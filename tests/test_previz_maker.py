"""Previz maker: the spec, and JPEG frames assembled into a shot."""

import hashlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from pickups.app.previz_maker import frame_count, parse_spec
from pickups.app.store import Store


EXAMPLE = {
    "version": 1, "template": "giant_orbit", "duration_s": 6, "fps": 24, "size": [1920, 1088],
    "lens": {"focal_mm": 24, "film_gauge_mm": 36},
    "environment": {"ground": "plane", "sky": "gradient", "fog": [50, 2000], "props": ["houses_small", "trees", "clouds"]},
    "subjects": [{"id": "giant", "kind": "humanoid", "height_m": 300, "pos": [0, 0, 0], "anim": "walk_slow"}],
    "camera": {"interp": "catmull_rom", "ease": "in_out",
               "keys": [{"t": 0, "pos": [3, 1, 4], "look": [0, 2, 0]},
                        {"t": 1, "pos": [350, 200, 350], "look": [0, 150, 0], "focal_mm": 35}]},
    "seed": 1,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    app = server.create_app("http://127.0.0.1:9")
    return TestClient(app), store


def _jpeg(mark: int) -> bytes:
    im = Image.new("RGB", (1920, 1088), (150, 150, 150))
    ImageDraw.Draw(im).rectangle([80, 80, 400 + mark, 900], fill=(30, 30, 30))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_frame0_digest_is_deterministic(web_node):
    import json
    import subprocess
    def once() -> str:
        r = subprocess.run(["node", "--experimental-strip-types", "src/previz/frame0-cli.ts"], input=json.dumps(EXAMPLE),
                           capture_output=True, text=True, cwd=web_node, check=True)
        return r.stdout.strip()
    a, b = once(), once()
    assert a == b and len(a) == 8 and int(a, 16) >= 0


def test_spec_schema_accepts_the_example_and_rejects_a_bad_template():
    spec = parse_spec(EXAMPLE)
    assert spec.template == "giant_orbit" and spec.speed == 1 and spec.notes == ""
    assert spec.camera.free is False and spec.camera.frame is None
    assert frame_count(spec.duration_s, spec.fps) == 144
    with pytest.raises(Exception):
        parse_spec({**EXAMPLE, "template": "freeform"})


def test_a_recompose_and_a_free_camera_parse():
    camera = {**EXAMPLE["camera"], "frame": [0.2, -0.4], "free": True}
    spec = parse_spec({**EXAMPLE, "camera": camera})
    assert spec.camera.frame == (0.2, -0.4) and spec.camera.free is True
    with pytest.raises(Exception):
        parse_spec({**EXAMPLE, "camera": {**EXAMPLE["camera"], "frame": [3, 0]}})


BENT = [{"t": round(i / 5, 3), "pos": [3 + i, 1.6, 4 - i * 0.5], "look": [0, 1.2, 0]} for i in range(6)]


def test_a_bent_following_path_parses_and_old_specs_still_do():
    spec = parse_spec({**EXAMPLE, "camera": {**EXAMPLE["camera"], "keys": BENT, "free": True, "follow": True}})
    assert len(spec.camera.keys) == 6 and spec.camera.follow is True
    # Old specs never had follow: they parse as before, and an unknown camera field is still refused.
    assert parse_spec(EXAMPLE).camera.follow is None
    with pytest.raises(Exception):
        parse_spec({**EXAMPLE, "camera": {**EXAMPLE["camera"], "follows": True}})


def test_a_bent_path_is_stored_unchanged(client):
    c, store = client
    sid = "pv_bbbbbbbbbbbbbbbb"
    files = [("files", (f"{i:04d}.jpg", _jpeg(i * 30), "image/jpeg")) for i in range(2)]
    assert c.post(f"/api/previz/{sid}/frames", data={"start": "0"}, files=files).status_code == 200
    spec = {**EXAMPLE, "duration_s": 2 / 24, "camera": {**EXAMPLE["camera"], "keys": BENT, "free": True, "follow": True}}
    r = c.post(f"/api/previz/{sid}/finish", json={"spec": spec, "name": "Bent"})
    assert r.status_code == 200, r.text
    saved = store.shot(r.json()["id"])["previz_spec"]["camera"]
    assert saved["follow"] is True and saved["free"] is True
    assert [k["pos"] for k in saved["keys"]] == [k["pos"] for k in BENT]


def test_over_the_gpu_cap_is_refused(client):
    c, _ = client
    r = c.post("/api/previz/pv_0123456789abcdef/finish", json={"spec": EXAMPLE, "name": "Giant orbit"})
    assert r.status_code == 400
    assert "97" in r.json()["detail"]


def test_frames_become_a_shot_and_frame_0_is_stable(client):
    c, store = client
    jpeg0, jpeg1 = _jpeg(0), _jpeg(40)
    spec = {**EXAMPLE, "duration_s": 2 / 24}
    assert frame_count(spec["duration_s"], 24) == 2

    def finish(session: str) -> dict:
        up = c.post(f"/api/previz/{session}/frames", data={"start": "0"}, files=[
            ("files", ("0000.jpg", jpeg0, "image/jpeg")),
            ("files", ("0001.jpg", jpeg1, "image/jpeg")),
        ])
        assert up.status_code == 200 and up.json()["received"] == 2
        prj = store.create_project("Layouts")
        r = c.post(f"/api/previz/{session}/finish", json={"spec": spec, "project_id": prj["id"], "name": "Street run"})
        assert r.status_code == 200, r.text
        return r.json()

    a = finish("pv_0123456789abcdef")
    b = finish("pv_fedcba9876543210")
    expect = hashlib.sha256(jpeg0).hexdigest()
    for shot in (a, b):
        assert shot["width"] == 1920 and shot["height"] == 1088
        assert shot["frames"] == 2
        assert abs(shot["fps"] - 24) < 0.02
        assert shot["previz_frame0"] == expect
        assert shot["previz_spec"]["template"] == "giant_orbit"
        assert shot["name"] == "Street run"
        assert (store.shot_dir(shot["id"]) / "meta.json").exists()
    assert a["previz_frame0"] == b["previz_frame0"]


def test_a_blank_frame_is_refused(client):
    c, _ = client
    flat = io.BytesIO()
    Image.new("RGB", (1920, 1088), (120, 120, 120)).save(flat, "JPEG", quality=95)
    raw = flat.getvalue()
    spec = {**EXAMPLE, "duration_s": 1 / 24}
    sid = "pv_aaaaaaaaaaaaaaaa"
    assert c.post(f"/api/previz/{sid}/frames", data={"start": "0"}, files=[("files", ("0000.jpg", raw, "image/jpeg"))]).status_code == 200
    r = c.post(f"/api/previz/{sid}/finish", json={"spec": spec, "name": "Blank"})
    assert r.status_code == 400 and "blank" in r.json()["detail"].lower()


def test_bad_session_and_batch(client):
    c, _ = client
    assert c.post("/api/previz/not-a-session/frames", data={"start": "0"}, files=[("files", ("a.jpg", b"\xff\xd8xx", "image/jpeg"))]).status_code == 400
    assert c.delete("/api/previz/nope").status_code == 400


def _compile(line: str, camera: str | None = None) -> dict:
    import json
    import subprocess
    from pathlib import Path
    web = Path(__file__).resolve().parents[1] / "web"
    env = dict(**__import__("os").environ)
    if camera:
        env["PREVIZ_CAMERA"] = camera
    r = subprocess.run(["node", "--experimental-strip-types", "src/previz/compile-cli.ts"], input=line,
                       capture_output=True, text=True, cwd=web, check=True, env=env)
    return json.loads(r.stdout)


@pytest.mark.usefixtures("web_modules")
def test_a_spoken_line_builds_a_world_and_a_camera_move():
    park = _compile(
        "A park with a bench and a person sitting at the bench. The camera is pushing in from behind "
        "and rotates around until we see them in front as they take a sip of coffee from their coffee cup."
    )
    spec = parse_spec(park["spec"])
    assert spec.template == "described"
    assert spec.environment.props[0] == "park"
    assert "bench" in spec.environment.props and "cup" in spec.environment.props
    assert spec.subjects[0].kind == "humanoid" and spec.subjects[0].anim == "sit_sip"
    assert spec.camera.keys[0].pos[2] < 0
    assert spec.camera.keys[-1].pos[2] > 0
    assert park["camera"] == "From behind, around to the front"
    assert park["missed"] == []
    again = _compile(park["spec"]["notes"])
    assert again["spec"]["seed"] == park["spec"]["seed"]
    assert again["spec"]["camera"] == park["spec"]["camera"]

    aerial = _compile(
        "a man walking a dog at a park, camera starts Medium on the man and slowly pulls out into a wide aerial, the camera follows the man"
    )
    aerial_spec = parse_spec(aerial["spec"])
    assert aerial["camera"] == "Medium to aerial"
    assert aerial["missed"] == []
    assert aerial_spec.subjects[0].kind == "humanoid" and aerial_spec.subjects[0].anim == "walk_follow"
    assert aerial_spec.subjects[1].kind == "dog"
    a0, a1 = aerial_spec.camera.keys[0], aerial_spec.camera.keys[-1]
    assert a0.pos[1] < 3 and a1.pos[1] > 8
    assert a0.focal_mm is not None and a1.focal_mm is not None and a0.focal_mm > a1.focal_mm
    assert (a0.pos[0] ** 2 + a0.pos[2] ** 2) < (a1.pos[0] ** 2 + a1.pos[2] ** 2)

    forest = _compile("A dog in the forest. The camera pushes in.")
    assert forest["spec"]["environment"]["props"][0] == "forest"
    assert forest["spec"]["subjects"][0]["kind"] == "dog"
    assert forest["spec"]["camera"]["keys"][0]["pos"] != forest["spec"]["camera"]["keys"][-1]["pos"]
    parse_spec(forest["spec"])

    suburbs = _compile("A car parked in the suburbs in front of a house.")
    assert suburbs["spec"]["environment"]["props"][0] == "suburbs"
    assert "house" in suburbs["spec"]["environment"]["props"]
    assert suburbs["spec"]["subjects"][0]["kind"] == "car"

    lake = _compile("A cat by the lake.")
    assert lake["spec"]["environment"]["props"][0] == "lake"
    assert lake["spec"]["subjects"][0]["kind"] == "cat"

    indoor = _compile("Inside a house, a person sitting.")
    assert indoor["spec"]["environment"]["props"][0] == "interior"
    assert indoor["spec"]["subjects"][0]["anim"] == "sit"
    assert indoor["spec"]["environment"]["ground"] == "none"

    desert = _compile("A person standing in the desert next to a fountain.")
    assert desert["spec"]["environment"]["props"][0] == "desert"
    assert "fountain" in desert["spec"]["environment"]["props"]
    assert "fountain" not in desert["missed"]

    skate = _compile("A kid riding a skateboard down the street.")
    assert skate["spec"]["environment"]["props"][0] == "city"
    assert skate["spec"]["subjects"][0]["kind"] == "skate"
    assert skate["spec"]["subjects"][0]["anim"] == "track"
    assert skate["spec"]["subjects"][0]["height_m"] == 1.2
    assert skate["missed"] == []

    side = _compile("A person walking on the sidewalk in a city.")
    assert side["spec"]["environment"]["props"][0] == "city"
    assert "sidewalk" in side["spec"]["environment"]["props"]
    assert side["spec"]["subjects"][0]["pos"][0] < -3
    assert side["spec"]["subjects"][0]["anim"] == "walk"
    assert side["missed"] == []

    parallel = _compile("A person walking on the sidewalk in a city as the camera tracks parallel along with it.")
    assert parallel["camera"] == "Parallel"
    assert parallel["spec"]["subjects"][0]["anim"] == "walk_follow"
    assert parallel["spec"]["subjects"][0]["pos"][0] < -3
    assert parallel["spec"]["camera"]["keys"][0]["pos"] == parallel["spec"]["camera"]["keys"][-1]["pos"]
    assert parallel["spec"]["camera"]["ease"] == "linear"
    assert parallel["missed"] == []

    jumping = _compile("A person jumping in the park.")
    assert jumping["spec"]["subjects"][0]["anim"] == "jump"
    laying = _compile("A person laying in the park.")
    assert laying["spec"]["subjects"][0]["anim"] == "lay"
    assert laying["missed"] == []

    running = _compile("A person running through the city.")
    assert running["spec"]["subjects"][0]["anim"] == "run"
    jogging = _compile("A person jogging in the park.")
    assert jogging["spec"]["environment"]["props"][0] == "park"
    assert jogging["spec"]["subjects"][0]["anim"] == "jog"

    kitchen = _compile("A chef in the kitchen.")
    assert kitchen["spec"]["environment"]["props"][0] == "interior"
    assert kitchen["spec"]["subjects"][0]["kind"] == "humanoid"
    assert kitchen["missed"] == []

    fpv = _compile("A kid riding a skateboard down the street.", "fpv")
    # FPV follow: a chase that stays behind the rider. Dodge and wrap is a different shot.
    assert fpv["camera"] == "Chase from behind"
    fpv_keys = fpv["spec"]["camera"]["keys"]
    assert len(fpv_keys) >= 12
    assert fpv["spec"]["camera"]["ease"] == "linear"
    assert fpv["spec"]["camera"]["speed"] == 1
    assert fpv["spec"]["camera"]["taper"] == 0.55
    # Behind the rider the whole way: about 3 m at the start, about 1.5 m at the end, never in front.
    behind = [k["look"][2] - k["pos"][2] for k in fpv_keys]
    assert all(b > 1.2 for b in behind)
    assert 2.6 < behind[0] < 3.4
    assert 1.2 < behind[-1] < 1.9
    assert behind[0] > behind[-1] + 1.0
    assert all(k["pos"][1] > k["look"][1] + 0.05 for k in fpv_keys)
    assert all(abs(k["look"][0]) < 0.4 and abs(k["look"][2]) < 0.4 for k in fpv_keys)
    rolls = [k["roll"] for k in fpv_keys]
    assert max(abs(b - a) for a, b in zip(rolls, rolls[1:])) < 12
    steps = [
        sum((b - a) ** 2 for a, b in zip(p["pos"], q["pos"])) ** 0.5
        for p, q in zip(fpv_keys, fpv_keys[1:])
    ]
    assert max(steps) < min(steps) * 4
    parse_spec(fpv["spec"])

    drone = _compile("A person standing in a park.", "drone")
    assert drone["camera"] == "Drone"
    keys = drone["spec"]["camera"]["keys"]
    assert keys[0]["pos"][1] > keys[-1]["pos"][1] + 3
    parse_spec(drone["spec"])

    web = __import__("pathlib").Path(__file__).resolve().parents[1] / "web"
    count = __import__("subprocess").run(["node", "--experimental-strip-types", "src/previz/count-cli.ts"],
                                         capture_output=True, text=True, cwd=web, check=True)
    assert int(count.stdout) >= 300


def test_gzip_library_files_only():
    """GLB, JSON and JS responses are gzipped. A png of the same size is not."""
    import asyncio

    from pickups.app.server import AssetGzip

    payload = b"x" * 400

    def inner(kind):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", kind)]})
            await send({"type": "http.response.body", "body": payload})
        return app

    async def once(path):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "method": "GET", "path": path, "headers": [(b"accept-encoding", b"gzip")], "query_string": b""}
        await AssetGzip(inner(b"application/octet-stream"))(scope, receive, send)
        return messages

    def run(path):
        return asyncio.run(once(path))

    packed = run("/previz/person.glb")
    assert packed[0]["headers"] and b"gzip" in dict(packed[0]["headers"]).get(b"content-encoding", b"")
    assert __import__("gzip").decompress(packed[1]["body"]) == payload
    plain = run("/preview.png")
    assert b"content-encoding" not in dict(plain[0]["headers"])
    assert plain[1]["body"] == payload

    library = run("/previz/assets/models/person__lite.glb")
    headers = dict(library[0]["headers"])
    assert headers.get(b"content-encoding") == b"gzip"
    assert headers.get(b"cache-control") == b"public, max-age=3600"
    data = run("/previz/data/person_poses.json")
    assert dict(data[0]["headers"]).get(b"cache-control") == b"public, max-age=3600"

    async def raw(path):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "method": "GET", "path": path, "headers": [], "query_string": b""}
        await AssetGzip(inner(b"application/octet-stream"))(scope, receive, send)
        return messages

    cached = asyncio.run(raw("/previz/data/animal_poses.json"))
    assert dict(cached[0]["headers"]).get(b"cache-control") == b"public, max-age=3600"
    assert b"content-encoding" not in dict(cached[0]["headers"])


def test_index_html_is_not_cached(client):
    c, _store = client
    page = c.get("/")
    if page.status_code == 200 and "html" in page.headers.get("content-type", ""):
        assert page.headers.get("cache-control") == "no-cache"


def test_a_qc_run_saves_to_the_archived_test_runs_project_and_old_test_shots_archive(client):
    import importlib.util
    from pathlib import Path

    c, store = client
    spec = {**EXAMPLE, "duration_s": 2 / 24}

    def make(session: str, headers: dict | None = None, project_id: str | None = None) -> dict:
        c.post(f"/api/previz/{session}/frames", data={"start": "0"}, files=[
            ("files", ("0000.jpg", _jpeg(0), "image/jpeg")), ("files", ("0001.jpg", _jpeg(40), "image/jpeg"))])
        r = c.post(f"/api/previz/{session}/finish", json={"spec": spec, "name": "Restaurant, person sitting", "project_id": project_id},
                   headers=headers or {})
        assert r.status_code == 200, r.text
        return r.json()

    assert not store.project_exists("qc")
    qc = make("pv_00000000000000aa", {"X-Pickups-QC": "1"}, project_id="unsorted")
    assert qc["project_id"] == "qc" and store.project("qc")["archived"] is True
    assert [p["id"] for p in store.projects() if not p["archived"]] == ["unsorted"]

    # Three identical leftovers in My shots: the newest stays, the others move (nothing deleted).
    left = [make(f"pv_00000000000000b{i}") for i in range(3)]
    spec_path = Path(__file__).resolve().parents[1] / "scripts" / "archive_test_takes.py"
    mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("archive_test_takes", spec_path))
    mod.__loader__.exec_module(mod)
    found = mod.test_shots(store, set())
    assert len(found) == 2 and left[-1]["id"] not in {s["id"] for s in found}
    for shot in found:
        store.move_shot(shot["id"], store.qc_project())
    assert all(store.shot_dir(s["id"]).exists() for s in left)
    assert [s["id"] for s in store.shots("unsorted")] == [left[-1]["id"]]
    assert len(mod.test_shots(store, {"Restaurant, person sitting"})) == 1
