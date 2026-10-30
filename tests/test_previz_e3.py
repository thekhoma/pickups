"""Fix E3: the hand-off from a previz to Previz to final (blank frames, the start-frame check, the look preview)."""

import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from pickups import framing
from pickups.app import previz_maker
from pickups.app.store import Store

SPEC = {
    "version": 1, "template": "giant_orbit", "duration_s": 97 / 24, "fps": 24, "size": [1920, 1088],
    "lens": {"focal_mm": 24, "film_gauge_mm": 36},
    "environment": {"ground": "plane", "sky": "gradient", "fog": [50, 2000], "props": []},
    "subjects": [{"id": "a", "kind": "humanoid", "height_m": 1.8, "pos": [0, 0, 0], "anim": "idle"}],
    "camera": {"interp": "catmull_rom", "ease": "in_out",
               "keys": [{"t": 0, "pos": [3, 1, 4], "look": [0, 1, 0]}, {"t": 1, "pos": [2, 1, 3], "look": [0, 1, 0]}]},
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
    monkeypatch.setattr(server.common, "previz_frames", lambda vram: 97)
    return TestClient(server.create_app("http://127.0.0.1:9")), store, server


def _scene(subject_x: float = 0.5, subject_h: float = 0.5, horizon: float = 0.45, size=(1920, 1088), photo=False) -> Image.Image:
    """A grey layout: sky, ground, a horizon line, a few set pieces and one standing figure."""
    w, h = size
    im = Image.new("RGB", size, (190, 190, 190))
    d = ImageDraw.Draw(im)
    hy = int(horizon * h)
    d.rectangle([0, hy, w, h], fill=(120, 118, 115) if photo else (110, 110, 110))
    d.line([0, hy, w, hy], fill=(60, 60, 60), width=4)
    for bx in (0.08, 0.3, 0.72, 0.9):
        d.rectangle([int(bx * w), hy - int(0.18 * h), int(bx * w) + int(0.06 * w), hy], fill=(150, 150, 150), outline=(70, 70, 70), width=3)
    fh = int(subject_h * h)
    cx = int(subject_x * w)
    top = int(0.85 * h) - fh
    d.ellipse([cx - fh // 12, top, cx + fh // 12, top + fh // 6], fill=(40, 40, 40))
    d.rectangle([cx - fh // 8, top + fh // 6, cx + fh // 8, top + fh * 6 // 10], fill=(220, 200, 40) if not photo else (90, 60, 50))
    d.rectangle([cx - fh // 9, top + fh * 6 // 10, cx - fh // 40, top + fh], fill=(30, 30, 30))
    d.rectangle([cx + fh // 40, top + fh * 6 // 10, cx + fh // 9, top + fh], fill=(30, 30, 30))
    return im


def _box(subject_x=0.5, subject_h=0.5):
    top = 0.85 - subject_h
    return [subject_x - subject_h / 8, top, subject_x + subject_h / 8, 0.85]


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def test_blank_checks_sample_every_8th_frame_and_the_last(tmp_path):
    good = _jpeg(_scene())
    black = _jpeg(Image.new("RGB", (1920, 1088), (0, 0, 0)))
    paths = []
    for i in range(97):
        p = tmp_path / f"{i:04d}.jpg"
        p.write_bytes(black if i == 40 else good)
        paths.append(p)
    with pytest.raises(previz_maker.PrevizError) as e:
        previz_maker.check_frames(paths)
    assert str(e.value) == "Frame 41 came out blank. Check the preview, then try Make shot again."
    paths[40].write_bytes(good)
    previz_maker.check_frames(paths)


def test_a_black_frame_at_40_is_refused_by_the_server(client):
    c, _, _ = client
    good = _jpeg(_scene())
    black = _jpeg(Image.new("RGB", (1920, 1088), (0, 0, 0)))
    sid = "pv_0123456789abcdef"
    for start in range(0, 97, 16):
        files = [("files", (f"{i:04d}.jpg", black if i == 40 else good, "image/jpeg")) for i in range(start, min(97, start + 16))]
        assert c.post(f"/api/previz/{sid}/frames", data={"start": str(start)}, files=files).status_code == 200
    r = c.post(f"/api/previz/{sid}/finish", json={"spec": SPEC, "name": "Blank at 40"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Frame 41 came out blank. Check the preview, then try Make shot again."


def test_framing_matches_itself_and_warns_when_the_subject_moves(tmp_path):
    f0 = tmp_path / "f0.jpg"
    f0.write_bytes(_jpeg(_scene()))
    fr = {"subject": _box(), "horizon": 0.45}
    same = framing.compare(f0, fr, f0)
    assert same["warning"] is None, same
    look = tmp_path / "look.jpg"
    look.write_bytes(_jpeg(_scene(photo=True)))
    assert framing.compare(f0, fr, look)["warning"] is None
    moved = tmp_path / "moved.jpg"
    moved.write_bytes(_jpeg(_scene(subject_x=0.7, photo=True)))
    out = framing.compare(f0, fr, moved)
    assert out["warning"] and "further right" in out["warning"], out
    assert abs(out["subject_x"][1] - 0.7) < 0.03
    small = tmp_path / "small.jpg"
    small.write_bytes(_jpeg(_scene(subject_x=0.3, subject_h=0.3)))
    out = framing.compare(f0, fr, small)
    assert "further left" in out["warning"] and "smaller" in out["warning"], out


def test_finish_keeps_frame_0_and_the_framing_endpoint_reads_it(client):
    c, store, _ = client
    jpeg0 = _jpeg(_scene())
    sid = "pv_fedcba9876543210"
    for start in range(0, 97, 16):
        files = [("files", (f"{i:04d}.jpg", jpeg0, "image/jpeg")) for i in range(start, min(97, start + 16))]
        assert c.post(f"/api/previz/{sid}/frames", data={"start": str(start)}, files=files).status_code == 200
    r = c.post(f"/api/previz/{sid}/finish", json={"spec": SPEC, "name": "Framing", "framing": {"subject": _box(), "horizon": 0.45}})
    assert r.status_code == 200, r.text
    shot = r.json()
    kept = store.shot_dir(shot["id"]) / previz_maker.FRAME0
    assert hashlib.sha256(kept.read_bytes()).hexdigest() == shot["previz_frame0"]
    assert store.shot_meta(shot["id"])["previz_framing"]["horizon"] == 0.45
    ok = store.add_upload(_jpeg(_scene(photo=True)), "look.jpg")
    off = store.add_upload(_jpeg(_scene(subject_x=0.7, photo=True)), "off.jpg")
    a = c.post("/api/previz/framing", json={"shot_id": shot["id"], "look_frame_upload_id": ok["id"]}).json()
    b = c.post("/api/previz/framing", json={"shot_id": shot["id"], "look_frame_upload_id": off["id"]}).json()
    assert a["checked"] and a["warning"] is None, a
    assert b["checked"] and "further right" in b["warning"], b
    # A frame 0 that isn't the exported one (hash mismatch) is never compared.
    kept.write_bytes(_jpeg(_scene(subject_x=0.2)))
    assert c.post("/api/previz/framing", json={"shot_id": shot["id"], "look_frame_upload_id": off["id"]}).json()["checked"] is False


def test_look_preview_reads_every_reference_and_the_seed(client, monkeypatch, capsys):
    c, store, server = client
    prj = store.create_project("Diner")
    src = Path(__file__).parent / "_tmp_unused.mov"
    seen = {}

    monkeypatch.setattr(store, "version_master", lambda shot_id, version_id=None: src)
    monkeypatch.setattr(store, "shot_meta", lambda shot_id: {"project_id": prj["id"]})
    monkeypatch.setattr(store, "shot_dir", lambda shot_id: store.root / "shots_fake" / shot_id)
    monkeypatch.setattr(store, "shot_cast", lambda shot_id: ([], lambda c: None))
    monkeypatch.setattr(store, "ref_path", lambda pid, r: Path(f"/refs/{pid}/{r['kind']}_{r['id']}.png"))
    (store.root / "shots_fake" / "sh1").mkdir(parents=True)

    def fake_look(runner, st, playblast, refs, look, out, seed=42, at=0, commercial_safe=False):
        seen.setdefault("calls", []).append({"refs": [str(r) for r in refs], "look": look, "seed": seed})
        Image.new("RGB", (8, 8), (seed % 255, 0, 0)).save(out)
        return out

    monkeypatch.setattr(server.previz, "look_frame", fake_look)
    u1 = store.add_upload(b"\x89PNG one", "a.png")
    u2 = store.add_upload(b"\x89PNG two", "b.png")
    seeds = []
    for seed in (11, 22, 33):
        r = c.post("/api/previz/look", json={"shot_id": "sh1", "look_upload_ids": [u1["id"], u2["id"]],
                                            "look_refs": [{"kind": "cast", "id": "maya"}, {"kind": "image", "id": "im1"}],
                                            "look_prompt": "The grey figure is a man in a diner.", "seed": seed})
        assert r.status_code == 200, r.text
        assert r.json()["seed"] == seed and r.json()["references"] == 4
        seeds.append(r.json()["upload_id"])
    assert len(set(seeds)) == 3
    assert [x["seed"] for x in seen["calls"]] == [11, 22, 33]
    refs = seen["calls"][0]["refs"]
    assert len(refs) == 4 and refs[2].endswith("cast_maya.png") and refs[3].endswith("image_im1.png")
    err = capsys.readouterr().err
    assert "seed 11, 4 reference(s)" in err and "cast_maya.png" in err
