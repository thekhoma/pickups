"""Delete a project or an asset (Run 6): nothing is erased. A project's folder and its shots, or a cast entry's or an image's
files, move into library/trash/<id>/ under their library paths (store._into_trash, the one way in), Restore puts them
back, and nothing else in the library is touched. Deleting My shots (or the only project) leaves an empty My shots."""

import json

import pytest

from pickups.app.store import Store

SHOT = {"name": "Harbour", "source": "source.mov", "width": 1920, "height": 1080, "fps": 24.0, "frames": 48,
        "duration_s": 2.0, "created_at": 0}


def make_shot(store: Store, shot_id: str, project_id: str, params: dict | None = None) -> None:
    d = store.shot_dir(shot_id)
    (d / "versions").mkdir(parents=True)
    (d / "source.mov").write_bytes(b"S" * 1000)
    (d / "thumb.jpg").write_bytes(b"J" * 10)
    (d / "meta.json").write_text(json.dumps({"id": shot_id, "project_id": project_id, **SHOT}))
    if params is not None:
        v = store.version_dir(shot_id, "v_1")
        v.mkdir(parents=True)
        (v / "master.mov").write_bytes(b"M" * 500)
        (v / "meta.json").write_text(json.dumps({"id": "v_1", "label": "Previz", "task": "previz", "created_at": 1, "params": params}))


def make_image(store: Store, project_id: str, image_id: str, prompt: str = "A red door") -> None:
    d = store.project_dir(project_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{image_id}.png").write_bytes(b"P" * 100)
    (d / f"{image_id}_thumb.jpg").write_bytes(b"T" * 10)
    f = store.project_dir(project_id) / "images.json"
    items = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps([*items, {"id": image_id, "file": f"{image_id}.png", "prompt": prompt, "created_at": len(items)}]))


def make_cast(store: Store, project_id: str, cast_id: str, name: str) -> None:
    d = store.project_dir(project_id) / "cast"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cast_id}.png").write_bytes(b"C" * 100)
    (d / f"{cast_id}_thumb.jpg").write_bytes(b"T" * 10)
    (d / f"{cast_id}_sheet_ab12.jpg").write_bytes(b"S" * 50)
    f = store.project_dir(project_id) / "cast.json"
    items = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps([*items, {"id": cast_id, "name": name, "kind": "person", "line": "", "file": f"{cast_id}.png",
                                      "created_at": len(items)}]))


def tree(path) -> dict:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: None)  # no ffmpeg thumbnails: the fixtures bring their own
    return Store(tmp_path / "library")


@pytest.fixture()
def client(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    monkeypatch.setattr(server, "ROOT", store.root)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(server, "Store", lambda: store)
    monkeypatch.setattr(server.JobManager, "list", lambda self: [])
    return TestClient(server.create_app("http://127.0.0.1:9")), server


@pytest.fixture()
def two(store):
    """Two projects with a shot each, and a cast entry and an image in the first."""
    a = store.create_project("Paris")["id"]
    b = store.create_project("Lisbon")["id"]
    make_shot(store, "shot_a1", a)
    make_shot(store, "shot_a2", a)
    make_shot(store, "shot_b1", b)
    make_cast(store, a, "cast_aa", "Maya")
    make_image(store, a, "img_aa")
    return a, b


# --- projects -------------------------------------------------------------------------------------------------------
def test_delete_a_project_moves_its_folder_and_shots_to_the_trash(client, store, two):
    c, _ = client
    a, b = two
    r = c.delete(f"/api/projects/{a}")
    assert r.status_code == 200, r.text
    entry, left = r.json()["entry"], r.json()["projects"]
    assert entry["kind"] == "project" and entry["name"] == "Paris" and entry["shots"] == 2 and entry["thumb_url"]
    assert a not in [p["id"] for p in left] and b in [p["id"] for p in left]
    # Everything is in library/trash/<id>/ under its library path: recoverable by hand.
    home = store.trash_root / entry["id"]
    assert (home / "trash.json").is_file()
    assert (home / "projects" / a / "cast" / "cast_aa.png").is_file() and (home / "projects" / a / "images" / "img_aa.png").is_file()
    assert (home / "shots" / "shot_a1" / "source.mov").is_file() and (home / "shots" / "shot_a2" / "meta.json").is_file()
    assert not store.project_exists(a) and not store.shot_dir("shot_a1").exists()
    assert [e["id"] for e in c.get("/api/trash").json()] == [entry["id"]]
    assert c.get(f"/api/projects/{a}").status_code == 404


def test_delete_a_project_leaves_the_others_alone(client, store, two):
    c, _ = client
    a, b = two
    before = tree(store.project_dir(b)), tree(store.shot_dir("shot_b1")), tree(store.project_dir("unsorted"))
    assert c.delete(f"/api/projects/{a}").status_code == 200
    assert (tree(store.project_dir(b)), tree(store.shot_dir("shot_b1")), tree(store.project_dir("unsorted"))) == before
    assert [s["id"] for s in c.get(f"/api/projects/{b}").json()["shots"]] == ["shot_b1"]


def test_delete_the_current_project_my_shots_comes_back_empty(client, store, two):
    """My shots is where the app starts: deleting it (with its shots) leaves a fresh empty My shots to work in."""
    c, _ = client
    make_shot(store, "shot_u1", "unsorted")
    r = c.delete("/api/projects/unsorted")
    assert r.status_code == 200, r.text
    mine = next(p for p in r.json()["projects"] if p["id"] == "unsorted")
    assert mine["name"] == "My shots" and mine["shots"] == 0
    assert (store.trash_root / r.json()["entry"]["id"] / "shots" / "shot_u1" / "source.mov").is_file()


def test_delete_the_only_project_leaves_an_empty_my_shots(client, store):
    c, _ = client
    make_shot(store, "shot_u1", "unsorted")
    make_image(store, "unsorted", "img_u1")
    assert [p["id"] for p in c.get("/api/projects").json()] == ["unsorted"]
    r = c.delete("/api/projects/unsorted")
    assert r.status_code == 200
    assert [(p["id"], p["shots"], p["images"]) for p in r.json()["projects"]] == [("unsorted", 0, 0)]
    # Restore puts the old My shots back into the new one: its shot and image return, nothing new is lost.
    make_image(store, "unsorted", "img_new")
    assert c.post(f"/api/trash/{r.json()['entry']['id']}/restore").status_code == 200
    detail = c.get("/api/projects/unsorted").json()
    assert [s["id"] for s in detail["shots"]] == ["shot_u1"]
    assert sorted(i["id"] for i in c.get("/api/projects/unsorted/images").json()) == ["img_new", "img_u1"]
    assert c.get("/api/trash").json() == []


def test_restore_a_project_puts_it_back_whole(client, store, two):
    c, _ = client
    a, _b = two
    files = lambda: ({k: v for k, v in tree(store.project_dir(a)).items() if k != "meta.json"}, tree(store.shot_dir("shot_a1")))  # noqa: E731
    before = files()
    entry = c.delete(f"/api/projects/{a}").json()["entry"]
    r = c.post(f"/api/trash/{entry['id']}/restore")
    assert r.status_code == 200, r.text
    assert files() == before  # meta.json too, but for its updated_at
    assert c.get(f"/api/projects/{a}").json()["project"]["name"] == "Paris"
    assert sorted(s["id"] for s in c.get(f"/api/projects/{a}").json()["shots"]) == ["shot_a1", "shot_a2"]


def test_delete_project_refuses_examples_unknown_and_busy(client, store, two, monkeypatch):
    c, server = client
    a, _b = two
    assert c.delete("/api/projects/examples").status_code == 403
    assert c.delete("/api/projects/prj_nope").status_code == 404
    monkeypatch.setattr(server.JobManager, "list", lambda self: [{"project_id": a, "shot_id": "shot_a1", "status": "running"}])
    assert c.delete(f"/api/projects/{a}").status_code == 409
    assert store.project_exists(a) and store.shot_dir("shot_a1").exists()


# --- assets ---------------------------------------------------------------------------------------------------------
def test_delete_an_unused_cast_entry(client, store, two):
    c, _ = client
    a, _b = two
    make_cast(store, a, "cast_bb", "Leo")
    assert {x["id"]: x["uses"] for x in c.get(f"/api/projects/{a}").json()["cast"]} == {"cast_aa": 0, "cast_bb": 0}
    keep = tree(store.project_dir(a) / "images")
    r = c.delete(f"/api/projects/{a}/cast/cast_aa")
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["kind"] == "cast" and entry["name"] == "Maya" and entry["thumb_url"] and "entry" not in entry
    home = store.trash_root / entry["id"] / "projects" / a / "cast"
    assert sorted(f.name for f in home.iterdir()) == ["cast_aa.png", "cast_aa_sheet_ab12.jpg", "cast_aa_thumb.jpg"]
    assert [x["id"] for x in c.get(f"/api/projects/{a}").json()["cast"]] == ["cast_bb"]
    assert (store.project_dir(a) / "cast" / "cast_bb.png").is_file() and tree(store.project_dir(a) / "images") == keep
    assert c.post(f"/api/trash/{entry['id']}/restore").status_code == 200
    assert sorted(x["id"] for x in c.get(f"/api/projects/{a}").json()["cast"]) == ["cast_aa", "cast_bb"]


def test_delete_an_image_a_shot_uses(client, store, two):
    """The shot keeps its take and settings; the gone reference is skipped, never an error."""
    c, _ = client
    a, _b = two
    make_image(store, a, "img_bb", "A blue car")
    refs = [{"kind": "image", "id": "img_aa"}, {"kind": "image", "id": "img_bb"}]
    make_shot(store, "shot_a3", a, params={"look_refs": refs})
    assert {i["id"]: i["uses"] for i in c.get(f"/api/projects/{a}/images").json()} == {"img_aa": 1, "img_bb": 1}
    r = c.delete(f"/api/projects/{a}/images/img_aa")
    assert r.status_code == 200, r.text
    assert (store.trash_root / r.json()["id"] / "projects" / a / "images" / "img_aa.png").is_file()
    assert [i["id"] for i in c.get(f"/api/projects/{a}/images").json()] == ["img_bb"]
    shot = c.get("/api/shots/shot_a3").json()
    assert shot["versions"][-1]["params"]["look_refs"] == refs
    assert store.live_refs(a, refs) == [store.project_dir(a) / "images" / "img_bb.png"]
    assert c.get(f"/api/projects/{a}/refs/image/img_aa").status_code == 404
    assert c.get(f"/api/projects/{a}").status_code == 200


def test_delete_asset_refuses_unknown_and_busy(client, store, two, monkeypatch):
    c, server = client
    a, b = two
    assert c.delete(f"/api/projects/{a}/cast/cast_nope").status_code == 404
    assert c.delete(f"/api/projects/{b}/images/img_aa").status_code == 404
    assert c.delete("/api/projects/prj_nope/images/img_aa").status_code == 404
    monkeypatch.setattr(server.JobManager, "list", lambda self: [{"project_id": a, "status": "queued", "params": {"cast_id": "cast_aa"}}])
    assert c.delete(f"/api/projects/{a}/cast/cast_aa").status_code == 409
    assert c.delete(f"/api/projects/{a}/images/img_aa").status_code == 200


def test_restore_an_asset_needs_its_project(client, store, two):
    c, _ = client
    a, _b = two
    image = c.delete(f"/api/projects/{a}/images/img_aa").json()
    c.delete(f"/api/projects/{a}")
    r = c.post(f"/api/trash/{image['id']}/restore")
    assert r.status_code == 404 and r.json()["detail"] == "Restore its project first."


def test_an_entry_from_before_run_6_still_lists_and_restores(store):
    """Older trash entries hold their one item at the top of the entry, not under its library path."""
    make_shot(store, "shot_old", "unsorted")
    entry = store.trash_root / "tr_0123456789"
    entry.mkdir(parents=True)
    store.shot_dir("shot_old").rename(entry / "shot_old")
    (entry / "trash.json").write_text(json.dumps({"id": "tr_0123456789", "kind": "shot", "shot_id": "shot_old", "take_id": None,
                                                  "path": "shots/shot_old", "name": "Harbour", "project_id": "unsorted",
                                                  "trashed_at": 1, "thumb": "thumb.jpg"}))
    [listed] = store.trash_list()
    assert listed["bytes"] >= 1000
    assert listed["thumb_url"].endswith("trash/tr_0123456789/shot_old/thumb.jpg")
    store.restore("tr_0123456789")
    assert (store.shot_dir("shot_old") / "source.mov").is_file()
