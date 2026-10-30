"""Asset lifecycle: a failed or canceled job leaves no orphan take, the startup sweep, Trash for takes and shots (rename in
and out, restore, empty with its path guard), the routes (409 while a job works on the shot, Examples blocked, a trashed
take gone from the shot), Storage, and a failed trim leaving no copy behind."""

import json
import os
import time
from types import SimpleNamespace

import pytest

from pickups.app.comfy import Canceled
from pickups.app.jobs import JobManager, Stage, TaskSpec
from pickups.app.store import Store

SHOT = {"name": "Harbour", "source": "source.mov", "width": 1920, "height": 1080, "fps": 24.0, "frames": 48,
        "duration_s": 2.0, "created_at": 0}


def make_shot(store: Store, shot_id: str, project_id: str = "unsorted") -> None:
    d = store.shot_dir(shot_id)
    (d / "versions").mkdir(parents=True)
    (d / "source.mov").write_bytes(b"S" * 1000)
    (d / "meta.json").write_text(json.dumps({"id": shot_id, "project_id": project_id, **SHOT}))


def make_take(store: Store, shot_id: str, vid: str, workflow: dict | None = None) -> None:
    d = store.version_dir(shot_id, vid)
    d.mkdir(parents=True)
    (d / "master.mov").write_bytes(b"M" * 5000)
    if workflow:
        (d / "workflows").mkdir()
        (d / "workflows" / "01_render.json").write_text(json.dumps(workflow))
    (d / "meta.json").write_text(json.dumps({"id": vid, "label": "Remove", "task": "remove", "created_at": time.time()}))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    monkeypatch.delenv("PICKUPS_KEEP_WORK", raising=False)
    s = Store(tmp_path / "library")
    make_shot(s, "shot_a")
    return s


def run(jm, shot_id, task):
    job = jm.submit(shot_id, task, {})
    for _ in range(300):
        if jm.jobs[job["id"]]["status"] in ("done", "failed", "canceled"):
            return jm.jobs[job["id"]]
        time.sleep(0.02)
    raise AssertionError(jm.jobs[job["id"]])


def manager(store, fn):
    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=lambda: None)
    jm = JobManager(store, conn, {"t": TaskSpec([Stage("Working", 1)], fn)})
    jm.broadcast = lambda kind, data: None
    return jm


# --- orphan cleanup -----------------------------------------------------------------------------------------------
def test_a_failed_or_canceled_job_leaves_no_take_folder(store):
    def broken(ctx):
        (ctx.vdir / "work").mkdir()
        (ctx.vdir / "work" / "f0001.png").write_bytes(b"x" * 100)
        raise RuntimeError("out of memory")

    def canceled(ctx):
        (ctx.vdir / "work").mkdir()
        raise Canceled()

    assert run(manager(store, broken), "shot_a", "t")["status"] == "failed"
    assert run(manager(store, canceled), "shot_a", "t")["status"] == "canceled"
    assert list((store.shot_dir("shot_a") / "versions").iterdir()) == []


def test_a_saved_take_survives_a_later_failure(store):
    """Data loss guard: once meta.json is written the folder is a take, and nothing in the fail path may touch it."""
    def saved_then_broken(ctx):
        (ctx.vdir / "master.mov").write_bytes(b"M")
        (ctx.vdir / "meta.json").write_text("{}")
        raise RuntimeError("after the save")

    assert run(manager(store, saved_then_broken), "shot_a", "t")["status"] == "failed"
    [kept] = list((store.shot_dir("shot_a") / "versions").iterdir())
    assert (kept / "master.mov").exists()


def test_keep_work_keeps_a_failed_take(store, monkeypatch):
    monkeypatch.setenv("PICKUPS_KEEP_WORK", "1")

    def broken(ctx):
        raise RuntimeError("x")

    run(manager(store, broken), "shot_a", "t")
    assert len(list((store.shot_dir("shot_a") / "versions").iterdir())) == 1


def test_the_startup_sweep_takes_only_old_orphans(store):
    versions = store.shot_dir("shot_a") / "versions"
    old, young = versions / "v_old", versions / "v_young"
    old.mkdir()
    young.mkdir()
    make_take(store, "shot_a", "v_saved")
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    os.utime(versions / "v_saved", (two_hours_ago, two_hours_ago))
    assert store.sweep_orphan_versions() == 1
    assert not old.exists() and young.exists() and (versions / "v_saved").exists()


def test_discard_never_leaves_the_library(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    assert store.discard_version_dir(outside) is False and outside.exists()
    link = store.shot_dir("shot_a") / "versions" / "v_link"
    link.symlink_to(outside, target_is_directory=True)
    assert store.discard_version_dir(link) is False and outside.exists()


# --- trash in the store -------------------------------------------------------------------------------------------
WORKFLOW = {"1": {"class_type": "LoadImage", "inputs": {"image": "shot_a_src.png"}},
            "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pickups/remove/frames"}},
            "3": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "pickups/remove/video"}}}


def test_trash_restore_and_empty_a_take(store):
    make_take(store, "shot_a", "v_1", WORKFLOW)
    entry = store.trash("take", "shot_a", "v_1")
    assert entry["kind"] == "take" and entry["path"] == "shots/shot_a/versions/v_1" and entry["bytes"] >= 5000
    assert entry["remote_files"] == 3
    assert [v["id"] for v in store.shot("shot_a")["versions"]] == ["orig"]
    trash_json = json.loads((store.trash_root / entry["id"] / "trash.json").read_text())
    assert trash_json["shot_id"] == "shot_a" and trash_json["take_id"] == "v_1"

    store.restore(entry["id"])
    assert [v["id"] for v in store.shot("shot_a")["versions"]] == ["orig", "v_1"] and store.trash_list() == []

    store.trash("take", "shot_a", "v_1")
    removed = store.empty_trash()
    assert removed == {"items": 1, "bytes": removed["bytes"], "remote_files": 3} and removed["bytes"] >= 5000
    assert store.trash_list() == [] and not store.version_dir("shot_a", "v_1").exists()
    assert (store.shot_dir("shot_a") / "source.mov").exists()  # the original is untouched


def test_trash_a_whole_shot_and_bring_it_back(store):
    make_take(store, "shot_a", "v_1")
    entry = store.trash("shot", "shot_a")
    assert [s["id"] for s in store.shots()] == [] and entry["bytes"] >= 6000
    store.restore(entry["id"])
    assert [s["id"] for s in store.shots()] == ["shot_a"]
    assert [v["id"] for v in store.shot("shot_a")["versions"]] == ["orig", "v_1"]


def test_what_cannot_go_to_the_trash(store):
    make_shot(store, "shot_ex", project_id="examples")
    with pytest.raises(PermissionError):
        store.trash("shot", "shot_ex")
    with pytest.raises(PermissionError):
        store.trash("take", "shot_a", "orig")
    with pytest.raises(FileNotFoundError):
        store.trash("take", "shot_a", "v_missing")
    with pytest.raises(FileNotFoundError):
        store.trash("take", "shot_a", "../../etc")


def test_restore_refuses_to_overwrite_and_needs_its_shot(store):
    make_take(store, "shot_a", "v_1")
    take = store.trash("take", "shot_a", "v_1")
    shot = store.trash("shot", "shot_a")
    with pytest.raises(FileNotFoundError):  # the take's shot is in the trash too
        store.restore(take["id"])
    store.restore(shot["id"])
    store.version_dir("shot_a", "v_1").mkdir()  # something new sits where the take was
    with pytest.raises(FileExistsError):
        store.restore(take["id"])
    with pytest.raises(FileNotFoundError):
        store.restore("../shots")


def test_empty_trash_never_follows_a_link_out_of_the_trash(store, tmp_path):
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "film.mov").write_bytes(b"keep")
    store.trash_root.mkdir(parents=True, exist_ok=True)
    (store.trash_root / "tr_evil").symlink_to(precious, target_is_directory=True)
    make_take(store, "shot_a", "v_1")
    entry = store.trash("take", "shot_a", "v_1")
    (store.trash_root / entry["id"] / entry["path"] / "escape").symlink_to(precious, target_is_directory=True)
    store.empty_trash()
    assert (precious / "film.mov").read_bytes() == b"keep"
    assert list(store.trash_root.iterdir()) == []


def test_storage_reports_the_library(store):
    info = store.storage()
    assert info["path"] == str(store.root) and info["bytes"] >= 1000


# --- routes -------------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    monkeypatch.setattr(server, "ROOT", store.root)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(server, "Store", lambda: store)
    return TestClient(server.create_app("http://127.0.0.1:9")), server


def test_routes_trash_restore_and_empty(client, store):
    c, _server = client
    make_take(store, "shot_a", "v_1")
    r = c.delete("/api/shots/shot_a/takes/v_1")
    assert r.status_code == 200, r.text
    assert [v["id"] for v in c.get("/api/shots/shot_a").json()["versions"]] == ["orig"]
    [entry] = c.get("/api/trash").json()
    assert entry["id"] == r.json()["id"]
    restored = c.post(f"/api/trash/{entry['id']}/restore").json()
    assert [v["id"] for v in restored["shot"]["versions"]] == ["orig", "v_1"]
    c.delete("/api/shots/shot_a/takes/v_1")
    gone = c.delete("/api/trash").json()
    assert gone["items"] == 1 and gone["bytes"] >= 5000
    assert c.get("/api/storage").json()["bytes"] >= 1000


def test_routes_refuse_while_a_job_works_on_the_shot_and_for_examples(client, store, monkeypatch):
    c, server = client
    make_take(store, "shot_a", "v_1")
    make_shot(store, "shot_ex", project_id="examples")
    monkeypatch.setattr(server.JobManager, "list", lambda self: [{"shot_id": "shot_a", "status": "running"}])
    assert c.delete("/api/shots/shot_a/takes/v_1").status_code == 409
    assert c.delete("/api/shots/shot_a").status_code == 409
    assert store.version_dir("shot_a", "v_1").exists()
    monkeypatch.setattr(server.JobManager, "list", lambda self: [{"shot_id": "shot_a", "status": "done"}])
    assert c.delete("/api/shots/shot_ex").status_code == 403
    assert c.delete("/api/shots/shot_a/takes/orig").status_code == 403
    assert c.delete("/api/shots/shot_a/takes/v_nope").status_code == 404
    assert c.delete("/api/shots/shot_a").status_code == 200


def test_a_failed_trim_leaves_no_copy_in_the_library(client, store, monkeypatch):
    import subprocess
    c, _server = client

    def ffmpeg_fails(cmd, **kw):
        out = cmd[-1]
        open(out, "wb").write(b"half a file")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", ffmpeg_fails)
    with pytest.raises(subprocess.CalledProcessError):
        c.post("/api/shots/shot_a/trim", json={"seconds": 1})
    assert not list(store.root.glob("trim_*"))
