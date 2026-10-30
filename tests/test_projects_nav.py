import subprocess

import pytest

from pickups.app.store import EXAMPLES, UNSORTED, Store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    return Store(tmp_path / "library")


@pytest.fixture()
def clip(tmp_path):
    f = tmp_path / "clip.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "1", str(f)], check=True)
    return f


def add_take(store, shot_id, task="remove", **meta):
    vid, _ = store.new_version_dir(shot_id)
    store.save_version(shot_id, vid, {"label": task.title(), "task": task} | meta)
    return vid


def test_examples_list_only_shots_with_takes(store, clip):
    queued = store.add_shot(clip, "Warehouse street", project_id=EXAMPLES)
    pier = store.add_shot(clip, "Pier", project_id=EXAMPLES)
    add_take(store, pier["id"])
    failed = store.add_shot(clip, "Failed", project_id=EXAMPLES)
    store.update_take(failed["id"], add_take(store, failed["id"]), hidden=True)
    store.add_shot(clip, "Mine", project_id=UNSORTED)  # elsewhere a shot without takes is listed as usual

    assert [s["name"] for s in store.shots(EXAMPLES)] == ["Pier"]
    p = store.project(EXAMPLES)
    assert (p["shots"], p["takes"]) == (1, 1)
    listed = {x["id"]: x for x in store.projects()}
    assert (listed[EXAMPLES]["shots"], listed[EXAMPLES]["takes"]) == (1, 1)
    assert listed[UNSORTED]["shots"] == 1
    assert {s["name"] for s in store.shots()} == {"Pier", "Mine"}
    assert store.shot(queued["id"])["name"] == "Warehouse street"  # still opens by id



def test_empty_examples_project_is_not_listed(store, clip):
    assert EXAMPLES not in {p["id"] for p in store.projects()}  # a fresh library: nothing to show there
    assert store.project(EXAMPLES)["shots"] == 0  # still opens by link
    store.add_shot(clip, "Queued", project_id=EXAMPLES)
    assert EXAMPLES not in {p["id"] for p in store.projects()}  # a shot without a take doesn't count
    pier = store.add_shot(clip, "Pier", project_id=EXAMPLES)
    add_take(store, pier["id"])
    assert store.projects()[-1]["id"] == EXAMPLES  # an old library with example takes keeps it, last

def test_finish_candidates(store, clip):
    prj = store.create_project("P")["id"]
    shots = {n: store.add_shot(clip, n, project_id=prj)["id"] for n in "ABCDEF"}
    t = {n: add_take(store, shots[n]) for n in "ABCF"}
    for n in "ABCF":
        store.update_take(shots[n], t[n], status="approved")
    add_take(store, shots["B"], "finish", input_version=t["B"])  # already finished
    hidden = add_take(store, shots["F"], "finish", input_version=t["F"])
    store.update_take(shots["F"], hidden, hidden=True)  # a hidden (failed) finish doesn't count
    store.update_take(shots["D"], add_take(store, shots["D"], "finish", input_version="orig"), status="approved")  # a Finish take
    store.update_take(shots["E"], "orig", status="approved")  # an approved original is finished too
    jobs = [{"task": "finish", "status": "queued", "shot_id": shots["C"], "version_id": t["C"]},
            {"task": "finish", "status": "done", "shot_id": shots["A"], "version_id": t["A"]},
            {"task": "remove", "status": "running", "shot_id": shots["E"], "version_id": None}]

    got = {(d["shot_id"], d["take"]["id"]) for d in store.finish_candidates(prj, jobs)}
    assert got == {(shots["A"], t["A"]), (shots["E"], "orig"), (shots["F"], t["F"])}
    assert len(store.finish_candidates(prj)) == 4  # without jobs, C is queued too
