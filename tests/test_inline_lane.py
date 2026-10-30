"""Step 21 D (SPEC section 3): the start frame and the look frame show in the queue as inline rows, without queuing."""

from pathlib import Path

import pytest
from PIL import Image

from pickups import start_frame as sf
from test_previz_e3 import SPEC, _jpeg, _scene, client  # noqa: F401 - the fixture
from test_start_frame import S04_FRAMING, _matte_url


@pytest.fixture()
def lane(client, monkeypatch):  # noqa: F811 - the fixture
    """The client, plus the app's JobManager and a count of free_memory calls."""
    c, store, server = client
    import pickups.app.comfy as comfy

    freed = []
    monkeypatch.setattr(comfy.Connection, "free_memory", lambda self: freed.append(1))
    made = []

    class Spy(server.JobManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(server, "JobManager", Spy)
    from fastapi.testclient import TestClient

    c = TestClient(server.create_app("http://127.0.0.1:9"))
    return c, store, server, made[-1], freed


def _previz_shot(c) -> str:
    sid = "pv_00aa11bb22cc33dd"
    jpeg0 = _jpeg(_scene())
    for start in range(0, 97, 16):
        files = [("files", (f"{i:04d}.jpg", jpeg0, "image/jpeg")) for i in range(start, min(97, start + 16))]
        assert c.post(f"/api/previz/{sid}/frames", data={"start": str(start)}, files=files).status_code == 200
    r = c.post(f"/api/previz/{sid}/finish", json={"spec": SPEC | {"notes": "A woman waves."}, "name": "Inline",
                                                   "framing": S04_FRAMING, "matte": _matte_url()})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_a_start_frame_is_a_running_inline_row_then_done_with_its_result(lane, monkeypatch):
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    seen = {}

    def fake_make(runner, plan, frame0, out, seed=42):  # the fake ComfyUI: a picture, and a look at the queue meanwhile
        rows = [j for j in jm.list() if j.get("lane") == "inline"]
        seen["running"] = [dict(j) for j in rows]
        Image.new("RGB", (16, 9), (seed % 255, 10, 10)).save(out)
        return out

    monkeypatch.setattr(sf, "make", fake_make)
    before = list(jm.order)
    r = c.post("/api/previz/start_frame", json={"shot_id": shot_id, "seed": 7, "draft_key": f"{shot_id}:previz"})
    assert r.status_code == 200, r.text
    res = r.json()
    # The response is today's: the picture, the seed and the plan's words.
    assert {"url", "upload_id", "seed", "positive", "negative", "method"} <= set(res) and res["seed"] == 7
    (row,) = seen["running"]
    assert row["status"] == "running" and row["lane"] == "inline" and row["stage"] == "Start frame" and row["stages"] == ["Start frame"]
    assert row["draft_key"] == f"{shot_id}:previz" and row["step"] == "start_frame" and row["params"] == {} and row["eta_s"] is None
    listed = [j for j in c.get("/api/jobs").json() if j.get("lane") == "inline"]
    assert len(listed) == 1 and listed[0]["status"] == "done"
    took = listed[0]["result"].pop("took_s")
    assert listed[0]["result"] == {"upload_id": res["upload_id"], "url": res["url"], "seed": 7}
    assert isinstance(took, float) and took >= 0 and listed[0]["took_s"] == took and listed[0]["kind"].startswith("start_frame|")
    # Never on the GPU queue: nothing queued, no free_memory, no take; only the one row added to the list.
    assert jm._q.qsize() == 0 and freed == [] and jm.order[: len(before)] == before and len(jm.order) == len(before) + 1
    assert [v for v in c.get(f"/api/shots/{shot_id}").json()["versions"] if v["id"] != "orig"] == []
    # Cancel on a finished row changes nothing (Run 8: a running one can be stopped).
    r = c.post(f"/api/jobs/{listed[0]['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "done"


def test_plan_only_and_old_clients_are_unchanged(lane):
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    p = c.post("/api/previz/start_frame", json={"shot_id": shot_id, "plan_only": True})
    assert p.status_code == 200 and "url" not in p.json()
    assert not [j for j in jm.list() if j.get("lane") == "inline"]  # a plan runs nothing


def test_a_failed_start_frame_is_a_failed_row_with_the_routes_line(lane, monkeypatch):
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)

    def broken(*a, **k):
        raise RuntimeError("ComfyUI went away")

    monkeypatch.setattr(sf, "make", broken)
    r = c.post("/api/previz/start_frame", json={"shot_id": shot_id})
    assert r.status_code == 502
    (row,) = [j for j in c.get("/api/jobs").json() if j.get("lane") == "inline"]
    assert row["status"] == "failed" and row["error"] == r.json()["detail"] and "draft_key" not in row


def test_the_look_frame_is_an_inline_row_too(lane, monkeypatch):
    c, store, server, jm, freed = lane
    prj = store.create_project("Diner")
    monkeypatch.setattr(store, "version_master", lambda shot_id, version_id=None: Path("unused.mov"))
    monkeypatch.setattr(store, "shot_meta", lambda shot_id: {"project_id": prj["id"], "name": "Booth"})
    monkeypatch.setattr(store, "shot_dir", lambda shot_id: store.root / "shots_fake" / shot_id)
    monkeypatch.setattr(store, "shot_cast", lambda shot_id: ([], lambda c: None))
    (store.root / "shots_fake" / "sh1").mkdir(parents=True)

    def fake_look(runner, st, playblast, refs, look, out, seed=42, at=0, commercial_safe=False):
        Image.new("RGB", (8, 8), (1, 2, 3)).save(out)
        return out

    monkeypatch.setattr(server.previz, "look_frame", fake_look)
    r = c.post("/api/previz/look", json={"shot_id": "sh1", "look_prompt": "A diner.", "seed": 5})
    assert r.status_code == 200 and set(r.json()) == {"url", "upload_id", "seed", "references"}
    (row,) = [j for j in c.get("/api/jobs").json() if j.get("lane") == "inline"]
    assert row["stage"] == "Done" and row["stages"] == ["Look frame"] and row["step"] == "look" and row["shot_name"] == "Booth"
    assert row["result"]["upload_id"] == r.json()["upload_id"] and jm._q.qsize() == 0 and freed == []


def test_cancel_stops_a_running_start_frame_and_keeps_nothing(lane, monkeypatch):
    """Run 8: the start frame's Cancel. The row is canceled, the route answers 409, its graphs carry the row's Stopper
    (so the quick runner stops this app's own prompt by id) and no picture is kept."""
    import threading
    import time

    from pickups.app import comfy

    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    seen = {}

    def slow_make(runner, plan, frame0, out, seed=42):
        seen["stopper"] = comfy.current_stopper()
        for _ in range(250):
            seen["stopper"].cancel.wait(0.02)
            if seen["stopper"].cancel.is_set():
                raise comfy.Canceled()
        Image.new("RGB", (16, 9)).save(out)
        return out

    monkeypatch.setattr(sf, "make", slow_make)

    def press_cancel():
        for _ in range(250):
            rows = [j for j in jm.list() if j.get("lane") == "inline" and j["status"] == "running"]
            if rows:
                seen["canceled"] = jm.cancel(rows[0]["id"])["id"]
                return
            time.sleep(0.02)

    t = threading.Thread(target=press_cancel)
    t.start()
    r = c.post("/api/previz/start_frame", json={"shot_id": shot_id, "seed": 7, "draft_key": f"{shot_id}:previz"})
    t.join()
    assert r.status_code == 409 and r.json()["detail"] == "Cancelled."
    (row,) = [j for j in c.get("/api/jobs").json() if j.get("lane") == "inline"]
    assert row["id"] == seen["canceled"] and row["status"] == "canceled" and not row.get("result")
    assert seen["stopper"] is not None and comfy.current_stopper() is None
    assert not list((store.shot_dir(shot_id) / "looks").glob("start_[0-9]*.png"))  # start_frame0.png is the input frame
    assert "took_s" not in row  # a canceled run is not a measured time


def test_the_quick_runner_hands_the_rows_stopper_to_its_graphs(lane, monkeypatch):
    from pickups.app import comfy

    c, store, server, jm, freed = lane
    got = []
    monkeypatch.setattr(server, "run_graph", lambda _client, graph, **k: got.append(comfy.current_stopper()))
    q = server.QuickRunner(comfy.Connection())
    q._client, q._gen = object(), q.conn.generation
    q.run({})  # outside a row: no stopper
    with jm.inline("previz", "nope", "Start frame"):
        q.run({})
    assert got[0] is None and isinstance(got[1], comfy.Stopper)
