"""Step 23: Replace a person as a step flow. The render path is unchanged (same params, same stages), the page's draft
rides on the job record and never in params (so never in a take's receipt), and a sheet's views show as they are made
(TaskContext.partial)."""

import time

from PIL import Image

from pickups import sheets
from pickups.app.jobs import TaskContext, TaskSpec
from pickups.tasks import TASKS, recast
from test_inline_lane import _previz_shot, lane  # noqa: F401 - the fixture
from test_previz_e3 import client  # noqa: F401 - the fixture

PARAMS = {"points": [{"x": 320, "y": 180, "positive": True}], "frame": 12, "engine": "ltx", "cast_id": "cast_0123456789",
          "describe": "a knight in shining armour", "relight": True, "start_frame": 0, "frames": 97}


def _wait(c, job_id, s=10.0):
    end = time.monotonic() + s
    while time.monotonic() < end:
        job = next(j for j in c.get("/api/jobs").json() if j["id"] == job_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def test_a_recast_job_is_unchanged_and_the_draft_key_rides_on_the_job_not_in_params(lane, monkeypatch):  # noqa: F811
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    real = TASKS["recast"]
    monkeypatch.setitem(TASKS, "recast", TaskSpec(real.stages, lambda ctx: {"details": {}}, stages_for=real.stages_for))
    stages = [s.label for s in recast.stages_for(PARAMS)]

    old = c.post("/api/jobs", json={"shot_id": shot_id, "task": "recast", "params": PARAMS}).json()
    assert old["params"] == PARAMS and old["stages"] == stages and "draft_key" not in old and "step" not in old
    flow = c.post("/api/jobs", json={"shot_id": shot_id, "task": "recast", "params": PARAMS, "draft_key": f"{shot_id}:recast",
                                     "step": "render"}).json()
    assert flow["params"] == PARAMS and flow["stages"] == stages and flow["eta_s"] == old["eta_s"]
    assert flow["draft_key"] == f"{shot_id}:recast" and flow["step"] == "render"

    for j in (old, flow):
        done = _wait(c, j["id"])
        assert done["status"] == "done", done
        receipt = store.shot(shot_id)["versions"]
        take = next(v for v in receipt if v["id"] == done["result_version_id"])
        meta = (store.shot_dir(shot_id) / "versions" / take["id"] / "meta.json").read_text()
        assert "draft_key" not in meta and '"step"' not in meta


def test_a_blank_or_odd_draft_key_is_left_off(lane):  # noqa: F811
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    j = c.post("/api/jobs", json={"shot_id": shot_id, "task": "remove", "params": {}, "draft_key": "  ", "step": 3}).json()
    assert "draft_key" not in j and "step" not in j and j["params"] == {}
    c.post(f"/api/jobs/{j['id']}/cancel")


def test_partial_puts_each_view_on_the_job_as_it_is_made(lane, tmp_path):  # noqa: F811
    c, store, server, jm, freed = lane
    job = {"id": "job_partial", "params": {}, "shot_id": None, "project_id": None}
    ctx = TaskContext(jm, job, sheets.STAGES)
    pic = tmp_path / "front.png"
    Image.new("RGB", (8, 12), (200, 10, 10)).save(pic)
    ctx.partial("front", pic)
    ctx.partial("back", pic)
    assert set(job["partial"]) == {"front", "back"} and all(u.startswith("/media/") for u in job["partial"].values())
    ctx.partial("face", tmp_path / "missing.png")  # best effort: a file that isn't there changes nothing
    assert set(job["partial"]) == {"front", "back"}


def test_sheets_make_shows_front_back_and_close_up_in_order(tmp_path, monkeypatch):
    seen: list[str] = []

    class Assets:
        def from_file(self, p):
            return p

    class Ctx:
        client = type("C", (), {"assets": Assets()})()

        def stage(self, label):
            seen.append(label)

        def partial(self, key, path):
            assert path.exists()
            seen.append(key)

    def fake_edit(ctx, im, instruction, seed, commercial_safe, out_dir, view, asset=None):
        out = Image.new("RGB", (64, 96), (10, 10, 10))
        out.save(out_dir / f"{view}.png")
        return out

    monkeypatch.setattr(sheets, "_edit", fake_edit)
    monkeypatch.setattr(sheets, "front_view", lambda im: True)  # already frontal: no front edit
    pic = tmp_path / "in.png"
    Image.new("RGB", (64, 96), (90, 90, 90)).save(pic)
    sheets.make(Ctx(), tmp_path / "out", picture=pic)
    assert seen == ["front", sheets.STAGE_BACK, "back", sheets.STAGE_FACE, "face", sheets.STAGE_SHEET]

    class Plain(Ctx):  # a one-off runner without partial still builds the sheet
        partial = None

    seen.clear()
    sheets.make(Plain(), tmp_path / "out2", picture=pic)
    assert seen == [sheets.STAGE_BACK, sheets.STAGE_FACE, sheets.STAGE_SHEET]
