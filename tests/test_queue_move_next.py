"""Move next (Run 14, walkthrough 25): a waiting run goes to the front of the GPU queue; the running one is untouched."""
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from pickups.app.jobs import JobManager, Stage, TaskSpec
from pickups.app.store import Store


def test_move_next_runs_that_job_right_after_the_running_one(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = Store(tmp_path / "library")
    clip = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.3", "-r", "24", str(clip)], check=True)
    sid = store.add_shot(clip, "Shot")["id"]
    order, gate = [], threading.Event()

    def fn(ctx):
        ctx.stage("Working")
        order.append(ctx.params["n"])
        if ctx.params["n"] == 1:
            gate.wait(5)  # the first one runs while the others wait
        return {"label": "Fake", "task": "fake", "downloads": [], "details": {}}

    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=lambda: None,
                           cancel_prompt=lambda pid: None)
    jm = JobManager(store, conn, {"fake": TaskSpec([Stage("Working", 1)], fn)})
    jobs = [jm.submit(sid, "fake", {"n": n}) for n in (1, 2, 3, 4)]
    for _ in range(100):
        if order:
            break
        time.sleep(0.05)
    with pytest.raises(ValueError):
        jm.move_next(jobs[0]["id"])  # running: never moves
    moved = jm.move_next(jobs[3]["id"])
    assert moved["next_at"] > 0
    gate.set()
    for _ in range(200):
        if all(jm.jobs[j["id"]]["status"] in ("done", "failed") for j in jobs):
            break
        time.sleep(0.05)
    assert order == [1, 4, 2, 3]
    with pytest.raises(ValueError):
        jm.move_next(jobs[1]["id"])  # done: nothing to move
