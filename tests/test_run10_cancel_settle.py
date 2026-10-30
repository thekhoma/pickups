"""Run 10 E: Cancel says Cancelled at once, but the next job waits until ComfyUI has really stopped the interrupted one.

R&D's Run 9 check: Finish showed "Cancelled" while ComfyUI kept loading a VAE for about 8 s before the interrupt
landed. The worker now holds the next job (still queued) until the prompt is gone from ComfyUI's /queue.
"""

import subprocess
import threading
import time
from types import SimpleNamespace

from pickups.app.comfy import Canceled, Connection
from pickups.app.jobs import JobManager, Stage, TaskSpec

from test_run8_seed_cancel_timings import FakeComfy  # noqa: E402 - the shared fake ComfyUI queue
from test_run9_cancel import _wait, store  # noqa: F401 - the fixture


def test_wait_stopped_returns_once_the_prompt_leaves_comfyui():
    f = FakeComfy(running=["p_stopping"])
    try:
        c = Connection(comfy_url=f.url)
        threading.Timer(0.4, lambda: f.running.clear()).start()  # the interrupt lands at the next node
        t = time.monotonic()
        assert c.wait_stopped("p_stopping", timeout=5, every=0.05)
        assert 0.3 < time.monotonic() - t < 3
        assert c.wait_stopped("p_gone")  # not there: at once
        assert c.wait_stopped(None)
        f.running.append("p_stuck")
        assert not c.wait_stopped("p_stuck", timeout=0.2, every=0.05)  # never holds the queue for ever
    finally:
        f.close()
    assert Connection(comfy_url="http://127.0.0.1:9").wait_stopped("p_any", timeout=1)  # ComfyUI offline: nothing runs


def test_the_next_job_starts_only_after_the_interrupted_prompt_stops(store, tmp_path):  # noqa: F811 - the fixture
    clip = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.5", "-r", "24", str(clip)], check=True)
    shot = store.add_shot(clip, "Shot")
    stopping = threading.Event()  # set: ComfyUI still runs the interrupted prompt
    started_at: dict[str, float] = {}
    in_graph = threading.Event()

    def finish(ctx):
        ctx.stage("Upscaling to 4K")
        ctx.stopper.prompt_id = "p_finish"  # what run_graph records for the graph it sent
        in_graph.set()
        ctx.cancel.wait(5)
        stopping.set()
        threading.Timer(0.5, stopping.clear).start()  # the VAE load ends, then the interrupt lands
        raise Canceled()

    def second(ctx):
        started_at["second"] = time.monotonic()
        started_at["comfy_busy"] = stopping.is_set()
        return {"label": "Finish", "task": "finish", "details": {}}

    def wait_stopped(pid):
        assert pid == "p_finish"
        while stopping.is_set():
            time.sleep(0.02)
        started_at["stopped"] = time.monotonic()
        return True

    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=lambda: None,
                           cancel_prompt=lambda pid: None, wait_stopped=wait_stopped)
    jm = JobManager(store, conn, {"finish": TaskSpec([Stage("Upscaling to 4K", 1)], finish), "second": TaskSpec([Stage("Go", 1)], second)})
    a = jm.submit(shot["id"], "finish", {"upscale_4k": True})
    assert in_graph.wait(5)
    b = jm.submit(shot["id"], "second", {})
    jm.cancel(a["id"])
    assert _wait(jm, a["id"])["status"] == "canceled"  # Cancelled at once
    assert jm.jobs[b["id"]]["status"] == "queued"  # while ComfyUI is still stopping
    assert _wait(jm, b["id"])["status"] == "done"
    assert started_at["comfy_busy"] is False and started_at["second"] >= started_at["stopped"]
