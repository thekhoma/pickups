"""Run 16: the GPU queue frees ComfyUI's memory before its first job (the inline lane's models may still be loaded),
then again only when the task changes."""
import subprocess
import time
from types import SimpleNamespace

from pickups.app.jobs import JobManager, Stage, TaskSpec
from pickups.app.store import Store


def test_memory_is_freed_before_the_first_job_and_on_each_task_change(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = Store(tmp_path / "library")
    clip = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.3", "-r", "24", str(clip)], check=True)
    sid = store.add_shot(clip, "Shot")["id"]
    events: list[str] = []

    def fn(ctx):
        ctx.stage("Working")
        events.append(ctx.job["task"])
        return {"label": "Fake", "task": ctx.job["task"], "downloads": [], "details": {}}

    conn = SimpleNamespace(free_memory=lambda: events.append("free"), client=lambda: None, comfy_url="http://127.0.0.1:9",
                           system_stats=lambda: None, cancel_prompt=lambda pid: None)
    spec = TaskSpec([Stage("Working", 1)], fn)
    jm = JobManager(store, conn, {"a": spec, "b": spec})
    jobs = [jm.submit(sid, t, {}) for t in ("a", "a", "b")]
    for _ in range(200):
        if all(jm.jobs[j["id"]]["status"] in ("done", "failed") for j in jobs):
            break
        time.sleep(0.05)
    assert events == ["free", "a", "a", "free", "b"]
