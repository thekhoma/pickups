"""Run 9: Cancel really stops a multi-step job (Finish carried on after its last graph and saved its take), and only ever
targets prompts this app sent."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pickups.app import comfy
from pickups.app.comfy import Canceled, Connection, Stopper, run_local
from pickups.app.jobs import JobManager, Stage, TaskSpec
from pickups.app.store import Store

from test_run8_seed_cancel_timings import FakeComfy  # noqa: E402 - the shared fake ComfyUI queue


@pytest.fixture()
def fake():
    made = []

    def make(**k):
        f = FakeComfy(**k)
        made.append(f)
        return f

    yield make
    for f in made:
        f.close()


def test_a_running_prompt_we_did_not_send_is_never_interrupted_even_by_its_id(fake):
    f = fake(running=["someone_elses"], pending=["also_theirs"])
    c = Connection(comfy_url=f.url)
    assert c.cancel_prompt("someone_elses") is None and c.cancel_prompt("also_theirs") is None
    assert f.posts == []  # no /interrupt, no /queue delete


def test_run_graph_marks_what_it_sends_as_ours():
    class Job:
        id = "p_run9_mark"

        def events(self):
            return iter(())

        def result(self):
            return self

    client = SimpleNamespace(submit=lambda wf: Job(), workflows=SimpleNamespace(from_json=lambda g: g))
    assert not comfy.is_ours("p_run9_mark")
    comfy.run_graph(client, {})
    assert comfy.is_ours("p_run9_mark")


def test_a_local_encode_stops_on_cancel():
    st = Stopper()
    comfy.set_current_stopper(st)
    try:
        threading.Timer(0.3, st.cancel.set).start()
        t = time.monotonic()
        with pytest.raises(Canceled):
            run_local([sys.executable, "-c", "import time; time.sleep(20)"])
        assert time.monotonic() - t < 3
    finally:
        comfy.set_current_stopper(None)


def test_a_local_step_that_fails_still_fails():
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        run_local([sys.executable, "-c", "raise SystemExit(3)"])


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    return Store(tmp_path / "library")


def _wait(jm, jid, states=("done", "failed", "canceled")):
    for _ in range(400):
        if jm.jobs[jid]["status"] in states:
            return jm.jobs[jid]
        time.sleep(0.02)
    raise AssertionError(jm.jobs[jid])


def test_cancel_after_the_last_graph_saves_no_take(store, tmp_path):
    """Finish's case: Cancel pressed while it writes its files (no graph running, no stage left to enter)."""
    clip = tmp_path / "in.mp4"
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.5", "-r", "24", str(clip)], check=True)
    shot = store.add_shot(clip, "Shot")
    started, release = threading.Event(), threading.Event()

    def fn(ctx):
        ctx.stage("Writing your files")
        started.set()
        release.wait(5)  # the long encode, during which Cancel is pressed
        return {"label": "Finish 4K", "task": "finish", "details": {}}

    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=lambda: None,
                           cancel_prompt=lambda pid: None)
    jm = JobManager(store, conn, {"finish": TaskSpec([Stage("Writing your files", 1)], fn)})
    job = jm.submit(shot["id"], "finish", {"upscale_4k": True, "hdr": False})
    assert started.wait(5)
    jm.cancel(job["id"])
    assert jm.jobs[job["id"]]["stage"] == "Canceling"
    release.set()
    done = _wait(jm, job["id"])
    assert done["status"] == "canceled" and not done.get("result_version_id")
    assert [v for v in store.shot(shot["id"])["versions"] if v["id"] != "orig"] == []


def test_finish_shows_only_the_stages_its_choice_runs():
    from pickups.tasks import finish

    labels = lambda p, up=None: [s.label for s in finish.stages_for(p, up)]  # noqa: E731
    assert labels({"upscale_4k": True, "hdr": False}) == ["Preparing the shot", "Upscaling to 4K", "Writing your files"]
    assert labels({"upscale_4k": False, "hdr": True}) == ["Preparing the shot", "Rebuilding the highlights", "Merging HDR at full resolution", "Writing your files"]
    assert labels({}) == [s.label for s in finish.STAGES]  # 4K HDR: all of them (old takes sent no params)
    assert "Upscaling to 4K" not in labels({"upscale_4k": True, "hdr": True}, up=False)  # a source already 4K
