"""Run 8: Lock seed (the seed used goes on the result; a locked one is sent again), Cancel (only this app's own ComfyUI
prompt, removed from the queue or interrupted; nothing half-made is kept) and measured render times."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from pickups import seeds
from pickups.app import comfy
from pickups.app.comfy import Canceled, Connection, Stopper, run_graph
from pickups.app.jobs import JobManager, Stage, TaskSpec
from pickups.app.store import Store
from pickups.app.timings import Timings, kind_of


# --- seeds -------------------------------------------------------------------------------------------------------------
def test_a_sent_seed_is_used_and_none_picks_a_fresh_one():
    assert seeds.pick(12) == 12 and seeds.pick("12") == 12 and seeds.pick(" 7 ") == 7 and seeds.pick(0) == 0
    for blank in (None, "", "None", "  "):
        n = seeds.pick(blank)
        assert isinstance(n, int) and 1 <= n <= seeds.MAX
    assert len({seeds.pick() for _ in range(20)}) > 1  # random each time
    with pytest.raises(ValueError):
        seeds.pick(-1)
    with pytest.raises(ValueError):
        seeds.pick("abc")


def test_image_and_video_tasks_use_the_seed_they_were_given():
    from pickups import video
    assert video.request({"prompt": "A path", "seed": 99})["seed"] == 99
    assert isinstance(video.request({"prompt": "A path"})["seed"], int)


# --- a fake ComfyUI: /queue and /interrupt --------------------------------------------------------------------------------
class FakeComfy:
    """ComfyUI's queue endpoints: GET /queue lists running and pending prompts; POST /queue {"delete": [...]} and POST
    /interrupt are recorded."""

    def __init__(self, running=(), pending=()):
        self.running, self.pending, self.posts = list(running), list(pending), []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                if self.path != "/queue":
                    self.send_error(404)
                    return
                body = json.dumps({"queue_running": [[0, p, {}, {}, []] for p in fake.running],
                                   "queue_pending": [[i + 1, p, {}, {}, []] for i, p in enumerate(fake.pending)]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(n) or b"{}")
                fake.posts.append((self.path, data))
                if self.path == "/queue":
                    fake.pending = [p for p in fake.pending if p not in data.get("delete", [])]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


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


def test_a_waiting_prompt_of_ours_is_removed_from_the_queue(fake):
    f = fake(running=["theirs"], pending=["p_other", "p_ours"])
    comfy.mark_ours("p_ours")  # run_graph marks every prompt it submits
    assert Connection(comfy_url=f.url).cancel_prompt("p_ours") == "removed"
    assert f.posts == [("/queue", {"delete": ["p_ours"]})]
    assert f.pending == ["p_other"]  # nobody else's prompt touched


def test_our_running_prompt_is_interrupted_by_its_id(fake):
    f = fake(running=["p_ours"], pending=["p_next"])
    comfy.mark_ours("p_ours")
    assert Connection(comfy_url=f.url).cancel_prompt("p_ours") == "interrupted"
    assert f.posts == [("/interrupt", {"prompt_id": "p_ours"})]


def test_a_job_we_did_not_send_is_never_interrupted(fake):
    f = fake(running=["someone_elses"], pending=[])
    c = Connection(comfy_url=f.url)
    assert c.cancel_prompt("p_ours") is None  # ours already finished: leave theirs running
    assert c.cancel_prompt("") is None
    assert f.posts == []


def test_offline_comfy_is_not_an_error():
    assert Connection(comfy_url="http://127.0.0.1:9").cancel_prompt("p_1") is None


# --- run_graph honours Cancel while ComfyUI says nothing -------------------------------------------------------------
class SilentJob:
    """A prompt waiting in ComfyUI's queue: no events until it is canceled."""

    def __init__(self, pid="p_1"):
        self.id, self.canceled, self._done = pid, 0, threading.Event()

    def events(self):
        self._done.wait(5)
        return iter(())

    def cancel(self):
        self.canceled += 1
        self._done.set()

    def result(self):
        raise RuntimeError("ended canceled")


class FakeClient:
    def __init__(self, job):
        self.job = job
        self.workflows = SimpleNamespace(from_json=lambda g: g)

    def submit(self, wf):
        return self.job


def test_cancel_stops_our_prompt_by_id_even_with_no_events():
    job, stopped = SilentJob("p_42"), []
    st = Stopper(threading.Event(), stopped.append)
    threading.Timer(0.3, st.cancel.set).start()
    t = time.monotonic()
    with pytest.raises(Canceled):
        run_graph(FakeClient(job), {}, stopper=st)
    assert time.monotonic() - t < 2
    assert st.prompt_id == "p_42" and stopped == ["p_42"] and job.canceled == 1


def test_a_canceled_stopper_never_submits():
    st = Stopper()
    st.cancel.set()
    client = SimpleNamespace(submit=lambda wf: pytest.fail("submitted"), workflows=SimpleNamespace(from_json=lambda g: g))
    with pytest.raises(Canceled):
        run_graph(client, {}, stopper=st)


# --- the queue: Cancel on a running image job keeps nothing; a queued one never runs -----------------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    return Store(tmp_path / "library")


def _manager(store, fn, stopped):
    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: FakeClient(SilentJob("p_img")),
                           comfy_url="http://127.0.0.1:9", system_stats=lambda: None, cancel_prompt=stopped.append)
    spec = TaskSpec(stages=[Stage("Making your images", 10)], fn=fn, scope="project")
    return JobManager(store, conn, {"image": spec, "slow": TaskSpec(stages=[Stage("Wait", 1)], fn=lambda ctx: time.sleep(0.5) or {"images": []}, scope="project")})


def _wait(jm, jid, states=("done", "failed", "canceled")):
    for _ in range(300):
        if jm.jobs[jid]["status"] in states:
            return jm.jobs[jid]
        time.sleep(0.02)
    raise AssertionError(jm.jobs[jid])


def test_cancel_a_running_job_stops_its_prompt_and_saves_nothing(store):
    pid = store.create_project("P")["id"]
    stopped = []

    def fn(ctx):
        ctx.stage("Making your images")
        ctx.run({})  # waits on the silent prompt until canceled
        return {"images": [{"path": "never"}]}

    jm = _manager(store, fn, stopped)
    job = jm.submit(None, "image", {"prompt": "x", "seed": 3}, project_id=pid)
    _wait(jm, job["id"], ("running",))
    time.sleep(0.3)
    jm.cancel(job["id"])
    done = _wait(jm, job["id"])
    assert done["status"] == "canceled" and done["result_image_ids"] == []
    assert stopped == ["p_img"]  # this app's own prompt, by id
    assert store.images(pid) == []
    assert jm.clock().runs(job["kind"]) == []  # a canceled run is never a measured time


def test_cancel_a_queued_job_never_runs_it(store):
    pid = store.create_project("P")["id"]
    ran = []
    jm = _manager(store, lambda ctx: ran.append(1) or {"images": []}, [])
    first = jm.submit(None, "slow", {}, project_id=pid)
    second = jm.submit(None, "image", {"prompt": "x"}, project_id=pid)
    jm.cancel(second["id"])
    assert jm.jobs[second["id"]]["status"] == "canceled"
    _wait(jm, first["id"])
    time.sleep(0.1)
    assert ran == [] and jm.jobs[second["id"]]["status"] == "canceled"


# --- measured times ------------------------------------------------------------------------------------------------------
def test_usual_is_the_median_of_the_last_measured_runs(tmp_path):
    t = Timings(tmp_path / "timings.json")
    assert t.usual("image|z|16:9|1K|1") is None
    t.record("image|z|16:9|1K|1", 10)
    assert t.usual("image|z|16:9|1K|1") == 10
    for s in (50, 12, 11, 9, 13):
        t.record("image|z|16:9|1K|1", s)
    assert t.usual("image|z|16:9|1K|1") == 12  # the last five: 50, 12, 11, 9, 13
    t.record("x", 0)
    t.record("", 5)
    assert t.runs("x") == [] and t.runs("") == []
    for _ in range(30):
        t.record("k", 1)
    assert len(t.runs("k")) == 10


def test_a_kind_is_the_task_model_and_size():
    a = kind_of("image", {"model": "z-image-turbo", "aspect": "16:9", "resolution": "1K", "count": 1})
    assert a == "image|z-image-turbo|16:9|1K|1"
    assert a != kind_of("image", {"model": "z-image-turbo", "aspect": "16:9", "resolution": "2K", "count": 1})
    assert kind_of("video", {"mode": "text", "width": 1280, "height": 704, "frames": 97}) == "video|text|1280x704|97"
    assert kind_of("start_frame", {"method": "zbuf"}) == "start_frame|zbuf"
    assert kind_of("recast", {"engine": "scail"}) == "recast|scail"


def test_a_done_job_says_how_long_it_took_and_the_next_one_how_long_it_usually_takes(store):
    pid = store.create_project("P")["id"]

    def fn(ctx):
        time.sleep(0.2)
        return {"images": []}

    jm = _manager(store, fn, [])
    params = {"model": "z-image-turbo", "aspect": "1:1", "resolution": "1K", "count": 1, "prompt": "x"}
    first = jm.submit(None, "image", params, project_id=pid)
    assert first["usual_s"] is None  # no measured run yet: no "usually"
    done = _wait(jm, first["id"])
    assert done["status"] == "done" and done["took_s"] >= 0.2
    second = jm.submit(None, "image", params, project_id=pid)
    assert second["kind"] == first["kind"] and second["usual_s"] == 1  # whole seconds, at least 1
    other = jm.submit(None, "image", params | {"resolution": "2K"}, project_id=pid)
    assert other["usual_s"] is None  # another size is another kind


def test_the_image_route_puts_the_seed_on_the_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    import pickups.app.store as st

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    s = Store(tmp_path / "library")
    for k, v in (("ROOT", s.root), ("SAMPLES", tmp_path / "samples"), ("EXAMPLES", tmp_path / "examples")):
        monkeypatch.setattr(server, k, v)
    monkeypatch.setattr(server, "Store", lambda: s)
    queued = []
    monkeypatch.setattr(server.JobManager, "submit", lambda self, shot_id, task, params, *a, **k: queued.append(params) or {"id": "j", "params": params})
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    pid = s.create_project("P")["id"]
    assert c.post(f"/api/projects/{pid}/images/generate", json={"prompt": "x", "seed": 5, "seed_locked": True}).status_code == 200
    assert c.post(f"/api/projects/{pid}/images/generate", json={"prompt": "x"}).status_code == 200
    assert c.post(f"/api/projects/{pid}/images/generate", json={"prompt": "x", "seed": -3}).status_code == 400
    assert queued[0]["seed"] == 5 and queued[0]["seed_locked"] is True
    assert isinstance(queued[1]["seed"], int) and queued[1]["seed"] >= 1
    assert c.get("/api/timings", params={"kind": "image|z|1:1|1K|1"}).json() == {"kind": "image|z|1:1|1K|1", "usual_s": None, "runs": 0}


def test_the_image_task_keeps_the_seed_and_the_lock_on_each_image(tmp_path):
    from pickups.tasks import image as task

    class Out:
        name = "a.png"

        def to_file(self, dst):
            dst.write_bytes(b"png")

    class Ctx:
        params = {"model": "z-image-turbo", "prompt": "a loch", "seed": 77, "seed_locked": True, "count": 2}
        project_id = None
        vdir = tmp_path
        store = SimpleNamespace(live_refs=lambda pid, refs: [], cast=lambda pid: [])
        conn = SimpleNamespace(model_files=lambda: set())
        client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: p))

        def stage(self, label):
            pass

        def progress(self, f):
            pass

        def run(self, g):
            return SimpleNamespace(get_outputs=lambda node: [Out()])

    out = task.run(Ctx())["images"]
    assert [o["seed"] for o in out] == [77, 78] and all(o["seed_locked"] is True for o in out)


def test_old_jobs_without_a_stopper_still_cancel(monkeypatch):
    # comfy.Stopper is optional on run_graph: the old cancel= Event still works.
    job = SilentJob()
    ev = threading.Event()
    threading.Timer(0.2, ev.set).start()
    with pytest.raises(Canceled):
        run_graph(FakeClient(job), {}, cancel=ev)
    assert job.canceled == 1
    assert comfy.current_stopper() is None


# --- licences (README Licences section) ------------------------------------------------------------------------------
def test_krea_2s_revenue_cap_is_said_where_the_app_names_its_terms():
    from pickups import licences
    assert "$1M" in licences.terms("Krea 2 Community License") and "$10M" in licences.terms(licences.LTX)
    assert licences.licence("Krea 2 Turbo")["commercial"] is None  # it depends on its terms
