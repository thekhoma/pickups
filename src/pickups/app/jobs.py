"""Job queue: one GPU worker runs tasks in order; every change is pushed to listeners (the UI's event stream).

A task declares human stages with an expected duration each ("Finding the person", 20 s). Progress and the
countdown come from those estimates, corrected by how fast the stages actually run on this machine.
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pickups.app import bench
from pickups.app.comfy import Canceled, Connection, Stopper, plain_error, run_graph, set_current_stopper
from pickups.app.timings import Timings, kind_of
from pickups import media
from pickups.app.store import Store, drop_work, new_id
from pickups.app.workflow_titles import workflow_title


@dataclass
class Stage:
    label: str
    est_s: float


class TaskContext:
    """What a task gets: paths, params, the Comfy client, and progress/cancel hooks."""

    def __init__(self, manager: "JobManager", job: dict, stages: list[Stage]):
        self.m = manager
        self.job = job
        self.store: Store = manager.store
        self.conn: Connection = manager.conn
        self.params: dict = job["params"]
        self.shot_id: str | None = job.get("shot_id")
        self.project_id: str | None = job.get("project_id")
        # Project-level tasks (image generation) have no shot and no input clip.
        self.input_path: Path | None = self.store.version_master(job["shot_id"], job.get("version_id")) if self.shot_id else None
        self.stages = stages
        self.cancel = threading.Event()
        # Cancel stops this job's own ComfyUI prompt by its id (Connection.cancel_prompt), never another one.
        self.stopper = Stopper(self.cancel, getattr(self.conn, "cancel_prompt", None))
        self._i = -1
        self._t0 = time.monotonic()
        self._stage_t0 = self._t0
        self._done_est = 0.0
        self._client = None
        self.gpu_seconds = 0.0
        self.workflows: list[dict] = []  # the ComfyUI workflows this task ran, in order
        self.last_activity = time.monotonic()  # any stage or progress update (the stall watchdog reads it)
        self.stage_log: list[tuple[str, float]] = []  # (label, start) of every stage entered, for the benchmarks
        self.qa: dict = {}  # added to the take's details._qa (e.g. common.checked_masks: tracking_from_check)

    def stage_seconds(self, end: float | None = None) -> dict[str, float]:
        """Seconds spent in each stage, up to `end` (now); a stage entered twice adds up."""
        end = time.monotonic() if end is None else end
        out: dict[str, float] = {}
        for (label, t), (_, nxt) in zip(self.stage_log, self.stage_log[1:] + [("", end)]):
            out[label] = out.get(label, 0.0) + max(0.0, nxt - t)
        return {k: round(v, 1) for k, v in out.items()}

    @property
    def client(self):
        if self._client is None:
            self._client = self.conn.client()
        return self._client

    def allow_rtx(self) -> bool:
        """NVIDIA RTX accelerators: only when the user opted in (they're proprietary) and the nodes are installed."""
        from pickups.app.comfy import load_settings

        if not load_settings().get("allow_nvidia_rtx"):
            return False
        if not hasattr(self, "_rtx"):
            self._rtx = "RTXVideoSuperResolution" in self.conn.node_types()
        return self._rtx

    def commercial_safe(self) -> bool:
        """Setup → Licences: edit frames with Apache-2.0 Qwen-Image-Edit 2511 instead of Qwen-Image 2.1 (non-commercial).
        A take's own `commercial_safe` param wins, so a take re-runs the way it was made."""
        from pickups.app.comfy import load_settings

        if "commercial_safe" in self.params:
            return bool(self.params["commercial_safe"])
        return bool(load_settings().get("commercial_safe"))

    def check(self) -> None:
        if self.cancel.is_set():
            raise Canceled()

    def stage(self, label: str) -> None:
        """Move to the stage with this label (stages may be skipped)."""
        self.check()
        idx = next(i for i, s in enumerate(self.stages) if s.label == label)
        self._done_est = sum(s.est_s for s in self.stages[:idx])
        self._i = idx
        self._stage_t0 = time.monotonic()
        self.last_activity = self._stage_t0
        self.stage_log.append((label, self._stage_t0))
        self.m._update(self.job, stage=label, progress=self._fraction(0.0), eta_s=self._eta(0.0))

    def partial(self, key: str, path: Path) -> None:
        """A picture the task has made before it ends (a sheet's front view): kept as an upload and put on the job as
        `partial[key]` (its URL), so the page can show it now. Best effort: never fails the task."""
        try:
            up = self.store.add_upload(Path(path).read_bytes(), Path(path).name)
            self.m._update(self.job, partial={**(self.job.get("partial") or {}), key: up["url"]})
        except Exception:  # noqa: BLE001 - a nicety, like the timecode
            pass

    def use_stages(self, stages: list[Stage]) -> None:
        """Swap the job's stage list mid-run, when the task learns what it is doing (Recast finds the clicked thing is a
        vehicle): the countdown and the UI's step list follow the new one; the stage under way keeps its place."""
        if [s.label for s in stages] == [s.label for s in self.stages]:
            return
        cur = self.stages[self._i].label if self._i >= 0 else None
        self.stages = stages
        labels = [s.label for s in stages]
        self._i = labels.index(cur) if cur in labels else min(self._i, len(stages) - 1)
        self._done_est = sum(s.est_s for s in stages[:max(self._i, 0)])
        self.m._update(self.job, stages=labels, progress=self._fraction(0.0), eta_s=self._eta(0.0))

    def progress(self, frac: float) -> None:
        self.last_activity = time.monotonic()
        if self.job.get("stalled"):
            self.m._update(self.job, stalled=False, stall_message=None)
        self.m._update(self.job, progress=self._fraction(frac), eta_s=self._eta(frac))

    def run(self, graph: dict):
        t = time.monotonic()
        try:
            job = run_graph(self.client, graph, on_progress=self.progress, cancel=self.cancel)  # the Stopper rides on the thread
        finally:
            self.gpu_seconds += time.monotonic() - t
        self._keep_workflow(job, graph)
        return job

    def _keep_workflow(self, job, graph: dict) -> None:
        """Save the exact API-format workflow ComfyUI ran (from its history: real node inputs and uploaded file names),
        so every take can be opened and tweaked in ComfyUI. Best effort: never fails a task."""
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.conn.comfy_url}/history/{job.id}", timeout=10) as r:  # noqa: S310
                item = json.loads(r.read()).get(job.id) or {}
            ran = item.get("prompt", [None, None, None])[2]
            if not ran:
                return
            prefixes = [n.get("inputs", {}).get("filename_prefix", "") for n in ran.values() if isinstance(n, dict)]
            step = next((x.split("/")[1] for x in prefixes if isinstance(x, str) and x.startswith("pickups/") and "/" in x[8:]), "step")
            self.workflows.append({"step": step, "graph": ran})
        except Exception:  # noqa: BLE001
            pass

    def _fraction(self, frac: float) -> float:
        total = sum(s.est_s for s in self.stages) or 1.0
        cur = self.stages[self._i].est_s if self._i >= 0 else 0.0
        return max(0.0, min(0.99, (self._done_est + cur * frac) / total))

    def _eta(self, frac: float) -> int:
        total = sum(s.est_s for s in self.stages)
        left = total - self._done_est - (self.stages[self._i].est_s * frac if self._i >= 0 else 0.0)
        elapsed = time.monotonic() - self._t0
        done = max(self._done_est + (self.stages[self._i].est_s * frac if self._i >= 0 else 0.0), 1e-6)
        speed = elapsed / done if done > 5 else 1.0  # seconds per estimated second, once there is signal
        return int(max(0.0, left * min(max(speed, 0.5), 3.0)))


class InlineRun:
    """What an inline run hands back: `result` ({upload_id, url, seed}) is set by the caller before the block ends."""

    def __init__(self, job: dict):
        self.job = job
        self.result: dict | None = None


class NotStoppable(Exception):
    """Cancel on a row that has nothing to stop (kept for callers; every inline row can be canceled now)."""


@dataclass
class TaskSpec:
    stages: list[Stage]
    fn: object  # fn(ctx: TaskContext) -> dict (version meta; for project tasks: {"images": [...]} or {"sheet": {...}})
    scope: str = "shot"  # "shot": makes a new take of a shot; "project": adds images to a project
    stages_for: object = None  # fn(params) -> list[Stage], when the stages depend on the job (Recast's engine)

    def stages_of(self, params: dict) -> list[Stage]:
        return self.stages_for(params) if self.stages_for else self.stages


class JobManager:
    def __init__(self, store: Store, conn: Connection, tasks: dict[str, TaskSpec]):
        self.store = store
        self.conn = conn
        self.tasks = tasks
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self._q: queue.Queue[str] = queue.Queue()
        self._ctx: dict[str, TaskContext] = {}
        self._inline_stop: dict[str, Stopper] = {}  # running inline rows, by job id
        self._listeners: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="pickups-gpu").start()
        threading.Thread(target=self._watchdog, daemon=True, name="pickups-watchdog").start()

    STALL_S = 600  # no progress from ComfyUI for this long → tell the user (a hung ComfyUI needs a restart)

    def _watchdog(self) -> None:
        while True:
            time.sleep(30)
            for job_id, ctx in list(self._ctx.items()):
                job = self.jobs.get(job_id)
                if not job or job.get("status") != "running" or job.get("stalled"):
                    continue
                quiet = time.monotonic() - ctx.last_activity
                if quiet > self.STALL_S:
                    self._update(job, stalled=True, stall_message=(
                        f"No progress from ComfyUI for {int(quiet // 60)} minutes. It may be stuck: restart ComfyUI on the "
                        "GPU machine, then run this again. Pickups carries on by itself if it starts moving."))

    # --- listeners (SSE) ---------------------------------------------------------------------------------------
    def listen(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.append((loop, q))
        return q

    def unlisten(self, q: asyncio.Queue) -> None:
        self._listeners = [x for x in self._listeners if x[1] is not q]

    def broadcast(self, kind: str, data: dict) -> None:
        for loop, q in list(self._listeners):
            loop.call_soon_threadsafe(q.put_nowait, (kind, data))

    def _update(self, job: dict, **fields) -> None:
        job.update(fields)
        self.broadcast("job", dict(job))

    # --- API ---------------------------------------------------------------------------------------------------
    def submit(self, shot_id: str | None, task: str, params: dict, version_id: str | None = None,
               project_id: str | None = None, draft_key: str | None = None, step: str | None = None) -> dict:
        """Queue a job. `draft_key` and `step` (the page's draft and flow step) ride on the job record, never in
        `params`, which every take's receipt copies."""
        if task not in self.tasks:
            raise KeyError(task)
        shot_name = None
        if shot_id:
            meta = self.store.shot_meta(shot_id)
            shot_name = meta.get("name")
            project_id = project_id or meta.get("project_id")
        project_name = None
        if project_id and self.store.project_exists(project_id):
            project_name = json.loads((self.store.project_dir(project_id) / "meta.json").read_text()).get("name")
        job = {"id": new_id("job"), "shot_id": shot_id, "project_id": project_id, "shot_name": shot_name, "project_name": project_name,
               "model": params.get("model"), "version_id": version_id, "task": task,
               "params": params, "result_image_ids": [],
               "status": "queued", "stage": "Waiting for the GPU", "stages": [s.label for s in self.tasks[task].stages_of(params)],
               "progress": 0.0, "eta_s": int(sum(s.est_s for s in self.tasks[task].stages_of(params))), "error": None,
               "result_version_id": None, "created_at": time.time(), **self._timing(task, params)}
        if draft_key:
            job["draft_key"] = draft_key
        if step:
            job["step"] = step
        with self._lock:
            self.jobs[job["id"]] = job
            self.order.append(job["id"])
        self.broadcast("job", dict(job))
        self._q.put(job["id"])
        return job

    def clock(self) -> Timings:
        """The measured run times (PICKUPS_HOME/timings.json, beside the library this queue works on)."""
        t = getattr(self, "timings", None)
        if t is None:
            t = self.timings = Timings(self.store.root.parent / "timings.json")
        return t

    def _timing(self, task: str, params: dict | None) -> dict:
        """The kind a job is timed as, and how long that kind usually takes here (None before a measured run)."""
        kind = kind_of(task, params)
        try:
            usual = self.clock().usual(kind)
        except Exception:  # noqa: BLE001 - a nicety; never fails a submit
            usual = None
        return {"kind": kind, "usual_s": usual}

    def _took(self, job: dict) -> float | None:
        """A finished job's measured seconds (started to now), counted for its kind. Done runs only."""
        start = job.get("started_at")
        if not start:
            return None
        took = round(max(0.0, time.time() - start), 1)
        self.clock().record(job.get("kind") or "", took)
        return took

    @contextmanager
    def inline(self, task: str, shot_id: str, label: str, draft_key: str | None = None, step: str | None = None,
               error: str = "Something went wrong.", kind: str | None = None):
        """The inline lane: work a route runs itself (the start frame, the look frame), shown as a job row so the queue
        and a reloaded page can see it. Never on the GPU queue, never `free_memory`, no take. `error` is the one line
        the route answers with when it fails; the row says the same. Cancel stops it (Canceled is raised in the route's
        thread at the next graph, and its prompt in ComfyUI is stopped); `kind` times it (timings.kind_of)."""
        shot_name = project_id = None
        try:
            meta = self.store.shot_meta(shot_id)
            shot_name, project_id = meta.get("name"), meta.get("project_id")
        except Exception:  # noqa: BLE001 - a label, never a reason to fail the work
            pass
        now = time.time()
        job = {"id": new_id("job"), "lane": "inline", "shot_id": shot_id, "project_id": project_id, "shot_name": shot_name,
               "project_name": None, "model": None, "version_id": None, "task": task, "params": {}, "result_image_ids": [],
               "status": "running", "stage": label, "stages": [label], "progress": 0.0, "eta_s": None, "error": None,
               "result_version_id": None, "created_at": now, "started_at": now}
        if kind:
            job["kind"] = kind
            try:
                job["usual_s"] = self.clock().usual(kind)
            except Exception:  # noqa: BLE001
                job["usual_s"] = None
        if draft_key:
            job["draft_key"] = draft_key
        if step:
            job["step"] = step
        with self._lock:
            self.jobs[job["id"]] = job
            self.order.append(job["id"])
        self.broadcast("job", dict(job))
        run = InlineRun(job)
        stopper = Stopper(threading.Event(), getattr(self.conn, "cancel_prompt", None))
        self._inline_stop[job["id"]] = stopper
        set_current_stopper(stopper)
        try:
            yield run
        except Canceled:
            self._update(job, status="canceled", stage="Canceled", eta_s=0, finished_at=time.time())
            raise
        except BaseException:
            if stopper.cancel.is_set():  # canceled mid-graph: ComfyUI's interrupt can surface as a failure
                self._update(job, status="canceled", stage="Canceled", eta_s=0, finished_at=time.time())
                raise Canceled() from None
            self._update(job, status="failed", error=error, eta_s=0, finished_at=time.time())
            raise
        finally:
            set_current_stopper(None)
            self._inline_stop.pop(job["id"], None)
        took = self._took(job) if kind else round(time.time() - now, 1)
        result = {**run.result, "took_s": took} if isinstance(run.result, dict) else run.result
        self._update(job, status="done", stage="Done", progress=1.0, eta_s=0, result=result, took_s=took, finished_at=time.time())

    def cancel(self, job_id: str) -> dict:
        """Stop a job: a queued one never starts; a running one stops its own ComfyUI prompt (removed from ComfyUI's
        queue if it hasn't started there, else interrupted) and saves nothing. Inline rows (start frame) too."""
        job = self.jobs[job_id]
        if job.get("lane") == "inline":
            stopper = self._inline_stop.get(job_id)
            if job["status"] == "running" and stopper is None:
                raise NotStoppable(job_id)
            if stopper is not None:
                stopper.cancel.set()
                self._update(job, stage="Canceling")
            return job
        if job["status"] == "queued":
            self._update(job, status="canceled", stage="Canceled")
        elif job["status"] == "running" and job_id in self._ctx:
            self._ctx[job_id].cancel.set()  # its prompt is stopped by id; local work stops at the next check
            self._update(job, stage="Canceling")
        return job

    def move_next(self, job_id: str) -> dict:
        """Right-click Move next in the queue: a waiting GPU job goes to the front, so it runs as soon as the GPU is free.
        The running job is never touched. ValueError for a job that isn't waiting in the GPU queue."""
        job = self.jobs[job_id]
        if job["status"] != "queued" or job.get("lane") == "inline":
            raise ValueError("Only a waiting run can move.")
        with self._q.mutex:
            waiting = self._q.queue
            if job_id not in waiting:
                raise ValueError("Only a waiting run can move.")
            waiting.remove(job_id)
            waiting.appendleft(job_id)
        # The page sorts the waiting runs by this (newest Move next first, then oldest first), as the worker takes them.
        self._update(job, next_at=time.time())
        return job

    def list(self) -> list[dict]:
        return [self.jobs[i] for i in reversed(self.order)]

    def _wait_stopped(self, ctx: TaskContext) -> None:
        """After a Cancel: Cancelled shows at once, but the worker takes the next job only once ComfyUI has really
        stopped this one's prompt (an interrupt lands at the next node), so two never overlap on the GPU (Run 10 E)."""
        wait = getattr(self.conn, "wait_stopped", None)
        if wait:
            try:
                wait(ctx.stopper.prompt_id)
            except Exception:  # noqa: BLE001 - never holds the queue on an error
                pass

    # --- worker ------------------------------------------------------------------------------------------------
    def _worker(self) -> None:
        last_task = None
        while True:
            job_id = self._q.get()
            job = self.jobs.get(job_id)
            if not job or job["status"] != "queued":
                continue
            if job["task"] != last_task:
                # A different task loads different models (LTX, SCAIL, SeedVR2, an image model...): let ComfyUI start
                # clean instead of juggling them on a 24 GB card. The first job too: the inline lane (start and look
                # frames) may have left its models loaded.
                self.conn.free_memory()
            last_task = job["task"]
            spec: TaskSpec = self.tasks[job["task"]]
            ctx = TaskContext(self, job, spec.stages_of(job["params"]))
            self._ctx[job_id] = ctx
            set_current_stopper(ctx.stopper)  # run_graph finds it: Cancel stops this job's own prompt by id
            self._use_gpu_limits()
            self._update(job, status="running", started_at=time.time())
            if spec.scope == "project":
                self._run_project_task(job, ctx, spec)
                continue
            try:
                vid, vdir = self.store.new_version_dir(job["shot_id"])
                ctx.vdir = vdir
                meta = spec.fn(ctx)
                ctx.check()  # canceled during the last local step (a download, an encode): nothing is saved
                meta.setdefault("details", {})["gpu_seconds"] = round(ctx.gpu_seconds, 1)
                if ctx.qa:
                    meta["details"]["_qa"] = {**(meta["details"].get("_qa") or {}), **ctx.qa}
                meta["details"]["total_seconds"] = round(time.monotonic() - ctx._t0, 1)
                meta["params"] = job["params"]
                meta["input_version"] = job.get("version_id") or "orig"
                meta["workflows"] = self._save_workflows(ctx, vdir)
                try:  # a take made from part of a long shot says so: "the first 4.0 s of 12.3 s"
                    made = media.made_from(media.probe(ctx.input_path), int(job["params"].get("start_frame", 0)),
                                           int(meta["details"].get("frames") or 0))
                    if made:
                        meta["details"]["made_from"] = made
                except Exception:  # noqa: BLE001 - a nicety, like the timecode
                    pass
                try:  # keep the source's start timecode (plus any trim offset) so a take lands over the original
                    tc = media.timecode(ctx.input_path)
                    if tc:
                        tc = media.offset_timecode(tc, int(job["params"].get("start_frame", 0)), media.probe(ctx.input_path).fps)
                        media.stamp_timecode(vdir, tc)
                        meta["details"]["timecode"] = tc
                except Exception:  # noqa: BLE001 - timecode is a nicety; never fail a finished take over it
                    pass
                run = self._bench(job, ctx, "done", details=meta["details"], take_id=vid)
                if run:  # the full line stays out of the take's Details (leading "_"); the GPU's name shows
                    meta["details"]["_bench"] = run
                    if run.get("gpu"):
                        meta["details"]["gpu"] = run["gpu"]
                took = self._took(job)
                version = self.store.save_version(job["shot_id"], vid, meta)
                try:  # Save videos to: a copy in the user's folder, per project (savecopy.py); never fails the take
                    from pickups.app.comfy import load_settings
                    from pickups.app.savecopy import copy_take_later
                    copy_take_later(self.store, job["shot_id"], vid, load_settings().get("save_dir"))
                except Exception:  # noqa: BLE001
                    pass
                self._log_bench(run)
                drop_work(vdir / "work")  # gigabytes of intermediate frames; kept with PICKUPS_KEEP_WORK
                self._update(job, status="done", stage="Done", progress=1.0, eta_s=0, result_version_id=version["id"],
                             took_s=took, finished_at=time.time())
                self.broadcast("shot", self.store.shot(job["shot_id"]))
            except Canceled:
                self._update(job, status="canceled", stage="Canceled", eta_s=0)
                self._log_bench(self._bench(job, ctx, "canceled"))
                self.store.discard_version_dir(getattr(ctx, "vdir", None))  # a take that never saved (no meta.json): its work frames go
                self._wait_stopped(ctx)
            except Exception as e:  # noqa: BLE001 - report every failure to the UI
                traceback.print_exc()
                said, raw = plain_error(e, self.conn.system_stats)
                self._update(job, status="failed", stage="Something went wrong", error=said, error_detail=raw[:4000], eta_s=0)
                self._log_bench(self._bench(job, ctx, "failed"))
                self.store.discard_version_dir(getattr(ctx, "vdir", None))
            finally:
                self._ctx.pop(job_id, None)
                set_current_stopper(None)

    # --- benchmarks (pickups/app/bench.py) -----------------------------------------------------------------------------
    def _bench(self, job: dict, ctx: TaskContext, status: str, details: dict | None = None, **fields) -> dict | None:
        """The benchmark line for a finished job: task, engine, GPU, size, per-stage seconds, GPU and total time. Never
        raises: a benchmark must not fail a job."""
        try:
            now = time.monotonic()  # before asking ComfyUI for its GPU (an offline one takes a few seconds to say so)
            d = details or {}
            stages, total = ctx.stage_seconds(now), d.get("total_seconds", now - ctx._t0)
            shot = self.store.shot_meta(job["shot_id"]) if job.get("shot_id") else None
            return bench.run_entry(task=job["task"], status=status, machine=bench.machine(self.conn), params=job["params"],
                                   details=d, stages=stages, gpu_seconds=d.get("gpu_seconds", ctx.gpu_seconds),
                                   total_seconds=total, shot=shot, shot_id=job.get("shot_id"), project_id=job.get("project_id"),
                                   job_id=job["id"], **fields)
        except Exception:  # noqa: BLE001
            return None

    def _log_bench(self, run: dict | None) -> None:
        """Add the line to PICKUPS_HOME/benchmarks.jsonl (beside the library this queue works on)."""
        if run:
            try:
                bench.append(run, bench.path(self.store.root.parent))
            except Exception:  # noqa: BLE001 - a full disk or a read-only folder must not fail a job
                pass

    @staticmethod
    def _made(added: list[dict], result: dict) -> dict:
        """A project task's output for its benchmark: how many images (or sheet views), their size, the image model."""
        if added:
            return {"frames": len(added), "width": added[0].get("width"), "height": added[0].get("height"),
                    "engine": added[0].get("model"), "image_ids": [a["id"] for a in added]}
        views = list(((result.get("sheet") or {}).get("views") or {}).values())
        if not views:
            return {}
        try:
            from PIL import Image

            with Image.open(views[0]) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            w = h = None
        return {"frames": len(views), "width": w, "height": h}

    @staticmethod
    def _save_workflows(ctx: TaskContext, vdir: Path) -> list[dict]:
        out = []
        d = vdir / "workflows"
        frame = ctx.params.get("frame")  # the frame the user clicked (Remove, Recast): names the pick and the tracks
        for i, w in enumerate(ctx.workflows):
            d.mkdir(exist_ok=True)
            name = f"{i + 1:02d}_{w['step']}.json"
            (d / name).write_text(json.dumps(w["graph"], indent=1))
            out.append({"file": f"workflows/{name}", "step": w["step"], "title": workflow_title(w["graph"], w["step"], frame)})
        return out

    def _use_gpu_limits(self) -> None:
        """Tasks read the longest take this GPU renders (tasks.common.MAX_FRAMES): set it for the connected card."""
        try:
            from pickups.app import bench
            from pickups.tasks import common

            common.use_gpu(bench.machine(self.conn).get("vram_gb"))
        except Exception:  # noqa: BLE001 - an unknown card keeps the default
            pass

    def _run_project_task(self, job: dict, ctx: TaskContext, spec: TaskSpec) -> None:
        """Image generation and other project-level work: results become project images (or a cast member's character
        sheet), not takes."""
        work = self.store.project_dir(job["project_id"]) / "images" / f"_work_{job['id']}"
        work.mkdir(parents=True, exist_ok=True)
        ctx.vdir = work
        try:
            result = spec.fn(ctx)
            ctx.check()  # canceled after its last graph: nothing is saved
            took = round(max(0.0, time.time() - job.get("started_at", time.time())), 1)  # measured: started to done
            added = [self.store.add_image(job["project_id"], Path(im.pop("path")), im | {"job_id": job["id"], "took_s": took})
                     for im in result.get("images", [])]
            ids = [a["id"] for a in added]
            made = {}
            if sheet := result.get("sheet"):  # a character sheet: kept with a new cast member, or the one it upgrades
                cast = self.store.save_sheet(job["project_id"], sheet["views"], sheet["sheet"], cast_id=sheet.get("cast_id"),
                                             name=sheet.get("name"), line=sheet.get("line", ""), source=sheet.get("source"),
                                             meta=sheet.get("meta", {}) | {"job_id": job["id"], "took_s": took})
                made = {"result_cast_id": cast["id"]}
            shot = None
            if new := result.get("shot"):  # a new video (text or image to video): a shot of its own, ready to edit
                shot = self.store.add_shot(Path(new["path"]), new["name"], move=True, project_id=job["project_id"],
                                           extra={"made": new.get("made", {}) | {"job_id": job["id"], "took_s": took}})
                for name in new.get("keep", []):
                    shutil.copyfile(work / name, self.store.shot_dir(shot["id"]) / name)
                made = {"result_shot_id": shot["id"]}
            bench_made = self._made(added, result) or ({"frames": shot["frames"], "width": shot["width"], "height": shot["height"]} if shot else {})
            run = self._bench(job, ctx, "done", cast_id=made.get("result_cast_id"), **bench_made)
            self.clock().record(job.get("kind") or "", took)
            self._update(job, status="done", stage="Done", progress=1.0, eta_s=0, result_image_ids=ids, finished_at=time.time(),
                         gpu_seconds=round(ctx.gpu_seconds, 1), took_s=took, **made)
            if shot:
                self.broadcast("shot", shot)
            self.broadcast("project", {"id": job["project_id"], "images": ids, **({"cast": [made["result_cast_id"]]} if "result_cast_id" in made else {}),
                                       **({"shots": [shot["id"]]} if shot else {})})
            self._log_bench(run)
        except Canceled:
            self._update(job, status="canceled", stage="Canceled", eta_s=0)
            self._log_bench(self._bench(job, ctx, "canceled"))
            self._wait_stopped(ctx)
        except Exception as e:  # noqa: BLE001 - report every failure to the UI
            traceback.print_exc()
            said, raw = plain_error(e, self.conn.system_stats)
            # A failure can say more about itself on the job (an image the model refused: `blocked`).
            self._update(job, status="failed", stage="Something went wrong", error=said, error_detail=raw[:4000], eta_s=0,
                         **(getattr(e, "job_fields", None) or {}))
            self._log_bench(self._bench(job, ctx, "failed"))
        finally:
            self._ctx.pop(job["id"], None)
            set_current_stopper(None)
            drop_work(work)  # R&D #16: kept with PICKUPS_KEEP_WORK, as the store keeps take folders

