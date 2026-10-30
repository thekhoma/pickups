"""Benchmarks: every finished job adds a line to benchmarks.jsonl; old takes are backfilled once; the summary's maths;
the README section; what Setup → Benchmarks gets."""

import json
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from pickups.app import bench
from pickups.app.jobs import JobManager, Stage, TaskContext, TaskSpec
from pickups.app.store import Store

STATS = {"system": {"comfyui_version": "0.41.1"},
         "devices": [{"name": "cuda:0 NVIDIA RTX PRO 6000 Blackwell Server Edition : cudaMallocAsync", "vram_total": 102e9}]}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(bench, "_machines", {})
    monkeypatch.setattr(bench, "_cache", {"key": None, "groups": []})
    monkeypatch.delenv("PICKUPS_GPU_LABEL", raising=False)
    monkeypatch.delenv("PICKUPS_PROXY_PORT", raising=False)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    s = Store(tmp_path / "library")
    d = s.shot_dir("shot_test")
    (d / "versions").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"id": "shot_test", "name": "Test", "source": "source.mov", "width": 3840, "height": 2748,
                                             "fps": 24.0, "frames": 202, "duration_s": 8.4, "created_at": 0}))
    return s


def conn(calls: list | None = None, stats=STATS):
    def system_stats():
        if calls is not None:
            calls.append(1)
        return stats
    return SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=system_stats)


def wait(jm, job, lines_file=None, n=0):
    for _ in range(200):
        done = jm.jobs[job["id"]]["status"] in ("done", "failed", "canceled")
        if done and (lines_file is None or len(bench.load(lines_file)) >= n):
            return jm.jobs[job["id"]]
        time.sleep(0.02)
    raise AssertionError(jm.jobs[job["id"]])


# --- the machine ------------------------------------------------------------------------------------------------------
def test_gpu_name_and_host(monkeypatch):
    assert bench.gpu_name("cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync") == "RTX 4090"
    assert bench.gpu_name("cuda:0 NVIDIA RTX PRO 6000 Blackwell Server Edition : cudaMallocAsync") == \
        "RTX PRO 6000 Blackwell Server Edition"
    assert bench.host_kind("http://127.0.0.1:8188") == "local"
    assert bench.host_kind("http://gpu-pc:8188") == "remote"
    assert bench.host_kind("https://abc123xyz-8188.proxy.runpod.net") == "runpod"
    monkeypatch.setenv("PICKUPS_GPU_LABEL", "Lab")  # a tunnel to a rented GPU looks local: the label names it
    assert bench.host_kind("http://127.0.0.1:8288") == "lab"


def test_a_pc_on_the_network_is_remote_whatever_the_proxy_port_or_label(monkeypatch):
    # a Finish on a 4090 over Tailscale, with Pickups on its own API proxy port, used to say "runpod"
    monkeypatch.setenv("PICKUPS_PROXY_PORT", "8190")
    monkeypatch.setenv("PICKUPS_GPU_LABEL", "runpod")
    assert bench.host_kind("http://100.64.0.7:8188") == "remote"
    assert bench.host_kind("http://gpu-pc:8188") == "remote"


def test_machine_is_read_once_per_comfyui_and_never_raises():
    calls = []
    c = conn(calls)
    m = bench.machine(c)
    assert m == {"gpu": "RTX PRO 6000 Blackwell Server Edition", "vram_gb": 102.0, "comfy_version": "0.41.1", "host": "local"}
    bench.machine(c)
    assert len(calls) == 1
    offline = SimpleNamespace(comfy_url="http://gpu-pc:8188", system_stats=lambda: None)
    assert bench.machine(offline)["gpu"] is None and "http://gpu-pc:8188" not in bench._machines  # asked again next time
    assert bench.machine(SimpleNamespace(comfy_url="x"))["gpu"] is None  # no system_stats at all


def test_stage_seconds():
    ctx = TaskContext(SimpleNamespace(store=None, conn=None, _update=lambda *a, **k: None), {"params": {}}, [Stage("A", 1)])
    ctx.stage_log = [("A", 0.0), ("B", 2.0), ("A", 5.0)]
    assert ctx.stage_seconds(6.25) == {"A": 3.2, "B": 3.0}  # a stage entered twice adds up


# --- recording --------------------------------------------------------------------------------------------------------
def test_a_finished_take_records_its_run(store, tmp_path):
    def fn(ctx):
        ctx.stage("Finding the person")
        time.sleep(0.08)
        ctx.stage("Filling the gap")
        return {"label": "Remove", "task": "remove", "details": {
            "resolution": "3840×2160", "rendered_at": "3840×2748", "frames": 202, "fps": 24.0, "camera": "locked-off",
            "method": "real pixels from other frames"}}

    def broken(ctx):
        ctx.stage("Finding the person")
        raise RuntimeError("no person")

    stages = [Stage("Finding the person", 10), Stage("Filling the gap", 20)]
    calls = []
    jm = JobManager(store, conn(calls), {"remove": TaskSpec(stages, fn), "broken": TaskSpec(stages, broken)})
    jm.broadcast = lambda kind, data: None
    log = tmp_path / "benchmarks.jsonl"  # beside the library (PICKUPS_HOME)
    done = wait(jm, jm.submit("shot_test", "remove", {"frame": 10}), log, 1)
    assert done["status"] == "done", done.get("error")
    [run] = bench.load(log)
    assert run["task"] == "remove" and run["engine"] == "locked-off" and run["status"] == "done"
    assert run["gpu"] == "RTX PRO 6000 Blackwell Server Edition" and run["vram_gb"] == 102.0 and run["host"] == "local"
    assert run["comfy_version"] == "0.41.1"
    assert (run["frames"], run["width"], run["height"], run["fps"], run["res_class"]) == (202, 3840, 2748, 24.0, "4K")
    assert list(run["stages"]) == ["Finding the person", "Filling the gap"] and run["stages"]["Finding the person"] >= 0.05
    assert run["total_seconds"] >= run["gpu_seconds"] == 0.0
    assert run["shot_id"] == "shot_test" and run["take_id"] == done["result_version_id"] and run["job_id"] == done["id"]
    assert run["ts"].endswith("Z")
    details = store.version_meta("shot_test", done["result_version_id"])["details"]
    assert details["_bench"] == run and details["gpu"] == "RTX PRO 6000 Blackwell Server Edition"

    failed = wait(jm, jm.submit("shot_test", "broken", {}), log, 2)
    assert failed["status"] == "failed"
    run = bench.load(log)[1]
    assert run["status"] == "failed" and run["task"] == "broken" and "take_id" not in run
    assert (run["width"], run["height"], run["frames"]) == (3840, 2748, None)  # the shot's size; nothing was made
    assert list(run["stages"]) == ["Finding the person"]
    assert len(calls) == 1  # /system_stats once for this ComfyUI


def test_a_project_task_records_its_images(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = Store(tmp_path / "library")
    pid = store.create_project("Bench")["id"]

    def fn(ctx):
        ctx.stage("Making your images")
        out = []
        for i in range(2):
            Image.new("RGB", (1344, 768)).save(ctx.vdir / f"{i}.png")
            out.append({"path": str(ctx.vdir / f"{i}.png"), "model": "flux3-klein", "prompt": "a test"})
        return {"images": out}

    jm = JobManager(store, conn(), {"image": TaskSpec([Stage("Making your images", 5)], fn, scope="project")})
    jm.broadcast = lambda kind, data: None
    done = wait(jm, jm.submit(None, "image", {"prompt": "a test"}, project_id=pid), tmp_path / "benchmarks.jsonl", 1)
    assert done["status"] == "done", done.get("error")
    [run] = bench.load(tmp_path / "benchmarks.jsonl")
    assert run["task"] == "image" and run["engine"] == "flux3-klein" and run["frames"] == 2
    assert (run["width"], run["height"], run["res_class"]) == (1344, 768, "other")
    assert run["project_id"] == pid and run["image_ids"] == done["result_image_ids"] and "shot_id" not in run


# --- backfill ---------------------------------------------------------------------------------------------------------
def _take(store, vid, task, details, params=None, created_at=1.7e9):
    d = store.shot_dir("shot_test") / "versions" / vid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"id": vid, "task": task, "created_at": created_at, "details": details,
                                             "params": params or {}}))


def test_backfill_adds_each_old_take_once(store, tmp_path):
    _take(store, "v_old", "recast", {"resolution": "3840×2160", "frames": 81, "gpu_seconds": 139.3, "total_seconds": 243.5,
                                     "models": ["SAM 3.1", "SCAIL-2 (Wan 2.1)"]})
    _take(store, "v_pod", "remove", {"resolution": "3840×2160", "rendered_at": "3840×2748", "frames": 202, "fps": 24.0,
                                     "method": "real pixels from other frames", "gpu_seconds": 101.8, "total_seconds": 470.0})
    _take(store, "v_new", "previz", {"frames": 97, "total_seconds": 220.0, "_bench": {"take_id": "v_new"}})  # recorded live
    _take(store, "v_untimed", "previz", {"frames": 97})
    pods = tmp_path / "runpod" / "pod_takes.txt"
    assert bench.backfill(store, pod_takes=pods) and pods.read_text() == ""  # created, empty, when missing
    log = tmp_path / "benchmarks.jsonl"
    log.unlink()
    pods.write_text("v_pod\n# a comment\n")

    added = bench.backfill(store, pod_takes=pods)
    assert sorted(r["take_id"] for r in added) == ["v_old", "v_pod"]
    rows = {r["take_id"]: r for r in bench.load(log)}
    assert rows["v_old"]["gpu"] == "RTX 4090" and rows["v_old"]["host"] == "remote" and rows["v_old"]["engine"] == "scail"
    assert rows["v_old"]["backfill"] is True and rows["v_old"]["ts"] == bench._iso(1.7e9)
    assert rows["v_pod"]["gpu"] == "RTX PRO 6000 Blackwell Server Edition" and rows["v_pod"]["host"] == "runpod"
    assert (rows["v_pod"]["width"], rows["v_pod"]["height"], rows["v_pod"]["engine"]) == (3840, 2748, "locked-off")

    assert bench.backfill(store, pod_takes=pods) == []  # again: nothing twice
    assert len(bench.load(log)) == 2


# --- the summary ------------------------------------------------------------------------------------------------------
def _run(task, total, gpu_s, frames=97, gpu="RTX 4090", size=(1920, 1088), status="done", engine=None, cls=None):
    return {"task": task, "engine": engine, "status": status, "gpu": gpu, "host": "remote", "frames": frames,
            "width": size[0], "height": size[1], "res_class": cls or bench.res_class(*size), "gpu_seconds": gpu_s,
            "total_seconds": total, "ts": "2026-10-17T10:00:00Z"}


def test_aggregate_maths():
    rows = [_run("previz", 200, 180), _run("previz", 300, 260), _run("previz", 2700, 2680),
            _run("previz", 999, 1, status="failed"),  # only finished runs count
            _run("previz", 150, 120, gpu="RTX PRO 6000 Blackwell Server Edition"),
            _run("recast", 240, 140, frames=81, size=(3840, 2160), engine="scail"),
            _run("recast", 160, 120, frames=81, size=(1920, 1080), engine="scail")]
    groups = bench.aggregate(rows)
    assert [(g["task"], g["engine"], g["gpu"], g["res_class"]) for g in groups] == [
        ("recast", "scail", "RTX 4090", "4K"), ("recast", "scail", "RTX 4090", "HD"),
        ("previz", None, "RTX 4090", "HD"), ("previz", None, "RTX PRO 6000 Blackwell Server Edition", "HD")]
    p = groups[2]
    assert p["label"] == "Previz to final" and p["runs"] == 3
    assert (p["total_s"], p["gpu_s"], p["fastest_s"], p["slowest_s"]) == (300.0, 260.0, 200.0, 2700.0)
    assert p["s_per_frame"] == round(300 / 97, 2) and p["frames"] == 97 and p["size"] == "1920×1088"
    assert groups[0]["label"] == "Replace a person (SCAIL-2)"
    assert bench.res_class(3840, 2748) == "4K" and bench.res_class(1344, 768) == "other" and bench.res_class(None, None) == "other"


def test_status_rows_put_the_connected_gpu_first(tmp_path):
    ref = [{"task": "Recast", "seconds": 300, "frames": 81, "size": "1920x1088"}]
    log = tmp_path / "benchmarks.jsonl"
    assert bench.status_rows("RTX 5090", ref, 0.7, log) == [
        {"task": "Recast", "seconds": 210, "frames": 81, "size": "1920x1088", "gpu": "RTX 5090", "runs": 0, "measured": False, "current": True}]
    bench.append([_run("previz", 200, 180), _run("previz", 120, 100, gpu="RTX PRO 6000 Blackwell Server Edition"),
                  _run("previz", 130, 110, gpu="RTX PRO 6000 Blackwell Server Edition")], log)
    rows = bench.status_rows("RTX 4090", ref, 1.0, log)
    assert [(r["gpu"], r["current"], r["measured"]) for r in rows] == [
        ("RTX 4090", True, True), ("RTX PRO 6000 Blackwell Server Edition", False, True)]
    assert rows[0]["task"] == "Previz to final" and rows[0]["seconds"] == 200 and rows[1]["runs"] == 2
    # a GPU with no runs yet: its estimate, then the GPUs measured (most runs first)
    rows = bench.status_rows("RTX 5090", ref, 0.7, log)
    assert [(r["gpu"], r["measured"]) for r in rows] == [
        ("RTX 5090", False), ("RTX PRO 6000 Blackwell Server Edition", True), ("RTX 4090", True)]
    assert [r["gpu"] for r in bench.status_rows(None, ref, 1.0, log)] == ["RTX PRO 6000 Blackwell Server Edition", "RTX 4090"]


def test_write_and_markdown(tmp_path):
    bench.append([_run("previz", 200, 180), _run("previz", 260, 230), _run("finish", 1500, 1260, size=(3840, 2160), engine="best")],
                 tmp_path / bench.FILE)
    data = bench.write(tmp_path)
    assert json.loads((tmp_path / "bench.json").read_text()) == data
    assert data["runs"] == 3 and data["gpus"] == [{"gpu": "RTX 4090", "runs": 3, "vram_gb": None, "host": "remote"}]
    md = bench.markdown(data["rows"]).splitlines()
    assert md[0].startswith("| Task | GPU | Shot | Runs |")
    assert md[2] == "| Previz to final | RTX 4090 | 97 frames, 1920×1088 | 2 | 3.4 min | 3.8 min | 2.4 s | 3.3 min – 4.3 min |"
    assert md[3].startswith("| Finish, best quality | RTX 4090 | 97 frames, 3840×2160 | 1 | 21.0 min | 25.0 min |")


def test_readme_section_is_rewritten_between_the_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Pickups\n\nIntro.\n\n{bench.START}\nold table\n{bench.END}\n\nAfter.\n")
    groups = bench.aggregate([_run("previz", 200, 180)])
    bench.rewrite_readme(readme, groups)
    text = readme.read_text()
    assert "old table" not in text and text.startswith("# Pickups\n\nIntro.\n\n") and text.endswith(f"{bench.END}\n\nAfter.\n")
    assert "| Previz to final | RTX 4090 | 97 frames, 1920×1088 | 1 |" in text and "median of 1 finished run on 1 GPU" in text
    bench.rewrite_readme(readme, groups)
    assert readme.read_text() == text  # again: the same section, once
    (tmp_path / "plain.md").write_text("# No markers\n")
    with pytest.raises(ValueError):
        bench.rewrite_readme(tmp_path / "plain.md", groups)
