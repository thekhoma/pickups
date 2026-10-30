"""New video (Create: text to video, image to video) on LTX-2.5: the size and length rules, the two graphs (PU_* node
inputs as in the tested recipe), the task with a fake ComfyUI, the job that turns the video into a shot, and the route.
No ComfyUI needed."""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

import pytest

from pickups import graphs, video
from pickups.app import manifest

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = {("16:9", "720p"): (1280, 704), ("16:9", "1080p"): (1920, 1088), ("9:16", "720p"): (704, 1280),
            ("1:1", "720p"): (960, 960), ("4:3", "720p"): (1024, 768), ("21:9", "720p"): (1344, 576)}


# --- sizes and lengths ---------------------------------------------------------------------------------------------------

def test_only_the_verified_sizes_are_offered():
    offered = {(a, r): s for r, row in video.SIZES.items() for a, s in row.items()}
    assert offered == VERIFIED
    for w, h in offered.values():
        assert video.valid_size(w, h) and (w // 2) % 32 == 0 and (h // 2) % 32 == 0


def test_every_aspect_has_720p_and_1080p_is_16_9_only():
    assert all(video.size_for(a, "720p") for a in video.ASPECTS)
    assert video.size_for("16:9", "1080p") == (1920, 1088)
    with pytest.raises(ValueError, match="1080p is 16:9 only"):
        video.size_for("9:16", "1080p")
    with pytest.raises(ValueError):
        video.size_for("4:5", "720p")
    with pytest.raises(ValueError):
        video.size_for("16:9", "4K")


def test_durations_are_8n_plus_1_at_24_fps_and_never_over_97():
    assert video.DURATIONS == {1: 25, 2: 49, 3: 73, 4: 97}
    for s, n in video.DURATIONS.items():
        assert (n - 1) % 8 == 0 and n <= video.MAX_FRAMES == 97 and n == s * video.FPS + 1
        assert video.frames_for(s) == n and video.frames_for(str(s)) == n
    for bad in (0, 5, 1.5, "x", None):
        with pytest.raises(ValueError):
            video.frames_for(bad)


@pytest.mark.parametrize(("frames", "ok"), [(9, True), (25, True), (97, True), (96, False), (105, False), (1, False), (24, False)])
def test_valid_length(frames, ok):
    assert video.valid_length(frames) is ok


def test_valid_size_needs_multiples_of_64():
    assert video.valid_size(1280, 704) and not video.valid_size(1280, 720) and not video.valid_size(1920, 1080)


def test_request_fills_defaults_and_checks():
    r = video.request({"prompt": "  A quiet forest path  ", "seed": 7})
    assert (r["mode"], r["aspect"], r["resolution"], r["width"], r["height"], r["frames"], r["fps"], r["seed"], r["image"]) == (
        "text", "16:9", "1080p", 1920, 1088, 97, 24, 7, None)
    assert r["prompt"] == "A quiet forest path" and r["model"] == "ltx-2.5"
    with pytest.raises(ValueError, match="Describe the shot"):
        video.request({"prompt": " "})
    with pytest.raises(ValueError, match="Describe the motion"):
        video.request({"mode": "image", "prompt": ""})
    with pytest.raises(ValueError, match="Choose a picture"):
        video.request({"mode": "image", "prompt": "She turns"})
    i = video.request({"mode": "image", "prompt": "She turns", "aspect": "9:16", "resolution": "720p", "duration": 2,
                       "image": {"kind": "image", "id": "img_1", "extra": 1}})
    assert (i["width"], i["height"], i["frames"], i["image"]) == (704, 1280, 49, {"kind": "image", "id": "img_1"})
    with pytest.raises(ValueError):
        video.request({"mode": "sound", "prompt": "x"})


def test_options_say_the_same():
    o = video.options()
    assert o["sizes"]["1080p"] == {"16:9": [1920, 1088]} and o["durations"][-1] == {"seconds": 4, "frames": 97} and o["fps"] == 24


def test_the_web_table_matches():
    """web/src/lib/video.ts offers the same sizes and lengths (the pickers and demo mode read it)."""
    ts = (ROOT / "web" / "src" / "lib" / "video.ts").read_text()
    for res, row in video.SIZES.items():
        line = next(x for x in ts.splitlines() if x.strip().startswith(f'"{res}":'))
        assert {a: (int(w), int(h)) for a, w, h in re.findall(r'"(\d+:\d+)": \[(\d+), (\d+)\]', line)} == row
    assert [(int(s), int(f)) for s, f in re.findall(r"\{ seconds: (\d+), frames: (\d+) \}", ts)] == list(video.DURATIONS.items())


# --- graphs -------------------------------------------------------------------------------------------------------------

def test_text_to_video_graph():
    g = graphs.ltx_text_to_video("A slow dolly", 1280, 704, 73, seed=11, fps=24)
    assert g["PU_Prompt"]["inputs"]["text"] == "A slow dolly" and g["PU_Seed"]["inputs"]["noise_seed"] == 11
    lat = g["PU_Latent"]["inputs"]
    assert (lat["width"], lat["height"], lat["length"], lat["batch_size"]) == (640, 352, 73, 1)  # pass 1 at half size
    assert g["PU_Video"]["inputs"]["fps"] == 24.0 and g["cond"]["inputs"]["frame_rate"] == 24.0  # fps in both
    save = g["PU_Save"]["inputs"]
    assert save["filename_prefix"] == "pickups/video/video" and save["format"] == "mp4" and save["format.codec"] == "h264"
    assert g["sample1"]["inputs"]["latent_image"] == ["PU_Latent", 0] and g["sample2"]["inputs"]["latent_image"] == ["up", 0]
    assert g["unet"]["inputs"]["unet_name"] == graphs.UNET and g["clip"]["inputs"]["clip_name"] == graphs.TEXT_ENCODER
    assert g["upmodel"]["inputs"]["model_name"] == graphs.LATENT_UPSCALER
    assert g["sigmas1"]["inputs"]["sigmas"] == graphs.DISTILLED_SIGMAS and g["sigmas2"]["inputs"]["sigmas"] == graphs.DFR_SIGMAS
    assert g["guider1"]["inputs"]["cfg"] == 1.0 and g["sampler"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert not any(k in g for k in ("PU_Image", "PU_FirstHalf", "PU_FirstFull"))


def test_image_to_video_graph_cover_crops_at_half_and_full_size():
    g = graphs.ltx_image_to_video("She turns", "start.png", 1920, 1088, 97, seed=3)
    assert g["PU_Image"] == {"class_type": "LoadImage", "inputs": {"image": "start.png"}}
    half, full = g["PU_FirstHalf"]["inputs"], g["PU_FirstFull"]["inputs"]
    assert (half["width"], half["height"], full["width"], full["height"]) == (960, 544, 1920, 1088)
    for n in (half, full):
        assert n["image"] == ["PU_Image", 0] and n["crop"] == "center" and n["upscale_method"] == "lanczos"
    assert g["pre_half"]["inputs"] == {"image": ["PU_FirstHalf", 0], "img_compression": 18}
    assert g["pre_full"]["inputs"] == {"image": ["PU_FirstFull", 0], "img_compression": 18}
    assert g["i2v1"]["inputs"]["latent"] == ["PU_Latent", 0] and g["i2v1"]["inputs"]["image"] == ["pre_half", 0]
    assert g["i2v2"]["inputs"]["latent"] == ["up", 0] and g["i2v2"]["inputs"]["image"] == ["pre_full", 0]
    assert g["sample1"]["inputs"]["latent_image"] == ["i2v1", 0] and g["sample2"]["inputs"]["latent_image"] == ["i2v2", 0]
    assert (g["PU_Latent"]["inputs"]["width"], g["PU_Latent"]["inputs"]["height"]) == (960, 544)
    assert g["PU_Video"]["inputs"]["fps"] == 24.0 and g["cond"]["inputs"]["frame_rate"] == 24.0


@pytest.mark.parametrize(("w", "h", "n"), [(1280, 720, 97), (1920, 1080, 97), (1280, 704, 96), (1280, 704, 105), (1280, 704, 1)])
def test_graphs_refuse_bad_sizes_and_lengths(w, h, n):
    with pytest.raises(ValueError):
        graphs.ltx_text_to_video("x", w, h, n)


def test_every_node_input_points_at_a_node():
    g = graphs.ltx_image_to_video("x", "a.png", 1024, 768, 25)
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g


def test_setup_knows_what_video_needs():
    from pickups.tasks import video as task
    assert manifest.TASK_MODELS["video"] == task.MODEL_FILES
    assert all(f in manifest.MODELS for f in task.MODEL_FILES)
    assert "video" in manifest.task_models() and "video" in manifest.TASK_ETA_4090
    used = {n["class_type"] for n in graphs.ltx_image_to_video("x", "a.png", 1280, 704, 25).values()}
    assert used - manifest.STOCK_NODES == set(manifest.TASK_NODES["video"])


# --- the task, with a fake ComfyUI --------------------------------------------------------------------------------------

def _mp4(path: Path, size: str = "256x144") -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24", "-frames:v", "25",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


class FakeOut:
    def __init__(self, src: Path, name: str = "video_00001_.mp4"):
        self.src, self.name = src, name

    def to_file(self, dst):
        Path(dst).write_bytes(self.src.read_bytes())


class FakeComfyJob:
    def __init__(self, src: Path):
        self.src = src
        self.id = "prompt-1"

    def get_outputs(self, node):
        return [FakeOut(self.src)] if node == "PU_Save" else []


class FakeClient:
    def __init__(self):
        self.uploaded: list[Path] = []
        self.assets = self

    def from_file(self, p):
        self.uploaded.append(Path(p))
        return f"uploaded/{Path(p).name}"


@pytest.fixture()
def clip(tmp_path):
    return _mp4(tmp_path / "comfy_out.mp4")


def _ctx(tmp_path, params, clip, store=None, installed=None):
    class Conn:
        def model_files(self):
            return installed if installed is not None else set(manifest.TASK_MODELS["video"])

    class Ctx:
        def __init__(self):
            self.params, self.project_id, self.vdir, self.conn = params, "prj_1", tmp_path, Conn()
            self.client, self.store, self.graphs, self.stages = FakeClient(), store, [], []

        def stage(self, label):
            self.stages.append(label)

        def progress(self, _f):
            pass

        def run(self, g):
            self.graphs.append(g)
            return FakeComfyJob(clip)

    return Ctx()


def test_text_task_runs_the_text_graph_and_hands_back_a_shot(tmp_path, clip):
    from pickups.tasks import video as task
    ctx = _ctx(tmp_path, video.request({"prompt": "A slow forward dolly along a quiet forest path", "aspect": "21:9", "resolution": "720p",
                                        "duration": 1, "seed": 5}), clip)
    out = task.run(ctx)
    (g,) = ctx.graphs
    assert "PU_Image" not in g and g["PU_Latent"]["inputs"]["length"] == 25 and g["PU_Latent"]["inputs"]["width"] == 672
    assert ctx.stages == ["Loading LTX-2.5", "Making your video"]
    shot = out["shot"]
    # Run 9: the whole first sentence, never cut after six words.
    assert Path(shot["path"]).read_bytes() == clip.read_bytes() and shot["name"] == "A slow forward dolly along a quiet forest path"
    assert shot["made"]["mode"] == "text" and shot["made"]["width"] == 1344 and shot["keep"] == []


def test_image_task_uploads_the_picture_and_keeps_it(tmp_path, clip):
    from pickups.tasks import video as task
    pic = tmp_path / "pic.png"
    from PIL import Image
    Image.new("RGB", (64, 64), (10, 20, 30)).save(pic)

    class Store:
        def ref_path(self, project_id, ref):
            assert project_id == "prj_1" and ref == {"kind": "image", "id": "img_9"}
            return pic

    ctx = _ctx(tmp_path, video.request({"mode": "image", "prompt": "She turns", "aspect": "1:1", "resolution": "720p", "duration": 3,
                                        "image": {"kind": "image", "id": "img_9"}}), clip, store=Store())
    out = task.run(ctx)
    (g,) = ctx.graphs
    assert ctx.client.uploaded == [pic] and g["PU_Image"]["inputs"]["image"] == "uploaded/pic.png"
    assert g["PU_FirstFull"]["inputs"]["width"] == 960 and g["PU_FirstHalf"]["inputs"]["height"] == 480
    assert out["shot"]["keep"] == ["start.png"] and (tmp_path / "start.png").read_bytes() == pic.read_bytes()


def test_missing_models_say_so_before_the_gpu(tmp_path, clip):
    from pickups.tasks import video as task
    ctx = _ctx(tmp_path, video.request({"prompt": "x"}), clip, installed={"something_else.safetensors"})
    with pytest.raises(RuntimeError, match="See Setup"):
        task.run(ctx)
    assert ctx.graphs == []


def test_a_gone_picture_is_a_plain_error(tmp_path, clip):
    from pickups.tasks import video as task

    class Store:
        def ref_path(self, *_a):
            raise FileNotFoundError("img_9")

    ctx = _ctx(tmp_path, video.request({"mode": "image", "prompt": "x", "image": {"kind": "image", "id": "img_9"}}), clip, store=Store())
    with pytest.raises(ValueError, match="That picture is gone"):
        task.run(ctx)


# --- the job: the video becomes a shot in the project -------------------------------------------------------------------

def test_the_job_makes_a_new_shot_in_the_project(tmp_path, clip, monkeypatch):
    import pickups.app.store as st
    from pickups.app import jobs as jobs_mod
    from pickups.app.store import Store
    from pickups.tasks import TASKS

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = Store(tmp_path / "library")
    pid = store.create_project("Forest")["id"]

    class Conn:
        comfy_url = "http://127.0.0.1:9"

        def model_files(self):
            return set(manifest.TASK_MODELS["video"])

        def client(self):
            return FakeClient()

        def system_stats(self):
            return None

    monkeypatch.setattr(jobs_mod, "run_graph", lambda client, graph, **k: FakeComfyJob(clip))
    m = jobs_mod.JobManager.__new__(jobs_mod.JobManager)
    m.store, m.conn, m.tasks, m.jobs, m.order, m._listeners, m._ctx, m._lock = store, Conn(), TASKS, {}, [], [], {}, threading.Lock()
    seen = []
    m.broadcast = lambda kind, data: seen.append((kind, data))
    job = {"id": "job_1", "shot_id": None, "project_id": pid, "task": "video", "status": "running",
           "params": video.request({"prompt": "Mist over a loch at dawn", "resolution": "720p", "duration": 1, "seed": 9})}
    m.jobs["job_1"] = job
    ctx = jobs_mod.TaskContext(m, job, TASKS["video"].stages_of(job["params"]))
    m._run_project_task(job, ctx, TASKS["video"])
    assert job["status"] == "done", job.get("error_detail")
    shot = store.shot(job["result_shot_id"])
    assert shot["project_id"] == pid and shot["name"] == "Mist over a loch at dawn" and shot["frames"] == 25
    assert shot["made"]["task"] == "video" and shot["made"]["job_id"] == "job_1" and shot["made"]["seed"] == 9
    assert any(k == "shot" and d["id"] == shot["id"] for k, d in seen)
    assert any(k == "project" and d.get("shots") == [shot["id"]] for k, d in seen)


# --- the route ----------------------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    import pickups.app.store as st
    from pickups.app.store import Store

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = Store(tmp_path / "library")
    monkeypatch.setattr(server, "ROOT", store.root)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(server, "Store", lambda: store)
    queued = []

    def submit(self, shot_id, task, params, version_id=None, project_id=None, **k):
        queued.append((shot_id, task, params, project_id))
        return {"id": "job_x", "task": task, "project_id": project_id, "params": params, "status": "queued"}

    monkeypatch.setattr(server.JobManager, "submit", submit)
    return TestClient(server.create_app("http://127.0.0.1:9")), store, queued


def test_route_queues_a_checked_video_job(client):
    c, store, queued = client
    pid = store.create_project("Forest")["id"]
    r = c.post(f"/api/projects/{pid}/videos/generate", json={"prompt": "A quiet path", "aspect": "4:3", "resolution": "720p", "duration": 2})
    assert r.status_code == 200, r.text
    (_shot, task, params, project) = queued[0]
    assert (task, project, params["width"], params["height"], params["frames"]) == ("video", pid, 1024, 768, 49)
    assert isinstance(params["seed"], int)


@pytest.mark.parametrize(("body", "said"), [
    ({"prompt": ""}, "Describe the shot"),
    ({"prompt": "x", "aspect": "9:16", "resolution": "1080p"}, "1080p is 16:9 only"),
    ({"prompt": "x", "duration": 5}, "Pick a length"),
    ({"mode": "image", "prompt": "x"}, "Choose a picture"),
    ({"mode": "image", "prompt": "x", "image": {"kind": "image", "id": "img_nope"}}, "That picture is gone"),
])
def test_route_says_what_is_wrong(client, body, said):
    c, store, queued = client
    pid = store.create_project("Forest")["id"]
    r = c.post(f"/api/projects/{pid}/videos/generate", json=body)
    assert r.status_code == 400 and said in r.json()["detail"] and not queued


def test_options_route(client):
    c, _store, _q = client
    o = c.get("/api/video/options").json()
    assert o["model_name"] == "LTX-2.5" and o["sizes"]["720p"]["21:9"] == [1344, 576]
