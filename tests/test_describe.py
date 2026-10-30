"""Describe the shot (POST /api/shots/{id}/describe): a local model that sees images describes a shot or a take for a
prompt field. A fake runner stands in for ComfyUI (its "history" answers with the text), a stub server for a local
OpenAI-compatible one."""

import json
import re
import subprocess
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from pickups import describe, director
from pickups.tasks import common

COMFY = "http://127.0.0.1:9"  # a dead local address: never a real ComfyUI
QWEN35 = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
GEMMA = "gemma4_e2b_it_int8_convrot.safetensors"
RECIPES = {
    "image_qwen_image_2_1_image_edit": {"node": "TextGenerate", "clip_name": QWEN35, "clip_type": "qwen_image", "system": "edit",
                                        "settings": {"max_length": 4096, "sampling_mode": "on", "temperature": 1.0, "top_k": 20,
                                                     "top_p": 0.95, "min_p": 0, "repetition_penalty": 1, "seed": 0,
                                                     "presence_penalty": 0, "thinking": True, "use_default_template": True}},
    "video_ltx2_5_i2v": {"node": "TextGenerateLTX2Prompt", "clip_name": GEMMA, "clip_type": "ltxv", "system": None,
                         "settings": {"max_length": 600, "sampling_mode": "on", "temperature": 0.7, "top_k": 64, "top_p": 0.95,
                                      "min_p": 0.05, "repetition_penalty": 1.15, "seed": 0, "presence_penalty": 0,
                                      "thinking": False, "use_default_template": True}},
}
SAID = "A woman in a white top rides a bicycle down a country road toward the mountains; the camera follows behind."


class _Job:
    def __init__(self, n):
        self.id = f"job_{n}"


class FakeRunner:
    """QuickRunner without ComfyUI: records each graph and the pictures it was given; ComfyUI's history answers with
    `replies` in turn."""

    def __init__(self):
        self.graphs: list[dict] = []
        self.pictures: list[list[tuple[int, int]]] = []
        self.replies = [SAID]
        self.fail: Exception | None = None
        assets = type("Assets", (), {"from_file": staticmethod(lambda p: str(p))})
        self.client = type("Client", (), {"assets": assets})()

    def run(self, graph):
        if self.fail:
            raise self.fail
        self.graphs.append(graph)
        sizes = []
        for k in sorted(k for k in graph if k.startswith("PU_Ref")):
            with Image.open(graph[k]["inputs"]["image"]) as im:
                sizes.append(im.size)
        self.pictures.append(sizes)
        return _Job(len(self.graphs) - 1)


@pytest.fixture()
def comfy(monkeypatch):
    """ComfyUI's HTTP answers: the templates' enhancers, CLIPLoader's model list and the history's text."""
    state = {"models": {describe.QWEN_TEXT, QWEN35, GEMMA}, "offline": False, "runner": None}

    def get(url, timeout=20.0):
        if state["offline"]:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        if url.endswith("/object_info/CLIPLoader"):
            return {"CLIPLoader": {"input": {"required": {"clip_name": [sorted(state["models"])]}}}}
        m = re.search(r"/history/job_(\d+)$", url)
        if m:
            r = state["runner"].replies
            return {f"job_{m.group(1)}": {"outputs": {"PU_Text": {"text": [r[min(int(m.group(1)), len(r) - 1)]]}}}}
        raise AssertionError(url)

    monkeypatch.setattr(director, "_get", get)
    monkeypatch.setattr(director, "recipe", lambda url, name: RECIPES[name])
    return state


def _clip(path, seconds=1.0, size="320x180", rate=24):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture()
def app(tmp_path, monkeypatch, comfy):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(common, "MAX_FRAMES", 97)
    settings = {"director": {"backend": "comfy"}}
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    runner = FakeRunner()
    comfy["runner"] = runner
    monkeypatch.setattr(server, "QuickRunner", lambda conn: runner)
    client = TestClient(server.create_app(COMFY))
    sid = store.add_shot(_clip(tmp_path / "ride.mov"), "Ride")["id"]
    return client, store, runner, sid, settings


def _system(graph) -> str:
    return graph["pd_sys"]["inputs"]["value"]


def test_describes_first_middle_and_last_frame(app):
    c, store, runner, sid, _ = app
    r = c.post(f"/api/shots/{sid}/describe", json={"target": "ltx", "field": "scene"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["text"] == SAID and res["cached"] is False
    assert res["frames"] == [0, 11, 23] and res["version_id"] == "orig"  # a 24-frame shot
    assert res["model"] == "Qwen3-VL 8B" and res["runs_on"] == "ComfyUI" and res["camera"] in describe.MOVES
    g = runner.graphs[0]
    # Qwen-Image 2.1's own text encoder, run as a chat model on the three frames, no thinking
    assert g["pd_llm"]["inputs"] == {"clip_name": describe.QWEN_TEXT, "type": "qwen_image", "device": "default"}
    assert g["pd_gen"]["class_type"] == "TextGenerate" and g["pd_gen"]["inputs"]["thinking"] is False
    assert len(runner.pictures[0]) == 3 and all(max(s) <= describe.FRAME_LONG for s in runner.pictures[0])
    assert g["pd_img"]["class_type"] == "BatchImagesNode"
    # an instruction to describe, not to rewrite: only what's visible, in English, one or two sentences for LTX
    system = _system(g)
    assert "never rewrite" in system and "Only what is clearly visible" in system and "English" in system
    assert "ONE or TWO sentences" in system and "camera" in system
    assert "first, the middle and the last" in g["pd_gen"]["inputs"]["prompt"]
    assert not list((store.shot_dir(sid) / "describe").glob("_work*"))  # frames are cleaned up


def test_no_body_describes_for_the_video_model(app):
    c, store, runner, sid, _ = app
    res = c.post(f"/api/shots/{sid}/describe").json()
    assert res["target"] == "ltx" and res["field"] == "scene" and res["text"] == SAID


def test_cached_per_shot_take_and_field(app, tmp_path):
    c, store, runner, sid, _ = app
    first = c.post(f"/api/shots/{sid}/describe", json={"field": "scene"}).json()
    again = c.post(f"/api/shots/{sid}/describe", json={"field": "scene", "version_id": "orig"}).json()
    assert again["cached"] is True and len(runner.graphs) == 1  # no model call
    assert {k: v for k, v in again.items() if k != "cached"} == {k: v for k, v in first.items() if k != "cached"}
    # another field is its own answer
    motion = c.post(f"/api/shots/{sid}/describe", json={"field": "motion"}).json()
    assert motion["cached"] is False and motion["field"] == "motion" and len(runner.graphs) == 2
    assert "TWO or THREE sentences" in _system(runner.graphs[1])
    assert c.post(f"/api/shots/{sid}/describe", json={"field": "scene"}).json()["cached"] is True
    # "Try again": a fresh answer with another seed, which then is the cached one
    runner.replies = [SAID, SAID, "A cyclist rides along a mountain road; the camera follows."]
    fresh = c.post(f"/api/shots/{sid}/describe", json={"field": "scene", "refresh": True, "seed": 3}).json()
    assert fresh["cached"] is False and fresh["text"].startswith("A cyclist")
    assert runner.graphs[-1]["pd_gen"]["inputs"]["sampling_mode.seed"] == 3
    assert c.post(f"/api/shots/{sid}/describe", json={"field": "scene"}).json()["text"].startswith("A cyclist")
    # a take has its own description, looked at in its own clip
    vid, vdir = store.new_version_dir(sid)
    _clip(vdir / "master.mov", 0.5)
    _clip(vdir / "video.mp4", 0.5)
    store.save_version(sid, vid, {"label": "Change weather", "task": "environment", "input_version": "orig", "downloads": []})
    take = c.post(f"/api/shots/{sid}/describe", json={"version_id": vid}).json()
    assert take["cached"] is False and take["version_id"] == vid and take["frames"] == [0, 5, 11]
    assert json.loads((store.shot_dir(sid) / "describe" / f"{vid}.json").read_text())["ltx:scene"]["result"]["text"]
    # a changed clip is looked at again
    _clip(store.shot_dir(sid) / "proxy.mp4", 0.5)
    assert c.post(f"/api/shots/{sid}/describe", json={"field": "scene"}).json()["cached"] is False


def test_frames_are_the_part_the_video_model_renders():
    # first, middle and last of the first MAX_FRAMES frames (the rest of a long shot isn't rendered)
    assert describe.pick_frames(98, 97) == [0, 48, 96]
    assert describe.pick_frames(400, 97) == [0, 48, 96]
    assert describe.pick_frames(400, 169) == [0, 84, 168]  # Render previz on a big card
    assert describe.pick_frames(24, 97) == [0, 11, 23]
    assert describe.pick_frames(2, 97) == [0, 1] and describe.pick_frames(1, 97) == [0] and describe.pick_frames(0, 97) == [0]


def test_grabbed_frames_are_the_right_ones(tmp_path):
    clip = tmp_path / "count.mp4"
    # each frame's brightness is its number x 10, so the PNGs say which frames they are
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=960x540:r=24:d=1",
                    "-vf", "geq=lum='N*10':cb=128:cr=128", "-pix_fmt", "yuv420p", "-crf", "1", str(clip)], check=True)
    assert describe._count(clip) == 24
    pngs = describe.grab(clip, [0, 11, 23], tmp_path / "f")
    assert len(pngs) == 3
    for png, n in zip(pngs, (0, 11, 23)):
        with Image.open(png) as im:
            assert im.size == (768, 432)  # long side FRAME_LONG, even sizes
            assert abs(float(np.asarray(im.convert("L")).mean()) - max(0.0, (n * 10 - 16) * 255 / 219)) < 12


def test_camera_is_measured_not_guessed(tmp_path):
    rng = np.random.default_rng(1)
    ys, xs = np.mgrid[0:360, 0:640]
    scene = Image.fromarray((127 + 120 * np.sin(xs / 23 + ys / 61) * np.cos(ys / 17 - xs / 97)).astype(np.uint8))  # a busy set

    def save(im, name):
        p = tmp_path / name
        im.convert("RGB").save(p)
        return p

    first = save(scene, "a.png")
    still = save(scene, "b.png")
    w, h = scene.size
    zoomed = save(scene.crop((w * 0.1, h * 0.1, w * 0.9, h * 0.9)).resize((w, h)), "c.png")  # 1.25x: a push in
    other = save(Image.fromarray((rng.random((360, 640)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), "d.png")
    assert describe.camera(first, still) == "static"
    assert describe.camera(first, zoomed) == "push"
    assert describe.camera(zoomed, first) == "pull"
    assert describe.camera(first, other) == "moving"


def test_the_measured_move_goes_with_the_request(app, monkeypatch):
    c, store, runner, sid, _ = app
    monkeypatch.setattr(describe, "camera", lambda a, b: "push")
    res = c.post(f"/api/shots/{sid}/describe", json={"field": "motion"}).json()
    assert res["camera"] == "push"
    assert runner.graphs[0]["pd_gen"]["inputs"]["prompt"].endswith(describe.MOVES["push"])


def test_a_layout_description_is_asked_again(app):
    """Render previz: an answer naming the layout's stand-ins is asked once more, firmly, and the cleaner one kept."""
    c, store, runner, sid, _ = app
    runner.replies = ["A yellow figure sits in a grey block car. The camera pushes in.",
                      "A driver sits behind the wheel of a car. The camera pushes in toward him."]
    res = c.post(f"/api/shots/{sid}/describe", json={"field": "motion"}).json()
    assert res["text"] == "A driver sits behind the wheel of a car. The camera pushes in toward him."
    assert len(runner.graphs) == 2 and "none of these words" in runner.graphs[1]["pd_gen"]["inputs"]["prompt"]
    assert "never write mannequin" in _system(runner.graphs[0])


def test_an_answer_in_another_language_is_asked_again_then_refused(app):
    c, store, runner, sid, _ = app
    runner.replies = ["一位女士骑自行车。", "A woman rides a bicycle along a road; the camera follows."]
    assert c.post(f"/api/shots/{sid}/describe").json()["text"].startswith("A woman")
    assert runner.graphs[1]["pd_gen"]["inputs"]["prompt"].startswith("Answer in English.")
    runner.replies = ["一位女士骑自行车。"]
    r = c.post(f"/api/shots/{sid}/describe", json={"refresh": True})
    assert r.status_code == 502 and r.json()["detail"] == describe.NOT_ENGLISH


def test_falls_back_to_the_next_model_on_the_machine(app, comfy):
    c, store, runner, sid, _ = app
    comfy["models"] = {QWEN35, GEMMA}  # no Qwen-Image 2.1 text encoder
    res = c.post(f"/api/shots/{sid}/describe").json()
    assert res["model"] == "Qwen 3.5 9B" and runner.graphs[0]["pd_llm"]["inputs"]["clip_name"] == QWEN35
    assert runner.graphs[0]["pd_gen"]["inputs"]["thinking"] is False and "never rewrite" in _system(runner.graphs[0])
    comfy["models"] = {GEMMA}
    res = c.post(f"/api/shots/{sid}/describe", json={"refresh": True}).json()
    # LTX's own director sees one picture: the middle frame
    assert res["model"] == "LTX-2.5 director" and runner.graphs[1]["pd_gen"]["class_type"] == "TextGenerateLTX2Prompt"
    assert len(runner.pictures[1]) == 1 and runner.graphs[1]["pd_img"]["class_type"] == "LTXVPreprocess"


def test_errors_in_plain_words(app, comfy):
    c, store, runner, sid, settings = app
    comfy["models"] = {"umt5_xxl_fp8_e4m3fn_scaled.safetensors"}
    r = c.post(f"/api/shots/{sid}/describe")
    assert r.status_code == 502
    assert r.json()["detail"] == describe.MISSING.format(file=describe.QWEN_TEXT)
    assert "isn't on your GPU machine" in r.json()["detail"] and "Setup" in r.json()["detail"]
    comfy["offline"] = True
    r = c.post(f"/api/shots/{sid}/describe")
    assert r.status_code == 502 and r.json()["detail"] == describe.OFFLINE.format(url=COMFY)
    assert r.json()["detail"].startswith("ComfyUI isn't answering")
    comfy["offline"] = False
    comfy["models"] = {describe.QWEN_TEXT}
    runner.fail = ConnectionRefusedError(61, "Connection refused")  # gone between the check and the run
    assert c.post(f"/api/shots/{sid}/describe").json()["detail"].startswith("ComfyUI isn't answering")
    runner.fail = None
    runner.replies = [""]
    r = c.post(f"/api/shots/{sid}/describe")
    assert r.status_code == 502 and r.json()["detail"] == describe.NO_TEXT
    assert not (store.shot_dir(sid) / "describe" / "orig.json").exists()  # nothing cached
    settings["director"] = {"backend": "off"}
    r = c.post(f"/api/shots/{sid}/describe")
    assert r.status_code == 409 and "turned off" in r.json()["detail"]


def test_unknown_shot_take_target_or_field(app):
    c, store, runner, sid, _ = app
    assert c.post("/api/shots/shot_missing/describe").status_code == 404
    for bad in ("v_missing", "../..", "v_x/../.."):
        assert c.post(f"/api/shots/{sid}/describe", json={"version_id": bad}).status_code == 404
    assert c.post(f"/api/shots/{sid}/describe", json={"target": "dalle"}).status_code == 400
    assert c.post(f"/api/shots/{sid}/describe", json={"field": "poem"}).status_code == 400
    assert runner.graphs == []


def test_image_targets_describe_one_still(app):
    c, store, runner, sid, _ = app
    res = c.post(f"/api/shots/{sid}/describe", json={"target": "qwen-edit", "frame": 7}).json()
    assert res["field"] == "look" and res["frames"] == [7] and res["camera"] is None
    assert len(runner.pictures[0]) == 1 and "still photograph" in _system(runner.graphs[0])


# --- a local OpenAI-compatible server (LM Studio, Ollama) ---------------------------------------------------------------

def _stub(replies, status=200, error=""):
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if status != 200:
                out = json.dumps({"error": error}).encode()
                self.send_response(status)
            else:
                out = json.dumps({"choices": [{"message": {"content": replies[min(len(calls) - 1, len(replies) - 1)]}}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def test_a_local_vision_model_describes(app):
    c, store, runner, sid, settings = app
    srv, calls = _stub([SAID])
    settings["director"] = {"backend": "openai", "describe_backend": "openai", "url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "text-8b",
                            "vision_model": "vision-12b"}
    res = c.post(f"/api/shots/{sid}/describe").json()
    srv.shutdown()
    assert res["text"] == SAID and res["runs_on"] == "vision-12b" and runner.graphs == []
    body = calls[0]
    assert body["model"] == "vision-12b"
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url", "image_url"]
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts[1:])
    assert "never rewrite" in body["messages"][0]["content"] and "English" in body["messages"][0]["content"]


def test_without_a_vision_model_the_gpu_machine_describes(app):
    c, store, runner, sid, settings = app
    settings["director"] = {"backend": "openai", "describe_backend": "openai", "url": "http://127.0.0.1:9/v1", "model": "text-8b", "vision_model": ""}
    assert c.post(f"/api/shots/{sid}/describe").json()["runs_on"] == "ComfyUI"


def test_local_server_errors_in_plain_words(app):
    c, store, runner, sid, settings = app
    settings["director"] = {"backend": "openai", "describe_backend": "openai", "url": "http://127.0.0.1:9/v1", "model": "t", "vision_model": "v"}
    r = c.post(f"/api/shots/{sid}/describe")
    assert r.status_code == 502 and r.json()["detail"] == describe.LOCAL_OFFLINE.format(url="http://127.0.0.1:9/v1")
    srv, _ = _stub([], status=400, error="Model does not support images")
    settings["director"]["url"] = f"http://127.0.0.1:{srv.server_port}"
    r = c.post(f"/api/shots/{sid}/describe")
    srv.shutdown()
    assert r.json()["detail"] == "v can't look at pictures. Pick a vision model for the prompt director in Setup."
