import json
from types import SimpleNamespace

import pytest

from pickups.app.jobs import JobManager
from pickups.app.store import Store
from pickups.app.workflow_titles import saved_title, step_title, workflow_title


def n(class_type: str, **inputs) -> dict:
    return {"class_type": class_type, "inputs": inputs}


def g(*nodes: dict) -> dict:
    return {str(i): node for i, node in enumerate(nodes)}


SAM = n("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
LTX = n("UNETLoader", unet_name="ltx-2.5-22b-distilled-transformer.safetensors")


def ic_lora(name: str) -> dict:
    return n("LTXICLoRALoaderModelOnly", lora_name=name, model=["unet", 0])


def test_pick_by_label_and_by_clicks():
    by_label = g(n("LoadImage", image="pick.png"), SAM, n("CLIPTextEncode", text="woman"), n("SAM3_Detect", threshold=0.4))
    assert workflow_title(by_label, "pick", frame=82) == "Pick · SAM 3.1 · woman"
    clicks = g(n("LoadImage", image="pick.png"), SAM, n("SAM3_Detect", positive_coords=json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}])))
    assert workflow_title(clicks, "pick", frame=12) == "Pick · SAM 3.1 · frame 12"
    assert workflow_title(clicks, "pick") == "Pick · SAM 3.1 · 2 clicks"
    long = g(n("LoadImage", image="pick.png"), SAM, n("CLIPTextEncode", text="the man in the long grey coat by the door"), n("SAM3_Detect"))
    assert workflow_title(long, "pick").endswith("…")


def test_find_on_video_and_tracks():
    shadow = g(n("LoadVideo", file="proxy.mp4"), SAM, n("CLIPTextEncode", text="person's shadow"), n("SAM3_Detect"))
    assert workflow_title(shadow, "shadow1") == "Find · SAM 3.1 · person's shadow"
    fwd = g(n("LoadVideo", file="fwd (2).mp4"), SAM, n("LoadImage", image="pick_mask_0.png"), n("SAM3_VideoTrack", max_objects=1))
    bwd = g(n("LoadVideo", file="bwd (1).mp4"), SAM, n("LoadImage", image="pick_mask_0.png"), n("SAM3_VideoTrack", max_objects=2))
    assert workflow_title(fwd, "track", frame=82) == "Track · SAM 3.1 · forward from frame 82"
    assert workflow_title(bwd, "track", frame=82) == "Track · SAM 3.1 · backward from frame 82, 2 objects"
    assert workflow_title(fwd, "track") == "Track · SAM 3.1 · forward"


def test_frame_edits_name_the_model_and_references():
    q21 = g(n("LoadImage", image="frame.png"), n("UNETLoader", unet_name="qwen_image_2.1_int8.safetensors"),
            n("TextEncodeQwenImage21", prompt="x"), n("LoadImage", image="ref2.png"), n("LoadImage", image="ref3.png"))
    assert workflow_title(q21, "look") == "Look frame · Qwen-Image 2.1 · 2 references"
    edit2511 = g(n("LoadImage", image="frame0.png"), n("UNETLoader", unet_name="qwen_image_edit_2511_fp8mixed.safetensors"),
                 n("TextEncodeQwenImageEditPlus", prompt="snow"))
    assert workflow_title(edit2511, "environment_key") == "First frame · Qwen-Image-Edit 2511"
    assert workflow_title(edit2511, "plate") == "Plate · Qwen-Image-Edit 2511"


def test_video_models():
    render = g(LTX, ic_lora("ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors"), n("LoadVideo", file="work.mp4"))
    assert workflow_title(render, "previz") == "Render · LTX-2.5 Layout-To-Render"
    weather = g(LTX, ic_lora("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"), n("LoadDA3Model"), n("DA3Inference"))
    assert workflow_title(weather, "environment") == "Weather · LTX-2.5 Union Control · depth"
    edges = g(LTX, ic_lora("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"), n("Canny"))
    assert workflow_title(edges, "environment") == "Weather · LTX-2.5 Union Control · edges"
    fill = g(LTX, ic_lora("ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"), SAM, n("SAM3_Detect"))
    assert workflow_title(fill, "remove") == "Fill · LTX-2.5 In/Outpaint"
    hdr = g(LTX, ic_lora("ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"), n("LTXVSDRToHDRWorkingSpace"))
    assert workflow_title(hdr, "finish_hdr") == "HDR · LTX SDR-to-HDR"
    # SCAIL-2 tracks people with SAM too: it is still a recast.
    scail = g(n("UNETLoader", unet_name="wan2.1_14B_SCAIL_2.safetensors"), SAM, n("SAM3_VideoTrack"), n("WanSCAILToVideo"))
    assert workflow_title(scail, "recast") == "Recast · SCAIL-2"
    seedvr2 = g(n("UNETLoader", unet_name="seedvr2_3b.safetensors"), n("SeedVR2Conditioning"), n("KSampler"))
    assert workflow_title(seedvr2, "finish4k") == "4K · SeedVR2"


def test_images_and_fallbacks():
    flux = g(n("UNETLoader", unet_name="flux3_image_klein_sd_bf16.safetensors"), n("EmptyFlux2LatentImage"))
    assert workflow_title(flux, "image") == "Image · FLUX 3 klein"
    assert workflow_title(flux, "cast") == "Reference · FLUX 3 klein"
    assert workflow_title(g(n("SomethingNew")), "look_key") == "Key frame"
    assert workflow_title({"bad": "graph", "1": None}, "custom_step2") == "Custom step"
    assert step_title("") == "Workflow"


def test_saved_title_reads_the_file(tmp_path):
    f = tmp_path / "01_pick.json"
    f.write_text(json.dumps(g(n("LoadImage", image="p.png"), SAM, n("CLIPTextEncode", text="sign"), n("SAM3_Detect"))))
    assert saved_title(f, "pick") == "Pick · SAM 3.1 · sign"
    (tmp_path / "broken.json").write_text("{not json")
    assert saved_title(tmp_path / "broken.json", "track") == "Track"
    assert saved_title(tmp_path / "missing.json", "plate") == "Plate"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    s = Store(tmp_path / "library")
    d = s.shot_dir("shot_test")
    (d / "versions").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"id": "shot_test", "name": "Test", "source": "source.mov", "width": 320, "height": 180,
                                             "fps": 24.0, "frames": 24, "duration_s": 1.0, "created_at": 0}))
    return s


def test_titles_saved_with_the_take_and_derived_for_older_ones(store):
    vid, vdir = store.new_version_dir("shot_test")
    pick = g(n("LoadImage", image="p.png"), SAM, n("SAM3_Detect", positive_coords='[{"x": 5, "y": 5}]'))
    track = g(n("LoadVideo", file="fwd.mp4"), SAM, n("SAM3_VideoTrack", max_objects=1))
    ctx = SimpleNamespace(params={"frame": 12}, workflows=[{"step": "pick", "graph": pick}, {"step": "track", "graph": track}])
    saved = JobManager._save_workflows(ctx, vdir)
    assert [w["title"] for w in saved] == ["Pick · SAM 3.1 · frame 12", "Track · SAM 3.1 · forward from frame 12"]
    # An older take: the same files, no titles in its meta.
    older = [{k: v for k, v in w.items() if k != "title"} for w in saved]
    v = store.save_version("shot_test", vid, {"label": "Remove", "task": "remove", "params": {"frame": 12}, "workflows": older})
    assert [w["title"] for w in v["workflows"]] == ["Pick · SAM 3.1 · frame 12", "Track · SAM 3.1 · forward from frame 12"]
