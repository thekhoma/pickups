"""Relight any scene: the tested graph and the looks."""

import json
import os
from pathlib import Path

import pytest

from pickups import graphs, images
from pickups.app import manifest
from pickups.tasks import scene_relight as sr


def test_every_look_has_its_ball_and_the_lora_words():
    for key, (label, ball, light) in sr.LOOKS.items():
        assert (sr.BALLS / f"ball_{ball}.png").exists(), key
        assert "sun" in light and label
    assert sr.look_of({"look": "custom", "custom": "warm evening light.", "direction": "left_low"}) == (
        "Warm evening light", "left_low", "warm evening light from the left, low sun")
    with pytest.raises(RuntimeError):
        sr.look_of({"look": "custom", "custom": "  "})


def test_the_graph_is_the_tested_one():
    g = graphs.scene_relight("in.mp4", "relight", 121)
    assert g["vlat"]["inputs"]["width"] == 1280 and g["vlat"]["inputs"]["height"] == 704 and g["vlat"]["inputs"]["length"] == 121
    assert g["sampler"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp" and g["cond"]["inputs"]["frame_rate"] == 24.0
    assert g["lora"]["inputs"]["lora_name"] == graphs.RELIGHT_LORA
    # every file it loads is in Setup's list for the feature, and resolves when ComfyUI keeps it in a subfolder
    fields = ("unet_name", "clip_name1", "clip_name2", "vae_name", "lora_name")
    loaded = {v for n in g.values() for k, v in n["inputs"].items() if k in fields}
    assert loaded <= set(manifest.TASK_MODELS["relight"])
    local = images.localize(g, {"rich_test\\\\" + graphs.RELIGHT_LORA} | set(manifest.TASK_MODELS["relight"]) - {graphs.RELIGHT_LORA})
    assert local["lora"]["inputs"]["lora_name"].endswith(graphs.RELIGHT_LORA) and "rich_test" in local["lora"]["inputs"]["lora_name"]


def test_relight_is_a_setup_feature():
    import pickups.tasks as tasks
    assert "relight" in tasks.TASKS and "relight" in manifest.task_models()
    assert all(f in manifest.MODELS for f in manifest.TASK_MODELS["relight"])


def test_the_sigmas_match_the_reference_run():
    # the reference run's API graph lives outside the repo: set PICKUPS_TEST_RELIGHT_DIR to the folder holding harbour_B_api.json
    folder = os.environ.get("PICKUPS_TEST_RELIGHT_DIR", "")
    ref_file = Path(folder) / "harbour_B_api.json" if folder else None
    if not ref_file or not ref_file.exists():
        pytest.skip("PICKUPS_TEST_RELIGHT_DIR unset or has no harbour_B_api.json")
    ref = json.loads(ref_file.read_text())
    assert ref["sigmas"]["inputs"]["sigmas"] == graphs.scene_relight("x", "y", 97)["sigmas"]["inputs"]["sigmas"]


def test_a_master_keeps_its_last_frame_when_the_audio_is_a_hair_short(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    from pickups.tasks import common

    (tmp_path / "f").mkdir()
    (tmp_path / "v").mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x32:rate=30000/1001", "-frames:v", "121",
                    str(tmp_path / "f" / "r_%05d.png")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=4.032", "-f", "lavfi", "-i", "color=size=64x32:rate=30000/1001",
                    "-frames:v", "121", "-shortest", str(tmp_path / "a.mp4")], check=True)
    common.finish_outputs(tmp_path / "v", tmp_path / "f" / "r_%05d.png", 30000 / 1001, audio_from=tmp_path / "a.mp4")
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", str(tmp_path / "v" / "master.mov")], capture_output=True, text=True, check=True).stdout.strip()
    assert n == "121"
