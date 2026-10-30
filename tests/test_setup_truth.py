"""The setup check tells the truth: an unread node list is never Ready, an unanswered loader makes its files unknown
(not missing), a missing core node says Update ComfyUI and a missing pack node names the pack, and the manifest has the
files Replace a person, Change the weather and Previz to final load."""

from types import SimpleNamespace

import pytest

from pickups.app import doctor, manifest

STATS = {"system": {"comfyui_version": "0.41.1", "ram_total": 64e9},
         "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda", "vram_total": 25.8e9, "vram_free": 20e9}]}
EVERY_FILE = set(manifest.MODELS)
EVERY_NODE = set().union(*manifest.TASK_NODES.values()) | {n for p in manifest.NODE_PACKS.values() for n in p["nodes"]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PICKUPS_HOME", str(tmp_path))
    monkeypatch.setattr("pickups.app.comfy.load_settings", lambda: {})


def conn(files=EVERY_FILE, unknown=frozenset(), nodes=EVERY_NODE):
    return SimpleNamespace(comfy_url="http://gpu:8188", system_stats=lambda: STATS, model_scan=lambda: (set(files), set(unknown)),
                           model_files=lambda: set(files), node_types=lambda: set(nodes))


def test_everything_in_place_is_ready():
    tiles = doctor.status(conn())["tiles"]
    assert all(t["ready"] for task, t in tiles.items() if task != "reshoot")


def test_an_unread_node_list_is_never_ready():
    tiles = doctor.status(conn(nodes=set()))["tiles"]
    assert not any(t["ready"] for t in tiles.values())
    assert {t["reason"] for t in tiles.values()} == {"nodes_unknown"} and not tiles["recast"]["nodes_checked"]


def test_a_loader_that_did_not_answer_makes_files_unknown_not_missing():
    files = {f for f in EVERY_FILE if manifest.MODELS[f][0] != "checkpoints"}
    t = doctor.status(conn(files=files, unknown={"checkpoints"}))["tiles"]["remove"]
    assert not t["ready"] and t["reason"] == "models_unknown"
    assert t["missing"] == [] and {u["file"] for u in t["unknown"]} == {"sam3.1_multiplex_fp16.safetensors"}


def test_missing_nodes_split_into_core_and_packs():
    t = doctor.status(conn(nodes=EVERY_NODE - {"SDPoseKeypointExtractor", "LTXVLaplacianPyramidBlend"}))["tiles"]["recast"]
    assert not t["ready"] and t["reason"] == "nodes"
    assert t["missing_core_nodes"] == ["SDPoseKeypointExtractor"]
    [pack] = t["missing_nodes"]
    assert pack["pack"] == "ComfyUI-LTXVideo" and pack["nodes"] == ["LTXVLaplacianPyramidBlend"]
    assert pack["install"] == "git clone https://github.com/Lightricks/ComfyUI-LTXVideo custom_nodes/ComfyUI-LTXVideo"


def test_the_files_three_features_load_are_now_checked():
    tiles = manifest.task_models()
    assert "sdpose_wholebody_fp16.safetensors" in tiles["recast"]
    assert "sam3.1_multiplex_fp16.safetensors" in tiles["environment"]
    assert {"depth_anything_3_mono_large.safetensors", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"} <= set(tiles["previz"])
    gone = doctor.status(conn(files=EVERY_FILE - {"sdpose_wholebody_fp16.safetensors"}))["tiles"]["recast"]
    assert not gone["ready"] and [m["file"] for m in gone["missing"]] == ["sdpose_wholebody_fp16.safetensors"]


def test_an_extra_manifest_in_a_scratch_home_adds_a_requirement(tmp_path):
    (tmp_path / "manifest_extra.json").write_text(
        '{"models": {"demo_missing.safetensors": ["loras", "demo/repo", "demo_missing.safetensors", 2000000000, true]},'
        ' "tasks": {"finish": ["demo_missing.safetensors"]}, "nodes": {"finish": ["DemoMissingNode"]}}')
    t = doctor.status(conn())["tiles"]["finish"]
    assert not t["ready"] and [m["file"] for m in t["missing"]] == ["demo_missing.safetensors"] and t["missing"][0]["gated"]
    assert t["missing_core_nodes"] == ["DemoMissingNode"]
