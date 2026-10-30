"""Image graph builders: every graph must validate against a saved ComfyUI /object_info snapshot (0.41.1)."""

import inspect
import json
from pathlib import Path

import pytest

from pickups import graphs, images

OBJECT_INFO = json.loads((Path(__file__).parent / "data" / "object_info_images.json").read_text())
# Loader inputs whose choices are the server's own files: checked against ImageModel.files instead.
FILE_INPUTS = {("LoadImage", "image"), ("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name"),
               ("LoraLoaderModelOnly", "lora_name"), ("CheckpointLoaderSimple", "ckpt_name")}
CASES = sorted(images.BUILDERS)


def _graph(model_id: str, mode: str, nrefs: int = 2, **kw) -> dict:
    m = images.MODELS[model_id]
    refs = [f"ref{i + 1}.png" for i in range(min(nrefs, m.max_refs))] if mode != "generate" else []
    w, h = images.size_for("16:9", "1K", m)
    g = images.build(model_id, "A photograph of a red bicycle", w, h, 42, refs=refs, **kw)
    assert images.mode_for(m, refs) == mode
    return g


def _spec(class_type: str, name: str):
    inp = OBJECT_INFO[class_type]["input"]
    return (inp.get("required") or {}).get(name) or (inp.get("optional") or {}).get(name)


def _options(spec) -> list | None:
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO":
        return spec[1].get("options")
    return None


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


@pytest.mark.parametrize("model_id,mode", CASES)
def test_graph_validates_against_object_info(model_id, mode):
    g = _graph(model_id, mode)
    for nid, node in g.items():
        ct = node["class_type"]
        assert ct in OBJECT_INFO, f"{nid}: unknown node type {ct}"
        info = OBJECT_INFO[ct]
        given = node["inputs"]
        for name, value in given.items():
            base = name.split(".")[0]
            spec = _spec(ct, base)
            assert spec is not None, f"{nid} ({ct}) has no input {base!r}"
            if _is_link(value):
                src = g.get(value[0])
                assert src is not None, f"{nid}.{name} links to missing node {value[0]}"
                assert value[1] < len(OBJECT_INFO[src["class_type"]]["output"]), f"{nid}.{name}: bad output slot"
                continue
            if spec[0] == "COMFY_DYNAMICCOMBO_V3" and name == base:
                keys = {o["key"]: o for o in spec[1]["options"]}
                assert value in keys, f"{nid}.{name}={value!r} not in {list(keys)}"
                for sub in keys[value]["inputs"].get("required", {}):
                    assert f"{name}.{sub}" in given, f"{nid}: {name}={value!r} needs {name}.{sub}"
            opts = _options(spec)
            if opts is not None and (ct, base) not in FILE_INPUTS:
                assert value in opts, f"{nid}.{name}={value!r} not a valid choice"
        for name, spec in (info["input"].get("required") or {}).items():
            if spec[0] == "COMFY_AUTOGROW_V3":
                continue  # e.g. images.image_1 ...; min 0 or checked via the refs below
            assert name in given, f"{nid} ({ct}) is missing required input {name!r}"


@pytest.mark.parametrize("model_id,mode", CASES)
def test_conventions_and_model_files(model_id, mode):
    m = images.MODELS[model_id]
    g = _graph(model_id, mode)
    assert g["PU_Out"]["class_type"] == "SaveImage"
    assert g["PU_Out"]["inputs"]["filename_prefix"] == f"pickups/image/{model_id}"
    assert "PU_Prompt" in g and "A photograph of a red bicycle" in json.dumps(g["PU_Prompt"]["inputs"])
    assert "PU_Seed" in g
    nrefs = 0 if mode == "generate" else min(2, m.max_refs)
    assert sorted(k for k in g if k.startswith("PU_Ref")) == [f"PU_Ref{i + 1}" for i in range(nrefs)]
    for i in range(nrefs):
        assert g[f"PU_Ref{i + 1}"] == {"class_type": "LoadImage", "inputs": {"image": f"ref{i + 1}.png"}}
    loaded = {v for n in g.values() for (ct, f) in FILE_INPUTS if n["class_type"] == ct and ct != "LoadImage"
              for k, v in n["inputs"].items() if k == f}
    assert loaded and loaded <= set(m.files), f"{model_id} {mode} loads {loaded - set(m.files)} not listed in files"


def test_every_listed_file_is_loaded_by_some_mode():
    for m in images.MODELS.values():
        loaded = set()
        for mode in m.modes:
            g = _graph(m.id, mode)
            loaded |= {v for n in g.values() for k, v in n["inputs"].items() if k.endswith("_name") and isinstance(v, str)}
        assert set(m.files) <= loaded, f"{m.id}: files never loaded: {set(m.files) - loaded}"


def test_registry_shape():
    assert images.DEFAULT == "flux3-klein" and images.MODELS[images.DEFAULT].sponsor
    for m in images.MODELS.values():
        assert m.modes and m.modes[0] == "generate"
        assert set(m.modes) <= {"generate", "edit", "style"}
        assert all((m.id, mode) in images.BUILDERS for mode in m.modes)
        assert (m.max_refs > 0) == (len(m.modes) > 1)
        assert m.speed and m.notes and m.licence and m.files


@pytest.mark.parametrize("model_id", sorted(images.MODELS))
def test_size_for_multiples_of_64(model_id):
    m = images.MODELS[model_id]
    for aspect, (aw, ah) in images.ASPECTS.items():
        for res, side in images.RESOLUTIONS.items():
            w, h = images.size_for(aspect, res, m)
            assert w % 64 == 0 and h % 64 == 0, (aspect, res, w, h)
            assert max(w, h) <= m.max_side
            assert abs((w / h) / (aw / ah) - 1) < 0.1, (aspect, res, w, h)
            if max(w, h) < m.max_side:  # not capped: about 1 MP or 4 MP
                assert 0.85 < w * h / side**2 < 1.15, (aspect, res, w, h)


def test_size_examples():
    flux, hidream = images.MODELS["flux3-klein"], images.MODELS["hidream-o1"]
    assert images.size_for("16:9", "1K", flux) == (1344, 768)
    assert images.size_for("16:9", "2K", flux) == (2048, 1152)  # capped by max_side
    assert images.size_for("1:1", "2K", flux) == (2048, 2048)
    assert images.size_for("16:9", "2K", hidream) == (2560, 1408)


def test_batching():
    g = _graph("flux3-klein", "generate", count=4)
    assert g["flat"]["inputs"]["batch_size"] == 4
    g = _graph("krea2-turbo", "style", count=2)
    assert g["klat"]["inputs"]["batch_size"] == 2
    for model_id, mode in images.NO_BATCH:
        assert not images.can_batch(model_id, ["ref1.png"])
        with pytest.raises(ValueError):
            _graph(model_id, mode, count=2)


def test_refs_limits_and_modes():
    with pytest.raises(ValueError):
        images.build("z-image-turbo", "p", 1024, 1024, 1, refs=["a.png"])
    with pytest.raises(ValueError):
        images.build("krea2-turbo", "p", 1024, 1024, 1, refs=["a.png"] * 4)
    g = images.build("qwen-image-2.1", "p", 1024, 1024, 1, refs=["a.png", "b.png", "c.png"])
    assert [g["PU_Prompt"]["inputs"][f"images.image_{i}"] for i in (1, 2, 3)] == [["PU_Ref1", 0], ["PU_Ref2", 0], ["PU_Ref3", 0]]
    assert g["PU_Prompt"]["inputs"]["resolution"] == 1024


def test_qwen_edit_wraps_graphs_builder():
    g = images.qwen21_edit("p", 1344, 768, 5, ["a.png"])
    raw = graphs.qwen21_edit("a.png", "p", seed=5, resolution=1024)
    assert {k: v["class_type"] for k, v in g.items()} == {
        {"PU_Frame": "PU_Ref1", "PU_Plate": "PU_Out"}.get(k, k): v["class_type"] for k, v in raw.items()}


def test_still_matches_template_and_keeps_signature():
    assert list(inspect.signature(graphs.still).parameters) == ["prompt", "width", "height", "seed", "steps", "prefix"]
    s = graphs.still("p")["PU_Seed"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]) == (15, 1.0, "euler", "simple")


def test_localize_finds_subfolder_copies():
    g = _graph("krea2-turbo", "style")
    installed = {images.KREA2_TURBO, "styles\\krea2_style_reference.safetensors"}
    out = images.localize(g, installed)
    assert out["klora"]["inputs"]["lora_name"] == "styles\\krea2_style_reference.safetensors"
    assert out["kunet"]["inputs"]["unet_name"] == images.KREA2_TURBO
    assert g["klora"]["inputs"]["lora_name"] == images.KREA2_STYLE_LORA  # input graph untouched
    have = installed | {images.KREA2_TEXT, images.KREA2_VAE}
    assert images.missing("krea2-turbo", have) == []
    assert images.missing("krea2-turbo", have - {images.KREA2_VAE}) == [images.KREA2_VAE]


def test_model_files_resolve_by_alias_and_subfolder():
    from pickups import images
    official = {"flux3-image-klein-bf16.safetensors", "qwen3vl_4b_int8_convrot.safetensors", "flux2-vae.safetensors"}
    assert images.missing("flux3-klein", official) == []
    sub = {"loras/me/krea2_style_reference.safetensors", "krea2_turbo_int8_convrot.safetensors", "qwen3vl_4b_fp8_scaled.safetensors",
           "qwen_image_vae.safetensors"}
    assert images.missing("krea2-turbo", sub) == []
    assert images.missing("hidream-o1", set()) == ["hidream_o1_image_bf16.safetensors"]
