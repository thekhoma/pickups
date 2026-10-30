"""Image models for Pickups: text-to-image and edit graphs, rebuilt from Comfy's official templates.

Every builder is a flattened copy of one ComfyUI template (named in its docstring): the same nodes, sampler,
scheduler, steps, CFG, shift/sigmas and latent node. Left out on purpose, the same way for every model: the templates'
optional prompt-enhancer branch (an LLM rewrite behind a switch; Pickups sends the prompt as written), their style-LoRA
switch (off) and their preview/compare nodes; every graph saves with SaveImage. Where the server has a better precision
of the same weights than the template names, the builder uses it (FLUX 3 klein and its text encoder in bf16).
Graphs use the graphs.py conventions: LoadImage nodes PU_Ref1, PU_Ref2, ..., the prompt node PU_Prompt, the
seed-bearing node PU_Seed and the SaveImage node PU_Out.

How to write prompts, per model (from each template's notes, enhancer system prompt and model card):
- FLUX 3 klein: one continuous prose caption of the finished image, medium and style first, then scene, light and
  camera, then each subject with concrete colours, materials and frame positions (100-350 words; short prompts give
  generic images). Edits: caption the whole result with the change folded in, not a one-line instruction; references
  are <ref_image_0>, <ref_image_1>, ...
- Qwen-Image 2.1 / Turbo: generation is one long English paragraph describing the finished image as seen, with any
  text to render in double quotes. Edits are instructions that start with the verb; image 1 is the canvas, and with 2+
  images refer to them as <image1>, <image2>, ...; name what must stay ("keep everything else unchanged").
- Krea 2 Turbo: one plain-prose paragraph with subject, action, setting, medium, framing and light; put words to render
  in quotes, otherwise say there is no text. Style reference: describe the content only; the style comes from the
  reference image(s).
- Z-Image Turbo: a detailed natural-language description (English or Chinese), photographic detail welcome; text to
  render goes in quotes. No negative prompt (CFG 1).
- Ideogram 4: a structured JSON caption (high_level_description, style_description with photo or art_style,
  compositional_deconstruction with background then elements; text elements carry the literal text). Plain text trips
  the model's built-in safety filter (a grey card, refusal_card below), so the image task has the director write the
  caption first (director.ideogram_caption); JSON the user wrote is sent as it is.
- HiDream-O1: one detailed English paragraph (subject, composition, action, location, style, camera specs), exact
  text in double quotes with font, colour and position. Edits: an instruction; references keep their order (image 1 is
  the canvas).

Sizes: size_for() gives multiples of 64 at ~1 MP ("1K") or ~4 MP ("2K"), capped by the model's max long side. In edit
modes the output follows reference 1's aspect ratio (as the templates do) and width x height only sets the pixel
budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pickups import graphs
from pickups.graphs import _ref


@dataclass(frozen=True)
class ImageModel:
    id: str
    name: str
    maker: str
    licence: str
    modes: tuple[str, ...]  # "generate", "edit" (reference images), "style" (style reference images)
    max_refs: int
    max_side: int  # max long side it handles well (1024 or 2048; 2560 for HiDream-O1, trained at ~4 MP)
    speed: str  # measured on the reference RTX 4090, warm, one image
    sponsor: bool
    notes: str  # one line for the UI
    files: tuple[str, ...]  # every model file its graphs load, for the setup check


# --- model files ----------------------------------------------------------------------------------------------------
FLUX_KLEIN = graphs.FLUX_KLEIN  # FLUX 3 klein (the step-distilled release), bf16
FLUX_TEXT = graphs.FLUX_TEXT
FLUX_VAE = graphs.FLUX_VAE
QWEN21, QWEN21_TURBO, QWEN21_TEXT, QWEN21_VAE = graphs.QWEN21, graphs.QWEN21_TURBO, graphs.QWEN21_TEXT, graphs.QWEN21_VAE
KREA2_TURBO = "krea2_turbo_int8_convrot.safetensors"
KREA2_TEXT = "qwen3vl_4b_fp8_scaled.safetensors"
KREA2_VAE = "qwen_image_vae.safetensors"
KREA2_STYLE_LORA = "krea2_style_reference.safetensors"
ZIMAGE = "z_image_turbo_bf16.safetensors"
ZIMAGE_TEXT = "qwen_3_4b.safetensors"
ZIMAGE_VAE = "ae.safetensors"
IDEOGRAM4 = "ideogram4_fp8_scaled.safetensors"
IDEOGRAM4_UNCOND = "ideogram4_unconditional_fp8_scaled.safetensors"
IDEOGRAM4_TEXT = "qwen3vl_8b_fp8_scaled.safetensors"
IDEOGRAM4_VAE = "flux2-vae.safetensors"
HIDREAM_O1 = "hidream_o1_image_bf16.safetensors"

MODELS: dict[str, ImageModel] = {m.id: m for m in [
    ImageModel("flux3-klein", "FLUX 3 klein", "Black Forest Labs", "FLUX Kommunity License", ("generate", "edit"), 10, 2048,
               "~5 s at 1K on a 4090", True, "Fast all-rounder: photoreal people, multi-reference edits",
               (FLUX_KLEIN, FLUX_TEXT, FLUX_VAE)),
    ImageModel("qwen-image-2.1", "Qwen-Image 2.1", "Qwen (Alibaba)", "Qwen Research License (non-commercial)", ("generate", "edit"),
               10, 2048, "~5 s at 1K on a 4090", False, "Precise instruction edits and lettering, native 2K",
               (QWEN21, QWEN21_TEXT, QWEN21_VAE)),
    ImageModel("qwen-image-2.1-turbo", "Qwen-Image 2.1 Turbo", "Qwen (Alibaba)", "Qwen Research License (non-commercial)",
               ("generate", "edit"), 10, 2048, "~2 s at 1K on a 4090", False, "Qwen-Image 2.1 in 8 steps, for quick drafts",
               (QWEN21_TURBO, QWEN21_TEXT, QWEN21_VAE)),
    ImageModel("krea2-turbo", "Krea 2 Turbo", "Krea", "Krea 2 Community License", ("generate", "style"), 3, 2048,
               "~4 s at 1K on a 4090", False, "Aesthetic stills; style references lend it a look",
               (KREA2_TURBO, KREA2_TEXT, KREA2_VAE, KREA2_STYLE_LORA)),
    ImageModel("z-image-turbo", "Z-Image Turbo", "Tongyi-MAI (Alibaba)", "Apache 2.0", ("generate",), 0, 2048,
               "~4 s at 1K on a 4090", False, "Fast photoreal, English and Chinese text",
               (ZIMAGE, ZIMAGE_TEXT, ZIMAGE_VAE)),
    ImageModel("ideogram4", "Ideogram 4", "Ideogram", "Ideogram 4 Non-Commercial", ("generate",), 0, 2048,
               "~16 s at 1K on a 4090", False, "Best for lettering and layout (JSON prompts)",
               (IDEOGRAM4, IDEOGRAM4_UNCOND, IDEOGRAM4_TEXT, IDEOGRAM4_VAE)),
    ImageModel("hidream-o1", "HiDream-O1", "HiDream.ai", "MIT", ("generate", "edit"), 10, 2560,
               "~45 s at 2K on a 4090", False, "Made for 2K: slow and detailed; 1K comes out soft",
               (HIDREAM_O1,)),
]}

DEFAULT = "flux3-klein"

ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1), "4:5": (4, 5), "3:4": (3, 4), "2:3": (2, 3), "9:16": (9, 16),
    "16:9": (16, 9), "3:2": (3, 2), "4:3": (4, 3), "21:9": (21, 9),
}
RESOLUTIONS: dict[str, int] = {"1K": 1024, "2K": 2048}  # side of the square with the same pixel count


def size_for(aspect: str, resolution: str, model: ImageModel) -> tuple[int, int]:
    """Width and height for an aspect ratio at ~1 MP ("1K") or ~4 MP ("2K"): multiples of 64, long side <= max_side."""
    aw, ah = ASPECTS[aspect]
    side = RESOLUTIONS[resolution]
    w = math.sqrt(side * side * aw / ah)
    h = w * ah / aw
    scale = min(1.0, model.max_side / max(w, h))
    w, h = w * scale, h * scale

    def snap(v: float) -> int:
        return max(64, min(model.max_side // 64 * 64, round(v / 64) * 64))

    return snap(w), snap(h)


# --- helpers ----------------------------------------------------------------------------------------------------------
def _rename(graph: dict, names: dict[str, str]) -> dict:
    """Rename node ids and every link that points at them."""
    out = {}
    for nid, node in graph.items():
        inputs = {k: ([names.get(v[0], v[0]), v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) else v)
                  for k, v in node["inputs"].items()}
        out[names.get(nid, nid)] = {**node, "inputs": inputs}
    return out


def _load_refs(refs: Sequence) -> dict:
    return {f"PU_Ref{i + 1}": {"class_type": "LoadImage", "inputs": {"image": r}} for i, r in enumerate(refs)}


def _save(images: list) -> dict:
    return {"PU_Out": {"class_type": "SaveImage", "inputs": {"images": images, "filename_prefix": "pickups/image"}}}


def _qwen_resolution(width: int, height: int) -> int:
    """TextEncodeQwenImage21's `resolution` is a pixel budget (resolution^2), in steps of 32."""
    return max(32, round(math.sqrt(width * height) / 32) * 32)


def _megapixels(width: int, height: int) -> float:
    """Comfy's "megapixels" inputs count 1024 x 1024 as 1.0."""
    return round(width * height / (1024 * 1024), 2)


# --- FLUX 3 klein -------------------------------------------------------------------------------------------------------
def flux3_klein_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_flux3_klein_distilled_t2i: 15 steps, CFG 1, euler / simple, EmptyFlux2LatentImage (graphs.still)."""
    g = graphs.still(prompt, width, height, seed=seed)
    g["flat"]["inputs"]["batch_size"] = count
    return g


def flux3_klein_edit(prompt: str, width: int, height: int, seed: int, refs: Sequence) -> dict:
    """image_flux3_klein_distilled_image_edit: references through ReferenceMultiLatent (resized to the pixel budget),
    negative = ConditioningZeroOut(positive), 16 steps, CFG 1, euler / simple. The latent is ReferenceMultiLatent's
    empty latent, so the output takes reference 1's aspect."""
    g = {
        "funet": {"class_type": "UNETLoader", "inputs": {"unet_name": FLUX_KLEIN, "weight_dtype": "default"}},
        "fclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX_TEXT, "type": "flux3", "device": "default"}},
        "fvae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        **_load_refs(refs),
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("fclip"), "text": prompt}},
        "fneg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("PU_Prompt")}},
        "frefs": {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": _ref("PU_Prompt"), "negative": _ref("fneg"), "vae": _ref("fvae"), "resolution": "resize",
            "resolution.megapixels": _megapixels(width, height),
            **{f"images.image{i}": _ref(f"PU_Ref{i + 1}") for i in range(len(refs))}}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("funet"), "positive": _ref("frefs", 0), "negative": _ref("frefs", 1),
                                                       "latent_image": _ref("frefs", 2), "seed": seed, "steps": 16, "cfg": 1.0,
                                                       "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "fdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("fvae")}},
    }
    return g | _save(_ref("fdec"))


# --- Qwen-Image 2.1 -----------------------------------------------------------------------------------------------------
def _qwen21_base(turbo: bool) -> dict:
    return {
        "q21": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN21_TURBO if turbo else QWEN21, "weight_dtype": "default"}},
        "q21cache": {"class_type": "QwenImage21Cache", "inputs": {"model": _ref("q21"), "device": "auto", "dtype": "default"}},
        "q21clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN21_TEXT, "type": "qwen_image", "device": "default"}},
        "q21vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_VAE}},
    }


def qwen21_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_qwen_image_2_1_t2i: TextEncodeQwenImage21 without images, EmptyLatentImage, KSampler 25 steps, CFG 1,
    euler / simple on QwenImage21Cache."""
    g = _qwen21_base(turbo=False) | {
        "PU_Prompt": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": _ref("q21clip"), "prompt": prompt, "negative_prompt": "",
                                                                       "resolution": 1024}},
        "q21lat": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("q21cache"), "positive": _ref("PU_Prompt", 0),
                                                       "negative": _ref("PU_Prompt", 1), "latent_image": _ref("q21lat"), "seed": seed,
                                                       "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                       "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("q21vae")}},
    }
    return g | _save(_ref("decode"))


def qwen21_turbo_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_qwen_image_2_1_turbo_t2i: the 8-step Turbo sigmas (ManualSigmas), euler, CFGGuider 1, EmptyLatentImage."""
    g = _qwen21_base(turbo=True) | {
        "PU_Prompt": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": _ref("q21clip"), "prompt": prompt, "negative_prompt": "",
                                                                       "resolution": 1024}},
        "q21lat": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": graphs.QWEN21_TURBO_SIGMAS}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("q21cache"), "positive": _ref("PU_Prompt", 0),
                                                         "negative": _ref("PU_Prompt", 1), "cfg": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("q21lat")}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("q21vae")}},
    }
    return g | _save(_ref("decode"))


def _qwen21_edit(prompt: str, width: int, height: int, seed: int, refs: Sequence, turbo: bool) -> dict:
    """image_qwen_image_2_1[_turbo]_image_edit through graphs.qwen21_edit, renamed to this module's conventions.

    `resolution` is the template's pixel budget (custom_size off: the canvas comes from image 1's encode latent).
    """
    g = graphs.qwen21_edit(refs[0], prompt, seed=seed, image2=refs[1] if len(refs) > 1 else None, turbo=turbo,
                           resolution=_qwen_resolution(width, height), extra=list(refs[2:]))
    names = {"PU_Frame": "PU_Ref1", "PU_Plate": "PU_Out"} | {f"PU_Frame{i}": f"PU_Ref{i}" for i in range(2, len(refs) + 1)}
    return _rename(g, names)


def qwen21_edit(prompt: str, width: int, height: int, seed: int, refs: Sequence) -> dict:
    """image_qwen_image_2_1_image_edit: 25 steps, CFG 1, euler / simple."""
    return _qwen21_edit(prompt, width, height, seed, refs, turbo=False)


def qwen21_turbo_edit(prompt: str, width: int, height: int, seed: int, refs: Sequence) -> dict:
    """image_qwen_image_2_1_turbo_image_edit: the 8-step Turbo sigmas, CFG 1, euler."""
    return _qwen21_edit(prompt, width, height, seed, refs, turbo=True)


# --- Krea 2 Turbo -------------------------------------------------------------------------------------------------------
def _krea2_base() -> dict:
    return {
        "kunet": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA2_TURBO, "weight_dtype": "default"}},
        "kclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA2_TEXT, "type": "krea2", "device": "default"}},
        "kvae": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
    }


def krea2_turbo_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_krea2_turbo_t2i_int8 (style LoRA off): KSampler 8 steps, CFG 1, euler / simple, EmptyLatentImage,
    negative = ConditioningZeroOut(positive). No ModelSampling node, as in the template."""
    g = _krea2_base() | {
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("kclip"), "text": prompt}},
        "kneg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("PU_Prompt")}},
        "klat": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("kunet"), "positive": _ref("PU_Prompt"), "negative": _ref("kneg"),
                                                       "latent_image": _ref("klat"), "seed": seed, "steps": 8, "cfg": 1.0,
                                                       "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "kdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("kvae")}},
    }
    return g | _save(_ref("kdec"))


def krea2_turbo_style(prompt: str, width: int, height: int, seed: int, refs: Sequence, count: int = 1) -> dict:
    """image_krea2_turbo_int8_image_style_reference: the style-reference LoRA (1.0), up to 3 references through
    TextEncodeQwenImageEditPlus + FluxKontextMultiReferenceLatentMethod (index_timestep_zero), ModelSamplingFlux
    (1.15 / 0.5), BasicScheduler simple 8 steps, euler, CFGGuider 1, EmptyLatentImage at the requested size."""
    g = _krea2_base() | _load_refs(refs) | {
        "klora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("kunet"), "lora_name": KREA2_STYLE_LORA, "strength_model": 1.0}},
        "kshift": {"class_type": "ModelSamplingFlux", "inputs": {"model": _ref("klora"), "max_shift": 1.15, "base_shift": 0.5,
                                                                 "width": width, "height": height}},
        "PU_Prompt": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": _ref("kclip"), "vae": _ref("kvae"), "prompt": prompt,
            **{f"image{i + 1}": _ref(f"PU_Ref{i + 1}") for i in range(len(refs))}}},
        "kref": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": _ref("PU_Prompt"),
                                                                                   "reference_latents_method": "index_timestep_zero"}},
        "kneg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("kref")}},
        "klat": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": _ref("kshift"), "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("kshift"), "positive": _ref("kref"), "negative": _ref("kneg"), "cfg": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("klat")}},
        "kdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("kvae")}},
    }
    return g | _save(_ref("kdec"))


# --- Z-Image Turbo ------------------------------------------------------------------------------------------------------
def z_image_turbo_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_z_image_turbo: ModelSamplingAuraFlow shift 3, KSampler 8 steps, CFG 1, res_multistep / simple,
    EmptySD3LatentImage, negative = ConditioningZeroOut(positive)."""
    g = {
        "zunet": {"class_type": "UNETLoader", "inputs": {"unet_name": ZIMAGE, "weight_dtype": "default"}},
        "zclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_TEXT, "type": "lumina2", "device": "default"}},
        "zvae": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "zshift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("zunet"), "shift": 3.0}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("zclip"), "text": prompt}},
        "zneg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("PU_Prompt")}},
        "zlat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("zshift"), "positive": _ref("PU_Prompt"), "negative": _ref("zneg"),
                                                       "latent_image": _ref("zlat"), "seed": seed, "steps": 8, "cfg": 1.0,
                                                       "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "zdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("zvae")}},
    }
    return g | _save(_ref("zdec"))


# --- Ideogram 4 ---------------------------------------------------------------------------------------------------------
IDEOGRAM4_PRESET = {"steps": 20, "mu": 0.0, "std": 1.75}  # the template's "Default" preset (V4_DEFAULT_20)


def ideogram4_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_ideogram4_t2i: asymmetric CFG with two models (DualModelGuider, CFG 7, unconditional model for the
    negative pass on zeroed text), CFGOverride 3 over the last 30% of sampling, Ideogram4Scheduler (Default preset:
    20 steps, mu 0, std 1.75), euler, EmptyFlux2LatentImage. The template's size math (round up to 16, min 256) is
    folded in here."""
    w, h = (max((v + 15) // 16 * 16, 256) for v in (width, height))
    g = {
        "iunet": {"class_type": "UNETLoader", "inputs": {"unet_name": IDEOGRAM4, "weight_dtype": "default"}},
        "iuncond": {"class_type": "UNETLoader", "inputs": {"unet_name": IDEOGRAM4_UNCOND, "weight_dtype": "default"}},
        "iclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": IDEOGRAM4_TEXT, "type": "ideogram4", "device": "default"}},
        "ivae": {"class_type": "VAELoader", "inputs": {"vae_name": IDEOGRAM4_VAE}},
        "icfg": {"class_type": "CFGOverride", "inputs": {"model": _ref("iunet"), "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("iclip"), "text": prompt}},
        "ineg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("PU_Prompt")}},
        "guider": {"class_type": "DualModelGuider", "inputs": {"model": _ref("icfg"), "positive": _ref("PU_Prompt"),
                                                               "model_negative": _ref("iuncond"), "negative": _ref("ineg"), "cfg": 7.0}},
        "sigmas": {"class_type": "Ideogram4Scheduler", "inputs": {"steps": IDEOGRAM4_PRESET["steps"], "width": w, "height": h,
                                                                  "mu": IDEOGRAM4_PRESET["mu"], "std": IDEOGRAM4_PRESET["std"]}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "ilat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": count}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("ilat")}},
        "idec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("ivae")}},
    }
    return g | _save(_ref("idec"))


# Models whose weights draw a refusal card instead of an image (a learned refusal, not a node).
REFUSES = {"ideogram4"}
REFUSED = "Ideogram's built-in filter refused this prompt. Try other words."


def refusal_card(path) -> bool:
    """Ideogram 4's grey "Image blocked by safety filter" card: luma mean about 111 and std at most ~10.5 on every one
    measured, where real images have std 60-75 and a flat checkerboard 14.6 (R&D, Oct 27 2026). Mean 100-120 and std
    under 12."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        g = im.convert("L")
        g.thumbnail((768, 768))
        st = ImageStat.Stat(g)
    return 100.0 <= st.mean[0] <= 120.0 and st.stddev[0] < 12.0


# --- HiDream-O1 ---------------------------------------------------------------------------------------------------------
def _hidream_o1(prompt: str, seed: int, latent: dict, refs: Sequence) -> dict:
    """image_hidream_o1, shared part: one pixel-space checkpoint, ModelNoiseScale 8, patch-seam smoothing over the last
    20% (single_shift, ramp_2_4, median), BasicScheduler normal 40 steps (on the noise-scaled model), SamplerCustom
    CFG 5 with dpmpp_2m_sde_gpu."""
    g = {
        "hckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": HIDREAM_O1}},
        "hnoise": {"class_type": "ModelNoiseScale", "inputs": {"model": _ref("hckpt", 0), "noise_scale": 8.0}},
        "hseam": {"class_type": "HiDreamO1PatchSeamSmoothing", "inputs": {"model": _ref("hnoise"), "start_percent": 0.8, "end_percent": 1.0,
                                                                         "pattern": "single_shift", "passes": "ramp_2_4", "blend": "median",
                                                                         "strength": 1.0}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("hckpt", 1), "text": prompt}},
        "hneg": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("hckpt", 1), "text": ""}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m_sde_gpu"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": _ref("hnoise"), "scheduler": "normal", "steps": 40, "denoise": 1.0}},
        **latent,
    }
    pos, neg = _ref("PU_Prompt"), _ref("hneg")
    if refs:
        g |= _load_refs(refs)
        g["hrefs"] = {"class_type": "HiDreamO1ReferenceImages", "inputs": {
            "positive": pos, "negative": neg, **{f"images.image_{i + 1}": _ref(f"PU_Ref{i + 1}") for i in range(len(refs))}}}
        pos, neg = _ref("hrefs", 0), _ref("hrefs", 1)
    g |= {
        "PU_Seed": {"class_type": "SamplerCustom", "inputs": {"model": _ref("hseam"), "add_noise": True, "noise_seed": seed, "cfg": 5.0,
                                                            "positive": pos, "negative": neg, "sampler": _ref("sampler"),
                                                            "sigmas": _ref("sigmas"), "latent_image": _ref("hlat")}},
        "hdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed", 0), "vae": _ref("hckpt", 2)}},
    }
    return g | _save(_ref("hdec"))


def hidream_o1_t2i(prompt: str, width: int, height: int, seed: int, count: int = 1) -> dict:
    """image_hidream_o1 (edit switch off): EmptyHiDreamO1LatentImage at the requested size."""
    lat = {"hlat": {"class_type": "EmptyHiDreamO1LatentImage", "inputs": {"width": width, "height": height, "batch_size": count}}}
    return _hidream_o1(prompt, seed, lat, ())


def hidream_o1_edit(prompt: str, width: int, height: int, seed: int, refs: Sequence, count: int = 1) -> dict:
    """image_hidream_o1 (edit switch on): references through HiDreamO1ReferenceImages; the latent is reference 1's size
    floored to 32, after the template's optional ImageScaleToTotalPixels (lanczos) set to the requested pixel budget."""
    lat = {
        "hscale": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": _ref("PU_Ref1"), "upscale_method": "lanczos",
                                                                     "megapixels": _megapixels(width, height), "resolution_steps": 1}},
        "hsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("hscale")}},
        "hw": {"class_type": "ComfyMathExpression", "inputs": {"expression": "floor(a/32)*32", "values.a": _ref("hsize", 0)}},
        "hh": {"class_type": "ComfyMathExpression", "inputs": {"expression": "floor(a/32)*32", "values.a": _ref("hsize", 1)}},
        "hlat": {"class_type": "EmptyHiDreamO1LatentImage", "inputs": {"width": _ref("hw", 1), "height": _ref("hh", 1), "batch_size": count}},
    }
    return _hidream_o1(prompt, seed, lat, refs)


# --- registry -----------------------------------------------------------------------------------------------------------
# (model id, mode) -> builder(prompt, width, height, seed, refs, count). Edit builders whose latent comes from the
# references (FLUX ReferenceMultiLatent, Qwen's encode latent) have no batch size, and HiDream-O1 edits (~2.5 min each
# at 2K) aren't batch-tested: callers loop seeds for count > 1.
BUILDERS: dict[tuple[str, str], Callable[..., dict]] = {
    ("flux3-klein", "generate"): lambda p, w, h, s, r, n: flux3_klein_t2i(p, w, h, s, n),
    ("flux3-klein", "edit"): lambda p, w, h, s, r, n: flux3_klein_edit(p, w, h, s, r),
    ("qwen-image-2.1", "generate"): lambda p, w, h, s, r, n: qwen21_t2i(p, w, h, s, n),
    ("qwen-image-2.1", "edit"): lambda p, w, h, s, r, n: qwen21_edit(p, w, h, s, r),
    ("qwen-image-2.1-turbo", "generate"): lambda p, w, h, s, r, n: qwen21_turbo_t2i(p, w, h, s, n),
    ("qwen-image-2.1-turbo", "edit"): lambda p, w, h, s, r, n: qwen21_turbo_edit(p, w, h, s, r),
    ("krea2-turbo", "generate"): lambda p, w, h, s, r, n: krea2_turbo_t2i(p, w, h, s, n),
    ("krea2-turbo", "style"): lambda p, w, h, s, r, n: krea2_turbo_style(p, w, h, s, r, n),
    ("z-image-turbo", "generate"): lambda p, w, h, s, r, n: z_image_turbo_t2i(p, w, h, s, n),
    ("ideogram4", "generate"): lambda p, w, h, s, r, n: ideogram4_t2i(p, w, h, s, n),
    ("hidream-o1", "generate"): lambda p, w, h, s, r, n: hidream_o1_t2i(p, w, h, s, n),
    ("hidream-o1", "edit"): lambda p, w, h, s, r, n: hidream_o1_edit(p, w, h, s, r, n),
}
NO_BATCH = {("flux3-klein", "edit"), ("qwen-image-2.1", "edit"), ("qwen-image-2.1-turbo", "edit"), ("hidream-o1", "edit")}


def mode_for(model: ImageModel, refs: Sequence = ()) -> str:
    """"generate" without references; with references, "edit" if the model edits, else "style"."""
    if not refs:
        return "generate"
    for mode in ("edit", "style"):
        if mode in model.modes:
            return mode
    raise ValueError(f"{model.name} doesn't take reference images")


def can_batch(model_id: str, refs: Sequence = ()) -> bool:
    """True if build(count > 1) makes one batched job; otherwise call build() once per seed."""
    return (model_id, mode_for(MODELS[model_id], refs)) not in NO_BATCH


def build(model_id: str, prompt: str, width: int, height: int, seed: int, refs: Sequence = (), count: int = 1,
          prefix: str = "pickups/image") -> dict:
    """The API-format graph for one generation. refs: Comfy SDK asset handles or ComfyUI input file names (LoadImage);
    with refs the model edits (or uses them as style references). count > 1 batches where the template's latent has a
    batch size (see can_batch); elsewhere it raises, and callers loop seeds. Output: PU_Out (SaveImage)."""
    model = MODELS[model_id]
    refs = list(refs)
    mode = mode_for(model, refs)
    if len(refs) > model.max_refs:
        raise ValueError(f"{model.name} takes at most {model.max_refs} reference images")
    if count < 1 or (count > 1 and (model_id, mode) in NO_BATCH):
        raise ValueError(f"{model.name} {mode} makes one image per job: call build() once per seed")
    g = BUILDERS[(model_id, mode)](prompt, width, height, seed, refs, count)
    g["PU_Out"]["inputs"]["filename_prefix"] = f"{prefix}/{model_id}"
    return g


# The same weights under other published names (Black Forest Labs' repo vs Comfy-Org repackages, bf16 vs int8). When the
# preferred file isn't installed, the first installed alias is used instead.
ALIASES: dict[str, tuple[str, ...]] = {
    "flux3_image_klein_sd_bf16.safetensors": ("flux3-image-klein-bf16.safetensors", "flux3_image_klein_sd_int8_convrot.safetensors",
                                              "flux3-image-klein-int8-convrot.safetensors"),
    "qwen3vl_4b_bf16.safetensors": ("flux3_qwen3vl_4b_bf16.safetensors", "qwen3vl_4b_int8_convrot.safetensors"),
    "krea2_turbo_int8_convrot.safetensors": ("krea2_turbo_bf16.safetensors", "krea2_turbo_fp8_scaled.safetensors"),
    "z_image_turbo_bf16.safetensors": ("z_image_turbo_int8_convrot.safetensors",),
}


def _aliases(name: str) -> tuple[str, ...]:
    """Every other published name of the same weights (aliases work in both directions)."""
    for key, alts in ALIASES.items():
        group = (key,) + alts
        if name in group:
            return tuple(g for g in group if g != name)
    return ()


def _resolve(name: str, installed: set[str], by_base: dict[str, list[str]]) -> str | None:
    """The installed file to use for `name`: itself, a single subfolder copy, or an installed alias."""
    for cand in (name,) + _aliases(name):
        if cand in installed:
            return cand
        if len(by_base.get(cand, [])) == 1:
            return by_base[cand][0]
    return None


def _by_basename(installed: set[str]) -> dict[str, list[str]]:
    by_base: dict[str, list[str]] = {}
    for name in installed:
        by_base.setdefault(name.replace("\\", "/").rsplit("/", 1)[-1], []).append(name)
    return by_base


def missing(model_id: str, installed: set[str]) -> list[str]:
    """Files the model needs that ComfyUI doesn't have (at the top level, once in a subfolder, or under an alias)."""
    by_base = _by_basename(installed)
    return [f for f in MODELS[model_id].files if _resolve(f, installed, by_base) is None]


def localize(graph: dict, installed: set[str]) -> dict:
    """Point model-file inputs at the installed copy when ComfyUI only has it in a subfolder (e.g. loras/sub/x.safetensors).

    `installed` is Connection.model_files(). Names already installed as-is are left alone.
    """
    by_base = _by_basename(installed)
    fields = ("unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name", "lora_name", "ckpt_name")
    out = {}
    for nid, node in graph.items():
        inputs = dict(node["inputs"])
        for f in fields:
            v = inputs.get(f)
            if isinstance(v, str) and v not in installed:
                inputs[f] = _resolve(v, installed, by_base) or v
        out[nid] = {**node, "inputs": inputs}
    return out


def localize_for(ctx, graph: dict) -> dict:
    """localize() against the model files ctx.conn's ComfyUI has; the graph as-is when there is no answer (no
    connection, as in a one-off runner, or ComfyUI not listing its files)."""
    conn = getattr(ctx, "conn", None)
    try:
        installed = conn.model_files() if conn is not None else set()
    except Exception:  # noqa: BLE001
        installed = set()
    return localize(graph, installed) if installed else graph
