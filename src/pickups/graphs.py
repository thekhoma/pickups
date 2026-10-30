"""Build ComfyUI API-format graphs for Pickups actions.

Graphs are plain dicts in ComfyUI's API format. Node ids are stable strings so
callers (and tests) can address inputs and outputs by name, e.g. graph["PU_Out"].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Model files as installed on the reference 4090 machine. Kept in one place so
# the setup check and backends can map them per host later.
UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"

# Distilled 8-step schedule from the official LTX-2.5 templates.
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

# VAE decode temporal tile, in frames. Tiles of 64 overlapped by 8 put a visible step at frame 56; one tile for the whole
# shot has no seam. 256 holds any take the app renders (97 to 241 frames); Lightricks' own graphs use 128 / 32.
DECODE_FRAMES = 256

DEFAULT_NEGATIVE = "cut, scene change, new objects, text, watermark, blurry, distorted"


def _ref(node: str, slot: int = 0) -> list:
    return [node, slot]


@dataclass(frozen=True)
class Models:
    unet: str = UNET
    text_encoder: str = TEXT_ENCODER
    video_vae: str = VIDEO_VAE
    audio_vae: str = AUDIO_VAE


def _loaders(models: Models) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": models.unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": models.text_encoder, "type": "ltxv", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": models.video_vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": models.audio_vae}},
    }


def _text(prompt: str, negative: str, fps: float) -> dict:
    return {
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": _ref("clip")}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": _ref("clip")}},
        "cond": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": _ref("PU_Prompt"), "negative": _ref("neg"), "frame_rate": fps},
        },
    }


def _sample_and_save(av_latent: list, positive: list, negative: list, seed: int, prefix: str) -> dict:
    return {
        "guider": {
            "class_type": "LTXVDualCFGGuider",
            "inputs": {"model": _ref("unet"), "positive": positive, "negative": negative, "video_cfg": 1.0, "audio_cfg": 1.0},
        },
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": _ref("PU_Seed"),
                "guider": _ref("guider"),
                "sampler": _ref("sampler"),
                "sigmas": _ref("sigmas"),
                "latent_image": av_latent,
            },
        },
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample")}},
        "decode": {
            "class_type": "VAEDecodeTiled",
            "inputs": {"samples": _ref("split", 0), "vae": _ref("vae"), "tile_size": 512, "overlap": 64, "temporal_size": DECODE_FRAMES, "temporal_overlap": 32},
        },
        "adecode": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": _ref("split", 1), "audio_vae": _ref("avae")}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
        "PU_OutAudio": {"class_type": "SaveAudio", "inputs": {"audio": _ref("adecode"), "filename_prefix": f"{prefix}/audio"}},
    }


def smoke_t2v(prompt: str, seed: int = 42, width: int = 512, height: int = 288, frames: int = 25, fps: float = 24.0,
              models: Models = Models(), prefix: str = "pickups/k0") -> dict:
    """Tiny text-to-video job used to prove the SDK -> proxy -> GPU path (test K0)."""
    g = _loaders(models) | _text(prompt, DEFAULT_NEGATIVE, fps)
    g |= {
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": _ref("avae")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("vlat"), "audio_latent": _ref("alat")}},
    }
    g |= _sample_and_save(_ref("av"), _ref("cond", 0), _ref("cond", 1), seed, prefix)
    return g


SAM_CKPT = "sam3.1_multiplex_fp16.safetensors"
INPAINT_LORA = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"


QWEN_EDIT = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_TEXT = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"


QWEN_LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def qwen_size(width: int, height: int, megapixels: float = 1.0, multiple: int = 16) -> tuple[int, int]:
    """Qwen-friendly working size with the SAME aspect as the source, so the plate maps back pixel-for-pixel."""
    scale = (megapixels * 1_000_000 / (width * height)) ** 0.5
    return max(multiple, round(width * scale / multiple) * multiple), max(multiple, round(height * scale / multiple) * multiple)


def clean_plate(image: object, instruction: str, seed: int = 42, fast: bool = True,
                prefix: str = "pickups/plate", image2: object | None = None, size: tuple[int, int] | None = None,
                image3: object | None = None, steps: int | None = None, cfg: float | None = None, negative: str = "") -> dict:
    """Qwen Image Edit 2511: remove something from one frame ("Remove the boom microphone. Keep everything else.").

    Mirrors Comfy's image_qwen_image_edit_2511 template. fast=True uses the Lightning 4-step LoRA (4 steps, CFG 1);
    fast=False is the full 40 steps at CFG 4, or the steps and CFG given (the commercial-safe start frame: 30 at CFG 3, with
    its negative). Output: PU_Plate (one image).
    """
    steps, cfg = (4, 1.0) if fast else (steps or 40, cfg if cfg is not None else 4.0)
    g = {
        "PU_Frame": {"class_type": "LoadImage", "inputs": {"image": image}},
        "qunet": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_EDIT, "weight_dtype": "default"}},
        "qclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TEXT, "type": "qwen_image", "device": "default"}},
        "qvae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    model = _ref("qunet")
    if fast:
        g["lightning"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": QWEN_LIGHTNING, "strength_model": 1.0}}
        model = _ref("lightning")
    g |= {
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": _ref("shift"), "strength": 1.0}},
        # FluxKontextImageScale snaps to its own aspect buckets, which shifts geometry; keep the source aspect instead.
        "scaled": ({"class_type": "ImageScale", "inputs": {"image": _ref("PU_Frame"), "upscale_method": "lanczos",
                                                          "width": size[0], "height": size[1], "crop": "disabled"}}
                   if size else {"class_type": "FluxKontextImageScale", "inputs": {"image": _ref("PU_Frame")}}),
        "PU_Instruction": {"class_type": "TextEncodeQwenImageEditPlus",
                           "inputs": {"clip": _ref("qclip"), "prompt": instruction, "vae": _ref("qvae"), "image1": _ref("scaled"),
                                      **({"image2": _ref("PU_Frame2")} if image2 is not None else {}),
                                      **({"image3": _ref("PU_Frame3")} if image3 is not None else {})}},
        "qneg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": _ref("qclip"), "prompt": negative, "vae": _ref("qvae"), "image1": _ref("scaled")}},
        "refp": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": _ref("PU_Instruction"), "reference_latents_method": "index_timestep_zero"}},
        "refn": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": _ref("qneg"), "reference_latents_method": "index_timestep_zero"}},
        "qlat": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("scaled"), "vae": _ref("qvae")}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("norm"), "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                                       "scheduler": "simple", "positive": _ref("refp"), "negative": _ref("refn"),
                                                       "latent_image": _ref("qlat"), "denoise": 1.0}},
        "qdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("qvae")}},
        "PU_Plate": {"class_type": "SaveImage", "inputs": {"images": _ref("qdec"), "filename_prefix": f"{prefix}/plate"}},
    }
    if image2 is not None:
        # Second reference (e.g. a character sheet): "render image 1 with the character from image 2".
        g["PU_Frame2"] = {"class_type": "LoadImage", "inputs": {"image": image2}}
    if image3 is not None:
        g["PU_Frame3"] = {"class_type": "LoadImage", "inputs": {"image": image3}}
    return g


def edit_size(width: int, height: int, megapixels: float = 1.0) -> tuple[int, int]:
    """Qwen-Image-Edit 2511's working size for a frame: about one megapixel, the frame's own aspect, multiples of 16."""
    k = (megapixels * 1_000_000 / (width * height)) ** 0.5
    return max(16, round(width * k / 16) * 16), max(16, round(height * k / 16) * 16)


def edit_model(commercial_safe: bool) -> str:
    """The frame editor's name for take receipts."""
    return "Qwen-Image-Edit 2511" if commercial_safe else "Qwen-Image 2.1"


def image_edit(image: object, instruction: str, seed: int = 42, refs: list | tuple = (), resolution: int = 0,
               prefix: str = "pickups/edit", commercial_safe: bool = False, size: tuple[int, int] | None = None) -> dict:
    """One-frame edits for Remove plates, previz look frames and Change weather first frames. Output: PU_Plate.

    Default: Qwen-Image 2.1 (best quality, Qwen Research License, non-commercial). commercial_safe=True uses
    Qwen-Image-Edit 2511 (Apache-2.0) with the Lightning 4-step LoRA instead: up to two reference images, and
    `<imageN>` tags become the "Picture N" names that model's encoder gives its inputs.
    """
    if not commercial_safe:
        return qwen21_edit(image, instruction, seed=seed, image2=refs[0] if refs else None, extra=list(refs[1:]),
                           resolution=resolution, prefix=prefix)
    text = re.sub(r"<image(\d+)>", lambda m: f"Picture {m.group(1)}", instruction)
    return clean_plate(image, text, seed=seed, fast=True, prefix=prefix, size=size,
                       image2=refs[0] if refs else None, image3=refs[1] if len(refs) > 1 else None)


RELIGHT_PROMPT = ("Relight Picture 1 so the person is lit exactly as the people in Picture 2: the same soft or hard light from the same "
                  "direction, the same colour and strength, with the same shading on skin, cloth and metal. Matte surfaces, no studio "
                  "light, no rim light, no glow, no added highlights. Keep the pose, the face, the clothes, every detail and the framing "
                  "of Picture 1 exactly.")


def relight(paint: object, source: object, seed: int = 42, prefix: str = "pickups/relight", size: tuple[int, int] | None = None) -> dict:
    """Person swap's optional Relight: Qwen-Image-Edit 2511 (Apache-2.0, Lightning 4-step) relights the painted first frame
    with the source frame as the light reference. Output: PU_Plate. The caller keeps only its low-frequency shading."""
    return image_edit(paint, RELIGHT_PROMPT, seed=seed, refs=[source], prefix=prefix, commercial_safe=True, size=size)


UNION_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
LAYOUT_LORA = "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors"  # Blender/Unreal playblast -> render (ref downscale 1)
REFINE_LORA = "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
DA3_MODEL = "depth_anything_3_mono_large.safetensors"


def drive(control: object, first_frame: object, prompt: str, fps: float, width: int, height: int,
          control_mode: str = "canny", union_strength: float = 1.0, canny_low: float = 0.15, canny_high: float = 0.4,
          seed: int = 42, negative: str = DEFAULT_NEGATIVE, models: Models = Models(), prefix: str = "pickups/drive",
          dev_first_pass: bool = False, dev_steps: int = 30, dev_cfg: float = 3.0, ic_lora: str = UNION_LORA,
          first_at: int = 0, first_strength: float = 1.0) -> dict:
    """Drive: a reference/previz video steers structure and motion; `first_frame` sets the look and character.

    Rebuilt from Lightricks' LTX-2.5_ICLoRA_Union_Control_Distilled with core preprocessing only:
    control_mode "canny" runs ComfyUI's Canny on `control`; "depth" runs Depth Anything 3 on it (softer: shapes and
    distance without the previz's hard edges); "video" means `control` is already a control pass (e.g. a Blender
    depth render). Pass 1 at half size with the Union IC-LoRA guide, pass 2 upsampled x2.
    `control` must be width x height (multiples of 128, so the guide's half-size latent stays even) with 8k+1 frames;
    `first_frame` any size (scaled here). ic_lora=LAYOUT_LORA uses Lightricks' Layout-To-Render IC-LoRA instead: pass
    the raw playblast with control_mode "video"; its reference is full size, so multiples of 64 are enough
    (1920x1088 is its known-good size). dev_first_pass runs pass 1 on the dev model (30 steps, real CFG), so the
    negative prompt takes effect; slower, fewer invented objects.
    first_at > 0: `first_frame` is the shot's frame `first_at` (a multiple of 8, so it lands on a latent frame), not its
    first: it is added as an LTXVAddGuide keyframe there in both passes (as previz pins a look frame painted over a later
    playblast frame), and the frames before it are generated towards it. Recast on LTX paints the new character over the
    frame where the person is biggest, which is often not the first.
    """
    hw, hh = width // 2, height // 2
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "union": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": ic_lora, "strength_model": union_strength}},
        "PU_Control": {"class_type": "LoadVideo", "inputs": {"file": control}},
        "cparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Control")}},
        "PU_First": {"class_type": "LoadImage", "inputs": {"image": first_frame}},
        "first_half": {"class_type": "ImageScale", "inputs": {"image": _ref("PU_First"), "upscale_method": "lanczos", "width": hw, "height": hh, "crop": "center"}},
        "first_full": {"class_type": "ImageScale", "inputs": {"image": _ref("PU_First"), "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}},
    }
    if control_mode == "canny":
        # Edges at the guide's own (half) size: the same guide, a quarter of the work. Canny over ~100 full-size frames
        # was the common factor in two ComfyUI hangs.
        g["cparts_half"] = {"class_type": "ImageScale", "inputs": {"image": _ref("cparts", 0), "upscale_method": "lanczos",
                                                                   "width": hw, "height": hh, "crop": "disabled"}}
        g["ctl"] = {"class_type": "Canny", "inputs": {"image": _ref("cparts_half"), "low_threshold": canny_low, "high_threshold": canny_high}}
        ctl = _ref("ctl")
    elif control_mode == "depth":
        g["da3"] = {"class_type": "LoadDA3Model", "inputs": {"model_name": DA3_MODEL, "weight_dtype": "default"}}
        g["geo"] = {"class_type": "DA3Inference", "inputs": {"da3_model": _ref("da3"), "image": _ref("cparts", 0), "resolution": 504,
                                                            "resize_method": "upper_bound_resize", "mode": "mono"}}
        g["ctl"] = {"class_type": "DA3Render", "inputs": {"da3_geometry": _ref("geo"), "output": "depth",
                                                         "output.normalization": "v2_style", "output.apply_sky_clip": True}}
        ctl = _ref("ctl")
    else:
        ctl = _ref("cparts", 0)
    g |= {
        "ctl_half": {"class_type": "ImageScale", "inputs": {"image": ctl, "upscale_method": "lanczos", "width": hw, "height": hh, "crop": "disabled"}},
        "PU_ControlPreview": {"class_type": "SaveImage", "inputs": {"images": _ref("ctl_half"), "filename_prefix": f"{prefix}/control"}},
        "csize": {"class_type": "GetImageSize", "inputs": {"image": _ref("ctl_half")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": hw, "height": hh, "length": _ref("csize", 2), "batch_size": 1}},
        "i2v1": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": _ref("vae"), "image": _ref("first_half"), "latent": _ref("vlat"),
                                                                  "strength": 1.0, "bypass": False}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("i2v1"), "image": _ref("ctl_half"),
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("union", 1), "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": _ref("csize", 2), "frame_rate": fps, "batch_size": 1, "audio_vae": _ref("avae")}},
        "av1": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("guide", 2), "audio_latent": _ref("alat")}},
        "guider1": ({"class_type": "LTXVDualCFGGuider", "inputs": {"model": _ref("union_dev"), "positive": _ref("guide", 0),
                                                                    "negative": _ref("guide", 1), "video_cfg": dev_cfg, "audio_cfg": 7.0}}
                    if dev_first_pass else
                    {"class_type": "CFGGuider", "inputs": {"model": _ref("union"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "cfg": 1.0}}),
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas1": ({"class_type": "LTXVScheduler", "inputs": {"steps": dev_steps, "max_shift": 2.05, "base_shift": 0.95,
                                                                "stretch": True, "terminal": 0.1, "latent": _ref("av1")}}
                    if dev_first_pass else {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}}),
        "sample1": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider1"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas1"), "latent_image": _ref("av1")}},
        "split1": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample1")}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "latent": _ref("split1", 0)}},
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
        "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": _ref("crop", 2), "upscale_model": _ref("upmodel"), "vae": _ref("vae")}},
        "i2v2": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": _ref("vae"), "image": _ref("first_full"), "latent": _ref("up"),
                                                                  "strength": 1.0, "bypass": False}},
        "av2": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("i2v2"), "audio_latent": _ref("split1", 1)}},
        "guider2": {"class_type": "CFGGuider", "inputs": {"model": _ref("union"), "positive": _ref("crop", 0), "negative": _ref("crop", 1), "cfg": 1.0}},
        "seed2": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}},
        "sigmas2": {"class_type": "ManualSigmas", "inputs": {"sigmas": DFR_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("seed2"), "guider": _ref("guider2"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas2"), "latent_image": _ref("av2")}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample")}},
        # One temporal tile for a whole 97-frame shot (128 covers it): at 64/8 the second tile began at frame 56, and
        # the pair of decodes blended there made every render stall for a few frames about two seconds in.
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("split", 0), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    if first_at > 0:
        # No first frame in place: the frame is a keyframe guide at `first_at` in both passes (cropped off after each).
        del g["i2v1"], g["i2v2"]
        g["guide"]["inputs"]["latent"] = _ref("vlat")
        g["key1"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": _ref("guide", 0), "negative": _ref("guide", 1), "vae": _ref("vae"), "latent": _ref("guide", 2),
            "image": _ref("first_half"), "frame_idx": first_at, "strength": first_strength}}
        g["av1"]["inputs"]["video_latent"] = _ref("key1", 2)
        for node in ("guider1", "crop"):
            g[node]["inputs"]["positive"], g[node]["inputs"]["negative"] = _ref("key1", 0), _ref("key1", 1)
        g["key2"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": _ref("crop", 0), "negative": _ref("crop", 1), "vae": _ref("vae"), "latent": _ref("up"),
            "image": _ref("first_full"), "frame_idx": first_at, "strength": first_strength}}
        g["av2"]["inputs"]["video_latent"] = _ref("key2", 2)
        g["guider2"]["inputs"]["positive"], g["guider2"]["inputs"]["negative"] = _ref("key2", 0), _ref("key2", 1)
        g["crop2"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("key2", 0), "negative": _ref("key2", 1),
                                                                  "latent": _ref("split", 0)}}
        g["decode"]["inputs"]["samples"] = _ref("crop2", 2)
    if dev_first_pass:
        g["dev"] = {"class_type": "UNETLoader", "inputs": {"unet_name": DEV_UNET, "weight_dtype": "default"}}
        g["union_dev"] = {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("dev"), "lora_name": ic_lora,
                                                                               "strength_model": union_strength}}
    return g


SDPOSE_CKPT = "sdpose_wholebody_fp16.safetensors"  # Comfy-Org/SDPose checkpoints/; built-in SDPose nodes, no custom nodes


def sdpose(clip: object, prefix: str = "pickups/person_pose", hands: bool = False) -> dict:
    """Body, feet and face as a pose map (ComfyUI's SDPose whole-body template), and the hands when asked. Output:
    PU_Pose. Drawn, the hands' red and blue finger sticks came back as red, smeared hands in the render (a swinging
    hand in shadow, a hand half behind a cloak); without them LTX draws the hands itself, at the wrists the body
    sticks give."""
    return {
        "clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("clip")}},
        "sdp": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDPOSE_CKPT}},
        "kp": {"class_type": "SDPoseKeypointExtractor", "inputs": {"model": _ref("sdp"), "vae": _ref("sdp", 2), "image": _ref("parts"), "batch_size": 16}},
        "draw": {"class_type": "SDPoseDrawKeypoints", "inputs": {
            "keypoints": _ref("kp"), "draw_body": True, "draw_hands": hands, "draw_face": True, "draw_feet": True,
            "stick_width": 4, "face_point_size": 2, "score_threshold": 0.3, "draw_head": True}},
        "PU_Pose": {"class_type": "SaveImage", "inputs": {"images": _ref("draw"), "filename_prefix": f"{prefix}/pose"}},
    }


def masked_pose(base: object, latmask: object, pose: object, prompt: str, fps: float, seed: int = 42,
                frames: int = 81, union_strength: float = 1.0, pose_strength: float = 1.0,
                negative: str = DEFAULT_NEGATIVE, prefix: str = "pickups/person_pose", tiled: bool = True,
                region: object | None = None, blend_dilation: int = 0) -> dict:
    """LTX-2.5 renders only inside a mask, following a pose map. One pass, at the size of `base`.

    `region` (optional): a white-on-black video, one frame per picture frame, of where LTX may change the picture.
    With it the render is laid onto `base` through Lightricks' `LTXVLaplacianPyramidBlend` (their In/Outpaint
    recipe blends after every stage); output PU_Blend. `blend_dilation` is that node's low-res mask dilation: 0
    measured best on the knight (T1.5), because the node dilates the mask at 64 px on the long side, so even 5
    grows it by about 8 percent of the picture and lets more of the decoded background through.

    `base` is the clean-plate crop with the painted character already on frame 0. `latmask` has one frame per latent
    frame; its first frame is black, so frame 0 is not denoised. The Union Control IC-LoRA follows `pose`.
    `tiled` encodes the base and the pose in tiles: on a 24 GB RTX 4090, 81 frames at 1024x1792 rendered in 87 s
    (the prototype's 96 GB size), with room to spare for the VAE.
    """
    g = _loaders(Models()) | _text(prompt, negative, fps)
    g |= {
        "PU_Base": {"class_type": "LoadVideo", "inputs": {"file": base}},
        "bparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Base")}},
        "PU_LatMask": {"class_type": "LoadVideo", "inputs": {"file": latmask}},
        "lmparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_LatMask")}},
        "lmask": {"class_type": "ImageToMask", "inputs": {"image": _ref("lmparts"), "channel": "red"}},
        "enc": ({"class_type": "VAEEncodeTiled", "inputs": {"pixels": _ref("bparts"), "vae": _ref("vae"), "tile_size": 512,
                                                             "overlap": 64, "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}}
                if tiled else {"class_type": "VAEEncode", "inputs": {"pixels": _ref("bparts"), "vae": _ref("vae")}}),
        "masked": {"class_type": "LTXVSetVideoLatentNoiseMasks", "inputs": {"samples": _ref("enc"), "masks": _ref("lmask")}},
        "PU_Pose": {"class_type": "LoadVideo", "inputs": {"file": pose}},
        "pparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Pose")}},
        "union": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": UNION_LORA, "strength_model": union_strength}},
        "pguide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("masked"),
            "image": _ref("pparts"), "frame_idx": 0, "strength": pose_strength, "latent_downscale_factor": _ref("union", 1),
            "crop": "disabled", "use_tiled_encode": tiled, "tile_size": 512 if tiled else 256, "tile_overlap": 64}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": _ref("avae")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("pguide", 2), "audio_latent": _ref("alat")}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("union"), "positive": _ref("pguide", 0), "negative": _ref("pguide", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas"), "latent_image": _ref("av")}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample", 0)}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("pguide", 0), "negative": _ref("pguide", 1), "latent": _ref("split", 0)}},
        # One temporal tile for a whole 97-frame shot (128 covers it): at 64/8 the second tile began at frame 56 and every
        # render showed a step there (the "frame skip" about two seconds in).
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop", 2), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    if region is not None:
        g |= {
            "PU_Region": {"class_type": "LoadVideo", "inputs": {"file": region}},
            "rparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Region")}},
            "rmask": {"class_type": "ImageToMask", "inputs": {"image": _ref("rparts"), "channel": "red"}},
            "blend": {"class_type": "LTXVLaplacianPyramidBlend", "inputs": {
                "image_a": _ref("decode"), "image_b": _ref("bparts"), "mask": _ref("rmask"), "trim_to_shortest": True,
                "mask_low_res_dilation": blend_dilation}},
            "PU_Blend": {"class_type": "SaveImage", "inputs": {"images": _ref("blend"), "filename_prefix": f"{prefix}/blend"}},
        }
    return g


QWEN21 = "qwen_image_2.1_int8_convrot.safetensors"
QWEN21_TURBO = "qwen_image_2.1_turbo_int8_convrot.safetensors"
QWEN21_TEXT = "qwen3vl_8b_int8_convrot.safetensors"
QWEN21_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN21_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"


def qwen21_edit(image: object, instruction: str, seed: int = 42, image2: object | None = None, turbo: bool = False,
                steps: int = 25, resolution: int = 0, negative: str = "", prefix: str = "pickups/edit",
                extra: list | tuple = (), cfg: float = 1.0) -> dict:
    """Qwen-Image 2.1 edit (one checkpoint for generation and editing), from Comfy's image_qwen_image_2_1_image_edit.

    resolution 0 keeps image 1 at its own size (multiple of 32; the model supports up to ~2K), so a crop is edited
    at its real pixel density. turbo uses the 8-step checkpoint. Output: PU_Plate (same name as clean_plate's, so
    callers can swap editors). extra: more reference images after image2 (image 3, 4, ...), e.g. character sheets and
    a location plate for a previz look frame.
    """
    refs = ([image2] if image2 is not None else []) + list(extra)
    g = {
        "PU_Frame": {"class_type": "LoadImage", "inputs": {"image": image}},
        "q21": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN21_TURBO if turbo else QWEN21, "weight_dtype": "default"}},
        "q21cache": {"class_type": "QwenImage21Cache", "inputs": {"model": _ref("q21"), "device": "auto", "dtype": "default"}},
        "q21clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN21_TEXT, "type": "qwen_image", "device": "default"}},
        "q21vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_VAE}},
        "PU_Prompt": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": _ref("q21clip"), "prompt": instruction, "negative_prompt": negative, "resolution": resolution,
            "vae": _ref("q21vae"), "images.image_1": _ref("PU_Frame"),
            **{f"images.image_{i + 2}": _ref(f"PU_Frame{i + 2}") for i in range(len(refs))}}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("q21cache"), "positive": _ref("PU_Prompt", 0),
                                                         "negative": _ref("PU_Prompt", 1), "cfg": cfg}},
        "sigmas": ({"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN21_TURBO_SIGMAS}} if turbo else
                   {"class_type": "BasicScheduler", "inputs": {"model": _ref("q21cache"), "scheduler": "simple", "steps": steps, "denoise": 1.0}}),
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("PU_Prompt", 2)}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("q21vae")}},
        "PU_Plate": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/plate"}},
    }
    for i, r in enumerate(refs):
        g[f"PU_Frame{i + 2}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
    return g


def start_frame_maps(frame: object, prefix: str = "pickups/start_frame", da3_model: str = "depth_anything_3_mono_large.safetensors",
                     da3_resolution: int = 1512, canny: tuple[float, float] = (0.08, 0.25)) -> dict:
    """The structure maps for a start frame (start_frame.py, method B3), as R&D built them (control_maps_api.json):
    Depth Anything 3 depth (v2_style normalisation, no sky clip) and a soft Canny of the playblast's frame 0.
    Outputs: PU_Depth, PU_Canny."""
    return {
        "PU_Frame": {"class_type": "LoadImage", "inputs": {"image": frame}},
        "da3m": {"class_type": "LoadDA3Model", "inputs": {"model_name": da3_model, "weight_dtype": "default"}},
        "da3": {"class_type": "DA3Inference", "inputs": {"da3_model": _ref("da3m"), "image": _ref("PU_Frame"), "resolution": da3_resolution,
                                                         "resize_method": "upper_bound_resize", "mode": "mono"}},
        "dep": {"class_type": "DA3Render", "inputs": {"da3_geometry": _ref("da3"), "output": "depth",
                                                      "output.normalization": "v2_style", "output.apply_sky_clip": False}},
        "cans": {"class_type": "Canny", "inputs": {"image": _ref("PU_Frame"), "low_threshold": canny[0], "high_threshold": canny[1]}},
        "PU_Depth": {"class_type": "SaveImage", "inputs": {"images": _ref("dep"), "filename_prefix": f"{prefix}/depth"}},
        "PU_Canny": {"class_type": "SaveImage", "inputs": {"images": _ref("cans"), "filename_prefix": f"{prefix}/canny"}},
    }


def start_frame_fresh(depth: object, canny: object, prompt: str, negative: str, seed: int = 42, refs: list | tuple = (),
                      steps: int = 30, cfg: float = 3.0, denoise: float = 1.0, resolution: int = 1536, sampler: str = "euler",
                      scheduler: str = "simple", prefix: str = "pickups/start_frame") -> dict:
    """A start frame made fresh by Qwen-Image 2.1 (start_frame.py, method B3), node for node R&D's B3 graph
    (run_b3/<shot>_v3_B3_api.json): image 1 the depth, image 2 the masked soft canny (image inputs, not a ControlNet), then the
    identity references. The latent comes from TextEncodeQwenImage21 at `resolution` (1536: 2048x1152 at 16:9).
    Output: PU_Plate."""
    g = {
        "q21": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN21, "weight_dtype": "default"}},
        "q21cache": {"class_type": "QwenImage21Cache", "inputs": {"model": _ref("q21"), "device": "auto", "dtype": "default"}},
        "q21clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN21_TEXT, "type": "qwen_image", "device": "default"}},
        "q21vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_VAE}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "PU_Depth": {"class_type": "LoadImage", "inputs": {"image": depth}},
        "PU_Canny": {"class_type": "LoadImage", "inputs": {"image": canny}},
        "PU_Prompt": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": _ref("q21clip"), "prompt": prompt, "negative_prompt": negative, "resolution": resolution, "vae": _ref("q21vae"),
            "images.image_1": _ref("PU_Depth"), "images.image_2": _ref("PU_Canny"),
            **{f"images.image_{i + 3}": _ref(f"PU_Ref{i + 1}") for i in range(len(refs))}}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("q21cache"), "positive": _ref("PU_Prompt", 0),
                                                         "negative": _ref("PU_Prompt", 1), "cfg": cfg}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": _ref("q21cache"), "scheduler": scheduler, "steps": steps, "denoise": denoise}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("PU_Prompt", 2)}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("q21vae")}},
        "PU_Plate": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/plate"}},
    }
    for i, r in enumerate(refs):
        g[f"PU_Ref{i + 1}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
    return g


ZIMAGE_UNET, ZIMAGE_TEXT, ZIMAGE_VAE = "z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"


def start_frame_zbuf(control: object, prompt: str, seed: int = 42, strength: float = 0.75,
                     patch: str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors", steps: int = 8,
                     size: tuple[int, int] = (1920, 1088), prefix: str = "pickups/start_frame") -> dict:
    """A start frame from the previz's own depth (start_frame.py, method zbuf), node for node R&D's l2r_round4 still
    graph: Z-Image Turbo (shift 3) through Fun ControlNet Union 2.1 2602-8steps (ModelPatchLoader + ZImageFunControlnet,
    control on every step), res_multistep / simple, one sampler, CFG 1 with a zeroed negative. Output: PU_Plate."""
    return {
        "zunet": {"class_type": "UNETLoader", "inputs": {"unet_name": ZIMAGE_UNET, "weight_dtype": "default"}},
        "zclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_TEXT, "type": "lumina2", "device": "default"}},
        "zvae": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "zms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("zunet"), "shift": 3.0}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("zclip"), "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("PU_Prompt")}},
        "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": _ref("zms"), "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "patch": {"class_type": "ModelPatchLoader", "inputs": {"name": patch}},
        "PU_Depth": {"class_type": "LoadImage", "inputs": {"image": control}},
        "PU_Control": {"class_type": "ZImageFunControlnet", "inputs": {
            "model": _ref("zms"), "model_patch": _ref("patch"), "vae": _ref("zvae"), "strength": strength, "image": _ref("PU_Depth"),
            "start_percent": 0.0, "end_percent": 1.0}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("PU_Control"), "positive": _ref("PU_Prompt"),
                                                         "negative": _ref("neg"), "cfg": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                     "sigmas": _ref("sigmas"), "latent_image": _ref("lat")}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": _ref("sample"), "vae": _ref("zvae")}},
        "PU_Plate": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/plate"}},
    }


def layout_render(layout: object, look: object, prompt: str, fps: float, width: int = 1920, height: int = 1088, seed: int = 42,
                  negative: str = "", strength: float = 1.0, models: Models = Models(), prefix: str = "pickups/layout",
                  keyframes: list[tuple[object, int, float]] | None = None) -> dict:
    """Previz -> render with Lightricks' Layout-To-Render IC-LoRA (rebuilt from their Two_Stage_Distilled workflow).

    `layout`: the raw playblast/viewport video (8k+1 frames). `look`: a "look still" made from the previz's first
    frame (palette, light, materials, character). The layout is an IC-LoRA guide at frame 0; the look still is a
    reference at frame -1 (outside the timeline), so it sets the look without being forced in as frame 1.
    Pass 1 at half size (8 distilled steps), then a 2x latent upsample and 3 refine steps with both guides re-added.
    width/height: multiples of 64 (1920x1088 is the known-good size). Video only, no audio.
    keyframes: optional (image, frame_idx, strength) look frames made from later playblast frames (e.g. the middle and
    the last), added with LTX's core keyframe guide so small or moving subjects keep their place and scale.
    """
    hw, hh = width // 2, height // 2
    g = _loaders(models) | _text(prompt, negative, fps)

    def guides(tag: str, pos: list, neg: list, latent: list, layout_img: list) -> dict:
        nodes = {
            f"{tag}_layout": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
                "positive": pos, "negative": neg, "vae": _ref("vae"), "latent": latent, "image": layout_img, "frame_idx": 0,
                "strength": strength, "latent_downscale_factor": _ref("lora", 1), "crop": "disabled", "use_tiled_encode": False,
                "tile_size": 256, "tile_overlap": 64}},
            f"{tag}_look": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
                "positive": _ref(f"{tag}_layout", 0), "negative": _ref(f"{tag}_layout", 1), "vae": _ref("vae"),
                "latent": _ref(f"{tag}_layout", 2), "image": _ref("PU_Look"), "frame_idx": -1, "strength": 1.0,
                "latent_downscale_factor": _ref("lora", 1), "crop": "center", "use_tiled_encode": False, "tile_size": 256,
                "tile_overlap": 64}},
        }
        last = f"{tag}_look"
        for i, (_img, idx, st) in enumerate(keyframes or []):
            nodes[f"{tag}_kf{i}"] = {"class_type": "LTXVAddGuide", "inputs": {
                "positive": _ref(last, 0), "negative": _ref(last, 1), "vae": _ref("vae"), "latent": _ref(last, 2),
                "image": _ref(f"{tag}_kfimg{i}"), "frame_idx": idx, "strength": st}}
            nodes[f"{tag}_kfimg{i}"] = {"class_type": "ImageScale", "inputs": {"image": _ref(f"PU_Key{i}"), "upscale_method": "lanczos",
                                                                              "width": hw if tag == "g1" else width,
                                                                              "height": hh if tag == "g1" else height, "crop": "center"}}
            last = f"{tag}_kf{i}"
        nodes[f"{tag}_out"] = last
        return nodes

    g |= {
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": LAYOUT_LORA, "strength_model": 1.0}},
        "PU_Layout": {"class_type": "LoadVideo", "inputs": {"file": layout}},
        "lparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Layout")}},
        "PU_Look": {"class_type": "LoadImage", "inputs": {"image": look}},
        "lay_half": {"class_type": "ImageScale", "inputs": {"image": _ref("lparts", 0), "upscale_method": "lanczos", "width": hw, "height": hh, "crop": "center"}},
        "lay_full": {"class_type": "ImageScale", "inputs": {"image": _ref("lparts", 0), "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}},
        "lsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("lay_half")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": hw, "height": hh, "length": _ref("lsize", 2), "batch_size": 1}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas1": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sigmas2": {"class_type": "ManualSigmas", "inputs": {"sigmas": DFR_SIGMAS}},
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
    }
    for i, (img, _idx, _st) in enumerate(keyframes or []):
        g[f"PU_Key{i}"] = {"class_type": "LoadImage", "inputs": {"image": img}}
    n1 = guides("g1", _ref("cond", 0), _ref("cond", 1), _ref("vlat"), _ref("lay_half"))
    o1 = n1.pop("g1_out")
    g |= n1
    g |= {
        "guider1": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref(o1, 0), "negative": _ref(o1, 1), "cfg": 1.0}},
        "sample1": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider1"), "sampler": _ref("sampler"),
                                                                       "sigmas": _ref("sigmas1"), "latent_image": _ref(o1, 2)}},
        "crop1": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref(o1, 0), "negative": _ref(o1, 1), "latent": _ref("sample1", 0)}},
        "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": _ref("crop1", 2), "upscale_model": _ref("upmodel"), "vae": _ref("vae")}},
    }
    n2 = guides("g2", _ref("crop1", 0), _ref("crop1", 1), _ref("up"), _ref("lay_full"))
    o2 = n2.pop("g2_out")
    g |= n2
    g |= {
        "guider2": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref(o2, 0), "negative": _ref(o2, 1), "cfg": 1.0}},
        "sample2": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider2"), "sampler": _ref("sampler"),
                                                                       "sigmas": _ref("sigmas2"), "latent_image": _ref(o2, 2)}},
        "crop2": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref(o2, 0), "negative": _ref(o2, 1), "latent": _ref("sample2", 0)}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop2", 2), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


CLEAN_PLATE_LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"


def clean_plate_video(clip: object, fps: float, width: int, height: int, prompt: str, negative: str = "", seed: int = 42,
                      models: Models = Models(), prefix: str = "pickups/cleanplate") -> dict:
    """LTX-2.5 Clean-Plate IC-LoRA (Lightricks): the clip with people and moving things gone and the empty scene rebuilt.

    No mask: the whole clip is the IC-LoRA reference (strength 1.0, one distilled pass of 8 steps, CFG 1, euler_ancestral).
    Trained at 1024x576 on 49 frames at 25 fps; width and height multiples of 32, 8k+1 frames. Name the thing in the
    prompt ("An empty road, no vehicle") and in the negative. Seed-dependent: the caller checks the thing is gone and
    retries. Output PU_Out (frames at width x height).
    """
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": CLEAN_PLATE_LORA, "strength_model": 1.0}},
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "ref": {"class_type": "ImageScale", "inputs": {"image": _ref("parts", 0), "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
        "rsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("ref")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": _ref("rsize", 2), "batch_size": 1}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("vlat"), "image": _ref("ref"),
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("lora", 1), "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas"), "latent_image": _ref("guide", 2)}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "latent": _ref("sample")}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop", 2), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


def refine_details(clip: object, prompt: str, fps: float, width: int, height: int, seed: int = 42, negative: str = "",
                   tile: tuple[int, int] = (1024, 576), models: Models = Models(), prefix: str = "pickups/refine",
                   tile_frames: int = 97) -> dict:
    """LTX-2.5 Refine-Details IC-LoRA: rebuild fine texture (skin, fabric, edges) on a soft clip, same framing and colour.

    The clip is resized (lanczos) to the canvas and used as a full-size IC-LoRA reference; LTXVTiledFusionSampler
    samples 1024x576 tiles (the LoRA's trained window) with 50% overlap, 8 distilled steps, CFG 1. Canvas width/height
    multiples of 32 (e.g. 1920x1088, or 3840x2176 for 4K). Video only. Frames are encoded and sampled in temporal windows of
    `tile_frames` (97, Lightricks' value), so a clip longer than one window is refined window by window.
    """
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": REFINE_LORA, "strength_model": 1.0}},
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "ref": {"class_type": "ImageScale", "inputs": {"image": _ref("parts", 0), "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
        "rsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("ref")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": _ref("rsize", 2), "batch_size": 1}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("vlat"), "image": _ref("ref"),
            # Lightricks: a spatially tiled guide encode imprints a grid into the conditioning, so it stays off; each
            # temporal window of tile_frames is encoded fresh instead (use_streaming, same value on the sampler).
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("lora", 1), "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "use_streaming": True, "tile_frames": tile_frames}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "PU_Seed": {"class_type": "LTXVTiledFusionSampler", "inputs": {
            "model": _ref("lora"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "latents": _ref("guide", 2),
            "sigmas": _ref("sigmas"), "sampler": _ref("sampler"), "seed": seed, "cfg": 1.0, "tile_width": tile[0], "tile_height": tile[1],
            "overlap_frac": 0.5, "blend_var": 0.05, "grid_cycle": 1, "tile_frames": tile_frames, "vae": _ref("vae")}},
        # The fusion sampler returns the canvas without the guide frames, so there is nothing to crop.
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


def depth_frames(clip: object, resolution: int = 518, prefix: str = "pickups/depth") -> dict:
    """Depth Anything 3 on every frame of `clip`: PU_Depth, raw relative depth as greyscale images (scaled per frame,
    so read them together), and PU_Sky, the model's sky probability."""
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "da3": {"class_type": "LoadDA3Model", "inputs": {"model_name": DA3_MODEL, "weight_dtype": "default"}},
        "geo": {"class_type": "DA3Inference", "inputs": {"da3_model": _ref("da3"), "image": _ref("parts", 0), "resolution": resolution,
                                                        "resize_method": "upper_bound_resize", "mode": "mono"}},
        "depth": {"class_type": "DA3Render", "inputs": {"da3_geometry": _ref("geo"), "output": "depth",
                                                       "output.normalization": "min_max", "output.apply_sky_clip": True}},
        "sky": {"class_type": "DA3Render", "inputs": {"da3_geometry": _ref("geo"), "output": "sky_mask"}},
        "PU_Depth": {"class_type": "SaveImage", "inputs": {"images": _ref("depth"), "filename_prefix": f"{prefix}/depth"}},
        "PU_Sky": {"class_type": "SaveImage", "inputs": {"images": _ref("sky"), "filename_prefix": f"{prefix}/sky"}},
    }


def select(clip: object, label: str, threshold: float = 0.5, prefix: str = "pickups/select") -> dict:
    """Find `label` on every frame with SAM 3.1 (text prompt). Outputs one white-on-black mask image per frame."""
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
        "PU_Label": {"class_type": "CLIPTextEncode", "inputs": {"text": counted(label), "clip": _ref("sam", 1)}},
        "detect": {
            "class_type": "SAM3_Detect",
            "inputs": {"model": _ref("sam"), "image": _ref("parts", 0), "threshold": threshold, "refine_iterations": 2,
                       "individual_masks": False, "conditioning": _ref("PU_Label")},
        },
        "maskimg": {"class_type": "MaskToImage", "inputs": {"mask": _ref("detect", 0)}},
        "PU_Mask": {"class_type": "SaveImage", "inputs": {"images": _ref("maskimg"), "filename_prefix": f"{prefix}/mask"}},
    }


MAX_PEOPLE = 8  # how many matches a text label may return


def counted(label: str, n: int = MAX_PEOPLE) -> str:
    """SAM 3's text encoder in ComfyUI keeps only the single best detection unless the prompt says how many
    ("person:8"; comfy/text_encoders/sam3_clip.py). A plain "person" finds one of four runners, even with individual
    masks and a low threshold. Add the count unless the label already has one."""
    label = label.strip()
    return label if not label or re.search(r":\d+$", label) else f"{label}:{n}"


def select_frame(frame: object, positive: list[dict], negative: list[dict] | None = None, text: str = "",
                 threshold: float = 0.4, prefix: str = "pickups/pick", individual: bool = False) -> dict:
    """Click-to-select on one frame: SAM 3.1 with point prompts (pixel coords) and/or a text prompt.

    Output PU_Mask: one white-on-black mask for the frame (the union of what was found), or with `individual` one mask
    per object found (e.g. each person of a couple, even when they touch).
    """
    detect = {"model": _ref("sam"), "image": _ref("PU_Frame"), "threshold": threshold, "refine_iterations": 2,
              "individual_masks": individual}
    if positive:
        detect["positive_coords"] = json.dumps([{"x": int(p["x"]), "y": int(p["y"])} for p in positive])
    if negative:
        detect["negative_coords"] = json.dumps([{"x": int(p["x"]), "y": int(p["y"])} for p in negative])
    g = {
        "PU_Frame": {"class_type": "LoadImage", "inputs": {"image": frame}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
        "detect": {"class_type": "SAM3_Detect", "inputs": detect},
        "maskimg": {"class_type": "MaskToImage", "inputs": {"mask": _ref("detect", 0)}},
        "PU_Mask": {"class_type": "SaveImage", "inputs": {"images": _ref("maskimg"), "filename_prefix": f"{prefix}/mask"}},
    }
    if text:
        g["PU_Label"] = {"class_type": "CLIPTextEncode", "inputs": {"text": counted(text), "clip": _ref("sam", 1)}}
        detect["conditioning"] = _ref("PU_Label")
    if individual:
        # Nobody found is an empty batch, which SaveImage can't save (the job fails as if SAM were down). One blank mask
        # after the found ones keeps the batch non-empty; every caller drops empty masks.
        g["frame_m"] = {"class_type": "ImageToMask", "inputs": {"image": _ref("PU_Frame"), "channel": "red"}}
        g["blank"] = {"class_type": "ThresholdMask", "inputs": {"mask": _ref("frame_m"), "value": 1.0}}
        g["blankimg"] = {"class_type": "MaskToImage", "inputs": {"mask": _ref("blank")}}
        g["found"] = {"class_type": "BatchImagesNode", "inputs": {"images.image0": _ref("maskimg"), "images.image1": _ref("blankimg")}}
        g["PU_Mask"]["inputs"]["images"] = _ref("found")
    return g


def track_mask(clip: object, first_mask: object | list, prefix: str = "pickups/track", each: bool = False) -> dict:
    """Follow first-frame mask(s) through the clip with SAM 3.1 video tracking. PU_Mask: one mask per frame (the union).

    `first_mask` is one mask image, or a list (one per object, e.g. each person of a couple): SAM 3.1 tracks each
    object separately, which holds separate people far better than one combined mask. each=True also saves every
    object's own masks (PU_Obj0, PU_Obj1, ...: Recast with several people swaps each one on their own)."""
    firsts = first_mask if isinstance(first_mask, list) else [first_mask]
    g = {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
    }
    for i, m in enumerate(firsts):
        g[f"PU_First{i}" if i else "PU_First"] = {"class_type": "LoadImage", "inputs": {"image": m}}
    if len(firsts) == 1:
        g["m0"] = {"class_type": "ImageToMask", "inputs": {"image": _ref("PU_First"), "channel": "red"}}
    else:
        g["firsts"] = {"class_type": "BatchImagesNode", "inputs": {
            f"images.image{i}": _ref(f"PU_First{i}" if i else "PU_First") for i in range(len(firsts))}}  # 0-based, as in Comfy's templates
        g["m0"] = {"class_type": "ImageToMask", "inputs": {"image": _ref("firsts"), "channel": "red"}}
    g |= {
        "track": {"class_type": "SAM3_VideoTrack", "inputs": {"images": _ref("parts", 0), "model": _ref("sam"), "detection_threshold": 0.5,
                                                             "max_objects": len(firsts), "detect_interval": 1, "initial_mask": _ref("m0")}},
        "masks": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": _ref("track"), "object_indices": "" if len(firsts) > 1 else "0"}},
        "maskimg": {"class_type": "MaskToImage", "inputs": {"mask": _ref("masks")}},
        "PU_Mask": {"class_type": "SaveImage", "inputs": {"images": _ref("maskimg"), "filename_prefix": f"{prefix}/mask"}},
    }
    if each:
        for i in range(len(firsts)):
            g[f"obj{i}"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": _ref("track"), "object_indices": str(i)}}
            g[f"objimg{i}"] = {"class_type": "MaskToImage", "inputs": {"mask": _ref(f"obj{i}")}}
            g[f"PU_Obj{i}"] = {"class_type": "SaveImage", "inputs": {"images": _ref(f"objimg{i}"), "filename_prefix": f"{prefix}/obj{i}/mask"}}
    return g


def remove(clip: object, label: str, fps: float, seed: int = 42, dilate: int = 15, threshold: float = 0.5,
           prompt: str = "Clean background, the scene continues naturally with nothing in that area.",
           negative: str = DEFAULT_NEGATIVE, models: Models = Models(), prefix: str = "pickups/remove",
           mask_clip: object | None = None, plate: object | None = None, plate_strength: float = 0.7) -> dict:
    """Inpaint only the masked area with LTX-2.5 + the In/Outpaint IC-LoRA.

    The mask comes from SAM 3.1 finding `label` on every frame, or from `mask_clip` (a white-on-black mask video
    the runner prepared, e.g. a box around the SAM mask). Single stage at the clip's own size (multiples of 32,
    8k+1 frames). Outputs the generated frames (PU_Out) and the mask used (PU_Mask).
    Ported from LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled, stage 1. Distilled CFG is 1, so `negative` has no
    effect; the fill is steered by `prompt` alone.
    """
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
    }
    if mask_clip is not None:
        g |= {
            "PU_MaskClip": {"class_type": "LoadVideo", "inputs": {"file": mask_clip}},
            "mparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_MaskClip")}},
            "rawmask": {"class_type": "ImageToMask", "inputs": {"image": _ref("mparts", 0), "channel": "red"}},
        }
        raw = _ref("rawmask")
    else:
        g |= {
            "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
            "PU_Label": {"class_type": "CLIPTextEncode", "inputs": {"text": counted(label), "clip": _ref("sam", 1)}},
            "detect": {
                "class_type": "SAM3_Detect",
                "inputs": {"model": _ref("sam"), "image": _ref("parts", 0), "threshold": threshold, "refine_iterations": 2,
                           "individual_masks": False, "conditioning": _ref("PU_Label")},
            },
        }
        raw = _ref("detect", 0)
    g |= {
        "dilate": {"class_type": "LTXVDilateVideoMask", "inputs": {"spatial_radius": dilate, "temporal_radius": 0, "mask": raw}},
        "PU_Mask": {"class_type": "SaveImage", "inputs": {"images": _ref("maskimg"), "filename_prefix": f"{prefix}/mask"}},
        "maskimg": {"class_type": "MaskToImage", "inputs": {"mask": _ref("dilate")}},
        # Fill: LTX-2.5 with the In/Outpaint IC-LoRA, guided by the source frames with the hole greyed out.
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": INPAINT_LORA, "strength_model": 1.0}},
        "holed": {"class_type": "LTXVInpaintPreprocess", "inputs": {"images": _ref("parts", 0), "mask": _ref("dilate")}},
        "size": {"class_type": "GetImageSize", "inputs": {"image": _ref("holed")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": _ref("size", 0), "height": _ref("size", 1), "length": _ref("size", 2), "batch_size": 1}},
    }
    start_latent = _ref("vlat")
    if plate is not None:
        # A clean first frame (e.g. from clean_plate) steers the whole fill toward "nothing there".
        g |= {
            "PU_Plate": {"class_type": "LoadImage", "inputs": {"image": plate}},
            "platepre": {"class_type": "LTXVPreprocess", "inputs": {"image": _ref("PU_Plate"), "img_compression": 18}},
            "i2v": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": _ref("vae"), "image": _ref("platepre"), "latent": _ref("vlat"),
                                                                     "strength": plate_strength, "bypass": False}},
        }
        start_latent = _ref("i2v")
    g |= {
        "guide": {
            "class_type": "LTXAddVideoICLoRAGuideAdvanced",
            "inputs": {"positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": start_latent,
                       "image": _ref("holed"), "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("lora", 1),
                       "crop": "disabled", "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "attention_strength": 1.0},
        },
        "aenc": {"class_type": "VAEEncodeAudio", "inputs": {"audio": _ref("parts", 1), "vae": _ref("avae")}},
        "areft": {"class_type": "LTXVSetAudioRefTokens", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "audio_latent": _ref("aenc")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("guide", 2), "audio_latent": _ref("areft", 2)}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref("areft", 0), "negative": _ref("areft", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"), "sigmas": _ref("sigmas"), "latent_image": _ref("av")}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample", 1)}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("areft", 0), "negative": _ref("areft", 1), "latent": _ref("split", 0)}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop", 2), "vae": _ref("vae"), "tile_size": 512, "overlap": 64, "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


LATENT_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


DETAIL_LORA = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
DEV_UNET = "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
DFR_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"


def retake_two_stage(canvas: object, fps: float, start_time: float, end_time: float, context_frames: int,
                     width: int, height: int, prompt: str, seed: int = 42, negative: str = DEFAULT_NEGATIVE,
                     models: Models = Models(), prefix: str = "pickups/retake2", detail: bool = False,
                     detail_strength: float = 1.0, dev_first_pass: bool = False, dev_steps: int = 30,
                     dev_cfg: float = 3.0) -> dict:
    """Two-pass retake, the way the official LTX-2.5 templates get detail.

    Pass 1 regenerates [start_time, end_time] at half size. Pass 2 upsamples that latent 2x, puts the real context
    frames back (encoded at full size), and refines only the window with the short stage-2 schedule. `canvas`
    must be at full working size (width x height, multiples of 64); `context_frames` must be 8k+1.
    """
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": canvas}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "half": {"class_type": "ImageScale", "inputs": {"image": _ref("parts", 0), "upscale_method": "lanczos",
                                                         "width": width // 2, "height": height // 2, "crop": "disabled"}},
        **({"dev": {"class_type": "UNETLoader", "inputs": {"unet_name": DEV_UNET, "weight_dtype": "default"}}} if dev_first_pass else {}),
        # Pass 1 at half size: same as retake().
        "venc1": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("half"), "vae": _ref("vae")}},
        "aenc": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": _ref("parts", 1), "audio_vae": _ref("avae")}},
        "av1": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("venc1"), "audio_latent": _ref("aenc")}},
        "win1": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {
            "av_latent": _ref("av1"), "positive": _ref("cond", 0), "negative": _ref("cond", 1), "model": _ref("unet"),
            "vae": _ref("vae"), "audio_vae": _ref("avae"), "start_time": start_time, "end_time": end_time, "video_fps": fps,
            "mask_video": True, "mask_audio": True, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "guider1": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": _ref("dev") if dev_first_pass else _ref("unet"),
                                                                   "positive": _ref("win1", 0), "negative": _ref("win1", 1),
                                                                   "video_cfg": dev_cfg if dev_first_pass else 1.0,
                                                                   "audio_cfg": 7.0 if dev_first_pass else 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas1": ({"class_type": "LTXVScheduler", "inputs": {"steps": dev_steps, "max_shift": 2.05, "base_shift": 0.95,
                                                                "stretch": True, "terminal": 0.1, "latent": _ref("win1", 2)}}
                    if dev_first_pass else {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}}),
        "sample1": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider1"), "sampler": _ref("sampler"),
                                                                       "sigmas": _ref("sigmas1"), "latent_image": _ref("win1", 2)}},
        "split1": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample1")}},
        # Pass 2: upsample, re-freeze the real context at full size, refine the window only.
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
        "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": _ref("split1", 0), "upscale_model": _ref("upmodel"), "vae": _ref("vae")}},
        "ctx": {"class_type": "ImageFromBatch", "inputs": {"image": _ref("parts", 0), "batch_index": 0, "length": context_frames}},
        "ctxenc": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("ctx"), "vae": _ref("vae")}},
        "refrozen": {"class_type": "ReplaceVideoLatentFrames", "inputs": {"destination": _ref("up"), "index": 0, "source": _ref("ctxenc")}},
    }
    pos, neg, vlat, model2 = _ref("cond", 0), _ref("cond", 1), _ref("refrozen"), _ref("unet")
    if detail:
        # Lightricks' DFR detailing: the pixel-spatial-upscaler IC-LoRA, with the half-size draft as an x2 reference.
        g |= {
            "detail_lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": DETAIL_LORA,
                                                                             "strength_model": detail_strength}},
            "dguide": {"class_type": "LTXVAddLatentGuide", "inputs": {"positive": pos, "negative": neg, "vae": _ref("vae"),
                                                                       "latent": vlat, "guiding_latent": _ref("split1", 0),
                                                                       "latent_idx": 0, "strength": 1.0}},
        }
        pos, neg, vlat, model2 = _ref("dguide", 0), _ref("dguide", 1), _ref("dguide", 2), _ref("detail_lora")
    g |= {
        "av2": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": vlat, "audio_latent": _ref("split1", 1)}},
        "PU_Window": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {
            "av_latent": _ref("av2"), "positive": pos, "negative": neg, "model": model2,
            "vae": _ref("vae"), "audio_vae": _ref("avae"), "start_time": start_time, "end_time": end_time, "video_fps": fps,
            "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "guider2": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": model2, "positive": _ref("PU_Window", 0), "negative": _ref("PU_Window", 1),
                                                                   "video_cfg": 1.0, "audio_cfg": 1.0}},
        "seed2": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}},
        "sigmas2": {"class_type": "ManualSigmas", "inputs": {"sigmas": DFR_SIGMAS if detail else STAGE2_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("seed2"), "guider": _ref("guider2"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas2"), "latent_image": _ref("PU_Window", 2)}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample")}},
    }
    final_video = _ref("split", 0)
    if detail:
        g["uncrop"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("PU_Window", 0), "negative": _ref("PU_Window", 1),
                                                                  "latent": _ref("split", 0)}}
        final_video = _ref("uncrop", 2)
    g |= {
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": final_video, "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "adecode": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": _ref("split", 1), "audio_vae": _ref("avae")}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
        "PU_OutAudio": {"class_type": "SaveAudio", "inputs": {"audio": _ref("adecode"), "filename_prefix": f"{prefix}/audio"}},
    }
    return g


def seedvr2_upscale(clip: object, scale: float = 2.0, seed: int = 42, prefix: str = "pickups/seedvr2") -> dict:
    """SeedVR2 restoration/upscale of a short clip (FL-SeedVR2 pack). Used as an optional finishing pass on AI frames."""
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "svr": {"class_type": "FLSeedVR2ModelLoader", "inputs": {"download_if_missing": False}},
        "PU_Upscale": {"class_type": "FLSeedVR2Upscale", "inputs": {"model": _ref("svr", 0), "vae": _ref("svr", 1), "images": _ref("parts", 0),
                                                                   "scale_multiplier": scale, "seed": seed, "color_correction": "lab",
                                                                   "vae_tile_size": "512"}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("PU_Upscale"), "filename_prefix": f"{prefix}/frame"}},
    }


SEEDVR2_NATIVE = "seedvr2_distill_6L_1.4B_sharp_fp16_comfyui.safetensors"
SEEDVR2_VAE = "seedvr2_ema_vae_fp16.safetensors"


def seedvr2_video(clip: object, scale: float = 2.0, seed: int = 42, model: str = SEEDVR2_NATIVE, overlap: int = 2,
                  color: str = "lab", prefix: str = "pickups/seedvr2v") -> dict:
    """SeedVR2 with ComfyUI's native nodes (from Comfy's utility_seedvr2_3b_int8_upscale_video template).

    Unlike the FL pack, the clip is encoded as one video latent and sampled in overlapping temporal chunks that are
    crossfaded on merge, so frames are restored together (less flicker). One step, CFG 1.
    """
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "svr": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "svae": {"class_type": "VAELoader", "inputs": {"vae_name": SEEDVR2_VAE}},
        "resized": {"class_type": "ResizeImageMaskNode", "inputs": {"input": _ref("parts", 0), "resize_type": "scale by multiplier",
                                                                     "resize_type.multiplier": scale, "scale_method": "lanczos"}},
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": _ref("resized")}},
        "enc": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": _ref("pre"), "vae": _ref("svae"), "tile_size": 512, "overlap": 128,
                                                           "temporal_size": 64, "temporal_overlap": 8}},
        "chunk": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": _ref("enc"), "temporal_overlap": overlap, "chunking_mode": "auto"}},
        "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": _ref("svr"), "vae_conditioning": _ref("chunk", 0)}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("svr"), "seed": seed, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
                                                         "scheduler": "simple", "positive": _ref("cond", 0), "negative": _ref("cond", 1),
                                                         "latent_image": _ref("chunk", 0), "denoise": 1.0}},
        "merge": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": _ref("PU_Seed"), "temporal_overlap": _ref("chunk", 1)}},
        "dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("merge"), "vae": _ref("svae"), "tile_size": 512, "overlap": 128,
                                                           "temporal_size": 64, "temporal_overlap": 8}},
        "post": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": _ref("dec"), "original_resized_images": _ref("resized"),
                                                                   "color_correction_method": color}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("post"), "filename_prefix": f"{prefix}/frame"}},
    }


def gan_upscale(clip: object, width: int, height: int, model: str = "RealESRGAN_x4plus.safetensors", prefix: str = "pickups/gan") -> dict:
    """Classic GAN upscale (Real-ESRGAN x4), then resize to the target. Fast, invents nothing, can look over-crisp."""
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "gan": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "up": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": _ref("gan"), "image": _ref("parts", 0)}},
        "fit": {"class_type": "ImageScale", "inputs": {"image": _ref("up"), "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("fit"), "filename_prefix": f"{prefix}/frame"}},
    }


def rtx_upscale(clip: object, scale: float = 2.0, quality: str = "HIGHBITRATE_ULTRA", prefix: str = "pickups/rtxvsr") -> dict:
    """NVIDIA RTX Video Super Resolution (Comfy-Org's Nvidia VFX nodes; NVIDIA's proprietary library, RTX GPUs only).

    Very fast, no diffusion: a preview-grade 4K. HIGHBITRATE_* modes are for clean camera sources.
    """
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "vsr": {"class_type": "RTXVideoSuperResolution", "inputs": {"images": _ref("parts", 0), "resize_type": "scale by multiplier",
                                                                   "resize_type.scale": scale, "quality": quality, "strength": 1.0,
                                                                   "image_encoding": "8-bit RGB"}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("vsr"), "filename_prefix": f"{prefix}/frame"}},
    }


def rtx_hdr(clip: object, fps: float, peak_nits: int = 1000, prefix: str = "pickups/rtxhdr") -> dict:
    """NVIDIA RTX Video HDR: SDR -> BT.2020 HLG. Saves an HLG master (10-bit) and scene-linear EXRs."""
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "hdr": {"class_type": "RTXTrueHDR", "inputs": {"images": _ref("parts", 0), "contrast": 100, "saturation": 100, "middle_gray": 50,
                                                       "luminance": peak_nits, "debanding": True, "output_colorspace": "HDR"}},
        "vid": {"class_type": "CreateVideo", "inputs": {"images": _ref("hdr"), "fps": fps, "bit_depth": 10, "color_space": "HDR"}},
        "PU_HLG": {"class_type": "SaveVideo", "inputs": {"video": _ref("vid"), "filename_prefix": f"{prefix}/hlg", "format": "mp4",
                                                         "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                         "format.codec.encoding.crf": 10.0}},
        "PU_EXR": {"class_type": "SaveImageAdvanced", "inputs": {"images": _ref("hdr"), "filename_prefix": f"{prefix}/exr/frame", "format": "exr",
                                                                 "format.bit_depth": "16-bit float", "format.input_color_space": "HDR"}},
    }


HDR_LORA = "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"
HDR_SCENE_EMB = "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"


def sdr_to_hdr(clip: object, fps: float, seed: int = 42, review_frame: int = 0, exposures: tuple = (-2.0, -4.0),
               exr: bool = False, models: Models = Models(), prefix: str = "pickups/hdr") -> dict:
    """LTX-2.5 SDR->HDR IC-LoRA, rebuilt from Lightricks' LTX-2.5_ICLoRA_SDR_to_HDR_Distilled graph.

    The whole SDR clip (Rec.709 sRGB-gamma, multiples of 32, 8k+1 frames) is a frame-aligned IC-LoRA reference in
    ACEScct; one 8-step distilled pass with fixed scene embeddings (no prompt). Outputs: PU_HLG (BT.2020 HLG
    10-bit HEVC master via LTXVSaveHLG), PU_Preview (tonemapped frames) and PU_EV_<n> (one frame tonemapped with an
    exposure pull, to show highlight range on SDR screens).
    """
    cond_fps = min(fps, 30.0)
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": models.unet, "weight_dtype": "default"}},
        "hdrlora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": HDR_LORA, "strength_model": 1.0}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": models.video_vae}},
        "vae32": {"class_type": "LTXVVAEForceFloat32", "inputs": {"vae": _ref("vae")}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": models.audio_vae}},
        "emb": {"class_type": "LTXVLoadConditioning", "inputs": {"file_name": HDR_SCENE_EMB, "device": "cpu"}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": _ref("emb"), "negative": _ref("emb"), "frame_rate": cond_fps}},
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "acescct": {"class_type": "LTXVSDRToHDRWorkingSpace", "inputs": {"image": _ref("parts", 0), "color_space": "srgb_gamma"}},
        "size": {"class_type": "GetImageSize", "inputs": {"image": _ref("acescct")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": _ref("size", 0), "height": _ref("size", 1), "length": _ref("size", 2), "batch_size": 1}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae32"), "latent": _ref("vlat"),
            "image": _ref("acescct"), "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("hdrlora", 1),
            "crop": "disabled", "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64,
            "use_streaming": False, "tile_frames": 97}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": _ref("size", 2), "frame_rate": fps, "batch_size": 1, "audio_vae": _ref("avae")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("guide", 2), "audio_latent": _ref("alat")}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("hdrlora"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas"), "latent_image": _ref("av")}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample", 0)}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "latent": _ref("split", 0)}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop", 2), "vae": _ref("vae32"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "post": {"class_type": "LTXVHDRDecodePostprocess", "inputs": {"image": _ref("decode"), "transfer": "acescct", "exposure": 0.0,
                                                                       "save_exr": False, "exr_color_space": "acescg",
                                                                       "output_dir": "output/hdr_exr", "filename_prefix": "frame", "half_precision": True}},
        # HDR master through core nodes so the files are registered job outputs (LTXVSaveHLG writes but doesn't register).
        "to_hlg": {"class_type": "ImageColorSpace", "inputs": {"image": _ref("decode"), "source": "HDR ACEScct", "destination": "HDR"}},
        "hlgvid": {"class_type": "CreateVideo", "inputs": {"images": _ref("to_hlg"), "fps": fps, "bit_depth": 10, "color_space": "HDR",
                                                           "codec": "none"}},
        "PU_HLG": {"class_type": "SaveVideo", "inputs": {"video": _ref("hlgvid"), "filename_prefix": f"{prefix}/hlg",
                                                        "format": "mp4", "format.codec": "h264",
                                                        "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 10.0}},
        "PU_Preview": {"class_type": "SaveImage", "inputs": {"images": _ref("post", 0), "filename_prefix": f"{prefix}/preview"}},
        "evframe": {"class_type": "ImageFromBatch", "inputs": {"image": _ref("decode"), "batch_index": review_frame, "length": 1}},
    }
    if exr:
        # Unclamped scene-linear ACEScg from Lightricks' postprocess (core ImageColorSpace clamps at 1.0). Written as-is;
        # the runner records the primaries (ACEScg) in the sidecar so Resolve/ACES reads them correctly.
        g["PU_EXR"] = {"class_type": "SaveImageAdvanced", "inputs": {"images": _ref("post", 1), "filename_prefix": f"{prefix}/exr/frame",
                                                                     "format": "exr", "format.bit_depth": "16-bit float",
                                                                     "format.input_color_space": "linear"}}
    for ev in exposures:
        tag = f"{ev:+.0f}".replace("+", "p").replace("-", "m")
        g[f"ev_{tag}"] = {"class_type": "LTXVHDRDecodePostprocess", "inputs": {"image": _ref("evframe"), "transfer": "acescct", "exposure": ev,
                                                                               "save_exr": False, "exr_color_space": "acescg",
                                                                               "output_dir": "output/hdr_exr", "filename_prefix": "ev", "half_precision": True}}
        g[f"PU_EV_{tag}"] = {"class_type": "SaveImage", "inputs": {"images": _ref(f"ev_{tag}", 0), "filename_prefix": f"{prefix}/ev_{tag}"}}
    return g


FLUX_KLEIN = "flux3_image_klein_sd_bf16.safetensors"
FLUX_TEXT = "qwen3vl_4b_bf16.safetensors"  # Qwen3-VL 4B, the bf16 file in BFL's flux-3-image-klein-comfy repo
FLUX_VAE = "flux2-vae.safetensors"


def still(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42, steps: int = 15,
          prefix: str = "pickups/still") -> dict:
    """FLUX 3 klein text-to-image (open weights). Used for reference characters, keyframes and plates.

    Comfy's image_flux3_klein_distilled_t2i without its optional prompt upsampler: 15 steps, CFG 1, euler / simple,
    empty negative, EmptyFlux2LatentImage. The bf16 weights stand in for the template's int8 ones (same model).
    Also the generate builder for images.py's "flux3-klein".
    """
    return {
        "funet": {"class_type": "UNETLoader", "inputs": {"unet_name": FLUX_KLEIN, "weight_dtype": "default"}},
        "fclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX_TEXT, "type": "flux3", "device": "default"}},
        "fvae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("fclip"), "text": prompt}},
        "fneg": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("fclip"), "text": ""}},
        "flat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "PU_Seed": {"class_type": "KSampler", "inputs": {"model": _ref("funet"), "positive": _ref("PU_Prompt"), "negative": _ref("fneg"),
                                                       "latent_image": _ref("flat"), "seed": seed, "steps": steps, "cfg": 1.0,
                                                       "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "fdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed"), "vae": _ref("fvae")}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("fdec"), "filename_prefix": f"{prefix}/still"}},
    }


SCAIL_UNET = "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"
SCAIL_DPO = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
WAN_LIGHTX2V = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
WAN_TEXT = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_CLIP_VISION = "clip_vision_h.safetensors"
WAN_VAE = "Wan2_1_VAE_bf16.safetensors"


def scail2(drive: object, reference: object, prompt: str, replacement: bool = True, person: str = "human",
           ref_person: str = "human", object_indices: str = "", seed: int = 42, steps: int = 6,
           pose_strength: float = 1.0, prefix: str = "pickups/scail2", reference_mask: object | None = None) -> dict:
    """SCAIL-2 (Wan 2.1 14B, MIT) from Comfy's video_wan21_scail2_character_replacement_int8 template, first segment.

    replacement=True: swap the tracked person in `drive` for the `reference` character, keeping camera and background
    (Swap). replacement=False: the reference character performs the drive video's motion (animation mode).
    `drive` must already be at the working size (multiples of 32) and 4k+1 frames, at most tasks.common.MAX_FRAMES (97
    rendered fine at 832x640 on the 4090). Output PU_Out frames.

    `reference_mask` (white-on-black image of the character on `reference`) replaces SAM's `ref_person` track of the
    reference: SCAIL2ColoredMask takes a plain MASK as one identity, so a robot or a costume SAM wouldn't call "human"
    still gets its mask. `object_indices` picks drive people (left to right); it filters SAM reference tracks too, so
    pass a reference_mask with it.
    """
    g = {
        "PU_Drive": {"class_type": "LoadVideo", "inputs": {"file": drive}},
        "dparts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Drive")}},
        "dsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("dparts", 0)}},
        "PU_Reference": {"class_type": "LoadImage", "inputs": {"image": reference}},
        "wunet": {"class_type": "UNETLoader", "inputs": {"unet_name": SCAIL_UNET, "weight_dtype": "default"}},
        "dpo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("wunet"), "lora_name": SCAIL_DPO, "strength_model": 1.0}},
        "distill": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("dpo"), "lora_name": WAN_LIGHTX2V, "strength_model": 0.8}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": _ref("distill"), "shift": 5.0}},
        "wclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TEXT, "type": "wan", "device": "default"}},
        "wvae": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "cvis": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": WAN_CLIP_VISION}},
        "cvenc": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": _ref("cvis"), "image": _ref("PU_Reference"), "crop": "none"}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("wclip"), "text": prompt}},
        "wneg": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("wclip"), "text": ""}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
        "dlabel": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("sam", 1), "text": person}},
        "rlabel": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("sam", 1), "text": ref_person}},
        "dtrack": {"class_type": "SAM3_VideoTrack", "inputs": {"images": _ref("dparts", 0), "model": _ref("sam"), "detection_threshold": 0.5,
                                                               "max_objects": 4, "detect_interval": 1, "conditioning": _ref("dlabel")}},
        "rtrack": {"class_type": "SAM3_VideoTrack", "inputs": {"images": _ref("PU_Reference"), "model": _ref("sam"), "detection_threshold": 0.5,
                                                               "max_objects": 4, "detect_interval": 1, "conditioning": _ref("rlabel")}},
        "cmask": {"class_type": "SCAIL2ColoredMask", "inputs": {"driving_track_data": _ref("dtrack"), "ref_track_data": _ref("rtrack"),
                                                                "object_indices": object_indices, "sort_by": "left_to_right",
                                                                "replacement_mode": replacement}},
        "scail": {"class_type": "WanSCAILToVideo", "inputs": {
            "positive": _ref("PU_Prompt"), "negative": _ref("wneg"), "vae": _ref("wvae"),
            "width": _ref("dsize", 0), "height": _ref("dsize", 1), "length": _ref("dsize", 2), "batch_size": 1,
            "pose_strength": pose_strength, "pose_start": 0.0, "pose_end": 1.0, "video_frame_offset": 0, "previous_frame_count": 5,
            "pose_video": _ref("dparts", 0), "pose_video_mask": _ref("cmask", 0), "replacement_mode": replacement,
            "reference_image": _ref("PU_Reference"), "reference_image_mask": _ref("cmask", 1), "clip_vision_output": _ref("cvenc")}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": _ref("distill"), "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "wsampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "PU_Seed": {"class_type": "SamplerCustom", "inputs": {"model": _ref("shift"), "add_noise": True, "noise_seed": seed, "cfg": 1.0,
                                                            "positive": _ref("scail", 0), "negative": _ref("scail", 1),
                                                            "sampler": _ref("wsampler"), "sigmas": _ref("sched"), "latent_image": _ref("scail", 2)}},
        "wdec": {"class_type": "VAEDecode", "inputs": {"samples": _ref("PU_Seed", 1), "vae": _ref("wvae")}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("wdec"), "filename_prefix": f"{prefix}/frame"}},
    }
    if reference_mask is not None:
        del g["rlabel"], g["rtrack"]
        g["PU_RefMask"] = {"class_type": "LoadImage", "inputs": {"image": reference_mask}}
        g["rmask"] = {"class_type": "ImageToMask", "inputs": {"image": _ref("PU_RefMask"), "channel": "red"}}
        g["cmask"]["inputs"]["ref_track_data"] = _ref("rmask")
    return g


def select_any(clip: object, labels: list[str] | tuple[str, ...], threshold: float = 0.4, prefix: str = "pickups/select") -> dict:
    """SAM 3.1 for several labels on every frame, in one job: PU_Mask for the first label, PU_Mask1, PU_Mask2, ... for
    the rest (one white-on-black mask per frame each). Recast finds the new character with it whatever it is: a robot
    or an inflatable T-rex may not answer to "person"."""
    g = {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
    }
    for j, label in enumerate(labels):
        k = str(j) if j else ""
        g |= {
            f"PU_Label{k}": {"class_type": "CLIPTextEncode", "inputs": {"text": counted(label), "clip": _ref("sam", 1)}},
            f"detect{k}": {"class_type": "SAM3_Detect", "inputs": {"model": _ref("sam"), "image": _ref("parts", 0), "threshold": threshold,
                                                                  "refine_iterations": 2, "individual_masks": False,
                                                                  "conditioning": _ref(f"PU_Label{k}")}},
            f"maskimg{k}": {"class_type": "MaskToImage", "inputs": {"mask": _ref(f"detect{k}", 0)}},
            f"PU_Mask{k}": {"class_type": "SaveImage", "inputs": {"images": _ref(f"maskimg{k}"), "filename_prefix": f"{prefix}/l{j}/mask"}},
        }
    return g


def retake(canvas: object, fps: float, start_time: float, end_time: float, prompt: str, seed: int = 42,
           regenerate_audio: bool = True, negative: str = DEFAULT_NEGATIVE, models: Models = Models(),
           prefix: str = "pickups/retake") -> dict:
    """Regenerate only [start_time, end_time] of a padded canvas clip; everything else is frozen.

    `canvas` is the LoadVideo file input: a filename on the ComfyUI host, or an SDK asset handle.
    Extend uses start_time = context_frames / fps and end_time = the canvas length.
    """
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": canvas}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "venc": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("parts", 0), "vae": _ref("vae")}},
        "aenc": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": _ref("parts", 1), "audio_vae": _ref("avae")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("venc"), "audio_latent": _ref("aenc")}},
        "PU_Window": {
            "class_type": "LTXVSetAudioVideoMaskByTime",
            "inputs": {
                "av_latent": _ref("av"),
                "positive": _ref("cond", 0),
                "negative": _ref("cond", 1),
                "model": _ref("unet"),
                "vae": _ref("vae"),
                "audio_vae": _ref("avae"),
                "start_time": start_time,
                "end_time": end_time,
                "video_fps": fps,
                "mask_video": True,
                "mask_audio": regenerate_audio,
                "mask_init_value_video": 0.0,
                "mask_init_value_audio": 0.0,
                "slope_len": 3,
            },
        },
    }
    g |= _sample_and_save(_ref("PU_Window", 2), _ref("PU_Window", 0), _ref("PU_Window", 1), seed, prefix)
    return g


EDIT_ANYTHING_MOTION_LORA = "edit_anything_30k_v0.1_motion_transfer_r128.safetensors"  # Apache-2.0, LTX-2.5 compatible
CLEAN_PLATE_LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
ALPHA_GEN_LORA = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
CLEAN_PLATE_SIZE = (1024, 576)  # the IC-LoRA's trained size: 4 of 5 seeds clean here, 1 of 3 at 1280x704


DEV_UNET = "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"


def edit_anything_motion(clip: object, ref: object, prompt: str, fps: float, seed: int = 42, negative: str = "",
                         models: Models = Models(), prefix: str = "pickups/edit_anything", tile: int = 80,
                         dev: bool = False) -> dict:
    """EditAnything v0.1 motion transfer on LTX-2.5: the clip's motion drives the picture `ref` (the edited first frame).
    `clip` and `ref` have the same size (multiples of 32; 1280x704 works), the clip 8k+1 frames with no cuts. Output PU_Out: one
    image per frame. The whole frame is re-rendered (the background drifts and softens), so the result is only ever used under a
    matte over the real footage.

    Distilled (default): 8 steps, CFG 1: fast, and the new car holds on a steady shot. `dev=True`: the dev model, 30 steps, CFG 4:
    about 3x slower, and the prompt keeps the new car through a turn, where the distilled model drifted back to the old car's shape
    within 25 frames of the painted frame (the canyon bend, camera and car both turning). The looping sampler
    (ComfyUI-BFSNodes, GPL-3.0, installed by the user: not bundled) tiles time `tile` frames wide with 24 overlap: its default 80
    cross-fades two renders of the car over frames 56-80 (a soft, doubled car), so pass the clip's length to render in one tile."""
    if dev:
        models = Models(unet=DEV_UNET, text_encoder=models.text_encoder, video_vae=models.video_vae, audio_vae=models.audio_vae)
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "size": {"class_type": "GetImageSize", "inputs": {"image": _ref("parts", 0)}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": _ref("size", 0), "height": _ref("size", 1),
                                                                   "length": _ref("size", 2), "batch_size": 1}},
        "PU_Ref": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": EDIT_ANYTHING_MOTION_LORA,
                                                                  "strength_model": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if dev else "euler_ancestral"}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref("cond", 0), "negative": _ref("cond", 1),
                                                         "cfg": 4.0 if dev else 1.0}},
        "sigmas": ({"class_type": "LTXVScheduler", "inputs": {"steps": 30, "max_shift": 2.05, "base_shift": 0.95, "stretch": True,
                                                              "terminal": 0.1, "latent": _ref("vlat")}} if dev else
                   {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}}),
        "loop": {"class_type": "LTXVEditAnythingLoopingSampler", "inputs": {
            "model": _ref("lora"), "vae": _ref("vae"), "noise": _ref("PU_Seed"), "sampler": _ref("sampler"), "sigmas": _ref("sigmas"),
            "guider": _ref("guider"), "positive": _ref("cond", 0), "negative": _ref("cond", 1), "latents": _ref("vlat"),
            "temporal_tile_size": tile, "temporal_overlap": 24, "blend_overlap": True, "ref_image": _ref("PU_Ref"),
            "guide_frames": _ref("parts", 0), "ref_resize_mode": "pad_to_fit", "enable_role_embedding": False,
            "enable_adaln": False, "enable_visual_crossattn": False}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("loop", 0), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": 128, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


def ic_lora_video(clip: object, lora: str, prompt: str, fps: float, width: int, height: int, seed: int = 42, negative: str = "",
                  models: Models = Models(), prefix: str = "pickups/iclora") -> dict:
    """An official LTX-2.5 video-to-video IC-LoRA (Clean-Plate, Alpha-Gen): the clip, scaled to width x height (multiples of 32),
    is the reference at strength 1; distilled 8 steps, CFG 1, one pass. Output PU_Out: one image per frame."""
    g = _loaders(models) | _text(prompt, negative, fps)
    g |= {
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": lora, "strength_model": 1.0}},
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "ref": {"class_type": "ImageScale", "inputs": {"image": _ref("parts", 0), "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
        "rsize": {"class_type": "GetImageSize", "inputs": {"image": _ref("ref")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": _ref("rsize", 2), "batch_size": 1}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("vlat"), "image": _ref("ref"),
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("lora", 1), "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas"), "latent_image": _ref("guide", 2)}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "latent": _ref("sample")}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("crop", 2), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                               "temporal_size": 128, "temporal_overlap": 32}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }
    return g


def clean_plate_road(clip: object, fps: float, seed: int = 42, size: tuple[int, int] = CLEAN_PLATE_SIZE,
                     models: Models = Models(), prefix: str = "pickups/clean_plate") -> dict:
    """LTX-2.5 Clean-Plate IC-LoRA for a car swap's road: the clip with its subject taken out and the ground and sky behind it drawn in
    (1024x576, the trained size; seed-dependent: the caller checks the subject is gone and retries)."""
    # The prompt it was tested with (4 of 5 seeds clean on the rural-road shot): "empty road, no vehicle", negative "car".
    return ic_lora_video(clip, CLEAN_PLATE_LORA, "empty road, no vehicle", fps, size[0], size[1], seed=seed, negative="car",
                         models=models, prefix=prefix)


def alpha_gen_video(clip: object, fps: float, width: int, height: int, seed: int = 42, models: Models = Models(),
                    prefix: str = "pickups/alpha_gen") -> dict:
    """LTX-2.5 Alpha-Gen IC-LoRA (empty prompt): a grey alpha video of the clip's foreground subject."""
    return ic_lora_video(clip, ALPHA_GEN_LORA, "", fps, width, height, seed=seed, models=models, prefix=prefix)


# Relight any scene: LTX-2.3 with Lightricks' Relight IC-LoRA, exactly as tested on the 4090 (Oct 22, 20 runs): the
# distilled 1.1 fp8 transformer, Gemma 3 12B fp8 with the LTX-2.3 text projection, 1280x704, 8 distilled steps with
# euler_ancestral_cfg_pp, CFG 1, conditioning at 24 fps. The light-direction ball is baked into the clip beforehand.
RELIGHT_LORA = "ltx-2.3-22b-ic-lora-relight-1.0.safetensors"
RELIGHT_SIZE = (1280, 704)
RELIGHT_MODELS = {"unet": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
                  "gemma": "gemma_3_12B_it_fp8_scaled.safetensors", "projection": "ltx-2.3_text_projection_bf16.safetensors",
                  "vae": "LTX23_video_vae_bf16.safetensors", "audio_vae": "LTX23_audio_vae_bf16.safetensors"}
RELIGHT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, deformed\n"


def scene_relight(clip: object, prompt: str, frames: int, fps: float = 24.0, seed: int = 42, prefix: str = "pickups/relight_scene") -> dict:
    """The clip (1280x704, ball baked in, `frames` = 8k+1) relit to match its ball and the prompt. Output PU_Out: one image
    per frame at 1280x704."""
    m = RELIGHT_MODELS
    w, h = RELIGHT_SIZE
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": m["gemma"], "clip_name2": m["projection"], "type": "ltxv", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "lora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": _ref("unet"), "lora_name": RELIGHT_LORA, "strength_model": 1.0}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": _ref("clip")}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": RELIGHT_NEGATIVE, "clip": _ref("clip")}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": _ref("PU_Prompt"), "negative": _ref("neg"), "frame_rate": fps}},
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": _ref("PU_Clip")}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1}},
        "guide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": _ref("cond", 0), "negative": _ref("cond", 1), "vae": _ref("vae"), "latent": _ref("vlat"), "image": _ref("parts", 0),
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": _ref("lora", 1), "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": _ref("avae")}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": _ref("guide", 2), "audio_latent": _ref("alat")}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": _ref("lora"), "positive": _ref("guide", 0), "negative": _ref("guide", 1), "cfg": 1.0}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider"), "sampler": _ref("sampler"),
                                                                      "sigmas": _ref("sigmas"), "latent_image": _ref("av")}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": _ref("sample")}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": _ref("guide", 0), "negative": _ref("guide", 1), "latent": _ref("split", 0)}},
        "decode": {"class_type": "LTXVTiledVAEDecode", "inputs": {"vae": _ref("vae"), "latents": _ref("crop", 2), "horizontal_tiles": 2, "vertical_tiles": 2,
                                                                   "overlap": 6, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "PU_Out": {"class_type": "SaveImage", "inputs": {"images": _ref("decode"), "filename_prefix": f"{prefix}/frame"}},
    }


# --- New video: text to video and image to video (LTX-2.5) -------------------------------------------------------------
# The previz_to_final pipeline without the IC-LoRA, keyframes or previz driver: 8 distilled steps at half size, the x2
# latent upscaler, 3 refine steps at full size, video only. Parameter nodes are PU_*: PU_Prompt.text, PU_Seed.noise_seed,
# PU_Latent.width/height (HALF the final size) and .length (8n+1), PU_Video.fps with cond.frame_rate, PU_Save, and for
# image to video PU_Image (LoadImage) feeding PU_FirstHalf / PU_FirstFull (ImageScale, crop center: the picture covers the
# frame and the overflow is trimmed, so any picture works at any aspect). Sizes and lengths: pickups/video.py.
VIDEO_COMPRESSION = 18  # LTXVPreprocess img_compression on the start picture (the app's plate value)


def ltx_text_to_video(prompt: str, width: int, height: int, length: int, seed: int = 42, fps: float = 24.0,
                      prefix: str = "pickups/video", negative: str = DEFAULT_NEGATIVE) -> dict:
    """LTX-2.5 text to video: `width` x `height` (multiples of 64), `length` frames (8n+1, at most 97), H.264 mp4 at PU_Save."""
    if width % 64 or height % 64 or width <= 0 or height <= 0:
        raise ValueError(f"{width}x{height}: width and height must be multiples of 64 (pass 1 runs at half size).")
    if length < 9 or length > 97 or (length - 1) % 8:
        raise ValueError(f"{length} frames: the length must be 8n+1, from 9 to 97.")
    fps = float(fps)
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "ltxv", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "PU_Prompt": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": _ref("clip")}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": _ref("clip")}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": _ref("PU_Prompt"), "negative": _ref("neg"), "frame_rate": fps}},
        "PU_Latent": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width // 2, "height": height // 2, "length": length, "batch_size": 1}},
        "PU_Seed": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sigmas1": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "guider1": {"class_type": "CFGGuider", "inputs": {"model": _ref("unet"), "positive": _ref("cond", 0), "negative": _ref("cond", 1), "cfg": 1.0}},
        "sample1": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider1"), "sampler": _ref("sampler"),
                                                                       "sigmas": _ref("sigmas1"), "latent_image": _ref("PU_Latent")}},
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
        "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": _ref("sample1", 0), "upscale_model": _ref("upmodel"), "vae": _ref("vae")}},
        "sigmas2": {"class_type": "ManualSigmas", "inputs": {"sigmas": DFR_SIGMAS}},
        "guider2": {"class_type": "CFGGuider", "inputs": {"model": _ref("unet"), "positive": _ref("cond", 0), "negative": _ref("cond", 1), "cfg": 1.0}},
        "sample2": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": _ref("PU_Seed"), "guider": _ref("guider2"), "sampler": _ref("sampler"),
                                                                       "sigmas": _ref("sigmas2"), "latent_image": _ref("up")}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _ref("sample2", 0), "vae": _ref("vae"), "tile_size": 512, "overlap": 64,
                                                              "temporal_size": DECODE_FRAMES, "temporal_overlap": 32}},
        "PU_Video": {"class_type": "CreateVideo", "inputs": {"images": _ref("decode"), "fps": fps}},
        "PU_Save": {"class_type": "SaveVideo", "inputs": {"video": _ref("PU_Video"), "filename_prefix": f"{prefix}/video", "format": "mp4",
                                                         "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                         "format.codec.encoding.crf": 18.0}},
    }


def ltx_image_to_video(prompt: str, image: object, width: int, height: int, length: int, seed: int = 42, fps: float = 24.0,
                       prefix: str = "pickups/video", negative: str = DEFAULT_NEGATIVE, img_compression: int = VIDEO_COMPRESSION) -> dict:
    """LTX-2.5 image to video: the text graph with `image` (a LoadImage input) as frame 0, cover-cropped to the size (ImageScale
    crop center), at half size into the empty latent before pass 1 and at full size into the upscaled latent before pass 2."""
    g = ltx_text_to_video(prompt, width, height, length, seed=seed, fps=fps, prefix=prefix, negative=negative)
    g |= {
        "PU_Image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "PU_FirstHalf": {"class_type": "ImageScale", "inputs": {"image": _ref("PU_Image"), "upscale_method": "lanczos",
                                                                "width": width // 2, "height": height // 2, "crop": "center"}},
        "PU_FirstFull": {"class_type": "ImageScale", "inputs": {"image": _ref("PU_Image"), "upscale_method": "lanczos",
                                                                "width": width, "height": height, "crop": "center"}},
        "pre_half": {"class_type": "LTXVPreprocess", "inputs": {"image": _ref("PU_FirstHalf"), "img_compression": img_compression}},
        "pre_full": {"class_type": "LTXVPreprocess", "inputs": {"image": _ref("PU_FirstFull"), "img_compression": img_compression}},
        "i2v1": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": _ref("vae"), "image": _ref("pre_half"), "latent": _ref("PU_Latent"),
                                                                  "strength": 1.0, "bypass": False}},
        "i2v2": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": _ref("vae"), "image": _ref("pre_full"), "latent": _ref("up"),
                                                                  "strength": 1.0, "bypass": False}},
    }
    g["sample1"]["inputs"]["latent_image"] = _ref("i2v1")
    g["sample2"]["inputs"]["latent_image"] = _ref("i2v2")
    return g
