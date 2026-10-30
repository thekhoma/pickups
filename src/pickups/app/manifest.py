"""What each task needs: model files (where they go, where they come from, how big) and measured speed.

Sizes and sources are the exact Hugging Face files. Lightricks repos are gated (auto-approved): accept the licence
once on the model page, then download with a Hugging Face token.
"""

from __future__ import annotations

HF = "https://huggingface.co"

# file name -> (ComfyUI models subfolder, repo, path in repo, bytes, gated)
MODELS: dict[str, tuple[str, str, str, int, bool]] = {
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": ("diffusion_models", "Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", 21504034224, True),
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": ("text_encoders", "Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", 15372969374, True),
    "ltx-2.5-video-vae-bf16.safetensors": ("vae", "Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors", 1472223346, True),
    "ltx-2.5-audio-vae-bf16.safetensors": ("vae", "Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors", 364866540, True),
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": ("latent_upscale_models", "Lightricks/LTX-2.5", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 995778752, True),
    # Image models (the Image page). FLUX 3 klein from Black Forest Labs (gated: accept the licence once); the rest are
    # Comfy-Org repackages, exactly the files Comfy's official templates load.
    "flux3-image-klein-bf16.safetensors": ("diffusion_models", "black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3-image-klein-bf16.safetensors", 11097404512, True),
    "qwen3vl_4b_bf16.safetensors": ("text_encoders", "black-forest-labs/flux-3-image-klein-comfy", "text_encoders/qwen3vl_4b_bf16.safetensors", 8875719408, True),
    "flux2-vae.safetensors": ("vae", "black-forest-labs/flux-3-image-klein-comfy", "vae/flux2-vae.safetensors", 336211292, True),
    "qwen_image_2.1_turbo_int8_convrot.safetensors": ("diffusion_models", "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", 7256787829, False),
    "krea2_turbo_int8_convrot.safetensors": ("diffusion_models", "Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_int8_convrot.safetensors", 13492686496, False),
    "qwen3vl_4b_fp8_scaled.safetensors": ("text_encoders", "Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5242467968, False),
    "qwen_image_vae.safetensors": ("vae", "Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", 253806246, False),
    # Qwen-Image-Edit 2511 (Apache-2.0): the frame editor when it's picked in Setup → Licences → Frame editor.
    "qwen_image_edit_2511_fp8mixed.safetensors": ("diffusion_models", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", 20533762817, False),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": ("text_encoders", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680, False),
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": ("loras", "lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 849608296, False),
    "krea2_style_reference.safetensors": ("loras", "Comfy-Org/Krea-2", "loras/krea2_style_reference.safetensors", 457111760, False),
    "z_image_turbo_bf16.safetensors": ("diffusion_models", "Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", 12309866400, False),
    "qwen_3_4b.safetensors": ("text_encoders", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 8044982048, False),
    "ae.safetensors": ("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", 335304388, False),
    # Previz to final's start still from the previz depth (start_frame.py, method zbuf): Fun ControlNet Union 2.1, 8 steps.
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors": ("model_patches", "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors", 6712485600, False),
    "ideogram4_fp8_scaled.safetensors": ("diffusion_models", "Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors", 9280741285, False),
    "ideogram4_unconditional_fp8_scaled.safetensors": ("diffusion_models", "Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors", 9280741293, False),
    "qwen3vl_8b_fp8_scaled.safetensors": ("text_encoders", "Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors", 10588637512, False),
    "hidream_o1_image_bf16.safetensors": ("checkpoints", "Comfy-Org/HiDream-O1-Image", "checkpoints/hidream_o1_image_bf16.safetensors", 16365209824, False),
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors": ("loras", "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", 654465352, True),
    "depth_anything_3_mono_large.safetensors": ("geometry_estimation", "Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_mono_large.safetensors", 1336748056, False),
    "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors": ("loras", "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting", "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors", 1308778338, True),
    "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors": ("loras", "Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render", "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors", 1308787439, True),
    "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors": ("loras", "Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate", "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors", 327318439, True),
    "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors": ("loras", "Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details", "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors", 1308787534, True),
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors": ("loras", "Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR", "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors", 1308787472, True),
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors": ("loras", "Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR", "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors", 12614165, True),
    "sam3.1_multiplex_fp16.safetensors": ("checkpoints", "Comfy-Org/SAM3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors", 1745546848, False),
    # Replace a person reads the performance with SDPose (built-in nodes; every person recast loads it).
    "sdpose_wholebody_fp16.safetensors": ("checkpoints", "Comfy-Org/SDPose", "checkpoints/sdpose_wholebody_fp16.safetensors", 1916645792, False),
    "qwen_image_2.1_int8_convrot.safetensors": ("diffusion_models", "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7256783064, False),
    "qwen3vl_8b_int8_convrot.safetensors": ("text_encoders", "Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9350798360, False),
    "qwen_image_2.1_vae_bf16.safetensors": ("vae", "Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 675509688, False),
    "wan2.1_14B_SCAIL_2_int8_convrot.safetensors": ("diffusion_models", "Comfy-Org/SCAIL-2", "diffusion_models/wan2.1_14B_SCAIL_2_int8_convrot.safetensors", 16654506256, False),
    "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors": ("loras", "Comfy-Org/SCAIL-2", "loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors", 1226936552, False),
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors": ("loras", "Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 738005744, False),
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": ("text_encoders", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897, False),
    "clip_vision_h.safetensors": ("clip_vision", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", 1264219396, False),
    "Wan2_1_VAE_bf16.safetensors": ("vae", "Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", 253806278, False),
    # Relight any scene: LTX-2.3 as tested with the Relight IC-LoRA (Kijai's single-file transformer, Comfy-Org's Gemma 3).
    "ltx-2.3-22b-ic-lora-relight-1.0.safetensors": ("loras", "Lightricks/LTX-2.3-22b-IC-LoRA-Relight", "ltx-2.3-22b-ic-lora-relight-1.0.safetensors", 327309305, True),
    "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors": ("diffusion_models", "Kijai/LTX2.3_comfy", "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors", 25226571988, False),
    "ltx-2.3_text_projection_bf16.safetensors": ("text_encoders", "Kijai/LTX2.3_comfy", "text_encoders/ltx-2.3_text_projection_bf16.safetensors", 2312149072, False),
    "LTX23_video_vae_bf16.safetensors": ("vae", "Kijai/LTX2.3_comfy", "vae/LTX23_video_vae_bf16.safetensors", 1452258578, False),
    "LTX23_audio_vae_bf16.safetensors": ("vae", "Kijai/LTX2.3_comfy", "vae/LTX23_audio_vae_bf16.safetensors", 364855188, False),
    "gemma_3_12B_it_fp8_scaled.safetensors": ("text_encoders", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", 13205434827, False),
    "seedvr2_3b_int8_convrot.safetensors": ("diffusion_models", "Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", 3458259704, False),
    "seedvr2_ema_vae_fp16.safetensors": ("vae", "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", 501324814, False),
}

# The published sha256 of files whose exact build matters (a different 2.1 build is the same size).
SHA256: dict[str, str] = {
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors": "d1251cc7bc3486bc61d25c3be498ef394c31c85ddf4ee9137d2e933411f4a689",
}

LTX_CORE = ["ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"]
QWEN21 = ["qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors", "qwen_image_2.1_vae_bf16.safetensors"]
SAM = ["sam3.1_multiplex_fp16.safetensors"]
# Previz to final's start still (method zbuf). Without them the start still is made the B3 way, so they're listed as
# missing but never block the feature (SOFT_MODELS).
START_DEPTH = ["Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors", "z_image_turbo_bf16.safetensors",
               "qwen_3_4b.safetensors", "ae.safetensors"]
QWEN_EDIT_2511 = ["qwen_image_edit_2511_fp8mixed.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image_vae.safetensors",
                  "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]

TASK_MODELS: dict[str, list[str]] = {
    "remove": SAM + QWEN21 + LTX_CORE + ["ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors", "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"],
    # Recast's default engine (LTX-2.5): the frame editor paints the character in, LTX renders the shot from its depth.
    "recast": SAM + QWEN21 + LTX_CORE + ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "depth_anything_3_mono_large.safetensors",
                                         "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "sdpose_wholebody_fp16.safetensors"],
    # Recast's SCAIL-2 engine (More options), an extra download: `pickups models --task recast-scail`.
    "recast-scail": SAM + ["wan2.1_14B_SCAIL_2_int8_convrot.safetensors", "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
                           "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                           "clip_vision_h.safetensors", "Wan2_1_VAE_bf16.safetensors"],
    # Previz to final: Follow "loose" drives from depth (DA3 and the union-control LoRA) instead of the layout LoRA.
    "previz": QWEN21 + LTX_CORE + ["ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors", "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                                   "depth_anything_3_mono_large.safetensors", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"] + START_DEPTH,
    # Change the weather picks what to keep with SAM 3.1.
    "environment": SAM + QWEN21 + LTX_CORE + ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "depth_anything_3_mono_large.safetensors",
                                              "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
    "reshoot": LTX_CORE + ["ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors", "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                           "depth_anything_3_mono_large.safetensors"],
    "relight": ["ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors", "gemma_3_12B_it_fp8_scaled.safetensors",
                "ltx-2.3_text_projection_bf16.safetensors", "LTX23_video_vae_bf16.safetensors", "LTX23_audio_vae_bf16.safetensors",
                "ltx-2.3-22b-ic-lora-relight-1.0.safetensors"],
    "finish": LTX_CORE + ["seedvr2_3b_int8_convrot.safetensors", "seedvr2_ema_vae_fp16.safetensors",
                          "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors", "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"],
    # New video (Create: text to video, image to video): LTX-2.5 alone, video only (no audio VAE).
    "video": LTX_CORE[:3] + ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
}

# Finish's three choices need different files: 4K is SeedVR2 alone, HDR is LTX-2.5 with the SDR-to-HDR LoRA, 4K HDR both.
FINISH_CHOICE_MODELS: dict[str, list[str]] = {
    "4k": ["seedvr2_3b_int8_convrot.safetensors", "seedvr2_ema_vae_fp16.safetensors"],
    "hdr": LTX_CORE + ["ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors", "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"],
}
FINISH_CHOICE_MODELS["4k-hdr"] = TASK_MODELS["finish"]

# Image models by registry id (pickups.images): what `pickups models --task image-<id>` downloads. "image" = the default.
IMAGE_MODEL_FILES: dict[str, list[str]] = {
    "flux3-klein": ["flux3-image-klein-bf16.safetensors", "qwen3vl_4b_bf16.safetensors", "flux2-vae.safetensors"],
    "qwen-image-2.1": QWEN21,
    "qwen-image-2.1-turbo": ["qwen_image_2.1_turbo_int8_convrot.safetensors"] + QWEN21[1:],
    "krea2-turbo": ["krea2_turbo_int8_convrot.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen_image_vae.safetensors",
                    "krea2_style_reference.safetensors"],
    "z-image-turbo": ["z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"],
    "ideogram4": ["ideogram4_fp8_scaled.safetensors", "ideogram4_unconditional_fp8_scaled.safetensors",
                  "qwen3vl_8b_fp8_scaled.safetensors", "flux2-vae.safetensors"],
    "hidream-o1": ["hidream_o1_image_bf16.safetensors"],
}
TASK_MODELS["image"] = IMAGE_MODEL_FILES["flux3-klein"]
for _mid, _files in IMAGE_MODEL_FILES.items():
    TASK_MODELS[f"image-{_mid}"] = _files

# Person swap's optional Relight (More options) always uses these four files, whatever the frame-editor setting; the setup tiles
# do not list them for it, so with the default frame editor run `pickups models --task commercial-safe` once before turning Relight on.
# `pickups models --task commercial-safe`: the Apache-2.0 frame editor (not part of the default download).
TASK_MODELS["commercial-safe"] = QWEN_EDIT_2511
OPTIONAL = ("image-", "commercial-safe", "recast-scail")  # extra downloads, not setup tiles
# Files a feature lists (Setup says when they're missing) but runs without, on its fallback.
SOFT_MODELS: dict[str, list[str]] = {"previz": START_DEPTH}


def task_models(commercial_safe: bool = False) -> dict[str, list[str]]:
    """The setup tiles' model lists. With commercial-safe edits on, the tasks that edit frames need Qwen-Image-Edit 2511
    instead of Qwen-Image 2.1."""
    extra = _extra().get("tasks") or {}
    tiles = {t: f + [x for x in extra.get(t, []) if x in MODELS] for t, f in TASK_MODELS.items() if not t.startswith(OPTIONAL)}
    if commercial_safe:
        for t in ("remove", "previz", "environment", "recast"):
            tiles[t] = [f for f in tiles[t] if f not in QWEN21] + QWEN_EDIT_2511
    return tiles


# The nodes each feature's graphs use beyond the stock loaders, samplers and savers every ComfyUI has. A node in a
# NODE_PACKS entry comes from that pack ("Install or update <pack>"); any other is core ("Update ComfyUI"). The setup
# check reads these against /object_info (doctor.py); tests/test_manifest_coverage.py keeps them in step with graphs.py.
_LTX_GUIDE = ["LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly"]
TASK_NODES: dict[str, list[str]] = {
    "remove": _LTX_GUIDE + ["SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask", "TextEncodeQwenImage21", "QwenImage21Cache",
                            "LTXVImgToVideoInplace", "LTXVPreprocess", "BatchImagesNode", "ThresholdMask",
                            "LTXAddVideoICLoRAGuideAdvanced", "LTXVDilateVideoMask", "LTXVInpaintPreprocess", "LTXVSetAudioRefTokens"],
    "recast": _LTX_GUIDE + ["SAM3_Detect", "SAM3_VideoTrack", "TextEncodeQwenImage21", "LTXVAddGuide", "LTXVCropGuides", "DA3Inference",
                            "DA3Render", "LoadDA3Model", "SDPoseKeypointExtractor", "SDPoseDrawKeypoints", "LTXVLatentUpsampler",
                            "LTXVImgToVideoInplace", "Canny", "LTXVLaplacianPyramidBlend", "LTXVSetVideoLatentNoiseMasks"],
    "recast-scail": ["SAM3_VideoTrack", "SCAIL2ColoredMask", "WanSCAILToVideo", "CLIPVisionEncode", "ModelSamplingSD3"],
    # Make start frame from previz (start_frame.py): method zbuf's Fun ControlNet (core nodes), method B3's depth, soft
    # canny and Qwen-Image 2.1 cache.
    "previz": _LTX_GUIDE + ["TextEncodeQwenImage21", "LTXVLatentUpsampler", "LTXVCropGuides", "LoadDA3Model", "DA3Inference", "DA3Render",
                            "Canny", "QwenImage21Cache", "ModelPatchLoader", "ZImageFunControlnet", "ConditioningZeroOut"],
    "environment": _LTX_GUIDE + ["TextEncodeQwenImage21", "LTXVImgToVideoInplace", "DA3Inference", "DA3Render", "LoadDA3Model", "Canny",
                                 "SAM3_Detect", "LTXVLatentUpsampler"],
    "reshoot": _LTX_GUIDE + ["DA3Inference"],
    "relight": _LTX_GUIDE + ["LTXVTiledVAEDecode", "LTXVConcatAVLatent", "DualCLIPLoader"],
    "finish": ["SeedVR2Conditioning", "SeedVR2TemporalChunk", "SeedVR2Preprocess", "SeedVR2PostProcessing", "SeedVR2TemporalMerge",
               "ImageColorSpace", "SaveImageAdvanced", "CreateVideo", "ResizeImageMaskNode", "LTXVSDRToHDRWorkingSpace",
               "LTXVHDRDecodePostprocess", "LTXVLoadConditioning", "LTXVVAEForceFloat32"],
    "image": ["ReferenceMultiLatent", "ConditioningZeroOut", "EmptyFlux2LatentImage"],
    "video": ["LTXVLatentUpsampler", "LTXVPreprocess", "LTXVImgToVideoInplace", "CreateVideo"],
}

# Custom node packs the workflows need (everything else is core ComfyUI). Never auto-installed: the setup check says
# what to add (ComfyUI Manager name, or the git command) and the user does it.
NODE_PACKS = {
    "ComfyUI-LTXVideo": {"url": "https://github.com/Lightricks/ComfyUI-LTXVideo", "by": "Lightricks", "manager_name": "ComfyUI-LTXVideo",
                         "nodes": ["LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly", "LTXVSDRToHDRWorkingSpace",
                                   "LTXVHDRDecodePostprocess", "LTXVLoadConditioning", "LTXVVAEForceFloat32", "LTXVTiledFusionSampler",
                                   "LTXAddVideoICLoRAGuideAdvanced", "LTXVDilateVideoMask", "LTXVInpaintPreprocess",
                                   "LTXVLaplacianPyramidBlend", "LTXVSetAudioRefTokens", "LTXVSetAudioVideoMaskByTime",
                                   "LTXVSetVideoLatentNoiseMasks", "LTXVTiledVAEDecode"]},
}
# Stock nodes every ComfyUI 0.41+ has (loaders, samplers, savers, image and LTX core plumbing): not checked per feature.
STOCK_NODES = {
    "BasicScheduler", "CFGGuider", "CFGOverride", "CLIPLoader", "CLIPTextEncode", "CLIPVisionLoader", "CheckpointLoaderSimple",
    "ComfyMathExpression", "DualModelGuider", "EmptyLTXVLatentVideo", "EmptyLatentImage", "EmptySD3LatentImage", "GetImageSize",
    "GetVideoComponents", "ImageFromBatch", "ImageScale", "ImageScaleToTotalPixels", "ImageToMask", "ImageUpscaleWithModel",
    "KSampler", "KSamplerSelect", "LTXVAddLatentGuide", "LTXVAudioVAEDecode", "LTXVAudioVAEEncode", "LTXVConditioning",
    "LTXVDualCFGGuider", "LTXVEmptyLatentAudio", "LTXVScheduler", "LTXVSeparateAVLatent", "LatentUpscaleModelLoader", "LoadImage",
    "LoadVideo", "LoraLoaderModelOnly", "ManualSigmas", "MaskToImage", "ModelNoiseScale", "ModelSamplingAuraFlow", "ModelSamplingFlux",
    "RandomNoise", "ReplaceVideoLatentFrames", "SamplerCustom", "SamplerCustomAdvanced", "SaveAudio", "SaveImage", "SaveVideo",
    "UNETLoader", "UpscaleModelLoader", "VAEDecode", "VAEDecodeTiled", "VAEEncode", "VAEEncodeAudio", "VAEEncodeTiled", "VAELoader",
}
# Nodes only opt-in paths use (non-default image models, the commercial-safe frame editor, the NVIDIA RTX upscaler, the
# alternative SeedVR2 pack, the car workflows behind PICKUPS_CARS): not part of any feature's setup check.
OPTIONAL_NODES = {
    "CFGNorm", "FluxKontextImageScale", "FluxKontextMultiReferenceLatentMethod", "TextEncodeQwenImageEditPlus",
    "EmptyHiDreamO1LatentImage", "HiDreamO1PatchSeamSmoothing", "HiDreamO1ReferenceImages", "Ideogram4Scheduler",
    "RTXTrueHDR", "RTXVideoSuperResolution", "FLSeedVR2ModelLoader", "FLSeedVR2Upscale", "LTXVEditAnythingLoopingSampler",
}
# Model files only builders no feature reaches use (kept for experiments): not downloaded or checked.
UNLISTED_MODELS = {
    "RealESRGAN_x4plus.safetensors",  # gan_upscale
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",  # dev_first_pass, edit_anything_motion
    "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors", "edit_anything_30k_v0.1_motion_transfer_r128.safetensors",  # cars
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",  # retake_two_stage
    "seedvr2_distill_6L_1.4B_sharp_fp16_comfyui.safetensors",  # seedvr2_upscale
}
UPDATE_COMFY = "git pull && pip install -r requirements.txt"  # run in the ComfyUI folder (portable: update/update_comfyui.bat)


def pack_install(pack: str) -> str:
    return f"git clone {NODE_PACKS[pack]['url']} custom_nodes/{pack}"


def missing_nodes(task: str, nodes: set[str]) -> tuple[list[str], list[dict]]:
    """A feature's nodes that this ComfyUI lacks: core ones (an update fixes them) and, per pack, its missing nodes with
    how to get the pack."""
    want = list(dict.fromkeys(TASK_NODES.get(task, []) + extra_nodes().get(task, [])))
    lacking = [n for n in want if n not in nodes]
    packs: dict[str, list[str]] = {}
    core: list[str] = []
    for n in lacking:
        pack = next((p for p, info in NODE_PACKS.items() if n in info["nodes"]), None)
        (packs.setdefault(pack, []) if pack else core).append(n)
    return core, [{"pack": p, "url": NODE_PACKS[p]["url"], "by": NODE_PACKS[p]["by"], "manager_name": NODE_PACKS[p]["manager_name"],
                   "nodes": ns, "install": pack_install(p)} for p, ns in packs.items()]


# --- extra requirements (tests and demos): PICKUPS_HOME/manifest_extra.json -------------------------------------------
# {"models": {"<file>": [folder, repo, path, bytes, gated]}, "tasks": {"<task>": ["<file>", ...]}, "nodes": {"<task>": ["<Node>"]}}
# Lets a scratch home show a missing file or node without touching a real ComfyUI. Absent on real installs.
def _extra() -> dict:
    import json
    import os
    from pathlib import Path

    f = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "manifest_extra.json"
    try:
        data = json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}
    for name, row in (data.get("models") or {}).items():
        MODELS.setdefault(name, tuple(row))
    return data


def extra_nodes() -> dict[str, list[str]]:
    return _extra().get("nodes") or {}
TASK_PACKS = {"remove": ["ComfyUI-LTXVideo"], "previz": ["ComfyUI-LTXVideo"], "environment": ["ComfyUI-LTXVideo"], "reshoot": ["ComfyUI-LTXVideo"],
              "finish": ["ComfyUI-LTXVideo"], "relight": ["ComfyUI-LTXVideo"], "recast": ["ComfyUI-LTXVideo"], "recast-scail": [], "video": []}

# Measured on an RTX 4090 24 GB (seconds, 4 s / 97 frames): drives the ETAs before a machine has its own numbers.
BENCH_4090 = [
    {"task": "Select (click + track)", "seconds": 10, "frames": 97, "size": "1024 proxy"},
    {"task": "Remove (locked-off, real pixels)", "seconds": 330, "frames": 97, "size": "3840x2752"},
    {"task": "Replace a person", "seconds": 300, "frames": 81, "size": "1920x1088"},
    {"task": "Previz to final", "seconds": 220, "frames": 97, "size": "1920x1088"},
    {"task": "Finish 4K (best)", "seconds": 890, "frames": 97, "size": "3840x2160"},
    {"task": "Finish HDR", "seconds": 360, "frames": 89, "size": "1920x1080"},
]
# Recast is its default engine, LTX-2.5 (97 frames of a 4K shot, composited at 3840 wide: 344 s); recast-scail is SCAIL-2.
TASK_ETA_4090 = {"remove": 330, "recast": 360, "recast-scail": 300, "previz": 220, "environment": 240, "reshoot": 260, "finish": 1300,
                 "relight": 380, "image": 10, "video": 60}  # video: 1080p, 97 frames, warm (cold adds about 45 s)

# Rough speed relative to a 4090 for GPUs we know (lower = faster). Unknown GPUs use 1.0 until benchmarked.
GPU_SPEED = {"5090": 0.7, "4090": 1.0, "6000 ada": 0.95, "pro 6000": 0.6, "a6000": 1.6, "3090": 1.8, "h100": 0.55, "a100": 0.9,
             "l40s": 1.05, "4080": 1.4}


def gpu_factor(name: str) -> float:
    n = name.lower()
    for key, f in GPU_SPEED.items():
        if key in n:
            return f
    return 1.0


def model_info(name: str) -> dict:
    folder, repo, path, size, gated = MODELS[name]
    return {"file": name, "folder": folder, "gb": round(size / 1e9, 2), "gated": gated, "repo": repo,
            "url": f"{HF}/{repo}/resolve/main/{path}", "page": f"{HF}/{repo}", **({"sha256": SHA256[name]} if name in SHA256 else {})}
