"""Plain names for the ComfyUI workflows a take ran, read from the saved API-format graph.

"01 PICK · 02 PICK · 03 PICK" says nothing; "Pick · SAM 3.1 · woman", "Pick · SAM 3.1 · backpack" and
"Track · SAM 3.1 · forward from frame 82" say what each one did. A title is "<what> · <model>[ · <detail>]".
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# What a step did, by its output folder (pickups/<step>/...). Trailing digits are dropped: shadow0 is shadow.
STEPS = {
    "pick": "Pick", "track": "Track", "select": "Find", "shadow": "Find", "recast_new": "Find", "recast_track": "Track",
    "recast_paint": "Paint", "recast_ltx": "Recast",
    "plate": "Plate", "edit": "Edit", "look": "Look frame", "look_key": "Key frame", "environment_key": "First frame",
    "environment": "Weather", "drive": "Drive", "previz": "Render", "layout": "Render", "remove": "Fill",
    "recast": "Recast", "scail2": "Recast", "cast": "Reference", "still": "Still", "image": "Image",
    "finish4k": "4K", "finish4k_rtx": "4K", "seedvr2": "4K", "seedvr2v": "4K", "rtxvsr": "4K", "gan": "Upscale",
    "finish_hdr": "HDR", "hdr": "HDR", "rtxhdr": "HDR", "refine": "Refine", "retake": "Reshoot", "retake2": "Reshoot",
    "relight_scene": "Relight",
}


def _key(step: str) -> str:
    return re.sub(r"\d+$", "", step or "")


def step_title(step: str) -> str:
    """The title from the step alone: "look_key" is "Key frame", an unknown "foo_bar" is "Foo bar"."""
    k = _key(step)
    return STEPS.get(k) or k.replace("_", " ").strip().capitalize() or "Workflow"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _short(text: str, n: int = 28) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _clicks(raw) -> int:
    try:
        return len(json.loads(raw)) if isinstance(raw, str) else len(raw or [])
    except ValueError:
        return 0


def _describe(nodes: list[dict], step: str, frame: int | None) -> list[str]:
    types = {n["class_type"] for n in nodes}
    kind = STEPS.get(step)

    def inputs(*classes: str) -> list[dict]:
        return [n.get("inputs") or {} for n in nodes if not classes or n["class_type"] in classes]

    loras = " ".join(str(i.get("lora_name", "")) for i in inputs()).lower()
    weights = " ".join(str(i.get(k, "")) for i in inputs() for k in ("unet_name", "ckpt_name")).lower()
    # The clicked frame names the person's pick and tracks; Recast on LTX follows the new character from its own frame.
    at = f"frame {frame}" if frame is not None and step != "recast_track" else ""

    # Order matters: SCAIL-2 tracks people with SAM, and every LTX graph loads the same base model.
    if "WanSCAILToVideo" in types:
        return [kind or "Recast", "SCAIL-2"]
    if types & {"SeedVR2Conditioning", "FLSeedVR2Upscale"}:
        return [kind or "4K", "SeedVR2"]
    if "RTXVideoSuperResolution" in types:
        return [kind or "4K", "RTX Video Super Resolution"]
    if "RTXTrueHDR" in types:
        return [kind or "HDR", "RTX Video HDR"]
    if "sdr-to-hdr" in loras:
        return [kind or "HDR", "LTX SDR-to-HDR"]
    if "layout-to-render" in loras:
        return [kind or "Render", "LTX-2.5 Layout-To-Render"]
    if "union-control" in loras:
        structure = "depth" if any(t.startswith("DA3") or "Depth" in t for t in types) else "edges" if "Canny" in types else ""
        return [kind or "Drive", "LTX-2.5 Union Control", structure]
    if "in-outpainting" in loras:
        return [kind or "Fill", "LTX-2.5 In/Outpaint"]
    if "ic-lora-relight" in loras:
        return [kind or "Relight", "LTX-2.3 Relight IC-LoRA"]
    if "refine-details" in loras:
        return [kind or "Refine", "LTX-2.5 Refine Details"]
    if "ltx" in weights or any(t.startswith("LTXV") for t in types):
        return [kind or "Video", "LTX-2.5"]

    qwen = ("Qwen-Image 2.1 Turbo" if "qwen_image_2.1_turbo" in weights else
            "Qwen-Image 2.1" if "TextEncodeQwenImage21" in types or "qwen_image_2.1" in weights else
            "Qwen-Image-Edit 2511" if "TextEncodeQwenImageEditPlus" in types else "")
    if qwen:
        refs = max(0, len(inputs("LoadImage")) - 1)  # image 1 is the frame being edited
        return [kind or ("Edit" if inputs("LoadImage") else "Image"), qwen, _plural(refs, "reference") if refs else ""]
    if "flux3" in weights or "EmptyFlux2LatentImage" in types:
        return [kind or "Image", "FLUX 3 klein"]

    if "SAM3_VideoTrack" in types:
        files = " ".join(str(i.get("file", "")) for i in inputs("LoadVideo")).lower()
        way = "backward" if re.search(r"\b(bwd|back)", files) else "forward" if re.search(r"\b(fwd|forward)", files) else ""
        where = f"{way} from {at}" if way and at else way or (f"from {at}" if at else "")
        objects = max([int(i.get("max_objects") or 1) for i in inputs("SAM3_VideoTrack")] or [1])
        return [kind or "Track", "SAM 3.1", ", ".join(x for x in (where, _plural(objects, "object") if objects > 1 else "") if x)]
    if "SAM3_Detect" in types:
        labels = [str(i.get("text", "")).strip() for i in inputs("CLIPTextEncode") if str(i.get("text", "")).strip()]
        clicks = sum(_clicks(i.get("positive_coords")) for i in inputs("SAM3_Detect"))
        detail = _short(labels[0]) if labels else (at or _plural(clicks, "click")) if clicks else ""
        return [kind or ("Find" if "LoadVideo" in types else "Pick"), "SAM 3.1", detail]

    if "ImageUpscaleWithModel" in types:
        name = " ".join(str(i.get("model_name", "")) for i in inputs("UpscaleModelLoader"))
        return [kind or "Upscale", "Real-ESRGAN" if "esrgan" in name.lower() else Path(name).stem]
    return []


def workflow_title(graph: dict, step: str = "", frame: int | None = None) -> str:
    """"Pick · SAM 3.1 · woman" for one saved workflow. `frame` is the frame the user clicked (Remove and Recast)."""
    try:
        nodes = [n for n in graph.values() if isinstance(n, dict) and isinstance(n.get("class_type"), str)]
        parts = _describe(nodes, _key(step), int(frame) if isinstance(frame, (int, float)) else None)
    except Exception:  # noqa: BLE001 - a name is a nicety; never fail a take over it
        parts = []
    return " · ".join(p for p in parts if p) or step_title(step)


@lru_cache(maxsize=1024)
def _file_title(path: str, mtime_ns: int, step: str, frame: int | None) -> str:
    try:
        graph = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return step_title(step)
    return workflow_title(graph if isinstance(graph, dict) else {}, step, frame)


def saved_title(path: Path, step: str, frame: int | None = None) -> str:
    """Older takes saved no title: name it from the saved file (read once while the file is unchanged)."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return step_title(step)
    return _file_title(str(path), mtime, step, int(frame) if isinstance(frame, (int, float)) else None)
