"""New video on LTX-2.5: text to video and image to video. The sizes, lengths and the checked request a job runs.

LTX-2.5 is the one video model. Its two-pass graph (graphs.ltx_text_to_video / ltx_image_to_video) runs pass 1 at half
size, so the final width and height are multiples of 64 (the half pass needs multiples of 32: true 1280x720 can't be
made, 1280x704 can). The length is 8n+1 frames, at most 97, at 24 fps. Every size here ran on an RTX 4090 (24 GB) at
33, 65 and 97 frames with no failures; other 1080p-class aspects were calculated, not run, so they aren't offered.
The web UI keeps the same table in web/src/lib/video.ts (tests/test_video.py keeps the two in step).
"""

from __future__ import annotations

import re

from pickups import seeds

MODEL = "ltx-2.5"
MODEL_NAME = "LTX-2.5"
FPS = 24
MAX_FRAMES = 97
MODES = ("text", "image")
ASPECTS = ("16:9", "9:16", "1:1", "4:3", "21:9")
RESOLUTIONS = ("720p", "1080p")
# Final sizes (pass 1 runs at half). 1080p is 16:9 only: the only 1080p-class size that was run.
SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960), "4:3": (1024, 768), "21:9": (1344, 576)},
    "1080p": {"16:9": (1920, 1088)},
}
# Seconds -> frames at 24 fps (8n+1): 1 s = 25, 2 s = 49, 3 s = 73, 4 s = 97.
DURATIONS: dict[int, int] = {1: 25, 2: 49, 3: 73, 4: 97}
DEFAULTS = {"mode": "text", "aspect": "16:9", "resolution": "1080p", "duration": 4}


def valid_size(width: int, height: int) -> bool:
    return width > 0 and height > 0 and width % 64 == 0 and height % 64 == 0


def valid_length(frames: int) -> bool:
    return 9 <= frames <= MAX_FRAMES and (frames - 1) % 8 == 0


def size_for(aspect: str, resolution: str = "720p") -> tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError(f"Pick an aspect ratio: {', '.join(ASPECTS)}.")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Pick a size: {', '.join(RESOLUTIONS)}.")
    if aspect not in SIZES[resolution]:
        raise ValueError(f"{resolution} is {', '.join(SIZES[resolution])} only. Pick 720p for {aspect}.")
    return SIZES[resolution][aspect]


def frames_for(duration: int | float | str) -> int:
    try:
        f = float(duration)
    except (TypeError, ValueError):
        f = -1.0
    d = int(f)
    if d not in DURATIONS or f != d:
        raise ValueError(f"Pick a length of {', '.join(f'{s} s' for s in DURATIONS)}.")
    return DURATIONS[d]


def options() -> dict:
    """What the pickers offer (GET /api/video/options; the CLI's `pickups video --help` says the same)."""
    return {"model": MODEL, "model_name": MODEL_NAME, "fps": FPS, "max_frames": MAX_FRAMES, "modes": list(MODES),
            "aspects": list(ASPECTS), "resolutions": list(RESOLUTIONS),
            "sizes": {r: {a: list(s) for a, s in SIZES[r].items()} for r in RESOLUTIONS},
            "durations": [{"seconds": s, "frames": f} for s, f in DURATIONS.items()], "defaults": dict(DEFAULTS)}


def request(params: dict) -> dict:
    """A video job's params, checked and filled in: mode, prompt, aspect, resolution, width, height, frames, seed and,
    for image to video, the picture (a {kind, id} reference). Raises ValueError with one plain sentence."""
    p = DEFAULTS | {k: v for k, v in (params or {}).items() if v is not None}
    mode = str(p["mode"])
    if mode not in MODES:
        raise ValueError("Mode is text or image.")
    prompt = str(p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Describe the shot first." if mode == "text" else "Describe the motion first.")
    width, height = size_for(str(p["aspect"]), str(p["resolution"]))
    frames = frames_for(p["duration"])
    assert valid_size(width, height) and valid_length(frames)
    seed = seeds.pick(p.get("seed"))
    image = p.get("image")
    if mode == "image":
        if not (isinstance(image, dict) and image.get("kind") in ("image", "cast", "upload") and image.get("id")):
            raise ValueError("Choose a picture to start from.")
        image = {"kind": image["kind"], "id": str(image["id"])}
    else:
        image = None
    return {"mode": mode, "prompt": prompt, "aspect": p["aspect"], "resolution": p["resolution"], "width": width, "height": height,
            "duration": int(float(p["duration"])), "frames": frames, "fps": FPS, "seed": seed, "image": image, "model": MODEL}


SHOT_NAME_MAX = 90


def shot_name(prompt: str) -> str:
    """The new shot's name: the prompt's whole first sentence ("A golden retriever runs along a beach at sunset"), never
    cut mid-sentence after six words (FableWalk: "A golden retriever runs along a"). A very long sentence stops at the
    last whole word within SHOT_NAME_MAX characters."""
    text = " ".join(prompt.split())
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip(" ,.;:")
    if len(first) > SHOT_NAME_MAX:
        first = first[:SHOT_NAME_MAX].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return first or "New video"
