"""Will this computer work: the per-feature rule from SYSTEM_REQUIREMENTS.md (c), in one place.

One VRAM unit everywhere: decimal GB, `vram_total / 1e9` (an RTX 4090 reads 25.8). NVIDIA means the device name says
NVIDIA; `type == "cuda"` alone isn't enough, since PyTorch on ROCm reports AMD cards as cuda too.
"""

from __future__ import annotations

import re
import shutil
import subprocess

MIN_COMFY = (0, 41)
MIN_VRAM_GB = 23.5  # the tested profile: an RTX 4090 (25.8 decimal GB)
RELIGHT_VRAM_GB = 25.0  # Relight any scene peaks about 25.5 of a 4090's 25.8: nothing else may be loaded
VRAM_GB = {"relight": RELIGHT_VRAM_GB}  # every other feature: MIN_VRAM_GB
# Free space on the models drive beyond the missing files, for ComfyUI's input/ and output/ growth (Finish keeps 4K
# PNG and EXR frames); SYSTEM_REQUIREMENTS.md (c).
HEADROOM_GB = {"finish": 20.0, "image": 5.0}
DEFAULT_HEADROOM_GB = 10.0
LIBRARY_MIN_FREE_GB = 11.0  # about ten shots at 1.1 GB each
MIN_FFMPEG = 8  # colour-managed scale (out_primaries): older ones skip converting HDR and wide-gamut sources


def vram_gb(stats: dict | None) -> float:
    dev = ((stats or {}).get("devices") or [{}])[0]
    return round(float(dev.get("vram_total") or 0) / 1e9, 1)


def is_nvidia(stats: dict | None) -> bool:
    dev = ((stats or {}).get("devices") or [{}])[0]
    return "nvidia" in str(dev.get("name") or "").lower()


def comfy_version(stats: dict | None) -> tuple[int, ...] | None:
    v = str(((stats or {}).get("system") or {}).get("comfyui_version") or "")
    nums = re.findall(r"\d+", v)
    return tuple(int(x) for x in nums[:3]) if nums else None


def comfy_ok(stats: dict | None) -> bool | None:
    v = comfy_version(stats)
    return None if v is None else v[:2] >= MIN_COMFY


def vram_needed(task: str) -> float:
    return VRAM_GB.get(task, MIN_VRAM_GB)


def support(task: str, stats: dict | None) -> str:
    """"ok", "untested" (NVIDIA under the tested VRAM: allowed, not promised) or "unsupported" (not NVIDIA)."""
    if not is_nvidia(stats):
        return "unsupported"
    return "ok" if vram_gb(stats) >= vram_needed(task) else "untested"


def headroom(tasks: list[str]) -> float:
    return max((HEADROOM_GB.get(t, DEFAULT_HEADROOM_GB) for t in tasks), default=0.0)


def disk_free_gb(path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return None


def ffmpeg_major() -> int | None:
    """ffmpeg's major version on this computer, or None when it isn't on PATH or can't be read."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"ffmpeg version n?(\d+)", out)
    return int(m.group(1)) if m else None
