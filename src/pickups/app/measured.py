"""What Pickups has actually measured on a GPU, per feature, from its own finished takes (Oct 2026).

Only these numbers are shown to someone setting up: a feature on a GPU that isn't here says "not tested on this GPU"
rather than an estimate. Source: the take records (task, GPU, frames, total seconds) in the development library.
Replace a person counts people only (car swaps are out of this build); `minutes` is the median for a ~4 s take (89-105
frames, the default); `max_frames` is the longest take that finished; `fps` is the usual frame rate of those clips.
PRO 6000 times include runs that shared the card with other jobs.
"""

from __future__ import annotations

MEASURED: dict[str, dict[str, dict]] = {
    "RTX 4090": {
        "recast": {"runs": 43, "max_frames": 145, "fps": 29.97, "minutes": 14.2},
        "remove": {"runs": 42, "max_frames": 202, "fps": 29.97, "minutes": 13.5},
        "environment": {"runs": 112, "max_frames": 145, "fps": 24.0, "minutes": 5.2},
        # Relight any scene: the 20 test renders of Oct 22 (render only, models cached or not; 97-frame median).
        "relight": {"runs": 20, "max_frames": 121, "fps": 24.0, "minutes": 3.5},
        # Previz to final: the first-five GPU run of Oct 22 (97 frames at 1920x1088, 24 fps, render 216 s).
        "previz": {"runs": 1, "max_frames": 97, "fps": 24.0, "minutes": 3.6},
    },
    "RTX PRO 6000": {
        "recast": {"runs": 65, "max_frames": 225, "fps": 29.97, "minutes": 22.9},
        "remove": {"runs": 2, "max_frames": 81, "fps": 29.97, "minutes": None},
        "environment": {"runs": 9, "max_frames": 209, "fps": 29.97, "minutes": 10.0},
        "previz": {"runs": 17, "max_frames": 169, "fps": 24.0, "minutes": 5.5},
    },
}


def gpu_key(name: str | None) -> str | None:
    """The MEASURED key for a GPU name as ComfyUI reports it ("cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync")."""
    n = (name or "").upper()
    if "RTX PRO 6000" in n and "MIG" not in n:
        return "RTX PRO 6000"
    if "RTX 4090" in n:
        return "RTX 4090"
    return None


def for_task(gpu_name: str | None, task: str) -> dict | None:
    """The measured row for a feature on this GPU, with its longest take in seconds, or None (not tested here)."""
    key = gpu_key(gpu_name)
    row = MEASURED.get(key or "", {}).get(task)
    if not row:
        return None
    return {"gpu": key, **row, "max_seconds": round(row["max_frames"] / row["fps"], 1)}
