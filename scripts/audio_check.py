"""Compare loudness envelopes of the canvas audio and the generated audio around the seam.

Usage: uv run python scripts/audio_check.py ~/.pickups/runs/l1 --seam-seconds 1.0417
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np


def pcm(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def envelope(x: np.ndarray, win: int = 2400) -> np.ndarray:
    n = len(x) // win
    return np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seam-seconds", type=float, required=True)
    a = ap.parse_args()
    canvas = pcm(a.run / "canvas.mp4")
    gen = pcm(next(a.run.glob("audio_*")))
    ec, eg = envelope(canvas), envelope(gen)
    n = min(len(ec), len(eg))
    seam = int(a.seam_seconds * 48000 / 2400)
    corr_before = float(np.corrcoef(ec[:seam], eg[:seam])[0, 1]) if seam > 2 else float("nan")
    print(f"windows: {n} (50 ms each), seam at window {seam}")
    print(f"envelope correlation before seam: {corr_before:.3f}")
    print("canvas :", " ".join(f"{v:.2f}" for v in ec[:n]))
    print("output :", " ".join(f"{v:.2f}" for v in eg[:n]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
