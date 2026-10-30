"""Make review material for an Extend run: a seam strip image and an MP4 (original frames, then AI frames).

Usage: uv run python scripts/review_extend.py ~/.pickups/runs/k1_57 [--fps 29.97]
The MP4 uses the ORIGINAL canvas frames up to the seam and the generated frames after it, which is what the
app will deliver, with a 4-frame red bar in the corner on the first AI frames so the seam is easy to find.
"""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    rep = json.loads((a.run / "report.json").read_text())
    seam, fps = rep["seam_frame"], rep["fps"]
    canvas = sorted((a.run / "canvas_frames").glob("c_*.png"))
    gen = sorted((a.run / "out").glob("frame_*.png"))
    frames_dir = a.run / "review_frames"
    frames_dir.mkdir(exist_ok=True)
    seq = canvas[:seam] + gen[seam:]
    for i, f in enumerate(seq):
        im = Image.open(f).convert("RGB")
        if seam <= i < seam + 4:
            ImageDraw.Draw(im).rectangle([0, 0, im.width // 20, im.height // 40], fill=(230, 40, 40))
        im.save(frames_dir / f"r_{i:04d}.png")
    mp4 = a.run / "review.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "r_%04d.png"),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    picks = [max(seam - 6, 0), seam - 1, seam, seam + 1, seam + 8, min(seam + 24, len(seq) - 1), len(seq) - 1]
    ims = [Image.open(seq[i]).convert("RGB") for i in picks]
    w, h = ims[0].size
    strip = Image.new("RGB", (w * len(ims), h + 28), (10, 10, 12))
    d = ImageDraw.Draw(strip)
    for j, (i, im) in enumerate(zip(picks, ims)):
        strip.paste(im, (j * w, 28))
        d.text((j * w + 8, 6), f"frame {i}{'  (first AI frame)' if i == seam else ''}", fill=(255, 220, 0))
    strip.save(a.run / "seam_strip.png")
    print(json.dumps({"review": str(mp4), "strip": str(a.run / "seam_strip.png"), "frames": len(seq)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
