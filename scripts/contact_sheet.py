"""Numbered contact sheets for a folder of clips (one frame per clip), for picking test shots.

Usage: uv run python scripts/contact_sheet.py OUT_DIR FOLDER [FOLDER ...] [--at 2.5] [--cols 10] [--rows 8]
Writes OUT_DIR/list.txt (number -> path) and OUT_DIR/sheet_N.png.
"""

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXTS = {".mov", ".mp4", ".mxf"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--at", type=float, default=2.5)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for f in a.folders for p in f.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith("."))
    (a.out / "list.txt").write_text("\n".join(f"{i + 1}\t{p}" for i, p in enumerate(clips)))
    tw, th = 360, 270
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    thumbs = []
    for i, p in enumerate(clips):
        t = a.out / f"t_{i + 1:03d}.png"
        if not t.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.at), "-i", str(p), "-frames:v", "1",
                            "-vf", f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2",
                            str(t)], check=False, stdin=subprocess.DEVNULL)
        im = Image.open(t).convert("RGB") if t.exists() else Image.new("RGB", (tw, th), (40, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 52, 36], fill=(0, 0, 0))
        d.text((6, 3), str(i + 1), fill=(255, 220, 0), font=font)
        thumbs.append(im)
    per = a.cols * a.rows
    for s in range(0, len(thumbs), per):
        page = thumbs[s:s + per]
        rows = (len(page) + a.cols - 1) // a.cols
        sheet = Image.new("RGB", (a.cols * (tw + 2), rows * (th + 2)), (10, 10, 12))
        for j, im in enumerate(page):
            sheet.paste(im, ((j % a.cols) * (tw + 2), (j // a.cols) * (th + 2)))
        sheet.save(a.out / f"sheet_{s // per + 1}.png")
    print(f"{len(clips)} clips, {(len(thumbs) + per - 1) // per} sheets in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
