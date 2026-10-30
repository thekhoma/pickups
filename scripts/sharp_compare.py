"""Build a 100%-crop sharpness comparison for one AI frame across Extend variants.

Usage: uv run python scripts/sharp_compare.py OUT.png --frame 49 --crop 980,560,560,420 \
          "label=path/to/frame.png" ["label=..."]
Each image is resized to 1920 wide first (so crops are at the same scale), then cropped (x,y,w,h) and labelled.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--crop", default="980,560,560,420")
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    x, y, w, h = (int(v) for v in a.crop.split(","))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for item in a.items:
        label, path = item.split("=", 1)
        im = Image.open(Path(path).expanduser()).convert("RGB")
        im = im.resize((1920, round(im.height * 1920 / im.width)), Image.LANCZOS)
        tile = im.crop((x, y, x + w, y + h))
        d = ImageDraw.Draw(tile)
        d.rectangle([0, 0, w, 34], fill=(0, 0, 0))
        d.text((10, 5), label, fill=(255, 220, 0), font=font)
        tiles.append(tile)
    cols = 3 if len(tiles) > 4 else 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), (10, 10, 12))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    sheet.save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
