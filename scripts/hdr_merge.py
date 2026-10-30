"""Detail-safe HDR: merge LTX's HDR highlights onto the full-resolution source and write an HLG master.

Usage: uv run python scripts/hdr_merge.py ~/.pickups/runs/hdr_clip --source ORIGINAL.mp4 [--long-side 3840]
Needs the run's out/frame_*.exr (unclamped ACEScg from hdr_clip.py --exr). Writes merged_HLG.mp4 + merge_compare.png.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pickups import hdr, media


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--long-side", type=int, default=0, help="0 = source resolution")
    ap.add_argument("--lo", type=float, default=0.55)
    ap.add_argument("--hi", type=float, default=0.9)
    a = ap.parse_args()
    exrs = sorted((a.run / "out").glob("frame_*.exr"))
    if not exrs:
        raise SystemExit("no EXR frames in run/out")
    p = media.probe(a.source)
    w, h = (p.width, p.height) if not a.long_side else media.work_size(p.width, p.height, a.long_side)
    w, h = w // 2 * 2, h // 2 * 2
    n = len(exrs)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.source), "-vf", f"select='gte(n\\,{a.start})',scale={w}:{h}:flags=lanczos",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    sdr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    hlg_frames, mid = [], n // 2
    peak = []
    for i in range(min(n, len(sdr))):
        merged, wgt = hdr.merge_frame(sdr[i], hdr.read_exr(exrs[i]), a.lo, a.hi)
        peak.append(float(np.percentile(merged.max(axis=-1), 99.9)))
        hlg_frames.append(hdr.hlg_encode(merged))
        if i == mid:
            mid_merged, mid_wgt = merged, wgt
    out = a.run / "merged_HLG.mp4"
    hdr.write_hlg(hlg_frames, f"{p.fps.numerator}/{p.fps.denominator}", out)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    src_lin = hdr.srgb_to_linear(sdr[mid].astype(np.float32) / 255.0) @ hdr.M_709_TO_AP1.T
    tiles = [("Source (SDR)", sdr[mid]), ("Where LTX highlights are used", (np.stack([mid_wgt] * 3, -1) * 255).astype(np.uint8)),
             ("Source pulled -3 stops", hdr.preview(src_lin, -3)), ("Detail-safe HDR pulled -3 stops", hdr.preview(mid_merged, -3))]
    tw, th = 960, round(960 * h / w)
    sheet = Image.new("RGB", (tw * 2 + 4, th * 2 + 4), (10, 10, 12))
    for i, (label, arr) in enumerate(tiles):
        im = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, tw, 38], fill=(0, 0, 0))
        d.text((10, 6), label, fill=(255, 220, 0), font=font)
        sheet.paste(im, ((i % 2) * (tw + 4), (i // 2) * (th + 4)))
    sheet.save(a.run / "merge_compare.png")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt,width,height,color_transfer,color_primaries,nb_frames", "-of", "compact", str(out)],
                           capture_output=True, text=True).stdout.strip()
    rep = {"size": [w, h], "frames": len(hlg_frames), "peak_linear_p999_median": round(float(np.median(peak)), 2), "master": str(out), "probe": probe}
    (a.run / "merge_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
