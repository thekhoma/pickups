"""To HDR test: LTX-2.5 SDR->HDR on a clip, plus an exposure-pull comparison against the SDR original.

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/hdr_clip.py --clip SRC --name hdr_137 [--frames 49] [--long-side 1920]
Writes the HLG master, tonemapped preview frames, exposure-pull PNGs and a comparison sheet to ~/.pickups/runs/<name>/.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comfy_sdk import Comfy, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def sdr_pull(img: np.ndarray, ev: float) -> np.ndarray:
    lin = srgb_to_linear(img.astype(np.float32) / 255.0) * (2.0 ** ev)
    return (linear_to_srgb(lin) * 255).round().astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--exr", action="store_true")
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    p = media.probe(a.clip)
    frames = media.ltx_frames(min(a.frames, p.frames - a.start))
    if frames > p.frames - a.start:
        frames -= 8
    size = media.work_size(p.width, p.height, a.long_side)
    work = run / "work.mp4"
    info = media.prepare_work_clip(a.clip, work, a.start, frames, size)
    review = frames // 2
    client = Comfy(client_info="pickups")
    t0 = time.monotonic()
    job = client.submit(client.workflows.from_json(graphs.sdr_to_hdr(client.assets.from_file(work), info["fps"],
                                                                         review_frame=review, exr=a.exr, prefix=f"pickups/{a.name}")))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    job = job.result()
    secs = round(time.monotonic() - t0, 1)
    out = run / "out"
    out.mkdir(exist_ok=True)
    names = []
    for o in job.outputs:
        o.to_file(out / o.name)
        names.append(o.name)

    # Comparison: SDR original vs HDR at 0, -2, -4 EV for the review frame.
    src_png = run / "src_review.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"select=eq(n\\,{review})", "-frames:v", "1", str(src_png)], check=True)
    sdr = np.asarray(Image.open(src_png).convert("RGB"))
    prev = sorted(out.glob("preview_*.png"))
    hdr0 = np.asarray(Image.open(prev[review]).convert("RGB")) if len(prev) > review else sdr
    ev2 = np.asarray(Image.open(sorted(out.glob("ev_m2_*.png"))[0]).convert("RGB"))
    ev4 = np.asarray(Image.open(sorted(out.glob("ev_m4_*.png"))[0]).convert("RGB"))
    tiles = [("SDR original", sdr), ("HDR (tonemapped)", hdr0),
             ("SDR pulled -2 stops", sdr_pull(sdr, -2)), ("HDR pulled -2 stops", ev2),
             ("SDR pulled -4 stops", sdr_pull(sdr, -4)), ("HDR pulled -4 stops", ev4)]
    w, h = 960, round(960 * sdr.shape[0] / sdr.shape[1])
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (w * 2 + 4, h * 3 + 8), (10, 10, 12))
    for i, (label, arr) in enumerate(tiles):
        im = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w, 38], fill=(0, 0, 0))
        d.text((10, 6), label, fill=(255, 220, 0), font=font)
        sheet.paste(im, ((i % 2) * (w + 4), (i // 2) * (h + 4)))
    sheet.save(run / "hdr_compare.png")
    hlg = [n for n in names if n.startswith("hlg") and n.endswith(".mp4")]
    hlg_info = None
    if hlg:
        hlg_info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=codec_name,pix_fmt,color_transfer,color_primaries,color_space,width,height,r_frame_rate,nb_frames",
                                   "-of", "compact", str(out / hlg[0])], capture_output=True, text=True).stdout.strip()
    rep = {"name": a.name, "size": list(size), "frames": frames, "fps": info["fps"], "seconds": secs, "outputs": len(names),
           "hlg": hlg, "hlg_probe": hlg_info}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
