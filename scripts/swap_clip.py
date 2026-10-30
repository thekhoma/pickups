"""Swap (SCAIL-2 replacement) or Animate (SCAIL-2 animation) test on one clip.

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/swap_clip.py --clip SRC --name swap_103 \
      --ref-prompt "Full-body photo of a fictional woman in a yellow raincoat..." --prompt "A woman in a yellow raincoat walks..."
  (or --ref-image path/to/character.png instead of --ref-prompt)
  --mode animate makes the reference character perform the clip's motion instead of replacing the person in place.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from comfy_sdk import Comfy, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"


def run_job(client, graph):
    job = client.submit(client.workflows.from_json(graph))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    return job.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref-image", type=Path)
    ap.add_argument("--ref-prompt")
    ap.add_argument("--prompt", required=True, help="what the output shot shows (scene + new character)")
    ap.add_argument("--mode", choices=["replace", "animate"], default="replace")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--long-side", type=int, default=896)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    timing = {}

    ref = a.ref_image
    if ref is None:
        if not a.ref_prompt:
            ap.error("pass --ref-image or --ref-prompt")
        t = time.monotonic()
        job = run_job(client, graphs.still(a.ref_prompt, 832, 1216, seed=a.seed, prefix=f"pickups/{a.name}_ref"))
        ref = run / "reference.png"
        job.get_outputs("PU_Out")[0].to_file(ref)
        timing["reference"] = round(time.monotonic() - t, 1)

    p = media.probe(a.clip)
    frames = min(a.frames, p.frames - a.start)
    frames = 4 * ((frames - 1) // 4) + 1  # Wan wants 4k+1
    size = media.work_size(p.width, p.height, a.long_side)
    work = run / "work.mp4"
    info = media.prepare_work_clip(a.clip, work, a.start, frames, size)

    t = time.monotonic()
    job = run_job(client, graphs.scail2(client.assets.from_file(work), client.assets.from_file(ref), a.prompt,
                                        replacement=(a.mode == "replace"), seed=a.seed, prefix=f"pickups/{a.name}"))
    timing["scail2"] = round(time.monotonic() - t, 1)
    out = run / "out"
    out.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(out / o.name)
    outv = run / "result.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(info["fps"]), "-pattern_type", "glob",
                    "-i", str(out / "frame_*.png"), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(outv)], check=True)
    # Side by side: reference | original | result.
    refv = run / "ref_still.mp4"
    w, h = size
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(ref), "-t", str(frames / info["fps"]),
                    "-vf", f"scale=-2:{h},pad=ceil(iw/2)*2:{h}", "-r", str(info["fps"]), "-pix_fmt", "yuv420p", str(refv)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(refv), "-i", str(work), "-i", str(outv), "-filter_complex",
                    "[0:v][1:v][2:v]hstack=inputs=3", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(run / "side_by_side.mp4")], check=True)
    n = len(list(out.glob("frame_*.png")))
    sheet = Image.new("RGB", (w * 2, h * 3))
    srcdir = run / "src"
    srcdir.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), str(srcdir / "s_%04d.png")], check=True)
    gen = sorted(out.glob("frame_*.png"))
    src = sorted(srcdir.glob("s_*.png"))
    for r, i in enumerate([0, n // 2, n - 1]):
        sheet.paste(Image.open(src[i]).convert("RGB"), (0, r * h))
        sheet.paste(Image.open(gen[i]).convert("RGB").resize((w, h)), (w, r * h))
    sheet.save(run / "sheet.png")
    rep = {"name": a.name, "mode": a.mode, "size": [w, h], "frames": frames, "fps": info["fps"], "frames_out": n, "seconds": timing}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
