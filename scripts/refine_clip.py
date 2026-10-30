"""Refine-Details pass: rebuild fine texture on a clip (e.g. a Layout-To-Render result) with LTX-2.5's Refine-Details IC-LoRA.

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/refine_clip.py --clip result.mp4 --prompt "..." --name refine_sh010 [--size 1920x1088]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from comfy_sdk import Comfy, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="", help="WxH canvas, multiples of 32 (default: the clip's size)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = RUNS / a.name
    out = run / "out"
    out.mkdir(parents=True, exist_ok=True)
    p = media.probe(a.clip)
    w, h = (int(v) for v in a.size.split("x")) if a.size else (p.width // 32 * 32, p.height // 32 * 32)
    frames = media.ltx_frames(p.frames)
    work = run / "work.mp4"
    info = media.prepare_work_clip(a.clip, work, 0, frames, (w, h))
    client = Comfy(client_info="pickups")
    t = time.monotonic()
    job = client.submit(client.workflows.from_json(graphs.refine_details(client.assets.from_file(work), a.prompt, info["fps"], w, h,
                                                                         seed=a.seed, prefix=f"pickups/{a.name}")))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    res = job.result()
    for o in res.get_outputs("PU_Out"):
        o.to_file(out / o.name)
    secs = round(time.monotonic() - t, 1)
    fps = str(info["fps"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-pattern_type", "glob", "-i", str(out / "frame_*.png"),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(run / "result.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-i", str(run / "result.mp4"), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2,scale=2400:-2", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(run / "side_by_side.mp4")], check=True)
    rep = {"name": a.name, "size": [w, h], "frames": frames, "seconds": secs}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
