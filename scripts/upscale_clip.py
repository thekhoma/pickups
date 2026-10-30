"""Upscale a clip with SeedVR2 (one-step 1.4B, FL pack) in chunks, and write a near-lossless master.

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/upscale_clip.py --clip SRC --name up_clip --scale 2 --frames 89
Writes ~/.pickups/runs/<name>/upscaled.mov (ProRes 422 HQ) and report.json.
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
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = to the end")
    ap.add_argument("--chunk", type=int, default=24, help="frames per job (RAM/VRAM)")
    ap.add_argument("--engine", choices=["fl", "native", "esrgan", "rtx"], default="fl",
                    help="fl: FL pack, frame by frame; native: ComfyUI SeedVR2 nodes, temporal chunks (whole clip per job)")
    ap.add_argument("--model", default=None, help="native: SeedVR2 diffusion model file (default: the 1.4B distill)")
    a = ap.parse_args()
    run = RUNS / a.name
    out = run / "frames"
    out.mkdir(parents=True, exist_ok=True)
    p = media.probe(a.clip)
    n = (p.frames - a.start) if not a.frames else min(a.frames, p.frames - a.start)
    client = Comfy(client_info="pickups")
    t = time.monotonic()
    k = 0
    for s in range(a.start, a.start + n, a.chunk):
        m = min(a.chunk, a.start + n - s)
        part = run / f"part_{s:05d}.mp4"
        media.prepare_work_clip(a.clip, part, s, m, (p.width // 2 * 2, p.height // 2 * 2))
        if a.engine == "rtx":
            graph = graphs.rtx_upscale(client.assets.from_file(part), a.scale, prefix=f"pickups/{a.name}")
        elif a.engine == "esrgan":
            graph = graphs.gan_upscale(client.assets.from_file(part), round(p.width * a.scale) // 2 * 2, round(p.height * a.scale) // 2 * 2,
                                       prefix=f"pickups/{a.name}")
        elif a.engine == "native":
            graph = graphs.seedvr2_video(client.assets.from_file(part), a.scale, prefix=f"pickups/{a.name}",
                                         **({"model": a.model} if a.model else {}))
        else:
            graph = graphs.seedvr2_upscale(client.assets.from_file(part), a.scale, prefix=f"pickups/{a.name}")
        job = run_job(client, graph)
        for o in job.get_outputs("PU_Out"):
            k += 1
            o.to_file(out / f"u_{k:05d}.png")
        part.unlink()
    secs = round(time.monotonic() - t, 1)
    fps = f"{p.fps.numerator}/{p.fps.denominator}"
    master = run / "upscaled.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-i", str(out / "u_%05d.png"), "-c:v", "prores_ks", "-profile:v", "3",
                    "-pix_fmt", "yuv422p10le", str(master)], check=True)
    size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(master)],
                          capture_output=True, text=True).stdout.strip()
    rep = {"name": a.name, "engine": a.engine, "frames": k, "seconds": secs, "sec_per_frame": round(secs / max(k, 1), 2), "size": size, "master": str(master)}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
