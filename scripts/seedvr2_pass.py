"""Sharpness test 3: run SeedVR2 on the AI frames of an Extend run (frames from the seam on).

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/seedvr2_pass.py ~/.pickups/runs/k1_57 --scale 2
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from comfy_sdk import Comfy, StatusChange

from pickups import graphs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    rep = json.loads((a.run / "report.json").read_text())
    seam, fps = rep["seam_frame"], rep["fps"]
    gen = sorted((a.run / "out").glob("frame_*.png"))[seam:]
    tmp = a.run / "ai_only"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(gen):
        (tmp / f"a_{i:04d}.png").write_bytes(f.read_bytes())
    clip = a.run / "ai_only.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "a_%04d.png"),
                    "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", "-an", str(clip)], check=True)
    client = Comfy(client_info="pickups")
    t0 = time.monotonic()
    job = client.submit(client.workflows.from_json(graphs.seedvr2_upscale(client.assets.from_file(clip), a.scale,
                                                                             prefix=f"pickups/{a.run.name}_svr")))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    job = job.result()
    out = a.run / "seedvr2"
    out.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(out / o.name)
    print(json.dumps({"frames": len(list(out.glob("frame_*.png"))), "seconds": round(time.monotonic() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
