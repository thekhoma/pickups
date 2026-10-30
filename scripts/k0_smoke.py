"""K0: prove the Comfy SDK -> comfy-api-proxy -> ComfyUI GPU path with a tiny LTX-2.5 job.

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/k0_smoke.py
"""

import json
import os
import sys
import time
from pathlib import Path

from comfy_sdk import Comfy, JobFailed, Progress, StatusChange

from pickups import graphs

OUT = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs" / "k0"


def main() -> int:
    if not os.environ.get("COMFY_BASE_URL"):
        print("Set COMFY_BASE_URL (e.g. http://127.0.0.1:8189 for the local proxy).", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    wf = client.workflows.from_json(
        graphs.smoke_t2v("A slow dolly shot across a sunlit wooden table with a coffee cup, natural light.", seed=7)
    )
    t0 = time.monotonic()
    job = client.submit(wf)
    print("submitted", job.id)
    for ev in job.events():
        match ev:
            case Progress() as p:
                print(f"  {p.value:.0%} {p.message or ''}".rstrip())
            case StatusChange(status=s) if s in ("succeeded", "failed", "canceled", "expired"):
                break
    try:
        job = job.result()
    except JobFailed as e:
        print("FAILED:", e.error)
        return 1
    elapsed = time.monotonic() - t0
    frames = job.get_outputs("PU_Out")
    audio = job.get_outputs("PU_OutAudio")
    for o in frames + audio:
        o.to_file(OUT / o.name)
    report = {"job": job.id, "seconds": round(elapsed, 1), "frames": len(frames), "audio_files": len(audio)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
