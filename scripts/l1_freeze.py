"""L1/K1: run the Extend retake graph on a clip and measure what stayed frozen.

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/l1_freeze.py --synthetic
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/l1_freeze.py --clip /path/to/clip.mov --name k1_clip1

Writes frames, audio, the canvas, the submitted graph and report.json to ~/.pickups/runs/<name>/.
Pass: context frames decode with PSNR > 30 dB against the canvas; the first clearly changed frame is
at (or within one latent step of) the seam.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_sdk import Comfy, JobFailed, Progress, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"


def synthetic_source(path: Path, fps: str = "24", seconds: float = 3.0) -> None:
    """Test pattern with a frame counter and a 1 kHz beep every 0.5 s."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate={fps}:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate=48000:duration={seconds}",
        "-filter_complex", "[1:a]volume='if(lt(mod(t,0.5),0.1),1,0)':eval=frame,pan=stereo|c0=c0|c1=c0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True)


def frames_of(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), str(out_dir / "c_%04d.png")], check=True)
    return sorted(out_dir.glob("c_*.png"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--name", default="l1")
    ap.add_argument("--context", type=int, default=25)
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--two-stage", action="store_true", help="half-size draft, 2x latent upscale, full-size refine")
    ap.add_argument("--detail", action="store_true", help="with --two-stage: LTX DFR detailing IC-LoRA in pass 2")
    ap.add_argument("--dev", action="store_true", help="with --two-stage: full dev model (30 steps, CFG 3) for pass 1")
    ap.add_argument("--prompt", default="The shot continues naturally. Same camera move, same lighting. No cuts, no new objects.")
    args = ap.parse_args()

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    src = args.clip
    if args.synthetic:
        src = run / "source.mp4"
        synthetic_source(src)
    if not src:
        ap.error("pass --clip or --synthetic")

    p = media.probe(src)
    size = media.work_size(p.width, p.height, args.long_side)
    canvas = run / "canvas.mp4"
    timing = media.build_extend_canvas(src, canvas, args.context, args.new, size)

    client = Comfy(client_info="pickups")
    if args.two_stage:
        w, h = timing["size"]
        if w % 64 or h % 64:
            raise SystemExit(f"two-stage needs multiples of 64, got {w}x{h}")
        graph = graphs.retake_two_stage(client.assets.from_file(canvas), timing["fps"], timing["start_time"],
                                        timing["end_time"], args.context, w, h, args.prompt, seed=args.seed,
                                        prefix=f"pickups/{args.name}", detail=args.detail,
                                        dev_first_pass=args.dev)
    else:
        graph = graphs.retake(client.assets.from_file(canvas), timing["fps"], timing["start_time"], timing["end_time"],
                              args.prompt, seed=args.seed, prefix=f"pickups/{args.name}")
    wf = client.workflows.from_json(graph)
    t0 = time.monotonic()
    job = client.submit(wf)
    (run / "graph.json").write_text(json.dumps({k: v for k, v in wf.json.items()}, default=str, indent=1))
    print("submitted", job.id)
    for ev in job.events():
        match ev:
            case Progress() as pr:
                print(f"  {pr.value:.0%}")
            case StatusChange(status=s) if s in ("succeeded", "failed", "canceled", "expired"):
                break
    try:
        job = job.result()
    except JobFailed as e:
        print("FAILED:", e.error)
        (run / "report.json").write_text(json.dumps({"job": job.id, "failed": str(e.error)}, indent=2))
        return 1
    elapsed = time.monotonic() - t0

    out_dir = run / "out"
    out_dir.mkdir(exist_ok=True)
    outs = job.get_outputs("PU_Out")
    for o in outs:
        o.to_file(out_dir / o.name)
    for o in job.get_outputs("PU_OutAudio"):
        o.to_file(run / o.name)

    gen = [np.asarray(Image.open(f).convert("RGB")) for f in sorted(out_dir.glob("frame_*.png"))]
    ref = [np.asarray(Image.open(f).convert("RGB")) for f in frames_of(canvas, run / "canvas_frames")]
    n = min(len(gen), len(ref))
    per_frame = [round(psnr(gen[i], ref[i]), 2) for i in range(n)]
    ctx = per_frame[: args.context]
    # Context frames are a VAE round trip of the canvas (the app pastes originals back, so their
    # absolute PSNR only bounds codec loss). The seam is the first frame that drops clearly below
    # the context level, since the canvas tail is a held frame and generation should move.
    finite = [v for v in ctx if v != float("inf")]
    ctx_level = float(np.median(finite)) if finite else 99.0
    threshold = ctx_level - 3.0
    first_changed = next((i for i, v in enumerate(per_frame) if v < threshold), None)
    report = {
        "job": job.id,
        "seconds": round(elapsed, 1),
        "size": timing["size"],
        "fps": timing["fps"],
        "frames_out": len(gen),
        "frames_canvas": len(ref),
        "seam_frame": args.context,
        "context_psnr_min": min(ctx) if ctx else None,
        "context_psnr_mean": round(float(np.mean([v for v in ctx if v != float("inf")])), 2) if ctx else None,
        "first_changed_frame": first_changed,
        "per_frame_psnr": per_frame,
        "context_level_db": round(ctx_level, 2),
        "pass": bool(ctx and min(ctx) > threshold and first_changed is not None and abs(first_changed - args.context) <= 1),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_frame_psnr"}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
