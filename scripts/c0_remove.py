"""C0/C1: Remove an object by name, then paste only the masked area back over the source frames.

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/c0_remove.py --make-test-clip --label "red coffee mug" --name c0
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/c0_remove.py --clip shot.mov --label "boom microphone" --name c1_boom

Writes source/generated/composited frames, masks, a before/after sheet and report.json to ~/.pickups/runs/<name>/.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from comfy_sdk import Comfy, JobFailed, Progress, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"

TEST_PROMPT = ("Static locked-off shot of a tidy wooden desk by a window, a red coffee mug sits on the desk next to "
               "an open notebook, soft daylight, gentle steam rising from the mug, no people.")


def run_job(client: Comfy, graph: dict):
    job = client.submit(client.workflows.from_json(graph))
    print("submitted", job.id)
    for ev in job.events():
        match ev:
            case Progress():
                pass
            case StatusChange(status=s) if s in ("succeeded", "failed", "canceled", "expired"):
                break
    return job.result()


def make_test_clip(client: Comfy, run: Path, w: int, h: int, frames: int, fps: float) -> Path:
    src_dir = run / "src_gen"
    if len(list(src_dir.glob("frame_*.png"))) != frames:
        job = run_job(client, graphs.smoke_t2v(TEST_PROMPT, seed=11, width=w, height=h, frames=frames, fps=fps,
                                               prefix="pickups/c0_src"))
        src_dir.mkdir(parents=True, exist_ok=True)
        for o in job.get_outputs("PU_Out") + job.get_outputs("PU_OutAudio"):
            o.to_file(src_dir / o.name)
    audio = next(src_dir.glob("audio_*"))
    clip = run / "source.mp4"
    # Keep every frame: pad the audio to the video length rather than cutting video to the audio.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(src_dir / "frame_*.png"),
                    "-i", str(audio), "-filter_complex", f"[1:a]apad,atrim=end={frames / fps:.6f}[a]", "-map", "0:v", "-map", "[a]",
                    "-frames:v", str(frames), "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], check=True)
    return clip


def frames_of(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), str(out_dir / "s_%04d.png")], check=True)
    return sorted(out_dir.glob("s_*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--make-test-clip", action="store_true")
    ap.add_argument("--label", required=True)
    ap.add_argument("--name", default="c0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--feather", type=float, default=4.0)
    ap.add_argument("--fill-prompt", default=None, help="What the masked area should become")
    ap.add_argument("--avoid", default="", help="Comma list of things that must not appear in the fill")
    args = ap.parse_args()

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    t_all = time.monotonic()
    clip = make_test_clip(client, run, 768, 448, 49, 24.0) if args.make_test_clip else args.clip
    if not clip:
        ap.error("pass --clip or --make-test-clip")
    p = media.probe(clip)
    if p.width % 32 or p.height % 32 or p.frames != media.ltx_frames(p.frames):
        raise SystemExit(f"C0 expects multiples of 32 and 8k+1 frames; got {p.width}x{p.height}, {p.frames} frames")

    t0 = time.monotonic()
    try:
        job = run_job(client, graphs.remove(client.assets.from_file(clip), args.label, float(p.fps), seed=args.seed,
                                             dilate=args.dilate, prefix=f"pickups/{args.name}",
                                             **({"prompt": args.fill_prompt} if args.fill_prompt else {}),
                                             negative=", ".join(filter(None, [args.avoid, graphs.DEFAULT_NEGATIVE]))))
    except JobFailed as e:
        print("FAILED:", e.error)
        (run / "report.json").write_text(json.dumps({"failed": str(e.error)}, indent=2))
        return 1
    gen_seconds = time.monotonic() - t0

    gen_dir, mask_dir, comp_dir = run / "gen", run / "mask", run / "comp"
    for d in (gen_dir, mask_dir, comp_dir):
        d.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gen_dir / o.name)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(mask_dir / o.name)

    src_frames = frames_of(clip, run / "src")
    gen_frames = sorted(gen_dir.glob("frame_*.png"))
    mask_frames = sorted(mask_dir.glob("mask_*.png"))
    n = min(len(src_frames), len(gen_frames), len(mask_frames))
    outside_changed = 0
    coverage = []
    for i in range(n):
        s = np.asarray(Image.open(src_frames[i]).convert("RGB")).astype(np.float32)
        g = np.asarray(Image.open(gen_frames[i]).convert("RGB").resize((s.shape[1], s.shape[0]), Image.LANCZOS)).astype(np.float32)
        m_img = Image.open(mask_frames[i]).convert("L").resize((s.shape[1], s.shape[0]), Image.BILINEAR)
        hard = np.asarray(m_img) > 127
        soft = np.asarray(m_img.filter(ImageFilter.GaussianBlur(args.feather))).astype(np.float32) / 255.0
        soft[hard] = 1.0
        a = soft[..., None]
        c = s * (1 - a) + g * a
        out = np.clip(np.rint(c), 0, 255).astype(np.uint8)
        outside = soft == 0
        outside_changed += int(np.count_nonzero(np.any(out[outside] != s[outside].astype(np.uint8), axis=-1)))
        coverage.append(float(hard.mean()))
        Image.fromarray(out).save(comp_dir / f"c_{i + 1:04d}.png")

    # Before/after sheet: source, mask, composite at 3 moments.
    picks = [0, n // 2, n - 1]
    tiles = []
    for i in picks:
        row = [Image.open(src_frames[i]).convert("RGB"), Image.open(mask_frames[i]).convert("RGB").resize(Image.open(src_frames[i]).size),
               Image.open(comp_dir / f"c_{i + 1:04d}.png")]
        tiles.append(row)
    tw, th = tiles[0][0].size
    sheet = Image.new("RGB", (tw * 3, th * len(tiles)), (20, 20, 23))
    for r, row in enumerate(tiles):
        for c_, im in enumerate(row):
            sheet.paste(im, (c_ * tw, r * th))
    sheet.save(run / "sheet.png")

    report = {
        "job": job.id,
        "label": args.label,
        "seconds_generate": round(gen_seconds, 1),
        "seconds_total": round(time.monotonic() - t_all, 1),
        "size": [p.width, p.height],
        "frames": n,
        "mask_coverage_mean": round(float(np.mean(coverage)), 4) if coverage else 0.0,
        "frames_with_detection": int(sum(1 for v in coverage if v > 0)),
        "outside_mask_pixels_changed_pre_encode": outside_changed,
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
