"""C0 variant: SAM select -> box-shaped mask (union over the shot, padded) -> LTX fill -> composite.

A mask shaped like the object invites the model to redraw the object; a padded box does not.
Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/c0_box.py --clip SRC --label "red coffee mug" \
         --fill-prompt "..." --name c0_box [--extra-label steam] [--pad 24] [--seed 42]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from comfy_sdk import Comfy, JobFailed, StatusChange

from pickups import graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"


def run_job(client, graph):
    job = client.submit(client.workflows.from_json(graph))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    return job.result()


def frames_of(video: Path, out_dir: Path, pattern: str = "s_%04d.png") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), str(out_dir / pattern)], check=True)
    return sorted(out_dir.glob(pattern.split("%")[0] + "*.png"))


def sam_masks(client, clip: Path, label: str, run: Path, tag: str) -> list[np.ndarray]:
    job = run_job(client, graphs.select(client.assets.from_file(clip), label, prefix=f"pickups/{run.name}_{tag}"))
    d = run / f"sam_{tag}"
    d.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(d / o.name)
    return [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(d.glob("mask_*.png"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--extra-label", default=None)
    ap.add_argument("--fill-prompt", required=True)
    ap.add_argument("--name", default="c0_box")
    ap.add_argument("--pad", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feather", type=float, default=6.0)
    ap.add_argument("--plate-instruction", default=None, help="Qwen Image Edit instruction for a clean first frame")
    ap.add_argument("--plate-strength", type=float, default=0.7)
    a = ap.parse_args()

    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    p = media.probe(a.clip)
    t0 = time.monotonic()
    masks = sam_masks(client, a.clip, a.label, run, "a")
    if a.extra_label:
        extra = sam_masks(client, a.clip, a.extra_label, run, "b")
        masks = [m | e for m, e in zip(masks, extra)]
    union = np.any(np.stack(masks), axis=0)
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        print(json.dumps({"error": "nothing found", "label": a.label}))
        return 1
    x0, x1 = max(xs.min() - a.pad, 0), min(xs.max() + a.pad, p.width - 1)
    y0, y1 = max(ys.min() - a.pad, 0), min(ys.max() + a.pad, p.height - 1)
    box = np.zeros((p.height, p.width), dtype=np.uint8)
    box[y0:y1 + 1, x0:x1 + 1] = 255
    mdir = run / "boxmask"
    mdir.mkdir(exist_ok=True)
    for i in range(p.frames):
        Image.fromarray(np.dstack([box, box, box])).save(mdir / f"m_{i + 1:04d}.png")
    mask_clip = run / "mask.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(float(p.fps)), "-i", str(mdir / "m_%04d.png"),
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(mask_clip)], check=True)
    select_s = time.monotonic() - t0

    plate_asset, plate_s, plate_img = None, 0.0, None
    if a.plate_instruction:
        tp = time.monotonic()
        first = run / "first.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.clip), "-frames:v", "1", str(first)], check=True)
        pj = run_job(client, graphs.clean_plate(client.assets.from_file(first), a.plate_instruction, seed=a.seed,
                                                prefix=f"pickups/{a.name}_plate"))
        raw = run / "plate_raw.png"
        pj.get_outputs("PU_Plate")[0].to_file(raw)
        plate_img = Image.open(raw).convert("RGB").resize((p.width, p.height), Image.LANCZOS)
        plate_path = run / "plate.png"
        plate_img.save(plate_path)
        plate_asset = client.assets.from_file(plate_path)
        plate_s = time.monotonic() - tp

    t1 = time.monotonic()
    try:
        job = run_job(client, graphs.remove(client.assets.from_file(a.clip), "", float(p.fps), seed=a.seed, dilate=0,
                                            prompt=a.fill_prompt, prefix=f"pickups/{a.name}",
                                            mask_clip=client.assets.from_file(mask_clip),
                                            plate=plate_asset, plate_strength=a.plate_strength))
    except JobFailed as e:
        print("FAILED:", e.error)
        return 1
    fill_s = time.monotonic() - t1
    gdir = run / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)

    src = frames_of(a.clip, run / "src")
    gen = sorted(gdir.glob("frame_*.png"))
    soft = np.asarray(Image.fromarray(box).filter(ImageFilter.GaussianBlur(a.feather))).astype(np.float32) / 255.0
    soft[box > 0] = 1.0
    alpha = soft[..., None]
    cdir = run / "comp"
    cdir.mkdir(exist_ok=True)
    changed = 0
    for i, (s_, g_) in enumerate(zip(src, gen)):
        s = np.asarray(Image.open(s_).convert("RGB")).astype(np.float32)
        g = np.asarray(Image.open(g_).convert("RGB")).astype(np.float32)
        out = np.clip(np.rint(s * (1 - alpha) + g * alpha), 0, 255).astype(np.uint8)
        changed += int(np.count_nonzero(np.any(out[soft == 0] != s[soft == 0].astype(np.uint8), axis=-1)))
        Image.fromarray(out).save(cdir / f"c_{i + 1:04d}.png")
    n = len(gen)
    picks = [0, n // 2, n - 1]
    w, h = p.width, p.height
    cols = 3 if plate_img is not None else 2
    sheet = Image.new("RGB", (w * cols, h * 3))
    for r, i in enumerate(picks):
        sheet.paste(Image.open(src[i]).convert("RGB"), (0, r * h))
        if plate_img is not None:
            s_ = np.asarray(Image.open(src[i]).convert("RGB")).astype(np.float32)
            pl = np.asarray(plate_img).astype(np.float32)
            sheet.paste(Image.fromarray(np.clip(np.rint(s_ * (1 - alpha) + pl * alpha), 0, 255).astype(np.uint8)), (w, r * h))
        sheet.paste(Image.open(cdir / f"c_{i + 1:04d}.png"), ((cols - 1) * w, r * h))
    sheet.save(run / "sheet.png")
    rep = {"name": a.name, "box": [int(x0), int(y0), int(x1), int(y1)], "box_coverage": round(float(box.mean() / 255), 4),
           "seconds_select": round(select_s, 1), "seconds_plate": round(plate_s, 1), "seconds_fill": round(fill_s, 1), "frames": n,
           "outside_mask_pixels_changed_pre_encode": changed}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
