"""Swap, local: replace only the person. Everything outside the person matte stays the original footage.

1. SAM 3.1 finds the person over the shot -> one crop box (padded) covering them in every frame.
2. SCAIL-2 runs on that crop only, at its working size (so the character gets more pixels than a whole-frame run).
3. SAM 3.1 tracks the original person and the new person in the crop; their union (grown + feathered) is the matte.
4. Only matte pixels are pasted back onto the full-resolution frames (colour-matched to their surroundings).

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/swap_local.py --clip SRC --ref-image ref.png \
      --prompt "A woman in a yellow raincoat walks..." --name swapL_103 [--long-side 1920] [--frames 81]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_sdk import Comfy, StatusChange

from pickups import composite, graphs, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"
SCAIL_LONG = 832  # SCAIL-2 working long side for the crop (multiple of 32)


def run_job(client, graph):
    job = client.submit(client.workflows.from_json(graph))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    return job.result()


def sam_masks(client, clip: Path, label: str, out_dir: Path, prefix: str) -> list[np.ndarray]:
    job = run_job(client, graphs.select(client.assets.from_file(clip), label, prefix=prefix))
    out_dir.mkdir(parents=True, exist_ok=True)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(out_dir / o.name)
    return [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(out_dir.glob("mask_*.png"))]


def frames_of(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), str(out_dir / "f_%04d.png")], check=True)
    return sorted(out_dir.glob("f_*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--ref-image", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", default="person")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--pad", type=float, default=0.35, help="crop padding as a fraction of the person box")
    ap.add_argument("--grow", type=int, default=10)
    ap.add_argument("--feather", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    timing = {}

    p = media.probe(a.clip)
    frames = min(a.frames, p.frames - a.start)
    frames = 4 * ((frames - 1) // 4) + 1  # Wan wants 4k+1
    W, H = media.work_size(p.width, p.height, a.long_side)
    work = run / "work.mp4"
    info = media.prepare_work_clip(a.clip, work, a.start, frames, (W, H))
    fps = info["fps"]

    # 1. Where is the person over the whole shot? (SAM on a small proxy for speed)
    t = time.monotonic()
    pw, ph = media.work_size(p.width, p.height, 768)
    proxy = run / "proxy.mp4"
    media.prepare_work_clip(a.clip, proxy, a.start, frames, (pw, ph))
    pm = sam_masks(client, proxy, a.label, run / "sam_proxy", f"pickups/{a.name}_proxy")
    union = np.any(np.stack(pm), axis=0)
    if not union.any():
        print(json.dumps({"error": "nobody found", "label": a.label}))
        return 1
    ys, xs = np.nonzero(union)
    sx, sy = W / pw, H / ph
    bx0, bx1, by0, by1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    bw, bh = bx1 - bx0, by1 - by0
    cx0 = int(max(0, bx0 - bw * a.pad)); cx1 = int(min(W, bx1 + bw * a.pad))
    cy0 = int(max(0, by0 - bh * a.pad)); cy1 = int(min(H, by1 + bh * a.pad))
    cw, ch = (cx1 - cx0) // 2 * 2, (cy1 - cy0) // 2 * 2
    timing["find"] = round(time.monotonic() - t, 1)

    # 2. SCAIL-2 on the crop only.
    scale = SCAIL_LONG / max(cw, ch)
    sw, sh = max(32, round(cw * scale / 32) * 32), max(32, round(ch * scale / 32) * 32)
    crop = run / "crop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"crop={cw}:{ch}:{cx0}:{cy0},scale={sw}:{sh}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", str(crop)], check=True)
    t = time.monotonic()
    job = run_job(client, graphs.scail2(client.assets.from_file(crop), client.assets.from_file(a.ref_image), a.prompt,
                                        replacement=True, seed=a.seed, prefix=f"pickups/{a.name}"))
    timing["scail2"] = round(time.monotonic() - t, 1)
    gdir = run / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)
    gen_clip = run / "gen_crop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(gdir / "frame_*.png"),
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", str(gen_clip)], check=True)

    # 3. Matte = original person + new person (in crop space).
    t = time.monotonic()
    m_orig = sam_masks(client, crop, a.label, run / "sam_orig", f"pickups/{a.name}_morig")
    m_new = sam_masks(client, gen_clip, a.label, run / "sam_new", f"pickups/{a.name}_mnew")
    timing["matte"] = round(time.monotonic() - t, 1)

    # 4. Paste only the matte back onto the full-resolution frames.
    src = frames_of(work, run / "src")
    gen = sorted(gdir.glob("frame_*.png"))
    cdir = run / "comp"
    cdir.mkdir(exist_ok=True)
    changed, areas = 0, []
    for i, sf in enumerate(src):
        full = np.asarray(Image.open(sf).convert("RGB"))
        if i >= len(gen):
            Image.fromarray(full).save(cdir / f"c_{i + 1:04d}.png")
            continue
        m = (m_orig[i] if i < len(m_orig) else np.zeros((sh, sw), bool)) | (m_new[i] if i < len(m_new) else np.zeros((sh, sw), bool))
        m_full = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((cw, ch), Image.NEAREST)) > 127
        matte_crop = composite.soft_matte(m_full, a.grow, a.feather)
        g = np.asarray(Image.open(gen[i]).convert("RGB").resize((cw, ch), Image.LANCZOS))
        region = full[cy0:cy0 + ch, cx0:cx0 + cw]
        g = composite.harmonize(g, region, matte_crop)
        pasted, ch_ = composite.paste(region, g, matte_crop)
        out = full.copy()
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        changed += ch_
        areas.append(float((matte_crop > 0).sum()) / (W * H))
        Image.fromarray(out).save(cdir / f"c_{i + 1:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(cdir / "c_%04d.png"),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(run / "result.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-i", str(run / "result.mp4"), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2,scale=2400:-2", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(run / "side_by_side.mp4")], check=True)
    n = len(src)
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * 2, th * 3))
    for r, i in enumerate([0, n // 2, n - 1]):
        sheet.paste(Image.open(src[i]).convert("RGB").resize((tw, th)), (0, r * th))
        sheet.paste(Image.open(cdir / f"c_{i + 1:04d}.png").resize((tw, th)), (tw, r * th))
    sheet.save(run / "sheet.png")
    rep = {"name": a.name, "size": [W, H], "crop": [cx0, cy0, cw, ch], "scail_size": [sw, sh], "frames": n, "fps": fps,
           "matte_area_mean": round(float(np.mean(areas)), 4) if areas else 0, "seconds": timing,
           "outside_matte_pixels_changed_pre_encode": changed}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
