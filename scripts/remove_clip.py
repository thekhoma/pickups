"""Remove on a real clip, at full resolution.

1. SAM 3.1 finds the object (+ optional attached shadow) on a small proxy; masks are scaled up to the work size.
2. Everything generative runs on a CROP around the object (object box + context), so the fill gets the crop's
   pixel density instead of a whole frame squeezed into ~1 MP.
3. Qwen Image Edit makes a clean plate of the crop. Locked-off shot: paste the plate (with matched grain) on every
   frame. Moving shot: LTX in/outpaint fills the crop, starting from the plate.
4. Only the matte is pasted back onto the full-resolution frames; the seam is blended with a smooth correction field.

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/remove_clip.py --clip SRC --label "person" \
     --plate-instruction "Remove the person on the pier. Keep everything else exactly the same." \
     --fill-prompt "Empty wooden pier by calm water." --name c4_pier [--shadow] [--long-side 3840] [--quality]

Writes to ~/.pickups/runs/<name>/: work.mp4, plate_crop.png, comp frames, review.mp4, side_by_side.mp4,
zoom_compare.mp4 (the crop, before | after), sheet.png and report.json.
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
PROXY_LONG = 1024   # SAM runs here
LTX_LONG = 1024     # LTX fill runs on the crop at this long side (moving shots only)


def run_job(client, graph):
    job = client.submit(client.workflows.from_json(graph))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    return job.result()


def frames_of(video: Path, out_dir: Path, vf: str = "") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video)] + (["-vf", vf] if vf else []) + [str(out_dir / "f_%04d.png")],
                   check=True)
    return sorted(out_dir.glob("f_*.png"))


def sam(client, clip: Path, label: str, run: Path, tag: str, reuse: bool = False) -> list[np.ndarray]:
    d = run / f"sam_{tag}"
    if reuse and any(d.glob("mask_*.png")):
        return [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(d.glob("mask_*.png"))]
    job = run_job(client, graphs.select(client.assets.from_file(clip), label, prefix=f"pickups/{run.name}_{tag}"))
    d.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(d / o.name)
    return [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(d.glob("mask_*.png"))]


def upscale_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127


def encode(pattern: Path, fps: float, out: Path, crf: int = 14) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(pattern), "-c:v", "libx264", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--shadow", action="store_true", help="also find and remove the object's shadow")
    ap.add_argument("--plate-instruction", required=True)
    ap.add_argument("--fill-prompt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--long-side", type=int, default=0, help="work size; 0 = source resolution")
    ap.add_argument("--context", type=float, default=0.75, help="crop context on each side, as a fraction of the object box")
    ap.add_argument("--grow", type=float, default=10, help="matte growth in px at 1920 wide (scaled to the work size)")
    ap.add_argument("--feather", type=float, default=6.0, help="matte feather in px at 1920 wide")
    ap.add_argument("--quality", action="store_true", help="qwen2511 only: 40 steps instead of the 4-step Lightning LoRA")
    ap.add_argument("--editor", choices=["qwen2511", "qwen21", "qwen21-turbo"], default="qwen2511",
                    help="clean-plate model; Qwen-Image 2.1 edits near the crop's own resolution (up to ~2K)")
    ap.add_argument("--edit-res", type=int, default=2048, help="qwen21: pixel budget side (0 = the crop's own size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse", action="store_true", help="reuse this run's selection masks and plate (re-composite only)")
    ap.add_argument("--mode", choices=["auto", "plate", "ltx"], default="auto",
                    help="plate: paste the clean plate (locked-off shots); ltx: LTX fills the crop (moving shots)")
    a = ap.parse_args()

    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    timing = {}
    p = media.probe(a.clip)
    frames = media.ltx_frames(min(a.frames, p.frames - a.start))
    if frames > p.frames - a.start:
        frames -= 8
    W, H = (p.width // 2 * 2, p.height // 2 * 2) if not a.long_side else media.work_size(p.width, p.height, a.long_side)
    k = max(W, H) / 1920
    work = run / "work.mp4"
    if a.reuse and work.exists():
        info = {"fps": float(p.fps)}
    else:
        info = media.prepare_work_clip(a.clip, work, a.start, frames, (W, H))
    fps = info["fps"]

    # 1. Find the object on a proxy.
    t = time.monotonic()
    pw, ph = media.work_size(p.width, p.height, PROXY_LONG)
    proxy = run / "proxy.mp4"
    if not (a.reuse and proxy.exists()):
        media.prepare_work_clip(a.clip, proxy, a.start, frames, (pw, ph))
    pm = sam(client, proxy, a.label, run, "obj", a.reuse)
    pm_obj = list(pm)
    if a.shadow:
        shadows = sam(client, proxy, f"shadow of the {a.label}", run, "shadow", a.reuse)
        # Only shadow that touches the object counts; SAM's "shadow of X" also finds trees, buildings, etc.
        pm = [m | composite.attached_shadow(m, s) for m, s in zip(pm, shadows)]
    timing["select"] = round(time.monotonic() - t, 1)
    found = [bool(m.any()) for m in pm]
    if not any(found):
        print(json.dumps({"error": "nothing found", "label": a.label}))
        return 1

    # 2. Crop = object box over the whole shot + context, in work pixels.
    ys, xs = np.nonzero(np.any(np.stack(pm), axis=0))
    sx, sy = W / pw, H / ph
    bx0, bx1, by0, by1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    padx = max((bx1 - bx0) * a.context, 192 * k)
    pady = max((by1 - by0) * a.context, 192 * k)
    cx0, cx1 = int(max(0, bx0 - padx)), int(min(W, bx1 + padx))
    cy0, cy1 = int(max(0, by0 - pady)), int(min(H, by1 + pady))
    cw, ch = (cx1 - cx0) // 2 * 2, (cy1 - cy0) // 2 * 2
    crop_vf = f"crop={cw}:{ch}:{cx0}:{cy0}"

    # 3. Clean plate of the crop (first frame).
    t = time.monotonic()
    first = run / "first_crop.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", crop_vf, "-frames:v", "1", str(first)], check=True)
    if a.editor == "qwen2511":
        qw, qh = graphs.qwen_size(cw, ch)
        graph = graphs.clean_plate(client.assets.from_file(first), a.plate_instruction, seed=a.seed, fast=not a.quality,
                                   prefix=f"pickups/{a.name}_plate", size=(qw, qh))
    else:
        graph = graphs.qwen21_edit(client.assets.from_file(first), a.plate_instruction, seed=a.seed, turbo=a.editor == "qwen21-turbo",
                                   resolution=a.edit_res, prefix=f"pickups/{a.name}_plate")
    if not (a.reuse and (run / "plate_raw.png").exists()):
        pj = run_job(client, graph)
        pj.get_outputs("PU_Plate")[0].to_file(run / "plate_raw.png")
    plate = Image.open(run / "plate_raw.png").convert("RGB")
    qw, qh = plate.size
    plate = plate.resize((cw, ch), Image.LANCZOS)
    plate.save(run / "plate_crop.png")
    timing["plate"] = round(time.monotonic() - t, 1)

    src = frames_of(work, run / "src")
    src_crop = frames_of(work, run / "src_crop", crop_vf)
    sa = [np.asarray(Image.open(f).convert("RGB")) for f in (src_crop[0], src_crop[-1])]
    crop_mask = lambda i: upscale_mask(pm[i], W, H)[cy0:cy0 + ch, cx0:cx0 + cw]  # noqa: E731
    locked = composite.camera_is_locked(sa, [crop_mask(0), crop_mask(len(pm) - 1)])
    mode = a.mode if a.mode != "auto" else ("plate" if locked else "ltx")

    mattes_hard = None
    # 4. Fill the crop.
    gdir = run / "gen"
    gdir.mkdir(exist_ok=True)
    t = time.monotonic()
    if mode == "plate":
        # Locked-off shot: real pixels first. Wherever the object moves away at some point, the background is
        # taken from the nearest frame that shows it; the clean plate (with matched grain) only fills what is
        # never uncovered. Nothing new is invented and LTX is not needed.
        crops = [np.asarray(Image.open(f).convert("RGB")) for f in src_crop]
        objs = [upscale_mask(pm_obj[min(i, len(pm_obj) - 1)], W, H)[cy0:cy0 + ch, cx0:cx0 + cw] for i in range(len(crops))]
        if a.shadow:
            # Locked-off: the cast shadow is found against the clean background (difference matte), not by SAM.
            shadows = composite.cast_shadows(crops, objs)
            objs = [o | s_ for o, s_ in zip(objs, shadows)]
        mattes_hard = objs
        covered = [composite.grow(m, round((a.grow + 6) * k)) for m in objs]
        filled, never = composite.temporal_fill(crops, covered)
        target = composite.grain_profile(crops[:8])
        plate_arr = np.asarray(plate)
        never_frac = []
        for i in range(len(crops)):
            g = filled[i]
            if never[i].any():
                grained = composite.add_grain(plate_arr, target, seed=i)
                g = np.where(never[i][..., None], grained, g)
            never_frac.append(float(never[i].mean()))
            Image.fromarray(g).save(gdir / f"frame_{i + 1:05d}_.png")
            np.save(gdir / f"never_{i + 1:05d}.npy", np.packbits(never[i]))
        del crops, filled
        timing["never_uncovered_mean"] = round(float(np.mean(never_frac)), 4)
    else:
        lw, lh = media.work_size(cw, ch, LTX_LONG)
        ltx_clip, mask_clip = run / "crop_ltx.mp4", run / "boxmask.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"{crop_vf},scale={lw}:{lh}:flags=lanczos",
                        "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", str(ltx_clip)], check=True)
        bx = np.zeros((lh, lw), np.uint8)
        x0, x1 = int((bx0 - cx0) * lw / cw), int((bx1 - cx0) * lw / cw)
        y0, y1 = int((by0 - cy0) * lh / ch), int((by1 - cy0) * lh / ch)
        bx[max(0, y0 - 16):y1 + 16, max(0, x0 - 16):x1 + 16] = 255
        mdir = run / "boxmask"
        mdir.mkdir(exist_ok=True)
        for i in range(frames):
            Image.fromarray(np.dstack([bx] * 3)).save(mdir / f"m_{i + 1:04d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(mdir / "m_%04d.png"),
                        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(mask_clip)], check=True)
        plate.resize((lw, lh), Image.LANCZOS).save(run / "plate_ltx.png")
        job = run_job(client, graphs.remove(client.assets.from_file(ltx_clip), "", fps, seed=a.seed, dilate=0,
                                            prompt=a.fill_prompt, prefix=f"pickups/{a.name}",
                                            mask_clip=client.assets.from_file(mask_clip),
                                            plate=client.assets.from_file(run / "plate_ltx.png")))
        for o in job.get_outputs("PU_Out"):
            o.to_file(gdir / o.name)
    timing["fill"] = round(time.monotonic() - t, 1)

    # 5. Paste only the matte back onto the full-resolution frames.
    t = time.monotonic()
    gen = sorted(gdir.glob("frame_*.png"))
    cdir, zdir = run / "comp", run / "zoom"
    cdir.mkdir(exist_ok=True)
    zdir.mkdir(exist_ok=True)
    changed, areas = 0, []
    rec_borrowed = rec_generated = 0
    rdir = run / "receipt"
    rdir.mkdir(exist_ok=True)
    for i, sf in enumerate(src):
        full = np.asarray(Image.open(sf).convert("RGB"))
        region = full[cy0:cy0 + ch, cx0:cx0 + cw]
        g = np.asarray(Image.open(gen[min(i, len(gen) - 1)]).convert("RGB").resize((cw, ch), Image.LANCZOS))
        hard = mattes_hard[i] if mattes_hard is not None else crop_mask(min(i, len(pm) - 1))
        matte = composite.soft_matte(hard, round(a.grow * k), a.feather * k)
        g = composite.harmonize(g, region, matte)
        pasted, c = composite.paste(region, g, matte)
        changed += c
        areas.append(float((matte > 0).sum()) / (W * H))
        # Receipt: every changed pixel is either real (borrowed from another frame) or generated.
        nf = gdir / f"never_{i + 1:05d}.npy"
        never_i = np.unpackbits(np.load(nf))[:ch * cw].reshape(ch, cw).astype(bool) if nf.exists() else np.ones((ch, cw), bool)
        touched = matte > 0
        rec_borrowed += int((touched & ~never_i).sum())
        rec_generated += int((touched & never_i).sum())
        rec = np.zeros((ch, cw, 3), np.uint8)
        rec[touched & ~never_i] = (40, 170, 255)
        rec[touched & never_i] = (255, 70, 200)
        Image.fromarray(rec).save(rdir / f"r_{i + 1:04d}.png")
        out = full.copy()
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        Image.fromarray(out).save(cdir / f"c_{i + 1:04d}.png")
        Image.fromarray(np.concatenate([region, pasted], axis=1)).save(zdir / f"z_{i + 1:04d}.png")
    timing["composite"] = round(time.monotonic() - t, 1)
    encode(cdir / "c_%04d.png", fps, run / "review.mp4")
    encode(zdir / "z_%04d.png", fps, run / "zoom_compare.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-i", str(run / "review.mp4"), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2,scale=2400:-2", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(run / "side_by_side.mp4")], check=True)
    n = len(src)
    zw = min(2 * cw, 2400)
    zh = round(ch * zw / (2 * cw))
    sheet = Image.new("RGB", (zw, zh * 3))
    for r, i in enumerate([0, n // 2, n - 1]):
        sheet.paste(Image.open(zdir / f"z_{i + 1:04d}.png").resize((zw, zh), Image.LANCZOS), (0, r * zh))
    sheet.save(run / "sheet.png")
    rep = {"name": a.name, "clip": str(a.clip), "label": a.label, "shadow": a.shadow, "size": [W, H], "fps": fps, "frames": n,
           "frames_found": sum(found), "crop": [cx0, cy0, cw, ch], "plate_size": [qw, qh], "editor": a.editor, "plate": "quality" if a.quality else "fast",
           "matte_area_mean": round(float(np.mean(areas)), 4), "seconds": timing,
           "outside_matte_pixels_changed_pre_encode": changed, "mode": mode, "camera_locked": locked,
           "receipt": {"changed_pixels_real_from_other_frames": rec_borrowed, "changed_pixels_generated": rec_generated,
                       "frame_pixels_untouched_pct": round(100 * (1 - (rec_borrowed + rec_generated) / (W * H * n)), 3)}}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
