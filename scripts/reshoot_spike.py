"""Reshoot feasibility spike: move the camera on a locked-off shot.

1. Depth Anything 3 (mono) gives per-frame depth on the GPU.
2. Each frame is unprojected with a pinhole camera and re-rendered from a moving virtual camera (push in, pull out,
   orbit, crane), z-buffered point splatting at 2x density + push-pull hole filling = a rough "layout" pass.
3. LTX-2.5 Layout-To-Render renders that layout with the original first frame as the look still.

Usage: COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/reshoot_spike.py --clip SRC --move orbit_left --amount 1.0 \
         --prompt "..." --name rs_pier_orbit [--frames 73] [--fov 50]
Writes ~/.pickups/runs/<name>/: layout.mp4, result.mp4, side_by_side.mp4 (original | layout | result), report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_sdk import Comfy, StatusChange

from pickups import graphs, hdr, media

RUNS = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "runs"
LW, LH = 960, 544     # layout render size (Layout-To-Render's stage 1)
W, H = 1920, 1088


def run_job(client, graph):
    job = client.submit(client.workflows.from_json(graph))
    for ev in job.events():
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    return job.result()


def depth_graph(clip, resolution: int = 518) -> dict:
    r = graphs._ref
    return {
        "PU_Clip": {"class_type": "LoadVideo", "inputs": {"file": clip}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": r("PU_Clip")}},
        "da3": {"class_type": "LoadDA3Model", "inputs": {"model_name": graphs.DA3_MODEL, "weight_dtype": "default"}},
        "geo": {"class_type": "DA3Inference", "inputs": {"da3_model": r("da3"), "image": r("parts", 0), "resolution": resolution,
                                                        "resize_method": "upper_bound_resize", "mode": "mono"}},
        "depth": {"class_type": "DA3Render", "inputs": {"da3_geometry": r("geo"), "output": "depth", "output.normalization": "raw",
                                                       "output.apply_sky_clip": True}},
        "PU_Depth": {"class_type": "SaveImageAdvanced", "inputs": {"images": r("depth"), "filename_prefix": "pickups/reshoot_depth/d",
                                                                    "format": "exr", "format.bit_depth": "32-bit float",
                                                                    "format.input_color_space": "linear"}},
    }


def camera(move: str, s: float, amount: float, pivot: float) -> tuple[np.ndarray, np.ndarray]:
    """World->camera rotation R and camera centre C at progress s (0..1). The original camera is R=I, C=0."""
    e = s * s * (3 - 2 * s)  # ease in-out
    R, C = np.eye(3), np.zeros(3)
    if move in ("push_in", "pull_out"):
        C[2] = (1 if move == "push_in" else -1) * e * amount * pivot * 0.10
    elif move in ("orbit_left", "orbit_right"):
        a = math.radians((8 if move == "orbit_left" else -8) * e * amount)
        # orbit around a point `pivot` ahead of the camera: camera moves on a circle and keeps looking at it
        C = np.array([-math.sin(a) * pivot, 0.0, pivot - math.cos(a) * pivot])
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]).T
    elif move == "crane_up":
        C[1] = -e * amount * pivot * 0.07  # image y points down
        t = math.radians(6 * e * amount)
        R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]).T
    return R, C


def push_pull_fill(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill holes from the surrounding pixels (pyramid push-pull)."""
    if valid.all():
        return img
    levels = []
    cur, m = img.astype(np.float32) * valid[..., None], valid.astype(np.float32)
    while min(cur.shape[:2]) > 4:
        levels.append((cur, m))
        h, w = cur.shape[:2]
        h2, w2 = h // 2 * 2, w // 2 * 2
        c = cur[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).sum(axis=(1, 3))
        mm = m[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum(axis=(1, 3))
        cur = np.where(mm[..., None] > 0, c / np.maximum(mm[..., None], 1e-6), 0)
        m = (mm > 0).astype(np.float32)
    for c_hi, m_hi in reversed(levels):
        up = np.asarray(Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8)).resize((c_hi.shape[1], c_hi.shape[0]), Image.BILINEAR), np.float32)
        cur = np.where(m_hi[..., None] > 0, c_hi, up)
    return np.clip(cur, 0, 255).astype(np.uint8)


def render(rgb: np.ndarray, depth: np.ndarray, R: np.ndarray, C: np.ndarray, f: float) -> np.ndarray:
    """Forward-splat a coloured depth map into a moved pinhole camera (2x density, z-buffer, hole fill)."""
    h, w = depth.shape
    s = 3
    rgb2 = np.asarray(Image.fromarray(rgb).resize((w * s, h * s), Image.BILINEAR))
    d2 = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((w * s, h * s), Image.BILINEAR))
    vs, us = np.mgrid[0:h * s, 0:w * s].astype(np.float32)
    us, vs = (us + 0.5) / s - 0.5, (vs + 0.5) / s - 0.5
    cx, cy = w / 2, h / 2
    X = np.stack([(us - cx) * d2 / f, (vs - cy) * d2 / f, d2], -1).reshape(-1, 3)
    Xc = (X - C) @ R.T
    z = Xc[:, 2]
    ok = z > 1e-3
    u = np.round(Xc[ok, 0] / z[ok] * f + cx).astype(np.int64)
    v = np.round(Xc[ok, 1] / z[ok] * f + cy).astype(np.int64)
    col = rgb2.reshape(-1, 3)[ok]
    zz = z[ok]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, col, zz = u[inside], v[inside], col[inside], zz[inside]
    order = np.argsort(-zz)  # far first, near overwrites
    out = np.zeros((h, w, 3), np.uint8)
    valid = np.zeros((h, w), bool)
    out[v[order], u[order]] = col[order]
    valid[v[order], u[order]] = True
    return push_pull_fill(out, valid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--move", default="orbit_left", choices=["push_in", "pull_out", "orbit_left", "orbit_right", "crane_up"])
    ap.add_argument("--amount", type=float, default=1.0)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=int, default=73)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--fov", type=float, default=50.0, help="assumed horizontal field of view")
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    t0 = time.monotonic()
    p = media.probe(a.clip)
    n = media.ltx_frames(min(a.frames, p.frames - a.start))
    small = run / "small.mp4"
    media.prepare_work_clip(a.clip, small, a.start, n, (LW, LH), cover=True)
    full = run / "full.mp4"
    media.prepare_work_clip(a.clip, full, a.start, n, (W, H), cover=True)
    fps = float(p.fps)

    job = run_job(client, depth_graph(client.assets.from_file(small)))
    ddir = run / "depth"
    ddir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Depth"):
        o.to_file(ddir / o.name)
    depths = [hdr.read_exr(f)[..., 0] for f in sorted(ddir.glob("*.exr"))]
    t_depth = time.monotonic() - t0
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(small), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, LH, LW, 3)
    # DA3 mono depth is relative; put the scene's median at 5 units. If values look like disparity (near = large), invert.
    d0 = np.asarray(Image.fromarray(depths[0]).resize((LW, LH), Image.BILINEAR))
    rep_depth = {"min": float(d0.min()), "median": float(np.median(d0)), "max": float(d0.max())}
    f = (LW / 2) / math.tan(math.radians(a.fov) / 2)
    ldir = run / "layout"
    ldir.mkdir(exist_ok=True)
    # The camera pivots on the subject: the median depth of the central region (where the subject usually is),
    # scaled to 5 units. Single-image depth is compressed, so moves stay modest.
    box = d0[int(LH * 0.35):int(LH * 0.9), int(LW * 0.3):int(LW * 0.7)]
    subject = float(np.median(box[box < np.percentile(d0, 90)])) if box.size else float(np.median(d0))
    rep_depth["subject"] = subject
    for i in range(len(frames)):
        d = np.asarray(Image.fromarray(depths[min(i, len(depths) - 1)]).resize((LW, LH), Image.BILINEAR)).astype(np.float32)
        d = 5.0 * d / max(subject, 1e-6)
        d = np.asarray(Image.fromarray(d).filter(__import__("PIL.ImageFilter", fromlist=["MedianFilter"]).MedianFilter(5)))  # calm edges
        d = np.clip(d, 0.3, 200.0)
        R, C = camera(a.move, i / max(len(frames) - 1, 1), a.amount, 5.0)
        Image.fromarray(render(frames[i], d, R, C, f)).save(ldir / f"l_{i + 1:04d}.png")
    layout = run / "layout.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(ldir / "l_%04d.png"), "-vf", f"scale={W}:{H}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(layout)], check=True)
    look = run / "look.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-frames:v", "1", str(look)], check=True)
    t1 = time.monotonic()
    job = run_job(client, graphs.layout_render(client.assets.from_file(layout), client.assets.from_file(look), a.prompt, fps, W, H,
                                               prefix=f"pickups/{a.name}"))
    out = run / "out"
    out.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(out / o.name)
    t_render = time.monotonic() - t1
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(out / "*.png"),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(run / "result.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-i", str(layout), "-i", str(run / "result.mp4"), "-filter_complex",
                    "[0:v][1:v][2:v]hstack=inputs=3,scale=2880:-2", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(run / "side_by_side.mp4")], check=True)
    rep = {"name": a.name, "move": a.move, "amount": a.amount, "frames": int(len(frames)), "depth_raw": rep_depth,
           "seconds": {"depth": round(t_depth, 1), "render": round(t_render, 1), "total": round(time.monotonic() - t0, 1)}}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
