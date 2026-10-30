"""Drive test: reference/previz video + character image -> rendered shot (LTX-2.5 + Union Control IC-LoRA).

Usage:
  COMFY_BASE_URL=http://127.0.0.1:8189 uv run python scripts/drive_clip.py --clip DRIVE.mov --ref-image character.png \
     --look "Render image 1 as ... with the character from image 2. Keep the camera framing." \
     --prompt "What the final shot shows..." --name drive_103 [--control canny|video] [--first-frame key.png]
--control depth: Depth Anything 3 on the clip (best for flat-shaded previz). --control video: the clip is already a
control pass (e.g. a Blender depth render). --first-frame skips the Qwen keyframe. --pre cleans the clip first
(e.g. "crop=1920:1034:0:46,delogo=x=20:y=60:w=440:h=140" for a previz HUD).
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

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
    ap.add_argument("--ref-image", type=Path)
    ap.add_argument("--first-frame", type=Path)
    ap.add_argument("--look", help="Qwen instruction that turns frame 1 (+ the character image) into the first frame")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--control", choices=["canny", "depth", "video"], default="canny")
    ap.add_argument("--ic-lora", choices=["union", "layout"], default="union",
                    help="layout: Lightricks' Layout-To-Render IC-LoRA (use --control video with a raw playblast)")
    ap.add_argument("--pre", default="", help="ffmpeg filters applied to the clip first (crop/delogo a HUD)")
    ap.add_argument("--size", help="WxH work size, multiples of 128 (default: from --long-side)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--dev", action="store_true", help="pass 1 on the dev model with real CFG (slower; the negative prompt works)")
    ap.add_argument("--negative", default=None, help="what to avoid (only used with --dev)")
    ap.add_argument("--editor", choices=["qwen2511", "qwen21", "qwen21-turbo"], default="qwen2511", help="keyframe model")
    ap.add_argument("--key-mp", type=float, default=1.0, help="Qwen keyframe size in megapixels (sharper first frame = sharper shot)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    client = Comfy(client_info="pickups")
    timing = {}

    p = media.probe(a.clip)
    frames = media.ltx_frames(min(a.frames, p.frames - a.start))
    if frames > p.frames - a.start:
        frames -= 8
    # Pass 1 runs at half size. The union guide halves that again in latent space (multiples of 128); the layout
    # LoRA's reference is full size (multiples of 64).
    step = 64 if a.ic_lora == "layout" else 128
    if a.size:
        w, h = (int(v) for v in a.size.lower().split("x"))
    else:
        w = max(step, round(a.long_side / step) * step)
        h = max(step, round(w * p.height / p.width / step) * step)
    if w % step or h % step:
        ap.error(f"--size must be multiples of {step}")
    work = run / "work.mp4"
    info = media.prepare_work_clip(a.clip, work, a.start, frames, (w, h), pre=a.pre, cover=True)

    first = a.first_frame
    if first is None:
        if not a.look:
            ap.error("pass --first-frame or --look")
        f0 = run / "frame0.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-frames:v", "1", str(f0)], check=True)
        t = time.monotonic()
        ref = client.assets.from_file(a.ref_image) if a.ref_image else None
        if a.editor == "qwen2511":
            graph = graphs.clean_plate(client.assets.from_file(f0), a.look, seed=a.seed, prefix=f"pickups/{a.name}_key", image2=ref,
                                       size=graphs.qwen_size(w, h, megapixels=a.key_mp))
        else:
            graph = graphs.qwen21_edit(client.assets.from_file(f0), a.look, seed=a.seed, image2=ref, turbo=a.editor == "qwen21-turbo",
                                       resolution=0, prefix=f"pickups/{a.name}_key")
        job = run_job(client, graph)
        first = run / "keyframe.png"
        job.get_outputs("PU_Plate")[0].to_file(first)
        timing["keyframe"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    if a.ic_lora == "layout":
        graph = graphs.layout_render(client.assets.from_file(work), client.assets.from_file(first), a.prompt, info["fps"], w, h,
                                     seed=a.seed, strength=a.strength, prefix=f"pickups/{a.name}",
                                     **({"negative": a.negative} if a.negative else {}))
    else:
        graph = graphs.drive(client.assets.from_file(work), client.assets.from_file(first), a.prompt, info["fps"], w, h,
                             control_mode=a.control, union_strength=a.strength, seed=a.seed, prefix=f"pickups/{a.name}",
                             dev_first_pass=a.dev, **({"negative": a.negative} if a.negative else {}))
    job = run_job(client, graph)
    timing["drive"] = round(time.monotonic() - t, 1)
    out, ctl = run / "out", run / "control"
    out.mkdir(exist_ok=True)
    ctl.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(out / o.name)
    for o in job.get_outputs("PU_ControlPreview"):
        o.to_file(ctl / o.name)
    fps = str(info["fps"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-pattern_type", "glob", "-i", str(out / "frame_*.png"),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(run / "result.mp4")], check=True)
    panels = [work]
    if any(ctl.glob("control_*.png")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-pattern_type", "glob", "-i", str(ctl / "control_*.png"),
                        "-vf", f"scale={w}:{h}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(run / "control.mp4")], check=True)
        panels.append(run / "control.mp4")
    panels.append(run / "result.mp4")
    ins = [x for pth in panels for x in ("-i", str(pth))]
    stack = "".join(f"[{i}:v]scale={w}:{h},setsar=1[p{i}];" for i in range(len(panels)))
    stack += "".join(f"[p{i}]" for i in range(len(panels))) + f"hstack=inputs={len(panels)},scale=2400:-2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", stack, "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", str(run / "side_by_side.mp4")], check=True)
    gen = sorted(out.glob("frame_*.png"))
    n = len(gen)
    srcd = run / "src"
    srcd.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), str(srcd / "s_%04d.png")], check=True)
    src = sorted(srcd.glob("s_*.png"))
    sheet = Image.new("RGB", (w * 2, h * 3))
    for r, i in enumerate([0, n // 2, n - 1]):
        sheet.paste(Image.open(src[i]).convert("RGB"), (0, r * h))
        sheet.paste(Image.open(gen[i]).convert("RGB"), (w, r * h))
    sheet.save(run / "sheet.png")
    rep = {"name": a.name, "control": a.control, "size": [w, h], "frames": frames, "fps": info["fps"], "frames_out": n, "seconds": timing}
    (run / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
