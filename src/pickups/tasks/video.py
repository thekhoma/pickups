"""New video: text to video, or image to video from a picture, on LTX-2.5. The video becomes a new shot in the project,
so every edit (replace a person, change the weather, finish to 4K HDR) works on it next.

Params (pickups.video.request checks and fills them): mode "text" | "image", prompt, aspect, resolution "720p" | "1080p",
duration 1-4 s (25, 49, 73 or 97 frames at 24 fps), seed, and for image to video `image`: {kind: image | cast | upload, id}.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pickups import graphs, video
from pickups.app.jobs import Stage

STAGES = [Stage("Loading LTX-2.5", 20), Stage("Making your video", 50)]
OUT = "PU_Save"


def run(ctx) -> dict:
    p = video.request(ctx.params)
    picture = None
    if p["mode"] == "image":
        try:
            picture = ctx.store.ref_path(ctx.project_id, p["image"])
        except FileNotFoundError as e:
            raise ValueError("That picture is gone. Choose another one.") from e

    ctx.stage("Loading LTX-2.5")
    try:
        installed = ctx.conn.model_files()
    except Exception:  # noqa: BLE001 - an unknown list never blocks the job; ComfyUI says what's missing
        installed = set()
    if installed and (gone := [f for f in MODEL_FILES if f not in installed and not any(i.endswith(f) for i in installed)]):
        raise RuntimeError(f"LTX-2.5 needs files ComfyUI doesn't have yet: {', '.join(gone)}. See Setup.")

    ctx.stage("Making your video")
    w, h, n = p["width"], p["height"], p["frames"]
    if picture is not None:
        graph = graphs.ltx_image_to_video(p["prompt"], ctx.client.assets.from_file(picture), w, h, n, seed=p["seed"], fps=p["fps"])
    else:
        graph = graphs.ltx_text_to_video(p["prompt"], w, h, n, seed=p["seed"], fps=p["fps"])
    job = ctx.run(graph)
    outs = job.get_outputs(OUT)
    if not outs:
        raise RuntimeError("ComfyUI finished but sent no video back.")
    pick = next((o for o in outs if str(getattr(o, "name", "")).lower().endswith(".mp4")), outs[0])
    dst = Path(ctx.vdir) / "video.mp4"
    pick.to_file(dst)
    ctx.progress(1.0)
    made = {"task": "video", "mode": p["mode"], "model": video.MODEL, "prompt": p["prompt"], "aspect": p["aspect"],
            "resolution": p["resolution"], "width": w, "height": h, "frames": n, "fps": p["fps"], "seed": p["seed"],
            "seed_locked": bool(ctx.params.get("seed_locked")),
            **({"image": p["image"]} if p["image"] else {})}
    if picture is not None:  # the start picture stays with the shot (a deleted asset never breaks it)
        keep = Path(ctx.vdir) / f"start{Path(picture).suffix.lower() or '.png'}"
        shutil.copyfile(picture, keep)
        made["start_file"] = keep.name
    return {"shot": {"path": str(dst), "name": video.shot_name(p["prompt"]), "made": made,
                     "keep": [made["start_file"]] if picture is not None else []}}


# What the graphs load (manifest.TASK_MODELS["video"] lists the same files for Setup).
MODEL_FILES = [graphs.UNET, graphs.TEXT_ENCODER, graphs.VIDEO_VAE, graphs.LATENT_UPSCALER]
