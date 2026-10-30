"""Finish: 4K and/or HDR masters for delivery.

4K: SeedVR2 3B restores and upscales the whole clip with ComfyUI's native temporal chunks (sharpest, low flicker).
Fast uses NVIDIA RTX Video Super Resolution when the user has allowed it (proprietary; ~12x faster, no new detail). HDR: LTX-2.5's SDR->HDR IC-LoRA reconstructs the highlights at
1920x1088; the detail-safe merge keeps the (upscaled) source pixels below a luma threshold and takes only the
highlight range from LTX, at the output resolution. Outputs a BT.2020 HLG 10-bit master, ACEScg EXRs and ProRes.

The receipt (receipts.finish_receipt): an upscale is its own share, "upscaled" (your footage resampled: new detail, no new
content), never AI; the only generated pixels are the HDR highlights LTX rebuilt.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import graphs, hdr, media
from pickups.app.comfy import check_canceled, run_local
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.receipts import finish_receipt
from pickups.tasks import common

STAGES = [
    Stage("Preparing the shot", 20),
    Stage("Upscaling to 4K", 450),
    Stage("Rebuilding the highlights", 240),
    Stage("Merging HDR at full resolution", 240),
    Stage("Writing your files", 90),
]
SEEDVR2_3B = "seedvr2_3b_int8_convrot.safetensors"


def stages_for(params: dict, upscale: bool | None = None) -> list[Stage]:
    """Only the stages the chosen finish runs: 4K alone never rebuilds highlights, HDR alone never upscales (nor 4K HDR
    on a source already 3840 wide: `upscale` False once the task knows)."""
    p = params or {}
    k4 = bool(p.get("upscale_4k", True)) if upscale is None else upscale
    hdr_on = bool(p.get("hdr", True))
    return [s for s in STAGES if (k4 or "4K" not in s.label) and (hdr_on or not ("HDR" in s.label or "highlights" in s.label))]


def _download(job, node: str, d: Path) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    for o in job.get_outputs(node):
        check_canceled()  # hundreds of 4K frames: a Cancel stops between them
        o.to_file(d / o.name)
    return sorted(d.iterdir())


def run(ctx) -> dict:
    p = ctx.params
    want_4k, want_hdr, best = bool(p.get("upscale_4k", True)), bool(p.get("hdr", True)), p.get("quality", "best") == "best"
    if not (want_4k or want_hdr):
        raise RuntimeError("Pick 4K, HDR or both.")
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)

    ctx.stage("Preparing the shot")
    work = run_dir / "work.mp4"
    info = common.work_clip(ctx, work, ltx=want_hdr)
    W, H, fps = info["width"], info["height"], info["fps"]
    n = info["frames"]
    scale = 3840 / max(W, H) if want_4k else 1.0
    upscale = want_4k and scale > 1.05
    rtx = upscale and not best and ctx.allow_rtx()
    sdr_dir = run_dir / "sdr"
    if hasattr(ctx, "use_stages"):
        ctx.use_stages(stages_for(p, upscale))  # a source already 4K: no upscale stage to show

    if upscale:
        ctx.stage("Upscaling to 4K")
        if rtx:
            # NVIDIA RTX Video Super Resolution: ~12x faster, very stable, adds no new detail (opt-in: proprietary).
            job = ctx.run(graphs.rtx_upscale(ctx.client.assets.from_file(work), scale, prefix="pickups/finish4k_rtx"))
        else:
            job = ctx.run(graphs.seedvr2_video(ctx.client.assets.from_file(work), scale, model=SEEDVR2_3B, prefix="pickups/finish4k"))
        _download(job, "PU_Out", sdr_dir)
        ctx.conn.free_memory()
    else:
        sdr_dir.mkdir(exist_ok=True)
        run_local(["ffmpeg", "-v", "error", "-y", "-i", str(work), str(sdr_dir / "u_%05d.png")])
    for i, f in enumerate(sorted(sdr_dir.glob("*.png"))):  # one naming scheme for every path
        f.rename(sdr_dir / f"s_{i + 1:05d}.png.tmp")
    for f in sdr_dir.glob("*.tmp"):
        f.rename(f.with_suffix(""))
    sdr_frames = sorted(sdr_dir.glob("s_*.png"))
    OW, OH = Image.open(sdr_frames[0]).size

    downloads: list[dict] = []
    details: dict = {"input": f"{W}×{H}", **common.size_details(OW, OH, len(sdr_frames), fps)}
    hl = 0.0  # % of pixels whose HDR highlights LTX-2.5 rebuilt: the take's only generated share

    if want_hdr:
        ctx.stage("Rebuilding the highlights")
        hw, hh = media.work_size(W, H, 1920)
        hdr_in = run_dir / "hdr_in.mp4"
        media.prepare_work_clip(work, hdr_in, 0, n, (hw, hh))
        job = ctx.run(graphs.sdr_to_hdr(ctx.client.assets.from_file(hdr_in), fps, exr=True, prefix="pickups/finish_hdr"))
        exrs = [f for f in _download(job, "PU_EXR", run_dir / "ltx_exr") if f.suffix == ".exr"]
        ctx.conn.free_memory()

        ctx.stage("Merging HDR at full resolution")
        exr_dir = ctx.vdir / "exr"
        exr_dir.mkdir(exist_ok=True)
        weights = []

        def merged_frames():
            for i, (sf, ef) in enumerate(zip(sdr_frames, exrs)):
                ctx.check()
                sdr = np.asarray(Image.open(sf).convert("RGB"))
                merged, wgt = hdr.merge_frame(sdr, hdr.read_exr(ef))
                weights.append(float((wgt > 0.02).mean()))
                hdr.write_exr(exr_dir / f"frame_{i + 1:05d}.exr", merged)
                ctx.progress((i + 1) / len(exrs))
                yield hdr.hlg_encode(merged)

        rate = media.probe(work).fps
        hdr.write_hlg(merged_frames(), f"{rate.numerator}/{rate.denominator}", ctx.vdir / "hdr_hlg.mp4")
        with zipfile.ZipFile(ctx.vdir / "exr_acescg.zip", "w", zipfile.ZIP_STORED) as z:
            for f in sorted(exr_dir.glob("*.exr")):
                z.write(f, f.name)
        shutil.rmtree(exr_dir, ignore_errors=True)  # the zip is the deliverable
        downloads += [{"label": "HDR master · HLG BT.2020 · H.265 10-bit", "file": "hdr_hlg.mp4", "codec": "hlg", "size": "UHD" if upscale else "HD"},
                      {"label": "EXR sequence · ACEScg (zip)", "file": "exr_acescg.zip", "codec": "exr", "size": "UHD" if upscale else "HD"}]
        hl = round(100 * float(np.mean(weights)), 2) if weights else 0.0
        # receipts.HDR_SHARE reads "on N% of pixels" back from older takes: keep the wording.
        details["hdr"] = f"highlights reconstructed by LTX-2.5 on {hl}% of pixels; everything else is your footage" + \
            (", upscaled" if upscale else "")
    frames_pattern = sdr_dir / "s_%05d.png"  # the SDR master (and viewer) is the (upscaled) source look

    ctx.stage("Writing your files")  # checks Cancel; the encodes below stop on it too (run_local)
    downloads = common.finish_outputs(ctx.vdir, frames_pattern, fps, audio_from=work) + downloads
    make_viewer(work, ctx.vdir / "before.mp4")
    details["models"] = (["NVIDIA RTX Video Super Resolution" if rtx else "SeedVR2 3B"] if upscale else []) + \
        (["LTX-2.5 SDR-to-HDR"] if want_hdr else [])
    receipt = finish_receipt(upscale, hl, details["models"][0] if upscale else None, OW / W if upscale else None)
    # As the form names the choice: 4K HDR, HDR (on a 1080p source, 1080p HDR), 4K.
    size = "4K" if upscale else ("1080p" if want_hdr and max(W, H) <= 1920 else "")
    label = " ".join(x for x in (size, "HDR" if want_hdr else "") if x) or "Finish"
    return {"label": f"Finish {label}", "task": "finish", "receipt": receipt, "downloads": downloads, "details": details}
