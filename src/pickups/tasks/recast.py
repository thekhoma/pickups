"""Recast: same performance, new person. Only the person changes.

Two engines (params.engine). "ltx", the default, is tasks/recast_ltx.py: the new character is painted into one frame and
LTX-2.5 renders the shot around them (the whole costume, lit by the shot). "scail" is this module's SCAIL-2 path, which
follows the exact performance:

The user clicks the person; SAM 3.1 follows them through the shot. The new character is cut out of its picture on its
own (reference.py: one figure, its own mask, even from a contact sheet). SCAIL-2 (Wan 2.1, replacement mode) renders
them on a crop around the person, at a good working size, re-rendering only that person when someone else shares the
crop. The union of the old person's matte and the new character's (whatever SAM calls it, plus what SCAIL-2 changed
next to the person) is pasted back onto the full-resolution frames; everything else stays the original footage.
"""

from __future__ import annotations

import subprocess

import numpy as np
from comfy_sdk import InvalidWorkflow
from PIL import Image

from pickups import composite, graphs, images, media, mentions, reference
from pickups.app.comfy import Canceled
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.tasks import common

# FLUX 3 klein reference still for "describe who's in" (BFL: klein expects a full caption; this reads like the output
# of its prompt upsampler). Whole body, facing camera, arms slightly away so SAM can separate them, plain backdrop.
STILL = ("Studio photograph, full-length shot of {who}, standing centered and facing the camera in a relaxed neutral "
         "stance, arms slightly away from the body, feet shoulder-width apart, the whole figure from the top of the head "
         "to the shoes inside the frame, hands empty. Soft, even light from the front, a gentle shadow under the chin. "
         "Plain light-grey seamless backdrop and floor, a faint contact shadow under the feet.")

STAGES = [
    Stage("Preparing the shot", 20),
    Stage("Finding the person", 25),
    Stage("Reading the character sheet", 40),
    Stage("Performing the scene", 180),
    Stage("Blending at full resolution", 120),
    Stage("Writing your files", 30),
]
SCAIL_LONG = 832
# Close-ups (crop long side > 3 x SCAIL_LONG, e.g. a 4K medium close-up) would be upscaled ~4.6x on paste and read soft.
# 1024 long side still fits a 24 GB card at 97 frames (int8 SCAIL-2 + the distill LoRA).
SCAIL_LONG_CLOSE = 1024
AI_TINT = (255, 79, 216)
DRIVE_LABEL = "human"  # graphs.scail2's own SAM label for the people on the drive
# The new character on SCAIL-2's render, whatever it is: a robot or an inflatable T-rex may not answer to "person".
NEW_LABELS = ("person", "character", "costume", "robot", "figure")
ENGINES = ("ltx", "scail")
DEFAULT_ENGINE = "ltx"


def engine(params: dict) -> str:
    """Which renderer a Recast job uses: "ltx" (the default) or "scail"."""
    e = str(params.get("engine") or DEFAULT_ENGINE).strip().lower()
    if e not in ENGINES:
        raise ValueError(f"Unknown Recast engine '{e}'. Use 'ltx' or 'scail'.")
    return e


def stages_for(params: dict) -> list[Stage]:
    """The job's stages (and so its countdown), by engine. Never raises: an unknown engine fails when the job runs."""
    if str(params.get("engine") or DEFAULT_ENGINE).strip().lower() == "scail":
        return STAGES
    from pickups.tasks import recast_ltx
    return recast_ltx.stages_for(params)


def run(ctx) -> dict:
    if engine(ctx.params) == "ltx":
        from pickups.tasks import recast_ltx
        return recast_ltx.run(ctx)
    from pickups.tasks import recast_many
    swaps = recast_many.parse_swaps(ctx.params)
    if len(swaps) > 1:
        raise RuntimeError("Swapping several people at once needs the default engine. Pick it under Engine.")
    if swaps:
        ctx.params = recast_many.single_params(ctx.params, swaps[0])
    return run_scail(ctx)


def who_is_in(ctx, p: dict):
    """The new character's picture, as the user gave it: (picture, params, made a still, cast entry).

    In order: a picked reference (who_ref), a cast member (cast_id, or the first person @-mentioned in `describe`, whose
    words then name them), an uploaded photo or sheet, the still an earlier take made ("Try again with changes"), or a
    FLUX 3 klein still made from the description (kept with the take as reference.png)."""
    cast, cast_file = ctx.store.shot_cast(ctx.shot_id)
    who = next((c for c in cast if c["id"] == p.get("cast_id")), None) or next(
        (c for c in mentions.mentioned(p.get("describe", ""), cast) if c.get("kind", "person") == "person"), None)
    made_still = False  # a FLUX 3 klein reference still, from a description
    if p.get("who_ref"):
        ref = ctx.store.ref_path(ctx.project_id, p["who_ref"])
    elif who:
        ref = cast_file(who)
        p = dict(p, describe=mentions.for_text(p.get("describe") or f"@{who['name']}", cast))
    elif p.get("ref_upload_id"):
        ref = ctx.store.upload_path(p["ref_upload_id"])
    elif p.get("reference_take") and (kept := ctx.store.take_reference(ctx.shot_id, p["reference_take"])):
        # "Try again with changes" keeping the person: that take's FLUX 3 klein still (kept with this take too).
        ref = ctx.vdir / "reference.png"
        ref.write_bytes(kept.read_bytes())
        made_still = True
    elif p.get("describe") or p.get("reference_prompt"):
        still = graphs.still(p.get("reference_prompt") or STILL.format(who=p["describe"].strip().rstrip(".")),
                             832, 1216, seed=int(p.get("seed", 42)), prefix="pickups/cast")
        job = ctx.run(images.localize_for(ctx, still))
        ref = ctx.vdir / "reference.png"  # kept with the take: "Try again with changes" shows and reuses the person
        job.get_outputs("PU_Out")[0].to_file(ref)
        made_still = True
    else:
        raise RuntimeError("Add a photo (or a description) of who's in.")
    return ref, p, made_still, who


def run_scail(ctx) -> dict:
    p = ctx.params
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)

    ctx.stage("Preparing the shot")
    src = media.probe(ctx.input_path)
    frames = 4 * ((min(common.MAX_FRAMES, src.frames) - 1) // 4) + 1  # Wan wants 4k+1
    if int(p.get("frame", 0)) >= frames:  # the people finder looks as far as the LTX engine does
        raise RuntimeError(f"SCAIL-2 works on the first {frames / max(float(src.fps), 1.0):.0f} s of a shot, and the person you picked "
                           "comes in later. Use the default engine (LTX-2.5), which finds them wherever they are.")
    work = run_dir / "work.mp4"
    info = common.work_clip(ctx, work, frames=frames)
    W, H, fps = info["width"], info["height"], info["fps"]
    k = max(W, H) / 1920
    proxy = run_dir / "proxy.mp4"
    common.proxy_clip(ctx, work, proxy)

    ctx.stage("Finding the person")
    pm = common.tracked_masks(ctx, proxy, int(p.get("frame", 0)), p.get("points", []), p.get("text", ""), src.width, run_dir)
    common.require_size(pm, src.height)  # too small for a good result: say so now, in the shot's own pixels
    ph, pw = pm[0].shape
    ys, xs = np.nonzero(np.any(np.stack(pm), axis=0))
    sx, sy = W / pw, H / ph
    bx0, bx1, by0, by1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    bw, bh = bx1 - bx0, by1 - by0
    cx0, cx1 = int(max(0, bx0 - bw * 0.35)), int(min(W, bx1 + bw * 0.35))
    cy0, cy1 = int(max(0, by0 - bh * 0.35)), int(min(H, by1 + bh * 0.35))
    cw, ch = (cx1 - cx0) // 2 * 2, (cy1 - cy0) // 2 * 2

    ctx.stage("Reading the character sheet")
    ref, p, made_still, _ = who_is_in(ctx, p)

    ctx.stage("Performing the scene")
    scail_long, sw, sh = performance_size(cw, ch)
    crop = run_dir / "crop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"crop={cw}:{ch}:{cx0}:{cy0},scale={sw}:{sh}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", str(crop)], check=True)
    pbox = (cx0 / sx, cy0 / sy, (cx0 + cw) / sx, (cy0 + ch) / sy)  # the crop, in proxy pixels
    old_drive = [mask_in(m, pbox, (sw, sh)) for m in pm]
    # One figure, alone and big, with its own mask: SCAIL-2 copies it, CLIP-vision reads it.
    cast_ref = _reference(ctx, ref, sw / sh, run_dir)
    ref_png, ref_mask = run_dir / "reference_in.png", run_dir / "reference_mask.png"
    cast_ref.image.save(ref_png)
    Image.fromarray(cast_ref.mask.astype(np.uint8) * 255).convert("RGB").save(ref_mask)
    indices, people = _only_them(ctx, crop, old_drive[0], run_dir)
    # SCAIL-2 wants a description of the finished video (its authors' prompt enhancer), not an instruction.
    who_text = (p.get("describe") or "").strip().rstrip(".")
    prompt = (p.get("prompt") or " ".join(x for x in (
        f"{who_text[0].upper() + who_text[1:]}, moves through the scene with the original performer's exact pose, timing, "
        "movement and position in the frame." if who_text else "",
        (p.get("scene") or "").strip(),
        "The same place, light, camera angle and shot scale as the original shot; a normal-sized head in proportion to the body.") if x))

    def perform(indices: str, mask):
        return ctx.run(graphs.scail2(ctx.client.assets.from_file(crop), ctx.client.assets.from_file(ref_png), prompt, replacement=True,
                                     object_indices=indices, reference_mask=mask, seed=int(p.get("seed", 42)), prefix="pickups/recast"))
    ref_masked = True
    try:
        job = perform(indices, ctx.client.assets.from_file(ref_mask))
    except InvalidWorkflow:
        # A ComfyUI whose SCAIL2ColoredMask won't take a plain MASK as the reference (0.27's does): SAM's "human" track of
        # the cut-out instead, for everyone in the crop (object_indices filters that track too, and could empty it).
        indices, ref_masked = "", False
        job = perform("", None)
    gdir = run_dir / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)
    gen = sorted(gdir.glob("frame_*.png"))
    gen_clip = run_dir / "gen_crop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(gdir / "frame_*.png"),
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", str(gen_clip)], check=True)
    sel = ctx.run(graphs.select_any(ctx.client.assets.from_file(gen_clip), NEW_LABELS, prefix="pickups/recast_new"))
    found = _found(sel, len(NEW_LABELS), run_dir / "sam_new", (sw, sh))
    # Someone else in the crop: the new character as one instance, so a partner is never taken for them.
    tracked = _follow(ctx, gen_clip, old_drive[0], gen[0], run_dir, (sw, sh)) if people > 1 and gen else None
    blank = np.zeros((sh, sw), bool)

    def matte_for(i: int, orig: np.ndarray):
        if i >= len(gen):
            return blank, blank, False
        g = np.asarray(Image.open(gen[i]).convert("RGB").resize((sw, sh), Image.BILINEAR))
        return new_matte(found[i] if i < len(found) else blank, old_drive[min(i, len(old_drive) - 1)], g, orig,
                         tracked[i] if tracked and i < len(tracked) else None)
    mattes = list(common.parallel_frames(common.read_frames(crop), matte_for))

    ctx.stage("Blending at full resolution")
    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, fps, kind="h264", scale_long=1920)
    qs = max(1, round(max(cw, ch) / 640))  # QA works on every qs-th pixel: cheap, and the same at any frame size

    def blend(i: int, full: np.ndarray):
        out = full.copy()
        if i >= len(gen):
            return out, out, 0, 0, None
        old = mask_in(pm[min(i, len(pm) - 1)], pbox, (cw, ch))
        new_d, subject_d, hit = mattes[i] if i < len(mattes) else (blank, blank, False)
        new, subject = upsize(new_d, (cw, ch)), upsize(subject_d, (cw, ch))
        # What is pasted is the old person plus the new character (a bulkier costume is never clipped to the old
        # outline); that matte is also what the receipt counts as AI pixels.
        matte = composite.soft_matte(old | new, round(10 * k), 6.0 * k)
        g = np.asarray(Image.open(gen[i]).convert("RGB").resize((cw, ch), Image.LANCZOS))
        if subject.any():
            g = composite.dehalo(g, subject, max(6, round(12 * k)), inner=max(2, round(5 * k)))  # SCAIL-2's glow (~10-25 px at 4K)
        region = full[cy0:cy0 + ch, cx0:cx0 + cw]
        g = composite.harmonize(g, region, matte)
        pasted, c = composite.paste(region, g, matte)
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        touched = matte > 0
        tinted = out.copy()
        reg = pasted.astype(np.float32)
        reg[touched] = reg[touched] * 0.45 + np.array(AI_TINT, np.float32) * 0.55
        tinted[cy0:cy0 + ch, cx0:cx0 + cw] = reg.astype(np.uint8)
        q = {"out": pasted[::qs, ::qs], "src": region[::qs, ::qs], "new": new[::qs, ::qs], "old": old[::qs, ::qs], "hit": hit,
             "coverage": coverage(new_d, old_drive[min(i, len(old_drive) - 1)]),
             "halo": halo_lift(pasted[::qs, ::qs], subject[::qs, ::qs], k / qs) if hit else None}
        return out, tinted, c, int(touched.sum()), q

    changed = touched_px = 0
    n = 0
    qa = QA()
    for i, (out, tinted, c, t, q) in enumerate(common.parallel_frames(common.read_frames(work), blend)):
        ctx.check()
        n += 1
        changed += c
        touched_px += t
        qa.add(q)
        out_w.write(out)
        rec_w.write(tinted)
        ctx.progress((i + 1) / max(len(gen), 1))
    out_w.close()
    rec_w.close()

    ctx.stage("Writing your files")
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=work)
    make_viewer(work, ctx.vdir / "before.mp4")
    total = W * H * n
    receipt = {"original": round(100 * (1 - touched_px / total), 2), "borrowed": 0.0, "ai": round(100 * touched_px / total, 2)}
    return {"label": "Replace a person", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, n, fps), "engine": "scail", "performance_size": f"{sw}×{sh}", "reference": cast_ref.path,
                        "pixels_changed_outside_selection": changed,
                        "models": ["SAM 3.1", "SCAIL-2 (Wan 2.1)"] + (["FLUX 3 klein"] if made_still else []),
                        "_qa": {**qa.summary(), "scail_long": scail_long, "reference": cast_ref.path, "reference_mask": ref_masked,
                                "object_indices": indices, "people_on_first_frame": people, "new_track": tracked is not None}}}


def performance_size(cw: int, ch: int) -> tuple[int, int, int]:
    """SCAIL-2's working size for a crop: (long side used, width, height), multiples of 32."""
    long = SCAIL_LONG_CLOSE if max(cw, ch) > 3 * SCAIL_LONG else SCAIL_LONG
    s = long / max(cw, ch)
    return long, max(32, round(cw * s / 32) * 32), max(32, round(ch * s / 32) * 32)


# --- masks -------------------------------------------------------------------------------------------------------


def mask_in(m: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """The `box` (x0, y0, x1, y1, in m's pixels) of a mask, resampled to `size` (w, h)."""
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.BILINEAR, box=box)) > 127


def upsize(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (m.shape[1], m.shape[0]) == size:
        return m
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127


def drive_rank(people: list[np.ndarray], old: np.ndarray, max_objects: int = 4) -> int | None:
    """Which of SCAIL-2's drive objects is the chosen person: their left-to-right rank among the people SAM 3.1 finds on
    the drive's first frame (SCAIL2ColoredMask orders by first appearance, then centroid x).

    None (SCAIL re-renders everyone, as before) when there is only one person, or when it isn't clear: the person isn't
    on the first frame, the selection covers two people, or there are more people than SCAIL tracks."""
    people = [m for m in people if m.any()]
    if not 2 <= len(people) <= max_objects or not old.any():
        return None
    iou = [float((m & old).sum()) / float((m | old).sum()) for m in people]
    best = int(np.argmax(iou))
    if iou[best] < 0.3 or sum((m & old).sum() > 0.5 * m.sum() for m in people) > 1:
        return None
    cx = [float(np.nonzero(m)[1].mean()) for m in people]
    return sorted(range(len(people)), key=lambda j: cx[j]).index(best)


def new_matte(found: np.ndarray, old: np.ndarray, gen: np.ndarray, orig: np.ndarray,
              track: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """The new character on one generated drive frame: (new matte, dehalo subject, SAM found them).

    `found` is SAM 3.1's union over NEW_LABELS on the render. `track`, when someone else shares the crop, is the new
    character alone (followed from where the old person stood): then only changes next to it count, so a partner
    SCAIL-2 redrew a little off (and the spot they used to fill) is never theirs. Without a track only what touches the
    original person counts. Past the old outline only pixels SCAIL-2 actually changed count, so a partner it left alone
    is not pasted.

    Changed pixels joined to the person are added even when SAM has no word for them, so a bulkier costume is never
    clipped to the old silhouette. Whitish, lighter pixels in a thin shell hugging the outline are SCAIL's glow
    (dehalo's job); a light part reaching beyond that shell is costume: a white spacesuit, chrome or steel armour wider
    than the performer. When SAM finds nothing, dehalo still works around the old outline plus any bulk."""
    blank = np.zeros_like(old)
    if not old.any():
        return blank, blank, False
    ys = np.nonzero(old)[0]
    tall = int(ys.max() - ys.min() + 1)
    near_px = max(2, round(0.02 * tall))
    near = composite.grow(old, near_px)
    reach = composite.grow(old, max(8, round(0.25 * tall)))
    g, o = gen.astype(np.float32), orig.astype(np.float32)
    d = composite.smooth(np.sqrt(((g - o) ** 2).sum(-1)), 1.0)
    far = ~reach & ~found
    base = float(np.percentile(d[far], 90)) if far.sum() > 200 else 0.0  # SCAIL's own drift on the background
    changed = d > max(28.0, 1.5 * base + 8.0)
    changed &= composite.grow(composite.shrink(changed, 1), 1)  # no specks
    lum = np.array([0.299, 0.587, 0.114], np.float32)
    glow = ((g @ lum) > (o @ lum)) & ((g.max(-1) - g.min(-1)) / np.maximum(g.max(-1), 1.0) < 0.28)
    cand = changed & reach & ~old
    if track is not None and track.any():
        around = composite.grow(track, near_px)
        hit = (track | (found & around)) & (near | changed) & reach
        cand &= around  # whatever else changed (a partner redrawn off, where they used to be) is someone else
    else:
        hit = reference.touching(found, old) & (near | changed) & reach  # never a whole re-rendered partner
    attach = composite.grow(old, 2) | hit
    body = old | hit | reference.touching(cand & ~glow, attach)
    # SCAIL's glow is a few drive pixels wide at any size: light pieces that stay inside this shell are glow.
    r = max(8, round(0.03 * tall))
    shell = composite.grow(body, r)
    light = reference.touching(cand & glow & ~body, composite.grow(body, 2))
    lab, n = reference.label(light)
    if n:
        keep = np.bincount(lab[light & ~shell], minlength=n + 1) >= r * r
        keep[0] = False
        light = keep[lab]
    bulk = reference.touching((cand & ~glow) | light, attach)  # a dark glove at the end of a white sleeve too
    new = hit | bulk
    return new, (new if hit.any() else old | bulk), bool(hit.any())


# --- QA (stored in details["_qa"] for the stress harness; never shown) ---------------------------------------------


def coverage(new: np.ndarray, old: np.ndarray) -> float | None:
    """New-matte area over old-matte area (a bulkier costume > 1, a missed character 0)."""
    return float(new.sum()) / float(old.sum()) if old.any() else None


def halo_lift(img: np.ndarray, mask: np.ndarray, k: float = 1.0) -> float | None:
    """Mean luminance lift (0-255) of the ring 6-12 px outside `mask` over the ring 12-24 px out (px scaled by k):
    about 0 when the edge is clean, clearly positive when a light glow surrounds the character."""
    if not mask.any():
        return None
    g6, g12, g24 = (composite.grow(mask, max(1, round(px * k))) for px in (6, 12, 24))
    r1, r2 = g12 & ~g6, g24 & ~g12
    if r1.sum() < 8 or r2.sum() < 8:
        return None
    y = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(y[r1].mean() - y[r2].mean())


def frame_change(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    """How much `b` changed from `a` inside `mask`: the mean absolute difference over b's own contrast there (its
    spread of levels). A white spacesuit swapped in for a dark sweater changes more in levels at the same steadiness;
    measured against its own contrast, steady is steady."""
    if not mask.any():
        return None
    inside = b.astype(np.float32)[mask]
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32))[mask].mean()) / max(float(inside.std()), 4.0)


class QA:
    """Per-frame QA from the blend loop, summarised as medians. Flicker compares how much the new character changes
    frame to frame with how much the original person did (about 1 = as steady as the real footage)."""

    def __init__(self):
        self.cov: list[float] = []
        self.halo: list[float] = []
        self.d_new: list[float] = []
        self.d_old: list[float] = []
        self.empty = 0
        self.frames = 0
        self.prev: dict | None = None

    def add(self, q: dict | None) -> None:
        if q is None:
            return
        self.frames += 1
        if q["old"].any() and not q["hit"]:
            self.empty += 1
        for name, v in (("cov", q["coverage"]), ("halo", q["halo"])):
            if v is not None:
                getattr(self, name).append(v)
        if self.prev is not None:
            dn, do = frame_change(self.prev["out"], q["out"], q["new"]), frame_change(self.prev["src"], q["src"], q["old"])
            if dn is not None and do is not None:
                self.d_new.append(dn)
                self.d_old.append(do)
        self.prev = q

    def summary(self) -> dict:
        def med(xs: list[float]) -> float | None:
            return round(float(np.median(xs)), 3) if xs else None
        flicker = None
        if self.d_new:
            flicker = round(float(np.median(self.d_new)) / max(float(np.median(self.d_old)), 0.02), 3)
        return {"coverage": med(self.cov), "halo": med(self.halo), "flicker": flicker, "empty_new_frames": self.empty,
                "frames": self.frames}


# --- steps that ask ComfyUI ----------------------------------------------------------------------------------------


def _reference(ctx, ref, aspect: float, run_dir):
    """The new character alone, for SCAIL-2: cut out against a plain backdrop (a still, a sheet's front view, a
    transparent PNG), or picked out of a busy photo by SAM 3.1."""
    try:
        prepared = reference.prepare(ref, aspect)
    except (OSError, ValueError) as e:  # not a picture we can read
        raise RuntimeError("That picture of who's in can't be opened. Try a PNG or JPEG.") from e
    if prepared is not None:
        return prepared
    im = Image.fromarray(reference.load(ref)[0])
    im.thumbnail((2048, 2048))
    png = run_dir / "reference_photo.png"
    im.save(png)
    for label in NEW_LABELS:
        try:
            people = common.select_instances(ctx, png, label, 1.0, run_dir / f"ref_{label}")
        except Exception:  # noqa: BLE001 - try the next word
            continue
        if people and (prepared := reference.prepare(ref, aspect, mask=people[0])) is not None:
            return prepared
    raise RuntimeError("We couldn't find anyone in that picture. Try one where the whole character stands against a plain "
                       "background.")


def _only_them(ctx, crop, old0: np.ndarray, run_dir) -> tuple[str, int]:
    """SCAIL-2's object_indices: just the chosen person when someone else shares the crop (a hugging couple), and how
    many people SAM 3.1 saw on the drive's first frame. "" (everyone, as before) when there's one person or it's unclear."""
    try:
        png = common._frame_png(crop, 0, run_dir / "drive_first.png")
        job = ctx.run(graphs.select_frame(ctx.client.assets.from_file(png), [], None, text=DRIVE_LABEL, threshold=0.5, individual=True))
        people = [m for m in common._download_masks(job, run_dir / "drive_people") if m.any()]
    except Exception:  # noqa: BLE001 - without it SCAIL re-renders everyone in the crop, as before
        return "", 0
    if people and people[0].shape != old0.shape:
        old0 = upsize(old0, (people[0].shape[1], people[0].shape[0]))
    rank = drive_rank(people, old0)
    return ("" if rank is None else str(rank)), len(people)


def _follow(ctx, gen_clip, old0: np.ndarray, first_gen, run_dir, size: tuple[int, int]) -> list[np.ndarray] | None:
    """The new character alone through the render: SAM 3.1 tracking seeded with where the old person stood on the
    first frame (no label, so a robot or a T-rex counts too). None when the person isn't on the first frame or the
    tracking fails; SAM's labels then decide, as before."""
    if not old0.any():
        return None
    try:
        seed = run_dir / "new_seed.png"
        Image.fromarray(old0.astype(np.uint8) * 255).resize(Image.open(first_gen).size, Image.NEAREST).convert("RGB").save(seed)
        job = ctx.run(graphs.track_mask(ctx.client.assets.from_file(gen_clip), ctx.client.assets.from_file(seed),
                                        prefix="pickups/recast_track"))
        masks = _found(job, 1, run_dir / "track_new", size)
    except Canceled:
        raise
    except Exception:  # noqa: BLE001 - without it SAM's labels decide, as before
        return None
    return masks if any(m.any() for m in masks) else None


def _found(job, n_labels: int, out_dir, size: tuple[int, int]) -> list[np.ndarray]:
    """Per frame, the union of what SAM found for every label (graphs.select_any's PU_Mask, PU_Mask1, ...)."""
    found: list[np.ndarray] = []
    for j in range(n_labels):
        d = out_dir / f"l{j}"
        d.mkdir(parents=True, exist_ok=True)
        for o in job.get_outputs(f"PU_Mask{j if j else ''}"):
            o.to_file(d / o.name)
        for i, f in enumerate(sorted(d.glob("mask_*.png"))):
            m = upsize(np.asarray(Image.open(f).convert("L")) > 127, size)
            if i < len(found):
                found[i] |= m
            else:
                found.append(m)
    return found
