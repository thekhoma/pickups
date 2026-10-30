"""Remove: erase what the user clicked (and its shadow) from a shot.

Real pixels first: on a locked-off shot every covered pixel is taken from the nearest frame where it is visible;
a Qwen-Image 2.1 clean plate fills only what is never uncovered (often nothing). Moving camera: LTX in/outpaint on a
crop around the object, started from the plate. Only the matte is pasted back onto the full-resolution frames, and
every changed pixel is recorded in the receipt (borrowed from other frames vs generated). Frames stream through
ffmpeg pipes; nothing full-resolution is written as images.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import align, composite, graphs, media
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.tasks import common

STAGES = [
    Stage("Preparing the shot", 20),
    Stage("Finding what you clicked", 25),
    Stage("Rebuilding the background", 60),
    Stage("Painting what was never seen", 60),
    Stage("Blending at full resolution", 90),
    Stage("Writing your files", 40),
]

LTX_LONG = 1024
# Locked-off shots are filled with real pixels from other frames (cheap), so they can run up to 10 s; a moving camera
# goes through LTX, which is tested at 97 frames, so those are trimmed to common.MAX_FRAMES.
LOCKED_MAX_S = 10.0
MAX_DRIFT = 0.02  # up to 2% of the frame of slow drift is aligned away and still filled with real pixels
BORROWED = np.array((61, 214, 245), np.float32)   # receipt: real pixels from other frames
GENERATED = np.array((255, 79, 216), np.float32)  # receipt: AI-generated pixels


def _upscale(m: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127


class Background:
    """What was behind the object, on a crop of the work clip: `gen[i]` fills `mattes_hard[i]` (grown and feathered when
    pasted); `never[i]` marks the generated (not borrowed) pixels. plate(i, frame) puts it back on a full frame."""

    def __init__(self, box, gen, mattes_hard, never, locked: bool, drift: float, never_any: bool, k: float, editor: str):
        self.box, self.gen, self.mattes_hard, self.never = box, gen, mattes_hard, never
        self.locked, self.drift, self.never_any, self.k, self.editor = locked, drift, never_any, k, editor
        self.n = len(gen)
        self.aligned = False
        self.clean_plate = False

    def plate(self, i: int, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Frame i with the object gone: (frame, touched, borrowed, generated (crop-sized masks), pixels changed
        outside the matte (0)). Past the filled frames the frame is returned as it is."""
        cx0, cy0, cw, ch = self.box
        if i >= self.n:
            z = np.zeros((ch, cw), bool)
            return full, z, z, z, 0
        region = full[cy0:cy0 + ch, cx0:cx0 + cw]
        matte = self.matte(i)
        g = composite.harmonize(self.gen[i], region, matte)
        pasted, c = composite.paste(region, g, matte)
        touched = matte > 0
        out = full.copy()
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        return out, touched, touched & ~self.never[i], touched & self.never[i], c

    cloned = False  # what is never uncovered was filled with texture from beside it, not a frame-editor plate

    def matte(self, i: int) -> np.ndarray:
        """Where frame i's fill is pasted (0..1, the crop's size): the object grown and feathered."""
        return composite.soft_matte(self.mattes_hard[i], round(10 * self.k), 6.0 * self.k)

    def method(self) -> str:
        if self.aligned and self.clean_plate:
            return "real pixels from other frames, aligned to the moving camera + LTX-2.5 Clean-Plate"
        if self.clean_plate:
            return "LTX-2.5 Clean-Plate, checked and put back under the mask"
        extra = " + texture from beside it" if self.cloned else f" + {self.editor} plate"
        return ("real pixels from other frames" + (", aligned to the moving camera" if self.aligned else "") +
                (extra if self.never_any else "")) if self.locked else \
            f"LTX-2.5 in/outpaint from a {self.editor} plate"

    def models(self) -> list[str]:
        if self.clean_plate:
            return ["LTX-2.5 Clean-Plate IC-LoRA"]
        return ([self.editor] if self.never_any and not self.cloned else []) + ([] if self.locked else ["LTX-2.5"])


NEVER_MAX = 0.60      # moving camera: most of what the object hid must be seen in some frame (else LTX fills it)
ALIGN_LONG = 960         # px: the long side frames are registered at (480 left the path edge a few px off)
ALIGN_MAX_RESID = 0.06  # mean luma difference (0..1) between a frame and the reference, around the object, after alignment


def _warp_back(img: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A picture in the reference's pixels put back in a frame's own (H: frame -> reference). A picture is edge-padded
    first: a frame that looks past the reference's edge (the camera moved forward) would otherwise get black there."""
    if img.dtype == np.uint8 and img.ndim == 3:
        P = round(0.2 * max(size))
        padded = np.pad(img, ((P, P), (P, P), (0, 0)), mode="edge")
        T = np.array([[1.0, 0, -P], [0, 1.0, -P], [0, 0, 1.0]])
        return align.to_ref(padded, np.linalg.inv(H) @ T, size)
    return align.to_ref(img, np.linalg.inv(H), size)


def _off_canvas(H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Where a frame's pixels fall outside the reference's picture (nothing real to borrow there)."""
    return ~align.valid(np.linalg.inv(H), size, size, margin=0)


def _blend_plate(real: np.ndarray, plate: np.ndarray, never: np.ndarray, hole: np.ndarray, footage: np.ndarray, k: float) -> np.ndarray:
    """`plate` laid over `real` where `never`: its low-frequency colour moved to match what surrounds it (the real fill,
    or the footage beyond the hole), then feathered in. Normalised convolution carries the colour step at the ring into
    the patch."""
    ref = np.where(hole[..., None], real, footage).astype(np.float32)
    ring = (composite.grow(never, round(16 * k)) & ~never).astype(np.float32)
    sigma = max(4.0, 24 * k)
    num = composite.blur((ref - plate.astype(np.float32)) * ring[..., None], sigma)
    den = composite.blur(ring, sigma)[..., None]
    field = num / np.maximum(den, 1e-3)
    field = np.where(den > 1e-3, field, 0.0)
    matched = np.clip(plate.astype(np.float32) + field, 0, 255)
    a = np.clip(composite.blur(composite.grow(never, round(3 * k)).astype(np.float32), max(1.5, 3 * k)), 0, 1)[..., None]
    return np.clip(np.rint(real.astype(np.float32) * (1 - a) + matched * a), 0, 255).astype(np.uint8)


def _smooth_fill(footage: np.ndarray, hole: np.ndarray, k: float) -> np.ndarray | None:
    """What lies behind a hole that is ringed by smooth picture (sky): the ring's colour carried in by normalised
    convolution at growing scales. None when the ring has texture (then borrowed or generated pixels are used)."""
    ring = composite.grow(hole, round(40 * k)) & ~composite.grow(hole, round(6 * k))
    if ring.sum() < 200:
        return None
    luma = composite.luma709(footage.astype(np.float32))
    if float((luma - composite.blur(luma, max(4.0, 12 * k)))[ring].std()) > SMOOTH_STD:
        return None
    known = (~composite.grow(hole, round(6 * k))).astype(np.float32)
    f = footage.astype(np.float32)
    out = np.zeros_like(f)
    filled = np.zeros(hole.shape, bool)
    for sigma in (8.0, 24.0, 64.0, 160.0):
        sg = max(3.0, sigma * k)
        w = composite.blur(known, sg)
        v = composite.blur(f * known[..., None], sg) / np.maximum(w, 1e-4)[..., None]
        take = (w > 0.15) & ~filled
        out[take] = v[take]
        filled |= take
    return np.clip(np.rint(np.where(filled[..., None], out, f)), 0, 255).astype(np.uint8)


def _ring_match(plate: np.ndarray, hard: np.ndarray, footage: np.ndarray, k: float) -> np.ndarray:
    """The plate's low-frequency colour moved to the footage on a ring just outside `hard`: the offset there (footage minus
    plate, both exist) is smoothed and carried over the whole frame, so the patch sits at the footage's level."""
    ring = (composite.grow(hard, round(40 * k)) & ~composite.grow(hard, round(8 * k))).astype(np.float32)
    if ring.sum() < 50:
        return plate
    sigma = max(8.0, 40 * k)
    num = composite.blur((footage.astype(np.float32) - plate.astype(np.float32)) * ring[..., None], sigma)
    den = np.maximum(composite.blur(ring, sigma), 1e-3)[..., None]
    return np.clip(np.rint(plate.astype(np.float32) + np.clip(num / den, -30, 30)), 0, 255).astype(np.uint8)


def _tone_match(fill: np.ndarray, hole: np.ndarray, footage: np.ndarray, k: float) -> np.ndarray:
    """A borrowed fill's low-frequency colour moved to the footage just outside the hole (another frame's sky or light is
    paler or darker than the one beside the hole: a ghost of the object's shape). The colour step at the edge is carried
    inward by normalised convolution and fades with distance from the edge."""
    if not hole.any():
        return fill
    inner = hole & ~composite.shrink(hole, max(2, round(10 * k)))
    outer = composite.grow(hole, max(2, round(10 * k))) & ~hole
    if not inner.any() or not outer.any():
        return fill
    sigma = max(6.0, 30 * k)
    f, o = fill.astype(np.float32), footage.astype(np.float32)
    a = composite.blur(f * inner[..., None], sigma) / np.maximum(composite.blur(inner.astype(np.float32), sigma), 1e-3)[..., None]
    b = composite.blur(o * outer[..., None], sigma) / np.maximum(composite.blur(outer.astype(np.float32), sigma), 1e-3)[..., None]
    reach = np.clip(composite.blur(inner.astype(np.float32), 2 * sigma) * 4.0, 0, 1)[..., None]
    field = np.clip(b - a, -40, 40) * reach
    return np.where(hole[..., None], np.clip(np.rint(f + field), 0, 255), f).astype(np.uint8)


def _aligned_fill(crops, crop_mask, shadow_crop, k: float, shadow: bool, make_plate, qa: dict, clean=None):
    """Moving camera, real pixels. Every frame is laid on one reference by a homography fitted in the ring around the
    object (align.clip); there the usual locked-off fill runs (what the object hid is taken from the nearest frame that
    shows it); the result goes back to each frame's own place. Returns (gen, mattes_hard, never, never_any) per
    frame in the crop's pixels, or None when the alignment doesn't hold or too much was never seen; `qa` (a dict, filled
    in place) says how it went."""
    n = len(crops)
    ch, cw = crops[0].shape[:2]
    size = (cw, ch)
    objs0 = [composite.grow(crop_mask(i), round(PERSON_PAD * k)) for i in range(n)]
    sh0 = [shadow_crop(i) if shadow_crop else np.zeros_like(objs0[0]) for i in range(n)]
    covers = [composite.grow(o | s_, round(16 * k)) for o, s_ in zip(objs0, sh0)]
    Hs, resid, m = align.clip(crops, covers, work_long=ALIGN_LONG, ring_px=round(80 * k))
    worst = max(resid)
    qa |= {"reference": m, "residual_max": round(float(worst), 4), "residual_mean": round(float(np.mean(resid)), 4),
           "residual": [round(float(r), 3) for r in resid]}
    if worst > ALIGN_MAX_RESID:
        qa["why_not"] = "the camera's move could not be followed around the object"
        return None
    al, valid, objs = [], [], []
    for i in range(n):
        al.append(align.to_ref(crops[i], Hs[i], size))
        valid.append(align.valid(Hs[i], size, size, margin=round(2 * k)))
        objs.append(align.to_ref(objs0[i] | sh0[i], Hs[i], size))
    obj_only = [align.to_ref(o, Hs[i], size) for i, o in enumerate(objs0)]
    if shadow:
        found = composite.cast_shadows(al, obj_only, feet_frac=0.4)
        band = composite.shadow_below(al, obj_only)
        objs = [o | f | b for o, f, b in zip(objs, found, band)]
    covered = [composite.grow(o, round(16 * k)) | ~v for o, v in zip(objs, valid)]
    gen_al, never_al = composite.temporal_fill(al, covered, tol=30)
    del al
    base_ref = gen_al[m].copy()  # the reference with everything seen filled in: what the plate is painted on
    gen = [_warp_back(g, Hs[i], size) for i, g in enumerate(gen_al)]
    del gen_al
    # Never-seen pixels still hold the object (temporal_fill leaves them as they were), and warping the fill back smears
    # a pixel or two of it past their edge: a dark hairline round the patch. Grown a few px, they cover it.
    never = [composite.grow(_warp_back(nv, Hs[i], size), round(4 * k)) for i, nv in enumerate(never_al)]
    mattes_hard = [_warp_back(o, Hs[i], size) | objs0[i] | sh0[i] for i, o in enumerate(objs)]
    never = [(nv | _off_canvas(Hs[i], size)) & composite.grow(mattes_hard[i], round(16 * k)) for i, nv in enumerate(never)]
    # One homography follows one depth. The upper part of a tall hole (a head against the sky, a wall) is far, and moves
    # differently from the ground the first fit followed, so that part is filled again from a fit made in the ring round
    # it alone, and replaces the first fill there, feathered. Without it the sky showed a pale ghost of the person.
    rows = np.nonzero(np.logical_or.reduce(mattes_hard).any(axis=1))[0]
    if len(rows) > 8 * k * 16:
        cut = int(rows.min() + FAR_SHARE * (rows.max() - rows.min()))
        top = np.zeros((ch, cw), bool)
        top[:cut] = True
        Hz, rz, _ = align.clip(crops, covers, ref=m, ring_px=round(80 * k), ring_from=[mh & top for mh in mattes_hard])
        qa["far_residual_max"] = round(float(max(rz)), 4)
        if max(rz) <= ALIGN_MAX_RESID:
            alz = [align.to_ref(crops[i], Hz[i], size) for i in range(n)]
            cov_z = [composite.grow(align.to_ref(mattes_hard[i], Hz[i], size), round(16 * k)) | ~align.valid(Hz[i], size, size, margin=round(2 * k))
                     for i in range(n)]
            gz, nz = composite.temporal_fill(alz, cov_z)
            del alz
            ramp = np.clip(((cut + 12 * k) - np.arange(ch, dtype=np.float32)) / max(24 * k, 1), 0, 1)[:, None, None]
            for i in range(n):
                g_i = _warp_back(gz[i], Hz[i], size)
                n_i = composite.grow(_warp_back(nz[i], Hz[i], size), round(4 * k)) & composite.grow(mattes_hard[i], round(16 * k))
                gen[i] = np.clip(np.rint(g_i * ramp + gen[i] * (1 - ramp)), 0, 255).astype(np.uint8)
                never[i] = np.where(np.arange(ch)[:, None] < cut, n_i | (never[i] & (np.arange(ch)[:, None] >= cut - 12 * k)), never[i])
            del gz, nz
            # Sky behind the head: borrowed pixels carry another frame's sky (a different place on its gradient), a pale ghost.
            # Where the ring round the top of the hole is smooth, the ring's own colour is carried in instead.
            for i in range(n):
                top_hole = mattes_hard[i] & top
                sf = _smooth_fill(crops[i], top_hole, k) if top_hole.any() else None
                if sf is not None:
                    a = composite.blur(composite.grow(top_hole, round(3 * k)).astype(np.float32), max(1.5, 3 * k))[..., None]
                    gen[i] = np.clip(np.rint(gen[i] * (1 - a) + sf * a), 0, 255).astype(np.uint8)
                    never[i] = never[i] & ~composite.grow(top_hole, round(6 * k))
    gen = [_tone_match(g, mh, crops[i], k) for i, (g, mh) in enumerate(zip(gen, mattes_hard))]
    hole = sum(int(mh.sum()) for mh in mattes_hard)
    unseen = sum(int(nv.sum()) for nv in never)
    share = unseen / max(hole, 1)
    qa |= {"never_seen_share": round(share, 4), "frames": n}
    if share > NEVER_MAX:
        qa["why_not"] = "too much of what the object hid is never seen in another frame"
        return None
    never_any = any(x.any() for x in never)
    cp = clean() if never_any and clean is not None else None
    if cp is not None:
        # What no frame shows comes from LTX Clean-Plate (it follows the camera, so no swim), colour-matched to the real
        # fill around it and feathered in; everything else in the hole stays real footage.
        for i in range(n):
            if never[i].any():
                gen[i] = _blend_plate(gen[i], cp[0][i], never[i], mattes_hard[i], crops[i], k)
        qa["never_seen_from"] = "clean-plate"
        qa["used"] = True
    elif never_any:
        union_al = np.logical_or.reduce([align.to_ref(nv, Hs[i], size) for i, nv in enumerate(never)])
        plate_arr = make_plate(composite.grow(union_al, round(8 * k)), base_ref)
        # The unseen part is often farther away than the ground the first fit followed (a hill behind someone walking
        # away): it moves differently, so the plate follows a fit made in the ring around that part alone.
        Hp, rp, _ = align.clip(crops, covers, ref=m, ring_px=round(80 * k), ring_from=never)
        qa["plate_residual_max"] = round(float(max(rp)), 4)
        if max(rp) > ALIGN_MAX_RESID:
            Hp = Hs
        target = composite.grain_profile(crops[:8])
        for i in range(n):
            if never[i].any():
                pl = composite.add_grain(_warp_back(plate_arr, Hp[i], size), target, seed=i)
                gen[i] = _blend_plate(gen[i], pl, never[i], mattes_hard[i], crops[i], k)
    return gen, mattes_hard, never, never_any


CLEAN_DIFF = 6.0      # levels: beside the hole, the plate differs from the shot by this much where a reflection or shadow was
EDGE_MAX = 0.45       # a render that keeps the thing's edges (their agreement with the shot's, in the hole) still has it; unrelated ground is far lower
SMOOTH_STD = 3.5      # levels: a ring whose fine detail (luma minus its blur) is this quiet is sky
FAR_SHARE = 0.45      # the top share of a hole (by height) is refilled with its own fit: it is far, the rest is ground
PERSON_PAD = 10       # px at 1920 wide: a person's mask is grown by this before anything is borrowed (hair against the sky, a strap)
GONE_MIN = 22.0       # a thing is gone when a render differs from the shot by at least this (levels) over where it was
CLEAN_PLATE_TRIES = 2  # one try and one retry on the next seed, then the inpaint fill


def _clean_plate(ctx, work: Path, crops: list, crop_vf: str, size: tuple[int, int], n: int, fps: float, run_dir: Path, seed: int,
                 label: str, fill: str, k: float, crop_mask, shadow_crop):
    """Moving camera, nothing to borrow: LTX Clean-Plate on the crop (no mask; it rebuilds the empty scene), checked that
    the thing is really gone on every frame, retried once. Returns (frames, mattes_hard) in the crop's pixels, or None
    (no LoRA on the machine, or still there after the retry) for the inpaint fill. The picture is put back only under
    the grown mask and shadow, so everything else stays the footage."""
    files = getattr(getattr(ctx, "conn", None), "model_files", None)
    try:
        have = set(files()) if callable(files) else set()
    except Exception:  # noqa: BLE001 - not knowing what is installed: the inpaint fill
        have = set()
    if not any(f.endswith(graphs.CLEAN_PLATE_LORA) for f in have):
        return None
    cw, ch = size
    lw, lh = media.work_size(cw, ch, LTX_LONG)
    clip = run_dir / "crop_clean.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"{crop_vf},scale={lw}:{lh}:flags=lanczos", "-frames:v", str(n),
                    "-an", "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", str(clip)], check=True)
    thing = (label or "object").strip() or "object"
    prompt = f"An empty scene with no {thing}. " + ((fill.rstrip(".") + ".") if fill else "Only the background remains.")
    holes = [crop_mask(i) | (shadow_crop(i) if shadow_crop else False) for i in range(n)]
    src = [np.asarray(Image.fromarray(f).resize((lw, lh), Image.BILINEAR)) for f in crops[:n]]
    small = [np.asarray(Image.fromarray(h.astype(np.uint8) * 255).resize((lw, lh), Image.NEAREST)) > 127 for h in holes]
    inner = [composite.shrink(h, max(2, round(6 * lw / cw * k))) for h in small]
    report = []
    for t in range(CLEAN_PLATE_TRIES):
        job = ctx.run(graphs.clean_plate_video(ctx.client.assets.from_file(clip), fps, lw, lh, prompt, negative=thing, seed=seed + t,
                                               prefix="pickups/remove_clean"))
        gdir = run_dir / f"clean{t}"
        gdir.mkdir(exist_ok=True)
        for o in job.get_outputs("PU_Out"):
            o.to_file(gdir / o.name)
        out = sorted(gdir.glob("frame_*.png"))
        if len(out) < n:
            report.append({"seed": seed + t, "frames": len(out), "gone": False})
            continue
        plate = [np.asarray(Image.open(f).convert("RGB")) for f in out[:n]]
        diffs = [float(np.abs(p.astype(np.int16) - s_.astype(np.int16)).max(axis=-1)[m].mean()) if m.any() else 255.0
                 for p, s_, m in zip(plate, src, inner)]
        # Gone means gone everywhere, and judged against the hole's own content, not the whole shot: a thing that is still there
        # keeps its edges (its folds, its hair), where a rebuilt scene does not. A pale top over cloud has little colour to
        # tell it from the plate, but its edges are all there.
        edges = []
        for p, s_, m in zip(plate, src, inner):
            e = composite.edge_agreement(composite.luma709(p.astype(np.float32)), composite.luma709(s_.astype(np.float32)), m)
            edges.append(0.0 if e is None else abs(float(e)))
        gone = min(diffs) >= GONE_MIN and min(diffs) >= 0.5 * float(np.median(diffs)) and max(edges) <= EDGE_MAX
        report.append({"seed": seed + t, "min_diff": round(min(diffs), 1), "median_diff": round(float(np.median(diffs)), 1),
                       "edge_agreement_max": round(max(edges), 3), "gone": gone})
        if gone:
            target = composite.grain_profile(crops[:8])
            full = [np.asarray(Image.fromarray(p).resize((cw, ch), Image.LANCZOS)) for p in plate]
            # The plate is clean, so it may also cover what the mask missed: a reflection or the soft edge of a shadow shows as
            # a clear difference from the shot beside the hole (within 150 px of it), at the car's rear edge too. Then grown.
            hard = []
            for i, h in enumerate(holes):
                d = np.abs(plate[i].astype(np.int16) - src[i].astype(np.int16)).max(axis=-1).astype(np.float32)
                d = composite.blur(d, max(2.0, 6 * lw / cw * k))
                extra = (d > CLEAN_DIFF) & composite.grow(small[i], max(4, round(150 * lw / cw * k)))
                extra_full = np.asarray(Image.fromarray(extra.astype(np.uint8) * 255).resize((cw, ch), Image.BILINEAR)) > 127
                hard.append(composite.grow(h | extra_full, round(20 * k)))
            # Frame by frame, the plate's colour is moved to the real picture on a ring just outside what it replaces (the
            # render is a little paler or darker than the footage, which showed as a haze at the old edge).
            gen = [composite.add_grain(_ring_match(f, hard[i], crops[i], k), target, seed=i) for i, f in enumerate(full)]
            if isinstance(getattr(ctx, "qa", None), dict):
                ctx.qa["remove_clean_plate"] = report
            return gen, hard
    if isinstance(getattr(ctx, "qa", None), dict):
        ctx.qa["remove_clean_plate"] = report
    return None


def background(ctx, work: Path, pm: list[np.ndarray], sh: list[np.ndarray] | None, size: tuple[int, int], fps: float,
               run_dir: Path, seed: int = 42, fill: str = "", shadow: bool = True, stage=None,
               camera: str | None = None, aligned_ok: bool = False, label: str = "", never_fill: str = "editor",
               given_locked: bool = False) -> Background:
    """Rebuild what is behind the object (masks `pm` at the proxy's size, its shadow `sh`) on the work clip `work`
    (size = (W, H)): real pixels from other frames on a locked-off shot, an LTX-2.5 in/outpaint from a frame-editor
    plate on a moving one. `fill` says what continues behind it. Remove runs it for a take; Recast's vehicle swap for
    the clean road its new vehicle drives on. `stage(label)` hears Remove's own stages. `camera` ("moving" or "locked")
    overrides the crop's own measure, which a big object that moves with the camera (a tracked car) reads as locked.
    `aligned_ok`: on a moving camera, try real pixels from other frames first (frames laid on one reference). `label`:
    what is being removed, in words ("SUV"): the LTX Clean-Plate prompt names it. `never_fill` "clone" fills what is never
    uncovered on a locked shot with texture from beside it instead of a frame-editor plate.
    `given_locked`: on a locked-off shot `sh` goes too (Remove finds its own shadows there; a vehicle swap's `sh` also
    holds the old car's reflection down a wet road, which no shadow search finds)."""
    stage = stage or (lambda label: None)
    W, H = size
    k = max(W, H) / 1920
    ph, pw = pm[0].shape
    # Crop = object (+ shadow) box over the whole shot + context, in work pixels.
    ys, xs = np.nonzero(np.any(np.stack([a | b for a, b in zip(pm, sh)] if sh else pm), axis=0))
    sx, sy = W / pw, H / ph
    bx0, bx1, by0, by1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    padx, pady = max((bx1 - bx0) * 0.75, 192 * k), max((by1 - by0) * 0.75, 192 * k)
    cx0, cx1 = int(max(0, bx0 - padx)), int(min(W, bx1 + padx))
    cy0, cy1 = int(max(0, by0 - pady)), int(min(H, by1 + pady))
    cw, ch = (cx1 - cx0) // 2 * 2, (cy1 - cy0) // 2 * 2
    crop_vf = f"crop={cw}:{ch}:{cx0}:{cy0}"

    stage("Rebuilding the background")
    crops = list(common.read_frames(work, crop_vf, (cw, ch)))
    n = len(crops)
    crop_mask = lambda i: _upscale(pm[min(i, len(pm) - 1)], W, H)[cy0:cy0 + ch, cx0:cx0 + cw]  # noqa: E731
    # Locked-off, or near enough: a tripod that creeps or a steady hand drifts a few px. Each frame's offset from frame 1
    # is measured and frames are aligned before pixels are borrowed, so those shots still get real-pixel fills.
    shifts = composite.frame_shifts(crops)
    drift = max(np.hypot(dx, dy) for dx, dy in shifts)
    locked = drift <= max(6.0, MAX_DRIFT * max(cw, ch)) if camera is None else camera == "locked"
    # Qwen-Image 2.1 never sees the app's selection, so the selection is painted pure red on the frame and the
    # instruction names "the red area" (Comfy's painted-mask method) and says what fills it. `fill` is the user's
    # optional description of what continues behind the object; LTX gets only that (Lightricks: describe the masked
    # area only, or an empty prompt; naming the scene paints people back in).
    fill = (fill or "").strip().rstrip(".")
    plate_instruction = ("Remove everything inside the red area and fill it with " +
                         (fill or "whatever continues behind it in the scene") +
                         ", matching the surrounding perspective, texture, grain and light. Keep everything outside the "
                         "red area exactly as it is.")

    safe = ctx.commercial_safe()
    first_crop = crops[0]

    def make_plate(red: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        first = run_dir / "first_crop.png"
        painted = (first_crop if base is None else base).copy()
        painted[red] = (255, 0, 0)
        Image.fromarray(painted).save(first)
        job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(first), plate_instruction, seed=seed, resolution=2048,
                                        prefix="pickups/plate", commercial_safe=safe, size=graphs.edit_size(cw, ch)))
        job.get_outputs("PU_Plate")[0].to_file(run_dir / "plate_raw.png")
        return np.asarray(Image.open(run_dir / "plate_raw.png").convert("RGB").resize((cw, ch), Image.LANCZOS))

    shadow_crop = (lambda i: _upscale(sh[min(i, len(sh) - 1)], W, H)[cy0:cy0 + ch, cx0:cx0 + cw]) if sh else None  # noqa: E731
    aligned = None
    if not locked and camera is None and aligned_ok:
        # A moving camera shows the ground again from another place: lay every frame on one reference and borrow real
        # pixels. Only when most of what the object hid is seen in some frame; else LTX fills it, below.
        stage("Rebuilding the background")
        report: dict = {}
        qa_clean_flag = report
        aligned = _aligned_fill(crops, crop_mask, shadow_crop, k, shadow, make_plate, report, clean=lambda c=crops, m=n: _clean_plate(
            ctx, work, c, crop_vf, (cw, ch), m, fps, run_dir, seed, label, fill, k, crop_mask, shadow_crop))
        if isinstance(getattr(ctx, "qa", None), dict):
            ctx.qa["remove_aligned"] = report
    if aligned is None and not locked:
        cap = common.length_cap(getattr(ctx, "params", None))
        if n > cap:
            # LTX fills the shot, at the length it is tested on.
            n = media.ltx_frames_down(cap)
            crops, pm = crops[:n], pm[:n]
        # LTX renders 8k+1 frames: a 60-frame clip came back as 57, and the last three frames repeated the 57th.
        n = max(min(n, media.ltx_frames_down(n)), 1)
        crops, pm = crops[:n], pm[:n]

    used_clean = False
    if aligned is not None:
        gen, mattes_hard, never, never_any = aligned
        used_clean = bool(qa_clean_flag.get("used"))
    elif locked:
        # Work in frame 1's coordinates (aligned), then put every result back where each frame has it.
        al = [composite.shift(crops[i], -dx, -dy) for i, (dx, dy) in enumerate(shifts)]
        objs = [composite.shift(crop_mask(i), -dx, -dy, fill=True) for i, (dx, dy) in enumerate(shifts)]
        if shadow:
            objs = [o | s | b for o, s, b in zip(objs, composite.cast_shadows(al, objs), composite.shadow_below(al, objs))]
        if sh and given_locked:  # what the caller says goes with it too (a vehicle's reflection down a wet road)
            given = lambda i: _upscale(sh[min(i, len(sh) - 1)], W, H)[cy0:cy0 + ch, cx0:cx0 + cw]  # noqa: E731
            objs = [o | composite.shift(given(i), -dx, -dy, fill=False) for i, (o, (dx, dy)) in enumerate(zip(objs, shifts))]
        covered = [composite.grow(m, round(16 * k)) for m in objs]
        gen, never = composite.temporal_fill(al, covered)
        del al
        gen = [composite.shift(g, dx, dy) for g, (dx, dy) in zip(gen, shifts)]
        never = [composite.shift(m, dx, dy, fill=True) for m, (dx, dy) in zip(never, shifts)]
        mattes_hard = [composite.shift(m, dx, dy, fill=False) | crop_mask(i) for i, (m, (dx, dy)) in enumerate(zip(objs, shifts))]
        never = [nv & composite.grow(mh, round(16 * k)) for nv, mh in zip(never, mattes_hard)]  # unknown borders only matter under the matte
        never_any = any(x.any() for x in never)
        stage("Painting what was never seen")
        if never_any and never_fill == "clone":
            # Texture from beside it in the same frame, nothing generated (Change weather's plate: LTX re-renders it,
            # and an object a frame editor imagined there would be drawn too).
            gen = [composite.clone_fill(gen[i], never[i]) if never[i].any() else gen[i] for i in range(n)]
        elif never_any:
            plate_arr = make_plate(composite.grow(np.logical_or.reduce(never), round(8 * k)))
            target = composite.grain_profile(crops[:8])
            gen = [np.where(never[i][..., None], composite.add_grain(plate_arr, target, seed=i), gen[i]) if never[i].any() else gen[i]
                   for i in range(n)]
    elif (cp := _clean_plate(ctx, work, crops, crop_vf, (cw, ch), n, fps, run_dir, seed, label, fill, k, crop_mask, shadow_crop)) is not None:
        gen, mattes_hard = cp
        never = [np.ones((ch, cw), bool)] * n
        never_any = True
        used_clean = True
    else:
        stage("Painting what was never seen")
        never_any = True
        mattes_hard = [crop_mask(i) | shadow_crop(i) if shadow_crop else crop_mask(i) for i in range(n)]
        never = [np.ones((ch, cw), bool)] * n
        plate = make_plate(composite.grow(mattes_hard[0], round(16 * k)))
        lw, lh = media.work_size(cw, ch, LTX_LONG)
        ltx_clip, mask_clip = run_dir / "crop_ltx.mp4", run_dir / "boxmask.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"{crop_vf},scale={lw}:{lh}:flags=lanczos",
                        "-frames:v", str(n), "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                        str(ltx_clip)], check=True)
        mw = common.FrameWriter(mask_clip, lw, lh, fps, kind="h264")
        for i in range(n):
            m = np.asarray(Image.fromarray(composite.grow(mattes_hard[i], round(12 * k)).astype(np.uint8) * 255).resize((lw, lh), Image.NEAREST))
            mw.write(np.dstack([m] * 3))
        mw.close()
        Image.fromarray(plate).resize((lw, lh), Image.LANCZOS).save(run_dir / "plate_ltx.png")
        job = ctx.run(graphs.remove(ctx.client.assets.from_file(ltx_clip), "", fps, seed=seed, dilate=0,
                                    prompt="The same place with nothing there, everything else unchanged.", prefix="pickups/remove",
                                    mask_clip=ctx.client.assets.from_file(mask_clip),
                                    plate=ctx.client.assets.from_file(run_dir / "plate_ltx.png")))
        gdir = run_dir / "gen"
        gdir.mkdir(exist_ok=True)
        for o in job.get_outputs("PU_Out"):
            o.to_file(gdir / o.name)
        files = sorted(gdir.glob("frame_*.png"))
        gen = [np.asarray(Image.open(files[min(i, len(files) - 1)]).convert("RGB").resize((cw, ch), Image.LANCZOS)) for i in range(n)]
    del crops
    bgd = Background((cx0, cy0, cw, ch), gen, mattes_hard, never, locked or aligned is not None, drift, never_any, k, graphs.edit_model(safe))
    bgd.aligned = aligned is not None
    bgd.clean_plate = used_clean
    bgd.cloned = locked and never_fill == "clone"
    return bgd


def run(ctx) -> dict:
    p = ctx.params
    seed = int(p.get("seed", 42))
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)

    ctx.stage("Preparing the shot")
    work = run_dir / "work.mp4"
    src = media.probe(ctx.input_path)
    start = max(0, min(int(p.get("start_frame") or 0), max(0, src.frames - 1)))
    # A bracket the user set (params.frames) is the length; otherwise a locked-off shot carries on up to LOCKED_MAX_S.
    span = common.length_cap(p) if p.get("frames") else max(common.length_cap(p), min(src.frames - start, int(LOCKED_MAX_S * float(src.fps))))
    info = common.work_clip(ctx, work, frames=span, start=start)
    W, H, fps = info["width"], info["height"], info["fps"]
    proxy = run_dir / "proxy.mp4"
    common.proxy_clip(ctx, work, proxy)
    src_w = media.probe(ctx.input_path).width

    ctx.stage("Finding what you clicked")
    # The click is a frame of the shot. The work clip begins at `start`, so the pick is that far into it.
    pm = common.tracked_masks(ctx, proxy, max(0, int(p.get("frame", start)) - start), p.get("points", []), p.get("text", ""), src_w, run_dir)
    if not any(m.any() for m in pm):
        raise RuntimeError("Couldn't follow the selection through the shot. Try clicking it again on a clearer frame.")

    # The object's own shadow from SAM (used on moving shots; locked-off shots find it against the clean background).
    sh = None
    if p.get("shadow", True):
        try:
            sh = common.shadow_masks(ctx, proxy, pm, run_dir)
        except Exception:  # noqa: BLE001 - a missing shadow is better than a failed take
            sh = None

    bg = background(ctx, work, pm, sh, (W, H), fps, run_dir, seed=seed, fill=p.get("fill") or "", shadow=bool(p.get("shadow", True)),
                    stage=ctx.stage, aligned_ok=True, label=str(p.get("text") or ""))
    n = bg.n

    ctx.stage("Blending at full resolution")
    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, fps, kind="h264", scale_long=1920)
    cx0, cy0, cw, ch = bg.box

    def blend(i: int, full: np.ndarray):
        out, _, b, a, c = bg.plate(i, full)
        reg = out[cy0:cy0 + ch, cx0:cx0 + cw].astype(np.float32)
        reg[b] = reg[b] * 0.45 + BORROWED * 0.55
        reg[a] = reg[a] * 0.45 + GENERATED * 0.55
        tinted = out.copy()
        tinted[cy0:cy0 + ch, cx0:cx0 + cw] = reg.astype(np.uint8)
        return out, tinted, c, int(b.sum()), int(a.sum())

    changed = borrowed = generated = 0
    frames = (f for i, f in enumerate(common.read_frames(work)) if i < n)
    for i, (out, tinted, c, nb, na) in enumerate(common.parallel_frames(frames, blend)):
        ctx.check()
        changed += c
        borrowed += nb
        generated += na
        out_w.write(out)
        rec_w.write(tinted)
        ctx.progress((i + 1) / n)
    out_w.close()
    rec_w.close()

    ctx.stage("Writing your files")
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=work)
    make_viewer(work, ctx.vdir / "before.mp4", frames=n)
    total = W * H * n
    receipt = {"original": round(100 * (1 - (borrowed + generated) / total), 2), "borrowed": round(100 * borrowed / total, 2),
               "ai": round(100 * generated / total, 2)}
    drift = bg.drift
    return {"label": "Remove", "task": "remove", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, n, fps), "camera": ("moving, aligned" if bg.aligned else "locked-off" if drift <= 6 else f"near-locked ({drift:.0f} px drift, aligned)")
                        if bg.locked else "moving", "method": bg.method(),
                        "pixels_changed_outside_selection": changed,
                        "models": ["SAM 3.1"] + bg.models()}}
