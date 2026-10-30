"""Car swap, v2: the pure image maths (no GPU, no ComfyUI): crops, the seam blend, the new car on the clean plate, and the
quality gate's measurements. tasks/car_swap.py is the job that feeds these.

The rules the blend keeps (owner review, Oct 19: "masking mixed AI and real backgrounds in a rectangle"):

* The region that changes is car-shaped: the new car's matte, its shadow and reflection band under it (the band follows the
  car's lower outline and fades out), and the old car's footprint (its outline, shadow and reflection), filled from the
  clean plate. Never a rectangle or a crop box.
* Outside that region's support the picture IS the footage, bit for bit.
* Plate and footage meet through a Laplacian-pyramid seam blend on a mask that is not dilated.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pickups import composite

MULT = 32                      # LTX sizes are multiples of this
EDIT_PX = 1280 * 704           # EditAnything v0.1 works at this many pixels (1280x704)
MIN_SCALE = 0.4                # a small car's crop is shown to the model at most 1/0.4 = 2.5x enlarged
CROP_MARGIN = 0.14             # of the car's path box: context around it
SEAM_LEVELS = 3                # Laplacian pyramid levels of the seam blend
OLD_DILATE_PX = 8              # px at HD: the old car's footprint grows this far (its soft edge)
SUPPORT_PX = 14                # px: the region's support grows this far past the mask for the seam's low frequencies


# --- the crop ------------------------------------------------------------------------------------------------------------


def crop_for(boxes: list[tuple[float, float, float, float]], size: tuple[int, int], max_px: int = EDIT_PX,
             margin: float = CROP_MARGIN) -> tuple[int, int, int, int, int, int]:
    """The crop around the car's whole path, for EditAnything: (x0, y0, cw, ch, w, h) in work pixels, the crop's own size
    (cw x ch) and the size the model sees (w x h, multiples of 32 with at most `max_px` pixels, the same aspect, so nothing is
    stretched). A path wider than the budget is shown smaller; a small car's crop is shown up to 1/MIN_SCALE enlarged. The
    crop is a box in the frame, never in the composite: only the car-shaped region it holds is used."""
    W, H = size
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    bw, bh = bw * (1 + 2 * margin) + 32, bh * (1 + 2 * margin) + 32
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    aspect = float(np.clip(bw / bh, 1.0, 2.4))
    h = max(MULT, round(np.sqrt(max_px / aspect) / MULT) * MULT)
    w = max(MULT, round(h * aspect / MULT) * MULT)
    while w * h > max_px * 1.02 and h > MULT:
        h -= MULT
        w = max(MULT, round(h * aspect / MULT) * MULT)
    a = w / h
    cw = max(bw, bh * a)
    cw = float(np.clip(cw, MIN_SCALE * w, max(W, H * a)))
    ch = cw / a
    if cw > W or ch > H:
        s = min(W / cw, H / ch)
        cw, ch = cw * s, ch * s
    cw, ch = int(round(cw)), int(round(ch))
    cw, ch = min(cw, W), min(ch, H)
    cx0 = int(np.clip(round(cx - cw / 2), 0, W - cw))
    cy0 = int(np.clip(round(cy - ch / 2), 0, H - ch))
    return cx0, cy0, cw, ch, w, h


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos resize of an RGB uint8 picture to (w, h)."""
    if (img.shape[1], img.shape[0]) == size:
        return img
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


def resize_float(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float map (H, W) to (w, h)."""
    if (a.shape[1], a.shape[0]) == size:
        return a.astype(np.float32)
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.BILINEAR))


def segments(n: int, k: int) -> list[tuple[int, int, int]]:
    """How to cover frames 0..n-1 with EditAnything runs that all start on the painted frame `k`: [(start, step, count)]
    with count = the run's real frames. k = 0: one run forward. Otherwise one forward from k and one backward from k
    (frame order reversed), each padded to 8m+1 frames by the caller."""
    if k <= 0:
        return [(0, 1, n)]
    out = [(k, 1, n - k)]
    out.append((k, -1, k + 1))
    return out


def windows(n: int, length: int = 49, stride: int = 24) -> list[tuple[int, int]]:
    """Overlapping windows (start, length) of 8m+1 frames covering n frames: starts every `stride`, the last one flush with the end.
    A clip shorter than `length` is one window of its own (rounded down to 8m+1)."""
    if n <= length:
        return [(0, 8 * ((n - 1) // 8) + 1)]
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] + length < n:
        starts.append(n - length)
    return [(a, length) for a in starts]


def window_weight(i: int, start: int, length: int, ramp: int = 12) -> float:
    """How much frame i of a window that begins at `start` counts: 1 in its middle, ramping down over `ramp` frames at both ends."""
    return float(np.clip(min(i - start + 1, start + length - i) / ramp, 0.0, 1.0))


def ltx_len(count: int) -> int:
    """The next 8m+1 length at or above `count`."""
    return 8 * (-(-(count - 1) // 8)) + 1


# --- seam blend ----------------------------------------------------------------------------------------------------------


def _down(a: np.ndarray) -> np.ndarray:
    return composite.gauss(a, 1.0)[::2, ::2]


def _up(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
    return np.stack([_up(a[..., c], shape) for c in range(a.shape[-1])], axis=-1)


def lap_blend(a: np.ndarray, b: np.ndarray, mask: np.ndarray, levels: int = SEAM_LEVELS) -> np.ndarray:
    """Laplacian-pyramid blend of b over a through `mask` (0..1; the mask is NOT dilated: the pyramid's coarse levels
    soften only the low frequencies across the seam, the fine detail changes exactly where the mask does). Floats."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    m = mask.astype(np.float32)
    pa, pb, pm = [a], [b], [m]
    for _ in range(levels):
        pa.append(_down(pa[-1]))
        pb.append(_down(pb[-1]))
        pm.append(_down(pm[-1]))
    out = pb[-1] * pm[-1][..., None] + pa[-1] * (1 - pm[-1][..., None])
    for i in range(levels - 1, -1, -1):
        shape = pa[i].shape[:2]
        la = pa[i] - _up(pa[i + 1], shape)
        lb = pb[i] - _up(pb[i + 1], shape)
        out = _up(out, shape) + lb * pm[i][..., None] + la * (1 - pm[i][..., None])
    return out


# --- matte ---------------------------------------------------------------------------------------------------------------


def matte_from(alpha: np.ndarray, k: float = 1.0, threshold: float = 0.5) -> np.ndarray:
    """A car's matte from a soft alpha (0..1): a cars's edge is crisp, so binarised, choked 1 px and feathered ~1 px: steadier
    frame to frame than the raw alpha's soft ramp, and no semi-transparent fringe of the render's own background."""
    core = alpha >= threshold
    core = composite.erode_box(core, max(1, round(1.0 * k))) if core.any() else core
    return np.clip(composite.gauss(core.astype(np.float32), 1.0 * k), 0.0, 1.0)


def jitter(mattes: list[np.ndarray], radius: int = 2) -> float:
    """How much the matte's area wobbles frame to frame: the mean of |area - median of its neighbours| over the area. A
    steady matte of a car that moves smoothly scores near 0 (it was ~0.7% for Alpha-Gen, 1-2% for SAM tracks)."""
    areas = np.array([float((m > 0.5).sum()) for m in mattes])
    if len(areas) < 3 or areas.max() <= 0:
        return 1.0
    dev = []
    for i in range(len(areas)):
        lo, hi = max(0, i - radius), min(len(areas), i + radius + 1)
        dev.append(abs(areas[i] - np.median(areas[lo:hi])))
    return float(np.mean(dev) / max(np.median(areas), 1.0))


def steadier(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[str, dict]:
    """Which of two mattes of the same car (Alpha-Gen `a`, SAM track `b`) is steadier over time. Alpha-Gen wins ties (and
    anything within 15%): its edge is cleaner. Returns ("alpha" | "sam", scores)."""
    ja, jb = jitter(a), jitter(b)
    return ("sam" if jb < 0.85 * ja else "alpha"), {"alpha_gen": round(ja, 4), "sam": round(jb, 4)}


# --- the shadow and reflection band --------------------------------------------------------------------------------------


def lower_band(car: np.ndarray, reach: float, side: float = 0.0, soft: float = 6.0) -> np.ndarray:
    """A band under the car that follows its lower outline: in every column the car spans (a little past it by `side` px),
    from just above the car's lowest point in that column to `reach` px below it, fading out downward, then feathered by
    `soft`. 0..1, the picture's size. Car-shaped (wheel arches and all), not a box."""
    h, w = car.shape
    out = np.zeros((h, w), np.float32)
    cols = np.flatnonzero(car.any(axis=0))
    if len(cols) == 0 or reach <= 0:
        return out
    bottom = h - 1 - np.argmax(car[::-1, cols], axis=0)
    lo, hi = max(0, int(cols[0] - side)), min(w, int(cols[-1] + side + 1))
    ext = np.arange(lo, hi)
    line = np.interp(ext, cols, bottom.astype(np.float32))
    yy = np.arange(h, dtype=np.float32)[:, None]
    fall = np.clip(1.0 - (yy - line[None, :]) / reach, 0.0, 1.0) * (yy >= (line[None, :] - 0.08 * reach))
    out[:, ext] = fall
    out = composite.gauss(out, soft) if soft > 0 else out
    out[out < 0.02] = 0.0
    return np.clip(out, 0.0, 1.0)


def road_zone(car: np.ndarray, line: tuple[np.ndarray, np.ndarray], down: float = 0.9, up: float = 0.12, side: float = 0.06) -> np.ndarray:
    """The ground around a car that can hold its shadow, spray and reflection (bool, the picture's size): in the columns the car
    spans (and `side` of its width past each end), from `up` of its height above its contact line `line` (xs, ys) down to `down`
    of its height below it. The wide SAM masks of "spray" and "shadow" took the sea and the sky round the car on a coast road; only
    this zone of them counts."""
    h, w = car.shape
    out = np.zeros((h, w), bool)
    rows = np.flatnonzero(car.any(axis=1))
    xs, ys = line
    if len(rows) == 0 or len(xs) == 0:
        return out
    tall, width = float(rows[-1] - rows[0] + 1), float(xs[-1] - xs[0] + 1)
    lo, hi = max(0, int(xs[0] - side * width)), min(w, int(xs[-1] + side * width + 1))
    ext = np.arange(lo, hi)
    ly = np.interp(ext, xs, ys)
    yy = np.arange(h, dtype=np.float32)[:, None]
    out[:, ext] = (yy >= (ly - up * tall)[None, :]) & (yy <= (ly + down * tall)[None, :])
    return out


def soft_weights(car: np.ndarray, shadow: np.ndarray, line: tuple[np.ndarray, np.ndarray], wet: bool, k: float = 1.0) -> np.ndarray:
    """How much of the old car's shadow, spray and reflection to take out of the footage, per pixel (0..1, no hard edge): where
    SAM found shadow or spray inside the road zone, and on a wet road a mirror image of the car under its contact line that fades
    to nothing a car-height down and at both ends. Not the car's body."""
    h, w = car.shape
    out = np.zeros((h, w), np.float32)
    rows = np.flatnonzero(car.any(axis=1))
    xs, ys = line
    if len(rows) == 0 or len(xs) == 0:
        return out
    tall, width = float(rows[-1] - rows[0] + 1), float(xs[-1] - xs[0] + 1)
    zone = road_zone(car, line)
    ws = np.clip(composite.gauss((shadow & zone).astype(np.float32), 16.0 * k) * 1.6, 0.0, 1.0)
    out = ws
    if wet:
        yy = np.arange(h, dtype=np.float32)[:, None]
        ext = np.arange(max(0, int(xs[0])), min(w, int(xs[-1] + 1)))
        ly = np.interp(ext, xs, ys)
        fall = np.clip(1.0 - (yy - ly[None, :]) / (0.9 * tall), 0.0, 1.0) * (yy >= (ly - 0.03 * tall)[None, :])
        t = np.clip(np.minimum(ext - xs[0], xs[-1] - ext) / (0.12 * width), 0.0, 1.0)
        fall = fall * (t * t * (3 - 2 * t))[None, :]
        out[:, ext] = np.maximum(out[:, ext], 0.9 * fall)
    out[car] = 0.0
    return np.clip(composite.gauss(out, 5.0 * k), 0.0, 1.0)


def shade_from_render(gen: np.ndarray, plate: np.ndarray, band: np.ndarray, car: np.ndarray, k: float = 1.0) -> np.ndarray:
    """The darkening the render drew under the new car (its contact shadow and its reflection, luminance only): the render's
    low-pass luma over the plate's, both blurred, measured against how the two roads differ elsewhere (a ring outside the car
    and its band), never above 1 (only darkens), and weighted by the band. (H, W) float: multiply the plate by it."""
    g = composite.gauss(composite.luma709(gen.astype(np.float32)), 4.0 * k)
    p = composite.gauss(composite.luma709(plate.astype(np.float32)), 4.0 * k)
    near = (band > 0) | car
    ring = composite.dilate_box(near, round(90 * k)) & ~composite.dilate_box(near, round(24 * k))
    gain = 1.0
    if ring.sum() > 500:
        gain = float(np.clip(np.median(g[ring]) / max(np.median(p[ring]), 1.0), 0.8, 1.25))
    ratio = np.clip(g / np.maximum(p * gain, 1.0), 0.25, 1.0)
    return 1.0 - (1.0 - ratio) * band


# --- plate matching --------------------------------------------------------------------------------------------------------


def lap_var(img: np.ndarray, where: np.ndarray) -> float:
    y = composite.luma709(img.astype(np.float32))
    lap = y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    m = where[1:-1, 1:-1]
    return float(lap[m].var()) if m.sum() > 200 else 0.0


def sharpen_to(img: np.ndarray, ref: np.ndarray, where_img: np.ndarray, where_ref: np.ndarray, sigma: float = 1.3,
               max_amount: float = 1.2) -> np.ndarray:
    """Unsharp mask on `img` (uint8) so its fine detail (Laplacian variance over `where_img`) matches `ref`'s over `where_ref`:
    an upscaled render or a plate is softer than the footage. Never softens. Returns uint8."""
    vi, vr = lap_var(img, where_img), lap_var(ref, where_ref)
    if vi <= 0 or vr <= vi * 1.15:
        return img
    lo, hi = 0.0, max_amount
    f = img.astype(np.float32)
    base = composite.gauss(f, sigma)
    for _ in range(6):
        mid = (lo + hi) / 2
        test = np.clip(np.rint(f + (f - base) * mid), 0, 255).astype(np.uint8)
        if lap_var(test, where_img) < vr:
            lo = mid
        else:
            hi = mid
    return np.clip(np.rint(f + (f - base) * hi), 0, 255).astype(np.uint8)


# --- the quality gate's measurements ---------------------------------------------------------------------------------------


def straight_seam(diff: np.ndarray, min_len: int) -> int:
    """The longest perfectly straight horizontal or vertical edge of the changed region (diff: bool, True where the output
    differs from the footage). A rectangle composite leaves one; a car-shaped region's outline has none that long. Returns the
    run's length in px (0 when none reaches 8)."""
    best = 0
    for d in (diff, diff.T):
        edge = d[:-1] != d[1:]
        for row in np.flatnonzero(edge.sum(axis=1) >= max(8, min_len)):
            x = edge[row]
            if not x.any():
                continue
            padded = np.r_[False, x, False]
            starts = np.flatnonzero(padded[1:] & ~padded[:-1])
            ends = np.flatnonzero(~padded[1:] & padded[:-1])
            best = max(best, int((ends - starts).max()))
    return best if best >= min_len else 0


def halo(out: np.ndarray, plate: np.ndarray, matte: np.ndarray, k: float = 1.0) -> float | None:
    """A light or dark fringe at the car's edge: the share of the car's 3 px inner rim whose luma is nearer the plate beside
    the car than the car's own body just inside it (the render's background showing at the edge). 0 = none."""
    core = matte > 0.5
    if core.sum() < 2000:
        return None
    ring = composite.erode_box(core, max(1, round(1 * k))) & ~composite.erode_box(core, max(2, round(4 * k)))
    inner = composite.erode_box(core, max(3, round(8 * k))) & ~composite.erode_box(core, max(5, round(16 * k)))
    out_ring = composite.dilate_box(core, max(3, round(6 * k))) & ~composite.dilate_box(core, max(2, round(2 * k)))
    if ring.sum() < 50 or inner.sum() < 50 or out_ring.sum() < 50:
        return None
    y = composite.luma709(out.astype(np.float32))
    p = composite.luma709(plate.astype(np.float32))
    body, bg = float(np.median(y[inner])), float(np.median(p[out_ring]))
    if abs(body - bg) < 12:  # no contrast: nothing to show
        return 0.0
    t = (y[ring] - body) / (bg - body)
    return float((t > 0.6).mean())


def flicker(mattes: list[np.ndarray], radius: int = 2) -> float:
    """Edge flicker: how far the matte's edge pops in and out against its neighbours in time, in px of edge per px of outline. For
    each frame, the pixels whose side of 0.5 differs from the median of the frames around it, over the length of the median's
    outline; the mean over frames. A steady edge (or one moving evenly) scores near 0; an edge that jumps a pixel back and forth
    scores about 1."""
    if len(mattes) < 3:
        return 0.0
    stack = np.stack([m.astype(np.float32) for m in mattes])
    vals = []
    for i in range(len(mattes)):
        lo, hi = max(0, i - radius), min(len(mattes), i + radius + 1)
        med = np.median(stack[lo:hi], axis=0) > 0.5
        outline = med & ~composite.erode_box(med, 1)
        if outline.sum() < 30:
            continue
        popped = (stack[i] > 0.5) != med
        vals.append(float(popped.sum() / outline.sum()))
    return float(np.mean(vals)) if vals else 0.0


def shape_stability(mattes: list[np.ndarray], moves: list[np.ndarray]) -> dict:
    """The car's shape from frame to frame: each matte moved onto the next frame by the old car's own motion (`moves[i]` maps
    frame i's picture to frame i+1's, 3x3), IoU of the two. A smooth car holds 0.9+; a morphing one dips."""
    ious = []
    for i in range(len(mattes) - 1):
        a = mattes[i] > 0.5
        b = mattes[i + 1] > 0.5
        if not a.any() or not b.any():
            ious.append(None)
            continue
        moved = composite.warp(a.astype(np.float32), moves[i], (a.shape[1], a.shape[0])) > 0.5
        inter = float((moved & b).sum())
        ious.append(round(inter / max(float((moved | b).sum()), 1.0), 4))
    seen = [x for x in ious if x is not None]
    return {"iou": ious, "min": min(seen) if seen else None, "mean": round(float(np.mean(seen)), 4) if seen else None}


def tyre_gaps(bottoms: list[tuple[float, float]], line_x: np.ndarray, line_y: np.ndarray, tall: float) -> dict:
    """Wheel contact: each visible tyre's bottom (x, y) against the road line the old car's tyres stood on (`line_x`, `line_y`,
    its contact line), as a share of the car's height. Negative: the tyre is below the line (nearer the camera, fine up to the
    tolerance); positive: it floats above the road."""
    gaps = []
    for x, y in bottoms:
        if line_x[0] - 0.05 * tall <= x <= line_x[-1] + 0.05 * tall:
            ly = float(np.interp(x, line_x, line_y))
            gaps.append(round((ly - y) / max(tall, 1.0), 4))
    return {"gaps": gaps, "max_float": max(gaps) if gaps else None, "max_sink": -min(gaps) if gaps else None}


def old_left_over(out: np.ndarray, src: np.ndarray, old: np.ndarray, new: np.ndarray) -> float | None:
    """Is the old car still showing: the share of its footprint (outside the new car) where the output is still the footage
    (within 6 levels) rather than the plate. A swap replaces them all; a car-coloured leftover is what this finds."""
    zone = old & ~new
    zone = composite.erode_box(zone, 2) if zone.sum() > 500 else zone
    if zone.sum() < 200:
        return None
    d = np.abs(out.astype(np.int16) - src.astype(np.int16)).max(axis=-1)
    return float((d[zone] <= 6).mean())


# --- the composite -------------------------------------------------------------------------------------------------------------


def grain_need(real: list[np.ndarray], other: list[np.ndarray]) -> np.ndarray | None:
    """Per-channel grain (std, 8-bit levels) that `other` lacks against the footage `real` (same regions, a few frames), or None."""
    need = np.sqrt(np.clip(composite.grain_profile(real) ** 2 - composite.grain_profile(other) ** 2, 0, None))
    return need if np.any(need > 0.05) else None


def compose(real: np.ndarray, plate: np.ndarray, gen: np.ndarray, matte: np.ndarray, old: np.ndarray, k: float, seed: int,
            tall: float, wet: bool = False, plate_need: np.ndarray | None = None, car_need: np.ndarray | None = None,
            wrap: float = 0.15, soft: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """One frame of the swap, in the crop. real/plate/gen: RGB uint8 of the crop's size; matte: the new car (0..1); old: bool,
    the old car's body outline; `soft`: its shadow, spray and reflection on the road as 0..1 weights (soft_weights), which fade
    into the footage with no hard edge (a hard one showed as a lighter rectangle of plate road beside the car). k: scale of edge widths (1 at HD); tall:
    the old car's height in px.

    The region that changes is car-shaped: the old footprint (from the plate), the new car, and a band under it that follows its
    lower outline (the render's shadow and reflection, luminance only). Returns (picture, support, stats): `support` (bool) is
    everything that may differ from `real`; outside it the picture IS `real`."""
    ch, cw = matte.shape
    footprint = composite.dilate_box(old, max(2, round(OLD_DILATE_PX * k)))
    o = np.clip(composite.gauss(footprint.astype(np.float32), 2.5 * k), 0.0, 1.0)
    if soft is not None and soft.any():
        o = np.maximum(o, soft.astype(np.float32))
    o[o < 0.03] = 0.0
    core = matte > 0.5
    reach = (0.45 if wet else 0.28) * tall
    band = lower_band(core, reach, side=0.0, soft=max(2.0, 6.0 * k)) * (matte < 0.99)
    live = (o > 0) | (matte > 0) | (band > 0)
    grown = composite.dilate_box(live, SUPPORT_PX)
    # the plate, toned to the footage by a membrane learnt around everything that changes, then as sharp and grainy as the footage
    fixed = composite.harmonize(plate, real, composite.dilate_box(live, max(2, round(6 * k))), grid=48, robust=True)
    fixed = np.clip(np.rint(fixed), 0, 255).astype(np.uint8)
    ring = composite.dilate_box(live, round(120 * k)) & ~composite.dilate_box(live, round(30 * k))
    fixed = sharpen_to(fixed, real, ring, ring)
    if plate_need is not None:
        fixed = composite.grain(fixed, plate_need, seed)
    fixed = retexture(fixed, real, live, o * (matte < 0.5), k)
    # the seam: the plate over the footage on the (undilated) footprint, through a Laplacian pyramid
    base = lap_blend(real, fixed, o)
    seam = composite.dilate_box(o > 0, SUPPORT_PX)
    base[~seam] = real[~seam]
    # what the render drew under the new car: darkening only, luminance only
    shade = shade_from_render(gen, fixed, band, core, k)
    base = base * shade[..., None]
    # the new car: as sharp and grainy as the footage, the plate's light wrapped round its edge
    car = sharpen_to(gen, real, ring, ring)
    if car_need is not None:
        car = composite.grain(car, car_need, seed + 7919)
    car = composite.light_wrap(car, matte, base, sigma=10 * k, width=max(3, round(6 * k)), amount=wrap)
    m = matte[..., None]
    out = np.clip(np.rint(car * m + base * (1 - m)), 0, 255).astype(np.uint8)
    support = grown | (band > 0)
    out[~support] = real[~support]
    stats = {"band_px": int((band > 0.05).sum()), "shade_min": round(float(shade.min()), 3)}
    return out, support, stats


def copy_score(real: np.ndarray, gen: np.ndarray, old: np.ndarray, new: np.ndarray) -> float | None:
    """Did the render just redraw the old car? The fine detail (luma minus a 3 px blur) of the render and of the footage,
    correlated over where the old and the new car overlap. A different car in the same place scores about 0 (-0.02 to 0.02 on four
    real swaps); the distilled model's SUV-again on the canyon bend scored 0.43."""
    inter = composite.erode_box(old & new, 6)
    if inter.sum() < 500:
        return None
    a = composite.luma709(real.astype(np.float32))
    b = composite.luma709(gen.astype(np.float32))
    a, b = (a - composite.gauss(a, 3.0))[inter], (b - composite.gauss(b, 3.0))[inter]
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


SPECK_LEVELS = 30.0       # a speck is at least this much brighter than the same pixel in the frames around it (luma, 0-255)
SPECK_LOCAL = 14.0        # ...and than its own neighbourhood in the frame
SPECK_UPPER = 0.55        # of the car's height from its top: the body, the bonnet, the roof (wheels and lights change legitimately)


def despeckle(gen: np.ndarray, around: list[np.ndarray], matte: np.ndarray, k: float = 1.0) -> tuple[np.ndarray, int]:
    """Render specks: isolated bright dots that pop onto the paint for a frame or two (an orange glint on C2's bonnet, frames 50-51),
    found as pixels inside the upper part of the car that are much brighter than the same pixel in the frames around it (the median
    of `around`: n-2, n-1, n+1, n+2) and than their own neighbourhood, and replaced from that median. Wheels and lamps, which change
    from frame to frame on purpose, are left alone (only the car's upper part is looked at). Returns (picture, pixels replaced)."""
    if len(around) < 2:
        return gen, 0
    core = composite.erode_box(matte > 0.5, max(2, round(3 * k)))
    rows = np.flatnonzero(core.any(axis=1))
    if len(rows) == 0:
        return gen, 0
    zone = core.copy()
    zone[int(rows[0] + SPECK_UPPER * (rows[-1] - rows[0])):] = False
    lum = composite.luma709(gen.astype(np.float32))
    stack = np.stack([composite.luma709(a.astype(np.float32)) for a in around])
    med = np.median(stack, axis=0)
    local = lum - composite.gauss(lum, 5.0 * k)
    bright = (lum - med) > SPECK_LEVELS
    above = sum(((lum - composite.luma709(a.astype(np.float32))) > SPECK_LEVELS / 2).astype(np.int8) for a in around)
    hit = zone & bright & (local > SPECK_LOCAL) & (above >= max(3, len(around) - 1))  # brighter than nearly every neighbour frame
    if not hit.any():
        return gen, 0
    # only small ones: a bright patch wider than ~12 px is a reflection moving over the paint, not a speck
    big = composite.dilate_box(composite.erode_box(bright, max(3, round(6 * k))), max(4, round(8 * k)))
    hit &= ~big
    if not hit.any():
        return gen, 0
    area = composite.dilate_box(hit, max(1, round(2 * k)))
    w = np.clip(composite.gauss(area.astype(np.float32), 1.2 * k) * 1.5, 0.0, 1.0)[..., None]
    fill = np.median(np.stack([a.astype(np.float32) for a in around]), axis=0)
    out = np.clip(np.rint(gen.astype(np.float32) * (1 - w) + fill * w), 0, 255).astype(np.uint8)
    return out, int(area.sum())


TEXTURE_SHIFTS = ((-0.28, 0.0), (0.28, 0.0), (0.0, -0.22), (0.0, 0.22), (-0.28, -0.22), (0.28, 0.22), (-0.28, 0.22), (0.28, -0.22))


def retexture(plate: np.ndarray, real: np.ndarray, live: np.ndarray, weight: np.ndarray, k: float = 1.0) -> np.ndarray:
    """The plate's smooth fill given the footage's own texture (the road under C2's bumper was a smoother, lighter patch than the
    gravel round it). The fine detail of the real picture (luma minus a 2 px blur) is copied in from other places of the same
    picture that are not part of the change (shifted by a quarter of the crop), at the energy the real picture has around the hole
    (its local mean |detail|, carried across the hole as a membrane) less what the plate already has. `weight`: 0..1 where the
    plate is used. uint8 in, uint8 out."""
    h, w = live.shape
    ry = composite.luma709(real.astype(np.float32))
    hp_r = ry - composite.gauss(ry, 2.0)
    py = composite.luma709(plate.astype(np.float32))
    hp_p = py - composite.gauss(py, 2.0)
    grown = composite.dilate_box(live, max(3, round(10 * k)))
    sig = 14.0 * k
    e_real = composite.gauss(np.abs(hp_r), sig)
    known = ~grown
    if known.sum() < 2000:
        return plate
    target = composite.harmonize(np.zeros((h, w, 3), np.float32), np.repeat(e_real[..., None], 3, axis=2), grown, grid=48)[..., 0]
    e_plate = composite.gauss(np.abs(hp_p), sig)
    need = np.sqrt(np.clip(target ** 2 - e_plate ** 2, 0.0, None))
    src = np.zeros((h, w), np.float32)
    have = np.zeros((h, w), bool)
    for fx, fy in TEXTURE_SHIFTS:
        dx, dy = int(round(fx * w)), int(round(fy * h))
        ys = np.clip(np.arange(h) + dy, 0, h - 1)
        xs = np.clip(np.arange(w) + dx, 0, w - 1)
        ok = known[ys][:, xs]
        take = ok & ~have
        if take.any():
            src[take] = hp_r[ys][:, xs][take]
            have |= take
    e_src = composite.gauss(np.abs(src), sig)
    noise = src / np.maximum(e_src, 1.0) * need * have
    noise = np.clip(noise, -24.0, 24.0) * np.clip(weight, 0.0, 1.0)
    return np.clip(np.rint(plate.astype(np.float32) + noise[..., None]), 0, 255).astype(np.uint8)


# --- anchored windows: the new car re-painted every ~24 frames --------------------------------------------------------------


ANCHOR_STRIDE, ANCHOR_WINDOW, ANCHOR_FADE = 24, 33, 6   # frames: a painted keyframe this often, each window this long, crossfade this long


def anchors(n: int, stride: int = ANCHOR_STRIDE) -> list[int]:
    """The frames the new car is painted on: 0, then every `stride`, leaving room for a window at the end (n=97: 0, 24, 48, 72)."""
    out = list(range(0, max(n - 8, 1), stride))
    return out or [0]


def anchored_weights(n: int, starts: list[int], fade: int = ANCHOR_FADE) -> list[list[float]]:
    """For each frame, the weight of each window (windows start at `starts`, each used until the next starts, then faded to the
    next over `fade` frames). Weights of a frame sum to 1."""
    out = []
    for i in range(n):
        w = [0.0] * len(starts)
        j = max(x for x in range(len(starts)) if starts[x] <= i)
        w[j] = 1.0
        if j > 0 and i - starts[j] < fade:
            t = (i - starts[j] + 0.5) / fade  # the new window's share: 0 at its first frame growing to 1
            w[j], w[j - 1] = t, 1.0 - t
        out.append(w)
    return out


# --- wheels that turn -------------------------------------------------------------------------------------------------------


def wheel_ellipse(mask: np.ndarray) -> tuple[float, float, np.ndarray, float] | None:
    """A wheel's centre (x, y), its covariance as a 2x2 matrix A that maps the unit circle onto the wheel's ellipse (the half axes),
    and its mean radius, from its mask. None when it is too small to turn visibly."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 150:
        return None
    pts = np.stack([xs, ys]).astype(np.float64)
    c = pts.mean(axis=1)
    cov = np.cov(pts - c[:, None])
    vals, vecs = np.linalg.eigh(cov)
    half = 2.0 * np.sqrt(np.maximum(vals, 1e-6))        # a filled ellipse has variance (half axis)^2 / 4
    A = vecs @ np.diag(half) @ vecs.T
    return float(c[0]), float(c[1]), A, float(np.sqrt(half[0] * half[1]))


def smooth_ellipses(masks: list[np.ndarray], radius: int = 3) -> list[tuple[float, float, np.ndarray, float] | None]:
    """A wheel's ellipse on every frame, steadied: its mask wobbles a little from frame to frame (a bit of arch or blur in, a bit of
    tyre out), which tilted the spun face. The ellipse's matrix and radius are the median over the frames around each one; the centre
    stays the frame's own (the wheel does move). None where the wheel is too small or missing."""
    raw = [wheel_ellipse(m) if m is not None and m.any() else None for m in masks]
    out: list = []
    for i, e in enumerate(raw):
        near = [x for x in raw[max(0, i - radius):i + radius + 1] if x is not None]
        if e is None or not near:
            out.append(None)
            continue
        A = np.median(np.stack([x[2] for x in near]), axis=0)
        out.append((e[0], e[1], A, float(np.median([x[3] for x in near]))))
    return out


def spin_wheel(img: np.ndarray, mask, phase: float, blur: float = 0.55, steps: int = 7, inner: float = 0.80, outer: float = 0.93
               ) -> np.ndarray:
    """Turn one wheel's face by `phase` radians (about its centre, in the wheel's own plane: the ellipse is unwarped to a circle,
    rotated, and warped back), with a rotational blur of `blur` radians (a shutter: the average of `steps` turns about it). The tyre's
    outer ring stays as it was, so the silhouette and the road contact don't move. A generated wheel keeps the same spoke
    pattern from frame to frame; this is what makes it roll. Returns a new uint8 picture."""
    e = mask if isinstance(mask, tuple) else wheel_ellipse(mask)   # an ellipse (smooth_ellipses), or a wheel's mask
    if e is None:
        return img
    cx, cy, A, rad = e
    if rad < 7:
        return img
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - 1.2 * rad * 1.6)), min(w, int(cx + 1.2 * rad * 1.6))
    y0, y1 = max(0, int(cy - 1.2 * rad * 1.6)), min(h, int(cy + 1.2 * rad * 1.6))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    Ainv = np.linalg.inv(A)
    u = Ainv[0, 0] * (xx - cx) + Ainv[0, 1] * (yy - cy)
    v = Ainv[1, 0] * (xx - cx) + Ainv[1, 1] * (yy - cy)
    r = np.sqrt(u * u + v * v)
    src = img[y0:y1, x0:x1].astype(np.float32)
    acc = np.zeros_like(src)
    for t in np.linspace(-0.5, 0.5, steps):
        a = phase + t * blur
        ca, sa = np.cos(a), np.sin(a)
        us, vs_ = ca * u + sa * v, -sa * u + ca * v
        sx = A[0, 0] * us + A[0, 1] * vs_ + cx
        sy = A[1, 0] * us + A[1, 1] * vs_ + cy
        sx, sy = np.clip(sx, 0, w - 1.001), np.clip(sy, 0, h - 1.001)
        ix, iy = sx.astype(np.int32), sy.astype(np.int32)
        fx, fy = (sx - ix)[..., None], (sy - iy)[..., None]
        s00, s10 = img[iy, ix].astype(np.float32), img[iy, np.minimum(ix + 1, w - 1)].astype(np.float32)
        s01, s11 = img[np.minimum(iy + 1, h - 1), ix].astype(np.float32), img[np.minimum(iy + 1, h - 1), np.minimum(ix + 1, w - 1)].astype(np.float32)
        acc += (s00 * (1 - fx) + s10 * fx) * (1 - fy) + (s01 * (1 - fx) + s11 * fx) * fy
    spun = acc / steps
    wt = np.clip((outer - r) / (outer - inner), 0.0, 1.0)[..., None]   # the face turns, the tyre's outer ring stays
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(np.rint(spun * wt + src * (1 - wt)), 0, 255).astype(np.uint8)
    return out


def drift_stats(real: np.ndarray, gen: np.ndarray, old: np.ndarray, new: np.ndarray, ref_colour: np.ndarray | None) -> dict:
    """Per frame: how much of the old car the render has gone back to (detail correlation with the footage's old car), how far the
    new car's colour is from the painted reference's, and how big the new car's matte is against the old car's."""
    core = composite.erode_box(new, 6)
    out: dict = {"copy": copy_score(real, gen, old, new), "colour": None, "area": float(new.sum()) / max(float(old.sum()), 1.0)}
    if ref_colour is not None and core.any():
        out["colour"] = float(np.abs(gen.astype(np.float32)[core].mean(axis=0) - ref_colour).mean())
    return out


def align_shift(a: np.ndarray, b: np.ndarray, region: np.ndarray, limit: float = 14.0) -> tuple[float, float]:
    """How far picture `b` sits from picture `a` inside `region` (bool), as the (dx, dy) that moves b onto a: phase correlation of their
    luma (mean-removed, only the region), the peak refined to a fraction of a pixel; (0, 0) when it can't tell or the shift is over
    `limit`. Two renders of the same car (neighbouring keyframe windows) differ by a few pixels."""
    if region.sum() < 400:
        return 0.0, 0.0
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    la, lb = composite.luma709(a[y0:y1, x0:x1].astype(np.float32)), composite.luma709(b[y0:y1, x0:x1].astype(np.float32))
    m = region[y0:y1, x0:x1].astype(np.float32)
    wy, wx = np.hanning(la.shape[0] + 2)[1:-1], np.hanning(la.shape[1] + 2)[1:-1]
    win = np.outer(wy, wx) * m
    la, lb = (la - (la * m).sum() / m.sum()) * win, (lb - (lb * m).sum() / m.sum()) * win
    f = np.fft.fft2(la) * np.conj(np.fft.fft2(lb))
    f /= np.maximum(np.abs(f), 1e-6)
    r = np.real(np.fft.ifft2(f))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)

    def refine(c: int, n: int, axis: int) -> float:
        lo, mid, hi = (r[(py - 1) % n, px], r[py, px], r[(py + 1) % n, px]) if axis == 0 else (r[py, (px - 1) % n], r[py, px], r[py, (px + 1) % n])
        d = lo - 2 * mid + hi
        return 0.0 if abs(d) < 1e-9 else 0.5 * (lo - hi) / d

    dy = py + refine(py, r.shape[0], 0)
    dx = px + refine(px, r.shape[1], 1)
    dy = dy - r.shape[0] if dy > r.shape[0] / 2 else dy
    dx = dx - r.shape[1] if dx > r.shape[1] / 2 else dx
    return (float(dx), float(dy)) if abs(dx) <= limit and abs(dy) <= limit else (0.0, 0.0)


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The picture moved by (dx, dy) pixels (bilinear; the edge it uncovers keeps the picture's own pixels)."""
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return img.astype(np.float32)
    M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    h, w = img.shape[:2]
    out = composite.warp(img.astype(np.float32), M, (w, h))
    cov = composite.warp(np.ones((h, w), np.float32), M, (w, h))
    return out + (1 - cov)[..., None] * img.astype(np.float32)
