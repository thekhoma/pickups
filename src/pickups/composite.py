"""Paste generated pixels back over source frames, only where the fix is.

The rule the app promises: outside the matte, output pixels are the source pixels (checked before encoding).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a boolean mask by roughly `px` pixels (Gaussian spread + threshold; fast and round-cornered)."""
    if px <= 0:
        return mask.copy()
    im = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(px / 2))
    return np.asarray(im) > 8


def shrink(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode a boolean mask by roughly `px` pixels."""
    return ~grow(~mask, px) if px > 0 else mask.copy()


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-like blur (three box passes) of a float array (H, W) or (H, W, C), zero outside the image. Floats
    throughout, unlike PIL's 8-bit blur, so small weights stay precise."""
    if sigma <= 0:
        return a.astype(np.float32)
    r = max(1, int(round((np.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    out = a.astype(np.float32)
    for axis in (0, 1):
        for _ in range(3):
            out = _box(out, r, axis)
    return out


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """blur() without the dark rim: normalised by how much of the kernel is inside the image."""
    ones = blur(np.ones(a.shape[:2], np.float32), sigma)
    return blur(a, sigma) / (ones[..., None] if a.ndim == 3 else ones)


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def dehalo(img: np.ndarray, mask: np.ndarray, width: int, inner: int | None = None) -> np.ndarray:
    """Remove the light outline around a generated subject (SCAIL-2 draws a soft white glow around replaced people).

    A band from `inner` px inside the subject's edge (SAM's mask often swallows part of the glow) to `width` px outside
    it is rebuilt from the background beyond the band (normalised blur; the subject itself is never sampled). Only
    whitish pixels that are lighter than that background change, so a saturated jacket, skin and dark hair strands
    stay. Inside the mask a pixel must also be lighter than the subject's own interior nearby: a haze is lighter than
    both sides, while a white suit, pale skin or silver hair matches itself. Everything else is untouched. `width` must
    cover the whole glow, or the rebuild samples the glow itself."""
    if width <= 0 or not mask.any():
        return img
    inner = max(1, width // 3) if inner is None else inner
    outer = grow(mask, width)
    deep = shrink(mask, inner)
    band = outer & ~deep
    f = img.astype(np.float32)
    lum = np.array([0.299, 0.587, 0.114], np.float32)
    y = f @ lum

    def spread(x: np.ndarray, known: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised blur of x (H, W, C) from the `known` pixels only, and how much known there is nearby."""
        k = known.astype(np.uint8) * 255
        w = np.asarray(Image.fromarray(k).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0
        xs = (x * known[..., None]).clip(0, 255).astype(np.uint8)
        b = np.stack([np.asarray(Image.fromarray(xs[..., c]).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
                      for c in range(x.shape[-1])], axis=-1)
        return b / np.maximum(w[..., None], 1e-3), w

    def two_scale(x: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        near, w_near = spread(x, known, max(1.0, width * 0.6))
        far, w_far = spread(x, known, max(2.0, width * 2.0))  # for gaps narrower than the band (between legs, under an arm)
        mix = np.clip(w_near / 0.15, 0.0, 1.0)[..., None]
        return near * mix + far * (1 - mix), w_far

    fill, w_far = two_scale(f, ~outer)  # background beyond the glow only
    alpha = np.asarray(Image.fromarray(band.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(max(1.0, width / 4))),
                       dtype=np.float32) / 255.0
    alpha = alpha * (w_far > 0.01) * ~deep  # never deeper inside than `inner`, whatever the blur
    lift = y - fill @ lum
    sat = (f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1.0)
    halo = np.clip(lift / 18.0, 0.0, 1.0) * np.clip((0.28 - sat) / 0.14, 0.0, 1.0)  # lighter and whitish
    core = shrink(mask, inner + width // 4)
    if core.any():  # glow from 12 levels above the interior's light tones, certainly from 30
        y_in, w_in = _light_tones(y, core, width)
        above = np.clip((y - y_in - 12.0) / 18.0, 0.0, 1.0) * (w_in > 1e-4)
    else:  # a sliver of a subject: nothing to compare with, so nothing inside it is glow
        above = np.zeros_like(y)
    halo = np.where(mask, halo * above, halo)
    alpha = (alpha * halo)[..., None]
    return np.rint(np.clip(f * (1 - alpha) + fill * alpha, 0, 255)).astype(np.uint8)


def _light_tones(y: np.ndarray, known: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """dehalo's interior reference: the light tones of luma `y` over `known` (a power mean, so dark folds and seams
    don't drag it down: a lit edge or a white sole is no lighter than the best-lit parts of the same costume), carried
    out over the edge band from the nearest interior, wider where there is none close. Worked out at reduced size (it
    is smooth anyway) in floats, so a thin limb's small weights stay precise. Returns (luma, weight)."""
    h, w = y.shape
    s = max(1, round(width / 8))
    size = (max(1, round(w / s)), max(1, round(h / s)))

    def down(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize(size, Image.BOX))

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.BILINEAR))

    num, den = down(np.where(known, (y / 255.0) ** 4, 0.0)), down(known)
    n_near, w_near = blur(num, max(1.0, width * 0.6) / s), blur(den, max(1.0, width * 0.6) / s)
    n_far, w_far = blur(num, max(2.0, width * 2.0) / s), blur(den, max(2.0, width * 2.0) / s)
    mix = np.clip(w_near / 0.005, 0.0, 1.0)  # the nearest interior wherever there is any
    p = n_near / np.maximum(w_near, 1e-9) * mix + n_far / np.maximum(w_far, 1e-9) * (1 - mix)
    p = np.where(w_far > 1e-9, p, 1.0)  # no interior in reach: as light as can be, so nothing there reads as glow
    return up(np.clip(p, 0.0, 1.0) ** 0.25 * 255.0), up(np.maximum(w_near, w_far))


def soft_matte(mask: np.ndarray, grow_px: int, feather_px: float) -> np.ndarray:
    """Grown mask with a feathered edge, as float 0..1; 1 everywhere inside the grown mask."""
    hard = grow(mask, grow_px)
    if feather_px <= 0:
        return hard.astype(np.float32)
    soft = np.asarray(Image.fromarray(hard.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather_px))) / 255.0
    soft = soft.astype(np.float32)
    # Feather only outward, so the whole grown mask is fully replaced.
    soft[hard] = 1.0
    return soft


def attached_shadow(obj: np.ndarray, shadow: np.ndarray, reach_frac: float = 0.6, min_px: int = 24) -> np.ndarray:
    """Keep only the part of a shadow mask near the object (SAM's "shadow of X" also finds unrelated shadows).

    Shadow pixels are kept within `reach_frac` x the object's height (at least `min_px`) of the object mask.
    """
    if not obj.any() or not shadow.any():
        return np.zeros_like(shadow)
    ys = np.nonzero(obj)[0]
    reach = max(min_px, int((ys.max() - ys.min() + 1) * reach_frac))
    return shadow & grow(obj, reach)


def camera_shift(a: np.ndarray, b: np.ndarray, size: int = 256) -> float:
    """Global translation (px at full size) between two frames by phase correlation on edges.

    Edges ignore smooth changes (exposure, clouds); water and foliage add noise but don't move the peak much.
    """
    def prep(x):
        g = Image.fromarray(x).convert("L").resize((size, round(size * x.shape[0] / x.shape[1])), Image.BILINEAR)
        e = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return (e - e.mean()) * np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1]))
    fa, fb = np.fft.fft2(prep(a)), np.fft.fft2(prep(b))
    r = fa * np.conj(fb)
    corr = np.abs(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = dy - h if dy > h // 2 else dy
    dx = dx - w if dx > w // 2 else dx
    return float(np.hypot(dx, dy)) * a.shape[1] / size


def frame_offset(a: np.ndarray, b: np.ndarray, size: int = 320) -> tuple[int, int]:
    """Signed global translation (dx, dy) in full-size px such that b[y, x] ≈ a[y - dy, x - dx] (content moved by dx, dy)."""
    def prep(x):
        g = Image.fromarray(x).convert("L").resize((size, round(size * x.shape[0] / x.shape[1])), Image.BILINEAR)
        e = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return (e - e.mean()) * np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1]))
    fa, fb = np.fft.fft2(prep(a)), np.fft.fft2(prep(b))
    r = fb * np.conj(fa)
    corr = np.abs(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape

    def refine(c_minus: float, c0: float, c_plus: float) -> float:  # parabolic sub-pixel peak
        d = c_minus - 2 * c0 + c_plus
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_minus - c_plus) / d

    fx = refine(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    fy = refine(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dy = (py - h if py > h // 2 else py) + fy
    dx = (px - w if px > w // 2 else px) + fx
    k = a.shape[1] / size
    return int(round(dx * k)), int(round(dy * k))


PROFILE_MIN = (0.5, 0.3)  # (x, y): how well two edge profiles must match for their shift to count


def _edge_profile(img: np.ndarray, keep: np.ndarray, axis: int) -> np.ndarray:
    """The picture's edges across `axis` (0: horizontal edges, row by row; 1: vertical edges, column by column), averaged
    over `keep` along the other axis, its slow trend taken out."""
    g = luma709(img.astype(np.float32))
    gx, gy = sobel(gauss(g, 1.5))
    e = (gy if axis == 0 else gx) * keep  # signed: a horizon keeps its sign, waves and foliage average out
    k = keep.sum(axis=1 - axis).astype(np.float32)
    prof = e.sum(axis=1 - axis) / np.maximum(k, 1)
    prof[k < 0.1 * keep.shape[1 - axis]] = 0  # rows (columns) almost all masked: nothing to say
    return prof - gauss(prof[None, :], 15)[0]


def _profile_shift(pa: np.ndarray, pb: np.ndarray, r: int) -> tuple[float, float]:
    """(shift, score): pb ≈ pa moved by shift (sub-pixel), score its normalised correlation."""
    n = len(pa)
    r = max(1, min(r, n // 3))
    scores = np.zeros(2 * r + 1)
    for j, d in enumerate(range(-r, r + 1)):
        x, y = pa[max(0, -d):n - max(0, d)], pb[max(0, d):n - max(0, -d)]
        scores[j] = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    j = int(np.argmax(scores))
    f = 0.0
    if 0 < j < 2 * r:
        c = scores[j - 1] - 2 * scores[j] + scores[j + 1]
        f = 0.0 if abs(c) < 1e-12 else 0.5 * (scores[j - 1] - scores[j + 1]) / c
    return j - r + f, float(scores[j])


def masked_offset(a: np.ndarray, b: np.ndarray, keep: np.ndarray, max_shift: int = 64) -> tuple[float, float] | None:
    """Signed sub-pixel translation (dx, dy) with b[y, x] ≈ a[y - dy, x - dx], measured only where `keep`: the shift of
    b's row profile of horizontal edges against a's (dy), and of its column profile of vertical ones (dx). Averaging along
    rows keeps a horizon or a road edge and drops a stormy sea's waves, which a 2-D phase correlation matched anywhere
    (readings jumped 30-90 px from frame to frame). A component whose profiles match worse than PROFILE_MIN is 0; None
    when neither matches."""
    k = keep.astype(np.float32)
    if k.mean() < 0.08:
        return None
    dy, sy = _profile_shift(_edge_profile(a, k, 0), _edge_profile(b, k, 0), max_shift)
    dx, sx = _profile_shift(_edge_profile(a, k, 1), _edge_profile(b, k, 1), max_shift)
    okx, oky = sx >= PROFILE_MIN[0], sy >= PROFILE_MIN[1]
    if not (okx or oky):
        return None
    return (float(dx) if okx else 0.0), (float(dy) if oky else 0.0)


def frame_shifts(frames: list[np.ndarray]) -> list[tuple[int, int]]:
    """Each frame's content offset from frame 0 (dx, dy), chained frame to frame so slow drift is followed."""
    out = [(0, 0)]
    for i in range(1, len(frames)):
        dx, dy = frame_offset(frames[i - 1], frames[i])
        px, py = out[-1]
        out.append((px + dx, py + dy))
    return out


def shift(x: np.ndarray, dx: int, dy: int, fill=0) -> np.ndarray:
    """Translate an image or mask by (dx, dy) px; uncovered borders get `fill`."""
    if dx == 0 and dy == 0:
        return x
    out = np.full_like(x, fill)
    h, w = x.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = x[ys, xs]
    return out


def camera_is_locked(frames: list[np.ndarray], masks: list[np.ndarray], max_shift_px: float = 6.0) -> bool:
    """True if the camera barely moves between the first and last frame (tripod/locked-off shot)."""
    if len(frames) < 2:
        return True
    return camera_shift(frames[0], frames[-1]) <= max_shift_px


def cast_shadows(frames: list[np.ndarray], objs: list[np.ndarray], reach_frac: float = 0.8, ratio: float = 0.82,
                 step: int = 4, feet_frac: float | None = None) -> list[np.ndarray]:
    """Locked-off shots: find each object's cast shadow by comparing every frame with the clean background.

    The background is the per-pixel temporal median (the object moves, so the median shows what is behind it).
    A shadow is ground near the object that is clearly darker than that background; exposure drift (clouds) is
    normalised per frame. More reliable than asking a segmenter for "shadow of X", which finds other shadows.
    `feet_frac`: only look from this share of the object's height above its lowest point down (a shadow lies on the
    ground; on a moving camera a far wall that doesn't line up with the ground would read as dark).
    """
    sub = np.stack(frames[::step] if len(frames[::step]) % 2 else frames[::step][:-1])
    k = len(sub) // 2
    bg = np.partition(sub, k, axis=0)[k]
    del sub
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    bl = bg.astype(np.float32) @ w + 2.0
    h, wd = bl.shape
    out = []
    for f, o in zip(frames, objs):
        sh = np.zeros((h, wd), bool)
        if o.any():
            r = (f.astype(np.float32) @ w + 2.0) / bl
            r /= max(float(np.median(r[~o])), 1e-3)
            ys, xs = np.nonzero(o)
            reach = int((ys.max() - ys.min()) * reach_frac)
            y0, y1 = max(0, ys.min() - reach // 4), min(h, ys.max() + reach)
            if feet_frac is not None:
                y0 = max(y0, ys.max() - int(feet_frac * (ys.max() - ys.min() + 1)))
            x0, x1 = max(0, xs.min() - reach), min(wd, xs.max() + reach)
            near = np.zeros((h, wd), bool)
            near[y0:y1, x0:x1] = True
            sh = near & (r < ratio) & ~o
            # drop speckle: keep pixels whose neighbourhood is mostly shadow
            dens = np.asarray(Image.fromarray(sh.astype(np.uint8) * 255).filter(ImageFilter.BoxBlur(6)))
            sh = dens > 110
        out.append(sh)
    return out


def shadow_below(frames: list[np.ndarray], objs: list[np.ndarray], ratio: float = 0.93, reach_frac: float = 0.35,
                 side_frac: float = 0.3, step: int = 4) -> list[np.ndarray]:
    """The soft edge of a cast shadow that cast_shadows' firm threshold leaves: a band under and beside the object (to
    `reach_frac` x its height below its feet, `side_frac` either side) where the ground is darker than the clip's own
    background there (a lower bar, `ratio`, than cast_shadows', but only in the band, and only where the dark is solid)."""
    sub = np.stack(frames[::step] if len(frames[::step]) % 2 else frames[::step][:-1])
    k = len(sub) // 2
    bg = np.partition(sub, k, axis=0)[k]
    del sub
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    bl = bg.astype(np.float32) @ w + 2.0
    h, wd = bl.shape
    out = []
    for f, o in zip(frames, objs):
        sh = np.zeros((h, wd), bool)
        if o.any():
            r = (f.astype(np.float32) @ w + 2.0) / bl
            r /= max(float(np.median(r[~o])), 1e-3)
            ys, xs = np.nonzero(o)
            oh = ys.max() - ys.min() + 1
            band = np.zeros((h, wd), bool)
            band[max(0, ys.max() - int(0.15 * oh)):min(h, ys.max() + int(reach_frac * oh)),
                 max(0, xs.min() - int(side_frac * oh)):min(wd, xs.max() + int(side_frac * oh))] = True
            sh = band & (r < ratio) & ~o
            dens = np.asarray(Image.fromarray(sh.astype(np.uint8) * 255).filter(ImageFilter.BoxBlur(6)))
            sh = dens > 140
        out.append(sh)
    return out


def _latest_free(frames: list[np.ndarray], covered: list[np.ndarray], order) -> dict:
    """Walking the frames in `order`, at each frame's covered pixels: the last uncovered value seen and when (-1: never)."""
    h, w = covered[0].shape
    v1, t1 = np.zeros((h, w, 3), np.uint8), np.full((h, w), -1, np.int32)
    out = {}
    for i in order:
        free = ~covered[i]
        v1[free] = frames[i][free]
        t1[free] = i
        c = covered[i]
        out[i] = (v1[c], t1[c])
    return out


def clean_median(frames: list[np.ndarray], covered: list[np.ndarray], where: np.ndarray,
                 chunk: int = 150_000) -> tuple[np.ndarray, np.ndarray]:
    """Each `where` pixel's median over the frames where it is uncovered (in np.nonzero(where) order): what is really
    there on a locked-off shot, even when a mask missed a coat tail for a run of frames. Returns (median uint8 k x 3,
    how many uncovered samples each pixel had)."""
    import warnings

    ys, xs = np.nonzero(where)
    med = np.zeros((len(ys), 3), np.uint8)
    count = np.zeros(len(ys), np.int32)
    for s in range(0, len(ys), chunk):
        yy, xx = ys[s:s + chunk], xs[s:s + chunk]
        vals = np.stack([f[yy, xx] for f in frames]).astype(np.float32)
        free = ~np.stack([c[yy, xx] for c in covered])
        vals[~free] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # never-uncovered pixels: all NaN, left at 0
            med[s:s + chunk] = np.clip(np.rint(np.nan_to_num(np.nanmedian(vals, axis=0))), 0, 255)
        count[s:s + chunk] = free.sum(axis=0)
    return med, count


def temporal_fill(frames: list[np.ndarray], covered: list[np.ndarray], tol: int = 30) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Locked-off shots: fill each covered pixel with the same pixel from the nearest frame where it is uncovered.

    A person walking through a static shot uncovers the background somewhere in the clip, so most of the fill is real
    footage at full resolution, not a generated plate. Returns (filled frames, never-uncovered masks); only the
    never-uncovered pixels still need a plate.

    The nearest frame's pixel is used only when it agrees (within `tol` levels) with that pixel's median over every
    frame where it is uncovered; otherwise the median is used. A mask that misses part of the person for a run of frames
    (a coat tail) would otherwise copy them into the fill: a dark ribbon. The cost: something that really changes right
    beside the person (a tailgate swinging shut) takes its most common state there. tol=0: always the nearest frame.
    """
    n = len(frames)
    fwd = _latest_free(frames, covered, range(n))
    bwd = _latest_free(frames, covered, range(n - 1, -1, -1))
    ever = np.logical_or.reduce(covered)
    med = count = idx = None
    if tol and ever.any():
        med, count = clean_median(frames, covered, ever)
        idx = np.full(ever.shape, -1, np.int64)
        idx[ever] = np.arange(int(ever.sum()))
    far = 1 << 30
    out, never = [], []
    for i in range(n):
        f1, ft = fwd.pop(i)
        b1, bt = bwd.pop(i)
        has_f, has_b = ft >= 0, bt >= 0
        use_b = np.where(has_b, bt - i, far) < np.where(has_f, i - ft, far)
        val = np.where(use_b[:, None], b1, f1)
        if med is not None:
            j = idx[covered[i]]
            m, ok = med[j], count[j] > 0
            off = ok & (np.abs(val.astype(np.int16) - m.astype(np.int16)).max(axis=1) > tol)
            val[off] = m[off]
        o = frames[i].copy()
        o[covered[i]] = val
        nv = np.zeros(covered[i].shape, bool)
        nv[covered[i]] = ~has_f & ~has_b
        out.append(o)
        never.append(nv)
    return out, never


def _nanmedian(a: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmedian, NaN (and silent) where a slice has no values at all."""
    empty = np.isnan(a).all(axis=axis, keepdims=True)
    out = np.nanmedian(np.where(empty, 0.0, a), axis=axis)
    return np.where(np.squeeze(empty, axis=axis), np.nan, out)


ROBUST_TINT = 4.0  # levels: the most colour (beyond tone) a robust membrane may add


def harmonize(gen: np.ndarray, src: np.ndarray, matte: np.ndarray, grid: int = 96, iters: int = 1500,
              robust: bool = False) -> np.ndarray:
    """Make a fill blend seamlessly: add the smooth source-minus-fill difference, carried in from around the matte.

    A plate or a VAE round trip shifts tone unevenly (a wall darker, a table brighter), which no single gain fixes.
    The difference is measured outside the matte, then interpolated across it as a membrane (Laplace fill on a
    coarse grid), so the fill keeps its own detail but meets the source with no visible edge.
    """
    h, w = matte.shape
    d = max(2, round(max(h, w) / grid))
    hs, ws = -(-h // d), -(-w // d)
    valid = np.zeros((hs * d, ws * d), np.float32)
    valid[:h, :w] = matte <= 0
    res = np.zeros((hs * d, ws * d, 3), np.float32)
    res[:h, :w] = src.astype(np.float32) - gen.astype(np.float32)
    v = valid.reshape(hs, d, ws, d).mean(axis=(1, 3))
    if robust:  # each cell's median: a road line one picture has and the other hasn't doesn't tint the membrane
        cells = np.where(valid[..., None] > 0, res, np.nan).reshape(hs, d, ws, d, 3).transpose(0, 2, 1, 3, 4).reshape(hs, ws, d * d, 3)
        r = _nanmedian(cells, axis=2)
        r[v <= 0.5] = np.nan  # then each known cell's 5x5 neighbourhood's median: a line a cell or two wide is voted
        pr = np.pad(r, ((2, 2), (2, 2), (0, 0)), constant_values=np.nan)  # out too, even at the hole's rim
        near = np.stack([pr[a:a + hs, b:b + ws] for a in range(5) for b in range(5)], axis=2)
        r = np.nan_to_num(_nanmedian(near, axis=2)).astype(np.float32)
        y = r.mean(axis=-1, keepdims=True)  # and a tint, unlike a tone, is never more than a few levels
        r = y + np.clip(r - y, -ROBUST_TINT, ROBUST_TINT)
    else:
        r = (res * valid[..., None]).reshape(hs, d, ws, d, 3).sum(axis=(1, 3)) / np.maximum(v * d * d, 1e-6)[..., None]
    known = v > 0.5
    if not known.any() or known.all():
        return gen.astype(np.float32)
    f = np.where(known[..., None], r, r[known].mean(axis=0))
    for _ in range(iters):
        p = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
        f = np.where(known[..., None], r, (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25)
    up = np.stack([np.asarray(Image.fromarray(f[..., c]).resize((ws * d, hs * d), Image.BILINEAR)) for c in range(3)], axis=-1)
    return gen.astype(np.float32) + up[:h, :w]


OCCLUDER_MARGIN = 12.0  # depth levels (0-255, near is bright) an occluder must stand in front of the person
OCCLUDER_FEET = 0.06  # of the person's height: the ground band at their feet, nearer but never in front of them


def occluders(depth: np.ndarray, person: np.ndarray, area: np.ndarray, speck: int = 2) -> np.ndarray:
    """What stands in front of the person inside `area`: not the person, clearly nearer than them (above their own
    nearest tenth), and not the ground at their feet (nearer too, but below them). A father's hand passing in front of
    the child being recast was redrawn by the render and pasted see-through; with this it stays real footage.

    `depth` is the shot's own depth on this frame (near bright), at the same size as the masks."""
    if person.sum() < 50:
        return np.zeros_like(person)
    core = shrink(person, 2)
    near = float(np.percentile(depth[core if core.sum() >= 25 else person], 90))
    occ = area & ~person & (depth.astype(np.float32) > near + OCCLUDER_MARGIN)
    ys = np.flatnonzero(person.any(axis=1))
    occ[max(0, ys[-1] - int(OCCLUDER_FEET * (ys[-1] - ys[0] + 1))):] = False
    return grow(shrink(occ, speck), speck) & occ  # no specks


SUBJECT_LUMA_MAX = 10.0  # levels: how far the tone match may brighten or darken a new character


def harmonize_around(gen: np.ndarray, src: np.ndarray, matte: np.ndarray, subject: np.ndarray,
                     luma_max: float = SUBJECT_LUMA_MAX) -> np.ndarray:
    """`harmonize`, but the new character keeps its own colours: the fill around them gets the full correction, the
    character (`subject`, 0..1) only its brightness part, clamped. The difference is learned from the render's
    re-drawn background, and LTX draws grass a warmer green than the footage: carried across a recast woman, that
    correction turned her brown skirt purple once grass surrounded her."""
    g = gen.astype(np.float32)
    off = harmonize(gen, src, matte) - g
    luma = np.clip(off.mean(axis=-1, keepdims=True), -luma_max, luma_max)
    s = np.clip(subject.astype(np.float32), 0.0, 1.0)[..., None]
    return g + off * (1.0 - s) + luma * s


def grain_profile(frames: list[np.ndarray]) -> np.ndarray:
    """Per-channel noise level of the source: robust (MAD) spread of frame minus a light blur, median over frames.

    The median ignores edges and texture, which are a minority of pixels, so this measures grain, not detail.
    """
    stds = []
    for f in frames:
        blur = np.asarray(Image.fromarray(f).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)
        hp = (f.astype(np.float32) - blur).reshape(-1, 3)
        stds.append(1.4826 * np.median(np.abs(hp - np.median(hp, axis=0)), axis=0))
    return np.median(np.stack(stds), axis=0)


def add_grain(frame: np.ndarray, target_std: np.ndarray, seed: int, size: float = 0.6) -> np.ndarray:
    """Top up a generated frame's fine detail to the source's level with fresh per-frame grain."""
    have = grain_profile([frame])
    need = np.sqrt(np.clip(target_std ** 2 - have ** 2, 0, None))
    if not np.any(need > 0.05):
        return frame
    return grain(frame, need, seed, size)


def grain(frame: np.ndarray, need: np.ndarray, seed: int, size: float = 0.6) -> np.ndarray:
    """Add fresh grain of `need` (per-channel std, 8-bit levels) to a frame, whatever it already has."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(frame.shape[:2]).astype(np.float32)
    if size > 0:
        noise = np.asarray(Image.fromarray(((noise * 40) + 128).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(size)),
                           dtype=np.float32)
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    # Luma-correlated grain (same pattern on all channels), scaled per channel.
    out = frame.astype(np.float32) + noise[..., None] * need[None, None, :]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def paste(src: np.ndarray, gen: np.ndarray, matte: np.ndarray) -> tuple[np.ndarray, int]:
    """Composite and report how many pixels outside the matte differ from the source (should be 0)."""
    a = matte[..., None]
    out = np.clip(np.rint(src.astype(np.float32) * (1 - a) + gen.astype(np.float32) * a), 0, 255).astype(np.uint8)
    outside = matte == 0
    changed = int(np.count_nonzero(np.any(out[outside] != src[outside], axis=-1)))
    return out, changed


# --- vehicle swap helpers (tasks/vehicle_swap.py): warps, exact morphology, a true Gaussian, the car-over-plate blend ---


def gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """A true Gaussian blur (sampled kernel to 3 sigma, mirrored edges) of a float array (H, W) or (H, W, C). blur()'s
    three box passes are a coarse stand-in at a pixel or two; a 1 px choke and 1.1 px feather need the real thing."""
    a = a.astype(np.float32)
    if sigma <= 0:
        return a
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    out = a
    for axis in (0, 1):
        n = out.shape[axis]
        rr = min(r, n - 1)
        pad = [(0, 0)] * out.ndim
        pad[axis] = (rr, rr)
        p = np.pad(out, pad, mode="reflect") if rr else out
        acc = np.zeros_like(out)
        for j in range(-rr, rr + 1):
            acc += k[j + r] * np.take(p, np.arange(rr + j, rr + j + n), axis=axis)
        out = acc / k[r - rr:r + rr + 1].sum()
    return out


def dilate_box(mask: np.ndarray, rx: int, ry: int | None = None) -> np.ndarray:
    """Exact dilation by a (2rx+1) x (2ry+1) rectangle (cumulative sums: any pixel set in the window)."""
    ry = rx if ry is None else ry
    out = mask.astype(np.int32)
    for axis, r in ((1, rx), (0, ry)):
        if r <= 0:
            continue
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad), axis=axis)
        n = out.shape[axis]
        out = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis) - np.take(c, np.arange(n), axis=axis)
    return out > 0


def erode_box(mask: np.ndarray, rx: int, ry: int | None = None) -> np.ndarray:
    """Exact erosion by a rectangle (outside the image counts as set, as OpenCV's default border does)."""
    ry = rx if ry is None else ry
    if rx <= 0 and ry <= 0:
        return mask.copy()
    p = np.pad(~mask, ((ry, ry), (rx, rx)), constant_values=False)
    return ~dilate_box(p, rx, ry)[ry:ry + mask.shape[0], rx:rx + mask.shape[1]]


def choke1(mask: np.ndarray) -> np.ndarray:
    """Erode by one pixel with a 3x3 cross (OpenCV's 3x3 ellipse)."""
    p = np.pad(mask, 1, constant_values=True)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def warp(img: np.ndarray, M: np.ndarray, size: tuple[int, int], cubic: bool = False) -> np.ndarray:
    """img (H, W) or (H, W, C) moved by the 3x3 matrix M (affine or perspective, source pixel -> output pixel, corner-
    origin coordinates), on an output of size (w, h); zero where nothing maps. Float32 per channel through PIL."""
    Mi = np.linalg.inv(np.asarray(M, np.float64))
    Mi = Mi / Mi[2, 2]
    affine = abs(Mi[2, 0]) < 1e-12 and abs(Mi[2, 1]) < 1e-12
    kind = Image.AFFINE if affine else Image.PERSPECTIVE
    data = tuple(Mi[:2].ravel()) if affine else tuple(Mi.ravel()[:8])
    rs = Image.BICUBIC if cubic else Image.BILINEAR
    a = img.astype(np.float32)
    chans = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    out = [np.asarray(Image.fromarray(np.ascontiguousarray(c), "F").transform(size, kind, data, resample=rs)) for c in chans]
    return out[0] if a.ndim == 2 else np.stack(out, axis=-1)


def sobel(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3x3 Sobel gradients (x, y) of a float image, mirrored edges."""
    p = np.pad(g.astype(np.float32), 1, mode="reflect")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx, gy


def edge_agreement(a: np.ndarray, b: np.ndarray, where: np.ndarray, sigma: float = 1.5, shift: int = 0) -> float | None:
    """How alike two greyscale pictures' edges are over `where`: the normalised dot product of their gradients (1 = the
    same shape, near 0 = unrelated). A body that morphs from frame to frame scores low against its first frame.

    `shift` > 0: the best over `a` moved by up to that many pixels each way, so a body that is the same shape but sits a
    pixel or two off (LTX draws every frame after a pinned keyframe ~2 px lower) isn't scored as a morph."""
    if not where.any():
        return None
    ys, xs = np.nonzero(where)
    y0, y1 = max(0, ys.min() - shift), min(where.shape[0], ys.max() + 1 + shift)
    x0, x1 = max(0, xs.min() - shift), min(where.shape[1], xs.max() + 1 + shift)
    a, b, where = a[y0:y1, x0:x1], b[y0:y1, x0:x1], where[y0:y1, x0:x1]
    ax, ay = sobel(gauss(a, sigma))
    bx, by = sobel(gauss(b, sigma))
    bw, bb = bx[where], by[where]
    den_b = float((bw ** 2 + bb ** 2).sum())
    best = None
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            sx, sy = (np.roll(ax, (dy, dx), (0, 1)), np.roll(ay, (dy, dx), (0, 1))) if dy or dx else (ax, ay)
            aw, ab = sx[where], sy[where]
            den = float(np.sqrt((aw ** 2 + ab ** 2).sum() * den_b))
            if den > 0:
                e = float((aw * bw + ab * bb).sum()) / den
                best = e if best is None else max(best, e)
    return best


def luma709(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def car_on_ground(gen: np.ndarray, plate: np.ndarray, matte: np.ndarray, ground: np.ndarray, hard: np.ndarray,
                  need: np.ndarray | None, seed: int, k: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """A rendered vehicle on the clean plate (both RGB uint8, same size), standing on the render's own road.

    `matte` (0..1) is the new vehicle (its smoothed, choked, feathered track); `ground` (0..1) the road it stands on:
    under its body, between and below its wheels, fading out a little under it. Both come from the render: there its
    tyres touch the road, the road darkens under it and a wet road mirrors it. Cut out on its own and laid on the plate,
    the same vehicle floated (the plate has no contact shadow) and the plate's centre line ran through a front tyre the
    render had drawn the line behind. The render's road is brought to the plate's tone first: their smooth difference,
    measured around (never in) the vehicle's ground, carried across it as a membrane (`harmonize`), so its shadow stays
    and its edge meets the plate unseen. `hard` is this frame's raw track (the vehicle's colour match is measured around
    it, clamped). Grain `need` (per channel) goes on all of the render that is used.

    Returns (picture, ground weight outside the vehicle 0..1, stats); where matte and ground are both 0 the picture IS
    the plate, bit for bit."""
    h, w = matte.shape
    g, p = gen.astype(np.float32), plate.astype(np.float32)
    car, gr = matte > 0, np.clip(ground.astype(np.float32), 0.0, 1.0)
    gain = np.ones(3, np.float32)
    stats: dict = {"gain": [1.0, 1.0, 1.0], "shadow_min": 1.0}
    if hard.any():  # colour: the render's own background against the plate, in a ring 24-96 px out (clamped)
        ring = dilate_box(hard, round(96 * k)) & ~dilate_box(hard, round(24 * k)) & (gr <= 0)
        if ring.sum() > 1000:
            gain = np.clip(np.median(p[ring], axis=0) / np.maximum(1, np.median(g[ring], axis=0)), 0.88, 1.12).astype(np.float32)
    road = g
    if (gr > 0).any():
        # The membrane is learnt clear of the render's shadow: around the ground and the vehicle, grown by a margin.
        away = dilate_box((gr > 0) | car, round(24 * k)).astype(np.float32)
        road = harmonize(gen, plate, away, grid=48, robust=True)
    gq = np.clip(np.rint(g * gain), 0, 255).astype(np.uint8)
    rq = np.clip(np.rint(road), 0, 255).astype(np.uint8)
    if need is not None and np.any(need > 0.05):
        gq, rq = grain(gq, need, seed), grain(rq, need, seed + 7919)
    m = matte[..., None]
    under = gr[..., None] * rq.astype(np.float32) + (1 - gr[..., None]) * p
    out = np.clip(np.rint(gq.astype(np.float32) * m + under * (1 - m)), 0, 255).astype(np.uint8)
    wt = gr * (1 - matte)
    keep = ~(car | (gr > 0))
    out[keep] = plate[keep]  # bit-exact: rounding can never touch the plate outside the vehicle and its ground
    shade = (wt > 0.5)
    if shade.sum() > 50:
        ratio = gauss(luma709(out.astype(np.float32)), 2.0 * k) / np.maximum(gauss(luma709(p), 2.0 * k), 1)
        stats["shadow_min"] = round(float(np.clip(ratio[shade], 0, 9).min()), 3)
    stats["gain"] = [round(float(x), 3) for x in gain]
    return out, wt, stats

# --- C3: the original subject as the reference for the new one -----------------------------------------------------------


def subject_stats(img: np.ndarray, mask: np.ndarray) -> dict:
    """Light, colour, grain and softness of `img` inside `mask` (a few thousand pixels of the subject's interior)."""
    f = img.astype(np.float32)
    y = f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
    hp = y - blur(y, 0.9)
    v = hp[mask]
    lap = y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    m = mask[1:-1, 1:-1]
    chroma = np.sqrt(((f - y[..., None]) ** 2).sum(-1))
    return {"mean": float(y[mask].mean()), "std": float(y[mask].std()), "chroma": float(chroma[mask].mean()),
            "sigma": float(1.4826 * np.median(np.abs(v - np.median(v)))), "lap": float(lap[m].var())}


def smooth_targets(stats: list[dict], radius: int = 4) -> list[dict]:
    """Each statistic averaged over a window of frames: the reference breathes with the real footage, not with its noise."""
    keys = stats[0].keys()
    out = []
    for i in range(len(stats)):
        lo, hi = max(0, i - radius), min(len(stats), i + radius + 1)
        out.append({k: float(np.mean([s[k] for s in stats[lo:hi]])) for k in keys})
    return out


def real_grain(plate: np.ndarray, shape: tuple[int, int], seed: int) -> np.ndarray:
    """The footage's own grain pattern: the plate minus its denoised self, a random patch of it per frame. float HxWx3, unit
    scale per channel left as measured."""
    f = plate.astype(np.float32)
    hp = f - smooth(f, 1.2)
    rng = np.random.default_rng(seed)
    h, w = shape
    ph, pw = hp.shape[:2]
    if ph < h or pw < w:
        hp = np.tile(hp, (-(-h // ph), -(-w // pw), 1))
        ph, pw = hp.shape[:2]
    y0, x0 = int(rng.integers(0, ph - h + 1)), int(rng.integers(0, pw - w + 1))
    return hp[y0:y0 + h, x0:x0 + w]


def match_subject(fg: np.ndarray, mask: np.ndarray, target: dict, plate: np.ndarray, seed: int, limit: float = 0.15,
                  max_soften: float = 1.4, lap_tol: float = 1.1, max_ratio: float = 2.5) -> np.ndarray:
    """Pull the new subject's pixels (`fg`, uint8, statistics measured inside `mask`) towards the original subject's `target`:
    exposure within +-`limit`, softness by Laplacian variance (only softens, up to `max_soften` px of blur),
    grain topped up to the footage's level with the plate's own grain. Never repaints colours: identity stays."""
    f = fg.astype(np.float32)
    st = subject_stats(fg, mask)
    y = f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
    g = float(np.clip(target["mean"] / max(st["mean"], 1.0), 1 - limit, 1 + limit))
    k = float(np.clip(target["std"] / max(st["std"], 1.0), 1 - limit, 1 + limit))
    ym = st["mean"]
    y2 = (y - ym) * k + ym * g
    # Colour strength is left alone: a different costume has its own saturation, and matching it would repaint the character.
    f = (y2[..., None] + (f - y[..., None]))
    f = np.clip(f, 0, 255)
    # softness
    cur = subject_stats(np.clip(np.rint(f), 0, 255).astype(np.uint8), mask)
    # (only when the render is moderately sharper than the original: a textured costume against a plain one would be smeared)
    if lap_tol * target["lap"] < cur["lap"] <= max_ratio * target["lap"]:
        lo, hi = 0.0, max_soften
        for _ in range(5):
            mid = (lo + hi) / 2
            test = np.clip(np.rint(smooth(f, mid) if mid > 0 else f), 0, 255).astype(np.uint8)
            if subject_stats(test, mask)["lap"] > lap_tol * target["lap"]:
                lo = mid
            else:
                hi = mid
        f = smooth(f, hi)
    # grain: what the footage has and the subject lacks, from the footage's own noise
    have = subject_stats(np.clip(np.rint(f), 0, 255).astype(np.uint8), mask)["sigma"]
    if have < target["sigma"]:
        amount = float(np.sqrt(max(target["sigma"] ** 2 - have ** 2, 0.0)) / max(subject_stats(plate, np.ones(plate.shape[:2], bool))["sigma"], 0.2))
        f = f + real_grain(plate, f.shape[:2], seed) * min(0.7 * amount, 3.0)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


# --- C1: the plate's light round the subject's edge ---------------------------------------------------------------------------


def light_wrap(fg: np.ndarray, alpha: np.ndarray, plate: np.ndarray, sigma: float = 12.0, width: int = 8, amount: float = 0.25
               ) -> np.ndarray:
    """C1: the plate's light spilling round the subject's edge: the blurred plate, screened in over a band just inside the edge."""
    core = alpha >= 0.5
    inner = core & ~shrink(core, width)
    w = blur(inner.astype(np.float32), width / 3.0)[..., None] * amount
    wrap = smooth(plate.astype(np.float32), sigma) / 255.0
    f = fg.astype(np.float32) / 255.0
    return np.clip((1 - (1 - f) * (1 - wrap * w)) * 255.0, 0, 255)



def clone_fill(frame: np.ndarray, hole: np.ndarray, margin: int = 8) -> np.ndarray:
    """Fill `hole` with the same frame's texture from beside it (a clone stamp from the left, or the right at the
    frame's edge), its tone blended into the surroundings (harmonize). No generated content: a meadow is filled with
    meadow, never with something a frame editor imagined there."""
    if not hole.any():
        return frame
    h, w = hole.shape
    xs = np.flatnonzero(hole.any(axis=0))
    dx = int(xs[-1] - xs[0] + 1 + margin)
    out = frame.copy()
    for d in (dx, -dx, 2 * dx, -2 * dx):
        src = shift(frame, d, 0)
        ok = shift(np.ones((h, w), bool), d, 0, fill=False) & ~shift(hole, d, 0, fill=True)
        take = hole & ok
        if take.any():
            g = harmonize(src, frame, hole.astype(np.float32), iters=300)
            out[take] = np.clip(np.rint(g[take]), 0, 255).astype(np.uint8)
            hole = hole & ~take
        if not hole.any():
            break
    return out


# --- Relight: the light of the shot on a painted first frame ------------------------------------------------------
# The frame editor paints the new person with its own studio-ish light. A second edit pass (Qwen-Image-Edit 2511,
# told to match the shot's light) relights it, but also re-draws fine detail. So only the relit frame's LOW-FREQUENCY
# shading is kept, blended into the paint, inside a tight subject mask; the paint keeps every fine detail. Then the
# subject's brightness is set to the original person's ratio to the plate, measured on this shot.

RELIGHT_SIGMA = 8.0      # px at the crop's size: shading, not detail
RELIGHT_STRENGTH = 0.7   # share of the relit frame's shading blended into the paint
RELIGHT_SHRINK = 2       # px: the tight mask. A grown mask lifted the sky at the edge into a halo.
RELIGHT_FEATHER = 1.5
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 luminance (0..255 floats) of an RGB image."""
    return img.astype(np.float32) @ _LUMA


def subject_ratio(src: np.ndarray, plate: np.ndarray, mask: np.ndarray) -> float | None:
    """The original person's mean luminance over the plate's at the same pixels, for this shot (0.846 on the knight:
    a person in the shade is darker than the ground behind them). None when the mask is empty."""
    if not mask.any():
        return None
    base = float(luma(plate)[mask].mean())
    return float(luma(src)[mask].mean()) / base if base > 1e-3 else None


def relight_weight(mask: np.ndarray) -> np.ndarray:
    """Where the relight may act: the subject mask shrunk, feathered inward, and never beyond the mask itself."""
    w = np.clip(blur(shrink(mask, RELIGHT_SHRINK).astype(np.float32), RELIGHT_FEATHER), 0, 1)
    return (w * mask).astype(np.float32)


def _shaded(paint: np.ndarray, relit: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """The paint's luminance times the relit frame's low-frequency gain (no mask yet)."""
    lp, lr = luma(paint), luma(relit)
    return lp * ((blur(lr, sigma) + 4) / (blur(lp, sigma) + 4)) ** strength


def relight_shading(paint: np.ndarray, relit: np.ndarray, mask: np.ndarray, sigma: float = RELIGHT_SIGMA,
                    strength: float = RELIGHT_STRENGTH) -> np.ndarray:
    """Luminance (H, W) of the paint with the relit frame's low-frequency gain applied inside the tight mask. Colour and
    fine detail stay the paint's; outside the mask the paint's own luminance is returned."""
    w = relight_weight(mask)
    return luma(paint) * (1 - w) + _shaded(paint, relit, sigma, strength) * w


def brightness_gamma(lum: np.ndarray, mask: np.ndarray, target: float) -> float:
    """The gamma on luminance (0..255) that brings its mean inside the mask to `target`; a gamma, not a gain, so the
    highlights stay compressed. Clamped to 0.4..2.0 (1.0 for an empty mask)."""
    if not mask.any():
        return 1.0
    x = np.clip(lum[mask], 0, 255) / 255
    lo, hi = 0.4, 2.0
    for _ in range(30):
        g = (lo + hi) / 2
        if (255 * x ** g).mean() > target:
            lo = g
        else:
            hi = g
    return (lo + hi) / 2


NEUTRAL_SAT = 0.25   # below this saturation a pixel is a neutral material (steel, grey, white): it takes the brightness gamma
EMBLEM_SHARE = 0.08  # a bright chromatic hue group smaller than this share of the subject is an emblem or trim: kept as painted
EMBLEM_VALUE = 0.45


def material_masks(paint: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the subject into (neutral, cloth, emblem) by colour: neutral = low saturation (steel, grey, white); the
    chromatic pixels grouped by hue (12 bins); a group under EMBLEM_SHARE of the subject and bright is an emblem (a gold
    lion), the rest are cloth (a red tabard). R&D's knight relight2 handled steel, tabard and lion this way."""
    f = paint.astype(np.float32) / 255
    mx, mn = f.max(-1), f.min(-1)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    neutral = mask & (sat < NEUTRAL_SAT)
    chrom = mask & ~neutral
    bins = (hue * 12).astype(np.int32) % 12
    area = max(int(mask.sum()), 1)
    cloth = np.zeros(mask.shape, bool)
    emblem = np.zeros(mask.shape, bool)
    for k in range(12):
        m = chrom & (bins == k)
        if not m.any():
            continue
        if m.sum() < EMBLEM_SHARE * area and float(mx[m].mean()) > EMBLEM_VALUE:
            emblem |= m
        else:
            cloth |= m
    return neutral, cloth, emblem


def relight_paint(paint: np.ndarray, relit: np.ndarray, mask: np.ndarray, src: np.ndarray, plate: np.ndarray,
                  src_mask: np.ndarray, strength: float = RELIGHT_STRENGTH, materials: bool = True) -> tuple[np.ndarray, dict]:
    """The painted first frame with the shot's light: the relit frame's low-frequency shading at `strength`, inside
    the tight subject `mask`, then the subject's brightness set to the original person's ratio to the plate
    (`src_mask`: where they were; `plate` and `src` are the same crop). Returns (uint8 image, what was measured)."""
    w = relight_weight(mask)
    lp = luma(paint)
    shaded = _shaded(paint, relit, RELIGHT_SIGMA, strength)
    ratio = subject_ratio(src, plate, src_mask)
    inner = w > 0.5
    info: dict = {"scene_ratio": None if ratio is None else round(ratio, 3)}
    neutral, cloth, emblem = material_masks(paint, inner) if materials else (inner, np.zeros_like(inner), np.zeros_like(inner))
    if not neutral.any():  # nothing neutral to set the brightness on: the whole subject takes the gamma
        neutral, cloth, emblem = inner, np.zeros_like(inner), np.zeros_like(inner)
    if ratio is not None and inner.any():
        target = ratio * float(luma(plate)[inner].mean())
        g = brightness_gamma(shaded, neutral, target)
        ls = 255 * (np.clip(shaded, 0, 255) / 255) ** g  # neutral materials: the shot's brightness, highlights compressed
        info |= {"target": round(target, 1), "gamma": round(g, 3)}
    else:
        ls = shaded
    if materials and (cloth.any() or emblem.any()):
        lt = shaded * (float(lp[cloth].mean()) / max(float(shaded[cloth].mean()), 1e-3)) if cloth.any() else shaded  # cloth: shading, own mean
        wl = np.clip(blur((grow(emblem, 3) & inner).astype(np.float32), 2.0), 0, 1)  # emblems keep the paint
        wt = np.clip(blur((grow(cloth, 6) & inner & ~neutral).astype(np.float32), 3.0) - wl, 0, 1)
        shaded = lt * wt + lp * wl + ls * np.clip(1 - wt - wl, 0, 1)
        info["materials"] = {"neutral": int(neutral.sum()), "cloth": int(cloth.sum()), "emblem": int(emblem.sum())}
    else:
        shaded = ls
    lo = lp * (1 - w) + shaded * w
    out = np.clip(paint.astype(np.float32) * np.clip(lo / np.maximum(lp, 1.0), 0.25, 4.0)[..., None], 0, 255)
    if inner.any():
        info["mean"] = {"paint": round(float(lp[inner].mean()), 1), "out": round(float(luma(out)[inner].mean()), 1)}
    return np.rint(out).astype(np.uint8), info
