"""Reference prep for Recast: one front-view figure and its mask, from a photo, a still or a character sheet.

SCAIL-2 copies the new character from a reference picture plus a mask of who to copy. Our stills and most character
sheets are shot on a plain light-grey backdrop, so the figure is simply what differs from the backdrop: no segmenter
label, which a chrome robot, an inflatable T-rex or a lizard-person may not answer to. A sheet is split into its
panels and figures and the front view is kept, so CLIP-vision sees one figure, big. A busy photo returns None and the
caller finds the person with SAM 3.1 instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pickups import composite
from pickups.composite import _box, blur, smooth

WORK = 512        # analysis size (long side)
OUT_LONG = 1536   # the prepared reference is at most this big; SCAIL-2 resizes it to the performance size anyway
IN_LONG = 2 * OUT_LONG  # pictures are read at most this big: a 48 MP portrait would add seconds and GBs for nothing
STANDING = 1.1    # a figure on a plain backdrop is at least this much taller than wide (a car, a landscape isn't)


@dataclass
class Reference:
    image: Image.Image   # RGB: one figure, padded to the performance's aspect
    mask: np.ndarray     # bool, the figure, same size as image
    path: str            # how it was made, for the take's Details
    views: int = 1       # figures found on the picture (3 on a front/back/close-up sheet)
    box: tuple[int, int, int, int] = (0, 0, 0, 0)  # the crop on the picture as read (x0, y0, x1, y1; may reach past it)


# --- small image tools (numpy only; blur and smooth live in composite) --------------------------------------------


def vclose(mask: np.ndarray, r: int) -> np.ndarray:
    """Close gaps up to 2r tall, vertically only: bridges a low-contrast stripe across a standing figure (a grey belt)
    without closing the gaps between legs or between an arm and the body."""
    if r < 1:
        return mask.copy()
    d = _box(mask.astype(np.float32), r, 0) > 1e-4
    return mask | (_box(d.astype(np.float32), r, 0) > 1 - 1e-4)


def fill_from(img: np.ndarray, known: np.ndarray, sigma: float) -> np.ndarray:
    """Smoothly fill an image from its `known` pixels (normalised blur): a backdrop estimate behind a figure."""
    w = known.astype(np.float32)
    num = blur(img.astype(np.float32) * w[..., None], sigma)
    den = blur(w, sigma)[..., None]
    out = num / np.maximum(den, 1e-6)
    if (den < 1e-6).any():  # far from anything known: the mean of what is known
        out = np.where(den < 1e-6, img[known].mean(axis=0) if known.any() else img.reshape(-1, img.shape[-1]).mean(axis=0), out)
    return out


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components (8-neighbour) of a boolean mask: labels 1..n (0 = background) and n.

    Works on row runs (a union-find over runs that overlap the row above), so it is fast at full size too."""
    h, w = mask.shape
    d = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, x0 = np.nonzero(d == 1)
    x1 = np.nonzero(d == -1)[1]
    n = len(rows)
    if n == 0:
        return np.zeros((h, w), np.int32), 0
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = np.searchsorted(rows, np.arange(h + 1)).tolist()
    xs0, xs1 = x0.tolist(), x1.tolist()
    for r in range(1, h):
        i, i1, j, j1 = first[r - 1], first[r], first[r], first[r + 1]
        while i < i1 and j < j1:
            if xs0[i] <= xs1[j] and xs0[j] <= xs1[i]:  # touching, diagonals included
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            if xs1[i] < xs1[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    lengths = x1 - x0
    start = rows * w + x0
    pos = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths) + np.repeat(start, lengths)
    out = np.zeros(h * w, np.int32)
    out[pos] = np.repeat(ids.astype(np.int32) + 1, lengths)
    return out.reshape(h, w), int(ids.max()) + 1


def touching(region: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """The parts of `region` connected to any `seed` pixel (whole objects that touch the seed)."""
    hit = region & seed
    if not hit.any():
        return np.zeros_like(region)
    lab, _ = label(region)
    keep = np.zeros(lab.max() + 1, bool)
    keep[np.unique(lab[hit])] = True
    keep[0] = False
    return keep[lab]


def largest(mask: np.ndarray, keep_frac: float = 0.0) -> np.ndarray:
    """The biggest connected piece, plus any piece at least `keep_frac` of its size (0 = the biggest only)."""
    lab, n = label(mask)
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= max(1, keep_frac * sizes.max()) if keep_frac > 0 else sizes == sizes.max()
    return keep[lab]


def fill_holes(mask: np.ndarray, max_frac: float = 1.0, solid: np.ndarray | None = None) -> np.ndarray:
    """Fill enclosed holes no bigger than `max_frac` of the mask's area (a gap under a hand on a hip stays open), and
    any enclosed hole that is mostly `solid` (part of the figure that happens to look like the backdrop)."""
    lab, n = label(~mask)
    if n == 0:
        return mask.copy()
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    fill = sizes <= max_frac * max(1, int(mask.sum()))
    if solid is not None:  # judged away from the hole's rim, where the figure's own edge blurs in
        inner = (~composite.grow(mask, 2)).ravel()
        deep = np.bincount(lab.ravel(), weights=inner.astype(np.float64), minlength=n + 1)
        hit = np.bincount(lab.ravel(), weights=(solid.ravel() & inner).astype(np.float64), minlength=n + 1)
        fill |= (deep > 0) & (hit > 0.5 * deep)
    fill[edge] = False
    fill[0] = False
    return mask | fill[lab]


def drop_specks(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = label(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def _luma(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


# --- backdrop and figures ------------------------------------------------------------------------------------------


@dataclass
class _Plain:
    fg: np.ndarray       # figure pixels at analysis size
    field: np.ndarray    # backdrop estimate at analysis size (float RGB)
    thr: float           # colour distance that counts as "not backdrop"


def foreground(img: np.ndarray) -> _Plain | None:
    """What isn't backdrop on a plain-backdrop picture (analysis size), or None when the backdrop isn't plain.

    The backdrop is estimated from the border, then refined from everything that isn't figure (studio backdrops have
    a vignette and a lighter floor). A pixel is figure when its colour is clearly off the backdrop, or when it has
    texture the smooth backdrop never has (a silver suit on a grey backdrop is close in colour, but full of folds).
    Soft shadows on the floor (darker, same colour, smooth) are backdrop."""
    h, w, _ = img.shape
    f = img.astype(np.float32)
    b = max(3, round(0.03 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    y = _luma(f)
    hp = y - smooth(y, 1.0)
    tex = np.sqrt(smooth(hp * hp, 1.0))
    tex_thr = max(3.0, 3.0 * float(np.median(tex[border])))

    floor = np.zeros((h, w), bool)  # contact shadows and a brighter floor: same colour, darker or lighter, smooth
    floor[int(0.7 * h):] = True
    known = border
    fg = np.zeros((h, w), bool)
    field = f
    thr = 24.0
    dist = np.zeros((h, w), np.float32)
    for sigma in (0.3, 0.12, 0.06):
        field = fill_from(f, known, sigma * max(h, w))
        dist = np.sqrt(((f - field) ** 2).sum(-1))
        res = dist[border & ~fg] if (border & ~fg).sum() > 50 else dist[border]
        thr = max(24.0, 5.0 * 1.4826 * float(np.median(np.abs(res - np.median(res)))))
        ratio = (f + 1.0) / (field + 1.0)
        shade = ratio.mean(-1)
        shading = floor & (shade > 0.4) & (shade < 1.6) & (ratio.max(-1) - ratio.min(-1) < 0.08) & (tex < tex_thr)
        raw = ((dist > thr) & ~shading) | ((tex > tex_thr) & (dist > 0.5 * thr))  # texture alone spreads past the edge
        fg = vclose(composite.shrink(composite.grow(raw, 2), 2), max(2, round(0.01 * h)))  # close small gaps (a grey belt)
        fg = drop_specks(fg, max(12, int(0.0006 * h * w)))
        known = ~composite.grow(fg, max(3, round(0.015 * max(h, w))))
        if known.sum() < 0.15 * h * w:
            return None
    # Plain means: most of the border is backdrop, the backdrop is smooth and neutral (a studio sweep, not a sky).
    bg = ~composite.grow(fg, 2)
    sat = (field.max(-1) - field.min(-1)) / np.maximum(field.max(-1), 1.0)
    if ((dist[border] < thr).mean() < 0.6 or fg[border].mean() > 0.25 or float(np.percentile(tex[bg], 90)) > 2.5 * tex_thr
            or float(np.median(sat[bg])) > 0.3 or fg.mean() > 0.6 or not fg.any()):
        return None
    fg = fill_holes(fg, 0.0, solid=dist > 0.5 * thr)  # small holes are filled per figure (_figures)
    return _Plain(fg, field, thr)


def seams(img: np.ndarray, axis: int = 1) -> list[tuple[int, int]]:
    """Panel seams of a contact sheet, as (start, end) of each divider: straight lines where the picture jumps on
    nearly every row (axis=1: vertical seams). A figure's own straight edge never spans the whole height."""
    y = _luma(img) if img.ndim == 3 else img.astype(np.float32)
    if axis == 0:
        y = y.T
    d = np.abs(np.diff(y, axis=1))
    med = np.median(d, axis=0)
    full = (d > 2.5).mean(axis=0)
    hits = np.flatnonzero((med > max(4.0, 6.0 * float(np.median(med)))) & (full > 0.95)) + 1
    n = y.shape[1]
    out: list[list[int]] = []
    for x in hits.tolist():
        if out and x - out[-1][1] <= max(3, n // 100):
            out[-1][1] = x
        else:
            out.append([x, x])
    out = [s for s in out if 0.1 * n < s[0] and s[1] < 0.9 * n]
    keep: list[tuple[int, int]] = []
    for a, b in out:
        if not keep or a - keep[-1][1] > 0.1 * n:
            keep.append((a, b))
    return keep


def spans(mask: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Separate figures side by side: column runs that hold figure pixels, split where a gap is at least `min_gap`."""
    h, w = mask.shape
    cols = mask.sum(axis=0) >= max(1, round(0.005 * h))
    runs: list[list[int]] = []
    for x in np.flatnonzero(cols):
        if runs and x - runs[-1][1] <= min_gap:
            runs[-1][1] = int(x)
        else:
            runs.append([int(x), int(x)])
    if not runs:
        return []
    mass = [int(mask[:, a:b + 1].sum()) for a, b in runs]
    return [(a, b + 1) for (a, b), m in zip(runs, mass) if m >= 0.05 * max(mass)]


def symmetry(img: np.ndarray, mask: np.ndarray) -> float:
    """How mirror-symmetric the top of a figure (its head) is, 0..1: a front or back view scores high, a profile low."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return 0.0
    y0 = ys.min()
    hh = max(4, int(0.14 * (ys.max() - y0 + 1)))
    band = mask[y0:y0 + hh]
    bx = np.nonzero(band.any(axis=0))[0]
    if len(bx) < 4:
        return 0.0
    c = int(round(np.nonzero(band)[1].mean()))
    half = min(c - bx.min(), bx.max() - c)
    if half < 2:
        return 0.0
    lum = _luma(img[y0:y0 + hh])
    left, right = lum[:, c - half:c][:, ::-1], lum[:, c + 1:c + 1 + half]
    ml, mr = band[:, c - half:c][:, ::-1], band[:, c + 1:c + 1 + half]
    both = ml & mr
    shape = both.sum() / max(1, (ml | mr).sum())
    if not both.any():
        return 0.0
    spread = float(np.abs(left[both] - left[both].mean()).mean() + np.abs(right[both] - right[both].mean()).mean()) + 1.0
    tone = 1.0 - min(1.0, float(np.abs(left[both] - right[both]).mean()) / spread)
    return 0.5 * float(shape) + 0.5 * tone


@dataclass
class _Figure:
    panel: tuple[int, int, int, int]   # x0, y0, x1, y1 in the full picture
    mask: np.ndarray                   # at the panel's analysis size
    plain: _Plain | None
    x: float                           # left edge in the full picture (for leftmost-first)
    whole: bool                        # head to feet inside its panel (not a close-up or cut off)
    height: float                      # share of its panel's height
    sym: float


def _figures(rgb: np.ndarray, alpha: np.ndarray | None) -> tuple[list[_Figure], int]:
    """Every figure on the picture (each panel of a sheet on its own), and how many panels there are."""
    H, W, _ = rgb.shape
    if alpha is not None:
        panels = [(0, 0, W, H)]
    else:
        def cuts(found: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
            edges = [0] + [v for a, b in found for v in (a, b)] + [n]
            return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
        xs, ys = cuts(seams(rgb, 1), W), cuts(seams(rgb, 0), H)
        t = max(2, round(0.003 * max(W, H)))  # past a thin frame line or divider edge
        panels = [(x0 + t, y0 + t, x1 - t, y1 - t) for y0, y1 in ys for x0, x1 in xs]
    out: list[_Figure] = []
    for x0, y0, x1, y1 in panels:
        s = min(1.0, WORK / max(x1 - x0, y1 - y0))
        size = (max(8, round((x1 - x0) * s)), max(8, round((y1 - y0) * s)))
        small = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize(size, Image.BILINEAR))
        if alpha is not None:
            plain = None
            fg = np.asarray(Image.fromarray(alpha.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
            fg = drop_specks(fg, max(12, int(0.0006 * fg.size)))
        else:
            plain = foreground(small)
            if plain is None:
                continue
            fg = plain.fg
        h, w = fg.shape
        for a, b in spans(fg, max(3, round(0.02 * w))):
            m = np.zeros_like(fg)
            m[:, a:b] = fg[:, a:b]
            # The figure plus pieces close to it: a light shoe on a light floor can come apart from the leg.
            m = fill_holes(touching(m, composite.grow(largest(m, keep_frac=0.15), max(3, round(0.025 * h)))), 0.03)
            ys, xs_ = np.nonzero(m)
            if len(ys) < 0.002 * h * w:
                continue
            bw = xs_.max() - xs_.min() + 1
            cut = (m[-2:].any(axis=0).sum() > 0.35 * bw or m[:2].any() or m[:, :2].sum() > 0.1 * h or m[:, -2:].sum() > 0.1 * h)
            out.append(_Figure((x0, y0, x1, y1), m, plain, x0 + a / s, not cut, (ys.max() - ys.min() + 1) / h,
                               symmetry(small, m)))
    return out, len(panels)


def _pick(figs: list[_Figure]) -> _Figure:
    """The front view: whole figures only (not a face close-up), leftmost by sheet convention, unless the leftmost is
    clearly a profile and another whole figure is clearly more symmetric (a front view; a back view scores about as
    high as a front view, so the margin keeps a front view on the left)."""
    whole = [f for f in figs if f.whole]
    if whole:
        tall = max(f.height for f in whole)
        whole = [f for f in whole if f.height >= 0.6 * tall]
    pool = sorted(whole or figs, key=lambda f: f.x)
    if not whole:
        return max(pool, key=lambda f: f.mask.sum())
    best = max(f.sym for f in pool)
    return next(f for f in pool if f.sym >= best - 0.2)


# --- cutting out --------------------------------------------------------------------------------------------------


def load(src) -> tuple[np.ndarray, np.ndarray | None]:
    """RGB pixels (EXIF orientation applied, at most IN_LONG on the long side) and the alpha cut-out when the picture
    really has a transparent background."""
    im = src if isinstance(src, Image.Image) else Image.fromarray(src) if isinstance(src, np.ndarray) else Image.open(Path(src))
    im = ImageOps.exif_transpose(im)
    transparent = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    im = im.convert("RGBA" if transparent else "RGB")
    if max(im.size) > IN_LONG:
        im.thumbnail((IN_LONG, IN_LONG), Image.LANCZOS)  # alpha with it
    alpha = None
    if transparent:
        a = np.asarray(im)[..., 3]
        if (a < 128).mean() > 0.05:
            alpha = a >= 128
    if alpha is not None:  # transparent pixels shown on our usual light grey
        rgba = np.asarray(im).astype(np.float32)
        k = rgba[..., 3:] / 255.0
        rgb = np.clip(rgba[..., :3] * k + 200.0 * (1 - k), 0, 255).astype(np.uint8)
    else:
        rgb = np.asarray(im.convert("RGB"))
    return rgb, alpha


def _refine(rgb: np.ndarray, coarse: np.ndarray, field: np.ndarray | None, thr: float, scale: float) -> np.ndarray:
    """Sharpen an upscaled analysis mask at full size, so no ring of light backdrop rides along (SCAIL-2 would copy it
    as a glow). Near the edge, figure is what is off the backdrop's colour or textured, at full-size precision."""
    if field is None:
        return coarse
    px = max(2, round(4 * scale))  # the analysis mask's own blur and closing, at full size
    band = composite.grow(coarse, px) & ~composite.shrink(coarse, px)
    f = rgb.astype(np.float32)
    dist = np.sqrt(((f - field) ** 2).sum(-1))
    y = _luma(f)
    hp = y - smooth(y, 0.8)
    tex = np.sqrt(smooth(hp * hp, 0.8))
    outside = ~composite.grow(coarse, px)
    t_thr = max(3.0, 3.0 * float(np.median(tex[outside]))) if outside.any() else 3.0
    ratio = (f + 1.0) / (field + 1.0)
    shade = ratio.mean(-1)
    shading = (shade > 0.45) & (shade < 1.15) & (ratio.max(-1) - ratio.min(-1) < 0.08)  # the floor in shadow
    shading[: int(0.7 * shading.shape[0])] = False
    out = (coarse & ~band) | (band & (((dist > 0.5 * thr) & ~shading) | ((tex > t_thr) & (dist > 0.3 * thr))))
    return fill_holes(largest(out, keep_frac=0.15) | touching(out, coarse & ~band), 0.01)  # nothing the rough mask held whole


def _cut(rgb: np.ndarray, mask: np.ndarray, aspect: float, backdrop: np.ndarray | None, path: str, views: int,
         origin: tuple[int, int] = (0, 0), others: np.ndarray | None = None) -> Reference:
    """Crop around the figure with a small margin and pad to `aspect` (width / height), so SCAIL-2's centre crop to the
    performance's shape keeps the whole figure. `others` (other figures beside it) are painted over with the backdrop;
    nothing else is, so a light boot the mask missed is still in the picture."""
    H, W, _ = rgb.shape
    ys, xs = np.nonzero(mask)
    fx0, fx1, fy0, fy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = max(4, round(0.06 * (fy1 - fy0)))
    bx0, bx1, by0, by1 = fx0 - m, fx1 + m, fy0 - m, fy1 + m
    bw, bh = bx1 - bx0, by1 - by0
    if bw / bh < aspect:
        grow = round(bh * aspect) - bw
        bx0 -= grow // 2
        bx1 = bx0 + round(bh * aspect)
    else:
        grow = round(bw / aspect) - bh
        by0 -= grow // 2
        by1 = by0 + round(bw / aspect)
    if backdrop is not None:
        src = rgb.copy()
        if others is not None and others.any():
            near = composite.grow(mask, max(2, round(0.01 * (fy1 - fy0))))
            paint = composite.grow(others & ~near, max(2, round(0.015 * (fy1 - fy0)))) & ~near  # their soft edges too
            src[paint] = backdrop[paint].astype(np.uint8)  # only the chosen figure on its backdrop
        pad = np.median(backdrop[~mask], axis=0) if (~mask).any() else np.array([200, 200, 200])
    else:
        src = rgb
        edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        pad = np.median(edge, axis=0)
    out = np.empty((by1 - by0, bx1 - bx0, 3), np.uint8)
    out[:] = np.asarray(pad, np.float32).round().astype(np.uint8)
    om = np.zeros(out.shape[:2], bool)
    sx0, sy0, sx1, sy1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
    out[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = src[sy0:sy1, sx0:sx1]
    om[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = mask[sy0:sy1, sx0:sx1]
    im = Image.fromarray(out)
    if max(im.size) > OUT_LONG:
        s = OUT_LONG / max(im.size)
        size = (max(1, round(im.width * s)), max(1, round(im.height * s)))
        im = im.resize(size, Image.LANCZOS)
        om = np.asarray(Image.fromarray(om.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
    return Reference(im, om, path, views, (bx0 + origin[0], by0 + origin[1], bx1 + origin[0], by1 + origin[1]))


def prepare(src, aspect: float, mask: np.ndarray | None = None) -> Reference | None:
    """One figure and its mask, padded to `aspect` (the performance's width / height).

    `mask` (a person SAM 3.1 found on a busy photo) skips the backdrop analysis. Without it: a transparent PNG uses its
    alpha; a plain-backdrop still or sheet is cut out against the backdrop (a sheet keeps its front view); anything
    else, a figure that isn't standing (wider than tall: a car on a sheet) included, returns None, so the caller can
    fall back to SAM."""
    rgb, alpha = load(src)
    H, W, _ = rgb.shape
    if min(H, W) < 16:
        return None  # a thumbnail this small holds no one to copy
    if mask is not None:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
        m = largest(m, keep_frac=0.15)
        if not m.any():
            return None
        return _cut(rgb, m, aspect, None, "person picked out of your photo", 1)
    figs, panels = _figures(rgb, alpha)
    if not figs:
        return None
    f = _pick(figs)
    ys, xs = np.nonzero(f.mask)
    if ys.max() - ys.min() + 1 < STANDING * (xs.max() - xs.min() + 1):
        return None  # not a standing figure (a car on a sheet, a landscape): SAM decides whether anyone is there
    x0, y0, x1, y1 = f.panel
    pw, ph = x1 - x0, y1 - y0
    coarse = np.asarray(Image.fromarray(f.mask.astype(np.uint8) * 255).resize((pw, ph), Image.BILINEAR)) > 127
    rough = coarse  # the figure before its edge is sharpened: a light boot the sharp mask drops is still theirs
    panel = rgb[y0:y1, x0:x1]
    field = None
    if f.plain is not None:
        field = np.stack([np.asarray(Image.fromarray(f.plain.field[..., c]).resize((pw, ph), Image.BILINEAR)) for c in range(3)], -1)
        sharp = _refine(panel, coarse, field, f.plain.thr, max(pw, ph) / max(f.mask.shape))
        coarse = sharp if sharp.sum() > 0.5 * coarse.sum() else coarse
        everyone = np.asarray(Image.fromarray(f.plain.fg.astype(np.uint8) * 255).resize((pw, ph), Image.BILINEAR)) > 127
    else:
        everyone = alpha[y0:y1, x0:x1] if alpha is not None else np.zeros((ph, pw), bool)
    if field is None:
        field = np.broadcast_to(np.array([200.0, 200.0, 200.0], np.float32), panel.shape).copy()
    views = max(len(figs), panels)
    if alpha is not None:
        path = "figure cut out from its transparent background" if views == 1 else f"front view cut out from a {views}-view sheet"
    else:
        path = "figure cut out from its plain backdrop" if views == 1 else f"front view cropped from a {views}-view sheet"
    # Other figures beside this one (a sheet without panel seams): big pieces that aren't it, even when a thin strand
    # joins them to it outside its own columns.
    r = max(2, round(0.01 * ph))
    xs = np.flatnonzero((rough | coarse).any(axis=0))
    cols = np.zeros_like(everyone)
    cols[:, max(0, xs.min() - r):xs.max() + 1 + r] = True
    lab, n = label(everyone & ~touching(everyone & cols, composite.grow(rough | coarse, r)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    big = sizes >= 0.05 * max(1, int(coarse.sum()))
    big[0] = False
    return _cut(panel, coarse, aspect, field, path, views, (x0, y0), others=big[lab])
