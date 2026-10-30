"""Relight a kept subject by frequency separation.

The kept thing is real footage; the render of the same shot (made with the thing in it, as a light reference only)
shows how the new light falls on it. The footage keeps all its fine detail, and its broad light and colour come from
the render:  out = real_high + reference_low, in log linear light. That is the same as laying the (small) ratio of the
two low-passes over the footage, once the footage has the scene's overall gain and veil.

The reference is a different drawing of the thing, a few px off and not always the same clothes, so:
  * its outline (SAM on the render) is lined up on the real one first (align),
  * the ratio is trusted only near the thing's usual shift and within a stop or so (clamp),
  * and it is carried frame to frame in the thing's own coordinates and averaged over a few frames, so it is steady.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pickups import composite

LUM = np.array([0.299, 0.587, 0.114], np.float32)
SIGMA_MIN = 5.0     # px: never finer than this, so a small thing's gain is a smooth light and none of its detail
SIGMA = 0.02        # of the subject's height: the finest light taken from the reference (about 1-3%)
ALIGN = 0.12        # of its height: how far the reference's outline may sit from the real one and still be lined up
SCALE = (0.6, 1.7)  # the reference's height over the real one, else it is a different thing
LUM_SPREAD = 0.55   # log: how far across the subject its light may differ from its median (a lit side, a shaded one)
TINT_SPREAD = 0.14  # log: ...and its colour
LUM_MED = (-1.0, 0.7)  # log: its median change over the footage that has already had the scene's gain and veil
TINT_MED = 0.3
GRID_H = 48         # cells down the subject (the fields are smooth at SIGMA, a cell is about 2% of its height)
REACH = 4           # frames either side the grids are averaged over
EPS = 1e-4


def to_linear(x: np.ndarray) -> np.ndarray:
    return (np.clip(x.astype(np.float32), 0, 255) / 255.0) ** 2.2


def lowpass(x: np.ndarray, sigma: float, weight: np.ndarray) -> np.ndarray:
    """A blur of only the weighted pixels (normalised), carried 4x wider over gaps with none: the thing's own light."""
    wt = weight.astype(np.float32)

    def spread(s: float) -> tuple[np.ndarray, np.ndarray]:
        wb = composite.blur(wt, s)
        return composite.blur(x * wt[..., None], s) / np.maximum(wb, 1e-6)[..., None], wb

    near, w_near = spread(sigma)
    far, _ = spread(4 * sigma)
    mix = np.clip(w_near / 0.2, 0.0, 1.0)[..., None]
    return near * mix + far * (1 - mix)


def box_of(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not ys.size:
        return None
    h, w = mask.shape
    return max(0, ys[0] - pad), min(h, ys[-1] + 1 + pad), max(0, xs[0] - pad), min(w, xs[-1] + 1 + pad)


def align(ref_mask: np.ndarray, real_mask: np.ndarray) -> np.ndarray | None:
    """The 3x3 matrix that moves the reference's outline onto the real one (scale by the heights, then centroid, then a
    cross-correlation of the two outlines within ALIGN of the height), or None when they are not the same sort of thing."""
    br, bf = box_of(real_mask), box_of(ref_mask)
    if br is None or bf is None:
        return None
    hr, hf = br[1] - br[0], bf[1] - bf[0]
    s = hr / hf
    if not SCALE[0] <= s <= SCALE[1]:
        return None
    cy_r, cx_r = (np.argwhere(real_mask).mean(axis=0))
    cy_f, cx_f = (np.argwhere(ref_mask).mean(axis=0))
    M = np.array([[s, 0, cx_r - s * cx_f], [0, s, cy_r - s * cy_f], [0, 0, 1]], np.float64)
    h, w = real_mask.shape
    moved = composite.warp(ref_mask.astype(np.float32), M, (w, h)) > 0.5
    y0, y1, x0, x1 = box_of(real_mask | moved, pad=int(ALIGN * hr) + 2)
    a = composite.blur(real_mask[y0:y1, x0:x1].astype(np.float32), max(1.0, 0.01 * hr))
    b = composite.blur(moved[y0:y1, x0:x1].astype(np.float32), max(1.0, 0.01 * hr))
    c = np.abs(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    reach = int(ALIGN * hr)
    # the circular correlation, read only for shifts within `reach`
    best, at = -1.0, (0, 0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            v = c[dy % c.shape[0], dx % c.shape[1]]
            if v > best:
                best, at = v, (dx, dy)
    T = np.array([[1, 0, at[0]], [0, 1, at[1]], [0, 0, 1]], np.float64)
    return T @ M


def padded_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The thing's box with room for its blur: the area a grid is made over and laid back on."""
    box = box_of(mask)
    if box is None:
        return None
    return box_of(mask, int(3 * max(SIGMA_MIN, SIGMA * (box[1] - box[0]))))


def frame_grid(base: np.ndarray, ref: np.ndarray, mask: np.ndarray, ref_mask: np.ndarray, shape: tuple[int, int],
               ) -> tuple[np.ndarray, np.ndarray] | None:
    """One frame's light on the subject from the reference, as a grid in its own coordinates.

    `base`: the footage with the scene's gain and veil already on it (uint8 or float, 0..255); `ref`: the render with
    the thing in it, same size; `mask`, `ref_mask`: the thing in each. Returns (log ratio, weight) where the ratio is
    low(reference) / low(base) per colour on a `shape` (rows, cols) grid over padded_box(mask), or None."""
    box = box_of(mask)
    M = align(ref_mask, mask)
    if box is None or M is None:
        return None
    h, w = mask.shape
    tall = box[1] - box[0]
    sigma = max(SIGMA_MIN, SIGMA * tall)
    y0, y1, x0, x1 = padded_box(mask)
    sl = (slice(y0, y1), slice(x0, x1))
    b_lin = to_linear(base[sl])
    r_lin = composite.warp(to_linear(ref), M, (w, h))[sl]
    r_msk = (composite.warp(ref_mask.astype(np.float32), M, (w, h)) > 0.5)[sl]
    core_b = composite.shrink(mask[sl], max(1, int(0.4 * sigma)))
    core_r = composite.shrink(r_msk, max(1, int(0.4 * sigma)))
    if core_b.sum() < 64 or core_r.sum() < 64:
        return None
    lb = lowpass(b_lin, sigma, core_b)
    lr = lowpass(r_lin, sigma, core_r)
    ratio = np.log(np.maximum(lr, EPS)) - np.log(np.maximum(lb, EPS))
    # where the two drawings overlap the ratio is the light; where only one has the thing it is a guess
    weight = (core_b & core_r).astype(np.float32) * 1.0 + 0.15 * core_b
    weight = composite.blur(weight, sigma)
    # to the grid, premultiplied so cells with nothing in them take their neighbours' (weight 0)
    gh, gw = shape
    num = np.stack([np.asarray(Image.fromarray((ratio[..., c] * weight).astype(np.float32), "F").resize((gw, gh), Image.BOX))
                    for c in range(3)], axis=-1)
    den = np.asarray(Image.fromarray(weight.astype(np.float32), "F").resize((gw, gh), Image.BOX))
    return num, den


def grid_shape(masks: list[np.ndarray]) -> tuple[int, int]:
    """Rows and columns of the grid: the subject's usual box (median over the clip) at about GRID_H cells tall."""
    boxes = [box_of(m) for m in masks if m.any()]
    if not boxes:
        return GRID_H, GRID_H
    hs = np.median([b[1] - b[0] for b in boxes])
    ws = np.median([b[3] - b[2] for b in boxes])
    gh = int(np.clip(round(hs / max(SIGMA_MIN, SIGMA * hs)), 6, GRID_H))
    gw = int(np.clip(round(gh * ws / hs), 8, 96))
    return gh, gw


def steady_grids(grids: list, reach: int = REACH) -> list[np.ndarray | None]:
    """Each frame's grid averaged over `reach` frames either side (weighted by how much of the thing both drawings
    showed), robustly clamped: its median change within the usual range, its spread across the subject within a lit
    side's and a shaded side's. Returns the log ratio grid per frame, or None where nothing was learnt."""
    n = len(grids)
    have = [i for i, g in enumerate(grids) if g is not None and g[1].sum() > 0]
    out: list = []
    for i in range(n):
        near = [grids[j] for j in have if abs(j - i) <= reach]
        if not near:
            if not have:
                out.append(None)
                continue
            near = [grids[min(have, key=lambda j: abs(j - i))]]
        num = sum(g[0] for g in near)
        den = sum(g[1] for g in near)
        r = num / np.maximum(den, 1e-6)[..., None]
        ok = den > 0.05 * den.max()
        if not ok.any():
            out.append(None)
            continue
        lum = r.mean(-1)
        chroma = r - lum[..., None]

        def wmed(x: np.ndarray) -> np.ndarray:
            return np.median(x[ok], axis=0)

        ml, mc = float(wmed(lum)), wmed(chroma)
        ml_c = float(np.clip(ml, *LUM_MED))
        mc_c = np.clip(mc, -TINT_MED, TINT_MED)
        lum2 = ml_c + np.clip(lum - ml, -LUM_SPREAD, LUM_SPREAD)
        chroma2 = mc_c + np.clip(chroma - mc, -TINT_SPREAD, TINT_SPREAD)
        # cells that learnt nothing take the median
        lum2 = np.where(ok, lum2, ml_c)
        chroma2 = np.where(ok[..., None], chroma2, mc_c)
        out.append((lum2[..., None] + chroma2).astype(np.float32))
    return out


def field(grid: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    """The grid's log ratio as an (h, w, 3) field over the thing's padded box (zero elsewhere); None without one."""
    box = padded_box(mask)
    if grid is None or box is None:
        return None
    y0, y1, x0, x1 = box
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[y0:y1, x0:x1, c] = np.asarray(Image.fromarray(grid[..., c].astype(np.float32), "F").resize((x1 - x0, y1 - y0), Image.BILINEAR))
    return out
