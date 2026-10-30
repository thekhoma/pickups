"""Camera alignment by direct image registration (numpy and PIL only).

A moving camera shows the ground and the far scene again and again at a different place and scale. If every frame is
warped onto one reference by a homography, what an object hid in one frame is visible, as real footage, in another, and
Remove can borrow it instead of generating it. The homography is fitted where it matters: the ring of picture around the
object (its ground), not the whole frame, so a near plane and a far hill don't pull against each other.

`register` is Lucas-Kanade on a pyramid (Gauss-Newton on the eight homography parameters, masked and weighted).
`clip` chains consecutive frames to a middle reference and scores each frame's fit.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pickups import composite


def gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 luma, 0..1."""
    return (composite.luma709(img.astype(np.float32)) / 255.0).astype(np.float32)


def down(a: np.ndarray, k: int) -> np.ndarray:
    """Average-pool by an integer factor (crops the remainder)."""
    if k <= 1:
        return a
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _scale(k: float) -> np.ndarray:
    return np.diag([k, k, 1.0])


def _norm(w: int, h: int) -> np.ndarray:
    """Pixel -> normalised (centred, about [-1, 1]) coordinates."""
    s = max(w, h) / 2.0
    return np.array([[1 / s, 0, -w / (2 * s)], [0, 1 / s, -h / (2 * s)], [0, 0, 1.0]])


def _gauss_newton(ref: np.ndarray, mov: np.ndarray, wt: np.ndarray, H: np.ndarray, iters: int) -> tuple[np.ndarray, float]:
    """Refine H (mov pixel -> ref pixel) on one pyramid level. Returns (H, weighted mean |ref - warped mov|)."""
    h, w = ref.shape
    T = _norm(w, h)
    Ti = np.linalg.inv(T)
    s = max(w, h) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - w / 2) / s, (ys - h / 2) / s
    ones = np.ones_like(ref)
    err = 1.0
    H = T @ H @ Ti  # to normalised coordinates
    for _ in range(iters):
        Hp = Ti @ H @ T
        iw = composite.warp(mov, Hp, (w, h))
        valid = composite.warp(ones, Hp, (w, h)) > 0.999
        W = wt * valid
        tot = float(W.sum())
        if tot < 50:
            return Ti @ H @ T, 1.0
        r = ref - iw
        err = float((np.abs(r) * W).sum() / tot)
        gx, gy = (g / 8.0 * s for g in composite.sobel(iw))  # per normalised unit
        t = gx * xn + gy * yn
        cols = [gx * xn, gx * yn, gx, gy * xn, gy * yn, gy, -t * xn, -t * yn]
        A = np.stack([c[W > 0] for c in cols], axis=1).astype(np.float64)
        wv = W[W > 0].astype(np.float64)
        rv = r[W > 0].astype(np.float64)
        AtW = A.T * wv
        try:
            p = -np.linalg.solve(AtW @ A + 1e-6 * np.eye(8), AtW @ rv)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 0.5:
            break
        D = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [p[6], p[7], 1.0]])
        H = D @ H
        H = H / H[2, 2]
        if np.abs(p).max() < 1e-4:
            break
    H = Ti @ H @ T
    return H / H[2, 2], err


def register(ref: np.ndarray, mov: np.ndarray, weight: np.ndarray, H0: np.ndarray | None = None, levels: int = 3,
             iters: int = 12) -> tuple[np.ndarray, float]:
    """The homography H (mov pixel -> ref pixel) that best lays `mov` over `ref` where `weight` > 0 (float32 HxW).
    Pyramid: the coarse levels find big moves, the fine level the last pixel. Returns (H, residual): the weighted mean
    absolute luma difference after the fit (0..1)."""
    H = np.eye(3) if H0 is None else np.asarray(H0, np.float64)
    err = 1.0
    for lvl in range(levels - 1, -1, -1):
        k = 2 ** lvl
        S = _scale(1.0 / k)
        Hl = S @ H @ np.linalg.inv(S)
        Hl, err = _gauss_newton(down(ref, k), down(mov, k), down(weight, k), Hl, iters)
        H = np.linalg.inv(S) @ Hl @ S
        H = H / H[2, 2]
    return H, err


def clip(frames: list[np.ndarray], covers: list[np.ndarray], work_long: int = 480, ref: int | None = None,
         ring_px: int = 0, ring_from: list[np.ndarray] | None = None) -> tuple[list[np.ndarray], list[float], int]:
    """Each frame's homography onto the middle frame (full-size pixels), from the frames in order.

    `covers[i]`: where frame i is hidden by the object (or its shadow): left out of the fit, and the fit is weighted
    towards the ring of picture around it (`ring_px`, full-size px; 0 = flat weights). Returns (H per frame mapping frame
    i's pixel to the reference's, residual per frame, reference index). `ring_from`: the masks the ring is drawn around
    (default: `covers`), so a second fit can follow one part of the hidden area, e.g. the far hill behind the object."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    k = max(1, int(np.ceil(max(h, w) / work_long)))
    gs = [down(gray(f), k) for f in frames]
    cs = [composite.grow(down(c.astype(np.float32), k) > 0.05, 2) for c in covers]
    sh, sw = gs[0].shape
    S = _scale(1.0 / k)
    near = np.zeros((sh, sw), np.float32)
    if ring_px > 0:
        union = np.logical_or.reduce(cs if ring_from is None else [composite.grow(down(r.astype(np.float32), k) > 0.05, 1) for r in ring_from])
        r = max(2, ring_px // k)
        near = (composite.grow(union, r) & ~np.logical_or.reduce(cs)).astype(np.float32)

    def weight(a: int, b: int) -> np.ndarray:
        free = (~cs[a] & ~cs[b]).astype(np.float32)
        return free * (0.25 + 0.75 * near if ring_px > 0 else 1.0)

    m = n // 2 if ref is None else ref
    # consecutive pairs: Hstep[i] maps frame i+1's pixels to frame i's
    steps, errs = [], []
    for i in range(n - 1):
        Hs, e = register(gs[i], gs[i + 1], weight(i, i + 1))
        steps.append(Hs)
        errs.append(e)
    Hs_small: list[np.ndarray] = [np.eye(3)] * n
    for i in range(m + 1, n):  # i -> i-1 -> ... -> m
        Hs_small[i] = Hs_small[i - 1] @ steps[i - 1]
    for i in range(m - 1, -1, -1):  # i -> i+1 (inverse of the step) -> ... -> m
        Hs_small[i] = Hs_small[i + 1] @ np.linalg.inv(steps[i])
    # one fine pass of each frame against the reference removes the chain's drift
    resid = []
    out = []
    for i in range(n):
        if i == m:
            Hf, e = np.eye(3), 0.0
        else:
            Hf, e = register(gs[m], gs[i], weight(m, i), H0=Hs_small[i], levels=1, iters=6)
            Hf = Hf if e <= errs[min(i, n - 2)] * 4 + 0.02 else Hs_small[i]
        resid.append(e)
        out.append(np.linalg.inv(S) @ Hf @ S)  # back to full-size pixels
    return out, resid, m


def to_ref(img: np.ndarray, H: np.ndarray, size: tuple[int, int], cubic: bool = False) -> np.ndarray:
    """A frame (or mask, as uint8/float) laid onto the reference by H; zero where the frame has no picture."""
    if img.dtype == np.bool_:
        return composite.warp(img.astype(np.float32), H, size) > 0.5
    if img.ndim == 3 and img.dtype == np.uint8:
        r = Image.fromarray(img)
        Hn = H / H[2, 2]
        Mi = np.linalg.inv(Hn)
        Mi = Mi / Mi[2, 2]
        affine = abs(Mi[2, 0]) < 1e-12 and abs(Mi[2, 1]) < 1e-12
        data = tuple(Mi[:2].ravel()) if affine else tuple(Mi.ravel()[:8])
        return np.asarray(r.transform(size, Image.AFFINE if affine else Image.PERSPECTIVE, data, resample=Image.BICUBIC if cubic else Image.BILINEAR))
    return composite.warp(img, H, size, cubic)


def valid(H: np.ndarray, size: tuple[int, int], src_size: tuple[int, int], margin: int = 2) -> np.ndarray:
    """Where the frame (of src_size = (w, h)) has picture once laid onto the reference by H (a small margin in)."""
    ones = np.ones((src_size[1], src_size[0]), np.float32)
    v = composite.warp(ones, H, size) > 0.999
    return composite.shrink(v, margin) if margin else v
