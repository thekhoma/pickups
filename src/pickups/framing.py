"""Does a pinned start frame match the previz framing? A plain image comparison: no model, no network.

The playblast's frame 0 comes with its framing from the browser (the main subject's box and the horizon height, as
fractions of the frame). The start frame is compared with it on edges, which a grey layout and a photoreal frame share:
the subject is found by matching the frame-0 subject's edge map across positions and sizes, and the horizon by sliding
the two frames' horizontal-edge profiles against each other. The subject's size and place are only reported (and warned
about) when the match is sure: one size clearly wins, and the patch it found leads back to the subject in frame 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Worth a warning above these (fractions of the frame).
MAX_X = 0.08
MAX_H = 0.10
MAX_HORIZON = 0.10
# Below this match score the subject wasn't found at all.
MIN_SCORE = 0.2
W, H = 320, 182
# Sizes tried for the subject (start frame against frame 0), even steps in log scale. A match counts only when it peaks
# inside this range (not on its last PEAK_EDGE steps) and stands PEAK_ABOVE over the median score of all the sizes.
SCALES = np.exp(np.linspace(np.log(0.5), np.log(2.0), 27))
PEAK_EDGE = 2
PEAK_ABOVE = 0.08
# ...and stands SHARP_ABOVE over the sizes SHARP_STEPS either side (3 steps: about 17% bigger or smaller).
SHARP_STEPS = 3
SHARP_ABOVE = 0.06
# The cycle check: back within this size ratio and this fraction of the frame of where the subject is in frame 0.
BACK_SIZE = 1.15
BACK_PLACE = 0.05


def _comes_back(g0: Image.Image, e1: np.ndarray, shape: tuple[int, int], s: float, x: int, y: int,
                home: tuple[float, float]) -> bool:
    """Cycle check: the patch the match found in the start frame, searched for in frame 0 the same way, must land back
    on the subject at the inverse size. A grey mannequin matched onto the wrong part of a photo doesn't."""
    th, tw = int(round(shape[0] * s)), int(round(shape[1] * s))
    patch = e1[max(0, y): max(0, y + th), max(0, x): max(0, x + tw)]
    home = (home[0] + max(0, x) - x, home[1] + max(0, y) - y)  # the part of the patch inside the frame
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return False
    tries = []
    for r in SCALES:
        score, bx, by = _ncc(_edges(g0, int(round(W / r)), int(round(H / r)))[0], patch)
        if score > -1:
            tries.append((score, float(r), bx * r, by * r))
    if not tries:
        return False
    _, r, bx, by = max(tries)
    return abs(np.log(r * s)) < np.log(BACK_SIZE) and abs(bx - home[0]) < BACK_PLACE * W and abs(by - home[1]) < BACK_PLACE * H


def _decisive(tries: list[tuple[float, float, int, int]]) -> bool:
    """Did one size clearly win? tries: (score, scale, x, y) for each size that fit, smallest scale first."""
    if len(tries) < 2 * PEAK_EDGE + 3:
        return False
    scores = [t[0] for t in tries]
    i = int(np.argmax(scores))
    if not (max(PEAK_EDGE, SHARP_STEPS) <= i < len(scores) - max(PEAK_EDGE, SHARP_STEPS)):
        return False
    # Sharp on both sides: a part of the subject (the mannequin's head on a man's cap) keeps scoring as the size drops,
    # a real match falls away either side of its size.
    sides = max(scores[i - SHARP_STEPS], scores[i + SHARP_STEPS])
    return scores[i] - float(np.median(scores)) >= PEAK_ABOVE and scores[i] - sides >= SHARP_ABOVE


def _gray(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("L")


def _edges(path: Path | Image.Image, w: int = W, h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """Edge magnitude and horizontal edges, worked out at w x h (never an edge map that was itself resized)."""
    im = _gray(path) if isinstance(path, Path) else path
    g = np.asarray(im.resize((max(8, w), max(8, h)), Image.BILINEAR), dtype=np.float32) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy), np.abs(gy)


def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((max(1, w), max(1, h)), Image.BILINEAR), dtype=np.float32)


def _ncc(image: np.ndarray, tpl: np.ndarray) -> tuple[float, int, int]:
    """Best normalized cross-correlation of tpl inside image: (score, x, y) of the top-left corner."""
    ih, iw = image.shape
    th, tw = tpl.shape
    if th > ih or tw > iw or th < 4 or tw < 4:
        return -1.0, 0, 0
    t = tpl - tpl.mean()
    tn = float(np.sqrt((t * t).sum()))
    if tn < 1e-6:
        return -1.0, 0, 0
    fh, fw = ih + th, iw + tw
    num = np.fft.irfft2(np.fft.rfft2(image, (fh, fw)) * np.conj(np.fft.rfft2(t, (fh, fw))), (fh, fw))[: ih - th + 1, : iw - tw + 1]
    ii = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(image * image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(s):
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]

    n = th * tw
    s1, s2 = box(ii), box(ii2)
    var = s2 - s1 * s1 / n
    # A window much flatter than the template (a bare wall, the sky) can't be the subject; without this its tiny
    # variance blows the score up.
    flat = var < (0.25 * tn) ** 2
    score = num / (np.sqrt(np.maximum(var, 1e-9)) * tn)
    score[flat] = -1.0
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[y, x]), int(x), int(y)


def _horizon_shift(a: np.ndarray, b: np.ndarray) -> int:
    """Rows to move a's horizontal-edge profile down to best match b's."""
    pa = a.mean(1)
    pb = b.mean(1)
    pa = (pa - pa.mean()) / (pa.std() + 1e-6)
    pb = (pb - pb.mean()) / (pb.std() + 1e-6)
    best, shift = -1e9, 0
    lim = H // 3
    for s in range(-lim, lim + 1):
        if s >= 0:
            c = float((pa[: H - s] * pb[s:]).mean())
        else:
            c = float((pa[-s:] * pb[: H + s]).mean())
        if c > best:
            best, shift = c, s
    return shift


def compare(frame0: Path, framing: dict, start: Path) -> dict:
    """Measure where the start frame puts the subject and the horizon, against the playblast's frame 0."""
    e0, gy0 = _edges(frame0)
    g1 = _gray(start)
    e1, gy1 = _edges(g1)
    out: dict = {"warning": None}
    box = framing.get("subject") if isinstance(framing, dict) else None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x0, y0, x1, y1 = (float(v) for v in box)
        cx0, h0 = (x0 + x1) / 2, y1 - y0
        pad = 0.15
        bx0, bx1 = max(0.0, x0 - (x1 - x0) * pad), min(1.0, x1 + (x1 - x0) * pad)
        by0, by1 = max(0.0, y0 - h0 * pad), min(1.0, y1 + h0 * pad)
        tpl = e0[int(by0 * H): max(int(by0 * H) + 1, int(round(by1 * H))), int(bx0 * W): max(int(bx0 * W) + 1, int(round(bx1 * W)))]
        # Like for like: the template keeps its size and the start frame is scaled by 1/s, its edges found again at that
        # size. Resizing an edge map (the template, as this did before) blurs or thins its edges, and the blur scored
        # better, so every warning said "smaller" (Fresh-9 user test: 6 of 6, even for a bigger subject).
        tries = []
        for s in SCALES:
            # Padded by half the template, so a subject that runs off the edge (a bigger one) still has a place to match.
            py, px = tpl.shape[0] // 2, tpl.shape[1] // 2
            img = np.pad(_edges(g1, int(round(W / s)), int(round(H / s)))[0], ((py, py), (px, px)))
            score, x, y = _ncc(img, tpl)
            if score > -1:
                tries.append((score, float(s), int(round((x - px) * s)), int(round((y - py) * s))))
        score, s, x, y = max(tries) if tries else (-1.0, 1.0, 0, 0)
        out["subject_sure"] = bool(_decisive(tries)) and bool(_comes_back(_gray(frame0), e1, tpl.shape, s, x, y, (bx0 * W, by0 * H)))
        # The match found the padded box; map back to the subject box inside it.
        sx0 = (x + (x0 - bx0) * W * s) / W
        sy0 = (y + (y0 - by0) * H * s) / H
        cx1 = sx0 + (x1 - x0) * s / 2
        h1 = h0 * s
        out.update(subject_score=round(score, 3), subject_x=[round(cx0, 3), round(cx1, 3)], subject_h=[round(h0, 3), round(h1, 3)],
                   subject_y=[round(y0, 3), round(sy0, 3)])
    hz = framing.get("horizon") if isinstance(framing, dict) else None
    if isinstance(hz, (int, float)):
        shift = _horizon_shift(gy0, gy1) / H
        out["horizon"] = [round(float(hz), 3), round(float(hz) + shift, 3)]
    out["warning"] = warning(out)
    return out


def warning(out: dict) -> str | None:
    """The plain warning for a compare() result, or None when nothing measured is both sure and off."""
    parts = []
    # A grey mannequin and a photo share few edges: unless the match stands out, its size and place say nothing, and a
    # wrong warning is worse than none (it sent the Fresh-9 test after 'smaller' start frames that were fine).
    if "subject_score" in out and out.get("subject_sure"):
        if out["subject_score"] < MIN_SCORE:
            parts.append("the subject isn't where the previz has it")
        else:
            dx = out["subject_x"][1] - out["subject_x"][0]
            dh = out["subject_h"][1] - out["subject_h"][0]
            if abs(dx) > MAX_X:
                parts.append(f"subject is further {'right' if dx > 0 else 'left'}")
            if abs(dh) > MAX_H:
                parts.append("bigger" if dh > 0 else "smaller")
    # The horizon is a whole-frame edge-profile slide: it locks onto the strongest photo line (a guardrail, a parapet)
    # as readily as the horizon, so it only counts when the subject match is sure and both horizons are in the frame
    # (Run 5: c07 [0.37, 0.672] and s06 [-0.407, -0.176], both unsure, both fine by eye).
    hz = out.get("horizon")
    if hz and out.get("subject_sure") and all(0 <= v <= 1 for v in hz) and abs(hz[1] - hz[0]) > MAX_HORIZON:
        parts.append(f"horizon is {'lower' if hz[1] > hz[0] else 'higher'}")
    if not parts:
        return None
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"This start frame doesn't match the previz framing ({text}). The render may drift at the start."
