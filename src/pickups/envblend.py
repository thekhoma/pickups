"""Environment blend: make a new subject sit in the light of the plate, after the matte, in linear light.

The render's subject comes out of LTX with the render's grade: its own white balance and black level, a little less
contrast than the plate, and the light of the painted frame, not of the shot's sun. Three steps, each measured:

1. grade: the render's own surroundings (a ring just outside the subject, inside the region LTX redrew) against the same
   ring on the real plate give the colour cast (per-channel gain) and the black level (an offset) the render drifted by.
   Applied to the subject. Smoothed over time.
2. shade: the original person stood in the real sun. Their lateral light (the column profile of their luminance across
   their width, blurred, normalised by its mean; a column mixes less clothing albedo than a row does) is put on the new
   subject at a part of its strength, over the new subject's width.
3. contrast: the render's local contrast against the plate's over the same ring (the render is softer: LTX's decode), put
   back on the subject about its own mean.

The light wrap from composite.light_wrap is applied after this, in person_swap.blend_frame.
"""

from __future__ import annotations

import numpy as np

from pickups import composite

RING_IN, RING_OUT = 0.004, 0.03  # of the subject's height: the ring just outside the matte the grade is measured on
HELD_IN, HELD_OUT = 0.03, 0.06  # and a held-out ring beyond it, to check the grade on pixels it was not fitted to
GAIN_LIMITS = (0.9, 1.1)  # the render's own drift from the plate was about 1.0 on the knight (T1.5): kept small
LIFT_LIMIT = 0.02  # linear: the most the subject's black level is lifted towards the plate's (airlight)
LIFT_SHARE = 0.7  # of the gap
CONTRAST_LIMITS = (0.9, 1.25)
SHADE_STRENGTH = 0.5
SHADE_LIMITS = (0.8, 1.2)
BINS = 48
MIN_RING_PX = 400


def to_lin(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0, 1)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def _luma(lin: np.ndarray) -> np.ndarray:
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def height(mask: np.ndarray) -> int:
    ys = np.flatnonzero(mask.any(axis=1))
    return int(ys[-1] - ys[0] + 1) if len(ys) else 0


def ring(mask: np.ndarray, region: np.ndarray, tall: int, near: float = RING_IN, far: float = RING_OUT) -> np.ndarray:
    """The ground just outside the subject, inside the region LTX redrew (a ring outside it is the real plate in both)."""
    inner = composite.grow(mask, max(2, round(near * tall)))
    return composite.grow(mask, max(4, round(far * tall))) & ~inner & region


def ring_stats(img: np.ndarray, rg: np.ndarray) -> dict | None:
    """Mean colour, black level and contrast (linear) of `img` on the ring `rg`; None when the ring is too small."""
    if int(rg.sum()) < MIN_RING_PX:
        return None
    lin = to_lin(img)
    px = lin[rg]
    y = _luma(lin)
    low = composite.smooth(y, 3.0)
    return {"mean": px.mean(axis=0), "black": float(np.percentile(y[rg], 2)), "contrast": float(low[rg].std() + 1e-6)}


def frame_params(render: np.ndarray, plate: np.ndarray, mask: np.ndarray, region: np.ndarray) -> dict | None:
    """The grade and contrast one frame needs: the plate's ring against the render's ring. None when not measurable."""
    tall = height(mask)
    if not tall:
        return None
    rg = ring(mask, region, tall)
    a, b = ring_stats(render, rg), ring_stats(plate, rg)
    if a is None or b is None:
        return None
    gain = np.clip(b["mean"] / np.maximum(a["mean"], 1e-4), *GAIN_LIMITS)
    # The subject's blacks against the plate's: studio-dark armour on a hazy meadow. Lifted towards the plate's own dark
    # level (an airlight veil, in the plate's tint), never lowered.
    inner = mask
    sub_black = float(np.percentile(_luma(to_lin(render))[inner], 2)) if int(inner.sum()) > MIN_RING_PX else b["black"]
    lift = float(np.clip(LIFT_SHARE * (b["black"] - sub_black), 0, LIFT_LIMIT))
    tint = b["mean"] / max(float(b["mean"].mean()), 1e-6)
    contrast = float(np.clip(b["contrast"] / a["contrast"], *CONTRAST_LIMITS))
    return {"gain": gain.astype(np.float32), "lift": lift, "tint": tint.astype(np.float32), "contrast": contrast}


def smooth_params(params: list[dict | None], radius: int = 4) -> list[dict | None]:
    """Each frame's parameters averaged over its neighbours (frames that could not be measured borrow theirs)."""
    ok = [i for i, p in enumerate(params) if p is not None]
    if not ok:
        return [None] * len(params)
    out: list[dict | None] = []
    for i in range(len(params)):
        near = [params[j] for j in range(max(0, i - radius), min(len(params), i + radius + 1)) if params[j] is not None]
        if not near:
            j = min(ok, key=lambda k: abs(k - i))
            near = [params[j]]
        out.append({"gain": np.mean([p["gain"] for p in near], axis=0).astype(np.float32),
                    "lift": float(np.mean([p["lift"] for p in near])),
                    "tint": np.mean([p["tint"] for p in near], axis=0).astype(np.float32),
                    "contrast": float(np.mean([p["contrast"] for p in near]))})
    return out


def shade_profile(old: np.ndarray, mask: np.ndarray, bins: int = BINS) -> np.ndarray | None:
    """The original person's lateral light: their luminance (linear) by column across their width, in `bins` steps,
    divided by its mean and blurred. None when there is too little of them."""
    if int(mask.sum()) < 2000:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y = _luma(to_lin(old))
    cols = np.array_split(np.arange(x0, x1), bins)
    prof = np.array([float(y[:, c][mask[:, c]].mean()) if mask[:, c].any() else np.nan for c in cols])
    if np.isnan(prof).all():
        return None
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    prof = prof / max(float(prof.mean()), 1e-6)
    k = np.ones(5) / 5
    return np.convolve(np.pad(prof, 2, mode="edge"), k, mode="valid").astype(np.float32)


def smooth_profiles(profiles: list[np.ndarray | None], radius: int = 4) -> list[np.ndarray | None]:
    out = []
    for i in range(len(profiles)):
        near = [profiles[j] for j in range(max(0, i - radius), min(len(profiles), i + radius + 1)) if profiles[j] is not None]
        out.append(np.mean(near, axis=0).astype(np.float32) if near else None)
    return out


def shade_map(profile: np.ndarray | None, mask: np.ndarray, strength: float = SHADE_STRENGTH) -> np.ndarray:
    """A multiplicative shading over the frame: the original's lateral profile over the new subject's width, `strength`
    of the way from flat, kept within SHADE_LIMITS."""
    out = np.ones(mask.shape, np.float32)
    if profile is None or not mask.any():
        return out
    xs = np.flatnonzero(mask.any(axis=0))
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    u = np.linspace(0, len(profile) - 1, x1 - x0)
    col = np.interp(u, np.arange(len(profile)), profile)
    col = np.clip(1.0 + strength * (col - 1.0), *SHADE_LIMITS)
    out[:, x0:x1] = col[None, :]
    return out


def apply(char: np.ndarray, matte: np.ndarray, params: dict | None, profile: np.ndarray | None,
          steps: tuple[str, ...] = ("grade", "shade", "contrast"), strength: float = SHADE_STRENGTH) -> np.ndarray:
    """`char` (the render at the crop's size) with the grade, shading and contrast steps applied, in linear light."""
    if params is None and profile is None:
        return char
    lin = to_lin(char)
    inner = matte > 0.5
    if "contrast" in steps and params is not None and inner.any():
        y = _luma(lin)
        mean = float(y[inner].mean())
        lin = np.maximum(mean + (lin - mean) * params["contrast"], 0)
    if "grade" in steps and params is not None:
        lin = np.maximum(lin * params["gain"][None, None, :] + params["lift"] * params["tint"][None, None, :], 0)
    if "shade" in steps and profile is not None:
        lin = lin * shade_map(profile, inner, strength)[..., None]
    return to_srgb(lin)


def black_gap(img: np.ndarray, plate: np.ndarray, mask: np.ndarray, region: np.ndarray) -> float | None:
    """How far the subject's black level (2nd percentile of its luminance, linear) is from the plate's on the ring round it,
    as a share of the plate's: the thing the grade step closes."""
    tall = height(mask)
    if not tall or int(mask.sum()) < MIN_RING_PX:
        return None
    rg = ring(mask, region, tall)
    if int(rg.sum()) < MIN_RING_PX:
        return None
    sub = float(np.percentile(_luma(to_lin(img))[mask], 2))
    pl = float(np.percentile(_luma(to_lin(plate))[rg], 2))
    return abs(pl - sub) / max(pl, 1e-6)


def lateral_agreement(new: np.ndarray, new_mask: np.ndarray, old: np.ndarray, old_mask: np.ndarray, bins: int = BINS) -> float | None:
    """Pearson correlation of the lateral luminance profiles (by column across the width) of the new subject and the original
    person: what the shade step moves towards 1."""
    def prof(img, m):
        if int(m.sum()) < 2000:
            return None
        xs = np.flatnonzero(m.any(axis=0))
        y = _luma(to_lin(img))
        cols = np.array_split(np.arange(xs[0], xs[-1] + 1), bins)
        return np.array([float(y[:, c][m[:, c]].mean()) if m[:, c].any() else np.nan for c in cols])
    a, b = prof(new, new_mask), prof(old, old_mask)
    if a is None or b is None or np.isnan(a).any() or np.isnan(b).any() or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])
