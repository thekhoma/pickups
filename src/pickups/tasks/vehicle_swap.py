"""Recast a vehicle: a new car drives the old one's path on the clean road, keeping its own shape in every frame.

Recast on LTX steers the render with the shot's depth. For a person that is the point (the new character takes the old
one's pose); for a vehicle it is the fault: the old car's depth pins the new body to the old car's shape, so a three-
window wagon slowly morphs into a two-window SUV, and pasting where the render changed the footage left a seam. Here:

1. Clear the road. Remove's own background fill (tasks/remove.background) takes the old vehicle, its shadow and (on a
   wet road) its reflection out of the window, as a stage of this job (no extra take). What fills it is said in the
   scene's own words (the road), and the plate is checked for what the old car left behind (a dark patch where it
   stood); a failed plate is made again with the next seed. A car the camera keeps in place gets the generated fill;
   one that comes, goes or crosses the frame lets Remove use real pixels where it can.
2. Paint the new vehicle onto the anchor frame (the middle of the shot for a car coming towards the camera), at the
   source's own shape, from the angle the old one is seen (view_words), a small one on an enlarged window. The painting
   must stand on the road at the old one's size and place (placement); else it is made again with the next seed.
3. Cut it out with SAM 3.1, and follow the old vehicle by aligning its SAM outline frame to frame (an affine fit: point
   tracks slide on reflective paint and glass), the corners' paths smoothed, then made rigid (no shear or squash),
   pinned where its tyres meet the road. A parked one stays put while a door swings.
4. Give its wheels a road-speed spin (rotational blur in the wheel's own plane), so the render doesn't copy frozen hubs.
5. A stand-in: the new vehicle, moved by the old one's motion, over the clean plate.
6. Render it with LTX-2.5's Layout-to-Render IC-LoRA on a crop around the vehicle's path and the old one's place, at
   1:1 pixels (multiples of 128), the painted frame as the look and as a keyframe at the anchor.
7. Follow the rendered vehicle with SAM 3.1 from the stand-in's outline, move the render back onto the footage where
   LTX drew its world a little off (smooth_offsets), and put it on the clean plate: the matte smoothed over +-2 frames
   along the vehicle's motion, choked 1 px, feathered ~1.1 px. It stands on the render's own road: under its body and
   wheels down past its tyres (ground_region), and wherever the old vehicle was, the render's pixels, toned to the
   footage by a membrane learnt around them, so its tyres touch the road, its contact shadow and reflection stay, and
   road lines pass behind its wheels as the render drew them. Grain matched. Everyone else stays footage, in front of
   it. Outside the vehicle and that ground the output is the clean plate, bit for bit.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite, graphs, media, reference
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.tasks import common, recast, remove

STAGES = [  # an RTX 4090, 97 frames of an HD shot (seconds)
    Stage("Preparing the shot", 15),
    Stage("Finding the vehicle", 35),
    Stage("Removing the old vehicle", 200),
    Stage("Getting the new vehicle", 10),
    Stage("Painting it into the shot", 40),
    Stage("Placing the new vehicle", 60),
    Stage("Rendering the shot", 110),
    Stage("Blending at full resolution", 90),
    Stage("Writing your files", 30),
]
FIND, CLEAR, GET, PAINT, PLACE, RENDER, BLEND, WRITE = (s.label for s in STAGES[1:])

SPRAY_LABELS = ("water spray", "splash")  # on a wet road: the spray the old vehicle throws up, removed with it (not "mist": the fog)
SHADOW_LABELS = ("shadow", "shadow of the car")  # what the old vehicle casts on the road (on a wet road its mirror image
#                                                 too: reflection_column; SAM's "reflection" took a stormy sea with it)
WET = re.compile(r"\b(wet|rain|rainy|raining|puddles?|flooded|slick|glistening)\b", re.IGNORECASE)
REFLECT_REACH = 1.5      # of the old vehicle's height: how far down a wet road its reflection (and lamp streaks) is cleared
HIGH = re.compile(r"\b(drone|aerial|from above|overhead|bird'?s[- ]eye|high angle|from high)\b", re.IGNORECASE)
HIGH_WORDS = (", seen from high above, looking steeply down onto its roof: the roof and bonnet fill most of it and only a "
              "thin strip of its side shows")
COMING = 1.15            # the old vehicle grows (shrinks) this much over the window: it comes (goes)
HEAD_ON_ASPECT, SIDE_ASPECT = 1.65, 2.7  # a vehicle's box, width over height: seen head-on below, side-on above
CONTACT_BAND = 0.05      # of the vehicle's height: the band about its contact line its painted contact shadows come from
PAINT_TRIES = 2        # paintings made at most (the next seed when the first one's tyres stand above the road)
ZOOM_FILL, ZOOM_MAX = 0.3, 4.0  # a small vehicle is painted enlarged until it fills this much of the height, at most this much
SIZE_RANGE = (1 / 1.6, 1.6)  # the painted vehicle's size against the old one's (square root of the areas)
PLACE_IOU = 0.4          # and how much at least it overlaps the old one
SAME_SIZE = (" It must be exactly as big in the picture as the {noun} it replaces, no bigger and no smaller, in the same place, "
             "with its tyres touching the road where the {noun}'s tyres touch it.")
FLOAT_MAX = 0.04         # of the old vehicle's height: how far above its contact line the new one's may stand
FAR_ROWS = 0.2            # of the vehicle's height from its top: the camera's move behind it is read above this
REAL_GROW = 36            # px at HD: how far past the old vehicle's outline real frames count as covered
MIN_SEEN = 4              # frames a pixel behind the old vehicle must show in to be taken from them
REAL_UPPER = 0.55       # of the old vehicle's height from its top: what is behind its body, refilled from real frames
INVENTED_MAX = 1.8       # the plate's edges there may be at most this dense against the footage's beside it
PLATE_TRIES = 2         # clean plates made at most (the next seed when the first leaves the old car behind)
REMNANT_LEVELS = 24.0    # the ground under where the old car was may be at most this much darker than beside it...
REMNANT_SHARE = 0.4      # ...or this share of how much darker the car itself made it
SPIN_SPAN, SPIN_STEP, SPIN_SAMPLES = 90.0, 200.0, 24  # degrees: a 180-degree shutter at ~200 deg/frame, as sampled
WHEEL_MIN = 0.3          # a wheel is at least this share of the biggest one (a steering wheel or a hidden far wheel isn't)
TOUCH_FRAMES = 3          # frames on which someone touching the car means they get in or out
DOOR_SWING = 1.12         # a parked car whose outline's area changes this much has a door or boot opening
DOORS_MESSAGE = ("This car has its doors opened, or people getting in or out. Replace a car works on cars that drive or "
                 "stay closed.")
PARKED_TOL = 0.015         # of its width: a vehicle whose lower body moves less than this is parked
SMOOTH_FRAMES = 3.0      # sigma (frames) of the corner paths
SIL_SIGMA = 3.0          # px (on the ~1024 px masks): the soft silhouettes that are aligned
MATTE_WEIGHTS = {-2: 1, -1: 2, 0: 3, 1: 2, 2: 1}
REGISTER_MAX = 0.04      # of the crop's long side: the most the render is moved back onto the footage
SHAPE_SHIFT = 3          # px: the shape check forgives a body sitting this far off (LTX's few-pixel drop after a keyframe)
FEATHER = 1.1            # px at HD
GROUND_REACH = 0.10      # of the vehicle's height: how far below the line its tyres stand on the render's road is used
GROUND_LIFT = 0.40       # ...and how far up from its lowest point (the sills, the bumpers' ends, the wheel arches)
GROUND_SOFT = 0.03       # of its height: the sigma of that road's soft edge into the plate
BEHIND = 0.3             # of the vehicle's height from its top: above it, frames whose render is off keep the plate
WORLD_OFF = 2.0          # a frame's render is off when its world differs from the footage this many times the median...
HOLD_MAX = 4               # at most this many such frames are stood in for by a neighbour
WORLD_MIN = 4.0          # ...and by at least this many levels
REFLECT_ALPHA, REFLECT_LEN = 0.35, 0.6  # a wet road's mirror image: its strength at the tyres, its length (of its height)
FILL_SOFT = 0.05         # of its height: the sigma of the edge where the old vehicle's place meets the footage
SIDE = 128               # Layout-to-Render's crop sides
SURFACE = re.compile(r"\b((?:[\w-]+[ ,]+){0,4}?(?:road|street|lane|highway|motorway|freeway|track|drive|driveway|avenue|"
                     r"boulevard|bridge|asphalt|tarmac|gravel|dirt|parking lot|car park|forecourt|runway|beach|sand|"
                     r"desert|snow|field|square|plaza|alley|tunnel|pavement)(?:\s+(?:road|track|lane|path|surface))?)\b",
                     re.IGNORECASE)


@dataclass
class Shot:
    """What Recast on LTX already worked out before it knew it was a vehicle."""
    p: dict
    run_dir: Path
    seed: int
    src: object  # media.Probe
    s0: int
    start: int | None
    pm: list[np.ndarray]  # the old vehicle in the window, at the proxy's size
    k: int  # the anchor frame
    full: Path  # the window at work size
    size: tuple[int, int]
    fps: float
    frames: int
    n_search: int
    noun: str
    qa: dict = field(default_factory=dict)
    om: list[np.ndarray] = field(default_factory=list)  # everyone else, tracked with it (proxy size): kept as footage


def stages_for(params: dict) -> list[Stage] | None:
    """The vehicle's stages when the selection's words name a vehicle (a clicked one switches when it is found)."""
    from pickups import features
    from pickups.tasks import recast_ltx
    if not features.cars():  # cut from this build: a person's steps, and the job refuses the car when it runs
        return None
    if str(params.get("kind", "")).lower() == "person":
        return None
    if not recast_ltx.vehicle_noun(params.get("text", "")):
        return None
    return STAGES if old_path(params) else car_stages()


def old_path(params: dict) -> bool:
    """The first car swap (Layout-to-Render on a stand-in) only when a take asks for it ("car_engine": "proxy")."""
    return str(params.get("car_engine") or "").lower() == "proxy"


def car_stages() -> list[Stage]:
    from pickups.tasks import car_swap
    return car_swap.STAGES


def use_stages(ctx) -> None:
    if hasattr(ctx, "use_stages"):
        ctx.use_stages(STAGES if old_path(getattr(ctx, "params", {}) or {}) else car_stages())


@contextmanager
def span(ctx, lo: float, hi: float):
    """Progress inside part of a stage: what reports 0..1 here moves the stage from lo to hi."""
    had = "progress" in vars(ctx)
    orig = ctx.progress
    ctx.progress = lambda f: orig(lo + (hi - lo) * max(0.0, min(1.0, float(f))))
    try:
        yield
    finally:
        if had:
            ctx.progress = orig
        else:
            del ctx.progress


# --- the old vehicle's motion: aligning its outline -------------------------------------------------------------------


def _align(t: np.ndarray, img: np.ndarray, A: np.ndarray, iters: int = 80, eps: float = 1e-4) -> np.ndarray:
    """Affine A (3x3, template px -> image px) with img(A x) ~ t(x): inverse-compositional Lucas-Kanade on two soft
    silhouettes. The template's gradients are fixed, so each step is one warp and a 6x6 solve."""
    h, w = t.shape
    gy, gx = np.gradient(t)
    sel = (np.abs(gx) + np.abs(gy)) > 1e-4
    if sel.sum() < 12:
        return A
    ys, xs = np.nonzero(sel)
    x, y = xs + 0.5, ys + 0.5
    gxs, gys = gx[sel].astype(np.float64), gy[sel].astype(np.float64)
    sd = np.stack([gxs * x, gxs * y, gxs, gys * x, gys * y, gys], axis=1)
    hess = sd.T @ sd
    tv = t[sel].astype(np.float64)
    A = A.astype(np.float64).copy()
    for _ in range(iters):
        iw = composite.warp(img, np.linalg.inv(A), (w, h))[sel]
        dp = np.linalg.solve(hess, sd.T @ (iw - tv))
        dW = np.array([[1 + dp[0], dp[1], dp[2]], [dp[3], 1 + dp[4], dp[5]], [0, 0, 1]])
        A = A @ np.linalg.inv(dW)
        if abs(dp[2]) + abs(dp[5]) < eps * 10 and np.abs(dp[[0, 1, 3, 4]]).max() < eps:
            break
    return A


def _soft(m: np.ndarray) -> np.ndarray:
    return composite.gauss(m.astype(np.float32), SIL_SIGMA)


def _rescale(A: np.ndarray, s: float) -> np.ndarray:
    S = np.diag([s, s, 1.0])
    return S @ A @ np.linalg.inv(S)


def align_outlines(masks: list[np.ndarray], ref: int) -> list[np.ndarray]:
    """Each frame's affine from frame `ref` (3x3, mask px): frame ref's soft outline aligned to each frame's, coarse
    to fine, each started from its neighbour's. A frame with no outline keeps its neighbour's."""
    n = len(masks)
    out: list[np.ndarray | None] = [None] * n
    out[ref] = np.eye(3)
    base = [_soft(m) for m in masks]
    half = [np.asarray(Image.fromarray(b, "F").resize((max(8, b.shape[1] // 2), max(8, b.shape[0] // 2)), Image.BILINEAR))
            for b in base]
    for order in (range(ref + 1, n), range(ref - 1, -1, -1)):
        prev = np.eye(3)
        for i in order:
            if masks[i].any():
                A = _align(half[ref], half[i], _rescale(prev, 0.5), iters=40)
                prev = _align(base[ref], base[i], _rescale(A, 2.0))
            out[i] = prev
    return out  # type: ignore[return-value]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def as_similarity(M: np.ndarray, about: tuple[float, float] | np.ndarray = (0.0, 0.0)) -> np.ndarray:
    """Move, turn and scale, without shear or squash, pinned at `about`: that point goes exactly where M takes it.

    The stand-in was a full affine fitted to the old SUV's outline, so a different body was stretched whenever that
    outline changed shape. The painted vehicle keeps its own proportions. Pinned at the picture's corner (as it first
    was), the dropped squash moved the car itself: 5% of scale lost 600 px from the corner is 30 px, and the new car
    sank and rose off the road through the shot. Pinned at the old car's contact line, its tyres follow the old ones.
    """
    A = np.asarray(M, float)
    linear = A[:2, :2]
    u, s, vt = np.linalg.svd(linear)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rot = u @ vt
    out = np.eye(3)
    out[:2, :2] = rot * float(np.mean(s))
    c = np.asarray(about, float)
    out[:2, 2] = linear @ c + A[:2, 2] - out[:2, :2] @ c
    return out


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine (3x3) taking points src (N x 2) to dst."""
    X = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return np.vstack([sol.T, [0, 0, 1]])


def smooth_path(Ms: list[np.ndarray], box: np.ndarray, ref: int, sigma: float = SMOOTH_FRAMES) -> list[np.ndarray]:
    """The motion with its jitter taken out: the box's corners, moved by each frame's fit, are smoothed over time
    (Gaussian, the path mirrored through its ends) and each frame's affine refit from them. Frame `ref` stays exactly where it is."""
    n = len(Ms)
    corners = np.stack([(M @ np.hstack([box, np.ones((4, 1))]).T).T[:, :2] for M in Ms])
    r = max(1, int(3 * sigma))
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    ker /= ker.sum()
    if n > r + 1:  # mirrored through the end points: a steady drive isn't pulled back at either end (held ends lag ~v*sigma)
        pad = np.concatenate([2 * corners[:1] - corners[r:0:-1], corners, 2 * corners[-1:] - corners[-2:-r - 2:-1]])
    else:
        pad = np.concatenate([np.repeat(corners[:1], r, 0), corners, np.repeat(corners[-1:], r, 0)])
    sm = np.stack([np.tensordot(ker, pad[j:j + 2 * r + 1], axes=(0, 0)) for j in range(n)])
    sm[ref] = corners[ref]
    return [fit_affine(box, sm[i]) for i in range(n)]


def parked(masks: list[np.ndarray], tol: float = PARKED_TOL) -> bool:
    """Does the vehicle stand still in the picture? Its lower body (wheels and sills, the bottom 40% of its outline,
    which no opening door or boot changes much) keeps its box within `tol` of its width on every frame."""
    boxes = []
    for m in masks:
        if not m.any():
            continue
        rows = np.flatnonzero(m.any(axis=1))
        low = m.copy()
        low[:int(rows[-1] - 0.4 * (rows[-1] - rows[0] + 1))] = False
        cols = np.flatnonzero(low.any(axis=0))
        boxes.append((float(cols[0]), float(cols[-1] + 1), float(rows[-1] + 1)))
    if len(boxes) < 2:
        return False
    b = np.array(boxes)
    width = float(np.median(b[:, 1] - b[:, 0]))
    return bool((np.ptp(b, axis=0) <= tol * width).all())


def track_vehicle(pm: list[np.ndarray], ref: int, size: tuple[int, int]) -> tuple[list[np.ndarray], dict]:
    """The old vehicle's motion from frame `ref`, as affines in work pixels (size = (W, H)), smoothed; and how well the
    moved outline lies on each frame's own (IoU)."""
    W, H = size
    ph, pw = pm[0].shape
    masks = [reference.largest(m) if m.any() else m for m in pm]
    if parked(masks):  # it doesn't move: a door or boot swinging open changes its outline, not where it is
        return [np.eye(3) for _ in pm], {"track_iou": [1.0] * len(pm), "track_iou_min": 1.0, "track_iou_mean": 1.0,
                                         "rigid_iou_min": 1.0, "rigid_iou_mean": 1.0, "parked": True}
    raw = align_outlines(masks, ref)
    ys, xs = np.nonzero(masks[ref])
    box = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]], float)
    sm = smooth_path(raw, box, ref)
    lx, line = contact_line(masks[ref])  # pinned where the old car meets the road: the middle of its contact line
    pin = np.array([(lx[0] + lx[-1]) / 2 + 0.5, float(np.interp((lx[0] + lx[-1]) / 2, lx, line))])
    rigid = [as_similarity(M, pin) for M in sm]
    ious, fits = [], []
    for i, (M, R) in enumerate(zip(sm, rigid)):
        moved = composite.warp(masks[ref].astype(np.float32), M, (pw, ph)) > 0.5
        ious.append(round(iou(moved, masks[i]), 4) if masks[i].any() else None)
        moved = composite.warp(masks[ref].astype(np.float32), R, (pw, ph)) > 0.5
        fits.append(round(iou(moved, masks[i]), 4) if masks[i].any() else None)
    S = np.diag([pw / W, ph / H, 1.0])
    full = [np.linalg.inv(S) @ R @ S for R in rigid]
    seen, held = [x for x in ious if x is not None], [x for x in fits if x is not None]
    return full, {"track_iou": ious, "track_iou_min": min(seen) if seen else None,
                  "track_iou_mean": round(float(np.mean(seen)), 4) if seen else None,
                  "rigid_iou_min": min(held) if held else None,
                  "rigid_iou_mean": round(float(np.mean(held)), 4) if held else None}


# --- wheels ------------------------------------------------------------------------------------------------------------


@dataclass
class Wheel:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of the region that is redrawn
    soft: np.ndarray  # where the blur goes, 0..1, the box's size
    far: bool  # the tread faces the camera: no hub to turn, a vertical blur
    centre: tuple[float, float] = (0.0, 0.0)  # in the box
    A: np.ndarray | None = None  # the wheel's ellipse -> a circle


def _grey(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def wheels_of(img: np.ndarray, found: list[np.ndarray], car: np.ndarray) -> list[Wheel]:
    """The vehicle's road wheels among SAM's "wheel"s on the painted frame: on the vehicle, in its lower half, and at
    least WHEEL_MIN of the biggest (a steering wheel and a hidden far wheel are not). A near wheel's bright rim is fitted
    with an ellipse (its plane, as the camera sees it); a wheel with no bright metal showing is a tread seen edge-on."""
    if not car.any():
        return []
    cy0, cy1 = np.flatnonzero(car.any(axis=1))[[0, -1]]
    on = composite.grow(car, 4)
    cands = [m for m in found if m.shape == car.shape and m.any() and (m & on).sum() >= 0.3 * m.sum()]
    cands = [m for m in cands if m.any() and np.nonzero(m)[0].mean() > cy0 + 0.5 * (cy1 - cy0)]
    if not cands:
        return []
    big = max(int(m.sum()) for m in cands)
    lum = _grey(img.astype(np.float32))
    H, W = car.shape
    out = []
    for m in cands:
        if m.sum() < WHEEL_MIN * big:
            continue
        ys, xs = np.nonzero(m)
        pad = 12
        x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)
        mb = m[y0:y1, x0:x1]
        bright = mb & (lum[y0:y1, x0:x1] > 110)
        far = bright.sum() < 0.08 * mb.sum()
        if far:
            soft = composite.gauss(composite.erode_box(mb, 2).astype(np.float32), 2.0)
            out.append(Wheel((x0, y0, x1, y1), soft, True))
            continue
        rim = composite.erode_box(composite.dilate_box(bright, 4), 4)
        rim = reference.fill_holes(reference.largest(rim))
        ry, rx = np.nonzero(rim)
        if len(ry) < 30:
            continue
        cx, cy = rx.mean() + 0.5, ry.mean() + 0.5
        cov = np.cov(np.stack([rx + 0.5 - cx, ry + 0.5 - cy]))
        ev, vec = np.linalg.eigh(cov)
        a, b = 4 * math.sqrt(max(ev[1], 1e-6)), 4 * math.sqrt(max(ev[0], 1e-6))  # full axes of a filled ellipse
        th = math.atan2(vec[1, 1], vec[0, 1])
        a, b = a * 1.12, b * 1.12  # a little of the sidewall turns with it
        R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])  # into the ellipse's frame
        yy, xx = np.mgrid[:y1 - y0, :x1 - x0]
        q = R @ np.stack([xx.ravel() + 0.5 - cx, yy.ravel() + 0.5 - cy])
        ell = ((q[0] / (a / 2)) ** 2 + (q[1] / (b / 2)) ** 2 <= 1).reshape(mb.shape)
        r = max(a, b) / 2
        A = np.diag([r / (a / 2), r / (b / 2)]) @ R
        out.append(Wheel((x0, y0, x1, y1), composite.gauss((ell & mb).astype(np.float32), 2.0), False, (cx, cy), A))
    return out


def spin(img: np.ndarray, wheels: list[Wheel], frame: int) -> np.ndarray:
    """The picture with its wheels turning at road speed on frame `frame`: each near wheel blurred along its own turn
    (ellipse -> circle -> rotate -> back, SPIN_SAMPLES samples over SPIN_SPAN degrees), a far one blurred up and down."""
    out = img.copy()
    for wh in wheels:
        x0, y0, x1, y1 = wh.box
        crop = img[y0:y1, x0:x1].astype(np.float32)
        if wh.far:
            p = np.pad(crop, ((7, 7), (0, 0), (0, 0)), mode="edge")
            blurred = sum(p[j:j + crop.shape[0]] for j in range(15)) / 15
        else:
            c = np.array(wh.centre)
            Ai = np.linalg.inv(wh.A)
            acc = np.zeros_like(crop)
            base = frame * SPIN_STEP
            for j in range(SPIN_SAMPLES):
                phi = math.radians(base + SPIN_SPAN * (j / (SPIN_SAMPLES - 1) - 0.5))
                rot = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
                L = Ai @ rot @ wh.A
                M = np.eye(3)
                M[:2, :2], M[:2, 2] = L, c - L @ c
                acc += composite.warp(crop, M, (x1 - x0, y1 - y0))
            blurred = acc / SPIN_SAMPLES
        s = wh.soft[..., None]
        out[y0:y1, x0:x1] = np.clip(np.rint(crop * (1 - s) + blurred * s), 0, 255).astype(np.uint8)
    return out


# --- the clean plate ---------------------------------------------------------------------------------------------------


def wet_road(p: dict) -> bool:
    """Does the scene's own description (or the take's `wet` setting) say the road mirrors what is on it?"""
    if p.get("wet") is not None:
        return bool(p.get("wet"))
    text = " ".join(str(p.get(k) or "") for k in ("scene", "prompt", "fill"))
    return bool(WET.search(text))


def ground_rows(car: np.ndarray) -> np.ndarray:
    """The rows a vehicle's shadow can be on (the mask's size): from its middle down."""
    out = np.zeros_like(car, dtype=bool)
    if car.any():
        rows = np.flatnonzero(car.any(axis=1))
        out[int((rows[0] + rows[-1]) / 2):] = True
    return out


def reflection_column(car: np.ndarray, reach: float = REFLECT_REACH) -> np.ndarray:
    """Where a vehicle mirrors itself on a wet road (the mask's size): straight down from its contact line, across the
    columns it spans, `reach` of its height (its body's reflection, and its lamps' streaks most of the way)."""
    out = np.zeros_like(car, dtype=bool)
    if not car.any():
        return out
    rows = np.flatnonzero(car.any(axis=1))
    xs, line = contact_line(car)
    h = car.shape[0]
    yy = np.arange(h)[:, None]
    top = np.floor(line - 0.02 * (rows[-1] - rows[0] + 1))
    out[:, xs] = (yy >= top[None, :]) & (yy <= (line + reach * (rows[-1] - rows[0] + 1))[None, :])
    return out & ~car


def fill_words(p: dict) -> str:
    """What the frame editor paints where the old vehicle was, from the scene's own words ("a wet two-lane rural road"
    -> "the empty wet two-lane rural road ..."): told nothing, it left a car-shaped dark patch on the asphalt."""
    text = " ".join(str(p.get(k) or "") for k in ("scene", "prompt"))
    m = SURFACE.search(text)
    ground = m.group(1).strip(" ,") if m else ""
    # only the surface's own words: "it drives along a gravel road" -> "gravel road"
    ground = re.split(r"\b(?:a|an|the|its|their|along|on|onto|down|across|over|through|in|into|at|by|past|round|around|"
                      r"drives?|driving|travels?|travelling|stands?|parked|is|it|seen|from)\b", ground, flags=re.I)[-1].strip(" ,")
    where = f"the empty {ground}" if ground else "the empty road"
    return (f"{where} and the scenery beyond it, continuing exactly as around it, evenly lit, with no vehicle, no shadow "
            "and no dark patch where it stood")


def under_darkness(frame: np.ndarray, car: np.ndarray) -> float | None:
    """How much darker the ground just under the vehicle's place is than the same rows beside it (levels): where a car
    stands its underside and shadow make it dark; on a clean plate it should be about as light as the road around."""
    if not car.any():
        return None
    y = composite.blur(composite.luma709(frame.astype(np.float32)), 3.0)
    H, W = car.shape
    ys, xs = np.nonzero(car)
    tall, x0, x1 = ys.max() - ys.min(), xs.min(), xs.max()
    cw = x1 - x0
    d = int(0.3 * cw)
    vals = []
    for r in range(int(ys.max() - 0.12 * tall), min(H, int(ys.max() + 0.2 * tall))):
        ins = y[r, x0 + cw // 10:x1 - cw // 10]
        out = np.r_[y[r, max(0, x0 - d):max(0, x0 - cw // 20)], y[r, min(W, x1 + cw // 20):min(W, x1 + d)]]
        if len(out) >= 20 and len(ins):
            vals.append(float(np.median(out) - np.median(ins)))
    return float(np.mean(vals)) if vals else None


def keeps_its_place(pm: list[np.ndarray], travel: float = 0.5) -> tuple[bool, float]:
    """Does the vehicle stay where it is on screen (its box's centre moves less than `travel` of its width over the
    window)? Then nothing behind it is ever uncovered, and real pixels from other frames can't clear it: Remove's
    generated fill does. Remove's own camera measure (the crop's shift) follows the car when it fills the crop, and a
    tracking shot read as locked-off got one still plate pasted under a moving road."""
    seen = [m for m in pm if m.any()]
    if len(seen) < 2:
        return True, 0.0
    (ax0, _, ax1, _), (bx0, _, bx1, _) = _box(seen[0]), _box(seen[-1])
    moved = abs((bx0 + bx1) - (ax0 + ax1)) / 2 / max(1.0, ax1 - ax0)
    return moved < travel, round(moved, 3)


def _box(m: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def remnants(bg, full: Path, pm: list[np.ndarray], size: tuple[int, int], samples: int = 5) -> dict:
    """The clean plate checked for what the old vehicle left behind, on a few frames: under_darkness on the plate and
    on the footage (with the car). ok when the plate is no darker there than REMNANT_LEVELS, or than REMNANT_SHARE of
    the footage."""
    W, H = size
    n = bg.n
    picks = sorted({round(j * (n - 1) / max(1, samples - 1)) for j in range(samples)})
    plate, foot = [], []
    for i, f in enumerate(common.read_frames(full)):
        if i in picks and i < len(pm) and pm[i].any():
            car = recast.upsize(pm[i], (W, H))
            a, b = under_darkness(bg.plate(i, f)[0], car), under_darkness(f, car)
            if a is not None and b is not None:
                plate.append(round(a, 1))
                foot.append(round(b, 1))
    if not plate:
        return {"under_darkness": [], "footage": [], "ok": True}
    dark, src = float(np.median(plate)), float(np.median(foot))
    share = dark / src if src > 8 else 0.0
    return {"under_darkness": plate, "footage": foot, "share": round(share, 2),
            "ok": bool(dark <= REMNANT_LEVELS or share <= REMNANT_SHARE)}


# --- the crop ------------------------------------------------------------------------------------------------------------


def crop_box(boxes: list[tuple[float, float, float, float]], size: tuple[int, int], max_px: int,
             min_px: int = 1024 * 1024) -> tuple[int, int, int, int, int, int]:
    """(x0, y0, cw, ch, w, h): a crop around everywhere the vehicle goes (`boxes`, work px, its shadow included) with
    room around it, sides multiples of SIDE, rendered at 1:1 (w, h = cw, ch). Grown for scene context up to `min_px`
    where the frame allows; scaled down only when it is over `max_px` (a 4K shot)."""
    W, H = size
    bx0, by0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    bx1, by1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    bw, bh = bx1 - bx0, by1 - by0
    k = max(W, H) / 1920
    mx, my = max(64 * k, 0.15 * bw), max(64 * k, 0.15 * bh)
    gw, gh = max(SIDE, W // SIDE * SIDE), max(SIDE, H // SIDE * SIDE)
    cw = min(gw, math.ceil((bw + 2 * mx) / SIDE) * SIDE)
    ch = min(gh, math.ceil((bh + 2 * my) / SIDE) * SIDE)
    while cw * ch < min_px and (cw < gw or ch < gh):
        if (cw / max(ch, 1) < W / H and cw < gw) or ch >= gh:
            cw = min(gw, cw + SIDE)
        else:
            ch = min(gh, ch + SIDE)
    cw, ch = min(cw, W // 2 * 2), min(ch, H // 2 * 2)
    x0 = int(min(max((bx0 + bx1) / 2 - cw / 2, 0), W - cw)) // 2 * 2
    y0 = int(min(max((by0 + by1) / 2 - ch / 2, 0), H - ch)) // 2 * 2
    w, h = cw, ch
    if w * h > max_px or w % SIDE or h % SIDE:
        s = min(1.0, math.sqrt(max_px / (cw * ch)))
        w, h = max(2 * SIDE, int(cw * s) // SIDE * SIDE), max(2 * SIDE, int(ch * s) // SIDE * SIDE)
    return x0, y0, cw, ch, w, h


def _shift(M: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """A full-frame affine as seen in a crop at (x0, y0): crop pixels -> crop pixels."""
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
    return np.linalg.inv(T) @ M @ T


def _into(M: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """A full-frame affine that lands in a crop at (x0, y0): full-frame pixels -> crop pixels."""
    return np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ M


# --- the ground it stands on ---------------------------------------------------------------------------------------------


def contact_line(car: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the vehicle meets the road: the lower hull of its outline (columns xs, rows ys, one per column from its
    first to its last), the line its tyres' bottoms span. A tyre stands on it; the body between them is above it."""
    cols = np.flatnonzero(car.any(axis=0))
    h = car.shape[0]
    bottom = h - 1 - np.argmax(car[::-1, cols], axis=0)
    hull: list[tuple[int, int]] = []
    for x, y in zip(cols.tolist(), bottom.tolist()):  # monotone chain: keep the hull that bulges downward (image y)
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0:  # the middle point lies above the new chord: drop it
                hull.pop()
            else:
                break
        hull.append((x, y))
    hx, hy = np.array(hull, float).T
    xs = np.arange(cols[0], cols[-1] + 1)
    return xs, np.interp(xs, hx, hy) + 1.0


def ground_region(car: np.ndarray) -> np.ndarray:
    """The road the vehicle stands on (0..1, the picture's size): in every column it spans, from GROUND_LIFT of its
    height above its lowest point (or its own top, if lower) down to GROUND_REACH under its contact line, a little past
    its ends, softened by GROUND_SOFT. car_on_ground takes this from the render, not the plate."""
    out = np.zeros(car.shape, np.float32)
    if not car.any():
        return out
    h, w = car.shape
    rows = np.flatnonzero(car.any(axis=1))
    tall = float(rows[-1] - rows[0] + 1)
    xs, line = contact_line(car)
    reach, soft = GROUND_REACH * tall, max(1.0, GROUND_SOFT * tall)
    top_of = np.argmax(car[:, xs], axis=0)
    lift = rows[-1] - GROUND_LIFT * tall
    pad = int(round(reach / 2))
    ext = np.arange(max(0, xs[0] - pad), min(w, xs[-1] + pad + 1))
    yy = np.arange(h)[:, None]
    lo = np.interp(ext, xs, line) + reach
    hi = np.interp(ext, xs, np.maximum(top_of, lift))
    hi = np.where((ext < xs[0]) | (ext > xs[-1]), np.interp(ext, xs, line) - reach, hi)  # past its ends: around the line
    out[:, ext] = ((yy >= hi[None, :]) & (yy <= lo[None, :])).astype(np.float32)
    out = np.clip(composite.gauss(out, soft), 0.0, 1.0)
    out[out < 0.01] = 0
    return out


# --- the matte ------------------------------------------------------------------------------------------------------------


def smooth_mattes(raw: list[np.ndarray], Mc: list[np.ndarray], i: int, k: float = 1.0) -> np.ndarray:
    """Frame i's matte: the raw tracks of frames i-2..i+2, each moved along the vehicle's own motion onto frame i, a
    weighted vote (the outline stops boiling), choked 1 px and feathered (FEATHER px at HD)."""
    h, w = raw[i].shape
    acc, ws = np.zeros((h, w), np.float32), 0.0
    for o, wt in MATTE_WEIGHTS.items():
        j = i + o
        if not 0 <= j < len(raw):
            continue
        m = raw[j].astype(np.float32)
        acc += wt * (m if j == i else composite.warp(m, Mc[i] @ np.linalg.inv(Mc[j]), (w, h)))
        ws += wt
    hard = acc / ws > 0.5
    for _ in range(max(1, round(k))):
        hard = composite.choke1(hard)
    soft = np.clip(composite.gauss(hard.astype(np.float32), FEATHER * k), 0.0, 1.0)
    soft[soft < 0.02] = 0
    return soft


def find_vehicle(ctx, png: Path, old: np.ndarray, out: Path) -> tuple[np.ndarray | None, str]:
    """The new vehicle on a painted frame: SAM 3.1's instance overlapping the old one most, by the first word of
    VEHICLE_LABELS that finds one; (None, "") when none does."""
    from pickups.tasks import recast_ltx
    for label in recast_ltx.VEHICLE_LABELS:
        try:
            found = common.select_instances(ctx, png, label, 1.0, Path(f"{out}_{label}"))
        except Exception:  # noqa: BLE001 - try the next word
            continue
        found = [m for m in found if m.shape == old.shape and (m & old).any()]
        if found:
            return max(found, key=lambda m: int((m & old).sum())), label
    return None, ""


def view_words(front: bool | None, aspect: float) -> str:
    """The angle a vehicle is seen from, in the still's and the painter's words: front or back (None: not known, and
    then not said: the painter follows the old vehicle), and how far round, from its box's width over height (head-on
    ~1.5, a three-quarter view ~2, side-on ~3)."""
    if aspect >= SIDE_ASPECT:
        return "a side view"
    if aspect <= HEAD_ON_ASPECT:
        return {True: "a front view, seen almost head-on", False: "a rear view, seen almost straight from behind",
                None: "a view from almost straight on"}[front]
    return {True: "a three-quarter front view", False: "a three-quarter rear view", None: "a three-quarter view"}[front]


def approach(pm: list[np.ndarray]) -> float:
    """How much bigger the old vehicle gets over the window (its last outline's area over its first): over 1 it comes
    towards the camera (so shows its front), under 1 it drives away (its back)."""
    seen = [int(m.sum()) for m in pm if m.any()]
    return seen[-1] / max(1, seen[0]) if len(seen) > 1 else 1.0


def vehicle_view(ctx, frame_png: Path, old: np.ndarray, out: Path, nearer: float = 1.0) -> str:
    """The angle the old vehicle is seen from on the anchor frame. Front or back: a grille SAM 3.1 finds on it is its
    front; else its headlights against its tail lights; else how it moves (`nearer`, approach: growing, it comes; shrinking,
    it goes), which a camera car pulling away from one coming after it gets wrong. An SUV's light bar was read as tail
    lights once, and a car coming head-on was painted from behind. How far round: its box's width over height."""
    if not old.any():
        return ""
    x0, y0, x1, y1 = _box(old)
    aspect = (x1 - x0) / max(1.0, y1 - y0)
    near = composite.grow(old, 4)
    count = {}
    for label in ("grille", "headlight", "tail light"):
        try:
            found = common.select_instances(ctx, frame_png, label, 1.0, Path(f"{out}_{label.replace(' ', '_')}"), min_frac=0.0)
        except Exception:  # noqa: BLE001 - not knowing: the next clue
            found = []
        count[label] = sum(1 for m in found if m.shape == old.shape and (m & near).sum() >= 0.5 * m.sum())
    front = count["grille"] + count["headlight"]
    back = count["tail light"]
    if front >= back + 2 and count["headlight"]:  # clear: both headlights, or a grille and a headlight, and few tail lights
        return view_words(True, aspect)
    if back >= front + 2 or (back >= 2 and not count["headlight"]):  # a grille alone was found on a rear bumper once
        return view_words(False, aspect)
    if not (front or back) and (nearer >= COMING or nearer <= 1 / COMING):
        return view_words(nearer >= COMING, aspect)
    return view_words(None, aspect)


def placement(new: np.ndarray, old: np.ndarray) -> dict:
    """How the painted vehicle sits on the old one's place: its contact line's height above the old one's (floats), its
    size (the square root of its area over the old one's) and their overlap (IoU); ok when it stands on the road, is
    within SIZE_RANGE and overlaps at least PLACE_IOU; `miss` scores how far off it is (0 = ok)."""
    lift = floats(new, old)
    size = float(np.sqrt(new.sum() / max(1, old.sum())))
    over = iou(new, old)
    miss = max(0.0, (lift or 0.0) - FLOAT_MAX) * 5 + max(0.0, abs(math.log(max(size, 1e-3))) - math.log(SIZE_RANGE[1])) + \
        max(0.0, PLACE_IOU - over)
    return {"floats": lift, "size": round(size, 3), "iou": round(over, 3), "miss": round(miss, 4), "ok": miss == 0}


def floats(new: np.ndarray, old: np.ndarray) -> float | None:
    """How far the painted vehicle's contact line stands above the old one's, as a share of the old one's height
    (median over the columns both span; 0 or less: on the road, or nearer the camera). The frame editor can paint a
    vehicle a little high, its tyres in the air, and every frame after inherits the gap."""
    if not new.any() or not old.any():
        return None
    xa, la = contact_line(new)
    xb, lb = contact_line(old)
    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    if hi - lo < 8:
        return None
    rows = np.flatnonzero(old.any(axis=1))
    gap = lb[lo - xb[0]:hi - xb[0] + 1] - la[lo - xa[0]:hi - xa[0] + 1]
    return round(float(np.median(gap)) / float(rows[-1] - rows[0] + 1), 4)


def smooth_offsets(readings: list, degree: int = 2) -> np.ndarray:
    """The render's drift off the footage, frame by frame (n x 2), from noisy readings (None: no reading): per axis a
    low-order curve through them, refitted twice without the readings more than max(2 px, 2.5 MADs) off it. A per-frame
    median left 5-frame bumps of 6 px that jolted the whole render; the drift itself is slow."""
    n = len(readings)
    out = np.zeros((n, 2))
    t = np.arange(n, dtype=float)
    for ax in range(2):
        ok = np.array([r is not None for r in readings])
        v = np.array([r[ax] if r is not None else 0.0 for r in readings], float)
        use = ok.copy()
        if use.sum() < max(5, n // 4):
            continue
        for _ in range(3):
            c = np.polyfit(t[use], v[use], min(degree, max(0, int(use.sum()) - 1)))
            res = v - np.polyval(c, t)
            mad = float(np.median(np.abs(res[use]))) if use.any() else 0.0
            keep = ok & (np.abs(res) <= max(2.0, 2.5 * 1.4826 * mad))
            if keep.sum() < max(5, n // 4) or np.array_equal(keep, use):
                break
            use = keep
        out[:, ax] = np.polyval(c, t)
    return out


def paint_anchor(pm: list[np.ndarray], k: int, n: int) -> int:
    """The frame the new vehicle is painted on: Recast's (`k`, where the old one is biggest), but for a car coming
    towards the camera, painted at the end, the middle latent frame instead: LTX drew the frames far from the keyframe
    where it liked (on the coast road 30 px low and too big). In the middle no frame is more than half the shot away."""
    mid = 8 * round((n - 1) / 16)
    if pm and approach(pm) >= COMING and k > (n - 1) / 2 and mid < len(pm) and pm[mid].any():
        return mid
    return k


def world_outliers(err: list[float]) -> np.ndarray:
    """The frames whose render draws the world behind the vehicle far off the footage (WORLD_OFF times the median, and
    WORLD_MIN levels): LTX's first frames, far from a keyframe, cross-faded two horizons on the coast road."""
    e = np.asarray(err, float)
    if not len(e):
        return np.zeros(0, bool)
    med = float(np.median(e))
    return (e > WORLD_OFF * med) & (e > med + WORLD_MIN)


def mirror_reflection(img: np.ndarray, matte: np.ndarray, line_x: np.ndarray, line_y: np.ndarray, tall: float,
                      where: np.ndarray) -> np.ndarray:
    """A wet road's mirror image of the vehicle: the picture flipped about its contact line (columns line_x, rows
    line_y), only the vehicle's own pixels (matte), streaked down the road, fading over REFLECT_LEN of its height, laid on
    the road at REFLECT_ALPHA, only inside `where` (the render's own ground: outside it the plate stays bit for bit) and
    never over the vehicle. The render's own reflection was a soft dark smear under the misty-road wagon."""
    h, w = matte.shape
    xs = np.arange(w)
    order = np.argsort(line_x)
    yl = np.interp(xs, line_x[order], line_y[order], left=np.nan, right=np.nan)
    ok = ~np.isnan(yl)
    if not ok.any():
        return img
    yl = np.where(ok, yl, 0.0)
    yy = np.arange(h, dtype=np.float32)[:, None]
    src = np.clip(np.rint(2 * yl[None, :] - yy), 0, h - 1).astype(int)
    f = img.astype(np.float32)
    refl = np.take_along_axis(f, np.repeat(src[..., None], 3, axis=2), axis=0)
    m = np.take_along_axis(matte.astype(np.float32), src, axis=0)
    reach = max(1.0, REFLECT_LEN * tall)
    d = yy - yl[None, :]
    m = m * np.clip(1.0 - d / reach, 0.0, 1.0) * (d > 0) * ok[None, :]
    sig = max(1.0, 0.015 * tall)
    has = (m > 0).astype(np.float32)
    refl = composite.gauss(refl * has[..., None], sig) / np.maximum(composite.gauss(has, sig), 1e-3)[..., None]
    m = composite.gauss(m, sig)
    a = (REFLECT_ALPHA * m * (1 - matte) * np.clip(where, 0, 1))[..., None]
    out = f * (1 - a) + refl * a
    res = img.copy()
    sel = a[..., 0] > 0
    res[sel] = np.clip(np.rint(out[sel]), 0, 255).astype(np.uint8)
    return res


def doors_or_people(pm: list[np.ndarray], om: list[np.ndarray]) -> dict | None:
    """Why a rigid new car can't stand in for this one, or None: parked, someone gets in or out (another tracked person
    touches it on TOUCH_FRAMES frames or more), or its outline changes shape (a door or the boot opening: its
    area swings by DOOR_SWING or more). The cliff-top shot has both; the new car came out with a man walking through it."""
    seen = [m for m in pm if m.any()]
    if not seen:
        return None
    touch = 0
    for m, o in zip(pm, om):
        if m.any() and o is not None and o.any() and (composite.grow(m, 3) & o).sum() >= 0.002 * m.sum():
            touch += 1
    areas = np.array([float(m.sum()) for m in seen])
    swing = float(areas.max() / max(1.0, areas.min()))
    big = [reference.largest(m) for m in seen]
    still = parked(big)
    if still and (touch >= TOUCH_FRAMES or swing >= DOOR_SWING):  # (someone passing a car that drives is fine)
        return {"touching_frames": touch, "area_swing": round(swing, 3), "parked": still}
    return None


def without_seam(still: np.ndarray, marked: np.ndarray, width: int = 6) -> np.ndarray:
    """The frame editor's plate with the seam it leaves along the edge of the red area it was asked to fill (`marked`,
    the frame it was shown) smoothed away: a band of `width` px each side of that edge, filled from the plate beside
    it (normalised blur). On the coast road the seam drew the old car's roof line across the sky behind the new one."""
    red = (marked[..., 0] > 240) & (marked[..., 1] < 20) & (marked[..., 2] < 20)
    if not red.any():
        return still
    band = composite.grow(red, width) & ~composite.shrink(red, width)
    ok = (~band).astype(np.float32)
    s = still.astype(np.float32)
    fill = composite.gauss(s * ok[..., None], width) / np.maximum(composite.gauss(ok, width), 1e-3)[..., None]
    out = still.copy()
    out[band] = np.clip(np.rint(fill[band]), 0, 255).astype(np.uint8)
    return out


def camera_path(readings: list, width: int = 5) -> np.ndarray:
    """The camera's move frame by frame (n x 2) from per-frame readings against the first (None: none): gaps filled
    by straight lines, then a running median over `width` frames (a stray reading goes, a real swing stays). A smooth
    curve through them (smooth_offsets) can't follow a camera that tilts down and then swings back up."""
    n = len(readings)
    out = np.zeros((n, 2))
    t = np.arange(n, dtype=float)
    ok = np.array([r is not None for r in readings])
    if ok.sum() < 2:
        return out
    for ax in range(2):
        v = np.array([r[ax] if r is not None else 0.0 for r in readings], float)
        v = np.interp(t, t[ok], v[ok])
        h = width // 2
        out[:, ax] = [np.median(v[i - min(h, i, n - 1 - i):i + min(h, i, n - 1 - i) + 1]) for i in range(n)]  # centred
    return out


def real_behind(bg, full: Path, size: tuple[int, int], still: np.ndarray | None = None, upper: float = REAL_UPPER) -> dict:
    """The world behind a vehicle that comes or goes, from real frames: the upper part of its place (`upper` of its
    height from its top: what is behind its body, sea, sky, trees, never the road it stands on) filled from the frames
    where that background shows, aligned for the camera's motion (measured on the world above and around it), over the
    generated plate in `bg` (changed in place). Where no frame ever shows it, the frame editor's still plate of the first
    frame (`still`, crop-sized), moved with the camera: the far background behind a car is one picture. LTX's own fill
    stays only where neither reaches. On the coast road LTX's fill invented bare trees and a car over the sea behind the
    oncoming SUV, and the render copied them."""
    cx0, cy0, cw, ch = bg.box
    k = bg.k
    crops = [c.copy() for i, c in enumerate(common.read_frames(full, f"crop={cw}:{ch}:{cx0}:{cy0}", (cw, ch))) if i < bg.n]
    n = len(crops)
    if n < 3:
        return {"used": False}
    covered = [composite.grow(bg.mattes_hard[i], round(REAL_GROW * k)) for i in range(n)]  # its trim and pillars too
    boxes = [_box(m) if m.any() else None for m in bg.mattes_hard[:n]]
    # The camera's move from the first frame: a tilt and a roll (on the coast road the horizon's slope across the frame
    # went from 6 to 31 px), read on the far world in the left and right halves, as a rotation about the crop's middle.
    left, right = (np.arange(cw) < cw // 2)[None, :], (np.arange(cw) >= cw // 2)[None, :]
    rl: list = [(0.0, 0.0)]
    rr: list = [(0.0, 0.0)]
    xl = xr = None
    for i in range(1, n):  # each frame against the first (chained, small per-frame drifts were lost to rounding)
        keep = ~composite.dilate_box(covered[0] | covered[i], round(24 * k))
        tops = [bx[1] + FAR_ROWS * (bx[3] - bx[1]) for bx in (boxes[0],) if bx]  # fixed rows: the horizon stays in
        if tops:  # the far world only, level with the vehicle's roof and above: a guardrail or verge near the camera
            keep[int(min(tops)):] = False  # slides by its own parallax (it read the coast road's 25 px tilt as 9)
        kl, kr = keep & left, keep & right
        if xl is None:
            xl = float(np.nonzero(kl)[1].mean()) if kl.any() else cw / 4
            xr = float(np.nonzero(kr)[1].mean()) if kr.any() else 3 * cw / 4
        rl.append(composite.masked_offset(crops[0], crops[i], kl, max_shift=round(96 * k)) if kl.mean() > 0.04 else None)
        rr.append(composite.masked_offset(crops[0], crops[i], kr, max_shift=round(96 * k)) if kr.mean() > 0.04 else None)
    pl, pr = camera_path(rl), camera_path(rr)
    xl, xr = xl or cw / 4, xr or 3 * cw / 4
    mid = np.array([cw / 2, ch / 2])
    moves = []
    for i in range(n):
        th = float(np.arctan2(pr[i][1] - pl[i][1], xr - xl))
        if not (any(r is not None for r in rl) and any(r is not None for r in rr)):
            th = 0.0
        c, s_ = np.cos(th), np.sin(th)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0, 0, 1.0]])
        t = (pl[i] + pr[i]) / 2
        A = np.array([[1, 0, mid[0] + t[0]], [0, 1, mid[1] + t[1]], [0, 0, 1.0]]) @ R @ \
            np.array([[1, 0, -mid[0]], [0, 1, -mid[1]], [0, 0, 1.0]])
        moves.append(A)  # frame-0 pixels -> frame-i pixels
    if still is not None:  # the editor's still is resized from its own grid: put its world on the first frame's
        keep = ~composite.dilate_box(covered[0], round(24 * k))
        o = composite.masked_offset(crops[0], still, keep, max_shift=round(48 * k)) if keep.mean() > 0.08 else None
        if o:
            still = composite.shift(still, -int(round(o[0])), -int(round(o[1])))
    ones = np.ones((ch, cw), np.float32)

    def to_first(img, A):
        return composite.warp(img, np.linalg.inv(A), (cw, ch))

    al = [np.clip(np.rint(to_first(crops[i], A)), 0, 255).astype(np.uint8) for i, A in enumerate(moves)]
    cov = [(to_first(covered[i].astype(np.float32), A) > 0.5) | (to_first(ones, A) < 0.99) for i, A in enumerate(moves)]
    fills, never = composite.temporal_fill(al, cov)
    # Seen on only a frame or two (the rim of what the old vehicle always hides): as good as never. The coast road's
    # fill drew the old SUV's roof line there from the one frame it was barely uncovered in.
    rare = np.sum([~c for c in cov], axis=0) < MIN_SEEN
    used = 0
    for i, A in enumerate(moves):
        if boxes[i] is None:
            continue
        f = composite.warp(fills[i].astype(np.float32), A, (cw, ch))
        valid = composite.warp(ones, A, (cw, ch)) > 0.99
        nv = (composite.warp((never[i] | (rare & cov[i])).astype(np.float32), A, (cw, ch)) > 0.5) | ~valid
        have = ~nv
        if still is not None:  # never seen: the still, moved with the camera
            st = composite.warp(still.astype(np.float32), A, (cw, ch))
            if (~nv & valid).any() and nv.any():  # toned to the real frames around it: the editor's sky came out lighter,
                st = composite.harmonize(np.clip(st, 0, 255), np.clip(f, 0, 255),  # an outline of the old car's place
                                         composite.grow(nv, round(8 * k)).astype(np.float32), grid=48, robust=True)
            f = np.where(nv[..., None], st, f)
            have = have | (nv & valid)
        x0, y0, x1, y1 = boxes[i]
        top = np.zeros((ch, cw), bool)
        top[:int(y0 + upper * (y1 - y0))] = True
        use = top & covered[i] & have
        if not use.any():
            continue
        wgt = np.clip(composite.gauss(use.astype(np.float32), 3.0 * k), 0.0, 1.0)[..., None] * have[..., None]
        bg.gen[i] = np.clip(np.rint(bg.gen[i].astype(np.float32) * (1 - wgt) + f * wgt), 0, 255).astype(np.uint8)
        used += int(use.sum())
    last = moves[-1]
    roll = float(np.degrees(np.arctan2(last[1, 0], last[0, 0])))
    return {"used": used > 0, "real_px_per_frame": round(used / max(1, n)),
            "camera_travel_px": [round((pl[-1][0] + pr[-1][0]) / 2), round((pl[-1][1] + pr[-1][1]) / 2)],
            "camera_roll_deg": round(roll, 2)}


def invented(bg, full: Path, samples: int = 5, upper: float = REAL_UPPER) -> dict:
    """Does the plate draw structure the footage around it hasn't got (bare trees over sea or sky)? On a few frames, the
    edge density of the plate in the upper part of the old vehicle's place against the footage's in a band beside it, at
    the same rows. ok when no frame's ratio passes INVENTED_MAX."""
    cx0, cy0, cw, ch = bg.box
    picks = sorted({round(j * (bg.n - 1) / max(1, samples - 1)) for j in range(samples)})
    ratios = []
    for i, f in enumerate(common.read_frames(full)):
        if i not in picks or i >= bg.n or not bg.mattes_hard[i].any():
            continue
        plate = bg.plate(i, f)[0][cy0:cy0 + ch, cx0:cx0 + cw]
        foot = f[cy0:cy0 + ch, cx0:cx0 + cw]
        m = bg.mattes_hard[i]
        x0, y0, x1, y1 = _box(m)
        rows = np.zeros(m.shape, bool)
        rows[int(y0):int(y0 + upper * (y1 - y0))] = True
        inside = m & rows
        beside = composite.dilate_box(m, round(0.4 * (x1 - x0)), 0) & ~composite.dilate_box(m, round(24 * bg.k)) & rows
        if inside.sum() < 200 or beside.sum() < 200:
            continue
        def edges(img):
            gx, gy = composite.sobel(composite.gauss(composite.luma709(img.astype(np.float32)), 1.0))
            return np.hypot(gx, gy)
        e_in = float(np.mean(edges(plate)[inside]))
        e_out = float(np.mean(edges(foot)[beside]))
        ratios.append(round(e_in / max(e_out, 1.0), 2))
    return {"edge_ratio": ratios, "ok": bool(not ratios or max(ratios) <= INVENTED_MAX)}


def contact_shadow(painted: np.ndarray, plate: np.ndarray, car: np.ndarray, band: float = CONTACT_BAND) -> np.ndarray:
    """The painted vehicle's contact shadows (H x W, 1 = none, darker below 1): its picture's luma over the clean
    plate's in a band of `band` of its height around its contact line, outside the vehicle, darken only, softened."""
    out = np.ones(car.shape, np.float32)
    if not car.any():
        return out
    rows = np.flatnonzero(car.any(axis=1))
    tall = float(rows[-1] - rows[0] + 1)
    xs, line = contact_line(car)
    yy = np.arange(car.shape[0], dtype=np.float32)[:, None]
    d = yy - line[None, :]
    near = np.zeros(car.shape, bool)
    near[:, xs] = (d > -band * tall) & (d < band * tall)
    near &= ~composite.grow(car, 1)
    pl = composite.gauss(composite.luma709(plate.astype(np.float32)), 2.0)
    pa = composite.gauss(composite.luma709(painted.astype(np.float32)), 2.0)
    r = np.clip(pa / np.maximum(pl, 1.0), 0.35, 1.0)
    out[near] = r[near]
    return np.clip(composite.gauss(out, 2.0), 0.35, 1.0)


# --- the task ------------------------------------------------------------------------------------------------------------


def _png(a: np.ndarray, path: Path) -> Path:
    Image.fromarray(a).save(path)
    return path


def _encode(folder: Path, fps: float, out: Path, size: tuple[int, int] | None = None) -> Path:
    vf = ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos"] if size else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(folder / "%04d.png"), *vf,
                    "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def zoom_box(old: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Where to paint a small vehicle (x0, y0, x1, y1, work px, the frame's own shape): a window around it in which it
    fills about ZOOM_FILL of the height, at most ZOOM_MAX times closer. None when it is big enough as it is. Painted on
    the whole frame, a car 50 px tall on a drone shot came back twice its size, whatever the instruction said."""
    if not old.any():
        return None
    W, H = size
    bx0, by0, bx1, by1 = _box(recast.upsize(old, (W, H)) if old.shape != (H, W) else old)
    z = min(ZOOM_MAX, ZOOM_FILL * H / max(1.0, by1 - by0), 0.6 * W / max(1.0, bx1 - bx0))
    if z < 1.5:
        return None
    cw, ch = W / z, H / z
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = min(max(0.0, cx - cw / 2), W - cw), min(max(0.0, cy - ch / 2), H - ch)
    return int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch))


def paint(ctx, anchor: np.ndarray, instruction: str, refs: list, seed: int, safe: bool, run_dir: Path,
          zoom: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The frame editor's new vehicle on the anchor frame, at the frame's own shape. Qwen-Image 2.1 edits at the
    picture's size rounded to 32 px: the frame is padded to that (edge pixels) and the padding cut off afterwards, so
    nothing is stretched (a 1080-line frame was painted squashed to 1024 lines before). With `zoom` (zoom_box) the
    window around a small vehicle is painted enlarged to the frame's size, then put back, its edge feathered."""
    if zoom is not None:
        x0, y0, x1, y1 = zoom
        H, W = anchor.shape[:2]
        win = np.asarray(Image.fromarray(anchor[y0:y1, x0:x1]).resize((W, H), Image.LANCZOS))
        done = paint(ctx, win, instruction, refs, seed, safe, run_dir)
        back = np.asarray(Image.fromarray(done).resize((x1 - x0, y1 - y0), Image.LANCZOS)).astype(np.float32)
        edge = np.zeros((y1 - y0, x1 - x0), np.float32)
        m = max(2, round(0.08 * min(x1 - x0, y1 - y0)))
        edge[m:-m, m:-m] = 1
        edge = np.clip(composite.gauss(edge, m / 2), 0, 1)[..., None]
        out = anchor.copy()
        out[y0:y1, x0:x1] = np.clip(np.rint(back * edge + anchor[y0:y1, x0:x1].astype(np.float32) * (1 - edge)), 0, 255).astype(np.uint8)
        return out
    H, W = anchor.shape[:2]
    pw, ph = -(-W // 32) * 32, -(-H // 32) * 32
    padded = np.pad(anchor, ((0, ph - H), (0, pw - W), (0, 0)), mode="edge")
    src = _png(padded, run_dir / "anchor_padded.png")
    job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(src), instruction, seed=seed, refs=refs, resolution=0,
                                    prefix="pickups/recast_paint", commercial_safe=safe, size=graphs.edit_size(pw, ph)))
    out = run_dir / "painted_raw.png"
    job.get_outputs("PU_Plate")[0].to_file(out)
    im = Image.open(out).convert("RGB")
    if im.size != (pw, ph):
        im = im.resize((pw, ph), Image.LANCZOS)
    return np.asarray(im)[:H, :W].copy()


def reused_painting(ctx, take: str, size: tuple[int, int], s0: int, k: int) -> np.ndarray | None:
    """An earlier Recast take's painted frame ("Try again with changes" keeping the new vehicle), put back at the work
    frame's own shape through that take's crop. None when that take painted another frame or kept none."""
    try:
        d = ctx.store.version_dir(ctx.shot_id, str(take))
        meta = ctx.store.version_meta(ctx.shot_id, str(take))
    except (FileNotFoundError, OSError, ValueError):
        return None
    f = d / "painted_frame.png"
    qa = (meta.get("details") or {}).get("_qa") or {}
    if not f.is_file() or int(qa.get("window_start", 0)) != s0 or int(qa.get("anchor_frame", 0)) != k:
        return None
    W, H = size
    im = Image.open(f).convert("RGB")
    crop = qa.get("crop") if qa.get("engine") != "vehicle-proxy" else None
    if crop and [crop[0], crop[1], crop[2], crop[3]] != [0, 0, W, H]:
        return None  # painted on part of the frame: the new path paints the whole frame
    return np.asarray(im.resize((W, H), Image.LANCZOS))


def run(ctx, s: Shot) -> dict:
    from pickups.tasks import recast_ltx

    p, run_dir, seed = s.p, s.run_dir, s.seed
    W, H = s.size
    n, k = s.frames, s.k
    k = paint_anchor(s.pm, k, n)
    kf = max(W, H) / 1920
    ke = max(1.0, kf)  # edge widths: a 1 px choke and ~1.1 px feather at HD, never thinner on a smaller shot
    qa: dict = {"engine": "vehicle-proxy"}
    if not p.get("doors_ok"):  # before any GPU time goes on it
        why = doors_or_people(s.pm, s.om)
        if why:
            raise RuntimeError(DOORS_MESSAGE)

    # 1. The clean plate: Remove's fill, as a stage of this job.
    ctx.stage(CLEAR)
    cdir = run_dir / "clear"
    cdir.mkdir(exist_ok=True)
    wproxy = cdir / "proxy.mp4"
    common.proxy_clip(ctx, s.full, wproxy)
    pw, ph = media.probe(wproxy).width, media.probe(wproxy).height
    pm = [m if m.shape == (ph, pw) else recast.upsize(m, (pw, ph)) for m in s.pm]
    sh = None
    wet = wet_road(p)
    if p.get("shadow", True):
        try:
            with span(ctx, 0.0, 0.1):
                sh = common.shadow_masks(ctx, wproxy, pm, cdir, labels=SHADOW_LABELS)
        except Exception:  # noqa: BLE001 - a missing shadow is better than a failed take
            sh = None
    if sh:  # on the ground only: nothing above the old vehicle's middle (a dark sea or hedge behind it isn't its shadow)
        sh = [x & ground_rows(m) for x, m in zip(sh, pm)]
    if wet:  # and the spray its tyres throw up, beside and behind it (left as a ghosted plume on the misty road)
        try:
            with span(ctx, 0.0, 0.1):
                spray = common.shadow_masks(ctx, wproxy, pm, cdir / "spray", labels=SPRAY_LABELS)
            sh = [a | b for a, b in zip(sh, spray)] if sh else spray
            qa["spray_removed"] = True
        except Exception:  # noqa: BLE001 - no spray found: as before
            pass
    if wet:  # its mirror image and its headlights' streaks under it go too: SAM's "reflection" finds only some of it
        cols = [reflection_column(m) for m in pm]
        sh = [a | b for a, b in zip(sh, cols)] if sh else cols
    qa["wet_road"] = wet
    fill = (p.get("fill") or "").strip() or fill_words(p)
    first = int(p.get("clear_seed", seed))
    stays, travel = keeps_its_place(pm)
    nearer = approach(pm)
    # Driving across the frame, or coming or going (what it hides on one frame shows on another): Remove decides, real
    # pixels where it can. On the coast road the generated fill invented a headland on the horizon behind the SUV's roof,
    # which the frames where it was still small and far show as open sea.
    sized = nearer >= COMING or nearer <= 1 / COMING
    camera = "moving" if stays and not sized else None
    tries, bg, best = [], None, None
    for t in range(PLATE_TRIES):
        tdir = cdir / f"try{t}"
        tdir.mkdir(exist_ok=True)
        with span(ctx, 0.1 + 0.9 * t / PLATE_TRIES, 0.1 + 0.9 * (t + 1) / PLATE_TRIES):
            b = remove.background(ctx, s.full, pm, sh, (W, H), s.fps, tdir, seed=first + t, fill=fill,
                                  shadow=bool(p.get("shadow", True)), camera=camera, given_locked=True)
        r = remnants(b, s.full, pm, (W, H))
        if sized and not b.locked:  # a car that comes or goes: what is behind it shows in other frames
            still = None
            if (tdir / "plate_raw.png").exists():
                cw_, ch_ = b.box[2], b.box[3]
                still = np.asarray(Image.open(tdir / "plate_raw.png").convert("RGB").resize((cw_, ch_), Image.LANCZOS))
                if (tdir / "first_crop.png").exists():
                    still = without_seam(still, np.asarray(Image.open(tdir / "first_crop.png").convert("RGB").resize(
                        (cw_, ch_), Image.NEAREST)))
            r["real_behind"] = real_behind(b, s.full, (W, H), still)
        r["invented"] = invented(b, s.full)
        r["ok"] = bool(r["ok"] and r["invented"]["ok"])
        tries.append({"seed": first + t, "camera": "locked" if b.locked else "moving", **r})
        score = float(np.median(r["under_darkness"])) if r["under_darkness"] else 0.0
        if best is None or score < best:
            bg, best = b, score
        if r["ok"]:
            break
        if b.locked:  # real pixels from other frames left the car behind: the generated fill instead
            camera = "moving"
    assert bg is not None
    qa["plate_remnants"] = {"fill": fill, "tries": tries, "method": bg.method(), "ok": any(x["ok"] for x in tries),
                            "travel_widths": travel, "approach": round(nearer, 3), "generated_fill": camera == "moving"}

    # 2. The new vehicle's picture.
    ctx.stage(GET)
    anchor_png = common._frame_png(s.full, k, run_dir / "anchor_full.png")
    anchor = np.asarray(Image.open(anchor_png).convert("RGB"))
    old_k = recast.upsize(pm[k], (W, H))
    with span(ctx, 0.0, 0.3):
        view = vehicle_view(ctx, anchor_png, old_k, run_dir / "view", approach(pm))
    if view and HIGH.search(" ".join(str(p.get(x) or "") for x in ("scene", "prompt"))):
        view += HIGH_WORDS  # a drone's view: painted level, the wagon showed its side, not its roof
    qa["view"] = view or None
    character, p, what, _views, made_still = recast_ltx.character_picture(ctx, p, run_dir, vehicle=True, view=view)

    # 3. Painted onto the anchor frame, whole and at the frame's own shape.
    ctx.stage(PAINT)
    who = (p.get("describe") or "").strip().rstrip(".")
    zoom = zoom_box(old_k, (W, H))
    qa["paint_zoom"] = list(zoom) if zoom else None
    seen_k = old_k[zoom[1]:zoom[3], zoom[0]:zoom[2]] if zoom else old_k  # where it is in what the painter sees
    instruction = (p.get("look_prompt") or "").strip() or recast_ltx.paint_vehicle(s.noun, recast_ltx.where_words(seen_k), who)
    if view and not (p.get("look_prompt") or "").strip():
        instruction = instruction.replace(" Keep everything else in <image1>", f" Here it is seen in {view}: paint it from that "
                                          "angle, not from the angle it has in <image2>. Keep everything else in <image1>")
    safe = ctx.commercial_safe()
    painted = reused_painting(ctx, p["painted_take"], (W, H), s.s0, k) if p.get("painted_take") else None
    qa["painting"] = "reused from " + str(p["painted_take"]) if painted is not None else "painted"
    reused = painted is not None
    gate: list[dict] = []
    tries = []
    for attempt in range(1 if reused else PAINT_TRIES):
        if not reused:
            said = instruction if attempt == 0 else instruction.replace(" Keep everything else in <image1>", SAME_SIZE.format(noun=recast_ltx.vehicle_words(s.noun)) +
                                                                        " Keep everything else in <image1>")
            painted = paint(ctx, anchor, said, [ctx.client.assets.from_file(character)], seed + attempt, safe, run_dir, zoom)
        png = _png(painted, run_dir / f"painted_{attempt}.png")
        # The new vehicle on it, cut out. A painting whose tyres stand above the road, or whose vehicle is far bigger or
        # smaller than the old one or somewhere else, is made again with the next seed (told the size outright).
        found, label = find_vehicle(ctx, png, old_k, run_dir / f"newcut{attempt or ''}")
        place = placement(found, old_k) if found is not None else None
        gate.append({"seed": seed + attempt, "found": label or None, **(place or {})})
        if place is not None:
            tries.append((place["miss"], attempt, painted, found, label))
            if place["ok"]:
                break
    qa["paint_gate"] = gate
    qa["paint_ok"] = any(g.get("ok") for g in gate)
    if not tries:
        raise RuntimeError("The new vehicle couldn't be found on the painted frame. Try again, or try another picture of it.")
    _, _, painted, car, how = min(tries, key=lambda t: (t[0], t[1]))  # the one placed best, the first on a tie
    painted_png = _png(painted, run_dir / "painted_full.png")

    # 4. The stand-in: the new vehicle cut out, its wheels spinning, moved along the old one's path over the plate.
    ctx.stage(PLACE)
    car = composite.choke1(reference.fill_holes(reference.largest(car)))
    alpha = np.clip(composite.gauss(car.astype(np.float32), 1.0 * ke), 0.0, 1.0)
    try:
        with span(ctx, 0.3, 0.4):
            wheel_masks = common.select_instances(ctx, painted_png, "wheel", 1.0, run_dir / "wheels", min_frac=0.0005)
    except Exception:  # noqa: BLE001 - no spin: the render may copy still hubs
        wheel_masks = []
    wheels = wheels_of(painted, wheel_masks, car)
    Ms, track_qa = track_vehicle(pm, k, (W, H))
    qa.update(track_qa)
    if track_qa.get("parked"):
        wheels = []  # a parked car's wheels stand still (the cliff-top wagon's spun)
    ys, xs = np.nonzero(car)
    corners = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]], float)
    ground = ground_region(car)  # the road it stands on, at the anchor: moved with it, taken from the render
    below = max(8, int(np.flatnonzero(ground.any(axis=1))[-1]) - int(ys.max())) if ground.any() else 8
    fill_soft = FILL_SOFT * float(ys.max() - ys.min() + 1)
    car_tall = float(ys.max() - ys.min() + 1)
    anchor_line = contact_line(car)  # where the painted vehicle meets the road, at the anchor (work px)
    boxes = []
    for M in Ms:
        c = (M @ np.hstack([corners, np.ones((4, 1))]).T).T
        boxes.append((c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max() + below))
    for j, m in enumerate(pm):  # and where the old one was, its shadow and reflection: the render redraws them too
        m = m | sh[j] if sh and j < len(sh) and sh[j].shape == m.shape else m
        if m.any():
            bx = _box(m)
            boxes.append((bx[0] * W / pw, bx[1] * H / ph, bx[2] * W / pw, bx[3] * H / ph))
    max_px = recast_ltx.max_pixels(recast_ltx.gpu_vram(ctx))
    x0, y0, cw, ch, w, h = crop_box(boxes, (W, H), max_px)
    Mc = [_shift(M, x0, y0) for M in Ms]
    rgba = np.dstack([painted.astype(np.float32) * alpha[..., None], alpha])
    pdir = run_dir / "proxy"
    pdir.mkdir(exist_ok=True)
    proxy_alpha: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    # Its tyres' contact shadows, from the painting: a darken-only luma ratio against the anchor's clean plate in a
    # thin band at its contact line, moved with it. Cut out alone, the far front tyre of the misty-road wagon had none,
    # and the render drew it hanging over the road.
    anchor_plate = None
    for j, f in enumerate(common.read_frames(s.full)):
        if j == k:
            anchor_plate = bg.plate(k, f)[0]
        if j >= k:
            break
    contact = contact_shadow(painted, anchor_plate, car) if anchor_plate is not None else None

    def place(i: int, frame: np.ndarray):
        src = spin(painted, wheels, i) if wheels else painted
        layer = np.dstack([src.astype(np.float32) * alpha[..., None], alpha]) if wheels else rgba
        wl = composite.warp(layer, _into(Ms[i], x0, y0), (cw, ch), cubic=True)
        a = np.clip(wl[..., 3:4], 0, 1)
        plate = bg.plate(i, frame)[0][y0:y0 + ch, x0:x0 + cw].astype(np.float32)
        if contact is not None:
            r = composite.warp(contact - 1.0, _into(Ms[i], x0, y0), (cw, ch)) + 1.0
            plate = plate * np.clip(r, 0.0, 1.0)[..., None]
        out = np.clip(np.rint(np.clip(wl[..., :3], 0, None) + plate * (1 - a)), 0, 255).astype(np.uint8)
        _png(out, pdir / f"{i + 1:04d}.png")
        return i, np.rint(a[..., 0] * 255).astype(np.uint8), src if i == k else None

    look = None
    with span(ctx, 0.4, 1.0):
        for j, (i, a, src) in enumerate(common.parallel_frames((f for i, f in enumerate(common.read_frames(s.full)) if i < n), place)):
            ctx.check()
            proxy_alpha[i] = a
            if src is not None:
                look = src
            ctx.progress((j + 1) / n)
    proxy_mp4 = _encode(pdir, s.fps, run_dir / "proxy_crop.mp4", None if (w, h) == (cw, ch) else (w, h))
    look_png = _png(np.asarray(Image.fromarray(look[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.LANCZOS)), run_dir / "look_crop.png")

    # 5. The render: Layout-to-Render follows the stand-in, the painted frame its look and its anchor keyframe.
    ctx.stage(RENDER)
    motion = recast_ltx.drive_words(recast_ltx.motion_words(pm))
    prompt = (p.get("prompt") or "").strip() or recast_ltx.ltx_prompt(who, motion, p.get("scene") or "", vehicle=True)
    look_asset = ctx.client.assets.from_file(look_png)
    job = ctx.run(graphs.layout_render(ctx.client.assets.from_file(proxy_mp4), look_asset, prompt, s.fps, w, h, seed=seed,
                                       prefix="pickups/recast_ltx", keyframes=[(look_asset, k, 1.0)]))
    gdir = run_dir / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)
    gen = sorted(gdir.glob("*.png"))[:n]
    if not gen:
        raise RuntimeError("LTX-2.5 didn't send the shot back. Try again.")
    seq = run_dir / "gen_seq"
    seq.mkdir(exist_ok=True)
    for i, f in enumerate(gen):
        shutil.copyfile(f, seq / f"{i + 1:04d}.png")
    gen_mp4 = _encode(seq, s.fps, run_dir / "gen.mp4")

    # 6. The rendered vehicle, followed from the stand-in's outline, onto the clean plate.
    ctx.stage(BLEND)
    seed_mask = recast.upsize(proxy_alpha[k] > 127, (w, h))
    with span(ctx, 0.0, 0.3):
        track = common.track_from(ctx, gen_mp4, k, [seed_mask], run_dir, folder="new_track", prefix="pickups/recast_track")
    near_r = round(40 * kf)
    raw = []
    for i in range(len(gen)):
        m = recast.upsize(track[i], (cw, ch)) if i < len(track) else np.zeros((ch, cw), bool)
        m = m & composite.dilate_box(proxy_alpha[i] > 127, near_r)  # only what is where the new vehicle should be
        raw.append(reference.fill_holes(m) if m.any() else m)
    gen_crops = [np.asarray(Image.open(f).convert("RGB").resize((cw, ch), Image.LANCZOS)) if (w, h) != (cw, ch)
                 else np.asarray(Image.open(f).convert("RGB")) for f in gen]
    grain_on = str(p.get("grain", "fine")).lower() != "off"
    need = None
    if grain_on:
        picks = sorted({0, len(gen) // 2, len(gen) - 1})
        plates = [bg.plate(i, f)[0][y0:y0 + ch, x0:x0 + cw] for i, f in enumerate(common.read_frames(s.full)) if i in picks]
        need = np.sqrt(np.clip(composite.grain_profile(plates) ** 2 - composite.grain_profile([gen_crops[i] for i in picks]) ** 2, 0, None))
        need = need if np.any(need > 0.05) else None
    # The render's own picture can sit a little off the footage (LTX draws the frames after a pinned keyframe a couple of
    # pixels low, and far from it up to 20 px: two horizons cross-faded into a band). Where both show the footage's
    # world, away from the vehicle and the old one's place, the offset is measured on each frame, smoothed over +-4
    # frames, and the whole render (vehicle, road and matte) moved back onto the footage.
    bx0, by0, bcw, bch = bg.box
    offsets, world_err = [], []
    for i, f in enumerate(common.read_frames(s.full)):
        if i >= len(gen):
            break
        ctx.check()
        plate_i = f[y0:y0 + ch, x0:x0 + cw]  # where it is measured (away from the old car's place) the plate IS the footage
        away = composite.dilate_box(proxy_alpha[i] > 0, round(48 * kf))
        if i < bg.n:
            fm = np.zeros((H, W), bool)
            fm[by0:by0 + bch, bx0:bx0 + bcw] = bg.matte(i) > 0
            away |= composite.dilate_box(fm[y0:y0 + ch, x0:x0 + cw], round(24 * kf))
        rows = np.flatnonzero((proxy_alpha[i] > 0).any(axis=1))
        if len(rows):  # the world behind and above it only: the render redraws the road it drives on its own way
            away[max(0, rows[0] + (rows[-1] - rows[0]) // 3):] = True
        offsets.append(composite.masked_offset(plate_i, gen_crops[i], ~away, max_shift=round(REGISTER_MAX * max(cw, ch))))
        # How far the render's own world (behind and above the vehicle) is from the footage's: LTX's first frames far
        # from the keyframe can cross-fade two horizons (the coast road), and those frames keep the plate there.
        g_l, f_l = composite.luma709(gen_crops[i].astype(np.float32)), composite.luma709(plate_i.astype(np.float32))
        world_err.append(float(np.median(np.abs(composite.gauss(g_l, 2.0) - composite.gauss(f_l, 2.0))[~away]))
                         if (~away).any() else 0.0)
    offs = np.clip(smooth_offsets(offsets), -REGISTER_MAX * max(cw, ch), REGISTER_MAX * max(cw, ch))
    off_world = world_outliers(world_err)
    qa["render_world_error"] = {"levels": [round(x, 1) for x in world_err],
                                "plate_behind_frames": [int(j) for j in np.flatnonzero(off_world)]}
    good = np.flatnonzero(~off_world)
    if len(good) and off_world.any() and off_world.sum() <= HOLD_MAX:
        # A few such frames (the render's first ones): the nearest good frame's render, moved with the vehicle, stands
        # in for each, with its track, so neither the cross-faded horizon nor a ghost of the old car's roof comes through.
        for j in np.flatnonzero(off_world):
            g = int(good[np.argmin(np.abs(good - j))])
            Mg = Mc[j] @ np.linalg.inv(Mc[g])
            gen_crops[j] = np.clip(np.rint(composite.warp(gen_crops[g], Mg, (cw, ch), cubic=True)), 0, 255).astype(np.uint8)
            raw[j] = composite.warp(raw[g].astype(np.float32), Mg, (cw, ch)) > 0.5
            offs[j] = offs[g]
        qa["render_world_error"]["held_from_neighbour"] = True
    qa["render_offset_px"] = {"max": round(float(np.abs(offs).max()), 2) if len(offs) else 0.0,
                              "mean": [round(float(x), 2) for x in offs.mean(axis=0)] if len(offs) else [0.0, 0.0],
                              "raw": [None if o is None else [round(o[0], 1), round(o[1], 1)] for o in offsets],
                              "used": [[round(float(a), 1), round(float(b), 1)] for a, b in offs]}

    # The render's road fades out before the crop's edge (where that edge isn't the frame's): cut off there, a wet
    # road's reflection column ended in a hard line across the misty road.
    inside = np.ones((ch, cw), np.float32)
    ramp = max(8, round(fill_soft * 2))
    r = np.arange(ramp, dtype=np.float32) / ramp
    if y0 > 0:
        inside[:ramp] *= r[:, None]
    if y0 + ch < H:
        inside[-ramp:] *= r[::-1][:, None]
    if x0 > 0:
        inside[:, :ramp] *= r[None, :]
    if x0 + cw < W:
        inside[:, -ramp:] *= r[::-1][None, :]

    def onto_footage(img: np.ndarray, i: int) -> np.ndarray:
        dx, dy = offs[i] if i < len(offs) else (0.0, 0.0)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return img
        T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]])
        size = (img.shape[1], img.shape[0])
        out = composite.warp(img, T, size)
        cov = composite.warp(np.ones(img.shape[:2], np.float32), T, size)  # the edge it uncovers: the render unmoved
        out = out + (1 - cov[..., None] if img.ndim == 3 else 1 - cov) * img.astype(np.float32)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out.astype(img.dtype)

    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, s.fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, s.fps, kind="h264", scale_long=1920)
    lumas: list[np.ndarray] = [None] * len(gen)  # type: ignore[list-item]

    def blend(i: int, frame: np.ndarray):
        plate, touched, _, _, c0 = bg.plate(i, frame)
        out = plate.copy()
        ai = np.zeros((H, W), bool)
        ai[by0:by0 + bch, bx0:bx0 + bcw] = touched
        st: dict = {}
        if i < len(gen):
            matte = smooth_mattes(raw, Mc, i, ke)
            region = plate[y0:y0 + ch, x0:x0 + cw]
            under = composite.warp(ground, _into(Ms[i], x0, y0), (cw, ch))
            refl = None
            if i < bg.n:  # where the old vehicle was, the render's road too: the plate's fill is generated there anyway
                fm = np.zeros((H, W), np.float32)
                fm[by0:by0 + bch, bx0:bx0 + bcw] = bg.matte(i) > 0.5
                fm = np.clip(composite.blur(fm[y0:y0 + ch, x0:x0 + cw], fill_soft) * 2.0, 0.0, 1.0)  # its edge, wide
                rows = np.flatnonzero((proxy_alpha[i] > 0).any(axis=1))
                if i < len(off_world) and off_world[i] and len(rows):  # the render's world is off here: the plate's above it
                    cut = rows[0] + BEHIND * (rows[-1] - rows[0])
                    fm *= np.clip((np.arange(ch, dtype=np.float32)[:, None] - cut) / max(4.0, 2 * fill_soft) + 0.5, 0.0, 1.0)
                if wet and i < len(pm) and pm[i].any():
                    # On a wet road the old vehicle's reflection streaks down the road: there, below the new one's own
                    # ground, the clean plate's wet road, not the render's (a blotchy dark smear on the misty road); the
                    # new vehicle's mirror image goes on it below.
                    col = np.zeros((H, W), np.float32)
                    col[:] = recast.upsize(reflection_column(pm[i]), (W, H))
                    refl = np.clip(composite.blur(col[y0:y0 + ch, x0:x0 + cw], fill_soft), 0.0, 1.0)
                    own = composite.warp(ground, _into(Ms[i], x0, y0), (cw, ch))
                    fm = fm * (1 - refl * (1 - np.clip(own * 2, 0, 1)))
                under = np.maximum(under, fm)
            under = np.where(under < 0.01, 0.0, under * inside).astype(np.float32)
            matte = np.clip(onto_footage(matte, i), 0.0, 1.0)
            matte[matte < 0.02] = 0
            comp, wt, st = composite.car_on_ground(onto_footage(gen_crops[i], i), region, matte, under,
                                                   onto_footage(raw[i].astype(np.float32), i) > 0.5, need, seed=i, k=kf)
            where = wt if refl is None else np.maximum(wt, refl * (1 - matte))
            if wet and where.any():
                cl = (_into(Ms[i], x0, y0) @ np.vstack([anchor_line[0] + 0.5, anchor_line[1], np.ones(len(anchor_line[0]))]))[:2]
                comp = mirror_reflection(comp, matte, cl[0] - offs[i][0], cl[1] - offs[i][1], car_tall, where)
            out[y0:y0 + ch, x0:x0 + cw] = comp
            sel = (matte > 0) | (where > 0)
            ai[y0:y0 + ch, x0:x0 + cw] |= sel
            outside = np.ones((H, W), bool)
            outside[y0:y0 + ch, x0:x0 + cw] = ~sel
            c0 += int(np.any(out[outside] != plate[outside], axis=-1).sum())  # 0: the plate, bit for bit
            lumas[i] = np.rint(composite.luma709(comp.astype(np.float32))).astype(np.uint8)
            hard, pa = matte > 0.5, proxy_alpha[i] > 127
            st |= {"area": int(hard.sum()), "proxy_area": int(pa.sum()), "iou": iou(hard, pa),
                   "edge_px": int(((matte > 0) & (matte < 1)).sum())}
        if i < len(s.om) and s.om[i].any():  # everyone else stays footage, in front of the new vehicle (a man at its boot)
            keep = composite.soft_matte(recast.upsize(s.om[i], (W, H)), round(2 * ke), 1.5 * ke)[..., None]
            out = np.clip(np.rint(out * (1 - keep) + frame * keep), 0, 255).astype(np.uint8)
            ai &= keep[..., 0] < 1
        tinted = out.copy()
        reg = tinted[ai].astype(np.float32) * 0.45 + np.array(recast.AI_TINT, np.float32) * 0.55
        tinted[ai] = reg.astype(np.uint8)
        return out, tinted, c0, int(ai.sum()), st

    changed = touched_px = done = 0
    stats = []
    with span(ctx, 0.3, 1.0):
        for i, (out, tinted, c, t, st) in enumerate(common.parallel_frames(common.read_frames(s.full), blend)):
            ctx.check()
            done += 1
            changed += c
            touched_px += t
            stats.append(st)
            out_w.write(out)
            rec_w.write(tinted)
            ctx.progress((i + 1) / max(n, 1))
    out_w.close()
    rec_w.close()

    # The new vehicle's shape, frame by frame: each frame moved back onto the anchor's by the motion, its body's edges
    # compared with the anchor's (wheels left out: they turn). 1 = the same body; a morphing render scores far lower.
    body = composite.erode_box(composite.warp(alpha, _into(Ms[k], x0, y0), (cw, ch)) > 0.78, round(7 * kf))
    for wh in wheels:
        wm = np.zeros((H, W), bool)
        bx, by, bx1, by1 = wh.box
        wm[by:by1, bx:bx1] = wh.soft > 0.05
        body &= ~composite.dilate_box(composite.warp(wm.astype(np.float32), _into(Ms[k], x0, y0), (cw, ch)) > 0.5, round(10 * kf))
    shape = []
    for i in range(len(gen)):
        if lumas[i] is None or lumas[k] is None:
            shape.append(None)
            continue
        back = composite.warp(lumas[i].astype(np.float32), Mc[k] @ np.linalg.inv(Mc[i]), (cw, ch))
        e = composite.edge_agreement(back, lumas[k].astype(np.float32), body, shift=SHAPE_SHIFT)
        shape.append(None if e is None else round(e, 3))
    seen = [x for x in shape if x is not None and x == x]
    st_ok = [x for x in stats if x]
    areas = np.array([x["area"] for x in st_ok], float)
    qa["shape_stability"] = shape
    qa["shape_stability_min"] = min(seen) if seen else None
    qa["shape_stability_mean"] = round(float(np.mean(seen)), 3) if seen else None
    qa["matte_edges"] = {
        "choke_px": max(1, round(ke)), "feather_px": round(FEATHER * ke, 2),
        "edge_px_mean": round(float(np.mean([x["edge_px"] for x in st_ok])), 1) if st_ok else None,
        "iou_with_stand_in_min": round(min(x["iou"] for x in st_ok), 4) if st_ok else None,
        "iou_with_stand_in_mean": round(float(np.mean([x["iou"] for x in st_ok])), 4) if st_ok else None,
        "area_to_stand_in": [round(float(min(x["area"] / max(1, x["proxy_area"]) for x in st_ok)), 3),
                             round(float(max(x["area"] / max(1, x["proxy_area"]) for x in st_ok)), 3)] if st_ok else None,
        "area_jump_max": round(float(np.max(np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1))), 4) if len(areas) > 1 else None,
        "colour_gain_first": st_ok[0]["gain"] if st_ok else None,
        "shadow_min_ratio": min(x["shadow_min"] for x in st_ok) if st_ok else None,
    }

    ctx.stage(WRITE)
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=s.full)
    make_viewer(s.full, ctx.vdir / "before.mp4")
    _png(painted, ctx.vdir / "painted_frame.png")  # the whole frame, at its own shape
    total = W * H * max(done, 1)
    receipt = {"original": round(100 * (1 - touched_px / total), 2), "borrowed": 0.0, "ai": round(100 * touched_px / total, 2)}
    made = media.made_from(s.src, s.s0, done)
    models = ["SAM 3.1", graphs.edit_model(safe)] + [m for m in bg.models() if m != graphs.edit_model(safe)] + \
        ["LTX-2.5 + Layout-to-Render IC-LoRA"] + (["FLUX 3 klein"] if made_still else [])
    qa |= {"window_start": s.s0, "window_chosen": "yours" if s.start is not None else "auto", "anchor_frame": k,
           "guide": "first frame" if k == 0 else f"frame {k}", "new_seed": how, "crop": [x0, y0, cw, ch], "work_size": [w, h],
           "one_to_one": (w, h) == (cw, ch), "max_pixels": max_px, "wheels": len(wheels), "reference": what,
           "searched_frames": s.n_search, "grain_added": None if need is None else [round(float(x), 2) for x in need],
           "ground_below_px": below}
    return {"label": "Replace a car", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, done, s.fps), "engine": "ltx", "performance_size": f"{w}×{h}",
                        "reference": what, "subject": "vehicle", "pixels_changed_outside_selection": changed,
                        "models": models, **({"made_from": made} if made else {}),
                        "grain": "matched to your shot" if grain_on else "off", "look_prompt": instruction, "prompt": prompt,
                        "_qa": qa}}
