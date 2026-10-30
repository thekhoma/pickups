"""Replace a person: a clean plate, a 1:1 crop that follows them, their pose, and a masked LTX render.

The depth-guided path in recast_ltx pastes an upscaled render back and leaves a soft edge. This is the path that
passed on the castle-hall prototype (an internal prototype, the 4k_c2 composite):

1. The background where they were. Frames are aligned and exposure-matched; each frame gets its background back only
   where the person was (grown 8 px, feathered 3 px); their cast shadow stays, carried over the restored ground as a light
   ratio. Each
   pixel comes from the nearest frame where that spot was clear, with that frame's brightness drift taken out; the
   per-pixel median of the clear frames, with the footage's grain, only where no frame shows it.
2. A moving crop at 1:1 that follows them. Nothing is scaled to fit a model grid until the render itself, and only
   when the card cannot hold the crop (a 24 GB card renders up to 1024×1792 at 1:1; past that the character is
   scaled back up).
3. SDPose draws their pose with ComfyUI's built-in nodes. The weights are sdpose_wholebody_fp16.safetensors.
4. The frame editor paints the new character onto the crop's first frame, in the scene's light, facing the way the
   person does (recast_ltx.paint_view), and the paint gate checks it did.
5. LTX-2.5 with the Union Control IC-LoRA renders only inside a blocky mask over the plate. Frame 0 is the painted
   character and is not denoised. The pose is the guide. Not the shot's depth.
6. SAM 3.1 mattes the rendered character, smoothed over 3 frames; the guided filter's soft alpha snaps it to their
   edges, and the edge colour comes from just inside them. Their contact shadow is added.
   Outside that matte and shadow the plate stays, and outside the crop the footage stays.

`guide` "pose" on the params opts into this path. Until it matches the prototype, the depth path stays the default.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite, envblend, graphs, media
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.tasks import common, recast

STAGES = [
    Stage("Preparing the shot", 15),
    Stage("Finding the person", 35),
    Stage("Building the clean plate", 40),
    Stage("Reading the character sheet", 10),
    Stage("Reading the pose", 25),
    Stage("Painting them into the shot", 40),
    Stage("Performing the scene", 180),
    Stage("Blending at full resolution", 90),
    Stage("Writing your files", 45),
]
PLATE, SHEET, POSE, PAINT, RENDER, BLEND, WRITE = (s.label for s in STAGES[2:])

# A 24 GB card renders the masked graph at 1:1 up to this many pixels: on the RTX 4090, 81 frames at 1024×1792 (the
# 96 GB prototype's size) took 87 s with tiled encodes, and 960×1728 took 99 s. A bigger crop is scaled to fit and the
# character scaled back up. Cards with NATIVE_VRAM_GB render any crop at 1:1.
SMALL_CARD_PIXELS = 1024 * 1792
NATIVE_VRAM_GB = 40.0
BIG_CARD_PIXELS = 1280 * 2240  # a 96 GB card: about 1.6x the 24 GB size
MIN_RENDER_PIXELS = 768 * 1344
WHOLE_CUT = 0.02  # of their outline on the frame's edge: whole in the picture
POSE_SIDE = 0.05  # of their height: right and left torso sticks closer than this are a profile
WHOLE_ASPECT = 3.4  # a whole standing figure's height over its width
LEGS_BELOW = (" Their legs and feet are not in <image1> yet: paint them on down to the ground at the bottom of the "
              "picture, at their size, walking the way they walk.")
GROW_AT = 1792  # the prototype grew the person 110 px inside a 1792-tall crop
GROW_PX = 110
BLOCK = 32
TEMPORAL = 3
# The plate goes back only where the person was: grown this many px, feathered this much. Their floor shadow is not
# replaced (shadow_ratio); FEET_BAND / SHADOW_REACH only keep lenders out of the zone where it falls.
RESTORE_GROW = 8
RESTORE_FEATHER = 3.0
FEET_BAND = 0.15
SHADOW_REACH = 0.12
DARKER = 5.0
# A far foot behind the near leg comes and goes in the person's track: a black heel and a white sole stayed beside the
# new character's boot. Their feet are taken FOOT_GROW of their height wider, on the frame and in the lending frames.
FOOT_GROW = 0.04
UNSURE_NEAR = 0.08  # of the character's height: how far from them the render's own ground may be used
FILL_COPY_TOL = 45.0  # levels: how far a copied ground pixel may be from the smooth fill
MIN_SEEN = 3  # frames that must show a spot of the background for the plate to count as having seen it
FILL_MARGIN = 6  # px round a hole the smooth fill takes no colour from
UNSURE_GROW = 3  # px round where no frame showed the background and the render's picture is used, feathered:
UNSURE_FEATHER = 2.0
SAME_OFFSETS = [(dy, dx) for dy in (-24, -12, 0, 12, 24) for dx in (-24, -12, 0, 12, 24) if dy or dx]
SAME_LEVELS = 16.0  # what came back this close to the person's own pixels, inside them, is them (black trousers: 15 vs 25)
# Remnants the mask missed: within REMNANT_REACH of their height, REMNANT_LEVELS from the background the nearest clear
# frames show (after a REMNANT_SMOOTH blur), touching them. A dark beaded strap on a shaded pillar differs by ~15-30.
REMNANT_REACH = 0.12
REMNANT_LEVELS = 12.0
REMNANT_SMOOTH = 0.5
# The background comes from the nearest frame where that spot was clear. A frame lends no pixel within CLEAR_GROW px
# of where the person was on it; a pixel that differs from the median by more than FILL_TOL levels (a missed coat
# tail) takes the median instead; each lending frame's brightness drift is measured over DRIFT_SIGMA px around.
CLEAR_GROW = 12
FILL_TOL = 16.0
FILL_SAMPLES = 5  # lenders read per pixel: the nearest that agrees with their median is used
DRIFT_SIGMA = 48
DRIFT_MAX = 10.0  # levels: a frame a few frames away is never further off than this
CELL = 16  # the drift is measured on cell means
MOTION_TILE = 128  # half-size tiles whose shifts give the camera's motion between frames
# The character's matte: the guided filter's own soft alpha (radius, eps), its 0.1-0.9 stretched to 0-1, and the edge
# colour taken from about DECON_PX inside so the render's background doesn't rim them.
MATTE_R, MATTE_EPS = 6, 1e-3
MATTE_LO, MATTE_HI = 0.1, 0.9
DECON_PX = 1.0
RIM_PX, RIM_LIFT = 2, 6.0  # an opaque edge this wide, this much lighter than inside, is a rim light
PLATE_BAND = 16  # px: the ramp from the render back to the real plate beyond the mask (knight T1.5: 16 beat 32)
MATTE_REACH = 2  # px past the track the matte may reach
SHADOW_SMOOTH, SHADOW_MAX = 10, 0.25  # the contact shadow: the render-vs-plate light ratio, smoothed; its strongest darkening
# The original person's cast shadow stays: the pose path drives the new character with the same pose and timing, so the
# shadow is about right. Where the person's own pixels go back to the ground, the ground is darkened by the light ratio
# measured round them (SHADOW_BAND of their height up from their feet, up to SHADOW_KEEP_REACH of it away), smoothed.
SHADOW_BAND = 0.30
SHADOW_KEEP_REACH = 0.45
SHADOW_FLOOR = 0.35  # a ratio under this is not a shadow (a black shoe the mask missed)
SHADOW_STEADY = 0.07
SHADOW_FAR = 0.2  # of their height: a shadow must run this far from them to be a cast shadow  # and the ratio must be this steady over a few px
# A soft contact shadow where the feet meet the ground, luminance only: AO_MAX at the feet, fading over AO_SIGMA of their height.
# When the new person's silhouette is not close to the old one's (IoU under SHADOW_IOU on the first frame), the old shadow is a
# remnant: it is removed (REDRAW_BAND of their height up from the feet, REDRAW_REACH of it away), that ground goes into the
# mask LTX may change, and the contact shadow follows what LTX drew (up to REDRAW_MAX).
SHADOW_IOU = 0.7
REDRAW_BAND, REDRAW_REACH, REDRAW_MAX = 0.20, 0.30, 0.6
AO_MAX, AO_SIGMA, AO_BAND = 0.35, 0.012, 0.08
UNMIX_ALPHA = 0.3  # an edge pixel at least this much the character: the background's share is taken out


@dataclass
class Shot:
    """What Recast on LTX already worked out before it took the person path."""
    p: dict
    run_dir: Path
    seed: int
    src: object
    s0: int
    start: int | None
    pm: list[np.ndarray]
    om: list[np.ndarray]
    k: int
    full: Path
    size: tuple[int, int]
    fps: float
    frames: int
    n_search: int
    qa: dict = field(default_factory=dict)


def stages_for(params: dict) -> list[Stage] | None:
    """The person stages, when the pose path was asked for (`guide` "pose") and the words don't name a vehicle."""
    if str(params.get("guide") or "").lower() != "pose":
        return None
    if str(params.get("kind", "")).lower() == "vehicle":
        return None
    from pickups.tasks.recast_ltx import vehicle_noun
    if str(params.get("kind", "")).lower() != "person" and vehicle_noun(params.get("text", "")):
        return None
    return STAGES


def use_stages(ctx) -> None:
    if hasattr(ctx, "use_stages"):
        ctx.use_stages(STAGES)


@contextmanager
def span(ctx, lo: float, hi: float):
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


# --- the crop ----------------------------------------------------------------------------------------------------------


def snap64(n: float, limit: int) -> int:
    """The largest multiple of 64 that is at most `limit` and at least 64, near `n`."""
    n = int(max(64, round(n)))
    n = min(n, limit // 64 * 64)
    return max(64, n // 64 * 64)


def _smooth(xs: np.ndarray) -> np.ndarray:
    """A quadratic fit plus a short blur, so the crop glides. Short clips are left as they are."""
    n = len(xs)
    if n < 4:
        return xs.astype(float)
    t = np.arange(n)
    fit = np.polyval(np.polyfit(t, xs, 2), t)
    rad = min(12, n // 2)
    g = np.exp(-0.5 * (np.arange(-rad, rad + 1) / 5.0) ** 2)
    g /= g.sum()
    pad = np.pad(xs - fit, rad, mode="edge")
    return fit + np.convolve(pad, g, mode="valid")


def crop_plan(masks: list[np.ndarray], fw: int, fh: int) -> tuple[list[tuple[int, int]], int, int]:
    """A 1:1 crop that follows the person. (boxes of even origins, crop width, crop height).

    The crop is about 1.5 times their height and 4:7, and holds them at their biggest, in multiples of 64. When their feet stay near the bottom of
    the frame the crop sits on the bottom, as it did for the castle-hall walk. Otherwise it stays centred on them.
    """
    cxs, cys, feet, heights, widths = [], [], [], [], []
    last = (fw / 2, fh / 2, fh * 0.7)
    for m in masks:
        if m.shape != (fh, fw):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((fw, fh), Image.NEAREST)) > 127
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            cxs.append(last[0])
            cys.append(last[1])
            feet.append(last[2])
            continue
        heights.append(int(ys.max() - ys.min() + 1))
        widths.append(int(xs.max() - xs.min() + 1))
        cxs.append(float(np.median(xs)))
        cys.append(float(np.median(ys)))
        feet.append(float(ys.max()))
        last = (cxs[-1], cys[-1], feet[-1])
    tall = int(np.median(heights)) if heights else max(64, fh // 2)
    # Big enough for them at their biggest too: walking towards the camera, they end twice as tall as they started.
    ch = snap64(max(tall * 1.5, max(heights, default=0) * 1.15), fh)
    cw = snap64(max(ch * 4 / 7, max(widths, default=0) * 1.15), fw)
    sx, sy = _smooth(np.array(cxs, float)), _smooth(np.array(cys, float))
    on_floor = float(np.median(feet)) >= fh - max(8, 0.08 * fh)
    boxes = []
    for x, y in zip(sx, sy):
        x0 = int(round(min(max(x - cw / 2, 0), fw - cw)))
        y0 = (fh - ch) if on_floor else int(round(min(max(y - ch / 2, 0), fh - ch)))
        boxes.append((x0 // 2 * 2, y0 // 2 * 2))
    return boxes, cw, ch


def visible_share(mask: np.ndarray) -> float:
    """How much of a person cut off by the bottom of the picture is in it, from the shape of what shows (a whole
    figure is about WHOLE_ASPECT times as tall as it is wide)."""
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(ys):
        return 1.0
    return float(np.clip((ys[-1] - ys[0] + 1) / max(1, xs[-1] - xs[0] + 1) / WHOLE_ASPECT, 0.35, 1.0))


PAINT_TOL = 0.08  # of their height: how far short of that the painted feet may fall
PAINT_REACH = 0.6  # of the rows below the picture their feet should be in, the painted feet must reach
HIP_AT = 0.47  # of a standing figure's height, from the top of the head down to the hips


def hip_share(pose: np.ndarray | None) -> float | None:
    """How far down the picture a drawn SDPose frame puts the hips (the low ends of the neck-to-hip sticks), as a share
    of its height; None when they aren't drawn."""
    if pose is None:
        return None
    hips = _pose_px(pose, [POSE_R[1], POSE_L[1]])
    if not hips.any():
        return None
    return float(np.flatnonzero(hips.any(axis=1))[-1] + 1) / pose.shape[0]


BOTTOM_CUT = 0.35  # the person's width at the picture's bottom rows, over their widest: more and the edge cuts them off


def cut_by_bottom(mask: np.ndarray, rows: int = 3) -> bool:
    """Does the bottom of the picture cut the person off? A shoe tip touching the edge (a man whole in the picture) does
    not: what reaches the last rows must be a good share of their widest part. Told it did, the painter was asked for a
    figure 832 rows taller than the man was, and the gate rejected the three clean knights."""
    live = np.flatnonzero(mask.any(axis=1))
    if not len(live):
        return False
    widest = max(int(mask[y].sum()) for y in live[::4])
    at_bottom = int(np.count_nonzero(mask[-rows:].any(axis=0)))
    return at_bottom >= BOTTOM_CUT * max(widest, 1)


def below_edge(mask: np.ndarray, pose: np.ndarray | None = None) -> int:
    """How many rows of a person cut off by the bottom of the picture are below it (a multiple of 32, at most the
    picture's height). From the hips in the pose when it has them (head to hips is about HIP_AT of a person), else from
    the shape of what shows: an arm held out made a man cut off at the thighs look half there, and the frame editor
    painted a small whole knight in the top half of a picture twice his height."""
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(ys):
        return 0
    seen = ys[-1] - ys[0] + 1
    hip = hip_share(pose)
    if hip is not None and hip * mask.shape[0] > ys[0] + 0.25 * seen:
        rows = ys[0] + (hip * mask.shape[0] - ys[0]) / HIP_AT - (ys[-1] + 1)
        rows = max(0.0, rows) * 1.1
    else:
        rows = seen * (1 / visible_share(mask) - 1) * 1.1
    return int(min(mask.shape[0], max(32, round(rows / 32) * 32)))


def painted_small(people_dir: Path, old: np.ndarray, rows: int, ext: int) -> str:
    """On a canvas carried on `ext` rows below the picture (`rows` tall): the painted person (the one over the old)
    must reach well down it, at the old one's size; "" when they do or it can't be told. Told to paint a man cut off
    at the thighs down to the ground below, the frame editor drew a knight two-thirds his size, feet just below the
    picture."""
    from pickups.tasks import recast
    masks = [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(people_dir.glob("*.png"))]
    masks = [recast.upsize(m, (old.shape[1], old.shape[0])) for m in masks]
    best = max(masks, key=lambda m: (m & old).sum(), default=None)
    if best is None or not (best & old).any():
        return ""
    ys = np.flatnonzero(best.any(axis=1))
    # the feet should reach `needed`; a rounding-level miss (within PAINT_TOL of their height) is not "smaller"
    needed = rows + PAINT_REACH * ext / 1.1
    top = int(np.flatnonzero(old.any(axis=1))[0]) if old.any() else 0
    return "" if ys[-1] >= needed - PAINT_TOL * max(1.0, needed - top) else "painted smaller than they are"


def extend_down(frame: np.ndarray, plate: np.ndarray, rows: int) -> np.ndarray:
    """The frame carried on `rows` below its bottom edge with the clean plate's ground: its bottom band mirrored back and
    forth, so it reads as more of the same ground (the whole plate mirrored put the sky and the hills upside down below
    the edge, and the painter stood the person on that seam)."""
    ground = plate[:-8]  # the last rows of a frame are often a codec's black
    band = ground[-max(8, min(len(ground), len(ground) // 6)):]
    out, flip = [], True
    while sum(len(b) for b in out) < rows:
        out.append(band[::-1] if flip else band)
        flip = not flip
    below = np.concatenate(out, axis=0)[:rows] if out else ground[:0]
    return np.concatenate([frame, below], axis=0)


POSE_R = [(255, 0, 0), (0, 255, 0)]  # SDPose's sticks from the neck to the right shoulder and the right hip
POSE_L = [(255, 85, 0), (0, 255, 255)]  # ... to the left shoulder and the left hip
POSE_FACE = (255, 255, 255)  # the face's points
POSE_NOSE = (0, 0, 255)  # SDPose's stick from the neck to the nose


def _pose_px(a: np.ndarray, cols: list[tuple[int, int, int]]) -> np.ndarray:
    m = np.zeros(a.shape[:2], bool)
    for c in cols:
        m |= np.all(np.abs(a.astype(np.int16) - np.array(c, np.int16)) <= 8, axis=-1)
    return m


def pose_facing(pose: np.ndarray) -> str:
    """Which way a person faces, from a drawn SDPose frame: "front", "back", "side-right", "side-left"; "" when the
    body isn't drawn. Their right shoulder and hip left of their left ones in the picture face the camera, right of
    them face away, on top of them is a profile, turned the way the face's points lie from the neck. (SDPose draws
    face points on backs of heads too, so a face's presence says nothing.)"""
    r, l = _pose_px(pose, POSE_R), _pose_px(pose, POSE_L)
    body = r | l
    if not r.any() or not l.any():
        return ""
    ys = np.flatnonzero(_pose_px(pose, POSE_R + POSE_L + [(255, 170, 0), (170, 255, 0), (0, 255, 85), (0, 170, 255),
                                                          (0, 255, 170), (0, 85, 255)]).any(axis=1))
    tall = max(1, int(ys[-1] - ys[0] + 1)) if len(ys) else max(1, int(np.ptp(np.flatnonzero(body.any(axis=1)))) + 1)
    rx, lx = np.nonzero(r)[1].mean(), np.nonzero(l)[1].mean()
    d = (rx - lx) / tall
    if d <= -POSE_SIDE:
        return "front"
    if d >= POSE_SIDE:
        return "back"
    face = _pose_px(pose, [POSE_FACE])
    if not face.any():
        return ""
    return "side-right" if np.nonzero(face)[1].mean() > (rx + lx) / 2 else "side-left"


def same_facing(want: str, got: str) -> bool:
    """Does the painted person face as the footage does? A profile may come out turned a little towards or away from
    the camera (front or back) but never the other way round."""
    if want == got or not want or not got:
        return True
    wb, _, ww = want.partition("-")
    gb, _, gw = got.partition("-")
    if wb == "side":
        return gb == "side" and gw == ww or gb == "back"
    if gb == "side":
        return wb == "back"
    return False


def painted_pose(ctx, png: Path, rw: int, rh: int, fps: float, out: Path) -> np.ndarray | None:
    """The drawn SDPose map of a painted frame (SDPose on a one-frame clip of it); None where it can't be had."""
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "pose_check.mp4"
    _rgb_video(clip, [_resize(np.asarray(Image.open(png).convert("RGB")), rw, rh)], fps)
    try:
        job = ctx.run(graphs.sdpose(ctx.client.assets.from_file(clip), prefix="pickups/person_pose_check"))
        outs = job.get_outputs("PU_Pose")
        if not outs:
            return None
        outs[0].to_file(out / "pose_check.png")
    except Exception:  # noqa: BLE001 - a check we can't run passes
        return None
    return np.asarray(Image.open(out / "pose_check.png").convert("RGB"))


def painted_facing(ctx, png: Path, rw: int, rh: int, fps: float, out: Path, pose: np.ndarray | None = None) -> str:
    """pose_facing of a painted frame."""
    pose = painted_pose(ctx, png, rw, rh, fps, out) if pose is None else pose
    return pose_facing(pose) if pose is not None else ""


NEVER_SEEN_MAX = 0.10  # of their area that no frame of the window shows clear: more, and the pose path isn't used
GUARD_SAMPLES = 5  # frames the never-seen share is measured on


# A take fails when a part of the body the pose shows is not in the final matte: share of that part's pose pixels inside it.
PARTS = {
    "torso": POSE_R[:1] + POSE_L[:1],
    "arms and hands": [(255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0)],
    "legs and feet": POSE_R[1:] + POSE_L[1:] + [(0, 255, 85), (0, 255, 170), (0, 170, 255), (0, 85, 255)],
}
PART_MIN = 0.6  # a part must be at least this inside the matte, on average over the frames it is drawn in
PART_MIN_PX = 12


HEAD_UP, HEAD_DOWN = 0.22, 0.2  # with a nose stick: the box from above to below its tip, in shoulder widths
HEAD_HALF, HEAD_NEAR, HEAD_FAR = 0.18, 0.12, 0.55  # the box above the neck, in shoulder widths: half-width, and from, to
TORSO_TO_SHOULDERS = 0.77  # a body's shoulder width against its neck-to-hip length (front on)
FORESHORTENED = 0.6        # shoulders drawn narrower than this share of that are seen side-on: the torso sets the head's size


def _stick_end(stick: np.ndarray, gap: int = 3) -> float:
    """The lowest row of a stick drawn down from the neck: its rows from the top while they run on (gaps of at most `gap`).
    A joint's dot drawn in the same colour further down (the right knee's is the left hip stick's colour) is not the stick."""
    rows = np.unique(np.nonzero(stick)[0])
    end = int(rows[0])
    for r in rows[1:]:
        if r - end > gap:
            break
        end = int(r)
    return float(end)


def _main_rows(stick: np.ndarray, neck: float | None = None, gap: int = 3) -> np.ndarray:
    """A stick's pixels in one run of rows (gaps of at most `gap`): the run nearest the `neck` row (where the neck-to-hip
    sticks start), else the biggest. Seen side-on a face line near the eye holds pixels of the right shoulder stick's red,
    and the nose's and feet's dots are red too; taken as the shoulders they put the head box off the head."""
    rows = np.unique(np.nonzero(stick)[0])
    if rows.size == 0:
        return stick
    runs, start = [], rows[0]
    for a, b in zip(rows[:-1], rows[1:]):
        if b - a > gap:
            runs.append((start, a))
            start = b
    runs.append((start, rows[-1]))
    if neck is None:
        y0, y1 = max(runs, key=lambda r: int(stick[r[0]:r[1] + 1].sum()))
    else:
        y0, y1 = min(runs, key=lambda r: max(r[0] - neck, neck - r[1], 0))
    out = np.zeros_like(stick)
    out[y0:y1 + 1] = stick[y0:y1 + 1]
    return out


def _stick_top(stick: np.ndarray, gap: int = 3) -> float:
    """The highest row of a stick drawn up from the neck: its rows from the bottom while they run on."""
    rows = np.unique(np.nonzero(stick)[0])[::-1]
    end = int(rows[0])
    for r in rows[1:]:
        if end - r > gap:
            break
        end = int(r)
    return float(end)


def head_box(pose: np.ndarray) -> np.ndarray | None:
    """Where a head must be: a box above the neck (where the two neck-to-shoulder sticks meet), as wide and as high as a
    head is to the shoulders. The face points SDPose draws sit anywhere on a back of a head, so they say nothing about
    one; the neck and shoulders are drawn steadily. Seen side-on the shoulders are foreshortened to a few pixels, so the
    head's size then comes from the neck-to-hip length instead. None when they aren't drawn, or the box leaves the picture."""
    hips = _pose_px(pose, POSE_R[1:] + POSE_L[1:])
    neck = float(np.nonzero(hips)[0].min()) if int(hips.sum()) >= PART_MIN_PX else None  # the hip sticks start at the neck
    r, l = _main_rows(_pose_px(pose, POSE_R[:1]), neck), _main_rows(_pose_px(pose, POSE_L[:1]), neck)
    if int(r.sum()) < PART_MIN_PX or int(l.sum()) < PART_MIN_PX:
        return None
    both = r | l
    ys, xs = np.nonzero(both)
    sw = float(xs.max() - xs.min() + 1)
    nx = float(np.nonzero(r)[1].mean() + np.nonzero(l)[1].mean()) / 2
    ny = float(ys.min())
    if int(hips.sum()) >= PART_MIN_PX:  # seen side-on the shoulders shrink to a few pixels, the head doesn't: size it from the torso
        est = TORSO_TO_SHOULDERS * (_stick_end(hips) - ny)
        if sw < FORESHORTENED * est:
            sw = est
    y0, y1 = int(ny - HEAD_FAR * sw), int(ny - HEAD_NEAR * sw)
    cx = nx
    nose = _pose_px(pose, [POSE_NOSE])
    nose[int(ny) + 3:] = False  # the left knee's dot is drawn in the same colour, further down
    if int(nose.sum()) >= PART_MIN_PX:  # the head is where the neck-to-nose stick ends: forward of the neck when seen side-on
        ty = _stick_top(nose)
        tx = float(np.nonzero(nose[int(ty):int(ty) + 3])[1].mean())
        cx = (tx + nx) / 2
        y0, y1 = int(ty - HEAD_UP * sw), int(ty + HEAD_DOWN * sw)
    if y0 < 0 or sw < 8:
        return None
    box = np.zeros(pose.shape[:2], bool)
    box[y0:y1, max(0, int(cx - HEAD_HALF * sw)):int(cx + HEAD_HALF * sw)] = True
    return box


def part_masks(pose: np.ndarray) -> dict[str, np.ndarray]:
    """The pose map's pixels for each body part (head, torso, arms and hands, legs and feet), where drawn."""
    out = {}
    head = head_box(pose)
    if head is not None:
        out["head"] = head
    for name, cols in PARTS.items():
        m = _pose_px(pose, cols)
        if int(m.sum()) >= PART_MIN_PX:
            out[name] = m
    return out


def part_shares(parts: dict[str, np.ndarray], matte: np.ndarray) -> dict[str, float]:
    """For each part, the share of its pose pixels inside the (slightly grown) matte."""
    inside = composite.grow(matte > 0.3, 3)
    return {k: float((m & inside).sum() / m.sum()) for k, m in parts.items()}


def missing_parts(shares: list[dict[str, float]], minimum: float = PART_MIN) -> dict[str, float]:
    """Parts whose mean share over the frames they appear in is under `minimum`: {part: mean}."""
    names = {k for s in shares for k in s}
    mean = {k: float(np.mean([s[k] for s in shares if k in s])) for k in names}
    return {k: round(v, 2) for k, v in mean.items() if v < minimum}


def never_seen(frames: list[np.ndarray], masks: list[np.ndarray], samples: int = GUARD_SAMPLES) -> float:
    """The share of the person's area that no other frame of the window shows clear (the camera's motion taken out),
    averaged over a few frames: the ground the background can't come back from. A camera following a walker, or
    someone walking on the spot, leaves most of it unseen; someone crossing a still or panning shot leaves none.
    `frames` and `masks` at the proxy's size."""
    n = len(frames)
    if n < 2 or not any(m.any() for m in masks):
        return 0.0
    H, W = frames[0].shape[:2]
    masks = [recast_upsize(m, (W, H)) for m in masks]
    ref = n // 2
    step = [np.eye(3)] + [frame_motion(frames[i - 1], frames[i], masks[i - 1], masks[i]) for i in range(1, n)]
    A = [np.eye(3) for _ in range(n)]
    for i in range(ref + 1, n):
        A[i] = A[i - 1] @ step[i]
    for i in range(ref - 1, -1, -1):
        A[i] = A[i + 1] @ np.linalg.inv(step[i + 1])
    ones = np.ones((H, W), np.float32)
    clear = [(composite.warp((~composite.grow(m, 2)).astype(np.float32), A[i], (W, H)) > 0.99)
             & (composite.warp(ones, A[i], (W, H)) > 0.99) for i, m in enumerate(masks)]
    shares = []
    for i in sorted({round(k * (n - 1) / max(1, samples - 1)) for k in range(samples)}):
        person = composite.warp(masks[i].astype(np.float32), A[i], (W, H)) > 0.5
        if not person.any():
            continue
        seen = np.zeros((H, W), bool)
        for j, c in enumerate(clear):
            if j != i:
                seen |= c
        shares.append(float((person & ~seen).sum()) / float(person.sum()))
    return float(np.mean(shares)) if shares else 0.0


def recast_upsize(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from pickups.tasks import recast
    return recast.upsize(m, size) if m.shape != (size[1], size[0]) else m


def pose_guard(masks: list[np.ndarray], frames: list[np.ndarray] | None) -> str:
    """Why the pose path can't take this window ("" when it can): they must be whole in the picture on its first or
    its last frame (it is painted there; on the last, rendered backwards), and, when `frames` are given, nearly all
    the ground behind them must be seen clear in some frame (it is put back from the footage)."""
    from pickups.tasks import recast_ltx
    shown = [m for m in masks if m.any()]
    if not shown:
        return "the person isn't in this part of the shot"
    if recast_ltx.cut_off(shown[0]) > WHOLE_CUT and recast_ltx.cut_off(shown[-1]) > WHOLE_CUT:
        return "the frame cuts them off all through this part of the shot"
    if frames and never_seen(frames, masks) > NEVER_SEEN_MAX:
        return "the ground behind them is never seen clear (the camera follows them, or they stay in one place)"
    return ""


def backwards(person: list[np.ndarray]) -> bool:
    """Render the window backwards: the frame cuts them off on its first frame but they are whole on its last."""
    from pickups.tasks import recast_ltx
    if len(person) < 2 or not person[0].any() or not person[-1].any():
        return False
    return recast_ltx.cut_off(person[0]) > WHOLE_CUT and recast_ltx.cut_off(person[-1]) <= WHOLE_CUT


def pose_window(masks: list[np.ndarray], frames: int, total: int, start: int | None = None) -> int:
    """The first frame of the window the pose path renders: the user's, else the window where the person is on screen
    most among those that start with them whole in the picture (the painted first frame is the character LTX keeps;
    painted cut off at the bottom of the frame, the walker on the mountain trail came back as a small whole figure in
    the wrong place). recast_ltx.choose_window when no frame shows them whole."""
    from pickups.tasks import recast_ltx
    if start is not None or not masks:
        return recast_ltx.choose_window(masks, frames, total, start)
    last = max(0, total - frames)
    score = recast_ltx.presence(masks)
    best, best_score = None, -1.0
    for s in range(min(last, len(masks) - 1) + 1):
        if not masks[s].any() or recast_ltx.cut_off(masks[s]) > WHOLE_CUT:
            continue
        got = float(score[s:s + frames].sum())
        if got > best_score:
            best, best_score = s, got
    if best is None:  # none starts with them whole: one that ends with them whole is rendered backwards
        for s in range(min(last, len(masks) - 1) + 1):
            e = min(len(masks) - 1, s + frames - 1)
            if not masks[e].any() or recast_ltx.cut_off(masks[e]) > WHOLE_CUT:
                continue
            got = float(score[s:s + frames].sum())
            if got > best_score:
                best, best_score = s, got
    return best if best is not None else recast_ltx.choose_window(masks, frames, total, start)


def render_size(cw: int, ch: int, vram_gb: float | None) -> tuple[int, int]:
    """The size LTX renders the crop at: its own size when the card holds it (SMALL_CARD_PIXELS on 24 GB,
    BIG_CARD_PIXELS from NATIVE_VRAM_GB), else the most that fits; and never under MIN_RENDER_PIXELS (a person a few
    hundred px tall is drawn poorly; the render is scaled back down to the crop, which keeps it sharp)."""
    cap = BIG_CARD_PIXELS if (vram_gb or 0) >= NATIVE_VRAM_GB else SMALL_CARD_PIXELS
    px = cw * ch
    if px > cap:
        s = math.sqrt(cap / px)
        return snap64(cw * s, cw), snap64(ch * s, ch)
    if px < MIN_RENDER_PIXELS:
        s = math.sqrt(MIN_RENDER_PIXELS / px)
        return max(64, round(cw * s / 64) * 64), max(64, round(ch * s / 64) * 64)
    return cw, ch


# --- the plate ---------------------------------------------------------------------------------------------------------


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dy, dx) that moves b onto a, sub-pixel."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    A, B = np.fft.rfft2(a * win), np.fft.rfft2(b * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-6
    r = np.fft.irfft2(R, s=a.shape)
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)

    def sub(c, cm, cp):
        d = cm - 2 * c + cp
        return 0.0 if abs(d) < 1e-9 else float(0.5 * (cm - cp) / d)

    h, w = r.shape
    dy = py + sub(r[py, px], r[(py - 1) % h, px], r[(py + 1) % h, px])
    dx = px + sub(r[py, px], r[py, (px - 1) % w], r[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def fshift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Sub-pixel translate of a float image by (dy, dx)."""
    h, w = img.shape[:2]
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    ramp = np.exp(-2j * np.pi * (ky * dy + kx * dx)).astype(np.complex64)
    if img.ndim == 2:
        return np.fft.irfft2(np.fft.rfft2(img) * ramp, s=(h, w)).astype(np.float32)
    return np.stack([np.fft.irfft2(np.fft.rfft2(img[..., c]) * ramp, s=(h, w)) for c in range(img.shape[2])], -1).astype(np.float32)


def _luma(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _tile_fit(ga: np.ndarray, gb: np.ndarray, skip: np.ndarray, t: int) -> np.ndarray | None:
    """The similarity taking gb's pixels to ga's, from tile shifts (phase correlation) where `skip` is clear; the
    worst-fitting tiles dropped. None with fewer than 2 tiles."""
    h, w = ga.shape
    step = max(8, t * 3 // 4)
    src, dst = [], []
    for y in range(0, h - t + 1, step):
        for x in range(0, w - t + 1, step):
            if skip[y:y + t, x:x + t].mean() > 0.02:
                continue
            ta, tb = ga[y:y + t, x:x + t], gb[y:y + t, x:x + t]
            if ta.std() < 0.5 or tb.std() < 0.5:
                continue
            dy, dx = phase_shift(ta - ta.mean(), tb - tb.mean())  # a(p) ~ b(p - d)
            cy, cx = y + t / 2, x + t / 2
            src.append((cx - dx, cy - dy))
            dst.append((cx, cy))
    if len(src) < 2:
        return None
    s, d = np.array(src), np.array(dst)
    # A similarity (pan, push-in, roll): x' = a x - b y + tx, y' = b x + a y + ty. Steadier than a full affine when the
    # person hides half the tiles.
    X = np.zeros((2 * len(s), 4))
    X[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(len(s)), np.zeros(len(s))])
    X[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(len(s)), np.ones(len(s))])
    keep = np.ones(len(s), bool)
    M = np.eye(3)
    for _ in range(3):
        k2 = np.repeat(keep, 2)
        sol, *_ = np.linalg.lstsq(X[k2], d.ravel()[k2], rcond=None)
        a, b, tx, ty = sol
        M = np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])
        res = np.hypot(*((X @ sol).reshape(-1, 2) - d).T)
        new = res <= max(0.5, 2.5 * float(np.median(res[keep])))
        if new.sum() < 2 or np.array_equal(new, keep):
            break
        keep = new
    return M


def frame_motion(a: np.ndarray, b: np.ndarray, ignore_a: np.ndarray, ignore_b: np.ndarray,
                 tile: int = MOTION_TILE) -> np.ndarray:
    """The similarity (3x3) that takes frame b's pixels to frame a's, from the shifts of many tiles of the set at half
    size, the tiles the person covers skipped. b is moved onto a and measured again twice, so the residual shifts
    are near zero, where phase correlation's sub-pixel estimate is unbiased (chained over 40 frames, a 0.2 px bias is
    8 px). A pan that also pushes in moves the edges of the frame more than its middle: one translation left the far
    side of the castle wall 20 px off after a dozen frames."""
    def prep(x: np.ndarray) -> np.ndarray:
        y = _luma(x)
        h2, w2 = y.shape[0] // 2 * 2, y.shape[1] // 2 * 2
        return y[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))  # half size without aliasing

    def hp(g: np.ndarray) -> np.ndarray:
        return g - composite.smooth(g, 3.0)

    ga, gb = prep(a), prep(b)
    h, w = ga.shape
    skip_a = composite.grow(ignore_a[:2 * h:2, :2 * w:2], 4)
    skip_b = composite.grow(ignore_b[:2 * h:2, :2 * w:2], 4).astype(np.float32)
    t = min(tile, max(16, min(h, w) // 2))
    S = np.diag([2.0, 2.0, 1.0])
    M = np.eye(3)  # half-size b -> half-size a
    ha = hp(ga)
    for _ in range(3):
        moved = composite.warp(gb, M, (w, h), cubic=True)
        cover = composite.warp(np.ones_like(gb), M, (w, h)) < 0.99
        skip = skip_a | (composite.warp(skip_b, M, (w, h)) > 0.01) | composite.grow(cover, 4)
        R = _tile_fit(ha, hp(moved), skip, t)
        if R is None:
            break
        M = R @ M
        if np.abs(R - np.eye(3)).max() < 1e-4:
            break
    return S @ M @ np.linalg.inv(S)


@dataclass
class Stack:
    """The frames (`frames`, uint8) and, per frame, the affine that takes it onto the middle one (`A`), its per-channel
    exposure gains to the middle one, and where it lends no pixel (`excl`: the person, a margin, their shadow).
    `al`/`valid` hold every frame moved onto the middle one, for the median."""
    frames: list[np.ndarray]
    A: list[np.ndarray]
    gains: list[list[float]]
    excl: list[np.ndarray]
    al: np.ndarray | None = None
    valid: np.ndarray | None = None
    cells: list = field(default_factory=list)  # per frame, (cell means, share known) on the middle frame, exposure-matched
    _plate: tuple | None = None
    plate_seen: np.ndarray | None = None  # where some frame showed the plate's background (median_of)

    def plate_cells(self, plate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._plate is None:
            self._plate = cells(plate, np.ones(plate.shape[:2], bool))
        return self._plate


def align_stack(frames: list[np.ndarray], cover: list[np.ndarray]) -> Stack:
    """Every frame onto the middle one (frame_motion, chained frame to frame), exposure-matched there. `cover` is where
    the person is."""
    n = len(frames)
    ref = n // 2
    H, W = frames[0].shape[:2]
    step = [np.eye(3)] + [frame_motion(frames[i - 1], frames[i], cover[i - 1], cover[i]) for i in range(1, n)]
    A = [np.eye(3) for _ in range(n)]
    for i in range(ref + 1, n):
        A[i] = A[i - 1] @ step[i]
    for i in range(ref - 1, -1, -1):
        A[i] = A[i + 1] @ np.linalg.inv(step[i + 1])
    al = np.empty((n, H, W, 3), np.float32)
    valid = np.empty((n, H, W), bool)
    ones = np.ones((H, W), np.float32)
    for i, f in enumerate(frames):
        al[i] = composite.warp(f, A[i], (W, H), cubic=True)
        # their feet go wider in the plate too: a far foot the track missed in most frames stood in the median
        clear = composite.warp((~(composite.grow(cover[i], 2) | feet_cover(cover[i]))).astype(np.float32), A[i], (W, H))
        valid[i] = (clear > 0.99) & (composite.warp(ones, A[i], (W, H)) > 0.99)
    gains = []
    for i in range(n):
        both = valid[i] & valid[ref]
        if int(both.sum()) < 16:
            gains.append([1.0, 1.0, 1.0])
            continue
        gi = [float(np.median(al[ref][..., c][both]) / max(1e-3, np.median(al[i][..., c][both]))) for c in range(3)]
        gains.append(gi)
        al[i] *= np.array(gi, np.float32)
    excl = [composite.grow(c, CLEAR_GROW) | feet_cover(c) for c in cover]
    # The drift is measured well clear of the person: cells that touch their edge read their dark coat as drift.
    far = [(composite.warp((~composite.grow(c, CLEAR_GROW + 4)).astype(np.float32), A[i], (W, H)) > 0.99) & valid[i]
           for i, c in enumerate(cover)]
    return Stack(frames, A, gains, excl, al, valid, [cells(al[i], far[i]) for i in range(n)])


def median_of(st: Stack) -> np.ndarray:
    """The per-pixel median of every frame where that pixel shows the background (the middle frame where none does),
    on the middle frame's pixels."""
    n, H = st.al.shape[:2]
    plate = st.al[n // 2].copy()
    st.plate_seen = np.zeros(plate.shape[:2], bool)
    for y in range(0, H, 64):
        v = st.valid[:, y:y + 64]
        if not v.any():
            continue
        blk = st.al[:, y:y + 64]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a pixel the person always covers has no samples
            got = np.nanmedian(np.where(v[..., None], blk, np.nan), axis=0)
        use = ~np.isnan(got[..., 0])
        plate[y:y + 64][use] = got[use]
        # one or two samples are not a background: where the person stood on most frames, a median of the few frames that
        # missed them still held their dark edge (T1.3: dark blocks along the bottom of the plate on every frame)
        st.plate_seen[y:y + 64] = use & (v.sum(axis=0) >= MIN_SEEN)
    return plate


def plate_in(st: Stack, plate: np.ndarray, i: int, win: tuple[slice, slice]) -> np.ndarray:
    """The plate as frame i sees it (its motion and exposure), over that frame's window `win`."""
    M = _offset(win) @ np.linalg.inv(st.A[i])
    p = _warp_window(plate, M, (win[1].stop - win[1].start, win[0].stop - win[0].start), cubic=True)
    return np.clip(p / np.array(st.gains[i], np.float32), 0, 255)


def _offset(win: tuple[slice, slice]) -> np.ndarray:
    T = np.eye(3)
    T[0, 2], T[1, 2] = -win[1].start, -win[0].start
    return T


def _warp_window(img: np.ndarray, M: np.ndarray, size: tuple[int, int], cubic: bool = False) -> np.ndarray:
    """composite.warp onto a small output, reading only the part of `img` that lands there."""
    w, h = size
    c = np.linalg.inv(M) @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], np.float64)
    c = c[:2] / c[2]
    H, W = img.shape[:2]
    x0, x1 = max(0, int(np.floor(c[0].min())) - 4), min(W, int(np.ceil(c[0].max())) + 4)
    y0, y1 = max(0, int(np.floor(c[1].min())) - 4), min(H, int(np.ceil(c[1].max())) + 4)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w) + img.shape[2:], np.float32)
    T = np.eye(3)
    T[0, 2], T[1, 2] = x0, y0
    return composite.warp(img[y0:y1, x0:x1], M @ T, size, cubic=cubic)


def replace_mask(cover: np.ndarray, src: np.ndarray, ref: np.ndarray, window: np.ndarray, redraw: bool = False) -> np.ndarray:
    """Where the background goes back on a frame: the person grown RESTORE_GROW px, their floor shadow (darker than the
    background `ref`) only in a band at their feet, and anything near them, touching that, that is clearly not the
    background (REMNANT_LEVELS from `ref`). Inside `window` (the crop). `ref` is the background as this frame would
    show it (fill_from_clear: the nearest clear frames, so the camera's parallax doesn't shift it).

    The darker-than-plate test alone, anywhere near them, also caught every crack and shadowed stone of the wall
    behind: a smooth plate patch, ten times the person's size, slid over the static wall with the character."""
    if not cover.any():
        return np.zeros_like(cover)
    kept = np.zeros_like(cover)
    rep = composite.grow(cover, RESTORE_GROW) | feet_cover(cover)
    ys = np.flatnonzero(cover.any(axis=1))
    tall = int(ys[-1] - ys[0] + 1)
    src_f = src.astype(np.float32)
    if redraw:  # their shadow goes (the new person's is drawn by LTX): darker than the background, on that ground
        dark = (_luma(composite.smooth(ref, 2)) - _luma(composite.smooth(src_f, 2))) > DARKER
        rep |= dark & redraw_zone(cover)
    elif cover.any():
        kept = cast_shadow(cover, src, ref, window)  # their cast shadow stays real footage, where it can be told from a remnant
        rep &= ~kept
    rep &= window  # otherwise their cast shadow is not replaced: `ref` carries it (shadow_ratio)
    # What they carry that the mask missed (a phone's lanyard, a strap, a strand of hair): pixels near them that are
    # clearly not the background and touch what is already replaced. Alone, it stayed as footage: a green lanyard
    # swung beside the new character's gown.
    near = composite.grow(cover, max(RESTORE_GROW, round(REMNANT_REACH * tall))) & window & ~rep
    if near.any():
        d = np.abs(composite.smooth(src_f, REMNANT_SMOOTH) - composite.smooth(ref, REMNANT_SMOOTH)).max(axis=-1)
        odd = near & (d > REMNANT_LEVELS) & ~kept
        rep |= connected(composite.grow(rep, 2) & odd, odd)
    return rep


def connected(seed: np.ndarray, allowed: np.ndarray, step: int = 4) -> np.ndarray:
    """The parts of `allowed` connected to `seed`, gaps up to about `step` px bridged (a beaded chain strap breaks into
    beads). Grown on step x step cells, however long the path."""
    h, w = allowed.shape
    hs, ws = -(-h // step), -(-w // step)

    def cells(m: np.ndarray) -> np.ndarray:
        pad = np.zeros((hs * step, ws * step), bool)
        pad[:h, :w] = m
        return pad.reshape(hs, step, ws, step).any(axis=(1, 3))

    a = cells(allowed)
    got = cells(seed) & a
    while True:
        more = got.copy()
        more[1:] |= got[:-1]
        more[:-1] |= got[1:]
        more[:, 1:] |= more[:, :-1].copy()
        more[:, :-1] |= more[:, 1:].copy()
        more &= a
        if np.array_equal(more, got):
            break
        got = more
    return np.repeat(np.repeat(got, step, 0), step, 1)[:h, :w] & allowed


def cells(img: np.ndarray, known: np.ndarray, cell: int = CELL) -> tuple[np.ndarray, np.ndarray]:
    """`img` (H, W, 3) averaged over its `known` pixels in cell x cell squares, and the share of each square known."""
    h, w = known.shape
    hc, wc = -(-h // cell), -(-w // cell)
    k = np.zeros((hc * cell, wc * cell), np.float32)
    k[:h, :w] = known
    v = np.zeros((hc * cell, wc * cell, 3), np.float32)
    v[:h, :w] = np.where(known[..., None], img, 0)
    ks = k.reshape(hc, cell, wc, cell).sum(axis=(1, 3))
    vs = v.reshape(hc, cell, wc, cell, 3).sum(axis=(1, 3))
    return vs / np.maximum(ks, 1)[..., None], ks / (cell * cell)


def spread(d: np.ndarray, ok: np.ndarray, sigma: float) -> np.ndarray:
    """A smooth field from `d` (h, w, 3) where `ok`, carried over the holes: a normalised blur of `sigma`, and wider
    ones where the narrow one finds nothing (a hole as wide as the person)."""
    out = np.zeros(d.shape, np.float32)
    weight = np.zeros(ok.shape, np.float32)
    k = ok.astype(np.float32)
    dk = d * k[..., None]
    for sig in (sigma, 4 * sigma, 16 * sigma):
        kw = composite.blur(k, sig)
        est = composite.blur(dk, sig) / np.maximum(kw[..., None], 1e-6)
        take = np.clip(kw / 0.05, 0, 1) * (1 - weight)
        out += est * take[..., None]
        weight += take
    return out / np.maximum(weight[..., None], 1e-6)


def cubic_at(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """img (H, W, C) read at the points (x, y), bicubic (Keys, a = -0.5, as PIL's): one resample of real footage."""
    H, W = img.shape[:2]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    tx, ty = (x - x0).astype(np.float32), (y - y0).astype(np.float32)

    def weights(t: np.ndarray) -> list[np.ndarray]:
        a = -0.5
        out = []
        for s in (t + 1, t, 1 - t, 2 - t):
            s = np.abs(s)
            out.append(np.where(s <= 1, ((a + 2) * s - (a + 3)) * s * s + 1, ((a * s - 5 * a) * s + 8 * a) * s - 4 * a))
        return out

    wx, wy = weights(tx), weights(ty)
    acc = np.zeros((len(x), img.shape[2]), np.float32)
    for m, wyy in zip(range(-1, 3), wy):
        yy = np.clip(y0 + m, 0, H - 1)
        for k, wxx in zip(range(-1, 3), wx):
            acc += img[yy, np.clip(x0 + k, 0, W - 1)].astype(np.float32) * (wyy * wxx)[:, None]
    return acc


def sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, cell: int = CELL) -> np.ndarray:
    """A cell field read at full-resolution points (xs, ys), bilinear between cell centres."""
    hc, wc = field.shape[:2]
    fy = np.clip((ys + 0.5) / cell - 0.5, 0, hc - 1)
    fx = np.clip((xs + 0.5) / cell - 0.5, 0, wc - 1)
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    y1, x1 = np.minimum(y0 + 1, hc - 1), np.minimum(x0 + 1, wc - 1)
    wy, wx = (fy - y0)[:, None], (fx - x0)[:, None]
    return ((field[y0, x0] * (1 - wx) + field[y0, x1] * wx) * (1 - wy) +
            (field[y1, x0] * (1 - wx) + field[y1, x1] * wx) * wy)


def fill_from_clear(st: Stack, plate: np.ndarray, i: int, area: np.ndarray, win: tuple[slice, slice],
                    need: np.ndarray | None = None, seed: int = 0, tol: float = 0,
                    trace: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame i's background over `win` where `area` (frame i's pixels, window-sized) is, from the frames nearest to it
    (i-1, i+1, i-2, ...) where that spot of the set was clear: each read where the camera's motion puts it there
    (bicubic, one resample) with that frame's brightness drift around the spot taken out (cell means on the middle
    frame, spread over the hole). The FILL_SAMPLES nearest are read; the nearest that agrees with their median within
    `tol` levels is used, else the median: one lender that still shows something of the person (a strap the mask
    missed) is outvoted, and nothing is compared with a plate the camera's parallax has shifted. Where no frame shows
    the spot, the plate, with the footage's grain (`need`). Returns (fill, window x 3 floats in frame i's exposure;
    where it took the plate)."""
    tol = tol or FILL_TOL
    n = len(st.frames)
    H, W = st.frames[0].shape[:2]
    plate_w = plate_in(st, plate, i, win)
    fill = plate_w.copy()
    took = np.zeros(area.shape, bool)
    ys, xs = np.nonzero(area)
    if not len(ys):
        return fill, took
    k = len(ys)
    q = st.A[i] @ np.stack([xs + win[1].start, ys + win[0].start, np.ones(k)])  # on the middle frame
    gi = np.array(st.gains[i], np.float32)
    mi, fi = st.cells[i]
    samples = np.zeros((FILL_SAMPLES, k, 3), np.float32)
    lender = np.full((FILL_SAMPLES, k), -1, np.int32)
    count = np.zeros(k, np.int64)
    todo = np.arange(k)
    for d in range(1, n):
        for j in (i - d, i + d):
            if not (0 <= j < n) or not len(todo):
                continue
            pj = np.linalg.inv(st.A[j]) @ q[:, todo]
            xj, yj = pj[0], pj[1]
            ok = (xj >= 1) & (xj <= W - 3) & (yj >= 1) & (yj <= H - 3)
            ok[ok] = ~st.excl[j][np.rint(yj[ok]).astype(int), np.rint(xj[ok]).astype(int)]
            if not ok.any():
                continue
            sel = todo[ok]
            mj, fj = st.cells[j]
            drift = np.clip(spread(mi - mj, (fi > 0.9) & (fj > 0.9), DRIFT_SIGMA / CELL), -DRIFT_MAX, DRIFT_MAX)
            raw = cubic_at(st.frames[j], xj[ok], yj[ok])
            samples[count[sel], sel] = (raw * np.array(st.gains[j], np.float32) + sample(drift, q[0, sel], q[1, sel])) / gi
            lender[count[sel], sel] = j
            count[sel] += 1
            todo = todo[count[todo] < FILL_SAMPLES]
        if not len(todo):
            break
    has = count > 0
    vals = np.zeros((k, 3), np.float32)
    if has.any():
        s = samples[:, has]
        c = count[has]
        valid = np.arange(FILL_SAMPLES)[:, None] < c[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(valid[..., None], s, np.nan), axis=0)
        agree = valid & (np.abs(s - med[None]).max(axis=-1) <= tol)
        # The nearest lender that agrees with the others; where none does (grass in the wind), the nearest itself:
        # always one real frame's pixel. A median of three frames' grass was a soft smudge that walked with her.
        first = np.where(agree.any(axis=0), np.argmax(agree, axis=0), 0)
        vals[has] = s[first, np.arange(s.shape[1])]
        if trace is not None:
            src = np.full(area.shape, -1, np.int32)
            src[ys[has], xs[has]] = lender[:, has][first, np.arange(s.shape[1])]
            trace["src"] = src
    fill[ys[has], xs[has]] = vals[has]
    if (~has).any():
        mp, _ = st.plate_cells(plate)
        drift = np.clip(spread(mi - mp, fi > 0.9, DRIFT_SIGMA / CELL), -DRIFT_MAX, DRIFT_MAX)
        kk = np.flatnonzero(~has)
        pv = plate_w[ys[kk], xs[kk]] + sample(drift, q[0, kk], q[1, kk]) / gi
        if need is not None:
            rng = np.random.default_rng(seed)
            pv = pv + rng.standard_normal((len(kk), 1)).astype(np.float32) * np.asarray(need, np.float32)[None, :]
        fill[ys[kk], xs[kk]] = pv
        took[ys[kk], xs[kk]] = True
    return fill, took


def _near(m: np.ndarray, pad: int) -> tuple[slice, slice]:
    """The box around a mask's pixels, `pad` px bigger each way, inside the image."""
    ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    h, w = m.shape
    return slice(max(0, ys[0] - pad), min(h, ys[-1] + pad + 1)), slice(max(0, xs[0] - pad), min(w, xs[-1] + pad + 1))


def feet_cover(cover: np.ndarray) -> np.ndarray:
    """The person's feet, grown FOOT_GROW of their height (only in the bottom FEET_BAND of them)."""
    ys = np.flatnonzero(cover.any(axis=1))
    if not len(ys):
        return np.zeros_like(cover)
    tall = int(ys[-1] - ys[0] + 1)
    feet = np.zeros_like(cover)
    feet[max(0, int(ys[-1] - FEET_BAND * tall)):] = True
    return composite.grow(cover & feet, max(RESTORE_GROW, round(FOOT_GROW * tall))) & feet


def shadow_zone(cover: np.ndarray) -> np.ndarray:
    """Where the person's floor shadow can be: the bottom FEET_BAND of their height, within SHADOW_REACH of them."""
    ys = np.flatnonzero(cover.any(axis=1))
    tall = int(ys[-1] - ys[0] + 1)
    feet = np.zeros_like(cover)
    feet[max(0, int(ys[-1] - FEET_BAND * tall)):] = True
    return feet & composite.grow(cover, max(RESTORE_GROW, round(SHADOW_REACH * tall)))


def light_ratio(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the frame's light over the background's per pixel, at most 1, and how steady it is over a few px)."""
    ls = _luma(composite.smooth(src.astype(np.float32), 2))
    lr = _luma(composite.smooth(np.clip(ref, 0, 255), 2))
    r = np.minimum((ls + 4) / (lr + 4), 1.0)
    mean = composite.smooth(r, 1.5)
    sd = np.sqrt(np.maximum(composite.smooth(r * r, 1.5) - mean * mean, 0))
    return r, sd


def cast_shadow(cover: np.ndarray, src: np.ndarray, ref: np.ndarray, window: np.ndarray) -> np.ndarray:
    """The person's cast shadow on the ground in the frame itself (real pixels worth keeping): steady, darker than the
    background but not black, at their feet, and running away from them (connected to a part beyond SHADOW_FAR of their
    height). A compact dark thing at the foot (a shoe the mask missed) is a remnant, not a shadow."""
    ys = np.flatnonzero(cover.any(axis=1))
    if not len(ys):
        return np.zeros_like(cover)
    tall = int(ys[-1] - ys[0] + 1)
    r, sd = light_ratio(src, ref)
    rows = np.arange(cover.shape[0])[:, None]
    band = rows >= (ys[-1] - SHADOW_BAND * tall)
    like = band & window & (r > SHADOW_FLOOR) & (r < 0.93) & (sd < SHADOW_STEADY) & ~composite.grow(cover, 3)
    like = composite.shrink(composite.grow(like, 2), 2)  # closes pinholes
    seed = like & ~composite.grow(cover, max(RESTORE_GROW + 8, round(SHADOW_FAR * tall)))
    return connected(seed, like, step=6) if seed.any() else np.zeros_like(cover)


def shadow_ratio(cover: np.ndarray, src: np.ndarray, ref: np.ndarray, window: np.ndarray) -> np.ndarray:
    """How much darker the frame is than the clean background `ref` in the person's cast shadow, as a light ratio (<= 1),
    over the frame: measured on the ground beside them (not on them), and carried in under them. 1 = no shadow.
    Multiplied onto `ref` where their pixels go back to the ground, so their shadow stays.

    A shadow is the ground's own texture times one number, so the ratio is only taken where it is steady over a few
    px (a shoe the mask missed, a remnant's edge, changes from pixel to pixel) and not near what is far darker."""
    ones = np.ones(cover.shape, np.float32)
    ys = np.flatnonzero(cover.any(axis=1))
    if not len(ys):
        return ones
    tall = int(ys[-1] - ys[0] + 1)
    rows = np.arange(cover.shape[0])[:, None]
    band = rows >= (ys[-1] - SHADOW_BAND * tall)
    reach = composite.grow(cover, max(RESTORE_GROW + 8, round(SHADOW_KEEP_REACH * tall)))
    ls = _luma(composite.smooth(src.astype(np.float32), 2))
    lr = _luma(composite.smooth(np.clip(ref, 0, 255), 2))
    r = np.minimum((ls + 4) / (lr + 4), 1.0)
    mean = composite.smooth(r, 1.5)
    sd = np.sqrt(np.maximum(composite.smooth(r * r, 1.5) - mean * mean, 0))
    dark = r <= SHADOW_FLOOR
    valid = band & reach & window & ~composite.grow(cover, 6) & ~composite.grow(dark, 4) & (sd < SHADOW_STEADY)
    if not valid.any():
        return ones
    v = valid.astype(np.float32)
    ext = np.ones(cover.shape, np.float32)
    got = np.zeros(cover.shape, np.float32)
    for sigma in (4.0, 12.0, max(24.0, 0.05 * tall)):  # the ratio close by first, a wider reach only where none is
        num, den = composite.blur(r * v, sigma), composite.blur(v, sigma)
        trust = np.clip(den / 0.1, 0, 1) * (1 - got)
        ext = ext * (1 - trust) + (num / np.maximum(den, 1e-4)) * trust
        got = got + trust
    ext = ext * got + (1 - got)  # no ratio near enough anywhere: none
    w = np.clip(composite.blur((band & window).astype(np.float32), 4.0), 0, 1)
    return np.minimum(1.0 - (1.0 - ext) * w, 1.0).astype(np.float32)


def redraw_zone(cover: np.ndarray) -> np.ndarray:
    """The ground where the old person's shadow can fall, which is taken out and given to LTX to redraw."""
    ys = np.flatnonzero(cover.any(axis=1))
    if not len(ys):
        return np.zeros_like(cover)
    tall = int(ys[-1] - ys[0] + 1)
    low = np.arange(cover.shape[0])[:, None] >= (ys[-1] - REDRAW_BAND * tall)
    return low & composite.grow(cover, max(RESTORE_GROW, round(REDRAW_REACH * tall)))


SHADOW_SEEN_PX, SHADOW_SEEN_SHARE = 400, 0.2


def has_shadow(keep: list[np.ndarray]) -> bool:
    """Did the old person cast a shadow worth keeping or redrawing? (The light ratio measured round them is under 0.9 over a
    real area on at least SHADOW_SEEN_SHARE of the frames.) An overcast shot has none: redrawing a shadow there only added
    32-px blocks of regenerated ground, with hard edges, round the feet."""
    if not keep:
        return False
    return float(np.mean([(k < 0.9).sum() >= SHADOW_SEEN_PX for k in keep])) >= SHADOW_SEEN_SHARE


HARD_LIGHT = "The light is harsh midday sun: deep crisp shadows on them, bright specular highlights, strong contrast."
HARD_NEGATIVE = ", soft light, overcast, flat light, low contrast"
HARD_PUNCH = 1.2  # contrast boost on the new subject for `lighting: hard`
DIM_LUMA = 0.10  # the plate's median luminance (linear) under which a scene is dim: the castle hall 0.065, a dusk trail 0.04
LIGHT_NEGATIVE = ", studio lighting, cinematic lighting, softbox, fill light, bright even light, overexposed"
TILT_MIN = 0.08  # a subject lit this much more on one side than the other is lit from that side
TILT_WRONG = 0.05  # and the painted one must not lean this much the other way


def lateral_tilt(img: np.ndarray, mask: np.ndarray) -> float | None:
    """How much brighter the right half of a subject is than the left, as a share of its mean (linear light); None when
    there is too little of it. Positive: lit from camera right."""
    if int(mask.sum()) < 2000:
        return None
    ys, xs = np.nonzero(mask)
    mid = (int(xs.min()) + int(xs.max())) // 2
    lin = envblend.to_lin(img)
    y = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    left, right = y[:, :mid][mask[:, :mid]], y[:, mid:][mask[:, mid:]]
    if not len(left) or not len(right):
        return None
    return float((right.mean() - left.mean()) / max(float(y[mask].mean()), 1e-6))


def key_light_words(crops_: list[np.ndarray], covers_: list[np.ndarray], bg_: list[np.ndarray], keep_: list[np.ndarray]) -> str:
    """The scene's key light in plain words, from the plate and the original person's shading: hard or soft, which side,
    warm or cool. Said to the painter and to LTX, so the new subject is lit right at the source."""
    idx = [i for i in range(0, len(covers_), max(1, len(covers_) // 8)) if covers_[i].any()]
    hard = has_shadow(keep_)
    px = np.concatenate([bg_[i][~composite.grow(covers_[i], 8)][::37] for i in idx]).astype(np.float32) if idx else None
    warm, dim = "", False
    if px is not None and len(px):
        rb = float(px[:, 0].mean() / max(float(px[:, 2].mean()), 1.0))
        warm = "warm " if rb > 1.12 else ("cool " if rb < 0.92 else "")
        lin = envblend.to_lin(px[None].astype(np.uint8))[0]
        dim = float(np.median(lin[:, 0] * 0.2126 + lin[:, 1] * 0.7152 + lin[:, 2] * 0.0722)) < DIM_LUMA
    # (No side: a subject's left-right luminance is its clothes more than the light, so the side is not said.)
    if dim:
        return (f"The scene is dim and underexposed, lit only by natural {warm}available light, soft and from one side, with deep "
                "shadows and no fill light: the character is as dark as the people in this shot, not brighter.")
    if hard:
        return f"The light is hard {warm}direct sun, with cast shadows on the ground."
    return f"The light is soft {warm}overcast light, with no cast shadows on the ground."


EXPOSURE_MAX = 0.6  # a painted subject's median luminance over the ground round them: the approved paints are 0.15-0.32


def exposure_ratio(img: np.ndarray, person: np.ndarray, scene: np.ndarray) -> float | None:
    """Median luminance (linear) of the subject in `img` over that of the ground ring round them in `scene`."""
    if int(person.sum()) < 2000:
        return None
    ring = composite.grow(person, 60) & ~composite.grow(person, 8)
    if int(ring.sum()) < 2000:
        return None
    def y(a):
        lin = envblend.to_lin(a)
        return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    return float(np.median(y(img)[person]) / max(float(np.median(y(scene)[ring])), 1e-6))


def exposure_problem(painted: np.ndarray, person: np.ndarray, scene: np.ndarray) -> str:
    """A painted subject far brighter than the scene it stands in (studio light on a dim hall)."""
    r = exposure_ratio(painted, person, scene)
    return "too bright for the scene" if r is not None and r > EXPOSURE_MAX else ""


def light_problem(painted: np.ndarray, person: np.ndarray, orig_tilt: float | None) -> str:
    """A painted subject lit from the other side than the original person was (a paint-gate criterion)."""
    if orig_tilt is None or abs(orig_tilt) < TILT_MIN:
        return ""
    got = lateral_tilt(painted, person)
    if got is None:
        return ""
    return "lit from the wrong side" if got * orig_tilt < 0 and abs(got) > TILT_WRONG else ""


def best_person(people_dir: Path, old: np.ndarray) -> np.ndarray | None:
    """The painted person's mask (the one overlapping the old person most) at the old mask's size, None when none does."""
    from pickups.tasks import recast
    masks = [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(people_dir.glob("*.png"))]
    masks = [recast.upsize(m, (old.shape[1], old.shape[0])) for m in masks]
    best = max(masks, key=lambda m: (m & old).sum(), default=None)
    return best if best is not None and (best & old).any() else None


def shape_match(people_dir: Path, old: np.ndarray) -> float:
    """IoU of the old person's mask and the painted person's (the painted mask that overlaps it most), 0 when none."""
    from pickups.tasks import recast
    masks = [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(people_dir.glob("*.png"))]
    masks = [recast.upsize(m, (old.shape[1], old.shape[0])) for m in masks]
    best = max(masks, key=lambda m: (m & old).sum(), default=None)
    if best is None or not (best & old).any():
        return 0.0
    return float((best & old).sum() / (best | old).sum())


def restore_frames(st: Stack, cover: list[np.ndarray], window: list[np.ndarray], plate: np.ndarray,
                   need: np.ndarray | None = None, seed: int = 0, nearest: bool = True,
                   unsure: list[np.ndarray] | None = None, redraw: bool = False,
                   keep: list[np.ndarray] | None = None, debug: dict | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each frame with the background put back only where the person was (replace_mask), feathered RESTORE_FEATHER px.

    The background comes from the nearest frames where each spot was clear (fill_from_clear): real footage with its
    own texture and grain, and also what replace_mask compares the frame with (their shadow, what the mask missed).
    Where no frame shows it (or `nearest` is off), the plate, given the footage's grain (`need`, per channel; a median
    has none). Returns the frames and how much each pixel took from elsewhere (0..1). A frame lends nothing within
    CLEAR_GROW px of the person, in their shadow's zone, or where it had to replace something itself (`st.excl`)."""
    frames = st.frames
    feather = 2 * round(RESTORE_FEATHER)
    wins = []
    for i in range(len(frames)):
        if not cover[i].any():
            wins.append(None)
            continue
        ys = np.flatnonzero(cover[i].any(axis=1))
        tall = int(ys[-1] - ys[0] + 1)
        reach = max(RESTORE_GROW, round(max(SHADOW_REACH, REMNANT_REACH) * tall))
        wins.append(_near(cover[i], reach + 2 * feather + 4))
        st.excl[i] |= shadow_zone(cover[i])
    def background(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(the background frame i would show over its window, what goes back there, where it was measured, where no
        frame showed it and the plate stood in)."""
        win = wins[i]
        cw, sw, ww = cover[i][win], frames[i][win], window[i][win]
        ys = np.flatnonzero(cw.any(axis=1))
        tall = int(ys[-1] - ys[0] + 1)
        cand = composite.grow(composite.grow(cw, max(RESTORE_GROW, round(REMNANT_REACH * tall))) | shadow_zone(cw), feather) & ww
        if nearest:
            tr = {} if debug is not None and i in debug else None
            ref, took = fill_from_clear(st, plate, i, cand, win, need=need, seed=seed + i, trace=tr)
            if tr is not None:
                debug[i] = {"src": tr.get("src"), "win": win, "took": took}
        else:
            took = np.zeros_like(cw)
            ref = plate_in(st, plate, i, win)
            if need is not None:
                ref = composite.grain(np.clip(np.rint(ref), 0, 255).astype(np.uint8), need, seed=seed + i).astype(np.float32)
        ratio = np.ones(cw.shape, np.float32) if redraw else shadow_ratio(cw, sw, ref, ww)
        ref = ref * ratio[..., None]
        return ref, replace_mask(cw, sw, ref, ww, redraw), cand, took, ratio

    if nearest:
        # What each frame's mask missed (a strap) lends nothing either: found on a first pass, then the background is
        # read again without it. Late in the castle-hall walk the strap crossed the same stones for frames on end and
        # its three nearest lenders all still showed a few of its beads.
        for i in range(len(frames)):
            if wins[i] is not None:
                _, rep, _, _, _ = background(i)
                st.excl[i][wins[i]] |= composite.grow(rep, 3)
    out, alphas = [], []
    for i, src in enumerate(frames):
        o = src.copy()
        a = np.zeros(cover[i].shape, np.float32)
        out.append(o)
        alphas.append(a)
        if unsure is not None:
            unsure.append(np.zeros(cover[i].shape, bool))
        if keep is not None:
            keep.append(np.ones(cover[i].shape, np.float32))
        if wins[i] is None:
            continue
        win = wins[i]
        ref, rep, cand, took, ratio = background(i)
        if keep is not None:
            kp = np.ones(cover[i].shape, np.float32)
            kp[win] = ratio
            keep[-1] = kp
        if not rep.any():
            continue
        if unsure is not None and st.plate_seen is not None:
            # only where the plate never saw the background either (it is the person there): a clean plate is
            # real footage, the render's ground is not
            seen = _warp_window(st.plate_seen.astype(np.float32), _offset(win) @ np.linalg.inv(st.A[i]),
                                (win[1].stop - win[1].start, win[0].stop - win[0].start)) > 0.5
            unsure[i][win] = rep & took & ~seen
        if unsure is not None:
            # and wherever what came back is the person themselves (lenders a camera following them can't place
            # right near its lens, the plate's few samples): the same as the frame inside their own outline
            rs, ss = composite.smooth(ref, 2), composite.smooth(src[win].astype(np.float32), 2)
            inside = composite.shrink(cover[i][win], 2)
            same = inside & (np.abs(rs - ss).max(axis=-1) < SAME_LEVELS)
            # or the person a few px off (a lender a little out of place near the lens), where the frame itself
            # shows something else: the old leg's black beside the new skirt's hem
            looks = np.zeros_like(same)
            for dy, dx in SAME_OFFSETS:
                sh = np.roll(ss, (dy, dx), axis=(0, 1))
                looks |= np.roll(inside, (dy, dx), axis=(0, 1)) & (np.abs(rs - sh).max(axis=-1) < SAME_LEVELS)
            looks &= np.abs(rs - ss).max(axis=-1) > 2 * SAME_LEVELS
            unsure[i][win] |= composite.grow(composite.shrink(same | looks, 2), 6) & rep
        aw = np.clip(composite.blur(rep.astype(np.float32), RESTORE_FEATHER) * 2.0, 0, 1) * composite.grow(rep, feather)
        aw[aw < 0.01] = 0
        aw *= cand | rep
        a[win] = aw
        o[win] = _mix(src[win], np.clip(ref, 0, 255), aw)
    return out, alphas


def fill_holes(img: np.ndarray, hole: np.ndarray, shift: int = 0) -> np.ndarray:
    """`img` with `hole` filled from around it. Where no frame showed the background, the plate is the person (a
    camera following them): LTX copied her black trousers from it under the new character's skirt.

    With `shift`, the ground beside the hole moved across by that many px (one shift for the whole shot, hole_shift,
    so it doesn't jump from frame to frame), where that is known: real texture for LTX to carry on. A smooth fill
    alone it copied as a grey patch between the new character's feet. Elsewhere, and eased in at the hole's edge, the
    smooth fill (pull-push: halved with the known pixels' weights until the hole is gone, the coarse levels poured
    back in)."""
    if not hole.any():
        return img
    # What lies just outside the hole is often still the person's edge (a dark fringe): the smooth fill ignores it.
    outside = ~composite.grow(hole, FILL_MARGIN)
    w = outside.astype(np.float32)
    levels = [(img.astype(np.float32) * w[..., None], w)]
    halved = []  # which axes each level halved (an axis stops at one cell, so a hole as tall as the picture still closes)
    while (levels[-1][1] == 0).any() and max(levels[-1][1].shape) > 1:
        c, cw_ = levels[-1]
        hy, hx = cw_.shape[0] > 1, cw_.shape[1] > 1
        c = np.pad(c, ((0, c.shape[0] % 2 * hy), (0, c.shape[1] % 2 * hx), (0, 0)))
        cw_ = np.pad(cw_, ((0, cw_.shape[0] % 2 * hy), (0, cw_.shape[1] % 2 * hx)))
        fy, fx = 2 if hy else 1, 2 if hx else 1
        c = c.reshape(c.shape[0] // fy, fy, c.shape[1] // fx, fx, 3).sum(axis=(1, 3))
        cw_ = cw_.reshape(cw_.shape[0] // fy, fy, cw_.shape[1] // fx, fx).sum(axis=(1, 3))
        levels.append((c, cw_))
        halved.append((fy, fx))
    c, cw_ = levels[-1]
    up = c / np.maximum(cw_, 1e-6)[..., None]
    for (c, cw_), (fy, fx) in zip(reversed(levels[:-1]), reversed(halved)):
        h, w_ = cw_.shape
        big = np.repeat(np.repeat(up, fy, axis=0), fx, axis=1)
        big = np.pad(big, ((0, max(0, h - big.shape[0])), (0, max(0, w_ - big.shape[1])), (0, 0)), mode="edge")[:h, :w_]
        big = composite.smooth(big, 1.0)  # not blur: its zero outside the picture darkened the fill at the bottom edge
        known = np.clip(cw_, 0, 1)[..., None]
        up = c / np.maximum(cw_, 1e-6)[..., None] * known + big * (1 - known)
    # The smooth fill is flat, and LTX copies a flat patch as a grey patch. Give it the ground's own fine texture: the
    # high-pass of the ground beside the hole, mirrored across the hole's edge on each row.
    base = np.where(outside[..., None], img.astype(np.float32), up)
    texture = mirror_texture(base - composite.smooth(base, 2.5), hole, ~composite.grow(hole, 2))
    fill = up + texture
    up = fill.copy()
    if shift:
        todo = hole.copy()
        for d in (shift, -shift):
            ys, xs = np.nonzero(todo)
            xsrc = xs - d
            ok = (xsrc >= 0) & (xsrc < hole.shape[1])
            ok[ok] &= ~hole[ys[ok], xsrc[ok]]
            # a copy that is far from the smooth fill is not ground: it is the old person's dark edge (T1.3: a dark block
            # with straight sides at the left of the hem on every frame of f80-84)
            ok[ok] &= np.abs(img[ys[ok], xsrc[ok]].astype(np.float32) - up[ys[ok], xs[ok]]).max(axis=-1) < FILL_COPY_TOL
            fill[ys[ok], xs[ok]] = img[ys[ok], xsrc[ok]]
            todo[ys[ok], xs[ok]] = False
        a = np.clip((composite.smooth(hole.astype(np.float32), 3.0) - 0.5) * 2.0, 0, 1)[..., None]
        fill = up * (1 - a) + fill * a
    out = img.copy()
    out[hole] = np.clip(np.rint(fill[hole]), 0, 255).astype(img.dtype)
    return out


def mirror_texture(hp: np.ndarray, hole: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """`hp` (a high-pass image) in the hole, taken from the known pixels beside it: each pixel of a run in a row is
    mirrored across the nearer end of the run (the far end where that falls outside the picture or on unknown
    pixels). Where neither is known the texture is 0."""
    tex = np.zeros_like(hp)
    wd = hole.shape[1]
    for y in np.flatnonzero(hole.any(axis=1)):
        edge = np.diff(np.concatenate([[0], hole[y].astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
            x = np.arange(a, b)
            near_left = x < (a + b) / 2
            first = np.where(near_left, 2 * a - 1 - x, 2 * b - 1 - x)
            second = np.where(near_left, 2 * b - 1 - x, 2 * a - 1 - x)
            src = np.full(x.shape, -1)
            for cand in (first, second):
                good = (src < 0) & (cand >= 0) & (cand < wd)
                good[good] &= ok[y, cand[good]]
                src[good] = cand[good]
            has = src >= 0
            tex[y, x[has]] = hp[y, src[has]]
    return tex


def hole_shift(holes: list[np.ndarray], gap: int = 6) -> int:
    """The one sideways shift fill_holes copies ground by for a whole shot: past the widest hole."""
    widths = [int(np.ptp(np.flatnonzero(h.any(axis=0)))) + 1 for h in holes if h.any()]
    return max(widths) + gap if widths else 0


def _mix(src: np.ndarray, p: np.ndarray, a: np.ndarray) -> np.ndarray:
    bg = np.clip(np.rint(src.astype(np.float32) * (1 - a[..., None]) + p * a[..., None]), 0, 255).astype(np.uint8)
    bg[a == 0] = src[a == 0]
    return bg


# --- the mask LTX is allowed to change --------------------------------------------------------------------------------


def blockify(m: np.ndarray, block: int = BLOCK) -> np.ndarray:
    h, w = m.shape
    hb, wb = -(-h // block), -(-w // block)
    pad = np.zeros((hb * block, wb * block), bool)
    pad[:h, :w] = m
    b = pad.reshape(hb, block, wb, block).any(axis=(1, 3))
    return np.repeat(np.repeat(b, block, 0), block, 1)[:h, :w]


def region_masks(person: list[np.ndarray], grow_px: int, temporal: int = TEMPORAL) -> list[np.ndarray]:
    """The person, grown, united with the neighbouring frames, then blocked so the mask sits on latent squares."""
    grown = [composite.grow(m, grow_px) for m in person]
    out = []
    for i in range(len(grown)):
        u = np.logical_or.reduce(grown[max(0, i - temporal):i + temporal + 1])
        out.append(blockify(u))
    return out


def pixel_region(lat: list[np.ndarray], n: int) -> list[np.ndarray]:
    """The latent masks back to one per picture frame: frame 0 is kept, frames 8k-7..8k belong to latent frame k."""
    return [lat[0] if f == 0 else lat[min(-(-f // 8), len(lat) - 1)] for f in range(n)]


def plate_lock(render: np.ndarray, plate: np.ndarray, region: np.ndarray, band: int = PLATE_BAND) -> np.ndarray:
    """Outside the mask LTX may change, plus a `band`-px ramp, the picture is the real plate's own pixels, not decoded
    latents: the VAE round trip had softened the grass and moved the levels by about 7 on the whole frame. Inside the
    mask it is the render. (Pipeline v2, R1.)"""
    if not region.any():
        return plate.copy()
    soft = composite.blur(composite.grow(region, band // 2).astype(np.float32), max(band / 4, 1.0))
    a = np.maximum(region.astype(np.float32), np.clip(soft, 0, 1))
    a = np.where(composite.grow(region, band), a, 0.0)[..., None]
    return np.clip(np.rint(render.astype(np.float32) * a + plate.astype(np.float32) * (1 - a)), 0, 255).astype(np.uint8)


def latent_masks(region: list[np.ndarray]) -> list[np.ndarray]:
    """One mask per latent frame. The first is empty, so the painted frame is kept. Each later one covers 8 frames."""
    h, w = region[0].shape
    n = len(region)
    lat = [np.zeros((h, w), bool)]
    for k in range((n - 1) // 8):
        lat.append(np.logical_or.reduce(region[1 + 8 * k:9 + 8 * k]))
    return lat


# --- the matte ---------------------------------------------------------------------------------------------------------


def guided(guide: np.ndarray, p: np.ndarray, r: int = MATTE_R, eps: float = MATTE_EPS) -> np.ndarray:
    """Snap a soft matte's edge to the image's edges (grey guided filter)."""
    I = guide.astype(np.float32).mean(axis=2) / 255.0
    mI, mp = composite._box(composite._box(I, r, 0), r, 1), composite._box(composite._box(p, r, 0), r, 1)
    cov = composite._box(composite._box(I * p, r, 0), r, 1) - mI * mp
    var = composite._box(composite._box(I * I, r, 0), r, 1) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return composite._box(composite._box(a, r, 0), r, 1) * I + composite._box(composite._box(b, r, 0), r, 1)


def character_matte(track: np.ndarray, char: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The guided filter's own soft alpha (radius MATTE_R, eps MATTE_EPS) from the lightly smoothed, 3-frame-smoothed
    track, kept inside the render region. No 0.5 threshold and re-blur: that turned the SAM mask's steps, upscaled from
    the proxy, into a lumpy edge that crawled from frame to frame."""
    soft = track.astype(np.float32)
    if soft.shape != region.shape:
        soft = np.asarray(Image.fromarray((np.clip(soft, 0, 1) * 255).astype(np.uint8)).resize(region.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255
    soft = composite.smooth(soft, 1.0)
    al = np.clip(guided(char, soft), 0, 1)
    al = np.clip((al - MATTE_LO) / (MATTE_HI - MATTE_LO), 0, 1)
    # Never past the track: where the ground matches the hem's colour the guided filter carried the alpha onto the
    # render's own ground, a soft dark patch that walked with her.
    al = np.minimum(al, composite.blur(composite.grow(soft > 0.5, MATTE_REACH).astype(np.float32), 1.5))
    al *= composite.blur(composite.shrink(region, 1).astype(np.float32), 1.0) * region
    al[al < 0.02] = 0
    return al


def decontaminate(char: np.ndarray, alpha: np.ndarray, bg: np.ndarray | None = None, px: float = DECON_PX) -> np.ndarray:
    """The character's own colour at its edge, without the render's background in it. Where the edge is mostly them
    (alpha UNMIX_ALPHA or more) and the background is known, the background's share is taken out: (C - (1 - a) B) / a.
    A fainter edge (hair, a motion blur) takes the colour of the character just inside it. The opaque RIM_PX just
    inside the soft edge takes it too where it is RIM_LIFT levels lighter than inside: LTX draws a thin light line down
    a dark cloak's edge. A dark outline stays.

    Copying the inside's colour onto every soft-edge pixel drew a light line down a dark sleeve: the nearest inside
    there was a lit hand."""
    opaque = alpha >= 0.95
    inner = composite.shrink(opaque, 1 + RIM_PX)
    if not (alpha > 0).any():
        return char
    f = char.astype(np.float32)
    out = f.copy()
    soft = (alpha > 0) & ~opaque
    mostly = soft & (alpha >= UNMIX_ALPHA) if bg is not None else np.zeros_like(soft)
    if mostly.any():
        a = alpha[mostly][:, None]
        out[mostly] = np.clip((f[mostly] - (1 - a) * bg.astype(np.float32)[mostly]) / a, 0, 255)
    if inner.any():
        sig = px + 0.5 + RIM_PX
        k = composite.blur(inner.astype(np.float32), sig)
        est = composite.blur(f * inner[..., None], sig) / np.maximum(k[..., None], 1e-6)
        reach = np.clip(k / 0.05, 0, 1)
        faint = soft & ~mostly
        rim = opaque & ~inner
        lift = np.clip((_luma(f) - _luma(est) - RIM_LIFT) / RIM_LIFT, 0, 1)
        w = (reach * (faint + rim * lift))[..., None]
        out = out * (1 - w) + est * w
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


EDGE_HOLD = 0.012  # of the render's height: the track is carried out to a picture edge it comes this close to


def edges_of(box: tuple[int, int], cw: int, ch: int, W: int, H: int) -> tuple[bool, bool, bool, bool]:
    """Which of the crop's sides (top, bottom, left, right) are the picture's edges."""
    x, y = box
    return y <= 0, y + ch >= H, x <= 0, x + cw >= W


def hold_edges(t: np.ndarray, edges: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """The render's track carried out to the picture's edges: SAM stops a row or two short of one, and the matte
    closed there over the old footage with a light rim, a line under a skirt the frame cuts off."""
    k = max(2, round(EDGE_HOLD * t.shape[0]))
    t = t.copy()
    top, bottom, left, right = edges
    if bottom:
        t[-k:] = np.maximum(t[-k:], t[-k - 1])
    if top:
        t[:k] = np.maximum(t[:k], t[k])
    if left:
        t[:, :k] = np.maximum(t[:, :k], t[:, k:k + 1])
    if right:
        t[:, -k:] = np.maximum(t[:, -k:], t[:, -k - 1:-k])
    return t


BORDER_ROWS = 4  # crop px at a picture edge where the render's own last rows are replaced by the one inside them


def hold_border(a: np.ndarray, edges: tuple[bool, bool, bool, bool], k: int = BORDER_ROWS) -> np.ndarray:
    """`a` with its last k rows or columns at each picture edge (top, bottom, left, right) copied from the one inside."""
    a = a.copy()
    top, bottom, left, right = edges
    if bottom:
        a[-k:] = a[-k - 1]
    if top:
        a[:k] = a[k]
    if left:
        a[:, :k] = a[:, k:k + 1]
    if right:
        a[:, -k:] = a[:, -k - 1:-k]
    return a


def temporal_median(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Each mask replaced by the median of itself and its two neighbours, so a one-frame flicker drops out."""
    if not masks:
        return []
    tr = np.stack([m.astype(np.float32) for m in masks])
    prev = np.concatenate([tr[:1], tr[:-1]])
    nxt = np.concatenate([tr[1:], tr[-1:]])
    return list(np.median(np.stack([prev, tr, nxt]), axis=0))


def contact_shadow(render: np.ndarray, plate: np.ndarray, hard: np.ndarray, region: np.ndarray, matte: np.ndarray,
                   shadow_max: float = SHADOW_MAX) -> np.ndarray:
    """A darkening of the plate near the feet, from how much darker the render is than the plate it was given. 1 = none."""
    ch, cw = matte.shape
    rh, rw = render.shape[:2]
    pr = np.asarray(Image.fromarray(plate).resize((rw, rh), Image.BILINEAR))
    ratio = (composite.smooth(_luma(render), SHADOW_SMOOTH) + 4) / (composite.smooth(_luma(pr), SHADOW_SMOOTH) + 4)
    ratio = np.asarray(Image.fromarray(np.clip(ratio * 127.5, 0, 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)).astype(np.float32) / 127.5
    ys = np.nonzero(hard.any(axis=1))[0]
    band = np.zeros((ch, cw), np.float32)
    near = np.zeros((ch, cw), np.float32)
    if len(ys):
        tall = ys.max() - ys.min() + 1
        wide = shadow_max > SHADOW_MAX  # LTX draws their shadow: it reaches further than a contact shadow
        top = int(ys.max() - (REDRAW_BAND if wide else 0.12) * tall)
        band[max(0, top):, :] = 1.0
        hard_c = _resize_mask(hard, cw, ch)
        # The shadow falls beside and below their feet, not across the whole band.
        feet = hard_c.copy()
        feet[:max(0, int((top / rh) * ch))] = False
        near = composite.blur(composite.grow(feet, max(4, round((REDRAW_REACH if wide else 0.03) * tall * ch / rh))).astype(np.float32), 6)
    inner = composite.shrink(region, 4).astype(np.float32)
    w = composite.blur(band * inner, 4) * near * (1 - matte)
    sh = 1 - np.clip(1 - ratio, 0, shadow_max) * w
    if len(ys):
        sh = np.minimum(sh, 1 - AO_MAX * contact_ao(matte, region, int(ys.max()), int(tall)))
    sh[sh > 0.985] = 1.0
    return sh


def contact_ao(matte: np.ndarray, region: np.ndarray, bottom: int, tall: int) -> np.ndarray:
    """A soft occlusion, 0..1, on the ground just outside the character's feet (the lowest AO_BAND of their height), a
    little more underneath: where a foot meets the ground there is always some dark. Luminance only; the caller scales it."""
    h, w = matte.shape
    rows = np.arange(h)[:, None]
    feet = (matte > 0.5) & (rows >= bottom - AO_BAND * tall)
    if not feet.any():
        return np.zeros((h, w), np.float32)
    down = max(1, round(0.004 * tall))
    f = feet.astype(np.float32)
    f = np.maximum(f, np.roll(f, down, axis=0) * (rows >= down))
    ao = np.clip(composite.blur(f, max(2.0, AO_SIGMA * tall)) * 2.0, 0, 1)
    ground = (rows >= bottom - 2 * AO_BAND * tall) & region
    return (ao * ground * (1 - np.clip(matte, 0, 1))).astype(np.float32)


def tone_ratio(bg: np.ndarray, char: np.ndarray, valid: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """The ratio (per channel, 0.6-1.6) of the plate's local mean to the render's, measured on `valid` ground only and spread
    smoothly, so the render's ground can be given the plate's tone. 1 where no ground is known near."""
    v = valid.astype(np.float32)
    den = composite.smooth(v, sigma)
    mb = composite.smooth(bg.astype(np.float32) * v[..., None], sigma) / np.maximum(den, 1e-4)[..., None]
    mc = composite.smooth(char.astype(np.float32) * v[..., None], sigma) / np.maximum(den, 1e-4)[..., None]
    ratio = np.clip((mb + 2.0) / (mc + 2.0), 0.6, 1.6)
    known = np.clip(den / 0.05, 0, 1)[..., None]
    return 1 + (ratio - 1) * known


FEET_BAND_G = 0.14  # of the character's height: the feet zone where the render's own floor may replace the plate's
FEET_REACH_G = 0.12  # of their height: how far round the feet
FEET_AGREE = 22.0  # levels: the render's floor (tone-matched) must agree with the plate's at low frequency to be used
TRAIL_SOFT = (0.55, 0.9)  # in the feet band, behind the motion, alpha under the first is cut and under the second ramped


def feet_ground(bg: np.ndarray, char: np.ndarray, matte: np.ndarray, need: np.ndarray | None, seed: int) -> np.ndarray:
    """`bg` with the render's own floor in the feet zone: the plate there is a resampled mix of other frames, softer than the real
    floor, and LTX drew the floor beside the feet crisp (the castle hall's pale ghosts under the shoe). The render's floor is
    graded to the plate's tone (tone_ratio), given the footage's grain, and used only where it agrees with the plate at low
    frequency (so it cannot move the floor's joints); outside the zone nothing changes."""
    solid = matte > 0.5
    ys = np.flatnonzero(solid.any(axis=1))
    if not len(ys):
        return bg
    tall = int(ys[-1] - ys[0] + 1)
    rows = np.arange(bg.shape[0])[:, None]
    zone = (rows >= ys[-1] - FEET_BAND_G * tall) & composite.grow(solid, max(8, round(FEET_REACH_G * tall)))
    w = composite.blur(zone.astype(np.float32), 6.0) * (1 - np.clip(matte, 0, 1))
    if float(w.max()) < 0.02:
        return bg
    ground = char.astype(np.float32) * tone_ratio(bg, char, (matte < 0.05) & ~composite.grow(solid, 6))
    if need is not None:
        ground = composite.grain(np.clip(np.rint(ground), 0, 255).astype(np.uint8), need, seed=seed).astype(np.float32)
    diff = np.abs(composite.smooth(ground, 6.0) - composite.smooth(bg.astype(np.float32), 6.0)).max(axis=-1)
    agree = np.clip(1.0 - np.maximum(diff - FEET_AGREE * 0.5, 0) / (FEET_AGREE * 0.5), 0, 1)
    w = (w * agree)[..., None]
    return np.clip(np.rint(bg.astype(np.float32) * (1 - w) + ground * w), 0, 255).astype(np.uint8)


def trim_trail(matte: np.ndarray, motion: tuple[float, float] | None) -> np.ndarray:
    """The matte without its motion-blur trail: in the feet band, behind the foot's motion, the semi-transparent pixels (a pale
    shoe ghost on the floor) are cut (alpha under TRAIL_SOFT[0]) or ramped (up to TRAIL_SOFT[1]); the sharp core stays."""
    if motion is None or (abs(motion[0]) + abs(motion[1])) < 1.0:
        return matte
    solid = matte > 0.5
    ys = np.flatnonzero(solid.any(axis=1))
    if not len(ys):
        return matte
    tall = int(ys[-1] - ys[0] + 1)
    rows = np.arange(matte.shape[0])[:, None]
    band = rows >= ys[-1] - FEET_BAND_G * tall
    if not (solid & band).any():
        return matte
    cy, cx = np.nonzero(solid & band)
    yy, xx = np.mgrid[0:matte.shape[0], 0:matte.shape[1]]
    behind = ((xx - cx.mean()) * motion[0] + (yy - cy.mean()) * motion[1]) < 0
    lo, hi = TRAIL_SOFT
    cut = np.clip((matte - lo) / (hi - lo), 0, 1)
    return np.where(band & behind & (matte < hi), cut, matte).astype(np.float32)


def foot_motions(tracks: list[np.ndarray], cw: int, ch: int) -> list[tuple[float, float] | None]:
    """Per frame, the motion (px per frame, in crop pixels) of the character's feet: the centroid of the lowest FEET_BAND_G of
    their track, between the frame before and the one after."""
    cen: list[tuple[float, float] | None] = []
    for t in tracks:
        m = _resize_mask(t > 0.5, cw, ch) if t.any() else np.zeros((ch, cw), bool)
        ys = np.flatnonzero(m.any(axis=1))
        if not len(ys):
            cen.append(None)
            continue
        band = m.copy()
        band[:int(ys[-1] - FEET_BAND_G * (ys[-1] - ys[0] + 1))] = False
        yy, xx = np.nonzero(band)
        cen.append((float(xx.mean()), float(yy.mean())))
    out = []
    for i in range(len(cen)):
        a, b = cen[max(i - 1, 0)], cen[min(i + 1, len(cen) - 1)]
        out.append(None if a is None or b is None else ((b[0] - a[0]) / 2, (b[1] - a[1]) / 2))
    return out


def composite_crop(bg: np.ndarray, char: np.ndarray, matte: np.ndarray, shadow: np.ndarray | None) -> np.ndarray:
    """bg where the matte is 0 and there is no shadow; the character where the matte is 1."""
    base = bg.astype(np.float32) if shadow is None else bg.astype(np.float32) * shadow[..., None]
    m = matte[..., None]
    return np.clip(np.rint(base * (1 - m) + char.astype(np.float32) * m), 0, 255).astype(np.uint8)


def blend_frame(bg: np.ndarray, char: np.ndarray, render: np.ndarray, track: np.ndarray, region: np.ndarray,
                others: np.ndarray | None, need: np.ndarray | None, seed: int, unsure: np.ndarray | None = None,
                edges: tuple[bool, bool, bool, bool] | None = None, keep: np.ndarray | None = None,
                shadow_max: float = SHADOW_MAX, target: dict | None = None,
                env: tuple[dict | None, np.ndarray | None] | None = None, punch: float = 1.0,
                feet_floor: bool = False, foot_motion: tuple[float, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One crop's composite: the character (`char`, the render at the crop's size) over the restored background `bg`,
    through its matte (from `track`, the 3-frame-smoothed SAM track at the render's size), never over anyone else
    (`others`), with its contact shadow and the footage's grain (`need`). Where no frame of the shot showed the
    background (`unsure`: the camera follows them, so the plate there is them), the render's own picture is used:
    the plate put the old person's black trousers back under the new one's skirt. With `target` (the original subject's statistics,
    `composite.subject_stats`) the subject's softness and grain are matched to it from the footage's own noise (C3), and the
    plate's light wraps round its edge (C1). Returns (composite, matte, shadow)."""
    rh, rw = render.shape[:2]
    ch, cw = bg.shape[:2]
    hard = composite.shrink(track > 0.5, 1) if track.any() else np.zeros((rh, rw), bool)
    soft = track
    if soft.shape != (ch, cw):
        soft = np.asarray(Image.fromarray((np.clip(soft, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)).astype(np.float32) / 255
    matte = character_matte(soft, char, region)
    if edges is not None and any(edges):  # the render's last rows at a picture edge are lighter (the VAE's border)
        char, matte = hold_border(char, edges), hold_border(matte, edges)
    if others is not None and others.any():
        matte = matte * (1 - composite.blur(others.astype(np.float32), 1.5))
    matte = trim_trail(matte, foot_motion)
    shadow = contact_shadow(render, _resize(bg, rw, rh), hard, _resize_mask(region, rw, rh),
                            np.asarray(Image.fromarray((matte * 255).astype(np.uint8)).resize((rw, rh), Image.BILINEAR)).astype(np.float32) / 255,
                            shadow_max)
    shadow = _resize_float(shadow, cw, ch)
    char = decontaminate(char, matte, bg)
    if punch != 1.0:  # the light ratio pushed up: contrast about the subject's own median, in linear light
        char = envblend.apply(char, matte, {"gain": np.ones(3, np.float32), "lift": 0.0, "tint": np.ones(3, np.float32), "contrast": punch}, None, ("contrast",))
    if env is not None:  # the plate's grade, the original's light and the plate's contrast, in linear light
        char = envblend.apply(char, matte, env[0], env[1])
    if keep is not None and (keep < 0.999).any():
        # Where the render's own ground is used (nobody showed it), the old person's cast shadow goes on it too.
        char = np.clip(np.rint(char.astype(np.float32) * (1 - (1 - keep) * (1 - matte))[..., None]), 0, 255).astype(np.uint8)
    if target is not None:
        inner = composite.shrink(matte > 0.95, 6)
        if int(inner.sum()) > 500:
            char = composite.match_subject(char, inner, target, bg, seed, limit=0.0)  # C3: grain and softness of the footage
        elif need is not None:
            char = composite.grain(char, need, seed=seed)
        char = np.clip(np.rint(composite.light_wrap(char, matte, bg)), 0, 255).astype(np.uint8)  # C1
    elif need is not None:
        char = composite.grain(char, need, seed=seed)
    if unsure is not None and unsure.any():
        u = np.clip(composite.blur(composite.grow(unsure & region, UNSURE_GROW).astype(np.float32), UNSURE_FEATHER)
                    * 2.0, 0, 1) * region
        # The render's picture of the ground goes in only close to the character (UNSURE_NEAR of their height round them): a
        # whole dilated region of LTX's ground made the floor round the feet warp from frame to frame. Beyond it the plate's
        # fill stays, the same on every frame.
        ys_ = np.flatnonzero((matte > 0.5).any(axis=1))
        if len(ys_):
            reach = max(8, round(UNSURE_NEAR * (ys_[-1] - ys_[0] + 1)))
            u = u * composite.blur(composite.grow(matte > 0.5, reach).astype(np.float32), max(3.0, reach / 4))
        # The render's ground is graded to the plate's own fill beside it (low frequencies only): LTX's floor came out lighter than
        # the footage's, with pale ghosts of the old person's shoes, round the castle hall's feet.
        ground = u * (1 - np.clip(matte, 0, 1))
        if float(ground.max()) > 0.02:
            ratio = tone_ratio(bg, char, (matte < 0.05) & ~composite.grow(matte > 0.5, 6))
            char = np.clip(np.rint(char.astype(np.float32) * (1 + ground[..., None] * (ratio - 1))), 0, 255).astype(np.uint8)
        matte = np.maximum(matte, u.astype(np.float32))
    if feet_floor:  # the render's crisp floor beside the feet instead of the plate's soft one (grain is already on `char`)
        bg = feet_ground(bg, char, matte, None, seed)
    return composite_crop(bg, char, matte, shadow), matte, shadow


# --- the job -----------------------------------------------------------------------------------------------------------


def environment_pass(p: dict, renders, chars, bg, tracks, region, crops, covers, ch: int, cw: int):
    """The per-frame environment blend parameters (envblend), measured before the blend loop so they can be smoothed over
    time: the render's grade against the plate's on the ring round the new subject, and the original person's lateral light."""
    raw, prof = [], []
    for i in range(len(chars)):
        m = _resize_mask(tracks[i] > 0.5, cw, ch) if tracks[i].any() else np.zeros((ch, cw), bool)
        raw.append(envblend.frame_params(chars[i], bg[i], m, region[i]) if m.any() else None)
        prof.append(envblend.shade_profile(crops[i], covers[i]))
    return envblend.smooth_params(raw), envblend.smooth_profiles(prof)


def _rgb_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """An intermediate LTX or SDPose reads as pixels. crf 4 keeps the mask and the pose sticks, as the prototype did."""
    h, w = frames[0].shape[:2]
    wtr = common.FrameWriter(path, w, h, fps, kind="h264", crf=4)
    for f in frames:
        img = f if f.ndim == 3 else np.dstack([f.astype(np.uint8)] * 3)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[0] != h or img.shape[1] != w:
            img = np.asarray(Image.fromarray(img).resize((w, h), Image.NEAREST))
        wtr.write(img)
    wtr.close()


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))


def require_pose_weights(ctx) -> None:
    """Stop before any plate or paint work when ComfyUI does not have the pose checkpoint."""
    conn = getattr(ctx, "conn", None)
    if conn is None or not hasattr(conn, "model_files"):
        return
    if graphs.SDPOSE_CKPT not in conn.model_files():
        raise RuntimeError(
            "SDPose's weights (sdpose_wholebody_fp16.safetensors) need to be in ComfyUI's checkpoints before a person can be replaced.")


def run(ctx, s: Shot) -> dict:
    from pickups.tasks import recast_ltx

    require_pose_weights(ctx)
    p, run_dir, seed = s.p, s.run_dir, s.seed
    W, H, n = s.size[0], s.size[1], s.frames
    full_frames = [f for i, f in enumerate(common.read_frames(s.full)) if i < n]
    if len(full_frames) < n:
        n = len(full_frames)
    person = [recast.upsize(m, (W, H)) if m.shape != (H, W) else m for m in s.pm[:n]]
    om = list(s.om[:n]) if s.om else []
    # Cut off by the frame at the window's start but whole at its end (walking away from a camera close behind): the
    # window is rendered backwards from its last frame, so the character is painted where they are whole. Painted cut
    # off, the frame editor drew a small whole figure, or turned them to the camera, on every try.
    reverse = backwards(person)
    if reverse:
        full_frames, person, om = full_frames[::-1], person[::-1], om[::-1]
    boxes, cw, ch = crop_plan(person, W, H)
    (run_dir / "crop.json").write_text(json.dumps({"boxes": boxes, "cw": cw, "ch": ch}))  # for a 1:1 review of the take
    grow_px = max(4, round(GROW_PX * ch / GROW_AT))
    crops = [f[y:y + ch, x:x + cw] for f, (x, y) in zip(full_frames, boxes)]
    covers = [m[y:y + ch, x:x + cw] for m, (x, y) in zip(person, boxes)]
    # The plate is built in one fixed window that holds every crop, so a moving crop still has one background.
    x0 = min(x for x, _ in boxes)
    y0 = min(y for _, y in boxes)
    x1 = max(x for x, _ in boxes) + cw
    y1 = max(y for _, y in boxes) + ch
    roi_frames = [f[y0:y1, x0:x1] for f in full_frames]
    roi_cover = [m[y0:y1, x0:x1] for m in person]
    ctx.stage(PLATE)
    stack = align_stack(roi_frames, roi_cover)
    plate = median_of(stack)
    stack.al = stack.valid = None  # the median's working set, 3-4 GB on a 4K walk
    region = region_masks(covers, grow_px)
    windows = []
    for x, y in boxes:
        canvas = np.zeros((y1 - y0, x1 - x0), bool)
        canvas[y - y0:y - y0 + ch, x - x0:x - x0 + cw] = True
        windows.append(canvas)
    picks = sorted({0, n // 2, n - 1})
    plate_grain = np.sqrt(np.clip(composite.grain_profile([crops[i] for i in picks]) ** 2 -
                                  composite.grain_profile([np.clip(np.rint(plate), 0, 255).astype(np.uint8)]) ** 2, 0, None))
    def build(redraw: bool):
        """The restored crops, where no frame showed the ground, what the render's picture fills, the old shadow kept as a
        light ratio (`redraw` False), and the holes filled for LTX."""
        unsure_roi: list[np.ndarray] = []
        keep_roi: list[np.ndarray] = []
        restored, plate_alpha = restore_frames(stack, roi_cover, windows, plate,
                                               need=plate_grain if np.any(plate_grain > 0.05) else None, seed=seed,
                                               unsure=unsure_roi, redraw=redraw, keep=keep_roi)
        crop_of = lambda arrs: [a[y - y0:y - y0 + ch, x - x0:x - x0 + cw] for a, (x, y) in zip(arrs, boxes)]  # noqa: E731
        bg_ = crop_of(restored)
        unsure_ = crop_of(unsure_roi)
        holes = [composite.grow(u, UNSURE_GROW) for u in unsure_]
        shift = hole_shift(holes)
        bg_ = [fill_holes(b, h, shift) for b, h in zip(bg_, holes)]
        return bg_, unsure_, [a > 0 for a in crop_of(plate_alpha)], crop_of(keep_roi)

    # The old person's cast shadow is kept when the new silhouette is about the old one's, and otherwise redrawn (below).
    mode = str(p.get("shadow") or "auto").lower()
    bg, unsure, borrowed, keep = build(redraw=(mode == "redraw"))

    ctx.stage(SHEET)
    character, p, what, views, made_still = recast_ltx.character_picture(ctx, p, run_dir, vehicle=False)
    rw, rh = render_size(cw, ch, recast_ltx.gpu_vram(ctx))
    ctx.stage(POSE)
    pose_clip = run_dir / "pose_in.mp4"
    _rgb_video(pose_clip, [_resize(f, rw, rh) for f in crops], s.fps)
    pose_job = ctx.run(graphs.sdpose(ctx.client.assets.from_file(pose_clip), prefix="pickups/person_pose"))
    pose_dir = run_dir / "pose"
    pose_dir.mkdir(exist_ok=True)
    for o in pose_job.get_outputs("PU_Pose"):
        o.to_file(pose_dir / o.name)
    pose_pngs = sorted(pose_dir.glob("*.png"))
    if len(pose_pngs) < n:
        raise RuntimeError("The pose map didn't come back. SDPose's weights (sdpose_wholebody_fp16.safetensors) need to be in ComfyUI's checkpoints.")
    pose_frames = [_resize(np.asarray(Image.open(f).convert("RGB")), rw, rh) for f in pose_pngs[:n]]

    # Which way they face, from the pose of the footage's first frame (shoulders and hips, the face's side), not from
    # a "face" detection: a helmet or a cap seen from behind answered to "face".
    seen = pose_facing(pose_frames[0])
    ctx.stage(PAINT)
    who = (p.get("describe") or "").strip().rstrip(".")
    motion = recast_ltx.motion_words(s.pm)
    if reverse:  # LTX sees the walk backwards: they come towards the camera, back first
        motion = "walks backwards towards the camera, seen from behind, with their back to the camera"
    # The masked render keeps its first frame, so the character is painted onto the window's first frame.
    instruction = (p.get("look_prompt") or "").strip() or recast_ltx.paint_instruction(
        recast_ltx.person_noun(p.get("text", "")), recast_ltx.where_words(covers[0]), motion.startswith("walks"),
        views > 1, who, recast_ltx.person_phrase(p.get("text", "")))
    hard_light = str(p.get("lighting", "auto")).lower() == "hard"
    light = HARD_LIGHT if hard_light else key_light_words(crops, covers, bg, keep)
    # (The light is said to LTX, not to the painter: told the light, Qwen relit the whole picture and the paint gate
    # rejected three paints on T1.3 as "someone else was changed".)
    anchor_png = run_dir / "anchor.png"
    Image.fromarray(crops[0]).save(anchor_png)
    refs = [ctx.client.assets.from_file(character)]
    view = ""
    if not (p.get("look_prompt") or "").strip():
        # Which way they face (a sheet's view, from behind, in profile walking across): said to the painter, and
        # checked on what it paints. Told nothing, it turned a woman walking across a castle hall to the camera.
        view, instruction, cut = recast_ltx.paint_view(ctx, anchor_png, covers[0], person[::-1] if reverse else person,
                                                       character, views, instruction, run_dir, pose_view=seen)
        if cut is not None:
            refs.append(ctx.client.assets.from_file(cut))
    # Cut off by the bottom of the picture (walking away from a camera close behind), the painter put a small whole
    # figure in the middle of the path on every try, told or not. It paints on the picture carried on below the edge
    # (the ground, mirrored from the clean plate), with their legs to draw down to it; the top is kept.
    ext = 0
    paint_old = covers[0]
    bottom_cut = cut_by_bottom(covers[0])
    if not (p.get("look_prompt") or "").strip() and bottom_cut and "bottom" in recast_ltx.cut_words(covers[0]).split("cut them off")[0]:
        ext = below_edge(covers[0], _resize(pose_frames[0], cw, ch))
        Image.fromarray(extend_down(crops[0], bg[0], ext)).save(anchor_png)
        # the check asks the painted person to cover the part of the old one in the picture; below it is new
        paint_old = np.vstack([covers[0], np.zeros((ext, cw), bool)])
        instruction = instruction.replace(recast_ltx.KEEP_REST, LEGS_BELOW + recast_ltx.KEEP_REST)
    else:
        cutoff = recast_ltx.cut_words(covers[0])
        if cutoff and not bottom_cut and recast_ltx.cut_words(covers[0]).split("cut them off")[0].strip().endswith("The bottom edge of the picture"):
            cutoff = ""  # only a shoe tip touches the bottom edge
        if cutoff:
            instruction = instruction.replace(recast_ltx.KEEP_REST, cutoff + recast_ltx.KEEP_REST)
    safe = ctx.commercial_safe()
    painted = run_dir / "painted.png"
    gate: list[str] = []
    for attempt in range(recast_ltx.PAINT_TRIES):
        job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(anchor_png), instruction, seed=seed + attempt,
                                        refs=refs, resolution=0, prefix="pickups/recast_paint", commercial_safe=safe,
                                        size=graphs.edit_size(cw, ch + ext)))
        try_png = run_dir / f"painted_{attempt}.png"
        job.get_outputs("PU_Plate")[0].to_file(try_png)
        ppose = painted_pose(ctx, try_png, rw, rh, s.fps, run_dir / f"gate_{attempt}")
        hands = None  # the arms and hands the pose draws, where the held thing must be
        if ppose is not None:
            hands = _resize_mask(composite.grow(_pose_px(ppose, PARTS["arms and hands"]), 6), cw, ch + ext)
        problem = recast_ltx.paint_problem(ctx, try_png, paint_old, "", f"{who} {p.get('prompt') or ''}",
                                           run_dir / f"gate_{attempt}", anchor=anchor_png, hands=hands)
        if not problem and ext:
            problem = painted_small(run_dir / f"gate_{attempt}" / "people", paint_old, ch, ext)
        want = seen if not (p.get("look_prompt") or "").strip() else ""
        if not problem and want:
            got = painted_facing(ctx, try_png, rw, rh, s.fps, run_dir / f"gate_{attempt}", pose=ppose)
            if got and not same_facing(want, got):
                problem = f"facing the wrong way ({got})"
        # (A paint-gate check on the side it is lit from was tried and dropped: a subject's left-right luminance is mostly
        # its clothes, not the light. It rejected three clean paints of T1.3.)
        if not problem:
            best = best_person(run_dir / f"gate_{attempt}" / "people", paint_old)
            if best is not None:
                shown = _resize(np.asarray(Image.open(try_png).convert("RGB")), cw, ch + ext)[:ch]
                problem = exposure_problem(shown, best[:ch], crops[0])
        gate.append(problem or "ok")
        if not problem:
            break
    if problem:
        # A paint that failed its check is never used: LTX follows it for the whole shot.
        raise RuntimeError(f"The character couldn't be painted into the first frame the right way ({'; '.join(gate)}). "
                           "Try another picture of them, or another part of the shot.")
    paint = _resize(np.asarray(Image.open(try_png).convert("RGB")), cw, ch + ext)[:ch]
    iou = shape_match(run_dir / f"gate_{attempt}" / "people", paint_old)
    relight_qa: dict = {}
    if wants_relight(p):
        ctx.check()
        paint, relight_qa = relight(ctx, paint, try_png, anchor_png, run_dir / f"gate_{attempt}" / "people", paint_old, crops[0],
                                    bg[0], covers[0], seed, graphs.edit_size(cw, ch + ext), run_dir)
    shadowed = has_shadow(keep)
    # Redrawing the shadow (LTX draws a new one inside a widened mask) was tried on the castle hall and on T1.3: a dark smeared band
    # across the floor, and hard 32-px blocks of regenerated ground. So the old shadow is kept unless a job asks for "redraw".
    redraw = mode == "redraw"
    stack = None  # free the working set
    if redraw:
        region = [r | redraw_zone(c) for r, c in zip(region, covers)]
    reach = composite.soft_matte(composite.grow(covers[0], grow_px), 1, 2.0)[..., None]
    anchor = np.clip(np.rint(bg[0].astype(np.float32) * (1 - reach) + paint.astype(np.float32) * reach), 0, 255).astype(np.uint8)
    Image.fromarray(anchor).save(painted)
    base = [anchor] + bg[1:]

    ctx.stage(RENDER)
    prompt = (p.get("prompt") or "").strip() or f'{recast_ltx.ltx_prompt(who, motion, p.get("scene") or "")} {light}'
    small_region = [_resize_mask(m, rw, rh) for m in region]
    lat = latent_masks(small_region)
    base_v, lat_v, pose_v = run_dir / "base.mp4", run_dir / "latmask.mp4", run_dir / "pose.mp4"
    _rgb_video(base_v, [_resize(f, rw, rh) for f in base], s.fps)
    _rgb_video(lat_v, [(m.astype(np.uint8) * 255) for m in lat], s.fps)
    _rgb_video(pose_v, pose_frames, s.fps)
    region_px = pixel_region(lat, n)
    region_v = run_dir / "region.mp4"
    _rgb_video(region_v, [(m.astype(np.uint8) * 255) for m in region_px], s.fps)
    job = ctx.run(graphs.masked_pose(ctx.client.assets.from_file(base_v), ctx.client.assets.from_file(lat_v),
                                     ctx.client.assets.from_file(pose_v), prompt, s.fps, seed=seed, frames=n,
                                     prefix="pickups/person_pose", region=ctx.client.assets.from_file(region_v),
                                     negative=graphs.DEFAULT_NEGATIVE + LIGHT_NEGATIVE + (HARD_NEGATIVE if hard_light else "")))
    gdir = run_dir / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Blend"):
        o.to_file(gdir / o.name)
    gen = sorted(gdir.glob("*.png"))[:n]
    if not gen:
        raise RuntimeError("LTX-2.5 didn't send the shot back. Try again.")
    # R1: beyond the mask the real plate's pixels, not the decoded latents.
    locked = []
    for i, f in enumerate(gen):
        locked.append(plate_lock(np.asarray(Image.open(f).convert("RGB")), _resize(base[i], rw, rh), region_px[i]))
        Image.fromarray(locked[-1]).save(f)
    gen_small = run_dir / "gen.mp4"
    sw, sh = media.work_size(rw, rh, common.PROXY_LONG)
    _rgb_video(gen_small, [_resize(x, sw, sh) for x in locked], s.fps)

    ctx.stage(BLEND)
    track, seeded = recast_ltx.find_new(ctx, gen_small, 0, _resize_mask(covers[0], sw, sh), run_dir)
    tracks = temporal_median([recast.upsize(track[i], (rw, rh)) if i < len(track) else np.zeros((rh, rw), bool) for i in range(n)])
    tracks = [hold_edges(t, edges_of(boxes[i], cw, ch, W, H)) for i, t in enumerate(tracks)]
    chars = [_resize(np.asarray(Image.open(f).convert("RGB")), cw, ch) for f in gen]
    renders = [np.asarray(Image.open(f).convert("RGB")) for f in gen]
    grain_on = str(p.get("grain", "fine")).lower() != "off"
    need = None
    if grain_on:
        picks = sorted({0, n // 2, n - 1})
        need = np.sqrt(np.clip(composite.grain_profile([crops[i] for i in picks]) ** 2 - composite.grain_profile([chars[i] for i in picks]) ** 2, 0, None))
        need = need if np.any(need > 0.05) else None
    # C3/C1: the original subject is the reference for the new one's softness and grain, and the plate's light wraps its edge.
    refs = composite.smooth_targets([composite.subject_stats(crops[i], composite.shrink(covers[i], 6)) for i in range(n)]) \
        if grain_on else None
    env_params, env_profiles = environment_pass(p, renders, chars, bg, tracks, region, crops, covers, ch, cw) if str(p.get("envblend", "off")).lower() != "off" else (None, None)
    others = [recast.upsize(m, (W, H)) if m.shape != (H, W) else m for m in om]
    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, s.fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, s.fps, kind="h264", scale_long=1920)
    held: list[tuple[np.ndarray, np.ndarray]] = []  # rendered backwards: written in the shot's order at the end
    changed = touched_px = borrowed_px = 0
    pose_parts = [part_masks(pf) for pf in pose_frames]
    shares: list[dict[str, float]] = []
    # Both are opt-in: on the castle hall the render's floor in the feet zone smeared grey clouds round the shoe, and the trail
    # trim only thinned the pale ghost a little.
    feet_floor_on = bool(p.get("feet_floor"))
    foot_mot = foot_motions(tracks, cw, ch) if p.get("feet_trim") else None
    for i in range(n):
        ctx.check()
        block = others[i][boxes[i][1]:boxes[i][1] + ch, boxes[i][0]:boxes[i][0] + cw] if i < len(others) else None
        comp, matte, shadow = blend_frame(bg[i], chars[i], renders[i], tracks[i], region[i], block, need, 1000 + i,
                                          unsure=unsure[i], edges=edges_of(boxes[i], cw, ch, W, H),
                                          keep=None if redraw else keep[i], shadow_max=REDRAW_MAX if redraw else SHADOW_MAX,
                                          target=refs[i] if refs else None,
                                          env=(env_params[i], env_profiles[i]) if env_params else None,
                                          punch=HARD_PUNCH if hard_light else 1.0,
                                          feet_floor=feet_floor_on, foot_motion=foot_mot[i] if foot_mot else None)
        shares.append(part_shares(pose_parts[i], _resize_float(matte, rw, rh)))
        frame = full_frames[i].copy()
        x, y = boxes[i]
        # Outside the crop the footage is the work frame. Inside, only the matte, the shadow and the plate put back
        # where the person was may differ from the footage.
        untouched = (matte == 0) & (shadow >= 0.999) & ~borrowed[i]
        changed += int(np.count_nonzero(np.any(comp[untouched] != crops[i][untouched], axis=-1)))
        frame[y:y + ch, x:x + cw] = comp
        touched = np.zeros((H, W), bool)
        touched[y:y + ch, x:x + cw] = matte > 0
        touched_px += int(touched.sum())
        borrowed_px += int(np.count_nonzero(borrowed[i] & (matte == 0)))
        tint = frame.copy()
        tint[touched] = (tint[touched].astype(np.float32) * 0.45 + np.array(recast.AI_TINT, np.float32) * 0.55).astype(np.uint8)
        if reverse:
            held.append((frame, tint))
        else:
            out_w.write(frame)
            rec_w.write(tint)
        ctx.progress((i + 1) / n)
    for frame, tint in reversed(held):
        out_w.write(frame)
        rec_w.write(tint)
    out_w.close()
    rec_w.close()
    missing = missing_parts(shares)
    if missing:
        raise RuntimeError("The new person is missing part of their body in this take (" +
                           ", ".join(f"{k}: {round(100 * v)}% of it is in the picture" for k, v in missing.items()) +
                           "). Try again, or another picture of them.")

    ctx.stage(WRITE)
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=s.full)
    make_viewer(s.full, ctx.vdir / "before.mp4")
    (ctx.vdir / "painted_frame.png").write_bytes(painted.read_bytes())
    total = W * H * max(n, 1)
    receipt = {"original": round(100 * (1 - (touched_px + borrowed_px) / total), 2),
               "borrowed": round(100 * borrowed_px / total, 2), "ai": round(100 * touched_px / total, 2)}
    made = media.made_from(s.src, s.s0, n)
    models = [graphs.edit_model(safe), *([graphs.edit_model(True)] if relight_qa.get("relight") == "on" and not safe else []),
              "LTX-2.5 + Union Control IC-LoRA", "SDPose", "SAM 3.1"] + (["FLUX 3 klein"] if made_still else [])
    one_to_one = (rw, rh) == (cw, ch)
    return {"label": "Replace a person", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, n, s.fps), "engine": "ltx", "performance_size": f"{rw}×{rh}",
                        "reference": what, "pixels_changed_outside_selection": changed, "models": models,
                        **({"made_from": made} if made else {}),
                        "grain": "matched to your shot" if grain_on else "off",
                        "look_prompt": instruction, "prompt": prompt,
                        "_qa": {"engine": "person-pose", "window_start": s.s0,
                                "window_chosen": "yours" if s.start is not None else "auto",
                                "anchor_frame": n - 1 if reverse else 0, "rendered_backwards": reverse, "guide": "pose",
                                "new_seed": seeded, **({"relight": relight_qa} if relight_qa else {}), "paint_gate": gate, "facing": view or None, "pose_facing": seen or None,
                                "crop": [boxes[0][0], boxes[0][1], cw, ch], "work_size": [rw, rh],
                                "one_to_one": one_to_one, "reference": what, "views": views,
                                "searched_frames": s.n_search,
                                "shadow": "redrawn by LTX" if redraw else ("kept from the footage" if shadowed else "none cast"),
                                "shape_iou": round(iou, 3),
                                "body_parts": {k: round(float(np.mean([x[k] for x in shares if k in x])), 2)
                                               for k in {k for x in shares for k in x}}}}}


def wants_relight(p: dict) -> bool:
    """The optional Relight step (`relight` true on the params; off by default)."""
    v = p.get("relight")
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "on")


def relight(ctx, paint: np.ndarray, painted_png: Path, anchor_png: Path, people_dir: Path, paint_old: np.ndarray,
            src: np.ndarray, plate: np.ndarray, cover: np.ndarray, seed: int, size: tuple[int, int],
            out_dir: Path) -> tuple[np.ndarray, dict]:
    """Relight the painted first frame: Qwen-Image-Edit 2511 relights it with the footage's first frame as the light
    reference, and only its low-frequency shading is blended (70%) into the paint, inside the painted person's tight mask,
    with the subject's brightness set to the original person's ratio to the plate (measured on this shot). The paint
    keeps all fine detail. When the painted person can't be found the paint is returned as it is."""
    ch, cw = paint.shape[:2]
    best = best_person(people_dir, paint_old)
    if best is None:
        return paint, {"relight": "skipped: no painted person found"}
    mask = best[:ch]
    job = ctx.run(graphs.relight(ctx.client.assets.from_file(painted_png), ctx.client.assets.from_file(anchor_png), seed=seed,
                                 size=size))
    relit_png = out_dir / "relit.png"
    job.get_outputs("PU_Plate")[0].to_file(relit_png)
    relit = _resize(np.asarray(Image.open(relit_png).convert("RGB")), cw, paint_old.shape[0])[:ch]
    out, info = composite.relight_paint(paint, relit, mask, src, plate, cover)
    return out, {"relight": "on", **info}


def _resize_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    if m.shape == (h, w):
        return m
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127


def _resize_float(a: np.ndarray, w: int, h: int) -> np.ndarray:
    if a.shape == (h, w):
        return a
    return np.asarray(Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255
