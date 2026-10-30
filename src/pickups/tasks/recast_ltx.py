"""Recast on LTX-2.5, the default engine.

A person goes through tasks/person_swap: a median plate, a 1:1 crop that follows them, their pose, and a masked LTX
render. The character is painted onto the crop's first frame, which is not denoised. Outside the matte and its contact
shadow the plate stays, and outside the crop the footage stays. `guide` "pose" opts into that path; the default is the older one, which paints the
frame where the person is biggest and steers the render with the shot's depth.

1. Window. LTX renders up to common.MAX_FRAMES frames (8k+1). On a longer shot SAM 3.1 follows the person through the
   first search_frames() and the window goes where they are on screen most (or from params.start_frame).
2. Anchor. On the depth path, the frame on LTX's 8-frame grid where the person is biggest and least cut off.
3. Character. A cast member's character sheet when they have one; otherwise the picture given (a photo, a still or a
   sheet, as it is: the frame editor reads every view of a sheet), or a FLUX 3 klein still made from the description.
4. Paint. The frame editor (graphs.image_edit: Qwen-Image 2.1, or Qwen-Image-Edit 2511 under Setup's commercial licence
   choice) puts the character in the person's place, in the scene's own light. The person path paints the window's
   first frame. The depth path paints the anchor.
5. Render. The person path is a single masked pass guided by the pose (graphs.masked_pose). The depth path re-renders
   a crop around everywhere the person goes, steered by Depth Anything 3, with the painted frame pinned at the anchor.
6. Composite. SAM 3.1 finds the new character and follows them. The person path chokes that matte and keeps a contact
   shadow; the depth path feathers the old person plus the new character. Delivered at the source's size (up to 4K).

A vehicle ("the SUV", or a click SAM 3.1 calls a vehicle) shares steps 1-2, then goes its own way (tasks/vehicle_swap):
the old one is removed for a clean plate, the new one drives its path as a stand-in, and Layout-to-Render renders that
with no depth of the old car to pull the new body back to its shape.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite, graphs, media, previz_facts, reference
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups import features
from pickups.tasks import common, person_swap, recast, vehicle_swap

STAGES = [  # measured on an RTX 4090: 97 frames of a 4K shot (seconds)
    Stage("Preparing the shot", 15),
    Stage("Finding the person", 35),
    Stage("Reading the character sheet", 10),
    Stage("Painting them into the shot", 40),
    Stage("Performing the scene", 120),
    Stage("Blending at full resolution", 95),
    Stage("Writing your files", 45),
]



def stages_for(params: dict) -> list[Stage]:
    """A person's stages, or a vehicle's when the selection's words name one (vehicle_swap.STAGES)."""
    return vehicle_swap.stages_for(params) or person_swap.stages_for(params) or STAGES




def search_frames() -> int:
    """How far into a long shot the window may move (SAM 3.1 tracks this many frames)."""
    return 3 * common.MAX_FRAMES



GRID = 8                    # LTX's latent frames: the painted frame is pinned on one
MAX_PIXELS = 1920 * 1088    # the largest frame the Union IC-LoRA graph has run at on a 24 GB card (environment.py)
MIN_PIXELS = 1024 * 1024    # a small crop is rendered bigger than the plate: LTX draws a person poorly at a few hundred px
# On a card with common.BIG_GPU_GB or more (a 96 GB RTX PRO 6000) LTX renders bigger: a 4K shot is scaled up about 1.7x
# instead of 2.3x, so faces and costumes stay sharper. PICKUPS_LTX_PIXELS, when set, decides for every card.
BIG_MAX_PIXELS = 2560 * 1408
SIDE = 128                  # drive() wants multiples of 128
MARGIN = 0.5                # room around everywhere the person goes, as a share of that box, each side
MIN_CROP = 0.45             # the crop is at least this share of the frame each way (LTX needs some of the scene)
WHOLE = 0.8                 # a crop this big both ways is the whole frame
ASPECT = (0.5, 2.2)         # crop shapes LTX is asked to render
KEEP_FIRST = 0.85           # frame 0 is the anchor when the person is nearly as big there (the in-place pin is strongest)
# The painted frame as a keyframe guide later in the shot. At 1.0 LTX keeps a frozen copy of the painted character in
# every frame beside the moving one (seen on a walk-in, anchored at frame 16); at 0.7 the character moves through it.
GUIDE_STRENGTH = 0.7
# Outside the new character, the render replaces the footage only where it differs from it (levels, after the tone
# match; a soft ramp between): the old person's or car's place, not a band of re-drawn background around it.
KEY_LO, KEY_HI = 10.0, 28.0
# The painted frame is checked before LTX follows it, and repainted with the next seed when it fails (paint_problem).
PAINT_TRIES = 3
PAINT_MIN_CHANGE = 18.0  # mean levels the old person's pixels must change by
PAINT_REACH = 0.15  # of the person's height: how far past them a costume may reach
PAINT_FAR_LEVELS, PAINT_FAR_MAX = 48.0, 0.02  # changed pixels well away from the person: at most this share
PAINT_MIN_COVER = 0.6  # of the old person's pixels the new one must cover: they stand where the old one stood
# Words in the character's description that mean they're meant to hold something: then a phone in hand is no fault.
HELD_WORDS = re.compile(r"\b(phone|camera|cup|mug|glass|bottle|holding|holds|carrying|carries|in (?:her|his|their) hands?)\b",
                        re.IGNORECASE)
BACK_VIEW_MIN_H = 0.2  # a person shorter than this share of the frame is too small to tell their face is hidden
SIDE_MIN_DX = 0.12  # of the picture's width: a person who crosses this much of it in the window walks across it
SIDE_TURN = 0.25  # of half their head's width: a face nearer the middle of the head than this faces the camera
AWAY_SHRINK = 0.75  # ending this much of their first height or less, they walked away from the camera
# Run 10 (R&D, Run 9): a previz camera pushing in on a man standing with his back to it read as "walks towards the
# camera", so the painter and LTX turned him round. With the camera moving, a person's size says nothing about walking.
CAMERA_MOVE_SHARE = 0.25  # of the background changed between frames CAMERA_STEP apart: the camera moves
CAMERA_STEP = 6
GAIT_SWING = 0.2  # the legs' width against the height swinging this much (of its mean) frame to frame: walking
STAND = "moves naturally where they stand"
KEEP_REST = " Keep everything else in <image1> exactly as it is."

# "Turn the man into ..., keep his exact pose" keeps a walker mid-stride; "Replace the man with ..." had Qwen-Image 2.1
# copy the sheet's standing front pose instead (seen on a walk towards the camera).
PAINT = ("Turn {person} <image1> into {character}: keep the {noun}'s exact pose{stride}, with the head, arms and legs exactly "
         "where they are, and the same position, size and direction in the picture. Change the face, hair, body and clothes "
         "to look exactly like the character in <image2>, from head to toe, lit by the same light as the scene in <image1>, "
         "with no rim light and no glow around them. Anything they hold or carry that the character in <image2> doesn't "
         "have, like a phone or a cup, is gone. Keep everything else in <image1> exactly as it is.")
PERSON_WORDS = re.compile(r"^(?:the |a |an )?(person|man|woman|girl|boy|child|kid|hiker|walker|runner|tourist|"
                          r"photographer|worker|cyclist|rider|driver)$", re.IGNORECASE)
# A vehicle is recast too ("the car", "the dark SUV"): the same steps, in a car's words.
NO_CARS = "Swapping a car isn't in this version yet. Pick a person to replace instead."  # features.cars() is off
VEHICLE_WORDS = re.compile(r"^(?:the |a |an )?(?:[\w-]+ ){0,2}(car|suv|truck|pickup|van|bus|jeep|taxi|cab|sedan|hatchback|"
                           r"wagon|coupe|convertible|vehicle|motorbike|motorcycle|scooter|tractor|lorry)s?$", re.IGNORECASE)
PAINT_VEHICLE = ("Replace the {noun} {where} <image1> with {vehicle}: the same position, size and angle in the picture, facing "
                 "and travelling the same way, its wheels on the road exactly where the {noun}'s wheels are. Make it look exactly "
                 "like the vehicle in <image2>, its body, colour, wheels and details, lit by the same light as the scene in "
                 "<image1>, with matching reflections and its own soft shadow on the road. Keep everything else in <image1> "
                 "exactly as it is.")
VEHICLE_STILL = ("Studio photograph of {who}: the whole vehicle in {view}, all four wheels and the full body "
                 "inside the frame, standing on a plain light-grey seamless floor and backdrop with a soft contact shadow, "
                 "soft even studio light, sharp detail, true colours, no people, no text.")
VEHICLE_LABELS = ("car", "vehicle", "truck")  # what SAM 3.1 calls the new vehicle on the render


# --- the window, the anchor, the crop ----------------------------------------------------------------------------------


def presence(masks: list[np.ndarray]) -> np.ndarray:
    """Per frame, how much the person is on screen: 0 when not, else 0.5-1 by size (the biggest frame is 1)."""
    area = np.array([float(m.sum()) for m in masks])
    if not area.any():
        return np.zeros(len(masks))
    return np.where(area > 0, 0.5 + 0.5 * area / area.max(), 0.0)


def choose_window(masks: list[np.ndarray], frames: int, total: int, start: int | None = None) -> int:
    """The first frame of the `frames` LTX renders, on a shot of `total` frames. `masks` is the person's track from
    frame 0 (as far as it was followed). The user's start wins; otherwise the window where the person is on screen
    most (frames they're in, bigger counts a little more), the earliest of equals."""
    last = max(0, total - frames)
    if start is not None:
        return max(0, min(int(start), last))
    if last == 0 or not masks:
        return 0
    score = presence(masks)
    sums = [float(score[s:s + frames].sum()) for s in range(min(last, len(masks) - 1) + 1)]
    return int(np.argmax(sums))


def cut_off(mask: np.ndarray) -> float:
    """How much of the person the frame's edge cuts: the share of their outline that lies on the edge (0 = all inside)."""
    if not mask.any():
        return 1.0
    edge = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    if not edge:
        return 0.0
    rim = mask & ~composite.shrink(mask, 1)
    return min(1.0, edge / max(1, int(rim.sum())))


def anchor_frame(masks: list[np.ndarray], grid: int = GRID, allowed=None) -> int:
    """The frame the new character is painted on: on LTX's latent grid, where the person is biggest and least cut off
    by the frame (a walk-in's first frames show half of them). Frame 0 when they're nearly as big there. `allowed`
    (frame -> bool), when true on some frame, keeps the pick to those: on a previz take, the frames where the way the
    figure faces is known (an orbit turns them; previz_facts.Facts.facing_at)."""
    cands = list(range(0, len(masks), grid))
    if allowed is not None and any(allowed(i) for i in cands):
        cands = [i for i in cands if allowed(i)]
        score = [float(masks[i].sum()) * max(0.0, 1.0 - 4.0 * cut_off(masks[i])) for i in cands]
        best = int(np.argmax(score))
        if score[best] <= 0:
            return cands[0]
        return cands[0] if score[0] >= KEEP_FIRST * score[best] else cands[best]
    if not cands:
        return 0
    score = [float(masks[i].sum()) * max(0.0, 1.0 - 4.0 * cut_off(masks[i])) for i in cands]
    best = int(np.argmax(score))
    if score[best] <= 0:
        return 0
    return 0 if score[0] >= KEEP_FIRST * score[best] else cands[best]


def max_pixels(vram_gb: float | None) -> int:
    """The biggest frame LTX renders on this GPU: MAX_PIXELS, or BIG_MAX_PIXELS on a card with BIG_GPU_GB or more."""
    if os.environ.get("PICKUPS_LTX_PIXELS"):
        return int(os.environ["PICKUPS_LTX_PIXELS"])
    return BIG_MAX_PIXELS if (vram_gb or 0) >= common.BIG_GPU_GB else MAX_PIXELS


def gpu_vram(ctx) -> float | None:
    """The GPU's memory in GB (None when ComfyUI doesn't say, or offline tests)."""
    conn = getattr(ctx, "conn", None)
    if conn is None:
        return None
    from pickups.app import bench
    return bench.machine(conn).get("vram_gb")


def ltx_size(cw: float, ch: float, max_px: int = MAX_PIXELS) -> tuple[int, int]:
    """LTX's working size for a crop: its shape, MIN_PIXELS to max_px, both sides multiples of SIDE."""
    px = min(max_px, max(MIN_PIXELS, cw * ch))
    s = math.sqrt(px / (cw * ch))
    return max(2 * SIDE, int(cw * s) // SIDE * SIDE), max(2 * SIDE, int(ch * s) // SIDE * SIDE)


def crop_box(masks: list[np.ndarray], size: tuple[int, int], max_px: int = MAX_PIXELS) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) in a size = (W, H) frame: everywhere the person goes in the window with room around them, at least
    MIN_CROP of the frame each way and in a shape LTX renders (ASPECT, then its own working size's), or the whole frame
    when that is most of it. Even numbers."""
    W, H = size
    union = np.any(np.stack(masks), axis=0)
    if not union.any():
        return 0, 0, W // 2 * 2, H // 2 * 2
    ph, pw = union.shape
    ys, xs = np.nonzero(union)
    sx, sy = W / pw, H / ph
    bx0, bx1, by0, by1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    cw = min(W, max((bx1 - bx0) * (1 + 2 * MARGIN), MIN_CROP * W))
    ch = min(H, max((by1 - by0) * (1 + 2 * MARGIN), MIN_CROP * H))
    if cw >= WHOLE * W and ch >= WHOLE * H:
        return 0, 0, W // 2 * 2, H // 2 * 2
    lo, hi = ASPECT
    if cw / ch < lo:
        cw = min(W, ch * lo)
    elif cw / ch > hi:
        ch = min(H, cw / hi)
    w, h = ltx_size(cw, ch, max_px)  # grow the crop to LTX's own shape where the frame allows, so nothing is stretched
    if cw / ch < w / h:
        cw = min(W, ch * w / h)
    else:
        ch = min(H, cw * h / w)
    if cw >= WHOLE * W and ch >= WHOLE * H:
        return 0, 0, W // 2 * 2, H // 2 * 2
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    x0 = int(min(max((bx0 + bx1) / 2 - cw / 2, 0), W - cw)) // 2 * 2
    y0 = int(min(max((by0 + by1) / 2 - ch / 2, 0), H - ch)) // 2 * 2
    return x0, y0, cw, ch


# --- words ---------------------------------------------------------------------------------------------------------------


def _box(m: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def camera_moves(frames: list[np.ndarray], masks: list[np.ndarray]) -> bool:
    """True when the shot's background changes a lot between frames CAMERA_STEP apart (a push, an orbit, a pan): then
    the person getting bigger or smaller is the camera, not their walk. frames: RGB, the masks' size."""
    shares = []
    for i in range(0, min(len(frames), len(masks)) - CAMERA_STEP, CAMERA_STEP):
        a, b = frames[i], frames[i + CAMERA_STEP]
        if a.shape[:2] != masks[i].shape or b.shape[:2] != masks[i].shape:
            continue
        bg = ~composite.grow(masks[i] | masks[i + CAMERA_STEP], 4)
        if bg.sum() < 0.2 * bg.size:
            continue
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 20
        shares.append(float(diff[bg].mean()))
    return bool(shares) and float(np.median(shares)) >= CAMERA_MOVE_SHARE


def strides(masks: list[np.ndarray]) -> bool:
    """True when their legs scissor: the width of the lowest third of the silhouette against its height swings from
    frame to frame (a walk), which no camera move does to someone standing."""
    r = []
    for m in masks:
        ys = np.flatnonzero(m.any(axis=1))
        if len(ys) < 8:
            continue
        h = ys[-1] - ys[0] + 1
        cols = np.flatnonzero(m[ys[0] + int(0.67 * h): ys[-1] + 1].any(axis=0))
        if len(cols):
            r.append((cols[-1] - cols[0] + 1) / h)
    if len(r) < 12:
        return False
    v = np.asarray(r)
    t = np.arange(len(v))
    swing = v - np.polyval(np.polyfit(t, v, 1), t)
    return float(swing.std() / max(v.mean(), 1e-6)) >= GAIT_SWING


def motion_words(masks: list[np.ndarray], moving_camera: bool = False) -> str:
    """What the person does through the window, from their track: "walks in from the left and away from the camera",
    "walks towards the camera", "moves naturally where they stand". With the camera moving, only their stride says they
    walk, and never which way."""
    seen = [m for m in masks if m.any()]
    if len(seen) < 2:
        return STAND
    if moving_camera:
        return "walks as in the original shot" if strides(seen) else STAND
    h, w = seen[0].shape
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = _box(seen[0]), _box(seen[-1])
    dx = ((bx0 + bx1) - (ax0 + ax1)) / 2 / w
    grow = (by1 - by0) / max(1.0, ay1 - ay0)
    ways = []
    if grow < 0.7:
        ways.append("away from the camera")
    elif grow > 1.4:
        ways.append("towards the camera")
    side = "to the right" if dx > 0 else "to the left"
    if abs(dx) > 0.12 and not ways:
        ways.append(side)
    if ax0 <= 1 and dx > 0.05:
        return "walks in from the left" + (f" and {ways[0]}" if ways and ways[0] != side else "")
    if ax1 >= w - 1 and dx < -0.05:
        return "walks in from the right" + (f" and {ways[0]}" if ways and ways[0] != side else "")
    return f"walks {' and '.join(ways)}" if ways else STAND


def where_words(mask: np.ndarray) -> str:
    """Where the person is on a picture, to go before it: "on the left of", "in the middle of", "on the right of"."""
    if not mask.any():
        return "in"
    x0, _, x1, _ = _box(mask)
    c = (x0 + x1) / 2 / mask.shape[1]
    return "on the left of" if c < 0.38 else "on the right of" if c > 0.62 else "in the middle of"


def person_noun(text: str) -> str:
    """The word for who is replaced: the user's own when it names a person ("man", "the man in the green parka"),
    else "person"."""
    label = (common.labels(text) or [""])[0]  # "woman, handbag": the woman
    m = PERSON_WORDS.match(label) or common.PERSON.search(label)
    return m.group(1).lower() if m else "person"


def person_phrase(text: str) -> str:
    """The user's whole description of who is replaced, when it says more than the noun ("man in green parka"), so the
    painter can tell him from the child he's kneeling by; else "". "The person on the right" picked the child."""
    label = re.sub(r"^(?:the|a|an)\s+", "", (common.labels(text) or [""])[0].strip(), flags=re.IGNORECASE)
    return label.lower() if common.PERSON.search(label) and label.lower() != person_noun(text) else ""


def paint_instruction(noun: str, where: str, walking: bool, sheet: bool, who: str, phrase: str = "", still: str = "") -> str:
    """The frame editor's instruction: turn the person (named by where they are, and by the user's own description
    when there is one) into the character, in their exact pose and the scene's light. `still`: their posture when it
    is known they stay where they are ("standing", from the previz), so the painter never makes a walk of it."""
    doing = " walking" if walking else (" " + still if still else "")
    person = f"the {phrase or noun}{doing} {where}"
    character = "the character from the character sheet in <image2>" if sheet else "the character in <image2>"
    if who:
        character += f" ({who})"
    return PAINT.format(person=person, character=character, noun=noun, stride=", mid-stride" if walking else "")


def facing_words(view: str) -> str:
    """The way the person faces on the anchor frame (paint_view's view), as LTX reads it, in plain words: "facing away
    from the camera, back to the camera", "facing the camera", "side on, in profile, facing left"; "" when unsure."""
    base, _, way = (view or "").partition("-")
    if base == "back":
        return ", facing away from the camera, back to the camera" + (f", turned to the {way}" if way else "")
    if base == "front":
        return ", facing the camera"
    if base == "side" and way:
        return f", side on, in profile, facing {way}"
    return ""


def ltx_prompt(who: str, motion: str, scene: str, vehicle: bool = False, view: str = "") -> str:
    """LTX-2.5's prompt: the character doing what the person does (facing the way the frame shows them), then the
    scene and its light, in the present tense."""
    subject = (who.strip().rstrip(".") or ("the new vehicle" if vehicle else "the new character"))
    subject = subject[0].upper() + subject[1:]
    parts = [f"{subject} {motion}{'' if vehicle else facing_words(view)}.", scene.strip()]
    # Nothing about the wheels: "its wheels keep their shape and roll steadily" had LTX steer a convertible's front
    # wheel nearly sideways out of its arch, on two seeds.
    parts.append("It is lit by the scene's own light, with real reflections on its paint and glass and a soft shadow on the "
                 "road; the camera and the place stay exactly as in the original shot." if vehicle else
                 "They are lit by the scene's own natural light, with soft, realistic shadows and no glow around them; "
                 "the camera and the place stay exactly as in the original shot.")
    return " ".join(x for x in parts if x)


def vehicle_noun(text: str) -> str | None:
    """The vehicle being recast ("car", "suv"), when the selection's words name one; None for a person."""
    m = VEHICLE_WORDS.match((text or "").strip())
    return m.group(1).lower() if m else None


def vehicle_words(noun: str) -> str:
    """How a vehicle is named in a sentence: "SUV", not "suv"."""
    return noun.upper() if noun in ("suv",) else noun


def drive_words(motion: str) -> str:
    """motion_words for a vehicle: it drives, and a car that holds its place in a tracking shot is driving alongside."""
    if motion.startswith("moves naturally"):
        return "drives steadily along the road, the camera travelling with it"
    return motion.replace("walks", "drives", 1)


def paint_vehicle(noun: str, where: str, who: str) -> str:
    """The frame editor's instruction for a vehicle: the new one in the old one's place, angle and direction."""
    vehicle = "the vehicle in <image2>" + (f" ({who})" if who else "")
    return PAINT_VEHICLE.format(noun=vehicle_words(noun), where=where, vehicle=vehicle)


# --- masks ---------------------------------------------------------------------------------------------------------------


def protect_others(others: np.ndarray, old: np.ndarray, feather: float) -> np.ndarray | None:
    """Other people (tracked with the person, never recast) as a 0..1 matte that always stays footage: their outline
    less the person's own, softened a little outward but never into the person (a parent's hands on a child's jacket
    are the parent's). None when no one else is there. The blend takes it out of the matte: LTX redraws a child the
    parent touches, and SAM's masks of the two bleed into each other where they touch."""
    keep = others & ~old
    if not keep.any():
        return None
    return composite.soft_matte(keep, 0, feather) * ~old


def new_character(found: np.ndarray, old: np.ndarray, reach: int) -> np.ndarray:
    """The new character on one rendered frame: what SAM 3.1 found or followed that touches where the old person was,
    grown by `reach`. Anything else LTX drew further off is left out (it isn't pasted). SAM often splits a costume at a
    thin gap (a helmet above its neck ring, a glove past a cuff), so pieces a small gap apart count as one."""
    if not found.any() or not old.any():
        return np.zeros_like(found)
    rows = np.flatnonzero(old.any(axis=1))
    gap = max(2, round(0.03 * (rows[-1] - rows[0] + 1)))
    return reference.touching(composite.grow(found, gap), composite.grow(old, reach)) & found


def find_new(ctx, gen_small: Path, k: int, old_k: np.ndarray, run_dir: Path,
             labels: tuple[str, ...] = recast.NEW_LABELS) -> tuple[list[np.ndarray], str]:
    """The new character on every rendered frame (at gen_small's size), and what seeded it: SAM 3.1 finds them on the
    anchor frame, where they stand in the painted frame (the first of recast.NEW_LABELS that finds someone there: a
    robot or a T-rex may not answer to "person"), then follows them forward and backward. Where SAM finds no one, the
    old person's own outline seeds the tracking: the painted character stands exactly there."""
    png = common._frame_png(gen_small, k, run_dir / "new_anchor.png")
    size = Image.open(png).size
    old = recast.upsize(old_k, size)
    seed, how = np.zeros_like(old), "the person's outline"
    for label in labels:
        try:
            found = common.select_instances(ctx, png, label, 1.0, run_dir / f"new_{label}")
        except Exception:  # noqa: BLE001 - try the next word
            continue
        hit = [m for m in found if m.shape == old.shape and (m & composite.grow(old, 3)).any()]
        if hit:
            seed, how = np.logical_or.reduce(hit), label
            break
    if not seed.any():
        seed = old
    masks = common.track_from(ctx, gen_small, k, common.split_objects(seed), run_dir, folder="new_track",
                              prefix="pickups/recast_track")
    return masks, how


def clicked_vehicle(ctx, proxy: Path, frame: int, track: list[np.ndarray], run_dir: Path) -> str | None:
    """"car" when what the user clicked is a vehicle (SAM 3.1 finds a vehicle that is mostly the selection), else None.
    A click has no words, so this is how a clicked car gets a car's instructions."""
    sel = track[frame] if 0 <= frame < len(track) else None
    if sel is None or not sel.any():
        return None
    try:
        found = common.select_instances(ctx, common._frame_png(proxy, frame, run_dir / "kind.png"), "vehicle", 1.0, run_dir / "kind")
    except Exception:  # noqa: BLE001 - not knowing means a person, as before
        return None
    for m in found:
        if m.shape == sel.shape and (m & sel).sum() >= 0.5 * min(m.sum(), sel.sum()):
            return "car"
    return None


def sheet_view(path: Path, which: str) -> Image.Image | None:
    """One view cut from a character sheet: "front" or "back" (its whole standing figures, left to right: the front
    view first). None when the picture isn't a sheet with both."""
    try:
        rgb, alpha = reference.load(path)
        figs, _ = reference._figures(rgb, alpha)
    except (OSError, ValueError):
        return None
    boxes = []
    for f in figs:
        ys, xs = np.nonzero(f.mask)
        if not f.whole or not len(ys):
            continue
        px0, py0, px1, py1 = f.panel  # the figure's own outline, scaled from its panel's analysis size
        sx, sy = (px1 - px0) / f.mask.shape[1], (py1 - py0) / f.mask.shape[0]
        bx0, by0, bx1, by1 = px0 + xs.min() * sx, py0 + ys.min() * sy, px0 + (xs.max() + 1) * sx, py0 + (ys.max() + 1) * sy
        if by1 - by0 >= reference.STANDING * (bx1 - bx0):  # standing figures, not a close-up
            boxes.append((bx0, by0, bx1, by1))
    if len(boxes) < 2:
        return None
    x0, y0, x1, y1 = sorted(boxes)[0 if which == "front" else 1]
    pad = 0.08 * (y1 - y0)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(W, int(x1 + pad)), min(H, int(y1 + pad)))
    return Image.fromarray(rgb[y0:y1, x0:x1])


def faces_on(ctx, frame_png: Path, run_dir: Path) -> list[np.ndarray] | None:
    """The faces SAM 3.1 finds on a picture; None when it couldn't look."""
    try:
        return common.select_instances(ctx, frame_png, "face", 1.0, run_dir / "faces")
    except Exception:  # noqa: BLE001 - not knowing: no view picked
        return None


def facing_from(faces: list[np.ndarray] | None, person: np.ndarray) -> str:
    """"front" when one of `faces` is in the top third of the person, else "back" (seen from behind, or too far off
    to tell); "" when faces couldn't be looked for."""
    if faces is None:
        return ""
    if not person.any():
        return "back"
    return "front" if _face_on(faces, person) is not None else "back"


def _face_on(faces: list[np.ndarray], person: np.ndarray) -> np.ndarray | None:
    ys = np.flatnonzero(person.any(axis=1))
    top = np.zeros_like(person)
    top[ys[0]:ys[0] + max(1, (ys[-1] - ys[0]) // 3)] = True
    near = composite.grow(person & top, 3)
    for f in faces:
        f = recast.upsize(f, (person.shape[1], person.shape[0]))
        if (f & near).any():
            return f
    return None


def facing(ctx, frame_png: Path, person: np.ndarray, run_dir: Path) -> str:
    """"front" when SAM 3.1 finds a face in the top third of the person on this frame, else "back" (seen from behind,
    or too far off to tell)."""
    if not person.any():
        return "back"
    return facing_from(faces_on(ctx, frame_png, run_dir), person)


def face_turn(faces: list[np.ndarray] | None, person: np.ndarray) -> float | None:
    """How far the person's face sits to the side of their head: about 0 facing the camera, towards +1 in profile
    facing right, -1 facing left. None when no face shows (from behind, or faces couldn't be looked for)."""
    if not faces or not person.any():
        return None
    f = _face_on(faces, person)
    if f is None:
        return None
    ys = np.flatnonzero(person.any(axis=1))
    head = person[ys[0]:ys[0] + max(1, (ys[-1] - ys[0] + 1) // 7)]
    hx = np.flatnonzero(head.any(axis=0))
    fx = np.flatnonzero(f.any(axis=0))
    if not len(hx) or not len(fx):
        return None
    half = max(1.0, (hx[-1] - hx[0] + 1) / 2)
    return float(np.clip(((fx[0] + fx[-1]) / 2 - (hx[0] + hx[-1]) / 2) / half, -1, 1))


def walk_side(track: list[np.ndarray]) -> str:
    """"right" or "left" when the person walks across the picture (SIDE_MIN_DX of its width or more, neither towards
    nor away from the camera), else ""."""
    seen = [m for m in track if m.any()]
    if len(seen) < 2:
        return ""
    w = seen[0].shape[1]
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = _box(seen[0]), _box(seen[-1])
    dx = ((bx0 + bx1) - (ax0 + ax1)) / 2 / w
    grow = (by1 - by0) / max(1.0, ay1 - ay0)
    if abs(dx) < SIDE_MIN_DX or not 0.7 <= grow <= 1.4:
        return ""
    return "right" if dx > 0 else "left"


def walks_away(track: list[np.ndarray], towards: bool = False) -> bool:
    """True when the person gets clearly smaller over the window (to AWAY_SHRINK of their height or less): walking
    away from the camera. With `towards`, when they get clearly bigger (by 1 / AWAY_SHRINK): walking towards it."""
    seen = [m for m in track if m.any()]
    if len(seen) < 2:
        return False
    first, last = _box(seen[0]), _box(seen[-1])
    a, b = first[3] - first[1], last[3] - last[1]
    return a <= AWAY_SHRINK * b if towards else b <= AWAY_SHRINK * a


def paint_view(ctx, anchor: Path, old: np.ndarray, track: list[np.ndarray], character: Path, views: int,
               instruction: str, run_dir: Path, pose_view: str = "", moving_camera: bool = False) -> tuple[str, str, Path | None]:
    """Which way the painter must turn the character on the anchor frame, said in the instruction (the view, the
    instruction, and a cut of the sheet's view to show it). "front"/"back" from a face on the anchor; walking across
    the picture adds the way they face: "side-right", "back-left" ...

    A sheet holds the character from the front and from behind: the painter is shown the one this frame needs (an
    astronaut seen from behind kept its sheet's backpack only once it was pointed at the back view). One front-facing
    still and a person walking away: told nothing, the painter turned a woman on a castle path to face the camera and
    LTX spun her round in the first second. A woman walking across a castle hall in profile was painted facing the
    camera too. moving_camera: their size changing is the camera's (camera_moves), so it never overrules the face."""
    if pose_view:  # from the pose's shoulders and hips (person_swap.pose_facing): no face detection to fool
        pbase, _, pway = pose_view.partition("-")
        faces, seen = [], "" if pbase == "side" else pbase
    else:
        pway = ""
        faces = faces_on(ctx, anchor, run_dir)
        seen = facing_from(faces, old)
    if not pose_view and not moving_camera and seen == "back" and walks_away(track, towards=True):
        # No face found, but they end the window walking towards the camera: they turn round at some point, and on
        # this frame they may be in profile (the man on the pier, looking out over the water). Nothing is asserted;
        # the painter keeps the direction the picture shows.
        seen = ""
    if not pose_view and not moving_camera and seen != "back" and walks_away(track):
        # Walking away from the camera they show their back: a cap seen from behind answered to "face", and the
        # knight was painted (and rendered) facing the camera while the man looked out over the valley. Also when
        # faces couldn't be looked for: told nothing, the painter turned the walker on the trail to the camera.
        seen, faces = "back", []
    big = old.any() and np.ptp(np.flatnonzero(old.any(axis=1))) >= BACK_VIEW_MIN_H * old.shape[0]
    view, cut_png = "", None
    cut = sheet_view(character, seen) if views > 1 and seen else None
    if cut is not None:
        view = seen
        cut_png = run_dir / f"view_{view}.png"
        cut.save(cut_png)
        seen_as = "from the front, facing the camera" if view == "front" else "from behind, facing away from the camera, back to the camera"
        instruction = instruction.replace(KEEP_REST, f" From this angle they are seen {seen_as}: "
                                          f"copy their {'front' if view == 'front' else 'back'} exactly as <image3> shows it." + KEEP_REST)
    elif big and seen == "back":  # also a sheet whose back view couldn't be cut: never left to turn them round
        view = "back"
        instruction = instruction.replace(KEEP_REST, " They are seen from behind, facing away from the camera, back to the camera: paint the back of the character's "
                                          "head and body, with no face showing. The head is a natural rounded head with the character's own hair, and no hat, "
                                          "cap or hood unless the character in <image2> wears one." + KEEP_REST)
    way = pway or (walk_side(track) if big and not pose_view and not moving_camera else "")
    if way:
        turn = None if pose_view else face_turn(faces, old)
        if turn is None or abs(turn) >= SIDE_TURN:  # a face square to the camera on a walk across: they look at us
            away = " and turned partly away from the camera" if seen == "back" else ""
            walking = f" and walking to the {way} of the picture" if walk_side(track) == way else ""
            instruction = instruction.replace(KEEP_REST, f" They are seen from the side, side on, in profile{away}, facing {way}{walking}: "
                                              "keep them turned exactly that way, not towards the camera." + KEEP_REST)
            view = f"{view or 'side'}-{way}" if view != "front" else f"side-{way}"
    return view, instruction, cut_png


def cut_words(mask: np.ndarray, edge_px: int = 2) -> str:
    """The sentence that tells the painter the frame's edge cuts the person off ("" when it doesn't): told nothing, it
    shrank a walker cut off at the hips into a small whole figure, in the wrong place, on three tries."""
    if not mask.any():
        return ""
    h, w = mask.shape
    sides = [name for name, band in (("bottom", mask[h - edge_px:]), ("top", mask[:edge_px]), ("left", mask[:, :edge_px]),
                                     ("right", mask[:, w - edge_px:])) if band.sum() >= max(2, 0.02 * mask.sum() ** 0.5)]
    if not sides:
        return ""
    where = " and ".join(sides[:2])
    edges = "edges" if len(sides) > 1 else "edge"
    return (f" The {where} {edges} of the picture cut them off: keep them exactly this size and in this place, cut off by "
            f"the {where} {edges} the same way; do not shrink them to fit the whole figure in.")


def view_problem(faces: list[np.ndarray] | None, person: np.ndarray, view: str) -> str:
    """"facing the wrong way" when the painted person doesn't face as `view` says; "" when they do, or when faces
    couldn't be looked for."""
    if not view or faces is None:
        return ""
    base, _, way = view.partition("-")
    if way:  # walking across: no face (from behind) or a face turned their way, never one square to the camera
        turn = face_turn(faces, person)
        if turn is not None and (abs(turn) < SIDE_TURN or (turn > 0) != (way == "right")):
            return "facing the wrong way"
        return ""
    if base in ("front", "back") and facing_from(faces, person) != base:
        return "facing the wrong way"
    return ""


def _rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)


def paint_reach(old_k: np.ndarray) -> np.ndarray:
    """Where the painted frame may differ from the footage: the old person and a margin for a bulkier costume."""
    ys = np.flatnonzero(old_k.any(axis=1))
    tall = int(ys[-1] - ys[0] + 1) if len(ys) else 1
    return composite.grow(old_k, max(4, round(PAINT_REACH * tall)))


def paint_changes(anchor: Path, painted: Path, old_k: np.ndarray) -> str:
    """Did the painter change the right person? The old person must change, and nothing well away from them may: told
    to turn a kneeling man into a woman, Qwen left him as he was and turned the child beside him into her instead."""
    size = (old_k.shape[1], old_k.shape[0])
    d = composite.blur(np.abs(_rgb(painted, size) - _rgb(anchor, size)).max(axis=-1), 2.0)
    if not old_k.any() or float(d[old_k].mean()) < PAINT_MIN_CHANGE:
        return "the person wasn't changed"
    far = ~paint_reach(old_k)
    if far.any() and float((d[far] > PAINT_FAR_LEVELS).mean()) > PAINT_FAR_MAX:
        return "someone else was changed"
    return ""


PHONE_MAX_SHARE = 0.03  # a phone's mask is under this share of the person's area (SAM's "phone" can answer with the whole person)
PHONE_ASPECT = (0.25, 0.95)  # and roughly phone-shaped: its box's short side over its long side, between these
PHONE_FILL = 0.4  # and it fills this much of that box
PHONE_HAND_PX = 16  # and it touches a hand: within this many px of the pose's arm and hand points, when the pose is known


def phone_counts(phone: np.ndarray, person: np.ndarray, hands: np.ndarray | None = None) -> bool:
    """Is this mask of "phone" a phone in their hand? Small (PHONE_MAX_SHARE of the person), phone-shaped, touching them
    and, when the pose's hand points are known, touching one. Anything else (the whole knight) is ignored."""
    area = int(phone.sum())
    if not area or not person.any() or area > PHONE_MAX_SHARE * person.sum():
        return False
    ys, xs = np.nonzero(phone)
    h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
    if not PHONE_ASPECT[0] <= min(h, w) / max(h, w) <= PHONE_ASPECT[1] or area / (h * w) < PHONE_FILL:
        return False
    if not (composite.grow(phone, 3) & person).any():
        return False
    return hands is None or bool((composite.grow(phone, PHONE_HAND_PX) & hands).any())


def paint_problem(ctx, painted: Path, old_k: np.ndarray, view: str, wanted: str, run_dir: Path,
                  anchor: Path | None = None, hands: np.ndarray | None = None) -> str:
    """What's wrong with the painted frame, before LTX spends two minutes following it ("" when nothing). The new person
    must stand where the old one stood (a sheet's pose copied instead), face the way the footage does (a walker painted
    facing the camera, then spun round by LTX), and hold no phone the character wasn't asked to have (a noblewoman kept
    the tourist's smartphone)."""
    if anchor is not None:
        changed = paint_changes(anchor, painted, old_k)
        if changed:
            return changed
    try:
        people = common.select_instances(ctx, painted, "person", 1.0, run_dir / "people")
    except Exception:  # noqa: BLE001 - a check we can't run passes
        return ""
    size = (old_k.shape[1], old_k.shape[0])
    people = [recast.upsize(m, size) for m in people]
    best = max(people, key=lambda m: (m & old_k).sum(), default=None)
    if best is None or (best & old_k).sum() < PAINT_MIN_COVER * old_k.sum():
        return "not where the person stands"
    if view:
        wrong = view_problem(faces_on(ctx, painted, run_dir), best, view)
        if wrong:
            return wrong
    if not HELD_WORDS.search(wanted or ""):
        try:
            held = common.select_instances(ctx, painted, "phone", 1.0, run_dir / "phone")
        except Exception:  # noqa: BLE001
            held = []
        hands = None if hands is None else recast.upsize(hands, size)
        if any(phone_counts(recast.upsize(m, size), best, hands) for m in held):
            return "holding a phone"
    return ""


# --- the character's picture -----------------------------------------------------------------------------------------------


def cast_sheet(entry: dict | None, cast_dir: Path) -> Path | None:
    """A cast member's character sheet (front | back | close-up), when one was built for them (Store.save_sheet)."""
    sheet = (entry or {}).get("sheet")
    if not isinstance(sheet, dict):
        return None
    f = sheet.get("file")
    if isinstance(f, str) and f and "/" not in f and "\\" not in f and f not in (".", ".."):
        path = cast_dir / f
        if path.is_file():
            return path
    return None


def character_picture(ctx, p: dict, run_dir: Path, vehicle: bool = False,
                      view: str = "") -> tuple[Path, dict, str, int, bool]:
    """The picture the frame editor copies the character from: (file, params, what it is, views on it, made a still).

    A cast member's sheet first; otherwise recast.who_is_in's picture, as it is (a sheet keeps all its views; a busy
    photo is narrowed to the one person SAM 3.1 finds there), or the FLUX 3 klein still it made. A vehicle's picture is
    used as it is, and one made from words is a landscape studio still of the whole vehicle, from `view` (the angle
    the old one is seen from: the frame editor copies the still's angle, and a three-quarter still of a convertible was
    painted side-on over an SUV coming head-on)."""
    given = any(p.get(k) for k in ("who_ref", "cast_id", "ref_upload_id", "reference_take"))
    if vehicle and not given and (p.get("describe") or p.get("reference_prompt")):
        job = ctx.run(graphs.still(p.get("reference_prompt") or VEHICLE_STILL.format(who=p["describe"].strip().rstrip("."),
                                                                                    view=view or "a three-quarter front view"),
                                   1216, 832, seed=int(p.get("seed", 42)), prefix="pickups/cast"))
        ref = ctx.vdir / "reference.png"  # kept with the take, like a person's still
        job.get_outputs("PU_Out")[0].to_file(ref)
        made_still, who = True, None
    else:
        ref, p, made_still, who = recast.who_is_in(ctx, p)
    if vehicle:
        try:
            im = Image.open(ref).convert("RGB")
        except OSError as e:
            raise RuntimeError("That picture of the vehicle can't be opened. Try a PNG or JPEG.") from e
        im.thumbnail((2048, 2048), Image.LANCZOS)
        out = run_dir / "character.png"
        im.save(out)
        return out, p, "a FLUX 3 klein still from your description" if made_still else "your picture of the vehicle", 1, made_still
    cast, cast_file = ctx.store.shot_cast(ctx.shot_id)
    if who is None and isinstance(p.get("who_ref"), dict) and p["who_ref"].get("kind") == "cast":
        who = next((c for c in cast if c["id"] == p["who_ref"].get("id")), None)
    if who is not None and (sheet := cast_sheet(who, cast_file(who).parent)):
        return sheet, p, f"{who['name']}'s character sheet", 3, made_still
    try:
        prepared = reference.prepare(ref, 2 / 3)
        rgb = reference.load(ref)[0]
    except (OSError, ValueError) as e:  # not a picture we can read
        raise RuntimeError("That picture of who's in can't be opened. Try a PNG or JPEG.") from e
    out = run_dir / "character.png"
    if prepared is None:  # a busy photo: just them
        picked = recast._reference(ctx, ref, 2 / 3, run_dir)
        picked.image.save(out)
        return out, p, picked.path, 1, made_still
    im = Image.fromarray(rgb)
    im.thumbnail((2048, 2048), Image.LANCZOS)
    im.save(out)
    if made_still:
        what = "a FLUX 3 klein still from your description"
    elif prepared.views > 1:
        what = f"a {prepared.views}-view character sheet"
    else:
        what = "one figure on a plain backdrop"
    return out, p, what, prepared.views, made_still


# --- the task ------------------------------------------------------------------------------------------------------------


def previz_of(ctx, track: list[np.ndarray], total: int) -> previz_facts.Facts | None:
    """What the previz knows about this take's figure and camera, when the take is an app-made previz (or a take made
    from one that keeps its camera and blocking); None for imported footage. The figure is the one where the person
    was picked on frame 0."""
    job = getattr(ctx, "job", None) or {}
    found = previz_facts.of_take(getattr(ctx, "store", None), getattr(ctx, "shot_id", None), job.get("version_id"))
    if found is None:
        return None
    spec, framing, offset, frames = found
    m0 = track[0] if track else None
    x = None
    if m0 is not None and m0.any():
        x0, _, x1, _ = _box(m0)
        x = (x0 + x1) / 2 / m0.shape[1]
    return previz_facts.facts(spec, framing, frames or total + offset, x=x, offset=offset)



def run(ctx) -> dict:
    from pickups.tasks import recast_many  # "Recast anyone": several people, each to their own character
    swaps = recast_many.parse_swaps(ctx.params)
    if len(swaps) > 1:
        return recast_many.run(ctx, swaps)
    p = recast_many.single_params(ctx.params, swaps[0]) if swaps else dict(ctx.params)
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)
    seed = int(p.get("seed", 42))

    ctx.stage("Preparing the shot")
    src = media.probe(ctx.input_path)
    frames = 8 * ((min(common.length_cap(p), src.frames) - 1) // 8) + 1  # LTX lengths: 8k+1
    if frames < media.MIN_FRAMES:
        raise RuntimeError("This shot is too short to recast. Try one that's at least half a second long.")
    given = p.get("start_frame")
    start = None if given in (None, "") else max(0, min(int(given), src.frames - frames))
    n_search = min(src.frames, max(search_frames(), (start or 0) + frames))
    pw, ph = media.work_size(src.width, src.height, common.PROXY_LONG)
    proxy = run_dir / "proxy.mp4"
    media.prepare_work_clip(ctx.input_path, proxy, 0, n_search, (pw, ph))
    common.check_not_blank(proxy)

    vnoun = vehicle_noun(p.get("text", "")) if str(p.get("kind", "")).lower() != "person" else None
    if vnoun and not features.cars():
        raise RuntimeError(NO_CARS)
    if vnoun:  # a vehicle drives the old one's path on a clean road (vehicle_swap): its own stages
        vehicle_swap.use_stages(ctx)
    ctx.stage(vehicle_swap.FIND if vnoun else "Finding the person")
    others: list[np.ndarray] = []  # everyone else in the shot, tracked with them: always kept as footage
    track = common.checked_masks(ctx, proxy, int(p.get("frame", 0)), p.get("points", []), p.get("text", ""), src.width, run_dir,
                                 others_out=others, one=True)
    if not vnoun and str(p.get("kind", "")).lower() != "person" and p.get("points") and not (p.get("text") or "").strip():
        vnoun = clicked_vehicle(ctx, proxy, int(p.get("frame", 0)), track, run_dir)
        if vnoun and not features.cars():
            raise RuntimeError(NO_CARS)
        if vnoun:
            vehicle_swap.use_stages(ctx)
    s0 = choose_window(track, frames, src.frames, start)
    guide = str(p.get("guide") or "").lower()
    pose = False
    if not vnoun and guide in ("pose", "auto"):
        # The pose path only where it holds up (person_swap.pose_guard). "auto" otherwise takes the standard one;
        # "pose" asks for it, so only what it can't do at all stops it (no frame to paint them whole on), and says so
        # before the clean plate and the paint spend half an hour on it.
        ps0 = person_swap.pose_window(track, frames, src.frames, start)
        window = track[ps0:ps0 + frames]
        clip = None
        if guide == "auto":
            clip = [f.copy() for i, f in enumerate(common.read_frames(proxy)) if ps0 <= i < ps0 + len(window)]
            ph, pw = window[0].shape
            clip = [f if f.shape[:2] == (ph, pw) else np.asarray(Image.fromarray(f).resize((pw, ph), Image.BILINEAR))
                    for f in clip]
        why = person_swap.pose_guard(window, clip)
        ctx.params["path_check"] = why or "ok"
        if not why:
            pose, s0 = True, ps0
        elif guide == "pose":
            raise RuntimeError(f"This shot can't use the pose swap: {why}. Try the standard swap.")
    blank = np.zeros_like(track[0])
    pm = (track[s0:s0 + frames] + [blank] * frames)[:frames]
    om = (others[s0:s0 + frames] + [blank] * frames)[:frames] if others else []
    common.require_size(pm, src.height)
    if start is None and s0:
        # The take says which part of the shot it is (jobs.py: made_from and the timecode offset read start_frame).
        ctx.params["start_frame"] = s0
    pv = previz_of(ctx, track, src.frames) if not vnoun else None
    k = anchor_frame(pm, allowed=(lambda i: bool(pv.facing_at(s0 + i))) if pv else None)
    full = run_dir / "work.mp4"
    info = common.work_clip(ctx, full, frames=frames, start=s0)
    W, H, fps = info["width"], info["height"], info["fps"]
    frames = info["frames"]
    pm = pm[:frames]
    om = om[:frames]
    if vnoun:
        if not vehicle_swap.old_path(p):
            from pickups.tasks import car_swap
            return car_swap.run(ctx, vehicle_swap.Shot(p=p, run_dir=run_dir, seed=seed, src=src, s0=s0, start=start, pm=pm, k=k,
                                                       full=full, size=(W, H), fps=fps, frames=frames, n_search=n_search,
                                                       noun=vnoun, om=om))
        return vehicle_swap.run(ctx, vehicle_swap.Shot(p=p, run_dir=run_dir, seed=seed, src=src, s0=s0, start=start, pm=pm, k=k,
                                                       full=full, size=(W, H), fps=fps, frames=frames, n_search=n_search,
                                                       noun=vnoun, om=om))
    if pose:  # opt-in ("pose", or "auto" where person_swap.pose_guard lets it)
        person_swap.use_stages(ctx)
        return person_swap.run(ctx, person_swap.Shot(p=p, run_dir=run_dir, seed=seed, src=src, s0=s0, start=start, pm=pm,
                                                     om=om, k=k, full=full, size=(W, H), fps=fps, frames=frames,
                                                     n_search=n_search))
    vram = gpu_vram(ctx)
    max_px = max_pixels(vram)
    cx0, cy0, cw, ch = crop_box(pm, (W, H), max_px)
    w, h = ltx_size(cw, ch, max_px)
    sx, sy = W / pm[0].shape[1], H / pm[0].shape[0]
    pbox = (cx0 / sx, cy0 / sy, (cx0 + cw) / sx, (cy0 + ch) / sy)  # the crop, in proxy pixels
    clip = run_dir / "ltx_in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-vf", f"crop={cw}:{ch}:{cx0}:{cy0},scale={w}:{h}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", str(clip)], check=True)
    old_w = [recast.mask_in(m, pbox, (w, h)) for m in pm]  # the old person on LTX's frames

    ctx.stage("Reading the character sheet")
    character, p, what, views, made_still = character_picture(ctx, p, run_dir, vehicle=bool(vnoun))

    ctx.stage("Painting them into the shot")
    who = (p.get("describe") or "").strip().rstrip(".")
    # The camera's own move (a previz push or orbit) is never read as the person walking (Run 10). On a previz take
    # the previz says how the camera moves, which way the figure faces and whether it walks (Run 11); the picture
    # checks are for imported footage.
    moving = pv.camera_moves if pv else None
    if moving is None:
        mh, mw = pm[0].shape
        window = [f if f.shape[:2] == (mh, mw) else np.asarray(Image.fromarray(f).resize((mw, mh), Image.BILINEAR))
                  for i, f in enumerate(common.read_frames(proxy)) if s0 <= i < s0 + len(pm)]
        moving = camera_moves(window, pm)
    motion = (pv.motion(s0 + k) if pv else "") or motion_words(pm, moving_camera=moving)
    still = pv.figure.posture if pv and pv.figure and not pv.figure.moving else ""
    if vnoun:
        motion = drive_words(motion)
        instruction = (p.get("look_prompt") or "").strip() or paint_vehicle(vnoun, where_words(old_w[k]), who)
    else:
        instruction = (p.get("look_prompt") or "").strip() or paint_instruction(
            person_noun(p.get("text", "")), where_words(old_w[k]), motion.startswith("walks"), views > 1, who,
            person_phrase(p.get("text", "")), still=still)
    anchor = common._frame_png(clip, k, run_dir / "anchor.png")
    refs = [ctx.client.assets.from_file(character)]
    view = sheet_cut = ""
    known = pv.view_at(s0 + k) if pv else ""
    if not vnoun and not (p.get("look_prompt") or "").strip() and (known or not pv):
        # A previz take that can't tell which way the figure faces says nothing about it (never a guess from a face).
        view, instruction, cut = paint_view(ctx, anchor, old_w[k], old_w, character, views, instruction, run_dir,
                                            pose_view=known, moving_camera=moving)
        if cut is not None:
            sheet_cut = cut.stem.removeprefix("view_")
            refs.append(ctx.client.assets.from_file(cut))
    safe = ctx.commercial_safe()
    painted = run_dir / "painted.png"
    gate: list[str] = []
    for attempt in range(PAINT_TRIES):
        job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(anchor), instruction, seed=seed + attempt,
                                        refs=refs, resolution=0, prefix="pickups/recast_paint",
                                        commercial_safe=safe, size=graphs.edit_size(w, h)))
        try_png = run_dir / f"painted_{attempt}.png"
        job.get_outputs("PU_Plate")[0].to_file(try_png)
        if attempt == 0:
            shutil.copyfile(try_png, painted)
        if vnoun:
            break
        problem = paint_problem(ctx, try_png, old_w[k], view, f"{who} {p.get('prompt') or ''}", run_dir / f"gate_{attempt}",
                                anchor=anchor)
        gate.append(problem or "ok")
        if not problem:
            shutil.copyfile(try_png, painted)
            break
    if old_w[k].any():  # only the person's place comes from the painter; a stray edit elsewhere never reaches LTX
        reach = composite.soft_matte(paint_reach(old_w[k]), 1, max(2.0, w / 200))[..., None]
        a, pt = _rgb(anchor, (w, h)), _rgb(painted, (w, h))
        Image.fromarray(np.clip(np.rint(a * (1 - reach) + pt * reach), 0, 255).astype(np.uint8)).save(painted)

    ctx.stage("Performing the scene")
    prompt = (p.get("prompt") or "").strip() or ltx_prompt(who, motion, p.get("scene") or "", vehicle=bool(vnoun), view=view)
    job = ctx.run(graphs.drive(ctx.client.assets.from_file(clip), ctx.client.assets.from_file(painted), prompt, fps, w, h,
                               control_mode="depth", seed=seed, prefix="pickups/recast_ltx", first_at=k,
                               first_strength=float(p.get("guide_strength", GUIDE_STRENGTH)) if k else 1.0))
    gdir = run_dir / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)
    gen = sorted(gdir.glob("*.png"))
    if not gen:
        raise RuntimeError("LTX-2.5 didn't send the shot back. Try again.")
    ddir = run_dir / "depth"  # the shot's own depth, the render's guide: what stands in front of the person
    ddir.mkdir(exist_ok=True)
    try:
        for o in job.get_outputs("PU_ControlPreview"):
            o.to_file(ddir / o.name)
    except Exception:  # noqa: BLE001 - no depth back: blend as before, nothing kept in front
        pass
    depth = sorted(ddir.glob("*.png"))
    if len(depth) != len(gen):
        depth = []
    gen_small = run_dir / "gen.mp4"  # SAM's copy of the render, at the proxy's size
    sw_, sh_ = media.work_size(w, h, common.PROXY_LONG)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(gdir / "*.png"),
                    "-vf", f"scale={sw_}:{sh_}:flags=lanczos", "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", str(gen_small)],
                   check=True)

    ctx.stage("Blending at full resolution")
    track_new, seeded = find_new(ctx, gen_small, k, old_w[k], run_dir, labels=VEHICLE_LABELS if vnoun else recast.NEW_LABELS)
    tall = max(1, common.person_height(old_w))
    reach = max(3, round(0.04 * tall))
    new_w = [new_character(recast.upsize(track_new[i], (w, h)) if i < len(track_new) else np.zeros((h, w), bool), old_w[i], reach)
             for i in range(len(old_w))]

    kf = max(W, H) / 1920
    grow_px, feather = max(2, round(6 * kf)), 4.0 * kf
    grain_on = str(p.get("grain", "fine")).lower() != "off"
    need = _grain_needed(full, gen, (cx0, cy0, cw, ch), run_dir) if grain_on else None
    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, fps, kind="h264", scale_long=1920)
    qs = max(1, round(max(cw, ch) / 640))  # QA on every qs-th pixel, as the SCAIL-2 engine

    kept_front = [0]  # frames where something in front of the person stayed real footage
    protected = [0]  # frames where someone else would have been pasted from the render and stayed footage

    def blend(i: int, frame: np.ndarray):
        out = frame.copy()
        if i >= len(gen) or i >= len(old_w):
            return out, out, 0, 0, None
        old = recast.mask_in(pm[i], pbox, (cw, ch))
        new = recast.upsize(new_w[i], (cw, ch))
        protect = protect_others(recast.mask_in(om[i], pbox, (cw, ch)), old, feather / 2) if i < len(om) else None
        # Everything SAM followed, pasted or not: the tone match must not learn from the character's own pixels.
        drawn = recast.upsize(recast.upsize(track_new[i], (w, h)), (cw, ch)) if i < len(track_new) else new
        # The old person plus the new character: a bulkier costume is never clipped to the old outline, and wherever
        # the old person isn't covered by the new one LTX's background fills in. Also the receipt's AI pixels.
        matte = composite.soft_matte(old | new, grow_px, feather) if (old | new).any() else np.zeros((ch, cw), np.float32)
        region = frame[cy0:cy0 + ch, cx0:cx0 + cw]
        if matte.any():
            g = np.asarray(Image.open(gen[i]).convert("RGB").resize((cw, ch), Image.LANCZOS))
            keep_out = np.maximum(matte, composite.grow(drawn, 2 * grow_px).astype(np.float32))
            occ = None
            if depth and old.any():  # a hand or a door frame in front of the person stays real footage
                dep = np.asarray(Image.open(depth[i]).convert("L").resize((cw, ch), Image.BILINEAR))
                occ = composite.occluders(dep, old, composite.grow(old | new, 2 * grow_px), speck=max(2, grow_px // 2))
                occ = composite.soft_matte(occ, 1, feather / 2) if occ.any() else None
                if occ is not None:
                    keep_out = np.maximum(keep_out, occ)
            if protect is not None:
                keep_out = np.maximum(keep_out, protect)
            subject = composite.soft_matte(drawn, max(1, grow_px // 2), feather) if drawn.any() else np.zeros_like(matte)
            g = np.clip(np.rint(composite.harmonize_around(g, region, keep_out, subject)), 0, 255).astype(np.uint8)
            # Only what changed. The new character is always the render; around them (the margin the grown matte adds,
            # and where the old person or car was but the render's background matches the footage) the footage stays.
            # A whole band of re-drawn background around a swapped car read as a soft outline that moved on its own.
            core = composite.soft_matte(new, max(1, grow_px // 2), feather) if new.any() else np.zeros_like(matte)
            diff = composite.blur(np.abs(g.astype(np.float32) - region.astype(np.float32)).max(axis=-1), 2.0 * kf)
            matte = np.maximum(core, matte * np.clip((diff - KEY_LO) / (KEY_HI - KEY_LO), 0.0, 1.0))
            if occ is not None:
                matte = matte * (1.0 - occ)
                kept_front[0] += 1
            if protect is not None:
                if float((matte * protect).max()) > 0.05:
                    protected[0] += 1
                matte = matte * (1.0 - protect)
            if need is not None:
                g = _grain_in(g, matte > 0, need, seed=i)
            pasted, c = composite.paste(region, g, matte)
        else:
            pasted, c = region.copy(), 0
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        touched = matte > 0
        tinted = out.copy()
        reg = pasted.astype(np.float32)
        reg[touched] = reg[touched] * 0.45 + np.array(recast.AI_TINT, np.float32) * 0.55
        tinted[cy0:cy0 + ch, cx0:cx0 + cw] = reg.astype(np.uint8)
        hit = bool(new_w[i].any())
        q = {"out": pasted[::qs, ::qs], "src": region[::qs, ::qs], "new": new[::qs, ::qs], "old": old[::qs, ::qs], "hit": hit,
             "coverage": recast.coverage(new_w[i], old_w[i]),
             "halo": recast.halo_lift(pasted[::qs, ::qs], new[::qs, ::qs], kf / qs) if hit else None}
        return out, tinted, c, int(touched.sum()), q

    changed = touched_px = n = 0
    qa = recast.QA()
    for i, (out, tinted, c, t, q) in enumerate(common.parallel_frames(common.read_frames(full), blend)):
        ctx.check()
        n += 1
        changed += c
        touched_px += t
        qa.add(q)
        out_w.write(out)
        rec_w.write(tinted)
        ctx.progress((i + 1) / max(frames, 1))
    out_w.close()
    rec_w.close()

    ctx.stage("Writing your files")
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=full)
    make_viewer(full, ctx.vdir / "before.mp4")
    (ctx.vdir / "painted_frame.png").write_bytes(painted.read_bytes())
    total = W * H * max(n, 1)
    receipt = {"original": round(100 * (1 - touched_px / total), 2), "borrowed": 0.0, "ai": round(100 * touched_px / total, 2)}
    made = media.made_from(src, s0, n)
    models = [graphs.edit_model(safe), "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3", "SAM 3.1"] + (
        ["FLUX 3 klein"] if made_still else [])
    return {"label": "Replace a person", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, n, fps), "engine": "ltx", "performance_size": f"{w}×{h}", "reference": what,
                        **({"subject": "vehicle"} if vnoun else {}),
                        "pixels_changed_outside_selection": changed, "models": models,
                        **({"made_from": made} if made else {}),
                        "grain": "matched to your shot" if grain_on else "off",
                        "look_prompt": instruction, "prompt": prompt,
                        "_qa": {**qa.summary(), "window_start": s0, "window_chosen": "yours" if start is not None else "auto",
                                "anchor_frame": k, "guide": "first frame" if k == 0 else f"frame {k}", "new_seed": seeded,
                                "sheet_view": sheet_cut or None, "facing": view or None, "camera_moves": moving,
                                "facing_from": ("previz" if known else "unsure") if pv else "picture", "standing": bool(still), "paint_gate": gate, "kept_in_front": kept_front[0], "protected_frames": protected[0],
                                "crop": [cx0, cy0, cw, ch], "work_size": [w, h], "max_pixels": max_px, "gpu_vram_gb": vram,
                                "reference": what, "views": views,
                                "searched_frames": n_search, "grain_added": None if need is None else [round(float(x), 2) for x in need]}}}


# --- grain ---------------------------------------------------------------------------------------------------------------


def _grain_needed(full: Path, gen: list[Path], crop: tuple[int, int, int, int], run_dir: Path,
                  samples: int = 3) -> np.ndarray | None:
    """The grain (per-channel std, 8-bit levels) to add to LTX's scaled-up pixels so they match the plate's: the plate's
    own noise over the crop, less what the render already has (a few frames of each). None when it needs none."""
    x0, y0, cw, ch = crop
    n = len(gen)
    picks = sorted({round(j * (n - 1) / max(1, samples - 1)) for j in range(samples)}) if n else []
    plate, ltx = [], []
    for i in picks:
        f = np.asarray(Image.open(common._frame_png(full, i, run_dir / f"plate_{i}.png")).convert("RGB"))
        plate.append(np.ascontiguousarray(f[y0:y0 + ch, x0:x0 + cw]))
        ltx.append(np.asarray(Image.open(gen[i]).convert("RGB").resize((cw, ch), Image.LANCZOS)))
    if not plate:
        return None
    need = np.sqrt(np.clip(composite.grain_profile(plate) ** 2 - composite.grain_profile(ltx) ** 2, 0, None))
    return need if np.any(need > 0.05) else None


def _grain_in(img: np.ndarray, where: np.ndarray, need: np.ndarray, seed: int) -> np.ndarray:
    """Add grain only over the box around `where` (the rest of the picture is never pasted)."""
    ys, xs = np.nonzero(where)
    if not len(ys):
        return img
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = img.copy()
    out[y0:y1, x0:x1] = composite.grain(np.ascontiguousarray(img[y0:y1, x0:x1]), need, seed)
    return out
