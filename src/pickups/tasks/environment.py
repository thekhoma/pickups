"""Change the weather, time of day or season of a real shot (snow, rain, fog, night, golden hour, autumn).

Step 1: Qwen-Image 2.1 restyles the shot's first frame (everything else told to stay exactly as it is).
Step 2: LTX-2.5 with the Union Control IC-LoRA re-renders the whole shot from that first frame, steered by the real
footage's structure (Depth Anything 3 depth for changes that alter surfaces, like snow or autumn leaves; Canny edges
for changes of light, like night or golden hour), so the camera move and every person's motion stay the shot's own.
Every pixel is generated (the receipt says 100% AI); chain Finish for a 4K master.

Keep exactly as shot (optional): one thing in the shot (a car, a product) can come back as filmed. SAM 3.1 finds it
(`keep_text`, or a click: `keep_frame` + `keep_points`) and follows it through the shot; both models are told to leave
it as it is, and after the render it is put back over every frame with detail transfer: the footage's own fine detail
(badge, grille, edges) on the render's broad light and colour, learnt only where the render drew it in the same place,
so it is the real thing in the new weather. Falling snow and rain from the render pass in front of it, and the render's
snow or spray covers where it meets the ground. The receipt counts its pixels as original footage (the weather in front
as AI), and the take's details say its light comes from the render.
"""

from __future__ import annotations

import itertools
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pickups import composite, graphs, media, mentions
from pickups.app.comfy import Canceled
from pickups.app.jobs import Stage
from pickups.tasks import common, subject_light

STAGES = [
    Stage("Preparing the shot", 20),
    Stage("Finding what to keep", 45),
    Stage("Taking it out of the shot", 60),
    Stage("Restyling the first frame", 90),
    Stage("Rendering the new weather", 140),
    Stage("Lighting what you kept", 120),
    Stage("Keeping it as shot", 45),
    Stage("Writing your files", 40),
]

# The largest frame the Union IC-LoRA graph has run at on a 24 GB card is ~1920x1088; sizes are multiples of 128.
MAX_PIXELS = 1920 * 1088

# Qwen fills an empty meadow with walkers when the light's description mentions people ("rim light on people"), and
# LTX then renders them: never invited.
NOTHING_NEW = "Add no people, animals, vehicles or objects that are not already in the picture."
KEEP = ("Keep everything else exactly the same: the framing and camera angle, the buildings and landscape, every person "
        "with the same face, pose and clothes, and every object in its place.")

# Per preset: the Qwen-Image 2.1 edit instruction for the first frame, the LTX-2.5 video sentence (appended to the
# shot description) and the structure control. Qwen instructions say what changes and then what must not; LTX prompts
# describe what is seen and how it moves, in the present tense.
PRESETS: dict[str, dict] = {
    "snow": {
        "light": ((0.7, 1.4), 0.08),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Snow",
        # Only what is already in the picture gets snow: "roofs and walls" had LTX build a brick house where a misty road
        # had bare trees.
        "qwen": "Make it a winter day in heavy snowfall: thick fresh snow covers the fields and settles on everything already "
                "in the picture, roads and paths stay where they are as snowy tracks, large snowflakes fall through the air, "
                "the sky is a flat overcast white-grey and the light is soft and cold. Add nothing new: no buildings, walls or "
                "objects that aren't there.",
        "ltx": "Heavy snow falls steadily over the scene; thick fresh snow covers the fields and settles on everything in it, "
               "the roads and paths run where they always do, and large flakes drift down through soft, cold, overcast light.",
        "control": "depth",
    },
    "rain": {
        "light": ((0.35, 1.0), 0.08),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Rain",
        "qwen": "Make it a dark, rainy day: heavy rain is falling, roads and paths are soaking wet, glistening and "
                "reflective along their whole length, with only a few small shallow puddles and no large pools of standing "
                "water, grass and verges are dark and wet but stay grass, the sky is low and dark grey, and the light is dim "
                "and cool.",
        "ltx": "Heavy rain pours down, streaking through the air and splashing on the wet, glistening road, which is wet "
               "all along its length with only small shallow puddles and no large pools, under a low, dark grey sky; the grass "
               "stays grass, and every surface glistens in the dim, cool light.",
        "control": "canny",
    },
    "fog": {
        "light_only": 0.35,  # every pixel stays the footage's, in the render's fog; fine texture far off mostly hidden
        "light": ((0.6, 1.3), 0.45),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Fog",
        "qwen": "Fill the scene with thick, soft morning fog: the background fades into pale grey mist, distant objects are "
                "barely visible, and the light is diffuse and cool.",
        "ltx": "Thick morning fog drifts slowly through the scene, fading the distance into pale grey mist under diffuse, "
               "cool light.",
        "control": "depth",
    },
    "night": {
        "light_only": 1.0,  # every pixel stays the footage's, in the render's light (light_transfer); its texture kept
        "light": ((0.05, 0.5), 0.0),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Night",
        "qwen": "Turn this into the same place at night: a deep blue night sky, warm street lamps, lit windows and lamps "
                "glowing, soft moonlight on surfaces, dark shadows, and every person lit by the nearest lights.",
        "ltx": "At night, warm street lamps and lit windows glow against a deep blue sky, and pools of lamplight fall across "
               "the ground and the people moving through the soft shadows.",
        "control": "canny",
    },
    "golden_hour": {
        "light_only": 1.0,  # every pixel stays the footage's, in the render's light (light_transfer); its texture kept
        "light": ((0.4, 1.6), 0.0),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Golden hour",
        "qwen": "Change the light to golden hour: low, warm sunlight from the side, with the sun itself outside the frame "
                "(no sun or sun flare in the picture), long soft shadows, a glowing amber and pink sky, and warm rim light "
                "along every edge.",
        "ltx": "In warm, steady golden-hour light from a low sun outside the frame, long soft shadows lie across the scene "
               "and every edge is rimmed with amber light under a glowing sky; the light does not change and no sun appears.",
        "control": "canny",
    },
    "autumn": {
        "light": ((0.8, 1.25), 0.0),  # a kept thing's plausible change of light (gain), and most veil
        "label": "Autumn",
        # Autumn changes the colours, not the sun: "soft, low, warm light" had LTX stretch a cyclist's midday shadow into
        # a long smear that no longer matched her.
        "qwen": "Make it mid-autumn: grass and meadows turn golden-brown and tawny, trees and shrubs turn orange, red and gold "
                "with some leaves scattered beneath them. Keep the sunlight, its direction and every shadow exactly as they "
                "are. Keep the colours natural, not saturated.",
        "ltx": "In mid-autumn, golden-brown grass and trees in muted orange, red and gold fill the scene, with natural, "
               "unsaturated colour; the sunlight and every shadow stay exactly as in the original shot.",
        "control": "depth",
    },
}


def fit(width: int, height: int) -> tuple[int, int, int, int, int, int]:
    """Where the shot goes on the render's canvas: (picture width, picture height, x, y, canvas width, canvas height).

    The whole frame at its own shape, as big as fits a canvas of multiples of 128 (what the Union guide needs) within
    MAX_PIXELS, centred on it; the spare edge is padding, cut off again on delivery. Filling the canvas and centre-cropping
    instead lost up to 8% of a 4:3 or 1.4:1 frame.
    """
    width, height = max(2, int(width)), max(2, int(height))
    best = (2, 2, 63, 63, 128, 128)  # a frame smaller than the canvas still lands on the 128 grid
    lo, hi = 0.0, math.sqrt(MAX_PIXELS / (width * height)) + 0.01
    for _ in range(40):
        s = (lo + hi) / 2
        cw, ch = max(2, round(width * s) // 2 * 2), max(2, round(height * s) // 2 * 2)
        W, H = max(128, math.ceil(cw / 128) * 128), max(128, math.ceil(ch / 128) * 128)
        if W * H <= MAX_PIXELS:
            best, lo = (cw, ch, (W - cw) // 4 * 2, (H - ch) // 4 * 2, W, H), s
        else:
            hi = s
    return best


def crop_pngs(folder: Path, pattern: str, box: tuple[int, int, int, int]) -> None:
    """Cut the canvas's padding off rendered frames (box = x, y, w, h), so delivery is the shot's own shape."""
    ox, oy, cw, ch = box
    for f in folder.glob(pattern):
        with Image.open(f) as im:
            if im.size == (cw, ch):
                continue
            im = im.convert("RGB").crop((ox, oy, ox + cw, oy + ch))
        im.save(f)


def crop_clip(src: Path, dst: Path, box: tuple[int, int, int, int]) -> None:
    """The work clip without the canvas's padding (for before.mp4 and the delivery's sound)."""
    ox, oy, cw, ch = box
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"crop={cw}:{ch}:{ox}:{oy}",
                    "-c:v", "libx264", "-crf", "2", "-preset", "veryfast", "-c:a", "copy", str(dst)], check=True)


# --- keep exactly as shot ---------------------------------------------------------------------------------------------

# Both models are told to leave the kept thing alone, so the render's copy is already close before it is put back.
KEEP_OBJECT = "Keep {obj} exactly as it is: the same model, colour and shape, with every detail unchanged."
KEEP_OBJECT_VIDEO = "{Obj} stays exactly as it is throughout, the same model, colour and shape."
# Detail transfer's blur, as a share of the kept thing's height (and at least LOW_MIN px): finer than this is the
# footage's own detail (a badge, a grille, every edge), coarser is the render's light and colour.
LOW_SIGMA = 0.03
LOW_MIN = 3.0
GROW, FEATHER = 1, 3.0  # the matte: SAM's outline grown a pixel, feathered outward (px at 1920 wide)
LIGHT_RANGE = 14.0  # levels the kept thing's light may differ, pixel to pixel, from its usual shift into the render's
LUM = np.array([0.299, 0.587, 0.114], np.float32)
DETERMINER = re.compile(r"^(the|a|an|this|that|these|those|my|our|your|his|her|their)\s+", re.I)
AI_TINT = np.array((255, 79, 216), np.float32)  # receipt.mp4: generated pixels, as Remove and Recast mark them


def keep_name(text: str) -> str:
    """What to keep, as the prompts say it: "car" -> "the car", "The blue SUV." -> "the blue SUV"."""
    t = " ".join((text or "").split()).rstrip(".").strip()
    if not t:
        return ""
    if not t[:2].isupper():  # "SUV" stays as it is
        t = t[0].lower() + t[1:]
    return t if DETERMINER.match(t) else f"the {t}"


def sam_words(text: str) -> str:
    """The label SAM 3.1 looks for: the words without "the" ("the car" -> "car")."""
    return DETERMINER.sub("", " ".join((text or "").split()).rstrip(".")).strip()


VEHICLES = frozenset({"suv", "car", "vehicle", "truck", "van", "wagon", "jeep", "pickup", "bus", "lorry"})


def without_subject(text: str, obj: str) -> str:
    """`text` without the sentences that put the kept thing in the picture (from 2a3da3d). When it is taken out of the
    frames the models see, "a small dark SUV on the far road" would have them draw one anyway, beside the real one."""
    name = sam_words(obj).lower()
    if not text or not name:
        return (text or "").strip()
    words = {name, *name.split()} - {"the", "a", "an", "of", "on", "in", "dark", "small", "big", "red", "blue", "white", "black"}
    if words & VEHICLES:
        words |= VEHICLES
    noun = r"(?:" + "|".join(re.escape(a) + "(?:e?s)?" for a in sorted(words, key=len, reverse=True)) + r")"
    pat = re.compile(rf"\b{noun}\b", re.I)
    # "...and a small dark SUV on the far road behind the trees" keeps "the far road behind the trees": the place it
    # was in is still part of the picture.
    where = re.compile(rf"\b(?:(?:a|an|the|one|two|some)\s+)?(?:[\w-]+\s+){{0,3}}?{noun}\b(?:\s+(?:is|are))?"
                       rf"(?:\s+(?:\w+ing|drives|travels|moves|passes|parked|sits|stands|waits))?"
                       rf"\s+(?:on|along|across|down|up|through|in|at|by|near|beside)\s+(?=(?:the|a|an)\b)", re.I)
    kept = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if words & VEHICLES:  # a car is somewhere ("on the road"); "a man in a white T-shirt" is not
            s = where.sub("", s)
        if s and not pat.search(s):
            kept.append(s[0].upper() + s[1:])
    return " ".join(kept).strip()


def where_name(mask: np.ndarray) -> str:
    """A clicked thing with no words, named by where it is on the first frame: "the object in the middle of the frame"."""
    if not mask.any():
        return "the selected object"
    xs = np.flatnonzero(mask.any(axis=0))
    c = (xs[0] + xs[-1] + 1) / 2 / mask.shape[1]
    return f"the object {'on the left of' if c < 0.38 else 'on the right of' if c > 0.62 else 'in the middle of'} the frame"


REF_NEAR = 0.3  # of the real thing's height: the reference's find must lie this close to it (a walker far off isn't it)


def reference_pass(ctx, work: Path, qwen_text: str, video_text: str, obj: str, label: str, kept: list[np.ndarray],
                   size: tuple[int, int], fps: float, control: str, seed: int, run_dir: Path
                   ) -> tuple[list[Path], list[np.ndarray]] | None:
    """A second render of the shot with the kept thing left in it, made only to show how the new light falls on it: its
    broad light and colour are then taken from that drawing (subject_light) and the detail stays the footage's. The
    first frame is the footage's own first frame restyled with the thing in it; its masks come from SAM on the render,
    kept only near the real thing. Returns (frames, masks per frame at the proxy's size), or None."""
    w, h = size
    d = run_dir / "reference"
    d.mkdir(exist_ok=True)
    f0 = d / "frame0.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-frames:v", "1", str(f0)], check=True)
    instruction = qwen_text
    if NOTHING_NEW not in instruction:
        instruction += " " + NOTHING_NEW
    instruction += " " + LAYOUT_KEEP
    if obj and KEEP_OBJECT.format(obj=obj) not in instruction:
        instruction += " " + KEEP_OBJECT.format(obj=obj)
    if not re.search(r"\bkeep\b", qwen_text, re.I):
        instruction += " " + KEEP
    job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(f0), instruction, seed=seed, resolution=0, prefix="pickups/environment_ref_key",
                                    commercial_safe=ctx.commercial_safe(), size=graphs.edit_size(w, h)))
    first = d / "first.png"
    job.get_outputs("PU_Plate")[0].to_file(first)
    job = ctx.run(graphs.drive(ctx.client.assets.from_file(work), ctx.client.assets.from_file(first), video_text, fps, w, h,
                               control_mode=control, seed=seed, prefix="pickups/environment_ref"))
    gd = d / "gen"
    gd.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gd / o.name)
    files = sorted(gd.glob("*.png"))
    if not files or not label:
        return None
    with Image.open(files[0]) as im:
        gw, gh = im.size
    pw, ph = media.work_size(gw, gh, common.PROXY_LONG)
    clip = d / "render.mp4"
    write_clip((np.asarray(Image.open(f).convert("RGB").resize((pw, ph), Image.BILINEAR)) for f in files), clip, (pw, ph), fps, work)
    job = ctx.run(graphs.select(ctx.client.assets.from_file(clip), label, threshold=0.4, prefix="pickups/environment_ref_find"))
    found = common._download_masks(job, d / "find")
    masks = []
    for i, f in enumerate(found):
        real = upsize(kept[min(i, len(kept) - 1)], (pw, ph)) if kept else np.zeros((ph, pw), bool)
        f = f if f.mean() < GHOST_MAX else np.zeros_like(f)
        box = subject_light.box_of(real)
        near = composite.grow(real, max(4, round(REF_NEAR * (box[1] - box[0])))) if box else real
        masks.append(f & near)
    if sum(1 for m in masks if m.any()) < 0.5 * len(masks):
        return None
    return files, masks


def keep_wanted(p: dict) -> bool:
    return bool((p.get("keep_text") or "").strip() or p.get("keep_points"))


def work_points(points: list[dict], src_size: tuple[int, int], content: tuple[int, int],
                origin: tuple[int, int] = (0, 0)) -> list[dict]:
    """Clicks on the shot (in its own pixels) on the work clip: the shot scaled to `content` (w, h) and placed at
    `origin` on the canvas (fit)."""
    (sw, sh), (cw, ch), (ox, oy) = src_size, content, origin
    return [{**pt, "x": ox + min(cw - 1.0, max(0.0, pt["x"] * cw / sw)), "y": oy + min(ch - 1.0, max(0.0, pt["y"] * ch / sh))}
            for pt in points]


def keep_masks(ctx, p: dict, work: Path, start: int, frames: int, place: tuple[int, int, int, int, int, int], src,
               run_dir: Path) -> list[np.ndarray]:
    """The kept thing on every frame of the work clip, at SAM's proxy size: found from the words or the click and
    followed forward and back. A click on a frame outside the part being changed still works: SAM follows it from there
    over a longer clip, and only the work clip's frames are kept."""
    cw, ch, ox, oy, W, H = place
    d = run_dir / "keep"
    d.mkdir(exist_ok=True)
    at = min(src.frames - 1, max(0, int(p.get("keep_frame", start) or 0)))
    c0, c1 = min(start, at), max(start + frames, at + 1)
    clip = work
    if (c0, c1) != (start, start + frames):
        clip = d / "longer.mp4"
        media.prepare_work_clip(ctx.input_path, clip, c0, c1 - c0, (cw, ch), pad=(W, H, ox, oy))
    proxy = d / "proxy.mp4"
    common.proxy_clip(ctx, clip, proxy)
    pts = work_points(p.get("keep_points") or [], (src.width, src.height), (cw, ch), (ox, oy))
    masks = common.tracked_masks(ctx, proxy, at - c0, pts, sam_words(p.get("keep_text", "")), W, d)
    masks = masks[start - c0:start - c0 + frames]
    if not pts:
        try:
            masks = fine_masks(ctx, work, masks, sam_words(p.get("keep_text", "")), (W, H), d) or masks
        except Exception:  # noqa: BLE001 - the proxy's outline is a fallback, never a failed take
            pass
    return masks


FINE_SMALL = 0.15  # of the frame's height: a kept thing smaller than this is followed again on a close crop
FINE_LONG = 1024  # the crop's long side for SAM


def fine_masks(ctx, work: Path, masks: list[np.ndarray], words: str, size: tuple[int, int], d: Path) -> list[np.ndarray] | None:
    """A small kept thing (a far cable car: 60 px on the 1024 proxy) followed again on a close crop of the work clip,
    so its outline follows the hanger's bars and cables instead of a blob around them. Returns full-size masks, or
    None when it isn't small or the crop loses it."""
    found = [m for m in masks if m.any()]
    if not found or not words:
        return None
    W, H = size
    ph, pw = found[0].shape
    union = np.logical_or.reduce(found)
    ys, xs = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
    tall = max(int(np.ptp(np.flatnonzero(m.any(axis=1)))) + 1 for m in found)
    if tall > FINE_SMALL * ph:
        return None
    sx, sy = W / pw, H / ph
    mx, my = max(32, 0.5 * (xs[-1] - xs[0])), max(32, 0.5 * (ys[-1] - ys[0]))
    x0, x1 = int(max(0, (xs[0] - mx) * sx)) // 2 * 2, int(min(W, (xs[-1] + 1 + mx) * sx)) // 2 * 2
    y0, y1 = int(max(0, (ys[0] - my) * sy)) // 2 * 2, int(min(H, (ys[-1] + 1 + my) * sy)) // 2 * 2
    cw, ch = x1 - x0, y1 - y0
    s = FINE_LONG / max(cw, ch)
    zw, zh = round(cw * s) // 2 * 2, round(ch * s) // 2 * 2
    crop = d / "fine.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"crop={cw}:{ch}:{x0}:{y0},scale={zw}:{zh}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", "-c:a", "copy", str(crop)], check=True)
    fd = d / "fine"
    fd.mkdir(exist_ok=True)
    first = next(i for i, m in enumerate(masks) if m.any())
    small = common.tracked_masks(ctx, crop, first, [], words, zw, fd)
    hits = sum(1 for m in small if m.any())
    if hits < 0.9 * len(found):
        return None
    out = []
    for m, coarse in zip(small, masks):
        full = np.zeros((H, W), bool)
        if m.any():
            full[y0:y1, x0:x1] = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((cw, ch), Image.BILINEAR)) > 127
            # the coarse outline bounds it: the crop may also find a second car on the same cable
            full &= composite.grow(upsize(coarse, (W, H)), max(4, round(0.02 * H)))
        out.append(full)
    return out


STABLE_REACH = 4  # frames either side a person's outline is voted over: SAM toggles the gaps between them frame to frame


def subject_offsets(frames: list[np.ndarray], masks: list[np.ndarray]) -> list[tuple[int, int]]:
    """Where the kept thing sits in each frame relative to the first (dx, dy px of the masks' own size), chained from
    phase correlation of the footage around it: it drifts a few px in a followed shot, and a vote over frames must line
    its outline up first."""
    off = [(0, 0)]
    for i in range(1, len(frames)):
        both = masks[i - 1] | masks[i]
        if not both.any():
            off.append(off[-1])
            continue
        ys, xs = np.nonzero(both)
        h, w = both.shape
        y0, y1, x0, x1 = max(0, ys.min() - 16), min(h, ys.max() + 17), max(0, xs.min() - 16), min(w, xs.max() + 17)
        a, b = (f[y0:y1, x0:x1].astype(np.float32) @ LUM for f in (frames[i - 1], frames[i]))
        win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        r = np.fft.fft2((b - b.mean()) * win) * np.conj(np.fft.fft2((a - a.mean()) * win))
        c = np.abs(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
        py, px = np.unravel_index(int(c.argmax()), c.shape)
        dy = py - c.shape[0] if py > c.shape[0] // 2 else py
        dx = px - c.shape[1] if px > c.shape[1] // 2 else px
        if abs(dx) > 12 or abs(dy) > 12:  # a wild peak is no drift
            dx = dy = 0
        off.append((off[-1][0] + int(dx), off[-1][1] + int(dy)))
    return off


def stable_masks(frames: list[np.ndarray], masks: list[np.ndarray], reach: int = STABLE_REACH) -> list[np.ndarray]:
    """The kept people's outlines voted over 2 * reach + 1 frames, each lined up on the footage's drift: SAM's
    answer for a thin gap between two people (their joined hands) changes from frame to frame, and the old and the new
    background showed through it by turns. A symmetric majority keeps a steadily moving edge where it is and decides a
    gap the same way on every frame."""
    n = len(masks)
    if n < 3 or not any(m.any() for m in masks):
        return masks
    off = subject_offsets(frames, masks)
    out = []
    for i in range(n):
        acc, cnt = np.zeros(masks[i].shape, np.float32), 0
        for j in range(max(0, i - reach), min(n, i + reach + 1)):
            if not masks[j].any():
                continue
            acc += composite.shift(masks[j].astype(np.uint8), off[i][0] - off[j][0], off[i][1] - off[j][1]).astype(np.float32)
            cnt += 1
        out.append(acc >= 0.5 * cnt if cnt else masks[i])
    return out


def small_frames(work: Path, size: tuple[int, int]) -> list[np.ndarray]:
    """The work clip's frames at a mask's size (w, h)."""
    return [np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in common.read_frames(work)]


def close_gaps(mask: np.ndarray, frac: float = 0.005) -> np.ndarray:
    """A person's mask with SAM's hairline gaps closed (between a hand and the hip): they come and go from frame to frame,
    and the new weather flickering through them reads as holes in the person. Only hairlines: a real gap between two
    people (their joined hands) is the new background, and filling it showed the old one, relit dark, as a smear."""
    if not mask.any():
        return mask
    r = max(2, round(frac * max(mask.shape)))
    return composite.shrink(composite.grow(mask, r), r) | mask


DROPOUT = 0.6  # a frame whose kept mask is under this share of its neighbours' lost part of it (one of a couple)
DROPOUT_REACH = 8  # frames either side that set what "its neighbours" are


def mend_dropouts(masks: list[np.ndarray], cover: bool = False) -> list[np.ndarray]:
    """SAM sometimes loses part of what it follows for a few frames (one of a couple, for five frames on the dusk road),
    so it would vanish into the new weather and back. On those frames the lost part is put back, between the nearest
    whole frames before and after: moved there from the nearer one, in step (`cover` False: the matte it is pasted
    with, which must not carry old background along), or as the union of both (`cover` True: what is taken out of
    the shot, which may be generous). At the shot's ends it is a car driving in or out, and left alone."""
    areas = np.array([float(m.sum()) for m in masks])
    n = len(masks)
    if n < 3 or not areas.any():
        return masks
    usual = np.array([np.median(areas[max(0, i - DROPOUT_REACH):i + DROPOUT_REACH + 1]) for i in range(n)])
    short = areas < DROPOUT * usual
    if not short.any():
        return masks
    whole = [i for i in range(n) if not short[i]]
    out = list(masks)
    for i in np.flatnonzero(short):
        before = max((j for j in whole if j < i), default=None)
        after = min((j for j in whole if j > i), default=None)
        if before is None or after is None:  # a car driving into or out of frame grows or shrinks: not a dropout
            continue
        here = composite.grow(masks[i], 3)
        lost_b, lost_a = masks[before] & ~here, masks[after] & ~here
        if not lost_b.any() or not lost_a.any():
            out[i] = masks[i] | (lost_b if lost_b.any() else lost_a) | (masks[before] | masks[after] if cover else False)
            continue
        cb, ca = (np.array(np.nonzero(x)).mean(axis=1) for x in (lost_b, lost_a))
        t = (i - before) / (after - before)
        src, c0 = (lost_b, cb) if t <= 0.5 else (lost_a, ca)
        dy, dx = np.rint(cb + t * (ca - cb) - c0).astype(int)
        moved = composite.grow(composite.shift(src, int(dx), int(dy), fill=False), 2)  # a little give either way
        out[i] = composite.shrink(composite.grow(masks[i] | moved, 4), 4) | masks[i]  # no pinholes where they meet
        if cover:
            out[i] |= masks[before] | masks[after]
    return out


def upsize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A proxy mask at the render's size, its outline smoothed (no proxy-pixel stair steps along a wheel)."""
    if mask.shape[::-1] == tuple(size):
        return mask
    m = np.asarray(Image.fromarray(mask.astype(np.float32), "F").resize(size, Image.BILINEAR))
    return composite.blur(m, 0.6 * size[0] / mask.shape[1]) >= 0.5


def low_pass(x: np.ndarray, sigma: float, weight: np.ndarray | None = None) -> np.ndarray:
    """Gaussian blur, in floats. With `weight`, a blur of only the weighted pixels (normalised), carried a few times
    wider over gaps with none, so a kept thing's broad light is its own, with none of the background's in it."""
    if weight is None:
        return composite.smooth(x, sigma)
    wt = weight.astype(np.float32)

    def spread(s: float) -> tuple[np.ndarray, np.ndarray]:
        wb = composite.blur(wt, s)
        return composite.blur(x * wt[..., None], s) / np.maximum(wb, 1e-6)[..., None], wb

    near, w_near = spread(sigma)
    far, _ = spread(4 * sigma)
    mix = np.clip(w_near / 0.2, 0.0, 1.0)[..., None]
    return near * mix + far * (1 - mix)


def detail_transfer(orig: np.ndarray, gen: np.ndarray, sigma: float, weight: np.ndarray | None = None) -> np.ndarray:
    """The footage's fine detail on the render's broad light and colour: low(gen) + (orig - low(orig)), as floats."""
    o, g = orig.astype(np.float32), gen.astype(np.float32)
    return low_pass(g, sigma, weight) + (o - low_pass(o, sigma, weight))


def agreement(orig: np.ndarray, gen: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Where the render's copy of the kept thing can lend it its light (0..1): inside the mask, where the change from
    footage to render (with the fine detail's own differences blurred away) is close to the usual change over the
    thing. Where LTX drew it a little elsewhere (snow where the bumper is) the change jumps, and those pixels are left
    out; a hood that catches more of the new sky than the grille is a smaller step, and stays."""
    dy = low_pass(((gen.astype(np.float32) - orig.astype(np.float32)) @ LUM)[..., None], sigma, mask)[..., 0]
    v = dy[mask]
    if not v.size:
        return mask.astype(np.float32)
    med = float(np.median(v))
    reach = 3 * (1.4826 * float(np.median(np.abs(v - med))) + 10.0)  # a robust spread, never under 30 levels
    return mask * np.clip(1 - ((dy - med) / reach) ** 2, 0.0, 1.0) ** 2  # Tukey's weights: none at all past `reach`


def rank(x: np.ndarray, n: int, fn=np.maximum) -> np.ndarray:
    """A square max (or min, fn=np.minimum) filter n px wide, one axis at a time: quick at any size, unlike PIL's."""
    r = n // 2
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r, r)
        p, size = np.pad(x, pad, mode="edge"), x.shape[axis]
        x = fn.reduce([p[i:i + size] if axis == 0 else p[:, i:i + size] for i in range(n)])
    return x


def specks(y: np.ndarray, size: int) -> np.ndarray:
    """What is lighter than its surroundings and smaller than `size` px (a white top-hat of luma `y`)."""
    return y - rank(rank(y, size, np.minimum), size)


def weather_in_front(gen: np.ndarray, kept: np.ndarray, mask: np.ndarray, k: float = 1.0,
                     around: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """How much of the render's weather passes in front of the kept thing (0..1 per pixel): falling flakes and streaks
    (small specks lighter and whiter than anything of its own there), and where it meets the ground, the render's snow
    or spray over its lowest part. `around` (the render's frames before and after) tells falling snow from the render's
    own copy of the thing: a speck there on both is stuck to it (snow on a grille, a wheel's rim), and is left out."""
    g = gen.astype(np.float32)
    yg, yk = g @ LUM, kept.astype(np.float32) @ LUM
    size = 2 * max(1, round(4 * k)) + 1

    still = rank(specks(yk, size), 3)  # its own glints, a little wider
    if around is not None:
        reach = 2 * max(1, round(6 * k)) + 1  # how far the thing itself moves in a frame
        near = [rank(specks(f.astype(np.float32) @ LUM, size), reach) for f in around]
        still = np.maximum(still, np.minimum(*near))
    sat = (g.max(-1) - g.min(-1)) / np.maximum(g.max(-1), 1.0)
    white = np.clip((0.3 - sat) / 0.15, 0.0, 1.0)
    lighter = yg - yk
    flakes = np.clip((specks(yg, size) - still - 10) / 20, 0.0, 1.0) * white * (lighter > 0)
    # No ground spray over the thing: the render's snow over its lowest part washed a car's wheels see-through.
    return flakes


def keep_over(orig: np.ndarray, gen: np.ndarray, mask: np.ndarray, k: float = 1.0,
              around: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """One frame: the kept thing (`mask`, at the frame's size) from the footage, in the render's light, over the render,
    with the render's weather in front of it. Returns (frame, how much of each pixel is the footage 0..1, the blur's
    sigma in px)."""
    h, w = mask.shape
    if not mask.any():
        return gen.copy(), np.zeros((h, w), np.float32), 0.0
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    sigma = max(LOW_MIN * k, LOW_SIGMA * (ys[-1] - ys[0] + 1))
    # Worked out on a box around the matte, not the whole frame (every blur here only reads the mask's own pixels).
    m = int(sigma + 4 * FEATHER * k + GROW * k + 16)
    y0, y1, x0, x1 = max(0, ys[0] - m), min(h, ys[-1] + 1 + m), max(0, xs[0] - m), min(w, xs[-1] + 1 + m)
    box = (slice(y0, y1), slice(x0, x1))
    mb, o, g = mask[box], orig[box].astype(np.float32), gen[box].astype(np.float32)
    # The broad light is learnt only where the render's copy agrees with the footage.
    kept = detail_transfer(o, g, sigma, weight=agreement(o, g, mb, sigma))
    # The render's light, but never far from its usual shift over the thing: where LTX drew the car a little elsewhere
    # (a spinning wheel, the sills in spray) its broad light was the snow's, and the footage's car showed through it
    # like a ghost.
    shift = kept - o
    usual = np.median(shift[mb], axis=0)
    kept = o + np.clip(shift, usual - LIGHT_RANGE, usual + LIGHT_RANGE)
    a = composite.soft_matte(mb, round(GROW * k), FEATHER * k)
    wf = weather_in_front(g, kept, mb, k, None if around is None else (around[0][box], around[1][box])) * (a > 0)
    kept = kept + wf[..., None] * (g - kept)
    out = gen.copy()
    out[box] = np.clip(np.rint(g + a[..., None] * (kept - g)), 0, 255).astype(np.uint8)
    own = np.zeros((h, w), np.float32)
    own[box] = a * (1 - wf)
    return out, own, float(sigma)


# --- keep exactly as shot, when it moves through the frame -------------------------------------------------------------
#
# A car crossing a field: told to keep it, LTX draws its own copy from the depth guide, and that copy runs a car-length
# ahead of the real one, so the take shows two cars. So a kept thing that travels through the frame is first taken out
# of the shot with Remove's clean plate (real pixels from the frames where the road was visible), the weather is
# rendered on that empty shot (no car in the guide, the first frame or the prompts), and the real thing is put back in
# the new light: the change from the clean plate to the render around it, as one gain per colour in linear light, so
# its own contrast and highlights stay as filmed. Its shadow comes back as the footage's own darkening of the plate.

TRAVELS = 0.45  # its usual size over all it covers in the shot: below this it moves through the frame (and is cleared)
STILL_RESIDUAL = 12.0  # levels apart, aligned, that the background may differ and still lend its real pixels
STILL = 0.02    # ...on a camera that moves at most this share of the frame (a tripod, a slow drift): the road behind it
                # is then real footage from other frames, aligned. On a moving camera a car keeps the render's own copy
                # (rigid, it lines up); a person is always cleared.
NEAR = 1.5      # the light is learnt from the plate and the render this many of its sizes around it
GAIN_RANGE = (1 / 10, 2.0)  # how much darker or lighter than as filmed the new light may make it (linear)
TINT_MAX = 0.2   # how far one colour's gain may stray from the others' (log): warmer or colder light, never green
TINT_SAME = 0.35  # a pixel whose colours changed more unevenly than this (log) is a new material, not a new light
LOCAL_SAME = 0.25  # the share of pixels around it that must still be the same materials to learn the light from them
FRAME_SAME = 0.08  # ...or, across the whole frame (sky, hills, forest), else no tint at all
HAZE_MAX = 0.3   # the most veil (fog, falling snow) laid over it, as a share of the light around it
THIN = 0.06  # of its height: the outline is trusted only this far inside; nearer the edge the picture decides
NOTCH_REACH = 0.05  # of its height: a person's outline takes in background (a neck's notch) only this near its edge
UNMIX_REACH = 0.01  # of its height: how far past SAM's outline its own soft edge (blur, hair) may still be some of it
UNMIX_MAX = 40.0  # levels an edge pixel's colour may move when what was behind it is only a guess (no real plate)
SAME_AS_PLATE = 6.0  # levels: a pixel inside the outline this close to the clean plate is background showing through
SHADE_EDGE = 0.015  # of its height: how far in from its edge the reference's light is laid in full
WRAP_WIDTH = 0.01  # of its height: the blur that sets how deep the wrap goes
WRAP = 0.2       # how much of the light around it spills over its very edge (0.25 read as a faint light rim on a dark coat
                 # against a wet road the render drew brighter than the footage's)
LIGHT_SMOOTH = 4  # frames either side the light is averaged over (a steady light, no flicker)
ATMOSPHERE = ("fog", "snow", "rain")  # presets that lay a veil over far things


def travels(masks: list[np.ndarray]) -> bool:
    """True when the kept thing moves through the frame (a car crossing a field) rather than staying in its place
    (a car the camera tracks): its usual size is a small part of everything it covers over the shot."""
    found = [m for m in masks if m.any()]
    if not found:
        return False
    union = np.logical_or.reduce(found)
    return float(np.mean([m.sum() for m in found])) < TRAVELS * float(union.sum())


def camera_still(work: Path, masks: list[np.ndarray], long: int = 640) -> bool:
    """True when the background behind the kept thing is the same from frame to frame once a slow drift is aligned
    away, so other frames can lend it real pixels: a tripod (0-2 levels apart) or a slow pan over far hills (about 9),
    not a camera dollying after a couple (17: the hills grow, and translation can't line them up). Measured outside
    the kept thing between the first frame and the middle and last ones."""
    pr = media.probe(work)
    sw, sh = media.work_size(pr.width, pr.height, long)
    fr = list(common.read_frames(work, f"scale={sw}:{sh}", (sw, sh)))
    if len(fr) < 2:
        return True
    shifts = composite.frame_shifts(fr)
    if max(float(np.hypot(dx, dy)) for dx, dy in shifts) > STILL * sw:
        return False
    ms = [composite.grow(upsize(m, (sw, sh)), 8) for m in masks] or [np.zeros((sh, sw), bool)]
    worst = 0.0
    for j in (len(fr) // 2, len(fr) - 1):
        dx, dy = shifts[j]
        back = composite.shift(fr[j], -dx, -dy)
        valid = (composite.shift(np.ones((sh, sw), bool), -dx, -dy, fill=False) & ~ms[0]
                 & ~composite.shift(ms[min(j, len(ms) - 1)], -dx, -dy, fill=True))
        if valid.any():
            worst = max(worst, float(np.median(np.abs(back.astype(np.float32) - fr[0]).max(-1)[valid])))
    return worst <= STILL_RESIDUAL


def write_clip(frames, path: Path, size: tuple[int, int], fps: float, audio_from: Path) -> None:
    """`frames` as a near-lossless H.264 with `audio_from`'s sound (LTX's graph reads a sound track)."""
    silent = path.with_name(path.stem + "_silent.mp4")
    rate = media.probe(audio_from).fps  # the work clip's exact rate (30000/1001, not 29.97002997)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                             "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-", "-c:v", "libx264", "-crf", "2", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                             str(silent)], stdin=subprocess.PIPE)
    assert proc.stdin
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("Couldn't write the shot without the kept thing.")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(audio_from), "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "copy", str(path)], check=True)  # every frame: -shortest cut 97 to 95 at 29.97
    silent.unlink(missing_ok=True)


def clear_shot(ctx, work: Path, kept: list[np.ndarray], size: tuple[int, int], fps: float, run_dir: Path, seed: int,
               still: bool = True):
    """The work clip with the kept thing (and its cast shadow) gone, by Remove's clean plate: on a `still` camera real
    pixels from the frames where it has moved on, aligned for a slow drift (a frame-editor plate fills only what is
    never seen); on a moving one Remove's LTX in/outpaint. Returns (the cleared clip, the Remove background: `locked`
    when its pixels are real)."""
    from pickups.tasks import remove

    d = run_dir / "clear"
    d.mkdir(exist_ok=True)
    bg = remove.background(ctx, work, kept, None, size, fps, d, seed=seed, shadow=True, camera="locked" if still else "moving",
                           never_fill="clone")
    clip = run_dir / "control.mp4"
    write_clip((bg.plate(i, f)[0] for i, f in enumerate(common.read_frames(work))), clip, size, fps, work)
    return clip, bg


GHOST_MIN = 0.0004  # of the frame: a smaller find is a speck, not a second car or person
GHOST_MAX = 0.25    # ...and a bigger one is SAM's whole-frame "nothing found", not a car


THIN_WORDS = ("cable", "gondola", "tram", "lift", "wire", "fence", "gate", "railing", "bridge", "pylon", "tower")


def thin_kind(keep_text: str) -> bool:
    """A kept thing with real gaps through it (a cable car's hanger bars, a fence): matted on its own picture. Anything
    else (a person, a car) is solid, and inside its matte every pixel is the footage's own."""
    words = sam_words(keep_text).lower()
    return any(w in words for w in THIN_WORDS)


def ghost_label(keep_text: str) -> str:
    """What SAM looks for in the render to catch a copy LTX drew anyway: its kind ("person", "car"), else its words."""
    words = sam_words(keep_text)
    if not words:
        return ""
    if common.PERSON.search(words) or common.MANY.search(words):
        return "person"
    if set(words.lower().split()) & VEHICLES:
        return "car"
    return words


def ghost_masks(found: list[np.ndarray], real: list[np.ndarray], k: float = 1.0) -> list[np.ndarray]:
    """What the render has of that kind that the shot (without the kept thing) doesn't: LTX's own invention, a second
    car or a walker in the distance. `found` on the render, `real` on the cleared shot, same size; finds smaller than
    GHOST_MIN of the frame are left alone."""
    out = []
    for i, f in enumerate(found):
        r = real[i] if i < len(real) and real[i].shape == f.shape else np.zeros_like(f)
        # SAM's answer when it finds nothing can be the whole frame: no car fills a quarter of a weather shot.
        f = f if f.mean() < GHOST_MAX else np.zeros_like(f)
        r = r if r.mean() < GHOST_MAX else np.zeros_like(r)
        g = f & ~composite.grow(r, max(2, round(6 * k)))
        rr = max(1, round(2 * k))
        g = composite.grow(composite.shrink(g, rr), rr + 1) & g  # specks and slivers along the real ones' edges go
        keep = np.zeros_like(g)
        for comp, (x0, y0, x1, y1) in blobs(g):
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            # A person or a car is a compact blob: a band along the horizon is a tree line SAM misread.
            if comp.sum() >= GHOST_MIN * g.size / 4 and bw <= 0.5 * g.shape[1] and bw <= 5 * bh:
                keep |= comp
        out.append(composite.grow(keep, max(2, round(4 * k))) if keep.any() else keep)
    return out


def moving_only(ghosts: list[np.ndarray], most: float = 0.5) -> list[np.ndarray]:
    """The ghosts without what is found in the same place on most frames: a walker LTX invents walks, a clump of trees
    SAM takes for people stays put (and filling it would cut a hole in the hillside)."""
    if not ghosts or not any(g.any() for g in ghosts):
        return ghosts
    static = np.mean(np.stack(ghosts), axis=0) > most
    out = []
    for g in ghosts:
        g = g & ~static
        keep = np.zeros_like(g)
        for comp, _ in blobs(g) if g.any() else []:  # the static find's flickering fringe is no walker either
            if comp.sum() >= GHOST_MIN * g.size / 4:
                keep |= comp
        out.append(keep)
    return out


def blobs(mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """The mask's connected parts (4-neighbour), each with its box (x0, y0, x1, y1). For small proxy masks."""
    seen = np.zeros_like(mask, bool)
    h, w = mask.shape
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, pts = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = np.array(pts).T
        comp = np.zeros_like(mask, bool)
        comp[ys, xs] = True
        out.append((comp, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return out


def fill_ghosts(frames: list[np.ndarray], ghosts: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """The render with the ghosts filled from its own frames where they have moved on (the camera is still: that is
    when a kept thing is cleared). Where one never moves off, it is left as rendered. Returns (frames, frames changed)."""
    if not any(g.any() for g in ghosts):
        return frames, 0
    filled, never = composite.temporal_fill(frames, ghosts)
    out, changed = [], 0
    for f, fl, g, nv in zip(frames, filled, ghosts, never):
        take = g & ~nv
        if take.any():
            f = f.copy()
            f[take] = fl[take]
            changed += 1
        out.append(f)
    return out, changed


def clear_ghosts(ctx, gen_files: list[Path], plate: Path, label: str, fps: float, run_dir: Path) -> int:
    """The safety net after a render made without the kept thing: SAM looks for its kind on the render and on the
    cleared shot; whatever the render has that the shot doesn't is filled away (fill_ghosts), in place. Returns how
    many frames had one."""
    if not label or not gen_files:
        return 0
    d = run_dir / "ghosts"
    d.mkdir(exist_ok=True)
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in gen_files]
    h, w = frames[0].shape[:2]
    pw, ph = media.work_size(w, h, common.PROXY_LONG)
    rclip = d / "render.mp4"
    write_clip((np.asarray(Image.fromarray(f).resize((pw, ph), Image.BILINEAR)) for f in frames), rclip, (pw, ph), fps, plate)
    pclip = d / "plate.mp4"
    media.prepare_work_clip(plate, pclip, 0, len(frames), (pw, ph))
    found = []
    for name, clip in (("render", rclip), ("plate", pclip)):
        job = ctx.run(graphs.select(ctx.client.assets.from_file(clip), label, threshold=0.4, prefix=f"pickups/ghosts_{name}"))
        found.append(common._download_masks(job, d / name))
    small = moving_only(ghost_masks(*found, k=pw / 1920))
    if not any(g.any() for g in small):
        return 0
    big = [upsize(g, (w, h)) for g in small]
    fixed, n = fill_ghosts(frames, big)
    raw = run_dir / "gen_before_ghosts"  # the render as it came, beside the take
    raw.mkdir(exist_ok=True)
    for f, img, orig, g in zip(gen_files, fixed, frames, big):
        if g.any() and not np.array_equal(img, orig):
            Image.fromarray(orig).save(raw / f.name)
            Image.fromarray(img).save(f)
    return n


def to_linear(x: np.ndarray) -> np.ndarray:
    return (np.clip(x.astype(np.float32), 0, 255) / 255.0) ** 2.2


def to_srgb(x: np.ndarray) -> np.ndarray:
    return 255.0 * np.clip(x, 0, None) ** (1 / 2.2)


def scene_light(plate: np.ndarray, gen: np.ndarray, mask: np.ndarray, k: float = 1.0, haze: bool = False,
                shadow: np.ndarray | None = None, gain_range: tuple[float, float] = GAIN_RANGE,
                haze_max: float = HAZE_MAX) -> np.ndarray:
    """The new light around the kept thing, from the clean plate (the scene as filmed, without it) and the render (the
    same scene in the new weather): (gain RGB, veil RGB) in linear light, as a (2, 3) array, so that new ≈ gain *
    filmed + veil.

    Learnt from the pixels within NEAR of its sizes around it whose materials stayed the same (blurred a little, so
    texture the render redrew doesn't count), as medians: a shaded strip of road beside it is a few of them, not all.
    Frost that became green grass is a new material, not green light: when that is all there is around it, the light
    comes from the rest of the frame where materials stayed (a golden sky, the hills). The gain's colour is held near
    grey (TINT_MAX). A veil only when `haze` (fog, snow, rain), and never more than `haze_max` of the light around it.
    Snow that covers a grey rock brightens it without changing its colour, so the gain is held to `gain_range`: what
    that weather can do to the light (a preset's own "light", or GAIN_RANGE)."""
    h, w = mask.shape
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not ys.size:
        return np.array([[1.0] * 3, [0.0] * 3], np.float32)
    r = max(48 * k, NEAR * max(ys[-1] - ys[0] + 1, xs[-1] - xs[0] + 1))
    y0, y1, x0, x1 = (max(0, int(ys[0] - r)), min(h, int(ys[-1] + 1 + r)), max(0, int(xs[0] - r)), min(w, int(xs[-1] + 1 + r)))
    box = (slice(y0, y1), slice(x0, x1))
    near = ~composite.grow(mask[box], max(2, round(4 * k)))
    if shadow is not None:
        near &= ~shadow[box]
    s = max(1.0, 1.5 * k)
    pl, gl = to_linear(composite.smooth(plate[box].astype(np.float32), s)), to_linear(composite.smooth(gen[box].astype(np.float32), s))
    pl, gl = pl[near], gl[near]
    py, gy = pl @ LUM, gl @ LUM
    ok = (py > 0.002) & (gy > 0.001)
    if ok.sum() < 32:
        return np.array([[1.0] * 3, [0.0] * 3], np.float32)
    pl, gl = pl[ok], gl[ok]
    # Only pixels whose material stayed (road, rock, sky) say what the light did: frost that became grass, grass that
    # became snow, changed colour unevenly and are left out. Too few of them around it: the rest of the frame's.
    _, _, same = light_change(pl, gl)
    if same.mean() < LOCAL_SAME:
        sub = max(1, round(w / 320))
        far = ~composite.grow(mask, max(2, round(4 * k)))[::sub, ::sub]
        if shadow is not None:
            far &= ~shadow[::sub, ::sub]
        fp = to_linear(composite.smooth(plate.astype(np.float32), s)[::sub, ::sub][far])
        fg = to_linear(composite.smooth(gen.astype(np.float32), s)[::sub, ::sub][far])
        good = ((fp @ LUM) > 0.002) & ((fg @ LUM) > 0.001)
        fp, fg = fp[good], fg[good]
        _, _, f_same = light_change(fp, fg)
        if f_same.size and f_same.mean() >= FRAME_SAME:
            pl, gl, same = fp, fg, f_same
    if not same.any() or same.mean() < FRAME_SAME:
        # No telling: its exposure from around it (bounded), no colour, no veil.
        ratio, _, _ = light_change(pl, gl)
        e = float(np.clip(np.median(ratio), *np.log(gain_range)))
        return np.array([[np.exp(e)] * 3, [0.0] * 3], np.float32)
    pl, gl = pl[same], gl[same]
    py, gy = pl @ LUM, gl @ LUM
    veil = np.zeros(3, np.float32)
    slope = 1.0
    if haze:
        # new = a * filmed + b on the light, fitted to the two sides' tones (quantile against quantile): b is the veil.
        q = np.linspace(0.05, 0.95, 19)
        qp, qg = np.quantile(py, q), np.quantile(gy, q)
        slope, bb = np.polyfit(qp, qg, 1) if np.ptp(qp) > 0.02 else (1.0, 0.0)
        bb = float(min(max(0.0, bb), haze_max * float(np.median(gy)))) if slope > 0 else 0.0
        dark = py <= np.quantile(py, 0.2)
        colour = np.median(gl[dark], axis=0) / max(float(np.median(gy[dark])), 1e-6)
        veil = (bb * np.clip(colour, 0.7, 1.4)).astype(np.float32)
        gl = np.maximum(gl - veil, 1e-4)
    ratio, shift, _ = light_change(pl, gl)
    e = float(np.log(slope)) if veil.any() else float(np.median(ratio))
    e = float(np.clip(e, *np.log(gain_range)))
    tint = np.clip(np.median(shift, axis=0), -TINT_MAX, TINT_MAX)
    tint = np.clip(tint - tint @ LUM, -TINT_MAX, TINT_MAX)
    return np.array([np.exp(e + tint), veil], np.float32)


def light_change(pl: np.ndarray, gl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel (N x 3 linear, filmed and new): the change in light (log), the change in colour beyond it (log, grey
    removed), and whether that colour change is a light's (small and even), not a new material's."""
    ratio = np.log(np.maximum(gl @ LUM, 1e-5) / np.maximum(pl @ LUM, 1e-5))
    shift = np.log(np.maximum(gl, 1e-5) / np.maximum(pl, 1e-5)) - ratio[:, None]
    shift -= (shift @ LUM)[:, None]
    return ratio, shift, np.abs(shift).max(axis=1) < TINT_SAME


def steady(lights: list[np.ndarray | None]) -> list[np.ndarray]:
    """Each frame's light, averaged over LIGHT_SMOOTH frames either side (frames without it borrow their neighbours'):
    the light on a car doesn't flicker from frame to frame, even where the render's grass does."""
    have = [i for i, x in enumerate(lights) if x is not None]
    if not have:
        return [np.array([[1.0] * 3, [0.0] * 3], np.float32)] * len(lights)
    out = []
    for i in range(len(lights)):
        near = [lights[j] for j in have if abs(j - i) <= LIGHT_SMOOTH] or [lights[min(have, key=lambda j: abs(j - i))]]
        out.append(np.mean(near, axis=0).astype(np.float32))
    return out


def relight(orig: np.ndarray, light: np.ndarray, shade: np.ndarray | None = None) -> np.ndarray:
    """The footage in the new light: gain and veil in linear light (its contrast and highlights scale with it, so it
    never goes flat or black); highlights pushed past white roll off instead of clipping. `shade`: the broad light the
    render's own drawing of the thing adds (subject_light, a log ratio per pixel and colour). Floats, 0..255."""
    lin = to_linear(orig) * light[0] + light[1]
    if shade is not None:
        lin = lin * np.exp(shade)
    knee = 0.85
    over = lin > knee
    lin[over] = knee + (1 - knee) * (1 - np.exp(-(lin[over] - knee) / (1 - knee)))
    return to_srgb(lin)


def shadow_back(orig: np.ndarray, plate: np.ndarray, gen: np.ndarray, mask: np.ndarray, k: float = 1.0) -> np.ndarray:
    """The render with the kept thing's own shadow laid on it: where the footage is darker than the clean plate near it
    (its contact shadow on the road), the render is darkened by the same ratio. Floats."""
    ys = np.flatnonzero(mask.any(axis=1))
    if not ys.size:
        return gen.astype(np.float32)
    tall = ys[-1] - ys[0] + 1
    # Its own contact shadow lies at and below its lowest part; higher up, footage darker than the plate beside it is
    # its motion blur or the plate's own guess, and darkening the render there drew a dark smear beside a cable car.
    low = np.zeros_like(mask)
    low[max(0, ys[-1] - int(0.3 * tall)):] = True
    reach = composite.grow(mask, max(3, round(0.6 * tall))) & ~mask & low
    s = max(1.0, 1.5 * k)
    yo = to_linear(composite.smooth(orig.astype(np.float32), s)) @ LUM
    yp = to_linear(composite.smooth(plate.astype(np.float32), s)) @ LUM
    ratio = np.clip((yo + 1e-3) / (yp + 1e-3), 0.2, 1.0)
    ratio = np.where(reach & (ratio < 0.94), ratio, 1.0)
    ratio = composite.smooth(ratio[..., None], s)[..., 0]
    return to_srgb(to_linear(gen) * ratio[..., None])


def unmix(orig: np.ndarray, plate: np.ndarray, mask: np.ndarray, k: float = 1.0,
          see_through: bool = True, cap: float | None = None, notch: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The kept thing's own share of each pixel near its outline (alpha 0..1) and its own colour there, from the
    footage and the clean plate behind it: footage = alpha * thing + (1 - alpha) * plate, with the thing's colour taken
    from just inside its outline. Where the two are too alike to tell apart, SAM's outline (softened) decides."""
    o, b = orig.astype(np.float32), plate.astype(np.float32)
    rows = np.flatnonzero(mask.any(axis=1))
    tall = rows[-1] - rows[0] + 1 if rows.size else 0
    # SAM's outline sits anywhere on a soft edge; on a small far car a wide edge band would soften the whole of it
    core = composite.shrink(mask, max(2, min(round(4 * k), round(0.06 * tall))))
    band = composite.grow(mask, max(1, round(2 * k), round(UNMIX_REACH * tall))) & ~core  # out as far as its own blur reaches
    if not core.any():
        return composite.soft_matte(mask, round(GROW * k), FEATHER * k), o
    f_est = low_pass(o, max(1.0, 1.5 * k), core)  # the thing's colour, carried out over its edge
    d, fb = o - b, f_est - b
    contrast = (fb * fb).sum(-1)
    alpha = np.clip((d * fb).sum(-1) / np.maximum(contrast, 1.0), 0.0, 1.0)
    soft = composite.blur(mask.astype(np.float32), 0.7 * max(1.0, k))
    alpha = np.where(contrast < 20.0 ** 2, soft, alpha)
    # Inside the outline, what is exactly the plate is the background seen through it (grass between a cable car's
    # hanger bars, sky through a window): not the thing.
    # (averaged over the thing's own pixels only: the plate-equal ground just outside must not wear its edge away)
    seen = low_pass(np.abs(d).max(axis=-1)[..., None], 0.7 * max(1.0, k), mask)[..., 0]
    through = np.clip((seen - SAME_AS_PLATE) / 10.0, 0.0, 1.0)
    # Only real openings (a gap between bars), not specks where glass happens to match: those read as holes.
    r = max(2, round(2 * k))
    opening = composite.grow(composite.shrink(through < 0.5, r), r)
    through = np.where(opening | ~core, through, 1.0)
    if not see_through and notch:
        # A person has no gaps; a white T-shirt in front of white cloud only looks like one. But the notch beside a neck,
        # or the space between an arm and the body, near their outline, is background when it equals the plate: SAM's
        # outline took it in, and the thing's brighter light on it drew a pale smudge there.
        near_edge = ~composite.shrink(mask, max(2, round(NOTCH_REACH * tall)))
        rn = max(2, round(2 * k))
        plate_like = through < 0.5
        notch_px = composite.grow(composite.shrink(plate_like, rn), rn) & plate_like & near_edge
        through = np.where(core & ~notch_px, 1.0, through)
    elif not see_through:  # a car or an object is solid: its core is the footage's own pixels, every one (the pixel lock)
        through = np.where(core, 1.0, through)
    alpha = np.where(core, through, np.where(band, np.minimum(alpha, through), 0.0)).astype(np.float32)
    # Its colour: unmixed where it is most of the pixel, its nearby colour where it is a trace (dividing by a small
    # share would blow the old background's grain up into a rim).
    sure = np.clip((alpha - 0.2) / 0.3, 0.0, 1.0)[..., None]
    unmixed = b + d / np.maximum(alpha, 0.2)[..., None]
    step = unmixed - o
    reach = np.linalg.norm(step, axis=-1, keepdims=True)
    if cap is not None:  # what was behind is a guess (no real plate): a wrong one mustn't throw the edge to white
        unmixed = o + step * np.minimum(1.0, cap / np.maximum(reach, 1e-6))
    fg = sure * np.clip(unmixed, 0, 255) + (1 - sure) * f_est
    fg = np.where(core[..., None], o, fg)
    alpha = composite.blur(alpha, 0.6 * max(1.0, k)).astype(np.float32)  # no stair steps along it
    return alpha, fg


def guided_colour(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter with a colour guide (HxWx3, 0..1): `src` (HxW) takes the guide's edges, colour by colour,
    so a dark grey bar in front of dark green trees still separates."""
    mi = [box_mean(guide[..., c], r) for c in range(3)]
    mp = box_mean(src, r)
    cov = np.stack([box_mean(guide[..., c] * src, r) - mi[c] * mp for c in range(3)], -1)
    var = np.empty(guide.shape[:2] + (3, 3), np.float32)
    for a in range(3):
        for b in range(3):
            var[..., a, b] = box_mean(guide[..., a] * guide[..., b], r) - mi[a] * mi[b]
    var += eps * np.eye(3, dtype=np.float32)
    A = np.linalg.solve(var, cov[..., None])[..., 0]
    B = mp - sum(A[..., c] * mi[c] for c in range(3))
    return sum(box_mean(A[..., c], r) * guide[..., c] for c in range(3)) + box_mean(B, r)


def thin_matte(orig: np.ndarray, plate: np.ndarray, mask: np.ndarray, k: float = 1.0) -> np.ndarray:
    """The kept thing's matte (0..1) when it has thin parts (bars, struts): SAM's outline refined on the footage itself
    with a colour guided filter, so it follows each bar; what still equals the clean plate (real pixels from other
    frames) is background showing through and drops out; pinholes that punches in its own edge are closed again.
    Inside its solid core it is all the thing."""
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return mask.astype(np.float32)
    tall = rows[-1] - rows[0] + 1
    alpha = np.clip(guided_colour(orig.astype(np.float32) / 255, mask.astype(np.float32), max(2, round(3 * k)), 1e-3), 0, 1)
    diff = composite.blur(np.abs(orig.astype(np.float32) - plate.astype(np.float32)).max(-1), max(1.0, 0.7 * k))
    # The core is the thing unless a real opening in it is just the plate (the gap between a hanger's legs); a dark window
    # matching the dark trees behind it is a speck, smaller than an opening, and stays.
    r_open = max(2, round(3 * k))
    plate_like = diff < SAME_AS_PLATE
    opening = composite.grow(composite.shrink(plate_like, r_open), r_open) & plate_like
    sure = composite.shrink(mask, max(2, round(THIN * tall))) & ~opening
    alpha = np.where(sure, 1.0, alpha) * composite.grow(mask, max(2, round(4 * k)))
    gated = np.where(sure | (composite.shrink(mask, 1) & ~opening), alpha, alpha * np.clip((diff - SAME_AS_PLATE) / 12.0, 0.0, 1.0))
    r = max(2, round(2 * k))
    closed = composite.shrink(composite.grow(gated > 0.5, r), r)
    return np.where(closed & (gated <= 0.5), np.maximum(gated, alpha), gated).astype(np.float32)


def keep_relit(orig: np.ndarray, plate: np.ndarray, gen: np.ndarray, mask: np.ndarray, light: np.ndarray, k: float = 1.0,
               around: tuple[np.ndarray, np.ndarray] | None = None, real_plate: bool = True,
               flakes: bool = False, see_through: bool = True, shade: np.ndarray | None = None,
               person: bool = False, matte: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """keep_over for a render made without the kept thing in it: the footage's thing, relit by `light` (scene_light),
    with a little of the light around it wrapped over its edge, its own shadow from the footage (`real_plate`: the
    plate is all real pixels, so footage over plate is exactly its shadow), and (`flakes`: snow, rain) the render's
    falling weather in front. Without flakes nothing of the render is laid over it: the render has the ground there, not
    weather, and its specks would read as dirt on the car.
    Returns (frame, how much of each pixel is the footage, 0..1, the blur's sigma in px)."""
    h, w = mask.shape
    if not mask.any():
        return gen.copy(), np.zeros((h, w), np.float32), 0.0
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    sigma = max(LOW_MIN * k, LOW_SIGMA * (ys[-1] - ys[0] + 1))
    m = int(max(sigma + 4 * FEATHER * k + GROW * k + 16, 0.8 * (ys[-1] - ys[0] + 1)))
    y0, y1, x0, x1 = max(0, ys[0] - m), min(h, ys[-1] + 1 + m), max(0, xs[0] - m), min(w, xs[-1] + 1 + m)
    box = (slice(y0, y1), slice(x0, x1))
    mb, o, g = mask[box], orig[box], gen[box].astype(np.float32)
    if real_plate:
        g = shadow_back(o, plate[box], g, mb, k)
        # Its edge as filmed is part old background (frost behind a blurred bumper): unmixed against the real plate,
        # so only the car's share comes over and the frost doesn't ring it in white on the new grass.
        if see_through:  # a thing with gaps (a cable car's hanger): matted on its own picture, kept at full sharpness
            # Its own colour at the edge from unmixing; its matte from the picture. Its solid core stays solid even where
            # it happens to match the plate (dark cabin windows over dark trees punched holes in it).
            _, fg = unmix(o, plate[box], mb, k, see_through=False)
            a = thin_matte(o, plate[box], mb, k)
        else:
            a, fg = unmix(o, plate[box], mb, k, see_through, notch=person)
    else:
        # No real plate (a moving camera's was generated): the background behind its edge is taken from the footage
        # just outside it, carried in, so a matte a little too wide (SAM, or a mended frame) doesn't bring a rim of it.
        behind = low_pass(o.astype(np.float32), max(2.0, 3 * k), ~composite.grow(mb, 1))
        a, fg = unmix(o, behind, mb, k, see_through=False, cap=UNMIX_MAX, notch=person)
    if matte is not None:  # the matte voted on the footage itself replaces the one-frame estimate; the colour stays unmix's
        a = matte[box]
    # Where the matte says it is the thing (alpha over 0.97, whole by 1, a few px in from SAM's outline) the colour is the footage's own pixel, not an unmixed
    # estimate: the pixel lock. Only the soft rim, which is part old background, takes the decontaminated colour.
    own_w = (np.clip((a - 0.97) / 0.03, 0.0, 1.0) * composite.shrink(mb, max(2, round(3 * k))))[..., None]
    fg = own_w * o.astype(np.float32) + (1 - own_w) * fg
    if shade is not None:
        # Edge-aware: the reference's broad light is laid on the thing's inside only. Its edge pixels are part old background
        # (sky or hills behind a neck), and brightening them drew a pale halo round it; there they keep the scene's overall
        # light, and the shade fades in over SHADE_EDGE of its height.
        r = max(2, round(SHADE_EDGE * (ys[-1] - ys[0] + 1)))
        inside = composite.blur(composite.shrink(mb, r).astype(np.float32), max(1.0, r / 2))
        shade_b = shade[box] * inside[..., None]
    else:
        shade_b = None
    kept = relight(fg, light, shade_b)
    # Light wrap: just inside its edge, a little of the new background's light (blurred from outside only), over a width
    # that follows the thing's size.
    tall = ys[-1] - ys[0] + 1
    edge = np.clip(1 - composite.blur(mb.astype(np.float32), max(1.0, 1.5 * k, WRAP_WIDTH * tall)), 0, 1) * mb
    # Inside its matte every pixel is the footage's own (the pixel lock): the wrap lives only on the soft rim where the new
    # background already shows through, not over the solid part.
    edge = np.clip(edge * 2, 0, 1) * (1 - np.clip((a - 0.9) / 0.1, 0, 1))
    wrap_from = low_pass(g, 2 * max(1.0, 1.5 * k), ~composite.grow(mb, 1))
    kept = to_srgb(to_linear(kept) * (1 - WRAP * edge[..., None]) + to_linear(wrap_from) * (WRAP * edge[..., None]))
    wf = np.zeros_like(a)
    if flakes:
        wf = weather_in_front(g, kept, mb, k, None if around is None else (around[0][box], around[1][box])) * (a > 0)
    kept = kept + wf[..., None] * (g - kept)
    out = gen.copy()
    out[box] = np.clip(np.rint(g + a[..., None] * (kept - g)), 0, 255).astype(np.uint8)
    own = np.zeros((h, w), np.float32)
    own[box] = a * (1 - wf)
    return out, own, float(sigma)


# --- the shot's layout survives the restyle ---------------------------------------------------------------------------

# Said of the ground in general: naming what SAM found ("keep the path") had Qwen pave the meadow's faint mown track
# into a tarmac path, while saying nothing had it paint the frosty field's road over with grass.
LAYOUT_KEEP = ("Keep the layout of the ground exactly as it is: every road, track, path, verge, wall and building stays in "
               "the same place with the same shape and width, and each keeps its own kind of surface: tarmac stays tarmac, a "
               "faint worn track stays a faint worn track, grass stays grass. Do not add, widen or pave anything.")
LAYOUT = ("road", "path", "building")  # what must stay where it is: only season, light and weather change
LAYOUT_TRIES = 3  # restyles tried (seed, seed + 1, ...) until the layout holds
LAYOUT_TRIES_MAX = 6  # while every try has moved the layout (and the check is sure), a few more seeds; then it stops
                      # rather than render a reframed shot (the harbour in the rain: 0.15, 0.29, 0.40 and the people gone)
LAYOUT_MIN = 0.8  # each found thing's overlap with itself after the restyle (a few px of give): a verge turned to
                  # tarmac scored 0.73 on the dusk road, the restyle that kept it 0.84
LAYOUT_UNSURE = 0.35  # every try below this: the check itself is wrong (SAM called a whole field "road")
LAYOUT_SMALL = 0.003  # of the frame: smaller finds (a far shed) aren't checked


TRANSFER_RADIUS = 0.08  # of the frame's width: the light is taken from the restyle at this scale (edge-aware), so a
                        # track it paints in (about 2% wide) is averaged away


def box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 box, edges normalised."""
    ones = np.ones(x.shape[:2], np.float32)
    k = composite._box(composite._box(ones, r, 0), r, 1)
    y = composite._box(composite._box(x.astype(np.float32), r, 0), r, 1)
    return y / (k[..., None] if x.ndim == 3 else k)


FIELD_STEP = 8  # the light fields are smooth at TRANSFER_RADIUS, so they are kept at 1/8 size
FIELD_SMOOTH = 4  # frames either side the fields are averaged over: LTX's light steps between its latent chunks


def light_fields(orig: np.ndarray, restyled: np.ndarray, radius: int, gain_max: float = 4.0,
                 texture: float = 1.0) -> np.ndarray:
    """The render's light as fields over the footage, (2, h/8, w/8, 3): gain and veil per colour in linear light, so
    that render ~ gain x footage + veil near each pixel (a guided filter with the footage as its own guide). Two
    regions in one window (a pink sky over a grey ridge) map exactly; fog is the veil. Where the render redrew a texture
    that doesn't match the footage's (grass, a crowd it re-imagined), the gain is the ratio of their local contrasts
    (times `texture`: fog hides a far sky's clouds, golden light doesn't hide grass)."""
    o = to_linear(orig)
    q = to_linear(restyled if restyled.shape == orig.shape else
                  np.asarray(Image.fromarray(restyled).resize(orig.shape[1::-1], Image.LANCZOS)))
    fields = np.empty((2, *orig.shape[:2], 3), np.float32)
    for c in range(3):
        oc, qc = o[..., c], q[..., c]
        mo, mq = box_mean(oc, radius), box_mean(qc, radius)
        vo = np.maximum(box_mean(oc * oc, radius) - mo * mo, 1e-7)
        vq = np.maximum(box_mean(qc * qc, radius) - mq * mq, 0.0)
        cov = box_mean(oc * qc, radius) - mo * mq
        r = np.clip(cov / np.sqrt(vo * np.maximum(vq, 1e-7)), -1.0, 1.0)
        gain = np.clip(np.sqrt(vq / vo) * (r ** 3 + texture * (1 - r * r)), 0.0, gain_max)
        fields[0, ..., c] = box_mean(gain, radius)
        fields[1, ..., c] = box_mean(mq - gain * mo, radius)
    return fields[:, FIELD_STEP // 2::FIELD_STEP, FIELD_STEP // 2::FIELD_STEP]


def apply_fields(orig: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """The footage in the light of `fields` (light_fields, upsampled smoothly)."""
    h, w = orig.shape[:2]
    up = [np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f[..., c])).resize((w, h), Image.BILINEAR))
                    for c in range(3)], -1) for f in fields]
    return np.clip(np.rint(to_srgb(up[0] * to_linear(orig) + up[1])), 0, 255).astype(np.uint8)


def light_transfer(orig: np.ndarray, restyled: np.ndarray, radius: int, gain_max: float = 4.0,
                   texture: float = 1.0) -> np.ndarray:
    """The original frame in the render's light, fog and colour, every pixel of it still the footage's detail; nothing
    the render drew finer than `radius` (a track it paved, a person it redrew) comes through."""
    return apply_fields(orig, light_fields(orig, restyled, radius, gain_max, texture))


EMISSIVE_MIN = 0.02  # linear light a render's lamp or window must stand above its surroundings to be added
LAMP_BLOOM = 0.5  # how strong a lamp's halo is
LAMP_MIN = 0.3  # linear: by night a lamp or a lit window in the render is at least this bright
EMISSIVE_RADIUS = 0.006  # of the frame's width: what counts as "its surroundings" for a lamp


FOG_EXTRA = 0.7  # how much further than the render's own thin fog the relit footage is sunk into the fog colour


def thicken_fog(relit: np.ndarray, fields: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Relit footage under denser fog: where the render's veil (fields[1]) is heaviest (the distance, the sky) it is
    blended furthest toward the fog colour, by FOG_EXTRA of the render's own veil, normalised. LTX draws fog thin;
    a user who picks Fog expects to see it. Near, clear things (the quay, its people) stay readable."""
    h, w = relit.shape[:2]
    veil = np.asarray(Image.fromarray(np.ascontiguousarray(fields[1] @ LUM)).resize((w, h), Image.BILINEAR))
    dens = (FOG_EXTRA * np.clip(veil / max(float(np.quantile(veil, 0.95)), 1e-3), 0.0, 1.0))[..., None]
    lin = to_linear(relit) * (1 - dens) + colour * dens
    return np.clip(np.rint(to_srgb(lin)), 0, 255).astype(np.uint8)


def add_lights(relit: np.ndarray, render: np.ndarray, radius: int, allow: np.ndarray | None = None) -> np.ndarray:
    """The render's own small lights (lit windows, street lamps) over the relit footage: what in the render stands
    brighter than its surroundings by more than EMISSIVE_MIN, added as light (it never darkens or replaces anything).
    `allow` (HxW 0..1): where a light may be; then only a bright, warm one counts (a lamp, a lit window; not a white
    lane line or the glow of a sky behind a roof)."""
    g = to_linear(render)
    glow = np.clip(g - box_mean(g, radius) - EMISSIVE_MIN, 0.0, None)
    if allow is not None:
        warm = np.clip((g[..., 0] / np.maximum(g[..., 2], 1e-4) - 1.1) / 0.3, 0, 1)
        lamp = np.clip((g @ LUM - LAMP_MIN) / LAMP_MIN, 0, 1) * np.maximum(warm, np.clip((g @ LUM - 0.6) / 0.2, 0, 1))  # a white-hot core too
        glow = np.maximum(glow, 0.8 * g) * (allow * lamp)[..., None]  # a lamp is light through and through, not a ring
        glow = glow + LAMP_BLOOM * composite.blur(glow, max(2.0, 0.004 * glow.shape[1])) * 8  # its halo in the night air
    return np.clip(np.rint(to_srgb(to_linear(relit) + glow)), 0, 255).astype(np.uint8)


# --- night: a moonlit grade of the footage, the render's lamps and windows, a soft fill on people --------------------
#
# Learnt from a night render, the light came out in blotches and left faces black. Night is a grade instead: the
# footage a few stops down, its colour drained toward moonlight blue; the sky a deep night blue that keeps the
# clouds' shapes; people lifted by a soft, warmer fill so their faces read; then the render's own lamps and lit
# windows laid on as light (add_lights).

MOON_GAIN = 0.12  # linear: about three stops down
MOON_TINT = np.array([0.72, 0.9, 1.3], np.float32)  # moonlight's cold cast
MOON_GREY = 0.6  # how much of the colour drains to grey (the eye sees little colour by moonlight)
FILL_GAIN = 0.32  # people: about a stop and a half above the scene
FILL_TINT = np.array([1.0, 0.96, 0.92], np.float32)
FILL_GREY = 0.3
NIGHT_SKY = np.array([0.004, 0.008, 0.026], np.float32)  # linear: deep night blue
NIGHT_LABELS = ("sky", "person", "window", "street lamp")
WINDOW_GLOW = np.array([0.30, 0.17, 0.06], np.float32)  # linear: a lit window's warm light, as the footage's own windows


def moon_grade(pic: np.ndarray, sky: np.ndarray, people: np.ndarray) -> np.ndarray:
    """The picture (uint8) by moonlight: `sky` and `people` (HxW 0..1) say where the night sky and the fill go."""
    lin = to_linear(pic)
    lum = (lin @ LUM)[..., None]
    moon = ((1 - MOON_GREY) * lin + MOON_GREY * lum) * MOON_TINT * MOON_GAIN
    fill = ((1 - FILL_GREY) * lin + FILL_GREY * lum) * FILL_TINT * FILL_GAIN
    s, pe = sky[..., None], people[..., None]
    level = float(np.median(lum[sky > 0.5])) if (sky > 0.5).any() else float(lum.mean())
    night_sky = NIGHT_SKY * np.clip(0.4 + 0.6 * lum / max(level, 1e-4), 0.2, 2.0)  # the clouds' shapes, dark
    out = moon * (1 - pe) + fill * pe
    out = out * (1 - s) + night_sky * s
    return np.clip(np.rint(to_srgb(out)), 0, 255).astype(np.uint8)


def night_masks(ctx, picture: Path, run_dir: Path) -> dict[str, list[np.ndarray]]:
    """SAM 3.1's sky, people, windows and street lamps on every frame of the picture (proxy size, 0..1)."""
    proxy = run_dir / "night_proxy.mp4"
    common.proxy_clip(ctx, picture, proxy)
    job = ctx.run(graphs.select_any(ctx.client.assets.from_file(proxy), NIGHT_LABELS, prefix="pickups/environment_night"))
    found = {}
    for j, label in enumerate(NIGHT_LABELS):
        d = run_dir / "night" / label
        d.mkdir(parents=True, exist_ok=True)
        for o in job.get_outputs(f"PU_Mask{j if j else ''}"):
            o.to_file(d / o.name)
        ms = [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(d.glob("mask_*.png"))]
        found[label] = [m.astype(np.float32) if m.mean() < NOTHING else np.zeros(m.shape, np.float32) for m in ms]
    found["person"] = [composite.blur(composite.grow(m > 0.5, 1).astype(np.float32), 1.5) if m.any() else m for m in found["person"]]
    return found


def relight_clip(ctx, work: Path, gen_files: list[Path], out: Path, box: tuple[int, int, int, int], texture: float,
                 plate: Path | None = None, mode: str = "light",
                 night: dict[str, list[np.ndarray]] | None = None) -> int:
    """Every frame of the work clip in the render's light, without the canvas's padding, as out/r_NNNNN.png: each
    frame's light fields (light_fields) averaged over FIELD_SMOOTH frames either side, so the light holds steady where
    LTX's steps between its chunks. The footage keeps its own grain, so none is added. `plate`: the clip the render
    was made from when a kept thing was taken out of it; the light is learnt between it and the render (neither has
    the thing) and laid on the footage, so the render's grass where the thing stands doesn't set its light. `mode`:
    "light" (golden hour, fog) or "night" (add_lights: the render's lamps and lit windows over the darkened footage).
    `night`: the sky and people per frame (night_masks); the footage is then graded by moonlight (moon_grade) instead
    of taking the render's light. Returns the frame count."""
    ox, oy, cw, ch = box
    n = len(gen_files)
    radius = max(2, round(TRANSFER_RADIUS * cw))

    def cut(f: np.ndarray) -> np.ndarray:
        return f[oy:oy + ch, ox:ox + cw]

    def fields(i: int, base: np.ndarray) -> np.ndarray:
        g = np.asarray(Image.open(gen_files[i]).convert("RGB"))
        if g.shape[:2] != base.shape[:2]:
            g = np.asarray(Image.fromarray(g).resize(base.shape[1::-1], Image.LANCZOS))
        return light_fields(cut(base), cut(g), radius, texture=texture)

    steady: list = [None] * n
    if night is None:
        every = list(common.parallel_frames((f for i, f in enumerate(common.read_frames(plate or work)) if i < n), fields))
        steady = [np.mean(every[max(0, i - FIELD_SMOOTH):i + FIELD_SMOOTH + 1], axis=0) for i in range(n)]

    small = max(2, round(EMISSIVE_RADIUS * cw))
    fog_colour = None
    if mode == "fog":  # the render's fog colour, from where its veil is heaviest, one for the whole shot
        g0 = np.asarray(Image.open(gen_files[n // 2]).convert("RGB"))
        v = steady[n // 2][1] @ LUM
        heavy = np.asarray(Image.fromarray((v > np.quantile(v, 0.8)).astype(np.uint8) * 255).resize(g0.shape[1::-1])) > 127
        fog_colour = np.median(to_linear(g0)[heavy], axis=0)

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((cw, ch), Image.BILINEAR))

    def one(i: int, orig: np.ndarray):
        if night is not None:
            j = min(i, len(night["sky"]) - 1)
            near = slice(max(0, j - 2), j + 3)
            sky = snap_sky(cut(orig), up(np.mean(night["sky"][near], axis=0)))
            person = up(night["person"][j])
            frame = moon_grade(cut(orig), sky, person)
            # The footage's own windows lit: a warm glow over the window's darker panes, steady over a few frames.
            win = up(np.mean(night["window"][near], axis=0))
            fixtures = np.clip(np.maximum(composite.blur(win, 2.0), up(np.mean(night["street lamp"][near], axis=0))) * 2, 0, 1)
            if win.any():
                lin = to_linear(frame) + (np.clip(win * 1.5 - 0.25, 0, 1) * (1 - person))[..., None] * WINDOW_GLOW
                frame = np.clip(np.rint(to_srgb(lin)), 0, 255).astype(np.uint8)
        else:
            frame = apply_fields(cut(orig), steady[i])
        if mode == "fog":
            frame = thicken_fog(frame, steady[i], fog_colour)
        if mode == "night":
            g = np.asarray(Image.open(gen_files[i]).convert("RGB"))
            if g.shape[:2] != orig.shape[:2]:
                g = np.asarray(Image.fromarray(g).resize(orig.shape[1::-1], Image.LANCZOS))
            # By the grade, a lamp is never in the sky or in front of a person (they stand before it).
            # Only on what the footage has (its windows, its lamps): a lamp the render stood in an empty field would
            # float there with no post.
            frame = add_lights(frame, cut(g), small, None if night is None else fixtures * (1 - sky) * (1 - person))
        Image.fromarray(frame).save(out / f"r_{i + 1:05d}.png")
        return i

    for i in common.parallel_frames((f for i, f in enumerate(common.read_frames(work)) if i < n), one):
        ctx.check()
        ctx.progress((i + 1) / n)
    return n


def crop_work(clip: Path, box: tuple[int, int, int, int], run_dir: Path) -> Path:
    """`clip` without the canvas's padding (the relit frames are already cut to the picture)."""
    ox, oy, cw, ch = box
    pr = media.probe(clip)
    if (pr.width, pr.height) == (cw, ch):
        return clip
    dst = run_dir / f"{clip.stem}_picture.mp4"
    if not dst.exists():
        crop_clip(clip, dst, box)
    return dst


def crop_masks(masks: list[np.ndarray], place: tuple[int, int, int, int, int, int]) -> list[np.ndarray]:
    """Proxy masks of the canvas, cut to the picture (at the proxy's scale)."""
    cw, ch, ox, oy, w, h = place
    out = []
    for m in masks:
        sy, sx = m.shape[0] / h, m.shape[1] / w
        out.append(m[round(oy * sy):round((oy + ch) * sy), round(ox * sx):round((ox + cw) * sx)])
    return out


# --- the shot's thin ground features hold their place and width in the new season ------------------------------------

HOLD_LABEL = "puddle"  # SAM's word that finds a far road's thin line and the icy strips in front of it
HOLD_MAX = 0.06  # of the frame: a find bigger than this is not a thin ground feature
HOLD_ROAD = 0.6  # a strip spanning more than this share of the frame's width is a road; shorter ones are standing water
WATER_REFLECT = 0.65  # how much of the sky still water reflects at a low angle
WATER_RIPPLE = 0.06  # the gentle wind ripple across the water, as a share of its light
WATER_SKYLINE = 0.3  # how far down each patch the reflected skyline lies (the hills above it, the sky below)


def stretch_rows(orig: np.ndarray, gen: np.ndarray, k: float) -> np.ndarray | None:
    """Per pixel of the footage's road (`orig`), the render's row to show there: the render's own road (`gen`, found
    on its first frame) stretched per column to the footage's top and bottom, so the new season's road keeps its old
    width. None when the render's road is missing over too much of it."""
    h, w = orig.shape
    have_o, have_g = orig.any(axis=0), gen.any(axis=0)
    both = have_o & have_g
    if both.sum() < 0.5 * have_o.sum():
        return None
    x = np.arange(w)
    r = max(3, round(15 * k))

    def ends(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.argmax(m, axis=0).astype(np.float32), (h - 1 - np.argmax(m[::-1], axis=0)).astype(np.float32)

    def along(v: np.ndarray) -> np.ndarray:  # filled where the render's road is missing, then smoothed along it
        v = np.interp(x, x[both], v[both]).astype(np.float32)
        pad = np.pad(v, r, mode="edge")
        return np.convolve(pad, np.ones(2 * r + 1, np.float32) / (2 * r + 1), mode="valid")

    t0, b0 = ends(orig)
    t1, b1 = ends(gen)
    t1, b1 = along(t1), along(b1)
    ys = np.arange(h, dtype=np.float32)[:, None]
    src = t1 + (ys - t0) * (b1 - t1) / np.maximum(b0 - t0, 1.0)
    return np.where(orig, np.clip(src, 0, h - 1), ys)


def sky_bottom(gen: np.ndarray) -> np.ndarray:
    """Per column, the row where the render's sky (the smooth bright top of the frame) ends: the skyline."""
    lum = to_linear(gen) @ LUM
    h = gen.shape[0]
    band = lum[: int(h * 0.6)]
    high = float(np.median(band[: max(2, h // 10)]))
    sky = (np.abs(band - box_mean(band, max(2, h // 200))) < 0.04 * high) & (band > 0.6 * high)
    sky = box_mean(sky.astype(np.float32), max(2, h // 200)) > 0.5
    bottom = np.where(sky.all(axis=0), band.shape[0], np.argmin(sky, axis=0)).astype(np.float32)
    r = max(2, gen.shape[1] // 200)
    return np.convolve(np.pad(bottom, r, mode="edge"), np.ones(2 * r + 1, np.float32) / (2 * r + 1), mode="valid")


def hold_ground(ctx, work: Path, out: Path, place: tuple[int, int, int, int, int, int], run_dir: Path,
                kept: list[np.ndarray] | None, frozen: bool = False) -> int:
    """On a still camera: what was there stays there in the new season. The shot's thin ground features (SAM's
    "puddle": a far road's line, icy strips in a field) are put back over every output frame at their own place and
    width, whatever the render made of them (frost to summer narrowed the road and grassed the strips over): a road
    is the footage in the new light (its own contrast, so it stays a road; frost and snow on it taken off), the strips
    are standing water reflecting the render's sky. In snow (`frozen`) water stays as rendered. Returns the features
    held."""
    cw, ch, ox, oy, w, h = place
    if not camera_still(work, kept or []):
        return 0
    d = run_dir / "hold"
    d.mkdir(exist_ok=True)
    frame0 = d / "frame0.png"
    first = next(common.read_frames(work))[oy:oy + ch, ox:ox + cw]
    Image.fromarray(first).save(frame0)
    found = common.select_preview(ctx, frame0, [], HOLD_LABEL, 1.0, d / "orig")
    if not found.any() or found.mean() > HOLD_MAX:
        return 0
    road = np.zeros_like(found)
    water = np.zeros_like(found)
    for comp, (x0, _, x1, _) in blobs(found):
        if comp.sum() < 0.0005 * comp.size:
            continue
        if x1 - x0 + 1 > HOLD_ROAD * found.shape[1]:
            road |= comp
        elif not frozen:
            water |= comp
    if not road.any() and not water.any():
        return 0
    k = cw / 1920
    outs = sorted(out.glob("r_*.png"))
    r0 = np.asarray(Image.open(outs[0]).convert("RGB"))
    guide = first.astype(np.float32) / 255

    def matte(m: np.ndarray) -> np.ndarray:
        if not m.any():
            return np.zeros(m.shape + (1,), np.float32)
        a = guided_colour(guide, composite.grow(m, max(1, round(2 * k))).astype(np.float32), max(2, round(3 * k)), 1e-3)
        return np.clip(a, 0.0, 1.0)[..., None]

    road_a, water_a = matte(road), matte(water)
    light = scene_light(first, r0, road | water, k)
    # The render's own road, stretched to the footage's width (frost on the footage's road would stay in summer).
    stretch = None
    if road.any():
        gen0 = d / "gen0.png"
        Image.fromarray(r0).save(gen0)
        try:
            on_gen = common.select_preview(ctx, gen0, [], HOLD_LABEL, 1.0, d / "gen")
        except Exception:  # noqa: BLE001 - then the footage's road in the new light
            on_gen = None
        if on_gen is not None and on_gen.shape == road.shape:
            stretch = stretch_rows(road, on_gen & composite.grow(road, max(4, round(20 * k))), k)
    # Still water reflects the scene's sky: upside down, the skyline (the hills, the tree line) a dark band along each
    # patch's far edge and the sky below it. Per water pixel, its source row.
    ys = np.arange(ch)[:, None]
    skyline = sky_bottom(r0)
    src_y = np.tile(ys, (1, cw)).astype(np.float32)
    below = np.full((ch, cw), 1e9, np.float32)  # how far below its patch's far edge each pixel lies
    for comp, _ in blobs(water):
        cols = np.flatnonzero(comp.any(axis=0))
        tops, tall = np.zeros(cw, np.int64), np.ones(cw, np.int64)
        tops[cols] = np.argmax(comp[:, cols], axis=0)
        tall[cols] = comp[:, cols].sum(axis=0)
        yy, xx = np.nonzero(composite.grow(comp, max(2, round(3 * k))))
        src_y[yy, xx] = np.clip(skyline[xx] + WATER_SKYLINE * tall[xx] - (yy - tops[xx]), 0, ch - 1)
        below[yy, xx] = yy - tops[xx]
    bank = np.clip(1 - below / max(1.0, 1.5 * k), 0, 1)  # the near bank's lip, a thin dark line along the far edge
    rng = np.random.default_rng(7)
    ripple = rng.standard_normal((max(8, ch // 6), max(8, cw // 24))).astype(np.float32)
    ripple = np.asarray(Image.fromarray(ripple).resize((cw * 2, ch), Image.BICUBIC))
    ripple = ripple / max(float(ripple.std()), 1e-6)
    sky_l = r0[: max(2, ch // 10)].astype(np.float32) @ LUM
    grain = float(np.std(sky_l - box_mean(sky_l, 2)))  # the render's own grain (in its sky), laid on the water
    xs = np.arange(cw)[None, :]

    def one(i: int, orig: np.ndarray):
        f = outs[min(i, len(outs) - 1)]
        g = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
        o = orig[oy:oy + ch, ox:ox + cw]
        # What was kept is the footage's own pixels and nothing paints over them: a car crossing the road kept the held
        # road's line over its wheels (they read as dissolving into the road: "not the same car").
        protect = np.zeros((ch, cw, 1), np.float32)
        if kept:
            km = upsize(kept[min(i, len(kept) - 1)], (w, h))[oy:oy + ch, ox:ox + cw]
            if km.any():
                protect = composite.blur(composite.grow(km, max(2, round(3 * k))).astype(np.float32), max(1.0, 1.5 * k))[..., None]
        road_a_i, water_a_i = road_a * (1 - protect), water_a * (1 - protect)
        if road.any():
            if stretch is not None:
                held_road = g[np.rint(stretch).astype(np.int64), xs]
            else:
                held_road = relight(o, light)
            g = g + road_a_i * (held_road - g)
        if water.any():
            shift = int(round(i * 0.8 * k)) % cw
            rip = ripple[:, shift:shift + cw]
            sy = np.clip(src_y + 1.5 * k * rip, 0, ch - 1)
            lin = to_linear(g)
            mirror = sum(lin[np.clip(np.rint(sy + dy * k), 0, ch - 1).astype(np.int64), xs] for dy in (-1, 0, 1)) / 3
            noise = np.random.default_rng(1000 + i).standard_normal((ch, cw)).astype(np.float32) * grain
            refl = WATER_REFLECT * mirror * (1 + WATER_RIPPLE * rip[..., None])
            wat = to_srgb(refl * (1 - 0.5 * bank[..., None])) + noise[..., None]
            g = g + water_a_i * (wat - g)
        Image.fromarray(np.clip(np.rint(g), 0, 255).astype(np.uint8)).save(f)
        return i

    for _ in common.parallel_frames((fr for i, fr in enumerate(common.read_frames(work)) if i < len(outs)), one):
        ctx.check()
    return len(blobs(road | water))


def layout_masks(ctx, frame_png: Path, out_dir: Path) -> dict[str, np.ndarray]:
    """The shot's roads, paths and buildings on its first frame (SAM 3.1, by words), each big enough to matter.
    SAM's whole-frame "nothing found" counts as nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for label in LAYOUT:
        try:
            m = common.select_preview(ctx, frame_png, [], label, 1.0, out_dir / label)
        except Exception:  # noqa: BLE001 - a missing check is better than a failed take
            continue
        if LAYOUT_SMALL <= m.mean() < GHOST_MAX * 2:
            found[label] = m
    return found


def overlap(a: np.ndarray, b: np.ndarray, give: int) -> float:
    """How well two masks of the same thing agree, allowing `give` px either way: the share of each that falls inside
    the other's grown outline, the smaller of the two (1 = the same place and size)."""
    if not a.any() or not b.any():
        return 0.0
    ga, gb = composite.grow(a, give), composite.grow(b, give)
    return float(min((a & gb).sum() / a.sum(), (b & ga).sum() / b.sum()))


def layout_match(ctx, candidate: Path, layout: dict[str, np.ndarray], out_dir: Path) -> float:
    """The worst overlap between the shot's layout and the same things found on a restyled frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = next(iter(layout.values()))
    with Image.open(candidate) as im:
        if im.size != ref.shape[::-1]:
            im.convert("RGB").resize(ref.shape[::-1], Image.LANCZOS).save(out_dir / "frame.png")
            candidate = out_dir / "frame.png"
    give = max(2, round(0.01 * ref.shape[0]))
    worst = 1.0
    for label, a in layout.items():
        try:
            b = common.select_preview(ctx, candidate, [], label, 1.0, out_dir / label)
        except Exception:  # noqa: BLE001
            continue
        b = b if b.mean() < GHOST_MAX * 2 else np.zeros_like(b)
        worst = min(worst, overlap(a, b, give))
    return worst


INVENT_LABELS = ("car", "person", "building", "animal")  # what a restyle tends to add far off: a car, a walker, a shed, a cow
INVENT_MIN = 0.0002  # of the frame: a find smaller than this is a speck, not an object
INVENT_PUDDLE = 0.004  # of the frame: in the rain a standing-water find bigger than this is a lake the render drew, not a puddle


def invent_masks(ctx, picture: Path, out_dir: Path, size: tuple[int, int] | None = None,
                 labels: tuple[str, ...] = INVENT_LABELS) -> dict[str, np.ndarray]:
    """SAM 3.1's find of each INVENT_LABELS on one picture (at `size` if given), SAM's whole-frame "nothing" as nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if size is not None:
        with Image.open(picture) as im:
            if im.size != tuple(size):
                im.convert("RGB").resize(size, Image.LANCZOS).save(out_dir / "frame.png")
                picture = out_dir / "frame.png"
    found = {}
    for label in labels:
        try:
            m = common.select_preview(ctx, picture, [], label, 1.0, out_dir / label)
        except Exception:  # noqa: BLE001 - a missing check is better than a failed take
            m = np.zeros(Image.open(picture).size[::-1], bool)
        found[label] = m if m.mean() < GHOST_MAX else np.zeros_like(m)
    return found


def invented_objects(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """What the restyle has of a kind (a car, a person, a building, an animal) that the shot didn't have: (label, share of
    the frame) for each find on `after` that lies away from every find of that kind on `before`, compact and not a speck.
    A restyle that puts a weird object far off at the edge of the road is caught here, and asked again."""
    out = []
    for label, b in after.items():
        a = before.get(label)
        if a is None or not b.any():
            continue
        k = b.shape[1] / 1920
        new = b & ~composite.grow(a, max(3, round(12 * k)))
        r = max(1, round(2 * k))
        new = composite.grow(composite.shrink(new, r), r + 1) & new
        if not new.any():
            continue
        for comp, (x0, y0, x1, y1) in blobs(new[::2, ::2]):
            share = float(comp.sum()) * 4 / new.size
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if label == "puddle":  # standing water has no compact shape; only its size says it is a lake
                if share >= INVENT_PUDDLE:
                    out.append((label, round(share, 5)))
            elif share >= INVENT_MIN and bw <= 0.5 * new.shape[1] / 2 and bw <= 5 * bh:
                out.append((label, round(share, 5)))
    return out


def run(ctx) -> dict:
    p = ctx.params
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)
    key = p.get("change", "snow")
    preset = PRESETS.get(key)
    custom = (p.get("custom") or "").strip()
    if preset is None and not custom:
        raise ValueError(f"Unknown change '{key}'. Pick one of: {', '.join(PRESETS)}, or describe your own.")
    label = preset["label"] if preset else custom[:40]
    qwen = (p.get("look_prompt") or (preset["qwen"] if preset else f"{custom.rstrip('.')}.")).strip()
    describe = (p.get("describe") or "").strip()
    video = (p.get("prompt") or " ".join(x for x in (describe, preset["ltx"] if preset else custom) if x)).strip()
    cast, _ = ctx.store.shot_cast(ctx.shot_id)
    if cast and "@" in video:
        video = mentions.for_text(video, cast)
    control = p.get("control") or (preset["control"] if preset else "depth")
    seed = int(p.get("seed", 42))
    grain = p.get("grain", "fine")

    ctx.stage("Preparing the shot")
    src = media.probe(ctx.input_path)
    start = int(p.get("start_frame", 0))
    frames = media.ltx_frames(min(common.length_cap(p, "environment"), src.frames - start))
    if frames > src.frames - start:
        frames -= 8
    place = fit(src.width, src.height)
    cw, ch, ox, oy, w, h = place
    box = (ox, oy, cw, ch)
    work = run_dir / "work.mp4"
    info = media.prepare_work_clip(ctx.input_path, work, start, frames, (cw, ch), pad=(w, h, ox, oy))
    fps = info["fps"]

    if key == "fog" and not p.get("look_prompt") and not p.get("prompt"):
        return fog_take(ctx, work, place, fps, label)
    if key == "autumn" and not p.get("look_prompt") and not p.get("prompt"):
        return autumn_take(ctx, work, place, fps, label)

    # Found before any rendering: a label that isn't in the shot stops here, in seconds.
    kept: list[np.ndarray] | None = None
    ref_text: tuple[str, str] | None = None
    obj = ""
    drive_clip, cleared, bg = work, False, None  # what the models see; whether the kept thing was taken out of it
    graded_night = key == "night" and not p.get("look_prompt") and not p.get("prompt")
    if keep_wanted(p) and not graded_night:  # by night everyone is graded and lit as filmed (moon_grade)
        ctx.stage("Finding what to keep")
        found = keep_masks(ctx, p, work, start, frames, place, src, run_dir)
        if ghost_label(p.get("keep_text", "")) == "person":
            found = stable_masks(small_frames(work, found[0].shape[::-1])[:len(found)], found)
            found = [close_gaps(m) for m in found]
        kept = mend_dropouts(found)
        np.savez_compressed(run_dir / "keep_masks.npz", masks=np.stack(kept))
        obj = keep_name(p.get("keep_text", "")) or where_name(kept[0])
        Obj = obj[0].upper() + obj[1:]
        mode = p.get("keep_mode", "auto")
        still = camera_still(work, kept)
        cleared = mode == "clear" or (mode == "auto" and (ghost_label(p.get("keep_text", "")) == "person" or
                                                          (travels(kept) and still)))
        if cleared:
            ref_text = (qwen, f"{video} {KEEP_OBJECT_VIDEO.format(Obj=Obj)}".strip())  # what the light reference is asked for
            # It moves through the frame, or it is a person (LTX's copy of a walking person runs off theirs: a second
            # head beside his): taken out of the shot, so neither model can draw a second one.
            ctx.stage("Taking it out of the shot")
            drive_clip, bg = clear_shot(ctx, work, mend_dropouts(found, cover=True), (w, h), fps, run_dir, seed, still)
            video, qwen = without_subject(video, obj), without_subject(qwen, obj)
        else:
            sentence = KEEP_OBJECT_VIDEO.format(Obj=Obj)
            if sentence not in video:
                video = f"{video} {sentence}".strip()

    ctx.stage("Restyling the first frame")
    frame0 = run_dir / "frame0.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(drive_clip), "-frames:v", "1", str(frame0)], check=True)
    layout_qa: dict = {}
    if p.get("first_frame_ref"):
        first = ctx.store.ref_path(ctx.project_id, p["first_frame_ref"])
    elif p.get("first_frame_upload_id"):
        first = ctx.store.upload_path(p["first_frame_upload_id"])
    else:
        layout = layout_masks(ctx, frame0, run_dir / "layout")
        instruction = qwen
        if NOTHING_NEW not in qwen:
            instruction += " " + NOTHING_NEW
        instruction += " " + LAYOUT_KEEP
        if obj and not cleared and KEEP_OBJECT.format(obj=obj) not in qwen:  # what changes, then what must not
            instruction += " " + KEEP_OBJECT.format(obj=obj)
        if not re.search(r"\bkeep\b", qwen, re.I):  # don't say it twice
            instruction += " " + KEEP
        # The restyled frame sets what LTX draws for the whole shot: a road it paints over stays gone (the frosty field
        # lost its road to meadow; the dusk road's grass verge became wet tarmac). Each try is checked against the
        # shot's own roads, paths and buildings, and a repaint is asked for until they are where they were.
        tries: list[float] = []
        invented_log: list = []
        best = None
        with Image.open(frame0) as im:
            frame0_size = im.size
        invent_labels = INVENT_LABELS + (("puddle",) if key == "rain" else ())
        base_finds = invent_masks(ctx, frame0, run_dir / "invent_base", labels=invent_labels)
        attempt = 0
        while attempt < LAYOUT_TRIES or (layout and attempt < LAYOUT_TRIES_MAX and max(tries) >= LAYOUT_UNSURE):
            job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(frame0), instruction, seed=seed + attempt, resolution=0,
                                            prefix="pickups/environment_key", commercial_safe=ctx.commercial_safe(),
                                            size=graphs.edit_size(w, h)))
            candidate = run_dir / f"first_{attempt}.png"
            job.get_outputs("PU_Plate")[0].to_file(candidate)
            score = layout_match(ctx, candidate, layout, run_dir / f"layout_{attempt}") if layout else 1.0
            # An object the restyle invented far off (the frosty field's "weird object on the right") fails the try too.
            extra = invented_objects(base_finds, invent_masks(ctx, candidate, run_dir / f"invent_{attempt}", frame0_size, invent_labels))
            invented_log.append(extra)
            tries.append(round(score, 3))
            rank_key = (not extra, score)
            if best is None or rank_key > best[2]:
                best = (score, candidate, rank_key)
            if score >= LAYOUT_MIN and not extra:
                break
            attempt += 1
        if layout and LAYOUT_UNSURE <= max(tries) and best[0] < LAYOUT_MIN:
            raise RuntimeError("The new weather kept moving the shot's layout (its roads, paths or buildings), so it "
                               "wasn't rendered. Try another seed or a lighter look.")
        if layout and best[0] < LAYOUT_UNSURE:
            # Nothing came close: SAM's idea of the road was wrong to begin with (a frosty field it called a road),
            # so the scores say nothing; the first try, made with the instruction, stands.
            best = (tries[0], run_dir / "first_0.png")
        first = run_dir / "first.png"
        first.write_bytes(best[1].read_bytes())
        if preset and preset.get("light_only"):
            f0 = np.asarray(Image.open(frame0).convert("RGB"))
            rs = np.asarray(Image.open(first).convert("RGB"))
            Image.fromarray(light_transfer(f0, rs, max(2, round(TRANSFER_RADIUS * f0.shape[1])),
                                           texture=float(preset["light_only"]))).save(first)
        layout_qa = ({"layout": sorted(layout), "layout_tries": tries,
                      "layout_check": "passed" if best[0] >= LAYOUT_MIN else "unsure" if max(tries) < LAYOUT_UNSURE else "best try"}
                     if layout else {})
        layout_qa["invented"] = invented_log  # per try: what the restyle added far off (label, share of the frame)

    ctx.stage("Rendering the new weather")
    job = ctx.run(graphs.drive(ctx.client.assets.from_file(drive_clip), ctx.client.assets.from_file(first), video, fps, w, h,
                               control_mode=control, seed=seed, prefix="pickups/environment"))
    out = run_dir / "out"
    out.mkdir(exist_ok=True)
    relit = bool(preset and preset.get("light_only"))

    def light_reference(to_picture: bool) -> tuple[list[Path], list[np.ndarray]] | None:
        """A second render with the kept thing in it, as the reference for its broad light (reference_pass); the frames
        and masks cut to the picture when `to_picture` (relit footage works without the canvas's padding)."""
        if ref_text is None or not p.get("subject_light", True):
            return None
        ctx.stage("Lighting what you kept")
        try:
            ref = reference_pass(ctx, work, ref_text[0], ref_text[1], obj, ghost_label(p.get("keep_text", "")) and (
                "person" if ghost_label(p.get("keep_text", "")) == "person" else sam_words(p.get("keep_text", ""))),
                kept, (w, h), fps, control, seed, run_dir)
        except Canceled:
            raise
        except Exception:  # noqa: BLE001 - without a reference the thing keeps the scene's overall light
            return None
        if ref is not None and to_picture:
            crop_pngs(ref[0][0].parent, "*.png", box)
            ref = (ref[0], crop_masks(ref[1], place))
        return ref

    if relit:
        # A change of light, fog or season-free sky: every frame is the footage, in the render's light. Roads, tracks,
        # people and cars stay exactly as filmed; nothing the render drew finer than TRANSFER_RADIUS comes through.
        gdir = run_dir / "gen"
        gdir.mkdir(exist_ok=True)
        for o in job.get_outputs("PU_Out"):
            o.to_file(gdir / o.name)
        ctx.stage("Keeping it as shot")
        ghosts = 0
        if kept is not None and cleared and bg.locked:
            ghosts = clear_ghosts(ctx, sorted(gdir.glob("*.png")), drive_clip, ghost_label(p.get("keep_text", "")), fps, run_dir)
        mode = key if key in ("night", "fog") else "light"
        if kept is None:
            night = night_masks(ctx, crop_work(work, box, run_dir), run_dir) if graded_night else None
            n_rel = relight_clip(ctx, work, sorted(gdir.glob("*.png")), out, box, float(preset["light_only"]), mode=mode,
                                 night=night)
            keep_details = {"_qa": {}}
        else:
            # The kept thing (taken out of what LTX saw) is relit on its own, with the light around it held to a
            # light's colour (keep_relit), over the relit footage: the field learnt over its place is the meadow's,
            # and laid on navy shorts it turned them teal.
            rdir = run_dir / "relit"
            rdir.mkdir(exist_ok=True)
            n_rel = relight_clip(ctx, work, sorted(gdir.glob("*.png")), rdir, box, float(preset["light_only"]), mode=mode,
                                 plate=drive_clip if cleared else None)
            unbox = (0, 0, cw, ch)
            if ghost_label(p.get("keep_text", "")) == "person":
                # People take a change of light the autumn way: their own footage, graded (no render of them at all).
                keep_details = grade_person_clip(ctx, crop_work(work, box, run_dir), sorted(rdir.glob("r_*.png")),
                                                 crop_masks(kept, place), out, cw / 1920)
            else:
                ref_pic = light_reference(True) if cleared else None
                for veil in (None, 0.0):  # a lock that fails with the veil is tried again without it
                    try:
                        _, keep_details = keep_pass(ctx, crop_work(work, box, run_dir), sorted(rdir.glob("r_*.png")), crop_masks(kept, place),
                                                    out, fps, "off", obj, unbox, plate=crop_work(drive_clip, box, run_dir) if cleared else None,
                                                    haze=key in ATMOSPHERE, real_plate=bool(bg and bg.locked),
                                                    light_bounds=preset["light"], see_through=thin_kind(p.get("keep_text", "")),
                                                    person=ghost_label(p.get("keep_text", "")) == "person", reference=ref_pic, veil_scale=veil)
                        break
                    except LockFailed:
                        if veil == 0.0:
                            raise
        receipt = {"original": 100.0, "borrowed": 0.0, "ai": 0.0}
        keep_details["relit"] = "every pixel is your footage; its light, colour and weather come from the render"
        keep_details["_qa"].update(method="moonlit grade" if graded_night else "relit footage", frames=n_rel, ghost_frames=ghosts)
    elif kept is None:
        for o in job.get_outputs("PU_Out"):
            o.to_file(out / o.name)
        for i, f in enumerate(sorted(out.glob("*.png"))):
            f.rename(out / f"r_{i + 1:05d}.png")
        crop_pngs(out, "r_*.png", box)
        common.grain_pngs(out, "r_*.png", grain)
        receipt = {"original": 0.0, "borrowed": 0.0, "ai": 100.0}
        keep_details: dict = {}
    else:
        gdir = run_dir / "gen"  # the render as it came, kept beside the take like Recast's
        gdir.mkdir(exist_ok=True)
        for o in job.get_outputs("PU_Out"):
            o.to_file(gdir / o.name)
        ctx.stage("Keeping it as shot")
        ghosts = 0
        if cleared and bg.locked:  # its frames line up only on a still camera
            ghosts = clear_ghosts(ctx, sorted(gdir.glob("*.png")), drive_clip, ghost_label(p.get("keep_text", "")), fps, run_dir)
        haze = key in ATMOSPHERE or bool(re.search(r"\b(fog|foggy|mist|misty|haze|hazy|snow|snowy|rain|rainy)\b", custom, re.I))
        ref_canvas = light_reference(False) if cleared else None
        for veil in (None, 0.0):  # a lock that fails with the veil is tried again without it
            try:
                receipt, keep_details = keep_pass(ctx, work, sorted(gdir.glob("*.png")), kept, out, fps, grain, obj, box,
                                                  plate=drive_clip if cleared else None, haze=haze, real_plate=bool(bg and bg.locked),
                                                  light_bounds=preset["light"] if preset else (GAIN_RANGE, HAZE_MAX),
                                                  see_through=thin_kind(p.get("keep_text", "")),
                                                  person=ghost_label(p.get("keep_text", "")) == "person", reference=ref_canvas,
                                                  veil_scale=veil)
                break
            except LockFailed:
                if veil == 0.0:
                    raise
        if cleared:
            keep_details["_qa"].update(keep_plate=bg.method(), ghost_frames=ghosts)

    held = 0
    # Rain wets the whole ground: the render draws the path glistening along its length and puddles where they lie. Holding the
    # footage's dry road and "puddles" over that left a puddle in front of the woman and a dry path further down (the held
    # road is the footage's own, its scree strips became flat standing water).
    if not relit and key != "rain":
        try:
            held = hold_ground(ctx, work, out, place, run_dir, kept, frozen=key == "snow")
        except Exception:  # noqa: BLE001 - a missed hold is the render as it was, never a failed take
            held = 0
    if held:
        keep_details.setdefault("_qa", {})["ground_held"] = held

    ctx.stage("Writing your files")
    shown = work
    if (cw, ch) != (w, h):
        shown = run_dir / "shown.mp4"
        crop_clip(work, shown, box)
    downloads = common.finish_outputs(ctx.vdir, out / "r_%05d.png", fps, audio_from=shown)
    from pickups.app.store import make_viewer
    make_viewer(shown, ctx.vdir / "before.mp4")
    with Image.open(first) as im:  # the restyled frame as the shot is framed, without the canvas's padding
        sx, sy = im.width / w, im.height / h
        im.convert("RGB").crop((round(ox * sx), round(oy * sy), round((ox + cw) * sx), round((oy + ch) * sy))).save(
            ctx.vdir / "first_frame.png")
    note = ("Every pixel is your footage, relit in the new weather; chain Finish for a 4K master." if relit else
            "Every pixel is generated from your shot's structure; chain Finish for a 4K master." if kept is None else
            f"{Obj} is your footage: its detail as shot, its light and colour from the render. Everything else is generated "
            "from your shot's structure; chain Finish for a 4K master.")
    return {"label": f"Weather · {label}", "task": "environment", "receipt": receipt, "downloads": downloads,
            "details": {**common.sequence_details(out, "r_*.png", fps), "change": key if preset else "custom", "control": control,
                        "models": [graphs.edit_model(ctx.commercial_safe()), "LTX-2.5 + Union Control IC-LoRA",
                                   "Depth Anything 3" if control == "depth" else "Canny"] + ([] if kept is None else ["SAM 3.1"]),
                        **keep_details, **({"_layout": layout_qa} if layout_qa else {}), "look_prompt": qwen, "prompt": video, "grain": grain, "note": note}}


# --- fog from depth: the footage itself, no render pixels ---------------------------------------------------------------
#
# LTX draws fog thin, and any render pixels laid over the footage ghost where they don't line up (a second row of
# roofs over the harbour). Fog is the footage seen through air: each pixel keeps exp(-FOG_K * distance) of itself and
# the rest is the fog colour, with distance from Depth Anything 3. Near people stay clear, far hills vanish, the sky
# is all fog.

FOG_K = 4.0  # extinction over the scene's depth range: the nearest ~0.75 clear, the middle ~0.35, the far ~0.05
FOG_COLOUR = np.array([0.56, 0.58, 0.60], np.float32)  # linear light: a soft, cool, bright grey
FOG_NEAR = 1.0  # the most of itself the nearest thing keeps (bright fog laid over dark near grass even at 15% read
                # as the whole moor in fog)
FOG_SMOOTH = 3  # frames either side the fog density is averaged over: depth flickers a little frame to frame
FOG_GROUND = 0.25  # the distance at the top of the frame's bottom quarter, for ground (0 at the bottom edge)


def sky_of(depth: np.ndarray) -> np.ndarray:
    """The sky in a DA3 depth image (near bright; the model's sky clip sets the sky to one flat value): the value most
    common along the top rows, wherever it fills the picture."""
    top = depth[: max(4, depth.shape[0] // 20)].ravel()
    vals, counts = np.unique(top, return_counts=True)
    v = vals[np.argmax(counts)]
    flat = np.abs(depth.astype(np.int16) - int(v)) <= 1
    return flat if flat[: max(4, depth.shape[0] // 20)].mean() > 0.3 else np.zeros_like(flat)


def ground_prior(dist: np.ndarray) -> np.ndarray:
    """Distance (0 near .. 1 far, HxW) held to the ground plane: down each column it can only come nearer (the ground
    in front of the camera is nearer the lower it is in the frame), and the bottom quarter of the frame is near ground
    rising from 0 at the bottom edge to FOG_GROUND at its top. The model misread the near moor at Glencoe's bottom
    edge as mid-distance, so it sat in fog."""
    h = dist.shape[0]
    dist = np.minimum.accumulate(dist, axis=0)
    y = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    ramp = np.where(y >= 0.75, FOG_GROUND * (1 - y) / 0.25, 1.0)
    return np.minimum(dist, ramp)


def fog_density(depth: np.ndarray, sky: np.ndarray | None = None) -> np.ndarray:
    """How much of each pixel is fog (0..1) from a DA3 depth image (uint8, near bright). `sky` (the sky, if already
    found) is all fog."""
    d = depth.astype(np.float32) / 255.0
    sky = sky_of(depth) if sky is None else sky
    dist = 1.0 - d
    if (~sky).any():  # stretch the scene (not the sky) to 0..1
        lo, hi = np.quantile(dist[~sky], [0.01, 0.995])
        dist = np.clip((dist - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    dist[sky] = 1.0
    dist = ground_prior(dist)
    keep = FOG_NEAR * np.exp(-FOG_K * dist ** 1.5)  # the near ground clears faster than linear: people stay readable
    keep[sky] = 0.0
    return 1.0 - keep


def guided(guide: np.ndarray, src: np.ndarray, radius: int, eps: float = 1e-3) -> np.ndarray:
    """A guided filter: `src` smoothed but with the edges of `guide` (both HxW float)."""
    mi, mp = box_mean(guide, radius), box_mean(src, radius)
    var = box_mean(guide * guide, radius) - mi * mi
    a = (box_mean(guide * src, radius) - mi * mp) / (var + eps)
    b = mp - a * mi
    return box_mean(a, radius) * guide + box_mean(b, radius)


def fog_over_footage(orig: np.ndarray, density: np.ndarray, radius: int, sky: np.ndarray | None = None) -> np.ndarray:
    """The footage under fog of `density` (HxW, smooth), its edge snapped to the footage's own (rooftops, people).
    `sky` (0..1, already sharp): kept as it is, not smoothed again. `radius` 0: `density` is already final."""
    lin = to_linear(orig)
    dens = density.astype(np.float32) if radius <= 0 else np.clip(guided(lin @ LUM, density.astype(np.float32), radius), 0.0, 1.0)
    if sky is not None:
        dens = np.maximum(dens * (1 - sky), sky)
    dens = dens[..., None]
    return np.clip(np.rint(to_srgb(lin * (1 - dens) + FOG_COLOUR * dens)), 0, 255).astype(np.uint8)


SKY_BAND = 0.006  # of the frame's width: how far off the depth model's coarse skyline can be


def snap_sky(pic: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """The sky (HxW 0..1) at the footage's own sharpness. `soft`: the depth model's sky, coarse (it works at 518 px).
    Near its skyline each pixel goes to whichever it looks more like by colour, the sure sky just above it or the
    sure land just below; a pixel that is part of both (a ridge's edge) gets its share."""
    h, w = soft.shape
    r = max(2, round(SKY_BAND * w))
    hard = soft > 0.5
    sure_sky, sure_land = composite.shrink(hard, r), ~composite.grow(hard, r)
    band = ~sure_sky & ~sure_land
    out = sure_sky.astype(np.float32)
    rows = np.flatnonzero(band.any(axis=1))
    if not len(rows):
        return out
    y0, y1 = max(0, rows[0] - 2 * r), min(h, rows[-1] + 2 * r + 1)  # only the rows around the skyline
    img = pic[y0:y1].astype(np.float32) / 255

    def near(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mf = m[y0:y1].astype(np.float32)
        share = box_mean(mf, 2 * r)
        return box_mean(img * mf[..., None], 2 * r) / np.maximum(share, 1e-4)[..., None], share > 1e-3

    sky_c, sky_ok = near(sure_sky)
    land_c, land_ok = near(sure_land)
    d_sky, d_land = ((img - sky_c) ** 2).sum(axis=-1), ((img - land_c) ** 2).sum(axis=-1)
    a = np.clip((d_land / np.maximum(d_sky + d_land, 1e-6) - 0.5) * 2.0 + 0.5, 0.0, 1.0)
    b = band[y0:y1]
    seg = out[y0:y1]
    ok = b & sky_ok & land_ok
    seg[ok] = a[ok]
    seg[b & ~ok] = soft[y0:y1][b & ~ok]
    return out


def fog_take(ctx, work: Path, place: tuple[int, int, int, int, int, int], fps: float, label: str) -> dict:
    """Fog with no generated pixels: depth on every frame, then the footage under fog by that depth (fog_over_footage),
    the density averaged over FOG_SMOOTH frames either side."""
    cw, ch, ox, oy, w, h = place
    box = (ox, oy, cw, ch)
    run_dir = ctx.vdir / "work"
    ctx.stage("Rendering the new weather")
    shown = work
    if (cw, ch) != (w, h):  # depth on the picture alone: the canvas's smeared padding reads as something near
        shown = run_dir / "shown.mp4"
        crop_clip(work, shown, box)
    job = ctx.run(graphs.depth_frames(ctx.client.assets.from_file(shown), prefix="pickups/environment_depth"))
    ddir = run_dir / "depth"
    ddir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Depth"):
        o.to_file(ddir / o.name)
    files = sorted(ddir.glob("*.png"))
    if not files:
        raise RuntimeError("Couldn't measure depth in the shot.")
    ctx.stage("Keeping it as shot")
    small = (max(64, cw // 8), max(36, ch // 8))

    def density(f: Path) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(Image.open(f).convert("L"))
        sky = sky_of(d)
        dens = fog_density(d, sky)
        # The sky is laid per frame at full size, sharp. Under it the scene's density is carried on from the land
        # nearby, so the coarse map has no step at the skyline to smear (the sky's own density there blurred a fog
        # halo into every ridge).
        land = (~sky).astype(np.float32)
        spread = max(2.0, 0.02 * d.shape[1])
        near = composite.blur(dens * land, spread) / np.maximum(composite.blur(land, spread), 1e-4)
        dens[sky] = np.where(composite.blur(land, spread)[sky] > 1e-3, near[sky], 1.0 - FOG_NEAR * float(np.exp(-FOG_K)))
        return np.asarray(Image.fromarray(dens).resize(small, Image.BILINEAR)), sky

    both = [density(f) for f in files]
    dens, skies = [b[0] for b in both], [b[1] for b in both]
    n = len(dens)
    out = run_dir / "out"
    out.mkdir(exist_ok=True)

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((cw, ch), Image.BILINEAR))

    def one(i: int, frame: np.ndarray):
        j = min(i, n - 1)
        scene = up(np.mean(dens[max(0, j - FOG_SMOOTH):j + FOG_SMOOTH + 1], axis=0))
        pic = frame[oy:oy + ch, ox:ox + cw]
        # The skyline at the footage's own sharpness: this frame's sky, its edge refined on the picture itself.
        sky = snap_sky(pic, up(skies[j]))
        full = sky + (1 - sky) * scene
        Image.fromarray(fog_over_footage(pic, full, 0)).save(out / f"r_{i + 1:05d}.png")
        return i

    total = media.probe(work).frames
    for i in common.parallel_frames(common.read_frames(work), one):
        ctx.check()
        ctx.progress((i + 1) / total)
    ctx.stage("Writing your files")
    downloads = common.finish_outputs(ctx.vdir, out / "r_%05d.png", fps, audio_from=shown)
    from pickups.app.store import make_viewer
    make_viewer(shown, ctx.vdir / "before.mp4")
    Image.open(out / "r_00001.png").save(ctx.vdir / "first_frame.png")
    return {"label": f"Weather · {label}", "task": "environment", "receipt": {"original": 100.0, "borrowed": 0.0, "ai": 0.0},
            "downloads": downloads,
            "details": {**common.sequence_details(out, "r_*.png", fps), "change": "fog", "models": ["Depth Anything 3"],
                        "_qa": {"method": "fog from depth", "frames": total},
                        "note": "Every pixel is your footage, seen through fog that thickens with distance."}}


# --- autumn: the footage's own trees and grass recoloured, no render pixels -----------------------------------------
#
# A render redraws the trees (and moves the road); autumn is a change of colour, so the footage keeps every pixel and
# only its foliage turns: SAM 3.1 finds the trees, bushes and grass on every frame, and within them each green pixel
# takes an autumn hue by its own lightness (the shade in a crown deep red-brown, its middle orange, its lit leaves
# gold; grass tawny), keeping its lightness and texture. Road, sky, water and people are never touched.

AUTUMN_LABELS = ("tree", "bush", "grass", "meadow", "person")
AUTUMN_SMOOTH = 3  # frames either side the vegetation is averaged over: SAM finds a far hillside on most frames, not all
LEAF_HUES = ((0.3, 40.0), (0.5, 58.0), (0.7, 80.0))  # OKLab lightness -> OKLab hue (degrees) for leaves
GRASS_HUES = ((0.3, 68.0), (0.5, 78.0), (0.7, 90.0))  # ... and for grass: olive-brown to straw
LEAF_CHROMA, GRASS_CHROMA = 1.1, 0.8  # of the green's own colourfulness: natural, not saturated
GREEN = (100.0, 180.0)  # the OKLab hues that are foliage (yellow-green to blue-green), fading out 20 degrees beyond
CLOTHES_GREEN = 0.07  # OKLab chroma: a green this strong on a person is clothing, not the grass behind them
NOTHING = 0.9  # a mask over more than this share of the frame is SAM's "nothing found"

_OK_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def to_oklab(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (..., 3) to OKLab (L 0..1, a, b)."""
    return np.cbrt(lin @ _OK_M1.T) @ _OK_M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    """OKLab to linear sRGB."""
    return (lab @ np.linalg.inv(_OK_M2).T) ** 3 @ np.linalg.inv(_OK_M1).T


def autumn_colour(pic: np.ndarray, leaf: np.ndarray, grass: np.ndarray, people: np.ndarray | None = None) -> np.ndarray:
    """The picture (uint8) with its green turned to autumn where `leaf` (trees, bushes) and `grass` (HxW 0..1) say:
    each green pixel's hue set by its lightness, its lightness kept, so the texture and light are the footage's.
    `people`: where a person is; a strong green there is their clothes and stays (the grass SAM's outline takes in
    between their legs still turns)."""
    lab = to_oklab(to_linear(pic))
    light, chroma = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    lo, hi = GREEN
    green = np.clip(np.minimum(hue - (lo - 20), (hi + 20) - hue) / 20, 0, 1) * np.clip((chroma - 0.004) / 0.012, 0, 1)
    if people is not None:
        green = green * (1 - people * np.clip((chroma - CLOTHES_GREEN) / 0.03, 0, 1))

    def turned(hues: tuple, gain: float) -> np.ndarray:
        h = np.radians(np.interp(light, [x for x, _ in hues], [y for _, y in hues]))
        return np.stack([light, gain * chroma * np.cos(h), gain * chroma * np.sin(h)], axis=-1)

    w_leaf = (leaf * green)[..., None]
    w_grass = (grass * (1 - leaf) * green)[..., None]
    new = lab + w_leaf * (turned(LEAF_HUES, LEAF_CHROMA) - lab) + w_grass * (turned(GRASS_HUES, GRASS_CHROMA) - lab)
    return np.clip(np.rint(to_srgb(np.clip(from_oklab(new), 0, None))), 0, 255).astype(np.uint8)


BOUNCE = 0.5  # how much of the turned ground's colour change a person standing in it takes (it is light bounced up from it)
GRADE_LUM_MAX = 0.08  # OKLab lightness: the most of the ground's change of brightness a person takes (golden hour)
GRADE_CHROMA_MAX = 0.12  # ...and of its colour
BOUNCE_MAX = 0.07  # OKLab chroma: the most of the ground's change of colour that is taken, before BOUNCE
GRADE_HEAD = 0.85  # the sun lights the whole of them: their head takes this much of what their feet do
GRADE_STRENGTH = 1.0  # how much of that change a person takes in a change of light (BOUNCE for the autumn's)
GRADE_LUM = 0.6  # ...and how much of the change of brightness
BOUNCE_HEAD = 0.35  # ...at their head: the sky's light dominates up there, the ground's at their feet


def person_bounce(pic: np.ndarray, new: np.ndarray, people: np.ndarray, strength: float = BOUNCE,
                  chroma_max: float = BOUNCE_MAX, lum: float = 0.0, head: float = BOUNCE_HEAD,
                  sky: np.ndarray | None = None) -> np.ndarray:
    """The recoloured picture with the people in it lit by the new ground: the change of colour that autumn made to the
    grass and bushes around them (their median change in OKLab, the ring 2-15% of their height out) laid on their own
    colour as a broad shift, stronger toward their feet. A shift of colour only, so every bit of their detail and
    brightness is the footage's, and a white shirt that read cool beside straw-coloured grass takes its warm bounce."""
    m = people > 0.5
    rows = np.flatnonzero(m.any(axis=1))
    if not rows.size:
        return new
    tall = rows[-1] - rows[0] + 1
    pad = int(0.17 * tall) + 2
    h, w = m.shape
    y0, y1 = max(0, rows[0] - pad), min(h, rows[-1] + 1 + pad)
    cols = np.flatnonzero(m.any(axis=0))
    x0, x1 = max(0, cols[0] - pad), min(w, cols[-1] + 1 + pad)
    sl = (slice(y0, y1), slice(x0, x1))
    mb = m[sl]
    ring = composite.grow(mb, int(0.15 * tall)) & ~composite.grow(mb, max(2, int(0.02 * tall)))
    lo, ln = to_oklab(to_linear(pic[sl])), to_oklab(to_linear(new[sl]))
    d = ln - lo
    changed = ring & (np.hypot(d[..., 1], d[..., 2]) > 0.01)
    if changed.sum() < 64:
        return new
    shift = np.median(d[changed][:, 1:], axis=0)
    shift = shift * min(1.0, chroma_max / max(float(np.hypot(*shift)), 1e-6))  # a tint, never a recolour
    d_light = float(np.clip(np.median(d[changed][:, 0]), -GRADE_LUM_MAX, GRADE_LUM_MAX)) * lum
    v = np.clip((np.arange(y0, y1) - rows[0]) / max(tall - 1, 1), 0, 1)
    weight = (head + (1 - head) * v)[:, None] * people[sl] * strength
    lab = ln.copy()
    lab[..., 1:] += weight[..., None] * shift
    lab[..., 0] += weight * d_light
    if sky is not None:
        # Their bright parts (a white shirt) are lit by the light itself: the colour of what the new light did to the
        # bright, neutral things in the frame (clouds), laid on in proportion to how bright they are.
        wb = np.clip((composite.blur(lo[..., 0], max(2.0, 0.02 * tall)) - 0.45) / 0.35, 0, 1) * people[sl] * strength
        lab += wb[..., None] * sky
    out = new.copy()
    out[sl] = np.clip(np.rint(to_srgb(np.clip(from_oklab(lab), 0, None))), 0, 255).astype(np.uint8)
    return out


def sky_shift(pic: np.ndarray, new: np.ndarray, people: np.ndarray) -> np.ndarray | None:
    """What the new light did to the bright, neutral things of the frame (clouds, bright sky): the median change of
    lightness and colour in OKLab, colour capped at GRADE_CHROMA_MAX. None if the frame has too few of them."""
    sub = max(1, pic.shape[1] // 480)
    lo, ln = to_oklab(to_linear(pic[::sub, ::sub])), to_oklab(to_linear(new[::sub, ::sub]))
    far = ~composite.grow(people[::sub, ::sub] > 0.5, 8)
    pick = far & (lo[..., 0] > 0.55) & (np.hypot(lo[..., 1], lo[..., 2]) < 0.035)
    if pick.sum() < 200:
        return None
    d = np.median((ln - lo)[pick], axis=0)
    c = float(np.hypot(d[1], d[2]))
    if c > GRADE_CHROMA_MAX:
        d[1:] *= GRADE_CHROMA_MAX / c
    d[0] = np.clip(d[0], -GRADE_LUM_MAX, GRADE_LUM_MAX)
    return d.astype(np.float32)


def grade_person_clip(ctx, work_picture: Path, relit_files: list[Path], masks: list[np.ndarray], out: Path, k: float) -> dict:
    """The footage's person in a change of light, the autumn way (footage only, no render pixels on them): the relit
    footage everywhere else, and on them their own pixels with the change of colour and brightness that the new light
    made to the ground around them (person_bounce) laid on as a broad, capped shift, stronger toward their feet. Every
    pixel of detail is the footage's. Returns the details dict."""
    lock = []

    def one(i: int, f: np.ndarray):
        r = np.asarray(Image.open(relit_files[i]).convert("RGB"))
        h, w = r.shape[:2]
        m = upsize(masks[min(i, len(masks) - 1)], (w, h))
        soft = composite.blur(m.astype(np.float32), max(1.0, 1.2 * k)) if m.any() else np.zeros((h, w), np.float32)
        base = np.clip(np.rint(r * (1 - soft[..., None]) + f * soft[..., None]), 0, 255).astype(np.uint8)
        res = person_bounce(f, base, soft, GRADE_STRENGTH, GRADE_CHROMA_MAX, lum=GRADE_LUM, head=GRADE_HEAD, sky=sky_shift(f, r, soft))
        Image.fromarray(res).save(out / f"r_{i + 1:05d}.png")
        core = composite.shrink(m, max(2, round(4 * k)))
        c = None
        if core.sum() > 64:
            hp = lambda a: (lambda y: y - composite.smooth(y[..., None], 1.5)[..., 0])(a.astype(np.float32) @ LUM)  # noqa: E731
            a_, b_ = hp(res)[core], hp(f)[core]
            if a_.std() > 1e-3 and b_.std() > 1e-3:
                c = float(np.corrcoef(a_, b_)[0, 1])
        return c

    n = len(relit_files)
    for c in common.parallel_frames((f for i, f in enumerate(common.read_frames(work_picture)) if i < n), one):
        ctx.check()
        if c is not None:
            lock.append(c)
    if lock and min(lock) < LOCK_MIN:
        raise LockFailed(f"The kept person no longer matches your footage inside their outline (lowest {min(lock):.2f}).")
    return {"kept_as_shot": "the person: their own pixels, with the new light's colour and brightness laid on as a broad shift",
            "_qa": {"keep_lock_min": round(min(lock), 3) if lock else None, "keep_lock_median": round(float(np.median(lock)), 3) if lock else None,
                    "keep_method": "footage with a graded light"}}


def vegetation(ctx, clip: Path, d: Path) -> dict[str, list[np.ndarray]]:
    """SAM 3.1's trees, bushes, grass and people on every frame of `clip` (one job), per label."""
    job = ctx.run(graphs.select_any(ctx.client.assets.from_file(clip), AUTUMN_LABELS, prefix="pickups/environment_autumn"))
    found = {}
    for j, label in enumerate(AUTUMN_LABELS):
        out = d / label
        out.mkdir(parents=True, exist_ok=True)
        for o in job.get_outputs(f"PU_Mask{j if j else ''}"):
            o.to_file(out / o.name)
        masks = [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(out.glob("mask_*.png"))]
        found[label] = [m if m.mean() < NOTHING else np.zeros_like(m) for m in masks]
    return found


def autumn_take(ctx, work: Path, place: tuple[int, int, int, int, int, int], fps: float, label: str) -> dict:
    """Autumn with no generated pixels: the footage, its trees and grass recoloured (autumn_colour)."""
    cw, ch, ox, oy, w, h = place
    run_dir = ctx.vdir / "work"
    ctx.stage("Rendering the new weather")
    shown = work
    if (cw, ch) != (w, h):
        shown = run_dir / "shown.mp4"
        crop_clip(work, shown, (ox, oy, cw, ch))
    proxy = run_dir / "proxy.mp4"
    common.proxy_clip(ctx, shown, proxy)
    found = vegetation(ctx, proxy, run_dir / "veg")
    n = min(len(v) for v in found.values())
    if not n or not any(m.any() for lb in ("tree", "bush", "grass", "meadow") for m in found[lb]):
        raise RuntimeError("Couldn't find trees or grass in the shot to turn.")
    ctx.stage("Keeping it as shot")
    pw = found["tree"][0].shape[1]
    r = max(1, round(3 * pw / 1024))

    def soft(ms: list[np.ndarray], grow: int, sigma: float) -> list[np.ndarray]:
        return [composite.blur(composite.grow(m, grow).astype(np.float32), sigma) if m.any() else np.zeros(m.shape, np.float32)
                for m in ms]

    grassy = [a | b for a, b in zip(found["grass"][:n], found["meadow"][:n])]
    # SAM calls a meadow's tussocks bushes too: a bush in the grass is grass, else it is leaves.
    leaf = soft([t | (b & ~g) for t, b, g in zip(found["tree"][:n], found["bush"][:n], grassy)], r, r)
    grass = soft(grassy, r, r)
    people = soft(found["person"][:n], 0, 1.0)
    out = run_dir / "out"
    out.mkdir(exist_ok=True)

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((cw, ch), Image.BILINEAR))

    def one(i: int, frame: np.ndarray):
        j = min(i, n - 1)
        win = slice(max(0, j - AUTUMN_SMOOTH), j + AUTUMN_SMOOTH + 1)
        pic = frame[oy:oy + ch, ox:ox + cw]
        # SAM's edge across a far meadow is no edge in the picture: the weights follow the picture's own edges (a
        # forest's, a road's) and fade gently where there is none.
        small = np.asarray(Image.fromarray(pic).resize(people[j].shape[::-1], Image.BILINEAR)).astype(np.float32) / 255
        lw, gw = (up(np.clip(guided_colour(small, np.clip(np.mean(m[win], axis=0), 0, 1), 4 * r, 1e-2), 0, 1))
                  for m in (leaf, grass))
        ppl = up(people[j])
        Image.fromarray(person_bounce(pic, autumn_colour(pic, lw, gw, ppl), ppl)).save(out / f"r_{i + 1:05d}.png")
        return i

    total = media.probe(work).frames
    for i in common.parallel_frames(common.read_frames(work), one):
        ctx.check()
        ctx.progress((i + 1) / total)
    ctx.stage("Writing your files")
    downloads = common.finish_outputs(ctx.vdir, out / "r_%05d.png", fps, audio_from=shown)
    from pickups.app.store import make_viewer
    make_viewer(shown, ctx.vdir / "before.mp4")
    Image.open(out / "r_00001.png").save(ctx.vdir / "first_frame.png")
    return {"label": f"Weather · {label}", "task": "environment", "receipt": {"original": 100.0, "borrowed": 0.0, "ai": 0.0},
            "downloads": downloads,
            "details": {**common.sequence_details(out, "r_*.png", fps), "change": "autumn", "models": ["SAM 3.1"],
                        "_qa": {"method": "vegetation recoloured", "frames": total},
                        "note": "Every pixel is your footage; only the trees and grass changed colour."}}


def halo_lift(img: np.ndarray, mask: np.ndarray, k: float = 1.0) -> float | None:
    """QA: mean luma of the ring 6-12 px outside `mask` over the ring 12-24 px out (px scaled by k). About 0 for a
    clean edge, clearly positive when a light glow surrounds it."""
    if not mask.any():
        return None
    g6, g12, g24 = (composite.grow(mask, max(1, round(px * k))) for px in (6, 12, 24))
    r1, r2 = g12 & ~g6, g24 & ~g12
    if r1.sum() < 8 or r2.sum() < 8:
        return None
    y = img.astype(np.float32) @ LUM
    return float(y[r1].mean() - y[r2].mean())


MATTE_MARGIN = 0.03  # of its height: how far past SAM's outline the picture's own difference from the plate may reach
MATTE_REACH = 4  # frames either side a thin thing's matte is voted over (in its own coordinates)


def person_matte(o: np.ndarray, plate: np.ndarray | None, mask: np.ndarray, k: float = 1.0) -> np.ndarray:
    """A solid kept thing's matte (a person) from the picture itself: SAM's outline refined by a colour guided filter on
    the footage, so it follows hair, a sleeve, a bag strap against the sky or the path; solid inside. With a real `plate`
    (pixels from other frames), ground inside SAM's outline that equals the plate over a real patch (the strip of scree
    between an arm and the body, which SAM took in) is not them and drops out; a person has no gaps, so only patches
    wider than a few px, and never the solid core."""
    rows = np.flatnonzero(mask.any(axis=1))
    tall = rows[-1] - rows[0] + 1
    core = composite.shrink(mask, max(2, round(0.04 * tall)))
    limit = composite.grow(mask, max(2, round(0.02 * tall)))
    ground = np.zeros_like(mask)
    if plate is not None:
        diff = composite.blur(np.abs(o.astype(np.float32) - plate.astype(np.float32)).max(-1), max(0.8, 0.7 * k))
        like = diff < SAME_AS_PLATE
        r = max(3, round(3 * k))
        ground = composite.grow(composite.shrink(like, r), r) & like & mask
        # only if it opens onto the outside (a patch beside them), not an island inside their body
        ground &= composite.grow(~composite.grow(mask, 1), max(4, round(0.05 * tall)))
        core = core & ~composite.grow(ground, 1)
    a = np.clip(guided_colour(o.astype(np.float32) / 255, composite.blur(mask.astype(np.float32), max(0.7, 0.7 * k)),
                              max(2, round(2 * k)), 2e-4), 0, 1)
    a = np.where(ground, 0.0, a)
    return np.where(core, 1.0, a * limit).astype(np.float32)


def object_mattes(pairs, masks: list[np.ndarray], k: float, reach: int = MATTE_REACH, raw_matte=None) -> list[np.ndarray]:
    """The kept thing's matte on every frame, full size: each frame's raw matte (thin_matte, or person_matte for a solid person), then voted (median) over 2 * reach + 1
    frames after moving each onto the frame's own position (phase correlation of the mattes themselves), then snapped
    to the footage's edges by a guided filter. A rigid thing's outline is the same through the window, so the ragged
    fringe of any one frame's guess drops out and the outline doesn't jitter or start late."""
    n = len(masks)
    h, w = masks[0].shape
    union = np.logical_or.reduce(masks)
    if not union.any():
        return [np.zeros((h, w), np.float32)] * n
    ys, xs = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
    pad = int(0.15 * (ys[-1] - ys[0] + 1)) + 8
    y0, y1, x0, x1 = max(0, ys[0] - pad), min(h, ys[-1] + 1 + pad), max(0, xs[0] - pad), min(w, xs[-1] + 1 + pad)
    sl = (slice(y0, y1), slice(x0, x1))
    bh, bw = y1 - y0, x1 - x0
    raw, crops, msk = [], [], []
    for (o, q), m in zip(pairs, masks):
        mb = m[sl]
        msk.append(mb)
        crops.append(o[sl])
        raw.append((raw_matte or thin_matte)(o[sl], None if q is None else q[sl], mb, k) if mb.any() else np.zeros((bh, bw), np.float32))
    win = np.outer(np.hanning(bh), np.hanning(bw))
    prep = [(a - a.mean()) * win for a in raw]
    off = [np.zeros(2)]
    for i in range(1, n):
        both = raw[i - 1].any() and raw[i].any()
        off.append(off[-1] + (phase_offset(prep[i - 1], prep[i], 1.0) if both else 0.0))
    out = []
    for i in range(n):
        if not msk[i].any():
            out.append(np.zeros((h, w), np.float32))
            continue
        stack = []
        for j in range(max(0, i - reach), min(n, i + reach + 1)):
            if raw[j].any():
                dx, dy = off[i] - off[j]
                stack.append(raw[j] if (abs(dx) < 0.05 and abs(dy) < 0.05) else
                             composite.warp(raw[j], np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64), (bw, bh)))
        agg = np.median(np.stack(stack), axis=0)
        r = max(2, round(2 * k))
        tall = np.flatnonzero(msk[i].any(axis=1))
        limit = composite.grow(msk[i], max(3, round(MATTE_MARGIN * (tall[-1] - tall[0] + 1))))
        a = np.clip(guided_colour(crops[i].astype(np.float32) / 255, agg.astype(np.float32), r, 2e-4), 0, 1) * limit
        full = np.zeros((h, w), np.float32)
        full[sl] = a
        out.append(full)
    return out


LOCK_MIN = 0.95  # the high-pass of what is kept, against the footage's own, on every frame (the pixel lock)
TINT_OBJECT = 0.5  # how much of the new light's colour a kept car or object takes (a person takes all of it): same car, same paint
DRIFT_MIN, DRIFT_MAX = 0.75, 0.06  # px, and share of the width: a render's camera that strays from the footage's by this much is put back


class LockFailed(RuntimeError):
    """What was kept no longer matches the footage inside its matte."""


def edge_image(img: np.ndarray, hide: np.ndarray | None, size: int = 1024) -> np.ndarray:
    """Edges of a picture at `size` wide, windowed, for phase correlation; `hide` (a mask) is flattened first so the kept
    thing's own motion does not pull the match."""
    g = img.astype(np.float32) @ LUM
    if hide is not None and hide.any():
        g = np.where(hide, g[~hide].mean(), g)
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).resize((size, round(size * g.shape[0] / g.shape[1])), Image.BILINEAR)
    e = np.asarray(im.filter(ImageFilter.FIND_EDGES), np.float32)
    return (e - e.mean()) * np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1]))


def phase_offset(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Sub-pixel (dx, dy) in full-size px such that b's content has moved by it from a's (edge images, see edge_image)."""
    r = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    c = np.abs(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
    py, px = np.unravel_index(int(c.argmax()), c.shape)
    h, w = c.shape

    def peak(m: float, c0: float, p: float) -> float:
        d = m - 2 * c0 + p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m - p) / d

    fx = peak(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
    fy = peak(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
    return np.array([(px - w if px > w // 2 else px) + fx, (py - h if py > h // 2 else py) + fy]) * scale


def shifted(img: np.ndarray, dx: float, dy: float, pad: int = 16) -> np.ndarray:
    """A picture's content moved by (dx, dy) px, the edge repeated into what it uncovers."""
    h, w = img.shape[:2]
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    out = composite.warp(p, M, (w + 2 * pad, h + 2 * pad))
    return np.clip(np.rint(out[pad:pad + h, pad:pad + w]), 0, 255).astype(np.uint8)


def smooth_offsets(d: np.ndarray, reach: int = 1) -> np.ndarray:
    """A median then a mean over 2 * reach + 1 frames (one bad match is no jump)."""
    n = len(d)
    med = np.stack([np.median(d[max(0, i - reach):i + reach + 1], axis=0) for i in range(n)])
    return np.stack([med[max(0, i - reach):i + reach + 1].mean(axis=0) for i in range(n)])


def keep_pass(ctx, work: Path, gen_files: list[Path], kept: list[np.ndarray], out: Path, fps: float, grain: str,
              obj: str, box: tuple[int, int, int, int] | None = None, plate: Path | None = None, haze: bool = False,
              real_plate: bool = True, light_bounds: tuple[tuple[float, float], float] = (GAIN_RANGE, HAZE_MAX),
              see_through: bool = True, reference: tuple[list[Path], list[np.ndarray]] | None = None,
              person: bool = False, veil_scale: float | None = None) -> tuple[dict, dict]:
    """Put the kept thing back over every rendered frame (grain goes on the render first, so the kept thing keeps the
    footage's own) and write the frames as out/r_NNNNN.png, without the canvas's padding (`box`: x, y, w, h), plus
    receipt.mp4: the render tinted, the kept thing not. Returns (receipt, details).

    `plate` (the cleared clip the render was made from): the render has no copy of it, so it is relit from the change
    from that plate to the render around it (keep_relit), one steady light per frame; otherwise detail transfer from
    the render's own copy (keep_over). `reference` (the frames of a render made with the thing in it, and its masks of
    the thing there): the broad light on the thing comes from that drawing, the detail stays the footage's
    (subject_light)."""
    with Image.open(gen_files[0]) as im:
        gw, gh = im.size
    n = len(gen_files)
    ox, oy, cw, ch = box or (0, 0, gw, gh)
    k = max(cw, ch) / 1920
    need = common.GRAIN.get(grain)
    rec = common.FrameWriter(ctx.vdir / "receipt.mp4", cw, ch, fps, kind="h264", scale_long=1920)
    blank = np.zeros_like(kept[0])
    qs = max(1, round(max(cw, ch) / 480))  # QA on every qs-th pixel

    def cut(a: np.ndarray) -> np.ndarray:
        return a[oy:oy + ch, ox:ox + cw]

    drift: np.ndarray | None = None

    def raw_render(j: int) -> np.ndarray:
        return cut(np.asarray(Image.open(gen_files[min(max(j, 0), n - 1)]).convert("RGB")))

    def render(j: int) -> np.ndarray:
        g = raw_render(j)
        if drift is not None:
            dx, dy = drift[min(max(j, 0), n - 1)]
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                g = shifted(g, float(dx), float(dy))
        return g

    def footage(f: np.ndarray) -> np.ndarray:  # at the render's size (the same framing: both are the work clip)
        return cut(f if f.shape[:2] == (gh, gw) else np.asarray(Image.fromarray(f).resize((gw, gh), Image.LANCZOS)))

    def mask_at(i: int) -> np.ndarray:
        return cut(upsize(kept[i] if i < len(kept) else blank, (gw, gh)))

    def pairs():
        src = common.read_frames(work)
        pl = common.read_frames(plate) if plate is not None else itertools.repeat(None)
        for i, (f, q) in enumerate(zip(src, pl)):
            if i >= n:
                break
            yield f, q

    if plate is not None and n > 2:
        # The render's own camera wanders from the footage's (a few px over the first frames), and the footage's kept thing
        # then seems to jump against it: the render is moved back onto the footage's camera, frame by frame.
        scale = cw / 1024
        cs = cg = np.zeros(2)
        prev = None
        offs = []
        for i, (f, _) in enumerate(pairs()):
            hide = composite.grow(mask_at(i), max(4, round(12 * k)))
            es, eg = edge_image(footage(f), hide), edge_image(raw_render(i), hide)
            if prev is not None:
                cs = cs + phase_offset(prev[0], es, scale)
                cg = cg + phase_offset(prev[1], eg, scale)
            prev = (es, eg)
            offs.append(cs - cg)
        d = smooth_offsets(np.array(offs))
        if DRIFT_MIN <= float(np.abs(d).max()) <= DRIFT_MAX * cw:
            drift = d

    lights: list[np.ndarray] = []
    if plate is not None:
        def learn(i: int, fq) -> np.ndarray | None:
            o, q = (footage(x) for x in fq)
            mask = mask_at(i)
            if not mask.any():
                return None
            changed = np.abs((o.astype(np.float32) - q) @ LUM) > 6  # its shadow: the plate is not the scene there
            return scene_light(q, render(i), mask, k, haze=haze, shadow=changed, gain_range=light_bounds[0],
                               haze_max=light_bounds[1])
        lights = steady(list(common.parallel_frames(pairs(), learn)))
        # An additive veil (haze, rain) flattens a thing's own detail unevenly, dark pixels most: a person takes none of it (their
        # detail correlated 0.945-0.97 with the footage's with it, 0.995 without); an object takes all of it unless the lock fails.
        vs = (0.0 if person else 1.0) if veil_scale is None else veil_scale
        lights = [np.stack([l[0], l[1] * vs]).astype(np.float32) for l in lights]
        if not person:  # same car, same paint: a kept car or object takes only part of the new light's colour
            lights = [np.stack([np.exp(np.log(l[0]).mean() + TINT_OBJECT * (np.log(l[0]) - np.log(l[0]).mean())), l[1]]).astype(np.float32)
                      for l in lights]

    mattes: list[np.ndarray] | None = None
    if plate is not None and see_through and real_plate:
        # A thing with real gaps through it is matted on the footage itself, voted over its own motion (object_mattes).
        mattes = object_mattes(((footage(f), footage(q)) for f, q in pairs()), [mask_at(i) for i in range(n)], k)
    elif plate is not None and person:
        # A person likewise: SAM's outline refined on their own frame (a colour guided filter), voted over 9 frames.
        mattes = object_mattes(((footage(f), footage(q) if real_plate else None) for f, q in pairs()),
                               [mask_at(i) for i in range(n)], k, raw_matte=person_matte)

    shades: list = []
    if plate is not None and reference is not None:
        ref_files, ref_masks = reference
        shape = subject_light.grid_shape([mask_at(i) for i in range(n)])

        def reference_frame(i: int) -> np.ndarray:
            return cut(np.asarray(Image.open(ref_files[min(i, len(ref_files) - 1)]).convert("RGB")))

        def learn_shade(i: int, fq):
            mask = mask_at(i)
            if not mask.any() or i >= len(ref_masks):
                return None
            base = relight(footage(fq[0]), lights[i])
            return subject_light.frame_grid(base, reference_frame(i), mask, cut(upsize(ref_masks[i], (gw, gh))), shape)
        shades = subject_light.steady_grids(list(common.parallel_frames(pairs(), learn_shade)))

    def tinted(f: np.ndarray | None) -> np.ndarray | None:
        if f is None or person:
            return f
        lum = f.mean(axis=-1, keepdims=True)
        return lum + TINT_OBJECT * (f - lum)

    def locked(frame: np.ndarray, orig: np.ndarray, own: np.ndarray) -> float | None:
        """Inside the kept matte (3 px in: its rim is part decontaminated colour) the high-pass of the result against the
        footage's, as a correlation: the pixel lock."""
        solid = composite.shrink(own > 0.98, max(2, round(3 * k)))
        if solid.sum() <= 64:
            return None
        ys, xs = np.flatnonzero(solid.any(axis=1)), np.flatnonzero(solid.any(axis=0))
        bx = (slice(max(0, ys[0] - 4), ys[-1] + 5), slice(max(0, xs[0] - 4), xs[-1] + 5))
        hp = lambda a: (lambda y: y - composite.smooth(y[..., None], 1.5)[..., 0])(a.astype(np.float32) @ LUM)  # noqa: E731
        a_, b_ = hp(frame[bx])[solid[bx]], hp(orig[bx])[solid[bx]]
        return float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 1e-3 and b_.std() > 1e-3 else None

    def one(i: int, fq):
        g = render(i)
        orig = footage(fq[0])
        if need is not None:
            g = composite.add_grain(g, need, seed=i, size=0.6)
        mask = mask_at(i)
        lock = None
        if plate is None:
            frame, own, sigma = keep_over(orig, g, mask, k, around=(render(i - 1), render(i + 1)))
            lock = locked(frame, orig, own)
        else:
            for use in ((mattes[i] if mattes else None), None):  # a voted matte that breaks the lock on a frame gives way to SAM's own
                frame, own, sigma = keep_relit(orig, footage(fq[1]), g, mask, lights[i], k, around=(render(i - 1), render(i + 1)),
                                               real_plate=real_plate, see_through=see_through,  # no flakes: they'd read as dirt
                                               shade=tinted(subject_light.field(shades[i], mask)) if shades else None, person=person,
                                               matte=use)
                lock = locked(frame, orig, own)
                if use is None or lock is None or lock >= LOCK_MIN:
                    break
        Image.fromarray(frame).save(out / f"r_{i + 1:05d}.png")
        ai = (1 - own)[..., None] * 0.55
        rec_frame = np.clip(frame * (1 - ai) + AI_TINT * ai, 0, 255).astype(np.uint8)
        core = composite.shrink(mask, round(6 * k))[::qs, ::qs]
        q = {"lock": lock, "out": frame[::qs, ::qs], "src": orig[::qs, ::qs], "core": core, "sigma": sigma, "found": bool(mask.any()),
             "halo": halo_lift(frame[::qs, ::qs], mask[::qs, ::qs], k / qs),
             "light": float(((frame[::qs, ::qs].astype(np.float32) - orig[::qs, ::qs]) @ LUM)[core].mean()) if core.any() else None}
        return rec_frame, float(own.sum()), q

    share = 0.0
    qa: dict[str, list] = {"sigma": [], "halo": [], "light": [], "d_out": [], "d_src": [], "lock": []}
    found, prev = 0, None
    for i, (rec_frame, own, q) in enumerate(common.parallel_frames(pairs(), one)):
        ctx.check()
        rec.write(rec_frame)
        share += own
        if q["found"]:
            found += 1
            if q["lock"] is not None:
                qa["lock"].append(q["lock"])
                if q["lock"] < LOCK_MIN:
                    qa.setdefault("lock_bad", []).append(i)
            qa["sigma"].append(q["sigma"])
            for key in ("halo", "light"):
                if q[key] is not None:
                    qa[key].append(q[key])
        if prev is not None:
            both = q["core"] & prev["core"]
            if both.any():
                qa["d_out"].append(float(np.abs(q["out"].astype(np.float32) - prev["out"])[both].mean()))
                qa["d_src"].append(float(np.abs(q["src"].astype(np.float32) - prev["src"])[both].mean()))
        prev = q
        ctx.progress((i + 1) / n)
    rec.close()
    if qa.get("lock_bad"):
        raise LockFailed(f"The kept {obj} no longer matches your footage inside its outline on frames "
                         f"{', '.join(str(i) for i in qa['lock_bad'][:8])}: its own detail correlates under {LOCK_MIN:g} "
                         f"(lowest {min(qa['lock']):.2f}).")

    def med(xs: list) -> float | None:
        return round(float(np.median(xs)), 2) if xs else None

    original = round(100 * share / (cw * ch * n), 2)
    receipt = {"original": original, "borrowed": 0.0, "ai": round(100 - original, 2)}
    flicker = round(med(qa["d_out"]) / max(med(qa["d_src"]), 0.5), 2) if qa["d_out"] else None
    details = {"kept_as_shot": f"{obj}: its detail as shot, its light from the render ({original:g}% of the picture)",
               # QA for the stress harness (never shown): flicker ~1 is as steady as the footage, halo ~0 is a clean edge.
               "_qa": {"keep": obj, "keep_frames_found": found, "frames": n, "keep_share": original,
                       "keep_sigma_px": med(qa["sigma"]), "keep_light_shift": med(qa["light"]), "keep_halo": med(qa["halo"]),
                       "keep_flicker": flicker, "keep_lock_min": round(min(qa["lock"]), 3) if qa["lock"] else None,
                       "keep_lock_median": med(qa["lock"]), "keep_lock_frames": [round(x, 3) for x in qa["lock"]],
                       "render_camera_moved_px": round(float(np.abs(drift).max()), 1) if drift is not None else 0.0, "keep_method": "detail transfer" if plate is None else "relit on a clean plate",
                       **({"keep_light_from": "the render's own drawing of it", "keep_shade_frames": sum(x is not None for x in shades),
                           "keep_shade_median": [round(float(x), 3) for x in np.median(
                               [g.mean(axis=(0, 1)) for g in shades if g is not None], axis=0)] if any(x is not None for x in shades) else None}
                          if shades else {}),
                       **({"keep_gain": [round(float(x), 3) for x in np.median(lights, axis=0)[0]],
                           "keep_veil": [round(float(x), 4) for x in np.median(lights, axis=0)[1]]} if lights else {})}}
    return receipt, details
