"""Replace a car, v2 (production path): a new car drives the old one's path, everything else stays real footage.

The first car swaps (tasks/vehicle_swap.py) drew the new car with Layout-to-Render on a stand-in and pasted it back into a
rectangle: two backgrounds mixed, warped bodies, edge flicker. This path follows the official recipes instead:

1. Track. SAM 3.1 follows the old car (Recast on LTX did that already); its shadow, spray and wet-road reflection are tracked as
   separate regions (they go with it).
2. Clear the road. LTX-2.5's Clean-Plate IC-LoRA (1024x576, its trained size) draws the road, sea and sky behind the old car.
   It is seed-dependent (4 of 5 seeds clean): a check looks for the old car left in it and retries once with the next seed.
3. Paint. The first frame with the new car (Qwen-Image 2.1 edit, the existing paint graph), matched to the old car's size,
   position and heading. A size-and-position check against the tracked car repaints on a miss, then fits the painted car to
   the old one's box as a last resort.
4. Generate. EditAnything v0.1 motion transfer on LTX-2.5 (distilled): the original clip is the guide, the painted frame the
   reference. It runs on a crop around the car's path at up to 1280x704 for resolution. The crop is only the model's window.
5. Matte. LTX Alpha-Gen on the generated clip, and a SAM 3.1 track of the new car; the steadier of the two frame to frame.
6. Composite. Per frame the region that changes is car-shaped: the new car, its shadow and reflection band (from the render,
   luminance only), and the old car's footprint, filled from the plate (toned to the footage by a membrane, sharpness and
   grain matched). A Laplacian-pyramid seam blend with no dilation, light wrap, grain. Outside the region's support the output
   is the footage bit for bit.
   The distilled model (8 steps) goes first: fast, and its wheels hold their spokes. The retry uses the dev model (30 steps, CFG 4;
   or `dev_render`): the distilled one drifted back to the old car's shape within 25 frames of the painted frame on the canyon
   bend, and the dev one held the new car through it (but drew the wheels as mushy discs on the side-on shot).
7. Gate. Zero pixels changed outside, halo, edge flicker, wheel contact, no straight (rectangle) seams, shape stability, the
   old car left over. One retry (the next seed) on a failure that is cheap to detect.
"""

from __future__ import annotations

import colorsys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite, graphs, media
from pickups.app.comfy import Canceled
from pickups.app.jobs import Stage
from pickups.app.store import make_viewer
from pickups.tasks import car_blend as cb
from pickups.tasks import common, recast, remove, vehicle_swap as vs

STAGES = [  # an RTX 4090, 97 frames of an HD shot (seconds)
    Stage("Preparing the shot", 15),
    Stage("Finding the vehicle", 35),
    Stage("Clearing the road", 120),
    Stage("Painting the new car", 60),
    Stage("Rendering the new car", 140),
    Stage("Cutting it out", 80),
    Stage("Blending at full resolution", 90),
    Stage("Checking the result", 40),
    Stage("Writing your files", 30),
]
FIND, CLEAR, PAINT, RENDER, CUT, BLEND, CHECK, WRITE = (s.label for s in STAGES[1:])

PAINT_TRIES = 3
PLATE_TRIES = 2
PLATE_FRAME_OK = 1.0         # a plate frame counts as clean when its difference from the footage is at least this times PLATE_GONE
PLATE_REPAIRS, PLATE_LEAD, PLATE_RAMP = 2, 8, 6   # extra windows for frames none shows clean, how far before them they start, the crossfade between windows (frames)
PLATE_WINDOW, PLATE_STRIDE = 49, 24  # Clean-Plate keeps the car out for about 49 frames; a longer clip gets it back in
RENDER_TRIES = 2
SIZE_OK = (0.85, 1.2)        # the painted car's width against the old car's
HEIGHT_OK = (0.5, 1.2)       # its height (a low coupe is shorter than an SUV)
CENTRE_OK = 0.10             # of the old car's width: how far off its centre may be
BASE_OK = 0.07               # of the old car's height: how far its contact line may be above or below the old one's
PLATE_GONE = 20.0            # levels: the clean plate differs from the footage at least this much (median) where the old car was
PLATE_FRAME_MIN = 0.35       # ...and on no frame less than this share of the median (a ghost of the car coming back)
OLD_LEFT_MAX = 0.35          # share of the old car's footprint still showing the footage
GAP_MAX = 0.09               # of the old car's height: a tyre's float above the road line
SINK_MAX = 0.5               # ...and how far below it (perspective: a nearer tyre stands well below the old car's lower hull; only an absurd sink counts)
HALO_MAX = 0.5
COPY_PEAK_MAX, COLOUR_DRIFT_MAX, SIZE_JUMP_MAX = 0.12, 24.0, 0.2  # a stretch of frames going back to the old car; colour off the painted car; size jumping in 4 frames
COPY_MAX = 0.2               # the render's fine detail correlating with the old car's this much is the old car again (real swaps: ~0)
FLICKER_MAX = 0.6
SHAPE_MIN = 0.70
SPIN_STEP, SPIN_BLUR, SPIN_SAMPLES = 0.6, 1.0, 17   # radians per frame the wheels are turned, and the rotational blur of one frame's shutter
OLD_DILATE = 3               # px at HD: the old car's footprint grows this far to take in its soft edge
GENERIC_NEGATIVE = "cut, scene change, blurry, distorted, extra wheels, text, watermark"


# --- small helpers -------------------------------------------------------------------------------------------------------


def colour_name(rgb: np.ndarray) -> str:
    """A car's colour in a word, from the median colour of its pixels (for "Replace the <colour> SUV ...")."""
    r, g, b = (float(x) / 255 for x in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.12:
        return "white" if v > 0.8 else "silver" if v > 0.5 else "dark grey"
    hue = h * 360
    for lim, name in ((15, "red"), (45, "orange"), (70, "yellow"), (165, "green"), (255, "blue"), (290, "purple"), (345, "pink"),
                      (361, "red")):
        if hue < lim:
            return ("dark " + name) if v < 0.45 and name not in ("black",) else name
    return "dark"


def _png(a: np.ndarray, path: Path) -> Path:
    Image.fromarray(a).save(path)
    return path


def _frames_to_mp4(folder: Path, fps: float, out: Path, pattern: str = "%04d.png") -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(folder / pattern), "-c:v", "libx264", "-crf", "8",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def fit_check(new: np.ndarray, old: np.ndarray, ratio: float | None = None) -> dict:
    """The painted car against the tracked one: width ratio, height ratio, centre offset and contact-line offset (as shares of
    the old car's width or height). ok when all are in range; `miss` scores how far off (0 = ok)."""
    if not new.any() or not old.any():
        return {"ok": False, "miss": 9.0, "why": "no car found"}
    nx0, ny0, nx1, ny1 = vs._box(new)
    ox0, oy0, ox1, oy1 = vs._box(old)
    ow, oh = max(ox1 - ox0, 1.0), max(oy1 - oy0, 1.0)
    wr, hr = (nx1 - nx0) / ow, (ny1 - ny0) / oh
    cen = abs((nx0 + nx1) - (ox0 + ox1)) / 2 / ow
    base = (ny1 - oy1) / oh  # + : the new car stands lower in the picture
    aspect = ((nx1 - nx0) / max(ny1 - ny0, 1.0)) / ((ox1 - ox0) / max(oy1 - oy0, 1.0))
    miss = max(0.0, ASPECT_OK[0] - aspect) + max(0.0, aspect - ASPECT_OK[1]) + max(0.0, SIZE_OK[0] - wr) + max(0.0, wr - SIZE_OK[1]) + max(0.0, HEIGHT_OK[0] - hr) + max(0.0, hr - HEIGHT_OK[1]) + \
        max(0.0, cen - CENTRE_OK) * 2 + max(0.0, abs(base) - BASE_OK) * 2
    return {"aspect_ratio": round(aspect, 3), "width_ratio": round(wr, 3), "height_ratio": round(hr, 3), "centre_off": round(cen, 3), "base_off": round(base, 3),
            "miss": round(miss, 4), "ok": miss == 0}


FRONT_LABELS, REAR_LABELS = ("headlight", "grille"), ("tail light",)
KEY_EXACT = 0.01             # exact_size: a later keyframe more than this share off the first one's width ratio is refit
KEY_SIZE_TOL = 0.06           # later keyframes' width, as a share of the first one's width against the old car's (the car must not change size between windows)
ASPECT_OK = (0.7, 1.7)      # the painted car's box aspect against the old car's (a side view stays a side view)


def view_end(ctx, png: Path, car: np.ndarray, out: Path, nearer: float = 1.0) -> str | None:
    """Which end of a car (not seen from the side) the picture shows: "front" or "rear", None when it can't tell. From SAM 3.1's
    headlights against its tail lights inside the car (a grille is read only when there are no tail lights: a rear bumper reads
    as one), else from how it moves (`nearer`, vs.approach: it grows as it comes at the camera, shrinks as it goes)."""
    if not car.any():
        return None
    near = composite.grow(car, 4)

    def count(label: str) -> int:
        try:
            found = common.select_instances(ctx, png, label, 1.0, Path(f"{out}_{label.replace(' ', '_')}"), min_frac=0.0)
        except Exception:  # noqa: BLE001
            return 0
        return sum(1 for m in found if m.shape == car.shape and (m & near).sum() >= 0.5 * m.sum())

    heads, tails = count("headlight"), count("tail light")
    if tails >= heads + 2 or (tails >= 1 and heads == 0):
        return "rear"
    if heads >= tails + 2 or (heads >= 1 and tails == 0):
        return "front"
    if not tails and not heads and count("grille") >= 1:
        return "front"
    if nearer >= vs.COMING:
        return "front"
    if nearer <= 1 / vs.COMING:
        return "rear"
    return None


def look_of(ctx, png: Path, car: np.ndarray, out: Path) -> tuple[str | None, str | None]:
    """How a car is seen: (side, end). `side` is which side of the picture a car seen from the side or three-quarters points its
    nose to ("left", "right"), `end` which end a car seen head-on or from behind shows ("front", "rear"); None when not told.
    From where SAM 3.1 finds its headlights against its tail lights, inside the car."""
    if not car.any():
        return None, None
    x0, y0, x1, y1 = vs._box(car)
    width = x1 - x0
    sideways = width / max(y1 - y0, 1.0) >= 1.9
    near = composite.grow(car, 4)

    def xs(labels) -> list[float]:
        got = []
        for label in labels:
            try:
                found = common.select_instances(ctx, png, label, 1.0, Path(f"{out}_{label.replace(' ', '_')}"), min_frac=0.0)
            except Exception:  # noqa: BLE001
                continue
            for m in found:
                if m.shape == car.shape and (m & near).sum() >= 0.5 * m.sum():
                    got.append(float(np.nonzero(m)[1].mean()))
        return got

    front, rear = xs(FRONT_LABELS[:1]), xs(REAR_LABELS)  # headlights against tail lights (a rear bumper reads as a "grille")
    if not front and not rear:
        front = xs(FRONT_LABELS[1:])
    if not sideways:
        return None, None
    cx = (x0 + x1) / 2
    if front and rear:
        d = np.mean(front) - np.mean(rear)
        return (None if abs(d) < 0.15 * width else ("left" if d < 0 else "right")), None
    if front and abs(np.mean(front) - cx) > 0.15 * width:
        return ("left" if np.mean(front) < cx else "right"), None
    if rear and abs(np.mean(rear) - cx) > 0.15 * width:
        return ("right" if np.mean(rear) < cx else "left"), None
    return None, None


def refit(painted: np.ndarray, car: np.ndarray, old: np.ndarray, ground: np.ndarray, flip: bool = False, ratio: float = 1.0
          ) -> tuple[np.ndarray, np.ndarray]:
    """The last resort for a painting that is the wrong size or in the wrong place: its car cut out, scaled about its contact line
    to the old car's width, moved onto the old car's centre and contact line, and set on `ground` (the anchor frame with the old
    car taken out). Returns (picture, the car's new mask)."""
    if flip:  # facing the wrong way: mirrored about its own centre
        nx0, _, nx1, _ = vs._box(car)
        cx = (nx0 + nx1) / 2
        W = car.shape[1]
        xs = np.arange(W)
        src = np.clip(np.rint(2 * cx - xs).astype(int), 0, W - 1)
        painted, car = painted[:, src], car[:, src]
    nx0, ny0, nx1, ny1 = vs._box(car)
    ox0, oy0, ox1, oy1 = vs._box(old)
    s = float(np.clip(ratio * (ox1 - ox0) / max(nx1 - nx0, 1.0), 0.5, 2.0))
    H, W = car.shape
    # the map from the painting to the result: scale s about the new car's bottom centre, then onto the old one's
    bx, by = (nx0 + nx1) / 2, ny1
    tx, ty = (ox0 + ox1) / 2, oy1
    M = np.array([[s, 0, tx - s * bx], [0, s, ty - s * by], [0, 0, 1.0]])
    layer = np.dstack([painted.astype(np.float32) * car[..., None], car.astype(np.float32)])
    wl = composite.warp(layer, M, (W, H), cubic=True)
    a = np.clip(wl[..., 3:4], 0, 1)
    out = np.clip(np.rint(wl[..., :3] + ground.astype(np.float32) * (1 - a)), 0, 255).astype(np.uint8)
    return out, wl[..., 3] > 0.5


# --- the clean plate -----------------------------------------------------------------------------------------------------


def plate_diffs(plate_files: list[Path], full: Path, pm: list[np.ndarray], size: tuple[int, int], start: int = 0) -> dict[int, float]:
    """For each plate frame (plate_files[j] is the work clip's frame start + j): how far it is from the footage where the old car
    was (median luma levels over the old car's eroded outline), by the work clip's frame number. Frames where the car isn't
    tracked are left out."""
    pw, ph = size
    diffs: dict[int, float] = {}
    for j, f in enumerate(common.read_frames(full, vf=f"trim=start_frame={start},setpts=PTS-STARTPTS,scale={pw}:{ph}:flags=bilinear", size=(pw, ph))):
        if j >= len(plate_files) or start + j >= len(pm):
            break
        m = composite.erode_box(recast.upsize(pm[start + j], (pw, ph)), 2)
        if m.sum() < 50:
            continue
        pl = np.asarray(Image.open(plate_files[j]).convert("RGB").resize((pw, ph), Image.BILINEAR))
        d = np.abs(composite.luma709(pl.astype(np.float32)) - composite.luma709(f.astype(np.float32)))
        diffs[start + j] = float(np.median(d[m]))
    return diffs


def plate_verdict(by_frame: dict[int, float]) -> dict:
    """Is the old car gone: the median of the differences against PLATE_GONE, and the weakest frame against the median (a
    translucent car ghosting back in on some frames)."""
    diffs = list(by_frame.values())
    if not diffs:
        return {"ok": False, "median": 0.0, "weakest": 0.0}
    med, low = float(np.median(diffs)), float(min(diffs))
    return {"ok": bool(med >= PLATE_GONE and low >= PLATE_FRAME_MIN * med), "median": round(med, 1), "weakest": round(low, 1)}


def plate_check(plate_dir: Path, full: Path, pm: list[np.ndarray], size: tuple[int, int]) -> dict:
    """plate_verdict for a whole plate folder (frames 0001.png ...)."""
    return plate_verdict(plate_diffs(sorted(plate_dir.glob("*.png")), full, pm, size))


def remove_plate(ctx, s: vs.Shot, pm: list[np.ndarray], gone: list[np.ndarray], pdir: Path, qa: dict) -> Path | None:
    """The fallback when Clean-Plate leaves the old car in on every seed: the first car swap's plate (Remove's fill: a painted
    clean first frame and real pixels from other frames where the camera moves, LTX inpainting where it doesn't), as full-size
    frames in pdir/remove. None when its own check finds the car left behind too."""
    W, H = s.size
    fill = (s.p.get("fill") or "").strip() or vs.fill_words(s.p)
    stays, _ = vs.keeps_its_place(pm)
    nearer = vs.approach(pm)
    sized = nearer >= vs.COMING or nearer <= 1 / vs.COMING
    tdir = pdir / "remove_try"
    tdir.mkdir(exist_ok=True)
    bg = remove.background(ctx, s.full, pm, gone, (W, H), s.fps, tdir, seed=s.seed, fill=fill, shadow=bool(s.p.get("shadow", True)),
                           camera="moving" if stays and not sized else None, given_locked=True)
    r = vs.remnants(bg, s.full, pm, (W, H))
    out = pdir / "remove"
    out.mkdir(exist_ok=True)
    for i, f in enumerate(common.read_frames(s.full)):
        Image.fromarray(bg.plate(i, f)[0]).save(out / f"{i + 1:04d}.png")
    qa["plate_fallback"] = {"method": bg.method(), "ok": bool(r["ok"])}
    return out if r["ok"] else None


def make_plate(ctx, s: vs.Shot, pm: list[np.ndarray], pdir: Path, qa: dict, gone: list[np.ndarray] | None = None) -> Path:
    """The clean plate (1024x576 frames in pdir/try<seed>): Clean-Plate on the whole work clip, checked, retried once with the
    next seed. Returns the folder of the better one."""
    cw, ch = graphs.CLEAN_PLATE_SIZE
    clip = pdir / "plate_in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(s.full), "-vf", f"scale={cw}:{ch}:flags=lanczos", "-c:v", "libx264",
                    "-crf", "4", "-pix_fmt", "yuv420p", "-an", str(clip)], check=True)
    first = int(s.p.get("clear_seed", s.seed))
    n = s.frames
    wins = cb.windows(n, PLATE_WINDOW, PLATE_STRIDE)
    d = pdir / "try0"
    d.mkdir(exist_ok=True)
    got: dict[int, list[tuple[int, int, np.ndarray, bool]]] = {}  # frame -> [(window start, its length, plate, the car is gone from it)]
    tries: list[dict] = []
    planned = len(wins) * PLATE_TRIES

    def window(j: int, a: int, length: int, seeds: list[int], done: int) -> None:
        """Clean-Plate on frames a..a+length-1, each seed checked on its own (the car comes back into some windows); the best one's
        frames are kept, each marked clean or not."""
        wclip = pdir / f"plate_win{j}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"trim=start_frame={a}:end_frame={a + length},setpts=PTS-STARTPTS",
                        "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", "-an", str(wclip)], check=True)
        asset = ctx.client.assets.from_file(wclip)
        best_w, best_score, best_d = None, -1.0, {}
        for t, seed_j in enumerate(seeds):
            ctx.check()
            with vs.span(ctx, min(1.0, (done + t) / planned), min(1.0, (done + t + 1) / planned)):
                job = ctx.run(graphs.clean_plate_road(asset, s.fps, seed=seed_j))
            wd = pdir / f"win{j}_s{seed_j}"
            wd.mkdir(exist_ok=True)
            files = []
            for i, o in enumerate(job.get_outputs("PU_Out")[:length]):
                f = wd / f"{i + 1:04d}.png"
                o.to_file(f)
                files.append(f)
            dif = plate_diffs(files, s.full, pm, (cw, ch), start=a)
            chk = plate_verdict(dif)
            tries.append({"window": j, "start": a, "seed": seed_j, **chk})
            score = chk["median"] * (1.0 if chk["ok"] else 0.5) + 0.3 * chk["weakest"]
            if best_w is None or score > best_score:
                best_w, best_score, best_d = files, score, dif
            if chk["ok"]:
                break
        assert best_w is not None
        for i, f in enumerate(best_w):
            # a frame of a window that still shows the car (the middle window of C1 did on every seed) is left out where another
            # window covers it cleanly
            clean = best_d.get(a + i, PLATE_GONE) >= PLATE_FRAME_OK * PLATE_GONE
            got.setdefault(a + i, []).append((a, length, np.asarray(Image.open(f).convert("RGB"), np.float32), clean))

    for j, (a, length) in enumerate(wins):
        window(j, a, length, [first + t + 10 * j for t in range(PLATE_TRIES)], j * PLATE_TRIES)
    # Frames no window shows clean (C1's frames 36-48: the old SUV's roof ghosted back in at the end of one window and the next
    # failed): windows that start just before each such run, where Clean-Plate keeps the car out.
    extra = len(wins)
    for rnd in range(PLATE_REPAIRS):
        bad = [i for i in range(n) if not any(c[3] for c in got.get(i, []))]
        if not bad:
            break
        planned += PLATE_TRIES
        a0 = bad[0]
        run_end = a0
        while run_end + 1 in bad:
            run_end += 1
        a = int(np.clip(a0 - PLATE_LEAD, 0, max(0, n - PLATE_WINDOW)))
        length = min(PLATE_WINDOW, n - a)
        length = 8 * ((length - 1) // 8) + 1
        window(extra, a, length, [first + 100 + 10 * rnd + t for t in range(PLATE_TRIES)], planned - PLATE_TRIES)
        extra += 1
    qa["plate_repair_windows"] = extra - len(wins)
    for i in range(n):  # each frame from the windows that show it clean, the freshest first (a short crossfade between two of them)
        cover = got.get(i, [])
        usable = [c for c in cover if c[3]] or cover
        ws = [(cb.window_weight(i, a, ln, PLATE_RAMP), img) for a, ln, img, _ in usable]
        tot = sum(w for w, _ in ws) or 1.0
        mix = sum(w * img for w, img in ws) / tot
        Image.fromarray(np.clip(np.rint(mix), 0, 255).astype(np.uint8)).save(d / f"{i + 1:04d}.png")
    chk = plate_check(d, s.full, pm, (cw, ch))
    tries.append({"window": "all", **chk})
    best = d
    qa["plate"] = {"tries": tries, "size": [cw, ch], "ok": bool(tries[-1]["ok"])}
    assert best is not None
    if not qa["plate"]["ok"] and gone is not None:  # Clean-Plate left the car in: the first swap's plate instead
        fb = remove_plate(ctx, s, pm, gone, pdir, qa)
        if fb is not None:
            qa["plate"]["used"] = "remove"
            return fb
    qa["plate"]["used"] = "clean-plate"
    return best


# --- the paint -----------------------------------------------------------------------------------------------------------


def paint_new_car(ctx, s: vs.Shot, pm_full: np.ndarray, k: int, anchor: np.ndarray, ground: np.ndarray, qa: dict, tag: str = "",
                  reference: np.ndarray | None = None, ratio: float | None = None, exact_size: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The anchor frame with the new car, size- and position-checked against the old car. Returns (picture, the new car's mask).
    `tag`: names this keyframe's files and qa ("" for the first). `reference`: a picture of the car as first painted (RGB): later
    keyframes are painted to look exactly like it, so the car is the same car all through the shot. `exact_size`: a later
    keyframe whose car passed the check but is not the first one's width against the old car (within KEY_EXACT) is refit to it
    anyway, so the car keeps one size through the shot (the painter tends to draw later keyframes ~10% bigger)."""
    from pickups.tasks import recast_ltx

    p, run_dir, seed = s.p, s.run_dir, s.seed
    W, H = s.size
    anchor_png = _png(anchor, run_dir / f"anchor_full{tag}.png")
    picture = bool(p.get("ref_upload_id") or p.get("who_ref") or p.get("reference_take")) or reference is not None
    old_face, _ = look_of(ctx, anchor_png, pm_full, run_dir / f"facing_old{tag}")
    old_end = None if old_face else view_end(ctx, anchor_png, pm_full, run_dir / f"view{tag}", vs.approach(s.pm))
    x0_, y0_, x1_, y1_ = vs._box(pm_full)
    aspect = (x1_ - x0_) / max(y1_ - y0_, 1.0)
    view = vs.view_words(None if old_end is None else old_end == "front", aspect) if not old_face else ""
    if view and vs.HIGH.search(" ".join(str(p.get(x) or "") for x in ("scene", "prompt"))):
        view += vs.HIGH_WORDS
    if old_face:
        view = (view or "a side view") + f", its nose pointing to the {old_face} of the picture"
    note = qa.setdefault("keyframes", {}).setdefault(tag or "first", {})
    note |= {"frame": k, "view": view or None, "facing": old_face or old_end}
    if not tag:
        qa["view"] = view or None
        qa["facing"] = old_face or old_end
    who = (p.get("describe") or "").strip().rstrip(".")
    zoom = vs.zoom_box(pm_full, (W, H))
    qa["paint_zoom"] = list(zoom) if zoom else None
    seen_k = pm_full[zoom[1]:zoom[3], zoom[0]:zoom[2]] if zoom else pm_full
    refs: list = []
    if picture:  # the user's own picture of the car, or the car as first painted: the painter copies it into the old car's place
        if reference is not None:
            character, what = _png(reference, run_dir / f"ref_car{tag}.png"), "the car as first painted"
        else:
            character, p, what, _views, _still = recast_ltx.character_picture(ctx, p, run_dir, vehicle=True, view=view)
        refs = [ctx.client.assets.from_file(character)]
        instruction = (p.get("look_prompt") or "").strip() or recast_ltx.paint_vehicle(s.noun, recast_ltx.where_words(seen_k), who)
        if view and not (p.get("look_prompt") or "").strip():
            instruction = instruction.replace(" Keep everything else in <image1>", f" Here it is seen in {view}: paint it from that "
                                              "angle, not from the angle it has in <image2>. Keep everything else in <image1>")
    else:  # described in words only: the painter takes its heading and angle from the old car (a studio still's 3/4 view was copied
        # onto a side-on SUV, facing the wrong way)
        what = "your description of the car"
        noun = recast_ltx.vehicle_words(s.noun)
        instruction = (p.get("look_prompt") or "").strip() or (
            f"Replace the {noun} in the picture with {who}: the same position, size and angle in the picture, facing and "
            f"travelling the same way{f', seen in {view}, exactly as the ' + noun + ' is' if view else ''}, its wheels on the road exactly where "
            f"the {noun}'s wheels are. Lit by the same light as the "
            "scene, with matching reflections and its own soft shadow on the road. No badges, logos or text. Keep everything else "
            "in the picture exactly as it is.")
    if not tag:
        qa["reference"] = what
    safe = ctx.commercial_safe()
    gate, tries = [], []
    for attempt in range(PAINT_TRIES):
        ctx.check()
        keep = " Keep everything else in <image1>" if picture else " Keep everything else in the picture"
        said = instruction if attempt == 0 else instruction.replace(keep, vs.SAME_SIZE.format(noun=recast_ltx.vehicle_words(s.noun)) + keep)
        if attempt and old_end:
            said = said.replace(keep, (" It is seen from behind: we see its tail lights and rear window, not its front." if old_end == "rear" else
                                       " It is seen from the front: we see its headlights and windscreen, not its rear.") + keep)
        painted = vs.paint(ctx, anchor, said, refs, seed + attempt, safe, run_dir, zoom)
        png = _png(painted, run_dir / f"painted_{tag}{attempt}.png")
        found, label = vs.find_vehicle(ctx, png, pm_full, run_dir / f"newcut{tag}{attempt or ''}")
        chk = fit_check(found, pm_full, ratio) if found is not None else {"ok": False, "miss": 9.0, "why": "new car not found"}
        new_face = new_end = None
        if found is not None and old_face:
            new_face, _ = look_of(ctx, png, found, run_dir / f"facing_new{tag}{attempt}")
        elif found is not None and old_end:
            new_end = view_end(ctx, png, found, run_dir / f"view_new{tag}{attempt}")
        if old_face and new_face and new_face != old_face:
            chk = {**chk, "ok": False, "miss": round(chk["miss"] + 1.0, 4), "why": f"faces {new_face}, the old car faces {old_face}"}
        if old_end and new_end and new_end != old_end:  # a car going away painted from the front
            chk = {**chk, "ok": False, "miss": round(chk["miss"] + 1.0, 4), "why": f"shows its {new_end}, the old car shows its {old_end}"}
        gate.append({"seed": seed + attempt, "found": label or None, "facing": new_face or new_end, **chk})
        if found is not None:
            tries.append((chk["miss"], attempt, painted, found, new_face))
        if chk["ok"]:
            break
    note["gate"] = gate
    if not tag:
        qa["paint_gate"] = gate
    if not tries:
        raise RuntimeError("The new car couldn't be found on the painted frame. Try again, or try another picture of it.")
    miss, _, painted, car, new_face = min(tries, key=lambda t: (t[0], t[1]))
    note["ok"] = miss == 0
    if not tag:
        qa["paint_ok"] = miss == 0
    if exact_size and tag and ratio and miss == 0:
        wr = fit_check(car, pm_full, ratio).get("width_ratio") or ratio
        if abs(wr - ratio) > KEY_EXACT * ratio:
            painted, car = refit(painted, car, pm_full, ground, False, ratio)
            note["exact_refit"] = {"was": wr, **fit_check(car, pm_full, ratio)}
    if miss > 0:  # none of them fits: the best one, scaled and moved onto the old car's box, set on the clean road
        flip = bool(old_face and new_face and new_face != old_face)  # a side view facing the wrong way is mirrored
        painted, car = refit(painted, car, pm_full, ground, flip, ratio or 1.0)
        note["refit"] = {**fit_check(car, pm_full, ratio), "flipped": flip}
        if not tag:
            qa["paint_refit"] = note["refit"]
    return painted, car


# --- the regions of the old car --------------------------------------------------------------------------------------------


def reflection_mirror(car: np.ndarray, reach: float = 1.0) -> np.ndarray:
    """Where a car mirrors itself on a wet road (the mask's size): below its contact line, in every column it spans, as far
    down as the car is tall in that column (times `reach`): its silhouette flipped about the contact line, so the region has the
    car's roofline, not a box's edge."""
    out = np.zeros_like(car, dtype=bool)
    if not car.any():
        return out
    xs, line = vs.contact_line(car)
    top = np.argmax(car[:, xs], axis=0).astype(np.float32)
    tall = np.maximum(line - top, 0.0)
    h = car.shape[0]
    yy = np.arange(h, dtype=np.float32)[:, None]
    out[:, xs] = (yy >= (line - 0.03 * tall)[None, :]) & (yy <= (line + reach * tall)[None, :])
    return out & ~car


def old_regions(ctx, s: vs.Shot, cdir: Path, qa: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], bool]:
    """(old car, shadow-and-spray, reflection) masks at the proxy's size, and whether the road is wet. The shadow and the
    reflection are separate regions: the shadow from SAM 3.1, the reflection (wet road only) straight down from the contact line."""
    wproxy = cdir / "proxy.mp4"
    common.proxy_clip(ctx, s.full, wproxy)
    pw, ph = media.probe(wproxy).width, media.probe(wproxy).height
    pm = [m if m.shape == (ph, pw) else recast.upsize(m, (pw, ph)) for m in s.pm]
    wet = vs.wet_road(s.p)
    sh: list[np.ndarray] = [np.zeros_like(m) for m in pm]
    if s.p.get("shadow", True):
        try:
            with vs.span(ctx, 0.0, 0.5):
                found = common.shadow_masks(ctx, wproxy, pm, cdir, labels=vs.SHADOW_LABELS)
            sh = [x & vs.ground_rows(m) for x, m in zip(found, pm)]
        except Exception:  # noqa: BLE001 - a missing shadow is better than a failed take
            pass
    if wet:
        try:
            with vs.span(ctx, 0.5, 0.9):
                spray = common.shadow_masks(ctx, wproxy, pm, cdir / "spray", labels=vs.SPRAY_LABELS)
            sh = [a | b for a, b in zip(sh, spray)]
            qa["spray_removed"] = True
        except Exception:  # noqa: BLE001
            pass
    lines = [vs.contact_line(m) if m.any() else None for m in pm]
    sh = [x & cb.road_zone(m, ln) if ln is not None else np.zeros_like(m) for x, m, ln in zip(sh, pm, lines)]  # the zone round the car
    refl = [reflection_mirror(m) if wet else np.zeros_like(m) for m in pm]
    qa["wet_road"] = wet
    qa["tracked"] = {"car_px": int(np.mean([m.sum() for m in pm])), "shadow_px": int(np.mean([m.sum() for m in sh])),
                     "reflection_px": int(np.mean([m.sum() for m in refl]))}
    return pm, sh, refl, wet


# --- EditAnything ------------------------------------------------------------------------------------------------------------


def _guide(full: Path, out: Path, crop: tuple[int, int, int, int], size: tuple[int, int], seg: tuple[int, int, int], total: int) -> Path:
    """The guide clip for one run: the crop of the work clip scaled to the model's size, frames start..start+/-count, reversed for
    a backward run, padded with copies of its last frame to `total` (8m+1)."""
    x0, y0, cw, ch = crop
    start, step, count = seg
    lo, hi = (start, start + count - 1) if step > 0 else (start - count + 1, start)
    vf = f"trim=start_frame={lo}:end_frame={hi + 1},setpts=PTS-STARTPTS,crop={cw}:{ch}:{x0}:{y0},scale={size[0]}:{size[1]}:flags=lanczos"
    if step < 0:
        vf += ",reverse"
    if total > count:
        vf += f",tpad=stop_mode=clone:stop={total - count}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-vf", vf, "-frames:v", str(total), "-c:v", "libx264", "-crf", "2",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    return out


def car_region(s: vs.Shot, i: int, crop: tuple) -> np.ndarray:
    """Where the car is on frame i, at the render's size (w, h): the old car's outline, grown (the new car sits where it was)."""
    x0, y0, cw, ch, w, h = crop
    W, H = s.size
    m = recast.upsize(s.pm[i], (W, H))[y0:y0 + ch, x0:x0 + cw]
    small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    return composite.dilate_box(small, 12)


def assemble_windows(s: vs.Shot, crop: tuple, windows: list[list[Path]], starts: list[int], gdir: Path) -> list[Path]:
    """One frame sequence (gdir/seq) from the keyframe windows' renders: each window takes over where it matches the one before best."""
    n = s.frames
    seq = gdir / "seq"
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir()

    def load(j: int, i: int) -> np.ndarray:
        return np.asarray(Image.open(windows[j][i - starts[j]]).convert("RGB"), np.float32)

    # Where each window takes over from the one before: not at its keyframe but at the frame within the next few where the two renders
    # of the car look most alike once the new one is moved onto the old one (a cross-fade of two renders doubled the wheels and the
    # car's edges); then a cut, the move fading out over 3 frames.
    cuts = [0]
    shifts: dict[int, tuple[float, float]] = {}
    for j in range(1, len(starts)):
        best = None
        for c in range(starts[j], min(starts[j] + cb.ANCHOR_FADE + 3, n, starts[j - 1] + cb.ANCHOR_WINDOW)):
            region = car_region(s, c, crop)
            old_f, new_f = load(j - 1, c), load(j, c)
            dx, dy = cb.align_shift(old_f, new_f, region)
            moved = cb.shifted(new_f, dx, dy)
            cost = float(np.abs(composite.luma709(moved) - composite.luma709(old_f))[region].mean()) if region.any() else 0.0
            if best is None or cost < best[0]:
                best = (cost, c, dx, dy)
        assert best is not None
        cuts.append(best[1])
        shifts[best[1]] = (best[2], best[3])
    out = []
    for i in range(n):
        j = max(x for x in range(len(starts)) if cuts[x] <= i)
        dst = seq / f"{i + 1:04d}.png"
        age = i - cuts[j]
        if j > 0 and age < 3 and shifts.get(cuts[j]) is not None:
            dx, dy = shifts[cuts[j]]
            f = 1.0 - age / 3.0
            Image.fromarray(np.clip(np.rint(cb.shifted(load(j, i), f * dx, f * dy)), 0, 255).astype(np.uint8)).save(dst)
        else:
            shutil.copyfile(windows[j][i - starts[j]], dst)
        out.append(dst)
    (gdir / "cuts.json").write_text(json.dumps(cuts))
    return out


def generate_anchored(ctx, s: vs.Shot, paints: list[tuple[int, np.ndarray]], crop: tuple[int, int, int, int, int, int], prompt: str,
                      seed: int, gdir: Path, dev: bool = False) -> list[Path]:
    """EditAnything with the new car re-painted every ~24 frames (one window per keyframe, each painted frame its reference), the
    windows faded into each other over 8 frames. One reference decays: the render drifts back to the old car's size and look
    within ~25 frames of it (C4's car grew back, C5's went back towards the blue SUV). Frames in gdir/seq, in time order."""
    x0, y0, cw, ch, w, h = crop
    n = s.frames
    gdir.mkdir(parents=True, exist_ok=True)
    starts = [a for a, _ in paints]
    windows: list[list[Path]] = []
    for j, (a, painted) in enumerate(paints):
        ctx.check()
        count = min(cb.ANCHOR_WINDOW, n - a)
        guide = _guide(s.full, gdir / f"guide{j}.mp4", (x0, y0, cw, ch), (w, h), (a, 1, count), cb.ANCHOR_WINDOW)
        ref = cb.resize(painted[y0:y0 + ch, x0:x0 + cw], (w, h))
        ref_asset = ctx.client.assets.from_file(_png(ref, gdir / f"ref{j}.png"))
        guide_asset = ctx.client.assets.from_file(guide)
        with vs.span(ctx, j / len(paints), (j + 1) / len(paints)):
            try:
                job = ctx.run(graphs.edit_anything_motion(guide_asset, ref_asset, prompt, s.fps, seed=seed, negative="" if dev else GENERIC_NEGATIVE,
                                                          tile=max(80, cb.ANCHOR_WINDOW), dev=dev))
            except Canceled:
                raise
            except Exception:  # noqa: BLE001 - the dev model isn't on every machine (or doesn't fit): the distilled one, the next seed
                job = ctx.run(graphs.edit_anything_motion(guide_asset, ref_asset, prompt, s.fps, seed=seed + 1, negative=GENERIC_NEGATIVE,
                                                          tile=max(80, cb.ANCHOR_WINDOW), dev=False))
        sd = gdir / f"win{j}"
        sd.mkdir(exist_ok=True)
        files = []
        for i, o in enumerate(job.get_outputs("PU_Out")):
            f = sd / f"{i + 1:04d}.png"
            o.to_file(f)
            files.append(f)
        if len(files) < count:
            raise RuntimeError("LTX-2.5 didn't send the shot back. Try again.")
        windows.append(files)
    return assemble_windows(s, crop, windows, starts, gdir)


def alpha_frames(ctx, gen_mp4: Path, fps: float, size: tuple[int, int], seed: int, adir: Path) -> list[np.ndarray]:
    """Alpha-Gen's alpha of the generated clip: one float map (0..1) per frame at the clip's size."""
    adir.mkdir(parents=True, exist_ok=True)
    job = ctx.run(graphs.alpha_gen_video(ctx.client.assets.from_file(gen_mp4), fps, size[0], size[1], seed=seed))
    out = []
    for i, o in enumerate(job.get_outputs("PU_Out")):
        f = adir / f"{i + 1:04d}.png"
        o.to_file(f)
        out.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0)
    return out


# --- the composite, the gate ---------------------------------------------------------------------------------------------------


class Plan:
    """Everything the blend and the gate need, fixed before the render: sizes, crop, the old car's regions (proxy size)."""

    def __init__(self, s: vs.Shot, pm, sh, refl, wet, crop, plate_dir: Path, painted, car_k, k: int, paints=None):
        self.s, self.pm, self.sh, self.refl, self.wet = s, pm, sh, refl, wet
        self.crop, self.plate_dir, self.painted, self.car_k, self.k = crop, plate_dir, painted, car_k, k
        # the keyframes: [(frame, painted picture, the new car's mask)], and the colour each one's car has (the drift gate's reference)
        self.paints = paints or [(k, painted, car_k)]
        self.wheels: list[list[np.ndarray]] | None = None   # tracked wheel masks per wheel per frame, at the render's size
        self.spin = 1.0                                       # +1 clockwise (nose to the right), -1 the other way
        self.W, self.H = s.size
        self.kf = max(self.W, self.H) / 1920
        self.ke = max(1.0, self.kf)
        rows = [np.flatnonzero(m.any(axis=1)) for m in pm]
        tall = [(r[-1] - r[0] + 1) * self.H / pm[0].shape[0] for r in rows if len(r)]
        self.tall = float(np.median(tall)) if tall else 100.0
        cols = [np.flatnonzero(m.any(axis=0)) for m in pm]
        wide = [(c[-1] - c[0] + 1) * self.W / pm[0].shape[1] for c in cols if len(c)]
        self.width = float(np.median(wide)) if wide else 200.0

    def ref_colour(self, i: int) -> np.ndarray | None:
        """The painted car's mean colour at the keyframe in force on frame i."""
        x0, y0, cw, ch = self.crop[:4]
        a, img, car = max((t for t in self.paints if t[0] <= i), key=lambda t: t[0], default=self.paints[0])
        m = composite.erode_box(car[y0:y0 + ch, x0:x0 + cw], 6)
        return img[y0:y0 + ch, x0:x0 + cw].astype(np.float32)[m].mean(axis=0) if m.any() else None

    def old_crop(self, i: int) -> np.ndarray:
        """The old car's footprint on frame i, in the crop's pixels."""
        x0, y0, cw, ch = self.crop[:4]
        m = self.pm[i] | self.sh[i] | self.refl[i]
        return recast.upsize(m, (self.W, self.H))[y0:y0 + ch, x0:x0 + cw]

    def soft_crop(self, i: int) -> np.ndarray:
        """How much of the old car's shadow, spray and reflection to take out on frame i (0..1), in the crop's pixels."""
        x0, y0, cw, ch = self.crop[:4]
        car = recast.upsize(self.pm[i], (self.W, self.H))
        if not car.any():
            return np.zeros((ch, cw), np.float32)
        shadow = recast.upsize(self.sh[i], (self.W, self.H))
        w = cb.soft_weights(car, shadow, vs.contact_line(car), self.wet, self.ke)
        return w[y0:y0 + ch, x0:x0 + cw]

    def car_crop(self, i: int) -> np.ndarray:
        x0, y0, cw, ch = self.crop[:4]
        return recast.upsize(self.pm[i], (self.W, self.H))[y0:y0 + ch, x0:x0 + cw]

    def plate_crop(self, i: int) -> np.ndarray:
        """Clean-plate frame i for the crop: the plate's own pixels for the crop's box, scaled to the crop's size."""
        x0, y0, cw, ch = self.crop[:4]
        f = sorted(self.plate_dir.glob("*.png"))[min(i, len(list(self.plate_dir.glob("*.png"))) - 1)]
        im = Image.open(f).convert("RGB")
        sx, sy = im.width / self.W, im.height / self.H
        return np.asarray(im.resize((cw, ch), Image.LANCZOS, box=(x0 * sx, y0 * sy, (x0 + cw) * sx, (y0 + ch) * sy)))


def save_state(plan: Plan, run_dir: Path) -> None:
    """What the blend needs, kept in the work folder: lets a take's composite be replayed without the GPU."""
    s = plan.s
    cars = {f"car{a}": c for a, _, c in plan.paints}
    np.savez_compressed(run_dir / "state.npz", pm=np.array(plan.pm), sh=np.array(plan.sh), refl=np.array(plan.refl), car=plan.car_k, **cars)
    meta = {"crop": list(plan.crop), "k": plan.k, "wet": plan.wet, "plate_dir": str(plan.plate_dir), "size": list(s.size), "fps": s.fps,
            "frames": s.frames, "noun": s.noun, "full": str(s.full), "anchors": [a for a, _, _ in plan.paints], "spin": plan.spin}
    (run_dir / "state.json").write_text(json.dumps(meta))
    Image.fromarray(plan.painted).save(run_dir / "painted_full.png")
    for a, img, _ in plan.paints:
        Image.fromarray(img).save(run_dir / f"painted_a{a}.png")


def load_state(run_dir: Path, shot_like) -> Plan:
    """The Plan saved by save_state, for a replay (`shot_like`: a vs.Shot, or anything with the same fields)."""
    meta = json.loads((run_dir / "state.json").read_text())
    z = np.load(run_dir / "state.npz")
    painted = np.asarray(Image.open(run_dir / "painted_full.png").convert("RGB"))
    paints = []
    for a in meta.get("anchors", [meta["k"]]):
        f = run_dir / f"painted_a{a}.png"
        paints.append((a, np.asarray(Image.open(f).convert("RGB")) if f.exists() else painted, z[f"car{a}"] if f"car{a}" in z else z["car"]))
    plan = Plan(shot_like, list(z["pm"]), list(z["sh"]), list(z["refl"]), meta["wet"], tuple(meta["crop"]), Path(meta["plate_dir"]),
                painted, z["car"], meta["k"], paints)
    plan.spin = float(meta.get("spin", 1.0))
    return plan


def matte_crops(ctx, plan: Plan, gen_files: list[Path], gdir: Path, fps: float, seed: int, qa: dict) -> list[np.ndarray]:
    """The new car's matte on every frame, at the crop's size: Alpha-Gen's and a SAM 3.1 track's, the steadier one."""
    x0, y0, cw, ch, w, h = plan.crop
    n = len(gen_files)
    gen_mp4 = _frames_to_mp4(gdir / "seq", fps, gdir / "gen.mp4")
    alpha = alpha_frames(ctx, gen_mp4, fps, (w, h), seed, gdir / "alpha")
    a_m = [cb.matte_from(cb.resize_float(a, (cw, ch)), plan.ke) for a in alpha[:n]]
    sam_m: list[np.ndarray] = []
    try:
        seed_mask = cb.resize_float((plan.car_k[y0:y0 + ch, x0:x0 + cw]).astype(np.float32), (w, h)) > 0.5
        track = common.track_from(ctx, gen_mp4, plan.k, [seed_mask], gdir, folder="new_track", prefix="pickups/recast_track")
        sam_m = [cb.matte_from(cb.resize_float(t.astype(np.float32), (cw, ch)), plan.ke) for t in track[:n]]
    except Exception as e:  # noqa: BLE001 - Alpha-Gen alone
        qa["sam_track_error"] = str(e)[:120]
    if len(a_m) < n:
        a_m += [np.zeros((ch, cw), np.float32)] * (n - len(a_m))
    if sam_m and len(sam_m) == n:
        which, score = cb.steadier(a_m, sam_m)
    else:
        which, score = "alpha", {"alpha_gen": round(cb.jitter(a_m), 4), "sam": None}
    qa["matte"] = {"source": which, "jitter": score}
    chosen = sam_m if which == "sam" else a_m
    plan.wheels = None
    md = gdir / "mattes"
    md.mkdir(exist_ok=True)
    for i, m in enumerate(chosen):
        Image.fromarray(np.rint(m * 255).astype(np.uint8)).save(md / f"{i + 1:04d}.png")
    try:
        track_wheels(ctx, plan, gen_files, gen_mp4, gdir, chosen, qa)
    except Canceled:
        raise
    except Exception as e:  # noqa: BLE001 - no spin, as before
        qa["wheel_track_error"] = str(e)[:120]
    return chosen


def track_wheels(ctx, plan: Plan, gen_files: list[Path], gen_mp4: Path, gdir: Path, mattes: list[np.ndarray], qa: dict) -> None:
    """The new car's wheels, followed through the render (SAM 3.1 finds them on the first frame, inside the car's lower half, and
    tracks each), so the blend can turn them: a generated wheel keeps its spoke pattern from frame to frame."""
    x0, y0, cw, ch, w, h = plan.crop
    n = len(gen_files)
    car0 = cb.resize_float(mattes[0].astype(np.float32), (w, h)) > 0.5
    rows = np.flatnonzero(car0.any(axis=1))
    if len(rows) == 0:
        return
    found = common.select_instances(ctx, gen_files[0], "wheel", 1.0, gdir / "wheel_seed", min_frac=0.0002)
    low = rows[0] + 0.45 * (rows[-1] - rows[0])
    keep = [m for m in found if m.shape == (h, w) and (m & car0).sum() >= 0.6 * m.sum() and np.nonzero(m)[0].mean() > low]
    keep = sorted(keep, key=lambda m: -int(m.sum()))[:4]
    qa["wheels_found"] = len(keep)
    if not keep:
        return
    tracks = common.track_each(ctx, gen_mp4, 0, keep, gdir, n=n, folder="wheel_track", prefix="pickups/recast_track")
    np.savez_compressed(gdir / "wheels.npz", tracks=np.array(tracks))
    plan.wheels = [list(t) for t in tracks]


def load_wheels(plan: Plan, gdir: Path) -> None:
    f = gdir / "wheels.npz"
    if f.exists():
        plan.wheels = [list(t) for t in np.load(f)["tracks"]]


def blend_all(ctx, plan: Plan, gen_files: list[Path], mattes: list[np.ndarray], master: Path, receipt: Path, qa: dict) -> dict:
    """Every frame composited and written (ProRes master + the tinted receipt), with the gate's per-frame measures."""
    s = plan.s
    W, H = plan.W, plan.H
    x0, y0, cw, ch, w, h = plan.crop
    n = s.frames
    picks = sorted({0, n // 2, n - 1})
    reals = {}
    for i, f in enumerate(common.read_frames(s.full)):
        if i in picks:
            reals[i] = f[y0:y0 + ch, x0:x0 + cw].copy()
        if i >= picks[-1]:
            break
    gens = {i: cb.resize(np.asarray(Image.open(gen_files[i]).convert("RGB")), (cw, ch)) for i in picks}
    ring_for = lambda i: composite.dilate_box(plan.old_crop(i) | (mattes[i] > 0.5), round(120 * plan.ke)) & \
        ~composite.dilate_box(plan.old_crop(i) | (mattes[i] > 0.5), round(30 * plan.ke))  # noqa: E731
    plate_need = cb.grain_need([reals[i][ring_for(i)].reshape(-1, 1, 3) for i in picks],
                               [cb.resize(plan.plate_crop(i), (cw, ch))[ring_for(i)].reshape(-1, 1, 3) for i in picks])
    car_need = cb.grain_need([reals[i][ring_for(i)].reshape(-1, 1, 3) for i in picks], [gens[i][ring_for(i)].reshape(-1, 1, 3) for i in picks])
    qa["grain_added"] = {"plate": None if plate_need is None else [round(float(x), 2) for x in plate_need],
                         "car": None if car_need is None else [round(float(x), 2) for x in car_need]}
    ellipses = [cb.smooth_ellipses([recast.upsize(m, (cw, ch)) for m in track]) for track in (plan.wheels or [])]
    out_w = common.FrameWriter(master, W, H, s.fps)
    rec_w = common.FrameWriter(receipt, W, H, s.fps, kind="h264", scale_long=1920)
    kept: dict[int, np.ndarray] = {}
    sample = set(np.linspace(0, n - 1, 6).astype(int).tolist())

    def blend(i: int, frame: np.ndarray):
        real = frame[y0:y0 + ch, x0:x0 + cw]
        gen = cb.resize(np.asarray(Image.open(gen_files[i]).convert("RGB")), (cw, ch))
        near = [cb.resize(np.asarray(Image.open(gen_files[j]).convert("RGB")), (cw, ch)) for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < n]
        specks = 0
        if s.p.get("despeckle"):  # off by default: a bright, isolated, short-lived dot on the paint (see cb.despeckle)
            gen, specks = cb.despeckle(gen, near, mattes[i], plan.ke)
        if plan.wheels and not s.p.get("no_spin"):  # the wheels turn: a generated wheel keeps one spoke pattern
            for track, ells in zip(plan.wheels, ellipses):
                areas = [int(m.sum()) for m in track if m.any()]
                if i < len(ells) and ells[i] is not None and track[i].sum() >= 0.6 * float(np.median(areas)):
                    gen = cb.spin_wheel(gen, ells[i], plan.spin * SPIN_STEP * i, SPIN_BLUR, SPIN_SAMPLES)
        comp, support, st = cb.compose(real, plan.plate_crop(i), gen, mattes[i], plan.car_crop(i), plan.ke, i, plan.tall, plan.wet,
                                       plate_need, car_need, soft=plan.soft_crop(i))
        out = frame.copy()
        out[y0:y0 + ch, x0:x0 + cw] = comp
        diff = np.abs(comp.astype(np.int16) - real.astype(np.int16)).max(axis=-1)
        full_live = np.zeros((H, W), bool)
        full_live[y0:y0 + ch, x0:x0 + cw] = support
        changed_outside = int(np.any(out != frame, axis=-1)[~full_live].sum())
        bodies = composite.dilate_box(plan.car_crop(i) | (mattes[i] > 0.5), 6)  # a car's own straight edges (roof, bonnet) don't count
        seam = cb.straight_seam((diff > 6) & ~bodies, max(48, int(0.4 * plan.width)))
        hl = cb.halo(comp, plan.plate_crop(i), mattes[i], plan.ke)
        left = cb.old_left_over(comp, real, plan.car_crop(i), mattes[i] > 0.5)
        tinted = out.copy()
        ai = np.zeros((H, W), bool)
        ai[y0:y0 + ch, x0:x0 + cw] = (diff > 2)
        reg = tinted[ai].astype(np.float32) * 0.45 + np.array(recast.AI_TINT, np.float32) * 0.55
        tinted[ai] = reg.astype(np.uint8)
        dr = cb.drift_stats(real, gen, plan.car_crop(i), mattes[i] > 0.5, plan.ref_colour(i))
        return out, tinted, {"specks": specks, "copy": dr["copy"], "colour": dr["colour"], "area": dr["area"], "outside": changed_outside, "touched": int(ai.sum()), "seam": seam, "halo": hl, "left": left, **st}

    stats = []
    with vs.span(ctx, 0.0, 1.0):
        for i, (out, tinted, st) in enumerate(common.parallel_frames(common.read_frames(s.full), blend)):
            ctx.check()
            out_w.write(out)
            rec_w.write(tinted)
            stats.append(st)
            if i in sample:
                kept[i] = out[y0:y0 + ch, x0:x0 + cw].copy()
            ctx.progress((i + 1) / max(n, 1))
    out_w.close()
    rec_w.close()
    return {"stats": stats, "samples": kept}


def wheel_contact(ctx, plan: Plan, master: Path, mattes: list[np.ndarray], wdir: Path) -> dict:
    """Every visible tyre bottom of the new car against the road line the old car's tyres stood on, on six frames of the result
    (SAM 3.1 finds the wheels; the lowest point of each wheel that belongs to the new car is its tyre's bottom)."""
    s = plan.s
    n = s.frames
    x0, y0, cw, ch = plan.crop[:4]
    picks = sorted(set(np.linspace(0, n - 1, 5).astype(int).tolist()))
    wdir.mkdir(parents=True, exist_ok=True)
    frames = {i: f for i, f in enumerate(common.read_frames(master)) if i in picks}
    per = {}
    worst_float, worst_sink = 0.0, 0.0
    for i in picks:
        png = _png(frames[i], wdir / f"f{i:03d}.png")
        try:
            found = common.select_instances(ctx, png, "wheel", 1.0, wdir / f"w{i:03d}", min_frac=0.0005)
        except Exception:  # noqa: BLE001
            continue
        new = np.zeros((plan.H, plan.W), bool)
        new[y0:y0 + ch, x0:x0 + cw] = composite.dilate_box(mattes[i] > 0.5, 2)
        old = recast.upsize(plan.pm[i], (plan.W, plan.H))
        if not old.any():
            continue
        lx, ly = vs.contact_line(old)
        rows = np.flatnonzero(old.any(axis=1))
        tall = float(rows[-1] - rows[0] + 1)
        bottoms = []
        ny = np.flatnonzero(new.any(axis=1))
        mid = ny[0] + 0.5 * (ny[-1] - ny[0]) if len(ny) else 0
        for m in found:
            if m.shape != new.shape or (m & new).sum() < 0.5 * m.sum():
                continue
            ys, xs = np.nonzero(m)
            if ys.mean() < mid:  # a wheel is in the car's lower half (an arch, a vent or a light read as one isn't)
                continue
            lowest = ys.max()
            bottoms.append((float(xs[ys >= lowest - 2].mean()), float(lowest)))
        g = cb.tyre_gaps(bottoms, lx.astype(float), ly.astype(float), tall)
        per[int(i)] = {"wheels": len(bottoms), **g}
        worst_float = max(worst_float, g["max_float"] or 0.0)
        worst_sink = max(worst_sink, g["max_sink"] or 0.0)
    seen = sum(v["wheels"] for v in per.values())
    return {"frames": per, "wheels_seen": seen, "max_float": round(worst_float, 4), "max_sink": round(worst_sink, 4),
            "ok": bool(seen == 0 or (worst_float <= GAP_MAX and worst_sink <= SINK_MAX))}


def judge(plan: Plan, mattes: list[np.ndarray], result: dict, wheels: dict | None, qa: dict) -> list[str]:
    """The quality gate: the numbers into qa["gate"], and the list of what failed (empty = pass)."""
    st = result["stats"]
    n = len(st)
    halos = [x["halo"] for x in st if x.get("halo") is not None]
    lefts = [x["left"] for x in st if x.get("left") is not None]
    moves = []
    for i in range(n - 1):
        a, b = plan.pm[i], plan.pm[min(i + 1, n - 1)]
        moves.append(np.eye(3))  # the matte's frame-to-frame IoU is read as it is: the car moves little in a frame
        if a.any() and b.any():
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = vs._box(a), vs._box(b)
            sx, sy = (bx1 - bx0) / max(ax1 - ax0, 1.0), (by1 - by0) / max(ay1 - ay0, 1.0)
            sx = sy = float(np.sqrt(sx * sy))
            cax, cay, cbx, cby = (ax0 + ax1) / 2, ay1, (bx0 + bx1) / 2, by1
            f = plan.W / a.shape[1]
            moves[-1] = np.array([[sx, 0, (cbx - sx * cax) * f], [0, sy, (cby - sy * cay) * f], [0, 0, 1.0]])
    shape = cb.shape_stability(mattes, moves)
    ious = []
    for i in range(0, n, 4):  # the new car's outline against the old one's: the same outline means the model copied the old car
        a, b = plan.car_crop(i), mattes[i] > 0.5
        if a.any() and b.any():
            ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
    same_shape = float(np.median(ious)) if ious else None
    copies = [x["copy"] for x in st if x.get("copy") is not None]
    copied = float(np.median(copies)) if copies else None

    def peak(vals: list, width: int = 5) -> float | None:
        """The highest running mean (over `width` frames) of a per-frame measure: one noisy frame doesn't count, a stretch does."""
        v = np.array([np.nan if x is None else x for x in vals], float)
        if np.isnan(v).all():
            return None
        runs = [np.nanmean(v[i:i + width]) for i in range(max(1, len(v) - width + 1)) if not np.isnan(v[i:i + width]).all()]
        return float(max(runs)) if runs else None

    copy_peak = peak([x.get("copy") for x in st])
    colour_peak = peak([x.get("colour") for x in st])
    areas = [x.get("area") for x in st]
    jumps = [abs(areas[i] - areas[i - 4]) / max(areas[i], 1e-6) for i in range(4, len(areas)) if areas[i] is not None and areas[i - 4] is not None]
    area_jump = float(max(jumps)) if jumps else None
    gate = {
        "outside_changed_px": int(sum(x["outside"] for x in st)),
        "straight_seam_px_max": int(max(x["seam"] for x in st)),
        "halo_mean": round(float(np.mean(halos)), 4) if halos else None,
        "halo_max": round(float(np.max(halos)), 4) if halos else None,
        "edge_flicker": round(cb.flicker(mattes), 4),
        "shape_stability": {"min": shape["min"], "mean": shape["mean"]},
        "specks_replaced_px": int(sum(x.get("specks", 0) for x in st)),
        "outline_iou_with_old_car": None if same_shape is None else round(same_shape, 3),
        "detail_copy_of_old_car": None if copied is None else round(copied, 3),
        "detail_copy_peak": None if copy_peak is None else round(copy_peak, 3),
        "colour_drift_peak": None if colour_peak is None else round(colour_peak, 1),
        "size_jump_max": None if area_jump is None else round(area_jump, 3),
        "old_car_left_over_max": round(float(np.max(lefts)), 4) if lefts else None,
        "old_car_left_over_mean": round(float(np.mean(lefts)), 4) if lefts else None,
        "wheel_contact": wheels,
    }
    failed = []
    if gate["outside_changed_px"]:
        failed.append("pixels changed outside the region")
    if gate["straight_seam_px_max"]:
        failed.append("a straight seam (rectangle edge)")
    if gate["halo_mean"] is not None and gate["halo_mean"] > HALO_MAX:
        failed.append("halo at the car's edge")
    if gate["edge_flicker"] > FLICKER_MAX:
        failed.append("edge flicker")
    if shape["min"] is not None and shape["min"] < SHAPE_MIN:
        failed.append("the car's shape jumps")
    if copied is not None and copied > COPY_MAX or copy_peak is not None and copy_peak > COPY_PEAK_MAX:
        failed.append("the render went back to the old car")
    if colour_peak is not None and colour_peak > COLOUR_DRIFT_MAX:
        failed.append("the new car's colour drifted from the painted one")
    if area_jump is not None and area_jump > SIZE_JUMP_MAX:
        failed.append("the new car's size jumps")
    if gate["old_car_left_over_mean"] is not None and gate["old_car_left_over_mean"] > OLD_LEFT_MAX:
        failed.append("the old car is still there")
    if wheels is not None and not wheels["ok"]:
        failed.append("a tyre off the road")
    gate["failed"] = failed
    gate["passed"] = not failed
    qa["gate"] = gate
    return failed


def run(ctx, s: vs.Shot) -> dict:
    p, run_dir, seed = s.p, s.run_dir, s.seed
    W, H = s.size
    n = s.frames
    k = 0  # the first keyframe; later ones are painted every ~24 frames (cb.anchors)
    qa: dict = {"engine": "car-v2"}
    if not p.get("doors_ok"):  # before any GPU time goes on it
        if vs.doors_or_people(s.pm, s.om):
            raise RuntimeError(vs.DOORS_MESSAGE)

    def frame_of(i: int, name: str) -> np.ndarray:
        return np.asarray(Image.open(common._frame_png(s.full, i, run_dir / name)).convert("RGB"))

    def ground_of(plate_dir: Path, i: int, anchor: np.ndarray) -> np.ndarray:
        """Frame i with the old car taken out (the plate in its footprint, toned to the footage)."""
        footprint = composite.dilate_box(recast.upsize(pm[i] | sh[i] | refl[i], (W, H)), 8)
        plate_i = np.asarray(Image.open(sorted(plate_dir.glob("*.png"))[i]).convert("RGB").resize((W, H), Image.LANCZOS))
        g = np.clip(np.rint(composite.harmonize(plate_i, anchor, footprint, grid=48, robust=True)), 0, 255).astype(np.uint8)
        return np.where(footprint[..., None], g, anchor)

    def crop_of(car: np.ndarray) -> tuple:
        boxes = []
        for i in range(n):
            m = pm[i] | sh[i] | refl[i]
            if m.any():
                bx = vs._box(m)
                boxes.append((bx[0] * W / m.shape[1], bx[1] * H / m.shape[0], bx[2] * W / m.shape[1], bx[3] * H / m.shape[0]))
        boxes.append(vs._box(car))
        return cb.crop_for(boxes, (W, H))

    def paint_keyframes(plate_dir: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """The new car painted on frame 0, then on every ~24th frame to look exactly like the first one."""
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        ref = None
        ratio = None
        for j, a in enumerate(cb.anchors(n)):
            ctx.stage(PAINT)
            frame = frame_of(a, f"anchor_full{'' if j == 0 else f'a{a}_'}.png")
            old = recast.upsize(pm[a], (W, H))
            painted, car = paint_new_car(ctx, s, old, a, frame, ground_of(plate_dir, a, frame), qa, tag="" if j == 0 else f"a{a}_", reference=ref, exact_size=bool(p.get("key_refit")),
                                         ratio=ratio)
            out.append((a, painted, car))
            if j == 0:  # the car as first painted, cut out on its own: what the later keyframes are told to copy
                ox0, _, ox1, _ = vs._box(old)
                x0, y0, x1, y1 = vs._box(car)
                ratio = (x1 - x0) / max(ox1 - ox0, 1.0)  # its width against the old car's: kept by every later keyframe
                m = int(0.12 * max(x1 - x0, y1 - y0))
                ref = painted[max(0, int(y0) - m):int(y1) + m, max(0, int(x0) - m):int(x1) + m]
        return out

    reuse = Path(p["reuse_work"]) if p.get("reuse_work") else None
    if reuse and (reuse / "state.json").exists():  # a kept work folder's track, plate and painting (iterating on the blend)
        plan = load_state(reuse, s)
        pm, sh, refl, wet, plate_dir, painted, car, crop, paints = (plan.pm, plan.sh, plan.refl, plan.wet, plan.plate_dir, plan.painted,
                                                                    plan.car_k, plan.crop, plan.paints)
        qa["reused_work"] = str(reuse)
        anchor = frame_of(k, "anchor_full.png")
        old_k = recast.upsize(pm[k], (W, H))
        if p.get("redo_plate"):  # the track and painting kept, the clean plate made again
            cdir = run_dir / "clear"
            cdir.mkdir(exist_ok=True)
            ctx.stage(CLEAR)
            plate_dir = make_plate(ctx, s, pm, cdir, qa, [a | b for a, b in zip(sh, refl)])
        if p.get("redo_paint"):  # ...or the painting made again on the kept plate
            paints = paint_keyframes(plate_dir)
            painted, car = paints[0][1], paints[0][2]
            crop = crop_of(car)
    else:
        # 1-2. The old car's regions, and the clean road.
        ctx.stage(CLEAR)
        cdir = run_dir / "clear"
        cdir.mkdir(exist_ok=True)
        pm, sh, refl, wet = old_regions(ctx, s, cdir, qa)
        with vs.span(ctx, 0.3, 1.0):
            plate_dir = make_plate(ctx, s, pm, cdir, qa, [a | b for a, b in zip(sh, refl)])
        anchor = frame_of(k, "anchor_full.png")
        old_k = recast.upsize(pm[k], (W, H))
        # 3. The new car on keyframes, size- and position-checked each time.
        paints = paint_keyframes(plate_dir)
        painted, car = paints[0][1], paints[0][2]
        _png(painted, run_dir / "painted_full.png")
        # 4. The crop around the car's path.
        crop = crop_of(car)
    plan = Plan(s, pm, sh, refl, wet, crop, plate_dir, painted, car, k, paints)
    kf = qa.get("keyframes", {}).get("first", {})
    plan.spin = -1.0 if kf.get("facing") == "left" else 1.0
    save_state(plan, run_dir)
    qa["keyframe_frames"] = [a for a, _, _ in paints]
    qa["crop"] = list(crop[:4])
    qa["work_size"] = [crop[4], crop[5]]
    qa["crop_scale"] = round(crop[4] / crop[2], 3)
    who = (p.get("describe") or "").strip().rstrip(".")
    ys, xs = np.nonzero(old_k)
    colour = colour_name(np.median(anchor[ys, xs], axis=0)) if len(ys) else ""
    prompt = (p.get("prompt") or "").strip() or (
        f"Replace the {colour + ' ' if colour else ''}{vs_noun(s.noun)} with {who}. It drives fast, its wheels spinning and its suspension "
        "working over the road, lit by the scene's own light and reflecting the scene around it. Keep the road, the camera move and "
        "everything else exactly as in the video.")
    qa["prompt"] = prompt

    best = None
    master = ctx.vdir / "master.mov"
    for attempt in range(RENDER_TRIES):
        ctx.stage(RENDER)
        gdir = run_dir / f"render{attempt}"
        rr = Path(p["reuse_render"]) if p.get("reuse_render") and attempt == 0 else None
        if rr and (rr / "mattes").exists():  # a kept render and its mattes
            gen_files = sorted((rr / "seq").glob("*.png"))
            mattes = [np.asarray(Image.open(f).convert("L"), np.float32) / 255 for f in sorted((rr / "mattes").glob("*.png"))]
            qa["reused_render"] = str(rr)
            load_wheels(plan, rr)
        else:
            with vs.span(ctx, 0.0, 1.0):
                gen_files = generate_anchored(ctx, s, [(a, img) for a, img, _ in paints], crop, prompt, seed + attempt, gdir,
                                              dev=bool(attempt) or bool(p.get("dev_render")))
            qa.setdefault("render_models", []).append("dev, CFG 4, 30 steps" if attempt or p.get("dev_render") else "distilled, 8 steps")
            ctx.stage(CUT)
            mattes = matte_crops(ctx, plan, gen_files, gdir, s.fps, seed + attempt, qa)
        ctx.stage(BLEND)
        result = blend_all(ctx, plan, gen_files, mattes, master, ctx.vdir / "receipt.mp4", qa)
        ctx.stage(CHECK)
        wheels = None
        try:
            wheels = wheel_contact(ctx, plan, master, mattes, run_dir / f"wheels{attempt}")
        except Exception as e:  # noqa: BLE001
            qa["wheel_contact_error"] = str(e)[:120]
        failed = judge(plan, mattes, result, wheels, qa)
        qa.setdefault("attempts", []).append({"seed": seed + attempt, "failed": failed, "gate": qa["gate"]})
        score = len(failed)
        if best is None or score < best[0]:
            keep = ctx.vdir / f"master_try{attempt}.mov"
            shutil.copyfile(master, keep)
            best = (score, attempt, keep, result, dict(qa["gate"]))
        if not failed:
            break
        # a failure that is cheap to detect: the next seed, once
    assert best is not None
    shutil.copyfile(best[2], master)
    for f in ctx.vdir.glob("master_try*.mov"):
        f.unlink()
    qa["gate"] = best[4]
    qa["render_used"] = best[1]
    result = best[3]

    ctx.stage(WRITE)
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=s.full)
    make_viewer(s.full, ctx.vdir / "before.mp4")
    _png(painted, ctx.vdir / "painted_frame.png")
    st = result["stats"]
    total = W * H * max(len(st), 1)
    touched = sum(x["touched"] for x in st)
    receipt = {"original": round(100 * (1 - touched / total), 2), "borrowed": 0.0, "ai": round(100 * touched / total, 2)}
    made = media.made_from(s.src, s.s0, len(st))
    safe = ctx.commercial_safe()
    models = ["SAM 3.1", graphs.edit_model(safe), "LTX-2.5 + EditAnything v0.1 motion transfer", "LTX-2.5 Clean-Plate IC-LoRA",
              "LTX-2.5 Alpha-Gen IC-LoRA"]
    qa |= {"window_start": s.s0, "anchor_frame": k, "guide": "first frame" if k == 0 else f"frame {k}"}
    return {"label": "Replace a car", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, len(st), s.fps), "engine": "ltx", "performance_size": f"{crop[4]}×{crop[5]}",
                        "reference": qa.get("reference", ""), "subject": "vehicle",
                        "pixels_changed_outside_selection": qa["gate"]["outside_changed_px"], "models": models,
                        **({"made_from": made} if made else {}), "grain": "matched to your shot", "prompt": prompt, "_qa": qa}}


def vs_noun(noun: str) -> str:
    from pickups.tasks import recast_ltx
    return recast_ltx.vehicle_words(noun)
