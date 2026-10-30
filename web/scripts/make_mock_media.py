"""Builds the demo-mode media in web/public/mock from a Pickups library: real shots, real takes, real receipts.

Every clip and still the mock API (web/src/mock.ts) shows is cut from takes Pickups made: the originals' proxies, the
takes' videos, their receipt overlays and aligned "before" clips, the people finder's cutouts, the cast's character
sheets and FLUX 3 klein stills. It reads the library by shot, take and cast id, so it only runs on the machine that
holds that library (PICKUPS_HOME, default ~/.pickups); the files it writes are what the repo ships.

    uv run python web/scripts/make_mock_media.py          (from the repo root; needs ffmpeg)

Videos: H.264 yuv420p, muted, +faststart, keyframe every 12 frames (scrubbing), 640–960 px wide.
Stills: JPEG at quality 80.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HOME = Path(os.environ.get("PICKUPS_HOME") or Path.home() / ".pickups")
SHOTS = HOME / "library" / "shots"
PROJECTS = HOME / "library" / "projects"
OUT = Path(__file__).resolve().parent.parent / "public" / "mock"

# Library ids (shots, their takes, the projects that hold the cast).
TRAIL, TRAIL_REMOVE = "shot_18c5b670f6", "v_aba37b78a7"
CASTLE_DUSK, CASTLE_DUSK_ENV = "shot_c65143169e", "v_a1ccffb667"
CASTLE_WALK, CASTLE_WALK_RECAST, CASTLE_WALK_ENV = "shot_308ef69675", "v_4d16742ab7", "v_a48a55908d"
ALPINE, ALPINE_ENV = "shot_e8b853045c", "v_19a399b17d"
GLEN, GLEN_RECAST, GLEN_RECAST_B = "shot_7e027b87b3", "v_6307c8e35b", "v_92817ef61a"
GLEN_LONG, GLEN_LONG_RECAST = "shot_8e86a27c74", "v_cd780bdae1"
OFFROAD, OFFROAD_RECAST = "shot_18741d1eff", "v_851856c201"
CURVE, CURVE_RECAST = "shot_404d7c62cd", "v_ce6e7fee91"
MISTY, MISTY_ENV, MISTY_ENV_B = "shot_de41926925", "v_afaeb992bd", "v_7f0aa85cdc"
COASTAL, COASTAL_FINISH = "shot_6bcd847447", "v_3a685fa4e3"
HOOD, HOOD_PREVIZ, HOOD_PREVIZ_A, HOOD_PREVIZ_B = "shot_2e8c8ddfde", "v_3dcb13ca4f", "v_78d2dddb53", "v_aafa29c4f4"
PROFILE, PROFILE_PREVIZ = "shot_b649b1f67b", "v_f51b94c3ef"
LAUNCH, LAUNCH_PREVIZ = "shot_97126f4009", "v_4b203115aa"
OCEAN, OCEAN_PREVIZ = "shot_46b7d4a92d", "v_04f18fd96e"
FUN_CAST, PREVIZ_CAST, LOCH_IMAGES = "prj_f13b7ace7d", "prj_2ee6fc70df", "prj_394d3c2227"


def take(shot: str, vid: str, name: str) -> Path:
    return SHOTS / shot / "versions" / vid / name


def proxy(shot: str) -> Path:
    return SHOTS / shot / "proxy.mp4"


# (output stem, source, width, crf, poster time in s or None for no poster)
VIDEOS: list[tuple[str, Path, int, int, float | None]] = [
    # Highlands & Alps
    ("trail", proxy(TRAIL), 960, 28, 2.7),
    ("trail_remove", take(TRAIL, TRAIL_REMOVE, "video.mp4"), 960, 28, 2.7),
    ("trail_remove_receipt", take(TRAIL, TRAIL_REMOVE, "receipt.mp4"), 640, 31, None),
    ("castle_dusk", proxy(CASTLE_DUSK), 960, 28, 1.2),
    ("castle_dusk_environment", take(CASTLE_DUSK, CASTLE_DUSK_ENV, "video.mp4"), 960, 28, 1.2),
    ("castle_dusk_environment_before", take(CASTLE_DUSK, CASTLE_DUSK_ENV, "before.mp4"), 960, 28, None),
    ("castle_walk", proxy(CASTLE_WALK), 768, 28, 1.5),
    ("castle_walk_recast", take(CASTLE_WALK, CASTLE_WALK_RECAST, "video.mp4"), 768, 28, 1.5),
    ("castle_walk_recast_receipt", take(CASTLE_WALK, CASTLE_WALK_RECAST, "receipt.mp4"), 640, 31, None),
    ("castle_walk_environment", take(CASTLE_WALK, CASTLE_WALK_ENV, "video.mp4"), 768, 28, 1.5),
    ("castle_walk_environment_before", take(CASTLE_WALK, CASTLE_WALK_ENV, "before.mp4"), 768, 28, None),
    ("alpine_ride", proxy(ALPINE), 768, 28, 1.6),
    ("alpine_ride_environment", take(ALPINE, ALPINE_ENV, "video.mp4"), 768, 28, 1.6),
    ("alpine_ride_environment_before", take(ALPINE, ALPINE_ENV, "before.mp4"), 768, 28, None),
    # Fun cast
    ("glen", proxy(GLEN), 960, 28, 1.6),
    ("glen_recast", take(GLEN, GLEN_RECAST, "video.mp4"), 960, 28, 1.6),
    ("glen_recast_receipt", take(GLEN, GLEN_RECAST, "receipt.mp4"), 640, 31, None),
    ("glen_recast_b", take(GLEN, GLEN_RECAST_B, "video.mp4"), 768, 28, 1.6),
    ("glen_recast_b_receipt", take(GLEN, GLEN_RECAST_B, "receipt.mp4"), 640, 31, None),
    ("glen_long", proxy(GLEN_LONG), 640, 30, 6.0),
    ("glen_long_recast", take(GLEN_LONG, GLEN_LONG_RECAST, "video.mp4"), 768, 28, 1.5),
    ("glen_long_recast_receipt", take(GLEN_LONG, GLEN_LONG_RECAST, "receipt.mp4"), 640, 31, None),
    ("glen_long_recast_before", take(GLEN_LONG, GLEN_LONG_RECAST, "before.mp4"), 768, 28, None),
    # Coast road
    ("offroad", proxy(OFFROAD), 768, 28, 1.6),
    ("offroad_recast", take(OFFROAD, OFFROAD_RECAST, "video.mp4"), 768, 28, 1.6),
    ("offroad_recast_receipt", take(OFFROAD, OFFROAD_RECAST, "receipt.mp4"), 640, 31, None),
    ("coast_curve", proxy(CURVE), 768, 28, 1.2),
    ("coast_curve_recast", take(CURVE, CURVE_RECAST, "video.mp4"), 768, 28, 1.2),
    ("coast_curve_recast_receipt", take(CURVE, CURVE_RECAST, "receipt.mp4"), 640, 31, None),
    ("misty_road", proxy(MISTY), 768, 28, 1.6),
    ("misty_road_environment", take(MISTY, MISTY_ENV, "video.mp4"), 768, 28, 1.6),
    ("misty_road_environment_receipt", take(MISTY, MISTY_ENV, "receipt.mp4"), 640, 31, None),
    ("misty_road_environment_before", take(MISTY, MISTY_ENV, "before.mp4"), 768, 28, None),
    ("misty_road_environment_b", take(MISTY, MISTY_ENV_B, "video.mp4"), 768, 28, 1.6),
    ("coastal_drive", proxy(COASTAL), 768, 28, 1.6),
    ("coastal_drive_finish", take(COASTAL, COASTAL_FINISH, "video.mp4"), 768, 28, 1.6),
    # Stunt previz
    ("hood_push", proxy(HOOD), 768, 28, 2.0),
    ("hood_push_previz", take(HOOD, HOOD_PREVIZ, "video.mp4"), 768, 28, 2.0),
    ("hood_push_previz_a", take(HOOD, HOOD_PREVIZ_A, "video.mp4"), 768, 28, 2.0),
    ("hood_push_previz_b", take(HOOD, HOOD_PREVIZ_B, "video.mp4"), 768, 28, 2.0),
    ("driver_profile", proxy(PROFILE), 640, 30, 2.0),
    ("driver_profile_previz", take(PROFILE, PROFILE_PREVIZ, "video.mp4"), 768, 28, 2.0),
    ("launch", proxy(LAUNCH), 768, 28, 1.5),
    ("launch_previz", take(LAUNCH, LAUNCH_PREVIZ, "video.mp4"), 768, 28, 1.5),
    # Seafront pier: the previz and its final
    ("ocean_person", take(OCEAN, OCEAN_PREVIZ, "before.mp4"), 768, 28, 3.0),
    ("ocean_person_previz", take(OCEAN, OCEAN_PREVIZ, "video.mp4"), 768, 28, 3.0),
]

# Stills cut from footage: (output, source video, time s, width, crop box as fractions (l, t, r, b) or None)
FRAMES: list[tuple[str, Path, float, int, tuple[float, float, float, float] | None]] = [
    # Cast: places and looks
    ("cast_castle_loch.jpg", proxy(CASTLE_DUSK), 1.0, 960, None),
    ("cast_winter_dusk.jpg", take(CASTLE_DUSK, CASTLE_DUSK_ENV, "video.mp4"), 2.0, 960, None),
    ("cast_offroader.jpg", take(OFFROAD, OFFROAD_RECAST, "video.mp4"), 2.0, 960, None),
    ("cast_convertible.jpg", take(CURVE, CURVE_RECAST, "video.mp4"), 1.6, 960, None),
    ("cast_winter_light.jpg", proxy(MISTY), 3.0, 960, None),
    # Image results (the demo's "generated" stills): frames of the stunt renders
    ("still_driver_1.jpg", take(PROFILE, PROFILE_PREVIZ, "video.mp4"), 2.5, 1280, None),
    ("still_driver_2.jpg", take(HOOD, HOOD_PREVIZ, "video.mp4"), 4.5, 1280, None),
    ("still_driver_3.jpg", take(HOOD, HOOD_PREVIZ_B, "video.mp4"), 1.0, 1280, None),
    ("still_driver_4.jpg", take(LAUNCH, LAUNCH_PREVIZ, "video.mp4"), 3.0, 1280, None),
]

# Stills from pictures: (output, source picture, width, crop box as fractions (l, t, r, b) or None)
PICTURES: list[tuple[str, Path, int, tuple[float, float, float, float] | None]] = [
    ("look_frame.jpg", take(HOOD, HOOD_PREVIZ_A, "look.png"), 960, None),
    # The stills FLUX 3 klein made from the Recast descriptions (kept with each take)
    ("offroad_recast_reference.jpg", take(OFFROAD, OFFROAD_RECAST, "reference.png"), 640, None),
    ("coast_curve_recast_reference.jpg", take(CURVE, CURVE_RECAST, "reference.png"), 640, None),
    ("look_frame_2.jpg", take(HOOD, HOOD_PREVIZ_B, "look.png"), 960, None),
    # Stunt previz cast: the driver's three-view sheet and the set's places
    ("cast_driver_sheet.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_e18bd629fb.png", 960, None),
    ("cast_driver_front.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_e18bd629fb.png", 360, (0.0, 0.0, 1 / 3, 1.0)),
    ("cast_driver_back.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_e18bd629fb.png", 360, (1 / 3, 0.0, 2 / 3, 1.0)),
    ("cast_driver_face.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_e18bd629fb.png", 360, (2 / 3, 0.0, 1.0, 1.0)),
    ("cast_hood_view.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_b0d6e57bfa.png", 960, None),
    ("cast_the_run.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_10272f576b.png", 960, None),
    ("cast_video_village.jpg", PROJECTS / PREVIZ_CAST / "cast" / "cast_2225376ebc.png", 960, None),
    # Highlands images (FLUX 3 klein)
    ("image_loch_1.jpg", PROJECTS / LOCH_IMAGES / "images" / "img_df88d36b03.png", 960, None),
    ("image_loch_2.jpg", PROJECTS / LOCH_IMAGES / "images" / "img_3522697b39.png", 960, None),
]

# Fun cast: the seven characters (never the two that look like real people), each with its sheet and views,
# and the FLUX 3 klein stills they were made from.
FUN = {
    "astronaut": ("cast_79b6968a55", "5bd7dd", ["img_00b794aeac", "img_f856702db0"]),
    "knight": ("cast_cd92e004ed", "a330d1", ["img_f91e69cc25", "img_36d91be619"]),
    "trex": ("cast_4ed3cafd3d", "022cc1", ["img_f793e3b126", "img_db1ca0500f"]),
    "android": ("cast_6b06a18aaf", "e53960", ["img_41fa3b160b", "img_9aeca10207"]),
    "alien": ("cast_6c5367fbb1", "841daa", ["img_a9d6ad9beb", "img_19ba26a7e5"]),
    "skeleton": ("cast_cf42c2138a", "09eacd", ["img_98587511fa", "img_48662a348d"]),
    "lizard": ("cast_47e68ad1f7", "f6134a", ["img_dcc0b95218", "img_bb45062b89"]),
}

# The people finder: the library's own answers where it has them, else each person's matte read off a take on the probe
# frame: what a Remove take changed (before vs after), or the tinted region of a Recast take's receipt.
# (stem, shot, take, frame, how)
PEOPLE_FROM_TAKES = [
    ("castle_walk", CASTLE_WALK, CASTLE_WALK_RECAST, 0, "receipt"),
    ("trail", TRAIL, TRAIL_REMOVE, 82, "removed"),
]
PEOPLE_FROM_LIBRARY = [("glen", GLEN), ("glen_long", GLEN_LONG)]

X264 = ["-c:v", "libx264", "-preset", "slow", "-g", "12", "-keyint_min", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]


def ff(*args: str | Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def frame(src: Path, t: float) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f.png"
        ff("-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", out)
        return Image.open(out).convert("RGB")


def frame_at(src: Path, n: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f.png"
        ff("-i", src, "-vf", f"select=eq(n\\,{n})", "-vsync", "0", "-frames:v", "1", out)
        return Image.open(out).convert("RGB")


def save_jpeg(im: Image.Image, name: str, width: int) -> None:
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width / 2) * 2), Image.LANCZOS)
    im.convert("RGB").save(OUT / name, "JPEG", quality=80, optimize=True, progressive=True)


def crop(im: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
    if not box:
        return im
    l, t, r, b = box
    return im.crop((round(l * im.width), round(t * im.height), round(r * im.width), round(b * im.height)))


def videos() -> None:
    for stem, src, width, crf, poster in VIDEOS:
        print("·", stem)
        ff("-i", src, "-vf", f"scale={width}:-2:flags=lanczos", *X264, "-crf", str(crf), OUT / f"{stem}.mp4")
        if poster is not None:
            save_jpeg(frame(src, poster), f"{stem}.jpg", width)


def stills() -> None:
    for name, src, t, width, box in FRAMES:
        print("·", name)
        save_jpeg(crop(frame(src, t), box), name, width)
    for name, src, width, box in PICTURES:
        print("·", name)
        save_jpeg(crop(Image.open(src).convert("RGB"), box), name, width)
    cast = PROJECTS / FUN_CAST / "cast"
    images = PROJECTS / FUN_CAST / "images"
    for key, (cid, tag, imgs) in FUN.items():
        print("·", key)
        save_jpeg(Image.open(cast / f"{cid}_sheet_{tag}_small.jpg"), f"fun_{key}_sheet.jpg", 960)
        for view in ("front", "back", "face"):
            save_jpeg(Image.open(cast / f"{cid}_{view}_{tag}.png"), f"fun_{key}_{view}.jpg", 360)
        for i, img in enumerate(imgs, 1):
            save_jpeg(Image.open(images / f"{img}.png"), f"fun_{key}_{i}.jpg", 600)


def tinted(rgb: np.ndarray) -> np.ndarray:
    """The receipt's tint on a frame: magenta (AI) or cyan (borrowed) pixels."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    magenta = (r > 140) & (b > 120) & (g < np.minimum(r, b) - 45)
    cyan = (b > 140) & (g > 120) & (r < np.minimum(g, b) - 50)
    return magenta | cyan


def components(mask: np.ndarray, min_px: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Connected regions of a boolean mask (4-neighbour), largest first: (x, y, w, h, region mask)."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        if len(pts) < min_px:
            continue
        ys, xs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        region = np.zeros_like(mask)
        region[ys, xs] = True
        out.append((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), region))
    return sorted(out, key=lambda c: -int(c[4].sum()))


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    h, w = mask.shape
    pad = np.pad(mask, r)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= pad[r + dy : r + dy + h, r + dx : r + dx + w]
    return out


def cutout(frame_im: Image.Image, region: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    """As people.py's thumbnail: a 3:4 crop around the person, them in colour, the rest grey and dim."""
    x0, y0, bw, bh = box
    w, h = bw * 1.28, bh * 1.28
    if w / h > 0.75:
        h = w / 0.75
    else:
        w = h * 0.75
    cx, cy = x0 + bw / 2, y0 + bh / 2
    crop = (round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2))
    colour = frame_im.crop(crop).resize((192, 256), Image.LANCZOS)
    m = Image.fromarray((region * 255).astype(np.uint8)).crop(crop).resize((192, 256), Image.LANCZOS)
    grey = np.asarray(colour.convert("L"), dtype=np.float32) * 0.32
    dim = Image.fromarray(np.stack([grey] * 3, -1).astype(np.uint8))
    return Image.composite(colour, dim, m)


def mask_png(region: np.ndarray, name: str) -> None:
    """A white matte with alpha, 1024 px wide, as the backend writes people mattes."""
    im = Image.fromarray((region * 255).astype(np.uint8)).resize((1024, round(1024 * region.shape[0] / region.shape[1])), Image.LANCZOS)
    a = np.asarray(im)
    rgba = np.zeros((*a.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(a > 100, 255, 0)
    Image.fromarray(rgba, "RGBA").save(OUT / name, optimize=True)


def people() -> dict:
    found: dict = {}
    for stem, shot in PEOPLE_FROM_LIBRARY:
        meta = json.loads((SHOTS / shot / "people" / "orig.json").read_text())["result"]
        for p in meta["people"]:
            src = SHOTS / shot / "people" / f"orig_{meta['frame']}_{p['index']}"
            Image.open(f"{src}.jpg").convert("RGB").save(OUT / f"{stem}_person_{p['index']}.jpg", "JPEG", quality=85)
            Image.open(f"{src}_mask.png").save(OUT / f"{stem}_person_{p['index']}_mask.png", optimize=True)
            for k in ("thumb_url", "mask_url"):
                p.pop(k, None)
        found[stem] = meta
    for stem, shot, vid, n, how in PEOPLE_FROM_TAKES:
        meta = json.loads((SHOTS / shot / "meta.json").read_text())
        W, H = meta["width"], meta["height"]
        before = frame_at(take(shot, vid, "before.mp4"), n)
        if how == "receipt":
            region = dilate(tinted(np.asarray(frame_at(take(shot, vid, "receipt.mp4"), n))), 2)
        else:
            diff = np.abs(np.asarray(before, dtype=np.int16) - np.asarray(frame_at(take(shot, vid, "video.mp4"), n), dtype=np.int16)).sum(-1)
            region = dilate(dilate(diff > 60, 3) & dilate(diff > 25, 1), 2)
        k = W / region.shape[1]
        regions = components(region, min_px=120)[:6]
        biggest = max(range(len(regions)), key=lambda j: int(regions[j][4].sum()))
        order = sorted(range(len(regions)), key=lambda j: regions[j][0])  # left to right, as the finder lists them
        people_out = []
        for i, j in enumerate(order):
            x, y, w, h, reg = regions[j]
            cutout(before, reg, (x, y, w, h)).save(OUT / f"{stem}_person_{i}.jpg", "JPEG", quality=85)
            mask_png(reg, f"{stem}_person_{i}_mask.png")
            ys, xs = np.nonzero(reg)
            mid = (int(np.median(xs) * k), int(np.median(ys) * k))
            low = (int(np.median(xs) * k), int(np.percentile(ys, 80) * k))
            people_out.append({
                "index": i, "box": [round(x * k), round(y * k), round(w * k), round(h * k)],
                "area": round(float(reg.sum()) / reg.size, 4),
                "point": {"x": mid[0], "y": mid[1]},
                "points": [{"x": mid[0], "y": mid[1], "positive": True}, {"x": low[0], "y": low[1], "positive": True}],
                "seen": 1, "main": j == biggest,
            })
        main = next(i for i, p in enumerate(people_out) if p["main"])
        found[stem] = {"frame": n, "people": people_out, "main": main, "width": W, "height": H}
    return found


def share() -> None:
    """The Share fallbacks (browsers that can't record): 9:16 before over after, and a 1:1 wipe, slate burned in."""
    before, after = OUT / "glen.mp4", OUT / "glen_recast.mp4"
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tall = Image.new("RGBA", (540, 960), (13, 12, 11, 0))
        d = ImageDraw.Draw(tall)
        d.ellipse((25, 115, 35, 125), fill=(245, 48, 30))
        d.text((44, 112), "RECAST · 5% AI", font=font, fill=(238, 233, 224))
        d.text((24, 150), "BEFORE", font=small, fill=(163, 157, 147))
        d.text((24, 176 + 2 * 304 + 22), "AFTER", font=small, fill=(163, 157, 147))
        d.text((24, 890), "MADE LOCALLY WITH PICKUPS", font=small, fill=(133, 127, 119))
        tall.save(t / "tall.png")
        ff("-i", before, "-i", after, "-i", t / "tall.png", "-filter_complex",
           "color=c=0x0D0C0B:s=540x960:d=4[bg];[0]scale=540:304:force_original_aspect_ratio=increase,crop=540:304[b];"
           "[1]scale=540:304:force_original_aspect_ratio=increase,crop=540:304[a];"
           "[bg][b]overlay=0:176[x];[x][a]overlay=0:488[y];[y][2]overlay=0:0,trim=duration=4",
           *X264, "-crf", "30", OUT / "share_9x16.mp4")
        sq = Image.new("RGBA", (720, 720), (0, 0, 0, 0))
        d = ImageDraw.Draw(sq)
        d.rectangle((0, 644, 720, 720), fill=(13, 12, 11, 210))
        d.ellipse((25, 669, 35, 679), fill=(245, 48, 30))
        d.text((44, 666), "RECAST · 5% AI", font=font, fill=(238, 233, 224))
        d.text((44, 692), "MADE LOCALLY WITH PICKUPS", font=small, fill=(133, 127, 119))
        sq.save(t / "sq.png")
        p = "W*(0.5+0.32*sin(T*1.4))"
        ff("-i", before, "-i", after, "-i", t / "sq.png", "-filter_complex",
           "[0]scale=720:720:force_original_aspect_ratio=increase,crop=720:720,format=gbrp[b];"
           "[1]scale=720:720:force_original_aspect_ratio=increase,crop=720:720,format=gbrp[a];"
           f"[b][a]blend=all_expr='if(lt(abs(X-{p}),1.5),236,if(lt(X,{p}),A,B))'[w];"
           "[w][2]overlay=0:0,trim=duration=4",
           *X264, "-crf", "30", OUT / "share_1x1.mp4")


def main() -> int:
    if not SHOTS.is_dir():
        print(f"No Pickups library at {HOME} (set PICKUPS_HOME).", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    if not only or "videos" in only:
        videos()
    if not only or "stills" in only:
        stills()
    if not only or "people" in only:
        print(json.dumps(people(), indent=1))
    if not only or "share" in only:
        share()
    return 0


if __name__ == "__main__":
    sys.exit(main())
