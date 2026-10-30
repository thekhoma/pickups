"""Who's in this shot: the people SAM 3.1 finds, so Recast can pick its person without a click.

`find()` (POST /api/shots/{id}/people) looks at 0, 1/4, 1/2 and 3/4 of the way through the first 4 s, then one frame
every 2 s as far into the shot as Recast searches (`window()`), one SAM 3.1 call each ("person:8", one mask each), and
keeps the frame where SAM finds the most separate people (ties: the one nearest the middle). For each person on that
frame it returns a box, click points inside them, a cutout thumbnail and a matte. The person who is largest across the
probed frames (big, and on screen for longest) is `main`, the one Recast replaces unless the user picks another.

The form turns a person into Recast's usual params: `frame` plus `points` (all positive, no `text`). That is exactly
what a click gives Recast today (`common.tracked_masks`: SAM 3.1 selects from the points on that frame, then tracks
forward and backward), so Recast itself needs no change. Send all the points: one chest point can take only part of a
half-hidden person.

Results are cached per shot and take in shots/<id>/people/ (a JSON file plus the thumbnails and mattes, served from
/media), so reopening the form is instant.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pickups import graphs, media
from pickups.app.store import media_url
from pickups.tasks import common

VERSION = 1           # bump when the result changes shape or meaning: older caches are recomputed
MAX_PEOPLE = 8
# What SAM 3.1 looks for. ComfyUI's SAM 3 text encoder keeps only the best detection per prompt unless the prompt says
# how many it may return ("person:8", comfy/text_encoders/sam3_clip.py): a plain "person" finds one person even with
# individual masks on. Checked on the 4090: plain "person" found 1 of the 4 runners, "person:8" all four.
LABEL = f"person:{MAX_PEOPLE}"
PROBES = 4            # SAM calls over the first 4 s (probe_frames: a few more on a long shot)
MIN_FRAC = 0.002      # smaller detections are specks, not people (as common.select_instances)
THUMB = (192, 256)    # 3:4, the person fitted inside with some room around them
MATTE_RGBA = (245, 184, 61, 110)  # the same overlay colour as /select's mask

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class PeopleError(RuntimeError):
    """SAM couldn't run (ComfyUI offline, a model missing). The message is shown to the user."""


def window(frames: int) -> int:
    """How far into the shot people are looked for: as far as Recast looks for them (3 x common.MAX_FRAMES, 12 s at
    24 fps; it then renders the 4 s where its person is on screen most), so someone who walks in late is still found."""
    return max(1, min(3 * int(common.MAX_FRAMES), int(frames)))


def probe_frames(w: int) -> list[int]:
    """0, 1/4, 1/2 and 3/4 of the way through the first common.MAX_FRAMES (fewer for very short shots), then one every
    half of that after them, up to w (someone who walks in late)."""
    n = max(1, min(w, int(common.MAX_FRAMES)))
    first = 4 * ((n - 1) // 4) + 1  # as Recast's own 4k+1 window
    late = range(int(common.MAX_FRAMES), w, max(1, int(common.MAX_FRAMES) // 2))
    return sorted({min(first - 1, (k * first) // PROBES) for k in range(PROBES)} | set(late))


# --- geometry ------------------------------------------------------------------------------------------------------

def _depth(mask: np.ndarray) -> np.ndarray:
    """How far each mask pixel is from the background: erosions survived, alternating 4- and 8-neighbour steps (close
    to a Euclidean distance transform, in plain numpy). The deepest pixel is the safest place to click."""
    cur = np.pad(mask.astype(bool), 1)
    depth = np.zeros(cur.shape, np.int32)
    k = 0
    while cur.any():
        k += 1
        depth += cur
        e = cur.copy()
        e[1:] &= cur[:-1]
        e[:-1] &= cur[1:]
        e[:, 1:] &= cur[:, :-1]
        e[:, :-1] &= cur[:, 1:]
        if k % 2 == 0:
            e[1:, 1:] &= cur[:-1, :-1]
            e[:-1, :-1] &= cur[1:, 1:]
            e[1:, :-1] &= cur[:-1, 1:]
            e[:-1, 1:] &= cur[1:, :-1]
        cur = e
    return depth[1:-1, 1:-1]


def inside_point(mask: np.ndarray, top: float = 0.0, bottom: float = 1.0) -> tuple[int, int, int] | None:
    """The deepest point of the mask (x, y, depth in px), optionally only between `top` and `bottom` of its height
    (0 = the top of the person, 1 = their feet). Always a mask pixel, even for shapes whose centre is outside them (an
    arm on a hip, a C). None when the band holds no mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    s = max(1, max(y1 - y0, x1 - x0) // 256)  # work on a small copy: every kept pixel is a real mask pixel
    crop = mask[y0:y1:s, x0:x1:s].astype(bool)
    d = _depth(crop)
    h = crop.shape[0]
    r0 = min(h - 1, int(top * h))
    r1 = max(r0 + 1, int(round(bottom * h)))
    band = np.zeros_like(d)
    band[r0:r1] = d[r0:r1]
    peak = int(band.max())
    if peak <= 0:
        return None
    best = np.argwhere(band == peak)
    by, bx = best[np.argmin(((best - best.mean(axis=0)) ** 2).sum(axis=1))]
    return x0 + int(bx) * s, y0 + int(by) * s, peak * s


def click_points(mask: np.ndarray) -> list[tuple[int, int]]:
    """Where to click so SAM selects the whole person, not their shirt: the deepest point (usually the chest) first,
    then, when there is room, the deepest point of the head end and of the legs. All inside the mask."""
    main = inside_point(mask)
    if main is None:
        return []
    ys = np.nonzero(mask)[0]
    height = int(ys.max() - ys.min() + 1)
    out = [main]
    for top, bottom in ((0.0, 0.15), (0.6, 0.9)):
        p = inside_point(mask, top, bottom)
        if p is None or p[2] < max(2, 0.2 * main[2]):
            continue
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= (0.15 * height) ** 2 for q in out):
            out.append(p)
    return [(x, y) for x, y, _ in out]


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _clean(masks: list[np.ndarray]) -> list[np.ndarray]:
    """People only: specks dropped, the same person found twice kept once, biggest first."""
    if not masks:
        return []
    total = masks[0].size
    out: list[np.ndarray] = []
    for m in sorted((m for m in masks if m.sum() >= MIN_FRAC * total), key=lambda m: -int(m.sum())):
        if any((m & o).sum() > 0.7 * m.sum() for o in out):
            continue
        out.append(m)
    return out[:MAX_PEOPLE]


# --- SAM --------------------------------------------------------------------------------------------------------------

def _frame(video: Path, index: int, size: tuple[int, int] | None, out: Path) -> Path:
    vf = f"select=eq(n\\,{index})" + (f",scale={size[0]}:{size[1]}" if size else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf, "-frames:v", "1", str(out)], check=True)
    if not out.exists():
        raise FileNotFoundError(f"frame {index} of {video.name}")
    return out


def _sam_people(runner, frame_png: Path, out_dir: Path, size: tuple[int, int]) -> list[np.ndarray]:
    """One SAM 3.1 call on one frame: one mask per person (graphs.select_frame with individual masks)."""
    job = runner.run(graphs.select_frame(runner.client.assets.from_file(frame_png), [], None, text=LABEL, individual=True,
                                         prefix="pickups/people"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(out_dir / o.name)
    masks = []
    for f in sorted(out_dir.glob("*.png")):
        im = Image.open(f).convert("L")
        if im.size != size:
            im = im.resize(size, Image.NEAREST)
        masks.append(np.asarray(im) > 127)
    return _clean(masks)


def count_people(runner, png: Path, out_dir: Path) -> int | None:
    """How many people SAM 3.1 finds in one picture (the masks _clean keeps), or None when the detector can't run."""
    try:
        with Image.open(png) as im:
            size = im.size
        return len(_sam_people(runner, png, out_dir, size))
    except Exception:  # noqa: BLE001 - no SAM on this ComfyUI: the count is unknown, never a failure
        return None


# --- pictures ---------------------------------------------------------------------------------------------------------

def _crop_window(box: tuple[float, float, float, float], aspect: float, frame: tuple[int, int],
                 pad: float = 0.14) -> tuple[int, int, int, int]:
    """A window of the given aspect around a box with some room, centred on it and kept inside the frame (a person
    taller than the frame is cropped a little rather than shown between black bars)."""
    fw, fh = frame
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    if h > fh:
        h, w = fh, fh * aspect
    if w > fw:
        w, h = fw, fw / aspect
    cx = min(max((x0 + x1) / 2, w / 2), fw - w / 2)
    cy = min(max((y0 + y1) / 2, h / 2), fh - h / 2)
    return round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)


def _thumb(frame: Image.Image, mask: np.ndarray, out: Path) -> None:
    """The person cut out: full colour with a thin light outline, the rest of the frame dimmed and grey, so it's clear
    who is meant. Dark shots are lifted (up to 2x) so the person reads at chip size."""
    vw, vh = frame.size
    alpha = Image.fromarray(mask.astype(np.uint8) * 255).resize((vw, vh), Image.BILINEAR)
    sx, sy = vw / mask.shape[1], vh / mask.shape[0]
    x0, y0, x1, y1 = _box(mask)
    win = _crop_window((x0 * sx, y0 * sy, x1 * sx, y1 * sy), THUMB[0] / THUMB[1], (vw, vh))
    pic = np.asarray(frame.crop(win).resize(THUMB, Image.LANCZOS), np.float32)
    a = np.asarray(alpha.crop(win).resize(THUMB, Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    on = a > 0.5
    if on.any():
        bright = float(np.percentile(pic[on].max(axis=1), 98))
        pic *= min(2.0, max(1.0, 200 / max(bright, 1.0)))
    grey = pic.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.38
    cut = pic * a[..., None] + grey * (1 - a[..., None])
    a_img = Image.fromarray((a * 255).astype(np.uint8))
    ring = (np.asarray(a_img.filter(ImageFilter.MaxFilter(5))) > 127) & ~on
    cut[ring] = cut[ring] * 0.15 + np.array([238, 233, 224], np.float32) * 0.85
    Image.fromarray(np.clip(cut, 0, 255).astype(np.uint8)).save(out, quality=86)


def _matte(mask: np.ndarray, out: Path) -> None:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = MATTE_RGBA
    Image.fromarray(rgba).save(out)


# --- the finder -------------------------------------------------------------------------------------------------------

def _source(store, shot_id: str, version_id: str | None) -> tuple[str, Path, int, int, int]:
    """(cache key, viewer video, source width, height, frames) for the original or a take. FileNotFoundError: no such
    shot or take."""
    meta = store.shot_meta(shot_id)
    if not version_id or version_id == "orig":
        return "orig", store.shot_dir(shot_id) / "proxy.mp4", int(meta["width"]), int(meta["height"]), int(meta["frames"])
    vdir = store.version_dir(shot_id, version_id)
    viewer, master = vdir / "video.mp4", vdir / "master.mov"
    if not viewer.exists() or not master.exists():
        raise FileNotFoundError(version_id)
    return (version_id, viewer, *_dims(master))


def _dims(path: Path) -> tuple[int, int, int]:
    """Width, height and frame count from the container header (no decode, so a cached answer stays instant); a full
    count only when the header has none."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    if str(s.get("nb_frames", "")).isdigit() and int(s["nb_frames"]) > 0:
        return int(s["width"]), int(s["height"]), int(s["nb_frames"])
    p = media.probe(path)
    return p.width, p.height, p.frames


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def find(runner, store, shot_id: str, version_id: str | None = None, refresh: bool = False) -> dict:
    """Who's in the shot (or a take of it). Returns

        {frame, people: [{index, box: [x, y, w, h], area, point: {x, y}, points: [{x, y, positive}], thumb_url,
                          mask_url, seen, main}], main: index | None, probed: [frames], window, width, height, cached}

    in source pixels (what Recast's `points` use). `area` is the share of the frame the person covers, `seen` on how
    many of the probed frames they were found. People are listed left to right. `partial: true` when SAM failed after
    the first frame (what it found is returned but not cached). Raises FileNotFoundError (no such shot or take) or
    PeopleError (SAM couldn't run at all)."""
    key, viewer, src_w, src_h, n = _source(store, shot_id, version_id)
    folder = store.shot_dir(shot_id) / "people"
    cache = folder / f"{key}.json"
    st = viewer.stat()
    w = window(n)
    stamp = [VERSION, LABEL, st.st_mtime_ns, st.st_size, w, src_w, src_h]
    with _lock(f"{shot_id}:{key}"):
        if not refresh and cache.exists():
            try:
                saved = json.loads(cache.read_text())
                hit = saved.get("stamp") == stamp and all((folder / Path(p["thumb_url"].split("?")[0]).name).exists()
                                                         for p in saved["result"]["people"])
            except (OSError, ValueError, KeyError, TypeError):
                hit = False
            if hit:
                return saved["result"] | {"cached": True}
        result = _find(runner, viewer, folder, key, src_w, src_h, w)
        if result.get("partial"):  # SAM failed part way: use what it found, but look again next time
            cache.unlink(missing_ok=True)
        else:
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps({"stamp": stamp, "result": result}, indent=1))
            tmp.replace(cache)
        return result | {"cached": False}


def _find(runner, viewer: Path, folder: Path, key: str, src_w: int, src_h: int, w: int) -> dict:
    pw, ph = media.work_size(src_w, src_h, common.PROXY_LONG)  # the size SAM sees in Recast (and in /select)
    work = folder / f"_work_{key}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        probed = probe_frames(w)
        found: dict[int, list[np.ndarray]] = {}
        for f in probed:
            try:
                png = _frame(viewer, f, (pw, ph), work / f"frame_{f}.png")
                found[f] = _sam_people(runner, png, work / f"masks_{f}", (pw, ph))
            except Exception as e:  # noqa: BLE001 - SAM down: say so at once, don't try three more frames
                if not found:
                    raise PeopleError(f"Couldn't look for people in this shot: {e}") from e
                break
        probed = sorted(found)
        mid = w // 2
        frame = max(probed, key=lambda f: (len(found[f]), -abs(f - mid)))
        people = found[frame]

        # Largest and longest on screen: each person's share of the frame, averaged over the probed frames (a frame
        # where nobody overlaps their box counts as zero).
        boxes = {f: [_box(m) for m in found[f]] for f in probed}
        scores, seen = [], []
        for m in people:
            b, total, count = _box(m), 0.0, 0
            for f in probed:
                ious = [_iou(b, o) for o in boxes[f]]
                if ious and max(ious) >= 0.2:
                    total += found[f][int(np.argmax(ious))].sum() / m.size
                    count += 1
            scores.append(total / len(probed))
            seen.append(count)

        order = sorted(range(len(people)), key=lambda i: (_box(people[i])[0] + _box(people[i])[2]) / 2)  # left to right
        main = order.index(int(np.argmax(scores))) if people else None
        for old in folder.glob(f"{key}_*"):
            old.unlink(missing_ok=True)
        picture = Image.open(_frame(viewer, frame, None, work / "thumb_frame.png")).convert("RGB") if people else None
        sx, sy = src_w / pw, src_h / ph
        tag = int(time.time())
        out = []
        for idx, i in enumerate(order):
            m = people[i]
            x0, y0, x1, y1 = _box(m)
            pts = [{"x": min(src_w - 1, int((x + 0.5) * sx)), "y": min(src_h - 1, int((y + 0.5) * sy)), "positive": True}
                   for x, y in click_points(m)]
            thumb, matte = folder / f"{key}_{frame}_{idx}.jpg", folder / f"{key}_{frame}_{idx}_mask.png"
            _thumb(picture, m, thumb)
            _matte(m, matte)
            out.append({"index": idx, "box": [round(x0 * sx), round(y0 * sy), round((x1 - x0) * sx), round((y1 - y0) * sy)],
                        "area": round(float(m.sum()) / m.size, 4), "point": {"x": pts[0]["x"], "y": pts[0]["y"]},
                        "points": pts, "thumb_url": f"{media_url(thumb)}?v={tag}", "mask_url": f"{media_url(matte)}?v={tag}",
                        "seen": seen[i], "main": idx == main})
        partial = {"partial": True} if len(probed) < len(probe_frames(w)) else {}
        return {"frame": frame, "people": out, "main": main, "probed": probed, "window": w, "width": src_w, "height": src_h} | partial
    finally:
        shutil.rmtree(work, ignore_errors=True)
