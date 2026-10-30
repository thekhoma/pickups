"""Helpers shared by the tasks: work clips, click-to-select masks, and encoding results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import graphs, media
from pickups.app.comfy import run_local
from pickups.app.store import make_thumb, make_viewer

# Tasks work frame for frame at the source's own rate (see media), on at most MAX_FRAMES frames from the start: what the
# video models are tested on. 4 s at 24 fps, 3.2 s at 30, 1.6 s at 60; the take says when that is less than the shot.
DEFAULT_FRAMES = 97  # a 24 GB card, until the GPU is known
# The longest take (frames, LTX lengths 8k+1) a GPU renders without quality loss, by VRAM: the largest row whose
# minimum VRAM the card meets. Measured in TEST_LOG.md ("clip length"). PICKUPS_MAX_FRAMES, when set, decides for every card.
# 24 GB (RTX 4090, reports 25.8): measured at 97, 121 and 145 frames on Replace a person, Change the weather and Remove, no
# memory error, no step in the picture, the character holds late in the clip. 80 GB and up: measured on a 96 GB card up to
# 225 frames (peak VRAM under 60 GB). The 32 GB and 48 GB rows are estimates between the two.
FRAMES_BY_VRAM: tuple[tuple[float, int], ...] = ((0.0, DEFAULT_FRAMES), (22.0, 145), (30.0, 169), (40.0, 193), (80.0, 225))
# Change the weather is shorter on a 24 GB card: at 145 frames its render took about 3.5x longer per frame and detail rose
# late in the clip, so it stays at 121 there until that is fixed. Other tiers follow the table above.
TASK_FRAMES: dict[str, tuple[tuple[float, int], ...]] = {
    "environment": ((0.0, DEFAULT_FRAMES), (22.0, 121), (30.0, 145), (40.0, 169), (80.0, 225)),
}
MAX_FRAMES = int(os.environ.get("PICKUPS_MAX_FRAMES", DEFAULT_FRAMES))
GPU_VRAM: float | None = None


def _row(table: tuple[tuple[float, int], ...], vram_gb: float | None) -> int:
    if os.environ.get("PICKUPS_MAX_FRAMES"):
        return int(os.environ["PICKUPS_MAX_FRAMES"])
    if not vram_gb:
        return DEFAULT_FRAMES
    return max((n for gb, n in table if vram_gb >= gb), default=DEFAULT_FRAMES)


def frames_for(vram_gb: float | None, task: str | None = None) -> int:
    """The longest take this GPU renders (8k+1 frames), for a task when it has its own table; DEFAULT_FRAMES when the
    card is unknown."""
    return _row(TASK_FRAMES.get(task or "", FRAMES_BY_VRAM), vram_gb)


def use_gpu(vram_gb: float | None) -> int:
    """Set MAX_FRAMES for the connected GPU (the status bar and every task read it). An unknown card leaves it as it is."""
    global MAX_FRAMES, GPU_VRAM
    if vram_gb or os.environ.get("PICKUPS_MAX_FRAMES"):
        MAX_FRAMES = frames_for(vram_gb)
        GPU_VRAM = vram_gb
    return MAX_FRAMES


def task_frames(task: str | None = None) -> int:
    """MAX_FRAMES for a task on the connected GPU (never more than MAX_FRAMES itself)."""
    return min(MAX_FRAMES, frames_for(GPU_VRAM, task)) if task in TASK_FRAMES else MAX_FRAMES


def length_cap(params: dict | None, task: str | None = None) -> int:
    """How many frames a task may render: the user's bracket (`params.frames`, at least MIN_FRAMES), never more than the
    task's limit on this GPU."""
    try:
        want = int((params or {}).get("frames") or 0)
    except (TypeError, ValueError):
        want = 0
    top = task_frames(task)
    return top if want <= 0 else max(media.MIN_FRAMES, min(top, want))


# Render previz on a big card: 169 frames (7 s at 24 fps) in one pass at 1920x1088, checked on a 96 GB RTX PRO 6000.
BIG_GPU_GB, BIG_GPU_FRAMES = 80, 169


def previz_frames(vram_gb: float | None) -> int:
    """How many frames Render previz makes in one pass on this GPU: MAX_FRAMES, or BIG_GPU_FRAMES on a card with at
    least BIG_GPU_GB. PICKUPS_MAX_FRAMES, when set, decides for every card. (Previz keeps its own measured length.)"""
    if os.environ.get("PICKUPS_MAX_FRAMES"):
        return MAX_FRAMES
    return BIG_GPU_FRAMES if (vram_gb or 0) >= BIG_GPU_GB else DEFAULT_FRAMES
MAX_LONG = 3840      # work resolution cap (4K)
PROXY_LONG = 1024    # SAM runs here
# Recast's smallest person, as tall on the 1024 px proxy (SAM 3.1 sees it at ~1008 px): below ~48 px (under 5% of the
# frame's long side, a face of a few pixels) SAM loses them during tracking and SCAIL-2's pose comes out mush.
MIN_PERSON_PX = 48
NOBODY = "No one found in this shot. Click the person in the picture, or try another shot."
LOST = "Couldn't follow the selection through the shot. Try clicking it again on a clearer frame."


def work_clip(ctx, out: Path, long_side: int = MAX_LONG, frames: int | None = None, ltx: bool = False, start: int = 0) -> dict:
    """Cut the task input to a work clip (<= frames from `start`, <= long side, even dims). ltx=True also snaps frames
    to 8k+1."""
    p = media.probe(ctx.input_path)
    n = min(MAX_FRAMES if frames is None else frames, p.frames - start)
    if ltx:
        n = media.ltx_frames(n)
        if n > p.frames - start:
            n -= 8
    s = min(1.0, long_side / max(p.width, p.height))
    size = (int(p.width * s) // 2 * 2, int(p.height * s) // 2 * 2)
    info = media.prepare_work_clip(ctx.input_path, out, start, n, size)
    info["width"], info["height"] = size
    return info


def proxy_clip(ctx, work: Path, out: Path) -> tuple[int, int]:
    """The small copy SAM runs on. A blank shot (all black, one flat colour) stops here, before any GPU work."""
    p = media.probe(work)
    pw, ph = media.work_size(p.width, p.height, PROXY_LONG)
    media.prepare_work_clip(work, out, 0, p.frames, (pw, ph))
    check_not_blank(out)
    return pw, ph


def check_not_blank(video: Path, every: int = 8) -> None:
    """Raise if every sampled frame is one flat tone (a black slate, a lens cap). A dark night shot still has lights
    and noise, so it spans far more than a few levels."""
    spans = []
    for i, f in enumerate(read_frames(video)):  # read to the end (small: the proxy), so ffmpeg finishes cleanly
        if i % every == 0:
            luma = f[::4, ::4].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            spans.append(np.percentile(luma, (0.1, 99.9)))  # a small thing on a plain backdrop still counts
    if not spans or any(b - a >= 8 for a, b in spans):
        return
    tone = "black" if max(b for _, b in spans) < 24 else "one flat colour"
    raise RuntimeError(f"This shot is {tone} all the way through, so there's nothing in it to work on. Try another shot.")


def _frame_png(video: Path, index: int, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(out)], check=True)
    return out


def _segment(video: Path, start: int, end: int, out: Path, reverse: bool = False) -> Path:
    vf = f"select='between(n\\,{start}\\,{end})',setpts=N/24/TB" + (",reverse" if reverse else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "4",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _download_masks(job, d: Path) -> list[np.ndarray]:
    d.mkdir(parents=True, exist_ok=True)
    for o in job.get_outputs("PU_Mask"):
        o.to_file(d / o.name)
    return [np.asarray(Image.open(f).convert("L")) > 127 for f in sorted(d.glob("mask_*.png"))]


def labels(text: str) -> list[str]:
    """The things a selection's words name: "person, handbag" is two labels, each found on its own (SAM 3.1 reads
    one noun phrase per prompt). Plain words are one label."""
    parts = [t.strip() for t in (text or "").split(",")]
    return [t for t in parts if t] or [""]


def select_preview(ctx_like, frame_png: Path, points: list[dict], text: str, scale: float, out_dir: Path) -> np.ndarray:
    """One-frame selection from clicks (points in source pixels, scaled into the frame's pixels) and/or words."""
    pos = [{"x": p["x"] * scale, "y": p["y"] * scale} for p in points if p.get("positive", True)]
    neg = [{"x": p["x"] * scale, "y": p["y"] * scale} for p in points if not p.get("positive", True)]
    out = np.zeros(Image.open(frame_png).size[::-1], bool)
    names = labels(text)
    for i, label in enumerate(names):
        g = graphs.select_frame(ctx_like.client.assets.from_file(frame_png), pos if i == 0 else [], neg if i == 0 else [], text=label)
        masks = _download_masks(ctx_like.run(g), out_dir if len(names) == 1 else out_dir / f"label_{i}")
        if masks and masks[0].shape == out.shape:
            out |= masks[0]
    return out


def select_instances(ctx_like, frame_png: Path, text: str, scale: float, out_dir: Path, min_frac: float = 0.002) -> list[np.ndarray]:
    """Every object SAM 3.1 finds for the label(s) on one frame, one mask each (biggest first, specks dropped)."""
    names = labels(text)
    masks = []
    for i, label in enumerate(names):
        job = ctx_like.run(graphs.select_frame(ctx_like.client.assets.from_file(frame_png), [], None, text=label, individual=True))
        masks += [m for m in _download_masks(job, out_dir if len(names) == 1 else out_dir / f"label_{i}") if m.any()]
    if not masks:
        return []
    total = masks[0].size
    masks = [m for m in masks if m.sum() >= min_frac * total]
    return sorted(masks, key=lambda m: -int(m.sum()))[:8]


def split_objects(mask: np.ndarray, min_frac: float = 0.03, max_n: int = 8) -> list[np.ndarray]:
    """Split a selection into its separate objects (connected pieces), biggest first, dropping specks: two people of
    a couple become two masks, so each is tracked on its own."""
    h, w = mask.shape
    s = max(1, max(h, w) // 256)  # label on a small copy: fast in pure numpy
    small = mask[::s, ::s]
    labels = np.zeros(small.shape, np.int32)
    n = 0
    for y, x in zip(*np.nonzero(small)):
        if labels[y, x]:
            continue
        n += 1
        stack = [(y, x)]
        labels[y, x] = n
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1), (cy - 1, cx - 1), (cy + 1, cx + 1), (cy - 1, cx + 1), (cy + 1, cx - 1)):
                if 0 <= ny < small.shape[0] and 0 <= nx < small.shape[1] and small[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    stack.append((ny, nx))
    if n <= 1:
        return [mask]
    sizes = np.bincount(labels.ravel())[1:]
    keep = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= min_frac * sizes.sum()][:max_n]
    full = np.repeat(np.repeat(labels, s, axis=0), s, axis=1)[:h, :w]
    parts = [mask & (full == k) for k in keep]
    return parts if len(parts) > 1 else [mask]


PERSON = re.compile(r"\b(person|people|man|men|woman|women|girl|boy|child|kid|hiker|walker|runner|tourist|couple|"
                    r"photographer|worker|cyclist|rider|driver|crew)\b", re.IGNORECASE)


# Labels that mean more than one person: then the frame with the most separate matches is the best start.
MANY = re.compile(r"\b(people|persons|couple|men|women|crew|kids|children|group|family|both|everyone|pair|runners|"
                  r"walkers|hikers|tourists)\b", re.IGNORECASE)
# What people carry that a "person" mask leaves out. A phone in the hand too: left out of the castle-hall woman's mask,
# it stayed as footage and floated by the new character's hands.
CARRIED = ("backpack", "bag", "suitcase", "umbrella", "phone")


def with_carried(ctx, frame_png: Path, person: np.ndarray, text: str, scale: float, run: Path) -> np.ndarray:
    """People carry things: add any bag, backpack or phone touching the selected person, so it goes with them (SAM's
    "woman" mask stops at the strap). Only when the selection is a person (by its label, or clicks with no label)."""
    if text and not PERSON.search(text):
        return person
    if not text:  # clicks only: is what they clicked a person?
        try:
            people = select_preview(ctx, frame_png, [], "person", scale, run / "pick_person")
        except Exception:  # noqa: BLE001
            return person
        if people.shape != person.shape or (people & person).sum() < 0.5 * person.sum():
            return person
    from pickups import composite

    ys = np.nonzero(person)[0]
    near = composite.grow(person, max(6, int(0.12 * (ys.max() - ys.min() + 1))))
    out = person.copy()
    for thing in CARRIED:
        try:
            m = select_preview(ctx, frame_png, [], thing, scale, run / f"pick_{thing}")
        except Exception:  # noqa: BLE001 - an accessory we can't find is not an error
            continue
        if m.shape == out.shape and (m & near).any():
            out |= m & near
    return out


def seed_frame(ctx, proxy: Path, n: int, frame: int, text: str, scale: float, run: Path,
               one: bool = False) -> tuple[int, list[np.ndarray]]:
    """Where to start tracking a label ("person"). On the chosen frame two people who touch (a hugging couple) are one
    SAM instance, and tracking that blob follows only one of them. Look at a few frames and start from the one where SAM
    finds the most separate objects (ties: the frame nearest the chosen one). Returns (frame, its instance masks)."""
    best, best_parts = frame, []
    if one:
        # One person ("man in green parka"): the most instances is the wrong goal. A second hit is a false one (the
        # child in his parka), and tracking it recast the child. The chosen frame, and its biggest match.
        # Further labels ("woman, handbag") are things that go with them: every match of those is kept.
        png = _frame_png(proxy, frame, run / f"seed_{frame}.png")
        found: list[list[np.ndarray]] = []
        for j, label in enumerate(labels(text)):
            try:
                found.append(select_instances(ctx, png, label, scale, run / f"seed_{frame}_{j}"))
            except Exception:  # noqa: BLE001
                found.append([])
        if found and found[0]:
            return frame, [max(found[0], key=lambda m: int(m.sum()))] + [m for ms in found[1:] for m in ms]
    for f in sorted({frame, 0, n // 4, n // 2, (3 * n) // 4, n - 1}, key=lambda f: abs(f - frame)):
        try:
            parts = select_instances(ctx, _frame_png(proxy, f, run / f"seed_{f}.png"), text, scale, run / f"seed_{f}")
        except Exception:  # noqa: BLE001
            continue
        if len(parts) > len(best_parts):
            best, best_parts = f, parts
    return best, best_parts


def tracked_masks(ctx, proxy: Path, frame: int, points: list[dict], text: str, src_w: int, run: Path,
                  others_out: list | None = None, one: bool = False) -> list[np.ndarray]:
    """Click on `frame` -> mask on that frame -> SAM 3.1 tracking forward and backward -> one proxy mask per frame.

    Nothing found (an empty shot, a label that isn't there) stops here, on single frames, before any tracking.

    `others_out` (a list, filled in place): everyone else SAM 3.1 finds on the pick frame is tracked too, together with
    the selection (SAM keeps people who touch apart far better that way: tracked alone, a parent's mask spread onto
    the child their hands were on), and the list gets their union on every frame, never overlapping the selection."""
    text = (text or "").strip()
    if not points and not text:
        raise RuntimeError("Click the person or thing in the picture first, or type what it is.")
    p = media.probe(proxy)
    n = p.frames
    frame = max(0, min(frame, n - 1))
    scale = p.width / src_w
    seeded: list[np.ndarray] = []
    # One person (a swap's target, `one`): a description picks its single best match. Elsewhere ("person" in Remove)
    # a label means every match.
    one = one and bool(text) and not points and not MANY.search(text)
    if text and not points:
        frame, seeded = seed_frame(ctx, proxy, n, frame, text, scale, run, one=one)
    pick_png = _frame_png(proxy, frame, run / "pick.png")
    first = select_preview(ctx, pick_png, points, text, scale, run / "pick")
    if one and seeded and first.shape == seeded[0].shape:
        first = first & np.logical_or.reduce(seeded)  # one person: only the matches seed_frame chose
    if not first.any() and seeded:  # the pick frame missed what another frame found
        first = np.logical_or.reduce(seeded)
    if not first.any():
        if points:
            raise RuntimeError("Nothing found where you clicked. Try clicking in the middle of it.")
        raise RuntimeError(NOBODY if PERSON.search(text) else
                           f"Couldn't find “{text}” in this shot. Click it in the picture, or try other words.")
    carried = with_carried(ctx, pick_png, first, text, scale, run) & ~first
    parts = seeded
    if text and not points and not parts:
        # A label ("person"): one mask per object SAM finds, so people who touch are still tracked separately.
        try:
            parts = select_instances(ctx, pick_png, text, scale, run / "pick_inst")
        except Exception:  # noqa: BLE001
            parts = []
    if len(parts) < 2:
        parts = split_objects(first)
    if carried.any():
        # A bag goes with the person it touches, tracked as an object of its own: merged into the person on this one
        # frame, SAM dropped a swinging bag a few frames later and its corner poked out from behind the new character.
        parts = parts + split_objects(carried)
    others = people_besides(ctx, pick_png, np.logical_or.reduce(parts), scale, run) if others_out is not None else []
    if others:
        each = track_each(ctx, proxy, frame, parts + others, run, n)
        masks = [np.logical_or.reduce([t[i] for t in each[:len(parts)]]) for i in range(n)]
        rest = [np.logical_or.reduce([t[i] for t in each[len(parts):]]) & ~masks[i] for i in range(n)]
    else:
        masks = track_from(ctx, proxy, frame, parts, run, n)
        rest = [np.zeros_like(m) for m in masks]
    if not any(m.any() for m in masks):
        raise RuntimeError(LOST)
    if others_out is not None:
        others_out[:] = rest
    return masks


OTHER_OVERLAP = 0.2  # a person found on the pick frame is someone else when less than this share of them is the selection


def people_besides(ctx, frame_png: Path, target: np.ndarray, scale: float, run: Path) -> list[np.ndarray]:
    """Everyone else SAM 3.1 finds as a "person" on this frame: each instance less than OTHER_OVERLAP the selection,
    without the selection's pixels. None found (or SAM failing) is no one."""
    try:
        found = select_instances(ctx, frame_png, "person", scale, run / "others")
    except Exception:  # noqa: BLE001 - not knowing who else is there: track the selection alone, as before
        return []
    out = []
    for m in found:
        if m.shape == target.shape and (m & target).sum() < OTHER_OVERLAP * m.sum():
            rest = m & ~target
            if rest.any():
                out.append(rest)
    return out


# --- "Check tracking": the tracked masks made before the render, kept for it ------------------------------------------
# The check (pickups/tracking.py) and the render key the masks the same way, so a checked selection costs nothing at
# render time. The render's proxy is cut afresh for every take, so the key names the footage it is cut from (the take's
# input file: size and modification time) and the proxy's own frame count and size, never the proxy file itself.


def track_key(shot_id: str | None, version_id: str | None, frame: int, points: list[dict], text: str, source: Path,
              proxy_frames: int, proxy_size: tuple[int, int]) -> str:
    """The cache key of one selection's tracked masks on one shot or take."""
    st = Path(source).stat()
    pts = [[round(float(p["x"]), 2), round(float(p["y"]), 2), bool(p.get("positive", True))] for p in points or []]
    blob = json.dumps({"shot": shot_id or "", "version": version_id or "orig", "frame": int(frame), "points": pts,
                       "text": (text or "").strip(), "source": [st.st_size, st.st_mtime_ns], "carried": list(CARRIED),
                       "proxy": [int(proxy_frames), int(proxy_size[0]), int(proxy_size[1])]}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def save_track(path: Path, masks: list[np.ndarray], others: list[np.ndarray] | None = None) -> None:
    """One bit per pixel, compressed: a few MB for 291 frames at 1024 px. `others`: the other people
    (tracked_masks' others_out), kept with them."""
    a = np.stack(masks).astype(bool)
    extra = {"others": np.packbits(np.stack(others).astype(bool), axis=None)} if others is not None and len(others) == len(a) else {}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, bits=np.packbits(a, axis=None), shape=np.array(a.shape), **extra)
    tmp.replace(path)


def load_track(path: Path | None, shape: tuple[int, int, int], others_out: list | None = None) -> list[np.ndarray] | None:
    """The masks saved at `path` when they are (frames, height, width) = `shape`, else None. With `others_out`, only
    masks saved with their other people count, and the list gets those."""
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as z:
            got = tuple(int(x) for x in z["shape"])
            if got != tuple(shape) or (others_out is not None and "others" not in z.files):
                return None
            count = int(np.prod(got))
            a = np.unpackbits(z["bits"], count=count).reshape(got).astype(bool)
            if others_out is not None:
                others_out[:] = list(np.unpackbits(z["others"], count=count).reshape(got).astype(bool))
    except (OSError, KeyError, ValueError):
        return None
    return list(a)


def _track_cache(ctx, proxy: Path, frame: int, points: list[dict], text: str) -> tuple[Path, tuple[int, int, int]] | None:
    """Where a task's selection would be cached, and the masks' shape; None outside a shot (or anything unreadable)."""
    try:
        p = media.probe(proxy)
        version = (getattr(ctx, "job", None) or {}).get("version_id")
        key = track_key(ctx.shot_id, version, frame, points, text, ctx.input_path, p.frames, (p.width, p.height))
        return ctx.store.shot_dir(ctx.shot_id) / "picks" / f"track_{key}.npz", (p.frames, p.height, p.width)
    except Exception:  # noqa: BLE001 - no cache is never an error: the masks are tracked as usual
        return None


def _note_cached(ctx, cached: bool) -> None:
    qa = getattr(ctx, "qa", None)
    if isinstance(qa, dict):
        qa["tracking_from_check"] = cached


def checked_masks(ctx, proxy: Path, frame: int, points: list[dict], text: str, src_w: int, run: Path,
                  others_out: list | None = None, one: bool = False) -> list[np.ndarray]:
    """tracked_masks, reusing "Check tracking"'s masks for the same selection on the same footage (and keeping the
    render's own for a later check or take). The take's _qa says which (tracking_from_check). `others_out`: as
    tracked_masks (a cached selection counts only when its other people were kept with it)."""
    where = _track_cache(ctx, proxy, frame, points, text)
    masks = load_track(*where, others_out=others_out) if where else None
    _note_cached(ctx, masks is not None)
    if masks is None:
        masks = tracked_masks(ctx, proxy, frame, points, text, src_w, run, others_out=others_out, one=one)
        if where:
            try:
                where[0].parent.mkdir(parents=True, exist_ok=True)
                save_track(where[0], masks, others_out)
            except OSError:
                pass
    return masks


def checked_tracks(ctx, proxy: Path, swaps: list[dict], others_out: list | None = None) -> list[list[np.ndarray]] | None:
    """Recast anyone: every person's masks from "Check tracking" when each of them was checked; else None (then they
    are tracked together, as usual). `others_out` gets the people no swap is: on each frame, who was someone else in
    every person's check, less every swapped person."""
    out, rest = [], []
    for s in swaps:
        where = _track_cache(ctx, proxy, s["frame"], s.get("points") or [], s.get("text", ""))
        o: list = []
        masks = load_track(*where, others_out=o if others_out is not None else None) if where else None
        if masks is None:
            _note_cached(ctx, False)
            return None
        out.append(masks)
        rest.append(o)
    _note_cached(ctx, bool(out))
    if out and others_out is not None:
        n = len(out[0])
        others_out[:] = [np.logical_and.reduce([o[i] for o in rest]) & ~np.logical_or.reduce([m[i] for m in out]) for i in range(n)]
    return out or None


def track_from(ctx, video: Path, frame: int, parts: list[np.ndarray], run: Path, n: int | None = None, folder: str = "",
               prefix: str = "pickups/track") -> list[np.ndarray]:
    """SAM 3.1 tracking from masks on `frame`, forward to the end and backward to the start: one mask per frame of
    `video` (n frames). Each part is an object of its own (one person of a couple). `folder` keeps a second tracking's
    files apart from the first's."""
    n = n or media.probe(video).frames
    if folder:
        run = run / folder
        run.mkdir(parents=True, exist_ok=True)
    assets = []
    for j, part in enumerate(parts):
        Image.fromarray(part.astype(np.uint8) * 255).convert("RGB").save(run / f"pick_mask_{j}.png")
        assets.append(ctx.client.assets.from_file(run / f"pick_mask_{j}.png"))
    mask_asset = assets if len(assets) > 1 else assets[0]
    fwd = ctx.run(graphs.track_mask(ctx.client.assets.from_file(_segment(video, frame, n - 1, run / "fwd.mp4")), mask_asset, prefix=prefix))
    masks_f = _download_masks(fwd, run / "track_fwd")
    masks_b: list[np.ndarray] = []
    if frame > 0:
        bwd = ctx.run(graphs.track_mask(ctx.client.assets.from_file(_segment(video, 0, frame, run / "bwd.mp4", reverse=True)),
                                        mask_asset, prefix=prefix))
        masks_b = _download_masks(bwd, run / "track_bwd")[::-1][:-1]  # back to forward order, drop the seed frame
    empty = np.zeros_like(parts[0])
    masks = [m if m.shape == empty.shape else np.asarray(Image.fromarray(m).resize(empty.shape[::-1], Image.NEAREST))
             for m in masks_b + masks_f]
    return (masks + [empty] * n)[:n]


def _download_each(job, d: Path, n_obj: int) -> list[list[np.ndarray]]:
    return [_download_masks(_Outputs(job, f"PU_Obj{j}"), d / f"obj{j}") for j in range(n_obj)]


class _Outputs:
    """One output node of a job, read as if it were PU_Mask (so _download_masks reads it)."""

    def __init__(self, job, name: str):
        self.job, self.name = job, name

    def get_outputs(self, _name: str):
        return self.job.get_outputs(self.name)


def track_each(ctx, video: Path, frame: int, parts: list[np.ndarray], run: Path, n: int | None = None, folder: str = "",
               prefix: str = "pickups/track") -> list[list[np.ndarray]]:
    """track_from, but every part comes back on its own: one list of per-frame masks per part, in the parts' order.
    The parts are tracked together (SAM 3.1 keeps people who touch apart far better that way); each object's track is
    matched back to the part it started from on `frame` (by overlap), whatever order SAM numbered them in."""
    n = n or media.probe(video).frames
    if folder:
        run = run / folder
        run.mkdir(parents=True, exist_ok=True)
    assets = []
    for j, part in enumerate(parts):
        Image.fromarray(part.astype(np.uint8) * 255).convert("RGB").save(run / f"pick_mask_{j}.png")
        assets.append(ctx.client.assets.from_file(run / f"pick_mask_{j}.png"))
    mask_asset = assets if len(assets) > 1 else assets[0]
    empty = np.zeros_like(parts[0])

    def fit(m: np.ndarray) -> np.ndarray:
        return m if m.shape == empty.shape else np.asarray(Image.fromarray(m).resize(empty.shape[::-1], Image.NEAREST))

    fwd = ctx.run(graphs.track_mask(ctx.client.assets.from_file(_segment(video, frame, n - 1, run / "fwd.mp4")), mask_asset,
                                    prefix=prefix, each=True))
    each_f = [[fit(m) for m in ms] for ms in _download_each(fwd, run / "track_fwd", len(parts))]
    order = _match_parts(parts, [ms[0] if ms else empty for ms in each_f])
    each_b: list[list[np.ndarray]] = [[] for _ in parts]
    if frame > 0:
        bwd = ctx.run(graphs.track_mask(ctx.client.assets.from_file(_segment(video, 0, frame, run / "bwd.mp4", reverse=True)),
                                        mask_asset, prefix=prefix, each=True))
        raw = [[fit(m) for m in ms] for ms in _download_each(bwd, run / "track_bwd", len(parts))]
        border = _match_parts(parts, [ms[0] if ms else empty for ms in raw])
        each_b = [raw[border[j]][::-1][:-1] for j in range(len(parts))]  # forward order, without the seed frame
    return [(each_b[j] + each_f[order[j]] + [empty] * n)[:n] for j in range(len(parts))]


def _match_parts(parts: list[np.ndarray], firsts: list[np.ndarray]) -> list[int]:
    """For each part, the index of the tracked object that starts on it: the pairs that overlap most first (in order
    when nothing overlaps)."""
    ov = {(j, i): int((f & p).sum()) if f.shape == p.shape else -1 for j, p in enumerate(parts) for i, f in enumerate(firsts)}
    out: dict[int, int] = {}
    for (j, i) in sorted(ov, key=lambda ji: (-ov[ji], ji[0], ji[1])):
        if j not in out and i not in out.values():
            out[j] = i
    return [out.get(j, j) for j in range(len(parts))]


def person_height(masks: list[np.ndarray]) -> int:
    """How tall the selection usually is, in mask pixels: the median height over the frames it is found on."""
    heights = []
    for m in masks:
        rows = np.flatnonzero(m.any(axis=1))
        if rows.size:
            heights.append(int(rows[-1] - rows[0] + 1))
    return int(np.median(heights)) if heights else 0


def require_size(masks: list[np.ndarray], frame_height: int) -> None:
    """Recast: stop before the render when the person is shorter than MIN_PERSON_PX on the 1024 px proxy. The message
    gives their height in the frame's own pixels (`frame_height` tall)."""
    h = person_height(masks)
    if not h:
        raise RuntimeError(LOST)
    mh, mw = masks[0].shape
    if h * PROXY_LONG / max(mh, mw) < MIN_PERSON_PX:
        raise RuntimeError(f"The person is too small in the frame for a good result (about {round(h * frame_height / mh)} px "
                           "tall). Try a closer shot.")


# Film grain for generated renders (target noise per channel, 8-bit). Generated video is grain-free, which reads as
# "plastic"; fresh per-frame grain breaks that up. "off" | "fine" (default) | "35mm".
GRAIN = {"fine": np.array([2.4, 2.4, 2.8], np.float32), "35mm": np.array([3.8, 3.8, 4.4], np.float32)}


def grain_pngs(folder: Path, pattern: str, preset: str = "fine") -> None:
    """Add film grain in place to every PNG frame matching `pattern` in `folder` (no-op for "off")."""
    if preset not in GRAIN:
        return
    from concurrent.futures import ThreadPoolExecutor

    from pickups import composite

    def add(i_f):
        i, f = i_f
        img = np.asarray(Image.open(f).convert("RGB"))
        Image.fromarray(composite.add_grain(img, GRAIN[preset], seed=i, size=0.6)).save(f)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(add, enumerate(sorted(folder.glob(pattern)))))


SHADOW_LABELS = ("shadow", "person's shadow")  # two phrasings catch more of a long, soft shadow


def shadow_masks(ctx, proxy: Path, objs: list[np.ndarray], run: Path, labels: tuple[str, ...] = SHADOW_LABELS) -> list[np.ndarray]:
    """Moving shots: the selected object's own cast shadow on each proxy frame.

    SAM 3.1 finds every "shadow" in the frame (or whatever `labels` name: a car's reflection on a wet road too); only
    pixels near the object (within ~60% of its height) are kept, so other people's or trees' shadows are left alone.
    (Locked-off shots use composite.cast_shadows instead.)"""
    from pickups import composite

    asset = ctx.client.assets.from_file(proxy)
    found: list[list[np.ndarray]] = []
    for k, label in enumerate(labels):
        job = ctx.run(graphs.select(asset, label, threshold=0.3, prefix=f"pickups/shadow{k}"))
        found.append(_download_masks(job, run / f"shadow{k}"))
    out = []
    for i, o in enumerate(objs):
        s = np.zeros_like(o)
        for f in found:
            if i < len(f) and f[i].shape == o.shape:
                s |= f[i]
        if not o.any():
            out.append(np.zeros_like(o))
            continue
        ys = np.nonzero(o)[0]
        near = composite.grow(o, max(4, int(0.6 * (ys.max() - ys.min() + 1))))
        out.append(s & near & ~o)
    return out


def encode_png_sequence(pattern: Path, fps: float, out: Path, crf: int = 14) -> None:
    run_local(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(pattern), "-c:v", "libx264", "-crf", str(crf),
               "-pix_fmt", "yuv420p", str(out)])


PRORES = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
# Lighter files for everyone else. H.264 plays everywhere; H.265 (10-bit, tagged hvc1 so QuickTime and Apple devices
# open it) is about half the size at the same quality.
H264 = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow", "-movflags", "+faststart"]
H265 = ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "18", "-preset", "medium", "-tag:v", "hvc1",
        "-x265-params", "log-level=error", "-movflags", "+faststart"]
DELIVERY = [("UHD", 3840, 2160), ("HD", 1920, 1080)]
# A master this close to 16:9 (or 9:16) is a 16:9 picture: the video models render at multiples of 32 or 128 px, so a
# 16:9 shot comes back 1920x1088 or 1920x1024, and is delivered at the standard sizes. Anything further off (4:3, 1:1,
# 4:5, 2.39:1, open gate) keeps its own shape.
WIDESCREEN_TOLERANCE = 0.06


def is_widescreen(width: int, height: int) -> bool:
    """16:9 or 9:16, give or take the render sizes' rounding (WIDESCREEN_TOLERANCE)."""
    long, short = max(width, height), max(1, min(width, height))
    return abs((long / short) / (16 / 9) - 1) <= WIDESCREEN_TOLERANCE


def delivery_sizes(width: int, height: int) -> list[tuple[str, int, int]]:
    """The sizes a master is delivered at: UHD and HD from a 4K-class master, HD from anything smaller.

    A 16:9 master gets the standard 3840x2160 and 1920x1080; a vertical one (a phone shot upright) 2160x3840 and
    1080x1920. Any other shape is delivered as it is, never cropped: its long side 3840 (UHD) or 1920 (HD), the other
    side to match, both even (a 4:3 1440x1080 master: HD 1920x1440)."""
    names = [(n, w, h) for n, w, h in DELIVERY if not (n == "UHD" and max(width, height) < 3000)]
    if is_widescreen(width, height):
        return [(n, h, w) for n, w, h in names] if height > width else names
    out = []
    for n, long, _ in names:
        s = long / max(width, height)
        out.append((n, int(round(width * s / 2)) * 2, int(round(height * s / 2)) * 2))
    return out


def size_details(width: int, height: int, frames: int, fps: float) -> dict:
    """A take's Details as delivered: the top delivery size, the size it was rendered at when that differs (previz
    renders 1920x1088 and delivers HD 1920x1080), and its real length in frames at its frame rate."""
    _, w, h = delivery_sizes(width, height)[0]
    out: dict = {"resolution": f"{w}×{h}"}
    if (width, height) != (w, h):
        out["rendered_at"] = f"{width}×{height}"
    return out | {"frames": int(frames), "fps": round(float(fps), 3)}


def sequence_details(folder: Path, pattern: str, fps: float) -> dict:
    """size_details for a rendered PNG sequence (its frame size and count)."""
    files = sorted(folder.glob(pattern))
    with Image.open(files[0]) as im:
        w, h = im.size
    return size_details(w, h, len(files), fps)


def _encode(master: Path, dst: Path, w: int, h: int, codec: list[str], audio: list[str], same_size: bool) -> None:
    # Fill w x h, then trim the overspill: for a 16:9 render size (1920x1088) the few rows LTX rounded up by; for every
    # other shape delivery_sizes keeps the master's own, so nothing but rounding is cut.
    vf = [] if same_size else ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1"]
    run_local(["ffmpeg", "-v", "error", "-y", "-i", str(master), *vf, *audio, *codec, str(dst)])


def deliverables(master: Path, vdir: Path) -> list[dict]:
    """Delivery files from the master, most-used first.

    ProRes 422 HQ .mov at UHD (4K-class sources) and HD, MP4 H.264 at UHD and HD, MP4 H.265 10-bit at the top size,
    and the native-resolution ProRes master. A master that isn't 16:9 (4:3, square, open gate) is delivered in its own
    shape (delivery_sizes), never cropped to 16:9. Each entry: label, file, codec, size.
    """
    p = media.probe(master)
    has_audio = p.has_audio
    sizes = delivery_sizes(p.width, p.height)
    out: list[dict] = []
    native_is_standard = False
    for name, w, h in sizes:  # ProRes
        same = (p.width, p.height) == (w, h)
        if same:
            native_is_standard = True
            out.append({"label": f"ProRes 422 HQ · {name} {w}×{h}", "file": master.name, "codec": "prores", "size": name})
            continue
        dst = vdir / f"{name.lower()}_{w}x{h}.mov"
        _encode(master, dst, w, h, PRORES, ["-c:a", "copy"] if has_audio else ["-an"], same)
        out.append({"label": f"ProRes 422 HQ · {name} {w}×{h}", "file": dst.name, "codec": "prores", "size": name})
    aac = ["-c:a", "aac", "-b:a", "320k"] if has_audio else ["-an"]
    for name, w, h in sizes:  # H.264 at every delivery size
        dst = vdir / f"{name.lower()}_{w}x{h}_h264.mp4"
        _encode(master, dst, w, h, H264, aac, (p.width, p.height) == (w, h))
        out.append({"label": f"MP4 · H.264 · {name} {w}×{h}", "file": dst.name, "codec": "h264", "size": name})
    name, w, h = sizes[0]  # H.265 at the top size
    dst = vdir / f"{name.lower()}_{w}x{h}_h265.mp4"
    _encode(master, dst, w, h, H265, aac, (p.width, p.height) == (w, h))
    out.append({"label": f"MP4 · H.265 (HEVC) 10-bit · {name} {w}×{h}", "file": dst.name, "codec": "h265", "size": name})
    if not native_is_standard:
        out.append({"label": f"ProRes 422 HQ · native {p.width}×{p.height}", "file": master.name, "codec": "prores", "size": "native"})
    return out


def finish_outputs(vdir: Path, frames_pattern: Path, fps: float, audio_from: Path | None = None) -> list[dict]:
    """From a PNG sequence: ProRes 422 HQ master + UHD/HD deliverables, H.264 viewer proxy, thumbnail."""
    master = vdir / "master.mov"
    args = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames_pattern)]
    if audio_from is not None and media.probe(audio_from).has_audio:
        # Audio padded with silence, so -shortest ends on the last frame (an audio track a hair short dropped it).
        args += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a", "-af", "apad", "-c:a", "pcm_s24le", "-shortest"]
    run_local(args + PRORES + [str(master)])  # a Cancel stops it (comfy.run_local)
    make_viewer(master, vdir / "video.mp4")
    make_thumb(vdir / "video.mp4", vdir / "thumb.jpg")
    return deliverables(master, vdir)


def read_frames(video: Path, vf: str = "", size: tuple[int, int] | None = None):
    """Yield RGB frames (uint8 HxWx3) from a video through an ffmpeg pipe (no PNGs on disk).

    If `vf` changes the frame size, pass the resulting `size` (w, h)."""
    p = media.probe(video)
    w, h = size or (p.width, p.height)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video)] + (["-vf", vf] if vf else []) +
                            ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    assert proc.stdout
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


class FrameWriter:
    """Pipe RGB frames into ffmpeg: ProRes 422 HQ master (default) or an H.264 review file (scaled)."""

    def __init__(self, out: Path, w: int, h: int, fps: float, kind: str = "prores", scale_long: int | None = None, crf: int = 17):
        vf = []
        if scale_long and max(w, h) > scale_long:
            s = scale_long / max(w, h)
            vf = ["-vf", f"scale={int(w * s) // 2 * 2}:{int(h * s) // 2 * 2}:flags=lanczos"]
        codec = PRORES if kind == "prores" else ["-c:v", "libx264", "-crf", str(crf), "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                                      "-i", "-", *vf, *codec, str(out)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing video")


def outputs_from_master(vdir: Path, master: Path, audio_from: Path | None = None) -> list[dict]:
    """Add the source audio to a piped master, then viewer proxy, thumbnail and UHD/HD deliverables."""
    if audio_from is not None and media.probe(audio_from).has_audio:
        tmp = master.with_name("master_av.mov")
        # The sound padded or cut to the picture's length: -shortest dropped the last frame of 29.97 fps masters.
        seconds = media.probe(master).seconds
        run_local(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-i", str(audio_from), "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-af", "apad", "-c:a", "pcm_s24le", "-t", f"{seconds:.6f}", str(tmp)])
        tmp.replace(master)
    make_viewer(master, vdir / "video.mp4")
    make_thumb(vdir / "video.mp4", vdir / "thumb.jpg")
    return deliverables(master, vdir)


WORKERS = max(2, min(12, (os.cpu_count() or 4) - 4))


def parallel_frames(frames, fn, batch: int = 24):
    """Apply fn(i, frame) to frames on a thread pool (numpy/PIL release the GIL), yielding results in order."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(WORKERS) as pool:
        buf: list = []
        i0 = 0
        for frame in frames:
            buf.append(frame)
            if len(buf) == batch:
                yield from pool.map(fn, range(i0, i0 + len(buf)), buf)
                i0 += len(buf)
                buf = []
        if buf:
            yield from pool.map(fn, range(i0, i0 + len(buf)), buf)
