"""Check tracking: Recast's selection followed through the whole shot before the render, to look at.

The same tracked masks the render makes (common.tracked_masks, on the same proxy recast_ltx.run cuts: the first
search_frames() frames of the shot or take at PROXY_LONG), shown as the proxy with the selection tinted amber, plus a
per-frame health check. The masks are cached under the shot (picks/track_<key>.npz) by common.track_key, so the render
of a checked selection skips tracking (common.checked_masks)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from pickups import media
from pickups.app.store import media_url
from pickups.tasks import common

AMBER = np.array([245, 184, 61], np.float32)  # the select overlay's colour
ALPHA = 0.45
OVERLAY_W = 960      # the overlay video is at most this wide
NEAR = 3             # a frame is compared with the median of this many frames each side
DROP = 0.35          # ...and flagged when its area is more than this much smaller
KEEP = 12            # checks kept per shot (masks and videos); older ones are deleted


def search_frames(total: int, start_frame=None) -> int:
    """How many frames from the start the render's proxy covers (as recast_ltx.run and recast_many.run cut it)."""
    from pickups.tasks import recast_ltx

    frames = 8 * ((min(common.MAX_FRAMES, total) - 1) // 8) + 1
    start = None if start_frame in (None, "") else max(0, min(int(start_frame), total - frames))
    return min(total, max(recast_ltx.search_frames(), (start or 0) + frames))


def health(masks: list[np.ndarray], near: int = NEAR, drop: float = DROP) -> dict:
    """Per frame: the selection's area (share of the frame); the frames where it is more than `drop` smaller than the
    median of the `near` frames each side (lost a part, or lost them); the frames where it is empty.

    Walking in or out of the picture isn't losing them: a drop next to empty frames is not flagged when the masks that
    border the empty frames touch the frame's edge."""
    area = np.array([int(m.sum()) for m in masks], np.float64)
    n = len(area)
    edge = [bool(a and (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())) for m, a in zip(masks, area)]
    border = [bool(area[j] and ((j > 0 and not area[j - 1]) or (j < n - 1 and not area[j + 1]))) for j in range(n)]
    flagged = []
    for i in range(n):
        around = np.concatenate([area[max(0, i - near):i], area[i + 1:i + 1 + near]])
        if not (around.size and (med := float(np.median(around))) > 0 and area[i] < (1 - drop) * med):
            continue
        lo, hi = max(0, i - near), min(n, i + near + 1)
        edges = [edge[j] for j in range(lo, hi) if border[j]]
        if edges and all(edges) and not area[lo:hi].all():
            continue  # entering or leaving the picture
        flagged.append(i)
    total = masks[0].size if masks else 1
    return {"area": [round(float(a) / total, 5) for a in area], "flagged": flagged,
            "empty": [i for i, a in enumerate(area) if a == 0]}


def overlay(proxy: Path, masks: list[np.ndarray], out: Path) -> None:
    """The proxy with the selection tinted amber, at most OVERLAY_W wide (H.264, for the browser)."""
    p = media.probe(proxy)
    s = min(1.0, OVERLAY_W / p.width)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    w = common.FrameWriter(tmp, p.width, p.height, float(p.fps), kind="h264",
                           scale_long=int(max(p.width, p.height) * s) if s < 1 else None)
    try:
        for i, f in enumerate(common.read_frames(proxy)):
            m = masks[i] if i < len(masks) else None
            if m is not None and m.shape == f.shape[:2] and m.any():
                f = f.copy()
                f[m] = (f[m] * (1 - ALPHA) + AMBER * ALPHA).astype(np.uint8)
            w.write(f)
    finally:
        w.close()
    tmp.replace(out)


def _prune(d: Path, keep: int = KEEP) -> None:
    """Keep the newest `keep` checks in the shot's picks folder."""
    done = sorted(d.glob("track_*.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in done[keep:]:
        for g in (f, f.with_suffix(".mp4")):
            g.unlink(missing_ok=True)


def proxy_for(store, shot_id: str, version_id: str | None, start_frame=None) -> tuple[Path, Path]:
    """(the shot's or take's input, the render's proxy of it): cut once into the shot's picks folder and cut again when
    the input changes."""
    src = store.version_master(shot_id, version_id)
    if not src.exists():
        raise FileNotFoundError(src)
    p = media.probe(src)
    n = search_frames(p.frames, start_frame)
    d = store.shot_dir(shot_id) / "picks"
    d.mkdir(exist_ok=True)
    vid = version_id if version_id and version_id != "orig" else "orig"
    proxy = d / f"track_proxy_{vid}_{n}.mp4"
    if not proxy.exists() or proxy.stat().st_mtime < src.stat().st_mtime:
        tmp = proxy.with_name("tmp_" + proxy.name)
        media.prepare_work_clip(src, tmp, 0, n, media.work_size(p.width, p.height, common.PROXY_LONG))
        common.check_not_blank(tmp)
        tmp.replace(proxy)
    return src, proxy


def check(runner, store, shot_id: str, version_id: str | None, frame: int, points: list[dict], text: str,
          start_frame=None) -> dict:
    """Track the selection through the shot (or reuse a check of the same selection) and return the overlay and the
    health: {video_url, frames, fps, area, flagged, empty, cache_key, cached}. Raises RuntimeError worded for the user
    when there's nothing to follow."""
    src, proxy = proxy_for(store, shot_id, version_id, start_frame)
    pp = media.probe(proxy)
    vid = version_id if version_id and version_id != "orig" else "orig"
    key = common.track_key(shot_id, vid, frame, points, text, src, pp.frames, (pp.width, pp.height))
    d = proxy.parent
    path = d / f"track_{key}.npz"
    others: list = []  # everyone else, tracked with them (the render keeps them as footage): cached together
    masks = common.load_track(path, (pp.frames, pp.height, pp.width), others_out=others)
    cached = masks is not None
    if masks is None:
        run = d / f"track_{key}"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir()
        try:
            masks = common.tracked_masks(runner, proxy, frame, points, text, media.probe(src).width, run, others_out=others,
                                             one=True)
        finally:
            shutil.rmtree(run, ignore_errors=True)
        common.save_track(path, masks, others)
    video = d / f"track_{key}.mp4"
    if not cached or not video.exists():  # tracked again: the picture of it too
        overlay(proxy, masks, video)
    path.touch()  # newest first when pruning
    _prune(d)
    h = health(masks)
    return {"video_url": media_url(video), "frames": len(masks), "fps": float(pp.fps), **h, "cache_key": key, "cached": cached}
