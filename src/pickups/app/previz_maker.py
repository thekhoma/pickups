"""Browser previz: JPEG frames from the page become a ProRes shot in the library.

The page renders each frame itself (so the preview and the file match) and uploads them in small batches.
This module checks the spec, rejects a blank frame, and assembles the sequence the way a trimmed clip is prepared.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pickups import media

SESSION = re.compile(r"pv_[0-9a-f]{16}")
BATCH = 16
# Biggest a previz shot is allowed to be, even before the GPU cap (169 is 7 s at 24 fps).
HARD_CAP = 169
SIZE = (1920, 1088)
FRAME0 = "previz_frame0.jpg"
# Frame 0's subject matte from the previz renderer (white: every person and vehicle; black: the rest), for the start
# frame's canny mask (start_frame.py). Never derived from colour: a car is the same grey as the road.
SUBJECTS0 = "previz_subjects0.png"
# The page's depth passes (runtime.ts depthPasses: frame 0, and 48 and 96 for the rescue keyframes): view-space z in 24-bit
# RGB, sky black. start_frame.DEPTH_FILE names them; start_frame.decode_depth reads them.
DEPTH_FRAMES = (0, 48, 96)
DEPTH_MAX_BYTES = 24_000_000
FACINGS = ("camera", "away", "left", "right")


class PrevizError(ValueError):
    """One line, safe to show in the page."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraKey(_Strict):
    t: float = Field(ge=0, le=1)
    pos: tuple[float, float, float]
    look: tuple[float, float, float]
    focal_mm: float | None = Field(default=None, ge=8, le=200)
    roll: float | None = Field(default=None, ge=-180, le=180)


class Subject(_Strict):
    id: str = Field(min_length=1, max_length=40)
    kind: Literal["humanoid", "car", "product", "dog", "cat", "bike", "skate", "boat"]
    height_m: float = Field(gt=0, le=2000)
    pos: tuple[float, float, float]
    anim: str = Field(max_length=40)
    speed: float | None = Field(default=None, gt=0, le=4)


class Lens(_Strict):
    focal_mm: float = Field(ge=8, le=200)
    film_gauge_mm: Literal[36]


class Environment(_Strict):
    ground: Literal["plane", "none"]
    sky: Literal["gradient"]
    fog: tuple[float, float]
    props: list[str] = Field(max_length=24)


class Camera(_Strict):
    interp: Literal["catmull_rom"]
    # How round the path is between keys (web cameraMath.ts curveOf): 0 straight, 1 smooth, 1.5 rounder.
    curve: float = Field(default=1, ge=0, le=1.5)
    ease: Literal["in_out", "linear"]
    speed: float = Field(default=1, ge=0.25, le=3)
    taper: float = Field(default=1, ge=0, le=1)
    distance: float = Field(default=1, ge=0.35, le=4)
    frame: tuple[float, float] | None = None
    free: bool = False
    # With free: the camera still rides with the subject (a bent tracking path keeps following). Stored, never read here.
    follow: bool | None = None
    pick: (
        Literal["close", "wide", "close-wide", "dolly-in", "dolly-out", "orbit", "drone", "fpv", "track", "parallel", "locked"]
        | None
    ) = None
    keys: list[CameraKey] = Field(min_length=2, max_length=24)
    segments: list["CameraSegment"] = Field(default_factory=list, max_length=3)
    handheld: "Handheld | None" = None

    @field_validator("frame")
    @classmethod
    def frame_in_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is None:
            return None
        if any(n < -1.5 or n > 1.5 for n in value):
            raise ValueError("frame is out of range")
        return value


class CameraSegment(_Strict):
    move: str = Field(min_length=1, max_length=40)
    from_t: float = Field(ge=0, le=1)
    to_t: float = Field(ge=0, le=1)
    params: dict[str, str | float | int | bool] = Field(default_factory=dict)
    tracking: float = Field(ge=0, le=1)


class Handheld(_Strict):
    seed: int
    pos_m: float = Field(ge=0, le=0.2)
    rot_deg: float = Field(ge=0, le=5)
    hz: float = Field(ge=0, le=4)


class Beat(_Strict):
    actor: str | None = Field(default=None, max_length=40)
    type: str = Field(min_length=1, max_length=40)
    from_t: float = Field(ge=0, le=1)
    to_t: float = Field(ge=0, le=1)
    params: dict[str, str | float | int | bool] = Field(default_factory=dict)


class PrevizSpecModel(_Strict):
    version: Literal[1]
    template: Literal["giant_orbit", "street_run", "tunnel", "car_track", "product_orbit", "described"]
    duration_s: float = Field(gt=0, le=60)
    fps: Literal[24]
    size: tuple[Literal[1920], Literal[1088]]
    lens: Lens
    environment: Environment
    subjects: list[Subject] = Field(max_length=6)
    camera: Camera
    beats: list[Beat] = Field(default_factory=list, max_length=8)
    seed: int
    speed: float = Field(default=1, gt=0, le=20)
    notes: str = Field(default="", max_length=2000)


def frame_count(duration_s: float, fps: int) -> int:
    """Frames the page will render. The small bias keeps 97/24 * 24 from rounding down."""
    return max(1, round(float(duration_s) * int(fps) + 1e-6))


def parse_spec(data: object) -> PrevizSpecModel:
    if not isinstance(data, dict):
        raise PrevizError("That previz spec isn't valid.")
    try:
        return PrevizSpecModel.model_validate(data)
    except ValidationError:
        raise PrevizError("That previz spec isn't valid.") from None


def _dir(root: Path, session: str) -> Path:
    if not SESSION.fullmatch(session):
        raise PrevizError("That previz session isn't valid.")
    return root / "previz" / session


def save_frames(root: Path, session: str, start: int, blobs: list[bytes]) -> int:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0 or not blobs or len(blobs) > BATCH:
        raise PrevizError("Send frames in batches of 16.")
    if start + len(blobs) > HARD_CAP:
        raise PrevizError("That's more frames than a previz shot can hold.")
    d = _dir(root, session) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(blobs):
        if len(data) < 32 or data[:2] != b"\xff\xd8" or len(data) > 12 * 1024 * 1024:
            raise PrevizError("A frame isn't a JPEG.")
        (d / f"{start + i:04d}.jpg").write_bytes(data)
    return start + len(blobs)


def discard(root: Path, session: str) -> None:
    if not SESSION.fullmatch(session):
        raise PrevizError("That previz session isn't valid.")
    shutil.rmtree(_dir(root, session), ignore_errors=True)


def _check_frame(path: Path, index: int | None = None) -> None:
    where = f"Frame {index + 1}" if index is not None else "A frame"
    try:
        with Image.open(path) as im:
            if im.size != SIZE or im.format != "JPEG":
                raise PrevizError("A frame isn't 1920×1088.")
            lo, hi = im.convert("L").getextrema()
    except PrevizError:
        raise
    except Exception:
        raise PrevizError("A frame couldn't be read.") from None
    if hi - lo < 8:
        raise PrevizError(f"{where} came out blank. Check the preview, then try Make shot again.")


def check_frames(paths: list[Path]) -> None:
    """Every 8th frame and the last: a context lost halfway through an export shows up as a blank frame."""
    for i in sorted({*range(0, len(paths), 8), len(paths) - 1}):
        _check_frame(paths[i], i)


def _framing(data: object) -> dict | None:
    """Frame 0's framing from the page: the main subject's box and the horizon, as fractions of the frame."""
    if not isinstance(data, dict):
        return None
    out: dict = {}
    box = data.get("subject")
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in box)
        if x1 > x0 and y1 > y0:
            out["subject"] = [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
    hz = data.get("horizon")
    if isinstance(hz, (int, float)) and -1 <= hz <= 2:
        out["horizon"] = round(float(hz), 4)
    view = _view(data.get("view"))
    if view:
        out["view"] = view
    place = data.get("place")
    if isinstance(place, str) and place.strip():
        out["place"] = " ".join(place.split())[:40]
    reveals = [
        {"word": r["word"], "at_s": round(float(r["at_s"]), 1), "side": r["side"]}
        for r in (data.get("reveals") or [])[:12]
        if isinstance(r, dict) and _word(r.get("word")) == r.get("word") and isinstance(r.get("at_s"), (int, float))
        and 0 <= r["at_s"] <= 60 and r.get("side") in ("left", "right", "middle")
    ] if isinstance(data.get("reveals"), list) else []
    if reveals:
        out["reveals"] = reveals
    cam = data.get("camera")
    if isinstance(cam, dict) and all(isinstance(cam.get(k), (int, float)) for k in ("height_m", "pitch_deg", "focal_mm")):
        out["camera"] = {"height_m": round(min(500.0, max(-50.0, float(cam["height_m"]))), 2),
                         "pitch_deg": round(min(90.0, max(-90.0, float(cam["pitch_deg"]))), 1),
                         "focal_mm": int(min(300, max(4, round(float(cam["focal_mm"])))))}
    subs = []
    for s in (data.get("subjects") or [])[:12] if isinstance(data.get("subjects"), list) else []:
        if not isinstance(s, dict) or s.get("kind") not in ("person", "vehicle", "animal"):
            continue
        if not all(isinstance(s.get(k), (int, float)) for k in ("x", "distance_m")):
            continue
        one = {"kind": s["kind"], "x": round(min(3.0, max(-2.0, float(s["x"]))), 2),
               "distance_m": round(min(5000.0, max(0.0, float(s["distance_m"]))), 1), "in_view": s.get("in_view") is True}
        for k in ("doing", "word"):
            if isinstance(s.get(k), str) and _word(s[k]) == s[k]:
                one[k] = s[k]
        if s.get("facing") in FACINGS:  # from the figure's rotation (runtime.ts facingOf)
            one["facing"] = s["facing"]
        if isinstance(s.get("speed_ms"), (int, float)) and 0 < s["speed_ms"] < 120:
            one["speed_ms"] = round(float(s["speed_ms"]), 1)
        for k in ("w", "h"):  # Run 2: its size on screen, fractions of the frame (the still's size sentence)
            if isinstance(s.get(k), (int, float)) and 0 < s[k] <= 1:
                one[k] = round(float(s[k]), 3)
        subs.append(one)
    if subs:
        out["subjects"] = subs
    move = data.get("move")  # the camera's real move, first and last frame (runtime.ts moveOf)
    ends = [move.get(k) for k in ("start", "end")] if isinstance(move, dict) else []
    if len(ends) == 2 and all(isinstance(e, dict) and all(_vec(e.get(k)) for k in ("pos", "look")) for e in ends):
        out["move"] = {k: {p: [round(min(1e4, max(-1e4, float(v))), 3) for v in e[p]] for p in ("pos", "look")}
                       for k, e in zip(("start", "end"), ends)}
    light = data.get("light")
    if isinstance(light, dict) and light.get("dir") in ("camera left", "camera right", "behind camera", "backlit"):
        out["light"] = {"dir": light["dir"]}
    world = data.get("world")
    if isinstance(world, dict):
        w = {k: world[k] for k in ("time", "weather") if isinstance(world.get(k), str) and _word(world[k]) == world[k]}
        if w:
            out["world"] = w
    return out or None


def _vec(v: object) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)


def _word(w: object) -> str | None:
    return " ".join(w.split())[:30] if isinstance(w, str) and w.strip() else None


def _view(data: object) -> dict | None:
    """What frame 0 shows (inView.ts): the share of the frame per thing, the things out of frame, the actors in view."""
    if not isinstance(data, dict):
        return None
    share = data.get("share") if isinstance(data.get("share"), dict) else {}
    out: dict = {
        "share": {k: round(min(1.0, max(0.0, float(v))), 3) for k, v in list(share.items())[:30]
                  if _word(k) == k and isinstance(v, (int, float))},
        "outside": [w for w in (data.get("outside") or [])[:30] if _word(w) == w] if isinstance(data.get("outside"), list) else [],
        "actors": [int(a) for a in (data.get("actors") or [])[:6] if isinstance(a, int) and 0 <= a < 6]
        if isinstance(data.get("actors"), list) else [],
    }
    xs = data.get("x") if isinstance(data.get("x"), dict) else {}
    out["x"] = {k: round(min(1.0, max(0.0, float(v))), 2) for k, v in xs.items() if k in out["share"] and isinstance(v, (int, float))}
    for k in ("blocked", "bare"):
        if isinstance(data.get(k), (int, float)):
            out[k] = round(min(1.0, max(0.0, float(data[k]))), 3)
    return out


def _matte(data: object) -> bytes | None:
    """The page's subject matte: a PNG data URL of the frame's size, kept as a one-channel PNG. Anything else is ignored."""
    if not isinstance(data, str) or not data.startswith("data:image/png;base64,") or len(data) > 8_000_000:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1], validate=True)))
        im.load()
    except (ValueError, OSError):
        return None
    if im.size != SIZE:
        return None
    buf = io.BytesIO()
    im.convert("L").point(lambda v: 255 if v > 127 else 0).save(buf, "PNG")
    return buf.getvalue()


def _depths(data: object) -> dict[int, bytes]:
    """The page's depth passes: {"<frame>": PNG data URL} at the frame's size, kept as RGB PNGs (lossless, so the 24-bit
    code survives). Frames other than DEPTH_FRAMES, and anything that isn't such a PNG, are ignored."""
    out: dict[int, bytes] = {}
    if not isinstance(data, dict):
        return out
    for key, url in list(data.items())[: len(DEPTH_FRAMES)]:
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        if frame not in DEPTH_FRAMES or not isinstance(url, str) or not url.startswith("data:image/png;base64,") or len(url) > DEPTH_MAX_BYTES:
            continue
        try:
            im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True)))
            im.load()
        except (ValueError, OSError):
            continue
        if im.size != SIZE:
            continue
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG")
        out[frame] = buf.getvalue()
    return out


def save_depths(folder: Path, data: object) -> list[int]:
    """Write the depth passes into a shot folder; the frames written."""
    from pickups.start_frame import DEPTH_FILE

    kept = _depths(data)
    for frame, png in kept.items():
        (folder / DEPTH_FILE.format(frame)).write_bytes(png)
    return sorted(kept)


def refresh_frame0(store, shot_id: str, framing: object, matte: object, depth: object = None) -> dict:
    """Give a shot made before frame 0 recorded its subjects, camera and matte what the page re-rendered from its spec.
    The new framing replaces the old one; frame 0's picture and hash stay as exported."""
    meta = store.shot_meta(shot_id)
    if not isinstance(meta.get("previz_spec"), dict):
        raise ValueError("not a previz shot")
    fr = _framing(framing)
    kept = _matte(matte)
    if fr:
        store.set_shot_meta(shot_id, previz_framing={**(meta.get("previz_framing") or {}), **fr})
    if kept:
        (store.shot_dir(shot_id) / SUBJECTS0).write_bytes(kept)
    save_depths(store.shot_dir(shot_id), depth)
    return store.shot(shot_id)


def finish(root: Path, store, session: str, spec_data: object, project_id: str | None, name: str, cap: int,
           framing: object = None, matte: object = None, depth: object = None) -> dict:
    spec = parse_spec(spec_data)
    n = frame_count(spec.duration_s, spec.fps)
    if n > int(cap):
        raise PrevizError(f"This GPU renders up to {int(cap)} frames. Shorten the shot.")
    folder = _dir(root, session) / "frames"
    paths = [folder / f"{i:04d}.jpg" for i in range(n)]
    if not all(p.is_file() for p in paths):
        raise PrevizError("Some frames didn't arrive. Try the export again.")
    check_frames(paths)
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    out = root / f"previz_{session}.mov"
    try:
        media.assemble_jpeg_frames(folder, out, n, Fraction(spec.fps, 1), SIZE)
        probed = media.probe(out)
        if (probed.width, probed.height) != SIZE or probed.frames != n or abs(float(probed.fps) - spec.fps) > 0.02:
            raise PrevizError("The shot didn't come out at the right size or length.")
        title = " ".join(str(name).split())[:120] or "Previz"
        pid = project_id if isinstance(project_id, str) else None
        extra = {"previz_spec": spec.model_dump(mode="json"), "previz_frame0": digest}
        fr = _framing(framing)
        if fr:
            extra["previz_framing"] = fr
        shot = store.add_shot(out, title, move=True, project_id=pid, extra=extra)
        # Frame 0 as exported: the start-frame check compares against these exact bytes (their hash is previz_frame0).
        shutil.copyfile(paths[0], store.shot_dir(shot["id"]) / FRAME0)
        kept = _matte(matte)
        if kept:
            (store.shot_dir(shot["id"]) / SUBJECTS0).write_bytes(kept)
        save_depths(store.shot_dir(shot["id"]), depth)
        store.touch_project(shot["project_id"])
        return shot
    finally:
        out.unlink(missing_ok=True)
        shutil.rmtree(_dir(root, session), ignore_errors=True)
