"""ffmpeg/ffprobe helpers: probe sources and build the padded "canvas" clips the retake graph needs.

Orientation: a phone clip shot upright is stored sideways with a display rotation. ffmpeg applies that rotation
whenever it decodes, so every size here is the picture as displayed, and every file Pickups writes is upright with no
rotation left in it (a vertical shot gives a vertical take).

Frame rate: tasks work frame for frame at the source's own rate, so a take cuts straight back over the original on the
same timeline with the same timecode. The video models get at most tasks.common.MAX_FRAMES frames from the start of a
shot (4 s at 24 fps, 1.6 s at 60), and a take made from part of a shot says how much (made_from).

Colour: phones record HDR video by default (BT.2020, HLG). Decoded as is it looks grey and washed out next to the
original, so wide-gamut clips are converted to the BT.709 everything else is made in (to_rec709).
"""

from __future__ import annotations

import functools
import json
import math
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# LTX latents are 32 px spatially and 8 frames temporally (+1).
SPATIAL = 32
MIN_FRAMES = 9  # the shortest clip the video models render (LTX: 8k+1 frames)
# Standard rates. Phones record a variable frame rate, and some files report a time base (1000/1, 90000/1) or a peak
# rate as the stream's rate; their average rate, snapped to one of these, is the rate they play at.
RATES = tuple(Fraction(r) for r in ("24000/1001", "24", "25", "30000/1001", "30", "48", "50", "60000/1001", "60",
                                    "100", "120000/1001", "120", "240"))


class MediaError(ValueError):
    """A file Pickups can't work on. The message is for the person who added it."""


UNREADABLE = "Pickups can't read that file as video. It may be damaged or still copying: try exporting it again as an MP4 or MOV."


@dataclass(frozen=True)
class Probe:
    width: int  # as displayed (rotation applied)
    height: int
    fps: Fraction
    frames: int
    has_audio: bool
    audio_rate: int | None
    rotation: int = 0  # the display rotation stored in the file (0, 90, 180, 270); ffmpeg applies it when decoding
    primaries: str = ""  # colour tags as ffprobe names them ("bt709", "bt2020"; "" when untagged)
    transfer: str = ""  # "bt709", "arib-std-b67" (HLG), "smpte2084" (PQ)...

    @property
    def seconds(self) -> float:
        return self.frames / float(self.fps)


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _fraction(text) -> Fraction | None:
    try:
        f = Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def frame_rate(r_rate: str | None, avg_rate: str | None) -> Fraction:
    """The rate a clip plays at: the stream's rate when it agrees with the average, else the average (snapped to a
    standard rate when it is within 2% of one)."""
    r, avg = _fraction(r_rate), _fraction(avg_rate)
    if r and r <= 240 and (avg is None or abs(r - avg) <= 0.02 * avg):
        return r
    if avg:
        near = min(RATES, key=lambda x: abs(x - avg))
        return near if abs(near - avg) <= 0.02 * avg else avg.limit_denominator(1001)
    if r:
        return r
    raise MediaError("Pickups can't tell this clip's frame rate. Try exporting it again as an MP4 or MOV.")


def _rotation(stream: dict) -> int:
    """Display rotation as 0, 90, 180 or 270: the display matrix (side data), or an older file's "rotate" tag."""
    r = next((sd["rotation"] for sd in stream.get("side_data_list") or [] if "rotation" in sd), None)
    if r is None:
        r = (stream.get("tags") or {}).get("rotate", 0)
    try:
        return round(float(r) / 90) * 90 % 360
    except (TypeError, ValueError):
        return 0


def _count(stream: dict) -> int:
    """Frames decoded (when counted: what ffmpeg will really get), else the container's count; 0 when unknown."""
    for key in ("nb_read_frames", "nb_frames"):
        if key in stream:
            try:
                return int(stream[key])
            except ValueError:  # "N/A"
                continue
    return 0


_probed: dict[tuple, Probe] = {}


def probe(path: Path, count: bool = True) -> Probe:
    """Size (as displayed), frame rate, frame count and audio of a clip. Raises MediaError, worded for the person who
    added the file, for anything that isn't a readable video. count=False trusts the container's frame count when it
    has one (no decode: fast on long files).

    Counting decodes the whole clip (seconds on a long 4K file) and a task asks about its input several times, so
    results are kept per file version (path, size, modification time)."""
    try:
        st = Path(path).stat()
        key = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns, count)
    except OSError:
        st, key = None, None
    if key in _probed:
        return _probed[key]
    if st is not None and st.st_size == 0:
        raise MediaError("That file is empty. Try exporting the clip again.")
    p = _probe(path, count)
    if key is not None:
        if len(_probed) > 512:
            _probed.clear()
        _probed[key] = p
    return p


def _probe(path: Path, count: bool) -> Probe:
    try:
        info = json.loads(_run(["ffprobe", "-v", "error", "-show_streams"] + (["-count_frames"] if count else []) +
                               ["-of", "json", str(path)]))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise MediaError(UNREADABLE) from e
    streams = info.get("streams") or []
    # Cover art in a music file is a one-picture "video" stream: not a video.
    video = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaError("That file has sound but no picture. Pick a video clip." if audio else UNREADABLE)
    frames = _count(video)
    if not frames and not count:
        return probe(path, count=True)
    if not frames or not video.get("width") or not video.get("height"):
        raise MediaError("That file has no frames of video in it. Try exporting it again.")
    rotation = _rotation(video)
    w, h = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        w, h = h, w
    return Probe(
        width=w,
        height=h,
        fps=frame_rate(video.get("r_frame_rate"), video.get("avg_frame_rate")),
        frames=frames,
        has_audio=audio is not None,
        audio_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        rotation=rotation,
        primaries=video.get("color_primaries") or "",
        transfer=video.get("color_transfer") or "",
    )


@functools.cache
def _colour_managed_scale() -> bool:
    """ffmpeg 8's scale filter converts between colour spaces (out_primaries/out_transfer); older ones can't."""
    try:
        return "out_primaries" in subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=scale"], capture_output=True,
                                                 text=True).stdout
    except OSError:
        return False


def to_rec709(p: Probe) -> str:
    """ffmpeg filter that brings a wide-gamut or HDR clip into BT.709 ("" for everything else, or when this ffmpeg
    can't). The colours are converted; levels stay as shot, with no tone mapping (HLG is designed to read on an SDR
    screen as it is)."""
    wide = p.primaries == "bt2020" or p.transfer in ("arib-std-b67", "smpte2084")
    if not wide or not _colour_managed_scale():
        return ""
    return "scale=out_primaries=bt709:out_transfer=bt709:out_color_matrix=bt709"


def check_shot(path: Path) -> Probe:
    """A quick probe() of a file someone is adding as a shot: also refuses a still picture or a clip too short to work on."""
    p = probe(path, count=False)
    if p.frames == 1:
        raise MediaError("That's a still picture, not a video clip. Add a video here; pictures go in as references, "
                         "like the new person in Recast.")
    if p.frames < MIN_FRAMES:
        raise MediaError(f"That clip is too short to work on (only {p.frames} frames). Use one at least {MIN_FRAMES} frames long.")
    return p


def made_from(src: Probe, start: int, frames: int) -> str | None:
    """How much of the source a take covers when it is less than all of it: "the first 4.0 s of 12.3 s"."""
    start = max(0, int(start))
    if frames <= 0 or (start == 0 and frames >= src.frames):
        return None
    fps = float(src.fps)
    took, total = frames / fps, src.seconds
    if start == 0:
        return f"the first {took:.1f} s of {total:.1f} s"
    return f"{took:.1f} s from {start / fps:.1f} s in, of {total:.1f} s"


def ltx_frames(n: int) -> int:
    """Smallest 8k+1 frame count >= n."""
    return 8 * math.ceil(max(n - 1, 0) / 8) + 1


def ltx_frames_down(n: int) -> int:
    """Largest 8k+1 frame count <= n (at least 1): what LTX renders from a clip of n frames."""
    return 8 * (max(n - 1, 0) // 8) + 1


def work_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Working size for generation: scale so the long side is `long_side`, both sides multiples of 32."""
    scale = long_side / max(width, height)
    w = max(SPATIAL, round(width * scale / SPATIAL) * SPATIAL)
    h = max(SPATIAL, round(height * scale / SPATIAL) * SPATIAL)
    return w, h


def prepare_work_clip(src: Path, out: Path, start: int, frames: int, size: tuple[int, int], pre: str = "",
                      cover: bool = False, pad: tuple[int, int, int, int] | None = None, hold: int = 0) -> dict:
    """Cut `frames` frames from `start` and scale to the working size (multiples of 32), keeping audio if present.

    `pre` is an ffmpeg filter chain applied first (e.g. crop/delogo to clean a previz HUD). `cover` fills the size
    and centre-crops instead of stretching when the aspect differs. `pad` is (canvas width, canvas height, x, y): the
    scaled picture is placed there on a bigger canvas, whose spare edge is the picture's own edge repeated, so nothing
    is cropped. `hold` is how many of the `frames` repeat the last source frame (a short playblast is padded up to
    8k+1, never trimmed).
    """
    p = probe(src)
    w, h = size
    fps = p.fps
    hold = max(0, int(hold))
    take = frames - hold
    if take < 1:
        raise MediaError("A work clip needs at least one source frame.")
    fit = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}" if cover else f"scale={w}:{h}:flags=lanczos"
    if pad:
        W, H, ox, oy = (int(v) for v in pad)
        if (W, H) != (w, h):
            fit += (f",pad={W}:{H}:{ox}:{oy},"
                    f"fillborders=left={ox}:right={W - ox - w}:top={oy}:bottom={H - oy - h}:mode=smear")
        w, h = W, H
    tail = f",tpad=stop_mode=clone:stop={hold}" if hold else ""
    vf = (f"select='between(n\\,{start}\\,{start + take - 1})',setpts=N/({fps.numerator}/{fps.denominator})/TB,"
          + "".join(f"{x}," for x in (to_rec709(p), pre) if x) + f"{fit}{tail},format=yuv420p")
    dur = float(frames / fps)
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if p.has_audio:
        t0 = start / fps
        af = f"[0:a]atrim=start={float(t0):.6f}:duration={dur:.6f},asetpts=PTS-STARTPTS,aresample=48000,apad,atrim=end={dur:.6f}[a]"
    else:
        # LTX graphs encode audio too, so silent clips get a silent track.
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        af = f"[1:a]atrim=end={dur:.6f}[a]"
    args += ["-filter_complex", f"[0:v]{vf}[v];{af}", "-map", "[v]", "-map", "[a]", "-frames:v", str(frames),
             "-r", f"{fps.numerator}/{fps.denominator}", "-c:a", "aac", "-b:a", "320k",
             "-c:v", "libx264", "-crf", "2", "-preset", "veryfast", str(out)]
    subprocess.run(args, check=True)
    return {"fps": float(fps), "start_frame": start, "frames": frames, "size": [w, h], "source_size": [p.width, p.height]}


def build_extend_canvas(src: Path, out: Path, context: int, new: int, size: tuple[int, int]) -> dict:
    """Canvas for a tail Extend: the source's last `context` frames, then `new` copies of its last frame.

    Audio is the matching source audio followed by silence (or silence throughout if the source has none).
    Returns the timing the retake graph needs.
    """
    p = probe(src)
    if context + new != ltx_frames(context + new):
        raise ValueError(f"context + new must be 8k+1 frames, got {context + new}")
    start = max(p.frames - context, 0)
    fps = p.fps
    w, h = size
    total = context + new
    vf = (
        f"select='gte(n\\,{start})',setpts=N/({fps.numerator}/{fps.denominator})/TB,"
        + (f"{to_rec709(p)}," if to_rec709(p) else "")
        + f"scale={w}:{h}:flags=lanczos,tpad=stop_mode=clone:stop={new},format=yuv420p"
    )
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if p.has_audio:
        t0 = start / fps
        af = f"atrim=start={float(t0):.6f},asetpts=PTS-STARTPTS,aresample=48000,apad,atrim=end={float(total / fps):.6f}"
        args += ["-filter_complex", f"[0:v]{vf}[v];[0:a]{af}[a]", "-map", "[v]", "-map", "[a]"]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex", f"[0:v]{vf}[v]",
                 "-map", "[v]", "-map", "1:a", "-shortest"]
    args += ["-frames:v", str(total), "-r", f"{fps.numerator}/{fps.denominator}",
             "-c:v", "libx264", "-crf", "2", "-preset", "medium", "-c:a", "aac", "-b:a", "320k", str(out)]
    subprocess.run(args, check=True)
    return {
        "fps": float(fps),
        "context_frames": context,
        "new_frames": new,
        "total_frames": total,
        "start_time": context / float(fps),
        "end_time": total / float(fps),
        "source_start_frame": start,
        "size": [w, h],
    }


def timecode(path: Path) -> str | None:
    """The clip's start timecode (stream or container tag), e.g. "01:00:32:10", or None."""
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format_tags=timecode:stream_tags=timecode", "-of", "csv=p=0", str(path)])
    for line in out.splitlines():
        for part in line.split(","):
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}[:;]\d{2}", part.strip()):
                return part.strip()
    return None


def offset_timecode(tc: str, frames: int, fps: float) -> str:
    """tc + frames, counted at the nominal (rounded) frame rate; the drop-frame separator is kept."""
    if not frames:
        return tc
    sep = ";" if ";" in tc else ":"
    hh, mm, ss, ff = (int(x) for x in re.split("[:;]", tc))
    base = max(1, round(fps))
    total = ((hh * 60 + mm) * 60 + ss) * base + ff + frames
    ff, total = total % base, total // base
    ss, total = total % 60, total // 60
    mm, hh = total % 60, (total // 60) % 24
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


# --- trim and crop a source into a new shot ("Trim & crop", POST /api/shots/{id}/prepare) ------------------------------

# The crop shapes the UI offers (width / height). A crop with no aspect is free.
ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5, "2.39:1": 2.39, "4:3": 4 / 3}
MIN_CROP = 64  # px, the smallest side a crop may have (two LTX latents)


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def crop_box(width: int, height: int, crop: dict | None) -> tuple[int, int, int, int] | None:
    """The crop to cut from a `width`x`height` frame as (x, y, w, h), even numbers inside the frame, or None for the
    whole frame. `crop` is {aspect?, x, y, w, h} in the frame's pixels; with an aspect the box becomes the largest of
    that shape inside the one given, about the same centre. Raises ValueError (worded for the person) for a bad one."""
    if not crop:
        return None
    aspect = crop.get("aspect")
    if aspect is not None and aspect not in ASPECTS:
        raise ValueError(f"Unknown shape {aspect}. Use one of: {', '.join(ASPECTS)}.")
    try:
        x, y, w, h = (float(crop.get(k, d)) for k, d in (("x", 0), ("y", 0), ("w", width), ("h", height)))
    except (TypeError, ValueError):
        raise ValueError("Couldn't read the crop. Draw it again.") from None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("Couldn't read the crop. Draw it again.")
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    w, h = x1 - x0, y1 - y0
    if aspect is not None:
        r = ASPECTS[aspect]
        cx, cy = x0 + w / 2, y0 + h / 2
        w, h = (h * r, h) if w / max(h, 1e-6) > r else (w, w / r)
        x0, y0 = cx - w / 2, cy - h / 2
    cw, ch = _even(w), _even(h)
    if min(cw, ch) < MIN_CROP:
        raise ValueError(f"That crop is too small. Make it at least {MIN_CROP} px on each side.")
    cx0 = min(max(0, int(round(x0)) // 2 * 2), (width - cw) // 2 * 2)
    cy0 = min(max(0, int(round(y0)) // 2 * 2), (height - ch) // 2 * 2)
    if (cx0, cy0, cw, ch) == (0, 0, _even(width), _even(height)) and width % 2 == 0 and height % 2 == 0:
        return None
    return cx0, cy0, cw, ch


def trim_range(frames: int, in_frame, out_frame) -> tuple[int, int]:
    """In and out points (the out frame is kept, as an editor's out point) checked against a clip of `frames` frames:
    (start, count). Raises ValueError for a range no task could work on."""
    try:
        a = 0 if in_frame is None else int(in_frame)
        b = frames - 1 if out_frame is None else int(out_frame)
    except (TypeError, ValueError):
        raise ValueError("In and out must be frame numbers.") from None
    if a < 0 or b > frames - 1 or b < a:
        raise ValueError(f"In and out must be inside the clip (frames 0 to {frames - 1}), with out after in.")
    if b - a + 1 < MIN_FRAMES:
        raise ValueError(f"That's too short to work on. Keep at least {MIN_FRAMES} frames.")
    return a, b - a + 1


def _matrix(path: Path) -> str:
    """The clip's colour matrix tag ("bt709", "bt2020nc"; "" when untagged or unreadable)."""
    try:
        return _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_space", "-of",
                     "csv=p=0", str(path)]).strip().split(",")[0]
    except (subprocess.CalledProcessError, OSError):
        return ""


def trim_and_crop(src: Path, out: Path, start: int, frames: int, box: tuple[int, int, int, int] | None) -> dict:
    """Cut frames [start, start + frames) of `src` to `out` (a .mov), frame for frame, cropped to `box` (x, y, w, h).

    Re-encoded, so in and out land on exact frames: ProRes 422 HQ (10-bit 4:2:2), the quality a source deserves, with
    the sound trimmed to match as 24-bit PCM. The source's own frame rate, size (as displayed: a phone's rotation is
    applied), colour tags and start timecode (moved on by `start` frames) are kept. Returns what was written."""
    p = probe(src)
    fps = p.fps
    rate = f"{fps.numerator}/{fps.denominator}"
    vf = f"trim=start_frame={start}:end_frame={start + frames},setpts=N/({rate})/TB"
    if box:
        x, y, w, h = box
        vf += f",crop={w}:{h}:{x}:{y}"
    vf += ",setsar=1"
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1"]
    graph = f"[0:v:0]{vf}[v]"
    maps = ["-map", "[v]"]
    if p.has_audio:
        t0, dur = start / fps, frames / fps
        graph += f";[0:a:0]atrim=start={float(t0):.6f}:duration={float(dur):.6f},asetpts=PTS-STARTPTS[a]"
        maps += ["-map", "[a]", "-c:a", "pcm_s24le"]
    colour = []
    for flag, tag in (("-color_primaries", p.primaries), ("-color_trc", p.transfer), ("-colorspace", _matrix(src))):
        if tag and tag != "unknown":
            colour += [flag, tag]
    tc = timecode(src)
    stamp = ["-timecode", offset_timecode(tc, start, float(fps))] if tc else []
    subprocess.run(args + ["-filter_complex", graph] + maps + ["-frames:v", str(frames), "-r", rate, "-c:v", "prores_ks",
                   "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0", *colour, *stamp, str(out)],
                   check=True, capture_output=True)
    w, h = (box[2], box[3]) if box else (p.width, p.height)
    return {"start_frame": start, "frames": frames, "width": w, "height": h, "timecode": stamp[1] if stamp else None}


def assemble_jpeg_frames(folder: Path, out: Path, frames: int, fps: Fraction, size: tuple[int, int]) -> None:
    """Turn a `%04d.jpg` sequence (starting at 0000) into a ProRes 422 HQ .mov.

    Same codec prepare writes (`prores_ks` profile 3, 10-bit 4:2:2), at the sequence's own rate and size, so a
    browser previz lands in the library as a shot the rest of the app can open."""
    w, h = size
    rate = f"{fps.numerator}/{fps.denominator}"
    args = ["ffmpeg", "-v", "error", "-y", "-framerate", rate, "-start_number", "0", "-i", str(folder / "%04d.jpg"),
            "-frames:v", str(frames), "-r", rate, "-s", f"{w}x{h}", "-c:v", "prores_ks", "-profile:v", "3",
            "-pix_fmt", "yuv422p10le", "-vendor", "apl0", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", str(out)]
    subprocess.run(args, check=True, capture_output=True)


def stamp_timecode(folder: Path, tc: str) -> int:
    """Write a start timecode into every .mov/.mp4 deliverable in `folder` (stream copy, no re-encode). Returns count."""
    n = 0
    for f in sorted(folder.glob("*")):
        if f.suffix.lower() not in (".mov", ".mp4") or f.name in ("video.mp4", "before.mp4", "receipt.mp4"):
            continue
        tmp = f.with_name(f".tc_{f.name}")
        # Picture and sound only: an existing timecode (tmcd) track would otherwise be copied and win over the new one.
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-map", "0:v", "-map", "0:a?", "-c", "copy",
                            "-map_metadata", "0", "-metadata", f"timecode={tc}", "-timecode", tc,
                            "-movflags", "+faststart", str(tmp)], capture_output=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(f)
            n += 1
        else:
            tmp.unlink(missing_ok=True)
    return n
