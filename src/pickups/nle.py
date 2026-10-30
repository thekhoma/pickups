"""Export a project's approved takes to an editing timeline: FCP7 XML (xmeml, imported by DaVinci Resolve and Premiere
Pro) and a CMX3600 EDL.

Each take's ProRes master is placed at its source start timecode, so the sequence lines up with (and can be stacked on
top of) the original clips in an edit. Takes whose timecodes overlap go on separate video tracks. Takes without
timecode are laid end to end after the last timed one. Non-drop-frame timecode throughout (drop-frame sources are
placed by frame count, which is exact for the frames but labelled non-drop).

A take's notes travel at their place in the edit: in the XML as sequence markers, which DaVinci Resolve (21.1, tested)
imports as timeline markers and Premiere reads too (clip markers would show every note twice there; Resolve ignores
them). In the EDL they are Avid-style locators ("* LOC: <record timecode> BLUE <text>").

Notes left on a shot's original belong to the shot, and the shot reaches the edit as its approved take: unless the
original is approved itself, they are folded into each approved take of the shot, at the take's frame that shows the
same moment (fold_original_notes); notes outside the take stay behind. The marker comment and the locator say
"original".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape


@dataclass
class Clip:
    name: str          # take name shown in the NLE ("Harbour pier · T02 Remove")
    reel: str          # the original clip's name (for the conform)
    path: Path         # the take's ProRes master
    fps: Fraction
    frames: int
    width: int
    height: int
    start_tc: str | None  # source start timecode, e.g. "01:00:32:10"
    notes: list[tuple[int, str]] = field(default_factory=list)  # (frame from the clip's first frame, text)
    # Notes left on the shot's original, already at this clip's frames (fold_original_notes).
    original_notes: list[tuple[int, str]] = field(default_factory=list)


LOCATOR_COLOUR = "BLUE"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _one_line(text: str) -> str:
    return " ".join(_CONTROL.sub("", text).split())


def markers(c: Clip) -> list[tuple[int, str, bool]]:
    """(frame, text, from the original) for the clip's notes in frame order. A take's own note is kept inside the clip
    (past the end it sits on the last frame); a note folded in from the original is already placed, and one outside the
    clip is dropped."""
    out = [(max(0, min(c.frames - 1, int(f))), _one_line(t), False) for f, t in c.notes]
    out += [(int(f), _one_line(t), True) for f, t in c.original_notes if 0 <= int(f) < c.frames]
    return sorted((m for m in out if m[1]), key=lambda m: m[0])


def take_start(shot: dict, take_id: str) -> int | None:
    """Where a take starts in its shot's original, in the original's frames. Each take is cut from its input (the
    original or another take) from params.start_frame (none: the input's first frame), so the starts add up along
    input_version. None when that chain is broken (an input that isn't in the shot) or loops."""
    by_id = {v.get("id"): v for v in shot.get("versions") or []}
    start, seen, vid = 0, set(), take_id
    while vid != "orig":
        v = by_id.get(vid)
        if v is None or vid in seen:
            return None
        seen.add(vid)
        try:
            start += max(0, int((v.get("params") or {}).get("start_frame") or 0))
        except (TypeError, ValueError):
            return None
        vid = v.get("input_version") or "orig"
    return start


def fold_original_notes(shot: dict, take_id: str, frames: int, fps: float | Fraction | None = None) -> list[tuple[int, str]]:
    """The shot's original notes as (frame from the take's first frame, text) for one of its takes: each at the take's
    frame showing the same moment (1:1 for a take of the whole original at its frame rate), in frame order. Notes
    outside the take stay behind. Empty for the original itself, when the original is approved (its notes go to the
    edit with it), or when the take can't be placed (take_start)."""
    orig = next((v for v in shot.get("versions") or [] if v.get("id") == "orig"), None)
    if take_id == "orig" or orig is None or orig.get("status") == "approved":
        return []
    start = take_start(shot, take_id)
    if start is None:
        return []
    src_fps, take_fps = float(shot.get("fps") or 0), float(fps or 0)
    ratio = take_fps / src_fps if src_fps > 0 and take_fps > 0 and abs(take_fps - src_fps) > 0.001 else 1.0
    out = []
    for n in orig.get("notes") or []:
        try:
            f = int(n.get("frame") or 0) - start
        except (TypeError, ValueError):
            continue
        f = round(f * ratio) if ratio != 1.0 else f
        text = str(n.get("text") or "")
        if 0 <= f < frames and text.strip():
            out.append((f, text))
    return sorted(out, key=lambda m: m[0])


def _base(fps: Fraction) -> int:
    return max(1, round(float(fps)))


def _ntsc(fps: Fraction) -> bool:
    return abs(float(fps) - round(float(fps))) > 0.001


def tc_to_frames(tc: str, fps: Fraction) -> int:
    hh, mm, ss, ff = (int(x) for x in re.split("[:;]", tc))
    return ((hh * 60 + mm) * 60 + ss) * _base(fps) + ff


def frames_to_tc(n: int, fps: Fraction) -> str:
    b = _base(fps)
    ff, n = n % b, n // b
    ss, n = n % 60, n // 60
    mm, hh = n % 60, (n // 60) % 24
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def layout(clips: list[Clip]) -> tuple[int, list[tuple[Clip, int, int]]]:
    """Returns (sequence start frame, [(clip, record start offset, track index)])."""
    timed = sorted((c for c in clips if c.start_tc), key=lambda c: tc_to_frames(c.start_tc, c.fps))
    untimed = [c for c in clips if not c.start_tc]
    seq_start = tc_to_frames(timed[0].start_tc, timed[0].fps) if timed else tc_to_frames("01:00:00:00", clips[0].fps)
    placed: list[tuple[Clip, int, int]] = []
    track_ends: list[int] = []
    for c in timed:
        start = tc_to_frames(c.start_tc, c.fps) - seq_start
        track = next((t for t, end in enumerate(track_ends) if end <= start), None)
        if track is None:
            track_ends.append(0)
            track = len(track_ends) - 1
        track_ends[track] = start + c.frames
        placed.append((c, start, track))
    cursor = max([s + c.frames for c, s, _ in placed], default=0)
    for c in untimed:
        placed.append((c, cursor, 0))
        cursor += c.frames
    return seq_start, placed


def _rate(fps: Fraction) -> str:
    return f"<rate><timebase>{_base(fps)}</timebase><ntsc>{'TRUE' if _ntsc(fps) else 'FALSE'}</ntsc></rate>"


def _short(text: str) -> str:
    return text if len(text) <= 60 else text[:59].rstrip() + "…"


def _marker(name: str, comment: str, frame: int) -> str:
    """One FCP7 marker, children in xmeml's order: name, comment, in, out (-1: a point, not a range)."""
    return f"<marker><name>{escape(name)}</name><comment>{escape(comment)}</comment><in>{frame}</in><out>-1</out></marker>"


def sequence_markers(placed: list[tuple[Clip, int, int]]) -> list[tuple[int, str, str]]:
    """(frame from the sequence start, name, comment) for every note, in order; the comment says which take the note
    was left on ("<shot> · original" for one folded in from the original). Notes of stacked takes on the same frame
    share one marker (Resolve keeps one marker per frame); an original's note folded into two stacked takes is one."""
    at: dict[int, list[tuple[str, str]]] = {}
    for c, start, _ in placed:
        for f, t, original in markers(c):
            entry = (t, f"{c.reel} · original" if original else c.name)
            if entry not in at.setdefault(start + f, []):
                at[start + f].append(entry)
    out = []
    for frame in sorted(at):
        text = " / ".join(t for t, _ in at[frame])
        takes = ", ".join(dict.fromkeys(n for _, n in at[frame]))
        out.append((frame, _short(text), f"{text} ({takes})"))
    return out


def xmeml(clips: list[Clip], title: str) -> str:
    """FCP7 XML (xmeml v4): one sequence, one video track per overlap level (picture only), notes as sequence markers
    (Resolve 21 imports only those; Premiere reads them too, so clip markers would show every note twice there)."""
    fps = clips[0].fps
    seq_start, placed = layout(clips)
    total = max(s + c.frames for c, s, _ in placed)
    w, h = clips[0].width, clips[0].height
    tracks: dict[int, list[str]] = {}
    for i, (c, start, track) in enumerate(placed, start=1):
        url = "file://localhost" + quote(str(c.path.resolve()))
        src_in = tc_to_frames(c.start_tc, c.fps) if c.start_tc else 0
        tracks.setdefault(track, []).append(f"""
        <clipitem id="clipitem-{i}">
          <name>{escape(c.name)}</name><duration>{c.frames}</duration>{_rate(c.fps)}
          <start>{start}</start><end>{start + c.frames}</end><in>0</in><out>{c.frames}</out>
          <file id="file-{i}">
            <name>{escape(c.path.name)}</name><pathurl>{url}</pathurl>{_rate(c.fps)}<duration>{c.frames}</duration>
            <timecode>{_rate(c.fps)}<string>{frames_to_tc(src_in, c.fps)}</string><frame>{src_in}</frame>
              <displayformat>NDF</displayformat><reel><name>{escape(c.reel)}</name></reel></timecode>
            <media><video><samplecharacteristics>{_rate(c.fps)}<width>{c.width}</width><height>{c.height}</height></samplecharacteristics></video>
              <audio><channelcount>2</channelcount></audio></media>
          </file>
        </clipitem>""")
    video_tracks = "".join(f"\n      <track>{''.join(items)}\n      </track>" for _, items in sorted(tracks.items()))
    marks = "".join(f"\n    {_marker(n, cm, f)}" for f, n, cm in sequence_markers(placed))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
  <sequence id="sequence-1">
    <name>{escape(title)}</name><duration>{total}</duration>{_rate(fps)}
    <timecode>{_rate(fps)}<string>{frames_to_tc(seq_start, fps)}</string><frame>{seq_start}</frame><displayformat>NDF</displayformat></timecode>{marks}
    <media>
      <video>
        <format><samplecharacteristics>{_rate(fps)}<width>{w}</width><height>{h}</height><pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>{video_tracks}
      </video>
    </media>
  </sequence>
</xmeml>
"""


def edl(clips: list[Clip], title: str) -> str:
    """CMX3600: one event per take on V, source in/out = the take's own timecode, record = its sequence position.
    Notes follow their event as Avid locators at their record timecode."""
    fps = clips[0].fps
    seq_start, placed = layout(clips)
    lines = [f"TITLE: {title[:70]}", "FCM: NON-DROP FRAME", ""]
    for i, (c, start, _track) in enumerate(sorted(placed, key=lambda x: x[1]), start=1):
        src_in = tc_to_frames(c.start_tc, c.fps) if c.start_tc else 0
        reel = re.sub(r"[^A-Za-z0-9_]", "", c.reel)[:32] or "AX"
        lines.append(f"{i:03d}  {reel:<8} V     C        {frames_to_tc(src_in, c.fps)} {frames_to_tc(src_in + c.frames, c.fps)} "
                     f"{frames_to_tc(seq_start + start, fps)} {frames_to_tc(seq_start + start + c.frames, fps)}")
        lines.append(f"* FROM CLIP NAME: {c.path.name}")
        lines.append(f"* COMMENT: {c.name}")
        for f, t, original in markers(c):
            lines.append(f"* LOC: {frames_to_tc(seq_start + start + f, fps)} {LOCATOR_COLOUR:<7} {t}{' (original)' if original else ''}")
        lines.append("")
    return "\n".join(lines) + "\n"
