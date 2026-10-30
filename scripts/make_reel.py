"""Make a reel: a silent, loop-friendly before/after reel (at most 15 s) from the local example gallery, for social posts.

No title card: the first frame is a picture. The reel cold-opens on the first example already split by the app's wipe
line (the before on the right, the after on the left), with the wordmark in its band for 0.6 s; then the line finishes
the reveal. Each example after it runs 2.7 s (the before plays 0.6 s, the wipe takes 0.7 s, the after holds 1.4 s)
under its slate and title. A 1.5 s end card (the wordmark, the tagline, credits, #OpenSourceHiggsfield and an optional
--cta line such as a repo or a handle, passed at render time) closes the loop.

- 9x16: 1080×1920. A square picture over a soft blurred fill; all text in the top band (the bottom band stays clear for
  the caption and handle that TikTok and Reels lay over it), never on the picture. For remove and recast the crop and
  the moment follow what changed (the person), so the subject stays in frame; otherwise centred.
- 16x9: 1920×1080, full bleed, slate and title top left on a soft scrim.

H.264 yuv420p (BT.709), 30 fps, no audio (add --silent-audio for a silent AAC track), well under 20 MB. Only ffmpeg and
Pillow; text is drawn with the bundled fonts (see pickups.share). Repeatable: the same examples and arguments give
the same file.

Usage: uv run python scripts/make_reel.py [--format 9x16|16x9|both] [--ids a,b,c] [--out DIR] [--cta TEXT]
                                          [--focus ID=0.3] [--start ID=1.5] [--no-sheet] [--silent-audio]
Examples come from $PICKUPS_HOME/examples (public ones only; missing ones are skipped). Writes
DIR/pickups_reel_<format>.mp4 and a contact sheet DIR/pickups_reel_<format>_sheet.jpg; DIR defaults to
$PICKUPS_HOME/reels.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from pickups import media, share

HOME = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups")))
FPS = 30
SEG_F, END_F = 81, 45  # 2.7 s an example, 1.5 s for the end card
MAX_F = 15 * FPS  # the whole reel: at most 15.0 s
HOLD_F, WIPE_F = 18, 21  # in each example the before plays 0.6 s, the wipe takes 0.7 s, the after holds 1.4 s
# The cold open: the first example starts with the line already COLD_AT across, holds that split for 0.6 s under the
# wordmark (INTRO_F), then sweeps the rest of the way in 0.5 s.
COLD_AT, COLD_HOLD_F, COLD_WIPE_F = 0.42, 18, 15
INTRO_F, INTRO_FADE_F = 18, 4  # the wordmark fades out first, then the slate and title fade in: never both at once
MAX_EXAMPLES = (MAX_F - END_F) // SEG_F  # 5
DEFAULT_IDS = ("env-eilean-donan-snow-dusk", "rm-tre-cime-hiker", "pv-k03-hood-push", "env-fleshmarket-fog",
               "rc-doune-noblewoman")
FORMATS = {"9x16": (1080, 1920), "16x9": (1920, 1080)}
LOCAL = {"remove", "recast"}  # tasks that change one subject: the crop and the moment follow the change
TAGLINE = "Power your creativity, locally."
CREDITS = "Open source · runs on ComfyUI · LTX-2.5 at the core"
HASHTAG = "#OpenSourceHiggsfield"
MAX_MB = 20

BG = (13, 12, 11)
IVORY = share.IVORY
INK = (238, 233, 224, 217)  # ivory 85%: body text
SLATE = share.IVORY_70  # the app's .slate: mono caps, 0.08em tracking, ivory 70%
DIM = (238, 233, 224, 128)
TALLY = share.TALLY
UNLIT = (58, 42, 40, 255)
PLATE = (13, 12, 11, 150)
TRACK = 0.08
ARROW = "→"  # not in the bundled latin subsets: drawn as a vector


# --- the plan -------------------------------------------------------------------------------------------------------

@dataclass
class Part:
    kind: str  # "example" or "end"
    start: int  # first frame in the reel
    frames: int
    example: dict | None = None
    index: int = 0  # 1-based position among the examples
    src_start: float = 0.0  # seconds into the example's loops
    focus: float = 0.5  # horizontal centre of the crop, 0..1
    cold: bool = False  # the reel's first frames: opens mid-wipe under the wordmark

    @property
    def seconds(self) -> float:
        return self.frames / FPS


def gallery(folder: Path) -> list[dict]:
    """The public examples whose before and after loops are on disk."""
    path = folder / "examples.json"
    if not path.exists():
        return []
    return [e for e in json.loads(path.read_text())
            if e.get("public") and e.get("before") and e.get("after")
            and (folder / e["before"]).exists() and (folder / e["after"]).exists()]


def plan(available: list[dict], ids: list[str]) -> tuple[list[Part], dict[str, str]]:
    """Lay the reel out: up to five examples in the order asked for (the first one is the cold open), end card.

    Returns the parts and {id: why it was skipped}. Raises ValueError when none of the ids can be used.
    """
    by_id = {e["id"]: e for e in available}
    chosen: list[dict] = []
    skipped: dict[str, str] = {}
    for i in ids:
        if i not in by_id:
            skipped[i] = "not in the public gallery on this machine"
        elif by_id[i] in chosen:
            skipped[i] = "already in the reel"
        elif len(chosen) >= MAX_EXAMPLES:
            skipped[i] = f"only {MAX_EXAMPLES} fit in 15 s"
        else:
            chosen.append(by_id[i])
    if not chosen:
        raise ValueError("none of those examples are in the public gallery on this machine")
    parts: list[Part] = []
    for n, e in enumerate(chosen, 1):
        parts.append(Part("example", parts[-1].start + parts[-1].frames if parts else 0, SEG_F, e, n, cold=n == 1))
    parts.append(Part("end", parts[-1].start + parts[-1].frames, END_F))
    assert parts[-1].start + parts[-1].frames <= MAX_F
    return parts, skipped


def total_frames(parts: list[Part]) -> int:
    return sum(p.frames for p in parts)


def wipe(n: int, cold: bool = False) -> float:
    """How much of the after is revealed at frame n of an example, 0..1 (smoothstep, like share.py's wipe). The cold
    open holds at COLD_AT, then sweeps the rest of the way."""
    if cold:
        p = min(max((n - COLD_HOLD_F) / COLD_WIPE_F, 0.0), 1.0)
        return COLD_AT + (1 - COLD_AT) * p * p * (3 - 2 * p)
    p = min(max((n - HOLD_F) / WIPE_F, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def _wipe_expr(var: str, cold: bool = False) -> str:
    """wipe() as an ffmpeg expression of the frame number `var` (commas escaped for a quoted filter option)."""
    hold, frames = (COLD_HOLD_F, COLD_WIPE_F) if cold else (HOLD_F, WIPE_F)
    p = f"clip(({var}-{hold})/{frames}\\,0\\,1)"
    eased = f"({p}*{p}*(3-2*{p}))"
    return f"({COLD_AT}+{1 - COLD_AT:.2f}*{eased})" if cold else eased


def line_frames(cold: bool = False) -> tuple[int, int]:
    """The frames of an example the wipe line shows on (first, last): only while it moves, and from the cold open's
    first frame."""
    return (0, COLD_HOLD_F + COLD_WIPE_F - 1) if cold else (HOLD_F + 1, HOLD_F + WIPE_F - 1)


# --- where to look --------------------------------------------------------------------------------------------------

COLS, ROWS = 64, 36


def _gray(path: Path) -> list[Image.Image]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"fps={FPS},scale={COLS}:{ROWS}:flags=area,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    size = COLS * ROWS
    return [Image.frombytes("L", (COLS, ROWS), raw[i * size:(i + 1) * size]) for i in range(len(raw) // size)]


def choose(profiles: list[list[int]], local: bool) -> tuple[int, float]:
    """(first frame, horizontal focus) from per-frame column profiles of |after - before| at 30 fps.

    Anything that changes the whole picture: the middle of the loop, centred. Remove and recast change one subject:
    the window where it is most visible while the before plays and the wipe crosses it, and the crop on its centre.
    """
    n = len(profiles)
    mid = max(0, (n - SEG_F) // 2)
    if not local or n == 0:
        return mid, 0.5
    look = HOLD_F + WIPE_F  # the subject should be on screen until the line has passed it
    energy = [sum(c) for c in profiles]
    starts = range(0, max(n - SEG_F, 0) + 1)
    first = max(starts, key=lambda s: (sum(energy[s:s + look]), -abs(s - mid)))
    prof = [sum(profiles[i][x] for i in range(first, min(first + look, n))) for x in range(COLS)]
    floor = sorted(prof)[COLS // 2]  # compression noise is everywhere; the subject is what stands above it
    weight = [max(0, v - floor) for v in prof]
    if sum(weight) <= 0:
        return first, 0.5
    return first, round(sum((x + 0.5) * w for x, w in enumerate(weight)) / sum(weight) / COLS, 3)


def analyse(before: Path, after: Path, task: str) -> tuple[float, float]:
    """(seconds into the loops, focus) for one example."""
    b, a = _gray(before), _gray(after)
    profiles = [list(ImageChops.difference(x, y).resize((COLS, 1), Image.Resampling.BOX).tobytes()) for x, y in zip(b, a)]
    first, focus = choose(profiles, task in LOCAL)
    return first / FPS, focus


# --- type -----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _f(path: Path, size: int) -> ImageFont.FreeTypeFont:
    f = share._font(path, size)
    if not isinstance(f, ImageFont.FreeTypeFont):
        raise SystemExit(f"Couldn't load {path.name}: the reel needs the bundled fonts.")
    return f


def _runs(text: str) -> list[str]:
    out, cur = [], ""
    for ch in text:
        if ch == ARROW:
            if cur:
                out.append(cur)
            out.append(ARROW)
            cur = ""
        else:
            cur += ch
    return out + ([cur] if cur else [])


def width(text: str, font: ImageFont.FreeTypeFont, track: float = 0.0) -> float:
    """Advance width of text as _draw sets it (letter-spacing in em)."""
    gap = track * font.size
    w = 0.0
    for run in _runs(text):
        if run == ARROW:
            w += font.getlength("M") + gap
        elif gap:
            w += sum(font.getlength(c) + gap for c in run)
        else:
            w += font.getlength(run)
    return w - gap if text else 0.0


def _arrow(d: ImageDraw.ImageDraw, x: float, base: float, adv: float, size: int, fill) -> None:
    y = base - size * 0.36
    t = max(2, round(size * 0.075))
    x0, x1 = x + adv * 0.1, x + adv * 0.9
    d.line((x0, y, x1 - t * 0.5, y), fill=fill, width=t)
    h = adv * 0.34
    d.line((x1 - h, y - h, x1, y, x1 - h, y + h), fill=fill, width=t, joint="curve")


def _draw(d: ImageDraw.ImageDraw, x: float, base: float, text: str, font: ImageFont.FreeTypeFont, fill,
          track: float = 0.0) -> float:
    """Draw text left-aligned on a baseline. Returns its width."""
    gap = track * font.size
    cx = x
    for run in _runs(text):
        if run == ARROW:
            adv = font.getlength("M")
            _arrow(d, cx, base, adv, font.size, fill)
            cx += adv + gap
        elif gap:
            for c in run:
                d.text((cx, base), c, font=font, fill=fill, anchor="ls")
                cx += font.getlength(c) + gap
        else:
            d.text((cx, base), run, font=font, fill=fill, anchor="ls")
            cx += font.getlength(run)
    return cx - x - (gap if text else 0.0)


def cap(font: ImageFont.FreeTypeFont) -> int:
    """Cap height in px."""
    return -font.getbbox("H", anchor="ls")[1]


def balance(text: str, font: ImageFont.FreeTypeFont, max_w: float, track: float = 0.0, sep: str = " ") -> list[str]:
    """One line if it fits, else the two-line break (at `sep`) with the shortest longer line."""
    if width(text, font, track) <= max_w:
        return [text]
    words = text.split(sep)
    best = None
    for i in range(1, len(words)):
        lines = [sep.join(words[:i]).strip(), sep.join(words[i:]).strip()]
        w = max(width(s, font, track) for s in lines)
        if best is None or w < best[0]:
            best = (w, lines)
    return best[1] if best else [text]


def fit(text: str, path: Path, max_w: float, size: int, smallest: int) -> ImageFont.FreeTypeFont:
    """The largest size from `size` down to `smallest` at which text fits on one line."""
    for s in range(size, smallest - 1, -2):
        if width(text, _f(path, s)) <= max_w:
            return _f(path, s)
    return _f(path, smallest)


def slate(d: ImageDraw.ImageDraw, x: float, base: float, text: str, size: int, dot: bool = False, fill=SLATE) -> float:
    """The burn-in slate: mono caps, tracked, an optional lit tally dot in front."""
    f = _f(share.MONO, size)
    x0 = x
    if dot:
        r = size * 0.2
        cy = base - cap(f) / 2
        d.ellipse((x, cy - r, x + 2 * r, cy + r), fill=TALLY)
        x += 2 * r + size * 0.6
    return x - x0 + _draw(d, x, base, text.upper(), f, fill, TRACK)


def tag(text: str, size: int) -> Image.Image:
    """BEFORE / AFTER: a small mono plate that rides on its own picture, so the wipe swaps it."""
    f = _f(share.MONO, size)
    text = text.upper()
    px, py = round(size * 0.6), round(size * 0.5)
    w = round(width(text, f, TRACK)) + 2 * px
    h = cap(f) + 2 * py
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=max(4, size // 4), fill=PLATE)
    _draw(d, px, py + cap(f), text, f, SLATE, TRACK)
    return img


@lru_cache(maxsize=None)
def wordmark(size: int, lit: bool = True) -> tuple[Image.Image, int, int]:
    """'Pickups' in Instrument Serif with the tally dot, as the app's top bar sets it (a dot just above the baseline,
    a hair after the s), with the dot a touch smaller at display sizes where the serif's strokes are thinner.
    Returns the sprite, the x of the word's left edge and the y of the baseline in it."""
    f = _f(share.SERIF, size)
    word = round(f.getlength("Pickups"))
    dot, gap, lift = round(size * 0.17), round(size * 0.07), round(size * 0.09)
    pad = dot * 2  # room for the bloom
    asc, desc = f.getmetrics()
    img = Image.new("RGBA", (word + gap + dot + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    base = pad + asc
    ImageDraw.Draw(img).text((pad, base), "Pickups", font=f, fill=IVORY, anchor="ls")
    box = (pad + word + gap, base - lift - dot, pad + word + gap + dot, base - lift)
    if lit:  # the app's 6 px bloom, scaled: a soft red halo under the dot
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        g = dot * 0.25
        ImageDraw.Draw(glow).ellipse((box[0] - g, box[1] - g, box[2] + g, box[3] + g), fill=(245, 48, 30, 90))
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(dot * 0.4)))
    ImageDraw.Draw(img).ellipse(box, fill=TALLY if lit else UNLIT)
    return img, pad, base


def _sprite(lines: list[str], font: ImageFont.FreeTypeFont, fill, leading: float, track: float = 0.0,
            caps: bool = False) -> tuple[Image.Image, int]:
    """Centred lines of text as one sprite. Returns it and the first baseline's y in it."""
    lines = [s.upper() for s in lines] if caps else lines
    asc, desc = font.getmetrics()
    w = round(max(width(s, font, track) for s in lines)) + 4
    step = round(font.size * leading)
    img = Image.new("RGBA", (w, asc + desc + step * (len(lines) - 1)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        _draw(d, (w - width(s, font, track)) / 2, asc + i * step, s, font, fill, track)
    return img, asc


def _fade(img: Image.Image, a: float) -> Image.Image:
    if a >= 1:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: round(v * max(a, 0.0))))
    return out


def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


# --- layout ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    fmt: str
    w: int
    h: int
    panel: tuple[int, int, int, int]  # x, y, w, h of the picture
    margin: int
    mark: int  # wordmark size on the cards
    mark_base: int  # its baseline (the same on both cards)


def layout(fmt: str) -> Layout:
    if fmt == "9x16":
        return Layout(fmt, 1080, 1920, (0, 420, 1080, 1080), 72, 200, 780)
    if fmt == "16x9":
        return Layout(fmt, 1920, 1080, (0, 0, 1920, 1080), 80, 180, 380)
    raise ValueError("format must be 9x16 or 16x9")


def _paste_centred(canvas: Image.Image, sprite: Image.Image, base_in: int, base_at: float) -> None:
    canvas.alpha_composite(sprite, (round(canvas.width / 2 - sprite.width / 2), round(base_at - base_in)))


def card(lay: Layout, kind: str, n: int, cta: str = "") -> Image.Image:
    """Frame n of the end card: the wordmark, the tagline, credits, the hashtag and the optional call to action, in
    that order, each easing in a beat after the one before."""
    if kind != "end":
        raise ValueError("the reel has an end card only: it opens on a picture")
    img = Image.new("RGBA", (lay.w, lay.h), BG + (255,))
    mark, mx, mb = wordmark(lay.mark, True)
    # centre the word (not the dot) so the wordmark sits optically centred
    word_w = _f(share.SERIF, lay.mark).getlength("Pickups")
    img.alpha_composite(mark, (round(lay.w / 2 - word_w / 2 - mx), lay.mark_base - mb))
    tall = lay.fmt == "9x16"
    y = lay.mark_base
    rows = [  # (font, text, fill, leading, gap above the first baseline, frame it starts easing in, separator)
        (_f(share.SERIF, 64 if tall else 60), TAGLINE, INK, 1.1, 120 if tall else 110, 2, " "),
        (_f(share.MONO, 32 if tall else 30), CREDITS, SLATE, 1.5, 120 if tall else 100, 5, " · "),
        (_f(share.SERIF, 96 if tall else 84), HASHTAG, IVORY, 1.0, 150 if tall else 125, 9, " "),
    ]
    if cta:
        rows.append((_f(share.MONO, 32 if tall else 30), cta, INK, 1.5, 110 if tall else 95, 12, " "))
    prev = None
    for f, text, fill, leading, gap, at, sep in rows:
        lines = balance(text, f, lay.w - 2 * lay.margin, sep=sep)
        spr, b = _sprite(lines, f, fill, leading)
        if prev is not None:
            y += round(prev[0].size * prev[1]) * (prev[2] - 1)  # past the previous row's extra lines
        y += gap
        q = _ease((n - at) / 10)
        _paste_centred(img, _fade(spr, q), b, y + 12 * (1 - q))
        prev = (f, leading, len(lines))
    return img.convert("RGB")


def _scrim(w: int, h: int, alpha) -> Image.Image:
    """A soft dark gradient, alpha(x, y) on 0..1 coordinates, computed small and scaled up."""
    sw, sh = 96, round(96 * h / w)
    m = Image.new("L", (sw, sh))
    m.putdata([round(255 * min(max(alpha((x + 0.5) / sw, (y + 0.5) / sh), 0.0), 1.0)) for y in range(sh) for x in range(sw)])
    out = Image.new("RGBA", (w, h), BG + (0,))
    out.putalpha(m.resize((w, h), Image.Resampling.BICUBIC))
    return out


def _shadowed(lay: Layout, fg: Image.Image) -> Image.Image:
    """Type with a soft shadow under it, so it holds on bright footage (fog, snow, sky) without a heavier scrim."""
    out = Image.new("RGBA", (lay.w, lay.h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (lay.w, lay.h), BG + (0,))
    shadow.putalpha(fg.getchannel("A").filter(ImageFilter.GaussianBlur(9)).point(lambda v: round(v * 0.75)))
    out.alpha_composite(shadow)
    out.alpha_composite(fg)
    return out


def backdrop(lay: Layout) -> Image.Image:
    """What sits still under the type of every example: the darkened bands (9x16) or the soft scrim (16x9)."""
    img = Image.new("RGBA", (lay.w, lay.h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    px, py, pw, ph = lay.panel
    if lay.fmt == "9x16":
        d.rectangle((0, 0, lay.w, py - 1), fill=BG + (165,))  # quiet the blurred fill so the type reads
        d.rectangle((0, py + ph, lay.w, lay.h), fill=BG + (165,))
        d.line((0, py - 1, lay.w, py - 1), fill=(238, 233, 224, 30))
        d.line((0, py + ph, lay.w, py + ph), fill=(238, 233, 224, 30))
    else:
        def shade(x: float, y: float) -> float:
            top = max(0.0, 1 - y / 0.38) ** 1.4 * min(1.0, max(0.0, (0.8 - x) / 0.32)) * 0.82
            low = max(0.0, (y - 0.78) / 0.22) ** 1.4 * min(1.0, max(0.0, (x - 0.6) / 0.22)) * 0.66
            return 1 - (1 - top) * (1 - low)
        img.alpha_composite(_scrim(lay.w, lay.h, shade))
    return img


def captions(lay: Layout, ex: dict, index: int, count: int) -> Image.Image:
    """An example's type: slate, title, counter and a small wordmark. In 9x16 all of it sits in the top band, the
    wordmark beside the counter, so the bottom band stays clear for the platform's own caption."""
    fg = Image.new("RGBA", (lay.w, lay.h), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    px, py, pw, ph = lay.panel
    m = lay.margin
    title = ex.get("title") or ""
    if lay.fmt == "9x16":
        tf = fit(title, share.SERIF, lay.w - 2 * m, 112, 72)
        tb = py - 70
        _draw(d, m, tb, title, tf, IVORY)
        sb = tb - cap(tf) - 46
        slate(d, m, sb, ex.get("slate") or "", 30, dot=True)
        cf = _f(share.MONO, 30)
        counter = f"{index:02d}/{count:02d}"
        cx = lay.w - m - width(counter, cf, TRACK)
        _draw(d, cx, sb, counter, cf, DIM, TRACK)
        size = 46
        mark, mx, mb = wordmark(size)
        word_w = _f(share.SERIF, size).getlength("Pickups")
        fg.alpha_composite(mark, (round(cx - 30 - word_w - size * 0.24 - mx), sb - mb))
        return fg
    sb = m + 18
    slate(d, m, sb, ex.get("slate") or "", 30, dot=True)
    tf = fit(title, share.SERIF, lay.w * 0.55, 104, 72)
    _draw(d, m, sb + 34 + cap(tf), title, tf, IVORY)
    mark, mx, mb = wordmark(56)
    word_w = _f(share.SERIF, 56).getlength("Pickups")
    fg.alpha_composite(mark, (round(lay.w - m - word_w - 56 * 0.24 - mx), lay.h - m + 6 - mb))
    return _shadowed(lay, fg)


def overlay(lay: Layout, ex: dict, index: int, count: int) -> Image.Image:
    """Everything that sits still over an example: the backdrop and its type."""
    img = backdrop(lay)
    img.alpha_composite(captions(lay, ex, index, count))
    return img


def intro(lay: Layout) -> Image.Image:
    """The cold open's first 0.6 s: the wordmark where the example's title will be (9x16: centred in the top band;
    16x9: top left), lit, as the app's top bar while a take renders."""
    fg = Image.new("RGBA", (lay.w, lay.h), (0, 0, 0, 0))
    size = 150 if lay.fmt == "9x16" else 140
    mark, mx, mb = wordmark(size, True)
    word_w = _f(share.SERIF, size).getlength("Pickups")
    if lay.fmt == "9x16":
        x, base = lay.w / 2 - word_w / 2, lay.panel[1] - 150
    else:
        x, base = lay.margin, lay.margin + 18 + cap(_f(share.SERIF, size))
    fg.alpha_composite(mark, (round(x - mx), round(base - mb)))
    return fg if lay.fmt == "9x16" else _shadowed(lay, fg)


def tag_spots(lay: Layout, before: Image.Image, after: Image.Image) -> tuple[tuple[int, int], tuple[int, int]]:
    """Both tags top right of the picture: whatever shows in that corner is what the tag says."""
    px, py, pw, ph = lay.panel
    inset = 28 if lay.fmt == "9x16" else lay.margin
    return ((pw - inset - before.width, inset), (pw - inset - after.width, inset))


# --- render ---------------------------------------------------------------------------------------------------------

def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


LOSSLESS = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
TO_709 = "scale=out_color_matrix=bt709:out_range=tv"


def render_card(lay: Layout, part: Part, tmp: Path, out: Path, cta: str = "") -> None:
    for n in range(part.frames):
        card(lay, part.kind, n, cta).save(tmp / f"{part.kind}_{n:03d}.png")
    _ffmpeg(["-framerate", str(FPS), "-i", str(tmp / f"{part.kind}_%03d.png"), "-vf", f"{TO_709},format=yuv420p",
             "-frames:v", str(part.frames), *LOSSLESS, str(out)])


def example_graph(lay: Layout, focus: float, spots: tuple[tuple[int, int], tuple[int, int]], cold: bool = False) -> str:
    """Inputs: 0 before, 1 after, 2 / 3 the BEFORE / AFTER tags, 4 the wipe line, 5 the backdrop, 6 the type; the cold
    open adds 7, the intro wordmark, which gives way to the type at INTRO_F."""
    px, py, pw, ph = lay.panel
    src = (f"fps={FPS},crop=w='min(iw\\,ih*{pw}/{ph})':h='min(ih\\,iw*{ph}/{pw})'"
           f":x='clip(iw*{focus:.3f}-ow/2\\,0\\,iw-ow)':y='(ih-oh)/2',"
           f"scale={pw}:{ph}:flags=lanczos,setsar=1,format=yuv420p,"
           f"tpad=stop_mode=clone:stop_duration=4,trim=end_frame={SEG_F},setpts=PTS-STARTPTS")
    (bx, by), (ax, ay) = spots
    g = [f"[0:v]{src}[b0]", f"[1:v]{src}[a0]",
         f"[2:v]format=rgba,{TO_709},format=yuva420p[tb]", f"[3:v]format=rgba,{TO_709},format=yuva420p[ta]",
         f"[b0][tb]overlay={bx}:{by}[b]", f"[a0][ta]overlay={ax}:{ay}[a]",
         f"[b][a]blend=all_expr='if(lt(X\\,W*{_wipe_expr('N', cold)})\\,B\\,A)'[w]"]
    if lay.fmt == "9x16":  # the picture as a square over a soft blurred, darkened copy of itself
        g += ["[w]split[p][q]",
              f"[q]scale=192:192:flags=area,crop=108:192,gblur=sigma=6,scale={lay.w}:{lay.h}:flags=bicubic,setsar=1[bg]",
              f"[bg][p]overlay={px}:{py}[c]"]
    else:
        g += ["[w]null[c]"]
    x = f"{px}+{pw}*{_wipe_expr('n', cold)}-3"
    first, last = line_frames(cold)
    out_at, fade_s = INTRO_F / FPS, INTRO_FADE_F / FPS
    in_at, in_s = (INTRO_F + INTRO_FADE_F) / FPS, 5 / FPS
    g += [f"[c][4:v]overlay=x='if(between(n\\,{first}\\,{last})\\,{x}\\,-100)':y={py}:eval=frame[d]",
          f"[5:v]format=rgba,{TO_709},format=yuva420p[k]",
          f"[6:v]format=rgba,{TO_709},format=yuva420p" + (f",fade=t=in:st={in_at:.3f}:d={in_s:.3f}:alpha=1" if cold else "") + "[o]",
          "[d][k]overlay=0:0[e]"]
    if cold:
        g += [f"[7:v]format=rgba,{TO_709},format=yuva420p,fade=t=out:st={out_at:.3f}:d={fade_s:.3f}:alpha=1[i]",
              "[e][o]overlay=0:0[f]", "[f][i]overlay=0:0,format=yuv420p[v]"]
    else:
        g += ["[e][o]overlay=0:0,format=yuv420p[v]"]
    return ";".join(g)


def wipe_line(height: int, path: Path) -> Path:
    """The app's signature line: 2 px ivory with a dark edge either side, so it holds over any footage."""
    im = Image.new("RGBA", (6, height), (13, 12, 11, 150))
    ImageDraw.Draw(im).rectangle((2, 0, 3, height), fill=(238, 233, 224, 245))
    im.save(path)
    return path


def render_example(lay: Layout, part: Part, count: int, folder: Path, tmp: Path, out: Path) -> None:
    ex = part.example or {}
    size = 26 if lay.fmt == "9x16" else 28
    tb, ta = tag("Before", size), tag("After", size)
    k = f"{part.index:02d}"
    tb.save(tmp / f"tag_b_{k}.png")
    ta.save(tmp / f"tag_a_{k}.png")
    backdrop(lay).save(tmp / f"back_{k}.png")
    captions(lay, ex, part.index, count).save(tmp / f"type_{k}.png")
    line = wipe_line(lay.panel[3], tmp / "line.png")
    graph = example_graph(lay, part.focus, tag_spots(lay, tb, ta), part.cold)
    ss = f"{part.src_start:.3f}"
    # Stills that fade are looped for the length of the example; the rest are single frames, held.
    still = ["-framerate", str(FPS), "-loop", "1", "-t", f"{part.frames / FPS:.3f}", "-i"]
    ins = ["-ss", ss, "-i", str(folder / ex["before"]), "-ss", ss, "-i", str(folder / ex["after"]),
           "-i", str(tmp / f"tag_b_{k}.png"), "-i", str(tmp / f"tag_a_{k}.png"), "-i", str(line),
           "-i", str(tmp / f"back_{k}.png"), *(still if part.cold else ["-i"]), str(tmp / f"type_{k}.png")]
    if part.cold:
        intro(lay).save(tmp / "intro.png")
        ins += [*still, str(tmp / "intro.png")]
    _ffmpeg([*ins, "-filter_complex", graph, "-map", "[v]", "-an", "-frames:v", str(part.frames), *LOSSLESS, str(out)])


def join(clips: list[Path], out: Path, seconds: float, silent_audio: bool) -> None:
    ins = [a for c in clips for a in ("-i", str(c))]
    tags = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"  # tag BT.709 end to end
    graph = "".join(f"[{i}:v]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,setsar=1,format=yuv420p,{tags}[v]"
    audio = ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"] if silent_audio else []
    amap = ["-map", f"{len(clips)}:a", "-c:a", "aac", "-b:a", "64k"] if silent_audio else ["-an"]
    _ffmpeg([*ins, *audio, "-filter_complex", graph, "-map", "[v]", *amap,
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "9M", "-bufsize", "18M",
             "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", str(2 * FPS), "-threads", "8",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
             "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
             "-movflags", "+faststart", str(out)])


def contact_sheet(video: Path, fmt: str, out: Path) -> Path:
    """Every half second, tiled: 9x16 as 10×3, 16x9 as 6×5."""
    tile, scale = ("10x3", "216:384") if fmt == "9x16" else ("6x5", "384:216")
    _ffmpeg(["-i", str(video), "-vf", f"select='not(mod(n\\,15))',scale={scale}:flags=area,"
             f"tile={tile}:padding=6:margin=6:color=0x0D0C0B", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "3",
             str(out)])
    return out


def render(parts: list[Part], fmt: str, folder: Path, out: Path, silent_audio: bool = False, cta: str = "") -> Path:
    lay = layout(fmt)
    count = sum(1 for p in parts if p.kind == "example")
    with tempfile.TemporaryDirectory(prefix="pickups-reel-") as t:
        tmp = Path(t)
        clips = []
        for i, p in enumerate(parts):
            clip = tmp / f"part_{i:02d}.mp4"
            if p.kind == "example":
                render_example(lay, p, count, folder, tmp, clip)
            else:
                render_card(lay, p, tmp, clip, cta)
            clips.append(clip)
        join(clips, out, total_frames(parts) / FPS, silent_audio)
    return out


def _pairs(items: list[str], what: str) -> dict[str, float]:
    out = {}
    for s in items:
        k, _, v = s.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--{what} takes ID=NUMBER, not {s!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render a 15-second before/after reel from the example gallery.")
    ap.add_argument("--format", choices=["9x16", "16x9", "both"], default="9x16")
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS), help="comma-separated example ids, in order")
    ap.add_argument("--examples", type=Path, default=HOME / "examples", help="the example gallery folder")
    ap.add_argument("--out", type=Path, default=HOME / "reels", help="output folder")
    ap.add_argument("--focus", action="append", default=[], metavar="ID=X", help="crop centre, 0 (left) to 1 (right)")
    ap.add_argument("--start", action="append", default=[], metavar="ID=S", help="seconds into the example's loops")
    ap.add_argument("--no-sheet", action="store_true", help="skip the contact sheet")
    ap.add_argument("--silent-audio", action="store_true", help="add a silent AAC track for platforms that want one")
    ap.add_argument("--cta", default="", help="a last line on the end card, e.g. where to get it (a repo or a handle)")
    a = ap.parse_args(argv)
    ids = [s.strip() for s in a.ids.split(",") if s.strip()]
    try:
        parts, skipped = plan(gallery(a.examples), ids)
    except ValueError as e:
        raise SystemExit(f"No reel: {e}.")
    for i, why in skipped.items():
        print(f"skipped {i}: {why}")
    focus, start = _pairs(a.focus, "focus"), _pairs(a.start, "start")
    for p in parts:
        if p.kind != "example" or p.example is None:
            continue
        e = p.example
        p.src_start, p.focus = analyse(a.examples / e["before"], a.examples / e["after"], e.get("task", ""))
        p.src_start = start.get(e["id"], p.src_start)
        p.focus = min(max(focus.get(e["id"], p.focus), 0.0), 1.0)
        print(f"{p.index}. {e['id']}: from {p.src_start:.2f} s, crop centred at {p.focus:.2f}")
    a.out.mkdir(parents=True, exist_ok=True)
    for fmt in (["9x16", "16x9"] if a.format == "both" else [a.format]):
        out = render(parts, fmt, a.examples, a.out / f"pickups_reel_{fmt}.mp4", a.silent_audio, a.cta.strip())
        pr = media.probe(out)
        mb = out.stat().st_size / 1e6
        print(f"{out.name}: {pr.width}×{pr.height}, {pr.frames / float(pr.fps):.2f} s, {mb:.1f} MB"
              + ("" if mb < MAX_MB else f" (over {MAX_MB} MB: too big to post everywhere)"))
        if not a.no_sheet:
            print(f"{contact_sheet(out, fmt, a.out / f'pickups_reel_{fmt}_sheet.jpg').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
