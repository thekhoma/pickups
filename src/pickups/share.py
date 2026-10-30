"""Share videos: a take's before/after as a social-ready MP4 with the slate burned in.

- "9x16": 1080×1920, before on top and after below, labelled; for Reels, Shorts and TikTok.
- "1x1": 1080×1080, an animated wipe from before to after (the app's signature wipe line), for feeds.
The slate carries what was done, by which models, how much of the picture is still the original footage (from the
take's receipt), and "Made locally with Pickups". Text is drawn with Pillow (bundled OFL fonts) as a transparent
overlay, so any ffmpeg build works. At most 10 s.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).parent / "assets" / "fonts"
MONO = FONTS / "geist-mono-latin-400-normal.woff"
SERIF = FONTS / "instrument-serif-latin-400-normal.woff"
SERIF_I = FONTS / "instrument-serif-latin-400-italic.woff"
IVORY = (238, 233, 224, 255)
IVORY_70 = (238, 233, 224, 178)
TALLY = (245, 48, 30, 255)
MAX_S = 10.0


def _font(path: Path, size: int):
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _label(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 26, dot: bool = False) -> None:
    """The burn-in slate: mono caps on a soft dark plate, optional tally dot."""
    f = _font(MONO, size)
    text = text.upper()
    x, y = xy
    l, t, r, b = d.textbbox((0, 0), text, font=f, spacing=0)
    pad = size // 2
    dx = (size // 2 + 10) if dot else 0
    d.rounded_rectangle((x, y, x + r + pad * 2 + dx, y + b + pad), radius=6, fill=(13, 12, 11, 150))
    if dot:
        cy = y + (b + pad) // 2
        d.ellipse((x + pad, cy - size // 5, x + pad + size // 2.5, cy + size // 5), fill=TALLY)
    d.text((x + pad + dx, y + pad // 2), text, font=f, fill=IVORY_70)


def stat_line(receipt: dict | None) -> str:
    """What the slate says about the picture: the share of original pixels when most of it is untouched; for an
    upscale (Finish), the factor and the AI share, since upscaled pixels are your footage, not new content. Shares are
    rounded as the app and the receipt sheet round them (receipts.percent_texts: "<1%", never 100% beside another share)."""
    from pickups.receipts import percent_texts, percents, scale_text  # lazily: receipts imports the store

    r = receipt or {}
    t, p = percent_texts(r), percents(r)
    if not t:
        return ""
    if float(r.get("upscaled") or 0) > 0:
        return f"Upscaled{scale_text(r)} · {t[3]} AI"
    if r.get("original") is not None and float(r.get("original") or 0) < 100 and p[3] < 50:
        return f"{t[0]} ORIGINAL PIXELS"
    return ""


def slate_overlay(w: int, h: int, fmt: str, task: str, models: list[str], receipt: dict | None, title: str, out: Path) -> Path:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = w / 1080
    m = int(36 * s)
    stat = stat_line(receipt)
    what = " · ".join(x for x in (task, stat) if x)
    if fmt == "9x16":
        half = h // 2
        _label(d, (m, m + int(120 * s)), "Before", int(26 * s))
        _label(d, (m, half + m), f"After · {what}" if what else "After", int(26 * s), dot=True)
        # header: the title in the serif, one italic word avoided on purpose (titles are user text)
        f = _font(SERIF, int(64 * s))
        d.text((m, int(28 * s)), title, font=f, fill=IVORY)
        foot = " · ".join(models[:3])
        _label(d, (m, h - m - int(120 * s)), foot or "Open models", int(22 * s))
        _label(d, (m, h - m - int(60 * s)), "Made locally with Pickups", int(22 * s))
    else:
        _label(d, (m, m), f"{what}" if what else task, int(24 * s), dot=True)
        _label(d, (m, h - m - int(52 * s)), "Made locally with Pickups · " + " · ".join(models[:2]), int(20 * s))
    img.save(out)
    return out


def _cover(w: int, h: int) -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1"


def make(before: Path, after: Path, out: Path, fmt: str, task: str, models: list[str], receipt: dict | None, title: str) -> Path:
    """Render the share MP4 (H.264, AAC-less, faststart)."""
    if fmt not in ("9x16", "1x1"):
        raise ValueError("format must be 9x16 or 1x1")
    w, h = (1080, 1920) if fmt == "9x16" else (1080, 1080)
    overlay = slate_overlay(w, h, fmt, task, models, receipt, title, out.with_suffix(".png"))
    line = out.with_name(out.stem + "_line.png")
    Image.new("RGBA", (6, h), (13, 12, 11, 150)).save(line)  # dark edges
    im = Image.open(line)
    ImageDraw.Draw(im).rectangle((2, 0, 3, h), fill=(238, 233, 224, 245))  # the 2 px ivory wipe line
    im.save(line)
    common = ["ffmpeg", "-v", "error", "-y", "-t", f"{MAX_S}", "-i", str(before), "-t", f"{MAX_S}", "-i", str(after), "-i", str(overlay),
              "-loop", "1", "-i", str(line)]
    if fmt == "9x16":
        graph = (f"[0:v]{_cover(w, h // 2)}[b];[1:v]{_cover(w, h // 2)}[a];[b][a]vstack=2,"
                 f"drawbox=x=0:y={h // 2 - 1}:w={w}:h=2:color=0xEEE9E0@0.9:t=fill[v];[v][2:v]overlay=0:0,format=yuv420p")
    else:
        # wipe: hold on before 0.6 s, sweep to after over 1.4 s (smoothstep), hold; ivory 2 px line with a dark edge
        p = "clip((T-0.6)/1.4\\,0\\,1)"
        ease = f"({p}*{p}*(3-2*{p}))"
        x = f"{w}*{ease.replace('T', 't')}-3"
        graph = (f"[0:v]{_cover(w, h)},format=yuv420p[b];[1:v]{_cover(w, h)},format=yuv420p[a];"
                 f"[b][a]blend=all_expr='if(lt(X\\,W*{ease})\\,B\\,A)'[wv];"
                 f"[wv][3:v]overlay=x='if(between(t\\,0.6\\,2.0)\\,{x}\\,-10)':y=0:shortest=1:eval=frame[v];"
                 f"[v][2:v]overlay=0:0,format=yuv420p")
    subprocess.run(common + ["-filter_complex", graph, "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                             "-movflags", "+faststart", "-shortest", str(out)], check=True)
    overlay.unlink(missing_ok=True)
    line.unlink(missing_ok=True)
    return out
