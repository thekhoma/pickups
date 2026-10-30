"""Character sheets: one person from the front, from behind and close up, side by side in one 16:9 picture.

A character made from a single front picture renders badly from behind: SCAIL-2 has to guess their back. It takes extra
reference views (a back view, a close-up) beside the main one, and Qwen-Image 2.1 reads a sheet like this well for
previz look frames. The layout is the one that has worked on real jobs: full-length FRONT | full-length BACK | head
CLOSE-UP, the same person and outfit, on a plain light-grey studio backdrop, three equal panels with thin white gaps.

From one picture (a photo, a selfie, an AI still) or a few words. Words: FLUX 3 klein makes the front view with Recast's
studio still template. A picture that isn't already a clean full-length front view on a plain light backdrop is first
turned into one by the frame editor. Then the editor makes the back view and the close-up from the front view, every
panel with the same seed. The editor is graphs.image_edit, so Setup's licence choice applies: Qwen-Image 2.1, or
Qwen-Image-Edit 2511 when commercial-safe models are asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pickups import composite, graphs, images, reference
from pickups.app.jobs import Stage

KEEP = ("Keep the same person, face, hair, clothing, colours, materials, details and proportions exactly. Plain light-grey "
        "seamless studio backdrop, soft even front light, a faint contact shadow.")
FRONT = ("Show the same person in a full-length front view on a plain light-grey seamless studio backdrop: standing straight "
         "and facing the camera, arms slightly away from the body, the whole figure from the top of the head to the shoes "
         "inside the frame. " + KEEP)
BACK = ("Show the same person from directly behind: a full-length back view, standing straight with arms slightly away from "
        "the body, the whole figure from the top of the head to the shoes inside the frame. " + KEEP)
FACE = ("A close-up of the same person's head and shoulders, facing the camera, the face filling most "
        "of the frame. " + KEEP)
FACE_GEAR = ("A close-up of the same person's head and shoulders, facing the camera, the face (or helmet and visor) filling most "
             "of the frame. " + KEEP)


def face_instruction(description: str | None) -> str:
    """The close-up line. Helmet and visor wording only when the description names a helmet, visor, mask or suit."""
    text = description or ""
    if re.search(r"\b(helmets?|visors?|masks?|suits?)\b", text, re.I):
        return FACE_GEAR
    return FACE

VIEWS = ("front", "back", "face")
SIZE = (2000, 1125)      # the sheet, 16:9
GAP = 8                  # white between the panels
STILL_SIZE = (832, 1280)  # FLUX 3 klein's front view from words (2:3, about 1 MP, as Recast's "Make one")
WORK_MP = 1.0            # the editors work at about one megapixel
SOURCES = ("upload_id", "image_id", "cast_id", "describe")
MAX_DESCRIBE = 600

STAGE_FRONT, STAGE_BACK, STAGE_FACE, STAGE_SHEET = ("Making the front view", "Making the back view", "Making the close-up",
                                                    "Putting the sheet together")
STAGES = [Stage(STAGE_FRONT, 12), Stage(STAGE_BACK, 13), Stage(STAGE_FACE, 12), Stage(STAGE_SHEET, 3)]
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def request(body: object) -> dict:
    """A sheet job's params from a request body, or ValueError saying what's wrong.

    The source is exactly one of: an uploaded picture (upload_id), a project image (image_id), a cast member to upgrade
    (cast_id), or a description alone (describe). A new cast member needs a name; a seed is optional."""
    if not isinstance(body, dict):
        raise ValueError("Send the sheet's settings as a JSON object.")
    given = {k: " ".join(str(body[k]).split()) for k in SOURCES if body.get(k) is not None and str(body[k]).strip()}
    if not given:
        raise ValueError("Add a picture of them, or describe them.")
    if len(given) > 1:
        raise ValueError("Use one picture or a description, not both.")
    if len(given.get("describe", "")) > MAX_DESCRIBE:
        raise ValueError(f"Keep the description under {MAX_DESCRIBE} characters.")
    params = dict(given)
    name = " ".join(str(body.get("name") or "").split())[:60]
    if name:
        params["name"] = name
    elif "cast_id" not in given:
        raise ValueError("Give them a name.")
    seed = body.get("seed")
    if seed is not None and str(seed).strip() != "":
        try:
            params["seed"] = int(seed)
        except (TypeError, ValueError):
            raise ValueError("The seed must be a whole number.") from None
        if params["seed"] < 0:
            raise ValueError("The seed must be a whole number, 0 or more.")
    if body.get("seed_locked") is True:  # Lock seed was on: the cast card shows it locked again
        params["seed_locked"] = True
    return params


# --- the picture ------------------------------------------------------------------------------------------------------


def load(src) -> Image.Image:
    """A picture as RGB at about one megapixel, the editors' working size (a 48 MP photo shrinks, a small selfie grows;
    one near that size is only trimmed to multiples of 16), EXIF orientation applied, transparency shown on light grey."""
    im = src if isinstance(src, Image.Image) else Image.open(Path(src))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (200, 200, 200, 255)), rgba)
    im = im.convert("RGB")
    if 0.5 <= im.width * im.height / (WORK_MP * 1e6) <= 1.5:
        w, h = im.width // 16 * 16, im.height // 16 * 16
        x, y = (im.width - w) // 2, (im.height - h) // 2
        return im if (w, h) == im.size else im.crop((x, y, x + w, y + h))
    return im.resize(graphs.edit_size(im.width, im.height, WORK_MP), Image.LANCZOS)


def front_view(picture: Image.Image | np.ndarray) -> bool:
    """Whether a picture already is a clean full-length front view: one standing figure on a plain, light backdrop, the
    head near the top and the feet near the bottom, nothing cut off by the frame. (Which way they face isn't checked:
    a picture that passes is used as the front view as it is.)"""
    im = picture if isinstance(picture, Image.Image) else Image.fromarray(np.asarray(picture))
    im = im.convert("RGB")
    s = min(1.0, reference.WORK / max(im.size))
    if s < 1.0:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BILINEAR)
    plain = reference.foreground(np.asarray(im))
    if plain is None:  # a busy photo, a sky, a dark set: not a studio still
        return False
    h, w = plain.fg.shape
    if len(reference.spans(plain.fg, max(3, round(0.02 * w)))) != 1:  # a group, or a sheet already
        return False
    # The figure, and pieces close to it (a light shoe on a light floor), never a vignette's dark corner at the edge.
    fg = reference.touching(plain.fg, composite.grow(reference.largest(plain.fg, keep_frac=0.15), max(3, round(0.025 * h))))
    ys, xs = np.nonzero(fg)
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    top, bottom = ys.min() / h, (ys.max() + 1) / h
    cut = (fg[:2].any() or fg[-2:].any(axis=0).sum() > 0.35 * bw  # through the top, or a body (not two feet) at the bottom
           or fg[:, :2].sum() > 0.1 * h or fg[:, -2:].sum() > 0.1 * h)
    light = float(np.median(plain.field[~plain.fg] @ _LUMA)) >= 150  # the sheet's backdrop is light grey
    return top <= 0.2 and bottom >= 0.8 and not cut and bh >= 1.3 * bw and light


def portrait(im: Image.Image, aspect: float = 2 / 3) -> Image.Image:
    """A picture wider than a standing figure needs (a landscape photo, a square crop), centred on a 2:3 canvas of its
    own edge colour, so the front view the editor makes is portrait like the rest of the sheet. Nothing is cut off."""
    if im.width / im.height <= 0.8:
        return im
    W, H = im.width, round(im.width / aspect)
    a = np.asarray(im)
    colour = tuple(int(v) for v in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0))
    canvas = Image.new("RGB", (W, H), colour)
    canvas.paste(im, (0, (H - im.height) // 2))
    return canvas.resize(graphs.edit_size(W, H, WORK_MP), Image.LANCZOS)


# --- the sheet ---------------------------------------------------------------------------------------------------------


def panels(size: tuple[int, int] = SIZE, n: int = 3, gap: int = GAP) -> list[tuple[int, int, int, int]]:
    """The panels' boxes (x0, y0, x1, y1): n equal columns (within a pixel) with `gap` px between them, edge to edge."""
    W, H = size
    return [(round(i * (W + gap) / n), 0, round((i + 1) * (W + gap) / n) - gap, H) for i in range(n)]


def fit_height(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """A full-length view in its panel: scaled to the panel's height (never a head or the shoes cropped), then cropped
    at the sides, or padded with its own backdrop when it is narrower than the panel."""
    w, h = size
    im = im.convert("RGB")
    im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
    if im.width >= w:
        x = (im.width - w) // 2
        return im.crop((x, 0, x + w, h))
    a = np.asarray(im)
    colour = tuple(int(v) for v in np.median(np.concatenate([a[:, 0], a[:, -1]]), axis=0))
    out = Image.new("RGB", (w, h), colour)
    out.paste(im, ((w - im.width) // 2, 0))
    return out


def compose(front: Image.Image, back: Image.Image, face: Image.Image, size: tuple[int, int] = SIZE, gap: int = GAP) -> Image.Image:
    """FRONT | BACK | CLOSE-UP on white: the full-length views fit by height, the close-up fills its panel."""
    sheet = Image.new("RGB", size, (255, 255, 255))
    for (x0, y0, x1, y1), im, whole in zip(panels(size, 3, gap), (front, back, face), (True, True, False)):
        box = (x1 - x0, y1 - y0)
        sheet.paste(fit_height(im, box) if whole else ImageOps.fit(im.convert("RGB"), box, Image.LANCZOS), (x0, y0))
    return sheet


# --- building one -----------------------------------------------------------------------------------------------------


@dataclass
class Made:
    views: dict[str, Path]      # front, back, face (PNG)
    sheet: Path                 # the 16:9 composite (JPEG)
    source: Path                # the picture it was built from (FLUX 3 klein's still when made from words)
    made_front: bool            # the front view is the editor's (the picture wasn't a clean front view)
    editor: str                 # "Qwen-Image 2.1" or "Qwen-Image-Edit 2511"
    seed: int
    models: list[str] = field(default_factory=list)


def make(ctx, out_dir: Path, picture: Path | None = None, still_prompt: str | None = None, seed: int = 42,
         commercial_safe: bool = False) -> Made:
    """Build a sheet in `out_dir` from a picture, or from FLUX 3 klein's still for `still_prompt`.

    `ctx` is anything with run(graph) -> finished Comfy job, client.assets.from_file(path) and stage(label): the job
    queue's TaskContext, or a small runner for a one-off build."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    editor = graphs.edit_model(commercial_safe)
    models: list[str] = []
    if picture is None:
        if not (still_prompt or "").strip():
            raise ValueError("Add a picture of them, or describe them.")
        ctx.stage(STAGE_FRONT)
        job = ctx.run(images.localize_for(ctx, graphs.still(still_prompt, *STILL_SIZE, seed=seed, prefix="pickups/sheet_still")))
        outs = job.get_outputs("PU_Out")
        if not outs:
            raise RuntimeError("FLUX 3 klein didn't send a picture back. Try again.")
        picture = out_dir / "still.png"
        outs[0].to_file(picture)
        models.append("FLUX 3 klein")
    try:
        im = load(picture)
    except (OSError, ValueError) as e:
        raise RuntimeError("That picture can't be opened. Try a PNG or JPEG.") from e

    made_front = not front_view(im)
    front_png = out_dir / "front.png"
    # Each view shows on the page as soon as it exists (the job queue's TaskContext.partial; a one-off runner has none).
    early = getattr(ctx, "partial", None) or (lambda key, path: None)
    if made_front:
        ctx.stage(STAGE_FRONT)
        front = _edit(ctx, portrait(im), FRONT, seed, commercial_safe, out_dir, "front")
    else:
        front = im
        front.save(front_png)
    early("front", front_png)
    asset = ctx.client.assets.from_file(front_png)  # one upload for both edits
    ctx.stage(STAGE_BACK)
    back = _edit(ctx, front, BACK, seed, commercial_safe, out_dir, "back", asset)
    early("back", out_dir / "back.png")
    ctx.stage(STAGE_FACE)
    face = _edit(ctx, front, face_instruction(still_prompt), seed, commercial_safe, out_dir, "face", asset)
    early("face", out_dir / "face.png")

    ctx.stage(STAGE_SHEET)
    sheet = out_dir / "sheet.jpg"
    compose(front, back, face).save(sheet, quality=92)
    models.append(editor)
    return Made({v: out_dir / f"{v}.png" for v in VIEWS}, sheet, Path(picture), made_front, editor, seed, models)


_NAMES = {"front": "front view", "back": "back view", "face": "close-up"}


def _edit(ctx, im: Image.Image, instruction: str, seed: int, commercial_safe: bool, out_dir: Path, view: str,
          asset=None) -> Image.Image:
    """One view from the frame editor (graphs.image_edit), saved as <view>.png."""
    if asset is None:
        src = out_dir / f"{view}_in.png"
        im.save(src)
        asset = ctx.client.assets.from_file(src)
    job = ctx.run(graphs.image_edit(asset, instruction, seed=seed, prefix=f"pickups/sheet_{view}", commercial_safe=commercial_safe,
                                    size=im.size if commercial_safe else None))
    outs = job.get_outputs("PU_Plate")
    if not outs:
        raise RuntimeError(f"The {_NAMES[view]} came back empty. Try again.")
    dst = out_dir / f"{view}.png"
    outs[0].to_file(dst)
    with Image.open(dst) as out:
        return out.convert("RGB")
