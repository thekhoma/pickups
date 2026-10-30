"""Describe the shot: a local model that sees images looks at a shot (or a take) and writes what is in it, in the words
a prompt field wants, so "What's in the shot" needn't start empty.

`describe()` (POST /api/shots/{id}/describe) takes the first, middle and last frame of the part the video model works on
(the first `span` frames: common.MAX_FRAMES, or Render previz's length) from the viewer clip and asks the model to
describe them. It describes, it doesn't rewrite: only what is visible, in English, in the field's style.

Nothing new is installed: the model is one the shot tasks or the Prompt Director already have.

- backend "comfy" (default), on the GPU machine, the first of these that ComfyUI has:
  1. Qwen3-VL 8B, Qwen-Image 2.1's own text encoder (every task with this field needs it), shown the three frames.
     Checked on a 4090 against the two below: ~1-3 s, short, factual, and it keeps to the rules.
  2. Qwen 3.5 9B, the Qwen-Image 2.1 edit enhancer that sees images (`image_qwen_image_2_1_image_edit`): accurate but
     wordy, and it tends to name the layout's stand-ins ("a yellow figure") and invent gestures.
  3. LTX-2.5's own director (Gemma 4 E2B, `video_ltx2_5_i2v`), given the middle frame: fast but thin, and it guessed
     (night, rain on the windscreen) on a misty daytime road.
- backend "openai": the vision model of a local OpenAI-compatible server (settings director.vision_model). With no
  vision model set, the GPU machine describes.

Answers are cached per shot or take, target and field in shots/<id>/describe/<take>.json, so a second press is instant
unless the user asks for another (`refresh`).
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import director

VERSION = 1           # bump when the instructions change meaning: older cached answers are made again
FRAME_LONG = 768      # the frames' long side as the model sees them (plenty to read a shot; fewer image tokens)
FIELDS = ("scene", "motion", "look")
# The models on the GPU machine, best first. A "template" engine is that official template's enhancer (director.recipe);
# the others are a text encoder Pickups' own graphs load, run as a chat model (ComfyUI's TextGenerate).
QWEN_TEXT = "qwen3vl_8b_int8_convrot.safetensors"  # graphs.QWEN21_TEXT
ENGINES = {
    "qwen3vl": {"clip_name": QWEN_TEXT, "clip_type": "qwen_image", "frames": "all", "label": "Qwen3-VL 8B"},
    "qwen": {"template": "image_qwen_image_2_1_image_edit", "frames": "all", "label": "Qwen 3.5 9B"},
    "ltx": {"template": "video_ltx2_5_i2v", "frames": "middle", "label": "LTX-2.5 director"},
}
ENGINE_ORDER = ("qwen3vl", "qwen", "ltx")
# Sampling for a description: cooler than a creative rewrite, no thinking (it took 15-30 s and wasn't better).
SAMPLING = {"max_length": 320, "sampling_mode": "on", "temperature": 0.4, "top_k": 64, "top_p": 0.95, "min_p": 0.05,
            "repetition_penalty": 1.05, "seed": 0, "presence_penalty": 0, "thinking": False, "use_default_template": True}

BASE = (
    "You write a short, factual description of one film shot, from a few of its frames, for the prompt of a video or "
    "image model. You describe what is there; you never rewrite, embellish or imagine.\n\n"
    "Rules:\n"
    "- Only what is clearly visible. Don't guess: if you aren't sure what something is or does, use a plain word (\"a "
    "dark car\", \"a person in a light jacket\") or leave it out.\n"
    "- Never name brands, makes, models, logos, places or people, and never add writing, sounds, feelings or a story.\n"
    "- Write about the shot itself: never mention frames, images or what stays the same between them.\n"
    "- Plain English in the present tense. No lists, headings, quotes or preamble (\"The image shows\", \"This shot\"). "
    "Reply with the description only.")
CAMERA = (
    "The camera: the request says how it moves, measured from the pictures; trust that and say it in your own words. "
    "When it only says the camera moves, name the move: if we see the subject's back, the camera follows behind; if the "
    "subject faces us, it leads ahead of them (moving back); if we see their side, it tracks alongside; if the view "
    "turns across the scene, it pans or tilts. Mention a gesture only if it plainly differs between the frames.")
# What camera() measured, as the request tells the model (a 4090's VLMs guessed the move from three frames: often wrong).
MOVES = {
    "static": "Measured from the pictures: the camera doesn't move; anything that grows or shrinks is moving itself.",
    "push": "Measured from the pictures: the camera pushes in.",
    "pull": "Measured from the pictures: the camera pulls back.",
    "moving": ("Measured from the pictures: the camera moves with the action. Which way the subject faces says how: we "
               "see their back, the camera follows behind; they face the camera, it leads ahead of them; we see their "
               "side, it tracks alongside."),
}
LAYOUT = (
    "If the frames are a rough 3D layout (plain coloured stand-in figures, boxes and blocks), describe the live-action "
    "shot they stand for: a figure is a person (\"a driver\", \"a person\"), a car-shaped box is a car, rows of blocks "
    "are buildings or walls. Leave out their surfaces, colours and faces, and never write mannequin, stand-in, figure, "
    "block, blocky, 3D, render, clay, grey, previz or playblast.")
STYLE = {
    "scene": (
        "Write ONE or TWO sentences, under 55 words in all: who and what is in it (people by apparent age, clothing "
        "and colours; vehicles by type and colour), what they do, where, the light, and the camera. Leave out small "
        "details.\n\n" + CAMERA + "\n\nThe length and style to match: A man in a grey suit walks down a wet city "
        "street at night past lit shop windows, holding a black umbrella; the camera tracks alongside him."),
    "motion": (
        "Write TWO or THREE sentences, under 70 words in all, about the action and the camera: who or what moves, how "
        "and where in the frame, in order, then the camera.\n\n" + CAMERA + "\n\n" + LAYOUT + "\n\nThe length and "
        "style to match: A car pulls away from the kerb and drives off down a long straight road toward the low sun. "
        "The camera follows behind it at bumper height, then slows and lets it go."),
    "look": (
        "Write ONE or TWO sentences, under 55 words in all, describing the frame as a still photograph: the subject and "
        "where it is in the frame, the setting, the light (direction, softness, time of day), the main colours, and the "
        "shot size and angle. Don't describe motion.\n\nThe length and style to match: A wide shot of a stone castle "
        "on a small island in a loch at dusk, the sky deep blue above dark hills, warm light in a few windows and calm "
        "water in the foreground."),
}
LENGTH = {"scene": "one or two sentences, under 55 words", "motion": "two or three sentences, under 70 words",
          "look": "one or two sentences, under 55 words"}
ASK = {
    "all": "Frames from one shot, in order: the first, the middle and the last. Describe the shot in {length}.",
    "middle": "The middle frame of one shot. Describe the shot in {length}.",
    "one": "One frame of a shot. Describe it in {length}.",
}
# Render previz describes a layout as the shot it stands for: an answer with the layout's own words is asked again once.
LAYOUT_WORDS = re.compile(r"\b(mannequins?|stand-?ins?|figures?|blocks?|blocky|3d|render(ed|s)?|clay|gr[ae]y|previz|"
                          r"playblast|cgi)\b", re.I)
AGAIN = ("Describe the live-action shot this layout stands for, in {length}: people, cars, buildings, the light and the "
         "camera, with no colours of the stand-ins and none of these words: mannequin, figure, block, 3D, render, grey.")

# Plain words for what went wrong (shown to the user as they are).
OFFLINE = "ComfyUI isn't answering at {url}. Start it, or check the address in Setup, then try again."
MISSING = ("The model that describes shots isn't on your GPU machine yet ({file}). It comes with Change weather and "
           "Render previz: download their models in Setup, then try again.")
LOCAL_OFFLINE = "Your local model server isn't answering at {url}. Start it, or pick another in Setup."
NO_VISION = "{model} can't look at pictures. Pick a vision model for the prompt director in Setup."
NO_TEXT = "The model looked but didn't write a description. Try again."
NOT_ENGLISH = "The model answered in another language. Try again."
NO_FRAMES = "Couldn't read frames from this shot."

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class DescribeError(RuntimeError):
    """The model couldn't describe the shot (ComfyUI offline, the model missing). The message is shown to the user."""


def default_field(target: str) -> str:
    return "scene" if target == "ltx" else "look"


def pick_frames(n: int, span: int) -> list[int]:
    """The first, middle and last of the first `span` frames of an n-frame clip (fewer when the clip is that short)."""
    last = max(0, min(int(n), int(span)) - 1)
    return sorted({0, last // 2, last})


def style(target: str, field: str) -> str:
    """The field whose style applies: image models read a still, so they always get "look". ValueError: unknown."""
    if field not in FIELDS:
        raise ValueError(f"Unknown field {field}")
    return field if target == "ltx" else "look"


def instruction(target: str, field: str) -> str:
    """The system prompt: describe (never rewrite) only what is visible, in the style the target's field wants."""
    return f"{BASE}\n\n{STYLE[style(target, field)]}"


def ask(target: str, field: str, frames: str) -> str:
    """The request beside the pictures; frames: "all" (first, middle, last), "middle" or "one"."""
    return ASK[frames].format(length=LENGTH[style(target, field)])


# --- frames -----------------------------------------------------------------------------------------------------------

def _source(store, shot_id: str, version_id: str | None) -> tuple[str, Path]:
    """(cache key, viewer clip) for the original or a take. FileNotFoundError: no such shot or take."""
    if not version_id or version_id == "orig":
        store.shot_meta(shot_id)  # an unknown shot raises here
        viewer = store.shot_dir(shot_id) / "proxy.mp4"
        key = "orig"
    else:
        viewer = store.version_dir(shot_id, version_id) / "video.mp4"
        key = version_id
    if not viewer.exists():
        raise FileNotFoundError(version_id or shot_id)
    return key, viewer


def _count(viewer: Path) -> int:
    """Frames in the clip, from the container header (no decode); a full count only when the header has none."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of",
                          "json", str(viewer)], check=True, capture_output=True, text=True).stdout
    nb = str((json.loads(out).get("streams") or [{}])[0].get("nb_frames", ""))
    if nb.isdigit() and int(nb) > 0:
        return int(nb)
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(viewer)], check=True, capture_output=True, text=True)
    return max(1, int(out.stdout.strip() or 1))


def grab(viewer: Path, frames: list[int], folder: Path) -> list[Path]:
    """The frames as PNGs, at most FRAME_LONG on the long side, in one decode."""
    folder.mkdir(parents=True, exist_ok=True)
    pick = "+".join(f"eq(n\\,{f})" for f in frames)
    fit = f"scale={FRAME_LONG}:{FRAME_LONG}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(viewer), "-vf", f"select={pick},{fit}", "-fps_mode",
                    "passthrough", "-frames:v", str(len(frames)), str(folder / "f_%02d.png")], check=True)
    out = sorted(folder.glob("f_*.png"))
    if len(out) != len(frames):
        raise DescribeError(NO_FRAMES)
    return out


# --- the camera -------------------------------------------------------------------------------------------------------

CAM_W = 160          # measured on small greyscale copies
STATIC_MAE = 8.0     # mean grey-level change of the frame's outer ring below which the camera hasn't moved
ZOOM_GAIN = 0.6      # a zoom must explain the change this well (error at most 60% of the unmoved one)


def _grey(png: Path) -> np.ndarray:
    im = Image.open(png).convert("L")
    return np.asarray(im.resize((CAM_W, max(8, round(im.height * CAM_W / im.width))), Image.BILINEAR), np.float32)


def _warp(a: np.ndarray, s: float, dx: int, dy: int) -> np.ndarray:
    """a scaled by s about its centre and shifted by (dx, dy), same size (edge pixels repeated)."""
    h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(np.round((xs - w / 2 - dx) / s + w / 2).astype(int), 0, w - 1)
    sy = np.clip(np.round((ys - h / 2 - dy) / s + h / 2).astype(int), 0, h - 1)
    return a[sy, sx]


def camera(first: Path, last: Path) -> str:
    """How the camera moves from the first frame to the last: "static", "push" (in), "pull" (back) or "moving". Read
    from the frame's outer ring (the background, mostly, not the subject in the middle): unchanged is static; explained
    by a zoom about the centre is a push or a pull; anything else is a move the model names."""
    a, b = _grey(first), _grey(last)
    h, w = a.shape
    ring = np.ones_like(a, bool)
    ring[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = False
    ring[:4] = ring[-4:] = False
    ring[:, :4] = ring[:, -4:] = False
    base = float(np.abs(a - b)[ring].mean())
    if base < STATIC_MAE:
        return "static"
    best, scale = base, 1.0
    for s in np.arange(0.8, 1.2501, 0.025):
        for dx in range(-12, 13, 3):
            for dy in range(-6, 7, 3):
                e = float(np.abs(_warp(a, float(s), dx, dy) - b)[ring].mean())
                if e < best:
                    best, scale = e, float(s)
    if best <= ZOOM_GAIN * base and abs(scale - 1) >= 0.06:
        return "push" if scale > 1 else "pull"
    return "moving"


# --- the models -------------------------------------------------------------------------------------------------------

def _offline(e: BaseException) -> bool:
    """Nothing answered (refused, unreachable, timed out), from urllib or the SDK's HTTP client."""
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError)):
        return True
    return any(w in type(e).__name__ for w in ("ConnectError", "ConnectTimeout", "ConnectionError"))


def installed(comfy_url: str) -> set[str]:
    """The text-encoder files ComfyUI can load (CLIPLoader's list). DescribeError when ComfyUI doesn't answer."""
    try:
        spec = director._get(f"{comfy_url}/object_info/CLIPLoader")["CLIPLoader"]["input"]["required"]["clip_name"]
    except Exception as e:  # noqa: BLE001
        if _offline(e):
            raise DescribeError(OFFLINE.format(url=comfy_url)) from e
        return set()  # an odd answer: let the run itself say what's wrong
    return set(spec[0] if isinstance(spec[0], list) else (spec[1] or {}).get("options", []))


def _engine(comfy_url: str, prefer: str | None = None) -> tuple[str, dict]:
    """The first engine whose model ComfyUI has: (name, recipe as director.graph reads it). DescribeError: offline, or
    none of them installed."""
    order = [prefer] + [e for e in ENGINE_ORDER if e != prefer] if prefer in ENGINES else list(ENGINE_ORDER)
    have = installed(comfy_url)
    first = None
    for name in order:
        spec = ENGINES[name]
        if "template" in spec:
            try:
                rec = director.recipe(comfy_url, spec["template"])
            except Exception as e:  # noqa: BLE001
                if _offline(e):
                    raise DescribeError(OFFLINE.format(url=comfy_url)) from e
                continue  # this ComfyUI has no such template
        else:
            rec = {"node": "TextGenerate", "clip_name": spec["clip_name"], "clip_type": spec["clip_type"], "system": None,
                   "settings": dict(SAMPLING)}
        first = first or rec
        if not have or rec["clip_name"] in have:
            return name, rec
    raise DescribeError(MISSING.format(file=(first or {}).get("clip_name", QWEN_TEXT)))


def run_comfy(runner, comfy_url: str, target: str, field: str, frames: list[Path], seed: int = 0,
              engine: str | None = None, move: str | None = None) -> tuple[str, str]:
    """Describe on the GPU machine. Returns (text, engine label)."""
    name, rec = _engine(comfy_url, engine)
    spec = ENGINES[name]
    images = frames if spec["frames"] == "all" else [frames[len(frames) // 2]]
    system = instruction(target, field)
    # Pickups' own instruction on the model and loader, with SAMPLING's length, temperature and no thinking.
    keep = {k: SAMPLING[k] for k in ("thinking", "max_length", "temperature", "use_default_template")}
    rec = rec | {"system": system, "settings": rec["settings"] | keep}

    def one(request: str, s: int) -> str:
        try:
            assets = [runner.client.assets.from_file(f) for f in images]
            text = director.generated_text(comfy_url, runner.run(director.graph(rec, request, assets, s)))
        except Exception as e:  # noqa: BLE001
            if _offline(e):
                raise DescribeError(OFFLINE.format(url=comfy_url)) from e
            raise DescribeError(f"The model couldn't describe the shot: {e}") from e
        if director._CJK.search(text):  # the Qwen enhancers are bilingual
            text = director.generated_text(comfy_url, runner.run(director.graph(rec, f"Answer in English. {request}", assets, s + 1)))
        return text

    return _answer(one, target, field, "one" if len(frames) == 1 else spec["frames"], seed, move), spec["label"]


def run_openai(cfg: dict, target: str, field: str, frames: list[Path], seed: int = 0,
               move: str | None = None) -> tuple[str, str]:
    """Describe with the local server's vision model. Returns (text, model)."""
    model = cfg["vision_model"]
    pictures = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"}}
                for f in frames]
    system = instruction(target, field) + "\n\nWrite in English only."

    def one(request: str, s: int) -> str:
        try:
            return director.chat_openai(cfg, model, system, [{"type": "text", "text": request}, *pictures], temperature=0.4,
                                        top_p=0.95, seed=s, max_tokens=400)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            if any(w in detail.lower() for w in ("image", "vision", "multimodal")):
                raise DescribeError(NO_VISION.format(model=model)) from e
            raise DescribeError(f"The local model couldn't describe the shot: {e.code} {detail[:200]}".strip()) from e
        except RuntimeError as e:  # answered in another language twice
            raise DescribeError(NOT_ENGLISH) from e
        except Exception as e:  # noqa: BLE001
            if _offline(e):
                raise DescribeError(LOCAL_OFFLINE.format(url=cfg.get("url", ""))) from e
            raise DescribeError(f"The local model couldn't describe the shot: {e}") from e

    return _answer(one, target, field, "one" if len(frames) == 1 else "all", seed, move), model


def _answer(one, target: str, field: str, frames: str, seed: int, move: str | None = None) -> str:
    """Ask once (with the measured camera move, when there is one); a previz description that still talks about the
    layout ("a yellow figure", "grey blocks") is asked again with a firmer line, and the cleaner of the two is kept."""
    request = ask(target, field, frames)
    if move in MOVES and style(target, field) != "look":
        request = f"{request} {MOVES[move]}"
    text = one(request, seed).strip()
    if style(target, field) == "motion" and LAYOUT_WORDS.search(text):
        again = one(f"{request} {AGAIN.format(length=LENGTH['motion'])}", seed + 1).strip()
        if again and not director._CJK.search(again) and len(LAYOUT_WORDS.findall(again)) < len(LAYOUT_WORDS.findall(text)):
            text = again
    return _checked(text)


def _checked(text: str) -> str:
    text = text.strip()
    if not text:
        raise DescribeError(NO_TEXT)
    if director._CJK.search(text):
        raise DescribeError(NOT_ENGLISH)
    return text


# --- the describer ----------------------------------------------------------------------------------------------------

def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def backend(cfg: dict | None) -> str:
    """Where shots are described: "comfy" (Qwen3-VL 8B on the GPU machine, already installed for Change weather and
    Render previz), always. A local LM Studio vision model (Gemma 4 12B) was tried and got the shots wrong: a rider
    riding away came "towards the camera", a second car was invented, stand-ins became "a person in a yellow suit".
    So a local server describes only when the director settings opt in with "describe_backend": "openai"."""
    cfg = cfg or {}
    opted = cfg.get("describe_backend") == "openai" and cfg.get("url") and cfg.get("vision_model")
    return "openai" if opted else "comfy"


def describe(runner, store, shot_id: str, version_id: str | None = None, target: str = "ltx", field: str | None = None,
             cfg: dict | None = None, comfy_url: str = "", span: int = 97, frame: int | None = None, refresh: bool = False,
             seed: int = 0, engine: str | None = None) -> dict:
    """What's in the shot (or a take of it), in the style `target`'s `field` wants. Returns

        {text, target, field, version_id, frames: [indices looked at], camera: static | push | pull | moving | None,
         model, runs_on, seconds, cached}

    `frame` (image targets): describe that one frame instead of first / middle / last. Raises FileNotFoundError (no
    such shot or take), ValueError (unknown target or field) or DescribeError (the model couldn't run)."""
    field = field or default_field(target)
    if target not in director.TARGETS:
        raise ValueError(f"Unknown target {target}")
    instruction(target, field)  # an unknown field raises here
    key, viewer = _source(store, shot_id, version_id)
    n = _count(viewer)
    frames = [max(0, min(n - 1, int(frame)))] if frame is not None and target != "ltx" else pick_frames(n, span)
    how = backend(cfg)
    who = (cfg or {}).get("vision_model", "") if how == "openai" else (engine or "")
    st = viewer.stat()
    stamp = [VERSION, st.st_mtime_ns, st.st_size, frames, how, who]
    folder = store.shot_dir(shot_id) / "describe"
    cache = folder / f"{key}.json"
    slot = f"{target}:{field}"
    with _lock(f"{shot_id}:{key}"):
        saved: dict = {}
        if cache.exists():
            try:
                saved = json.loads(cache.read_text())
            except (OSError, ValueError):
                saved = {}
        hit = saved.get(slot) if isinstance(saved.get(slot), dict) else None
        if not refresh and hit and hit.get("stamp") == stamp:
            return hit["result"] | {"cached": True}
        work = folder / f"_work_{key}"
        shutil.rmtree(work, ignore_errors=True)
        t0 = time.monotonic()
        try:
            pngs = grab(viewer, frames, work)
            move = camera(pngs[0], pngs[-1]) if len(pngs) > 1 and style(target, field) != "look" else None
            if how == "openai":
                text, model = run_openai(cfg or {}, target, field, pngs, seed, move)
                runs_on = model
            else:
                text, model = run_comfy(runner, comfy_url, target, field, pngs, seed, engine, move)
                runs_on = "ComfyUI"
        finally:
            shutil.rmtree(work, ignore_errors=True)
        result = {"text": text, "target": target, "field": field, "version_id": key, "frames": frames, "camera": move,
                  "model": model, "runs_on": runs_on, "seconds": round(time.monotonic() - t0, 1)}
        saved[slot] = {"stamp": stamp, "result": result}
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved, indent=1))
        tmp.replace(cache)
        return result | {"cached": False}
