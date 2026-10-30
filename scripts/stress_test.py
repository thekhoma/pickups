"""Stress test: run a matrix of shots × characters × tasks × seeds through Pickups, one cell at a time, and grade every take.

Cells run through the app's own job queue inside this process (JobManager and the same tasks the server runs, as in
scripts/run_task.py): restarting the server never kills a cell in flight, and every run uses the code on disk. Before each
cell the harness waits until the app's queue and ComfyUI are idle, so cells never share the GPU and timings stay fair.
Takes land in the library (a "Stress test" project), so each one can be opened in the app.

Per cell, in <results>/<stage>/<cell>/: result.json (take id, timings, scores, error), contact.jpg (8 frames, before and
after rows, plus close-ups), sbs.mp4 (2 s side by side) and sbs.gif. Per stage: report.html (opens from disk) and
report.csv. Matrix format, checks and grades: docs/stress-test.md.

Usage: uv run python scripts/stress_test.py MATRIX.json [--dry-run] [--judge] [--only GLOB,...] [--retry-failed]
       [--force] [--rescore] [--report-only] [--limit N]
"""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import html
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pickups import director, media
from pickups.app import store as store_mod

SRC = Path(__file__).resolve().parents[1] / "src" / "pickups"
THRESHOLDS = {"coverage_min": 0.7, "coverage_max": 1.6, "halo_max": 2.0, "flicker_max": 1.5, "empty_new_frames_max": 0,
              "outside_max": 0, "eta_factor": 1.5, "judge_a": 4, "judge_fail": 2}
QA_KEYS = {"coverage": ("coverage", "coverage_ratio"), "halo": ("halo", "halo_score"), "flicker": ("flicker", "flicker_score"),
           "empty_new_frames": ("empty_new_frames", "empty_frames")}
# How Recast went about it (details._qa, not graded): which reference path, whose mask, who was re-rendered, what size.
HOW_KEYS = ("reference", "reference_mask", "object_indices", "people_on_first_frame", "new_track", "scail_long")
FINISHED = ("done", "failed", "timeout", "error")  # the run ended; "running" and "interrupted" cells run again
CHARACTER_TASKS = ("recast",)
SHEET_FRAMES = 8
# A failure message a person can act on has none of these (risk 6: every refusal needs a fast, worded outcome).
TECHNICAL = re.compile(r"Traceback|\b[A-Z]\w*(Error|Exception)\b|NoneType|has no attribute|out of bounds|index out of range|"
                       r"zero-size array|could not be broadcast|returned non-zero|\[Errno|\.py\b|\bline \d+|\bffmpeg\b")
BG, PANEL, INK, MUTED, MARK = (18, 16, 14), (32, 29, 25), (239, 233, 221), (163, 155, 141), (239, 74, 63)


class StopRun(Exception):
    """Stop the whole run (the GPU is stuck, or busy for too long). Finished cells stay finished; run again to carry on."""


# --- the matrix ---------------------------------------------------------------------------------------------------
def slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "x"


def who_from_prompt(prompt: str | None) -> str:
    """The person in Recast's STILL prompt ("...full-length shot of {who}, standing centered...")."""
    m = re.search(r"full-length shot of (.+?), standing centered", prompt or "")
    return m.group(1).strip() if m else ""


def character(e: dict, base: Path) -> dict:
    c = {"id": slug(e.get("id") or e.get("name") or "character"), "name": e.get("name") or e.get("id") or "Character",
         "describe": e.get("describe", ""), "params": e.get("params", {})}
    if e.get("image"):
        p = Path(e["image"]).expanduser()
        c["image"] = str(p if p.is_absolute() else base / p)
    for k in ("cast_image", "ref_upload_id"):
        if e.get(k):
            c[k] = e[k]
    return c


def from_cast(data: dict, variant: int | str | list = "chosen", only: list | None = None) -> tuple[list[dict], list[str]]:
    """Characters from the fun-cast file: {"project", "cast": {name: {"status", "images": [image ids], "error"}}, "prompts",
    "chosen": {name: {"image_id"}}}. variant: "chosen" (the picked picture, else the first), a picture number from 0, or a list."""
    entries = data.get("cast") or {}
    if isinstance(entries, list):
        entries = {e.get("name") or e.get("id"): e for e in entries}
    variants = variant if isinstance(variant, list) else [variant]
    out, notes = [], []
    for name, e in entries.items():
        if only and name not in only:
            continue
        imgs = e.get("images") or []
        if e.get("status") not in (None, "done") or not imgs:
            notes.append(f"{name} has no picture in the cast ({e.get('error') or e.get('status') or 'empty'}), so it's left out.")
            continue
        who = e.get("describe") or who_from_prompt((data.get("prompts") or {}).get(name)) or name
        for v in variants:
            if v == "chosen":
                image = ((data.get("chosen") or {}).get(name) or {}).get("image_id") or imgs[0]
                n = imgs.index(image) if image in imgs else 0
            elif isinstance(v, int) and 0 <= v < len(imgs):
                image, n = imgs[v], v
            else:
                notes.append(f"{name} has no picture {v!r}.")
                continue
            many = len(variants) > 1
            out.append({"id": slug(name) + (f"-v{n + 1}" if many else ""), "name": name + (f" ({n + 1})" if many else ""),
                        "describe": who, "params": {}, "cast_image": {"project": data.get("project"), "id": image}})
    return out, notes


def load_matrix(path: Path) -> tuple[dict, list[str]]:
    """The matrix spec, with characters expanded (a cast file is read now, so one written after the spec still counts)."""
    m = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    m["_base"] = str(base)
    m["stage"] = slug(m.get("stage") or Path(path).stem)
    m["thresholds"] = THRESHOLDS | (m.get("thresholds") or {})
    chars, notes = [], []
    for e in m.get("characters", []):
        if "from_cast" not in e:
            chars.append(character(e, base))
            continue
        f = base / e["from_cast"]
        if not f.is_file():
            notes.append(f"{e['from_cast']} isn't there yet: its characters join the matrix once it exists.")
            continue
        got, why = from_cast(json.loads(f.read_text()), e.get("variant", "chosen"), e.get("only"))
        chars += got
        notes += why
    m["_characters"] = chars
    return m, notes


def plan(m: dict) -> list[dict]:
    """Every cell, shots outermost: {"id", "shot", "character", "task", "seed", "params", "expect"}."""
    cells, seen = [], set()
    for shot in m.get("shots", []):
        for t in m.get("tasks") or ["recast"]:
            t = {"task": t} if isinstance(t, str) else t
            task = t["task"]
            cast = m.get("_characters", []) if t.get("characters", task in CHARACTER_TASKS) else [None]
            for ch in cast:
                for seed in shot.get("seeds") or t.get("seeds") or m.get("seeds") or [42]:
                    params = {**t.get("params", {}), **shot.get("params", {}), **(shot.get("task_params") or {}).get(task, {})}
                    if ch:
                        params |= ({"describe": ch["describe"]} if ch.get("describe") else {}) | ch.get("params", {})
                    params["seed"] = int(seed)
                    cid = "__".join((slug(shot["id"]), ch["id"] if ch else "none", slug(task), f"s{seed}"))
                    if cid in seen:
                        raise ValueError(f"Two cells are both called {cid}: give the shots or characters different ids.")
                    seen.add(cid)
                    cells.append({"id": cid, "shot": shot, "character": ch, "task": task, "seed": int(seed), "params": params,
                                  "expect": shot.get("expect") or t.get("expect") or {}})
    return cells


def select(cells: list[dict], only: str) -> list[dict]:
    pats = [p.strip() for p in only.split(",") if p.strip()]
    return [c for c in cells if not pats or any(fnmatch.fnmatch(c["id"], p) for p in pats)]


def decide(prev: dict | None, o) -> str | None:
    """What a cell needs now: "run" (render a take), "post" (pictures, scores or a verdict for the take it has) or None."""
    st = (prev or {}).get("status")
    if o.force or st not in FINISHED:
        return "run"
    if st != "done":
        return "run" if o.retry_failed else None
    if o.rescore or not prev.get("artefacts"):
        return "post"
    if getattr(o, "judge", False) and prev.get("character") and not (prev.get("judge") or {}).get("identity_match"):
        return "post"
    return None


def reference(ch: dict | None) -> Path | None:
    """The character's picture on disk (read-only), or None for a described character (Recast makes a FLUX 3 klein still)."""
    if not ch:
        return None
    lib = store_mod.ROOT
    if ch.get("image"):
        p = Path(ch["image"])
        if not p.is_file():
            raise FileNotFoundError(f"Can't find {ch['name']}'s picture ({p.name}).")
        return p
    if ch.get("cast_image"):
        pid, iid = str(ch["cast_image"].get("project")), str(ch["cast_image"].get("id"))
        if not (store_mod.ITEM_ID.fullmatch(pid) and store_mod.ITEM_ID.fullmatch(iid)):
            raise FileNotFoundError(f"{ch['name']}'s cast picture isn't a library image ({pid}/{iid}).")
        items = _read_json(lib / "projects" / pid / "images.json") or []
        hit = next((x for x in items if x.get("id") == iid), None)
        p = lib / "projects" / pid / "images" / hit["file"] if hit else None
        if p is None or not p.is_file():
            raise FileNotFoundError(f"{ch['name']}'s cast picture ({iid}) is gone from the library.")
        return p
    if ch.get("ref_upload_id"):
        uid = str(ch["ref_upload_id"])
        hits = sorted((lib / "uploads").glob(f"{uid}.*")) if store_mod.UPLOAD_ID.fullmatch(uid) else []
        if not hits:
            raise FileNotFoundError(f"{ch['name']}'s upload ({uid}) is gone from the library.")
        return hits[0]
    return None


def resolve_uploads(runner, value, base: Path):
    """"@upload:<path>" anywhere in params becomes an upload id (as scripts/run_examples.py)."""
    if isinstance(value, str) and value.startswith("@upload:"):
        p = Path(value[8:]).expanduser()
        return runner.upload(p if p.is_absolute() else base / p)
    if isinstance(value, list):
        return [resolve_uploads(runner, v, base) for v in value]
    if isinstance(value, dict):
        return {k: resolve_uploads(runner, v, base) for k, v in value.items()}
    return value


# --- small helpers ------------------------------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - the user's own local servers
        return json.loads(r.read())


def fmt_dur(s) -> str:
    if s is None:
        return "–"
    s = int(round(float(s)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def worded(text: str | None) -> bool:
    return bool((text or "").strip()) and not TECHNICAL.search(text)


def code_stamp() -> str:
    """When the code on disk last changed: tells which fixes a take was made with."""
    t = max((p.stat().st_mtime for p in SRC.rglob("*.py")), default=0)
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def read_qa(details: dict) -> dict:
    """The take's automatic QA (details._qa), numbers only; missing keys stay missing."""
    qa = (details or {}).get("_qa")
    out = {}
    if not isinstance(qa, dict):
        return out
    for key, aliases in QA_KEYS.items():
        raw = next((qa[a] for a in aliases if a in qa), None)
        try:
            if raw is not None and not isinstance(raw, bool) and math.isfinite(float(raw)):
                out[key] = float(raw)
        except (TypeError, ValueError):
            pass
    if "empty_new_frames" in out:
        out["empty_new_frames"] = int(out["empty_new_frames"])
    return out


# --- checks and grades --------------------------------------------------------------------------------------------
def checks(r: dict, th: dict = THRESHOLDS) -> list[dict]:
    """The automatic checks of a finished take: ok is True, False, or None when it couldn't be measured."""
    out = []
    qa = r.get("qa") or {}

    def add(key, label, value, ok, limit):
        out.append({"key": key, "label": label, "value": value, "ok": ok, "limit": limit})

    if r.get("task") == "recast":
        v = qa.get("coverage")
        add("coverage", "Coverage", v, None if v is None else th["coverage_min"] <= v <= th["coverage_max"],
            f"{th['coverage_min']}–{th['coverage_max']}")
        v = qa.get("halo")
        add("halo", "Halo", v, None if v is None else v < th["halo_max"], f"under {th['halo_max']}")
        v = qa.get("flicker")
        add("flicker", "Flicker", v, None if v is None else v < th["flicker_max"], f"under {th['flicker_max']}")
        v = qa.get("empty_new_frames")
        add("empty_new_frames", "Empty frames", v, None if v is None else v <= th["empty_new_frames_max"],
            f"at most {th['empty_new_frames_max']}")
    v = r.get("outside_selection")
    if v is not None or r.get("task") in ("recast", "remove"):
        add("outside", "Outside", v, None if v is None else v <= th["outside_max"], f"at most {th['outside_max']} px")
    t = r.get("timings") or {}
    limit = th["eta_factor"] * t["eta_s"] if t.get("eta_s") else None
    run_s = t.get("run_s")
    add("time", "Time", run_s, None if run_s is None or limit is None or r.get("contended") else run_s <= limit, limit)
    return out


def _check_reason(c: dict) -> str:
    v, k = c["value"], c["key"]
    if c["ok"] is None:
        if k == "time":
            return "Time not counted: the app rendered something at the same time." if v is not None else "Time not measured."
        return f"{c['label']} not measured."
    if k == "coverage":
        return f"Coverage {v:.2f}: the new character is {'smaller' if v < 1 else 'bigger'} than the old one (should be {c['limit']})."
    if k in ("halo", "flicker"):
        return f"{c['label']} {v:.2f} (should be {c['limit']})."
    if k == "empty_new_frames":
        return f"{v} frame{'s' if v != 1 else ''} with no new character found."
    if k == "outside":
        return f"{int(v):,} pixels changed outside the selection."
    return f"Took {fmt_dur(v)} (limit {fmt_dur(c['limit'])})."


def _expected_failure(r: dict, exp: dict) -> tuple[str, list[str]]:
    st, err = r.get("status"), (r.get("error") or "").strip()
    secs, within = (r.get("timings") or {}).get("run_s"), float(exp.get("within_s", 120))
    if st == "done":
        return "F", ["Made a take where it should have stopped with a message."]
    why = []
    if st == "timeout":
        why.append("Timed out instead of stopping with a message.")
    elif st == "error":
        why.append(f"The harness couldn't start it: {err}")
    if secs is not None and secs > within:
        why.append(f"Took {fmt_dur(secs)} to say so (limit {fmt_dur(within)}).")
    if st == "failed" and not worded(err):
        why.append(f"The message isn't written for a person: {err[:160] or '(empty)'}")
    if st == "failed" and exp.get("message") and not re.search(exp["message"], err, re.I):
        why.append(f"Unexpected message: {err[:160]}")
    return ("F", why) if why else ("A", [f"Stopped in {fmt_dur(secs)} with: {err}"])


def grade(r: dict, th: dict = THRESHOLDS) -> tuple[str, list[str], list[dict]]:
    """A, B or F ("–" for a cell that hasn't finished), the reasons in words, and the checks. Rules: docs/stress-test.md."""
    st = r.get("status")
    if st not in FINISHED:
        return "–", [{"running": "Started, not finished yet.", "interrupted": "Interrupted; it runs again next time."}.get(st, "Not run yet.")], []
    exp = r.get("expect") or {}
    if exp.get("status") in ("failed", "error"):
        g, why = _expected_failure(r, exp)
        return g, why, []
    if st != "done":
        what = {"failed": "Failed", "timeout": "Timed out", "error": "Lost its take" if r.get("take_id") else "Couldn't start"}[st]
        at = f" at “{r['failed_stage']}”" if r.get("failed_stage") else ""
        why = [f"{what}{at}" + (f": {r['error']}" if r.get("error") else ".")]
        if st == "failed" and not worded(r.get("error")):
            why.append("The message isn't written for a person.")
        return "F", why, []
    cs = checks(r, th)
    fails = [_check_reason(c) for c in cs if c["ok"] is False]
    unmeasured = [c for c in cs if c["ok"] is None]
    qa_keys = set(QA_KEYS)
    qa_missing = qa_keys <= {c["key"] for c in unmeasured}  # one line for "no QA at all", not four
    soft = ["QA scores missing (an older take, or QA failed)."] if qa_missing else []
    soft += [_check_reason(c) for c in unmeasured if not (qa_missing and c["key"] in qa_keys)]
    if r.get("artefact_error"):
        soft.append(f"Couldn't make the contact sheet: {r['artefact_error']}")
    if r.get("character"):
        j = r.get("judge") or {}
        if j.get("identity_match") and j.get("artefacts"):
            text = f"Judge: identity {j['identity_match']}/5, artefacts {j['artefacts']}/5" + (f" ({j['notes']})" if j.get("notes") else ".")
            if min(j["identity_match"], j["artefacts"]) <= th["judge_fail"]:
                fails.append(text)
            elif min(j["identity_match"], j["artefacts"]) < th["judge_a"]:
                soft.append(text)
        else:
            soft.append(f"Not judged ({j['error']})" if j.get("error") else "Not judged.")
    if fails:
        return "F", fails + soft, cs
    return ("B", soft, cs) if soft else ("A", [], cs)


# --- pictures -----------------------------------------------------------------------------------------------------
def _dims(video: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True, check=True).stdout
    w, h = out.strip().splitlines()[0].split(",")[:2]
    return int(w), int(h)


def spread(n: int, k: int = SHEET_FRAMES) -> list[int]:
    if n <= 0:
        return []
    return sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(min(k, n))})


def grab(video: Path, idx: list[int], long_side: int) -> list[np.ndarray]:
    """These frames (by index) as RGB arrays, no bigger than long_side."""
    w0, h0 = _dims(video)
    s = min(1.0, long_side / max(w0, h0))
    w, h = max(2, round(w0 * s / 2) * 2), max(2, round(h0 * s / 2) * 2)
    order = sorted(set(idx))
    sel = "+".join(f"eq(n,{i})" for i in order)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{sel}',scale={w}:{h}:flags=area",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    size = w * h * 3
    got = [np.frombuffer(raw, np.uint8, size, k * size).reshape(h, w, 3) for k in range(len(raw) // size)]
    if not got:
        raise RuntimeError(f"No frames came out of {video.name}.")
    at = {i: got[min(k, len(got) - 1)] for k, i in enumerate(order)}
    return [at[i] for i in idx]


def changed(before: np.ndarray, after: np.ndarray, thr: int = 40) -> np.ndarray:
    """Pixels the take changed (well above compression noise)."""
    if before.shape != after.shape:
        return np.zeros(before.shape[:2], bool)
    return np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2) > thr


def change_box(mask: np.ndarray, pad: float = 1.6, min_frac: float = 0.15) -> list[float] | None:
    """A padded box around the changed pixels (normalised x0, y0, x1, y1), between 1:2 and 2:1, or None when nothing
    changed or the change fills most of the frame."""
    h, w = mask.shape
    if mask.mean() < 0.0002:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max((x1 - x0 + 1) * pad, w * min_frac), max((y1 - y0 + 1) * pad, h * min_frac)
    bw, bh = max(bw, bh / 2), max(bh, bw / 2)  # never a sliver
    bw, bh = min(bw, w), min(bh, h)
    if bw * bh >= 0.7 * w * h:
        return None
    x0, y0 = min(max(cx - bw / 2, 0), w - bw), min(max(cy - bh / 2, 0), h - bh)
    return [round(x0 / w, 4), round(y0 / h, 4), round((x0 + bw) / w, 4), round((y0 + bh) / h, 4)]


def focus_boxes(before: list[np.ndarray], after: list[np.ndarray]) -> tuple[list[float] | None, list[list[float] | None]]:
    """The box around everything that changed, and one per frame (a walker coming closer gets a bigger box each frame).
    A frame with no change of its own borrows the overall box; all None when the change isn't local."""
    masks = [changed(b, a) for b, a in zip(before, after)]
    overall = change_box(np.logical_or.reduce(masks)) if masks else None
    if overall is None and not any(change_box(m) for m in masks):
        return None, [None] * len(masks)
    return overall, [change_box(m) or overall for m in masks]


def crop(frame: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = max(x0 + 2, int(round(box[2] * w))), max(y0 + 2, int(round(box[3] * h)))
    return frame[y0:y1, x0:x1]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType
        return ImageFont.load_default()


def _fit(f: np.ndarray, tw: int, th: int) -> Image.Image:
    """A frame or close-up inside a tw × th tile, letterboxed on the sheet's background."""
    im = Image.fromarray(np.ascontiguousarray(f))
    s = min(tw / im.width, th / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    tile = Image.new("RGB", (tw, th), BG)
    tile.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    return tile


def contact_sheet(before: list[np.ndarray], after: list[np.ndarray], idx: list[int], boxes: list, title: str) -> Image.Image:
    """Before and after rows, plus close-ups on the change when it is local (one box per frame), one column per frame."""
    h, w = before[0].shape[:2]
    tw = 300 if w >= h else max(60, round(300 * w / h))
    th = round(tw * h / w)
    rows = [("Before", before, th), ("After", after, th)]
    if any(boxes):
        # Close-ups of a standing figure are tall: give those rows up to twice the height so they read.
        ratio = float(np.median([(b[3] - b[1]) * h / ((b[2] - b[0]) * w) for b in boxes if b]))
        ch = int(min(max(th, tw * ratio), 2 * th))
        rows += [("Before, close", [crop(f, b) if b else f for f, b in zip(before, boxes)], ch),
                 ("After, close", [crop(f, b) if b else f for f, b in zip(after, boxes)], ch)]
    lab, gap, head, colhead = 124, 4, 52, 24
    sheet = Image.new("RGB", (lab + len(idx) * (tw + gap), head + colhead + sum(rh + gap for *_, rh in rows)), BG)
    d = ImageDraw.Draw(sheet)
    d.text((14, 14), title, fill=INK, font=_font(22))
    for c, i in enumerate(idx):
        d.text((lab + c * (tw + gap) + 4, head + 3), f"frame {i}", fill=MUTED, font=_font(13))
    y = head + colhead
    for name, frames, rh in rows:
        d.text((14, y + rh // 2 - 9), name, fill=MUTED if "close" in name else INK, font=_font(16))
        for c, f in enumerate(frames):
            x = lab + c * (tw + gap)
            sheet.paste(_fit(f, tw, rh), (x, y))
            if name == "After" and (b := boxes[c]):  # where the close-up below looks
                d.rectangle([x + b[0] * tw, y + b[1] * th, x + b[2] * tw - 1, y + b[3] * th - 1], outline=MARK, width=2)
        y += rh + gap
    return sheet


def side_by_side(before: Path, after: Path, out: Path, fps: float, n: int, seconds: float = 2.0) -> None:
    start = max(0.0, (n / max(fps, 1.0) - seconds) / 2)  # the middle of the clip
    graph = (f"[0:v]scale=-2:540,pad=iw+8:ih:0:0:color=0x{'%02x%02x%02x' % BG}[a];[1:v]scale=-2:540[b];"
             "[a][b]hstack=inputs=2,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{seconds}", "-i", str(before),
                    "-ss", f"{start:.3f}", "-t", f"{seconds}", "-i", str(after), "-filter_complex", graph, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-movflags", "+faststart", str(out)],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)


def gif(video: Path, out: Path, width: int = 640) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    f"fps=12,scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    str(out)], check=True, capture_output=True, stdin=subprocess.DEVNULL)


def artefacts(tdir: Path, d: Path, title: str, ref: Path | None) -> dict:
    """The cell's pictures from a take folder: contact sheet, side by side (MP4 + GIF), judge close-ups, reference."""
    before, after = tdir / "before.mp4", tdir / "video.mp4"
    if not before.is_file():  # every task writes one today; an older take falls back to the shot's own viewer
        before = tdir.parent.parent / "proxy.mp4"
    pa = media.probe(after)
    n = min(pa.frames, media.probe(before).frames)
    idx = spread(n)
    fb, fa = grab(before, idx, 960), grab(after, idx, 960)
    overall, boxes = focus_boxes(fb, fa)
    sheet = contact_sheet(fb, fa, idx, boxes, title)
    sheet.save(d / "contact.jpg", quality=88)
    small = sheet.copy()
    small.thumbnail((760, 760), Image.LANCZOS)
    small.save(d / "contact_thumb.jpg", quality=72)
    side_by_side(before, after, d / "sbs.mp4", float(pa.fps), n)
    gif(d / "sbs.mp4", d / "sbs.gif")
    # The judge sees three frames close on the new character, cut from the full-resolution master.
    jidx = [round((n - 1) * x) for x in (0.2, 0.5, 0.8)]
    _, jboxes = focus_boxes(grab(before, jidx, 960), grab(after, jidx, 960))
    master = tdir / "master.mov"
    judge_frames = []
    for k, (f, b) in enumerate(zip(grab(master if master.is_file() else after, jidx, 3840), jboxes)):
        im = Image.fromarray(np.ascontiguousarray(crop(f, b) if b else f))
        im.thumbnail((1024, 1024), Image.LANCZOS)
        im.save(d / f"judge_{k + 1}.jpg", quality=90)
        judge_frames.append(f"judge_{k + 1}.jpg")
    if ref is not None and ref.is_file():
        im = Image.open(ref).convert("RGB")
        im.thumbnail((1024, 1024), Image.LANCZOS)
        im.save(d / "reference.jpg", quality=90)
    return {"frames": n, "sheet_frames": idx, "focus": overall, "contact": "contact.jpg", "thumb": "contact_thumb.jpg",
            "sbs": "sbs.mp4", "gif": "sbs.gif", "judge_frames": judge_frames,
            "reference": "reference.jpg" if (d / "reference.jpg").is_file() else None}


# --- the judge (optional: the user's local vision model through LM Studio's OpenAI-compatible API) --------------------
JUDGE_SYSTEM = (
    "You are a strict visual-effects supervisor checking an AI character replacement in real footage. Image 1 is the "
    "character reference: who should appear. Images 2 to 4 are frames from the finished shot, close on the replaced person.\n"
    "Score two things from 1 to 5.\n"
    "identity_match: does the figure in images 2 to 4 read as the character in image 1 (species or skin, face or head, "
    "costume, colours, materials, body proportions)? 5 = unmistakably the same character in every frame; 3 = the same idea "
    "with clear differences (wrong colours, missing costume parts, a human face where the reference has none); 1 = a "
    "different person, or the original performer still there.\n"
    "artefacts: how clean is the composite? 5 = could pass as filmed; 4 = tiny flaws you only see when looking for them; "
    "3 = noticeable flaws (a light outline or glow around the figure, smeared hands or feet, a soft or pasted look, wrong "
    "scale); 2 = obvious damage (a ghost of the original person, melted limbs, a box or patch around the figure); 1 = broken.\n"
    'Reply with JSON only, in English: {"identity_match": <1-5>, "artefacts": <1-5>, "notes": "<one short sentence: the '
    'biggest problem, or clean>"}')
JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["identity_match", "artefacts", "notes"],
    "properties": {"identity_match": {"type": "integer", "minimum": 1, "maximum": 5},
                   "artefacts": {"type": "integer", "minimum": 1, "maximum": 5}, "notes": {"type": "string"}}}}}
RETRY = 'Reply with the JSON object only, in English: {"identity_match": 1-5, "artefacts": 1-5, "notes": "..."}'


def parse_verdict(text: str | None) -> dict | None:
    """A strict verdict, or None for anything else (prose, out-of-range scores, notes in another language)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
    except ValueError:
        return None
    if not isinstance(v, dict):
        return None
    out = {}
    for key, aliases in (("identity_match", ("identity_match", "identity")), ("artefacts", ("artefacts", "artifacts"))):
        raw = next((v[k] for k in aliases if k in v), None)
        if isinstance(raw, bool):
            return None
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        if n != int(n) or not 1 <= n <= 5:
            return None
        out[key] = int(n)
    notes = v.get("notes", "")
    if not isinstance(notes, str) or director._CJK.search(notes):
        return None
    out["notes"] = " ".join(notes.split())[:300]
    return out


def _v1(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def _image_part(path: Path, long_side: int = 768) -> dict:
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def judge_config(url: str | None = None, model: str | None = None, timeout: float = 180.0) -> dict:
    from pickups.app.comfy import load_settings

    cfg = dict(load_settings().get("director") or {})
    return {"url": url or cfg.get("url") or "http://127.0.0.1:1234/v1", "model": model or cfg.get("vision_model") or cfg.get("model"),
            "timeout": timeout}


def judge_ready(cfg: dict, timeout: float = 5.0) -> str | None:
    """None when the judge's server answers and lists the model; otherwise what's wrong, in words."""
    if not cfg.get("model"):
        return "no vision model is set (Setup → Director)"
    try:
        ids = [x.get("id") for x in _get_json(f"{_v1(cfg['url'])}/models", timeout).get("data", [])]
    except Exception:  # noqa: BLE001
        return f"the judge's server isn't answering at {cfg['url']}"
    return None if cfg["model"] in ids else f"{cfg['model']} isn't available on the judge's server (load it in LM Studio)"


def _chat(url: str, body: dict, timeout: float) -> str:
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the user's own local server
        out = json.loads(r.read())
    return ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def judge(cfg: dict, ref: Path, frames: list[Path]) -> dict:
    """{identity_match, artefacts, notes, model, seconds} from the vision model. Never raises: problems come back as error."""
    url, model, timeout = _v1(cfg["url"]), cfg.get("model"), float(cfg.get("timeout", 180.0))
    if not model:
        return {"error": "no vision model is set"}
    try:
        content = ([{"type": "text", "text": "Image 1, the character:"}, _image_part(ref),
                    {"type": "text", "text": "Images 2 to 4, the finished shot:"}] + [_image_part(f) for f in frames])
    except OSError as e:
        return {"error": f"couldn't read the pictures ({e})", "model": model}
    body = {"model": model, "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 400, "seed": 7, "reasoning_effort": "none", "response_format": JUDGE_SCHEMA}
    t0, last = time.monotonic(), ""
    for _ in range(3):
        try:
            text = _chat(url, body, timeout)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "response_format" in body:  # a server without structured output: ask in words
                body.pop("response_format")
                continue
            return {"error": f"the judge answered HTTP {e.code}", "model": model}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            return {"error": f"the judge didn't answer ({'timed out' if 'timed out' in str(reason) else reason})", "model": model}
        except ValueError:
            text = ""
        v = parse_verdict(text)
        if v:
            return v | {"model": model, "seconds": round(time.monotonic() - t0, 1)}
        last = text
        body["messages"] = body["messages"][:2] + [{"role": "assistant", "content": (text or "")[:500]}, {"role": "user", "content": RETRY}]
    return {"error": "the judge's answer wasn't a usable verdict", "raw": " ".join((last or "").split())[:200], "model": model}


# --- running cells through the app ----------------------------------------------------------------------------------
class AppRunner:
    """Cells through the app's own job queue in this process (JobManager + TASKS, as scripts/run_task.py)."""

    def __init__(self, results: Path, project: str, server: str, wait: bool = True, idle_timeout_s: float = 4 * 3600,
                 poll_s: float = 2.0):
        from pickups.app.comfy import Connection

        self.store = store_mod.Store(store_mod.ROOT)
        self.conn = Connection.from_settings()
        self.server, self.wait, self.idle_timeout_s, self.poll_s = server.rstrip("/"), wait, idle_timeout_s, poll_s
        self.project = project
        self.lib_file = results / "_library.json"
        self.lib = _read_json(self.lib_file) or {}
        self.others = True  # free ComfyUI's memory before the first cell, and after anyone else used the GPU
        self.cancel_wait_s = 120.0  # after a timeout: how long a cancel may take before ComfyUI counts as stuck
        self.job = None
        self._jm = None

    @property
    def jm(self):
        if self._jm is None:  # only when a cell renders: rescoring and judging never start a GPU worker
            from pickups.app.jobs import JobManager
            from pickups.tasks import TASKS

            self._jm = JobManager(self.store, self.conn, TASKS)
        return self._jm

    def _save_lib(self) -> None:
        self.lib_file.parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.lib_file, self.lib)

    def project_id(self) -> str:
        known = self.lib.setdefault("projects", {})
        pid = known.get(self.project)
        if not (pid and self.store.project_exists(pid)):
            pid = next((m["id"] for f in sorted((self.store.root / "projects").glob("*/meta.json"))
                        if (m := _read_json(f)) and m.get("name") == self.project and not m.get("archived")), None)
            pid = pid or self.store.create_project(self.project)["id"]
            known[self.project] = pid
            self._save_lib()
        return pid

    def shot_for(self, spec: dict) -> str:
        """The library shot a matrix shot runs on: by default a copy in the stress project, so the user's own shot gets no
        stress takes ("copy": false uses it as it is)."""
        if spec.get("shot_id") and spec.get("copy") is False:
            self.store.shot_meta(spec["shot_id"])
            return spec["shot_id"]
        key = f"{spec['id']}|{spec.get('shot_id') or spec.get('path')}"
        sid = self.lib.setdefault("shots", {}).get(key)
        if sid and (self.store.root / "shots" / sid / "meta.json").is_file():
            return sid
        if spec.get("shot_id"):
            src = self.store.source_path(spec["shot_id"])
        else:
            src = Path(spec["path"]).expanduser()
            if not src.is_file():
                raise FileNotFoundError(f"Can't find the video for {spec['id']} ({src.name}).")
        sid = self.store.add_shot(src, spec.get("name") or spec["id"], False, self.project_id())["id"]
        self.lib["shots"][key] = sid
        self._save_lib()
        return sid

    def upload(self, path: Path) -> str:
        """As POST /api/uploads (Store.add_upload); the same file is uploaded once across runs."""
        data = Path(path).read_bytes()
        h = hashlib.sha1(data).hexdigest()
        uploads = self.lib.setdefault("uploads", {})
        if uploads.get(h):
            try:
                self.store.upload_path(uploads[h])
                return uploads[h]
            except FileNotFoundError:
                pass
        uploads[h] = self.store.add_upload(data, Path(path).name)["id"]
        self._save_lib()
        return uploads[h]

    def server_busy(self) -> bool:
        try:
            return any(j.get("status") in ("queued", "running") for j in _get_json(f"{self.server}/api/jobs", 5))
        except Exception:  # noqa: BLE001 - no server running: nothing of the app's is on the GPU
            return False

    def comfy_state(self) -> str | None:
        try:
            q = _get_json(f"{self.conn.comfy_url}/queue", 5)
        except Exception:  # noqa: BLE001
            return "ComfyUI isn't answering"
        return "ComfyUI is busy with other work" if q.get("queue_running") or q.get("queue_pending") else None

    def wait_idle(self, log) -> float:
        """Wait until neither the app nor anyone else is using ComfyUI. Returns the seconds waited."""
        t0, said = time.monotonic(), None
        while self.wait:
            why = ("the app is rendering" if self.server_busy() else None) or self.comfy_state()
            if not why:
                break
            self.others = True
            if time.monotonic() - t0 > self.idle_timeout_s:
                raise StopRun(f"Waited {fmt_dur(time.monotonic() - t0)} for the GPU ({why}).")
            if why != said:
                log(f"waiting: {why}")
                said = why
            time.sleep(30)
        if self.others:
            self.conn.free_memory()
            self.others = False
        return time.monotonic() - t0

    def run(self, shot_id: str, task: str, params: dict, timeout_s: float, log) -> dict:
        t0 = time.monotonic()
        job = self.job = self.jm.submit(shot_id, task, params)
        eta = int(job.get("eta_s") or 0)
        limit = max(float(timeout_s), 3.0 * eta)
        stage, contended, next_look = None, False, 0.0
        out = {"job_id": job["id"], "eta_s": eta}
        while job["status"] in ("queued", "running"):
            if job["status"] == "running" and job.get("stage") != stage:
                stage = job["stage"]
                log(stage)
            now = time.monotonic()
            if now >= next_look:
                contended = contended or self.server_busy()
                next_look = now + 30
            if now - t0 > limit:
                stuck = not self._cancel(job, self.cancel_wait_s)
                self.job = None
                return out | {"status": "timeout", "error": f"Didn't finish in {fmt_dur(limit)}.", "failed_stage": stage,
                              "run_s": round(time.monotonic() - t0, 1), "contended": contended, "stuck": stuck}
            time.sleep(self.poll_s)
        self.job = None  # kept until here so Ctrl-C can cancel it
        out |= {"run_s": round(time.monotonic() - t0, 1), "contended": contended}
        if job["status"] == "done":
            return out | {"status": "done", "take_id": job["result_version_id"]}
        return out | {"status": "failed", "error": job.get("error") or ("Canceled." if job["status"] == "canceled" else ""),
                      "failed_stage": stage}

    def _cancel(self, job: dict, wait_s: float) -> bool:
        """Cancel and wait for it to land (at ComfyUI's next progress event). False: it didn't stop in time."""
        self.jm.cancel(job["id"])
        end = time.monotonic() + wait_s
        while job["status"] in ("queued", "running") and time.monotonic() < end:
            time.sleep(1)
        return job["status"] not in ("queued", "running")

    def interrupt(self) -> None:
        job, self.job = self.job, None
        if job and job["status"] in ("queued", "running"):
            self._cancel(job, 15)  # so the prompt on ComfyUI is cancelled too, not left running

    def take(self, shot_id: str, take_id: str) -> tuple[dict, Path]:
        return self.store.version_meta(shot_id, take_id), self.store.version_dir(shot_id, take_id)


# --- the run ------------------------------------------------------------------------------------------------------
def base_result(c: dict, m: dict) -> dict:
    ch = c.get("character") or {}
    return {"cell": c["id"], "stage": m["stage"], "shot": c["shot"]["id"], "shot_name": c["shot"].get("name") or c["shot"]["id"],
            "character": ch.get("name"), "character_id": ch.get("id"), "task": c["task"], "seed": c["seed"],
            "params": c["params"], "expect": c.get("expect") or {}}


def title(r: dict) -> str:
    return " · ".join(x for x in (r.get("shot_name"), r.get("character"), r.get("task", "").capitalize(), f"seed {r.get('seed')}") if x)


def post(r: dict, runner, d: Path, judge_cfg: dict | None) -> dict:
    """Scores, pictures and (with a judge) a verdict for a cell's finished take. No GPU. A take that has gone from the
    library (deleted in the app) makes the cell an error, so the run carries on."""
    try:
        meta, tdir = runner.take(r["shot_id"], r["take_id"])
    except (KeyError, OSError) as e:  # FileNotFoundError included
        r |= {"status": "error", "error": "The take is gone from the library. Run with --retry-failed or --force to make a new one.",
              "error_detail": str(e)[:300]}
        return r
    det = meta.get("details") or {}
    r["qa"] = read_qa(det)
    raw = det.get("_qa") if isinstance(det.get("_qa"), dict) else {}
    r["how"] = {k: raw[k] for k in HOW_KEYS if k in raw}
    r["outside_selection"] = det.get("pixels_changed_outside_selection")
    r["timings"] = (r.get("timings") or {}) | {"gpu_s": det.get("gpu_seconds"), "take_s": det.get("total_seconds")}
    ref = Path(r["reference"]) if r.get("reference") else None
    if (ref is None or not ref.is_file()) and (tdir / "reference.png").is_file():
        ref = tdir / "reference.png"  # a still Recast made from a description
    try:
        r["artefacts"] = artefacts(tdir, d, title(r), ref)
        r.pop("artefact_error", None)
    except Exception as e:  # noqa: BLE001 - the take still counts; the report says the pictures are missing
        r["artefact_error"] = str(e)[:300]
    a = r.get("artefacts") or {}
    if judge_cfg and r.get("character") and a.get("reference") and a.get("judge_frames"):
        r["judge"] = judge(judge_cfg, d / a["reference"], [d / f for f in a["judge_frames"]])
    return r


def run_cell(c: dict, runner, d: Path, m: dict, judge_cfg: dict | None, log) -> dict:
    r = base_result(c, m) | {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code_stamp()}
    _write_json(d / "result.json", r)
    try:
        ref = reference(c.get("character"))
        params = resolve_uploads(runner, dict(c["params"]), Path(m["_base"]))
        if ref is not None:
            params["ref_upload_id"] = c["character"].get("ref_upload_id") or runner.upload(ref)
            r["reference"] = str(ref)
        r["shot_id"] = runner.shot_for(c["shot"])
        r["params"] = params
        r["timings"] = {"queue_wait_s": round(runner.wait_idle(log), 1)}
        _write_json(d / "result.json", r)
        out = runner.run(r["shot_id"], c["task"], params, float(m.get("cell_timeout_s", 2400)), log)
    except StopRun:
        r["status"] = "interrupted"
        _write_json(d / "result.json", r)
        raise
    except Exception as e:  # noqa: BLE001 - record it and carry on with the next cell
        r |= {"status": "error", "error": str(e)[:500], "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        return r
    r["timings"] = (r.get("timings") or {}) | {"eta_s": out.get("eta_s"), "run_s": out.get("run_s")}
    r |= {k: out[k] for k in ("status", "error", "take_id", "job_id", "failed_stage", "contended", "stuck") if out.get(k) is not None}
    r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if r["status"] == "done":
        post(r, runner, d, judge_cfg)
    return r


def run_stage(m: dict, chosen: list[dict], cells: list[dict], out: Path, runner, o, judge_cfg: dict | None = None, log=print) -> int:
    """Each chosen cell in order; finished cells are skipped. Returns 0, or 2 when the run had to stop (run again to resume)."""
    ran, current = 0, None
    try:
        for i, c in enumerate(chosen, 1):
            d = out / c["id"]
            prev = _read_json(d / "result.json")
            act = decide(prev, o)
            if act is None:
                continue
            if act == "run" and o.limit and ran >= o.limit:
                log(f"Stopping after {o.limit} cell{'s' if o.limit != 1 else ''} (--limit).")
                break

            def say(text, c=c, i=i):
                log(f"{time.strftime('%H:%M:%S')}  [{i}/{len(chosen)}] {c['id']}  {text}")

            d.mkdir(parents=True, exist_ok=True)
            current = d
            if act == "run":
                ran += 1
                say("starting")
                r = run_cell(c, runner, d, m, judge_cfg, say)
            else:
                say("scoring the take it has")
                r = post(prev, runner, d, judge_cfg)
            r["grade"], r["reasons"], _ = grade(r, m["thresholds"])
            _write_json(d / "result.json", r)
            current = None
            say(f"{r['status']} · {r['grade']}" + (f" · {r['reasons'][0]}" if r["reasons"] else ""))
            write_report(out, m, cells)
            if r.get("stuck"):
                raise StopRun("ComfyUI looks stuck: restart it on the GPU machine.")
    except StopRun as e:
        log(f"Stopped: {e} Run the same command again to carry on.")
        write_report(out, m, cells)
        return 2
    except KeyboardInterrupt:
        runner.interrupt()
        if current is not None and (r := _read_json(current / "result.json")) and r.get("status") == "running":
            _write_json(current / "result.json", r | {"status": "interrupted"})
        write_report(out, m, cells)
        log("Interrupted. Run the same command again to carry on.")
        return 130
    write_report(out, m, cells)
    return 0


# --- the report ---------------------------------------------------------------------------------------------------
CSV_COLUMNS = ["stage", "cell", "shot", "character", "task", "seed", "status", "grade", "reasons", "shot_id", "take_id", "coverage",
               "halo", "flicker", "empty_new_frames", "outside_selection", "run_s", "eta_s", "gpu_s", "queue_wait_s", "contended",
               "identity_match", "artefacts", "judge_notes", "error", "code", *HOW_KEYS]


def how_text(r: dict) -> str:
    """One plain line on how Recast made the take, e.g. "Figure cut out from its plain backdrop · SCAIL-2 at 832 px"."""
    h = r.get("how") or {}
    bits = [str(h["reference"])[:1].upper() + str(h["reference"])[1:]] if h.get("reference") else []
    if h.get("reference_mask") is False:
        bits.append("SAM's mask of the reference (ComfyUI refused ours)")
    n = h.get("people_on_first_frame")
    if isinstance(n, int) and n > 1:
        idx = str(h.get("object_indices", ""))
        bits.append(f"{n} people in the crop, " + (f"only number {int(idx) + 1} from the left re-rendered" if idx.isdigit() else "all re-rendered"))
        if h.get("new_track") is not None:
            bits.append("the new character followed on their own" if h["new_track"] else "couldn't follow the new character on their own")
    if h.get("scail_long"):
        bits.append(f"SCAIL-2 at {h['scail_long']} px")
    return " · ".join(bits)


def collect(out: Path, m: dict, cells: list[dict]) -> list[dict]:
    rows = []
    for c in cells:
        r = _read_json(out / c["id"] / "result.json") or (base_result(c, m) | {"status": "pending"})
        r["grade"], r["reasons"], r["checks"] = grade(r, m["thresholds"])
        rows.append(r)
    return rows


def _data_uri(path: Path) -> str | None:
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def _num(c: dict) -> str:
    v, k = c["value"], c["key"]
    if v is None:
        return "–"
    if k == "time":
        return fmt_dur(v)  # the limit is in the tooltip; the colour says pass or fail
    if k == "outside":
        return f"{int(v):,} px"
    if k == "empty_new_frames":
        return str(int(v))
    return f"{v:.2f}"


def _stage_name(stage: str) -> str:
    return re.sub(r"^stage-?(\d+)$", r"Stage \1", stage)


CSS = """
:root{--bg:#12100e;--panel:#1b1815;--line:#2e2a24;--ink:#efe9dd;--muted:#a39b8d;--a:#55c27f;--b:#e3a63b;--f:#ef4a3f;--p:#6f685d;
--mono:ui-monospace,"Geist Mono",SFMono-Regular,Menlo,monospace;--sans:"Instrument Sans",-apple-system,"Segoe UI",system-ui,sans-serif}
@media (prefers-color-scheme:light){:root{--bg:#f6f2ea;--panel:#fffdf8;--line:#e2dccf;--ink:#1d1a16;--muted:#6d665b;--p:#a9a194}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 var(--sans)}
a{color:inherit}
header{padding:40px 28px 8px;max-width:1500px;margin:0 auto}
.eyebrow{font:12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;display:flex;align-items:center;gap:8px}
.eyebrow:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--f)}
h1{font:400 40px/1.1 "Instrument Serif",Georgia,serif;margin:0 0 6px}
h1 span{color:var(--muted)}
.about{color:var(--muted);max-width:72ch;margin:6px 0 20px}
.tally{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 16px}
.t{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 16px;min-width:96px}
.t b{display:block;font:500 28px/1 var(--sans)}.t span{font-size:12px;color:var(--muted)}
.t.A b{color:var(--a)}.t.B b{color:var(--b)}.t.F b{color:var(--f)}.t.P b{color:var(--p)}
.meta,.rules{font:12px/1.6 var(--mono);color:var(--muted);margin:0}
.overview{max-width:1500px;margin:18px auto 0;padding:0 28px;overflow-x:auto}
table{border-collapse:collapse;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;white-space:nowrap}
th{font-weight:500;color:var(--muted)}
.chip{display:inline-grid;place-items:center;min-width:24px;height:24px;border-radius:6px;font:600 12px var(--mono);
text-decoration:none;margin-right:3px;color:#12100e}
.g-A .badge,.chip.g-A{background:var(--a)}.g-B .badge,.chip.g-B{background:var(--b)}.g-F .badge,.chip.g-F{background:var(--f)}
.g-P .badge,.chip.g-P{background:var(--p)}
.filters{max-width:1500px;margin:20px auto 0;padding:0 28px;display:flex;gap:8px;flex-wrap:wrap}
.filters button{font:13px var(--sans);color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:6px 14px;cursor:pointer}.filters button.on{background:var(--ink);color:var(--bg)}
main{max-width:1500px;margin:16px auto 60px;padding:0 28px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(100%,380px),1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.pic{position:relative;display:block;background:#000;min-height:120px}
.pic img{width:100%;height:auto;display:block}
.pic .none{position:absolute;inset:0;display:grid;place-items:center;color:var(--p);font:12px var(--mono)}
.badge{position:absolute;top:10px;left:10px;width:38px;height:38px;border-radius:10px;display:grid;place-items:center;
font:700 20px var(--sans);color:#12100e}
.body{padding:14px 16px 16px;display:flex;flex-direction:column;gap:10px;flex:1}
.head{display:flex;gap:12px;align-items:flex-start;justify-content:space-between}
h3{margin:0;font:500 18px/1.2 var(--sans)}.sub{margin:2px 0 0;color:var(--muted);font-size:13px}
.ref{width:44px;height:64px;object-fit:cover;border-radius:6px;border:1px solid var(--line);flex:none}
dl{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:0}
dl div{background:var(--bg);border-radius:8px;padding:6px 8px;min-width:0}
dt{font-size:11px;color:var(--muted)}dd{margin:0;font:500 13px var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ok dd{color:var(--a)}.bad dd{color:var(--f)}.na dd{color:var(--p)}
.why{margin:0;padding-left:18px;font-size:13px}.why li{margin:2px 0}
.err{margin:0;font:12px/1.5 var(--mono);color:var(--f);background:var(--bg);border-radius:8px;padding:8px 10px;white-space:pre-wrap;
word-break:break-word;max-height:9em;overflow:auto}
.links{margin-top:auto;display:flex;gap:12px;flex-wrap:wrap;font-size:13px;align-items:center}
.links code{font:11px var(--mono);color:var(--muted);margin-left:auto}
body:not([data-filter=all]) .card{display:none}
body[data-filter=A] .g-A,body[data-filter=B] .g-B,body[data-filter=F] .g-F,body[data-filter=P] .g-P{display:flex}
@media (max-width:600px){header,main,.filters,.overview{padding-left:16px;padding-right:16px}h1{font-size:30px}dl{grid-template-columns:repeat(2,1fr)}}
"""

JS = """
document.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{
document.body.dataset.filter=b.dataset.f;document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b))}));
"""


def _g(grade: str) -> str:
    return "P" if grade == "–" else grade  # the CSS class of a grade ("P": not run yet)


def _card(r: dict, out: Path, th: dict) -> str:
    e, cid = html.escape, r["cell"]
    a = r.get("artefacts") or {}
    thumb = _data_uri(out / cid / (a.get("thumb") or "contact_thumb.jpg"))
    pic = f'<img src="{thumb}" alt="Contact sheet">' if thumb else '<span class="none">no pictures yet</span>'
    ref = None
    if a.get("reference"):
        p = out / cid / a["reference"]
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((120, 180), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            ref = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except OSError:
            pass
    nums = "".join(f'<div class="{"na" if c["ok"] is None else "ok" if c["ok"] else "bad"}"><dt>{e(c["label"])}</dt>'
                   f'<dd title="limit {e(fmt_dur(c["limit"]) if c["key"] == "time" else str(c["limit"]))}">{e(_num(c))}</dd></div>'
                   for c in r.get("checks", []))
    j = r.get("judge") or {}
    if j.get("identity_match"):
        lo = min(j["identity_match"], j["artefacts"])
        cls = "bad" if lo <= th["judge_fail"] else "ok" if lo >= th["judge_a"] else "na"
        nums += (f'<div class="{cls}"><dt>Identity</dt><dd>{j["identity_match"]}/5</dd></div>'
                 f'<div class="{cls}"><dt>Artefacts</dt><dd>{j["artefacts"]}/5</dd></div>')
    reasons, err = list(r.get("reasons", [])), ""
    if r.get("error") and r.get("status") in ("failed", "timeout", "error") and reasons:
        reasons[0] = reasons[0].replace(f": {r['error']}", ".")  # the message itself goes in the box below
        err = f'<pre class="err">{e(r["error"])}</pre>'
    why = "".join(f"<li>{e(x)}</li>" for x in reasons)
    links = "".join(f'<a href="{e(cid)}/{e(a[k])}">{label}</a>' for k, label in
                    (("contact", "Contact sheet"), ("sbs", "Side by side"), ("gif", "GIF"), ("reference", "Reference")) if a.get(k))
    take = f'<code>{e(r.get("take_id") or "")}</code>' if r.get("take_id") else ""
    sub = " · ".join(x for x in (r.get("shot_name"), r.get("task", "").capitalize(), f"seed {r.get('seed')}") if x)
    return (f'<article class="card g-{_g(r["grade"])}" id="{e(cid)}"><a class="pic" href="{e(cid)}/contact.jpg">{pic}'
            f'<span class="badge">{e(r["grade"])}</span></a><div class="body"><div class="head"><div>'
            f'<h3>{e(r.get("character") or r.get("task", "").capitalize())}</h3><p class="sub">{e(sub)}</p>'
            + (f'<p class="sub">{e(how_text(r))}</p>' if how_text(r) else "") + "</div>"
            + (f'<img class="ref" src="{ref}" alt="Reference">' if ref else "") + "</div>"
            + (f"<dl>{nums}</dl>" if nums else "") + (f'<ul class="why">{why}</ul>' if why else "") + err
            + f'<div class="links">{links}{take}</div></div></article>')


def _overview(rows: list[dict]) -> str:
    shots = list(dict.fromkeys(r["shot_name"] for r in rows))
    who = list(dict.fromkeys(r.get("character") or "—" for r in rows))
    e = html.escape
    head = "".join(f"<th>{e(w)}</th>" for w in who)
    body = ""
    for s in shots:
        tds = ""
        for w in who:
            chips = "".join(f'<a class="chip g-{_g(r["grade"])}" href="#{e(r["cell"])}" title="{e(r["cell"])}">{e(r["grade"])}</a>'
                            for r in rows if r["shot_name"] == s and (r.get("character") or "—") == w)
            tds += f"<td>{chips}</td>"
        body += f"<tr><th>{e(s)}</th>{tds}</tr>"
    return f'<section class="overview"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></section>'


def render_report(out: Path, m: dict, rows: list[dict]) -> str:
    e, th = html.escape, m["thresholds"]
    n = {g: sum(r["grade"] == g for r in rows) for g in ("A", "B", "F", "–")}
    tally = "".join(f'<div class="t {k}"><b>{n[g]}</b><span>{label}</span></div>'
                    for g, k, label in (("A", "A", "A · clean"), ("B", "B", "B · passes, check"), ("F", "F", "F · failed"),
                                        ("–", "P", "not run yet")))
    gpu = sum(float((r.get("timings") or {}).get("gpu_s") or 0) for r in rows)
    wall = sum(float((r.get("timings") or {}).get("run_s") or 0) for r in rows)
    judges = sorted({(r.get("judge") or {}).get("model") for r in rows} - {None})
    codes = sorted({r.get("code") for r in rows} - {None})
    meta = [f"Updated {time.strftime('%d %b %Y %H:%M')}", f"{len(rows)} cells", f"render time {fmt_dur(wall)}", f"GPU {fmt_dur(gpu)}"]
    if codes:
        meta.append("code from " + (codes[0] if len(codes) == 1 else f"{codes[0]} to {codes[-1]}"))
    meta.append("judge: " + ", ".join(judges) if judges else "not judged")
    rules = (f"Coverage {th['coverage_min']}–{th['coverage_max']} · Halo &lt; {th['halo_max']} · Flicker &lt; {th['flicker_max']} · "
             f"Empty frames {th['empty_new_frames_max']} · Outside the selection {th['outside_max']} px · "
             f"Time ≤ {th['eta_factor']} × estimate · Judge ≥ {th['judge_a']} for A, ≤ {th['judge_fail']} fails")
    filters = "".join(f'<button data-f="{f}"{" class=on" if f == "all" else ""}>{label} {count}</button>' for f, label, count in
                      (("all", "All", len(rows)), ("A", "A", n["A"]), ("B", "B", n["B"]), ("F", "F", n["F"]), ("P", "Not run", n["–"])))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,"
            f"initial-scale=1\"><title>Stress test · {e(_stage_name(m['stage']))}</title><style>{CSS}</style></head>"
            f'<body data-filter="all"><header><p class="eyebrow">Pickups stress test</p>'
            f"<h1>{e(_stage_name(m['stage']))} <span>{e(m.get('title') or '')}</span></h1>"
            + (f'<p class="about">{e(m["about"])}</p>' if m.get("about") else "")
            + f'<div class="tally">{tally}</div><p class="meta">{e(" · ".join(meta))}</p><p class="rules">{rules}</p></header>'
            + (_overview(rows) if rows else "") + f'<nav class="filters">{filters}</nav><main>'
            + "".join(_card(r, out, th) for r in rows) + f"</main><script>{JS}</script></body></html>")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            qa, t, j = r.get("qa") or {}, r.get("timings") or {}, r.get("judge") or {}
            w.writerow({"stage": r.get("stage"), "cell": r["cell"], "shot": r.get("shot"), "character": r.get("character") or "",
                        "task": r.get("task"), "seed": r.get("seed"), "status": r.get("status"), "grade": r["grade"],
                        "reasons": " | ".join(r.get("reasons", [])), "shot_id": r.get("shot_id") or "", "take_id": r.get("take_id") or "",
                        **{k: qa.get(k, "") for k in ("coverage", "halo", "flicker", "empty_new_frames")},
                        "outside_selection": "" if r.get("outside_selection") is None else r["outside_selection"],
                        **{k: "" if t.get(k) is None else t[k] for k in ("run_s", "eta_s", "gpu_s", "queue_wait_s")},
                        "contended": bool(r.get("contended")), "identity_match": j.get("identity_match", ""),
                        "artefacts": j.get("artefacts", ""), "judge_notes": j.get("notes") or j.get("error") or "",
                        "error": r.get("error") or "", "code": r.get("code") or "",
                        **{k: "" if (r.get("how") or {}).get(k) is None else r["how"][k] for k in HOW_KEYS}})


def write_report(out: Path, m: dict, cells: list[dict]) -> list[dict]:
    rows = collect(out, m, cells)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(render_report(out, m, rows))
    write_csv(out / "report.csv", rows)
    return rows


# --- dry run ------------------------------------------------------------------------------------------------------
def describe_plan(m: dict, chosen: list[dict], out: Path, o, notes: list[str]) -> list[str]:
    """The plan in words, read-only: what each cell would do and anything missing."""
    lib = store_mod.ROOT
    known = (_read_json(out.parent / "_library.json") or {}).get("shots", {})
    lines = [f"{_stage_name(m['stage'])}: {m.get('title') or ''} ({len(chosen)} cell{'s' if len(chosen) != 1 else ''}; results in {out})"]
    lines += [f"  note: {n}" for n in notes]
    for s in m.get("shots", []):
        key = f"{s['id']}|{s.get('shot_id') or s.get('path')}"
        if s.get("shot_id"):
            ok = (lib / "shots" / s["shot_id"] / "meta.json").is_file()
            how = ("uses it as it is" if s.get("copy") is False else
                   "copy made earlier" if known.get(key) else f"copies it into the “{m.get('project', 'Stress test')}” project")
            lines.append(f"  shot {s['id']}: library shot {s['shot_id']} ({'found' if ok else 'MISSING'}), {how}")
        else:
            ok = Path(s.get("path", "")).expanduser().is_file()
            lines.append(f"  shot {s['id']}: {Path(s.get('path', '')).name} ({'found' if ok else 'MISSING'})")
    for c in chosen:
        act = decide(_read_json(out / c["id"] / "result.json"), o) or "skip"
        try:
            ref = reference(c.get("character"))
            who = f"ref {ref.name}" if ref else ("FLUX 3 klein still from the description" if c.get("character") else "")
        except FileNotFoundError as e:
            who = f"MISSING: {e}"
        lines.append(f"  {act:<5} {c['id']}  {who}")
    if chosen:
        lines.append("  params of the first cell: " + json.dumps(chosen[0]["params"], ensure_ascii=False)[:600])
    return lines


def main(argv: list[str] | None = None, runner=None) -> int:
    ap = argparse.ArgumentParser(description="Run a stress matrix through Pickups and grade every take.")
    ap.add_argument("matrix", type=Path)
    ap.add_argument("--out", type=Path, help="results folder (default: results/ next to the matrix)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan; change nothing")
    ap.add_argument("--only", default="", help="comma-separated cell id patterns, e.g. 'skye-walker__alien*'")
    ap.add_argument("--retry-failed", action="store_true", help="run failed, timed-out and unstartable cells again")
    ap.add_argument("--force", action="store_true", help="render every chosen cell again (new takes)")
    ap.add_argument("--rescore", action="store_true", help="redo pictures, scores (and verdicts) of finished takes; no GPU")
    ap.add_argument("--report-only", action="store_true", help="rebuild report.html and report.csv only")
    ap.add_argument("--judge", action="store_true", help="score each take with the local vision model (LM Studio)")
    ap.add_argument("--judge-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0, help="render at most this many cells this time")
    ap.add_argument("--server", default="http://127.0.0.1:7788", help="the app, to wait for its queue")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the app and ComfyUI to be idle")
    ap.add_argument("--idle-timeout", type=float, default=4 * 3600, help="seconds to wait for an idle GPU before stopping")
    o = ap.parse_args(argv)

    m, notes = load_matrix(o.matrix)
    cells = plan(m)
    chosen = select(cells, o.only)
    root = (o.out or Path(m["_base"]) / "results").resolve()
    out = root / m["stage"]
    if o.dry_run:
        print("\n".join(describe_plan(m, chosen, out, o, notes)))
        return 0
    for n in notes:
        print("note:", n)
    if o.report_only:
        rows = write_report(out, m, cells)
        print(f"{out / 'report.html'} ({len(rows)} cells)")
        return 0
    judge_cfg = None
    if o.judge:
        judge_cfg = judge_config(o.judge_url, o.judge_model, o.judge_timeout)
        why = judge_ready(judge_cfg)
        if why:
            print(f"Judge off for this run: {why}. Run with --judge again later to judge these takes.")
            judge_cfg = None
    runner = runner or AppRunner(root, m.get("project", "Stress test"), o.server, not o.no_wait, o.idle_timeout)
    code = run_stage(m, chosen, cells, out, runner, o, judge_cfg)
    print(out / "report.html")
    return code


if __name__ == "__main__":
    sys.exit(main())
