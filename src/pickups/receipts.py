"""Receipt sheets: one take on one printable page, the receipt you hand a client.

take_page() renders a single self-contained HTML page (inline CSS, the bundled OFL fonts and JPEG stills embedded as
base64, a Content-Security-Policy that allows no requests at all): what was done, before and after at the same frame,
how much of the picture is still the original footage, how it was made, the models and their licences, the prompts
each model received (read from the saved ComfyUI workflows), the source timecode, the delivered files, notes, and the
workflows that made it. project_zip() bundles a project's approved takes: index.html, takes/<name>.html and
workflows/<name>/<NN>_<title>.json.

Pages name files, never paths on this computer. Stills are cached next to the take (receipt_*.jpg).
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from pickups import licences, media, nle
from pickups.app.store import UNSORTED, Store, _safe
from pickups.app.workflow_titles import step_title

FONTS = Path(__file__).parent / "assets" / "fonts"
STILL_W = 1280
EMBED_MAX = 512 * 1024  # a workflow larger than this isn't embedded in a single page (the zip always has it)

TASK_VERB = {"remove": "Remove a person", "recast": "Replace a person", "environment": "Change the weather", "previz": "Previz to final",
             "reshoot": "Reshoot", "finish": "Finish"}
SEEDED = {"remove", "recast", "environment", "previz"}  # tasks that seed with 42 unless told otherwise
DELIVERY_SIZE = {"UHD": "3840×2160", "HD": "1920×1080"}
DETAIL_LABELS = {"method": "Method", "camera": "Camera", "input": "Input", "performance_size": "Performance size",
                 "change": "Change", "control": "Structure", "kept_as_shot": "Kept as shot", "grain": "Film grain", "hdr": "HDR", "quality": "Quality",
                 "move": "Move", "references": "References", "pinned_first_frame": "First frame locked",
                 "look_at": "Look frame at", "note": "Note"}
VALUE_LABELS = {
    "change": {"snow": "Snow", "rain": "Rain", "fog": "Fog", "night": "Night", "golden_hour": "Golden hour", "autumn": "Autumn",
               "custom": "Custom"},
    "control": {"depth": "Depth (surfaces change)", "canny": "Edges (light only)"},
    "grain": {"off": "Off", "fine": "Fine", "35mm": "35mm"},
}
# Older takes (or ones without workflows): the prompts their details and settings kept.
FALLBACK_PROMPTS = (("text", "Selection"), ("fill", "Fill"), ("reference_prompt", "Reference still"),
                    ("look_prompt", "Look prompt"), ("prompt", "Video prompt"), ("prompt_sent", "Prompt"))
SELECTION = re.compile(r"^(Pick|Track|Find|Shadow)\b")
PRINT_JS = 'document.getElementById("print").addEventListener("click",function(){window.print()});'

e = html.escape


# --- what a receipt says ---------------------------------------------------------------------------------------------
# A receipt's shares of the picture, in the bar's order: (key, label, CSS class). "upscaled" is Finish's upscale: every
# pixel resampled from your footage (new detail, no new content), so it is its own share and never counts as AI.
SHARES = (("original", "Original", "o"), ("borrowed", "Other frames", "b"), ("upscaled", "Upscaled", "u"), ("ai", "AI", "a"))
HDR_SHARE = re.compile(r"\bon ([\d.]+)% of pixels")  # Finish's details["hdr"]: "highlights reconstructed … on 13.24% of pixels"


def finish_receipt(upscaled: bool, highlights: float = 0.0, upscaler: str | None = None, scale: float | None = None) -> dict:
    """A Finish take's receipt. `highlights`: the % of pixels whose HDR highlights LTX-2.5 rebuilt (0 without HDR), the
    only generated share. With an upscale nothing is the untouched original, and the rest is upscaled from it:
    {"original": 0, "borrowed": 0, "upscaled": 100 - hl, "ai": hl, "upscaler", "scale"}. Without one:
    {"original": 100 - hl, "borrowed": 0, "ai": hl}."""
    hl = round(min(100.0, max(0.0, float(highlights or 0))), 2)
    if not upscaled:
        return {"original": round(100 - hl, 2), "borrowed": 0.0, "ai": hl}
    out = {"original": 0.0, "borrowed": 0.0, "upscaled": round(100 - hl, 2), "ai": hl}
    if upscaler:  # unknown on an older take that didn't name its models: the wording then leaves it out
        out["upscaler"] = upscaler
    if scale and scale > 0:
        out["scale"] = round(float(scale), 2)
    return out


def _wh(v) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", v, flags=re.I) if isinstance(v, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def finish_receipt_on_read(receipt: dict | None, details: dict | None) -> dict | None:
    """Finish takes made before upscales had a share of their own said "100% AI" whenever they upscaled. Read them as
    finish_receipt() writes them now, from what their details recorded: the HDR share (details["hdr"]), the upscaler
    (the first model that isn't LTX) and the factor (output over input width). Any other receipt is returned as it is."""
    r, d = receipt or {}, details if isinstance(details, dict) else {}
    try:
        if not isinstance(r, dict) or "upscaled" in r or float(r.get("ai") or 0) < 100 or \
                float(r.get("original") or 0) > 0 or float(r.get("borrowed") or 0) > 0:
            return receipt
        hit = HDR_SHARE.search(str(d.get("hdr") or ""))
        hl = float(hit.group(1)) if hit else 0.0
    except (TypeError, ValueError):  # a receipt Pickups didn't write: shown as it is
        return receipt
    models = [m for m in d.get("models") if isinstance(m, str) and m and not m.startswith("LTX")] if isinstance(d.get("models"), list) else []
    size_in, size_out = _wh(d.get("input")), _wh(d.get("rendered_at")) or _wh(d.get("output")) or _wh(d.get("resolution"))
    scale = size_out[0] / size_in[0] if size_in and size_out and size_in[0] else None
    return finish_receipt(True, hl, re.sub(r"\s*\([^)]*\)$", "", models[0]) if models else None, scale if scale and scale > 1 else None)


def compose(own: dict | None, source: dict | None, take: str, label: str = "") -> dict | None:
    """A Finish take made from another take (`take`: its number, "02"), as shares of the shot's original footage: the
    input's pixels go through the Finish, so its other frames and AI stay what they were. With an upscale:
    ai = hl + (100 - hl) × in.ai, borrowed = (100 - hl) × in.borrowed, upscaled = the rest (hl: the % of pixels whose
    HDR highlights LTX-2.5 rebuilt, assumed spread evenly over the input's shares). Without one, the input's original,
    other frames and AI each scale by (100 - hl) and hl is added to the AI. Adds "highlights" (hl, for the wording) and
    "source": {"take", "label", "receipt"} (the input's shares). Read time only: Store.shot() calls it, nothing is
    written. Without the input's receipt the shares stay as they are and only the source is named."""
    if not isinstance(own, dict):
        return own
    out = dict(own)
    src = {"take": take, "label": label}
    ins = [float(source.get(k) or 0) for k, _, _ in SHARES] if isinstance(source, dict) else []
    if sum(ins) > 0:
        i_o, i_b, i_u, i_a = (x / sum(ins) for x in ins)
        o, b, u, a = (float(own.get(k) or 0) for k, _, _ in SHARES)
        upscaled = o * i_u + u * (i_o + i_u)
        out.update(original=round(o * i_o, 2), borrowed=round(o * i_b + b * (i_o + i_b + i_u) + u * i_b, 2),
                   ai=round(a + (o + b + u) * i_a, 2), highlights=round(a, 2))
        if "upscaled" in own or upscaled > 0:
            out["upscaled"] = round(upscaled, 2)
        if "upscaled" not in own and upscaled > 0:  # this take didn't upscale; its input did
            out["source_upscaled"] = True
            out.update({k: source[k] for k in ("upscaler", "scale") if source.get(k)})
        src["receipt"] = {k: round(float(source.get(k) or 0), 2) for k, _, _ in SHARES if k in source}
    out["source"] = src
    return out


def percents(r: dict | None) -> tuple[int, int, int, int] | None:
    """(original, other frames, upscaled, AI) as whole percentages adding up to 100, the way the app shows a receipt."""
    if not r:
        return None
    vals = [float(r.get(k) or 0) for k, _, _ in SHARES]
    total = sum(vals)
    if total <= 0:
        return None
    scaled = [v / total * 100 for v in vals]
    out = [int(v) for v in scaled]
    for i in sorted(range(len(out)), key=lambda i: scaled[i] - out[i], reverse=True)[:100 - sum(out)]:
        out[i] += 1
    return out[0], out[1], out[2], out[3]


def percent_texts(r: dict | None) -> tuple[str, str, str, str] | None:
    """percents() as shown: a share that is there but rounds to nothing reads "<1%", and a share rounded up to 100%
    while another is there reads ">99%" (the app's receiptShares() does the same)."""
    p = percents(r)
    if not p:
        return None
    raw = [float(r.get(k) or 0) for k, _, _ in SHARES]
    out = []
    for i, v in enumerate(p):
        if v == 0 and raw[i] > 0:
            out.append("<1%")
        elif v == 100 and any(x > 0 for j, x in enumerate(raw) if j != i):
            out.append(">99%")
        else:
            out.append(f"{v}%")
    return out[0], out[1], out[2], out[3]


def shares(r: dict | None) -> list[tuple[str, str, int]]:
    """The shares a receipt names, as (CSS class, label, whole percent): Original, Other frames and AI; for an upscale,
    Upscaled and AI (and Original or Other frames only when some is left). As the app's receiptShares()."""
    p = percents(r)
    if not p:
        return []
    up = float(r.get("upscaled") or 0) > 0

    def named(key: str, v: int) -> bool:
        return True if key == "ai" else up if key == "upscaled" else not up or v > 0

    return [(cls, label, v) for (key, label, cls), v in zip(SHARES, p) if named(key, v)]


def share_texts(r: dict | None) -> list[tuple[str, str, str]]:
    """shares() with each share as shown (percent_texts()): (CSS class, label, "13%" | "<1%" | ">99%")."""
    t = percent_texts(r)
    if not t:
        return []
    by_cls = {cls: t[i] for i, (_, _, cls) in enumerate(SHARES)}
    return [(cls, label, by_cls[cls]) for cls, label, _ in shares(r)]


def all_ai(r: dict | None) -> bool:
    """A fully generated take: nothing of the original, of other frames or upscaled from your footage is left."""
    return bool(percents(r)) and all(float(r.get(k) or 0) <= 0 for k in ("original", "borrowed", "upscaled"))


def _times(scale) -> str:
    """2.0 → "2×", 2.67 → "2.7×"."""
    return f"{round(float(scale), 1):g}×"


def scale_text(r: dict) -> str:
    """" 2×" for a receipt that knows its factor, else ""."""
    try:
        return f" {_times(r['scale'])}" if float(r.get("scale") or 0) > 0 else ""
    except (TypeError, ValueError):
        return ""


def upscaled(r: dict | None) -> bool:
    """The take was upscaled (Finish 4K): an upscaled share, or (a whole frame generated upstream) the upscaler named."""
    return bool(r) and (float(r.get("upscaled") or 0) > 0 or bool(r.get("upscaler")) or bool(r.get("scale")))


def _source(r: dict | None) -> dict:
    s = (r or {}).get("source")
    return s if isinstance(s, dict) and s.get("take") else {}


def upscale_sentence(r: dict | None) -> str | None:
    """"Upscaled 2× from your footage by SeedVR2 3B: new detail, no new content." ("from Take 02" for a Finish made from
    a take; None: no upscale)."""
    if not upscaled(r):
        return None
    who = str(r.get("upscaler") or "")
    x, n = scale_text(r), _source(r).get("take")
    # SeedVR2 restores detail as it upscales; NVIDIA's RTX Video Super Resolution only sharpens.
    what = "new detail, no new content" if not who or who.startswith("SeedVR2") else "no new content"
    by = f" by {who}" if who else ""
    if r.get("source_upscaled") and n:  # an HDR-only Finish of a take that was upscaled
        return f"Take {n} was upscaled{x} from your footage{by}: {what}."
    return f"Upscaled{x} from {f'Take {n}' if n else 'your footage'}{by}: {what}."


def source_sentence(r: dict | None) -> str | None:
    """What the take a Finish was made from already was: "Take 02 (Remove) is 95% original · 4% other frames · 1% AI."
    (None: made from the original, the input's shares unknown, or a whole frame generated, which says it all)."""
    s = _source(r)
    if not s or not isinstance(s.get("receipt"), dict) or all_ai(r):
        return None
    what = summary(s["receipt"])
    if what == "No receipt":
        return None
    label = f" ({s['label']})" if s.get("label") else ""
    return f"Take {s['take']}{label} is {what}."


def highlights_text(r: dict | None) -> str | None:
    """A Finish take's HDR share as shown: "13%" or "under 1%" (None: no HDR). Its own highlights when it was made from
    a take (receipt "highlights"), else its AI share as the bar rounds it."""
    if not r:
        return None
    if "highlights" in r:
        hl = float(r.get("highlights") or 0)
        whole = int(hl + 0.5)
    else:
        p = percents(r)
        hl, whole = float(r.get("ai") or 0), (p[3] if p else 0)
    if hl <= 0:
        return None
    return f"{whole}%" if whole else "under 1%"


def highlights_sentence(r: dict | None) -> str | None:
    """A Finish take's generated share: "HDR highlights rebuilt by LTX-2.5 on 13% of pixels." (None: no HDR)."""
    t = highlights_text(r)
    return f"HDR highlights rebuilt by LTX-2.5 on {t} of pixels." if t else None


def hdr_detail(r: dict | None) -> str | None:
    """A Finish take's details["hdr"] as read: "highlights rebuilt by LTX-2.5 on 13% of pixels; everything else is your
    footage, upscaled" (Take 02 for one made from a take), rounded as the receipt rounds it."""
    t = highlights_text(r)
    if not t:
        return None
    n = _source(r).get("take")
    return f"highlights rebuilt by LTX-2.5 on {t} of pixels; everything else is {f'Take {n}' if n else 'your footage'}" + \
        (", upscaled" if upscaled(r) and not r.get("source_upscaled") else "")


def summary(r: dict | None) -> str:
    if all_ai(r):
        return "100% AI · whole frame generated" + (f" (upscaled{scale_text(r)})" if upscaled(r) else "")
    parts = share_texts(r)
    return " · ".join(f"{t} {label if label == 'AI' else label.lower()}" for _, label, t in parts) if parts else "No receipt"


def _duration(seconds) -> str | None:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    m, r = divmod(s, 60)
    if m >= 60:
        return f"{m // 60} h {m % 60} min"
    return f"{m} min {r} s" if r else f"{m} min"


def _bytes(n) -> str:
    if not n:
        return ""
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{round(n / 1024 ** 2)} MB"
    return f"{max(1, round(n / 1024))} KB"


def _dims(v) -> str | None:
    return re.sub(r"^(\d+)\s*x\s*(\d+)$", r"\1×\2", v, flags=re.I) if isinstance(v, str) and v else None


def _date(ts) -> str:
    try:
        d = datetime.fromtimestamp(float(ts))
    except (TypeError, ValueError, OSError):
        return "—"
    return f"{d.day} {d:%b %Y · %H:%M}"


def _today() -> str:
    d = datetime.now()
    return f"{d.day} {d:%B %Y}"


def _fps(f) -> str:
    return f"{round(float(f), 3):g} fps" if f else ""


def _length(frames, fps) -> str | None:
    if not frames or not fps:
        return None
    s = frames / float(fps)
    return f"{f'{s:.1f} s' if s < 10 else _duration(s)} · {frames} frames"


def _value(key: str, v) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if key == "look_at":
        return f"Frame {v}"
    if key in VALUE_LABELS and isinstance(v, str):
        return VALUE_LABELS[key].get(v, v)
    return (_dims(v) or v) if isinstance(v, str) else str(v)


def _path_like(v) -> bool:
    return isinstance(v, str) and bool(re.match(r"^(/|~|[A-Za-z]:[\\/]|\\\\)", v))


# --- library files ---------------------------------------------------------------------------------------------------
def _file(store: Store, url: str | None) -> Path | None:
    """The library file behind a /media/ URL from Store.shot() (never outside the library)."""
    if not url or not url.startswith("/media/"):
        return None
    root = store.root.resolve()
    p = (root / url[len("/media/"):]).resolve()
    return p if p.is_relative_to(root) and p.is_file() else None


def _grab(src: Path | None, frame: int, out: Path) -> bytes | None:
    """One frame of `src` as a ~1280 px JPEG (q 4), cached at `out`; the first frame if the clip is shorter."""
    if src is None:
        return None
    try:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime and out.stat().st_size:
            return out.read_bytes()
        tmp = out.with_name(f"_{uuid.uuid4().hex[:8]}_{out.name}")  # two exports at once never share a file
        for n in dict.fromkeys((max(0, frame), 0)):
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"select=eq(n\\,{n}),scale='min({STILL_W},iw)':-2",
                                "-frames:v", "1", "-q:v", "4", str(tmp)], capture_output=True)
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size:
                tmp.replace(out)
                return out.read_bytes()
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return None


def _graph_text(path: Path | None) -> dict[str, list]:
    """What a saved API-format graph sent to its text encoders: prompts, negative prompts and the labels SAM looked for
    (an encoder fed by a SAM checkpoint), and its seeds."""
    out: dict[str, list] = {"texts": [], "negatives": [], "labels": [], "seeds": []}
    try:
        graph = json.loads(path.read_text()) if path else {}
    except (OSError, ValueError):
        graph = {}
    nodes = {k: n for k, n in graph.items() if isinstance(n, dict)} if isinstance(graph, dict) else {}
    sam = {k for k, n in nodes.items() if any(isinstance(v, str) and re.match(r"sam\d", v.lower()) for v in (n.get("inputs") or {}).values())}
    for nid, node in nodes.items():
        inp = node.get("inputs") or {}
        if "TextEncode" in str(node.get("class_type", "")):
            text = inp.get("text", inp.get("prompt"))
            clip = inp.get("clip")
            kind = "negatives" if str(nid).lower().endswith("neg") else "labels" if isinstance(clip, list) and clip and clip[0] in sam \
                else "texts"
            if isinstance(text, str) and text.strip() and text.strip() not in out[kind]:
                out[kind].append(text.strip())
            if isinstance(inp.get("negative_prompt"), str) and inp["negative_prompt"].strip():
                out["negatives"].append(inp["negative_prompt"].strip())
        for k in ("seed", "noise_seed"):
            if isinstance(inp.get(k), int) and not isinstance(inp.get(k), bool) and inp[k] not in out["seeds"]:
                out["seeds"].append(inp[k])
    return out


def workflow_rows(take: dict) -> list[dict]:
    """The take's workflows as the app lists them: identical neighbours merged, Pickups' own steps in place.
    Each row: {"title", "files": [(run index from 0, workflow)]}."""
    rows: list[dict] = []
    for i, w in enumerate(take.get("workflows") or []):
        title = w.get("title") or step_title(w.get("step", ""))
        if rows and rows[-1]["title"] == title:
            rows[-1]["files"].append((i, w))
        else:
            rows.append({"title": title, "files": [(i, w)]})
    d = take.get("details") or {}
    if take.get("task") == "remove" and (re.match(r"(locked|near-locked)", str(d.get("camera", "")))
                                         or str(d.get("method", "")).startswith("real pixels")):
        at = max((i + 1 for i, r in enumerate(rows) if SELECTION.match(r["title"])), default=0)
        rows.insert(at, {"title": "Rebuild · real pixels, in Pickups", "files": []})
    return rows


# --- what a page shows -----------------------------------------------------------------------------------------------
def _project_name(store: Store, project_id: str | None) -> str:
    try:
        return store.project(project_id or UNSORTED)["name"]
    except (OSError, ValueError, KeyError):
        return ""


def sheet(store: Store, shot: dict, take_id: str, project_name: str) -> dict:
    """Everything one receipt page shows (FileNotFoundError: no such take, or the original, which has no receipt)."""
    n, take = next(((i + 1, v) for i, v in enumerate(shot["versions"]) if v["id"] == take_id), (0, None))
    if take is None or take_id == "orig":
        raise FileNotFoundError(take_id)
    d, p = take.get("details") or {}, take.get("params") or {}
    vdir = store.shot_dir(shot["id"]) / "versions" / take_id
    verb = TASK_VERB.get(take.get("task") or "", "Take")
    fps = d.get("fps") or shot.get("fps") or 24
    frames = d.get("frames") if isinstance(d.get("frames"), int) else shot.get("frames")

    # The same frame of before and after: the one the user clicked (Remove, Recast), else the middle.
    clicked = p.get("frame")
    frame = clicked if isinstance(clicked, int) and frames and 0 <= clicked < frames else (frames or 1) // 2
    src_n, src = next(((i + 1, v) for i, v in enumerate(shot["versions"]) if v["id"] == (take.get("input_version") or "orig")),
                      (1, shot["versions"][0]))
    before = _file(store, take.get("before_url")) or _file(store, src.get("video_url"))
    after = _file(store, take.get("video_url"))
    rmap = _file(store, take.get("receipt_url")) if str(take.get("receipt_url", "")).endswith(".mp4") else None

    tc_in = d.get("timecode") if isinstance(d.get("timecode"), str) else None
    if not tc_in and (vdir / "master.mov").exists():
        try:
            tc_in = media.timecode(vdir / "master.mov")
        except (OSError, subprocess.CalledProcessError):
            tc_in = None
    tc_out = media.offset_timecode(tc_in, frames, float(fps)) if tc_in and frames else None

    made = _dims(d.get("rendered_at")) or _dims(d.get("resolution")) or _dims(d.get("output"))
    files = take.get("downloads") or []
    top = next((s for s in ("UHD", "HD") if any(f.get("size") == s or re.search(rf"\b{s}\b", f.get("label", "")) for f in files)), None)
    delivered = DELIVERY_SIZE.get(top or "") or _dims(d.get("resolution")) or made
    rendered = made if made and made != delivered else None
    native = rendered == f"{shot.get('width')}×{shot.get('height')}" and take.get("task") not in ("previz", "environment")

    rows = workflow_rows(take)
    prompts = []
    for r in rows:
        row = {"title": r["title"], "texts": [], "negatives": [], "labels": [], "seeds": []}
        for _, w in r["files"]:
            for k, vals in _graph_text(_file(store, w.get("url"))).items():
                row[k] += [x for x in vals if x not in row[k]]
        if row["texts"] or row["labels"] or row["seeds"]:
            prompts.append(row)
    if not any(x["texts"] or x["labels"] for x in prompts):
        prompts = [{"title": label, "texts": [str(d.get(k) or p.get(k)).strip()], "negatives": [], "labels": [], "seeds": []}
                   for k, label in FALLBACK_PROMPTS if isinstance(d.get(k) or p.get(k), str) and str(d.get(k) or p.get(k)).strip()]
    seed = p.get("seed", d.get("seed"))
    seed_note = None
    if not any(x["seeds"] for x in prompts):
        # The seed only matters to a generative model (not SAM, depth or edges).
        generative = any(m.get("name") not in ("SAM 3.1", "Depth Anything 3", "Canny") and not m.get("via") for m in take.get("licences") or [])
        if str(seed or "").strip() not in ("", "None"):
            seed_note = str(seed)
        elif take.get("task") in SEEDED and generative:
            seed_note = "42 (the default)"

    # Each note at the timecode Pickups shows (from the take's first frame) and at its source timecode, kept inside the
    # take as the timeline export keeps it (a note past the end sits on the last frame). An approved take also carries
    # the original's notes at its own frames, as the export folds them in (nle.fold_original_notes).
    notes = []
    folded = [{"frame": f, "text": f"{t} (original)"} for f, t in nle.fold_original_notes(shot, take_id, frames, fps)] \
        if take.get("status") == "approved" and frames else []
    for note in (take.get("notes") or []) + folded:
        try:
            f = max(0, int(note.get("frame") or 0))
        except (TypeError, ValueError):
            f = 0
        f = min(frames - 1, f) if frames else f
        notes.append({"tc": media.offset_timecode("00:00:00:00", f, float(fps)), "frame": f,
                      "src_tc": media.offset_timecode(tc_in, f, float(fps)) if tc_in else None, "text": str(note.get("text", ""))})
    notes.sort(key=lambda x: x["frame"])

    try:
        kept = "workflows" in store.version_meta(shot["id"], take_id)  # takes made before Pickups kept them have none
    except (OSError, ValueError):
        kept = False

    outside = d.get("pixels_changed_outside_selection")
    return {
        "project": project_name, "shot": shot, "take": take, "n": n, "num": f"{n:02d}", "label": take.get("label") or verb,
        "verb": verb, "stem": f"{_safe(shot['name'])}_T{n:02d}_{take.get('task') or 'take'}",
        "made": _date(take.get("created_at")), "total": _duration(d.get("total_seconds")), "gpu": _duration(d.get("gpu_seconds")),
        "frame": frame, "frame_tc": media.offset_timecode("00:00:00:00", frame, float(fps)),
        "frame_src_tc": media.offset_timecode(tc_in, frame, float(fps)) if tc_in else None,
        "before_label": f"Take {src_n:02d} · {src.get('label') or 'Original'}",
        "before": _grab(before, frame, vdir / "receipt_before.jpg"), "after": _grab(after, frame, vdir / "receipt_after.jpg"),
        "map": _grab(rmap, frame, vdir / "receipt_map.jpg"),
        "receipt": take.get("receipt"), "percents": percents(take.get("receipt")),
        "outside": outside if isinstance(outside, int) and not isinstance(outside, bool) else None,
        "delivered": delivered, "rendered": (("Native" if native else "Rendered at"), rendered) if rendered else None,
        "length": _length(frames, fps), "fps": fps, "frames": frames, "tc_in": tc_in, "tc_out": tc_out,
        "details": [(label, _value(k, d[k])) for k, label in DETAIL_LABELS.items()
                    if d.get(k) not in (None, "") and not _path_like(d[k]) and not isinstance(d[k], (dict, list))],
        "licences": take.get("licences") or [], "prompts": prompts, "seed_note": seed_note,
        "files": [(f.get("filename") or Path(f.get("url", "")).name, f.get("label", ""), f.get("bytes")) for f in files],
        "notes": notes, "workflows": rows, "workflows_kept": kept,
    }


# --- HTML ------------------------------------------------------------------------------------------------------------
def _font_face() -> str:
    out = []
    for family, file, style in (("Instrument Serif", "instrument-serif-latin-400-normal.woff", "normal"),
                                ("Instrument Serif", "instrument-serif-latin-400-italic.woff", "italic"),
                                ("Geist Mono", "geist-mono-latin-400-normal.woff", "normal")):
        try:
            data = base64.b64encode((FONTS / file).read_bytes()).decode()
        except OSError:
            continue
        out.append(f'@font-face{{font-family:"{family}";src:url(data:font/woff;base64,{data}) format("woff");'
                   f"font-weight:400;font-style:{style};font-display:block}}")
    return "".join(out)


CSS = """
:root{--paper:#f6f3ee;--card:#fffdf9;--ink:#171513;--ink2:#56514a;--ink3:#857f77;--line:#e1dacf;--line2:#c9c1b4;
--orig:#cbc3b6;--borrowed:#3dd6f5;--upscaled:#99948d;--ai:#ff4fd8;--tally:#f5301e;
--serif:"Instrument Serif",Georgia,"Times New Roman",serif;--mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:"Instrument Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html{background:var(--paper);color:var(--ink);-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-underline-offset:3px}
.sheet{max-width:980px;margin:0 auto;padding:44px 48px 40px}
.mono{font-family:var(--mono)}
.slate{font-family:var(--mono);font-size:11px;line-height:1.5;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--tally);margin-right:9px;vertical-align:1px}
.top{display:flex;justify-content:space-between;align-items:center;gap:16px;padding-bottom:12px;border-bottom:1px solid var(--ink)}
.top .slate:first-child{color:var(--ink)}
.pill{font:500 13px/1 var(--sans);color:var(--ink);background:none;border:1px solid var(--line2);border-radius:999px;padding:8px 14px;
cursor:pointer;text-decoration:none}
.pill:hover{border-color:var(--ink)}
.tools{display:flex;justify-content:flex-end;gap:8px;max-width:980px;margin:0 auto;padding:18px 48px 0}
.kicker{margin:30px 0 0}
h1{font:400 64px/1 var(--serif);letter-spacing:-.01em;margin:10px 0 0;overflow-wrap:anywhere}
.sub{font:400 30px/1.15 var(--serif);margin:8px 0 0;color:var(--ink2)}
.sub em{color:var(--ink)}
.facts{display:grid;grid-template-columns:minmax(0,1.5fr) repeat(4,minmax(0,1fr));margin:30px 0 0;border-top:1px solid var(--line);
border-bottom:1px solid var(--line)}
.facts div{padding:12px 14px 13px 0}
.facts dt{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.facts dd{margin:5px 0 0;font:13.5px/1.35 var(--mono);white-space:nowrap}
section{margin-top:46px}
h2{display:flex;align-items:baseline;gap:12px;font:400 30px/1.1 var(--serif);margin:0 0 16px;break-after:avoid;page-break-after:avoid}
.stills,figure,.stats,.zero,.mapfig,.tc,.facts,.callout,.prompt,tr,.notes li,.flows li{break-inside:avoid;page-break-inside:avoid}
h2 .n{font:11px/1 var(--mono);letter-spacing:.08em;color:var(--ink3)}
.stills{display:grid;grid-template-columns:1fr 1fr;gap:14px}
figure{margin:0}
figure img{display:block;width:100%;height:auto;border-radius:3px;background:#0d0c0b}
figure .none{display:flex;align-items:center;justify-content:center;aspect-ratio:16/9;border-radius:3px;background:#e9e3d9;
color:var(--ink3)}
figcaption{display:flex;justify-content:space-between;gap:12px;margin-top:8px}
figcaption .live::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--tally);
margin-right:7px;vertical-align:1px}
.frame{margin:10px 0 0;text-align:center}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line)}
.bar i{display:block;height:100%}
.bar .o{background:var(--orig)}.bar .b{background:var(--borrowed)}.bar .u{background:var(--upscaled)}.bar .a{background:var(--ai)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin-top:18px}
.stat b{display:block;font:400 52px/1 var(--serif)}
.stat span{display:inline-flex;align-items:center;gap:7px;margin-top:6px}
.sw{display:inline-block;width:8px;height:8px;border-radius:50%}
.sw.o{background:var(--orig)}.sw.b{background:var(--borrowed)}.sw.u{background:var(--upscaled)}.sw.a{background:var(--ai)}
.plain{font-size:15px;color:var(--ink2);margin:18px 0 0;max-width:62ch}
.plain strong{color:var(--ink);font-weight:600}
.zero{display:inline-flex;align-items:baseline;gap:10px;margin-top:16px;padding:10px 14px;border:1px solid var(--line2);border-radius:6px}
.zero b{font:400 28px/1 var(--serif)}
.mapfig{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start;margin-top:22px}
.mapfig p{margin:0;color:var(--ink2);font-size:14px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:9px 28px;margin:0}
dl.kv dt{color:var(--ink3);font-size:14px}
dl.kv dd{margin:0;font:13px/1.55 var(--mono);overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font:400 10.5px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);
padding:0 14px 8px 0;border-bottom:1px solid var(--line2)}
td{padding:11px 14px 11px 0;border-bottom:1px solid var(--line);vertical-align:top}
td.num,th.num{text-align:right;padding-right:0;white-space:nowrap}
td.file{width:50%;font:12.5px/1.5 var(--mono);overflow-wrap:break-word}
.terms{font-size:13px;line-height:1.45;color:var(--ink2)}
.tag{display:inline-block;font:10px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;padding:4px 6px 3px;
border:1px solid var(--line2);border-radius:3px;color:var(--ink2);white-space:nowrap}
.tag.nc{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.tag.ok{border-color:var(--line2)}
.callout{margin:0 0 16px;padding:12px 16px;border-left:3px solid var(--ink);background:var(--card);font-size:14px}
.prompt{margin-top:18px}
.prompt:first-of-type{margin-top:0}
.prompt pre{margin:7px 0 0;padding:13px 15px;font:12.5px/1.65 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere;
background:var(--card);border:1px solid var(--line);border-radius:4px}
.prompt pre.neg{color:var(--ink3);font-size:11.5px;padding:9px 15px}
.tc{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.tc b{display:block;margin-top:6px;font:400 26px/1.1 var(--mono);letter-spacing:.01em}
.notes{list-style:none;margin:0;padding:0}
.notes li{display:grid;grid-template-columns:170px 1fr;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)}
.notes .when b{display:block;font:400 13.5px/1.4 var(--mono)}
.notes p{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.flows{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}
.flows li{display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)}
.flows .t{min-width:0;overflow-wrap:anywhere}
.flows .t .n{font:12px var(--mono);color:var(--ink3);margin-right:10px}
.flows .t .more{color:var(--ink3)}
.flows .dl{font:12px var(--mono);white-space:nowrap;color:var(--ink2)}
.flows .dl a{margin-left:10px}
.hint{margin:12px 0 0;color:var(--ink2);font-size:14px}
.muted{color:var(--ink3)}
footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 16px;margin-top:60px;padding-top:12px;border-top:1px solid var(--ink)}
.idx td{vertical-align:middle}
.idx .muted{font-size:13px}
.idx img{display:block;width:132px;height:auto;border-radius:3px;background:#0d0c0b}
.idx .bar{height:6px;width:150px;margin-bottom:7px}
.idx .shot{font:400 22px/1.15 var(--serif)}
.idx .sum{font:11.5px/1.5 var(--mono);color:var(--ink2)}
.nw{white-space:nowrap}
@page{size:A4;margin:14mm 14mm 16mm}
@media print{html{background:#fff}.sheet{max-width:none;padding:0}.noprint{display:none!important}h1{font-size:52px}
section{margin-top:30px}footer{margin-top:28px;break-inside:avoid}a{text-decoration:none}.page+.page{break-before:page}}
/* Phones only: an A4 page is narrower than 760 px too, and prints the desktop layout. */
@media screen and (max-width:760px){.sheet{padding:28px 16px}.tools{padding:14px 16px 0}h1{font-size:44px}.sub{font-size:24px}
.facts{grid-template-columns:repeat(2,minmax(0,1fr))}.facts dd{white-space:normal}.stills,.mapfig{grid-template-columns:1fr}.stat b{font-size:38px}
.flows li{flex-wrap:wrap;gap:2px 16px}.flows .dl{white-space:normal}.flows .dl a:first-child{margin-left:0}
.tc{grid-template-columns:1fr 1fr}.tc b{font-size:18px}.notes li{grid-template-columns:1fr;gap:4px}
.idx thead{display:none}.idx tr{display:grid;grid-template-columns:96px minmax(0,1fr);gap:6px 14px;padding:16px 0;
border-bottom:1px solid var(--line)}.idx td{display:block;padding:0;border:0}.idx .th{grid-row:span 5}.idx img{width:96px}
.idx .num{text-align:left}.idx .bar{width:100%}}
"""


def _doc(title: str, body: str, back: str | None = None) -> str:
    js_hash = base64.b64encode(hashlib.sha256(PRINT_JS.encode()).digest()).decode()
    csp = (f"default-src 'none'; img-src data:; font-src data:; style-src 'unsafe-inline'; script-src 'sha256-{js_hash}'; "
           "base-uri 'none'; form-action 'none'")
    back_link = f'<a class="pill" href="{e(back)}">← All receipts</a>' if back else ""
    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<meta name="color-scheme" content="light">
<meta name="generator" content="Pickups">
<title>{e(title)}</title>
<style>{_font_face()}{CSS}</style>
</head>
<body>
<div class="tools noprint">{back_link}<button class="pill" id="print" type="button">Print or save as PDF</button></div>
{body}
<script>{PRINT_JS}</script>
</body>
</html>
"""


def _img(data: bytes | None, alt: str) -> str:
    if not data:
        return '<div class="none slate">No picture</div>'
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(data).decode()}" alt="{e(alt)}">'


def _bar(r: dict | None) -> str:
    p = percents(r)
    if not p:
        return ""
    label = ", ".join(f"{t} {name if name == 'AI' else name.lower()}" for _, name, t in share_texts(r))
    segs = "".join(f'<i class="{cls}" style="width:{v}%"></i>' for (_, _, cls), v in zip(SHARES, p) if v)
    return f'<div class="bar" role="img" aria-label="{e(label)}">{segs}</div>'


def _licence_tag(m: dict) -> str:
    if m.get("commercial") is False:
        return '<span class="tag nc">Non-commercial</span>'
    if m.get("commercial") is True:
        return '<span class="tag ok">Commercial use OK</span>'
    if licences.terms(m.get("licence")):
        return f'<span class="terms">{e(licences.terms(m.get("licence")))}</span>'
    return '<span class="muted">Check its terms</span>' if m.get("licence") else '<span class="muted">No model license</span>'


def _embedded(path: Path | None, filename: str) -> str:
    if not path or path.stat().st_size > EMBED_MAX:
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<a href="data:application/json;base64,{data}" download="{e(filename)}">JSON</a>'


def take_body(s: dict, store: Store | None = None, links: dict[int, str] | None = None, zip_dir: str = "") -> str:
    """One receipt as an <article>. `links` maps a workflow's run index to its file in the zip; without it (a single
    page) each workflow is embedded as a data: link."""
    shot = s["shot"]
    sections: list[tuple[str, str]] = []

    after_label = f"Take {s['num']} · {s['label']}"
    frame = f"Frame {s['frame']} · {s['frame_tc']}" + (f" · Source {s['frame_src_tc']}" if s["frame_src_tc"] else "")
    sections.append(("Before and after", f"""
<div class="stills">
<figure>{_img(s['before'], 'Before')}<figcaption><span class="slate">Before</span><span class="slate">{e(s['before_label'])}</span></figcaption></figure>
<figure>{_img(s['after'], 'After')}<figcaption><span class="slate live">After</span><span class="slate">{e(after_label)}</span></figcaption></figure>
</div>
<p class="frame slate">{e(frame)}</p>"""))

    p = s["percents"]
    if p:
        # Whole percents as the app shows them; the sentence never rounds a share it names down to nothing.
        r = s["receipt"] or {}
        raw = [float(r.get(k) or 0) for k, _, _ in SHARES]
        pct = [x.replace("<1%", "Under 1%").replace(">99%", "Over 99%") for x in percent_texts(r)]
        # Finish: an upscale is the footage resampled (new detail, no new content); HDR highlights are its own AI share.
        # Made from a take, it also says what that take already was (its other frames and AI are in the shares above).
        finish = s["take"].get("task") == "finish"
        upscale = upscale_sentence(r) if finish else None
        extra = " ".join(e(x) for x in ((source_sentence(r), highlights_sentence(r)) if finish else ()) if x)
        if all_ai(r):
            plain = "<strong>100% AI: the whole frame is generated.</strong>" + \
                "".join(f" {e(x)}" for x in ((upscale, highlights_sentence(r)) if finish else ()) if x)
        elif upscale and not r.get("source_upscaled"):
            plain = f"<strong>{e(upscale)}</strong>" + (f" {extra}" if extra else "")
        else:
            plain = f"<strong>{pct[0]} of the picture is the original footage, untouched.</strong>"
            if raw[1] > 0:
                plain += f" {pct[1]} is real footage borrowed from other frames of the same shot."
            if upscale:  # an HDR-only Finish of an upscaled take
                plain += f" {e(upscale)}"
            plain += f" {extra}" if extra else f" {pct[3]} is generated by AI." if raw[3] > 0 else " None of it is generated by AI."
        zero = ""
        if s["outside"] is not None:
            zero = (f'<div class="zero"><b>{s["outside"]:,}</b><span class="slate">'
                    f'{"pixel" if s["outside"] == 1 else "pixels"} changed outside the selection</span></div>')
        mapfig = ""
        if s["map"]:
            mapfig = f"""<div class="mapfig"><figure>{_img(s['map'], 'Where the pixels came from')}</figure>
<p><span class="slate">On the picture</span><br>Your footage shows as it is. Pixels borrowed from other frames are tinted
cyan; generated pixels are tinted magenta. Same frame as above.</p></div>"""
        stats = "\n".join(f'<div class="stat"><b>{t}</b><span class="slate"><i class="sw {cls}"></i>{label}</span></div>'
                          for cls, label, t in share_texts(r))
        sections.append(("What changed", f"""{_bar(r)}
<div class="stats">
{stats}
</div>
<p class="plain">{plain}</p>{zero}{mapfig}"""))

    rows = [("Task", s["verb"])]
    rows += [r for r in s["details"] if r[0] in ("Method", "Camera")]
    if s["delivered"]:
        rows.append(("Delivered", s["delivered"]))
    if s["rendered"]:
        rows.append(s["rendered"])
    if s["length"]:
        rows.append(("Length", s["length"]))
    if s["fps"]:
        rows.append(("Frame rate", _fps(s["fps"])))
    rows += [r for r in s["details"] if r[0] not in ("Method", "Camera")]
    sections.append(("How it was made", '<dl class="kv">' + "".join(
        f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in rows) + "</dl>"))

    if s["licences"]:
        nc = [licence_name(m) for m in s["licences"] if m.get("commercial") is False]
        callout = ""
        if nc:  # what to do about it is for the app's take Details, not for the client's sheet
            callout = (f'<p class="callout"><strong>Not for paid work as it stands:</strong> {e(", ".join(nc))} '
                       f'{"is" if len(nc) == 1 else "are"} licensed for non-commercial use only.</p>')
        # A model that made the take this one was made from (a Finish of a Remove) says so: "via Take 02".
        def via(m: dict) -> str:
            return f' <span class="muted">via {e(m["via"])}</span>' if m.get("via") else ""

        trs = "".join(f"<tr><td>{e(m['name'])}{via(m)}</td><td>{e(m.get('licence') or '—')}</td><td>{_licence_tag(m)}</td></tr>"
                      for m in s["licences"])
        sections.append(("Models and licenses", f"""{callout}<table><thead><tr><th>Model</th><th>License</th><th>Commercial use</th></tr></thead>
<tbody>{trs}</tbody></table>"""))

    if s["prompts"] or s["seed_note"]:
        blocks = []
        for pr in s["prompts"]:
            seeds = f" · Seed {', '.join(str(x) for x in pr['seeds'])}" if pr["seeds"] else ""
            texts = "".join(f"<pre>{e(t)}</pre>" for t in pr["texts"])
            texts += "".join(f'<pre class="neg">Selection: {e(t)}</pre>' for t in pr["labels"])
            negs = "".join(f'<pre class="neg">Negative: {e(t)}</pre>' for t in pr["negatives"])
            if not texts:
                texts = '<p class="muted" style="margin:6px 0 0">No text prompt.</p>'
            blocks.append(f'<div class="prompt"><div class="slate">{e(pr["title"] + seeds)}</div>{texts}{negs}</div>')
        if s["seed_note"]:
            blocks.append(f'<p class="hint">Seed {e(s["seed_note"])}.</p>')
        sections.append(("Prompts, as each model received them", "".join(blocks)))

    if s["tc_in"]:
        tc = f"""<div class="tc"><div><span class="slate">In</span><b>{e(s['tc_in'])}</b></div>
<div><span class="slate">Out</span><b>{e(s['tc_out'] or '—')}</b></div>
<div><span class="slate">Length</span><b>{e(str(s['frames'] or '—'))} fr</b></div></div>
<p class="hint">Source timecode, so the take drops straight over the original in the edit (out is exclusive, as in an EDL).</p>"""
    else:
        tc = '<p class="hint" style="margin:0">The source has no timecode; the take starts at its first frame.</p>'
    sections.append(("Timecode", tc))

    if s["files"]:
        # Long names break after an underscore, never inside a word or the extension.
        trs = "".join(f'<tr><td class="file">{e(name).replace("_", "_<wbr>")}</td><td>{e(label)}</td>'
                      f'<td class="num mono">{e(_bytes(size))}</td></tr>' for name, label, size in s["files"])
        sections.append(("Files delivered", f"""<table><thead><tr><th>File</th><th>What it is</th><th class="num">Size</th></tr></thead>
<tbody>{trs}</tbody></table>"""))

    if s["notes"]:
        # The timecode Pickups shows first (from the take's first frame), so a client can find the note in the app.
        lis = "".join(f'<li><span class="when"><b>{e(n["tc"])}</b><span class="slate">'
                      f'{e("Source " + n["src_tc"] if n["src_tc"] else "Frame " + str(n["frame"]))}</span></span><p>{e(n["text"])}</p></li>'
                      for n in s["notes"])
        where = "; the source timecode under it is where the note sits in the edit" if s["tc_in"] else ""
        sections.append(("Notes", f'<ul class="notes">{lis}</ul><p class="hint">Timecode from the take\'s first frame, as Pickups '
                                  f"shows it{where}.</p>"))

    # Workflows only when the take kept some (as the app's ComfyUI list), else one line that says why there are none.
    saved = bool(links) if links is not None else any(r["files"] for r in s["workflows"])
    if saved:
        lis = []
        for i, r in enumerate(s["workflows"], start=1):
            what, *rest = r["title"].split(" · ")
            title = e(what) + (f'<span class="more"> · {e(" · ".join(rest))}</span>' if rest else "")
            count = f'<span class="more"> ×{len(r["files"])}</span>' if len(r["files"]) > 1 else ""
            if not r["files"]:
                dl = '<span class="muted">In Pickups, no workflow</span>'
            else:
                anchors = []
                for j, w in r["files"]:
                    if links is not None:
                        a = f'<a href="{e(links[j])}">JSON</a>' if j in links else ""
                    else:
                        a = _embedded(_file(store, w.get("url")) if store else None, w.get("filename") or f"{j + 1:02d}_{w.get('step')}.json")
                    anchors.append(a)
                dl = "".join(anchors)
            lis.append(f'<li><span class="t"><span class="n">{i:02d}</span>{title}{count}</span><span class="dl">{dl}</span></li>')
        where = (f"The JSON files are in this zip, in <span class=\"mono\">{e(zip_dir)}</span>. " if links is not None else "")
        also = "" if links is not None else ", and opens from the take in Pickups too"
        sections.append(("ComfyUI workflows", f'<ul class="flows">{"".join(lis)}</ul><p class="hint">{where}Each one is the exact '
                                              f"ComfyUI workflow this take ran{also}. Drag one onto ComfyUI to see what ran; "
                                              "running it again needs the source media and the same models.</p>"))
    else:
        why = ("This take was made before Pickups kept its ComfyUI workflows." if not s["workflows_kept"] else
               "No ComfyUI workflow ran for this take: Pickups made it itself.")
        sections.append(("ComfyUI workflows", f'<p class="hint" style="margin:0">{why}</p>'))

    facts = [("Made", s["made"]), ("Total time", s["total"] or "—"), ("GPU time", s["gpu"] or "—"),
             ("Delivered", s["delivered"] or "—"), ("Length", (s["length"] or "—").split(" · ")[0])]
    head = f"""<header class="top"><span class="slate"><i class="dot"></i>Pickups receipt</span>
<span class="slate">{e(s['stem'])}</span></header>
<p class="kicker slate">{e(s['project'] or 'Project')}</p>
<h1>{e(shot['name'])}</h1>
<p class="sub">Take {s['num']} · <em>{e(s['label'])}</em></p>
<dl class="facts">{"".join(f"<div><dt>{k}</dt><dd>{e(v)}</dd></div>" for k, v in facts)}</dl>"""
    body = "".join(f'<section><h2><span class="n">{i:02d}</span>{e(t)}</h2>{c}</section>' for i, (t, c) in enumerate(sections, start=1))
    foot = f"""<footer><span class="slate">Made with Pickups · open source</span><span class="slate">{e(_today())}</span></footer>"""
    return f'<article class="sheet page">{head}{body}{foot}</article>'


def take_page(store: Store, shot_id: str, take_id: str) -> str:
    """One take's receipt as a page (FileNotFoundError: no such shot or take, or the original)."""
    shot = store.shot(shot_id)
    s = sheet(store, shot, take_id, _project_name(store, shot.get("project_id")))
    return _doc(f"Receipt · {shot['name']} · Take {s['num']} {s['label']}", take_body(s, store))


def licence_name(m: dict) -> str:
    """A model as a receipt names it: "Qwen-Image 2.1", or "Qwen-Image 2.1 (via Take 02)" for one that made the take this
    take was made from (Store.shot() adds those to a take's licences)."""
    return f"{m.get('name')} (via {m['via']})" if m.get("via") else str(m.get("name"))


def _licence_flags(lic: list[dict]) -> str:
    nc = [licence_name(m) for m in lic if m.get("commercial") is False]
    if nc:
        return f'<span class="tag nc">Non-commercial</span><div class="muted" style="margin-top:5px">{e(", ".join(nc))}</div>'
    # Commercial use on a licence's own terms: the ones Pickups can state (each receipt says what), and the rest.
    cond = [licence_name(m) for m in lic if m.get("commercial") is None and licences.terms(m.get("licence"))]
    check = [licence_name(m) for m in lic if m.get("commercial") is None and m.get("licence") and not licences.terms(m.get("licence"))]
    lines = ([f"Conditions apply: {', '.join(cond)}"] if cond else []) + ([f"Check terms: {', '.join(check)}"] if check else [])
    if lines:
        return f'<span class="muted">{"<br>".join(e(x) for x in lines)}</span>'
    return '<span class="tag ok">Commercial use OK</span>' if lic else '<span class="muted">—</span>'


def project_zip(store: Store, project_id: str) -> bytes | None:
    """The receipts of a project's approved takes as a zip (index.html, takes/, workflows/); None when there are none.
    Approved originals are skipped: they have nothing to receipt."""
    project_name = _project_name(store, project_id)
    shots = {s["id"]: s for s in store.shots(project_id)}
    items: list[tuple[str, dict]] = []
    used: set[str] = set()
    for d in store.deliveries(project_id):
        if d["take"]["id"] == "orig" or d["shot_id"] not in shots:
            continue
        s = sheet(store, shots[d["shot_id"]], d["take"]["id"], project_name)
        name, k = s["stem"], 2
        while name.lower() in used:  # two shots with the same name
            name, k = f"{s['stem']}_{k}", k + 1
        used.add(name.lower())
        items.append((name, s))
    if not items:
        return None
    items.sort(key=lambda x: (x[1]["shot"]["name"].lower(), x[1]["n"]))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        rows = []
        any_flows = False
        for name, s in items:
            links = {}
            for i, w in enumerate(s["take"].get("workflows") or []):
                f = _file(store, w.get("url"))
                if f:
                    arc = f"workflows/{name}/{i + 1:02d}_{_safe(w.get('title') or w.get('step') or 'workflow')}.json"
                    z.writestr(arc, f.read_bytes())
                    links[i] = f"../{arc}"
            any_flows = any_flows or bool(links)
            title =f"Receipt · {s['shot']['name']} · Take {s['num']} {s['label']}"
            z.writestr(f"takes/{name}.html", _doc(title, take_body(s, store, links, f"workflows/{name}/"), back="../index.html"))
            thumb = _file(store, s["take"].get("thumb_url"))
            label = f'<div class="muted">{e(s["label"])}</div>' if s["label"] != s["verb"] else ""
            pieces = " · ".join(f'<span class="nw">{e(x)}</span>' for x in summary(s["receipt"]).split(" · "))
            # What the grey share is, in words, and what the take it was made from already was.
            upscale = " ".join(x for x in (upscale_sentence(s["receipt"]), source_sentence(s["receipt"])) if x)
            words = f'<div class="muted" style="margin-top:4px">{e(upscale)}</div>' if upscale else ""
            rows.append(f"""<tr><td class="th">{_img(thumb.read_bytes() if thumb else s['after'], s['shot']['name'])}</td>
<td class="sh"><div class="shot">{e(s['shot']['name'])}</div></td>
<td class="tk"><span class="mono">T{s['num']}</span> · {e(s['verb'])}{label}</td>
<td class="wc">{_bar(s['receipt'])}<span class="sum">{pieces}</span>{words}</td>
<td class="lc">{_licence_flags(s['licences'])}</td>
<td class="num lk"><a href="takes/{e(name)}.html">Receipt →</a></td></tr>""")
        nc = sum(1 for _, s in items if any(m.get("commercial") is False for m in s["licences"]))
        warn = (f'<p class="callout"><strong>{nc} of {len(items)} {"take uses" if nc == 1 else "takes use"}</strong> a model licensed for '
                "non-commercial use only. Their receipts say which.</p>") if nc else ""
        flows_hint = 'One page per take in <span class="mono">takes/</span>' + (
            '; the ComfyUI workflows that made them in <span class="mono">workflows/</span>, ready to drag onto ComfyUI.'
            if any_flows else ".")
        body = f"""<article class="sheet page"><header class="top"><span class="slate"><i class="dot"></i>Pickups receipts</span>
<span class="slate">{len(items)} approved {"take" if len(items) == 1 else "takes"}</span></header>
<p class="kicker slate">Project</p><h1>{e(project_name)}</h1>
<p class="sub">Every approved take: what changed, the models that made it, and their licenses.</p>
<section>{warn}<table class="idx"><thead><tr><th></th><th>Shot</th><th>Take</th><th>What changed</th><th>Licenses</th>
<th class="num"></th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p class="hint">{flows_hint}</p></section>
<footer><span class="slate">Made with Pickups · open source</span><span class="slate">{e(_today())}</span></footer></article>"""
        z.writestr("index.html", _doc(f"Receipts · {project_name}", body))
    return buf.getvalue()
