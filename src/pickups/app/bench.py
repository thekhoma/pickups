"""Benchmarks from real runs, per task and per GPU.

Every finished job (a take or a project task: done, failed or cancelled) adds one JSON line to
PICKUPS_HOME/benchmarks.jsonl: the task and its engine, the GPU it ran on, the length and size of the work, the
seconds spent in each stage, the GPU time and the total. `pickups bench` sums them up per task × engine × GPU × size
class (HD, 4K, other): runs, median total, median GPU time, median seconds per output frame, fastest and slowest. It
writes PICKUPS_HOME/bench.json, prints a Markdown table, and with `--md README.md` rewrites the table between the
<!-- bench:start --> and <!-- bench:end --> markers. Setup → Benchmarks shows the same numbers.

Where a run happened (`host`) comes from the ComfyUI address: a RunPod proxy address is "runpod", a PC on the LAN or over
Tailscale is "remote", this computer is "local". PICKUPS_GPU_LABEL names a local address only (a tunnel to a rented GPU
looks local); it never overrides an address that says where the GPU is. The API proxy port says nothing about the GPU.

`pickups bench --backfill` adds lines for takes made before this recording existed (they kept their GPU and total
time, not the GPU's name): see backfill().
"""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from pickups.app.comfy import HOME

FILE = "benchmarks.jsonl"
SUMMARY = "bench.json"
START, END = "<!-- bench:start -->", "<!-- bench:end -->"
POD_GPU = "RTX PRO 6000 Blackwell Server Edition"
TASK_ORDER = ("remove", "recast", "environment", "previz", "relight", "finish", "image", "video", "sheet", "reshoot")
TASK_LABELS = {"remove": "Remove a person", "recast": "Replace a person", "environment": "Change the weather",
               "previz": "Previz to final", "relight": "Relight", "finish": "Finish", "image": "Image", "video": "New video",
               "sheet": "Character sheet",
               "reshoot": "Reshoot"}
VARIANT_LABELS = {("remove", "locked-off"): "Remove, locked-off", ("remove", "moving"): "Remove, moving camera",
                  ("recast", "ltx"): "Replace a person (LTX-2.5)", ("recast", "scail"): "Replace a person (SCAIL-2)",
                  ("finish", "best"): "Finish, best quality"}
CLASS_ORDER = ("4K", "HD", "other")

_lock = threading.Lock()
_machines: dict[str, dict] = {}  # comfy_url -> GPU facts (read once per ComfyUI address)
_cache: dict = {"key": None, "groups": []}


def path(home: Path | str | None = None) -> Path:
    """The run log: PICKUPS_HOME/benchmarks.jsonl (or beside another library's folder)."""
    return Path(home or HOME) / FILE


# --- the machine a run happened on ------------------------------------------------------------------------------------
def gpu_name(raw: str) -> str:
    """ComfyUI's device name, short: "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync" → "RTX 4090"."""
    return re.sub(r"^\w+:\d+\s+", "", raw or "").split(" : ")[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()


def host_kind(comfy_url: str) -> str:
    """"local", "remote", "runpod", or PICKUPS_GPU_LABEL for a local address (see the module notes)."""
    host = urlparse(comfy_url or "").hostname or ""
    if host.endswith(".proxy.runpod.net"):
        return "runpod"
    if host not in ("127.0.0.1", "localhost", "::1"):
        return "remote"
    return os.environ.get("PICKUPS_GPU_LABEL", "").strip().lower() or "local"


def machine(conn) -> dict:
    """GPU, VRAM, ComfyUI version and host for a run, from ComfyUI's /system_stats once per address. Never raises; an
    offline ComfyUI gives no GPU (and is asked again next time)."""
    url = str(getattr(conn, "comfy_url", "") or "")
    if url in _machines:
        return dict(_machines[url])
    info = {"gpu": None, "vram_gb": None, "comfy_version": None, "host": host_kind(url)}
    try:
        stats = conn.system_stats()
    except Exception:  # noqa: BLE001 - no stats is a normal state
        stats = None
    if not stats:
        return info
    dev = (stats.get("devices") or [{}])[0]
    info |= {"gpu": gpu_name(dev.get("name", "")) or None, "vram_gb": round(dev.get("vram_total", 0) / 1e9, 1) or None,
             "comfy_version": (stats.get("system") or {}).get("comfyui_version")}
    _machines[url] = info
    return dict(info)


# --- one run ---------------------------------------------------------------------------------------------------------
def size_of(text) -> tuple[int | None, int | None]:
    """"3840×2160" or "3840x2160" → (3840, 2160)."""
    m = re.match(r"\s*(\d+)\s*[x×]\s*(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def res_class(w: int | None, h: int | None) -> str:
    """HD (long side 1800-3199 px), 4K (3200 and up) or other."""
    if not w or not h:
        return "other"
    long = max(int(w), int(h))
    return "4K" if long >= 3200 else "HD" if long >= 1800 else "other"


def variant(task: str | None, params: dict | None, details: dict | None) -> str | None:
    """What makes runs of a task comparable: the engine asked for (Recast: ltx or scail), Remove's camera (locked-off
    shots borrow real pixels in seconds, moving ones render with LTX-2.5), the image model, Finish's quality."""
    params, details = params or {}, details or {}
    e = params.get("engine") or details.get("engine")
    if e:
        return str(e).strip().lower()
    if task == "recast":  # older takes: the engine shows in the models used
        names = " ".join(m if isinstance(m, str) else str((m or {}).get("name", "")) for m in details.get("models") or []).lower()
        return "scail" if "scail" in names else "ltx" if "ltx" in names else None
    if task == "remove":
        method = str(details.get("method") or "")
        if details.get("camera") == "moving" or method.startswith("LTX"):
            return "moving"
        return "locked-off" if method else None
    if task == "image":
        return params.get("model") or None
    if task == "video":  # text or image to video, at a size: "image-1080p"
        return f"{params.get('mode') or 'text'}-{params.get('resolution') or '1080p'}"
    if task == "finish":
        return params.get("quality") or None
    return None


def _iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t if t is not None else time.time()))


def _num(x, nd: int = 1):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def run_entry(*, task: str | None, status: str, machine: dict, params: dict | None = None, details: dict | None = None,
              stages: dict | None = None, gpu_seconds=None, total_seconds=None, shot: dict | None = None,
              frames: int | None = None, width: int | None = None, height: int | None = None, engine: str | None = None,
              ts: float | None = None, **ids) -> dict:
    """One benchmark line. The work size is the size the GPU made (a take's `rendered_at`, else its output or delivery
    size; the shot's own size when nothing was made); the size class goes by the delivery size."""
    d = details or {}
    shot = shot or {}
    if width is None:
        width, height = size_of(d.get("rendered_at") or d.get("output") or d.get("resolution"))
    if width is None:
        width, height = shot.get("width"), shot.get("height")
    dw, dh = size_of(d.get("output") or d.get("resolution"))
    if frames is None:
        frames = d.get("frames")
    fps = d.get("fps") or shot.get("fps")
    out = {"ts": _iso(ts), "task": task, "engine": engine or variant(task, params, d), "status": status,
           "gpu": machine.get("gpu"), "vram_gb": machine.get("vram_gb"), "host": machine.get("host"),
           "comfy_version": machine.get("comfy_version"),
           "frames": int(frames) if frames else None, "width": int(width) if width else None, "height": int(height) if height else None,
           "res_class": res_class(dw or width, dh or height), "fps": _num(fps, 3),
           "stages": stages or {}, "gpu_seconds": _num(gpu_seconds), "total_seconds": _num(total_seconds)}
    return out | {k: v for k, v in ids.items() if v not in (None, [], "")}


def append(entries: dict | list[dict], file: Path | None = None) -> None:
    """Add lines to the run log (one write per call, so the app and a script can both add runs)."""
    entries = [entries] if isinstance(entries, dict) else entries
    if not entries:
        return
    f = Path(file or path())
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    with _lock:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(text)


def load(file: Path | None = None) -> list[dict]:
    """Every run in the log (bad lines are skipped)."""
    try:
        text = Path(file or path()).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


# --- takes made before recording -------------------------------------------------------------------------------------
def pod_take_ids(file: Path | str | None) -> set[str]:
    """Take ids from a text file, one per line (# comments). The file is created, empty, when missing."""
    if not file:
        return set()
    f = Path(file)
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    return {line.split("#", 1)[0].strip() for line in f.read_text(encoding="utf-8").splitlines()} - {""}


def backfill(store=None, file: Path | None = None, gpu: str = "RTX 4090", host: str = "remote",
             pod_takes: Path | str | None = None, pod_gpu: str = POD_GPU, pod_host: str = "runpod") -> list[dict]:
    """Lines for takes that kept their times (details.total_seconds) but have no benchmark of their own (no _bench).
    They ran on `gpu`, except the take ids listed in `pod_takes` (`pod_gpu`). Keyed by take id: a take already in
    the log is never added twice. Returns the lines added."""
    from pickups.app.store import Store

    store = store or Store()
    f = Path(file or path(store.root.parent))
    pods = pod_take_ids(pod_takes)
    have = {r.get("take_id") for r in load(f) if r.get("take_id")}
    added = []
    for meta_file in sorted((store.root / "shots").glob("*/versions/*/meta.json")):
        try:
            m = json.loads(meta_file.read_text())
        except (OSError, ValueError):
            continue
        d = m.get("details") or {}
        vid = m.get("id") or meta_file.parent.name
        if not isinstance(d, dict) or "total_seconds" not in d or "_bench" in d or vid in have:
            continue
        shot_id = meta_file.parent.parent.parent.name
        try:
            shot = json.loads((meta_file.parent.parent.parent / "meta.json").read_text())
        except (OSError, ValueError):
            shot = {}
        pod = vid in pods
        where = {"gpu": pod_gpu if pod else gpu, "vram_gb": None, "comfy_version": None, "host": pod_host if pod else host}
        added.append(run_entry(task=m.get("task"), status="done", machine=where, params=m.get("params"), details=d,
                               gpu_seconds=d.get("gpu_seconds"), total_seconds=d.get("total_seconds"), shot=shot,
                               ts=m.get("created_at"), shot_id=shot_id, take_id=vid, backfill=True))
        have.add(vid)
    added.sort(key=lambda e: e["ts"])
    append(added, f)
    return added


# --- the summary -----------------------------------------------------------------------------------------------------
def label(task: str, engine: str | None) -> str:
    """"Recast (SCAIL-2)", "Remove, locked-off", "Image (FLUX 3 klein)"."""
    if (task, engine) in VARIANT_LABELS:
        return VARIANT_LABELS[(task, engine)]
    base = TASK_LABELS.get(task, str(task).replace("-", " ").capitalize())
    if engine and task == "image":
        try:
            from pickups.images import MODELS
            return f"Image ({MODELS[engine].name})"
        except Exception:  # noqa: BLE001 - an image model this build doesn't know
            return f"Image ({engine})"
    return f"{base} ({engine})" if engine else base


LONG_FRAMES = 100  # 4 s at 24 fps (97 frames) and below, or longer


def _med(xs: list[float], nd: int = 1):
    return round(statistics.median(xs), nd) if xs else None


def aggregate(rows: list[dict]) -> list[dict]:
    """Finished runs grouped by task × engine × GPU × size class × length (up to LONG_FRAMES, or longer: a 7 s
    render costs more per frame than a 4 s one, so they're never averaged together), in task order."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("status") != "done" or not r.get("task") or not _num(r.get("total_seconds")):
            continue
        long_ = bool(r.get("frames")) and int(r["frames"]) > LONG_FRAMES
        key = (r["task"], r.get("engine") or "", r.get("gpu") or "unknown", r.get("res_class") or res_class(r.get("width"), r.get("height")),
               long_)
        groups.setdefault(key, []).append(r)
    out = []
    for (task, engine, gpu, cls, _), rs in groups.items():
        totals = [float(r["total_seconds"]) for r in rs]
        gpu_s = [float(r["gpu_seconds"]) for r in rs if _num(r.get("gpu_seconds")) is not None]
        frames = [int(r["frames"]) for r in rs if r.get("frames")]
        per = [float(r["total_seconds"]) / int(r["frames"]) for r in rs if r.get("frames")]
        sizes = Counter(f"{r['width']}×{r['height']}" for r in rs if r.get("width") and r.get("height"))
        hosts = Counter(r["host"] for r in rs if r.get("host"))
        vram = [float(r["vram_gb"]) for r in rs if r.get("vram_gb")]
        out.append({"task": task, "engine": engine or None, "label": label(task, engine or None), "gpu": gpu,
                    "res_class": cls, "runs": len(rs), "total_s": _med(totals), "gpu_s": _med(gpu_s),
                    "s_per_frame": _med(per, 2), "fastest_s": round(min(totals), 1), "slowest_s": round(max(totals), 1),
                    "frames": statistics.median_low(frames) if frames else None,
                    "size": sizes.most_common(1)[0][0] if sizes else None,
                    "host": hosts.most_common(1)[0][0] if hosts else None, "vram_gb": max(vram) if vram else None,
                    "last": max(str(r.get("ts") or "") for r in rs) or None})

    def rank(g):
        t = TASK_ORDER.index(g["task"]) if g["task"] in TASK_ORDER else len(TASK_ORDER)
        c = CLASS_ORDER.index(g["res_class"]) if g["res_class"] in CLASS_ORDER else len(CLASS_ORDER)
        return (t, g["task"], g["engine"] or "", c, g["gpu"], g["frames"] or 0)

    return sorted(out, key=rank)


def gpus(groups: list[dict]) -> list[dict]:
    """The GPUs measured, most runs first."""
    by: dict[str, dict] = {}
    for g in groups:
        x = by.setdefault(g["gpu"], {"gpu": g["gpu"], "runs": 0, "vram_gb": None, "host": g["host"]})
        x["runs"] += g["runs"]
        x["vram_gb"] = g["vram_gb"] or x["vram_gb"]
    return sorted(by.values(), key=lambda x: -x["runs"])


def measured(file: Path | None = None) -> list[dict]:
    """aggregate() of the run log, read again only when the log changes (the setup check asks often)."""
    f = Path(file or path())
    try:
        st = f.stat()
        key = (str(f), st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    if _cache["key"] != key:
        _cache.update(key=key, groups=aggregate(load(f)))
    return _cache["groups"]


def status_rows(gpu: str | None, reference: list[dict], factor: float = 1.0, file: Path | None = None) -> list[dict]:
    """Setup → Benchmarks: the connected GPU's measured rows first (or, when it has none yet, the RTX 4090 reference
    scaled to it), then every other GPU measured, most runs first. Nothing measured at all: the reference only."""
    groups = measured(file)

    def row(g: dict, current: bool) -> dict:
        return {"task": g["label"], "seconds": int(round(g["total_s"])), "frames": g["frames"], "size": g["size"] or "",
                "gpu": g["gpu"], "runs": g["runs"], "gpu_seconds": g["gpu_s"], "s_per_frame": g["s_per_frame"],
                "fastest": g["fastest_s"], "slowest": g["slowest_s"], "measured": True, "current": current, "host": g["host"]}

    def estimate(name: str | None, f: float) -> list[dict]:
        return [b | {"seconds": int(b["seconds"] * f), "gpu": name or "RTX 4090", "runs": 0, "measured": False, "current": bool(name)}
                for b in reference]

    if not groups:
        return estimate(gpu, factor if gpu else 1.0)
    mine = [g for g in groups if gpu and g["gpu"] == gpu]
    out = [row(g, True) for g in mine] if mine else (estimate(gpu, factor) if gpu else [])
    for other in gpus([g for g in groups if g not in mine]):
        out += [row(g, False) for g in groups if g["gpu"] == other["gpu"] and g not in mine]
    return out


# --- output ----------------------------------------------------------------------------------------------------------
def _t(s) -> str:
    if s is None:
        return "–"
    s = float(s)
    if s < 10:
        return f"{s:.1f} s"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _shot(g: dict) -> str:
    n = g["frames"]
    unit = {"image": "image", "sheet": "view"}.get(g["task"], "frame")
    what = f"{n} {unit}{'' if n == 1 else 's'}" if n else ""
    return ", ".join(x for x in (what, g["size"] or "") if x) or "–"


def markdown(groups: list[dict]) -> str:
    """The summary as a Markdown table, a task's GPUs side by side."""
    lines = ["| Task | GPU | Shot | Runs | GPU time | Total | Per frame | Fastest – slowest |",
             "|---|---|---|--:|--:|--:|--:|---|"]
    for g in groups:
        spread = f"{_t(g['fastest_s'])} – {_t(g['slowest_s'])}" if g["runs"] > 1 else "–"
        lines.append(f"| {g['label']} | {g['gpu']} | {_shot(g)} | {g['runs']} | {_t(g['gpu_s'])} | {_t(g['total_s'])} | "
                     f"{_t(g['s_per_frame'])} | {spread} |")
    return "\n".join(lines)


def section(groups: list[dict]) -> str:
    """What goes between the README's bench markers."""
    runs = sum(g["runs"] for g in groups)
    n = len({g["gpu"] for g in groups})
    intro = (f"Measured end to end through the app: the median of {runs} finished run{'s' if runs != 1 else ''} on "
             f"{n} GPU{'s' if n != 1 else ''}, updated {time.strftime('%Y-%m-%d')} by `pickups bench --md README.md`. "
             "GPU time is ComfyUI's share; the rest is compositing and encoding on the client.")
    return f"{intro}\n\n{markdown(groups)}" if groups else "No runs measured yet."


def rewrite_readme(readme: Path | str, groups: list[dict]) -> None:
    """Replace what sits between <!-- bench:start --> and <!-- bench:end --> (ValueError when the markers are missing)."""
    f = Path(readme)
    text = f.read_text(encoding="utf-8")
    a, b = text.find(START), text.find(END)
    if a < 0 or b < a:
        raise ValueError(f"{f.name} has no {START} … {END} section")
    f.write_text(f"{text[:a]}{START}\n{section(groups)}\n{text[b:]}", encoding="utf-8")


def write(home: Path | str | None = None) -> dict:
    """Summarise the run log into PICKUPS_HOME/bench.json. Returns the summary."""
    groups = aggregate(load(path(home)))
    data = {"updated": _iso(), "runs": sum(g["runs"] for g in groups), "gpus": gpus(groups), "rows": groups}
    f = Path(home or HOME) / SUMMARY
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


if __name__ == "__main__":
    print(markdown(write()["rows"]))
