"""How long generations really took on this machine, by kind (the task, its model and its size).

Only measured runs go in: a job's seconds from the moment it started on the GPU to the moment it finished. A canceled
or failed run is never counted. After one measured run of a kind, the page may say "usually about N s" (the median of
the last few). Kept in PICKUPS_HOME/timings.json, beside benchmarks.jsonl.
"""

from __future__ import annotations

import json
import statistics
import threading
from pathlib import Path

FILE = "timings.json"
KEEP = 10  # runs kept per kind
MEDIAN_OF = 5  # "usually" is the median of the last this many

_lock = threading.Lock()


def _size(p: dict) -> str:
    w, h = p.get("width"), p.get("height")
    return f"{w}x{h}" if w and h else ""


def kind_of(task: str, params: dict | None) -> str:
    """The kind a run is timed as: the same task, model and size take about the same time.

    image: model, aspect, resolution, count. video: mode, size and frames. sheet: from a picture or from words. The
    start frame and the look frame: their method. Any other task (a shot's take): the task and its engine."""
    p = params or {}
    if task == "image":
        return "|".join(["image", str(p.get("model") or ""), str(p.get("aspect") or "16:9"), str(p.get("resolution") or "1K"),
                         str(p.get("count") or 1)])
    if task == "video":
        return "|".join(["video", str(p.get("mode") or "text"), _size(p) or f"{p.get('aspect', '')}@{p.get('resolution', '')}",
                         str(p.get("frames") or p.get("duration") or "")])
    if task == "sheet":
        return "|".join(["sheet", "words" if p.get("describe") and not (p.get("upload_id") or p.get("image_id") or p.get("cast_id")) else "picture"])
    if task in ("start_frame", "look_frame"):
        return "|".join([task, str(p.get("method") or "")])
    return "|".join([task, str(p.get("engine") or p.get("model") or "")])


class Timings:
    def __init__(self, file: Path):
        self.file = Path(file)

    def _load(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self.file.read_text())
            return {k: [float(x) for x in v] for k, v in data.items() if isinstance(v, list)}
        except (OSError, ValueError, TypeError):
            return {}

    def record(self, kind: str, seconds: float) -> None:
        """Add one measured run. Never raises: a full disk must not fail a job."""
        if not kind or not seconds or seconds <= 0:
            return
        try:
            with _lock:
                data = self._load()
                data[kind] = (data.get(kind, []) + [round(float(seconds), 1)])[-KEEP:]
                self.file.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=1))
                tmp.replace(self.file)
        except OSError:
            pass

    def runs(self, kind: str) -> list[float]:
        return self._load().get(kind, [])

    def usual(self, kind: str) -> int | None:
        """The median of the last few measured runs of this kind, in whole seconds; None before the first one."""
        got = self.runs(kind)[-MEDIAN_OF:]
        return max(1, round(statistics.median(got))) if got else None
