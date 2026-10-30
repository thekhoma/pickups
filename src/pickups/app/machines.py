"""Saved GPU machines: this computer, a PC on the LAN, or a rented RunPod pod.

settings.json keeps `machines` ([{id, name, kind, url}]) and `active_machine` (an id). `comfy_url` stays in step with
the active machine, so older builds still find it. A settings file from before machines existed gets one machine
from its `comfy_url`. Nothing secret is stored: a RunPod pod is reached through RunPod's public HTTP proxy for its
port, `https://<pod-id>-<port>.proxy.runpod.net`, so no API key is needed.
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from pickups.app import bench, comfy
from pickups.app.comfy import _get_json

DEFAULT_URL = "http://127.0.0.1:8188"
KINDS = ("local", "lan", "runpod")
RUNPOD_PORT = 8188
_POD_ID = re.compile(r"^[a-z0-9]{6,32}$")
_RUNPOD_HOST = re.compile(r"^([a-z0-9]+)-(\d{1,5})\.proxy\.runpod\.net$")


class Busy(Exception):
    """A job is running on the current machine, so it can't be switched now."""


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def kind_of(url: str) -> str:
    host = urlparse(url).hostname or ""
    if host.endswith(".proxy.runpod.net"):
        return "runpod"
    return "local" if _is_local(url) else "lan"


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _id_path() -> Path:
    return comfy.HOME / "machine_id"


def first_id() -> str:
    """The id of the machine settings.json stands for before any machine is saved: made once and kept in the app home,
    so /api/settings and /api/machines answer with the same id on every call."""
    path = _id_path()
    try:
        saved = path.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{8}", saved):
            return saved
    except OSError:
        pass
    mid = new_id()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mid + "\n")
    except OSError:  # a read-only home still works, with a new id each time
        pass
    return mid


def runpod_url(text: str, port: int | str | None = None) -> str:
    """A pod id ("abc123xyz") or its proxy address ("https://abc123xyz-8188.proxy.runpod.net/...") → the ComfyUI URL.
    A port given here wins over one in the address. Raises ValueError with a plain message."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Paste a pod ID or its address.")
    found_port = None
    if "." in raw or "/" in raw:
        host = (urlparse(raw if "://" in raw else f"https://{raw}").hostname or "").lower()
        m = _RUNPOD_HOST.match(host)
        if not m:
            raise ValueError("That isn't a RunPod address. It looks like https://<pod-id>-8188.proxy.runpod.net")
        pod, found_port = m.group(1), int(m.group(2))
    else:
        pod = raw.lower()
    if not _POD_ID.match(pod):
        raise ValueError("A pod ID is letters and numbers only, like abc123xyz.")
    p = port if port not in (None, "") else (found_port or RUNPOD_PORT)
    try:
        p = int(p)
    except (TypeError, ValueError):
        raise ValueError("The port is a number, like 8188.") from None
    if not 1 <= p <= 65535:
        raise ValueError("The port is a number from 1 to 65535.")
    return f"https://{pod}-{p}.proxy.runpod.net"


def normalize(st: dict) -> tuple[list[dict], str]:
    """The saved machines and the active id, migrating an older settings file (comfy_url only) to one machine."""
    machines = [m for m in (st.get("machines") or []) if isinstance(m, dict) and m.get("url") and m.get("id")]
    if not machines:
        url = (st.get("comfy_url") or DEFAULT_URL).rstrip("/")
        kind = kind_of(url)
        name = st.get("machine_name") or ("This computer" if kind == "local" else "RunPod" if kind == "runpod" else "GPU PC")
        machines = [{"id": first_id(), "name": name, "kind": kind, "url": url}]
    active = st.get("active_machine")
    if active not in {m["id"] for m in machines}:
        cur = (st.get("comfy_url") or "").rstrip("/")
        active = next((m["id"] for m in machines if m["url"] == cur), machines[0]["id"])
    return machines, active


def with_machines(st: dict, machines: list[dict], active: str) -> dict:
    url = next(m["url"] for m in machines if m["id"] == active)
    return st | {"machines": machines, "active_machine": active, "comfy_url": url}


def active_machine(st: dict) -> dict:
    machines, active = normalize(st)
    return next(m for m in machines if m["id"] == active)


def probe(url: str, timeout: float = 6.0) -> dict:
    """Reachable? GPU and VRAM from /system_stats. Never raises."""
    try:
        stats = _get_json(f"{url.rstrip('/')}/system_stats", timeout=timeout)
    except Exception:  # noqa: BLE001 - offline or a stopped pod is a normal state
        return {"reachable": False, "gpu": None, "vram_gb": None}
    dev = (stats.get("devices") or [{}])[0]
    return {"reachable": True, "gpu": bench.gpu_name(dev.get("name", "")) or None,
            "vram_gb": round(dev.get("vram_total", 0) / 1e9, 1) or None}


def probe_all(machines: list[dict]) -> dict[str, dict]:
    if not machines:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(machines))) as ex:
        return dict(zip([m["id"] for m in machines], ex.map(lambda m: probe(m["url"]), machines)))


def switch(conn, jobs: list[dict], url: str) -> str:
    """Point the connection at another ComfyUI (restarting the API proxy for it), unless a job is running."""
    if any(j.get("status") == "running" for j in jobs):
        raise Busy("A job is running. Switch machines when it finishes.")
    if url.rstrip("/") == conn.comfy_url.rstrip("/"):
        return conn.comfy_url
    conn.switch(url)
    return conn.comfy_url
