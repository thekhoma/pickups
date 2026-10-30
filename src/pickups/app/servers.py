"""Connecting to a ComfyUI server: read an address the way people type it, test it without saving, find one on this
computer, and remember the last five used.

Pickups runs on any computer and points at any ComfyUI: this one, another on the network (an IP or a Tailscale name,
"http://gpu-pc:8188"), or a RunPod pod through its HTTPS proxy. The browser never talks to ComfyUI; Pickups' server does
(and passes ComfyUI's own pages through /comfy/), so ComfyUI needs no CORS settings, only --listen to be reached from
another computer.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from pickups.app import bench, machines

PORT = 8188
LOCAL = ("http://127.0.0.1:8188", "http://127.0.0.1:8000")  # ComfyUI portable/manual, then ComfyUI Desktop's default
RECENT_MAX = 5
UA = {"User-Agent": "Pickups/1.0"}  # RunPod's proxy (Cloudflare) refuses urllib's default agent


def address(text: str) -> str:
    """The ComfyUI URL for what someone typed: "gpu-pc", "gpu-pc:8188", "192.168.1.20", "http://gpu-pc:8188/",
    a RunPod proxy address or pod ID ("abc123xyz-8188.proxy.runpod.net"). Raises ValueError with a plain message."""
    raw = (text or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Type the address of the computer running ComfyUI, like http://gpu-pc:8188")
    if "proxy.runpod.net" in raw:
        return machines.runpod_url(raw)
    url = raw if "://" in raw else f"http://{raw}"
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname or " " in raw:
        raise ValueError("That doesn't look like an address. Try http://gpu-pc:8188 or http://192.168.1.20:8188")
    port = f":{p.port}" if p.port else ("" if p.scheme == "https" else f":{PORT}")
    host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
    return f"{p.scheme}://{host}{port}"


def label(url: str) -> str:
    """How the app names a server: "This computer", its host ("gpu-pc", "192.168.1.20") or "RunPod <pod-id>"."""
    host = urlparse(url).hostname or url
    kind = machines.kind_of(url)
    if kind == "local":
        return "This computer"
    return f"RunPod {host.split('-')[0]}" if kind == "runpod" else host


def test(url: str, timeout: float = 5.0) -> dict:
    """Reach ComfyUI at `url` and read /system_stats. Never raises. `reason` when it fails: "dns" (name not found),
    "refused" (nothing listening), "timeout", "not_comfy" (something else answered), "http" (an error status)."""
    t0 = time.monotonic()
    out: dict = {"ok": False, "url": url, "kind": machines.kind_of(url), "reason": None, "message": ""}
    try:
        req = urllib.request.Request(f"{url}/system_stats", headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the address the person typed
            body = r.read()
        stats = json.loads(body)
        if not isinstance(stats, dict) or "system" not in stats:
            raise ValueError("not ComfyUI")
    except urllib.error.HTTPError as e:
        if out["kind"] == "runpod":
            return out | {"reason": "http", "message": "No running pod answers at that address. Check the pod is running "
                                                       "and that it's the address for port 8188."}
        return out | {"reason": "http", "message": f"The server answered with an error ({e.code}). Is this ComfyUI?"}
    except urllib.error.URLError as e:
        r = e.reason
        if isinstance(r, socket.gaierror):
            return out | {"reason": "dns", "message": f"Can't find a computer called {urlparse(url).hostname}."}
        if isinstance(r, (TimeoutError, socket.timeout)):
            return out | {"reason": "timeout", "message": "That computer didn't answer. It may be off, or a firewall is blocking the port."}
        if isinstance(r, ConnectionRefusedError):
            return out | {"reason": "refused", "message": "That computer is there, but ComfyUI isn't listening on this port."}
        return out | {"reason": "refused", "message": "Couldn't connect to that address."}
    except (TimeoutError, socket.timeout):
        return out | {"reason": "timeout", "message": "That computer didn't answer. It may be off, or a firewall is blocking the port."}
    except (ValueError, json.JSONDecodeError):
        return out | {"reason": "not_comfy", "message": "Something answered at that address, but it isn't ComfyUI."}
    except Exception:  # noqa: BLE001 - any other failure is "couldn't connect", never a crash
        return out | {"reason": "refused", "message": "Couldn't connect to that address."}
    return out | {"ok": True, "ms": round((time.monotonic() - t0) * 1000), **facts(stats)}


def facts(stats: dict) -> dict:
    """GPU, VRAM, RAM, OS and ComfyUI version from a /system_stats answer."""
    dev = (stats.get("devices") or [{}])[0]
    sysi = stats.get("system") or {}
    return {"gpu": bench.gpu_name(dev.get("name", "")) or None, "gpu_raw": dev.get("name") or None,
            "device": dev.get("type") or None,
            "vram_gb": round(dev.get("vram_total", 0) / 1e9, 1) or None,  # decimal GB, the unit every gate uses (4090: 25.8)
            "ram_gb": round(sysi.get("ram_total", 0) / 1e9, 1) or None,
            "os": sysi.get("os") or None, "comfy_version": sysi.get("comfyui_version") or None}


def detect() -> dict:
    """ComfyUI on this computer: the first local address that answers, and every one tried."""
    tried = [test(u, timeout=2.0) for u in LOCAL]
    return {"found": next((t for t in tried if t["ok"]), None), "tried": tried}


def recents(st: dict) -> list[dict]:
    return [r for r in st.get("recent_servers") or [] if isinstance(r, dict) and r.get("url")][:RECENT_MAX]


def remember(st: dict, url: str, gpu: str | None = None) -> dict:
    """Settings with `url` first in the recent servers (at most RECENT_MAX)."""
    rest = [r for r in recents(st) if r["url"] != url]
    entry = {"url": url, "kind": machines.kind_of(url), "gpu": gpu, "used_at": round(time.time())}
    return st | {"recent_servers": [entry] + rest[: RECENT_MAX - 1]}


def listen_commands(port: int = PORT) -> list[dict]:
    """How to start ComfyUI so other computers can reach it, per OS (run in the ComfyUI folder)."""
    return [
        {"os": "Windows", "label": "ComfyUI portable",
         "command": rf".\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0 --port {port}",
         "note": "Run it in the ComfyUI_windows_portable folder. Allow Python through Windows Firewall when asked."},
        {"os": "Mac", "label": "ComfyUI from source",
         "command": f"python main.py --listen 0.0.0.0 --port {port}",
         "note": "Run it in the ComfyUI folder, with its Python environment active."},
        {"os": "Linux", "label": "ComfyUI from source",
         "command": f"python main.py --listen 0.0.0.0 --port {port}",
         "note": "Run it in the ComfyUI folder, with its Python environment active. Open the port if a firewall is on."},
        {"os": "Desktop app", "label": "ComfyUI Desktop (Windows or Mac)",
         "command": "Settings → Server-Config → Host: 0.0.0.0",
         "note": "Set the host to 0.0.0.0, restart ComfyUI Desktop, then use its port (8000 by default)."},
    ]
