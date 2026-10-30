"""Use on your phone: Pickups on this computer's Tailscale address, so a phone signed in to the same tailnet opens it from
anywhere. Off by default. When on, Pickups also listens on the Tailscale IP only (never 0.0.0.0), so nothing on the
local network or the internet can reach it that Tailscale doesn't already let in.

Detection is read-only: `tailscale status --json`, else a 100.64.0.0/10 address on a network interface.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

CGNAT = ipaddress.ip_network("100.64.0.0/10")  # the range Tailscale gives each device
CLI = ["tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale", r"C:\Program Files\Tailscale\tailscale.exe"]


def _tailnet(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip) in CGNAT
    except ValueError:
        return False


def detect() -> dict | None:
    """{"ip": "100.x.y.z", "name": "my-mac.tail1234.ts.net" or None} when Tailscale is up on this computer, else None.
    PICKUPS_TAILSCALE=0 turns phone access off entirely (it's then never offered)."""
    import os

    if os.environ.get("PICKUPS_TAILSCALE", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    for cli in CLI:
        exe = shutil.which(cli) or (cli if Path(cli).exists() else None)
        if not exe:
            continue
        try:
            out = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=3, check=False)
            data = json.loads(out.stdout or "{}")
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        me = data.get("Self") or {}
        ip = next((a for a in me.get("TailscaleIPs") or [] if _tailnet(a)), None)
        if ip and data.get("BackendState", "Running") == "Running":
            return {"ip": ip, "name": (me.get("DNSName") or "").rstrip(".") or None}
    # No CLI: a CGNAT address on an interface (Tailscale's), without its name.
    cmd = ["ipconfig"] if sys.platform.startswith("win") else ["ifconfig"] if shutil.which("ifconfig") else ["ip", "-4", "addr"]
    try:
        text = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    ip = next((a for a in re.findall(r"(?<![\d.])(100\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\d.])", text) if _tailnet(a)), None)
    return {"ip": ip, "name": None} if ip else None


class PhoneListener:
    """A second uvicorn server for the same app, bound to the Tailscale IP only. Lifespan off: the app is already up."""

    def __init__(self) -> None:
        self._server = None
        self._thread: threading.Thread | None = None
        self.bound: str | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, app, ip: str, port: int) -> None:
        if not _tailnet(ip):
            raise ValueError("Not a Tailscale address.")
        if self.running and self.bound == ip:
            return
        self.stop()
        import uvicorn

        self._server = uvicorn.Server(uvicorn.Config(app, host=ip, port=port, log_level="warning", lifespan="off"))
        self._thread = threading.Thread(target=self._server.run, name="pickups-phone", daemon=True)
        self._thread.start()
        self.bound = ip

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = self._thread = None
        self.bound = None
