"""Find ComfyUI on the network: "Find ComfyUI on my network" on the first screen.

Scope, and nothing more:
- this computer's own private networks (10/8, 172.16/12, 192.168/16) on its active interfaces, at most a /24 each;
- the devices in this computer's Tailscale network, read from `tailscale status --json` when the CLI is installed
  (the only Tailscale command it runs).
Every address is checked before it's tried: never a public one. Each host gets one GET of /system_stats (the same
check servers.test makes), with a short timeout and a bounded number at once.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from pickups.app import servers

PRIVATE = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
PORTS = tuple(int(p) for p in os.environ.get("PICKUPS_SCAN_PORTS", str(servers.PORT)).split(",") if p.strip())
TIMEOUT = 0.4  # seconds per host: a computer on the same network answers in a few ms
PATIENT_TIMEOUT = 1.5  # the second look, when the first found nothing: a slow Wi-Fi, a busy GPU computer
PATIENT_WORKERS = 128
WORKERS = 64
MAX_PREFIX = 24  # a wider network is cut to the /24 around this computer
MAX_NETWORKS = 4
MAX_PEERS = 64
MAX_HOSTS = MAX_NETWORKS * 254 + MAX_PEERS  # the hard cap on hosts tried in one scan
TAILSCALE_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"

_lock = threading.Lock()  # one scan at a time; a second click waits for the first


def is_private(ip: ipaddress.IPv4Address) -> bool:
    return any(ip in n for n in PRIVATE)


def _tailnet_ok(ip: ipaddress.IPv4Address) -> bool:
    """A Tailscale peer's address: its shared (carrier-grade NAT) range or a private one, never a public address."""
    return not (ip.is_global or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified)


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


# --- this computer's networks ------------------------------------------------------------------------------------------

def parse_ifconfig(text: str) -> list[ipaddress.IPv4Interface]:
    """macOS/BSD `ifconfig`: "inet 192.168.1.5 netmask 0xffffff00", only for interfaces that aren't inactive."""
    out: list[ipaddress.IPv4Interface] = []
    for block in re.split(r"\n(?=\S)", text):
        if "status: inactive" in block:
            continue
        for ip, mask in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)(?: --> \S+)? netmask (0x[0-9a-fA-F]+|\d+\.\d+\.\d+\.\d+)", block):
            m = str(ipaddress.IPv4Address(int(mask, 16))) if mask.startswith("0x") else mask
            out.append(ipaddress.IPv4Interface(f"{ip}/{m}"))
    return out


def parse_ip_addr(text: str) -> list[ipaddress.IPv4Interface]:
    """Linux `ip -4 -o addr show up`: "inet 192.168.1.5/24"."""
    return [ipaddress.IPv4Interface(a) for a in re.findall(r"inet (\d+\.\d+\.\d+\.\d+/\d+)", text)]


def parse_ipconfig(text: str) -> list[ipaddress.IPv4Interface]:
    """Windows `ipconfig`: each "IPv4 Address" line followed by its "Subnet Mask" line."""
    out: list[ipaddress.IPv4Interface] = []
    ip = None
    for line in text.splitlines():
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
        if not m:
            continue
        if "IPv4" in line:
            ip = m.group(1)
        elif ip and ("Mask" in line or "mask" in line):
            try:
                out.append(ipaddress.IPv4Interface(f"{ip}/{m.group(1)}"))
            except ValueError:
                pass
            ip = None
    return out


def _route_address() -> ipaddress.IPv4Interface | None:
    """The address this computer would use to reach a private network. A UDP "connect" sends nothing."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 9))
            return ipaddress.IPv4Interface(f"{s.getsockname()[0]}/{MAX_PREFIX}")
    except OSError:
        return None


def interfaces() -> list[ipaddress.IPv4Interface]:
    """This computer's IPv4 addresses with their netmasks (any kind; networks() keeps the private ones)."""
    if sys.platform == "win32":
        found = parse_ipconfig(_run(["ipconfig"]))
    elif sys.platform.startswith("linux") and shutil.which("ip"):
        found = parse_ip_addr(_run(["ip", "-4", "-o", "addr", "show", "up"]))
    else:
        found = parse_ifconfig(_run(["ifconfig"]))
    route = _route_address()
    if route and route.ip not in {i.ip for i in found}:
        found.append(route)
    return found


def networks(addrs: list[ipaddress.IPv4Interface]) -> list[ipaddress.IPv4Network]:
    """The private networks to scan: each at most a /24 (the one around this computer), at most MAX_NETWORKS."""
    nets: list[ipaddress.IPv4Network] = []
    for a in addrs:
        if not is_private(a.ip):
            continue
        net = a.network if a.network.prefixlen >= MAX_PREFIX else ipaddress.ip_interface(f"{a.ip}/{MAX_PREFIX}").network
        if net.prefixlen > 30 or net in nets:  # a point-to-point link has no one else on it
            continue
        nets.append(net)
    return nets[:MAX_NETWORKS]


# --- Tailscale ---------------------------------------------------------------------------------------------------------

def tailscale_cli() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return TAILSCALE_APP if sys.platform == "darwin" and os.path.exists(TAILSCALE_APP) else None


def parse_tailscale(text: str) -> list[dict]:
    """The online peers in `tailscale status --json`: [{"name", "ip"}], IPv4 only, never a public address."""
    try:
        peers = (json.loads(text or "{}").get("Peer") or {}).values()
    except (ValueError, AttributeError):
        return []
    out: list[dict] = []
    for p in peers:
        if not isinstance(p, dict) or p.get("Online") is False:
            continue
        for raw in p.get("TailscaleIPs") or []:
            try:
                ip = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if ip.version == 4 and _tailnet_ok(ip):
                name = p.get("HostName") or (p.get("DNSName") or "").split(".")[0] or str(ip)
                out.append({"name": name, "ip": str(ip)})
                break
    return out[:MAX_PEERS]


def tailscale_peers() -> list[dict]:
    cli = tailscale_cli()
    return parse_tailscale(_run([cli, "status", "--json"])) if cli else []


# --- the scan ----------------------------------------------------------------------------------------------------------

def targets(addrs: list[ipaddress.IPv4Interface], peers: list[dict]) -> list[dict]:
    """Every host to try, as {"ip", "name", "self"}: the private networks first, then the Tailscale peers."""
    mine = {str(a.ip) for a in addrs}
    seen: set[str] = set()
    out: list[dict] = []
    for net in networks(addrs):
        for ip in net.hosts():
            s = str(ip)
            if s not in seen:
                seen.add(s)
                out.append({"ip": s, "name": "This computer" if s in mine else s, "self": s in mine})
    for p in peers:
        ip = ipaddress.ip_address(p["ip"])
        if p["ip"] not in seen and ip.version == 4 and _tailnet_ok(ip):
            seen.add(p["ip"])
            out.append({"ip": p["ip"], "name": p["name"], "self": p["ip"] in mine})
    return out[:MAX_HOSTS]


def _probe(t: dict, port: int, timeout: float) -> dict | None:
    res = servers.test(f"http://{t['ip']}:{port}", timeout=timeout)
    return res | {"name": t["name"], "self": t["self"]} if res["ok"] else None


def scan(addrs: list[ipaddress.IPv4Interface] | None = None, peers: list[dict] | None = None,
         ports: tuple[int, ...] = PORTS, timeout: float = TIMEOUT) -> dict:
    """ComfyUI on this computer's private networks and its Tailscale devices: {"found": [test + name, self], "tried",
    and what was looked at, said plainly when nothing answers: "networks" (as 192.168.1.0/24), "peers" (a count),
    "ports" and "timeout"}."""
    with _lock:
        addrs = interfaces() if addrs is None else addrs
        peers = tailscale_peers() if peers is None else peers
        todo = [(t, p) for t in targets(addrs, peers) for p in ports]
        found: list[dict] = []
        workers = PATIENT_WORKERS if timeout > 1 else WORKERS
        if todo:
            with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
                found = [r for r in ex.map(lambda tp: _probe(tp[0], tp[1], timeout), todo) if r]
        return {"found": found, "tried": len(todo), "networks": [str(n) for n in networks(addrs)], "peers": len(peers),
                "ports": list(ports), "timeout": timeout}
