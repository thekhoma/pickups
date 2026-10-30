"""Find ComfyUI on my network: what the scan may touch (private networks, Tailscale peers, never a public address),
its caps, its timeouts, and a live run against a stub ComfyUI that answers /system_stats."""

import ipaddress
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pickups.app import discover, servers

STATS = {"system": {"os": "nt", "ram_total": 64 * 2**30, "comfyui_version": "0.41.0"},
         "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda", "vram_total": 24 * 2**30}]}
# A Tailscale address, built from the shared range's first octets so no real device address is written down.
SHARED = str(ipaddress.IPv4Address((100 << 24) | (64 << 16) | 7))
IF = ipaddress.IPv4Interface

IFCONFIG = """lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.1.5 netmask 0xffffff00 broadcast 192.168.1.255
\tstatus: active
en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 10.0.0.9 netmask 0xffffff00 broadcast 10.0.0.255
\tstatus: inactive
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.3.4 netmask 0xffff0000 broadcast 169.254.255.255
utun4: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 172.20.0.2 --> 172.20.0.2 netmask 0xffffffff
"""
IPCONFIG = """Ethernet adapter Ethernet:
   IPv4 Address. . . . . . . . . . . : 192.168.50.20
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.50.1
"""


# --- this computer's networks ------------------------------------------------------------------------------------------
def test_interfaces_are_read_on_each_os():
    mac = discover.parse_ifconfig(IFCONFIG)
    assert IF("192.168.1.5/24") in mac and IF("10.0.0.9/24") not in mac  # an inactive interface is skipped
    assert IF("172.20.0.2/32") in mac  # a point-to-point link (networks() drops it)
    assert discover.parse_ip_addr("2: eth0    inet 10.1.2.3/16 brd 10.1.255.255 scope global eth0") == [IF("10.1.2.3/16")]
    assert discover.parse_ipconfig(IPCONFIG) == [IF("192.168.50.20/24")]


def test_only_private_networks_are_scanned_and_each_is_at_most_a_slash_24():
    nets = discover.networks([IF("127.0.0.1/8"), IF("8.8.8.8/24"), IF("169.254.3.4/16"), IF(f"{SHARED}/32"),
                              IF("10.1.2.3/8"), IF("172.20.0.2/32"), IF("192.168.1.5/24"), IF("172.16.4.4/28")])
    assert nets == [ipaddress.ip_network("10.1.2.0/24"), ipaddress.ip_network("192.168.1.0/24"),
                    ipaddress.ip_network("172.16.4.0/28")]
    many = discover.networks([IF(f"10.{i}.0.1/24") for i in range(9)])
    assert len(many) == discover.MAX_NETWORKS


# --- Tailscale ---------------------------------------------------------------------------------------------------------
def test_tailscale_peers_are_online_ipv4_and_never_public():
    status = {"Peer": {
        "a": {"HostName": "gpu-pc", "DNSName": "gpu-pc.example.ts.net.", "Online": True, "TailscaleIPs": [SHARED, "fd7a::1"]},
        "b": {"HostName": "laptop", "Online": False, "TailscaleIPs": [SHARED]},
        "c": {"HostName": "odd", "Online": True, "TailscaleIPs": ["8.8.8.8"]},
        "d": {"DNSName": "render.example.ts.net.", "Online": True, "TailscaleIPs": ["192.168.9.9"]},
    }}
    assert discover.parse_tailscale(json.dumps(status)) == [{"name": "gpu-pc", "ip": SHARED}, {"name": "render", "ip": "192.168.9.9"}]
    assert discover.parse_tailscale("not json") == [] and discover.parse_tailscale("") == []


def test_the_only_tailscale_command_is_status_json(monkeypatch):
    ran = []
    monkeypatch.setattr(discover, "tailscale_cli", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(discover, "_run", lambda cmd, timeout=3.0: ran.append(cmd) or "{}")
    assert discover.tailscale_peers() == [] and ran == [["/usr/bin/tailscale", "status", "--json"]]
    ran.clear()
    monkeypatch.setattr(discover, "tailscale_cli", lambda: None)
    assert discover.tailscale_peers() == [] and ran == []


# --- what gets tried ---------------------------------------------------------------------------------------------------
def test_targets_never_include_a_public_address_and_are_capped():
    peers = [{"name": "gpu-pc", "ip": SHARED}, {"name": "web", "ip": "1.1.1.1"}, {"name": "dup", "ip": "192.168.1.20"}]
    ts = discover.targets([IF("192.168.1.5/24"), IF("203.0.113.5/24")], peers)
    ips = [ipaddress.ip_address(t["ip"]) for t in ts]
    assert all(not ip.is_global for ip in ips)
    assert len(ts) == 254 + 1  # the /24 and the one Tailscale peer; the duplicate and the public one are dropped
    assert next(t for t in ts if t["ip"] == "192.168.1.5") == {"ip": "192.168.1.5", "name": "This computer", "self": True}
    assert ts[-1] == {"ip": SHARED, "name": "gpu-pc", "self": False}
    wide = discover.targets([IF(f"10.{i}.0.1/16") for i in range(9)], [{"name": f"p{i}", "ip": f"192.168.200.{i + 1}"} for i in range(99)])
    assert len(wide) <= discover.MAX_HOSTS


def test_each_host_gets_one_system_stats_get_with_the_short_timeout(monkeypatch):
    calls = []

    def fake_test(url, timeout=5.0):
        calls.append((url, timeout))
        return {"ok": url.startswith("http://192.168.1.20:"), "url": url, "gpu": "RTX 4090"}

    monkeypatch.setattr(servers, "test", fake_test)
    out = discover.scan([IF("192.168.1.5/24")], [{"name": "gpu-pc", "ip": SHARED}], ports=(8188,))
    assert out["tried"] == 255 and len(calls) == 255
    assert all(t == discover.TIMEOUT and u.endswith(":8188") for u, t in calls)
    assert [(f["url"], f["name"], f["self"]) for f in out["found"]] == [("http://192.168.1.20:8188", "192.168.1.20", False)]
    assert discover.TIMEOUT <= 0.5 and discover.WORKERS <= 64


# --- live, against stubs -----------------------------------------------------------------------------------------------
class StubComfy(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        StubComfy.seen.append(("GET", self.path))
        body = json.dumps(STATS).encode() if self.path == "/system_stats" else b"{}"
        self.send_response(200 if self.path == "/system_stats" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        StubComfy.seen.append(("POST", self.path))
        self.send_response(405)
        self.end_headers()


class Silent(StubComfy):
    def do_GET(self):
        time.sleep(2)  # accepts, never answers in time


def stub(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


@pytest.fixture()
def loopback_is_the_lan(monkeypatch):
    """The stubs listen on 127.0.0.1, so for these tests that one address stands in for a private network."""
    monkeypatch.setattr(discover, "PRIVATE", (*discover.PRIVATE, ipaddress.ip_network("127.0.0.0/30")))
    return [IF("127.0.0.1/30")]


def test_a_stub_comfyui_is_found_with_its_gpu_and_only_asked_for_system_stats(loopback_is_the_lan):
    srv, port = stub(StubComfy)
    StubComfy.seen = []
    try:
        out = discover.scan(loopback_is_the_lan, [], ports=(port,))
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["tried"] == 2  # 127.0.0.1 and 127.0.0.2: the /30's hosts
    assert [(f["url"], f["gpu"], f["vram_gb"], f["self"], f["name"]) for f in out["found"]] == [
        (f"http://127.0.0.1:{port}", "RTX 4090", 25.8, True, "This computer")]
    assert StubComfy.seen == [("GET", "/system_stats")]


def test_nothing_found_and_a_silent_host_times_out_fast(loopback_is_the_lan):
    srv, port = stub(Silent)
    try:
        t0 = time.monotonic()
        out = discover.scan(loopback_is_the_lan, [], ports=(port,))
        took = time.monotonic() - t0
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["found"] == [] and out["tried"] == 2
    # What was looked at, for the plain line under the button (Run 14): the network, no peers, the port, the wait.
    assert out["networks"] and out["peers"] == 0 and out["ports"] == [port] and out["timeout"] == discover.TIMEOUT
    assert took < 1.5  # 0.4 s each, at once, never the 2 s the silent host would take


def test_the_scan_route_answers(monkeypatch, tmp_path):
    import pickups.app.server as server
    from fastapi.testclient import TestClient

    from pickups.app import comfy

    monkeypatch.setattr(comfy, "HOME", tmp_path)
    monkeypatch.setattr(comfy, "SETTINGS", tmp_path / "settings.json")
    waits = []

    def fake(timeout=discover.TIMEOUT):
        waits.append(timeout)
        return {"found": [{"ok": True, "url": "http://192.168.1.20:8188", "name": "192.168.1.20"}], "tried": 254}

    monkeypatch.setattr(discover, "scan", fake)
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    assert c.get("/api/servers/scan").json()["found"][0]["url"] == "http://192.168.1.20:8188"
    # The second, patient look (after one that found nothing) waits 1.5 s for each computer.
    c.get("/api/servers/scan?patient=1")
    assert waits == [discover.TIMEOUT, discover.PATIENT_TIMEOUT] and discover.PATIENT_TIMEOUT == 1.5
