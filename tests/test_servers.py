"""Connecting to any ComfyUI: addresses as people type them, testing without saving, the recent five, the capability
scan (measured numbers only) and the /comfy/ pass-through (REST, uploads, /view, websocket)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from pickups.app import comfy, measured, servers

STATS = {"system": {"os": "nt", "ram_total": 64 * 2**30, "comfyui_version": "0.41.0"},
         "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda", "vram_total": 24 * 2**30}]}


class FakeComfy(BaseHTTPRequestHandler):
    seen: list = []  # each request's headers, as (name, value) pairs

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeComfy.seen.append(list(self.headers.items()))
        if self.path == "/system_stats":
            return self._send(200, json.dumps(STATS).encode())
        if self.path.startswith("/view?filename=frame.png"):
            return self._send(200, b"\x89PNG-bytes", "image/png")
        if self.path == "/internal/folder_paths":
            return self._send(200, json.dumps({"checkpoints": ["/nowhere/ComfyUI/models/checkpoints"]}).encode())
        return self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        if self.path == "/upload/image":
            return self._send(200, json.dumps({"name": "up.png", "bytes": len(data)}).encode())
        return self._send(404, b"{}")


class NotComfy(FakeComfy):
    def do_GET(self):
        return self._send(200, b"<html>router</html>", "text/html")


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture()
def fake_comfy():
    srv, url = serve(FakeComfy)
    yield url
    srv.shutdown()


# --- addresses and tests ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("typed, want", [
    ("gpu-pc", "http://gpu-pc:8188"), ("gpu-pc:8188", "http://gpu-pc:8188"), ("http://gpu-pc:8188/", "http://gpu-pc:8188"),
    ("192.168.1.20", "http://192.168.1.20:8188"), ("http://192.168.1.20:8000", "http://192.168.1.20:8000"),
    ("https://abc123xyz-8188.proxy.runpod.net/", "https://abc123xyz-8188.proxy.runpod.net"),
])
def test_addresses_are_read_the_way_people_type_them(typed, want):
    assert servers.address(typed) == want


@pytest.mark.parametrize("typed", ["", "   ", "ftp://gpu-pc", "gpu-pc 8188"])
def test_a_bad_address_says_what_to_type(typed):
    with pytest.raises(ValueError):
        servers.address(typed)


def test_a_comfyui_answers_with_its_gpu(fake_comfy):
    r = servers.test(fake_comfy)
    assert r["ok"] and r["gpu"] == "RTX 4090" and r["vram_gb"] == 25.8 and r["ram_gb"] == 68.7  # decimal GB, as every gate reads it and r["comfy_version"] == "0.41.0"


def test_failures_say_why():
    srv, url = serve(NotComfy)
    try:
        assert servers.test(url)["reason"] == "not_comfy"
    finally:
        srv.shutdown()
    assert servers.test("http://127.0.0.1:9")["reason"] == "refused"
    assert servers.test("http://no-such-host.invalid:8188")["reason"] == "dns"
    cmds = {c["os"]: c["command"] for c in servers.listen_commands()}
    assert "--listen 0.0.0.0 --port 8188" in cmds["Windows"] and "--listen 0.0.0.0" in cmds["Mac"] and "--listen" in cmds["Linux"]


def test_the_recent_five_are_kept_newest_first():
    st: dict = {}
    for i in range(7):
        st = servers.remember(st, f"http://pc{i}:8188", "RTX 4090")
    st = servers.remember(st, "http://pc3:8188")
    urls = [r["url"] for r in servers.recents(st)]
    assert urls == ["http://pc3:8188", "http://pc6:8188", "http://pc5:8188", "http://pc4:8188", "http://pc2:8188"]


def test_only_measured_numbers_are_shown():
    row = measured.for_task("cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "recast")
    assert row["gpu"] == "RTX 4090" and row["max_frames"] == 145 and row["max_seconds"] == round(145 / 29.97, 1)
    assert measured.for_task("cuda:0 NVIDIA GeForce RTX 3090 : cudaMallocAsync", "recast") is None  # not tested here
    assert measured.for_task("cuda:0 NVIDIA GeForce RTX 4090", "finish") is None  # no measured run with its GPU recorded
    assert measured.gpu_key("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb") is None  # a slice isn't the card


# --- the routes --------------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch, fake_comfy):
    import pickups.app.server as server
    import pickups.app.store as st
    from pickups.app.store import Store
    from fastapi.testclient import TestClient

    monkeypatch.setattr(comfy, "HOME", tmp_path)
    monkeypatch.setattr(comfy, "SETTINGS", tmp_path / "settings.json")
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    monkeypatch.setattr(comfy.Connection, "ensure_proxy", lambda self: "http://127.0.0.1:1")
    monkeypatch.setattr(comfy.Connection, "switch", lambda self, url: setattr(self, "comfy_url", url.rstrip("/")))
    monkeypatch.setattr(server.doctor, "status", lambda conn: {"connected": True, "verdict": "ready", "comfy_url": conn.comfy_url,
                                                               "machine": {}, "tiles": {"recast": {"ready": True, "missing": []}}})
    (tmp_path / "settings.json").write_text(json.dumps({}))
    return TestClient(server.create_app("http://127.0.0.1:9")), fake_comfy


def test_first_open_has_no_welcome_done_and_connecting_remembers_the_server(client, monkeypatch):
    c, url = client
    import pickups.app.server as server
    fresh = TestClient(server.create_app(None))
    assert fresh.get("/api/servers").json()["welcome_done"] is False
    assert c.get("/api/servers").json()["welcome_done"] is True  # started with --comfy: already chosen
    bad = c.post("/api/servers/test", json={"address": "http://127.0.0.1:9"}).json()
    assert not bad["ok"] and bad["reason"] == "refused" and bad["listen"]
    assert c.post("/api/servers/use", json={"address": "http://127.0.0.1:9"}).status_code == 409  # not switched to
    good = c.post("/api/servers/test", json={"address": url}).json()
    assert good["ok"] and good["gpu"] == "RTX 4090"
    r = c.post("/api/servers/use", json={"address": url}).json()
    assert r["active"]["url"] == url and r["welcome_done"] is True and r["recent"][0]["url"] == url
    saved = json.loads((comfy.SETTINGS).read_text())
    assert saved["comfy_url"] == url
    assert next(m for m in saved["machines"] if m["id"] == saved["active_machine"])["name"] == "This computer"  # 127.0.0.1
    # a machine auto-named before (here "GPU PC") takes the address's name when used again
    saved["machines"][0]["name"] = "GPU PC"
    comfy.SETTINGS.write_text(json.dumps(saved))
    c.post("/api/servers/use", json={"address": url})
    saved = json.loads(comfy.SETTINGS.read_text())
    assert saved["machines"][0]["name"] == "This computer"
    assert servers.label("http://gpu-pc:8188") == "gpu-pc" and servers.label("https://abc123xyz-8188.proxy.runpod.net") == "RunPod abc123xyz"


def test_capabilities_show_measured_numbers_for_this_gpu(client):
    c, url = client
    c.post("/api/servers/use", json={"address": url})
    cap = c.get("/api/capabilities").json()
    assert cap["connected"] and cap["machine"]["gpu"] == "RTX 4090" and cap["measured_gpu"] == "RTX 4090"
    by = {f["task"]: f for f in cap["features"]}
    assert by["recast"]["measured"]["max_frames"] == 145 and by["finish"]["measured"] is None
    assert by["recast"]["ready"] and not by["finish"]["ready"]


def test_the_comfy_pass_through_carries_view_and_uploads(client):
    c, url = client
    c.post("/api/servers/use", json={"address": url})
    v = c.get("/comfy/view", params={"filename": "frame.png"})
    assert v.status_code == 200 and v.content == b"\x89PNG-bytes" and v.headers["content-type"] == "image/png"
    up = c.post("/comfy/upload/image", files={"image": ("a.png", b"12345", "image/png")})
    assert up.status_code == 200 and up.json()["name"] == "up.png"
    FakeComfy.seen.clear()
    r = c.get("/comfy/system_stats", headers={"Origin": "http://localhost:7788", "User-Agent": "Mozilla/5.0"})
    assert r.json()["devices"][0]["type"] == "cuda"
    sent = [k.lower() for k, _ in FakeComfy.seen[-1]]
    # one agent (ours), and no Origin: ComfyUI's cross-origin guard would refuse this page's address
    assert sent.count("user-agent") == 1 and "origin" not in sent


def test_the_comfy_websocket_is_passed_through(client, monkeypatch):
    from websockets.sync.server import serve as ws_serve

    def echo(sock):
        for msg in sock:
            sock.send(f"echo:{msg}")
    server = ws_serve(echo, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.socket.getsockname()[1]
    c, _ = client
    c.app.state.conn.comfy_url = f"http://127.0.0.1:{port}"  # a websocket-only server: no /system_stats to test
    with c.websocket_connect("/comfy/ws?clientId=abc") as ws:
        ws.send_text("hello")
        assert ws.receive_text() == "echo:hello"
    server.shutdown()
