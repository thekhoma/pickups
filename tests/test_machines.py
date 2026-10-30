"""Saved machines and RunPod: settings migration, the RunPod address, and switching (proxy mocked, no GPU)."""

import json

import pytest
from fastapi.testclient import TestClient

from pickups.app import comfy, machines
from pickups.app.comfy import Connection


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "HOME", tmp_path)
    monkeypatch.setattr(comfy, "SETTINGS", tmp_path / "settings.json")
    return tmp_path


# --- migration -----------------------------------------------------------------------------------------------------
def test_old_settings_become_one_machine():
    ms, active = machines.normalize({"comfy_url": "http://100.64.0.5:8188/", "commercial_safe": True})
    assert len(ms) == 1 and ms[0]["url"] == "http://100.64.0.5:8188" and ms[0]["kind"] == "lan"
    assert active == ms[0]["id"]


def test_empty_settings_default_to_this_computer():
    ms, active = machines.normalize({})
    assert ms[0] | {"id": None} == {"id": None, "name": "This computer", "kind": "local", "url": machines.DEFAULT_URL}


def test_active_follows_comfy_url_when_id_missing():
    ms = [{"id": "a", "name": "A", "kind": "local", "url": "http://127.0.0.1:8188"},
          {"id": "b", "name": "B", "kind": "runpod", "url": "https://abc123xyz-8188.proxy.runpod.net"}]
    assert machines.normalize({"machines": ms, "comfy_url": ms[1]["url"]})[1] == "b"
    st = machines.with_machines({"x": 1}, ms, "a")
    assert st["comfy_url"] == ms[0]["url"] and st["x"] == 1 and st["active_machine"] == "a"


# --- the RunPod address --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,port,want", [
    ("abc123xyz", None, "https://abc123xyz-8188.proxy.runpod.net"),
    ("  ABC123XYZ ", None, "https://abc123xyz-8188.proxy.runpod.net"),
    ("abc123xyz", "3000", "https://abc123xyz-3000.proxy.runpod.net"),
    ("https://abc123xyz-8188.proxy.runpod.net/", None, "https://abc123xyz-8188.proxy.runpod.net"),
    ("abc123xyz-8288.proxy.runpod.net/system_stats", None, "https://abc123xyz-8288.proxy.runpod.net"),
    ("https://abc123xyz-8188.proxy.runpod.net", 9000, "https://abc123xyz-9000.proxy.runpod.net"),
])
def test_runpod_url(text, port, want):
    assert machines.runpod_url(text, port) == want


@pytest.mark.parametrize("text,port", [
    ("", None), ("abc", None), ("abc_123xyz", None), ("abc123xyz;rm", None), ("abc 123 xyz", None),
    ("https://evil.example.com/abc123xyz-8188.proxy.runpod.net", None), ("https://abc123xyz.proxy.runpod.net", None),
    ("abc123xyz", "0"), ("abc123xyz", "70000"), ("abc123xyz", "http"),
])
def test_runpod_url_rejects(text, port):
    with pytest.raises(ValueError):
        machines.runpod_url(text, port)


def test_runpod_is_its_own_kind():
    assert machines.kind_of("https://abc123xyz-8188.proxy.runpod.net") == "runpod"
    assert machines.kind_of("http://localhost:8188") == "local"


# --- switching -----------------------------------------------------------------------------------------------------
class FakeProc:
    def __init__(self, args, **_):
        self.args, self.done = args, False
        FakeProc.started.append(self)

    def poll(self):
        return 0 if self.done else None

    def terminate(self):
        self.done = True
        FakeProc.listening.discard(int(self.args[self.args.index("--port") + 1]))

    def wait(self, timeout=None):
        return 0


@pytest.fixture()
def fake_proxy(home, monkeypatch):
    FakeProc.started, FakeProc.listening = [], set()

    def popen(args, **kw):
        p = FakeProc(args, **kw)
        FakeProc.listening.add(int(args[args.index("--port") + 1]))
        return p

    monkeypatch.setattr(comfy.shutil, "which", lambda _: "/bin/comfy-api-proxy")
    monkeypatch.setattr(comfy.subprocess, "Popen", popen)
    monkeypatch.setattr(comfy, "_port_open", lambda port: port in FakeProc.listening)
    monkeypatch.setattr(comfy, "_free_port", lambda: 18999)
    monkeypatch.setattr(comfy.time, "sleep", lambda _: None)
    return FakeProc


def test_switch_stops_our_proxy_and_starts_one_for_the_new_url(fake_proxy):
    conn = Connection("http://127.0.0.1:8188", proxy_port=8189)
    conn.ensure_proxy()
    first = fake_proxy.started[0]
    assert "http://127.0.0.1:8188" in first.args
    base = machines.switch(conn, [{"status": "done"}, {"status": "queued"}], "https://abc123xyz-8188.proxy.runpod.net/")
    conn.ensure_proxy()  # the new proxy starts in the background: wait for it as a job would
    assert first.done
    second = fake_proxy.started[1]
    assert "https://abc123xyz-8188.proxy.runpod.net" in second.args and "8189" in second.args
    assert base == conn.comfy_url == "https://abc123xyz-8188.proxy.runpod.net" and conn.generation == 1


def test_switch_takes_a_free_port_when_another_proxy_holds_it(fake_proxy):
    fake_proxy.listening.add(8189)  # a proxy this app didn't start
    conn = Connection("http://127.0.0.1:8188", proxy_port=8189)
    conn.switch("http://127.0.0.1:8288")
    conn.ensure_proxy()  # started in the background
    assert conn.proxy_port == 18999 and 8189 in fake_proxy.listening
    assert "18999" in fake_proxy.started[0].args and "http://127.0.0.1:8288" in fake_proxy.started[0].args


def test_no_switch_while_a_job_runs(fake_proxy):
    conn = Connection("http://127.0.0.1:8188", proxy_port=8189)
    with pytest.raises(machines.Busy):
        machines.switch(conn, [{"status": "running"}], "http://127.0.0.1:8288")
    assert conn.comfy_url == "http://127.0.0.1:8188" and not fake_proxy.started


# --- the routes ----------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(fake_proxy, tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    from pickups.app.store import Store
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    monkeypatch.setattr(machines, "probe", lambda url, timeout=6.0: {"reachable": False, "gpu": None, "vram_gb": None})
    (tmp_path / "settings.json").write_text(json.dumps({"comfy_url": "http://127.0.0.1:9"}))
    app = server.create_app("http://127.0.0.1:9")
    return TestClient(app), app


def test_add_runpod_and_switch(client, home):
    c, _ = client
    r = c.post("/api/machines", json={"kind": "runpod", "pod": "abc123xyz"})
    assert r.status_code == 200
    body = r.json()
    pod = next(m for m in body["machines"] if m["id"] == body["added"])
    assert pod["url"] == "https://abc123xyz-8188.proxy.runpod.net" and pod["kind"] == "runpod"
    assert c.post("/api/machines", json={"kind": "runpod", "pod": "bad_id!"}).status_code == 400
    r = c.post(f"/api/machines/{pod['id']}/use")
    assert r.status_code == 200 and r.json()["active"] == pod["id"]
    saved = json.loads((home / "settings.json").read_text())
    assert saved["comfy_url"] == pod["url"] and saved["active_machine"] == pod["id"]
    assert not any(k in json.dumps(saved).lower() for k in ("api_key", "token", "secret"))
    assert c.delete(f"/api/machines/{pod['id']}").status_code == 409  # the active one stays


def test_the_unsaved_machine_keeps_one_id(client, home):
    """Before any machine is saved, /api/settings and /api/machines name the same machine on every call, and after a
    restart: its id is made once and kept in the app home. Adding a machine saves it under that id."""
    import pickups.app.server as server
    c, _ = client
    ids = {c.get("/api/settings").json()["active_machine"]["id"] for _ in range(3)}
    ids |= {c.get("/api/machines").json()["active"] for _ in range(3)}
    ids |= {m["id"] for m in c.get("/api/machines").json()["machines"]}
    assert len(ids) == 1
    (mid,) = ids
    assert (home / "machine_id").read_text().strip() == mid
    assert "machines" not in json.loads((home / "settings.json").read_text())  # a GET saves nothing in settings
    again = TestClient(server.create_app("http://127.0.0.1:9"))
    assert again.get("/api/settings").json()["active_machine"]["id"] == mid
    added = c.post("/api/machines", json={"kind": "lan", "url": "http://127.0.0.1:8288"}).json()
    assert added["active"] == mid and mid in [m["id"] for m in json.loads((home / "settings.json").read_text())["machines"]]


def test_use_refused_mid_job(client, monkeypatch):
    c, app = client
    pod = c.post("/api/machines", json={"kind": "lan", "url": "http://127.0.0.1:8288"}).json()["added"]
    import pickups.app.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod.JobManager, "list", lambda self: [{"status": "running"}])
    r = c.post(f"/api/machines/{pod}/use")
    assert r.status_code == 409 and "job is running" in r.json()["detail"]
