"""Use on your phone: Tailscale detection is read-only and only ever yields a 100.64.0.0/10 address; the listener refuses
anything else (never 0.0.0.0); the setting is off by default and can't be turned on without Tailscale."""

import json
from types import SimpleNamespace

import pytest

from pickups.app import phone


def test_detect_reads_the_cli(monkeypatch):
    out = json.dumps({"BackendState": "Running", "Self": {"DNSName": "mac.tail1.ts.net.", "TailscaleIPs": ["100.64.1.2", "fd7a::1"]}})
    monkeypatch.setattr(phone.shutil, "which", lambda c: "/usr/bin/tailscale" if c == "tailscale" else None)
    monkeypatch.setattr(phone.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert phone.detect() == {"ip": "100.64.1.2", "name": "mac.tail1.ts.net"}


def test_detect_falls_back_to_an_interface_and_ignores_other_100_addresses(monkeypatch):
    monkeypatch.setattr(phone.shutil, "which", lambda c: "/sbin/ifconfig" if c == "ifconfig" else None)
    monkeypatch.setattr(phone.Path, "exists", lambda self: False)
    text = "inet 100.12.0.4 netmask\ninet 192.168.1.20\ninet 100.64.5.6 --> 100.64.5.6"
    monkeypatch.setattr(phone.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=text))
    assert phone.detect() == {"ip": "100.64.5.6", "name": None}  # 100.12.x is outside Tailscale's range


def test_the_off_switch(monkeypatch):
    monkeypatch.setenv("PICKUPS_TAILSCALE", "0")
    assert phone.detect() is None


def test_the_listener_only_binds_a_tailscale_address():
    lst = phone.PhoneListener()
    for host in ("0.0.0.0", "192.168.1.20", "127.0.0.1"):
        with pytest.raises(ValueError):
            lst.start(None, host, 7788)
    assert not lst.running


def test_routes_off_by_default_and_need_tailscale(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    from pickups.app.store import Store

    monkeypatch.setattr(server, "load_settings", lambda: {})
    monkeypatch.setattr(server, "Store", lambda: Store(tmp_path / "library"))
    monkeypatch.setattr(phone, "detect", lambda: None)
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    assert c.get("/api/phone").json() == {"tailscale": False, "enabled": False, "url": None, "ip": None, "name": None}
    assert c.post("/api/phone", json={"enabled": True}).status_code == 409
    assert c.get("/api/phone/qr.svg").status_code == 404
