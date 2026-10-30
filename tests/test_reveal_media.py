"""Reveal in Finder for any file the app shows (Run 14, walkthrough 13): only library files, never a path outside it."""
import pytest
from fastapi.testclient import TestClient

from pickups.app.store import Store


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    shown = []
    monkeypatch.setattr(server, "_reveal", lambda target, folder=False: shown.append(target))
    app = server.create_app("http://127.0.0.1:9")
    return TestClient(app), lib, shown


def test_reveals_a_library_file_by_its_media_url(client):
    c, lib, shown = client
    f = lib / "projects" / "p1" / "images" / "a b.png"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"png")
    r = c.post("/api/reveal", json={"url": "/media/projects/p1/images/a%20b.png?v=3"})
    assert r.status_code == 200
    assert shown == [f.resolve()]


@pytest.mark.parametrize("url", ["/media/../secret.txt", "/media/%2e%2e/secret.txt", "/etc/passwd", "", "/media/missing.png", "/media/"])
def test_refuses_anything_else(client, url):
    c, lib, shown = client
    (lib.parent / "secret.txt").write_text("no")
    r = c.post("/api/reveal", json={"url": url})
    assert r.status_code == 404
    assert shown == []
