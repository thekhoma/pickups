"""Fresh install (Run 14, walkthrough 29): sample thumbnails installed after the app started are served, not index.html."""
from fastapi.testclient import TestClient

from pickups.app.store import Store


def test_samples_installed_after_start_are_served(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    samples = tmp_path / "samples"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", samples)
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    assert not samples.exists()  # a fresh home: no samples folder when the app starts
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    (samples / "beach.jpg").write_bytes(b"\xff\xd8\xff jpeg")  # the pack lands later (first-run install, Add examples)
    r = c.get("/samples/beach.jpg")
    assert r.status_code == 200 and r.content.startswith(b"\xff\xd8")
    assert r.headers["content-type"] == "image/jpeg"
