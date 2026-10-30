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
    # A dead local address: the test never talks to a real ComfyUI.
    app = server.create_app("http://127.0.0.1:9")
    return TestClient(app), store


def test_upload_served_by_id(client):
    c, store = client
    up = store.add_upload(b"\x89PNG fake", "ref.png")
    r = c.get(f"/api/uploads/{up['id']}")
    assert r.status_code == 200 and r.content == b"\x89PNG fake"
    assert r.headers["content-type"] == "image/png"


@pytest.mark.parametrize("bad", ["notanid", "up_XYZ", "up_*", "up_..", "cast_0123abcd"])
def test_bad_upload_ids_rejected(client, bad):
    c, store = client
    store.add_upload(b"x", "a.png")
    assert c.get(f"/api/uploads/{bad}").status_code == 400


def test_no_path_traversal(client, tmp_path):
    c, store = client
    (tmp_path / "library" / "secret.txt").write_text("no")
    for path in ("/api/uploads/..%2Fsecret.txt", "/api/uploads/up_..%2F..%2Fsecret.txt"):
        r = c.get(path)
        assert r.status_code in (400, 404) and b"no" != r.content
    with pytest.raises(FileNotFoundError):
        store.upload_path("../secret")


def test_missing_upload_404(client):
    c, _ = client
    assert c.get("/api/uploads/up_0123456789").status_code == 404


def test_ref_pictures_resolve_to_their_files(client):
    c, store = client
    prj = store.create_project("P")
    cast = store.add_cast(prj["id"], b"img", "maya.png", "Maya")
    up = store.add_upload(b"abc", "look.png")
    r = c.get(f"/api/projects/{prj['id']}/refs/cast/{cast['id']}", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == cast["thumb_url"]
    assert c.get(f"/api/projects/{prj['id']}/refs/upload/{up['id']}").content == b"abc"
    assert c.get(f"/api/projects/{prj['id']}/refs/image/img_0123456789").status_code == 404
    assert c.get(f"/api/projects/{prj['id']}/refs/video/x").status_code == 400
    assert c.get(f"/api/projects/nope/refs/cast/{cast['id']}").status_code == 404
