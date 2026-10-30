"""Save videos to (Run 14, walkthrough 14): finished takes are copied into the chosen folder, one folder per project."""
import pytest
from fastapi.testclient import TestClient

from pickups.app import savecopy
from pickups.app.store import Store


class FakeStore:
    def __init__(self, root, project="Beach day"):
        self.root = root
        self.project_name = project

    def shot(self, shot_id):
        return {"id": shot_id, "name": "man at the beach", "project_id": "p1",
                "versions": [{"id": "orig"}, {"id": "v1", "label": "Previz"}, {"id": "v2", "label": "Finish 4K HDR"}]}

    def version_dir(self, shot_id, vid):
        return self.root / shot_id / vid

    def project(self, pid):
        return {"name": self.project_name}


def test_copy_lands_in_the_project_folder_with_a_plain_name(tmp_path):
    store = FakeStore(tmp_path / "lib")
    master = store.version_dir("s1", "v2") / "master.mov"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"prores")
    out = savecopy.copy_take(store, "s1", "v2", str(tmp_path / "Movies"))
    assert out == tmp_path / "Movies" / "Beach_day" / "man_at_the_beach_T02_Finish_4K_HDR.mov"
    assert out.read_bytes() == b"prores"


def test_nothing_copied_without_a_folder_or_a_master(tmp_path):
    store = FakeStore(tmp_path / "lib")
    assert savecopy.copy_take(store, "s1", "v2", None) is None
    assert savecopy.copy_take(store, "s1", "v2", str(tmp_path / "Movies")) is None
    assert not (tmp_path / "Movies").exists()


def test_check_dir_rules(tmp_path):
    assert savecopy.check_dir(str(tmp_path / "new" / "Pickups")).is_dir()
    for bad in ("", "  ", "relative/folder"):
        with pytest.raises(ValueError):
            savecopy.check_dir(bad)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.comfy as comfy
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(comfy, "SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(comfy, "HOME", tmp_path)
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    return TestClient(server.create_app("http://127.0.0.1:9")), tmp_path


def test_the_routes_keep_the_folder(client):
    c, tmp = client
    r = c.get("/api/save-folder").json()
    assert r["path"] is None and r["suggested"].endswith("Pickups")
    assert c.post("/api/save-folder", json={"path": "not/absolute"}).status_code == 400
    target = tmp / "Videos" / "Pickups"
    assert c.post("/api/save-folder", json={"path": str(target)}).json()["path"] == str(target)
    assert c.get("/api/save-folder").json()["path"] == str(target)
    assert target.is_dir()


def test_folders_read_with_the_home_as_a_tilde(tmp_path, monkeypatch):
    monkeypatch.setattr(savecopy.Path, "home", classmethod(lambda cls: tmp_path))
    assert savecopy.shown(tmp_path / "Movies" / "Pickups") == "~/Movies/Pickups"
    assert savecopy.shown("/Volumes/Media/Pickups") == "/Volumes/Media/Pickups"
    assert savecopy.shown(None) is None
