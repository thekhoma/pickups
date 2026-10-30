"""The examples pack: install from a zip or a folder, the old flat format, nothing written outside its two folders,
the import route, and finding a pack next to the app."""

import json
import zipfile

import pytest

from pickups.app import examples_pack

EXAMPLES = [{"id": "rl-walk-golden", "task": "relight", "sample_id": "rl-walk-golden", "after": "rl-walk-golden_after.mp4"}]


def make_zip(path, extra=()):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("pack.json", json.dumps({"version": 1}))
        z.writestr("examples/examples.json", json.dumps(EXAMPLES))
        z.writestr("examples/rl-walk-golden_after.mp4", b"AFTER-BYTES")
        z.writestr("samples/samples.json", json.dumps({"rl-walk-golden": {"name": "Evening walk"}}))
        z.writestr("samples/rl-walk-golden.mp4", b"CLIP")
        for name, data in extra:
            z.writestr(name, data)
    return path


def test_a_zip_installs_byte_for_byte_and_merges_sample_names(tmp_path):
    ex, smp = tmp_path / "examples", tmp_path / "samples"
    smp.mkdir()
    (smp / "samples.json").write_text(json.dumps({"old": {"name": "Old"}}))
    res = examples_pack.install(make_zip(tmp_path / "p.zip", [("../evil.json", b"x"), ("examples/../../evil2.json", b"x")]), ex, smp)
    assert res == {"examples": 1, "samples": 1}
    assert (ex / "rl-walk-golden_after.mp4").read_bytes() == b"AFTER-BYTES"
    assert json.loads((smp / "samples.json").read_text()) == {"old": {"name": "Old"}, "rl-walk-golden": {"name": "Evening walk"}}
    assert not (tmp_path / "evil.json").exists() and not (tmp_path.parent / "evil2.json").exists()


def test_a_folder_and_an_old_flat_pack_install_too(tmp_path):
    folder = tmp_path / "examples_pack"
    with zipfile.ZipFile(make_zip(tmp_path / "p.zip")) as z:
        z.extractall(folder)
    res = examples_pack.install(folder, tmp_path / "e1", tmp_path / "s1")
    assert res["examples"] == 1 and (tmp_path / "s1" / "rl-walk-golden.mp4").read_bytes() == b"CLIP"
    flat = tmp_path / "flat.zip"
    with zipfile.ZipFile(flat, "w") as z:
        z.writestr("examples.json", json.dumps(EXAMPLES))
        z.writestr("rl-walk-golden_after.mp4", b"x")
    assert examples_pack.install(flat, tmp_path / "e2", tmp_path / "s2")["examples"] == 1


def test_bad_packs_say_why(tmp_path):
    (tmp_path / "not.zip").write_text("hello")
    with pytest.raises(ValueError):
        examples_pack.install(tmp_path / "not.zip", tmp_path / "e", tmp_path / "s")
    with pytest.raises(FileNotFoundError):
        examples_pack.install(tmp_path / "missing.zip", tmp_path / "e", tmp_path / "s")
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("samples/a.mp4", b"x")
    with pytest.raises(ValueError):
        examples_pack.install(empty, tmp_path / "e3", tmp_path / "s3")


def test_a_pack_next_to_the_app_is_found(tmp_path):
    assert examples_pack.find_next_to([tmp_path]) is None
    make_zip(tmp_path / "pickups_examples_v1.zip")
    make_zip(tmp_path / "pickups_examples_v2.zip")
    assert examples_pack.find_next_to([tmp_path / "nowhere", tmp_path]).name == "pickups_examples_v2.zip"


def test_the_import_route_takes_a_path_or_an_upload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    import pickups.app.store as st
    from pickups.app.store import Store

    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(server, "load_settings", lambda: {})
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    monkeypatch.setattr(examples_pack, "find_next_to", lambda roots: None)
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    assert c.get("/api/examples/pack").json()["installed"] is False
    pack = make_zip(tmp_path / "p.zip")
    assert c.post("/api/examples/import", json={"path": str(tmp_path / "nope.zip")}).status_code == 400
    r = c.post("/api/examples/import", json={"path": str(pack)}).json()
    assert r["examples"] == 1 and c.get("/api/examples/pack").json()["examples"] == 1
    (tmp_path / "examples" / "examples.json").unlink()
    r = c.post("/api/examples/import", files={"file": ("pickups_examples_v1.zip", pack.read_bytes(), "application/zip")}).json()
    assert r["examples"] == 1
