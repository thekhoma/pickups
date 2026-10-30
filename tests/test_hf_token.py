"""Pickups never handles a Hugging Face token: the app's download leaves gated files to the browser, and a token in the
environment is never returned by an API, written to the library or settings, or put into an error the UI shows."""

import io
import json
import urllib.error

import pytest

from pickups.app import manifest, models

SECRET = "hf_SENTINELsentinelSENTINEL0000"


def test_the_downloader_never_leaks_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    seen = []

    def refuse(req, timeout=0):
        seen.append(req)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"no"))

    monkeypatch.setattr(models.urllib.request, "urlopen", refuse)
    gated = next(f for f, row in manifest.MODELS.items() if row[4])
    with pytest.raises(urllib.error.HTTPError) as e:
        models.download([gated], tmp_path)
    assert SECRET not in str(e.value) and SECRET not in repr(e.value)
    assert seen[0].full_url.startswith("https://huggingface.co/")  # the only place it may go
    assert not any(SECRET in p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file())


def test_no_api_says_or_stores_anything_about_the_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    from pickups.app.store import Store

    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("PICKUPS_HOME", str(tmp_path))
    monkeypatch.setattr(server, "Store", lambda: Store(tmp_path / "library"))
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    for path in ("/api/status", "/api/capabilities", "/api/settings", "/api/servers", "/api/models/download", "/api/storage"):
        r = c.get(path)
        assert SECRET not in r.text, path
        assert "hf_token" not in r.text, path
    assert not any(SECRET in p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file() and p.stat().st_size < 5e6)


def test_the_app_download_leaves_gated_files_to_the_browser(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pickups.app.server as server
    from pickups.app.store import Store

    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setattr(server, "Store", lambda: Store(tmp_path / "library"))
    monkeypatch.setattr(server, "load_settings", lambda: {"comfy_models_dir": str(tmp_path)})
    started = []
    monkeypatch.setattr(models, "download", lambda files, d, cb=None: started.append(list(files)))
    app = server.create_app("http://127.0.0.1:9")
    gated = next(f for f, row in manifest.MODELS.items() if row[4])
    tiles = {"finish": {"missing": [manifest.model_info(gated)]}}
    monkeypatch.setattr(server.doctor, "status", lambda conn: {"tiles": tiles, "machine": None})
    r = TestClient(app).post("/api/models/download", json={"tasks": ["finish"]})
    assert r.status_code == 409 and SECRET not in r.text and started == []
    assert json.dumps(r.json()).count("HF_TOKEN") == 0
