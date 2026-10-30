"""Recast keeps the still it made from a description with the take, so "Try again with changes" can show the same
person and use it again (params.reference_take)."""

import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

PNG = b"\x89PNG\r\n\x1a\n made by FLUX 3 klein"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    return TestClient(server.create_app("http://127.0.0.1:9")), store  # a dead local address: no ComfyUI


def _shot(store, tmp_path):
    src = tmp_path / "hall.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24", "-t", "0.5",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    return store.add_shot(src, "Castle hall")["id"]


def _recast_take(store, sid, still: bool):
    vid, vdir = store.new_version_dir(sid)
    if still:
        (vdir / "reference.png").write_bytes(PNG)
    store.save_version(sid, vid, {"label": "Recast", "task": "recast", "input_version": "orig", "downloads": [],
                                  "params": {"frame": 4, "points": [], "text": "woman", "reference_prompt": "Studio photograph of a woman"}})
    return vid, vdir


def test_a_take_with_a_still_serves_it(client, tmp_path):
    c, store = client
    sid = _shot(store, tmp_path)
    vid, vdir = _recast_take(store, sid, still=True)
    take = next(v for v in c.get(f"/api/shots/{sid}").json()["versions"] if v["id"] == vid)
    assert take["reference_url"] == f"/media/shots/{sid}/versions/{vid}/reference.png"
    r = c.get(take["reference_url"])
    assert r.status_code == 200 and r.content == PNG
    assert store.take_reference(sid, vid) == vdir / "reference.png"


def test_older_takes_have_none_and_ids_stay_ids(client, tmp_path):
    c, store = client
    sid = _shot(store, tmp_path)
    vid, _ = _recast_take(store, sid, still=False)
    assert "reference_url" not in next(v for v in c.get(f"/api/shots/{sid}").json()["versions"] if v["id"] == vid)
    (tmp_path / "library" / "shots" / sid / "reference.png").write_bytes(PNG)  # never reachable through a take id
    for bad in (vid, "orig", "..", "../..", f"{vid}/..", "", "v_missing", None):
        assert store.take_reference(sid, bad) is None
    assert store.take_reference("..", vid) is None


# --- the task: keep the still, use it again ---------------------------------------------------------------------------


class _Stop(Exception):
    pass


class _Out:
    def __init__(self, data):
        self.data = data

    def to_file(self, path):
        path.write_bytes(self.data)


class _Job:
    def get_outputs(self, _name):
        return [_Out(b"\x89PNG new still")]


class _Ctx:
    """Enough of a job's context to run Recast up to the performance (no ComfyUI)."""

    def __init__(self, store, sid, params):
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.ran = []

    def stage(self, label):
        if label == "Performing the scene":
            raise _Stop

    def run(self, graph):
        self.ran.append(graph)
        return _Job()


def _run_until_the_performance(ctx, monkeypatch):
    from pickups.tasks import common, recast
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **k: [np.ones((96, 160), bool)] * 4)
    with pytest.raises(_Stop):
        recast.run(ctx)


def test_a_still_made_from_words_is_kept_with_the_take(client, tmp_path, monkeypatch):
    _, store = client
    sid = _shot(store, tmp_path)
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "woman", "describe": "a woman in a green wool gown"})
    _run_until_the_performance(ctx, monkeypatch)
    assert len(ctx.ran) == 1 and "woman in a green wool gown" in ctx.ran[0]["PU_Prompt"]["inputs"]["text"]
    assert (ctx.vdir / "reference.png").read_bytes() == b"\x89PNG new still"  # beside the take, not in work/


def test_reference_take_uses_that_takes_still_again(client, tmp_path, monkeypatch):
    _, store = client
    sid = _shot(store, tmp_path)
    vid, _ = _recast_take(store, sid, still=True)
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "woman", "reference_take": vid, "reference_prompt": "Studio photograph"})
    _run_until_the_performance(ctx, monkeypatch)
    assert ctx.ran == []  # no new still
    assert (ctx.vdir / "reference.png").read_bytes() == PNG


def test_reference_take_without_a_still_makes_it_again_from_the_words(client, tmp_path, monkeypatch):
    _, store = client
    sid = _shot(store, tmp_path)
    vid, _ = _recast_take(store, sid, still=False)
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "woman", "reference_take": vid, "reference_prompt": "Studio photograph",
                            "seed": 7})
    _run_until_the_performance(ctx, monkeypatch)
    assert len(ctx.ran) == 1 and ctx.ran[0]["PU_Prompt"]["inputs"]["text"] == "Studio photograph"
    assert ctx.ran[0]["PU_Seed"]["inputs"]["seed"] == 7
