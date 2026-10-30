"""Run 16: a Previz to final take lists only the still models that made it, and a start or look frame is kept in the
project under the model that made it (not always "FLUX 3 klein")."""

from pathlib import Path

from PIL import Image

from pickups import graphs
from pickups import start_frame as sf
from pickups.tasks import previz
from test_inline_lane import _previz_shot, lane  # noqa: F401 - the fixture
from test_previz_e3 import client  # noqa: F401 - the fixture

ZIMAGE, QWEN21, EDIT2511 = previz.ZBUF_STILL, graphs.edit_model(False), graphs.edit_model(True)


def test_the_frame_editor_is_listed_only_when_it_ran():
    # The demo path: the start frame came from the inline lane (Z-Image Turbo), no look frame was made here.
    assert previz.frame_models(False, False, sf.ZBUF, False) == [ZIMAGE]
    assert QWEN21 not in previz.frame_models(False, False, sf.ZBUF, True)
    # A look frame made in the job: the editor, in the licence the setting picked.
    assert previz.frame_models(True, False, None, False) == [QWEN21]
    assert previz.frame_models(True, True, None, False) == [EDIT2511]
    # The inline lane's other methods name their own model; an own picture names none.
    assert previz.frame_models(False, False, sf.METHOD, False) == [QWEN21]
    assert previz.frame_models(False, True, "commercial-safe", False) == [EDIT2511]
    assert previz.frame_models(False, False, None, False) == []
    assert previz.frame_models(True, False, sf.METHOD, True) == [QWEN21, ZIMAGE]  # no repeats


def _kept(store, shot_id: str) -> list[dict]:
    pid = store.shot(shot_id)["project_id"]
    return store.images(pid)


def test_a_start_frame_is_kept_under_the_model_that_made_it(lane, monkeypatch):  # noqa: F811 - the fixture
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    if store.shot(shot_id).get("project_id") in (None, "examples"):
        prj = store.create_project("Pier")
        monkeypatch.setattr(store, "shot", lambda sid, _s=store.shot: {**_s(sid), "project_id": prj["id"]})

    def fake_make(runner, plan, frame0, out, seed=42):
        Image.new("RGB", (16, 9), (seed % 255, 10, 10)).save(out)
        return out

    monkeypatch.setattr(sf, "make", fake_make)
    r = c.post("/api/previz/start_frame", json={"shot_id": shot_id, "seed": 7})
    assert r.status_code == 200, r.text
    method = r.json()["method"]
    kept = [im for im in _kept(store, shot_id) if im.get("kind") == "Start frame"]
    assert kept and kept[0]["model"] == previz.START_MODELS[method] != "FLUX 3 klein"


def test_a_look_frame_is_kept_under_the_frame_editor(lane, monkeypatch):  # noqa: F811 - the fixture
    c, store, server, jm, freed = lane
    prj = store.create_project("Diner")
    monkeypatch.setattr(store, "version_master", lambda shot_id, version_id=None: Path("unused.mov"))
    monkeypatch.setattr(store, "shot_meta", lambda shot_id: {"project_id": prj["id"], "name": "Booth"})
    monkeypatch.setattr(store, "shot", lambda shot_id: {"project_id": prj["id"], "name": "Booth"})
    monkeypatch.setattr(store, "shot_dir", lambda shot_id: store.root / "shots_fake" / shot_id)
    monkeypatch.setattr(store, "shot_cast", lambda shot_id: ([], lambda c: None))
    (store.root / "shots_fake" / "sh1").mkdir(parents=True)

    def fake_look(runner, st, playblast, refs, look, out, seed=42, at=0, commercial_safe=False):
        Image.new("RGB", (8, 8), (1, 2, 3)).save(out)
        return out

    monkeypatch.setattr(server.previz, "look_frame", fake_look)
    r = c.post("/api/previz/look", json={"shot_id": "sh1", "look_prompt": "A diner.", "seed": 5})
    assert r.status_code == 200, r.text
    kept = [im for im in store.images(prj["id"]) if im.get("kind") == "Look frame"]
    assert kept and kept[0]["model"] == QWEN21
