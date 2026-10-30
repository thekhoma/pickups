"""The default-models catalog (the gallery's model chips) names models the way the tasks record them."""

from pathlib import Path

import pytest

from pickups import graphs, images, licences, models_catalog

SRC = Path(models_catalog.__file__).parent
# Where each task writes its details["models"] (or the names it builds them from).
RECORDS = {
    "recast": ["tasks/recast_ltx.py", "tasks/recast.py"],
    "relight": ["tasks/scene_relight.py"],
    "remove": ["tasks/remove.py"],
    "environment": ["tasks/environment.py"],
    "previz": ["tasks/previz.py"],
    "sheet": ["sheets.py"],
    "finish": ["tasks/finish.py"],
    "image": ["images.py"],
}


def _recorded(task: str) -> str:
    return "\n".join((SRC / f).read_text() for f in RECORDS[task])


@pytest.mark.parametrize("safe", [False, True])
def test_every_workflow_has_models(safe):
    cat = models_catalog.catalog(safe)
    assert set(cat) >= {"character", "weather", "remove", "previz", "sheet", "finish", "image"}  # "car" only with features.cars()
    for wid, w in cat.items():
        assert w["task"] in RECORDS, wid
        assert w["models"], f"{wid} has no models"
        for m in w["models"]:
            assert m["role"] and m["name"] and m["short"], (wid, m)
            assert m["options"][0] == m["name"], (wid, m)


@pytest.mark.parametrize("safe", [False, True])
def test_names_match_what_tasks_record(safe):
    editors = {graphs.edit_model(False), graphs.edit_model(True)}
    image_names = {m.name for m in images.MODELS.values()}
    for wid, w in models_catalog.workflows(safe).items():
        src = _recorded(w["task"])
        for s in w["models"]:
            for name in (s.name, *s.options):
                assert name in licences.LICENCES, f"{wid}: {name} has no licence entry"
                # a literal in the task's code, or a name it gets from graphs.edit_model() / images.MODELS
                assert f'"{name}"' in src or name in editors or name in image_names, f"{wid}: {name} isn't what {w['task']} records"


def test_frame_editor_follows_the_licence_choice():
    assert models_catalog.for_task("recast")[1].name == "Qwen-Image 2.1"
    assert models_catalog.for_task("recast", commercial_safe=True)[1].name == "Qwen-Image-Edit 2511"
    assert models_catalog.for_task("image")[0].name == images.MODELS[images.DEFAULT].name
    assert models_catalog.for_task("reshoot") == []


def test_served_and_filled_into_examples(tmp_path, monkeypatch):
    import json

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
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "examples.json").write_text(json.dumps([{"id": "a", "task": "remove", "after": "a_after.mp4"}]))
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    cat = c.get("/api/models/defaults").json()
    assert cat["weather"]["models"][0]["short"] == "LTX-2.5"
    ex = c.get("/api/examples").json()[0]
    assert [m["name"] for m in ex["models"]] == [m["name"] for m in cat["remove"]["models"]]


def test_hidden_examples_are_listed_by_id(tmp_path, monkeypatch):
    import json

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
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "examples.json").write_text(json.dumps([
        {"id": "env-car-snow", "task": "environment", "params": {"change": "snow", "keep_text": "the car"}},  # kept
        {"id": "rc-car-offroader", "task": "recast", "params": {"text": "SUV"}},  # a car swap: hidden
        {"id": "walker", "task": "recast", "params": {"text": "woman"}},
        {"id": "launch", "task": "previz", "params": {"text": "car"}},
        {"id": "pv-k03-hood-push", "task": "previz", "params": {}},  # the driver's hand to his face: hidden by id
        {"id": "env-fleshmarket-fog", "task": "environment", "params": {"change": "custom"}},  # hidden by id
        {"id": "rl-harbour-sunset", "task": "relight", "params": {}},  # denied in review: hidden by id
        {"id": "rl-walk-golden", "task": "relight", "params": {}},  # approved: kept
        {"id": "held", "task": "environment", "params": {}, "hold_reason": "not offered yet"},
    ]))
    monkeypatch.delenv("PICKUPS_CARS", raising=False)
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    assert [e["id"] for e in c.get("/api/examples").json()] == ["env-car-snow", "walker", "launch", "rl-walk-golden"]
    monkeypatch.setenv("PICKUPS_CARS", "1")
    assert [e["id"] for e in c.get("/api/examples").json()] == ["env-car-snow", "rc-car-offroader", "walker", "launch", "rl-walk-golden"]


def test_the_seafront_pier_previz_leads_previz_to_final_with_its_own_models(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path

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
    used = [{"role": "video", "name": "LTX-2.5 Layout-To-Render", "short": "LTX-2.5", "options": ["LTX-2.5 Layout-To-Render"]}]
    pack = [{"id": "rm-trail", "task": "remove"}, {"id": "pv-k05-driver", "task": "previz"}, {"id": "pv-k03-hood-push", "task": "previz"},
            {"id": "pv-ocean-person", "task": "previz", "models_used": used, "after": "pv-ocean-person_after.mp4"}]
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "examples.json").write_text(json.dumps(pack))
    c = TestClient(server.create_app("http://127.0.0.1:9"))
    got = c.get("/api/examples").json()
    assert [e["id"] for e in got if e["task"] == "previz"] == ["pv-ocean-person", "pv-k05-driver"]
    assert got[0]["models_used"] == used and got[0]["after_url"] == "/examples/pv-ocean-person_after.mp4"
    # make_examples_pack writes the pack in the same order, without the hood push
    spec = importlib.util.spec_from_file_location("make_examples_pack", Path(__file__).parent.parent / "scripts" / "make_examples_pack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [e["id"] for e in mod.shown(pack)] == ["pv-ocean-person", "rm-trail", "pv-k05-driver"]
