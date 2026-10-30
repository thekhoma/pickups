"""Run 10 A: the Look on screen (auto or edited) is the Look the models get.

R&D's Run 9 check: the auto Look ("It is dusk. The man wears a long coat. ...") was never sent, so the start still was
rebuilt from the previz alone (a t-shirt, no pier) and the take saved look_prompt "". These tests read the graphs a
mocked ComfyUI is sent, including the words the previz kit can't draw (coat, pier).
"""

import json
from fractions import Fraction
from pathlib import Path

from PIL import Image

from pickups import prompt_words
from pickups import start_frame as sf
from pickups.media import Probe
from pickups.tasks import previz
from test_inline_lane import _previz_shot, lane  # noqa: F401 - the fixture
from test_previz_e3 import client  # noqa: F401 - the fixture

# The Look the previz page writes for R&D's sentence (lookPrompt.ts), fixed look line and all.
AUTO_LOOK = ("It is dusk. The man wears a long coat. The grey figure is a man in a long coat who stands on a pier at dusk. "
             "A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean. "
             "Practical lights glow against deep blue shadows, soft highlight roll-off, fine grain.")
STANDING = {"camera": {"height_m": 1.6, "pitch_deg": -2, "focal_mm": 35},
            "subjects": [{"kind": "person", "word": "man", "x": 0.5, "distance_m": 6.0, "in_view": True, "facing": "away"}]}


def test_look_words_drop_the_look_line_and_say_the_figure_as_what_is_there():
    assert prompt_words.look_words(AUTO_LOOK) == ("It is dusk. The man wears a long coat. "
                                                  "A man in a long coat stands on a pier at dusk.")
    # Run 9's doubled verb, saved on older drafts and takes, reads right too.
    assert prompt_words.look_words("The grey figure is a man in a long coat stands on a pier at dusk.") == \
        "A man in a long coat stands on a pier at dusk."
    assert prompt_words.look_words("The grey figures are two women who walk along the beach.") == "Two women walk along the beach."
    assert prompt_words.look_words("He wears a grey flat cap.") == "He wears a grey flat cap."  # a real grey stays


def test_the_start_still_prompt_has_the_auto_look_wardrobe_and_place():
    p = sf.plan({"notes": "A man in a long coat stands on a pier at dusk."}, STANDING, [], [], [], AUTO_LOOK)
    for words in ("long coat", "pier", "dusk"):
        assert words in p.positive, (words, p.positive)
    assert "grey figure" not in p.positive.lower() and p.positive.count("65mm commercial film frame") == 1
    # The same for the zbuf method (no images, the depth pass only).
    z = sf.plan({"notes": "A man stands by a lake."}, STANDING, [], [], [], AUTO_LOOK, method=sf.ZBUF)
    assert "long coat" in z.positive and "pier" in z.positive


def test_the_start_frame_route_sends_the_look_to_comfyui(lane, monkeypatch):  # noqa: F811 - the fixture
    c, store, server, jm, freed = lane
    shot_id = _previz_shot(c)
    sent = {}

    def fake_make(runner, plan, frame0, out, seed=42):  # the mocked ComfyUI: the real graph, then a picture
        sent["graph"] = json.dumps(sf.graph_for(plan, "depth.png", "canny.png", seed))
        Image.new("RGB", (16, 9)).save(out)
        return out

    monkeypatch.setattr(sf, "make", fake_make)
    r = c.post("/api/previz/start_frame", json={"shot_id": shot_id, "seed": 7, "user_look": AUTO_LOOK, "method": "B3"})
    assert r.status_code == 200, r.text
    for words in ("long coat", "pier", "dusk"):
        assert words in sent["graph"], words
    # Its recipe keeps the Look shown, so the shot's later keyframes are made from the same words.
    recipes = json.loads((store.shot_dir(shot_id) / previz.RECIPES).read_text())
    assert recipes[r.json()["upload_id"]]["user_look"] == AUTO_LOOK


def test_the_video_line_takes_the_looks_facts_in_order():
    man = [{"kind": "person", "word": "man"}]
    out = prompt_words.l2r_prompt("A man stands on the shore of a lake. The camera pushes in.", man, look=AUTO_LOOK)
    # Run 11: the Look's one sentence with the coat, the dusk and the pier, said once (not "It is dusk. The man wears a
    # long coat." before it), in place of frame 1's "stands on the shore", where the start still doesn't show him.
    assert out == ("A man in a long coat stands on a pier at dusk. The camera pushes in. "
                   "Keep the same man and camera as the driver.")
    assert "grey" not in out and "65mm" not in out
    # Already said: nothing is doubled.
    said = "A man in a long coat stands on a pier at dusk. The camera pushes in."
    assert prompt_words.l2r_prompt(said, man, look=AUTO_LOOK) == f"{said} Keep the same man and camera as the driver."
    assert prompt_words.l2r_prompt(said, man) == prompt_words.l2r_prompt(said, man, look="")


def _ctx(tmp_path, params, seen):
    class Store:
        def shot_cast(self, shot_id):
            return [], lambda c: None

        def shot_meta(self, shot_id):
            return {"previz_framing": STANDING}

        def shot_dir(self, shot_id):
            return tmp_path

        def upload_path(self, uid):
            return tmp_path / f"{uid}.png"

    class Job:
        def get_outputs(self, name):
            return []

    class Ctx:
        input_path = Path("in.mp4")
        vdir = tmp_path
        conn = None
        project_id = "prj"
        shot_id = "sht"
        store = Store()
        client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

        def stage(self, label):
            pass

        def run(self, graph):
            seen.append(json.dumps(graph))
            return Job()

        def commercial_safe(self):
            return False

    ctx = Ctx()
    ctx.params = params
    return ctx


def test_a_take_from_an_accepted_start_frame_sends_and_saves_the_look(tmp_path, monkeypatch):
    seen: list[str] = []
    Image.new("RGB", (16, 9)).save(tmp_path / "mine.png")

    def prepare(src, out, start, frames, size, pre="", cover=False, pad=None, hold=0):
        out.write_bytes(b"work")
        return {"fps": 24.0, "frames": frames, "start_frame": start, "size": list(size)}

    monkeypatch.setattr(previz.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), 97, False, None))
    monkeypatch.setattr(previz.media, "prepare_work_clip", prepare)
    monkeypatch.setattr(previz.bench, "machine", lambda conn: {"vram_gb": 24})
    monkeypatch.setattr(previz.common, "grain_pngs", lambda *a, **k: None)
    monkeypatch.setattr(previz.common, "finish_outputs", lambda *a, **k: [])
    monkeypatch.setattr(previz.common, "sequence_details", lambda *a, **k: {})
    import pickups.app.store as app_store
    monkeypatch.setattr(app_store, "make_viewer", lambda *a, **k: None)
    # What the form sends now: the start frame it accepted, the Look shown and the Advanced motion line.
    params = {"look_frame_upload_id": "mine", "look_prompt": AUTO_LOOK, "pin_start": False,
              "prompt": "A man stands on the shore of a lake. The camera pushes in, then orbits around him."}
    out = previz.run(_ctx(tmp_path, params, seen))
    (graph,) = seen  # one LTX graph, no look frame made
    for words in ("long coat", "pier", "dusk"):
        assert words in graph, words
    assert out["details"]["look_prompt"] == AUTO_LOOK
    assert "long coat" in out["details"]["prompt"] and "pier" in out["details"]["prompt"]
