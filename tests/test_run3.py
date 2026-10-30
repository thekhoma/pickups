"""Run 3: the closed prompt set and the last pipeline fixes before the freeze (QA's Run 3 list, the director's RUN3_FIXES and
RUN3_ADDENDUM, R&D's BUGS_RUN2).

s04 and s06 are filled from R&D's own test shots (tests/fixtures/run3, their Run 2 home's spec and frame 0 framing) with
the Run 3 words, and must read word for word as the director's closed set. Also: the "These N are the only people" sentence,
the s04 layout without its car, and a keyed render that cuts is made again start-only.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickups import graphs
from pickups import start_frame as sf
from pickups.media import Probe
from pickups.tasks import previz

FIX = Path(__file__).parent / "fixtures" / "run3"


def _fix(shot: str) -> dict:
    return json.loads((FIX / f"{shot}.json").read_text())


def _plan(shot: str) -> sf.Plan:
    d = _fix(shot)
    return sf.plan(d["spec"], d["framing"], d["cast"], [], d["cast"], user_look=d["user_look"], method=sf.ZBUF)


def _paras(text: str) -> list[str]:
    return text.split("\n\n")


# --- s04: the small hero, no car ---------------------------------------------------------------------------------------

S04 = [
    "A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean. "
    "EXACTLY 1 person in frame: Marcus in the centre.",
    "Marcus: centre of frame, far across the park, a small figure with the bench, lawn and trees filling the frame around him.",
    "Viewpoint at about 2.6 metres above the ground, level, 30mm lens, natural perspective, background slightly soft.",
    "Marcus: a small distant man in a navy peacoat and grey flat cap, sitting on the bench.",
]


def test_s04_reads_word_for_word_as_the_closed_set():
    d, p = _fix("s04"), _plan("s04")
    paras = _paras(p.positive)
    assert paras[:4] == S04
    assert paras[4] == d["user_look"].split(" Clear morning sun")[0]  # the Look box, its light sentences in paragraph 6
    assert paras[5].startswith("Clear morning sun from frame left, low in the sky,")
    assert p.positive == d["positive"]
    assert d["spec"]["notes"] == "A man sitting on a bench in the park. The camera pushes in."


def test_s04_small_hero_has_no_number_no_distance_no_gesture():
    p = _plan("s04")
    blocking = _paras(p.positive)[1]
    assert not any(ch.isdigit() for ch in blocking) and "metres" not in blocking and "profile" not in blocking
    assert "percent" not in p.positive and "60s" not in p.positive and "beard" not in p.positive


def test_s04_has_no_car_anywhere():
    d, p = _fix("s04"), _plan("s04")
    assert "cars" not in json.dumps(d["framing"]["view"])
    assert not any(w in (p.positive + p.negative).lower() for w in (" car", "sedan", "vehicle"))
    old = json.loads((Path(__file__).parent / "fixtures" / "step20" / "s04.json").read_text())["framing"]["view"]
    assert "cars" not in json.dumps(old)  # R&D's step 20 s04 fixture lost its car too
    hand = (Path(__file__).parent / "fixtures" / "start_frame" / "s04" / "prompt_v2.txt").read_text()
    assert "sedan" not in hand and "The car" not in hand


def test_the_small_hero_rule_is_under_15_percent_of_the_frame_height():
    d = _fix("s04")
    tall = {**d["framing"], "subjects": [dict(d["framing"]["subjects"][0], h=0.16)]}
    p = sf.plan(d["spec"], tall, d["cast"], [], d["cast"], user_look=d["user_look"], method=sf.ZBUF)
    assert "far across" not in p.positive and "small distant" not in p.positive and "head is about" in p.positive
    assert sf.worn_words("a man in his 60s in a navy wool peacoat and grey flat cap") == " in a navy peacoat and grey flat cap"


# --- s06: three people, positive count words -------------------------------------------------------------------------------

def test_s06_reads_word_for_word_as_the_closed_set():
    d, p = _fix("s06"), _plan("s06")
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 3 people in frame: Ana on the left, Kofi in the centre, Jun on the right.")
    assert paras[1].endswith("Jun's head is about 7 percent of the frame height. These three are the only people on the roof; "
                             "the rest of the roof is open tar, gravel and sky.")
    assert "Kofi: a man in his 30s in a white t-shirt, black jeans, a black bucket hat pulled low and a gold chain." in paras[3]
    assert "shaved" not in p.positive
    assert ("The three friends dance in a loose row, evenly spaced with clear floor between them, each with both arms raised, "
            "loose and joyful.") in paras[4]
    assert "weathered and real" not in p.positive and "A weathered building" not in p.positive  # R2-7
    assert p.positive == d["positive"]


def test_three_people_never_say_third_person_or_nobody():
    p = _plan("s06")
    assert "third person" not in p.negative and "duplicate person, twins, clone" in p.negative
    assert not any(w in p.positive.lower() for w in ("no one else", "nobody", "third person"))


@pytest.mark.parametrize("n,word", [(2, "two"), (3, "three"), (4, "four")])
def test_two_or_more_people_add_the_only_people_sentence(n, word):
    xs = [0.2, 0.4, 0.6, 0.8][:n]
    framing = {"camera": {"height_m": 1.6, "pitch_deg": 0, "focal_mm": 35},
               "subjects": [{"kind": "person", "x": x, "distance_m": 6, "in_view": True} for x in xs]}
    p = sf.plan({"notes": "Friends stand on a lawn."}, framing, [], [], [], method=sf.ZBUF)
    assert f"These {word} are the only people in the frame; the rest of the frame is open ground." in _paras(p.positive)[1]
    one = sf.plan({"notes": "A man stands."}, {**framing, "subjects": framing["subjects"][:1]}, [], [], [], method=sf.ZBUF)
    assert "only people" not in one.positive


# --- a keyed render that cuts is made again from the start still alone -----------------------------------------------------

class _Out:
    def __init__(self, arr: np.ndarray, name: str):
        self.arr, self.name = arr, name

    def to_file(self, path):
        Image.fromarray(self.arr).save(path)


class _Outs:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames

    def get_outputs(self, name):
        return [_Out(a, f"f_{i:05d}.png") for i, a in enumerate(self.frames)]


def _clip(cut: bool) -> list[np.ndarray]:
    out = []
    for i in range(17):
        a = np.zeros((36, 64, 3), np.uint8)
        a[:, i * 3:i * 3 + 8] = 200
        out.append(255 - a if cut and i >= 9 else a)
    return out


def test_a_keyed_render_with_a_cut_is_made_again_start_only(tmp_path, monkeypatch):
    renders = []
    (tmp_path / "look.png").write_bytes(b"")
    monkeypatch.setattr(previz.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), 97, False, None))
    monkeypatch.setattr(previz.media, "prepare_work_clip", lambda src, dst, *a, **k: Path(dst).write_bytes(b"") and {"fps": Fraction(24)} or {"fps": Fraction(24)})
    monkeypatch.setattr(previz.bench, "machine", lambda conn: {"vram_gb": 24})
    monkeypatch.setattr(previz.subprocess, "run", lambda cmd, **k: Image.new("RGB", (64, 36)).save(cmd[-1]))
    monkeypatch.setattr(previz.common, "grain_pngs", lambda *a, **k: None)
    monkeypatch.setattr(previz.common, "finish_outputs", lambda *a, **k: [])
    monkeypatch.setattr(previz.common, "sequence_details", lambda *a, **k: {})
    monkeypatch.setattr("pickups.app.store.make_viewer", lambda *a, **k: None)
    monkeypatch.setattr(graphs, "layout_render", lambda *a, keyframes=None, **k: {"PU_Layout": {"keys": [x[1] for x in keyframes or []]}})
    monkeypatch.setattr(graphs, "image_edit", lambda *a, **k: {"PU_Plate": {}})

    class Store:
        def shot_cast(self, shot_id):
            return [], lambda c: None

        def shot_meta(self, shot_id):
            return {}

        def shot_dir(self, shot_id):
            return tmp_path

        def upload_path(self, uid):
            return tmp_path / "look.png"

    class Ctx:
        input_path = Path("in.mp4")
        vdir = tmp_path
        conn = None
        project_id = "prj"
        shot_id = "sht"
        store = Store()
        params = {"prompt": "She walks.", "look_frame_upload_id": "mine", "keyframes": True}
        client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

        def stage(self, _label):
            return None

        def commercial_safe(self):
            return False

        def run(self, g):
            if "PU_Layout" in g:
                renders.append(g["PU_Layout"]["keys"])
                return _Outs(_clip(cut=len(renders) == 1))
            if "PU_Plate" in g:
                return _Outs([np.zeros((36, 64, 3), np.uint8)])
            raise RuntimeError("no DA3 here")

    (tmp_path / "work").mkdir()
    out = previz.run(Ctx())
    assert renders == [[48, 96], []]  # keyed, a cut, then the start still alone
    assert out["details"]["cut_check"]["cuts"] == [9] and out["details"]["keyframes"] == []
    assert (tmp_path / "work" / "out_keys").is_dir() and not list(tmp_path.glob("key_*.png"))
