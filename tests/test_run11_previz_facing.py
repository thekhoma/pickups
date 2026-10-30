"""Run 11 C: Replace a person on a previz take takes the figure's facing, its motion and the camera from the previz.

R&D's Run 10 check 2: Take 01, the yellow mannequin with his back to the camera, the camera pushing in and orbiting.
The take's _qa said facing "front", anchor 96, sheet view "front", camera_moves false; the painter was told "walking ...
mid-stride ... seen from the front" and LTX "walks towards the camera, facing the camera". The new man faced the camera
and walked. The previz recorded all of it (previz_framing: facing, doing, speed, the camera's move).
"""

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickups import previz_facts as pf
from pickups.media import Probe
from pickups.tasks import recast_ltx

N, H, W = 97, 64, 114
WALK_WORDS = ("walking", "mid-stride", "towards the camera", "walks")


def _orbit_move(turn_deg: float, push: float = 0.6) -> dict:
    """The previz camera at frame 0 behind the figure 6 m off, and at the end `push` as far, swung `turn_deg` round."""
    r0, r1 = 6.0, 6.0 * push
    a0 = math.pi / 2  # the camera on +z of the figure
    a1 = a0 + math.radians(turn_deg)
    return {"start": {"pos": [r0 * math.cos(a0), 1.9, r0 * math.sin(a0)], "look": [0, 1, 0]},
            "end": {"pos": [r1 * math.cos(a1), 1.9, r1 * math.sin(a1)], "look": [0, 1, 0]}}


def _framing(facing="away", doing="stand", speed=None, turn=35.0, push=0.6) -> dict:
    s = {"kind": "person", "x": 0.5, "distance_m": 6.0, "in_view": True, "doing": doing, "facing": facing, "h": 0.46}
    if speed:
        s["speed_ms"] = speed
    return {"subjects": [s], "move": _orbit_move(turn, push), "camera": {"height_m": 1.9, "pitch_deg": -8, "focal_mm": 30}}


SPEC = {"camera": {"ease": "in_out", "keys": [{"t": 0, "pos": [0, 1.9, 6], "look": [0, 1, 0]},
                                             {"t": 1, "pos": [2, 1.7, 3], "look": [0, 1, 0]}]}}


# --- the previz facts --------------------------------------------------------------------------------------------------


def test_a_push_in_and_orbit_is_a_moving_camera_and_a_locked_one_is_not():
    assert pf.camera_moves(SPEC, _framing()) is True
    assert pf.camera_moves(SPEC, _framing(turn=10, push=1.0)) is True  # an orbit alone
    assert pf.camera_moves(SPEC, _framing(turn=0, push=0.6)) is True  # a push alone
    assert pf.camera_moves(SPEC, _framing(turn=0, push=1.0)) is False
    assert pf.camera_moves({}, {"subjects": []}) is None  # nothing recorded: the picture check decides


def test_a_previz_figure_with_no_speed_stands_and_a_walker_walks():
    assert pf.figure(_framing()) == pf.Figure(facing="away", moving=False, posture="standing")
    assert pf.figure(_framing(doing="walk")).moving is False  # a walk with no path is standing
    assert pf.figure(_framing(doing="walk", speed=1.4)).moving is True
    assert pf.figure(_framing(doing="sit")).posture == "sitting"


def test_the_orbit_turns_the_facing_only_past_a_clear_line():
    f = pf.facts(SPEC, _framing(turn=35), N)
    assert [f.facing_at(i) for i in (0, 48, 96)] == ["away", "away", "away"]  # 35 degrees round: still his back
    f = pf.facts(SPEC, _framing(turn=60), N)
    assert f.facing_at(0) == "away" and f.facing_at(96) == ""  # three-quarter back at the end: unsure, nothing said
    f = pf.facts(SPEC, _framing(turn=90), N)
    assert f.facing_at(96) in ("left", "right")
    assert pf.facts(SPEC, _framing(facing="camera", turn=0), N).facing_at(96) == "camera"


def test_several_figures_name_the_one_where_the_click_was_or_none():
    two = {"subjects": [{"kind": "person", "x": 0.2, "in_view": True, "facing": "away"},
                        {"kind": "person", "x": 0.8, "in_view": True, "facing": "camera"}]}
    assert pf.figure(two, x=0.75).facing == "camera"
    assert pf.figure(two, x=0.5) is None and pf.figure(two) is None


class _Store:
    def __init__(self, meta, versions=None):
        self.meta, self.versions = meta, versions or {}

    def shot_meta(self, shot_id):
        return self.meta

    def version_meta(self, shot_id, vid):
        if vid not in self.versions:
            raise FileNotFoundError(vid)
        return self.versions[vid]


def test_only_a_previz_take_or_one_with_its_blocking_counts():
    meta = {"previz_framing": _framing(), "previz_spec": SPEC, "frames": N}
    store = _Store(meta, {"v_final": {"task": "previz", "input_version": "orig", "params": {"start_frame": 8}},
                          "v_4k": {"task": "finish", "input_version": "v_final"},
                          "v_swap": {"task": "recast", "input_version": "orig"}})
    assert pf.of_take(store, "s", None)[2:] == (0, N)
    assert pf.of_take(store, "s", "v_4k")[2] == 8
    assert pf.of_take(store, "s", "v_swap") is None  # someone else stands there now
    assert pf.of_take(_Store({"frames": N}), "s", None) is None  # imported footage


# --- the whole run, mocked to the two prompts --------------------------------------------------------------------------


class Sent(Exception):
    """The LTX prompt was built: the run stops before rendering."""


def _push_in_track():
    """The mannequin standing mid-frame, growing as the camera pushes in."""
    out = []
    for i in range(N):
        m = np.zeros((H, W), bool)
        tall = int(20 + 20 * i / (N - 1))
        m[H - 4 - tall:H - 4, 52:62] = True
        out.append(m)
    return out


@pytest.fixture()
def recast_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(recast_ltx.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), N, False, None))
    monkeypatch.setattr(recast_ltx.media, "prepare_work_clip", lambda *a, **k: {"fps": 24.0})
    monkeypatch.setattr(recast_ltx.common, "check_not_blank", lambda p: None)
    monkeypatch.setattr(recast_ltx.common, "checked_masks", lambda *a, **k: _push_in_track())
    monkeypatch.setattr(recast_ltx.common, "work_clip", lambda *a, **k: {"width": 1920, "height": 1088, "fps": 24.0, "frames": N})
    monkeypatch.setattr(recast_ltx.subprocess, "run", lambda *a, **k: None)

    def frame_png(video, index, out):
        seen.setdefault("anchor_at", index)
        Image.new("RGB", (64, 36), (90, 90, 90)).save(out)
        return out

    monkeypatch.setattr(recast_ltx.common, "_frame_png", frame_png)
    monkeypatch.setattr(recast_ltx, "character_picture",
                        lambda ctx, p, run_dir, vehicle=False: (tmp_path / "sheet.png", p, "Elderly man's character sheet", 3, False))
    monkeypatch.setattr(recast_ltx, "sheet_view", lambda path, which: seen.setdefault("sheet", which) and Image.new("RGB", (8, 16)))
    monkeypatch.setattr(recast_ltx, "paint_problem", lambda *a, **k: "")
    # Imported footage reads the picture: a face on the mannequin's head (Run 10's "front") and the frames.
    monkeypatch.setattr(recast_ltx, "faces_on", lambda ctx, png, run_dir: [np.ones((H, W), bool)])
    monkeypatch.setattr(recast_ltx.common, "read_frames", lambda *a, **k: iter([np.zeros((H, W, 3), np.uint8)] * N))

    def image_edit(src, instruction, **k):
        seen["paint"] = instruction
        return {}

    def drive(clip, painted, prompt, *a, **k):
        seen["video"] = prompt
        raise Sent()

    monkeypatch.setattr(recast_ltx.graphs, "image_edit", image_edit)
    monkeypatch.setattr(recast_ltx.graphs, "drive", drive)

    class Out:
        def to_file(self, path):
            Image.new("RGB", (64, 36), (120, 100, 80)).save(path)

    class Job:
        def get_outputs(self, name):
            return [Out()]

    class Assets:
        def from_file(self, path):
            return str(path)

    class Client:
        assets = Assets()

    class Ctx:
        input_path = Path("previz_take01.mov")
        vdir = tmp_path
        conn = None
        client = Client()
        shot_id = "shot_1"
        job = {"version_id": None}

        def stage(self, label):
            pass

        def run(self, graph):
            return Job()

        def commercial_safe(self):
            return False

    def go(meta, params=None):
        ctx = Ctx()
        ctx.store = _Store(meta)
        ctx.params = params or {"points": [{"x": 960, "y": 540, "positive": True}], "kind": "person", "frame": 0,
                                "engine": "ltx", "describe": "Elderly man, an elderly man with a white beard"}
        with pytest.raises(Sent):
            recast_ltx.run(ctx)
        return seen

    return go


def test_the_back_to_camera_mannequin_on_a_push_in_and_orbit_stays_standing_with_his_back_to_us(recast_run, monkeypatch):
    qa = {}
    monkeypatch.setattr(recast_ltx, "camera_moves", lambda *a: qa.setdefault("picture_check", True))
    seen = recast_run({"previz_framing": _framing("away", "stand", turn=35), "previz_spec": SPEC, "frames": N})
    paint, video = seen["paint"], seen["video"]
    assert "picture_check" not in qa  # the previz said the camera moves; the frames weren't read for it
    assert seen["sheet"] == "back"  # the back view of the sheet is shown to the painter
    for words in (paint, video):
        assert "back to the camera" in words and "facing away from the camera" in words
        assert not any(w in words for w in WALK_WORDS), words
        assert "facing the camera" not in words and "from the front" not in words
    assert "the person standing in the middle of" in paint
    assert "is standing in place, facing away from the camera, back to the camera." in video


def test_the_previz_wins_over_the_face_found_on_the_mannequins_head(recast_run):
    # Imported footage with the same picture: the face check says "front" (Run 10's reading, kept for real footage).
    seen = recast_run({"frames": N})
    assert "facing the camera" in seen["video"] and seen["sheet"] == "front"


def test_a_front_facing_previz_walker_still_walks_towards_the_camera(recast_run):
    seen = recast_run({"previz_framing": _framing("camera", "walk", speed=1.4, turn=0, push=1.0), "previz_spec": SPEC, "frames": N})
    assert "walking" in seen["paint"] and "mid-stride" in seen["paint"] and "facing the camera" in seen["paint"]
    assert "walks towards the camera, facing the camera." in seen["video"]
    assert "back to the camera" not in seen["video"]


def test_an_unsure_facing_says_nothing_about_facing(recast_run):
    # Two figures equally near the click: which one is being replaced can't be told, so neither's facing is said.
    framing = _framing("away", "stand", turn=35)
    framing["subjects"] = [{**framing["subjects"][0], "x": 0.45}, {**framing["subjects"][0], "x": 0.55, "facing": "camera"}]
    seen = recast_run({"previz_framing": framing, "previz_spec": SPEC, "frames": N})
    for words in (seen["paint"], seen["video"]):
        assert not any(w in words for w in ("facing", "back to the camera", "from behind", "from the front", "side on"))
    assert "sheet" not in seen


def test_an_orbit_past_the_line_paints_on_a_frame_whose_facing_is_known(recast_run):
    # 60 degrees round: the end of the shot is three-quarter back (unsure), so the paint frame stays where he is
    # clearly back to the camera, and the words still say so.
    seen = recast_run({"previz_framing": _framing("away", "stand", turn=60), "previz_spec": SPEC, "frames": N})
    assert seen["anchor_at"] < 96
    assert "back to the camera" in seen["paint"] and "back to the camera" in seen["video"]
