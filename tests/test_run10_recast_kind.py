"""Run 10 B: Replace a person on a previz mannequin take never asks whether the click is a car.

R&D's Run 9 check: the flow sent click points with no kind, so recast_ltx.clicked_vehicle asked SAM 3.1 for "vehicle",
which matched the yellow mannequin, and the render failed with "Swapping a car isn't in this version yet". The flow now
sends kind "person" (flowParams.recastParams), and the server honours it for one person and for several.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from pickups.media import Probe
from pickups.tasks import recast_ltx, recast_many


class Past(Exception):
    """The run got past the vehicle check."""


def _mannequin_track(n=97, h=64, w=114):
    m = np.zeros((h, w), bool)
    m[20:56, 50:62] = True  # one standing figure, mid-frame (the previz's yellow mannequin)
    return [m.copy() for _ in range(n)]


@pytest.fixture()
def recast(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(recast_ltx.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), 97, False, None))
    monkeypatch.setattr(recast_ltx.media, "prepare_work_clip", lambda *a, **k: {"fps": 24.0})
    monkeypatch.setattr(recast_ltx.common, "check_not_blank", lambda p: None)
    monkeypatch.setattr(recast_ltx.common, "checked_masks", lambda *a, **k: _mannequin_track())
    monkeypatch.setattr(recast_ltx.features, "cars", lambda: False)

    def segmenter(ctx, png, text, *a, **k):  # SAM 3.1 asked for "vehicle": the mannequin comes back as one
        asked.append(text)
        return [_mannequin_track()[0]]

    monkeypatch.setattr(recast_ltx.common, "select_instances", segmenter)
    monkeypatch.setattr(recast_ltx.common, "_frame_png", lambda proxy, frame, out: out)
    monkeypatch.setattr(recast_ltx, "choose_window", lambda *a, **k: (_ for _ in ()).throw(Past()))

    class Ctx:
        input_path = Path("previz_take01.mp4")
        vdir = tmp_path
        conn = None
        stages: list[str] = []

        def stage(self, label):
            self.stages.append(label)

    def go(params):
        ctx = Ctx()
        ctx.params = params
        return recast_ltx.run(ctx)

    return go, asked


CLICK = [{"x": 960, "y": 540, "positive": True}]


def test_a_person_click_on_a_mannequin_take_is_never_checked_as_a_car(recast):
    go, asked = recast
    with pytest.raises(Past):
        go({"points": CLICK, "kind": "person", "frame": 0, "engine": "ltx", "cast_id": "cast_1"})
    assert asked == []  # no "vehicle" question at all


def test_without_a_kind_the_old_check_still_runs_for_agents(recast):
    go, asked = recast
    with pytest.raises(RuntimeError, match="Swapping a car isn't in this version yet"):
        go({"points": CLICK, "frame": 0, "engine": "ltx", "cast_id": "cast_1"})
    assert asked == ["vehicle"]


def test_several_people_keep_their_kind_person():
    # recast_many asks clicked_vehicle only for a swap with no kind, so the kind must survive parsing.
    swaps = recast_many.parse_swaps({"swaps": [{"frame": 0, "points": CLICK, "kind": "person", "cast_id": "a"},
                                               {"frame": 0, "points": CLICK, "kind": "person", "cast_id": "b"}]})
    assert [s.get("kind") for s in swaps] == ["person", "person"]
