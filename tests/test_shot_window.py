"""A long shot's bracket is the part Remove, weather and previz work on, not always the first frames."""

from fractions import Fraction
from pathlib import Path

import pytest

from pickups.media import Probe
from pickups.tasks import previz, remove


def test_a_look_frame_is_measured_from_the_window():
    assert previz.look_in_window(88, 80, 97) == 8
    assert previz.look_in_window(0, 80, 97) == 0  # before the window: its first frame
    assert previz.look_in_window(200, 80, 97) == 96  # past the window: its last snapped frame
    assert previz.look_in_window(0, 0, 97) == 0


def test_remove_starts_the_clip_at_the_bracket(tmp_path, monkeypatch):
    seen = {}

    def work_clip(ctx, out, frames=0, start=0, **_k):
        seen["start"] = start
        seen["frames"] = frames
        return {"width": 16, "height": 16, "fps": 24.0}

    def tracked(ctx, proxy, frame, *a, **k):
        seen["frame"] = frame
        raise RuntimeError("stop after the pick")

    probe = Probe(16, 16, Fraction(24), 400, False, None)
    monkeypatch.setattr(remove.media, "probe", lambda path, count=True: probe)
    monkeypatch.setattr(remove.common, "work_clip", work_clip)
    monkeypatch.setattr(remove.common, "proxy_clip", lambda *a, **k: (16, 16))
    monkeypatch.setattr(remove.common, "tracked_masks", tracked)

    class Ctx:
        params = {"start_frame": 80, "frame": 90, "points": [{"x": 1, "y": 1, "positive": True}], "shadow": False}
        input_path = Path("in.mp4")
        vdir = tmp_path

        def stage(self, _label):
            return None

    with pytest.raises(RuntimeError, match="stop after the pick"):
        remove.run(Ctx())
    assert seen["start"] == 80
    assert seen["frame"] == 10  # the click is 10 frames into the window
    assert seen["frames"] == max(remove.common.MAX_FRAMES, min(400 - 80, int(remove.LOCKED_MAX_S * 24)))
