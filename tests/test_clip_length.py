"""How long a take a GPU renders (tasks.common.MAX_FRAMES from a per-VRAM table) and the user's bracket length."""

from __future__ import annotations

import pytest

from pickups import media
from pickups.tasks import common


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.delenv("PICKUPS_MAX_FRAMES", raising=False)
    monkeypatch.setattr(common, "MAX_FRAMES", common.MAX_FRAMES)


def test_the_table_has_a_row_for_every_card_and_ltx_lengths():
    assert common.FRAMES_BY_VRAM[0][0] == 0.0
    assert all(n % 8 == 1 for _gb, n in common.FRAMES_BY_VRAM)
    rows = [n for _gb, n in sorted(common.FRAMES_BY_VRAM)]
    assert rows == sorted(rows)  # a bigger card never renders less


def test_frames_for_picks_the_largest_row_the_card_meets(monkeypatch):
    monkeypatch.setattr(common, "FRAMES_BY_VRAM", ((0.0, 97), (30.0, 121), (80.0, 241)))
    assert common.frames_for(None) == common.DEFAULT_FRAMES
    assert common.frames_for(24.0) == 97
    assert common.frames_for(32.0) == 121
    assert common.frames_for(96.0) == 241


def test_the_env_override_decides_for_every_card(monkeypatch):
    monkeypatch.setenv("PICKUPS_MAX_FRAMES", "49")
    assert common.frames_for(96.0) == 49
    assert common.use_gpu(24.0) == common.MAX_FRAMES == 49


def test_use_gpu_sets_max_frames_and_an_unknown_card_leaves_it(monkeypatch):
    monkeypatch.setattr(common, "FRAMES_BY_VRAM", ((0.0, 97), (80.0, 241)))
    assert common.use_gpu(96.0) == 241 == common.MAX_FRAMES
    assert common.use_gpu(None) == 241  # offline: the last known


def test_length_cap_is_the_bracket_never_over_the_gpus_limit(monkeypatch):
    monkeypatch.setattr(common, "MAX_FRAMES", 121)
    assert common.length_cap({}) == 121
    assert common.length_cap({"frames": 49}) == 49
    assert common.length_cap({"frames": 500}) == 121
    assert common.length_cap({"frames": 3}) == media.MIN_FRAMES
    assert common.length_cap({"frames": "x"}) == 121
    assert common.length_cap(None) == 121


def test_work_clip_default_follows_max_frames_at_call_time(monkeypatch):
    import inspect

    assert inspect.signature(common.work_clip).parameters["frames"].default is None


def test_remove_renders_the_length_the_bracket_asks_for(tmp_path):
    import subprocess
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from pickups.tasks import remove

    work = tmp_path / "work.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=256x192:rate=24", "-frames:v", "60",
                    "-pix_fmt", "yuv420p", str(work)], check=True)
    pm = [np.zeros((48, 64), bool) for _ in range(60)]
    for m in pm:
        m[20:30, 25:40] = True

    class Out:
        name = "frame_00001.png"

        def to_file(self, path):
            Image.fromarray(np.zeros((192, 256, 3), np.uint8)).save(path)

    class Job:
        def get_outputs(self, node):
            return [Out()]

    class Ctx:
        params = {"frames": 33}

        class client:  # noqa: N801
            class assets:  # noqa: N801
                @staticmethod
                def from_file(path):
                    return Path(path).name

        def commercial_safe(self):
            return False

        def run(self, g):
            return Job()

    bg = remove.background(Ctx(), work, pm, None, (256, 192), 24.0, tmp_path, camera="moving", shadow=False)
    assert media.probe(tmp_path / "crop_ltx.mp4").frames == 33
    assert bg.n == 33


def test_the_measured_table():
    assert common.frames_for(None) == 97
    assert common.frames_for(16.0) == 97
    assert common.frames_for(25.8) == 145  # an RTX 4090
    assert common.frames_for(96.0) == 225  # RTX PRO 6000
    assert media.ltx_frames(225) == 225 and media.ltx_frames(145) == 145


def test_weather_is_shorter_than_the_rest_on_a_24_gb_card(monkeypatch):
    assert common.frames_for(25.8, "environment") == 121 and common.frames_for(25.8, "remove") == 145
    assert common.frames_for(96.0, "environment") == 225
    common.use_gpu(25.8)
    assert common.task_frames("environment") == 121 and common.task_frames("remove") == common.MAX_FRAMES == 145
    assert common.length_cap({}, "environment") == 121 and common.length_cap({"frames": 49}, "environment") == 49
    assert common.length_cap({"frames": 500}, "environment") == 121 and common.length_cap({"frames": 500}) == 145
