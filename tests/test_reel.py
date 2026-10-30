"""The reel's plan and timing (scripts/make_reel.py); the ffmpeg render itself isn't run here."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_reel.py"
_spec = importlib.util.spec_from_file_location("make_reel", _PATH)
reel = importlib.util.module_from_spec(_spec)
sys.modules["make_reel"] = reel
_spec.loader.exec_module(reel)


def _gallery(tmp_path, ids, private=(), missing=()):
    items = []
    for i in ids:
        e = {"id": i, "task": "remove", "title": f"Title {i}", "slate": "REMOVE · REAL PIXELS", "public": i not in private,
             "before": f"{i}_before.mp4", "after": f"{i}_after.mp4"}
        items.append(e)
        if i not in missing:
            (tmp_path / e["before"]).write_bytes(b"")
            (tmp_path / e["after"]).write_bytes(b"")
    (tmp_path / "examples.json").write_text(json.dumps(items))
    return reel.gallery(tmp_path)


def test_plan_fits_fifteen_seconds(tmp_path):
    ids = [f"ex{i}" for i in range(7)]
    parts, skipped = reel.plan(_gallery(tmp_path, ids), ids)
    # No title card: frame 0 (the feed's thumbnail) is the first example, opening mid-wipe.
    assert [p.kind for p in parts] == ["example"] * 5 + ["end"]
    assert parts[0].start == 0 and parts[0].cold and not any(p.cold for p in parts[1:])
    assert [p.example["id"] for p in parts[:-1]] == ids[:5]  # in the order asked for
    assert parts[-1].seconds == pytest.approx(1.5)
    assert all(p.seconds == pytest.approx(2.7) for p in parts[:-1])
    assert all(b.start == a.start + a.frames for a, b in zip(parts, parts[1:]))  # back to back, no gaps
    assert reel.total_frames(parts) / reel.FPS <= 15.0
    assert set(skipped) == {"ex5", "ex6"}


def test_plan_skips_private_and_missing(tmp_path):
    ids = ["a", "b", "c", "d"]
    available = _gallery(tmp_path, ids, private={"b"}, missing={"c"})
    parts, skipped = reel.plan(available, ["a", "b", "c", "d", "a", "nope"])
    assert [p.example["id"] for p in parts if p.kind == "example"] == ["a", "d"]
    assert set(skipped) == {"b", "c", "nope", "a"}  # private, no loops, unknown, asked twice
    with pytest.raises(ValueError):
        reel.plan(available, ["b", "c"])


def test_wipe_holds_sweeps_and_lands():
    assert reel.wipe(0) == 0 and reel.wipe(reel.HOLD_F) == 0
    assert reel.wipe(reel.HOLD_F + reel.WIPE_F) == 1 and reel.wipe(reel.SEG_F - 1) == 1
    steps = [reel.wipe(n) for n in range(reel.SEG_F)]
    assert steps == sorted(steps)
    # About 0.6 s of before, 0.7 s of wipe and the longest look for the after.
    assert (reel.HOLD_F, reel.WIPE_F) == (18, 21) and reel.SEG_F - reel.HOLD_F - reel.WIPE_F >= reel.FPS * 1.3


def test_cold_open_starts_split():
    # The first frame is already split (the after on the left), holds under the wordmark, then finishes the reveal.
    assert reel.wipe(0, cold=True) == reel.COLD_AT == reel.wipe(reel.COLD_HOLD_F, cold=True)
    assert 0.3 <= reel.COLD_AT <= 0.6 and reel.wipe(reel.COLD_HOLD_F + reel.COLD_WIPE_F, cold=True) == 1
    steps = [reel.wipe(n, cold=True) for n in range(reel.SEG_F)]
    assert steps == sorted(steps)
    assert reel.line_frames(cold=True)[0] == 0 and reel.INTRO_F / reel.FPS <= 0.6
    assert reel.SEG_F - reel.COLD_HOLD_F - reel.COLD_WIPE_F >= reel.FPS * 1.3


def test_crop_follows_the_change():
    n, cols = 120, reel.COLS
    left = [[200 if 8 <= x < 14 else 2 for x in range(cols)] if 40 <= i < 90 else [2] * cols for i in range(n)]
    first, focus = reel.choose(left, local=True)
    assert focus < 0.3  # the subject is on the left
    assert 30 <= first <= 45  # the window starts as the subject comes in
    assert reel.choose(left, local=False) == ((n - reel.SEG_F) // 2, 0.5)  # weather etc.: middle, centred
    assert reel.choose([[2] * cols] * n, local=True)[1] == 0.5  # nothing stands out: centred


def test_frames_draw_at_size():
    ex = {"title": "Previz to final", "slate": "PREVIZ → FINAL · LTX-2.5"}  # → isn't in the font: drawn by hand
    for fmt, size in reel.FORMATS.items():
        lay = reel.layout(fmt)
        assert reel.overlay(lay, ex, 3, 5).size == size and reel.intro(lay).size == size
        assert reel.card(lay, "end", 0).size == size and reel.card(lay, "end", reel.END_F - 1, cta="example.org/pickups").size == size
    with pytest.raises(ValueError):
        reel.layout("1x1")
    with pytest.raises(ValueError):
        reel.card(reel.layout("9x16"), "open", 0)  # the reel opens on a picture


def test_nine_by_sixteen_keeps_the_bottom_band_clear():
    # TikTok and Reels lay their caption and handle over the bottom band: no type of ours goes there.
    lay = reel.layout("9x16")
    type_layer = reel.captions(lay, {"title": "Castle in a winter dusk", "slate": "CHANGE WEATHER · LTX-2.5"}, 1, 5)
    px, py, pw, ph = lay.panel
    assert type_layer.getchannel("A").crop((0, py, lay.w, lay.h)).getbbox() is None
    assert type_layer.getchannel("A").crop((0, 0, lay.w, py)).getbbox() is not None
