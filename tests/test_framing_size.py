"""The start-frame check measures size like for like: a photo of a same-size subject is not "smaller".

Fresh-9 user test: every warning said "smaller", even for a subject as big as the mannequin or bigger. The check resized
the template's edge map, and part of the mannequin (its head on a man's cap) kept scoring as the size dropped. Now the
template keeps its size, the start frame is scaled and its edges found again at that size, and a size or place is only
claimed when one size clearly wins and the match comes back on the subject (on the test's real frames: 0 of 6 false).
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import framing


def _layout(subject_h: float, subject_x: float = 0.5, photo: bool = False, seed: int = 0) -> Image.Image:
    """Frame 0 of a playblast (flat grey, a bright mannequin with raised arms) or a photo of the same blocking (texture,
    grain, a person in dark clothes with arms down, so the outline and the edges inside it differ)."""
    w, h = 1920, 1088
    rng = np.random.default_rng(seed)
    hy = int(0.45 * h)
    if photo:
        a = np.empty((h, w, 3), np.float32)
        a[:hy] = (150, 170, 195)
        a[hy:] = (105, 98, 84)
        a += rng.normal(0, 9, a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    else:
        im = Image.new("RGB", (w, h), (190, 190, 190))
        ImageDraw.Draw(im).rectangle([0, hy, w, h], fill=(110, 110, 110))
    d = ImageDraw.Draw(im)
    d.line([0, hy, w, hy], fill=(60, 60, 60) if not photo else (80, 76, 70), width=3)
    for bx in (0.08, 0.3, 0.72, 0.9):
        d.rectangle([int(bx * w), hy - int(0.18 * h), int(bx * w) + int(0.06 * w), hy],
                    fill=(150, 150, 150) if not photo else (128, 112, 96), outline=(70, 70, 70), width=3)
    fh, cx = int(subject_h * h), int(subject_x * w)
    top = int(0.85 * h) - fh
    body = (225, 200, 40) if not photo else (70, 58, 52)
    d.ellipse([cx - fh // 12, top, cx + fh // 12, top + fh // 6], fill=(40, 40, 40) if not photo else (180, 140, 120))
    d.rectangle([cx - fh // 8, top + fh // 6, cx + fh // 8, top + fh * 6 // 10], fill=body)
    if photo:  # a jacket's seams and pockets: edges inside the figure the mannequin doesn't have
        for k in range(1, 5):
            y = top + fh // 6 + k * fh // 12
            d.line([cx - fh // 10, y, cx + fh // 10, y], fill=(40, 34, 30), width=2)
    else:  # raised arms
        d.rectangle([cx - fh // 8 - fh // 20, top - fh // 10, cx - fh // 8, top + fh // 3], fill=body)
        d.rectangle([cx + fh // 8, top - fh // 10, cx + fh // 8 + fh // 20, top + fh // 3], fill=body)
    d.rectangle([cx - fh // 9, top + fh * 6 // 10, cx - fh // 40, top + fh], fill=(30, 30, 30))
    d.rectangle([cx + fh // 40, top + fh * 6 // 10, cx + fh // 9, top + fh], fill=(30, 30, 30))
    return im


def _box(subject_h: float, subject_x: float = 0.5) -> list[float]:
    top = 0.85 - subject_h
    return [subject_x - subject_h / 8, top - subject_h / 10, subject_x + subject_h / 8, 0.85]


def _save(im: Image.Image, path):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    path.write_bytes(buf.getvalue())
    return path


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_same_size_photo_is_not_smaller(tmp_path, seed):
    f0 = _save(_layout(0.5), tmp_path / "f0.jpg")
    start = _save(_layout(0.5, photo=True, seed=seed), tmp_path / "start.jpg")
    out = framing.compare(f0, {"subject": _box(0.5), "horizon": 0.45}, start)
    assert out["warning"] is None, out
    h0, h1 = out["subject_h"]
    assert abs(h1 - h0) < 0.06, out


def test_a_bigger_subject_reads_bigger_and_a_smaller_one_smaller(tmp_path):
    f0 = _save(_layout(0.5), tmp_path / "f0.jpg")
    fr = {"subject": _box(0.5)}
    big = framing.compare(f0, fr, _save(_layout(0.68, photo=True), tmp_path / "big.jpg"))
    assert big["warning"] and "bigger" in big["warning"], big
    small = framing.compare(f0, fr, _save(_layout(0.34, photo=True), tmp_path / "small.jpg"))
    assert small["warning"] and "smaller" in small["warning"], small


# Run 5: the horizon is a whole-frame edge-profile slide and locks onto a guardrail or a parapet. These are the measured
# results of two start frames that match the previz by eye (c07 seed 1001, s06 seed 2002): both warned "horizon is lower".
C07 = {"subject_sure": False, "subject_score": 0.244, "subject_x": [0.5, 0.993], "subject_h": [0.359, 0.445],
       "subject_y": [0.342, 0.775], "horizon": [0.37, 0.672]}
S06 = {"subject_sure": False, "subject_score": 0.254, "subject_x": [0.399, 0.389], "subject_h": [0.516, 0.516],
       "subject_y": [0.194, -0.126], "horizon": [-0.407, -0.176]}


@pytest.mark.parametrize("out", [C07, S06], ids=["c07", "s06"])
def test_an_unsure_match_gives_no_horizon_warning(out):
    assert framing.warning(out) is None


def test_a_horizon_outside_the_frame_gives_no_warning_even_when_sure():
    assert framing.warning({**S06, "subject_sure": True, "subject_x": [0.4, 0.4], "subject_h": [0.5, 0.5]}) is None


def test_a_sure_match_with_a_real_horizon_shift_still_warns():
    sure = {"subject_sure": True, "subject_score": 0.5, "subject_x": [0.5, 0.5], "subject_h": [0.4, 0.4], "horizon": [0.45, 0.62]}
    assert "horizon is lower" in framing.warning(sure)
    assert "horizon is higher" in framing.warning({**sure, "horizon": [0.45, 0.3]})
