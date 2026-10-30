"""C3 (grain and softness of the footage) and C1 (the plate's light round the edge) in the default composite."""

import numpy as np

from pickups import composite
from pickups.tasks import person_swap as ps

H, W = 128, 160


def box_alpha(x0=40, x1=120, y0=30, y1=100, soft=1.5):
    a = np.zeros((H, W), np.float32)
    a[y0:y1, x0:x1] = 1
    return np.clip(composite.blur(a, soft), 0, 1)


def test_subject_match_tops_up_grain_to_the_originals_level_and_keeps_colour():
    rng = np.random.default_rng(8)
    plate = np.clip(rng.normal(90, 3, (H, W, 3)), 0, 255).astype(np.uint8)
    mask = box_alpha() > 0.95
    orig = np.clip(rng.normal(120, 3, (H, W, 3)), 0, 255).astype(np.uint8)
    target = composite.subject_stats(orig, mask)
    flat = np.full((H, W, 3), (60, 140, 60), np.uint8)
    out = composite.match_subject(flat, mask, target, plate, seed=1, limit=0.0)
    s = composite.subject_stats(out, mask)
    assert 0.6 * target["sigma"] < s["sigma"] < 1.15 * target["sigma"]
    assert s["sigma"] > 3 * composite.subject_stats(flat, mask)["sigma"] + 0.5
    assert abs(float(out[mask][:, 1].mean()) - 140) < 6 and float(out[mask][:, 0].mean()) < 80


def test_subject_match_softens_a_moderately_sharper_render_and_leaves_a_different_texture_alone():
    rng = np.random.default_rng(2)
    plate = np.clip(rng.normal(90, 3, (H, W, 3)), 0, 255).astype(np.uint8)
    mask = box_alpha() > 0.95
    sharp = np.zeros((H, W, 3), np.uint8)
    sharp[:, ::2] = 200
    lap = composite.subject_stats(sharp, mask)["lap"]
    base = composite.subject_stats(plate, mask)
    soft = composite.match_subject(sharp, mask, {**base, "lap": lap / 2.0}, plate, 2, limit=0.0)
    assert composite.subject_stats(soft, mask)["lap"] < lap
    wild = composite.match_subject(sharp, mask, {**base, "lap": 1.0}, plate, 2, limit=0.0)  # armour against a t-shirt
    assert composite.subject_stats(wild, mask)["lap"] > 0.5 * lap


def test_the_target_is_smoothed_over_frames():
    st = [{"mean": 100.0, "std": 10.0, "chroma": 5.0, "sigma": s, "lap": 20.0} for s in (1.0, 3.0, 1.0, 3.0, 1.0)]
    sm = composite.smooth_targets(st, radius=2)
    assert abs(sm[2]["sigma"] - 1.8) < 1e-6 and max(x["sigma"] for x in sm) < 3.0


def test_light_wrap_adds_plate_light_only_just_inside_the_edge():
    plate = np.full((H, W, 3), 200, np.uint8)
    fg = np.full((H, W, 3), 30, np.uint8)
    a = box_alpha(soft=1.0)
    out = composite.light_wrap(fg, a, plate)
    assert out[31:34, 80].mean() > 40
    assert abs(out[65, 80].mean() - 30) < 1 and abs(out[5, 5].mean() - 30) < 1
    assert np.allclose(composite.light_wrap(fg, np.zeros((H, W), np.float32), plate), fg.astype(np.float32), atol=0.01)


def test_the_default_blend_takes_a_target_and_leaves_the_plate_alone_outside_the_subject():
    rng = np.random.default_rng(3)
    bg = np.clip(rng.normal(90, 3, (H, W, 3)), 0, 255).astype(np.uint8)
    char = np.full((H, W, 3), (60, 140, 60), np.uint8)
    track = (box_alpha() > 0.5).astype(np.float32)
    region = np.ones((H, W), bool)
    target = composite.subject_stats(np.clip(rng.normal(120, 3, (H, W, 3)), 0, 255).astype(np.uint8), box_alpha() > 0.95)
    plain, m0, s0 = ps.blend_frame(bg, char, char, track, region, None, None, 1)
    fin, m1, s1 = ps.blend_frame(bg, char, char, track, region, None, None, 1, target=target)
    far = ~composite.grow(m1 > 0, 4) & (s1 >= 0.999)
    assert np.array_equal(fin[far], bg[far])  # nothing outside the matte changes
    core = composite.shrink(m1 > 0.95, 6)
    assert composite.subject_stats(fin, core)["sigma"] > composite.subject_stats(plain, core)["sigma"] + 0.5
