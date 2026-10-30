"""Environment blend (envblend): the grade, the shading from the original and the contrast, in linear light."""

import numpy as np

from pickups import envblend


def _scene():
    rng = np.random.default_rng(1)
    plate = np.clip(rng.normal(120, 20, (400, 300, 3)), 0, 255).astype(np.uint8)  # hazy ground: no deep blacks
    mask = np.zeros((400, 300), bool)
    mask[100:380, 110:190] = True
    region = np.ones((400, 300), bool)
    char = plate.copy()
    char[mask] = np.clip(rng.normal(25, 40, (int(mask.sum()), 3)), 0, 255).astype(np.uint8)  # studio-dark, contrasty
    return plate, char, mask, region


def test_srgb_round_trip():
    x = np.arange(0, 256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert np.abs(envblend.to_srgb(envblend.to_lin(x)).astype(int) - x.astype(int)).max() <= 1


def test_the_subjects_blacks_are_lifted_towards_the_plates_and_the_gap_closes():
    plate, char, mask, region = _scene()
    p = envblend.frame_params(char, plate, mask, region)
    assert p is not None and 0 < p["lift"] <= envblend.LIFT_LIMIT
    before = envblend.black_gap(char, plate, mask, region)
    after = envblend.black_gap(envblend.apply(char, mask.astype(np.float32), p, None, ("grade",)), plate, mask, region)
    assert after < before


def test_a_subject_already_as_light_as_the_plate_is_not_lifted():
    plate, _, mask, region = _scene()
    p = envblend.frame_params(plate.copy(), plate, mask, region)
    assert p is not None and p["lift"] < 0.001 and np.allclose(p["gain"], 1.0, atol=0.02)


def test_shading_follows_the_originals_lateral_light_and_stays_in_limits():
    old = np.zeros((300, 200, 3), np.uint8)
    old[:] = 100
    old[:, 100:] = 200  # lit from the right
    m = np.zeros((300, 200), bool)
    m[20:280, 40:160] = True
    prof = envblend.shade_profile(old, m)
    assert prof is not None and prof[-1] > prof[0]
    new = np.zeros((300, 200), bool)
    new[20:280, 60:140] = True
    sm = envblend.shade_map(prof, new)
    assert sm[100, 130] > sm[100, 70] and sm.min() >= envblend.SHADE_LIMITS[0] and sm.max() <= envblend.SHADE_LIMITS[1]
    assert (sm[:, :60] == 1).all()
    assert envblend.lateral_agreement(np.repeat(np.where(sm[..., None] > 1, 200, 100), 3, axis=2).astype(np.uint8), new, old, m) is not None
    assert envblend.shade_profile(old, np.zeros_like(m)) is None


def test_params_are_smoothed_over_time_and_unmeasurable_frames_borrow():
    a = {"gain": np.ones(3, np.float32), "lift": 0.0, "tint": np.ones(3, np.float32), "contrast": 1.0}
    b = {"gain": np.ones(3, np.float32), "lift": 0.02, "tint": np.ones(3, np.float32), "contrast": 1.0}
    out = envblend.smooth_params([a, None, b, None], radius=1)
    assert all(o is not None for o in out) and 0 < out[1]["lift"] < 0.02
    assert envblend.smooth_params([None, None]) == [None, None]


def test_nothing_changes_without_parameters_or_steps():
    plate, char, mask, region = _scene()
    assert envblend.apply(char, mask.astype(np.float32), None, None) is char
    p = envblend.frame_params(char, plate, mask, region)
    out = envblend.apply(char, mask.astype(np.float32), p, None, ())
    assert np.abs(out.astype(int) - char.astype(int)).max() <= 1
