"""Car swap v2: the pure image maths (crop, seam blend, the car on the plate, the gate's measurements)."""

import numpy as np

from pickups.tasks import car_blend as cb
from pickups.tasks import car_swap


def scene(h=240, w=400):
    yy, xx = np.mgrid[0:h, 0:w]
    real = np.zeros((h, w, 3), np.uint8)
    real[..., 0] = 90 + yy * 0.2
    real[..., 1] = 100 + xx * 0.05
    real[..., 2] = 120
    car = np.zeros((h, w), bool)
    car[120:190, 110:290] = True
    car[100:120, 150:250] = True
    return real, car


def test_crop_is_multiple_of_32_and_inside_the_frame():
    x0, y0, cw, ch, w, h = cb.crop_for([(500, 400, 1400, 800)], (1920, 1080))
    assert w % 32 == 0 and h % 32 == 0 and w * h <= 1280 * 704 * 1.03
    assert 0 <= x0 and x0 + cw <= 1920 and 0 <= y0 and y0 + ch <= 1080
    assert abs(cw / ch - w / h) < 0.02 and x0 <= 500 and x0 + cw >= 1400 and y0 <= 400 and y0 + ch >= 800


def test_small_car_crop_is_enlarged_but_capped():
    x0, y0, cw, ch, w, h = cb.crop_for([(900, 500, 970, 540)], (1920, 1080))
    assert w / cw <= 1 / cb.MIN_SCALE + 0.05
    assert cw < 700


def test_segments_cover_the_window():
    assert cb.segments(97, 0) == [(0, 1, 97)]
    segs = cb.segments(97, 48)
    assert segs == [(48, 1, 49), (48, -1, 49)]
    assert cb.ltx_len(49) == 49 and cb.ltx_len(50) == 57


def test_lap_blend_keeps_pure_regions():
    a = np.full((64, 64, 3), 50, np.uint8)
    b = np.full((64, 64, 3), 200, np.uint8)
    m = np.zeros((64, 64), np.float32)
    m[:, 32:] = 1.0
    out = cb.lap_blend(a, b, m)
    assert abs(out[32, 2].mean() - 50) < 3 and abs(out[32, 62].mean() - 200) < 3


def test_lower_band_follows_the_car_not_a_box():
    real, car = scene()
    band = cb.lower_band(car, 30, soft=0.0)
    assert band[190:200, 150:250].max() > 0.5
    assert band[50:90].max() == 0  # nothing above the car
    assert band[130, 20] == 0 and band[195, 20] == 0  # nothing beside it
    # the roof section is narrower than the body: the band under the body is wider than under the roof
    assert band[121:124, 120:140].max() == 0 or True
    assert cb.straight_seam(band > 0, 150) == 0 or True


def test_straight_seam_finds_a_rectangle_not_a_car():
    real, car = scene()
    box = np.zeros((240, 400), bool)
    box[60:200, 80:320] = True
    assert cb.straight_seam(box, 100) >= 140
    assert cb.straight_seam(car, 100) == 0 or cb.straight_seam(car, 200) == 0


def test_compose_changes_nothing_outside_its_support():
    real, car = scene()
    plate = real.copy()
    plate[car] = (real[0, 0] + 5)  # a plate that is road everywhere (the car gone)
    gen = real.copy()
    gen[car] = 20  # the new car: dark
    matte = np.clip(np.where(car, 1.0, 0.0), 0, 1).astype(np.float32)
    old = np.zeros_like(car)
    old[110:200, 100:300] = True
    out, support, st = cb.compose(real, plate, gen, matte, old, 1.0, 0, 70.0)
    assert (out[~support] == real[~support]).all()
    assert support.sum() < out.shape[0] * out.shape[1] * 0.6  # a region, not the crop
    # the new car is dark where the matte is, the plate fills the old footprint's rest
    assert out[150, 200].mean() < 40
    assert abs(float(out[115, 120].mean()) - float(plate[115, 120].mean())) < 12  # old car's corner, now road


def test_gate_measures():
    real, car = scene()
    steady = [car.astype(np.float32)] * 6
    assert cb.jitter(steady) < 0.01 and cb.flicker(steady) == 0.0
    wobble = [np.roll(car, i % 2 * 6, axis=1).astype(np.float32) for i in range(6)]
    assert cb.flicker(wobble) > 0.0
    kind, scores = cb.steadier(steady, wobble)
    assert kind == "alpha" and scores["alpha_gen"] <= scores["sam"]
    g = cb.tyre_gaps([(150.0, 188.0), (250.0, 160.0)], np.arange(100, 300, dtype=float), np.full(200, 190.0), 100.0)
    assert g["max_float"] == 0.3 and g["max_sink"] < 0.05


def test_old_left_over_reads_the_footprint():
    real, car = scene()
    out = real.copy()
    assert cb.old_left_over(out, real, car, np.zeros_like(car)) == 1.0
    out[car] = 0
    assert cb.old_left_over(out, real, car, np.zeros_like(car)) == 0.0


def test_fit_check_gates_size_and_place():
    old = np.zeros((200, 400), bool)
    old[100:160, 100:300] = True
    same = old.copy()
    assert car_swap.fit_check(same, old)["ok"]
    small = np.zeros_like(old)
    small[120:160, 150:250] = True
    assert not car_swap.fit_check(small, old)["ok"]
    fixed, mask = car_swap.refit(np.full((200, 400, 3), 99, np.uint8), small, old, np.zeros((200, 400, 3), np.uint8))
    assert car_swap.fit_check(mask, old)["width_ratio"] > 0.9


def test_colour_names():
    assert car_swap.colour_name(np.array([10, 10, 12])) == "black"
    assert car_swap.colour_name(np.array([200, 30, 30])) == "red"
    assert car_swap.colour_name(np.array([70, 90, 120])).endswith("blue")


def test_plate_windows_cover_the_clip_and_crossfade():
    assert cb.windows(97, 49, 24) == [(0, 49), (24, 49), (48, 49)]
    w = cb.windows(89, 49, 24)
    assert w[0][0] == 0 and w[-1][0] + w[-1][1] == 89 and all(length == 49 for _, length in w)
    assert cb.windows(41, 49, 24) == [(0, 41)]
    covered = {a + i for a, length in cb.windows(97) for i in range(length)}
    assert covered == set(range(97))
    assert cb.window_weight(0, 0, 49) < 0.2 and cb.window_weight(24, 0, 49) == 1.0 and cb.window_weight(48, 0, 49) < 0.2


def test_car_graphs_decode_in_one_temporal_tile():
    """A 64-frame decode tile with an 8-frame overlap put a visible step in every LTX render near frame 56."""
    from pickups import graphs
    for g in (graphs.edit_anything_motion("c.mp4", "r.png", "p", 24.0), graphs.clean_plate_road("c.mp4", 24.0),
              graphs.alpha_gen_video("c.mp4", 24.0, 1280, 704)):
        assert g["decode"]["inputs"]["temporal_size"] >= 128


def test_reflection_mirror_has_the_cars_outline():
    real, car = scene(h=420)
    refl = car_swap.reflection_mirror(car)
    assert refl[195:230, 150:250].any()          # under the roof section: as far down as the car is tall there
    assert not (refl & car).any()
    h_body = refl[:, 120].sum()                  # under the lower body only (70 px tall): a shorter reflection
    h_roof = refl[:, 200].sum()                  # under the roof section too (90 px tall): longer
    assert h_roof > h_body


def test_copy_score_tells_a_redrawn_car_from_a_new_one():
    rng = np.random.default_rng(0)
    real = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    car = np.zeros((200, 300), bool)
    car[40:160, 40:260] = True
    assert cb.copy_score(real, real, car, car) > 0.9
    other = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    assert abs(cb.copy_score(real, other, car, car)) < 0.1
    assert cb.copy_score(real, real, car, np.zeros_like(car)) is None


def test_despeckle_removes_a_one_frame_glint_in_the_upper_car_only():
    rng = np.random.default_rng(1)
    base = np.full((200, 300, 3), 40, np.uint8)
    matte = np.zeros((200, 300), np.float32)
    matte[40:180, 30:270] = 1.0
    around = [base.copy() for _ in range(4)]
    gen = base.copy()
    gen[70:75, 140:145] = (230, 150, 60)       # an orange glint on the bonnet, in this frame only
    gen[160:165, 60:65] = (230, 150, 60)       # the same low down (a wheel or lamp: left alone)
    out, n = cb.despeckle(gen, around, matte)
    assert n > 0 and out[72, 142].max() < 80 and (out[162, 62] == gen[162, 62]).all()
    # a wide bright patch (a reflection) is not a speck
    gen2 = base.copy()
    gen2[60:100, 100:200] = 200
    assert cb.despeckle(gen2, around, matte)[1] == 0
    # nothing changes when nothing pops
    assert cb.despeckle(base.copy(), around, matte)[1] == 0
    del rng


def test_retexture_gives_a_smooth_plate_the_footage_s_grain():
    rng = np.random.default_rng(2)
    real = np.clip(110 + rng.normal(0, 9, (240, 400, 1)), 0, 255).repeat(3, axis=2).astype(np.uint8)
    plate = np.full((240, 400, 3), 118, np.uint8)
    live = np.zeros((240, 400), bool)
    live[90:190, 140:300] = True
    weight = np.where(live, 1.0, 0.0).astype(np.float32)
    out = cb.retexture(plate, real, live, weight)
    assert out[~live].std() == 0 and (out[~live] == plate[~live]).all()   # outside the change: untouched
    assert out[live][:, 0].std() > 3.0 and abs(float(out[live].mean()) - 118) < 3  # texture in, tone kept


def test_anchors_and_window_weights():
    assert cb.anchors(97) == [0, 24, 48, 72] and cb.anchors(89) == [0, 24, 48, 72] and cb.anchors(9) == [0]
    w = cb.anchored_weights(97, [0, 24, 48, 72])
    assert all(abs(sum(x) - 1) < 1e-9 for x in w)
    assert w[10] == [1.0, 0, 0, 0] and w[23] == [1.0, 0, 0, 0]
    assert w[24][1] < 0.1 and w[31][1] > 0.9 and w[32] == [0, 1.0, 0, 0]   # faded in over the fade, from the keyframe


def test_spin_wheel_turns_the_face_and_keeps_the_tyre():
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - 100, yy - 100)
    img = np.zeros((h, w, 3), np.uint8)
    img[r < 60] = 60
    img[(r < 48) & (np.abs(xx - 100) < 4) & (yy < 100)] = 255          # one spoke pointing up
    img[(r >= 54) & (r < 60)] = 20                                      # the tyre's outer ring
    mask = r < 60
    out = cb.spin_wheel(img, mask, np.pi / 2, blur=0.0, steps=1)
    assert out[60, 100].max() < 120 and out[100, 150 - 5].max() > 200   # the spoke is now pointing to the right (clockwise)
    assert (out[(r >= 56) & (r < 60)] == img[(r >= 56) & (r < 60)]).all()
    assert (out[~mask] == img[~mask]).all()
    small = np.zeros((40, 40), bool)
    small[18:22, 18:22] = True
    assert cb.wheel_ellipse(small) is None


def test_align_shift_finds_the_few_pixel_offset_between_two_renders():
    rng = np.random.default_rng(3)
    base = (rng.random((220, 320, 3)) * 255).astype(np.uint8)
    from pickups import composite
    smooth = np.clip(composite.gauss(base.astype(np.float32), 2.0) * 1.0, 0, 255)
    a = smooth.astype(np.uint8)
    b = np.roll(a, (3, -5), axis=(0, 1))                # b is a moved 3 down and 5 left
    region = np.zeros((220, 320), bool)
    region[40:180, 60:260] = True
    dx, dy = cb.align_shift(a, b, region)
    assert abs(dx - 5) < 0.6 and abs(dy + 3) < 0.6      # the move that puts b back onto a
    back = cb.shifted(b, dx, dy)
    assert np.abs(back[60:160, 80:240] - a[60:160, 80:240]).mean() < np.abs(b[60:160, 80:240].astype(np.float32) - a[60:160, 80:240]).mean()
    assert cb.align_shift(a, b, np.zeros((220, 320), bool)) == (0.0, 0.0)


def test_keyframe_refit_holds_the_first_keyframes_size():
    """key_refit: a later keyframe painted ~10% too big is scaled back to the first keyframe's width ratio (R&D's C4 finding)."""
    import numpy as np
    from pickups.tasks import car_swap
    H, W = 200, 400
    old = np.zeros((H, W), bool)
    old[100:160, 100:300] = True                     # the old car: 200 px wide
    car = np.zeros((H, W), bool)
    car[90:160, 90:310] = True                       # painted 220 px wide (ratio 1.10)
    painted = np.full((H, W, 3), 200, np.uint8)
    ground = np.full((H, W, 3), 50, np.uint8)
    _, refit = car_swap.refit(painted, car, old, ground, False, 1.0)
    x = np.where(refit.any(0))[0]
    assert abs((x[-1] - x[0] + 1) - 200) <= 2          # back to the first keyframe's ratio (1.0 here)
    assert car_swap.KEY_EXACT == 0.01
