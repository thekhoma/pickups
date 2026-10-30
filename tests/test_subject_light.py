"""Relight by frequency separation: the footage's detail, the reference's broad light."""

import numpy as np

from pickups import composite
from pickups.tasks import environment as env
from pickups.tasks import subject_light as sl


def person(h=240, w=320, top=20, bottom=220, left=130, right=190):
    m = np.zeros((h, w), bool)
    m[top:bottom, left:right] = True
    return m


def textured(mask, level=120):
    rng = np.random.default_rng(3)
    img = np.full(mask.shape + (3,), 90.0, np.float32)
    img[mask] = level + rng.normal(0, 18, (int(mask.sum()), 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_the_reference_outline_is_lined_up_on_the_real_one_even_when_it_sits_elsewhere_and_a_little_bigger():
    real = person()
    ref = person(top=10, bottom=222, left=60, right=125)  # 12 px taller, 70 px left
    M = sl.align(ref, real)
    assert M is not None
    moved = composite.warp(ref.astype(np.float32), M, real.shape[::-1]) > 0.5
    assert (moved & real).sum() / (moved | real).sum() > 0.9


def test_something_half_the_size_is_not_the_same_thing():
    assert sl.align(person(top=100, bottom=160), person()) is None


def test_the_footage_takes_the_references_broad_light_and_keeps_its_own_detail():
    real = person()
    base = textured(real)
    # the reference: the same thing, lit warm on its left half and cooler, darker on its right (a broad step)
    ref = base.astype(np.float32).copy()
    left = real.copy()
    left[:, 160:] = False
    right = real & ~left
    ref[left] *= np.array([1.25, 1.0, 0.8], np.float32)
    ref[right] *= 0.7
    ref = np.clip(ref, 0, 255).astype(np.uint8)
    shape = sl.grid_shape([real])
    g = sl.frame_grid(base, ref, real, real, shape)
    assert g is not None
    steady = sl.steady_grids([g] * 9)
    fld = sl.field(steady[4], real)
    lit = base.astype(np.float32) / 255
    out = np.clip(((lit ** 2.2) * np.exp(fld)) ** (1 / 2.2), 0, 1)
    # the left half is warmer than the right, as in the reference
    lo, ro = out[real & left.any(0)[None, :] & (np.arange(320)[None, :] < 150)], out[real & (np.arange(320)[None, :] > 170)]
    assert lo[:, 0].mean() / lo[:, 2].mean() > ro[:, 0].mean() / ro[:, 2].mean() + 0.1
    assert lo.mean() > ro.mean()
    # and its fine detail is the footage's: high-pass of the result follows the footage's high-pass
    hp = lambda a: a - composite.smooth(a, 4)  # noqa: E731
    a = hp(out[..., 1])[real][::7]
    b = hp(lit[..., 1])[real][::7]
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_a_wild_reference_is_held_within_a_lit_side_and_a_shaded_one():
    real = person()
    base = textured(real, 120)
    ref = np.where(real[..., None], 250, 5).astype(np.uint8)  # a white drawing: a different thing's colours
    g = sl.frame_grid(base, ref, real, real, sl.grid_shape([real]))
    fld = sl.field(sl.steady_grids([g])[0], real)
    assert fld[real].max() <= sl.LUM_MED[1] + sl.LUM_SPREAD + sl.TINT_MED + sl.TINT_SPREAD + 1e-3


def test_no_reference_where_it_was_not_found():
    real = person()
    assert sl.frame_grid(textured(real), textured(real), real, np.zeros_like(real), (16, 8)) is None
    assert sl.steady_grids([None, None]) == [None, None]


def test_a_gap_between_two_people_is_decided_the_same_on_every_frame():
    a = np.zeros((120, 200), bool)
    a[10:110, 20:80] = True
    a[10:110, 120:180] = True
    gap = np.zeros_like(a)
    gap[60:100, 80:120] = True  # the pocket between their hands
    masks = []
    for i in range(9):
        m = a.copy()
        m[60:100, 80:120] = i % 3 == 0  # SAM fills it on every third frame
        masks.append(m)
    frames = [np.full(a.shape + (3,), 100, np.uint8)] * 9
    out = env.stable_masks(frames, masks, reach=4)
    filled = [bool(o[60:100, 80:120].all()) for o in out]
    assert len(set(filled)) == 1
    assert all(o[10:110, 20:80].all() for o in out)  # the people themselves stay


def test_an_object_the_restyle_adds_far_off_is_found_and_the_cars_that_were_there_are_not():
    before = {"car": np.zeros((540, 960), bool), "person": np.zeros((540, 960), bool)}
    before["car"][300:320, 100:160] = True
    after = {"car": before["car"].copy(), "person": np.zeros((540, 960), bool)}
    assert env.invented_objects(before, after) == []
    after["car"][290:310, 800:860] = True  # a second one, far on the right
    found = env.invented_objects(before, after)
    assert [lb for lb, _ in found] == ["car"]


def test_the_held_road_is_not_painted_over_the_kept_car(tmp_path, monkeypatch):
    """hold_ground puts a thin road back over every frame; a kept car crossing it must stay the footage's own pixels."""
    from PIL import Image

    from pickups.tasks import common

    cw, ch = 192, 108
    work = tmp_path / "w.mp4"
    frames = [np.full((ch, cw, 3), 90, np.uint8) for _ in range(3)]
    for f in frames:
        f[62:66, 0:cw] = 40  # the road's line
        f[50:70, 80:110] = (200, 30, 30)  # the car, across it
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cw}x{ch}", "-r", "24", "-i", "-",
                    "-pix_fmt", "yuv420p", str(work)], input=b"".join(f.tobytes() for f in frames), check=True)
    out = tmp_path / "out"
    out.mkdir()
    for i in range(3):
        g = np.full((ch, cw, 3), (60, 140, 50), np.uint8)
        g[50:70, 80:110] = (200, 30, 30)
        Image.fromarray(g).save(out / f"r_{i + 1:05d}.png")
    road = np.zeros((ch, cw), bool)
    road[62:66, :] = True
    monkeypatch.setattr(env.common, "select_preview", lambda *a, **k: road if "orig" in str(a[-1]) else np.zeros((ch, cw), bool))
    monkeypatch.setattr(env, "camera_still", lambda *a, **k: True)
    car = np.zeros((ch, cw), bool)
    car[50:70, 80:110] = True

    class Ctx:
        def check(self): pass
    held = env.hold_ground(Ctx(), work, out, (cw, ch, 0, 0, cw, ch), tmp_path, [car] * 3)
    assert held
    img = np.asarray(Image.open(out / "r_00002.png").convert("RGB")).astype(int)
    assert np.abs(img[53:67, 85:105] - np.array([200, 30, 30])).max() <= 2  # the car as it was
    assert np.abs(img[64, 10] - np.array([60, 140, 50])).max() > 10  # the road line was held elsewhere
    _ = common


def test_a_render_that_strays_from_the_footages_camera_is_put_back_onto_it():
    rng = np.random.default_rng(5)
    from PIL import Image
    base = np.asarray(Image.fromarray((rng.random((60, 80, 3)) * 255).astype(np.uint8)).resize((640, 480), Image.BILINEAR))
    moved = env.shifted(base, 7.0, -3.0)
    a, b = env.edge_image(base, None), env.edge_image(moved, None)
    dx, dy = env.phase_offset(a, b, 640 / 1024)
    assert abs(dx - 7.0) < 1.0 and abs(dy + 3.0) < 1.0
    back = env.shifted(moved, -dx, -dy)
    assert np.abs(back[20:-20, 20:-20].astype(int) - base[20:-20, 20:-20]).mean() < 6
    d = env.smooth_offsets(np.array([[0.0, 0], [0, 0], [9, 0], [0, 0], [0, 0]]))
    assert abs(d[2][0]) < 4  # one bad match is no jump


def test_a_kept_car_is_solid_and_a_cable_car_is_not():
    assert not env.thin_kind("the SUV") and not env.thin_kind("the car") and not env.thin_kind("the man")
    assert env.thin_kind("the cable car") and env.thin_kind("a wire fence")


def test_what_a_person_takes_from_a_change_of_light_leaves_their_detail_the_footages_own():
    rng = np.random.default_rng(1)
    h, w = 200, 160
    pic = np.full((h, w, 3), (90, 120, 70), np.uint8)  # green ground
    person = np.zeros((h, w), bool)
    person[30:190, 60:100] = True
    pic[person] = np.clip(170 + rng.normal(0, 14, (int(person.sum()), 3)), 0, 255)  # a pale shirt with texture
    warm = pic.copy()
    warm[~person] = (150, 120, 40)  # the ground turns amber
    res = env.person_bounce(pic, warm, person.astype(np.float32), 1.0, env.GRADE_CHROMA_MAX, lum=0.6, head=0.85)
    hp = lambda a: a.astype(float)[..., 1] - composite.smooth(a.astype(float)[..., 1:2], 1.5)[..., 0]  # noqa: E731
    core = composite.shrink(person, 4)
    assert np.corrcoef(hp(res)[core], hp(pic)[core])[0, 1] > env.LOCK_MIN
    assert res[core][:, 0].mean() / res[core][:, 2].mean() > pic[core][:, 0].mean() / pic[core][:, 2].mean() + 0.02  # warmer


def test_a_large_standing_water_find_in_the_rain_is_an_invention_and_a_small_one_is_not():
    before = {"puddle": np.zeros((540, 960), bool)}
    after = {"puddle": np.zeros((540, 960), bool)}
    after["puddle"][400:420, 100:160] = True  # a small shallow puddle
    assert env.invented_objects(before, after) == []
    after["puddle"][380:480, 300:700] = True  # a lake across the path
    found = env.invented_objects(before, after)
    assert found and found[0][0] == "puddle"


def test_a_persons_matte_follows_their_edge_on_the_footage_and_is_solid_inside():
    h, w = 160, 120
    yy, xx = np.mgrid[:h, :w]
    person = (yy > 20) & (yy < 150) & (xx > 40) & (xx < 80)
    o = np.full((h, w, 3), 150, np.uint8)
    o[person] = (30, 30, 40)
    wide = composite.grow(person, 4)  # SAM's outline a little wide
    a = env.person_matte(o, o, wide)
    assert (a[composite.shrink(person, 3)] > 0.98).all()
    assert a[~composite.grow(person, 10)].max() < 0.01  # nothing beyond a little past SAM's outline
