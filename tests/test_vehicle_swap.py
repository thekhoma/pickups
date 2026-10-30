"""Recast a vehicle (tasks/vehicle_swap): the old car's path from its outline, spinning wheels, the stand-in's warp, the
composite's promise (the clean plate, bit for bit, outside the new car and its shadow band), the clean plate's check,
and a whole swap with ComfyUI faked."""

import math
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import composite, reference
from pickups.app.jobs import TaskContext
from pickups.tasks import common, person_swap, recast, recast_ltx, vehicle_swap as vs


def car_mask(h, w, cx, cy, s=1.0):
    """A car's side outline: a body with a cabin on top and two wheel arches' worth of round bottom."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    bw, bh = 220 * s, 60 * s
    d.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], fill=255)
    d.polygon([(cx - 80 * s, cy - bh / 2), (cx - 55 * s, cy - bh / 2 - 45 * s), (cx + 40 * s, cy - bh / 2 - 45 * s),
               (cx + 75 * s, cy - bh / 2)], fill=255)
    for wx in (-70, 70):
        d.ellipse([cx + (wx - 24) * s, cy + bh / 2 - 24 * s, cx + (wx + 24) * s, cy + bh / 2 + 24 * s], fill=255)
    return np.asarray(im) > 127


def test_a_sheared_path_keeps_the_vehicles_shape():
    shear = np.array([[1.0, 0.45, 12.0], [0.05, 0.8, -4.0], [0.0, 0.0, 1.0]])
    S = vs.as_similarity(shear)
    cols = S[:2, :2]
    assert abs(float(np.dot(cols[:, 0], cols[:, 1]))) < 1e-6
    assert abs(float(np.linalg.norm(cols[:, 0]) - np.linalg.norm(cols[:, 1]))) < 1e-6
    assert np.allclose(S[:2, 2], [12.0, -4.0])


# --- the old car's path ---------------------------------------------------------------------------------------------------


def test_the_outline_tracker_follows_an_affine_path_and_smooths_its_jitter():
    W, H, n, s = 1280, 720, 13, 0.5  # work size; masks at half size, as on the 1024 px proxy
    truth, masks = [], []
    rng = np.random.default_rng(3)
    for i in range(n):
        A = np.array([[1 + 0.004 * i, 0.0, 6.0 * i], [0.0, 1 + 0.004 * i, 1.5 * i], [0, 0, 1]])
        truth.append(A)
        m = composite.warp(car_mask(H, W, 560, 400, 2.0).astype(np.float32), A, (W, H)) > 0.5  # frame 0's outline, moved by A
        if i % 3 == 1:  # SAM's outline boils a pixel now and then
            m = composite.dilate_box(m, 1) if rng.random() < 0.5 else composite.erode_box(m, 1)
        masks.append(np.asarray(Image.fromarray(m).resize((int(W * s), int(H * s)), Image.BILINEAR)))
    Ms, qa = vs.track_vehicle(masks, 0, (W, H))
    assert np.allclose(Ms[0], np.eye(3))
    box = np.array([[340, 250, 1], [780, 250, 1], [780, 470, 1], [340, 470, 1]], float).T
    for M, T in zip(Ms, truth):
        err = np.linalg.norm((M @ box - T @ box)[:2], axis=0)
        assert err.max() < 2.0, err
    assert qa["track_iou_min"] > 0.93 and len(qa["track_iou"]) == n


def test_a_squashing_outline_moves_the_rigid_car_with_its_tyres_on_the_old_ones():
    W, H, n, s = 1280, 720, 25, 0.5
    truth, masks = [], []
    for i in range(n):  # the old car's outline flattens a little as it yaws, far from the picture's corner
        A = np.array([[1.0, 0.0, 2.0 * i], [0.0, 1 - 0.004 * i, 0.0], [0, 0, 1]])
        truth.append(A)
        m = composite.warp(car_mask(H, W, 640, 520, 2.0).astype(np.float32), A, (W, H)) > 0.5
        masks.append(np.asarray(Image.fromarray(m).resize((int(W * s), int(H * s)), Image.BILINEAR)))
    Ms, qa = vs.track_vehicle(masks, 0, (W, H))
    xs, line = vs.contact_line(car_mask(H, W, 640, 520, 2.0))
    pin = np.array([(xs[0] + xs[-1]) / 2, line[len(line) // 2], 1.0])
    for M, T in zip(Ms, truth):
        assert np.linalg.norm((M @ pin - T @ pin)[:2]) < 3.0  # where its tyres meet the road, it follows the old one
    corner = vs.as_similarity(truth[-1])  # pinned at the picture's corner instead, it sinks off the road
    assert np.linalg.norm((corner @ pin - truth[-1] @ pin)[:2]) > 30
    assert np.allclose(vs.as_similarity(truth[-1], pin[:2]) @ pin, truth[-1] @ pin)
    for M in Ms:  # and it stays rigid: no shear, no squash
        L = M[:2, :2]
        assert abs(float(np.dot(L[:, 0], L[:, 1]))) < 1e-6 and abs(np.linalg.norm(L[:, 0]) - np.linalg.norm(L[:, 1])) < 1e-6
    assert qa["rigid_iou_min"] > 0.85 and qa["rigid_iou_min"] <= qa["track_iou_min"]


def test_the_path_is_smoothed_but_the_anchor_frame_stays_put():
    box = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], float)
    rng = np.random.default_rng(0)
    raw = []
    for i in range(30):
        M = np.eye(3)
        M[0, 2], M[1, 2] = 3.0 * i + rng.normal(0, 2.0), rng.normal(0, 2.0)
        raw.append(M)
    sm = vs.smooth_path(raw, box, ref=10)
    assert np.allclose(sm[10], raw[10])
    jitter_raw = np.std(np.diff([M[1, 2] for M in raw]))
    jitter_sm = np.std(np.diff([M[1, 2] for M in sm]))
    assert jitter_sm < 0.3 * jitter_raw
    assert abs(sm[25][0, 2] - 75) < 4  # the drive itself is kept


# --- wheels --------------------------------------------------------------------------------------------------------------


def wheel_picture():
    """A grey car body with one near wheel (a bright five-spoke rim in a dark tyre) and one far wheel (all tread)."""
    img = np.full((240, 400, 3), 120, np.uint8)
    car = np.zeros((240, 400), bool)
    car[40:200, 20:380] = True
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.ellipse([60, 110, 160, 210], fill=(25, 25, 25))  # near tyre
    d.ellipse([80, 130, 140, 190], fill=(210, 210, 210))  # rim
    for j in range(5):
        a = 2 * math.pi * j / 5
        d.line([(110, 160), (110 + 28 * math.cos(a), 160 + 28 * math.sin(a))], fill=(40, 40, 40), width=5)
    d.ellipse([260, 120, 330, 200], fill=(30, 30, 30))  # far wheel: tread only
    d.rectangle([285, 120, 305, 200], fill=(70, 70, 70))
    img = np.asarray(im).copy()
    near = np.zeros((240, 400), bool)
    yy, xx = np.mgrid[:240, :400]
    near[(xx - 110) ** 2 + (yy - 160) ** 2 <= 50 ** 2] = True
    far = ((xx - 295) / 35.0) ** 2 + ((yy - 160) / 40.0) ** 2 <= 1
    steering = ((xx - 200) ** 2 + (yy - 70) ** 2) <= 12 ** 2  # small and high up: not a road wheel
    return img, car, [near, far, steering]


def test_wheels_spin_in_their_own_plane_and_nothing_else_changes():
    img, car, found = wheel_picture()
    wheels = vs.wheels_of(img, found, car)
    assert sorted(w.far for w in wheels) == [False, True]  # the steering wheel is left out
    out = vs.spin(img, wheels, frame=3)
    # Around the near rim the spokes are smeared into a ring: far less change along the turn than before.
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ring = (np.rint(160 + 20 * np.sin(ang)).astype(int), np.rint(110 + 20 * np.cos(ang)).astype(int))
    assert out[ring].astype(float).std(axis=0).mean() < 0.4 * img[ring].astype(float).std(axis=0).mean()
    # The far wheel is blurred up and down only: its columns keep their mean.
    assert np.abs(out[130:190, 270:320].astype(float).mean(0) - img[130:190, 270:320].astype(float).mean(0)).max() < 6
    # Off the wheels nothing moves.
    touched = np.zeros(car.shape, bool)
    for w in wheels:
        x0, y0, x1, y1 = w.box
        touched[y0:y1, x0:x1] |= w.soft > 0
    assert np.array_equal(out[~touched], img[~touched])
    assert not np.array_equal(vs.spin(img, wheels, 1), vs.spin(img, wheels, 2))  # it turns from frame to frame


# --- the stand-in's warp --------------------------------------------------------------------------------------------------


def test_the_warp_moves_pixels_exactly_and_into_a_crop():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (90, 160, 3)).astype(np.float32)
    M = np.array([[1, 0, 7], [0, 1, -4], [0, 0, 1.0]])
    out = composite.warp(img, M, (160, 90))
    assert np.allclose(out[10:80, 7:150], img[14:84, 0:143], atol=1e-3)  # content moved by (7, -4)
    # Warping straight into a crop is the same as cropping the full warp.
    S = np.array([[1.05, 0.02, 3.5], [-0.01, 0.97, 2.0], [0, 0, 1.0]])
    full = composite.warp(img, S, (160, 90))
    crop = composite.warp(img, vs._into(S, 32, 18), (64, 40))
    assert np.allclose(crop, full[18:58, 32:96], atol=1e-3)
    # A frame-to-frame move inside the crop is the full move seen from the crop.
    Mc = vs._shift(S, 32, 18)
    p = np.array([10.0, 12.0, 1.0])
    assert np.allclose(Mc @ p + [32, 18, 0], S @ (p + [32, 18, 0]))


def test_the_crop_is_one_to_one_in_multiples_of_128_around_the_whole_path():
    boxes = [(451 + 4 * i, 276, 1574 + 4 * i, 883 + 110) for i in range(25)]
    x0, y0, cw, ch, w, h = vs.crop_box(boxes, (1920, 1080), 1920 * 1088)
    assert (w, h) == (cw, ch) and w % 128 == 0 and h % 128 == 0
    assert x0 <= 451 and x0 + cw >= 1574 + 96 and y0 <= 276 and y0 + ch >= 993 and y0 + ch <= 1080
    # A 4K shot over the pixel budget is scaled, never stretched past the crop's own shape.
    boxes4 = [(b[0] * 2, b[1] * 2, b[2] * 2, b[3] * 2) for b in boxes]
    x0, y0, cw, ch, w, h = vs.crop_box(boxes4, (3840, 2160), 1920 * 1088)
    assert w * h <= 1920 * 1088 and w % 128 == 0 and h % 128 == 0 and abs(w / h - cw / ch) < 0.12


# --- the composite ---------------------------------------------------------------------------------------------------------


def test_the_contact_line_runs_along_the_tyres_bottoms_not_the_body():
    car = car_mask(300, 480, 240, 140, 1.0)  # wheels' bottoms at row 194, the body's bottom between them at 170
    xs, line = vs.contact_line(car)
    bottom = 299 - np.argmax(car[::-1, xs], axis=0)
    assert (line >= bottom + 1 - 1e-6).all()  # never above the outline: it is the outline's lower hull
    at = lambda x: float(line[np.searchsorted(xs, x)])  # noqa: E731
    assert at(170) == pytest.approx(195, abs=0.5) and at(310) == pytest.approx(195, abs=0.5)
    assert at(240) == pytest.approx(195, abs=0.5)  # straight from tyre to tyre, under the body


def test_the_ground_is_under_the_vehicle_down_from_its_tyres_and_never_above_it():
    car = car_mask(300, 480, 240, 140, 1.0)
    rows = np.flatnonzero(car.any(axis=1))
    tall = rows[-1] - rows[0] + 1
    g = vs.ground_region(car)
    assert g[195:195 + int(0.06 * tall), 170:311].min() > 0.9  # the road just under its tyres and between them
    assert g[171:190, 200:280].min() > 0.9  # under the body, between the wheels
    assert g[:int(rows[-1] - (vs.GROUND_LIFT + 0.12) * tall)].max() == 0  # nothing around its cabin or roof
    assert g[int(195 + (vs.GROUND_REACH + 3 * vs.GROUND_SOFT) * tall) + 2:].max() == 0  # it ends a little under it


def test_the_vehicle_stands_on_the_renders_road_and_outside_it_the_plate_is_untouched():
    rng = np.random.default_rng(5)
    h, w = 300, 480
    plate = rng.integers(90, 160, (h, w, 3)).astype(np.uint8)
    gen = np.clip(rng.integers(90, 160, (h, w, 3)) + 25, 0, 255).astype(np.uint8)  # its own road, a little lighter
    hard = car_mask(h, w, 240, 140, 0.8)
    gen[hard] = (60, 110, 50)
    ys, xs = np.nonzero(hard)
    gen[ys.max() - 4:ys.max() + 10, xs.min():xs.max()] = (40, 40, 40)  # the render's contact shadow under it
    matte = composite.gauss(composite.choke1(hard).astype(np.float32), 1.1)
    matte[matte < 0.02] = 0
    ground = vs.ground_region(hard)
    out, wt, st = composite.car_on_ground(gen, plate, matte, ground, hard, need=np.array([2.0, 2.0, 2.0]), seed=0)
    keep = ~((matte > 0) | (ground > 0))
    assert keep.sum() > 0.5 * h * w
    assert np.array_equal(out[keep], plate[keep])
    assert ((wt > 0) == ((ground > 0) & (matte < 1))).all()
    inner = matte >= 1
    assert 0.88 <= min(st["gain"]) and max(st["gain"]) <= 1.12
    assert np.abs(out[inner].astype(float) - np.array([60, 110, 50]) * st["gain"]).mean() < 3
    # Under its tyres: the render's shadow, brought to the plate's tone (25 darker), not the plate's bright road.
    contact = (ground > 0.95) & ~composite.grow(hard, 2)
    contact[:ys.max() - 4] = False
    contact[ys.max() + 10:] = False
    assert out[contact].mean() < 40 and plate[contact].mean() > 110
    # Where the ground fades into the plate the render's road matches it: no lighter ring, no edge.
    edge = (ground > 0.1) & (ground < 0.6) & ~composite.grow(hard, 20)
    edge[ys.max() - 4:ys.max() + 10] = False
    assert abs(out[edge].astype(float).mean() - plate[edge].astype(float).mean()) < 4


def test_a_road_line_only_the_plate_has_does_not_tint_the_renders_road():
    rng = np.random.default_rng(1)
    h, w = 300, 480
    plate = rng.integers(95, 105, (h, w, 3)).astype(np.uint8)
    gen = plate.copy()
    plate[226:238, :] = (220, 180, 40)  # a yellow centre line on the plate, just under the vehicle's ground...
    hard = car_mask(h, w, 240, 140, 0.8)  # ...that the render drew elsewhere (here: not at all)
    gen[hard] = (60, 110, 50)
    matte = composite.gauss(hard.astype(np.float32), 1.1)
    ground = vs.ground_region(hard)
    out, _, _ = composite.car_on_ground(gen, plate, matte, ground, hard, need=None, seed=0)
    under = (ground > 0.9) & (matte == 0)
    tint = out[under].astype(float).mean(axis=0) - 100
    assert np.abs(tint).max() < 6, tint


def test_the_matte_is_smoothed_along_the_cars_motion_then_choked_and_feathered():
    h, w = 200, 320
    Mc, raw = [], []
    for i in range(5):
        M = np.eye(3)
        M[0, 2] = 4 * i
        Mc.append(M)
        m = car_mask(h, w, 140 + 4 * i, 100, 0.6)
        if i == 2:
            m = m.copy()
            m[60:70, 100:180] = False  # SAM drops a strip on one frame
            m[150:160, 20:40] = True  # and grabs a speck
        raw.append(m)
    matte = vs.smooth_mattes(raw, Mc, 2)
    truth = car_mask(h, w, 148, 100, 0.6)
    assert (matte[62:70, 125:160] > 0.99).mean() > 0.9  # the dropped strip is voted back in
    assert matte[150:160, 20:40].max() == 0  # the speck is voted out
    assert matte[~composite.dilate_box(truth, 4)].max() == 0  # the feather stays at the edge
    assert ((matte > 0) & (matte < 1)).sum() > 0  # soft edge


def test_a_plate_with_a_dark_patch_where_the_car_stood_fails_the_check():
    h, w = 360, 640
    road = np.zeros((h, w, 3), np.uint8)
    road[:180] = (205, 208, 210)  # sky
    road[180:] = (110, 112, 115)  # road
    car = car_mask(h, w, 320, 200, 0.8)
    footage = road.copy()
    footage[car] = (40, 40, 45)
    ys, xs = np.nonzero(car)
    footage[ys.max():ys.max() + 20, xs.min() + 10:xs.max() - 10] = (50, 50, 55)  # its shadow
    smudged = road.copy()
    smudged[ys.max() - 8:ys.max() + 18, xs.min() + 20:xs.max() - 20] = (78, 78, 80)
    assert vs.under_darkness(road, car) == pytest.approx(0, abs=2)
    assert vs.under_darkness(footage, car) > 30
    assert vs.under_darkness(smudged, car) > vs.REMNANT_LEVELS


def test_the_old_cars_shadow_is_looked_for_on_the_ground_only(tmp_path):
    asked = []

    class Ctx:
        client = _Client()

        def run(self, graph):
            asked.append(next(n["inputs"]["text"] for n in graph.values() if n["class_type"] == "CLIPTextEncode"))
            return _Job({})

    objs = [car_mask(120, 200, 100, 60, 0.4)] * 3
    out = common.shadow_masks(Ctx(), tmp_path / "proxy.mp4", objs, tmp_path, labels=vs.SHADOW_LABELS)
    assert [a.split(":")[0] for a in asked] == list(vs.SHADOW_LABELS) and len(out) == 3
    rows = vs.ground_rows(objs[0])
    ys = np.flatnonzero(objs[0].any(axis=1))
    assert not rows[:(ys[0] + ys[-1]) // 2].any() and rows[(ys[0] + ys[-1]) // 2 + 1:].all()


def test_the_view_is_said_from_the_old_cars_lights_and_its_boxs_shape():
    assert vs.view_words(True, 1.5) == "a front view, seen almost head-on"
    assert vs.view_words(False, 1.5) == "a rear view, seen almost straight from behind"
    assert vs.view_words(None, 1.9) == "a three-quarter view"  # not known: not said (the painter follows the old car)
    assert vs.view_words(False, 2.0) == "a three-quarter rear view"
    assert vs.view_words(True, 3.1) == "a side view"
    assert vs.HIGH.search("Seen from a drone, it drives along a gravel road") and not vs.HIGH.search("It drives along a wet road")
    # No grille or lights to go on: a car that grows over the shot comes towards the camera, one that shrinks goes.
    coming = [car_mask(300, 480, 240, 140, 0.5 + 0.02 * i) for i in range(20)]
    assert vs.approach(coming) > vs.COMING and vs.approach(coming[::-1]) < 1 / vs.COMING
    head_on = np.zeros((300, 480), bool)
    head_on[100:220, 160:330] = True

    class Sam:
        def __init__(self, finds=()):
            self.finds, self.asked = finds, []

        def run(self, graph):
            label = next(n["inputs"]["text"] for n in graph.values() if n["class_type"] == "CLIPTextEncode").split(":")[0]
            self.asked.append(label)
            raise RuntimeError("nothing") if label not in self.finds else None

    assert vs.vehicle_view(Sam(), None, head_on, None, nearer=1.6) == "a front view, seen almost head-on"
    assert vs.vehicle_view(Sam(), None, head_on, None, nearer=0.6) == "a rear view, seen almost straight from behind"
    assert "the whole vehicle in a rear view, seen" in recast_ltx.VEHICLE_STILL.format(who="a wagon", view=vs.view_words(False, 1.4))


def test_a_render_drawn_off_the_footage_is_measured_away_from_the_car():
    rng = np.random.default_rng(0)
    h, w = 512, 768
    world = (composite.gauss(rng.normal(0, 1, (h + 80, w + 80)).astype(np.float32), 3) * 400 + 128).clip(0, 255)
    footage = np.dstack([world[40:40 + h, 40:40 + w]] * 3).astype(np.uint8)
    keep = np.ones((h, w), bool)
    keep[180:370, 230:570] = False  # the car and the old one's place: not measured
    for dx, dy in ((3.0, -5.0), (-12.0, 20.0)):  # the render's world drawn this far off
        render = composite.warp(world, np.array([[1, 0, dx - 40], [0, 1, dy - 40], [0, 0, 1.0]]), (w, h))
        render = np.dstack([np.clip(render, 0, 255)] * 3).astype(np.uint8)
        render[200:350, 250:550] = (60, 110, 50)  # its car, which the footage doesn't have
        got = composite.masked_offset(footage, render, keep)
        assert abs(got[0] - dx) < 0.5 and abs(got[1] - dy) < 0.5
    assert composite.masked_offset(footage, footage, np.zeros((h, w), bool)) is None  # nothing to go on
    # A horizon over a stormy sea whose waves the render redraws: the horizon says how far off it is, the waves nothing.
    def sea(top, seed):
        img = np.full((h, w), 190.0, np.float32)
        img[top:] = 70 + rng.normal(0, 18, (h - top, w))
        return np.dstack([np.clip(composite.gauss(img, 1.0), 0, 255)] * 3).astype(np.uint8)
    got = composite.masked_offset(sea(220, 1), sea(205, 2), keep)
    assert got is not None and abs(got[1] + 15) < 1.0 and got[0] == 0.0


def test_the_renders_drift_is_a_smooth_curve_through_noisy_readings():
    n = 57
    truth = [(0.0, -10 + 10 * i / 56) for i in range(n)]  # it drifts back onto the footage by the keyframe
    readings = [(a + 0.3 * ((i * 7) % 3 - 1), b + 0.3 * ((i * 5) % 3 - 1)) for i, (a, b) in enumerate(truth)]
    for i in range(13, 18):
        readings[i] = (0.0, 9.6)  # a 5-frame bump of wrong readings
    readings[3] = None
    readings[30] = (40.0, -50.0)
    got = vs.smooth_offsets(readings)
    assert np.abs(got - np.array(truth)).max() < 1.0
    assert np.abs(np.diff(got[:, 1])).max() < 0.5  # no jolt from one frame to the next
    assert np.array_equal(vs.smooth_offsets([None] * 10), np.zeros((10, 2)))


def test_the_shape_check_forgives_a_body_two_pixels_off_but_not_a_morph():
    a = car_mask(200, 320, 160, 100, 0.8).astype(np.float32) * 200
    a[90:96, 100:220] = 60  # a window line
    off = np.roll(a, 2, axis=0)  # the same body, drawn 2 px lower
    other = car_mask(200, 320, 160, 100, 0.8).astype(np.float32) * 200
    other[60:140, 150:156] = 60  # a different body: its line runs the other way
    where = composite.dilate_box(a > 0, 3)
    assert composite.edge_agreement(off, a, where) < 0.8
    assert composite.edge_agreement(off, a, where, shift=3) > 0.99
    assert composite.edge_agreement(other, a, where, shift=3) < composite.edge_agreement(off, a, where, shift=3) - 0.05


def test_on_a_wet_road_the_old_cars_reflection_is_cleared_straight_down_from_its_tyres():
    car = car_mask(300, 480, 240, 120, 0.8)
    rows = np.flatnonzero(car.any(axis=1))
    tall = rows[-1] - rows[0] + 1
    col = vs.reflection_column(car, reach=0.8)
    xs, line = vs.contact_line(car)
    assert not (col & car).any()
    assert col[int(line.max()) + 2:int(line.min() + 0.75 * tall), xs[5]:xs[-5]].all()  # under it, all the way across
    assert not col[:, :xs[0]].any() and not col[:, xs[-1] + 1:].any()  # never beside it
    assert not col[int(line.max() + 0.85 * tall):].any()
    assert vs.reflection_column(car)[-1, xs[len(xs) // 2]]  # by default its lamps' streaks, to the frame's foot
    assert vs.wet_road({"scene": "It drives along a wet two-lane rural road."})
    assert vs.wet_road({"scene": "Rain falls on the coast road."}) and not vs.wet_road({"scene": "A dusty gravel road."})
    assert not vs.wet_road({"scene": "A wet road.", "wet": False})


def test_the_fill_is_said_in_the_scenes_own_words():
    p = {"scene": "It drives along a wet two-lane rural road across flat, frosty farmland under a flat grey winter sky."}
    assert vs.fill_words(p).startswith("the empty wet two-lane rural road and the scenery beyond it")
    assert "no dark patch" in vs.fill_words({})
    assert vs.fill_words({"scene": "Seen from a drone, it drives along a gravel road across grassland."}).startswith("the empty gravel road and")


def test_a_clicked_vehicle_switches_the_jobs_stages_and_keeps_its_place():
    class M:
        def __init__(self):
            self.updates = []

        def _update(self, job, **f):
            job.update(f)
            self.updates.append(f)

    m = M()
    ctx = TaskContext.__new__(TaskContext)
    ctx.m, ctx.job, ctx.stages, ctx._i, ctx._done_est = m, {}, recast_ltx.STAGES, 1, 15.0
    import time
    ctx._t0 = time.monotonic()
    ctx.use_stages(vs.STAGES)
    assert ctx.job["stages"] == [s.label for s in vs.STAGES] and ctx._i == 1
    from pickups.tasks import car_swap
    assert recast_ltx.stages_for({"text": "the SUV"}) == car_swap.STAGES and recast_ltx.stages_for({"text": "man", "guide": "pose"}) == person_swap.STAGES
    assert recast_ltx.stages_for({"text": "the SUV", "car_engine": "proxy"}) == vs.STAGES  # the first car swap, on request
    assert recast.stages_for({"text": "SUV"}) == car_swap.STAGES and recast.stages_for({"text": "SUV", "kind": "person", "guide": "pose"}) == person_swap.STAGES
    assert recast_ltx.stages_for({"text": "man", "guide": "depth"}) == recast_ltx.STAGES


# --- a whole swap, ComfyUI faked ------------------------------------------------------------------------------------------


W, H, N = 640, 384, 25
BLUE, GREEN, HUB, TYRE = (30, 40, 200), (40, 190, 60), (235, 190, 60), (22, 22, 22)
RED = (255, 0, 0)


def colour(img, c, tol=40):
    return np.abs(img.astype(int) - np.array(c)).max(-1) < tol


def car_x(i):
    return 230 + 3 * i


def background(i):
    """Sky, fields and road in bands, and fence posts sliding by (the camera tracks with the car)."""
    yy, xx = np.mgrid[:H, :W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:150] = (200, 204, 208)
    bg[150:230] = (150, 135, 110)
    bg[230:] = (100, 102, 106)
    posts = ((xx + 5 * i) % 60 < 4) & (yy > 120) & (yy < 175)
    bg[posts] = (70, 60, 50)
    return bg


def car_at(i):
    m = car_mask(H, W, car_x(i) + 110, 230, 0.7)
    return m


def draw_car(f, i, body):
    m = car_at(i)
    f[m] = body
    for wx in (-49, 49):
        cx, cy = car_x(i) + 110 + wx, 230 + 21
        yy, xx = np.mgrid[:H, :W]
        f[(xx - cx) ** 2 + (yy - cy) ** 2 <= 15 ** 2] = TYRE
        f[(xx - cx) ** 2 + (yy - cy) ** 2 <= 8 ** 2] = HUB
    return f


def source():
    frames, masks = [], []
    for i in range(N):
        f = draw_car(background(i), i, BLUE)
        frames.append(f)
        m = car_at(i)
        masks.append(m)
    return frames, masks


def row_fill(img, hole):
    """Each hole pixel takes the nearest pixel outside the hole on its own row (the fakes' in-painter)."""
    out = img.copy()
    for y in np.flatnonzero(hole.any(axis=1)):
        ok = np.flatnonzero(~hole[y])
        if not len(ok):
            continue
        xs = np.flatnonzero(hole[y])
        idx = np.clip(np.searchsorted(ok, xs), 0, len(ok) - 1)
        left = ok[np.clip(idx - 1, 0, len(ok) - 1)]
        right = ok[idx]
        near = np.where(np.abs(xs - left) <= np.abs(right - xs), left, right)
        out[y, xs] = img[y, near]
    return out


class _Out:
    def __init__(self, path):
        self.path, self.name = path, path.name

    def to_file(self, dst):
        dst.write_bytes(self.path.read_bytes())


class _Job:
    def __init__(self, outs):
        self.outs = outs

    def get_outputs(self, name):
        return self.outs.get(name, [])


class _Assets:
    def from_file(self, path):
        return str(path)


class _Client:
    assets = _Assets()


class _Writer:
    made: dict = {}
    real = common.FrameWriter

    def __init__(self, out, w, h, fps, kind="prores", scale_long=None):
        self.frames = []
        self.inner = None if out.name in ("master.mov", "receipt.mp4") else _Writer.real(out, w, h, fps, kind, scale_long)
        _Writer.made[out.name] = self

    def write(self, frame):
        self.frames.append(frame.copy())
        if self.inner:
            self.inner.write(frame)

    def close(self):
        if self.inner:
            self.inner.close()


def _save_masks(out, masks):
    outs = []
    for j, m in enumerate(masks):
        Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"mask_{j + 1:05d}_.png")
        outs.append(_Out(p))
    return _Job({"PU_Mask": outs})


class _Ctx:
    def __init__(self, store, sid, params, tmp):
        from pathlib import Path
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.client, self.ran, self.tmp, self.entered, self.labels = _Client(), [], Path(tmp), [], []
        self.stages = list(recast_ltx.STAGES)
        self.progressed = []

    def use_stages(self, stages):
        self.stages = list(stages)

    def stage(self, label):
        assert label in [s.label for s in self.stages], label
        self.entered.append(label)

    def check(self):
        pass

    def progress(self, frac):
        self.progressed.append(frac)

    def commercial_safe(self):
        return False

    def run(self, graph):
        self.ran.append(graph)
        out = self.tmp / f"job{len(self.ran)}"
        out.mkdir()
        types = {n["class_type"] for n in graph.values()}
        if "TextEncodeQwenImage21" in types:
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB")).copy()
            if colour(f, RED, 5).any():  # Remove's plate: the red area filled from beside it
                f = row_fill(f, colour(f, RED, 5))
            else:  # the painter: the new vehicle (green) where the old one (blue) is
                assert "PU_Frame2" in graph
                f[colour(f, BLUE)] = GREEN
            Image.fromarray(f).save(p := out / "plate_00001_.png")
            return _Job({"PU_Plate": [_Out(p)]})
        if "LTXVInpaintPreprocess" in types:  # Remove's LTX in/outpaint: the masked area filled from beside it
            masks = [m[..., 0] > 127 for m in common.read_frames(graph["PU_MaskClip"]["inputs"]["file"])]
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                Image.fromarray(row_fill(f, masks[min(i, len(masks) - 1)])).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Out": files})
        if "LTXAddVideoICLoRAGuide" in types:  # Layout-to-Render: the stand-in, re-rendered (the background a bit off)
            assert "PU_Layout" in graph
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Layout"]["inputs"]["file"])):
                g = f.astype(int)
                g[~colour(f, GREEN, 60)] += (9, -5, 7)
                Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Out": files})
        if "SAM3_Detect" in types:
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB"))
            label = graph["PU_Label"]["inputs"]["text"].split(":")[0]
            self.labels.append(label)
            if label == "car":
                ys = colour(f, GREEN) | colour(f, TYRE, 20) | colour(f, HUB, 20)
                return _save_masks(out, [reference.largest(composite.grow(colour(f, GREEN), 3) & ys)])
            if label == "wheel":
                lab, n = reference.label(composite.grow(colour(f, TYRE, 20) | colour(f, HUB, 20), 1))
                return _save_masks(out, [lab == j for j in range(1, n + 1)])
            return _save_masks(out, [np.zeros(f.shape[:2], bool)])
        if "SAM3_VideoTrack" in types:  # following the rendered car: everything green, and its wheels
            seed = np.asarray(Image.open(graph["PU_First"]["inputs"]["image"]).convert("L")) > 127
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                m = colour(f, GREEN, 60) | colour(f, TYRE, 30) | colour(f, HUB, 40)
                m = composite.grow(m, 1) & ~colour(f, (100, 102, 106), 12)
                if i == 0:
                    assert (seed & m).sum() > 0.8 * seed.sum()  # seeded on the stand-in's outline
                Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"mask_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Mask": files})
        raise AssertionError(f"unexpected graph {sorted(types)}")


@pytest.fixture()
def shot(tmp_path, monkeypatch):
    import pickups.app.store as st
    from pickups.tasks import vehicle_swap
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    frames, masks = source()
    src = tmp_path / "road.mov"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(src)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    sid = store.add_shot(src, "Road")["id"]
    car = Image.new("RGB", (600, 300), (200, 200, 200))
    ImageDraw.Draw(car).rectangle([100, 100, 500, 220], fill=GREEN)
    from io import BytesIO
    b = BytesIO()
    car.save(b, "PNG")
    up = store.add_upload(b.getvalue(), "wagon.png")["id"]
    monkeypatch.setattr(common, "MAX_FRAMES", 17)
    small = [np.asarray(Image.fromarray(m).resize((W, H))) for m in masks]
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **k: small)
    monkeypatch.setattr(common, "shadow_masks", lambda ctx, proxy, objs, run, labels=(): [np.zeros_like(o) for o in objs])
    monkeypatch.setattr(common, "FrameWriter", _Writer)
    monkeypatch.setattr(common, "outputs_from_master", lambda *a, **k: [])
    monkeypatch.setattr(recast_ltx, "make_viewer", lambda *a, **k: None)
    monkeypatch.setattr(vehicle_swap, "make_viewer", lambda *a, **k: None)
    _Writer.made = {}
    return store, sid, up, frames


def test_on_a_wet_road_the_old_cars_reflection_area_is_the_plates_road_not_the_renders(shot, tmp_path):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It drives along a wet country road."}, tmp_path / "jobs")
    res = recast.run(ctx)
    assert res["details"]["pixels_changed_outside_selection"] == 0
    out = _Writer.made["master.mov"].frames
    for i in (4, 12):
        ys, xs = np.nonzero(car_at(i))
        far = out[i][ys.max() + 70:ys.max() + 100, xs.min() + 40:xs.max() - 40].astype(int)
        # the fake render's road is (9, -5, 7) off the footage's (100, 102, 106); the plate's is the footage's
        assert np.abs(far.mean(axis=(0, 1)) - np.array([100, 102, 106])).max() < 3


def test_a_whole_vehicle_swap_offline(shot, tmp_path):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It drives along a wet country road."}, tmp_path / "jobs")
    res = recast.run(ctx)
    d, qa = res["details"], res["details"]["_qa"]
    assert ctx.entered == [s.label for s in vs.STAGES]
    assert qa["engine"] == "vehicle-proxy" and d["subject"] == "vehicle" and d["engine"] == "ltx" and res["label"] == "Replace a car"
    # The road was cleared inside this job, with the scene's own words, and checked.
    rem = qa["plate_remnants"]
    assert rem["ok"] and rem["fill"].startswith("the empty wet country road") and len(rem["tries"]) == 1
    plate_edit = next(g for g in ctx.ran if "TextEncodeQwenImage21" in {n["class_type"] for n in g.values()} and "PU_Frame2" not in g)
    assert "the empty wet country road" in plate_edit["PU_Prompt"]["inputs"]["prompt"]
    # Painted whole, at the frame's own shape (padded to 32 px, never squashed).
    painter = next(g for g in ctx.ran if "PU_Frame2" in g)
    assert Image.open(painter["PU_Frame"]["inputs"]["image"]).size == (640, 384)
    assert Image.open(ctx.vdir / "painted_frame.png").size == (W, H)
    # Layout-to-Render on a 1:1 crop in multiples of 128, the painted look as the keyframe at the anchor.
    render = next(g for g in ctx.ran if "PU_Layout" in g)
    x0, y0, cw, ch = qa["crop"]
    assert qa["one_to_one"] and qa["work_size"] == [cw, ch] and cw % 128 == 0 and ch % 128 == 0
    for i in range(17):  # the old car's place is inside the render's crop on every frame (the render redraws it)
        oy, ox = np.nonzero(car_at(i) | vs.reflection_column(car_at(i)))  # and its reflection down the wet road
        assert x0 <= ox.min() and ox.max() < x0 + cw and y0 <= oy.min() and oy.max() < y0 + ch
    assert render["lay_full"]["inputs"]["width"] == cw and render["g2_kf0"]["inputs"]["frame_idx"] == qa["anchor_frame"] == 0
    assert render["g1_kf0"]["inputs"]["strength"] == 1.0
    assert "LTX-2.5 + Layout-to-Render IC-LoRA" in d["models"] and "Depth Anything 3" not in " ".join(d["models"])
    assert qa["wheels"] == 2 and ctx.labels[:5] == ["grille", "headlight", "tail light", "car", "wheel"]
    # The angle the old car is seen from goes to the painter (no lights found: which end isn't said; three-quarter by
    # its box).
    assert qa["view"] == "a three-quarter view"
    assert "Here it is seen in a three-quarter view" in painter["PU_Prompt"]["inputs"]["prompt"]
    # The old car's path, from its outline.
    assert qa["track_iou_min"] > 0.9
    # The composite: the clean plate outside the new car and its shadow; no trace of the old car; the new one where it was.
    assert d["pixels_changed_outside_selection"] == 0
    out = _Writer.made["master.mov"].frames
    assert len(out) == 17 and out[0].shape == (H, W, 3)
    for i in range(17):
        assert not colour(out[i], BLUE, 30).any()
        body = composite.shrink(car_at(i) & colour(frames[i], BLUE, 5), 4)
        assert colour(out[i][body], GREEN, 30).mean() > 0.95
        assert np.array_equal(out[i][:100], frames[i][:100])  # the sky far above: the footage
    assert qa["shape_stability_min"] > 0.9 and len(qa["shape_stability"]) == 17
    me = qa["matte_edges"]
    assert me["iou_with_stand_in_min"] > 0.85 and me["choke_px"] == 1 and me["feather_px"] == 1.1
    receipt = res["receipt"]
    assert receipt["ai"] > 0 and receipt["original"] + receipt["ai"] == pytest.approx(100, abs=0.02)
    tracks = [g for g in ctx.ran if "track" in g and g["track"]["class_type"] == "SAM3_VideoTrack"]
    assert len(tracks) == 1 and tracks[0]["PU_Mask"]["inputs"]["filename_prefix"].startswith("pickups/recast_track/")


def test_someone_in_front_of_the_car_stays_footage(shot, tmp_path, monkeypatch):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    _, masks = source()
    small = [np.asarray(Image.fromarray(m).resize((W, H))) for m in masks]

    def man(i):  # someone standing in front of the car's back half
        m = np.zeros((H, W), bool)
        m[195:290, car_x(i) + 60:car_x(i) + 80] = True
        return m

    def tracked(*a, others_out=None, **k):
        if others_out is not None:
            others_out[:] = [man(i) for i in range(len(small))]
        return [m & ~man(i) for i, m in enumerate(small)]

    monkeypatch.setattr(common, "tracked_masks", tracked)
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It drives along a wet country road."}, tmp_path / "jobs")
    res = recast.run(ctx)
    out = _Writer.made["master.mov"].frames
    work = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))  # the footage, as the job decoded it
    for i in range(len(out)):
        core = composite.shrink(man(i), 3)
        assert np.array_equal(out[i][core], work[i][core])  # in front of the new car, untouched
        assert colour(out[i][composite.shrink(car_at(i) & ~composite.grow(man(i), 6) & colour(frames[i], BLUE, 5), 4)], GREEN, 30).mean() > 0.9
    assert res["details"]["pixels_changed_outside_selection"] == 0


def test_a_painting_whose_tyres_float_is_made_again_with_the_next_seed(shot, tmp_path, monkeypatch):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    lifts = iter([0.12, 0.0])  # the first painting stands 12% of the car's height above the road; the second on it
    monkeypatch.setattr(vs, "floats", lambda new, old: next(lifts))
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It drives along a wet country road."}, tmp_path / "jobs")
    res = recast.run(ctx)
    gate = res["details"]["_qa"]["paint_gate"]
    painters = [g for g in ctx.ran if "PU_Frame2" in g]
    assert len(painters) == 2 and [x["floats"] for x in gate] == [0.12, 0.0]
    assert [x["seed"] for x in gate] == [42, 43]
    prompts = [g["PU_Prompt"]["inputs"]["prompt"] for g in painters]
    assert "exactly as big in the picture" not in prompts[0] and "exactly as big in the picture as the SUV" in prompts[1]


def test_a_small_car_is_painted_on_an_enlarged_window_and_put_back(tmp_path):
    W, H = 640, 360
    small = car_mask(H, W, 420, 200, 0.12)  # a drone shot's car, ~16 px tall
    assert vs.zoom_box(car_mask(H, W, 320, 180, 1.0), (W, H)) is None  # a big one is painted as it is
    x0, y0, x1, y1 = vs.zoom_box(small, (W, H))
    ys, xs = np.nonzero(small)
    assert x0 <= xs.min() and xs.max() < x1 and y0 <= ys.min() and ys.max() < y1
    assert abs((x1 - x0) / (y1 - y0) - W / H) < 0.05 and (x1 - x0) == round(W / vs.ZOOM_MAX)  # the frame's shape, 4x closer
    seen = []

    class Ctx:
        client = _Client()

        def run(self, graph):  # the painter: hands back what it was shown, a little redder
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB")).astype(int)
            seen.append(f.shape)
            f[..., 0] = np.clip(f[..., 0] + 40, 0, 255)
            Image.fromarray(f.astype(np.uint8)).save(tmp_path / "p.png")
            return _Job({"PU_Plate": [_Out(tmp_path / "p.png")]})

    anchor = np.full((H, W, 3), 100, np.uint8)
    out = vs.paint(Ctx(), anchor, "paint it", [], 1, False, tmp_path, zoom=(x0, y0, x1, y1))
    assert seen[0][:2] == (384, 640)  # the window, shown at the frame's size (padded to 32 px)
    outside = np.ones((H, W), bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(out[outside], anchor[outside])
    assert out[(y0 + y1) // 2, (x0 + x1) // 2, 0] == 140  # the painted window's middle, put back


def test_a_painted_car_far_bigger_than_the_old_one_or_elsewhere_fails_the_gate():
    old = car_mask(300, 480, 240, 140, 0.6)
    assert vs.placement(old, old)["ok"] and vs.placement(old, old)["miss"] == 0
    big = car_mask(300, 480, 240, 120, 1.25)  # painted twice the size (the drone shot's wagon)
    away = car_mask(300, 480, 400, 140, 0.6)  # or beside it
    assert not vs.placement(big, old)["ok"] and vs.placement(big, old)["size"] > 1.6
    assert not vs.placement(away, old)["ok"] and vs.placement(away, old)["iou"] < vs.PLACE_IOU
    assert vs.placement(big, old)["miss"] > 0 and vs.placement(away, old)["miss"] > 0


def test_the_float_check_measures_the_new_contact_line_against_the_old():
    old = car_mask(300, 480, 240, 140, 1.0)
    rows = np.flatnonzero(old.any(axis=1))
    tall = rows[-1] - rows[0] + 1
    high = np.roll(old, -20, axis=0)  # the same car painted 20 px up: in the air
    low = np.roll(old, 12, axis=0)  # or a little nearer the camera: still on the road
    assert vs.floats(old, old) == pytest.approx(0, abs=1e-6)
    assert vs.floats(high, old) == pytest.approx(20 / tall, abs=0.01) and vs.floats(high, old) > vs.FLOAT_MAX
    assert vs.floats(low, old) < 0
    assert vs.floats(np.zeros_like(old), old) is None


def test_a_car_the_camera_follows_is_cleared_by_the_generated_fill():
    h, w = 360, 640
    tracked = [car_mask(h, w, 300 + i, 200, 0.8) for i in range(30)]  # it stays put on screen: nothing behind it shows
    across = [car_mask(h, w, 120 + 12 * i, 200, 0.8) for i in range(30)]  # it drives through a still frame
    assert vs.keeps_its_place(tracked)[0] and not vs.keeps_its_place(across)[0]


def test_on_a_locked_off_shot_the_old_cars_reflection_goes_with_it(tmp_path):
    from pickups.tasks import remove
    h, w, n = 192, 320, 9
    frames = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:90] = (200, 204, 208)
        f[90:] = (100, 102, 106)
        frames.append(f)
    work = tmp_path / "work.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(work)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    car = [car_mask(h, w, 100 + 10 * i, 80, 0.3) for i in range(n)]  # it drives through a still frame
    col = [vs.reflection_column(m) for m in car]

    class Ctx:
        client = _Client()

        def commercial_safe(self):
            return False

    for given in (False, True):
        bg = remove.background(Ctx(), work, car, col, (w, h), 24.0, tmp_path, camera="locked", given_locked=given)
        cx0, cy0, cw, ch = bg.box
        below = np.zeros((h, w), bool)
        below[cy0:cy0 + ch, cx0:cx0 + cw] = bg.mattes_hard[4]
        assert below[col[4]].mean() > (0.95 if given else -1) and (given or below[col[4]].mean() < 0.5)


def test_the_cars_end_is_said_only_when_its_lights_agree(tmp_path):
    head_on = np.zeros((300, 480), bool)
    head_on[100:220, 160:330] = True
    part = lambda y0, x0: np.pad(np.ones((20, 30), bool), ((y0, 280 - y0), (x0, 450 - x0)))  # noqa: E731
    Image.fromarray(np.zeros((300, 480, 3), np.uint8)).save(tmp_path / "f.png")

    def view(finds, nearer):
        class Ctx:
            client = _Client()
            n = 0

            def run(self, graph):
                label = graph["PU_Label"]["inputs"]["text"].split(":")[0]
                out = tmp_path / f"job{len(finds)}_{nearer}_{Ctx.n}"
                Ctx.n += 1
                out.mkdir()
                return _save_masks(out, finds.get(label) or [np.zeros_like(head_on)])
        return vs.vehicle_view(Ctx(), tmp_path / "f.png", head_on, tmp_path / f"v{len(finds)}{nearer}", nearer=nearer)

    # The misty road: the camera car pulls away from the SUV coming after it, so it shrinks, but its grille and
    # headlights say it shows its front.
    front = {"grille": [part(160, 210)], "headlight": [part(140, 170), part(140, 290)]}
    assert view(front, 0.68) == "a front view, seen almost head-on"
    # The cliff top: SAM called the parked SUV's rear diffuser a grille, but both tail lights say its back.
    back = {"grille": [part(190, 210)], "tail light": [part(130, 170), part(130, 290)]}
    assert view(back, 0.98) == "a rear view, seen almost straight from behind"
    # One light, nothing else: which end isn't said.
    assert view({"tail light": [part(130, 170)]}, 1.0) == "a view from almost straight on"



def test_a_car_coming_towards_the_camera_is_painted_in_the_middle_of_the_shot():
    coming = [car_mask(300, 480, 240, 140, 0.5 + 0.01 * i) for i in range(57)]
    assert vs.paint_anchor(coming, 56, 57) in (24, 32)  # the middle, on LTX's latent grid
    assert vs.paint_anchor(coming[::-1], 0, 57) == 0  # going away: painted where it is biggest, as before
    steady = [car_mask(300, 480, 240, 140, 0.8)] * 97
    assert vs.paint_anchor(steady, 0, 97) == 0


def test_a_parked_car_with_a_door_swinging_open_stays_put():
    h, w = 360, 640
    frames = []
    for i in range(20):
        m = car_mask(h, w, 300, 200, 0.9)
        m[120:175, 395:395 + 3 * i] = True  # a door (or boot) opening on its upper body, wider each frame
        frames.append(m)
    assert vs.parked(frames)
    Ms, qa = vs.track_vehicle(frames, 0, (w, h))
    assert qa["parked"] and all(np.allclose(M, np.eye(3)) for M in Ms)
    driving = [car_mask(h, w, 200 + 6 * i, 200, 0.9) for i in range(20)]
    assert not vs.parked(driving)


def test_a_parked_cars_wheels_are_not_spun(shot, tmp_path, monkeypatch):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    real = vs.track_vehicle
    monkeypatch.setattr(vs, "track_vehicle", lambda pm, ref, size: (real(pm, ref, size)[0], {"track_iou": [1.0], "parked": True}))
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It stands on a cliff top."}, tmp_path / "jobs")
    qa = recast.run(ctx)["details"]["_qa"]
    assert qa["parked"] and qa["wheels"] == 0


def test_a_parked_car_someone_gets_out_of_is_turned_away_before_any_gpu_time(shot, tmp_path, monkeypatch):
    h, w = 360, 640
    parked_car = [car_mask(h, w, 300, 200, 0.9)] * 20

    def man(i):  # walks up out of the car's side over the shot
        m = np.zeros((h, w), bool)
        m[140:300, 380 + i:400 + i] = True
        return m
    why = vs.doors_or_people(parked_car, [man(i) for i in range(20)])
    assert why and why["parked"] and why["touching_frames"] >= vs.TOUCH_FRAMES
    driving = [car_mask(h, w, 150 + 8 * i, 200, 0.9) for i in range(20)]
    assert vs.doors_or_people(driving, [man(i) for i in range(20)]) is None  # someone passing a car that drives: fine
    doors = []
    for i in range(20):
        m = car_mask(h, w, 300, 200, 0.9)
        m[130:185, 400:400 + 4 * i] = True  # a door swinging open (its upper part); nobody else
        doors.append(m)
    assert vs.doors_or_people(doors, []) is not None
    assert vs.doors_or_people(parked_car, []) is None

    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(vs, "doors_or_people", lambda pm, om: {"touching_frames": 6})
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It stands on a cliff top."}, tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="doors opened, or people getting in or out"):
        recast.run(ctx)
    assert not any("PU_Frame2" in g or "PU_Layout" in g or "LTXVInpaintPreprocess" in {n["class_type"] for n in g.values()}
                   for g in ctx.ran)  # nothing painted, filled or rendered


def test_frames_whose_render_draws_the_world_off_are_found():
    err = [8.3, 7.9, 3.5, 3.3, 3.2, 3.2, 3.1, 3.1, 3.4, 3.0]  # the coast road's first two frames
    assert list(np.flatnonzero(vs.world_outliers(err))) == [0, 1]
    assert not vs.world_outliers([3.0, 3.2, 5.5, 3.1]).any()  # a little worse is not off
    assert len(vs.world_outliers([])) == 0


def test_a_wet_roads_mirror_image_lies_under_the_car_and_nowhere_else():
    h, w = 300, 480
    img = np.full((h, w, 3), 90, np.uint8)
    car = car_mask(h, w, 240, 120, 0.8)
    img[car] = (40, 160, 40)
    matte = car.astype(np.float32)
    xs, line = vs.contact_line(car)
    where = np.zeros((h, w), np.float32)
    where[int(line.min()) - 5:, :] = 1
    out = vs.mirror_reflection(img, matte, xs.astype(float), line, 100.0, where)
    assert np.array_equal(out[car], img[car])  # never over the car
    assert np.array_equal(out[where == 0], img[where == 0])  # nor outside the render's ground
    below = out[int(line.max()) + 4:int(line.max()) + 20, xs[20]:xs[-20]].astype(int)
    assert (below[..., 1] - below[..., 0]).mean() > 10  # greener: the car's mirror image
    assert np.array_equal(out[int(line.max()) + 70:], img[int(line.max()) + 70:])  # it fades out


def _sea_shot(tmp_path, n=12, h=192, w=320):
    """A sea under a sky (horizon at row 80), the camera tilting down 1 px a frame, and a car coming towards it (its
    box grows from the middle)."""
    rng = np.random.default_rng(2)
    world = np.zeros((h + 40, w, 3), np.uint8)
    world[:100] = (190, 195, 200)
    world[100:] = (60, 75, 90)
    world[100:] += rng.integers(0, 25, (h - 60, w, 1)).astype(np.uint8)
    frames, cars = [], []
    for i in range(n):
        f = world[20 - i:20 - i + h].copy()
        c = np.zeros((h, w), bool)
        c[70 - 2 * i:150, 130 - 3 * i:190 + 3 * i] = True
        f[c] = (30, 40, 200)
        frames.append(f)
        cars.append(c)
    work = tmp_path / "sea.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(work)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    return work, frames, cars


def test_behind_an_oncoming_car_the_plate_is_the_real_world_not_an_invented_one(tmp_path):
    from pickups.tasks import remove
    work, frames, cars = _sea_shot(tmp_path)
    h, w = cars[0].shape
    gen = []
    for f, c in zip(frames, cars):
        g = f.copy()
        g[c] = (60, 75, 90)
        tree = c & (np.arange(h)[:, None] < 110) & ((np.arange(w)[None, :] // 3) % 2 == 0)
        g[tree] = (20, 20, 20)  # invented: bare branches over the sea and sky behind the car
        gen.append(g)
    bg = remove.Background((0, 0, w, h), gen, cars, [np.ones((h, w), bool)] * len(gen), False, 0.0, True, 0.2, "editor")
    still = frames[0].copy()
    still[cars[0] & (np.arange(h)[:, None] < 80)] = (190, 195, 200)  # the editor's still of the first frame: sky
    still[cars[0] & (np.arange(h)[:, None] >= 80)] = (70, 85, 100)  # and sea behind the car
    before = vs.invented(bg, work)
    info = vs.real_behind(bg, work, (w, h), still)
    after = vs.invented(bg, work)
    assert info["used"] and not before["ok"] and after["ok"]
    for i in (0, len(gen) - 1):
        top = cars[i] & (np.arange(h)[:, None] < 100)  # the sky behind the car's roof: no tree left
        assert bg.gen[i][top].mean() > 120


def test_the_painted_cars_contact_shadows_come_with_it_and_only_darken():
    h, w = 300, 480
    plate = np.full((h, w, 3), 120, np.uint8)
    car = car_mask(h, w, 240, 120, 0.8)
    painted = plate.copy()
    painted[car] = (60, 110, 50)
    xs, line = vs.contact_line(car)
    for x, y in zip(xs, line):  # a dark contact shadow under each tyre, painted on the road
        painted[int(y):int(y) + 4, x] = 50
    painted[200:210, 10:40] = 200  # something lighter elsewhere on the road: never a "shadow"
    c = vs.contact_shadow(painted, plate, car)
    assert c.max() <= 1.0 and c.min() >= 0.35
    under = [c[int(y) + 1, x] for x, y in zip(xs[10:-10:10], line[10:-10:10])]
    assert np.mean(under) < 0.8  # under its tyres: darker
    assert (c[:int(line.min()) - 20] == 1).all() and (c[200:210, 10:40] == 1).all()  # nowhere else


def test_on_a_wet_road_the_old_cars_spray_is_looked_for_too(shot, tmp_path, monkeypatch):
    store, sid, up, frames = shot
    (tmp_path / "jobs").mkdir()
    asked = []

    def masks(ctx, proxy, objs, run, labels=()):
        asked.append(tuple(labels))
        return [np.zeros_like(o) for o in objs]
    monkeypatch.setattr(common, "shadow_masks", masks)
    ctx = _Ctx(store, sid, {"car_engine": "proxy", "frame": 0, "points": [], "text": "the SUV", "ref_upload_id": up, "describe": "an olive wagon",
                            "scene": "It drives along a wet country road."}, tmp_path / "jobs")
    qa = recast.run(ctx)["details"]["_qa"]
    assert vs.SPRAY_LABELS in asked and qa["spray_removed"]
    assert all("mist" not in labels for labels in asked)  # the fog itself stays


def test_the_camera_path_follows_a_tilt_that_swings_back_and_drops_a_stray_reading():
    truth = [(0.0, float(min(i, 40) * 0.6 - max(0, i - 44) * 5)) for i in range(57)]  # down, then fast back up
    readings = list(truth)
    readings[4] = (63.0, 3.5)  # a stray reading
    readings[20] = None
    got = vs.camera_path(readings)
    assert np.abs(got - np.array(truth)).max() < 3.0
    assert got[-1][1] < -20  # the swing back is kept (a smooth curve lost it)


def test_the_editors_seam_along_the_red_area_is_smoothed_away():
    h, w = 120, 200
    still = np.full((h, w, 3), 150, np.uint8)
    marked = still.copy()
    marked[40:90, 60:140] = (255, 0, 0)
    still[40, 60:140] = 40  # the seam the editor left along the red area's top edge
    out = vs.without_seam(still, marked)
    assert np.abs(out.astype(int) - 150).max() < 25 and out[40, 100].mean() > 125
    assert np.array_equal(out[:20], still[:20])  # away from the edge: untouched


def test_behind_a_rolling_camera_the_real_fill_keeps_the_horizon_straight(tmp_path):
    from pickups.tasks import remove
    rng = np.random.default_rng(4)
    n, h, w = 12, 192, 320
    big = np.zeros((h * 2, w * 2, 3), np.uint8)
    big[:h] = (190, 195, 200)
    big[h:] = (60, 75, 90)
    big[h:] += rng.integers(0, 20, (h, w * 2, 1)).astype(np.uint8)
    frames, cars = [], []
    for i in range(n):
        th = np.radians(0.15 * i)  # the camera rolls and tilts
        c, s_ = np.cos(th), np.sin(th)
        A = np.array([[c, -s_, -w / 2], [s_, c, -h / 2 - 2 * i], [0, 0, 1.0]])
        f = np.clip(np.rint(composite.warp(big.astype(np.float32), np.linalg.inv(np.linalg.inv(A)), (w, h))), 0, 255).astype(np.uint8)
        car = np.zeros((h, w), bool)
        car[60 - 2 * i:170, 130 - 3 * i:190 + 3 * i] = True
        f[car] = (30, 40, 200)
        frames.append(f)
        cars.append(car)
    work = tmp_path / "roll.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(work)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    gen = [np.where(c[..., None], np.uint8(20), f) for f, c in zip(frames, cars)]  # a dark invented blob behind it
    bg = remove.Background((0, 0, w, h), gen, cars, [np.ones((h, w), bool)] * n, False, 0.0, True, 0.2, "editor")
    info = vs.real_behind(bg, work, (w, h))
    assert info["used"]
    last = bg.gen[-1]
    sky = cars[-1] & (np.arange(h)[:, None] < 40)
    assert last[sky].mean() > 120  # behind the car's roof: the real sky, not the blob


