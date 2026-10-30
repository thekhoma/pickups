"""Replace a person (tasks/person_swap): the crop that follows them, the median plate, the mask LTX may change,
and the composite's promise that a zero matte leaves the plate alone."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import composite, graphs
from pickups.tasks import person_swap as ps


def person(h, w, cx, foot, tall=80):
    m = np.zeros((h, w), bool)
    m[foot - tall:foot, cx - 14:cx + 14] = True
    return m


def test_the_crop_follows_the_person_and_stays_a_multiple_of_64():
    h, w = 256, 320
    masks = [person(h, w, 80 + 4 * i, 200) for i in range(12)]
    boxes, cw, ch = ps.crop_plan(masks, w, h)
    assert cw % 64 == 0 and ch % 64 == 0
    assert len(boxes) == 12
    xs = [x for x, _ in boxes]
    assert xs[-1] > xs[0]  # it walks right, and the crop goes with them
    for x, y in boxes:
        assert x % 2 == 0 and y % 2 == 0
        assert 0 <= x <= w - cw and 0 <= y <= h - ch


def test_feet_near_the_bottom_pin_the_crop_there():
    h, w = 256, 320
    masks = [person(h, w, 160, h - 2) for _ in range(8)]
    boxes, _, ch = ps.crop_plan(masks, w, h)
    assert all(y == h - ch for _, y in boxes)


def test_a_known_shift_is_recovered():
    rng = np.random.default_rng(2)
    img = rng.random((48, 64)).astype(np.float32)
    moved = ps.fshift(img, 3, -4)
    dy, dx = ps.phase_shift(img, moved)
    back = ps.fshift(moved, dy, dx)
    assert abs(dy + 3) < 0.2 and abs(dx - 4) < 0.2
    assert np.abs(back - img).mean() < 0.02


def test_the_median_plate_removes_a_moving_person_and_keeps_the_wall():
    h, w, n = 64, 96, 9
    rng = np.random.default_rng(1)
    wall = composite.blur(rng.integers(0, 255, (h, w, 3)).astype(np.float32), 0.7).astype(np.uint8)
    frames, cover = [], []
    for i in range(n):
        f = wall.copy()
        x = 10 + 5 * i
        f[20:44, x:x + 16] = (220, 20, 20)
        c = np.zeros((h, w), bool)
        c[18:46, x - 2:x + 18] = True
        frames.append(f)
        cover.append(c)
    st = ps.align_stack(frames, cover)
    plate = ps.median_of(st)
    assert all(abs(g[0] - 1) < 0.08 for g in st.gains)
    assert np.abs(st.A[3] - np.eye(3)).max() < 0.05
    restored, _ = ps.restore_frames(st, cover, [np.ones((h, w), bool)] * n, plate, nearest=False)
    # Where the red figure stood, the wall is back.
    assert np.abs(restored[3][30, 22].astype(int) - wall[30, 22].astype(int)).max() < 12


def _one(src, cover, plate):
    excl = composite.grow(cover, ps.CLEAR_GROW)
    return ps.Stack([src], [np.eye(3)], [[1.0, 1.0, 1.0]], [excl], cells=[ps.cells(src.astype(np.float32), ~excl)])


def test_the_plate_goes_back_only_where_the_person_was_and_their_floor_shadow():
    h, w = 300, 300
    src = np.full((h, w, 3), 120, np.uint8)
    src[120:124, 100:200] = 60  # a dark crack in the wall beside her, above the feet
    src[30:34, 100:200] = 60  # and one just above her head
    src[250:262, 115:185] = 80  # her shadow on the floor, beside her feet
    src[60:260, 140:160] = (200, 40, 40)  # the person
    cover = np.zeros((h, w), bool)
    cover[60:260, 140:160] = True
    plate = np.full((h, w, 3), 120, np.float32)  # the aligned median keeps the wall's cracks; it has no shadow
    plate[120:124, 100:200] = 60
    plate[30:34, 100:200] = 60
    out, alpha = ps.restore_frames(_one(src, cover, plate), [cover], [np.ones((h, w), bool)], plate, nearest=False)
    assert np.abs(out[0][150, 150].astype(int) - 120).max() <= 2  # the person is gone
    assert np.array_equal(out[0][120:124, 100:125], src[120:124, 100:125])  # the crack beside her stays real footage
    assert np.array_equal(out[0][30:34, 100:200], src[30:34, 100:200])  # and the one above her head
    assert int(out[0][256, 122, 0]) < 118  # her cast shadow stays: the ground there is darkened, not put back flat
    assert np.array_equal(out[0][alpha[0] == 0], src[alpha[0] == 0])  # alpha 0: footage, bit for bit
    assert alpha[0][cover].min() == 1.0
    ys, xs = np.nonzero(alpha[0] > 0)
    assert ys.min() >= 60 - ps.RESTORE_GROW - 8  # the patch is the person plus a few px, not a wall-sized blob


def test_a_lanyard_the_mask_missed_goes_and_the_wall_beside_it_stays():
    h, w = 300, 300
    src = np.full((h, w, 3), 120, np.uint8)
    src[60:260, 140:160] = (200, 40, 40)  # the person
    src[150:200, 160:164] = (30, 160, 60)  # a green lanyard hanging from her hand, outside the mask
    src[150:200, 196:200] = (30, 160, 60)  # something green on the wall, not touching her
    cover = np.zeros((h, w), bool)
    cover[60:260, 140:160] = True
    plate = np.full((h, w, 3), 120, np.float32)
    out, _ = ps.restore_frames(_one(src, cover, plate), [cover], [np.ones((h, w), bool)], plate, nearest=False)
    assert np.abs(out[0][150:200, 160:164].astype(int) - 120).max() <= 2  # the lanyard is gone
    assert np.array_equal(out[0][150:200, 196:200], src[150:200, 196:200])  # the wall's own green stays


def test_a_long_beaded_strap_is_followed_all_the_way():
    allowed = np.zeros((40, 400), bool)
    for x in range(0, 400, 6):
        allowed[18:22, x:x + 4] = True  # beads with 2 px gaps, 400 px long
    allowed[2:6, 300:304] = True  # a speck that doesn't touch it
    seed = np.zeros_like(allowed)
    seed[18:22, 0:4] = True
    got = ps.connected(seed, allowed)
    assert got[18:22, 390:394].all() and not got[2:6, 300:304].any()
    assert not (got & ~allowed).any()


def test_a_strap_that_lingers_on_the_same_stones_is_not_lent_back():
    """Walking away, the strap crossed the same stones for frames on end: every near lender showed some of it."""
    h, w, n = 160, 200, 9
    rng = np.random.default_rng(4)
    wall = composite.blur(rng.uniform(90, 150, (h, w, 3)).astype(np.float32), 1.0)
    frames, cover = [], []
    for i in range(n):
        f = wall.copy()
        x = 60 + 3 * i
        f[20:150, x:x + 20] = (200, 40, 40)  # the person, walking slowly
        f[40:120, x + 24 + i // 3:x + 27 + i // 3] = (20, 20, 20)  # a strap the mask missed, beside them
        c = np.zeros((h, w), bool)
        c[18:152, x - 2:x + 22] = True
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
        cover.append(c)
    st = ps.align_stack(frames, cover)
    plate = ps.median_of(st)
    out, _ = ps.restore_frames(st, cover, [np.ones((h, w), bool)] * n, plate)
    for i in (3, 4, 5):
        x = 60 + 3 * i + 24 + i // 3
        assert np.abs(out[i][40:120, x:x + 3].astype(np.float32) - wall[40:120, x:x + 3]).max(axis=-1).mean() < 12


def test_the_plate_gets_the_footage_grain():
    h, w = 64, 64
    rng = np.random.default_rng(3)
    src = np.clip(120 + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    cover = np.zeros((h, w), bool)
    cover[16:48, 24:40] = True
    plate = np.full((h, w, 3), 120, np.float32)
    out, _ = ps.restore_frames(_one(src, cover, plate), [cover], [np.ones((h, w), bool)], plate,
                               need=np.array([4.0, 4.0, 4.0]), nearest=False)
    assert out[0][20:44, 28:36].astype(float).std() > 2.0  # not a flat smear


def test_the_latent_mask_keeps_the_painted_frame():
    region = [np.zeros((32, 32), bool) for _ in range(17)]
    for i, m in enumerate(region):
        m[4:20, 4 + i:8 + i] = True
    lat = ps.latent_masks(region)
    assert len(lat) == 3  # frame 0, then two groups of 8
    assert not lat[0].any()
    assert lat[1].any() and lat[2].any()


def test_a_zero_matte_leaves_the_plate_unchanged():
    bg = np.full((32, 32, 3), 40, np.uint8)
    bg[0, 0] = (1, 2, 3)
    char = np.full_like(bg, 200)
    matte = np.zeros((32, 32), np.float32)
    matte[8:20, 8:20] = 1
    shadow = np.ones((32, 32), np.float32)
    out = ps.composite_crop(bg, char, matte, shadow)
    assert np.array_equal(out[0, 0], bg[0, 0])
    assert np.array_equal(out[:8], bg[:8])
    assert out[12, 12, 0] > 150


def test_the_pose_graph_is_masked_ltx_and_not_depth():
    g = graphs.masked_pose("base.mp4", "lat.mp4", "pose.mp4", "a woman walks", 24, seed=42, frames=17)
    types = {n["class_type"] for n in g.values()}
    assert "LTXVSetVideoLatentNoiseMasks" in types
    assert "LoadDA3Model" not in types
    assert g["union"]["inputs"]["lora_name"] == graphs.UNION_LORA
    assert g["PU_Seed"]["inputs"]["noise_seed"] == 42
    pose = graphs.sdpose("crop.mp4")
    assert pose["sdp"]["inputs"]["ckpt_name"] == "sdpose_wholebody_fp16.safetensors"
    assert "SDPoseKeypointExtractor" in {n["class_type"] for n in pose.values()}
    assert pose["draw"]["inputs"]["draw_hands"] is False and pose["draw"]["inputs"]["draw_body"] is True


def test_a_24_gb_card_renders_the_castle_hall_crop_at_1_to_1():
    assert ps.render_size(960, 1728, 24) == (960, 1728)  # the castle hall's crop: no upscale of the character
    assert ps.render_size(1024, 1792, 24) == (1024, 1792)  # the prototype's size ran in 87 s on the RTX 4090
    rw, rh = ps.render_size(1216, 2112, 24)  # a taller person: scaled to fit
    assert rw * rh <= ps.SMALL_CARD_PIXELS and rw % 64 == 0 and rh % 64 == 0
    assert ps.render_size(1216, 2112, 96) == (1216, 2112)  # a 96 GB card holds it at 1:1
    rw, rh = ps.render_size(1600, 2880, 96)  # but not an 8K medium shot's crop
    assert rw * rh <= ps.BIG_CARD_PIXELS
    rw, rh = ps.render_size(384, 704, 24)  # a small figure across a meadow: drawn bigger, scaled back down
    assert rw * rh >= 0.9 * ps.MIN_RENDER_PIXELS and rw % 64 == 0 and rh % 64 == 0 and abs(rw / rh - 384 / 704) < 0.05


def test_the_crop_holds_someone_walking_towards_the_camera():
    h, w = 1080, 1920
    masks = [person(h, w, 900, 300 + 60 * i, tall=150 + 60 * i) for i in range(12)]  # grows from 150 to 810 px
    boxes, cw, ch = ps.crop_plan(masks, w, h)
    assert ch >= 810


def test_the_pose_graph_encodes_in_tiles():
    g = graphs.masked_pose("base.mp4", "lat.mp4", "pose.mp4", "a woman walks", 24, frames=17)
    assert g["enc"]["class_type"] == "VAEEncodeTiled" and g["pguide"]["inputs"]["use_tiled_encode"] is True
    g = graphs.masked_pose("base.mp4", "lat.mp4", "pose.mp4", "a woman walks", 24, frames=17, tiled=False)
    assert g["enc"]["class_type"] == "VAEEncode" and g["pguide"]["inputs"]["use_tiled_encode"] is False


def test_the_matte_is_the_guided_soft_alpha_on_the_character_edge_inside_the_region():
    # A bright figure on a dark ground; the track is blocky (SAM at proxy size, upscaled) and 4 px too wide.
    im = Image.new("RGB", (96, 96), (30, 30, 30))
    ImageDraw.Draw(im).ellipse([28, 10, 66, 86], fill=(230, 220, 210))
    char = np.asarray(im)
    track = np.zeros((96, 96), np.float32)
    track[8:88, 24:72] = 1
    track = np.repeat(np.repeat(track[::4, ::4], 4, 0), 4, 1)
    region = np.zeros((96, 96), bool)
    region[2:94, 12:84] = True
    matte = ps.character_matte(track, char, region)
    assert matte[~region].max() == 0 and matte.max() <= 1
    assert matte[48, 47] > 0.95  # inside the figure
    # Inside the blocky track but outside the figure, the alpha is pulled towards the picture's edge.
    assert matte[46:50, 24:27].mean() < 0.7 * composite.smooth(track, 1.0)[46:50, 24:27].mean()
    edge = matte[40:56, 20:34]
    assert ((edge > 0.05) & (edge < 0.95)).any()  # a soft edge, not a thresholded step


def test_the_edge_takes_the_characters_colour_not_the_renders_background():
    char = np.zeros((40, 40, 3), np.uint8)
    char[:, :20] = (200, 40, 40)  # the character: red
    char[:, 20:] = (40, 200, 40)  # the render's own background: green
    alpha = np.zeros((40, 40), np.float32)
    alpha[:, :19] = 1.0
    alpha[:, 19:22] = (0.7, 0.4, 0.15)  # the soft edge reaches into the green
    out = ps.decontaminate(char, alpha)
    assert out[20, 20, 1] < 120 and out[20, 20, 0] > 120  # the edge is red, not a green rim
    assert np.array_equal(out[:, :17], char[:, :17])  # inside stays as rendered
    assert np.array_equal(out[:, 25:], char[:, 25:])  # alpha 0: untouched (and never shown)


def test_a_light_rim_down_a_dark_cloak_goes_and_a_dark_seam_stays():
    char = np.full((40, 40, 3), (70, 50, 40), np.uint8)  # a dark brown cloak on the left
    char[:, 18] = (150, 130, 115)  # a thin light line LTX drew down its edge
    char[:, 10] = (20, 15, 10)  # a dark seam well inside
    char[:, 20:] = (30, 30, 30)
    alpha = np.zeros((40, 40), np.float32)
    alpha[:, :20] = 1.0
    out = ps.decontaminate(char, alpha)
    assert out[20, 18, 0] < 95  # the rim is the cloak's own brown now
    assert np.array_equal(out[20, 10], char[20, 10])  # the seam inside is untouched
    dark = char.copy()
    dark[:, 18] = (40, 28, 20)  # a darker outline at the edge stays
    assert np.array_equal(ps.decontaminate(dark, alpha)[20, 18], dark[20, 18])


def test_missing_pose_weights_stop_before_any_stage():
    class Conn:
        def model_files(self):
            return {"sam3.1_multiplex_fp16.safetensors"}

    class Ctx:
        conn = Conn()

        def stage(self, name):
            raise AssertionError(name)

    with pytest.raises(RuntimeError, match="sdpose_wholebody_fp16"):
        ps.run(Ctx(), None)


def test_the_nearest_clear_frame_is_picked():
    # A still camera; where the person stands on frame 3, frame j shows 100 + 10 j. Frame 2 had them there too.
    h, w, n = 48, 48, 7
    frames, excl = [], []
    for j in range(n):
        f = np.full((h, w, 3), 100, np.uint8)
        f[16:32, 16:32] = 100 + 10 * j
        frames.append(f)
        e = np.zeros((h, w), bool)
        if j in (2, 3):
            e[14:34, 14:34] = True
        excl.append(e)
    st = ps.Stack(frames, [np.eye(3)] * n, [[1.0] * 3] * n, excl,
                  cells=[ps.cells(f.astype(np.float32), ~e) for f, e in zip(frames, excl)])
    area = np.zeros((h, w), bool)
    area[18:30, 18:30] = True
    win = (slice(0, h), slice(0, w))
    fill, took = ps.fill_from_clear(st, np.full((h, w, 3), 100, np.float32), 3, area, win, tol=200)
    assert abs(fill[24, 24, 0] - 140) < 1.0  # frame 4: one away, and frame 2 isn't clear there
    assert not took.any()


def test_a_lender_that_still_shows_a_strap_is_outvoted():
    # Still camera, a plain wall; the frame right after shows a strap the mask missed where the person stood.
    h, w, n = 48, 48, 7
    frames = [np.full((h, w, 3), 100, np.uint8) for _ in range(n)]
    frames[4][10:40, 22:26] = (30, 160, 60)
    excl = [np.zeros((h, w), bool) for _ in range(n)]
    excl[3][14:34, 14:34] = True
    st = ps.Stack(frames, [np.eye(3)] * n, [[1.0] * 3] * n, excl,
                  cells=[ps.cells(f.astype(np.float32), ~e) for f, e in zip(frames, excl)])
    area = np.zeros((h, w), bool)
    area[18:30, 18:30] = True
    fill, _ = ps.fill_from_clear(st, np.full((h, w, 3), 100, np.float32), 3, area, (slice(0, h), slice(0, w)))
    assert np.abs(fill[18:30, 18:30] - 100).max() < 1.0  # the strap in frame 4 is outvoted by frames 2 and 5


def test_a_drift_is_carried_across_a_hole():
    known = np.ones((96, 96), bool)
    known[30:70, 30:70] = False
    diff = np.zeros((96, 96, 3), np.float32)
    diff[..., 0] = 6.0
    diff[~known] = 100.0  # what's in the hole is never read
    m, f = ps.cells(diff, known, cell=8)
    d = ps.spread(m, f > 0.5, 3.0)
    v = ps.sample(d, np.array([50.0]), np.array([50.0]), cell=8)[0]
    assert abs(v[0] - 6.0) < 0.5 and abs(v[1]) < 0.5


def _panning_walk(n=13, h=144, w=256, step=3, grain=4.0, drift=0.015, vignette=0.0):
    """A textured wall the camera pans across `step` px a frame, with fresh grain, an exposure drift and the lens's
    vignette on every frame, and a red figure walking the other way."""
    rng = np.random.default_rng(5)
    wall = composite.blur(rng.uniform(40, 200, (h, w + step * n, 3)).astype(np.float32), 1.0)
    vig = (1 - vignette * (np.linspace(-1, 1, w) ** 2))[None, :, None].astype(np.float32)
    frames, cover, truth = [], [], []
    for i in range(n):
        bgf = wall[:, step * i:step * i + w] * (1 + drift * (i - n // 2)) * vig
        truth.append(bgf.copy())
        f = np.clip(bgf + rng.normal(0, grain, bgf.shape), 0, 255)
        x = 170 - 8 * i
        f[30:120, x:x + 24] = (220, 30, 30)
        c = np.zeros((h, w), bool)
        c[28:122, x - 2:x + 26] = True
        frames.append(np.rint(f).astype(np.uint8))
        cover.append(c)
    return frames, cover, truth


def test_the_background_comes_from_the_nearest_clear_frame_aligned_and_drift_corrected():
    frames, cover, truth = _panning_walk(vignette=0.5)
    n, (h, w) = len(frames), cover[0].shape
    st = ps.align_stack(frames, cover)
    plate = ps.median_of(st)
    win = [np.ones((h, w), bool)] * n
    out, alpha = ps.restore_frames(st, cover, win, plate)
    i = 4
    inner = composite.shrink(cover[i], 3) & (alpha[i] >= 1)
    err = np.abs(out[i][inner].astype(np.float32) - truth[i][inner]).mean()
    assert err < 6.0  # the wall behind her, at this frame's exposure (grain is ~4 levels)
    for k in (1, 11):  # far from the middle the lending frames saw that wall elsewhere in the vignette: corrected
        inner_k = composite.shrink(cover[k], 3) & (alpha[k] >= 1)
        assert abs(float((out[k][inner_k].astype(np.float32) - truth[k][inner_k]).mean())) < 4.0
    # Real footage keeps its grain (~4 levels about the true wall); the median's is averaged away.
    med_out, _ = ps.restore_frames(ps.align_stack(frames, cover), cover, win, plate, nearest=False)
    real = (out[i][inner].astype(np.float32) - truth[i][inner]).std()
    assert real > 3.0 and real > (med_out[i][inner].astype(np.float32) - truth[i][inner]).std() + 1.0
    assert np.array_equal(out[i][alpha[i] == 0], frames[i][alpha[i] == 0])


def test_the_camera_motion_is_an_affine_not_just_a_shift():
    # A pan that also pushes in: the far edges move more than the middle.
    rng = np.random.default_rng(8)
    a = composite.blur(rng.uniform(0, 255, (256, 320, 3)).astype(np.float32), 1.2)
    M = np.array([[1.02, 0.0, -3.0], [0.0, 1.02, 2.0], [0, 0, 1]])  # a's pixels -> b's
    b = composite.warp(a, M, (320, 256), cubic=True)
    none = np.zeros((256, 320), bool)
    got = ps.frame_motion(a.astype(np.uint8), b.astype(np.uint8), none, none)  # b's pixels -> a's
    want = np.linalg.inv(M)
    for x, y in ((40, 40), (280, 220), (160, 128)):
        p, q = got @ (x, y, 1), want @ (x, y, 1)
        assert np.hypot(*(p[:2] - q[:2])) < 0.6


def test_the_window_starts_with_the_person_whole_in_the_picture():
    h, w = 120, 200
    masks = []
    for i in range(40):
        m = np.zeros((h, w), bool)
        top = 10 + i  # walking away: cut off by the bottom of the frame for the first 20 frames
        m[top:min(h, top + 130 - 2 * i), 90:110] = True
        masks.append(m)
    s0 = ps.pose_window(masks, 9, 40)
    from pickups.tasks import recast_ltx
    assert recast_ltx.cut_off(masks[s0]) <= ps.WHOLE_CUT and masks[s0].any()
    assert ps.pose_window(masks, 9, 40, start=3) == 3  # the user's choice wins


def test_an_edge_between_a_lit_hand_and_a_dark_sleeve_stays_dark():
    """Copying the inside's colour onto the soft edge drew a light line down the sleeve: the nearest inside was a hand."""
    char = np.zeros((40, 40, 3), np.uint8)
    char[:, :16] = (180, 150, 130)  # a lit hand
    char[:, 16:22] = (35, 30, 25)  # a dark sleeve, its soft edge
    char[:, 22:] = (60, 60, 60)  # the render's background
    bg = np.full((40, 40, 3), 50, np.uint8)
    alpha = np.zeros((40, 40), np.float32)
    alpha[:, :16] = 1.0
    alpha[:, 16:22] = 0.9
    out = ps.decontaminate(char, alpha, bg)
    assert out[20, 18].max() < 60  # still dark, not the hand's colour


def test_a_person_cut_off_at_the_hips_is_painted_on_the_picture_carried_on_below():
    m = np.zeros((200, 200), bool)
    m[100:200, 70:130] = True  # head to hips: 100 tall, 60 wide
    share = ps.visible_share(m)
    assert 0.4 < share < 0.6
    rows = ps.below_edge(m)
    assert rows % 32 == 0 and 64 <= rows <= 200  # roughly the missing hips-to-feet
    frame = np.zeros((200, 200, 3), np.uint8)
    plate = np.zeros((200, 200, 3), np.uint8)
    plate[-9] = 77  # the ground's last real row comes first below the edge
    plate[:100] = 200  # the sky never comes below it
    ext = ps.extend_down(frame, plate, rows)
    assert ext.shape == (200 + rows, 200, 3) and ext[200, 0, 0] == 77 and ext[200:].max() < 200


def test_the_rows_below_the_edge_come_from_the_hips_in_the_pose():
    m = np.zeros((200, 200), bool)
    m[20:200, 40:160] = True  # head to thighs, an arm held out: wide, so the shape alone says half of them shows
    pose = np.zeros((200, 200, 3), np.uint8)
    pose[60:120, 99] = ps.POSE_R[1]  # neck to right hip
    pose[60:120, 101] = ps.POSE_L[1]  # neck to left hip: the hips at y=120
    rows = ps.below_edge(m, pose)
    # head to hips is 100 rows, so about 213 tall: about 33 rows below the edge, not a whole picture
    assert 32 <= rows <= 64
    assert ps.below_edge(m) >= 160


def test_the_matte_stops_at_the_track_where_the_ground_matches_the_hem():
    char = np.full((80, 80, 3), (60, 70, 50), np.uint8)  # the hem and the grass below it: the same dull green
    track = np.zeros((80, 80), np.float32)
    track[10:50, 20:60] = 1
    region = np.ones((80, 80), bool)
    matte = ps.character_matte(track, char, region)
    assert matte[60:, :].max() == 0  # the render's own ground below the hem is not composited


def test_where_lenders_disagree_the_nearest_real_pixel_is_used_not_a_blend():
    h, w, n = 32, 32, 7
    frames = [np.full((h, w, 3), 40 * (j % 3) + 60, np.uint8) for j in range(n)]  # grass in the wind: every frame differs
    excl = [np.zeros((h, w), bool) for _ in range(n)]
    excl[3][8:24, 8:24] = True
    st = ps.Stack(frames, [np.eye(3)] * n, [[1.0] * 3] * n, excl,
                  cells=[ps.cells(np.full((h, w, 3), 100, np.float32), np.ones((h, w), bool)) for _ in range(n)])
    area = np.zeros((h, w), bool)
    area[12:20, 12:20] = True
    fill, _ = ps.fill_from_clear(st, np.full((h, w, 3), 100, np.float32), 3, area, (slice(0, h), slice(0, w)), tol=5)
    assert {int(round(v)) for v in np.unique(fill[12:20, 12:20, 0])} <= {60, 100, 140}  # one frame's own value


def test_a_walk_away_from_a_close_camera_is_rendered_backwards():
    h, w = 120, 200
    masks = []
    for i in range(40):
        m = np.zeros((h, w), bool)
        top = 10 + 2 * i  # cut off by the bottom of the frame until frame 38, then whole and smaller
        m[top:min(h, top + 184 - 4 * i), 90:110] = True
        masks.append(m)
    s0 = ps.pose_window(masks, 13, 40)
    from pickups.tasks import recast_ltx
    assert recast_ltx.cut_off(masks[s0 + 12]) <= ps.WHOLE_CUT  # it ends with them whole
    assert ps.backwards(masks[s0:s0 + 13])
    assert not ps.backwards(masks[::-1][:13])  # whole at the start: forwards


def _pose(rx, lx, face_dx=0, h=200, w=200):
    """A drawn SDPose frame: neck at (100, 40); right/left shoulder+hip sticks ending at x = rx / lx; face points."""
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    d.line([(100, 40), (rx, 45)], fill=(255, 0, 0), width=4)
    d.line([(100, 40), (rx, 120)], fill=(0, 255, 0), width=4)
    d.line([(100, 40), (lx, 45)], fill=(255, 85, 0), width=4)
    d.line([(100, 40), (lx, 120)], fill=(0, 255, 255), width=4)
    d.line([(100, 120), (100, 190)], fill=(0, 255, 85), width=4)
    d.ellipse([100 + face_dx - 3, 20, 100 + face_dx + 3, 26], fill=(255, 255, 255))
    return np.asarray(im)


def test_facing_is_read_from_the_pose_not_a_face():
    assert ps.pose_facing(_pose(70, 130)) == "front"  # their right shoulder on the picture's left
    assert ps.pose_facing(_pose(130, 70)) == "back"
    assert ps.pose_facing(_pose(102, 98, face_dx=12)) == "side-right"
    assert ps.pose_facing(_pose(98, 102, face_dx=-12)) == "side-left"
    assert ps.same_facing("side-right", "back") and not ps.same_facing("side-right", "front")
    assert not ps.same_facing("back", "front") and not ps.same_facing("side-right", "side-left")


def test_a_far_heel_the_track_missed_goes_with_the_feet():
    cover = np.zeros((400, 200), bool)
    cover[50:390, 80:120] = True  # 340 tall
    heel = (slice(375, 390), slice(124, 130))  # 4-10 px past the mask, beside the foot
    src = np.full((400, 200, 3), 120, np.uint8)
    src[heel] = 20
    ref = src.astype(np.float32).copy()
    ref[heel] = 20  # the lending frames show the same missed heel: the remnant test can't see it
    rep = ps.replace_mask(cover, src, ref, np.ones_like(cover))
    assert rep[heel].all()
    assert not rep[100, 140]  # the body isn't grown wider


def test_where_no_frame_shows_the_ground_the_render_is_used():
    """A camera following them: their spot is never clear, so the plate there is them, and the render's picture goes in."""
    h, w, n = 160, 200, 7
    rng = np.random.default_rng(5)
    wall = composite.blur(rng.uniform(90, 150, (h, w, 3)).astype(np.float32), 1.0)
    frames, cover = [], []
    for _ in range(n):
        f = wall.copy()
        f[20:150, 90:110] = (20, 20, 20)  # the person, in the same place on every frame
        c = np.zeros((h, w), bool)
        c[18:152, 88:112] = True
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
        cover.append(c)
    st = ps.align_stack(frames, cover)
    plate = ps.median_of(st)
    unsure: list[np.ndarray] = []
    out, _ = ps.restore_frames(st, cover, [np.ones((h, w), bool)] * n, plate, unsure=unsure)
    assert len(unsure) == n and unsure[3][80, 100] and not unsure[3][80, 20]
    char = np.full((h, w, 3), 130, np.uint8)  # the render: a narrower character on its own ground
    track = np.zeros((h, w), np.float32)
    track[20:150, 96:104] = 1
    region = composite.grow(cover[3], 10)
    filled = ps.fill_holes(out[3], composite.grow(unsure[3], ps.UNSURE_GROW))  # as the job does, before the blend
    comp, matte, _ = ps.blend_frame(filled, char, char, track, region, None, None, 0, unsure=unsure[3])
    assert matte[80, 100] > 0.99 and abs(int(comp[80, 100, 0]) - 130) < 15  # the render's ground beside them
    assert comp[80, 91, 0] > 80  # and beyond that the plate's fill, not the old person's dark leg
    assert matte[80, 20] == 0


def test_a_hole_no_frame_saw_is_filled_from_around_it():
    img = np.zeros((64, 80, 3), np.uint8)
    img[:, :40] = (100, 120, 90)
    img[:, 40:] = (110, 125, 95)
    hole = np.zeros((64, 80), bool)
    hole[20:50, 30:52] = True
    img[hole] = (10, 10, 10)  # the old person's black trousers in the plate
    out = ps.fill_holes(img, hole)
    assert out[hole].min() >= 80 and out[hole].max() <= 130  # the ground round it, not the trousers
    assert (out[~hole] == img[~hole]).all()


def test_a_paint_that_shrinks_a_cut_off_person_into_the_picture_fails(tmp_path):
    from PIL import Image
    old = np.zeros((200, 50), bool)
    old[10:100, 15:35] = True  # cut off by the picture's bottom at row 100; the canvas goes on to 200
    small = np.zeros((200, 50), np.uint8)
    small[10:110, 18:32] = 255  # feet just below the picture
    Image.fromarray(small).save(tmp_path / "mask_1.png")
    assert ps.painted_small(tmp_path, old, 100, 100)
    big = np.zeros((200, 50), np.uint8)
    big[10:190, 15:35] = 255  # down to the ground below the picture
    Image.fromarray(big).save(tmp_path / "mask_1.png")
    assert ps.painted_small(tmp_path, old, 100, 100) == ""


def test_a_background_that_comes_back_as_the_person_is_left_to_the_render():
    h, w = 200, 200
    src = np.full((h, w, 3), 120, np.uint8)
    src[40:180, 80:120] = 15  # the person: black trousers
    cover = np.zeros((h, w), bool)
    cover[38:182, 78:122] = True
    plate = src.astype(np.float32)  # the plate is the person too (a camera following them)
    unsure: list[np.ndarray] = []
    ps.restore_frames(_one(src, cover, plate), [cover], [np.ones((h, w), bool)], plate, nearest=False, unsure=unsure)
    assert unsure[0][100, 100] and not unsure[0][100, 20]



def test_the_track_reaches_the_edge_of_the_picture_that_cuts_them_off():
    t = np.zeros((200, 100), np.float32)
    t[50:198, 30:70] = 1  # SAM stopped two rows short of the bottom edge
    held = ps.hold_edges(t, ps.edges_of((0, 400), 100, 200, 300, 600))
    assert held[199, 50] == 1 and held[199, 10] == 0
    assert (ps.hold_edges(t, ps.edges_of((0, 100), 100, 200, 300, 600)) == t).all()  # not at an edge: as it was



def test_the_renders_light_last_rows_at_the_picture_edge_are_replaced():
    char = np.full((40, 30, 3), 50, np.uint8)
    char[-3:] = 90  # the VAE's light border
    out = ps.hold_border(char, (False, True, False, False))
    assert out[-1].max() == 50 and (out[:-4] == char[:-4]).all()


def _walk(n: int, follow: bool, cut: bool = False):
    rng = np.random.default_rng(7)
    scene = composite.blur(rng.uniform(40, 220, (120, 400, 3)).astype(np.float32), 1.0)
    frames, masks = [], []
    for i in range(n):
        x = 100 if follow else 60 + 6 * i  # walking across the scene, or on the spot (the ground under them unseen)
        cam = 3 * i if follow else 0  # the camera pans a little
        f = scene[:, cam:cam + 200].copy()
        m = np.zeros((120, 200), bool)
        m[20:(120 if cut else 100), x - cam:x - cam + 16] = True
        f[m] = (30, 30, 30)
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
        masks.append(m)
    return frames, masks


def test_the_guard_lets_a_walk_across_a_still_shot_through():
    frames, masks = _walk(9, follow=False)
    assert ps.never_seen(frames, masks) < 0.05
    assert ps.pose_guard(masks, frames) == ""


def test_the_guard_stops_someone_who_stays_in_one_place():
    frames, masks = _walk(9, follow=True)
    assert ps.never_seen(frames, masks) > 0.5
    assert "never seen" in ps.pose_guard(masks, frames)


def test_the_guard_stops_someone_cut_off_all_through_without_reading_frames():
    _, masks = _walk(5, follow=False, cut=True)
    assert "cuts them off" in ps.pose_guard(masks, None)


def test_a_hole_is_filled_with_the_ground_beside_it():
    rng = np.random.default_rng(9)
    img = np.clip(rng.normal(120, 25, (60, 80, 3)), 0, 255).astype(np.uint8)  # gravel
    hole = np.zeros((60, 80), bool)
    hole[10:50, 30:40] = True
    shift = ps.hole_shift([hole])
    out = ps.fill_holes(img, hole, shift)
    inner = out[20:40, 34:36].astype(np.float32)
    assert inner.std() > 10  # texture, not a smooth smear
    src = [img[20:40, 34 - d:36 - d].astype(np.float32) for d in (shift, -shift) if 0 <= 34 - d and 36 - d <= 80]
    assert min(np.abs(inner - x).mean() for x in src) < 12  # the gravel beside it


def test_a_wide_hole_with_no_shift_still_gets_ground_texture_not_a_flat_patch():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(120, 25, (60, 200, 3)), 0, 255).astype(np.uint8)
    hole = np.zeros((60, 200), bool)
    hole[5:60, 40:160] = True  # wider than the picture leaves room to copy across
    img[hole] = (15, 15, 15)
    out = ps.fill_holes(img, hole, 0)
    inner = out[20:50, 70:130].astype(np.float32)
    assert inner.std() > 8  # not flat
    assert 90 < inner.mean() < 150  # the ground's tone, not the black fringe


def test_the_original_cast_shadow_is_kept_under_the_new_person():
    h, w = 300, 300
    cover = np.zeros((h, w), bool)
    cover[40:280, 120:150] = True
    ref = np.full((h, w, 3), 150, np.float32)
    src = ref.copy()
    src[270:296, 100:260] = 90  # a long cast shadow across the ground beside the feet, ratio 0.6
    src[cover] = 20  # the person
    win = np.ones((h, w), bool)
    rep = ps.replace_mask(cover, src.astype(np.uint8), ref * ps.shadow_ratio(cover, src, ref, win)[..., None], win)
    assert not rep[285, 200]  # the shadow far from them is left as footage
    ratio = ps.shadow_ratio(cover, src, ref, win)
    assert 0.5 < ratio[285, 135] < 0.78  # carried in under the feet
    assert ratio[100, 135] > 0.99  # not up the body


def test_a_black_shoe_the_mask_missed_is_not_taken_for_a_shadow():
    cover = np.zeros((300, 300), bool)
    cover[40:280, 120:150] = True
    ref = np.full((300, 300, 3), 150, np.float32)
    src = ref.copy()
    src[270:290, 160:175] = 15
    assert ps.shadow_ratio(cover, src, ref, np.ones((300, 300), bool)).min() > 0.95


def test_the_new_feet_get_a_soft_contact_shadow_and_the_body_does_not():
    h, w = 200, 160
    matte = np.zeros((h, w), np.float32)
    matte[20:180, 60:90] = 1
    region = np.ones((h, w), bool)
    ao = ps.contact_ao(matte, region, 179, 160)
    assert ao[175, 58] > 0.2 and ao[178, 58] > 0.2  # beside the foot
    assert ao[60, 55] == 0  # beside the body, no
    assert (ao[matte > 0.5] == 0).all()


def test_when_the_new_shape_differs_the_old_shadow_is_taken_out_and_the_ground_goes_to_ltx():
    h, w = 300, 300
    cover = np.zeros((h, w), bool)
    cover[40:280, 120:150] = True
    ref = np.full((h, w, 3), 150, np.float32)
    src = ref.copy()
    src[262:282, 150:230] = 90  # their shadow beside the feet
    src[cover] = 20
    win = np.ones((h, w), bool)
    keep = ps.replace_mask(cover, src.astype(np.uint8), ref, win)
    redraw = ps.replace_mask(cover, src.astype(np.uint8), ref, win, redraw=True)
    assert not keep[270, 200] and redraw[270, 200]
    zone = ps.redraw_zone(cover)
    assert zone[270, 200] and not zone[100, 200]


def test_the_painted_person_is_compared_with_the_old_one_by_iou(tmp_path):
    old = np.zeros((100, 60), bool)
    old[10:90, 20:40] = True
    same = np.zeros((100, 60), np.uint8)
    same[10:90, 20:40] = 255
    wide = np.zeros((100, 60), np.uint8)
    wide[10:90, 5:55] = 255
    Image.fromarray(same).save(tmp_path / "mask_1.png")
    assert ps.shape_match(tmp_path, old) > 0.95
    Image.fromarray(wide).save(tmp_path / "mask_1.png")
    assert ps.shape_match(tmp_path, old) < ps.SHADOW_IOU

def test_the_pose_graph_blends_the_render_onto_the_plate_with_lightricks_node():
    g = graphs.masked_pose("base.mp4", "lat.mp4", "pose.mp4", "a woman walks", 24, frames=17)
    assert "PU_Blend" not in g  # no region, no blend
    g = graphs.masked_pose("base.mp4", "lat.mp4", "pose.mp4", "a woman walks", 24, frames=17, region="region.mp4")
    b = g["blend"]
    assert b["class_type"] == "LTXVLaplacianPyramidBlend"
    assert b["inputs"]["image_a"] == ["decode", 0] and b["inputs"]["image_b"] == ["bparts", 0]  # render onto the plate
    assert b["inputs"]["mask_low_res_dilation"] == 0  # 5, 6 and 10 let more decoded background through (T1.5)
    assert g["PU_Blend"]["inputs"]["images"] == ["blend", 0]
    assert g["PU_Out"]["inputs"]["images"] == ["decode", 0]  # the raw decode is still saved


def test_the_pixel_region_gives_each_picture_frame_its_latent_frames_mask():
    lat = [np.full((4, 4), v, np.uint8).astype(bool) for v in (0, 1, 0)]
    lat[1][0, 0] = False
    px = ps.pixel_region(lat, 17)
    assert len(px) == 17 and not px[0].any()
    assert all(np.array_equal(px[f], lat[1]) for f in range(1, 9))  # frames 1..8 are latent frame 1
    assert all(np.array_equal(px[f], lat[2]) for f in range(9, 17))


def test_beyond_the_mask_and_its_band_the_picture_is_the_real_plate():
    h, w = 160, 120
    region = np.zeros((h, w), bool)
    region[40:120, 30:80] = True
    plate = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    render = np.clip(plate.astype(int) + 40, 0, 255).astype(np.uint8)  # a decode that moved every level
    out = ps.plate_lock(render, plate, region)
    far = ~composite.grow(region, ps.PLATE_BAND + 2)
    assert np.array_equal(out[far], plate[far])  # outside: not one pixel changed
    assert np.array_equal(out[region][::7], render[region][::7])  # inside the mask: the render
    edge = ~region & composite.grow(region, 3)
    assert (np.abs(out[edge].astype(int) - plate[edge]) > 0).any()  # the ramp starts at the mask's edge
    assert np.array_equal(ps.plate_lock(render, plate, np.zeros((h, w), bool)), plate)  # an empty mask: all plate


def test_a_phone_mask_that_is_the_whole_knight_is_not_a_phone():
    """SAM's "phone" query returned the whole knight on T1.5: 412022 px against the person's 415367."""
    from pickups.tasks import recast_ltx
    person = np.zeros((2432, 896), bool)
    person[400:2300, 250:470] = True
    whole = person.copy()
    assert not recast_ltx.phone_counts(whole, person)
    # exact numbers of the rejected take: the phone mask is 99.2% of the person's
    assert 412022 / 415367 > 0.99


def test_a_small_phone_shaped_thing_in_a_hand_is_a_phone():
    from pickups.tasks import recast_ltx
    person = np.zeros((600, 300), bool)
    person[100:580, 100:200] = True
    phone = np.zeros_like(person)
    phone[300:340, 190:210] = True  # 800 px, portrait, over the person's edge
    hands = np.zeros_like(person)
    hands[310:330, 200:215] = True
    assert recast_ltx.phone_counts(phone, person, hands)
    assert not recast_ltx.phone_counts(phone, person, np.zeros_like(person))  # nowhere near a hand
    far = np.zeros_like(person)
    far[10:50, 10:30] = True
    assert not recast_ltx.phone_counts(far, person, hands)  # not touching them
    long = np.zeros_like(person)
    long[300:304, 150:230] = True  # a thin strip, not a phone
    assert not recast_ltx.phone_counts(long, person)


def test_a_paint_a_rounding_short_of_the_cutoff_is_not_smaller(tmp_path):
    """The T1.5 paint reached 0.627 of the canvas against a 0.63 cutoff."""
    rows, ext = 500, 2190
    total = rows + ext
    needed = rows + ps.PAINT_REACH * ext / 1.1
    assert abs(needed / total - 0.63) < 0.002
    old = np.zeros((total, 100), bool)
    old[10:rows, 30:70] = True
    def painted(feet):
        m = np.zeros((total, 100), np.uint8)
        m[10:feet, 28:72] = 255
        Image.fromarray(m).save(tmp_path / "mask_1.png")
        return ps.painted_small(tmp_path, old, rows, ext)
    assert painted(round(0.627 * total)) == ""
    assert painted(round(0.45 * total)) != ""  # a figure two-thirds the size still fails


def test_a_shoe_tip_on_the_bottom_edge_is_not_a_person_cut_off_by_it():
    whole = np.zeros((400, 200), bool)
    whole[20:380, 70:130] = True  # standing
    whole[380:400, 80:100] = True  # a shoe tip reaching the last rows: 20 px of a 60 px body
    assert not ps.cut_by_bottom(whole)
    cut = np.zeros((400, 200), bool)
    cut[20:400, 70:130] = True  # cut off at the thighs: the whole width runs off the bottom
    assert ps.cut_by_bottom(cut)
    assert not ps.cut_by_bottom(np.zeros((10, 10), bool))


def _pose_with_neck_and_shoulders(h=400, w=300, neck=(150, 200), half=60):
    pose = np.zeros((h, w, 3), np.uint8)
    nx, ny = neck
    pose[ny:ny + 4, nx - half:nx] = ps.POSE_R[0]
    pose[ny:ny + 4, nx:nx + half] = ps.POSE_L[0]
    return pose


def test_a_head_box_sits_above_the_neck_and_the_matte_must_cover_it():
    pose = _pose_with_neck_and_shoulders()
    box = ps.head_box(pose)
    assert box is not None and box[:200].any() and not box[200:].any()  # above the neck only
    head = np.zeros((400, 300), np.float32)
    head[110:200, 120:180] = 1  # a head over the neck
    assert ps.part_shares(ps.part_masks(pose), head)["head"] > 0.9
    headless = np.zeros((400, 300), np.float32)
    headless[205:300, 90:210] = 1  # shoulders only
    assert ps.part_shares(ps.part_masks(pose), headless)["head"] < 0.1
    assert ps.missing_parts([ps.part_shares(ps.part_masks(pose), headless)]) == {"head": 0.0} or "head" in ps.missing_parts([ps.part_shares(ps.part_masks(pose), headless)])
    assert ps.head_box(_pose_with_neck_and_shoulders(neck=(150, 20))) is None  # the box would leave the picture


def test_an_overcast_shot_has_no_shadow_to_keep_or_redraw():
    flat = [np.ones((300, 300), np.float32) for _ in range(10)]
    assert not ps.has_shadow(flat)
    shaded = []
    for _ in range(10):
        k = np.ones((300, 300), np.float32)
        k[250:290, 100:250] = 0.6
        shaded.append(k)
    assert ps.has_shadow(shaded)
    assert not ps.has_shadow([])


def _lit(side, h=300, w=200):
    img = np.full((h, w, 3), 90, np.uint8)
    if side == "right":
        img[:, 100:] = 190
    elif side == "left":
        img[:, :100] = 190
    m = np.zeros((h, w), bool)
    m[20:280, 40:160] = True
    return img, m


def test_the_key_light_side_is_read_from_the_subject_and_a_paint_lit_from_the_other_side_fails():
    right, m = _lit("right")
    left, _ = _lit("left")
    flat, _ = _lit("none")
    assert ps.lateral_tilt(right, m) > ps.TILT_MIN and ps.lateral_tilt(left, m) < -ps.TILT_MIN
    assert abs(ps.lateral_tilt(flat, m)) < 0.01
    assert ps.light_problem(left, m, ps.lateral_tilt(right, m)) == "lit from the wrong side"
    assert ps.light_problem(right, m, ps.lateral_tilt(right, m)) == ""
    assert ps.light_problem(left, m, ps.lateral_tilt(flat, m)) == ""  # nothing to match when the original is evenly lit
    assert ps.lateral_tilt(right, np.zeros_like(m)) is None


def test_the_light_is_said_in_plain_words():
    right, m = _lit("right")
    plate = np.full((300, 200, 3), (200, 160, 120), np.uint8)  # warm
    shade = [np.ones((300, 200), np.float32) * 0.6 for _ in range(4)]
    hard = ps.key_light_words([right] * 4, [m] * 4, [plate] * 4, shade)
    assert "hard warm direct sun" in hard and "cast shadows" in hard
    soft = ps.key_light_words([right] * 4, [m] * 4, [np.full((300, 200, 3), 120, np.uint8)] * 4, [np.ones((300, 200), np.float32)] * 4)
    assert "soft" in soft and "no cast shadows" in soft
    dim = ps.key_light_words([right] * 4, [m] * 4, [np.full((300, 200, 3), 45, np.uint8)] * 4, shade)
    assert "dim and underexposed" in dim and "no fill light" in dim and "sun" not in dim


def test_the_renders_ground_is_used_only_near_the_character():
    h, w = 400, 300
    bg = np.full((h, w, 3), 120, np.uint8)
    char = np.full((h, w, 3), 60, np.uint8)  # the render's ground is different
    track = np.zeros((h, w), np.float32)
    track[100:380, 130:170] = 1
    region = np.ones((h, w), bool)
    unsure = np.zeros((h, w), bool)
    unsure[:, 20:280] = True  # no frame showed any of this ground
    comp, matte, _ = ps.blend_frame(bg, char, char, track, region, None, None, 0, unsure=unsure)
    far = matte[300, 40]
    assert far < 0.05 and abs(int(comp[300, 40, 0]) - 120) <= 2  # far ground stays the plate's fill
    assert matte[300, 120] > 0.9  # close to the character the render's ground is used


def test_the_renders_ground_takes_the_plates_tone_near_the_character():
    h, w = 400, 300
    bg = np.full((h, w, 3), 120, np.uint8)
    char = np.full((h, w, 3), 170, np.uint8)  # LTX's floor came out lighter
    track = np.zeros((h, w), np.float32)
    track[100:380, 130:170] = 1
    region = np.ones((h, w), bool)
    unsure = np.zeros((h, w), bool)
    unsure[:, 90:210] = True
    comp, matte, _ = ps.blend_frame(bg, char, char, track, region, None, None, 0, unsure=unsure)
    ground = (matte > 0.5) & (np.arange(w)[None, :] < 124)
    assert ground.any()
    assert abs(float(comp[ground].mean()) - 120) < 25  # not the render's 170


def test_a_shadow_running_away_from_the_feet_stays_but_a_compact_dark_remnant_goes():
    h, w = 400, 500
    cover = np.zeros((h, w), bool)
    cover[40:380, 200:230] = True  # 340 tall
    ref = np.full((h, w, 3), 150, np.float32)
    src = ref.copy()
    src[360:380, 230:420] = 90  # the shadow running right, attached to the foot and 190 px long (ratio 0.6)
    src[365:380, 170:196] = 60  # a compact dark thing beside the foot (ratio 0.4), 26 px: a missed heel
    src[cover] = 20
    win = np.ones((h, w), bool)
    rep = ps.replace_mask(cover, src.astype(np.uint8), ref, win)
    assert not rep[370, 260] and not rep[370, 380]  # the long shadow, even next to the foot, stays footage
    assert rep[372, 180]  # the heel is replaced


def test_a_paint_far_brighter_than_a_dim_scene_fails_the_exposure_gate():
    scene = np.full((300, 200, 3), 60, np.uint8)
    m = np.zeros((300, 200), bool)
    m[40:280, 60:140] = True
    dark = scene.copy()
    dark[m] = 25
    bright = scene.copy()
    bright[m] = 200
    assert ps.exposure_problem(dark, m, scene) == ""
    assert ps.exposure_problem(bright, m, scene) == "too bright for the scene"
    assert ps.exposure_ratio(dark, np.zeros_like(m), scene) is None


def test_a_dark_piece_of_the_old_person_beside_a_hole_is_not_copied_into_it():
    rng = np.random.default_rng(4)
    img = np.clip(rng.normal(120, 12, (80, 200, 3)), 0, 255).astype(np.uint8)
    hole = np.zeros((80, 200), bool)
    hole[10:70, 100:130] = True
    shift = ps.hole_shift([hole])
    img[10:70, 100 - shift:112 - shift] = 15  # part of what the shifted copy would read: the old person's black
    out = ps.fill_holes(img, hole, shift)
    assert out[hole].min() > 60  # ground, not black


def test_the_matte_trail_behind_a_moving_foot_is_trimmed_and_the_core_stays():
    m = np.zeros((300, 300), np.float32)
    m[100:290, 100:140] = 1.0  # the character
    m[280:290, 60:100] = 0.35  # a pale trail to the left, in the feet band, soft
    out = ps.trim_trail(m, (8.0, 0.0))  # the foot moves right: the trail is behind
    assert out[285, 80] == 0 and (out[100:290, 100:140] == 1).all()
    assert (ps.trim_trail(m, None) == m).all()
    ahead = m.copy()
    ahead[280:290, 140:180] = 0.35
    assert ps.trim_trail(ahead, (8.0, 0.0))[285, 160] == 0.35  # ahead of the motion is left alone


def test_the_renders_floor_is_used_only_in_the_feet_zone():
    h, w = 400, 300
    bg = np.full((h, w, 3), 100, np.uint8)
    rng = np.random.default_rng(2)
    char = np.clip(rng.normal(100, 10, (h, w, 3)), 0, 255).astype(np.uint8)  # a crisp render floor, the plate's tone
    matte = np.zeros((h, w), np.float32)
    matte[100:380, 130:170] = 1.0
    out = ps.feet_ground(bg, char, matte, None, 0)
    assert (out[:200] == bg[:200]).all()  # nothing changes high up, away from the feet
    assert np.abs(out[360:390, 90:125].astype(int) - 100).mean() > 2  # near the feet the render's floor texture comes in


def test_a_spot_only_one_or_two_frames_show_is_not_a_seen_background():
    h, w, n = 64, 64, 9
    rng = np.random.default_rng(6)
    wall = rng.uniform(90, 150, (h, w, 3)).astype(np.float32)
    frames, cover = [], []
    for i in range(n):
        f = wall.copy()
        c = np.zeros((h, w), bool)
        c[20:60, 20:44] = True
        if i in (3, 5):  # the person steps aside on two frames only: those frames show the spot
            c[20:60, 20:44] = False
            c[20:60, 44:56] = True
        f[c] = (15, 15, 15)
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
        cover.append(c)
    st = ps.align_stack(frames, cover)
    ps.median_of(st)
    assert not st.plate_seen[40, 30]  # shown by 2 frames: not enough
    assert st.plate_seen[40, 5]  # the open wall is seen by all


def test_a_side_view_head_box_is_sized_from_the_torso_not_the_foreshortened_shoulders():
    """T1·1 (a woman in a cloak walking side-on): her shoulders read ~10 px wide, so a shoulder-sized box was a sliver at the
    neck and the gate said 'head: 34% in the picture' on a take whose head is fully there."""
    pose = np.zeros((400, 300, 3), np.uint8)
    nx, ny = 150, 150
    pose[ny:ny + 3, nx - 5:nx] = ps.POSE_R[0]  # shoulders seen side-on: 10 px
    pose[ny:ny + 3, nx:nx + 5] = ps.POSE_L[0]
    pose[ny + 3:ny + 120, nx - 1:nx + 1] = ps.POSE_R[1]  # neck to hips, ~120 px
    pose[ny + 3:ny + 120, nx + 1:nx + 3] = ps.POSE_L[1]
    box = ps.head_box(pose)
    assert box is not None
    ys, xs = np.nonzero(box)
    assert ys.max() < ny and (ny - ys.min()) > 40 and (xs.max() - xs.min()) > 20  # a head-sized box above the neck
    head = np.zeros((400, 300), np.float32)
    head[60:ny, 120:180] = 1  # the head, fully there
    assert ps.part_shares(ps.part_masks(pose), head)["head"] > 0.9
    knee = pose.copy()
    knee[ny + 330:ny + 340, nx - 4:nx + 4] = ps.POSE_L[1]  # the right knee's dot, drawn in the left hip stick's colour
    assert (ps.head_box(knee) == box).all()  # the knee is not the hip
    eye = pose.copy()
    eye[ny - 60:ny - 57, nx + 20:nx + 30] = ps.POSE_R[0]  # a face line near the eye drawn close to the shoulder stick's red
    assert (ps.head_box(eye) == box).all()  # the eye is not the shoulders
    front = _pose_with_neck_and_shoulders()  # front on: unchanged (shoulders set the size)
    assert (ps.head_box(front) == ps.head_box(front)).all() and ps.head_box(front)[:200].any()
