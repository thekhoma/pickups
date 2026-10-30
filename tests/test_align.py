import numpy as np

from pickups import composite


def _scene(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):  # blocks with edges, like a landscape
        y, x = rng.integers(0, h - 40), rng.integers(0, w - 40)
        img[y:y + rng.integers(10, 40), x:x + rng.integers(10, 40)] = rng.integers(0, 255, 3)
    return img


def test_shift_and_offset_agree():
    a = _scene()
    b = composite.shift(a, 7, -4)  # content moves right 7, up 4
    assert composite.frame_offset(a, b) == (7, -4)
    back = composite.shift(b, -7, 4)
    assert np.array_equal(back[10:-10, 10:-10], a[10:-10, 10:-10])


def test_frame_shifts_chain():
    a = _scene(1)
    frames = [composite.shift(a, d, 0) for d in (0, 2, 5, 9)]
    assert composite.frame_shifts(frames) == [(0, 0), (2, 0), (5, 0), (9, 0)]


def test_shift_mask_fill():
    m = np.zeros((10, 10), bool)
    out = composite.shift(m, 3, 0, fill=True)
    assert out[:, :3].all() and not out[:, 3:].any()


def test_split_objects_two_people():
    from pickups.tasks.common import split_objects
    m = np.zeros((200, 300), bool)
    m[50:150, 20:60] = True     # person 1
    m[60:160, 200:240] = True   # person 2
    m[10:12, 290:292] = True    # a speck
    parts = split_objects(m)
    assert len(parts) == 2 and all(p.any() for p in parts)
    assert (parts[0] | parts[1]).sum() == m[50:160, :280].sum()
    one = np.zeros((50, 50), bool); one[10:20, 10:20] = True
    assert len(split_objects(one)) == 1


def test_aligned_temporal_fill_recovers_background_under_drift():
    """A tripod that creeps: the background drifts 1 px per frame while an object walks through. Aligning to frame 1,
    filling from other frames and shifting back must give the real background where the object was."""
    bg = _scene(3, h=200, w=320)
    n, frames, masks, truth = 10, [], [], []
    for i in range(n):
        dx = i  # slow drift
        f = composite.shift(bg, dx, 0)
        m = np.zeros(f.shape[:2], bool)
        m[80:140, 20 + 25 * i:60 + 25 * i] = True  # the walker
        truth.append(f.copy())
        f = f.copy()
        f[m] = (0, 0, 0)
        frames.append(f)
        masks.append(m)
    shifts = composite.frame_shifts(frames)
    assert max(abs(dx - i) for i, (dx, _) in enumerate(shifts)) <= 1
    al = [composite.shift(frames[i], -dx, -dy) for i, (dx, dy) in enumerate(shifts)]
    objs = [composite.shift(masks[i], -dx, -dy, fill=True) for i, (dx, dy) in enumerate(shifts)]
    gen, never = composite.temporal_fill(al, objs)
    gen = [composite.shift(g, dx, dy) for g, (dx, dy) in zip(gen, shifts)]
    i = 5
    hole = masks[i] & ~composite.shift(never[i], shifts[i][0], shifts[i][1], fill=True)
    err = np.abs(gen[i][hole].astype(int) - truth[i][hole].astype(int)).mean()
    assert err < 2.0, err


def test_temporal_fill_ignores_a_frame_whose_mask_missed_part_of_the_person():
    """A walker crosses a locked-off shot; on frame 6 the mask misses their trailing coat tail (dark pixels marked as
    free). Frame 5's hole borrows from frame 6 first: those dark pixels must not end up in the fill."""
    bg = np.full((120, 200, 3), 200, np.uint8)
    bg[:, ::7] = 150  # some texture
    n, frames, masks = 10, [], []
    for i in range(n):
        f, m = bg.copy(), np.zeros((120, 200), bool)
        m[40:100, 10 + 18 * i:40 + 18 * i] = True
        f[m] = 20
        if i == 6:
            f[40:100, 10 + 18 * 5:10 + 18 * 6] = 20  # the coat tail, where frame 5's walker stood, missed by the mask
        frames.append(f)
        masks.append(m)
    gen, never = composite.temporal_fill(frames, masks)
    hole = masks[5] & ~never[5]
    assert np.abs(gen[5][hole].astype(int) - bg[hole].astype(int)).max() <= 1
    plain, _ = composite.temporal_fill(frames, masks, tol=0)  # the old nearest-frame fill copies the coat tail
    assert (plain[5][hole] == 20).any()


def test_temporal_fill_ignores_a_run_of_frames_whose_mask_missed_the_coat():
    """The castle couple: the mask lost a dark coat tail for several frames in a row (a steady run, not a one-frame
    jump). The fill must still be the background seen on most frames."""
    bg = np.full((120, 200, 3), 200, np.uint8)
    n, frames, masks = 14, [], []
    for i in range(n):
        f, m = bg.copy(), np.zeros((120, 200), bool)
        m[40:100, 10 + 12 * i:40 + 12 * i] = True
        f[m] = 20
        if 6 <= i <= 8:
            f[40:100, 70:82] = 20  # the coat tail behind frame 5's walker, missed on three frames
        frames.append(f)
        masks.append(m)
    gen, never = composite.temporal_fill(frames, masks)
    hole = masks[5] & ~never[5]
    assert np.abs(gen[5][hole].astype(int) - bg[hole].astype(int)).max() <= 1


def test_seed_frame_prefers_the_frame_where_people_are_apart(monkeypatch, tmp_path):
    """A hugging couple is one SAM instance on the chosen frame; tracking starts where SAM sees two people."""
    from pickups.tasks import common

    counts = {0: 2, 25: 1, 50: 1, 75: 2, 99: 1}  # frame -> people SAM finds

    def fake_instances(ctx, png, text, scale, out):
        f = int(png.stem.split("_")[1])
        return [np.ones((4, 4), bool)] * counts.get(f, 1)

    monkeypatch.setattr(common, "select_instances", fake_instances)
    monkeypatch.setattr(common, "_frame_png", lambda video, i, out: out)
    frame, parts = common.seed_frame(None, tmp_path / "p.mp4", 100, 60, "person", 1.0, tmp_path)
    assert frame == 75 and len(parts) == 2  # the nearest frame with two people
    frame, parts = common.seed_frame(None, tmp_path / "p.mp4", 100, 50, "person", 1.0, tmp_path)
    assert frame == 75 and len(parts) == 2


def test_sam_labels_carry_a_count():
    """SAM 3 in ComfyUI keeps only its best match unless the label says how many ("person:8")."""
    from pickups import graphs

    assert graphs.counted("person") == f"person:{graphs.MAX_PEOPLE}"
    assert graphs.counted("person:3") == "person:3" and graphs.counted("") == ""
    g = graphs.select_frame("f.png", [], text="person", individual=True)
    assert g["PU_Label"]["inputs"]["text"] == f"person:{graphs.MAX_PEOPLE}" and g["detect"]["inputs"]["individual_masks"]
    assert graphs.select("c.mp4", "woman")["PU_Label"]["inputs"]["text"] == f"woman:{graphs.MAX_PEOPLE}"


def test_selection_words_can_name_several_things():
    from pickups.tasks import common

    assert common.labels("person, handbag") == ["person", "handbag"]
    assert common.labels("man in dark coat") == ["man in dark coat"]
    assert common.labels("") == [""] and common.labels(" , ") == [""]


def test_harmonize_around_keeps_the_subjects_colour():
    """A render whose grass is warmer than the footage must not tint the new character (the purple skirt)."""
    src = np.full((96, 96, 3), (60, 120, 50), np.float32)  # the footage's grass
    gen = np.full((96, 96, 3), (80, 140, 30), np.float32)  # the render's warmer grass
    gen[32:64, 32:64] = (90, 60, 40)  # the new character's brown skirt
    hole = np.zeros((96, 96), np.float32)
    hole[24:72, 24:72] = 1.0
    subject = np.zeros((96, 96), np.float32)
    subject[32:64, 32:64] = 1.0
    plain = composite.harmonize(gen, src, hole)
    kept = composite.harmonize_around(gen, src, hole, subject)
    assert plain[48, 48, 2] > plain[48, 48, 1]  # the old way: bluer than green, i.e. purple
    np.testing.assert_allclose(kept[48, 48], gen[48, 48] + np.clip((src - gen)[0, 0].mean(), -10, 10), atol=1.5)
    np.testing.assert_allclose(kept[26, 26], plain[26, 26], atol=1e-3)  # the fill around still meets the footage


def test_occluders_keep_a_hand_in_front_but_not_the_ground_or_the_wall():
    depth = np.full((120, 80), 60, np.uint8)  # the wall behind
    depth[100:] = 150  # the floor, nearer toward the bottom
    child = np.zeros((120, 80), bool)
    child[30:100, 30:50] = True
    depth[child] = 110
    depth[55:65, 20:45] = 170  # the father's hand, in front of the child
    child[55:65, 30:45] = False  # SAM's child mask stops at the hand
    area = composite.grow(child, 10)
    occ = composite.occluders(depth, child, area)
    assert occ[60, 35] and occ[60, 25]  # the hand, over the child and beside them
    assert not occ[105, 40]  # the floor at the feet
    assert not occ[40, 25] and not occ[40, 40]  # the wall and the child
