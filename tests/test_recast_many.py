"""Recast anyone: one Recast take that swaps several people, each to their own character. The swaps params, the window
and the anchor for everyone, keeping people (and their new characters) apart, the paint checks, and a whole run with
ComfyUI faked: each person gets their own character, a paint that spills onto the other person is tried again, and
nothing outside the people changes."""

import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from pickups import graphs
from pickups.tasks import common, recast, recast_ltx, recast_many


def box(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


Z = np.zeros((60, 100), bool)


# --- params ----------------------------------------------------------------------------------------------------------


def test_swaps_are_read_and_a_person_with_no_character_stays_as_filmed():
    p = {"seed": 7, "swaps": [
        {"frame": 40, "points": [{"x": 10, "y": 20}], "cast_id": "cast_a", "label": "the man"},
        {"frame": 40, "points": [{"x": 90, "y": 20, "positive": True}], "ref_upload_id": "up_1", "describe": "an alien"},
        {"frame": 40, "points": [{"x": 50, "y": 20}]},  # no character: as filmed
    ]}
    sw = recast_many.parse_swaps(p)
    assert len(sw) == 2
    assert sw[0] == {"frame": 40, "points": [{"x": 10.0, "y": 20.0, "positive": True}], "text": "", "cast_id": "cast_a", "label": "the man"}
    assert sw[1]["ref_upload_id"] == "up_1" and sw[1]["describe"] == "an alien"
    assert recast_many.parse_swaps({"frame": 3}) == [] and recast_many.parse_swaps({"swaps": []}) == []


def test_one_swap_is_todays_single_recast():
    p = {"seed": 7, "scene": "a loch", "cast_id": "cast_top", "swaps": [{"frame": 12, "points": [{"x": 1, "y": 2}], "text": "man"}]}
    sw = recast_many.parse_swaps(p)
    assert sw[0]["cast_id"] == "cast_top"  # one swap with no character of its own: the form's character
    q = recast_many.single_params(p, sw[0])
    assert q == {"seed": 7, "scene": "a loch", "frame": 12, "points": [{"x": 1.0, "y": 2.0, "positive": True}], "text": "man",
                 "cast_id": "cast_top"}


def test_bad_swaps_say_what_to_do():
    with pytest.raises(RuntimeError, match="click"):
        recast_many.parse_swaps({"swaps": [{"frame": 0, "points": [], "cast_id": "c"}]})
    with pytest.raises(RuntimeError, match="at least one"):
        recast_many.parse_swaps({"swaps": [{"frame": 0, "points": [{"x": 1, "y": 1}]}, {"frame": 0, "points": [{"x": 5, "y": 1}]}]})
    with pytest.raises(RuntimeError, match="up to"):
        recast_many.parse_swaps({"swaps": [{"points": [{"x": i, "y": 1}], "describe": "x"} for i in range(recast_many.MAX_SWAPS + 1)]})
    with pytest.raises(RuntimeError):
        recast_many.parse_swaps({"swaps": "everyone"})


# --- window and anchor for everyone --------------------------------------------------------------------------------------


def test_the_window_goes_where_most_of_them_are_on_screen():
    h, w = 60, 100
    a = [box(h, w, 10, 10, 20, 50) for _ in range(40)]  # there all along
    b = [box(h, w, 70, 10, 80, 50) if 20 <= i <= 35 else Z for i in range(40)]  # only later
    assert recast_many.choose_window_many([a, b], 9, 40) == 20  # the earliest window with both
    assert recast_many.choose_window_many([a, b], 9, 40, start=3) == 3
    assert recast_many.choose_window_many([a], 9, 40) == recast_ltx.choose_window(a, 9, 40) == 0


def test_someone_walking_in_through_the_edge_counts_as_only_partly_there():
    h, w = 60, 100
    a = [box(h, w, 40, 10, 60, 50)] * 30
    b = [box(h, w, -10, 10, 10, 50) if i < 15 else box(h, w, 20, 10, 40, 50) for i in range(30)]  # cut off, then whole
    assert recast_many.choose_window_many([a, b], 9, 30) == 15


def test_the_anchor_is_where_they_are_all_big_and_whole():
    h, w = 60, 100
    a = [box(h, w, 10, 5, 30, 58)] * 25  # big all along
    b = [box(h, w, 85, 20, 105, 50) if i < 16 else box(h, w, 60, 5, 80, 58) for i in range(25)]  # cut off, then whole
    k = recast_many.anchor_many([a, b])
    assert k == 16 and k % recast_ltx.GRID == 0
    assert recast_many.anchor_many([a, a]) == 0  # as good on the first frame: the in-place pin
    assert recast_many.anchor_many([[Z] * 9, [Z] * 9]) == 0


def test_the_crop_is_around_everyone():
    h, w = 90, 160
    a = [box(h, w, 10, 30, 20, 70)] * 9
    b = [box(h, w, 130, 30, 140, 70)] * 9
    union = [x | y for x, y in zip(a, b)]
    x0, y0, cw, ch = recast_ltx.crop_box(union, (1920, 1080))
    assert x0 <= 10 * 12 and x0 + cw >= 140 * 12  # both of them inside


def test_people_are_named_by_where_they_stand():
    h, w = 60, 100
    l, m, r = box(h, w, 5, 5, 15, 50), box(h, w, 45, 5, 55, 50), box(h, w, 85, 5, 95, 50)
    assert recast_many.positions([r, l]) == ["on the right", "on the left"]
    assert recast_many.positions([m, r, l]) == ["in the middle", "on the right", "on the left"]
    four = [box(h, w, x, 5, x + 5, 50) for x in (60, 5, 30, 90)]
    assert recast_many.positions(four) == ["second from the right", "on the far left", "second from the left", "on the far right"]
    assert recast_many.of_picture("on the left") == "on the left of"
    assert recast_many.of_picture("second from the left") == "second from the left in"


# --- keeping them apart --------------------------------------------------------------------------------------------------


def test_touching_people_are_kept_apart():
    h, w = 60, 100
    a, b = box(h, w, 10, 10, 50, 50), box(h, w, 40, 10, 80, 50)  # overlap at x 40-50
    da, db = recast_many.disjoint([a, b])
    assert not (da & db).any() and ((da | db) == (a | b)).all()
    assert da[30, 42] and db[30, 48]  # each keeps the side nearer their own middle


def test_two_tracks_that_swap_people_are_swapped_back():
    h, w = 60, 100
    left = [box(h, w, 10 + i, 10, 30 + i, 50) for i in range(12)]
    right = [box(h, w, 70 - i, 10, 90 - i, 50) for i in range(12)]
    t0 = left[:6] + right[6:]  # SAM's object 0 jumps onto the other person at frame 6
    t1 = right[:6] + left[6:]
    (f0, f1), fixes = recast_many.keep_identities([t0, t1])
    assert fixes == 1
    assert all((f0[i] == left[i]).all() and (f1[i] == right[i]).all() for i in range(12))
    same, none = recast_many.keep_identities([left, right])
    assert none == 0 and all((same[0][i] == left[i]).all() for i in range(12))
    gone = left[:4] + [Z.copy() for _ in range(8)]  # someone leaving the frame isn't a swap
    _, fixes = recast_many.keep_identities([gone, right])
    assert fixes == 0


def test_each_new_character_stays_on_its_own_person():
    h, w = 60, 100
    a, b = box(h, w, 10, 10, 30, 50), box(h, w, 60, 10, 80, 50)
    (n0, n1), swapped = recast_many.own_person([b, a], [a, b], 3)  # the tracker handed them over
    assert swapped and (n0 == a).all() and (n1 == b).all()
    (n0, n1), swapped = recast_many.own_person([a, b], [a, b], 3)
    assert not swapped and (n0 == a).all()


def test_detections_go_to_the_person_whose_place_they_are_on():
    h, w = 60, 100
    a, b = box(h, w, 10, 10, 30, 50), box(h, w, 60, 10, 80, 50)
    big_a, far = box(h, w, 6, 4, 34, 52), box(h, w, 90, 0, 99, 9)
    sa, sb = recast_many.assign([far, big_a], [a, b])
    assert (sa == big_a).all() and not sb.any()


# --- painting one at a time ------------------------------------------------------------------------------------------------


def test_a_paint_that_spills_onto_the_other_person_is_caught_and_put_back():
    h, w = 60, 100
    before = np.full((h, w, 3), 100, np.uint8)
    a, b = box(h, w, 10, 10, 30, 50), box(h, w, 60, 10, 80, 50)
    after = before.copy()
    after[a] = (0, 200, 0)
    changed, leak = recast_many.paint_effect(before, after, a, b, 3)
    assert changed == 1.0 and leak == 0.0
    after[b] = (200, 200, 0)  # the painter changed the other person too
    changed, leak = recast_many.paint_effect(before, after, a, b, 3)
    assert leak == 1.0 and leak > recast_many.LEAK_MAX
    after[box(h, w, 85, 0, 99, 10)] = (255, 0, 0)  # and drew something in a corner
    fixed = recast_many.keep_local(before, after, a, b, 3, 4)
    assert (fixed[b & ~recast_many.composite.grow(a, 12)] == 100).all()  # the other person as they were
    assert (fixed[0:10, 85:99] == 100).all()  # nothing added away from the person
    assert (fixed[a] == (0, 200, 0)).all()  # this one painted


def test_each_person_is_painted_on_a_crop_with_them_in_the_middle():
    t = box(600, 1000, 700, 100, 760, 500)
    x0, y0, x1, y1 = recast_many.paint_box(t)
    assert x0 <= 700 and x1 >= 760 and y0 <= 100 and y1 >= 500 and x1 <= 1000 and y1 <= 600
    assert abs((x0 + x1) / 2 - 730) < 40 and 0.6 <= (x1 - x0) / (y1 - y0) <= 1.8
    assert all(v % 2 == 0 for v in (x0, y0))
    pw, ph = recast_many.paint_size(x1 - x0, y1 - y0)
    assert pw % 32 == 0 and ph % 32 == 0 and 0.8e6 < pw * ph < 1.2e6
    assert recast_many.paint_order([box(60, 100, 0, 0, 5, 55), box(60, 100, 50, 0, 55, 30)]) == [1, 0]  # the furthest first


def test_the_ltx_prompt_names_everyone_by_place():
    text = recast_many.ltx_prompt_many([
        {"who": "A knight in plate armour", "motion": "walks away from the camera", "where": "on the left", "vehicle": False},
        {"who": "", "motion": "moves naturally where they stand", "where": "on the right", "vehicle": False}], "A castle by a loch.")
    assert text.startswith("On the left, A knight in plate armour walks away from the camera. On the right, the new character "
                           "moves naturally where they stand. A castle by a loch.")


def test_the_tracker_returns_every_person_on_their_own():
    g = graphs.track_mask("clip.mp4", ["a.png", "b.png"], each=True)
    assert g["obj0"]["inputs"]["object_indices"] == "0" and g["obj1"]["inputs"]["object_indices"] == "1"
    assert g["PU_Obj1"]["inputs"]["filename_prefix"] == "pickups/track/obj1/mask"
    assert "PU_Obj0" not in graphs.track_mask("clip.mp4", ["a.png", "b.png"])  # as before unless asked
    assert common._match_parts([box(10, 10, 0, 0, 3, 3), box(10, 10, 6, 6, 9, 9)],
                               [box(10, 10, 6, 6, 9, 9), box(10, 10, 0, 0, 3, 3)]) == [1, 0]


# --- a whole two-person Recast, ComfyUI faked ------------------------------------------------------------------------------

W, H, N = 320, 192, 25
BLUE, RED = (30, 40, 200), (200, 40, 40)          # the man on the left, the woman on the right
GREEN, YELLOW = (40, 190, 60), (225, 205, 40)     # who they become
DECOY = (slice(10, 40), slice(140, 180))          # someone else LTX drew, far from both (LTX's own pixels)


def colour(img, c):
    return np.abs(img.astype(int) - np.array(c)).max(-1) < 60


def blob(img, c):
    """colour() as SAM would see it: the solid figure, not the odd stray pixel at an edge."""
    m = colour(img, c)
    return m & (m.sum(1) > 3)[:, None] & (m.sum(0) > 3)[None, :]


def man_x(i):
    return 40 + i


def woman_x(i):
    return 230 - i


def source(woman_h=140):
    yy, xx = np.mgrid[:H, :W]
    bg = np.stack([110 + 30 * np.sin(xx / 9.0), 115 + 20 * np.cos(yy / 7.0), 105 + 25 * np.sin((xx - yy) / 11.0)], -1).astype(np.uint8)
    frames = []
    for i in range(N):
        f = bg.copy()
        f[box(H, W, man_x(i), 36, man_x(i) + 40, 176)] = BLUE
        f[box(H, W, woman_x(i), 176 - woman_h, woman_x(i) + 36, 176)] = RED
        frames.append(f)
    return frames


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

    def __init__(self, out, w, h, fps, kind="prores", scale_long=None):
        self.frames = []
        _Writer.made[out.name] = self

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        pass


def _blob_at(f, pt):
    for c in (BLUE, RED, GREEN, YELLOW):
        m = blob(f, c)
        if m[min(f.shape[0] - 1, int(pt["y"])), min(f.shape[1] - 1, int(pt["x"]))]:
            return m
    return np.zeros(f.shape[:2], bool)


class _Ctx:
    def __init__(self, store, sid, params, tmp):
        from pathlib import Path
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.client, self.ran, self.tmp, self.stages, self.paints = _Client(), [], Path(tmp), [], []

    def stage(self, label):
        self.stages.append(label)

    def check(self):
        pass

    def progress(self, frac):
        pass

    def commercial_safe(self):
        return False

    def _save(self, out, masks, name="mask"):
        files = []
        for j, m in enumerate(masks):
            Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"{name}_{j + 1:05d}_.png")
            files.append(_Out(p))
        return files

    def run(self, graph):
        self.ran.append(graph)
        out = self.tmp / f"job{len(self.ran)}"
        out.mkdir()
        types = {n["class_type"] for n in graph.values()}
        if "TextEncodeQwenImage21" in types:  # the frame editor: the named person becomes their character
            text = graph["PU_Prompt"]["inputs"]["prompt"]
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB")).copy()
            mid = (f.shape[0] // 2, f.shape[1] // 2)  # the person in the middle of the crop

            def dist(c):
                ys, xs = np.nonzero(blob(f, c))
                return abs(xs.mean() - mid[1]) if len(xs) else 1e9
            old, other = (BLUE, RED) if dist(BLUE) < dist(RED) else (RED, BLUE)
            new = GREEN if "green" in text else YELLOW
            self.paints.append(text)
            if len(self.paints) == 1:  # the first try adds a character in a corner instead of changing the person
                f[:20, :20] = new
            else:
                f[colour(f, old)] = new
            Image.fromarray(f).resize((W * 2, H * 2), Image.NEAREST).save(p := out / "plate_00001_.png")  # the editor's own size
            return _Job({"PU_Plate": [_Out(p)]})
        if "LTXAddVideoICLoRAGuide" in types:  # LTX: everything re-rendered; each person as their (bulkier) character
            files, depth = [], []
            key = np.asarray(Image.open(graph["PU_First"]["inputs"]["image"]).convert("RGB")) if "PU_First" in graph else None
            assert key is not None and colour(key, GREEN).any()  # painted on the keyframe
            self.keyframe = key
            for i, f in enumerate(common.read_frames(graph["PU_Control"]["inputs"]["file"])):
                g = np.clip(f.astype(int) + (14, -6, 8), 0, 255).astype(np.uint8)
                for old, new in ((BLUE, GREEN), (RED, YELLOW)):
                    m = colour(f, old)
                    xs, ys = np.flatnonzero(m.sum(0) > 20), np.flatnonzero(m.sum(1) > 20)  # not the odd stray edge pixel
                    if len(xs) and len(ys):
                        g[box(*m.shape, xs.min() - 4, ys.min() - 3, xs.max() + 4, ys.max() + 1)] = new
                g[DECOY] = GREEN
                Image.fromarray(g).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
                d = np.full(f.shape[:2], 60, np.uint8)  # the shot's depth (near bright): the people a little nearer
                d[colour(f, BLUE) | colour(f, RED)] = 120
                Image.fromarray(d).save(p := out / f"control_{i + 1:05d}_.png")
                depth.append(_Out(p))
            return _Job({"PU_Out": files, "PU_ControlPreview": depth})
        if "SAM3_VideoTrack" in types:  # following each object by its colour; the objects come back in reverse order
            firsts = [graph[k]["inputs"]["image"] for k in sorted(graph) if k.startswith("PU_First")]
            frames = list(common.read_frames(graph["PU_Clip"]["inputs"]["file"]))
            cols = []
            for fp in firsts:
                seed = np.asarray(Image.open(fp).convert("L")) > 127
                f0 = np.asarray(Image.fromarray(frames[0]).resize(seed.shape[::-1]))
                cols.append(max((BLUE, RED, GREEN, YELLOW), key=lambda c: int((colour(f0, c) & seed).sum())))
            each = [[blob(f, c) for f in frames] for c in cols][::-1]
            outs = {"PU_Mask": self._save(out, [np.logical_or.reduce(fr) for fr in zip(*each)])}
            if "PU_Obj0" in graph:
                for j, ms in enumerate(each):
                    (out / f"o{j}").mkdir()
                    outs[f"PU_Obj{j}"] = self._save(out / f"o{j}", ms)
            return _Job(outs)
        if "SAM3_Detect" in types:
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB"))
            d = graph["detect"]["inputs"]
            label = graph["PU_Label"]["inputs"]["text"].split(":")[0] if "PU_Label" in graph else ""
            if "positive_coords" in d:  # a click: the blob under the clicks
                m = np.logical_or.reduce([_blob_at(f, pt) for pt in json.loads(d["positive_coords"])])
                return _Job({"PU_Mask": self._save(out, [m])})
            found = []
            if label in ("person", "character"):
                found = [blob(f, c) for c in (BLUE, RED, GREEN, YELLOW) if blob(f, c).any()]
            return _Job({"PU_Mask": self._save(out, found + [np.zeros(f.shape[:2], bool)])})
        raise AssertionError(f"unexpected graph {sorted(types)}")


@pytest.fixture()
def couple(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")

    def make(woman_h=140):
        src = tmp_path / f"couple_{woman_h}.mov"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                              "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(src)], stdin=subprocess.PIPE)
        p.communicate(b"".join(f.tobytes() for f in source(woman_h)))
        return store.add_shot(src, "Couple")["id"]

    from io import BytesIO
    b = BytesIO()
    Image.new("RGB", (300, 500), (200, 200, 200)).save(b, "PNG")
    up = store.add_upload(b.getvalue(), "char.png")["id"]
    monkeypatch.setattr(common, "MAX_FRAMES", 9)
    monkeypatch.setattr(common, "FrameWriter", _Writer)
    monkeypatch.setattr(common, "outputs_from_master", lambda *a, **k: [])
    monkeypatch.setattr(recast_many, "make_viewer", lambda *a, **k: None)
    monkeypatch.setattr(recast_ltx.reference, "prepare", lambda *a, **k: type("P", (), {"views": 1})())
    _Writer.made = {}
    return store, make, up


def _swaps(up):
    return [{"frame": 4, "points": [{"x": man_x(4) + 20, "y": 100}], "kind": "person", "ref_upload_id": up,
             "describe": "a man in a green suit", "name": "Knight", "label": "the man"},
            {"frame": 4, "points": [{"x": woman_x(4) + 18, "y": 170}], "kind": "person", "ref_upload_id": up,
             "describe": "a woman in a yellow coat", "name": "Alien"}]


def test_two_people_each_become_their_own_character(couple, tmp_path):
    store, make, up = couple
    sid = make()
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"swaps": _swaps(up), "seed": 5}, tmp_path / "jobs")
    res = recast.run(ctx)
    d, qa = res["details"], res["details"]["_qa"]
    assert ctx.stages == [s.label for s in recast_ltx.STAGES]
    assert d["swaps"] == [{"who": "the man", "character": "Knight", "reference": "one figure on a plain backdrop"},
                          {"who": "the person on the right", "character": "Alien", "reference": "one figure on a plain backdrop"}]
    assert "left_as_filmed" not in d
    # Painted one at a time, each in the middle of their own crop; a first try that missed the person is tried again.
    assert len(ctx.paints) == 3 and all("in the middle of <image1>" in t for t in ctx.paints)
    assert "red" not in ctx.paints[0] and "green suit" in ctx.paints[0] and "yellow coat" in ctx.paints[2]
    people = {q["character"]: q for q in qa["people"]}
    assert people["Knight"]["paint"]["attempts"] == 2 and people["Knight"]["paint"]["ok"]
    assert people["Alien"]["paint"]["attempts"] == 1 and people["Alien"]["paint"]["found"] == "person"
    assert all(q["empty_new_frames"] == 0 and q["coverage"] > 1 for q in qa["people"])
    tracks = [g for g in ctx.ran if "track" in g and g["track"]["class_type"] == "SAM3_VideoTrack"]
    assert tracks and all("PU_Obj1" in g for g in tracks)  # each person followed on their own
    assert colour(ctx.keyframe, GREEN).any() and colour(ctx.keyframe, YELLOW).any()  # both painted on the one keyframe
    drive = next(g for g in ctx.ran if "PU_Control" in g)
    assert drive["PU_Prompt"]["inputs"]["text"].startswith("On the left, a man in a green suit")
    assert "On the right, a woman in a yellow coat" in drive["PU_Prompt"]["inputs"]["text"]

    assert d["pixels_changed_outside_selection"] == 0
    out = _Writer.made["master.mov"].frames
    src = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    s0 = qa["window_start"]
    for i in range(len(out)):
        mx, wx = man_x(s0 + i), woman_x(s0 + i)
        assert colour(out[i][90:110, mx + 8:mx + 32], GREEN).all()  # the man became the green character
        assert colour(out[i][90:110, wx + 8:wx + 28], YELLOW).all()  # the woman, hers
        assert not colour(out[i][90:110, wx + 8:wx + 28], GREEN).any()  # nobody half-swapped or swapped over
        assert np.array_equal(out[i][DECOY], src[i][DECOY])  # someone else LTX drew is not pasted
        assert np.array_equal(out[i][:, :mx - 12], src[i][:, :mx - 12])  # beyond the people: the footage
        assert np.array_equal(out[i][:, wx + 50:], src[i][:, wx + 50:])
        assert np.array_equal(out[i][180:], src[i][180:])
    receipt = res["receipt"]
    assert receipt["ai"] > 0 and receipt["original"] + receipt["ai"] == pytest.approx(100, abs=0.02)
    assert (ctx.vdir / "painted_frame.png").exists()


def test_someone_too_small_to_draw_is_left_as_filmed(couple, tmp_path):
    store, make, up = couple
    sid = make(woman_h=12)
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"swaps": _swaps(up)}, tmp_path / "jobs")
    res = recast.run(ctx)
    d = res["details"]
    assert [s["character"] for s in d["swaps"]] == ["Knight"]
    assert len(d["left_as_filmed"]) == 1 and "too small" in d["left_as_filmed"][0] and "left as filmed" in d["left_as_filmed"][0]
    out, src = _Writer.made["master.mov"].frames, list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    s0 = d["_qa"]["window_start"]
    for i in range(len(out)):
        wx = woman_x(s0 + i)
        assert np.array_equal(out[i][150:180, wx - 4:wx + 40], src[i][150:180, wx - 4:wx + 40])  # she is as filmed


def test_a_big_gpu_renders_bigger(monkeypatch):
    monkeypatch.delenv("PICKUPS_LTX_PIXELS", raising=False)
    assert recast_ltx.max_pixels(24) == recast_ltx.max_pixels(None) == recast_ltx.MAX_PIXELS  # the 4090 as before
    assert recast_ltx.max_pixels(96) == recast_ltx.BIG_MAX_PIXELS
    w, h = recast_ltx.ltx_size(3840, 2748, recast_ltx.max_pixels(96))
    assert w % 128 == 0 and h % 128 == 0 and recast_ltx.MAX_PIXELS < w * h <= recast_ltx.BIG_MAX_PIXELS
    monkeypatch.setenv("PICKUPS_LTX_PIXELS", str(1920 * 1088))
    assert recast_ltx.max_pixels(96) == 1920 * 1088
