"""Recast on LTX-2.5 (the default engine): where in a long shot it works, which frame the character is painted on, the
crop LTX renders, the graph that pins the painted frame there, and a whole run (ComfyUI faked) that pastes only the
new character back, leaving every other pixel of the footage as it was."""

import subprocess
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import graphs
from pickups.app.jobs import Stage, TaskSpec
from pickups.tasks import TASKS, common, person_swap, recast, recast_ltx


def box(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


# --- the window ------------------------------------------------------------------------------------------------------


def test_the_window_goes_where_the_person_is_on_screen():
    h, w = 60, 100
    masks = [box(h, w, 40, 10, 50, 50) if 12 <= i <= 25 else np.zeros((h, w), bool) for i in range(40)]
    assert recast_ltx.choose_window(masks, 9, 40) == 12  # the earliest of the windows they fill
    assert recast_ltx.choose_window(masks, 9, 40, start=30) == 30  # the user's start wins
    assert recast_ltx.choose_window(masks, 9, 40, start=99) == 31  # ...within the shot
    assert recast_ltx.choose_window(masks, 9, 9) == 0  # a shot no longer than the window
    assert recast_ltx.choose_window([np.zeros((h, w), bool)] * 40, 9, 40) == 0  # nobody: from the start


def test_bigger_counts_when_they_are_on_screen_as_long():
    h, w = 60, 100
    small, big = box(h, w, 40, 30, 46, 42), box(h, w, 30, 5, 60, 58)
    masks = [small] * 10 + [np.zeros((h, w), bool)] * 5 + [big] * 10
    assert recast_ltx.choose_window(masks, 9, 25) == 15  # nine frames each way: the window where they're big


# --- the anchor ------------------------------------------------------------------------------------------------------


def test_cut_off_is_the_share_of_the_outline_on_the_frame_edge():
    assert recast_ltx.cut_off(box(100, 100, 20, 20, 60, 90)) == 0
    half = box(100, 100, 0, 20, 20, 90)  # walking in through the left edge
    assert 0.2 < recast_ltx.cut_off(half) <= 1
    assert recast_ltx.cut_off(np.zeros((100, 100), bool)) == 1


def test_the_anchor_is_where_they_are_biggest_and_whole_on_ltxs_grid():
    h, w = 100, 160
    walk_in = [box(h, w, -40 + 6 * i, 10, 10 + 6 * i, 95) for i in range(8)]  # half of them through the left edge
    whole = [box(h, w, 30 + i, 10, 80 + i, 95) for i in range(8, 20)]
    away = [box(h, w, 60, 30, 75, 70) for _ in range(20, 33)]  # walking away: smaller
    k = recast_ltx.anchor_frame(walk_in + whole + away)
    assert k % recast_ltx.GRID == 0 and k in (8, 16)
    standing = [box(h, w, 50, 10, 90, 95)] * 33
    assert recast_ltx.anchor_frame(standing) == 0  # as big on the first frame: the in-place pin
    assert recast_ltx.anchor_frame([np.zeros((h, w), bool)] * 9) == 0


# --- the crop --------------------------------------------------------------------------------------------------------


def test_a_person_who_goes_everywhere_gets_the_whole_frame():
    masks = [box(90, 160, 5 * i, 5, 5 * i + 30, 88) for i in range(26)]
    assert recast_ltx.crop_box(masks, (3840, 2160)) == (0, 0, 3840, 2160)


def test_a_small_person_gets_a_crop_around_them_in_ltxs_shape():
    masks = [box(90, 160, 60 + i // 4, 30, 72 + i // 4, 70) for i in range(25)]  # standing, a bit off centre
    x0, y0, cw, ch = recast_ltx.crop_box(masks, (3840, 2160))
    assert cw < 3840 and ch <= 2160 and cw % 2 == 0 and ch % 2 == 0
    assert cw >= recast_ltx.MIN_CROP * 3840 - 2 and ch >= recast_ltx.MIN_CROP * 2160 - 2
    person = (60 * 24, 30 * 24, (72 + 6) * 24, 70 * 24)  # their box in the frame's pixels
    assert x0 <= person[0] and y0 <= person[1] and x0 + cw >= person[2] and y0 + ch >= person[3]
    assert 0 <= x0 and 0 <= y0 and x0 + cw <= 3840 and y0 + ch <= 2160
    w, h = recast_ltx.ltx_size(cw, ch)
    assert abs(cw / ch - w / h) < 0.03  # LTX renders the crop's own shape: nothing is stretched


def test_ltx_sizes_are_multiples_of_128_within_the_pixel_budget():
    for cw, ch in ((3840, 2748), (1800, 2160), (500, 400), (2400, 900)):
        w, h = recast_ltx.ltx_size(cw, ch)
        assert w % 128 == 0 and h % 128 == 0
        assert w * h <= recast_ltx.MAX_PIXELS
        assert abs(w / h - cw / ch) / (cw / ch) < 0.15
    assert recast_ltx.ltx_size(3840, 2748) == (1664, 1152)  # the full 4K frame, as Change weather renders it
    w, h = recast_ltx.ltx_size(500, 400)
    assert w * h > 500 * 400  # a small crop is rendered bigger than the plate


# --- words -----------------------------------------------------------------------------------------------------------


def test_what_they_do_and_where_they_are():
    h, w = 100, 200
    towards = [box(h, w, 90, 40 - 2 * i, 100 + i, 50 + 2 * i) for i in range(20)]
    assert recast_ltx.motion_words(towards) == "walks towards the camera"
    walk_in = [box(h, w, 0, 10 + 2 * i, 40 + 6 * i, 95 - 2 * i) for i in range(20)]
    assert recast_ltx.motion_words(walk_in) == "walks in from the left and away from the camera"
    assert recast_ltx.motion_words([box(h, w, 90, 10, 110, 90)] * 20) == "moves naturally where they stand"
    assert recast_ltx.where_words(box(h, w, 10, 10, 40, 90)) == "on the left of"
    assert recast_ltx.where_words(box(h, w, 160, 10, 190, 90)) == "on the right of"
    assert recast_ltx.person_noun("man") == "man" and recast_ltx.person_noun("the woman") == "woman"
    assert recast_ltx.person_noun("the guy in the coat") == "person"
    assert recast_ltx.person_noun("man in green parka") == "man"
    assert recast_ltx.person_phrase("the man in green parka") == "man in green parka"
    assert recast_ltx.person_phrase("woman, handbag") == "" and recast_ltx.person_phrase("guy in a coat") == ""
    assert "Turn the man in green parka on the right of <image1>" in recast_ltx.paint_instruction(
        "man", "on the right of", False, False, "", "man in green parka")


def test_the_paint_instruction_names_the_person_by_place_keeps_their_pose_and_asks_for_the_scenes_light():
    text = recast_ltx.paint_instruction("man", "on the left of", True, True, "an astronaut in a white suit")
    assert text.startswith("Turn the man walking on the left of <image1> into the character from the character sheet "
                           "in <image2> (an astronaut in a white suit): keep the man's exact pose, mid-stride,")
    assert "no rim light" in text and "no glow" in text and text.endswith("Keep everything else in <image1> exactly as it is.")
    still = recast_ltx.paint_instruction("person", "in the middle of", False, False, "")
    assert still.startswith("Turn the person in the middle of <image1> into the character in <image2>: keep the person's exact "
                            "pose, with the head")
    prompt = recast_ltx.ltx_prompt("a knight in steel armour", "walks towards the camera", "A ridge at dusk.")
    assert prompt.startswith("A knight in steel armour walks towards the camera. A ridge at dusk.")


def test_only_the_new_character_touching_the_old_person_counts():
    old = box(100, 100, 40, 10, 60, 90)
    costume = box(100, 100, 34, 8, 66, 92)
    someone_else = box(100, 100, 80, 20, 95, 90)
    new = recast_ltx.new_character(costume | someone_else, old, 3)
    assert new[costume].all() and not new[someone_else].any()
    assert not recast_ltx.new_character(np.zeros_like(old), old, 3).any()


def test_a_helmet_sam_split_off_at_the_neck_stays_on():
    old = box(100, 100, 40, 30, 60, 90)
    body = box(100, 100, 36, 22, 64, 92)
    helmet = box(100, 100, 42, 5, 58, 19)  # 3 px above the body, beyond the old person's reach
    new = recast_ltx.new_character(body | helmet, old, 3)
    assert new[body].all() and new[helmet].all()
    assert not (new & ~(body | helmet)).any()


# --- the graph -------------------------------------------------------------------------------------------------------


def test_the_painted_frame_is_a_keyframe_guide_at_the_anchor():
    plain = graphs.drive("c.mp4", "f.png", "p", 24, 1664, 1152, control_mode="depth")
    assert graphs.drive("c.mp4", "f.png", "p", 24, 1664, 1152, control_mode="depth", first_at=0) == plain  # unchanged
    g = graphs.drive("c.mp4", "f.png", "p", 24, 1664, 1152, control_mode="depth", first_at=16)
    assert "i2v1" not in g and "i2v2" not in g  # no first frame in place
    assert g["guide"]["inputs"]["latent"] == ["vlat", 0]
    for key, img, latent in (("key1", "first_half", ["guide", 2]), ("key2", "first_full", ["up", 0])):
        n = g[key]
        assert n["class_type"] == "LTXVAddGuide" and n["inputs"]["frame_idx"] == 16 and n["inputs"]["strength"] == 1.0
        assert n["inputs"]["image"] == [img, 0] and n["inputs"]["latent"] == latent
    assert g["key1"]["inputs"]["positive"] == ["guide", 0] and g["av1"]["inputs"]["video_latent"] == ["key1", 2]
    assert g["guider1"]["inputs"]["positive"] == ["key1", 0] and g["crop"]["inputs"]["positive"] == ["key1", 0]
    assert g["key2"]["inputs"]["positive"] == ["crop", 0] and g["av2"]["inputs"]["video_latent"] == ["key2", 2]
    assert g["guider2"]["inputs"]["negative"] == ["key2", 1]
    assert g["crop2"]["inputs"] == {"positive": ["key2", 0], "negative": ["key2", 1], "latent": ["split", 0]}
    assert g["decode"]["inputs"]["samples"] == ["crop2", 2]  # the guide frames are cut off before decoding
    for node in g.values():  # every reference points at a node that exists
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, v


# --- the engine ------------------------------------------------------------------------------------------------------


def test_ltx_is_the_default_engine_and_scail_keeps_its_own_path(monkeypatch):
    monkeypatch.setattr(recast_ltx, "run", lambda ctx: "ltx")
    monkeypatch.setattr(recast, "run_scail", lambda ctx: "scail")

    class Ctx:
        def __init__(self, params):
            self.params = params
    assert recast.run(Ctx({})) == "ltx"
    assert recast.run(Ctx({"engine": "LTX"})) == "ltx"
    assert recast.run(Ctx({"engine": "scail"})) == "scail"
    with pytest.raises(ValueError, match="Unknown Recast engine"):
        recast.run(Ctx({"engine": "wan"}))
    spec = TASKS["recast"]
    assert spec.stages_of({}) is recast_ltx.STAGES and spec.stages_of({"engine": "scail"}) is recast.STAGES
    assert spec.stages_of({"guide": "pose"}) is person_swap.STAGES  # the pose path is opt-in for now
    assert spec.stages_of({"engine": "wan"}) is recast_ltx.STAGES  # never raises at submit; the job says what's wrong
    plain = TaskSpec([Stage("A", 1)], lambda ctx: {})
    assert plain.stages_of({"engine": "scail"}) == plain.stages


# --- a cast member's sheet -------------------------------------------------------------------------------------------


def test_a_cast_members_sheet_is_used_when_they_have_one(tmp_path):
    d = tmp_path / "cast"
    d.mkdir()
    (d / "c1_sheet_ab12.jpg").write_bytes(b"sheet")
    assert recast_ltx.cast_sheet({"sheet": {"file": "c1_sheet_ab12.jpg"}}, d) == d / "c1_sheet_ab12.jpg"
    assert recast_ltx.cast_sheet({"sheet": {"file": "missing.jpg"}}, d) is None
    assert recast_ltx.cast_sheet({"sheet": {"file": "../c1_sheet_ab12.jpg"}}, d) is None  # never outside the cast folder
    assert recast_ltx.cast_sheet({"file": "c1.png"}, d) is None  # no sheet yet
    assert recast_ltx.cast_sheet(None, d) is None


# --- a whole Recast on LTX, ComfyUI faked -----------------------------------------------------------------------------

W, H, N = 320, 192, 25
BLUE, GREEN = (30, 40, 200), (40, 190, 60)


def colour(img, c):
    return np.abs(img.astype(int) - np.array(c)).max(-1) < 60


def person_x(i):
    """Off screen until frame 10, then walking in through the left edge, 10 px a frame."""
    return None if i < 10 else -20 + 10 * (i - 10)


def source():
    yy, xx = np.mgrid[:H, :W]
    bg = np.stack([110 + 30 * np.sin(xx / 9.0), 115 + 20 * np.cos(yy / 7.0), 105 + 25 * np.sin((xx - yy) / 11.0)], -1).astype(np.uint8)
    frames, masks = [], []
    for i in range(N):
        f, x = bg.copy(), person_x(i)
        m = np.zeros((H, W), bool) if x is None else box(H, W, x, 36, x + 40, 176)
        f[m] = BLUE
        frames.append(f)
        masks.append(m)
    return frames, masks


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


DECOY = (slice(40, 110), slice(880, 990))  # someone else LTX drew, far from the person (LTX's own pixels)


class _Ctx:
    def __init__(self, store, sid, params, tmp):
        from pathlib import Path
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.client, self.ran, self.tmp, self.stages, self.labels = _Client(), [], Path(tmp), [], []

    def stage(self, label):
        self.stages.append(label)

    def check(self):
        pass

    def progress(self, frac):
        pass

    def commercial_safe(self):
        return False

    def run(self, graph):
        self.ran.append(graph)
        out = self.tmp / f"job{len(self.ran)}"
        out.mkdir()
        types = {n["class_type"] for n in graph.values()}
        if "TextEncodeQwenImage21" in types:  # the frame editor: the character over the person
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB")).copy()
            f[colour(f, BLUE)] = GREEN
            Image.fromarray(f).save(p := out / "plate_00001_.png")
            return _Job({"PU_Plate": [_Out(p)]})
        if "LTXAddVideoICLoRAGuide" in types:  # LTX: every pixel re-rendered, the person as a bulkier green costume
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Control"]["inputs"]["file"])):
                g = np.clip(f.astype(int) + (14, -6, 8), 0, 255).astype(np.uint8)
                person = colour(f, BLUE)
                if person.any():
                    ys, xs = np.nonzero(person)
                    pad = (xs.max() - xs.min()) // 4
                    g[box(*person.shape, xs.min() - pad, ys.min() - pad // 2, xs.max() + pad, ys.max() + 2)] = GREEN
                g[DECOY] = GREEN
                Image.fromarray(g).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Out": files})
        if "SAM3_Detect" in types:  # SAM 3.1 on the render's anchor frame: "person" finds everyone green, one by one
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB"))
            label = graph["PU_Label"]["inputs"]["text"].split(":")[0]
            self.labels.append(label)
            green = colour(f, GREEN) if label == "person" else np.zeros(f.shape[:2], bool)
            decoy = np.zeros_like(green)
            decoy[round(DECOY[0].start * f.shape[0] / 896):, round(DECOY[1].start * f.shape[1] / 1024):] = True
            outs = []
            for j, m in enumerate((green & ~decoy, green & decoy)):
                Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"mask_{j + 1:05d}_.png")
                outs.append(_Out(p))
            return _Job({"PU_Mask": outs})
        if "SAM3_VideoTrack" in types:  # SAM 3.1 following them: everything green (the decoy too: the task drops it)
            seed = np.asarray(Image.open(graph["PU_First"]["inputs"]["image"]).convert("L")) > 127
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                m = colour(f, GREEN)
                if i == 0:
                    assert seed.any() and (seed & m).sum() > 0.8 * seed.sum()  # seeded on the new character
                Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"mask_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Mask": files})
        raise AssertionError(f"unexpected graph {sorted(types)}")


def _sheet_png() -> bytes:
    """A two-view sheet on a plain backdrop: front | back."""
    im = Image.new("RGB", (800, 600), (205, 202, 198))
    d = ImageDraw.Draw(im)
    for x0 in (130, 530):
        d.ellipse([x0 + 40, 40, x0 + 100, 120], fill=(200, 160, 130))
        d.rectangle([x0, 120, x0 + 140, 330], fill=GREEN)
        d.rectangle([x0 + 20, 330, x0 + 60, 560], fill=(50, 50, 60))
        d.rectangle([x0 + 80, 330, x0 + 120, 560], fill=(50, 50, 60))
    d.line([(400, 0), (400, 600)], fill=(255, 255, 255), width=8)
    b = BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def shot(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    frames, masks = source()
    src = tmp_path / "walk.mov"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(src)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    sid = store.add_shot(src, "Walk")["id"]
    up = store.add_upload(_sheet_png(), "sheet.png")["id"]
    monkeypatch.setattr(common, "MAX_FRAMES", 9)  # a shot longer than the window
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **k: masks)
    monkeypatch.setattr(common, "FrameWriter", _Writer)
    monkeypatch.setattr(common, "outputs_from_master", lambda *a, **k: [])
    monkeypatch.setattr(recast_ltx, "make_viewer", lambda *a, **k: None)
    _Writer.made = {}
    return store, sid, up


def test_a_whole_recast_on_ltx_offline(shot, tmp_path):
    store, sid, up = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"frame": 14, "points": [], "text": "man", "ref_upload_id": up, "describe": "a man in a green suit",
                            "guide": "depth"},
               tmp_path / "jobs")
    res = recast.run(ctx)  # no engine given: LTX
    assert ctx.stages == [s.label for s in recast_ltx.STAGES]
    d, qa = res["details"], res["details"]["_qa"]
    assert d["engine"] == "ltx" and res["label"] == "Replace a person"
    # The window: where he is on screen most (he walks in at frame 10, whole from 12); the take says which part it is.
    assert qa["window_start"] == 12 and qa["window_chosen"] == "auto" and ctx.params["start_frame"] == 12
    assert d["made_from"] == "0.4 s from 0.5 s in, of 1.0 s" and d["frames"] == 9
    # The anchor: on the first window frame he touches the edge; eight frames on he is whole.
    assert qa["anchor_frame"] == 8 and qa["guide"] == "frame 8"
    paint = next(g for g in ctx.ran if "PU_Prompt" in g and "TextEncodeQwenImage21" in g["PU_Prompt"]["class_type"])
    text = paint["PU_Prompt"]["inputs"]["prompt"]
    assert text.startswith("Turn the man walking in the middle of <image1> into the character from the character sheet in "
                           "<image2> (a man in a green suit)")  # where he is on the anchor frame of the crop
    assert Image.open(paint["PU_Frame2"]["inputs"]["image"]).size == (800, 600)  # the whole sheet, every view
    assert qa["reference"] == "a 2-view character sheet" and qa["views"] == 2
    drive = next(g for g in ctx.ran if "PU_Control" in g)
    assert drive["key1"]["inputs"]["frame_idx"] == 8 and drive["key2"]["inputs"]["frame_idx"] == 8
    assert drive["key1"]["inputs"]["strength"] == recast_ltx.GUIDE_STRENGTH == 0.7  # no frozen copy of the painted frame
    # Which way he faces on the anchor frame (a face?) picks the sheet's view; then the new character is found there.
    assert qa["new_seed"] == "person" and qa["paint_gate"] == ["ok"]
    # the anchor's facing; the paint gate (who, which way, a phone); then the new character in the render
    assert ctx.labels == ["face", "person", "face", "phone", "person"]
    assert qa["sheet_view"] in ("front", "back", None)
    tracks = [g for g in ctx.ran if "track" in g and g["track"]["class_type"] == "SAM3_VideoTrack"]
    assert len(tracks) == 2 and all(g["PU_Mask"]["inputs"]["filename_prefix"].startswith("pickups/recast_track/") for g in tracks)
    assert drive["ctl"]["class_type"] == "DA3Render"  # depth from the shot
    assert drive["PU_Prompt"]["inputs"]["text"].startswith("A man in a green suit walks")
    assert d["models"] == ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3", "SAM 3.1"]

    # The composite: only the matte changed, at the source's size, and the receipt counts exactly that.
    assert d["pixels_changed_outside_selection"] == 0
    out = _Writer.made["master.mov"].frames
    src = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    assert len(out) == 9 and out[0].shape == (H, W, 3)
    x0, y0, cw, ch = qa["crop"]
    assert (x0, y0) == (0, 0) and cw < W  # a crop around where he walks
    for i in range(9):
        assert np.array_equal(out[i][:, x0 + cw:], src[i][:, x0 + cw:])  # outside the crop: the footage
        x = person_x(12 + i)
        assert colour(out[i][100:110, x + 12:x + 28], GREEN).all()  # the new character where he was
        if x - 6 > 0:
            assert colour(out[i][100:110, x - 6:x - 2], GREEN).all()  # the bulkier costume, past his old outline
        (dy, dx), (lw, lh) = DECOY, qa["work_size"]
        sy, sx = slice(dy.start * ch // lh + 1, dy.stop * ch // lh), slice(dx.start * cw // lw + 1, dx.stop * cw // lw)
        assert np.array_equal(out[i][sy, sx], src[i][sy, sx])  # someone else LTX drew is not pasted
        assert np.array_equal(out[i][150:, x + 60:x0 + cw], src[i][150:, x + 60:x0 + cw])  # LTX's re-rendered ground isn't
    touched = sum(int((_Writer.made["receipt.mp4"].frames[i] != out[i]).any(-1).sum()) for i in range(9))
    receipt = res["receipt"]
    assert receipt["ai"] == pytest.approx(100 * touched / (W * H * 9), abs=0.05) and receipt["ai"] > 0
    assert receipt["original"] + receipt["ai"] == pytest.approx(100, abs=0.02)
    assert qa["coverage"] > 1.1 and qa["empty_new_frames"] == 0 and qa["flicker"] is not None and qa["frames"] == 9
    assert (ctx.vdir / "painted_frame.png").exists()


def test_the_users_start_frame_wins(shot, tmp_path):
    store, sid, up = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"frame": 14, "points": [], "text": "man", "ref_upload_id": up, "start_frame": 14, "grain": "off",
                            "guide": "depth"},
               tmp_path / "jobs")
    res = recast.run(ctx)
    qa = res["details"]["_qa"]
    assert qa["window_start"] == 14 and qa["window_chosen"] == "yours" and res["details"]["grain"] == "off"
    assert qa["grain_added"] is None and res["details"]["pixels_changed_outside_selection"] == 0


def test_its_workflows_have_plain_names():
    from pickups.app.workflow_titles import workflow_title
    paint = graphs.image_edit("anchor.png", "Replace the man", refs=["sheet.png"], prefix="pickups/recast_paint")
    render = graphs.drive("c.mp4", "f.png", "p", 24, 1664, 1152, control_mode="depth", prefix="pickups/recast_ltx", first_at=16)
    track = graphs.track_mask("fwd.mp4", "pick_mask_0.png", prefix="pickups/recast_track")
    assert workflow_title(paint, "recast_paint", 20) == "Paint · Qwen-Image 2.1 · 1 reference"
    assert workflow_title(render, "recast_ltx", 20) == "Recast · LTX-2.5 Union Control · depth"
    assert workflow_title(track, "recast_track", 20) == "Track · SAM 3.1 · forward"  # from the anchor, not the clicked frame


def test_a_vehicle_is_recast_in_a_cars_words():
    assert [recast_ltx.vehicle_noun(t) for t in ("car", "the SUV", "the dark blue suv", "a truck", "man", "person", "")] == [
        "car", "suv", "suv", "truck", None, None, None]
    text = recast_ltx.paint_vehicle("suv", "in the middle of", "a vintage olive-green off-roader")
    assert text.startswith("Replace the SUV in the middle of <image1> with the vehicle in <image2> (a vintage olive-green off-roader): "
                           "the same position, size and angle")
    assert "wheels on the road exactly where the SUV's wheels are" in text
    assert recast_ltx.drive_words("walks to the left") == "drives to the left"
    assert recast_ltx.drive_words("moves naturally where they stand").startswith("drives steadily along the road")
    prompt = recast_ltx.ltx_prompt("a vintage off-roader", "drives to the left", "A wet rural road.", vehicle=True)
    assert prompt.startswith("A vintage off-roader drives to the left. A wet rural road. It is lit by the scene's own light")
    assert "wheel" not in prompt  # naming the wheels had LTX steer them out of their arches


def test_a_sheets_back_view_is_cut_for_a_figure_seen_from_behind(tmp_path):
    """Front | back | close-up on a plain backdrop: "back" is the second standing figure from the left."""
    sheet = np.full((300, 600, 3), 225, np.uint8)
    sheet[40:280, 60:130] = (200, 30, 30)    # front: red
    sheet[40:280, 260:330] = (30, 30, 200)   # back: blue
    sheet[60:200, 430:560] = (30, 200, 30)   # close-up: a head, not a whole figure
    p = tmp_path / "sheet.png"
    Image.fromarray(sheet).save(p)
    back = recast_ltx.sheet_view(p, "back")
    front = recast_ltx.sheet_view(p, "front")
    assert back is not None and front is not None
    assert np.asarray(back).reshape(-1, 3).mean(0)[2] > np.asarray(back).reshape(-1, 3).mean(0)[0]  # blue
    assert np.asarray(front).reshape(-1, 3).mean(0)[0] > np.asarray(front).reshape(-1, 3).mean(0)[2]  # red
    single = tmp_path / "one.png"
    Image.fromarray(sheet[:, :200].copy()).save(single)
    assert recast_ltx.sheet_view(single, "back") is None  # one figure: not a sheet


def test_paint_gate_catches_a_misplaced_person_a_wrong_facing_and_a_phone(monkeypatch, tmp_path):
    old = np.zeros((100, 60), bool)
    old[20:90, 20:40] = True
    found: dict[str, list[np.ndarray]] = {}
    monkeypatch.setattr(recast_ltx.common, "select_instances", lambda ctx, png, label, scale, out: found.get(label, []))
    png = tmp_path / "p.png"
    elsewhere = np.zeros_like(old)
    elsewhere[20:90, 42:58] = True
    found["person"] = [elsewhere]
    assert recast_ltx.paint_problem(None, png, old, "", "a knight", tmp_path) == "not where the person stands"
    found["person"] = [old.copy()]
    face = np.zeros_like(old)
    face[22:30, 25:35] = True
    found["face"] = [face]  # painted facing the camera; the footage is a back view
    assert recast_ltx.paint_problem(None, png, old, "back", "a knight", tmp_path) == "facing the wrong way"
    assert recast_ltx.paint_problem(None, png, old, "front", "a knight", tmp_path) == ""
    phone = np.zeros_like(old)
    phone[55:60, 40:43] = True
    found["phone"] = [phone]
    assert recast_ltx.paint_problem(None, png, old, "", "a noblewoman", tmp_path) == "holding a phone"
    assert recast_ltx.paint_problem(None, png, old, "", "a reporter holding a phone", tmp_path) == ""


def test_paint_change_check_catches_the_wrong_person(tmp_path):
    """Told to recast the man, the painter changed the child beside him: caught before LTX follows it."""
    anchor = np.full((120, 200, 3), 120, np.uint8)
    man = np.zeros((120, 200), bool)
    man[10:120, 120:180] = True
    anchor[man] = (40, 60, 40)
    anchor[60:120, 20:50] = (200, 160, 40)  # the child, well to the left
    Image.fromarray(anchor).save(tmp_path / "a.png")

    def painted(edit):
        im = anchor.copy()
        edit(im)
        Image.fromarray(im).save(tmp_path / "p.png")
        return tmp_path / "p.png"

    assert recast_ltx.paint_changes(tmp_path / "a.png", painted(lambda im: None), man) == "the person wasn't changed"
    child_swapped = painted(lambda im: im.__setitem__((slice(40, 120), slice(10, 60)), (90, 30, 120)))
    assert recast_ltx.paint_changes(tmp_path / "a.png", child_swapped, man) == "the person wasn't changed"

    def both(im):
        im[man] = (150, 90, 120)
        im[40:120, 10:60] = (90, 30, 120)
    assert recast_ltx.paint_changes(tmp_path / "a.png", painted(both), man) == "someone else was changed"
    assert recast_ltx.paint_changes(tmp_path / "a.png", painted(lambda im: im.__setitem__(man, (150, 90, 120))), man) == ""


def _walker(h=200, w=400, x=60, top=20, tall=170):
    m = np.zeros((h, w), bool)
    m[top:top + tall, x:x + 40] = True
    return m


def _face(person, offset):
    """A face on the person's head: offset 0 square to the camera, +1 at the right edge of the head (profile right)."""
    f = np.zeros_like(person)
    ys = np.flatnonzero(person.any(axis=1))
    xs = np.flatnonzero(person[ys[0]:ys[0] + 10].any(axis=0))
    cx = (xs[0] + xs[-1]) / 2 + offset * (xs[-1] - xs[0]) / 2
    f[ys[0] + 4:ys[0] + 18, int(cx) - 5:int(cx) + 5] = True
    return f


def test_a_face_square_to_the_camera_and_one_in_profile_are_told_apart():
    p = _walker()
    assert abs(recast_ltx.face_turn([_face(p, 0.0)], p)) < recast_ltx.SIDE_TURN
    assert recast_ltx.face_turn([_face(p, 0.8)], p) > recast_ltx.SIDE_TURN
    assert recast_ltx.face_turn([_face(p, -0.8)], p) < -recast_ltx.SIDE_TURN
    assert recast_ltx.face_turn([], p) is None


def test_a_walk_across_the_picture_is_sideways_and_a_walk_towards_the_camera_is_not():
    across = [_walker(x=40 + 30 * i) for i in range(9)]
    assert recast_ltx.walk_side(across) == "right"
    assert recast_ltx.walk_side(across[::-1]) == "left"
    towards = [_walker(top=100 - 10 * i, tall=60 + 12 * i) for i in range(9)]
    assert recast_ltx.walk_side(towards) == ""


def test_a_woman_walking_across_in_profile_is_painted_in_profile(monkeypatch, tmp_path):
    """The castle hall: one front-facing still, a woman walking right in profile. Told nothing, the painter turned her
    to the camera; now the instruction says which way she faces, and the gate rejects a face square to the camera."""
    track = [_walker(x=40 + 30 * i) for i in range(9)]
    old = track[0]
    found: dict[str, list[np.ndarray]] = {"face": [_face(old, 0.7)]}
    monkeypatch.setattr(recast_ltx.common, "select_instances", lambda ctx, png, label, scale, out: found.get(label, []))
    base = recast_ltx.paint_instruction("woman", "on the left of", True, False, "a noblewoman")
    view, text, cut = recast_ltx.paint_view(None, tmp_path / "a.png", old, track, tmp_path / "c.png", 1, base, tmp_path)
    assert view == "side-right" and cut is None
    assert "in profile, facing right and walking to the right of the picture" in text and "not towards the camera" in text
    assert text.endswith(recast_ltx.KEEP_REST)
    # Seen from behind (no face): turned away as well, still facing the way she walks.
    found["face"] = []
    view, text, _ = recast_ltx.paint_view(None, tmp_path / "a.png", old, track, tmp_path / "c.png", 1, base, tmp_path)
    assert view == "back-right" and "turned partly away from the camera" in text and "facing away" in text
    # The gate: a face square to the camera, or turned the other way, is the wrong way; profile or no face is right.
    assert recast_ltx.view_problem([_face(old, 0.0)], old, "side-right") == "facing the wrong way"
    assert recast_ltx.view_problem([_face(old, -0.7)], old, "side-right") == "facing the wrong way"
    assert recast_ltx.view_problem([_face(old, 0.7)], old, "side-right") == ""
    assert recast_ltx.view_problem([], old, "back-right") == ""
    assert recast_ltx.view_problem(None, old, "side-right") == ""  # faces couldn't be looked for: no verdict
    # Standing still, a face square to the camera: nothing about a side.
    found["face"] = [_face(old, 0.0)]
    view, text, _ = recast_ltx.paint_view(None, tmp_path / "a.png", old, [old] * 9, tmp_path / "c.png", 1, base, tmp_path)
    assert view == "" and text == base


def test_a_person_cut_off_by_the_frame_is_painted_cut_off():
    m = np.zeros((100, 60), bool)
    m[40:100, 20:40] = True  # walking away, the bottom of the picture cuts her off at the hips
    text = recast_ltx.cut_words(m)
    assert "bottom edge of the picture cut them off" in text and "do not shrink them" in text
    whole = np.zeros((100, 60), bool)
    whole[20:90, 20:40] = True
    assert recast_ltx.cut_words(whole) == ""


def test_someone_walking_away_is_painted_from_behind_even_if_a_face_is_found(monkeypatch, tmp_path):
    """A cap seen from behind answered to "face": the knight was painted facing the camera."""
    track = [_walker(top=20 + 6 * i, tall=170 - 12 * i) for i in range(9)]  # smaller and smaller
    old = track[0]
    monkeypatch.setattr(recast_ltx.common, "select_instances", lambda ctx, png, label, scale, out: [_face(old, 0.0)])
    base = recast_ltx.paint_instruction("man", "in the middle of", True, False, "a knight")
    view, text, _ = recast_ltx.paint_view(None, tmp_path / "a.png", old, track, tmp_path / "c.png", 1, base, tmp_path)
    assert view == "back" and "facing away" in text
    assert recast_ltx.walks_away(track) and not recast_ltx.walks_away(track[::-1])


def test_no_face_on_someone_who_turns_and_walks_towards_the_camera_asserts_nothing(monkeypatch, tmp_path):
    """The man on the pier looks out over the water in profile (no face found), then walks towards the camera."""
    track = [_walker(top=60 - 5 * i, tall=80 + 12 * i) for i in range(9)]  # bigger and bigger
    monkeypatch.setattr(recast_ltx.common, "select_instances", lambda ctx, png, label, scale, out: [])
    base = recast_ltx.paint_instruction("man", "in the middle of", True, False, "a knight")
    view, text, _ = recast_ltx.paint_view(None, tmp_path / "a.png", track[0], track, tmp_path / "c.png", 1, base, tmp_path)
    assert view == "" and text == base
    assert recast_ltx.walks_away(track, towards=True)
