"""Recast's masks: only the chosen person is re-rendered, the new character's matte survives non-humans and bulky
costumes, the QA numbers measure what they say, and a whole run (fake ComfyUI) keeps the receipt honest."""

import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import graphs
from pickups.tasks import recast

# --- which drive person SCAIL-2 re-renders -------------------------------------------------------------------------


def box(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_drive_rank_is_the_chosen_persons_place_from_the_left():
    left, mid, right = box(100, 300, 20, 10, 70, 95), box(100, 300, 120, 10, 170, 95), box(100, 300, 220, 10, 270, 95)
    old = box(100, 300, 118, 8, 168, 96)  # the tracked person, a little off SAM's outline
    assert recast.drive_rank([right, left, mid], old) == 1  # SAM's order doesn't matter; left to right does
    assert recast.drive_rank([right, left], box(100, 300, 222, 12, 268, 95)) == 1
    assert recast.drive_rank([mid], old) is None  # one person: everyone, as before
    assert recast.drive_rank([left, mid, right], np.zeros((100, 300), bool)) is None  # not on the first frame
    assert recast.drive_rank([left, mid], box(100, 300, 15, 5, 175, 98)) is None  # the selection is both of them
    five = [box(100, 300, x, 10, x + 40, 90) for x in (0, 60, 120, 180, 240)]
    assert recast.drive_rank(five, five[2]) is None  # more people than SCAIL-2 tracks (4)


# --- the new character's matte -------------------------------------------------------------------------------------


def scene(h=200, w=200, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    bg = np.stack([90 + 40 * np.sin(xx / 7.0), 100 + 30 * np.cos(yy / 5.0), 110 + 20 * np.sin((xx + yy) / 9.0)], -1)
    return np.clip(bg + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)


def test_a_bulky_costume_sam_cant_name_is_not_clipped_to_the_old_outline():
    orig = scene()
    old = box(200, 200, 80, 40, 120, 180)
    gen = orig.copy()
    costume = box(200, 200, 60, 30, 140, 185)  # an inflatable T-rex: wider and taller than the performer
    gen[costume] = (40, 190, 60)
    new, subject, hit = recast.new_matte(np.zeros_like(old), old, gen, orig)
    assert not hit  # SAM found nothing...
    assert (new & costume).sum() > 0.9 * (costume & ~old).sum()  # ...the changed pixels still bring the whole costume
    assert np.array_equal(subject, old | new)  # dehalo works around the costume, not just the old outline
    assert not (new & ~recast.composite.grow(costume, 3)).any()


def test_a_partner_scail_left_alone_is_not_pasted():
    orig = scene(seed=1)
    old = box(200, 200, 60, 40, 100, 180)
    partner = box(200, 200, 100, 45, 150, 180)  # hugging: touches the chosen person
    gen = orig.copy()
    gen[old] = (200, 60, 60)  # SCAIL-2 re-rendered only the chosen person (object_indices)
    found = old | partner  # SAM's "person" union sees one blob
    new, _, hit = recast.new_matte(found, old, gen, orig)
    assert hit
    far_partner = partner & ~recast.composite.grow(old, 6)
    assert (new & far_partner).sum() < 0.02 * far_partner.sum()


@pytest.mark.parametrize("light", [(236, 236, 232), (200, 205, 210)])  # a white spacesuit, chrome
def test_a_white_or_chrome_costume_sam_cant_name_is_not_clipped(light):
    orig = scene(seed=3)
    old = box(200, 200, 80, 40, 120, 180)
    costume = box(200, 200, 60, 30, 140, 185)  # 20 px wider than the performer, lighter than the background
    gen = orig.copy()
    rng = np.random.default_rng(3)
    gen[costume] = np.clip(np.array(light) + rng.normal(0, 3, (int(costume.sum()), 3)), 0, 255).astype(np.uint8)
    glow = recast.composite.grow(costume, 4) & ~costume
    gen[glow] = (gen[glow] * 0.4 + 235 * 0.6).astype(np.uint8)
    new, subject, hit = recast.new_matte(np.zeros_like(old), old, gen, orig)
    assert not hit
    beyond = costume & ~old
    assert (new & beyond).sum() > 0.95 * beyond.sum()  # whitish and lighter, but far wider than a glow: costume
    assert not (new & ~recast.composite.grow(costume, 6)).any()
    assert np.array_equal(subject, old | new)


def test_a_partner_scail_redrew_a_little_off_is_not_taken_for_the_new_character():
    bg = scene(seed=4)
    old = box(200, 200, 60, 40, 100, 180)
    partner = box(200, 200, 100, 45, 150, 180)  # hugging
    orig = bg.copy()
    orig[partner], orig[old] = (180, 150, 60), (60, 60, 160)
    moved = np.roll(np.roll(partner, 8, 0), 3, 1)  # SCAIL-2 regenerates every pixel: the partner comes back a little off
    gen = bg.copy()
    gen[moved], gen[old] = (180, 150, 60), (200, 60, 60)
    far_partner = partner & ~recast.composite.grow(old, 6)
    leaked, _, _ = recast.new_matte(old | moved, old, gen, orig)  # SAM's union sees one blob
    new, _, hit = recast.new_matte(old | moved, old, gen, orig, track=old.copy())  # the new character, tracked alone
    assert hit and (new & old).sum() == old.sum()
    assert (new & far_partner).sum() == 0 < (leaked & far_partner).sum()


def test_scails_glow_and_background_drift_are_not_new_character():
    rng = np.random.default_rng(2)
    orig = scene(seed=2)
    old = box(200, 200, 80, 40, 120, 180)
    gen = np.clip(orig.astype(int) + rng.normal(0, 4, orig.shape), 0, 255).astype(np.uint8)  # SCAIL's own drift
    gen[old] = (60, 60, 200)
    ring = recast.composite.grow(old, 5) & ~old
    gen[ring] = (gen[ring] * 0.3 + 235 * 0.7).astype(np.uint8)  # a whitish glow round the new person
    new, subject, hit = recast.new_matte(old.copy(), old, gen, orig)
    assert hit and (new & old).sum() == old.sum()
    assert (new & ring).sum() < 0.35 * ring.sum()  # the glow is dehalo's job, not part of the character
    assert not (new & ~recast.composite.grow(old, 8)).any()  # drift far away adds nothing


# --- QA numbers ----------------------------------------------------------------------------------------------------


def disk(h, w, r):
    yy, xx = np.mgrid[:h, :w]
    return np.hypot(yy - h / 2, xx - w / 2) < r


def test_halo_lift_sees_a_light_ring_and_nothing_else():
    img = np.full((200, 200, 3), 100, np.uint8)
    m = disk(200, 200, 40)
    img[m] = (40, 60, 120)
    assert abs(recast.halo_lift(img, m)) < 1
    ring = recast.composite.grow(m, 12) & ~recast.composite.grow(m, 6)
    img[ring] = 170
    assert recast.halo_lift(img, m) > 50
    assert recast.halo_lift(img, np.zeros_like(m)) is None


def test_coverage_change_and_flicker():
    old, new = box(50, 50, 10, 10, 30, 30), box(50, 50, 8, 8, 34, 32)
    assert recast.coverage(new, old) == pytest.approx(new.sum() / old.sum())
    assert recast.coverage(new, np.zeros_like(old)) is None
    a, b = np.zeros((50, 50, 3), np.uint8), np.full((50, 50, 3), 6, np.uint8)
    assert recast.frame_change(a, b, old) == 6 / 4 and recast.frame_change(a, b, np.zeros_like(old)) is None  # flat: contrast floor 4
    # A bright, detailed suit that moves like a dark sweater did scores the same: change is measured against contrast.
    rng = np.random.default_rng(0)
    dark = rng.integers(10, 30, (50, 50, 3)).astype(np.uint8)
    bright = (dark.astype(np.int32) * 10).clip(0, 255).astype(np.uint8)
    shift = lambda x: np.roll(x, 1, axis=1)  # noqa: E731 - the same small motion in both
    assert recast.frame_change(bright, shift(bright), old) == pytest.approx(recast.frame_change(dark, shift(dark), old), rel=0.1)
    qa = recast.QA()
    for i in range(5):  # the new character changes twice as much frame to frame as the real person did
        qa.add({"out": np.full((50, 50, 3), 10 * (i % 2), np.uint8), "src": np.full((50, 50, 3), 5 * (i % 2), np.uint8),
                "new": new, "old": old, "hit": i != 3, "coverage": 1.3, "halo": 2.0})
    qa.add(None)  # past the end of the render
    s = qa.summary()
    assert s == {"coverage": 1.3, "halo": 2.0, "flicker": 2.0, "empty_new_frames": 1, "frames": 5}


def test_close_ups_render_bigger():
    assert recast.performance_size(600, 1000) == (832, 512, 832)
    assert recast.performance_size(2400, 1300)[0] == 832
    long, w, h = recast.performance_size(2600, 1400)
    assert long == recast.SCAIL_LONG_CLOSE == 1024 and max(w, h) == 1024 and w % 32 == 0 and h % 32 == 0


# --- graphs --------------------------------------------------------------------------------------------------------


def test_scail2_takes_our_reference_mask_and_one_person():
    g = graphs.scail2("crop.mp4", "ref.png", "A knight", object_indices="1", reference_mask="ref_mask.png")
    assert "rtrack" not in g and "rlabel" not in g
    assert g["PU_RefMask"]["inputs"]["image"] == "ref_mask.png"
    assert g["rmask"] == {"class_type": "ImageToMask", "inputs": {"image": ["PU_RefMask", 0], "channel": "red"}}
    cm = g["cmask"]["inputs"]
    assert cm["ref_track_data"] == ["rmask", 0] and cm["object_indices"] == "1" and cm["sort_by"] == "left_to_right"
    assert g["cvenc"]["inputs"]["image"] == ["PU_Reference", 0]  # CLIP-vision reads the one cropped figure
    old = graphs.scail2("crop.mp4", "ref.png", "A knight")  # scripts still get the template's SAM reference track
    assert old["cmask"]["inputs"]["ref_track_data"] == ["rtrack", 0] and "PU_RefMask" not in old
    assert set(old) - set(g) == {"rtrack", "rlabel"} and set(g) - set(old) == {"PU_RefMask", "rmask"}


def test_select_any_finds_every_label_in_one_job():
    g = graphs.select_any("gen.mp4", recast.NEW_LABELS, prefix="pickups/recast_new")
    outs = [k for k in g if k.startswith("PU_Mask")]
    assert outs == ["PU_Mask", "PU_Mask1", "PU_Mask2", "PU_Mask3", "PU_Mask4"]
    assert [g[f"PU_Label{k}"]["inputs"]["text"] for k in ("", "1", "2", "3", "4")] == [graphs.counted(x) for x in recast.NEW_LABELS]
    assert len({g[k]["inputs"]["filename_prefix"] for k in outs}) == 5


# --- a whole Recast, ComfyUI faked ---------------------------------------------------------------------------------

W, H, N = 320, 192, 9
BLUE, GOLD, GREEN = (30, 40, 200), (220, 200, 40), (40, 190, 60)


def colour(img, c):
    d = np.abs(img.astype(int) - np.array(c)).max(-1)
    return d < 60


def source_frames():
    frames, masks = [], []
    yy, xx = np.mgrid[:H, :W]
    bg = np.stack([110 + 30 * np.sin(xx / 9.0), 115 + 20 * np.cos(yy / 7.0), 105 + 25 * np.sin((xx - yy) / 11.0)], -1).astype(np.uint8)
    for i in range(N):
        f = bg.copy()
        f[40:172, 84:118] = GOLD  # the partner, on the left, hugging
        x = 120 + i
        f[36:176, x:x + 40] = BLUE  # the chosen person
        frames.append(f)
        masks.append(box(H, W, x, 36, x + 40, 176))
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


class _Ctx:
    def __init__(self, store, sid, params, tmp, sam_names_it=True, refuse_mask=False, tracks=True):
        from pathlib import Path
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.client, self.ran, self.tmp, self.sam_names_it = _Client(), [], Path(tmp), sam_names_it
        self.refuse_mask = refuse_mask  # a ComfyUI whose SCAIL2ColoredMask won't take a plain MASK
        self.tracks = tracks  # SAM 3.1 can follow the new character from the old person's first-frame mask

    def stage(self, label):
        pass

    def check(self):
        pass

    def progress(self, frac):
        pass

    def run(self, graph):
        from comfy_sdk import InvalidWorkflow

        from pickups.tasks import common
        self.ran.append(graph)
        if self.refuse_mask and "PU_RefMask" in graph:
            raise InvalidWorkflow("Prompt outputs failed validation: SCAIL2ColoredMask ref_track_data")
        out = self.tmp / f"job{len(self.ran)}"
        out.mkdir()
        types = {n["class_type"] for n in graph.values()}
        if "WanSCAILToVideo" in types:  # the new character: a bulkier green costume with a light glow round it
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Drive"]["inputs"]["file"])):
                g = f.copy()
                person = colour(f, BLUE)
                ys, xs = np.nonzero(person)
                pad = (xs.max() - xs.min()) // 4
                costume = box(*person.shape, xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + 2)
                glow = recast.composite.grow(costume, 4) & ~costume
                g[glow] = (g[glow] * 0.4 + 230 * 0.6).astype(np.uint8)
                g[costume] = GREEN
                Image.fromarray(g).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Out": files})
        if "PU_Frame" in graph:  # SAM 3.1 on the drive's first frame: one mask per person
            f = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB"))
            outs = []
            for j, c in enumerate((GOLD, BLUE)):
                Image.fromarray(colour(f, c).astype(np.uint8) * 255).save(p := out / f"mask_{j + 1:05d}_.png")
                outs.append(_Out(p))
            return _Job({"PU_Mask": outs})
        if "PU_First" in graph:  # SAM 3.1 tracking the new character from where the old person stood
            if not self.tracks:
                raise RuntimeError("SAM3_VideoTrack: no object in the initial mask")
            seed = np.asarray(Image.open(graph["PU_First"]["inputs"]["image"]).convert("L")) > 127
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                if i == 0:
                    assert (seed & colour(f, GREEN)).sum() > 0.9 * seed.sum()  # seeded on the new character, not the partner
                Image.fromarray(colour(f, GREEN).astype(np.uint8) * 255).save(p := out / f"mask_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Mask": files})
        if "SAM3_Detect" in types:  # SAM 3.1 over NEW_LABELS on the render: only "costume" knows a T-rex
            outs = {}
            for k in [n for n in graph if n.startswith("PU_Mask")]:
                label = graph[k.replace("PU_Mask", "PU_Label")]["inputs"]["text"].split(":")[0]  # "costume:8"
                (d := out / k).mkdir()
                files = []
                for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                    m = colour(f, GREEN) if (label == "costume" and self.sam_names_it) else np.zeros(f.shape[:2], bool)
                    Image.fromarray(m.astype(np.uint8) * 255).save(p := d / f"mask_{i + 1:05d}_.png")
                    files.append(_Out(p))
                outs[k] = files
            return _Job(outs)
        raise AssertionError(f"unexpected graph {sorted(types)}")


def _figure_png() -> bytes:
    """A plain-backdrop still of the new character."""
    im = Image.new("RGB", (400, 600), (205, 202, 198))
    d = ImageDraw.Draw(im)
    d.ellipse([170, 40, 230, 120], fill=(200, 160, 130))
    d.rectangle([130, 120, 270, 330], fill=GREEN)
    d.rectangle([150, 330, 190, 560], fill=(50, 50, 60))
    d.rectangle([210, 330, 250, 560], fill=(50, 50, 60))
    from io import BytesIO
    b = BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def shot(tmp_path, monkeypatch):
    import pickups.app.store as st
    from pickups.tasks import common
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    frames, masks = source_frames()
    src = tmp_path / "hug.mov"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(src)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    sid = store.add_shot(src, "Hug")["id"]
    up = store.add_upload(_figure_png(), "trex.png")["id"]
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **k: masks)
    monkeypatch.setattr(common, "FrameWriter", _Writer)
    monkeypatch.setattr(common, "outputs_from_master", lambda *a, **k: [])
    monkeypatch.setattr(recast, "make_viewer", lambda *a, **k: None)
    _Writer.made = {}
    return store, sid, up, masks


@pytest.mark.parametrize("sam_names_it, tracks", [(True, True), (False, True), (True, False), (False, False)])
def test_a_whole_recast_offline(shot, tmp_path, sam_names_it, tracks):
    from pickups.tasks import common
    store, sid, up, masks = shot
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "person", "ref_upload_id": up, "describe": "an inflatable T-rex"},
               tmp_path / "jobs", sam_names_it, tracks=tracks)
    (tmp_path / "jobs").mkdir()
    res = recast.run(ctx)
    d = res["details"]
    assert d["pixels_changed_outside_selection"] == 0
    assert d["reference"] == "figure cut out from its plain backdrop"
    qa = d["_qa"]
    assert qa["object_indices"] == "1" and qa["people_on_first_frame"] == 2  # only the chosen person is re-rendered
    assert qa["reference_mask"] is True
    assert qa["scail_long"] == 832 and qa["frames"] == N and qa["reference"] == d["reference"]
    assert qa["new_track"] is tracks  # a partner shares the crop: the new character is followed as one instance
    if sam_names_it or tracks:
        assert qa["coverage"] > 1.2 and qa["empty_new_frames"] == 0  # the costume is bigger than the performer
        assert qa["halo"] is not None and qa["halo"] < 10  # the fake glow, mostly gone after dehalo
    else:  # SAM had no word for it: only the changed pixels past the old outline count, and the QA says so
        assert 0.3 < qa["coverage"] < 1 and qa["empty_new_frames"] == N and qa["halo"] is None
    assert qa["flicker"] is not None

    scail = next(g for g in ctx.ran if "PU_Drive" in g)
    assert scail["cmask"]["inputs"]["object_indices"] == "1" and scail["cmask"]["inputs"]["ref_track_data"] == ["rmask", 0]
    ref = Image.open(scail["PU_Reference"]["inputs"]["image"])
    drive = Image.open(next(iter(ctx.vdir.glob("work/drive_first.png"))))
    assert abs(ref.width / ref.height - drive.width / drive.height) < 0.02  # padded to SCAIL-2's shape: nothing cropped

    out, src = _Writer.made["master.mov"].frames, list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    assert len(out) == N
    for i in (0, N // 2, N - 1):
        x = 120 + i
        assert colour(out[i][100:110, x - 6:x - 2], GREEN).all()  # costume past the old outline, not clipped
        partner = (slice(60, 160), slice(84, 100))  # the partner, away from the performer: the real footage
        assert np.array_equal(out[i][partner], src[i][partner])
        assert np.array_equal(out[i][:, 200:], src[i][:, 200:])
    receipt = res["receipt"]
    assert receipt["ai"] > 0 and receipt["original"] + receipt["ai"] == pytest.approx(100, abs=0.02)


def test_a_comfyui_that_refuses_our_reference_mask_still_recasts(shot, tmp_path):
    """Checked against ComfyUI 0.27's SCAIL2ColoredMask only: if a newer one refuses a plain MASK (at submit, before any
    GPU work), Recast falls back to SAM's track of the cut-out reference, for everyone in the crop."""
    store, sid, up, masks = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "person", "ref_upload_id": up, "describe": "an inflatable T-rex"},
               tmp_path / "jobs", refuse_mask=True)
    res = recast.run(ctx)
    scails = [g for g in ctx.ran if "PU_Drive" in g]
    assert len(scails) == 2 and "PU_RefMask" in scails[0] and "PU_RefMask" not in scails[1]
    assert scails[1]["cmask"]["inputs"]["ref_track_data"] == ["rtrack", 0] and scails[1]["cmask"]["inputs"]["object_indices"] == ""
    qa = res["details"]["_qa"]
    assert qa["reference_mask"] is False and qa["object_indices"] == ""
    assert res["details"]["pixels_changed_outside_selection"] == 0


def test_too_small_gives_the_height_in_the_shots_own_pixels(shot, tmp_path, monkeypatch):
    """An 8K shot works on a 4K clip: "about N px tall" must be the person in the file the user is looking at."""
    from pickups.tasks import common
    store, sid, up, masks = shot
    real = common.work_clip
    monkeypatch.setattr(common, "work_clip", lambda ctx, out, **kw: real(ctx, out, long_side=W // 2, **kw))
    seen = []

    def stop(pm, frame_height):
        seen.append(frame_height)
        raise RuntimeError("stop here")
    monkeypatch.setattr(common, "require_size", stop)
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"engine": "scail", "frame": 0, "points": [], "text": "person", "ref_upload_id": up}, tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="stop here"):
        recast.run(ctx)
    assert seen == [H]  # the source's height, not the work clip's (H // 2)


def test_one_person_seeds_from_the_chosen_frame_with_its_biggest_match(monkeypatch, tmp_path):
    """"man in green parka" also hit the child in his parka on another frame; tracking both recast the child."""
    from pickups.tasks import common
    man = np.zeros((40, 60), bool)
    man[5:40, 30:55] = True
    child = np.zeros((40, 60), bool)
    child[20:40, 5:15] = True
    seen: list[int] = []
    monkeypatch.setattr(common, "_frame_png", lambda video, f, out: (seen.append(f), out)[1])
    monkeypatch.setattr(common, "select_instances", lambda ctx, png, text, scale, out: [child, man])
    frame, parts = common.seed_frame(None, tmp_path / "v.mp4", 80, 39, "man in green parka", 1.0, tmp_path, one=True)
    assert frame == 39 and len(parts) == 1 and (parts[0] == man).all() and seen == [39]
    bag = np.zeros((40, 60), bool)
    bag[25:30, 52:58] = True
    monkeypatch.setattr(common, "select_instances", lambda ctx, png, text, scale, out: [bag] if text == "handbag" else [child, man])
    frame, parts = common.seed_frame(None, tmp_path / "v.mp4", 80, 39, "woman, handbag", 1.0, tmp_path, one=True)
    assert len(parts) == 2 and (parts[0] == man).all() and (parts[1] == bag).all()  # her bag still goes with her
    monkeypatch.setattr(common, "select_instances", lambda ctx, png, text, scale, out: [child, man])
    assert not common.MANY.search("man in green parka") and common.MANY.search("the couple")
    frame, parts = common.seed_frame(None, tmp_path / "v.mp4", 80, 39, "the couple", 1.0, tmp_path)
    assert len(parts) == 2  # several people: every match, from the frame with the most
