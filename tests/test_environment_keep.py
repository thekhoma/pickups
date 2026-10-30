"""Change weather's "Keep exactly as shot": the words and clicks that say what to keep, detail transfer (the footage's
fine detail on the render's broad light), the matte it is put back in, and a whole run (ComfyUI faked) where the kept
thing comes back as shot, everything else is the render, and the receipt counts exactly the kept share as original."""

import subprocess

import numpy as np
import pytest
from PIL import Image

from pickups import composite
from pickups.tasks import common, environment as env


# --- what to keep ------------------------------------------------------------------------------------------------------


def test_what_to_keep_in_the_prompts_words_and_sams():
    assert env.keep_name("car") == "the car"
    assert env.keep_name("The car.") == "the car"
    assert env.keep_name("  the   blue SUV ") == "the blue SUV"
    assert env.keep_name("SUV") == "the SUV"  # an all-caps word stays as it is
    assert env.keep_name("my bike") == "my bike"
    assert env.keep_name("") == ""
    assert env.sam_words("the car") == "car" and env.sam_words("A red van.") == "red van" and env.sam_words("car") == "car"


def test_a_click_with_no_words_is_named_by_where_it_is():
    m = np.zeros((90, 160), bool)
    m[30:60, 10:40] = True
    assert env.where_name(m) == "the object on the left of the frame"
    m[:] = False
    m[30:60, 60:100] = True
    assert env.where_name(m) == "the object in the middle of the frame"
    assert env.where_name(np.zeros((90, 160), bool)) == "the selected object"


def test_the_whole_frame_is_padded_onto_the_canvas_not_cropped():
    for sw, sh in ((1920, 1080), (1440, 1080), (4224, 3024), (1080, 1920), (7680, 4320)):
        cw, ch, ox, oy, W, H = env.fit(sw, sh)
        assert W % 128 == 0 and H % 128 == 0 and W * H <= env.MAX_PIXELS
        assert cw / sw == pytest.approx(ch / sh, rel=0.01)  # its own shape: scaled, nothing cut off
        assert 0 <= ox and ox + cw <= W and 0 <= oy and oy + ch <= H and ox % 2 == oy % 2 == 0
        assert cw * ch > 0.8 * W * H  # most of the canvas is picture
    cw, ch, *_ = env.fit(1440, 1080)
    assert ch > 0.9 * 1024  # a 4:3 frame keeps its height


def test_clicks_land_on_the_padded_picture():
    cw, ch, ox, oy, W, H = env.fit(1920, 1080)
    pts = env.work_points([{"x": 960, "y": 540, "positive": True}, {"x": 0, "y": 0, "positive": False}], (1920, 1080),
                          (cw, ch), (ox, oy))
    assert pts[0]["x"] == pytest.approx(ox + cw / 2) and pts[0]["y"] == pytest.approx(oy + ch / 2)
    assert pts[1] == {"x": ox, "y": oy, "positive": False}


def test_keep_is_asked_for_by_words_or_clicks():
    assert not env.keep_wanted({}) and not env.keep_wanted({"keep_text": "  ", "keep_points": []})
    assert env.keep_wanted({"keep_text": "the car"}) and env.keep_wanted({"keep_points": [{"x": 1, "y": 2}]})


# --- the blend ---------------------------------------------------------------------------------------------------------


def car_scene(h=160, w=240):
    """The footage: a dark blue car with a fine pattern (its badge and grille) on a grey road. The render: the same car
    lit brighter and bluer but its pattern gone (an AI copy), on white snow."""
    yy, xx = np.mgrid[:h, :w]
    car = (yy > 50) & (yy < 120) & (xx > 70) & (xx < 170)
    pattern = 25.0 * np.sign(np.sin(xx / 1.5) * np.sin(yy / 1.5))
    orig = np.full((h, w, 3), 120.0)
    orig[car] = np.array([40.0, 60.0, 140.0]) + pattern[car][:, None]
    gen = np.full((h, w, 3), 235.0)
    gen[car] = (80.0, 100.0, 185.0)
    return np.clip(orig, 0, 255).astype(np.uint8), gen.astype(np.uint8), car


def test_detail_transfer_keeps_the_footages_detail_in_the_renders_light():
    orig, gen, car = car_scene()
    sigma = 4.0
    out = env.detail_transfer(orig, gen, sigma, weight=car)
    core = composite.shrink(car, int(3 * sigma))
    # The pattern is the footage's, to the level...
    assert np.abs((out - orig)[core] - (out - orig)[core].mean(axis=0)).max() < 1.0
    assert out[core].std(axis=0) == pytest.approx(orig[core].astype(float).std(axis=0), abs=0.5)
    # ...and the light is the render's.
    assert np.abs(out[core].mean(axis=0) - gen[core].mean(axis=0)).max() < 2.0


def test_the_snow_around_it_doesnt_bleed_into_its_edges():
    orig, gen, car = car_scene()
    sigma = 4.0
    rim = car & ~composite.shrink(car, int(sigma))
    own = env.detail_transfer(orig, gen, sigma, weight=car)
    plain = env.detail_transfer(orig, gen, sigma)
    car_light = gen[car].astype(float).mean()
    assert abs(own[rim].mean() - car_light) < 4  # its own light right up to the edge
    assert plain[rim].mean() - car_light > 15  # a plain blur lifts the edge toward the white snow: a glow


def test_where_the_render_drew_it_elsewhere_its_own_light_fills_in():
    # LTX drew the car 15 px higher: inside the footage's outline, its lowest rows show the render's snow.
    orig, _, car = car_scene()
    gen = np.full(orig.shape, 235, np.uint8)
    drawn = np.roll(car, -15, axis=0)
    gen[drawn] = (80, 100, 185)
    out, own, _ = env.keep_over(orig, gen, car)
    rows = slice(108, 114)  # the footage's car, below the render's, above the ground band
    lit = gen[drawn].astype(float).mean()
    assert out[rows][car[rows]].astype(float).mean() < lit + 10  # still the car, in the car's new light
    plain = env.detail_transfer(orig, gen, 4.0, weight=car)
    assert plain[rows][car[rows]].mean() > lit + 40  # learning from every pixel in the outline fogs it white
    assert (own[car] > 0.9).mean() > 0.85  # it is still the footage (a little ground snow aside)


def test_falling_snow_passes_in_front_but_snow_stuck_to_it_does_not():
    orig, gen, car = car_scene()
    gen = gen.copy()
    falling, stuck = (slice(70, 73), slice(100, 103)), (slice(80, 83), slice(140, 143))
    prev, nxt = gen.copy(), gen.copy()
    for g in (gen, prev, nxt):
        g[stuck] = 250  # on the render's car in every frame: it moves with the car
    gen[falling] = 250  # only on this frame: it is falling
    out, own, _ = env.keep_over(orig, gen, car, around=(prev, nxt))
    assert out[falling].min() > 200 and own[falling].max() < 0.2  # the flake, over the car (and counted as AI)
    assert (out[stuck].astype(float) @ env.LUM).max() < 160  # the car as filmed (the flake would be 250)
    alone, _, _ = env.keep_over(orig, gen, car)
    assert alone[stuck].min() > 200  # without the frames around it there's no telling them apart


def test_keep_over_changes_only_the_matte_and_leaves_no_halo():
    orig, gen, car = car_scene()
    out, own, sigma = env.keep_over(orig, gen, car)
    assert sigma == pytest.approx(max(env.LOW_MIN, env.LOW_SIGMA * 69))  # 3% of its height, at least a few px
    assert own.max() <= 1 and (own[composite.shrink(car, 2)] > 0.99).mean() > 0.9  # its own pixels are the footage's
    outside = own == 0
    assert outside.sum() > 0.5 * car.size and np.array_equal(out[outside], gen[outside])  # the render, untouched
    core = composite.shrink(car, 12)
    assert out[core].astype(float).std() > 15  # the pattern is back
    far = composite.grow(car, 14) & ~composite.grow(car, 10)  # the feather is over a few px out
    assert np.abs(out[far].astype(float) - gen[far]).max() <= 1
    # No glow: the edge just inside its outline is no lighter than its middle (the snow doesn't wrap into it).
    rim, middle = car & ~composite.shrink(car, 3), composite.shrink(car, 10)
    assert (out[rim].astype(float) @ env.LUM).mean() < (out[middle].astype(float) @ env.LUM).mean() + 3
    nothing, a0, _ = env.keep_over(orig, gen, np.zeros_like(car))
    assert np.array_equal(nothing, gen) and not a0.any()


# --- a whole Change weather with a kept car, ComfyUI faked -------------------------------------------------------------

SW, SH, N = 512, 288, 12  # the shot: 16:9, padded onto a 384x256 canvas (MAX_PIXELS below); delivery is the picture


SPEED = 4  # px a frame the car drives; the crossing tests set it faster
GHOST = False  # the fake LTX draws a second car ahead of any car it is shown (what LTX did on the frosty field)


def car_box(i):
    """The car in the shot's pixels on frame i: driving right, SPEED px a frame."""
    return 150 + SPEED * i, 110, 290 + SPEED * i, 200


def shot_frames():
    yy, xx = np.mgrid[:SH, :SW]
    bg = np.stack([100 + 20 * np.sin(xx / 13.0), 105 + 15 * np.cos(yy / 11.0), 110 + 10 * np.sin((xx + yy) / 17.0)], -1)
    bg += (12 * np.sin(xx / 2.3) * np.cos(yy / 3.1))[..., None]  # fine texture, as a real road or field has
    pattern = 30.0 * np.sign(np.sin(xx / 1.7) * np.sin(yy / 1.7))
    frames = []
    for i in range(N):
        f = bg.copy()
        x0, y0, x1, y1 = car_box(i)
        car = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        f[car] = np.array([40.0, 60.0, 150.0]) + pattern[car][:, None]
        f[240:268] = 70.0  # a flat grey road along the bottom
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


PAINT_OVER = 0  # how many first restyles the fake frame editor paints the road over with grass
PAINT_PART = 0  # how many first restyles paint over half the road (a layout that moved, but the check is sure)


def is_road(img: np.ndarray) -> np.ndarray:
    return (np.abs(img.astype(int) - 70) < 8).all(axis=-1)


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


def render(f, i=0):
    """The fake LTX: every pixel re-rendered brighter and colder, the car's pattern smoothed away (an AI copy)."""
    g = np.asarray(Image.fromarray(f).filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(3)), dtype=np.float32)
    g = np.clip(g * 1.15 + (25, 30, 40), 0, 255).astype(np.uint8)
    blue = (f[..., 2].astype(int) > f[..., 0].astype(int) + 50) & (f[..., 2] > 120)
    if GHOST and blue.sum() > 80:  # the copy it draws runs ahead of the car: a green one, 48 px on
        ys, xs = np.nonzero(blue)
        g[ys.min():ys.max() + 1, min(g.shape[1], xs.min() + 48):min(g.shape[1], xs.max() + 49)] = (0, 255, 0)
    if GHOST == "always":  # a car of its own, though it was shown none: driving across the far side
        g[20:50, 30 + 30 * i:90 + 30 * i] = (0, 255, 0)
    return g


class _Ctx:
    def __init__(self, store, sid, params, tmp):
        from pathlib import Path
        self.store, self.shot_id, self.params = store, sid, params
        self.project_id = store.shot_meta(sid)["project_id"]
        self.input_path = store.source_path(sid)
        self.vid, self.vdir = store.new_version_dir(sid)
        self.client, self.ran, self.tmp, self.stages = _Client(), [], Path(tmp), []

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
        if "TextEncodeQwenImage21" in types:  # the frame editor: the first frame back as it was
            img = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB")).copy()
            self.edits = getattr(self, "edits", 0) + 1
            if self.edits <= PAINT_OVER:
                img[is_road(img)] = (60, 140, 50)  # the road painted over with meadow
            elif self.edits <= PAINT_PART:
                ys, xs = np.nonzero(is_road(img))
                half = is_road(img) & (np.arange(img.shape[1])[None, :] < np.median(xs))
                img[half] = (60, 140, 50)
            Image.fromarray(img).save(p := out / "plate_00001_.png")
            return _Job({"PU_Plate": [_Out(p)]})
        if "SAM3_Detect" in types and "PU_Frame" in graph:  # SAM on one frame: the road by its grey, nothing else
            img = np.asarray(Image.open(graph["PU_Frame"]["inputs"]["image"]).convert("RGB"))
            label = graph.get("PU_Label", {}).get("inputs", {}).get("text", "")
            m = is_road(img) if label.startswith("road") else np.zeros(img.shape[:2], bool)
            Image.fromarray(m.astype(np.uint8) * 255).save(p := out / "mask_00001_.png")
            return _Job({"PU_Mask": [_Out(p)]})
        if "LTXAddVideoICLoRAGuide" in types:
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Control"]["inputs"]["file"])):
                Image.fromarray(render(f, i)).save(p := out / f"frame_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Out": files})
        if "DA3Render" in types and "PU_Depth" in graph:  # depth: near (bottom) bright, a flat sky over the top third
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                h, w = f.shape[:2]
                d = np.tile(np.linspace(0, 255, h)[:, None], (1, w)).astype(np.uint8)
                d[: h // 3] = 40
                Image.fromarray(d).save(p := out / f"depth_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Depth": files})
        if "SAM3_Detect" in types and "PU_Clip" in graph:  # SAM looking for cars on a whole clip: the fake LTX's green ones
            files = []
            for i, f in enumerate(common.read_frames(graph["PU_Clip"]["inputs"]["file"])):
                m = (f[..., 1] > 200) & (f[..., 0] < 80)
                Image.fromarray(m.astype(np.uint8) * 255).save(p := out / f"mask_{i + 1:05d}_.png")
                files.append(_Out(p))
            return _Job({"PU_Mask": files})
        raise AssertionError(f"unexpected graph {sorted(types)}")


@pytest.fixture()
def shot(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    src = tmp_path / "road.mov"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SW}x{SH}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(src)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in shot_frames()))
    sid = store.add_shot(src, "Road")["id"]
    monkeypatch.setattr(common, "MAX_FRAMES", 9)  # a shot longer than the part being changed
    monkeypatch.setattr(env, "MAX_PIXELS", 512 * 256)
    monkeypatch.setattr(common, "FrameWriter", _Writer)
    monkeypatch.setattr(common, "finish_outputs", lambda *a, **k: [])
    monkeypatch.setattr(st, "make_viewer", lambda *a, **k: None)
    tracked = {}

    def fake_track(ctx, proxy, frame, points, text, src_w, run):
        """SAM 3.1, faked: the car on every proxy frame (the shot's car box, in the proxy's pixels)."""
        from pickups import media
        pr = media.probe(proxy)
        tracked.update(proxy_frames=pr.frames, frame=frame, points=points, text=text, src_w=src_w, size=(pr.width, pr.height))
        cw, ch, ox, oy, W, H = env.fit(SW, SH)  # the shot -> its place on the work canvas
        sx, sy, kx, ky = cw / SW, ch / SH, pr.width / W, pr.height / H
        masks = []
        for i in range(pr.frames):
            x0, y0, x1, y1 = car_box(i)
            m = np.zeros((pr.height, pr.width), bool)
            m[round((y0 * sy + oy) * ky):round((y1 * sy + oy) * ky), round((x0 * sx + ox) * kx):round((x1 * sx + ox) * kx)] = True
            masks.append(m)
        return masks

    monkeypatch.setattr(common, "tracked_masks", fake_track)
    _Writer.made = {}
    return store, sid, tracked


def test_a_whole_change_of_weather_that_keeps_the_car(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "snow", "keep_text": "car", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert ctx.stages == [s.label for s in env.STAGES if s.label not in ("Taking it out of the shot", "Lighting what you kept")]  # it stays in its place
    cw, ch, ox, oy, W, H = env.fit(SW, SH)
    sx, sy = cw / SW, ch / SH
    assert (W, H) == (512, 256) and (cw, ch) != (W, H)  # the test's canvas has padding
    assert tracked["text"] == "car" and tracked["frame"] == 0 and tracked["proxy_frames"] == 9 and tracked["src_w"] == W

    # Both models are told to leave it as it is: the frame editor before "everything else", the video model at the end.
    edit = next(g for g in ctx.ran if "PU_Prompt" in g and g["PU_Prompt"]["class_type"] == "TextEncodeQwenImage21")
    text = edit["PU_Prompt"]["inputs"]["prompt"]
    assert text.startswith(env.PRESETS["snow"]["qwen"])
    assert "Keep the car exactly as it is: the same model, colour and shape" in text and text.endswith(env.KEEP)
    assert env.NOTHING_NEW in text  # no walkers or cars invited into the picture
    assert "people" not in env.PRESETS["golden_hour"]["qwen"]
    drive = next(g for g in ctx.ran if "PU_Control" in g)
    assert drive["PU_Prompt"]["inputs"]["text"].endswith("The car stays exactly as it is throughout, the same model, colour and shape.")

    # The frames: the car is the footage's (its pattern back, in the render's light), everything else is the render.
    work = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    files = sorted((ctx.vdir / "work" / "out").glob("r_*.png"))
    gens = sorted((ctx.vdir / "work" / "gen").glob("*.png"))
    assert len(files) == len(gens) == len(work) == 9
    for i in (0, 4, 8):
        out = np.asarray(Image.open(files[i]).convert("RGB")).astype(float)
        gen = np.asarray(Image.open(gens[i]).convert("RGB"))[oy:oy + ch, ox:ox + cw].astype(float)
        src = work[i][oy:oy + ch, ox:ox + cw].astype(float)
        assert out.shape == gen.shape == (ch, cw, 3)  # delivery is the picture, without the canvas's padding
        x0, y0, x1, y1 = car_box(i)
        cy, cx = slice(round(y0 * sy) + 12, round(y1 * sy) - 12), slice(round(x0 * sx) + 12, round(x1 * sx) - 12)
        assert out[cy, cx].std() > 0.8 * src[cy, cx].std() > 10  # the pattern the render smoothed away is back
        hp = lambda a: a - composite.smooth(a, 3)  # noqa: E731
        assert np.corrcoef(hp(out[cy, cx]).ravel(), hp(src[cy, cx]).ravel())[0, 1] > 0.95  # ...the footage's own
        assert np.abs(out[cy, cx].mean((0, 1)) - gen[cy, cx].mean((0, 1))).max() < 4  # in the render's light
        far = np.ones(out.shape[:2], bool)
        far[max(0, cy.start - 40):cy.stop + 40, max(0, cx.start - 40):cx.stop + 40] = False
        assert np.array_equal(out[far], gen[far])  # away from the car: the render as it came

    # The receipt: the kept car is original, the rest AI; receipt.mp4 tints only the AI.
    r, d = res["receipt"], res["details"]
    car_share = sum(round((car_box(i)[2] - car_box(i)[0]) * sx) * round((car_box(i)[3] - car_box(i)[1]) * sy) for i in range(9)) / (cw * ch * 9)
    assert r["original"] == pytest.approx(100 * car_share, rel=0.15) and r["original"] + r["ai"] == pytest.approx(100, abs=0.02)
    assert d["kept_as_shot"].startswith("the car: its detail as shot, its light from the render (")
    assert d["models"][-1] == "SAM 3.1" and "your footage" in d["note"]
    qa = d["_qa"]
    assert qa["keep"] == "the car" and qa["keep_frames_found"] == qa["frames"] == 9 and qa["keep_method"] == "detail transfer"
    assert qa["keep_sigma_px"] == pytest.approx(env.LOW_SIGMA * 90 * sy, rel=0.1)  # 3% of the car's height on the work clip
    assert qa["keep_flicker"] is not None and qa["keep_halo"] is not None
    assert len(_Writer.made["receipt.mp4"].frames) == 9


def test_a_click_later_in_the_shot_is_followed_from_there(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "snow", "keep_points": [{"x": 256, "y": 144, "positive": True}], "keep_frame": 11, "grain": "off"},
               tmp_path / "jobs")
    res = env.run(ctx)
    # SAM gets the whole stretch up to the click (12 frames for a 9-frame change), the click in the work clip's pixels.
    assert tracked["proxy_frames"] == 12 and tracked["frame"] == 11 and tracked["text"] == ""
    cw, ch, ox, oy, _, _ = env.fit(SW, SH)
    assert tracked["points"][0]["x"] == pytest.approx(256 * cw / SW + ox)
    assert tracked["points"][0]["y"] == pytest.approx(144 * ch / SH + oy)
    assert res["details"]["_qa"]["keep"] == "the object in the middle of the frame"  # named by where it is
    assert res["receipt"]["original"] > 0


def test_without_keep_every_pixel_is_still_the_render(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "snow", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert not tracked and res["receipt"] == {"original": 0.0, "borrowed": 0.0, "ai": 100.0}
    assert "Finding what to keep" not in ctx.stages and "kept_as_shot" not in res["details"]
    drive = next(g for g in ctx.ran if "PU_Control" in g)
    assert "stays exactly as it is" not in drive["PU_Prompt"]["inputs"]["text"]
    assert not (ctx.vdir / "receipt.mp4").exists() and "receipt.mp4" not in _Writer.made


# --- a kept thing that moves through the frame: taken out, the weather rendered without it, put back relit -----------


def test_a_car_crossing_the_frame_travels_but_one_the_camera_tracks_does_not():
    def boxes(step):
        out = []
        for i in range(20):
            m = np.zeros((90, 320), bool)
            m[40:60, 10 + step * i:50 + step * i] = True
            out.append(m)
        return out
    assert env.travels(boxes(12))  # crosses the frame: its own size is a small part of all it covers
    assert not env.travels(boxes(0)) and not env.travels(boxes(1))  # tracked: it stays where it is
    assert not env.travels([np.zeros((9, 9), bool)] * 3)


def test_a_panning_camera_is_measured_so_its_car_keeps_the_renders_copy(tmp_path):
    rng = np.random.default_rng(1)
    tex = np.asarray(Image.fromarray(rng.integers(0, 255, (300, 700, 3), np.uint8)).resize((1400, 600), Image.BILINEAR))

    def clip(step, name):
        path = tmp_path / name
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360", "-r", "24", "-i", "-",
                              "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)], stdin=subprocess.PIPE)
        p.communicate(b"".join(np.ascontiguousarray(tex[100:460, 50 + step * i:690 + step * i]).tobytes() for i in range(12)))
        return path
    none = [np.zeros((360, 640), bool)] * 12
    assert env.camera_still(clip(0, "still.mp4"), none)
    assert not env.camera_still(clip(6, "pan.mp4"), none)  # 66 px over the shot: too far to borrow the road
    assert env.camera_still(clip(1, "creep.mp4"), none)  # a slow creep is aligned away


def test_what_is_never_uncovered_is_filled_with_the_textures_beside_it():
    # Where a walker ends the shot, the meadow is never seen: it is filled with meadow from beside it, nothing new.
    h, w = 80, 200
    yy, xx = np.mgrid[:h, :w]
    meadow = np.stack([70 + 20 * np.sin(xx / 2.0), 110 + 20 * np.sin(xx / 2.0 + yy / 3.0), 40 + 0 * xx], -1)
    meadow = np.clip(meadow + (yy * 0.3)[..., None], 0, 255).astype(np.uint8)
    frame = meadow.copy()
    hole = (yy > 30) & (yy < 60) & (xx > 120) & (xx < 140)
    frame[hole] = (255, 120, 0)  # what a frame editor might have drawn: an orange kite
    out = composite.clone_fill(frame, hole)
    assert not ((out[..., 0] > 200) & (out[..., 2] < 30)).any()
    assert np.abs(out[hole].astype(float).mean(0) - meadow[hole].astype(float).mean(0)).max() < 12  # meadow-coloured
    assert out[hole].std() > 5  # with texture, not a smooth blob
    assert np.array_equal(out[~hole], frame[~hole])


def test_a_person_sam_lost_for_a_few_frames_is_put_back():
    h, w, n = 60, 160, 20
    masks = []
    for i in range(n):
        m = np.zeros((h, w), bool)
        m[10:50, 20 + i:35 + i] = True  # her
        if not 8 <= i < 13:  # SAM lost him for five frames (he walks 3 px a frame)
            m[10:50, 60 + 3 * i:75 + 3 * i] = True
        masks.append(m)
    mended = env.mend_dropouts(masks)
    for i in range(8, 13):
        him = np.zeros((h, w), bool)
        him[10:50, 60 + 3 * i:75 + 3 * i] = True
        assert (mended[i] & him).sum() >= 0.9 * him.sum()  # he is back, where he is on that frame...
        assert (mended[i] & ~him & ~masks[i]).sum() <= 0.7 * him.sum()  # ...a few px of give, not a smear of his path
        if i == 10:
            union = masks[7] | masks[13]
            assert (union & ~him & ~masks[i]).sum() > 1.1 * him.sum()  # (the union of either side would be)
        assert mended[i][10:50, 20 + i:35 + i].all()
    covered = env.mend_dropouts(masks, cover=True)
    for i in range(8, 13):
        assert covered[i][masks[7] | masks[13] | mended[i]].all()  # taking him out may be generous: was, is, will be
    assert all(np.array_equal(a, b) for i, (a, b) in enumerate(zip(masks, mended)) if not 8 <= i < 13)  # the rest as found
    assert env.mend_dropouts([np.zeros((4, 4), bool)] * 3)[0].sum() == 0
    entering = []
    for i in range(n):
        m = np.zeros((h, w), bool)
        m[20:40, 0:min(w, 3 + 6 * i)] = True  # a car driving in from the left edge: small at first, not lost
        entering.append(m)
    assert all(np.array_equal(a, b) for a, b in zip(entering, env.mend_dropouts(entering)))


def test_small_gaps_in_a_persons_mask_are_closed_but_not_the_space_beside_them():
    m = np.zeros((200, 300), bool)
    m[20:190, 60:120] = True  # her
    m[20:190, 140:200] = True  # him, a wide gap away
    m[100:110, 120:140] = True  # their joined hands
    m[80:84, 90:94] = False  # a speck SAM dropped in her coat
    m[150:153, 65:115] = False  # a sliver across her coat
    out = env.close_gaps(m)
    assert out[80:84, 90:94].all() and out[150:153, 65:115].all()
    assert not out[30:90, 126:134].any()  # the gap between them stays open
    assert (out | m).sum() == out.sum()


def test_the_cleared_shot_keeps_every_frame_at_29_97(tmp_path):
    from pickups import media
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=128x64:rate=30000/1001", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=48000", "-frames:v", "97", "-c:v", "libx264", "-c:a", "aac", "-shortest",
                    str(src)], check=True)
    work = tmp_path / "work.mp4"
    media.prepare_work_clip(src, work, 0, 97, (128, 64))
    frames = list(common.read_frames(work))
    out = tmp_path / "control.mp4"
    env.write_clip(iter(frames), out, (128, 64), 30000 / 1001, work)
    assert media.probe(out).frames == len(frames) == 97  # LTX needs all 8k+1 of them


def test_sentences_that_put_the_kept_car_in_the_picture_are_dropped():
    text = ("Golden hour on a summer evening, a lush green meadow. Tall golden grass in the foreground and a small dark SUV "
            "on the far road behind the trees. The grass sways in the breeze.")
    assert env.without_subject(text, "the SUV") == ("Golden hour on a summer evening, a lush green meadow. Tall golden grass in "
                                                     "the foreground and the far road behind the trees. The grass sways in the breeze.")
    assert env.without_subject("A blue car drives along a winding road.", "the car") == "A winding road."
    assert env.without_subject("Two cars pass. Snow falls.", "the car") == "Snow falls."  # plurals, and any vehicle word
    assert env.without_subject("A van waits. Snow falls.", "the blue SUV") == "Snow falls."
    assert env.without_subject("The couple walk on. Snow falls.", "the couple") == "Snow falls."
    assert env.without_subject("A man in a white T-shirt walks away through a meadow. Clouds.", "the man") == "Clouds."
    assert env.without_subject("Snow falls on the road.", "the car") == "Snow falls on the road."
    assert env.without_subject("", "the car") == "" and env.without_subject("Snow.", "") == "Snow."


def road_scene(h=200, w=320):
    """A road (textured grey) and a dark blue car with a bright glint; the render: the same road in warmer, dimmer
    light, with a strip of deep shade right beside the car (the golden-hour road that turned the last try black)."""
    yy, xx = np.mgrid[:h, :w]
    plate = np.stack([110 + 25 * np.sin(xx / 5.0) * np.cos(yy / 7.0)] * 3, -1) + (0, 4, 10)
    car = (yy > 80) & (yy < 120) & (xx > 120) & (xx < 200)
    orig = plate.copy()
    orig[car] = (30, 38, 70)
    orig[(yy > 88) & (yy < 92) & (xx > 140) & (xx < 150)] = 245  # a glint on the roof
    gain = np.array([0.72, 0.6, 0.5])  # warmer, most of a stop darker (linear)
    gen = env.to_srgb(env.to_linear(plate) * gain)
    shade = (yy >= 120) & (yy < 132) & (xx > 100) & (xx < 220)
    gen[shade] *= 0.45
    return [np.clip(np.rint(a), 0, 255).astype(np.uint8) for a in (orig, plate, gen)], car, gain


def lum(a):
    return a.astype(float) @ env.LUM


def test_the_new_light_is_learnt_from_the_scene_around_it_not_a_shaded_ring():
    (orig, plate, gen), car, gain = road_scene()
    light = env.scene_light(plate, gen, car)
    assert light[0] == pytest.approx(gain, rel=0.12) and not light[1].any()  # the shade beside it is a few pixels, not all
    kept = env.relight(orig, light)
    yy = np.mgrid[:200, :320][0]
    body = car & ~((yy > 86) & (yy < 94))
    assert 0.6 * lum(orig)[body].mean() < lum(kept)[body].mean() < lum(orig)[body].mean()  # a stop darker is not black
    # Its own contrast holds: the glint is still far brighter than the body.
    assert lum(kept)[90, 145] - lum(kept)[body].mean() > 0.7 * (lum(orig)[90, 145] - lum(orig)[body].mean())
    assert kept[car][:, 0].mean() / kept[car][:, 2].mean() > orig[car][:, 0].mean() / orig[car][:, 2].mean()  # warmer


def test_a_new_material_is_not_taken_for_coloured_light():
    # Frost that turned to green grass: the plate is white-grey, the render green. The car must not turn green.
    (orig, plate, _), car, _ = road_scene()
    grass = np.zeros_like(plate)
    grass[:] = (60, 140, 40)
    light = env.scene_light(plate, grass, car)
    exposure = float(np.exp(np.log(light[0]) @ env.LUM))
    g = light[0] / exposure
    assert g.max() / g.min() < np.exp(2 * env.TINT_MAX) + 1e-3
    assert env.GAIN_RANGE[0] <= exposure <= env.GAIN_RANGE[1]


def test_when_all_around_it_changed_material_the_light_comes_from_the_rest_of_the_frame():
    # Frost to summer at golden hour: the field around the car went from white frost to green grass (a new material),
    # but the sky and the hills, the same materials, show the warm light.
    h, w = 200, 320
    yy, xx = np.mgrid[:h, :w]
    plate = np.zeros((h, w, 3))
    plate[:] = (215, 218, 224)  # frost
    plate[yy < 80] = (150, 152, 160)  # sky and hills
    plate += (10 * np.sin(xx / 3.0) * np.cos(yy / 4.0))[..., None]
    plate = np.clip(plate, 0, 255).astype(np.uint8)
    warm = np.array([1.3, 1.1, 0.8])
    gen = np.clip(env.to_srgb(env.to_linear(plate) * warm), 0, 255).astype(np.uint8)
    gen[yy >= 80] = (70, 130, 40)  # green grass where the frost was
    car = (yy > 140) & (yy < 160) & (xx > 140) & (xx < 180)
    light = env.scene_light(plate, gen, car)
    assert light[0] == pytest.approx(warm * np.exp(np.clip(np.log(warm) - np.log(warm) @ env.LUM, -env.TINT_MAX, env.TINT_MAX)
                                                   - (np.log(warm) - np.log(warm) @ env.LUM)), rel=0.08)


def test_fog_lays_a_veil_over_it():
    (orig, plate, _), car, _ = road_scene()
    fog = np.clip(np.rint(env.to_srgb(env.to_linear(plate) * 0.6 + 0.25)), 0, 255).astype(np.uint8)
    light = env.scene_light(plate, fog, car, haze=True)
    assert light[1].min() > 0.05 and light[0].mean() < 0.9
    assert not env.scene_light(plate, fog, car, haze=False)[1].any()


def test_the_light_holds_steady_from_frame_to_frame():
    lights = [np.array([[1.0 + 0.2 * (-1) ** i] * 3, [0.0] * 3]) for i in range(20)]
    lights[7] = None
    out = env.steady(lights)
    assert len(out) == 20 and all(abs(o[0, 0] - 1.0) < 0.05 for o in out[4:16])  # the jitter averages out
    assert env.steady([None, None])[0][0].tolist() == [1.0, 1.0, 1.0]


def test_its_shadow_comes_back_from_the_footage():
    (orig, plate, gen), car, _ = road_scene()
    yy, xx = np.mgrid[:200, :320]
    under = (yy >= 120) & (yy < 128) & (xx > 125) & (xx < 195)
    orig[under] = (orig[under] * 0.5).astype(np.uint8)  # its contact shadow on the road, as filmed
    out = env.shadow_back(orig, plate, gen, car)
    inner = under & (xx > 135) & (xx < 185) & (yy >= 121) & (yy < 127)
    assert lum(out)[inner].mean() < 0.85 * lum(gen)[inner].mean()  # darker where the footage was
    far = (yy < 30) | (yy > 180)
    assert np.abs(out[far] - gen[far]).max() < 1.0  # nowhere else


def test_keep_relit_puts_the_footages_car_over_the_render_in_its_light():
    (orig, plate, gen), car, gain = road_scene()
    light = env.scene_light(plate, gen, car)
    out, own, _ = env.keep_relit(orig, plate, gen, car, light)
    core = composite.shrink(car, 4)
    hp = lambda a: a.astype(float) - composite.smooth(a.astype(float), 2)  # noqa: E731
    assert np.corrcoef(hp(out)[core].ravel(), hp(orig)[core].ravel())[0, 1] > 0.95  # its detail as filmed
    assert (own[core] > 0.99).all()
    far = ~composite.grow(car, 50)
    assert np.array_equal(out[far], gen[far])


def test_its_edge_brings_no_old_background_onto_the_new_one():
    # A dark car with a soft (motion-blurred) edge over white frost; SAM's outline is a little wide, so its edge pixels
    # are half frost. Over the new green grass that frost must not ring the car in white.
    h, w = 120, 200
    yy, xx = np.mgrid[:h, :w]
    car = (yy > 40) & (yy < 80) & (xx > 60) & (xx < 140)
    plate = np.full((h, w, 3), (220, 222, 228), np.float64)
    a_true = composite.blur(car.astype(np.float32), 1.5)
    orig = a_true[..., None] * np.array([40.0, 45, 60]) + (1 - a_true[..., None]) * plate
    orig, plate = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (orig, plate))
    gen = np.zeros((h, w, 3), np.uint8)
    gen[:] = (60, 130, 40)
    sam = composite.grow(car, 2)
    light = np.array([[1.0] * 3, [0.0] * 3], np.float32)
    out, own, _ = env.keep_relit(orig, plate, gen, sam, light)
    ring = composite.grow(car, 3) & ~composite.shrink(car, 1)
    assert lum(out)[ring].max() < lum(gen)[0, 0] + 12  # no white rim: never much lighter than the grass
    alpha, fg = env.unmix(orig, plate, sam)
    inside = composite.shrink(car, 3)
    assert (alpha[inside] > 0.9).all() and alpha[~composite.grow(car, 5)].max() < 0.05
    assert np.abs(fg[ring & (alpha > 0.3)] - (40, 45, 60)).max() < 35  # its own colour, unmixed from the frost


def test_the_safety_net_finds_only_what_the_render_invented():
    assert env.ghost_label("the man") == "person" and env.ghost_label("the couple") == "person"
    assert env.ghost_label("the blue SUV") == "car" and env.ghost_label("the cable car") == "car"
    assert env.ghost_label("the red kite") == "red kite" and env.ghost_label("") == ""
    h, w = 90, 160
    parked, ghost, speck = np.zeros((h, w), bool), np.zeros((h, w), bool), np.zeros((h, w), bool)
    parked[10:30, 10:40] = True  # a car really parked there: in the shot and in the render
    ghost[50:70, 90:130] = True  # a car only the render has
    speck[5, 150] = True
    out = env.ghost_masks([parked | ghost | speck], [parked])[0]
    assert out[55:65, 95:125].all() and not out[15:25, 15:35].any() and not out[5, 150]
    band = np.zeros((h, w), bool)
    band[80:84, 5:155] = True  # a far tree line SAM took for people: a band, not a person
    assert not env.ghost_masks([band], [np.zeros((h, w), bool)])[0].any()
    everything = np.ones((h, w), bool)  # SAM's whole-frame answer when it finds nothing
    assert not env.ghost_masks([everything], [parked])[0].any()
    assert env.ghost_masks([ghost], [everything])[0][55:65, 95:125].all()


def test_a_ghost_that_moves_is_filled_from_the_renders_own_frames():
    h, w, n = 60, 160, 8
    road = np.zeros((h, w, 3), np.uint8)
    road[:] = (90, 120, 60)
    road[:, ::7] = (70, 100, 50)  # a pattern, so a fill shows if it is wrong
    frames, ghosts = [], []
    for i in range(n):
        f = road.copy()
        g = np.zeros((h, w), bool)
        g[20:40, 10 + 15 * i:40 + 15 * i] = True
        f[g] = (0, 255, 0)
        frames.append(f)
        ghosts.append(composite.grow(g, 2))
    fixed, changed = env.fill_ghosts(frames, ghosts)
    assert changed == n
    for f in fixed:
        assert np.abs(f.astype(int) - road).max() <= 2  # the road as it is in the render's other frames
    still = [np.zeros((h, w), bool)] * n
    same, none = env.fill_ghosts(frames, still)
    assert none == 0 and same is frames
    # One found in the same place on every frame is scenery SAM misread (a clump of trees): left as rendered.
    spot = np.zeros((h, w), bool)
    spot[3:17, 138:152] = True
    assert not any(g.any() for g in env.moving_only([spot] * n))
    walking = env.moving_only(ghosts)
    assert all(np.array_equal(a, b) for a, b in zip(walking, ghosts))
    kept, changed = env.fill_ghosts(frames, [spot] * n)
    assert changed == 0 and all(np.array_equal(a, b) for a, b in zip(kept, frames))


def test_without_a_real_plate_a_too_wide_matte_brings_no_rim_of_old_background():
    # A dark coat on a pale dusk sky, SAM's outline 4 px too wide; the render (made without him) is wet dark road.
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    coat = (yy > 20) & (yy < 100) & (xx > 50) & (xx < 110)
    orig = np.full((h, w, 3), (150, 160, 180), np.uint8)
    orig[coat] = (25, 28, 35)
    gen = np.full((h, w, 3), (60, 62, 66), np.uint8)
    wide = composite.grow(coat, 4)
    out, _, _ = env.keep_relit(orig, gen, gen, wide, np.array([[1.0] * 3, [0.0] * 3], np.float32), real_plate=False)
    rim = wide & ~coat
    assert lum(out)[rim].mean() < lum(gen)[0, 0] + 15  # the road, not a pale ring of sky around him
    assert lum(out)[composite.shrink(coat, 3)].max() < 45  # and his coat as filmed


def test_background_seen_through_it_is_the_new_background():
    # A cable car's hanger: bars inside SAM's outline, with the old grass between them. In snow, the gaps are snow.
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    outline = (yy > 20) & (yy < 100) & (xx > 40) & (xx < 120)
    bars = outline & (((xx - 40) % 20 < 4) | (yy > 80))
    rng = np.random.default_rng(3)
    plate = np.clip(np.stack([60 + 0 * xx, 110 + 0 * xx, 50 + 0 * xx], -1) + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)
    orig = plate.copy()
    orig[bars] = (30, 30, 35)
    gen = np.full((h, w, 3), 235, np.uint8)
    out, own, _ = env.keep_relit(orig, plate, gen, outline, np.array([[1.0] * 3, [0.0] * 3], np.float32))
    gap = outline & ~composite.grow(bars, 3)
    assert gap.sum() > 500 and np.percentile(lum(out)[gap], 5) > 200  # snow between the bars, not the old grass
    assert lum(out)[composite.shrink(bars, 2)].max() < 60  # the bars themselves, as filmed
    solid, _, _ = env.keep_relit(orig, plate, gen, outline, np.array([[1.0] * 3, [0.0] * 3], np.float32), see_through=False)
    assert lum(solid)[composite.shrink(outline, 6)].max() < 140  # a person: no holes where they match the background


@pytest.fixture()
def crossing(monkeypatch):
    """The car drives 25 px a frame (across the frame), and the fake LTX draws a ghost ahead of any car it sees."""
    mod = __import__("sys").modules[__name__]
    monkeypatch.setattr(mod, "SPEED", 25)
    monkeypatch.setattr(mod, "GHOST", True)


def test_a_car_that_crosses_the_frame_is_rendered_out_and_put_back_with_no_ghost(crossing, shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "snow", "keep_text": "car", "grain": "off", "subject_light": False,
                            "describe": "A blue car drives along a road. Snow falls on the road."}, tmp_path / "jobs")
    res = env.run(ctx)
    assert "Taking it out of the shot" in ctx.stages
    cw, ch, ox, oy, W, H = env.fit(SW, SH)
    sx, sy = cw / SW, ch / SH
    # Neither model is shown or told about the car.
    drive = next(g for g in ctx.ran if "PU_Control" in g)
    assert drive["PU_Control"]["inputs"]["file"].endswith("control.mp4")
    text = drive["PU_Prompt"]["inputs"]["text"]
    assert "car" not in text.lower() and "Snow falls on the road." in text
    edit = next(g for g in ctx.ran if "PU_Prompt" in g and g["PU_Prompt"]["class_type"] == "TextEncodeQwenImage21")
    assert "car" not in edit["PU_Prompt"]["inputs"]["prompt"].lower()
    files = sorted((ctx.vdir / "work" / "out").glob("r_*.png"))
    work = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    for i in (0, 4, 8):
        out = np.asarray(Image.open(files[i]).convert("RGB")).astype(float)
        assert not ((out[..., 1] > 240) & (out[..., 0] < 30)).any()  # no second (green) car anywhere
        x0, y0, x1, y1 = car_box(i)
        cy, cx = slice(round(y0 * sy) + 8, round(y1 * sy) - 8), slice(round(x0 * sx) + 8, round(min(x1, SW) * sx) - 8)
        src = work[i][oy:oy + ch, ox:ox + cw].astype(float)
        hp = lambda a: a - composite.smooth(a, 3)  # noqa: E731
        assert np.corrcoef(hp(out[cy, cx]).ravel(), hp(src[cy, cx]).ravel())[0, 1] > 0.9  # the real car, as filmed
        assert lum(out[cy, cx]).mean() > lum(src[cy, cx]).mean()  # in the render's brighter light
    qa = res["details"]["_qa"]
    assert qa["keep_method"] == "relit on a clean plate" and qa["keep_plate"] == "real pixels from other frames"
    assert qa["ghost_frames"] == 0
    assert all(g > 1.0 for g in qa["keep_gain"])  # the fake render is lighter everywhere



def test_a_car_the_render_invents_on_its_own_is_filled_away(crossing, shot, tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys").modules[__name__], "GHOST", "always")
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "snow", "keep_text": "car", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert res["details"]["_qa"]["ghost_frames"] == 9
    for f in sorted((ctx.vdir / "work" / "out").glob("r_*.png")):
        out = np.asarray(Image.open(f).convert("RGB")).astype(int)
        assert ((out[..., 1] > 200) & (out[..., 0] < 80)).sum() < 20  # no green car left


def test_a_kept_person_is_always_taken_out_even_when_they_stay_in_place(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "rain", "keep_text": "the woman", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert "Taking it out of the shot" in ctx.stages  # a render's copy of a walking person runs off theirs
    assert res["details"]["_qa"]["keep_method"] == "relit on a clean plate"



# --- the layout survives the restyle -----------------------------------------------------------------------------------


def test_layout_overlap_allows_a_few_px_but_not_a_road_gone_or_widened():
    a = np.zeros((100, 200), bool)
    a[60:70, :] = True
    assert env.overlap(a, a, 2) == 1.0
    assert env.overlap(a, np.roll(a, 2, axis=0), 2) > 0.95  # a couple of px off: the same road
    assert env.overlap(a, np.zeros_like(a), 2) == 0.0  # painted over
    wide = np.zeros_like(a)
    wide[50:90, :] = True  # the verge turned to tarmac
    assert env.overlap(a, wide, 2) < env.LAYOUT_MIN


def test_the_restyle_is_redone_until_the_road_is_where_it_was(shot, tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys").modules[__name__], "PAINT_OVER", 1)
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "rain", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    edits = [g for g in ctx.ran if "PU_Prompt" in g and g["PU_Prompt"]["class_type"] == "TextEncodeQwenImage21"]
    assert len(edits) == 2  # the first painted the road over; the second kept it
    assert env.LAYOUT_KEEP in edits[0]["PU_Prompt"]["inputs"]["prompt"]
    lay = res["details"]["_layout"]
    assert lay["layout"] == ["road"] and lay["layout_tries"][0] < env.LAYOUT_MIN <= lay["layout_tries"][1]
    first = np.asarray(Image.open(ctx.vdir / "work" / "first.png").convert("RGB"))
    assert is_road(first).sum() > 1000  # the frame LTX starts from has its road


def test_when_every_restyle_misses_the_layout_the_check_is_unsure_and_the_first_try_stands(shot, tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys").modules[__name__], "PAINT_OVER", 9)  # never kept: as if SAM's road were wrong
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "rain", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    lay = res["details"]["_layout"]
    assert len(lay["layout_tries"]) == env.LAYOUT_TRIES and lay["layout_check"] == "unsure"
    assert (ctx.vdir / "work" / "first.png").read_bytes() == (ctx.vdir / "work" / "first_0.png").read_bytes()


def test_while_the_layout_keeps_moving_more_seeds_are_tried(shot, tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys").modules[__name__], "PAINT_PART", env.LAYOUT_TRIES + 1)
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "rain", "grain": "off"}, tmp_path / "jobs")
    lay = env.run(ctx)["details"]["_layout"]
    assert len(lay["layout_tries"]) == env.LAYOUT_TRIES + 2 and lay["layout_check"] == "passed"


def test_a_layout_that_never_holds_stops_before_the_render(shot, tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys").modules[__name__], "PAINT_PART", 99)
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "rain", "grain": "off"}, tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="layout"):
        env.run(ctx)
    edits = [g for g in ctx.ran if "PU_Prompt" in g and g["PU_Prompt"]["class_type"] == "TextEncodeQwenImage21"]
    assert len(edits) == env.LAYOUT_TRIES_MAX
    assert not any("LTXAddVideoICLoRAGuide" in {n["class_type"] for n in g.values()} for g in ctx.ran)


def test_presets_change_the_weather_not_the_layout_or_the_sun():
    gold = env.PRESETS["golden_hour"]
    assert "outside the frame" in gold["qwen"] and "outside the frame" in gold["ltx"] and "near the horizon" not in gold["ltx"]
    assert "stay grass" in env.PRESETS["rain"]["qwen"] and "every surface are soaking" not in env.PRESETS["rain"]["qwen"]
    assert "roads and paths" in env.PRESETS["snow"]["qwen"]


def test_a_change_of_light_keeps_the_ground_as_filmed_and_takes_only_the_light():
    rng = np.random.default_rng(5)
    h, w = 120, 240
    meadow = np.clip(np.stack([90 + 0 * rng.random((h, w)), 130 + 25 * rng.random((h, w)), 60 + 0 * rng.random((h, w))], -1), 0, 255)
    meadow = meadow.astype(np.uint8)
    warm = np.clip(env.to_srgb(env.to_linear(meadow) * np.array([1.4, 1.0, 0.6])), 0, 255).astype(np.uint8)
    restyled = warm.copy()
    restyled[:, 118:124] = (120, 90, 60)  # a dirt track the restyle painted in
    out = env.light_transfer(meadow, restyled, round(env.TRANSFER_RADIUS * w))
    hp = lambda a: lum(a) - composite.smooth(lum(a), 2)  # noqa: E731
    assert np.corrcoef(hp(out).ravel(), hp(meadow).ravel())[0, 1] > 0.9  # every blade of grass as filmed
    assert out[..., 0].mean() / out[..., 2].mean() > 1.3 * meadow[..., 0].mean() / meadow[..., 2].mean()  # in warm light
    track, beside = lum(out)[:, 118:124].mean(), lum(out)[:, 100:110].mean()
    assert abs(track - beside) < 6  # the painted-in track is not carried over
    assert env.PRESETS["golden_hour"].get("light_only") and env.PRESETS["night"].get("light_only")
    assert not env.PRESETS["snow"].get("light_only")


def test_a_change_of_light_or_fog_is_the_footage_relit_every_road_and_person_as_filmed(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "golden_hour", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert res["receipt"]["original"] == 100.0 and res["details"]["_qa"]["method"] == "relit footage"
    cw, ch, ox, oy, W, H = env.fit(SW, SH)
    work = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    files = sorted((ctx.vdir / "work" / "out").glob("r_*.png"))
    assert len(files) == 9
    for i in (0, 8):
        out = np.asarray(Image.open(files[i]).convert("RGB")).astype(float)
        src = work[i][oy:oy + ch, ox:ox + cw].astype(float)
        assert out.shape == src.shape
        hp = lambda a: a - composite.smooth(a, 2)  # noqa: E731
        assert np.corrcoef(hp(out).ravel(), hp(src).ravel())[0, 1] > 0.9  # the footage's own detail everywhere
        assert lum(out).mean() > lum(src).mean() + 10  # in the fake render's brighter light


def test_a_kept_person_in_a_change_of_light_is_their_own_footage_graded_over_the_relit_footage(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "golden_hour", "keep_text": "the woman", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    qa = res["details"]["_qa"]
    assert qa["method"] == "relit footage" and qa["keep_method"] == "footage with a graded light"
    assert qa["keep_lock_min"] is None or qa["keep_lock_min"] >= env.LOCK_MIN  # the pixel lock
    assert res["receipt"]["original"] == 100.0
    files = sorted((ctx.vdir / "work" / "out").glob("r_*.png"))
    cw, ch, *_ = env.fit(SW, SH)
    assert len(files) == 9 and Image.open(files[0]).size == (cw, ch)


def test_night_adds_the_renders_own_lamps_without_darkening_anything():
    relit = np.full((60, 80, 3), 20, np.uint8)
    render = np.full((60, 80, 3), 25, np.uint8)
    render[28:32, 38:42] = (255, 220, 150)  # a lit window the render drew
    out = env.add_lights(relit, render, 6)
    assert lum(out)[29:31, 39:41].min() > 120 and (out.astype(int) >= relit.astype(int)).all()
    assert np.abs(out[:10].astype(int) - relit[:10]).max() <= 1  # nothing added where the render is plain


def test_fog_is_thickened_where_the_renders_veil_is_heaviest():
    relit = np.full((40, 80, 3), 60, np.uint8)
    fields = np.zeros((2, 5, 10, 3), np.float32)
    fields[0] = 1.0
    fields[1, :, 5:] = 0.4  # the render's fog is heavy on the right (the distance)
    out = env.thicken_fog(relit, fields, np.array([0.8, 0.8, 0.8], np.float32))
    assert lum(out)[:, 70:].mean() > lum(out)[:, :10].mean() + 60  # right sunk into fog, left still the footage
    assert abs(lum(out)[:, :5].mean() - 60) < 3


def test_fog_from_depth_hides_the_sky_and_the_distance_and_keeps_the_near_ground():
    h, w = 90, 120
    d = np.tile(np.linspace(0, 255, h)[:, None], (1, w)).astype(np.uint8)
    d[:30] = 40  # the model's flat sky
    assert env.sky_of(d)[:30].all() and not env.sky_of(d)[40:].any()
    dens = env.fog_density(d)
    assert (dens[:30] == 1).all()  # the sky is all fog
    assert dens[-1].mean() < 0.3 and dens[35].mean() > 0.9  # near ground clear, far ground gone
    assert (np.diff(dens[30:].mean(axis=1)) <= 1e-6).all()  # thinner all the way toward the camera


def test_a_fog_take_is_the_footage_under_fog_by_depth_with_no_render_pixels(shot, tmp_path):
    store, sid, tracked = shot
    (tmp_path / "jobs").mkdir()
    ctx = _Ctx(store, sid, {"change": "fog", "grain": "off"}, tmp_path / "jobs")
    res = env.run(ctx)
    assert res["receipt"]["original"] == 100.0 and res["details"]["_qa"]["method"] == "fog from depth"
    kinds = [{n["class_type"] for n in g.values()} for g in ctx.ran]
    assert not any("LTXAddVideoICLoRAGuide" in k or "TextEncodeQwenImage21" in k for k in kinds)  # no render at all
    cw, ch, ox, oy, W, H = env.fit(SW, SH)
    work = list(common.read_frames(ctx.vdir / "work" / "work.mp4"))
    out = np.asarray(Image.open(sorted((ctx.vdir / "work" / "out").glob("r_*.png"))[4]).convert("RGB")).astype(float)
    src = work[4][oy:oy + ch, ox:ox + cw].astype(float)
    assert out.shape == src.shape
    fog = env.to_srgb(env.FOG_COLOUR)
    assert np.abs(out[: ch // 4].mean((0, 1)) - fog).max() < 6  # the sky is the fog colour
    hp = lambda a: a - composite.smooth(a, 2)  # noqa: E731
    near = slice(ch - ch // 6, ch)
    assert np.corrcoef(hp(out[near]).ravel(), hp(src[near]).ravel())[0, 1] > 0.9  # the near ground is the footage


def test_the_thin_matte_follows_the_bars_and_keeps_a_solid_cabin_solid():
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    rng = np.random.default_rng(4)
    plate = np.clip(np.stack([40 + 0 * xx, 70 + 0 * xx, 40 + 0 * xx], -1) + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)
    cabin = (yy > 70) & (yy < 110) & (xx > 40) & (xx < 120)
    legs = (yy > 20) & (yy <= 70) & ((np.abs(xx - (60 + (yy - 20) * 0.3)) < 2) | (np.abs(xx - (100 - (yy - 20) * 0.3)) < 2))
    orig = plate.copy()
    orig[cabin] = (90, 70, 50)
    orig[legs] = (30, 30, 32)
    window = (yy > 85) & (yy < 89) & (xx > 70) & (xx < 74)
    orig[window] = plate[window]  # a dark window that happens to match the trees behind
    sam = composite.grow(cabin | legs, 4) | ((yy > 20) & (yy <= 70) & (xx > 55) & (xx < 105))  # a blob over the legs
    a = env.thin_matte(orig, plate, sam)
    assert (a[composite.shrink(cabin, 3)] > 0.95).all()  # the cabin, window and all
    between = (yy > 35) & (yy < 60) & (xx > 76) & (xx < 84)
    assert a[between].max() < 0.1  # the gap between the legs is background
    assert a[legs & (yy > 30)].mean() > 0.7  # the legs themselves are kept


def test_a_road_and_a_puddle_hold_their_place_and_width_in_the_new_season(tmp_path, monkeypatch):
    h, w, n = 128, 128, 4
    yy, xx = np.mgrid[:h, :w]
    road = (yy >= 20) & (yy < 23)  # spans the frame
    puddle = (yy >= 40) & (yy < 46) & (xx > 30) & (xx < 70)
    frames = []
    for _ in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:] = (200, 205, 210)  # frost
        f[road] = (70, 70, 72)
        f[puddle] = (72, 71, 70)
        frames.append(f)
    work = tmp_path / "work.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(work)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    out = tmp_path / "out"
    out.mkdir()
    for i in range(n):  # the render: a blue-grey sky over summer grass, the road and the puddle gone
        r = np.full((h, w, 3), (70, 140, 50), np.uint8)
        r[:8] = (150, 170, 210)
        Image.fromarray(r).save(out / f"r_{i + 1:05d}.png")

    def fake_select(ctx, png, points, text, scale, out_dir):
        img = np.asarray(Image.open(png).convert("RGB"))
        return (np.abs(img.astype(int) - 71) < 10).all(axis=-1)

    monkeypatch.setattr(common, "select_preview", fake_select)
    monkeypatch.setattr(env, "camera_still", lambda work, masks: True)
    ctx = type("C", (), {"check": lambda self: None})()
    assert env.hold_ground(ctx, work, out, (w, h, 0, 0, w, h), tmp_path, None) == 2
    res = np.asarray(Image.open(out / "r_00002.png").convert("RGB")).astype(float)
    assert lum(res)[21].mean() < lum(res)[30].mean() - 30  # the road is back, its full width, darker than the field
    water = res[42, 40:60]
    assert water[:, 2].mean() > water[:, 1].mean() * 0.8 and abs(lum(water).mean() - lum(res[30:31, 40:60]).mean()) > 3  # not grass
    assert np.abs(res[30] - (70, 140, 50)).max() < 3  # the field is the render's
    frozen = env.hold_ground(ctx, work, out, (w, h, 0, 0, w, h), tmp_path / "f", None, frozen=True) if (tmp_path / "f").mkdir() is None else 0
    assert frozen == 1  # in snow the water is left as rendered; the road still holds


def test_the_renders_road_is_stretched_to_the_footages_width():
    h, w = 64, 128
    orig = np.zeros((h, w), bool)
    orig[20:27] = True  # the footage's road: 7 rows
    gen = np.zeros((h, w), bool)
    gen[22:25, 10:] = True  # the render's: 3 rows, missing at the left
    src = env.stretch_rows(orig, gen, 1.0)
    assert src is not None
    assert abs(src[20, 64] - 22) < 0.5 and abs(src[26, 64] - 24) < 0.5  # its top and bottom land on the render's
    assert abs(src[20, 2] - 22) < 0.5  # filled in where the render's road is missing
    assert src[40, 64] == 40  # off the road: the render as it is
    assert env.stretch_rows(orig, np.zeros_like(gen), 1.0) is None


def test_the_fogs_skyline_snaps_to_the_footages_own_edge():
    h, w = 120, 2000
    pic = np.full((h, w, 3), 60, np.uint8)  # dark land
    pic[:50] = (150, 155, 160)  # sky down to row 50, sharp
    soft = np.clip((60 - np.arange(h, dtype=np.float32)) / 8, 0, 1)[:, None] * np.ones((1, w), np.float32)
    # the depth model's sky: coarse, its middle 6 rows too far down
    sky = env.snap_sky(pic, soft)
    assert sky[:48].min() > 0.95 and sky[51:].max() < 0.05  # the edge is the footage's, one pixel wide


def test_autumn_turns_only_the_trees_and_grass_of_the_footage(tmp_path, monkeypatch):
    import pickups.app.store as st
    h, w, n = 96, 160, 5
    yy, xx = np.mgrid[:h, :w]
    tree = (yy < 50) & (xx < 60) & (yy > 10)
    grass = yy >= 70
    person = (yy > 40) & (yy < 90) & (xx > 100) & (xx < 115)
    rng = np.random.default_rng(3)
    f = np.zeros((h, w, 3), np.float32)
    f[:] = (140, 170, 215)  # sky
    f[50:70] = (90, 90, 92)  # road
    f[tree] = (40, 95, 35)
    f[grass] = (80, 140, 60)
    f[person] = (30, 120, 40)  # a green coat
    f = np.clip(f + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)
    work = tmp_path / "work.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(work)], stdin=subprocess.PIPE)
    p.communicate(f.tobytes() * n)
    regions = {"tree": tree, "bush": grass, "grass": grass, "meadow": np.zeros_like(tree), "person": person}

    class C:
        vdir, client = tmp_path, _Client()

        def stage(self, s):
            pass

        def check(self):
            pass

        def progress(self, x):
            pass

        def run(self, graph):
            labels = [graph[k]["inputs"]["text"].split(":")[0] for k in sorted(graph) if k.startswith("PU_Label")]
            shape = next(iter(common.read_frames(graph["PU_Clip"]["inputs"]["file"]))).shape[:2]
            outs = {}
            for j, label in enumerate(labels):
                m = np.asarray(Image.fromarray(regions[label].astype(np.uint8) * 255).resize(shape[::-1], Image.NEAREST))
                d = tmp_path / f"sam_{j}"
                d.mkdir(exist_ok=True)
                files = []
                for i in range(n):
                    Image.fromarray(m).save(q := d / f"mask_{i + 1:05d}_.png")
                    files.append(_Out(q))
                outs[f"PU_Mask{j if j else ''}"] = files
            return _Job(outs)

    (tmp_path / "work").mkdir()
    monkeypatch.setattr(common, "finish_outputs", lambda *a, **k: [])
    monkeypatch.setattr(st, "make_viewer", lambda *a, **k: None)
    res = env.autumn_take(C(), work, (w, h, 0, 0, w, h), 24.0, "Autumn")
    assert res["receipt"]["original"] == 100.0
    out = np.asarray(Image.open(tmp_path / "work" / "out" / "r_00003.png").convert("RGB")).astype(float)
    src = f.astype(float)
    t = out[composite.shrink(tree, 4)].mean(0)
    assert t[0] > t[1] and t[0] > t[2] + 10  # the tree is orange-red now
    g, g0 = out[composite.shrink(grass, 4)].mean(0), src[composite.shrink(grass, 4)].mean(0)
    assert g[0] > g0[0] + 15 and g[1] < g0[1]  # the grass is tawny
    for keep in ((yy > 54) & (yy < 66), yy < 8):  # the road and the sky as filmed
        assert np.abs(out[keep] - src[keep]).mean() < 2.5
    coat = composite.shrink(person, 4)  # the coat: its detail as filmed, with a faint tint of the new ground
    assert np.abs(out[coat] - src[coat]).mean() < 16
    assert abs(lum(out[composite.shrink(tree, 4)]).mean() - lum(src[composite.shrink(tree, 4)]).mean()) < 12  # its light kept


def test_night_is_a_moonlit_grade_with_a_night_sky_and_a_fill_on_people():
    h, w = 60, 80
    pic = np.full((h, w, 3), (120, 130, 110), np.uint8)  # a field by day
    pic[:20] = (150, 180, 230)  # the sky
    sky = np.zeros((h, w), np.float32)
    sky[:20] = 1
    people = np.zeros((h, w), np.float32)
    people[30:50, 30:40] = 1  # someone standing in the same field colour
    out = env.moon_grade(pic, sky, people).astype(float)
    field, person, night = out[55, 5], out[40, 35], out[5, 40]
    assert lum(field[None])[0] < 0.5 * lum(pic[55:56, 5].astype(float))[0]  # a few stops down
    assert field[2] > field[0]  # moonlight blue
    assert night[2] > night[0] + 10 and lum(night[None])[0] < 60  # a deep blue night sky
    assert lum(person[None])[0] > lum(field[None])[0] + 15  # the fill: a face still reads
