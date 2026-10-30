"""The render follows the previz (Run 1): the start still from the previz's own depth (method zbuf), the strength and its
one retry, the start still at position -1 with no keyframes by default, the prompt word filter, facing from the previz
and the size bands."""

import base64
import io
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickups import graphs, prompt_words
from pickups import start_frame as sf
from pickups.app import manifest, previz_maker
from pickups.media import Probe
from pickups.tasks import previz

W, H = previz_maker.SIZE
FIX = Path(__file__).parent / "fixtures"


def encode_depth(z: np.ndarray) -> Image.Image:
    """What runtime.ts depthPasses writes: z / DEPTH_FAR in 24 bits (red high), the sky (inf) pure black."""
    d = np.clip(np.where(np.isfinite(z), z, 0) / sf.DEPTH_FAR, 0, 0.999999)
    hi = np.floor(d * 255)
    r1 = (d - hi / 255) * 255
    mid = np.floor(r1 * 255)
    lo = np.round((r1 - mid / 255) * 255 * 255)
    rgb = np.stack([hi, mid, np.clip(lo, 0, 255)], -1).astype(np.uint8)
    rgb[~np.isfinite(z)] = 0
    return Image.fromarray(rgb)


def scene_z() -> np.ndarray:
    """A ground plane getting further up the frame (2 m at the bottom), a person 1.5 m away in the middle, sky in the top
    third."""
    rows = np.linspace(60, 2, H)[:, None] * np.ones((1, W))
    z = rows.astype(np.float32)
    z[: H // 3] = np.inf
    z[400:900, 900:1020] = 1.5
    return z


# --- 1. the depth export ----------------------------------------------------------------------------------------------

def test_the_depth_pass_round_trips_to_metres_and_the_sky_is_inf():
    z = scene_z()
    back = sf.decode_depth(encode_depth(z))
    assert back.shape == (H, W)
    assert np.isinf(back[: H // 3]).all() and np.isfinite(back[H // 3:]).all()
    assert np.allclose(back[np.isfinite(z)], z[np.isfinite(z)], atol=0.01)


def test_the_control_is_the_frame_size_sky_0_and_near_bright_like_the_da3_control():
    z = scene_z()
    da3 = Image.fromarray(np.linspace(250, 20, H)[::-1, None].repeat(W, 1).astype(np.uint8))  # DA3: near (bottom) bright
    da3_arr = np.asarray(da3).copy()
    da3_arr[: H // 3] = 0  # its sky
    tones = sf.da3_tones(Image.fromarray(da3_arr))
    ctl = sf.depth_control(z, tones, blur=0)
    a = np.asarray(ctl.convert("L"))
    assert ctl.size == (W, H) and ctl.mode == "RGB"
    assert (a[: H // 3] == 0).all()  # sky 0
    assert a[600, 960] > a[H - 1, 100] > a[H // 3 + 5, 100] > 0  # the person (1.5 m) over the near ground over the far ground
    # Rank-mapped onto the DA3 control's own tonal range (its non-sky values), same polarity: near bright.
    assert a[H // 3:].max() <= tones.max() and a[H // 3:].min() >= tones.min()
    blurred = np.asarray(sf.depth_control(z, tones).convert("L"))
    assert sf.ZBUF_BLUR <= 4 and abs(int(blurred[10, 10])) == 0  # the blur is 4 px or less: deep sky stays 0


def test_the_page_depth_passes_are_kept_lossless_at_the_frame_size(tmp_path):
    buf = io.BytesIO()
    encode_depth(scene_z()).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    small = io.BytesIO()
    Image.new("RGB", (64, 36)).save(small, "PNG")
    bad = "data:image/png;base64," + base64.b64encode(small.getvalue()).decode()
    assert previz_maker.save_depths(tmp_path, {"0": url, "48": bad, "7": url, "96": "nope"}) == [0]
    assert np.array_equal(np.asarray(Image.open(tmp_path / sf.DEPTH_FILE.format(0))), np.asarray(encode_depth(scene_z())))


# --- 1. strength by subject, the one retry ----------------------------------------------------------------------------

PERSON = {"kind": "person", "x": 0.5, "distance_m": 4, "in_view": True}
CAR = {"kind": "vehicle", "x": 0.3, "distance_m": 9, "in_view": True}


@pytest.mark.parametrize("subjects,first,retry", [
    ([PERSON], 0.75, 0.65), ([CAR], 0.65, 0.75),
    ([CAR, PERSON], 0.65, 0.75),  # mixed: the user-placed hero (the first subject in view) decides
    ([dict(PERSON, in_view=False), CAR], 0.65, 0.75), ([], 0.75, 0.65)])
def test_strength_by_the_hero_subject(subjects, first, retry):
    assert sf.strengths(sf.hero_kind({"subjects": subjects})) == (first, retry)


class FakeJob:
    def __init__(self, tag):
        self.tag = tag

    def get_outputs(self, _name):
        tag = self.tag

        class Out:
            def to_file(self, path):
                Image.new("RGB", (8, 8), (tag, tag, tag)).save(path)
        return [Out()]


class Runner:
    def __init__(self):
        self.graphs = []
        self.client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

    def run(self, g):
        self.graphs.append(g)
        strength = next((n["inputs"]["strength"] for n in g.values() if n["class_type"] == "ZImageFunControlnet"), None)
        return FakeJob(200 if strength is None else int(strength * 100))


def _run(tmp_path, monkeypatch, ratios, framing=None):
    encode_depth(scene_z()).save(tmp_path / "depth.png")
    Image.new("RGB", (W, H)).save(tmp_path / "f0.jpg")
    seq = iter(ratios)
    monkeypatch.setattr(sf, "size_from_depth", lambda d0, d1, box, matte=None: (lambda r: {"ratio": r, "state": sf.size_state(r), "warning": None})(next(seq)))
    runner = Runner()
    p = sf.Plan(positive="A woman on a street.", negative="", method=sf.ZBUF)
    framing = framing or {"subjects": [PERSON], "subject": [0.4, 0.3, 0.6, 0.9]}
    made = sf.make_zbuf(runner, p, tmp_path / "depth.png", tmp_path / "f0.jpg", framing, tmp_path / "start.png", seed=5,
                        patch="sub\\" + sf.ZBUF_PATCH)
    return made, runner


def test_a_passing_still_is_one_try_at_the_hero_strength(tmp_path, monkeypatch):
    made, runner = _run(tmp_path, monkeypatch, [1.02])
    assert [t["strength"] for t in made["tries"]] == [0.75] and made["flag"] is None and made["size"]["state"] == "pass"
    zb = [g for g in runner.graphs if "PU_Control" in g]
    assert len(zb) == 1 and zb[0]["patch"]["inputs"]["name"] == "sub\\" + sf.ZBUF_PATCH


def test_a_failed_size_retries_once_at_the_other_strength_and_keeps_the_better(tmp_path, monkeypatch):
    made, runner = _run(tmp_path, monkeypatch, [1.62, 1.04])
    assert [t["strength"] for t in made["tries"]] == [0.75, 0.65]
    assert made["strength"] == 0.65 and made["flag"] is None
    assert Image.open(tmp_path / "start.png").getpixel((0, 0))[0] == 65  # the retry's picture is the start frame
    assert Image.open(tmp_path / "start_try1.png").getpixel((0, 0))[0] == 75  # R2-10: try 1 is kept, never overwritten


def test_two_failures_are_flagged_and_never_a_third_try(tmp_path, monkeypatch):
    made, _ = _run(tmp_path, monkeypatch, [1.9, 1.6], framing={"subjects": [CAR], "subject": [0.1, 0.3, 0.9, 0.7]})
    assert [t["strength"] for t in made["tries"]] == [0.65, 0.75] and made["flag"] == "size" and made["strength"] == 0.75


def test_a_warning_alone_does_not_retry(tmp_path, monkeypatch):
    made, _ = _run(tmp_path, monkeypatch, [1.4])
    assert len(made["tries"]) == 1 and made["size"]["state"] == "warn"


def test_a_ghost_retries_like_a_failed_size():
    assert sf.needs_retry("pass", ghost=True) and sf.needs_retry("fail") and not sf.needs_retry("warn")


def test_the_zbuf_graph_is_the_reference_recipe():
    g = sf.zbuf_graph(sf.Plan(positive="x", negative="y"), "c.png", seed=7, strength=0.75)
    cn = g["PU_Control"]["inputs"]
    assert (cn["start_percent"], cn["end_percent"], cn["strength"]) == (0.0, 1.0, 0.75)  # control on all 8 steps
    assert g["sigmas"]["inputs"]["steps"] == 8 and g["guider"]["inputs"]["cfg"] == 1.0
    assert sum(n["class_type"] == "SamplerCustomAdvanced" for n in g.values()) == 1  # one sampler
    assert g["sampler"]["inputs"]["sampler_name"] == "res_multistep" and g["zms"]["inputs"]["shift"] == 3.0
    assert (g["lat"]["inputs"]["width"], g["lat"]["inputs"]["height"]) == (1920, 1088)


def test_the_2602_model_is_registered_with_its_public_source_and_sha():
    info = manifest.model_info(sf.ZBUF_PATCH)
    assert info["folder"] == "model_patches" and not info["gated"]
    assert info["url"] == "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/" + sf.ZBUF_PATCH
    assert info["sha256"] == "d1251cc7bc3486bc61d25c3be498ef394c31c85ddf4ee9137d2e933411f4a689"
    assert sf.ZBUF_PATCH in manifest.TASK_MODELS["previz"] and sf.ZBUF_PATCH in manifest.SOFT_MODELS["previz"]


class Conn:
    def __init__(self, names):
        self.names = set(names)

    def model_scan(self):
        return self.names, set()


def test_the_patch_is_found_in_a_subfolder_and_missing_falls_back():
    base = set(manifest.START_DEPTH[1:])
    assert sf.zbuf_patch(Conn(base | {"_test\\" + sf.ZBUF_PATCH})) == "_test\\" + sf.ZBUF_PATCH
    assert sf.zbuf_patch(Conn(base)) is None  # missing: the start frame goes the B3 way, never an error
    assert sf.zbuf_patch(Conn({sf.ZBUF_PATCH})) is None  # without Z-Image Turbo itself


def test_a_missing_2602_is_reported_but_never_blocks_previz(monkeypatch, tmp_path):
    from pickups.app import doctor
    have = {f for f in manifest.TASK_MODELS["previz"] if f != sf.ZBUF_PATCH} | set(manifest.QWEN21)

    class C:
        comfy_url = "http://127.0.0.1:8190"

        def system_stats(self):
            return {"system": {"comfyui_version": "0.41.0", "os": "nt", "ram_total": 64e9},
                    "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090", "type": "cuda", "vram_total": 24e9, "vram_free": 20e9}]}

        def model_scan(self):
            return have, set()

        def node_types(self):
            return set(manifest.TASK_NODES["previz"]) | manifest.STOCK_NODES

    monkeypatch.setattr(doctor, "load_settings", lambda: {}, raising=False)
    out = doctor.status(C()) if hasattr(doctor, "status") else None
    if out is None:
        pytest.skip("doctor has no status(conn)")
    tile = out["tiles"]["previz"]
    assert [m["file"] for m in tile["missing"]] == [sf.ZBUF_PATCH] and tile["reason"] is None


# --- 2. the video: position -1, no keyframes by default -----------------------------------------------------------------

def _drive(tmp_path, monkeypatch, params, depths=False, meta=None):
    seen = {}
    real_layout = graphs.layout_render

    def prepare(src, dst, start, frames, size, **kw):
        Path(dst).write_bytes(b"")
        return {"fps": Fraction(24)}

    def layout(layout_, look, prompt, fps, *a, keyframes=None, **k):
        seen.update(look=look, prompt=prompt, keyframes=keyframes)
        return real_layout(layout_, look, prompt, fps, keyframes=keyframes)

    monkeypatch.setattr(previz.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), 97, False, None))
    monkeypatch.setattr(previz.media, "prepare_work_clip", prepare)
    monkeypatch.setattr(previz.bench, "machine", lambda conn: {"vram_gb": 24})
    monkeypatch.setattr(previz.graphs, "layout_render", layout)
    monkeypatch.setattr(previz.subprocess, "run", lambda cmd, **k: Image.new("RGB", (W, H)).save(cmd[-1]))
    monkeypatch.setattr(sf, "zbuf_patch", lambda conn: sf.ZBUF_PATCH)
    if depths:
        for f in previz_maker.DEPTH_FRAMES:
            encode_depth(scene_z()).save(tmp_path / sf.DEPTH_FILE.format(f))
    (tmp_path / "look.png").write_bytes(b"")

    class Store:
        def shot_cast(self, shot_id):
            return [], lambda c: None

        def shot_meta(self, shot_id):
            return meta or {}

        def shot_dir(self, shot_id):
            return tmp_path

        def upload_path(self, uid):
            return tmp_path / "look.png"

    class Ctx:
        input_path = Path("in.mp4")
        vdir = tmp_path
        conn = None
        project_id = "prj"
        shot_id = "sht"
        store = Store()
        client = Runner().client

        def __init__(self):
            self.params = params
            self.graphs = []

        def stage(self, _label):
            return None

        def commercial_safe(self):
            return False

        def run(self, g):
            self.graphs.append(g)
            if "PU_Layout" in g:
                seen["graph"] = g
                raise RuntimeError("stop after the graph")
            return FakeJob(120)

    ctx = Ctx()
    with pytest.raises(RuntimeError, match="stop after the graph"):
        previz.run(ctx)
    seen["all"] = ctx.graphs
    return seen


def test_the_start_still_goes_in_at_position_minus_1_only_by_default(tmp_path, monkeypatch):
    seen = _drive(tmp_path, monkeypatch, {"prompt": "She walks.", "look_frame_upload_id": "mine"}, depths=True)
    assert seen["keyframes"] == []
    g = seen["graph"]
    looks = [n for n in g.values() if n["class_type"] == "LTXAddVideoICLoRAGuide" and n["inputs"]["image"] == ["PU_Look", 0]]
    assert len(looks) == 2 and all(n["inputs"]["frame_idx"] == -1 for n in looks)  # both stages
    assert not any(n["class_type"] in ("LTXVAddGuide", "LTXVImgToVideoInplace") for n in g.values())  # no pin, no keys
    assert not any("PU_Control" in x for x in seen["all"])  # no stills made


def test_the_rescue_keyframes_are_new_zimage_stills_at_48_and_96(tmp_path, monkeypatch):
    meta = {"previz_spec": {"notes": "A woman walks."}, "previz_framing": {"subjects": [PERSON], "camera": {"height_m": 1.6, "pitch_deg": 0, "focal_mm": 35}}}
    seen = _drive(tmp_path, monkeypatch, {"prompt": "She walks.", "look_frame_upload_id": "mine", "keyframes": True}, depths=True, meta=meta)
    assert [k[1] for k in seen["keyframes"]] == [48, 96]
    assert all(k[0] != seen["look"] for k in seen["keyframes"])  # never the start still again
    stills = [x for x in seen["all"] if "PU_Control" in x]
    assert len(stills) == 2 and all(x["PU_Control"]["inputs"]["strength"] == 0.75 for x in stills)
    adds = [n for n in seen["graph"].values() if n["class_type"] == "LTXVAddGuide"]
    assert sorted(n["inputs"]["frame_idx"] for n in adds) == [48, 48, 96, 96]  # both stages


def test_the_video_prompt_is_the_short_layout_to_render_one(tmp_path, monkeypatch):
    meta = {"previz_framing": {"subjects": [PERSON, dict(PERSON, x=0.7)]}}
    seen = _drive(tmp_path, monkeypatch, {"prompt": "Two friends hug on a grey previz street. The camera pulls back. "
                                                    "They stay side by side; nobody crosses in front of anyone. Warm sun.",
                                          "look_frame_upload_id": "mine"}, meta=meta)
    assert seen["prompt"] == "The camera pulls back. They stay side by side. Keep the same two people and camera as the driver."


# --- 5. the prompt word filter ------------------------------------------------------------------------------------------

BANNED = ("previz", "mannequin", "blockout", "backdrop", "studio", "clay", "3D", "small vehicles")


def _no_banned(text: str):
    for w in BANNED:
        assert not re.search(rf"(?i)(?<![\w-]){re.escape(w)}(?![\w-])", text), w
    # Run 2: a real grey stays grey; a grey stand-in never reaches a positive.
    assert not any(re.search(r"(?i)\bgr[ae]y\b", s) and re.search(prompt_words.STAND_IN, s, re.I)
                   for s in re.split(r"(?<=[.!?])\s+", text)), text
    assert not re.search(r"(?i)\b(no|not|never|without|nobody)\b", text), text


@pytest.mark.parametrize("dirty", [
    "A grey mannequin stands on a clay blockout street in front of a studio backdrop.",
    "Further ahead, small vehicles in a 3D previz.",
    "Tyres with real tread and no readable badges, never pure black, nobody crosses.",
])
def test_the_filter_takes_every_banned_word_and_negation_out(dirty):
    clean = prompt_words.clean_positive(dirty)
    _no_banned(clean)
    assert prompt_words.banned_in(dirty) and not prompt_words.banned_in(clean)


def test_a_real_grey_stays_grey_and_the_rest_of_the_sentence_stays():
    # Run 2 (R&D #10): a grey cap came out charcoal, near black.
    assert prompt_words.clean_positive("He has a short grey beard and a grey flat cap.") == "He has a short grey beard and a grey flat cap."
    assert prompt_words.clean_positive("Tyres with real tread and no readable badges.") == "Tyres with real tread and clean unbadged bodywork."
    assert prompt_words.clean_positive("A grey figure stands by grey shapes.") == ""


SHOT20 = sorted((FIX / "step20").glob("*.json"))


@pytest.mark.parametrize("method", [sf.METHOD, sf.ZBUF, "commercial-safe"])
@pytest.mark.parametrize("shot", SHOT20, ids=lambda p: p.stem)
def test_no_start_frame_positive_has_a_banned_word_and_the_negative_keeps_them(shot, method):
    d = json.loads(shot.read_text())
    refs = [Path(f"{c['name']}.png") for c in d["cast"]]
    p = sf.plan(d["spec"], d["framing"], d["cast"], refs, d["cast"], d["user_look"] + " A grey mannequin, no badges.",
                matte=Path("m.png"), method=method)
    _no_banned(p.positive)
    assert "grey mannequin" in p.negative and "clay render" in p.negative and "3D render" in p.negative


def test_zbuf_names_references_in_words_only():
    d = json.loads((FIX / "step20" / "s05.json").read_text())
    p = sf.plan(d["spec"], d["framing"], d["cast"], [Path("a.png")] * len(d["cast"]), d["cast"], "", method=sf.ZBUF)
    assert p.refs == [] and "image 3" not in p.positive and "in the layout" not in p.positive and p.action is None


def test_the_l2r_prompt_is_one_or_two_sentences_then_the_keep_line():
    two = [PERSON, PERSON]
    out = prompt_words.l2r_prompt("A woman waves. The camera orbits left. A boat drifts. Warm light.", [PERSON])
    assert out == "A woman waves. The camera orbits left. Keep the same person and camera as the driver."
    assert prompt_words.l2r_prompt("A car races. Keep the same car, road and camera as the driver.", [CAR]).endswith("road and camera as the driver.")
    assert prompt_words.l2r_prompt("They hug.", two + [CAR]).endswith("Keep the same two people and car and camera as the driver.")
    _no_banned(prompt_words.l2r_prompt("A grey previz man walks; nobody follows. He waves.", [PERSON]))


def test_the_keep_line_names_a_lone_person_by_the_previz_word():
    assert prompt_words.keep_subjects([dict(PERSON, word="woman")]) == "woman"
    assert prompt_words.keep_subjects([dict(PERSON, word="man"), CAR]) == "man and car"
    assert prompt_words.keep_subjects([dict(PERSON, word="friends"), PERSON]) == "two people"
    assert prompt_words.keep_subjects([PERSON]) == "person"  # no word from the previz: nothing better to say
    assert prompt_words.keep_subjects([dict(CAR, word="truck")]) == "truck"


# The short lines the page writes (motionPrompt.ts motionPromptFor; web/tests/previz/handoff_words.mjs builds them):
# subject, action, place and light, the camera, the Keep line with the place noun.
SHORT_LINES = [
    "A woman in her 40s with long wavy auburn hair in a cream chunky knit sweater standing on the shore of a lake waving. "
    "The camera orbits to the left around them and pushes in. Keep the same woman, lake and camera as the driver.",
    "A man in his 60s with a short silver beard in a navy wool peacoat sitting on a bench along a path in the park. The "
    "camera pushes in. Keep the same man, park and camera as the driver.",
    "A woman sits alone in a diner booth at night. The camera pushes in. Keep the same woman, booth and camera as the "
    "driver.",
    "Three friends dancing on a city rooftop in low golden-hour sun. The camera cranes down. Keep the same three people, "
    "rooftop and camera as the driver.",
    "A dark green 1970s fastback muscle coupe driving along an open highway. Filmed from a camera car driving alongside at "
    "the same speed, the road and background streaming past. Keep the same car, road and camera as the driver.",
]


@pytest.mark.parametrize("line", SHORT_LINES)
def test_the_short_line_passes_the_word_filter_and_the_server_keeps_it_whole(line):
    assert prompt_words.banned_in(line) == []
    assert prompt_words.l2r_prompt(line, [PERSON]) == line
    assert "km/h" not in line and not re.search(r"(?i)\blight\w* (sweep|change|fade|shift)", line)


# --- 3. facing from the previz rotation -------------------------------------------------------------------------------

S05 = {"notes": "Two friends hug on a city sidewalk."}


def _framing(facing, kind="person"):
    return {"subjects": [{"kind": kind, "x": 0.5, "distance_m": 6, "in_view": True, "facing": facing}],
            "camera": {"height_m": 1.6, "pitch_deg": 0, "focal_mm": 35}}


@pytest.mark.parametrize("facing,words", [("away", "seen from behind, back to the viewer"),
                                          ("camera", "facing the viewer, face toward the viewer"),
                                          ("left", "seen in profile, head and body turned to frame left"),
                                          ("right", "seen in profile, head and body turned to frame right")])
def test_facing_comes_from_the_previz_rotation(facing, words):
    p = sf.plan(S05, _framing(facing), [], [], [], method=sf.ZBUF)
    assert f"Person 1: centre of frame, about 6 metres from the viewer, {words}" in p.positive and p.flags == []


@pytest.mark.parametrize("facing,words", [
    ("camera", "seen dead on from the front, squared to the viewer, headlights evenly either side of the centre line"),
    ("away", "seen directly from behind, driving away from the viewer"),
    ("left", "seen side-on in profile, its front end and headlights pointing to frame left"),
    ("right", "seen side-on in profile, its front end and headlights pointing to frame right")])
def test_a_vehicle_faces_by_its_front_end_and_headlights(facing, words):
    p = sf.plan({"notes": "A car drives."}, _framing(facing, "vehicle"), [], [], [], method=sf.ZBUF)
    assert words in p.positive and "nose" not in p.positive


def test_the_facing_words_pass_the_word_filter_and_read_back_as_their_own_facing():
    for words in [*sf.FACING_PERSON.values(), *sf.FACING_VEHICLE.values()]:
        assert prompt_words.banned_in(words) == [] and prompt_words.clean_positive(words) == words[0].upper() + words[1:] + "."
    # The words the app writes agree with the user-words reader, so a plan never flags its own facing.
    for facing in ("camera", "away"):
        assert sf.said_facing(sf.FACING_PERSON[facing]) == facing
    assert sf.said_facing(sf.FACING_VEHICLE["camera"]) == "camera"


@pytest.mark.parametrize("notes,doing", [("A woman hugs her friend on a city sidewalk.", "hug"),
                                         ("They dance together on a rooftop.", "dance"), ("A man waves.", None)])
def test_a_lone_person_is_never_called_two_different_people(notes, doing):
    framing = _framing("camera")
    framing["subjects"][0]["doing"] = doing
    for method in (sf.ZBUF, sf.METHOD):
        p = sf.plan({"notes": notes}, framing, [], [], [], matte=Path("m"), method=method)
        assert "different people" not in p.positive and sf.GROUP_NO_REF not in p.positive
        assert sf.GROUP_NEGATIVE not in p.negative and "EXACTLY 1 person" in p.positive
    # A lone person beside a car is still one person.
    framing["subjects"].append(dict(_framing("left", "vehicle")["subjects"][0], x=0.2))
    p = sf.plan({"notes": notes}, framing, [], [], [], method=sf.ZBUF)
    assert "different people" not in p.positive and sf.GROUP_NO_REF not in p.positive


def test_when_the_words_disagree_the_previz_wins_and_it_is_flagged():
    p = sf.plan(S05, _framing("away"), [], [], [], user_look="She walks toward the camera. Warm evening light on her coat.", method=sf.ZBUF)
    assert "toward the camera" not in p.positive and "seen from behind" in p.positive
    assert len(p.flags) == 1 and p.flags[0].startswith("Facing: the previz has person 1 seen from behind")
    agree = sf.plan(S05, _framing("away"), [], [], [], user_look="Seen from behind, she walks away.", method=sf.ZBUF)
    assert agree.flags == []


def test_a_pair_facing_two_ways_agrees_with_words_for_either():
    pair = {"subjects": [dict(_framing("away")["subjects"][0], x=0.3), dict(_framing("camera")["subjects"][0], x=0.7)],
            "camera": _framing("away")["camera"]}
    p = sf.plan(S05, pair, [], [], [], user_look="They walk toward the camera.", method=sf.ZBUF)
    assert p.flags == [] and "They walk toward the viewer." in p.positive  # Run 2 (d): no camera words in a still
    p = sf.plan(S05, pair, [], [], [], user_look="Both in side profile, facing frame left.", method=sf.ZBUF)
    assert p.flags == ["Facing: no one in the previz faces the way the words said, so the previz wins."]


def test_the_page_facing_is_kept_by_the_shot():
    fr = previz_maker._framing({"subjects": [{"kind": "person", "x": 0.5, "distance_m": 3, "in_view": True, "facing": "left"},
                                             {"kind": "person", "x": 0.2, "distance_m": 3, "in_view": True, "facing": "up"}]})
    assert fr["subjects"][0]["facing"] == "left" and "facing" not in fr["subjects"][1]


# --- 4. the size bands --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ratio,state", [
    (1.0, "pass"), (1.15, "pass"), (1.1501, "warn"), (1.5, "warn"), (1.5001, "fail"), (2.4, "fail"),
    (0.85, "pass"), (0.8499, "warn"), (0.67, "warn"), (0.66, "small"), (0.5, "small"), (None, "unmeasured"), (0, "unmeasured")])
def test_the_size_bands_at_their_edges(ratio, state):
    assert sf.size_state(ratio) == state
    assert sf.needs_retry(state) is (state in ("fail", "small"))  # R2-5: far too small retries as well as far too big


def test_the_size_check_reads_the_existing_edge_match():
    got = {"subject_sure": True, "subject_h": [0.4, 0.62], "warning": None}
    assert sf.size_from(got) == {"ratio": 1.55, "state": "fail", "warning": "The subject is 55% bigger than the previz."}
    assert sf.size_from({"subject_sure": False, "subject_h": [0.4, 0.8]})["state"] == "unmeasured"
    assert sf.size_from({"subject_sure": True, "subject_h": [0.4, 0.3]})["warning"] == "The subject is about 0.8x the previz size."


def test_a_size_warning_with_no_retry_says_the_size_in_plain_words():
    # s04 on e91bf96 measured 1.455: a warning, no retry (SIZE_FAIL stays 0.50), and the line says what it means
    assert sf.SIZE_FAIL == 0.50 and sf.size_state(1.455) == "warn" and not sf.needs_retry("warn")
    assert sf.size_warning("warn", 1.455) == "The subject is about 1.5x the previz size."
    assert sf.size_warning("warn", 0.7) == "The subject is about 0.7x the previz size."
    assert sf.size_warning("pass", 1.01) is None and sf.size_warning("unmeasured", None) is None


# --- graphs against ComfyUI's own node specs (an RTX 4090 ComfyUI's /object_info, trimmed to these nodes, no file lists) ---------------

SPECS = json.loads((FIX / "object_info_previz.json").read_text())


def _valid(g: dict):
    for nid, n in g.items():
        spec = SPECS[n["class_type"]]["input"]
        req, opt = spec.get("required", {}), spec.get("optional", {})
        for k in req:
            assert k in n["inputs"], (nid, k)
        for k, v in n["inputs"].items():
            assert k in req or k in opt or k.split(".")[0] in req, (nid, k)  # "output.normalization": a dynamic combo's own
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g and v[1] < len(SPECS[g[v[0]]["class_type"]]["output"]), (nid, k, v)


@pytest.mark.parametrize("build", [
    lambda: graphs.start_frame_zbuf("c.png", "x", strength=0.65),
    lambda: graphs.start_frame_maps("f.png"),
    lambda: graphs.layout_render("l.mp4", "look.png", "x", 24.0),
    lambda: graphs.layout_render("l.mp4", "look.png", "x", 24.0, keyframes=[("a.png", 48, 1.0), ("b.png", 96, 1.0)]),
])
def test_the_graphs_match_comfyuis_node_specs(build):
    _valid(build())
