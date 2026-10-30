"""Run 2: the prompt and pipeline fixes from Run 1/1b (R&D's BUGS_FINAL.md, the director's PROMPT_BUG_FIXES and verdict).

Each test names its bug: facing drops only a subject's clause (#2), no camera words in a still (#3), the Look text's
ground wins over the location base (#1), the size check measures and its retry fires (#5), the previz size in words,
a previz sentence goes whole (#9), positive phrasings (#8), grey stays grey (#10), no km/h (#11), one distance (#12),
orbit and crane keyframes by default with the start still's recipe (#13), and PICKUPS_KEEP_WORK keeps work (#16).
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickups import prompt_words
from pickups import start_frame as sf
from pickups.app import store as store_mod
from pickups.tasks import previz

CAM = {"height_m": 1.6, "pitch_deg": 0, "focal_mm": 35}


def _framing(*subjects, **more):
    return {"camera": CAM, "subjects": list(subjects), **more}


def _person(facing="away", **kw):
    return {"kind": "person", "x": 0.5, "distance_m": 6, "in_view": True, "facing": facing, **kw}


def _car(facing="left", **kw):
    return {"kind": "vehicle", "x": 0.5, "distance_m": 8, "in_view": True, "facing": facing, **kw}


# --- #2 facing: only a subject's clause, never the whole sentence ---------------------------------------------------------

def test_s02_bench_turned_toward_camera_is_not_a_facing_claim():
    look = ("At the right of the frame a green slatted wooden park bench sits empty on the lawn, turned about forty-five "
            "degrees toward camera, and beyond it a single large maple.")
    p = sf.plan({"notes": "A man jogs in the park."}, _framing(_person("left")), [], [], [], user_look=look, method=sf.ZBUF)
    assert p.flags == [] and "single large maple" in p.positive and "park bench sits empty" in p.positive


def test_s01_shadow_toward_frame_left_keeps_its_light_sentence():
    look = "Late-afternoon sun from camera right casts a long soft shadow across the pavement toward frame left."
    p = sf.plan({"notes": "Maya walks."}, _framing(_person("camera")), [], [], [], user_look=look, method=sf.ZBUF)
    assert p.flags == [] and "long soft shadow across the pavement toward frame left" in p.positive


def test_only_the_subjects_clause_goes_and_the_rest_of_the_sentence_stays():
    p = sf.plan({"notes": "A woman walks."}, _framing(_person("away")), [], [], [],
                user_look="She walks toward the camera, her red coat bright against the wall.", method=sf.ZBUF)
    assert len(p.flags) == 1 and "toward the viewer" not in p.positive
    assert "Her red coat bright against the wall." in p.positive


# --- #3 cameras as props -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", [sf.ZBUF, sf.METHOD, "commercial-safe"])
def test_a_still_never_names_a_camera_a_crew_or_filming(method):
    look = "Shot on a cinema camera with lens flare. Light from camera left. She looks toward camera. A film crew nearby."
    framing = _framing(_person("camera", x=0.3), _car("left", x=0.7, speed_ms=20, doing="drive"), light={"dir": "behind camera"})
    p = sf.plan({"notes": "A woman by a car."}, framing, [], [], [], user_look=look, matte=Path("m.png"), method=method)
    assert not re.search(r"(?i)\b(camera|filmed|crew|lens flare|shot on)\b", p.positive), p.positive
    assert "frame left" in p.positive and sf.CAMERA_NEGATIVE in p.negative
    assert p.positive.startswith(sf.FRAME_LINE.split(" EXACTLY")[0])


# --- #1 the world paragraph ------------------------------------------------------------------------------------------------

def _world(p):
    return next(x for x in p.positive.split("\n\n") if x.startswith(("Pale packed", "Worn real", "Cracked", "A ", "Still lake")) or ", weathered and real" in x)


def test_c09_look_text_names_the_ground_so_no_location_base():
    look = "A dirt farm track runs past an old farmhouse with pale clapboard walls."
    w = sf.world_sentence(look, "A car parked on the outskirts.", "", ["buildings", "a post", "the road"])
    assert "asphalt" not in w and "lane paint" not in w and "packed earth" not in w
    # R2-7: the Look describes the place, so the post gets no generic "weathered and real" phrase.
    assert w == "A small weathered house with painted clapboard or plaster walls and a shingle roof."


def test_the_rural_row_comes_before_the_street_and_no_road_is_never_a_street():
    assert sf.location("", "A car parked on the outskirts.", "")[1].startswith("pale packed earth")
    assert sf.location("", "A woman waves.", "", ["a post"]) == (sf.NEUTRAL_KEY, sf.NEUTRAL_BASE)
    assert "reeds" not in sf.location("", "By the lake.", "")[1] and "granite" in sf.location("", "By the lake.", "")[1]


def test_a_light_sentence_naming_the_roof_does_not_skip_the_base():
    w = sf.world_sentence("Low golden sun rakes across the rooftop.", "Three friends dance on a rooftop.", "", ["a bench"])
    assert w.startswith("Tar and gravel roof surface")


# --- #5 the size check measures, and the retry fires -------------------------------------------------------------------------

def _depth(h_frac: float, box=(0.45, 0.4, 0.55, 0.8)) -> Image.Image:
    """A depth map (near bright): ground brighter toward the bottom, sky 0, a figure of h_frac standing on the box's feet."""
    H, W = 272, 480
    d = np.zeros((H, W), np.float32)
    rows = np.arange(H) / H
    d[rows > 0.5] = (40 + 200 * (rows[rows > 0.5] - 0.5) / 0.5)[:, None]
    feet = int(box[3] * H)
    top = max(0, feet - int(h_frac * H))
    x0, x1 = int(box[0] * W), int(box[2] * W)
    d[top:feet, x0:x1] = d[feet - 1, 0] + 30  # the figure stands out from the ground behind it
    return Image.fromarray(d.clip(0, 255).astype(np.uint8))


def _matte(h_frac: float, box=(0.45, 0.4, 0.55, 0.8)) -> Image.Image:
    """The previz subject matte: the figure white, standing on the box's feet."""
    m = Image.new("L", (480, 272), 0)
    feet = int(box[3] * 272)
    m.paste(255, (int(box[0] * 480), max(0, feet - int(h_frac * 272)), int(box[2] * 480), feet))
    return m


def test_depth_height_measures_like_for_like():
    box = [0.45, 0.4, 0.55, 0.8]
    m = _matte(0.4)
    same = sf.size_from_depth(_depth(0.4), _depth(0.4), box, m)
    big = sf.size_from_depth(_depth(0.2), _depth(0.4), box, _matte(0.2))  # previz 20% tall, the still 40%: twice the size
    small = sf.size_from_depth(_depth(0.4), _depth(0.2), box, m)  # half the size
    assert same["state"] == "pass" and same["ratio"] == pytest.approx(1.0, abs=0.05)
    assert big["state"] == "fail" and big["ratio"] == pytest.approx(2.0, rel=0.15)
    assert small["state"] == "small" and small["ratio"] == pytest.approx(0.5, rel=0.15) and sf.needs_retry(small["state"])
    flat = Image.new("L", (480, 272), 90)
    assert sf.size_from_depth(flat, flat, box, m)["state"] == "unmeasured"


def test_r2_5_the_seed_is_the_subject_not_the_nearest_ground():
    """R&D's s04: the box's nearest depth is the lawn, which ran the frame's width and passed a 3.6x still. The matte's
    median depth is the figure; a region that runs the frame's width is no measure."""
    box = [0.45, 0.4, 0.55, 0.8]
    d = np.asarray(_depth(0.2)).copy()
    d[int(0.79 * 272):int(0.81 * 272), :] = 255  # the nearest ground in the box, a band across the whole frame
    seeded = sf.depth_height(Image.fromarray(d), box, _matte(0.2))
    assert seeded == pytest.approx(0.2, abs=0.03)
    ground = np.zeros((272, 480), np.uint8)
    ground[136:] = np.linspace(40, 240, 136, dtype=np.uint8)[:, None]  # no figure: the matte sits on bare ground
    assert sf.depth_height(Image.fromarray(ground), [0.0, 0.5, 1.0, 1.0], Image.new("L", (480, 272), 255)) is None


class _Out:
    def __init__(self, img):
        self.img = img

    def to_file(self, path):
        self.img.save(path)


class _Job:
    def __init__(self, outs):
        self.outs = outs

    def get_outputs(self, name):
        return [_Out(self.outs[name])]


class _Runner:
    """The fake ComfyUI for make_zbuf: DA3 jobs answer depth maps in order, a still job a plain picture."""

    def __init__(self, depths):
        self.depths = list(depths)
        self.stills = 0
        self.client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

    def run(self, g):
        if "PU_Plate" in g:
            self.stills += 1
            return _Job({"PU_Plate": Image.new("RGB", (64, 36), (90, 90, 90))})
        d = self.depths.pop(0)
        return _Job({"PU_Depth": d, "PU_Canny": d})


def test_a_still_twice_the_previz_size_fails_and_the_one_retry_fires(tmp_path):
    box = [0.45, 0.4, 0.55, 0.8]
    framing = {"subject": box, "subjects": [_person("camera")]}
    frame0 = tmp_path / "f0.png"
    Image.new("RGB", (64, 36)).save(frame0)
    zbuf = tmp_path / "depth0.png"
    Image.new("RGB", (480, 272), (1, 0, 0)).save(zbuf)
    p = sf.Plan(positive="p", negative="n")
    matte = tmp_path / "subjects0.png"
    _matte(0.2).save(matte)
    r = _Runner([_depth(0.2), _depth(0.4), _depth(0.21)])  # frame 0, the first still (2x), the retry (about right)
    made = sf.make_zbuf(r, p, zbuf, frame0, framing, tmp_path / "start.png", matte=matte)
    assert r.stills == 2 and [t["size"]["state"] for t in made["tries"]] == ["fail", "pass"]
    assert made["size"]["state"] == "pass" and made["strength"] == sf.strengths("person")[1] and made["flag"] is None
    assert (tmp_path / "start_try1.png").is_file()  # R2-10


def test_a_wrong_people_count_retries_once_with_the_next_seed_then_flags(tmp_path):
    """Run 3 (s06): the still's people (the app's detector) against the previz's; one retry at the next seed, try 1 kept,
    then a flag."""
    box = [0.45, 0.4, 0.55, 0.8]
    framing = {"subject": box, "subjects": [_person("camera"), _person("camera", x=0.2), _person("camera", x=0.8)]}
    frame0 = tmp_path / "f0.png"
    Image.new("RGB", (64, 36)).save(frame0)
    zbuf = tmp_path / "depth0.png"
    Image.new("RGB", (480, 272), (1, 0, 0)).save(zbuf)
    p = sf.Plan(positive="p", negative="n")
    seeds = []

    class R(_Runner):
        def run(self, g):
            if "PU_Plate" in g:
                seeds.append(next(n["inputs"]["noise_seed"] for n in g.values() if "noise_seed" in n.get("inputs", {})))
            return super().run(g)

    counts = iter([4, 3])
    made = sf.make_zbuf(R([_depth(0.4)] * 3), p, zbuf, frame0, framing, tmp_path / "s.png", seed=2002,
                        count=lambda png: next(counts))
    assert seeds == [2002, 2003] and [t["people"] for t in made["tries"]] == [4, 3]
    assert made["seed"] == 2003 and made["flag"] is None and (tmp_path / "s_try1.png").is_file()
    counts = iter([4, 4])
    flagged = sf.make_zbuf(R([_depth(0.4)] * 3), p, zbuf, frame0, framing, tmp_path / "t.png", seed=1, count=lambda png: next(counts))
    assert len(flagged["tries"]) == 2 and flagged["flag"] == "count" and flagged["warning"] == "The still has 4 people; the previz has 3."
    one = sf.make_zbuf(R([_depth(0.4)] * 2), p, zbuf, frame0, framing, tmp_path / "u.png", count=lambda png: 3)
    assert len(one["tries"]) == 1 and not (tmp_path / "u_try1.png").exists()


# --- size, distance and speed in the words ------------------------------------------------------------------------------------

def test_the_previz_size_is_a_frame_relative_number():
    p = sf.plan({"notes": "Marcus sits."}, _framing(_person("camera", doing="sit", h=0.25)), [], [], [], method=sf.ZBUF)
    assert "Person 1's head is about 4 percent of the frame height." in p.positive  # 0.25 x 0.18 sitting
    c = sf.plan({"notes": "A car drives."}, _framing(_car("left", w=0.12, speed_ms=20)), [], [], [], method=sf.ZBUF)
    assert "Vehicle 1 fills about 12 percent of the frame width." in c.positive
    old = sf.plan({"notes": "A woman."}, {**_framing(_person("camera")), "subject": [0.4, 0.2, 0.6, 0.66]}, [], [], [], method=sf.ZBUF)
    assert "head is about 6 percent of the frame height" in old.positive  # an older shot: the hero's framing box


def test_one_distance_the_previz_wins_and_no_kmh():
    look = ("The car is about fourteen metres from camera. It fills about a third of the frame width. It does 80 km/h. "
            "The car sits centre frame with the farmhouse at the right.")
    p = sf.plan({"notes": "A car drives."}, _framing(_car("left", speed_ms=22)), [], [], [], user_look=look, method=sf.ZBUF)
    assert p.positive.count("metres from") == 1 and "about 8 metres from the viewer" in p.positive
    assert "fourteen" not in p.positive and "third of the frame" not in p.positive
    assert not re.search(r"(?i)km/h|\b80\b|camera car", p.positive)
    assert "The car sits centre frame with the farmhouse at the right." in p.positive


# --- #9 previz cut, #8 positive phrasings, #10 grey --------------------------------------------------------------------------------

def test_r2_8_only_the_previz_clause_goes_never_the_sentence():
    look = "A white car is parked where the previz places it, on the park lane behind the bench. Autumn leaves on the lawn."
    p = sf.plan({"notes": "Marcus sits."}, _framing(_person("camera")), [], [], [], user_look=look, method=sf.ZBUF)
    assert "A white car is parked on the park lane behind the bench. Autumn leaves on the lawn." in p.positive
    assert "previz" not in p.positive and "where the" not in p.positive
    assert any("where the previz places it" in t for t in p.todo)  # the user sees what was taken out
    assert sf.strip_banned("Keep the previz framing, with warm light.") == "Keep the framing, with warm light."
    assert sf.strip_banned("A cinematic rainy look. Real grain.") == "A look. Real grain."


C07_LOOK = ("The car is about seven metres away and fills roughly the middle third of the frame width, with a driver at the "
            "wheel and its wheels blurred with motion.")


def test_r2_12_c07_keeps_the_driver_and_drops_only_the_distance_clauses():
    got, gone = sf.strip_clauses(C07_LOOK, sf.USER_DISTANCE)
    assert got == "The car has a driver at the wheel and its wheels blurred with motion."
    assert gone == ["The car is about seven metres away", "fills roughly the middle third of the frame width"]
    p = sf.plan({"notes": "A car drives."}, _framing(_car("left", speed_ms=20)), [], [], [], user_look=C07_LOOK, method=sf.ZBUF)
    assert "The car has a driver at the wheel and its wheels blurred with motion." in p.positive
    assert sum("Taken out of your words" in t for t in p.todo) == 2


def test_a_clause_strip_keeps_whatever_is_left_and_drops_an_empty_sentence():
    text = "It fills about a third of the frame width. A boy waves, about 3 metres away, in a red coat. Leaves fall."
    got, gone = sf.strip_clauses(text, sf.USER_DISTANCE)
    assert got == "A boy waves, in a red coat. Leaves fall." and len(gone) == 2
    assert sf.strip_clauses("He is 40 percent of the frame tall.", sf.USER_DISTANCE)[0] == ""


def test_run4_an_and_inside_a_noun_phrase_never_splits_and_the_verb_stays():
    got, gone = sf.strip_clauses("A man in a navy coat and grey cap stands about three metres from the viewer.", sf.USER_DISTANCE)
    assert got == "A man in a navy coat and grey cap stands." and gone == ["about three metres from the viewer"]
    # an " and " after the verb still splits (c07 above), and a sentence with no match is untouched
    got, gone = sf.strip_clauses("A woman in a red coat and black boots waits. The car is parked about ten metres away.", sf.USER_DISTANCE)
    assert got == "A woman in a red coat and black boots waits. The car is parked." and gone == ["about ten metres away"]


@pytest.mark.parametrize("text", ["The car fills a quarter of the frame.", "She fills half the frame.", "It fills a third of the frame width.",
                                  "The van fills two thirds of the frame."])
def test_run4_user_distance_matches_a_bare_fraction_of_the_frame(text):
    assert re.search(sf.USER_DISTANCE, text)
    assert sf.strip_clauses(text, sf.USER_DISTANCE)[0] == ""
    assert not re.search(sf.USER_DISTANCE, "Half the frame is sky. A quarter of the crowd waves.")


def test_c07_cast_line_keeps_its_constraints_in_positive_words():
    line = ("a dark green 1970s two-door coupe with chrome bumpers, a smooth, unbroken tail with a flat trunk lid and no "
            "spoiler, and clean flanks with no badges or emblems")
    got = prompt_words.clean_positive(line + ".")
    assert "a smooth rounded tail with a plain flat trunk lid" in got and "clean unbadged flanks" in got
    assert not re.search(r"(?i)\bno\b", got)
    assert {"spoiler", "badges", "emblems"} <= set(prompt_words.negated(line))
    car = {"id": "c", "name": "Green coupe", "kind": "vehicle", "line": line}
    p = sf.plan({"notes": "The Green coupe drives."}, _framing(_car("left", speed_ms=20)), [car], [], [car], method=sf.ZBUF)
    assert "spoiler" in p.negative and "a smooth rounded tail" in p.positive
    assert prompt_words.clean_positive("He wears no hat and no glasses.") == "He wears bare-headed and a bare face."


def test_grey_stays_grey_on_a_cast_line_and_a_stand_in_still_goes():
    assert prompt_words.clean_positive("A short grey beard and a grey flat cap.") == "A short grey beard and a grey flat cap."
    assert prompt_words.clean_positive("Every grey figure is a person.") == ""
    marcus = {"id": "m", "name": "Marcus", "kind": "person", "line": "a man in his 60s with a short grey beard and a grey flat cap"}
    p = sf.plan({"notes": "Marcus sits."}, _framing(_person("camera")), [marcus], [], [marcus], method=sf.ZBUF)
    assert "a short grey beard and a grey flat cap" in p.positive and "charcoal" not in p.positive and "silver" not in p.positive


# --- #13 keyframes by move, with the start still's recipe ----------------------------------------------------------------------

def _move(a, b, look=(0, 1, 0)):
    return {"move": {"start": {"pos": list(a), "look": list(look)}, "end": {"pos": list(b), "look": list(look)}}}


@pytest.mark.parametrize("a,b,kind", [((0, 1.5, 6), (6, 1.5, 0), "orbit"), ((0, 4, 6), (0, 1.5, 6), "crane"),
                                      ((0, 1.5, 6), (0, 1.5, 3), None), ((0, 1.5, 6), (2, 1.5, 6), None),
                                      ((0, 1.5, 6), (0, 1.5, -6), "orbit")])
def test_r3_keyframes_are_off_for_every_move_by_default(a, b, kind):
    framing = _move(a, b)
    assert previz.camera_move({}, framing) == kind
    assert previz.KEYS_AUTO is False and previz.keyframes_on({}, {}, framing) is False
    assert previz.keyframes_on({"keyframes": True}, {}, framing) is True  # an agent's explicit ask only


def test_r3_the_auto_on_gate_is_an_orbit_over_90_degrees_only(monkeypatch):
    monkeypatch.setattr(previz, "KEYS_AUTO", True)
    assert previz.orbit_degrees({}, _move((0, 1.5, 6), (6, 1.5, 0))) == pytest.approx(90.0)
    assert previz.keyframes_on({}, {}, _move((0, 1.5, 6), (6, 1.5, 0))) is False  # 90 is not over 90
    assert previz.keyframes_on({}, {}, _move((0, 1.5, 6), (-5, 1.5, -3))) is True  # about 121 degrees
    assert previz.keyframes_on({}, {}, _move((0, 4, 6), (0, 1.5, 6))) is False  # a crane never
    assert previz.keyframes_on({"keyframes": False}, {}, _move((0, 1.5, 6), (0, 1.5, -6))) is False


def _frames(tmp_path, cut_at=None, n=24):
    out = []
    for i in range(n):
        a = np.zeros((90, 160), np.uint8)
        a[:, (i * 3) % 160:(i * 3) % 160 + 20] = 200  # a bar drifting right: a steady small change
        if cut_at is not None and i >= cut_at:
            a = 255 - a  # a new picture from here on
        f = tmp_path / f"r_{i + 1:05d}.png"
        Image.fromarray(a).save(f)
        out.append(f)
    return out


def test_r3_the_cut_check_finds_a_hard_cut_and_passes_a_smooth_shot(tmp_path):
    assert previz.cut_frames(_frames(tmp_path)) == []
    assert previz.cut_frames(_frames(tmp_path, cut_at=12)) == [12]


def test_r3_a_key_plan_never_carries_frame_0s_size_or_facing():
    framing = _framing(_person("left", doing="sit", h=0.4))
    full = sf.plan({"notes": "A man sits."}, framing, [], [], [], method=sf.ZBUF)
    key = sf.plan({"notes": "A man sits."}, framing, [], [], [], method=sf.ZBUF, at_key=True)
    assert "percent of the frame height" in full.positive and "turned to frame left" in full.positive
    assert "percent of the frame" not in key.positive and "turned to frame left" not in key.positive and "sitting" in key.positive


def test_the_spec_keys_decide_when_frame_0_recorded_no_move():
    spec = {"camera": {"keys": [{"pos": [0, 1.5, 6], "look": [0, 1, 0]}, {"pos": [-6, 1.5, 0], "look": [0, 1, 0]}]}}
    assert previz.camera_move(spec, {}) == "orbit"


def test_the_start_recipe_is_read_back_by_upload_id(tmp_path):
    class S:
        def shot_dir(self, _):
            return tmp_path
    (tmp_path / "looks").mkdir()
    (tmp_path / previz.RECIPES).write_text(json.dumps({"up_1": {"user_look": "Warm dusk.", "seed": 1001, "look_refs": []}}))
    assert previz.start_recipe(S(), "s", "up_1")["seed"] == 1001
    assert previz.start_recipe(S(), "s", "up_2") == {} and previz.start_recipe(S(), "s", None) == {}


# --- REVIEW F3: a rejected workflow says so in plain words ---------------------------------------------------------------

def test_a_rejected_workflow_names_its_node_and_cant_reach_is_only_for_a_dead_server():
    from pickups.app.comfy import REJECTED, UNREACHABLE, UPLOAD_REFUSED, plain_error

    class InvalidWorkflow(Exception):
        details = {"node_errors": {"7": {"class_type": "TextEncodeQwenImage21", "errors": [{"message": "Required input is missing"}]}}}

    said, raw = plain_error(InvalidWorkflow("Prompt outputs failed validation"))
    assert said == REJECTED.format(what=": TextEncodeQwenImage21 (Required input is missing)") and "failed validation" in raw
    up = plain_error(RuntimeError("Failed to reach ComfyUI for upload."), reachable=lambda: {"system": {}})[0]
    down = plain_error(RuntimeError("Failed to reach ComfyUI for upload."), reachable=lambda: None)[0]
    assert up == UPLOAD_REFUSED and down == UNREACHABLE
    assert plain_error(ValueError("This GPU renders up to 97 frames. Shorten the shot."))[0].startswith("This GPU renders")


# --- #16 PICKUPS_KEEP_WORK ------------------------------------------------------------------------------------------------------

def test_keep_work_keeps_every_work_folder(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "frame.png").write_bytes(b"x")
    monkeypatch.setenv("PICKUPS_KEEP_WORK", "1")
    assert store_mod.drop_work(work) is False and (work / "frame.png").is_file()
    monkeypatch.setenv("PICKUPS_KEEP_WORK", "0")
    assert store_mod.keep_work() is False
    monkeypatch.delenv("PICKUPS_KEEP_WORK")
    assert store_mod.drop_work(work) is True and not work.exists()
    src = (Path(__file__).resolve().parents[1] / "src/pickups/app/jobs.py").read_text()
    assert "shutil.rmtree" not in src and src.count("drop_work(") == 2
