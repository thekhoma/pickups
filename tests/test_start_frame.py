"""Make start frame from previz (step 17b): the creative director's template v2.2 with method B3, the only method.

Fixtures (tests/fixtures/start_frame): R&D's submitted B3 graphs (c08, one reference; s06, three) and their control-map
graph, and the per-shot prompts the creative director filled by hand (v2) for s04 (one man on a bench) and c08 (a red
sedan from a chase car). v2.2 overrides v2's look, light and negative, so those check the structure and the facts.
"""

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import graphs
from pickups import start_frame as sf
from pickups.prompt_words import clean_positive as _c  # every positive is cleaned (no previz words, no negations)
from test_previz_e3 import SPEC, _jpeg, _scene, client  # noqa: F401 - the shared server fixture and a made previz

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "start_frame"
TEMPLATE = (ROOT / sf.TEMPLATE_FILE).read_text()
RULES = (ROOT / sf.RULES_FILE).read_text()


def _in_template(line: str) -> bool:
    """A config line, with each {slot} as one or more of the template's [brackets], is in the template word for word."""
    parts = re.split(r"\{\w+\}", line)
    pattern = r"(?:\[[^\]]+\] ?)+".join(re.escape(p) for p in parts)
    return re.search(pattern, TEMPLATE) is not None


def test_the_config_is_template_v2_2():
    assert sf.TEMPLATE_VERSION == "v2.2" and sf.TEMPLATE_FILE.endswith("start_frame_template_v2_2.md")
    assert "v2.2" in TEMPLATE.splitlines()[0] and sf.METHOD == "B3"
    assert (ROOT / "docs/prompts/start_frame_template_v2_1.md").is_file()  # kept for history


@pytest.mark.parametrize("name", ["FRAME_LINE", "CAMERA_LINE", "VEHICLE_LINE", "LIGHT_LINE", "LOOK_LINE", "TEST_LINE",
                                  "NEGATIVE", "VEHICLE_NEGATIVE"])
def test_every_line_is_the_templates_word_for_word(name):
    assert _in_template(getattr(sf, name)), f"{name} isn't in {sf.TEMPLATE_FILE} word for word"


def test_the_grey_line_the_light_bracket_and_the_banned_words_are_the_templates():
    assert "Every grey figure in the layout is a real person and every grey shape is a real object: [name each object" in TEMPLATE
    assert sf.GREY_LINES["people"] == "Every grey figure in the layout is a real person and every grey shape is a real object."
    assert f'6. Light: "[Source and direction]. {sf.LIGHT_LINE}"' in TEMPLATE
    assert "BANNED in the positive: " + ", ".join(sf.BANNED) + "." in TEMPLATE
    assert f"{sf.SIZE[0]}x{sf.SIZE[1]}, CFG {sf.CFG:g}, {sf.STEPS} steps" in TEMPLATE


# --- the graph, node for node --------------------------------------------------------------------------------------

VOLATILE = {"prompt", "negative_prompt", "image", "noise_seed", "filename_prefix"}


def _shape(g: dict) -> list:
    """A graph without its node ids: each node as its class, its settings (less the prompt, the inputs, the seed and the
    save name) and what feeds it, recursively. Two graphs with the same shape are the same graph."""
    memo = {}

    def sig(nid):
        if nid in memo:
            return memo[nid]
        n = g[nid]
        ins = []
        for k, v in sorted(n["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in g:
                ins.append((k, sig(v[0]), v[1]))
            elif k not in VOLATILE:
                ins.append((k, json.dumps(v)))
        memo[nid] = (n["class_type"], tuple(ins))
        return memo[nid]

    return sorted(json.dumps(sig(nid)) for nid in g)


def _ref(name: str) -> dict:
    return json.loads((FIX / name).read_text())


PLAN = sf.Plan(positive="p", negative="n")


def test_b3_is_the_reference_b3_graph_node_for_node():
    ref = _ref("c08_v3_B3_api.json")
    ours = sf.graph_for(PLAN, "d.png", "c.png", seed=1, refs=["r1.png"])
    assert len(ours) == len(ref)
    assert _shape(ours) == _shape(ref)


def test_b3_with_three_refs_is_the_reference_s06_graph_but_the_app_sends_two():
    # R&D's s06 ran three references (905 s: they spill a 24 GB card); the builder makes that graph, the plan caps at 2.
    assert _shape(sf.graph_for(PLAN, "d", "c", seed=1, refs=["a", "b", "c"])) == _shape(_ref("s06_v3_B3_api.json"))
    assert len(sf.plan(S05_SPEC, S05_FRAMING, [], [Path(f"r{i}") for i in range(3)], [None] * 3).refs) == sf.MAX_REFS == 2


def test_the_structure_maps_are_the_reference_depth_and_soft_canny():
    ref = _ref("s04_control_maps_api.json")
    keep = {"img", "da3m", "da3", "dep", "cans", "s1", "s4"}  # depth (v2_style) and the soft canny, and their saves
    ref = {k: v for k, v in ref.items() if k in keep}
    ours = graphs.start_frame_maps("f.png", da3_model=sf.DA3_MODEL, da3_resolution=sf.DA3_RESOLUTION, canny=sf.CANNY_SOFT)
    assert _shape(ours) == _shape(ref)


def test_never_the_turbo_checkpoint_or_the_lightning_lora():
    g = json.dumps(sf.graph_for(PLAN, "d", "c", seed=1, refs=["r"]))
    assert "turbo" not in g.lower() and "lightning" not in g.lower()
    assert '"cfg": 3.0' in g and f'"steps": {sf.STEPS}' in g and '"resolution": 1536' in g


def test_there_is_one_method():
    assert not hasattr(sf, "ROUTE") and not hasattr(sf, "main_kind") and not hasattr(sf, "masked_frame")
    assert not hasattr(sf.Plan(positive="", negative=""), "arm")


# --- the fill rules (docs/prompts/startframe_rules_v2.2.md), table by table ---------------------------------------------

def _quoted(text: str) -> bool:
    """Quoted in the rules, in double quotes (v2.2) or single (the v2.4 addendum)."""
    return f'"{text}"' in RULES or f"'{text}'" in RULES


def test_the_rules_file_is_kept_word_for_word_and_live_code_has_no_rich_blacks():
    assert RULES.startswith("# Start frame fill rules v2.2")
    live = [ROOT / "src/pickups/start_frame.py", ROOT / "src/pickups/graphs.py", ROOT / "src/pickups/app/server.py"]
    assert not any("rich blacks" in p.read_text() for p in live)
    assert "rich blacks" not in sf.LOOK_LINE + sf.NEGATIVE


@pytest.mark.parametrize("keys,base", sf.LOCATION_BASE)
def test_each_location_base_is_the_rules_and_its_keywords_pick_it(keys, base):
    assert _quoted(base) and all(k in RULES for k in keys)
    for k in keys:
        assert sf.location(f"Two people meet at the {k} at noon.", "", "")[1] == base


@pytest.mark.parametrize("word,phrase", sorted(sf.OBJECT_PHRASE.items()))
def test_each_object_phrase_is_the_rules(word, phrase):
    assert _quoted(phrase)
    assert sf.world_sentence("", "A street.", "", [word]).endswith(", " + phrase + ".")


def test_anything_else_is_named_weathered_and_real_and_ground_is_left_to_the_base():
    assert "'[object name], weathered and real'" in RULES  # Run 2 (a) 5
    w = sf.world_sentence("", "A man on a street.", "", ["the road", "a post", "a bench", "a bench", "people"])
    assert w == (sf._cap(sf.location("", "street", "")[1]) + ", a post, weathered and real, "
                 "a bench of weathered timber slats on metal legs.")  # v2.4 4: never "park bench" outside a park


@pytest.mark.parametrize("keys,source", sf.LIGHT_SOURCE)
def test_each_light_source_is_the_rules_and_its_time_cue_picks_it(keys, source):
    assert _quoted(source.replace("{dir}", "[dir]"))
    framing = {"light": {"dir": "camera right"}}
    got = sf.light_sentence(f"A walk in the {keys[0]}.", "", framing, interior=False)
    assert got == sf._cap(source.format(dir="camera right")) + ". " + sf.LIGHT_LINE


def test_no_cue_is_the_window_indoors_and_the_afternoon_line_outdoors():
    assert _quoted(sf.LIGHT_INTERIOR.replace("{dir}", "[dir]"))
    assert sf.light_sentence("", "", {}, interior=True) == "Soft daylight through a window at camera left. " + sf.LIGHT_LINE
    assert sf.light_sentence("", "", {}, interior=False).startswith("Late-afternoon sun from camera left, about 30 degrees high. ")
    assert f'"{sf.LIGHT_LINE}"' in RULES  # "Always follow it with"


@pytest.mark.parametrize("focal,words", [(18, sf.LENS_BANDS[0][1]), (24, sf.LENS_BANDS[0][1]), (25, sf.LENS_BANDS[1][1]),
                                         (40, sf.LENS_BANDS[1][1]), (41, sf.LENS_BANDS[2][1]), (65, sf.LENS_BANDS[2][1]),
                                         (66, sf.LENS_BANDS[3][1]), (135, sf.LENS_BANDS[3][1])])
def test_each_lens_band_is_the_rules(focal, words):
    assert _quoted(words) and sf.lens_words(focal) == words


def test_the_vehicle_line_has_no_speed_and_no_camera_car():
    assert _in_template(sf.VEHICLE_LINE) and not hasattr(sf, "VEHICLE_KMH")  # Run 2 (c)
    for facing, line in ((None, sf.VEHICLE_LINE), ("left", sf.VEHICLE_LINE), ("away", sf.VEHICLE_LINE_END_ON)):
        framing = {**C08_FRAMING, "subjects": [dict(C08_FRAMING["subjects"][0], facing=facing)]}
        p = sf.plan(C08_SPEC, framing, [], [], [], user_look="The car does about 80 km/h. Dust behind it.")
        assert line + "." in p.positive and "Dust behind it." in p.positive
        assert not re.search(r"(?i)km/h|\bmph\b|camera car|alongside|\b80\b", p.positive)


def test_the_group_and_fallback_lines_are_the_rules():
    assert _quoted(sf.GROUP_NO_REF.rstrip(".")) and _quoted(sf.FALLBACK_LINE)


# --- precedence: the user's words, then the previz scene data, then the defaults ----------------------------------------

def test_precedence_user_then_previz_then_default():
    framing = {"place": "Park", "world": {"time": "night"}, "light": {"dir": "backlit"}}
    # Location: the user's words beat the sentence, the sentence beats the previz place, the place beats the default.
    assert sf.location("on a rooftop", "in a park", "Highway")[0] == "rooftop"
    assert sf.location("", "in a park", "Highway")[0] == "park"
    assert sf.location("", "", "Highway")[0] == "highway"
    assert sf.location("", "", "")[0] == "street"
    # Light: the user's time cue beats the previz time; the previz key light gives the direction; else camera left.
    assert sf.light_sentence("at dawn", "", framing, False).startswith("Low soft dawn sun from backlit")
    assert sf.light_sentence("", "", framing, False).startswith("Street lamps and lit windows as the light sources, from backlit")
    assert sf.light_sentence("", "", {"light": {"dir": "sideways"}}, False).startswith("Late-afternoon sun from camera left")


def test_weather_words_never_reach_the_start_frame():
    framing = {**S04_FRAMING, "world": {"time": "night", "weather": "rain"}}
    p = sf.plan(S04_SPEC | {"notes": "A man sits on a bench in the rain and fog."}, framing, [], [], [],
                user_look="Heavy rain, thick fog and snow at night.")
    for w in sf.WEATHER:
        assert not re.search(rf"(?i)(?<![\w-]){w}(?![\w-])", p.positive), w


@pytest.mark.parametrize("spec,framing", [("S04", "S04"), ("C08", "C08")])
def test_the_light_line_closes_paragraph_6_on_every_shot_vehicles_too(spec, framing):
    paras = _paras(sf.plan(globals()[f"{spec}_SPEC"], globals()[f"{framing}_FRAMING"], [], [], [], matte=Path("m")).positive)
    assert paras[-2].endswith(". " + _c(sf.LIGHT_LINE)) and paras[-1] == _c(sf.LOOK_LINE) and sf.TEST_LINE not in paras


# --- groups: counted from the previz figures, numbered left to right ---------------------------------------------------

S05_SPEC = {"notes": "Two friends hugging on a city sidewalk. Close to wide.", "lens": {"focal_mm": 50}}
S05_FRAMING = {  # as the previz recorded it (step 17b): one plural parse ("friends"), two figures
    "view": {"share": {"buildings": 0.2, "cars": 0.1, "the pavement": 0.3}, "outside": [], "actors": [0, 1]},
    "camera": {"height_m": 1.7, "pitch_deg": -9, "focal_mm": 50},
    "subjects": [{"kind": "person", "x": 0.56, "distance_m": 1.6, "in_view": True, "doing": "hug", "word": "friends"},
                 {"kind": "person", "x": 0.42, "distance_m": 1.5, "in_view": True}],
}


def test_s05_two_friends_are_two_people_numbered_left_to_right():
    p = sf.plan(S05_SPEC, S05_FRAMING, [], [], [], matte=Path("m"))
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 2 people in frame: person 1 on the left, person 2 on the right.")
    assert "friends" not in paras[0]
    # v2.3 (a): the hug is one shared line naming both; no face bracket, as the previz doesn't say who faces camera.
    assert paras[1].startswith("Person 1: left of frame, about 1.5 metres from the viewer. Person 2: right of frame, about 1.6 "
                               "metres from the viewer. Person 1 and person 2 hug each other, every hand resting on the other's "
                               "back and traceable to a shoulder.")
    assert paras[1].count("hug") == 1 and "hugging" not in paras[1]
    assert paras[3] == sf.GROUP_NO_REF + " Person 1 and person 2 are two different people."  # v2.4 7


def test_the_shared_line_and_the_default_action_are_the_rules():
    assert _quoted("Person 1 and person 2 hug each other, [her face / his face] toward camera, every hand resting on the "
                   "other's back and traceable to a shoulder.")
    assert _quoted(sf.DEFAULT_ACTION + ".") and _quoted(sf.MANY_VEHICLES)
    for k in ("hug", "embrace", "handshake", "kiss", "dance together", "hold hands", "talk to each other"):
        assert k in sf.SHARED_ACTIONS and k in RULES


def test_the_face_bracket_only_when_the_previz_says_who_faces_camera():
    faced = {**S05_FRAMING, "subjects": [dict(S05_FRAMING["subjects"][0], faces_camera="her face"), S05_FRAMING["subjects"][1]]}
    para = _paras(sf.plan(S05_SPEC, faced, [], [], [], matte=Path("m")).positive)[1]
    assert "Person 1 and person 2 hug each other, her face toward the viewer, every hand resting on the other's back" in para
    assert "toward the viewer" not in _paras(sf.plan(S05_SPEC, S05_FRAMING, [], [], [], matte=Path("m")).positive)[1]


@pytest.mark.parametrize("subjects,notes", [
    (S05_FRAMING["subjects"], "Two friends hugging on a city sidewalk."),
    ([{"kind": "person", "x": 0.3, "in_view": True}, {"kind": "person", "x": 0.7, "in_view": True, "doing": "walk"}], "Two people."),
    ([{"kind": "person", "x": 0.5, "in_view": True}], "A woman."),
    ([{"kind": "person", "x": x, "in_view": True} for x in (0.1, 0.3, 0.5, 0.7, 0.9)], "A crowd."),
    ([{"kind": "person", "x": 0.2, "in_view": True}, {"kind": "vehicle", "x": 0.7, "in_view": True}], "A man by a car."),
])
def test_no_person_is_ever_without_an_action_line(subjects, notes):
    p = sf.plan({"notes": notes}, {**S05_FRAMING, "subjects": subjects}, [], [], [], matte=Path("m"))
    para = _paras(p.positive)[1]
    people = [s for s in p.subjects if s.kind == "person"]
    shared = sf.shared_action(people, notes) or ""
    assert shared in para
    for s in people:
        own = next(b for b in para.split(". ") if b.lower().startswith(f"{s.label.lower()}:"))
        acted = own.rstrip(".").endswith((sf.DEFAULT_ACTION, *sf.DOING.values()))
        assert acted or s.label.lower() in shared.lower(), (s.label, para)


def test_four_people_read_far_left_to_far_right_and_five_are_numbered():
    four = {**S05_FRAMING, "subjects": [{"kind": "person", "x": x, "in_view": True} for x in (0.9, 0.1, 0.4, 0.6)]}
    assert _paras(sf.plan({"notes": "Four people."}, four, [], [], [], matte=Path("m")).positive)[0].endswith(
        "EXACTLY 4 people in frame: person 1 on the far left, person 2 left of centre, person 3 right of centre, "
        "person 4 on the far right.")
    assert _quoted("far left, left of centre, right of centre, far right")
    five = {**S05_FRAMING, "subjects": [{"kind": "person", "x": x, "in_view": True} for x in (0.1, 0.3, 0.5, 0.7, 0.9)]}
    assert _paras(sf.plan({"notes": "A crowd."}, five, [], [], [], matte=Path("m")).positive)[0].endswith(
        "EXACTLY 5 people in frame: person 1, person 2, person 3, person 4, person 5.")


def test_a_group_reads_left_to_right_with_ref_names_where_there_are_refs():
    framing = {**S05_FRAMING, "subjects": [{"kind": "person", "x": 0.8, "distance_m": 4, "in_view": True},
                                            {"kind": "person", "x": 0.1, "distance_m": 5, "in_view": True},
                                            {"kind": "person", "x": 0.5, "distance_m": 6, "in_view": True}]}
    ana = {"id": "c1", "name": "Ana", "kind": "person", "line": "curly black hair, a yellow raincoat"}
    p = sf.plan({"notes": "Ana and her friends dance."}, framing, [ana], [Path("ana.jpg")], [ana], matte=Path("m"))
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 3 people in frame: Ana on the left, person 2 in the centre, person 3 on the right.")
    assert [b.split(":")[0] for b in paras[1].split(". ")[:3]] == ["Ana", "Person 2", "Person 3"]
    assert paras[3] == "Ana is the one in image 3: curly black hair, a yellow raincoat. " + sf.GROUP_NO_REF


def test_vehicles_are_numbered_the_same_way():
    framing = {**C08_FRAMING, "subjects": [{"kind": "vehicle", "x": 0.7, "distance_m": 9, "in_view": True},
                                            {"kind": "vehicle", "x": 0.2, "distance_m": 12, "in_view": True, "word": "van"}]}
    paras = _paras(sf.plan(C08_SPEC, framing, [], [], []).positive)
    assert paras[0].endswith("EXACTLY 2 vehicles in frame: vehicle 1 on the left, vehicle 2 on the right.")
    # v2.3 (c): each vehicle with no ref gets the identity line, typed from its previz word; two or more differ.
    assert paras[3] == ("Vehicle 1 is a real, everyday van in a plain solid colour, with glass reflections, tyres with real "
                        "tread and clean unbadged bodywork. Vehicle 2 is a real, everyday saloon in a plain solid colour, with "
                        "glass reflections, tyres with real tread and clean unbadged bodywork. Each vehicle a different make and colour.")


def test_the_no_ref_vehicle_line_is_the_rules_and_its_type_comes_from_the_previz():
    assert _quoted(sf.VEHICLE_NO_REF.format(label="Vehicle 1", type="[saloon / hatchback / SUV / pickup / van, picked from "
                                            "the previz vehicle type, default saloon]"))
    assert [sf.vehicle_type(w) for w in ("", "car", "sedan", "suv", "truck", "hatchback", "van", "boat")] == [
        "saloon", "saloon", "saloon", "SUV", "pickup", "hatchback", "van", "saloon"]
    one = _paras(sf.plan(C08_SPEC, C08_FRAMING, [], [], []).positive)[3]
    assert one == _c(sf.VEHICLE_NO_REF.format(label="Vehicle 1", type="saloon")) and sf.MANY_VEHICLES not in one


def test_camera_left_stays_the_default_light():
    no_light = {k: v for k, v in S04_FRAMING.items() if k != "light"}
    assert "from frame left" in _paras(sf.plan(S04_SPEC, no_light, [], [], [], matte=Path("m")).positive)[-2]  # Run 2 (d)
    assert sf.LIGHT_DIRECTION_DEFAULT == "camera left"


# --- the fills, against the creative director's own pages --------------------------------------------------------------

S04_SPEC = {"notes": "Marcus sits alone on a park bench.", "lens": {"focal_mm": 60}, "subjects": [{"kind": "humanoid"}]}
S04_FRAMING = {
    "view": {"share": {"trees": 0.2, "a path": 0.15, "a bench": 0.05, "grass": 0.3}, "outside": [], "actors": [0, 1]},  # no car (Run 3)
    "camera": {"height_m": 1.6, "pitch_deg": -2, "focal_mm": 60},
    "subjects": [{"kind": "person", "x": 0.45, "distance_m": 9.2, "in_view": True, "doing": "sit"}],
}
MARCUS = {"id": "cast_m", "name": "Marcus", "kind": "person", "line": "a man in his 60s with a short gray beard, navy wool peacoat, flat grey tweed cap"}
C08_SPEC = {"notes": "A red sedan speeds away down the highway. The camera follows low behind it.", "lens": {"focal_mm": 35},
            "subjects": [{"kind": "car"}]}
C08_FRAMING = {
    "view": {"share": {"the road": 0.5, "a fence": 0.1, "a post": 0.02}, "outside": [], "actors": [0]},
    "camera": {"height_m": 0.5, "pitch_deg": 0, "focal_mm": 35},
    "subjects": [{"kind": "vehicle", "x": 0.5, "distance_m": 8.0, "in_view": True, "doing": "drive", "speed_ms": 25.0}],
    "light": {"dir": "camera left"},
}
SEDAN = {"id": "cast_s", "name": "Red sedan", "kind": "prop", "line": "glossy red paint, black wheels, slim full-width tail lights"}


def _paras(text: str) -> list[str]:
    return text.split("\n\n")


def test_s04_a_man_on_a_bench_fills_like_the_creative_directors_page():
    p = sf.plan(S04_SPEC, S04_FRAMING, [MARCUS], [Path("marcus.jpg")], [MARCUS], matte=Path("m.png"))
    hand = (FIX / "s04" / "prompt_v2.txt").read_text()
    paras = _paras(p.positive)
    assert paras[0] == ("A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp "
                        "and clean. EXACTLY 1 person in frame: Marcus in the centre.")
    assert "EXACTLY ONE PERSON" in hand
    assert "Marcus: centre of frame, about 9 metres from the viewer, sitting." in paras[1] and "about 9 metres" in hand
    assert sf.DEFAULT_ACTION not in paras[1]  # his own action
    assert paras[2] == ("Viewpoint at about 1.6 metres above the ground, level, 60mm lens, natural perspective with gentle "
                        "compression, background softly out of focus.")
    assert paras[3] == _c(f"Marcus is the one in image 3: {MARCUS['line']}.") and "is the one in the reference image" in hand
    # v2.3 (b): the tree phrase stays, the park base's own trees go. v2.4 4: the tree phrase. Run 3: s04 has no car.
    assert paras[4] == ("Real grass with worn patches and fallen leaves, packed gravel and dirt paths, real trees with full, "
                        "irregular canopies, rough bark and uneven branches, a park bench of weathered timber slats on "
                        "cast-iron legs.")
    assert paras[5] == "Late-afternoon sun from frame left, about 30 degrees high. " + _c(sf.LIGHT_LINE)
    assert paras[-1] == _c(sf.LOOK_LINE) and sf.TEST_LINE not in p.positive
    neg = (FIX / "s04" / "negative.txt").read_text().strip()
    assert p.negative == sf.NEGATIVE + ", " + sf.CAMERA_NEGATIVE + ", " + sf.REF_NEGATIVE and set(neg.split(", ")) <= set(p.negative.split(", "))


def test_c08_a_sedan_from_a_chase_car_has_the_vehicle_words():
    p = sf.plan(C08_SPEC, C08_FRAMING, [SEDAN], [Path("sedan.png")], [SEDAN], matte=Path("m.png"))
    hand = (FIX / "c08" / "prompt_v2.txt").read_text()
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 1 vehicle in frame: Red sedan in the centre.") and "EXACTLY ONE CAR" in hand
    assert "Red sedan: centre of frame, about 8 metres from the viewer, driving." in paras[1] and "about 8 metres ahead" in hand
    assert paras[2] == ("Viewpoint at about 50 cm above the ground, level, 35mm lens, natural perspective, background slightly soft, "
                        "seen side-on with the road and background streaking with motion blur, wheels blurred.")
    assert paras[3] == f"Red sedan is the one in image 3: {SEDAN['line']}."  # B3: depth, canny, then the references
    assert paras[4].startswith("Worn asphalt with tyre marks and cracks")
    neg = (FIX / "c08" / "negative.txt").read_text().strip()
    assert p.negative == sf.NEGATIVE + ", " + sf.CAMERA_NEGATIVE + ", " + sf.VEHICLE_NEGATIVE + ", " + sf.REF_NEGATIVE
    assert set(neg.split(", ")) <= set(p.negative.split(", "))


def test_paragraph_7_is_exactly_the_locked_look_and_nothing_is_spliced():
    for spec, framing in ((S04_SPEC, S04_FRAMING), (C08_SPEC, C08_FRAMING), (S05_SPEC, S05_FRAMING)):
        paras = _paras(sf.plan(spec, framing, [], [], [], matte=Path("m.png")).positive)
        assert paras[-1] == _c(sf.LOOK_LINE) and paras[-2].endswith(_c(sf.LIGHT_LINE))
        assert "Exposed" not in paras[-1] and ".," not in "\n\n".join(paras) and ".." not in "\n\n".join(paras)


def test_people_and_vehicles_together_are_counted_both_and_the_vehicle_words_apply():
    framing = {**S04_FRAMING, "subjects": [S04_FRAMING["subjects"][0],
                                            {"kind": "vehicle", "x": 0.8, "distance_m": 6, "in_view": True},
                                            {"kind": "vehicle", "x": 0.2, "distance_m": 40, "in_view": False}]}
    p = sf.plan(S04_SPEC, framing, [], [], [], matte=Path("m.png"))
    assert _paras(p.positive)[0].endswith("EXACTLY 1 person and 1 vehicle in frame: person 1 in the centre, vehicle 1 on the right.")
    # R2-4: no vehicle in it moves, so "parked car" is no negative.
    assert p.negative.endswith(sf.VEHICLE_NEGATIVE.replace("parked car, ", "")) and "parked car" not in p.negative
    moving = {**framing, "subjects": [framing["subjects"][0], dict(framing["subjects"][1], speed_ms=12)]}
    assert sf.plan(S04_SPEC, moving, [], [], [], matte=Path("m.png")).negative.endswith(sf.VEHICLE_NEGATIVE)


def test_banned_words_never_reach_the_positive():
    dirty = dict(MARCUS, line="a cinematic hyperreal 8K man in a peacoat")
    p = sf.plan(S04_SPEC, S04_FRAMING, [dirty], [Path("m.jpg")], [dirty], user_look="Cinematic pore-level warm light. A blockout proxy.")
    for w in sf.BANNED:
        assert not re.search(rf"(?i)(?<![\w-]){re.escape(w)}(?![\w-])", p.positive), w
    assert "in a peacoat" in p.positive and "Warm light." in p.positive and "blockout" not in p.positive


def test_the_only_todo_left_is_an_old_shots_subject_mask():
    assert sf.plan(S04_SPEC, S04_FRAMING, [], [], [], matte=Path("m.png")).todo == []
    assert any("subject mask" in t for t in sf.plan(S04_SPEC, S04_FRAMING, [], [], []).todo)


# --- the commercial-safe fallback (rule 5) -------------------------------------------------------------------------------

def test_the_fallback_is_the_same_words_the_a2_way_with_the_rules_sentence():
    b3 = sf.plan(S04_SPEC, S04_FRAMING, [MARCUS], [Path("m.jpg")], [MARCUS], matte=Path("m.png"))
    fb = sf.plan(S04_SPEC, S04_FRAMING, [MARCUS], [Path("m.jpg")], [MARCUS], method="commercial-safe")
    assert fb.method == "commercial-safe" and b3.method == "B3" and fb.negative == b3.negative
    assert _paras(fb.positive)[1].endswith(sf.FALLBACK_LINE_CLEAN) and sf.FALLBACK_LINE_CLEAN not in b3.positive
    assert "Marcus is the one in image 2:" in fb.positive  # frame 0 is image 1
    g = sf.fallback_graph(fb, "f.png", seed=3, refs=["m.jpg"])
    sampler = next(n for n in g.values() if n["class_type"] == "KSampler")["inputs"]
    assert sampler["steps"] == 30 and sampler["cfg"] == 3.0
    assert "lightning" not in json.dumps(g).lower()
    assert g["PU_Frame"]["inputs"]["image"] == "f.png" and g["PU_Frame2"]["inputs"]["image"] == "m.jpg"
    assert g["PU_Instruction"]["inputs"]["prompt"] == fb.positive and g["qneg"]["inputs"]["prompt"] == fb.negative
    # v2.3 (e): generated at 1664x928, then a Lanczos upscale to 2048x1152 before the save.
    assert sf.FALLBACK_SIZE == (1664, 928) and "generate at 1664x928, then Lanczos upscale to 2048x1152" in RULES
    assert "1664" in json.dumps(g) and "928" in json.dumps(g)
    up = g["upscale"]["inputs"]
    assert (g["upscale"]["class_type"], up["upscale_method"], up["width"], up["height"]) == ("ImageScale", "lanczos", 2048, 1152)
    assert g["PU_Plate"]["inputs"]["images"] == ["upscale", 0]


def test_the_fallback_paints_over_the_joint_circles(tmp_path):
    frame, _ = _street()
    a = np.asarray(sf.painted_frame(frame))
    assert tuple(a[200, 290]) == (225, 220, 60) and tuple(a[180, 90]) == GREY
    frame.save(tmp_path / "frame0.png")
    r = _Runner()
    fb = sf.plan(S04_SPEC, S04_FRAMING, [], [], [], method="commercial-safe")
    sf.make_fallback(r, fb, tmp_path / "frame0.png", tmp_path / "out.png", seed=1)
    assert len(r.graphs) == 1 and r.graphs[0]["PU_Frame"]["inputs"]["image"] == "start_frame0_painted.png"


# --- the canny mask: the subjects from the renderer's matte, and the joint circles --------------------------------------

GREY = (128, 128, 128)


def _street() -> tuple[Image.Image, Image.Image]:
    """A grey car on a grey road (the same grey: colour can't find it) and a yellow mannequin with joint balls, and the
    renderer's subject matte for both."""
    frame = Image.new("RGB", (400, 300), GREY)
    d = ImageDraw.Draw(frame)
    d.rectangle([30, 150, 150, 210], fill=GREY, outline=(120, 120, 120))  # the car: road grey
    d.rectangle([250, 60, 330, 260], fill=(225, 220, 60))  # the mannequin
    for x, y in ((250, 120), (330, 120), (290, 200)):
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(20, 20, 20))  # joint balls
    matte = Image.new("L", (400, 300), 0)
    md = ImageDraw.Draw(matte)
    md.rectangle([30, 150, 150, 210], fill=255)
    md.rectangle([250, 60, 330, 260], fill=255)
    return frame, matte


def test_the_subject_mask_comes_from_the_matte_grown_12_px_square():
    frame, matte = _street()
    m = sf.subject_mask(matte, frame.size)
    assert m[180, 90] and m[180, 30 - 12] and m[150 - 12, 30 - 12]  # inside, and 12 px out at an edge and a corner
    assert not m[180, 30 - 13] and not m[150 - 13, 90]
    assert not m[20, 200]


def test_the_canny_keeps_depth_only_on_the_subjects_and_loses_the_joints():
    frame, matte = _street()
    canny = Image.new("RGB", (400, 300), (255, 255, 255))
    c = np.asarray(sf.masked_canny(canny, frame, matte))
    assert (c[180, 90] == 0).all()  # the grey car: masked by its ID, not its colour
    assert (c[150, 290] == 0).all() and (c[200, 290] == 0).all()
    assert (c[20, 200] == 255).all() and (c[280, 10] == 255).all()  # the rest of the frame keeps its canny
    old = np.asarray(sf.masked_canny(canny, frame, None))  # an older shot without a matte: the joints only
    assert (old[180, 90] == 255).all() and (old[200, 290] == 0).all()


# --- running it, on a stand-in ComfyUI ---------------------------------------------------------------------------------

class _Out:
    def __init__(self, size):
        self.size = size

    def to_file(self, path):
        Image.new("RGB", self.size, (255, 255, 255)).save(path)


class _Job:
    def get_outputs(self, name):
        return [_Out((400, 300) if name in ("PU_Depth", "PU_Canny") else sf.SIZE)]


class _Runner:
    def __init__(self):
        self.graphs = []
        self.client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: Path(p).name)})()})()

    def run(self, g):
        self.graphs.append(g)
        return _Job()


def test_b3_runs_the_maps_masks_the_canny_then_generates(tmp_path):
    frame, matte = _street()
    frame.save(tmp_path / "frame0.png")
    matte.save(tmp_path / "subjects.png")
    r = _Runner()
    p = sf.plan(C08_SPEC, C08_FRAMING, [SEDAN], [tmp_path / "s.png"], [SEDAN], matte=tmp_path / "subjects.png")
    sf.make(r, p, tmp_path / "frame0.png", tmp_path / "out.png", seed=7)
    assert len(r.graphs) == 2
    assert _shape(r.graphs[0]) == _shape(graphs.start_frame_maps("x", canny=sf.CANNY_SOFT))
    gen = r.graphs[1]
    assert gen["PU_Canny"]["inputs"]["image"] == "start_canny_masked.png" and gen["PU_Prompt"]["inputs"]["prompt"] == p.positive
    masked = np.asarray(Image.open(tmp_path / "start_canny_masked.png"))
    assert (masked[180, 90] == 0).all() and (masked[20, 200] == 255).all()
    assert (tmp_path / "out.png").exists()


# --- the endpoint: plan_only shows the words without running anything --------------------------------------------------

def _matte_url(size=(1920, 1088)) -> str:
    im = Image.new("L", size, 0)
    ImageDraw.Draw(im).rectangle([800, 300, 1100, 900], fill=255)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("with_matte", [True, False])
def test_the_endpoint_plans_from_the_made_previz_without_running(client, with_matte):  # noqa: F811 - the fixture
    c, store, _ = client
    jpeg0 = _jpeg(_scene())
    sid = "pv_aa00bb11cc22dd3" + ("3" if with_matte else "4")
    for start in range(0, 97, 16):
        files = [("files", (f"{i:04d}.jpg", jpeg0, "image/jpeg")) for i in range(start, min(97, start + 16))]
        assert c.post(f"/api/previz/{sid}/frames", data={"start": str(start)}, files=files).status_code == 200
    body = {"spec": SPEC | {"notes": "A woman waves."}, "name": "Plan", "framing": S04_FRAMING}
    if with_matte:
        body["matte"] = _matte_url()
    r = c.post(f"/api/previz/{sid}/finish", json=body)
    assert r.status_code == 200, r.text
    shot = r.json()
    assert ("previz_matte_url" in c.get(f"/api/shots/{shot['id']}").json()) is with_matte
    p = c.post("/api/previz/start_frame", json={"shot_id": shot["id"], "plan_only": True, "user_look": "A cinematic dusk."}).json()
    assert p["method"] == "B3" and "arm" not in p and p["steps"] == 30 and p["cfg"] == 3.0 and p["template"] == "v2.2"
    assert p["subject_mask"] is with_matte
    assert "EXACTLY 1 person in frame" in p["positive"] and "cinematic" not in p["positive"].lower()
    assert p["negative"] == sf.NEGATIVE + ", " + sf.CAMERA_NEGATIVE


def test_a_matte_of_the_wrong_size_or_kind_is_not_kept(client):  # noqa: F811 - the fixture
    from pickups.app import previz_maker
    assert previz_maker._matte(_matte_url((640, 360))) is None
    assert previz_maker._matte("data:image/jpeg;base64,AAAA") is None and previz_maker._matte(None) is None
    kept = Image.open(io.BytesIO(previz_maker._matte(_matte_url())))
    assert kept.mode == "L" and kept.size == (1920, 1088)


# --- step 18b: a thin scene (an older shot) with a cast reference ---------------------------------------------------------

C08_THIN = {"subject": [0.4, 0.31, 0.61, 0.66], "horizon": 0.5}  # what c08's old frame 0 kept: a box and a horizon


def test_a_thin_c08_with_the_red_sedan_ref_names_the_car_and_has_the_vehicle_and_backdrop_negatives():
    ref, chase = SEDAN, {"notes": "A car chase through traffic on the highway. Low angle."}
    # v2.3 (f): a prop is a vehicle only on a vehicle figure. c08's old frame 0 has none, so it is refused (the page
    # re-renders frame 0 first, which brings the figure); with the figure, the Red sedan is the car.
    with pytest.raises(sf.NoSubjects):
        sf.plan(chase, C08_THIN, [ref], [Path("sedan.png")], [ref])
    thin = {**C08_THIN, "subjects": [{"kind": "vehicle", "x": 0.5, "in_view": True, "doing": "drive"}]}
    p = sf.plan(chase, thin, [ref], [Path("sedan.png")], [ref])
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 1 vehicle in frame: Red sedan in the centre.") and "ZERO" not in p.positive
    assert paras[1] == ("Red sedan: centre of frame, driving. Every vehicle in the layout is a real car on a real road, "
                        "and every other shape is a real object.")  # the rules' clause, its grey taken out
    assert paras[3] == f"Red sedan is the one in image 3: {SEDAN['line']}."
    assert paras[4].startswith("Worn asphalt with tyre marks and cracks")  # the sentence's "highway"
    assert paras[2] == sf._cap(sf.VEHICLE_LINE) + "."  # no camera recorded; Run 2: no speed, no camera car
    for words in (sf.VEHICLE_NEGATIVE, sf.REF_NEGATIVE):
        assert words in p.negative
    assert "studio background" in sf.REF_NEGATIVE and "seamless backdrop" in sf.REF_NEGATIVE
    assert p.action == sf.REMAKE  # no camera: the page re-renders frame 0, else Remake this shot


def test_a_person_ref_on_a_thin_scene_is_a_person_and_props_or_places_are_not_subjects():
    place = {"id": "c3", "name": "Old town square", "kind": "place", "line": "cobbles"}
    p = sf.plan({"notes": "Marcus waits."}, C08_THIN, [MARCUS, place], [Path("m.jpg"), Path("p.jpg")], [MARCUS, place])
    assert _paras(p.positive)[0].endswith("EXACTLY 1 person in frame: Marcus in the centre.")
    assert sf.VEHICLE_NEGATIVE not in p.negative and sf.REF_NEGATIVE in p.negative
    assert sf.ref_kind(place) is None and sf.ref_kind({"name": "Phone", "kind": "prop", "line": "black glass"}) is None


def test_a_refs_saved_type_beats_its_name():
    # v2.3 (e): no car-word guess. Saved as a person, the Red sedan is a person.
    assert sf.ref_kind(dict(SEDAN, kind="person")) == "person" and sf.ref_kind(SEDAN) == "prop vehicle"
    assert sf.ref_kind({"name": "Van Morrison", "kind": "person", "line": "a singer"}) == "person"


KEYS = {"id": "c_k", "name": "car keys", "kind": "prop", "line": "a silver fob"}
TOY = {"id": "c_t", "name": "toy truck", "kind": "prop", "line": "red plastic"}
ON_A_CAR = {**C08_FRAMING, "subjects": [dict(C08_FRAMING["subjects"][0], doing=None, speed_ms=None)]}


def test_the_prop_rule_is_the_rules():
    assert _quoted("A Prop ref becomes a vehicle subject only when its name has a vehicle word AND it is placed on a "
                   "vehicle-shaped previz object. Otherwise it stays a prop and is written only as an object phrase in paragraph 5.")


@pytest.mark.parametrize("prop", [KEYS, TOY])
def test_car_keys_and_a_toy_truck_stay_props_written_only_in_paragraph_5(prop):
    assert sf.ref_kind(prop) is None
    for framing in (ON_A_CAR, S04_FRAMING):  # on a vehicle figure or not
        p = sf.plan({"notes": "Marcus waits by the road."}, framing, [prop], [Path("p.png")], [prop], matte=Path("m"))
        world = _paras(p.positive)[-3]  # paragraph 5: then light and look
        assert p.positive.count(prop["name"]) == 1 and prop["line"] not in p.positive
        assert world.endswith(f", {prop['name']} as in image 3.")
        assert all(s.label != prop["name"] for s in p.subjects)


def test_the_red_sedan_on_a_vehicle_figure_is_the_vehicle():
    p = sf.plan({"notes": "A red sedan waits."}, ON_A_CAR, [SEDAN], [Path("s.png")], [SEDAN], matte=Path("m"))
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 1 vehicle in frame: Red sedan in the centre.")
    assert paras[3] == f"Red sedan is the one in image 3: {SEDAN['line']}." and "as in image" not in paras[4]
    # Without a vehicle figure (a man on a bench) the Red sedan stays a prop.
    p = sf.plan(S04_SPEC, S04_FRAMING, [SEDAN], [Path("s.png")], [SEDAN], matte=Path("m"))
    assert _paras(p.positive)[0].endswith("EXACTLY 1 person in frame: person 1 in the centre.")
    assert _paras(p.positive)[-3].endswith(", Red sedan as in image 3.")


# --- v2.3 (g): the paragraph 2 grey clause by what's in the shot ---------------------------------------------------------

def _grey(subjects, refs=()):
    """Paragraph 2 with the grey put back where the rules' clause has it (the positive never says grey)."""
    framing = {**S04_FRAMING, "subjects": subjects}
    p2 = _paras(sf.plan({"notes": "On the road."}, framing, list(refs), [Path("r.png")] * len(refs), list(refs),
                        matte=Path("m")).positive)[1]
    assert "grey" not in p2.lower()
    return re.sub(r"\b(figure|vehicle|shape)\b(?= (in the layout )?is a real)", r"grey \1", p2)


PERSON = {"kind": "person", "x": 0.3, "in_view": True}
CAR = {"kind": "vehicle", "x": 0.7, "in_view": True}
OBJECTS = ": grass, trees, a path, a bench."  # S04_FRAMING, no car (Run 3)


def test_each_grey_clause_is_the_rules():
    for key, line in sf.GREY_LINES.items():
        assert _quoted(line.format(vehicle="car")), key


def test_people_and_objects_get_the_people_clause():
    assert _grey([PERSON]).endswith(sf.GREY_LINES["people"][:-1] + OBJECTS)


def test_vehicles_and_objects_only_get_the_vehicle_clause():
    assert _grey([CAR]).endswith("Every grey vehicle in the layout is a real car on a real road, and every other grey "
                                 "shape is a real object" + OBJECTS)


def test_people_and_vehicles_get_both():
    assert _grey([PERSON, CAR]).endswith("Every grey figure in the layout is a real person, every grey vehicle is a real "
                                         "car, and every other grey shape is a real object" + OBJECTS)


def test_objects_only_get_the_object_clause():
    assert sf.grey_line([], ["a bench"]) == "Every grey shape in the layout is a real object: a bench."
    assert sf.grey_line([], []) == sf.GREY_LINES["objects"]


@pytest.mark.parametrize("word", ["van", "truck", "bus"])
def test_a_known_vehicle_type_replaces_car(word):
    assert f"is a real {word} on a real road" in _grey([dict(CAR, word=word)])
    assert f"every grey vehicle is a real {word}," in _grey([PERSON, dict(CAR, word=word)])
    blue = {"id": "c_v", "name": f"Blue {word}", "kind": "prop", "line": "dented"}
    assert f"is a real {word} on a real road" in _grey([CAR], [blue])  # from the ref's name


def test_mixed_or_unknown_vehicle_types_say_car():
    assert "is a real car on a real road" in _grey([dict(CAR, word="van"), dict(CAR, x=0.2, word="bus")])
    assert "is a real car on a real road" in _grey([dict(CAR, word="sedan")])


def test_never_exactly_zero_nothing_to_make_is_refused():
    with pytest.raises(sf.NoSubjects) as e:
        sf.plan({"notes": "An empty highway."}, C08_THIN, [], [], [])
    assert str(e.value) == sf.NO_SUBJECTS and "Make shot again" in sf.NO_SUBJECTS
    with pytest.raises(sf.NoSubjects):
        sf.plan({}, {}, [], [Path("u.png")], [None])  # an uploaded picture alone names no subject


def test_no_refs_means_no_backdrop_words_and_a_full_scene_needs_no_remake():
    p = sf.plan(S04_SPEC, S04_FRAMING, [], [], [], matte=Path("m.png"))
    assert sf.REF_NEGATIVE not in p.negative and p.action is None


def test_the_endpoint_refuses_an_empty_frame_with_the_guidance(client):  # noqa: F811 - the fixture
    c, store, _ = client
    jpeg0 = _jpeg(_scene())
    sid = "pv_aa00bb11cc22dd35"
    for start in range(0, 97, 16):
        files = [("files", (f"{i:04d}.jpg", jpeg0, "image/jpeg")) for i in range(start, min(97, start + 16))]
        assert c.post(f"/api/previz/{sid}/frames", data={"start": str(start)}, files=files).status_code == 200
    shot = c.post(f"/api/previz/{sid}/finish", json={"spec": SPEC | {"notes": "An empty road."}, "name": "Empty",
                                                      "framing": C08_THIN}).json()
    r = c.post("/api/previz/start_frame", json={"shot_id": shot["id"], "plan_only": True})
    assert r.status_code == 422 and r.json()["detail"] == sf.NO_SUBJECTS
    # The page re-renders the old shot's frame 0 from its spec: the framing and the matte arrive, and the plan is full.
    r = c.post(f"/api/shots/{shot['id']}/previz_frame0", json={"framing": S04_FRAMING, "matte": _matte_url()})
    assert r.status_code == 200 and r.json()["previz_framing"]["subjects"] and r.json()["previz_matte_url"]
    assert r.json()["previz_framing"]["horizon"] == C08_THIN["horizon"]  # what the old frame 0 had stays
    p = c.post("/api/previz/start_frame", json={"shot_id": shot["id"], "plan_only": True}).json()
    assert p["subject_mask"] is True and p["action"] is None and "EXACTLY 1 person in frame" in p["positive"]


# --- step 20: R&D's eb0cb5f shots (tests/fixtures/step20, copied from that run) -------------------------------------------

def _shot20(shot: str) -> dict:
    return json.loads((FIX.parent / "step20" / f"{shot}.json").read_text())


def _plan20(shot: str) -> "sf.Plan":
    d = _shot20(shot)
    refs = [Path(f"{c['name']}.png") for c in d["cast"]]
    return sf.plan(d["spec"], d["framing"], d["cast"], refs, d["cast"], d["user_look"], matte=Path("m.png"))


def _world(p: "sf.Plan") -> str:
    return _paras(p.positive)[-3]  # paragraph 5: then light and look


# v2.4 3 and 4: location bases and object phrases by location
def test_3_s03_by_the_lake_comes_out_lakeside():
    assert "Cracked asphalt" in _shot20("s03")["before"]["positive"]
    assert _world(_plan20("s03")) == ("Still lake water with real reflections, wet pebbled granite shoreline, packed earth path, "
                                      "real trees with full, irregular canopies, rough bark and uneven branches.")


@pytest.mark.parametrize("key,base", [("lake", "still lake water"), ("pond", "still lake water"), ("river", "still lake water"),
                                      ("water", "still lake water"), ("forest", "forest floor"), ("field", "long uneven grass")])
def test_3_the_new_location_rows(key, base):
    assert sf.location("", f"A woman by the {key}.", "")[1].startswith(base)


def test_3_with_no_keyword_the_previz_objects_decide_before_the_street():
    park = sf.location("", "A woman waves.", "", ["trees", "grass", "a path"])[1]
    assert park.startswith("real grass with worn patches")
    assert sf.location("", "A woman waves.", "", ["the road", "cars", "trees"])[1].startswith("cracked asphalt")  # a road: street
    assert sf.location("", "A woman waves.", "", ["a post"])[1] == sf.NEUTRAL_BASE  # Run 2 (a) 2: no road, no cars
    assert sf.location("", "A woman waves.", "", ["cars"])[1].startswith("cracked asphalt")  # cars: the street


def test_4_s06_rooftop_bench_and_tree():
    before = _shot20("s06")["before"]["positive"]
    assert "a park bench" in before and "real trees with rough bark" in before
    world = _world(_plan20("s06"))
    assert world == ("Tar and gravel roof surface with stains, weathered concrete parapet with rust streaks, galvanised metal "
                     "vents, a weathered building with real wall texture, weathered concrete, a small tree in a large concrete planter, a scuffed metal table, a bench of "
                     "weathered timber slats on metal legs.")
    assert "park bench" not in _plan20("s06").positive


def test_4_the_parked_car_and_the_tree_phrase_everywhere():
    world = _world(_plan20("s04"))  # Run 3: s04 has no car
    assert "car" not in world and "bodywork" not in world
    assert "real trees with full, irregular canopies, rough bark and uneven branches" in world
    parked = sf.world_sentence("", "A man on a bench in the park.", "Park", ["trees", "cars"])
    assert "a parked car with painted bodywork and road dust" in parked and "painted metal bodywork" not in parked
    for phrase in ("a bench of weathered timber slats on metal legs", "a small tree in a large concrete planter",
                   "a parked car with painted bodywork and road dust", "real trees with full, irregular canopies, rough bark and uneven branches"):
        assert _quoted(phrase), phrase


def test_4_every_paragraph_5_phrase_is_the_rules_own():
    for shot in ("s03", "s04", "s05", "s06", "c07", "c08"):
        other = sf.OBJECT_OTHER.format(name="")
        world = _world(_plan20(shot)).rstrip(".").replace(other, other.replace(",", ""))
        for phrase in sorted({*sf.OBJECT_PHRASE.values(), *(p for at in sf.OBJECT_PHRASE_AT.values() for p in at.values())}, key=len, reverse=True):
            world = world.replace(phrase[0].upper() + phrase[1:], "").replace(phrase, "")  # a phrase may hold commas
        for part in (p for p in world.split(", ") if p.strip(", ")):
            assert part.lower() in RULES.lower() or part.endswith(other.replace(",", "")), (shot, part)


# v2.4 6: the light override
def test_6_s06_light_override_has_no_default_sun_and_the_exposure_sentence_once():
    d = _shot20("s06")
    assert "Late-afternoon sun" in d["before"]["positive"]
    p = _plan20("s06").positive
    assert _paras(p)[-2] == _c(sf.no_camera(d["user_look"]) + " " + sf.EXPOSURE)  # Run 2 (d): frame right, not camera right
    assert "Late-afternoon" not in p and p.count("Exposed for the main subject") == 1 and p.count(sf.no_camera(d["user_look"])) == 1
    assert sf.LIGHT_LINE.endswith(sf.EXPOSURE)


def test_6_look_words_that_are_not_light_stay_where_they_were():
    p = sf.plan(S04_SPEC, S04_FRAMING, [MARCUS], [Path("m.jpg")], [MARCUS], user_look="He wears a red scarf. Soft overcast light from the left.", matte=Path("m"))
    paras = _paras(p.positive)
    assert "He wears a red scarf." in paras and paras[-2] == "Soft overcast light from the left. " + _c(sf.EXPOSURE)


# v2.4 7: groups
def test_7_s06_kofi_and_jun_dance_and_every_picked_person_has_their_line_past_the_cap():
    p = _plan20("s06")
    assert len(p.refs) == sf.MAX_REFS == 2  # the VRAM cap stays: Jun is named by his line, not sent as a picture
    paras = _paras(p.positive)
    assert paras[0].endswith("EXACTLY 3 people in frame: Ana on the left, Kofi in the centre, Jun on the right.")
    assert re.sub(r" \w+'s head is about \d+ percent of the frame height\.", "", paras[1]).startswith(
        "Ana: left of frame, about 7 metres from the viewer, dancing. Kofi: centre of frame, about 6 metres "
        "from the viewer, dancing. Jun: right of frame, about 6 metres from the viewer, dancing.")
    assert sf.DEFAULT_ACTION not in paras[1] and sf.DEFAULT_ACTION in _shot20("s06")["before"]["positive"]
    cast = _shot20("s06")["cast"]
    assert paras[3] == _c(f"Ana is the one in image 3: {cast[0]['line']}. Kofi is the one in image 4: {cast[1]['line']}. "
                          f"Jun: {cast[2]['line']}.")


@pytest.mark.parametrize("sentence", ["Three dancers on a rooftop.", "On the rooftop they dance."])
def test_7_a_group_action_in_the_users_words_applies_to_everyone(sentence):
    framing = {**_shot20("s06")["framing"], "subjects": [dict(s, doing=None, word=None) for s in _shot20("s06")["framing"]["subjects"]]}
    para = _paras(sf.plan({"notes": sentence}, framing, [], [], [], matte=Path("m")).positive)[1]
    assert para.count("dancing") == 3 and sf.DEFAULT_ACTION not in para


def test_7_a_mutual_action_names_two_different_people_and_2plus_people_add_the_group_negative():
    p = _plan20("s05")
    assert _paras(p.positive)[3].endswith("Tom is the one in image 4: " + _shot20("s05")["cast"][1]["line"] + ". Priya and Tom are two different people.")
    assert _quoted(sf.DIFFERENT_PEOPLE.format(names="[A] and [B]", count="two"))
    assert _quoted(f"{sf.GROUP_NEGATIVE}, {sf.GROUP_NEGATIVE_PAIR}")
    assert "duplicate person, twins, clone, third person" in _plan20("s05").negative  # a pair
    # Run 3: three people never name the unwanted count ("third person") anywhere.
    assert "duplicate person, twins, clone" in _plan20("s06").negative and "third person" not in _plan20("s06").negative
    for shot in ("s03", "s04", "c07"):
        assert sf.GROUP_NEGATIVE not in _plan20(shot).negative, shot
    assert "two different people" not in _plan20("s06").positive  # dancing together isn't contact


# v2.4 1: ref captions
def test_1_c07_caption_reads_as_written_from_the_saved_line():
    d = _shot20("c07")
    assert d["before"]["positive"].count("shown three") == 1 and len(d["cast"][0]["line"]) == 140
    p = _plan20("c07")
    assert _paras(p.positive)[3] == "Green coupe car is the one in image 3: a dark green 1970s fastback muscle coupe with chrome bumpers."
    assert "reference sheet" not in p.positive and "the same" not in p.positive and "shown" not in p.positive


def test_1_the_full_sheet_prompt_and_the_c08_cut():
    full = ("Car reference sheet on a plain light gray studio background: the same dark green 1970s fastback muscle coupe with "
            "chrome bumpers shown three times, front three-quarter view, side profile, rear three-quarter view. Photoreal, even studio light.")
    assert sf.cast_line(full) == "a dark green 1970s fastback muscle coupe with chrome bumpers"
    assert _paras(_plan20("c08").positive)[3] == "Red sedan is the one in image 3: a glossy cherry red modern four-door sports sedan with black wheels."


def test_1_cast_lines_never_cut_mid_sentence():
    assert sf.cast_line("Car reference sheet on a plain light gray studio background: the same red coupe, black wheels") == "a red coupe, black wheels"
    cut = ("a man in his 60s, navy wool peacoat, flat grey tweed cap, brown leather boots, " * 2)[:140]  # an old save's cut
    assert len(cut) == 140 and sf.cast_line(cut) == cut[: cut.rfind(", ")]
    assert sf.cast_line(MARCUS["line"]) == MARCUS["line"]  # a whole line stays whole


def test_d_the_sheet_and_images_page_save_whole_clauses():
    from pickups.tasks.sheet import _whole
    text = "a woman in her 20s, " * 12
    assert _whole(text.strip(), 200).endswith("in her 20s") and len(_whole(text.strip(), 200)) <= 200
    assert _whole("short line", 200) == "short line"


def test_e_frame_0_keeps_the_cameras_real_move():
    from pickups.app import previz_maker
    move = {"start": {"pos": [0, 4.3, 6], "look": [0, 1, 0]}, "end": {"pos": [0, 1.5, 6], "look": [0, 1, 0]}}
    kept = previz_maker._framing({"horizon": 0.4, "move": move})
    assert kept["move"] == {"start": {"pos": [0.0, 4.3, 6.0], "look": [0.0, 1.0, 0.0]}, "end": {"pos": [0.0, 1.5, 6.0], "look": [0.0, 1.0, 0.0]}}
    assert "move" not in previz_maker._framing({"horizon": 0.4, "move": {"start": {"pos": [0, 1]}, "end": move["end"]}})

