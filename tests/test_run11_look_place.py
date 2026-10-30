"""Run 11 G: the Look's named place wins over the previz scene's ground words, in the start still and the video line.

R&D's Run 10 check 1: the Look said "A man in a long coat stands on a pier at dusk", but that sentence went to the
light paragraph (for its "dusk") and the lake's ground ("wet pebbled granite shoreline, packed earth path") was the only
ground the start still named. The man stood on the shore path beside the pier, not on it.
"""

from pickups import prompt_words
from pickups import start_frame as sf

AUTO_LOOK = ("It is dusk. The man wears a long coat. The grey figure is a man in a long coat who stands on a pier at dusk. "
             "A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean. "
             "Practical lights glow against deep blue shadows, soft highlight roll-off, fine grain.")
SENTENCE = "A man in a long coat stands on a pier at dusk. The camera pushes in slowly from behind and circles to his face."
FRAMING = {"camera": {"height_m": 1.9, "pitch_deg": -10, "focal_mm": 30}, "place": "lake",
           "subjects": [{"kind": "person", "word": "man", "x": 0.5, "distance_m": 6.0, "in_view": True, "facing": "away",
                         "doing": "stand", "h": 0.46}]}
SHORE = ("wet pebbled granite shoreline", "packed earth path")


def _paras(text):
    return text.split("\n\n")


def test_the_pier_look_stands_him_on_the_pier_not_the_shore():
    p = sf.plan({"notes": SENTENCE}, FRAMING, [], [], [], AUTO_LOOK)
    paras = _paras(p.positive)
    blocking = next(x for x in paras if x.startswith("Person 1:"))
    # The place sentence is beside the subject's own line, before the size sentence.
    assert "back to the viewer, standing. A man in a long coat stands on a pier at dusk. Person 1's head" in blocking
    assert not any(w in p.positive for w in SHORE), p.positive
    world = next(x for x in paras if "lake water" in x)
    assert world.startswith("The weathered timber boards of the pier underfoot, still lake water with real reflections")
    assert "It is dusk." in p.positive and "long coat" in p.positive
    assert p.positive.count("stands on a pier") == 1


def test_a_look_with_no_place_keeps_the_scenes_ground():
    p = sf.plan({"notes": SENTENCE}, FRAMING, [], [], [], "It is dusk. The man wears a long coat.")
    assert all(w in p.positive for w in SHORE)


def test_every_standing_place_takes_its_own_surface():
    for look, word in (("She waits on the old stone bridge.", "bridge"), ("He stands on a balcony at night.", "balcony"),
                       ("A woman sits on the wooden jetty.", "jetty")):
        assert sf.standing_place(look) == (word, look)
        p = sf.plan({"notes": "A person by a lake."}, FRAMING, [], [], [], look)
        assert not any(w in p.positive for w in SHORE), (word, p.positive)
        assert word in p.positive


def test_the_light_across_a_rooftop_is_no_place_to_stand():
    assert sf.standing_place("Low golden sun rakes across the rooftop.") is None
    assert sf.standing_place("The man looks at the pier.") is None  # looking at it, not standing on it


def test_the_rooftop_look_keeps_the_rooftop_base():
    p = sf.plan({"notes": "A man on a roof."}, {**FRAMING, "place": ""}, [], [], [], "A man stands on a rooftop at noon.")
    assert "Tar and gravel roof surface" in p.positive
    assert "High midday sun" in p.positive  # its "at noon" still lights the still from beside the subject


def test_the_video_line_says_the_coat_the_dusk_and_the_pier_once():
    man = [{"kind": "person", "word": "man", "in_view": True}]
    out = prompt_words.l2r_prompt("A man stands still. The camera pushes in, then orbits around him.", man, look=AUTO_LOOK)
    assert out == ("A man in a long coat stands on a pier at dusk. The camera pushes in, then orbits around him. "
                   "Keep the same man and camera as the driver.")
    # Facts the Look adds that the line lacks, each once and in the fewest sentences.
    out = prompt_words.l2r_prompt("A man waits. The camera pushes in.", man, look="It is dusk. The man wears a long coat.")
    assert out == "A man waits. It is dusk. The man wears a long coat. The camera pushes in. Keep the same man and camera as the driver."
    # R&D's own sentence already says it all: unchanged.
    said = "A man in a long coat stands on a pier at dusk. The camera pushes in, then orbits around him."
    assert prompt_words.l2r_prompt(said, man, look=AUTO_LOOK) == f"{said} Keep the same man and camera as the driver."
