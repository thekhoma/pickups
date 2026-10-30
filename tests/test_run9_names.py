"""Run 9: a made shot keeps its whole first sentence as its name (FableWalk: "A golden retriever runs along a")."""

from pickups import video


def test_a_made_shot_is_named_by_its_whole_first_sentence():
    assert video.shot_name("A golden retriever runs along a beach at sunset, kicking up spray. Slow push in.") == \
        "A golden retriever runs along a beach at sunset, kicking up spray"
    assert video.shot_name("  A woman in a yellow raincoat walks along a pier  ") == "A woman in a yellow raincoat walks along a pier"
    assert video.shot_name("") == "New video"


def test_a_very_long_sentence_stops_at_a_whole_word():
    long = "A " + " ".join(["very"] * 40) + " long shot"
    name = video.shot_name(long)
    assert len(name) <= video.SHOT_NAME_MAX and name.endswith("very") and "  " not in name
