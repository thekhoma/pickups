from pickups import mentions

CAST = [
    {"id": "c1", "name": "Maya", "kind": "person", "line": "a woman with short black hair"},
    {"id": "c2", "name": "Old town", "kind": "place", "line": "a cobbled square in an old harbour town"},
    {"id": "c3", "name": "Mayan guide", "kind": "person", "line": "a tour guide in a green jacket"},
    {"id": "c4", "name": "Dusk", "kind": "look", "line": "warm low evening sun and dense blacks"},
]


def test_mentioned_order_and_longest_name():
    t = "@Mayan guide pours while @Maya waits in @Old town; @Maya sighs."
    assert [c["id"] for c in mentions.mentioned(t, CAST)] == ["c3", "c1", "c2"]


def test_for_images_numbers_from_two():
    t, refs = mentions.for_images("@Maya sits in @Old town. @Maya sighs. Graded like @Dusk.", CAST)
    assert t == ("Maya (the person in <image2>) sits in Old town (the place in <image3>). Maya (the person in <image2>) sighs. "
                 "Graded like the look of <image4>.")
    assert [c["id"] for c in refs] == ["c1", "c2", "c4"]


def test_for_images_respects_given_order():
    _t, refs = mentions.for_images("@Old town with @Maya", CAST, order=[CAST[0]])
    assert [c["id"] for c in refs] == ["c1", "c2"]


def test_for_text_describes_first_mention_only():
    t = mentions.for_text("@Maya sits in @Old town as the camera pushes in. @Maya sighs.", CAST)
    assert t == ("Maya, a woman with short black hair, sits in a cobbled square in an old harbour town as the camera "
                 "pushes in. Maya sighs.")


def test_for_text_punctuation_and_unknown():
    assert mentions.for_text("Close on @Maya.", CAST) == "Close on Maya, a woman with short black hair."
    assert mentions.for_text("email me @home", CAST) == "email me @home"
    assert mentions.for_text("@maya waves", CAST).startswith("Maya, a woman")


def test_tag_images():
    assert mentions.tag_images("The man in image 2 sits in Image 3, not <image2>.", 3) == "The man in <image2> sits in <image3>, not <image2>."
    assert mentions.tag_images("Keep image 1's framing.", 1) == "Keep the image's framing."
    assert mentions.tag_images("Keep <image1>'s framing.", 1) == "Keep the image's framing."
    assert mentions.tag_images("the image 2 reference", 2) == "<image2> reference"
