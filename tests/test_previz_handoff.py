"""Frame 0's view (what it shows, what's out of frame, what comes into view later) is kept with a made previz, cleaned."""

from pickups.app import previz_maker


def test_the_frame_0_view_place_and_reveals_are_kept_and_cleaned():
    fr = previz_maker._framing({
        "subject": [0.4, 0.2, 0.6, 0.9],
        "horizon": 0.45,
        "view": {"share": {"grass": 0.52, "trees": 0.2, "x" * 50: 0.1, "bad": "a lot"}, "x": {"grass": 0.5, "nope": 0.1},
                 "outside": ["water", 7, "  "], "actors": [0, 9, "1"], "blocked": 0.0, "bare": 1.7, "inside": None},
        "place": "  Lake  ",
        "reveals": [{"word": "water", "at_s": 2, "side": "left"}, {"word": "trees", "at_s": 2, "side": "up"},
                    {"word": "rocks", "at_s": 99, "side": "right"}],
    })
    assert fr["subject"] == [0.4, 0.2, 0.6, 0.9] and fr["horizon"] == 0.45
    assert fr["view"]["share"] == {"grass": 0.52, "trees": 0.2}
    assert fr["view"]["x"] == {"grass": 0.5}
    assert fr["view"]["outside"] == ["water"]
    assert fr["view"]["actors"] == [0]
    assert fr["view"]["bare"] == 1.0
    assert fr["place"] == "Lake"
    assert fr["reveals"] == [{"word": "water", "at_s": 2.0, "side": "left"}]


def test_an_old_page_without_a_view_still_keeps_its_framing():
    assert previz_maker._framing({"subject": [0.1, 0.1, 0.3, 0.5]}) == {"subject": [0.1, 0.1, 0.3, 0.5]}
    assert previz_maker._framing({"view": "nonsense"}) is None
