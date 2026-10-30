import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from pickups import nle

F24 = Fraction(24000, 1001)


def clip(name, tc, frames=97, fps=F24):
    return nle.Clip(name=name, reel=name, path=Path(f"/tmp/{name}.mov"), fps=fps, frames=frames, width=1920, height=1080, start_tc=tc)


def test_timecode_math():
    assert nle.tc_to_frames("01:00:00:00", F24) == 86400
    assert nle.frames_to_tc(86400 + 25, F24) == "01:00:01:01"
    assert nle.frames_to_tc(nle.tc_to_frames("10:59:59:29", Fraction(30000, 1001)) + 1, Fraction(30000, 1001)) == "11:00:00:00"


def test_layout_places_by_timecode_and_stacks_overlaps():
    a, b, c = clip("a", "01:00:00:00"), clip("b", "01:00:02:00"), clip("c", "01:00:10:00")
    untimed = clip("u", None, frames=10)
    start, placed = nle.layout([c, a, b, untimed])
    pos = {cl.name: (s, t) for cl, s, t in placed}
    assert start == 86400
    assert pos["a"] == (0, 0)
    assert pos["b"] == (48, 1)  # overlaps a (97 frames) -> second track
    assert pos["c"] == (240, 0)
    assert pos["u"] == (240 + 97, 0)  # untimed after the last timed clip


def test_xmeml_is_valid_xml_with_ntsc_rates():
    x = nle.xmeml([clip("Pier & co", "01:00:00:00"), clip("b", "01:00:02:00")], "Test <seq>")
    root = ET.fromstring(x.split("\n", 2)[2])  # skip the XML declaration and DOCTYPE
    assert root.tag == "xmeml"
    assert root.find("./sequence/name").text == "Test <seq>"
    assert len(root.findall("./sequence/media/video/track")) == 2
    assert root.find(".//rate/ntsc").text == "TRUE"
    assert root.find(".//file/pathurl").text.startswith("file://localhost/")


def test_edl_lines():
    e = nle.edl([clip("Harbour pier", "01:00:00:00")], "T")
    line = [l for l in e.splitlines() if l.startswith("001")][0]
    assert "01:00:00:00 01:00:04:01 01:00:00:00 01:00:04:01" in line
    assert "* FROM CLIP NAME: Harbour pier.mov" in e


def seq_markers(x):
    root = ET.fromstring(x.split("\n", 2)[2])
    return root, [(int(m.find("in").text), m.find("name").text, m.find("comment").text) for m in root.findall("./sequence/marker")]


def test_notes_become_sequence_markers_at_their_record_frame():
    a = clip("a", "01:00:00:00")  # record 0..97
    a.notes = [(36, "Fix the <halo> & the sign"), (4, "Too dark\non the left")]
    b = clip("b", "01:00:10:00", frames=50)  # record 240..290
    b.notes = [(200, "Past the end"), (10, "Sign")]
    root, ms = seq_markers(nle.xmeml([b, a], "T"))
    # In frame order, at the clip's record position + the note's frame; a note past the end sits on the last frame.
    assert [(f, n) for f, n, _ in ms] == [(4, "Too dark on the left"), (36, "Fix the <halo> & the sign"), (250, "Sign"), (289, "Past the end")]
    assert ms[1][2] == "Fix the <halo> & the sign (a)"  # the comment names the take
    m = root.find("./sequence/marker")
    assert [c.tag for c in m] == ["name", "comment", "in", "out"] and m.find("out").text == "-1"
    # Sequence markers only: clip markers would show each note twice in Premiere.
    assert not list(root.iter("clipitem"))[0].findall("marker") and not root.findall(".//file/marker")
    # Before <media>, after the sequence's timecode.
    tags = [c.tag for c in root.find("sequence")]
    assert tags.index("timecode") < tags.index("marker") < tags.index("media")


def test_stacked_takes_share_a_marker_on_the_same_frame():
    a, b = clip("a", "01:00:00:00"), clip("b", "01:00:02:00")  # b overlaps a: track 2, record 48
    a.notes, b.notes = [(60, "Halo")], [(12, "Flicker")]
    _, ms = seq_markers(nle.xmeml([a, b], "T"))
    assert ms == [(60, "Halo / Flicker", "Halo / Flicker (a, b)")]


def test_marker_text_is_escaped_and_cleaned():
    a = clip("a", "01:00:00:00")
    a.notes = [(0, 'Say "cut" & <go>\x07'), (1, "   "), (2, "x" * 80)]
    x = nle.xmeml([a], "T")
    assert "<name>Say \"cut\" &amp; &lt;go&gt;</name>" in x
    assert "\x07" not in x
    _, ms = seq_markers(x)
    assert len(ms) == 2  # blank notes are dropped
    assert ms[1][1] == "x" * 59 + "…" and ms[1][2] == "x" * 80 + " (a)"


def test_edl_locators_at_record_timecode_across_clips():
    a = clip("a", "01:00:00:00")  # record 01:00:00:00, 97 frames
    a.notes = [(12, "Halo on the\r\nmast")]
    b = clip("b", "01:00:10:00")  # record 01:00:10:00 (240 frames in)
    b.notes = [(30, "Sign still readable"), (0, "Starts here")]
    u = clip("u", None, frames=24)  # untimed: after the last timed clip, at record 01:00:14:01
    u.notes = [(5, "Flash")]
    lines = nle.edl([b, a, u], "T").splitlines()
    locs = [line for line in lines if line.startswith("* LOC:")]
    assert locs == [
        "* LOC: 01:00:00:12 BLUE    Halo on the mast",
        "* LOC: 01:00:10:00 BLUE    Starts here",
        "* LOC: 01:00:11:06 BLUE    Sign still readable",
        "* LOC: 01:00:14:06 BLUE    Flash",
    ]
    # Each locator follows its own event, before the blank line that ends it.
    i = lines.index("* LOC: 01:00:00:12 BLUE    Halo on the mast")
    assert lines[i - 1] == "* COMMENT: a" and lines[i + 1] == "" and lines[i + 2].startswith("002")


def test_no_notes_no_markers():
    x = nle.xmeml([clip("a", "01:00:00:00")], "T")
    assert "<marker>" not in x and "LOC:" not in nle.edl([clip("a", "01:00:00:00")], "T")


def test_export_route_carries_take_notes(tmp_path, monkeypatch):
    import subprocess

    from fastapi.testclient import TestClient

    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    c = TestClient(server.create_app("http://127.0.0.1:9"))  # a dead local address: no ComfyUI

    src = tmp_path / "pier.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "1",
                    "-timecode", "01:00:10:00", str(src)], check=True)
    prj = store.create_project("Notes test")
    shot = store.add_shot(src, "Pier", project_id=prj["id"])
    sid = shot["id"]
    assert c.post(f"/api/shots/{sid}/takes/orig/notes", json={"frame": 6, "text": "Boom in frame"}).status_code == 200
    gone = c.post(f"/api/shots/{sid}/takes/orig/notes", json={"frame": 9, "text": "Deleted"}).json()
    assert c.delete(f"/api/shots/{sid}/takes/orig/notes/{gone['id']}").status_code == 200
    assert c.post(f"/api/shots/{sid}/takes/orig/notes", json={"frame": 1, "text": "  "}).status_code == 400
    store.update_take(sid, "orig", status="approved")

    x = c.get(f"/api/projects/{prj['id']}/export/xml")
    assert x.status_code == 200
    _, ms = seq_markers(x.text)
    assert ms == [(6, "Boom in frame", "Boom in frame (Pier · T01 Original)")]
    e = c.get(f"/api/projects/{prj['id']}/export/edl").text
    assert "* LOC: 01:00:10:06 BLUE    Boom in frame" in e and "Deleted" not in e
    assert str(tmp_path) not in e  # file names only in the EDL


def test_markers_keep_a_takes_notes_inside_and_drop_folded_ones_outside():
    a = clip("a", "01:00:00:00", frames=10)
    a.notes = [(25, "Past the end")]
    a.original_notes = [(25, "Outside"), (-1, "Before"), (3, "Inside")]  # nle.fold_original_notes drops these too
    assert nle.markers(a) == [(3, "Inside", True), (9, "Past the end", False)]
