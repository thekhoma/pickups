"""Notes on a shot's original reach the edit with the shot's approved takes (Export timeline)."""

import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pickups import nle

F24 = Fraction(24)


def _shot(orig_notes, *takes, orig_status="", fps=24.0):
    """A shot dict as Store.shot returns it: the original (with notes) and takes (id, input_version, params, status)."""
    orig = {"id": "orig", "status": orig_status, "input_version": None, "params": {},
            "notes": [{"id": f"n{i}", "frame": f, "text": t} for i, (f, t) in enumerate(orig_notes)]}
    return {"id": "shot_1", "name": "Pier", "fps": fps, "versions": [orig, *takes]}


def _take(vid, input_version="orig", status="approved", **params):
    return {"id": vid, "status": status, "notes": [], "input_version": input_version, "params": params}


# --- the fold and its frame mapping -----------------------------------------------------------------------------------


def test_a_take_of_the_whole_original_maps_one_to_one():
    shot = _shot([(30, "Boom in frame"), (4, "Too dark")], _take("v_a"))
    assert nle.take_start(shot, "v_a") == 0
    assert nle.fold_original_notes(shot, "v_a", 97, 24.0) == [(4, "Too dark"), (30, "Boom in frame")]


def test_a_trimmed_take_maps_from_its_start_and_drops_notes_outside_it():
    # The take is frames 40..136 of the original (params.start_frame = 40, 97 frames).
    shot = _shot([(39, "Before it"), (40, "First frame"), (100, "Inside"), (136, "Last frame"), (137, "After it")],
                 _take("v_a", start_frame=40))
    assert nle.take_start(shot, "v_a") == 40
    assert nle.fold_original_notes(shot, "v_a", 97) == [(0, "First frame"), (60, "Inside"), (96, "Last frame")]


def test_a_take_shorter_than_the_original_leaves_later_notes_behind():
    # Recast renders at most 97 frames of a longer shot, from its first frame.
    shot = _shot([(10, "Early"), (150, "Late")], _take("v_a"))
    assert nle.fold_original_notes(shot, "v_a", 97) == [(10, "Early")]


def test_starts_add_up_along_the_takes_it_was_made_from():
    # Finish (no trim) of a weather take cut from frame 12 of a Remove take cut from frame 8: 20 frames in.
    shot = _shot([(25, "Halo")], _take("v_rm", start_frame=8, status=""), _take("v_env", "v_rm", status="", start_frame=12),
                 _take("v_fin", "v_env"))
    assert nle.take_start(shot, "v_fin") == 20
    assert nle.fold_original_notes(shot, "v_fin", 50) == [(5, "Halo")]


def test_nothing_folds_when_the_original_is_approved_or_the_take_cant_be_placed():
    assert nle.fold_original_notes(_shot([(5, "Boom")], _take("v_a"), orig_status="approved"), "v_a", 97) == []
    assert nle.fold_original_notes(_shot([(5, "Boom")], _take("v_a")), "orig", 97) == []
    broken = _shot([(5, "Boom")], _take("v_a", "v_gone"))
    assert nle.take_start(broken, "v_a") is None and nle.fold_original_notes(broken, "v_a", 97) == []
    loop = _shot([(5, "Boom")], _take("v_a", "v_b"), _take("v_b", "v_a"))
    assert nle.take_start(loop, "v_a") is None and nle.fold_original_notes(loop, "v_a", 97) == []
    assert nle.fold_original_notes(_shot([(5, "Boom")], _take("v_a", start_frame="soon")), "v_a", 97) == []


def test_bad_or_blank_notes_are_skipped():
    shot = _shot([(5, "Boom"), ("x", "Bad frame"), (6, "   "), (None, "No frame")], _take("v_a"))
    assert nle.fold_original_notes(shot, "v_a", 97) == [(0, "No frame"), (5, "Boom")]


def test_a_take_at_another_frame_rate_maps_by_time():
    shot = _shot([(48, "Two seconds in")], _take("v_a"), fps=24.0)
    assert nle.fold_original_notes(shot, "v_a", 200, 48.0) == [(96, "Two seconds in")]
    assert nle.fold_original_notes(shot, "v_a", 200, 24.0) == [(48, "Two seconds in")]


# --- markers and locators ---------------------------------------------------------------------------------------------


def _clip(name, tc, frames=97, reel="Pier"):
    return nle.Clip(name=name, reel=reel, path=Path(f"/tmp/{name}.mov"), fps=F24, frames=frames, width=1920, height=1080, start_tc=tc)


def _markers(x):
    root = ET.fromstring(x.split("\n", 2)[2])
    return [(int(m.find("in").text), m.find("name").text, m.find("comment").text) for m in root.findall("./sequence/marker")]


def test_folded_notes_say_original_in_the_comment_and_the_locator():
    a = _clip("Pier · T02 Recast", "01:00:00:00")
    a.notes = [(10, "Hair too dark")]
    a.original_notes = [(4, "Boom in frame"), (200, "Past the take")]  # outside the clip: dropped, never clamped
    ms = _markers(nle.xmeml([a], "T"))
    assert ms == [(4, "Boom in frame", "Boom in frame (Pier · original)"), (10, "Hair too dark", "Hair too dark (Pier · T02 Recast)")]
    locs = [line for line in nle.edl([a], "T").splitlines() if line.startswith("* LOC:")]
    assert locs == ["* LOC: 01:00:00:04 BLUE    Boom in frame (original)", "* LOC: 01:00:00:10 BLUE    Hair too dark"]


def test_an_original_note_folded_into_stacked_takes_is_one_marker():
    # Two approved takes of the same shot over the same source timecode: stacked on two tracks, one marker.
    a, b = _clip("Pier · T02 Remove", "01:00:00:00"), _clip("Pier · T03 Recast", "01:00:00:00")
    a.original_notes = b.original_notes = [(12, "Sign still readable")]
    b.notes = [(12, "Halo")]
    ms = _markers(nle.xmeml([a, b], "T"))
    assert ms == [(12, "Sign still readable / Halo", "Sign still readable / Halo (Pier · original, Pier · T03 Recast)")]


# --- the export route -------------------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    return TestClient(server.create_app("http://127.0.0.1:9")), store  # a dead local address: no ComfyUI


def _video(path, seconds, tc):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", "-timecode", tc, str(path)], check=True)
    return path


def _add_take(store, sid, label, task, seconds, tc, params=None, input_version="orig"):
    """A finished take as a job leaves it: a master stamped with its source timecode, params, approved."""
    vid, vdir = store.new_version_dir(sid)
    _video(vdir / "master.mov", seconds, tc)
    store.save_version(sid, vid, {"label": label, "task": task, "params": params or {}, "input_version": input_version,
                                  "details": {"frames": int(seconds * 24)}, "downloads": []})
    store.update_take(sid, vid, status="approved")
    return vid


def test_export_folds_the_originals_notes_into_the_approved_take(client, tmp_path):
    c, store = client
    prj = store.create_project("Notes test")
    sid = store.add_shot(_video(tmp_path / "pier.mov", 3, "01:00:10:00"), "Pier", project_id=prj["id"])["id"]  # 72 frames
    for f, t in ((6, "Boom in frame"), (30, "Sign"), (60, "After the take")):
        assert c.post(f"/api/shots/{sid}/takes/orig/notes", json={"frame": f, "text": t}).status_code == 200
    # Change weather from frame 24 for 1 s (frames 24..47 of the original), stamped 01:00:11:00 as jobs do.
    vid = _add_take(store, sid, "Weather · Snow", "environment", 1, "01:00:11:00", {"start_frame": 24})
    store.add_note(sid, vid, 2, "Too much snow")

    ms = _markers(c.get(f"/api/projects/{prj['id']}/export/xml").text)
    assert ms == [(2, "Too much snow", "Too much snow (Pier · T02 Weather · Snow)"), (6, "Sign", "Sign (Pier · original)")]
    edl = c.get(f"/api/projects/{prj['id']}/export/edl").text
    locs = [line for line in edl.splitlines() if line.startswith("* LOC:")]
    assert locs == ["* LOC: 01:00:11:02 BLUE    Too much snow", "* LOC: 01:00:11:06 BLUE    Sign (original)"]
    assert "Boom in frame" not in edl and "After the take" not in edl  # before and after the take: they stay behind


def test_an_approved_original_keeps_its_notes_to_itself(client, tmp_path):
    c, store = client
    prj = store.create_project("Notes test")
    sid = store.add_shot(_video(tmp_path / "pier.mov", 1, "01:00:10:00"), "Pier", project_id=prj["id"])["id"]
    c.post(f"/api/shots/{sid}/takes/orig/notes", json={"frame": 6, "text": "Boom in frame"})
    store.update_take(sid, "orig", status="approved")
    _add_take(store, sid, "Remove", "remove", 1, "01:00:10:00")
    ms = _markers(c.get(f"/api/projects/{prj['id']}/export/xml").text)
    assert ms == [(6, "Boom in frame", "Boom in frame (Pier · T01 Original)")]  # once, with the original
    assert c.get(f"/api/projects/{prj['id']}/export/edl").text.count("Boom in frame") == 1


def test_an_approved_takes_receipt_lists_the_originals_notes_too(client, tmp_path):
    c, store = client
    prj = store.create_project("Notes test")
    sid = store.add_shot(_video(tmp_path / "pier.mov", 3, "01:00:10:00"), "Pier", project_id=prj["id"])["id"]
    for f, t in ((6, "Boom in frame"), (30, "Sign")):
        store.add_note(sid, "orig", f, t)
    vid = _add_take(store, sid, "Weather · Snow", "environment", 1, "01:00:11:00", {"start_frame": 24})
    page = c.get(f"/api/shots/{sid}/takes/{vid}/receipt").text
    notes = page[page.index('<ul class="notes">'):]
    assert "<b>00:00:00:06</b>" in notes and "Sign (original)" in notes and "Boom in frame" not in notes
    store.update_take(sid, vid, status="")  # not approved: it isn't going to the edit, so it carries none
    assert "Sign (original)" not in c.get(f"/api/shots/{sid}/takes/{vid}/receipt").text
