import subprocess

import pytest

from pickups import media
from pickups.app.store import EXAMPLES, UNSORTED, Store, download_name


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    return Store(tmp_path / "library")


@pytest.fixture()
def clip(tmp_path):
    f = tmp_path / "clip.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "1",
                    "-timecode", "01:00:10:00", str(f)], check=True)
    return f


def test_builtin_projects(store):
    assert store.project(EXAMPLES)["name"] == "Examples"  # always there, and opens by link
    names = {p["id"]: p["name"] for p in store.projects()}
    assert names == {UNSORTED: "My shots"}  # Examples lists only once it holds something (test_projects_nav)


def test_shot_lands_in_project_and_moves(store, clip):
    prj = store.create_project("Coastal spot")
    shot = store.add_shot(clip, "Pier", project_id=prj["id"])
    assert shot["project_id"] == prj["id"]
    assert store.add_shot(clip, "stray", project_id="nope")["project_id"] == UNSORTED
    store.move_shot(shot["id"], UNSORTED)
    assert {s["name"] for s in store.shots(UNSORTED)} == {"stray", "Pier"}
    with pytest.raises(FileNotFoundError):
        store.move_shot(shot["id"], "missing")


def test_cast_names_unique_and_editable(store):
    prj = store.create_project("P")
    c = store.add_cast(prj["id"], b"img", "maya.png", "  Maya  ", "person", "a woman in a red coat")
    assert c["name"] == "Maya" and c["url"].endswith(".png")
    with pytest.raises(ValueError):
        store.add_cast(prj["id"], b"img", "b.png", "maya")
    with pytest.raises(ValueError):
        store.add_cast(prj["id"], b"img", "b.png", "Car", kind="vehicle")
    c2 = store.update_cast(prj["id"], c["id"], line="a woman in a coat", kind="person")
    assert c2["line"] == "a woman in a coat"


def test_take_status_hidden_notes_and_deliveries(store, clip):
    prj = store.create_project("P")
    shot = store.add_shot(clip, "Shot one", project_id=prj["id"])
    vid, vdir = store.new_version_dir(shot["id"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-c:v", "libx264", str(vdir / "video.mp4")], check=True)
    (vdir / "master.mov").write_bytes(clip.read_bytes())
    store.save_version(shot["id"], vid, {"label": "Remove", "task": "remove",
                                         "downloads": [{"label": "ProRes", "file": "master.mov", "codec": "prores", "size": "HD"}]})
    s = store.update_take(shot["id"], vid, status="approved")
    take = s["versions"][1]
    assert take["status"] == "approved" and s["approved"] == 1
    assert take["downloads"][0]["filename"] == "Shot_one_T02_remove_ProRes422HQ_HD.mov"
    with pytest.raises(ValueError):
        store.update_take(shot["id"], vid, status="review")
    assert store.update_take(shot["id"], vid, hidden=True)["versions"][1]["hidden"] is True
    n = store.add_note(shot["id"], vid, 12, "  more grain  ")
    assert store.shot(shot["id"])["versions"][1]["notes"][0]["text"] == "more grain"
    store.delete_note(shot["id"], vid, n["id"])
    assert store.shot(shot["id"])["versions"][1]["notes"] == []
    d = store.deliveries(prj["id"])
    assert len(d) == 1 and d[0]["take"]["id"] == vid
    p = store.project(prj["id"])
    assert (p["shots"], p["takes"], p["approved"]) == (1, 1, 1)


def test_take_order_survives_writes(store, clip):
    """Takes keep the order they were made in when an older take gets a note, an approval or a cached file."""
    import os
    shot = store.add_shot(clip, "Order")
    ids = []
    for i in range(3):
        vid, vdir = store.new_version_dir(shot["id"])
        store.save_version(shot["id"], vid, {"label": f"T{i}", "task": "remove", "created_at": 1000.0 + i})
        os.utime(vdir, (2000.0 + i, 2000.0 + i))
        ids.append(vid)
    first = store.shot_dir(shot["id"]) / "versions" / ids[0]
    store.add_note(shot["id"], ids[0], 3, "later")
    store.update_take(shot["id"], ids[0], status="approved")
    (first / "receipt_after.jpg").write_bytes(b"x")
    os.utime(first, (3000.0, 3000.0))
    assert [v["id"] for v in store.shot(shot["id"])["versions"][1:]] == ids


def test_download_name():
    dl = {"url": "/media/shots/s/versions/v/uhd_3840x2160_h265.mp4", "codec": "h265", "size": "UHD"}
    assert download_name("Harbour pier!", 3, "finish", dl) == "Harbour_pier_T03_finish_H265_UHD.mp4"
    assert download_name("x", 1, None, {"url": "/a/exr_acescg.zip", "codec": "exr"}) == "x_T01_original_exr_acescg.zip"


def test_timecode_helpers(clip, tmp_path):
    assert media.timecode(clip) == "01:00:10:00"
    assert media.offset_timecode("01:00:10:00", 30, 24) == "01:00:11:06"
    out = tmp_path / "t"
    out.mkdir()
    (out / "master.mov").write_bytes(clip.read_bytes())
    assert media.stamp_timecode(out, "02:00:00:00") == 1 and media.timecode(out / "master.mov") == "02:00:00:00"
