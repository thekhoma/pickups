"""Trim & crop (POST /api/shots/{id}/prepare): a new shot from part of a source, frame-exact, cropped to a shape, with
its sound and timecode; the original untouched. And delivery in a master's own shape (4:3 stays 4:3)."""

import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pickups import media
from pickups.tasks import common


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="module")
def four_three(tmp_path_factory):
    """3 s of 4:3 picture (72 frames at 24 fps, every frame different) with sound and a start timecode."""
    d = tmp_path_factory.mktemp("prep")
    out = d / "interview.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-t", 3, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-timecode", "01:00:10:00", out)
    return out


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


def _frames(path, vf="", size=None):
    return list(common.read_frames(path, vf, size))


def test_crop_box_keeps_the_shape_inside_the_frame():
    # 16:9 from a 4:3 frame: the whole width, centred, even.
    assert media.crop_box(1440, 1080, {"aspect": "16:9", "x": 0, "y": 0, "w": 1440, "h": 1080}) == (0, 134, 1440, 810)
    # A box dragged past the edge is pulled back in, keeping its shape.
    x, y, w, h = media.crop_box(1920, 1080, {"aspect": "9:16", "x": 1700, "y": -50, "w": 608, "h": 1200})
    assert x + w <= 1920 and y >= 0 and h <= 1080 and w % 2 == 0 and h % 2 == 0 and abs(w / h - 9 / 16) < 0.01
    # Free crop (no aspect): the box as drawn, in even pixels.
    assert media.crop_box(1920, 1080, {"x": 101, "y": 51, "w": 801, "h": 401}) == (100, 50, 800, 400)
    # The whole frame, or nothing asked: no crop at all.
    assert media.crop_box(1920, 1080, {"aspect": "16:9", "x": 0, "y": 0, "w": 1920, "h": 1080}) is None
    assert media.crop_box(1920, 1080, None) is None
    with pytest.raises(ValueError, match="too small"):
        media.crop_box(1920, 1080, {"x": 0, "y": 0, "w": 20, "h": 20})
    with pytest.raises(ValueError, match="Unknown shape"):
        media.crop_box(1920, 1080, {"aspect": "3:1", "x": 0, "y": 0, "w": 100, "h": 100})


def test_trim_range_is_inclusive_and_long_enough():
    assert media.trim_range(264, 24, 119) == (24, 96)  # in 24, out 119 kept: 96 frames
    assert media.trim_range(264, None, None) == (0, 264)
    with pytest.raises(ValueError, match="too short"):
        media.trim_range(264, 10, 10 + media.MIN_FRAMES - 2)
    for a, b in ((-1, 50), (10, 264), (50, 40)):
        with pytest.raises(ValueError, match="inside the clip"):
            media.trim_range(264, a, b)


def test_prepare_makes_a_new_trimmed_cropped_shot(client, four_three):
    c, store = client
    project = store.create_project("Interview")
    src = store.add_shot(four_three, "interview", project_id=project["id"])
    before = (store.source_path(src["id"]).stat().st_size, store.shot_meta(src["id"]))

    r = c.post(f"/api/shots/{src['id']}/prepare",
               json={"in_frame": 12, "out_frame": 35, "crop": {"aspect": "16:9", "x": 0, "y": 30, "w": 320, "h": 180}})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["id"] != src["id"] and new["project_id"] == project["id"]
    assert (new["width"], new["height"], new["frames"]) == (320, 180, 24)  # exactly in..out, cropped to 16:9
    assert new["name"] == "interview (0.5–1.5 s, 16:9)"

    out = store.source_path(new["id"])
    p = media.probe(out)
    assert out.suffix == ".mov" and (p.width, p.height, p.frames, p.fps) == (320, 180, 24, 24)
    assert media.timecode(out) == "01:00:10:12"  # the in point's timecode
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "prores" in codec and "HQ" in codec
    # The sound is trimmed to the picture (1 s).
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0",
                        str(out)], capture_output=True, text=True).stdout
    assert abs(float(a) - 1.0) < 0.03

    # Frame for frame: the new shot's first and last frames are source frames 12 and 35, cropped (y 30 to 210).
    got = _frames(out)
    srcf = _frames(four_three, "crop=320:180:0:30", (320, 180))
    err = lambda i, j: float(np.abs(got[i].astype(np.float32) - srcf[j].astype(np.float32)).mean())  # noqa: E731
    assert err(0, 12) < 3 and err(0, 12) < min(err(0, 11), err(0, 13))
    assert err(23, 35) < 3 and err(23, 35) < min(err(23, 34), err(23, 36))

    # The original is untouched and still in the project, and the new shot says where it came from.
    assert (store.source_path(src["id"]).stat().st_size, store.shot_meta(src["id"])) == before
    assert {s["id"] for s in store.shots(project["id"])} == {src["id"], new["id"]}
    assert store.shot_meta(new["id"])["prepared_from"] == {"shot_id": src["id"], "in_frame": 12, "out_frame": 35,
                                                           "crop": [0, 30, 320, 180], "aspect": "16:9"}
    assert not list((store.root).glob("prepare_*"))  # no temporary file left behind


def test_prepare_trim_only_and_bad_requests(client, four_three):
    c, store = client
    src = store.add_shot(four_three, "interview")
    r = c.post(f"/api/shots/{src['id']}/prepare", json={"in_frame": 0, "out_frame": 47, "crop": None, "name": "Answer one"})
    assert r.status_code == 200 and (r.json()["width"], r.json()["height"], r.json()["frames"], r.json()["name"]) == (320, 240, 48, "Answer one")
    assert media.timecode(store.source_path(r.json()["id"])) == "01:00:10:00"
    assert c.post(f"/api/shots/{src['id']}/prepare", json={"in_frame": 30, "out_frame": 32}).status_code == 400
    assert c.post(f"/api/shots/{src['id']}/prepare", json={"in_frame": 0, "out_frame": 999}).status_code == 400
    bad = c.post(f"/api/shots/{src['id']}/prepare", json={"crop": {"aspect": "16:9", "x": 0, "y": 0, "w": 10, "h": 10}})
    assert bad.status_code == 400 and "too small" in bad.json()["detail"]
    assert c.post("/api/shots/shot_nope/prepare", json={}).status_code == 404
    assert len(store.shots()) == 2  # only the good request made a shot


def test_the_old_trim_route_still_works(client, four_three):
    c, store = client
    src = store.add_shot(four_three, "interview")
    r = c.post(f"/api/shots/{src['id']}/trim", json={"seconds": 1})
    assert r.status_code == 200 and r.json()["id"] != src["id"] and r.json()["duration_s"] <= 1.1


def test_status_gives_the_source_limits(client):
    c, _ = client
    st = c.get("/api/status").json()
    assert (st["max_frames"], st["max_long"], st["min_frames"]) == (common.MAX_FRAMES, common.MAX_LONG, media.MIN_FRAMES) == (97, 3840, 9)


# --- delivery in the master's own shape ---------------------------------------------------------------------------
def test_delivery_keeps_the_masters_shape():
    assert common.delivery_sizes(3840, 2160) == [("UHD", 3840, 2160), ("HD", 1920, 1080)]
    assert common.delivery_sizes(1920, 1088) == [("HD", 1920, 1080)]  # a 16:9 render size: standard HD
    assert common.delivery_sizes(1080, 1920) == [("HD", 1080, 1920)]  # vertical stays vertical
    assert common.delivery_sizes(1440, 1080) == [("HD", 1920, 1440)]  # 4:3 stays 4:3
    assert common.delivery_sizes(3840, 2880) == [("UHD", 3840, 2880), ("HD", 1920, 1440)]
    assert common.delivery_sizes(3840, 2748) == [("UHD", 3840, 2748), ("HD", 1920, 1374)]  # open gate
    assert common.delivery_sizes(1080, 1350) == [("HD", 1536, 1920)]  # 4:5
    assert common.delivery_sizes(1080, 1080) == [("HD", 1920, 1920)]
    assert common.delivery_sizes(1920, 804) == [("HD", 1920, 804)]  # 2.39:1
    # Change weather on a 4:3 shot renders 1664x1152 and delivers that shape at HD.
    assert common.size_details(1664, 1152, 97, 24.0) == {"resolution": "1920×1330", "rendered_at": "1664×1152", "frames": 97, "fps": 24.0}


def test_a_four_three_master_is_delivered_four_three(tmp_path):
    master = tmp_path / "master.mov"
    fw = common.FrameWriter(master, 320, 240, 24.0)
    for i in range(9):
        fw.write(np.full((240, 320, 3), 20 * i, np.uint8))
    fw.close()
    files = common.deliverables(master, tmp_path)
    assert {(f["size"], f["codec"]) for f in files} >= {("HD", "prores"), ("HD", "h264"), ("HD", "h265"), ("native", "prores")}
    for f in files:
        p = media.probe(tmp_path / f["file"])
        assert p.width * 3 == p.height * 4, f["file"]  # nothing cropped off the top and bottom
    assert any("1920×1440" in f["label"] for f in files)
