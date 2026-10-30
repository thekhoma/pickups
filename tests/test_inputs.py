"""What people drop on Pickups: phone clips shot upright, 60 fps, long clips, black clips, nobody in shot, tiny people,
sound-only and broken files. Each one gives a worded outcome before any GPU work (or works properly).

Clips are tiny synthetic ones made with ffmpeg's lavfi sources; SAM is stood in for by a fake runner."""

import subprocess
import time
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pickups import media
from pickups.tasks import common


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def _clip(path, src="testsrc=size=320x180:rate=24", seconds=1.0, audio=False):
    extra = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"] if audio else []
    _ff("-f", "lavfi", "-i", src, *extra, "-t", seconds, "-c:v", "libx264", "-pix_fmt", "yuv420p", path)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    land = _clip(d / "land.mp4", seconds=1)
    rot = d / "phone.mp4"  # a phone held upright: stored 320x180, displayed 180x320
    try:
        _ff("-display_rotation", 90, "-i", land, "-c", "copy", rot)
    except subprocess.CalledProcessError:  # older ffmpeg: the rotate tag
        _ff("-i", land, "-c", "copy", "-metadata:s:v:0", "rotate=90", rot)
    return SimpleNamespace(
        dir=d, land=land, rot=rot,
        fps60=_clip(d / "sixty.mp4", "testsrc=size=64x36:rate=60", seconds=3),
        long=_clip(d / "long.mp4", "testsrc=size=64x36:rate=24", seconds=12, audio=True),
        black=_clip(d / "black.mp4", "color=c=black:size=64x36:rate=24", seconds=1),
        grey=_clip(d / "grey.mp4", "color=c=0x808080:size=64x36:rate=24", seconds=1),
        dark=_clip(d / "dark.mp4", "testsrc=size=64x36:rate=24,eq=brightness=-0.45", seconds=1),
    )


# --- vertical phone video ------------------------------------------------------------------------------------------
def test_probe_reads_a_phone_clip_upright(clips):
    p = media.probe(clips.rot)
    assert (p.width, p.height) == (180, 320) and p.rotation in (90, 270)
    assert media.probe(clips.land).rotation == 0 and (media.probe(clips.land).width, media.probe(clips.land).height) == (320, 180)


def test_rotation_from_side_data_or_tag():
    assert media._rotation({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}) == 270
    assert media._rotation({"side_data_list": [{"rotation": 180}]}) == 180
    assert media._rotation({"tags": {"rotate": "90"}}) == 90
    assert media._rotation({"tags": {"rotate": "nonsense"}}) == 0
    assert media._rotation({}) == 0


def test_work_clip_is_upright_not_stretched(clips, tmp_path):
    info = common.work_clip(SimpleNamespace(input_path=clips.rot), tmp_path / "work.mp4")
    assert (info["width"], info["height"]) == (180, 320)
    w = media.probe(tmp_path / "work.mp4")
    assert (w.width, w.height, w.rotation) == (180, 320, 0)  # upright, with no rotation left in the file
    frame = list(common.read_frames(tmp_path / "work.mp4"))[0].astype(np.float32)
    land = list(common.read_frames(clips.land))[0].astype(np.float32)
    # The picture is the landscape frame turned a quarter (not squeezed into a tall frame).
    err = min(np.abs(frame - np.rot90(land, k)).mean() for k in (1, -1))
    assert err < 10


def test_a_vertical_master_is_delivered_vertical(tmp_path):
    assert common.delivery_sizes(1080, 1920) == [("HD", 1080, 1920)]
    assert common.delivery_sizes(2160, 3840) == [("UHD", 2160, 3840), ("HD", 1080, 1920)]
    assert common.delivery_sizes(1920, 1080) == [("HD", 1920, 1080)]
    assert common.size_details(1080, 1920, 97, 30.0)["resolution"] == "1080×1920"
    master = tmp_path / "master.mov"
    fw = common.FrameWriter(master, 180, 320, 30.0)
    for i in range(9):
        fw.write(np.full((320, 180, 3), 20 * i, np.uint8))
    fw.close()
    files = common.deliverables(master, tmp_path)
    assert files and all("1080×1920" in f["label"] or "180×320" in f["label"] for f in files)
    for f in files:
        p = media.probe(tmp_path / f["file"])
        assert p.height > p.width, f["file"]


@pytest.mark.skipif(not media._colour_managed_scale(), reason="this ffmpeg can't convert colour spaces")
def test_a_phones_hdr_clip_keeps_its_colours(tmp_path):
    from pickups.app.store import make_viewer

    ref, hlg = tmp_path / "ref.mp4", tmp_path / "phone_hdr.mp4"  # what a phone records by default: BT.2020 HLG, 10-bit
    _clip(ref, "testsrc2=size=320x180:rate=24", seconds=0.5)
    _ff("-i", ref, "-vf", "scale=in_primaries=bt709:in_transfer=bt709:in_color_matrix=bt709:out_primaries=bt2020:"
        "out_transfer=arib-std-b67:out_color_matrix=bt2020,format=yuv420p10le", "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-c:v", "libx265", "-x265-params", "log-level=error", hlg)
    p = media.probe(hlg)
    assert (p.primaries, p.transfer) == ("bt2020", "arib-std-b67") and media.to_rec709(p)
    assert media.to_rec709(media.probe(ref)) == ""  # everyday footage goes through untouched
    common.work_clip(SimpleNamespace(input_path=hlg), tmp_path / "work.mp4")
    make_viewer(hlg, tmp_path / "viewer.mp4")
    want = list(common.read_frames(ref))[0].astype(np.float32)
    naive = list(common.read_frames(hlg, "format=rgb24"))[0].astype(np.float32)
    for made in ("work.mp4", "viewer.mp4"):
        got = list(common.read_frames(tmp_path / made))[0].astype(np.float32)
        assert media.probe(tmp_path / made).primaries in ("bt709", "")
        # Saturation (spread between channels) comes back; decoded as is it's about half, grey and washed out.
        assert got.std(axis=2).mean() > 0.75 * want.std(axis=2).mean() > naive.std(axis=2).mean()


# --- frame rates ---------------------------------------------------------------------------------------------------
def test_60fps_works_frame_for_frame_at_its_own_rate(clips, tmp_path):
    src = media.probe(clips.fps60)
    assert src.fps == 60 and src.frames == 180
    info = common.work_clip(SimpleNamespace(input_path=clips.fps60), tmp_path / "work.mp4")
    w = media.probe(tmp_path / "work.mp4")
    assert info["fps"] == 60.0 and w.fps == 60 and w.frames == common.MAX_FRAMES  # 1.6 s, not conformed to 30
    assert media.made_from(src, 0, w.frames) == "the first 1.6 s of 3.0 s"


def test_phone_frame_rates_read_as_they_play():
    assert media.frame_rate("24/1", "24/1") == 24
    assert media.frame_rate("30000/1001", "30000/1001") == Fraction(30000, 1001)
    assert media.frame_rate("30/1", "2997/100") == 30  # a little variable: the stream rate
    assert media.frame_rate("90000/1", "30000/1001") == Fraction(30000, 1001)  # a time base, not a rate
    assert media.frame_rate("60/1", "7183/240") == Fraction(30000, 1001)  # peak 60, plays at ~29.93
    assert media.frame_rate("1000/1", "1747/100") == Fraction(1747, 100)  # no standard rate near: the average
    assert media.frame_rate("25/1", "0/0") == 25  # a still picture
    with pytest.raises(media.MediaError):
        media.frame_rate("0/0", "0/0")


# --- long clips ----------------------------------------------------------------------------------------------------
def test_a_long_clip_is_cut_to_the_cap_and_says_so(clips, tmp_path):
    src = media.probe(clips.long)
    assert src.frames == 288 and src.has_audio
    common.work_clip(SimpleNamespace(input_path=clips.long), tmp_path / "work.mp4")
    w = media.probe(tmp_path / "work.mp4")
    assert w.frames == common.MAX_FRAMES and w.has_audio
    assert media.made_from(src, 0, w.frames) == "the first 4.0 s of 12.0 s"
    assert media.made_from(src, 48, w.frames) == "4.0 s from 2.0 s in, of 12.0 s"
    assert media.made_from(src, 0, src.frames) is None  # the whole shot: nothing to say


def test_a_take_from_part_of_a_shot_records_it(clips, tmp_path, monkeypatch):
    import pickups.app.store as st
    from pickups.app.jobs import JobManager, Stage, TaskSpec

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    sid = store.add_shot(clips.long, "Long walk")["id"]

    def fake(ctx):
        ctx.stage("Working")
        return {"label": "Fake", "task": "fake", "downloads": [], "details": common.size_details(64, 36, 97, 24.0)}

    jm = JobManager(store, SimpleNamespace(free_memory=lambda: None), {"fake": TaskSpec([Stage("Working", 1)], fake)})
    job = jm.submit(sid, "fake", {})
    for _ in range(200):
        if jm.jobs[job["id"]]["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert jm.jobs[job["id"]]["status"] == "done", jm.jobs[job["id"]].get("error")
    take = store.version(sid, jm.jobs[job["id"]]["result_version_id"])
    assert take["details"]["made_from"] == "the first 4.0 s of 12.0 s"


# --- blank shots, nobody in shot, tiny people ----------------------------------------------------------------------
def test_a_black_shot_stops_while_preparing(clips, tmp_path):
    ctx = SimpleNamespace(input_path=clips.black)
    common.work_clip(ctx, tmp_path / "work.mp4")
    with pytest.raises(RuntimeError, match="black all the way through"):
        common.proxy_clip(ctx, tmp_path / "work.mp4", tmp_path / "proxy.mp4")
    with pytest.raises(RuntimeError, match="one flat colour"):
        common.check_not_blank(clips.grey)


def test_a_dark_shot_is_not_blank(clips, tmp_path):
    ctx = SimpleNamespace(input_path=clips.dark)
    common.work_clip(ctx, tmp_path / "work.mp4")
    assert common.proxy_clip(ctx, tmp_path / "work.mp4", tmp_path / "proxy.mp4") == (1024, 576)
    # A small thing on a plain backdrop (a bird on a clear sky) is something to work on too.
    common.check_not_blank(_clip(tmp_path / "bird.mp4", "color=c=0x808080:size=64x36:rate=24,drawbox=x=30:y=15:w=3:h=3:"
                                 "color=white:t=fill", seconds=1))


class FakeSam:
    """Stands in for ComfyUI: single-frame selects find `found` (a mask or None), tracking finds `tracked`."""

    def __init__(self, found=None, tracked=None):
        self.found, self.tracked, self.graphs = found, tracked, []
        self.client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: str(p)))

    def run(self, graph):
        self.graphs.append(graph)
        tracking = any(k.startswith("PU_First") for k in graph)
        masks = ([self.tracked] * 3 if self.tracked is not None else []) if tracking else ([self.found] if self.found is not None else [])

        def out(i, m):
            return SimpleNamespace(name=f"mask_{i:05d}.png", to_file=lambda p: Image.fromarray(m.astype(np.uint8) * 255).save(p))
        return SimpleNamespace(get_outputs=lambda node: [out(i, m) for i, m in enumerate(masks)])

    def tracked_runs(self):
        return sum(any(k.startswith("PU_First") for k in g) for g in self.graphs)


@pytest.fixture()
def proxy(clips, tmp_path):
    ctx = SimpleNamespace(input_path=clips.dark)
    common.work_clip(ctx, tmp_path / "work.mp4")
    common.proxy_clip(ctx, tmp_path / "work.mp4", tmp_path / "proxy.mp4")
    run = tmp_path / "run"
    run.mkdir()
    return tmp_path / "proxy.mp4", run


def test_nobody_in_shot_fails_before_tracking(proxy):
    p, run = proxy
    sam = FakeSam()
    with pytest.raises(RuntimeError) as e:
        common.tracked_masks(sam, p, 0, [], "person", 64, run)
    assert str(e.value) == common.NOBODY
    assert sam.graphs and sam.tracked_runs() == 0  # single frames only: no tracking, no render


def test_nothing_found_is_worded_for_what_was_asked(proxy):
    p, run = proxy
    with pytest.raises(RuntimeError, match="Couldn't find “red umbrella” in this shot"):
        common.tracked_masks(FakeSam(), p, 0, [], "red umbrella", 64, run)
    with pytest.raises(RuntimeError, match="Nothing found where you clicked"):
        common.tracked_masks(FakeSam(), p, 3, [{"x": 30, "y": 20}], "", 64, run)
    with pytest.raises(RuntimeError, match="Click the person or thing in the picture first"):
        common.tracked_masks(FakeSam(), p, 0, [], "  ", 64, run)


def test_losing_the_person_while_tracking_is_worded(proxy):
    p, run = proxy
    found = np.zeros((576, 1024), bool)
    found[200:400, 500:560] = True
    with pytest.raises(RuntimeError) as e:
        common.tracked_masks(FakeSam(found, tracked=np.zeros_like(found)), p, 0, [{"x": 33, "y": 11}], "", 64, run)
    assert str(e.value) == common.LOST


def _person(h, shape=(576, 1024), frames=5):
    m = np.zeros(shape, bool)
    m[100:100 + h, 400:420] = True
    return [m.copy() for _ in range(frames)] + [np.zeros(shape, bool)] * 3  # lost on a few frames: ignored


def test_a_tiny_person_stops_before_the_render():
    assert common.person_height(_person(30)) == 30
    with pytest.raises(RuntimeError, match=r"too small in the frame for a good result \(about 56 px tall\)"):
        common.require_size(_person(30), 1080)  # 30 of 576 proxy px in a 1080 px tall frame
    common.require_size(_person(120), 1080)  # big enough: no complaint
    common.require_size(_person(common.MIN_PERSON_PX), 1080)
    with pytest.raises(RuntimeError, match="too small"):
        common.require_size(_person(40, shape=(1024, 576)), 1920)  # vertical: 40 of 1024
    with pytest.raises(RuntimeError, match="Couldn't follow"):
        common.require_size([np.zeros((576, 1024), bool)] * 4, 1080)


# --- files that aren't shots ---------------------------------------------------------------------------------------
def test_files_that_are_not_shots_are_refused_in_words(clips, tmp_path):
    d = tmp_path
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", d / "voice.m4a")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(d / "cover.png")
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-i", d / "cover.png", "-map", "0", "-map", "1",
        "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v", "attached_pic", d / "song.mp3")
    (d / "empty.mp4").write_bytes(b"")
    (d / "junk.mov").write_bytes(np.random.default_rng(1).integers(0, 255, 4096, np.uint8).tobytes())
    (d / "cut_off.mp4").write_bytes(clips.long.read_bytes()[:2048])
    Image.new("RGB", (320, 480), (90, 90, 90)).save(d / "sheet.png")
    _clip(d / "blink.mp4", "testsrc=size=64x36:rate=24", seconds=3 / 24)
    expect = {
        "voice.m4a": "sound but no picture",
        "song.mp3": "sound but no picture",  # its cover art is not a video
        "empty.mp4": "That file is empty",
        "junk.mov": "can't read that file as video",
        "cut_off.mp4": "can't read that file as video",
        "sheet.png": "still picture, not a video clip",
        "blink.mp4": "too short to work on (only 3 frames)",
    }
    for name, words in expect.items():
        with pytest.raises(media.MediaError, match=words.replace("(", r"\(").replace(")", r"\)")):
            media.check_shot(d / name)
    assert media.check_shot(clips.land).frames == 24


@pytest.fixture()
def client(tmp_path, monkeypatch):
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
    return TestClient(server.create_app("http://127.0.0.1:9")), lib  # a dead local address: no ComfyUI


def test_adding_a_bad_file_says_why_and_leaves_no_shot(client, tmp_path, clips):
    c, lib = client
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", tmp_path / "voice.m4a")
    r = c.post("/api/shots", files={"file": ("voice.m4a", (tmp_path / "voice.m4a").read_bytes(), "audio/mp4")})
    assert r.status_code == 400 and r.json()["detail"] == "That file has sound but no picture. Pick a video clip."
    r = c.post("/api/shots", files={"file": ("broken.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 400 and "ffmpeg" not in r.json()["detail"] and r.json()["detail"] == media.UNREADABLE
    Image.new("RGB", (320, 480)).save(tmp_path / "sheet.png")
    r = c.post("/api/shots/from_path", json={"path": str(tmp_path / "sheet.png")})
    assert r.status_code == 400 and "still picture" in r.json()["detail"]
    assert not list((lib / "shots").iterdir())  # nothing half-made in the library
    assert not list(lib.glob("tmp*"))  # and the upload is gone
    r = c.post("/api/shots", files={"file": ("phone.mp4", clips.rot.read_bytes(), "video/mp4")})
    assert r.status_code == 200 and (r.json()["width"], r.json()["height"]) == (180, 320)  # vertical stays vertical


def test_status_says_how_much_of_a_shot_the_models_use(client):
    c, _ = client
    assert c.get("/api/status").json()["max_frames"] == common.MAX_FRAMES == 97
