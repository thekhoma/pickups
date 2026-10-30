"""Who's in this shot (POST /api/shots/{id}/people): the people SAM 3.1 finds, so Recast needs no click. A fake runner
stands in for ComfyUI and returns synthetic masks per frame."""

import io
import json
import re
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pickups import people
from pickups.tasks import common

W, H = 1024, 576  # what SAM sees for a 320x180 shot (media.work_size(320, 180, common.PROXY_LONG))
K = 320 / W       # proxy px -> source px


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def bracket(dx=0):
    """A person whose middle isn't on them: a "C" (the centre of its box is in the gap)."""
    return rect(100 + dx, 100, 300 + dx, 500) & ~rect(160 + dx, 180, 300 + dx, 420)


B = rect(700, 300, 760, 460)  # a smaller person on the right
SCENE = {
    0: [bracket()],
    5: [bracket(), B, rect(10, 10, 20, 20)],          # plus a speck
    10: [bracket(10), B, rect(705, 310, 755, 450)],   # plus B found twice
    15: [B],
}


class _Out:
    def __init__(self, name, mask):
        self.name, self.mask = name, mask

    def to_file(self, path):
        Image.fromarray(self.mask.astype(np.uint8) * 255).save(path)


class _Job:
    def __init__(self, masks):
        self.masks = masks

    def get_outputs(self, node):
        assert node == "PU_Mask"
        return [_Out(f"mask_{i + 1:05d}_.png", m) for i, m in enumerate(self.masks)]


class FakeRunner:
    """QuickRunner without ComfyUI: masks per frame from `scene` (the frame is read from the uploaded PNG's name)."""

    def __init__(self, scene=None, fail_on=None):
        self.scene = SCENE if scene is None else scene
        self.fail_on = fail_on  # a frame number, or "all"
        self.frames: list[int] = []
        self.graphs: list[dict] = []
        assets = type("Assets", (), {"from_file": staticmethod(lambda p: str(p))})
        self.client = type("Client", (), {"assets": assets})()

    def run(self, graph):
        self.graphs.append(graph)
        name = graph["PU_Frame"]["inputs"]["image"]
        frame = int(re.search(r"frame_(\d+)\.png$", name).group(1))
        self.frames.append(frame)
        with Image.open(name) as im:
            assert im.size == (W, H)  # SAM sees the frame at Recast's proxy size
        if self.fail_on == "all" or self.fail_on == frame:
            raise RuntimeError("SAM 3.1 isn't installed on this ComfyUI")
        return _Job(self.scene.get(frame, []))


def _clip(path, seconds=1.0, size="320x180"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(common, "MAX_FRAMES", 97)
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    runner = FakeRunner()
    monkeypatch.setattr(server, "QuickRunner", lambda conn: runner)
    client = TestClient(server.create_app("http://127.0.0.1:9"))  # a dead local address: never a real ComfyUI
    sid = store.add_shot(_clip(tmp_path / "hall.mov"), "Hall")["id"]
    return client, store, runner, sid


def _proxy(p):
    """A point in source px as Recast's SAM sees it (common.select_preview: x * proxy.width / source width)."""
    return int(p["y"] / K), int(p["x"] / K)


def test_finds_people_with_points_inside_and_the_largest_as_main(app):
    c, store, runner, sid = app
    r = c.post(f"/api/shots/{sid}/people", json={})
    assert r.status_code == 200, r.text
    res = r.json()
    assert runner.frames == [0, 5, 10, 15] and len(runner.graphs) <= 4  # 0, 1/4, 1/2, 3/4 of 21 frames
    # Each person as their own mask: SAM3_Detect with individual masks, or SAM 3.1's tracker run on the one frame
    # (graphs.select_frame picks either; both save one mask per person found).
    assert all(g["detect"]["inputs"]["individual_masks"] if "detect" in g else
               any(n["class_type"] == "SAM3_VideoTrack" for n in g.values()) for g in runner.graphs)
    # ComfyUI's SAM 3 text encoder returns only the best detection unless the prompt says how many ("person:8")
    assert all(g["PU_Label"]["inputs"]["text"] == "person:8" for g in runner.graphs)
    assert res["frame"] == 10  # two people on 5 and 10: the one nearest the middle
    assert res["probed"] == [0, 5, 10, 15] and res["window"] == 24 and (res["width"], res["height"]) == (320, 180)
    assert len(res["people"]) == 2  # the speck and the second find of B are gone
    a, b = res["people"]  # left to right
    assert [a["index"], b["index"]] == [0, 1]
    assert res["main"] == 0 and a["main"] and not b["main"]
    assert a["area"] > b["area"] > 0
    assert (a["seen"], b["seen"]) == (3, 3)
    masks = {0: bracket(10), 1: B}
    for i, p in enumerate(res["people"]):
        assert masks[i][_proxy(p["point"])], "the click point must be on the person"
        assert p["points"][0] == {"x": p["point"]["x"], "y": p["point"]["y"], "positive": True}
        assert all(q["positive"] and masks[i][_proxy(q)] for q in p["points"])
        x, y, w, h = p["box"]
        assert 0 <= x and 0 <= y and x + w <= 320 and y + h <= 180
    assert a["box"] == [round(110 * K), round(100 * K), round(200 * K), round(400 * K)]
    # the C's box centre is in its gap: the point is not just the middle of the box
    assert not bracket(10)[int((a["box"][1] + a["box"][3] / 2) / K), int((a["box"][0] + a["box"][2] / 2) / K)]


def test_thumbnails_and_mattes_are_served(app):
    c, store, runner, sid = app
    res = c.post(f"/api/shots/{sid}/people").json()  # no body at all is fine too
    for p in res["people"]:
        assert p["thumb_url"].startswith(f"/media/shots/{sid}/people/orig_10_")
        t = c.get(p["thumb_url"])
        assert t.status_code == 200
        with Image.open(io.BytesIO(t.content)) as im:
            assert im.format == "JPEG" and im.size == people.THUMB
        m = c.get(p["mask_url"])
        with Image.open(io.BytesIO(m.content)) as im:
            assert im.mode == "RGBA" and im.size == (W, H)
    assert not list((store.shot_dir(sid) / "people").glob("_work*"))  # frames and raw masks are cleaned up


def test_points_land_on_the_person_through_recasts_own_selection(app, tmp_path):
    """The form sends {frame, points} as Recast's params: common.select_preview scales them into the proxy exactly as
    a click, and they land inside the person SAM found."""
    c, store, runner, sid = app
    res = c.post(f"/api/shots/{sid}/people").json()
    main = res["people"][res["main"]]
    sent = []

    class Ctx:
        client = runner.client

        def run(self, graph):
            sent.append(graph)
            return _Job([])

    frame = tmp_path / "pick.png"
    Image.new("RGB", (W, H)).save(frame)
    common.select_preview(Ctx(), frame, main["points"], "", W / 320, tmp_path / "pick")
    coords = next(json.loads(n["inputs"]["positive_coords"]) for n in sent[0].values() if "positive_coords" in n.get("inputs", {}))
    assert len(coords) == len(main["points"]) >= 1
    assert all(bracket(10)[q["y"], q["x"]] for q in coords)


def test_cached_per_shot_and_take(app):
    c, store, runner, sid = app
    first = c.post(f"/api/shots/{sid}/people", json={}).json()
    assert first["cached"] is False and len(runner.frames) == 4
    again = c.post(f"/api/shots/{sid}/people", json={"version_id": "orig"}).json()
    assert again["cached"] is True and len(runner.frames) == 4  # no SAM call
    assert {k: v for k, v in again.items() if k != "cached"} == {k: v for k, v in first.items() if k != "cached"}
    fresh = c.post(f"/api/shots/{sid}/people", json={"refresh": True}).json()
    assert fresh["cached"] is False and len(runner.frames) == 8
    # a thumbnail gone from disk: looked for again
    thumb = store.shot_dir(sid) / "people" / fresh["people"][0]["thumb_url"].split("/")[-1].split("?")[0]
    thumb.unlink()
    assert c.post(f"/api/shots/{sid}/people").json()["cached"] is False and len(runner.frames) == 12


def test_a_take_has_its_own_people(app, tmp_path):
    c, store, runner, sid = app
    vid, vdir = store.new_version_dir(sid)
    _clip(vdir / "master.mov", 0.5)
    _clip(vdir / "video.mp4", 0.5)
    store.save_version(sid, vid, {"label": "Remove", "task": "remove", "input_version": "orig", "downloads": []})
    c.post(f"/api/shots/{sid}/people", json={})
    res = c.post(f"/api/shots/{sid}/people", json={"version_id": vid}).json()
    assert res["cached"] is False and res["window"] == 12 and res["probed"] == [0, 2, 4, 6]
    assert all(p["thumb_url"].startswith(f"/media/shots/{sid}/people/{vid}_") for p in res["people"])
    assert c.post(f"/api/shots/{sid}/people").json()["cached"] is True  # the original's people are still there


def test_frames_past_the_first_window_are_searched_too(app, monkeypatch):
    c, store, runner, sid = app
    monkeypatch.setattr(common, "MAX_FRAMES", 13)  # a shorter cap: 4 probes in its frames, then one every 6 frames
    res = c.post(f"/api/shots/{sid}/people").json()
    assert res["window"] == 24 and runner.frames == [0, 3, 6, 9, 13, 19]


def test_nobody_found(app):
    c, store, runner, sid = app
    runner.scene = {}
    res = c.post(f"/api/shots/{sid}/people").json()
    assert res["people"] == [] and res["main"] is None and res["frame"] == 10
    assert c.post(f"/api/shots/{sid}/people").json()["cached"] is True  # "no one" is an answer too


def test_main_is_the_one_on_screen_longest_not_one_big_frame(app):
    c, store, runner, sid = app
    big, steady = rect(100, 40, 400, 560), rect(600, 100, 800, 500)  # 15% on one frame vs 7% on all four
    runner.scene = {0: [steady], 5: [steady], 10: [big, steady], 15: [steady]}
    res = c.post(f"/api/shots/{sid}/people").json()
    assert res["frame"] == 10 and len(res["people"]) == 2
    assert res["people"][res["main"]]["seen"] == 4 and res["people"][res["main"]]["box"][0] == round(600 * K)


def test_sam_failing_is_a_quick_clear_error(app):
    c, store, runner, sid = app
    runner.fail_on = "all"
    r = c.post(f"/api/shots/{sid}/people")
    assert r.status_code == 502
    assert r.json()["detail"].startswith("Couldn't look for people in this shot") and "SAM 3.1" in r.json()["detail"]
    assert runner.frames == [0]  # gave up at once, not after four tries
    assert not (store.shot_dir(sid) / "people" / "orig.json").exists()


def test_a_later_failure_keeps_what_was_found_but_isnt_cached(app):
    c, store, runner, sid = app
    runner.fail_on = 10
    res = c.post(f"/api/shots/{sid}/people").json()
    assert res["probed"] == [0, 5] and res["frame"] == 5 and len(res["people"]) == 2 and res["partial"] is True
    runner.fail_on = None
    again = c.post(f"/api/shots/{sid}/people").json()
    assert again["cached"] is False and again["probed"] == [0, 5, 10, 15] and "partial" not in again


def test_unknown_shot_or_take(app):
    c, store, runner, sid = app
    assert c.post("/api/shots/shot_missing/people").status_code == 404
    assert c.post("/api/shots/..%2F..%2Fx/people").status_code in (404, 405)
    for bad in ("v_missing", "../..", "v_x/../.."):
        assert c.post(f"/api/shots/{sid}/people", json={"version_id": bad}).status_code == 404
    assert runner.frames == []


# --- the geometry -----------------------------------------------------------------------------------------------------

def test_window_and_probes(monkeypatch):
    monkeypatch.setattr(common, "MAX_FRAMES", 97)
    # as far as LTX Recast looks for its person (3 x MAX_FRAMES): someone who walks in late is found
    assert [people.window(n) for n in (1, 2, 5, 24, 96, 97, 98, 2400)] == [1, 2, 5, 24, 96, 97, 98, 291]
    assert people.probe_frames(97) == [0, 24, 48, 72]
    assert people.probe_frames(291) == [0, 24, 48, 72, 97, 145, 193, 241, 289]
    assert people.probe_frames(1) == [0] and people.probe_frames(2) == [0]


@pytest.mark.parametrize("mask", [
    bracket(),
    rect(100, 100, 400, 400) & ~rect(180, 180, 320, 320),  # a ring: its centre is a hole
    rect(500, 50, 520, 560),                               # a thin pole of a person, far away
    rect(0, 0, W, H),                                      # someone filling the frame
])
def test_points_are_always_on_the_mask(mask):
    pts = people.click_points(mask)
    assert 1 <= len(pts) <= 3
    assert all(mask[y, x] for x, y in pts)
    x, y, depth = people.inside_point(mask)
    assert mask[y, x] and depth >= 1


def test_a_standing_person_gets_head_chest_and_legs():
    figure = rect(480, 60, 540, 130) | rect(440, 130, 580, 330) | rect(450, 330, 500, 540) | rect(520, 330, 570, 540)
    pts = people.click_points(figure)
    assert len(pts) == 3 and all(figure[y, x] for x, y in pts)
    ys = sorted(y for _, y in pts)
    assert ys[0] < 130 and 130 <= ys[1] < 330 and ys[2] >= 330  # head, chest, a leg


@pytest.mark.parametrize("box", [(870, 101, 1695, 1080), (0, 0, 40, 90), (1880, 900, 1920, 1080), (0, 0, 1920, 1080)])
def test_thumbnail_window_stays_in_the_frame(box):
    """A 3:4 window, never past the frame's edges (a tall person is cropped a little, not shown between black bars)."""
    x0, y0, x1, y1 = people._crop_window(box, 0.75, (1920, 1080))
    assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
    assert abs((x1 - x0) / (y1 - y0) - 0.75) < 0.01


def test_nothing_to_point_at():
    assert people.click_points(np.zeros((H, W), bool)) == []
    assert people.inside_point(np.zeros((H, W), bool)) is None
