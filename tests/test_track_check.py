"""Check tracking: the cache key (stable for the same selection, new for any change), the health flags on synthetic
masks, the render reusing a checked selection, and the endpoint end to end with a fake ComfyUI."""

import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pickups import tracking
from pickups.tasks import common

# --- the cache key -------------------------------------------------------------------------------------------------


@pytest.fixture()
def source(tmp_path):
    f = tmp_path / "source.mov"
    f.write_bytes(b"x" * 100)
    return f


def key(source, **over):
    a = {"shot_id": "shot_a", "version_id": None, "frame": 39, "points": [{"x": 10, "y": 20}], "text": "man in green parka",
         "source": source, "proxy_frames": 291, "proxy_size": (1024, 576)}
    a.update(over)
    return common.track_key(**a)


def test_the_key_is_the_same_for_the_same_selection(source):
    k = key(source)
    assert k == key(source)
    assert k == key(source, version_id="orig")  # the original, however it's named
    assert k == key(source, points=[{"x": 10.0, "y": 20.0, "positive": True}])  # ints or floats, positive by default
    assert k == key(source, text="  man in green parka ")


@pytest.mark.parametrize("change", [
    {"shot_id": "shot_b"}, {"version_id": "v_0123456789"}, {"frame": 40}, {"points": [{"x": 11, "y": 20}]},
    {"points": [{"x": 10, "y": 20, "positive": False}]}, {"points": []}, {"text": "child"}, {"proxy_frames": 97},
    {"proxy_size": (1024, 544)},
])
def test_any_change_to_the_selection_or_proxy_is_a_new_key(source, change):
    assert key(source, **change) != key(source)


def test_new_footage_is_a_new_key(source):
    k = key(source)
    source.write_bytes(b"y" * 101)  # replaced (size)
    k2 = key(source)
    st = source.stat()
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # touched (mtime)
    assert len({k, k2, key(source)}) == 3


def test_masks_round_trip_and_a_wrong_shape_is_a_miss(tmp_path):
    rng = np.random.default_rng(0)
    masks = [rng.random((36, 64)) > 0.5 for _ in range(5)]
    f = tmp_path / "track_x.npz"
    common.save_track(f, masks)
    back = common.load_track(f, (5, 36, 64))
    assert back is not None and all(np.array_equal(a, b) for a, b in zip(masks, back))
    assert common.load_track(f, (6, 36, 64)) is None and common.load_track(f, (5, 64, 36)) is None
    assert common.load_track(tmp_path / "none.npz", (5, 36, 64)) is None
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert common.load_track(tmp_path / "bad.npz", (5, 36, 64)) is None


# --- health --------------------------------------------------------------------------------------------------------


def person(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_a_sudden_drop_and_empty_frames_are_flagged():
    masks = [person(100, 100, 20, 10, 60, 90) for _ in range(20)]  # 40 x 80
    masks[6] = person(100, 100, 20, 10, 60, 50)  # half of them for one frame: the bag, an arm, lost
    masks[12] = np.zeros((100, 100), bool)  # gone for a frame
    h = tracking.health(masks)
    assert h["flagged"] == [6, 12] and h["empty"] == [12]
    assert len(h["area"]) == 20 and h["area"][0] == pytest.approx(0.32) and h["area"][12] == 0


def test_a_steady_track_and_slow_changes_are_not_flagged():
    steady = [person(100, 100, 20, 10, 60, 90) for _ in range(12)]
    assert tracking.health(steady)["flagged"] == [] and tracking.health(steady)["empty"] == []
    # Walking away: 5% smaller each frame is never a 35% drop against the frames around it.
    shrinking, w = [], 60.0
    for _ in range(15):
        shrinking.append(person(100, 100, 0, 0, int(w), 80))
        w *= 0.95
    assert tracking.health(shrinking)["flagged"] == []
    # 30% smaller for one frame is under the line; 40% is over it.
    m = [person(100, 100, 0, 0, 50, 100) for _ in range(9)]
    m[4] = person(100, 100, 0, 0, 35, 100)
    assert tracking.health(m)["flagged"] == []
    m[4] = person(100, 100, 0, 0, 30, 100)
    assert tracking.health(m)["flagged"] == [4]


def test_walking_in_or_out_through_the_edge_is_not_losing_them():
    blank = np.zeros((50, 50), bool)
    walk_in = [blank] * 4 + [person(50, 50, 0, 10, 8, 40), person(50, 50, 0, 10, 16, 40)] + [person(50, 50, 0, 10, 24, 40)] * 4
    h = tracking.health(walk_in)
    assert h["empty"] == [0, 1, 2, 3] and h["flagged"] == []
    assert tracking.health(walk_in[::-1])["flagged"] == []  # walking out
    # Found again in the middle of the picture after frames of nothing: that's the tracker, not the person.
    pops = [blank] * 4 + [person(50, 50, 10, 10, 30, 40)] * 6
    assert tracking.health(pops)["flagged"] == [3]
    lost = [person(50, 50, 10, 10, 30, 40)] * 6 + [blank] * 4
    assert tracking.health(lost)["flagged"] == [6] and tracking.health(lost)["empty"] == [6, 7, 8, 9]


# --- the render reuses a checked selection -------------------------------------------------------------------------


def _clip(path, seconds=1.0, size="64x36"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24", "-t", str(seconds),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    root = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", root)
    monkeypatch.setattr(server, "ROOT", root)
    monkeypatch.setattr(tracking, "media_url", lambda p: "/media/" + p.relative_to(root).as_posix())
    store = st.Store(root)
    sid = store.add_shot(_clip(tmp_path / "walk.mp4"), "Walk")["id"]
    return store, sid


def _ctx(store, sid, version=None):
    return SimpleNamespace(store=store, shot_id=sid, input_path=store.version_master(sid, version), job={"version_id": version},
                           qa={})


def test_the_render_uses_the_checked_masks_and_says_so(lib, tmp_path, monkeypatch):
    store, sid = lib
    _, proxy = tracking.proxy_for(store, sid, None)
    p = common.media.probe(proxy)
    checked = [person(p.height, p.width, 100, 50, 300, 500)] * p.frames
    k = common.track_key(sid, None, 5, [{"x": 30, "y": 20}], "", store.source_path(sid), p.frames, (p.width, p.height))
    common.save_track(store.shot_dir(sid) / "picks" / f"track_{k}.npz", checked)
    calls = []
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **kw: calls.append(a) or [np.zeros((p.height, p.width), bool)] * p.frames)

    # The render cuts its own proxy (a new file) from the same footage: same key.
    render_proxy = tmp_path / "proxy.mp4"
    common.media.prepare_work_clip(store.source_path(sid), render_proxy, 0, p.frames, (p.width, p.height))
    ctx = _ctx(store, sid)
    got = common.checked_masks(ctx, render_proxy, 5, [{"x": 30, "y": 20, "positive": True}], "", 64, tmp_path)
    assert not calls and np.array_equal(got[0], checked[0]) and ctx.qa == {"tracking_from_check": True}

    # Another click: tracked as usual, and kept for next time.
    ctx = _ctx(store, sid)
    common.checked_masks(ctx, render_proxy, 5, [{"x": 31, "y": 20}], "", 64, tmp_path)
    assert len(calls) == 1 and ctx.qa == {"tracking_from_check": False}
    ctx = _ctx(store, sid)
    common.checked_masks(ctx, render_proxy, 5, [{"x": 31, "y": 20}], "", 64, tmp_path)
    assert len(calls) == 1 and ctx.qa == {"tracking_from_check": True}

    # Recast anyone: only when everyone was checked.
    swaps = [{"frame": 5, "points": [{"x": 30, "y": 20}], "text": ""}, {"frame": 5, "points": [{"x": 31, "y": 20}], "text": ""}]
    ctx = _ctx(store, sid)
    both = common.checked_tracks(ctx, render_proxy, swaps)
    assert both is not None and len(both) == 2 and ctx.qa == {"tracking_from_check": True}
    ctx = _ctx(store, sid)
    assert common.checked_tracks(ctx, render_proxy, swaps + [{"frame": 5, "points": [], "text": "dog"}]) is None
    assert ctx.qa == {"tracking_from_check": False}


def test_no_shot_no_cache(tmp_path, monkeypatch):
    proxy = _clip(tmp_path / "p.mp4")
    monkeypatch.setattr(common, "tracked_masks", lambda *a, **kw: ["tracked"])
    ctx = SimpleNamespace(input_path=proxy)  # a task run outside a shot (and the tests' bare contexts)
    assert common.checked_masks(ctx, proxy, 0, [], "person", 64, tmp_path) == ["tracked"]


# --- the endpoint, with a fake ComfyUI -----------------------------------------------------------------------------


class FakeSam:
    """Stands in for ComfyUI: single-frame selects find `found`, tracking follows it for 3 frames (then loses it)."""

    def __init__(self, found):
        self.found, self.graphs = found, []
        self.client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: str(p)))

    def run(self, graph):
        self.graphs.append(graph)
        masks = [self.found] * 3 if any(k.startswith("PU_First") for k in graph) else [self.found]

        def out(i, m):
            return SimpleNamespace(name=f"mask_{i:05d}.png", to_file=lambda p: Image.fromarray(m.astype(np.uint8) * 255).save(p))
        return SimpleNamespace(get_outputs=lambda node: [out(i, m) for i, m in enumerate(masks)])


@pytest.fixture()
def api(lib, tmp_path, monkeypatch):
    import pickups.app.server as server
    store, sid = lib
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(server, "Store", lambda: store)
    sam = FakeSam(person(576, 1024, 400, 100, 600, 500))
    monkeypatch.setattr(server, "QuickRunner", lambda conn: sam)
    return TestClient(server.create_app("http://127.0.0.1:9")), store, sid, sam


def test_the_endpoint_tracks_once_then_reuses_it(api):
    c, store, sid, sam = api
    body = {"frame": 0, "points": [{"x": 30, "y": 18, "positive": True}], "text": ""}
    r = c.post(f"/api/shots/{sid}/track", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["frames"] == 24 and res["fps"] == 24 and res["cached"] is False and len(res["area"]) == 24
    assert res["flagged"] == [3] and res["empty"] == list(range(3, 24))  # the fake loses them after 3 frames
    picks = store.shot_dir(sid) / "picks"
    video = picks / f"track_{res['cache_key']}.mp4"
    assert res["video_url"].endswith(video.name) and (picks / f"track_{res['cache_key']}.npz").exists()
    p = common.media.probe(video)
    assert p.width == 960 and p.height == 540 and p.frames == 24
    # Amber at 45% over the selection (x 400-600, y 100-500 on the 1024 px proxy), the picture as it is elsewhere.
    proxy = next(picks.glob("track_proxy_orig_*.mp4"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(proxy), "-vf", "select=eq(n\\,1),scale=960:540", "-frames:v", "1",
                    str(picks / "plain.png")], check=True)
    plain = np.asarray(Image.open(picks / "plain.png").convert("RGB")).astype(float)
    shown = np.asarray(Image.open(common._frame_png(video, 1, picks / "f.png")).convert("RGB")).astype(float)
    inside, outside = (slice(120, 440), slice(400, 540)), (slice(0, 80), slice(0, 300))
    want = plain[inside] * 0.55 + np.array([245, 184, 61]) * 0.45
    assert np.abs(shown[inside] - want).mean() < 10 and np.abs(shown[inside] - plain[inside]).mean() > 30
    assert np.abs(shown[outside] - plain[outside]).mean() < 10

    ran = len(sam.graphs)
    again = c.post(f"/api/shots/{sid}/track", json=body).json()
    assert again["cached"] is True and again["cache_key"] == res["cache_key"] and len(sam.graphs) == ran
    other = c.post(f"/api/shots/{sid}/track", json={**body, "text": "person"}).json()
    assert other["cache_key"] != res["cache_key"] and len(sam.graphs) > ran


def test_the_endpoint_words_its_errors(api):
    c, _, sid, sam = api
    assert c.post("/api/shots/shot_0000000000/track", json={"frame": 0, "text": "man"}).status_code == 404
    r = c.post(f"/api/shots/{sid}/track", json={"frame": 0, "points": [], "text": ""})
    assert r.status_code == 422 and "Click the person" in r.json()["detail"]
    sam.found = np.zeros((576, 1024), bool)
    r = c.post(f"/api/shots/{sid}/track", json={"frame": 0, "points": [{"x": 3, "y": 3}]})
    assert r.status_code == 422 and "Nothing found where you clicked" in r.json()["detail"]
    assert c.post(f"/api/shots/{sid}/track", json={"frame": 0, "points": [{"x": "a", "y": 1}]}).status_code == 400


def test_the_take_records_whether_its_masks_came_from_a_check(lib):
    import time

    from pickups.app.jobs import JobManager, Stage, TaskSpec
    store, sid = lib

    def fake(ctx):
        ctx.stage("Working")
        ctx.qa["tracking_from_check"] = True  # as common.checked_masks does
        return {"label": "Fake", "task": "fake", "downloads": [], "details": {"_qa": {"anchor_frame": 8}}}

    jm = JobManager(store, SimpleNamespace(free_memory=lambda: None), {"fake": TaskSpec([Stage("Working", 1)], fake)})
    job = jm.submit(sid, "fake", {})
    for _ in range(200):
        if jm.jobs[job["id"]]["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert jm.jobs[job["id"]]["status"] == "done", jm.jobs[job["id"]].get("error")
    take = store.version(sid, jm.jobs[job["id"]]["result_version_id"])
    assert take["details"]["_qa"] == {"anchor_frame": 8, "tracking_from_check": True}


# --- other people stay footage -------------------------------------------------------------------------------------


def test_a_touching_second_person_stays_footage():
    """The mudroom: the parent's hands on the child's jacket. The parent's grown matte and LTX's new character bleed
    onto the child; with the child protected, every child pixel is the footage and the parent is still the render."""
    from pickups import composite
    from pickups.tasks import recast_ltx

    h, w = 120, 160
    parent = person(h, w, 80, 10, 130, 115)
    parent[50:60, 60:80] = True  # hands reaching onto the child
    child = person(h, w, 30, 40, 80, 115) & ~parent
    new = composite.grow(parent, 8)  # the new character drawn a little bigger: over the child where they touch
    matte = composite.soft_matte(parent | new, 4, 3.0)
    assert (matte[child] > 0.5).mean() > 0.2  # unprotected, a good part of the child would be pasted
    rng = np.random.default_rng(1)
    footage = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    render = 255 - footage
    protect = recast_ltx.protect_others(child, parent, 1.5)
    out, _ = composite.paste(footage, render, matte * (1 - protect))
    assert np.array_equal(out[child], footage[child])
    inner = composite.shrink(parent, 2)
    assert np.array_equal(out[inner], render[inner])  # the hands on the jacket are the parent's: still recast
    assert recast_ltx.protect_others(np.zeros_like(child), parent, 1.5) is None
    assert recast_ltx.protect_others(parent, parent, 1.5) is None  # someone else is never the person


class PeopleSam:
    """Stands in for ComfyUI on the mudroom: "person" finds the parent and the child, a click finds the parent, bags
    aren't there; tracking follows each object as it started for 3 frames."""

    def __init__(self, parent, child):
        self.parent, self.child, self.graphs = parent, child, []
        self.client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: str(p)))

    def run(self, graph):
        self.graphs.append(graph)
        firsts = [v["inputs"]["image"] for k, v in graph.items() if k.startswith("PU_First")]
        if firsts:
            objs = [np.asarray(Image.open(f).convert("L")) > 127 for f in firsts]
            per = {"PU_Mask": [np.logical_or.reduce(objs)] * 3, **{f"PU_Obj{j}": [m] * 3 for j, m in enumerate(objs)}}
        else:
            label = graph.get("PU_Label", {}).get("inputs", {}).get("text", "")
            individual = any(v.get("inputs", {}).get("individual_masks") for v in graph.values())
            if label.startswith("person"):
                masks = [self.parent, self.child] if individual else [self.parent | self.child]
            else:
                masks = [] if label else [self.parent]
            per = {"PU_Mask": masks}

        def out(i, m):
            return SimpleNamespace(name=f"mask_{i:05d}.png", to_file=lambda p: Image.fromarray(m.astype(np.uint8) * 255).save(p))
        return SimpleNamespace(get_outputs=lambda node: [out(i, m) for i, m in enumerate(per.get(node, []))])


def test_tracked_masks_tracks_the_others_with_them(lib, tmp_path):
    store, sid = lib
    _, proxy = tracking.proxy_for(store, sid, None)
    parent = person(576, 1024, 500, 100, 760, 560)
    child = person(576, 1024, 300, 250, 520, 560)  # overlaps the parent a little (the hands)
    sam = PeopleSam(parent, child)
    others: list = []
    run = tmp_path / "run"
    run.mkdir()
    masks = common.tracked_masks(sam, proxy, 0, [{"x": 40, "y": 15}], "", 64, run, others_out=others)
    assert len(masks) == len(others) == 24
    assert np.array_equal(masks[0], parent) and np.array_equal(others[0], child & ~parent) and not (others[1] & masks[1]).any()
    assert not others[5].any()  # the fake tracking stops after 3 frames
    tracks = [g for g in sam.graphs if "PU_First1" in g]
    assert tracks and all("PU_Obj1" in g for g in tracks)  # tracked together, each on its own
    # Without others_out: the selection alone, as before.
    sam.graphs.clear()
    common.tracked_masks(sam, proxy, 0, [{"x": 40, "y": 15}], "", 64, run)
    assert not any("PU_First1" in g for g in sam.graphs)
    # Cached together: a check's masks serve a render that wants the others; masks kept without them don't.
    f = tmp_path / "t.npz"
    common.save_track(f, masks, others)
    back: list = []
    assert common.load_track(f, (24, 576, 1024), others_out=back) is not None and np.array_equal(back[0], others[0])
    common.save_track(f, masks)
    assert common.load_track(f, (24, 576, 1024), others_out=[]) is None and common.load_track(f, (24, 576, 1024)) is not None


def test_recast_anyone_keeps_only_the_unswapped_as_others(lib, tmp_path):
    store, sid = lib
    _, proxy = tracking.proxy_for(store, sid, None)
    p = common.media.probe(proxy)
    a, b, c = (person(p.height, p.width, x, 100, x + 150, 500) for x in (50, 400, 750))
    src = store.source_path(sid)
    swaps = [{"frame": 3, "points": [{"x": 10, "y": 18}], "text": ""}, {"frame": 3, "points": [{"x": 30, "y": 18}], "text": ""}]
    for s, me, rest in ((swaps[0], a, b | c), (swaps[1], b, a | c)):  # each person's own check: everyone else is "other"
        k = common.track_key(sid, None, 3, s["points"], "", src, p.frames, (p.width, p.height))
        common.save_track(store.shot_dir(sid) / "picks" / f"track_{k}.npz", [me] * p.frames, [rest] * p.frames)
    others: list = []
    tracks = common.checked_tracks(_ctx(store, sid), proxy, swaps, others_out=others)
    assert tracks is not None and len(others) == p.frames and np.array_equal(others[0], c)  # the swapped are never "other"


def test_a_change_to_what_people_carry_is_a_new_key(source, monkeypatch):
    k = key(source)
    monkeypatch.setattr(common, "CARRIED", common.CARRIED + ("kite",))
    assert key(source) != k  # masks tracked before the phone went with the person are not reused


def test_a_phone_in_the_hand_goes_with_the_person(monkeypatch, tmp_path):
    """The castle-hall woman's mask left out the phone in her hand: it stayed as footage by the new character's hands."""
    person = np.zeros((100, 80), bool)
    person[10:95, 30:50] = True
    in_hand = np.zeros_like(person)
    in_hand[50:58, 50:56] = True  # touching her hand
    elsewhere = np.zeros_like(person)
    elsewhere[5:12, 0:6] = True  # on a ledge across the room
    found = {"phone": in_hand | elsewhere}
    monkeypatch.setattr(common, "select_preview", lambda ctx, png, pts, thing, scale, run: found.get(thing, np.zeros_like(person)))
    out = common.with_carried(None, tmp_path / "f.png", person, "woman, handbag", 1.0, tmp_path)
    assert out[in_hand].all() and not out[elsewhere].any()
    assert "phone" in common.CARRIED
