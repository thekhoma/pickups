"""Remove with real pixels on a moving camera (align.py, remove._aligned_fill) and the Clean-Plate fallback."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pickups import align, composite, graphs
from pickups.tasks import remove


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((h // 4 + 1, w // 4 + 1, 3)).astype(np.float32)
    big = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32)
    fine = rng.random((h, w, 1)).astype(np.float32) * 40
    return np.clip(big + fine - 20, 0, 255).astype(np.uint8)


def test_register_recovers_a_known_homography():
    g = align.gray(_texture(240, 320))
    H = np.array([[1.03, 0.01, 4.0], [-0.01, 1.02, 3.0], [1e-5, 0.0, 1.0]])
    mov = composite.warp(g, H, (320, 240))
    ok = (composite.warp(np.ones_like(g), H, (320, 240)) > 0.99).astype(np.float32)
    Hhat, err = align.register(g, mov, ok)
    inv = np.linalg.inv(H)
    inv /= inv[2, 2]
    assert err < 0.04
    assert np.abs(Hhat[:2, :2] - inv[:2, :2]).max() < 0.01 and np.abs(Hhat[:2, 2] - inv[:2, 2]).max() < 1.0


def test_a_panning_camera_shows_what_an_object_hid_and_remove_borrows_it():
    h, w, n = 120, 200, 9
    world = _texture(h + 20, w + 80, seed=3)
    crops, objs = [], []
    for i in range(n):
        x0 = 4 * i
        f = world[10:10 + h, x0:x0 + w].copy()
        m = np.zeros((h, w), bool)
        cx = 150 - 14 * i  # the object crosses the frame the other way: it uncovers the ground it covered
        m[50:90, cx:cx + 24] = True
        f[m] = (250, 20, 20)
        crops.append(f)
        objs.append(m)
    truth = [world[10:10 + h, 4 * i:4 * i + w] for i in range(n)]
    report: dict = {}
    out = remove._aligned_fill(crops, lambda i: objs[i], None, 0.1, False, lambda *a, **k: None, report)
    assert out is not None, report
    gen, hard, never, _ = out
    for i in (0, n // 2, n - 1):
        m = objs[i]
        err = np.abs(gen[i][m].astype(np.int16) - truth[i][m].astype(np.int16)).mean()
        assert err < 18, (i, err)
        assert not (gen[i][m] == (250, 20, 20)).all(axis=-1).any()
    assert report["never_seen_share"] < 0.2


def test_the_far_wall_is_not_taken_for_a_shadow():
    f = [np.full((100, 100, 3), 160, np.uint8) for _ in range(6)]
    for i, a in enumerate(f):
        a[10:40, 30 + 3 * i:60 + 3 * i] = 60  # a dark stripe that shifts about up high: a wall that doesn't line up
    objs = [np.zeros((100, 100), bool) for _ in f]
    for o in objs:
        o[50:90, 20:30] = True
    feet = composite.cast_shadows(f, objs, feet_frac=0.4)
    assert not feet[2][:45].any()


class _Out:
    def __init__(self, arr):
        self.arr = arr
        self.name = "frame_00001.png"

    def to_file(self, path):
        Image.fromarray(self.arr).save(path)


def _ctx(tmp_path, plates, have=True):
    class Job:
        def get_outputs(self, node):
            outs = []
            for k, a in enumerate(plates):
                o = _Out(a)
                o.name = f"frame_{k + 1:05d}.png"
                outs.append(o)
            return outs

    class Ctx:
        qa: dict = {}

        class client:  # noqa: N801
            class assets:  # noqa: N801
                @staticmethod
                def from_file(p):
                    return str(p)

        class conn:  # noqa: N801
            @staticmethod
            def model_files():
                return {graphs.CLEAN_PLATE_LORA} if have else set()

        def run(self, g):
            assert g["lora"]["inputs"]["lora_name"] == graphs.CLEAN_PLATE_LORA
            return Job()

    return Ctx()


def _clip(tmp_path):
    import subprocess

    work = tmp_path / "work.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24", "-frames:v", "9", "-pix_fmt", "yuv420p",
                    str(work)], check=True)
    from pickups.tasks import common

    return work, list(common.read_frames(work))


def test_clean_plate_is_used_only_when_the_thing_is_really_gone(tmp_path):
    work, crops = _clip(tmp_path)
    holes = [np.zeros((96, 128), bool) for _ in crops]
    for h in holes:
        h[30:60, 40:80] = True
    args = (work, crops, "crop=128:96:0:0", (128, 96), 9, 24.0, tmp_path, 7, "SUV", "", 1.0, lambda i: holes[i], None)
    # the render still shows the thing (identical to the shot): not gone, tried twice, then None for the inpaint fill
    same = [np.asarray(Image.fromarray(c).resize((1024, 768))) for c in crops]
    assert remove._clean_plate(_ctx(tmp_path, same), *args) is None
    # the render has it gone (a flat road over the hole): used
    gone = [np.full((768, 1024, 3), 90, np.uint8) for _ in crops]
    ctx = _ctx(tmp_path, gone)
    got = remove._clean_plate(ctx, *args)
    assert got is not None and len(got[0]) == 9
    assert ctx.qa["remove_clean_plate"][-1]["gone"] is True
    # no LoRA on the machine: nothing runs
    assert remove._clean_plate(_ctx(tmp_path, gone, have=False), *args) is None


def test_clean_plate_graph_follows_the_card():
    g = graphs.clean_plate_video("c.mp4", 24, 1024, 576, "An empty road, no vehicle.", negative="car")
    assert g["lora"]["inputs"]["lora_name"] == graphs.CLEAN_PLATE_LORA and g["lora"]["inputs"]["strength_model"] == 1.0
    assert g["guider"]["inputs"]["cfg"] == 1.0 and g["sampler"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert g["decode"]["inputs"]["temporal_size"] >= 256
