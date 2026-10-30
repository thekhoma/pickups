"""Settings audited against Lightricks' official LTX-2.5 workflows."""

from __future__ import annotations

from pickups import graphs, media


def _ltx_graphs() -> dict[str, dict]:
    return {
        "smoke": graphs.smoke_t2v("p"),
        "drive": graphs.drive("c.mp4", "f.png", "p", 24, 1664, 1152, control_mode="depth"),
        "masked_pose": graphs.masked_pose("b.mp4", "m.mp4", "p.mp4", "p", 24),
        "layout": graphs.layout_render("l.mp4", "look.png", "p", 24),
        "refine": graphs.refine_details("c.mp4", "p", 24, 1920, 1088),
        "remove": graphs.remove("c.mp4", "", 24),
        "retake": graphs.retake("c.mp4", 24, 1.0, 2.0, "p"),
        "retake2": graphs.retake_two_stage("c.mp4", 24, 1.0, 2.0, 25, 1920, 1088, "p"),
        "hdr": graphs.sdr_to_hdr("c.mp4", 24),
    }


def test_ltx_decodes_use_one_temporal_tile_per_shot():
    """64-frame tiles overlapped by 8 put a seam at frame 56 (a stall about 2 s in at 24 fps). Lightricks' own inpaint,
    tiled-fusion and Refine-Details graphs decode at 128 / 32, which holds a whole 97-frame shot in one tile."""
    for name, g in _ltx_graphs().items():
        decodes = [n for n in g.values() if n["class_type"] == "VAEDecodeTiled"]
        assert decodes, name
        for n in decodes:
            assert n["inputs"]["temporal_size"] >= 256, name
            assert n["inputs"]["temporal_overlap"] >= 32, name
            assert n["inputs"]["temporal_size"] > media.ltx_frames(241), name


def test_ltx_frames_down_is_the_largest_8k_plus_1():
    assert [media.ltx_frames_down(n) for n in (1, 9, 10, 16, 17, 57, 60, 64, 65, 97, 98)] == [1, 9, 9, 9, 17, 57, 57, 57, 65, 97, 97]
    assert all(media.ltx_frames_down(n) % 8 == 1 for n in range(1, 200))


def test_remove_on_a_moving_camera_sends_ltx_an_8k_plus_1_clip(tmp_path):
    """A 60-frame shot used to go to LTX as 60 frames; it comes back as 57, and the take froze on its 57th for the last 3."""
    import subprocess
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from pickups.tasks import remove

    work = tmp_path / "work.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=256x192:rate=24", "-frames:v", "60",
                    "-pix_fmt", "yuv420p", str(work)], check=True)
    pm = [np.zeros((48, 64), bool) for _ in range(60)]
    for m in pm:
        m[20:30, 25:40] = True
    seen = {}

    class Out:
        def __init__(self, name):
            self.name = name

        def to_file(self, path):
            Image.fromarray(np.zeros((192, 256, 3), np.uint8)).save(path)

    class Job:
        def get_outputs(self, node):
            return [Out("frame_00001.png")]

    class Ctx:
        class client:  # noqa: N801
            class assets:  # noqa: N801
                @staticmethod
                def from_file(path):
                    return Path(path).name

        def commercial_safe(self):
            return False

        def run(self, g):
            if "PU_Clip" in g:
                seen["ltx_clip"] = Path(tmp_path / "crop_ltx.mp4")
            return Job()

    remove.background(Ctx(), work, pm, None, (256, 192), 24.0, tmp_path, camera="moving", shadow=False)
    assert media.probe(seen["ltx_clip"]).frames == 57


def test_refine_details_guide_is_not_spatially_tiled_and_matches_the_sampler_window():
    """Lightricks' tiled-fusion graphs: use_tiled_encode off, use_streaming on, the same tile_frames on guide and sampler."""
    g = graphs.refine_details("c.mp4", "p", 24, 1920, 1088)
    guide, sampler = g["guide"]["inputs"], g["PU_Seed"]["inputs"]
    assert guide["use_tiled_encode"] is False
    assert guide["use_streaming"] is True
    assert guide["tile_frames"] == sampler["tile_frames"] == 97
    assert (sampler["tile_width"], sampler["tile_height"], sampler["overlap_frac"]) == (1024, 576, 0.5)
