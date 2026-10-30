"""Run 10 C: Replace a person keeps the way the person faces and never invents a walk.

R&D's Run 9 check: on a previz render the camera pushes in on a man standing with his back to it. His growing size
read as "walks towards the camera", which (1) cancelled the "seen from behind" the missing face had found, and (2) told
the painter "walking ... mid-stride" and LTX "walks towards the camera". The new man faced the camera.
"""

from pathlib import Path

import numpy as np

from pickups.tasks import recast_ltx as rl

H, W = 90, 160


def _figure(scale: float, legs: float = 0.18, cx: int = 80) -> np.ndarray:
    """A standing silhouette, `scale` of the frame's height, its legs `legs` of its height apart at the feet."""
    m = np.zeros((H, W), bool)
    h = int(H * scale)
    top = H - 4 - h
    bw = max(3, h // 5)
    m[top:top + int(0.67 * h), cx - bw // 2:cx + bw // 2 + 1] = True  # head and body
    spread = max(1, int(legs * h / 2))
    m[top + int(0.67 * h):top + h, cx - spread:cx - spread + 2] = True  # two legs
    m[top + int(0.67 * h):top + h, cx + spread - 2:cx + spread] = True
    return m


def _push_in(n=49):
    """The camera pushing in on someone standing still: they grow from 35% to 60% of the frame, legs as they were."""
    return [_figure(0.35 + 0.25 * i / (n - 1)) for i in range(n)]


def _walker(n=49):
    """A locked camera, someone walking across on the spot: the legs scissor (wide, narrow, wide)."""
    return [_figure(0.5, legs=0.05 + 0.3 * abs(np.sin(i * np.pi / 6))) for i in range(n)]


def _frames(n, shift_px):
    rng = np.random.default_rng(3)
    plate = rng.integers(0, 255, (H, W + n * shift_px + 8, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        f = plate[:, i * shift_px:i * shift_px + W].copy()
        if not shift_px:  # a locked camera still has grain
            f = np.clip(f.astype(int) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)
        out.append(f)
    return out


def test_a_moving_background_is_the_camera_and_a_still_one_is_not():
    masks = _push_in()
    assert rl.camera_moves(_frames(49, 4), masks)
    assert not rl.camera_moves(_frames(49, 0), masks)


def test_a_stride_is_the_legs_never_the_size():
    assert rl.strides(_walker())
    assert not rl.strides(_push_in())


def test_a_push_in_on_someone_standing_is_never_walking():
    masks = _push_in()
    assert rl.motion_words(masks) == "walks towards the camera"  # a locked camera: today's reading
    assert rl.motion_words(masks, moving_camera=True) == rl.STAND
    assert rl.motion_words(_walker(), moving_camera=True).startswith("walks")
    instruction = rl.paint_instruction("person", "in the middle of", rl.motion_words(masks, moving_camera=True).startswith("walks"),
                                       True, "Elderly man")
    assert "walking" not in instruction and "mid-stride" not in instruction


def test_back_to_the_camera_stays_back_to_the_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(rl, "faces_on", lambda ctx, png, run_dir: [])  # faces looked for, none found: seen from behind
    monkeypatch.setattr(rl, "sheet_view", lambda path, which: None)  # a sheet whose back view couldn't be cut
    track = _push_in()
    base = rl.paint_instruction("person", "in the middle of", False, True, "Elderly man")
    # Before Run 10 the push-in cancelled "back" (walks_away towards) and nothing was said.
    view_old, said_old, _ = rl.paint_view(None, Path("a.png"), track[24], track, Path("sheet.png"), 2, base, tmp_path)
    assert view_old == "" and said_old == base
    view, said, _ = rl.paint_view(None, Path("a.png"), track[24], track, Path("sheet.png"), 2, base, tmp_path, moving_camera=True)
    assert view == "back"
    assert "seen from behind, facing away" in said and "no face showing" in said
    prompt = rl.ltx_prompt("Elderly man, an elderly man with a white beard", rl.STAND, "", view=view)
    assert prompt.startswith("Elderly man, an elderly man with a white beard moves naturally where they stand, facing away from "
                             "the camera, back to the camera.")
    assert "walk" not in prompt and "towards the camera" not in prompt


def test_facing_words_say_only_what_the_frame_showed():
    # Run 11: the plain words, the same in the paint and the video prompt.
    assert rl.facing_words("back") == ", facing away from the camera, back to the camera"
    assert rl.facing_words("back-left") == ", facing away from the camera, back to the camera, turned to the left"
    assert rl.facing_words("front") == ", facing the camera"
    assert rl.facing_words("side-right") == ", side on, in profile, facing right"
    assert rl.facing_words("") == ""
    assert "back" not in rl.ltx_prompt("The new SUV", "drives steadily", "", vehicle=True, view="back")
