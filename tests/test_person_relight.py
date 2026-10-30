"""Person swap's optional Relight: the low-frequency blend, the per-shot brightness ratio, the graphs and the param."""

import hashlib
import json

import numpy as np

from pickups import composite, graphs
from pickups.tasks import person_swap as ps


def scene(h=96, w=96):
    rng = np.random.default_rng(1)
    paint = np.full((h, w, 3), 90, np.float32)
    paint += rng.normal(0, 12, (h, w, 1))  # fine detail (texture)
    mask = np.zeros((h, w), bool)
    mask[16:80, 30:66] = True
    return np.clip(paint, 0, 255).astype(np.uint8), mask


def relit_of(paint):
    """A relit frame: darker on the left, its own fine detail (a different texture, none of the paint's) on top."""
    h, w = paint.shape[:2]
    ramp = np.linspace(0.5, 1.2, w)[None, :, None]
    other = np.random.default_rng(9).normal(0, 25, (h, w, 1))
    return np.clip(np.full(paint.shape, 90.0, np.float32) * ramp + other, 0, 255).astype(np.uint8)


def test_only_the_low_frequency_shading_comes_from_the_relit_frame():
    paint, mask = scene()
    relit = relit_of(paint)
    lum = composite.relight_shading(paint, relit, mask)
    lp = composite.luma(paint)
    inside = composite.shrink(mask, 6)
    # shading changed: the left of the subject got darker than the right
    assert lum[inside][:, ].mean() != lp[inside].mean()
    left, right = inside.copy(), inside.copy()
    left[:, 48:] = False
    right[:, :48] = False
    assert (lum[left].mean() / lp[left].mean()) < (lum[right].mean() / lp[right].mean()) - 0.05
    # fine detail is the paint's: the high-pass of the result follows the paint's, not the relit frame's
    hp = lambda a: a - composite.blur(a, 3)  # noqa: E731
    r_paint = np.corrcoef(hp(lum)[inside], hp(lp)[inside])[0, 1]
    r_relit = np.corrcoef(hp(lum)[inside], hp(composite.luma(relit))[inside])[0, 1]
    assert r_paint > 0.9 and abs(r_relit) < 0.3


def test_nothing_changes_outside_the_mask_and_the_mask_is_tight():
    paint, mask = scene()
    relit = relit_of(paint)
    w = composite.relight_weight(mask)
    assert (w[~mask] == 0).all()                      # never beyond the mask
    assert w[composite.shrink(mask, 6)].min() > 0.99  # whole inside
    assert w[mask & ~composite.shrink(mask, 2)].max() < 1.0  # the rim is held back / feathered
    out, _ = composite.relight_paint(paint, relit, mask, paint, paint, mask)
    assert np.array_equal(out[~mask], paint[~mask])


def test_a_wide_mask_is_not_used():
    """A mask grown past the subject is not grown further: the weight is the subject's, shrunk."""
    paint, mask = scene()
    wide = composite.grow(mask, 12)
    assert composite.relight_weight(mask).sum() < mask.sum()
    assert composite.relight_weight(wide).sum() <= wide.sum()
    out, _ = composite.relight_paint(paint, relit_of(paint), mask, paint, paint, mask)
    assert np.array_equal(out[wide & ~mask], paint[wide & ~mask])  # the halo ring around the subject is untouched


def test_the_brightness_ratio_is_measured_per_shot():
    plate = np.full((64, 64, 3), 120, np.uint8)
    m = np.zeros((64, 64), bool)
    m[10:50, 20:40] = True
    dark = plate.copy()
    dark[m] = 120 * 0.846
    assert abs(composite.subject_ratio(dark, plate, m) - 0.846) < 0.01
    bright = plate.copy()
    bright[m] = 150
    assert abs(composite.subject_ratio(bright, plate, m) - 1.25) < 0.01
    assert composite.subject_ratio(dark, plate, np.zeros_like(m)) is None


def test_the_subject_lands_on_the_shots_ratio_to_the_plate():
    paint, mask = scene()
    plate = np.full(paint.shape, 140, np.uint8)
    for ratio in (0.846, 1.1):
        src = plate.copy()
        src[mask] = np.uint8(140 * ratio)
        out, info = composite.relight_paint(paint, relit_of(paint), mask, src, plate, mask)
        inner = composite.shrink(mask, 6)  # whole weight (the feathered rim blends back to the paint)
        want = ratio * 140
        assert abs(composite.luma(out)[inner].mean() - want) < 2.5
        assert abs(info["scene_ratio"] - ratio) < 0.01 and abs(info["target"] - want) < 1


def knight():
    """A bright steel figure with a red tabard and a small gold emblem on it."""
    paint, mask = scene()
    paint[mask] = 200  # steel: neutral and bright
    tab = np.zeros_like(mask)
    tab[38:74, 38:58] = True
    paint[tab] = (170, 20, 25)
    lion = np.zeros_like(mask)
    lion[42:48, 45:51] = True
    paint[lion] = (230, 180, 40)
    return paint, mask, tab, lion


def test_each_material_is_relit_on_its_own():
    paint, mask, tab, lion = knight()
    neutral, cloth, emblem = composite.material_masks(paint, mask)
    assert neutral[20, 33] and cloth[66, 42] and emblem[44, 47]
    plate = np.full(paint.shape, 120, np.uint8)
    src = plate.copy()
    src[mask] = 96  # the person sits at 0.8 of the plate
    flat = np.full(paint.shape, 110, np.uint8)  # a relit frame with even, overcast shading
    out, info = composite.relight_paint(paint, flat, mask, src, plate, mask)
    lum, lp = composite.luma(out), composite.luma(paint)
    steel = composite.shrink(mask, 6) & ~composite.grow(tab, 4)
    assert abs(lum[steel].mean() - 96) < 6  # the steel takes the shot's brightness
    core_tab = composite.shrink(tab, 5) & ~composite.grow(lion, 8)
    assert abs(lum[core_tab].mean() - lp[core_tab].mean()) < 4  # the tabard keeps its own level
    core_lion = composite.shrink(lion, 2)
    assert np.abs(out[core_lion].astype(int) - paint[core_lion]).max() <= 3  # the emblem is untouched
    assert info["materials"]["emblem"] > 0


def test_no_source_person_means_shading_only():
    paint, mask = scene()
    out, info = composite.relight_paint(paint, relit_of(paint), mask, paint, paint, np.zeros_like(mask))
    assert info["scene_ratio"] is None and "target" not in info
    assert out.shape == paint.shape and out.dtype == np.uint8


def sha(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


def test_graphs_without_relight_are_unchanged():
    """Hashes taken on `release` before Relight existed."""
    g = graphs.masked_pose("b.mp4", "l.mp4", "p.mp4", "a prompt", 24.0, seed=42, frames=97, prefix="pickups/person_pose",
                           region="r.mp4", negative=graphs.DEFAULT_NEGATIVE)
    assert sha(g) == "738b38d8a5eec4d713ac6a0ef789051d16fe4d43b66ca8cf1640a7472c3cc8a0"
    g = graphs.image_edit("a.png", "Replace the man", refs=["sheet.png"], prefix="pickups/recast_paint", commercial_safe=True,
                          size=(704, 1280))
    assert sha(g) == "262dba9f214ed48e8eaf2ec37d5ba88703724d555c9ca4ea626531a144b0093e"


def test_the_relight_graph_is_qwen_edit_2511_with_the_lightning_lora_and_two_pictures():
    g = graphs.relight("paint.png", "source.png", seed=7, size=(704, 1280))
    files = {n["inputs"].get(k) for n in g.values() for k in ("unet_name", "clip_name", "vae_name", "lora_name")}
    assert {"qwen_image_edit_2511_fp8mixed.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image_vae.safetensors",
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"} <= files
    ks = g["PU_Seed"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["seed"]) == (4, 1.0, 7)
    assert g["PU_Frame"]["inputs"]["image"] == "paint.png" and g["PU_Frame2"]["inputs"]["image"] == "source.png"
    assert "Picture 2" in g["PU_Instruction"]["inputs"]["prompt"] and "<image" not in g["PU_Instruction"]["inputs"]["prompt"]
    assert "PU_Plate" in g


def test_the_relight_param_is_off_unless_asked_for():
    assert not ps.wants_relight({}) and not ps.wants_relight({"relight": False}) and not ps.wants_relight({"relight": None})
    assert ps.wants_relight({"relight": True}) and ps.wants_relight({"relight": "true"})
    assert graphs.edit_model(True) == "Qwen-Image-Edit 2511"  # what the take's models list names when Relight runs
