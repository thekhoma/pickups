"""Reference prep for Recast: one front-view figure and a clean mask, without a SAM label, from a plain-backdrop still,
a contact sheet or a transparent PNG; busy photos are left to SAM (None)."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pickups import reference as R

SKIN, HAIR = (205, 160, 130), (45, 35, 30)


def backdrop(w: int, h: int, seed: int = 0, base: float = 0.0) -> np.ndarray:
    """A light-grey studio sweep: darker at the top, a lighter floor, a vignette, a little noise."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    v = 172 + 40 * yy / h - 18 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) + base
    img = np.stack([v + 4, v, v - 4], -1) + np.random.default_rng(seed).normal(0, 1.2, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def person(img: np.ndarray, cx: int, top: int, tall: int, torso, view: str = "front", shadow: bool = True) -> np.ndarray:
    """Draw a simple full-length figure on img (in place) and return its mask. view: front, back or profile."""
    h, w, _ = img.shape
    im = Image.fromarray(img)
    if shadow:  # soft contact shadow on the floor, wider than the feet
        sh = Image.new("L", (w, h), 0)
        ImageDraw.Draw(sh).ellipse([cx - 0.28 * tall, top + 0.95 * tall, cx + 0.34 * tall, top + 1.02 * tall], fill=255)
        a = np.asarray(sh.filter(ImageFilter.GaussianBlur(tall * 0.01)), np.float32)[..., None] / 255
        im = Image.fromarray((np.asarray(im, np.float32) * (1 - 0.22 * a)).astype(np.uint8))
    mask = Image.new("L", (w, h), 0)
    d, m = ImageDraw.Draw(im), ImageDraw.Draw(mask)

    def part(box, fill, kind="rect"):
        (d.ellipse if kind == "ellipse" else d.rectangle)(box, fill=fill)
        (m.ellipse if kind == "ellipse" else m.rectangle)(box, fill=255)

    t = tall
    hw = 0.075 * t
    if view == "profile":
        part([cx - hw, top, cx + hw, top + 0.15 * t], SKIN, "ellipse")
        part([cx - hw, top, cx + 0.2 * hw, top + 0.15 * t], HAIR, "ellipse")  # hair at the back, face at the front
        part([cx + hw * 0.8, top + 0.06 * t, cx + hw * 1.5, top + 0.09 * t], SKIN)  # nose
    else:
        part([cx - hw, top, cx + hw, top + 0.15 * t], HAIR if view == "back" else SKIN, "ellipse")
        if view == "front":
            part([cx - hw, top, cx + hw, top + 0.05 * t], HAIR, "ellipse")
            for ex in (-0.035, 0.035):
                d.ellipse([cx + ex * t - 4, top + 0.065 * t - 3, cx + ex * t + 4, top + 0.065 * t + 3], fill=(30, 30, 30))
    bw = 0.12 * t if view == "profile" else 0.15 * t
    part([cx - bw, top + 0.15 * t, cx + bw, top + 0.52 * t], torso)
    if view != "profile":
        part([cx - bw - 0.09 * t, top + 0.15 * t, cx + bw + 0.09 * t, top + 0.2 * t], torso)  # shoulders
        for s in (-1, 1):  # arms, slightly away from the body
            x0 = cx + s * (bw + 0.03 * t)
            part([min(x0, x0 + s * 0.07 * t), top + 0.17 * t, max(x0, x0 + s * 0.07 * t), top + 0.5 * t], torso)
    part([cx - bw, top + 0.45 * t, cx + bw, top + 0.49 * t], (176, 174, 171))  # a grey belt, close to the backdrop
    for s in (-1, 1):  # legs and shoes
        part([min(cx + s * 0.02 * t, cx + s * 0.12 * t), top + 0.52 * t, max(cx + s * 0.02 * t, cx + s * 0.12 * t), top + 0.95 * t],
             (50, 52, 60))
        part([min(cx + s * 0.02 * t, cx + s * 0.13 * t), top + 0.93 * t, max(cx + s * 0.02 * t, cx + s * 0.13 * t), top + 0.98 * t],
             (25, 20, 18))
    img[:] = np.asarray(im)
    return np.asarray(mask) > 127


def in_crop(gt: np.ndarray, ref: R.Reference) -> np.ndarray:
    """The ground-truth mask, cut and scaled the way the reference was."""
    x0, y0, x1, y1 = ref.box
    canvas = np.zeros((y1 - y0, x1 - x0), bool)
    h, w = gt.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gt[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(canvas.astype(np.uint8) * 255).resize(ref.image.size, Image.BILINEAR)) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def test_a_still_on_a_plain_backdrop_is_cut_out_without_its_shadow():
    img = backdrop(640, 960)
    gt = person(img, 320, 90, 760, (170, 40, 40))
    ref = R.prepare(Image.fromarray(img), aspect=0.75)
    assert ref is not None and ref.path == "figure cut out from its plain backdrop" and ref.views == 1
    assert abs(ref.image.width / ref.image.height - 0.75) < 0.01  # SCAIL-2's centre crop keeps the whole figure
    assert ref.mask.shape == (ref.image.height, ref.image.width)
    assert iou(ref.mask, in_crop(gt, ref)) > 0.93  # the grey belt (close to the backdrop) included, the shadow not
    assert R.label(ref.mask)[1] == 1


def test_a_three_view_sheet_keeps_the_front_view():
    img = backdrop(1500, 800, seed=1)
    gts = [person(img, x, 70, 650, c, v) for x, c, v in ((260, (170, 40, 40), "front"), (750, (40, 140, 60), "profile"),
                                                          (1240, (40, 60, 160), "back"))]
    ref = R.prepare(img, aspect=0.6)
    assert ref is not None and ref.path == "front view cropped from a 3-view sheet" and ref.views == 3
    assert iou(ref.mask, in_crop(gts[0], ref)) > 0.9
    shown = np.asarray(ref.image)
    assert not in_crop(gts[1], ref).any() and not in_crop(gts[2], ref).any()  # only one figure in the picture
    torso = shown[ref.mask].astype(int)
    assert (torso[:, 0] > torso[:, 2] + 20).mean() > 0.3  # the red front view, not the blue back


def test_a_profile_on_the_left_loses_to_the_front_view():
    img = backdrop(1500, 800, seed=2)
    person(img, 260, 70, 650, (40, 140, 60), "profile")
    front = person(img, 750, 70, 650, (170, 40, 40), "front")
    person(img, 1240, 70, 650, (40, 60, 160), "back")
    ref = R.prepare(img, aspect=0.6)
    assert ref is not None and iou(ref.mask, in_crop(front, ref)) > 0.9


def test_panels_with_a_close_up_keep_the_whole_front_figure():
    panels = []
    for i, shade in enumerate((0.0, -12.0, 8.0)):
        p = backdrop(500, 900, seed=3 + i, base=shade)
        if i == 0:
            gt = person(p, 250, 80, 760, (170, 40, 40))
        elif i == 1:
            person(p, 250, 80, 760, (40, 60, 160), "back")
        else:  # a face close-up: head and shoulders cut by the bottom of the panel
            d = ImageDraw.Draw(im := Image.fromarray(p))
            d.rectangle([40, 600, 460, 900], fill=(170, 40, 40))
            d.ellipse([140, 150, 360, 640], fill=SKIN)
            p[:] = np.asarray(im)
        panels.append(p)
    sheet = np.concatenate([panels[0], np.full((900, 4, 3), 60, np.uint8), panels[1], np.full((900, 4, 3), 60, np.uint8), panels[2]], 1)
    assert [a for a, _ in R.seams(sheet)] == [500, 1004]
    ref = R.prepare(sheet, aspect=0.5)
    assert ref is not None and ref.path == "front view cropped from a 3-view sheet"
    gt_full = np.zeros(sheet.shape[:2], bool)
    gt_full[:, :500] = gt
    assert iou(ref.mask, in_crop(gt_full, ref)) > 0.9


def test_busy_photos_are_left_to_sam():
    rng = np.random.default_rng(4)
    blocks = rng.integers(0, 255, (12, 8, 3)).astype(np.uint8)
    img = np.asarray(Image.fromarray(blocks).resize((640, 960), Image.NEAREST)).copy()
    img = np.clip(img.astype(int) + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
    gt = person(img, 320, 90, 760, (170, 40, 40), shadow=False)
    assert R.prepare(img, aspect=0.75) is None
    picked = R.prepare(img, aspect=0.75, mask=gt)  # what SAM 3.1 found on it
    assert picked is not None and picked.path == "person picked out of your photo"
    assert iou(picked.mask, in_crop(gt, picked)) > 0.97


def test_a_transparent_png_uses_its_alpha():
    img = backdrop(400, 600)
    gt = person(img, 200, 40, 520, (170, 40, 40), shadow=False)
    rgba = np.dstack([img, gt.astype(np.uint8) * 255])
    ref = R.prepare(Image.fromarray(rgba, "RGBA"), aspect=0.5)
    assert ref is not None and ref.path == "figure cut out from its transparent background"
    assert iou(ref.mask, in_crop(gt, ref)) > 0.97


def test_label_fill_and_touching():
    m = np.zeros((40, 40), bool)
    m[2:10, 2:10] = True
    m[10, 10] = True  # touches the square diagonally: same object
    m[20:30, 20:35] = True
    m[24:26, 24:26] = False  # a small hole
    lab, n = R.label(m)
    assert n == 2 and lab[2, 2] == lab[10, 10] != lab[20, 20]
    assert R.fill_holes(m, 0.05)[25, 25] and not R.fill_holes(m, 0.001)[25, 25]
    seed = np.zeros_like(m)
    seed[22, 30] = True
    assert np.array_equal(R.touching(m, seed), lab == lab[20, 20])
    assert R.largest(m).sum() == (lab == lab[20, 20]).sum()


def test_a_figures_straight_edge_is_not_a_seam():
    img = backdrop(600, 600)
    img[60:560, 280:330] = (40, 40, 50)  # a tall straight-sided figure, not reaching top or bottom
    assert R.seams(img) == []


def test_only_other_figures_are_painted_out():
    """A lone figure's picture is the original pixels: what its mask may miss (a white shoe on a light floor) stays."""
    img = backdrop(640, 960, seed=5)
    person(img, 320, 90, 760, (170, 40, 40))
    ImageDraw.Draw(im := Image.fromarray(img)).ellipse([330, 805, 420, 840], fill=(226, 224, 222))  # a white trainer
    img = np.asarray(im).copy()
    ref = R.prepare(img, aspect=0.75)
    assert ref is not None and max(ref.image.size) <= R.OUT_LONG  # not resized, so pixels compare 1:1
    x0, y0, x1, y1 = ref.box
    t = 4  # a still's outermost few pixels are trimmed like a sheet's frame line
    sx0, sy0, sx1, sy1 = max(t, x0), max(t, y0), min(640 - t, x1), min(960 - t, y1)
    shown = np.asarray(ref.image)[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    assert np.array_equal(shown, img[sy0:sy1, sx0:sx1])


def car(img: np.ndarray, cx: int, cy: int, long: int) -> None:
    """A grey saloon side-on: wider than tall, like a car on a vehicle sheet."""
    d = ImageDraw.Draw(im := Image.fromarray(img))
    d.rectangle([cx - long / 2, cy - 0.08 * long, cx + long / 2, cy + 0.1 * long], fill=(120, 122, 126))
    d.polygon([(cx - 0.25 * long, cy - 0.08 * long), (cx - 0.12 * long, cy - 0.2 * long), (cx + 0.18 * long, cy - 0.2 * long),
               (cx + 0.3 * long, cy - 0.08 * long)], fill=(90, 95, 104))
    for wx in (-0.3, 0.3):
        d.ellipse([cx + wx * long - 0.08 * long, cy + 0.02 * long, cx + wx * long + 0.08 * long, cy + 0.18 * long], fill=(25, 25, 28))
    img[:] = np.asarray(im)


def test_a_car_on_a_plain_backdrop_is_not_taken_for_a_character():
    img = backdrop(1200, 800, seed=6)
    car(img, 600, 420, 800)
    assert R.prepare(img, aspect=0.6) is None  # left to SAM, which finds no one: a fast, worded "couldn't find anyone"
    sheet = np.concatenate([img, np.full((800, 4, 3), 60, np.uint8), img[:, ::-1]], 1)
    assert R.prepare(sheet, aspect=0.6) is None


def test_a_huge_picture_is_read_at_a_useful_size():
    img = backdrop(640, 960, seed=7)
    gt = person(img, 320, 90, 760, (170, 40, 40))
    big = Image.fromarray(np.dstack([img, gt.astype(np.uint8) * 255]), "RGBA").resize((4000, 6000), Image.BILINEAR)
    rgb, alpha = R.load(big)
    assert rgb.shape == (R.IN_LONG, R.IN_LONG * 2 // 3, 3) and alpha is not None and alpha.shape == rgb.shape[:2]
    ref = R.prepare(big, aspect=0.75)
    assert ref is not None and ref.path == "figure cut out from its transparent background" and max(ref.image.size) <= R.OUT_LONG
    assert R.prepare(Image.new("RGB", (1, 1), (200, 200, 200)), aspect=0.75) is None
