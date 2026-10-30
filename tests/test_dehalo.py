"""dehalo removes a light glow around a subject without touching the subject, darker detail or the far background."""

import numpy as np

from pickups import composite


def scene():
    img = np.full((200, 200, 3), 90, np.uint8)  # mid-grey background
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(yy - 100, xx - 100)
    subject = r < 40
    img[subject] = (40, 60, 120)
    glow = (r >= 40) & (r < 52)
    img[glow] = 235  # a 12 px white halo
    img[100, 145:150] = 20  # a dark strand inside the halo zone
    return img, subject, glow


def test_glow_is_removed_and_subject_kept():
    img, subject, glow = scene()
    out = composite.dehalo(img, subject, width=16, inner=1)
    assert out[glow].astype(float).mean() < 120  # the 235 halo comes down to the background's level
    assert np.array_equal(out[composite.shrink(subject, 2)], img[composite.shrink(subject, 2)])
    far = ~composite.grow(subject, 24)
    assert np.array_equal(out[far], img[far])


def test_darker_detail_survives():
    img, subject, _ = scene()
    out = composite.dehalo(img, subject, width=16, inner=1)
    assert out[100, 145:150].max() <= 20  # lighten-only removal never brightens or smears dark strands


def test_narrow_band_misses_a_wide_glow():
    img, subject, glow = scene()
    narrow = composite.dehalo(img, subject, width=4, inner=1)
    wide = composite.dehalo(img, subject, width=16, inner=1)
    assert wide[glow].astype(float).mean() < narrow[glow].astype(float).mean()


def grass(h=260, w=260, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    g = np.stack([60 + 12 * np.sin(xx / 3.0) * np.cos(yy / 5.0), 95 + 18 * np.sin(yy / 4.0 + xx / 9.0),
                  45 + 8 * np.cos(xx / 6.0)], -1)
    return np.clip(g + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)


def pale_figure():
    """A white spacesuit lit from the left (with darker folds), a pale face and silver hair on grass, exact mask."""
    img = grass()
    yy, xx = np.mgrid[:260, :260]
    suit = (np.abs(xx - 130) < 45) & (yy > 70) & (yy < 250)
    head = np.hypot(yy - 45, xx - 130) < 26
    hair = head & (yy < 38)
    shade = 246 - 30 * (xx - 85) / 90.0  # 246 on the lit left edge, 216 on the right
    img[suit] = np.stack([shade, shade - 2, shade - 5], -1)[suit].astype(np.uint8)
    img[suit & (xx % 23 < 2)] = (165, 163, 160)  # folds
    img[head] = (232, 204, 190)
    img[hair] = (196, 196, 202)
    return img, suit | head


def test_a_white_or_pale_figure_keeps_its_edges():
    img, subject = pale_figure()
    out = composite.dehalo(img, subject, width=24, inner=10)  # Recast's numbers at 4K
    diff = np.abs(out.astype(int) - img.astype(int)).max(-1)
    assert diff[subject].max() <= 2  # suit, skin and hair are the character, not glow
    ring = composite.grow(subject, 10) & ~subject
    glowing = img.copy()
    glowing[ring] = (glowing[ring] * 0.35 + 240 * 0.65).astype(np.uint8)
    out = composite.dehalo(glowing, subject, width=24, inner=10)
    y = lambda a: a.astype(float) @ [0.299, 0.587, 0.114]  # noqa: E731
    assert y(out)[ring].mean() - y(img)[ring].mean() < 12 < y(glowing)[ring].mean() - y(img)[ring].mean()
    diff = np.abs(out.astype(int) - glowing.astype(int)).max(-1)
    assert diff[subject].max() <= 2


def test_glow_the_mask_swallowed_is_removed_but_never_deeper_than_inner():
    img, subject, glow = scene()
    swallowed = composite.grow(subject, 6)  # SAM's mask took half the glow
    img[composite.shrink(subject, 14) & ~composite.shrink(subject, 20)] = 225  # a white trim deep inside
    out = composite.dehalo(img, swallowed, width=16, inner=8)
    inside = glow & swallowed
    assert out[inside].astype(float).mean() < 130  # 235 inside the mask comes down too: lighter than the dark subject
    deep = composite.shrink(swallowed, 8)
    assert np.array_equal(out[deep], img[deep])
