import subprocess

from pickups import media, share


def _clip(path, color):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d=2.5:r=24", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def test_share_formats(tmp_path):
    before, after = _clip(tmp_path / "b.mp4", "red"), _clip(tmp_path / "a.mp4", "blue")
    receipt = {"original": 96.2, "borrowed": 3.8, "ai": 0.0}
    for fmt, size in (("9x16", (1080, 1920)), ("1x1", (1080, 1080))):
        out = share.make(before, after, tmp_path / f"s_{fmt}.mp4", fmt, "Remove", ["SAM 3.1"], receipt, "Harbour pier")
        p = media.probe(out)
        assert (p.width, p.height) == size
        assert 2.0 <= p.frames / float(p.fps) <= 2.6
        assert not out.with_suffix(".png").exists()  # the slate overlay is cleaned up


def test_slate_mentions_original_pixels(tmp_path):
    img = share.slate_overlay(1080, 1920, "9x16", "Remove", ["SAM 3.1"], {"original": 97.0, "ai": 0.0}, "Pier", tmp_path / "o.png")
    assert img.exists()
