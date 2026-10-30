from PIL import Image

from pickups.tasks import common


def test_previz_size_is_delivered_as_hd():
    # Previz and Change weather render at 1920x1088 (LTX sizes) and deliver HD 1920x1080.
    assert common.size_details(1920, 1088, 97, 24.0) == {
        "resolution": "1920×1080", "rendered_at": "1920×1088", "frames": 97, "fps": 24.0}


def test_delivery_size_matches_the_master():
    assert common.size_details(3840, 2160, 97, 24.0) == {"resolution": "3840×2160", "frames": 97, "fps": 24.0}
    assert common.size_details(1920, 1080, 89, 30000 / 1001)["fps"] == 29.97
    # Open gate keeps its own shape (never cropped to 16:9): delivered as made.
    assert common.size_details(3840, 2748, 202, 24.0) == {"resolution": "3840×2748", "frames": 202, "fps": 24.0}
    # Small sources: the top delivery size, and what the take was really made at.
    assert common.size_details(1280, 720, 145, 24.0)["resolution"] == "1920×1080"
    assert common.size_details(1280, 720, 145, 24.0)["rendered_at"] == "1280×720"


def test_delivery_sizes():
    assert [n for n, _, _ in common.delivery_sizes(3840, 2160)] == ["UHD", "HD"]
    assert [n for n, _, _ in common.delivery_sizes(1920, 1088)] == ["HD"]


def test_sequence_details(tmp_path):
    for i in range(5):
        Image.new("RGB", (1920, 1024)).save(tmp_path / f"r_{i + 1:05d}.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "look.png")  # not part of the sequence
    d = common.sequence_details(tmp_path, "r_*.png", 24.0)
    assert d == {"resolution": "1920×1080", "rendered_at": "1920×1024", "frames": 5, "fps": 24.0}
