"""The commercial-safe switch swaps the frame editor; the default stays Qwen-Image 2.1."""

from pickups import graphs


def _classes(g: dict) -> set[str]:
    return {n["class_type"] for n in g.values()}


def test_default_is_qwen_image_21():
    g = graphs.image_edit("a.png", "Turn <image1> to snow, like <image2>.", refs=["b.png"])
    assert "TextEncodeQwenImage21" in _classes(g)
    assert g["PU_Prompt"]["inputs"]["prompt"] == "Turn <image1> to snow, like <image2>."
    assert "PU_Plate" in g


def test_commercial_safe_uses_edit_2511_with_picture_names():
    g = graphs.image_edit("a.png", "Render <image1> with the driver from <image2> and the car from <image3>.",
                          refs=["b.png", "c.png", "d.png"], commercial_safe=True, size=(1328, 752))
    assert "TextEncodeQwenImageEditPlus" in _classes(g) and "TextEncodeQwenImage21" not in _classes(g)
    enc = g["PU_Instruction"]["inputs"]
    assert enc["prompt"] == "Render Picture 1 with the driver from Picture 2 and the car from Picture 3."
    assert {"image1", "image2", "image3"} <= enc.keys()  # 2511 reads three images; the fourth is dropped
    assert g["PU_Frame3"]["inputs"]["image"] == "c.png" and "PU_Frame4" not in g
    assert g["scaled"]["inputs"]["width"] == 1328 and "PU_Plate" in g


def test_edit_size_is_one_megapixel_in_sixteens():
    w, h = graphs.edit_size(1920, 1088)
    assert w % 16 == 0 and h % 16 == 0 and abs(w * h - 1_000_000) < 40_000 and abs(w / h - 1920 / 1088) < 0.02


def test_receipt_names():
    assert graphs.edit_model(False) == "Qwen-Image 2.1" and graphs.edit_model(True) == "Qwen-Image-Edit 2511"


def test_setup_tiles_swap_the_editor():
    from pickups.app import manifest

    off, on = manifest.task_models(False), manifest.task_models(True)
    assert "commercial-safe" not in off and not any(t.startswith("image-") for t in off)
    assert "qwen_image_2.1_int8_convrot.safetensors" in off["remove"]
    assert "qwen_image_2.1_int8_convrot.safetensors" not in on["previz"]
    assert "qwen_image_edit_2511_fp8mixed.safetensors" in on["environment"]
    # Recast on LTX paints the character in with the frame editor; its SCAIL-2 engine is an extra download, not a tile.
    assert "qwen_image_edit_2511_fp8mixed.safetensors" in on["recast"] and "qwen_image_2.1_int8_convrot.safetensors" in off["recast"]
    assert "recast-scail" not in off and "recast-scail" in manifest.TASK_MODELS
    assert all(f in manifest.MODELS for f in manifest.QWEN_EDIT_2511)
