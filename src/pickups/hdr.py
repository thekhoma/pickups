"""Detail-safe HDR: keep the source pixels and take only the highlights from LTX's HDR reconstruction.

LTX-2.5 SDR->HDR regenerates the whole frame through a VAE at ~1080p, which softens texture. Here the output is
built at the SOURCE resolution: below a luma threshold it is the source itself (linearised); towards the highlights
it blends into LTX's scene-linear ACEScg values, which carry the reconstructed range. Then it is encoded as a
BT.2020 / HLG 10-bit HEVC master (reference white at HLG 0.75, BT.2408).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# Linear Rec.709 -> ACES AP1 (ACEScg) and AP1 -> linear Rec.2020 (D65; Bradford for the AP1 white point).
M_709_TO_AP1 = np.array([[0.6130974, 0.3395231, 0.0473795],
                         [0.0701937, 0.9163539, 0.0134524],
                         [0.0206156, 0.1095698, 0.8698147]], dtype=np.float32)
M_AP1_TO_2020 = np.array([[1.0258247, -0.0200532, -0.0057715],
                          [-0.0022344, 1.0045865, -0.0023521],
                          [-0.0050134, -0.0252901, 1.0303035]], dtype=np.float32)

HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073
REF_WHITE_E = (np.exp((0.75 - HLG_C) / HLG_A) + HLG_B) / 12.0  # scene-linear E giving HLG 0.75


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def read_frames(path: Path, width: int, height: int, pix_fmt: str, dtype) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=dtype)


def read_exr(path: Path) -> np.ndarray:
    """One EXR frame -> float32 HxWx3. Uses OpenEXR: ffmpeg's EXR decoder clamps half-float values to 1.0."""
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch:
            return np.asarray(ch["RGB"].pixels, dtype=np.float32)
        return np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in ("R", "G", "B")], axis=-1)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(img[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], axis=-1)


def merge_frame(sdr_rgb8: np.ndarray, hdr_ap1: np.ndarray, lo: float = 0.55, hi: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Return (merged scene-linear ACEScg at source size, blend weight). Source pixels below `lo` luma are kept."""
    h, w = sdr_rgb8.shape[:2]
    sdr = sdr_rgb8.astype(np.float32) / 255.0
    luma = sdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    sdr_ap1 = srgb_to_linear(sdr) @ M_709_TO_AP1.T
    hdr = resize(hdr_ap1, w, h) if hdr_ap1.shape[:2] != (h, w) else hdr_ap1
    # Match LTX's exposure to the source in the midtones so the blend is seamless.
    mid = (luma > 0.15) & (luma < 0.5)
    if mid.sum() > 1000:
        k = float(np.median(hdr[mid].mean(axis=1)) / max(np.median(sdr_ap1[mid].mean(axis=1)), 1e-4))
        hdr = hdr / max(k, 1e-3)
    wgt = smoothstep(lo, hi, luma)[..., None]
    merged = sdr_ap1 * (1 - wgt) + np.maximum(hdr, sdr_ap1) * wgt
    return merged, wgt[..., 0]


def hlg_encode(ap1_linear: np.ndarray) -> np.ndarray:
    """Scene-linear ACEScg (1.0 = diffuse white) -> HLG BT.2020 signal 0..1 with a soft roll-off near peak."""
    rgb2020 = np.clip(ap1_linear @ M_AP1_TO_2020.T, 0.0, None)
    e = rgb2020 * REF_WHITE_E
    knee = 0.75
    e = np.where(e < knee, e, knee + (1 - knee) * (1 - np.exp(-(e - knee) / (1 - knee))))
    return np.where(e <= 1 / 12, np.sqrt(3 * e), HLG_A * np.log(np.maximum(12 * e - HLG_B, 1e-6)) + HLG_C)


def write_hlg(frames_hlg, fps: str, out: Path, crf: int = 10) -> None:
    """Encode HLG RGB signal frames (0..1; a list or any iterable) as 10-bit 4:2:0 HEVC tagged BT.2020 / arib-std-b67."""
    it = iter(frames_hlg)
    first = next(it)
    h, w = first.shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", fps, "-i", "-",
         "-vf", "scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le",
         "-c:v", "libx265", "-crf", str(crf), "-preset", "medium", "-tag:v", "hvc1",
         "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
         "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv",
         str(out)], stdin=subprocess.PIPE)
    assert proc.stdin
    for f in (first, *it):
        proc.stdin.write(np.clip(np.rint(f * 65535), 0, 65535).astype("<u2").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg HLG encode failed")


def preview(ap1_linear: np.ndarray, ev: float = 0.0) -> np.ndarray:
    """Simple display preview of scene-linear ACEScg: exposure pull, AP1 -> 709, sRGB encode (clips above white)."""
    m_ap1_to_709 = np.linalg.inv(M_709_TO_AP1)
    lin709 = (ap1_linear * (2.0 ** ev)) @ m_ap1_to_709.T
    return (linear_to_srgb(lin709) * 255).round().astype(np.uint8)


def write_exr(path: Path, ap1_linear: np.ndarray) -> None:
    """Write scene-linear ACEScg as a half-float RGB EXR (OpenEXR)."""
    import OpenEXR

    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"RGB": ap1_linear.astype(np.float16)}) as f:
        f.write(str(path))
