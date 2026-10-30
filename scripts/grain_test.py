"""Add source-matched grain to the AI frames of an Extend run; writes out_grain/ next to out/.

Usage: uv run python scripts/grain_test.py ~/.pickups/runs/sharp_57_dfr
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    rep = json.loads((a.run / "report.json").read_text())
    seam = rep["seam_frame"]
    ctx = [np.asarray(Image.open(f).convert("RGB")) for f in sorted((a.run / "canvas_frames").glob("c_*.png"))[:seam]]
    target = composite.grain_profile(ctx[-8:])
    gen = sorted((a.run / "out").glob("frame_*.png"))
    out = a.run / "out_grain"
    out.mkdir(exist_ok=True)
    before = composite.grain_profile([np.asarray(Image.open(gen[seam + 12]).convert("RGB"))])
    for i, f in enumerate(gen):
        im = np.asarray(Image.open(f).convert("RGB"))
        if i >= seam:
            im = composite.add_grain(im, target, seed=i)
        Image.fromarray(im).save(out / f.name)
    after = composite.grain_profile([np.asarray(Image.open(out / gen[seam + 12].name).convert("RGB"))])
    print(json.dumps({"source_detail_std": target.round(2).tolist(), "ai_before": before.round(2).tolist(),
                      "ai_after": after.round(2).tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
