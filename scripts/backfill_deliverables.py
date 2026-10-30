"""Re-make the delivery files (ProRes UHD/HD + MP4 H.264/H.265) for takes made before the current delivery spec.

Keeps task-specific extras (HDR master, EXR zip). Usage: uv run python scripts/backfill_deliverables.py
"""

import json

from pickups.app.store import Store
from pickups.tasks import common


def main() -> int:
    store = Store()
    for shot in store.shots():
        for v in shot["versions"][1:]:
            vdir = store.shot_dir(shot["id"]) / "versions" / v["id"]
            meta_f = vdir / "meta.json"
            meta = json.loads(meta_f.read_text())
            if any(d.get("codec") == "h265" for d in meta.get("downloads", [])) or not (vdir / "master.mov").exists():
                continue
            extras = [d for d in meta.get("downloads", []) if d["file"] in ("hdr_hlg.mp4", "exr_acescg.zip")]
            for d in extras:
                d.setdefault("codec", "hlg" if d["file"].endswith(".mp4") else "exr")
            meta["downloads"] = common.deliverables(vdir / "master.mov", vdir) + extras
            meta_f.write_text(json.dumps(meta, indent=2))
            print("ok", shot["name"], v["label"], [d["file"] for d in meta["downloads"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
