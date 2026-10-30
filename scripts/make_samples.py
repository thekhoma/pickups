"""Cut example clips from your own footage into Pickups samples (the shots behind the home screen's examples).

Reads a JSON list of {id, path, start_s, duration_s, name, hint?, try?, pre?} and writes ~/.pickups/samples/<id>.mp4
(<= 3840 px long side, H.264 CRF 12, audio kept) + <id>.jpg + samples.json. Source footage is read, never changed, and
nothing is written into the repo.

Usage: uv run python scripts/make_samples.py spec.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from pickups import media

OUT = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "samples"
LONG = 3840


def cut(item: dict) -> Path:
    src = Path(item["path"])
    p = media.probe(src)
    s = min(1.0, LONG / max(p.width, p.height))
    w, h = int(p.width * s) // 2 * 2, int(p.height * s) // 2 * 2
    vf = ",".join(x for x in (item.get("pre", ""), f"scale={w}:{h}:flags=lanczos", "format=yuv420p") if x)
    dst = OUT / f"{item['id']}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{float(item.get('start_s', 0)):.3f}", "-i", str(src),
                    "-t", f"{float(item['duration_s']):.3f}", "-vf", vf, "-c:v", "libx264", "-crf", "12", "-preset", "slow",
                    "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", str(dst)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-vf", "thumbnail=24,scale=640:-2", "-frames:v", "1",
                    str(dst.with_suffix(".jpg"))], check=True)
    return dst


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    index_f = OUT / "samples.json"
    index = json.loads(index_f.read_text()) if index_f.exists() else {}
    for item in spec:
        dst = OUT / f"{item['id']}.mp4"
        if not dst.exists() or item.get("force"):
            cut(item)
        index[item["id"]] = {k: item[k] for k in ("name", "hint", "try") if item.get(k)}
        print("ok", item["id"], media.probe(dst).frames, "frames")
    index_f.write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
