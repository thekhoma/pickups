"""Pack the home screen's example gallery for people who install Pickups fresh (the loops live on your machine).

Writes pickups-examples.zip with examples.json and the loops/posters of every example marked public (nothing else):
upload it as a release asset and point PICKUPS_EXAMPLES_URL (or EXAMPLES_PACK_URL in pickups/app/server.py) at it.
The app then offers "Get the example gallery" on first run.

Usage: uv run python scripts/pack_examples.py [out.zip]
"""

import json
import os
import sys
import zipfile
from pathlib import Path

HOME = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups")))
SRC = HOME / "examples"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "pickups-examples.zip")
    items = [e for e in json.loads((SRC / "examples.json").read_text()) if e.get("public")]
    files = set()
    for e in items:
        e.pop("shot_id", None)  # local library ids mean nothing on another machine
        e.pop("params", None)   # take settings can carry local upload ids
        for k in ("before", "after", "receipt", "poster"):
            if e.get(k):
                files.add(e[k])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # MP4/JPEG are already compressed
        z.writestr("examples.json", json.dumps(items, indent=2))
        for f in sorted(files):
            if (SRC / f).exists():
                z.write(SRC / f, f)
    print(f"{out}: {len(items)} examples, {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
