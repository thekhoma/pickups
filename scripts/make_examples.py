"""Build the home screen's "What you can make" gallery from finished takes in your library.

Takes the same examples.json that run_examples.py ran, plus ~/.pickups/examples/runs.json (which shot and takes each
example produced). An example can instead name an existing take with {"shot_name", "take_label"}.
Writes ~/.pickups/examples/<id>_{before,after,receipt}.mp4 + <id>.jpg and examples.json (read by /api/examples),
each example with its workflow's default models (models_catalog) for the tile's model chips.
Loops are 1280 px wide, muted H.264. Only examples marked "public": true are included when --public is given
(the build for the published demo), so brand or client footage never leaves this machine by accident.

Usage: uv run python scripts/make_examples.py examples.json [--public]
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from pickups import models_catalog
from pickups.app.store import Store

HOME = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups")))
OUT = HOME / "examples"


def loop(src: Path, dst: Path, w: int = 1280, frames: int | None = None, trim: list | None = None) -> None:
    """A muted web loop. `trim` [first, last] (frames, inclusive) keeps only that part (the clean part of a take)."""
    cut = f"trim=start_frame={int(trim[0])}:end_frame={int(trim[1]) + 1},setpts=PTS-STARTPTS," if trim else ""
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"{cut}scale={w}:-2:flags=lanczos,format=yuv420p", "-an"]
    if frames:
        args += ["-frames:v", str(frames)]
    subprocess.run(args + ["-c:v", "libx264", "-crf", "21", "-preset", "slow", "-movflags", "+faststart", str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--public", action="store_true", help="only examples marked public (for the published demo)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    store = Store()
    runs = json.loads((OUT / "runs.json").read_text()) if (OUT / "runs.json").exists() else {}
    shots_by_name = {s["name"]: s for s in store.shots()}
    out = []
    for ex in json.loads(a.spec.read_text()):
        if ex.get("hold_reason"):  # kept out of the gallery until a better take exists
            print(f"hold {ex['id']}: {ex['hold_reason']}")
            continue
        if a.public and not ex.get("public"):
            continue
        shot = take = None
        if ex.get("shot_name"):
            shot = shots_by_name.get(ex["shot_name"])
            take = next((v for v in reversed(shot["versions"]) if v["label"].startswith(ex["take_label"])), None) if shot else None
        elif runs.get(ex["id"], {}).get("status") == "done":
            r = runs[ex["id"]]
            shot = store.shot(r["shot_id"])
            pick = r["takes"][-1 if ex.get("gallery_take") == "last" else 0]["take_id"]
            take = next((v for v in shot["versions"] if v["id"] == pick), None)
        if not take:
            print(f"skip {ex['id']}: no finished take yet")
            continue
        vdir = store.shot_dir(shot["id"]) / "versions" / take["id"]
        before = vdir / "before.mp4"
        if not before.exists():
            src = take.get("input_version") or "orig"
            before = (store.shot_dir(shot["id"]) / "proxy.mp4") if src == "orig" else (store.shot_dir(shot["id"]) / "versions" / src / "video.mp4")
        n = None  # before and after share a length so the wipe stays in sync
        loop(before, OUT / f"{ex['id']}_before.mp4", frames=n, trim=ex.get("trim"))
        loop(vdir / "video.mp4", OUT / f"{ex['id']}_after.mp4", frames=n, trim=ex.get("trim"))
        item = {"id": ex["id"], "task": ex["task"], "title": ex["title"], "line": ex["line"], "slate": ex.get("slate", ""),
                "sample_id": ex.get("sample_id"), "hero": bool(ex.get("hero")), "public": bool(ex.get("public")),
                "shot_id": shot["id"] if ex.get("open_shot") else None, "params": take.get("params", {}),
                "project": ex.get("project"),
                # the workflow's default models (the tile's model chips), not this take's own
                "models": models_catalog.as_json(models_catalog.for_task(ex["task"])),
                "before": f"{ex['id']}_before.mp4", "after": f"{ex['id']}_after.mp4", "poster": f"{ex['id']}.jpg"}
        if (vdir / "receipt.mp4").exists():
            loop(vdir / "receipt.mp4", OUT / f"{ex['id']}_receipt.mp4", trim=ex.get("trim"))
            item["receipt"] = f"{ex['id']}_receipt.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / f"{ex['id']}_after.mp4"), "-vf", "thumbnail=24", "-frames:v", "1",
                        str(OUT / f"{ex['id']}.jpg")], check=True)
        out.append(item)
        print("ok", ex["id"])
    # heroes first, then the spec's order
    out.sort(key=lambda x: not x["hero"])
    (OUT / "examples.json").write_text(json.dumps(out, indent=2))
    print(len(out), "examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
