"""The examples pack: Home's example gallery and the clips "Try it" opens, as one zip (or its unzipped folder).

A fresh install has no video examples: they're the author's renders, too big for the repo. The pack carries them:

    pack.json                     {"version": 1, "examples": N, "samples": M}
    examples/examples.json        the gallery (the entries the app shows)
    examples/<id>_before.mp4 ...  each example's loops and poster, byte for byte as made
    samples/samples.json          names for the copied shots
    samples/<sample_id>.mp4       the clip Try it opens

Setup → Examples imports one (a zip or a folder), and on first run the app looks for one next to itself. An older
flat pack (example files only, no folders) still installs. Nothing outside the two folders is ever written.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

PACK_NAMES = ("pickups_examples*.zip", "examples_pack")  # what "next to the app" looks for, in this order
MEDIA = {".mp4", ".jpg", ".jpeg", ".png", ".json"}


def _safe(name: str) -> str | None:
    """A plain file name, or None for anything that could leave its folder or isn't pack media."""
    n = Path(name).name
    if not n or n != name or n.startswith(".") or Path(n).suffix.lower() not in MEDIA:
        return None
    return n


def _members(src: Path):
    """(folder, file name, read()) for each file in a zip or a folder: folder is "examples", "samples" or "" (flat)."""
    if src.is_dir():
        for sub in ("examples", "samples"):
            for f in sorted((src / sub).glob("*")) if (src / sub).is_dir() else []:
                if f.is_file():
                    yield sub, f.name, f.read_bytes
        for f in sorted(src.glob("*")):
            if f.is_file() and f.name != "pack.json":
                yield "", f.name, f.read_bytes
        return
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if len(parts) == 2 and parts[0] in ("examples", "samples"):
                yield parts[0], parts[1], (lambda i=info: z.read(i))
            elif len(parts) == 1 and parts[0] != "pack.json":
                yield "", parts[0], (lambda i=info: z.read(i))


def install(src: Path, examples_dir: Path, samples_dir: Path) -> dict:
    """Unpack `src` (a pack zip or folder) into the app's examples and samples folders. samples.json is merged
    (the pack's names win); examples.json is replaced. Returns {"examples": n, "samples": m}."""
    src = Path(src).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Pickups can't find {src}.")
    if src.is_file() and not zipfile.is_zipfile(src):
        raise ValueError("That isn't a zip. Pick the pickups_examples zip, or the folder it unzips to.")
    examples_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    n_ex = n_smp = 0
    names: dict = {}
    for folder, name, read in _members(src):
        safe = _safe(name)
        if safe is None:
            continue
        if folder == "samples":
            if safe == "samples.json":
                names.update(json.loads(read()))
            else:
                (samples_dir / safe).write_bytes(read())
                n_smp += 1
        else:  # "examples", or a flat pack's files
            (examples_dir / safe).write_bytes(read())
            n_ex += safe == "examples.json"
    if not (examples_dir / "examples.json").exists():
        raise ValueError("That pack has no examples.json, so there's nothing to show.")
    if names:
        f = samples_dir / "samples.json"
        have = json.loads(f.read_text()) if f.exists() else {}
        f.write_text(json.dumps(have | names, indent=2, ensure_ascii=False) + "\n")
    shown = len(json.loads((examples_dir / "examples.json").read_text()))
    return {"examples": shown, "samples": n_smp}


def find_next_to(roots: list[Path]) -> Path | None:
    """A pack beside the app: pickups_examples*.zip (newest name first) or an examples_pack folder, in the first root
    that has one."""
    for root in roots:
        for pattern in PACK_NAMES:
            found = sorted(root.glob(pattern), reverse=True)
            for f in found:
                if f.is_file() or (f / "examples" / "examples.json").exists():
                    return f
    return None
