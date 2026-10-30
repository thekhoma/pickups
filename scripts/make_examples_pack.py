"""Build the examples pack from a Pickups home: only the examples the app shows, their files byte for byte (stored,
not recompressed), the sample clips Try it opens, and a SHA-256 checksum beside the zip.

    PICKUPS_HOME=/path/to/pickups-home uv run python scripts/make_examples_pack.py OUT.zip

The format is pickups.app.examples_pack's (pack.json, examples/, samples/).
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

from pickups import features
from pickups.app import server


def shown(examples: list[dict]) -> list[dict]:
    """The gallery as the app filters it (server._examples): held, hidden and (with cars off) car-swap entries out."""
    return server.first_examples([e for e in examples if not e.get("hold_reason") and e.get("id") not in server.HIDDEN_EXAMPLES
                                  and (features.cars() or e.get("id") not in server.CAR_SWAP_EXAMPLES)])


def main(out: Path) -> None:
    ex_dir, smp_dir = server.EXAMPLES, server.SAMPLES
    examples = shown(json.loads((ex_dir / "examples.json").read_text()))
    sample_names = json.loads((smp_dir / "samples.json").read_text()) if (smp_dir / "samples.json").exists() else {}
    files: list[tuple[str, Path]] = []
    samples: dict = {}
    for e in examples:
        for k in ("before", "after", "poster", "receipt"):
            if e.get(k) and (ex_dir / e[k]).exists():
                files.append((f"examples/{e[k]}", ex_dir / e[k]))
        sid = e.get("sample_id")
        if sid and (smp_dir / f"{sid}.mp4").exists():
            files.append((f"samples/{sid}.mp4", smp_dir / f"{sid}.mp4"))
            if sid in sample_names:
                samples[sid] = sample_names[sid]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("pack.json", json.dumps({"version": 1, "examples": len(examples), "samples": len(samples)}, indent=2) + "\n")
        z.writestr("examples/examples.json", json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
        z.writestr("samples/samples.json", json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
        for arc, f in sorted(set(files)):
            z.write(f, arc)
    tmp.replace(out)
    h = hashlib.sha256()
    with out.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    (out.parent / (out.name + ".sha256")).write_text(f"{h.hexdigest()}  {out.name}\n")
    print(f"{out}: {len(examples)} examples, {len(samples)} sample clips, {out.stat().st_size / 1e6:.0f} MB, sha256 {h.hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
