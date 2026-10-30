"""Download the models Pickups needs into a ComfyUI models folder (run on the GPU machine).

Gated repositories (Lightricks, Black Forest Labs): the app leaves them to the browser. The `pickups models` command on a
GPU machine uses a token from the environment if one is set there; it's only ever sent to huggingface.co in the request
header, never printed, logged, returned or written anywhere.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from pickups.app import manifest


def download(files: set[str] | list[str], models_dir: Path, on_progress=None) -> list[str]:
    """Download each missing file into models_dir/<folder>/. Returns the files written."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    done = []
    for name in files:
        if name not in manifest.MODELS:
            continue
        info = manifest.model_info(name)
        dst = models_dir / info["folder"] / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(info["url"], headers={"Authorization": f"Bearer {token}"} if token else {})
        if not info["url"].startswith("https://huggingface.co/"):  # the token only ever goes to Hugging Face
            req = urllib.request.Request(info["url"])
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 - Hugging Face URLs from the manifest
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while chunk := r.read(8 << 20):
                f.write(chunk)
                got += len(chunk)
                if on_progress:
                    on_progress(name, got, total)
        tmp.replace(dst)
        done.append(name)
    return done


def missing_for(models_dir: Path, tasks: list[str] | None = None) -> list[str]:
    """Files to download for `tasks` (default: every video task plus the default image model, not every image model),
    skipping any already present under the same name or a known alias (BFL vs Comfy-Org names, bf16 vs int8)."""
    from pickups.images import _aliases

    want = set()
    for t, files in manifest.TASK_MODELS.items():
        if (tasks is None and not t.startswith(manifest.OPTIONAL)) or (tasks and t in tasks):
            want.update(files)

    def have(f: str) -> bool:
        folder = models_dir / manifest.MODELS[f][0]
        return any(folder.rglob(name) for name in (f,) + _aliases(f)) if folder.exists() else False
    return sorted(f for f in want if not have(f))
