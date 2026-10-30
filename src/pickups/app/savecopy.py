"""Save videos to: a folder the user picks (first asked on the first Save to), where every finished take is copied,
one folder per project: <folder>/<project>/<shot>_T03_Finish_4K_HDR.mov. The library stays the working copy; this is
the copy the user keeps, finds and edits with. A copy that fails is logged and never fails the take.

settings.json keeps `save_dir`; nothing is copied until it is set."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from pickups.app.store import _safe


def default_dir() -> Path:
    """The suggestion: Movies/Pickups on a Mac, Videos/Pickups elsewhere."""
    home = Path.home()
    return home / ("Movies" if sys.platform == "darwin" else "Videos") / "Pickups"


def shown(path: str | Path | None) -> str | None:
    """A folder as a person reads it: the home folder as ~ (~/Movies/Pickups)."""
    if not path:
        return None
    p, home = Path(path), Path.home()
    try:
        return "~/" + p.relative_to(home).as_posix() if p != home else "~"
    except ValueError:
        return str(p)


def check_dir(raw: str) -> Path:
    """A usable folder from what the user typed or picked (made if missing). ValueError says what's wrong."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Choose a folder.")
    p = Path(os.path.expanduser(text))
    if not p.is_absolute():
        raise ValueError("Use a full path, like ~/Movies/Pickups or /Volumes/Media/Pickups.")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Pickups can't make that folder: {e.strerror or e}.") from e
    if not os.access(p, os.W_OK):
        raise ValueError("Pickups can't write to that folder.")
    return p


def copy_name(shot_name: str, take_no: int, label: str, suffix: str = ".mov") -> str:
    return f"{_safe(shot_name)}_T{take_no:02d}_{_safe(label or 'take')}{suffix}"


def copy_take(store, shot_id: str, vid: str, save_dir: str | None) -> Path | None:
    """Copy a finished take's master into <save_dir>/<project>/. Returns the copy, or None (no folder, no master)."""
    if not save_dir:
        return None
    shot = store.shot(shot_id)
    src = store.version_dir(shot_id, vid) / "master.mov"
    if not src.exists():
        return None
    ids = [v["id"] for v in shot["versions"]]
    num = ids.index(vid) if vid in ids else len(ids)
    version = next((v for v in shot["versions"] if v["id"] == vid), {})
    try:
        project = store.project(shot["project_id"])["name"]
    except (OSError, ValueError, KeyError):
        project = "My shots"
    folder = Path(save_dir) / (_safe(project) if project else "My_shots")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / copy_name(shot["name"], num, version.get("label") or version.get("task") or "take")
    shutil.copy2(src, dest)
    return dest


def copy_take_later(store, shot_id: str, vid: str, save_dir: str | None) -> None:
    """copy_take in the background (a ProRes master can be gigabytes); a failure is printed, never raised."""
    if not save_dir:
        return

    def run():
        try:
            copy_take(store, shot_id, vid, save_dir)
        except Exception as e:  # noqa: BLE001 - the take is saved in the library either way
            print(f"Save videos to: couldn't copy {shot_id}/{vid}: {e}", flush=True)

    threading.Thread(target=run, daemon=True).start()


def choose_dir(start: str | None = None, timeout: float = 300) -> str | None:
    """The computer's own folder picker (Pickups on this computer only). The chosen path, or None when canceled or
    when there is no picker here (the typed path still works)."""
    try:
        if sys.platform == "darwin":
            where = f' default location (POSIX file "{start}")' if start and Path(start).is_dir() and '"' not in start else ""
            r = subprocess.run(["osascript", "-e", f'POSIX path of (choose folder with prompt "Save videos to"{where})'],
                               capture_output=True, text=True, timeout=timeout)
        elif sys.platform.startswith("win"):
            ps = ("Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                  "$d.Description = 'Save videos to'; if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }")
            r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True, timeout=timeout)
        elif shutil.which("zenity"):
            r = subprocess.run(["zenity", "--file-selection", "--directory", "--title=Save videos to"], capture_output=True, text=True, timeout=timeout)
        else:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        return None
    return out.rstrip("/") or out
