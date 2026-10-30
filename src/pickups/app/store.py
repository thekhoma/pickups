"""The Pickups library on disk: shots, their versions, uploads.

Layout (under ~/.pickups/library by default; nothing here is ever written into the repo):
  shots/<shot>/meta.json, source.<ext>, proxy.mp4, thumb.jpg
  shots/<shot>/versions/<version>/meta.json, video.mp4 (viewer), master.mov (ProRes), receipt.mp4, thumb.jpg, ...
  uploads/<id>.<ext>
  projects/<project>/meta.json, cast.json, cast/<id>.<ext>, images.json, images/<id>.png (+ <id>_thumb.jpg)
  projects/<project>/cast/<id>_sheet_<v>.jpg (+ _small.jpg), <id>_{front,back,face}_<v>.png: a character sheet (sheets.py)
  trash/<id>/trash.json, and what went there under its library path (a project's folder and its shots' folders, a
  cast entry's or an image's files, a shot, a take): nothing is erased until Empty Trash
Shots belong to a project (meta.project_id; none = "Unsorted"). Takes (versions) carry a review status and notes.
The original is never modified; every task writes a new version.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from pickups import licences, media
from pickups.app.workflow_titles import saved_title

ROOT = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "library"
VIEW_LONG = 1920  # viewer proxies are H.264 at up to this long side
UNSORTED = "unsorted"  # id of the built-in "My shots" project (shots without a project land here)
EXAMPLES = "examples"  # built-in project the home screen's examples open into; never a default target
QC = "qc"  # "Test runs": an archived project QC and test runs save into, so they never land in a person's projects
STATUSES = ("", "approved")
CAST_KINDS = ("person", "place", "prop", "look")
UPLOAD_ID = re.compile(r"up_[0-9a-f]+")  # new_id("up")
ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # a shot or take id from a URL: a folder name, never "..", never a path

_lock = threading.Lock()


def _tree_bytes(path: Path) -> int:
    """Bytes under a folder, without following symlinks."""
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def _remote_names(path: Path) -> int:
    """How many files the ComfyUI machine kept for these takes: every output prefix and input file named in their saved
    workflows (workflows/*.json). Counted from the graphs; nothing is asked of the server."""
    names: set[str] = set()
    for wf in Path(path).rglob("workflows/*.json"):
        try:
            graph = json.loads(wf.read_text())
        except (OSError, ValueError):
            continue
        nodes = graph.get("prompt", graph) if isinstance(graph, dict) else {}
        for node in nodes.values() if isinstance(nodes, dict) else []:
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            for key in ("filename_prefix", "image", "video", "audio", "file"):
                v = inputs.get(key)
                if isinstance(v, str) and v:
                    names.add(f"{key}:{v}")
    return len(names)


def keep_work() -> bool:
    """PICKUPS_KEEP_WORK set (and not 0 or false): no work folder, take folder or orphan is removed, so a test run can
    read every intermediate (R&D #16). The one switch the jobs and the store both read."""
    return os.environ.get("PICKUPS_KEEP_WORK", "").strip().lower() not in ("", "0", "false", "no", "off")


def drop_work(path: Path) -> bool:
    """Remove a job's own scratch folder unless keep_work(). True when it went."""
    if keep_work():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not Path(path).exists()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def media_url(path: Path) -> str:
    return "/media/" + path.relative_to(ROOT).as_posix()


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def make_viewer(src: Path, out: Path, frames: int | None = None) -> None:
    """H.264 proxy for the browser (even dims, <= VIEW_LONG, yuv420p, faststart)."""
    p = media.probe(src)
    s = min(1.0, VIEW_LONG / max(p.width, p.height))
    w, h = int(p.width * s) // 2 * 2, int(p.height * s) // 2 * 2
    rec709 = media.to_rec709(p)  # a phone's HDR clip: the same colours as the work clips the takes are made from
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", (f"{rec709}," if rec709 else "") +
            f"scale={w}:{h}:flags=lanczos,format=yuv420p", "-an"]
    if frames:
        args += ["-frames:v", str(frames)]
    subprocess.run(args + ["-c:v", "libx264", "-crf", "17", "-preset", "fast", "-movflags", "+faststart", str(out)], check=True)


def make_thumb(src: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "thumbnail=12,scale=480:-2", "-frames:v", "1", str(out)], check=True)


CODEC_NAMES = {"prores": "ProRes422HQ", "h264": "H264", "h265": "H265"}


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "shot"


def download_name(shot_name: str, take_no: int, task: str | None, dl: dict) -> str:
    """NLE-safe, self-describing file name: harbour_pier_T02_remove_ProRes422HQ_UHD.mov."""
    ext = Path(dl["url"]).suffix
    kind = CODEC_NAMES.get(dl.get("codec") or "") or _safe(Path(dl["url"]).stem)
    size = dl.get("size") or ""
    parts = [_safe(shot_name), f"T{take_no:02d}", task or "original", kind] + ([size] if size and size != "native" else [])
    return "_".join(parts) + ext


FRAME_EDITORS = ("Qwen-Image 2.1", "Qwen-Image-Edit 2511")


def _models_on_read(m: dict) -> list:
    """The models a take ran, from its meta.json, with what older takes got wrong put right (read time only; the file
    is never rewritten):
    - a Recast whose person FLUX 3 klein made from a description (a saved "cast" workflow, or a reference prompt and no
      cast, project or dropped image) before Recast named it;
    - a Remove made of real pixels only (0% AI, no plate step, no plate in its method): older Removes listed the frame
      editor although it added nothing."""
    d = m.get("details") if isinstance(m.get("details"), dict) else {}
    models = [x for x in d.get("models") or [] if isinstance(x, str)] if isinstance(d.get("models"), list) else []
    steps = {str(w.get("step")) for w in m.get("workflows") or [] if isinstance(w, dict)}
    p = m.get("params") if isinstance(m.get("params"), dict) else {}
    if m.get("task") == "recast" and "FLUX 3 klein" not in models and (
            "cast" in steps or (str(p.get("reference_prompt") or "").strip() and not any(p.get(k) for k in ("cast_id", "who_ref", "ref_upload_id")))):
        models.append("FLUX 3 klein")
    r = m.get("receipt") if isinstance(m.get("receipt"), dict) else {}
    try:
        no_ai = r and float(r.get("ai") or 0) == 0
    except (TypeError, ValueError):
        no_ai = False
    if m.get("task") == "remove" and no_ai and "plate" not in steps and "plate" not in str(d.get("method") or ""):
        models = [x for x in models if x not in FRAME_EDITORS]
    return models


def _follow_inputs(versions: list[dict]) -> None:
    """A take made from another take carries that take's pixels, so (read time, in place):
    - its licences gain the input chain's models it didn't run itself, each marked "via Take NN";
    - a Finish take's receipt is composed with its input's (receipts.compose) so its shares are of the original footage,
      and its details["hdr"] says the same as its receipt (receipts.hdr_detail).
    A chain that is broken (an input that isn't in the shot) or loops stops where it breaks."""
    from pickups import receipts

    index = {v["id"]: i for i, v in enumerate(versions)}
    own = {v["id"]: list(v.get("licences") or []) for v in versions}
    shares: dict[str, dict | None] = {}

    def receipt(vid: str, seen: frozenset) -> dict | None:
        """A take's receipt as it reads, composed along its inputs (Finish of a Finish of a Remove, …)."""
        if vid in shares:
            return shares[vid]
        v = versions[index[vid]]
        r = v.get("receipt")
        src = v.get("input_version") or "orig"
        if v.get("task") == "finish" and src != "orig" and src in index and src not in seen:
            s = versions[index[src]]
            r = receipts.compose(r, receipt(src, seen | {vid}), f"{index[src] + 1:02d}", s.get("label") or "")
        shares[vid] = r
        return r

    for v in versions:
        if v["id"] == "orig":
            continue
        if v.get("receipt") is not None:
            v["receipt"] = receipt(v["id"], frozenset())
            if v.get("task") == "finish" and isinstance(v.get("details"), dict) and v["details"].get("hdr"):
                v["details"] = {**v["details"], "hdr": receipts.hdr_detail(v["receipt"]) or v["details"]["hdr"]}
        names = {x.get("name") for x in v.get("licences") or []}
        seen, src = {v["id"]}, v.get("input_version") or "orig"
        while src != "orig" and src in index and src not in seen:
            seen.add(src)
            for x in own[src]:
                if x.get("name") not in names:
                    names.add(x.get("name"))
                    v["licences"] = [*(v.get("licences") or []), {**x, "via": f"Take {index[src] + 1:02d}"}]
            src = versions[index[src]].get("input_version") or "orig"


def listed(shot: dict) -> bool:
    """The Examples project lists only shots with something to compare (a take that isn't hidden)."""
    return shot["project_id"] != EXAMPLES or any(v["id"] != "orig" and not v.get("hidden") for v in shot["versions"])


class Store:
    def __init__(self, root: Path = ROOT):
        self.root = root
        (root / "shots").mkdir(parents=True, exist_ok=True)
        (root / "uploads").mkdir(parents=True, exist_ok=True)
        (root / "projects").mkdir(parents=True, exist_ok=True)
        self._builtin_projects()

    def _builtin_projects(self) -> None:
        """My shots and Examples always exist: a deleted My shots comes back empty, so there's always a project."""
        for pid, name in ((UNSORTED, "My shots"), (EXAMPLES, "Examples")):
            if not (self.root / "projects" / pid / "meta.json").exists():
                self._write_project(pid, {"id": pid, "name": name, "created_at": time.time(), "updated_at": time.time(), "archived": False})

    # --- shots -------------------------------------------------------------------------------------------------
    def shot_dir(self, shot_id: str) -> Path:
        if not ITEM_ID.fullmatch(str(shot_id)):
            raise FileNotFoundError(shot_id)
        return self.root / "shots" / shot_id

    def add_shot(self, src: Path, name: str, move: bool = False, project_id: str | None = None,
                 extra: dict | None = None) -> dict:
        """A new shot from `src` (copied, or moved), in `project_id` (else My shots). `extra` goes into its meta.json
        (e.g. "prepared_from": the shot and range it was trimmed and cropped from)."""
        shot_id = new_id("shot")
        d = self.shot_dir(shot_id)
        (d / "versions").mkdir(parents=True)
        dst = d / f"source{src.suffix.lower()}"
        if move:
            shutil.move(str(src), dst)
        else:
            shutil.copy2(src, dst)
        p = media.probe(dst)
        make_viewer(dst, d / "proxy.mp4")
        make_thumb(d / "proxy.mp4", d / "thumb.jpg")
        meta = {"id": shot_id, "name": name, "source": dst.name, "width": p.width, "height": p.height,
                "fps": float(p.fps), "fps_num": p.fps.numerator, "fps_den": p.fps.denominator, "frames": p.frames,
                "duration_s": round(p.frames / float(p.fps), 3), "has_audio": p.has_audio, "created_at": time.time(),
                "project_id": project_id if project_id and self.project_exists(project_id) else UNSORTED}
        meta = {k: v for k, v in (extra or {}).items() if k not in meta} | meta
        _write_json(d / "meta.json", meta)
        return self.shot(shot_id)

    def shot_meta(self, shot_id: str) -> dict:
        return json.loads((self.shot_dir(shot_id) / "meta.json").read_text())

    def source_path(self, shot_id: str) -> Path:
        return self.shot_dir(shot_id) / self.shot_meta(shot_id)["source"]

    def shot(self, shot_id: str) -> dict:
        d = self.shot_dir(shot_id)
        m = self.shot_meta(shot_id)
        orig = {"id": "orig", "label": "Original", "task": None, "created_at": m["created_at"],
                "video_url": media_url(d / "proxy.mp4"), "thumb_url": media_url(d / "thumb.jpg"), "downloads": [], "details": {},
                "status": m.get("orig_status", ""), "notes": m.get("orig_notes", []), "input_version": None, "params": {},
                "workflows": [], "licences": []}
        versions = [orig] + self.versions(shot_id)
        _follow_inputs(versions)  # a take made from a take: its input's models, and a Finish's shares of the footage
        for i, v in enumerate(versions):
            for dl in v["downloads"]:
                dl["filename"] = download_name(m["name"], i + 1, v.get("task"), dl)
            for j, w in enumerate(v.get("workflows", [])):  # ComfyUI workflows, named like the take's files
                w["filename"] = f"{_safe(m['name'])}_T{i + 1:02d}_{v.get('task') or 'original'}_{j + 1:02d}_{_safe(w['step'])}_comfyui.json"
        out = {"id": m["id"], "name": m["name"], "width": m["width"], "height": m["height"], "fps": m["fps"], "frames": m["frames"],
               "duration_s": m["duration_s"], "thumb_url": orig["thumb_url"], "video_url": orig["video_url"],
               "project_id": m.get("project_id", UNSORTED), "created_at": m["created_at"],
               "latest_thumb_url": versions[-1]["thumb_url"], "approved": sum(v.get("status") == "approved" for v in versions),
               "versions": versions}
        # A shot Create made (text or image to video): how, so it can be made again and listed with what Pickups made.
        if isinstance(m.get("made"), dict):
            out["made"] = m["made"]
        # A shot made in the Previz tab keeps the layout that rendered it, so it can be opened again.
        if isinstance(m.get("previz_spec"), dict):
            out["previz_spec"] = m["previz_spec"]
        if isinstance(m.get("previz_frame0"), str):
            out["previz_frame0"] = m["previz_frame0"]
        if isinstance(m.get("previz_framing"), dict):
            out["previz_framing"] = m["previz_framing"]  # what frame 0 shows: the start frame is made from it
        matte = self.shot_dir(shot_id) / "previz_subjects0.png"  # previz_maker.SUBJECTS0: frame 0's people and vehicles
        if matte.is_file():
            out["previz_matte_url"] = media_url(matte)
        depth = self.shot_dir(shot_id) / "previz_depth_0000.png"  # start_frame.DEPTH_FILE of frame 0: the start still's depth
        if depth.is_file():
            out["previz_depth_url"] = media_url(depth)
        return out

    def shots(self, project_id: str | None = None) -> list[dict]:
        out = []
        for d in sorted((self.root / "shots").iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if (d / "meta.json").exists():
                try:
                    shot = self.shot(d.name)
                except Exception:  # noqa: BLE001 - a half-written shot must not break the library
                    continue
                if (project_id is None or shot["project_id"] == project_id) and listed(shot):
                    out.append(shot)
        return out

    def set_shot_meta(self, shot_id: str, **fields) -> None:
        """Set fields in a shot's meta.json (e.g. a refreshed previz_framing)."""
        with _lock:
            m = self.shot_meta(shot_id)
            m.update(fields)
            _write_json(self.shot_dir(shot_id) / "meta.json", m)

    def rename_shot(self, shot_id: str, name: str) -> dict:
        with _lock:
            m = self.shot_meta(shot_id)
            m["name"] = name.strip() or m["name"]
            _write_json(self.shot_dir(shot_id) / "meta.json", m)
        return self.shot(shot_id)

    def move_shot(self, shot_id: str, project_id: str) -> dict:
        if not self.project_exists(project_id):
            raise FileNotFoundError(project_id)
        with _lock:
            m = self.shot_meta(shot_id)
            m["project_id"] = project_id
            _write_json(self.shot_dir(shot_id) / "meta.json", m)
        self.touch_project(project_id)
        return self.shot(shot_id)

    # --- versions ----------------------------------------------------------------------------------------------
    def new_version_dir(self, shot_id: str) -> tuple[str, Path]:
        vid = new_id("v")
        d = self.shot_dir(shot_id) / "versions" / vid
        d.mkdir(parents=True)
        return vid, d

    def save_version(self, shot_id: str, vid: str, meta: dict) -> dict:
        d = self.shot_dir(shot_id) / "versions" / vid
        meta = {"id": vid, "created_at": time.time()} | meta
        _write_json(d / "meta.json", meta)
        return self.version(shot_id, vid)

    def discard_version_dir(self, vdir: Path | None) -> bool:
        """A failed or canceled job's take folder goes, but only while it has no meta.json (a saved take is never
        touched) and PICKUPS_KEEP_WORK is unset. True when it was removed."""
        if vdir is None or keep_work():
            return False
        d = Path(vdir)
        shots = (self.root / "shots").resolve()
        if d.is_symlink() or not d.is_dir() or (d / "meta.json").exists() or not d.resolve().is_relative_to(shots):
            return False
        shutil.rmtree(d, ignore_errors=True)
        return not d.exists()

    def sweep_orphan_versions(self, min_age_s: float = 3600) -> int:
        """At start: take folders with no meta.json (left by a crash, or by an older build's failed jobs), older than
        min_age_s so a second running app's live job is never swept. Returns how many went."""
        if keep_work():
            return 0
        now, n = time.time(), 0
        for vdir in (self.root / "shots").glob("*/versions/*"):
            try:
                if vdir.is_dir() and not (vdir / "meta.json").exists() and now - vdir.stat().st_mtime >= min_age_s:
                    n += self.discard_version_dir(vdir)
            except OSError:
                continue
        return n

    # --- trash: shots, takes, projects, cast entries and images, under library/trash/<id>/; os.rename in and out ------
    @property
    def trash_root(self) -> Path:
        return self.root / "trash"

    def _trash_entry_dir(self, trash_id: str) -> Path:
        if not re.fullmatch(r"tr_[0-9a-f]+", str(trash_id)):
            raise FileNotFoundError(trash_id)
        return self.trash_root / trash_id

    def _into_trash(self, kind: str, sources: list[Path], meta: dict, thumb: Path | None = None) -> dict:
        """The one way into the trash. Each file or folder keeps its library path under trash/<id>/ (so it can also be put
        back by hand), with trash.json beside them; nothing is erased. Same volume, so a rename: instant and atomic. If
        one move fails, the ones already moved go back and the library is as it was."""
        tid = new_id("tr")
        entry = self._trash_entry_dir(tid)
        entry.mkdir(parents=True)
        rels = [src.relative_to(self.root).as_posix() for src in sources]
        moved: list[tuple[Path, Path]] = []
        try:
            for src, rel in zip(sources, rels):
                dst = entry / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, dst)
                moved.append((src, dst))
        except OSError:
            for src, dst in reversed(moved):
                os.rename(dst, src)
            shutil.rmtree(entry, ignore_errors=True)
            raise
        thumb_rel = thumb.relative_to(self.root).as_posix() if thumb is not None and thumb.is_relative_to(self.root) else None
        _write_json(entry / "trash.json", {"id": tid, "kind": kind, "path": rels[0], "paths": rels, "trashed_at": time.time(),
                                           "thumb_path": thumb_rel if thumb_rel and (entry / thumb_rel).is_file() else None} | meta)
        return self._trash_entry(entry)

    @staticmethod
    def _trash_items(entry: Path, meta: dict) -> list[Path]:
        if "paths" in meta:
            return [entry / p for p in meta["paths"]]
        return [entry / Path(meta["path"]).name]  # an entry from before Run 6: its one item at the top

    def trash(self, kind: str, shot_id: str, take_id: str | None = None) -> dict:
        """Move a take (kind "take") or a whole shot (kind "shot") into the trash. The Examples project and the original
        can't go."""
        shot = self.shot_meta(shot_id)
        if shot.get("project_id") == EXAMPLES:
            raise PermissionError("Examples can't be moved to the trash.")
        if kind == "take":
            if not take_id or take_id == "orig":
                raise PermissionError("The original stays with its shot. Move the shot to the trash instead.")
            src = self.version_dir(shot_id, take_id)
            if not (src / "meta.json").exists():
                raise FileNotFoundError(take_id)
            take = self.version(shot_id, take_id)
            name = f"{shot['name']} · {take.get('label') or take_id}"
        elif kind == "shot":
            src = self.shot_dir(shot_id)
            name = shot["name"]
        else:
            raise ValueError(kind)
        return self._into_trash(kind, [src], {"shot_id": shot_id, "take_id": take_id if kind == "take" else None, "name": name,
                                              "project_id": shot.get("project_id", UNSORTED)}, thumb=src / "thumb.jpg")

    def trash_project(self, project_id: str) -> dict:
        """Move a project to the trash: its folder (cast, images, deliveries) and every shot in it, takes and all. My shots
        comes back empty straight away, so there's always a project; Examples can't go."""
        if project_id == EXAMPLES:
            raise PermissionError("Examples can't be moved to the trash.")
        if not ITEM_ID.fullmatch(str(project_id)) or not self.project_exists(project_id):
            raise FileNotFoundError(project_id)
        name = json.loads((self.project_dir(project_id) / "meta.json").read_text()).get("name") or project_id
        shots = [self.shot_dir(s["id"]) for s in self.shots(project_id)]
        cover = next((d / "thumb.jpg" for d in shots if (d / "thumb.jpg").is_file()), None)
        entry = self._into_trash("project", [self.project_dir(project_id), *shots],
                                 {"project_id": project_id, "name": name, "shots": len(shots)}, thumb=cover)
        self._builtin_projects()
        return entry

    def _asset_list(self, project_id: str, kind: str) -> tuple[Path, Path]:
        """A cast entry's or an image's folder and the list it sits in (cast.json or images.json)."""
        d = self.project_dir(project_id)
        return (d / "cast", d / "cast.json") if kind == "cast" else (d / "images", d / "images.json")

    def trash_asset(self, project_id: str, kind: str, asset_id: str) -> dict:
        """Move a cast entry (kind "cast") or a project image (kind "image") into the trash: every file of it (picture,
        thumbnail, character sheet and views); its entry is kept in trash.json to put back. Takes that used it keep
        their settings; a reference that's gone is skipped."""
        if kind not in ("cast", "image") or not ITEM_ID.fullmatch(str(asset_id)) or not self.project_exists(project_id):
            raise FileNotFoundError(asset_id)
        d, f = self._asset_list(project_id, kind)
        with _lock:
            items = json.loads(f.read_text()) if f.exists() else []
            hit = next((x for x in items if x.get("id") == asset_id), None)
            if hit is None:
                raise FileNotFoundError(asset_id)
            files = sorted(x for x in d.iterdir() if x.is_file() and (x.name == hit.get("file") or x.name.startswith((f"{asset_id}.", f"{asset_id}_"))))
            thumb = d / f"{asset_id}_thumb.jpg"
            name = hit.get("name") if kind == "cast" else (str(hit.get("prompt") or "").strip()[:80] or "Image")
            entry = self._into_trash(kind, files, {"project_id": project_id, "asset_id": asset_id, "name": name, "entry": hit},
                                     thumb=thumb if thumb in files else d / str(hit.get("file")))
            _write_json(f, [x for x in items if x.get("id") != asset_id])
        self.touch_project(project_id)
        return entry

    def _trash_entry(self, entry: Path) -> dict:
        meta = json.loads((entry / "trash.json").read_text())
        items = [i for i in self._trash_items(entry, meta) if i.exists()]
        size = sum(_tree_bytes(i) if i.is_dir() else i.lstat().st_size for i in items)
        out = {**meta, "bytes": size, "remote_files": sum(_remote_names(i) for i in items if i.is_dir())}
        out.pop("entry", None)  # a cast entry's or image's own record: only Restore reads it
        thumb = entry / meta["thumb_path"] if meta.get("thumb_path") else items[0] / meta["thumb"] if meta.get("thumb") and items else None
        if thumb is not None and thumb.is_file():
            out["thumb_url"] = media_url(thumb)
        return out

    def trash_list(self) -> list[dict]:
        out = []
        if self.trash_root.is_dir():
            for entry in self.trash_root.iterdir():
                if entry.is_dir() and not entry.is_symlink() and (entry / "trash.json").exists():
                    try:
                        out.append(self._trash_entry(entry))
                    except (OSError, ValueError, KeyError):
                        continue
        return sorted(out, key=lambda e: -e["trashed_at"])

    def restore(self, trash_id: str) -> dict:
        """Put a trashed item back where it was. FileExistsError when something now sits there; FileNotFoundError when
        what it belongs to is gone (a take's shot, or a cast entry's or an image's project, is itself in the trash)."""
        entry = self._trash_entry_dir(trash_id)
        meta = json.loads((entry / "trash.json").read_text())
        if meta["kind"] == "project":
            self._restore_project(entry, meta)
        elif meta["kind"] in ("cast", "image"):
            self._restore_asset(entry, meta)
        else:
            dest = self._back_to(meta["path"], "shots")
            if dest.exists():
                raise FileExistsError(meta["path"])
            if not dest.parent.exists():
                raise FileNotFoundError(meta["shot_id"])
            os.rename(self._trash_items(entry, meta)[0], dest)
            if meta["kind"] == "shot":
                self.touch_project(meta.get("project_id") or UNSORTED)
        shutil.rmtree(entry, ignore_errors=True)
        meta.pop("entry", None)
        return meta

    def _back_to(self, rel: str, under: str) -> Path:
        """Where a trashed path goes back to: inside the library's `under` folder, never anywhere else."""
        dest = (self.root / rel).resolve()
        if not dest.is_relative_to((self.root / under).resolve()):
            raise PermissionError(rel)
        return dest

    def _restore_project(self, entry: Path, meta: dict) -> None:
        pid = meta["project_id"]
        folder, dest = entry / meta["paths"][0], self._back_to(meta["paths"][0], "projects")
        shots = [(entry / r, self._back_to(r, "shots")) for r in meta["paths"][1:]]
        if any(d.exists() for _, d in shots):
            raise FileExistsError(pid)
        merge = dest.exists()
        lists = ("meta.json", "cast.json", "images.json")
        if merge:
            # Only My shots comes back by itself once deleted: what's in it now stays, and the old things join it.
            if pid != UNSORTED:
                raise FileExistsError(pid)
            if any(f.is_file() and f.name not in lists and (dest / f.relative_to(folder)).exists() for f in folder.rglob("*")):
                raise FileExistsError(pid)
        for src, d in shots:
            os.rename(src, d)
        if not merge:
            os.rename(folder, dest)
        else:
            with _lock:
                for name in lists[1:]:
                    old = json.loads((folder / name).read_text()) if (folder / name).exists() else []
                    new = json.loads((dest / name).read_text()) if (dest / name).exists() else []
                    if old:
                        ids = {o.get("id") for o in old}
                        _write_json(dest / name, old + [x for x in new if x.get("id") not in ids])
                if (folder / "meta.json").exists():
                    os.replace(folder / "meta.json", dest / "meta.json")
                for f in sorted(folder.rglob("*")):
                    if f.is_file() and f.name not in lists:
                        (dest / f.relative_to(folder)).parent.mkdir(parents=True, exist_ok=True)
                        os.rename(f, dest / f.relative_to(folder))
        self.touch_project(pid)

    def _restore_asset(self, entry: Path, meta: dict) -> None:
        pid, kind, item = meta["project_id"], meta["kind"], meta["entry"]
        if not self.project_exists(pid):
            raise FileNotFoundError("Restore its project first.")
        moves = [(entry / r, self._back_to(r, "projects")) for r in meta["paths"]]
        if any(d.exists() for _, d in moves):
            raise FileExistsError(meta["path"])
        _d, f = self._asset_list(pid, kind)
        with _lock:
            items = json.loads(f.read_text()) if f.exists() else []
            if kind == "cast" and any(str(c.get("name", "")).lower() == str(item.get("name", "")).lower() for c in items):
                raise FileExistsError(item.get("name"))
            for src, d in moves:
                d.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, d)
            _write_json(f, sorted([*items, item], key=lambda x: x.get("created_at") or 0))
        self.touch_project(pid)

    def empty_trash(self) -> dict:
        """Remove everything in the trash from this computer. Each folder is resolved and checked to be inside the
        trash before it's removed, and symlinks are never followed. remote_files: names the ComfyUI machine still holds
        (from the takes' saved workflows); Pickups can't remove those."""
        removed = {"items": 0, "bytes": 0, "remote_files": 0}
        if not self.trash_root.is_dir():
            return removed
        root = self.trash_root.resolve()
        for entry in list(self.trash_root.iterdir()):
            if entry.is_symlink():
                entry.unlink()  # the link only, never what it points at
                continue
            target = entry.resolve()
            if not target.is_dir() or target == root or not target.is_relative_to(root):
                continue
            removed["bytes"] += _tree_bytes(target)
            removed["remote_files"] += _remote_names(target)
            shutil.rmtree(target)
            removed["items"] += 1
        return removed

    def storage(self) -> dict:
        """The library's folder and its total size (walks the tree: call it off the main thread)."""
        return {"path": str(self.root), "bytes": _tree_bytes(self.root)}

    def version(self, shot_id: str, vid: str) -> dict:
        d = self.shot_dir(shot_id) / "versions" / vid
        m = json.loads((d / "meta.json").read_text())
        # Change weather takes were once labelled "Environment · Snow"; they read "Weather · Snow" now.
        label = re.sub(r"^Environment · ", "Weather · ", m.get("label", vid))
        out = {"id": vid, "label": label, "task": m.get("task"), "created_at": m.get("created_at"),
               "video_url": media_url(d / "video.mp4"), "thumb_url": media_url(d / "thumb.jpg"),
               "downloads": [{"label": x["label"], "url": media_url(d / x["file"]), "codec": x.get("codec"), "size": x.get("size"),
                              "bytes": (d / x["file"]).stat().st_size}
                             for x in m.get("downloads", []) if (d / x["file"]).exists()],
               "details": m.get("details", {}), "licences": licences.licences_for(_models_on_read(m)),
               "status": m.get("status", ""), "notes": m.get("notes", []), "input_version": m.get("input_version"),
               "params": m.get("params", {}), "hidden": m.get("hidden", False),
               "workflows": [{"step": w["step"], "url": media_url(d / w["file"]), "filename": f"{vid}_{Path(w['file']).name}",
                              "title": w.get("title") or saved_title(d / w["file"], w["step"], (m.get("params") or {}).get("frame"))}
                             for w in m.get("workflows", []) if (d / w["file"]).exists()]}
        for extra in ("look.png", "first_frame.png", "reference.png"):  # reference.png: Recast's still made from a description
            if (d / extra).exists():
                out[extra.split(".")[0] + "_url"] = media_url(d / extra)
        if (d / "receipt.mp4").exists():
            out["receipt_url"] = media_url(d / "receipt.mp4")
        if m.get("receipt"):
            out["receipt"] = m["receipt"]
            if m.get("task") == "finish":  # older Finish takes said "100% AI" for an upscale: read them as they are
                from pickups.receipts import finish_receipt_on_read
                out["receipt"] = finish_receipt_on_read(m["receipt"], m.get("details"))
        if (d / "before.mp4").exists():
            out["before_url"] = media_url(d / "before.mp4")
        if isinstance(out["details"], dict) and isinstance(out["details"].get("models"), list):
            out["details"] = {**out["details"], "models": _models_on_read(m)}  # the corrected list, as the licences
        return out

    def versions(self, shot_id: str) -> list[dict]:
        vd = self.shot_dir(shot_id) / "versions"
        out = []
        for d in vd.iterdir():
            if (d / "meta.json").exists():
                out.append((d.stat().st_mtime, self.version(shot_id, d.name)))
        # In the order they were made: a folder's mtime moves whenever anything in it is written (a note, an approval,
        # a receipt still), and that must never renumber the takes.
        return [v for _, v in sorted(out, key=lambda x: x[1].get("created_at") or x[0])]

    def version_master(self, shot_id: str, vid: str | None) -> Path:
        """The full-quality input for a task: the original source, or a version's master."""
        if not vid or vid == "orig":
            return self.source_path(shot_id)
        return self.version_dir(shot_id, vid) / "master.mov"

    def version_dir(self, shot_id: str, vid: str) -> Path:
        """A take's folder (FileNotFoundError for an id that isn't a plain folder name; the folder may not exist)."""
        if not ITEM_ID.fullmatch(str(vid)):
            raise FileNotFoundError(vid)
        return self.shot_dir(shot_id) / "versions" / vid

    def _version_meta_path(self, shot_id: str, vid: str) -> Path:
        f = self.version_dir(shot_id, vid) / "meta.json"
        if not f.exists():
            raise FileNotFoundError(vid)
        return f

    def take_reference(self, shot_id: str, vid: str) -> Path | None:
        """A Recast take's reference still (the person FLUX 3 klein made from a description), or None: no such take, an
        id that isn't a plain folder name, or a take that kept no still."""
        try:
            f = self.version_dir(shot_id, str(vid)) / "reference.png"
        except FileNotFoundError:
            return None
        return f if f.is_file() else None

    def version_meta(self, shot_id: str, vid: str) -> dict:
        """A take's meta.json as saved (FileNotFoundError: no such take)."""
        return json.loads(self._version_meta_path(shot_id, vid).read_text())

    def update_take(self, shot_id: str, vid: str, status: str | None = None, hidden: bool | None = None) -> dict:
        """A take's status ("" or "approved"; the original can be approved too) and whether it is hidden (failed takes)."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with _lock:
            if vid == "orig":
                f, prefix = self.shot_dir(shot_id) / "meta.json", "orig_"
            else:
                f, prefix = self._version_meta_path(shot_id, vid), ""
            m = json.loads(f.read_text())
            if status is not None:
                m[prefix + "status"] = status
            if hidden is not None and vid != "orig":
                m["hidden"] = bool(hidden)
            _write_json(f, m)
        self.touch_project(self.shot_meta(shot_id).get("project_id", UNSORTED))
        return self.shot(shot_id)

    def set_status(self, shot_id: str, vid: str, status: str) -> dict:
        return self.update_take(shot_id, vid, status=status)

    def add_note(self, shot_id: str, vid: str, frame: int, text: str) -> dict:
        note = {"id": new_id("n"), "frame": int(frame), "text": text.strip(), "created_at": time.time()}
        with _lock:
            if vid == "orig":
                m = self.shot_meta(shot_id)
                m.setdefault("orig_notes", []).append(note)
                _write_json(self.shot_dir(shot_id) / "meta.json", m)
            else:
                f = self._version_meta_path(shot_id, vid)
                m = json.loads(f.read_text())
                m.setdefault("notes", []).append(note)
                _write_json(f, m)
        return note

    def delete_note(self, shot_id: str, vid: str, note_id: str) -> None:
        with _lock:
            if vid == "orig":
                f = self.shot_dir(shot_id) / "meta.json"
                key = "orig_notes"
            else:
                f = self._version_meta_path(shot_id, vid)
                key = "notes"
            m = json.loads(f.read_text())
            m[key] = [n for n in m.get(key, []) if n["id"] != note_id]
            _write_json(f, m)

    # --- projects and cast -------------------------------------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        return self.root / "projects" / project_id

    def project_exists(self, project_id: str) -> bool:
        return (self.project_dir(project_id) / "meta.json").exists()

    def _write_project(self, project_id: str, meta: dict) -> None:
        d = self.project_dir(project_id)
        (d / "cast").mkdir(parents=True, exist_ok=True)
        _write_json(d / "meta.json", meta)

    def create_project(self, name: str) -> dict:
        pid = new_id("prj")
        now = time.time()
        self._write_project(pid, {"id": pid, "name": name.strip() or "Untitled project", "created_at": now, "updated_at": now,
                                  "archived": False})
        return self.project(pid)

    def qc_project(self) -> str:
        """The archived "Test runs" project, made the first time a test run saves something."""
        if not self.project_exists(QC):
            now = time.time()
            self._write_project(QC, {"id": QC, "name": "Test runs", "created_at": now, "updated_at": now, "archived": True})
        return QC

    def touch_project(self, project_id: str) -> None:
        if not self.project_exists(project_id):
            return
        with _lock:
            f = self.project_dir(project_id) / "meta.json"
            m = json.loads(f.read_text())
            m["updated_at"] = time.time()
            _write_json(f, m)

    def update_project(self, project_id: str, name: str | None = None, archived: bool | None = None) -> dict:
        with _lock:
            f = self.project_dir(project_id) / "meta.json"
            m = json.loads(f.read_text())
            if name is not None and name.strip():
                m["name"] = name.strip()
            if archived is not None and project_id not in (UNSORTED, EXAMPLES):
                m["archived"] = bool(archived)
            m["updated_at"] = time.time()
            _write_json(f, m)
        return self.project(project_id)

    def project(self, project_id: str, shots: list[dict] | None = None) -> dict:
        m = json.loads((self.project_dir(project_id) / "meta.json").read_text())
        shots = self.shots(project_id) if shots is None else shots
        takes = sum(len(s["versions"]) - 1 for s in shots)
        latest = max(shots, key=lambda s: s["versions"][-1].get("created_at") or 0, default=None)
        return {"id": m["id"], "name": m["name"], "created_at": m["created_at"], "updated_at": m.get("updated_at", m["created_at"]),
                "archived": m.get("archived", False), "builtin": project_id in (UNSORTED, EXAMPLES),
                "shots": len(shots), "takes": takes, "approved": sum(s["approved"] for s in shots),
                "cover_url": latest["latest_thumb_url"] if latest else (self.images(project_id)[:1] or [{}])[0].get("thumb_url"),
                "cover_video_url": latest["versions"][-1]["video_url"] if latest else None,  # plays when the card is hovered
                "cast": len(self.cast(project_id)), "images": len(self.images(project_id))}

    def projects(self) -> list[dict]:
        all_shots = self.shots()
        out = []
        for d in (self.root / "projects").iterdir():
            if (d / "meta.json").exists():
                p = self.project(d.name, [s for s in all_shots if s["project_id"] == d.name])
                # "Try it" copies examples into My shots, so Examples only lists in libraries that still hold some.
                if p["id"] == EXAMPLES and not (p["shots"] or p["images"] or p["cast"]):
                    continue
                out.append(p)
        return sorted(out, key=lambda p: (p["id"] == EXAMPLES, -p["updated_at"]))

    def cast(self, project_id: str) -> list[dict]:
        f = self.project_dir(project_id) / "cast.json"
        items = json.loads(f.read_text()) if f.exists() else []
        d = self.project_dir(project_id) / "cast"
        out = []
        for c in items:
            if not (d / c["file"]).exists():
                continue
            thumb = d / f"{c['id']}_thumb.jpg"
            sheet = c.get("sheet") if isinstance(c.get("sheet"), dict) else None
            if not thumb.exists():  # character sheets are big PNGs; grids and chips use a small JPEG
                # A sheet built here: chips show its front view, not the middle (back) panel of the composite.
                front = d / str((sheet or {}).get("view_files", {}).get("front", ""))
                src = front if sheet and front.is_file() else d / c["file"]
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale='min(720,iw)':-2", "-q:v", "3",
                                str(thumb)], check=False)
            extra = {"sheet": self._sheet_urls(d, sheet)} if sheet else {}
            out.append(c | {"url": media_url(d / c["file"]), "thumb_url": media_url(thumb) if thumb.exists() else media_url(d / c["file"])} | extra)
        return out

    @staticmethod
    def _sheet_urls(d: Path, sheet: dict) -> dict:
        """A cast entry's character sheet as the API sends it: what's on disk plus url, thumb_url, the views' urls and the
        licences of the models that made it (as a take's)."""
        small = d / str(sheet.get("small", ""))
        return sheet | {"url": media_url(d / sheet["file"]), "thumb_url": media_url(small if small.is_file() else d / sheet["file"]),
                        "views": {k: media_url(d / f) for k, f in sheet.get("view_files", {}).items() if (d / f).is_file()},
                        "licences": licences.licences_for(sheet.get("models"))}

    def _write_cast(self, project_id: str, items: list[dict]) -> None:
        def on_disk(c: dict) -> dict:
            c = {k: v for k, v in c.items() if k not in ("url", "thumb_url")}
            if isinstance(c.get("sheet"), dict):  # urls are worked out on read
                c["sheet"] = {k: v for k, v in c["sheet"].items() if k not in ("url", "thumb_url", "views", "licences")}
            return c
        _write_json(self.project_dir(project_id) / "cast.json", [on_disk(c) for c in items])

    def add_cast(self, project_id: str, data: bytes, filename: str, name: str, kind: str = "person", line: str = "") -> dict:
        if kind not in CAST_KINDS:
            raise ValueError(f"kind must be one of {CAST_KINDS}")
        name = " ".join(name.split()) or "Untitled"
        cid = new_id("cast")
        ext = Path(filename).suffix.lower() or ".png"
        d = self.project_dir(project_id) / "cast"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cid}{ext}").write_bytes(data)
        with _lock:
            items = self.cast(project_id)
            if any(c["name"].lower() == name.lower() for c in items):
                raise ValueError(f"There's already a cast entry called {name}.")
            items.append({"id": cid, "name": name, "kind": kind, "line": line.strip(), "file": f"{cid}{ext}", "created_at": time.time()})
            self._write_cast(project_id, items)
        self.touch_project(project_id)
        return next(c for c in self.cast(project_id) if c["id"] == cid)

    def update_cast(self, project_id: str, cast_id: str, **fields) -> dict:
        with _lock:
            items = self.cast(project_id)
            c = next((c for c in items if c["id"] == cast_id), None)
            if c is None:
                raise FileNotFoundError(cast_id)
            for k in ("name", "kind", "line"):
                if fields.get(k) is not None:
                    c[k] = " ".join(str(fields[k]).split()) if k == "name" else str(fields[k]).strip()
            if c["kind"] not in CAST_KINDS:
                raise ValueError(f"kind must be one of {CAST_KINDS}")
            self._write_cast(project_id, items)
        return next(c for c in self.cast(project_id) if c["id"] == cast_id)

    def import_cast(self, project_id: str, from_project_id: str, cast_id: str) -> dict:
        """Reuse a cast entry from another project (a copy: editing one doesn't change the other)."""
        src = next((c for c in self.cast(from_project_id) if c["id"] == cast_id), None)
        if src is None:
            raise FileNotFoundError(cast_id)
        base = self.project_dir(from_project_id) / "cast"
        f = base / src["file"]
        new = self.add_cast(project_id, f.read_bytes(), f.name, src["name"], src.get("kind", "person"), src.get("line", ""))
        sheet = src.get("sheet")
        if isinstance(sheet, dict) and sheet.get("view_files"):  # its character sheet comes along, views and all
            source = base / str(sheet.get("source_file") or "")
            new = self.save_sheet(project_id, {k: base / v for k, v in sheet["view_files"].items()}, base / sheet["file"],
                                  cast_id=new["id"], source=source if source.is_file() else None, meta=sheet)
        return new

    def save_sheet(self, project_id: str, views: dict, sheet: Path, cast_id: str | None = None, name: str | None = None,
                   line: str = "", source: Path | None = None, meta: dict | None = None) -> dict:
        """Keep a character sheet (sheets.py) with a cast member: a new person called `name`, or `cast_id` upgraded (or
        made again). The composite becomes the member's picture (Recast cuts out its front view, image models read the
        whole sheet); the front, back and close-up PNGs are kept beside it as `sheet.view_files`. `source`, the picture
        it was built from, is copied in for a new member; an existing member keeps its own picture as the source, so
        a redo starts again from it. An earlier sheet's files are removed."""
        d = self.project_dir(project_id) / "cast"
        d.mkdir(parents=True, exist_ok=True)
        ver = uuid.uuid4().hex[:6]  # new file names each time, so no browser shows a cached older sheet
        reserved = ("file", "small", "view_files", "source_file", "url", "thumb_url", "views", "licences", "created_at")
        with _lock:
            items = self.cast(project_id)
            if cast_id:
                c = next((x for x in items if x["id"] == cast_id), None)
                if c is None:
                    raise FileNotFoundError(cast_id)
            else:
                name = " ".join(str(name or "").split())
                if not name:
                    raise ValueError("Give them a name.")
                if any(x["name"].lower() == name.lower() for x in items):
                    raise ValueError(f"There's already a cast entry called {name}.")
                c = {"id": new_id("cast"), "name": name, "kind": "person", "line": "", "created_at": time.time()}
                items.append(c)
            cid = c["id"]
            old = c.get("sheet") if isinstance(c.get("sheet"), dict) else {}
            old_files = {c.get("file"), old.get("file"), old.get("small"), *old.get("view_files", {}).values()} - {None}
            if source is not None:
                keep = f"{cid}_source_{ver}{Path(source).suffix.lower() or '.png'}"
                shutil.copy2(source, d / keep)
            else:
                keep = old.get("source_file") or c.get("file")
            files = {k: f"{cid}_{k}_{ver}.png" for k in views}
            for k, f in files.items():
                shutil.copy2(views[k], d / f)
            sheet_file, small = f"{cid}_sheet_{ver}.jpg", f"{cid}_sheet_{ver}_small.jpg"
            shutil.copy2(sheet, d / sheet_file)
            try:  # the cast grid shows the sheet at card size
                from PIL import Image as _Image
                with _Image.open(d / sheet_file) as im:
                    im = im.convert("RGB")
                    im.thumbnail((960, 960))
                    im.save(d / small, quality=85)
            except Exception:  # noqa: BLE001 - the full sheet stands in
                small = None
            c["file"] = sheet_file
            c["sheet"] = {"file": sheet_file, **({"small": small} if small else {}), "view_files": files, "source_file": keep,
                          **{k: v for k, v in (meta or {}).items() if k not in reserved}, "created_at": time.time()}
            if line.strip() and not c.get("line"):
                c["line"] = line.strip()
            self._write_cast(project_id, items)
        for f in old_files - {keep, sheet_file, small, *files.values()}:
            (d / f).unlink(missing_ok=True)
        (d / f"{cid}_thumb.jpg").unlink(missing_ok=True)  # made again, from the new front view
        self.touch_project(project_id)
        return next(x for x in self.cast(project_id) if x["id"] == cid)

    def sheet_source(self, project_id: str, cast_id: str) -> Path:
        """The picture a cast member's sheet is built from: the one its current sheet came from, else its own picture."""
        c = next((x for x in self.cast(project_id) if x["id"] == cast_id), None)
        if c is None:
            raise FileNotFoundError(cast_id)
        d = self.project_dir(project_id) / "cast"
        src = d / str((c.get("sheet") or {}).get("source_file") or c["file"])
        return src if src.is_file() else d / c["file"]

    def cast_uses(self, project_id: str, shots: list[dict] | None = None) -> dict[str, int]:
        """How many shots of the project use each cast entry (an @-mention or a reference in any take's settings)."""
        cast = self.cast(project_id)
        shots = self.shots(project_id) if shots is None else shots
        uses = {c["id"]: 0 for c in cast}
        for shot in shots:
            blob = json.dumps([v.get("params", {}) for v in shot["versions"]]).lower()
            for c in cast:
                if f"@{c['name'].lower()}" in blob or c["id"] in blob:
                    uses[c["id"]] += 1
        return uses

    def image_uses(self, project_id: str, shots: list[dict] | None = None) -> dict[str, int]:
        """How many shots of the project use each project image (a reference in any take's settings)."""
        ids = [i["id"] for i in self.images(project_id)]
        shots = self.shots(project_id) if shots is None else shots
        uses = dict.fromkeys(ids, 0)
        for shot in shots:
            blob = json.dumps([v.get("params", {}) for v in shot["versions"]])
            for i in ids:
                uses[i] += i in blob
        return uses

    def cast_path(self, project_id: str, cast_id: str) -> Path:
        c = next((c for c in self.cast(project_id) if c["id"] == cast_id), None)
        if c is None:
            raise FileNotFoundError(cast_id)
        return self.project_dir(project_id) / "cast" / c["file"]

    def shot_cast(self, shot_id: str) -> tuple[list[dict], callable]:
        """The cast of a shot's project, and a function mapping a cast entry to its image file."""
        pid = self.shot_meta(shot_id).get("project_id", UNSORTED)
        return self.cast(pid), (lambda c: self.project_dir(pid) / "cast" / c["file"])

    # --- generated images -------------------------------------------------------------------------------------
    def images(self, project_id: str) -> list[dict]:
        f = self.project_dir(project_id) / "images.json"
        items = json.loads(f.read_text()) if f.exists() else []
        d = self.project_dir(project_id) / "images"
        out = []
        for im in reversed(items):  # newest first
            if (d / im["file"]).exists():
                thumb = d / f"{im['id']}_thumb.jpg"
                out.append(im | {"url": media_url(d / im["file"]), "thumb_url": media_url(thumb) if thumb.exists() else media_url(d / im["file"])})
        return out

    def add_image(self, project_id: str, src: Path, meta: dict) -> dict:
        """Keep a generated image (PNG) in the project, with a small JPEG for grids, and its recipe (model, prompt...)."""
        iid = new_id("img")
        d = self.project_dir(project_id) / "images"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{iid}{src.suffix.lower() or '.png'}"
        shutil.copy2(src, dst)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-vf", "scale='min(640,iw)':-2", "-q:v", "3",
                        str(d / f"{iid}_thumb.jpg")], check=False)
        try:
            from PIL import Image as _Image
            with _Image.open(dst) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            w = h = None
        entry = {"id": iid, "file": dst.name, "width": w, "height": h, "created_at": time.time()} | {
            k: v for k, v in meta.items() if k not in ("id", "file", "url", "thumb_url")}
        with _lock:
            f = self.project_dir(project_id) / "images.json"
            items = json.loads(f.read_text()) if f.exists() else []
            items.append(entry)
            _write_json(f, items)
        self.touch_project(project_id)
        return next(i for i in self.images(project_id) if i["id"] == iid)

    def image_path(self, project_id: str, image_id: str) -> Path:
        im = next((i for i in self.images(project_id) if i["id"] == image_id), None)
        if im is None:
            raise FileNotFoundError(image_id)
        return self.project_dir(project_id) / "images" / im["file"]

    def image_to_cast(self, project_id: str, image_id: str, name: str, kind: str = "person", line: str = "") -> dict:
        src = self.image_path(project_id, image_id)
        return self.add_cast(project_id, src.read_bytes(), src.name, name, kind, line)

    def ref_path(self, project_id: str | None, ref: dict) -> Path:
        """A reference picked in the UI: {"kind": "cast" | "image" | "upload", "id"}."""
        kind, rid = ref.get("kind"), ref.get("id")
        if kind == "cast" and project_id:
            return self.cast_path(project_id, rid)
        if kind == "image" and project_id:
            return self.image_path(project_id, rid)
        if kind == "upload":
            return self.upload_path(rid)
        raise FileNotFoundError(f"{kind}:{rid}")

    def live_refs(self, project_id: str | None, refs: list) -> list[Path]:
        """The pictures of a list of picked references, skipping any that's gone (a deleted cast entry or image): the
        take just goes without it."""
        out = []
        for r in refs or []:
            try:
                out.append(self.ref_path(project_id, r))
            except (FileNotFoundError, AttributeError):
                continue
        return out

    def friendly_master(self, shot_id: str, vid: str) -> Path:
        """The take's master under its self-describing name (a hard link, no extra space), for NLE timelines and bins."""
        master = self.version_master(shot_id, vid)
        shot = self.shot(shot_id)
        n = next(i for i, v in enumerate(shot["versions"]) if v["id"] == vid) + 1
        take = shot["versions"][n - 1]
        name = f"{_safe(shot['name'])}_T{n:02d}_{take.get('task') or 'original'}{master.suffix.lower()}"
        link = master.with_name(name)
        if not link.exists():
            try:
                os.link(master, link)
            except OSError:
                return master
        return link

    def deliveries(self, project_id: str) -> list[dict]:
        """Approved takes of a project, newest first, with their files."""
        out = []
        for shot in self.shots(project_id):
            for v in shot["versions"]:
                if v.get("status") == "approved":
                    out.append({"shot_id": shot["id"], "shot_name": shot["name"], "take": v})
        return sorted(out, key=lambda x: x["take"].get("created_at") or 0, reverse=True)

    def finish_candidates(self, project_id: str, jobs: list[dict] | None = None) -> list[dict]:
        """The approved takes "Finish all approved" queues: not Finish takes themselves, not already finished (a Finish
        take made from them that isn't hidden) and not being finished now (a queued or running Finish job in `jobs`)."""
        skip = {(j.get("shot_id"), j.get("version_id") or "orig") for j in jobs or []
                if j.get("task") == "finish" and j.get("status") in ("queued", "running")}
        skip |= {(s["id"], v.get("input_version") or "orig") for s in self.shots(project_id) for v in s["versions"]
                 if v.get("task") == "finish" and not v.get("hidden")}
        return [d for d in self.deliveries(project_id)
                if d["take"].get("task") != "finish" and (d["shot_id"], d["take"]["id"]) not in skip]

    # --- uploads -----------------------------------------------------------------------------------------------
    def add_upload(self, data: bytes, filename: str) -> dict:
        ext = Path(filename).suffix.lower() or ".png"
        uid = new_id("up")
        path = self.root / "uploads" / f"{uid}{ext}"
        path.write_bytes(data)
        return {"id": uid, "url": media_url(path)}

    def upload_path(self, uid: str) -> Path:
        if not UPLOAD_ID.fullmatch(str(uid)):  # an id, never a path or a glob
            raise FileNotFoundError(uid)
        hits = list((self.root / "uploads").glob(f"{uid}.*"))
        if not hits:
            raise FileNotFoundError(uid)
        return hits[0]
