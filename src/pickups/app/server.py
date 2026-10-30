"""The Pickups web app: a small local server for the UI, the library and the job queue.

Run with `pickups` (see cli.py). Everything is local: media is served from ~/.pickups/library, jobs go to your
ComfyUI through the Comfy SDK.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pickups import media
from pickups.app import discover, examples_pack, doctor, machines, manifest, measured, requirements, servers
from pickups.app.comfy import Canceled, Connection, _get_json, load_settings, run_graph, save_settings
from pickups.app.jobs import JobManager, NotStoppable
from pickups.app.timings import kind_of
from pickups.app.store import ROOT, UNSORTED, UPLOAD_ID, Store, media_url
from pickups.tasks import TASKS
from pickups import features, mentions, models_catalog
from pickups import framing as framing_check
from pickups import graphs, seeds, start_frame
from pickups.app import previz_maker
from pickups.tasks import common, previz

WEB = Path(__file__).resolve().parent.parent / "web" / "dist"
# Source copy of the previs GLBs. Vite serves web/public in dev and copies it into dist on build.
PREVIZ_ASSETS = Path(__file__).resolve().parents[3] / "web" / "public" / "previz"
_GZIP_SUFFIXES = (".glb", ".json", ".js")
_LIBRARY_CACHE = b"public, max-age=3600"


def _previz_library(path: str) -> bool:
    return path.startswith("/previz/assets/") or path.startswith("/previz/data/")


def _with_library_cache(path: str, headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    if not _previz_library(path):
        return headers
    out = [(key, value) for key, value in headers if key.lower() != b"cache-control"]
    out.append((b"cache-control", _LIBRARY_CACHE))
    return out


class AssetGzip:
    """Gzip only the previs library and the script bundles. Video and images stay as they are."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.endswith(_GZIP_SUFFIXES):
            await self.app(scope, receive, send)
            return
        accept = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"accept-encoding":
                accept = value.decode("latin1")
                break
        if "gzip" not in accept:
            if not _previz_library(path):
                await self.app(scope, receive, send)
                return

            async def stamp(message):
                if message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": _with_library_cache(path, list(message.get("headers") or [])),
                    }
                await send(message)

            await self.app(scope, receive, stamp)
            return
        status = 200
        headers: list[tuple[bytes, bytes]] = []
        body = bytearray()
        started = False

        async def capture(message):
            nonlocal status, headers, started
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                started = True
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))

        await self.app(scope, receive, capture)
        if not started:
            return
        raw = bytes(body)
        already = any(key.lower() == b"content-encoding" for key, _ in headers)
        if status != 200 or already or len(raw) < 256:
            headers = _with_library_cache(path, headers)
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": raw})
            return
        packed = gzip.compress(raw, compresslevel=5)
        out = [(key, value) for key, value in headers if key.lower() not in (b"content-length", b"content-encoding")]
        out.append((b"content-encoding", b"gzip"))
        out.append((b"content-length", str(len(packed)).encode()))
        out.append((b"vary", b"Accept-Encoding"))
        out = _with_library_cache(path, out)
        await send({"type": "http.response.start", "status": status, "headers": out})
        await send({"type": "http.response.body", "body": packed})
SAMPLES = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "samples"
EXAMPLES = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups"))) / "examples"
# The public example gallery as a release asset (set when the repo is published; PICKUPS_EXAMPLES_URL overrides).
EXAMPLES_PACK_URL = ""
EXAMPLES_PACK_MB = 40
CANCELED = "Cancelled."


class QuickRunner:
    """Run one small graph right away (selection preview, look frame) outside the job queue."""

    def __init__(self, conn: Connection):
        self.conn = conn
        self._client = None
        self._gen = -1
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None or self._gen != self.conn.generation:  # rebuilt after a machine switch
            self._client = self.conn.client()
            self._gen = self.conn.generation
        return self._client

    def run(self, graph: dict):
        # An inline row's graphs (the start frame) find its Stopper on this thread (comfy.current_stopper), so Cancel
        # stops them; other quick runs have none.
        with self._lock:
            return run_graph(self.client, graph)


def attachment(project_name: str, suffix: str) -> dict:
    """Content-Disposition for a project download, "<Project><suffix>" with underscores for spaces: an ASCII name (a
    project named only in another script reads "Pickups") plus, when the name has other letters ("東京ロケ", "Café"),
    the name in filename* (RFC 5987), which browsers and the UI prefer."""
    from urllib.parse import quote
    plain = re.sub(r"[^A-Za-z0-9]+", "_", project_name).strip("_")[:60] or "Pickups"
    typed = re.sub(r"\W+", "_", project_name).strip("_")[:60] or plain
    header = f'attachment; filename="{plain}{suffix}"'
    return {"Content-Disposition": header + (f"; filename*=UTF-8''{quote(typed + suffix)}" if typed != plain else "")}


# Examples kept out of the gallery by id (car swap is out of this build; vehicles in other examples are fine).
CAR_SWAP_EXAMPLES = frozenset({"rc-car-offroader", "rc-car-convertible"})
# Fog isn't offered yet (the owner, Oct 22). The harbour pier relights were denied in the owner's review (Oct 23); their files stay.
# The K03 hood push previz: the driver raises his hand to his face mid-shot, which reads badly as a showcase (Oct 28).
HIDDEN_EXAMPLES = frozenset({"env-fleshmarket-fog", "rl-harbour-sunset", "rl-harbour-midday", "pv-k03-hood-push"})
# Examples that lead their tile wherever the pack lists them: the seafront pier previz opens Previz to final.
FIRST_EXAMPLES = ("pv-ocean-person",)


def first_examples(examples: list[dict]) -> list[dict]:
    """The examples with FIRST_EXAMPLES moved to the front, in that order; the rest keep the pack's order."""
    rank = {i: n for n, i in enumerate(FIRST_EXAMPLES)}
    return sorted(examples, key=lambda e: rank.get(e.get("id"), len(rank)))


def _reveal(target: Path, folder: bool = False) -> None:
    """Show a file (selected) or a folder in Finder / Explorer. Only meaningful when Pickups runs on this computer."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)] if folder else ["open", "-R", str(target)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(target)] if folder else ["explorer", "/select,", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target if folder else target.parent)])


def _media_file(url: str) -> Path | None:
    """The library file behind a /media/ URL (query and fragment dropped), or None if it isn't one or doesn't exist."""
    from urllib.parse import unquote, urlsplit

    path = unquote(urlsplit(url).path)
    if not path.startswith("/media/"):
        return None
    base = ROOT.resolve()
    target = (base / path[len("/media/"):]).resolve()
    if target != base and base not in target.parents:
        return None
    return target if target.is_file() else None


def create_app(comfy_url: str | None = None) -> FastAPI:
    app = FastAPI(title="Pickups", docs_url=None, redoc_url=None)
    app.add_middleware(AssetGzip)
    store = Store()
    try:  # take folders a crash or an older build left with no meta.json, older than an hour
        store.sweep_orphan_versions()
    except OSError:
        pass
    conn = Connection.from_settings(comfy_url)
    app.state.conn = conn  # the selected ComfyUI (tests point it at a fake one)
    jobs = JobManager(store, conn, TASKS)
    quick = QuickRunner(conn)
    state = {"status": None, "pack": {"busy": False, "error": None, "found": None}}

    # First run: a pack next to the app (the repo folder, or where it was started), in the app's home, or just downloaded
    # (Downloads, Desktop) installs by itself. Only while no examples are installed: it never replaces what is there.
    def pack_roots() -> list[Path]:
        repo = Path(__file__).resolve().parents[3]  # the pickups folder (a clone)
        home = Path.home()
        return [repo, repo.parent, Path.cwd(), EXAMPLES.parent, home / "Downloads", home / "Desktop"]

    def auto_pack() -> None:
        if state["pack"]["busy"]:
            return
        found = examples_pack.find_next_to([r for r in pack_roots() if r.is_dir()])
        state["pack"]["found"] = str(found) if found else None
        if found and not (EXAMPLES / "examples.json").exists():
            state["pack"]["busy"] = True
            try:
                examples_pack.install(found, EXAMPLES, SAMPLES)
            except Exception as e:  # noqa: BLE001 - a bad pack leaves the app as it was; Setup says why
                state["pack"]["error"] = str(e)
            finally:
                state["pack"]["busy"] = False

    if not os.environ.get("PYTEST_CURRENT_TEST"):  # tests never pick up a real pack lying next to the repo
        threading.Thread(target=auto_pack, daemon=True).start()

    def refresh_status() -> dict:
        # max_frames: the video models work on at most this many frames from the start of a shot (4 s at 24 fps), so the
        # UI can say "Works on the first 4 s" from the shot's own frame rate. max_long and min_frames: the largest size
        # work is done at (long side) and the shortest clip a task takes, for the source limits shown when adding a clip.
        st = doctor.status(conn)
        vram = (st.get("machine") or {}).get("vram_gb")
        common.use_gpu(vram)
        state["status"] = st | {"comfy_queue": state.get("queue"), "max_frames": common.MAX_FRAMES,
                                "max_frames_by_task": {k: common.task_frames(k) for k in common.TASK_FRAMES},
                                "previz_frames": common.previz_frames(vram), "max_long": common.MAX_LONG,
                                "min_frames": media.MIN_FRAMES}
        return state["status"]

    def watch_comfy_queue() -> None:
        """ComfyUI can be busy with work Pickups didn't send (another app, or a job left over from before a restart, maybe
        hung). The status bar says so instead of "Ready", since new renders wait behind it."""
        while True:
            time.sleep(15)
            try:
                q = _get_json(f"{conn.comfy_url}/queue")
                running, pending = len(q.get("queue_running", [])), len(q.get("queue_pending", []))
            except Exception:  # noqa: BLE001 - offline is reported by the status check
                running = pending = 0
            ours = any(j.get("status") == "running" for j in jobs.list())
            busy = {"running": running, "pending": pending, "outside": bool(running and not ours)}
            if busy != state.get("queue"):
                state["queue"] = busy
                if state["status"]:
                    state["status"] = state["status"] | {"comfy_queue": busy}
                    jobs.broadcast("status", state["status"])

    threading.Thread(target=watch_comfy_queue, daemon=True, name="pickups-comfy-queue").start()

    # --- status / setup ------------------------------------------------------------------------------------------
    @app.get("/api/status")
    def api_status():
        return state["status"] or refresh_status()

    @app.post("/api/status/refresh")
    def api_status_refresh():
        return refresh_status()

    def use_url(url: str, save) -> dict:
        """Switch to a ComfyUI (refused while a job runs), save the choice, then refresh the status bar and tiles for
        its GPU (previz length and the verdict depend on its VRAM). Saved even if the proxy fails to start: the app
        now points there."""
        try:
            machines.switch(conn, jobs.list(), url)
        except machines.Busy as e:
            raise HTTPException(409, str(e)) from None
        except RuntimeError as e:  # the API proxy couldn't start
            save()
            jobs.broadcast("status", refresh_status())
            raise HTTPException(502, str(e)) from None
        save()
        jobs.broadcast("status", refresh_status())
        return state["status"]

    @app.post("/api/connect")
    async def api_connect(req: Request):
        body = await req.json()
        url = (body.get("comfy_url") or "").strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "Use a full address, like http://127.0.0.1:8188")
        def save():
            st = load_settings()
            ms, active = machines.normalize(st)
            # The typed address replaces the active machine's (or picks a saved machine with that address).
            same = next((m for m in ms if m["url"] == url), None)
            if same:
                active = same["id"]
            else:
                ms = [m | {"url": url, "kind": machines.kind_of(url)} if m["id"] == active else m for m in ms]
            save_settings(machines.with_machines(st, ms, active))

        return use_url(url, save)

    # --- first run: find, test and pick a ComfyUI; what this machine can do -----------------------------------------
    FEATURES = (("recast", "Replace a person"), ("remove", "Remove a person"), ("environment", "Change the weather"),
                ("previz", "Previz to final"), ("relight", "Relight"), ("finish", "Finish"),
                ("image", "New image"), ("video", "New video"))
    AUTO_NAMES = {"This computer", "GPU PC", "RunPod"}  # names the app gave a machine itself (machines.normalize)

    pinned = bool(comfy_url or os.environ.get("PICKUPS_COMFY_URL"))  # --comfy / the env var: the server is already chosen

    def servers_payload() -> dict:
        st = load_settings()
        return {"active": {"url": conn.comfy_url, "kind": machines.kind_of(conn.comfy_url)},
                "recent": servers.recents(st), "welcome_done": bool(st.get("welcome_done")) or pinned,
                "listen": servers.listen_commands()}

    @app.get("/api/servers")
    def api_servers():
        return servers_payload()

    @app.post("/api/servers/test")
    async def api_servers_test(req: Request):
        """Test an address without saving it. The answer says why it failed and how to start ComfyUI for other computers."""
        body = await req.json()
        try:
            url = servers.address(body.get("address", ""))
        except ValueError as e:
            return {"ok": False, "url": None, "reason": "address", "message": str(e), "listen": servers.listen_commands()}
        res = await asyncio.to_thread(servers.test, url)
        return res if res["ok"] else res | {"listen": servers.listen_commands(urlparse_port(url))}

    def urlparse_port(url: str) -> int:
        from urllib.parse import urlparse
        return urlparse(url).port or servers.PORT

    @app.get("/api/servers/detect")
    async def api_servers_detect():
        return await asyncio.to_thread(servers.detect) | {"listen": servers.listen_commands()}

    @app.get("/api/servers/scan")
    async def api_servers_scan(patient: int = 0):
        """ComfyUI on this computer's private networks and Tailscale devices (discover.py: what it may touch).
        patient=1: the second look after one that found nothing, waiting longer for each computer (1.5 s)."""
        return await asyncio.to_thread(discover.scan, timeout=discover.PATIENT_TIMEOUT if patient else discover.TIMEOUT)

    @app.post("/api/servers/use")
    async def api_servers_use(req: Request):
        """Connect to a server: test it, switch to it (refused while a job runs), remember it in the recent five and
        mark the welcome as seen. A server that doesn't answer isn't switched to."""
        body = await req.json()
        try:
            url = servers.address(body.get("address", ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        res = await asyncio.to_thread(servers.test, url)
        if not res["ok"]:
            raise HTTPException(409, res["message"] or "That server didn't answer.")

        def save():
            st = load_settings()
            ms, active = machines.normalize(st)
            same = next((m for m in ms if m["url"] == url), None)
            if same:
                active = same["id"]
                if same["name"] in AUTO_NAMES or same["name"].startswith("RunPod "):  # ours, not one they typed
                    ms = [m | {"name": servers.label(url)} if m["id"] == active else m for m in ms]
            else:  # the active machine now points here, and is named for it (the status pill shows the name)
                ms = [m | {"url": url, "kind": machines.kind_of(url), "name": servers.label(url)} if m["id"] == active else m
                      for m in ms]
            st = machines.with_machines(st, ms, active)
            save_settings(servers.remember(st, url, res.get("gpu")) | {"welcome_done": True})

        status = await asyncio.to_thread(use_url, url, save)
        return {"status": status, "test": res} | servers_payload()

    @app.post("/api/servers/forget")
    async def api_servers_forget(req: Request):
        url = ((await req.json()).get("url") or "").rstrip("/")
        st = load_settings()
        save_settings(st | {"recent_servers": [r for r in servers.recents(st) if r["url"] != url]})
        return servers_payload()

    @app.post("/api/welcome")
    def api_welcome_done():
        save_settings(load_settings() | {"welcome_done": True})
        return servers_payload()

    @app.get("/api/capabilities")
    def api_capabilities():
        """What the connected machine can do, per feature: models and nodes in place, and the numbers Pickups has
        measured on this GPU (none when it hasn't: the UI says "not tested on this GPU", never a guess)."""
        stats = conn.system_stats()
        status = state["status"] or refresh_status()
        if not stats:
            return {"connected": False, "server": {"url": conn.comfy_url, "kind": machines.kind_of(conn.comfy_url)}, "features": []}
        machine = servers.facts(stats)
        tiles = status.get("tiles") or {}
        out = []
        for task, label in FEATURES:
            t = tiles.get(task) or {}
            missing = t.get("missing") or []
            gone = {m["file"] for m in missing}
            unknown = {m["file"] for m in t.get("unknown") or []}
            core = t.get("missing_core_nodes") or []
            blocked = "vram" if t.get("vram_needed_gb") else None

            def needs(files, unknown=unknown, gone=gone):
                return [manifest.model_info(f) | {"installed": None if f in unknown else f not in gone} for f in files if f in manifest.MODELS]

            row = {"task": task, "label": label, "ready": bool(t.get("ready")), "reason": t.get("reason"),
                   "support": t.get("support") or requirements.support(task, stats),
                   "vram_gb": requirements.vram_needed(task), "headroom_gb": requirements.headroom([task]),
                   "needs": needs(manifest.task_models(bool(load_settings().get("commercial_safe"))).get(task, [])),
                   "missing_gb": round(sum(float(m.get("gb") or 0) for m in missing), 1),
                   # a list of packs, as before; core misses and the split view sit beside it (agents read nodes)
                   "missing_nodes": t.get("missing_nodes") or [],
                   "nodes": {"checked": bool(t.get("nodes_checked", True)), "core": core,
                             "update": manifest.UPDATE_COMFY if core else None, "packs": t.get("missing_nodes") or []},
                   "blocked": blocked, "vram_needed_gb": t.get("vram_needed_gb"),
                   "measured": measured.for_task(machine.get("gpu_raw"), task)}
            if t.get("choices"):  # Finish: what each choice needs (4K alone is SeedVR2 only)
                row["choices"] = {c: {"ready": v["ready"] and bool(t.get("ready") or t.get("reason") == "models"),
                                      "needs": needs(manifest.FINISH_CHOICE_MODELS[c]),
                                      "missing_gb": round(sum(float(m.get("gb") or 0) for m in v["missing"]), 1)}
                                  for c, v in t["choices"].items()}
            out.append(row)
        dirs = comfy_models_dirs()
        here = models_dir_here()
        return {"connected": True, "server": {"url": conn.comfy_url, "kind": machines.kind_of(conn.comfy_url)},
                "machine": machine, "measured_gpu": measured.gpu_key(machine.get("gpu_raw")), "features": out,
                "comfy_version": machine.get("comfy_version") or (status.get("machine") or {}).get("comfy_version"),
                "models_dir": comfy_models_dir(), "models_dirs": dirs, "models_dir_here": str(here or "") or None,
                "downloads": {"here": bool(here), "command": download_command()},
                "requirements": requirements_block(stats, here)}

    def requirements_block(stats: dict, here: Path | None) -> dict:
        """This computer against the minimum (SYSTEM_REQUIREMENTS.md): found values beside the rule, no prose."""
        f = servers.facts(stats)
        return {"gpu": f.get("gpu"), "nvidia": requirements.is_nvidia(stats), "vram_gb": requirements.vram_gb(stats),
                "min_vram_gb": requirements.MIN_VRAM_GB, "comfy_version": f.get("comfy_version"),
                "comfy_ok": requirements.comfy_ok(stats), "min_comfy": ".".join(map(str, requirements.MIN_COMFY)),
                # free space where downloads land (only when that's this computer) and where the library lives
                "models_free_gb": requirements.disk_free_gb(here) if here else None,
                "library_free_gb": requirements.disk_free_gb(ROOT), "library_min_free_gb": requirements.LIBRARY_MIN_FREE_GB,
                "ffmpeg": requirements.ffmpeg_major(), "min_ffmpeg": requirements.MIN_FFMPEG}

    # --- ComfyUI pass-through: the selected server's own pages, API, uploads, /view and websocket under /comfy/ -------
    # The browser never needs to reach ComfyUI itself (no CORS, no Tailscale name to resolve on this device).
    HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade", "accept-encoding",
           "proxy-connection", "te", "trailer"}
    # Not passed on: the agent is ours, and ComfyUI's cross-origin guard compares Origin with its own host, which this
    # browser's page (Pickups' address) never matches.
    BROWSER_ONLY = {"user-agent", "origin", "referer", "cookie"}

    @app.api_route("/comfy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def comfy_proxy(path: str, req: Request):
        import httpx

        target = f"{conn.comfy_url}/{path}"
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP | BROWSER_ONLY} | servers.UA
        # One pooled client for every pass-through: ComfyUI's page loads ~100 files at once.
        client = getattr(app.state, "comfy_http", None)
        if client is None:
            client = app.state.comfy_http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), follow_redirects=False,
                                                              limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
        content = await req.body()
        up = None
        for attempt in range(2 if req.method in ("GET", "HEAD") else 1):  # a dropped keep-alive is worth one retry
            try:
                up = await client.send(client.build_request(req.method, target, params=req.query_params, headers=headers,
                                                            content=content), stream=True)
                break
            except httpx.HTTPError as e:
                if attempt or req.method not in ("GET", "HEAD"):
                    raise HTTPException(502, f"ComfyUI at {conn.comfy_url} didn't answer ({type(e).__name__}).") from None

        async def body():
            try:
                async for chunk in up.aiter_raw():
                    yield chunk
            finally:
                await up.aclose()
        keep = {k: v for k, v in up.headers.items() if k.lower() not in HOP | {"content-encoding"}}
        return StreamingResponse(body(), status_code=up.status_code, headers=keep)

    @app.websocket("/comfy/ws")
    async def comfy_ws(ws: WebSocket):
        import websockets

        await ws.accept()
        target = conn.comfy_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        if ws.url.query:
            target += f"?{ws.url.query}"
        try:
            async with websockets.connect(target, max_size=None, additional_headers=servers.UA) as up:
                async def down():
                    async for msg in up:
                        await (ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_text(msg))

                async def upward():
                    while True:
                        m = await ws.receive()
                        if m.get("type") == "websocket.disconnect":
                            return
                        await up.send(m["bytes"] if m.get("bytes") is not None else m.get("text", ""))
                tasks = [asyncio.create_task(down()), asyncio.create_task(upward())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
        except Exception:  # noqa: BLE001 - a dropped or unreachable server closes this socket too
            pass
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass

    # --- machines (this computer, a LAN PC, a RunPod pod) ---------------------------------------------------------
    def machines_payload(probe: bool = True) -> dict:
        ms, active = machines.normalize(load_settings())
        stats = machines.probe_all(ms) if probe else {}
        return {"active": active, "machines": [m | {"status": stats.get(m["id"])} for m in ms]}

    @app.get("/api/machines")
    def api_machines():
        return machines_payload()

    @app.post("/api/machines")
    async def api_machine_add(req: Request):
        body = await req.json()
        kind = body.get("kind") if body.get("kind") in machines.KINDS else "lan"
        try:
            if kind == "runpod":
                url = machines.runpod_url(str(body.get("pod") or body.get("url") or ""), body.get("port"))
            else:
                url = str(body.get("url") or "").strip().rstrip("/")
                if not url.startswith(("http://", "https://")):
                    raise ValueError("Use a full address, like http://192.168.1.20:8188")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        st = load_settings()
        ms, active = machines.normalize(st)
        if any(m["url"] == url for m in ms):
            raise HTTPException(409, "That machine is already saved.")
        default = {"runpod": "RunPod", "local": "This computer"}.get(kind, "GPU PC")
        m = {"id": machines.new_id(), "name": str(body.get("name") or "").strip()[:40] or default, "kind": kind, "url": url}
        save_settings(machines.with_machines(st, [*ms, m], active))
        return machines_payload() | {"added": m["id"], "probe": machines.probe(url)}

    @app.delete("/api/machines/{machine_id}")
    def api_machine_delete(machine_id: str):
        st = load_settings()
        ms, active = machines.normalize(st)
        if machine_id == active:
            raise HTTPException(409, "Switch to another machine first.")
        save_settings(machines.with_machines(st, [m for m in ms if m["id"] != machine_id], active))
        return machines_payload(probe=False)

    @app.post("/api/machines/{machine_id}/use")
    def api_machine_use(machine_id: str):
        st = load_settings()
        ms, _ = machines.normalize(st)
        m = next((m for m in ms if m["id"] == machine_id), None)
        if not m:
            raise HTTPException(404, "No such machine.")
        status = use_url(m["url"], lambda: save_settings(machines.with_machines(load_settings(), ms, machine_id)))
        return {"status": status} | machines_payload(probe=False)

    @app.get("/api/settings")
    def api_settings():
        st = load_settings()
        d = st.get("director") or {}
        return {"comfy_url": conn.comfy_url, "active_machine": machines.active_machine(st), "allow_nvidia_rtx": bool(st.get("allow_nvidia_rtx")),
                "open_weights_only": not st.get("allow_nvidia_rtx"),
                "commercial_safe": bool(st.get("commercial_safe")),
                "director": {"backend": d.get("backend", "comfy"), "url": d.get("url", ""), "model": d.get("model", ""),
                             "vision_model": d.get("vision_model", ""), "off_acknowledged": bool(d.get("off_acknowledged"))}}

    @app.post("/api/settings")
    async def api_settings_set(req: Request):
        body = await req.json()
        st = load_settings()
        if "allow_nvidia_rtx" in body:
            st["allow_nvidia_rtx"] = bool(body["allow_nvidia_rtx"])
        retile = "commercial_safe" in body and bool(body["commercial_safe"]) != bool(st.get("commercial_safe"))
        if "commercial_safe" in body:
            st["commercial_safe"] = bool(body["commercial_safe"])
        if isinstance(body.get("director"), dict):
            d = body["director"]
            st["director"] = {"backend": d.get("backend") if d.get("backend") in ("openai", "comfy", "off") else "comfy",
                              **{k: str(d.get(k, "")).strip() for k in ("url", "model", "vision_model")},
                              # the user turned it off themselves (Setup asks politely and records the choice)
                              "off_acknowledged": bool(d.get("off_acknowledged")) if d.get("backend") == "off" else False}
        save_settings(st)
        if retile:  # the frame editor changed, so the setup tiles need different model files
            jobs.broadcast("status", refresh_status())
        return api_settings()

    # --- samples -------------------------------------------------------------------------------------------------
    @app.get("/api/samples")
    def api_samples():
        out = []
        meta = {}
        if (SAMPLES / "samples.json").exists():
            meta = json.loads((SAMPLES / "samples.json").read_text())
        for f in sorted(SAMPLES.glob("*.mp4")) if SAMPLES.exists() else []:
            thumb = f.with_suffix(".jpg")
            m = meta.get(f.stem, {})
            out.append({"id": f.stem, "name": m.get("name", f.stem.replace("_", " ").title()), "hint": m.get("hint"),
                        "try": m.get("try"), "thumb_url": f"/samples/{thumb.name}" if thumb.exists() else None})
        return out

    @app.post("/api/samples/{sample_id}/open")
    def api_sample_open(sample_id: str):
        f = SAMPLES / f"{sample_id}.mp4"
        if not f.exists():
            raise HTTPException(404, "No such sample")
        meta = json.loads((SAMPLES / "samples.json").read_text()) if (SAMPLES / "samples.json").exists() else {}
        # "Try it" copies the example's clip into My shots, the user's own workspace (the UI says so).
        return store.add_shot(f, meta.get(sample_id, {}).get("name", sample_id.replace("_", " ").title()), project_id=UNSORTED)

    # --- examples (the home screen's "What you can make") ---------------------------------------------------------
    def _examples() -> list[dict]:
        f = EXAMPLES / "examples.json"
        if not f.exists():
            return []
        out = []
        for ex in json.loads(f.read_text()):
            ex = dict(ex)
            if ex.get("id") in HIDDEN_EXAMPLES or (not features.cars() and ex.get("id") in CAR_SWAP_EXAMPLES):
                continue
            if ex.get("hold_reason"):  # held back from the gallery (make_examples' word), not deleted
                continue
            for k in ("before", "after", "receipt", "poster"):
                if ex.get(k):
                    ex[f"{k}_url"] = f"/examples/{ex.pop(k)}"
            # "Try it" needs the example's own clip (or its shot) on this machine; a downloaded pack has only the loops.
            ex["can_try"] = bool((ex.get("sample_id") and (SAMPLES / f"{ex['sample_id']}.mp4").exists())
                                 or (ex.get("shot_id") and (store.shot_dir(ex["shot_id"]) / "meta.json").exists()))
            if not ex.get("models"):  # built before make_examples wrote them, or a downloaded pack
                ex["models"] = models_catalog.as_json(models_catalog.for_task(ex.get("task", "")))
            out.append(ex)
        return first_examples(out)

    @app.get("/api/examples")
    def api_examples():
        return _examples()

    @app.get("/api/models/defaults")
    def api_models_defaults():
        """Each workflow's default models (models_catalog), with the frame editor Setup's licence choice picks."""
        return models_catalog.catalog(bool(load_settings().get("commercial_safe")))

    @app.get("/api/examples/pack")
    def api_examples_pack():
        """Whether the example gallery is installed, can be downloaded, or is being imported (a fresh install has none:
        the loops come in the examples pack)."""
        url = os.environ.get("PICKUPS_EXAMPLES_URL", EXAMPLES_PACK_URL)
        installed = (EXAMPLES / "examples.json").exists()
        # Still none: look again (a pack downloaded since the app started installs now, the same safe way).
        if (not installed and not state["pack"]["busy"] and not os.environ.get("PYTEST_CURRENT_TEST")
                and time.time() - state["pack"].get("scanned", 0) > 10):
            state["pack"]["scanned"] = time.time()
            threading.Thread(target=auto_pack, daemon=True).start()
        return {"available": bool(url), "installed": installed, "size_mb": EXAMPLES_PACK_MB,
                "examples": len(_examples()) if installed else 0, "importing": state["pack"]["busy"],
                "error": state["pack"]["error"], "found": state["pack"]["found"]}

    @app.post("/api/examples/import")
    async def api_examples_import(request: Request):
        """Install an examples pack: an uploaded zip (multipart "file"), or {"path": a zip or folder on this machine}."""
        ctype = request.headers.get("content-type", "")
        tmp: Path | None = None
        try:
            if ctype.startswith("multipart/"):
                form = await request.form()
                up = form.get("file")
                if up is None or not hasattr(up, "read"):
                    raise HTTPException(400, "Pick the examples zip.")
                tmp = Path(tempfile.mkstemp(suffix=".zip")[1])
                with tmp.open("wb") as fh:
                    while chunk := await up.read(1 << 22):
                        fh.write(chunk)
                src = tmp
            else:
                body = await request.json()
                src = Path(os.path.expanduser(str(body.get("path", "")).strip().strip('"').strip("'")))
                if not str(body.get("path", "")).strip():
                    raise HTTPException(400, "Type where the pack is: the zip, or the folder it unzips to.")
            state["pack"]["busy"], state["pack"]["error"] = True, None
            try:
                res = await asyncio.to_thread(examples_pack.install, src, EXAMPLES, SAMPLES)
            finally:
                state["pack"]["busy"] = False
            return {**res, "examples": len(_examples())}
        except (FileNotFoundError, ValueError, zipfile.BadZipFile) as e:
            raise HTTPException(400, str(e)) from None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    @app.post("/api/examples/pack")
    async def api_examples_pack_download():
        """Download and unpack the public example gallery into ~/.pickups/examples (only after the user asks)."""
        import io
        import zipfile
        url = os.environ.get("PICKUPS_EXAMPLES_URL", EXAMPLES_PACK_URL)
        if not url:
            raise HTTPException(404, "No example pack is published yet.")

        def work():
            with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - the project's own release asset
                data = r.read()
            EXAMPLES.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    name = Path(info.filename).name  # flat pack: never write outside the examples folder
                    if info.is_dir() or not name or name != info.filename or name.startswith("."):
                        continue
                    if Path(name).suffix.lower() not in (".mp4", ".jpg", ".json"):
                        continue
                    (EXAMPLES / name).write_bytes(z.read(info))
            return len(_examples())
        try:
            n = await asyncio.to_thread(work)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Couldn't download the examples: {e}") from e
        return {"examples": n}

    @app.post("/api/examples/{example_id}/open")
    def api_example_open(example_id: str):
        ex = next((e for e in _examples() if e["id"] == example_id), None)
        if not ex:
            raise HTTPException(404, "No such example")
        if ex.get("shot_id") and (store.shot_dir(ex["shot_id"]) / "meta.json").exists():
            # Examples that need their project (e.g. previz with its cast) open the real shot, with the take's settings.
            return {"shot": store.shot(ex["shot_id"]), "task": ex["task"], "params": ex.get("params", {})}
        if not ex.get("sample_id"):
            raise HTTPException(404, "This example has no shot to open on this machine")
        shot = api_sample_open(ex["sample_id"])
        return {"shot": shot, "task": ex["task"], "params": ex.get("params", {})}

    # --- shots ---------------------------------------------------------------------------------------------------
    @app.post("/api/shots/from_path")
    async def api_shot_from_path(req: Request):
        """Open a file the server can read directly (big camera files, network shares): no upload."""
        body = await req.json()
        path = Path(os.path.expanduser(str(body.get("path", "")).strip().strip('"').strip("'")))
        if not path.is_file():
            raise HTTPException(404, "Pickups can't find that file. Check the path (it must be readable on this machine).")
        try:
            return await asyncio.to_thread(add_checked, path, path.stem, False, body.get("project_id"))
        except media.MediaError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, media.UNREADABLE) from e

    def add_checked(path: Path, name: str, move: bool, project_id: str | None) -> dict:
        """Refuse, in words, what no task can work on (sound only, a still, a damaged file) before it enters the library."""
        media.check_shot(path)
        try:
            return store.add_shot(path, name, move, project_id)
        except Exception as e:  # the person gets a plain message; the details go to the log
            print(f"Couldn't add {path.name} as a shot: {e!r}", file=sys.stderr)
            raise

    @app.post("/api/shots")
    async def api_add_shot(file: UploadFile = File(...), project_id: str | None = Form(None)):
        suffix = Path(file.filename or "shot.mp4").suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=ROOT) as tmp:
            while chunk := await file.read(8 << 20):
                tmp.write(chunk)
        name = Path(file.filename or "shot").stem
        try:
            shot = await asyncio.to_thread(add_checked, Path(tmp.name), name, True, project_id)
        except Exception as e:  # noqa: BLE001
            Path(tmp.name).unlink(missing_ok=True)
            raise HTTPException(400, str(e) if isinstance(e, media.MediaError) else media.UNREADABLE) from e
        return shot

    @app.get("/api/shots")
    def api_shots(project_id: str | None = None):
        return store.shots(project_id)

    @app.patch("/api/shots/{shot_id}")
    async def api_shot_update(shot_id: str, req: Request):
        body = await req.json()
        try:
            if body.get("project_id"):
                store.move_shot(shot_id, body["project_id"])
            if body.get("name"):
                store.rename_shot(shot_id, body["name"])
            return store.shot(shot_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot or project") from e

    @app.patch("/api/shots/{shot_id}/takes/{take_id}")
    async def api_take_update(shot_id: str, take_id: str, req: Request):
        body = await req.json()
        try:
            return store.update_take(shot_id, take_id, body.get("status"), body.get("hidden"))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/shots/{shot_id}/takes/{take_id}/notes")
    async def api_note_add(shot_id: str, take_id: str, req: Request):
        body = await req.json()
        if not str(body.get("text", "")).strip():
            raise HTTPException(400, "A note needs some text.")
        frame = body.get("frame", 0)
        whole = (isinstance(frame, int) and not isinstance(frame, bool)) or (isinstance(frame, float) and frame.is_integer())
        if not whole or frame < 0:
            raise HTTPException(400, "Frame must be a whole number, 0 or more.")
        try:
            return store.add_note(shot_id, take_id, int(frame), str(body["text"]))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e

    @app.delete("/api/shots/{shot_id}/takes/{take_id}/notes/{note_id}")
    def api_note_delete(shot_id: str, take_id: str, note_id: str):
        try:
            store.delete_note(shot_id, take_id, note_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e
        return {"ok": True}

    # --- projects and cast -------------------------------------------------------------------------------------
    @app.get("/api/projects")
    def api_projects():
        return store.projects()

    @app.post("/api/projects")
    async def api_project_create(req: Request):
        body = await req.json()
        return store.create_project(str(body.get("name", "")))

    def _project_or_404(project_id: str) -> None:
        if not store.project_exists(project_id):
            raise HTTPException(404, "No such project")

    @app.get("/api/projects/{project_id}")
    def api_project(project_id: str):
        _project_or_404(project_id)
        shots = store.shots(project_id)
        uses = store.cast_uses(project_id, shots)
        return {"project": store.project(project_id, shots), "shots": shots,
                "cast": [c | {"uses": uses.get(c["id"], 0)} for c in store.cast(project_id)]}

    @app.patch("/api/projects/{project_id}")
    async def api_project_update(project_id: str, req: Request):
        _project_or_404(project_id)
        body = await req.json()
        return store.update_project(project_id, body.get("name"), body.get("archived"))

    @app.post("/api/projects/{project_id}/cast")
    async def api_cast_add(project_id: str, file: UploadFile = File(...), name: str = Form(...), kind: str = Form("person"),
                           line: str = Form("")):
        _project_or_404(project_id)
        try:
            return store.add_cast(project_id, await file.read(), file.filename or "cast.png", name, kind, line)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/cast")
    def api_all_cast():
        """Every cast entry in every project (for "Add from another project")."""
        out = []
        for p in store.projects():
            out += [c | {"project_id": p["id"], "project_name": p["name"]} for c in store.cast(p["id"])]
        return out

    @app.post("/api/projects/{project_id}/cast/import")
    async def api_cast_import(project_id: str, req: Request):
        _project_or_404(project_id)
        body = await req.json()
        try:
            return store.import_cast(project_id, body["from_project_id"], body["cast_id"])
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, "No such cast entry") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.patch("/api/projects/{project_id}/cast/{cast_id}")
    async def api_cast_update(project_id: str, cast_id: str, req: Request):
        _project_or_404(project_id)
        body = await req.json()
        try:
            return store.update_cast(project_id, cast_id, **{k: body.get(k) for k in ("name", "kind", "line")})
        except FileNotFoundError as e:
            raise HTTPException(404, "No such cast entry") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/projects/{project_id}/cast/sheet")
    async def api_cast_sheet(project_id: str, req: Request):
        """Build a character sheet (front, back, close-up; pickups/sheets.py) into the cast: from an upload, a project
        image, a cast member (upgraded, or made again with a new seed) or a description alone. Queues a "sheet" job."""
        from pickups import sheets
        _project_or_404(project_id)
        try:
            body = await req.json()
        except ValueError:
            body = None  # not JSON: sheets.request says what to send
        try:
            params = sheets.request(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        cast = store.cast(project_id)
        if params.get("upload_id"):
            if not UPLOAD_ID.fullmatch(params["upload_id"]):
                raise HTTPException(400, "That isn't an upload id.")
            try:
                store.upload_path(params["upload_id"])
            except FileNotFoundError as e:
                raise HTTPException(404, "That picture is gone. Add it again.") from e
        elif params.get("image_id") and not any(i["id"] == params["image_id"] for i in store.images(project_id)):
            raise HTTPException(404, "No such image")
        if params.get("cast_id"):
            member = next((c for c in cast if c["id"] == params["cast_id"]), None)
            if member is None:
                raise HTTPException(404, "No such cast entry")
            if member.get("kind", "person") != "person":
                raise HTTPException(400, "Character sheets are for people.")
            params["name"] = member["name"]  # for the queue's label; the member keeps its own name
        elif any(c["name"].lower() == params["name"].lower() for c in cast):
            raise HTTPException(400, f"There's already a cast entry called {params['name']}.")
        params["seed"] = seeds.pick(params.get("seed"))  # the seed it uses rides on the job, so the page can show it
        return jobs.submit(None, "sheet", params, project_id=project_id)

    @app.get("/api/projects/{project_id}/deliveries")
    def api_deliveries(project_id: str):
        _project_or_404(project_id)
        return store.deliveries(project_id)

    @app.post("/api/projects/{project_id}/finish_approved")
    def api_finish_approved(project_id: str):
        """Queue Finish (4K + HDR, best quality) for every approved take that isn't finished or being finished."""
        _project_or_404(project_id)
        queued = [jobs.submit(d["shot_id"], "finish", {"upscale_4k": True, "hdr": True, "quality": "best"}, d["take"]["id"])
                  for d in store.finish_candidates(project_id, jobs.list())]
        return {"queued": len(queued), "jobs": queued}

    @app.post("/api/shots/{shot_id}/takes/{take_id}/share")
    async def api_share(shot_id: str, take_id: str, req: Request):
        """A social-ready before/after MP4 of a take with the slate burned in: format "9x16" (stacked) or "1x1" (wipe)."""
        from pickups import share
        from pickups.app.store import _safe
        body = await req.json()
        fmt = body.get("format", "9x16")
        if fmt not in ("9x16", "1x1"):
            raise HTTPException(400, "format must be 9x16 or 1x1")
        if take_id == "orig":
            raise HTTPException(400, "Share a take, not the original")
        shot = store.shot(shot_id)
        n, take = next(((i + 1, v) for i, v in enumerate(shot["versions"]) if v["id"] == take_id), (None, None))
        if not take:
            raise HTTPException(404, "No such take")
        vdir = store.shot_dir(shot_id) / "versions" / take_id
        before = vdir / "before.mp4"
        if not before.exists():
            src = take.get("input_version") or "orig"
            before = store.shot_dir(shot_id) / "proxy.mp4" if src == "orig" else store.shot_dir(shot_id) / "versions" / src / "video.mp4"
        out = vdir / f"share_{fmt}.mp4"
        task = (take.get("label") or take.get("task") or "").split(" · ")[0]
        models = take.get("details", {}).get("models", [])
        try:
            await asyncio.to_thread(share.make, before, vdir / "video.mp4", out, fmt, task, models, take.get("receipt"), shot["name"])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Couldn't make the share video: {e}") from e
        return {"url": media_url(out), "filename": f"{_safe(shot['name'])}_T{n:02d}_{take.get('task')}_share_{fmt}.mp4"}

    @app.post("/api/shots/{shot_id}/takes/{take_id}/reveal")
    def api_reveal(shot_id: str, take_id: str):
        """Show a take's files in Finder / Explorer (only meaningful when Pickups runs on this computer)."""
        try:
            d = store.shot_dir(shot_id) if take_id == "orig" else store.version_dir(shot_id, take_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e
        if not d.exists():
            raise HTTPException(404, "No such take")
        _reveal(d / ("master.mov" if (d / "master.mov").exists() else "meta.json"))
        return {"ok": True}

    # --- trash: takes and shots (store.trash); nothing goes while a job for the shot is queued or running --------------
    def _busy(shot_id: str) -> bool:
        return any(j.get("shot_id") == shot_id and j.get("status") in ("queued", "running") for j in jobs.list())

    def _to_trash(kind: str, shot_id: str, take_id: str | None = None) -> dict:
        if _busy(shot_id):
            raise HTTPException(409, "A job is working on this shot. Cancel it or wait for it to finish.")
        try:
            entry = store.trash(kind, shot_id, take_id)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take" if kind == "take" else "No such shot") from e
        if kind == "take":
            jobs.broadcast("shot", store.shot(shot_id))
        else:
            jobs.broadcast("shot_removed", {"id": shot_id, "project_id": entry.get("project_id")})
        return entry

    @app.delete("/api/shots/{shot_id}/takes/{take_id}")
    def api_trash_take(shot_id: str, take_id: str):
        return _to_trash("take", shot_id, take_id)

    @app.delete("/api/shots/{shot_id}")
    def api_trash_shot(shot_id: str):
        return _to_trash("shot", shot_id)

    def _project_busy(project_id: str, uses: str | None = None) -> bool:
        """A queued or running job in the project (that names `uses` in its settings, when given)."""
        return any(j.get("project_id") == project_id and j.get("status") in ("queued", "running")
                   and (uses is None or uses in json.dumps(j.get("params") or {})) for j in jobs.list())

    @app.delete("/api/projects/{project_id}")
    def api_trash_project(project_id: str):
        """Delete a project: its folder and every shot in it move to the trash (Restore puts them back). Deleting My
        shots leaves an empty My shots. Returns the trash entry and the projects that are left."""
        _project_or_404(project_id)
        if _project_busy(project_id):
            raise HTTPException(409, "A job is working in this project. Cancel it or wait for it to finish.")
        try:
            entry = store.trash_project(project_id)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None
        for p in entry["paths"][1:]:
            jobs.broadcast("shot_removed", {"id": Path(p).name, "project_id": project_id})
        return {"entry": entry, "projects": store.projects()}

    def _asset_to_trash(project_id: str, kind: str, asset_id: str) -> dict:
        _project_or_404(project_id)
        if _project_busy(project_id, asset_id):
            raise HTTPException(409, "A job is using this. Cancel it or wait for it to finish.")
        try:
            entry = store.trash_asset(project_id, kind, asset_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such cast entry" if kind == "cast" else "No such image") from e
        jobs.broadcast("project", {"id": project_id, kind: [asset_id]})
        return entry

    @app.delete("/api/projects/{project_id}/cast/{cast_id}")
    def api_trash_cast(project_id: str, cast_id: str):
        """Delete a cast entry: its pictures and sheet move to the trash. Takes that used it keep their settings."""
        return _asset_to_trash(project_id, "cast", cast_id)

    @app.delete("/api/projects/{project_id}/images/{image_id}")
    def api_trash_image(project_id: str, image_id: str):
        """Delete a project image: it moves to the trash. Takes that used it keep their settings."""
        return _asset_to_trash(project_id, "image", image_id)

    @app.get("/api/trash")
    async def api_trash_list():
        return await asyncio.to_thread(store.trash_list)

    @app.post("/api/trash/{trash_id}/restore")
    def api_trash_restore(trash_id: str):
        try:
            meta = store.restore(trash_id)
        except FileExistsError:
            raise HTTPException(409, "Something is already in its place.") from None
        except FileNotFoundError as e:
            said = str(e.args[0]) if e.args else ""
            raise HTTPException(404, said if said.startswith("Restore") else "Restore its shot first.") from None
        except PermissionError:
            raise HTTPException(403, "That can't be restored.") from None
        if meta["kind"] in ("shot", "take"):
            shot = store.shot(meta["shot_id"])
            jobs.broadcast("shot", shot)
            return {"entry": meta, "shot": shot}
        if meta["kind"] == "project":
            for p in meta["paths"][1:]:
                jobs.broadcast("shot", store.shot(Path(p).name))
        else:
            jobs.broadcast("project", {"id": meta["project_id"], meta["kind"]: [meta["asset_id"]]})
        return {"entry": meta}

    @app.delete("/api/trash")
    async def api_trash_empty():
        return await asyncio.to_thread(store.empty_trash)

    # --- storage: where the library is and how big (walked off the main thread) -------------------------------------
    @app.get("/api/storage")
    async def api_storage():
        return await asyncio.to_thread(store.storage)

    # --- use on your phone: Pickups on this computer's Tailscale address (phone.py), off by default ----------------------
    from pickups.app import phone as phone_mod

    listener = phone_mod.PhoneListener()

    def phone_url(info: dict | None) -> str | None:
        if not info:
            return None
        return f"http://{info.get('name') or info['ip']}:{getattr(app.state, 'port', 7788)}"

    def phone_state(info: dict | None) -> dict:
        return {"tailscale": bool(info), "enabled": bool(load_settings().get("phone_access")) and listener.running,
                "url": phone_url(info), "ip": (info or {}).get("ip"), "name": (info or {}).get("name")}

    def start_phone() -> None:
        """At start: the saved choice, if Tailscale is up."""
        if not load_settings().get("phone_access"):
            return
        info = phone_mod.detect()
        if info:
            try:
                listener.start(app, info["ip"], getattr(app.state, "port", 7788))
            except (OSError, ValueError) as e:
                print(f"Use on your phone: couldn't listen on {info['ip']}: {e}", file=sys.stderr)

    app.state.start_phone = start_phone

    @app.get("/api/phone")
    async def api_phone():
        return phone_state(await asyncio.to_thread(phone_mod.detect))

    @app.post("/api/phone")
    async def api_phone_set(req: Request):
        body = await req.json()
        on = bool(body.get("enabled"))
        info = await asyncio.to_thread(phone_mod.detect)
        if on and not info:
            raise HTTPException(409, "Tailscale isn't running on this computer.")
        st = load_settings()
        st["phone_access"] = on
        save_settings(st)
        if on:
            try:
                listener.start(app, info["ip"], getattr(app.state, "port", 7788))
            except (OSError, ValueError) as e:
                raise HTTPException(500, f"Couldn't listen on {info['ip']}: {e}") from None
            for _ in range(20):  # the listener answers within a moment
                if listener.running:
                    break
                await asyncio.sleep(0.05)
        else:
            await asyncio.to_thread(listener.stop)
        return phone_state(info)

    @app.get("/api/phone/qr.svg")
    async def api_phone_qr():
        import io

        import segno

        url = phone_url(await asyncio.to_thread(phone_mod.detect))
        if not url:
            raise HTTPException(404, "Tailscale isn't running on this computer.")
        buf = io.BytesIO()
        segno.make(url, error="m").save(buf, kind="svg", scale=6, border=2, dark="#0D0C0B", light="#EEE9E0", xmldecl=False)
        return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.post("/api/reveal")
    def api_reveal_media(body: dict):
        """Reveal in Finder / Explorer any file the app shows (an image, a clip, a sheet), by its /media/ URL. Only files
        inside the library: a URL that points anywhere else is refused."""
        target = _media_file(str((body or {}).get("url") or ""))
        if target is None:
            raise HTTPException(404, "That file isn't in your library.")
        _reveal(target)
        return {"ok": True, "path": str(target)}

    # --- Save videos to (savecopy.py): the folder finished takes are copied into, one folder per project -----------
    @app.get("/api/save-folder")
    def api_save_folder():
        from pickups.app.savecopy import default_dir, shown
        path = load_settings().get("save_dir") or None
        return {"path": path, "suggested": str(default_dir()), "shown": shown(path), "suggested_shown": shown(default_dir())}

    @app.post("/api/save-folder")
    def api_set_save_folder(body: dict):
        from pickups.app.savecopy import check_dir
        try:
            p = check_dir(str((body or {}).get("path") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        save_settings(load_settings() | {"save_dir": str(p)})
        from pickups.app.savecopy import shown
        return {"path": str(p), "shown": shown(p)}

    @app.post("/api/save-folder/choose")
    def api_choose_save_folder():
        """The computer's folder picker, on the computer Pickups runs on. {path: null} when canceled or unavailable."""
        from pickups.app.savecopy import choose_dir, default_dir
        return {"path": choose_dir(load_settings().get("save_dir") or str(default_dir().parent))}

    @app.post("/api/save-folder/reveal")
    def api_reveal_save_folder():
        d = load_settings().get("save_dir")
        if not d or not Path(d).is_dir():
            raise HTTPException(404, "No folder chosen yet.")
        _reveal(Path(d), folder=True)
        return {"ok": True}

    @app.post("/api/storage/reveal")
    def api_storage_reveal():
        store.root.mkdir(parents=True, exist_ok=True)
        _reveal(store.root, folder=True)
        return {"ok": True}

    # --- images (create and edit) ------------------------------------------------------------------------------
    @app.get("/api/image/models")
    def api_image_models():
        from pickups import images as im
        try:
            have = conn.model_files()
        except Exception:  # noqa: BLE001 - offline: show the list, mark readiness unknown
            have = None
        return {"default": im.DEFAULT, "aspects": list(im.ASPECTS), "resolutions": ["1K", "2K"],
                "models": [{"id": m.id, "name": m.name, "maker": m.maker, "licence": m.licence, "modes": list(m.modes),
                            "commercial": not any(x in m.licence.lower() for x in ("non-commercial", "research")),
                            "max_refs": m.max_refs, "max_side": m.max_side, "speed": m.speed, "sponsor": m.sponsor,
                            "notes": m.notes, "ready": None if have is None else not im.missing(m.id, have),
                            "sizes": {r: {a: im.size_for(a, r, m) for a in im.ASPECTS} for r in ("1K", "2K")}}
                           for m in im.MODELS.values()]}

    @app.post("/api/projects/{project_id}/images/generate")
    async def api_image_generate(project_id: str, req: Request):
        _project_or_404(project_id)
        body = await req.json()
        try:  # the seed it uses rides on the job (and on each image), so Lock seed can send it again
            body["seed"] = seeds.pick(body.get("seed"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return jobs.submit(None, "image", body, project_id=project_id)

    # --- new video (Create: text to video, image to video, on LTX-2.5) -----------------------------------------------
    @app.get("/api/video/options")
    def api_video_options():
        """The pickers: aspects, sizes per resolution (1080p is 16:9 only), durations as 8n+1 frames at 24 fps."""
        from pickups import video
        return video.options()

    @app.post("/api/projects/{project_id}/videos/generate")
    async def api_video_generate(project_id: str, req: Request):
        """Queue a new video. It becomes a new shot in the project (job.result_shot_id when done)."""
        from pickups import video
        _project_or_404(project_id)
        body = await req.json()
        try:
            checked = video.request(body)
            if checked["image"]:
                store.ref_path(project_id, checked["image"])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(400, "That picture is gone. Choose another one.") from e
        return jobs.submit(None, "video", checked, project_id=project_id)

    @app.get("/api/projects/{project_id}/images")
    def api_images(project_id: str):
        _project_or_404(project_id)
        uses = store.image_uses(project_id)
        return [im | {"uses": uses.get(im["id"], 0)} for im in store.images(project_id)]

    @app.post("/api/projects/{project_id}/images/{image_id}/to_cast")
    async def api_image_to_cast(project_id: str, image_id: str, req: Request):
        _project_or_404(project_id)
        body = await req.json()
        try:
            return store.image_to_cast(project_id, image_id, str(body.get("name", "")), body.get("kind", "person"), body.get("line", ""))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such image") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # --- prompt director ---------------------------------------------------------------------------------------
    @app.get("/api/prompt/models")
    def api_prompt_models(url: str = ""):
        """Models offered by a local OpenAI-compatible server (for the Setup picker)."""
        u = (url or (load_settings().get("director") or {}).get("url", "")).rstrip("/")
        if not u:
            return []
        u = u if u.endswith("/v1") else u + "/v1"
        try:
            with urllib.request.urlopen(f"{u}/models", timeout=5) as r:  # noqa: S310 - the user's own local server
                return [m["id"] for m in json.loads(r.read()).get("data", []) if "embed" not in m["id"]]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Couldn't reach {u}: {e}") from e

    def _probe_llm(u: str) -> dict | None:
        u = u.rstrip("/")
        u = u if u.endswith("/v1") else u + "/v1"
        try:
            with urllib.request.urlopen(f"{u}/models", timeout=2.5) as r:  # noqa: S310 - local servers only
                ids = [m["id"] for m in json.loads(r.read()).get("data", []) if "embed" not in m["id"].lower()]
        except Exception:  # noqa: BLE001
            return None
        return {"url": u, "models": ids, "kind": "Ollama" if ":11434" in u else "LM Studio" if ":1234" in u else "OpenAI-compatible"}

    def _suggest(ids: list[str]) -> tuple[str, str]:
        """A sensible default text model and vision model from what the server offers (small, instruction-tuned)."""
        bad = ("uncensored", "abliterated", "heretic", "embed", "coder")
        ok = [i for i in ids if not any(b in i.lower() for b in bad)]
        rank = lambda i: (("a3b" in i or "flash" in i or "e4b" in i or "mini" in i or "small" in i) * -1, len(i))  # noqa: E731
        text = sorted(ok, key=rank)[0] if ok else (ids[0] if ids else "")
        vision = next((i for i in ok if any(v in i.lower() for v in ("gemma-4", "gemma-3", "vl", "vision", "llava", "pixtral"))), "")
        return text, vision

    @app.get("/api/prompt/detect")
    def api_prompt_detect():
        """Look for a local LLM server: LM Studio (:1234) and Ollama (:11434) on this computer, plus the saved URL."""
        saved = (load_settings().get("director") or {}).get("url", "")
        cands = [c for c in ([saved] if saved else []) + ["http://127.0.0.1:1234/v1", "http://127.0.0.1:11434/v1"] if c]
        found, seen = [], set()
        cands = sorted(dict.fromkeys(cands), key=lambda c: "127.0.0.1" not in c)  # same server twice: prefer localhost
        for c in cands:
            r = _probe_llm(c)
            if r and (r["url"] not in seen) and (tuple(r["models"]) not in seen):
                seen.update({r["url"], tuple(r["models"])})
                r["suggested_model"], r["suggested_vision_model"] = _suggest(r["models"])
                found.append(r)
        return {"found": found}

    @app.post("/api/prompt/test")
    async def api_prompt_test(req: Request):
        """Connect to a URL the user typed: lists its models and times a tiny request with the chosen model."""
        import time as _time
        body = await req.json()
        r = _probe_llm(str(body.get("url", "")))
        if not r:
            raise HTTPException(502, "Couldn't reach that server. Is LM Studio's local server running (Developer → Start Server)?")
        model = body.get("model") or _suggest(r["models"])[0]
        t = _time.monotonic()
        try:
            q = urllib.request.Request(f"{r['url']}/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps(
                {"model": model, "messages": [{"role": "user", "content": "Reply with the word ready."}], "max_tokens": 8,
                 "reasoning_effort": "none"}).encode())
            await asyncio.to_thread(lambda: urllib.request.urlopen(q, timeout=120).read())  # noqa: S310
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Connected, but {model} didn't answer: {e}") from e
        r["suggested_model"], r["suggested_vision_model"] = _suggest(r["models"])
        return r | {"model": model, "seconds": round(_time.monotonic() - t, 1)}

    @app.post("/api/prompt/enhance")
    async def api_prompt_enhance(req: Request):
        """Rewrite a plain request into the prompt a model reads best, with that model's official local enhancer.

        body: {target: qwen-image | qwen-edit | flux3 | ltx, text, project_id?, shot_id?, version_id?, frame?,
               refs?: [{kind, id}], seed?}. A shot gives the enhancer the shot's frame (Qwen edit canvas, LTX first frame);
        @-mentions are expanded first (images for Qwen edit, descriptions for the rest)."""
        from pickups import director
        body = await req.json()
        if (load_settings().get("director") or {}).get("backend") == "off":
            raise HTTPException(409, "The prompt director is turned off. You can turn it on in Setup.")
        target = body.get("target", "ltx")
        if target not in director.TARGETS:
            raise HTTPException(400, f"Unknown target {target}")
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "Write something first.")
        pid = body.get("project_id") or (store.shot_meta(body["shot_id"]).get("project_id") if body.get("shot_id") else None)
        cast = store.cast(pid) if pid else []
        files: list[Path] = []
        if body.get("shot_id") and target in ("qwen-edit", "ltx"):
            src = store.version_master(body["shot_id"], body.get("version_id"))
            frame = store.shot_dir(body["shot_id"]) / "picks" / f"enhance_{int(body.get('frame', 0))}.png"
            frame.parent.mkdir(exist_ok=True)
            await asyncio.to_thread(subprocess.run, ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                                    f"select=eq(n\\,{int(body.get('frame', 0))}),scale='min(1536,iw)':-2", "-frames:v", "1", str(frame)], check=True)
            files.append(frame)
        files += store.live_refs(pid, body.get("refs", []))
        if target == "qwen-edit":
            text, used = mentions.for_images(text, cast, first_image=1 + len(files))
            files += [store.cast_path(pid, c["id"]) for c in used]
            text = mentions.tag_images(text, len(files))
        else:
            text = mentions.for_text(text, cast)
            if target != "ltx":
                files = []

        cfg = load_settings().get("director") or {}

        def work():
            if cfg.get("backend") == "openai" and cfg.get("url"):
                return director.enhance_openai(cfg, conn.comfy_url, target, text, files, seed=int(body.get("seed", 0)))
            assets = [quick.client.assets.from_file(f) for f in files[:6]]
            return director.enhance(quick, conn.comfy_url, target, text, assets, seed=int(body.get("seed", 0)))
        try:
            out = await asyncio.to_thread(work)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"The prompt director couldn't run: {e}") from e
        return {"target": target, "model": director.TARGETS[target]["label"], "text": out, "sent": text,
                "runs_on": cfg.get("model") if cfg.get("backend") == "openai" else "ComfyUI"}

    @app.get("/api/projects/{project_id}/export/{fmt}")
    def api_export_timeline(project_id: str, fmt: str):
        """The project's approved takes as an editing timeline at their source timecode: FCP7 XML (Resolve, Premiere)
        or a CMX3600 EDL. Media paths point at the takes' ProRes masters on this computer. Each take's notes go along
        as markers (XML) or locators (EDL), with the original's notes folded into the shot's approved takes
        (nle.fold_original_notes) unless the original is approved itself."""
        from fractions import Fraction
        from fastapi.responses import Response
        from pickups import nle
        _project_or_404(project_id)
        if fmt not in ("xml", "edl"):
            raise HTTPException(400, "Format must be xml or edl")
        clips = []
        for d in reversed(store.deliveries(project_id)):  # oldest first
            take = d["take"]
            if not store.version_master(d["shot_id"], take["id"]).exists():
                continue
            master = store.friendly_master(d["shot_id"], take["id"])
            pr = media.probe(master)
            shot = store.shot(d["shot_id"])
            n = next(i for i, v in enumerate(shot["versions"]) if v["id"] == take["id"]) + 1
            clips.append(nle.Clip(name=f"{d['shot_name']} · T{n:02d} {take['label']}", reel=d["shot_name"], path=master,
                                  fps=Fraction(pr.fps), frames=pr.frames, width=pr.width, height=pr.height,
                                  start_tc=media.timecode(master),
                                  notes=[(int(x.get("frame", 0)), str(x.get("text", ""))) for x in take.get("notes") or []],
                                  original_notes=nle.fold_original_notes(shot, take["id"], pr.frames, pr.fps)))
        if not clips:
            raise HTTPException(409, "Approve some takes first (⇧A on a take); the timeline is made from approved takes.")
        title = f"{store.project(project_id)['name']} · approved takes"
        body = nle.xmeml(clips, title) if fmt == "xml" else nle.edl(clips, title)
        return Response(body, media_type="application/xml" if fmt == "xml" else "text/plain",
                        headers=attachment(store.project(project_id)["name"], f"_approved.{fmt}"))

    @app.get("/api/shots/{shot_id}/takes/{take_id}/receipt")
    def api_take_receipt(shot_id: str, take_id: str):
        """A take's receipt: one printable, self-contained HTML page (opens in a new tab). The original has none."""
        from fastapi.responses import HTMLResponse
        from pickups import receipts
        if take_id == "orig":
            raise HTTPException(404, "The original has no receipt; open a take.")
        try:
            page = receipts.take_page(store, shot_id, take_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/api/shots/{shot_id}/versions/{take_id}/workflows.zip")
    def api_take_workflows(shot_id: str, take_id: str):
        """Every ComfyUI workflow that made this take, as one zip (the take panel's Download workflows)."""
        import io
        import zipfile

        from fastapi.responses import Response
        try:
            d = store.version_dir(shot_id, take_id)  # a plain take id only
            v = store.version(shot_id, take_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such take") from e
        meta = json.loads((d / "meta.json").read_text())
        files = [(d / w["file"], f"{i + 1:02d}_{Path(w['file']).name}") for i, w in enumerate(meta.get("workflows") or [])
                 if isinstance(w, dict) and w.get("file") and (d / w["file"]).exists()]
        if not files:
            raise HTTPException(404, "This take has no ComfyUI workflows.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for src, name in files:
                z.write(src, name)
        name = f"{store.shot_meta(shot_id).get('name') or shot_id}_{v.get('label') or take_id}"
        return Response(buf.getvalue(), media_type="application/zip", headers=attachment(name, "_workflows.zip"))

    @app.get("/api/projects/{project_id}/receipts")
    def api_project_receipts(project_id: str):
        """The receipts of the project's approved takes as a zip: index.html, takes/<name>.html and the takes' ComfyUI
        workflows (workflows/<name>/<NN>_<title>.json)."""
        from fastapi.responses import Response
        from pickups import receipts
        _project_or_404(project_id)
        data = receipts.project_zip(store, project_id)
        if data is None:
            raise HTTPException(409, "Approve some takes first (⇧A on a take); receipts are made from approved takes.")
        return Response(data, media_type="application/zip", headers=attachment(store.project(project_id)["name"], "_receipts.zip"))

    @app.post("/api/mentions/preview")
    async def api_mentions_preview(req: Request):
        """What each model will actually receive for a prompt with @-mentions (shown under prompt fields)."""
        body = await req.json()
        cast = store.cast(body.get("project_id") or "unsorted")
        text = str(body.get("text", ""))
        uploads = int(body.get("uploads", 0))  # references added as files before the mentions (images 2..)
        img_text, refs = mentions.for_images(text, cast, first_image=2 + uploads)
        img_text = mentions.tag_images(img_text, 1 + uploads + len(refs))
        return {"images": img_text, "text": mentions.for_text(text, cast),
                "refs": [{"id": c["id"], "name": c["name"], "url": c["url"], "image": i + 2 + uploads} for i, c in enumerate(refs)]}

    @app.get("/api/shots/{shot_id}")
    def api_shot(shot_id: str):
        try:
            return store.shot(shot_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot") from e

    @app.post("/api/shots/{shot_id}/select")
    async def api_select(shot_id: str, req: Request):
        """Selection preview on one frame: returns a mask PNG (in viewer space) to draw over the video."""
        body = await req.json()
        frame, points, text = int(body.get("frame", 0)), body.get("points", []), body.get("text", "")
        version = body.get("version_id")
        meta = store.shot_meta(shot_id)
        vdir = store.shot_dir(shot_id) / "versions" / version if version and version != "orig" else None
        viewer = (vdir / "video.mp4") if vdir else store.shot_dir(shot_id) / "proxy.mp4"  # fast to decode
        src_w, src_h = (media.probe(vdir / "master.mov").width, media.probe(vdir / "master.mov").height) if vdir else (meta["width"], meta["height"])
        d = store.shot_dir(shot_id) / "picks"
        d.mkdir(exist_ok=True)

        def work():
            import subprocess
            p = type("P", (), {"width": src_w, "height": src_h})
            pw, ph = media.work_size(p.width, p.height, common.PROXY_LONG)
            fr = d / "frame.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(viewer), "-vf", f"select=eq(n\\,{frame}),scale={pw}:{ph}",
                            "-frames:v", "1", str(fr)], check=True)
            mask = common.select_preview(quick, fr, points, text, pw / p.width, d / "pick")
            out = d / f"mask_{frame}_{len(points)}.png"
            rgba = np.zeros((*mask.shape, 4), np.uint8)
            rgba[mask] = (245, 184, 61, 110)
            Image.fromarray(rgba).save(out)
            ys, xs = np.nonzero(mask)
            bbox = None
            if len(xs):
                sx = p.width / pw
                bbox = [int(xs.min() * sx), int(ys.min() * sx), int((xs.max() + 1) * sx), int((ys.max() + 1) * sx)]
            return {"mask_url": media_url(out), "bbox": bbox, "found": bool(len(xs))}

        try:
            return await asyncio.to_thread(work)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Selection failed: {e}") from e

    @app.post("/api/shots/{shot_id}/track")
    async def api_track(shot_id: str, req: Request):
        """Check tracking (pickups/tracking.py): the selection followed through the shot, as the render would, with an
        amber overlay video and per-frame health. Body: {frame, points, text, version_id?, start_frame?}. The masks are
        kept, so the render of the same selection skips tracking."""
        from pickups import tracking
        try:
            body = await req.json()
        except ValueError:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            frame = max(0, int(body.get("frame") or 0))
            points = [{"x": float(p["x"]), "y": float(p["y"]), "positive": bool(p.get("positive", True))}
                      for p in body.get("points") or [] if isinstance(p, dict) and "x" in p and "y" in p]
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "Couldn't read the selection. Click it again.") from e
        text = str(body.get("text") or "").strip()
        try:
            store.shot_meta(shot_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot") from e
        try:
            return await asyncio.to_thread(tracking.check, quick, store, shot_id, body.get("version_id"), frame, points, text,
                                           body.get("start_frame"))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot or take") from e
        except RuntimeError as e:  # worded for the user: nothing found, lost them, a blank shot
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # noqa: BLE001 - ComfyUI unreachable, a frame that wouldn't decode
            raise HTTPException(502, f"Couldn't check the tracking: {e}") from e

    @app.post("/api/shots/{shot_id}/people")
    async def api_people(shot_id: str, req: Request):
        """Who's in the shot, so Recast picks its person with no click (pickups/people.py). Body (optional):
        {version_id?, refresh?}. Cached per shot and take; the first look is up to four SAM 3.1 calls."""
        from pickups import people
        try:
            body = await req.json()
        except ValueError:  # no body
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            return await asyncio.to_thread(people.find, quick, store, shot_id, body.get("version_id"), bool(body.get("refresh")))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot or take") from e
        except people.PeopleError as e:
            raise HTTPException(502, str(e)) from e
        except Exception as e:  # noqa: BLE001 - a frame that wouldn't decode
            raise HTTPException(502, f"Couldn't look for people in this shot: {e}") from e

    @app.post("/api/shots/{shot_id}/describe")
    async def api_describe(shot_id: str, req: Request):
        """A local model looks at the shot (or a take) and describes it for a prompt field (pickups/describe.py). Body
        (optional): {version_id?, target?: ltx | qwen-edit | qwen-image | flux3, field?: scene | motion | look, frame?,
        refresh?, seed?}. Cached per shot or take, target and field."""
        from pickups import describe, director
        try:
            body = await req.json()
        except ValueError:  # no body
            body = {}
        body = body if isinstance(body, dict) else {}
        cfg = load_settings().get("director") or {}
        if cfg.get("backend") == "off":
            raise HTTPException(409, "The prompt director is turned off. You can turn it on in Setup.")
        target = str(body.get("target") or "ltx")
        field = str(body.get("field") or describe.default_field(target))
        if target not in director.TARGETS:
            raise HTTPException(400, f"Unknown target {target}")
        if field not in describe.FIELDS:
            raise HTTPException(400, f"Unknown field {field}")
        # The part the video model renders: Render previz's length on this GPU, else the tasks' usual MAX_FRAMES.
        span = int((state.get("status") or {}).get("previz_frames") or common.MAX_FRAMES) if field == "motion" else common.MAX_FRAMES
        frame = body.get("frame")
        try:
            return await asyncio.to_thread(
                describe.describe, quick, store, shot_id, body.get("version_id"), target, field, cfg=cfg,
                comfy_url=conn.comfy_url, span=span, frame=int(frame) if isinstance(frame, (int, float)) else None,
                refresh=bool(body.get("refresh")), seed=int(body.get("seed") or 0))
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot or take") from e
        except describe.DescribeError as e:
            raise HTTPException(502, str(e)) from e
        except Exception as e:  # noqa: BLE001 - a frame that wouldn't decode
            raise HTTPException(502, f"Couldn't describe the shot: {e}") from e

    @app.post("/api/shots/{shot_id}/trim")
    async def api_trim(shot_id: str, req: Request):
        """Keep the first N seconds as a new shot (the original stays in the library)."""
        body = await req.json()
        seconds = float(body.get("seconds", 4))
        meta = store.shot_meta(shot_id)
        src = store.source_path(shot_id)
        frames = max(1, int(round(seconds * meta["fps"])))

        def work():
            import subprocess
            out = ROOT / f"trim_{shot_id}{src.suffix}"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", str(frames), "-c", "copy", str(out)], check=True)
                return store.add_shot(out, f"{meta['name']} (first {seconds:g} s)", move=True)
            finally:
                out.unlink(missing_ok=True)  # moved into the shot on success; a failed trim leaves no copy in the library

        return await asyncio.to_thread(work)

    @app.post("/api/shots/{shot_id}/prepare")
    async def api_prepare(shot_id: str, req: Request):
        """Trim & crop: a new shot made from part of this one's source, in the same project (the original stays as it
        is). Body: {in_frame, out_frame (kept, as an editor's out point), crop: {aspect?, x, y, w, h} in source pixels
        or null, name?}. Frame-accurate and re-encoded as ProRes 422 HQ with the sound trimmed to match and the source
        timecode moved on to the in point (media.trim_and_crop)."""
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(400, "Send the in and out points as JSON.")
        try:
            meta = store.shot_meta(shot_id)
            src = store.source_path(shot_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "No such shot") from e
        try:
            start, frames = media.trim_range(int(meta["frames"]), body.get("in_frame"), body.get("out_frame"))
            box = media.crop_box(int(meta["width"]), int(meta["height"]), body.get("crop") if isinstance(body.get("crop"), dict) else None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        crop = body.get("crop") if isinstance(body.get("crop"), dict) else {}
        name = " ".join(str(body.get("name") or "").split())[:120]
        if not name:
            fps = float(meta["fps"]) or 24.0
            parts = [f"{start / fps:.1f}–{(start + frames) / fps:.1f} s"] if frames < int(meta["frames"]) else []
            parts += [str(crop.get("aspect") or "crop")] if box else []
            name = f"{meta['name']} ({', '.join(parts or ['copy'])})"
        extra = {"prepared_from": {"shot_id": shot_id, "in_frame": start, "out_frame": start + frames - 1,
                                   "crop": list(box) if box else None, "aspect": crop.get("aspect") if box else None}}

        def work():
            out = ROOT / f"prepare_{shot_id}_{time.time_ns()}.mov"
            try:
                media.trim_and_crop(src, out, start, frames, box)
                return store.add_shot(out, name, move=True, project_id=meta.get("project_id"), extra=extra)
            finally:
                out.unlink(missing_ok=True)

        try:
            shot = await asyncio.to_thread(work)
        except Exception as e:  # noqa: BLE001 - the person gets a plain message; the details go to the log
            print(f"Couldn't trim {shot_id}: {e!r} {getattr(e, 'stderr', '') or ''}", file=sys.stderr)
            raise HTTPException(500, "Couldn't make the new clip. The original is unchanged; try again.") from e
        store.touch_project(shot["project_id"])
        return shot

    def comfy_models_dir() -> str | None:
        """The connected ComfyUI's models folder as it reports it (/internal/folder_paths), on whichever machine it runs."""
        try:
            paths = _get_json(f"{conn.comfy_url}/internal/folder_paths", timeout=5)
            first = (paths.get("checkpoints") or [None])[0]
            return str(Path(first).parent) if first and "/" in first else (first.rsplit("\\", 1)[0] if first else None)
        except Exception:  # noqa: BLE001 - an older ComfyUI, or offline
            return None

    def models_dir_here() -> Path | None:
        """Where downloads can be written from here: the saved folder, or this computer's ComfyUI's own models folder."""
        saved = load_settings().get("comfy_models_dir")
        if saved and Path(saved).exists():
            return Path(saved)
        if machines.kind_of(conn.comfy_url) == "local":
            d = comfy_models_dir()
            if d and Path(d).exists():
                return Path(d)
        return None

    def comfy_models_dirs() -> list[str]:
        """Every models folder the connected ComfyUI reports (/internal/folder_paths), first one first."""
        try:
            paths = _get_json(f"{conn.comfy_url}/internal/folder_paths", timeout=5)
        except Exception:  # noqa: BLE001 - an older ComfyUI, or offline
            return []
        out: list[str] = []
        for entries in paths.values():
            for p in entries if isinstance(entries, list) else []:
                parent = p.rsplit("/", 1)[0] if "/" in p else p.rsplit("\\", 1)[0]
                if parent and parent not in out:
                    out.append(parent)
        return out

    def download_command(task: str | list[str] | None = None) -> str:
        """The one command to run on the GPU machine (from a Pickups checkout there) when Pickups can't write its folder:
        one --task per feature picked."""
        d = comfy_models_dir() or "<path to ComfyUI>/models"
        tasks = [task] if isinstance(task, str) else list(task or [])
        return f'uv run pickups models {"".join(f"--task {t} " for t in tasks)}--dir "{d}"'

    @app.post("/api/models/reveal")
    async def api_models_reveal(req: Request):
        """Open a models subfolder (diffusion_models, loras, ...) of this computer's ComfyUI, made if missing, to save a
        file the browser downloaded into."""
        body = await req.json()
        target = models_dir_here()
        folder = str(body.get("folder") or "")
        if not target or not re.fullmatch(r"[A-Za-z0-9_\-]+", folder):
            raise HTTPException(409, "ComfyUI isn't on this computer.")
        d = target / folder
        d.mkdir(parents=True, exist_ok=True)
        _reveal(d, folder=True)
        return {"path": str(d)}

    @app.get("/api/models/download")
    def api_models_download_progress():
        return state.get("download") or {"running": False}

    @app.post("/api/models/download")
    async def api_models_download(req: Request):
        """Download the picked features' missing models into the connected ComfyUI's models folder, when it's on this
        computer. Otherwise 409 with the file links. Gated files (Lightricks, Black Forest Labs) are left out: the person
        accepts the license and downloads them in their own browser, which is signed in. Pickups never handles a token."""
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001 - no body: everything
            body = {}
        task = body.get("task")
        picked = [t for t in (body.get("tasks") or ([task] if task else [])) if isinstance(t, str)]  # the features ticked
        tiles = (state["status"] or refresh_status())["tiles"]
        choice = body.get("finish_choice")  # Finish 4K alone needs only SeedVR2: fetch that choice's files

        def missing_of(t: str, v: dict) -> list[dict]:
            return (v.get("choices") or {}).get(choice, {}).get("missing", v.get("missing", [])) if t == "finish" and choice \
                else v.get("missing", [])

        missing = [m for t, v in tiles.items() if not picked or t in picked for m in missing_of(t, v)]
        files = sorted({m["file"] for m in missing})
        target = models_dir_here()
        if not target:
            raise HTTPException(409, {"message": "This ComfyUI runs on another computer. Run this there, from a Pickups "
                                                 "folder, or download the files into its models folder.",
                                      "command": download_command(picked),
                                      "files": [manifest.model_info(f) for f in files]})
        if (state.get("download") or {}).get("running"):
            raise HTTPException(409, {"message": "A download is already running."})
        files = [f for f in files if not manifest.MODELS[f][4]]  # gated: the browser downloads them
        if not files:
            raise HTTPException(409, {"message": "These need the license accepted and a download in your browser."})
        from pickups.app import models as model_dl
        total = sum(manifest.MODELS[f][3] for f in files)
        prog = {"running": True, "task": task, "tasks": picked, "files": files, "total": total, "done": 0, "file": None, "error": None}
        state["download"] = prog
        finished: dict[str, int] = {}

        def on_progress(name: str, got: int, size: int) -> None:
            finished[name] = got
            prog.update(file=name, done=sum(finished.values()))

        def run() -> None:
            try:
                model_dl.download(files, target, on_progress)
            except Exception as e:  # noqa: BLE001 - the panel shows it; the half file stays as .part
                prog["error"] = f"{type(e).__name__}: {e}"[:300]
            prog["running"] = False
            jobs.broadcast("status", refresh_status())
        threading.Thread(target=run, daemon=True).start()
        return prog

    # --- uploads -------------------------------------------------------------------------------------------------
    @app.post("/api/uploads")
    async def api_upload(file: UploadFile = File(...)):
        return store.add_upload(await file.read(), file.filename or "upload.png")

    @app.get("/api/uploads/{upload_id}")
    def api_upload_file(upload_id: str):
        """An uploaded file by id: a take's settings keep ids, so a reopened form can still show the pictures."""
        if not UPLOAD_ID.fullmatch(upload_id):
            raise HTTPException(400, "That isn't an upload id.")
        try:
            return FileResponse(store.upload_path(upload_id))
        except FileNotFoundError as e:
            raise HTTPException(404, "That file is gone.") from e

    @app.get("/api/projects/{project_id}/refs/{kind}/{ref_id}")
    def api_ref_picture(project_id: str, kind: str, ref_id: str):
        """The picture of a picked reference ({kind, id} in a take's settings): an upload, or the cast entry's or
        project image's own thumbnail."""
        if kind == "upload":
            return api_upload_file(ref_id)
        if kind not in ("cast", "image"):
            raise HTTPException(400, "kind must be upload, cast or image")
        _project_or_404(project_id)
        hit = next((x for x in (store.cast(project_id) if kind == "cast" else store.images(project_id)) if x["id"] == ref_id), None)
        if hit is None:
            raise HTTPException(404, "That reference is gone.")
        return RedirectResponse(hit.get("thumb_url") or hit["url"])

    def _draft_key(body: dict) -> str | None:
        """The page's draft for this work (optional): it travels on the inline row so a reloaded page finds it."""
        k = body.get("draft_key")
        return k[:200] if isinstance(k, str) and k.strip() else None

    def _look_refs(shot_id: str, body: dict) -> list[Path]:
        """Every reference the look preview reads, in the final render's order: uploads, then picks from the project."""
        ids = body.get("look_upload_ids")
        if not isinstance(ids, list):
            ids = [body[k] for k in ("look_upload_id", "look_upload_id_2") if body.get(k)]
        refs = [store.upload_path(str(u)) for u in ids if u]
        pid = store.shot_meta(shot_id).get("project_id")
        picks = [r for r in body.get("look_refs") or [] if isinstance(r, dict) and r.get("kind") in ("cast", "image", "upload") and r.get("id")]
        return refs + store.live_refs(pid, picks)

    @app.post("/api/previz/look")
    async def api_previz_look(req: Request):
        body = await req.json()
        shot_id = body["shot_id"]
        src = store.version_master(shot_id, body.get("version_id"))
        refs = _look_refs(shot_id, body)
        look = str(body.get("look_prompt") or body.get("describe") or "")
        # @-mentions of the cast add their pictures after the references, as the final render does.
        cast, cast_file = store.shot_cast(shot_id)
        if cast and "@" in look:
            look, used = mentions.for_images(look, cast, first_image=2 + len(refs))
            refs += [cast_file(c) for c in used]
        seed = int(body.get("seed", 42))
        print(f"Look preview for {shot_id}: seed {seed}, {len(refs)} reference(s): {', '.join(r.name for r in refs) or 'none'}",
              file=sys.stderr)

        failed = "The look frame didn't come out. Check that ComfyUI is running, then try again."

        def work():
            # The inline lane: a running row in the queue while it works, never on the GPU queue (SPEC section 3).
            with jobs.inline("previz", shot_id, "Look frame", draft_key=_draft_key(body), step="look", error=failed,
                             kind=kind_of("look_frame", {})) as run:
                d = store.shot_dir(shot_id) / "looks"
                d.mkdir(exist_ok=True)
                out = d / f"look_{len(list(d.glob('look_*.png')))}.png"
                safe = bool(load_settings().get("commercial_safe"))
                previz.look_frame(quick, store, src, refs, look, out, seed=seed, commercial_safe=safe)
                up = store.add_upload(out.read_bytes(), out.name)
                _keep_frame(shot_id, out, "Look frame", str(look or ""), seed, graphs.edit_model(safe))
                run.result = {"upload_id": up["id"], "url": up["url"], "seed": seed}
                return {"url": up["url"], "upload_id": up["id"], "seed": seed, "references": len(refs)}

        try:
            return await asyncio.to_thread(work)
        except Canceled:
            raise HTTPException(409, CANCELED) from None
        except Exception as e:  # noqa: BLE001 - the person gets one line; the detail goes to the log
            print(f"Couldn't make the look frame for {shot_id}: {e!r}", file=sys.stderr)
            raise HTTPException(502, failed) from e

    def _keep_frame(shot_id: str, png: Path, kind: str, prompt: str, seed: int | None, model: str) -> None:
        """A start or look frame Previz to final made is also a picture in the shot's project (Images, and Assets >
        Made in Pickups), so it can be found, reused as a reference and revealed. Never fails the frame."""
        try:
            pid = store.shot(shot_id).get("project_id")
            if not pid or pid == "examples":
                return
            store.add_image(pid, png, {"prompt": prompt or f"{kind} for {store.shot(shot_id)['name']}", "kind": kind,
                                       "shot_id": shot_id, "seed": seed, "model": model})
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't keep the {kind.lower()} in the project: {e!r}", file=sys.stderr)

    @app.post("/api/previz/start_frame")
    async def api_previz_start_frame(req: Request):
        """Make start frame from previz (start_frame.py): the start-frame template filled from the shot's previz and cast,
        made with method B3 (depth everywhere, the canny outside the subjects). `plan_only` returns the words without
        running anything, for a person or an agent to read first."""
        body = await req.json()
        shot_id = str(body.get("shot_id") or "")
        try:
            meta = store.shot_meta(shot_id)
        except (OSError, ValueError):
            raise HTTPException(404, "That shot isn't here any more.") from None
        refs = _look_refs(shot_id, body)
        cast, _ = store.shot_cast(shot_id)
        by_id = {c.get("id"): c for c in cast}
        picks = [r for r in body.get("look_refs") or [] if isinstance(r, dict) and r.get("kind") in ("cast", "image", "upload") and r.get("id")]
        ref_cast = [None] * (len(refs) - len(picks)) + [by_id.get(r["id"]) if r["kind"] == "cast" else None for r in picks]
        matte = store.shot_dir(shot_id) / previz_maker.SUBJECTS0
        depth0 = store.shot_dir(shot_id) / start_frame.DEPTH_FILE.format(0)
        # Commercial-safe edits on: the rules' fallback (Qwen-Image-Edit 2511, the A2 way). Otherwise method zbuf (the
        # previz's own depth) when the shot has its depth pass and ComfyUI has the 2602 ControlNet, else B3. The switch:
        # body or settings start_frame_method ("zbuf" or "B3").
        safe = bool(load_settings().get("commercial_safe"))
        framing = meta.get("previz_framing") or {}
        want = str(body.get("method") or load_settings().get("start_frame_method") or start_frame.DEFAULT_METHOD)
        patch = start_frame.zbuf_patch(conn) if want == start_frame.ZBUF and depth0.is_file() and not safe else None
        method = "commercial-safe" if safe else start_frame.ZBUF if patch else start_frame.METHOD
        try:
            p = start_frame.plan(meta.get("previz_spec") or {}, framing, cast, refs, ref_cast,
                                 str(body.get("user_look") or ""), matte=matte if matte.is_file() else None, method=method)
        except start_frame.NoSubjects as e:
            raise HTTPException(422, str(e)) from None
        zbuf = method == start_frame.ZBUF
        said = {"method": p.method, "positive": p.positive, "negative": p.negative, "references": len(p.refs),
                "subject_mask": p.matte is not None, "action": p.action, "todo": p.todo, "flags": p.flags,
                "steps": start_frame.ZBUF_STEPS if zbuf else start_frame.STEPS, "cfg": 1.0 if zbuf else start_frame.CFG,
                "template": start_frame.TEMPLATE_VERSION,
                **({"strengths": list(start_frame.strengths(start_frame.hero_kind(framing)))} if zbuf else {})}
        if body.get("plan_only"):
            return said
        seed = int(body.get("seed", 42))
        src = store.version_master(shot_id, body.get("version_id"))

        failed = "The start frame didn't come out. Check that ComfyUI is running, then try again."

        def work():
            # The inline lane: a running row in the queue while it works, never on the GPU queue (SPEC section 3).
            with jobs.inline("previz", shot_id, "Start frame", draft_key=_draft_key(body), step="start_frame", error=failed,
                             kind=kind_of("start_frame", {"method": method})) as run:
                d = store.shot_dir(shot_id) / "looks"
                d.mkdir(exist_ok=True)
                out = d / f"start_{len(list(d.glob('start_*.png')))}.png"
                frame0 = d / "start_frame0.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "select=eq(n\\,0),scale=1920:1088:force_original_aspect_ratio=increase,crop=1920:1088",
                                "-frames:v", "1", str(frame0)], check=True)
                made: dict = {}
                if zbuf:
                    from pickups import people
                    colour0 = store.shot_dir(shot_id) / previz_maker.FRAME0
                    made = start_frame.make_zbuf(quick, p, depth0, colour0 if colour0.is_file() else frame0, framing, out,
                                                 seed=seed, patch=patch, matte=matte if matte.is_file() else None,
                                                 count=lambda png: people.count_people(quick, png, d / f"{png.stem}_people"))
                    made = {"strength": made["strength"], "size_check": made["size"], "tries": made["tries"], "flag": made["flag"],
                            "people": made["people"], **({"warning": made["warning"]} if made.get("warning") else {})}
                else:
                    (start_frame.make_fallback if safe else start_frame.make)(quick, p, frame0, out, seed=seed)
                up = store.add_upload(out.read_bytes(), out.name)
                _keep_frame(shot_id, out, "Start frame", str(body.get("user_look") or ""), seed, previz.START_MODELS[method])
                # R&D #13: the still's recipe, so its mid and last keyframes are made the same way (tasks/previz.py).
                book = store.shot_dir(shot_id) / previz.RECIPES
                try:
                    recipes = json.loads(book.read_text()) if book.is_file() else {}
                except ValueError:
                    recipes = {}
                recipes[up["id"]] = {"user_look": str(body.get("user_look") or ""), "look_refs": picks, "seed": seed, "method": p.method,
                                     **({"size_check": made["size_check"]} if made.get("size_check") else {}),
                                     **({"warning": made["warning"]} if made.get("warning") else {})}
                book.write_text(json.dumps(recipes, indent=1))
                run.result = {"upload_id": up["id"], "url": up["url"], "seed": seed, **made}
                return {"url": up["url"], "upload_id": up["id"], "seed": seed, **said, **made}

        try:
            return await asyncio.to_thread(work)
        except Canceled:
            raise HTTPException(409, CANCELED) from None
        except Exception as e:  # noqa: BLE001 - the person gets one line; the detail goes to the log
            print(f"Couldn't make the start frame for {shot_id}: {e!r}", file=sys.stderr)
            raise HTTPException(502, failed) from e

    @app.post("/api/shots/{shot_id}/previz_frame0")
    async def api_shot_previz_frame0(shot_id: str, req: Request):
        """An older previz shot's frame 0, re-rendered by the page from its own spec: the framing (subjects, camera, key
        light, what's in view) and the subject matte, so its start frame has the words and the mask a new shot has."""
        body = await req.json()
        try:
            return previz_maker.refresh_frame0(store, shot_id, body.get("framing"), body.get("matte"), body.get("depth"))
        except (OSError, ValueError):
            raise HTTPException(404, "That shot isn't here any more.") from None

    @app.post("/api/previz/framing")
    async def api_previz_framing(req: Request):
        """Compare a start frame pinned at frame 0 with the playblast's first frame. Warns, never blocks."""
        body = await req.json()
        shot_id = str(body.get("shot_id") or "")
        try:
            meta = store.shot_meta(shot_id)
        except (OSError, ValueError):
            raise HTTPException(404, "That shot isn't here any more.") from None
        framing = meta.get("previz_framing")
        if not meta.get("previz_frame0") or not isinstance(framing, dict):
            return {"warning": None, "checked": False}
        if body.get("look_frame_ref"):
            start = store.ref_path(meta.get("project_id"), body["look_frame_ref"])
        elif body.get("look_frame_upload_id"):
            start = store.upload_path(str(body["look_frame_upload_id"]))
        else:
            raise HTTPException(400, "Send the start frame.")

        def work():
            frame0 = store.shot_dir(shot_id) / previz_maker.FRAME0
            # previz_frame0 is the hash of the frame the page exported: only that frame is compared against.
            if not frame0.is_file() or hashlib.sha256(frame0.read_bytes()).hexdigest() != meta["previz_frame0"]:
                return {"warning": None, "checked": False}
            got = framing_check.compare(frame0, framing, start)
            # R2-3: the size is the start-frame endpoint's own depth measure (start_frame.size_from_depth): the one saved
            # with a still the app made, else measured now; the place and horizon stay framing.compare's.
            recipe = previz.start_recipe(store, shot_id, body.get("look_frame_upload_id"))
            size = recipe.get("size_check") or start_frame.size_check_now(quick, store.shot_dir(shot_id), framing, start)
            place = got["warning"] if got.get("warning") and "bigger" not in got["warning"] and "smaller" not in got["warning"] else None
            warning = recipe.get("warning") or size.get("warning") or place  # the people count first (Run 3)
            return {**got, "warning": warning, "size_check": size, "checked": True, "frame0": meta["previz_frame0"][:12]}

        try:
            return await asyncio.to_thread(work)
        except (OSError, ValueError) as e:
            print(f"Couldn't compare the start frame for {shot_id}: {e!r}", file=sys.stderr)
            return {"warning": None, "checked": False}

    # --- previz maker (browser layout → a shot) ------------------------------------------------------------------
    @app.post("/api/previz/{session}/frames")
    async def api_previz_frames(session: str, start: int = Form(...), files: list[UploadFile] = File(...)):
        """One batch of JPEG frames (at most 16). `start` is the index of the first frame in this batch."""
        blobs = [await f.read() for f in files]
        try:
            received = previz_maker.save_frames(ROOT, session, start, blobs)
        except previz_maker.PrevizError as e:
            raise HTTPException(400, str(e)) from None
        return {"received": received}

    @app.post("/api/previz/{session}/finish")
    async def api_previz_finish(session: str, req: Request):
        """Assemble the uploaded frames into a ProRes shot and store the spec on it."""
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("spec"), dict):
            raise HTTPException(400, "Send the previz spec as JSON.")
        vram = ((state.get("status") or {}).get("machine") or {}).get("vram_gb")
        cap = common.previz_frames(vram)
        spec, project_id, name = body["spec"], body.get("project_id"), str(body.get("name") or "Previz")
        if req.headers.get("x-pickups-qc"):
            project_id = store.qc_project()  # a QC or test run: never into a person's projects

        def work():
            return previz_maker.finish(ROOT, store, session, spec, project_id, name, cap, framing=body.get("framing"),
                                       matte=body.get("matte"), depth=body.get("depth"))

        try:
            return await asyncio.to_thread(work)
        except previz_maker.PrevizError as e:
            raise HTTPException(400, str(e)) from None
        except Exception as e:  # noqa: BLE001 - the person gets one line; the ffmpeg detail goes to the log
            print(f"Couldn't assemble previz {session}: {e!r} {getattr(e, 'stderr', b'')!r}", file=sys.stderr)
            raise HTTPException(500, "Couldn't make the shot. Try again.") from e

    @app.delete("/api/previz/{session}")
    def api_previz_cancel(session: str):
        try:
            previz_maker.discard(ROOT, session)
        except previz_maker.PrevizError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    # --- jobs ----------------------------------------------------------------------------------------------------
    @app.post("/api/jobs")
    async def api_submit(req: Request):
        body = await req.json()
        try:
            step = body.get("step")
            return jobs.submit(body["shot_id"], body["task"], body.get("params", {}), body.get("version_id"),
                               draft_key=_draft_key(body), step=step[:40] if isinstance(step, str) and step.strip() else None)
        except KeyError as e:
            raise HTTPException(400, f"Unknown task {e}") from e

    @app.get("/api/jobs")
    def api_jobs():
        return jobs.list()

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError as e:
            raise HTTPException(404, "No such job") from e
        except NotStoppable:
            raise HTTPException(409, "This one can't be stopped") from None

    @app.post("/api/jobs/{job_id}/next")
    def api_job_next(job_id: str):
        try:
            return jobs.move_next(job_id)
        except KeyError as e:
            raise HTTPException(404, "No such job") from e
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/timings")
    def api_timings(kind: str):
        """How long a kind of run (timings.kind_of) usually takes here, measured: {kind, usual_s, runs}."""
        return {"kind": kind, "usual_s": jobs.clock().usual(kind), "runs": len(jobs.clock().runs(kind))}

    @app.get("/api/events")
    async def api_events(req: Request):
        loop = asyncio.get_running_loop()
        q = jobs.listen(loop)

        async def stream():
            try:
                yield "event: status\ndata: " + json.dumps(state["status"] or refresh_status()) + "\n\n"
                while True:
                    if await req.is_disconnected():
                        break
                    try:
                        kind, data = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                jobs.unlisten(q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # --- media and the UI ----------------------------------------------------------------------------------------
    if PREVIZ_ASSETS.is_dir():
        app.mount("/previz", StaticFiles(directory=PREVIZ_ASSETS), name="previz")
    app.mount("/media", StaticFiles(directory=ROOT), name="media")
    # Made now, even empty: a pack installed after start (first run, or Add examples) must be served without a restart.
    # A fresh install mounted /samples only if it already existed, so its thumbnails fell through to index.html.
    SAMPLES.mkdir(parents=True, exist_ok=True)
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
    if WEB.exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets") if (WEB / "assets").exists() else None

        @app.get("/{path:path}")
        def spa(path: str):
            if path == "api" or path.startswith("api/"):  # an unknown API route is a 404, never the app's page
                raise HTTPException(404, f"No such API route: /{path}")
            f = WEB / path
            if path and f.is_file():
                return FileResponse(f, headers={"Cache-Control": "no-cache"} if f.name == "index.html" else None)
            return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/")
        def no_ui():
            return JSONResponse({"pickups": "running", "ui": "not built (cd web && npm run build)"})

    return app
