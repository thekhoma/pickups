"""`pickups` command: start the app (default), check the setup, or benchmark this machine."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import webbrowser


def _port_free(host: str, port: int) -> bool:
    """Whether Pickups can listen on host:port (the same bind uvicorn does)."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(prog="pickups", description="Start Pickups, or run one of its commands.")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--comfy", help="your ComfyUI address (default: last used, or http://127.0.0.1:8188)")
    ap.add_argument("--port", type=int, default=7788, help="the port Pickups opens on (default: 7788)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open Pickups to your network")
    ap.add_argument("--no-browser", action="store_true")
    sub.add_parser("doctor", help="check ComfyUI, the GPU and the models, then exit")
    bp = sub.add_parser("bench", help="summarise the measured times of your finished runs, per task and GPU (writes bench.json)")
    bp.add_argument("--backfill", action="store_true", help="first add the takes made before runs were recorded")
    bp.add_argument("--gpu", default="RTX 4090", help="with --backfill: the GPU those takes ran on (default: RTX 4090)")
    bp.add_argument("--pod-takes", help="with --backfill: a text file of take ids (one per line) that ran on the rented GPU")
    bp.add_argument("--pod-gpu", default="RTX PRO 6000 Blackwell Server Edition", help="with --pod-takes: that GPU's name")
    bp.add_argument("--md", metavar="README", help="rewrite the table between the bench markers in this Markdown file")
    mp = sub.add_parser("models", help="download the models Pickups needs into a ComfyUI models folder (run on the GPU machine)")
    mp.add_argument("--dir", required=True, help="path to ComfyUI/models")
    mp.add_argument("--task", action="append", help="only for these tasks (repeat for several): remove, recast, environment, previz, relight, finish, image (FLUX 3 klein), image-<model>, recast-scail (Recast's SCAIL-2 engine), or commercial-safe (Qwen-Image-Edit 2511)")
    a = ap.parse_args()

    if a.cmd == "doctor":
        from pickups.app import doctor
        from pickups.app.comfy import Connection

        print(json.dumps(doctor.status(Connection.from_settings(a.comfy)), indent=2))
        return 0

    if a.cmd == "bench":  # reads the run log only: no ComfyUI needed
        from pathlib import Path

        from pickups.app import bench

        if a.backfill:
            added = bench.backfill(gpu=a.gpu, pod_takes=a.pod_takes, pod_gpu=a.pod_gpu)
            print(f"Backfilled {len(added)} take{'' if len(added) == 1 else 's'} into {bench.path()}\n")
        data = bench.write()
        print(bench.markdown(data["rows"]) if data["rows"] else "No runs measured yet. Finish a job, or run with --backfill.")
        if a.md:
            readme = Path(a.md)
            if not readme.exists() and readme.with_suffix(".md").exists():
                readme = readme.with_suffix(".md")
            bench.rewrite_readme(readme, data["rows"])
            print(f"\nUpdated the benchmarks in {readme}")
        return 0

    if a.cmd == "models":
        from pathlib import Path

        from pickups.app import manifest, models

        d = Path(a.dir)
        todo = models.missing_for(d, a.task)
        if not todo:
            print("All models are in place.")
            return 0
        gb = sum(manifest.MODELS[f][3] for f in todo) / 1e9
        print(f"Downloading {len(todo)} files ({gb:.1f} GB) into {d}")
        if any(manifest.MODELS[f][4] for f in todo) and not os.environ.get("HF_TOKEN"):
            print("Some LTX files are gated: accept the license on each model page, then set HF_TOKEN in this shell first.")

        def show(name, got, total):
            pct = f"{100 * got / total:5.1f}%" if total else f"{got / 1e9:.1f} GB"
            print(f"\r  {name}: {pct}", end="", flush=True)

        for f in todo:
            models.download([f], d, show)
            print()
        return 0

    if not _port_free(a.host, a.port):
        print(f"Port {a.port} is already in use (is Pickups already running?). "
              f"Start on another port: uv run pickups --port {a.port + 2}")
        return 1

    import uvicorn

    from pickups.app.server import create_app

    app = create_app(a.comfy)
    app.state.port = a.port  # Use on your phone listens on the Tailscale address at the same port
    app.state.start_phone()
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Pickups is running at {url}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0
