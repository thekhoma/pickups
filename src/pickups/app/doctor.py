"""The setup check: is ComfyUI reachable, what machine is it, and can each task run here?

Three outcomes only: ready (24 GB+ NVIDIA, everything installed), missing models (with sizes and links), or needs a
remote GPU (no ComfyUI, or a GPU below the tested 24 GB profile). Nothing is guessed: facts come from ComfyUI's
/system_stats and the model lists its loader nodes expose.
"""

from __future__ import annotations

from pickups.app import bench, manifest, requirements
from pickups.app.comfy import Connection
from pickups.app.requirements import MIN_VRAM_GB, RELIGHT_VRAM_GB  # noqa: F401 - kept importable from here


def status(conn: Connection) -> dict:
    stats = conn.system_stats()
    if not stats:
        return {"connected": False, "comfy_url": conn.comfy_url, "machine": None, "verdict": "offline",
                "tiles": {t: {"ready": False, "eta_s": None, "missing": []} for t in manifest.task_models()},
                "benchmarks": bench.status_rows(None, manifest.BENCH_4090)}
    dev = (stats.get("devices") or [{}])[0]
    raw = dev.get("name", "")  # e.g. "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync"
    gpu = bench.gpu_name(raw) or "unknown"
    vram = round(dev.get("vram_total", 0) / 1e9, 1)
    ram = round(stats.get("system", {}).get("ram_total", 0) / 1e9, 1)
    from urllib.parse import urlparse

    from pickups.app.comfy import load_settings

    from pickups.app import machines

    host = urlparse(conn.comfy_url).hostname or ""
    st = load_settings()
    active = machines.active_machine(st)
    kind = active["kind"] if active["url"] == conn.comfy_url else machines.kind_of(conn.comfy_url)
    name = (active["name"] if active["url"] == conn.comfy_url else None) or st.get("machine_name") or (
        "This computer" if host in ("127.0.0.1", "localhost") else "GPU")
    machine = {"name": name, "kind": kind, "gpu": gpu, "vram_gb": vram, "vram_free_gb": round(dev.get("vram_free", 0) / 1e9, 1), "ram_gb": ram,
               "comfy_version": stats.get("system", {}).get("comfyui_version"), "os": stats.get("system", {}).get("os")}
    scan = getattr(conn, "model_scan", None)
    have, unknown_folders = scan() if scan else (conn.model_files(), set())
    # /object_info failed or timed out: nodes are unknown, and a feature is never Ready on an unread node list.
    nodes = conn.node_types()
    nodes_checked = bool(nodes)
    factor = manifest.gpu_factor(gpu)
    tiles = {}
    from pickups.images import _by_basename, _resolve  # file names may differ by repo, or sit in subfolders
    by_base = _by_basename(have)
    # per-model image downloads are left out (the Image page reads each model's readiness itself)
    for task, files in manifest.task_models(bool(load_settings().get("commercial_safe"))).items():
        absent = [f for f in files if _resolve(f, have, by_base) is None]
        unknown = [manifest.model_info(f) for f in absent if manifest.MODELS[f][0] in unknown_folders]
        missing = [manifest.model_info(f) for f in absent if manifest.MODELS[f][0] not in unknown_folders]
        # A soft file (one the feature has a fallback for) is listed as missing but never holds the feature back.
        soft = set(manifest.SOFT_MODELS.get(task, []))
        blocking_unknown = [m for m in unknown if m["file"] not in soft]
        blocking = [m for m in missing if m["file"] not in soft]
        eta = manifest.TASK_ETA_4090.get(task, 15) * factor
        core, packs = manifest.missing_nodes(task, nodes) if nodes_checked else ([], [])
        old_comfy = requirements.comfy_ok(stats) is False  # under 0.41: Update ComfyUI first
        reason = ("nodes_unknown" if not nodes_checked else "comfy" if old_comfy else "models_unknown" if blocking_unknown
                  else "nodes" if core or packs else "models" if blocking else None)
        tiles[task] = {"ready": reason is None and task != "reshoot", "eta_s": int(eta), "missing": missing,
                       "missing_nodes": packs, "missing_core_nodes": core, "unknown": unknown, "nodes_checked": nodes_checked,
                       "reason": reason, "support": requirements.support(task, stats)}
        # NVIDIA below Relight's peak can't hold it, whatever is installed (other features are only "not tested")
        if task == "relight" and requirements.is_nvidia(stats) and vram < RELIGHT_VRAM_GB:
            tiles[task]["ready"] = False
            tiles[task]["vram_needed_gb"] = RELIGHT_VRAM_GB
        if task == "finish":  # each choice has its own files: 4K alone needs only SeedVR2
            tiles[task]["choices"] = {
                c: {"ready": reason in (None, "models") and all(_resolve(f, have, by_base) is not None for f in fs),
                    "missing": [m for m in missing if m["file"] in fs]}
                for c, fs in manifest.FINISH_CHOICE_MODELS.items()}
        if task == "recast":  # the form says the time of the engine picked (eta_s is the default's, LTX-2.5)
            tiles[task]["eta_by_engine"] = {"ltx": int(eta), "scail": int(manifest.TASK_ETA_4090["recast-scail"] * factor)}
    # NVIDIA under the tested VRAM runs, untested; another GPU needs a remote one
    verdict = {"ok": "ready", "untested": "untested"}.get(requirements.support("remove", stats), "remote")
    return {"connected": True, "comfy_url": conn.comfy_url, "machine": machine, "verdict": verdict, "tiles": tiles,
            # measured runs (PICKUPS_HOME/benchmarks.jsonl), this GPU first; the 4090 reference scaled when none yet
            "benchmarks": bench.status_rows(gpu, manifest.BENCH_4090, factor)}
