"""Will this computer work: finding ComfyUI (none, 8000, 8188, a typed address), the per-feature GPU rule (NVIDIA by
name, one decimal GB unit, Relight's own gate), free space, ffmpeg's version, and Finish's per-choice model lists."""

import subprocess
from types import SimpleNamespace

import pytest

from pickups.app import doctor, manifest, requirements, servers

NVIDIA_4090 = {"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda", "vram_total": 25.8e9}
EVERY_FILE = set(manifest.MODELS)
EVERY_NODE = set().union(*manifest.TASK_NODES.values()) | {n for p in manifest.NODE_PACKS.values() for n in p["nodes"]}


def stats(dev=NVIDIA_4090, version="0.41.1"):
    return {"system": {"comfyui_version": version, "ram_total": 64e9}, "devices": [dev | {"vram_free": 1e9}]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PICKUPS_HOME", str(tmp_path))
    monkeypatch.setattr("pickups.app.comfy.load_settings", lambda: {})


def conn(st, files=EVERY_FILE):
    return SimpleNamespace(comfy_url="http://gpu:8188", system_stats=lambda: st, model_scan=lambda: (set(files), set()),
                           model_files=lambda: set(files), node_types=lambda: set(EVERY_NODE))


# --- finding ComfyUI: 8188 then 8000, plus what was typed before ------------------------------------------------------

def answers(*ok_urls):
    def test(url, timeout=5.0):
        if url in ok_urls:
            return {"url": url, "ok": True, "gpu": "RTX 4090"}
        return {"url": url, "ok": False, "reason": "refused"}
    return test


def test_none_found(monkeypatch):
    monkeypatch.setattr(servers, "test", answers())
    d = servers.detect()
    assert d["found"] is None and [t["url"] for t in d["tried"]] == ["http://127.0.0.1:8188", "http://127.0.0.1:8000"]


def test_found_on_8000_the_desktop_app(monkeypatch):
    monkeypatch.setattr(servers, "test", answers("http://127.0.0.1:8000"))
    assert servers.detect()["found"]["url"] == "http://127.0.0.1:8000"


def test_found_on_8188_first(monkeypatch):
    monkeypatch.setattr(servers, "test", answers("http://127.0.0.1:8188", "http://127.0.0.1:8000"))
    assert servers.detect()["found"]["url"] == "http://127.0.0.1:8188"


def test_a_typed_address_is_kept_and_probed_next_time():
    st = servers.remember({}, "http://gpu-pc:8188", "RTX 4090")
    assert [r["url"] for r in servers.recents(st)] == ["http://gpu-pc:8188"]


# --- the GPU rule (SYSTEM_REQUIREMENTS.md c) -------------------------------------------------------------------------

def test_one_vram_unit_decimal_gb():
    assert requirements.vram_gb(stats()) == 25.8 and servers.facts(stats())["vram_gb"] == 25.8


def test_amd_with_24_gb_is_not_supported_even_as_cuda():
    amd = stats({"name": "cuda:0 AMD Radeon RX 7900 XTX : native", "type": "cuda", "vram_total": 25.8e9})
    s = doctor.status(conn(amd))
    assert s["verdict"] == "remote"
    assert {t["support"] for t in s["tiles"].values()} == {"unsupported"}
    # installing anyway stays possible: the files are in place, so nothing is blocked on VRAM
    assert "vram_needed_gb" not in s["tiles"]["relight"]


def test_an_nvidia_16_gb_card_is_not_tested_not_blocked():
    s = doctor.status(conn(stats(NVIDIA_4090 | {"name": "cuda:0 NVIDIA GeForce RTX 4080 : cudaMallocAsync", "vram_total": 17.2e9})))
    assert s["verdict"] == "untested"
    assert s["tiles"]["remove"]["support"] == "untested" and s["tiles"]["remove"]["ready"]
    # Relight's own gate, in the same unit, says the same number as the tile
    assert not s["tiles"]["relight"]["ready"] and s["tiles"]["relight"]["vram_needed_gb"] == requirements.RELIGHT_VRAM_GB == 25.0


def test_a_4090_is_ok_everywhere():
    s = doctor.status(conn(stats()))
    assert s["verdict"] == "ready" and {t["support"] for t in s["tiles"].values()} == {"ok"}
    assert s["tiles"]["relight"]["ready"]


def test_comfyui_older_than_0_41_says_update():
    t = doctor.status(conn(stats(version="0.3.68")))["tiles"]["remove"]
    assert not t["ready"] and t["reason"] == "comfy"


# --- free space and ffmpeg -------------------------------------------------------------------------------------------

def test_low_disk(monkeypatch):
    monkeypatch.setattr(requirements.shutil, "disk_usage", lambda p: SimpleNamespace(total=500e9, used=470e9, free=30e9))
    free = requirements.disk_free_gb("/models")
    # Replace a person missing everything: 62.7 GB of files plus 10 GB headroom doesn't fit in 30 GB
    need = sum(manifest.MODELS[f][3] for f in manifest.TASK_MODELS["recast"]) / 1e9 + requirements.headroom(["recast"])
    assert free == 30.0 and need > free
    assert requirements.headroom(["recast", "finish"]) == 20.0 and requirements.headroom(["image"]) == 5.0


def test_old_ffmpeg(monkeypatch):
    monkeypatch.setattr(requirements.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    out = {"v": "ffmpeg version 7.1.1 Copyright (c) 2000-2025"}
    monkeypatch.setattr(requirements.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out["v"], ""))
    assert requirements.ffmpeg_major() == 7 < requirements.MIN_FFMPEG
    out["v"] = "ffmpeg version n8.0 Copyright"
    assert requirements.ffmpeg_major() == 8
    monkeypatch.setattr(requirements.shutil, "which", lambda name: None)
    assert requirements.ffmpeg_major() is None


# --- Finish: each choice its own files -------------------------------------------------------------------------------

def test_finish_4k_only_needs_seedvr2_alone():
    assert manifest.FINISH_CHOICE_MODELS["4k"] == ["seedvr2_3b_int8_convrot.safetensors", "seedvr2_ema_vae_fp16.safetensors"]
    assert not set(manifest.FINISH_CHOICE_MODELS["4k"]) & {f for f in manifest.MODELS if "sdr-to-hdr" in f or f in manifest.LTX_CORE}
    only_seedvr = (EVERY_FILE - set(manifest.TASK_MODELS["finish"])) | set(manifest.FINISH_CHOICE_MODELS["4k"])
    t = doctor.status(conn(stats(), files=only_seedvr))["tiles"]["finish"]
    assert not t["ready"] and t["choices"]["4k"]["ready"] and t["choices"]["4k"]["missing"] == []
    assert not t["choices"]["hdr"]["ready"] and not t["choices"]["4k-hdr"]["ready"]
    assert {m["file"] for m in t["choices"]["4k-hdr"]["missing"]} == set(manifest.FINISH_CHOICE_MODELS["hdr"])
