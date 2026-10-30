"""Finish's receipt: an upscale is its own share ("upscaled": your footage resampled, no new content), never AI; the
only generated pixels are the HDR highlights LTX-2.5 rebuilt. Older takes on disk said "100% AI" for any upscale and
are read the new way, without rewriting their files."""

import io
import json
import re
import subprocess
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pickups import receipts, share
from pickups.app.store import Store
from pickups.tasks import finish

OLD_HDR = "highlights reconstructed by LTX-2.5 on 13.24% of pixels; everything else is your footage"


def test_finish_receipt_shape():
    # Upscale and HDR: nothing is the untouched original, nothing is borrowed; the rest is upscaled.
    assert receipts.finish_receipt(True, 13.24, "SeedVR2 3B", 2.0) == {
        "original": 0.0, "borrowed": 0.0, "upscaled": 86.76, "ai": 13.24, "upscaler": "SeedVR2 3B", "scale": 2.0}
    # Upscale only: no AI share at all.
    assert receipts.finish_receipt(True, 0.0, "SeedVR2 3B", 3.0) == {
        "original": 0.0, "borrowed": 0.0, "upscaled": 100.0, "ai": 0.0, "upscaler": "SeedVR2 3B", "scale": 3.0}
    # HDR only: the original footage, with the rebuilt highlights as the AI share.
    assert receipts.finish_receipt(False, 13.24) == {"original": 86.76, "borrowed": 0.0, "ai": 13.24}
    # Neither (a 4K source with HDR off): untouched.
    assert receipts.finish_receipt(False, 0.0) == {"original": 100.0, "borrowed": 0.0, "ai": 0.0}
    for r in (receipts.finish_receipt(True, 13.24, "SeedVR2 3B", 2.0), receipts.finish_receipt(False, 13.24)):
        assert sum(receipts.percents(r)) == 100 and receipts.percents(r)[3] == 13 and not receipts.all_ai(r)


def test_finish_words():
    both = receipts.finish_receipt(True, 13.24, "SeedVR2 3B", 2.0)
    assert receipts.upscale_sentence(both) == "Upscaled 2× from your footage by SeedVR2 3B: new detail, no new content."
    assert receipts.highlights_sentence(both) == "HDR highlights rebuilt by LTX-2.5 on 13% of pixels."
    assert receipts.summary(both) == "87% upscaled · 13% AI"
    rtx = receipts.finish_receipt(True, 0.0, "NVIDIA RTX Video Super Resolution", 2.6667)
    assert receipts.upscale_sentence(rtx) == "Upscaled 2.7× from your footage by NVIDIA RTX Video Super Resolution: no new content."
    assert receipts.highlights_sentence(rtx) is None and receipts.summary(rtx) == "100% upscaled · 0% AI"
    assert receipts.highlights_sentence(receipts.finish_receipt(False, 0.3)) == "HDR highlights rebuilt by LTX-2.5 on under 1% of pixels."
    assert receipts.upscale_sentence(receipts.finish_receipt(False, 13.24)) is None
    # An older take that named no models or sizes: nothing is guessed.
    bare = receipts.finish_receipt_on_read({"original": 0.0, "borrowed": 0.0, "ai": 100.0}, {})
    assert bare == {"original": 0.0, "borrowed": 0.0, "upscaled": 100.0, "ai": 0.0}
    assert receipts.upscale_sentence(bare) == "Upscaled from your footage: new detail, no new content."
    # The share video's slate: never "0% original pixels" for an upscale.
    assert share.stat_line(both) == "Upscaled 2× · 13% AI"
    assert share.stat_line({"original": 97.0, "borrowed": 3.0, "ai": 0.0}) == "97% ORIGINAL PIXELS"
    assert share.stat_line({"original": 0.0, "borrowed": 0.0, "ai": 100.0}) == ""


def test_one_rounding_everywhere():
    # The share slate rounds as the app and the sheet do (largest remainder), never round() on its own.
    half = receipts.finish_receipt(True, 13.5, "SeedVR2 3B", 2.0)
    assert receipts.summary(half) == "87% upscaled · 13% AI" and share.stat_line(half) == "Upscaled 2× · 13% AI"
    assert receipts.highlights_sentence(half) == "HDR highlights rebuilt by LTX-2.5 on 13% of pixels."
    # A share that is there never reads 0%, and nothing reads 100% beside it.
    tiny = receipts.finish_receipt(False, 0.4)
    assert receipts.percent_texts(tiny) == (">99%", "0%", "0%", "<1%")
    assert receipts.summary(tiny) == ">99% original · 0% other frames · <1% AI"
    assert share.stat_line(tiny) == ">99% ORIGINAL PIXELS"
    assert receipts.highlights_sentence(tiny) == "HDR highlights rebuilt by LTX-2.5 on under 1% of pixels."
    assert receipts.summary({"original": 96.77, "borrowed": 3.23, "ai": 0.0}) == "97% original · 3% other frames · 0% AI"


def test_compose_along_the_input():
    own = receipts.finish_receipt(True, 8.6, "SeedVR2 3B", 2.0)
    remove = {"original": 95.2, "borrowed": 3.9, "ai": 0.9}
    r = receipts.compose(own, remove, "02", "Remove")
    # ai = hl + (100 - hl) × in.ai; borrowed = (100 - hl) × in.borrowed; upscaled = the rest.
    assert (r["original"], r["borrowed"], r["upscaled"], r["ai"], r["highlights"]) == (0.0, 3.56, 87.01, 9.42, 8.6)
    assert receipts.summary(r) == "4% other frames · 87% upscaled · 9% AI"
    assert receipts.upscale_sentence(r) == "Upscaled 2× from Take 02 by SeedVR2 3B: new detail, no new content."
    assert receipts.source_sentence(r) == "Take 02 (Remove) is 95% original · 4% other frames · 1% AI."
    assert receipts.highlights_sentence(r) == "HDR highlights rebuilt by LTX-2.5 on 9% of pixels."
    assert share.stat_line(r) == "Upscaled 2× · 9% AI"
    # Without an upscale the input's shares scale by (100 - hl), and hl joins the AI.
    hdr = receipts.compose(receipts.finish_receipt(False, 8.6), remove, "02", "Remove")
    assert (hdr["original"], hdr["borrowed"], hdr["ai"]) == (87.01, 3.56, 9.42) and "upscaled" not in hdr
    assert receipts.upscale_sentence(hdr) is None
    assert receipts.hdr_detail(hdr) == "highlights rebuilt by LTX-2.5 on 9% of pixels; everything else is Take 02"
    # A previz input is generated through and through: so is its Finish, upscaled or not.
    pv = receipts.compose(own, {"original": 0.0, "borrowed": 0.0, "ai": 100.0}, "02", "Render previz")
    assert receipts.all_ai(pv) and receipts.summary(pv) == "100% AI · whole frame generated (upscaled 2×)"
    assert receipts.source_sentence(pv) is None and share.stat_line(pv) == ""
    # A Finish of a Finish says where the upscale happened.
    again = receipts.compose(receipts.finish_receipt(False, 5.0), own, "03", "Finish 4K + HDR")
    assert again["source_upscaled"] and receipts.upscale_sentence(again) == (
        "Take 03 was upscaled 2× from your footage by SeedVR2 3B: new detail, no new content.")


# --- Finish's own run, with ComfyUI and the HDR maths stood in for -------------------------------------------------
class _Out:
    def __init__(self, name, write):
        self.name, self._write = name, write

    def to_file(self, path):
        self._write(path)


class _Job:
    def __init__(self, node, outs):
        self.node, self.outs = node, outs

    def get_outputs(self, node):
        return self.outs if node == self.node else []


def _png(w, h):
    return lambda path: Image.new("RGB", (w, h), (128, 128, 128)).save(path)


@pytest.fixture()
def run_finish(tmp_path, monkeypatch):
    """finish.run() on a 64×36, 3-frame clip: SeedVR2 / RTX give 128×72 frames, LTX's highlights cover 25% of pixels."""
    def work_clip(ctx, out, ltx=False):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=24", "-frames:v", "3",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        return {"width": 64, "height": 36, "fps": 24.0, "frames": 3}

    upscaled = lambda *a, **k: _Job("PU_Out", [_Out(f"up_{i:05d}.png", _png(128, 72)) for i in range(3)])  # noqa: E731
    calls = []
    monkeypatch.setattr(finish.common, "work_clip", work_clip)
    monkeypatch.setattr(finish.common, "finish_outputs", lambda *a, **k: [])
    monkeypatch.setattr(finish, "make_viewer", lambda *a, **k: None)
    monkeypatch.setattr(finish.graphs, "seedvr2_video", lambda *a, **k: calls.append("seedvr2") or upscaled())
    monkeypatch.setattr(finish.graphs, "rtx_upscale", lambda *a, **k: calls.append("rtx") or upscaled())
    monkeypatch.setattr(finish.graphs, "sdr_to_hdr", lambda *a, **k: _Job("PU_EXR", [_Out(f"h_{i:05d}.exr", lambda p: p.write_bytes(b"x"))
                                                                                    for i in range(3)]))
    monkeypatch.setattr(finish.media, "prepare_work_clip", lambda *a, **k: None)
    monkeypatch.setattr(finish.hdr, "read_exr", lambda path: None)
    monkeypatch.setattr(finish.hdr, "write_exr", lambda path, img: None)

    def merge_frame(sdr, _hdr):
        h, w = sdr.shape[:2]
        wgt = np.zeros((h, w), np.float32)
        wgt[: h // 4] = 1.0  # a quarter of the rows are highlights
        return np.zeros((h, w, 3), np.float32), wgt

    def write_hlg(frames, fps, out):
        list(frames)
        out.write_bytes(b"")

    monkeypatch.setattr(finish.hdr, "merge_frame", merge_frame)
    monkeypatch.setattr(finish.hdr, "write_hlg", write_hlg)

    def run(params, rtx=False):
        vdir = tmp_path / f"take_{len(list(tmp_path.glob('take_*')))}"
        vdir.mkdir()
        ctx = SimpleNamespace(params=params, vdir=vdir, input_path=None, stage=lambda s: None, check=lambda: None,
                              progress=lambda x: None, run=lambda job: job, allow_rtx=lambda: rtx,
                              client=SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: p)),
                              conn=SimpleNamespace(free_memory=lambda: None))
        return finish.run(ctx)

    run.calls = calls
    return run


def test_finish_run_receipts(run_finish):
    both = run_finish({"upscale_4k": True, "hdr": True, "quality": "best"})
    assert both["receipt"] == {"original": 0.0, "borrowed": 0.0, "upscaled": 75.0, "ai": 25.0, "upscaler": "SeedVR2 3B", "scale": 2.0}
    assert both["label"] == "Finish 4K HDR" and both["details"]["models"] == ["SeedVR2 3B", "LTX-2.5 SDR-to-HDR"]
    # Details and the receipt say the same thing; an older take's details read back to the same receipt.
    assert "on 25.0% of pixels" in both["details"]["hdr"] and both["details"]["hdr"].endswith("your footage, upscaled")
    assert receipts.finish_receipt_on_read({"original": 0.0, "borrowed": 0.0, "ai": 100.0}, both["details"]) == both["receipt"]

    up = run_finish({"upscale_4k": True, "hdr": False, "quality": "best"})
    assert up["receipt"] == {"original": 0.0, "borrowed": 0.0, "upscaled": 100.0, "ai": 0.0, "upscaler": "SeedVR2 3B", "scale": 2.0}
    assert "hdr" not in up["details"]

    hdr_only = run_finish({"upscale_4k": False, "hdr": True, "quality": "best"})
    assert hdr_only["receipt"] == {"original": 75.0, "borrowed": 0.0, "ai": 25.0} and hdr_only["label"] in ("Finish 1080p HDR", "Finish HDR")

    fast = run_finish({"upscale_4k": True, "hdr": False, "quality": "fast"}, rtx=True)
    assert fast["receipt"]["upscaler"] == "NVIDIA RTX Video Super Resolution" and fast["receipt"]["ai"] == 0.0
    assert fast["details"]["models"] == ["NVIDIA RTX Video Super Resolution"]
    assert run_finish.calls == ["seedvr2", "seedvr2", "rtx"]


# --- the library: read-time migration and the receipt sheet ----------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    return TestClient(server.create_app("http://127.0.0.1:9")), store


def _video(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "1",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _take(store, shot_id, task, receipt, details, **meta):
    vid, vdir = store.new_version_dir(shot_id)
    _video(vdir / "video.mp4")
    store.save_version(shot_id, vid, {"label": meta.pop("label", "Finish 4K + HDR"), "task": task, "receipt": receipt,
                                      "details": details, **meta})
    return vid, vdir


OLD_FINISH = {"input": "1920x1080", "output": "3840x2160", "frames": 24, "hdr": OLD_HDR, "models": ["SeedVR2 3B", "LTX-2.5 SDR-to-HDR"]}


def test_store_reads_old_finish_takes_the_new_way(client, tmp_path):
    _, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Coast road car")
    old = {"original": 0.0, "borrowed": 0.0, "ai": 100.0}
    vid, vdir = _take(store, shot["id"], "finish", old, OLD_FINISH)
    on_disk = (vdir / "meta.json").read_text()
    assert store.version(shot["id"], vid)["receipt"] == {
        "original": 0.0, "borrowed": 0.0, "upscaled": 86.76, "ai": 13.24, "upscaler": "SeedVR2 3B", "scale": 2.0}
    assert (vdir / "meta.json").read_text() == on_disk  # read, never rewritten

    # 4K only (no HDR detail), an older "(4K)" model name, and today's details (resolution, × sizes).
    plain = {"input": "1280×720", "resolution": "3840×2160", "frames": 24, "models": ["SeedVR2 3B (4K)"]}
    vid2, _ = _take(store, shot["id"], "finish", old, plain, label="Finish 4K")
    assert store.version(shot["id"], vid2)["receipt"] == {
        "original": 0.0, "borrowed": 0.0, "upscaled": 100.0, "ai": 0.0, "upscaler": "SeedVR2 3B", "scale": 3.0}

    # Already the new shape, HDR-only Finish takes and fully generated takes of other tasks are left as they are.
    new = receipts.finish_receipt(True, 30.73, "SeedVR2 3B", 2.0)
    for task, r, d in (("finish", new, OLD_FINISH), ("finish", {"original": 86.76, "borrowed": 0.0, "ai": 13.24}, {"hdr": OLD_HDR}),
                       ("environment", old, {"models": ["Qwen-Image 2.1"]}), ("previz", old, OLD_FINISH)):
        v, _ = _take(store, shot["id"], task, r, d)
        assert store.version(shot["id"], v)["receipt"] == r, task
    # The whole shot (what the app and the receipt sheets read) carries the migrated receipt too.
    assert next(v for v in store.shot(shot["id"])["versions"] if v["id"] == vid)["receipt"]["upscaled"] == 86.76


def test_finish_receipt_sheet_is_self_consistent(client, tmp_path):
    c, store = client
    prj = store.create_project("Coast")
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Coast road car", project_id=prj["id"])
    vid, _ = _take(store, shot["id"], "finish", {"original": 0.0, "borrowed": 0.0, "ai": 100.0}, OLD_FINISH)
    page = c.get(f"/api/shots/{shot['id']}/takes/{vid}/receipt").text
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))
    assert "whole frame is generated" not in page and "100% AI" not in text
    changed = page[page.index("What changed"):page.index("How it was made")]
    assert "Upscaled 2× from your footage by SeedVR2 3B: new detail, no new content." in changed
    assert "HDR highlights rebuilt by LTX-2.5 on 13% of pixels." in changed
    assert re.search(r"<b>87%</b><span class=\"slate\"><i class=\"sw u\"></i>Upscaled</span>", changed)
    assert re.search(r"<b>13%</b><span class=\"slate\"><i class=\"sw a\"></i>AI</span>", changed)
    assert "Original</span>" not in changed and "Other frames</span>" not in changed  # nothing of either to name
    assert 'aria-label="87% upscaled, 13% AI"' in changed and '<i class="u" style="width:87%">' in changed
    # Section 03 agrees with section 02: the same rounding, and the upscale said there too.
    assert "highlights rebuilt by LTX-2.5 on 13% of pixels; everything else is your footage, upscaled" in page
    assert "13.24%" not in page

    # The project's receipts: the index names the grey share in words.
    store.update_take(shot["id"], vid, status="approved")
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/projects/{prj['id']}/receipts").content))
    index = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", z.read("index.html").decode()))
    assert "87% upscaled · 13% AI" in index and "whole frame generated" not in index
    assert "Upscaled 2× from your footage by SeedVR2 3B: new detail, no new content." in index


def test_hdr_only_finish_sheet(client, tmp_path):
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Alpine meadow")
    vid, _ = _take(store, shot["id"], "finish", receipts.finish_receipt(False, 13.24), {"hdr": OLD_HDR, "models": ["LTX-2.5 SDR-to-HDR"]},
                   label="Finish HDR")
    page = c.get(f"/api/shots/{shot['id']}/takes/{vid}/receipt").text
    changed = page[page.index("What changed"):page.index("How it was made")]
    assert ("87% of the picture is the original footage, untouched.</strong> HDR highlights rebuilt by LTX-2.5 on 13% of pixels."
            in changed)
    assert "generated by AI" not in changed and "Upscaled" not in changed
    assert json.loads(json.dumps(store.version(shot["id"], vid)["receipt"])) == {"original": 86.76, "borrowed": 0.0, "ai": 13.24}


# --- a Finish made from a take: its input's pixels and models come with it -------------------------------------------
PREVIZ = {"models": ["Qwen-Image 2.1", "LTX-2.5 Layout-To-Render"], "frames": 24}
REMOVE = {"models": ["SAM 3.1", "Qwen-Image 2.1", "LTX-2.5"], "method": "LTX-2.5 in/outpaint from a Qwen-Image 2.1 plate", "frames": 24}
NEW_FINISH = {"input": "1920x1080", "resolution": "3840×2160", "frames": 24, "hdr": OLD_HDR, "models": ["SeedVR2 3B", "LTX-2.5 SDR-to-HDR"]}


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def test_finish_of_a_previz_take(client, tmp_path):
    c, store = client
    prj = store.create_project("Bart")
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "K03 hood push", project_id=prj["id"])
    pv, _ = _take(store, shot["id"], "previz", {"original": 0.0, "borrowed": 0.0, "ai": 100.0}, PREVIZ, label="Render previz",
                  status="approved")
    # An older Finish (ai: 100 for any upscale) and a new one, both made from the previz take.
    old, _ = _take(store, shot["id"], "finish", {"original": 0.0, "borrowed": 0.0, "ai": 100.0}, OLD_FINISH, input_version=pv)
    new, vdir = _take(store, shot["id"], "finish", receipts.finish_receipt(True, 13.24, "SeedVR2 3B", 2.0), NEW_FINISH,
                      input_version=pv, status="approved")
    on_disk = (vdir / "meta.json").read_text()
    takes = {v["id"]: v for v in store.shot(shot["id"])["versions"]}
    for vid in (old, new):
        r = takes[vid]["receipt"]
        assert receipts.all_ai(r) and r["ai"] == 100.0, vid
        assert receipts.summary(r) == "100% AI · whole frame generated (upscaled 2×)"
        lic = {m["name"]: m for m in takes[vid]["licences"]}
        assert lic["Qwen-Image 2.1"]["commercial"] is False and lic["Qwen-Image 2.1"]["via"] == "Take 02"
        assert "via" not in lic["SeedVR2 3B"] and lic["LTX-2.5 Layout-To-Render"]["via"] == "Take 02"
    assert (vdir / "meta.json").read_text() == on_disk  # read, never rewritten

    page = c.get(f"/api/shots/{shot['id']}/takes/{new}/receipt").text
    changed = _text(page[page.index("What changed"):page.index("How it was made")])
    assert "100% AI: the whole frame is generated. Upscaled 2× from Take 02 by SeedVR2 3B" in changed
    assert "from your footage" not in page and "Upscaled</span>" not in page
    assert "Not for paid work as it stands: Qwen-Image 2.1 (via Take 02)" in _text(page)
    assert re.search(r"<td>Qwen-Image 2.1 <span class=\"muted\">via Take 02</span></td>.*Non-commercial", page)

    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/projects/{prj['id']}/receipts").content))
    index = _text(z.read("index.html").decode())
    assert "2 of 2 takes use a model licensed for non-commercial use only" in index
    assert "100% AI · whole frame generated (upscaled 2×)" in index and "Qwen-Image 2.1 (via Take 02)" in index


def test_finish_of_a_remove_take(client, tmp_path):
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Old town")
    rm, _ = _take(store, shot["id"], "remove", {"original": 95.2, "borrowed": 3.9, "ai": 0.9}, REMOVE, label="Remove",
                  workflows=[{"file": "workflows/04_plate.json", "step": "plate"}])
    fin, _ = _take(store, shot["id"], "finish", receipts.finish_receipt(True, 8.6, "SeedVR2 3B", 2.0),
                   NEW_FINISH | {"hdr": "highlights reconstructed by LTX-2.5 on 8.6% of pixels; everything else is your footage, upscaled"},
                   input_version=rm)
    take = next(v for v in store.shot(shot["id"])["versions"] if v["id"] == fin)
    r = take["receipt"]
    assert r["borrowed"] > 0 and r["ai"] > 8.6 and r["source"] == {"take": "02", "label": "Remove", "receipt": {
        "original": 95.2, "borrowed": 3.9, "ai": 0.9}}
    assert receipts.summary(r) == "4% other frames · 87% upscaled · 9% AI"
    assert take["details"]["hdr"] == "highlights rebuilt by LTX-2.5 on 9% of pixels; everything else is Take 02, upscaled"
    assert [(m["name"], m.get("via")) for m in take["licences"]] == [
        ("SeedVR2 3B", None), ("LTX-2.5 SDR-to-HDR", None), ("SAM 3.1", "Take 02"), ("Qwen-Image 2.1", "Take 02"), ("LTX-2.5", "Take 02")]
    # The Remove itself is read as it is.
    assert next(v for v in store.shot(shot["id"])["versions"] if v["id"] == rm)["receipt"] == {"original": 95.2, "borrowed": 3.9, "ai": 0.9}

    page = c.get(f"/api/shots/{shot['id']}/takes/{fin}/receipt").text
    changed = _text(page[page.index("What changed"):page.index("How it was made")])
    assert ("Upscaled 2× from Take 02 by SeedVR2 3B: new detail, no new content. Take 02 (Remove) is 95% original · "
            "4% other frames · 1% AI. HDR highlights rebuilt by LTX-2.5 on 9% of pixels.") in changed
    for share_ in ("4% Other frames", "87% Upscaled", "9% AI"):
        assert share_ in changed
    assert "from your footage" not in changed and "Not for paid work as it stands: Qwen-Image 2.1 (via Take 02)" in _text(page)


def test_store_puts_older_models_right_on_read(client, tmp_path):
    _, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Castle hall")
    scail = {"models": ["SAM 3.1", "SCAIL-2 (Wan 2.1)"]}
    # A Recast whose person FLUX 3 klein made from a description, before Recast named it: a saved "cast" step...
    made, vdir = _take(store, shot["id"], "recast", {"original": 94.0, "borrowed": 0.0, "ai": 6.0}, scail, label="Recast",
                       workflows=[{"file": "workflows/06_cast.json", "step": "cast"}], params={"reference_prompt": "a noblewoman"})
    on_disk = (vdir / "meta.json").read_text()
    # ... or a reference prompt and no cast, project image or dropped photo. A cast entry is a photo: no FLUX.
    worded, _ = _take(store, shot["id"], "recast", {"original": 94.0, "borrowed": 0.0, "ai": 6.0}, scail, label="Recast",
                      params={"reference_prompt": "a noblewoman"})
    cast, _ = _take(store, shot["id"], "recast", {"original": 94.0, "borrowed": 0.0, "ai": 6.0}, scail, label="Recast",
                    params={"reference_prompt": "a noblewoman", "cast_id": "cast_1"})
    # A Remove of real pixels only: older ones listed the frame editor although it added nothing.
    real = {"models": ["SAM 3.1", "Qwen-Image 2.1"], "method": "real pixels from other frames"}
    rm, _ = _take(store, shot["id"], "remove", {"original": 96.24, "borrowed": 3.76, "ai": 0.0}, real, label="Remove",
                  workflows=[{"file": "workflows/01_pick.json", "step": "pick"}, {"file": "workflows/02_track.json", "step": "track"}])
    plate, _ = _take(store, shot["id"], "remove", {"original": 96.0, "borrowed": 3.0, "ai": 1.0},
                     real | {"method": "real pixels from other frames + Qwen-Image 2.1 plate"}, label="Remove")
    v = {x["id"]: x for x in store.shot(shot["id"])["versions"]}
    names = {k: [m["name"] for m in v[k]["licences"]] for k in (made, worded, cast, rm, plate)}
    assert names[made] == names[worded] == ["SAM 3.1", "SCAIL-2 (Wan 2.1)", "FLUX 3 klein"]
    assert names[cast] == ["SAM 3.1", "SCAIL-2 (Wan 2.1)"]
    assert names[rm] == ["SAM 3.1"] and v[rm]["details"]["models"] == ["SAM 3.1"]
    assert names[plate] == ["SAM 3.1", "Qwen-Image 2.1"]
    flux = next(m for m in v[made]["licences"] if m["name"] == "FLUX 3 klein")
    assert flux["licence"] and flux["commercial"] is not True
    assert (vdir / "meta.json").read_text() == on_disk  # read, never rewritten
