import io
import json
import re
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient

from pickups import receipts
from pickups.app.store import Store

XSS = "<script>alert(1)</script>"


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
    app = server.create_app("http://127.0.0.1:9")  # a dead local address: no ComfyUI in tests
    return TestClient(app), store


def _video(path, src="testsrc", tc=None):
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=size=320x180:rate=24", "-t", "1", "-pix_fmt", "yuv420p"]
    subprocess.run(args + (["-timecode", tc] if tc else []) + [str(path)], check=True)
    return path


def _remove_take(store, tmp_path, project_id):
    """A tiny Remove take, the way a finished job leaves it: viewer, before, master, receipt, a workflow and notes."""
    shot = store.add_shot(_video(tmp_path / "clip.mov", tc="01:00:10:00"), f"Pier {XSS}", project_id=project_id)
    vid, vdir = store.new_version_dir(shot["id"])
    _video(vdir / "video.mp4", "testsrc2")
    _video(vdir / "before.mp4")
    _video(vdir / "receipt.mp4", "rgbtestsrc")
    _video(vdir / "master.mov", "testsrc2", tc="01:00:10:00")
    (vdir / "workflows").mkdir()
    pick = {"PU_Frame": {"class_type": "LoadImage", "inputs": {"image": "pick.png"}},
            "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1.safetensors"}},
            "detect": {"class_type": "SAM3_Detect", "inputs": {"model": ["sam", 0], "image": ["PU_Frame", 0], "conditioning": ["PU_Label", 0]}},
            "PU_Label": {"class_type": "CLIPTextEncode", "inputs": {"text": f"woman {XSS}", "clip": ["sam", 1]}}}
    plate = {"PU_Prompt": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "Fill the red area with the pier", "negative_prompt": ""}},
             "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
             "k": {"class_type": "KSampler", "inputs": {"seed": 1234}}}
    (vdir / "workflows" / "01_pick.json").write_text(json.dumps(pick))
    (vdir / "workflows" / "02_plate.json").write_text(json.dumps(plate))
    store.save_version(shot["id"], vid, {
        "label": "Remove", "task": "remove", "receipt": {"original": 96.77, "borrowed": 3.23, "ai": 0.0},
        "downloads": [{"label": "ProRes 422 HQ · HD 1920×1080", "file": "master.mov", "codec": "prores", "size": "HD"}],
        "details": {"resolution": "320x180", "frames": 24, "fps": 24.0, "camera": "locked-off", "method": "real pixels from other frames",
                    "pixels_changed_outside_selection": 0, "models": ["SAM 3.1", "Qwen-Image 2.1"], "gpu_seconds": 25.2,
                    "total_seconds": 507.7, "timecode": "01:00:10:00", "first_frame": str(tmp_path / "secret.png")},
        "params": {"frame": 12, "text": "woman"}, "input_version": "orig",
        "workflows": [{"file": "workflows/01_pick.json", "step": "pick"}, {"file": "workflows/02_plate.json", "step": "plate"}]})
    store.add_note(shot["id"], vid, 6, f"Grain looks off {XSS}")
    return shot["id"], vid, vdir


def _no_leaks(text: str, tmp_path) -> None:
    assert str(tmp_path) not in text and "/Users/" not in text and "/private/" not in text
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:|//)|url\((https?:|//)""", text)  # no external requests


def test_take_receipt_page(client, tmp_path):
    c, store = client
    prj = store.create_project("Coastal spot")
    sid, vid, vdir = _remove_take(store, tmp_path, prj["id"])
    r = c.get(f"/api/shots/{sid}/takes/{vid}/receipt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    for want in ("Pickups receipt", "Coastal spot", "Take 02", "<em>Remove</em>", "Before and after", "What changed",
                 "97%", "3%", "<b>0</b>", "pixels changed outside the selection", "How it was made", "real pixels from other frames", "locked-off",
                 "Models and licenses", "Qwen Research License", "Non-commercial", "Prompts, as each model received them",
                 "Fill the red area with the pier", "Seed 1234", "Negative: blurry", "Selection: woman", "01:00:10:00", "01:00:11:00",
                 "Files delivered", "_<wbr>T02_<wbr>remove_<wbr>ProRes422HQ_<wbr>HD.mov", "Notes", "<b>00:00:00:06</b>",
                 "Source 01:00:10:06", "ComfyUI workflows", "Rebuild<span class=\"more\"> · real pixels, in Pickups",
                 "In Pickups, no workflow", "Drag one onto ComfyUI to see what ran", "Made with Pickups",
                 "Allowed, within Meta&#x27;s trade-control and end-use limits"):
        assert want in page, want
    assert page.index("Before and after") < page.index("What changed") < page.index("Models and licenses") < page.index("Prompts")
    assert page.index("Timecode") < page.index("Files delivered") < page.index("Notes") < page.index("ComfyUI workflows")
    assert "Setup → Licenses" not in page  # what to do about a non-commercial model is for the app, not the client's sheet
    assert re.search(r"<dt>Made</dt><dd>\d{1,2} [A-Z][a-z]{2} \d{4} · \d\d:\d\d</dd>", page)
    assert page.count("data:image/jpeg;base64,") == 3  # before, after and the receipt on the picture
    assert page.count('href="data:application/json;base64,') == 2  # both workflows, embedded
    assert "Content-Security-Policy" in page and "default-src 'none'" in page
    # Everything people typed is escaped: shot names, prompts, workflow titles and notes.
    assert XSS not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert len(re.findall(r"<script", page)) == 1  # only the page's own print button
    _no_leaks(page, tmp_path)
    assert len(r.content) < 2 * 1024 * 1024
    # Stills are cached next to the take, so the next export is quick.
    assert (vdir / "receipt_before.jpg").exists() and (vdir / "receipt_after.jpg").exists() and (vdir / "receipt_map.jpg").exists()
    assert c.get(f"/api/shots/{sid}/takes/{vid}/receipt").text.count("data:image/jpeg;base64,") == 3


def test_fully_generated_take_says_so(client, tmp_path):
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Castle")
    vid, vdir = store.new_version_dir(shot["id"])
    _video(vdir / "video.mp4", "testsrc2")
    store.save_version(shot["id"], vid, {"label": "Weather · Snow", "task": "environment",
                                         "receipt": {"original": 0.0, "borrowed": 0.0, "ai": 100.0},
                                         "details": {"frames": 24, "look_prompt": "Snow <b>everywhere</b>", "prompt": "Snow falls",
                                                     "models": ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA"]}})
    page = c.get(f"/api/shots/{shot['id']}/takes/{vid}/receipt").text
    assert "100% AI: the whole frame is generated." in page
    assert "Weather · Snow" in page and "Change the weather" in page
    # No workflows saved: the prompts come from the take's details, and the seed is the default.
    assert "Snow &lt;b&gt;everywhere&lt;/b&gt;" in page and "Snow falls" in page and "Seed 42 (the default)" in page
    assert "The source has no timecode" in page
    # Made before Pickups kept workflows: the section says so instead of listing none (or pointing at a folder).
    assert "This take was made before Pickups kept its ComfyUI workflows." in page and "Drag one onto ComfyUI" not in page
    assert "Allowed for a business under $10M annual revenue" in page  # LTX's commercial terms, not "Check its terms"
    _no_leaks(page, tmp_path)

    # Its place in a project's receipts: plain words in the index, and no promise of a workflows/ folder.
    prj = store.create_project("Castle")
    store.move_shot(shot["id"], prj["id"])
    store.update_take(shot["id"], vid, status="approved")
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/projects/{prj['id']}/receipts").content))
    assert not any(n.startswith("workflows/") for n in z.namelist())
    index = re.sub(r"<[^>]+>", "", z.read("index.html").decode())
    assert "100% AI · whole frame generated" in index and "workflows/" not in index and "One page per take in takes/." in index
    assert "Conditions apply: LTX-2.5 + Union Control IC-LoRA" not in index  # non-commercial wins the flag
    page = z.read(next(n for n in z.namelist() if n.startswith("takes/"))).decode()
    assert "made before Pickups kept its ComfyUI workflows" in page and "in this zip" not in page


def test_workflows_kept_but_none_ran(client, tmp_path):
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Pier")
    vid, vdir = store.new_version_dir(shot["id"])
    _video(vdir / "video.mp4", "testsrc2")
    store.save_version(shot["id"], vid, {"label": "Remove", "task": "remove", "workflows": [],
                                         "details": {"frames": 24, "camera": "locked-off", "models": []}})
    page = c.get(f"/api/shots/{shot['id']}/takes/{vid}/receipt").text
    assert "No ComfyUI workflow ran for this take: Pickups made it itself." in page and "Rebuild" not in page


def test_note_timecodes(client, tmp_path):
    """Notes read as the app shows them (from the take's first frame), stay inside the take, and need no source TC."""
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Castle")
    vid, vdir = store.new_version_dir(shot["id"])
    _video(vdir / "video.mp4", "testsrc2")
    store.save_version(shot["id"], vid, {"label": "Recast", "task": "recast", "details": {"frames": 24, "fps": 24.0}})
    store.add_note(shot["id"], vid, 30, "Past the end")
    store.add_note(shot["id"], vid, 5, "Early")
    page = c.get(f"/api/shots/{shot['id']}/takes/{vid}/receipt").text
    notes = re.search(r'<ul class="notes">.*?</ul>', page, re.S).group(0)
    assert notes.index("<b>00:00:00:05</b>") < notes.index("<b>00:00:00:23</b>")  # in frame order; 30 sits on the last frame
    assert "Frame 23" in notes and "Source" not in notes
    # The API takes whole frames from 0 up.
    for bad in (-5, 2.5, "12", True):
        r = c.post(f"/api/shots/{shot['id']}/takes/{vid}/notes", json={"frame": bad, "text": "x"})
        assert r.status_code == 400, bad
    assert c.post(f"/api/shots/{shot['id']}/takes/{vid}/notes", json={"frame": 3.0, "text": "x"}).json()["frame"] == 3


def test_ids_are_folder_names(client, tmp_path):
    """A take or shot id from a URL is a folder name, never "..": the shot's own meta.json stays untouched."""
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Pier")
    meta = store.shot_dir(shot["id"]) / "meta.json"
    before = meta.read_text()
    for call in (lambda: store.add_note(shot["id"], "..", 1, "x"), lambda: store.delete_note(shot["id"], "..", "n_1"),
                 lambda: store.update_take(shot["id"], "..", status="approved"), lambda: store.version_master(shot["id"], ".."),
                 lambda: store.shot("..")):
        with pytest.raises(FileNotFoundError):
            call()
    r = c.post(f"/api/shots/{shot['id']}/takes/%2E%2E/notes", json={"frame": 1, "text": "x"})
    assert r.status_code == 404 and r.json()["detail"] == "No such take"  # the route matched; the id was refused
    assert c.patch(f"/api/shots/{shot['id']}/takes/%2E%2E", json={"status": "approved"}).status_code == 404
    assert meta.read_text() == before


def test_download_names():
    from pickups.app.server import attachment
    assert attachment("Coastal spot", "_receipts.zip") == {"Content-Disposition": 'attachment; filename="Coastal_spot_receipts.zip"'}
    h = attachment("東京ロケ", "_receipts.zip")["Content-Disposition"]
    assert 'filename="Pickups_receipts.zip"' in h and "filename*=UTF-8''%E6%9D%B1%E4%BA%AC%E3%83%AD%E3%82%B1_receipts.zip" in h
    assert "filename*=UTF-8''Caf%C3%A9_Shoot_approved.xml" in attachment("Café / Shoot", "_approved.xml")["Content-Disposition"]


def test_receipt_404s(client, tmp_path):
    c, store = client
    shot = store.add_shot(_video(tmp_path / "clip.mov"), "Pier")
    assert c.get(f"/api/shots/{shot['id']}/takes/orig/receipt").status_code == 404
    assert c.get(f"/api/shots/{shot['id']}/takes/v_nope/receipt").status_code == 404
    assert c.get("/api/shots/shot_nope/takes/v_nope/receipt").status_code == 404


def test_project_receipts_zip(client, tmp_path):
    c, store = client
    prj = store.create_project("Coastal spot")
    sid, vid, _ = _remove_take(store, tmp_path, prj["id"])
    r = c.get(f"/api/projects/{prj['id']}/receipts")
    assert r.status_code == 409 and "Approve some takes first" in r.json()["detail"]
    store.update_take(sid, "orig", status="approved")  # an approved original alone has nothing to receipt
    assert c.get(f"/api/projects/{prj['id']}/receipts").status_code == 409
    store.update_take(sid, vid, status="approved")

    r = c.get(f"/api/projects/{prj['id']}/receipts")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert 'filename="Coastal_spot_receipts.zip"' in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    stem = "Pier_script_alert_1_script_T02_remove"
    assert {"index.html", f"takes/{stem}.html"} <= names
    flows = sorted(n for n in names if n.startswith("workflows/"))
    assert len(flows) == 2 and all(re.fullmatch(rf"workflows/{stem}/0[12]_[A-Za-z0-9_]+\.json", n) for n in flows)
    assert json.loads(z.read(flows[0]))["PU_Label"]["class_type"] == "CLIPTextEncode"  # the exact workflow

    index = z.read("index.html").decode()
    assert "Coastal spot" in index and f'href="takes/{stem}.html"' in index and "Non-commercial" in index
    assert re.sub(r"<[^>]+>", "", index).count("97% original · 3% other frames · 0% AI") == 1
    assert "data:image/jpeg;base64," in index
    page = z.read(f"takes/{stem}.html").decode()
    assert f'href="../{flows[0]}"' in page and 'href="../index.html"' in page and "in this zip" in page
    assert "data:application/json" not in page  # the zip links its files instead of embedding them
    for n in ("index.html", f"takes/{stem}.html"):
        text = z.read(n).decode()
        assert XSS not in text
        _no_leaks(text, tmp_path)
    assert c.get("/api/projects/prj_nope/receipts").status_code == 404


def test_percents_match_the_app():
    # (original, other frames, upscaled, AI), as format.ts receiptPercents()
    assert receipts.percents({"original": 96.77, "borrowed": 3.23, "ai": 0.0}) == (97, 3, 0, 0)
    assert receipts.percents({"original": 0.0, "borrowed": 0.0, "ai": 100.0}) == (0, 0, 0, 100)
    assert receipts.percents({"original": 0.0, "borrowed": 0.0, "upscaled": 86.76, "ai": 13.24}) == (0, 0, 87, 13)
    assert sum(receipts.percents({"original": 33.3, "borrowed": 33.3, "ai": 33.4})) == 100
    assert receipts.percents(None) is None and receipts.percents({"original": 0, "ai": 0}) is None
    # The shares a receipt names: an upscale (Finish) shows Upscaled and AI, everything else Original, Other frames, AI.
    assert [label for _, label, _ in receipts.shares({"original": 96.77, "borrowed": 3.23, "ai": 0.0})] == ["Original", "Other frames", "AI"]
    assert receipts.shares({"original": 0, "borrowed": 0, "upscaled": 86.76, "ai": 13.24}) == [("u", "Upscaled", 87), ("a", "AI", 13)]
    assert receipts.summary({"original": 0, "borrowed": 0, "upscaled": 100.0, "ai": 0.0}) == "100% upscaled · 0% AI"
    assert receipts.summary({"original": 96.77, "borrowed": 3.23, "ai": 0.0}) == "97% original · 3% other frames · 0% AI"
    assert not receipts.all_ai({"original": 0, "borrowed": 0, "upscaled": 100.0, "ai": 0.0})


def test_take_workflows_zip(client, tmp_path):
    """Run 9: a take's ComfyUI workflows are one Download workflows (a zip, in order), not a row per workflow."""
    c, store = client
    prj = store.create_project("Coastal spot")
    sid, vid, vdir = _remove_take(store, tmp_path, prj["id"])
    r = c.get(f"/api/shots/{sid}/versions/{vid}/workflows.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.namelist() == ["01_01_pick.json", "02_02_plate.json"]
    assert json.loads(z.read("01_01_pick.json"))["PU_Frame"]["inputs"]["image"] == "pick.png"
    assert c.get(f"/api/shots/{sid}/versions/..%2Fx/workflows.zip").status_code == 404
    assert c.get(f"/api/shots/{sid}/versions/v_nope/workflows.zip").status_code == 404
