"""The stress harness (scripts/stress_test.py): the plan, resuming, grades, pictures, the judge and the report. Nothing here
touches ComfyUI or the GPU: cells run through a fake runner, and the judge talks to a stub server."""

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stress_test.py"
_spec = importlib.util.spec_from_file_location("stress_test", _PATH)
st = importlib.util.module_from_spec(_spec)
sys.modules["stress_test"] = st
_spec.loader.exec_module(st)

STILL = ("Studio photograph, full-length shot of {who}, standing centered and facing the camera in a relaxed neutral stance.")
QA = {"coverage": 1.05, "halo": 0.8, "flicker": 0.4, "empty_new_frames": 0}


def opts(**kw):
    return argparse.Namespace(**({"force": False, "retry_failed": False, "rescore": False, "judge": False, "limit": 0} | kw))


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    """A library with a "Fun cast" project holding three pictures."""
    root = tmp_path / "library"
    monkeypatch.setattr(st.store_mod, "ROOT", root)
    d = root / "projects" / "prj_fun" / "images"
    d.mkdir(parents=True)
    items = []
    for i, colour in enumerate(("green", "silver", "red", "blue")):
        Image.new("RGB", (64, 96), colour).save(d / f"img_{i}.png")
        items.append({"id": f"img_{i}", "file": f"img_{i}.png"})
    (root / "projects" / "prj_fun" / "images.json").write_text(json.dumps(items))
    return root


def _matrix(tmp_path, cast: dict | None = None, **extra) -> Path:
    stress = tmp_path / "stress"
    stress.mkdir(exist_ok=True)
    if cast is not None:
        (stress / "cast.json").write_text(json.dumps(cast))
    m = {"stage": "stage1", "title": "Cast smoke", "seeds": [42], "tasks": ["recast"],
         "shots": [{"id": "skye-walker", "name": "Skye ridge walker", "shot_id": "shot_abc", "copy": True,
                    "params": {"frame": 40, "points": [], "text": "man", "scene": "On a ridge."}}],
         "characters": [{"from_cast": "cast.json"}]} | extra
    (stress / "matrix.json").write_text(json.dumps(m))
    return stress / "matrix.json"


CAST = {"project": "prj_fun",
        "cast": {"Alien": {"status": "done", "images": ["img_0", "img_1"], "error": None},
                 "Android": {"status": "done", "images": ["img_1", "img_2"], "error": None},
                 "Knight": {"status": "failed", "images": [], "error": "out of memory"}},
        "prompts": {"Alien": STILL.format(who="a slim humanoid alien in a silver flight suit"),
                    "Android": STILL.format(who="a chrome android in a navy suit")},
        "chosen": {"Android": {"image_id": "img_2"}}}


# --- the plan -----------------------------------------------------------------------------------------------------
def test_plan_reads_the_cast_and_merges_params(tmp_path, lib):
    m, notes = st.load_matrix(_matrix(tmp_path, CAST))
    cells = st.plan(m)
    assert [c["id"] for c in cells] == ["skye-walker__alien__recast__s42", "skye-walker__android__recast__s42"]
    alien, android = cells
    assert alien["params"] == {"frame": 40, "points": [], "text": "man", "scene": "On a ridge.",
                               "describe": "a slim humanoid alien in a silver flight suit", "seed": 42}
    assert alien["character"]["cast_image"] == {"project": "prj_fun", "id": "img_0"}  # no pick: the first picture
    assert android["character"]["cast_image"]["id"] == "img_2"  # the chosen picture
    assert any("Knight" in n and "out of memory" in n for n in notes)
    assert st.reference(android["character"]) == lib / "projects" / "prj_fun" / "images" / "img_2.png"


def test_a_cast_file_written_later_is_read_at_run_time(tmp_path, lib):
    path = _matrix(tmp_path, None)
    m, notes = st.load_matrix(path)
    assert st.plan(m) == [] and "isn't there yet" in notes[0]
    (path.parent / "cast.json").write_text(json.dumps(CAST))
    assert len(st.plan(st.load_matrix(path)[0])) == 2


def test_variants_seeds_and_tasks_without_characters(tmp_path, lib):
    path = _matrix(tmp_path, CAST, seeds=[1, 2],
                   characters=[{"from_cast": "cast.json", "variant": [0, 1], "only": ["Alien"]},
                               {"name": "Ghost lady", "describe": "a Victorian ghost lady"}],
                   tasks=["recast", {"task": "remove", "params": {"text": "car"}}])
    cells = st.plan(st.load_matrix(path)[0])
    ids = [c["id"] for c in cells]
    assert ids[:4] == ["skye-walker__alien-v1__recast__s1", "skye-walker__alien-v1__recast__s2",
                       "skye-walker__alien-v2__recast__s1", "skye-walker__alien-v2__recast__s2"]
    assert "skye-walker__ghost-lady__recast__s2" in ids
    removes = [c for c in cells if c["task"] == "remove"]
    assert [c["id"] for c in removes] == ["skye-walker__none__remove__s1", "skye-walker__none__remove__s2"]
    assert removes[0]["params"]["text"] == "man"  # the shot's own pick wins over the task's default
    ghost = next(c for c in cells if c["id"].startswith("skye-walker__ghost-lady"))
    assert st.reference(ghost["character"]) is None  # described only: Recast makes the still


def test_duplicate_cells_are_refused(tmp_path, lib):
    path = _matrix(tmp_path, None, characters=[{"name": "Finn", "describe": "a"}, {"name": "finn", "describe": "b"}])
    with pytest.raises(ValueError, match="Two cells"):
        st.plan(st.load_matrix(path)[0])


def test_who_comes_out_of_the_still_prompt():
    assert st.who_from_prompt(STILL.format(who="a knight in plate armour, a red tabard")) == "a knight in plate armour, a red tabard"
    assert st.who_from_prompt("something else") == "" and st.who_from_prompt(None) == ""


@pytest.mark.parametrize("prev,o,want", [
    (None, opts(), "run"),
    ({"status": "running"}, opts(), "run"),
    ({"status": "interrupted"}, opts(), "run"),
    ({"status": "done", "artefacts": {"contact": "contact.jpg"}}, opts(), None),
    ({"status": "done", "artefacts": {"contact": "contact.jpg"}}, opts(force=True), "run"),
    ({"status": "done"}, opts(), "post"),  # the pictures failed last time
    ({"status": "done", "artefacts": {"c": 1}}, opts(rescore=True), "post"),
    ({"status": "done", "artefacts": {"c": 1}, "character": "Alien"}, opts(judge=True), "post"),
    ({"status": "done", "artefacts": {"c": 1}, "character": "Alien", "judge": {"identity_match": 4}}, opts(judge=True), None),
    ({"status": "failed"}, opts(), None),
    ({"status": "timeout"}, opts(retry_failed=True), "run"),
    ({"status": "error"}, opts(retry_failed=True), "run"),
])
def test_resume_rules(prev, o, want):
    assert st.decide(prev, o) == want


# --- running cells (fake runner) --------------------------------------------------------------------------------------
def _clip(path: Path, box: str | None = None, frames: int = 24):
    vf = ["-vf", f"drawbox={box}:color=0xd4a020:t=fill"] if box else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x4a6a3a:s=320x180:r=24", "-frames:v", str(frames),
                    *vf, "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    _clip(d / "before.mp4")
    _clip(d / "after.mp4", "x=200:y=40:w=30:h=90")
    return d


class FakeRunner:
    """Stands in for AppRunner: outcomes by character name; a done cell gets a take folder made of synthetic clips."""

    def __init__(self, root: Path, clips: Path, outcomes: dict, stop_on: str | None = None):
        self.root, self.clips, self.outcomes, self.stop_on = root, clips, outcomes, stop_on
        self.ran, self.uploads = [], []

    def shot_for(self, spec):
        return "shot_copy"

    def upload(self, path):
        self.uploads.append(Path(path).name)
        return "up_" + Path(path).stem.replace("img_", "")

    def wait_idle(self, log):
        return 0.0

    def run(self, shot_id, task, params, timeout_s, log):
        who = params.get("describe", "")
        if self.stop_on and self.stop_on in who:
            raise st.StopRun("Waited 4:00:00 for the GPU (ComfyUI is busy with other work).")
        self.ran.append(params)
        out = next(v for k, v in self.outcomes.items() if k in who)
        if isinstance(out, Exception):
            raise out
        if out.get("status") == "done":
            vid = f"v_{len(self.ran)}"
            tdir = self.root / "shots" / shot_id / "versions" / vid
            tdir.mkdir(parents=True)
            (tdir / "before.mp4").write_bytes((self.clips / "before.mp4").read_bytes())
            (tdir / "video.mp4").write_bytes((self.clips / "after.mp4").read_bytes())
            (tdir / "meta.json").write_text(json.dumps({"details": out["details"]}))
            return {"status": "done", "take_id": vid, "job_id": "job_1", "eta_s": 415, "run_s": 300.0, "contended": False}
        return {"job_id": "job_1", "eta_s": 415, "run_s": 12.0, "contended": False} | out

    def take(self, shot_id, take_id):
        tdir = self.root / "shots" / shot_id / "versions" / take_id
        return json.loads((tdir / "meta.json").read_text()), tdir

    def interrupt(self):
        pass


def test_a_stage_runs_records_failures_and_resumes(tmp_path, lib, clips, monkeypatch):
    cast = dict(CAST, cast=dict(CAST["cast"], Rosa={"status": "done", "images": ["img_3"]}),
                prompts=dict(CAST["prompts"], Rosa=STILL.format(who="a woman in her sixties in a red wool coat")))
    m, _ = st.load_matrix(_matrix(tmp_path, cast))
    cells = st.plan(m)
    out = tmp_path / "results" / "stage1"
    runner = FakeRunner(lib, clips, {
        "alien": {"status": "done", "details": {"pixels_changed_outside_selection": 0, "gpu_seconds": 140.0, "_qa": QA}},
        "android": {"status": "failed", "error": "Couldn't find anyone in this shot. Click the person you want to recast.",
                    "failed_stage": "Finding the person"},
        "woman": RuntimeError("upload went wrong")})
    verdicts = []
    monkeypatch.setattr(st, "judge", lambda cfg, ref, frames: verdicts.append((ref.name, [f.name for f in frames]))
                        or {"identity_match": 5, "artefacts": 4, "notes": "clean", "model": "vis"})
    logs = []
    assert st.run_stage(m, cells, cells, out, runner, opts(judge=True), {"url": "x", "model": "vis"}, logs.append) == 0

    alien = json.loads((out / "skye-walker__alien__recast__s42" / "result.json").read_text())
    assert alien["status"] == "done" and alien["grade"] == "A" and alien["take_id"] == "v_1"
    assert alien["params"]["ref_upload_id"] == "up_0" and alien["qa"] == QA and alien["outside_selection"] == 0
    for f in ("contact.jpg", "contact_thumb.jpg", "sbs.mp4", "sbs.gif", "judge_1.jpg", "judge_3.jpg", "reference.jpg"):
        assert (out / alien["cell"] / f).is_file(), f
    assert verdicts == [("reference.jpg", ["judge_1.jpg", "judge_2.jpg", "judge_3.jpg"])]
    android = json.loads((out / "skye-walker__android__recast__s42" / "result.json").read_text())
    assert android["status"] == "failed" and android["grade"] == "F"
    assert android["reasons"][0].startswith("Failed at “Finding the person”: Couldn't find anyone")
    rosa = json.loads((out / "skye-walker__rosa__recast__s42" / "result.json").read_text())
    assert rosa["status"] == "error" and "upload went wrong" in rosa["error"] and rosa["grade"] == "F"
    assert (out / "report.html").is_file() and (out / "report.csv").is_file()

    runner.ran.clear()  # a second run skips everything that finished
    st.run_stage(m, cells, cells, out, runner, opts(), None, logs.append)
    assert runner.ran == []
    st.run_stage(m, cells, cells, out, runner, opts(retry_failed=True), None, logs.append)
    assert sorted(p["describe"][:9] for p in runner.ran) == ["a chrome ", "a woman i"]


def test_a_take_deleted_in_the_app_is_an_error_not_a_crash(tmp_path, lib, clips):
    m, _ = st.load_matrix(_matrix(tmp_path, CAST))
    cells = st.plan(m)
    out = tmp_path / "results" / "stage1"
    done = {"status": "done", "details": {"pixels_changed_outside_selection": 0, "_qa": QA}}
    runner = FakeRunner(lib, clips, {"alien": done, "android": done})
    assert st.run_stage(m, cells, cells, out, runner, opts(), None, lambda s: None) == 0

    class Gone(FakeRunner):
        def take(self, shot_id, take_id):
            if take_id == "v_1":
                raise FileNotFoundError(f"[Errno 2] No such file or directory: '{take_id}/meta.json'")
            return super().take(shot_id, take_id)

    gone = Gone(lib, clips, {})
    logs = []
    assert st.run_stage(m, cells, cells, out, gone, opts(rescore=True), None, logs.append) == 0  # carries on
    first = json.loads((out / cells[0]["id"] / "result.json").read_text())
    assert first["status"] == "error" and first["grade"] == "F" and first["take_id"] == "v_1"
    assert first["reasons"] == ["Lost its take: The take is gone from the library. Run with --retry-failed or --force to make a new one."]
    second = json.loads((out / cells[1]["id"] / "result.json").read_text())
    assert second["status"] == "done" and second["grade"] != "F"  # the next cell was still scored
    assert "Lost its take" in (out / "report.html").read_text() and gone.ran == []
    st._write_json(out / cells[1]["id"] / "result.json", {k: v for k, v in second.items() if k != "artefacts"})
    (gone.root / "shots" / "shot_copy" / "versions" / "v_2" / "meta.json").unlink()
    assert st.run_stage(m, cells, cells, out, gone, opts(), None, logs.append) == 0  # a plain resume re-scores it: same guard
    assert json.loads((out / cells[1]["id"] / "result.json").read_text())["status"] == "error"


def test_a_stop_marks_the_cell_and_the_next_run_carries_on(tmp_path, lib, clips):
    m, _ = st.load_matrix(_matrix(tmp_path, CAST))
    cells = st.plan(m)
    out = tmp_path / "results" / "stage1"
    done = {"status": "done", "details": {"pixels_changed_outside_selection": 0, "_qa": QA}}
    runner = FakeRunner(lib, clips, {"alien": done, "android": done}, stop_on="android")
    assert st.run_stage(m, cells, cells, out, runner, opts(), None, lambda s: None) == 2
    assert json.loads((out / cells[1]["id"] / "result.json").read_text())["status"] == "interrupted"
    runner.stop_on = None
    runner.ran.clear()
    assert st.run_stage(m, cells, cells, out, runner, opts(), None, lambda s: None) == 0
    assert [p["describe"] for p in runner.ran] == ["a chrome android in a navy suit"]


def test_limit_stops_after_n_renders(tmp_path, lib, clips):
    m, _ = st.load_matrix(_matrix(tmp_path, CAST))
    cells = st.plan(m)
    runner = FakeRunner(lib, clips, {"alien": {"status": "failed", "error": "x"}, "android": {"status": "failed", "error": "y"}})
    st.run_stage(m, cells, cells, tmp_path / "out", runner, opts(limit=1), None, lambda s: None)
    assert len(runner.ran) == 1


# --- grades -------------------------------------------------------------------------------------------------------
def _done(**kw):
    r = {"status": "done", "task": "recast", "character": "Alien", "qa": dict(QA), "outside_selection": 0,
         "timings": {"eta_s": 415, "run_s": 300}, "judge": {"identity_match": 4, "artefacts": 5, "notes": "clean"}}
    for k, v in kw.items():
        r[k] = (r.get(k) or {}) | v if isinstance(v, dict) and k in ("qa", "timings", "judge") else v
    return r


@pytest.mark.parametrize("r,want,reason", [
    (_done(), "A", None),
    (_done(judge={"identity_match": 3}), "B", "Judge: identity 3/5"),
    (_done(judge={"artefacts": 2}), "F", "artefacts 2/5"),
    (_done(judge=None), "B", "Not judged"),
    (_done(judge={"identity_match": None, "error": "the judge didn't answer (timed out)"}), "B", "timed out"),
    (_done(qa={"halo": 2.4}), "F", "Halo 2.40"),
    (_done(qa={"halo": 2.0}), "F", "Halo 2.00"),  # the limit is "under 2.0"
    (_done(qa={"flicker": 1.6}), "F", "Flicker 1.60"),
    (_done(qa={"coverage": 0.55}), "F", "smaller than the old one"),
    (_done(qa={"coverage": 1.7}), "F", "bigger than the old one"),
    (_done(qa={"empty_new_frames": 3}), "F", "3 frames with no new character"),
    (_done(outside_selection=1204), "F", "1,204 pixels changed outside"),
    (_done(timings={"run_s": 700}), "F", "Took 11:40 (limit 10:22)"),
    (_done(timings={"run_s": 700}, contended=True), "B", "Time not counted"),
    (_done() | {"qa": {}}, "B", "QA scores missing"),
    (_done() | {"qa": None}, "B", "QA scores missing"),
    (_done(qa={"halo": None}) | {"qa": {"coverage": 1.0, "flicker": 0.2, "empty_new_frames": 0}}, "B", "Halo not measured"),
    ({"status": "failed", "error": "Couldn't find anyone in this shot.", "failed_stage": "Finding the person"}, "F", "Failed at"),
    ({"status": "failed", "error": "IndexError: index 0 is out of bounds"}, "F", "isn't written for a person"),
    ({"status": "timeout", "error": "Didn't finish in 40:00."}, "F", "Timed out"),
    ({"status": "pending"}, "–", "Not run yet"),
    ({"status": "interrupted"}, "–", "Interrupted"),
])
def test_grades(r, want, reason):
    g, why, _ = st.grade(r)
    assert g == want, why
    if reason:
        assert any(reason in w for w in why), why
    if want == "A":
        assert why == []


def test_tasks_without_a_character_need_no_judge():
    r = {"status": "done", "task": "environment", "timings": {"eta_s": 300, "run_s": 200}}
    assert st.grade(r)[0] == "A"
    assert [c["key"] for c in st.checks(r)] == ["time"]


@pytest.mark.parametrize("r,want,reason", [
    ({"status": "failed", "error": "There's nobody in this shot to recast. Try a shot with a person in it.", "timings": {"run_s": 20}},
     "A", "Stopped in 0:20"),
    ({"status": "failed", "error": "There's nobody in this shot.", "timings": {"run_s": 300}}, "F", "Took 5:00 to say so"),
    ({"status": "failed", "error": "ValueError: zero-size array to reduction", "timings": {"run_s": 5}}, "F", "isn't written for a person"),
    ({"status": "done", "timings": {"run_s": 5}}, "F", "should have stopped"),
    ({"status": "failed", "error": "Something else entirely.", "timings": {"run_s": 5}}, "F", "Unexpected message"),
])
def test_expected_failures_must_be_fast_and_worded(r, want, reason):
    r = r | {"task": "recast", "expect": {"status": "failed", "within_s": 60, "message": "nobody"}}
    g, why, _ = st.grade(r)
    assert g == want and any(reason in w for w in why), why


def test_qa_is_read_tolerantly():
    assert st.read_qa({}) == {} and st.read_qa({"_qa": "broken"}) == {}
    assert st.read_qa({"_qa": {"coverage_ratio": "1.2", "halo_score": 0.5, "flicker": None, "empty_frames": 2.0, "x": 1}}) == {
        "coverage": 1.2, "halo": 0.5, "empty_new_frames": 2}
    assert st.read_qa({"_qa": {"halo": float("nan"), "flicker": "inf", "coverage": True}}) == {}  # not measured, not a failure


# --- pictures -----------------------------------------------------------------------------------------------------
def test_pictures_zoom_on_what_changed(tmp_path, clips):
    tdir = tmp_path / "shots" / "shot_x" / "versions" / "v_1"
    tdir.mkdir(parents=True)
    (tdir / "before.mp4").write_bytes((clips / "before.mp4").read_bytes())
    (tdir / "video.mp4").write_bytes((clips / "after.mp4").read_bytes())
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 96), "green").save(ref)
    out = tmp_path / "cell"
    out.mkdir()
    a = st.artefacts(tdir, out, "Shot · Alien · Recast · seed 42", ref)
    assert a["frames"] == 24 and a["sheet_frames"] == [0, 3, 7, 10, 13, 16, 20, 23]
    x0, y0, x1, y1 = a["focus"]
    assert x0 <= 200 / 320 and x1 >= 230 / 320 and y0 <= 40 / 180 and y1 >= 130 / 180  # the changed box, padded
    assert x1 - x0 < 0.5  # a close-up, not the whole frame
    sheet = Image.open(out / "contact.jpg")
    assert sheet.width > 8 * 300 and sheet.height > 4 * 169  # before, after and two close-up rows
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=height", "-of", "csv=p=0", str(out / "sbs.mp4")],
                           capture_output=True, text=True).stdout
    assert probe.strip() == "540"
    assert (out / "sbs.gif").stat().st_size > 0 and (out / "reference.jpg").is_file()
    assert Image.open(out / "judge_2.jpg").width < Image.open(out / "judge_2.jpg").height  # cut close on the standing figure


def test_no_change_means_no_close_ups(clips):
    frames = st.grab(clips / "before.mp4", [0, 5, 10], 320)
    assert st.focus_boxes(frames, frames) == (None, [None, None, None])


# --- the judge ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,want", [
    ('{"identity_match": 4, "artefacts": 3, "notes": "Faint glow on the left arm."}',
     {"identity_match": 4, "artefacts": 3, "notes": "Faint glow on the left arm."}),
    ('<think>hmm</think>```json\n{"identity_match": "5", "artifacts": 4.0, "notes": "clean"}\n```',
     {"identity_match": 5, "artefacts": 4, "notes": "clean"}),
    ('{"identity_match": 6, "artefacts": 3, "notes": ""}', None),
    ('{"identity_match": 3.5, "artefacts": 3, "notes": ""}', None),
    ('{"identity_match": true, "artefacts": 3, "notes": ""}', None),
    ('{"identity_match": 4, "artefacts": 3, "notes": "左臂有光晕"}', None),
    ("The character looks great, I'd say 4 out of 5.", None),
    ("", None),
    ("[1, 2]", None),
])
def test_verdicts_are_strict(text, want):
    assert st.parse_verdict(text) == want


def _stub(replies, delay: float = 0.0):
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            time.sleep(delay)
            reply = replies[min(len(calls) - 1, len(replies) - 1)]
            if reply == 400:
                self.send_response(400)
                self.end_headers()
                return
            out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):  # noqa: N802
            out = json.dumps({"data": [{"id": "google/vision-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def _pictures(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (200, 300), (40 * i, 90, 60)).save(p)
        paths.append(p)
    return paths[0], paths[1:]


def test_judge_falls_back_without_structured_output_and_rejects_garbage(tmp_path):
    srv, calls = _stub([400, "Looks good to me!", '{"identity_match": 4, "artefacts": 2, "notes": "A ghost of the hiker."}'])
    ref, frames = _pictures(tmp_path)
    cfg = {"url": f"http://127.0.0.1:{srv.server_port}", "model": "google/vision-model", "timeout": 5}
    v = st.judge(cfg, ref, frames)
    srv.shutdown()
    assert {k: v[k] for k in ("identity_match", "artefacts", "notes", "model")} == {
        "identity_match": 4, "artefacts": 2, "notes": "A ghost of the hiker.", "model": "google/vision-model"}
    assert len(calls) == 3
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    images = [p for p in calls[0]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 4 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert calls[0]["model"] == "google/vision-model" and calls[0]["temperature"] == 0
    assert "JSON object only" in calls[2]["messages"][-1]["content"]  # the retry asks again, plainly


def test_judge_gives_up_in_words(tmp_path):
    ref, frames = _pictures(tmp_path)
    srv, calls = _stub(["no idea", "still no", "nope"])
    v = st.judge({"url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "m", "timeout": 5}, ref, frames)
    srv.shutdown()
    assert v["error"] == "the judge's answer wasn't a usable verdict" and len(calls) == 3
    slow, _ = _stub(['{"identity_match": 4, "artefacts": 4, "notes": ""}'], delay=1.5)
    v = st.judge({"url": f"http://127.0.0.1:{slow.server_port}", "model": "m", "timeout": 0.3}, ref, frames)
    slow.shutdown()
    assert v["error"].startswith("the judge didn't answer")
    assert st.judge({"url": "http://127.0.0.1:9", "model": "m", "timeout": 1}, ref, frames)["error"].startswith("the judge didn't answer")


def test_judge_settings_and_readiness(monkeypatch):
    import pickups.app.comfy as comfy
    monkeypatch.setattr(comfy, "load_settings", lambda: {"director": {"url": "http://127.0.0.1:1234/v1", "model": "text-model",
                                                                      "vision_model": "google/vision-model"}})
    assert st.judge_config()["model"] == "google/vision-model"
    assert st.judge_config(model="other")["model"] == "other"
    srv, _ = _stub([""])
    url = f"http://127.0.0.1:{srv.server_port}"
    assert st.judge_ready({"url": url, "model": "google/vision-model"}) is None
    assert "isn't available" in st.judge_ready({"url": url, "model": "missing"})
    srv.shutdown()
    assert "isn't answering" in st.judge_ready({"url": "http://127.0.0.1:9", "model": "m"}, timeout=1)


# --- the report ---------------------------------------------------------------------------------------------------
def test_report_shows_every_cell(tmp_path, lib):
    m, _ = st.load_matrix(_matrix(tmp_path, CAST, characters=[{"from_cast": "cast.json"}, {"name": "Finn", "describe": "a"},
                                                                {"name": "Rosa", "describe": "b"}]))
    cells = st.plan(m)
    out = tmp_path / "results" / "stage1"
    results = {
        cells[0]["id"]: _done() | {"take_id": "v_good", "artefacts": {"contact": "contact.jpg", "thumb": "contact_thumb.jpg",
                                                                       "sbs": "sbs.mp4", "gif": "sbs.gif", "reference": "reference.jpg"}},
        cells[1]["id"]: _done(judge={"identity_match": 3, "notes": "The visor reads silver, not gold."}),
        cells[2]["id"]: {"status": "failed", "task": "recast", "character": "Finn", "error": "Couldn't find anyone in this shot."},
    }
    for cid, r in results.items():
        (out / cid).mkdir(parents=True)
        base = st.base_result(next(c for c in cells if c["id"] == cid), m)
        (out / cid / "result.json").write_text(json.dumps(base | r | {"timings": r.get("timings", {}) | {"gpu_s": 140}}))
    Image.new("RGB", (760, 300), "navy").save(out / cells[0]["id"] / "contact_thumb.jpg")
    Image.new("RGB", (100, 150), "green").save(out / cells[0]["id"] / "reference.jpg")
    rows = st.write_report(out, m, cells)

    assert [r["grade"] for r in rows] == ["A", "B", "F", "–"]
    page = (out / "report.html").read_text()
    assert page.count('<article class="card') == 4
    assert 'class="card g-A"' in page and 'class="card g-B"' in page and 'class="card g-F"' in page and 'class="card g-P"' in page
    assert page.count("data:image/jpeg;base64,") == 2  # the contact sheet and the reference, inside the page
    assert f'href="{cells[0]["id"]}/sbs.mp4"' in page  # files next to the report, by relative links
    assert "The visor reads silver, not gold." in page and "Couldn&#x27;t find anyone in this shot." in page
    assert str(tmp_path) not in page and "<script src" not in page and "<link" not in page
    with (out / "report.csv").open() as f:
        lines = f.read().splitlines()
    assert lines[0].split(",")[:8] == ["stage", "cell", "shot", "character", "task", "seed", "status", "grade"]
    assert len(lines) == 5 and ",A," in lines[1] and ",F," in lines[3]


def test_how_recast_made_the_take_reads_plainly():
    assert st.how_text({"how": {"reference": "front view cropped from a 3-view sheet", "reference_mask": True,
                                "object_indices": "1", "people_on_first_frame": 2, "scail_long": 832}}) == (
        "Front view cropped from a 3-view sheet · 2 people in the crop, only number 2 from the left re-rendered · SCAIL-2 at 832 px")
    assert "ComfyUI refused ours" in st.how_text({"how": {"reference_mask": False, "object_indices": "", "people_on_first_frame": 3}})
    assert "all re-rendered" in st.how_text({"how": {"object_indices": "", "people_on_first_frame": 3}})
    assert "followed on their own" in st.how_text({"how": {"object_indices": "0", "people_on_first_frame": 2, "new_track": True}})
    assert "couldn't follow" in st.how_text({"how": {"object_indices": "0", "people_on_first_frame": 2, "new_track": False}})
    assert st.how_text({}) == "" and st.how_text({"how": {"people_on_first_frame": 1}}) == ""


# --- the command --------------------------------------------------------------------------------------------------
def test_dry_run_changes_nothing(tmp_path, lib, capsys):
    path = _matrix(tmp_path, CAST)
    before = sorted(p for p in tmp_path.rglob("*"))
    assert st.main([str(path), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "2 cells" in text and "run   skye-walker__alien__recast__s42  ref img_0.png" in text
    assert "library shot shot_abc (MISSING)" in text and "Knight" in text
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_the_app_runner_copies_the_shot_once_and_uploads_once(tmp_path, lib, monkeypatch):
    from pickups.app.store import Store
    src = tmp_path / "walker.mp4"
    _clip(src, frames=6)
    store = Store(lib)
    sid = store.add_shot(src, "Skye ridge walker")["id"]
    runner = st.AppRunner(tmp_path / "results", "Stress test", "http://127.0.0.1:9", wait=False)
    spec = {"id": "skye-walker", "shot_id": sid, "copy": True, "name": "Skye ridge walker"}
    copy = runner.shot_for(spec)
    assert copy != sid and runner.shot_for(spec) == copy
    assert store.shot_meta(copy)["project_id"] == runner.project_id()
    assert json.loads((store.project_dir(runner.project_id()) / "meta.json").read_text())["name"] == "Stress test"
    assert runner.shot_for(spec | {"copy": False}) == sid
    pic = lib / "projects" / "prj_fun" / "images" / "img_0.png"
    up = runner.upload(pic)
    assert runner.upload(pic) == up and store.upload_path(up).read_bytes() == pic.read_bytes()
    again = st.AppRunner(tmp_path / "results", "Stress test", "http://127.0.0.1:9", wait=False)  # remembered across runs
    assert again.shot_for(spec) == copy and again.upload(pic) == up
    assert runner._jm is None  # nothing started a GPU worker


class FakeJM:
    """The job queue's side of a cell: the job dict changes on a thread, as the GPU worker does."""

    def __init__(self, steps, honour_cancel=True):
        self.steps, self.honour_cancel, self.canceled = steps, honour_cancel, []

    def submit(self, shot_id, task, params):
        self.job = {"id": "job_1", "status": "queued", "stage": "Waiting for the GPU", "eta_s": 0, "error": None}

        def work():
            for delay, change in self.steps:
                time.sleep(delay)
                if self.job["status"] == "canceled":
                    return
                self.job.update(change)

        threading.Thread(target=work, daemon=True).start()
        return self.job

    def cancel(self, job_id):
        self.canceled.append(job_id)
        if self.honour_cancel:
            self.job.update(status="canceled", stage="Canceled")


def _app_runner(tmp_path, lib, steps, **kw):
    runner = st.AppRunner(tmp_path / "results", "Stress test", "http://127.0.0.1:9", wait=False, poll_s=0.02)
    runner._jm = FakeJM(steps, **kw)
    runner.cancel_wait_s = 0.3
    return runner


def test_the_app_runner_follows_a_job_to_the_end(tmp_path, lib):
    run = [(0.05, {"status": "running", "stage": "Finding the person"}), (0.05, {"stage": "Performing the scene"})]
    stages = []
    out = _app_runner(tmp_path, lib, run + [(0.05, {"status": "done", "stage": "Done", "result_version_id": "v_9"})]).run(
        "shot_1", "recast", {}, 30, stages.append)
    assert out["status"] == "done" and out["take_id"] == "v_9" and stages == ["Finding the person", "Performing the scene"]
    out = _app_runner(tmp_path, lib, run[:1] + [(0.05, {"status": "failed", "stage": "Something went wrong",
                                                        "error": "There's nobody in this shot."})]).run("shot_1", "recast", {}, 30, print)
    assert out == out | {"status": "failed", "error": "There's nobody in this shot.", "failed_stage": "Finding the person"}


def test_the_app_runner_stops_a_cell_that_runs_too_long(tmp_path, lib):
    forever = [(0.02, {"status": "running", "stage": "Performing the scene"})]
    runner = _app_runner(tmp_path, lib, forever)
    out = runner.run("shot_1", "recast", {}, 0.3, lambda s: None)
    assert out["status"] == "timeout" and out["failed_stage"] == "Performing the scene" and not out["stuck"]
    assert runner._jm.canceled == ["job_1"] and runner.job is None
    out = _app_runner(tmp_path, lib, forever, honour_cancel=False).run("shot_1", "recast", {}, 0.3, lambda s: None)
    assert out["status"] == "timeout" and out["stuck"]  # ComfyUI ignored the cancel: the run stops after this cell
    assert st.grade({"status": "timeout", "error": out["error"]})[0] == "F"


def test_ctrl_c_cancels_the_job_in_flight(tmp_path, lib):
    runner = _app_runner(tmp_path, lib, [(0.02, {"status": "running", "stage": "Performing the scene"})])
    runner.job = runner.jm.submit("shot_1", "recast", {})
    time.sleep(0.1)
    runner.interrupt()
    assert runner._jm.canceled == ["job_1"] and runner._jm.job["status"] == "canceled"
