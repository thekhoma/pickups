"""Character sheets (pickups/sheets.py, tasks/sheet.py): FRONT | BACK | CLOSE-UP in one 16:9 picture, kept with a cast
member. ComfyUI is stood in for: the fake editor answers each instruction with its own colour of figure."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pickups import sheets

GREY = (205, 205, 205)
BLUE, GREEN, RED, TAN = (40, 60, 150), (40, 140, 60), (170, 40, 40), (190, 140, 100)


def studio(size=(832, 1280), colour=BLUE, top=0.06, bottom=0.94, backdrop=GREY, seed=0, people=1) -> Image.Image:
    """A studio still: a standing figure (head, body, legs) on a plain backdrop with a little noise and a soft vignette."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.array(backdrop, np.float32) - 12 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)[..., None]
    img = base + rng.normal(0, 1.2, (h, w, 1))
    tall = (bottom - top) * h
    for k in range(people):
        cx = w * (k + 1) / (people + 1)
        y0 = top * h
        head = ((xx - cx) / (0.07 * tall)) ** 2 + ((yy - (y0 + 0.07 * tall)) / (0.075 * tall)) ** 2 <= 1
        body = (np.abs(xx - cx) <= 0.13 * tall) & (yy >= y0 + 0.14 * tall) & (yy <= y0 + 0.55 * tall)
        legs = ((np.abs(xx - (cx - 0.06 * tall)) <= 0.045 * tall) | (np.abs(xx - (cx + 0.06 * tall)) <= 0.045 * tall)) & (
            yy > y0 + 0.55 * tall) & (yy <= bottom * h)
        fig = head | body | legs
        img[fig] = np.array(colour, np.float32) + rng.normal(0, 6, (int(fig.sum()), 1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def busy(size=(1280, 720), seed=1) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[1] // 40, size[0] // 40, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BILINEAR)


# --- what the request may be ----------------------------------------------------------------------------------------


def test_request_wants_exactly_one_source():
    with pytest.raises(ValueError, match="Add a picture of them, or describe them"):
        sheets.request({"name": "Finn"})
    with pytest.raises(ValueError, match="Add a picture"):
        sheets.request({"upload_id": "  ", "describe": "", "name": "Finn"})  # blanks aren't sources
    for two in ({"upload_id": "up_1", "describe": "a man"}, {"image_id": "img_1", "cast_id": "cast_1"},
                {"upload_id": "up_1", "image_id": "img_1"}):
        with pytest.raises(ValueError, match="not both"):
            sheets.request(two | {"name": "Finn"})
    with pytest.raises(ValueError, match="JSON object"):
        sheets.request(["up_1"])
    with pytest.raises(ValueError, match="JSON object"):
        sheets.request(None)


def test_request_names_seeds_and_descriptions():
    assert sheets.request({"describe": "  a knight   in armour ", "name": "  Sir   Kay "}) == {"describe": "a knight in armour", "name": "Sir Kay"}
    with pytest.raises(ValueError, match="Give them a name"):
        sheets.request({"upload_id": "up_0123456789"})
    # Upgrading a cast member keeps its own name: none needed.
    assert sheets.request({"cast_id": "cast_1", "seed": "12"}) == {"cast_id": "cast_1", "seed": 12}
    assert sheets.request({"image_id": "img_1", "name": "Finn", "seed": None}) == {"image_id": "img_1", "name": "Finn"}
    for bad in ("abc", "1.5", -1):
        with pytest.raises(ValueError, match="seed"):
            sheets.request({"image_id": "img_1", "name": "Finn", "seed": bad})
    with pytest.raises(ValueError, match="under 600"):
        sheets.request({"describe": "x" * 601, "name": "Finn"})


# --- is it already a clean front view? -------------------------------------------------------------------------------


def test_a_studio_still_is_a_front_view():
    assert sheets.front_view(studio())
    assert sheets.front_view(np.asarray(studio(colour=RED, seed=3)))
    assert sheets.front_view(sheets.load(studio(size=(1664, 2560))))  # a 4 MP still, at working size


@pytest.mark.parametrize("picture", [
    studio(top=0.0),                         # head through the top of the frame
    studio(top=0.35, bottom=1.4),            # a close-up: the body runs out of the bottom
    studio(top=0.45, bottom=0.94),           # someone small, far down the frame
    studio(size=(1600, 1280), people=2),     # two people (or a sheet already)
    studio(backdrop=(40, 40, 40), colour=TAN),  # a dark set: the sheet's backdrop is light grey
    busy(),                                  # an ordinary photo
], ids=["head-cut", "close-up", "small", "two-people", "dark-backdrop", "busy-photo"])
def test_anything_else_is_not(picture):
    assert not sheets.front_view(picture)


def test_load_works_at_about_a_megapixel(tmp_path):
    big = sheets.load(studio(size=(3000, 4500)))
    assert 0.8e6 <= big.width * big.height <= 1.2e6 and big.width % 16 == 0 and big.height % 16 == 0
    small = sheets.load(studio(size=(300, 450)))
    assert 0.8e6 <= small.width * small.height <= 1.2e6
    assert sheets.load(studio()).size == (832, 1280)  # already the working size: untouched
    rgba = Image.new("RGBA", (900, 1200), (0, 0, 0, 0))
    rgba.paste((200, 0, 0, 255), (300, 100, 600, 1100))
    rgba.save(tmp_path / "cutout.png")
    out = sheets.load(tmp_path / "cutout.png")
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (200, 200, 200)  # transparency on light grey


def test_portrait_canvas_keeps_the_whole_picture():
    wide = busy((1600, 900))
    p = sheets.portrait(wide)
    assert abs(p.width / p.height - 2 / 3) < 0.02 and 0.8e6 <= p.width * p.height <= 1.2e6
    tall = studio()
    assert sheets.portrait(tall) is tall


# --- the composite -----------------------------------------------------------------------------------------------------


def test_panels_are_equal_with_8px_gaps():
    boxes = sheets.panels()
    assert len(boxes) == 3 and boxes[0][0] == 0 and boxes[-1][2] == 2000
    widths = [x1 - x0 for x0, _, x1, _ in boxes]
    assert max(widths) - min(widths) <= 1
    assert [boxes[i + 1][0] - boxes[i][2] for i in range(2)] == [8, 8]
    assert all(y0 == 0 and y1 == 1125 for _, y0, _, y1 in boxes)


def test_compose_puts_front_back_close_up_in_order():
    sheet = sheets.compose(studio(colour=BLUE), studio(colour=GREEN), Image.new("RGB", (832, 1280), RED))
    assert sheet.size == (2000, 1125)
    a = np.asarray(sheet).astype(int)
    for (x0, _, x1, _), colour in zip(sheets.panels(), (BLUE, GREEN, RED)):
        cx = (x0 + x1) // 2
        assert np.abs(a[600, cx] - colour).max() < 40  # the figure's middle (the close-up fills its panel)
    for x0, _, _, _ in sheets.panels()[1:]:
        assert (a[:, x0 - 8:x0] == 255).all()  # white gaps, top to bottom


def test_full_length_views_are_never_cropped_top_or_bottom():
    # A tall 9:16 view (narrower than its panel) keeps its head and shoes: padded at the sides instead.
    tall = Image.new("RGB", (540, 960), GREY)
    tall.paste(RED, (200, 0, 340, 6))      # the top of the head
    tall.paste(GREEN, (200, 954, 340, 960))  # the shoes
    out = np.asarray(sheets.fit_height(tall, (661, 1125))).astype(int)
    assert out.shape == (1125, 661, 3)
    assert np.abs(out[1, 330] - RED).max() < 30 and np.abs(out[-2, 330] - GREEN).max() < 30
    assert np.abs(out[500, 5] - GREY).max() < 6  # side padding in its own backdrop colour


# --- building one, with ComfyUI stood in for -------------------------------------------------------------------------


class _Out:
    def __init__(self, picture):
        self.picture = picture

    def to_file(self, path):
        self.picture.save(path)


class _Job:
    def __init__(self, node, picture):
        self.node, self.picture = node, picture

    def get_outputs(self, node):
        return [_Out(self.picture)] if node == self.node else []


def instruction(graph: dict) -> str:
    node = graph.get("PU_Instruction") or graph["PU_Prompt"]
    return node["inputs"].get("prompt") or node["inputs"].get("text")


def seed_of(graph: dict) -> int:
    i = graph["PU_Seed"]["inputs"]
    return i.get("noise_seed", i.get("seed"))


class FakeComfy:
    """run(graph) like a TaskContext: FLUX 3 klein gives a blue studio still, the editor a figure coloured by the view."""

    def __init__(self, still=None):
        self.graphs: list[dict] = []
        self.stages: list[str] = []
        self.still = still or studio(colour=BLUE, seed=7)
        self.client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: f"asset:{Path(p).name}"))

    def stage(self, label):
        self.stages.append(label)

    def run(self, graph):
        self.graphs.append(graph)
        if "PU_Out" in graph:
            return _Job("PU_Out", self.still)
        text = instruction(graph)
        view = "back" if text.startswith(sheets.BACK[:30]) else "face" if text.startswith(sheets.FACE[:30]) else "front"
        picture = {"front": studio(colour=BLUE, seed=11), "back": studio(colour=GREEN, seed=12),
                   "face": Image.new("RGB", (832, 1280), RED)}[view]
        return _Job("PU_Plate", picture)


def test_a_clean_front_view_skips_the_front_edit(tmp_path):
    src = tmp_path / "finn.png"
    studio(colour=BLUE).save(src)
    fake = FakeComfy()
    made = sheets.make(fake, tmp_path / "out", picture=src, seed=1234)
    assert fake.stages == [sheets.STAGE_BACK, sheets.STAGE_FACE, sheets.STAGE_SHEET]
    assert [instruction(g) for g in fake.graphs] == [sheets.BACK, sheets.FACE]
    assert all(g["PU_Frame"]["inputs"]["image"] == "asset:front.png" for g in fake.graphs)  # both from the front view
    assert {seed_of(g) for g in fake.graphs} == {1234}
    assert not made.made_front and made.editor == "Qwen-Image 2.1" and made.models == ["Qwen-Image 2.1"]
    assert made.source == src and set(made.views) == {"front", "back", "face"}
    assert all(p.is_file() for p in made.views.values())
    with Image.open(made.sheet) as im:
        assert im.size == (2000, 1125) and im.format == "JPEG"


def test_a_photo_gets_a_front_view_first(tmp_path):
    src = tmp_path / "selfie.jpg"
    busy((1600, 900)).save(src)
    fake = FakeComfy()
    made = sheets.make(fake, tmp_path / "out", picture=src, seed=5)
    assert fake.stages == [sheets.STAGE_FRONT, sheets.STAGE_BACK, sheets.STAGE_FACE, sheets.STAGE_SHEET]
    assert [instruction(g) for g in fake.graphs] == [sheets.FRONT, sheets.BACK, sheets.FACE]
    assert fake.graphs[0]["PU_Frame"]["inputs"]["image"] == "asset:front_in.png"
    with Image.open(tmp_path / "out" / "front_in.png") as im:  # a landscape photo goes in on a portrait canvas
        assert im.width < im.height
    assert made.made_front and {seed_of(g) for g in fake.graphs} == {5}
    a = np.asarray(Image.open(made.sheet)).astype(int)
    for (x0, _, x1, _), colour in zip(sheets.panels(), (BLUE, GREEN, RED)):
        assert np.abs(a[600, (x0 + x1) // 2] - colour).max() < 40


def test_words_make_a_flux_still_then_the_views(tmp_path):
    fake = FakeComfy()
    made = sheets.make(fake, tmp_path / "out", still_prompt="Studio photograph, full-length shot of a knight", seed=9)
    flux = fake.graphs[0]
    assert flux["PU_Prompt"]["inputs"]["text"] == "Studio photograph, full-length shot of a knight"
    assert (flux["flat"]["inputs"]["width"], flux["flat"]["inputs"]["height"]) == sheets.STILL_SIZE
    assert {seed_of(g) for g in fake.graphs} == {9}
    assert fake.stages[0] == sheets.STAGE_FRONT and not made.made_front  # FLUX's still is already a clean front view
    assert made.models == ["FLUX 3 klein", "Qwen-Image 2.1"] and made.source == tmp_path / "out" / "still.png"


def test_the_still_asks_for_the_flux_file_setup_installs(tmp_path):
    """Setup downloads FLUX 3 klein under the name in the manifest; the still must ask ComfyUI for that file."""
    from pickups.app.manifest import IMAGE_MODEL_FILES

    have = set(IMAGE_MODEL_FILES["flux3-klein"])
    fake = FakeComfy()
    fake.conn = SimpleNamespace(model_files=lambda: have)
    sheets.make(fake, tmp_path / "out", still_prompt="Studio photograph, full-length shot of a knight", seed=9)
    asked = {v for node in fake.graphs[0].values() for k, v in node["inputs"].items() if k in ("unet_name", "clip_name", "vae_name")}
    assert asked and asked <= have


def test_the_close_up_names_a_helmet_only_when_the_description_does(tmp_path):
    plain = sheets.face_instruction("a woman in a red coat, full length")
    gear = sheets.face_instruction("a pilot in a white suit, helmet and visor")
    assert "helmet" not in plain and "visor" not in plain and plain == sheets.FACE
    assert gear == sheets.FACE_GEAR and "helmet" in gear and "visor" in gear
    assert sheets.face_instruction("a figure in a mask") == sheets.FACE_GEAR
    fake = FakeComfy()
    sheets.make(fake, tmp_path / "gear", still_prompt="Studio photograph of a pilot in a helmet and visor", seed=4)
    assert sheets.FACE_GEAR in [instruction(g) for g in fake.graphs]
    bare = FakeComfy()
    sheets.make(bare, tmp_path / "plain", still_prompt="Studio photograph of a woman in a red coat", seed=4)
    assert sheets.FACE in [instruction(g) for g in bare.graphs]
    assert sheets.FACE_GEAR not in [instruction(g) for g in bare.graphs]


def test_commercial_safe_uses_edit_2511(tmp_path):
    src = tmp_path / "finn.png"
    studio().save(src)
    fake = FakeComfy()
    made = sheets.make(fake, tmp_path / "out", picture=src, seed=3, commercial_safe=True)
    assert made.editor == "Qwen-Image-Edit 2511" and made.models == ["Qwen-Image-Edit 2511"]
    assert all(g["PU_Instruction"]["class_type"] == "TextEncodeQwenImageEditPlus" for g in fake.graphs)
    assert all((g["scaled"]["inputs"]["width"], g["scaled"]["inputs"]["height"]) == (832, 1280) for g in fake.graphs)


def test_an_unreadable_picture_says_so(tmp_path):
    bad = tmp_path / "notes.png"
    bad.write_bytes(b"not a picture")
    with pytest.raises(RuntimeError, match="can't be opened"):
        sheets.make(FakeComfy(), tmp_path / "out", picture=bad)


# --- kept with the cast ----------------------------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import pickups.app.store as st
    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    return st.Store(tmp_path / "library")


def _built(tmp_path, tag="a") -> tuple[dict, Path, Path]:
    d = tmp_path / f"built_{tag}"
    d.mkdir()
    views = {}
    for v, colour in zip(sheets.VIEWS, (BLUE, GREEN, RED)):
        views[v] = d / f"{v}.png"
        studio(colour=colour).save(views[v])
    sheet = d / "sheet.jpg"
    sheets.compose(*(Image.open(views[v]) for v in sheets.VIEWS)).save(sheet, quality=92)
    source = d / "source.jpg"
    busy().save(source)
    return views, sheet, source


def test_a_new_member_keeps_the_sheet_views_and_source(store, tmp_path):
    pid = store.create_project("P")["id"]
    views, sheet, source = _built(tmp_path)
    c = store.save_sheet(pid, views, sheet, name="Finn", line="Red curls", source=source,
                         meta={"editor": "Qwen-Image 2.1", "models": ["Qwen-Image 2.1"], "seed": 7, "url": "/nope"})
    d = store.project_dir(pid) / "cast"
    assert c["name"] == "Finn" and c["kind"] == "person" and c["line"] == "Red curls"
    assert c["file"].endswith(".jpg") and c["url"] == c["sheet"]["url"] and c["sheet"]["url"].startswith(f"/media/projects/{pid}/cast/")
    assert set(c["sheet"]["views"]) == {"front", "back", "face"} and all(u.endswith(".png") for u in c["sheet"]["views"].values())
    assert c["sheet"]["editor"] == "Qwen-Image 2.1" and c["sheet"]["seed"] == 7
    assert c["sheet"]["licences"] == [{"name": "Qwen-Image 2.1", "licence": "Qwen Research License", "commercial": False}]
    assert (d / c["sheet"]["source_file"]).read_bytes() == source.read_bytes()
    with Image.open(d / f"{c['id']}_thumb.jpg") as t:  # chips show the front view, not the composite's middle panel
        assert t.width < t.height
    # On disk: files only, the urls are worked out on read.
    disk = json.loads((store.project_dir(pid) / "cast.json").read_text())[0]
    assert "url" not in disk and not {"url", "thumb_url", "views", "licences"} & set(disk["sheet"])
    assert disk["sheet"]["view_files"]["front"] == f"{c['id']}_front_{disk['sheet']['file'].split('_')[-1][:-4]}.png"
    # Editing the member keeps its sheet.
    c2 = store.update_cast(pid, c["id"], line="Red curls, green bomber")
    assert c2["sheet"]["views"] == c["sheet"]["views"] and c2["line"] == "Red curls, green bomber"
    with pytest.raises(ValueError, match="already a cast entry called finn"):
        store.save_sheet(pid, views, sheet, name="finn", source=source)
    with pytest.raises(ValueError, match="name"):
        store.save_sheet(pid, views, sheet, name="  ", source=source)


def test_redo_and_upgrade(store, tmp_path):
    pid = store.create_project("P")["id"]
    d = store.project_dir(pid) / "cast"
    plain = store.add_cast(pid, b"\x89PNG photo", "rosa.png", "Rosa", "person", "Silver bun")
    assert "sheet" not in plain and store.sheet_source(pid, plain["id"]) == d / plain["file"]
    views, sheet, _ = _built(tmp_path, "a")
    up = store.save_sheet(pid, views, sheet, cast_id=plain["id"], name="ignored", meta={"seed": 1})
    assert up["name"] == "Rosa" and up["line"] == "Silver bun" and up["file"] != plain["file"]
    assert up["sheet"]["source_file"] == plain["file"] and (d / plain["file"]).is_file()  # the photo stays, as the source
    assert store.sheet_source(pid, plain["id"]) == d / plain["file"]
    first = set(up["sheet"]["view_files"].values()) | {up["file"]}
    views2, sheet2, _ = _built(tmp_path, "b")
    redo = store.save_sheet(pid, views2, sheet2, cast_id=plain["id"], meta={"seed": 2})
    assert redo["sheet"]["seed"] == 2 and redo["sheet"]["source_file"] == plain["file"]
    assert not any((d / f).exists() for f in first)  # the older sheet's files went
    assert all((d / f).is_file() for f in redo["sheet"]["view_files"].values())
    assert len(store.cast(pid)) == 1
    with pytest.raises(FileNotFoundError):
        store.save_sheet(pid, views2, sheet2, cast_id="cast_missing")
    with pytest.raises(FileNotFoundError):
        store.sheet_source(pid, "cast_missing")


def test_importing_a_member_brings_its_sheet(store, tmp_path):
    a, b = store.create_project("A")["id"], store.create_project("B")["id"]
    views, sheet, source = _built(tmp_path)
    c = store.save_sheet(a, views, sheet, name="Finn", source=source, meta={"editor": "Qwen-Image 2.1"})
    copy = store.import_cast(b, a, c["id"])
    assert copy["id"] != c["id"] and copy["sheet"]["editor"] == "Qwen-Image 2.1" and set(copy["sheet"]["views"]) == set(sheets.VIEWS)
    d = store.project_dir(b) / "cast"
    assert sorted(p.name for p in d.iterdir() if not p.name.endswith("_thumb.jpg")) == sorted(
        [copy["file"], copy["sheet"]["small"], copy["sheet"]["source_file"], *copy["sheet"]["view_files"].values()])


# --- the job, end to end (the real task and queue, ComfyUI stood in for) ----------------------------------------------


def test_the_sheet_job_adds_a_cast_member(store, tmp_path, monkeypatch):
    import pickups.app.comfy as comfy
    import pickups.app.jobs as jobs_mod
    from pickups.tasks import TASKS

    pid = store.create_project("P")["id"]
    up = store.add_upload(b"", "finn.png")
    studio().save(store.upload_path(up["id"]))
    fake = FakeComfy()
    monkeypatch.setattr(jobs_mod, "run_graph", lambda client, graph, on_progress=None, cancel=None: fake.run(graph))
    monkeypatch.setattr(comfy, "load_settings", lambda: {})
    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: fake.client, comfy_url="http://127.0.0.1:9")
    jm = jobs_mod.JobManager(store, conn, {"sheet": TASKS["sheet"]})
    events = []
    jm.broadcast = lambda kind, data: events.append((kind, data))
    job = jm.submit(None, "sheet", {"upload_id": up["id"], "name": "Finn", "seed": 42}, project_id=pid)
    assert job["stages"] == [s.label for s in sheets.STAGES] and job["eta_s"] == 40
    for _ in range(400):
        if jm.jobs[job["id"]]["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    done = jm.jobs[job["id"]]
    assert done["status"] == "done", done.get("error")
    c = next(x for x in store.cast(pid) if x["id"] == done["result_cast_id"])
    assert c["name"] == "Finn" and c["sheet"]["made_from"] == "upload" and c["sheet"]["seed"] == 42
    assert c["sheet"]["job_id"] == job["id"] and c["sheet"]["made_front"] is False
    assert ("project", {"id": pid, "images": [], "cast": [c["id"]]}) in events
    assert not list((store.project_dir(pid) / "images").glob("_work_*"))  # the work folder is cleaned up


# --- the endpoint ------------------------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import pickups.app.server as server
    import pickups.app.store as st
    from pickups.app.jobs import Stage, TaskSpec

    lib = tmp_path / "library"
    monkeypatch.setattr(st, "ROOT", lib)
    monkeypatch.setattr(server, "ROOT", lib)
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "EXAMPLES", tmp_path / "examples")
    store = st.Store(lib)
    monkeypatch.setattr(server, "Store", lambda: store)
    # Never a real GPU job: the queued sheet does nothing.
    monkeypatch.setitem(server.TASKS, "sheet", TaskSpec([Stage("Making the back view", 1)], lambda ctx: {}, scope="project"))
    return TestClient(server.create_app("http://127.0.0.1:9")), store  # a dead local address: no ComfyUI


def test_sheet_endpoint_queues_a_job(client):
    c, store = client
    pid = store.create_project("P")["id"]
    up = store.add_upload(b"x", "me.jpg")
    r = c.post(f"/api/projects/{pid}/cast/sheet", json={"upload_id": up["id"], "name": "Me", "seed": 3})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["task"] == "sheet" and job["project_id"] == pid and job["shot_id"] is None
    assert job["params"] == {"upload_id": up["id"], "name": "Me", "seed": 3}
    rosa = store.add_cast(pid, b"img", "rosa.png", "Rosa")
    r = c.post(f"/api/projects/{pid}/cast/sheet", json={"cast_id": rosa["id"]})
    params = r.json()["params"]
    assert r.status_code == 200 and isinstance(params.pop("seed"), int)  # Run 8: the seed it uses rides on the job
    assert params == {"cast_id": rosa["id"], "name": "Rosa"}  # named for the queue
    r = c.post(f"/api/projects/{pid}/cast/sheet", json={"describe": "a knight in armour", "name": "Kay"})
    assert r.status_code == 200 and r.json()["params"]["describe"] == "a knight in armour"


def test_sheet_endpoint_errors(client):
    c, store = client
    pid = store.create_project("P")["id"]
    place = store.add_cast(pid, b"img", "square.png", "Square", "place")
    finn = store.add_cast(pid, b"img", "finn.png", "Finn")
    url = f"/api/projects/{pid}/cast/sheet"
    assert c.post("/api/projects/prj_nope/cast/sheet", json={"describe": "x", "name": "X"}).status_code == 404
    cases = [
        ({}, 400, "Add a picture"),
        ({"upload_id": "up_0123456789", "describe": "a man", "name": "X"}, 400, "not both"),
        ({"describe": "a man"}, 400, "name"),
        ({"describe": "a man", "name": "FINN"}, 400, "already a cast entry called FINN"),
        ({"upload_id": "../../secret", "name": "X"}, 400, "upload id"),
        ({"upload_id": "up_0123456789", "name": "X"}, 404, "gone"),
        ({"image_id": "img_missing", "name": "X"}, 404, "No such image"),
        ({"cast_id": "cast_missing"}, 404, "No such cast entry"),
        ({"cast_id": place["id"]}, 400, "for people"),
        ({"cast_id": finn["id"], "seed": "soon"}, 400, "seed"),
    ]
    for body, status, words in cases:
        r = c.post(url, json=body)
        assert r.status_code == status and words in r.json()["detail"], (body, r.status_code, r.text)
    r = c.post(url, content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "JSON" in r.json()["detail"]


def test_cast_payloads_carry_the_sheet(client, tmp_path):
    c, store = client
    pid = store.create_project("P")["id"]
    views, sheet, source = _built(tmp_path)
    made = store.save_sheet(pid, views, sheet, name="Finn", source=source, meta={"editor": "Qwen-Image 2.1"})
    member = next(x for x in c.get(f"/api/projects/{pid}").json()["cast"] if x["id"] == made["id"])
    assert member["sheet"]["url"] == made["sheet"]["url"] and member["uses"] == 0
    assert set(member["sheet"]["views"]) == {"front", "back", "face"}
    for u in [member["sheet"]["url"], member["sheet"]["thumb_url"], *member["sheet"]["views"].values()]:
        assert c.get(u).status_code == 200  # served by the media route
    everyone = c.get("/api/cast").json()
    assert next(x for x in everyone if x["id"] == made["id"])["sheet"]["views"] == member["sheet"]["views"]
