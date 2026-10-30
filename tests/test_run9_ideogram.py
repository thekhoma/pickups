"""Run 9: Ideogram 4's refusal card is a failed job (nothing saved, a new seed once), and plain words become its JSON
caption first (the local "ideogram director"). All mocked: no ComfyUI, no LLM, no network."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pickups import director, images
from pickups.tasks import image as task

# --- the refusal card detector ----------------------------------------------------------------------------------------


def _card(path, w=1024, h=576):
    """A stand-in for the grey card: flat grey ~111 with faint blotches and one line of centred text."""
    rng = np.random.default_rng(1)
    a = np.full((h, w), 111.0) + rng.normal(0, 3, (h, w))
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    d.text((w // 2 - 120, h // 2), "Image blocked by safety filter", fill=(235, 235, 235))
    im.save(path)
    return path


def _photo(path, w=1024, h=576):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)
    Image.fromarray(a).resize((w, h), Image.NEAREST).save(path)
    return path


def _checker(path, w=1024, h=576):
    """Run G's poor caption: red text on a checkerboard (mean ~178, std ~14.6)."""
    a = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    a[:] = np.where(((yy // 32) + (xx // 32)) % 2, 192, 164)
    Image.fromarray(a.astype(np.uint8)).convert("RGB").save(path)
    return path


def test_the_refusal_card_is_found_and_real_pictures_are_not(tmp_path):
    assert images.refusal_card(_card(tmp_path / "card.png"))
    assert not images.refusal_card(_photo(tmp_path / "photo.png"))
    assert not images.refusal_card(_checker(tmp_path / "checker.png"))
    # Flat grey that isn't mid grey (a foggy sky at 160) is no card either.
    Image.new("RGB", (512, 512), (160, 160, 160)).save(tmp_path / "fog.png")
    assert not images.refusal_card(tmp_path / "fog.png")
    assert images.REFUSES == {"ideogram4"}


# --- the image task: refused once → a new seed; refused twice → a failed job, nothing kept -------------------------------


class _Out:
    def __init__(self, src):
        self.src = src
        self.name = "a.png"

    def to_file(self, dst):
        dst.write_bytes(self.src.read_bytes())


def _ctx(tmp_path, model, prompt, renders, seed=7, count=1):
    """renders: the picture each graph run returns, in order. Graphs (and the text they were sent) are recorded."""
    sent = []

    class Ctx:
        params = {"model": model, "prompt": prompt, "seed": seed, "seed_locked": True, "count": count, "aspect": "16:9"}
        project_id = None
        vdir = tmp_path / "work"
        store = SimpleNamespace(live_refs=lambda pid, refs: [], cast=lambda pid: [])
        conn = SimpleNamespace(model_files=lambda: set(), comfy_url="http://127.0.0.1:9")
        client = SimpleNamespace(assets=SimpleNamespace(from_file=lambda p: p))
        stages = task.stages_for(params)
        entered: list[str] = []

        def stage(self, label):
            assert label in [s.label for s in self.stages]
            self.entered.append(label)

        def progress(self, f):
            pass

        def run(self, g):
            sent.append(g)
            src = renders[min(len(sent) - 1, len(renders) - 1)]
            return SimpleNamespace(get_outputs=lambda node: [_Out(src)])

    Ctx.vdir.mkdir()
    return Ctx(), sent


def _prompt_of(g):
    return g["PU_Prompt"]["inputs"]["text"]


def test_a_refused_picture_is_made_again_once_with_a_new_seed(tmp_path, monkeypatch):
    card, photo = _card(tmp_path / "card.png"), _photo(tmp_path / "photo.png")
    monkeypatch.setattr(director, "ideogram_caption", lambda runner, url, text, w, h, s, cfg=None: json.dumps({"high_level_description": f"{text} #{s}"}))
    ctx, sent = _ctx(tmp_path, "ideogram4", "a bakery storefront", [card, photo])
    out = task.run(ctx)["images"]
    assert len(sent) == 2 and len(out) == 1
    assert out[0]["seed"] != 7  # never the refused seed again (ComfyUI would serve the cached card)
    assert out[0]["prompt"] == "a bakery storefront"  # the user's words stay the image's prompt
    assert json.loads(out[0]["prompt_sent"])["high_level_description"].endswith(f"#{out[0]['seed']}")  # its own caption
    assert _prompt_of(sent[0]) != _prompt_of(sent[1])
    assert [p.name for p in ctx.vdir.iterdir()] == [out[0]["path"].rsplit("/", 1)[-1]]  # the card is gone
    assert ctx.entered == ["Loading the model", "Writing the caption", "Making your images"]


def test_refused_twice_is_a_failed_job_marked_blocked(tmp_path, monkeypatch):
    card = _card(tmp_path / "card.png")
    monkeypatch.setattr(director, "ideogram_caption", lambda *a, **k: '{"high_level_description":"x"}')
    ctx, sent = _ctx(tmp_path, "ideogram4", "a seaside travel poster", [card])
    with pytest.raises(task.Refused) as e:
        task.run(ctx)
    assert len(sent) == 2 and e.value.job_fields == {"blocked": True}
    assert str(e.value) == images.REFUSED
    assert not list(ctx.vdir.iterdir())  # nothing kept


def test_other_models_are_never_checked_for_the_card(tmp_path):
    card = _card(tmp_path / "card.png")
    ctx, sent = _ctx(tmp_path, "z-image-turbo", "a grey wall", [card])
    assert len(task.run(ctx)["images"]) == 1 and len(sent) == 1
    assert task.stages_for({"model": "z-image-turbo"}) is task.STAGES


def test_a_failed_project_job_carries_blocked(tmp_path):
    from pickups.app.jobs import JobManager, Stage, TaskSpec
    from pickups.app.store import Store

    s = Store(tmp_path / "library")
    pid = s.create_project("P")["id"]

    def refuse(ctx):
        raise task.Refused(images.REFUSED)

    conn = SimpleNamespace(free_memory=lambda: None, client=lambda: None, comfy_url="http://127.0.0.1:9", system_stats=lambda: None,
                           cancel_prompt=lambda pid: None)
    m = JobManager(s, conn, {"image": TaskSpec([Stage("Making your images", 1)], refuse, scope="project")})
    job = m.submit(None, "image", {"prompt": "x", "model": "ideogram4"}, project_id=pid)
    import time

    for _ in range(100):
        if job["status"] in ("failed", "done"):
            break
        time.sleep(0.05)
    assert job["status"] == "failed" and job["blocked"] is True and job["error"] == images.REFUSED
    assert s.images(pid) == []  # nothing saved


# --- the ideogram director ----------------------------------------------------------------------------------------------

TEMPLATE = "[META]\nthinking_mode: disabled\n\n[SYSTEM]\nYou write captions.\n\n[USER]\nTARGET IMAGE ASPECT RATIO: {{width}}:{{height}} (width:height).\nUser idea: {{original_prompt}}"
IDEOGRAM_JSON = {"nodes": [], "definitions": {"subgraphs": [{"nodes": [
    {"id": 113, "type": "PrimitiveStringMultiline", "widgets_values": ["not this one"]},
    {"id": 114, "type": "PrimitiveStringMultiline", "widgets_values": [TEMPLATE]},
]}]}}
QWEN_JSON = {"nodes": [
    {"id": 1, "type": "CLIPLoader", "widgets_values": ["qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors", "stable_diffusion", "default"]},
    {"id": 2, "type": "PrimitiveStringMultiline", "widgets_values": ["Qwen's own system prompt"]},
    {"id": 3, "type": "TextGenerate", "inputs": [{"name": "clip", "link": 10}, {"name": "system_prompt", "link": 11}],
     "widgets_values": ["", 2048, "on", 0.7, 20, 0.95, 0.0, 1.0, 1, None, False, True]},
], "links": [[10, 1, 0, 3, 0, "CLIP"], [11, 2, 0, 3, 1, "STRING"]]}

# R&D's run H, trimmed: invented signs, a duplicate, bbox_2d and text on an obj, a bbox out of range, lower-case colours.
RUN_H = {
    "aspect_ratio": "16:9",
    "high_level_description": "A vintage travel poster of a sunny seaside town with bold lettering reading 'FABLE WALK'.",
    "style_description": {"color_palette": ["#1b2a49", "f2e3c6", "not a colour"], "aesthetics": "flat screen print", "mood": "dropped"},
    "compositional_deconstruction": {"background": "Cream paper.", "elements": [
        {"type": "text", "bbox": [78, 120, 930, 340], "text": "FABLE WALK", "desc": "Large arched headline."},
        {"type": "text", "bbox": [680, 360, 860, 430], "text": "EST. 1872", "desc": "Invented badge."},
        {"type": "text", "bbox": [130, 920, 220, 960], "text": "Fable  Walk", "desc": "The same words again."},
        {"type": "obj", "bbox": [280, 450, 420, 1200], "text": "FABLE WALK", "desc": "A lighthouse.", "bbox_2d": [1, 2, 3, 4]},
        {"type": "obj", "bbox": [500, 500, 400, 600], "desc": "A box upside down."},
        {"type": "shape", "desc": "Not a type."},
    ]},
    "extra": "dropped",
}


def _fake_get(url, timeout=20.0):
    if url.endswith("/templates/image_ideogram4_t2i.json"):
        return IDEOGRAM_JSON
    if url.endswith("/templates/image_qwen_image_2_1_t2i.json"):
        return QWEN_JSON
    raise AssertionError(f"unexpected GET {url}")


@pytest.fixture
def offline(monkeypatch):
    director._cache.clear()
    monkeypatch.setattr(director, "_get", _fake_get)
    yield
    director._cache.clear()


def test_sanitize_keeps_the_schema_and_only_the_words_asked_for():
    out = director.sanitize_caption(RUN_H, 'A vintage travel poster of a seaside town, bold lettering that says "FABLE WALK", flat colours')
    assert "," in out and ": " not in out.split('"high_level_description"')[0]  # minified
    c = json.loads(out)
    assert list(c) == ["aspect_ratio", "high_level_description", "style_description", "compositional_deconstruction"]
    assert c["style_description"] == {"aesthetics": "flat screen print", "color_palette": ["#1B2A49", "#F2E3C6"]}
    els = c["compositional_deconstruction"]["elements"]
    assert [e.get("text") for e in els if e["type"] == "text"] == ["FABLE WALK"]  # no invented sign, no duplicate
    lighthouse = next(e for e in els if e["type"] == "obj" and e["desc"] == "A lighthouse.")
    assert lighthouse == {"type": "obj", "bbox": [280, 450, 420, 1000], "desc": "A lighthouse."}  # clamped; no text, no bbox_2d
    assert next(e for e in els if e["desc"] == "A box upside down.") == {"type": "obj", "desc": "A box upside down."}
    assert all(list(e) == [k for k in director.ELEMENT_KEYS if k in e] for e in els)


def test_unquoted_words_count_when_nothing_is_quoted():
    c = json.loads(director.sanitize_caption(RUN_H, "a travel poster with the words FABLE WALK on it"))
    assert [e["text"] for e in c["compositional_deconstruction"]["elements"] if e["type"] == "text"] == ["FABLE WALK"]
    c = json.loads(director.sanitize_caption(RUN_H, "a quiet harbour at dawn"))
    assert not [e for e in c["compositional_deconstruction"]["elements"] if e["type"] == "text"]


def test_an_empty_caption_is_refused():
    with pytest.raises(ValueError):
        director.sanitize_caption({"aspect_ratio": "1:1", "compositional_deconstruction": {"elements": []}}, "x")


def test_the_users_own_json_goes_through_untouched():
    mine = '{ "high_level_description": "My own caption",  "compositional_deconstruction": {"elements": []} }'

    class NoRun:
        def run(self, g):
            raise AssertionError("the director must not run")

    assert director.ideogram_caption(NoRun(), "http://127.0.0.1:9", mine, 1024, 576, 3) == mine
    assert director.is_json_caption(mine) and not director.is_json_caption("{not json") and not director.is_json_caption("a cat")
    assert task.stages_for({"model": "ideogram4", "prompt": mine}) is task.STAGES
    assert task.stages_for({"model": "ideogram4", "prompt": "a cat"}) is task.CAPTION_STAGES


def test_the_director_fills_comfys_caption_prompt_and_asks_for_4096_tokens(offline, monkeypatch):
    graphs = []
    replies = iter(["Sure! Here it is: {broken", "<think>plan</think>```json\n" + json.dumps(RUN_H) + "\n```"])
    monkeypatch.setattr(director, "raw_text", lambda url, job: next(replies))

    class Runner:
        def run(self, g):
            graphs.append(g)
            return SimpleNamespace(id="p1")

    out = director.ideogram_caption(Runner(), "http://127.0.0.1:9", 'a poster that says "FABLE WALK"', 2048, 1150, 41)
    assert json.loads(out)["compositional_deconstruction"]["elements"][0]["text"] == "FABLE WALK"
    assert len(graphs) == 2  # the first answer didn't parse: once more
    gen = graphs[0]["pd_gen"]["inputs"]
    assert gen["max_length"] == 4096 and gen["thinking"] is False
    assert "system_prompt" not in gen and "pd_sys" not in graphs[0]  # the whole template is the prompt, as Comfy sends it
    assert "TARGET IMAGE ASPECT RATIO: 2048:1152" in gen["prompt"] and 'User idea: a poster that says "FABLE WALK"' in gen["prompt"]
    assert graphs[0]["pd_llm"]["inputs"]["clip_name"].startswith("qwen3.5_9b")  # Qwen-Image's own enhancer, local
    assert [g["pd_gen"]["inputs"]["sampling_mode.seed"] for g in graphs] == [41, 42]  # the image's seed, then the next
    assert not any(n["class_type"].startswith("Ideogram") and "Api" in n["class_type"] for g in graphs for n in g.values())


def test_the_director_fails_in_words_after_two_bad_answers(offline, monkeypatch):
    monkeypatch.setattr(director, "raw_text", lambda url, job: "no json here")
    runner = SimpleNamespace(run=lambda g: SimpleNamespace(id="p"))
    with pytest.raises(RuntimeError, match="Ideogram caption"):
        director.ideogram_caption(runner, "http://127.0.0.1:9", "a cat", 1024, 1024, 1)


def test_a_local_chat_server_gets_the_system_and_user_parts(offline, monkeypatch):
    calls = []

    def chat(cfg, model, system, content, temperature, top_p, seed=0, max_tokens=1200, english=True):
        calls.append((system, content, max_tokens, english))
        return json.dumps({"high_level_description": "A cat on a mat."})

    monkeypatch.setattr(director, "chat_openai", chat)
    out = director.ideogram_caption(None, "http://127.0.0.1:9", "a cat", 1000, 500, 1, {"backend": "openai", "url": "http://127.0.0.1:1234", "model": "m"})
    assert json.loads(out) == {"high_level_description": "A cat on a mat."}
    system, content, max_tokens, english = calls[0]
    assert system == "You write captions." and content.startswith("TARGET IMAGE ASPECT RATIO: 1008:512") and content.endswith("a cat")
    assert max_tokens == 4096 and english is False
