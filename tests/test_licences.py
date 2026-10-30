import ast
from pathlib import Path

from pickups import graphs, images, licences

TASKS = Path(licences.__file__).parent / "tasks"


def _strings(node: ast.AST) -> set[str]:
    """String literals a "models" expression can evaluate to (list items, either side of an if/else or a +)."""
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.List):
        return set().union(*map(_strings, node.elts)) if node.elts else set()
    if isinstance(node, ast.IfExp):
        return _strings(node.body) | _strings(node.orelse)
    if isinstance(node, ast.BinOp):
        return _strings(node.left) | _strings(node.right)
    return set()  # names and calls (graphs.edit_model) are added separately


def task_model_names() -> set[str]:
    """Every model name a task can write into details["models"]: the string literals in each "models" value."""
    names: set[str] = set()
    for f in TASKS.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value == "models":
                        names |= _strings(v)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "models":
                        names |= _strings(node.value)
    return names | {graphs.edit_model(True), graphs.edit_model(False)}


def test_scan_finds_the_tasks_models():
    names = task_model_names()
    # A refactor that hides the names from the scan must fail here, not pass silently.
    assert {"SAM 3.1", "LTX-2.5 Layout-To-Render", "SCAIL-2 (Wan 2.1)", "SeedVR2 3B", "Canny",
            "Qwen-Image 2.1", "Qwen-Image-Edit 2511"} <= names


def test_every_task_model_has_a_licence():
    for name in task_model_names():
        assert name in licences.LICENCES, f"{name} has no entry in licences.LICENCES"
        if name != "Canny":  # a ComfyUI built-in, not a model
            assert licences.LICENCES[name]["licence"], name


def test_known_licences():
    L = licences.LICENCES
    assert L["Qwen-Image 2.1"] == {"licence": "Qwen Research License", "commercial": False}
    assert L["Qwen-Image-Edit 2511"] == {"licence": "Apache-2.0", "commercial": True}
    assert L["LTX-2.5 + Union Control IC-LoRA"]["licence"] == "LTX-2.x Community License"
    assert L["SAM 3.1"]["licence"] == "SAM License"
    assert L["SCAIL-2 (Wan 2.1)"]["licence"] == "MIT"
    assert L["SeedVR2 3B"]["licence"] == L["Depth Anything 3"]["licence"] == "Apache-2.0"


def test_image_models_are_covered():
    for m in images.MODELS.values():
        assert licences.LICENCES[m.name]["licence"], m.name
    assert licences.LICENCES["Ideogram 4"]["commercial"] is False
    assert licences.LICENCES["Z-Image Turbo"] == {"licence": "Apache-2.0", "commercial": True}


def test_licences_for():
    out = licences.licences_for(["Qwen-Image 2.1", "LTX-2.5 Layout-To-Render", "Qwen-Image 2.1", "Canny", "Mystery 9"])
    assert [x["name"] for x in out] == ["Qwen-Image 2.1", "LTX-2.5 Layout-To-Render", "Canny", "Mystery 9"]
    assert out[0] == {"name": "Qwen-Image 2.1", "licence": "Qwen Research License", "commercial": False}
    assert out[2]["licence"] is None and out[3] == {"name": "Mystery 9", "licence": None, "commercial": None}
    assert licences.licences_for(None) == [] and licences.licences_for([]) == []
    # Older takes named models with a note in brackets.
    assert licences.licences_for(["SeedVR2 3B (4K)"])[0]["licence"] == "Apache-2.0"


def test_store_adds_licences(tmp_path, monkeypatch):
    import json

    import pickups.app.store as st

    monkeypatch.setattr(st, "ROOT", tmp_path / "library")
    store = st.Store(tmp_path / "library")
    d = store.shot_dir("shot_x")
    (d / "versions" / "v_1").mkdir(parents=True)
    (d / "versions" / "v_1" / "meta.json").write_text(json.dumps({
        "label": "Environment · Snow", "task": "environment",
        "details": {"models": ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3"]}}))
    v = store.version("shot_x", "v_1")
    assert v["label"] == "Weather · Snow"  # older takes on disk keep their label; it reads as the task is named now
    assert [x["licence"] for x in v["licences"]] == ["Qwen Research License", "LTX-2.x Community License", "Apache-2.0"]
    meta = json.loads((d / "versions" / "v_1" / "meta.json").read_text())
    assert "licences" not in meta and meta["label"] == "Environment · Snow"  # derived, never written back
