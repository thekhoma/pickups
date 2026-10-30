"""The models each workflow runs by default, for the home gallery's model chips and the task list.

One entry per workflow on the home screen (the gallery's tiles; "character" and "car" are both Recast). Each slot is one
job a model does in that workflow: `name` is exactly what the task writes into a take's details["models"] (so licences
and receipts find it), `short` is the chip label, `role` says what it does in a word or two. `options` lists what the
slot can run instead today (the first is the default); a later build can let the user pick from it.

Defaults are what a take runs with nothing changed: Recast on the LTX-2.5 engine, Weather on depth with nothing kept,
Render previz followed exactly, Finish at best quality with 4K and HDR, New image on images.DEFAULT. Some slots run only
when they're needed (Remove's plate and fill, Recast's still from a description); `when` says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pickups import features, graphs

LTX_UNION = "LTX-2.5 + Union Control IC-LoRA"
ZBUF_STILL = "Z-Image Turbo + Fun ControlNet Union 2.1"  # Previz to final's start frame from the previz depth


@dataclass(frozen=True)
class Slot:
    role: str
    name: str
    short: str
    options: tuple[str, ...] = ()
    when: str = ""


def _editor(role: str, safe: bool, when: str = "") -> Slot:
    """The frame editor (graphs.edit_model): Qwen-Image 2.1, or Qwen-Image-Edit 2511 under Setup's commercial choice."""
    name = graphs.edit_model(safe)
    return Slot(role, name, name, (name, graphs.edit_model(not safe)), when)


def _flux(role: str, when: str = "") -> Slot:
    return Slot(role, "FLUX 3 klein", "FLUX 3 klein", when=when)


def _image_default() -> Slot:
    from pickups import images

    m = images.MODELS[images.DEFAULT]
    return Slot("image", m.name, m.name, (m.name, *(x.name for x in images.MODELS.values() if x is not m)))


def _recast(safe: bool) -> list[Slot]:
    return [Slot("video", LTX_UNION, "LTX-2.5", (LTX_UNION, "SCAIL-2 (Wan 2.1)")),
            _editor("frame paint", safe),
            Slot("tracking", "SAM 3.1", "SAM 3.1"),
            Slot("depth", "Depth Anything 3", "Depth Anything 3"),
            _flux("still from text", "when you describe them")]


def workflows(commercial_safe: bool = False) -> dict[str, dict]:
    """{workflow id: {"task", "models": [Slot]}} in the gallery's order."""
    safe = commercial_safe
    car = {"car": {"task": "recast", "models": _recast(safe)}} if features.cars() else {}
    return {
        "character": {"task": "recast", "models": _recast(safe)},
        **car,
        "weather": {"task": "environment", "models": [
            Slot("video", LTX_UNION, "LTX-2.5"),
            _editor("first frame", safe),
            Slot("depth", "Depth Anything 3", "Depth Anything 3", ("Depth Anything 3", "Canny")),
        ]},
        "remove": {"task": "remove", "models": [
            Slot("tracking", "SAM 3.1", "SAM 3.1"),
            _editor("clean plate", safe, "when part is never seen"),
            Slot("fill", "LTX-2.5", "LTX-2.5", when="on moving shots"),
        ]},
        "previz": {"task": "previz", "models": [
            Slot("video", "LTX-2.5 Layout-To-Render", "LTX-2.5", ("LTX-2.5 Layout-To-Render", LTX_UNION)),
            Slot("start frame", ZBUF_STILL, "Z-Image Turbo", (ZBUF_STILL, "Qwen-Image 2.1")),
            _editor("look frame", safe, "without a start frame"),
        ]},
        "sheet": {"task": "sheet", "models": [
            _editor("views", safe),
            _flux("front from text", "when you describe them"),
        ]},
        "relight": {"task": "relight", "models": [Slot("video", "LTX-2.3 + Relight IC-LoRA", "LTX-2.3 Relight")]},
        "finish": {"task": "finish", "models": [
            Slot("4K upscale", "SeedVR2 3B", "SeedVR2", ("SeedVR2 3B", "NVIDIA RTX Video Super Resolution")),
            Slot("HDR", "LTX-2.5 SDR-to-HDR", "LTX-2.5 HDR"),
        ]},
        "image": {"task": "image", "models": [_image_default()]},
    }


def for_task(task: str, commercial_safe: bool = False) -> list[Slot]:
    """The default models of the first workflow that runs `task` ([] for one with none, like Reshoot)."""
    return next((w["models"] for w in workflows(commercial_safe).values() if w["task"] == task), [])


def as_json(slots: list[Slot]) -> list[dict]:
    return [{k: v for k, v in asdict(s).items() if v or k in ("role", "name", "short")} | {"options": list(s.options or (s.name,))}
            for s in slots]


def catalog(commercial_safe: bool = False) -> dict[str, dict]:
    """GET /api/models/defaults: {workflow id: {"task", "models": [{"role", "name", "short", "options", "when"?}]}}."""
    return {wid: {"task": w["task"], "models": as_json(w["models"])} for wid, w in workflows(commercial_safe).items()}
