"""The licence of every model a take can name, for its Details.

Keys are the names the tasks write into a take's details["models"] (graphs.edit_model() included) and the image models'
names from images.MODELS. Each maps to {"licence", "commercial"}: commercial is True when the licence allows paid work,
False when it doesn't, and None when it depends on its terms (revenue caps, use policies) or there is no model licence
(a ComfyUI built-in). Sources: the README's licence table and images.MODELS.
"""

from __future__ import annotations

import re

LTX = "LTX-2.x Community License"
OPEN = ("Apache-2.0", "MIT")

TASK_MODELS: dict[str, dict] = {
    "LTX-2.5": {"licence": LTX, "commercial": None},
    "LTX-2.5 + Union Control IC-LoRA": {"licence": LTX, "commercial": None},
    "LTX-2.3 + Relight IC-LoRA": {"licence": LTX, "commercial": None},
    "LTX-2.5 Layout-To-Render": {"licence": LTX, "commercial": None},
    "LTX-2.5 SDR-to-HDR": {"licence": LTX, "commercial": None},
    "SAM 3.1": {"licence": "SAM License", "commercial": None},
    "Qwen-Image 2.1": {"licence": "Qwen Research License", "commercial": False},
    "Qwen-Image-Edit 2511": {"licence": "Apache-2.0", "commercial": True},
    "SCAIL-2 (Wan 2.1)": {"licence": "MIT", "commercial": True},
    "SeedVR2 3B": {"licence": "Apache-2.0", "commercial": True},
    "Depth Anything 3": {"licence": "Apache-2.0", "commercial": True},
    "Z-Image Turbo + Fun ControlNet Union 2.1": {"licence": "Apache-2.0", "commercial": True},
    "NVIDIA RTX Video Super Resolution": {"licence": "NVIDIA proprietary", "commercial": None},
    "Canny": {"licence": None, "commercial": None},  # ComfyUI's built-in edge filter, not a model
}


def _image_models() -> dict[str, dict]:
    from pickups import images

    out = {}
    for m in images.MODELS.values():
        low = m.licence.lower()
        name = re.sub(r"\s*\(non-commercial\)$", "", m.licence).replace("Apache 2.0", "Apache-2.0")
        commercial = False if ("non-commercial" in low or "research" in low) else True if name in OPEN else None
        out[m.name] = {"licence": name, "commercial": commercial}
    return out


LICENCES: dict[str, dict] = _image_models() | TASK_MODELS

# What "it depends on its terms" means, in a line, for the licences the tasks use (receipts print it). Checked against
# the licence texts (LTX-2 and LTX-2.x Community License; Meta's SAM License; the Krea 2 Community License, section 2.3).
TERMS: dict[str, str] = {
    LTX: "Allowed for a business under $10M annual revenue (with affiliates); above that, a paid license from Lightricks",
    "SAM License": "Allowed, within Meta's trade-control and end-use limits",
    "Krea 2 Community License": "Allowed only for a business under $1M yearly revenue (with affiliates), with content "
                                "filtering; above that, an enterprise license from Krea",
}


def terms(licence_name: str | None) -> str | None:
    """The commercial-use condition of a licence that allows paid work only on its terms, or None if Pickups has none."""
    return TERMS.get(licence_name or "")


def licence(name: str) -> dict:
    """{"name", "licence", "commercial"} for one model; older takes' "SeedVR2 3B (4K)" finds "SeedVR2 3B"."""
    hit = LICENCES.get(name) or LICENCES.get(re.sub(r"\s*\([^)]*\)$", "", name)) or {}
    return {"name": name, "licence": hit.get("licence"), "commercial": hit.get("commercial")}


def licences_for(models: list | None) -> list[dict]:
    """The licence of each model a take ran, in its order (unknown names: licence None)."""
    out: list[dict] = []
    for m in models or []:
        if isinstance(m, str) and m and all(x["name"] != m for x in out):
            out.append(licence(m))
    return out
