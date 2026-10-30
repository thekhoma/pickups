"""Create or edit images: pick a model, an aspect ratio and a resolution, write a prompt, generate.

Images belong to a project (they show up in its Images tab and can become cast, look frames or first frames).
References (cast entries, earlier images, uploads) switch a model that can edit into edit mode; @-mentions of the
project's cast become references for edit models and descriptions for generate-only ones.
"""

from __future__ import annotations

import re

from pickups import mentions, seeds
from pickups.app.jobs import Stage

STAGES = [Stage("Loading the model", 15), Stage("Making your images", 30)]
# Ideogram 4 reads JSON captions: plain words are written up as one first (director.ideogram_caption).
CAPTION_STAGES = [Stage("Loading the model", 15), Stage("Writing the caption", 25), Stage("Making your images", 30)]
MAX_COUNT = 4


class Refused(RuntimeError):
    """The model drew its refusal card instead of an image: the job fails (nothing saved) and says it was refused, so the
    page doesn't keep the seed that was refused."""

    job_fields = {"blocked": True}


def stages_for(params: dict) -> list[Stage]:
    from pickups import director

    p = params or {}
    return CAPTION_STAGES if p.get("model") == "ideogram4" and not director.is_json_caption(str(p.get("prompt") or "")) else STAGES


def run(ctx) -> dict:
    from pickups import images  # the model registry (imported here so the other tasks never depend on it)

    p = ctx.params
    model = images.MODELS.get(p.get("model") or images.DEFAULT)
    if model is None:
        raise ValueError(f"Unknown image model '{p.get('model')}'.")
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Write a prompt first.")
    count = max(1, min(MAX_COUNT, int(p.get("count", 1))))
    seed = seeds.pick(p.get("seed"))
    w, h = images.size_for(p.get("aspect", "16:9"), p.get("resolution", "1K"), model)

    refs = ctx.store.live_refs(ctx.project_id, p.get("refs", []))
    cast = ctx.store.cast(ctx.project_id) if ctx.project_id else []
    used: list[dict] = []
    can_ref = "edit" in model.modes or "style" in model.modes
    if can_ref and (refs or mentions.mentioned(prompt, cast)):
        # The canvas (first reference) is <image1>; mentioned cast follow it.
        text, used = mentions.for_images(prompt, cast, first_image=1 + len(refs))
        refs += [ctx.store.cast_path(ctx.project_id, c["id"]) for c in used]
        refs = refs[: model.max_refs]
        text = mentions.tag_images(text, len(refs)) if refs else mentions.for_text(prompt, cast)
        if model.id.startswith("flux3") and refs:
            # FLUX 3 klein names its references <ref_image_0>, <ref_image_1>, ... (image 1 is ref_image_0).
            text = re.sub(r"<image(\d+)>", lambda m: f"<ref_image_{int(m.group(1)) - 1}>", text)
    else:
        text = mentions.for_text(prompt, cast)
        refs = []
    mode = images.mode_for(model, refs)

    ctx.stage("Loading the model")
    try:
        installed = ctx.conn.model_files()  # model files may sit in subfolders on the user's ComfyUI
    except Exception:  # noqa: BLE001
        installed = set()
    if installed and (gone := images.missing(model.id, installed)):
        raise RuntimeError(f"{model.name} needs files ComfyUI doesn't have yet: {', '.join(gone)}. See Setup.")
    assets = [ctx.client.assets.from_file(r) for r in refs]
    refuses = model.id in images.REFUSES
    words = text

    def caption(s: int) -> str:
        """What the model reads for this seed: Ideogram 4's JSON caption (the LLM's seed is the image's, so a locked
        seed writes the same caption), else the words as they are."""
        if model.id != "ideogram4":
            return words
        from pickups import director
        from pickups.app.comfy import Canceled, load_settings, plain_error

        try:
            return director.ideogram_caption(ctx, ctx.conn.comfy_url, words, w, h, s, load_settings().get("director") or {})
        except Canceled:
            raise
        except Exception as e:  # noqa: BLE001 - said in words: plain text would only bring back the refusal card
            if "Ideogram caption" in str(e):
                raise
            raise RuntimeError(f"Ideogram 4 reads a JSON caption, and the prompt director couldn't write one here ({plain_error(e)[0]}). "
                               "Write the caption as JSON, or check the prompt enhancer in Setup.") from e

    if model.id == "ideogram4" and len(ctx.stages) == len(CAPTION_STAGES):
        ctx.stage("Writing the caption")
    text = caption(seed)
    out = []
    ctx.stage("Making your images")
    for i in range(count):
        s = seed + i
        for attempt in range(2):
            graph = images.build(model.id, text, w, h, s, refs=assets, prefix="pickups/image")
            job = ctx.run(images.localize(graph, installed) if installed else graph)
            made = []
            for o in job.get_outputs("PU_Out"):
                dst = ctx.vdir / f"{i:02d}_{attempt}_{o.name}"
                o.to_file(dst)
                made.append(dst)
            if not (refuses and any(images.refusal_card(f) for f in made)):
                break
            for f in made:  # the refusal card is never kept
                f.unlink(missing_ok=True)
            if attempt:
                raise Refused(images.REFUSED)
            # The same graph and seed would only bring back ComfyUI's cached card: once more with a new seed (and its
            # own caption).
            s = seeds.pick(None)
            text = caption(s)
        for dst in made:
            out.append({"path": str(dst), "model": model.id, "model_name": model.name, "mode": mode, "prompt": prompt,
                        "prompt_sent": text, "seed": s, "seed_locked": bool(p.get("seed_locked")), "aspect": p.get("aspect", "16:9"),
                        "resolution": p.get("resolution", "1K"), "refs": p.get("refs", []) + [{"kind": "cast", "id": c["id"]} for c in used]})
        ctx.progress((i + 1) / count)
    return {"images": out}
