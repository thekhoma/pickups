"""Build character sheet: front, back and close-up of one person, kept with the project's cast (pickups/sheets.py).

A project task (no shot): the job queue saves the result with the cast member (Store.save_sheet), a new one named in
the request or the one it upgrades. The source is an upload, a project image, a cast member or a description (FLUX 3
klein makes the front view with Recast's studio still template).
"""

from __future__ import annotations


from pickups import seeds, sheets
from pickups.tasks.recast import STILL

STAGES = sheets.STAGES


def _whole(text: str, n: int) -> str:
    """The words, never cut mid-sentence: a long one keeps its whole clauses (or all of it, with no clause to end on)."""
    if len(text) <= n:
        return text
    at = max(text.rfind(". ", 0, n), text.rfind(", ", 0, n), text.rfind("; ", 0, n))
    return text[:at] if at > 0 else text


def run(ctx) -> dict:
    p = ctx.params
    seed = seeds.pick(p.get("seed"))
    store, pid = ctx.store, ctx.project_id
    picture = still_prompt = None
    if p.get("upload_id"):
        picture, made_from = store.upload_path(p["upload_id"]), "upload"
    elif p.get("image_id"):
        picture, made_from = store.image_path(pid, p["image_id"]), "image"
    elif p.get("cast_id"):
        # A redo starts again from the picture the sheet was built from; a first sheet from the member's own picture.
        picture, made_from = store.sheet_source(pid, p["cast_id"]), "cast"
    elif (p.get("describe") or "").strip():
        still_prompt, made_from = STILL.format(who=p["describe"].strip().rstrip(".")), "words"
    else:
        raise ValueError("Add a picture of them, or describe them.")
    made = sheets.make(ctx, ctx.vdir, picture=picture, still_prompt=still_prompt, seed=seed, commercial_safe=ctx.commercial_safe())
    return {"sheet": {
        "views": made.views, "sheet": made.sheet, "cast_id": p.get("cast_id"), "name": p.get("name"),
        # A cast member keeps its own picture as the source; anything else is copied in with the sheet.
        "source": None if p.get("cast_id") else made.source,
        "line": _whole((p.get("describe") or "").strip(), 200),
        "meta": {"editor": made.editor, "models": made.models, "seed": made.seed, "seed_locked": bool(p.get("seed_locked")), "made_from": made_from, "made_front": made.made_front},
    }}
