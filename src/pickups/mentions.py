"""@-mentions of a project's cast in prompts, expanded for each model.

A cast entry is {"id", "name", "kind": person|place|prop|look, "line", "file"}. `line` is one noun phrase that
describes it ("a woman in her thirties with short black hair and a red wool coat").

- Image models (Qwen-Image 2.1) get the cast images as references; "@Maya" becomes "Maya (the person in <image2>)",
  numbered in order of first mention after the `first_image` slot (<image1> is always the shot's own frame). Qwen-Image
  2.1 requires `<imageN>` tags whenever there are two or more images (Comfy's official edit template and enhancer), so
  `tag_images` also converts a user's plain "image 2" into "<image2>".
- Text-only video models (LTX-2.5) get the description instead: the first "@Maya" becomes "Maya, a woman in her
  thirties with ...", later ones just "Maya"; places and looks become their line.
"""

from __future__ import annotations

import re

NOUN = {"person": "person", "place": "place", "prop": "object", "look": "look"}
_END = ",.;:!?)"


def _match_at(text: str, i: int, cast: list[dict]) -> dict | None:
    """The cast entry whose name follows the '@' at text[i] (longest name wins, case-insensitive, whole words)."""
    best = None
    for c in cast:
        name = c["name"]
        j = i + 1 + len(name)
        if text[i + 1:j].lower() == name.lower() and (j == len(text) or not (text[j].isalnum() or text[j] == "_")):
            if best is None or len(name) > len(best["name"]):
                best = c
    return best


def _scan(text: str, cast: list[dict]):
    """Yield (start, end, entry) for each @-mention of a cast entry, left to right."""
    i = 0
    while (i := text.find("@", i)) != -1:
        c = _match_at(text, i, cast)
        if c:
            end = i + 1 + len(c["name"])
            yield i, end, c
            i = end
        else:
            i += 1


def mentioned(text: str, cast: list[dict]) -> list[dict]:
    """Cast entries mentioned in `text`, in order of first mention, without repeats."""
    seen: dict[str, dict] = {}
    for _s, _e, c in _scan(text or "", cast):
        seen.setdefault(c["id"], c)
    return list(seen.values())


def for_images(text: str, cast: list[dict], first_image: int = 2, order: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Expand mentions for an image-editing model. Returns (text, entries in image order).

    `order` fixes the image numbering (e.g. the order a previous field assigned); unknown mentions are appended."""
    refs = list(order or [])
    ids = [c["id"] for c in refs]
    out, last = [], 0
    for s, e, c in _scan(text or "", cast):
        if c["id"] not in ids:
            refs.append(c)
            ids.append(c["id"])
        n = first_image + ids.index(c["id"])
        kind = c.get("kind", "person")
        if kind == "look":
            sub = f"the look of <image{n}>"
        elif kind == "place":
            sub = f"{c['name']} (the place in <image{n}>)"
        else:
            sub = f"{c['name']} (the {NOUN.get(kind, 'person')} in <image{n}>)"
        out += [text[last:s], sub]
        last = e
    out.append((text or "")[last:])
    return "".join(out), refs


def for_text(text: str, cast: list[dict]) -> str:
    """Expand mentions for a text-only model: people and props get their description on first mention."""
    out, last, done = [], 0, set()
    text = text or ""
    for s, e, c in _scan(text, cast):
        line = (c.get("line") or "").strip().rstrip(".")
        kind = c.get("kind", "person")
        if kind in ("place", "look"):
            sub = line or c["name"]
        elif c["id"] in done or not line:
            sub = c["name"]
        else:
            nxt = text[e:e + 1]
            sub = f"{c['name']}, {line}" + ("" if (not nxt or nxt in _END) else ",")
        done.add(c["id"])
        out += [text[last:s], sub]
        last = e
    out.append(text[last:])
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


_PLAIN = re.compile(r"(?<![<\w])(?:the\s+)?(?:image|picture)\s*#?\s*(\d+)(?!\w)", re.IGNORECASE)
_TAG = re.compile(r"<image(\d+)>")


def tag_images(text: str, n_images: int) -> str:
    """Qwen-Image 2.1 reference style: `<imageN>` tags with two or more images; "the image" with exactly one."""
    if n_images >= 2:
        return _PLAIN.sub(lambda m: f"<image{m.group(1)}>", text)
    text = _TAG.sub("the image", text)
    return _PLAIN.sub("the image", text)
