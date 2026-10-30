"""The one place positive prompts are cleaned (Previz to final: the start still and the LTX video prompt).

A positive never names the previz itself or what it is made of (grey, previz, mannequin, blockout, backdrop, studio, clay,
3D, "small vehicles"), and never says what isn't there ("no badges", "never black", "nobody crosses"): models read those
words as things to draw. Those words belong in the negative, which is never filtered.

clean_positive drops every sentence that names the previz (a grey on a figure, shape or surface is the layout's stand-in),
and every clause (the part of a sentence between commas, semicolons, "and", "but" or "with") that is a negation, after
rewriting the ones the director's table says as what is there ("no spoiler" -> "a smooth tail with a plain flat lid"); a grey
that is a real colour ("a grey flat cap") stays grey (Run 2: charcoal rendered near black). l2r_prompt is the short
Layout-to-Render video prompt: one or two sentences, then "Keep the same <subjects> and camera as the driver."
"""

from __future__ import annotations

import re

BANNED = ("grey", "gray", "previz", "pre-viz", "mannequin", "mannequins", "blockout", "block-out", "backdrop", "studio",
          "clay", "3D", "small vehicles")
NEGATION = r"\b(no|not|never|without|nobody|nothing|none|neither|nor)\b|n't\b"
# A grey on one of these is the layout's stand-in, never a real colour: the sentence goes.
STAND_IN = r"\b(figures?|shapes?|surfaces?|blocks?|layout|proxy|proxies|stand-ins?|desaturation)\b"
L2R_END = "Keep the same {subjects} and camera as the driver."
L2R_SENTENCES = 2
PERSON_WORDS = ("woman", "man", "girl", "boy")
VEHICLE_WORDS = ("car", "truck", "van", "bus", "taxi", "motorbike", "motorcycle")

_BANNED_RE = re.compile(r"(?i)(?<![\w-])(" + "|".join(re.escape(w) for w in BANNED if w not in ("grey", "gray")) + r")(?![\w-])")
_GREY_RE = re.compile(r"(?i)(?<![\w-])gr[ae]y(?![\w-])")
_NEG_RE = re.compile(r"(?i)" + NEGATION)
_STAND_RE = re.compile(r"(?i)" + STAND_IN)


# Run 2 (b): what isn't there, said as what is (the director's table), before the negation filter; the specific lines first.
POSITIVE_REWRITES: tuple[tuple[str, str], ...] = (
    (r"a smooth,? unbroken tail with a flat trunk lid and no spoiler", "a smooth rounded tail with a plain flat trunk lid"),
    (r"clean flanks with no (badges|emblems|logos)( or (badges|emblems|logos))*", "clean unbadged flanks and a bare smooth bonnet and boot lid"),
    (r"(with |and )?no (readable )?(badges|emblems|logos)( or (badges|emblems|logos))*", r"\1clean unbadged bodywork"),
    (r"(with |and )?no spoiler", r"\1a smooth tail with a plain flat lid"),
    (r"(with |and )?no roof rack", r"\1a smooth bare roof"),
    (r"(with |and )?no (hat|cap)", r"\1bare-headed"),
    (r"(with |and )?no glasses", r"\1a bare face"),
    (r"(with |and )?no (readable )?(text|letters) on (the )?signs", r"\1blank unmarked signs"),
)
# What the words said isn't there, for the negative ("no spoiler" -> spoiler); "no other people" is EXACTLY N's job.
NEGATED_NOUNS = r"(?i)\b(?:no|without)\s+(?:readable\s+|visible\s+)?([a-z][a-z -]{1,30}?)(?=\s*(?:[,.;:]|$|\band\b|\bor\b|\bwith\b))"


def banned_in(text: str) -> list[str]:
    """Every banned word or negation in a positive, as written (empty when it's clean). A grey is banned only on the
    layout's stand-ins (Run 2: a real grey stays grey)."""
    hits = [m.group(0) for m in _BANNED_RE.finditer(text or "")]
    hits += [m.group(0) for s in re.split(r"(?<=[.!?])\s+", text or "") if _STAND_RE.search(s) for m in _GREY_RE.finditer(s)]
    hits += [m.group(0) for m in _NEG_RE.finditer(text or "")]
    return hits


def positive_words(text: str) -> str:
    """The director's positive phrasings for what isn't there ("no spoiler" -> "a smooth tail with a plain flat lid")."""
    for rx, to in POSITIVE_REWRITES:
        text = re.sub(rf"(?i)\b{rx}\b", to, text)
    return text


def negated(text: str) -> list[str]:
    """The things the words say aren't there, each once, for the negative: "no spoiler and no badges or emblems" ->
    ["spoiler", "badges", "emblems"]. Never other people or cars (the count says that)."""
    out: list[str] = []
    for m in re.finditer(NEGATED_NOUNS, text or ""):
        tail = text[m.end():]
        words = [m.group(1), *re.findall(r"(?i)^\s+or\s+([a-z][a-z -]{1,30}?)(?=\s*(?:[,.;:]|$|\band\b))", tail)]
        for w in words:
            w = w.strip().lower()
            if w and not re.search(r"\b(other|one|people|person|cars?|vehicles?)\b", w) and w not in out:
                out.append(w)
    return out


def _sentence(s: str) -> str:
    """One sentence with its negations gone; "" when it names the previz or nothing is left of it."""
    if _BANNED_RE.search(s) or (_GREY_RE.search(s) and _STAND_RE.search(s)):
        return ""
    end = s[-1] if s and s[-1] in ".!?" else ""
    body = s[:-1] if end else s
    # Clauses: commas, semicolons, colons, and " and " / " but " (a list item: "tyres with real tread and no badges").
    parts = re.split(r"(\s*[,;:]\s*|\s+(?:and|but|with)\s+)", body)
    kept: list[str] = []
    sep = ""
    cut = False
    for i, part in enumerate(parts):
        if i % 2:
            sep = part
            continue
        if not part.strip():
            continue
        if not _NEG_RE.search(part):
            kept.append((sep if kept else "") + part)
        else:
            cut = True
        sep = ""
    out = "".join(kept).strip(" ,;:")
    out = re.sub(r"\s+(and|but|with)$", "", out)
    if not out or (cut and len(out.split()) < 2):
        return ""
    return out[:1].upper() + out[1:] + (end or ".")


def clean_positive(text: str) -> str:
    """The positive with every sentence naming the previz and every negation taken out (paragraphs kept); a negation
    the director's table can say as what is there is rewritten first, so the user's own constraint survives."""
    paras = []
    for para in re.split(r"\n\s*\n", positive_words(text or "")):
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(para.split())) if x]
        kept = [c for c in (_sentence(s) for s in sentences) if c]
        if kept:
            paras.append(" ".join(kept))
    return "\n\n".join(paras)


def keep_subjects(subjects: list[dict]) -> str:
    """"woman", "two people", "car", "man and car": what the driver must keep, from frame 0's subjects in view (a lone
    person by the previz's word, "person" only when it has none; motionPrompt.ts keepWords says it the same way)."""
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    seen = [s for s in subjects if isinstance(s, dict) and s.get("in_view", True)] or [s for s in subjects if isinstance(s, dict)]
    people = [s for s in seen if s.get("kind") == "person"]
    cars = [s for s in seen if s.get("kind") == "vehicle"]
    word = str(people[0].get("word") or "").lower() if people else ""
    car = str(cars[0].get("word") or "").lower() if cars else ""
    car = car if car in VEHICLE_WORDS else "car"
    parts = []
    if people:
        parts.append((word if word in PERSON_WORDS else "person") if len(people) == 1 else f"{words.get(len(people), 'many')} people")
    if cars:
        parts.append(car if len(cars) == 1 else f"{words.get(len(cars), 'many')} {car}s")
    return " and ".join(parts) or "shot"


# The Look box's fixed look line (web/src/previz/lookPrompt.ts LOOKS: the film sentence, then a light sentence ending
# "fine grain."). The start frame's template has its own, and it never steers the video line.
LOOK_LINE_RX = r"A real photograph in the look of a large-format 65mm commercial film frame[^.]*\.(?:\s+[^.]*?fine grain\.)?"
# The Look's "who the grey figure is" sentence (lookPrompt.ts), and the older one that doubled the verb.
FIGURE_WHO_RX = r"(?i)\bThe gr[ae]y (?:figures? (?:is|are))\s+([^.!?]*?)\s+(?:who|that)\s+([^.!?]*[.!?]?)"
FIGURE_RX = r"(?i)\bThe gr[ae]y (?:figures? (?:is|are))\s+([^.!?]*[.!?]?)"
LOOK_MAX = 3  # the Look's sentences the video line takes, at most
_LINK = {"is", "are", "was", "were", "wears", "wear", "wearing", "has", "have", "the", "and", "with", "who", "that",
         "its", "his", "her", "their", "into", "onto", "from"}


def look_words(look: str) -> str:
    """The Look box as the models read it: no fixed look line, and "The grey figure is a man in a long coat who stands
    on a pier" said as what is there ("A man in a long coat stands on a pier"), so the cleaner never drops the sentence
    for its grey."""
    out = re.sub(LOOK_LINE_RX, " ", look or "")
    out = re.sub(FIGURE_WHO_RX, lambda m: f"{m.group(1)[:1].upper()}{m.group(1)[1:]} {m.group(2)}", out)
    out = re.sub(FIGURE_RX, lambda m: f"{m.group(1)[:1].upper()}{m.group(1)[1:]}", out)
    return " ".join(out.split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z-]+", s.lower()) if len(w) > 2 and w not in _LINK}


LEAD_RX = (r"(?i)^(?:a|an|the)\s+(?:[\w-]+\s+){0,2}?(man|woman|girl|boy|person|couple|car|truck|van)\b[^.!?]*?\b"
           r"(stand|sit|walk|run|wait|lean|drive|ride|dance)s?\b")


def _lead(s: str) -> tuple[str, str] | None:
    """(who, what they do) a sentence starts with: "A man in a long coat stands on a pier" -> ("man", "stand")."""
    m = re.match(LEAD_RX, s or "")
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def l2r_prompt(text: str, subjects: list[dict], look: str = "") -> str:
    """The Layout-to-Render video prompt: the first one or two clean sentences (frame 1 and the one action), then the
    Keep line. A Keep line the user wrote is kept as theirs. look: the Look box; its sentences the video line doesn't
    already say (the time, the wardrobe, the place) go after frame 1 and before the move, so LTX reads what the start
    frame shows. Run 11: the sentence that says the most of what's missing goes first ("A man in a long coat stands on a
    pier at dusk" over "It is dusk." and "The man wears a long coat."), and one that says all of frame 1 and more takes
    its place, so no fact is said twice."""
    clean = clean_positive(text).replace("\n\n", " ")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", clean) if s]
    keep = next((s for s in sentences if re.match(r"(?i)keep the same\b.*\bas the driver\b", s)), None)
    body = [s for s in sentences if s is not keep][:L2R_SENTENCES]
    said = _content(" ".join(body))
    order = [s for s in re.split(r"(?<=[.!?])\s+", clean_positive(look_words(look)).replace("\n\n", " ")) if _content(s)]
    pool, extra = list(order), []
    while pool and len(extra) < LOOK_MAX:
        best = max(pool, key=lambda s: len(_content(s) - said))  # the first of equals: the Look's own order
        if not _content(best) - said:
            break
        extra.append(best)
        said |= _content(best)
        pool.remove(best)
    extra.sort(key=order.index)
    covered = set().union(*map(_content, extra)) if extra else set()
    first = body[0] if body and not re.search(r"(?i)\bcamera\b", body[0]) else ""
    if first and extra and (_content(first) <= covered or (_lead(first) and any(_lead(s) == _lead(first) for s in extra))):
        # Frame 1's sentence is all in the Look's, or the Look says the same person doing the same thing (where the
        # start still shows them: "on a pier", not "on the shore"): said once, in the Look's words.
        body = extra + body[1:]
    else:
        body = body[:1] + extra + body[1:]
    return " ".join(body + [keep or L2R_END.format(subjects=keep_subjects(subjects))])
