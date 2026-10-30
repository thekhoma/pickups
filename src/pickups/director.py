"""Prompt Director: turn a plain request into the prompt each model was trained to read, with the model makers' own
prompt enhancers, run locally on the same ComfyUI.

Each target uses the enhancer, system prompt and sampling settings from Comfy's official template for that model, read
live from the ComfyUI server (nothing is copied into this repo, and it follows ComfyUI updates):

- "qwen-image": Qwen-Image 2.1 text-to-image enhancer (Qwen 3.5 9B, `image_qwen_image_2_1_t2i`)
- "qwen-edit":  Qwen-Image 2.1 edit enhancer (Qwen 3.5 9B, sees the images; `image_qwen_image_2_1_image_edit`)
- "flux3":      FLUX 3 prompt upsampler (Qwen 3.5 9B, `image_flux3_klein_t2i`)
- "ltx":        LTX-2.5's own prompt director (Gemma, TextGenerateLTX2Prompt with LTX's built-in template; with a first
                frame it uses the image-to-video template `video_ltx2_5_i2v`, otherwise `video_ltx2_5_t2v`)
- Ideogram 4:   a JSON caption with the caption prompt in `image_ideogram4_t2i`, on the Qwen 3.5 9B enhancer, checked
                and cleaned (ideogram_caption; the image task runs it by itself for Ideogram 4)

The text comes back through ComfyUI's history for the job (the SDK returns files, and a prompt isn't one).

Backends (settings "director"): "comfy" (default, the official enhancer models above on the GPU machine) or "openai":
any local OpenAI-compatible server (LM Studio, Ollama, llama.cpp) given the same official system prompts, so the GPU
stays free for video. LTX's director template is built into ComfyUI rather than the template file, so the "openai"
backend uses LTX_SYSTEM below, written from Lightricks' prompting guide.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request

from pickups.graphs import _ref

TARGETS = {
    "qwen-image": {"template": "image_qwen_image_2_1_t2i", "label": "Qwen-Image 2.1"},
    "qwen-edit": {"template": "image_qwen_image_2_1_image_edit", "label": "Qwen-Image 2.1 edit", "images": True},
    "flux3": {"template": "image_flux3_klein_t2i", "label": "FLUX 3"},
    "ltx": {"template": "video_ltx2_5_t2v", "template_image": "video_ltx2_5_i2v", "label": "LTX-2.5"},
}
# Image models without an enhancer of their own read full scene captions best; the FLUX 3 upsampler writes those.
# Ideogram 4 reads only JSON captions: the image task writes one itself (ideogram_caption, below).
IMAGE_MODEL_TARGET = {"qwen": "qwen-image", "flux": "flux3", "ideogram": "ideogram"}
SAMPLING_KEYS = ["temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed", "presence_penalty"]

LTX_SYSTEM = (
    "You write prompts for LTX-2.5, an open video model. Rewrite the user's request as ONE flowing paragraph of 4 to 8 "
    "sentences (under 200 words) in the present tense, describing the finished shot as it plays: start with the main "
    "action; then specific movements and gestures in order; the people's and objects' appearance (age, clothing, "
    "colours, materials); the setting and background; the camera (shot size, angle, lens feel and one clear camera "
    "move); the light and colour; and the film look (e.g. 35mm, fine grain). Be concrete and visual: no lists, no "
    "headings, no camera-jargon soup, no negatives, no 'the scene opens', no sound or music. Keep every fact the user "
    "gave (names, numbers, colours, quoted words) and invent only what a cinematographer would need. If an image is "
    "attached it is the first frame: describe what continues from it, not a different picture. Reply with the prompt "
    "only.")
EDIT_FALLBACK = ("You rewrite a user's image-edit request as one precise English instruction for an image-editing "
                 "model: lead with the operation (Remove, Change, Replace, Convert), say exactly what changes and how it "
                 "should look, then one closing clause naming what stays unchanged. Refer to images as <image1>, "
                 "<image2> when there are several. Reply with the instruction only.")

# Pickups' prompts are English: the official Qwen enhancers are bilingual and may answer in Chinese.
ENGLISH = ("\n\nOUTPUT LANGUAGE: Write the final prompt in English only, whatever language the request uses, and keep any "
           "words the user quoted exactly as they are. Reply with the final prompt only.")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _get(url: str, timeout: float = 20.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - the user's own ComfyUI
        return json.loads(r.read())


def _graphs(template: dict):
    yield template
    yield from (template.get("definitions") or {}).get("subgraphs", [])


def recipe(comfy_url: str, template_name: str) -> dict:
    """The enhancer as the template wires it: node type, loader (model file, type), system prompt, settings."""
    with _lock:
        if template_name in _cache:
            return _cache[template_name]
    t = _get(f"{comfy_url}/templates/{template_name}.json")
    for g in _graphs(t):
        nodes = {n["id"]: n for n in g.get("nodes", [])}
        links = {}
        for link in g.get("links", []):
            if isinstance(link, list):
                links[link[0]] = link[1]
            elif isinstance(link, dict):
                links[link["id"]] = link["origin_id"]
        for n in nodes.values():
            if n.get("type") not in ("TextGenerate", "TextGenerateLTX2Prompt"):
                continue
            src = {i["name"]: nodes.get(links.get(i.get("link"))) for i in n.get("inputs", []) if i.get("link") is not None}
            loader = src.get("clip") or {}
            wv = n.get("widgets_values") or []
            # widgets: prompt, max_length, sampling_mode, temperature, top_k, top_p, min_p, repetition_penalty, seed,
            # presence_penalty, thinking, use_default_template[, mtp]
            settings = {"max_length": wv[1], "sampling_mode": wv[2], **dict(zip(SAMPLING_KEYS, wv[3:10])),
                        "thinking": bool(wv[10]) if len(wv) > 10 else False,
                        "use_default_template": bool(wv[11]) if len(wv) > 11 else True}
            if len(wv) > 12:
                settings["mtp"] = wv[12]
            system = (src.get("system_prompt") or {}).get("widgets_values", [None])[0]
            out = {"node": n["type"], "clip_name": loader["widgets_values"][0], "clip_type": loader["widgets_values"][1],
                   "system": system, "settings": settings, "image_prep": (src.get("image") or {}).get("type")}
            with _lock:
                _cache[template_name] = out
            return out
    raise RuntimeError(f"No prompt enhancer found in the {template_name} template")


def graph(rec: dict, text: str, images: list | None = None, seed: int = 0) -> dict:
    s = rec["settings"]
    inputs = {"clip": _ref("pd_llm"), "prompt": text, "max_length": s["max_length"], "sampling_mode": s["sampling_mode"],
              "thinking": s["thinking"], "use_default_template": s["use_default_template"]}
    if s["sampling_mode"] == "on":
        inputs |= {f"sampling_mode.{k}": s[k] for k in SAMPLING_KEYS if s.get(k) is not None and k != "presence_penalty"}
        inputs["sampling_mode.seed"] = seed
        if s.get("presence_penalty") is not None:
            inputs["sampling_mode.presence_penalty"] = s["presence_penalty"]
    if s.get("mtp"):
        inputs["mtp"] = s["mtp"]
    g = {"pd_llm": {"class_type": "CLIPLoader", "inputs": {"clip_name": rec["clip_name"], "type": rec["clip_type"], "device": "default"}}}
    if rec.get("system"):
        g["pd_sys"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": rec["system"] + ENGLISH}}
        inputs["system_prompt"] = _ref("pd_sys")
    if images:
        for i, im in enumerate(images):
            g[f"PU_Ref{i + 1}"] = {"class_type": "LoadImage", "inputs": {"image": im}}
        if rec["node"] == "TextGenerateLTX2Prompt":
            g["pd_img"] = {"class_type": "LTXVPreprocess", "inputs": {"image": _ref("PU_Ref1"), "img_compression": 18}}
        else:
            g["pd_img"] = {"class_type": "BatchImagesNode", "inputs": {f"images.image{i}": _ref(f"PU_Ref{i + 1}") for i in range(len(images))}}
        inputs["image"] = _ref("pd_img")
    g["pd_gen"] = {"class_type": rec["node"], "inputs": inputs}
    g["PU_Text"] = {"class_type": "PreviewAny", "inputs": {"source": _ref("pd_gen")}}
    return g


def clean(text: str) -> str:
    """Drop thinking blocks, an aspect-ratio line and wrapping quotes; keep one paragraph."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*(aspect ratio|ratio)\s*[:：]", ln, re.I)]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    return text.strip().strip('"').strip()


def enhance_openai(cfg: dict, comfy_url: str, target: str, text: str, files: list | None = None, seed: int = 0) -> str:
    """The same official system prompts, run on a local OpenAI-compatible server (cfg: url, model, vision_model)."""
    import base64
    import mimetypes

    spec = TARGETS[target]
    if target == "ltx":
        system, settings = LTX_SYSTEM, {"temperature": 0.7, "top_p": 0.95}
    else:
        try:
            rec = recipe(comfy_url, spec["template"])
            system, settings = rec["system"], rec["settings"]
        except Exception:  # noqa: BLE001 - ComfyUI offline: still direct the prompt
            system, settings = (EDIT_FALLBACK if target == "qwen-edit" else LTX_SYSTEM), {"temperature": 0.8, "top_p": 0.95}
    use_images = bool(files) and (spec.get("images") or target == "ltx")
    content: list | str = text
    if use_images:
        content = [{"type": "text", "text": text}]
        for f in files[:4]:
            mime = mimetypes.guess_type(str(f))[0] or "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}})
    model = cfg.get("vision_model") if use_images and cfg.get("vision_model") else cfg.get("model")
    return chat_openai(cfg, model, (system or "") + ENGLISH, content, float(settings.get("temperature", 0.8)),
                       float(settings.get("top_p", 0.95)), seed)


def chat_openai(cfg: dict, model: str, system: str, content: list | str, temperature: float, top_p: float, seed: int = 0,
                max_tokens: int = 1200, english: bool = True) -> str:
    """One chat completion on the local OpenAI-compatible server in cfg["url"], cleaned; asked once more when the answer
    isn't English (Pickups' prompts are English). english=False: any language is fine (a caption's quoted text)."""
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed, "reasoning_effort": "none"}
    url = cfg["url"].rstrip("/")
    url = url if url.endswith("/v1") else url + "/v1"
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(2):
        with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310 - the user's own local server
            out = json.loads(r.read())
        text = clean(out["choices"][0]["message"].get("content") or "")
        if text and (not english or not _CJK.search(text)):
            return text
        retry = ([{"type": "text", "text": "Answer in English."}] + content) if isinstance(content, list) else f"Answer in English.\n\n{content}"
        body["messages"][-1] = {"role": "user", "content": retry}
        body["seed"] = seed + 1 + attempt
        req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    raise RuntimeError("The director answered in another language twice; try again or use your own prompt.")


def raw_text(comfy_url: str, job) -> str:
    """The text a finished TextGenerate graph (PU_Text) wrote, as it wrote it. It comes back through ComfyUI's history
    for the job: the SDK returns files, and text isn't one."""
    hist = _get(f"{comfy_url}/history/{job.id}")
    out = (hist.get(job.id) or {}).get("outputs", {}).get("PU_Text", {}).get("text", [""])
    return out[0] if out else ""


def generated_text(comfy_url: str, job) -> str:
    """raw_text, cleaned to one paragraph."""
    return clean(raw_text(comfy_url, job))


def enhance(runner, comfy_url: str, target: str, text: str, images: list | None = None, seed: int = 0) -> str:
    """runner has .run(graph) (the app's QuickRunner or a TaskContext); images are Comfy SDK assets or file names."""
    spec = TARGETS[target]
    name = spec["template_image"] if images and spec.get("template_image") else spec["template"]
    rec = recipe(comfy_url, name)
    job = runner.run(graph(rec, text, images if (images and (spec.get("images") or spec.get("template_image"))) else None, seed))
    text = generated_text(comfy_url, job)
    if _CJK.search(text):
        raise RuntimeError("The director answered in another language; try again or use your own prompt.")
    return text


# --- Ideogram 4: a JSON caption from plain words ----------------------------------------------------------------------
# Ideogram 4 was trained only on structured JSON captions. Sent plain text, it often draws its built-in filter's grey
# "Image blocked by safety filter" card instead (Ideogram's prompting guide; R&D's renders, Oct 27 2026). So plain words
# become a full caption first: Comfy's own caption prompt (the "Ideogram4 Caption Prompt Template" in the
# image_ideogram4_t2i template, read live like the other targets) on the same local enhancer LLM as Qwen-Image's, then
# checked and cleaned. JSON the user wrote goes through untouched. The graph that renders is never changed.
IDEOGRAM_TEMPLATE = "image_ideogram4_t2i"
IDEOGRAM_LLM_FROM = "image_qwen_image_2_1_t2i"  # the Qwen 3.5 9B prompt enhancer's loader and sampling
IDEOGRAM_MAX_LENGTH = 4096  # 2048 cut a full caption short (R&D's run H)
CAPTION_KEYS = ("aspect_ratio", "high_level_description", "style_description", "compositional_deconstruction")
STYLE_KEYS = ("aesthetics", "lighting", "photo", "medium", "art_style", "color_palette")
ELEMENT_KEYS = ("type", "bbox", "text", "desc", "color_palette")
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|‘([^’]+)’|(?<![\w'])'([^']+)'(?![\w'])")


def caption_template(comfy_url: str) -> str:
    """Comfy's caption prompt for Ideogram 4: the multiline string in the template that takes {{original_prompt}}."""
    key = f"{IDEOGRAM_TEMPLATE}#caption"
    with _lock:
        if key in _cache:
            return _cache[key]["text"]
    t = _get(f"{comfy_url}/templates/{IDEOGRAM_TEMPLATE}.json")
    for g in _graphs(t):
        for n in g.get("nodes", []):
            wv = n.get("widgets_values") or []
            if n.get("type") == "PrimitiveStringMultiline" and wv and isinstance(wv[0], str) and "{{original_prompt}}" in wv[0]:
                with _lock:
                    _cache[key] = {"text": wv[0]}
                return wv[0]
    raise RuntimeError(f"No caption prompt found in the {IDEOGRAM_TEMPLATE} template")


def caption_request(template: str, text: str, width: int, height: int) -> str:
    """The template filled in: the size as the template rounds it (up to 16, at least 256), then the user's words."""
    w, h = (max((v + 15) // 16 * 16, 256) for v in (width, height))
    return template.replace("{{width}}", str(w)).replace("{{height}}", str(h)).replace("{{original_prompt}}", text)


def _section(template: str, name: str) -> str:
    """One [SECTION] of the caption prompt (for a chat server that takes a system and a user message)."""
    m = re.search(rf"^\[{name}\]\s*$(.*?)(?=^\[[A-Z]+\]\s*$|\Z)", template, flags=re.S | re.M)
    return m.group(1).strip() if m else ""


def is_json_caption(text: str) -> bool:
    """The user wrote the caption themselves: a JSON object."""
    s = (text or "").strip()
    if not s.startswith("{"):
        return False
    try:
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def parse_caption(raw: str) -> dict | None:
    """The JSON object in what the LLM wrote (thinking, code fences and chatter around it dropped), else None."""
    s = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\\n", " ")).strip().casefold()


def _palette(v) -> list[str]:
    out: list[str] = []
    for c in v if isinstance(v, list) else []:
        m = _HEX.match(str(c).strip())
        if m and f"#{m.group(1).upper()}" not in out:
            out.append(f"#{m.group(1).upper()}")
    return out


def _bbox(v) -> list[int] | None:
    """[y_min, x_min, y_max, x_max] on 0-1000, clamped; None when it isn't a box."""
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        y0, x0, y1, x1 = (max(0, min(1000, int(round(float(n))))) for n in v)
    except (TypeError, ValueError):
        return None
    return [y0, x0, y1, x1] if y1 > y0 and x1 > x0 else None


def sanitize_caption(obj: dict, user_text: str) -> str:
    """The caption as Ideogram 4 reads it, minified: only the schema's keys, in its order; bboxes clamped to 0-1000;
    colours as #RRGGBB; text only on text elements, and only words the user asked for (their quotes, or their own words
    when they quoted nothing), each once, so no sign is invented. Raises ValueError when nothing is left."""
    asked = [_norm(next(g for g in m.groups() if g)) for m in _QUOTED.finditer(user_text or "")]
    said = _norm(user_text or "")
    out: dict = {}
    ar = obj.get("aspect_ratio")
    if isinstance(ar, str) and re.fullmatch(r"\s*\d+\s*:\s*\d+\s*", ar):
        out["aspect_ratio"] = ar.replace(" ", "")
    if isinstance(obj.get("high_level_description"), str) and obj["high_level_description"].strip():
        out["high_level_description"] = obj["high_level_description"].strip()
    style = obj.get("style_description")
    if isinstance(style, dict):
        st: dict = {}
        for k in STYLE_KEYS:
            v = style.get(k)
            if k == "color_palette":
                if pal := _palette(v):
                    st[k] = pal
            elif isinstance(v, str) and v.strip():
                st[k] = v.strip()
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                st[k] = v
        if st:
            out["style_description"] = st
    comp = obj.get("compositional_deconstruction")
    if isinstance(comp, dict):
        cd: dict = {}
        if isinstance(comp.get("background"), str) and comp["background"].strip():
            cd["background"] = comp["background"].strip()
        elements, seen = [], set()
        for e in comp.get("elements") or []:
            if not isinstance(e, dict) or e.get("type") not in ("obj", "text"):
                continue
            el: dict = {"type": e["type"]}
            if box := _bbox(e.get("bbox")):
                el["bbox"] = box
            if e["type"] == "text":
                text = e.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                n = _norm(text)
                if n in seen or not (n in asked if asked else n in said):
                    continue
                seen.add(n)
                el["text"] = text
            if isinstance(e.get("desc"), str) and e["desc"].strip():
                el["desc"] = e["desc"].strip()
            if pal := _palette(e.get("color_palette")):
                el["color_palette"] = pal
            elements.append({k: el[k] for k in ELEMENT_KEYS if k in el})
        cd["elements"] = elements
        out["compositional_deconstruction"] = cd
    if not out.get("high_level_description") and not (out.get("compositional_deconstruction") or {}).get("elements"):
        raise ValueError("The caption is empty.")
    return json.dumps({k: out[k] for k in CAPTION_KEYS if k in out}, separators=(",", ":"), ensure_ascii=False)


def ideogram_graph(rec: dict, request: str, seed: int) -> dict:
    """The TextGenerate graph that writes a caption: the enhancer LLM as `rec` loads it, the whole caption prompt as the
    prompt (as Comfy's template sends it), thinking off, up to IDEOGRAM_MAX_LENGTH tokens."""
    settings = {**rec["settings"], "max_length": IDEOGRAM_MAX_LENGTH, "thinking": False}
    return graph({**rec, "system": None, "settings": settings}, request, None, seed)


def ideogram_caption(runner, comfy_url: str, text: str, width: int, height: int, seed: int = 0, cfg: dict | None = None) -> str:
    """Plain words → Ideogram 4's JSON caption, written by a local LLM and cleaned (sanitize_caption). The user's own JSON
    comes back as it is. The LLM's seed follows the image's, so a locked seed gives the same caption again; a caption
    that won't parse is asked for once more with the next seed, then it fails in words."""
    if is_json_caption(text):
        return text
    template = caption_template(comfy_url)
    cfg = cfg or {}
    for attempt in range(2):
        s = seed + attempt
        if cfg.get("backend") == "openai" and cfg.get("url"):
            raw = chat_openai(cfg, cfg.get("model") or "", _section(template, "SYSTEM"),
                              caption_request(_section(template, "USER"), text, width, height), 0.7, 0.95, s,
                              max_tokens=IDEOGRAM_MAX_LENGTH, english=False)
        else:
            rec = recipe(comfy_url, IDEOGRAM_LLM_FROM)
            raw = raw_text(comfy_url, runner.run(ideogram_graph(rec, caption_request(template, text, width, height), s)))
        obj = parse_caption(raw)
        if obj is not None:
            try:
                return sanitize_caption(obj, text)
            except ValueError:
                pass
    raise RuntimeError("The director couldn't write an Ideogram caption for this prompt. Try again, or write the caption as JSON.")
