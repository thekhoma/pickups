# Pickups: Image (create and edit), spec v1

The user's brief: "pick an image model, pick a resolution, pick an aspect ratio, enter a prompt and generate", intuitive,
with the sponsors' and the best open image models. Images live in the current project, next to its shots and cast, so a
generated still can become cast, a look frame or a first frame in one click. That's the bridge between the image tools
and the video tools, and it's what makes them one app instead of two.

## Where it lives

Top bar: `Pickups● · [Project ▾] · Make · Image · Setup`. **Image** opens the generator for the current project.
The project page gains an **Images** tab (grid of everything generated, newest first).

## The generator (one screen, no wizard)

Left column (fixed width ~360 px), top to bottom; right: results.

1. **Model**: a compact list of model rows (name, maker, one-line note, mono speed), the default first. Rows for
   models that can edit show a small mono `EDIT` tag (the catalog's `sponsor` flag isn't shown). Models whose
   files aren't installed are greyed with "Not installed · Setup".
2. **Aspect**: chips `1:1 · 4:5 · 3:4 · 2:3 · 9:16 · 16:9 · 3:2 · 4:3 · 21:9`, each with a tiny proportional frame
   icon. Default 16:9.
3. **Resolution**: a two-way segmented control `1K · 2K` with the exact size in mono under it (`1920 × 1088`), from
   the models endpoint. 2K is disabled for models whose max side is 1K.
4. **Prompt**: a large textarea with @cast chips (same component and prompt receipt as the video tools).
   Placeholder per model (FLUX 3: "Studio photograph of …"; Qwen edit: "Change the jacket to red leather").
5. **References** (optional, collapsed by default as "+ Add reference"): drop images, or pick from Cast / this
   project's Images. Adding one switches the button to **Edit** for models that edit (and to the first edit-capable
   model if the current one can't, with a one-line note). The first reference is the canvas (image 1).
6. **Count** `1 · 2 · 4`, **Seed** (in a small "More" row, empty = random).
7. **Generate** (ivory pill, `⌘↵`). The button shows the mono ETA from the model's speed × count.

Results (right): the newest batch large at the top, then earlier batches as a grid. Each image on hover:
**Download PNG** · **Add to cast** (inline name + kind + line, prefilled from the prompt) · **Edit** (uses it as the
canvas reference) · **Use as look frame** / **Use as first frame** (opens a shot picker → Render previz with `look_frame_ref` / Change
weather with `first_frame_ref`, both `{kind: "image", id}`) · **Use as reference** (previz `look_refs: [{kind, id}]`,
Recast `who_ref`) · **Same settings**
(restores model, aspect, resolution, prompt, seed). The recipe (model, size, seed, prompt sent) shows in mono under
the image in the detail view.

## API (live after the next server restart)

- `GET /api/image/models` → `{default, aspects, resolutions, models: [{id, name, maker, licence, modes, max_refs,
  max_side, speed, sponsor, notes, ready, sizes: {"1K": {"16:9": [w, h], ...}, "2K": {...}}}]}`.
- `POST /api/projects/{pid}/images/generate {model, prompt, aspect, resolution, count, seed?, refs: [{kind: "cast" |
  "image" | "upload", id}]}` → a job in the normal queue (`task: "image"`, `project_id`, `shot_id: null`); when done,
  `job.result_image_ids` and an SSE `project` event `{id, images}`.
- `GET /api/projects/{pid}/images` → `[{id, url, thumb_url, width, height, model, model_name, mode, prompt,
  prompt_sent, seed, aspect, resolution, refs, created_at}]` (newest first).
- `POST /api/projects/{pid}/images/{iid}/to_cast {name, kind, line}` → the new cast entry.
- `POST /api/mentions/preview` works for image prompts too (`uploads` = number of references before mentions).

## Rules

Same design system (v2.1): ivory pill for the one primary action, hairline containers, mono numbers. No model jargon
in the main flow (no sampler, steps or CFG). Everything else stays in the recipe line.
