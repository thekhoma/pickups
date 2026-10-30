# Pickups: projects, cast and delivery (spec v2)

Mission check: Pickups gives creators and editors powerful open-model tools for their clips (remove, recast, change
the weather, render previz, finish) in one simple, easy-to-navigate app on their own GPU. Frame.io's language
(projects, versions, approvals, deliveries) is inspiration only; Pickups adds what review tools can't do: **make the
next take**, and keep people and places **consistent across every task**.

## Words

| Word | Means |
|---|---|
| **Project** | A job: a film, a spot, a trip. Holds shots, cast and deliveries. |
| **Shot** | One source clip. Never modified. Badge: `ORIGINAL + 2 TAKES`. |
| **Take** | A result made from a shot or from another take. Slate always carries the verb: `TAKE 02 · REMOVE`, plus lineage `FROM TAKE 01`. |
| **Cast** | People, places, props and looks in a project: name, image, one describing line. |
| **Approved** | The only status (solo tool). Toggle with `⇧A`. |
| **Deliveries** | All approved takes with their files. |

## Navigation

Top bar: `Pickups●` · **[Project ▾]** switcher · **Make** · **Setup**; status at right. Breadcrumb on the shot page:
`Coastal spot ▾ / Harbour pier`. The queue stays in the bottom drawer.

- **Project switcher** (dropdown): recent projects with cover thumbnails, `New project…`, `All projects` (a grid of
  project cards: cover, serif name, mono `12 SHOTS · 31 TAKES · 4 APPROVED`). First run creates **My shots**.
  Built-in **Examples** is never a default target.
- **Make (home)**: hero example, examples grid, recipes. Below: the current project's shots. Dropping a file anywhere
  adds it to the **current project** (no picker).
- **Project page**: tabs **Shots** · **Cast** · **Deliveries**.
  - Shots: asset grid; each card shows its latest take, the stack badge and an approved dot.
  - Cast ("People, places, props, looks"): drop an image, give it a name and one line ("a woman in her thirties with short black
    hair and a red wool coat"), pick a kind.
  - Deliveries: approved takes, files per take, and one button **Finish all approved** (queues Finish 4K + HDR for each).
- **Shot page**: viewer, take stack, rail of verbs. `↑/↓` switch takes; `⇧A` approves; the wipe's left side can be any
  take (default: the take it was made from). Failed takes can be hidden. "Try again with changes" reopens the form with
  the take's params.

## Consistency: cast chips and the prompt receipt (the demo beat)

- Typing `@` in any prompt field opens the project's cast; a mention renders as an **inline chip** (16 px thumbnail +
  name).
- Under the field, the **prompt receipt** shows what each model will get, one mono row per model, live:
  - `QWEN-IMAGE 2.1 · image 2 = Maya · image 3 = Rooftop` (thumbnails) and the expanded instruction
  - `LTX-2.5 · "Maya, a woman in her thirties with short black hair…"`
  - `SCAIL-2 · reference = Maya` (Recast)
- Backend: `POST /api/mentions/preview {project_id, text}` → `{images, text, refs}`; tasks expand mentions server-side
  with `pickups/mentions.py` (unit-tested).
- The story for the demo: the same `@Maya` in Recast and in Render previz, in one project.

## Export

- Download names are NLE-safe and self-describing: `harbour_pier_T02_remove_ProRes422HQ_UHD.mov`.
- Masters keep the source's frame rate and start timecode so a take drops over the original in the timeline.
- `Reveal in Finder` for a take's folder (local installs).

## Notes (only if time)

Frame-accurate notes on a take, with **Make a take from this note** (the note prefills the next prompt). Backend
exists (`POST/DELETE /api/shots/{sid}/takes/{vid}/notes`).

## Out of scope for the contest build

A separate Projects page beyond the switcher's grid, grid/list toggle, sort/filter, archive UI, moving shots between
projects in the UI, zip exports, side-by-side compare, "In review", share links, multi-user, folders, NLE round-trip.
