# Pickups UI spec (v1)

Pickups: **power your creativity, locally.**
Audience: editors, colourists, filmmakers. Goal: *stupid simple*. Someone who has never seen it understands every screen in
two seconds and wants to do another shot as soon as one finishes.

## Principles

1. **One obvious next action per screen.** One primary button, big. Everything else is quiet.
2. **Verbs, not features.** Tiles are things you *do*: Remove, Recast, Render previz, Reshoot, Finish.
3. **Plain words only.** No model names, steps, CFG, LoRA, latent, seed in the main UI. A "Details" disclosure on a
   finished result shows the nerd facts (models, GPU time, settings) for those who want them.
4. **Always say what will happen and how long.** "About 4 min on your RTX 4090." Use the ETA the backend gives.
5. **Progress in human stages.** "Finding the person… Building the clean background… Blending… Done."
6. **The original is never touched.** Every result is a new *version* of the shot (like takes). A version strip shows them.
7. **Results invite the next step.** After Remove: "Keep going → Finish to 4K HDR".
8. **Proof, not promises.** Every result has a *Receipt*: which pixels are original, borrowed from other frames, or AI.

## Visual language

- Dark, calm, cinematic. Background `#0B0B0D`, surfaces `#141417`, raised `#1B1B1F`, borders `#26262B`.
- Text `#F4F4F5`, secondary `#A1A1AA`, tertiary `#71717A`.
- One accent: warm slate gold `#F5B83D` (primary buttons, selected tile, focus ring). Hover `#FFC857`.
- Receipt colours (overlay at ~55% opacity): original = no tint; borrowed from other frames = cyan `#3DD6F5`;
  AI-generated = magenta `#FF4FD8`.
- Status: success `#4ADE80`, warning `#FBBF24`, error `#F87171`.
- Font: Inter (bundle `@fontsource/inter`, weights 400/500/600/700). Numbers/timecode: `font-variant-numeric: tabular-nums`.
- Radii 12px (cards), 10px (buttons), 999px (pills). Generous spacing (8px grid). Subtle 1px borders, no heavy shadows.
- Icons: `lucide-react`, 20px, stroke 1.75.
- Motion: 150–200ms ease-out for hovers/panels; progress bars animate smoothly. Respect `prefers-reduced-motion`.
- Desktop-first (1280–2560 wide), must not break at 1024. Not a phone app.

## Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ● Pickups                 [Library]                 ● Studio · RTX 4090 · Ready │  ← top bar (48px)
├───────────────────────────────────────────────────────┬──────────────────────┤
│                                                       │                      │
│                    VIEWER                             │     RIGHT RAIL       │
│        (video, wipe handle, receipt overlay)          │  (tiles / task form  │
│                                                       │   / progress / result│
│ ───────────────────────────────────────────────────── │   actions)  ~380px   │
│  ▶  00:00:02:14 / 00:00:04:01      [Before|After|Receipt]                     │
│  ▕ filmstrip / scrubber with version strip below ▏     │                      │
├───────────────────────────────────────────────────────┴──────────────────────┤
│ Queue: ● Remove · pier_C115 · 2:10 left        ● Finish · castle · queued    │  ← queue bar (collapsible)
└──────────────────────────────────────────────────────────────────────────────┘
```

## Screens and states

### 1. Home (no shot open)
- Centre: a large drop zone card. Title **"Drop a shot"**, sub "MOV, MP4 or a Blender previz. Up to 4K, a few seconds is best."
  Button **"Choose file"**. Drag-over state: gold dashed border, slight glow.
- Under it: **"No footage handy? Try a sample:"** 3 sample chips with thumbnails (from `/api/samples`).
- Below: **Recent** — grid of shot cards (thumbnail, name, duration, small badges of finished versions e.g. "Remove ✓").
- Top-right status pill reflects `/api/status`:
  - Ready: green dot "Studio · RTX 4090 · Ready".
  - Missing models: amber dot "Setup needed" → opens Setup.
  - Not connected: red dot "Connect a GPU" → opens Setup.

### 2. Shot open, choosing what to do (right rail = Tiles)
- Viewer shows the shot (H.264 proxy) with play/pause, scrub, timecode, frame step.
- Rail title: **"What do you want to do?"**. Five big tiles (icon, verb, one line, time badge):
  1. **Remove** — "Erase a person, boom or sign." — `Eraser` icon
  2. **Recast** — "Same performance, new person." — `UserRoundCog`
  3. **Render previz** — "Turn a Blender previz into a real shot." — `Clapperboard`
  4. **Reshoot** `Beta` — "Move the camera after the fact." — `Move3d`
  5. **Finish** — "4K and HDR masters for delivery." — `Sparkles`
  Each tile shows a small time badge from status (`~4 min`) or a muted "Needs setup" if not ready (click → Setup).
- Tile hover: lift + gold border. Click → the rail turns into that task's form (with a back arrow).

### 3. Task forms (rail). Max two steps each. Primary button at the bottom, full width, gold.

**Remove**
- Step 1 "Click what should disappear." Instruction overlay on the viewer: cursor becomes a crosshair; click adds a
  green point (positive), Alt/Option-click adds a red point (negative). After each click call
  `POST /api/shots/{id}/select` and draw the returned mask on the viewer (gold 40% tint + 2px outline).
  Alternative for people who prefer words: a small text field "or type it: person, boom mic, sign…" + suggestion chips.
- Toggle (on by default): **"Include its shadow"**.
- Footer line: "Locked-off shots use real pixels from other frames. About 5 min."
- Primary: **"Remove it"**.

**Recast**
- Step 1 "Click the person to replace." (same click-to-select as Remove).
- Step 2 "Who's in?" — drop zone for a reference photo / character sheet (`POST /api/uploads`), shows thumbnail.
  Optional one line: "Describe them (optional)".
- Primary: **"Recast"**.

**Render previz**
- Step 1 "Add the look." Drop a look reference (character sheet, mood still or graded frame). Optional second image.
  One text line: "Describe the shot in a sentence." (placeholder: "A woman in a red coat crosses a rainy street at dusk.")
  Button **"Make the look frame"** → shows the generated frame large in the rail with **"Use this"** / **"Try another"**.
  Option chip: "I already have a graded first frame" → lets them drop it instead (skips generation).
- Step 2 Primary: **"Render the shot"**. Note: "Hide effects proxies (dust spheres etc.) in Blender first — they render as objects."

**Reshoot** (Beta badge)
- Step 1 pick a move: 5 preset cards with small animated arrows: **Push in**, **Pull out**, **Orbit left**, **Orbit right**,
  **Crane up**. One slider "Amount" (Small / Medium / Large).
- Primary: **"Reshoot"**. Muted note: "Works best on locked-off shots."

**Finish**
- Two big toggles: **4K** ("Sharper, cleaner, 3840×2160") and **HDR** ("HLG master + EXR for grading"). Both on by default.
- Quality segmented control: **Fast preview** / **Best** (default Best).
- Primary: **"Finish"**.

### 4. Running (rail)
- Card with the task verb + shot name, a big progress ring/bar, current human stage, ETA countdown ("about 2 min left").
- Stage list (checkmarks as they complete).
- "Cancel" text button. The user can go back to tiles or open another shot while it runs; it moves to the queue bar.
- The viewer keeps showing the original.

### 5. Result
- Viewer switches to the new version automatically with a **wipe** handle (before | after), draggable, starting at 50%.
- Viewer toggle: **Before / After / Receipt** (keys B / A / R). Receipt shows the colour overlay + a legend.
- Rail:
  - Title "Done in 3 min 12 s" + a small receipt summary bar: e.g. `96% original · 4% from other frames · 0% AI`
    (segmented horizontal bar using the receipt colours).
  - **Download** primary with a menu: "ProRes 422 HQ", "EXR sequence (zip)", "HDR master (HLG)" (only those available).
  - **"Keep going →"** suggestions (secondary buttons) e.g. "Finish to 4K HDR", "Remove something else".
  - "Try again" (new variation) as a quiet text button.
  - "Details" disclosure: GPU time, models used, resolution, frames.
- Version strip under the scrubber: Original · Remove v1 · Finish v1 … click to switch.

### 6. Setup (modal or full panel from the status pill)
- **Connect**: one field "Where is your ComfyUI?" default `http://127.0.0.1:8188`, button **Connect**; shows the GPU found.
- **Your machine**: GPU name, VRAM, RAM, ComfyUI version, and the verdict:
  "Ready — everything runs here" / "Runs here, slower" / "Needs a remote GPU (use the RunPod template)".
- **Models**: checklist per tile (✓ installed / ✗ missing with size). Button **"Get missing models"** (shows total GB).
  Gated models show "Accept license" link + note.
- **Benchmarks**: median measured times per task, grouped by GPU, the connected GPU first (from `/api/status`;
  the RTX 4090 reference scaled to this GPU until it has runs of its own).

### 7. Library
- Grid of all shots and versions. Filter chips: All / Remove / Recast / Previz / Reshoot / Finish.

## Keyboard
Space play/pause · ←/→ frame step · Shift+←/→ 1 s · B/A/R view mode · W toggle wipe · Esc back · ⌘O open file.

## Empty / error states (always one sentence + one action)
- Not connected: "Pickups can't reach your ComfyUI." [Open setup]
- Missing models: "Remove needs 2 models (9.4 GB)." [Get them]
- Job failed: "Something went wrong while blending." [Try again] [Details]
- Shot too long: "Shots up to 4 s work best. Trim to the first 4 s?" [Trim] [Use anyway]

## API contract (backend at the same origin; dev server proxies to http://127.0.0.1:7788)
All JSON. Media URLs are relative (`/media/...`).

- `GET /api/status` →
  `{ connected, comfy_url, machine: {gpu, vram_gb, ram_gb, comfy_version}, verdict: "ready"|"slower"|"remote"|"offline",
     tiles: { remove: {ready, eta_s, missing: [{file, gb, gated, url}]}, recast: {...}, previz: {...}, reshoot: {...}, finish: {...} },
     benchmarks: [{task, seconds, frames, size, gpu, runs, gpu_seconds, s_per_frame, fastest, slowest, measured, current, host}] }`
- `POST /api/connect {comfy_url}` → status
- `GET /api/samples` → `[{id, name, thumb_url}]`; `POST /api/samples/{id}/open` → shot
- `POST /api/shots` (multipart `file`) → shot
- `GET /api/shots` → `[shot]`; `GET /api/shots/{id}` → shot
  shot = `{ id, name, width, height, fps, frames, duration_s, thumb_url, video_url, versions: [version] }`
  version = `{ id, label, task, created_at, video_url, thumb_url, receipt_url?, receipt?: {original, borrowed, ai}, downloads: [{label, url}], details: {...} }`
  (the original is version id "orig", label "Original")
- `POST /api/shots/{id}/select {frame, points: [{x, y, positive}], text?}` → `{ mask_url, bbox }` (x/y in source pixels)
- `POST /api/uploads` (multipart `file`) → `{ id, url }`
- `POST /api/jobs {shot_id, version_id?, task, params}` → job
  - remove params: `{points, text?, shadow: bool}`
  - recast params: `{points, ref_upload_id, describe?}`
  - previz params: `{look_upload_id, look_upload_id_2?, describe, look_frame_upload_id?}`; keyframes are off and not
    supported in this release: leave `keyframes` unset (it is a test hook for the cut check, not a feature);
    `POST /api/previz/look {shot_id, look_upload_id, look_upload_id_2?, describe}` → `{url, upload_id}` for step 1
  - reshoot params: `{move: "push_in"|"pull_out"|"orbit_left"|"orbit_right"|"crane_up", amount: "small"|"medium"|"large"}`
  - finish params: `{upscale_4k: bool, hdr: bool, quality: "fast"|"best"}`
- `GET /api/jobs` → `[job]`; `POST /api/jobs/{id}/cancel`
  job = `{ id, shot_id, task, status: "queued"|"running"|"done"|"failed"|"canceled", stage, stages: [string],
           progress: 0..1, eta_s, error?, version_id? }`
- `GET /api/events` (Server-Sent Events): `event: job` with a job JSON on every change; `event: status` on status change.

## Tech
Vite + React 18 + TypeScript + Tailwind CSS, `lucide-react`, `@fontsource/inter`. No component library.
Source in `web/`, production build to `src/pickups/web/dist` (committed). `npm run dev` proxies `/api` and `/media` to
`http://127.0.0.1:7788`. A mock API (`VITE_MOCK=1`) with fake shots, simulated progress and sample media so the UI can be
built and demoed without the backend.
