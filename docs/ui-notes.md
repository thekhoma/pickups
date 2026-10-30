# UI notes for the backend

The web UI (`web/`) follows `ui-spec.md`. These are the places where it reads or sends something the v1 contract
doesn't spell out, so the backend can match them. Everything here is optional: the UI degrades gracefully without it.

## Read by the UI if present

| Field | Where | Use |
| --- | --- | --- |
| `machine.name` | `GET /api/status` | Status pill "Studio · RTX 4090 · Ready". Without it the pill reads "RTX 4090 · Ready". |
| `job.result_version_id` | job | Preferred over `job.version_id` for "the version this job made" (the backend uses `version_id` for the input). |
| `job.params`, `job.created_at`, `started_at`, `finished_at` | job | "Try again" resubmits `params`; queue order; "Done in …" fallback. Epoch seconds or ISO both work. |
| `version.before_url` | version | Left side of the wipe. Falls back to the original's `video_url`. |
| `version.details.total_seconds` / `gpu_seconds` | version | "Done in 3 min 12 s". Every `details` key is shown under **Details** (snake_case keys are humanised; `*_seconds` formatted as durations; arrays joined). |
| `version.details.resolution` / `rendered_at` / `frames` / `fps` | version | Details read Resolution (the top delivery size), Rendered at (only when the take was made at another size, e.g. previz 1920×1088 delivered as HD 1920×1080) and Length ("4.0 s · 97 frames"). Older takes recorded the working size as `resolution`; the rail reads their downloads for the delivered size. |
| `version.licences` | version | `[{name, licence, commercial}]` for `details.models`, derived by the server (`pickups/licences.py`), never stored. Details list each model with its license; `commercial: false` gets a NON-COMMERCIAL tag. |
| `receipt` | version | Percentages (0–100, as the backend writes them) or fractions (0–1); the UI normalises to whole percents. Finish takes that upscale add `upscaled` (its own share, never AI) plus `upscaler` and `scale` for the wording; see "Finish receipts" below. |
| `event: shot` | `GET /api/events` | Replaces the shot in place (new versions appear without a refetch). |

Version labels: the UI numbers versions per task in order ("Remove v1", "Finish 4K HDR v2") unless the label
already ends in `vN`.

Failures: the running card says "Something went wrong while *<last stage, lower-cased>*." using the last real stage
it saw, so the backend can set `stage` to anything on failure. `error` is shown under **Details**.

## Sent by the UI beyond the contract

- `params.frame` on `remove` and `recast` jobs: the frame the points were clicked on (same as the last `select` call).
  Safe to ignore if the backend tracks it from `select`.
- `version_id` on `POST /api/jobs` is only sent when the base is not the original.

## Endpoints the UI calls that aren't in v1

Each fails softly (a toast, or a "Missing" tile) if missing.

- `POST /api/models/download` `{}` → status (or anything). "Get missing models" / "Get them" in Setup. The UI then
  waits for `event: status` with no `missing` entries.
- `POST /api/shots/{id}/trim` `{seconds: 4}` → shot. The **Trim** action of "Shots up to 4 s work best. Trim to the
  first 4 s?" (shown when `duration_s > 4.05`).
- `GET /api/uploads/{upload_id}` (ids `up_<hex>` only: 400 otherwise, 404 when the file is gone) and
  `GET /api/projects/{pid}/refs/{kind}/{id}` (`upload`, or a redirect to the cast entry's or project image's
  thumbnail): the pictures behind the ids a take's settings keep, so a reopened form shows its references. A missing
  file shows a quiet "Missing" tile.

## v2 additions (home screen, recipes, downloads)

- `GET /api/examples` → `[{id, task, title, line, slate, sample_id, before_url, after_url, receipt_url?, poster_url, hero}]`.
  If the endpoint is missing (the SPA fallback answers with HTML), Home builds examples from `/api/samples` entries
  that carry `try` + `hint` (poster only, no loop). `POST /api/examples/{id}/open` → `{shot, task}`; on 404 the UI
  falls back to `POST /api/samples/{sample_id}/open`.
- `POST /api/shots/from_path {path}` → shot (the "or paste a file path" field). Errors show the backend's `detail`.
- `machine.vram_free_gb` (optional) → "18 GB FREE" in the status button.
- `tiles[task].missing_nodes` → listed under Models in Setup; a tile with missing nodes reads "Setup".
- Recipes are client-side (`web/src/lib/recipes.ts`). After a recipe step's job is done, the next step runs on
  `result_version_id`: Finish is submitted straight away with `{upscale_4k: true, hdr: true, quality: "best"}`;
  other steps open their form on the new take.
- Downloads: `{label, url, codec?, size?, bytes?}` with codec `prores | h264 | h265 | hlg | exr` and size
  `UHD | HD | native`. Without `codec`/`size` the UI guesses from the label. The menu groups ProRes, H.264, H.265, then
  HLG + EXR; the pill downloads the last choice (localStorage `pickups.download`), else ProRes at the top size.

## Change weather and the new Render previz form (Oct 17)

- **Change weather** (`task: "environment"`), in the rail between Recast and Render previz. Params:
  `change` (`snow | rain | fog | night | golden_hour | autumn`) or `custom` (text), `describe?`, and under Advanced
  `look_prompt?`, `prompt?`, `control` (`""` = auto, `"depth"`, `"canny"`), `seed?`. The Advanced placeholders mirror
  the presets in `tasks/environment.py` (copied into `web/src/components/FormsCreative.tsx`; keep them in step).
  Takes read "Weather · Snow" (older takes saved "Environment · Snow"; the store and `takeLabel()` show them the new
  way); the receipt is 100% AI. `details.first_frame` isn't shown until it has a URL.
- **Render previz** sends `look_upload_ids` (up to 4, shown as Image 2–5; image 1 is the playblast frame),
  `look_prompt`, `prompt`, `pin_start` (default off) and `look_frame_upload_id` when the user previews and keeps a look
  frame or drops a graded first frame. For older backends it also sends `look_upload_id` (the first reference). The
  look preview (`POST /api/previz/look`) sends `look_upload_ids`, `look_prompt`, `version_id?` plus the older
  `look_upload_id`, `look_upload_id_2` and `describe`; today's server reads only the older ones (one reference).
- The preview is optional now: without one, the render job makes its own look frame ("Making the look frame" stage).
- Take details show `look_prompt` and `prompt` as read-only, copyable blocks; `references`, `pinned_first_frame`,
  `change` and `control` get plain labels.
- **Prompt tips** popovers are static (condensed from `docs/prompting-guide.md`). Mismatch to settle: the guide says
  Qwen needs `<image1>`/`<image2>` tags, while `previz.py` and the UI's placeholder/hint say "image 2". If the backend
  switches to tags, it should translate "image N" in the user's look prompt, or the UI copy changes.
- **Reshoot** is a Labs tile and is hidden wherever this machine can't run it (`tiles.reshoot.ready` false): the rail,
  Setup's model list and the Library filter (unless there are reshoot takes). Tile times read `~00:04:00`.
- Copy (v2.2 positioning): hero "Power your creativity, *locally*.", recipes renamed ("Same scene, new lead"; added
  "New weather, 4K master"), the take's receipt section is titled "What changed".

## Projects, cast, approvals and deliveries (Oct 17, docs/projects-spec.md)

- **Navigation:** `Pickups● · [Project ▾] · Make · Setup`. The switcher lists recent projects (cover, counts),
  "New project…" (inline) and "All projects" (`#/projects`). The project page is `#/project/{id}` with tabs
  `/cast` and `/deliveries`; the old `#/library` opens the current project. The current project is remembered per
  browser (`localStorage pickups.project`); with nothing remembered it starts in the most recent project that has shots,
  else `unsorted`. `examples` is never a target: uploads and paths go to My shots then.
- **The current project changes only by an explicit choice:** picking it in the switcher, opening it from the
  Projects page, creating a project, or adding a new shot into a project (drop, file path, "Try it" copying a sample).
  Opening a shot (Home's examples, a deep link, a Deliveries row) or a project page reached by a link never changes
  it. When an explicit flow switches projects as a side effect (e.g. "Try it" lands in My shots), a toast says
  "Now working in <name>." Home's "Your shots" and the switcher always show the current project.
- **Examples project:** "Try it" copies into My shots, so the built-in `examples` project is left out of
  `GET /api/projects` (switcher, Projects page, counts) unless it holds something, as old libraries can.
- **New shots** go to the current project: `POST /api/shots` sends a `project_id` form field, `from_path` sends
  `project_id`. Home's "Your shots" is `GET /api/shots?project_id=…`.
- **Takes:** `status`, `hidden`, `input_version`, `params` and `downloads[].filename` are used. ⇧A / the Approve pill
  → `PATCH …/takes/{vid} {status}`; the hide button on a take → `{hidden}` (with "Show hidden"). Slates read
  `TAKE 03 · FINISH · FROM TAKE 02`. The wipe's left side is a take picker (default `input_version`; when that's the
  input, the take's `before_url` is used). ↑/↓ switch takes. "Try again with changes" reopens the form from `params`
  on the same input take (previz references are kept by id). Remove/Recast put the take's pick back (`frame`,
  `points`, `text`) as selected, "Selected · “woman” · as in the last take", without calling `select`: no find runs
  until the pick changes (a click, new words, Clear) or the user presses Show, which runs the same `select` to draw its
  mask (`selection.kept` in the store; the viewer reads "Same pick as the last take · click to add more"). While the
  pick is unchanged Remove keeps the take's `fill`; it always keeps `seed`. The prefill also carries `_take` (the
  take's id) and `_reference_url`.
- **Reveal in Finder** shows only when the page's host is localhost / 127.0.0.1.
- **Cast:** add (image + name + kind + one line), inline edit, duplicate-name errors shown as returned. Recast's
  "Who's in" offers the cast's people (`params.cast_id`), a photo, or a description alone (FLUX 3 klein makes a still
  first; the drop zone says "Or describe them below"). Recast sends `text` for a typed pick.
- **Recast's made person** (Oct 17): a Recast that made its person from words (`describe`, or FLUX's caption
  `reference_prompt`) keeps the still as the take's `reference.png`, served as `version.reference_url` (through
  `/media`; `Store.take_reference` validates the take id). "Try again with changes" shows it in "Who's in": "The same
  person as last time · Made from a description with FLUX 3 klein" with Change, and the words in the field ("Describe
  them", or "Their description" for a caption, which goes back as `reference_prompt`). Keeping it sends
  `params.reference_take` = that take's id: the task uses that still again (no new still; FLUX 3 klein stays in the
  take's models and licenses, since the person is its picture) and keeps a copy with the new take. Older takes kept no
  still: "The same description as last time · This take didn't keep its still, so FLUX 3 klein makes it again." (the
  words and seed go back). Editing the words means someone new (a new still; "Use the same person as last time" puts
  them back). While the person and words are unchanged the take's `prompt`, `scene` and `seed` go back too, and the
  prompt receipt shows that prompt.
- **@-mentions:** previz Look and Motion, Change weather "What's in the shot" (and its Advanced prompts), Recast's
  describe. Typing `@` opens the cast; mentions draw as ivory chips. The prompt receipt under each field calls
  `POST /api/mentions/preview` (350 ms debounce) with `uploads` = the number of previz references.
- **Deliveries** lists approved takes with their files (each file's label, e.g. "ProRes 422 HQ · HD 1920×1080",
  with `filename` as its tooltip; a native ProRes within 16 px of a delivery size stays in the take's Download menu
  only) and a secondary "Finish all approved". It asks first, inline: "Finish N approved takes · about X of GPU"
  (N × the Finish tile's `eta_s`; 90 min and over in hours, "about 2 h 40 min"), Queue / Cancel (Esc cancels), then
  `POST …/finish_approved`, which skips Finish takes and takes already finished or being finished. When every
  approved take is finished or finishing, a quiet mono line says so in the button's place.
- **Notes** (Oct 17): N on the shot view (not while typing; also the transport's Note button, "NOTE · N") pauses and
  opens a small input above the transport at the playhead, "Note on Take 02 at 00:00:03:12" ("on the original" for
  `orig`, with a line under it saying where it reaches the edit: "Goes to the edit with this shot's approved take"
  before any is approved, "Goes to the edit with Take 03" (or "Takes 02 and 03"), or "This moment isn't in Take 03, so
  it stays out of the edit"; none when the original is approved itself. The original's notes list says "They go to the
  edit with this shot's approved take." and marks a note outside the approved takes "· not in the approved take"; the
  frames come from `params.start_frame` along `input_version` and `details.frames`, as the export's fold; the take on
  screen even while Before shows), with "↵ save · esc cancel" under it; the Note button is lit and
  drops its tooltip while the input is open. Enter →
  `POST /api/shots/{sid}/takes/{vid}/notes {frame, text}` (the take on screen, `vid` = `orig` for the original; `frame`
  a whole number from 0, else 400 "Frame must be a whole number, 0 or more."; a take or shot id that isn't a plain
  folder name, such as `..`, is a 404), Esc cancels (the app's Esc-back doesn't fire) unless that Esc belongs to another
  field, a menu, a dialog or Setup, which keep it (the draft stays open); empty text does nothing. `frame` is the viewer's frame from the take's
  first frame, the same count the transport's timecode shows. Notes draw as small ivory ticks on the scrubber (hover
  shows the text, a press lands on the note's frame, dragging on scrubs) and as "Notes · N" rows in the take panel
  (and under the tiles once there are some); a row seeks, × → `DELETE …/notes/{nid}` with an Undo toast (Undo posts
  the note again, so it gets a new id). Toasts with an action (Undo, View) stay 10 s, and every toast's timer pauses
  while the pointer or focus is on it. Hovering a tick shows the note's text up to three lines (360 px wide). Optimistic both ways: a pending note has a `tmp_` id until the POST returns,
  and a failed call is rolled back with an error toast.

**Backend issue seen on :7788 (Oct 17):** `/api/mentions/preview` ignores `uploads` (a mention is always image 2, even
with 2 uploaded references) and returns plain "image 2" instead of `<image2>` tags. The current
`mentions.py`/`server.py` code does both, so the running process looks older than the code.

## Image page, Enhance and the Prompt director (Oct 17, docs/images-spec.md)

- **Navigation:** `Pickups● · [Project ▾] · Make · Image · Setup`. `#/image` is the Image page; the project page has
  an **Images** tab (`GET /api/projects/{pid}/images`, newest first).
- **Image page:** model rows from `GET /api/image/models` (name, maker, note, mono speed, EDIT / NON-COMMERCIAL tags;
  `ready === false` → greyed "Not installed · Setup"), aspect chips with frame icons, 1K · 2K with the exact size,
  prompt with @cast chips and the receipt, "+ Add reference" (upload, Cast, project Images), Count 1 · 2 · 4, Seed
  under More, Generate (⌘↵, mono ETA) → `POST /api/projects/{pid}/images/generate`. The job is task `"image"`; the UI
  shows the images from `result_image_ids` (or reloads the list) when it's done. Hover actions: Download PNG, Add to
  cast (`POST …/images/{iid}/to_cast`), Edit, Same settings, Use as look frame (previz `look_frame_ref`), Use as first
  frame (Change weather `first_frame_ref`), Use as reference (previz `look_refs` / Recast `who_ref`). The "Use as …"
  actions ask for a shot and open its form prefilled.
- **Enhance** (pen icon) on previz Look (`qwen-edit`, with `shot_id`, `version_id`, `frame` 0) and Motion (`ltx`),
  Change weather (`qwen-edit` / `ltx`), Recast (`ltx`), the Image prompt (`flux3` for FLUX, `qwen-image` for Qwen
  generate, `qwen-edit` with references). `POST /api/prompt/enhance` → the suggestion shows as a word diff under the
  field ("Director's version · via <runs_on>", Use / Keep mine / Try again); nothing is replaced silently. While it
  runs: mono "Directing… 00:00:03" and "Stop waiting". A 409 (director off) hides the buttons and links to Setup.
  First-run hint (once, `localStorage pickups.directorHint`) when the director is `comfy` with no `url`.
- **Setup · Prompt director:** three rows (Local LLM / On the GPU machine / Off). Local: "Find my LM Studio" →
  `GET /api/prompt/detect`; Text model + Vision model (optional) prefilled from `suggested_model` /
  `suggested_vision_model`; "Test & save" → `POST /api/prompt/test {url, model}` then
  `POST /api/settings {director:{backend:"openai", url, model, vision_model}}`. "Enter address manually" takes a URL.
  Nothing found → the polite panel ("Use the GPU machine" · "Turn it off" · "How to set up LM Studio"). Off asks
  first, then `{director:{backend:"off", off_acknowledged:true}}`; all Enhance buttons hide.
- **Previz:** `pin_start` defaults on; Advanced has "Paint the look over frame" (`look_at`, default 0 / "First frame",
  "Use current frame" takes the viewer's playhead).
- **Film grain:** previz and Change weather send `grain: "off" | "fine" | "35mm"` (default `"fine"`), under Advanced;
  the take details show `details.grain` as "Film grain".
- Cast grids, chips, pickers and the receipt use `cast[].thumb_url` (falling back to `url`). Queue rows read
  `REMOVE · <shot_name> · <project_name>` / `IMAGE · <model name> · <project_name>` from the job's `shot_name`,
  `project_name` and `model`; clicking one switches project first (image jobs open the Image page).

## ComfyUI workflows, timeline export and Share (Oct 17)

- **ComfyUI section** on a take's panel when `take.workflows` is non-empty: one row per workflow, named by its `title`
  ("Pick · SAM 3.1 · woman", saved by `jobs._save_workflows`, worked out from the graph for older takes; else the
  step), with Download (`filename`); identical rows in a row merge into one with a count and an "N files" toggle, and
  locked or near-locked Remove adds "Rebuild · real pixels, in Pickups" (no file), "Open ComfyUI ↗" to `GET /api/settings → comfy_url` (new tab; falls back to the status
  `comfy_url`; a bare `host:port` gets `http://`), and "Drag the JSON onto ComfyUI to tweak it". Hidden for `[]`.
- **Export timeline ▾** on Deliveries, next to "Finish all approved": "Resolve / Premiere (XML)" →
  `GET /api/projects/{id}/export/xml`, "EDL (CMX3600)" → `…/export/edl`. The UI fetches the file and saves it under the
  `Content-Disposition` name (fallback `timeline.xml` / `.edl`). Mono hint: "Approved takes at their source timecode.
  Verified importing into DaVinci Resolve." A 409's message replaces the hint for a few seconds (no error styling);
  other failures show an ERROR notice with Try again. While one export runs the other control is disabled. The file
  name is `<Project>_approved.xml` with an ASCII `filename` ("Pickups" for a name in another script) and, when the
  project's name has other letters, the name as typed in `filename*` (RFC 5987), which the UI prefers. Each approved
  take's notes go along (`nle.Clip.notes`, frames from the take's first frame) at their place in the edit: in the XML as sequence markers,
  `<marker><name/><comment/><in>frame</in><out>-1</out></marker>` in `<sequence>` with `in` = the take's record
  offset + the note's frame (name: the text cut at 60 characters; comment: the whole text and the take's name; notes of
  stacked takes on one frame share a marker; sequence markers only, since Premiere would show clip markers twice); in
  the EDL as Avid locators after the event, `* LOC: <record timecode> BLUE    <text>`. Text is one line; a note past
  the end of a take sits on its last frame. **Notes on the original** (Oct 17) go to the edit with the shot's approved
  takes, unless the original is approved itself (then they go with it, once): `nle.fold_original_notes` places each at
  the take's frame showing the same moment (the take's start in the original is `params.start_frame` added up along
  `input_version`; a take of the whole original maps 1:1; another frame rate maps by time), drops notes outside the
  take, and the export passes them as `nle.Clip.original_notes`. Their marker comment names `<shot> · original`
  instead of the take ("Boom in frame (Pier · original)"; folded into two stacked takes it is one marker), the EDL
  locator ends " (original)", and an approved take's receipt lists them as "<text> (original)".
  DaVinci Resolve 21.1 imports `<marker>` only in `<sequence>`: in a `<clipitem>` (in any child order), a `<file>` or
  a master `<clip>` it's ignored (tested Oct 17; they land as Blue timeline markers with name and note; re-tested with
  both kinds in one file: 4 notes gave exactly 4 timeline markers and no clip or media pool markers).
  Resolve's EDL import doesn't read `* LOC:` lines (they're for Avid).
- **Share ▾** on a take (not the original): "9:16 stacked (Reels, Shorts)" / "1:1 wipe (feeds)" →
  `POST /api/shots/{sid}/takes/{vid}/share {format: "9x16" | "1x1"}` → `{url, filename}`. "Rendering… 00:00:03" while
  it runs, then a small looping preview and Download (`filename`). Errors show an ERROR notice with Try again.
- Mock: takes carry generated `workflows` (data: URLs, generic node graphs), `share_9x16.mp4` / `share_1x1.mp4` are
  synthetic (`web/scripts/make-mock-media.sh`), and export builds a small XML / EDL from the approved takes (409 when none),
  with their notes as markers / locators. Mock notes live in the mock session like everything else.

## Review fixes, asset management and the demo autopilot (Oct 17)

- **Enhance diff:** your text (struck through) and the director's version render as two pre-wrap blocks with spaces
  kept and @mentions as inline chips; new words are lightly highlighted. Busy copy: "Directing… usually 5–20 s".
- **Mock in every build:** `?mock=1`, `?mock=fresh` and `?tour=1` work from :7788 (lazy `mock` chunk; `dist/mock`
  holds ~3 MB of synthetic clips, served by the SPA route). Mock sessions remember their project under
  `pickups.mock.project`. With `/examples/examples.json` reachable, the mock uses the real loops.
- **Long shot** moved into the task forms where a limit applies: Recast, Change weather, Render previz and Reshoot at
  4 s, Remove at 10 s, Finish none. Hidden once the shot has a take; "Use anyway" is remembered per shot
  (`localStorage pickups.longShotOk`); "Trim" calls the trim endpoint with that task's limit.
- **Queue:** verb names everywhere ("CHANGE WEATHER", "RENDER PREVIZ", …) and remaining time ("~5 min left", "~40 s
  left"), collapsed and expanded.
- **Image page:** Prompt first, then references, then the model as one row with "Change" (opens the list), aspect,
  and size + count side by side; all of it fits at 1440×900.
- **Receipt** is only offered on takes that have `receipt_url`. **Take thumbnails** are preloaded when a shot,
  project or project list loads, and a thumbnail whose URL changes no longer stays black.
- **Examples:** the card dims and says "Opening… (copies the example into My shots)" (or "Opening…" for examples with
  `shot_id`). `POST /api/examples/{id}/open` → `{shot, task, params}`: the UI opens the task form on that shot and
  prefills it from `params` (kept references read "From the example"). A copied sample is a new shot in My shots,
  so the UI switches there with the toast "Now working in My shots."; an example's own shot just opens.
- **Form header:** shot name, then the take as a chip ("On take 02 · Render previz"), wrapping instead of truncating.
- P2: the @-list closes after a pick; the receipt collapses double spaces; Esc only closes Setup, the queue, menus,
  dialogs and forms (it never changes page); "Shown in Finder" toast; "Open…" is disabled until a path is typed;
  accessible names on shot cards, verb rows, project cards, download menu items, model radios and 4K / HDR; the
  clean-up recipe reads "Remove, then remove again"; Home rows have edge fades and arrow buttons; VRAM reads the same
  way in the header and Setup.
- Fixed while testing: `navigate({name: "image"})` wrote `#/`, so the Image link sent you Home from a shot page.

### Asset management

- **Projects:** "Rename · Archive" under the title (`PATCH /api/projects/{id} {name}` / `{archived}`), with Undo on
  the toast. Archived projects leave the switcher; All projects has "Show archived" (only when there are any) and
  marks them ARCHIVED; an archived project's page offers "Unarchive".
- **Delete (Run 6):** "…" on a project card, and on an asset card in Assets (a clip, a take, a project image or a cast
  entry), ends with Delete. It asks first in place (`ConfirmDelete`: "Delete <name>?", one line for the shots that go
  with it or use it, Cancel / Delete project or Delete asset). Nothing is erased: everything moves into
  `library/trash/<id>/` under its library path (`store._into_trash`), with an Undo toast and Restore in Assets > Trash.
  `DELETE /api/projects/{id}` (→ `{entry, projects}`; My shots comes back empty, Examples is 403, 409 while a job works
  in it), `DELETE /api/projects/{pid}/cast/{cid}`, `DELETE /api/projects/{pid}/images/{iid}` (409 while a job uses it).
  Deleting the current project moves work to the most recent project left, else the empty My shots. Takes that used a
  deleted picture keep their settings; a new take, or Try again, goes without it (`store.live_refs`, `withoutGoneRefs`).
- **Shots:** the shot name in the breadcrumb is a menu: Rename (inline) and "Move to another project" (non-archived
  projects) → `PATCH /api/shots/{id} {name}` / `{project_id}`, with Undo on the move toast.
- **Shots grid:** mono segmented controls: Show All · Approved · Has takes, Sort Newest · Name.
- **Cast:** "Add from another project" → `GET /api/cast` (grouped by `project_name`, names already in this cast are
  disabled) → `POST /api/projects/{pid}/cast/import {from_project_id, cast_id}`. Cards show "USED IN 3 SHOTS" /
  "NOT USED YET" from `cast[].uses`. An older server without `/api/cast` gets "Restart Pickups to update it".

### Stalled jobs and licenses

- A job with `stalled: true` shows a calm mono STALLED slate plus `stall_message` (queue chip: "STALLED" with the
  message as its tooltip; expanded queue and the running view: the message in full, in a hairline box). The tally
  dot stays steady, the countdown stops and dims, nothing turns red. It clears when the server drops the flag.
  `?mock=stall` shows it.
- Image models with `commercial: false` get a mono NON-COMMERCIAL tag; the license name shows under the row in the
  model list and as the row's tooltip.
- **Setup → 04 Licenses** holds the Frame editor choice, two radio rows: Qwen-Image 2.1 (default, research license)
  or the alternate editor. It saves as `commercial_safe` in `GET/POST /api/settings` (true = the alternate editor; the
  setup tiles swap model files, and the server re-broadcasts status). Below it, the NVIDIA RTX accelerators switch
  (`allow_nvidia_rtx`, off by default). Both save optimistically and roll back on error. A take's own
  `commercial_safe` param overrides the setting.
- **ComfyUI busy with outside work:** `status.comfy_queue` = `{running, pending, outside}`, refreshed by the server
  every 15 s. `outside` (ComfyUI is running something and no Pickups job is) turns the status line into
  "<machine> · <GPU> · Busy with another ComfyUI job" (dot unlit), and Setup → Your machine's line into "Busy: ComfyUI
  is running a job Pickups didn't send" (dot unlit, never "Ready" above it) with the advice under it once: "New renders
  wait for it. If it never finishes, restart ComfyUI on the GPU machine." `?mock=busy` shows it.
- **Top bar under 1000 px:** the status button says only the state ("Ready", "Busy", "Setup needed", "Connect a GPU",
  "Checking…"); the tooltip and aria-label carry the whole line. Under 640 px only the dot shows (it still opens Setup).
- **Toasts:** four tones, each with its slate: `success` DONE (finished work only), `queued` QUEUED (sent to the GPU
  queue, nothing done yet: "Finish 4K HDR for 2 approved takes."), `info` NOTE, `error` ERROR. A take that lands
  while you're elsewhere says it in the app's take naming: "Take 03 is ready · Finish 4K HDR · <shot>" (View).
- **Stills before frames:** every take video shows its own take's `thumb_url` as its `poster` until it has a frame
  (the viewer's after side; the wipe's before side shows the before take's still, and none when that still is the
  same image as the after's). The first load's shimmer sits under the videos. Home example loops rest on the after
  (with `poster_url`, which must be the after's still) and the hero starts sweeping once the before clip has a frame.
  Deliveries warms its cards' stills when the list loads.

### Example gallery download

- When `GET /api/examples` is empty, Home asks `GET /api/examples/pack`; if `available && !installed` the gallery's
  place shows "See what it can do." with "Get the examples" → `POST /api/examples/pack` (mono "Downloading… usually
  10–30 s"), then the examples reload. Errors show inline with an ERROR slate. With shots, a compact version leads the
  examples row. Examples with `can_try: false` still play but aren't clickable and drop "Try it on this shot".
  `?mock=fresh` starts without a gallery.

### Setup models list

- Setup → Models lists Remove, Recast, Change weather, Render previz, Finish, then **Images** (the `image` tile:
  the default image model's readiness). Reshoot isn't listed (it stays in Labs); "Get them" counts only these rows.

### Demo autopilot

`?tour=1` (docs/demo-autopilot.md, implementation notes there). Its words come from `/examples/tour.json` when the machine has one (generic built-ins otherwise); with real examples the mock's Home lists all of them, each opening as its own shot with the real before/after. Mock only; renders take 4 s with the countdown
running from the real estimate; Share in the mock composes the 9:16 / 1:1 cut-down in the browser from the take;
Export timeline doesn't save a file during the tour. Fires `tour:done`. `npm run tour` records it (Playwright 1.63.0
dev dependency).

## Receipt sheets (Oct 17)

The receipt you hand a client: one printable page per take (`src/pickups/receipts.py`).

- `GET /api/shots/{sid}/takes/{vid}/receipt` → `text/html`, inline (404 for an unknown shot or take, and for `orig`,
  which has no receipt). One self-contained page: inline CSS, the bundled OFL fonts and the JPEG stills as base64, a
  Content-Security-Policy that allows no requests, a screen-only "Print or save as PDF" button; A4 print styles (the
  phone layout is `screen` only, so print keeps the stills side by side: about 3 A4 pages a take).
  Sections in order: header (Pickups receipt, project, shot, "Take NN · <label>", made, total and GPU time), before /
  after at the same frame (the clicked `params.frame`, else the middle; ~1280 px, cached as `receipt_before.jpg` /
  `receipt_after.jpg` / `receipt_map.jpg` in the take's folder), what changed (the bar and whole percents as the app
  shows them, "100% AI: the whole frame is generated." for fully generated takes, pixels changed outside the
  selection, and the receipt on the picture when the take has `receipt.mp4`), how it was made, models and licenses
  (`take.licences`, NON-COMMERCIAL tagged, a callout when any that says only that it isn't for paid work: what to do
  about it is in the take's Details in the app, with a Setup link; licenses that allow paid work on their own terms
  print them from `licences.TERMS`, e.g. LTX's revenue threshold), prompts as each model received them (the text-encoder
  inputs of the saved workflows, with seeds and negatives, SAM labels as "Selection"; older takes fall back to
  `details` / `params`), timecode in / out (`details.timecode` or the master's, out exclusive as in an EDL), files
  delivered (download file names and sizes, never paths; names break only after an underscore), notes (the timecode
  Pickups shows, from the take's first frame, with the source timecode or frame under it; a note past the end sits on
  the last frame), ComfyUI workflows by title (embedded as data: JSON links; "Rebuild · real pixels, in Pickups" as in
  the take panel; a take without saved workflows gets one line instead: "This take was made before Pickups kept its
  ComfyUI workflows."), footer. "Made" reads "17 Oct 2026 · 01:35".
- `GET /api/projects/{pid}/receipts` → `application/zip`, `<Project>_receipts.zip` (named as the timeline export:
  ASCII `filename`, "Pickups" for a name in another script, plus `filename*`): `index.html` (thumbnail, shot,
  take, what changed ("100% AI · whole frame generated" for fully generated takes), license flags, a link per take;
  it mentions `workflows/` only when the zip has some), `takes/<Shot>_T02_remove.html` (the page above, linking its
  workflows instead of embedding them) and `workflows/<Shot>_T02_remove/<NN>_<title>.json`. Approved takes only
  (approved originals are skipped); none → 409 "Approve some takes first (⇧A on a take); receipts are made from
  approved takes."
- **Take panel:** a "Receipt sheet ↗" pill beside Share opens the page in a new tab (`api.takeReceiptUrl`); "sheet"
  keeps it apart from the transport's Receipt view (R), which tints the picture. Deliveries rows have a quiet
  "Receipt sheet ↗" link beside Reveal in Finder (not for originals).
- Take order: `Store.versions()` sorts takes by `created_at` (folder mtime only as a fallback), so writing into a take's
  folder (a note, an approval, the receipt stills, the timeline's hard-linked master) never renumbers takes.
- **Deliveries:** "Export receipts" (secondary, beside Export timeline) downloads the zip under its
  `Content-Disposition` name. While it runs the hint reads "Making the receipts… a moment per take."; 409 and errors
  behave as for the timeline (calm message in the hint, else an ERROR notice with Try again). An older server gets
  "This server can't export receipts yet. Restart Pickups to update it."
- Mock: the receipt is built in the browser from the mock take with the backend's layout (`RECEIPT_CSS` in mock.ts is
  a copy of `receipts.CSS`: change both), numbered sections, the print button and the non-commercial callout. It's a
  Blob URL on the app's origin, so it uses the app's own font files (the @font-face rules, made absolute); the takes'
  thumbnails stand in for the stills. "Export receipts" saves one combined HTML page (`<Project>_receipts.html`: the
  index with thumbnails, then every approved take's receipt) instead of a zip.
- Phones (under 768 px): the shot view stacks the take panel under the viewer (440 px tall) and scrolls; the viewer's
  size/fps slate is hidden.

## Finish receipts and the non-commercial pill (Oct 17)

An upscale is the footage resampled (new detail, no new content), so it is a share of its own, never AI. The only
generated pixels of a Finish take are the HDR highlights LTX-2.5 rebuilt (`hl`: % of pixels, 0 with HDR off).

- **Shape** (`receipts.finish_receipt`, written by `tasks/finish.py`). With an upscale:
  `{"original": 0, "borrowed": 0, "upscaled": 100 - hl, "ai": hl, "upscaler": "SeedVR2 3B", "scale": 2}` (`upscaler`
  is "NVIDIA RTX Video Super Resolution" on the fast path; `scale` is the real factor, output over input width).
  HDR only: `{"original": 100 - hl, "borrowed": 0, "ai": hl}`.
- **Older takes** said `ai: 100` for any upscale. `Store.version()` reads them the new way
  (`receipts.finish_receipt_on_read`: `hl` from `details.hdr` "… on 13.24% of pixels …", the upscaler from the first
  non-LTX model, the factor from `details.input` and `output` / `rendered_at` / `resolution`) without rewriting
  meta.json, so the API, the receipt sheets and the zip all see the new shape.
- **Wording** (the same in `receipts.py`, `lib/format.ts` and the mock): "Upscaled 2× from your footage by SeedVR2 3B:
  new detail, no new content." ("no new content" for RTX, which only sharpens) and "HDR highlights rebuilt by LTX-2.5
  on 13% of pixels." The take panel's "What changed" and Deliveries cards put both under the bar
  (`finishSentence()`); the receipt sheet's section 02 leads with them (an HDR-only take keeps "87% of the picture is
  the original footage, untouched." and adds the highlights line); the zip index summary reads
  "87% upscaled · 13% AI" with the upscale sentence under it.
- **Bar and legends:** an Upscaled segment in neutral grey (`bg-ink2` in the app, `--upscaled` on the sheet), never the
  AI magenta. A receipt with an upscaled share names Upscaled and AI (Original or Other frames only when some is
  left): `receiptShares()` / `receipts.shares()`. Every "% AI" leaves the upscaled share out; `aiShare()` reads
  "Upscaled 2× · 9% AI" for an upscale (as the share slate) and "13% AI" otherwise. "100% AI · whole frame generated"
  needs nothing original, borrowed or upscaled ("… (upscaled 2×)" for a Finish of a previz take).
- **One rounding everywhere** (Python `receipts.percents()` / `percent_texts()`, TS `receiptPercents()` /
  `percentText()`, the mock, the share slate): largest remainder to whole percents; a share that is there but rounds to
  0 reads "<1%" ("Under 1%" at the start of a sentence), and a share rounded up to 100% while another is there reads
  ">99%" ("Over 99%"). `receiptShares()` items carry `text`; the bar, legend, summaries and the sheet's stats use it.
- **Share videos:** the slate reads "Upscaled 2× · 13% AI" for an upscale (`share.stat_line`, from `percent_texts`),
  never "0% original pixels", and ">99% ORIGINAL PIXELS" rather than 100% when any AI is there.
- **NON-COMMERCIAL pill:** the take header in the take panel and Deliveries cards show the mono NON-COMMERCIAL tag when
  any of `version.licences` has `commercial: false` (inherited ones included). It is a focusable button: a click,
  Enter or a tap opens a small note (`role="note"`, Esc or a click outside closes it) naming the model(s) ("Qwen-Image
  2.1 (via Take 02)") and, for the default frame editor: "For paid work, set Setup → Licenses → Frame editor to
  the alternate editor and make the take again." ("make Take 02, then this take, again" when it came with the input),
  with "Open Setup → Licenses". That guidance stays in the app; the client's receipt sheet only says the take isn't
  for paid work. Deliveries adds one quiet mono line under the export status, "N of M approved takes use a
  non-commercial model", and the "Finish all approved" confirm row says "· N use a non-commercial model".
- Mock: Finish takes get the same receipt shape (the real take's highlights share where it has one, else 8.6%) and
  `details.hdr`; the receipt page, the combined receipts page and the share slate mirror the backend. `RECEIPT_CSS`
  gained `--upscaled`, `.bar .u` and `.sw.u` (as `receipts.CSS`). Remove and Recast takes get a `receipt_url` (as the
  backend, which writes receipt.mp4 for those two), and so does a Change weather that keeps something as shot; Finish,
  plain Change weather and Render previz never offer the R view. A Recast made from words keeps `reference_url`: the
  real take's FLUX 3 klein still, else the Fun cast still the words ask for.
- Mock media (Oct 17): everything in `web/public/mock` is cut from real takes by `web/scripts/make_mock_media.py`
  (shots, takes, receipt overlays, aligned before clips, the people finder's cutouts and mattes, the Fun cast's sheets
  and FLUX stills). Four demo projects: Highlands & Alps, Coast road, Stunt previz (the tour's) and Fun cast.

## A take made from a take (Oct 17, read time)

A take made from another take (`input_version` not `orig`) carries that take's pixels. `Store.shot()` (which has every
version) puts this right at read time, without writing meta.json; the API, receipt sheets, the zip and share videos
all read it. `Store.version()` on its own does not (it can't see the other takes).

- **Licenses:** a take's `licences` gain every model of its input chain it didn't run itself, each with
  `via: "Take 02"` (`ModelLicence.via`). The pill, the receipt sheet's callout ("Qwen-Image 2.1 (via Take 02)"), its
  table ("Qwen-Image 2.1 via Take 02") and the zip's non-commercial count all see them (`receipts.licence_name`,
  `licenceName()`). The take panel's Details list says "· via Take 02".
- **Finish receipts** are composed with their input's (`receipts.compose`, mirrored by `composeReceipt` in the mock):
  with an upscale `ai = hl + (100 - hl) × in.ai`, `borrowed = (100 - hl) × in.borrowed`, `upscaled` = the rest; without
  one the input's shares scale by `(100 - hl)` and `hl` joins the AI (the highlights are assumed spread evenly over the
  input). Added keys: `highlights` (the Finish's own HDR share, for "HDR highlights rebuilt … on 9% of pixels") and
  `source: {take: "02", label, receipt}`; `source_upscaled` marks an HDR-only Finish of an upscaled take. A previz input
  stays 100% AI: "100% AI · whole frame generated (upscaled 2×)". Wording: "Upscaled 2× from Take 02 by SeedVR2 3B: new
  detail, no new content. Take 02 (Remove) is 95% original · 4% other frames · 1% AI. HDR highlights rebuilt by
  LTX-2.5 on 9% of pixels." (`upscaleSentence`, `sourceSentence`, `highlightsSentence`, `finishSentence`). Other tasks
  keep their own shares (their receipt map shows only their own pixels); only their licenses are inherited.
- **details.hdr** of a Finish reads as its receipt rounds it: "highlights rebuilt by LTX-2.5 on 13% of pixels;
  everything else is your footage, upscaled" ("… is Take 02, upscaled"; `receipts.hdr_detail`), so the sheet's sections
  02 and 03 agree.
- **Older takes' models** (`store._models_on_read`, also `details.models` as read): a Recast whose person FLUX 3 klein
  made from a description (a saved `cast` workflow step, or `params.reference_prompt` without `cast_id` / `who_ref` /
  `ref_upload_id`) gains "FLUX 3 klein"; a Remove of real pixels only (receipt `ai == 0`, no `plate` step, no "plate"
  in `details.method`) drops the frame editor it listed without using.

## Deliveries: finished takes, notes, top bar, tour (Oct 17)

- **Finished takes:** a Deliveries card whose take has a Finish that is done (visible, not approved) shows "Finished as
  Take 03 · 4K HDR" with **Use this** (`approveFinishedTake`: approves the Finish, unapproves the take; toast with
  Undo). While any wait, the header says "2 finished takes are waiting for approval" in place of "Every approved take
  is finished or finishing.". The "Finish all approved" toast with nothing to queue says "Every approved take is
  finished or finishing." (info).
- **Notes composer:** on the original with no approved take: "Goes to the edit once a take of this shot is approved".
  On a take that isn't approved while a take made from it is (its Finish): "Not in the edit: Take 03 is approved" (its
  notes aren't folded into later takes; only the original's are).
- **Kept pick row:** "Selected · 1 click" on one line and "Same as the last take" (or "the example") under it.
- **Top bar under 640 px:** the project switcher's button is at most `min(112px, 100%)` wide and truncates; the status
  dot keeps its 24 × 32 box but a transparent `::after` makes a 44 px target.
- **Tour caption:** off the shot page it tries the Image page's empty results canvas first, then bottom-left,
  top-right, top-left and bottom-right, and takes the first spot that covers no visible control (buttons, links,
  fields, radios, menu items; re-read every 200 ms), else the one that covers least.

## Who's in the shot: Recast's people finder (Oct 17, `pickups/people.py`)

`POST /api/shots/{id}/people` body (all optional) `{version_id?, refresh?}` →
`{frame, people: [{index, box, area, point, points, thumb_url, mask_url, seen, main}], main, probed, window, width,
height, cached, partial?}`. Types: `PeopleRequest` / `PeopleResult` / `PersonFound`; `api.people()`; the mock answers
per media key (one person on the pier and the playblast, none on the castle or after a Remove, two on an upload, one
in the middle otherwise).

- **How:** up to four SAM 3.1 calls ("person:8", one mask per person) on frames 0, ¼, ½ and ¾ of `window`, the first
  frames Recast renders (`common.MAX_FRAMES`, as 4k+1, read at call time), at the size SAM sees in Recast and
  `/select`. The frame with the most people wins (ties: nearest the middle). About 5 s the first time on the 4090,
  then cached in `shots/<id>/people/<orig|take>.json` (instant; `refresh: true` looks again).
- **Fields:** source pixels throughout. `box` `[x, y, w, h]`; `area` is the share of the frame (0–1); `point` the
  deepest point inside the person (usually the chest); `points` that point plus, when there's room, one in the head and
  one in the legs, all `positive: true`; `seen` on how many probed frames they were found; `main` the person largest
  across the probed frames (big and on screen longest). People are listed left to right; `main` is an `index` or
  `null` when nobody was found. `thumb_url` is a 192 × 256 JPEG (the person in colour with a thin light outline, the
  rest dimmed); `mask_url` their matte on `frame` in the same RGBA as `/select`'s (works with `MaskOverlay`).
- **Use with Recast:** send `params.frame = frame` and `params.points = person.points`, with **no `text`**. That is
  what a click gives Recast already (`common.tracked_masks`: SAM selects from the points on that frame, then tracks
  both ways), so recast.py needs no change. Send all of `points`, not just `point`: on the 4090 one chest point picked
  only part of a half-hidden man in the Eilean Donan couple (IoU 0.19 with his matte), the three points 0.90; every
  other person tested re-selected at 0.93–1.00 either way.
- **Errors:** 404 `{detail}` for an unknown shot or take; 502 `{detail: "Couldn't look for people in this shot: …"}`
  when SAM can't run (it gives up after the first failed call, so this is quick). If SAM fails after the first frame
  the answer has `partial: true` and isn't cached. Nobody found is an answer: `people: []`, `main: null` (cached).

## The simple Recast form (Oct 17)

Three things, one button: who to replace (found for you), **01 Your character**, **02 Describe them**, **Recast**.
Everything else sits under **More options** (closed).

- **Who to replace:** on open the form calls `api.people(shot, {version_id})` (the take it works on) and shows a quiet
  "Finding who's in the shot…" row. The `main` person is picked with no click: `pickPerson` sets the selection to
  `points` on `frame` and uses their `mask_url` as the viewer matte (no `/select` call; without a `mask_url` it runs
  `/select` like a click). The row reads "Replacing [cutout] the only person · Not them?" ("the person most on screen",
  "the person on the left" … when there are several, with a strip of cutouts: one tap switches). "Not them?" clears
  the pick and shows the click panel ("Click them in the picture."). Nobody found: "No one found. Click them in the
  picture." A finder error (offline, older backend) falls back to the click panel quietly. A click in the picture or
  words take over from the finder (the strip stays, one tap goes back). "Try again with changes": the take's pick
  wins; when it is someone the finder found, their matte is drawn at once. Answers are cached per shot and take for
  the session (the backend caches them too).
- **Clicks after an automatic pick** (fixes, Oct 17): `pickPerson` marks the selection `auto: true` (with `autoBase` =
  how many of its `points` are the finder's). While it is `auto`, a plain click never adds to it: `addPoint` asks the
  form (`setAutoClick`) where it landed. Inside another found person's box on the finder's frame: switch to them
  (their points and matte, no `/select`). Inside the picked person's box: add to them (a refine; `auto` goes false,
  `autoBase` stays, Undo back to the finder's points makes it `auto` again). Anywhere else, or on another frame: a new
  pick `[point]`. Option-click negatives refine as before. Words ("Pick who to replace by words", `selectByText`) drop
  the finder's points, so `/select` and Recast get the words (plus only the user's own clicks). The viewer chip reads
  "Click someone else to switch" on an automatic pick, and "Finding who's in the shot… or click them" while the finder
  looks (`selection.finding`, set by the form). The status row speaks one language throughout: "Replacing [cutout]
  the person most on screen · Not them?", "Replacing the person you clicked · Undo · Clear" (a found person refined by
  clicks keeps their cutout and label), "Replacing “the man in the coat”", "Replacing the person from the last take
  · Show"; no click counts. At phone width (≤ 767 px) the copy says "Tap them in the picture above." and "Not them?"
  scrolls the picture (`[data-stage]` on the viewer frame) back into view.
- **Your character:** the project's cast people as chips (`cast_id`), "Drop a photo or sheet" (`ref_upload_id`) or
  "Make one" (focuses Describe them). The choice shows as one card (cast, project image `who_ref`, photo, the last
  take's reference or made person) with Change / Remove / Replace.
- **A still from words:** words with no picture make a FLUX 3 klein still before the render, through the Images path:
  `POST /api/projects/{pid}/images/generate` with `{model: "flux3-klein", aspect: "2:3", resolution: "1K", count: 1,
  seed, refs: [], prompt}` where the prompt is `lib/still.ts` `stillPrompt(words)`, the same framing as
  `tasks/recast.py` `STILL` (keep the two in step). It starts on Enter, on leaving the field, from "Make one", or when
  Recast is pressed; the card shows "Making your character… · FLUX 3 klein · about 10 s", then the still with "Try
  another" (a new seed) and "Use a photo". The image job is quiet (`makeCharacterStill`: no "image ready" toast, and it
  leaves the queue bar when it ends; a dropped one is cancelled if it hasn't run). Recast then sends
  `who_ref: {kind: "image", id}` plus `describe`. Pressed before the still exists, Recast waits for it ("Making your
  character first…") and goes by itself. A take whose `who_ref` image has a `stillPrompt` comes back from "Try again
  with changes" as that still (new words make a new one). Words that @-mention a cast person use that person; no still.
- **Recast is enabled** when there is a person and a character (cast, picture, still, or words). Until then the line
  above the button says what is missing instead of the time: "Finding who's in the shot…", "Pick who to replace
  first.", "Uploading your picture…", "Add your character: a photo, a cast member or a few words." A Recast pressed
  before the still exists goes when it lands only if someone is still picked ("Not them?" and Clear call it off).
- **Words as an instruction:** "Describe them" goes through `lib/still.ts` `personWords` before FLUX 3 klein and
  `params.describe`: a lead-in such as "turn him into", "make her", "replace him with", "recast him as" comes off
  ("turn him into a knight in shining armour" → "a knight in shining armour"); the field keeps what was typed and says
  "Pickups reads this as “…”." The Prompt fold shows SCAIL-2's prompt as `tasks/recast.py` writes it
  (`scailPrompt`, keep the two in step; a cast pick with no words is named by its @-mention).
- **Words and pictures:** a photo, sheet or cast pick in place of a still made from the current words clears the
  words ("Your words went with the made still, so they're cleared. Undo"); words typed before any still stay, with
  "These words go with your picture. Clear them if they don't describe it." A wide picture (more than 1.3:1) is
  labelled "Your sheet · Pickups uses the front view". `useUploadSlot().clear()` also clears an earlier file's error,
  and making a still or picking a cast member clears it too.
- **While it runs:** the running panel shows the character going in (cast thumb, `who_ref`, `ref_upload_id` or the
  `reference_take` still) with "Not who you wanted? Stop now and try another." Words-only Recasts (FLUX inside the
  job) have no picture to show.
- **Long shots:** one line under the button, "Works on the first 4 s of your shot.", from `status.max_frames / fps`
  (4 s when the status has no `max_frames`), rounded to the nearest 0.5 s, shown only when the part left out is at
  least 0.25 s and 5 % of the shot (a 4.0 s shot against the 97-frame window gets no line). The "Long shot" notice is
  off for Recast (`FormShell lengthNote={false}`).
- **More options:** pick by words (`params.text`), the Option-click hint, Enhance for the description, and "Prompt"
  (the prompt receipt, collapsed). The viewer's chip leaves the keep-an-area hint out for Recast too ("Click the
  person to replace", "Got it · click to add more"); Remove still shows it. Option-click itself still works.
- **Backend follow-up:** a Recast whose person is one of these stills runs FLUX 3 klein in the image job, not in the
  task, so the take's models and licenses (made_still) don't list FLUX 3 klein yet; the image in Images does.
- **Home:** with no shots yet, a recipe card also opens the file picker; the dropped shot opens on the recipe's first
  form (for "Same scene, new lead", Recast with the person already found).
- **Mock:** `people()` as described above (cutouts drawn from the real frame, mattes from the mock's masks, cached
  per take); image jobs with a `stillPrompt` come out as the synthetic person still in a new outfit each time.

## Character sheets (Oct 17, `pickups/sheets.py`)

One person as FRONT | BACK | CLOSE-UP in one 16:9 picture (2000×1125, three equal panels, 8 px white gaps, a plain
light-grey backdrop). A character made from one front picture renders badly from behind; SCAIL-2 takes extra views
(back, close-up) beside its main reference, and Qwen-Image 2.1 reads a sheet well for previz look frames.

- **Endpoint:** `POST /api/projects/{pid}/cast/sheet {upload_id? | image_id? | cast_id? | describe?, name?, seed?}` →
  the queued job (`task: "sheet"`, `shot_id: null`, `project_id`, stages "Making the front view", "Making the back
  view", "Making the close-up", "Putting the sheet together", about 40 s). Exactly one source: an upload, a project
  image, a cast member (a first sheet, or "Redo" with a new seed), or a description alone. A new member needs a
  `name` that isn't taken. 400: no source, two sources, no name, a taken name, a bad seed, a non-person cast entry, a
  malformed upload id; 404: no project, upload, image or cast entry. For a `cast_id` the job's `params.name` is set to
  the member's name (queue labels).
- **The job:** words → FLUX 3 klein makes the front view (Recast's `STILL` template, 832×1280). A picture that isn't a
  clean full-length front view on a plain light backdrop (`reference.foreground`, one figure, head near the top, feet
  near the bottom, nothing cut off) is first turned into one by the frame editor (`graphs.image_edit`, so Setup's
  license choice applies: Qwen-Image 2.1, or the alternate editor for commercial-safe). Then the back view and the
  close-up from the front view; one seed for every panel. Pictures work at about 1 MP; a landscape picture goes in on a
  2:3 canvas of its own edge colour. When it's done the job carries `result_cast_id` and `event: project` carries
  `{id, images: [], cast: [cast_id]}`.
- **Cast payloads** (`GET /api/projects/{pid}`, `GET /api/cast`) carry `sheet` when a member has one:
  `{url, thumb_url (≈960 px), views: {front, back, face}, editor, models, licences, seed, made_from
  (upload | image | cast | words), made_front, created_at}`. The member's own `url` is the sheet too (Recast cuts out
  its front view, image models read the whole sheet); `thumb_url` is its front view, so chips show a face, not the
  back panel. On disk: `cast/<id>_sheet_<v>.jpg` (+ `_small.jpg`), `<id>_{front,back,face}_<v>.png`, and the picture it
  was built from (`sheet.source_file`; an upgraded member's own picture stays as that). "Redo" starts again from that
  source with a new seed; an older sheet's files are removed. "Add from another project" copies a sheet with its views.
- **Cast tab:** "Build character sheet" (primary, beside "Add from another project") opens a dialog: drop a picture
  (a photo, a selfie or an AI image) or describe them, plus a name; "About 40 s. They join this project's cast." While
  it runs, a card for the new member shows the stage, the time left and Stop. A member with a sheet shows it
  letterboxed in the 4:3 card (slate "PERSON · SHEET"); a click opens it big with "Made with … · seed N", the
  NON-COMMERCIAL tag when the editor's license needs it, and Redo. People without one get a quiet "Build a sheet" (not
  when their picture is already wide, likely a sheet); a sheet member gets "Redo sheet". A redo shows its progress over
  the card and the open sheet. When a sheet finishes away from the Cast tab, a toast says "<Name>'s sheet is ready ·
  View"; a failure says why.
- **Recast's "Your character":** under one dropped photo (not a wide sheet) or a still made from words, one line and
  one button: "Optional: a full sheet (front, back, close-up) makes them look right from behind. It joins your cast."
  · "Build a full sheet · about 40 s". It names the new member from the words ("a friendly green alien in a silver
  flight suit" → "Friendly green alien") or the file name (camera names such as IMG_1234 → "Character"), made unique
  in the cast. While it runs, the line shows the stage, the time left and Stop (a quiet job: no toast, it leaves the
  queue bar when it ends). Done, and still on the same picture: the new member becomes the character (`cast_id`), its
  card reading "Character sheet · front, back and a close-up".
- **Mock:** `buildSheet` validates like the server and builds a half-size sheet on a canvas from the dropped picture,
  the project image, the member's picture or (from words) the synthetic person: the picture as the front, mirrored in
  cool grey as the "back", its upper middle as the "close-up", and a MOCK slate.
- **Not here yet:** feeding `sheet.views` to SCAIL-2 as extra reference views (with matching masks) is separate work.

## Build

`cd web && npm run build` writes `src/pickups/web/dist` (index.html + hashed assets, no mock media). The backend
should serve `dist/index.html` at `/` and `dist/assets/*`, `dist/favicon.svg` statically.

Note: the repo-root `.gitignore` ignores `dist/` (and `*.mp4`). To commit the built UI as the spec intends, add
`!src/pickups/web/dist/` to the root `.gitignore`. `web/.gitignore` already re-includes the synthetic mock clips in
`web/public/mock/`.
