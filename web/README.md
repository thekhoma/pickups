# Pickups web UI

Vite + React 18 + TypeScript + Tailwind CSS 3. Spec: [`../docs/ui-spec.md`](../docs/ui-spec.md); look and feel
("the cutting room"): warm black, ivory type, Instrument Serif / Instrument Sans / Geist Mono, tally red on dots
only, the wipe line as the signature.
Notes for the backend (contract details the UI relies on): [`../docs/ui-notes.md`](../docs/ui-notes.md).

## Commands

```bash
cd web
npm install

npm run dev         # http://localhost:5173, proxies /api and /media to the backend on http://127.0.0.1:7788
npm run dev:mock    # same, but always on the mock API (no backend, no GPU)
npm run build       # type-checks, then builds into ../src/pickups/web/dist (served by the backend)
npm run typecheck
npm run mock-media  # rebuilds public/mock from a Pickups library's real takes (needs ffmpeg and that library)
```

`npm run dev` falls back to the mock by itself when the backend isn't answering `/api/status`
(the top bar then shows **Demo data**). You can also force it with `VITE_MOCK=1` or `?mock=1` in the URL.
The mock ships in every build as a lazy chunk, so `?mock=1`, `?mock=fresh` and `?tour=1` also work from the
backend's own page (http://127.0.0.1:7788/?mock=1). Mock sessions remember their project under their own key, so
they never change the real workspace. The demo's media is real: shots, takes, receipt overlays, the people finder's
cutouts and the cast's character sheets, cut from takes Pickups made (`scripts/make_mock_media.py`, about 17 MB).

## Mock scenarios

| URL | Shows |
| --- | --- |
| `/` | Ready: Studio · RTX 4090 · 24 GB; four projects: "Highlands & Alps" (a castle on a loch, a castle walk recast with SCAIL-2 and snowed in, a trail emptied with real pixels, an alpine ride turned to autumn), "Coast road" (a snowy road that keeps the car as shot, a 4K HDR finish), "Stunt previz" (Blender previz shots rendered into finished shots, the stunt driver's character sheet and the set) and "Fun cast" (seven FLUX 3 klein characters with sheets, recast into two walks across a glen) |
| `/?mock=fresh` | First run: no shots and no example gallery yet, so Home offers "Get the examples" (then shows them) |
| `/?mock=setup` | Missing models (amber "Setup needed", "Remove needs 2 models (1.2 GB)", gated license link) |
| `/?mock=offline` | No ComfyUI ("Connect a GPU"); Connect in Setup brings it back |
| `/?mock=fail` | Remove fails while blending ("Something went wrong while blending…") |
| `/?mock=stall` | The first video job stalls at 40 % (calm STALLED note in the queue and the running view) |
| `/?mock=busy` | ComfyUI is busy with someone else's job: the first job waits behind it, then runs |
| `/?mock=refuse` | Ideogram 4 refuses every image (its grey "blocked" card, twice): a red error, nothing saved, the seed lock lets go |
| `/?director=none` | Setup · Prompt director finds no local LLM server |
| `/?tour=1` | The demo autopilot (docs/demo-autopilot.md); `&loop=1` loops, `&pace=1.3` plays faster, Esc stops |

Mock jobs run 20× faster than the ETAs they report, so a "5 min" job finishes in about 15 s, and return the real take
the shot has for that task (with its receipt). Recast's people finder answers with the real cutout and matte on the glen,
castle walk and trail shots; the Glen walker is 10.8 s, so Recast's 4 s bracket shows on its timeline. Uploaded files
play from the browser (MP4 / H.264 works best in demo mode).

## Keyboard

Space play/pause · ←/→ frame · Shift+←/→ 1 s · B / A / R before / after / receipt · W wipe · Esc back · ⌘O open file ·
⌘Z undo last click while selecting.

## Layout of the code

```
src/
  api.ts          typed client for the contract (+ SSE on /api/events) and the mock/live switch
  mock.ts         mock API: status, samples, uploads, click-to-select masks, jobs with stages/ETA, versions
  types.ts        contract types
  store.ts        tiny global store (useSyncExternalStore)
  actions.ts      everything that calls the API and changes state
  lib/            formatting (timecode, ETAs, receipts), player sync, hooks
  components/
    TopBar         wordmark, Library, status pill
    Home           hero (first example + drop zone + path field), examples, recipes, your shots
    WipeMedia      before/after loops with the wipe line (hero sweep, card hover scrub, idle sweeps)
    DownloadControl downloads grouped by purpose; the pill repeats your last choice
    ShotView       viewer column + right rail
    Viewer         stacked before/after videos, wipe handle, receipt overlay, click-to-select + mask
    Transport      play / frame step / timecode / view toggle, filmstrip scrubber, version strip
    RightRail      tiles → task form → running → result
    Forms          Remove, Recast, Render previz, Reshoot, Finish
    QueueBar       running and finished jobs (expandable)
    SetupPanel     connect, machine, models, benchmarks
    ProjectPages   all projects; a project's Shots · Cast · Deliveries
    ProjectSwitcher the [Project ▾] menu
    Mentions       @-mention fields (cast chips) and the prompt receipt
```

The production build includes `public/mock` (about 3 MB of synthetic clips) so demo mode and the tour work from the
backend.

## Recording the tour

    npm run tour -- "http://127.0.0.1:7788/?tour=1" ../pickups_tour.mp4

`scripts/record_tour.mjs` drives headless Chromium (Playwright 1.63.0, which uses the cached Chromium build 1243),
waits for the page's `tour:done` event and writes an H.264 MP4 plus a 1280-wide loop. A full run takes about 90 s.
