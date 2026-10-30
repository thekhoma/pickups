# Demo autopilot (the product demo)

Inspired by Frame.io's homepage demo: the real UI plays itself. It's used three ways: looping on the README/launch page,
as the spine of the ≤2-minute contest video, and as a "Take the tour" link on first run.

## How it works

- `?tour=1` starts the autopilot in **mock mode** (no GPU, no network): the app's own mock data, whose media is cut from
  real takes into `web/public/mock` (only footage the user has cleared; `web/scripts/make_mock_media.py`).
- A scripted timeline of steps; each step = wait for a selector, move a drawn cursor (ivory ring, 160 ms ease) to it,
  click/type, hold. Typing is human-paced (35–60 ms per character, small random jitter). Renders are faked by the mock
  queue with an accelerated countdown (a 3-minute render plays in 4 s, timecode visibly ticking).
- One mono caption per beat (the slate style), e.g. `01 · DROP A SHOT`. On a shot it sits inside the picture
  (bottom-left, or bottom-right when the cursor's target is there), so it never covers the transport, the take strip
  or the rail's buttons, and it waits while the shot loads; elsewhere it sits bottom-left of the page (top-right under
  the bar when the target is there). It follows layout changes. No voice-over in the UI; the video adds it.
- `?tour=1&loop=1` loops; `Esc` ends the tour and leaves the user in a clean mock workspace.

## The beats (about 60 s)

1. **Home**: the hero wipe (a hiker erased from the Tre Cime trail, `0% AI · REAL PIXELS`); cursor glides over the
   examples grid, one card scrubs.
2. **Project**: switcher → "Stunt previz" → Shots grid with take stacks → Cast tab (Stunt driver with his sheet, Hood
   view, The run, Video village).
3. **Consistency**: open the `K03 hood push` previz → Render previz → type `@Stunt driver sits behind the wheel in
   @Hood view as the camera pushes in; …` Chips appear; the **prompt receipt** fills in (Qwen image numbers with
   thumbnails, LTX description).
4. **Prompt Director**: Enhance → "Directing…" → the director's version slides in → Use.
5. **Render**: the tally dot lights, red progress line and countdown → the take lands: wipe from previz to final.
6. **Approve & deliver**: ⇧A → Deliveries → Export timeline (XML) → the file list (ProRes 422 HQ, H.264, H.265) with
   mono sizes.
7. **Image**: Image page → FLUX 3 · 16:9 · 2K → prompt → four images → "Add to cast" on one.
8. **Change weather**: switcher → "Highlands & Alps" → Castle on the loch → Snow → result wipe.
9. End card: wordmark + "Power your creativity, locally." + "Open source · runs on ComfyUI".

## Recording

`web/scripts/record_tour.mjs` (Playwright, Chromium headless, 1920×1080, `recordVideo`) opens `?tour=1`, waits for the
`tour:done` event, and converts the WebM to H.264 MP4 (and a 1280-wide loop for the README). Deterministic: fixed
seeds for typing jitter, fonts preloaded, reduced motion off.

## Implementation notes (web UI, Oct 17)

- Code: `web/src/tour/autopilot.ts` (a lazy chunk loaded only with `?tour`); the mock's tour mode lives in
  `web/src/mock.ts` (`TOUR`). Repo hygiene: the tour uses generic demo names, a **Stunt previz** project with the
  **Stunt driver** (an old stunt driver with a white walrus moustache and a cream helmet with navy stars) and **Hood
  view**; the previz shot is `K03 hood push`, the weather shot `Castle on the loch`.
- Copy: if `/examples/tour.json` exists (the backend serves `~/.pickups/examples`), the tour (and the mock, during the
  tour only) take the project name, cast (name, kind, line, image relative to /examples/), previz shot name, typed
  prompt, the director's answer, the image prompt and the image's cast name/kind from it (`web/src/lib/tourCopy.ts`;
  optional keys `weather_shot`, `image_cast_kind`, default "person"). Without it the generic built-ins play, which
  match the mock's own media; `?mock=1` always shows the built-ins. `@Name` mentions in the
  prompts are picked from the @-list as they're typed.
- Media: the mock's own (`web/public/mock`, real takes): Home's gallery, the K03 previz and its render, the castle
  and its winter dusk, and the four images are frames of the stunt renders. `/examples/` is no longer read for media.
- Beats as built: hero (+ a card scrub) → switcher → the demo project's shots → Cast → previz shot → Render previz →
  type with @-mentions (chips + receipt) → Enhance ("via LM Studio") → Use → Render (3.2 s, countdown from the real
  estimate) → wipe sweep while the take plays → ⇧A → Deliveries → Export timeline (XML; no file is saved in tour
  mode) → Share 9:16 (composed in the browser from the take itself, slate burned in) → Change weather: switch to
  Highlands & Alps, open Castle on the loch → Snow → wipe → back to the demo
  project → Image: FLUX 3 · 16:9 · 2K → four images → Add to cast → end card.
- Parameters: `?tour=1&loop=1` loops; `&pace=1.3` plays faster (0.25–4); `?tour=home|project|mention|weather|image` starts at
  that beat (for checking one; later beats depend on earlier ones). Esc stops and leaves a normal mock workspace.
  On the end card a click or Esc plays the whole tour again; its hint fades in after `tour:done` (not under webdriver).
  The "Demo data" badge is hidden during the tour. A beat that can't find its control is skipped (logged as
  `window.__tourSkipped`), so a recording always reaches the end card and `tour:done`.
- The cursor glides at roughly 0.85 ms per pixel (240–820 ms, ease in-out); the 160 ms ease is the click press.
- Recorded headless with `npm run tour`: about 90 s including page load (Playwright 1.63.0 uses the cached
  Chromium 1243).
