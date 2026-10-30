// Run 9, in a real browser (Playwright) on the mock API (`npm run dev:mock`, or any build with ?mock=1):
//   RUN9_URL=http://127.0.0.1:5199 node tests/previz/run9_browser.mjs --out <dir> [--width 1440 --height 900] [--only flows,recast,...]
// Each part checks one Run 9 item on screen and saves <dir>/<part>/<size>_<nn>_<what>.png.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = opt('--url', process.env.RUN9_URL || 'http://localhost:5199');
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const ONLY = (opt('--only', '') || '').split(',').filter(Boolean);
const SIZE = `${W}x${H}`;
let fails = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) fails++;
};

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log(`PAGEERROR ${e}`));
let part = 'misc';
let n = 0;
const shoot = async (what) => {
  n += 1;
  if (!OUT) return;
  const dir = join(OUT, part);
  mkdirSync(dir, { recursive: true });
  await page.mouse.move(1, 1);
  await page.waitForTimeout(350);
  await page.screenshot({ path: join(dir, `${SIZE}_${String(n).padStart(2, '0')}_${what}.png`) });
};
const until = async (fn, ms = 20000, step = 200) => {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (await fn().catch(() => false)) return true;
    await page.waitForTimeout(step);
  }
  return false;
};
const api = (path, init) => page.evaluate(async ([p, i]) => (await fetch(p, i)).json(), [path, init]);
const rail = () => page.locator('aside[aria-label="Tasks"]');
const go = async (hash = '', query = 'mock=1') => {
  await page.goto(`${BASE}/?${query}#${hash}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
};
// The mock's shots by project (web/README.md, Mock scenarios): opened through the UI, as a user would.
const PROJECT_OF = [
  [/trail/i, 'Highlands & Alps'],
  [/castle walk/i, 'Fun cast'],
  [/glen/i, 'Fun cast'],
  [/coast|snowy road/i, 'Coast road'],
];
const openShot = async (match, query = 'mock=1') => {
  await go('/projects', query);
  const project = PROJECT_OF.find(([re]) => re.test(match.source))?.[1] ?? 'Highlands & Alps';
  await page.getByText(project, { exact: true }).first().click();
  await page.waitForTimeout(1000);
  const card = page.locator('main').getByText(match).first();
  await card.click({ timeout: 20000 });
  await rail().waitFor({ state: 'visible', timeout: 20000 });
  // A shot with work under way (or a draft) opens on it (Run 9 item 7): back to the task list.
  for (let i = 0; i < 3; i++) {
    const back = rail().getByRole('button', { name: 'Back', exact: true }).first();
    if (!(await back.isVisible().catch(() => false))) break;
    await back.click();
    await page.waitForTimeout(400);
  }
};
const openTile = async (name) => {
  const tile = rail().getByRole('button', { name: new RegExp(`^${name}`) }).first();
  await tile.click({ timeout: 20000 });
  await page.waitForTimeout(600);
};
const section = (name) => {
  part = name;
  n = 0;
  console.log(`--- ${name} (${SIZE})`);
};
const want = (name) => !ONLY.length || ONLY.includes(name);

// --- 1 step flows by default: Remove, Relight, Finish ------------------------------------------------------------------
if (want('flows')) {
  section('01_flows');
  await page.goto(`${BASE}/?mock=1`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    for (const k of Object.keys(localStorage)) if (k.startsWith('pickups.flow.') || k === 'pickups.drafts.v1') localStorage.removeItem(k);
  });
  await openShot(/trail/i);
  for (const [tile, first] of [
    ['Remove', 'Who'],
    ['Relight', 'Light'],
    ['Finish', 'Finish'],
  ]) {
    await openTile(tile);
    const steps = rail().getByRole('navigation', { name: 'Steps' });
    check(await until(() => steps.isVisible()), `${tile}: opens as steps with no flag set`);
    check(await until(async () => (await steps.innerText()).includes(`1 ${first}`)), `${tile}: the first step is "${first}"`);
    await shoot(tile.toLowerCase().replace(/\W+/g, '_'));
    await rail().getByRole('button', { name: 'Back', exact: true }).first().click();
    await page.waitForTimeout(500);
  }
}

// --- 2 Replace a person: on by default; Make 4 options and the sheet run with Cancel; the new character is picked and
// the render starts by itself with Cancel ---------------------------------------------------------------------------------
if (want('recast')) {
  section('02_recast');
  await page.goto(`${BASE}/?mock=1`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    for (const k of Object.keys(localStorage)) if (k.startsWith('pickups.flow.') || k === 'pickups.drafts.v1') localStorage.removeItem(k);
  });
  await openShot(/castle walk/i);
  await openTile('Replace a person');
  const steps = rail().getByRole('navigation', { name: 'Steps' });
  check(await until(() => steps.isVisible()), 'recast: opens as steps with no flag set');
  const usePerson = rail().getByRole('button', { name: /^Use Person \d|^Use this pick/ }).last();
  check(await until(() => usePerson.isEnabled(), 30000), 'who: a person is picked for you');
  await shoot('who');
  await usePerson.click();
  await page.waitForTimeout(500);
  const newTab = rail().getByRole('radio', { name: 'New', exact: true });
  if (await newTab.count()) await newTab.click();
  await rail().getByRole('textbox', { name: 'Who are they?' }).fill('a tall knight in silver armour');
  await rail().getByRole('button', { name: 'Make 4 options', exact: true }).click();
  const strip = rail().getByTestId('run-strip').first();
  check(await until(() => strip.isVisible(), 10000), 'make 4 options: a run strip while it runs');
  check(await strip.getByRole('button', { name: 'Cancel' }).isVisible(), 'make 4 options: Cancel');
  await shoot('options_running');
  await strip.getByRole('button', { name: 'Cancel' }).click();
  check(await until(async () => (await rail().getByRole('button', { name: 'Make 4 options', exact: true }).isEnabled()), 10000), 'make 4 options: Cancel stops it, and Make 4 options is back');
  await rail().getByRole('button', { name: 'Make 4 options', exact: true }).click();
  const opt1 = rail().getByRole('radio', { name: 'Option 1' });
  check(await until(() => opt1.isVisible(), 60000), 'make 4 options: the options come back');
  await opt1.click();
  await page.waitForTimeout(300);
  await shoot('option_picked');
  await rail().getByRole('button', { name: 'Make sheet', exact: true }).click();
  const sheetStrip = rail().getByTestId('run-strip').filter({ hasText: /making the sheet|queued/i }).first();
  check(await until(() => sheetStrip.isVisible(), 15000), 'make sheet: the new character is picked and their sheet shows a run strip with Cancel');
  check(await until(async () => (await rail().getByRole('radio', { checked: true }).count()) > 0, 5000), 'make sheet: the new character is selected in the cast');
  await shoot('sheet_making');
  const rendering = steps.locator('button[aria-current="step"]');
  const renderStrip = rail().getByTestId('run-strip').first();
  let sawStrip = false;
  check(
    await until(async () => {
      if (await renderStrip.getByRole('button', { name: 'Cancel' }).isVisible().catch(() => false)) {
        if (!sawStrip) await shoot('render');
        sawStrip = true;
      }
      return /Render|Review/.test(await rendering.innerText());
    }, 90000, 100),
    'sheet ready: the render starts by itself',
  );
  if (/Render/.test(await rendering.innerText()) && (await renderStrip.isVisible())) {
    sawStrip = true;
    await shoot('render');
  }
  check(await until(async () => /Review/.test(await rendering.innerText()), 90000), 'render: done, on to Review');
  await shoot('review');
  console.log(`note: render run strip with Cancel seen on screen: ${sawStrip}`);
}

// --- Render with Cancel on a longer task (Finish) ------------------------------------------------------------------------
if (want('render')) {
  section('02_render');
  await openShot(/trail/i);
  await openTile('Finish');
  await rail().getByRole('button', { name: /^Finish/ }).last().click();
  const strip = rail().getByTestId('run-strip').first();
  check(await until(() => strip.isVisible(), 15000), 'render: a run strip (elapsed, usually, Cancel)');
  await shoot('render_strip');
  await strip.getByRole('button', { name: 'Cancel' }).click();
  check(await until(async () => (await rail().getByText('Cancelled').count()) > 0, 15000), 'render: one press of Cancel stops it');
  await shoot('render_cancelled');
}

// --- 3 a refused image (Ideogram's grey card, ?mock=refuse): an error, nothing saved, the seed let go ------------------
if (want('refuse')) {
  section('03_refused');
  await go('/create/image', 'mock=refuse');
  const panel = page.locator('aside[aria-label="Make an image"]');
  await panel.waitFor({ state: 'visible', timeout: 20000 });
  await panel.getByRole('textbox', { name: 'Prompt' }).fill('A vintage travel poster of a seaside town, bold lettering that says "FABLE WALK", flat colours');
  const count = async () => {
    await page.waitForTimeout(800);
    return page.locator('section[aria-label="Results"] img').count();
  };
  // Run once on the default model so there is a seed (Run 10: no lock before one), lock it, then run Ideogram 4 into
  // the refusal.
  const lock = panel.getByRole('button', { name: /^Lock seed/ });
  await panel.getByRole('button', { name: 'Generate', exact: true }).click();
  check(await until(() => lock.isVisible(), 40000), 'refused: a first run gives a seed to lock');
  await lock.click();
  await page.waitForTimeout(300);
  await panel.getByRole('button', { name: 'Change', exact: true }).click();
  await panel.getByRole('radio', { name: /^Ideogram 4/ }).click();
  await until(async () => !(await page.getByTestId('run-strip').first().isVisible().catch(() => false)), 60000); // the first run's pictures are in
  const before = await count();
  await panel.getByRole('button', { name: 'Generate', exact: true }).click();
  const strip = page.getByTestId('run-strip').first();
  check(await until(() => strip.isVisible(), 10000), 'refused: it runs with the caption step');
  const notice = page.getByText("Ideogram's built-in filter refused this prompt. Try other words.");
  check(await until(() => notice.isVisible(), 40000), 'refused: a red notice says so');
  check((await lock.count()) === 0, 'refused: no seed is kept, so no lock shows');
  const after = await count();
  check(before === after, `refused: nothing saved (${before} → ${after} images)`);
  await shoot('refused');
}

// --- 4 one Cancel everywhere: the running panel and the queue stop a job with one press ---------------------------------
if (want('cancel')) {
  section('04_cancel');
  await openShot(/trail/i);
  await openTile('Change the weather');
  await rail().getByRole('button', { name: 'Snow', exact: true }).click().catch(() => {});
  await rail().getByRole('button', { name: 'Change the weather', exact: true }).last().click();
  const cancel = rail().getByRole('button', { name: 'Cancel', exact: true });
  check(await until(() => cancel.isVisible(), 15000), 'running panel: Cancel');
  check((await rail().getByText('Stop this job?').count()) === 0, 'running panel: no "Stop this job?" step');
  await shoot('running');
  await cancel.click();
  check(await until(async () => (await rail().getByText('Cancelled').count()) > 0, 15000), 'running panel: one press stops it (Cancelled)');
  await shoot('cancelled');
  // The queue: start another, cancel it from the queue menu with one press.
  await rail().getByRole('button', { name: 'Try again', exact: true }).first().click();
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: /queue|00:/i }).first().click();
  const qCancel = page.locator('li').getByRole('button', { name: 'Cancel', exact: true }).first();
  check(await until(() => qCancel.isVisible(), 10000), 'queue: Cancel on the running job');
  await shoot('queue_running');
  await qCancel.click();
  check(await until(async () => (await page.getByText('Keep going').count()) === 0 && !(await qCancel.isVisible().catch(() => false)), 10000), 'queue: one press, no "Keep going / Stop"');
  await shoot('queue_cancelled');
}

// --- 5 Finish: the tile is "Finish"; header, stages and label follow the choice --------------------------------------------
if (want('finish')) {
  section('05_finish');
  await openShot(/^Coastal drive$/);
  check((await rail().getByRole('button', { name: /^Finish\b/ }).first().innerText()).split('\n')[0].trim() === 'Finish', 'tile: "Finish"');
  await shoot('tiles');
  await openTile('Finish');
  const title = async () => (await rail().locator('h2, h1').first().innerText()).trim();
  await rail().getByRole('radio', { name: '4K', exact: true }).click();
  check(await until(async () => (await title()) === 'Finish to 4K'), `header: "Finish to 4K" with 4K picked (${await title()})`);
  await shoot('choice_4k');
  await rail().getByRole('radio', { name: '4K HDR', exact: true }).click();
  check(await until(async () => (await title()) === 'Finish to 4K HDR'), 'header: "Finish to 4K HDR" with 4K HDR picked');
  await rail().getByRole('radio', { name: '4K', exact: true }).click();
  await rail().getByRole('button', { name: 'Finish', exact: true }).last().click();
  const stages = rail().getByRole('list', { name: 'Stages' });
  check(await until(() => stages.isVisible(), 15000), 'render: the stages show');
  const labels = await stages.innerText();
  check(/Upscaling to 4K/.test(labels) && !/highlights|HDR/.test(labels), `render: 4K's stages only (${labels.replace(/\n+/g, ' / ')})`);
  check((await title()) === 'Finish to 4K', 'render: the header still says Finish to 4K');
  await shoot('render_4k');
}

// --- 6 never a prompt from another shot: the weather and previz fields open empty, with plain placeholders ---------------
if (want('prefill')) {
  section('06_no_prefill');
  await openShot(/^Coastal drive$/);
  await openTile('Change the weather');
  await rail().getByRole('button', { name: 'Advanced' }).last().click(); // item 8: the field sits under Advanced
  await page.waitForTimeout(300);
  const scene = rail().getByRole('textbox', { name: /What's in the shot/ }).first();
  const sceneBox = (await scene.count()) ? scene : rail().locator('textarea').first();
  check((await sceneBox.inputValue()) === '', `weather: "What's in the shot" opens empty on a shot with no words of its own`);
  check(!/raincoat|castle|Maya/i.test((await sceneBox.getAttribute('placeholder')) || ''), `weather: no other scene in the placeholder (${await sceneBox.getAttribute('placeholder')})`);
  await shoot('weather');
  await rail().getByRole('button', { name: 'Back', exact: true }).first().click();
  await page.waitForTimeout(400);
  await openTile('Previz to final');
  const boxes = rail().locator('textarea');
  const texts = await boxes.evaluateAll((els) => els.map((e) => ({ v: e.value, p: e.placeholder })));
  check(texts.every((t) => !/Maya|raincoat|castle|cobbled/i.test(t.v + t.p)), `previz to final: no other shot's words (${texts.map((t) => t.p).join(' | ')})`);
  await shoot('previz_to_final');
}

// --- 7 a job keeps going when its panel closes; its take chip (and its tile) bring the live progress back ----------------
if (want('reopen')) {
  section('07_reopen');
  await openShot(/^Coastal drive$/);
  const back = () => rail().getByRole('button', { name: 'Back', exact: true }).first().click();
  // Each task's own chip: an earlier part's job still running on this shot has a chip too (Run 10: the full run clicked it).
  const chipOf = (task) => page.getByRole('button', { name: new RegExp(`, ${task}, .*: show progress$`) }).first();
  let chip = chipOf('Change the weather');
  // Previz to final's and the weather's own panel (not a step flow).
  await openTile('Change the weather');
  await rail().getByRole('button', { name: 'Change the weather', exact: true }).last().click();
  await rail().getByText('Run in background').waitFor({ timeout: 10000 });
  await back();
  check(await until(() => chip.isVisible(), 5000), 'weather: closed mid-job, its take chip is in the strip');
  await shoot('closed');
  await chip.click();
  check(await until(() => rail().getByRole('button', { name: 'Run in background' }).isVisible(), 5000), 'weather: the chip reopens its live progress');
  await shoot('chip_reopens');
  await back();
  await openTile('Change the weather');
  check(await until(() => rail().getByRole('button', { name: 'Run in background' }).isVisible(), 5000), 'weather: its tile reopens the progress, not an empty form');
  await rail().getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.waitForTimeout(800);
  await back();
  // A step flow: the chip opens its own Render step.
  chip = chipOf('Relight');
  await openTile('Relight');
  await rail().getByRole('button', { name: 'Relight', exact: true }).last().click();
  await rail().getByTestId('run-strip').waitFor({ timeout: 10000 });
  await back();
  check(await until(() => chip.isVisible(), 5000), 'relight: closed mid-job, its take chip is in the strip');
  await chip.click();
  const current = rail().getByRole('navigation', { name: 'Steps' }).locator('button[aria-current="step"]');
  check(await until(async () => /Render/.test(await current.innerText()) && (await rail().getByTestId('run-strip').isVisible()), 5000), 'relight: the chip reopens its Render step, live');
  await shoot('flow_render_reopened');
  await rail().getByTestId('run-strip').getByRole('button', { name: 'Cancel' }).click();
}

// --- 8 receipts, Prompt tips, Enhance and footnotes wait under Advanced; results offer two ComfyUI actions ---------------
if (want('fold')) {
  section('08_advanced');
  const shown = async (re, root = page) => {
    for (const el of await root.getByText(re).all()) if (await el.isVisible()) return true;
    return false;
  };
  await openShot(/^Coastal drive$/);
  await openTile('Change the weather');
  check(!(await shown(/What each model gets|Prompt tips/i, rail())) && !(await rail().getByRole('button', { name: 'Enhance' }).isVisible().catch(() => false)), 'weather: no receipt, tips or Enhance on the first screen');
  check((await rail().locator('textarea:visible').count()) === 0, 'weather: no text box on the first screen (What\'s in the shot is under Advanced)');
  check(!(await shown(/in the bracket/i, rail())), 'weather: no footnote under the button');
  await shoot('weather');
  await rail().getByRole('button', { name: 'Advanced' }).last().click();
  check(await until(() => rail().getByText("What's in the shot").isVisible(), 3000), "weather: What's in the shot is under Advanced");
  await shoot('weather_advanced');
  await rail().getByRole('button', { name: 'Back', exact: true }).first().click();
  await page.waitForTimeout(400);
  await openTile('Previz to final');
  check(!(await shown(/What each model gets|Prompt tips/i, rail())) && !(await rail().getByRole('button', { name: 'Enhance' }).isVisible().catch(() => false)), 'previz to final: no receipt, tips or Enhance on the first screen');
  check((await rail().locator('textarea:visible').count()) === 1, 'previz to final: one text box (Look); Motion is under Advanced');
  await shoot('previz_to_final');
  await rail().getByRole('button', { name: 'Back', exact: true }).first().click();
  await page.waitForTimeout(400);
  await openTile('Relight');
  check(!(await shown(/24 GB/i, rail())), 'relight: no GPU warning on a 24 GB card');
  await shoot('relight');
  // Results: the take's ComfyUI workflows are two actions, not a row per workflow.
  await rail().getByRole('button', { name: 'Back', exact: true }).first().click();
  await page.getByRole('tab').filter({ hasText: /Take 02/ }).first().click();
  await page.waitForTimeout(800);
  const wf = rail().getByRole('region', { name: 'ComfyUI workflows' });
  if (await wf.count()) {
    check((await wf.getByRole('button').count()) + (await wf.getByRole('link').count()) <= 2, 'result: at most two ComfyUI actions');
    check(await wf.getByText('Download workflows').isVisible(), 'result: Download workflows');
  } else check(false, 'result: the take has a ComfyUI section');
  await shoot('result');
  // The Image page: Enhance and what each model gets are under Advanced.
  await go('/create/image');
  const imgPanel = page.locator('aside[aria-label="Make an image"]');
  await imgPanel.waitFor({ timeout: 15000 });
  check(!(await shown(/What each model gets/i, imgPanel)) && !(await imgPanel.getByRole('button', { name: 'Enhance' }).isVisible().catch(() => false)), 'image: no receipt or Enhance until Advanced');
  await shoot('image');
}

// --- 9 Home: no presets or recipes rows, no file footnote; New image first when there are no shots -------------------
if (want('home')) {
  section('09_home');
  const checked = () => page.locator('[role="radio"][aria-checked="true"]').filter({ hasText: /Edit clip|New image|Text to video|Image to video|New character|New previz/ }).first().innerText();
  await go('/', 'mock=fresh');
  await page.waitForTimeout(800);
  check(/New image/.test(await checked()), `no shots: the composer opens on New image (${(await checked()).trim()})`);
  check((await page.getByText(/^Presets$|^Recipes$/).count()) === 0, 'no shots: no Presets or Recipes rows');
  await shoot('fresh');
  await go('/', 'mock=1');
  await page.waitForTimeout(800);
  check(/Edit clip/.test(await checked()), `with shots: the composer opens on Edit clip (${(await checked()).trim()})`);
  check((await page.getByText(/^Presets$|^Recipes$/).count()) === 0, 'with shots: no Presets or Recipes rows');
  check((await page.getByRole('button', { name: 'What clips work' }).count()) === 0, 'with shots: no file footnote by the drop zone');
  await shoot('with_shots');
}

// --- 10 Assets takes pictures: Import (and the empty state's chooser) accept clips and pictures --------------------------
if (want('assets')) {
  section('10_assets');
  await go('/assets', 'mock=fresh');
  await page.getByRole('radio', { name: 'Imported' }).first().click().catch(() => {});
  await page.waitForTimeout(500);
  const input = page.locator('input[type=file][aria-label="Import clips or pictures"]');
  const accept = (await input.getAttribute('accept')) || '';
  check(/image\//.test(accept) && /video\//.test(accept), `the chooser takes clips and pictures (${accept})`);
  check(await page.getByRole('button', { name: 'Import', exact: true }).isVisible(), 'an Import button in the header');
  await shoot('empty');
  const picture = fileURLToPath(new URL('../../public/mock/look_frame.jpg', import.meta.url));
  await input.setInputFiles(picture);
  check(await until(async () => (await page.getByText(/look frame/i).count()) > 0, 10000), 'a picture chosen there shows in Imported');
  await shoot('picture_imported');
}

// --- 11 Previz: clicks set the camera path (with feedback), its points show in Top, missed words are said ----------------
if (want('previz')) {
  section('11_previz');
  await go('/previz');
  const box = page.getByRole('textbox', { name: 'Describe the shot' }).first();
  await box.waitFor({ timeout: 30000 });
  await box.fill('A man in a long coat stands on a pier at dusk. The camera slowly pushes in.');
  await box.press('Enter');
  const canvas = page.locator('canvas').first();
  await canvas.waitFor({ timeout: 30000 });
  await page.waitForTimeout(4000);
  const missed = page.getByTestId('previz-missed');
  const missedText = (await missed.count()) ? await missed.innerText() : '';
  check(!(await page.locator('.line-through').count()), 'missed words: never only crossed out');
  console.log(`note: missed-words line: "${missedText}"`);
  await shoot('built');
  await page.getByRole('button', { name: 'Set start and end' }).click();
  const status = page.getByTestId('path-status');
  check(await until(async () => /set the start/i.test(await status.innerText()), 5000), 'path: "Click the shot to set the start"');
  const b = await canvas.boundingBox();
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
  check(await until(async () => /Start set/i.test(await status.innerText()), 5000), 'path: a click sets the start and says so ("Start set · click to set the end")');
  await shoot('start_set');
  await page.mouse.click(b.x + b.width / 2 + 10, b.y + b.height / 2);
  check(await until(async () => /Path set/i.test(await status.innerText().catch(() => '')), 5000), 'path: a second click sets the end ("Path set")');
  await shoot('path_set');
  check(await page.getByRole('button', { name: 'Reset', exact: true }).isVisible(), 'path: the custom path is on (Reset shows)');
  await page.getByRole('button', { name: 'Top', exact: true }).click();
  await page.waitForTimeout(1500);
  const top = await page.evaluate(() => window.__previzTop || null);
  check(!!top && top.points.length >= 2, `top: the path's points are drawn (${top ? top.points.length : 0})`);
  if (top && top.points.length >= 2) {
    const p0 = top.points[top.points.length - 1];
    const cb = await canvas.boundingBox();
    await page.mouse.move(cb.x + p0.x, cb.y + p0.y);
    await page.mouse.down();
    await page.mouse.move(cb.x + p0.x + 60, cb.y + p0.y + 30, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(800);
    const after = await page.evaluate(() => window.__previzTop || null);
    const q = after.points[after.points.length - 1];
    check(Math.hypot(q.x - p0.x, q.y - p0.y) > 20, `top: dragging the end point moves it (${p0.x},${p0.y} → ${q.x},${q.y})`);
  }
  await shoot('top_points');
}

// --- 12 one name per task: tile, take chip, panel header and queue say the same; made shots keep their whole name ------
if (want('names')) {
  section('12_names');
  await openShot(/trail/i);
  const tiles = await rail().getByRole('button').evaluateAll((els) => els.map((e) => (e.querySelector('span span span')?.textContent || '').trim()).filter(Boolean));
  console.log(`note: tiles: ${tiles.join(' | ')}`);
  check(tiles.includes('Relight') && tiles.includes('Remove a person') && tiles.includes('Finish'), 'tiles: Remove a person, Relight, Finish');
  const chip = page.getByRole('tab').filter({ hasText: /Take 02/ }).first();
  const chipText = (await chip.innerText()).replace(/\s+/g, ' ');
  check(/Remove a person/.test(chipText), `take chip: "Remove a person" (${chipText})`);
  await chip.click();
  await page.waitForTimeout(700);
  const header = (await rail().locator('h2').first().innerText()).trim();
  check(/^Remove a person/.test(header), `result header: "Remove a person" (${header})`);
  await shoot('remove_take');
  await openShot(/^Coastal drive$/);
  const chip2 = (await page.getByRole('tab').filter({ hasText: /Take 02/ }).first().innerText()).replace(/\s+/g, ' ');
  check(/Finish 4K HDR/.test(chip2), `finish take chip keeps its detail (${chip2})`);
  await shoot('finish_take');
  // A made shot keeps its whole first sentence as its name.
  await go('/create', 'mock=1');
  await page.getByRole('radio', { name: 'Text to video' }).first().click();
  const sentence = 'A golden retriever runs along a beach at sunset kicking up spray';
  await page.getByRole('textbox').filter({ hasNot: page.locator('[disabled]') }).last().fill(`${sentence}. Slow push in.`);
  await page.getByRole('button', { name: /^Generate/ }).last().click();
  check(await until(async () => (await page.getByText(new RegExp(`ready · ${sentence}$`)).count()) > 0, 60000), 'made shot: named by its whole first sentence (the toast says it)');
  await shoot('made_shot_name');
}

console.log(fails ? `${fails} FAILED` : 'ALL PASS');
await browser.close();
process.exit(fails ? 1 : 0);
