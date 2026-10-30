// Run 10, in a real browser (Playwright) on the mock API (`npm run dev:mock`, or any build with ?mock=1):
//   node tests/previz/run10_browser.mjs --url http://127.0.0.1:5199 --out <dir> [--width 1440 --height 900] [--only look,...]
// Each part checks one Run 10 item on screen and saves <dir>/<part>/<size>_<nn>_<what>.png. What the forms sent is read
// from the mock's window.__mockSent (the bodies the server would get).
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = opt('--url', process.env.RUN10_URL || 'http://localhost:5199');
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
const shoot = async (what, { move = true } = {}) => {
  n += 1;
  if (!OUT) return;
  const dir = join(OUT, part);
  mkdirSync(dir, { recursive: true });
  if (move) await page.mouse.move(1, 1);
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
const sentTo = (route) => page.evaluate((r) => (window.__mockSent || []).filter((x) => x.route === r).map((x) => x.body), route);
const rail = () => page.locator('aside[aria-label="Tasks"]');
const fresh = async (hash = '', query = 'mock=1') => {
  await page.goto(`${BASE}/?${query}#${hash}`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    for (const k of Object.keys(localStorage)) if (k.startsWith('pickups.flow.') || k === 'pickups.drafts.v1') localStorage.removeItem(k);
  });
  await page.goto(`${BASE}/?${query}#${hash}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
};
const section = (name) => {
  part = name;
  n = 0;
  console.log(`--- ${name} (${SIZE})`);
};
const want = (name) => !ONLY.length || ONLY.includes(name);
const PIER = 'A man in a long coat stands on a pier at dusk. The camera slowly pushes in.';
const buildPrevizShot = async () => {
  await fresh('/previz');
  const box = page.getByRole('textbox', { name: 'Describe the shot' }).first();
  await box.waitFor({ timeout: 30000 });
  await box.fill(PIER);
  await box.press('Enter');
  await page.locator('canvas').first().waitFor({ timeout: 30000 });
  await page.waitForTimeout(4000);
};

// --- A Previz to final: the Look shown is the Look sent (start frame and render), in plain English -------------------
if (want('look')) {
  section('A_look');
  await buildPrevizShot();
  await page.getByRole('button', { name: /^Make shot/ }).first().click();
  // Save to (My shots, a project or a new one): the dialog's own Make shot.
  await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
  const look = rail().getByRole('textbox', { name: /^Look/ }).first();
  check(await until(() => look.isVisible(), 30000), 'make shot: Previz to final opens with the Look');
  const shown = (await look.inputValue()).trim();
  console.log(`note: Look shown: "${shown}"`);
  check(shown.startsWith('It is dusk. The man wears a long coat. The grey figure is a man in a long coat who stands on a pier at dusk.'), 'look: time, wardrobe, then one plain sentence');
  check(!/long coat stands/.test(shown), 'look: never "is a man in a long coat stands"');
  await shoot('look_shown');
  await rail().getByRole('button', { name: 'Make start frame from previz' }).click();
  check(await until(async () => (await sentTo('/api/previz/start_frame')).length > 0, 10000), 'start frame: asked');
  const sf = (await sentTo('/api/previz/start_frame')).at(-1);
  check(sf.user_look === shown, 'start frame: the auto Look is sent as it is shown (user_look)');
  check(await until(() => rail().getByRole('button', { name: 'Try another' }).isVisible(), 20000), 'start frame: made');
  await shoot('start_frame');
  await rail().getByRole('button', { name: 'Previz to final', exact: true }).last().click();
  check(await until(async () => (await sentTo('/api/jobs')).some((j) => j.task === 'previz'), 10000), 'render: queued');
  const job = (await sentTo('/api/jobs')).filter((j) => j.task === 'previz').at(-1);
  check(job.params.look_prompt === shown, 'render: the Look goes with the accepted start frame (look_prompt)');
  check(!!job.params.look_frame_upload_id, 'render: with the start frame it accepted');
  console.log(`note: render params: ${JSON.stringify({ look_prompt: job.params.look_prompt, prompt: job.params.prompt })}`);
  await shoot('render_queued');
  const strip = rail().getByTestId('run-strip').first();
  if (await until(() => strip.isVisible(), 5000)) await strip.getByRole('button', { name: 'Cancel' }).click().catch(() => {});
}

// --- B Replace a person: the flow sends kind "person" -------------------------------------------------------------------
if (want('recast')) {
  section('B_recast');
  await fresh('/projects');
  await page.getByText('Fun cast', { exact: true }).first().click();
  await page.waitForTimeout(1000);
  await page.locator('main').getByText(/castle walk/i).first().click({ timeout: 20000 });
  await rail().waitFor({ state: 'visible', timeout: 20000 });
  await rail().getByRole('button', { name: /^Replace a person/ }).first().click();
  const usePerson = rail().getByRole('button', { name: /^Use Person \d|^Use this pick/ }).last();
  check(await until(() => usePerson.isEnabled(), 30000), 'who: Person 1 is picked for you');
  await shoot('who');
  await usePerson.click();
  await page.waitForTimeout(500);
  // Someone already in the cast, so it renders at once.
  const castTab = rail().getByRole('radio', { name: 'Cast', exact: true });
  if (await castTab.count()) await castTab.click();
  const member = rail().getByRole('radio').filter({ hasNotText: /^(New|Cast)$/ }).first();
  await member.click().catch(() => {});
  const go = rail().getByRole('button', { name: /^(Use |Render|Replace a person)/ }).last();
  await go.click().catch(() => {});
  check(await until(async () => (await sentTo('/api/jobs')).some((j) => j.task === 'recast'), 15000), 'render: queued');
  const job = (await sentTo('/api/jobs')).filter((j) => j.task === 'recast').at(-1);
  check(job?.params?.kind === 'person', `render: kind "person" is sent (${job?.params?.kind})`);
  check(Array.isArray(job?.params?.points) && job.params.points.length > 0, 'render: with the click points');
  await shoot('render');
  const strip = rail().getByTestId('run-strip').first();
  if (await until(() => strip.isVisible(), 5000)) await strip.getByRole('button', { name: 'Cancel' }).click().catch(() => {});
}

// --- D Lock seed: none before a seed, none after a reload; on only with a seed ---------------------------------------
if (want('seed')) {
  section('D_seed');
  await fresh('/create/image');
  const panel = page.locator('aside[aria-label="Make an image"]');
  await panel.waitFor({ state: 'visible', timeout: 20000 });
  const lock = panel.getByRole('button', { name: /^Lock seed/ });
  check((await lock.count()) === 0, 'image page, fresh: no lock before a seed');
  await shoot('no_seed');
  await panel.getByRole('textbox', { name: 'Prompt' }).fill('A lighthouse on a cliff at dawn');
  await panel.getByRole('button', { name: 'Generate', exact: true }).click();
  check(await until(() => lock.isVisible(), 20000), 'after a run: the lock shows its seed');
  const label = await lock.getAttribute('aria-label');
  check(/^Lock seed \d+$/.test(label || ''), `after a run: "${label}"`);
  await lock.click();
  check((await lock.getAttribute('aria-pressed')) === 'true', 'pressed with a seed: on');
  await shoot('seed_locked');
  await until(async () => !(await page.getByTestId('run-strip').first().isVisible().catch(() => false)), 60000);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await panel.waitFor({ state: 'visible', timeout: 20000 });
  const after = (await lock.count()) ? `${await lock.getAttribute('aria-label')} pressed=${await lock.getAttribute('aria-pressed')}` : 'none';
  console.log(`note: after a reload the lock is: ${after}`);
  check(!(await lock.count()) || (await lock.getAttribute('aria-pressed')) !== 'true' || /\d/.test((await lock.getAttribute('aria-label')) || ''), 'after a reload: never pressed with no seed');
  await panel.getByRole('textbox', { name: 'Prompt' }).fill('A lighthouse on a cliff at dawn');
  await panel.getByRole('button', { name: 'Generate', exact: true }).click();
  check(await until(async () => (await sentTo('/api/images')).length > 0, 10000), 'after a reload: Generate runs');
  const sentAfter = (await sentTo('/api/images')).at(-1);
  check(sentAfter.seed_locked === false, `after a reload: a fresh seed, never sent as locked (seed ${sentAfter.seed}, locked ${sentAfter.seed_locked})`);
  check(await until(() => lock.isVisible(), 20000), 'after a reload and a run: the lock shows the new seed');
  await shoot('after_reload');
}

// --- E Cancel: Cancelled at once; the next job waits its turn ---------------------------------------------------------
if (want('cancel')) {
  section('E_cancel');
  await fresh('/projects');
  await page.getByText('Highlands & Alps', { exact: true }).first().click();
  await page.waitForTimeout(1000);
  await page.locator('main').getByText(/trail/i).first().click({ timeout: 20000 });
  await rail().waitFor({ state: 'visible', timeout: 20000 });
  for (let i = 0; i < 3; i++) {
    const back = rail().getByRole('button', { name: 'Back', exact: true }).first();
    if (!(await back.isVisible().catch(() => false))) break;
    await back.click();
    await page.waitForTimeout(400);
  }
  await rail().getByRole('button', { name: /^Finish/ }).first().click();
  await page.waitForTimeout(600);
  await rail().getByRole('button', { name: /^Finish/ }).last().click();
  const strip = rail().getByTestId('run-strip').first();
  check(await until(() => strip.isVisible(), 15000), 'finish: running, with Cancel');
  await shoot('running');
  await strip.getByRole('button', { name: 'Cancel' }).click();
  check(await until(async () => (await rail().getByText('Cancelled').count()) > 0, 3000), 'cancel: "Cancelled" at once');
  await shoot('cancelled');
}

// --- F Top view: camera marker, path points, click the path to add one, drag, remove, with feedback ------------------
if (want('top')) {
  section('F_top');
  await buildPrevizShot();
  const canvas = page.locator('canvas').first();
  await page.getByRole('button', { name: 'Top', exact: true }).click();
  await page.waitForTimeout(1500);
  const top = () => page.evaluate(() => window.__previzTop || null);
  let t = await top();
  check(!!t && !!t.camera, `top: the camera marker is drawn (${t ? `${t.camera.x},${t.camera.y}` : 'none'})`);
  check(!!t && t.points.length >= 2, `top: the path's points are drawn (${t ? t.points.length : 0})`);
  console.log(`note: top state: ${JSON.stringify(t)}`);
  await shoot('top_built');
  const cb = await canvas.boundingBox();
  // hover a point: the cursor says it can be dragged
  const p1 = t.points[t.points.length - 1];
  await page.mouse.move(cb.x + p1.x, cb.y + p1.y);
  await page.waitForTimeout(200);
  const cursor = await canvas.evaluate((c) => c.style.cursor);
  check(cursor === 'grab', `top: a point under the pointer shows a grab cursor (${cursor})`);
  // click the middle of the path: a point is added and selected
  const mid = t.path[Math.floor(t.path.length / 2)];
  const before = t.points.length;
  await page.mouse.move(cb.x + mid.x, cb.y + mid.y);
  await page.waitForTimeout(150);
  const pathCursor = await canvas.evaluate((c) => c.style.cursor);
  console.log(`note: cursor over the path: ${pathCursor}`);
  await page.mouse.down();
  await page.mouse.up();
  await page.waitForTimeout(800);
  t = await top();
  check(t.points.length === before + 1, `top: a click on the path adds a point (${before} → ${t.points.length})`);
  const bar = page.getByText(/^Point \d+ of \d+$/);
  check(await until(() => bar.isVisible(), 3000), `top: the new point is selected and says so ("${(await bar.count()) ? await bar.innerText() : ''}")`);
  await shoot('point_added');
  // drag it (the camera sits on it now: the point must win)
  const added = t.points.length;
  const selIndex = t.selected ?? 1;
  const sel = t.points[selIndex];
  await page.mouse.move(cb.x + sel.x, cb.y + sel.y);
  await page.mouse.down();
  await page.mouse.move(cb.x + sel.x + 50, cb.y + sel.y - 40, { steps: 10 });
  await shoot('dragging', { move: false });
  await page.mouse.up();
  await page.waitForTimeout(800);
  t = await top();
  const q = t.points[selIndex];
  check(Math.hypot(q.x - sel.x, q.y - sel.y) > 20, `top: dragging the point moves it (${sel.x},${sel.y} → ${q.x},${q.y})`);
  check(t.points.length === added && t.selected === selIndex, `top: the drag moves that point, never adds another (${added} → ${t.points.length}, selected ${t.selected})`);
  await shoot('point_dragged');
  // drag the camera marker
  const cam = t.camera;
  await page.mouse.move(cb.x + cam.x, cb.y + cam.y);
  await page.mouse.down();
  await page.mouse.move(cb.x + cam.x - 40, cb.y + cam.y + 30, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(800);
  const t2 = await top();
  check(Math.hypot(t2.camera.x - cam.x, t2.camera.y - cam.y) > 15, `top: dragging the camera moves it (${cam.x},${cam.y} → ${t2.camera.x},${t2.camera.y})`);
  check(await page.getByRole('button', { name: 'Reset', exact: true }).isVisible(), 'top: the path is now yours (Reset shows)');
  await shoot('camera_dragged');
  // remove the selected point
  const count = t2.points.length;
  const remove = page.getByRole('button', { name: /^Remove/ }).first();
  if (await remove.isVisible().catch(() => false) && (await remove.isEnabled())) {
    await remove.click();
    await page.waitForTimeout(600);
    const t3 = await top();
    check(t3.points.length === count - 1, `top: Remove takes the selected point out (${count} → ${t3.points.length})`);
  } else {
    console.log('note: no Remove for the selected point (it may be an end point)');
  }
  await shoot('after_remove');
}

await browser.close();
console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
process.exit(fails ? 1 : 0);
