// Step 21, in a real browser (Playwright) against a temporary Pickups with the fake ComfyUI and the three flow flags on:
//   PREVIZ_URL=http://127.0.0.1:7894 node tests/previz/step21_browser.mjs --out <dir> [--width 1440 --height 900]
// 1. Remove, Relight and Finish as step flows: a screenshot of each step reached (the fake ComfyUI renders nothing real,
//    so Render shows the job's real stages; Review needs a take the fake run made).
// 2. Previz to final: Look and motion words typed, another shot opened, a task switched, back: the words are still there.
// 3. Make start frame from previz: the open queue shows it as one running inline row, with no Cancel.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7894';
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
if (OUT) mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
await ctx.addInitScript(() => { for (const t of ['remove', 'relight', 'finish']) localStorage.setItem(`pickups.flow.${t}`, '1'); });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log(`PAGEERROR ${e}`));
const shoot = async (name) => {
  if (!OUT) return;
  await page.mouse.move(1, 1);
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(OUT, `${W}_${name}.png`) });
};
const api = (path) => page.evaluate(async (p) => (await fetch(p)).json(), path);

await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
const shots = (await api('/api/shots')).filter((s) => s && s.id);
const plain = shots.find((s) => !s.previz_spec) ?? shots[0];
const made = shots.find((s) => s.previz_spec);
check(!!plain && !!made, `a shot and a previz shot to work on (${shots.length} shots)`);
const openTask = async (shotId, task) => {
  await page.goto(`${BASE}/#/shot/${encodeURIComponent(shotId)}`, { waitUntil: 'domcontentloaded' });
  const tile = page.getByRole('button', { name: new RegExp(`^${task === 'remove' ? 'Remove a person' : task === 'relight' ? 'Relight' : 'Finish'}`) }).first();
  await page.waitForTimeout(1500);
  const back = page.getByRole('button', { name: 'Back', exact: true });
  if (!(await tile.isVisible().catch(() => false)) && (await back.count())) { await back.first().click(); await page.waitForTimeout(800); }
  await tile.click({ timeout: 30000 });
};

// --- 1. the three flows ---------------------------------------------------------------------------------------------
for (const [task, primary] of [['remove', /^Remove /], ['relight', /^Relight$/], ['finish', /^Finish$/]]) {
  await openTask(plain.id, task);
  const steps = page.getByRole('navigation', { name: 'Steps' });
  await steps.waitFor({ state: 'visible', timeout: 30000 });
  check(true, `${task}: the step flow opens`);
  await page.waitForTimeout(1500);
  await shoot(`${task}_1`);
  const go = page.getByRole('button', { name: primary }).last();
  if (await go.isEnabled().catch(() => false)) {
    await go.click();
    await page.getByRole('button', { name: 'Back to the shot' }).waitFor({ state: 'visible', timeout: 30000 }).catch(() => undefined);
    check(await page.getByRole('list', { name: 'Stages' }).count() > 0, `${task}: Render shows the job's stages`);
    await shoot(`${task}_2_render`);
    const review = page.getByRole('button', { name: 'Approve' });
    if (await review.waitFor({ state: 'visible', timeout: 15000 }).then(() => true).catch(() => false)) {
      check((await page.getByRole('button', { name: 'Try again' }).count()) > 0 && (await page.getByRole('button', { name: 'Note' }).count()) > 0, `${task}: Review has Try again, Note, Approve`);
      await shoot(`${task}_3_review`);
    } else console.log(`NOTE  ${task}: no take from the fake run, so no Review screenshot`);
  } else console.log(`NOTE  ${task}: the primary is off here (not ready on this machine, or nothing picked)`);
}

// --- 2. Previz words survive a switch of shot and task -----------------------------------------------------------------
await page.goto(`${BASE}/#/shot/${encodeURIComponent(made.id)}`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500); if (!(await page.getByRole('button', { name: /^Previz to final/ }).first().isVisible().catch(() => false)) && (await page.getByRole('button', { name: 'Back', exact: true }).count())) { await page.getByRole('button', { name: 'Back', exact: true }).first().click(); await page.waitForTimeout(800); } await page.getByRole('button', { name: /^Previz to final/ }).first().click({ timeout: 30000 });
const look = page.getByRole('textbox', { name: 'Look', exact: true });
await look.waitFor({ state: 'visible', timeout: 30000 });
await look.fill('Warm dusk, wet cobbles, a red scarf.');
const motion = page.getByRole('textbox', { name: 'Motion and camera' });
await motion.fill('She walks to camera as it pulls back.');
await page.waitForTimeout(500);
await openTask(plain.id, 'relight');
await page.goto(`${BASE}/#/shot/${encodeURIComponent(made.id)}`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500); if (!(await page.getByRole('button', { name: /^Previz to final/ }).first().isVisible().catch(() => false)) && (await page.getByRole('button', { name: 'Back', exact: true }).count())) { await page.getByRole('button', { name: 'Back', exact: true }).first().click(); await page.waitForTimeout(800); } await page.getByRole('button', { name: /^Previz to final/ }).first().click({ timeout: 30000 });
await look.waitFor({ state: 'visible', timeout: 30000 });
check((await look.inputValue()) === 'Warm dusk, wet cobbles, a red scarf.', 'previz: the Look words survive a switch of shot and task');
check((await motion.inputValue()) === 'She walks to camera as it pulls back.', 'previz: the motion words too');
await shoot('previz_words_after_switch');

// --- 3. the start frame is an inline row in the open queue ------------------------------------------------------------
await page.getByRole('button', { name: 'Make start frame from previz' }).click();
await page.getByRole('button', { name: /^Queue/ }).first().click();
const row = page.locator('li', { hasText: 'Start frame' }).first();
const seen = await row.waitFor({ state: 'visible', timeout: 20000 }).then(() => true).catch(() => false);
check(seen, 'queue: the start frame shows as a row');
if (seen) {
  check((await row.getByRole('button', { name: 'Cancel' }).count()) === 0, 'queue: the inline row has no Cancel');
  await shoot('queue_start_frame_inline');
}
const jobs = await api('/api/jobs');
check(jobs.some((j) => j.lane === 'inline' && j.step === 'start_frame'), '/api/jobs has the inline start-frame row');

console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
await browser.close();
process.exit(fails ? 1 : 0);
