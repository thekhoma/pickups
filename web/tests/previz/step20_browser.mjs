// Step 20, in a real browser (Playwright) against a running Pickups with any ComfyUI (no GPU job is sent: the start frame
// is answered in the browser from the server's own plan):
//   PREVIZ_URL=http://127.0.0.1:7893 node tests/previz/step20_browser.mjs [--out <dir>] [--width 1440 --height 900]
// 1. A new rooftop shot that cranes down: frame 0 records the camera's real move, and the motion words say "cranes down"
//    (R&D's s06 said "tracks to the left and pushes in").
// 2. Three cast picked: the start frame takes two as pictures, names all three, and the third tile is dimmed.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7893';
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const SENTENCE = 'Three friends dancing on a rooftop. The camera cranes down.';
const CAST = [['Nia', 'a woman in her 20s with curly black hair, pink blazer', '#d98aa0'], ['Sol', 'a man in his 30s with a shaved head, white t-shirt', '#8ab4d9'],
  ['Teo', 'a man in his 20s with round glasses, orange hoodie', '#e0a050']];
if (OUT) mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const page = await (await browser.newContext({ viewport: { width: W, height: H } })).newPage();
page.on('pageerror', (e) => console.log(`PAGEERROR ${e}`));

await page.goto(`${BASE}/#/previz`, { waitUntil: 'domcontentloaded' });
const box = page.getByRole('textbox', { name: 'Describe the shot' });
await box.waitFor({ state: 'visible', timeout: 30000 });
await box.fill(SENTENCE);
await page.getByRole('button', { name: /^(Build|Rebuild)$/ }).first().click();
await page.waitForFunction(() => !!window.__previzShot?.label, null, { timeout: 45000 });
await page.waitForTimeout(1500);
await page.getByRole('button', { name: 'Make shot' }).click();
// Save to (My shots, a project or a new one): the dialog's own Make shot.
await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
await page.waitForURL(/#\/shot\//, { timeout: 180000 });
const id = decodeURIComponent(page.url().split('#/shot/')[1].split(/[/?]/)[0]);
const shotRow = await page.evaluate(async (s) => (await fetch(`/api/shots/${s}`)).json(), id);

// --- 1. the real move ---------------------------------------------------------------------------------------------------
const move = shotRow.previz_framing?.move;
check(!!move && move.end.pos[1] < move.start.pos[1] - 0.8, `frame 0 records the real move, high to low (${JSON.stringify(move)})`);
const motion = page.getByRole('textbox', { name: 'Motion and camera' });
await motion.waitFor({ state: 'visible', timeout: 30000 });
const words = await motion.inputValue();
console.log(`motion: ${words}`);
check(/cranes down/.test(words) && !/tracks to the left/.test(words), 'the motion words say cranes down');
check(await page.getByText('Previz mismatch').count() === 0, 'the path agrees with the words: no mismatch flag');

// --- 2. three cast, two sent ------------------------------------------------------------------------------------------
await page.evaluate(async ([pid, cast]) => {
  for (const [name, line, colour] of cast) {
    const c = document.createElement('canvas');
    c.width = 160; c.height = 90;
    const x = c.getContext('2d');
    x.fillStyle = colour;
    x.fillRect(0, 0, 160, 90);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    const f = new FormData();
    f.append('file', blob, `${name}.png`); f.append('name', name); f.append('kind', 'person'); f.append('line', line);
    await fetch(`/api/projects/${pid}/cast`, { method: 'POST', body: f });
  }
}, [shotRow.project_id, CAST]);
await page.reload({ waitUntil: 'domcontentloaded' });
const look = page.getByRole('textbox', { name: 'Look', exact: true });
await look.waitFor({ state: 'visible', timeout: 8000 }).catch(async () => {
  await page.getByRole('button', { name: 'Previz to final' }).first().click();
  await look.waitFor({ state: 'visible', timeout: 30000 });
});
await page.getByRole('button', { name: 'Dismiss' }).click({ timeout: 3000 }).catch(() => undefined);
for (const [name] of CAST) await page.getByRole('group', { name: 'Add from the cast' }).getByRole('button', { name }).click();
let planned = null;
await page.route('**/api/previz/start_frame', async (route) => {
  const body = route.request().postDataJSON();
  if (body.plan_only) return route.continue();
  planned = await (await fetch(`${BASE}/api/previz/start_frame`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, plan_only: true }) })).json();
  return route.fulfill({ status: 200, json: { ...planned, url: shotRow.thumb_url, upload_id: 'up_standin', seed: body.seed } });
});
await page.getByRole('button', { name: 'Make start frame from previz' }).click();
await page.getByAltText('Start frame').waitFor({ state: 'visible', timeout: 60000 });
check(planned?.references === 2, `two references sent as pictures (${planned?.references})`);
check(CAST.every(([n, line]) => planned?.positive.includes(n) && planned.positive.includes(line)), 'all three named, each with their line');
await page.waitForTimeout(800); // the fade
const tiles = await page.locator('[data-sent]').evaluateAll((els) => els.map((e) => [e.getAttribute('data-sent'), getComputedStyle(e).opacity]));
console.log(`tiles: ${JSON.stringify(tiles)}`);
check(tiles.length === 3 && tiles[0][0] === 'true' && tiles[1][0] === 'true' && tiles[2][0] === 'false' && Number(tiles[2][1]) < 0.5,
  'the third reference is dimmed: not sent as a picture');
if (OUT) {
  await page.locator('[data-sent]').first().scrollIntoViewIfNeeded();
  await page.mouse.move(1, 1);
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(OUT, `${W}_01_three_cast_two_sent.png`) });
}

// --- 3. (step 20b, v2.4 9) a previz path that disagrees with the words: the words stay, the form flags it -------------
// The recorded move is swapped for R&D's s06 stored path (tracks to the left and pushes in), as the old s06 had it.
await page.route('**/api/previz/*/finish', async (route) => {
  const body = route.request().postDataJSON();
  body.framing = { ...body.framing, move: { start: { pos: [3.35, 1.435, 3.977], look: [0, 1.085, 0] }, end: { pos: [1.063, 1.435, 2.912], look: [0, 1.085, 0] } } };
  return route.continue({ postData: JSON.stringify(body) });
});
await page.goto(`${BASE}/#/previz`, { waitUntil: 'domcontentloaded' });
await box.waitFor({ state: 'visible', timeout: 30000 });
await box.fill(SENTENCE);
await page.getByRole('button', { name: /^(Build|Rebuild)$/ }).first().click();
await page.waitForFunction(() => !!window.__previzShot?.label, null, { timeout: 45000 });
await page.waitForTimeout(1500);
await page.getByRole('button', { name: 'Make shot' }).click();
// Save to (My shots, a project or a new one): the dialog's own Make shot.
await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
await page.waitForURL(/#\/shot\//, { timeout: 180000 });
await page.unroute('**/api/previz/*/finish');
await motion.waitFor({ state: 'visible', timeout: 30000 });
const said = await motion.inputValue();
console.log(`motion: ${said}`);
check(/The camera cranes down\./.test(said) && !/tracks to the left/.test(said), 'the words keep the user\'s camera move');
const flag = page.getByText('Previz mismatch');
check(await flag.count() === 1, 'the form flags the previz mismatch');
check(await page.getByText('The path tracks to the left and pushes in.').count() === 1, 'the flag says what the path does');
if (OUT) {
  await flag.scrollIntoViewIfNeeded();
  await page.mouse.move(1, 1);
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(OUT, `${W}_02_previz_mismatch.png`) });
}

console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
await browser.close();
process.exit(fails ? 1 : 0);
