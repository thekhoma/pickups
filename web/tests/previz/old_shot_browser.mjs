// Step 18b, in a real browser (Playwright) against a running Pickups with any ComfyUI (no GPU job is ever sent: the start
// frame is answered in the browser from the server's own plan):
//   PREVIZ_URL=http://127.0.0.1:7892 node tests/previz/old_shot_browser.mjs [--out <dir>] [--width 1440 --height 900]
// 1. Make shot sends the previz renderer's ID-pass subject matte: frame size, black and white, the car inside its box.
// 2. An older shot (frame 0 without its figures, camera or matte, as c08 was): Make start frame re-renders frame 0 from
//    the shot's spec and sends it first, so the plan names the vehicle and needs no remake.
// 3. When frame 0 can't be re-rendered and the old frame names nobody: refused with the guidance and one action,
//    Remake this shot, which opens Previz with the shot's sentence.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7892';
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const SENTENCE = 'A car chase through traffic on the highway. Low angle.';
if (OUT) mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const page = await (await browser.newContext({ viewport: { width: W, height: H } })).newPage();
page.on('pageerror', (e) => console.log(`PAGEERROR ${e}`));
const shot = async (name) => {
  if (!OUT) return;
  await page.mouse.move(1, 1);
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(OUT, `${W}_${name}.png`) });
};

// Make shot, optionally as an older page would have: frame 0 with only its box and horizon, no matte.
let finishBody = null;
async function makeShot({ thin }) {
  await page.route('**/api/previz/*/finish', async (route) => {
    const body = route.request().postDataJSON();
    finishBody = body;
    if (!thin) return route.continue();
    const old = { ...body, framing: { subject: body.framing?.subject, horizon: body.framing?.horizon } };
    delete old.matte;
    return route.continue({ postData: JSON.stringify(old) });
  });
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
  await page.unroute('**/api/previz/*/finish');
  const id = decodeURIComponent(page.url().split('#/shot/')[1].split(/[/?]/)[0]);
  await page.getByRole('textbox', { name: 'Look', exact: true }).waitFor({ state: 'visible', timeout: 30000 });
  await page.getByRole('button', { name: 'Dismiss' }).click({ timeout: 3000 }).catch(() => undefined);
  return id;
}
const api = (path, body) => page.evaluate(async ([p, b]) => {
  const r = await fetch(p, b ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) } : undefined);
  return { status: r.status, json: await r.json() };
}, [path, body]);

// --- 1. the ID-pass matte at Make shot --------------------------------------------------------------------------------
await makeShot({ thin: false });
const matte = finishBody?.matte;
check(typeof matte === 'string' && matte.startsWith('data:image/png;base64,'), 'Make shot sends a PNG subject matte');
const stats = await page.evaluate(async ([url, box]) => {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const x = c.getContext('2d');
  x.drawImage(img, 0, 0);
  const d = x.getImageData(0, 0, c.width, c.height).data;
  let white = 0, grey = 0, inBox = 0;
  const [x0, y0, x1, y1] = box.map((v, i) => v * (i % 2 ? c.height : c.width));
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i];
    if (v > 127) {
      white++;
      const p = i / 4, px = p % c.width, py = Math.floor(p / c.width);
      if (px >= x0 - 20 && px <= x1 + 20 && py >= y0 - 20 && py <= y1 + 20) inBox++;
    }
    if (v > 8 && v < 247) grey++;
  }
  return { w: c.width, h: c.height, white: white / (c.width * c.height), grey, inBox: white ? inBox / white : 0 };
}, [matte, finishBody.framing.subject]);
console.log(`matte: ${JSON.stringify(stats)}`);
check(stats.w === 1920 && stats.h === 1088, 'the matte is the frame size');
check(stats.grey === 0, 'the matte is black and white only (an ID pass, no shading)');
check(stats.white > 0.005 && stats.white < 0.6, 'the matte holds the subjects and not the frame');
check(stats.inBox > 0.8, 'the white is the car: inside its frame-0 box');

// --- 2. an older shot: frame 0 re-rendered at Make start frame ---------------------------------------------------------
const oldId = await makeShot({ thin: true });
const before = (await api(`/api/shots/${oldId}`)).json;
check(!before.previz_framing?.subjects && !before.previz_matte_url, 'the old shot has no figures and no matte (as c08)');
const sent = [];
await page.route('**/api/shots/*/previz_frame0', async (route) => { sent.push(route.request().postDataJSON()); return route.continue(); });
await page.route('**/api/previz/start_frame', async (route) => {
  const body = route.request().postDataJSON();
  if (body.plan_only) return route.continue();
  const plan = await (await fetch(`${BASE}/api/previz/start_frame`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, plan_only: true }) })).json();
  return route.fulfill({ status: plan.detail ? 422 : 200, json: plan.detail ? plan : { ...plan, url: before.thumb_url, upload_id: 'up_standin', seed: body.seed } });
});
await page.getByRole('button', { name: 'Make start frame from previz' }).click();
await page.getByAltText('Start frame').waitFor({ state: 'visible', timeout: 60000 });
check(sent.length === 1 && !!sent[0].framing?.subjects?.length && !!sent[0].matte, 'the page re-rendered frame 0 and sent its figures and matte');
const plan = (await api('/api/previz/start_frame', { shot_id: oldId, plan_only: true })).json;
check(/EXACTLY 1 vehicle in frame: vehicle 1/.test(plan.positive) && plan.negative.includes('frozen wheels'), 'the plan now names the vehicle, with the vehicle negative');
check(plan.subject_mask === true && !plan.action, 'with the matte and the camera back, no remake');
check(await page.getByRole('button', { name: 'Remake this shot' }).count() === 0, 'no Remake this shot after a re-render');

// --- 3. when frame 0 can't be re-rendered: refused, Remake this shot ---------------------------------------------------
const emptyId = await makeShot({ thin: true });
await page.route('**/api/shots/*/previz_frame0', (route) => route.fulfill({ status: 500, json: { detail: 'held' } }));
const refused = await api('/api/previz/start_frame', { shot_id: emptyId, plan_only: true });
check(refused.status === 422 && /no person or vehicle/.test(refused.json.detail), 'an old frame 0 that names nobody is refused, never EXACTLY ZERO');
await page.getByRole('button', { name: 'Make start frame from previz' }).click();
const remake = page.getByRole('button', { name: 'Remake this shot' });
await remake.waitFor({ state: 'visible', timeout: 60000 });
check(await page.getByText(/no person or vehicle/).count() > 0, 'the guidance shows');
await remake.scrollIntoViewIfNeeded();
await shot('01_remake_this_shot');
await remake.click();
const composer = page.getByRole('textbox', { name: 'Describe the shot' });
await composer.waitFor({ state: 'visible', timeout: 30000 });
check((await composer.inputValue()) === SENTENCE, 'Remake this shot opens Previz with the shot\'s sentence');
await shot('02_remake_opens_previz');

console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
await browser.close();
process.exit(fails ? 1 : 0);
