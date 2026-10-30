// Run 1 ("the render follows the previz"), in a real browser (Playwright) against a temporary Pickups and a fake ComfyUI
// (nothing queued): each sentence is built and made into a shot, which must come with its depth passes (frames 0, 48,
// 96) and every subject's facing; then the Previz to final form is shot with Advanced open, and with words that disagree
// with the previz facing. The start frame is only planned (plan_only), never made. Only previz frames, finish, frame 0,
// the framing check and the plan go through; anything else posting is refused.
//   PREVIZ_URL=http://127.0.0.1:7893 node tests/previz/run1_browser.mjs --out <dir> [--width 1440 --height 900]
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7893';
const OUT = opt('--out', 'run1_out');
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const SENTENCES = [
  ['hug', 'Two friends hugging on a city sidewalk. Close to wide.', 'They walk toward the camera.'],
  ['car', 'A car driving on the highway. Tracking shot.', 'The car drives toward the camera.'],
];
mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, extraHTTPHeaders: { 'X-Pickups-QC': '1' } });
const page = await ctx.newPage();
page.on('pageerror', (e) => { console.log(`PAGEERROR ${e}`); fails++; });
const SAFE_POST = [/\/api\/previz\/pv_[0-9a-f]+\/(frames|finish)$/, /\/api\/previz\/framing$/, /\/api\/projects$/, /\/api\/shots\/[^/]+\/previz_frame0$/,
  /\/api\/mentions\/preview$/];
const refused = [];
await page.route('**/api/**', (route) => {
  const req = route.request();
  const path = new URL(req.url()).pathname;
  const planOnly = path === '/api/previz/start_frame' && /"plan_only":true/.test(req.postData() || '');
  if (req.method() === 'GET' || planOnly || SAFE_POST.some((r) => r.test(path))) return route.continue();
  refused.push(`${req.method()} ${path}`);
  return route.fulfill({ status: 503, body: '{"error":"refused by the run 1 test"}', contentType: 'application/json' });
});
const made = [];
let n = 0;
for (const [key, text, against] of SENTENCES) {
  await page.goto(`${BASE}/?s=${key}#/previz`, { waitUntil: 'domcontentloaded' });
  const box = page.getByRole('textbox', { name: 'Describe the shot' });
  await box.waitFor({ state: 'visible', timeout: 30000 });
  await box.fill(text);
  await page.getByRole('button', { name: 'Build', exact: true }).click();
  await page.locator('canvas[aria-label="Previz preview"]').waitFor({ state: 'visible', timeout: 20000 });
  await page.waitForFunction(() => !!window.__previzShot?.label, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: 'Make shot' }).click();
  // Save to (My shots, a project or a new one): the dialog's own Make shot.
  await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
  await page.waitForURL(/#\/shot\//, { timeout: 180000 });
  const id = decodeURIComponent(page.url().split('#/shot/')[1].split(/[/?]/)[0]);
  const shot = await page.evaluate(async (s) => (await fetch(`/api/shots/${encodeURIComponent(s)}`)).json(), id);
  const subjects = shot.previz_framing?.subjects ?? [];
  check(!!shot.previz_depth_url, `${key}: made ${id} with its depth pass`);
  check(subjects.length > 0 && subjects.every((s) => ['camera', 'away', 'left', 'right'].includes(s.facing)),
    `${key}: every subject has a facing (${subjects.map((s) => `${s.kind} ${s.facing}`).join(', ')})`);
  // The plan the start frame would run (nothing is made): method zbuf, its strengths, and clean words.
  const plan = await page.evaluate(async (s) => (await fetch('/api/previz/start_frame', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ shot_id: s, plan_only: true }) })).json(), id).catch(() => null);
  made.push({ key, text, id, framing: shot.previz_framing, plan });
  await page.waitForTimeout(1500);
  await page.mouse.move(1, 1);
  n += 1;
  await page.screenshot({ path: join(OUT, `${W}_${String(n).padStart(2, '0')}_${key}_form.png`) });
  const adv = page.getByRole('button', { name: 'Advanced', exact: true });
  await adv.scrollIntoViewIfNeeded();
  await adv.click();
  await page.waitForTimeout(400);
  const lock = page.getByRole('switch', { name: 'Lock the first frame' });
  const keys = page.getByRole('switch', { name: 'Mid and last keyframes' });
  check(await lock.count() === 1 && (await lock.getAttribute('aria-checked')) === 'false', `${key}: Lock the first frame is off by default`);
  check(await keys.count() === 1 && (await keys.getAttribute('aria-checked')) === 'false', `${key}: Mid and last keyframes is off by default`);
  await keys.scrollIntoViewIfNeeded();
  n += 1;
  await page.screenshot({ path: join(OUT, `${W}_${String(n).padStart(2, '0')}_${key}_advanced.png`) });
  const motion = page.getByRole('textbox', { name: 'Motion and camera' });
  const words = await motion.inputValue();
  check(/Keep the same .* and camera as the driver\.$/.test(words) && words.split(/(?<=[.!?])\s+/).length <= 3, `${key}: motion is the short L2R prompt ("${words}")`);
  const disagrees = !subjects.some((s) => s.in_view && s.facing === 'camera');
  await motion.fill(`${against} ${words}`);
  await page.waitForTimeout(300);
  const notice = page.getByText(/the previz wins/);
  if (disagrees) {
    await motion.scrollIntoViewIfNeeded();
    check(await notice.count() === 1, `${key}: words that say "toward the camera" are flagged when nobody faces the camera in the previz`);
    n += 1;
    await page.screenshot({ path: join(OUT, `${W}_${String(n).padStart(2, '0')}_${key}_facing_mismatch.png`) });
  } else {
    check(await notice.count() === 0, `${key}: someone faces the camera in the previz: no flag`);
  }
}
check(!refused.length, `nothing else posted (${refused.join(', ') || 'none'})`);
writeFileSync(join(OUT, `run1_${W}x${H}.json`), JSON.stringify({ made, refused }, null, 2));
await browser.close();
console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
process.exit(fails ? 1 : 0);
