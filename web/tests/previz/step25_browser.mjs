// Step 25, in a real browser (Playwright) against a temporary Pickups (fake ComfyUI, nothing queued): each sentence is
// built in Previz (the editor screenshot at this size) and, with --make, made into a shot, whose layout frames 0, 48 and
// 96 are saved for a strip. Only the previz frame and finish posts go through; anything else posting is refused.
//   PREVIZ_URL=http://127.0.0.1:7896 node tests/previz/step25_browser.mjs --out <dir> [--width 1440 --height 900] [--make]
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7896';
const OUT = opt('--out', 'step25_out');
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const MAKE = args.includes('--make');
const SIZE = `${W}x${H}`;
export const SENTENCES = [
  ['car_tracking', 'A car driving on the highway. Tracking shot.'],
  ['sports_car_tracking', 'A sports car driving on the highway. Tracking shot.'],
  ['person_hug', 'Two friends hugging on a city sidewalk. Close to wide.'],
  ['lake', 'A woman standing by a lake.'],
  ['woods', 'A hiker walking through the woods.'],
];
mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, extraHTTPHeaders: { 'X-Pickups-QC': '1' } });
const page = await ctx.newPage();
page.on('pageerror', (e) => { console.log(`PAGEERROR ${e}`); fails++; });
const SAFE_POST = [/\/api\/previz\/pv_[0-9a-f]+\/(frames|finish)$/, /\/api\/previz\/framing$/, /\/api\/projects$/];
const refused = [];
await page.route('**/api/**', (route) => {
  const req = route.request();
  const path = new URL(req.url()).pathname;
  if (req.method() === 'GET' || SAFE_POST.some((r) => r.test(path))) return route.continue();
  refused.push(`${req.method()} ${path}`);
  return route.fulfill({ status: 503, body: '{"error":"refused by the step 25 test"}', contentType: 'application/json' });
});
const made = [];
for (const [key, text] of SENTENCES) {
  await page.goto(`${BASE}/?s=${key}#/previz`, { waitUntil: "domcontentloaded" }); // a fresh page per sentence
  const box = page.getByRole('textbox', { name: 'Describe the shot' });
  await box.waitFor({ state: 'visible', timeout: 30000 });
  await box.fill(text);
  await page.getByRole('button', { name: 'Build', exact: true }).click();
  await page.locator('canvas[aria-label="Previz preview"]').waitFor({ state: 'visible', timeout: 20000 });
  await page.waitForFunction(() => !!window.__previzShot?.label, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.mouse.move(1, 1);
  await page.screenshot({ path: join(OUT, `${SIZE}_editor_${key}.png`) });
  check(true, `${key}: built "${await page.evaluate(() => window.__previzShot?.label)}"`);
  if (!MAKE) continue;
  await page.getByRole('button', { name: 'Make shot' }).click();
  // Save to (My shots, a project or a new one): the dialog's own Make shot.
  await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
  await page.waitForURL(/#\/shot\//, { timeout: 180000 });
  const id = decodeURIComponent(page.url().split('#/shot/')[1].split(/[/?]/)[0]);
  const shot = await page.evaluate(async (s) => (await fetch(`/api/shots/${encodeURIComponent(s)}`)).json(), id);
  check(shot.frames === 97, `${key}: made ${id}, ${shot.frames} frames at ${shot.width}x${shot.height}`);
  made.push({ key, text, id, shot });
}
check(!refused.length, `nothing else posted (${refused.join(', ') || 'none'})`);
writeFileSync(join(OUT, `step25_${SIZE}${MAKE ? '_made' : ''}.json`), JSON.stringify({ made, refused }, null, 2));
await browser.close();
console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
process.exit(fails ? 1 : 0);
