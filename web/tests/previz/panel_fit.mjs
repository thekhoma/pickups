// The Previz panel and the render form fit the window: at 1440x900 and 1280x800 the Camera path row, the Still camera
// chip and Make shot are inside the panel's visible area and on top (the docked composer never covers them), and
// Previz to final is on screen on the render form. Runs in a real browser on the mock API (`npm run dev:mock`, or any
// build with ?mock=1):
//   node tests/previz/panel_fit.mjs --url http://127.0.0.1:5199 [--out <dir>] [--sizes 1440x900,1280x800]
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = opt('--url', process.env.PANEL_FIT_URL || 'http://localhost:5199');
const OUT = opt('--out', null);
const SIZES = opt('--sizes', '1440x900,1280x800').split(',').map((s) => s.split('x').map(Number));
let fails = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) fails++;
};

/** Is the element fully inside the box (the panel's scroll area, or the window) and the topmost thing at its centre? */
const fits = (el, box) =>
  el.evaluate((node, inner) => {
    const r = node.getBoundingClientRect();
    const b = inner ? document.querySelector(inner).getBoundingClientRect() : { top: 0, left: 0, bottom: innerHeight, right: innerWidth };
    const inside = r.width > 0 && r.height > 0 && r.top >= b.top && r.bottom <= b.bottom && r.left >= b.left && r.right <= b.right;
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return { inside, onTop: !!hit && (node === hit || node.contains(hit)), rect: [r.top, r.bottom, b.top, b.bottom].map(Math.round) };
  }, box);

const browser = await chromium.launch({ headless: true });
for (const [W, H] of SIZES) {
  const size = `${W}x${H}`;
  console.log(`--- ${size}`);
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  const shoot = async (what) => {
    if (!OUT) return;
    mkdirSync(OUT, { recursive: true });
    await page.mouse.move(1, 1);
    await page.waitForTimeout(350);
    await page.screenshot({ path: join(OUT, `${size}_${what}.png`) });
  };

  await page.goto(`${BASE}/?mock=1#/previz`, { waitUntil: 'domcontentloaded' });
  const box = page.getByRole('textbox', { name: 'Describe the shot' }).first();
  await box.waitFor({ timeout: 30000 });
  await box.fill('A man in a long coat stands on a pier at dusk. The camera slowly pushes in.');
  await box.press('Enter');
  await page.locator('canvas').first().waitFor({ timeout: 30000 });
  await page.waitForTimeout(3000);
  // The composer keeps focus after Build, so the dock is at its tallest: the worst case for the panel.
  const panel = page.getByRole('complementary', { name: 'Previz' });
  const scroller = 'aside[aria-label="Previz"] > div';
  const parts = [
    ['Camera path row', panel.getByText('Camera path', { exact: true })],
    ['Still camera chip', panel.getByRole('button', { name: 'Still camera', exact: true })],
    ['Make shot', panel.getByRole('button', { name: /^Make shot/ })],
  ];
  for (const [what, el] of parts) {
    const f = await fits(el, what === 'Make shot' ? null : scroller);
    check(f.inside && f.onTop, `${size} ${what}: in the panel's visible area and not covered (${JSON.stringify(f)})`);
  }
  await shoot('previz_panel');

  await panel.getByRole('button', { name: /^Make shot/ }).click();
  await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
  const rail = page.getByRole('complementary', { name: 'Tasks' });
  const final = rail.getByRole('button', { name: 'Previz to final', exact: true }).last();
  await final.waitFor({ timeout: 30000 });
  await page.waitForTimeout(1500);
  const f = await fits(final, null);
  check(f.inside && f.onTop, `${size} Previz to final: on screen on the render form and not covered (${JSON.stringify(f)})`);
  await shoot('render_form');

  check(!errors.length, `${size} no console or page errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
  await ctx.close();
}
await browser.close();
console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
process.exit(fails ? 1 : 0);
