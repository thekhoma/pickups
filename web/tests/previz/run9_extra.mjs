// Run 9 follow-ups on screen (mock API): Previz's Advanced against Make shot, and the result panel's Notes when empty.
//   node tests/previz/run9_extra.mjs --out <dir> [--width 1440 --height 900] [--url http://localhost:5199]
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = opt('--url', 'http://localhost:5199');
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const SIZE = `${W}x${H}`;
let fails = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) fails++;
};
const browser = await chromium.launch({ headless: true });
const page = await (await browser.newContext({ viewport: { width: W, height: H } })).newPage();
const shoot = async (name) => {
  if (!OUT) return;
  mkdirSync(join(OUT, '13_extra'), { recursive: true });
  await page.mouse.move(1, 1);
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, '13_extra', `${SIZE}_${name}.png`) });
};

// Previz: open Advanced; no control sits under the pinned Make shot button.
await page.goto(`${BASE}/?mock=1#/previz`, { waitUntil: 'domcontentloaded' });
const box = page.getByRole('textbox', { name: 'Describe the shot' }).first();
await box.waitFor({ timeout: 30000 });
await box.fill('A man walks along a pier at dusk. The camera slowly pushes in.');
await box.press('Enter');
await page.waitForTimeout(4000);
await page.locator('aside[aria-label="Previz"] summary').filter({ hasText: 'Advanced' }).click();
await page.waitForTimeout(900);
await shoot('previz_advanced');
// Opening it brings it into view; then every control in it can be scrolled to and is never under the pinned Make shot.
const lastSlider = page.locator('aside[aria-label="Previz"] details input[type=range]').last();
const lb = await lastSlider.boundingBox();
const panel = await page.locator('aside[aria-label="Previz"] > div').first().boundingBox();
check(!!lb && !!panel && lb.y + lb.height <= panel.y + panel.height + 1, 'previz Advanced: opening it scrolls its last slider into view');
const covered = [];
for (const el of await page.locator('aside[aria-label="Previz"] details input').all()) {
  await el.scrollIntoViewIfNeeded();
  const hidden = await el.evaluate((n) => {
    const r = n.getBoundingClientRect();
    const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return !(top && n.contains(top)) && !(top && top === n);
  });
  if (hidden) covered.push((await el.getAttribute('aria-label')) || 'input');
}
check(!covered.length, `previz Advanced: every control can be reached (${covered.join(', ')})`);
await shoot('previz_advanced_scrolled');

console.log(fails ? `${fails} FAILED` : 'ALL PASS');
await browser.close();
process.exit(fails ? 1 : 0);
