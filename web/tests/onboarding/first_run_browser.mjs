// The first run, in a real browser (Playwright) on a live app with an empty PICKUPS_HOME and no ComfyUI on this computer:
//   node tests/onboarding/first_run_browser.mjs --url http://127.0.0.1:7788 --expect none|found [--out <dir>]
//     [--width 1440 --height 900] [--platform Windows] [--demo]
// Checks the setup screen comes first, "Find ComfyUI on my network" shows a spinner and then either the servers found as
// Connect rows (--expect found: start the app with PICKUPS_SCAN_PORTS on a port where a stub answers /system_stats) or
// the scanned line and its reason with the address box still there, and the plain line under the address box. --demo then opens Try the
// demo and checks Previz to final starts on the seafront pier slide. Screenshots go to <dir>/<size>_<nn>_<what>.png.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = opt('--url', 'http://127.0.0.1:7788');
const OUT = opt('--out', null);
const EXPECT = opt('--expect', 'none');
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const PLATFORM = opt('--platform', null);
const DEMO = args.includes('--demo');
const TAG = `${W}x${H}${PLATFORM ? `_${PLATFORM.toLowerCase()}` : ''}_${EXPECT}`;
let fails = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) fails++;
};

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
if (PLATFORM) await ctx.addInitScript((p) => Object.defineProperty(navigator, 'platform', { get: () => p }), PLATFORM === 'Windows' ? 'Win32' : PLATFORM);
const page = await ctx.newPage();
const errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
page.on('pageerror', (e) => errors.push(String(e)));
let n = 0;
const shoot = async (what, { move = true } = {}) => {
  n += 1;
  if (!OUT) return;
  mkdirSync(OUT, { recursive: true });
  if (move) await page.mouse.move(1, 1);
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, `${TAG}_${String(n).padStart(2, '0')}_${what}.png`) });
};

await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
const dialog = page.getByRole('dialog', { name: 'Set up Pickups' });
await dialog.waitFor({ timeout: 20000 });
await page.getByRole('button', { name: 'Try the demo' }).waitFor({ timeout: 20000 });
check(await dialog.isVisible(), 'the setup screen is first');
console.log(`      heading: ${await page.locator('h1').first().innerText()}`);
if (PLATFORM === 'Windows') {
  // Windows and Linux open on Install ComfyUI; the scan sits with the other computer's address.
  await page.getByRole('button', { name: 'ComfyUI runs on another computer' }).click();
}
const find = page.getByRole('button', { name: /Find ComfyUI on my network|Looking on your network|Looking again/ });
check(await find.isVisible(), 'Find ComfyUI on my network is on screen');
check(await page.getByText("The GPU computer's name or IP, then :8188. Start ComfyUI there with --listen.").isVisible(), 'the plain line under the address box');
await shoot('first_screen');

await find.click();
await page.waitForTimeout(120);
const busy = (await find.getAttribute('aria-busy')) === 'true';
check(busy, 'a spinner while it looks');
await shoot('scanning', { move: false });
await page.waitForFunction(() => ![...document.querySelectorAll('button')].some((b) => /Looking on your network|Looking again/.test(b.textContent || '')), null, { timeout: 60000 });
await page.waitForTimeout(400);
if (EXPECT === 'none') {
  // Run 14: nothing on the quick look, nothing on the patient one: one plain line of what was looked at, and why.
  const note = page.getByTestId('scan-note');
  check(await note.isVisible(), 'the line saying what was scanned');
  const said = (await note.innerText()).replace(/\s+/g, ' ');
  check(/Looked at \d+ address/.test(said) && /twice\. Nothing answered\./.test(said) && /--listen/.test(said), `the scanned line and the reason: ${said}`);
  check(await page.getByRole('textbox', { name: 'Address' }).isVisible(), 'the address box is still there');
} else {
  const row = page.locator('ul button', { hasText: 'RTX 4090' }).first();
  check(await row.isVisible(), 'the server found is a Connect row with its GPU');
  console.log(`      row: ${(await row.innerText()).replace(/\s+/g, ' ')}`);
  check(await page.getByRole('textbox', { name: 'Address' }).isVisible(), 'the address box is still there');
}
await shoot('scan_result');

if (DEMO) {
  await page.getByRole('button', { name: 'Try the demo' }).click();
  await page.getByRole('button', { name: 'Exit demo' }).waitFor({ timeout: 20000 });
  const title = page.getByRole('heading', { name: 'Previz to final', exact: true });
  await title.waitFor({ timeout: 20000 });
  await title.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -240));
  await page.waitForTimeout(1500);
  const srcs = await title.evaluate((h) => {
    const tile = h.closest('.group');
    const first = tile?.querySelector('.absolute.inset-0[style*="z-index: 2"]') ?? tile;
    const all = (el) => [...(el?.querySelectorAll('img,video,source') ?? [])].map((m) => m.getAttribute('src') || m.getAttribute('poster') || '').filter(Boolean);
    return { first: all(first), tile: all(tile) };
  });
  console.log(`      previz tile first slide: ${srcs.first.join(', ')}`);
  check(srcs.first.some((s) => s.includes('ocean_person')), 'Previz to final starts on the seafront pier');
  check(!srcs.tile.some((s) => s.includes('hood_push')), 'no hood push slide in Previz to final');
  await shoot('demo_previz_tile');
}

check(!errors.length, `no console errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
await browser.close();
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
