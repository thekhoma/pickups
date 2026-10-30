// Steps 23 and 24, in a real browser (Playwright) against a temporary Pickups with the fake ComfyUI and stand-in image, sheet and
// recast tasks (a test-only launcher, never shipped), with pickups.flow.recast and pickups.flow.character on:
//   PREVIZ_URL=http://127.0.0.1:7896 node tests/previz/step23_browser.mjs --out <dir> [--width 1440 --height 900]
//   [--fail-flag <path>]   (the file the stand-in sheet task reads: when it exists, the next sheet fails at the back view)
// Replace a person: Who, Character (Cast: a sheet not made, making "x of 3", ready, failed, Try again by cast_id; New: 4
// options, the viewer on an option, pick, name, Make sheet), the viewer on a sheet (arrows, Esc, focus back), the draft
// surviving a reload and a switch, Render with the job's own stages, Review; then the cast page's sheet states and New
// character. Step 24 adds: the step tabs stay pinned while the rail scrolls, Advanced scrolls clear of the primary, the
// viewer's "Use <name>" starts the render, one word ("Failed") for a failure with labelled empty views, no position line
// in the viewer, one Approve on Review, New's empty and loading states, and the cast page's one New character.
// Every screenshot is <size>_<nn>_<what>.png.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7896';
const OUT = opt('--out', null);
const W = Number(opt('--width', 1440));
const H = Number(opt('--height', 900));
const FAIL_FLAG = opt('--fail-flag', null);
const SIZE = `${W}x${H}`;
if (OUT) mkdirSync(OUT, { recursive: true });
let fails = 0;
let n = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) fails++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
await ctx.addInitScript(() => {
  for (const t of ['recast', 'character']) localStorage.setItem(`pickups.flow.${t}`, '1');
});
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log(`PAGEERROR ${e}`));
const sheetPosts = [];
page.on('request', (r) => { if (r.method() === 'POST' && /\/cast\/sheet$/.test(r.url())) sheetPosts.push(JSON.parse(r.postData() || '{}')); });
const shoot = async (what, { hover = false } = {}) => {
  n += 1;
  if (!OUT) return;
  if (!hover) await page.mouse.move(1, 1);
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(OUT, `${SIZE}_${String(n).padStart(2, '0')}_${what}.png`) });
};
const api = (path, init) => page.evaluate(async ([p, i]) => { const r = await fetch(p, i); return r.json(); }, [path, init]);
const until = async (fn, ms = 30000, step = 250) => {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (await fn()) return true; await page.waitForTimeout(step); }
  return false;
};
const rail = () => page.locator('aside').last();
const primary = (name) => page.getByRole('button', { name, exact: true }).last();
const inViewport = async (loc) => {
  const b = await loc.boundingBox();
  return !!b && b.y >= 0 && b.y + b.height <= H && b.x >= 0 && b.x + b.width <= W;
};
// The rail's scrolling body to its end: the step tabs stay put (pinned above it).
const scrollRail = (to) => page.evaluate((y) => { for (const el of document.querySelectorAll('aside *')) { const cs = getComputedStyle(el); if (/(auto|scroll)/.test(cs.overflowY) && el.scrollHeight > el.clientHeight) el.scrollTop = y < 0 ? el.scrollHeight : y; } }, to);
// Step 25: the selected sheet's name and state sit on screen without scrolling the rail: below the pinned step tabs and
// above the footer's primary.
const sheetHeadShown = async (name, foot) => {
  const head = page.getByRole('region', { name: `${name}'s sheet` }).locator(':scope > div').first();
  const [h, t, f] = [await head.boundingBox(), await page.getByRole('navigation', { name: 'Steps' }).boundingBox(), await foot.boundingBox()];
  return !!h && !!t && !!f && h.y >= t.y + t.height - 1 && h.y + h.height <= f.y + 1;
};
const tabsShown = () => inViewport(page.getByRole('navigation', { name: 'Steps' }));
const activeLabel = () => page.evaluate(() => document.activeElement?.getAttribute('aria-label') || document.activeElement?.textContent?.trim() || '');

await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
await page.evaluate(() => localStorage.removeItem('pickups.drafts.v1'));
const shots = (await api('/api/shots')).filter((s) => s && s.id);
const shot = shots.find((s) => /Park, person walking/.test(s.name)) ?? shots[0];
const other = shots.find((s) => s.id !== shot.id);
const pid = shot.project_id || 'unsorted';
const castNow = async () => (await api(`/api/projects/${pid}`)).cast.filter((c) => c.kind === 'person');
// People with a portrait picture and no sheet (a wide picture is a sheet already).
const portrait = (url) => page.evaluate((u) => new Promise((res) => { const i = new Image(); i.onload = () => res(i.naturalWidth <= i.naturalHeight * 1.3); i.onerror = () => res(false); i.src = u; }), url);
const bare = [];
for (const c of await castNow()) if (!c.sheet && (await portrait(c.url))) bare.push(c);
check(bare.length >= 2, `two cast members without a sheet to work on (${bare.length})`);
const [first, second] = bare;

const openRecast = async (shotId = shot.id) => {
  await page.goto(`${BASE}/#/shot/${encodeURIComponent(shotId)}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  const tile = page.getByRole('button', { name: /^Replace a person/ }).first();
  const back = page.getByRole('button', { name: 'Back', exact: true });
  if (!(await tile.isVisible().catch(() => false)) && (await back.count())) { await back.first().click(); await page.waitForTimeout(800); }
  await tile.click({ timeout: 30000 });
  await page.getByRole('navigation', { name: 'Steps' }).waitFor({ state: 'visible', timeout: 30000 });
};

// --- 1 Who -----------------------------------------------------------------------------------------------------------
await openRecast();
check(true, 'recast: the step flow opens');
const person1 = page.getByRole('radio', { name: 'Person 1' });
check(await until(async () => (await person1.getAttribute('aria-checked').catch(() => null)) === 'true'), 'who: the person most on screen is picked for you (Person 1)');
const usePerson = primary('Use Person 1');
check(await until(() => usePerson.isEnabled()), 'who: the primary says "Use Person 1"');
check(await inViewport(usePerson), 'who: the primary is on screen without scrolling');
await shoot('who');
check((await page.getByRole('button', { name: 'Check tracking' }).count()) === 1, 'who: Check tracking, with no "?" beside it');
await usePerson.click();

// --- 2 Character: Cast, not made ---------------------------------------------------------------------------------------
await page.getByRole('radiogroup', { name: 'Your cast' }).waitFor({ state: 'visible', timeout: 15000 });
check((await page.getByRole('button', { name: 'Edit', exact: true }).count()) > 0, 'character: Who folds to a summary with Edit');
await page.getByRole('radio', { name: first.name, exact: true }).click();
const makeSheet = primary('Make sheet');
check(await until(() => makeSheet.isVisible()), `character: ${first.name} has no sheet, so the primary is "Make sheet"`);
check((await page.getByText('No sheet', { exact: true }).count()) > 0, 'character: the sheet panel says No sheet');
check((await page.getByRole('img', { name: 'Back: not made' }).count()) === 1 && (await page.getByRole('img', { name: 'Close-up: not made' }).count()) === 1, 'character: the empty views are labelled Back and Close-up');
check((await page.getByRole('radiogroup', { name: 'Your cast' }).getByRole('img', { name: 'No sheet' }).count()) > 0, 'character: a cast tile without a sheet has a quiet dot, not a badge');
await scrollRail(-1);
await page.waitForTimeout(300);
check(await tabsShown(), 'character: the step tabs stay on screen with the rail scrolled to the end');
await scrollRail(0);
check(await sheetHeadShown(first.name, makeSheet), `character: ${first.name}'s sheet name and state show without scrolling (${SIZE})`);
await shoot('character_cast_not_made');

// The viewer from a cast tile: Esc closes it and focus goes back to the button that opened it.
const openFirst = page.getByRole('button', { name: `Open ${first.name} full screen` });
await openFirst.focus();
await openFirst.press('Enter');
const viewer = page.getByRole('dialog', { name: new RegExp(`^${first.name}:`) });
check(await viewer.isVisible().catch(() => false), 'viewer: opens from a cast tile');
await page.keyboard.press('Escape');
check(await until(async () => !(await viewer.isVisible().catch(() => false)), 3000), 'viewer: Esc closes it');
check((await activeLabel()) === `Open ${first.name} full screen`, `viewer: focus returns to the tile's button (${await activeLabel()})`);

// --- Making, x of 3 ------------------------------------------------------------------------------------------------------
await makeSheet.click();
const making = primary('Making the sheet');
check(await until(() => making.isVisible(), 10000), 'making: the primary is "Making the sheet"');
check(await making.isDisabled(), 'making: and it is off');
check(await until(async () => (await page.getByText('2 of 3', { exact: true }).count()) > 0, 20000), 'making: the back view shows "2 of 3"');
await shoot('character_sheet_making');
check(await until(async () => (await page.getByText('3 of 3', { exact: true }).count()) > 0, 20000), 'making: the close-up and the composite show "3 of 3"');

// --- Ready ------------------------------------------------------------------------------------------------------------------
const useFirst = primary(`Use ${first.name}`);
check(await until(() => useFirst.isVisible(), 30000), `ready: the primary is "Use ${first.name}"`);
check(sheetPosts.at(-1)?.cast_id === first.id && !sheetPosts.at(-1)?.name, 'make sheet: the request goes by cast_id');
check(await sheetHeadShown(first.name, useFirst), `ready: the sheet name and Ready show without scrolling (${SIZE})`);
await shoot('character_sheet_ready');

// The viewer on the sheet: Front · 1 of 4 (its name for screen readers; no line on screen), arrows both ways (and round),
// the on-screen buttons, Use <name>, Esc, focus back.
const frontTile = page.getByRole('button', { name: `${first.name}, Front: open full screen` });
await frontTile.focus();
await frontTile.press('Enter');
const pos = async () => ((await viewer.getAttribute('aria-label').catch(() => '')) || '').replace(/^[^:]*:\s*/, '');
check(await until(async () => (await pos()) === 'Front · 1 of 4', 3000), `viewer: "Front · 1 of 4" (${await pos()})`);
check((await viewer.locator('.slate', { hasText: / of \d/ }).count()) === 0, 'viewer: no "· 1 of 4" line on screen');
check(await viewer.getByRole('button', { name: `Use ${first.name}`, exact: true }).isVisible().catch(() => false), `viewer: a "Use ${first.name}" button`);
await shoot('viewer_sheet');
await page.keyboard.press('ArrowRight');
check((await pos()) === 'Back · 2 of 4', 'viewer: Right goes to Back · 2 of 4');
await page.keyboard.press('ArrowLeft');
await page.keyboard.press('ArrowLeft');
check((await pos()) === 'Sheet · 4 of 4', 'viewer: Left from the first goes round to Sheet · 4 of 4');
await viewer.getByRole('button', { name: 'Next' }).click();
check((await pos()) === 'Front · 1 of 4', 'viewer: the on-screen Next goes round to Front');
await viewer.getByRole('tab', { name: 'Close-up' }).click();
check((await pos()) === 'Close-up · 3 of 4', 'viewer: a tab jumps to Close-up · 3 of 4');
await viewer.getByRole('button', { name: 'Close' }).click();
check(await until(async () => !(await viewer.isVisible().catch(() => false)), 3000), 'viewer: the close button closes it');
check((await activeLabel()) === `${first.name}, Front: open full screen`, 'viewer: focus returns to the sheet view that opened it');

// --- Failed, then Try again by cast_id -------------------------------------------------------------------------------------
if (FAIL_FLAG) writeFileSync(FAIL_FLAG, '1');
await page.getByRole('radio', { name: second.name, exact: true }).click();
await primary('Make sheet').click();
const tryAgain = primary('Try again');
check(await until(() => tryAgain.isVisible(), 30000), 'failed: the primary is "Try again"');
check((await page.getByText('Failed', { exact: true }).count()) > 0, 'failed: the panel says Failed');
check((await page.getByRole('img', { name: 'Back: failed' }).count()) === 1, 'failed: the views it never made say Failed, not a black box');
check(!(await rail().innerText()).includes('Try again.'), 'failed: the error line has no second "Try again."');
await shoot('character_sheet_failed');
// Advanced: "Use without a sheet" (Make sheet stays the primary), and the fold scrolls clear of the pinned primary.
await page.getByRole('button', { name: 'Advanced', exact: true }).click();
const without = page.getByRole('button', { name: 'Use without a sheet', exact: true });
check(await until(() => without.isVisible(), 3000), 'failed: Advanced has "Use without a sheet"');
await scrollRail(-1);
await page.waitForTimeout(500);
const footTop = (await tryAgain.boundingBox())?.y ?? 0;
const lastRow = page.getByRole('switch').last().or(page.getByRole('button', { name: 'Prompt' }).last()).last();
const lastBox = await lastRow.boundingBox().catch(() => null);
check(!!lastBox && lastBox.y + lastBox.height <= footTop - 8, `advanced: its last row scrolls clear of the pinned primary (${Math.round(lastBox?.y + lastBox?.height || 0)} < ${Math.round(footTop)})`);
check(await tabsShown(), 'advanced: the step tabs stay on screen');
await shoot('character_advanced_scrolled');
await page.getByRole('button', { name: 'Advanced', exact: true }).click();
await scrollRail(0);
const before = sheetPosts.length;
await tryAgain.click();
check(await until(async () => sheetPosts.length > before, 5000) && sheetPosts.at(-1).cast_id === second.id && !sheetPosts.at(-1).name && typeof sheetPosts.at(-1).seed === 'number', 'try again: resubmits with cast_id (and a new seed), never a name');
check(await until(() => primary('Making the sheet').isVisible(), 10000), 'try again: making again');

// --- New: words, 4 options, the viewer on an option, pick, name -----------------------------------------------------------
await page.getByRole('radio', { name: 'New', exact: true }).click();
const words = page.getByRole('textbox', { name: 'Who are they?' });
await words.fill('');
check(await primary('Make 4 options').isDisabled(), 'new: empty, Make 4 options waits for words');
await shoot('character_new_empty');
await words.fill('A knight in shining armour, with a red cloak');
const make4 = primary('Make 4 options');
check(await make4.isEnabled(), 'new: the primary is "Make 4 options"');
await make4.click();
check(await until(async () => (await page.getByRole('status', { name: /Option 1: being made/ }).count()) > 0, 5000), 'new: 4 tiles show while the options are made');
await shoot('character_new_options_loading');
const jobsNow = await api('/api/jobs');
check(jobsNow.filter((j) => j.task === 'image' && j.params?.count === 4).length >= 1, 'new: the 4 options are one image job with count 4');
check(await until(async () => (await page.getByRole('radio', { name: 'Option 4' }).count()) > 0, 30000), 'new: 4 options came back');
await page.getByRole('radio', { name: 'Option 2' }).click();
const name = page.getByRole('textbox', { name: 'Name' });
check((await name.inputValue()).startsWith('Knight'), `new: the name comes from the words (${await name.inputValue()})`);
const autoName = (await name.inputValue()).toLowerCase();
check(!(await castNow()).some((c) => c.name.toLowerCase() === autoName), `new: the auto name is free, the next free number (${autoName})`);
await shoot('character_new_options');

const openOpt = page.getByRole('button', { name: 'Open option 1 full screen' });
await openOpt.focus();
await openOpt.press('Enter');
const optViewer = page.getByRole('dialog', { name: /^New character:/ });
const optPos = async () => ((await optViewer.getAttribute('aria-label').catch(() => '')) || '').replace(/^[^:]*:\s*/, '');
check(await until(async () => (await optPos()) === 'Option 1 · 1 of 4', 3000), 'viewer: an option is "Option 1 · 1 of 4" (for screen readers)');
await shoot('viewer_option');
await page.keyboard.press('ArrowRight');
await page.keyboard.press('ArrowRight');
check((await optPos()) === 'Option 3 · 3 of 4', 'viewer: arrows move between the options');
await optViewer.getByRole('button', { name: 'Use this one' }).click();
check(await until(async () => !(await optViewer.isVisible().catch(() => false)), 3000), 'viewer: Use this one closes it');
check((await page.getByRole('radio', { name: 'Option 3' }).getAttribute('aria-checked')) === 'true', 'viewer: Use this one picks Option 3');
check((await activeLabel()) === 'Open option 1 full screen', 'viewer: focus returns to the option that opened it');

// --- the draft survives a reload and a switch of shot and task ---------------------------------------------------------------
await page.reload({ waitUntil: 'domcontentloaded' });
await openRecast();
const back3 = page.getByRole('radio', { name: 'Option 3' });
check(await until(async () => (await back3.getAttribute('aria-checked').catch(() => null)) === 'true', 15000), 'draft: after a reload, the same step with the 4 options and the pick');
await openRecast(other.id);
await page.waitForTimeout(800);
await openRecast();
check(await until(async () => (await page.getByRole('radio', { name: 'Option 3' }).getAttribute('aria-checked').catch(() => null)) === 'true', 15000), 'draft: after another shot, back to the same pick');
check((await page.getByRole('textbox', { name: 'Who are they?' }).inputValue()).startsWith('A knight'), 'draft: the words are still there');

// --- Make sheet from New, then Use <name> ----------------------------------------------------------------------------------
const knight = await name.inputValue();
await primary('Make sheet').click();
check(await until(() => primary('Making the sheet').isVisible(), 15000), 'new: Make sheet adds them to the cast and makes the sheet');
const useKnight = primary(`Use ${knight}`);
check(await until(() => useKnight.isVisible(), 60000), `new: ready, the primary is "Use ${knight}"`);
const made = (await castNow()).find((c) => c.name === knight);
check(!!made?.sheet, `new: ${knight} is in the cast with a sheet`);
check(sheetPosts.some((b) => b.cast_id === made?.id), 'new: its sheet went by cast_id');

// --- 3 Render, from the viewer's Use <name> ------------------------------------------------------------------------------------
await page.getByRole('button', { name: `${knight}, Front: open full screen` }).click();
const knightViewer = page.getByRole('dialog', { name: new RegExp(`^${knight}:`) });
const useInViewer = knightViewer.getByRole('button', { name: `Use ${knight}`, exact: true });
check(await until(() => useInViewer.isEnabled().catch(() => false), 5000), `viewer: "Use ${knight}" on the new character's sheet`);
await useInViewer.click();
check(await until(async () => !(await knightViewer.isVisible().catch(() => false)), 3000), 'viewer: Use closes it');
check(await until(async () => (await page.getByRole('list', { name: 'Stages' }).count()) > 0, 15000), 'render: the job\'s own stages');
const rj = (await api('/api/jobs')).find((j) => j.task === 'recast' && j.shot_id === shot.id);
check(!!rj && rj.params.cast_id === made?.id && rj.params.engine === 'ltx' && rj.params.frame != null && rj.params.points?.length > 0, 'render: the job is Person 1 with the new character, LTX-2.5');
check(!!rj && rj.draft_key === `${shot.id}:recast` && rj.step === 'render' && !('draft_key' in rj.params) && !('step' in rj.params), 'render: the draft key is on the job, not in params');
check(await inViewport(page.getByRole('button', { name: 'Back to the shot' })), 'render: "Back to the shot" is on screen');
await shoot('render');

// --- 4 Review ------------------------------------------------------------------------------------------------------------------
// The rail's primary is the one Approve (the player bar leaves its own to it while Review shows).
const approve = page.getByRole('button', { name: 'Approve', exact: true }).last();
if (await approve.waitFor({ state: 'visible', timeout: 150000 }).then(() => true).catch(() => false)) {
  check((await page.getByRole('button', { name: 'Try again' }).count()) > 0 && (await page.getByRole('button', { name: 'Note' }).count()) > 0, 'review: Try again, Note and Approve');
  await page.waitForTimeout(500);
  check((await page.getByRole('button', { name: 'Approve', exact: true }).count()) === 1, 'review: one Approve on screen');
  check((await page.getByText(/^On take \d/).count()) === 0, 'review: no "On take" chip under the title');
  await shoot('review');
} else {
  check(false, 'review: reached');
  console.log('NOTE  draft', await page.evaluate((k) => JSON.stringify((JSON.parse(localStorage.getItem('pickups.drafts.v1') || '{}')[k] || {}).step), `${shot.id}:recast`));
  await shoot('review_missed');
}

// --- the cast page: sheet states and New character ------------------------------------------------------------------------------
await page.goto(`${BASE}/#/project/${pid}/cast`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
check((await page.getByRole('button', { name: 'Make sheet', exact: true }).count()) > 0 && (await page.getByRole('button', { name: 'Redo sheet', exact: true }).count()) > 0, 'cast page: Make sheet and Redo sheet are buttons');
check((await page.getByText(/Type @|Drop a photo or sheet, or describe/).count()) === 0, 'cast page: no taglines');
check((await page.getByText('person', { exact: true }).count()) === 0, 'cast page: no PERSON badges');
check((await page.getByRole('button', { name: 'New character' }).count()) === 1, 'cast page: one New character');
await shoot('cast_page_states');
const card = page.getByRole('button', { name: `${first.name}: open full screen` });
await card.click();
check(await until(async () => (await page.getByRole('dialog', { name: new RegExp(`^${first.name}:`) }).count()) > 0, 3000), 'cast page: a card opens the same full-screen viewer');
await page.keyboard.press('Escape');
await page.getByRole('button', { name: 'New character' }).first().click();
const modal = page.getByRole('dialog', { name: 'New character' });
check(await modal.isVisible().catch(() => false), 'cast page: New character opens the Character flow');
await modal.getByRole('textbox', { name: 'Who are they?' }).fill('An astronaut in a white suit');
await shoot('cast_page_new_character');

await browser.close();
console.log(fails ? `FAILED (${fails})` : 'ALL PASS');
process.exit(fails ? 1 : 0);
