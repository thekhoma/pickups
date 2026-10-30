// A judge's first five minutes, in the real app (Playwright): Previz, the Window Seat sentence, Make shot, the Look
// preview and Try another, then queue Previz to final.
//
//   PREVIZ_URL=http://127.0.0.1:7841 PICKUPS_HOME=<that server's home> npm run test:previz:first5 -- --out <dir>
//
// No GPU job runs unless --gpu is passed (only with the owner's OK and the GPU free). Without it, the two GPU calls are held
// in the browser: the look previews come back as stand-in pictures made from the playblast's own first frame, and the
// queued job is recorded and its parameters asserted instead of being sent. The render graph is then built locally
// (first_five_graph.py) to show how many frames reach it.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const opt = (name, fallback) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] : fallback);
const BASE = process.env.PREVIZ_URL || 'http://127.0.0.1:7841';
const OUT = resolve(opt('--out', join(REPO, 'web/test-results/first_five')));
const GPU = flag('--gpu');
const SENTENCE = 'A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and slowly pushes in, and at the end he turns and looks at the camera.';
mkdirSync(OUT, { recursive: true });

const log = [];
const say = (line) => {
  const stamp = new Date().toISOString().slice(11, 19);
  log.push(`${stamp} ${line}`);
  console.log(`${stamp} ${line}`);
};
let failures = 0;
const check = (ok, what) => {
  say(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) failures++;
};

const browser = await chromium.launch({ channel: 'chrome', headless: !flag('--headed'), args: ['--use-angle=metal', '--enable-gpu'] });
// The QC header sends the made shot to the archived Test runs project, never a person's projects.
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, extraHTTPHeaders: { 'X-Pickups-QC': '1' } });
await ctx.addInitScript(() => {
  try {
    localStorage.setItem('pickups.chrome', 'top');
  } catch { /* private mode */ }
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));

// --- the GPU fence -------------------------------------------------------------------------------------------------
const looks = [];
const lookCalls = [];
const queued = [];
const SAFE_POST = [/\/api\/previz\/pv_[0-9a-f]+\/(frames|finish)$/, /\/api\/previz\/framing$/, /\/api\/uploads$/, /\/api\/projects$/, /\/api\/mentions\/preview$/];
await page.route('**/api/**', async (route) => {
  const req = route.request();
  const url = new URL(req.url());
  if (req.method() === 'GET' || req.method() === 'DELETE' || SAFE_POST.some((r) => r.test(url.pathname))) return route.continue();
  const body = (() => {
    try {
      return req.postDataJSON();
    } catch {
      return null;
    }
  })();
  // With --gpu the real requests go through; their bodies are still recorded and asserted.
  if (GPU) {
    if (url.pathname === '/api/previz/look') lookCalls.push(body);
    if (url.pathname === '/api/jobs') queued.push(body);
    return route.continue();
  }
  if (url.pathname === '/api/previz/look') {
    lookCalls.push(body);
    const look = looks[(lookCalls.length - 1) % looks.length];
    return route.fulfill({ json: { url: look.url, upload_id: look.id, seed: body?.seed, references: (body?.look_upload_ids?.length ?? 0) + (body?.look_refs?.length ?? 0) } });
  }
  if (url.pathname === '/api/jobs') {
    queued.push(body);
    return route.fulfill({ json: { id: 'job_first_five', shot_id: body?.shot_id, task: body?.task, status: 'queued', stage: 'Queued (not sent: no GPU OK)', stages: [], progress: 0, eta_s: null, params: body?.params ?? {} } });
  }
  say(`held a POST the run doesn't need: ${url.pathname}`);
  return route.fulfill({ status: 503, json: { detail: 'Held by first_five (no GPU OK).' } });
});

const shot = async (name) => {
  const file = join(OUT, name);
  await page.screenshot({ path: file });
  say(`screenshot ${name}`);
};

// --- 1. Previz, type the sentence, Build ---------------------------------------------------------------------------
const t0 = Date.now();
await page.goto(`${BASE}/#/previz`, { waitUntil: 'domcontentloaded' });
const box = page.getByRole('textbox', { name: 'Describe the shot' });
await box.waitFor({ state: 'visible', timeout: 30000 });
await page.waitForFunction(() => (window.__previzTiles?.built ?? 0) >= 11, null, { timeout: 45000 }).catch(() => say('not every starter tile had built after 45 s'));
await shot('01_gate.png');
check(await page.getByRole('heading', { name: 'Examples' }).count() === 1 && await page.getByRole('heading', { name: 'Your previz' }).count() === 0, 'gate shows Examples, and no Your previz row before the first one');
check(await page.getByRole('radio', { name: /New previz/ }).getAttribute('aria-checked') === 'true', 'gate has the shared composer on New previz');
await box.click();
await box.fill(SENTENCE);
await page.getByRole('button', { name: 'Build', exact: true }).click();
await page.waitForTimeout(300);
check((await page.getByRole('dialog').count()) === 0, 'Build shows the scene with no project popup');
await page.locator('canvas[aria-label="Previz preview"]').waitFor({ state: 'visible', timeout: 20000 });
await page.waitForFunction(() => !!window.__previzShot?.label, null, { timeout: 30000 });
await page.waitForTimeout(1200);
const label = await page.evaluate(() => window.__previzShot?.label);
say(`built: ${label}`);
await shot('02_editor.png');

// --- 2. Make shot ---------------------------------------------------------------------------------------------------
await page.getByRole('button', { name: 'Make shot' }).click();
// Save to (My shots, a project or a new one): the dialog's own Make shot.
await page.getByRole('dialog', { name: 'Save the shot to' }).getByRole('button', { name: 'Make shot' }).click();
await page.waitForTimeout(900);
await shot('03_make_shot.png');
await page.waitForURL(/#\/shot\//, { timeout: 120000 });
const shotId = decodeURIComponent(page.url().split('#/shot/')[1].split(/[/?]/)[0]);
const made = await page.evaluate(async (id) => (await fetch(`/api/shots/${encodeURIComponent(id)}`)).json(), shotId);
say(`made shot ${shotId}: ${made.frames} frames, ${made.width}x${made.height}, project ${made.project_id}`);
check(made.frames === 97 && made.width === 1920 && made.height === 1088, 'the playblast is 97 frames at 1920x1088');
check(made.project_id === 'qc', 'the test shot is saved to the archived Test runs project');

// Stand-in look pictures (no GPU): the playblast's first frame, warmed three different ways, uploaded to the app.
if (!GPU) {
  const made3 = await page.evaluate(async (thumb) => {
    const img = new Image();
    img.src = thumb;
    await img.decode();
    const out = [];
    for (const [r, g, b] of [[60, 30, 0], [20, 10, 40], [0, 30, 20]]) {
      const c = document.createElement('canvas');
      c.width = 1920;
      c.height = 1088;
      const x = c.getContext('2d');
      x.drawImage(img, 0, 0, 1920, 1088);
      x.globalCompositeOperation = 'overlay';
      x.fillStyle = `rgb(${128 + r}, ${128 + g}, ${128 + b})`;
      x.fillRect(0, 0, 1920, 1088);
      const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9));
      const fd = new FormData();
      fd.append('file', blob, 'standin_look.jpg');
      out.push(await (await fetch('/api/uploads', { method: 'POST', body: fd })).json());
    }
    return out;
  }, made.thumb_url);
  looks.push(...made3);
}

// --- 3. Look prefilled, preview, Try another (twice) ---------------------------------------------------------------
const lookField = page.getByRole('textbox', { name: 'Look', exact: true });
await lookField.waitFor({ state: 'visible', timeout: 20000 });
await page.waitForTimeout(800);
const lookText = await lookField.inputValue();
say(`Look prompt: "${lookText}"`);
check(lookText.startsWith('It is night. The grey figure is a man sitting alone in a booth in a diner at night'), 'the Look field leads with the time, then describes the man in the diner');
await lookField.scrollIntoViewIfNeeded();
await shot('04_look_prefilled.png');
const lookSeconds = [];
let lookT = Date.now();
await page.getByRole('button', { name: 'Make start frame from previz' }).click();
const seedText = page.getByTestId('look-seed');
await seedText.waitFor({ state: 'visible', timeout: 120000 });
lookSeconds.push((Date.now() - lookT) / 1000);
const seeds = [await seedText.innerText()];
await page.waitForTimeout(500);
await shot('05_preview.png');
for (const n of [6, 7]) {
  lookT = Date.now();
  await page.getByRole('button', { name: 'Try another' }).click();
  await page.waitForFunction((prev) => {
    const el = document.querySelector('[data-testid="look-seed"]');
    return el && el.textContent !== prev;
  }, seeds[seeds.length - 1], { timeout: 120000 });
  lookSeconds.push((Date.now() - lookT) / 1000);
  seeds.push(await seedText.innerText());
  await page.waitForTimeout(500);
  await shot(n === 6 ? '06_try_another.png' : '06b_try_another_again.png');
}
say(`seeds: ${seeds.join(', ')}`);
check(new Set(seeds).size === 3, 'Try another twice: three different seeds');
say(`look previews: ${lookSeconds.map((x) => x.toFixed(1)).join(' s, ')} s`);
check(lookCalls.length === 3 && new Set(lookCalls.map((c) => c?.seed)).size === 3, 'three look requests, each with its own seed');
check(lookCalls.length === 3 && lookCalls.every((c) => c?.look_prompt?.startsWith(lookText) && c.look_prompt.includes("Add nothing that isn't in the frame.")), 'every start frame request carries the Look prompt, then only what frame 0 shows');

// --- 4. The made start frame is pinned; queue Previz to final -------------------------------------------------------
await page.waitForTimeout(1500);
const drift = await page.getByText("doesn't match the previz framing").count();
check(drift === 0, 'a look frame painted over frame 0 raises no framing warning');
await page.getByRole('complementary', { name: 'Tasks' }).getByRole('button', { name: 'Previz to final' }).click();
await page.waitForTimeout(500);
check((await page.getByRole('dialog').count()) === 0, 'Previz to final queues with no popup');
const renderT = Date.now();
await page.waitForTimeout(1500);
await shot('07_queue.png');
let graph = null;
if (!GPU) {
  check(queued.length === 1, `one job queued (held, not sent): ${queued.length}`);
  const job = queued[0] ?? {};
  const p = job.params ?? {};
  say(`queued: task ${job.task}, shot ${job.shot_id}, params ${JSON.stringify(p)}`);
  check(job.task === 'previz' && job.shot_id === shotId, 'the job is Previz to final on the made shot');
  check(typeof p.look_frame_upload_id === 'string' && p.look_frame_upload_id === looks[2]?.id, 'the chosen look frame (the third preview) is the job\'s look frame');
  check(p.pin_start === true && (p.look_at ?? 0) === 0, 'pinned first frame, look at frame 0');
  check(p.look_prompt === undefined && p.look_upload_ids === undefined && p.look_refs === undefined, 'with a start frame no references or look words ride along');
  if (process.env.PICKUPS_HOME) {
    const r = spawnSync('uv', ['run', 'python', 'web/tests/previz/first_five_graph.py', shotId, JSON.stringify(p)], { cwd: REPO, env: process.env, encoding: 'utf8' });
    try {
      graph = JSON.parse(r.stdout.trim().split('\n').pop());
    } catch {
      say(`graph dry run failed: ${r.stderr.slice(-400)}`);
    }
    if (graph) say(`render graph (built, not sent): layout clip ${graph.graph_layout_frames} frames, ${graph.graph_nodes} nodes, guides ${graph.guides.join(' ')}`);
    check(graph?.graph_layout_frames === 97, '97 frames reach the render graph');
  } else say('PICKUPS_HOME not set: render graph dry run skipped');
}
let render = null;
if (GPU) {
  // The real render: wait for the job (10 min at most), then check the take it made.
  check(queued.length === 1 && queued[0]?.task === 'previz' && queued[0]?.shot_id === shotId, 'one Previz to final job sent for the made shot');
  let job = null;
  while (Date.now() - renderT < 600000) {
    const jobs = await page.evaluate(async () => (await fetch('/api/jobs')).json());
    job = (Array.isArray(jobs) ? jobs : jobs.jobs ?? []).filter((j) => j.shot_id === shotId && j.task === 'previz').pop() ?? null;
    if (job && (job.status === 'done' || job.status === 'failed')) break;
    await page.waitForTimeout(5000);
  }
  const renderSeconds = (Date.now() - renderT) / 1000;
  say(`render: ${job?.status ?? 'not found'} after ${renderSeconds.toFixed(0)} s${job?.error ? ': ' + job.error : ''}`);
  check(job?.status === 'done', 'the render finished (done, within 10 minutes)');
  check(typeof job?.result_version_id === 'string', 'the job names the take it made');
  let probe = null;
  if (job?.result_version_id) {
    const take = await page.evaluate(async ([id, vid]) => (await (await fetch(`/api/shots/${encodeURIComponent(id)}`)).json()).versions.find((v) => v.id === vid), [shotId, job.result_version_id]);
    const local = process.env.PICKUPS_HOME ? join(process.env.PICKUPS_HOME, 'library/shots', shotId, 'versions', job.result_version_id, 'master.mov') : null;
    let file = local && existsSync(local) ? local : null;
    if (!file && take?.downloads?.[0]?.url) {
      file = join(OUT, 'final_take' + (take.downloads[0].url.match(/\.\w+$/)?.[0] ?? '.mov'));
      writeFileSync(file, Buffer.from(await (await fetch(new URL(take.downloads[0].url, BASE))).arrayBuffer()));
    }
    if (file) {
      const r = spawnSync('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', file], { encoding: 'utf8' });
      try {
        const st = JSON.parse(r.stdout).streams[0];
        probe = { file, width: st.width, height: st.height, frames: Number(st.nb_read_frames) };
      } catch {
        say(`ffprobe failed: ${r.stderr.slice(-300)}`);
      }
    }
  }
  say(`final: ${probe ? `${probe.frames} frames at ${probe.width}x${probe.height}` : 'no video found'}`);
  check(probe?.frames === 97 && probe?.width === 1920 && probe?.height === 1088, 'the final is a 97-frame 1920x1088 video');
  render = { status: job?.status ?? null, result_version_id: job?.result_version_id ?? null, seconds: renderSeconds, video: probe };
}
check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.join(' | ') : ''}`);
say(`first five: ${((Date.now() - t0) / 1000).toFixed(1)} s, ${failures ? failures + ' failed' : 'all passed'}; GPU job ${GPU ? 'allowed (--gpu)' : 'not run (no OK): look previews stood in, job held'}`);
writeFileSync(join(OUT, 'first_five.json'), JSON.stringify({ shotId, made: { frames: made.frames, width: made.width, height: made.height, project: made.project_id }, label, lookText, seeds, lookCalls, lookSeconds, queued, graph, render, renderSeconds: render?.seconds ?? null, failures, gpu: GPU, log }, null, 2));
writeFileSync(join(OUT, 'first_five.log'), log.join('\n') + '\n');
await browser.close();
process.exit(failures ? 1 : 0);
