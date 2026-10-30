// Step 21 (SPEC sections 1, 2, 4): the step flows send exactly the old forms' params for the same picks, and drafts keep
// what was typed across a switch of shot and task (and a reload), capped at 50 and 14 days, a prefill winning.
import { readFileSync } from 'node:fs';
import { finishParams, relightParams, removeParams, RELIGHT_DEFAULT, RELIGHT_FROM, RELIGHT_LOOKS } from '../../src/lib/flowParams.ts';
import { paramsOf } from '../../src/lib/finish.ts';
import { lastInlineAt, newestInline } from '../../src/lib/inlineJobs.ts';
import { DRAFTS_KEY, draftKey, loadDrafts, MAX_DRAFTS, putDraft, saveDrafts, seedDraft, DRAFT_TTL_MS } from '../../src/lib/draftStore.ts';

// The old forms' own construction, as it was at 2e0b2e9 (Forms.tsx RemoveForm.submit, FinishForm, SceneRelightForm).
function oldRemove(sel, shadow, bracket, pick, pre) {
  const params = { points: sel.points, ...(sel.text ? { text: sel.text } : {}), shadow, frame: sel.frame };
  if (bracket.show && bracket.win) {
    params.start_frame = bracket.win.start;
    if (bracket.win.sized) params.frames = bracket.win.frames;
  }
  const same = !!pick && sel.frame === pick.frame && sel.text === pick.text && JSON.stringify(sel.points) === JSON.stringify(pick.points);
  if (same && typeof pre?.fill === 'string' && pre.fill) params.fill = pre.fill;
  if (typeof pre?.seed === 'number') params.seed = pre.seed;
  return params;
}
const oldRelight = (look, custom, from) => (look === 'custom' ? { look, custom: custom.trim(), direction: from } : { look });
const oldFinish = (pick, quality) => ({ ...paramsOf(pick), quality });

export function runFlowParams() {
  const problems = [];
  const eq = (a, b, what) => { if (JSON.stringify(a) !== JSON.stringify(b)) problems.push(`${what}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };

  // Remove: Person 1 by the finder's points, words, a sized and an unsized bracket, a take's fill and seed.
  const p1 = { points: [{ x: 410, y: 300, positive: true }, { x: 412, y: 210, positive: true }], text: '', frame: 12 };
  const words = { points: [], text: 'boom mic', frame: 40 };
  const win = { start: 8, frames: 97, sized: true };
  for (const [sel, shadow, bracket, pick, pre, what] of [
    [p1, true, { show: false, win: null }, null, null, 'person 1, shadow, no bracket'],
    [p1, false, { show: true, win }, null, null, 'no shadow, a sized bracket'],
    [words, true, { show: true, win: { start: 0, frames: 121, sized: false } }, null, { seed: 7 }, 'words, an unsized bracket, a seed'],
    [p1, true, { show: false, win: null }, { ...p1, from: 'the last take' }, { fill: 'grass and path', seed: 3 }, 'the take\'s pick: its fill words'],
    [words, true, { show: false, win: null }, { ...p1, from: 'the last take' }, { fill: 'grass and path' }, 'a changed pick: no fill words'],
  ]) {
    const same = !!pick && sel.frame === pick.frame && sel.text === pick.text && JSON.stringify(sel.points) === JSON.stringify(pick.points);
    const flow = removeParams(sel, shadow, bracket.show && bracket.win ? bracket.win : null, { fill: same ? pre?.fill || undefined : undefined, seed: typeof pre?.seed === 'number' ? pre.seed : undefined });
    eq(flow, oldRemove(sel, shadow, bracket, pick, pre), `remove: ${what}`);
  }
  eq(removeParams(p1, true, null).shadow, true, 'remove: "Include shadow" is the param shadow');

  // Relight: each tested look by its key; your own words with each direction; the old form's defaults.
  for (const l of RELIGHT_LOOKS) eq(relightParams(l.id, '', 'left_low'), oldRelight(l.id, '', 'left_low'), `relight: ${l.label}`);
  for (const d of RELIGHT_FROM) eq(relightParams('custom', ' soft warm evening light ', d.value), oldRelight('custom', ' soft warm evening light ', d.value), `relight: custom from ${d.label}`);
  eq(RELIGHT_DEFAULT, { look: 'sunset', direction: 'front_mid' }, 'relight: the form\'s starting picks');
  eq(RELIGHT_LOOKS.map((l) => l.id), ['sunset', 'golden', 'cool', 'midday', 'afternoon'], 'relight: scene_relight.py LOOKS keys');
  eq(RELIGHT_FROM.map((d) => d.value), ['behind_low', 'left_low', 'back_right_low', 'front_right_high', 'front_mid'], 'relight: DIRECTIONS keys');

  // Finish: each choice with each quality; Best unless changed.
  for (const c of ['4k-hdr', 'hdr', '4k']) for (const q of ['best', 'fast']) eq(finishParams(c, q), oldFinish(c, q), `finish: ${c} ${q}`);

  // Run 9: each flow builds its job from the same function the old form did, and is the only version (no flag).
  const src = readFileSync(new URL('../../src/components/Forms.tsx', import.meta.url), 'utf8');
  const body = (name) => { const i = src.indexOf(`function ${name}(`); const j = src.indexOf('\nfunction ', i + 10); return i < 0 ? '' : src.slice(i, j < 0 ? undefined : j); };
  for (const [form, flow, fn] of [['RemoveForm', 'RemoveFlow', 'removeParams('], ['SceneRelightForm', 'RelightFlow', 'relightParams('], ['FinishForm', 'FinishFlow', 'finishParams(']]) {
    if (body(form)) problems.push(`${form} is still there (the flow is the only version)`);
    if (!body(flow).includes(fn)) problems.push(`${flow} doesn't build its params with ${fn}`);
  }
  for (const task of ['remove', 'relight', 'finish']) if (src.includes(`flowOn("${task}")`)) problems.push(`${task} is still behind flowOn("${task}")`);
  // Finish's Quality is in the flow's Advanced and starts at Best, as the form.
  if (!/quality: pre\?\.quality === "fast" \? "fast" : "best"/.test(body('FinishFlow'))) problems.push('the Finish flow doesn\'t start at Best');

  // --- drafts (lib/draftStore.ts) ---
  const mem = new Map();
  const storage = { getItem: (k) => (mem.has(k) ? mem.get(k) : null), setItem: (k, v) => mem.set(k, v) };
  const t0 = 1_800_000_000_000;
  // The repro: words typed in Previz on shot A, then shot B's Remove, then back to A's Previz (and a reload).
  let all = seedDraft({}, draftKey('shot_a', 'previz'), '', { look: '', motion: 'The camera pushes in.', accepted: null, seed: null }, t0);
  all = putDraft(all, draftKey('shot_a', 'previz'), { data: { ...all['shot_a:previz'].data, look: 'Warm dusk, wet cobbles.', accepted: { upload_id: 'up_1', url: '/u/1.png' }, seed: 42 } }, t0 + 1000);
  all = seedDraft(all, draftKey('shot_b', 'remove'), 'who', { shadow: true }, t0 + 2000);
  all = putDraft(all, draftKey('shot_b', 'remove'), { data: { shadow: false } }, t0 + 3000);
  const back = all['shot_a:previz'];
  eq([back.data.look, back.data.motion, back.data.accepted, back.data.seed], ['Warm dusk, wet cobbles.', 'The camera pushes in.', { upload_id: 'up_1', url: '/u/1.png' }, 42], 'drafts: Previz words and frame survive a switch');
  saveDrafts(storage, all);
  const reloaded = loadDrafts(storage, t0 + 5000);
  eq(reloaded['shot_a:previz'].data.look, 'Warm dusk, wet cobbles.', 'drafts: and a reload');
  eq(mem.has(DRAFTS_KEY), true, 'drafts: localStorage pickups.drafts.v1');
  // A prefill (Try again with changes) starts fresh and wins over the stale draft.
  const fresh = seedDraft(reloaded, 'shot_a:previz', '', { look: 'From the take', motion: '', accepted: null, seed: null }, t0 + 6000);
  eq(fresh['shot_a:previz'].data.look, 'From the take', 'drafts: a prefill wins');
  // 50 at most, the most recently touched kept; nothing older than 14 days.
  let many = {};
  for (let i = 0; i < 60; i++) many = putDraft(many, `s${i}:remove`, { data: { i } }, t0 + i);
  eq(Object.keys(many).length, MAX_DRAFTS, 'drafts: capped at 50');
  eq(['s59:remove' in many, 's0:remove' in many], [true, false], 'drafts: least recently used goes first');
  eq(Object.keys(loadDrafts({ getItem: () => JSON.stringify({ old: { step: '', data: {}, outputs: {}, jobs: {}, updatedAt: t0 } }) }, t0 + DRAFT_TTL_MS + 1)), [], 'drafts: 14 days');
  eq(loadDrafts({ getItem: () => '{not json' }), {}, 'drafts: a broken store is empty, never a crash');

  // Step 22 (QA's QC, a false "The start frame didn't come out"): after a failed try, the next try (and a re-attach
  // after a reload) only takes an inline row made after its click, never the older failed one.
  const key = 'shot_a:previz';
  const row = (id, created_at, status, extra = {}) => ({ id, lane: 'inline', draft_key: key, step: 'start_frame', created_at, status, stage: 'Start frame', stages: ['Start frame'], progress: 0, eta_s: null, ...extra });
  const failed = row('job_old', 1000.5, 'failed', { error: "The start frame didn't come out. Check that ComfyUI is running, then try again." });
  const other = { ...row('job_other', 1500, 'running'), draft_key: 'shot_b:previz' };
  let jobs = { job_old: failed, job_other: other };
  const after = lastInlineAt(jobs, key, 'start_frame');
  eq(after, 1000.5, 'start frame: the click remembers the newest row so far');
  eq(newestInline(jobs, key, 'start_frame', after), undefined, 'start frame: the last failed try is never the new one (no false error)');
  jobs = { ...jobs, job_new: row('job_new', 1001.2, 'running') };
  eq(newestInline(jobs, key, 'start_frame', after)?.id, 'job_new', 'start frame: the new try\'s row, running');
  jobs = { ...jobs, job_new: row('job_new', 1001.2, 'done', { result: { upload_id: 'up_2', url: '/u/2.png', seed: 9 } }) };
  eq(newestInline(jobs, key, 'start_frame', after)?.result, { upload_id: 'up_2', url: '/u/2.png', seed: 9 }, 'start frame: re-attach after a reload finds it done, by draft key');
  eq(newestInline(jobs, 'shot_b:previz', 'start_frame', 0)?.id, 'job_other', 'start frame: another draft\'s row is its own');
  eq(newestInline({ job_old: failed }, key, 'start_frame', null)?.id, 'job_old', 'start frame: with no cutoff the newest row is found');

  return { fail: problems.length, problems, line: `flow params and drafts: ${problems.length ? 'FAIL' : 'pass'} (remove, relight, finish same as the forms; drafts survive a switch and a reload, 50, 14 days, prefill wins; no false start-frame error)` };
}
