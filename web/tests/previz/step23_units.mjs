// Step 23 (SPEC section 4, Replace a person and Character): the recast flow sends exactly the old form's params for the
// same picks; the sheet's state, its count ("2 of 3") and its one primary per state; Try again goes by cast_id; the
// viewer's position and arrows; a recast draft's picks and New character's outputs survive a reload.
import { recastParams } from '../../src/lib/flowParams.ts';
import { SHEET_STAGES, sheetError, sheetJobFor, sheetLabel, sheetPrimary, sheetRequest, sheetStateOf, sheetStep, sheetViews } from '../../src/lib/sheetState.ts';
import { stepIndex, viewerPosition } from '../../src/lib/viewer.ts';
import { freeName } from '../../src/lib/castName.ts';
import { draftKey, loadDrafts, putDraft, saveDrafts } from '../../src/lib/draftStore.ts';

// The old form's own construction, as it was at 55be15d (Forms.tsx RecastForm.doSubmit, one person), plus Run 10's
// kind "person" (Replace a person never asks whether a click is a car).
function oldRecast(sel, eng, source, o) {
  const params = { points: sel.points, ...(sel.text ? { text: sel.text } : {}), kind: 'person', frame: sel.frame, engine: eng };
  if (source.cast) params.cast_id = source.cast;
  else if (source.image) params.who_ref = source.image;
  else if (source.photo) params.ref_upload_id = source.photo;
  else if (source.kept) params.ref_upload_id = source.kept;
  else if (source.madeTake) params.reference_take = source.madeTake;
  else if (source.still) params.who_ref = { kind: 'image', id: source.still };
  if (source.madeCaption) params.reference_prompt = source.madeCaption;
  if (o.describe) params.describe = o.describe;
  if (o.relight) params.relight = true;
  if (o.same && o.pre) {
    if (o.keptPrompt) params.prompt = o.keptPrompt;
    if (o.pre.scene) params.scene = o.pre.scene;
    if (typeof o.pre.seed === 'number') params.seed = o.pre.seed;
  }
  if (eng === 'ltx' && o.showWindow && o.win) {
    params.start_frame = o.win.start;
    if (o.win.sized) params.frames = o.win.frames;
  }
  return params;
}

const job = (id, status, stage, extra = {}) => ({ id, task: 'sheet', status, stage, stages: Object.values(SHEET_STAGES), progress: 0.4, eta_s: 20, created_at: 1000, params: { cast_id: 'cast_rosa' }, ...extra });

export function runStep23Units() {
  const problems = [];
  const eq = (a, b, what) => { if (JSON.stringify(a) !== JSON.stringify(b)) problems.push(`${what}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };

  // --- the recast flow's params are the old form's for the same picks ---
  const p1 = { points: [{ x: 410, y: 300, positive: true }, { x: 412, y: 210, positive: true }], text: '', frame: 12 };
  const words = { points: [], text: 'the man in the coat', frame: 30 };
  const win = { start: 8, frames: 97, sized: true };
  for (const [sel, eng, cast, o, what] of [
    [p1, 'ltx', 'cast_rosa', {}, 'Person 1 with Rosa, LTX-2.5, no bracket'],
    [p1, 'ltx', 'cast_rosa', { relight: true, showWindow: true, win }, 'Relight the new person and a sized bracket'],
    [words, 'scail', 'cast_rosa', { showWindow: true, win }, 'SCAIL-2: no bracket is sent'],
    [p1, 'ltx', 'cast_rosa', { showWindow: true, win: { start: 0, frames: 121, sized: false } }, 'an unsized bracket: the start only'],
    [p1, 'ltx', 'cast_rosa', { same: true, pre: { scene: 'a grey dusk', seed: 5 }, keptPrompt: 'Rosa walks under the viaduct.' }, 'Try again: the take\'s prompt, scene and seed while Rosa stays'],
  ]) {
    const flow = recastParams(sel, eng, { cast_id: cast }, {
      relight: !!o.relight,
      kept: o.same ? { prompt: o.keptPrompt, scene: o.pre.scene, seed: o.pre.seed } : undefined,
      win: o.showWindow ? o.win : null,
    });
    eq(flow, oldRecast(sel, eng, { cast }, o), `recast params: ${what}`);
  }
  // The old form now builds its params with recastParams too: its other ways in still come out as before.
  for (const [source, who, o, what] of [
    [{ image: { kind: 'image', id: 'img_1' } }, { who_ref: { kind: 'image', id: 'img_1' } }, { describe: 'a knight' }, 'a project image and words'],
    [{ photo: 'up_1' }, { ref_upload_id: 'up_1' }, {}, 'a photo'],
    [{ madeTake: 'v_1', madeCaption: 'a knight, front' }, { reference_take: 'v_1', reference_prompt: 'a knight, front' }, { describe: 'a knight' }, 'a made character with its caption'],
    [{ madeCaption: 'a knight, front' }, { reference_prompt: 'a knight, front' }, {}, 'a made character from its caption alone'],
    [{ still: 'img_2' }, { who_ref: { kind: 'image', id: 'img_2' } }, { describe: 'a knight, no helmet' }, 'a still from words, with changes'],
  ]) {
    eq(recastParams(p1, 'ltx', who, { describe: o.describe }), oldRecast(p1, 'ltx', source, o), `recast params (old form): ${what}`);
  }

  // --- the sheet: x of 3 from the job's stage ---
  eq([SHEET_STAGES.front, SHEET_STAGES.back, SHEET_STAGES.face, SHEET_STAGES.sheet, 'Waiting for the GPU'].map(sheetStep), [1, 2, 3, 3, 0], 'sheet: front 1, back 2, close-up 3, putting it together 3 of 3');
  const rosa = { name: 'Rosa', url: '/m/rosa.png', thumb_url: '/m/rosa_t.jpg' };
  const withSheet = { ...rosa, sheet: { url: '/m/sheet.jpg', views: { front: '/m/f.png', back: '/m/b.png', face: '/m/c.png' } } };

  // Not made: Make sheet.
  let st = sheetStateOf(rosa, undefined);
  eq([st.s, sheetLabel(st), sheetPrimary(st, 'Rosa')], ['none', 'No sheet', { label: 'Make sheet', disabled: false }], 'sheet: not made');
  // Making: queued, then each view, the primary off.
  st = sheetStateOf(rosa, job('j1', 'queued', 'Waiting for the GPU'));
  eq([st.s, sheetLabel(st), sheetPrimary(st, 'Rosa')], ['making', 'Queued', { label: 'Making the sheet', disabled: true }], 'sheet: queued');
  st = sheetStateOf(rosa, job('j1', 'running', SHEET_STAGES.back, { partial: { front: '/u/f.png' } }));
  eq([st.s, sheetLabel(st)], ['making', '2 of 3'], 'sheet: making the back view is 2 of 3');
  eq(sheetViews(rosa, st).map((v) => [v.label, v.url]), [['Front', '/u/f.png'], ['Back', null], ['Close-up', null]], 'sheet: the front view shows as soon as it exists, the rest spin');
  st = sheetStateOf(rosa, job('j1', 'running', SHEET_STAGES.sheet));
  eq(sheetLabel(st), '3 of 3', 'sheet: putting the sheet together is 3 of 3');
  // Ready: Use <name>, and the viewer's 4 tabs (front first).
  st = sheetStateOf(withSheet, job('j1', 'done', 'Done'));
  eq([st.s, sheetLabel(st), sheetPrimary(st, 'Rosa')], ['ready', 'Sheet', { label: 'Use Rosa', disabled: false }], 'sheet: ready');
  eq(sheetViews(withSheet, st).map((v) => v.label), ['Front', 'Back', 'Close-up', 'Sheet'], 'sheet: the viewer has Front, Back, Close-up and Sheet');
  eq(sheetStateOf(rosa, undefined, true).s, 'ready', 'sheet: a wide picture is a sheet already');
  // Failed: Try again, by cast_id (the route refuses a second Rosa), with a new seed.
  st = sheetStateOf(rosa, job('j1', 'failed', 'Something went wrong', { error: 'The back view came back empty. Try again.' }));
  eq([st.s, sheetLabel(st), sheetPrimary(st, 'Rosa'), st.error], ['failed', 'Failed', { label: 'Try again', disabled: false }, 'The back view came back empty. Try again.'], 'sheet: failed');
  // Step 24: one word for a failure, and its line without a second "Try again" (the button says it).
  eq([sheetError(st.error), sheetError('Out of memory.'), sheetError('Please try again'), sheetError(null)], ['The back view came back empty.', 'Out of memory.', '', ''], 'sheet: the error line drops a trailing Try again');
  eq(sheetRequest('cast_rosa', true, 42), { cast_id: 'cast_rosa', seed: 42 }, 'sheet: Try again resubmits with cast_id and a new seed, never a name');
  eq(sheetRequest('cast_rosa', false, 42), { cast_id: 'cast_rosa' }, 'sheet: Make sheet goes by cast_id');
  eq(sheetStateOf(withSheet, job('j1', 'failed', 'x')).s, 'ready', 'sheet: a failed redo leaves the good sheet ready');
  eq(sheetStateOf(rosa, job('j1', 'canceled', 'Canceled')).s, 'none', 'sheet: a canceled one is as if it never ran');
  // The newest job for the member wins (a Try again after a failure).
  const jobs = [job('j_old', 'failed', 'x', { created_at: 100 }), job('j_new', 'running', SHEET_STAGES.front, { created_at: 200 }), job('j_other', 'running', 'x', { params: { cast_id: 'cast_ko' }, created_at: 300 })];
  eq(sheetJobFor('cast_rosa', jobs)?.id, 'j_new', 'sheet: the newest job for the member, not another member\'s');
  eq(sheetJobFor('cast_new', [job('j_made', 'done', 'Done', { params: { name: 'New' }, result_cast_id: 'cast_new' })])?.id, 'j_made', 'sheet: a job that made the member counts too');

  // --- the viewer ---
  const views = sheetViews(withSheet, sheetStateOf(withSheet, undefined));
  eq(viewerPosition(views, 0), 'Front · 1 of 4', 'viewer: Front · 1 of 4');
  eq(viewerPosition([{ key: 'p', label: 'Rosa', url: '/x' }], 0), 'Rosa', 'viewer: one picture is just its label');
  eq([stepIndex(0, -1, 4), stepIndex(3, 1, 4), stepIndex(1, 1, 4), stepIndex(0, 1, 0)], [3, 0, 2, 0], 'viewer: Left and Right go round');

  // --- auto names never collide (step 24): the next free number ---
  eq([freeName('Knight', []), freeName('Knight', ['knight']), freeName('Knight', ['Knight', 'Knight 2', 'Knight 4']), freeName('Knight', ['Knight 2'])], ['Knight', 'Knight 2', 'Knight 3', 'Knight'], 'names: the next free number, case-insensitive');

  // --- a recast draft survives a reload: the picks, the step, and New character's 4 options (its outputs) ---
  const mem = new Map();
  const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  const key = draftKey('shot_a', 'recast');
  const data = { pick: { ...p1, from: 'before' }, who: 'Person 1', mode: 'new', castId: null, fresh: { words: 'a knight', job: null, options: ['img_1', 'img_2', 'img_3', 'img_4'], picked: 'img_3', name: 'Knight', castId: null, error: null }, engine: 'ltx', relight: false };
  let all = putDraft({}, key, { step: 'character', data, jobs: { render: '' } }, 5000);
  all = putDraft(all, draftKey('shot_b', 'remove'), { step: 'who', data: { shadow: true } }, 6000);
  saveDrafts(storage, all);
  const back = loadDrafts(storage, 7000)[key];
  eq([back.step, back.data.fresh.options, back.data.fresh.picked, back.data.who], ['character', ['img_1', 'img_2', 'img_3', 'img_4'], 'img_3', 'Person 1'], 'drafts: the recast step, the pick and the 4 options survive a reload and another task');

  return { fail: problems.length, problems, line: `step 23 units: ${problems.length ? 'FAIL' : 'pass'} (recast params same as the form; sheet not made, making x of 3, ready, failed and Try again by cast_id; viewer position and arrows; recast drafts survive)` };
}
