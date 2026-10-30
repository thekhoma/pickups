// Run 6: deleting a project or an asset. Where new work goes once the current project is deleted (the most recent
// project left, else the fresh empty My shots), the one confirm line, and a reopened form without the deleted pictures.
import { deleteLine, projectAfterDelete, withoutGoneRefs } from '../../src/lib/deleted.ts';

export function runRun6Delete() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };
  const p = (id, updated_at, extra = {}) => ({ id, name: id, updated_at, shots: 0, takes: 0, ...extra });

  // --- where new work goes ---
  const left = [p('unsorted', 100), p('prj_old', 200), p('prj_new', 300), p('examples', 900), p('qc', 950, { archived: true }), p('prj_arch', 990, { archived: true })];
  ok(projectAfterDelete(left, 'prj_gone', 'prj_old') === 'prj_old', 'delete non-current: current stays');
  ok(projectAfterDelete(left, 'prj_gone', 'prj_gone') === 'prj_new', 'delete current: the most recent project left (never Examples, Test runs or archived)');
  ok(projectAfterDelete([p('unsorted', 999), p('prj_a', 10)], 'unsorted', 'unsorted') === 'prj_a', 'delete My shots: another project, not the fresh empty one');
  ok(projectAfterDelete([p('unsorted', 999)], 'unsorted', 'unsorted') === 'unsorted', 'delete the only project: the fresh empty My shots');
  ok(projectAfterDelete([p('unsorted', 5), p('examples', 9)], 'prj_only', 'prj_only') === 'unsorted', 'delete the only user project: My shots');
  ok(projectAfterDelete([p('unsorted', 5), p('prj_iso', '2026-10-25T10:00:00Z'), p('prj_num', 1)], 'x', 'x') === 'prj_iso', 'ISO and epoch times compare');

  // --- the confirm's line ---
  ok(deleteLine('project', 0) === 'No shots in it.', `project 0: ${deleteLine('project', 0)}`);
  ok(deleteLine('project', 1) === '1 shot goes with it.', `project 1: ${deleteLine('project', 1)}`);
  ok(deleteLine('project', 4) === '4 shots go with it.', `project 4: ${deleteLine('project', 4)}`);
  ok(deleteLine('shot', 0) === null && deleteLine('shot', 2) === '2 takes go with it.', 'a clip says its takes');
  ok(deleteLine('asset', 0) === null, 'an unused asset says nothing more');
  ok(deleteLine('asset', 1) === 'Used in 1 shot.' && deleteLine('asset', 3) === 'Used in 3 shots.', 'a used asset says so');

  // --- a reopened form without the deleted pictures ---
  const params = {
    prompt: 'Maya walks.',
    look_refs: [{ kind: 'cast', id: 'cast_gone' }, { kind: 'image', id: 'img_kept' }, { kind: 'upload', id: 'up_1' }],
    refs: [{ kind: 'image', id: 'img_gone' }],
    look_frame_ref: { kind: 'image', id: 'img_gone' },
    who_ref: { kind: 'cast', id: 'cast_kept' },
    cast_id: 'cast_gone',
  };
  const got = withoutGoneRefs(params, ['cast_kept'], ['img_kept']);
  ok(JSON.stringify(got.look_refs) === JSON.stringify([{ kind: 'image', id: 'img_kept' }, { kind: 'upload', id: 'up_1' }]), `look_refs: ${JSON.stringify(got.look_refs)}`);
  ok(Array.isArray(got.refs) && got.refs.length === 0, 'refs: the gone image drops');
  ok(!('look_frame_ref' in got) && !('cast_id' in got), 'a gone start frame and cast_id drop');
  ok(got.who_ref?.id === 'cast_kept' && got.prompt === 'Maya walks.', 'what is still there stays');
  ok(params.look_refs.length === 3 && params.cast_id === 'cast_gone', 'the take\'s own settings are never changed');
  const unknown = withoutGoneRefs(params, null, null);
  ok(unknown.look_refs.length === 3 && unknown.cast_id === 'cast_gone', 'nothing drops before the project is loaded');

  return problems;
}
