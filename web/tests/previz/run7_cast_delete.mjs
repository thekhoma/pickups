// Run 7: delete a character from the Cast tab. The confirm's words (the same ConfirmDelete as projects and assets), and
// "Try again with changes" opening without the gone character however the take named them (a picked reference, cast_id,
// or a person of a multi-person Recast, who keeps their words).
import { castDeleteLabel, deleteLine, withoutGoneRefs } from '../../src/lib/deleted.ts';

export function runRun7CastDelete() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };

  ok(castDeleteLabel('person') === 'Delete character', `person: ${castDeleteLabel('person')}`);
  ok(castDeleteLabel('prop') === 'Delete asset' && castDeleteLabel('place') === 'Delete asset', 'a prop or a place is an asset');
  ok(deleteLine('asset', 2) === 'Used in 2 shots.' && deleteLine('asset', 0) === null, 'the one line says who uses it');

  const params = {
    describe: 'a tall woman',
    who_ref: { kind: 'cast', id: 'cast_gone' },
    look_refs: [{ kind: 'cast', id: 'cast_gone' }, { kind: 'cast', id: 'cast_kept' }],
    swaps: [{ frame: 0, cast_id: 'cast_gone', name: 'Maya' }, { frame: 0, cast_id: 'cast_kept' }, { frame: 4, describe: 'a boy' }],
  };
  const got = withoutGoneRefs(params, ['cast_kept'], []);
  ok(!('who_ref' in got), 'a gone picked person drops');
  ok(JSON.stringify(got.look_refs) === JSON.stringify([{ kind: 'cast', id: 'cast_kept' }]), `look_refs: ${JSON.stringify(got.look_refs)}`);
  ok(!('cast_id' in got.swaps[0]) && got.swaps[0].name === 'Maya' && got.swaps[0].frame === 0, `swap 1: ${JSON.stringify(got.swaps[0])}`);
  ok(got.swaps[1].cast_id === 'cast_kept' && got.swaps[2].describe === 'a boy', 'the other people stay as they were');
  ok(params.swaps[0].cast_id === 'cast_gone' && params.who_ref.id === 'cast_gone', "the take's own settings are never changed");
  ok(withoutGoneRefs(params, null, null).swaps[0].cast_id === 'cast_gone', 'nothing drops before the cast is loaded');
  return problems;
}
