// Run 10: the auto Look's "who the grey figure is" sentence is plain English with one verb (Run 9 wrote "is a man in a
// long coat stands on a pier"), and its lead still names the time and the wardrobe.
import { figureLine, lookPromptFor } from '../../src/previz/lookPrompt.ts';
import { isLocked, nextSeed, seedFrom } from '../../src/lib/seed.ts';

export function runRun10Units() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };

  const pier = lookPromptFor('A man in a long coat stands on a pier at dusk. The camera slowly pushes in.');
  ok(pier.startsWith('It is dusk. The man wears a long coat. The grey figure is a man in a long coat who stands on a pier at dusk.'), `pier look: "${pier}"`);
  ok(!/\bis a man in a long coat stands\b/.test(pier), 'never "is ... stands"');
  ok(figureLine('A man is standing on a pier.') === 'The grey figure is a man who is standing on a pier.', figureLine('A man is standing on a pier.'));
  ok(figureLine('A man standing on a pier.') === 'The grey figure is a man standing on a pier.', figureLine('A man standing on a pier.'));
  ok(figureLine('Two women in red dresses walk along the beach.', 2) === 'The grey figures are two women in red dresses who walk along the beach.', figureLine('Two women in red dresses walk along the beach.', 2));
  ok(figureLine('A woman in boots carries a bag.') === 'The grey figure is a woman in boots who carries a bag.', figureLine('A woman in boots carries a bag.'));
  ok(figureLine('A man who waits by the door.') === 'The grey figure is a man who waits by the door.', figureLine('A man who waits by the door.'));
  ok(figureLine('A man in a long coat on a pier.') === 'The grey figure is a man in a long coat on a pier.', figureLine('A man in a long coat on a pier.'));

  // Run 10 D: after a reload there is no seed, so the lock is never on and never sends a random seed as "locked".
  ok(!isLocked({ seed: null, locked: true }), 'no seed: the lock is off');
  ok(isLocked({ seed: 42, locked: true }) && !isLocked({ seed: 42, locked: false }), 'a seed: the lock is what was pressed');
  const reloaded = { ...seedFrom('', true), locked: true }; // the Image page's form after a reload, the lock pressed
  ok(!isLocked(reloaded) && nextSeed({ ...reloaded, locked: isLocked(reloaded) }, () => 0.25) !== null, 'reloaded and pressed: a fresh seed, sent as not locked');
  ok(nextSeed({ seed: 77, locked: isLocked({ seed: 77, locked: true }) }) === 77, 'a real seed locked: the same one');
  return problems;
}
