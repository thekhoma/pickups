// Finish to 4K HDR's choice: the three choices map to the task's params and back, a saved preset or recipe (old
// upscale_4k / hdr params) opens on the right choice, and a 4K source only offers HDR.
import { choiceOf, finishChoices, finishTitle, paramsOf } from '../../src/lib/finish.ts';
import { readFileSync } from 'node:fs';

// lib/recipes.ts's DEFAULT_FINISH (read as text: it imports modules plain Node can't resolve).
const src = readFileSync(new URL('../../src/lib/recipes.ts', import.meta.url), 'utf8');
const DEFAULT_FINISH = { upscale_4k: /upscale_4k: true/.test(src), hdr: /DEFAULT_FINISH[^\n]*hdr: true/.test(src), quality: 'best' };

export function runFinishChoice() {
  const problems = [];
  const eq = (a, b, what) => { if (JSON.stringify(a) !== JSON.stringify(b)) problems.push(`${what}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
  eq(paramsOf('4k-hdr'), { upscale_4k: true, hdr: true }, '4K HDR');
  eq(paramsOf('hdr'), { upscale_4k: false, hdr: true }, '1080p HDR');
  eq(paramsOf('4k'), { upscale_4k: true, hdr: false }, '4K');
  for (const c of ['4k-hdr', 'hdr', '4k']) eq(choiceOf(paramsOf(c), 1920), c, `round trip ${c}`);
  eq(choiceOf(DEFAULT_FINISH, 1920), '4k-hdr', 'the default recipe');
  eq(choiceOf({ upscale_4k: true, hdr: false, quality: 'fast' }, 1920), '4k', 'a saved 4K-only preset');
  eq(choiceOf(null, 1920), '4k-hdr', 'no prefill');
  eq(finishChoices(3840).map((c) => c.value), ['hdr'], 'a 4K source offers HDR only');
  eq(choiceOf(DEFAULT_FINISH, 4096), 'hdr', 'a 4K source never upscales');
  eq(finishTitle('hdr', 1920), 'Finish to 1080p HDR', 'title');
  return { fail: problems.length, problems, line: `finish choice: ${problems.length ? 'FAIL' : 'pass'} (3 choices, presets, 4K source)` };
}
