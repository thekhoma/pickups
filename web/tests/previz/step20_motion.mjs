// Step 20 proof: the motion words for R&D's eb0cb5f shots (tests/fixtures/step20), one JSON line per shot.
// node --experimental-strip-types tests/previz/step20_motion.mjs s03 s04 ...
import { readFileSync } from 'node:fs';
import { cameraMismatch, motionPromptFor, motionSpecOf } from '../../src/previz/motionPrompt.ts';

const out = {};
for (const shot of process.argv.slice(2)) {
  const d = JSON.parse(readFileSync(new URL(`../../../tests/fixtures/step20/${shot}.json`, import.meta.url), 'utf8'));
  out[shot] = { motion: motionPromptFor(motionSpecOf(d.spec), d.framing), mismatch: cameraMismatch(motionSpecOf(d.spec), d.framing) };
}
console.log(JSON.stringify(out));
