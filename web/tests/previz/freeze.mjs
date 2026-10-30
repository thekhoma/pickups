// Regenerate tests/shot_specs.json from tests/sentences.txt with the CURRENT parser.
// Only run this after reviewing the diff of every changed sentence (node tests/run.mjs shows them first).
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, parse, essential } from './common.mjs';
const lines = fs.readFileSync(path.join(ROOT, 'sentences.txt'), 'utf8').split('\n').map((s) => s.trim()).filter((s) => s && !s.startsWith('#'));
const cases = lines.map((text) => ({ text, expect: essential(parse(text)) }));
fs.writeFileSync(path.join(ROOT, 'shot_specs.json'), JSON.stringify(cases, null, 1) + '\n');
console.log('froze', cases.length, 'sentences');
