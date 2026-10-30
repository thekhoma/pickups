// Shared loader for the offline tests (Node 18+, no dependencies).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLexicon } from '../../src/previz/runtime/lexicon.js';
import { parseShot } from '../../src/previz/runtime/grammar.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.resolve(HERE, '../../public/previz/data');
/** run.mjs and freeze.mjs join this with tests/… and data/…. Those folders live beside this file and under public/previz. */
export const ROOT = HERE;

export const readJSON = (p) => {
  const file = p.startsWith('data/') ? path.join(DATA, p.slice('data/'.length))
    : p.startsWith('tests/') ? path.join(HERE, p.slice('tests/'.length))
    : path.join(HERE, p);
  return JSON.parse(fs.readFileSync(file, 'utf8'));
};

export const manifest = readJSON('data/library_manifest.json');
export const grammar = readJSON('data/grammar.json');
export const lex = buildLexicon(manifest, grammar);
export const parse = (text) => parseShot(text, lex, grammar);
// the part of a spec the tests pin down (ids, assets, counts, poses, relations, place, camera, unmatched words)
export function essential(spec) {
  const rel = (r) => { const o = { type: r.type, target: r.target }; for (const k of ['seats', 'slots', 'side', 'hand']) if (r[k] != null) o[k] = r[k]; if (r.pair) o.pair = r.pair.id; return o; };
  return {
    place: spec.place.id + (spec.place.standin ? ' (standin)' : ''),
    actors: spec.actors.map((a) => { const o = { asset: a.asset, count: a.count, pose: a.pose }; if (a.preset && a.preset !== 'adult') o.preset = a.preset; if (a.presets) o.presets = a.presets; if (a.size) o.size = a.size; if (a.formation) o.formation = a.formation; if (a.pair) o.pair = a.pair.id + ':' + a.pair.poses.join('/'); if (a.relations.length) o.relations = a.relations.map(rel); return o; }),
    props: spec.props.map((p) => { const o = { asset: p.asset, count: p.count }; if (p.implied) o.implied = true; if (p.size) o.size = p.size; if (p.relations) o.relations = p.relations.map(rel); return o; }),
    camera: Object.fromEntries(Object.entries(spec.camera).filter(([k, v]) => v != null && !['segments', 'handheld', 'lensMm', 'facing', 'dutch', 'reveal', 'subjectSpeed'].includes(k))),
    ...(spec.world && (spec.world.time || spec.world.weather) ? { world: Object.fromEntries(Object.entries(spec.world).filter(([, v]) => v != null)) } : {}),
    ...(spec.unmatched.length ? { unmatched: spec.unmatched } : {}),
  };
}
