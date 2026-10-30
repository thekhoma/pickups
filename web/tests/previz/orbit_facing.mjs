// Orbit facing (Run 14, walkthrough bug 6): "a man standing at the beach, the camera starts from behind and orbits
// around to the front" starts behind him and ends in front of him. Parse, real builder, frameShot. Headless.
import './node_dom.mjs';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { ROOT, parse, readJSON } from './common.mjs';
import { createLibrary } from '../../src/previz/runtime/loader.js';
import { createShotBuilder } from '../../src/previz/runtime/builder.js';
import { frameShot } from '../../src/previz/runtime/camera.js';
import { registerAnimalPoses, registerPoses } from '../../src/previz/runtime/poser.js';

const PUBLIC = path.resolve(ROOT, '../../public/previz');
const LOCAL = 'http://previz.local';
if (!globalThis.__previzFetch) {
  globalThis.__previzFetch = true;
  const OrigRequest = globalThis.Request;
  globalThis.Request = class extends OrigRequest {
    constructor(input, init) {
      if (typeof input === 'string' && input.startsWith('/')) input = LOCAL + input;
      super(input, init);
    }
  };
  const origFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    const mark = LOCAL + '/previz/';
    if (url.startsWith(mark)) {
      const file = path.join(PUBLIC, url.slice(mark.length));
      if (!fs.existsSync(file)) return new Response('missing ' + url, { status: 404 });
      return new Response(fs.readFileSync(file));
    }
    return origFetch(input, init);
  };
}

let builder = null;
function shots() {
  if (builder) return builder;
  registerPoses(readJSON('data/person_poses.json'));
  registerAnimalPoses(readJSON('data/animal_poses.json'));
  const layouts = new Map();
  builder = createShotBuilder({
    lib: createLibrary({ baseUrl: '/previz/assets/' }),
    getLayout: async (id) => {
      if (layouts.has(id)) return layouts.get(id);
      const layout = JSON.parse(fs.readFileSync(path.join(PUBLIC, 'data/kits', id + '.json'), 'utf8'));
      layouts.set(id, layout);
      return layout;
    },
  });
  return builder;
}

/** Where the camera stands around the subject, in degrees: 0 in front of them (they face it), 180 behind. */
function sideOf(key, subject) {
  const q = new THREE.Quaternion();
  subject.getWorldQuaternion(q);
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q).setY(0).normalize();
  const c = new THREE.Vector3();
  new THREE.Box3().setFromObject(subject).getCenter(c);
  const to = new THREE.Vector3(key.pos[0] - c.x, 0, key.pos[2] - c.z).normalize();
  return Math.round((Math.acos(Math.max(-1, Math.min(1, fwd.dot(to)))) * 180) / Math.PI);
}

export async function orbitSides(text) {
  const shot = parse(text);
  const built = await shots().build(shot, { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
  const framed = frameShot(built, shot, { aspect: 1920 / 1088 });
  const keys = framed.keys && framed.keys.length >= 2 ? framed.keys : [framed.start, framed.end];
  const subject = (built.actors || []).find((a) => a.object)?.object;
  return { move: shot.camera && shot.camera.move, start: sideOf(keys[0], subject), end: sideOf(keys[keys.length - 1], subject), keys: keys.length };
}

const ROWS = [
  // [sentence, start side, end side]: within 35 degrees of behind (180) or in front (0)
  ['A man standing at the beach, camera starts from behind and orbits around to the front, crane down.', 'behind', 'front'],
  ['A man standing at the beach. The camera orbits from behind him to the front.', 'behind', 'front'],
  ['A woman standing in a park, the camera starts in front of her and orbits around behind her.', 'front', 'behind'],
];
const near = (deg, side) => (side === 'behind' ? deg >= 145 : deg <= 35);

export async function runOrbitFacing() {
  let fail = 0;
  for (const [text, from, to] of ROWS) {
    try {
      const got = await orbitSides(text);
      if (got.move !== 'orbit' || !near(got.start, from) || !near(got.end, to)) {
        fail++;
        console.log(`FAIL  orbit facing: "${text}" → ${got.move}, starts ${got.start}°, ends ${got.end}° (want ${from} → ${to})`);
      }
    } catch (e) {
      fail++;
      console.log(`FAIL  orbit facing: "${text}" → ${e && e.message}`);
    }
  }
  return { fail, line: `orbit facing: ${ROWS.length - fail}/${ROWS.length}` };
}

if (process.argv[1] && process.argv[1].endsWith('orbit_facing.mjs')) {
  for (const [text] of ROWS) console.log(text, await orbitSides(text));
  const r = await runOrbitFacing();
  console.log(r.line);
  process.exit(r.fail ? 1 : 0);
}
