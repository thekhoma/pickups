// Step 16: two previz animation glitches, on real builds (parser, kit, mannequins, gait, frameShot).
// 1. A pause between moves is a hold of its own: the walker stops travelling and stands, then walks on. It used to be
//    merged into the move before it (one pose for the whole shot; "pauses" wasn't even a word).
// 2. A first-person rider's two shoes read as two shoes across the frame. They used to stack with the deck into one
//    strip down the middle (the eye looked straight down the board's centre line at a 100 degree lens).
import './node_dom.mjs';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { ROOT, parse, readJSON } from './common.mjs';
import { createLibrary } from '../../src/previz/runtime/loader.js';
import { createShotBuilder } from '../../src/previz/runtime/builder.js';
import { frameShot } from '../../src/previz/runtime/camera.js';
import { applyGait, movingTime, strideAmount } from '../../src/previz/runtime/gait.js';
import { holdSpans } from '../../src/previz/runtime/grammar.js';
import { registerAnimalPoses, registerPoses } from '../../src/previz/runtime/poser.js';

const PUBLIC = path.resolve(ROOT, '../../public/previz');
const ASPECT = 1920 / 1088;
const LOCAL = 'http://previz.local';

// The runtime fetches its kit and models by URL: serve them from public/previz (as shot_guard.mjs does).
function serveLocal() {
  if (globalThis.__previzLocal) return;
  globalThis.__previzLocal = true;
  const OrigRequest = globalThis.Request;
  globalThis.Request = class extends OrigRequest {
    constructor(input, init) {
      super(typeof input === 'string' && input.startsWith('/') ? LOCAL + input : input, init);
    }
  };
  const origFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    const mark = LOCAL + '/previz/';
    if (!url.startsWith(mark)) return origFetch(input, init);
    const file = path.join(PUBLIC, url.slice(mark.length));
    if (!fs.existsSync(file)) return new Response('missing ' + url, { status: 404 });
    return new Response(fs.readFileSync(file), { status: 200, headers: { 'Content-Type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream' } });
  };
}

let builder = null;
function shots() {
  if (builder) return builder;
  serveLocal();
  registerPoses(readJSON('data/person_poses.json'));
  registerAnimalPoses(readJSON('data/animal_poses.json'));
  const lib = createLibrary({ baseUrl: '/previz/assets/' });
  const layouts = new Map();
  const getLayout = async (id) => {
    if (!layouts.has(id)) layouts.set(id, JSON.parse(fs.readFileSync(path.join(PUBLIC, 'data/kits', id + '.json'), 'utf8')));
    return layouts.get(id);
  };
  builder = createShotBuilder({ lib, getLayout });
  return builder;
}

const deg = (r) => (r * 180) / Math.PI;

export async function runPausePov() {
  const problems = [];
  const check = (ok, what) => { if (!ok) problems.push(what); };

  // --- 1. pauses ------------------------------------------------------------------------------------------------
  check(holdSpans('A man waits at the bus stop.').length === 0, 'a lone wait is a pose, not a hold');
  check(holdSpans('The car stops at the light, then a woman crosses the street.').length === 0, 'a car stopping holds no person');
  check(holdSpans('She walks to the bench. The camera stops on her.').length === 0, 'camera sentences hold nothing');
  const span = holdSpans('A woman walks along the path, pauses, then walks on.');
  check(span.length === 1 && span[0][0] > 0.2 && span[0][1] < 0.8 && span[0][1] - span[0][0] >= 0.12, `the pause is the middle third: ${JSON.stringify(span)}`);
  check(Math.abs(movingTime([[0.4, 0.6]], 1) - 0.8) < 1e-9 && Math.abs(movingTime([[0.4, 0.6]], 0.5) - 0.4) < 1e-9, 'moving time leaves the hold out');
  check(strideAmount([[0.4, 0.6]], 0.5) === 0 && strideAmount([[0.4, 0.6]], 0.3) === 1, 'the stride eases to nothing mid-hold');

  const text = 'A woman walks along the path, pauses, then walks on.';
  const shot = parse(text);
  const hold = (shot.beats || []).find((b) => b.type === 'hold');
  check(!!hold && !!hold.actor, `the parsed shot has the person's hold: ${JSON.stringify(shot.beats)}`);
  check(!(shot.unmatched || []).includes('pauses'), 'the pause is understood, not left over');
  const built = await shots().build(shot, { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
  const walker = built.actors.map((a) => a.object).find((o) => o && o.userData.gait && o.userData.gait.holds && o.userData.gait.holds.length);
  check(!!walker, 'the walker carries the hold');
  if (walker && hold) {
    const dur = 4;
    const at = (t) => { applyGait(walker, t, 1, dur); return new THREE.Vector2(walker.position.x, walker.position.z); };
    const into = at(hold.from_t + 0.01);
    const outOf = at(hold.to_t - 0.01);
    check(into.distanceTo(outOf) < 0.01, `no travel during the pause: ${into.distanceTo(outOf).toFixed(3)} m`);
    check(at(hold.from_t - 0.12).distanceTo(into) > 0.15, 'walking before the pause');
    check(outOf.distanceTo(at(Math.min(1, hold.to_t + 0.12))) > 0.15, 'walking again after it');
    applyGait(walker, (hold.from_t + hold.to_t) / 2, 1, dur);
    const L = walker.getObjectByName('hip_L');
    const R = walker.getObjectByName('hip_R');
    check(L && R && Math.abs(deg(L.rotation.x - R.rotation.x)) < 3, `standing mid-pause, legs together: ${L && R ? deg(L.rotation.x - R.rotation.x).toFixed(1) : '?'} deg apart`);
  }

  // --- 2. first-person feet -------------------------------------------------------------------------------------
  for (const words of ['first person', 'first-person', 'point of view']) {
    check(parse(`A man skateboarding down a hill, ${words}.`).camera.angle === 'pov', `"${words}" asks for a point of view`);
  }
  const pov = parse('A man skateboarding down a hill, point of view.');
  const rider = await shots().build(pov, { subjectTier: 'hero', liteCount: 1, crowdCount: 12, instancing: true });
  const framed = frameShot(rider, pov, { aspect: ASPECT });
  check(framed.label === 'point of view', `the rider gets the true point of view: ${framed.label}`);
  const cam = new THREE.PerspectiveCamera(framed.start.fov, ASPECT, 0.01, 500);
  cam.position.fromArray(framed.start.pos);
  cam.lookAt(new THREE.Vector3().fromArray(framed.start.target));
  cam.updateMatrixWorld(true);
  rider.root.updateMatrixWorld(true);
  const shoes = [];
  rider.root.traverse((o) => { if (o.name === 'pov_shoe') shoes.push(o); });
  check(shoes.length === 2, `two shoes on the board: ${shoes.length}`);
  const boxes = shoes.map((s) => {
    const b = new THREE.Box3().setFromObject(s);
    let x0 = 9, x1 = -9, y0 = 9, y1 = -9;
    for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
      const p = new THREE.Vector3(x, y, z).project(cam);
      x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
    }
    return { x0, x1, y0, y1, cx: (x0 + x1) / 2 };
  });
  if (boxes.length === 2) {
    const [a, b] = boxes;
    const overlap = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) / Math.min(a.x1 - a.x0, b.x1 - b.x0);
    check(overlap < 0.5, `the shoes sit side by side across the frame, not stacked in one strip: ${(overlap * 100).toFixed(0)}% overlap`);
    check(boxes.every((q) => q.x1 - q.x0 > 0.11), `each shoe reads at its size: ${boxes.map((q) => (q.x1 - q.x0).toFixed(2)).join(', ')} of the frame`);
    check(boxes.every((q) => q.x0 > -1 && q.x1 < 1 && q.y0 > -1 && q.y1 < 1), `both shoes in frame: ${JSON.stringify(boxes.map((q) => [q.x0, q.x1, q.y0, q.y1].map((v) => +v.toFixed(2))))}`);
  }
  for (const u of rider.undo || []) u();

  return { fail: problems.length, problems, line: `pause and pov: ${problems.length ? 'FAIL' : 'pass'} (a pause holds, then walks on; two shoes across a rider's view)` };
}
