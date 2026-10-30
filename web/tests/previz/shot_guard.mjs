// Shot guard. The headline table is unmodified real builds (parser, kit, mannequins, frameShot).
// The fault table injects a broken stand-in on purpose and is printed separately.
import './node_dom.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { ROOT, parse, readJSON } from './common.mjs';
import { createLibrary } from '../../src/previz/runtime/loader.js';
import { createShotBuilder } from '../../src/previz/runtime/builder.js';
import { frameShot, roomClamp } from '../../src/previz/runtime/camera.js';
import { registerAnimalPoses, registerPoses } from '../../src/previz/runtime/poser.js';
import { auditMotion, auditPlayback, auditShot, guardShot } from '../../src/previz/runtime/shotGuard.js';
import { createWorld } from '../../src/previz/runtime/world.js';

const PUBLIC = path.resolve(ROOT, '../../public/previz');
const ASPECT = 1920 / 1088;
const LOCAL = 'http://previz.local';
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
    const buf = fs.readFileSync(file);
    const type = file.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    return new Response(buf, { status: 200, headers: { 'Content-Type': type } });
  }
  return origFetch(input, init);
};
void fileURLToPath;

const SENTENCES = [
  'A person walking in the desert.',
  'A man walking in the park.',
  'A woman walks away down the street.',
  'A man walks toward the camera on the beach.',
  'A person walking across the field.',
  'A man sitting at a table, seen through a doorway.',
  'A kid flying a kite in the field.',
  'Two people standing in the park.',
  'A couple jogging on the beach.',
  'A crowd of people walking down the street.',
  'A person riding a skateboard. FPV follow.',
  'A person walking in the desert. The camera pulls wide.',
  'Two people in the rainy city.',
  'A man walking in a field in the fog.',
  'A woman sitting in a car at night.',
  'A man driving a truck through the desert.',
  'A child standing in the doorway.',
  'A dog running across the park.',
  'A horse walking in the field.',
  'A woman on a bicycle at night.',
  'Three people walking down the street.',
  'A man leaning on a car in the city.',
  'A girl running through the rain.',
  'Two kids sitting on a bench.',
  'A person standing on a boat.',
  'A cat sitting on a table inside a house.',
  'A man walking a dog on the beach.',
  'A crowd walking at night.',
  'A woman reading a book in the park.',
  'A person skateboarding down the street. The camera follows.',
  'A man and a woman walking in the rain.',
  'A close shot of a woman in a doorway.',
  'A wide shot of a man standing in the desert at night.',
  'A child flying a kite on the beach.',
  'A cow standing in a field.',
  'A man sitting at a desk inside.',
  'Two people jogging in the park.',
  'A woman walking toward the camera in the fog.',
  'A person running across the street.',
  'A man on a horse in the desert.',
  'A girl sitting at a table in a cafe.',
  'A couple walking away down the beach.',
  'Tracking shot left to right alongside a jogger on a beach at golden hour.',
  'Orbit 360 around a dancer in an empty warehouse.',
  'Medium shot, a couple slow dances in a living room, camera slowly circles them.',
  'Truck right past a row of trees revealing a hiker on a ridge.',
  'A surfer carries his board into the waves at sunrise, camera booms up.',
];

function person(name, x, z) {
  const root = new THREE.Group();
  root.name = name;
  root.position.set(x, 0, z);
  const add = (bone, px, py, pz) => {
    const o = new THREE.Object3D();
    o.name = bone;
    o.position.set(px, py, pz);
    root.add(o);
  };
  add('hips', 0, 0.95, 0);
  add('shoulder_L', -0.2, 1.4, 0);
  add('shoulder_R', 0.2, 1.4, 0);
  add('elbow_L', -0.2, 1.1, 0);
  add('elbow_R', 0.2, 1.1, 0);
  add('hip_L', -0.1, 0.95, 0);
  add('hip_R', 0.1, 0.95, 0);
  add('knee_L', -0.1, 0.5, 0);
  add('knee_R', 0.1, 0.5, 0);
  add('ankle_L', -0.1, 0.05, 0);
  add('ankle_R', 0.1, 0.05, 0);
  add('head', 0, 1.65, 0);
  root.updateMatrixWorld(true);
  return root;
}

function sceneFor(text, index) {
  const spec = parse(text);
  const root = new THREE.Group();
  const count = Math.max(1, Math.min(3, spec.actors.filter((a) => a.kind === 'person').reduce((n, a) => n + (a.count || 1), 0) || 1));
  const actors = [];
  for (let i = 0; i < count; i++) {
    const obj = person('person_' + i, 0, 0);
    if (/skate|fpv/i.test(text)) obj.userData.activity = 'ride';
    root.add(obj);
    actors.push({ object: obj });
  }
  const extra = ['goat', 'sheep', 'giraffe', 'elephant'].find((n) => !new RegExp('\\b' + n).test(text.toLowerCase())) || 'goat';
  const cow = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.8), new THREE.MeshBasicMaterial());
  cow.name = extra;
  cow.userData.asset = extra;
  cow.position.set(1.2, 0, 0.4);
  root.add(cow);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshBasicMaterial());
  ground.name = 'geo_ground';
  ground.rotation.x = -Math.PI / 2;
  root.add(ground);
  root.updateMatrixWorld(true);
  const shot = spec.camera.angle === 'pov' ? { camera: { shot: spec.camera.shot, angle: 'pov' } } : { camera: { shot: spec.camera.shot || 'full' } };
  const keys = [
    { t: 0, pos: [0.4, 1.35, 2.1], look: [0, 1.05, 0], focal_mm: 55 },
    { t: 1, pos: [0.2, 1.25, 1.7], look: [0, 1.05, 0], focal_mm: 55 },
  ];
  return {
    text,
    index,
    shot,
    built: { actors, root, kit: { set: root }, spec: { duration_s: 4 }, kitWorld: { fogFar: 18 } },
    keys,
    aspect: 1920 / 1088,
    sentence: text,
  };
}

function focalOf(fov) {
  const filmH = 36 / ASPECT;
  return Math.round(((filmH / 2) / Math.tan((fov * Math.PI) / 360)) * 10) / 10;
}

function table(title, rows) {
  console.log(title);
  console.log('| Sentence | Corrections | Problems left |');
  console.log('|---|---|---|');
  for (const row of rows) {
    console.log('| ' + row.text + ' | ' + (row.notes.length ? row.notes.join('; ') : 'none') + ' | ' + (row.problems.length ? row.problems.join(', ') : 'none') + ' |');
  }
}

let builder = null;
export function shots() {
  if (builder) return builder;
  registerPoses(readJSON('data/person_poses.json'));
  registerAnimalPoses(readJSON('data/animal_poses.json'));
  const lib = createLibrary({ baseUrl: '/previz/assets/' });
  const layouts = new Map();
  const getLayout = async (id) => {
    if (layouts.has(id)) return layouts.get(id);
    const layout = JSON.parse(fs.readFileSync(path.join(PUBLIC, 'data/kits', id + '.json'), 'utf8'));
    layouts.set(id, layout);
    return layout;
  };
  builder = createShotBuilder({ lib, getLayout });
  return builder;
}

async function realOne(text) {
  const shot = parse(text);
  if (/\bFPV\b/i.test(text)) shot.camera = { ...shot.camera, angle: 'pov' };
  const close = shot.camera.shot === 'close' || shot.camera.shot === 'extreme_close';
  const built = await shots().build(shot, { subjectTier: close ? 'hero' : 'lite', liteCount: 1, crowdCount: 12, instancing: true });
  const move = shot.camera.move;
  const people = shot.actors.filter((a) => a.kind === 'person').reduce((n, a) => n + (a.count || 1), 0);
  const stays = (move === 'boom_up' || move === 'truck_left' || move === 'truck_right') && people < 6 && shot.camera.facing !== 'toward';
  const follow = move == null || move === 'push_in' || move === 'pull_out' || move === 'follow' || move === 'track' || move === 'body_mount' || stays;
  for (const actor of built.actors) if (actor.object.userData.gait) actor.object.userData.gait.follow = follow;
  const framed = frameShot(built, shot, { aspect: ASPECT });
  const states = framed.keys && framed.keys.length >= 2 ? framed.keys : framed.mid ? [framed.start, framed.mid, framed.end] : [framed.start, framed.end];
  let keys = states.map((state, i) => ({
    t: states.length === 1 ? 0 : i / (states.length - 1),
    pos: state.pos,
    look: state.target,
    focal_mm: focalOf(state.fov),
  }));
  const guarded = guardShot({ built, keys, aspect: ASPECT, sentence: text, shot });
  keys = guarded.keys;
  const spec = {
    lens: { focal_mm: keys[0].focal_mm },
    camera: { interp: 'catmull_rom', ease: framed.label === 'fpv follow' ? 'linear' : 'in_out', keys },
  };
  const problems = auditPlayback({ built, spec, aspect: ASPECT, sentence: text, shot });
  return { text, notes: guarded.notes, problems, label: framed.label };
}

function faultRows() {
  const rows = [];
  let fail = 0;
  for (const text of SENTENCES) {
    const scene = sceneFor(text);
    const guarded = guardShot(scene);
    const problems = auditShot({ ...scene, keys: guarded.keys });
    if (problems.length) fail += 1;
    rows.push({ text, notes: guarded.notes, problems });
  }
  return { fail, rows };
}

export async function runShotGuard() {
  const rows = [];
  let fail = 0;
  for (const text of SENTENCES) {
    try {
      const row = await realOne(text);
      if (row.problems.length) {
        fail += 1;
        console.log('FAIL  real "' + text + '" still ' + row.problems.join(', '));
      }
      rows.push(row);
    } catch (err) {
      fail += 1;
      console.log('FAIL  real "' + text + '" ' + (err && err.stack ? err.stack.split('\n')[0] : err));
      rows.push({ text, notes: [], problems: ['build'] });
    }
  }
  table('shot guard, real builds:', rows);
  const fault = faultRows();
  table('shot guard, fault injection (not the headline):', fault.rows);
  if (fault.fail) console.log('FAIL  fault injection ' + fault.fail);
  const passed = SENTENCES.length - fail;
  return {
    fail: fail + fault.fail,
    line: `shot guard: ${passed}/${SENTENCES.length} real builds pass; fault injection ${SENTENCES.length - fault.fail}/${SENTENCES.length}`,
    rows,
    fault: fault.rows,
  };
}

// Previz never adds weather. Weather words still parse, and the built shot has no rain, snow or fog objects, and the
// world look (fog distances, fog colour, sky) is exactly what the same shot gets with no weather word at all.
const WEATHER = [
  ['Two people in the rainy city.', 'rain'],
  ['A girl running through the rain.', 'rain'],
  ["Close up on a woman's face in the rain, rack focus.", 'rain'],
  ['A man walking in a field in the fog.', 'fog'],
  'Extreme wide, a lone figure walks out of the fog toward camera.',
  ['A woman walking toward the camera in the fog, misty morning.', 'fog'],
  ['A man standing in a snowy field.', 'snow'],
  ['A stormy night, a man walks down the street.', 'rain'],
  ['A hazy afternoon in the park, two people standing.', 'fog'],
];

export async function runNoWeather() {
  const problems = [];
  const stub = () => ({ shadowMap: {}, userData: {} });
  const lookOf = (w, look) => {
    const scene = new THREE.Scene();
    createWorld(stub(), scene, { environment: false }).apply(w || {}, look);
    const sky = scene.getObjectByName('world_sky');
    return JSON.stringify([scene.fog.near, scene.fog.far, scene.fog.color.getHex(), scene.background.getHex(), sky.material.uniforms.zen.value.getHex(), sky.material.uniforms.hor.value.getHex()]);
  };
  for (const row of WEATHER) {
    const [text, want] = Array.isArray(row) ? row : [row, null];
    const shot = parse(text);
    if (want && (!shot.world || shot.world.weather !== want)) problems.push(text + ': weather word did not parse (' + JSON.stringify(shot.world) + ')');
    if (shot.unmatched.some((u) => /rain|fog|snow|storm|mist|haz/i.test(u))) problems.push(text + ': weather word unmatched');
    let built;
    try {
      built = await shots().build(shot, { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
    } catch (err) {
      problems.push(text + ': build ' + (err && err.message));
      continue;
    }
    built.root.traverse((o) => {
      if (/rain|snow|fog|mist|haze|storm|drizzle|precip/i.test(o.name || '')) problems.push(text + ': weather object ' + o.name);
      else if (o.isPoints || o.isLineSegments) problems.push(text + ': particle or streak object ' + (o.name || o.type));
    });
    const w = built.layout && built.layout.world;
    const time = shot.world ? shot.world.time : null;
    if (lookOf(w, { time, weather: shot.world && shot.world.weather }) !== lookOf(w, { time, weather: null })) problems.push(text + ': weather changed the world look (fog or sky)');
    if (built.dispose) built.dispose();
  }
  return { fail: problems.length, problems, line: `no weather: ${WEATHER.length - new Set(problems.map((p) => p.split(':')[0])).size}/${WEATHER.length} weather sentences add no rain, snow, fog or sky change` };
}

// Every walker keeps walking through the shot, and nobody walks into anybody: all 97 frames of every director sentence
// and the review clips.
export async function runMotionAudit() {
  const texts = [...readJSON('tests/director30.json').map((r) => r.text), 'A crowd of people walking down the street.', 'A crowd walking at night.',
    'Three people walking down the street.', 'A crowd of people crossing the street.', 'People walking both ways on a busy street.'];
  const problems = [];
  for (const text of texts) {
    const shot = parse(text);
    const built = await shots().build(shot, { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
    const found = auditMotion(built);
    if (found.length) problems.push(text + ': ' + found.join(', '));
    if (built.dispose) built.dispose();
  }
  return { fail: problems.length, problems, line: `motion: ${texts.length - problems.length}/${texts.length} shots keep every walker moving with no collisions (97 frames)` };
}

// "Alone" clears the people and benches around the subject, never the subject or the seat it sits on (Fix E3: the
// Window Seat sentence built an empty diner).
const LONE = [
  'A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and slowly pushes in, and at the end he turns and looks at the camera.',
  'A woman sitting alone on a bench in the park.',
  'Extreme wide, a lone figure walks out of the fog toward camera.',
];

export async function runLoneFigure() {
  const problems = [];
  for (const text of LONE) {
    let built;
    try {
      built = await shots().build(parse(text), { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
    } catch (err) {
      problems.push(text + ': build ' + (err && err.message));
      continue;
    }
    const who = (built.actors || []).find((a) => a.kind === 'person' && a.object);
    let shown = !!who;
    for (let n = who && who.object; n; n = n.parent) if (n.visible === false) shown = false;
    if (!shown) problems.push(text + ': the subject is hidden');
    if (built.dispose) built.dispose();
  }
  return { fail: problems.length, problems, line: `lone figure: ${LONE.length - problems.length}/${LONE.length} keep the subject in the shot` };
}
