// Window Seat: "A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and
// slowly pushes in, and at the end he turns and looks at the camera." Built with the real parser, kit, builder, framing,
// shot guard and gaze (headless, no WebGL), then measured frame by frame the way the app plays it (97 frames, t = i / 96).
//  1. a booth (benches with high backs round a table) against a window wall; a large opening beside him; a dark exterior
//  2. the wide shows a diner counter and a row of stools; pendant lamps hang over the booth and the counter
//  3. the push-in is clearly visible (25 percent or more)
//  4. his face is toward the window from frame 0, the turn to the lens starts in the last second and ends on it
//  5. same sentence, same frames; no weather; no lettering
import './node_dom.mjs';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { ROOT, parse, readJSON } from './common.mjs';
import { createLibrary } from '../../src/previz/runtime/loader.js';
import { createShotBuilder } from '../../src/previz/runtime/builder.js';
import { frameShot } from '../../src/previz/runtime/camera.js';
import { aimGaze, applyMotion } from '../../src/previz/runtime/motion.js';
import { registerAnimalPoses, registerPoses } from '../../src/previz/runtime/poser.js';
import { guardShot } from '../../src/previz/runtime/shotGuard.js';
import { sampleCamera } from '../../src/previz/cameraMath.ts';

const PUBLIC = path.resolve(ROOT, '../../public/previz');
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
    return new Response(fs.readFileSync(file));
  }
  return origFetch(input, init);
};

export const WINDOW_SEAT = 'A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and slowly pushes in, and at the end he turns and looks at the camera.';
const ASPECT = 1920 / 1088;
const LAST = 96;
const DEG = 180 / Math.PI;

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

const focalOf = (fov) => Math.round(((36 / ASPECT / 2) / Math.tan((fov * Math.PI) / 360)) * 10) / 10;
const fovOf = (focal) => 2 * Math.atan((36 / ASPECT / 2) / focal) * DEG;

/** The sentence as the app assembles it: parse, build, frame, guard, aim the gaze. */
async function assemble(text) {
  const shot = parse(text);
  const built = await shots().build(shot, { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
  const framed = frameShot(built, shot, { aspect: ASPECT });
  const states = framed.keys && framed.keys.length >= 2 ? framed.keys : framed.mid ? [framed.start, framed.mid, framed.end] : [framed.start, framed.end];
  let keys = states.map((s, i) => ({ t: typeof s.t === 'number' ? s.t : i / (states.length - 1), pos: s.pos, look: s.target, focal_mm: s.focal_mm ?? focalOf(s.fov) }));
  const guarded = guardShot({ built, keys, aspect: ASPECT, sentence: text, shot });
  keys = guarded.keys;
  aimGaze(built, keys[keys.length - 1].pos);
  const spec = { lens: { focal_mm: keys[0].focal_mm }, camera: { ease: 'in_out', keys } };
  return { shot, built, keys, spec, notes: guarded.notes };
}

const worldOf = (o) => o.getWorldPosition(new THREE.Vector3());
/** Visible itself and in every parent (the shot hides kit pieces by hiding their group). */
const shown = (n) => { for (let p = n; p; p = p.parent) if (p.visible === false) return false; return true; };
function headOf(built) {
  const who = built.subject.object;
  return { who, head: who.getObjectByName('head') };
}
/** Horizontal direction the face points (the head bone's +Z, as the seated framing reads it) as an angle about Y. */
function faceAngle(head) {
  head.updateWorldMatrix(true, false);
  const f = new THREE.Vector3(0, 0, 1).applyQuaternion(head.getWorldQuaternion(new THREE.Quaternion()));
  return Math.atan2(f.x, f.z);
}
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const bearing = (from, to) => Math.atan2(to.x - from.x, to.z - from.z);

function cameraAt(rig, f) {
  const s = sampleCamera(rig.spec, f / LAST);
  const cam = new THREE.PerspectiveCamera(fovOf(s.focal), ASPECT, 0.05, 400);
  cam.position.fromArray(s.pos);
  cam.up.set(0, 1, 0);
  cam.lookAt(s.look[0], s.look[1], s.look[2]);
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  return cam;
}
/** Share of a world box that lands inside the frame, and its projected height in NDC units. */
function onScreen(cam, box) {
  const xs = [], ys = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const p = new THREE.Vector3(x, y, z).project(cam);
    xs.push(p.x); ys.push(p.y);
  }
  const w = Math.min(1, Math.max(...xs)) - Math.max(-1, Math.min(...xs));
  const h = Math.min(1, Math.max(...ys)) - Math.max(-1, Math.min(...ys));
  const full = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
  return { inside: full > 0 ? Math.max(0, w) * Math.max(0, h) / full : 0, height: Math.max(...ys) - Math.min(...ys) };
}

export async function runWindowSeat() {
  const problems = [];
  const say = [];
  const fail = (m) => problems.push(m);
  let rig;
  try {
    rig = await assemble(WINDOW_SEAT);
  } catch (err) {
    return { fail: 1, problems: ['window seat: build ' + (err && err.stack ? err.stack.split('\n').slice(0, 3).join(' | ') : err)], line: 'window seat: build failed' };
  }
  const { built, shot, keys } = rig;
  const root = built.root;
  const { who, head } = headOf(built);
  const headP = worldOf(head);

  // ---- 1. the booth, the window wall, the dark outside ----
  const ancestors = [];
  for (let n = who.parent; n; n = n.parent) ancestors.push(n);
  const seat = ancestors.find((n) => n.userData && n.userData.booth && n.userData.asset === 'bench');
  if (!seat) fail('he is not sitting on a booth bench');
  const booths = [];
  built.kit.dressing.children.forEach((c) => { if (shown(c) && c.userData.booth) booths.push(c); });
  const benches = booths.filter((c) => c.userData.asset === 'bench');
  const table = booths.find((c) => c.userData.asset === 'table');
  if (benches.length < 2) fail('the booth has ' + benches.length + ' benches, wanted 2 facing each other');
  if (!table) fail('the booth has no table');
  if (table && seat) {
    const across = benches.find((b) => b !== seat);
    const tp = worldOf(table);
    const fwd = (o) => new THREE.Vector3(0, 0, 1).applyQuaternion(o.getWorldQuaternion(new THREE.Quaternion())).setY(0).normalize();
    if (!across || fwd(seat).dot(fwd(across)) > -0.9) fail('the two booth benches do not face each other');
    const toTable = tp.clone().sub(worldOf(seat)).setY(0);
    if (toTable.length() > 1.3 || toTable.normalize().dot(fwd(seat)) < 0.9) fail('the table is not in front of his bench');
  }
  const meshes = [];
  built.kit.set.traverse((n) => { if (n.isMesh && shown(n)) meshes.push(n); });
  const heightOf = (n) => { n.geometry.computeBoundingBox(); return new THREE.Box3().setFromObject(n).getSize(new THREE.Vector3()).y; };
  const backs = meshes.filter((n) => /^booth_back/.test(n.name));
  const hiBack = backs.find((n) => heightOf(n) >= 1.0 && new THREE.Box3().setFromObject(n).distanceToPoint(headP) < 1.2);
  if (!hiBack) fail('no high booth back behind him');
  const ww = built.kit.windowWall;
  if (!ww) fail('the kit has no window wall');
  let toWindow = null;
  if (ww) {
    const n = new THREE.Vector3(ww.normal[0], 0, ww.normal[1]);
    const along = new THREE.Vector3(-n.z, 0, n.x);
    const dist = headP.dot(n) - ww.at;
    const u = headP.dot(along);
    if (!(dist > 0 && dist < 1.6)) fail('he is ' + dist.toFixed(2) + ' m from the window wall, wanted under 1.6');
    if (!(u > ww.span[0] && u < ww.span[1])) fail('he is not beside the window opening');
    if (ww.span[1] - ww.span[0] < 3 || ww.y[1] - ww.y[0] < 1.2) fail('the window opening is small: ' + (ww.span[1] - ww.span[0]).toFixed(1) + ' x ' + (ww.y[1] - ww.y[0]).toFixed(1) + ' m');
    // the wall really is open there: a ray from his head straight through the opening hits nothing in the set before the dark outside
    const ray = new THREE.Raycaster(new THREE.Vector3(headP.x, 1.5, headP.z), n.clone().negate(), 0, 12);
    const hit = ray.intersectObjects(meshes, false).filter((h) => !/^night_|^geo_ground$|^window_sill$/.test(h.object.name));
    if (hit.length && !/^night_/.test(hit[0].object.name)) fail('the window is blocked by ' + hit[0].object.name);
    toWindow = new THREE.Vector3(headP.x, 1.5, headP.z).addScaledVector(n, -(dist + 0.2));
    toWindow.y = headP.y;
  }
  const outside = meshes.filter((n) => /^night_outside$/.test(n.name));
  const lum = (m) => { const c = [].concat(m.material)[0].color; return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; };
  if (!outside.length || lum(outside[0]) > 0.02) fail('no dark exterior behind the window');
  if (!shot.world || shot.world.time !== 'night') fail('the sentence did not read as night');

  // ---- 2. counter, stools, lamps ----
  const counters = [];
  built.kit.dressing.children.forEach((c) => { if (shown(c) && c.userData.asset === 'counter') counters.push(c); });
  const stools = meshes.filter((n) => /^stool_seat_/.test(n.name));
  const lamps = meshes.filter((n) => /^lamp_shade_/.test(n.name) && [].concat(n.material)[0].emissiveIntensity > 0);
  if (!counters.length) fail('no counter in the diner');
  if (stools.length < 4) fail('only ' + stools.length + ' stools');
  if (lamps.length < 3) fail('only ' + lamps.length + ' lit lamps');
  const lampNearBooth = lamps.some((l) => table && Math.hypot(worldOf(l).x - worldOf(table).x, worldOf(l).z - worldOf(table).z) < 1.0 && worldOf(l).y > 1.8);
  if (!lampNearBooth) fail('no pendant lamp over the booth');
  const lampNearCounter = lamps.some((l) => counters.some((c) => Math.abs(worldOf(l).x - worldOf(c).x) < 2 && Math.abs(worldOf(l).z - worldOf(c).z) < 1.4));
  if (!lampNearCounter) fail('no pendant lamp over the counter');
  const cam0 = cameraAt(rig, 0);
  const seen = (objs) => objs.filter((o) => onScreen(cam0, new THREE.Box3().setFromObject(o)).inside > 0.6).length;
  const stoolsSeen = seen(stools), countersSeen = seen(counters), lampsSeen = seen(lamps);
  if (stoolsSeen < 3) fail('the wide shows ' + stoolsSeen + ' stools, wanted 3 or more');
  if (!countersSeen) fail('the wide does not show the counter');
  if (lampsSeen < 2) fail('the wide shows ' + lampsSeen + ' lamps');
  const heroBox = new THREE.Box3().setFromObject(who);
  if (onScreen(cam0, heroBox).inside < 0.99) fail('he is not fully in the wide');
  const opening = ww ? new THREE.Box3(new THREE.Vector3(ww.at - 0.05, ww.y[0], ww.span[0]), new THREE.Vector3(ww.at + 0.05, ww.y[1], ww.span[1])) : null;
  const openShare = opening ? onScreen(cam0, opening).inside : 0;
  if (openShare < 0.5) fail('only ' + Math.round(openShare * 100) + ' percent of the window opening is in the wide');

  // ---- 3. the push-in ----
  const camStart = cameraAt(rig, 0), camEnd = cameraAt(rig, LAST);
  const dist0 = camStart.position.distanceTo(headP), dist1 = camEnd.position.distanceTo(headP);
  const h0 = onScreen(camStart, heroBox).height, h1 = onScreen(camEnd, heroBox).height;
  const grow = h1 / h0 - 1, shrink = 1 - dist1 / dist0;
  if (!(grow >= 0.25)) fail('he grows ' + Math.round(grow * 100) + ' percent on screen, wanted 25 or more');
  if (!(shrink >= 0.25)) fail('the camera closes ' + Math.round(shrink * 100) + ' percent of the distance, wanted 25 or more');
  if (onScreen(camEnd, heroBox).inside < 0.97) fail('he leaves the frame at the end of the push-in');
  // nothing between the lens and his face at the start and the end
  const solids = [];
  root.traverse((n) => {
    if (!n.isMesh || !shown(n)) return;
    for (let p = n; p; p = p.parent) if (p === who) return;
    if (/^night_|^geo_ground$|^lamp_/.test(n.name)) return;
    solids.push(n);
  });
  for (const [label, cam] of [['start', camStart], ['end', camEnd]]) {
    const dir = headP.clone().sub(cam.position);
    const hit = new THREE.Raycaster(cam.position.clone(), dir.clone().normalize(), 0.05, dir.length() - 0.25).intersectObjects(solids, false);
    if (hit.length) fail('at the ' + label + ' ' + hit[0].object.name + ' hides his face');
  }

  // ---- 4. the gaze, frame by frame ----
  const beats = (shot.beats || []);
  const camBeat = beats.find((b) => b.type === 'look_at_camera');
  if (!camBeat || camBeat.from_t < 0.74 || camBeat.to_t > 1) fail('the turn to camera is scheduled ' + (camBeat ? camBeat.from_t + '..' + camBeat.to_t : 'nowhere') + ', wanted the last quarter');
  const look = (f) => {
    applyMotion(who, f / LAST, 4);
    const c = cameraAt(rig, f).position;
    const hp = worldOf(head);
    const face = faceAngle(head);
    return {
      face,
      toWindow: toWindow ? Math.abs(wrap(face - bearing(hp, toWindow))) * DEG : NaN,
      toCamera: Math.abs(wrap(face - bearing(hp, c))) * DEG,
    };
  };
  const rows = [];
  for (let f = 0; f <= LAST; f++) rows[f] = look(f);
  const r0 = rows[0];
  if (!(r0.toWindow < 25)) fail('at frame 0 his face is ' + r0.toWindow.toFixed(0) + ' degrees off the window');
  if (!(r0.toCamera > 40)) fail('at frame 0 his face is already within ' + r0.toCamera.toFixed(0) + ' degrees of the lens');
  for (const f of [60, 72]) if (!(rows[f].toWindow < 25)) fail('at frame ' + f + ' he has already left the window (' + rows[f].toWindow.toFixed(0) + ' degrees off)');
  const moved = (f) => Math.abs(wrap(rows[f].face - r0.face)) * DEG;
  let startF = rows.findIndex((r, f) => moved(f) > 3);
  const endF = rows.findIndex((r) => r.toCamera < 12);
  if (!(startF >= 76)) fail('the head turn starts at frame ' + startF + ', wanted 76 or later');
  if (!(endF >= 0 && endF <= 92)) fail('the head turn reaches the lens at frame ' + endF + ', wanted 92 or earlier (' + rows[92].toCamera.toFixed(0) + ' degrees off at frame 92)');
  for (let f = 92; f <= LAST; f++) if (!(rows[f].toCamera < 8)) fail('at frame ' + f + ' his face is ' + rows[f].toCamera.toFixed(0) + ' degrees off the lens (the look should hold to 96)');
  // the upper body turns with the head, so the turn reads on a mannequin: the chest swings 25 degrees or more toward the lens
  const chestYaw = (f) => { applyMotion(who, f / LAST, 4); return faceAngle(who.getObjectByName('chest')); };
  const chestTurn = Math.abs(wrap(chestYaw(92) - chestYaw(76))) * DEG;
  if (!(chestTurn >= 25)) fail('the chest turns only ' + chestTurn.toFixed(0) + ' degrees with the head');
  // a smooth turn: no frame jumps more than 15 degrees
  let worst = 0;
  for (let f = 1; f <= LAST; f++) worst = Math.max(worst, Math.abs(wrap(rows[f].face - rows[f - 1].face)) * DEG);
  if (worst > 15) fail('the head jumps ' + worst.toFixed(1) + ' degrees in one frame');
  // the body stays seated: hips do not move
  const hipsY = (f) => { applyMotion(who, f / LAST, 4); return worldOf(who.getObjectByName('hips') || who); };
  const h00 = hipsY(0), h96 = hipsY(LAST);
  if (h00.distanceTo(h96) > 0.05) fail('his hips moved ' + h00.distanceTo(h96).toFixed(2) + ' m during the turn');

  // ---- 5. same sentence, same frames; no weather; no lettering ----
  const sig = (r) => JSON.stringify([r.keys, [0, 20, 60, 72, 84, 96].map((f) => { applyMotion(r.built.subject.object, f / LAST, 4); return faceAngle(r.built.subject.object.getObjectByName('head')).toFixed(6); })]);
  const a = sig(rig);
  if (rig.built.dispose) rig.built.dispose();
  const again = await assemble(WINDOW_SEAT);
  if (sig(again) !== a) fail('building the same sentence twice gave different frames');
  root.traverse((n) => {
    if (/rain|snow|fog|mist|haze|storm|precip/i.test(n.name || '')) fail('weather object ' + n.name);
    if (n.isPoints || n.isLineSegments || n.isSprite) fail('particle, line or sprite object ' + (n.name || n.type));
    if (/text|letter|label|neon|lettering|logo|menu/i.test(n.name || '')) fail('lettering object ' + n.name);
  });
  // the generic rule behind the timing: "at the end" puts a look in the last quarter, any other look keeps its place
  const endLook = parse('A woman standing in the park, and at the end she looks at the camera.').beats.find((b) => b.type === 'look_at_camera');
  if (!endLook || endLook.from_t < 0.74) fail('"at the end ... looks at the camera" starts at ' + (endLook && endLook.from_t));
  const plainLook = parse('A woman standing in the park, she looks at the camera.').beats.find((b) => b.type === 'look_at_camera');
  if (!plainLook || plainLook.from_t !== 0.7) fail('a plain "looks at the camera" moved to ' + (plainLook && plainLook.from_t));
  if (again.built.dispose) again.built.dispose();

  const fmt = (n) => (Number.isFinite(n) ? n.toFixed(0) : 'n/a');
  say.push(`push-in ${Math.round(grow * 100)} percent taller on screen, camera ${dist0.toFixed(1)} m to ${dist1.toFixed(1)} m (${Math.round(shrink * 100)} percent closer)`);
  say.push(`face off the window ${fmt(rows[0].toWindow)}/${fmt(rows[60].toWindow)}/${fmt(rows[72].toWindow)}/${fmt(rows[96].toWindow)} degrees and off the lens ${fmt(rows[0].toCamera)}/${fmt(rows[60].toCamera)}/${fmt(rows[72].toCamera)}/${fmt(rows[96].toCamera)} degrees at frames 0/60/72/96`);
  say.push(`turn starts frame ${startF}, on the lens by frame ${endF} and holds to 96, chest turns ${chestTurn.toFixed(0)} degrees`);
  say.push(`wide shows ${stoolsSeen} stools, ${countersSeen} counter, ${lampsSeen} lamps, ${Math.round(openShare * 100)} percent of the window`);
  const line = problems.length ? `window seat: ${problems.length} problem(s)` : 'window seat: booth + window wall + counter + lamps, ' + say.join('; ');
  return { fail: problems.length, problems, line, say };
}

if (process.argv[1] && process.argv[1].endsWith('window_seat.mjs')) {
  const r = await runWindowSeat();
  for (const p of r.problems) console.log('FAIL  window seat ' + p);
  for (const s of r.say || []) console.log('  ' + s);
  console.log(r.line);
  process.exit(r.fail ? 1 : 0);
}
