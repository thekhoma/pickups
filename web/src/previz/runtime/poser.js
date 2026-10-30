// Poser: applies poses from data/person_poses.json (and data/animal_poses.json) to LOADED GLBs by joint name.
// Conventions (library v1.2): degrees, Euler XYZ per joint node, rest = all zero; joints carry userData.isJoint and
// userData.limits ({ x: [lo, hi], y, z } in degrees); the person rig node is person.userData.rig ('person_rig').
import * as THREE from 'three';

const DEG = Math.PI / 180;
const POSES = {};
const ALIASES = { lay: 'lie_down', lie: 'lie_down', sit_sip: 'sip', phone: 'hold_phone', idle: 'stand' };
const ANIMAL = { dog: {}, cat: {} };
let FROZEN_VEHICLE = {}, FROZEN_RIDER = {};

/** Register person poses (the whole person_poses.json object). Returns the pose ids. */
export function registerPoses(json) {
  Object.assign(POSES, json.actions || {});
  Object.assign(ALIASES, json.aliases || {});
  FROZEN_VEHICLE = json.in_vehicle_frozen || FROZEN_VEHICLE;
  FROZEN_RIDER = json.rider_frozen || FROZEN_RIDER;
  Object.assign(PAIRS, json.pairs || {});
  return Object.keys(POSES);
}
const PAIRS = {};
/** Two-person pair table (library v1.4 person_poses.json pairs): { a, b, offset, facing }. */
export function getPair(id) { return PAIRS[id] || null; }
export function pairIds() { return Object.keys(PAIRS); }
/** Register animal poses (animal_poses.json: { dog: {...}, cat: {...} }). */
export function registerAnimalPoses(json) { for (const [k, v] of Object.entries(json || {})) if (v && typeof v === 'object' && k !== 'convention') ANIMAL[k] = Object.assign(ANIMAL[k] || {}, v); }
/** Animal kinds with a pose table (dog, cat; library v1.4: cow, giraffe, elephant, bird, horse). */
export function animalKinds() { return Object.keys(ANIMAL); }
/** One animal pose, or a fallback (walk -> stand) when that animal has no clip by this name. */
export function animalPose(kind, id) {
  const table = ANIMAL[kind];
  if (!table || !id) return null;
  if (table[id]) return table[id];
  for (const alt of ANIMAL_FALLBACK[id] || []) if (table[alt]) return table[alt];
  return null;
}
// generic pose names onto each animal's own table (a bird has no 'stand': it perches)
const ANIMAL_FALLBACK = { stand: ['stand', 'perch'], walk: ['walk', 'stand', 'perch'], run: ['run', 'walk', 'fly'], sit: ['sit', 'lie', 'stand', 'perch'], lie: ['lie', 'sit', 'stand', 'perch'], eat: ['graze', 'drink', 'peck', 'stand'], drink: ['drink', 'graze', 'peck', 'stand'], fly: ['fly', 'perch'] };
export function poseIds() { return Object.keys(POSES); }
export function resolvePoseId(id) { return POSES[id] ? id : ALIASES[id] && POSES[ALIASES[id]] ? ALIASES[id] : null; }
export function getPoseData(id) { const r = resolvePoseId(id); return r ? POSES[r] : null; }
export function frozenVehiclePose(kind, scaleKey, key) { const a = FROZEN_VEHICLE[kind]; return (a && a[scaleKey] && a[scaleKey][key]) || null; }
export function frozenRider(kind, scaleKey) { const a = FROZEN_RIDER[kind]; return (a && a[scaleKey]) || null; }

// ---- per-object lookups (cached; clones get their own entry) ----
const INFO = new WeakMap();
function info(obj) {
  let i = INFO.get(obj);
  if (i) return i;
  const rigName = obj.userData && obj.userData.rig;
  const rig = (rigName && obj.getObjectByName(rigName)) || obj;
  const joints = new Map(), attach = new Map();
  rig.traverse((n) => {
    if (n !== rig && n.userData && n.userData.previsAsset) return; // skip things parented to the body (props)
    if (n.userData && n.userData.isJoint && !joints.has(n.name)) joints.set(n.name, n);
    if (n.name && n.name.startsWith('attach_') && !attach.has(n.name)) attach.set(n.name, n);
  });
  // ground samples: a few hundred local points per body mesh, kept for fast re-grounding
  const samples = [];
  rig.traverse((n) => {
    if (!n.isMesh) return;
    let p = n; while (p && p !== rig) { if (p.userData && p.userData.previsAsset) return; p = p.parent; }
    samples.push({ mesh: n, pts: samplePoints(n.geometry) });
  });
  i = { rig, joints, attach, samples };
  INFO.set(obj, i);
  return i;
}
const SAMPLE_CACHE = new WeakMap();
function samplePoints(geom) {
  let s = SAMPLE_CACHE.get(geom);
  if (s) return s;
  // Every vertex. A stride missed the contact point on lie_down and left the body about 3 cm low in the loose box,
  // and a millimetre low on the real mesh. The ground snap still moves the rig so the lowest of these points is y = 0.
  const pos = geom.attributes.position, n = pos.count;
  s = new Float32Array(n * 3);
  for (let i = 0, k = 0; i < n; i++, k += 3) { s[k] = pos.getX(i); s[k + 1] = pos.getY(i); s[k + 2] = pos.getZ(i); }
  SAMPLE_CACHE.set(geom, s);
  return s;
}

export function rigOf(person) { return info(person).rig; }
export function jointsOf(obj) { return info(obj).joints; }
/** First node with this attach name that belongs to the object itself (not to something parented to it). */
export function attachPoint(obj, name) {
  const i = info(obj);
  if (i.attach.has(name)) return i.attach.get(name);
  let hit = null;
  obj.traverse((n) => {
    if (hit || n.name !== name) return;
    let p = n.parent, own = true;
    while (p && p !== obj) { if (p.userData && p.userData.previsAsset) { own = false; break; } p = p.parent; }
    if (own) hit = n;
  });
  return hit;
}

function clampDeg(lim, d) {
  if (!lim) return d;
  const c = (v, r) => (r ? Math.min(r[1], Math.max(r[0], v)) : v);
  return [c(d[0], lim.x), c(d[1], lim.y), c(d[2], lim.z)];
}

/** Scale a person (adult 1, child 0.7). The scale lives on the rig so attach points scale with the body. */
export function setPersonScale(person, s) { rigOf(person).scale.setScalar(s); person.userData.scale = s; return person; }

/** A kid is about 1.3 m, with a slightly larger head than a uniform scale would give. */
export function setChildProportions(person, extra = 1) {
  const s = (1.3 / 1.78) * (extra || 1);
  setPersonScale(person, s);
  const head = person.getObjectByName('head');
  if (head) head.scale.setScalar(1.16);
  return person;
}
export function scaleKey(person) { const s = rigOf(person).scale.x; return Math.abs(s - 1) < 0.01 ? 'adult' : Math.abs(s - 0.7) < 0.01 ? 'child' : null; }

/**
 * setPose(person, pose, { ground = true, clamp = true })
 * pose: an id from person_poses.json (old app names lay, sit_sip, phone are accepted) or a { joint: [x, y, z] } object.
 */
export function setPose(person, pose, options = {}) {
  const p = typeof pose === 'string' ? getPoseData(pose) : pose;
  if (!p) throw new Error('Unknown pose: ' + pose);
  const { rig, joints } = info(person);
  rig.position.set(0, 0, 0); rig.rotation.set(0, 0, 0);
  for (const [name, j] of joints) {
    let d = p[name];
    if (!d) { j.rotation.set(0, 0, 0); continue; }
    if (options.clamp !== false) d = clampDeg(j.userData.limits, d);
    j.rotation.set(d[0] * DEG, d[1] * DEG, d[2] * DEG);
  }
  person.userData.pose = typeof pose === 'string' ? (resolvePoseId(pose) || pose) : 'custom';
  if (options.ground !== false) groundObject(person, p._lift || 0);
  else person.updateMatrixWorld(true);
  return person;
}

/** Lowest sampled body point to y = lift (in the object's own space). Moves the rig (or animal root). */
export function groundObject(obj, lift = 0, mover = null) {
  const i = info(obj);
  const m = mover || i.rig;
  obj.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(obj.matrixWorld).invert(), mw = new THREE.Matrix4(), v = new THREE.Vector3();
  let minY = Infinity;
  for (const { mesh, pts } of i.samples) {
    mw.multiplyMatrices(inv, mesh.matrixWorld);
    const e = mw.elements;
    for (let k = 0; k < pts.length; k += 3) {
      const y = e[1] * pts[k] + e[5] * pts[k + 1] + e[9] * pts[k + 2] + e[13];
      if (y < minY) minY = y;
    }
  }
  if (minY !== Infinity) { m.position.y += lift - minY; obj.updateMatrixWorld(true); }
  void v;
  return minY;
}

/** setAnimalPose(dog | cat, 'stand' | 'walk' | 'run' | 'sit' | 'lie') */
export function setAnimalPose(animal, pose) {
  const asset = animal.userData.asset || String(animal.userData.libraryId || '').split('__')[0];
  const kind = ANIMAL[asset] ? asset : asset === 'cat' ? 'cat' : 'dog';
  const table = ANIMAL[kind] || {};
  let p = typeof pose === 'string' ? table[pose] : pose;
  if (!p && typeof pose === 'string') for (const alt of ANIMAL_FALLBACK[pose] || ['stand', 'perch']) if (table[alt]) { p = table[alt]; pose = alt; break; }
  if (!p) return animal;
  if (kind !== 'dog' && kind !== 'cat') { // v1.4 quadrupeds and the bird: joints flagged isJoint, rest = zero, then back on the ground (a flying bird keeps its height)
    animal.traverse((n) => { if (n.userData && n.userData.isJoint) n.rotation.set(0, 0, 0); });
    for (const [k, v] of Object.entries(p)) { const j = animal.getObjectByName(k); if (j) j.rotation.set(v[0] * DEG, v[1] * DEG, v[2] * DEG); }
    if (pose !== 'fly') groundObject(animal, 0);
    animal.userData.pose = pose; return animal;
  }
  const i = info(animal);
  const rootName = kind + '_root';
  for (const [name, j] of i.joints) if (name !== rootName) j.rotation.set(0, 0, 0);
  for (const [k, v] of Object.entries(p)) { const j = i.joints.get(k) || animal.getObjectByName(k); if (j) j.rotation.set(v[0] * DEG, v[1] * DEG, v[2] * DEG); }
  const root = animal.getObjectByName(rootName) || i.rig;
  root.position.y = 0;
  groundObject(animal, 0, root);
  animal.userData.pose = typeof pose === 'string' ? pose : 'custom';
  return animal;
}

/** Left-right mirror of a pose object (Y and Z flip, _L <-> _R). */
export function mirrorPose(p) {
  const out = {};
  for (const [k, v] of Object.entries(p)) {
    if (!Array.isArray(v)) { out[k] = v; continue; }
    const m = k.endsWith('_L') ? k.slice(0, -2) + '_R' : k.endsWith('_R') ? k.slice(0, -2) + '_L' : k;
    out[m] = [v[0], -v[1], -v[2]];
  }
  return out;
}
