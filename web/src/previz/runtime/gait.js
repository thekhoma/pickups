// Procedural gait. Frame t always poses the same way: no clock, no random, no mocap.
// The root rises and falls on one smooth curve. Each leg reaches its own foot target.
// Arms swing from the same phase as the legs and hang at the sides.
import * as THREE from 'three';
import { animalPose, rigOf } from './poser.js';

const DEG = Math.PI / 180;
const GAIT = {
  walk: { hz: 1, stance: 0.52, speed: 1.35, bob: 0.04, front: 0.20, heel: 0.16 },
  jog: { hz: 1.45, stance: 0.44, speed: 2.5, bob: 0.05, front: 0.24, heel: 0.18 },
  run: { hz: 1.75, stance: 0.36, speed: 4, bob: 0.06, front: 0.26, heel: 0.22 },
};
const FOLLOW = new Set([undefined, null, 'push_in', 'pull_out', 'follow', 'track']);
const skipped = new Set();
const _ray = new THREE.Raycaster();
const _v = new THREE.Vector3();

/** Animals whose walk/run pose has no leg pivots on the rig. */
export function gaitSkipped() { return [...skipped]; }

function phaseOf(seed, id, index) {
  let h = (seed >>> 0) ^ Math.imul(index + 1, 374761393);
  const s = String(id || '');
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return ((h >>> 0) % 1000) / 1000;
}

function clearPath(obj, yaw, kit) {
  if (!kit || !kit.dressing) return 24;
  const origin = obj.position.clone();
  origin.y += 0.9;
  const dir = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  _ray.set(origin, dir); _ray.near = 0.4; _ray.far = 14;
  const hit = _ray.intersectObjects(kit.dressing.children, true).find((h) => h.object.visible !== false && !/ground|floor|rug|path|road|water|dune|shore|swash|wave|dash/i.test(h.object.name));
  return hit ? Math.max(0.6, hit.distance - 0.7) : 24;
}

function hasLeg(obj, pose, asset) {
  const sample = animalPose(asset, pose);
  if (!sample || !Object.keys(sample).some((k) => /^(leg|knee|hip|ankle)_/.test(k))) return false;
  return ['leg_FL', 'knee_FL', 'hip_L', 'leg_BL', 'ankle_L'].some((n) => obj.getObjectByName(n));
}

function legLengths(obj) {
  const knee = obj.getObjectByName('knee_L') || obj.getObjectByName('knee_FL');
  const ankle = obj.getObjectByName('ankle_L') || obj.getObjectByName('ankle_FL');
  const hip = obj.getObjectByName('hip_L') || obj.getObjectByName('leg_FL');
  if (!knee || !ankle) return null;
  const L1 = knee.position.length();
  const L2 = ankle.position.length();
  if (L1 < 0.05 || L2 < 0.05) return null;
  return { L1, L2, hipX: hip ? Math.abs(hip.position.x) : 0.09 };
}

/** Stride is the step the cadence covers. The foot is under the hip at mid-stance. */
function personStride(speed, hz, stance) {
  const stride = speed / hz;
  return { stride, plant: stride * stance * 0.5, speed };
}

/** Straight-leg hip and ankle, measured with the walk pose cleared so a bent rest pose cannot crouch the gait. */
function straightMeasure(obj) {
  const names = ['hips', 'spine', 'chest', 'neck', 'hip_L', 'knee_L', 'ankle_L', 'hip_R', 'knee_R', 'ankle_R'];
  const saved = names.map((n) => {
    const j = obj.getObjectByName(n);
    return j ? j.rotation.clone() : null;
  });
  for (const n of names) {
    const j = obj.getObjectByName(n);
    if (j) j.rotation.set(0, 0, 0);
  }
  obj.updateMatrixWorld(true);
  let ankleY = obj.position.y + 0.08;
  let hipAboveRig = 0.915;
  const ankle = obj.getObjectByName('ankle_L');
  const hip = obj.getObjectByName('hip_L');
  if (ankle && hip) {
    ankle.getWorldPosition(_v);
    ankleY = _v.y;
    hip.getWorldPosition(_v);
    let rigY = 0;
    try { rigY = rigOf(obj).position.y; } catch { rigY = 0; }
    hipAboveRig = _v.y - obj.position.y - rigY;
  }
  names.forEach((n, i) => {
    const j = obj.getObjectByName(n);
    if (j && saved[i]) j.rotation.copy(saved[i]);
  });
  return { ankleY, hipAboveRig };
}

/** Attach a gait to a placed actor. Static poses are left alone. */
export function bindGait(obj, actor, index, spec, kit) {
  const pose = actor.pose;
  const g = GAIT[pose];
  if (!g || !obj) return;
  const asset = obj.userData.asset || String(obj.userData.libraryId || '').split('__')[0];
  const animal = actor.kind !== 'person';
  if (animal && !hasLeg(obj, pose, asset)) { if (asset) skipped.add(asset); return; }
  const yaw = obj.rotation.y;
  let rigY = 0;
  try { rigY = rigOf(obj).position.y; } catch { rigY = 0; }
  const crowd = String(obj.userData.libraryId || '').includes('crowd');
  const leg = animal ? null : legLengths(obj);
  const stance = crowd ? 0.5 : g.stance;
  let stride = g.speed / g.hz;
  let plant = stride * stance * 0.5;
  let legLen = leg ? leg.L1 + leg.L2 : 0.835;
  const speed = g.speed;
  if (leg) {
    const s = personStride(g.speed, g.hz, stance);
    stride = s.stride; plant = s.plant; legLen = leg.L1 + leg.L2;
  }
  const straight = straightMeasure(obj);
  const standV = straight.hipAboveRig + rigY + obj.position.y - straight.ankleY;
  obj.userData.gait = {
    pose, hz: g.hz, speed, stride, plant, crouch: 0, legLen, stance,
    bob: g.bob, front: g.front, heel: g.heel,
    animal, asset, leg,
    phase: phaseOf(spec.seed || 0, actor.id, index),
    yaw, start: [obj.position.x, obj.position.z], baseY: obj.position.y, rigY,
    ankleY: straight.ankleY, hipAboveRig: straight.hipAboveRig, standV,
    hipX: leg ? leg.hipX : 0.09,
    maxDist: clearPath(obj, yaw, kit),
    crowd,
    // "walks, pauses, then sits": the sentence's holds for this actor, as shot-time spans (grammar.js holdBeats).
    holds: (spec.beats || []).filter((b) => b.type === 'hold' && b.actor === actor.id).map((b) => [b.from_t, b.to_t]),
    follow: FOLLOW.has(spec.camera && spec.camera.move),
  };
}

/** Where a walker's gait puts them at shot time t (0-1), ignoring the clear-path cap. */
function walkAt(g, t, dur) {
  const d = Math.abs(g.speed) * t * dur;
  return [g.start[0] + Math.sin(g.yaw) * d, g.start[1] + Math.cos(g.yaw) * d];
}

/** Closest two walkers come, centre to centre, over the shot (people standing still count as walkers with speed 0). */
function closest(g, h, dur) {
  let best = Infinity;
  for (let k = 0; k <= 32; k++) {
    const t = k / 32;
    const a = walkAt(g, t, dur);
    const b = walkAt(h, t, dur);
    best = Math.min(best, Math.hypot(a[0] - b[0], a[1] - b[1]));
  }
  return best;
}

/**
 * Every walker keeps walking for the whole shot and nobody walks into anybody.
 * clearPath stops a walker short of the first set piece ahead, and a walker that reaches it marches on the spot for
 * the rest of the shot (the frozen mannequin). Walkers that start inside one another or pass within a shoulder of each
 * other bump. Walkers are settled in order (the subject first, never moved unless their own way is blocked): each one
 * that is blocked or would touch someone already settled sidesteps off their line or starts further back along it,
 * whichever small move first gives them a clear walk. `people` are everyone in the shot (standing people too).
 * Returns the walkers that could not be settled.
 */
export function settleWalkers(people, kit, duration = 4, gap = 0.75) {
  const dur = duration > 0 ? duration : 4;
  const still = (o) => ({ speed: 0, yaw: 0, start: [o.position.x, o.position.z] });
  const settled = people.filter((o) => !(o.userData.gait && !o.userData.gait.animal)).map(still);
  const left = [];
  for (const o of people) {
    const g = o.userData.gait;
    if (!g || g.animal) continue;
    const need = Math.abs(g.speed) * dur + 0.3;
    const dir = [Math.sin(g.yaw), Math.cos(g.yaw)];
    const side = [dir[1], -dir[0]];
    const home = [g.start[0], g.start[1]];
    const ok = (cand) => settled.every((h) => closest(cand, h, dur) >= gap);
    const tryAt = (lat, back) => {
      const x = home[0] + side[0] * lat - dir[0] * back;
      const z = home[1] + side[1] * lat - dir[1] * back;
      const cand = { speed: g.speed, yaw: g.yaw, start: [x, z] };
      if (!ok(cand)) return false;
      o.position.x = x; o.position.z = z;
      o.updateMatrixWorld(true);
      const room = clearPath(o, g.yaw, kit);
      if (room < need) return false;
      g.start = [x, z];
      g.maxDist = room;
      return true;
    };
    let done = false;
    const lats = [0, 0.45, -0.45, 0.9, -0.9, 1.35, -1.35];
    const backs = [0, 0.8, 1.6, 2.4, 3.2, 4.8, 6.4];
    for (const back of backs) {
      for (const lat of lats) if ((done = tryAt(lat, back))) break;
      if (done) break;
    }
    if (!done) {
      o.position.x = home[0]; o.position.z = home[1];
      o.updateMatrixWorld(true);
      g.start = home;
      left.push(o);
    }
    settled.push({ speed: g.speed, yaw: g.yaw, start: g.start.slice() });
  }
  return left;
}

function setE(obj, name, x, y, z) {
  const j = obj.getObjectByName(name);
  if (j) j.rotation.set(x * DEG, (y || 0) * DEG, (z || 0) * DEG);
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

/** Hip pitch that puts the ankle `relZ` ahead of the hip at a prescribed knee bend. Positive knee X flexes. */
function solveTheta(L1, L2, knee, relZ) {
  let theta = -relZ / Math.max(0.2, L1 + L2);
  for (let i = 0; i < 8; i++) {
    const z = -L1 * Math.sin(theta) - L2 * Math.sin(theta + knee);
    const dz = -L1 * Math.cos(theta) - L2 * Math.cos(theta + knee);
    if (Math.abs(dz) < 1e-6) break;
    theta -= (z - relZ) / dz;
  }
  return theta;
}

/** 1 at the centre, 0 at the edge, flat derivative at both. */
function lobe(phase, center, width) {
  const d = Math.abs(((phase - center + 1.5) % 1) - 0.5);
  if (d >= width) return 0;
  return 0.5 * (1 + Math.cos(Math.PI * d / width));
}

/** Dorsiflexed at heel strike, toes down at toe-off. One smooth lobe each, so the wrap doesn't kink. */
function ankleDeg(phase, stance, pose) {
  const plantar = pose === 'run' ? 34 : pose === 'jog' ? 30 : 26;
  return -16 * lobe(phase, 0, 0.22) + plantar * lobe(phase, stance, 0.2);
}

/** Positive is the arm back. A jog stays short so the hand never thrusts out past the chest. */
function shoulderDeg(phase, pose) {
  const back = pose === 'run' ? 24 : pose === 'jog' ? 12 : 15;
  const fwd = pose === 'run' ? 46 : pose === 'jog' ? 12 : 25;
  const B = (back + fwd) / 2;
  const M = (back - fwd) / 4;
  const w = phase * Math.PI * 2;
  return M + B * Math.cos(w) + M * Math.cos(2 * w);
}

function elbowDeg(phase, pose) {
  if (pose === 'run') return -90;
  // A jog keeps the hand between hip and chest. A walk hangs, it does not fold across the pelvis.
  if (pose === 'jog') return -72 + 4 * Math.cos(phase * Math.PI * 2);
  return -22 + 8 * Math.cos(phase * Math.PI * 2);
}

/** Palms toward the thighs. Z is the mirrored inward roll already used by the stand pose; X is a small flex. */
function wristDeg(phase, pose, side) {
  const roll = (side === 'L' ? 15 : -15);
  if (pose !== 'walk') return [-7, 0, roll];
  const lagged = phase - 0.06;
  const wave = Math.sin(lagged * Math.PI * 2);
  return [-7 + 5 * wave, 0, roll];
}

function armPose(obj, g, phase, k = 1) {
  const pose = g.pose;
  const right = (phase + 0.5) % 1;
  // Same swing on both sides, held out from the ribs so a three-quarter view doesn't park a forearm across the pelvis.
  // k eases the swing out in a hold: arms come down to the sides, a little out from the ribs.
  const out = (pose === 'jog' ? 22 : 16) * (0.4 + 0.6 * k);
  setE(obj, 'shoulder_L', k * shoulderDeg(phase, pose), 0, out);
  setE(obj, 'shoulder_R', k * shoulderDeg(right, pose), 0, -out);
  setE(obj, 'elbow_L', k * elbowDeg(phase, pose), 0, 0);
  setE(obj, 'elbow_R', k * elbowDeg(right, pose), 0, 0);
  const wL = wristDeg(phase, pose, 'L');
  const wR = wristDeg(right, pose, 'R');
  setE(obj, 'wrist_L', k * wL[0], k * wL[1], k * wL[2]);
  setE(obj, 'wrist_R', k * wR[0], k * wR[1], k * wR[2]);
}

/** Highest when the stance foot is under the hip, lowest a quarter cycle away. Peak to peak is the gait bob. */
function rootDrop(phase, g) {
  const mid = g.front / g.stride;
  return g.bob * 0.5 * (1 - Math.cos(4 * Math.PI * (phase - mid)));
}

/** Approach the straight-leg limit without arriving, and without a kink in the slope. */
function softCap(d, cap, margin) {
  const start = cap - margin;
  if (d <= start) return d;
  const limit = margin * 0.9;
  return start + limit * (1 - Math.exp(-(d - start) / limit));
}

function hermite(u, p0, p1, m0, m1) {
  const u2 = u * u;
  const u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1;
}

/** Metres the ankle sits ahead of the root. Stance keeps the foot in the world; the swing matches that speed at both ends. */
function footOffset(phase, g) {
  const front = g.front;
  const travel = g.stride * g.stance;
  const back = travel - front;
  if (phase < g.stance) return front - g.stride * phase;
  const span = 1 - g.stance;
  const u = (phase - g.stance) / span;
  const slope = -g.stride * span * 0.35;
  return hermite(u, -back, front, slope, slope);
}

/** Ankle height above a flat plant. Zero slope at contact, toe-off and the wrap. */
function footLift(phase, g) {
  const toe = g.heel;
  const clear = g.pose === 'run' ? 0.05 : g.pose === 'jog' ? 0.04 : 0.03;
  if (phase < g.stance) {
    const u = phase / g.stance;
    const t = clamp((u - 0.4) / 0.6, 0, 1);
    const s = t * t * (3 - 2 * t);
    return toe * s;
  }
  const u = (phase - g.stance) / (1 - g.stance);
  const s = u * u * (3 - 2 * u);
  return toe * (1 - s) + clear * Math.sin(Math.PI * u) ** 2;
}

/** Keep the foot's forward plant. Raise the ankle if the leg cannot reach, and ease the last centimetre. */
function fitReach(L1, L2, rel, v) {
  const cap = L1 + L2;
  const margin = 0.018;
  let d = Math.hypot(rel, v);
  const want = softCap(d, cap, margin);
  if (d <= want + 1e-6) return { rel, v };
  const vKeep = Math.sqrt(Math.max(0, want * want - rel * rel));
  if (v > 0 && vKeep < v) v = vKeep;
  d = Math.hypot(rel, v);
  if (d > want && d > 1e-6) {
    const s = want / d;
    rel *= s;
    v *= s;
  }
  return { rel, v };
}

function solveLeg(L1, L2, rel, v) {
  const fit = fitReach(L1, L2, rel, v);
  const d = Math.max(1e-4, Math.hypot(fit.rel, fit.v));
  const cosK = clamp((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1);
  const knee = Math.PI - Math.acos(cosK);
  const theta = solveTheta(L1, L2, knee, fit.rel);
  return { theta, knee, rel: fit.rel, v: fit.v };
}

/** Diagonal pairs: front-left with back-right, front-right with back-left. */
function diagSign(name, swing) {
  if (/_FL$|_BR$/.test(name)) return swing;
  if (/_FR$|_BL$/.test(name)) return -swing;
  return swing * 0.2;
}

function driveAnimal(obj, asset, pose, phase) {
  const data = animalPose(asset, pose) || animalPose(asset, 'walk');
  if (!data) return;
  const swing = Math.sin(phase * Math.PI * 2);
  for (const [name, deg] of Object.entries(data)) {
    if (!Array.isArray(deg)) continue;
    const sign = diagSign(name, swing);
    setE(obj, name, deg[0] * sign, (deg[1] || 0) * sign * 0.2, (deg[2] || 0) * sign * 0.2);
  }
}

function drivePerson(obj, g, phase, k = 1) {
  // k: how much of the stride shows (1 walking, 0 standing still in a hold).
  const s = k * Math.sin(phase * Math.PI * 2);
  const yaw = k * 2 * Math.cos(phase * Math.PI * 2);
  setE(obj, 'hips', 0, yaw, 0);
  setE(obj, 'spine', 4, 4 * s, 0);
  setE(obj, 'chest', 0, -5 * s, 0);
  setE(obj, 'neck', -4, 0, 0);
  const L1 = g.leg ? g.leg.L1 : 0.425;
  const L2 = g.leg ? g.leg.L2 : 0.41;
  const drop = k * rootDrop(phase, g);
  const stand = g.standV > 0.2 ? g.standV : (L1 + L2);
  try {
    const rig = rigOf(obj);
    rig.position.y = g.rigY - drop;
    rig.position.x = 0;
    rig.position.z = 0;
  } catch { /* no rig */ }
  const yawRad = yaw * DEG;
  for (const side of ['L', 'R']) {
    const ph = side === 'L' ? phase : (phase + 0.5) % 1;
    const sign = side === 'L' ? 1 : -1;
    const ahead = (k * footOffset(ph, g) + sign * g.hipX * Math.sin(yawRad)) / Math.cos(yawRad);
    const lift = k * footLift(ph, g);
    const fit = solveLeg(L1, L2, ahead, stand - drop - lift);
    const hip = obj.getObjectByName(`hip_${side}`);
    const knee = obj.getObjectByName(`knee_${side}`);
    const ankle = obj.getObjectByName(`ankle_${side}`);
    if (hip) hip.rotation.set(fit.theta, 0, 0);
    if (knee) knee.rotation.set(fit.knee, 0, 0);
    if (ankle) ankle.rotation.set(k * ankleDeg(ph, g.stance, g.pose) * DEG, 0, 0);
  }
  armPose(obj, g, phase, k);
}

/** Shot time spent moving by t: t less every hold before it. A hold is a pause, not part of the move before it. */
export function movingTime(holds, t) {
  const u = Math.max(0, Math.min(1, t));
  let held = 0;
  for (const [a, b] of holds || []) held += Math.max(0, Math.min(u, b) - Math.max(0, a));
  return u - held;
}

/** How much of the stride shows at t: 1 while moving, easing to 0 (standing) over HOLD_EASE into a hold and back. */
export const HOLD_EASE = 0.06;
export function strideAmount(holds, t) {
  let k = 1;
  for (const [a, b] of holds || []) {
    if (t <= a || t >= b) continue;
    const e = Math.min(1, (t - a) / HOLD_EASE, (b - t) / HOLD_EASE);
    k = Math.min(k, 1 - e * e * (3 - 2 * e));
  }
  return k;
}

/** Pose and place the actor at shot time t (0–1). Speed scales cadence and root speed together. */
export function applyGait(obj, t, speedScale, duration) {
  const g = obj.userData && obj.userData.gait;
  if (!g) return false;
  const scale = speedScale > 0 ? speedScale : 1;
  const dur = duration > 0 ? duration : 4;
  const tm = movingTime(g.holds, t);
  const k = strideAmount(g.holds, t);
  const dist = Math.min(g.maxDist, g.speed * scale * tm * dur);
  obj.position.x = g.start[0] + Math.sin(g.yaw) * dist;
  obj.position.z = g.start[1] + Math.cos(g.yaw) * dist;
  obj.position.y = g.baseY;
  obj.rotation.y = g.yaw;
  const cycles = tm * dur * g.hz * scale + g.phase;
  g._cycles = cycles;
  g._dist = dist;
  const phase = cycles - Math.floor(cycles);
  if (g.animal) driveAnimal(obj, g.asset, g.pose, phase);
  else drivePerson(obj, g, phase, k);
  return true;
}
