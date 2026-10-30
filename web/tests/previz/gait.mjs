// Smoothness of the procedural gait. 240 frames over 4 s, three clips.
// A pop at a footfall shows up as a spike in the second difference.
import * as THREE from 'three';
import { bindGait, applyGait } from '../../src/previz/runtime/gait.js';

const DEG = 180 / Math.PI;
const FRAMES = 240;
const HEIGHT_CM = 0.6;
const ANGLE_DEG = 12;
const JOINTS = ['knee_L', 'knee_R', 'hip_L', 'hip_R', 'ankle_L', 'ankle_R', 'shoulder_L', 'shoulder_R', 'elbow_L', 'elbow_R', 'wrist_L', 'wrist_R', 'hips', 'spine', 'chest', 'neck'];

function bone(name, x, y, z, parent) {
  const o = new THREE.Object3D();
  o.name = name;
  o.position.set(x, y, z);
  parent.add(o);
  return o;
}

function build() {
  const obj = new THREE.Group();
  obj.userData.rig = 'person_rig';
  obj.userData.asset = 'person';
  obj.userData.libraryId = 'person';
  const rig = bone('person_rig', 0, 0, 0, obj);
  const hips = bone('hips', 0, 0.96, 0, rig);
  const spine = bone('spine', 0, 0.1, 0, hips);
  const chest = bone('chest', 0, 0.2, 0, spine);
  bone('neck', 0, 0.2, 0, chest);
  const shL = bone('shoulder_L', 0.198, 0.205, -0.008, chest);
  const shR = bone('shoulder_R', -0.198, 0.205, -0.008, chest);
  bone('wrist_L', 0, -0.25, 0, bone('elbow_L', 0, -0.305, 0, shL));
  bone('wrist_R', 0, -0.25, 0, bone('elbow_R', 0, -0.305, 0, shR));
  const hipL = bone('hip_L', 0.092, -0.045, 0, hips);
  const hipR = bone('hip_R', -0.092, -0.045, 0, hips);
  bone('ankle_L', 0, -0.410, 0, bone('knee_L', 0, -0.425, 0, hipL));
  bone('ankle_R', 0, -0.410, 0, bone('knee_R', 0, -0.425, 0, hipR));
  return obj;
}

function secondMax(values) {
  let max = 0;
  for (let i = 1; i < values.length - 1; i++) {
    max = Math.max(max, Math.abs(values[i + 1] - 2 * values[i] + values[i - 1]));
  }
  return max;
}

function sample(pose) {
  const obj = build();
  bindGait(obj, { pose, kind: 'person', id: 'a' }, 0, { seed: 1, camera: { move: 'static' } }, null);
  const g = obj.userData.gait;
  const rows = [];
  for (let i = 0; i < FRAMES; i++) {
    applyGait(obj, i / (FRAMES - 1), 1, 4);
    obj.updateMatrixWorld(true);
    const ang = {};
    for (const name of JOINTS) {
      const j = obj.getObjectByName(name);
      ang[name] = j ? [j.rotation.x, j.rotation.y, j.rotation.z] : [0, 0, 0];
    }
    const ankle = new THREE.Vector3();
    obj.getObjectByName('ankle_L').getWorldPosition(ankle);
    rows.push({
      rigY: obj.getObjectByName('person_rig').position.y,
      root: [obj.position.x, obj.position.z],
      ankle: [ankle.x, ankle.z],
      ang,
    });
  }
  applyGait(obj, 0.37, 1, 4);
  const kneeA = obj.getObjectByName('knee_L').rotation.x;
  applyGait(obj, 0.8, 1, 4);
  applyGait(obj, 0.37, 1, 4);
  const scrub = Math.abs(kneeA - obj.getObjectByName('knee_L').rotation.x);
  return { g, rows, scrub };
}

function judge(pose, expectBob) {
  const { g, rows, scrub } = sample(pose);
  const problems = [];
  const fx = Math.sin(g.yaw);
  const fz = Math.cos(g.yaw);
  const sx = Math.cos(g.yaw);
  const sz = -Math.sin(g.yaw);
  const speed = ((rows[FRAMES - 1].root[0] - rows[0].root[0]) * fx + (rows[FRAMES - 1].root[1] - rows[0].root[1]) * fz) / 4;
  if (Math.abs(speed - g.speed) > 0.02) problems.push(`${pose} speed ${speed.toFixed(3)} wanted ${g.speed}`);
  const rigs = rows.map((r) => r.rigY);
  const bob = (Math.max(...rigs) - Math.min(...rigs)) * 100;
  if (Math.abs(bob - expectBob) > 0.15) problems.push(`${pose} bob ${bob.toFixed(2)} cm wanted ${expectBob}`);
  const heightD2 = secondMax(rigs) * 100;
  if (heightD2 > HEIGHT_CM) problems.push(`${pose} root second difference ${heightD2.toFixed(3)} cm`);
  let angleD2 = 0;
  for (const name of JOINTS) {
    for (const axis of [0, 1, 2]) {
      angleD2 = Math.max(angleD2, secondMax(rows.map((r) => r.ang[name][axis])) * DEG);
    }
  }
  if (angleD2 > ANGLE_DEG) problems.push(`${pose} joint second difference ${angleD2.toFixed(2)} deg`);
  if (scrub > 1e-6) problems.push(`${pose} scrub missed`);
  let slide = 0;
  let best = Infinity;
  let phase0 = 0;
  for (let k = 0; k < 360; k++) {
    const p0 = k / 360;
    let err = 0;
    for (let i = 0; i < FRAMES; i++) {
      const phase = ((i / (FRAMES - 1)) * 4 * g.hz + p0) % 1;
      if (phase >= g.stance) continue;
      const want = g.front - g.stride * phase;
      const got = (rows[i].ankle[0] - rows[i].root[0]) * fx + (rows[i].ankle[1] - rows[i].root[1]) * fz;
      err += Math.abs(got - want);
    }
    if (err < best) { best = err; phase0 = p0; }
  }
  for (let i = 0; i < FRAMES; i++) {
    const phase = ((i / (FRAMES - 1)) * 4 * g.hz + phase0) % 1;
    if (phase >= g.stance) continue;
    const gotF = (rows[i].ankle[0] - rows[i].root[0]) * fx + (rows[i].ankle[1] - rows[i].root[1]) * fz;
    const gotS = (rows[i].ankle[0] - rows[i].root[0]) * sx + (rows[i].ankle[1] - rows[i].root[1]) * sz;
    slide = Math.max(slide, Math.hypot(gotF - (g.front - g.stride * phase), gotS - g.hipX) * 100);
  }
  if (slide > 2) problems.push(`${pose} foot slide ${slide.toFixed(2)} cm`);
  const roll = Math.max(...rows.map((r) => Math.max(Math.abs(r.ang.hips[2]), Math.abs(r.ang.chest[2]), Math.abs(r.ang.neck[2])))) * DEG;
  if (roll > 0.05) problems.push(`${pose} rolled ${roll.toFixed(2)} deg`);
  return { problems, bob, slide, heightD2, angleD2, speed };
}

export function runGaitChecks() {
  const all = [
    ['walk', 4],
    ['jog', 5],
    ['run', 6],
  ].map(([pose, bob]) => ({ pose, ...judge(pose, bob) }));
  const problems = all.flatMap((r) => r.problems);
  const line = all.map((r) => `${r.pose} bob ${r.bob.toFixed(2)} cm, slide ${r.slide.toFixed(2)} cm, height Δ² ${r.heightD2.toFixed(3)} cm, angle Δ² ${r.angleD2.toFixed(2)}°`).join('; ');
  return { fail: problems.length, line, problems };
}
