// Quiet checks after a shot is built. One pass. Nothing here is shown as a popup.
import * as THREE from 'three';
import { sampleCamera } from '../cameraMath.ts';
import { applyGait } from './gait.js';
import { roomClamp } from './camera.js';

const ANIMALS = ['horse', 'cow', 'dog', 'cat', 'bird', 'giraffe', 'elephant', 'goat', 'sheep'];
const ANIMAL_RE = /^(horse|cow|dog|cat|bird|giraffe|elephant|goat|sheep)$/;

export function askedAnimals(sentence) {
  const s = String(sentence || '').toLowerCase();
  const out = new Set();
  for (const name of ANIMALS) if (new RegExp('\\b' + name + 's?\\b').test(s)) out.add(name);
  return out;
}

function shotSize(shot) {
  return (shot && shot.camera && shot.camera.shot) || 'full';
}

function wantsFeet(shot) {
  const cam = (shot && shot.camera) || {};
  if (cam.angle === 'over_shoulder' || cam.angle === 'overhead' || cam.move === 'body_mount') return false;
  const size = shotSize(shot);
  return size !== 'close' && size !== 'extreme_close' && size !== 'medium';
}

function isPov(shot) {
  const cam = (shot && shot.camera) || {};
  return cam.angle === 'pov' || cam.move === 'pov';
}

function animalId(obj) {
  const id = String((obj.userData && (obj.userData.asset || obj.userData.libraryId || obj.userData.kind)) || obj.name || '').split('__')[0].toLowerCase();
  return ANIMAL_RE.test(id) ? id : '';
}

function bonePoint(obj, name, fallback) {
  const j = obj.getObjectByName(name);
  const p = new THREE.Vector3();
  if (j) j.getWorldPosition(p);
  else obj.getWorldPosition(p).add(fallback);
  return p;
}

function predicted(obj, t, duration) {
  const p = new THREE.Vector3();
  obj.getWorldPosition(p);
  const g = obj.userData && obj.userData.gait;
  if (!g) return p;
  const dist = Math.abs(g.speed || 0) * t * (duration || 4);
  const yaw = g.yaw != null ? g.yaw : obj.rotation.y;
  const startX = g.start ? g.start[0] : p.x;
  const startZ = g.start ? g.start[1] : p.z;
  p.x = startX + Math.sin(yaw) * dist;
  p.z = startZ + Math.cos(yaw) * dist;
  return p;
}

/** Same catmull-rom, focal and roll as cameraMath.sampleCamera. t is 0–1 along the shot. */
function cameraAt(keys, t, aspect, curve = 1) {
  const list = keys && keys.length ? keys : [{ t: 0, pos: [0, 1.6, 4], look: [0, 1, 0], focal_mm: 35, roll: 0 }];
  const focal0 = list[0].focal_mm || 35;
  const spec = {
    lens: { focal_mm: focal0 },
    camera: {
      keys: list.map((k) => ({
        t: typeof k.t === 'number' ? k.t : 0,
        pos: k.pos,
        look: k.look,
        focal_mm: k.focal_mm ?? focal0,
        roll: k.roll ?? 0,
      })),
      ease: 'linear',
      taper: 0,
      distance: 1,
      speed: 1,
      curve,
    },
  };
  const sample = sampleCamera(spec, t);
  const cam = camFromSample(sample, aspect);
  if (sample.roll) cam.rotateZ((sample.roll * Math.PI) / 180);
  cam.updateMatrixWorld(true);
  return cam;
}

/** Every 12 frames of a 97-frame shot, plus each key (segment joins live on key t). */
function sampleTimes(keys) {
  const marks = new Set();
  for (let frame = 0; frame <= 96; frame += 12) marks.add(frame / 96);
  for (const key of keys || []) if (typeof key.t === 'number') marks.add(Math.max(0, Math.min(1, key.t)));
  return [...marks].sort((a, b) => a - b);
}

function ndcOf(cam, point) {
  return point.clone().project(cam);
}

function inFrame(ndc) {
  return ndc.z >= -1 && ndc.z <= 1 && Math.abs(ndc.x) <= 1 && Math.abs(ndc.y) <= 1;
}

function isTPose(obj) {
  const names = ['shoulder_L', 'elbow_L', 'hip_L', 'knee_L'];
  let n = 0;
  let flat = 0;
  for (const name of names) {
    const j = obj.getObjectByName(name);
    if (!j) continue;
    n += 1;
    if (Math.hypot(j.rotation.x, j.rotation.y, j.rotation.z) < 0.08) flat += 1;
  }
  return n >= 3 && flat === n;
}

function relax(obj) {
  const set = (name, x, y, z) => {
    const j = obj.getObjectByName(name);
    if (j) j.rotation.set(x, y, z);
  };
  const pose = obj.userData && obj.userData.activity;
  if (pose === 'ride') {
    set('hip_L', -0.55, 0.1, 0.2);
    set('hip_R', -0.5, -0.1, -0.2);
    set('knee_L', 0.9, 0, 0);
    set('knee_R', 0.85, 0, 0);
    set('shoulder_L', -0.1, 0, 0.45);
    set('shoulder_R', 0.1, 0, -0.42);
    set('elbow_L', -0.35, 0, 0);
    set('elbow_R', -0.32, 0, 0);
    return;
  }
  set('hip_L', 0.04, 0, 0.02);
  set('hip_R', -0.02, 0, -0.02);
  set('knee_L', 0.06, 0, 0);
  set('knee_R', 0.05, 0, 0);
  set('shoulder_L', 0.15, 0, 0.18);
  set('shoulder_R', 0.12, 0, -0.16);
  set('elbow_L', -0.35, 0, 0);
  set('elbow_R', -0.32, 0, 0);
}

/** A dance hold, a hug or a kiss is meant to touch. */
function held(a, b) {
  const ha = a.userData && a.userData.holdWith, hb = b.userData && b.userData.holdWith;
  return !!ha && ha === hb;
}

function moveSpawn(obj, dx, dz) {
  obj.position.x += dx;
  obj.position.z += dz;
  obj.updateMatrixWorld(true);
  const g = obj.userData && obj.userData.gait;
  if (g && g.start) {
    g.start[0] = obj.position.x;
    g.start[1] = obj.position.z;
  }
}

function peopleOf(built) {
  return (built.actors || []).map((a) => a.object || a).filter((o) => o && o.getObjectByName);
}

function fogFarOf(built) {
  const fog = built.scene && built.scene.fog;
  if (fog && fog.far) return fog.far;
  const w = built.kitWorld || (built.kit && built.kit.world) || {};
  return w.fogFar || 80;
}

/**
 * guardShot({ built, keys, aspect, sentence, shot, locked }) -> { keys, notes }
 * One correction each. Azimuth, lens and shot type stay as the sentence asked.
 */
export function guardShot({ built, keys, aspect, sentence, shot, locked }) {
  const notes = [];
  const actors = peopleOf(built);
  const asked = askedAnimals(sentence);
  const duration = (built.spec && built.spec.duration_s) || 4;
  const root = (built.kit && (built.kit.set || built.kit.dressing)) || built.root;

  for (const obj of actors) {
    if (!isTPose(obj)) continue;
    relax(obj);
    notes.push(obj.userData && obj.userData.activity === 'ride' ? 'Pose set to a riding stance' : 'Pose set to a relaxed idle');
  }

  const samples = sampleTimes(keys);
  let separated = false;
  for (let pass = 0; pass < 4; pass++) {
    let moved = false;
    for (let i = 0; i < actors.length; i++) {
      for (let j = i + 1; j < actors.length; j++) {
        let hit = false;
        if (held(actors[i], actors[j])) continue;
        for (const t of samples) {
          const a = predicted(actors[i], t, duration);
          const b = predicted(actors[j], t, duration);
          if (Math.hypot(a.x - b.x, a.z - b.z) < 0.6) hit = true;
        }
        if (!hit) continue;
        moveSpawn(actors[j], 0.7, 0.2 * (pass + 1));
        moved = true;
        separated = true;
      }
    }
    if (!moved) break;
  }
  if (separated) notes.push('Moved a person clear of another');

  const blockers = [];
  if (root && root.traverse) {
    root.traverse((n) => {
      if (!n.isMesh && !(n.userData && n.userData.asset)) return;
      const id = String((n.userData && (n.userData.asset || n.userData.libraryId)) || '');
      if (/table|chair|bench|car|suv|truck|van|bus|bicycle|skateboard/.test(id)) blockers.push(n);
    });
  }
  for (const obj of actors) {
    if (obj.userData && obj.userData.seated) continue;
    const p = predicted(obj, 0, duration);
    for (const block of blockers) {
      if (obj.parent === block || block.parent === obj) continue;
      const bp = new THREE.Vector3();
      block.getWorldPosition(bp);
      if (Math.hypot(p.x - bp.x, p.z - bp.z) >= 0.45) continue;
      moveSpawn(obj, 0.8, 0);
      notes.push('Moved a person clear of a prop');
      break;
    }
  }

  const hideAnimal = (obj) => {
    let node = obj;
    while (node.parent && node.parent !== root && !animalId(node)) node = node.parent;
    const target = animalId(node) ? node : obj;
    target.visible = false;
    target.position.set(80, -20, 80);
    if (target.userData) target.userData.guardHidden = true;
  };
  const seen = new Set();
  const cams = sampleTimes(keys).map((t) => cameraAt(keys, t, aspect));
  const scan = (obj) => {
    if (!obj || seen.has(obj) || obj.userData?.guardHidden) return;
    const id = animalId(obj);
    if (!id || asked.has(id)) return;
    seen.add(obj);
    let close = false;
    const ap = new THREE.Vector3();
    obj.getWorldPosition(ap);
    for (const person of actors) {
      for (const t of samples) {
        const p = predicted(person, t, duration);
        if (Math.hypot(ap.x - p.x, ap.z - p.z) < 12) close = true;
      }
    }
    let framed = false;
    for (const cam of cams) {
      if (inFrame(ndcOf(cam, ap))) framed = true;
    }
    if (close || framed) {
      hideAnimal(obj);
      notes.push('Hid an animal the sentence didn\'t ask for');
    }
  };
  if (root && root.traverse) root.traverse(scan);
  for (const extra of built.extras || []) scan(extra);

  const fogFar = fogFarOf(built);
  if (root && root.traverse) {
    root.traverse((n) => {
      if (!n.isMesh) return;
      const name = String(n.name || '');
      if (!/ground|backdrop|slab/i.test(name) && !(n.userData && n.userData.guardEdge)) return;
      const box = new THREE.Box3().setFromObject(n);
      if (!Number.isFinite(box.min.x)) return;
      const size = box.getSize(new THREE.Vector3());
      if (size.x < 8 && size.z < 8) return;
      const corners = [
        new THREE.Vector3(box.min.x, 0.02, box.min.z),
        new THREE.Vector3(box.max.x, 0.02, box.min.z),
        new THREE.Vector3(box.min.x, 0.02, box.max.z),
        new THREE.Vector3(box.max.x, 0.02, box.max.z),
      ];
      let exposed = false;
      for (const cam of cams) {
        for (const c of corners) {
          const dist = cam.position.distanceTo(c);
          if (dist < fogFar * 0.92 && inFrame(ndcOf(cam, c))) exposed = true;
        }
      }
      if (!exposed) return;
      const need = Math.max(1.4, (fogFar * 1.15) / Math.max(size.x, size.z, 1));
      n.scale.multiplyScalar(Math.min(need, 4));
      n.updateMatrixWorld(true);
      notes.push('Ground extended past the fog');
    });
  }

  let next = keys;
  const skipFit = (shot && shot.camera && (shot.camera.shot === 'extreme_close' || shot.camera.angle === 'overhead' || shot.camera.angle === 'over_shoulder' || shot.camera.move === 'body_mount')) || /\b(insert|whip pan|over the shoulder|snorricam|balcony)\b/i.test(sentence || '');
  if (!locked && !isPov(shot) && !skipFit && actors.length && keys && keys.length) {
    const feet = wantsFeet(shot);
    const clipped = (t) => {
      const cam = cameraAt(keys, t, aspect);
      const obj = actors[0];
      const head = bonePoint(obj, 'head', new THREE.Vector3(0, 1.6, 0));
      const foot = bonePoint(obj, 'ankle_L', new THREE.Vector3(0, 0.05, 0));
      const hp = ndcOf(cam, head);
      const fp = ndcOf(cam, foot);
      const height = Math.abs(hp.y - fp.y) / 2;
      const cut = hp.y > 1.02 || Math.abs(hp.x) > 1.02 || (feet && (fp.y < -1.02 || Math.abs(fp.x) > 1.05)) || (t === 0 && height > 0 && height < 0.05);
      return { cut, head, foot, cam };
    };
    const start = clipped(0);
    const end = clipped(1);
    const head = start.head;
    const foot = start.foot;
    const cam0 = start.cam;
    if (start.cut || end.cut) {
      const body = Math.max(0.4, head.y - foot.y);
      const half = Math.tan(cam0.fov * Math.PI / 360);
      const dist = feet ? body / (0.62 * 2 * half) : Math.max(0.8, body / (0.42 * 2 * half));
      const lookY = feet ? (head.y + foot.y) / 2 : head.y - body * 0.15;
      let minH = Infinity;
      for (const k of keys) minH = Math.min(minH, Math.max(0.4, Math.hypot(k.pos[0] - k.look[0], k.pos[2] - k.look[2])));
      const s = dist / minH;
      next = keys.map((k) => {
        const dx = k.pos[0] - k.look[0];
        const y = k.pos[1] - k.look[1];
        const dz = k.pos[2] - k.look[2];
        return {
          ...k,
          look: [k.look[0], lookY, k.look[2]],
          pos: [k.look[0] + dx * s, lookY + y * s, k.look[2] + dz * s],
        };
      });
      notes.push(feet ? 'Framing adjusted to keep the feet in' : 'Framing adjusted to keep the subject readable');
    }
  }

  if (built && built.kit && built.kit.set) next = roomClamp(next, built);
  if (!locked && !isPov(shot) && !skipFit && actors.length && next && next.length && !/\b(reveal|leaves|out of frame)\b/i.test(sentence || '')) {
    const head = bonePoint(actors[0], 'head', new THREE.Vector3(0, 1.6, 0));
    let off = false;
    for (const t of sampleTimes(next)) {
      const p = ndcOf(cameraAt(next, t, aspect), head);
      const x = (p.x + 1) / 2;
      const y = (p.y + 1) / 2;
      if (p.z < -1 || p.z > 1 || x < 0.1 || x > 0.9 || y < 0.1 || y > 0.9) off = true;
    }
    if (off) {
      next = next.map((k) => {
        const pos = new THREE.Vector3().fromArray(k.pos);
        const look = new THREE.Vector3(head.x, head.y * 0.9, head.z);
        const dir = pos.clone().sub(new THREE.Vector3().fromArray(k.look));
        if (dir.length() < 0.5) dir.set(0, 0.3, 2.4);
        const np = look.clone().add(dir);
        np.y = Math.max(0.35, np.y);
        return { ...k, look: [look.x, look.y, look.z], pos: [np.x, np.y, np.z] };
      });
      notes.push('Camera moved so the subject stays in frame');
    }
  }

  return { keys: next, notes: [...new Set(notes)] };
}

/** Problems still present after a guard pass. Empty means the shot passes. */
export function auditShot({ built, keys, aspect, sentence, shot }) {
  const problems = [];
  const actors = peopleOf(built);
  const asked = askedAnimals(sentence);
  const duration = (built.spec && built.spec.duration_s) || 4;
  const root = (built.kit && (built.kit.set || built.kit.dressing)) || built.root;
  for (const obj of actors) if (isTPose(obj)) problems.push('t-pose');
  for (let i = 0; i < actors.length; i++) {
    for (let j = i + 1; j < actors.length; j++) {
      if (held(actors[i], actors[j])) continue;
      for (const t of [0, 0.5, 1]) {
        const a = predicted(actors[i], t, duration);
        const b = predicted(actors[j], t, duration);
        if (Math.hypot(a.x - b.x, a.z - b.z) < 0.6) problems.push('overlap');
      }
    }
  }
  const cams = sampleTimes(keys).map((t) => cameraAt(keys, t, aspect));
  if (root && root.traverse) {
    root.traverse((n) => {
      const id = animalId(n);
      if (!id || asked.has(id) || !n.visible) return;
      const ap = new THREE.Vector3();
      n.getWorldPosition(ap);
      for (const person of actors) {
        for (const t of [0, 0.5, 1]) {
          const p = predicted(person, t, duration);
          if (Math.hypot(ap.x - p.x, ap.z - p.z) < 12) problems.push('animal');
        }
      }
      for (const cam of cams) if (inFrame(ndcOf(cam, ap))) problems.push('animal-frame');
    });
    const fogFar = fogFarOf(built);
    root.traverse((n) => {
      if (!n.isMesh || !/ground|backdrop|slab/i.test(String(n.name || ''))) return;
      const box = new THREE.Box3().setFromObject(n);
      const size = box.getSize(new THREE.Vector3());
      if (size.x < 8 && size.z < 8) return;
      const corners = [
        new THREE.Vector3(box.min.x, 0.02, box.min.z),
        new THREE.Vector3(box.max.x, 0.02, box.max.z),
      ];
      for (const cam of cams) {
        for (const c of corners) {
          if (cam.position.distanceTo(c) < fogFar * 0.92 && inFrame(ndcOf(cam, c))) problems.push('edge');
        }
      }
    });
  }
  if (!isPov(shot) && actors.length && keys && keys.length) {
    const feet = wantsFeet(shot);
    const cam0 = cameraAt(keys, 0, aspect);
    for (const obj of actors) {
      const head = bonePoint(obj, 'head', new THREE.Vector3(0, 1.6, 0));
      const foot = bonePoint(obj, 'ankle_L', new THREE.Vector3(0, 0.05, 0));
      const hp = ndcOf(cam0, head);
      const fp = ndcOf(cam0, foot);
      if (hp.y > 1.02 || Math.abs(hp.x) > 1.02) problems.push('framing');
      if (feet && (fp.y < -1.02 || Math.abs(fp.x) > 1.05)) problems.push('framing');
      if (Math.abs(hp.y - fp.y) / 2 < 0.05) problems.push('framing');
    }
  }
  return [...new Set(problems)];
}

const PLAY_FRAMES = [0, 24, 48, 72, 95];

function camFromSample(sample, aspect) {
  const filmH = 36 / Math.max(aspect || 1, 1);
  const fov = 2 * Math.atan((filmH / 2) / Math.max(8, sample.focal)) * 180 / Math.PI;
  const cam = new THREE.PerspectiveCamera(fov, aspect || 16 / 9, 0.05, 4000);
  cam.position.set(sample.pos[0], sample.pos[1], sample.pos[2]);
  cam.up.set(0, 1, 0);
  cam.lookAt(sample.look[0], sample.look[1], sample.look[2]);
  cam.updateMatrixWorld(true);
  return cam;
}

function shiftOf(built, t) {
  const obj = built.subject && built.subject.object;
  if (!obj) return new THREE.Vector3();
  const dur = (built.spec && built.spec.duration_s) || 4;
  const ride = obj.parent && obj.parent.userData && obj.parent.userData.cruise;
  if (ride) {
    const dist = ride.speed * t * dur;
    return new THREE.Vector3(Math.sin(ride.yaw) * dist, 0, Math.cos(ride.yaw) * dist);
  }
  const g = obj.userData && obj.userData.gait;
  if (!g || g.follow === false) return new THREE.Vector3();
  const dist = (g.speed || 0) * t * dur;
  const yaw = g.yaw != null ? g.yaw : obj.rotation.y || 0;
  return new THREE.Vector3(Math.sin(yaw) * dist, 0, Math.cos(yaw) * dist);
}

function posePlayback(built, t) {
  const dur = (built.spec && built.spec.duration_s) || 4;
  for (const obj of peopleOf(built)) {
    const ride = obj.parent && obj.parent.userData && obj.parent.userData.cruise;
    if (ride) {
      const dist = ride.speed * t * dur;
      obj.parent.position.x = ride.start[0] + Math.sin(ride.yaw) * dist;
      obj.parent.position.z = ride.start[1] + Math.cos(ride.yaw) * dist;
      obj.parent.updateMatrixWorld(true);
    } else if (obj.userData && obj.userData.gait) applyGait(obj, t, 1, dur);
  }
  const root = built.root;
  if (root && root.traverse) {
    const w = t * Math.PI * 2;
    root.traverse((n) => {
      const home = n.userData && n.userData.kiteHome;
      if (!n.userData || !n.userData.kiteFloat || !home) return;
      n.position.set(home.x + Math.sin(w) * 0.16, home.y + Math.sin(w * 2) * 0.1, home.z + Math.cos(w) * 0.12);
    });
    root.updateMatrixWorld(true);
  }
}

/**
 * auditMotion(built) -> problems. Every frame of the playblast, not five:
 *  - 'frozen': a walker whose path runs out before the shot does (they would march on the spot), or a person in a
 *    walk, jog or run pose with no gait at all.
 *  - 'collision': two people who are not holding each other come closer than 0.6 m, centre to centre, at any frame.
 */
export function auditMotion(built, { frames = 97 } = {}) {
  const problems = [];
  const dur = (built.spec && built.spec.duration_s) || 4;
  const people = peopleOf(built);
  const walkers = people.filter((o) => o.userData && o.userData.gait && !o.userData.gait.animal);
  for (const o of people) {
    const g = o.userData && o.userData.gait;
    if (!g && /^(walk|jog|run)$/.test(String(o.userData && o.userData.pose))) problems.push('frozen');
    // a dancer or a typist is a moving subject too: they need a timed motion, not a held pose
    if (!o.userData.extra && /^(dance|type)$/.test(String(o.userData && o.userData.pose)) && !o.userData.motion) problems.push('frozen');
    if (g && !g.animal && g.maxDist < Math.abs(g.speed) * dur - 0.02) problems.push('frozen');
  }
  if (walkers.length) {
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    for (let f = 0; f < frames; f++) {
      posePlayback(built, f / (frames - 1));
      for (let i = 0; i < people.length; i++) {
        people[i].getWorldPosition(a);
        for (let j = i + 1; j < people.length; j++) {
          if (held(people[i], people[j])) continue;
          if (!people[i].userData.gait && !people[j].userData.gait) continue;
          people[j].getWorldPosition(b);
          if (Math.hypot(a.x - b.x, a.z - b.z) < 0.6) problems.push('collision');
        }
      }
    }
    posePlayback(built, 0);
  }
  return [...new Set(problems)];
}

/**
 * Audit a real build at frames 0, 24, 48, 72 and 95.
 * The camera is cameraMath sampleCamera (catmull-rom, with focal_mm), plus the same follow shift the app adds.
 */
export function auditPlayback({ built, spec, aspect, sentence, shot }) {
  const problems = [];
  const actors = peopleOf(built);
  const asked = askedAnimals(sentence);
  const pov = isPov(shot);
  const feet = wantsFeet(shot) || pov;
  const root = (built.kit && built.kit.set) || built.root;
  const dressing = built.kit && built.kit.dressing;
  for (const frame of PLAY_FRAMES) {
    const t = frame / 95;
    posePlayback(built, t);
    const shift = shiftOf(built, t);
    const sample = sampleCamera(spec, t);
    const cam = camFromSample({
      pos: [sample.pos[0] + shift.x, sample.pos[1] + shift.y, sample.pos[2] + shift.z],
      look: [sample.look[0] + shift.x, sample.look[1] + shift.y, sample.look[2] + shift.z],
      focal: sample.focal,
    }, aspect);
    let tallest = 0;
    for (const obj of actors) {
      if (isTPose(obj)) problems.push('t-pose');
      const head = bonePoint(obj, 'head', new THREE.Vector3(0, 1.6, 0));
      const top = head.clone();
      top.y += 0.14;
      const fl = bonePoint(obj, 'ankle_L', new THREE.Vector3(0, 0.05, 0));
      const fr = bonePoint(obj, 'ankle_R', new THREE.Vector3(0, 0.05, 0));
      const limit = pov ? 0.94 : 1.02;
      const outside = (p, yMax) => p.z < -1 || p.z > 1 || Math.abs(p.x) > 1.02 || p.y > yMax || p.y < -1.02;
      const headN = ndcOf(cam, pov ? top : head);
      const flN = ndcOf(cam, fl);
      const frN = ndcOf(cam, fr);
      const sx = (headN.x + 1) / 2;
      const sy = (headN.y + 1) / 2;
      if (!pov && actors[0] === obj && !/\b(reveal|leaves)\b/i.test(sentence || '') && (headN.z < -1 || headN.z > 1 || sx < 0.1 || sx > 0.9 || sy < 0.1 || sy > 0.9)) problems.push('subject');
      const passing = obj.userData && obj.userData.passing; // an oncoming passer-by is meant to walk out of frame
      if (!passing && (outside(headN, limit) || (feet && (outside(flN, 1.02) || outside(frN, 1.02))))) {
        if (typeof process !== 'undefined' && process.env && process.env.GUARD_DEBUG) console.log('frame', frame, 'head', headN.x.toFixed(2), headN.y.toFixed(2), 'L', flN.x.toFixed(2), flN.y.toFixed(2), 'R', frN.x.toFixed(2), frN.y.toFixed(2), 'cam', cam.position.x.toFixed(2), cam.position.y.toFixed(2), cam.position.z.toFixed(2));
        problems.push('framing');
      }
      const height = Math.abs(ndcOf(cam, head).y - ndcOf(cam, fl).y) / 2;
      if (height > tallest) tallest = height;
      if (height > 0 && height < 0.05) problems.push('framing');
    }
    if (frame === 95 && actors.length >= 6 && tallest < 0.2) problems.push('crowd-height');
    for (let i = 0; i < actors.length; i++) {
      for (let j = i + 1; j < actors.length; j++) {
        if (held(actors[i], actors[j])) continue;
        const a = new THREE.Vector3();
        const b = new THREE.Vector3();
        actors[i].getWorldPosition(a);
        actors[j].getWorldPosition(b);
        if (Math.hypot(a.x - b.x, a.z - b.z) < 0.6) problems.push('overlap');
      }
    }
    if (dressing) {
      for (const c of dressing.children) {
        if (!c.visible) continue;
        const id = String((c.userData && (c.userData.asset || c.userData.libraryId)) || '').split('__')[0];
        if (!/^(car|suv|truck|van|bus)$/.test(id)) continue;
        const box = new THREE.Box3().setFromObject(c);
        if (box.isEmpty()) continue;
        const closest = box.clampPoint(cam.position, new THREE.Vector3());
        if (cam.position.distanceTo(closest) < 1.5 || box.containsPoint(cam.position)) problems.push('lens');
      }
    }
    const fogFar = fogFarOf(built);
    const scanEdge = (n) => {
      if (!n.visible || !n.isMesh) return;
      const name = String(n.name || '');
      if (/^(hill_|barn|barn_roof|geo_dune)/.test(name)) {
        const p = new THREE.Vector3();
        n.getWorldPosition(p);
        if (inFrame(ndcOf(cam, p))) problems.push('edge');
        return;
      }
      if (!/ground|backdrop|slab/i.test(name)) return;
      const box = new THREE.Box3().setFromObject(n);
      const size = box.getSize(new THREE.Vector3());
      if (size.x < 8 && size.z < 8) return;
      for (const c of [new THREE.Vector3(box.min.x, 0.02, box.min.z), new THREE.Vector3(box.max.x, 0.02, box.max.z)]) {
        if (cam.position.distanceTo(c) < fogFar * 0.92 && inFrame(ndcOf(cam, c))) problems.push('edge');
      }
    };
    if (root && root.traverse) root.traverse(scanEdge);
    const scanAnimal = (n) => {
      const id = animalId(n);
      if (!id || asked.has(id) || n.visible === false) return;
      const ap = new THREE.Vector3();
      n.getWorldPosition(ap);
      for (const person of actors) {
        const p = new THREE.Vector3();
        person.getWorldPosition(p);
        if (Math.hypot(ap.x - p.x, ap.z - p.z) < 12) problems.push('animal');
      }
      if (inFrame(ndcOf(cam, ap))) problems.push('animal-frame');
    };
    if (root && root.traverse) root.traverse(scanAnimal);
    if (dressing && dressing.traverse) dressing.traverse(scanAnimal);
  }
  problems.push(...auditMotion(built));
  return [...new Set(problems)];
}
