// Small timed body motion on top of a held pose: an argument, a slow-dance sway, hands typing.
// Pure function of t (0..1) and the clip length, so the same sentence always gives the same frames.
import * as THREE from 'three';

const DEG = Math.PI / 180;
const TAU = Math.PI * 2;
const JOINTS = ['spine', 'chest', 'neck', 'head', 'shoulder_L', 'shoulder_R', 'elbow_L', 'elbow_R', 'wrist_L', 'wrist_R'];

/** bindMotion(person, kind, index): remember the held pose, so applyMotion adds to it. kind: argue | sway | type | dance | chat | gaze. */
export function bindMotion(person, kind, index = 0) {
  const base = {};
  for (const name of JOINTS) {
    const j = person.getObjectByName(name);
    if (j) base[name] = [j.rotation.x, j.rotation.y, j.rotation.z];
  }
  const rig = person.getObjectByName((person.userData && person.userData.rig) || 'person_rig');
  person.userData.motion = {
    kind,
    index,
    phase: index * 1.9,
    base,
    rig: rig ? rig.position.toArray() : null,
    home: person.position.toArray(),
    yaw: person.rotation.y,
    last: person.position.toArray(),
  };
  return person;
}

function turn(person, name, base, dx, dy, dz) {
  const j = person.getObjectByName(name);
  if (!j || !base[name]) return;
  const b = base[name];
  j.rotation.set(b[0] + dx * DEG, b[1] + dy * DEG, b[2] + dz * DEG);
}

const ease = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/** applyMotion(person, t, duration_s): pose the joints for this moment. No-op without bindMotion. */
export function applyMotion(person, t, duration = 4) {
  const m = person && person.userData && person.userData.motion;
  if (!m) return;
  const s = Math.max(0, Math.min(1, t)) * duration;
  const p = m.phase;
  const b = m.base;
  // someone moved the person after binding (the shot guard's spacing): carry the move into home
  if (m.last && (Math.abs(person.position.x - m.last[0]) > 1e-6 || Math.abs(person.position.z - m.last[2]) > 1e-6)) {
    m.home = [m.home[0] + person.position.x - m.last[0], m.home[1], m.home[2] + person.position.z - m.last[2]];
    if (m.center) m.center = [m.center[0] + person.position.x - m.last[0], m.center[1], m.center[2] + person.position.z - m.last[2]];
  }
  if (m.kind === 'argue') {
    // Hands cut the air, the chest turns into each point, the head pushes forward, the weight rocks.
    const beat = Math.sin(TAU * 0.85 * s + p);
    const jab = Math.max(0, Math.sin(TAU * 0.55 * s + p + 0.8));
    const lean = ease((s - 0.6) / 1.6) * (m.index === 0 ? 1 : -0.6);
    turn(person, 'shoulder_R', b, -22 * beat - 14, 6 * beat, 10 * jab);
    turn(person, 'elbow_R', b, 28 * Math.sin(TAU * 1.2 * s + p), 0, 0);
    turn(person, 'wrist_R', b, 14 * Math.sin(TAU * 1.6 * s + p), 0, 0);
    turn(person, 'shoulder_L', b, -62 * jab, -10 * jab, -26 * jab);
    turn(person, 'elbow_L', b, -70 * jab, 0, 0);
    turn(person, 'spine', b, 6 * lean, 0, 3 * Math.sin(TAU * 0.4 * s + p));
    turn(person, 'chest', b, 4 * lean + 3 * jab, 10 * Math.sin(TAU * 0.55 * s + p), 0);
    turn(person, 'neck', b, 6 * Math.sin(TAU * 1.4 * s + p), 0, 0);
    turn(person, 'head', b, 5 * Math.sin(TAU * 1.4 * s + p + 0.6), 8 * Math.sin(TAU * 0.5 * s + p), 0);
    const rig = person.getObjectByName((person.userData && person.userData.rig) || 'person_rig');
    if (rig && m.rig) rig.position.set(m.rig[0] + 0.035 * Math.sin(TAU * 0.45 * s + p), m.rig[1], m.rig[2]);
    // one presses in half a step, the other gives a little ground
    const step = ease((s - 1.2) / 1.4) * (m.index === 0 ? 0.32 : -0.12);
    person.position.set(m.home[0] + Math.sin(m.yaw) * step, m.home[1], m.home[2] + Math.cos(m.yaw) * step);
  } else if (m.kind === 'sway') {
    // Slow dance: the pair rocks side to side together and turns slowly on the spot.
    const rock = Math.sin(TAU * 0.5 * s);
    turn(person, 'spine', b, 0, 0, 4 * rock);
    turn(person, 'chest', b, 0, 3 * rock, -2 * rock);
    turn(person, 'head', b, 0, 4 * rock, 3 * rock);
    const rig = person.getObjectByName((person.userData && person.userData.rig) || 'person_rig');
    if (rig && m.rig) rig.position.set(m.rig[0] + 0.03 * rock * (m.index === 0 ? 1 : -1), m.rig[1], m.rig[2]);
    if (m.center) {
      const a = (m.turn0 || 0) + (m.spin || 0) * ease(s / duration);
      const c = new THREE.Vector3().fromArray(m.center);
      const off = new THREE.Vector3().fromArray(m.home).sub(c).applyAxisAngle(new THREE.Vector3(0, 1, 0), a);
      person.position.set(c.x + off.x, m.home[1], c.z + off.z);
      person.rotation.y = m.yaw + a;
    }
  } else if (m.kind === 'chat') {
    // Seated conversation: nods, a turn of the head, one hand lifting off the table on a point.
    const talk = Math.max(0, Math.sin(TAU * 0.45 * s + p));
    turn(person, 'neck', b, 5 * Math.sin(TAU * 1.1 * s + p), 0, 0);
    turn(person, 'head', b, 4 * Math.sin(TAU * 0.9 * s + p + 0.5), 7 * Math.sin(TAU * 0.35 * s + p), 0);
    turn(person, 'chest', b, -3 * talk, 5 * Math.sin(TAU * 0.3 * s + p), 0);
    turn(person, 'shoulder_R', b, -18 * talk, 0, 6 * talk);
    turn(person, 'elbow_R', b, -30 * talk, 0, 0);
    turn(person, 'wrist_R', b, 12 * talk * Math.sin(TAU * 1.5 * s), 0, 0);
  } else if (m.kind === 'dance') {
    // A solo dancer on a 120 bpm beat: the weight rocks side to side, the chest turns into it, the head nods on the
    // beat and the arms swing and lift in turn, so the figure is never still.
    const beat = TAU * 2 * s + p;
    const bar = TAU * 0.5 * s + p;
    const rock = Math.sin(TAU * s + p);
    turn(person, 'spine', b, 0, 0, 6 * rock);
    turn(person, 'chest', b, 0, 12 * Math.sin(bar), -4 * rock);
    turn(person, 'neck', b, 5 * Math.sin(beat), 0, 0);
    turn(person, 'head', b, 4 * Math.sin(beat + 0.4), 10 * Math.sin(bar + 0.6), 0);
    turn(person, 'shoulder_L', b, -28 * Math.sin(bar), 0, 14 * Math.max(0, Math.sin(bar + 1)));
    turn(person, 'shoulder_R', b, 28 * Math.sin(bar), 0, -14 * Math.max(0, Math.sin(bar + 1 + Math.PI)));
    turn(person, 'elbow_L', b, -22 * Math.max(0, Math.sin(beat)), 0, 0);
    turn(person, 'elbow_R', b, -22 * Math.max(0, Math.sin(beat + Math.PI)), 0, 0);
    turn(person, 'wrist_L', b, 10 * Math.sin(beat), 0, 0);
    turn(person, 'wrist_R', b, 10 * Math.sin(beat + Math.PI), 0, 0);
    const rig = person.getObjectByName((person.userData && person.userData.rig) || 'person_rig');
    if (rig && m.rig) rig.position.set(m.rig[0] + 0.04 * rock, m.rig[1], m.rig[2]);
  } else if (m.kind === 'gaze') {
    // the eyes are on the window from the first frame; at the end the head comes round to the lens and settles
    const g = m.gaze;
    const from = g.hasOut ? g.outDeg : 0;
    const to = g.hasCam && g.camDeg != null ? g.camDeg : from;
    const u = ease((Math.max(0, Math.min(1, t)) - g.t0) / Math.max(1e-3, g.t1 - g.t0));
    setGazeDeg(person, b, from + (to - from) * u);
  } else if (m.kind === 'type') {
    // Slow typing: hands flat on the keys, palms down, fingers forward. Each hand lifts its fingers a little on its own
    // easy rhythm (the wrist's z pivot: up is +z on the left, -z on the right) and drifts a touch across the keys.
    // Nothing rolls the hands on their side.
    const l = Math.max(0, Math.sin(TAU * 1.6 * s + p));
    const r = Math.max(0, Math.sin(TAU * 1.9 * s + p + 1.3));
    turn(person, 'wrist_L', b, 0, 0, 5 * l);
    turn(person, 'wrist_R', b, 0, 0, -5 * r);
    turn(person, 'shoulder_L', b, 0, 2 * Math.sin(TAU * 0.35 * s), 0);
    turn(person, 'shoulder_R', b, 0, -2 * Math.sin(TAU * 0.3 * s + 2), 0);
    turn(person, 'elbow_L', b, 1.5 * l, 0, 0);
    turn(person, 'elbow_R', b, 1.5 * r, 0, 0);
  }
  m.last = person.position.toArray();
  person.updateMatrixWorld(true);
}


// ---------- gaze: "looking out the window ... at the end he turns and looks at the camera" ----------
/** How a head turn is shared out along the spine. The upper body carries close to half, so on a mannequin (no face) the
 * turn reads from the shoulders, not only from the head. */
const GAZE_SHARE = { head: 0.35, neck: 0.2, chest: 0.25, spine: 0.2 };
/** A turn to the lens settles here (frame 92 of 97) and holds, so the last frames look straight into the lens. */
const GAZE_SETTLE = 0.955;
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function faceYaw(person) {
  const head = person.getObjectByName('head');
  person.updateMatrixWorld(true);
  const f = new THREE.Vector3(0, 0, 1).applyQuaternion(head.getWorldQuaternion(new THREE.Quaternion()));
  return Math.atan2(f.x, f.z);
}
function setGazeDeg(person, base, deg) {
  for (const [name, share] of Object.entries(GAZE_SHARE)) turn(person, name, base, 0, deg * share, 0);
  person.updateMatrixWorld(true);
}
/** Degrees of shared head turn that put the face toward the world point (sign and gain measured, not assumed). */
function solveGaze(person, base, point) {
  const head = person.getObjectByName('head');
  setGazeDeg(person, base, 0);
  const y0 = faceYaw(person);
  setGazeDeg(person, base, 10);
  const gain = Math.min(2, Math.max(0.3, Math.abs(wrapPi(faceYaw(person) - y0)) / (10 * DEG))) * (wrapPi(faceYaw(person) - y0) < 0 ? -1 : 1);
  const hp = head.getWorldPosition(new THREE.Vector3());
  const want = Math.atan2(point.x - hp.x, point.z - hp.z);
  let deg = 0;
  for (let k = 0; k < 8; k++) {
    setGazeDeg(person, base, deg);
    const err = wrapPi(want - faceYaw(person));
    if (Math.abs(err) < 0.004) break;
    deg = Math.max(-150, Math.min(150, deg + err / DEG / gain));
  }
  setGazeDeg(person, base, 0);
  return deg;
}

/**
 * bindGaze(person, beats, outPoint): hold a pose and turn the head on a schedule. beats: this person's { type, from_t, to_t }.
 * 'look_out' = the face is toward outPoint (a window) from the first frame. 'look_at_camera' = it turns to the lens between
 * from_t and to_t. The lens turn needs the camera, so it waits for aimGaze. Returns true when something was bound.
 */
export function bindGaze(person, beats, outPoint = null) {
  const cam = (beats || []).find((b) => b.type === 'look_at_camera');
  const out = outPoint && (beats || []).some((b) => b.type === 'look_out');
  if (!cam && !out) return false;
  bindMotion(person, 'gaze');
  const m = person.userData.motion;
  m.gaze = { outDeg: 0, camDeg: null, t0: cam ? cam.from_t : 0, t1: cam ? Math.max(cam.from_t + 0.05, Math.min(cam.to_t, GAZE_SETTLE)) : 1, hasCam: !!cam, hasOut: !!out };
  if (out) m.gaze.outDeg = solveGaze(person, m.base, outPoint);
  applyMotion(person, 0, 4);
  return true;
}

/** aimGaze(built, camPos): the lens turn of every bound gaze goes to the camera's last position. */
export function aimGaze(built, camPos) {
  const p = Array.isArray(camPos) ? new THREE.Vector3(...camPos) : camPos;
  let n = 0;
  for (const a of (built && built.actors) || []) {
    const o = a.object, m = o && o.userData && o.userData.motion;
    if (!m || m.kind !== 'gaze' || !m.gaze.hasCam) continue;
    m.gaze.camDeg = solveGaze(o, m.base, p);
    applyMotion(o, 0, 4);
    n++;
  }
  return n;
}

/**
 * Vehicles on a timed drive: userData.drive = { start: [x, z], yaw, speed, lanes: [[t0, t1, sideMetres], ...] }.
 * The lane shifts ease in and out and the body turns into each one. Returns the lateral and forward travel at t.
 */
/** How far a timed drive has eased sideways at shot time t (0-1): the sum of its lane changes. */
export function driveSide(d, t) {
  const u = Math.max(0, Math.min(1, t));
  let side = 0;
  for (const [t0, t1, off] of d.lanes || []) side += off * ease((u - t0) / Math.max(1e-3, t1 - t0));
  return side;
}

export function applyDrive(obj, t, duration = 4) {
  const d = obj && obj.userData && obj.userData.drive;
  if (!d) return null;
  const u = Math.max(0, Math.min(1, t));
  const sec = u * duration;
  const along = d.speed * sec + 0.5 * (d.accel || 0) * sec * sec;
  let side = 0, slope = 0;
  for (const [t0, t1, off] of d.lanes || []) {
    const k = (u - t0) / Math.max(1e-3, t1 - t0);
    side += off * ease(k);
    if (k > 0 && k < 1) slope += off * 6 * k * (1 - k) / Math.max(1e-3, (t1 - t0) * duration * d.speed);
  }
  const f = [Math.sin(d.yaw), Math.cos(d.yaw)], r = [Math.cos(d.yaw), -Math.sin(d.yaw)];
  obj.position.x = d.start[0] + f[0] * along - r[0] * side;
  obj.position.z = d.start[1] + f[1] * along - r[1] * side;
  obj.rotation.y = d.yaw - Math.atan(slope);
  obj.updateMatrixWorld(true);
  return { along, side, dx: obj.position.x - d.start[0], dz: obj.position.z - d.start[1] };
}
