// Camera framing from a built shot: look height at 62% of the subject's height (the Previz app's rule), shot sizes,
// angles and simple moves as start / end camera states. Pure math on the built objects; deterministic.
import * as THREE from 'three';

const DEG = Math.PI / 180;
export const SHOT_HEIGHT = { extreme_close: 0.4, close: 0.8, medium: 1.25, full: null, wide: null };

// world box of obj, skipping subtrees marked userData.noFit (a v1.4 airborne kite 6 m up would shrink the person)
function worldBox(obj) {
  const box = new THREE.Box3(), b = new THREE.Box3();
  obj.updateWorldMatrix(true, false);
  (function walk(o) {
    if (o.userData.noFit) return;
    if (o.geometry && !o.children.length && !o.userData.noFit) { b.setFromObject(o, false); box.union(b); } // same rule as Box3.setFromObject
    else if (o.geometry) { const k = o.children; o.children = []; b.setFromObject(o, false); o.children = k; box.union(b); }
    for (const c of o.children) walk(c);
  })(obj);
  return box;
}

/** One move, as keys in subject space. A chain sticks each move onto the end of the one before it. */
function chainMoves(segments, rig) {
  const { at, az, el, dist, target, fov, fwd, right, endPos } = rig;
  const signOf = (params) => {
    const dir = params && params.direction;
    if (dir === 'left' || dir === 'right_to_left' || dir === 'counterclockwise') return -1;
    return 1;
  };
  const alone = (move, params) => {
    const speed = Math.max(0.2, Math.min(1.6, (params && params.speed) || 1));
    const sign = signOf(params);
    const look = target.toArray();
    const pair = (a, b) => [a, b];
    let keys;
    if (move === 'orbit' || move === 'swoop') {
      let sweep = ((params && params.angle) || (move === 'swoop' ? 50 : 90)) * DEG * sign;
      let a0 = az;
      // "starts from behind and orbits around to the front": the sentence's own start and end sides (0 front, 180 behind)
      const from = params && params.from_deg != null ? params.from_deg * DEG : null;
      const to = params && params.to_deg != null ? params.to_deg * DEG : null;
      if (move === 'orbit' && from != null && to != null) {
        const d = (((from - to) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) || 2 * Math.PI;
        a0 = from;
        sweep = sign > 0 ? d : d - 2 * Math.PI;
      } else if (move === 'orbit' && from != null) a0 = from;
      else if (move === 'orbit' && to != null) a0 = to + sweep;
      const steps = Math.max(1, Math.min(8, Math.round(Math.abs(sweep) / (45 * DEG))));
      keys = [];
      for (let i = 0; i <= steps; i++) {
        const a = a0 - sweep * (i / steps);
        const e = move === 'swoop' ? el - 8 * DEG * (i / steps) : el;
        const d = move === 'swoop' ? dist * (1 + 0.35 * (i / steps)) : dist;
        keys.push({ pos: at(a, e, d).toArray(), target: look.slice(), fov });
      }
    } else if (move === 'push_in') {
      keys = pair(
        { pos: at(az, el, dist * 1.45).toArray(), target: look.slice(), fov },
        { pos: at(az, el, dist).toArray(), target: look.slice(), fov },
      );
    } else if (move === 'pull_out') {
      keys = pair(
        { pos: at(az, el, dist).toArray(), target: look.slice(), fov },
        { pos: at(az, el + 4 * DEG, dist * 2.2).toArray(), target: look.slice(), fov: fov + 4 },
      );
    } else if (move === 'track' || move === 'dolly_along') {
      const side = Math.max(1.6, dist * 0.45);
      const from = endPos.clone().addScaledVector(right, -side);
      const to = endPos.clone().addScaledVector(right, side * 0.15);
      const aim0 = target.clone().addScaledVector(right, -side * 0.35);
      const aim1 = target.clone().addScaledVector(right, side * 0.05);
      keys = pair(
        { pos: from.toArray(), target: aim0.toArray(), fov },
        { pos: to.toArray(), target: aim1.toArray(), fov },
      );
    } else if (move === 'truck_left' || move === 'truck_right') {
      const side = 1.15 * (move === 'truck_left' ? -1 : 1) * sign;
      const from = endPos.clone().addScaledVector(right, -side);
      const to = endPos.clone().addScaledVector(right, side);
      keys = pair(
        { pos: from.toArray(), target: look.slice(), fov },
        { pos: to.toArray(), target: look.slice(), fov },
      );
    } else if (move === 'follow') {
      keys = pair(
        { pos: at(180 * DEG, 9 * DEG, dist * 1.15).toArray(), target: look.slice(), fov },
        { pos: at(172 * DEG, 8 * DEG, dist).toArray(), target: look.slice(), fov },
      );
    } else if (move === 'crane_up' || move === 'crane_reveal') {
      keys = pair(
        { pos: at(az, -4 * DEG, dist).toArray(), target: look.slice(), fov },
        { pos: at(az, 34 * DEG, dist * (move === 'crane_reveal' ? 2.2 : 1.2)).toArray(), target: look.slice(), fov },
      );
    } else if (move === 'crane_down') {
      keys = pair(
        { pos: at(az, 32 * DEG, dist * 1.3).toArray(), target: look.slice(), fov },
        { pos: at(az, 2 * DEG, dist).toArray(), target: look.slice(), fov },
      );
    } else if (move === 'boom_up' || move === 'boom_down') {
      const base = at(az, 0, dist);
      const up = move === 'boom_up' ? 3.2 : -1.4;
      keys = pair(
        { pos: base.toArray(), target: look.slice(), fov },
        { pos: [base.x, Math.max(0.4, base.y + up), base.z], target: look.slice(), fov },
      );
    } else if (move === 'rise') {
      const x = target.x, z = target.z;
      keys = pair(
        { pos: [x, target.y + 8, z], target: [x, target.y, z], fov: 42 },
        { pos: [x, target.y + 22, z], target: [x, target.y, z], fov: 42 },
      );
    } else if (move === 'tilt_up' || move === 'tilt_down') {
      const pos = at(az, el, dist).toArray();
      const lift = move === 'tilt_up' ? 2.8 : -1.2;
      keys = pair(
        { pos, target: look.slice(), fov },
        { pos: pos.slice(), target: [target.x, target.y + lift, target.z], fov },
      );
    } else if (move === 'pan' || move === 'pan_left' || move === 'pan_right') {
      const pos = endPos.toArray();
      const side = (move === 'pan_left' ? -1 : 1) * sign * 2.4;
      keys = pair(
        { pos, target: target.clone().addScaledVector(right, -side).toArray(), fov },
        { pos: pos.slice(), target: target.clone().addScaledVector(right, side).toArray(), fov },
      );
    } else if (move === 'aerial') {
      const back = fwd.clone().multiplyScalar(-6);
      const start = target.clone().add(back); start.y = 22;
      const end = start.clone().addScaledVector(fwd, 8);
      keys = pair(
        { pos: start.toArray(), target: look.slice(), fov: 38 },
        { pos: end.toArray(), target: look.slice(), fov: 38 },
      );
    } else if (move === 'body_mount') {
      const chest = target.clone();
      const pos = at(0, 0, 1.05, chest);
      pos.y = chest.y;
      const aim = chest.toArray();
      keys = pair({ pos: pos.toArray(), target: aim, fov: 28 }, { pos: pos.toArray(), target: aim.slice(), fov: 28 });
    } else if (move === 'dolly_zoom') {
      const f0 = (params && params.focal_from) || 70;
      const f1 = (params && params.focal_to) || 24;
      const base = at(az, el, dist);
      const horiz = Math.max(0.01, Math.hypot(base.x - target.x, base.z - target.z));
      const span = Math.max(8.5, horiz);
      const dx = (base.x - target.x) / horiz;
      const dz = (base.z - target.z) / horiz;
      const y = base.y;
      const span1 = Math.max(0.9, span * (f1 / f0));
      // focal tracks distance key by key, so the subject's size holds while the background stretches
      keys = [];
      for (let i = 0; i <= 12; i++) {
        const d = span + (span1 - span) * (i / 12);
        keys.push({ pos: [target.x + dx * d, y, target.z + dz * d], target: look.slice(), fov, focal_mm: f0 * (d / span) });
      }
    } else if (move === 'static') {
      const pos = at(az, el, dist).toArray();
      keys = pair({ pos, target: look.slice(), fov }, { pos: pos.slice(), target: look.slice(), fov });
    } else {
      keys = pair(
        { pos: at(az, el, dist * 1.35).toArray(), target: look.slice(), fov },
        { pos: at(az, el, dist).toArray(), target: look.slice(), fov },
      );
    }
    if (speed !== 1 && keys.length >= 2) {
      const a = keys[0];
      const b = keys[keys.length - 1];
      const blend = (u, v) => u.map((n, i) => n + (v[i] - n) * speed);
      keys[keys.length - 1] = { ...b, pos: blend(a.pos, b.pos), target: blend(a.target, b.target), fov: a.fov + (b.fov - a.fov) * speed };
      if (a.focal_mm && b.focal_mm) keys[keys.length - 1].focal_mm = a.focal_mm + (b.focal_mm - a.focal_mm) * speed;
    }
    return keys;
  };
  const out = [];
  let origin = null;
  for (const seg of segments) {
    const local = alone(seg.move, seg.params || {});
    const base = local[0];
    const placed = local.map((k) => {
      if (!origin) return { pos: k.pos.slice(), target: k.target.slice(), fov: k.fov, focal_mm: k.focal_mm };
      return {
        pos: origin.pos.map((v, i) => v + (k.pos[i] - base.pos[i])),
        target: origin.target.map((v, i) => v + (k.target[i] - base.target[i])),
        fov: k.fov,
        focal_mm: k.focal_mm,
      };
    });
    placed.forEach((k, i) => {
      if (origin && i === 0) return;
      const span = placed.length === 1 ? 1 : i / (placed.length - 1);
      const t = seg.from_t + (seg.to_t - seg.from_t) * span;
      out.push({ ...k, t });
    });
    origin = placed[placed.length - 1];
  }
  return out;
}

/**
 * frameShot(built, spec, { aspect = 16 / 9, fov = 40, obstacles }) -> { start: { pos, target, fov }, end: { pos, target, fov }, label }
 * obstacles: objects the camera must see past (walls, houses); by default the built kit's set and dressing.
 * built: result of buildShot (subject, actors, root). Positions are plain [x, y, z] arrays.
 */
export function frameShot(built, spec, options = {}) {
  const aspect = options.aspect || 16 / 9;
  if (built) built.sentence = spec.text || '';
  if (!options.obstacles && built.kit) options = Object.assign({}, options, { obstacles: defaultObstacles(built.kit, built) });
  const cam = spec.camera || {};
  const subj = built.subject;
  let fov = options.fov || 38;
  if (!subj) return still([12, 8, 14], [0, 1, 0], fov, 'empty');
  const sb = worldBox(subj.object);
  const seated = subj.object.userData.seated && subj.object.userData.seated.vehicle;
  const yaw = subj.yaw;
  const fwd = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  const right = new THREE.Vector3(fwd.z, 0, -fwd.x);
  // people: 62% of the pose's real head height (a sit, lie or crouch is shorter than a stand); things: box center
  const h = Math.max(0.2, sb.max.y - sb.min.y);
  let headY = sb.max.y;
  if (subj.kind === 'person') {
    const headBone = subj.object.getObjectByName('head');
    if (headBone) { const hv = new THREE.Vector3(); headBone.getWorldPosition(hv); headY = hv.y; }
  }
  const span = Math.max(0.2, headY - sb.min.y);
  const lookH = subj.kind === 'person' ? span : h;
  const target = new THREE.Vector3((sb.min.x + sb.max.x) / 2, sb.min.y + lookH * (subj.kind === 'person' ? 0.62 : 0.5), (sb.min.z + sb.max.z) / 2);
  if (cam.shot === 'close' || cam.shot === 'extreme_close') {
    const head = subj.kind === 'person' && subj.object.getObjectByName('head');
    if (head) { head.getWorldPosition(target); target.y += (cam.shot === 'close' ? 0.06 : 0.11) * (h / 1.75); } // head pivot is the neck: head and shoulders, whatever the pose
    else target.y = sb.max.y - h * 0.3;
  }

  // in a vehicle: through the windshield by default (the angle the library's car_interior kit proved reads best)
  if (seated && !cam.angle || cam.angle === 'through_windshield' && seated) {
    const veh = subj.vehicle;
    const cab = veh.userData.cabin || { zf: 1.2 };
    const head = new THREE.Vector3(); subj.object.getObjectByName('head').getWorldPosition(head);
    const vfwd = new THREE.Vector3(0, 0, 1).applyQuaternion(veh.getWorldQuaternion(new THREE.Quaternion())).setY(0).normalize();
    const vright = new THREE.Vector3(vfwd.z, 0, -vfwd.x);
    const tgt = head.clone().addScaledVector(new THREE.Vector3(0, 1, 0), -0.28);
    const vpos = veh.getWorldPosition(new THREE.Vector3());
    const seatName = (subj.object.userData.seated && subj.object.userData.seated.seat) || 'driver';
    const sp = (veh.userData.seats || {})[seatName];
    if (veh.userData.asset !== 'bus' && /^rear/.test(seatName) && sp) {
      // back seat: from between the front seats looking back (tinted side glass and the front seats hide a child from outside)
      veh.updateMatrixWorld(true);
      const local = veh.worldToLocal(head.clone());
      const pos = veh.localToWorld(new THREE.Vector3(local.x * 0.15, local.y + 0.12, local.z + 1.15));
      const t2 = head.clone().setY(head.y - 0.08);
      const end = { pos: pos.toArray(), target: t2.toArray(), fov: 58 };
      return { start: end, end, label: 'from the front seats, looking back' };
    }
    const front = vpos.clone().addScaledVector(vfwd, cab.zf + (veh.userData.asset === 'bus' ? 2.4 : 2.3));
    const pos = front.addScaledVector(vright, -0.15).setY(head.y + 0.6);
    const d = pos.distanceTo(tgt);
    const end = { pos: pos.toArray(), target: tgt.toArray(), fov: 32 };
    const start = { pos: tgt.clone().add(pos.clone().sub(tgt).setLength(d * (cam.move === 'pull_out' ? 0.7 : 2.2))).toArray(), target: tgt.toArray(), fov: 32 };
    if (cam.move === 'pull_out') return { start: end, end: start, label: 'through the windshield, pull out' };
    return { start: cam.move === 'static' ? end : start, end, label: 'through the windshield' + (cam.move === 'static' ? '' : ', push in') };
  }

  // how much to fit: the subject (close / medium / full) or the whole group (wide or several actors)
  const group = new THREE.Box3();
  for (const a of built.actors) group.union(worldBox(a.object));
  for (const p of built.props) if (fitsProp(p)) group.union(worldBox(p.object));
  const gsize = group.getSize(new THREE.Vector3());
  const shot = cam.shot || (built.actors.length > 1 && Math.max(gsize.x, gsize.z) > 4.5 ? 'wide' : built.actors.length > 1 ? 'group' : 'full');
  let az = 32 * DEG, el = 7 * DEG;
  let lockAz = false;
  let interiorSit = false;
  switch (cam.angle) {
    case 'low': el = -6 * DEG; break;
    case 'high': el = 32 * DEG; break;
    case 'overhead': el = 78 * DEG; az = 5 * DEG; break;
    case 'behind': az = 160 * DEG; el = 10 * DEG; break;
    case 'side': az = 90 * DEG; break;
    case 'front': az = 4 * DEG; break;
    case 'over_shoulder': az = 155 * DEG; el = 9 * DEG; break;
    default: break;
  }
  // v1.4 face-to-face pair (hug, kiss, shake_hands, fight): a two-shot from the side, not over one shoulder
  const sa = (spec.actors || []).find((a) => a.id === cam.subject);
  const pr = sa && (sa.pair || ((sa.relations || []).find((r) => r.type === 'pair') || {}).pair);
  if (!cam.angle && pr && Math.abs(Math.abs(pr.facing) - 180) < 1) { az = 74 * DEG; el = 8 * DEG; }
  // A walker with no "away" or "toward" faces across the frame or at a three-quarter, never straight away.
  const gaitPose = sa && sa.pose;
  if (gaitPose === 'walk' || gaitPose === 'jog' || gaitPose === 'run') {
    const sideFixed = ['behind', 'over_shoulder', 'side', 'front', 'pov'].includes(cam.angle);
    if (!sideFixed) {
      const said = String(spec.text || '');
      const facing = cam.facing;
      const away = facing === 'away' || /\b(?:walks?|walking|jogs?|jogging|runs?|running)\s+away\b|\baway\s+from\s+(?:the\s+)?(?:camera|lens)\b/i.test(said);
      const toward = facing === 'toward' || /\b(?:toward|towards)\s+(?:the\s+)?(?:camera|lens|us)\b|\b(?:walks?|walking|jogs?|jogging|runs?|running)\s+(?:toward|towards)\b/i.test(said);
      const across = facing === 'across' || /\bacross\b/i.test(said);
      const screenRight = facing === 'left_to_right' || /\bleft to right\b/i.test(said);
      if (away) az = 172 * DEG;
      else if (toward) az = (subj.kind === 'animal' ? 38 : 8) * DEG; // a four-legged runner reads at a three-quarter, not head-on
      else if (screenRight) az = 90 * DEG;
      else if (across) az = 90 * DEG;
      else if (!cam.angle) az = ((spec.seed >>> 0) % 2 === 0) ? 90 * DEG : 48 * DEG;
    }
    lockAz = true;
  }
  // a low hero angle on a person: the lens near the ground looking up at the chest, not level at the hips
  if (cam.angle === 'low' && subj.kind === 'person' && !seated) { el = -13 * DEG; target.y = Math.max(target.y, headY * 0.9); }
  const dirOf = (a, e) => fwd.clone().multiplyScalar(Math.cos(a)).addScaledVector(right, Math.sin(a)).multiplyScalar(Math.cos(e)).setY(Math.sin(e));
  let distFor;
  let holdFit = false;
  if (SHOT_HEIGHT[shot]) {
    const frameH = SHOT_HEIGHT[shot] * (h / 1.75 > 0.5 ? h / 1.75 : 1) * (subj.kind === 'person' ? 1 : 1.6);
    if (shot === 'medium') target.y = sb.max.y - frameH * 0.4; // waist up with a little headroom
    const d0 = (frameH / 2) / Math.tan(fov * DEG / 2);
    distFor = () => d0;
  } else {
    // full / group / wide: fit the box of the actors and the props they use (bench, boat, bike, leash dog) in frame
    if (shot === 'wide') fov = 40;
    if (cam.angle === 'overhead') fov = 40;
    if (shot !== 'full') { const gc = group.getCenter(new THREE.Vector3()); target.x = gc.x; target.z = gc.z; target.y = Math.min(target.y, group.min.y + gsize.y * 0.55); }
    const fit = shot === 'full' ? sb.clone().union(relatedBox(built, subj)) : group.clone();
    // The kite is not part of the fit. Framing it made the kid a speck.
    holdFit = false;
    const margin = cam.angle === 'overhead' ? 4 : shot === 'wide' ? 0.72 : shot === 'full' ? 1.15 : 0.84;
    distFor = (a) => Math.max(shot === 'full' ? 2.2 : 2.6, fitDistance(fit, target, dirOf(a, cam.angle === 'overhead' ? 80 * DEG : el), fov, aspect, margin));
  }
  let dist = distFor(az);
  if (/\bextreme wide\b/i.test(spec.text || '')) dist *= 2.3;
  const kite = (built.props || []).find((p) => p.object && (p.object.userData.asset === 'kite' || String(p.object.userData.libraryId || '') === 'kite'));
  let kiteShot = false;
  let kiteStart = 0;
  interiorSit = !!(subj.kind === 'person' && spec.place && spec.place.id === 'interior' && (subj.object.userData.seated && !subj.object.userData.seated.vehicle || subj.object.userData.pose === 'sit'));
  if (subj.kind === 'person' && !SHOT_HEIGHT[shot] && !kite) {
    const half = Math.tan(fov * DEG / 2);
    const need = Math.max(0, headY + 0.16 - target.y) / (half * 0.78);
    if (need > dist) dist = need;
    const sitting = span < 1.45 || subj.object.userData.seated || subj.object.userData.pose === 'sit';
    if (sitting && interiorSit && !cam.angle && !/\bwhip pan\b/i.test(spec.text || '')) {
      return frameSeatedInterior({ built, subj, sb, headY, yaw, fwd, right, target, aspect });
    } else if (sitting && !interiorSit) {
      const tight = Math.tan(7 * DEG);
      const seated = Math.max(4.2, (headY + 0.22 - target.y) / (tight * 0.75));
      if (seated > dist) dist = seated;
      if (!cam.angle) az = 72 * DEG;
    }
    const minDist = dist;
    const baseDist = distFor;
    distFor = (a) => Math.max(minDist, baseDist(a));
  }
  if (kite && subj.kind === 'person' && !SHOT_HEIGHT[shot]) {
    fov = 22;
    const half = Math.tan(fov * DEG / 2);
    const body = Math.max(0.9, headY - sb.min.y);
    const kb = new THREE.Box3().setFromObject(kite.object);
    // The kid stays near half the frame. The kite sits in the headroom above.
    const kc = kb.getCenter(new THREE.Vector3());
    const to = kc.clone().sub(target); to.y = 0;
    let lean = 14 * DEG;
    if (to.lengthSq() > 0.04) {
      const side = Math.atan2(to.dot(right), to.dot(fwd));
      lean = side > 0 ? 14 * DEG : -14 * DEG;
    }
    az = Math.PI + lean;
    // Feet near the bottom, the kite near the top, the kid about half the frame.
    const frameBot = sb.min.y - body * 0.06;
    const frameTop = kb.max.y + body * 0.08;
    const frameH = Math.max(body / 0.48, frameTop - frameBot);
    target.y = frameBot + frameH / 2;
    let kdist = (frameH / 2) / half;
    kdist = Math.max(2.3, kdist);
    dist = kdist;
    distFor = () => kdist;
    lockAz = true;
    el = 0;
    kiteStart = kdist * 1.04;
    kiteShot = true;
  }
  const at = (a, e, d, tgt = target) => {
    const dir = dirOf(a, e);
    const p = tgt.clone().addScaledVector(dir, d);
    if (p.y < 0.25) p.y = 0.25;
    return p;
  };
  if (built.shotCam && !cam.angle) return clearLens({ ...built.shotCam }, built, aspect);
  if (built.stagedPair && !cam.angle && (!cam.move || cam.move === 'static' || cam.move === 'push_in')) {
    return framePair(built, cam, group, aspect);
  }
  if (cam.angle === 'pov' && subj.kind === 'person' && /\b(pov|point of view|first[- ]person)\b/i.test(spec.text || '') && !/\bfpv\b/i.test(spec.text || '')) {
    return truePov(built, subj, aspect);
  }
  if ((cam.angle === 'pov' || cam.move === 'fpv') && subj.kind === 'person') {
    // FPV follow: low, just behind and above the shoulder, drifting forward with a small handheld wobble
    const headObj = subj.object.getObjectByName('head');
    const head = new THREE.Vector3();
    if (headObj) headObj.getWorldPosition(head);
    else head.set((sb.min.x + sb.max.x) / 2, sb.max.y - h * 0.1, (sb.min.z + sb.max.z) / 2);
    const ride = subj.object.parent && subj.object.parent.userData && subj.object.parent.userData.cruise;
    const travelYaw = ride ? ride.yaw : yaw;
    const travel = new THREE.Vector3(Math.sin(travelYaw), 0, Math.cos(travelYaw));
    const shoulder = head.clone();
    shoulder.y = head.y - 0.22;
    const keys = [];
    for (let i = 0; i < 5; i++) {
      const t = i / 4;
      const wobble = Math.sin(t * Math.PI * 2) * 0.08;
      const bob = Math.sin(t * Math.PI * 3) * 0.03;
      const back = 3 - t * 1.5;
      const side = new THREE.Vector3(travel.z, 0, -travel.x);
      const pos = shoulder.clone().addScaledVector(travel, -back).addScaledVector(side, wobble);
      pos.y = Math.max(0.55, shoulder.y + 0.16 + bob);
      const look = head.clone().addScaledVector(travel, 0.15).addScaledVector(side, wobble * 0.25);
      look.y = head.y - 0.55;
      keys.push({ pos: pos.toArray(), target: look.toArray(), fov: 62 });
    }
    const headTop = head.clone();
    headTop.y += 0.16;
    const foot = new THREE.Vector3();
    const ankle = subj.object.getObjectByName('ankle_L') || subj.object.getObjectByName('ankle_R');
    if (ankle) ankle.getWorldPosition(foot);
    else foot.set(head.x, sb.min.y + 0.04, head.z);
    const probe = new THREE.PerspectiveCamera(56, aspect, 0.05, 200);
    for (const k of keys) {
      for (let n = 0; n < 8; n++) {
        probe.fov = k.fov;
        probe.updateProjectionMatrix();
        probe.position.fromArray(k.pos);
        probe.up.set(0, 1, 0);
        probe.lookAt(k.target[0], k.target[1], k.target[2]);
        probe.updateMatrixWorld(true);
        const hn = headTop.clone().project(probe);
        const fn = foot.clone().project(probe);
        if (hn.y > 0.92) k.target[1] += 0.045;
        else if (fn.y < -0.96) k.target[1] -= 0.04;
        else break;
      }
    }
    return clearLens({ start: keys[0], end: keys[4], keys, label: 'fpv follow' }, built, aspect);
  }
  // keep walls, houses and buildings out of the lens: try other sides when the default view is blocked
  if (!cam.angle && options.obstacles) {
    const best = lockAz
      ? holdAzimuth(target, az, el, distFor, options.obstacles, dirOf, built, fov, aspect)
      : clearAzimuth(target, az, el, distFor, options.obstacles, dirOf, built, fov, aspect);
    az = best.az; dist = best.dist;
  }
  if (!kiteShot && (shot === 'full' || !shot) && cam.angle !== 'overhead') dist *= 1.38;
  if (cam.angle === 'over_shoulder') {
    const headObj = subj.object && subj.object.getObjectByName && subj.object.getObjectByName('head');
    const head = new THREE.Vector3();
    if (headObj) headObj.getWorldPosition(head);
    else head.set(target.x, headY, target.z);
    const face = new THREE.Vector3(0, 0, 1);
    if (headObj) face.applyQuaternion(headObj.getWorldQuaternion(new THREE.Quaternion()));
    else face.copy(fwd);
    face.y = 0;
    if (face.lengthSq() < 0.01) face.copy(fwd);
    face.normalize();
    // far enough back that the shoulder and head hold one side of the frame, the street ahead reads past them
    const shoulder = head.clone().addScaledVector(face, -1.45).addScaledVector(right, 0.72);
    shoulder.y = head.y + 0.04;
    const past = head.clone().addScaledVector(face, 7).addScaledVector(right, -1.3);
    past.y = head.y - 0.35;
    const closer = shoulder.clone().addScaledVector(face, 0.12);
    return clearLens({
      start: { pos: shoulder.toArray(), target: past.toArray(), fov: 40 },
      end: { pos: closer.toArray(), target: past.toArray(), fov: 40 },
      label: 'over the shoulder',
    }, built, aspect);
  }
  if (cam.angle === 'low' && /\bbehind\b/i.test(spec.text || '')) az = 168 * DEG;
  if (cam.move === 'body_mount') {
    const headObj = subj.object && subj.object.getObjectByName && subj.object.getObjectByName('head');
    const head = new THREE.Vector3();
    if (headObj) headObj.getWorldPosition(head);
    else head.set(target.x, headY, target.z);
    const face = new THREE.Vector3(0, 0, 1);
    if (headObj) face.applyQuaternion(headObj.getWorldQuaternion(new THREE.Quaternion()));
    else face.copy(fwd);
    face.y = 0;
    if (face.lengthSq() < 0.01) face.copy(fwd);
    face.normalize();
    const pos = head.clone().addScaledVector(face, 1.15);
    pos.y = head.y;
    const aim = [head.x, head.y, head.z];
    const state = { pos: pos.toArray(), target: aim, fov: 32 };
    return clearLens({ start: state, end: { pos: state.pos.slice(), target: aim.slice(), fov: 32 }, label: (shot || 'wide') + ', body_mount' }, built, aspect);
  }
  if (subj.kind !== 'person' && dist > 14) dist = 11;
  if (holdFit) dist = Math.max(dist, distFor(az));
  let endPos = at(az, el, dist);
  if (interiorSit && built.kit) endPos = insideRoom(endPos, target, built.kit);
  const end = { pos: endPos.toArray(), target: target.toArray(), fov };
  const wideDist = Math.max(2.6, fitDistance(group.clone(), target, dirOf(az, 12 * DEG), 40, aspect, 0.72));
  const fromDoorway = (framed) => {
    if (cam.from !== 'doorway' || !spec.place || spec.place.id !== 'interior') return framed;
    const ks = built.kit && built.kit.cameras && built.kit.cameras.start;
    if (!ks || !ks.pos || !ks.target) return framed;
    return { ...framed, start: { pos: ks.pos.slice(), target: ks.target.slice(), fov: ks.fov || framed.start.fov }, label: framed.label + ', from the doorway' };
  };
  const ship = (framed) => {
    const out = fromDoorway(clearEnds(framed, options.obstacles));
    if (interiorSit && built.kit) {
      cutaway(out.end.pos, out.end.target, built);
      cutaway(out.start.pos, out.start.target, built);
    }
    return clearLens(out, built, aspect);
  };
  if (/\bcar chase\b/i.test(spec.text || '') || (subj.object.userData.seated && subj.object.userData.seated.vehicle && subj.object.parent && (cam.angle === 'low' || /\bbehind\b/i.test(spec.text || '')))) {
    const car = (subj.object.parent && subj.object.parent !== built.root && subj.object.parent.userData && (subj.object.parent.userData.asset || subj.object.parent.userData.seats)) ? subj.object.parent : subj.object;
    car.updateMatrixWorld(true);
    const p = new THREE.Vector3();
    car.getWorldPosition(p);
    const back = new THREE.Vector3(0, 0, -1).applyQuaternion(car.getWorldQuaternion(new THREE.Quaternion()));
    back.y = 0;
    if (back.lengthSq() < 0.01) back.set(0, 0, -1);
    back.normalize();
    // low behind: the car's rear about 40% of the frame width, the road and the traffic ahead over its roof
    const look = p.clone().addScaledVector(back, -9);
    look.y = 0.85;
    const start = p.clone().addScaledVector(back, 8.6);
    start.y = 0.62;
    const end = p.clone().addScaledVector(back, 6.6);
    end.y = 0.55;
    return clearLens({
      start: { pos: start.toArray(), target: look.toArray(), fov: 34 },
      end: { pos: end.toArray(), target: look.toArray(), fov: 34 },
      label: 'low, behind',
    }, built, aspect);
  }
  const streetCrowd = !cam.angle
    && (gaitPose === 'walk' || gaitPose === 'jog' || gaitPose === 'run')
    && (built.actors || []).filter((a) => a.kind === 'person').length >= 8;
  const crowdCam = !cam.move || cam.move === 'push_in' || cam.move === 'follow' || cam.move === 'track';
  if (streetCrowd && crowdCam && cam.angle !== 'overhead') return ship(sidewalkCrowd(built, cam.facing || (/\b(toward|towards)\b/i.test(spec.text || '') ? 'toward' : /\baway\b/i.test(spec.text || '') ? 'away' : null)));
  if (kiteShot) {
    const endPos = at(az, el, dist);
    const startPos = at(az, el, Math.max(dist, kiteStart));
    return ship({ start: { pos: startPos.toArray(), target: target.toArray(), fov }, end: { pos: endPos.toArray(), target: target.toArray(), fov }, label: 'full, kite' });
  }
  const segs = Array.isArray(cam.segments) ? cam.segments.filter((s) => s && s.move) : [];
  const freshMove = ['crane_down', 'boom_up', 'boom_down', 'tilt_up', 'tilt_down', 'pan_left', 'pan_right', 'truck_left', 'truck_right', 'rise', 'dolly_zoom', 'body_mount', 'aerial', 'swoop'].includes(cam.move);
  const orbitAngle = segs[0] && segs[0].params && segs[0].params.angle;
  if ((cam.move === 'orbit' || (segs[0] && segs[0].move === 'orbit')) && (shot === 'close' || shot === 'medium' || shot === 'extreme_close')) target.y = headY;
  if ((cam.move === 'pan_right' || cam.move === 'pan_left' || cam.move === 'pan') && built.balcony) {
    // From across the street, a crane height up: the pan starts on the skyline down the avenue and turns onto her.
    const b = built.balcony;
    const w = new THREE.Vector3().fromArray(b.pos);
    const n = new THREE.Vector3().fromArray(b.normal);
    const al = new THREE.Vector3().fromArray(b.along);
    const sign = cam.move === 'pan_left' ? -1 : 1;
    const pos = w.clone().addScaledVector(n, 15).addScaledVector(al, -4 * sign);
    pos.y = 4.2;
    const aimEnd = w.clone(); aimEnd.y = w.y + 0.95;
    const toEnd = aimEnd.clone().sub(pos);
    const dEnd = toEnd.length();
    const yawEnd = Math.atan2(toEnd.x, toEnd.z);
    const pitchEnd = Math.asin(toEnd.y / dEnd);
    const yawStart = yawEnd + sign * 62 * DEG;
    const keys = [];
    for (let i = 0; i <= 6; i++) {
      const u = i / 6;
      const yawU = yawStart + (yawEnd - yawStart) * u;
      const pitchU = 8 * DEG + (pitchEnd - 8 * DEG) * u;
      const d = 60 + (dEnd - 60) * u;
      const aim = pos.clone().add(new THREE.Vector3(Math.sin(yawU) * Math.cos(pitchU), Math.sin(pitchU), Math.cos(yawU) * Math.cos(pitchU)).multiplyScalar(d));
      const focal = 30 + (70 - 30) * u * u;
      keys.push({ t: u, pos: pos.toArray(), target: aim.toArray(), fov, focal_mm: focal });
    }
    return clearLens({ start: keys[0], end: keys[6], keys, label: 'wide, pan right to the balcony' }, built, aspect);
  }
  const orbitSided = segs.some((s) => s.move === 'orbit' && s.params && (s.params.from_deg != null || s.params.to_deg != null));
  if (segs.length > 1 || freshMove || (cam.move === 'orbit' && orbitAngle >= 90) || orbitSided) {
    const chained = chainMoves(segs.length ? segs : [{ move: cam.move, from_t: 0, to_t: 1, params: cam.segments && cam.segments[0] && cam.segments[0].params || {}, tracking: 0 }], { at, az, el, dist, target, fov, fwd, right, endPos });
    if (chained.length >= 2) {
      let keys = chained;
      if (cam.move === 'orbit' || segs.some((s) => s.move === 'orbit')) keys = roomClamp(keys, built);
      if (cam.move === 'dolly_zoom' || segs.some((s) => s.move === 'dolly_zoom')) keys = holdDolly(keys, built, aspect);
      if (cam.move === 'push_in' || segs.some((s) => s.move === 'push_in')) keys = spreadPush(keys);
      const label = (shot ? shot + ', ' : '') + (segs.length ? segs.map((s) => s.move).filter(Boolean).join(' then ') : cam.move);
      return clearLens({ start: keys[0], end: keys[keys.length - 1], keys, label }, built, aspect);
    }
  }
  const crowdN = (spec.actors || []).reduce((n, a) => n + ((a.kind === 'person' || a.cat === 'person') ? (a.count || 1) : 0), 0);
  if (cam.angle === 'overhead' && crowdN >= 8) {
    const gc = group.getCenter(new THREE.Vector3());
    // straight down, low enough that each walker reads; a slow settle gives the frame life
    // centred on where the crowd is halfway through its walk
    let mx = 0, mz = 0, nw = 0;
    for (const a of built.actors || []) {
      const g = a.object && a.object.userData.gait;
      if (!g || g.animal) continue;
      const half = (g.speed || 0) * ((built.spec && built.spec.duration_s) || 4) * 0.5;
      mx += Math.sin(g.yaw) * half; mz += Math.cos(g.yaw) * half; nw++;
    }
    if (nw) { gc.x += mx / nw; gc.z += mz / nw; }
    const look = [gc.x, 0, gc.z];
    const along = Math.max(gsize.x, gsize.z);
    const h = Math.max(8, Math.min(14, along * 0.9 + 3));
    return clearLens({
      start: { pos: [gc.x, h + 1.2, gc.z + 0.3], target: look, fov: 50 },
      end: { pos: [gc.x, h, gc.z + 0.3], target: look, fov: 50 },
      label: 'overhead',
    }, built, aspect);
  }
  if (/\bwhip pan\b/i.test(spec.text || '')) {
    const door = kitLook(built, /shot_door/) || new THREE.Vector3(-4.1, 1.15, 0.1);
    const pos = new THREE.Vector3(door.x + 2.2, 1.35, door.z);
    // land with her whole head in: aim between chest and head, a little wider than the door
    const land = target.clone();
    const head = subj.object.getObjectByName('head');
    if (head) land.y = Math.max(land.y, head.getWorldPosition(new THREE.Vector3()).y - 0.3);
    const keys = [
      { t: 0, pos: pos.toArray(), target: [door.x, 1.15, door.z], fov: 28 },
      { t: 0.1, pos: pos.toArray(), target: land.toArray(), fov: 36 },
      { t: 1, pos: pos.toArray(), target: land.toArray(), fov: 36 },
    ];
    return clearLens({ start: keys[0], end: keys[2], keys, label: (shot || 'full') + ', pan' }, built, aspect);
  }
  if (cam.move === 'pan_right' && /\bbalcony\b/i.test(spec.text || '')) {
    const pos = endPos.toArray();
    const away = target.clone().addScaledVector(right, -14);
    const keys = [
      { t: 0, pos, target: away.toArray(), fov },
      { t: 1, pos: pos.slice(), target: target.toArray(), fov },
    ];
    return clearLens({ start: keys[0], end: keys[1], keys, label: (shot || 'full') + ', pan right' }, built, aspect);
  }
  if (shot === 'extreme_close' && /\b(laptop|hands|typing|insert)\b/i.test(spec.text || '')) {
    const lap = (built.props || []).find((p) => p.object && /laptop/i.test(String(p.object.userData.asset || p.object.userData.libraryId || p.object.name || '')));
    const aim = new THREE.Vector3();
    if (lap && lap.object) lap.object.getWorldPosition(aim);
    else aim.set(target.x, headY - 0.42, target.z);
    const chest = new THREE.Vector3();
    subj.object.getWorldPosition(chest);
    const toward = chest.clone().sub(aim);
    toward.y = 0;
    if (toward.lengthSq() < 0.01) toward.copy(fwd);
    toward.normalize();
    // three-quarter from the typist's side, above the hands: the forearms and both hands on the keys, not the back of a body
    const side = new THREE.Vector3(toward.z, 0, -toward.x);
    const keys = lap && lap.object && lap.object.getObjectByName('attach_keys');
    if (keys) keys.getWorldPosition(aim);
    aim.addScaledVector(toward, 0.08);
    const dirCam = toward.clone().multiplyScalar(0.55).addScaledVector(side, 0.84).normalize();
    const cam = aim.clone().addScaledVector(dirCam, 0.46);
    cam.y = aim.y + 0.3;
    const far = aim.clone().addScaledVector(dirCam, 0.62);
    far.y = aim.y + 0.36;
    return clearLens({
      start: { pos: far.toArray(), target: aim.toArray(), fov: 32 },
      end: { pos: cam.toArray(), target: aim.toArray(), fov: 32 },
      label: 'extreme close, insert',
    }, built, aspect);
  }
  if (/\bface\b/i.test(spec.text || '') && (shot === 'close' || /\bclose up\b/i.test(spec.text || ''))) {
    const aim = [target.x, headY, target.z];
    const pos = at(6 * DEG, 0, 0.78);
    pos.y = headY;
    const closer = pos.clone().addScaledVector(dirOf(6 * DEG, 0), -0.06);
    return clearLens({
      start: { pos: pos.toArray(), target: aim, fov: 36 },
      end: { pos: closer.toArray(), target: aim, fov: 36 },
      label: 'close, face',
    }, built, aspect);
  }
  let start;
  switch (cam.move) {
    case 'kit': { // the kit card's own move (data/kits/<id>.json cameras.start / end, label = layout.move), framed on the set, not the subject
      const kc = built.kit && built.kit.cameras;
      if (kc && kc.start && kc.end) return { start: { pos: kc.start.pos.slice(), target: kc.start.target.slice(), fov: kc.start.fov || fov }, end: { pos: kc.end.pos.slice(), target: kc.end.target.slice(), fov: kc.end.fov || fov }, label: (built.layout && built.layout.move) || 'kit move' };
      break;
    }
    case 'dolly_along': { // low lateral dolly past the subject (Lake card: low dolly along the shoreline)
      const low = at(az, el, dist); low.y = Math.max(0.45, Math.min(low.y, target.y * 0.6 + 0.2));
      const side = Math.max(2.4, dist * 0.55);
      start = { pos: low.clone().addScaledVector(right, -side).toArray(), target: target.clone().addScaledVector(right, -side * 0.35).toArray(), fov };
      return ship({ start, end: { pos: low.clone().addScaledVector(right, side).toArray(), target: target.clone().addScaledVector(right, side * 0.35).toArray(), fov }, label: shot + ', low dolly along' });
    }
    case 'crane_reveal': // crane up and back from near the ground to a high wide view (River card: crane up to reveal the river)
      start = { pos: at(az, -2 * DEG, dist * 0.9).toArray(), target: target.toArray(), fov };
      return ship({ start, end: { pos: at(az, 38 * DEG, dist * 2.4).toArray(), target: target.clone().addScaledVector(fwd, -dist * 0.6).setY(Math.max(0, target.y - 0.6)).toArray(), fov: fov + 6 }, label: shot + ', crane up to reveal' });
    case 'pull_out': {
      const closeShot = cam.shot === 'close' || cam.shot === 'extreme_close';
      const far = closeShot ? wideDist : dist * 2.2;
      const farFov = closeShot ? 40 : fov;
      return ship({ start: end, end: { pos: at(az, closeShot ? 8 * DEG : el + 4 * DEG, far).toArray(), target: target.toArray(), fov: farFov }, label: shot + (closeShot ? ', close to wide' : ', pull out') });
    }
    case 'orbit': {
      const framed = clearEnds({ start: { pos: at(az - 70 * DEG, el, dist).toArray(), target: end.target, fov }, end, label: shot + ', orbit' }, options.obstacles);
      return fromDoorway({ ...framed, mid: { pos: at(az - 35 * DEG, el, dist).toArray(), target: end.target, fov } });
    }
    case 'pan': { const t0 = target.clone().addScaledVector(right, -2.5); start = { pos: endPos.toArray(), target: t0.toArray(), fov }; return ship({ start, end, label: shot + ', pan' }); }
    case 'track': {
      const radius = Math.max(4.6, dist);
      const y = Math.max(1.35, target.y);
      // left to right: the camera rides on the subject's left, so the travel reads screen right
      const sideOf = cam.facing === 'right_to_left' ? 1 : cam.facing === 'left_to_right' ? -1 : 1;
      const drive = subj.object.userData.drive;
      if (drive && drive.follow) {
        // Run 3 (c07): a camera car beside a driving car keeps its place; the drive carries both, so nothing turns and
        // the road and the far background stream the same way (builder.js CAMERA_CAR_MOVES)
        const beside = target.clone().addScaledVector(right, radius * sideOf);
        beside.y = y;
        return ship({ start: { pos: beside.toArray(), target: target.toArray(), fov }, end: { pos: beside.toArray(), target: target.toArray(), fov }, label: shot + ', camera car' });
      }
      const from = target.clone().addScaledVector(right, radius * sideOf).addScaledVector(fwd, -1.1);
      const to = target.clone().addScaledVector(right, radius * sideOf).addScaledVector(fwd, 1.1);
      from.y = y;
      to.y = y;
      return ship({ start: { pos: from.toArray(), target: target.toArray(), fov }, end: { pos: to.toArray(), target: target.toArray(), fov }, label: shot + ', track' });
    }
    case 'follow': start = { pos: at(180 * DEG, 9 * DEG, dist * 1.2).toArray(), target: end.target, fov }; return ship({ start, end: { pos: at(170 * DEG, 9 * DEG, dist).toArray(), target: end.target, fov }, label: 'follow from behind' });
    case 'crane_up': start = { pos: at(az, -4 * DEG, dist).toArray(), target: end.target, fov }; return ship({ start, end: { pos: at(az, 34 * DEG, dist * 1.2).toArray(), target: end.target, fov }, label: shot + ', crane up' });
    case 'static': return ship(still(end.pos, end.target, fov, shot));
    default: break;
  }
  let farD = clearDist(target, dirOf(az, el), dist, dist * 1.55, options.obstacles);
  let nearD = dist;
  if (farD < dist + 0.25) { farD = dist; nearD = Math.max(1.15, dist * 0.62); }
  if (cam.angle === 'low' && subj.kind === 'person') farD = Math.min(farD, dist * 1.12); // the push stays close enough to keep looking up
  if (subj.kind === 'animal' && h < 1.3) { farD = Math.min(Math.max(farD, 3.6), 4.6); nearD = Math.min(Math.max(nearD, 2.6), farD - 0.8); } // a dog-sized animal fills a low frame
  else if (subj.kind !== 'person' || seated) { farD = Math.max(farD, 9); nearD = Math.max(5, Math.min(nearD, farD - 2)); }
  start = { pos: at(az, el, farD).toArray(), target: end.target, fov };
  return ship({ start, end: { pos: at(az, el, nearD).toArray(), target: end.target, fov }, label: shot + (cam.move === 'push_in' ? ', push in' : ', slow push in') });
}
const _ray = new THREE.Raycaster();

/** The rider's head over the heel edge (m), turned this far from the line of travel toward the toes (deg), and the
 * head-cam's lens: at 100 degrees two shoes were a tenth of the frame wide each. */
export const POV_HEEL = 0.15;
export const POV_TURN = 32;
export const POV_FOV_RIDE = 78;

/** A true point of view: the lens at the subject's eyes, looking down the line of travel at their own hands, feet and board. Head and neck hidden. */
function truePov(built, subj, aspect) {
  const obj = subj.object;
  const J = (n) => obj.getObjectByName(n);
  const head = J('head');
  const hp = new THREE.Vector3();
  if (head) head.getWorldPosition(hp); else obj.getWorldPosition(hp).setY(1.6);
  const ride = obj.parent && obj.parent.userData && obj.parent.userData.cruise;
  const yaw = ride ? ride.yaw : subj.yaw;
  const f = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  // the chest faces across the board (a sideways stance): the toes point that way
  const chest = J('chest');
  const face = new THREE.Vector3(0, 0, 1);
  if (chest) face.applyQuaternion(chest.getWorldQuaternion(new THREE.Quaternion()));
  face.y = 0;
  if (face.lengthSq() < 0.01) face.set(f.z, 0, -f.x);
  face.normalize();
  // the eyes of a rider bombing a hill: crouched, weight forward, head over the front foot about 1.3 m above it
  // (standing riders and walkers keep their own eye height)
  const ankles = ['L', 'R'].map((s) => J('ankle_' + s)).filter(Boolean).map((j) => j.getWorldPosition(new THREE.Vector3()));
  const frontFoot = ankles.sort((u, v) => v.dot(f) - u.dot(f))[0];
  // A rider's head sits over the heel edge, turned part way down the board (a sideways stance looks over the front
  // shoulder): the two shoes then lie across the frame. Straight down the board's centre line they stacked into one
  // strip with the deck (step 16).
  const eye = ride && frontFoot
    ? frontFoot.clone().addScaledVector(f, 0.05).addScaledVector(face, -POV_HEEL).setY(frontFoot.y + 1.3)
    : hp.clone().add(new THREE.Vector3(0, 0.05, 0)).addScaledVector(f, 0.09);
  const gaze = ride ? f.clone().multiplyScalar(Math.cos(POV_TURN * DEG)).addScaledVector(face, Math.sin(POV_TURN * DEG)).normalize() : f;
  // a rider's head-cam tips 67 degrees down: the road rushing up from the top of frame, the board and shoes
  // below. On foot the eyes look ahead, a little down.
  const pitch = (ride ? 67 : 12) * DEG;
  const look = eye.clone().addScaledVector(gaze, 3 * Math.cos(pitch)).add(new THREE.Vector3(0, -3 * Math.sin(pitch), 0));
  // From up here a mannequin's shoulder, thighs and knees read as blobs across the lens. What the rider sees is the
  // road coming at them and their own two shoes planted across the deck, the front one turned a little down the hill:
  // draw just that (the shoes ride on the board), the body itself is under and behind the lens.
  const board = ride ? obj.parent : null;
  const deck = board && board.getObjectByName('attach_feet');
  const shoe = J('ankle_L');
  if (board && deck && shoe) {
    board.updateMatrixWorld(true);
    const bq = board.getWorldQuaternion(new THREE.Quaternion()).invert();
    const toLocal = (v) => v.clone().applyQuaternion(bq);
    const up = toLocal(new THREE.Vector3(0, 1, 0));
    const along = toLocal(f);
    for (const [ahead, turn] of [[0.17, 28], [-0.17, 0]]) {
      const toe = toLocal(face.clone().multiplyScalar(Math.cos(turn * DEG)).addScaledVector(f, Math.sin(turn * DEG)).normalize());
      const side = new THREE.Vector3().crossVectors(up, toe).normalize();
      const copy = shoe.clone(true);
      copy.name = 'pov_shoe';
      copy.scale.set(1, 1, 1);
      copy.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(side, up, toe));
      // the sole is 8 cm under the ankle and the foot runs 6 cm back to 20 cm forward of it: centre it on the deck
      copy.position.copy(deck.position).addScaledVector(along, ahead).addScaledVector(up, 0.08).addScaledVector(toe, -0.07);
      copy.traverse((n) => { if (n.isMesh) n.visible = true; });
      board.add(copy);
      if (built.undo) built.undo.push(() => { board.remove(copy); });
    }
  }
  obj.traverse((n) => {
    if (!n.isMesh || !n.visible) return;
    n.visible = false;
    if (built.undo) built.undo.push(() => { n.visible = true; });
  });
  const wob = (i) => [Math.sin(i * 1.7) * 0.015, Math.sin(i * 2.3) * 0.01];
  const keys = [];
  for (let i = 0; i <= 4; i++) {
    const [a, b] = wob(i);
    const r = new THREE.Vector3(f.z, 0, -f.x);
    keys.push({ t: i / 4, pos: eye.clone().addScaledVector(r, a).add(new THREE.Vector3(0, b, 0)).toArray(), target: look.clone().addScaledVector(r, a * 2).toArray(), fov: ride ? POV_FOV_RIDE : 100 });
  }
  void aspect;
  return { start: keys[0], end: keys[4], keys, label: 'point of view' };
}

/** A staged indoor pair (builder stagedPair): side-on from the room's own camera side, both bodies in, nothing in front. */
function framePair(built, cam, group, aspect) {
  const sp = built.stagedPair;
  const c = new THREE.Vector3().fromArray(sp.center);
  const view = new THREE.Vector3().fromArray(sp.view);
  const medium = cam.shot === 'medium' || cam.shot === 'close';
  const fov = medium ? 30 : 38;
  const target = new THREE.Vector3(c.x, sp.seated ? 0.95 : medium ? 1.3 : 1.0, c.z);
  const back = view.clone().negate();
  const dir = back.clone().multiplyScalar(Math.cos(6 * DEG)).setY(Math.sin(6 * DEG)).normalize();
  const fit = group.clone();
  if (medium) fit.min.y = Math.max(fit.min.y, sp.seated ? 0.55 : 0.85);
  const dist = Math.max(2.4, fitDistance(fit, target, dir, fov, aspect, medium ? 0.86 : 0.8));
  const end = target.clone().addScaledVector(dir, dist);
  const slowPush = !cam.move && sp.seated;
  const far = target.clone().addScaledVector(dir, dist * (slowPush ? 1.12 : 1.25));
  const meshes = [];
  for (const root of [built.kit && built.kit.dressing, built.kit && built.kit.set]) {
    if (!root) continue;
    root.traverse((n) => { if (n.isMesh && n.visible && !/floor|ground|rug/i.test(n.name || '')) meshes.push(n); });
  }
  const own = new Set((built.actors || []).map((a) => a.object).concat(sp.keep || []));
  const aims = [];
  for (const a of built.actors || []) {
    for (const bone of ['head', 'chest', 'hips', 'knee_L', 'knee_R']) {
      const j = a.object.getObjectByName(bone);
      if (j) aims.push(j.getWorldPosition(new THREE.Vector3()));
    }
  }
  for (const from of [end, far]) {
    for (const aim of aims) {
      const d = aim.clone().sub(from);
      const len = d.length();
      _ray.set(from, d.multiplyScalar(1 / len)); _ray.near = 0.05; _ray.far = len - 0.25;
      for (const hit of _ray.intersectObjects(meshes, false)) {
        if (isActorNode(hit.object, own)) continue;
        let n = hit.object;
        while (n.parent && n.parent !== built.kit.dressing && n.parent !== built.kit.set) n = n.parent;
        if (n.visible && !n.userData.cutaway) {
          n.visible = false;
          n.userData.cutaway = true;
          if (built.undo) built.undo.push(() => { n.visible = true; n.userData.cutaway = false; });
        }
      }
    }
  }
  // dressing between the lens and the pair, or filling the near corners
  const probe = new THREE.PerspectiveCamera(fov, aspect, 0.05, 200);
  probe.position.copy(end); probe.lookAt(target); probe.updateMatrixWorld(true);
  if (built.kit && built.kit.dressing) for (const ch of built.kit.dressing.children) {
    if (!ch.visible || ch.isInstancedMesh || own.has(ch)) continue;
    const bx = new THREE.Box3().setFromObject(ch);
    if (bx.isEmpty()) continue;
    const cc = bx.getCenter(new THREE.Vector3());
    const along = cc.clone().sub(end).dot(target.clone().sub(end).normalize());
    const n = cc.clone().project(probe);
    if (along > 0 && along < dist - 1.0 && Math.abs(n.x) < 1.25 && Math.abs(n.y) < 1.25) {
      ch.visible = false;
      if (built.undo) built.undo.push(() => { ch.visible = true; });
    }
  }
  const s = cam.move === 'push_in' || slowPush ? far : end;
  return {
    start: { pos: s.toArray(), target: target.toArray(), fov },
    end: { pos: end.toArray(), target: target.toArray(), fov },
    label: (cam.shot || 'wide') + ', two-shot' + (cam.move === 'push_in' ? ', push in' : slowPush ? ', slow push in' : ''),
  };
}
const FLAT =/ground|floor|rug|path|road|water|dune|shore|swash|wave|dash|mountain/i;

/** A street crowd is filmed from the roadway side, behind the pack, so a tree or a wall never sits on the lens. */
function sidewalkCrowd(built, facing) {
  const sub0 = built.subject && built.subject.object;
  const origin = new THREE.Vector3();
  if (sub0) sub0.getWorldPosition(origin);
  const sub = sub0;
  const yaw = sub ? sub.rotation.y : Math.PI;
  const fwd = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  const rear = fwd.clone().multiplyScalar(-1);
  const right = new THREE.Vector3(fwd.z, 0, -fwd.x);
  const along = facing === 'toward' ? fwd : rear;
  const cam = origin.clone().addScaledVector(along, 5.4).addScaledVector(right, 5.0);
  cam.y = 1.85;
  const look = origin.clone().addScaledVector(fwd, 1.8);
  look.y = 0.95;
  const end = { pos: cam.toArray(), target: look.toArray(), fov: 56 };
  const startPos = cam.clone().addScaledVector(rear, 0.35);
  return { start: { pos: startPos.toArray(), target: look.toArray(), fov: 56 }, end, label: 'full, street' };
}

function spreadPush(keys) {
  if (!keys || keys.length < 2) return keys;
  const a = keys[0].pos;
  const b = keys[keys.length - 1].pos;
  if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > 0.25) return keys;
  const end = keys[keys.length - 1];
  const pos = end.pos.map((v, i) => v + ((end.target[i] || 0) - v) * 0.34);
  const copy = keys.slice();
  copy[copy.length - 1] = { ...end, pos };
  return copy;
}

export function roomClamp(keys, built) {
  if (!built || !built.kit || !built.kit.set) return keys;
  let walls = false;
  const room = new THREE.Box3();
  built.kit.set.traverse((n) => { if (n.isMesh && /wall_/.test(n.name || '')) { walls = true; room.union(new THREE.Box3().setFromObject(n)); } });
  if (!walls) return keys;
  return keys.map((k) => {
    const aim = k.target || k.look;
    if (!k.pos || !aim) return k;
    // only a room the subject is inside (a tunnel mouth far down the road is not a room)
    if (aim[0] < room.min.x || aim[0] > room.max.x || aim[2] < room.min.z || aim[2] > room.max.z) return k;
    const aimV = new THREE.Vector3().fromArray(aim);
    const orig = new THREE.Vector3().fromArray(k.pos);
    let placed = insideRoom(orig.clone(), aimV, built.kit);
    const flat = placed.clone().sub(aimV);
    flat.y = 0;
    if (flat.length() < 2.15) {
      const keep = orig.clone().sub(aimV);
      keep.y = 0;
      if (keep.length() < 0.2) keep.set(0, 0, 2.15);
      keep.setLength(Math.max(2.15, Math.min(keep.length(), 4.2)));
      placed = aimV.clone().add(keep);
      placed.y = Math.max(0.8, orig.y);
      built.kit.set.traverse((n) => {
        if (!n.visible || !n.isMesh || !/wall_/.test(n.name || '')) return;
        const box = new THREE.Box3().setFromObject(n);
        if (box.isEmpty() || !box.expandByScalar(0.2).containsPoint(placed)) return;
        n.visible = false;
        if (built.undo) built.undo.push(() => { n.visible = true; });
      });
    }
    return { ...k, pos: [placed.x, Math.max(0.45, placed.y), placed.z] };
  });
}

function holdDolly(keys, built, aspect) {
  const sub = built && built.subject && built.subject.object;
  if (!sub || !keys || keys.length < 2 || keys.length > 2) return keys;
  const head = sub.getObjectByName('head');
  const foot = sub.getObjectByName('ankle_L') || sub.getObjectByName('ankle_R');
  if (!head || !foot) return keys;
  sub.updateMatrixWorld(true);
  const hp = new THREE.Vector3();
  const fp = new THREE.Vector3();
  head.getWorldPosition(hp);
  foot.getWorldPosition(fp);
  const height = (camPos, focal) => {
    const filmH = 36 / Math.max(aspect || 1.76, 1);
    const fov = 2 * Math.atan((filmH / 2) / Math.max(8, focal || 35)) * 180 / Math.PI;
    const cam = new THREE.PerspectiveCamera(fov, aspect || 1.76, 0.05, 400);
    const look = new THREE.Vector3().fromArray(keys[0].target || keys[0].look);
    cam.position.copy(camPos);
    cam.lookAt(look);
    cam.updateMatrixWorld(true);
    const a = hp.clone().project(cam);
    const b = fp.clone().project(cam);
    return Math.hypot((a.x - b.x) * (aspect || 1.76), a.y - b.y) / 2;
  };
  const start = new THREE.Vector3().fromArray(keys[0].pos);
  const end = new THREE.Vector3().fromArray(keys[keys.length - 1].pos);
  const look = new THREE.Vector3().fromArray(keys[0].target || keys[0].look);
  const h0 = height(start, keys[0].focal_mm || 70);
  if (!(h0 > 0)) return keys;
  let lo = 0.12;
  let hi = 1.15;
  let best = end.clone();
  for (let i = 0; i < 10; i++) {
    const m = (lo + hi) / 2;
    const pos = look.clone().lerp(start, m);
    pos.y = start.y;
    const h = height(pos, keys[keys.length - 1].focal_mm || 24);
    best = pos;
    if (h > h0) lo = m;
    else hi = m;
  }
  const copy = keys.slice();
  copy[copy.length - 1] = { ...keys[keys.length - 1], pos: [best.x, start.y, best.z] };
  return copy;
}

function cutHorizon(framed, built) {
  if (!built || built._horizonCut || !/desert/i.test(built.sentence || '')) return;
  built._horizonCut = true;
  const cam = new THREE.Vector3().fromArray(framed.start.pos);
  const look = new THREE.Vector3().fromArray(framed.start.target).sub(cam);
  look.y = 0;
  if (look.lengthSq() < 1e-6) return;
  look.normalize();
  const left = new THREE.Vector3(-look.z, 0, look.x);
  const root = built.root;
  if (!root) return;
  const names = [];
  for (const side of [0.55, 0.85, 1.15]) {
    for (const drop of [-0.02, -0.08, -0.16]) {
      const dir = look.clone().multiplyScalar(0.35).addScaledVector(left, side).setY(drop).normalize();
      _ray.set(cam, dir);
      _ray.near = 0.3;
      _ray.far = 800;
      const hits = _ray.intersectObject(root, true).filter((h) => h.object && h.object.visible !== false);
      for (const h of hits.slice(0, 3)) {
        const name = String(h.object.name || (h.object.userData && h.object.userData.asset) || 'mesh');
        if (!names.includes(name)) names.push(name);
        if (/sand_relief|geo_dune|wedge|rim|horizon/.test(name)) {
          h.object.visible = false;
          if (built.undo) built.undo.push(() => { h.object.visible = true; });
        }
      }
    }
  }
  built.horizonHit = names;
}

function kitLook(built, pattern) {
  const roots = [built && built.kit && built.kit.set, built && built.kit && built.kit.dressing, built && built.root].filter(Boolean);
  if (!roots.length) return null;
  let found = null;
  for (const root of roots) root.traverse((n) => {
    if (found || !n.visible) return;
    const id = String(n.name || '') + ' ' + String((n.userData && (n.userData.asset || n.userData.libraryId)) || '');
    if (!pattern.test(id)) return;
    found = new THREE.Vector3();
    n.getWorldPosition(found);
  });
  return found;
}

function followTravel(built) {
  const obj = built && built.subject && built.subject.object;
  if (!obj) return new THREE.Vector3();
  const ride = obj.parent && obj.parent.userData && obj.parent.userData.cruise;
  const g = obj.userData && obj.userData.gait;
  const src = ride || (g && g.follow !== false ? g : null);
  if (!src) return new THREE.Vector3();
  const dur = (built.spec && built.spec.duration_s) || 4;
  const dist = (src.speed || 0) * dur;
  const yaw = src.yaw != null ? src.yaw : obj.rotation.y || 0;
  return new THREE.Vector3(Math.sin(yaw) * dist, 0, Math.cos(yaw) * dist);
}

function lensVehicles(built) {
  const dressing = built && built.kit && built.kit.dressing;
  if (!dressing) return [];
  const out = [];
  for (const c of dressing.children) {
    if (!c.visible) continue;
    const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
    if (/^(car|suv|truck|van|bus)$/.test(id)) out.push(c);
  }
  return out;
}

/** Hide a vehicle that enters the lens or fills the foreground along the follow. Walls stay. */
function clearLens(framed, built, aspect) {
  if (!framed || !framed.start || !framed.end || !built) return framed;
  cutHorizon(framed, built);
  const travel = followTravel(built);
  const own = built.subject && (built.subject.vehicle || built.subject.object);
  const vehicles = lensVehicles(built).filter((v) => v !== own && (!own || v !== own.parent));
  if (!vehicles.length) return framed;
  const fov = framed.end.fov || framed.start.fov || 40;
  const bad = new Set();
  for (const v of vehicles) {
    const box = new THREE.Box3().setFromObject(v);
    if (box.isEmpty()) continue;
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() * 0.5;
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      const pos = new THREE.Vector3().fromArray(framed.start.pos).lerp(new THREE.Vector3().fromArray(framed.end.pos), t).addScaledVector(travel, t);
      const lookPt = new THREE.Vector3().fromArray(framed.start.target).lerp(new THREE.Vector3().fromArray(framed.end.target), t).addScaledVector(travel, t);
      const look = lookPt.clone().sub(pos);
      const dist = look.length() || 1;
      const dir = look.clone().multiplyScalar(1 / dist);
      const closest = box.clampPoint(pos, new THREE.Vector3());
      if (pos.distanceTo(closest) < 1.5 || box.containsPoint(pos)) { bad.add(v); break; }
      const along = center.clone().sub(pos).dot(dir);
      if (along < 0.3 || along > dist - 0.4) continue;
      const halfH = Math.tan(fov * DEG / 2) * Math.max(0.4, along);
      const halfW = halfH * (aspect || 1.76);
      const cover = (Math.PI * radius * radius) / Math.max(0.2, 4 * halfW * halfH);
      if (cover > 0.15) { bad.add(v); break; }
    }
  }
  for (const v of bad) hideBlocker(v, null, built);
  hideNearTrees(framed, built, travel);
  if (framed.label && /fpv/i.test(framed.label)) {
    clearNearDressing(framed, built);
    // a flat path across the ride reads as a pale band at FPV height
    if (built.kit && built.kit.set) built.kit.set.traverse((n) => {
      if (!n.isMesh || !n.visible || !/path|track|sidewalk|paving/i.test(n.name || '')) return;
      n.visible = false;
      if (built.undo) built.undo.push(() => { n.visible = true; });
    });
  }
  return framed;
}

/** Props whose box comes within 1.5 m of the lens. The rider and the board stay. */
function clearNearDressing(framed, built) {
  const dressing = built && built.kit && built.kit.dressing;
  if (!dressing) return;
  const cam = new THREE.Vector3().fromArray(framed.start.pos);
  const m4 = new THREE.Matrix4();
  const ip = new THREE.Vector3();
  const iq = new THREE.Quaternion();
  const is = new THREE.Vector3();
  dressing.traverse((o) => {
    if (!o.isInstancedMesh) return;
    const id = String((o.parent && (o.parent.userData.libraryId || o.parent.userData.asset || o.parent.name)) || '');
    if (!/bench|tree|table|chair/.test(id)) return;
    for (let n = 0; n < o.count; n++) {
      o.getMatrixAt(n, m4);
      new THREE.Matrix4().multiplyMatrices(o.matrixWorld, m4).decompose(ip, iq, is);
      if (Math.hypot(cam.x - ip.x, cam.z - ip.z) < 6) hideBlocker(o, n, built);
    }
  });
  for (const c of dressing.children) {
    if (!c.visible || c.isInstancedMesh) continue;
    const id = String((c.userData && (c.userData.asset || c.userData.libraryId)) || '');
    if (/person|skate|bicycle/.test(id)) continue;
    const box = new THREE.Box3().setFromObject(c);
    if (box.isEmpty()) continue;
    const closest = box.clampPoint(cam, new THREE.Vector3());
    if (cam.distanceTo(closest) < 4) hideBlocker(c, null, built);
  }
}

/** A sidewalk tree whose canopy reaches the lens is a prop in the shot. Drop that instance. */
function hideNearTrees(framed, built, travel) {
  if (!built || !built.layout || built.layout.id !== 'city') return;
  const dressing = built && built.kit && built.kit.dressing;
  if (!dressing) return;
  dressing.updateMatrixWorld(true);
  const pos = new THREE.Vector3();
  const m4 = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  dressing.traverse((o) => {
    if (!o.isInstancedMesh) return;
    const id = String((o.parent && (o.parent.userData.libraryId || o.parent.name)) || '');
    if (!/tree/.test(id)) return;
    for (let n = 0; n < o.count; n++) {
      o.getMatrixAt(n, m4);
      const world = new THREE.Matrix4().multiplyMatrices(o.matrixWorld, m4);
      world.decompose(p, q, s);
      let close = false;
      for (let i = 0; i <= 8; i++) {
        const t = i / 8;
        pos.fromArray(framed.start.pos).lerp(new THREE.Vector3().fromArray(framed.end.pos), t).addScaledVector(travel, t);
        if (Math.hypot(pos.x - p.x, pos.z - p.z) < 8) { close = true; break; }
      }
      if (close) hideBlocker(o, n, built);
    }
  });
}
/** Everything in the set except flat ground. Houses, trees, animals and dressing all block a view. */
function defaultObstacles(kit, built) {
  const out = [];
  const take = (o) => {
    const id = String(o.userData.asset || o.userData.libraryId || o.name || '');
    if (FLAT.test(o.name) || FLAT.test(id)) return;
    out.push(o);
  };
  kit.set.children.forEach(take);
  kit.dressing.children.forEach(take);
  const subject = built && built.subject && built.subject.object;
  for (const a of (built && built.actors) || []) if (a.object && a.object !== subject) out.push(a.object);
  return out;
}
/** A camera is inside an obstacle when the surface is within this many metres of it, not when a tree sits in the middle of the view. */
const INSIDE = 1.1;
function nearestHit(target, pos, obstacles) {
  const tgt = new THREE.Vector3(target[0], target[1], target[2]);
  const p = new THREE.Vector3(pos[0], pos[1], pos[2]);
  const dir = p.clone().sub(tgt);
  const dist = dir.length();
  if (dist < 0.4 || !obstacles || !obstacles.length) return null;
  dir.multiplyScalar(1 / dist);
  _ray.set(tgt, dir); _ray.near = 0.3; _ray.far = dist + 0.2;
  const hit = _ray.intersectObjects(obstacles, true).find((h) => h.object.visible !== false && !/ground|floor|rug|path|road|water|dash|wave|shore|swash|dune|leash/i.test(h.object.name));
  return hit ? { dist, hit: hit.distance, dir, tgt } : null;
}
/** Pull a camera that sits inside an obstacle back to the near side of it. */
function pullOut(state, obstacles) {
  const n = nearestHit(state.target, state.pos, obstacles);
  if (!n || n.dist - n.hit > INSIDE) return state.pos;
  const safe = Math.max(1.2, n.hit - 0.6);
  if (safe >= n.dist - 0.05) return state.pos;
  const p = n.tgt.clone().addScaledVector(n.dir, safe);
  if (p.y < 0.3) p.y = 0.3;
  return [p.x, p.y, p.z];
}
function pointBlocked(target, pos, obstacles) {
  const n = nearestHit(target, pos, obstacles);
  return !!n && n.dist - n.hit <= INSIDE;
}
/** Start and end stay outside obstacles. If the straight move between them clips, step the path sideways. */
function clearEnds(framed, obstacles) {
  if (!framed || !obstacles || !obstacles.length) return framed;
  const place = (state) => ({ ...state, pos: pullOut(state, obstacles) });
  let start = place(framed.start), end = place(framed.end);
  const blocked = (a, b) => {
    if (pointBlocked(a.target, a.pos, obstacles) || pointBlocked(b.target, b.pos, obstacles)) return true;
    for (let i = 1; i <= 3; i++) {
      const t = i / 4;
      const pos = a.pos.map((v, k) => v * (1 - t) + b.pos[k] * t);
      const target = a.target.map((v, k) => v * (1 - t) + b.target[k] * t);
      if (pointBlocked(target, pos, obstacles)) return true;
    }
    return false;
  };
  const note = (a, b, path) => {
    const startIn = pointBlocked(a.target, a.pos, obstacles);
    const endIn = pointBlocked(b.target, b.pos, obstacles);
    if ((startIn || endIn || path) && typeof window !== 'undefined') {
      (window.__previzClips || (window.__previzClips = [])).push({ label: framed.label, startIn, endIn, path: !!path, start: a.pos.map((v) => +v.toFixed(2)), end: b.pos.map((v) => +v.toFixed(2)) });
    }
    return { ...framed, start: a, end: b };
  };
  if (!blocked(start, end)) return note(start, end, false);
  const along = new THREE.Vector3(end.pos[0] - start.pos[0], 0, end.pos[2] - start.pos[2]);
  const lateral = along.length() > 0.4 ? new THREE.Vector3(-along.z, 0, along.x).normalize() : new THREE.Vector3(1, 0, 0);
  for (let step = 1; step <= 4; step++) {
    const off = lateral.clone().multiplyScalar((step % 2 ? 1 : -1) * Math.ceil(step / 2) * 1.1);
    const shift = (state) => place({ ...state, pos: [state.pos[0] + off.x, state.pos[1], state.pos[2] + off.z] });
    const s = shift(start), e = shift(end);
    if (!blocked(s, e)) return note(s, e, false);
  }
  return note(start, end, true);
}
/** Longest distance up to want (never below min) with a clear line from the target: keeps move starts out of trees and walls. */
function clearDist(target, dir, min, want, obstacles) {
  if (!obstacles || want <= min) return want;
  _ray.set(target, dir); _ray.near = 0.3; _ray.far = want + 0.6;
  const hit = _ray.intersectObjects(obstacles, true).find((h) => h.object.visible !== false && !/ground|floor|rug|path|road|water|dash|wave|shore|swash|dune|leash/i.test(h.object.name));
  return hit ? Math.max(min, hit.distance - 0.6) : want;
}
function aimPoints(built, target) {
  const pts = [target.clone()];
  const sub = built && built.subject && built.subject.object;
  if (!sub) return pts;
  const put = (name) => { const n = sub.getObjectByName(name); if (!n) return; const p = new THREE.Vector3(); n.getWorldPosition(p); pts.push(p); };
  put('head'); put('chest'); put('hips'); put('ankle_L'); put('ankle_R');
  const box = new THREE.Box3();
  for (const a of built.actors || []) if (a.object) box.union(new THREE.Box3().setFromObject(a.object));
  if (!box.isEmpty()) {
    const y = box.max.y;
    pts.push(new THREE.Vector3(box.min.x, y, box.min.z), new THREE.Vector3(box.max.x, y, box.min.z), new THREE.Vector3(box.min.x, y, box.max.z), new THREE.Vector3(box.max.x, y, box.max.z));
  }
  return pts;
}
function isActorNode(obj, actors) {
  let n = obj;
  while (n) { if (actors.has(n)) return true; n = n.parent; }
  return false;
}
function rayHit(from, to, obstacles, actors) {
  const dir = to.clone().sub(from);
  const dist = dir.length();
  if (dist < 0.3 || !obstacles || !obstacles.length) return null;
  dir.multiplyScalar(1 / dist);
  _ray.set(from, dir); _ray.near = 0.05; _ray.far = dist - 0.05;
  const hit = _ray.intersectObjects(obstacles, true).find((h) => h.object.visible !== false && !FLAT.test(h.object.name) && !isActorNode(h.object, actors));
  return hit || null;
}
/** A near obstacle covers too much of the frame, or sits in the centre of the view. */
function coverageBlock(camPos, target, obstacles, fov, aspect, actors) {
  const look = target.clone().sub(camPos);
  const dist = look.length();
  if (dist < 0.4) return null;
  look.multiplyScalar(1 / dist);
  const up = new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(look, up);
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();
  const camUp = new THREE.Vector3().crossVectors(right, look).normalize();
  let worst = null;
  for (const o of obstacles) {
    if (!o.visible || o.isInstancedMesh || isActorNode(o, actors)) continue;
    const box = new THREE.Box3().setFromObject(o);
    if (box.isEmpty()) continue;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() * 0.5;
    const to = c.clone().sub(camPos);
    const along = to.dot(look);
    if (along < 0.25 || along > dist + 2) continue;
    const perp = to.addScaledVector(look, -along);
    const halfH = Math.tan(fov * DEG / 2) * Math.max(0.4, along);
    const halfW = halfH * aspect;
    const cover = (Math.PI * r * r) / Math.max(0.2, 4 * halfW * halfH);
    const mid = along > dist * 0.2 && along < dist * 0.8;
    const centre = Math.abs(perp.dot(right)) < halfW * 0.6 && Math.abs(perp.dot(camUp)) < halfH * 0.6;
    if (cover > 0.15 || (mid && centre && along < dist - 0.3)) {
      if (!worst || cover > worst.cover) worst = { cover, object: o };
    }
  }
  return worst;
}
function hideBlocker(obj, instanceId, built) {
  if (!built || !built.undo || !obj) return;
  if (obj.isInstancedMesh && instanceId != null) {
    const saved = new THREE.Matrix4();
    obj.getMatrixAt(instanceId, saved);
    obj.setMatrixAt(instanceId, new THREE.Matrix4().makeTranslation(0, -400, 0));
    obj.instanceMatrix.needsUpdate = true;
    const id = instanceId;
    built.undo.push(() => { obj.setMatrixAt(id, saved); obj.instanceMatrix.needsUpdate = true; });
    return;
  }
  let n = obj;
  while (n.parent && n.parent.name !== 'items' && n.parent.name !== 'set') n = n.parent;
  const id = String((n.userData && (n.userData.asset || n.userData.libraryId)) || n.name || '');
  if (/bench|sofa|bed|table|chair|counter/.test(id)) return;
  if (!n.visible || n.userData.cutaway) return;
  n.userData.cutaway = true;
  n.visible = false;
  built.undo.push(() => { n.visible = true; n.userData.cutaway = false; });
}
const SHELL = /wall|ceiling|sill|skirting|lintel|doorframe|door_frame/i;
/** Keep this azimuth. Hide whatever sits in the lens instead of swinging around behind the subject. */
function holdAzimuth(target, az, el, distFor, obstacles, dirOf, built, fov, aspect) {
  const dist = distFor(az);
  const cam = target.clone().addScaledVector(dirOf(az, el), dist);
  const actors = new Set((built && built.actors || []).map((a) => a.object).filter(Boolean));
  const seen = new Set();
  for (const p of aimPoints(built, target)) {
    const hit = rayHit(p, cam, obstacles, actors);
    if (hit && hit.distance < dist - 0.3 && !seen.has(hit.object)) {
      seen.add(hit.object);
      hideBlocker(hit.object, hit.instanceId, built);
    }
  }
  const cover = coverageBlock(cam, target, obstacles, fov, aspect, actors);
  if (cover && !seen.has(cover.object)) hideBlocker(cover.object, null, built);
  return { az, dist };
}
/** Pull a point back toward the subject until it sits inside the room walls. */
function insideRoom(pos, target, kit) {
  const box = new THREE.Box3();
  let any = false;
  kit.set.traverse((n) => {
    if (n.isMesh && /wall_/.test(n.name)) { box.union(new THREE.Box3().setFromObject(n)); any = true; }
  });
  if (!any || box.isEmpty()) return pos;
  const minX = box.min.x + 0.55, maxX = box.max.x - 0.55, minZ = box.min.z + 0.55, maxZ = box.max.z - 0.45;
  const inside = (p) => p.x >= minX && p.x <= maxX && p.z >= minZ && p.z <= maxZ;
  if (inside(pos)) return pos;
  let lo = 0.05, hi = 1, best = target.clone().lerp(pos, 0.2);
  for (let i = 0; i < 14; i++) {
    const m = (lo + hi) / 2;
    const p = target.clone().lerp(pos, m);
    if (inside(p)) { best = p; lo = m; } else hi = m;
  }
  best.y = Math.min(2.05, Math.max(0.9, pos.y));
  return best;
}
/**
 * Seated interior. The kit's doorway camera sits outside the side wall, and that sill covers the bottom of the frame.
 * Stay inside, on a three-quarter front, far enough that an 85 mm lens still holds the head. Hide any shell in the way.
 */
function frameSeatedInterior({ built, sb, headY, fwd, right, aspect }) {
  // A sit turns the chest toward the table while the root yaw stays put. Aim from the face.
  const headObj = built.subject && built.subject.object && built.subject.object.getObjectByName('head');
  if (headObj) {
    const face = new THREE.Vector3(0, 0, 1).applyQuaternion(headObj.getWorldQuaternion(new THREE.Quaternion()));
    face.y = 0;
    if (face.lengthSq() > 0.05) {
      fwd = face.normalize();
      right = new THREE.Vector3(fwd.z, 0, -fwd.x);
    }
  }
  const span = Math.max(0.4, headY - sb.min.y);
  const fov = 30;
  const bounds = roomBounds(built.kit);
  let hx = (sb.min.x + sb.max.x) / 2;
  let hz = (sb.min.z + sb.max.z) / 2;
  if (headObj) {
    const e = headObj.matrixWorld.elements;
    hx = e[12];
    hz = e[14];
  }
  const headPos = new THREE.Vector3(hx, headY, hz);
  // 38° off the face: chest and one side of the face, not a profile and not straight on.
  const dist = 4.55;
  const ang = 38 * DEG;
  let sign = 1;
  let best = -Infinity;
  for (const s of [1, -1]) {
    const raw = headPos.clone().addScaledVector(fwd, Math.cos(ang) * dist).addScaledVector(right, Math.sin(ang) * s * dist);
    let margin = 2;
    if (bounds) margin = Math.min(raw.x - bounds.minX, bounds.maxX - raw.x, raw.z - bounds.minZ, bounds.maxZ - raw.z);
    if (margin > best) { best = margin; sign = s; }
  }
  const lookY = sb.min.y + span * 0.40;
  const place = (ahead, side, up) => {
    const p = headPos.clone().addScaledVector(fwd, ahead).addScaledVector(right, side);
    p.y = Math.max(1.15, headY + up);
    if (bounds) {
      p.x = Math.min(bounds.maxX, Math.max(bounds.minX, p.x));
      p.z = Math.min(bounds.maxZ, Math.max(bounds.minZ, p.z));
    }
    return p;
  };
  const end = place(Math.cos(ang) * dist, Math.sin(ang) * sign * dist, -0.18);
  const start = place(Math.cos(ang) * (dist + 0.85), Math.sin(ang) * sign * (dist + 0.85), -0.02);
  const look = new THREE.Vector3(hx, lookY, hz);
  const head = new THREE.Vector3(hx, headY + 0.08, hz);
  const lap = headPos.clone().setY(sb.min.y + span * 0.28).addScaledVector(fwd, 0.12);
  if (built.subject && built.subject.object) {
    built.subject.object.userData.seatFrame = {
      wideLook: lookY,
      headTop: headY + 0.16,
      tableY: sb.min.y + span * 0.55,
    };
  }
  const framed = {
    start: { pos: start.toArray(), target: look.toArray(), fov },
    end: { pos: end.toArray(), target: look.toArray(), fov },
    label: 'three-quarter, inside',
  };
  for (let i = 0; i <= 4; i++) {
    const p = start.clone().lerp(end, i / 4).toArray();
    cutaway(p, head.toArray(), built);
    cutaway(p, lap.toArray(), built);
    hideShell(p, head.toArray(), built, /window|doorway|door_frame|doorframe/i);
    hideShell(p, lap.toArray(), built, /window|doorway|door_frame|doorframe/i);
  }
  backWall(built, end, headPos);
  return framed;
}
/**
 * A set built with three walls is open on its fourth side. When the camera looks across the subject toward that side,
 * a long lens sees past the room to the horizon. Close it with a wall a step behind the room's edge (removed on
 * teardown). Nothing changes when a wall is already there.
 */
function backWall(built, camPos, subjectPos) {
  const set = built && built.kit && built.kit.set;
  if (!set) return;
  const dir = subjectPos.clone().sub(camPos).setY(0);
  if (dir.lengthSq() < 1e-4) return;
  dir.normalize();
  const walls = [];
  let ref = null;
  set.traverse((n) => { if (n.isMesh && n.visible && /wall_/.test(n.name || '')) { walls.push(n); if (!ref) ref = n; } });
  if (!ref) return;
  const from = subjectPos.clone().setY(1.4);
  _ray.set(from, dir); _ray.near = 0; _ray.far = 9;
  if (_ray.intersectObjects(walls, false).length) return;
  const room = new THREE.Box3();
  for (const w of walls) room.union(new THREE.Box3().setFromObject(w));
  // the wall stands where the ray leaves the room's footprint, square to whichever side that is
  const xSide = Math.abs(dir.x) > Math.abs(dir.z);
  const edge = xSide ? (dir.x > 0 ? room.max.x : room.min.x) : (dir.z > 0 ? room.max.z : room.min.z);
  const span = xSide ? room.max.z - room.min.z : room.max.x - room.min.x;
  const height = Math.max(2.6, room.max.y - room.min.y);
  const geo = new THREE.BoxGeometry(xSide ? 0.2 : span + 0.4, height, xSide ? span + 0.4 : 0.2);
  const wall = new THREE.Mesh(geo, ref.material);
  wall.name = 'set_back_wall';
  wall.receiveShadow = true;
  wall.position.set(xSide ? edge + Math.sign(dir.x) * 0.1 : (room.min.x + room.max.x) / 2, room.min.y + height / 2, xSide ? (room.min.z + room.max.z) / 2 : edge + Math.sign(dir.z) * 0.1);
  const parent = built.root || set;
  parent.add(wall);
  if (built.undo) built.undo.push(() => { parent.remove(wall); geo.dispose(); });
}
function roomBounds(kit) {
  const box = new THREE.Box3();
  let any = false;
  if (kit && kit.set) kit.set.traverse((n) => {
    if (n.isMesh && /wall_/.test(n.name || '')) { box.union(new THREE.Box3().setFromObject(n)); any = true; }
  });
  if (!any || box.isEmpty()) return null;
  return { minX: box.min.x + 0.7, maxX: box.max.x - 0.7, minZ: box.min.z + 0.7, maxZ: box.max.z - 0.55 };
}
function clampRoom(pos, look, bounds) {
  const y = Math.min(1.65, Math.max(1.15, pos.y));
  if (!bounds) { const p = pos.clone(); p.y = y; return p; }
  const inside = (p) => p.x >= bounds.minX && p.x <= bounds.maxX && p.z >= bounds.minZ && p.z <= bounds.maxZ;
  const goal = pos.clone();
  goal.y = y;
  if (inside(goal)) return goal;
  let lo = 0.05, hi = 1, best = look.clone().lerp(goal, 0.15);
  for (let i = 0; i < 16; i++) {
    const m = (lo + hi) / 2;
    const q = look.clone().lerp(goal, m);
    q.y = y;
    if (inside(q)) { best = q; lo = m; } else hi = m;
  }
  best.y = y;
  return best;
}
function furnitureMeshes(built) {
  const out = [];
  const roots = built && built.kit ? [built.kit.dressing, built.kit.set] : [];
  for (const root of roots) {
    if (!root) continue;
    root.traverse((n) => {
      if (!n.isMesh) return;
      const id = String((n.userData && (n.userData.asset || n.userData.libraryId)) || n.name || '');
      if (/table|bench|sofa|counter|cabinet/.test(id)) out.push(n);
    });
  }
  return out;
}
function rayBlocks(from, to, meshes) {
  if (!meshes.length) return false;
  const dir = to.clone().sub(from);
  const dist = dir.length();
  if (dist < 0.3) return false;
  dir.multiplyScalar(1 / dist);
  _ray.set(from, dir);
  _ray.near = 0.05;
  _ray.far = dist - 0.2;
  return _ray.intersectObjects(meshes, false).some((h) => h.object.visible !== false);
}
function hideShell(pos, target, built, pattern) {
  if (!built || !built.kit) return;
  const meshes = [];
  for (const root of [built.kit.set, built.kit.dressing]) {
    if (!root) continue;
    root.traverse((n) => {
      if (!n.isMesh) return;
      const id = String(n.userData.asset || n.userData.libraryId || '');
      if (pattern.test(n.name || '') || pattern.test(id)) meshes.push(n);
    });
  }
  if (!meshes.length) return;
  const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);
  const aim = new THREE.Vector3(target[0], target[1], target[2]);
  const dir = aim.clone().sub(origin);
  const dist = dir.length();
  if (dist < 0.3) return;
  dir.multiplyScalar(1 / dist);
  _ray.set(origin, dir);
  _ray.near = 0.02;
  _ray.far = dist - 0.12;
  for (const hit of _ray.intersectObjects(meshes, false)) hideBlocker(hit.object, null, built);
}
/** Dollhouse: hide a wall, ceiling or wall edge between the lens and the subject. Undo on the next build. */
function cutaway(pos, target, built) {
  if (!built || !built.kit || !pos || !target) return;
  const meshes = [];
  for (const root of [built.kit.set, built.kit.dressing]) {
    if (!root) continue;
    root.traverse((n) => { if (n.isMesh && SHELL.test(n.name || '')) meshes.push(n); });
  }
  if (!meshes.length) return;
  const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);
  const look = new THREE.Vector3(target[0], target[1], target[2]);
  const aims = [look, new THREE.Vector3(look.x, Math.min(look.y, 0.4), look.z)];
  for (const aim of aims) {
    const dir = aim.clone().sub(origin);
    const dist = dir.length();
    if (dist < 0.3) continue;
    dir.multiplyScalar(1 / dist);
    _ray.set(origin, dir); _ray.near = 0.02; _ray.far = dist - 0.12;
    for (const hit of _ray.intersectObjects(meshes, false)) hideBlocker(hit.object, null, built);
  }
}
/** First azimuth whose head, chest, feet and group corners are clear, and whose frame is not filled by a near obstacle. */
function clearAzimuth(target, az, el, distFor, obstacles, dirOf, built, fov, aspect) {
  const tries = [az, -az, 4 * DEG, az + 40 * DEG, -az - 40 * DEG, 100 * DEG, -100 * DEG, 150 * DEG, -150 * DEG, 180 * DEG];
  const points = aimPoints(built, target);
  const actors = new Set((built && built.actors || []).map((a) => a.object).filter(Boolean));
  let fallback = null;
  for (const a of tries) {
    const dir = dirOf(a, el), dist = distFor(a);
    const cam = target.clone().addScaledVector(dir, dist);
    let blocked = null;
    for (const p of points) {
      const hit = rayHit(p, cam, obstacles, actors);
      if (hit && hit.distance < dist - 0.3) { blocked = hit; break; }
    }
    const cover = blocked ? null : coverageBlock(cam, target, obstacles, fov, aspect, actors);
    if (!blocked && !cover) return { az: a, dist };
    const offender = blocked ? blocked.object : cover.object;
    const instanceId = blocked ? blocked.instanceId : undefined;
    if (!fallback) fallback = { az: a, dist, offender, instanceId };
  }
  if (fallback && fallback.offender) {
    hideBlocker(fallback.offender, fallback.instanceId, built);
    if (typeof window !== 'undefined') (window.__previzClips || (window.__previzClips = [])).push({ label: 'hid near dressing', az: +fallback.az.toFixed(2) });
  }
  return fallback ? { az: fallback.az, dist: fallback.dist } : { az, dist: distFor(az) };
}

/** The kite (and anything else marked noFit) so a full shot can back up far enough to keep it in frame. */
function airborneBox(built) {
  const box = new THREE.Box3();
  for (const p of built.props || []) if (p.object && p.object.userData.noFit) box.union(new THREE.Box3().setFromObject(p.object));
  return box;
}

/** Box of the props the subject uses (rides, sits on, holds, walks). Empty box when none. */
function relatedBox(built, subj) {
  const b = new THREE.Box3();
  void subj;
  for (const p of built.props) if (fitsProp(p)) b.union(worldBox(p.object));
  return b;
}
// big set pieces (a house, a bus) stay partly out of frame; people-sized props are fitted
const NO_FIT = ['house', 'building', 'tree', 'fountain', 'bus', 'truck'];
function fitsProp(p) {
  const a = p.object.userData.asset || String(p.object.userData.libraryId || '').split('__')[0];
  return p.related && !NO_FIT.includes(a);
}

/**
 * Smallest camera distance along dir (unit, from target toward the camera) that keeps every corner of box inside
 * the frame (margin 0..1 of the half-frame). Vertical fov in degrees.
 */
export function fitDistance(box, target, dir, fov, aspect, margin = 0.85) {
  if (box.isEmpty()) return 0;
  const ty = Math.tan(fov * DEG / 2) * margin, tx = ty * aspect;
  const back = dir.clone().negate();
  const rgt = new THREE.Vector3().crossVectors(back, new THREE.Vector3(0, 1, 0));
  if (rgt.lengthSq() < 1e-6) rgt.set(1, 0, 0); rgt.normalize();
  const up = new THREE.Vector3().crossVectors(rgt, back).normalize();
  let d = 0;
  const c = new THREE.Vector3();
  for (let i = 0; i < 8; i++) {
    c.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(target);
    const along = c.dot(dir); // toward the camera
    d = Math.max(d, along + Math.abs(c.dot(rgt)) / tx, along + Math.abs(c.dot(up)) / ty);
  }
  return d;
}

function still(pos, target, fov, label) { const s = { pos, target, fov }; return { start: s, end: s, label }; }

/** Interpolate two camera states (t in 0..1, eased) onto a PerspectiveCamera. */
export function applyCamera(camera, a, b, t = 1) {
  const e = t * t * (3 - 2 * t);
  const lerp = (u, v) => u.map((x, i) => x + (v[i] - x) * e);
  camera.position.fromArray(lerp(a.pos, b.pos));
  camera.fov = a.fov + (b.fov - a.fov) * e;
  camera.updateProjectionMatrix();
  camera.lookAt(...lerp(a.target, b.target));
}
