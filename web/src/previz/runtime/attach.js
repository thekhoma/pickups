// Attach helpers on LOADED GLBs, driven by the named attach points that survive in the meshopt GLBs:
//   person: attach_hand_L/R, attach_foot_L/R, attach_seat, attach_back, attach_head
//   bench: attach_seat_0..2 (attach_seat = middle), table: attach_top, cup / phone: attach_grip
//   bicycle: attach_seat, attach_hand_L/R, attach_foot_L/R, skateboard: attach_feet, boat: attach_seat
//   car / suv / truck / van / bus: attach_seat_<driver|passenger|rear_L|rear_R|row..>, attach_wheel_grip_L/R,
//   attach_pedal_gas / brake, attach_footrest, attach_roof (vehicle.userData.seats holds the seat table)
// Every placement is a table lookup or a rigid alignment: no solver runs at build time, so results are instant and
// deterministic. Poses for vehicles and riders were solved offline (person_poses.json: in_vehicle_frozen, rider_frozen).
import * as THREE from 'three';
import { setPose, rigOf, attachPoint, scaleKey, mirrorPose, frozenVehiclePose, frozenRider, getPair } from './poser.js';

const DEG = Math.PI / 180;
const _v = new THREE.Vector3();
const wp = (o) => o.getWorldPosition(new THREE.Vector3());

export const SEAT_ALIASES = { driver: 'driver', drivers: 'driver', front: 'passenger', passenger: 'passenger', shotgun: 'passenger',
  back: 'rear_R', rear: 'rear_R', backseat: 'rear_R', back_left: 'rear_L', back_right: 'rear_R', rear_L: 'rear_L', rear_R: 'rear_R' };
export const VEHICLES = ['car', 'suv', 'truck', 'van', 'bus'];

function reparentAt(child, parent) { if (child.parent) child.parent.remove(child); parent.add(child); child.position.set(0, 0, 0); child.rotation.set(0, 0, 0); }
// A seat a kit stretched (a bench scaled to a stool, [0.3, 1, 1]) must not stretch the person on it: they sit under an
// anchor that cancels the seat's world scale, so the body keeps its true proportions while it still moves with the seat.
function seatAnchor(seat) {
  seat.updateMatrixWorld(true);
  const s = new THREE.Vector3();
  seat.matrixWorld.decompose(new THREE.Vector3(), new THREE.Quaternion(), s);
  if (Math.abs(s.x - 1) < 0.005 && Math.abs(s.y - 1) < 0.005 && Math.abs(s.z - 1) < 0.005) return seat;
  let anchor = seat.children.find((c) => c.userData.seatAnchor);
  if (!anchor) { anchor = new THREE.Group(); anchor.name = 'seat_anchor'; anchor.userData.seatAnchor = true; seat.add(anchor); }
  anchor.scale.set(1 / s.x, 1 / s.y, 1 / s.z);
  anchor.updateMatrixWorld(true);
  return anchor;
}
// move `obj` so its point `from` (a descendant) lands on world position `to`
function alignPoint(obj, from, to) {
  obj.updateMatrixWorld(true);
  const d = to.clone().sub(wp(from));
  const w = wp(obj).add(d);
  obj.position.copy(obj.parent ? obj.parent.worldToLocal(w) : w);
  obj.updateMatrixWorld(true);
}

/**
 * attachProp(person, prop, point = 'attach_hand_R', { upright = false })
 * Parents the prop to the person's attach point through the prop's attach_grip. upright keeps drinks level in world space.
 */
export function attachProp(person, prop, point = 'attach_hand_R', options = {}) {
  const a = attachPoint(person, point);
  if (!a) throw new Error('No attach point ' + point + ' on ' + person.name);
  prop.position.set(0, 0, 0); prop.rotation.set(0, 0, 0); prop.scale.set(1, 1, 1);
  prop.updateMatrixWorld(true);
  const grip = attachPoint(prop, 'attach_grip');
  const gm = new THREE.Matrix4();
  if (grip) gm.copy(prop.matrixWorld).invert().multiply(grip.matrixWorld);
  gm.invert().decompose(prop.position, prop.quaternion, prop.scale);
  if (prop.parent) prop.parent.remove(prop);
  a.add(prop);
  prop.updateMatrixWorld(true);
  if (options.upright) {
    const q = new THREE.Quaternion(); prop.getWorldQuaternion(q);
    const e = new THREE.Euler().setFromQuaternion(q, 'YXZ');
    const target = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, e.y, 0, 'YXZ'));
    const pq = new THREE.Quaternion(); a.getWorldQuaternion(pq);
    prop.quaternion.copy(pq.invert().multiply(target));
    prop.updateMatrixWorld(true);
  }
  return prop;
}

/**
 * placePair(pair, a, b) (library v1.4): pair is an id ('hug', 'kiss', 'shake_hands', 'fight', 'couple') or a spec pair
 * { poses: [a, b], offset, facing }. Poses both, then puts b at offset in a's frame (metres, +Z = where a faces) turned by
 * facing degrees. a and b end up with the same parent. Returns [a, b].
 */
export function placePair(pair, a, b) {
  const pr = typeof pair === 'string' ? getPair(pair) : pair;
  if (!pr) throw new Error('Unknown pair ' + pair);
  const poses = pr.poses || [pr.a, pr.b];
  if (poses[0]) setPose(a, poses[0]);
  if (poses[1]) setPose(b, poses[1]);
  if (a.parent && b.parent !== a.parent) a.parent.add(b);
  const yaw = a.rotation.y;
  const off = new THREE.Vector3(...(pr.offset || [0, 0, 0.9])).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  b.position.copy(a.position).add(off);
  b.rotation.set(0, yaw + (pr.facing || 0) * Math.PI / 180, 0);
  b.updateMatrixWorld(true); // same floor height as a (setPose already grounds each rig on its own origin)
  return [a, b];
}
/** Guitar on the strap (library v1.3.1 GUITAR_MOUNT, person-local metres and degrees). */
export const GUITAR_MOUNT = { pos: [-0.3, 0.86, 0.15], rot: [0, 0, -58] };
/** Two kid-heights up and in front of the flyer. The tail hangs free of the ground. */
export const KITE_MOUNT = { pos: [0.35, 2.40, 1.85], rot: [-10, 148, 3] };
/** Open a book GLB (half_L / half_R hinges) to deg degrees. */
export function setBookOpen(book, deg) {
  const a = (90 - deg / 2) * Math.PI / 180;
  const L = book.getObjectByName('half_L'), R = book.getObjectByName('half_R');
  if (L) L.rotation.y = a;
  if (R) R.rotation.y = -a;
  book.userData.open = deg; book.updateMatrixWorld(true);
  return book;
}
/**
 * attachActionProp(person, slot, prop, { upright }) (library v1.3.1): one call for every held-prop slot.
 * 'hand_L' / 'hand_R': attachProp on that palm. 'hands': centred between both palms, square to the body (carry).
 * 'foot_L' / 'foot_R': resting just ahead of the toe (kick). 'strap': the guitar on GUITAR_MOUNT. The prop ends up parented to the person.
 */
export function attachActionProp(person, slot, prop, options = {}) {
  if (prop.getObjectByName('half_L') && !prop.userData.open) setBookOpen(prop, 140); // a held book is read open
  if (!slot || slot === 'hand_L' || slot === 'hand_R') return attachProp(person, prop, 'attach_' + (slot || 'hand_R'), options);
  if (prop.parent) prop.parent.remove(prop);
  if (slot === 'push') { // v1.4: the cart's attach_push at the person's origin, same facing (push_cart puts both hands on the handle)
    prop.position.set(0, 0, 0); prop.rotation.set(0, 0, 0); prop.scale.set(1, 1, 1); prop.updateMatrixWorld(true);
    const ap = attachPoint(prop, 'attach_push'); if (ap) prop.position.sub(wp(ap));
    person.add(prop); prop.updateMatrixWorld(true); return prop;
  }
  if (slot === 'airborne') { // v1.4: kite in the air ahead of the person; the caller draws the line (makeLeash hand -> attach_line)
    prop.position.set(...KITE_MOUNT.pos); prop.rotation.set(...KITE_MOUNT.rot.map((v) => v * Math.PI / 180)); prop.scale.set(0.28, 0.28, 0.28);
    prop.userData.noFit = true; // camera framing ignores it (camera.js worldBox)
    prop.userData.kiteFloat = true;
    prop.userData.kiteHome = prop.position.clone();
    const tail = prop.getObjectByName('geo_tail');
    if (tail) tail.scale.set(1, 2.2, 1);
    person.add(prop); prop.updateMatrixWorld(true);
    const sail = prop.getObjectByName('geo_sail') || prop.getObjectByName('sail');
    if (tail && sail) {
      const box = new THREE.Box3().setFromObject(sail);
      const corner = prop.worldToLocal(new THREE.Vector3(box.min.x, box.min.y, (box.min.z + box.max.z) / 2));
      tail.position.copy(corner);
    }
    prop.traverse((n) => {
      if (!n.isMesh || !n.material) return;
      const name = String(n.name || '').toLowerCase();
      if (/tail|line|string/.test(name)) return;
      const mats = [].concat(n.material);
      n.material = mats.map((m) => {
        if (!m || !m.color) return m;
        const copy = m.clone();
        copy.color.setHex(0x1B4F8A);
        return copy;
      });
      if (n.material.length === 1) n.material = n.material[0];
    });
    return prop;
  }
  if (slot === 'strap') {
    prop.position.set(...GUITAR_MOUNT.pos); prop.rotation.set(...GUITAR_MOUNT.rot.map((v) => v * Math.PI / 180)); prop.scale.set(1, 1, 1);
    person.add(prop); prop.updateMatrixWorld(true); return prop;
  }
  person.updateMatrixWorld(true);
  prop.position.set(0, 0, 0); prop.rotation.set(0, 0, 0); prop.scale.set(1, 1, 1); prop.updateMatrixWorld(true);
  const q = person.getWorldQuaternion(new THREE.Quaternion()), w = new THREE.Vector3();
  if (slot === 'hands') {
    const a = attachPoint(person, 'attach_hand_L'), b = attachPoint(person, 'attach_hand_R');
    if (!a || !b) return attachProp(person, prop, 'attach_hand_R', options);
    const c = new THREE.Box3().setFromObject(prop, true).getCenter(new THREE.Vector3());
    w.addVectors(a.getWorldPosition(new THREE.Vector3()), b.getWorldPosition(new THREE.Vector3())).multiplyScalar(0.5).sub(c.applyQuaternion(q));
  } else { // foot_L / foot_R
    const ank = attachPoint(person, 'attach_foot_' + slot.slice(-1));
    if (!ank) return attachProp(person, prop, 'attach_hand_R', options);
    const f = ank.getWorldPosition(new THREE.Vector3()), aq = ank.getWorldQuaternion(new THREE.Quaternion());
    const toe = new THREE.Vector3(0, 0, 1).applyQuaternion(aq).setY(0).normalize();
    const r = new THREE.Box3().setFromObject(prop, true).getSize(new THREE.Vector3()).y / 2;
    w.copy(f).addScaledVector(toe, 0.13 + r); w.y = Math.max(r, f.y + 0.02) - r; // ball centre just ahead of the toe
  }
  prop.position.copy(w); prop.quaternion.copy(q);
  const parent = person.parent;
  if (parent) { parent.add(prop); prop.position.copy(parent.worldToLocal(w.clone())); }
  prop.updateMatrixWorld(true);
  person.attach(prop); // keeps the world transform
  return prop;
}

/** Put an object on a surface point (cup on a table): attach_top or the object's top. */
export function placeOnTop(base, obj, offset = [0, 0]) {
  base.updateMatrixWorld(true);
  const top = attachPoint(base, 'attach_top');
  const p = top ? wp(top) : (() => { const b = new THREE.Box3().setFromObject(base); return new THREE.Vector3((b.min.x + b.max.x) / 2, b.max.y, (b.min.z + b.max.z) / 2); })();
  if (obj.parent) obj.parent.remove(obj);
  base.parent.add(obj);
  obj.position.copy(base.parent.worldToLocal(p.add(new THREE.Vector3(offset[0], 0, offset[1]))));
  obj.updateMatrixWorld(true);
  return obj;
}

/**
 * seatOn(seat, person, { slot = null, pose = 'sit' }): bench / chair. slot 0..2 on the bench (null = attach_seat, the middle).
 * The person is parented to the seat and faces its +Z.
 */
export function seatOn(seat, person, options = {}) {
  const name = options.point || (options.slot == null ? 'attach_seat' : 'attach_seat_' + options.slot); // v1.4 bed: attach_seat_L / _R, attach_lie
  const sp = attachPoint(seat, name) || attachPoint(seat, 'attach_seat');
  if (!sp) throw new Error(seat.name + ' has no ' + name);
  setPose(person, options.pose || 'sit', { ground: false });
  reparentAt(person, seatAnchor(seat));
  seat.updateMatrixWorld(true);
  alignPoint(person, attachPoint(person, 'attach_seat'), wp(sp));
  person.userData.seated = { on: seat.userData.asset || seat.name, slot: options.slot ?? null };
  return person;
}

/**
 * attachRider(vehicle, person): bicycle, skateboard, boat. Uses the frozen placement solved by the library
 * (hands on the bars, feet on the pedals), so nothing is solved at build time.
 */
export function attachRider(vehicle, person, options = {}) {
  const kind = vehicle.userData.asset;
  const sk = scaleKey(person) || 'adult';
  const fr = frozenRider(kind, sk);
  const seatPose = options.pose || { bicycle: 'ride_bike', skateboard: 'ride_skate', board: 'ride_skate', boat: 'sit_boat', horse: 'ride_horse', skis: 'ride_ski' }[kind] || 'sit';
  const skate = kind === 'skateboard' || kind === 'board';
  setPose(person, skate ? 'ride_skate' : (fr && !options.pose ? fr.pose : seatPose), { ground: false });
  if (skate) {
    const bend = (name, x, y, z) => { const j = person.getObjectByName(name); if (j) j.rotation.set(x * DEG, y * DEG, z * DEG); };
    // Sideways on the deck: the right foot points along the nose, the left foot sits across the tail.
    bend('hip_L', -36, 22, 16);
    bend('hip_R', -40, -48, -8);
    bend('knee_L', 70, 0, 0);
    bend('knee_R', 50, 0, 0);
    bend('ankle_L', 8, -12, -8);
    bend('ankle_R', 6, 32, 6);
    bend('shoulder_L', -6, 0, 28);
    bend('shoulder_R', 6, 0, -26);
    bend('elbow_L', -22, 0, 0);
    bend('elbow_R', -20, 0, 0);
  }
  reparentAt(person, vehicle);
  if (fr && !options.pose) {
    person.position.fromArray(fr.personPosition);
    person.rotation.y = skate ? -Math.PI / 2 : fr.personRotationY * DEG;
    rigOf(person).position.fromArray(fr.rigPosition);
    person.updateMatrixWorld(true);
  } else {
    person.rotation.y = kind === 'skateboard' || kind === 'board' ? -Math.PI / 2 : 0;
    person.updateMatrixWorld(true);
    const seat = attachPoint(vehicle, 'attach_seat'), feet = attachPoint(vehicle, 'attach_feet');
    if (seat) alignPoint(person, attachPoint(person, 'attach_seat'), wp(seat));
    else if (feet) { // skis, board (v1.4): the middle of the feet on the deck
      const mid = wp(attachPoint(person, 'attach_foot_L')).add(wp(attachPoint(person, 'attach_foot_R'))).multiplyScalar(0.5);
      const d = wp(feet).sub(mid), w = person.getWorldPosition(new THREE.Vector3()).add(d);
      person.position.copy(person.parent.worldToLocal(w)); person.updateMatrixWorld(true);
    }
  }
  if (skate) {
    person.updateMatrixWorld(true);
    const deck = attachPoint(vehicle, 'attach_feet') || vehicle;
    const deckY = wp(deck).y;
    const soleY = (side) => {
      let best = Infinity;
      const v = new THREE.Vector3();
      person.traverse((n) => {
        if (!n.isMesh) return;
        // the foot mesh sits under a geo_foot_<side> node (its own name is mesh_<n>)
        const name = (String(n.name || '') + ' ' + String((n.parent && n.parent.name) || '')).toLowerCase();
        if (!/foot|shoe|sole/.test(name)) return;
        if (side === 'L' && /_r\b|right/.test(name)) return;
        if (side === 'R' && /_l\b|left/.test(name)) return;
        const pos = n.geometry && n.geometry.attributes.position;
        if (!pos) return;
        for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i).applyMatrix4(n.matrixWorld); if (v.y < best) best = v.y; }
      });
      if (best < Infinity) return best;
      const ankle = person.getObjectByName('ankle_' + side);
      return ankle ? wp(ankle).y - 0.09 : Infinity;
    };
    let low = Math.min(soleY('L'), soleY('R'));
    if (low < Infinity && Math.abs(low - deckY) > 0.005) {
      person.position.y -= low - deckY;
      person.updateMatrixWorld(true);
    }
    for (const side of ['L', 'R']) {
      const knee = person.getObjectByName('knee_' + side);
      if (!knee) continue;
      for (let n = 0; n < 8; n++) {
        const y = soleY(side);
        if (Math.abs(y - deckY) <= 0.03) break;
        const before = knee.rotation.x;
        knee.rotation.x = before + (y > deckY ? 4 : -4) * DEG;
        person.updateMatrixWorld(true);
        const next = soleY(side);
        if (Math.abs(next - deckY) > Math.abs(y - deckY)) knee.rotation.x = before;
        person.updateMatrixWorld(true);
      }
    }
    // level both soles on the deck: a lean of a few degrees through the rig, then the lift
    const rig = rigOf(person);
    const base = rig.rotation.clone();
    const baseY = rig.position.y;
    let best = null;
    for (let ax = -8; ax <= 8; ax += 1) {
      for (let az = -8; az <= 8; az += 1) {
        rig.rotation.set(base.x + ax * DEG, base.y, base.z + az * DEG);
        rig.position.y = baseY;
        person.updateMatrixWorld(true);
        const l = soleY('L'), r = soleY('R');
        const score = Math.abs(l - r) + (Math.abs(ax) + Math.abs(az)) * 0.0008;
        if (!best || score < best.score) best = { score, ax, az, mid: (l + r) / 2 };
      }
    }
    if (best) {
      rig.rotation.set(base.x + best.ax * DEG, base.y, base.z + best.az * DEG);
      rig.position.y = baseY + (deckY + 0.004 - best.mid) / Math.max(1e-3, rig.parent ? rig.parent.getWorldScale(new THREE.Vector3()).y : 1);
      person.updateMatrixWorld(true);
    }
  }
  person.userData.rider = { vehicle: kind, frozen: !!fr };
  return person;
}

function vehicleKind(v) { const a = v.userData.assetId || v.userData.asset; return VEHICLES.includes(a) ? a : (v.userData.variant === 'sedan' ? 'car' : v.userData.variant); }

/** Seats a vehicle offers (driver, passenger, rear_L, rear_R, bus rows). */
export function vehicleSeats(vehicle) { return Object.keys((vehicle.userData && vehicle.userData.seats) || {}); }

/**
 * seatInVehicle(person, vehicle, seat = 'driver', pose = null, { litInside = true })
 * seat: driver | passenger | rear_L | rear_R | bus row<i>_<L|R>_<win|aisle> (aliases: front, shotgun, back, rear)
 * pose: drive (driver default) | sit_passenger (default elsewhere) | lean_window
 * Needs the interior GLB (car.glb etc., not the __exterior LOD). The person is parented to the vehicle.
 */
export function seatInVehicle(person, vehicle, seat = 'driver', pose = null, options = {}) {
  if (!vehicle.userData.interior) throw new Error(vehicle.name + ' has no interior (load ' + vehicle.userData.asset + '.glb, not the __exterior LOD)');
  seat = SEAT_ALIASES[seat] || seat;
  const seats = vehicle.userData.seats || {};
  const sp = seats[seat];
  if (!sp) throw new Error(vehicle.name + ' has no seat ' + seat + ' (has ' + Object.keys(seats).join(', ') + ')');
  pose = pose || (seat === 'driver' ? 'drive' : 'sit_passenger');
  if (pose === 'drive' && seat !== 'driver') pose = 'sit_passenger';
  const kind = vehicleKind(vehicle);
  const sk = scaleKey(person) || (rigOf(person).scale.x < 0.85 ? 'child' : 'adult');
  const base = sp.alias || seat;
  const key = (seat === 'driver' ? 'driver' : (/^rear|row/.test(base) && kind !== 'bus') ? 'rear' : (kind === 'bus' ? 'row' : 'passenger')) + ':' + pose;
  let frozen = frozenVehiclePose(kind, sk, key) || frozenVehiclePose(kind, sk, key.split(':')[0] + ':sit_passenger') || frozenVehiclePose(kind, 'adult', key);
  if (!frozen) throw new Error('No frozen pose for ' + kind + ' ' + sk + ' ' + key);
  if (pose === 'lean_window' && sp.x > 0 && seat !== 'driver') frozen = mirrorPose(frozen);
  reparentAt(person, vehicle);
  setPose(person, frozen, { ground: false });
  rigOf(person).position.set(0, 0, 0);
  const seatNode = attachPoint(vehicle, 'attach_seat_' + seat);
  vehicle.updateMatrixWorld(true);
  alignPoint(person, attachPoint(person, 'attach_seat'), wp(seatNode));
  const sc = rigOf(person).scale.x;
  if (seat === 'driver' && sc < 0.85) { person.position.z += (1 - sc) * 0.36; person.position.y += 0.025; person.updateMatrixWorld(true); }
  if (options.litInside !== false) rigOf(person).traverse((m) => { if (m.isMesh) m.receiveShadow = false; });
  person.userData.pose = pose;
  person.userData.seated = { vehicle: kind, seat, pose };
  return person;
}

/** Leash between two attach points (person attach_hand_R to dog attach_leash). Returns the mesh (add it to the scene root). */
export function makeLeash(fromObj, fromPoint, toObj, toPoint, { sag = 0.18, material = null, radius = 0.006 } = {}) {
  const mat = material || new THREE.MeshStandardMaterial({ name: 't2_detail', color: 0x1A1A1A, roughness: 0.5 });
  const m = new THREE.Mesh(new THREE.BufferGeometry(), mat);
  m.name = 'leash'; m.castShadow = true;
  m.userData.leash = { from: fromObj, fromPoint, to: toObj, toPoint, sag, radius };
  updateLeash(m);
  return m;
}

/** Rebuild a leash from its attach points. A kite line rises as the kite bobs, so the tube cannot stay baked. */
export function updateLeash(mesh) {
  const u = mesh && mesh.userData && mesh.userData.leash;
  if (!u) return mesh;
  const a = wp(attachPoint(u.from, u.fromPoint)), b = wp(attachPoint(u.to, u.toPoint));
  const mid = a.clone().lerp(b, 0.5);
  mid.y -= u.sag || 0;
  if (mesh.geometry) mesh.geometry.dispose();
  mesh.geometry = new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 24, u.radius || 0.006, 5);
  return mesh;
}
void _v;
