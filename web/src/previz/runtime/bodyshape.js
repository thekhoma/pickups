// Car body types for the one car model the library has (a notchback sedan: a roof, a rear window, then a flat trunk lid).
// The render takes the layout's depth as the car's silhouette, so a sentence that says "coupe" or "sports car" should not
// send a three-box sedan. The sleek words reshape a per-instance copy of the sedan's body into a fastback: the greenhouse
// sits a little lower and the roof runs in one line down to the tail, filling the notch over the trunk.
// SUV, pickup truck and van have their own models (library words), so they are left alone. "hatchback" takes the
// fastback line too: one slope from the roof to the tail is nearer a hatch than the sedan's notch.
import * as THREE from 'three';

/** Body type for a vehicle word, or null when the library model already is that body. */
export function bodyTypeOf(word, asset) {
  if (asset !== 'car') return null;
  const w = String(word || '').toLowerCase();
  if (/\b(coupe|sports? car|race ?car|racecar|fastback|supercar|muscle car|hatchback|gt|roadster|convertible|electric car)\b/.test(w)) return 'fastback';
  return null; // car, sedan, taxi, police car, limo: the sedan as modelled (notchback)
}

const BELT = 0.68; // metres: below this the body is never moved (wheels, sills, bumpers, the side's lower half)
const ROOF_DROP = 0.94; // the greenhouse above the beltline sits at 94 percent of its height (driver's head still clears)

/**
 * Reshape a car instance (fresh from lib.instance, not yet placed) into the given body type. Every mesh above the
 * beltline gets its own geometry copy, so the shared template and other cars stay the sedan. Returns the new
 * geometries (the caller disposes them on teardown).
 */
export function shapeBody(car, type) {
  if (type !== 'fastback') return [];
  car.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(car.matrixWorld).invert();
  const meshes = [];
  car.traverse((o) => { if (o.isMesh && !o.isInstancedMesh && o.geometry && o.geometry.attributes.position) meshes.push(o); });
  const skip = (o) => { const mats = Array.isArray(o.material) ? o.material : [o.material]; return mats.some((m) => /seat/.test(String(m && m.name))); };
  // the car's own frame: +Z forward, +Y up, ground at y = 0 (library convention)
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  const toCar = new Map();
  for (const m of meshes) {
    const mm = new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld);
    toCar.set(m, mm);
    if (skip(m)) continue;
    const p = m.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) box.expandByPoint(v.fromBufferAttribute(p, i).applyMatrix4(mm));
  }
  if (box.isEmpty()) return [];
  const tail = box.min.z, len = box.max.z - box.min.z, H = box.max.y;
  // the top of the body along its length (5 cm bins), before any change
  const BINS = Math.max(8, Math.ceil(len / 0.05));
  const top = new Float32Array(BINS).fill(-Infinity);
  const bin = (z) => Math.max(0, Math.min(BINS - 1, Math.floor(((z - tail) / len) * BINS)));
  for (const m of meshes) {
    if (skip(m)) continue;
    const p = m.geometry.attributes.position, mm = toCar.get(m);
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(mm); const b = bin(v.z); if (v.y > top[b]) top[b] = v.y; }
  }
  for (let b = 1; b < BINS; b++) if (top[b] === -Infinity) top[b] = top[b - 1];
  // the roof's rear edge: the rearmost bin still within 3 cm of the roof height
  let roofRear = BINS - 1;
  for (let b = 0; b < BINS; b++) if (top[b] >= H - 0.03) { roofRear = b; break; }
  const zRoof = tail + ((roofRear + 0.5) / BINS) * len;
  const yRoof = BELT + (H - BELT) * ROOF_DROP;
  const yTail = Math.max(top[0], BELT + 0.36); // the tail's top edge: a little above the sedan's trunk lid line
  const line = (z) => yTail + (yRoof - yTail) * Math.max(0, Math.min(1, (z - tail) / Math.max(0.01, zRoof - tail)));
  const made = [];
  const back = new THREE.Matrix4();
  for (const m of meshes) {
    if (skip(m)) continue;
    const mm = toCar.get(m);
    back.copy(mm).invert();
    const g = m.geometry.clone();
    // quantized GLB positions (Uint16, Int16) can't hold the moved values: this copy is plain floats
    const q = g.attributes.position;
    const f = new Float32Array(q.count * 3);
    for (let i = 0; i < q.count; i++) { f[i * 3] = q.getX(i); f[i * 3 + 1] = q.getY(i); f[i * 3 + 2] = q.getZ(i); }
    g.setAttribute('position', new THREE.BufferAttribute(f, 3));
    g.deleteAttribute('normal');
    const p = g.attributes.position;
    let moved = false;
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).applyMatrix4(mm);
      if (v.y <= BELT) continue;
      let y = BELT + (v.y - BELT) * ROOF_DROP;
      if (v.z < zRoof) {
        // behind the roof: stretch the body's top at this length up (or down) onto the roof-to-tail line
        const t = top[bin(v.z)] - BELT;
        if (t > 0.02) y = BELT + (v.y - BELT) * ((line(v.z) - BELT) / t);
      }
      if (Math.abs(y - v.y) > 1e-5) { v.y = y; v.applyMatrix4(back); p.setXYZ(i, v.x, v.y, v.z); moved = true; }
    }
    if (!moved) { g.dispose(); continue; }
    p.needsUpdate = true;
    g.computeVertexNormals();
    g.computeBoundingBox(); g.computeBoundingSphere();
    m.geometry = g;
    made.push(g);
  }
  car.userData.body = type;
  return made;
}

/** The body's top height (car frame) at fractions of its length from the tail, for tests and the shot notes. */
export function roofProfile(car, fractions = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8]) {
  car.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(car.matrixWorld).invert();
  const pts = [];
  const v = new THREE.Vector3();
  car.traverse((o) => {
    if (!o.isMesh || o.isInstancedMesh || !o.geometry || !o.geometry.attributes.position) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    if (mats.some((m) => /seat/.test(String(m && m.name)))) return;
    const mm = new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld), p = o.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(mm); pts.push(v.z, v.y); }
  });
  let z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < pts.length; i += 2) { z0 = Math.min(z0, pts[i]); z1 = Math.max(z1, pts[i]); }
  return fractions.map((f) => {
    const a = z0 + (z1 - z0) * (f - 0.03), b = z0 + (z1 - z0) * (f + 0.03);
    let y = 0;
    for (let i = 0; i < pts.length; i += 2) if (pts[i] >= a && pts[i] <= b && pts[i + 1] > y) y = pts[i + 1];
    return +y.toFixed(3);
  });
}
