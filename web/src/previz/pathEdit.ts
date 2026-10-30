// Editing a camera path in Top view: pure functions on the keys, no Three.js. Every edit returns a new array; the
// keys stay in time order, first t = 0, last t = 1, 2 to 24 of them (the spec's own limits).
import { keyTime, sampleCamera } from "./cameraMath.ts";
import type { CameraKey, PrevizSpec, Vec3 } from "./spec";

export type Keys = CameraKey[];
export const MAX_KEYS = 24;

function copyKey(k: CameraKey): CameraKey {
  return { ...k, pos: [k.pos[0], k.pos[1], k.pos[2]], look: [k.look[0], k.look[1], k.look[2]] };
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

/** The path at key time t (no speed, no ease, no distance): where a key placed at t sits on the drawn curve. */
export function curveAt(keys: Keys, t: number, lensFocal: number, curve = 1): { pos: Vec3; look: Vec3; focal: number; roll: number } {
  const bare = {
    lens: { focal_mm: lensFocal, film_gauge_mm: 36 },
    camera: { interp: "catmull_rom", ease: "linear", speed: 1, taper: 0, distance: 1, keys, curve },
  } as unknown as PrevizSpec;
  const s = sampleCamera(bare, t);
  return { pos: s.pos, look: s.look, focal: s.focal, roll: s.roll };
}

/** Shot time (0 to 1) at which the camera reaches key time t: the inverse of keyTime, which only ever climbs. */
export function shotTimeAt(camera: Pick<PrevizSpec["camera"], "speed" | "taper" | "ease">, t: number): number {
  const target = Math.min(1, Math.max(0, t));
  let lo = 0;
  let hi = 1;
  if (keyTime(camera, 1) < target) return 1;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (keyTime(camera, mid) < target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** The nearest point of the curve to (x, z) on the ground: its key time and how far it is, in metres. */
export function keyTimeAt(keys: Keys, x: number, z: number, lensFocal = 35): { t: number; dist: number } {
  const steps = 96;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i <= steps; i++) {
    const p = curveAt(keys, i / steps, lensFocal).pos;
    const d = Math.hypot(p[0] - x, p[2] - z);
    if (d < bestD) {
      bestD = d;
      best = i / steps;
    }
  }
  // refine around the best sample
  let lo = Math.max(0, best - 1 / steps);
  let hi = Math.min(1, best + 1 / steps);
  for (let i = 0; i < 24; i++) {
    const a = lo + (hi - lo) / 3;
    const b = hi - (hi - lo) / 3;
    const pa = curveAt(keys, a, lensFocal).pos;
    const pb = curveAt(keys, b, lensFocal).pos;
    if (Math.hypot(pa[0] - x, pa[2] - z) < Math.hypot(pb[0] - x, pb[2] - z)) hi = b;
    else lo = a;
  }
  const t = (lo + hi) / 2;
  const p = curveAt(keys, t, lensFocal).pos;
  return { t, dist: Math.hypot(p[0] - x, p[2] - z) };
}

/** Which key sits at key time t (within tol), or -1. */
export function keyAtTime(keys: Keys, t: number, tol: number): number {
  let best = -1;
  let bestD = tol;
  keys.forEach((k, i) => {
    const d = Math.abs(k.t - t);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/**
 * A new point on the curve at key time t: its height, aim, lens and roll are the curve's own there, so the path does not
 * change shape until the point is moved. Null at 24 points, or where a point already is.
 */
export function insertPoint(keys: Keys, t: number, lensFocal = 35, curve = 1): { keys: Keys; index: number } | null {
  if (keys.length >= MAX_KEYS) return null;
  const at = round3(Math.min(1, Math.max(0, t)));
  if (keys.some((k) => Math.abs(k.t - at) < 0.004)) return null;
  const s = curveAt(keys, at, lensFocal, curve);
  const key: CameraKey = { t: at, pos: s.pos.map(round3) as Vec3, look: s.look.map(round3) as Vec3 };
  if (keys.some((k) => k.focal_mm != null)) key.focal_mm = Math.round(s.focal * 10) / 10;
  if (keys.some((k) => k.roll != null)) key.roll = Math.round(s.roll * 10) / 10;
  const out = [...keys.map(copyKey), key].sort((a, b) => a.t - b.t);
  return { keys: out, index: out.indexOf(key) };
}

/**
 * The camera moved by hand on a frame (W A S D, a drag): that frame's key holds the new view. The key at time t (within
 * tol) takes it, else a new key is made there. Null at 24 points. Every other key is untouched (keys stay the truth).
 */
export function setViewAt(keys: Keys, t: number, view: { pos: Vec3; look: Vec3 }, tol: number, lensFocal = 35, curve = 1): { keys: Keys; index: number } | null {
  let index = keyAtTime(keys, t, tol);
  let out = keys.map(copyKey);
  if (index < 0) {
    const made = insertPoint(keys, t, lensFocal, curve);
    if (!made) return null;
    out = made.keys;
    index = made.index;
  }
  out[index] = { ...out[index], pos: view.pos.map(round3) as Vec3, look: view.look.map(round3) as Vec3 };
  return { keys: out, index };
}

/** Move one point on the ground. Its height, aim, lens and roll stay; every other point is untouched. */
export function movePoint(keys: Keys, index: number, x: number, z: number): Keys {
  return keys.map((k, i) => (i === index ? { ...copyKey(k), pos: [round3(x), k.pos[1], round3(z)] } : k));
}

/** Move every point by the same amount, aims too: the whole move slides. */
export function movePath(keys: Keys, dx: number, dz: number): Keys {
  return keys.map((k) => ({
    ...copyKey(k),
    pos: [round3(k.pos[0] + dx), k.pos[1], round3(k.pos[2] + dz)],
    look: [round3(k.look[0] + dx), k.look[1], round3(k.look[2] + dz)],
  }));
}

/** Turn one point's lens to face (x, z) on the ground. It keeps how far ahead it looks and the tilt. */
export function aimPoint(keys: Keys, index: number, x: number, z: number): Keys {
  const k = keys[index];
  if (!k) return keys;
  const dx = x - k.pos[0];
  const dz = z - k.pos[2];
  const len = Math.hypot(dx, dz);
  if (len < 1e-4) return keys;
  const reach = Math.max(0.5, Math.hypot(k.look[0] - k.pos[0], k.look[2] - k.pos[2]));
  const look: Vec3 = [round3(k.pos[0] + (dx / len) * reach), k.look[1], round3(k.pos[2] + (dz / len) * reach)];
  return keys.map((key, i) => (i === index ? { ...copyKey(key), look } : key));
}

/** Rotate one point's aim around it by yaw radians (positive turns left, seen from above). */
export function turnPoint(keys: Keys, index: number, yawRad: number): Keys {
  const k = keys[index];
  if (!k) return keys;
  const ox = k.look[0] - k.pos[0];
  const oz = k.look[2] - k.pos[2];
  const c = Math.cos(yawRad);
  const s = Math.sin(yawRad);
  return aimPoint(keys, index, k.pos[0] + ox * c + oz * s, k.pos[2] - ox * s + oz * c);
}

/** Can this point go? Never the first or the last, and a path keeps two. */
export function canRemove(keys: Keys, index: number): boolean {
  return keys.length > 2 && index > 0 && index < keys.length - 1;
}

/** Remove a middle point. The ends and the last two points stay. */
export function removePoint(keys: Keys, index: number): Keys {
  if (!canRemove(keys, index)) return keys;
  return keys.filter((_, i) => i !== index).map(copyKey);
}

/** A ground box around everything that matters on the map, with a margin: half is the half-height in metres. */
export function fitBounds(points: Array<[number, number]>, opts: { minHalf?: number; margin?: number; aspect?: number } = {}): { cx: number; cz: number; half: number } {
  const minHalf = opts.minHalf ?? 4;
  const margin = opts.margin ?? 1.25;
  const aspect = Math.max(0.2, opts.aspect ?? 1);
  const finite = points.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!finite.length) return { cx: 0, cz: 0, half: minHalf };
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const [x, z] of finite) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  // z runs up the screen and x across it; the width is aspect times the height
  const half = Math.max(minHalf, ((maxZ - minZ) / 2) * margin, ((maxX - minX) / 2 / aspect) * margin);
  return { cx: (minX + maxX) / 2, cz: (minZ + maxZ) / 2, half };
}
