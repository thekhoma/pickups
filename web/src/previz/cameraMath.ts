// Camera curves with no Three.js and no clock, so a frame index always lands on the same pose.
import type { PrevizSpec, Vec3 } from "./spec";

export interface CameraSample {
  pos: Vec3;
  look: Vec3;
  focal: number;
  roll: number;
  /** Aim offset in fractions of the half-frame. Positive aims right and up. */
  frame?: [number, number];
}

export interface CameraPose {
  pos: Vec3;
  look: Vec3;
}

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Easy ease: no slope at the start or the end. */
function smootherstep(t: number): number {
  const x = clamp01(t);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u;
}

/** Catmull-Rom through the keys. curve scales the tangents (Run 14, walkthrough 22): 1 is the classic smooth path,
 * 0 straight lines from key to key, up to 1.5 rounder. The path always passes through every key. */
function catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, u: number, curve = 1): Vec3 {
  const u2 = u * u;
  const u3 = u2 * u;
  const out: Vec3 = [0, 0, 0];
  if (curve === 1) {
    for (let i = 0; i < 3; i++) {
      out[i] =
        0.5 *
        (2 * p1[i] +
          (-p0[i] + p2[i]) * u +
          (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * u2 +
          (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * u3);
    }
    return out;
  }
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;
  for (let i = 0; i < 3; i++) {
    const m1 = 0.5 * curve * (p2[i] - p0[i]);
    const m2 = 0.5 * curve * (p3[i] - p1[i]);
    out[i] = h00 * p1[i] + h10 * m1 + h01 * p2[i] + h11 * m2;
  }
  return out;
}

/** The path's curve setting, kept in range: 1 (smooth) unless set. */
export function curveOf(camera: { curve?: number } | undefined): number {
  const c = camera?.curve;
  return typeof c === "number" && Number.isFinite(c) ? Math.min(1.5, Math.max(0, c)) : 1;
}

/** Key time (the keys' own t, 0 to 1) at normalised shot time: the move's speed, then its ease. */
export function keyTime(camera: Pick<PrevizSpec["camera"], "speed" | "taper" | "ease">, tNorm: number): number {
  const speed = camera.speed ?? 1;
  const along = clamp01(clamp01(tNorm) * (speed > 0 ? speed : 1));
  const taper = camera.taper ?? (camera.ease === "linear" ? 0 : 1);
  const eased = smootherstep(along);
  return along + (eased - along) * clamp01(taper);
}

/** Pose at normalised time. t = 0 is the first key, whatever the easing. */
export function sampleCamera(spec: PrevizSpec, tNorm: number): CameraSample {
  const camera = spec.camera;
  const lensFocal = spec.lens.focal_mm;
  const keys = camera.keys.slice().sort((a, b) => a.t - b.t);
  const t = keyTime(camera, tNorm);
  const distance = camera.distance ?? 1;
  const first = keys[0];
  if (!first || keys.length < 2) {
    const look = first ? ([...first.look] as Vec3) : ([0, 0, 1] as Vec3);
    const pos = first ? ([...first.pos] as Vec3) : ([0, 0, 0] as Vec3);
    return { pos: scaleFrom(look, pos, distance), look, focal: first?.focal_mm ?? lensFocal, roll: first?.roll ?? 0 };
  }
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1].t) i += 1;
  const a = keys[i];
  const b = keys[i + 1];
  const span = b.t - a.t || 1;
  const u = clamp01((t - a.t) / span);
  const p0 = keys[i - 1] ?? a;
  const p3 = keys[i + 2] ?? b;
  const curve = curveOf(camera);
  const look = catmull(p0.look, a.look, b.look, p3.look, u, curve);
  return {
    pos: scaleFrom(look, catmull(p0.pos, a.pos, b.pos, p3.pos, u, curve), distance),
    look,
    focal: lerp(a.focal_mm ?? lensFocal, b.focal_mm ?? lensFocal, u),
    roll: lerp(a.roll ?? 0, b.roll ?? 0, u),
  };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function vlen(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function norm(a: Vec3): Vec3 {
  const l = vlen(a) || 1;
  return scale(a, 1 / l);
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Aim elsewhere. The camera stays put, so the subject leaves the middle of the frame. dx and dy are fractions of the frame, right and up. */
export function panLook(pose: CameraPose, dx: number, dy: number, focal: number, aspect: number): CameraPose {
  const forward = norm(sub(pose.look, pose.pos));
  let right = cross(forward, [0, 1, 0]);
  if (vlen(right) < 1e-4) right = [1, 0, 0];
  right = norm(right);
  const up = norm(cross(right, forward));
  const dist = vlen(sub(pose.look, pose.pos)) || 1;
  const halfW = (dist * 36) / (2 * Math.max(8, focal));
  const halfH = halfW / (aspect || 1);
  return { pos: [...pose.pos], look: add(add(pose.look, scale(right, dx * halfW * 2)), scale(up, dy * halfH * 2)) };
}

/** Swing the camera around its look point. Yaw and pitch are radians. */
export function orbitPose(pose: CameraPose, yaw: number, pitch: number): CameraPose {
  const ox = pose.pos[0] - pose.look[0];
  const oy = pose.pos[1] - pose.look[1];
  const oz = pose.pos[2] - pose.look[2];
  const dist = Math.hypot(ox, oy, oz) || 1;
  const az = Math.atan2(ox, oz) + yaw;
  let elev = Math.atan2(oy, Math.hypot(ox, oz) || 1e-6) + pitch;
  if (elev > 1.48) elev = 1.48;
  if (elev < -1.15) elev = -1.15;
  const nh = Math.cos(elev) * dist;
  const y = Math.max(0.12, pose.look[1] + Math.sin(elev) * dist);
  return {
    look: [...pose.look],
    pos: [pose.look[0] + Math.sin(az) * nh, y, pose.look[2] + Math.cos(az) * nh],
  };
}

/** Walk. Forward follows the gaze along the ground, so looking up does not fly you into the sky. */
export function walkPose(pose: CameraPose, forward: number, strafe: number, lift: number): CameraPose {
  const lookDir = sub(pose.look, pose.pos);
  let fx = lookDir[0];
  let fz = lookDir[2];
  const fh = Math.hypot(fx, fz);
  if (fh < 1e-3) {
    fx = 0;
    fz = 1;
  } else {
    fx /= fh;
    fz /= fh;
  }
  const dx = fx * forward - fz * strafe;
  const dz = fz * forward + fx * strafe;
  const y = Math.min(30, Math.max(0.25, pose.pos[1] + lift));
  const pos: Vec3 = [Math.min(60, Math.max(-60, pose.pos[0] + dx)), y, Math.min(60, Math.max(-60, pose.pos[2] + dz))];
  const dy = y - pose.pos[1];
  return {
    pos,
    look: [pose.look[0] + (pos[0] - pose.pos[0]), pose.look[1] + dy, pose.look[2] + (pos[2] - pose.pos[2])],
  };
}

/** Turn your head. The camera stays where it is. Yaw and pitch are radians. */
export function lookPose(pose: CameraPose, yaw: number, pitch: number): CameraPose {
  const ox = pose.look[0] - pose.pos[0];
  const oy = pose.look[1] - pose.pos[1];
  const oz = pose.look[2] - pose.pos[2];
  const dist = Math.hypot(ox, oy, oz) || 1;
  const x1 = ox * Math.cos(yaw) + oz * Math.sin(yaw);
  const z1 = -ox * Math.sin(yaw) + oz * Math.cos(yaw);
  let elev = Math.atan2(oy, Math.hypot(x1, z1) || 1e-6) + pitch;
  if (elev > 1.35) elev = 1.35;
  if (elev < -1.2) elev = -1.2;
  const az = Math.atan2(x1, z1);
  const nh = Math.cos(elev) * dist;
  return {
    pos: [...pose.pos],
    look: [pose.pos[0] + Math.sin(az) * nh, pose.pos[1] + Math.sin(elev) * dist, pose.pos[2] + Math.cos(az) * nh],
  };
}

/** Slide the camera and its aim together, so the world follows the drag. */
export function truckPose(pose: CameraPose, dx: number, dy: number, focal: number, aspect: number): CameraPose {
  const panned = panLook(pose, -dx, -dy, focal, aspect);
  const shift = sub(panned.look, pose.look);
  return { pos: add(pose.pos, shift), look: panned.look };
}

/** Step closer or farther along the aim. A factor under 1 moves in. */
export function dollyPose(pose: CameraPose, factor: number): CameraPose {
  const off = sub(pose.pos, pose.look);
  const dist = vlen(off) || 1;
  const next = Math.min(80, Math.max(0.35, dist * (factor > 0 ? factor : 1)));
  const pos = add(pose.look, scale(off, next / dist));
  if (pos[1] < 0.2) pos[1] = 0.2;
  return { pos, look: [...pose.look] };
}

/** Pull the camera in or out without changing where it looks. 1 keeps the written move. */
function scaleFrom(look: Vec3, pos: Vec3, distance: number): Vec3 {
  return [
    look[0] + (pos[0] - look[0]) * distance,
    look[1] + (pos[1] - look[1]) * distance,
    look[2] + (pos[2] - look[2]) * distance,
  ];
}

/** Stable id of the camera at frame 0. Same spec, same text. */
export function frameZeroDigest(spec: PrevizSpec): string {
  const s = sampleCamera(spec, 0);
  const line = [...s.pos, ...s.look, s.focal, s.roll, Number(spec.seed)].map((n) => Number(n).toFixed(6)).join(",");
  let h = 0x811c9dc5;
  for (let i = 0; i < line.length; i++) {
    h ^= line.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}
