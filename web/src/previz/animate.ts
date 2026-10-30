// Procedural motion. `t` is 0–1 across the shot; nothing here reads the clock.
import type { Object3D } from "three";

export interface Limbs {
  leftLeg: Object3D;
  rightLeg: Object3D;
  leftArm: Object3D;
  rightArm: Object3D;
  bob: Object3D;
}

export function walk(limbs: Limbs, t: number, speed: number, height: number, style: "walk" | "jog" | "run" = "walk"): void {
  const rate = style === "run" ? 14 : style === "jog" ? 11 : 8;
  const leg = style === "run" ? 1.05 : style === "jog" ? 0.8 : 0.55;
  const arm = style === "run" ? 1 : style === "jog" ? 0.7 : 0.45;
  const phase = t * Math.max(0.05, speed) * Math.PI * rate;
  const swing = Math.sin(phase);
  limbs.leftLeg.rotation.x = swing * leg;
  limbs.rightLeg.rotation.x = -swing * leg;
  limbs.leftArm.rotation.x = -swing * arm;
  limbs.rightArm.rotation.x = swing * arm;
  limbs.bob.position.y = Math.abs(Math.sin(phase * 2)) * height * (style === "run" ? 0.03 : style === "jog" ? 0.02 : 0.012);
  limbs.bob.rotation.x = style === "run" ? 0.22 : style === "jog" ? 0.1 : 0.04;
}

/** Wheel pivots spin on their axle (local X). */
export function spinWheels(pivots: Object3D[], t: number, speed: number): void {
  const spin = t * Math.max(0.05, speed) * Math.PI * 28;
  for (const p of pivots) p.rotation.x = spin;
}

export function spinY(object: Object3D, t: number, speed: number): void {
  object.rotation.y = t * Math.max(0.05, speed) * Math.PI * 2;
}

/** Metres of travel shared by a subject and, for a tracking shot, the camera. */
export function travelMetres(t: number, speed: number, metres: number): number {
  return t * Math.max(0.05, speed) * metres;
}
