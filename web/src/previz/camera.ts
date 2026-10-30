// Puts the sampled pose on a Three.js camera. filmGauge 36 mm, then the focal length, so a 24 mm lens is actually wide.
import * as THREE from "three";
import { type CameraSample, sampleCamera as sampleRaw } from "./cameraMath";
import type { PrevizSpec } from "./spec";

export { frameZeroDigest, sampleCamera } from "./cameraMath";

/** Steeper than this (|view dir y|) the world up is too close to the view to aim by: a top-down or drone shot. */
const STEEP = 0.9;
const STEPS = 240;
const upTables = new WeakMap<PrevizSpec, THREE.Vector3[]>();

/**
 * The camera's up through a shot. Level views use the world up, as always. A view steeper than STEEP carries the up it
 * had a moment before (parallel transport along the keyed path), so a drone passing over the car keeps the top of frame
 * where it was instead of spinning the picture 180 degrees as the aim crosses straight down.
 */
export function steadyUp(spec: PrevizSpec, t: number): THREE.Vector3 {
  let table = upTables.get(spec);
  if (!table) {
    table = [];
    let prev = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i <= STEPS; i++) {
      const s = sampleRaw(spec, i / STEPS);
      const dir = new THREE.Vector3(s.look[0] - s.pos[0], s.look[1] - s.pos[1], s.look[2] - s.pos[2]);
      if (dir.lengthSq() < 1e-10) { table.push(prev.clone()); continue; }
      dir.normalize();
      let up = new THREE.Vector3(0, 1, 0);
      if (Math.abs(dir.y) >= STEEP) {
        const carried = prev.clone().addScaledVector(dir, -prev.dot(dir));
        if (carried.lengthSq() > 1e-6) up = carried.normalize();
      }
      table.push(up);
      prev = up;
    }
    upTables.set(spec, table);
  }
  const i = Math.max(0, Math.min(STEPS, Math.round(t * STEPS)));
  return table[i];
}

export function applyCamera(cam: THREE.PerspectiveCamera, sample: CameraSample, aspect: number, near: number, far: number, steady?: THREE.Vector3): void {
  cam.filmGauge = 36;
  cam.setFocalLength(sample.focal);
  cam.aspect = aspect;
  cam.near = near;
  cam.far = far;
  cam.updateProjectionMatrix();
  cam.position.set(sample.pos[0], sample.pos[1], sample.pos[2]);
  cam.up.set(0, 1, 0);
  if (steady) {
    const dir = new THREE.Vector3(sample.look[0] - sample.pos[0], sample.look[1] - sample.pos[1], sample.look[2] - sample.pos[2]).normalize();
    if (Math.abs(dir.y) >= STEEP) {
      const up = steady.clone().addScaledVector(dir, -steady.dot(dir));
      if (up.lengthSq() > 1e-6) cam.up.copy(up.normalize());
    }
  }
  cam.lookAt(sample.look[0], sample.look[1], sample.look[2]);
  cam.rotateZ((sample.roll * Math.PI) / 180);
  const frame = sample.frame;
  if (frame && (frame[0] || frame[1])) {
    const halfY = (cam.fov * Math.PI) / 360;
    const halfX = Math.atan(Math.tan(halfY) * aspect);
    cam.rotateY(-frame[0] * halfX);
    cam.rotateX(-frame[1] * halfY);
  }
}
