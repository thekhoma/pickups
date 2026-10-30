// The library tree is a ball on a pole: in a layout sent to the render it reads as a grey sphere (over a lake, a ring of
// balloons), never as a tree. The tree templates (tree, tree__far) are rebuilt once, on load, as a pine: a tapered trunk
// and three stacked, faceted cones, in the model's own height, meshes and materials, so instancing, shadows, the
// editor's look and every caller stay as they were.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/** Tiers bottom to top, as fractions of the tree's height: [bottom, top, base radius]. */
export const PINE_TIERS = [[0.34, 0.72, 0.22], [0.52, 0.88, 0.17], [0.68, 1, 0.11]]; // the lowest tier clears a walker
const SEGMENTS = 9;

/** Is this library id a tree template? */
export const isTreeId = (id) => String(id || '').split('__')[0] === 'tree';

/** The pine's canopy, in tree space (ground at y 0, height h), faceted (flat normals). */
export function pineCanopy(h) {
  const parts = PINE_TIERS.map(([b, t, r], k) => {
    const g = new THREE.ConeGeometry(r * h, (t - b) * h, SEGMENTS, 1, false);
    g.rotateY((k * Math.PI) / SEGMENTS); // the tiers' facets staggered
    g.translate(0, ((b + t) / 2) * h, 0);
    return g.toNonIndexed();
  });
  const geo = mergeGeometries(parts);
  geo.computeVertexNormals();
  return geo;
}
/** The pine's trunk, in tree space: base on the ground, into the lowest tier. */
export function pineTrunk(h) {
  const g = new THREE.CylinderGeometry(0.018 * h, 0.032 * h, 0.5 * h, 7, 1, false);
  g.translate(0, 0.25 * h, 0);
  const geo = g.toNonIndexed();
  geo.computeVertexNormals();
  return geo;
}

/**
 * Rebuild a loaded tree template as a pine, in place. The lowest mesh is the trunk, the rest the canopy (one mesh: both,
 * merged, in its material). Returns the template.
 */
export function pineTemplate(root) {
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const meshes = [];
  root.traverse((o) => { if (o.isMesh) meshes.push(o); });
  if (!meshes.length) return root;
  const box = new THREE.Box3();
  const low = (m) => { m.geometry.computeBoundingBox(); return box.copy(m.geometry.boundingBox).applyMatrix4(new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld)).min.y; };
  const whole = new THREE.Box3();
  for (const m of meshes) { m.geometry.computeBoundingBox(); whole.union(box.copy(m.geometry.boundingBox).applyMatrix4(new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld))); }
  const h = Math.max(0.5, whole.max.y);
  meshes.sort((a, b) => low(a) - low(b));
  meshes.forEach((m, i) => {
    const geo = meshes.length === 1 ? mergeGeometries([pineTrunk(h), pineCanopy(h)]) : i === 0 ? pineTrunk(h) : i === 1 ? pineCanopy(h) : null;
    const rel = new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld).invert();
    if (!geo) { m.visible = false; return; }
    geo.applyMatrix4(rel);
    geo.computeBoundingBox(); geo.computeBoundingSphere();
    m.geometry = geo;
    m.castShadow = true; m.receiveShadow = true;
  });
  root.userData.proxy = 'pine';
  return root;
}
