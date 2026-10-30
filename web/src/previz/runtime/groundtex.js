// Ground texture: a soft, irregular, low-contrast mottle on the ground, roads, paths, sand and snow, so a moving camera
// or a driving car reads against the floor (a flat grey floor gives the eye nothing to track). Not a grid: a seeded,
// tileable value noise (several octaves) sampled in world space at three unrelated scales and rotations, so it never
// lines up with the axes and never visibly repeats. Multiplies the material colour about 1 +/- 8 percent, so the
// palette, the editor's look and the depth guide stay as they were. Needs no UVs (the kit's ribbons and boxes have none).
import * as THREE from 'three';

export const GROUND_NOISE_SIZE = 256;
const OCTAVES = [[4, 0.5], [8, 0.25], [16, 0.15], [32, 0.1]]; // lattice cells across the tile, weight

function mulberry(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

/** The noise tile as numbers in [0, 1], mean 0.5 (row-major, size x size). Seeded, so every build and test sees the same. */
export function groundNoise(size = GROUND_NOISE_SIZE, seed = 25) {
  const rnd = mulberry(seed);
  const out = new Float32Array(size * size);
  for (const [cells, w] of OCTAVES) {
    const lat = Float32Array.from({ length: cells * cells }, rnd);
    const at = (i, j) => lat[((j % cells) * cells) + (i % cells)];
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const fx = x / size * cells, fy = y / size * cells, ix = Math.floor(fx), iy = Math.floor(fy);
      let tx = fx - ix, ty = fy - iy; tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
      const a = at(ix, iy) + (at(ix + 1, iy) - at(ix, iy)) * tx, b = at(ix, iy + 1) + (at(ix + 1, iy + 1) - at(ix, iy + 1)) * tx;
      out[y * size + x] += w * (a + (b - a) * ty);
    }
  }
  let mean = 0; for (const v of out) mean += v; mean /= out.length;
  let sd = 0; for (const v of out) sd += (v - mean) ** 2; sd = Math.sqrt(sd / out.length) || 1;
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, Math.max(0, 0.5 + (out[i] - mean) / sd * 0.16));
  return out;
}

let TEX = null;
/** The shared noise tile as a repeat-wrapped, mipmapped texture (data, not colour). */
export function groundTexture() {
  if (!TEX) {
    const n = groundNoise(), data = new Uint8Array(n.length * 4);
    n.forEach((v, i) => { const b = Math.round(v * 255); data.set([b, b, b, 255], i * 4); });
    TEX = new THREE.DataTexture(data, GROUND_NOISE_SIZE, GROUND_NOISE_SIZE, THREE.RGBAFormat);
    TEX.name = 'pz_ground_noise';
    TEX.wrapS = TEX.wrapT = THREE.RepeatWrapping;
    TEX.magFilter = THREE.LinearFilter;
    TEX.minFilter = THREE.LinearMipmapLinearFilter;
    TEX.generateMipmaps = true;
    TEX.colorSpace = THREE.NoColorSpace;
    TEX.needsUpdate = true;
  }
  return TEX;
}

/** World metres per tile for the three samples, and how much each moves the colour. */
export const GROUND_SCALES = [[23.0, 0.42], [6.1, 0.34], [1.7, 0.26]];

/** Give a lit material the ground mottle (in place; returns it). Marked with userData.groundNoise for tests and tools. */
export function texturedGround(mat) {
  if (!mat || mat.userData.groundNoise) return mat;
  const tex = groundTexture();
  mat.userData.groundNoise = tex;
  const [[s1, w1], [s2, w2], [s3, w3]] = GROUND_SCALES;
  const f = (x) => x.toFixed(3);
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.pzNoise = { value: tex };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vPzWorld;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n  vec4 pzP = vec4(transformed, 1.0);\n  #ifdef USE_INSTANCING\n  pzP = instanceMatrix * pzP;\n  #endif\n  vPzWorld = (modelMatrix * pzP).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vPzWorld;\nuniform sampler2D pzNoise;\n'
        + 'float pzGround(vec2 p) {\n'
        + `  float a = texture2D(pzNoise, p / ${f(s1)}).r;\n`
        + `  float b = texture2D(pzNoise, mat2(0.8, -0.6, 0.6, 0.8) * p / ${f(s2)} + 0.31).r;\n`
        + `  float c = texture2D(pzNoise, mat2(0.28, 0.96, -0.96, 0.28) * p / ${f(s3)} + 0.73).r;\n`
        + `  return 1.0 + ${f(w1)} * (a - 0.5) + ${f(w2)} * (b - 0.5) + ${f(w3)} * (c - 0.5);\n}`)
      .replace('#include <map_fragment>', '#include <map_fragment>\n  diffuseColor.rgb *= pzGround(vPzWorld.xz);');
  };
  mat.customProgramCacheKey = () => 'pz_ground_v1';
  return mat;
}

/** Set materials that get the mottle: everything you stand, drive or walk on. */
export const GROUND_MATERIALS = ['ground', 'path', 'road', 'sand', 'sandLee', 'snow'];
