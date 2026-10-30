// World look (library v1.2): warm grey sky dome with a soft horizon, matching fog, hemisphere fill, one warm key light
// with soft shadows, a faint room environment for highlights. Neutral tone mapping keeps the people yellow (#F4FF75) on hue.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export const WORLD = {
  skyZenith: 0xA9A59E, skyHorizon: 0xD8D4CC, fog: 0xCFCBC3, fogNear: 22, fogFar: 75,
  exposure: 1.0, hemiSky: 0xF3EFE8, hemiGround: 0x8E8B85, hemiIntensity: 1.25,
  keyColor: 0xFFF4E6, keyIntensity: 2.1, keyDir: [-0.55, 1.0, 0.45], envIntensity: 0.35,
};

/** Time of day and weather looks for the spec's world block (rule tables, still in the grey previs palette). */
export const LOOKS = {
  time: {
    day: {},
    golden: { skyZenith: 0x9C958C, skyHorizon: 0xE6CBA6, fog: 0xD9C4A6, keyColor: 0xFFD3A1, keyIntensity: 2.3, keyDir: [-0.9, 0.32, 0.35], hemiIntensity: 1.0 },
    dusk: { skyZenith: 0x5F6170, skyHorizon: 0xB49E93, fog: 0x9E918C, keyColor: 0xFFC59A, keyIntensity: 1.1, keyDir: [-0.95, 0.22, 0.3], hemiIntensity: 0.75, exposure: 0.95 },
    night: { skyZenith: 0x1E222B, skyHorizon: 0x3A3F4A, fog: 0x30343D, keyColor: 0xBCCBFF, keyIntensity: 0.55, keyDir: [0.4, 1.0, -0.3], hemiSky: 0x8E9AB8, hemiGround: 0x2E3038, hemiIntensity: 0.55, envIntensity: 0.15, exposure: 1.1 },
  },
  // Previz never adds weather: rain, fog, snow, storm and overcast still parse and add nothing (no fog, haze or sky
  // change). Weather comes later, in previz-to-final, from a reference image.
  weather: { clear: {}, overcast: {}, rain: {}, fog: {}, snow: {} },
};

/** Renderer settings for the look. quality 'high' = 4096 VSM shadows (stills), 'fast' = 2048 (live preview). */
export function setupRenderer(renderer, { quality = 'fast' } = {}) {
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = WORLD.exposure;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  renderer.shadowMap.autoUpdate = false;
  renderer.userData = Object.assign(renderer.userData || {}, { previzQuality: quality });
}

const envCache = new WeakMap();
function environmentFor(renderer) {
  const cached = envCache.get(renderer);
  if (cached) return cached;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const tex = pmrem.fromScene(room, 0.04).texture;
  room.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    const m = o.material;
    if (Array.isArray(m)) m.forEach((x) => x.dispose());
    else if (m) m.dispose();
  });
  pmrem.dispose();
  tex.userData.shared = true;
  envCache.set(renderer, tex);
  return tex;
}
/** Drop the cached environment when its renderer is going away. */
export function disposeEnvironment(renderer) {
  const tex = envCache.get(renderer);
  if (!tex) return;
  tex.dispose();
  envCache.delete(renderer);
}

function skyDome(radius) {
  const mat = new THREE.ShaderMaterial({
    name: 'world_sky', side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      zen: { value: new THREE.Color(WORLD.skyZenith) },
      hor: { value: new THREE.Color(WORLD.skyHorizon) },
      fogCol: { value: new THREE.Color(WORLD.fog) },
    },
    vertexShader: 'varying vec3 vW; void main(){ vW = (modelMatrix * vec4(position, 1.0)).xyz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 zen; uniform vec3 hor; uniform vec3 fogCol; varying vec3 vW; void main(){ vec3 vP = normalize(vW - cameraPosition); float h = clamp(vP.y, 0.0, 1.0); float k = pow(h, 0.55); vec3 sky = mix(hor, zen, k); float band = smoothstep(0.16, 0.0, vP.y); gl_FragColor = vec4(mix(sky, fogCol, band), 1.0); \n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n }',
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 96, 48), mat);
  m.name = 'world_sky'; m.renderOrder = -1; m.frustumCulled = false;
  return m;
}

/**
 * createWorld(renderer, scene, { quality, environment = true }) -> { key, hemi, sky, apply(kitWorld, specWorld) }
 */
export function createWorld(renderer, scene, options = {}) {
  const quality = options.quality || 'fast';
  setupRenderer(renderer, { quality });
  scene.background = new THREE.Color(WORLD.skyHorizon);
  scene.fog = new THREE.Fog(WORLD.fog, WORLD.fogNear, WORLD.fogFar);
  const sky = skyDome(90); scene.add(sky);
  if (options.environment !== false) {
    scene.environment = environmentFor(renderer);
    scene.environmentIntensity = WORLD.envIntensity;
  }
  const hemi = new THREE.HemisphereLight(WORLD.hemiSky, WORLD.hemiGround, WORLD.hemiIntensity); hemi.name = 'world_ambient';
  const key = new THREE.DirectionalLight(WORLD.keyColor, WORLD.keyIntensity); key.name = 'world_key';
  key.castShadow = true;
  const size = quality === 'high' ? 4096 : 2048;
  key.shadow.mapSize.set(size, size);
  key.shadow.radius = quality === 'high' ? 9 : 6; key.shadow.blurSamples = quality === 'high' ? 24 : 10;
  key.shadow.bias = -0.0002; key.shadow.normalBias = 0.02;
  scene.add(hemi, key, key.target);
  /** apply(kitWorld, specWorld): kit layout's world block (fogNear, fogFar, shadowExtent, shadowCenter, keyDist, skyScale), then the shot's time-of-day look (weather is ignored). */
  function apply(w = {}, look = {}) {
    const L = Object.assign({}, WORLD, LOOKS.time[look && look.time] || {});
    // the kit's own distance falloff only: no look ever pulls the fog in
    const fogNear = w.fogNear ?? WORLD.fogNear;
    const fogFar = w.fogFar ?? WORLD.fogFar;
    scene.fog.color.setHex(L.fog); scene.fog.near = fogNear; scene.fog.far = fogFar;
    scene.background.setHex(L.skyHorizon);
    sky.material.uniforms.zen.value.setHex(L.skyZenith); sky.material.uniforms.hor.value.setHex(L.skyHorizon);
    if (sky.material.uniforms.fogCol) sky.material.uniforms.fogCol.value.setHex(L.fog);
    hemi.color.setHex(L.hemiSky); hemi.groundColor.setHex(L.hemiGround); hemi.intensity = L.hemiIntensity;
    key.color.setHex(L.keyColor); key.intensity = L.keyIntensity;
    key.shadow.radius = (quality === 'high' ? 9 : 6) * (L.shadowSoft ? 2.2 : 1);
    if (scene.environment) scene.environmentIntensity = L.envIntensity;
    renderer.toneMappingExposure = L.exposure;
    sky.scale.setScalar(w.skyScale || 1.5);
    const ext = Math.max(w.shadowExtent || 18, 140), c = new THREE.Vector3(...(w.shadowCenter || [0, 0, 0])), dist = w.keyDist || 40;
    const dir = new THREE.Vector3(...L.keyDir).normalize();
    key.position.copy(c).addScaledVector(dir, dist); key.target.position.copy(c);
    Object.assign(key.shadow.camera, { left: -ext, right: ext, top: ext, bottom: -ext, near: 1, far: dist * 2.4 });
    key.shadow.camera.updateProjectionMatrix();
    key.target.updateMatrixWorld(); key.updateMatrixWorld();
    key.shadow.needsUpdate = true;
    renderer.shadowMap.needsUpdate = true;
  }
  apply();
  return { key, hemi, sky, apply };
}
