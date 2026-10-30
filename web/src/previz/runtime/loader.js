// Library loader: GLTFLoader + MeshoptDecoder, one cached template per model, cheap clones, preload.
// Works with three r170 (no other dependency). Models live in <baseUrl>models/<id>.glb.
// Kits are assembled from layout JSON. Baked scene GLBs are not loaded.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { isTreeId, pineTemplate } from './pine.js';

/** Person GLB tiers: hero (full detail, close-ups), lite (default subject), crowd (extras). */
export const PERSON_TIERS = { hero: 'person', lite: 'person__lite', crowd: 'person__crowd' };

/** Startup preload set (person, car, tree, plus the bench and dog the default park shots use). Ground is procedural. */
export const CORE_MODELS = ['person__lite', 'car', 'tree', 'bench', 'tree__far', 'car__exterior'];

/**
 * createLibrary({ baseUrl, manager })
 * @param {{ baseUrl?: string, manager?: import('three').LoadingManager }} [options]
 */
export function createLibrary(options = {}) {
  const baseUrl = options.baseUrl ?? './assets/';
  const loader = new GLTFLoader(options.manager);
  loader.setMeshoptDecoder(MeshoptDecoder);
  const pending = new Map();   // id -> Promise<Object3D>
  const ready = new Map();     // id -> Object3D template
  const timings = {};          // id -> ms to fetch + decode

  const url = (id) => baseUrl + 'models/' + id + '.glb';

  function load(id) {
    if (String(id).startsWith('scene:')) return Promise.reject(new Error('baked scene GLBs are not loaded'));
    let p = pending.get(id);
    if (!p) {
      const t0 = performance.now();
      p = loader.loadAsync(url(id)).then((gltf) => {
        const root = prepareTemplate(gltf.scene, id);
        ready.set(id, root);
        timings[id] = Math.round(performance.now() - t0);
        return root;
      });
      p.catch(() => pending.delete(id)); // allow a retry after a failed fetch
      pending.set(id, p);
    }
    return p;
  }
  /** Synchronous clone of a loaded model (geometry and materials are shared with the template). */
  function instance(id) {
    const t = ready.get(id);
    if (!t) throw new Error('Model not loaded yet: ' + id + ' (await lib.load(id) or lib.preload([...]) first)');
    const c = t.clone(true);
    c.userData.libraryId = id;
    return c;
  }
  async function create(id) { await load(id); return instance(id); }
  function preload(ids) { return Promise.all([...new Set(ids)].map(load)); }
  return {
    url, load, create, instance, preload,
    isLoaded: (id) => ready.has(id),
    template: (id) => ready.get(id) || null,
    loadedIds: () => [...ready.keys()],
    timings, gltfLoader: loader,
  };
}

/** The dark trim's colour after loading, linear (a charcoal, about #5E5E5E on screen before light). */
const TRIM_LINEAR = 0.12;

// One-time fix-ups on a freshly loaded model: shadows (glTF has no shadow flags), tinted glass reflections,
// the asset root unwrapped from the glTF scene, and the tree as a pine.
export function prepareTemplate(scene, id) {
  const root = scene.children.length === 1 ? scene.children[0] : scene;
  if (root !== scene) scene.remove(root);
  root.traverse((o) => {
    if (!o.isMesh) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const clear = mats.some((m) => m.transparent);
    const owner = o.name || (o.parent && o.parent.name) || '';
    o.castShadow = !clear && !/^(stripe|face|seam|line_)/.test(owner);
    o.receiveShadow = !clear;
    for (const m of mats) {
      if (/glass/.test(m.name)) m.envMapIntensity = 0.15; // the library's low-reflection tint
      // The library's dark trim (t2_detail: grilles, book covers, cups; t2_rim: bumpers, rims) is 1 to 4 percent, so after
      // tone mapping it read as flat black blocks, worst on the small scene tiles. A charcoal keeps it dark trim with shape.
      if ((m.name === 't2_detail' || m.name === 't2_rim') && m.color) m.color.setScalar(TRIM_LINEAR);
    }
  });
  root.userData.libraryId = id;
  root.updateMatrixWorld(true);
  if (isTreeId(id)) pineTemplate(root); // a pine, not a ball on a pole
  return root;
}
