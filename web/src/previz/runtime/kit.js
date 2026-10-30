// Kit loader: a JSON layout (data/kits/<id>.json) -> a THREE.Group built from the library GLBs plus simple procedural
// set geometry (ground, paths, roads, water, dunes, boxes). Repeated dressing (trees, rocks, posts, buildings, houses)
// becomes one InstancedMesh per sub-mesh, so a 150-tree forest is a handful of draw calls.
// Layout schema: see data/kits/*.json and README.md (items: { id, asset, pos, rot, scale, options, pose, props, rider,
// seat, inVehicle, onTop, upright }, links: leash, cameras: { hero, start, end, ... }).
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { setPose, setPersonScale, setAnimalPose, animalKinds } from './poser.js';
import { attachProp, attachActionProp, placePair, seatOn, attachRider, seatInVehicle, placeOnTop, makeLeash, VEHICLES } from './attach.js';
import { PERSON_TIERS } from './loader.js';
import { texturedGround, GROUND_MATERIALS } from './groundtex.js';

const DEG = Math.PI / 180;
export const ACTOR_ASSETS = ['person', 'dog', 'cat'];
export const INSTANCE_ASSETS = ['tree', 'rock', 'post', 'building', 'house', 'fountain'];

export const SET_COLORS = { ground: 0x8A8884, path: 0xA3A19C, set: 0xB8B6B1, setDark: 0x6E6C68, post: 0x4A4845, road: 0x787672, sand: 0xA8A398, water: 0x6E8A90, waterLine: 0x637E84,
  sandLee: 0x878275, sandCrest: 0xC6C1B5, waterDeep: 0x4E6A72, foam: 0x5E7980, snow: 0xD6D4CF, exterior: 0x0B0E14 }; // library palette tier3 (v1.3). exterior: the dark outside a lit room's window at night. Water is cooler than the sand so a beach reads as water.
// The water's edges and lines (foam, waterLine) are a soft tone a little darker than the water, never light: a light shore
// ring or swash came back from the render as a white line on the water.
const ROUGH = { ground: 1, path: 0.95, set: 0.85, setDark: 0.85, post: 0.8, road: 0.95, sand: 1, water: 0.32, waterLine: 0.5, sandLee: 1, sandCrest: 1, waterDeep: 0.4, foam: 0.7, snow: 0.9, exterior: 1 };
let MATS = null;
export function setMaterials() {
  if (!MATS) {
    MATS = {};
    for (const [k, c] of Object.entries(SET_COLORS)) MATS[k] = new THREE.MeshStandardMaterial({ name: 't3_' + k, color: c, roughness: ROUGH[k], metalness: 0 });
    // a practical: a pendant shade that glows warm by itself (diner lamps over the booth and the counter)
    for (const k of GROUND_MATERIALS) texturedGround(MATS[k]); // soft mottle so motion reads (runtime/groundtex.js)
    MATS.lamp = new THREE.MeshStandardMaterial({ name: 't3_lamp', color: 0xEADDBE, emissive: 0xFFD58A, emissiveIntensity: 0.9, roughness: 0.6, metalness: 0 });
  }
  return MATS;
}

/** Assets that have a display-only far LOD (<id>__far.glb: merged meshes, about 30 percent of the triangles). */
export const FAR_ASSETS = ['tree', 'rock', 'post', 'building', 'house', 'fountain', 'bench', ...VEHICLES];

/** Model id for a layout item (handles the house storeys variant, the vehicle exterior LOD and the far LOD). */
export function modelIdFor(it, personTier = 'lite', vehicleLOD = false, far = false) {
  if (it.asset === 'person') return isCrowdItem(it) ? PERSON_TIERS.crowd : (PERSON_TIERS[personTier] || PERSON_TIERS.lite);
  let id = it.asset;
  if (vehicleLOD && VEHICLES.includes(it.asset) && !it.keepInterior) id = it.asset + '__exterior';
  if (it.asset === 'house' && it.options && it.options.storeys === 1) id = 'house__1storey';
  if (far && FAR_ASSETS.includes(it.asset) && (!VEHICLES.includes(it.asset) || id.endsWith('__exterior'))) id += '__far';
  return id;
}

/** Ids of layout items that other items ride, sit on, sit in, stand on or link to (never instanced or swapped for a LOD). */
export function referencedIds(layout) {
  const referenced = new Set();
  for (const it of layout.items || []) for (const k of ['rider', 'onTop']) if (it[k]) referenced.add(typeof it[k] === 'string' ? it[k] : it[k].of);
  for (const it of layout.items || []) { if (it.seat) referenced.add(it.seat.on); if (it.inVehicle) referenced.add(it.inVehicle.of); }
  for (const l of layout.links || []) { referenced.add(l.from[0]); referenced.add(l.to[0]); }
  return referenced;
}

function resolveOptions(options) {
  const o = Object.assign({ skipActors: false, instancing: true, personTier: 'lite', maxCrowd: 6 }, options);
  if (o.vehicleLOD == null) o.vehicleLOD = o.skipActors; // empty parked vehicles in the set use the exterior-only GLB
  if (o.farDist == null) o.farDist = o.skipActors ? 12 : 0; // metres from the stage; 0 turns the far LOD off
  return o;
}

/** Is this item far enough from the stage to use the far LOD? */
function isFar(it, o, stage, referenced) {
  if (!o.farDist || !it.pos || (it.id && referenced.has(it.id)) || !FAR_ASSETS.includes(it.asset)) return false;
  const x = it.pos[0], z = it.pos[it.pos.length - 1];
  return Math.hypot(x - stage.pos[0], z - stage.pos[1]) > o.farDist;
}

/** Every model a layout needs (for preloading). Takes the same options as buildKit. */
/** The crowd extras kept (maxCrowd nearest the stage); the rest are left out to hold the triangle budget. */
function keptCrowd(layout, o, stage) {
  const crowd = (layout.items || []).filter(isCrowdItem);
  if (crowd.length <= o.maxCrowd) return new Set(crowd);
  const d = (it) => { const p = it.pos || [stage.pos[0], stage.pos[1]]; return Math.hypot(p[0] - stage.pos[0], p[p.length - 1] - stage.pos[1]); };
  return new Set(crowd.slice().sort((a, b) => d(a) - d(b)).slice(0, o.maxCrowd));
}

export function kitModelIds(layout, options = {}) {
  const o = resolveOptions(options), ids = new Set(), referenced = referencedIds(layout), { stage } = kitStage(layout);
  const crowd = keptCrowd(layout, o, stage);
  for (const it of layout.items || []) {
    if (o.skipActors && isActorItem(it, layout)) continue;
    if (isCrowdItem(it) && !crowd.has(it)) continue;
    ids.add(it.pair ? (PERSON_TIERS[o.personTier] || PERSON_TIERS.lite) : modelIdFor(it, o.personTier, o.vehicleLOD, isFar(it, o, stage, referenced)));
    if (it.props) for (const a of Object.values(it.props)) ids.add(a);
  }
  return [...ids];
}
const riderTargets = (layout) => new Set((layout.items || []).filter((i) => i.rider).map((i) => i.rider));
/** v1.4 subjectSwap: an item yields when the subject's pose or one of its props (array or slot map) equals the swap id. */
export function swapMatches(swap, subject) {
  if (!subject) return false;
  const props = Array.isArray(subject.props) ? subject.props : Object.values(subject.props || {});
  return subject.pose === swap || props.includes(swap);
}
/** Background extras (options.preset 'crowd', library v1.3) stay in the set as dressing; everyone else is an actor. */
export function isCrowdItem(it) { return it.asset === 'person' && !!(it.options && it.options.preset === 'crowd'); }
function isActorItem(it, layout) {
  if (isCrowdItem(it)) return false;
  if (ACTOR_ASSETS.includes(it.asset) || it.rider || it.inVehicle || it.seat || it.pair) return true; // v1.4: pair items are two people
  // a vehicle that carries a seated person in the layout is part of the hero composition, keep it as dressing anyway
  void layout; return false;
}

/** The kit's hero spot (where the layout's main person stands) and a spot for a vehicle. */
export function kitStage(layout) {
  const items = layout.items || [];
  const pref = ['adult', 'walker', 'visitor', 'hero', 'man', 'woman', 'person', 'driver', 'runner', 'sitter'];
  let p = null;
  for (const id of pref) { p = items.find((i) => i.id === id && i.asset === 'person' && i.pos && !isCrowdItem(i)); if (p) break; }
  p = p || items.find((i) => i.pair && i.pos) || items.find((i) => i.asset === 'person' && i.pos && !isCrowdItem(i)) || null; // v1.4: a kit pair spot (gym trainer) is open floor
  const v = items.find((i) => VEHICLES.includes(i.asset) && i.pos) || null;
  const pos = p ? [p.pos[0], p.pos[p.pos.length - 1]] : (v ? [v.pos[0] + 2.5, v.pos[v.pos.length - 1]] : [0, 0]);
  const stage = { pos, rot: p ? (p.rot || 0) : 0, y: p && p.pos.length === 3 ? p.pos[1] : 0 }; // y: a raised stage or platform
  const vehicleStage = v ? { pos: [v.pos[0], v.pos[v.pos.length - 1]], rot: v.rot || 0, kitItem: v.id || null } : { pos: [pos[0] + 3.2, pos[1] - 1.2], rot: (stage.rot || 0) + 90, kitItem: null };
  return { stage, vehicleStage };
}

// ---------- procedural set geometry ----------
function ribbon(points, width, height, samples = 160) {
  const curve = new THREE.CatmullRomCurve3(points.map(([x, z]) => new THREE.Vector3(x, 0, z)), false, 'centripetal');
  const prof = [[-0.5, 0], [-0.48, 1], [0.48, 1], [0.5, 0]], up = new THREE.Vector3(0, 1, 0), pos = [], idx = [];
  const side = new THREE.Vector3();
  for (let i = 0; i <= samples; i++) {
    const t = i / samples, p = curve.getPointAt(t), tan = curve.getTangentAt(t);
    side.crossVectors(up, tan).normalize();
    for (const [a, hf] of prof) pos.push(p.x + side.x * a * width, hf * height, p.z + side.z * a * width);
  }
  for (let i = 0; i < samples; i++) for (let j = 0; j < 3; j++) { const a = i * 4 + j, b = a + 1, c = a + 4, d = c + 1; idx.push(a, c, b, b, c, d); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return { geometry: g, curve };
}
function meshOf(name, geom, mat, parent, cast = false) { const m = new THREE.Mesh(geom, mat); m.name = name; m.castShadow = cast; m.receiveShadow = true; parent.add(m); return m; }

function buildGeometry(g, root, m) {
  if (g.type === 'ground') {
    const d = meshOf(g.name || 'geo_ground', new THREE.CircleGeometry(Math.max(g.radius || 80, 560), 128), m[g.material && g.material !== 'ground' ? g.material : 'ground'], root);
    d.rotation.x = -Math.PI / 2; if (g.pos) d.position.set(g.pos[0], 0, g.pos[1]);
    if (g.y) d.position.y = g.y; // raised or sunken ground (rooftop street level)
  } else if (g.type === 'path' || g.type === 'road') {
    const road = g.type === 'road', h = g.height || (road ? 0.012 : 0.02), w = g.width || (road ? 7 : 1.8);
    const { geometry, curve } = ribbon(g.points, w, h);
    const p = meshOf(g.name || g.type, geometry, m[g.material || (road ? 'road' : 'path')], root);
    if (g.y) p.position.y = g.y;
    if (road && g.centerLine !== false) {
      const len = curve.getLength(), dash = 2.4, gap = 3.2, mats = [];
      for (let s = gap / 2; s + dash < len; s += dash + gap) {
        const a = curve.getPointAt(s / len), b = curve.getPointAt((s + dash) / len);
        mats.push(new THREE.Matrix4().compose(a.clone().lerp(b, 0.5).setY(h + 0.004 + (g.y || 0)), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(b.x - a.x, b.z - a.z)), new THREE.Vector3(1, 1, a.distanceTo(b))));
      }
      if (mats.length) { const im = new THREE.InstancedMesh(new THREE.BoxGeometry(0.14, 0.006, 1), m.path, mats.length); mats.forEach((x, i) => im.setMatrixAt(i, x)); im.name = 'geo_dash'; im.receiveShadow = true; im.computeBoundingSphere(); root.add(im); }
    }
  } else if (g.type === 'water') {
    const y = g.y || 0.014;
    if (g.shape === 'path') {
      const { geometry, curve } = ribbon(g.points, g.width, 0);
      const p = meshOf(g.name || 'water', geometry, m.water, root); p.position.y = y;
      if (g.lines) {
        const mats = [];
        for (const off of [-0.22, 0.18]) for (let i = 0; i < g.lines; i++) {
          const t0 = (i + 0.15) / g.lines, a = curve.getPointAt(t0), b = curve.getPointAt(Math.min(1, t0 + 0.35 / g.lines));
          const side = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), curve.getTangentAt(t0)).normalize().multiplyScalar(off * g.width);
          mats.push(new THREE.Matrix4().compose(a.clone().lerp(b, 0.5).add(side).setY(y + 0.004), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(b.x - a.x, b.z - a.z)), new THREE.Vector3(1, 1, a.distanceTo(b))));
        }
        addInstancedBoxes(root, mats, new THREE.BoxGeometry(0.06, 0.004, 1), m.waterLine, 'geo_waterLine');
      }
      return;
    }
    let geom;
    if (g.shape === 'disc') geom = new THREE.CircleGeometry(g.radius, 96);
    else if (g.shape === 'ellipse') geom = new THREE.CircleGeometry(1, 96).scale(g.size[0], g.size[1], 1);
    else geom = new THREE.PlaneGeometry(g.size[0], g.size[1]);
    const w = meshOf(g.name || 'geo_water', geom, m.water, root);
    w.rotation.x = -Math.PI / 2; w.position.set(g.pos ? g.pos[0] : 0, y, g.pos ? g.pos[1] : 0);
    waterShore(g, root, m, y);
    if (g.lines) {
      let s = 7; const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
      const [cx, cz] = g.pos || [0, 0], ex = (g.size ? g.size[0] : g.radius * 1.4) * 0.45, ez = (g.size ? g.size[1] : g.radius * 1.4) * 0.45, mats = [];
      for (let i = 0; i < g.lines; i++) {
        const x = cx + (r() * 2 - 1) * ex, z = cz + (r() * 2 - 1) * ez, L = 1.5 + r() * 5;
        if (g.shape === 'disc' && Math.hypot(x - cx, z - cz) > g.radius * 0.85) continue;
        mats.push(new THREE.Matrix4().compose(new THREE.Vector3(x, y + 0.004, z), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (g.lineYaw || 0) * DEG), new THREE.Vector3(L, 1, 1)));
      }
      addInstancedBoxes(root, mats, new THREE.BoxGeometry(1, 0.004, 0.12), m.waterLine, 'geo_waterLine');
    }
  } else if (g.type === 'mountain') {
    mountain(g, root, m);
  } else if (g.type === 'dune' && g.style !== 'mound') {
    duneRidge(g, root, m);
  } else if (g.type === 'dune') {
    const [rx, h, rz] = g.size, prof = [new THREE.Vector2(1.0, -0.03)];
    for (let i = 1; i <= 16; i++) { const t = 1 - i / 16; prof.push(new THREE.Vector2(t, h * Math.pow(Math.cos(t * Math.PI / 2), 1.5))); }
    const d = meshOf(g.name || 'geo_dune', new THREE.LatheGeometry(prof, 48), m[g.material || 'sand'], root, true);
    d.scale.set(rx, 1, rz); d.position.set(g.pos[0], 0, g.pos[1]); d.rotation.y = (g.rot || 0) * DEG;
  } else if (g.type === 'box') {
    const [w, h, d] = g.size, r = Math.min(g.radius ?? 0.04, w / 2, h / 2, d / 2);
    const b = meshOf(g.name || 'geo_box', new RoundedBoxGeometry(w, h, d, 2, r), m[g.material || 'set'], root, g.cast !== false);
    const p = g.pos; b.position.set(p[0], (p.length === 3 ? p[1] : 0) + h / 2, p[p.length - 1]); b.rotation.y = (g.rot || 0) * DEG;
  }
}
// library v1.3: a shoreline edge (a soft tone darker than the water, never a light stripe), a darker far band toward the horizon and wave lines near shore
function waterShore(g, root, m, wy) {
  if (g.shore === false) return;
  const [px, pz] = g.pos || [0, 0];
  const flat = (name, geom, mat, x, y, z) => { const e = meshOf(name, geom, mat, root); e.rotation.x = -Math.PI / 2; e.position.set(x, y, z); return e; };
  if (g.shape === 'ellipse' || g.shape === 'disc') {
    const [ax, az] = g.shape === 'disc' ? [g.radius, g.radius] : g.size;
    const ring = new THREE.RingGeometry(1, 1 + 0.3 / Math.min(ax, az), 96, 1); ring.scale(ax, az, 1);
    flat('geo_shoreEdge', ring, m.foam, px, wy + 0.003, pz);
    const inner = new THREE.CircleGeometry(1, 64); inner.scale(ax * 0.62, az * 0.62, 1);
    flat('geo_waterDeep', inner, m.waterDeep, px, wy + 0.001, pz);
  } else if (g.size) {
    const [sx, sz] = g.size, shoreZ = pz + sz / 2;
    flat('geo_shoreEdge', new THREE.PlaneGeometry(sx, 0.55), m.foam, px, wy + 0.003, shoreZ - 0.3);
    for (const [i, dz, wdt] of [[0, 2.2, 0.16], [1, 4.6, 0.12]]) flat('geo_swash_' + i, new THREE.PlaneGeometry(sx, wdt), m.foam, px, wy + 0.003, shoreZ - dz);
    if (g.nearLines) {
      let q = 31; const rq = () => ((q = (q * 1664525 + 1013904223) >>> 0) / 4294967296);
      const mats = [];
      for (let i = 0; i < g.nearLines; i++) {
        const dz = 6 + Math.pow(rq(), 1.6) * 60, x = px + (rq() * 2 - 1) * Math.min(sx / 2, 30 + dz * 1.5), L = 2 + rq() * 6 + dz * 0.08;
        mats.push(new THREE.Matrix4().compose(new THREE.Vector3(x, wy + 0.004, shoreZ - dz), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (g.lineYaw || 0) * DEG), new THREE.Vector3(L, 1, 0.1 + dz * 0.004)));
      }
      addInstancedBoxes(root, mats, new THREE.BoxGeometry(1, 0.004, 1), m.waterLine, 'geo_waveNear');
    }
    const far = g.horizon ?? sz * 0.55;
    flat('geo_waterDeep', new THREE.PlaneGeometry(sx, sz - far), m.waterDeep, px, wy + 0.001, shoreZ - far - (sz - far) / 2);
  }
}
// library v1.3 crested dune: gentle windward slope, steep darker lee face, light crest line. size = [length, height, width]
function duneRidge(g, root, m) {
  const [L, h, W] = g.size, crest = 0.25, nu = 20, nv = 6;
  const grp = new THREE.Group(); grp.name = g.name || 'geo_dune'; root.add(grp);
  const H = (u, v) => {
    const t = Math.pow(Math.max(0, 1 - u * u), 0.7), c = crest - 0.35 * (1 - t);
    const prof = v <= c ? Math.pow((v + 1) / (c + 1), 1.6) : Math.pow(Math.max(0, (1 - v) / (1 - c)), 0.75);
    return h * t * prof - 0.03;
  };
  const side = (lee) => {
    const pos = [], idx = [];
    for (let i = 0; i <= nu; i++) {
      const u = -1 + 2 * i / nu, t = Math.pow(Math.max(0, 1 - u * u), 0.7), c = crest - 0.35 * (1 - t);
      for (let j = 0; j <= nv; j++) { const f = j / nv, v = lee ? c + (1 - c) * f : -1 + (c + 1) * f; pos.push(u * L / 2, H(u, v), v * W / 2 + 0.35 * (1 - t) * W / 2); }
    }
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) { const a = i * (nv + 1) + j, b = a + nv + 1; idx.push(a, a + 1, b, b, a + 1, b + 1); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(idx);
    geo.computeVertexNormals();
    let ny = 0; for (let k = 0; k < geo.attributes.normal.count; k++) ny += geo.attributes.normal.getY(k);
    if (ny < 0) { geo.index.array.reverse(); geo.computeVertexNormals(); }
    return geo;
  };
  meshOf('geo_duneWind', side(false), m[g.material || 'sand'], grp, true);
  meshOf('geo_duneLee', side(true), m.sandLee, grp, true);
  const pts = [];
  for (let i = 1; i < nu; i++) { const u = -1 + 2 * i / nu, t = Math.pow(Math.max(0, 1 - u * u), 0.7), c = crest - 0.35 * (1 - t); pts.push(new THREE.Vector3(u * L / 2, H(u, c) + 0.02, c * W / 2 + 0.35 * (1 - t) * W / 2)); }
  meshOf('geo_duneCrest', new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), nu * 2, Math.max(0.05, h * 0.012), 4), m.sandCrest, grp);
  grp.position.set(g.pos[0], 0, g.pos[1]); grp.rotation.y = (g.rot || 0) * DEG;
}
// library v1.3 snow mountain: faceted peak with a snow cap. size = [radius, height]
function mountain(g, root, m) {
  const [r, h] = g.size, seg = g.facets || 9;
  let s = g.seed || 1; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const geo = new THREE.ConeGeometry(r, h, seg, 4, true);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) { const y = p.getY(i); if (y < h / 2 - 0.01) { const k = 1 + (rnd() - 0.5) * 0.35 * (1 - (y + h / 2) / h * 0.5); p.setX(i, p.getX(i) * k); p.setZ(i, p.getZ(i) * k); } }
  geo.translate(0, h / 2 - 0.05, 0);
  const grp = new THREE.Group(); grp.name = g.name || 'mountain'; root.add(grp);
  const body = geo.toNonIndexed(); body.computeVertexNormals();
  meshOf('geo_mountain', body, m[g.material || 'setDark'], grp, true);
  const capH = h * (g.snowline ?? 0.42), capR = r * capH / h;
  const cap = new THREE.ConeGeometry(capR * 1.035, capH, seg, 1, true); cap.translate(0, h - capH / 2 - 0.04, 0);
  const capG = cap.toNonIndexed(); capG.computeVertexNormals();
  meshOf('geo_snowCap', capG, m.snow, grp);
  grp.position.set(g.pos[0], 0, g.pos[1]); grp.rotation.y = (g.rot || 0) * DEG;
}
function addInstancedBoxes(root, mats, geom, mat, name) {
  if (!mats.length) return;
  const im = new THREE.InstancedMesh(geom, mat, mats.length); mats.forEach((x, i) => im.setMatrixAt(i, x));
  im.name = name; im.receiveShadow = true; im.computeBoundingSphere(); root.add(im);
}

// ---------- instancing ----------
function itemMatrix(it) {
  const p = it.pos || [0, 0];
  let yaw = (it.rot || 0) * DEG;
  if (it.asset === 'tree' && it.options && it.options.seed != null) yaw += it.options.seed;          // library: canopy turned by seed (radians)
  if (it.asset === 'rock' && it.options && it.options.seed != null) yaw += it.options.seed * 2.3999; // golden-angle spin stands in for the shape seed
  const s = it.scale == null ? [1, 1, 1] : Array.isArray(it.scale) ? it.scale : [it.scale, it.scale, it.scale];
  const sc = new THREE.Vector3(...s);
  if (it.asset === 'building' && it.options && it.options.floors) sc.y *= it.options.floors / 8;
  return new THREE.Matrix4().compose(new THREE.Vector3(p[0], p.length === 3 ? p[1] : 0, p[p.length - 1]), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw), sc);
}
/** One InstancedMesh per sub-mesh of the template, one instance per matrix. */
export function instanceModel(template, matrices, name) {
  const g = new THREE.Group(); g.name = 'inst_' + name;
  template.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(template.matrixWorld).invert(), rel = new THREE.Matrix4(), m = new THREE.Matrix4();
  template.traverse((o) => {
    if (!o.isMesh) return;
    rel.multiplyMatrices(inv, o.matrixWorld);
    const im = new THREE.InstancedMesh(o.geometry, o.material, matrices.length);
    matrices.forEach((x, i) => im.setMatrixAt(i, m.multiplyMatrices(x, rel)));
    im.castShadow = o.castShadow; im.receiveShadow = o.receiveShadow; im.name = o.name || o.parent.name;
    im.computeBoundingSphere();
    g.add(im);
  });
  return g;
}

/**
 * buildKit(layout, lib, { skipActors = false, instancing = true, personTier = 'lite', vehicleLOD = skipActors, farDist = skipActors ? 12 : 0 })
 * lib: from createLibrary(); every model the layout uses must be loaded (await lib.preload(kitModelIds(layout))).
 * Returns { root, set, dressing, byId, cameras, stage, vehicleStage, stats }.
 */
export function buildKit(layout, lib, options = {}) {
  const o = resolveOptions(options);
  const { stage, vehicleStage } = kitStage(layout);
  const root = new THREE.Group(); root.name = 'kit_' + layout.id;
  root.userData = { kit: layout.id, label: layout.label, move: layout.move };
  const m = setMaterials();
  const set = new THREE.Group(); set.name = 'set'; root.add(set);
  for (const g of layout.geometry || []) buildGeometry(g, set, m);
  const dressing = new THREE.Group(); dressing.name = 'items'; root.add(dressing);
  const byId = {}, referenced = referencedIds(layout);
  const groups = new Map();
  let instanced = 0, cloned = 0;
  const crowd = keptCrowd(layout, o, stage);
  let crowdSkipped = 0;
  const dropped = [];
  for (const it of layout.items || []) {
    if (o.skipActors && isActorItem(it, layout)) continue;
    if (o.skipActors && it.id && riderTargets(layout).has(it.id)) continue; // riderless kit horses, skis and other mounts are clutter
    if (isCrowdItem(it) && !crowd.has(it)) { crowdSkipped++; continue; }
    if (it.subjectSwap && o.subject && swapMatches(it.subjectSwap, o.subject)) { dropped.push(it.id || it.asset); continue; } // v1.4
    if (it.pair) { // v1.4: { id, pair, pos, rot } -> <id>_a and <id>_b with placePair
      const a = lib.instance(PERSON_TIERS[o.personTier] || PERSON_TIERS.lite), b = lib.instance(PERSON_TIERS[o.personTier] || PERSON_TIERS.lite);
      a.name = (it.id || it.pair) + '_a'; b.name = (it.id || it.pair) + '_b';
      const p = it.pos || [0, 0];
      a.position.set(p[0], p.length === 3 ? p[1] : 0, p[p.length - 1]); a.rotation.y = (it.rot || 0) * DEG;
      dressing.add(a); placePair(it.pair, a, b); cloned += 2;
      if (it.id) { byId[it.id + '_a'] = a; byId[it.id + '_b'] = b; }
      continue;
    }
    const far = isFar(it, o, stage, referenced);
    const mid = modelIdFor(it, o.personTier, o.vehicleLOD, far);
    if (o.instancing && (INSTANCE_ASSETS.includes(it.asset) || (far && it.asset === 'bench')) && !(it.id && referenced.has(it.id))) {
      if (!groups.has(mid)) groups.set(mid, []);
      groups.get(mid).push(itemMatrix(it)); instanced++;
      continue;
    }
    const obj = lib.instance(mid); cloned++;
    obj.name = it.id || it.asset;
    if (it.id) byId[it.id] = obj;
    if (it.subjectSwap) obj.userData.subjectSwap = it.subjectSwap; // the shot builder hides it when its subject brings the same thing
    if (it.booth) obj.userData.booth = true; // a booth's bench or table: a sentence that says "booth" sits on these, not on a loose chair
    if (it.asset === 'person') { setPersonScale(obj, typeof it.scale === 'number' ? it.scale : 1); }
    if (it.rider && byId[it.rider]) { attachRider(byId[it.rider], obj, it.riderOptions || {}); continue; }
    if (it.inVehicle && byId[it.inVehicle.of]) { seatInVehicle(obj, byId[it.inVehicle.of], it.inVehicle.seat || 'driver', it.pose || null); continue; }
    if (it.seat && byId[it.seat.on]) { seatOn(byId[it.seat.on], obj, { slot: it.seat.slot ?? null, pose: it.pose || 'sit' }); continue; }
    if (it.onTop && byId[it.onTop.of]) {
      placeOnTop(byId[it.onTop.of], obj, it.onTop.offset || [0, 0]);
      obj.position.y += it.onTop.lift || 0; obj.rotation.set((it.onTop.rotX || 0) * DEG, (it.rot || 0) * DEG, 0, 'YXZ'); continue;
    }
    const p = it.pos || [0, 0];
    obj.position.set(p[0], p.length === 3 ? p[1] : 0, p[p.length - 1]);
    obj.rotation.y = (it.rot || 0) * DEG;
    if (it.scale && it.asset !== 'person') Array.isArray(it.scale) ? obj.scale.set(...it.scale) : obj.scale.setScalar(it.scale);
    dressing.add(obj);
    if (it.asset === 'person') setPose(obj, it.pose || 'stand');
    else if (animalKinds().includes(it.asset) && it.asset !== 'horse' || it.asset === 'horse' && it.pose) setAnimalPose(obj, it.pose || 'stand');
    obj.updateMatrixWorld(true);
    if (it.props) for (const [hand, asset] of Object.entries(it.props)) {
      const prop = lib.instance(asset); prop.name = (it.id || 'person') + '_' + asset;
      attachActionProp(obj, hand, prop, { upright: !!it.upright }); // hand_L/R, hands, foot_L/R, strap (library v1.3.1), push, airborne (v1.4)
    }
  }
  for (const [mid, mats] of groups) dressing.add(instanceModel(lib.template(mid), mats, mid));
  root.updateMatrixWorld(true);
  for (const l of layout.links || []) {
    if ((l.type !== 'leash' && l.type !== 'line') || !byId[l.from[0]] || !byId[l.to[0]]) continue;
    dressing.add(makeLeash(byId[l.from[0]], l.from[1], byId[l.to[0]], l.to[1], { sag: l.sag ?? (l.type === 'line' ? 0.4 : 0.18) })); // v1.4 'line': kite string
  }
  root.userData.dropped = dropped;
  return { root, set, dressing, byId, dropped, cameras: layout.cameras || {}, stage, vehicleStage, world: layout.world || {}, windowWall: layout.windowWall || null, stats: { instanced, cloned, instancedModels: groups.size, farDist: o.farDist, crowdSkipped } };
}
