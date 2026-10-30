// Shot builder: shot spec (runtime/grammar.js) -> THREE scene content, using the kit loader and the attach helpers.
// Deterministic: every placement choice comes from a seeded RNG (spec.seed = hash of the sentence).
// The kit's set dressing is built once per kit and cached; a rebuild only clones and poses the actors and props.
import * as THREE from 'three';
import { swapMatches, buildKit, kitModelIds, setMaterials } from './kit.js';
import { SEATING, SURFACES } from './lexicon.js';
import { setPose, setPersonScale, setChildProportions, setAnimalPose, rigOf, attachPoint } from './poser.js';
import { bindGait, settleWalkers } from './gait.js';
import { bindMotion, bindGaze } from './motion.js';

/** Pull a sitter back onto the seat when their hips have landed on the table in front of it. */
function tuckIntoSeat(person, seat, kit) {
  const table = kit.dressing.children.find((c) => (c.userData.asset === 'table' || c.userData.libraryId === 'table') && c.position.distanceTo(seat.position) < 2.4);
  if (!table) return;
  const d = Math.PI / 180;
  const setJ = (n, x, y, z) => { const j = person.getObjectByName(n); if (j) j.rotation.set(x * d, y * d, z * d); };
  // Sit pose already levels the thighs and bends the knees. These pivots lay the forearms on the tabletop.
  setJ('shoulder_L', -60, 8, 14);
  setJ('shoulder_R', -60, -8, -14);
  setJ('elbow_L', -20, 0, 0);
  setJ('elbow_R', -20, 0, 0);
  setJ('wrist_L', -4, 0, 6);
  setJ('wrist_R', -4, 0, -6);
  person.updateMatrixWorld(true);
  const hips = person.getObjectByName('hips') || person;
  const hp = new THREE.Vector3(); hips.getWorldPosition(hp);
  const sp = seat.getWorldPosition(new THREE.Vector3());
  const tp = table.getWorldPosition(new THREE.Vector3());
  const toTable = tp.clone().sub(sp); toTable.y = 0;
  const along = hp.clone().sub(sp); along.y = 0;
  const len = toTable.length();
  if (len < 0.15) return;
  const proj = along.dot(toTable) / len;
  if (proj <= 0.12) return;
  const back = toTable.multiplyScalar((proj - 0.02) / len);
  const w = person.getWorldPosition(new THREE.Vector3()).sub(back);
  person.position.copy(person.parent.worldToLocal(w));
  person.updateMatrixWorld(true);
}
import { attachProp, attachActionProp, placePair, seatOn, attachRider, seatInVehicle, placeOnTop, makeLeash } from './attach.js';
import { PERSON_TIERS } from './loader.js';
import { bodyTypeOf, shapeBody } from './bodyshape.js';

const DEG = Math.PI / 180;
export const BUILD_DEFAULTS = {
  subjectTier: 'lite',     // 'hero' for close-ups (the 324k-triangle person), 'lite' (91k) is the default subject
  liteCount: 1,            // people up to liteCount use the lite GLB (only the subject by default), the next ones the crowd GLB
  crowdCount: 6,           // after that, instanced low-detail figures (about 600 triangles each)
  instancing: true,
};

export function mulberry32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

/** Model ids a spec needs (kit dressing + actors + props) so they can be fetched in parallel before building. */
function kitBuildOptions(layout, extra = {}) {
  // farDist 0: the __far glbs are kilometre-scale and draw as a slab in the sky.
  return { skipActors: true, farDist: 0, ...(layout && layout.id === 'car_interior' ? { vehicleLOD: false } : {}), ...extra };
}
export function shotModelIds(spec, layout, options = {}) {
  const o = Object.assign({}, BUILD_DEFAULTS, options);
  const ids = new Set(layout ? kitModelIds(layout, kitBuildOptions(layout)) : []);
  let people = 0;
  for (const a of spec.actors) {
    if (a.kind === 'person') { for (let i = 0; i < a.count; i++) { people++; ids.add(tierFor(people, a.id === spec.camera.subject && i === 0, o)); } }
    else ids.add(a.asset);
  }
  for (const p of spec.props) ids.add(p.asset);
  if (/\b(busy|crowded|bustling) (street|sidewalk|city)\b/i.test(spec.text || '')) ids.add(PERSON_TIERS.crowd);
  if (/\bbus stop\b/i.test(spec.text || '')) ids.add('bench');
  if (/\b(waiter|busy kitchen|restaurant kitchen)\b/i.test(spec.text || '') && layout && layout.id === 'kitchen') ids.add(PERSON_TIERS.crowd);
  ids.delete(null);
  return [...ids];
}
function tierFor(n, isSubject, o) {
  if (isSubject) return PERSON_TIERS[o.subjectTier] || PERSON_TIERS.lite;
  if (n <= o.crowdCount) return PERSON_TIERS.crowd;
  return null; // instanced figure
}

/**
 * createShotBuilder({ lib, getLayout }) -> { build(spec, options) -> Promise<Built>, current, dispose() }
 * getLayout(kitId) -> Promise<layout JSON> (cached by the caller or by fetch's HTTP cache).
 */
export function createShotBuilder({ lib, getLayout }) {
  const kits = new Map(); // kitId -> buildKit result (set + instanced dressing), reused across rebuilds
  let current = null;
  async function build(spec, options = {}) {
    const o = Object.assign({}, BUILD_DEFAULTS, options);
    const t0 = performance.now();
    const layout = await getLayout(spec.place.id);
    await lib.preload(shotModelIds(spec, layout, o));
    const t1 = performance.now();
    teardown();
    let kit = kits.get(layout.id);
    if (!kit) {
      kit = buildKit(layout, lib, kitBuildOptions(layout, { instancing: o.instancing }));
      kits.set(layout.id, kit);
    }
    const built = place(spec, layout, kit, lib, o);
    built.timing = { loadMs: Math.round(t1 - t0), buildMs: +(performance.now() - t1).toFixed(1) };
    current = built;
    return built;
  }
  function teardown() {
    if (!current) return;
    for (const f of current.undo) f();
    current = null;
  }
  return { build, teardown, get current() { return current; }, kits };
}

function interiorCar(kit, objs) {
  for (const list of objs.values()) for (const o of list) if (o && o.userData && o.userData.interior) return o;
  for (const c of kit.dressing.children) if (c.visible && c.userData && c.userData.interior) return c;
  return null;
}

// ------------------------------------------------------------------------------------------------
function place(spec, layout, kit, lib, o) {
  const rnd = mulberry32(spec.seed || 1);
  const root = new THREE.Group(); root.name = 'shot';
  root.add(kit.root);
  const shotGroup = new THREE.Group(); shotGroup.name = 'shot_actors'; root.add(shotGroup);
  const undo = [() => { root.remove(kit.root); }];
  const disposables = [];
  // an empty warehouse is the stage kit with the show taken out: the dancer stands on the bare floor mid-room
  const warehouse = layout.id === 'stage' && /\bwarehouse\b/i.test(spec.text || '');
  if (warehouse) {
    const show = kit.stage;
    kit.stage = { pos: [0, 1.5], rot: 0, y: 0 };
    undo.push(() => { kit.stage = show; });
  }
  const S = new THREE.Vector3(kit.stage.pos[0], kit.stage.y || 0, kit.stage.pos[1]); // stage height carries into every spot found around it
  const yaw0 = (kit.stage.rot || 0) * DEG;
  const fwd = new THREE.Vector3(Math.sin(yaw0), 0, Math.cos(yaw0)), right = new THREE.Vector3(fwd.z, 0, -fwd.x);
  const at = (r, f) => S.clone().addScaledVector(right, r).addScaledVector(fwd, f);
  const objs = new Map();  // spec id -> [objects]
  const actorsOut = [], propsOut = [];
  const occupied = [];     // [x, z, radius] footprints to avoid
  const free = (p, r) => occupied.every(([x, z, rr]) => Math.hypot(p.x - x, p.z - z) >= r + rr);
  const claim = (p, r) => occupied.push([p.x, p.z, r]);
  const findSpot = (center, r, minD = 0, maxD = 3) => {
    for (let k = 0; k < 40; k++) {
      const a = rnd() * Math.PI * 2, d = minD + (maxD - minD) * Math.sqrt(rnd()) + k * 0.05;
      const p = center.clone().add(new THREE.Vector3(Math.cos(a) * d, 0, Math.sin(a) * d));
      if (free(p, r)) return p;
    }
    return center.clone().addScaledVector(right, maxD + 1);
  };
  const created = [];
  const add = (obj, pos, yaw) => { shotGroup.add(obj); obj.position.copy(pos); obj.rotation.set(0, yaw, 0); obj.updateMatrixWorld(true); };
  const relOf = (e, t) => (e.relations || []).filter((r) => r.type === t);
  const targetsOf = new Map();
  for (const e of [...spec.actors, ...spec.props]) for (const r of e.relations || []) { if (!targetsOf.has(r.target)) targetsOf.set(r.target, []); targetsOf.get(r.target).push({ from: e, rel: r }); }
  // kit objects we may reuse or hide (restored on teardown)
  const kitByAsset = (asset) => { const out = []; kit.dressing.children.forEach((c) => { if (c.userData.libraryId === asset || c.userData.asset === asset) out.push(c); }); return out; };
  const hide = (obj) => { if (obj.visible) { obj.visible = false; undo.push(() => { obj.visible = true; }); } };
  // v1.4 subjectSwap: the kit's own ball (or striker) yields when the subject kicks or holds one
  const subjA = spec.actors.find((x) => x.id === spec.camera.subject);
  if (subjA) {
    const subj = { pose: subjA.pose, props: (subjA.relations || []).filter((r) => r.type === 'holds').map((r) => (spec.props.find((p) => p.id === r.target) || {}).asset) };
    kit.dressing.children.forEach((c) => { if (c.userData.subjectSwap && swapMatches(c.userData.subjectSwap, subj)) hide(c); });
  }
  // A kit copy of an animal or prop the spec also asks for steps aside for this shot. Animals anywhere in the
  // set (the field's cows sit well past the stage). Props within 6 m of the stage (skis, a loose ball).
  // Seating, tables and the big set pieces stay: the shot reuses those. Restored on the next build.
  const asked = new Set();
  for (const a of spec.actors) if (a.kind !== 'person') asked.add(a.asset);
  for (const p of spec.props) asked.add(p.asset);
  const keepDressing = new Set([...SEATING, ...SURFACES, 'house', 'building', 'tree', 'fountain', 'post', 'rock']);
  if (asked.size) {
    kit.dressing.children.forEach((c) => {
      if (c.isInstancedMesh) return;
      const id = c.userData.asset || c.userData.libraryId;
      if (!id || !asked.has(id) || keepDressing.has(id)) return;
      const animal = spec.actors.some((a) => a.kind !== 'person' && a.asset === id);
      if (!animal && Math.hypot(c.position.x - S.x, c.position.z - S.z) >= 6) return;
      hide(c);
    });
  }
  // A walking crowd brings its own people. Kit extras standing on the stage would be walked through.
  const walkingCrowd = spec.actors.some((a) => a.kind === 'person' && a.count >= 3 && (a.pose === 'walk' || a.pose === 'jog' || a.pose === 'run'));
  if (walkingCrowd) {
    kit.dressing.children.forEach((c) => {
      const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
      if (id !== 'person') return;
      if (Math.hypot(c.position.x - S.x, c.position.z - S.z) < 14) hide(c);
    });
  }
  // A kit animal the sentence did not ask for, near the stage, blocks the shot (the field horses in fog).
  // 11 m catches the grazer just off the stage. The field cows sit further out and stay.
  const ANIMAL = /^(horse|cow|dog|cat|bird|giraffe|elephant|goat|sheep)$/;
  const kiteShot = /\bkite\b/i.test(spec.text || '');
  kit.dressing.children.forEach((c) => {
    const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
    if (!ANIMAL.test(id) || asked.has(id)) return;
    if (Math.hypot(c.position.x - S.x, c.position.z - S.z) < (kiteShot ? 80 : 12)) hide(c);
  });
  // instanced kit dressing of this asset nearest the stage (houses, trees, ...) -> an invisible proxy box in its place,
  // so 'in front of a house' can use the house the kit already shows
  const instProxy = (asset, maxD) => {
    const base = asset === 'house' ? ['house', 'house__1storey'] : [asset], ids = [...base, ...base.map((b) => b + '__far')];
    let best = null;
    for (const id of ids) {
      const g = kit.dressing.getObjectByName('inst_' + id);
      if (!g || !g.children[0]) continue;
      const im = g.children[0], m = new THREE.Matrix4(), p = new THREE.Vector3();
      for (let i = 0; i < im.count; i++) { im.getMatrixAt(i, m); p.setFromMatrixPosition(m); const d = p.distanceTo(S); if (d < maxD && (!best || d < best.d)) best = { d, m: m.clone(), id }; }
    }
    if (!best) return null;
    const t = lib.template(best.id), bb = new THREE.Box3().setFromObject(t), size = bb.getSize(new THREE.Vector3()), c = bb.getCenter(new THREE.Vector3());
    const proxy = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z).translate(c.x, c.y, c.z), new THREE.MeshBasicMaterial());
    proxy.visible = false; proxy.name = 'proxy_' + asset; proxy.userData.asset = asset; proxy.userData.proxy = true;
    best.m.decompose(proxy.position, proxy.quaternion, proxy.scale);
    proxy.rotation.setFromQuaternion(proxy.quaternion);
    disposables.push(proxy.geometry, proxy.material);
    return proxy;
  };

  // ---- 1. props that people sit in or on: vehicles, benches, rideables ----
  let peopleN = 0;
  const makePerson = (a, i) => {
    peopleN++;
    // the one person the sentence singles out ("one person in red stands still") is never an instanced figure
    const singled = a.count === 1 && /\bin red\b/i.test(spec.text || '');
    const id = singled ? PERSON_TIERS.crowd : tierFor(peopleN, a.id === spec.camera.subject && i === 0, o);
    if (!id) return null; // instanced figure, handled later
    const p = lib.instance(id);
    p.name = a.id + '_' + i;
    const preset = (a.presets && a.presets[i]) || a.preset || 'adult';
    setPersonScale(p, (a.size || 1));
    if (preset === 'child') setChildProportions(p, a.size || 1);
    p.userData.preset = preset;
    return p;
  };
  const vehicleSpot = { used: false };
  for (const pr of spec.props) {
    const list = [];
    for (let i = 0; i < pr.count; i++) {
      // reuse: a house / building / tree / fountain the kit already shows near the stage stands in for the mentioned one
      if (i === 0 && ['house', 'building', 'tree', 'fountain', 'post', 'rock'].includes(pr.asset) && !pr.size) {
        const px = instProxy(pr.asset, pr.asset === 'house' || pr.asset === 'building' ? 40 : 15);
        if (px) { shotGroup.add(px); px.updateMatrixWorld(true); claim(px.position, 3); list.push(px); continue; }
      }
      const obj = lib.instance(pr.asset === 'house' ? 'house' : pr.asset);
      obj.name = pr.id + '_' + i;
      const sc = (Array.isArray(pr.scale) ? pr.scale : [pr.scale || 1, pr.scale || 1, pr.scale || 1]).map((v) => v * (pr.size || 1));
      obj.scale.set(...sc);
      // "a coupe", "a sports car": the sedan model reshaped to a fastback (runtime/bodyshape.js)
      const body = pr.kind === 'vehicle' ? bodyTypeOf(pr.word, pr.asset) : null;
      if (body) disposables.push(...shapeBody(obj, body));
      list.push(obj);
    }
    objs.set(pr.id, list);
  }
  // vehicles: the kit's vehicle spot (road lane, parking) for the first occupied or mentioned vehicle
  for (const pr of spec.props.filter((p) => p.kind === 'vehicle')) {
    objs.get(pr.id).forEach((v, i) => {
      let pos, yaw;
      if (!vehicleSpot.used && i === 0) {
        vehicleSpot.used = true;
        const vs = kit.vehicleStage;
        pos = new THREE.Vector3(vs.pos[0], 0, vs.pos[1]); yaw = (vs.rot || 0) * DEG;
        // hide kit vehicles parked on that spot
        kit.dressing.children.forEach((c) => { if (/^(car|suv|truck|van|bus)(__exterior)?(__far)?$/.test(c.userData.libraryId) && c.position.distanceTo(pos) < 6) hide(c); });
        const rel = (targetsOf.get(pr.id) || []).find((t) => t.rel.type === 'near');
        if (rel && !(targetsOf.get(pr.id) || []).some((t) => t.rel.type === 'in')) { pos = at(-2.6 - rnd(), 1.0 + rnd() * 0.5); yaw = yaw0 + 90 * DEG; } // someone stands next to it: bring it to the stage
      } else { pos = findSpot(at(-4, 2), 2.6, 4, 9); yaw = yaw0 + (rnd() < 0.5 ? 90 : -90) * DEG; }
      add(v, pos, yaw); claim(pos, 2.4);
    });
  }
  // seating (bench, sofa): reuse the kit's own piece near the stage, else place one at the stage
  const reuseKit = (asset, maxD = 8) => kitByAsset(asset).filter((b) => b.visible && b.position.distanceTo(S) < maxD).sort((a, b) => a.position.distanceTo(S) - b.position.distanceTo(S));
  const tableBench = () => kitByAsset('bench').filter((b) => kitByAsset('table').some((t) => t.position.distanceTo(b.position) < 2.4)).sort((a, b) => a.position.distanceTo(S) - b.position.distanceTo(S));
  for (const pr of spec.props.filter((p) => SEATING.includes(p.asset))) {
    const nearTable = layout.id === 'interior' && pr.asset === 'bench' ? tableBench() : [];
    // "booth": the kit's own booth bench (high back, against the window wall), not the loose bench nearest the stage
    const booth = pr.word === 'booth' ? kitByAsset('bench').filter((b) => b.visible && b.userData.booth) : [];
    const kb = booth.length ? booth : nearTable.length ? nearTable : reuseKit(pr.asset);
    objs.get(pr.id).forEach((b, i) => {
      if (i === 0 && kb.length) { // swap our clone for the kit's piece (same model), so the scene keeps its composition
        const k = kb[0]; objs.get(pr.id)[0] = k; claim(k.position, 1.4); return;
      }
      const pos = i === 0 ? at(0, -0.6) : at(2.3 * i, -0.6);
      add(b, pos, yaw0); claim(pos, 1.4);
    });
  }
  // tables and counters: the kit's own near the stage, else in front of the seat of whoever sits at it, else near the stage
  for (const pr of spec.props.filter((p) => SURFACES.includes(p.asset))) {
    const kt = reuseKit(pr.asset);
    objs.get(pr.id).forEach((t, i) => {
      if (i === 0 && kt.length) { objs.get(pr.id)[0] = kt[0]; claim(kt[0].position, 1.0); return; }
      const sitter = (targetsOf.get(pr.id) || []).map((x) => x.from).find((a) => relOf(a, 'on')[0]);
      const seat = sitter && (objs.get(relOf(sitter, 'on')[0].target) || [])[0];
      if (i === 0 && seat && seat.parent) {
        const f = new THREE.Vector3(0, 0, 1).applyQuaternion(seat.getWorldQuaternion(new THREE.Quaternion())).setY(0).normalize();
        const pos = seat.getWorldPosition(new THREE.Vector3()).addScaledVector(f, 0.95);
        add(t, pos, seat.rotation.y); claim(pos, 0.9); return;
      }
      const pos = findSpot(at(1.2, 1.2), 0.9, 0, 2.5 + i); add(t, pos, yaw0); claim(pos, 0.9);
    });
  }

  // ---- 2. people and animals ----
  const figureMatrices = [];
  const actorPos = new Map();
  let slot = 0;
  const footprint = (obj) => { const sz = new THREE.Box3().setFromObject(obj).getSize(new THREE.Vector3()); return Math.max(0.3, Math.min(2.2, Math.max(sz.x, sz.z) * 0.55)); };
  const nextFree = (r = 0.45) => { // a row to the stage's right, then a loose cluster
    for (let k = 0; k < 60; k++) {
      const p = slot < 6 ? at(slot * 0.95 * (slot % 2 ? 1 : -1) * 0.5 + (slot ? 0.4 : 0), (rnd() - 0.5) * 0.4) : findSpot(S, r, 1.2, 1.2 + Math.sqrt(slot) * 0.9);
      slot++;
      if (free(p, r)) return p;
    }
    return findSpot(S, r, 1, 6);
  };
  for (const a of spec.actors) {
    const list = [];
    for (let i = 0; i < a.count; i++) {
      if (a.kind === 'person') {
        const p = makePerson(a, i);
        if (!p) {
          const col = i % 8;
          const row = Math.floor(i / 8);
          const pos = (a.count >= 6 && /\bcrossing\b/i.test(spec.text || '')) ? at((col - 3.5) * 1.15, row * 1.3) : nextFree(0.4);
          claim(pos, 0.35);
          const child = ((a.presets && a.presets[i]) || a.preset) === 'child';
          const yaw = /\bcrossing\b/i.test(spec.text || '') ? yaw0 + Math.PI / 2 : yaw0 + (rnd() - 0.5) * 2.4;
          figureMatrices.push(new THREE.Matrix4().compose(pos, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw), new THREE.Vector3(1, 1, 1).multiplyScalar(child ? 1.3 / 1.78 : 0.94 + rnd() * 0.12)));
          list.push(null);
          continue;
        }
        list.push(p);
      } else {
        const an = lib.instance(a.asset); an.name = a.id + '_' + i;
        if (a.size) an.scale.setScalar(a.size);
        list.push(an);
      }
    }
    objs.set(a.id, list);
  }
  for (const a of spec.actors) {
    const list = objs.get(a.id);
    const ins = relOf(a, 'in')[0], on = relOf(a, 'on')[0], rides = relOf(a, 'rides')[0], walks = relOf(a, 'walks')[0];
    const near = relOf(a, 'near')[0], facing = relOf(a, 'facing')[0];
    list.forEach((obj, i) => {
      if (!obj) return;
      const isPerson = a.kind === 'person';
      // seated in a vehicle
      if (isPerson && ins && ins.seats && ins.seats[i]) {
        const v = objs.get(ins.target)[0];
        try { seatInVehicle(obj, v, ins.seats[i], (ins.poses && ins.poses[i]) || null); actorPos.set(obj, v.position.clone()); return; }
        catch (err) { console.warn(err.message); }
      }
      // a car interior with no seat named in the sentence: the person drives, unless they were said to be a passenger
      if (isPerson && layout.id === 'car_interior' && !obj.userData.seated) {
        const car = interiorCar(kit, objs);
        if (car) {
          const said = ins && ins.seats ? ins.seats[i] : null;
          const seat = said && said !== 'driver' ? said : (i === 0 ? 'driver' : 'passenger');
          try { seatInVehicle(obj, car, seat, seat === 'driver' ? 'drive' : null); actorPos.set(obj, car.position.clone()); return; }
          catch (err) { console.warn(err.message); }
        }
      }
      // on a bench (3 slots per bench)
      if (on) {
        const benches = objs.get(on.target) || [];
        const idx = on.slots ? on.slots[i] : i;
        const b = benches[Math.floor((idx ?? 0) / 3)];
        if (b && SEATING.includes(b.userData.asset)) {
          if (isPerson && b.userData.asset === 'bed') { // v1.4 bed: lying on attach_lie, sitting on the sides (attach_seat_L / _R)
            const lie = a.pose === 'lie_down' && attachPoint(b, 'attach_lie');
            if (lie) { // along the mattress, head toward the pillows (-Z of the bed), like the library's bedroom sleeper
              b.updateMatrixWorld(true);
              const n = objs.get(a.id).length, local = b.worldToLocal(lie.getWorldPosition(new THREE.Vector3()));
              local.x = (idx ?? 0) % 2 ? 0.38 : -0.38; // the library sleeper's side; the teddy sits on the other pillow
              const w = shotGroup.worldToLocal(b.localToWorld(local)), q = b.getWorldQuaternion(new THREE.Quaternion());
              add(obj, w, new THREE.Euler().setFromQuaternion(q, 'YXZ').y); setPose(obj, 'lie_down'); obj.position.y = w.y; obj.updateMatrixWorld(true); return;
            }
            seatOn(b, obj, { slot: (idx ?? 0) % 2 ? 'R' : 'L', pose: 'sit' });
            return;
          }
          if (isPerson) {
            seatOn(b, obj, { slot: (idx ?? 1) % 3, pose: a.pose === 'lie_down' ? 'lie_down' : 'sit' });
            tuckIntoSeat(obj, b, kit);
            return;
          }
          setAnimalPose(obj, 'sit'); b.updateMatrixWorld(true);
          const sp = attachPoint(b, 'attach_seat_' + ((idx ?? 1) % 3)) || attachPoint(b, 'attach_seat');
          b.add(obj); obj.position.copy(b.worldToLocal(sp.getWorldPosition(new THREE.Vector3()))); obj.position.y -= 0.02; return;
        }
        if (b && b.userData.asset === 'table' && !isPerson) { placeOnTop(b, obj); return; }
      }
      // riding a bicycle / skateboard / boat
      if (isPerson && rides) {
        const r = objs.get(rides.target)[i] || objs.get(rides.target)[0];
        if (r && !r.parent) {
          let pos = i === 0 ? at(0.2, 0.2) : nextFree(0.9), yaw = yaw0 + 90 * DEG;
          if (r.userData.asset === 'boat') { const w = waterSpot(layout, S); if (w) { pos = w.clone().add(new THREE.Vector3(i * 1.6, 0, i * 0.8)); yaw = yaw0; } }
          if (r.userData.asset === 'skateboard') {
            yaw = yaw0;
            r.userData.cruise = { speed: 5, yaw, start: [pos.x, pos.z] };
          }
          add(r, pos, yaw); claim(pos, 1.0);
        }
        if (r) { attachRider(r, obj); actorPos.set(obj, r.position.clone()); return; }
      }
      // standing / walking actors
      let pos, yaw = yaw0;
      if (a.kind === 'animal') {
        const walker = (targetsOf.get(a.id) || []).find((t) => t.rel.type === 'walks');
        if (walker) {
          const wobj = objs.get(walker.from.id)[0];
          if (wobj) { const wy = wobj.rotation.y; const wf = new THREE.Vector3(Math.sin(wy), 0, Math.cos(wy)), wr = new THREE.Vector3(wf.z, 0, -wf.x); pos = wobj.position.clone().addScaledVector(wf, 1.05).addScaledVector(wr, -0.55 - i * 0.6); yaw = wy; }
        }
      }
      if (!pos && (near || facing)) {
        const tid = (near || facing).target;
        const tobj = (objs.get(tid) || [])[0];
        if (tobj && tobj.parent) {
          tobj.updateMatrixWorld(true);
          const bx = new THREE.Box3().setFromObject(tobj), c = bx.getCenter(new THREE.Vector3()).setY(0), sz = bx.getSize(new THREE.Vector3());
          // the target's facing from its world rotation (instanced proxies decompose to x / z flips, rotation.y alone is not enough)
          const tf = new THREE.Vector3(0, 0, 1).applyQuaternion(tobj.getWorldQuaternion(new THREE.Quaternion())).setY(0).normalize(), tr = new THREE.Vector3(tf.z, 0, -tf.x);
          const ty = Math.atan2(tf.x, tf.z);
          const side = near ? near.side : 'front';
          const halfW = Math.max(sz.x, sz.z) / 2;
          const atSurface = SURFACES.includes(tobj.userData.asset) && side === 'front'; // standing at a counter or a table: close enough to work at it
          const gap = side === 'against' ? 0.28 : atSurface ? 0.1 : 0.55;
          if (facing) { pos = c.clone().addScaledVector(tf, Math.max(1.4, halfW * 0.6 + 1.1)).addScaledVector(tr, i * 0.8); }
          else if (side === 'behind') pos = c.clone().addScaledVector(tf, -(Math.abs(tf.dot(new THREE.Vector3(sz.x, 0, sz.z))) / 2 + gap)).addScaledVector(tr, i * 0.8);
          else if (side === 'front') pos = c.clone().addScaledVector(tf, Math.abs(tf.dot(new THREE.Vector3(sz.x, 0, sz.z))) / 2 + gap + (atSurface ? 0.2 : 0.3)).addScaledVector(tr, i * 0.8);
          else { const half = Math.abs(tr.x) * sz.x / 2 + Math.abs(tr.z) * sz.z / 2; pos = c.clone().addScaledVector(tr, -(half + gap + i * 0.75)).addScaledVector(tf, (rnd() - 0.5) * 0.4); }
          // face the target (people), animals face the same way as a person they stand by
          const look = c.clone().sub(pos); yaw = Math.atan2(look.x, look.z);
          if (side === 'beside' && tobj.userData.asset === 'person') yaw = ty;
          if (side === 'against') yaw += Math.PI;
          // in front of a house or a building: back to it, facing out the way it faces (the classic establishing frame)
          if (side === 'front' && ['house', 'building', 'fountain'].includes(tobj.userData.asset)) yaw = ty;
          if (!atSurface && !free(pos, 0.3)) pos = findSpot(pos, 0.35, 0.3, 1.5);
        }
      }
      if (!pos && a.formation === 'facing' && i === 1) { const f = objs.get(a.id)[0]; const fy = f.rotation.y; pos = f.position.clone().add(new THREE.Vector3(Math.sin(fy), 0, Math.cos(fy)).multiplyScalar(1.15)); yaw = fy + Math.PI; }
      // A walking group shares one facing and a loose grid, so nobody starts inside anyone else.
      if (!pos && isPerson && !a.pair && a.count > 1 && (a.pose === 'walk' || a.pose === 'jog' || a.pose === 'run')) {
        const bothWays = /\b(?:both ways|in both directions|against the crowd|busy sidewalk|busy street)\b/i.test(spec.text || '');
        if (a.count >= 6 && /\bthrough a crowd\b/i.test(spec.text || '')) {
          const ang = (i / a.count) * Math.PI * 2;
          const rad = (/\bsnorricam\b|\bbody mount\b/i.test(spec.text || '') ? 3.5 : 1.6) + (i % 3) * 0.5;
          pos = at(Math.sin(ang) * rad, Math.cos(ang) * Math.abs(rad) + 1.2);
          yaw = yaw0;
        } else if (a.count >= 6 && /\bcrossing\b/i.test(spec.text || '')) {
          const cross = crossingOf(layout, S);
          if (cross) {
            // on the crosswalk, loosely staggered, everyone heading the same way across the road
            const col = i % 4;
            const row = Math.floor(i / 4);
            const jx = (((i * 7919) % 13) / 13 - 0.5) * 0.5;
            const jz = (((i * 104729) % 11) / 11 - 0.5) * 0.2;
            pos = cross.center.clone().addScaledVector(cross.across, 0.6 + col * 1.25 + (row % 2) * 0.6 + jx).addScaledVector(cross.along, (row - 1) * 1.45 + jz);
            yaw = Math.atan2(-cross.across.x, -cross.across.z);
          } else {
            const col = i % 8;
            const row = Math.floor(i / 8);
            pos = at((col - 3.5) * 1.2, row * 1.35);
            yaw = yaw0 + Math.PI / 2;
          }
        } else if (a.count >= 6 && (bothWays || !['toward', 'away'].includes(spec.camera.facing))) {
          // a street crowd walks both ways in lanes, keeping right: two lanes each way, 1.2 m between the two directions
          // so swinging arms never touch, staggered in each lane, so people pass each other and never walk through anyone
          const k = Math.floor(i / 2);
          const against = i % 2 === 1;
          const lane = k % 2;
          const row = Math.floor(k / 2);
          const jit = ((i * 7919) % 13) / 13 - 0.5;
          const side = (0.62 + lane * 0.95 + jit * 0.1) * (against ? 1 : -1);
          const along = row * 2.6 + lane * 1.3 + jit * 0.5;
          pos = at(side, against ? 6 + along : along);
          yaw = against ? yaw0 + Math.PI : yaw0;
          if (against) obj.userData.passing = true; // walks past the lens and out of frame, as a passer-by does
        } else if (a.count >= 6) {
          // a crowd all coming toward the lens (or all walking away): one way, four loose lanes
          const lane = i % 4;
          const row = Math.floor(i / 4);
          const side = (lane - 1.5) * 0.95;
          const along = row * 2.35;
          pos = at(side, along);
          yaw = yaw0;
        } else if (bothWays) {
          // a few people passing both ways: one lane each way
          const against = i % 2 === 1;
          const along = Math.floor(i / 2) * 2.2;
          pos = at(against ? 0.5 : -0.5, against ? 4 + along : -along);
          yaw = against ? yaw0 + Math.PI : yaw0;
          if (against) obj.userData.passing = true;
        } else {
          const group = Math.floor(i / 3);
          const member = i % 3;
          const along = group * 2.2 + member * 0.22;
          const side = (member - 1) * 0.82 + (group % 2 ? 0.18 : -0.12);
          pos = at(side, -along);
          yaw = yaw0;
        }
      }
      const rad = isPerson ? 0.4 : footprint(obj); // v1.4: cows, giraffes and elephants need more room than a dog
      if (!pos) { pos = nextFree(rad); if (a.formation === 'facing' && i === 0) yaw = yaw0 - 90 * DEG; else if (i > 0 || slot > 1) yaw = yaw0 + (rnd() - 0.5) * 0.5; }
      add(obj, pos, yaw); claim(pos, isPerson ? 0.35 : rad);
      if (isPerson) setPose(obj, a.pose && a.pose !== 'drive' && a.pose !== 'sit_passenger' && a.pose !== 'lean_window' ? a.pose : 'stand');
      else { setAnimalPose(obj, a.pose || 'stand'); if (a.pose === 'fly') { obj.position.y += 2.4 + i * 0.5; obj.updateMatrixWorld(true); } } // a flying bird is up in the air
      actorPos.set(obj, pos);
    });
  }
  // ---- 2b. pairs (library v1.4): b at the pair offset in a's frame, turned by facing; poses from the spec ----
  for (const a of spec.actors) {
    if (a.pair) {
      const l = objs.get(a.id) || [];
      if (l[0] && l[1]) {
        placePair(a.pair, l[0], l[1]);
        if (['slow_dance', 'hug', 'kiss'].includes(a.pair.id)) { l[0].userData.holdWith = a.id; l[1].userData.holdWith = a.id; }
        if (a.pair.id === 'slow_dance') {
          // a closed dance hold: his right hand around her waist, fingers on her back, her left hand on his shoulder, his left and her right
          // joined out to the side, heads only just past each other so they read face to face. Arm angles solved offline
          // against those hand targets.
          const setJ = (o, n, x, y, z) => { const j = o.getObjectByName(n); if (j) j.rotation.set(x * DEG, y * DEG, z * DEG); };
          const [A, B] = l;
          setJ(A, 'shoulder_L', -44, 38, 61); setJ(A, 'elbow_L', -101, 0, 0); setJ(A, 'wrist_L', 0, 0, 0);
          setJ(A, 'shoulder_R', -56, 73, -7); setJ(A, 'elbow_R', -13, 0, 0); setJ(A, 'wrist_R', -37, 0, 0);
          setJ(B, 'shoulder_L', -55, -19, 20); setJ(B, 'elbow_L', -71, 0, 0); setJ(B, 'wrist_L', 0, 0, 0);
          setJ(B, 'shoulder_R', -68, -55, -46); setJ(B, 'elbow_R', -101, 0, 0); setJ(B, 'wrist_R', 0, 0, 0);
          for (const o of [A, B]) { setJ(o, 'neck', 2, 6, 0); setJ(o, 'head', 4, 4, 0); setJ(o, 'spine', -3, 0, 0); setJ(o, 'chest', 0, 0, 0); o.updateMatrixWorld(true); }
        }
      }
    }
    for (const r of relOf(a, 'pair')) { const A = (objs.get(a.id) || [])[0], B = (objs.get(r.target) || [])[0]; if (A && B && r.pair) placePair(r.pair, A, B); }
  }
  // A walking pair stands apart across the frame, not one behind the other.
  for (const a of spec.actors) {
    const walking = a.pose === 'walk' || a.pose === 'jog' || a.pose === 'run';
    const couple = walking && ((a.pair && a.count >= 2) || (a.count === 2 && /\bcouple\b/i.test(spec.text || '')));
    if (!couple) continue;
    const l = objs.get(a.id) || [];
    const A = l[0], B = l[1];
    if (!A || !B) continue;
    const yaw = A.rotation.y;
    const fwd = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    const right = new THREE.Vector3(fwd.z, 0, -fwd.x);
    const said = String(spec.text || '');
    const away = /\b(?:walks?|walking|jogs?|jogging|runs?|running)\s+away\b/i.test(said);
    const toward = /\b(?:toward|towards)\b/i.test(said);
    const across = /\bacross\b/i.test(said);
    const az = (away ? 172 : toward ? 8 : across ? 90 : ((spec.seed >>> 0) % 2 === 0 ? 90 : 48)) * DEG;
    const cam = fwd.clone().multiplyScalar(Math.cos(az)).addScaledVector(right, Math.sin(az));
    const screenR = new THREE.Vector3(-cam.z, 0, cam.x);
    if (screenR.lengthSq() > 1e-6) screenR.normalize();
    const gap = 0.7;
    B.position.copy(A.position).addScaledVector(screenR, gap);
    B.rotation.y = yaw;
    B.updateMatrixWorld(true);
  }
  // A face-to-face pair indoors stands on the most open floor, side-on to the room's own camera, with nothing in front.
  let stagedPair = null;
  for (const a of spec.actors) {
    const pr = a.pair;
    if (!pr || Math.abs(Math.abs(pr.facing || 0) - 180) > 1 || pr.id === 'hug' || pr.id === 'kiss') continue;
    const l = objs.get(a.id) || [];
    const A = l[0], B = l[1];
    const room = roomBox(kit);
    const kc = kit.cameras || (layout.cameras || {});
    const start = kc.start || kc.hero;
    if (!A || !B || !room || !start) continue;
    const view = new THREE.Vector3(start.target[0] - start.pos[0], 0, start.target[2] - start.pos[2]);
    if (view.lengthSq() < 1e-4) continue;
    view.normalize();
    const blocks = [];
    kit.dressing.children.forEach((c) => {
      if (!c.visible || c.isInstancedMesh) return;
      const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
      if (/^(person|cat|dog)$/.test(id)) return;
      const bx = new THREE.Box3().setFromObject(c);
      if (!bx.isEmpty() && bx.max.y > 0.3) blocks.push(bx);
    });
    let best = null;
    const mid = room.getCenter(new THREE.Vector3()).setY(0);
    const across = new THREE.Vector3(-view.z, 0, view.x);
    for (let x = room.min.x + 1.6; x <= room.max.x - 1.6; x += 0.2) {
      for (let z = room.min.z + 1.6; z <= room.max.z - 1.6; z += 0.2) {
        const p = new THREE.Vector3(x, 0, z);
        let clear = 9;
        for (const bx of blocks) clear = Math.min(clear, Math.hypot(Math.max(bx.min.x - x, 0, x - bx.max.x), Math.max(bx.min.z - z, 0, z - bx.max.z)));
        // in front of the room's back wall, not out at its open side
        const depth = p.clone().sub(mid).dot(view);
        const lateral = Math.abs(p.clone().sub(mid).dot(across));
        const score = Math.min(clear, 1.6) + depth * 0.1 - lateral * 0.15;
        if (!best || score > best.score) best = { p, score };
      }
    }
    if (!best) continue;
    const side = new THREE.Vector3(-view.z, 0, view.x);
    const half = (pr.offset ? Math.hypot(pr.offset[0], pr.offset[2]) : 1.2) / 2;
    const yawA = Math.atan2(side.x, side.z);
    A.position.copy(best.p).addScaledVector(side, -half);
    A.rotation.set(0, yawA, 0);
    B.position.copy(best.p).addScaledVector(side, half);
    B.rotation.set(0, yawA + Math.PI, 0);
    A.updateMatrixWorld(true);
    B.updateMatrixWorld(true);
    stagedPair = { center: best.p.toArray(), view: view.toArray(), span: half * 2 };
    // kit extras would be standing in the argument
    kit.dressing.children.forEach((c) => {
      const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
      if (id === 'person' && c !== A && c !== B) hide(c);
    });
    break;
  }
  // ---- 3. held props, leashes, loose props ----
  const kites = [];
  const clearPush = (person, cart) => { // v1.4: kit dressing the cart runs into steps aside (hidden for this shot, like a parked car on the vehicle spot)
    const bx = new THREE.Box3().setFromObject(cart).expandByScalar(-0.03);
    for (const c of kit.dressing.children) if (c.visible && !c.isInstancedMesh && c.userData.asset !== 'person' && new THREE.Box3().setFromObject(c).intersectsBox(bx)) hide(c);
  };
  for (const a of spec.actors) for (const r of relOf(a, 'holds')) {
    const props = objs.get(r.target) || [];
    const slot = r.hand === 'L' ? 'hand_L' : r.hand === 'R' || !r.hand ? 'hand_R' : r.hand; // hand_L, hand_R, hands, foot_L, foot_R, strap
    objs.get(a.id).forEach((p, i) => {
      const pr = props[i] || null;
      if (!p || !pr || pr.parent) return;
      attachActionProp(p, slot, pr, { upright: a.pose !== 'sip' });
      if (slot === 'airborne') kites.push([p, pr]);
      if (slot === 'push') clearPush(p, pr); // v1.4: a cart never runs through a counter
    });
  }
  const leashMat = new THREE.MeshStandardMaterial({ name: 't2_detail', color: 0x1A1A1A, roughness: 0.5 });
  disposables.push(leashMat);
  root.updateMatrixWorld(true);
  for (const [p, kite] of kites) if (attachPoint(kite, 'attach_line')) { const l = makeLeash(p, 'attach_hand_R', kite, 'attach_line', { sag: 1.1, radius: 0.0016, material: leashMat }); l.name = 'kite_line'; l.userData.noFit = true; shotGroup.add(l); }
  for (const a of spec.actors) for (const r of relOf(a, 'walks')) {
    const people = objs.get(a.id), dogs = objs.get(r.target) || [];
    people.forEach((p, i) => { const d = dogs[i]; if (p && d && attachPoint(d, 'attach_leash')) { const l = makeLeash(p, 'attach_hand_R', d, 'attach_leash', { material: leashMat }); shotGroup.add(l); disposables.push(l.geometry); } });
  }
  for (const pr of spec.props) objs.get(pr.id).forEach((obj, i) => {
    if (obj.parent) return;
    const rel = (pr.relations || [])[0];
    if (rel && rel.type === 'on') { const t = (objs.get(rel.target) || [])[0]; if (t) { placeOnTop(t, obj); return; } }
    // a kit with a window opening in its wall: that opening is the window, so the loose panel is kept out of the shot
    if (pr.asset === 'window_panel' && kit.windowWall && !targetsOf.has(pr.id)) { add(obj, S.clone(), yaw0); obj.visible = false; return; }
    const big = ['house', 'building', 'tree', 'fountain'].includes(pr.asset);
    const hasRel = targetsOf.has(pr.id);
    const pos = big ? findSpot(at(0, -9), 5, 0, 6) : hasRel ? findSpot(at(-1.2, 0.6), 0.8, 0.4, 2) : findSpot(at(-2, 1.5), 0.8, 1, 3.5);
    add(obj, pos, yaw0 + (big ? 0 : (rnd() - 0.5) * 1.2) + (pr.asset === 'house' ? Math.PI : 0)); claim(pos, big ? 5 : 0.8);
    void i;
  });
  // ---- 4. instanced crowd figures beyond the crowdCount ----
  if (figureMatrices.length) { const f = crowdFigures(figureMatrices); shotGroup.add(f.group); disposables.push(...f.disposables); }
  // teardown: detach everything this shot created (people seated on a reused kit bench included), keep the kit's own objects
  for (const list of objs.values()) for (const ob of list) if (ob && ob.parent !== kit.dressing) created.push(ob);
  undo.push(() => { for (const ob of created) ob.removeFromParent(); root.remove(shotGroup); for (const d of disposables) d.dispose && d.dispose(); });
  if (/\bcrossing\b/i.test(spec.text || '') && spec.actors.some((x) => x.kind === 'person' && x.count >= 6)) {
    const cross = crossingOf(layout, S);
    if (cross) {
      const white = new THREE.MeshStandardMaterial({ color: 0xE2E0DA, roughness: 0.8 });
      const bar = new THREE.BoxGeometry(0.5, 0.01, 3.4);
      disposables.push(white, bar);
      for (let k = -Math.floor(cross.width / 2 / 0.9); k <= Math.floor(cross.width / 2 / 0.9); k++) {
        const m = new THREE.Mesh(bar, white);
        m.name = 'geo_zebra';
        m.position.copy(cross.center).addScaledVector(cross.across, k * 0.9);
        m.position.y = 0.012;
        m.rotation.y = Math.atan2(cross.along.x, cross.along.z);
        m.receiveShadow = true;
        shotGroup.add(m);
      }
    }
  }
  let kitchenFollow = null;
  // a waiter through a busy kitchen: the length of the room between the counter and the table, staff at work on both sides
  if (layout.id === 'kitchen' && /\b(waiter|server|busy)\b/i.test(spec.text || '')) {
    const sa = spec.actors.find((x) => x.id === spec.camera.subject);
    const obj = sa && (objs.get(sa.id) || [])[0];
    if (obj && ['walk', 'jog', 'run'].includes(sa.pose)) {
      obj.position.set(3.3, obj.position.y, -0.45);
      obj.rotation.set(0, -Math.PI / 2, 0);
      obj.updateMatrixWorld(true);
      kit.dressing.children.forEach((c) => {
        const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
        if (id === 'person' || id === 'cat') hide(c);
      });
      kit.set.traverse((n) => { if (n.name === 'wall_right') hide(n); });
      kit.dressing.children.forEach((c) => { if (c.userData.asset === 'counter' && c.position.x > 2) hide(c); });
      kitchenFollow = obj;
      // every cook faces their station (the back counter, the prep table), never the lens, and clear of his lane
      const staff = [[-1.4, -1.55, Math.PI, 'cook'], [0.6, -1.6, Math.PI, 'cook'], [-2.6, -1.6, Math.PI, 'cook'], [-0.15, 0.98, 0, 'cook'], [-1.15, 0.98, 0, 'cook']];
      staff.forEach(([x, z, yaw, pose], k) => {
        const p = lib.instance(PERSON_TIERS.crowd);
        p.name = 'staff_' + k;
        p.userData.extra = true;
        add(p, new THREE.Vector3(x, 0, z), yaw);
        try { setPose(p, pose); } catch { setPose(p, 'stand'); }
        created.push(p);
      });
    }
  }
  // A tracked walker on a shore runs along the waterline, the water behind them as the kit's own camera sees it.
  const trackMove = ['track', 'dolly_along'].includes(spec.camera.move) || (spec.camera.segments || []).some((x) => x.move === 'track');
  const water = (layout.geometry || []).find((g) => g.type === 'water');
  const kc0 = (kit.cameras || layout.cameras || {}).start;
  if (trackMove && water && kc0) {
    const sa = spec.actors.find((x) => x.id === spec.camera.subject);
    const obj = sa && (objs.get(sa.id) || [])[0];
    if (obj && ['walk', 'jog', 'run'].includes(sa.pose)) {
      const v = new THREE.Vector3(kc0.target[0] - kc0.pos[0], 0, kc0.target[2] - kc0.pos[2]).normalize();
      const screenRight = new THREE.Vector3(-v.z, 0, v.x);
      const dir = spec.camera.facing === 'right_to_left' ? screenRight.clone().negate() : screenRight;
      obj.rotation.y = Math.atan2(dir.x, dir.z);
      obj.updateMatrixWorld(true);
    }
  }
  // ---- output ----
  // where "looking out the window" looks: through the kit's window opening, a little ahead of the person; else at the nearest window panel
  const windowPoint = (obj) => {
    const p = obj.getWorldPosition(new THREE.Vector3());
    const ww = kit.windowWall;
    if (ww) {
      const n = new THREE.Vector3(ww.normal[0], 0, ww.normal[1]), along = new THREE.Vector3(-n.z, 0, n.x);
      const f = new THREE.Vector3(0, 0, 1).applyQuaternion(obj.getWorldQuaternion(new THREE.Quaternion())).setY(0);
      const u = Math.min(ww.span[1] - 0.3, Math.max(ww.span[0] + 0.3, p.dot(along) + Math.sign(f.dot(along) || 1) * 0.6));
      return along.multiplyScalar(u).addScaledVector(n, ww.at - 2.5).setY(1.25); // 2.5 m outside the wall's inner face
    }
    const panes = propsOut.map((x) => x.object).concat(kitByAsset('window_panel')).filter((x) => x && x.userData && x.userData.asset === 'window_panel' && x.parent);
    panes.sort((u, v) => u.getWorldPosition(new THREE.Vector3()).distanceTo(p) - v.getWorldPosition(new THREE.Vector3()).distanceTo(p));
    return panes[0] ? panes[0].getWorldPosition(new THREE.Vector3()) : null;
  };
  let gaitN = 0;
  for (const a of spec.actors) objs.get(a.id).forEach((obj, i) => {
    if (!obj) return;
    bindGait(obj, a, i, spec, kit);
    const g = obj.userData.gait;
    const crowdWalk = a.count >= 3 && (a.pose === 'walk' || a.pose === 'jog' || a.pose === 'run');
    if (g && !g.animal) {
      const n = gaitN++;
      const vary = crowdWalk || n > 0;
      if (vary) {
        const h = Math.imul((spec.seed || 1) ^ Math.imul(n + 1, 374761393), 16777619) >>> 0;
        const span = crowdWalk ? 0.2 : 0.1;
        const scale = (1 - span / 2) + ((h % 1000) / 999) * span;
        g.speed *= scale;
        g.stride *= scale;
        g.phase = (g.phase + ((h >>> 8) % (crowdWalk ? 700 : 280)) / 1000) % 1;
        if (crowdWalk) {
          const hs = 0.95 + ((h % 997) / 996) * 0.1;
          try { rigOf(obj).scale.multiplyScalar(hs); } catch { /* no rig */ }
        }
      }
    }
    if (!g && a.kind === 'person') {
      const pairId = a.pair && a.pair.id;
      if (pairId === 'argue' || (a.pose === 'talk' && a.count >= 2)) bindMotion(obj, 'argue', i);
      else if (pairId === 'slow_dance') bindMotion(obj, 'sway', i);
      else if (a.pose === 'type') bindMotion(obj, 'type', i);
      else if (a.pose === 'dance') bindMotion(obj, 'dance', i);
      else if ((spec.beats || []).some((b) => b.actor === a.id && (b.type === 'look_at_camera' || b.type === 'look_out'))) bindGaze(obj, spec.beats.filter((b) => b.actor === a.id), windowPoint(obj));
    }
    actorsOut.push({ id: a.id, index: i, kind: a.kind === 'person' ? 'person' : 'animal', object: obj });
  });
  // a slow dance turns slowly on the spot around the pair's middle
  const dancers = actorsOut.filter((x) => x.object && x.object.userData.motion && x.object.userData.motion.kind === 'sway');
  if (dancers.length === 2) {
    const c = dancers[0].object.position.clone().add(dancers[1].object.position).multiplyScalar(0.5);
    for (const d of dancers) { d.object.userData.motion.center = c.toArray(); d.object.userData.motion.spin = 35 * DEG; }
  }
  for (const p of spec.props) objs.get(p.id).forEach((obj, i) => { if (obj) propsOut.push({ id: p.id, index: i, kind: p.kind, object: obj, related: targetsOf.has(p.id) }); });
  const sub = actorsOut.find((x) => x.id === spec.camera.subject) || propsOut.find((x) => x.id === spec.camera.subject) || actorsOut[0] || propsOut[0] || null;
  let subject = null;
  if (sub) {
    const q = new THREE.Quaternion(); sub.object.getWorldQuaternion(q);
    const e = new THREE.Euler().setFromQuaternion(q, 'YXZ');
    const seated = sub.object.userData.seated && sub.object.userData.seated.vehicle;
    subject = { id: sub.id, kind: sub.kind, object: sub.object, yaw: e.y, vehicle: seated ? sub.object.parent : null };
  }
  // Field hills (hill_A/B/C lathes) and the barn read as a pale slab through fog and behind a kite.
  // Desert dune ridges (geo_dune, geo_duneLee) read as a pale curve and a hard folded facet.
  if (layout.id === 'field' || layout.id === 'desert') {
    const drop = layout.id === 'field' ? /^(hill_[A-Z]|barn|barn_roof)$/ : /^geo_dune/;
    kit.set.traverse((n) => { if (drop.test(n.name || '')) hide(n); });
  }
  if (layout.id === 'desert') {
    kit.set.traverse((n) => {
      if (n.name === 'geo_ground') n.scale.multiplyScalar(6);
      if (/horizon|rim|wedge/i.test(n.name || '')) hide(n);
    });
  }
  root.updateMatrixWorld(true);
  const skater = actorsOut.find((a) => a.object && a.object.parent && a.object.parent.userData && a.object.parent.userData.cruise);
  if (skater) {
    const home = new THREE.Vector3();
    skater.object.parent.getWorldPosition(home);
    kit.dressing.traverse((n) => {
      const id = String((n.userData && (n.userData.asset || n.userData.libraryId)) || n.name || '');
      if (!/bench/.test(id) || n.userData.guardHidden) return;
      if (n.isInstancedMesh) {
        const m4 = new THREE.Matrix4();
        const p = new THREE.Vector3();
        const q = new THREE.Quaternion();
        const s = new THREE.Vector3();
        for (let i = 0; i < n.count; i++) {
          n.getMatrixAt(i, m4);
          new THREE.Matrix4().multiplyMatrices(n.matrixWorld, m4).decompose(p, q, s);
          if (Math.hypot(p.x - home.x, p.z - home.z) < 8) hide(n);
        }
      } else if (n.isMesh || n.userData.asset) {
        const p = new THREE.Vector3();
        n.getWorldPosition(p);
        if (Math.hypot(p.x - home.x, p.z - home.z) < 8) hide(n);
      }
    });
  }
  const said = spec.text || '';
  const quietPeople = /\bempty\b|\brooftop\b|\btwo-shot\b|\bacross from\b|\binsert shot\b|\ba man standing on a rooftop\b/i.test(said);
  if (quietPeople) {
    const owned = (n) => { let p = n; while (p) { if (actorsOut.some((a) => a.object === p)) return true; p = p.parent; } return false; };
    kit.dressing.traverse((n) => {
      if (owned(n)) return;
      const id = String((n.userData && (n.userData.asset || n.userData.libraryId)) || '').split('__')[0];
      if (id === 'person') hide(n);
    });
  }
  if (/\bin red\b/i.test(said)) {
    const stand = (spec.actors || []).find((a) => (a.kind === 'person' || a.cat === 'person') && a.pose === 'stand');
    const who = (stand && actorsOut.find((a) => a.id === stand.id && a.object)) || actorsOut.find((a) => a.kind === 'person' && a.object);
    if (who) who.object.traverse((n) => {
      if (!n.isMesh || !n.material) return;
      const mats = [].concat(n.material).map((m) => {
        const c = m.clone();
        if (c.color) c.color.setHex(0xD23A3A);
        if (c.emissive) { c.emissive.setHex(0xD23A3A); c.emissiveIntensity = 0.65; }
        c.vertexColors = false;
        disposables.push(c);
        return c;
      });
      n.material = mats.length === 1 ? mats[0] : mats;
    });
    if (who) {
      const mates = actorsOut.filter((a) => a.object && a.object !== who.object);
      if (mates.length && /\bcrossing\b/i.test(said)) {
        // mid-stream: where the crowd is halfway across, between two rows, so they part around the one standing still
        const spot = new THREE.Vector3();
        for (const m of mates) spot.add(m.object.getWorldPosition(new THREE.Vector3()));
        spot.multiplyScalar(1 / mates.length);
        const g = mates[0].object.userData.gait;
        const walk = g ? new THREE.Vector3(Math.sin(g.yaw), 0, Math.cos(g.yaw)) : new THREE.Vector3(1, 0, 0);
        const half = g ? g.speed * (spec.duration_s || 4) * 0.5 : 0;
        spot.addScaledVector(walk, half).addScaledVector(new THREE.Vector3(walk.z, 0, -walk.x), 0.72);
        spot.y = 0;
        const parent = who.object.parent;
        who.object.position.copy(parent ? parent.worldToLocal(spot.clone()) : spot);
        who.object.updateMatrixWorld(true);
      }
      const head = who.object.getObjectByName && who.object.getObjectByName('head');
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), new THREE.MeshStandardMaterial({ color: 0xD23A3A, emissive: 0xD23A3A, emissiveIntensity: 0.4 }));
      cap.name = 'red_mark';
      cap.position.y = 0.16;
      (head || who.object).add(cap);
      disposables.push(cap.geometry, cap.material);
    }
  }
  if (/\bfrom the door\b/i.test(said)) {
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    if (who) {
      const p = new THREE.Vector3();
      who.object.getWorldPosition(p);
      const yaw = who.object.rotation.y;
      const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
      const door = new THREE.Mesh(new THREE.BoxGeometry(1.15, 2.25, 0.08), new THREE.MeshStandardMaterial({ color: 0x4A3428, roughness: 0.62 }));
      door.name = 'shot_door';
      if (spec.place && spec.place.id === 'interior') {
        door.position.set(-4.05, 1.15, 0.1);
        door.rotation.y = Math.PI / 2;
      } else {
        door.position.copy(p).addScaledVector(right, -6);
        door.position.y = 1.15;
        door.lookAt(p.x, door.position.y, p.z);
      }
      shotGroup.add(door);
      disposables.push(door.geometry, door.material);
    }
  }
  if (/\b(surf|his board|her board)\b/i.test(said)) {
    // A surfboard (2.1 m, on its rail) under the far arm, in place of the carried box.
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    for (const p of propsOut) {
      const id = String((p.object && (p.object.userData.asset || p.object.userData.libraryId || p.object.name)) || '');
      if (p.object && /bag|box/i.test(id)) hide(p.object);
    }
    if (who) {
      const geo = new THREE.CapsuleGeometry(0.26, 1.58, 6, 16);
      geo.scale(1, 1, 0.13);
      geo.rotateX(Math.PI / 2);
      geo.rotateZ(Math.PI / 2);
      const board = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xF4F1EA, roughness: 0.5 }));
      board.name = 'surfboard';
      board.userData.asset = 'surfboard';
      board.userData.noFit = true;
      board.castShadow = true;
      // the person's local -x is their left when they walk away from the side camera
      board.position.set(-0.3, 0.98, 0.12);
      board.rotation.set(0, 0, -0.12);
      who.object.add(board);
      disposables.push(geo, board.material);
      if (layout.id === 'ocean') who.object.userData.wade = true;
    }
  }
  // the surf: a broken foam line along the waterline and one further out, so the shore is not a straight slab
  if (layout.id === 'ocean' && /\b(surf|surfer|waves?)\b/i.test(said)) {
    const water = (layout.geometry || []).find((g) => g.type === 'water' && g.size);
    if (water) {
      const shoreZ = (water.pos ? water.pos[1] : 0) + water.size[1] / 2;
      const foam = new THREE.MeshStandardMaterial({ color: 0xE4E6E3, roughness: 0.75 });
      disposables.push(foam);
      const band = (z0, width, amp, seed, name) => {
        const n = 600, half = 60;
        const pos = [], idx = [];
        for (let k = 0; k <= n; k++) {
          const x = -half + (2 * half * k) / n;
          const wob = Math.sin(x * 0.45 + seed) * amp + Math.sin(x * 1.3 + seed * 2) * amp * 0.45 + Math.sin(x * 3.1 + seed) * amp * 0.18;
          const w = width * (0.75 + 0.25 * Math.sin(x * 0.8 + seed * 3));
          pos.push(x, 0, z0 + wob, x, 0, z0 + wob - w);
          if (k < n) { const a = k * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setIndex(idx);
        g.computeVertexNormals();
        const mesh = new THREE.Mesh(g, foam);
        mesh.name = name;
        mesh.position.y = 0.02;
        mesh.receiveShadow = true;
        shotGroup.add(mesh);
        disposables.push(g);
      };
      band(shoreZ + 0.25, 0.9, 0.35, 1.3, 'surf_line');
      band(shoreZ - 3.2, 0.35, 0.6, 4.1, 'surf_break');
      band(shoreZ - 7.5, 0.25, 0.9, 2.2, 'surf_far');
      for (const a of actorsOut) if (a.object && a.object.userData.wade) a.object.userData.wade = { shoreZ: shoreZ - 0.2 };
    }
  }
  // on the phone (a call, not texting): the hand at the ear, the elbow out
  if (/\b(?:on (?:a|the|his|her) (?:cell ?)?phone|phone call|talking on)\b/i.test(said)) {
    for (const a of actorsOut) {
      if (a.kind !== 'person' || !a.object || a.object.userData.pose !== 'hold_phone') continue;
      const sh = a.object.getObjectByName('shoulder_R'), el = a.object.getObjectByName('elbow_R'), wr = a.object.getObjectByName('wrist_R');
      if (sh) sh.rotation.set(-80 * DEG, 30 * DEG, -64 * DEG);
      if (el) el.rotation.set(-138 * DEG, -60 * DEG, 0);
      if (wr) wr.rotation.set(0, 0, 0);
      const nk = a.object.getObjectByName('neck'), hd = a.object.getObjectByName('head');
      if (nk) nk.rotation.set(6 * DEG, 0, 0);
      if (hd) hd.rotation.set(2 * DEG, 0, 0);
      a.object.updateMatrixWorld(true);
      a.object.userData.call = true;
    }
  }
  // a busy street: passers-by on the sidewalk ahead, walking both ways along it
  if (/\b(busy|crowded|bustling) (street|sidewalk|city)\b/i.test(said) && layout.id === 'city') {
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    if (who) {
      const o = who.object;
      const yaw = o.rotation.y;
      const f = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw)), r = new THREE.Vector3(f.z, 0, -f.x);
      const base = o.getWorldPosition(new THREE.Vector3());
      const spots = [[2.6, -0.9, 0], [4.6, 0.7, 0], [6.6, -0.2, 0], [8.8, 1.1, 0], [11.0, 0.2, 0], [9.5, -1.7, 1], [12.5, -1.1, 1], [15.5, -2.1, 1]];
      spots.forEach(([ahead, side, back], k) => {
        const id = PERSON_TIERS.crowd;
        if (!id) return;
        const p = lib.instance(id);
        p.name = 'passerby_' + k;
        p.userData.preset = 'adult';
        p.userData.extra = true;
        const pos = base.clone().addScaledVector(f, ahead).addScaledVector(r, side);
        add(p, pos, back ? yaw + Math.PI : yaw);
        setPose(p, 'walk');
        bindGait(p, { id: 'passerby', kind: 'person', pose: 'walk', count: 1 }, k, spec, kit);
        if (p.userData.gait) { p.userData.gait.follow = false; p.userData.gait.speed *= 0.9 + 0.04 * (k % 4); p.userData.gait.phase = (p.userData.gait.phase + k * 0.37) % 1; }
        created.push(p);
        actorsOut.push({ id: 'passerby', index: k, kind: 'person', object: p, extra: true });
      });
      // the kit's own street extras stay out of the way of these
    }
  }
  // a chase through traffic: the hero car weaves past two slower vehicles, oncoming traffic passes the other way
  if (/\b(car chase|through traffic|weaving)\b/i.test(said) && layout.id === 'highway') {
    const heroP = propsOut.find((p) => p.id === spec.camera.subject) || propsOut.find((p) => p.object && p.object.userData.asset === 'car');
    const drive = (obj, speed, lanes, extra = {}) => {
      const prev = { pos: obj.position.clone(), rot: obj.rotation.y, drive: obj.userData.drive };
      obj.userData.drive = { start: [obj.position.x, obj.position.z], yaw: obj.rotation.y, speed, lanes, ...extra };
      undo.push(() => { obj.position.copy(prev.pos); obj.rotation.y = prev.rot; obj.userData.drive = prev.drive; obj.updateMatrixWorld(true); });
    };
    if (heroP && heroP.object) {
      // lanes: +x is the fast lane here (side < 0 moves toward +x for a car heading +z)
      // each lane change takes over a second; out before the slow car, back in once clear of it and well before the
      // truck; the camera follows each one 0.35 s late
      drive(heroP.object, 16, [[0.12, 0.4, -3.5], [0.6, 0.86, 3.5]], { follow: true, delay: 0.35 });
      const kitCar = (id) => kit.dressing.children.find((c) => c.visible && String(c.userData.asset || c.userData.libraryId || '').split('__')[0] === id);
      const truck = kitCar('truck');
      if (truck) { truck.position.set(5.25, truck.position.y, 40); drive(truck, 8, []); }
      const van = kitCar('van');
      if (van) { van.position.set(-5.25, van.position.y, 40); van.rotation.y = Math.PI; drive(van, 15, []); }
      const suv = kitCar('suv');
      if (suv) { suv.position.set(-1.75, suv.position.y, 78); suv.rotation.y = Math.PI; drive(suv, 13, []); }
      const slow = lib.instance('car');
      slow.name = 'traffic_car';
      add(slow, new THREE.Vector3(1.75, 0, 22), 0);
      slow.userData.drive = { start: [1.75, 22], yaw: 0, speed: 7, lanes: [] };
      created.push(slow);
      // the parked second car of the sentence stays out of the chase
      for (const p of propsOut) if (p !== heroP && p.object && p.object.userData.asset === 'car') hide(p.object);
    }
  }
  // Run 3 (R&D's c07): a car the sentence drives, filmed tracking or following, drives on its own heading and the camera
  // rides with it (a camera car), so the road and the far background stream past the same way. A parked car with the
  // camera swinging past it moved the hills one way and the road the other.
  const camMove = spec.camera && spec.camera.move;
  if (CAMERA_CAR_MOVES.includes(camMove) && DRIVES.test(said) && !/\b(park(ed|s|ing)?|stationary)\b/i.test(said) && layout.id !== 'car_interior') {
    const heroP = propsOut.find((p) => p.id === spec.camera.subject) || propsOut.find((p) => p.object && /^(car|suv|truck|van|bus|taxi)$/.test(String(p.object.userData.asset)));
    const car = heroP && heroP.object;
    if (car && /^(car|suv|truck|van|bus|taxi)$/.test(String(car.userData.asset)) && !car.userData.drive) {
      const prev = car.userData.drive;
      car.userData.drive = { start: [car.position.x, car.position.z], yaw: car.rotation.y, speed: CAMERA_CAR_SPEED, lanes: [], follow: true };
      undo.push(() => { car.userData.drive = prev; });
      // Run 5 (R&D's c07): on the highway the camera car sees only the road, the rail and the far land. The kit's parking
      // garage is set geometry (never filler) and came back as a ghost scaffold over the hills; the roadside posts stand
      // on the camera car's line (x 8.6 against 8.66), so one passed through the lens and wiped the render at f15.
      if (layout.id === 'highway') {
        kit.set.children.forEach((g) => { if (/^garage_/.test(g.name)) hide(g); });
        const yaw = car.rotation.y, cx = car.position.x, cz = car.position.z;
        const across = (x, z) => Math.abs((x - cx) * Math.cos(yaw) - (z - cz) * Math.sin(yaw));
        kit.dressing.updateMatrixWorld(true);
        const m4 = new THREE.Matrix4(), pp = new THREE.Vector3();
        kit.dressing.traverse((o) => {
          const id = String((o.parent && (o.parent.userData.libraryId || o.parent.name)) || '').replace(/^inst_/, '').split('__')[0];
          if (o.isInstancedMesh && id === 'post') {
            for (let n = 0; n < o.count; n++) {
              o.getMatrixAt(n, m4);
              pp.setFromMatrixPosition(new THREE.Matrix4().multiplyMatrices(o.matrixWorld, m4));
              if (across(pp.x, pp.z) >= CAMERA_CAR_REACH) continue;
              const saved = m4.clone(), at = n;
              o.setMatrixAt(n, new THREE.Matrix4().makeTranslation(0, -400, 0));
              o.instanceMatrix.needsUpdate = true;
              undo.push(() => { o.setMatrixAt(at, saved); o.instanceMatrix.needsUpdate = true; });
            }
          } else if (o.parent === kit.dressing && String(o.userData.asset || o.userData.libraryId || '').split('__')[0] === 'post' && across(o.position.x, o.position.z) < CAMERA_CAR_REACH) hide(o);
        });
      }
    }
  }
  // a bus stop: he sits on the stop's bench with his back to the road, the bus behind him pulls away
  let shotCam = null;
  if (/\bbus stop\b/i.test(said) && layout.id === 'city') {
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    const busP = propsOut.find((p) => p.object && p.object.userData.asset === 'bus');
    const cross = crossingOf(layout, S);
    if (who && cross) {
      const curb = cross.center.clone().addScaledVector(cross.across, Math.sign(S.clone().sub(cross.center).dot(cross.across)) * (cross.width / 2 + 1.6));
      const out = curb.clone().sub(cross.center).setY(0).normalize(); // from the road toward the sidewalk
      const bench = lib.instance('bench');
      bench.name = 'busstop_bench';
      add(bench, curb, Math.atan2(out.x, out.z));
      created.push(bench);
      seatOn(bench, who.object, { slot: attachPoint(bench, 'attach_seat_1') ? 1 : null, pose: 'sit' });
      if (busP) {
        const bus = busP.object;
        const lane = cross.center.clone().addScaledVector(out, cross.width / 4);
        lane.addScaledVector(cross.along, 1.0);
        bus.position.set(lane.x, bus.position.y, lane.z);
        bus.rotation.set(0, Math.atan2(cross.along.x, cross.along.z), 0);
        bus.updateMatrixWorld(true);
        bus.userData.drive = { start: [lane.x, lane.z], yaw: bus.rotation.y, speed: 1.0, accel: 3.2, lanes: [] };
        // other buses and cars parked on that stretch of road leave the frame clear
        kit.dressing.children.forEach((c) => {
          const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
          if (/^(bus|car|suv|van|truck)$/.test(id) && c.position.distanceTo(lane) < 16) hide(c);
        });
      }
      const head = who.object.getObjectByName('head').getWorldPosition(new THREE.Vector3());
      const look = head.clone(); look.y = 0.95;
      look.addScaledVector(out, -1.2);
      const camAt = (d, y) => { const c = head.clone().addScaledVector(out, d).addScaledVector(cross.along, -0.9); c.y = y; return c.toArray(); };
      shotCam = {
        start: { pos: camAt(3.4, 1.45), target: look.toArray(), fov: 40 },
        end: { pos: camAt(2.2, 1.3), target: look.toArray(), fov: 40 },
        label: 'full, push in, the bus pulls away',
      };
      // street extras near the stop stay out of the shot
      kit.dressing.children.forEach((c) => {
        const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
        if (id === 'person' && c.position.distanceTo(curb) < 12) hide(c);
      });
    }
  }
  // bombing a hill: a 7 degree ramp the board runs down, wide and long enough to fill a point of view
  if (/\b(hill|downhill|bombing)\b/i.test(said) && skater) {
    const board = skater.object.parent;
    const ride = board.userData.cruise;
    if (/\bbomb(ing|s)?\b/i.test(said)) ride.speed = 9; // bombing is flat out, about 32 km/h, not a cruise
    const slope = 7 * DEG;
    const run = ride.speed * (spec.duration_s || 4);
    const L0 = -6, L1 = run + 40;
    const top = (L1 - 0) * Math.tan(slope);
    const f = new THREE.Vector3(Math.sin(ride.yaw), 0, Math.cos(ride.yaw)), r = new THREE.Vector3(f.z, 0, -f.x);
    const o = new THREE.Vector3(ride.start[0], 0, ride.start[1]);
    const hAt = (u) => Math.max(0, (L1 - u) * Math.tan(slope));
    const W = 7;
    const P = (u, side, y) => o.clone().addScaledVector(f, u).addScaledVector(r, side).setY(y);
    const verts = [];
    const quad = (a, b, c, d) => verts.push(...a.toArray(), ...b.toArray(), ...c.toArray(), ...a.toArray(), ...c.toArray(), ...d.toArray());
    // top, two sides and the back face of the wedge
    quad(P(L0, -W, hAt(L0)), P(L0, W, hAt(L0)), P(L1, W, 0.005), P(L1, -W, 0.005));
    quad(P(L0, W, hAt(L0)), P(L0, W, 0), P(L1, W, 0), P(L1, W, 0.005));
    quad(P(L0, -W, 0), P(L0, -W, hAt(L0)), P(L1, -W, 0.005), P(L1, -W, 0));
    quad(P(L0, -W, 0), P(L0, W, 0), P(L0, W, hAt(L0)), P(L0, -W, hAt(L0)));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: 0x77756F, roughness: 0.95, side: THREE.DoubleSide });
    const hill = new THREE.Mesh(g, mat);
    hill.name = 'hill_road';
    hill.receiveShadow = true;
    shotGroup.add(hill);
    disposables.push(g, mat);
    // lane dashes down the middle so the speed reads
    const dashMat = new THREE.MeshStandardMaterial({ color: 0xD8D6CF, roughness: 0.8 });
    const dashGeo = new THREE.BoxGeometry(0.14, 0.01, 1.6);
    disposables.push(dashMat, dashGeo);
    for (let u = L0 + 2; u < L1 - 1; u += 4) { // 4 m apart: never a whole number of dashes per second at 9 m/s, so stills read as motion too
      const d = new THREE.Mesh(dashGeo, dashMat);
      d.name = 'hill_dash';
      d.position.copy(P(u, 1.6, hAt(u) + 0.012));
      d.rotation.order = 'YXZ';
      d.rotation.set(slope, ride.yaw, 0);
      shotGroup.add(d);
    }
    ride.slope = { y0: hAt(0), tan: Math.tan(slope), pitch: slope };
    board.position.y = hAt(0);
    board.rotation.order = 'YXZ';
    board.rotation.x = slope;
    board.updateMatrixWorld(true);
    undo.push(() => { board.rotation.x = 0; board.position.y = 0; delete ride.slope; });
    // dressing on the run is out of the way
    kit.dressing.children.forEach((c) => {
      if (c === board || !c.visible) return;
      const cp = c.getWorldPosition(new THREE.Vector3()).sub(o);
      const u = cp.dot(f), side = Math.abs(cp.dot(r));
      if (u > -8 && u < L1 + 4 && side < W + 6) hide(c);
    });
  }
  // a truck past a row of trees reveals a person on a ridge: the ridge out ahead, the trees close to the lens
  if (/\btrees?\b/i.test(said) && /\breveal/i.test(said) && /\b(ridge|hill|crest)\b/i.test(said) && /\btruck/i.test(said)) {
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    const kc = (kit.cameras || layout.cameras || {}).start;
    if (who && kc) {
      const v = new THREE.Vector3(kc.target[0] - kc.pos[0], 0, kc.target[2] - kc.pos[2]).normalize();
      const r = new THREE.Vector3(-v.z, 0, v.x);
      const sign = /\btruck left\b/i.test(said) ? -1 : 1;
      const H = S.clone().setY(0);
      const ridgeTop = 2.4;
      // ridge: a long rounded bank across the view
      const R = 4.2;
      const rg = new THREE.SphereGeometry(R, 48, 24);
      rg.scale(1, 1, 5.5);
      const ridge = new THREE.Mesh(rg, new THREE.MeshStandardMaterial({ color: 0x8C8A80, roughness: 1 }));
      ridge.name = 'ridge';
      ridge.position.copy(H).setY(ridgeTop - R);
      ridge.rotation.y = Math.atan2(r.x, r.z);
      ridge.receiveShadow = true; ridge.castShadow = true;
      shotGroup.add(ridge);
      disposables.push(rg, ridge.material);
      // the hiker stands on the crest, looking out along it
      const o = who.object;
      delete o.userData.gait;
      setPose(o, 'stand');
      const parent = o.parent;
      const top = H.clone().setY(ridgeTop);
      o.position.copy(parent ? parent.worldToLocal(top.clone()) : top);
      o.rotation.set(0, Math.atan2(r.x * sign, r.z * sign) + 0.5, 0);
      o.updateMatrixWorld(true);
      // the camera path and the trees between it and the ridge
      const dist = 13, travel = 4.2;
      const look1 = top.clone().setY(ridgeTop + 0.95);
      const C1 = H.clone().addScaledVector(v, -dist).setY(1.5);
      const C0 = C1.clone().addScaledVector(r, -travel * sign);
      for (const p of propsOut) if (p.object && p.object.userData.asset === 'tree') hide(p.object);
      const P0 = C0.clone().add(look1.clone().sub(C0).setY(0).multiplyScalar(3.4 / dist)).setY(0);
      [0, 1, 2, 3].forEach((k) => {
        const t = lib.instance('tree');
        t.name = 'reveal_tree_' + k;
        t.scale.setScalar(0.62 + 0.05 * (k % 2));
        add(t, P0.clone().addScaledVector(r, -2.6 * k * sign).addScaledVector(v, (k % 2) * 0.6 - 0.3), k * 1.3);
        created.push(t);
      });
      // the view out to the ridge is clear of kit dressing
      kit.dressing.children.forEach((c) => {
        if (!c.visible) return;
        const cp = c.getWorldPosition(new THREE.Vector3()).sub(C1);
        const ahead = cp.dot(v), side = cp.dot(r) * sign;
        if (ahead > -2 && ahead < dist + 10 && side > -travel - 6 && side < 9) hide(c);
      });
      shotCam = {
        start: { pos: C0.toArray(), target: look1.clone().addScaledVector(r, -travel * sign).toArray(), fov: 34 },
        end: { pos: C1.toArray(), target: look1.toArray(), fov: 34 },
        label: 'full, truck ' + (sign > 0 ? 'right' : 'left') + ', reveal',
      };
    }
  }
  if (kitchenFollow && !shotCam) {
    // Steadicam: close behind at shoulder height, the waiter's back and the kitchen ahead of him
    const p = kitchenFollow.getWorldPosition(new THREE.Vector3());
    const yaw = kitchenFollow.rotation.y;
    const f = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    const at = (back, y) => p.clone().addScaledVector(f, -back).setY(y).toArray();
    const aim = p.clone().addScaledVector(f, 3).setY(1.15).toArray();
    shotCam = { start: { pos: at(2.7, 1.68), target: aim, fov: 44 }, end: { pos: at(2.35, 1.62), target: aim.slice(), fov: 44 }, label: 'steadicam, follow from behind' };
  }
  // a booth by a window (the kit's authored booth cameras): wide on the booth, the window wall and the counter, then in on the sitter
  const bc = kit.cameras || layout.cameras || {};
  if (!shotCam && bc.booth_start && bc.booth_end && !spec.camera.angle && !['close', 'extreme_close', 'medium'].includes(spec.camera.shot) && [null, undefined, 'push_in', 'pull_out', 'static'].includes(spec.camera.move)) {
    const onBooth = (o) => { for (let n = o && o.parent; n; n = n.parent) if (n.userData && n.userData.booth) return true; return false; };
    const sitter = actorsOut.find((a) => a.kind === 'person' && onBooth(a.object));
    if (sitter) {
      const st = bc.booth_start, en = bc.booth_end, out = spec.camera.move === 'pull_out';
      const a = { pos: st.pos.slice(), target: st.target.slice(), fov: st.fov }, b = { pos: en.pos.slice(), target: en.target.slice(), fov: en.fov };
      shotCam = spec.camera.move === 'static'
        ? { start: a, end: { pos: st.pos.slice(), target: st.target.slice(), fov: st.fov }, label: 'wide, booth by the window' }
        : { start: out ? b : a, end: out ? a : b, label: 'wide, ' + (out ? 'pull out' : 'push in') + ', booth by the window' };
      // the room's other tables and benches between the lens and the sitter step aside for the shot (the diner is quiet: he is alone)
      const hp = sitter.object.getObjectByName('head').getWorldPosition(new THREE.Vector3());
      const reach = (c) => {
        const q = c.getWorldPosition(new THREE.Vector3());
        return [shotCam.start, shotCam.end].some((k) => {
          const p0 = new THREE.Vector3(k.pos[0], 0, k.pos[2]), d = new THREE.Vector3(hp.x - p0.x, 0, hp.z - p0.z), L = d.length();
          const u = Math.max(0, Math.min(1, new THREE.Vector3(q.x - p0.x, 0, q.z - p0.z).dot(d) / (L * L)));
          return Math.hypot(q.x - (p0.x + d.x * u), q.z - (p0.z + d.z * u)) < 1.15 && u < 1 - 0.8 / L;
        });
      };
      const cleared = [];
      kit.dressing.children.forEach((c) => {
        const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
        if (c.isInstancedMesh || c.userData.booth || !c.visible || !/^(table|bench|person)$/.test(id)) return;
        if (reach(c)) { hide(c); if (id === 'table') cleared.push(c.getWorldPosition(new THREE.Vector3())); }
      });
      // what stood on a table that stepped aside goes with it
      kit.dressing.children.forEach((c) => {
        const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
        if (c.isInstancedMesh || c.userData.booth || !c.visible || !/^(plate_food|cup)$/.test(id)) return;
        const q = c.getWorldPosition(new THREE.Vector3());
        if (cleared.some((t) => Math.hypot(q.x - t.x, q.z - t.z) < 0.8)) hide(c);
      });
    }
  }
  // a lone figure: nothing parked or standing near them
  if (/\b(lone|alone|solitary)\b/i.test(said)) {
    const who = actorsOut.find((a) => a.kind === 'person' && a.object);
    if (who) {
      const p = who.object.getWorldPosition(new THREE.Vector3());
      // never the figure itself, or the kit seat it was put on ("alone in a booth")
      const mine = new Set();
      for (let n = who.object; n; n = n.parent) mine.add(n);
      kit.dressing.children.forEach((c) => {
        if (mine.has(c)) return;
        const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
        if (c.userData.booth) return; // the booth's other bench stays: a man alone in a booth, not a booth with one bench
        if (/^(car|suv|van|truck|bus|person|bench|bicycle)$/.test(id) && c.getWorldPosition(new THREE.Vector3()).distanceTo(p) < 14) hide(c);
      });
    }
  }
  // typing: the typist sits square to the laptop, palms resting on the keys
  const lapProp = propsOut.find((p) => p.object && p.object.userData.asset === 'laptop');
  const typist = actorsOut.find((a) => a.kind === 'person' && a.object && a.object.userData.pose === 'type');
  if (lapProp && typist) {
    const lap = lapProp.object, who = typist.object;
    lap.updateMatrixWorld(true);
    const keys = attachPoint(lap, 'attach_keys');
    const kp = (keys || lap).getWorldPosition(new THREE.Vector3());
    const lq = lap.getWorldQuaternion(new THREE.Quaternion());
    const front = new THREE.Vector3(0, 0, 1).applyQuaternion(lq).setY(0).normalize(); // away from the screen
    const yaw = Math.atan2(-front.x, -front.z);
    const parent = who.parent;
    who.rotation.set(0, yaw - (parent ? new THREE.Euler().setFromQuaternion(parent.getWorldQuaternion(new THREE.Quaternion()), 'YXZ').y : 0), 0);
    who.updateMatrixWorld(true);
    const hl = (attachPoint(who, 'attach_hand_L') || who).getWorldPosition(new THREE.Vector3());
    const hr = (attachPoint(who, 'attach_hand_R') || who).getWorldPosition(new THREE.Vector3());
    const hands = hl.add(hr).multiplyScalar(0.5);
    const goal = kp.clone().addScaledVector(front, 0.06);
    goal.y = kp.y + 0.03;
    const w = who.getWorldPosition(new THREE.Vector3()).add(goal.sub(hands));
    who.position.copy(parent ? parent.worldToLocal(w) : w);
    who.updateMatrixWorld(true);
  }
  if (/\bacross from\b/i.test(said)) {
    // Both seated, one on each side of the same table, square to it and facing each other.
    const sit = actorsOut.filter((a) => a.kind === 'person' && a.object).map((a) => a.object);
    if (sit.length >= 2) {
      const A = sit[0];
      const B = sit[1];
      const wa = A.getWorldPosition(new THREE.Vector3());
      const tables = kitByAsset('table').filter((t) => t.visible).concat(propsOut.filter((p) => p.object && p.object.userData.asset === 'table').map((p) => p.object));
      const table = tables.sort((u, v) => u.getWorldPosition(new THREE.Vector3()).distanceTo(wa) - v.getWorldPosition(new THREE.Vector3()).distanceTo(wa))[0];
      if (table) {
        const tp = table.getWorldPosition(new THREE.Vector3());
        const seats = kitByAsset('bench').concat(kitByAsset('chair')).filter((b) => b.visible && b.getWorldPosition(new THREE.Vector3()).distanceTo(tp) < 1.3);
        const sides = [1, -1].map((sgn) => seats.find((b) => { const bp = b.getWorldPosition(new THREE.Vector3()).sub(tp); return Math.abs(bp.x) + Math.abs(bp.z) > 0.3 && Math.sign(Math.abs(bp.z) >= Math.abs(bp.x) ? bp.z : bp.x) === sgn; }));
        if (sides[0] && sides[1]) {
          [[A, sides[0]], [B, sides[1]]].forEach(([who, seat]) => {
            const slot = attachPoint(seat, 'attach_seat_1') ? 1 : null;
            seatOn(seat, who, { slot, pose: 'sit' });
            tuckIntoSeat(who, seat, kit);
            bindMotion(who, 'chat', who === A ? 0 : 1);
            if (!created.includes(who)) created.push(who);
          });
          // a side-on two-shot of the table: the camera on the side nearer the middle of the room
          const ha = A.getObjectByName('head').getWorldPosition(new THREE.Vector3());
          const hb = B.getObjectByName('head').getWorldPosition(new THREE.Vector3());
          const center = ha.clone().add(hb).multiplyScalar(0.5).setY(0);
          const axis = hb.clone().sub(ha).setY(0).normalize();
          let view = new THREE.Vector3(-axis.z, 0, axis.x);
          const room = roomBox(kit);
          const mid = room ? room.getCenter(new THREE.Vector3()).setY(0) : S.clone().setY(0);
          if (mid.clone().sub(center).dot(view) > 0) view.negate();
          stagedPair = { center: center.toArray(), view: view.toArray(), span: ha.distanceTo(hb), seated: true, keep: [table, sides[0], sides[1]] };
        }
      }
    }
  }
  if (/\b(circles|orbit)\b/i.test(said) && layout.id === 'interior') {
    const box = new THREE.Box3();
    let any = false;
    kit.set.traverse((n) => { if (n.isMesh && /wall_/.test(n.name || '')) { box.union(new THREE.Box3().setFromObject(n)); any = true; } });
    if (any && !box.isEmpty()) {
      const c = box.getCenter(new THREE.Vector3());
      c.y = 0;
      const folk = actorsOut.filter((a) => a.kind === 'person' && a.object);
      const mid = new THREE.Vector3();
      for (const a of folk) mid.add(a.object.getWorldPosition(new THREE.Vector3()));
      if (folk.length) mid.multiplyScalar(1 / folk.length);
      // the group moves toward the middle of the room together; the spacing inside it stays
      const shift = new THREE.Vector3((c.x - mid.x) * 0.85, 0, (c.z - mid.z) * 0.85);
      for (const a of folk) {
        const world = new THREE.Vector3();
        a.object.getWorldPosition(world);
        world.add(shift);
        const parent = a.object.parent;
        a.object.position.copy(parent ? parent.worldToLocal(world.clone()) : world);
        a.object.updateMatrixWorld(true);
      }
    }
  }
  // at night the buildings' window bands light up, most floors but not all, so a skyline reads
  if (spec.world && spec.world.time === 'night') {
    // window panes from world position (the band meshes carry no UVs): dark frames, most panes lit warm, some dark
    const lit = new THREE.MeshStandardMaterial({ name: 'windows_lit', color: 0x1C1C1E, emissive: 0xF0C470, emissiveIntensity: 0.8, roughness: 0.6 });
    lit.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWinPos;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n  vec4 winP = vec4(transformed, 1.0);\n  #ifdef USE_INSTANCING\n  winP = instanceMatrix * winP;\n  #endif\n  vWinPos = (modelMatrix * winP).xyz;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWinPos;')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n  float wu = (vWinPos.x + vWinPos.z) / 1.7;\n  float cell = floor(wu);\n  float row = floor(vWinPos.y / 3.0);\n  float on = step(0.38, fract(sin(cell * 12.9898 + row * 78.233) * 43758.5453));\n  float pane = step(0.16, fract(wu)) * step(fract(wu), 0.86);\n  totalEmissiveRadiance *= on * pane * (0.75 + 0.25 * fract(sin(cell * 3.7 + row * 9.1) * 911.3));');
    };
    lit.customProgramCacheKey = () => 'windows_lit_v1';
    disposables.push(lit);
    let f = 0;
    kit.dressing.traverse((n) => {
      if (!n.isMesh || !n.parent || !/building/.test(n.parent.name || '')) return;
      const mat = [].concat(n.material)[0];
      if (!mat || mat.name !== 't3_setDark') return;
      if (!n.geometry.boundingBox) n.geometry.computeBoundingBox();
      const size = n.geometry.boundingBox.getSize(new THREE.Vector3());
      const wide = Math.max(size.x, size.z);
      if (size.y > 0.2 * wide || Math.min(size.x, size.z) > 0.05 * wide) return; // long thin window bands only, not the entrance, canopy or roof
      f++;
      if (((f * 2654435761) >>> 0) % 5 === 0) return;
      const prev = n.material;
      n.material = lit;
      undo.push(() => { n.material = prev; });
    });
  }
  let balcony = null;
  if (/\bbalcony\b/i.test(said)) {
    // A real balcony on the nearest building's street face, a floor and a half up, and the person standing on it.
    const w = actorsOut.find((a) => a.kind === 'person' && a.object);
    let bx = null;
    kit.dressing.updateMatrixWorld(true);
    // one box per building item: the meshes whose centres sit on that item's footprint
    const items = (layout.items || []).filter((it) => it.asset === 'building' && it.pos);
    const meshes = [];
    kit.dressing.traverse((n) => {
      if (!n.isMesh) return;
      let p = n, ok = false;
      while (p) { if (/building/.test(String(p.userData.asset || p.userData.libraryId || p.name || ''))) { ok = true; break; } p = p.parent; }
      if (!ok) return;
      if (n.isInstancedMesh) {
        const m4 = new THREE.Matrix4();
        for (let k = 0; k < n.count; k++) {
          n.getMatrixAt(k, m4);
          if (!n.geometry.boundingBox) n.geometry.computeBoundingBox();
          meshes.push(n.geometry.boundingBox.clone().applyMatrix4(new THREE.Matrix4().multiplyMatrices(n.matrixWorld, m4)));
        }
      } else meshes.push(new THREE.Box3().setFromObject(n));
    });
    for (const it of items) {
      const b = new THREE.Box3();
      for (const m of meshes) { const mc = m.getCenter(new THREE.Vector3()); if (Math.hypot(mc.x - it.pos[0], mc.z - it.pos[1]) < 5 && m.max.y > 3) b.union(m); }
      if (b.isEmpty() || b.max.y < 8) continue;
      const d = b.distanceToPoint(S);
      if (!bx || d < bx.d) bx = Object.assign(b, { d });
    }
    if (typeof process !== 'undefined' && process.env && process.env.BALC) console.log('balcony box', bx && bx.min.toArray(), bx && bx.max.toArray());
    if (w && bx) {
      const c = bx.getCenter(new THREE.Vector3());
      const street = S.clone();
      const dx = street.x - c.x, dz = street.z - c.z;
      const n = Math.abs(dx) * (bx.max.z - bx.min.z) > Math.abs(dz) * (bx.max.x - bx.min.x) ? new THREE.Vector3(Math.sign(dx), 0, 0) : new THREE.Vector3(0, 0, Math.sign(dz));
      const along = new THREE.Vector3(-n.z, 0, n.x);
      const face = n.x ? (n.x > 0 ? bx.max.x : bx.min.x) : (n.z > 0 ? bx.max.z : bx.min.z);
      const y = 6.0;
      const mid = c.clone();
      if (n.x) mid.x = face; else mid.z = face;
      const u = Math.max(-((n.x ? bx.max.z - bx.min.z : bx.max.x - bx.min.x) / 2 - 2), Math.min((n.x ? bx.max.z - bx.min.z : bx.max.x - bx.min.x) / 2 - 2, street.clone().sub(c).dot(along)));
      mid.addScaledVector(along, u);
      const mat = new THREE.MeshStandardMaterial({ color: 0x8E8B86, roughness: 0.8 });
      const dark = new THREE.MeshStandardMaterial({ color: 0x3A3936, roughness: 0.6 });
      disposables.push(mat, dark);
      const yawN = Math.atan2(n.x, n.z);
      const part = (geo, m2, off, h) => {
        const mesh = new THREE.Mesh(geo, m2);
        mesh.name = 'balcony';
        mesh.position.copy(mid).addScaledVector(n, off[0]).addScaledVector(along, off[1]);
        mesh.position.y = h;
        mesh.rotation.y = yawN;
        mesh.castShadow = true; mesh.receiveShadow = true;
        shotGroup.add(mesh);
        disposables.push(geo);
        return mesh;
      };
      part(new THREE.BoxGeometry(2.6, 0.16, 1.3), mat, [0.65, 0], y - 0.08);
      part(new THREE.BoxGeometry(2.6, 0.05, 0.05), dark, [1.27, 0], y + 1.0);
      part(new THREE.BoxGeometry(0.05, 0.05, 1.3), dark, [0.65, 1.28], y + 1.0);
      part(new THREE.BoxGeometry(0.05, 0.05, 1.3), dark, [0.65, -1.28], y + 1.0);
      for (let k = -6; k <= 6; k++) part(new THREE.BoxGeometry(0.025, 1.0, 0.025), dark, [1.27, k * 0.2], y + 0.5);
      part(new THREE.BoxGeometry(3.4, 3.1, 0.04), new THREE.MeshStandardMaterial({ color: 0x55534F, roughness: 0.9 }), [0.03, 0], y + 1.45);
      part(new THREE.BoxGeometry(1.0, 2.2, 0.06), new THREE.MeshStandardMaterial({ color: 0xE8C77A, emissive: 0xE8B860, emissiveIntensity: 0.9 }), [0.07, -0.75], y + 1.1);
      const spot = mid.clone().addScaledVector(n, 0.75).addScaledVector(along, 0.45);
      spot.y = y;
      const parent = w.object.parent;
      w.object.position.copy(parent ? parent.worldToLocal(spot.clone()) : spot);
      w.object.rotation.set(0, yawN, 0);
      w.object.updateMatrixWorld(true);
      balcony = { pos: spot.toArray(), normal: n.toArray(), along: along.toArray() };
      // the street furniture below never reaches the lens
      kit.dressing.children.forEach((ch) => {
        const id = String(ch.userData.asset || ch.userData.libraryId || '').split('__')[0];
        if (id === 'person' && ch !== w.object) hide(ch);
      });
    }
  }
  if (/\b(waiter|over the shoulder)\b/i.test(said)) {
    const w = actorsOut.find((a) => a.kind === 'person' && a.object);
    if (w) {
      const p = new THREE.Vector3();
      w.object.getWorldPosition(p);
      kit.dressing.traverse((n) => {
        const id = String(n.name || '') + String((n.userData && (n.userData.asset || n.userData.libraryId)) || '');
        if (!/vase|plant|pot|tree|lamp/.test(id)) return;
        const q = new THREE.Vector3();
        n.getWorldPosition(q);
        if (p.distanceTo(q) < 3.2) hide(n);
      });
    }
  }
  if (layout.id === 'ocean' || layout.id === 'lake') {
    // a matte sea: at a low sun the glossy water flared into a white slab across the lower frame
    kit.set.traverse((n) => {
      if (!n.isMesh || !/water|sea/i.test(n.name || '')) return;
      for (const m of [].concat(n.material)) {
        if (!m || m.roughness == null || m.roughness >= 0.7) continue;
        const prev = m.roughness;
        m.roughness = 0.78;
        undo.push(() => { m.roughness = prev; });
      }
    });
  }
  if (layout.id === 'desert') {
    // sand a step darker than the pale sky, so the horizon and the ground read apart
    kit.set.traverse((n) => {
      if (!n.isMesh || n.name !== 'geo_ground') return;
      for (const m of [].concat(n.material)) {
        if (!m || !m.color) continue;
        const prev = m.color.getHex();
        m.color.setHex(0x9A8C72);
        undo.push(() => { m.color.setHex(prev); });
      }
    });
  }
  if (layout.id === 'ocean') {
    const dropLane = (n) => { if (/lane|dash|road|waterLine|waveNear|swash|shoreEdge/i.test(n.name || '')) hide(n); };
    kit.set.traverse(dropLane);
    kit.dressing.traverse(dropLane);
  }
  if (warehouse) {
    // no stage, amps, truss or mic stand; the walls stay, and a fourth wall closes the room for a full orbit
    kit.set.traverse((n) => { if (/^(stage|amp_|truss_)/.test(n.name || '') && n.name !== 'stage_back') hide(n); });
    kit.dressing.children.forEach((c) => { if (/^(inst_)?(post|microphone|guitar)/.test(String(c.userData.asset || c.userData.libraryId || c.name || ''))) hide(c); });
    const side = kit.set.getObjectByName('wall_left');
    const mat = side && side.isMesh ? side.material : side && side.children.find((m) => m.isMesh)?.material;
    const front = new THREE.Mesh(new THREE.BoxGeometry(18.4, 7, 0.2), mat || new THREE.MeshStandardMaterial({ color: 0xB9B4AA }));
    front.name = 'wall_front';
    front.position.set(0, 3.5, 9.1);
    front.receiveShadow = true;
    shotGroup.add(front);
    disposables.push(front.geometry);
  } else if (layout.id === 'stage' && /\b(empty|orbit)\b/i.test(said)) {
    kit.set.traverse((n) => { if (/wall_|stage_back/.test(n.name || '')) hide(n); });
  }
  const ANIMAL_ID = /^(horse|cow|dog|cat|bird|giraffe|elephant|goat|sheep)$/;
  const people = actorsOut.filter((x) => x.kind === 'person' && x.object);
  kit.dressing.children.forEach((c) => {
    const id = String(c.userData.asset || c.userData.libraryId || '').split('__')[0];
    if (!ANIMAL_ID.test(id) || asked.has(id) || !c.visible) return;
    const ap = new THREE.Vector3(); c.getWorldPosition(ap);
    for (const person of people) {
      const g = person.object.userData.gait;
      const start = new THREE.Vector3(); person.object.getWorldPosition(start);
      const yaw = person.object.rotation.y;
      const travel = Math.abs((g && g.speed) || 0) * (spec.duration_s || 4) + 2;
      const dirx = Math.sin(yaw), dirz = Math.cos(yaw);
      const dx = ap.x - start.x, dz = ap.z - start.z;
      const along = dx * dirx + dz * dirz;
      const clamped = Math.max(-2, Math.min(travel, along));
      if (Math.hypot(dx - dirx * clamped, dz - dirz * clamped) < 12) { hide(c); break; }
    }
  });
  // a kit tree crowding a lamp post (within 1.8 m, the kit jitters them): from most angles the post hides the trunk
  // and the canopy floats on top of it like a ball on a pole, so drop it
  const posts = (layout.items || []).filter((i) => i.asset === 'post' && i.pos).map((i) => [i.pos[0], i.pos[i.pos.length - 1]]);
  if (posts.length) {
    kit.dressing.updateMatrixWorld(true);
    const m4 = new THREE.Matrix4(), tp = new THREE.Vector3();
    kit.dressing.traverse((o) => {
      if (!o.isInstancedMesh || !/tree/.test(String((o.parent && (o.parent.userData.libraryId || o.parent.name)) || ''))) return;
      for (let n = 0; n < o.count; n++) {
        o.getMatrixAt(n, m4);
        tp.setFromMatrixPosition(new THREE.Matrix4().multiplyMatrices(o.matrixWorld, m4));
        if (!posts.some(([x, z]) => Math.hypot(x - tp.x, z - tp.z) < 1.8)) continue;
        const saved = m4.clone();
        o.setMatrixAt(n, new THREE.Matrix4().makeTranslation(0, -400, 0));
        o.instanceMatrix.needsUpdate = true;
        const id = n;
        undo.push(() => { o.setMatrixAt(id, saved); o.instanceMatrix.needsUpdate = true; });
      }
    });
  }
  hideFiller(kit, spec, layout, actorsOut, propsOut, hide);
  // every walker walks the whole shot and passes everyone else clear: the subject is settled first, then the rest
  const order = [...people].sort((a, b) => (b.object === (subject && subject.object)) - (a.object === (subject && subject.object)));
  settleWalkers(order.map((x) => x.object).filter((o) => !(o.parent && o.parent.userData && o.parent.userData.cruise)), kit, spec.duration_s || 4);
  return { root, subject, actors: actorsOut, props: propsOut, kit, layout, undo, stats: sceneStats(root), ...(stagedPair ? { stagedPair } : {}), ...(balcony ? { balcony } : {}), ...(shotCam ? { shotCam } : {}) };
}

/** A camera car: the moves that ride with a driving car, the words that drive it, and its speed (m/s, about 50 km/h). */
export const CAMERA_CAR_MOVES = ['track', 'follow'];
export const DRIVES = /\b(driv(e|es|ing)|speed(s|ing)|cruis(e|es|ing)|rac(es|ing))\b/i;
export const CAMERA_CAR_SPEED = 14;
/** How far across the road (m from the car's line) the camera car can ride: kit posts nearer than this step aside. */
export const CAMERA_CAR_REACH = 12;

/** Kit dressing that is filler: parked vehicles, bikes and road or shop signs (lamp posts, trees and buildings are the set). */
export const FILLER_ASSETS = ['car', 'suv', 'truck', 'van', 'bus', 'bicycle', 'sign'];
/** Kit extras (a shopper at the gas station): background people and animals the sentence never asked for. */
export const FILLER_EXTRAS = ['person', 'dog', 'cat'];
const WANTS_EXTRAS = /\b(crowds?|crowded|pedestrians?|passers?-?by|bystanders?|onlookers?|shoppers?|busy|people around|other people|background people)\b/i;
const WANTS_PARKED = /\b(parking (lot|garage)|car park|parked (cars?|vehicles?)|car lot|traffic|rush hour|busy|dealership|used car)\b/i;
/** Kit trees stay only when the sentence places them (or the place is a forest, where the trees are the place). */
export const WANTS_TREES = /\b(trees?|tree-lined|treeline|forests?|woods|woodland|orchard|grove|jungle|pines?|palms?|oaks?|birch(es)?|willows?)\b/i;
export const TREE_KITS = ['forest'];

/**
 * The layout sent to the render is a depth and colour guide: a parked van or a road sign in it comes back as a van or a
 * sign the user never asked for. So the kit's parked vehicles, bikes and signs step aside unless the sentence asks for
 * them (a parking lot, a busy street, traffic); the kit's background people and animals likewise (a crowd, passers-by),
 * and the kit's trees unless the sentence places them (trees, woods, a grove) or the place is a forest: a ring of kit trees
 * round a lake came back as grey balls over the water. Anything the shot drives (a chase's traffic) or anyone sits in or on stays.
 * The sentence's own vehicles and props are clones in the shot group, never kit dressing, so they always stay.
 * Applies to the editor and the export alike, so what the editor shows is what is sent. Restored on the next build.
 */
function hideFiller(kit, spec, layout, actorsOut, propsOut, hide) {
  if (layout.id === 'car_interior') return; // the car is the set
  const said = String(spec.text || '');
  const parked = WANTS_PARKED.test(said), trees = WANTS_TREES.test(said) || TREE_KITS.includes(layout.id);
  const used = [...actorsOut, ...propsOut].map((x) => x.object).filter(Boolean);
  const holds = (c) => used.some((o) => { for (let n = o; n; n = n.parent) if (n === c) return true; return false; });
  kit.dressing.children.forEach((c) => {
    if (!c.visible || c.userData.drive) return;
    const id = String(c.userData.asset || c.userData.libraryId || String(c.name || '').replace(/^inst_/, '')).split('__')[0];
    const filler = (FILLER_ASSETS.includes(id) && !parked) || (FILLER_EXTRAS.includes(id) && !WANTS_EXTRAS.test(said) && !parked) || (id === 'tree' && !trees);
    if (!filler || holds(c)) return;
    hide(c);
  });
}

/** The road nearest the stage: its centre point there, the road direction and the direction across it. */
function crossingOf(layout, S) {
  let best = null;
  for (const g of layout.geometry || []) {
    if (g.type !== 'road' || !g.points) continue;
    for (let k = 0; k + 1 < g.points.length; k++) {
      const a = new THREE.Vector3(g.points[k][0], 0, g.points[k][1]);
      const b = new THREE.Vector3(g.points[k + 1][0], 0, g.points[k + 1][1]);
      const ab = b.clone().sub(a);
      const u = Math.max(0, Math.min(1, S.clone().setY(0).sub(a).dot(ab) / Math.max(1e-6, ab.lengthSq())));
      const c = a.clone().addScaledVector(ab, u);
      const d = c.distanceTo(S.clone().setY(0));
      if (!best || d < best.d) {
        const along = ab.normalize();
        best = { d, center: c, along, across: new THREE.Vector3(-along.z, 0, along.x), width: g.width || 8 };
      }
    }
  }
  return best;
}

/** Floor box inside the room walls (wall_ meshes), or null on an open set. */
function roomBox(kit) {
  const box = new THREE.Box3();
  let any = false;
  kit.set.traverse((n) => { if (n.isMesh && /^wall_(back|left|right|front)$/.test(n.name || '')) { box.union(new THREE.Box3().setFromObject(n)); any = true; } });
  if (!any || box.isEmpty()) return null;
  box.min.y = 0; box.max.y = 0;
  return box;
}

function waterSpot(layout, S) {
  const w = (layout.geometry || []).find((g) => g.type === 'water');
  if (!w) return null;
  if (w.pos) return new THREE.Vector3(w.pos[0], 0, w.pos[1]).lerp(S, 0.35).setY(0);
  if (w.points) { const m = w.points[Math.floor(w.points.length / 2)]; return new THREE.Vector3(m[0], 0, m[1]); }
  return null;
}

/** Low-detail instanced figures for big crowds (yellow body, black joints), about 600 triangles each. */
export function crowdFigures(matrices) {
  const group = new THREE.Group(); group.name = 'crowd_figures';
  const body = new THREE.MeshStandardMaterial({ name: 'mat_body', color: 0xF4FF75, roughness: 0.55 });
  const joint = new THREE.MeshStandardMaterial({ name: 'mat_joint', color: 0x1A1A1A, roughness: 0.4 });
  const parts = [
    [new THREE.CapsuleGeometry(0.17, 0.42, 3, 10), body, [0, 1.2, 0], [1, 1, 0.62]],
    [new THREE.SphereGeometry(0.115, 12, 8), body, [0, 1.62, 0], [0.86, 1.1, 0.95]],
    [new THREE.CapsuleGeometry(0.065, 0.74, 2, 8), body, [0.1, 0.47, 0], [1, 1, 1]],
    [new THREE.CapsuleGeometry(0.065, 0.74, 2, 8), body, [-0.1, 0.47, 0], [1, 1, 1]],
    [new THREE.CapsuleGeometry(0.05, 0.56, 2, 8), body, [0.23, 1.12, 0], [1, 1, 1]],
    [new THREE.CapsuleGeometry(0.05, 0.56, 2, 8), body, [-0.23, 1.12, 0], [1, 1, 1]],
    [new THREE.SphereGeometry(0.06, 10, 6), joint, [0.21, 1.42, 0], [1, 1, 1]],
    [new THREE.SphereGeometry(0.06, 10, 6), joint, [-0.21, 1.42, 0], [1, 1, 1]],
  ];
  const disposables = [body, joint];
  const m = new THREE.Matrix4(), local = new THREE.Matrix4();
  for (const [g, mat, p, s] of parts) {
    const im = new THREE.InstancedMesh(g, mat, matrices.length);
    local.compose(new THREE.Vector3(...p), new THREE.Quaternion(), new THREE.Vector3(...s));
    matrices.forEach((x, i) => im.setMatrixAt(i, m.multiplyMatrices(x, local)));
    im.castShadow = true; im.receiveShadow = true; im.computeBoundingSphere();
    group.add(im); disposables.push(g);
  }
  return { group, disposables };
}

/** Triangles, draw calls (approximate: one per mesh / instanced mesh), meshes and instances under a root. */
export function sceneStats(root) {
  let tris = 0, draws = 0, instances = 0;
  root.traverseVisible((o) => {
    if (!o.isMesh) return;
    const g = o.geometry, t = (g.index ? g.index.count : g.attributes.position.count) / 3;
    const n = o.isInstancedMesh ? o.count : 1;
    tris += t * n; draws += 1; instances += n;
  });
  return { triangles: Math.round(tris), drawCalls: draws, instances };
}
void setMaterials; void rigOf;
