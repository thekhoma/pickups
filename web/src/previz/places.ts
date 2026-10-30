// Gray places for a described shot. The first prop is the place. The rest are things the sentence named.
import * as THREE from "three";
import { travelMetres } from "./animate";
import { bench, bicycle, boat, building, car, cat, dog, fountain, house, laneLine, person, road, skateboard, table, tree, type Actor, type Mats } from "./primitives";
import { mulberry32 } from "./seed";
import type { PrevizSpec } from "./spec";
import type { Dress } from "./templates";

const PLACES = new Set(["park", "suburbs", "house", "interior", "city", "outskirts", "desert", "forest", "ocean", "lake", "river"]);

/** A short thin arrow, plus a wider invisible grab so it stays easy to catch. */
function axisArrow(axis: "x" | "y" | "z", color: number, length: number): THREE.Group {
  const g = new THREE.Group();
  const shaftLen = length * 0.66;
  const headLen = length * 0.2;
  const gap = length * 0.16;
  const radius = Math.max(0.005, length * 0.028);
  const show = new THREE.MeshBasicMaterial({ color, depthTest: false });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, shaftLen, 6), show);
  const head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.2, headLen, 8), show);
  shaft.position.y = gap + shaftLen / 2;
  head.position.y = gap + shaftLen + headLen / 2;
  const grab = new THREE.Mesh(
    new THREE.CylinderGeometry(length * 0.1, length * 0.1, length, 6),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthTest: false }),
  );
  grab.position.y = length * 0.5;
  g.add(shaft, head, grab);
  if (axis === "x") g.rotation.z = -Math.PI / 2;
  if (axis === "z") g.rotation.x = Math.PI / 2;
  g.traverse((obj) => {
    obj.userData.axis = axis;
    obj.renderOrder = 3;
  });
  return g;
}

function axisHandles(height: number): THREE.Group {
  const length = Math.min(0.48, Math.max(0.18, height * 0.32));
  const g = new THREE.Group();
  g.name = "handles";
  g.userData.overlay = true; // editor-only: hidden in every exported frame
  g.visible = false;
  g.userData.length = length;
  g.add(axisArrow("x", 0xf4f1ea, length), axisArrow("y", 0xffffff, length), axisArrow("z", 0xcfcfcf, length));
  return g;
}

function box(mats: Mats, w: number, h: number, d: number, tone: "gray" | "dark" | "light" = "gray"): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mats[tone]);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function addTrees(api: Dress, mats: Mats, rng: () => number, n: number, radius: number, skipFront: boolean) {
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + rng() * 0.2;
    const dist = radius * (0.75 + rng() * 0.45);
    const x = Math.sin(a) * dist;
    const z = Math.cos(a) * dist;
    if (skipFront && (Math.cos(a) > 0.12 || (Math.abs(x) < 3.2 && z < 2))) continue;
    const t = tree(mats, 4 + rng() * 3.5);
    t.position.set(x, 0, z);
    api.add(t);
  }
}

function dressPlace(place: string, mats: Mats, api: Dress, rng: () => number, trees = true) {
  if (place === "park") {
    const path = box(mats, 1.8, 0.06, 14, "dark");
    path.position.set(0, 0.03, 3);
    api.add(path);
    if (trees) addTrees(api, mats, rng, 8, 9, true);
  } else if (place === "suburbs") {
    const street = road(mats, 80, 8);
    street.position.z = 8;
    api.add(street);
    for (let i = 0; i < 4; i++) {
      const h = house(mats, 7, 5, 6);
      h.position.set(i % 2 === 0 ? -10 : 10, 0, i * 14 - 8);
      h.rotation.y = i % 2 === 0 ? Math.PI / 2 : -Math.PI / 2;
      api.add(h);
    }
    if (trees) addTrees(api, mats, rng, 4, 14, false);
  } else if (place === "house") {
    const home = house(mats, 9, 6.5, 8);
    home.position.set(0, 0, -8);
    api.add(home);
    const path = box(mats, 1.4, 0.05, 8, "dark");
    path.position.set(0, 0.03, -3);
    api.add(path);
    const lawn = box(mats, 18, 0.04, 10, "gray");
    lawn.position.set(0, 0.02, -2);
    api.add(lawn);
  } else if (place === "interior") {
    const floor = box(mats, 8, 0.08, 8, "dark");
    floor.position.y = 0.04;
    api.add(floor);
    const wall = (w: number, h: number, d: number, x: number, y: number, z: number) => {
      const m = box(mats, w, h, d, "gray");
      m.position.set(x, y, z);
      api.add(m);
    };
    wall(8, 3.1, 0.12, 0, 1.55, -3.9);
    wall(0.12, 3.1, 8, -3.9, 1.55, 0);
    wall(0.12, 3.1, 8, 3.9, 1.55, 0);
    wall(2.4, 3.1, 0.12, -2.6, 1.55, 3.9);
    wall(2.4, 3.1, 0.12, 2.6, 1.55, 3.9);
    const window = box(mats, 1.6, 1.1, 0.08, "light");
    window.position.set(0, 1.8, -3.82);
    api.add(window);
    const sofa = box(mats, 2.2, 0.7, 0.8, "light");
    sofa.position.set(-1.6, 0.4, -2.6);
    api.add(sofa);
    const table = box(mats, 1.1, 0.42, 0.7, "dark");
    table.position.set(0.2, 0.25, -0.4);
    api.add(table);
  } else if (place === "outskirts") {
    const street = road(mats, 120, 7);
    street.position.z = 16;
    api.add(street);
    const far = house(mats, 8, 5, 7);
    far.position.set(-16, 0, 28);
    api.add(far);
    for (let i = 0; i < 5; i++) {
      const pole = box(mats, 0.15, 4, 0.15, "dark");
      pole.position.set(6, 2, i * 16 - 4);
      api.add(pole);
    }
  } else if (place === "desert") {
    const sand = box(mats, 80, 0.08, 80, "light");
    sand.position.y = 0.02;
    api.add(sand);
    for (let i = 0; i < 5; i++) {
      const dune = new THREE.Mesh(new THREE.SphereGeometry(4 + rng() * 4, 12, 8), mats.light);
      dune.scale.y = 0.28;
      dune.position.set((rng() - 0.5) * 36, 0.4, 8 + rng() * 28);
      dune.castShadow = true;
      dune.receiveShadow = true;
      api.add(dune);
    }
    for (let i = 0; i < 4; i++) {
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.4 + rng() * 0.5), mats.dark);
      rock.position.set((rng() - 0.5) * 10, 0.3, (rng() - 0.4) * 8);
      rock.castShadow = true;
      api.add(rock);
    }
  } else if (place === "forest") {
    addTrees(api, mats, rng, 16, 11, false);
    for (let i = 0; i < 5; i++) {
      const trunk = tree(mats, 7 + rng() * 3);
      trunk.position.set((i % 2 === 0 ? -1 : 1) * (2.2 + rng()), 0, 3 + i * 2.4);
      api.add(trunk);
    }
  } else if (place === "ocean") {
    const beach = box(mats, 40, 0.1, 16, "light");
    beach.position.set(0, 0.04, -4);
    api.add(beach);
    const water = box(mats, 80, 0.12, 70, "dark");
    water.position.set(0, 0.02, 28);
    api.add(water);
    for (let i = 0; i < 4; i++) {
      const wave = box(mats, 18, 0.08, 0.4, "light");
      wave.position.set((i % 2 === 0 ? -4 : 5), 0.12, 8 + i * 6);
      api.add(wave);
    }
  } else if (place === "lake") {
    const water = new THREE.Mesh(new THREE.CircleGeometry(14, 28), mats.dark);
    water.rotation.x = -Math.PI / 2;
    water.position.set(0, 0.05, 8);
    water.receiveShadow = true;
    api.add(water);
    if (trees) addTrees(api, mats, rng, 5, 12, true);
  } else if (place === "river") {
    const water = box(mats, 6, 0.08, 70, "dark");
    water.position.set(2.5, 0.04, 8);
    api.add(water);
    const bank = box(mats, 3, 0.12, 70, "light");
    bank.position.set(-2.2, 0.04, 8);
    api.add(bank);
    if (trees) addTrees(api, mats, rng, 6, 10, true);
  }
}

/** Avenue, sidewalk, alley, or plaza. The sidewalk keeps the person off the lanes. */
function dressCity(kind: string, mats: Mats, api: Dress, rng: () => number): void {
  if (kind === "sidewalk") {
    const pavement = box(mats, 3.4, 0.12, 80, "light");
    pavement.position.set(-4.2, 0.08, 8);
    api.add(pavement);
    const curb = box(mats, 0.28, 0.16, 80, "gray");
    curb.position.set(-2.4, 0.1, 8);
    api.add(curb);
    const street = road(mats, 80, 8);
    street.position.set(2.6, 0, 8);
    api.add(street);
    for (let i = 0; i < 8; i++) {
      const line = laneLine(mats);
      line.position.set(2.6, 0.1, i * 9 - 8);
      api.add(line);
      const lamp = box(mats, 0.12, 3.2, 0.12, "dark");
      lamp.position.set(-5.6, 1.6, i * 10 - 6);
      api.add(lamp);
    }
    for (let i = 0; i < 5; i++) {
      const near = building(mats, 7, 8 + rng() * 10, 10);
      near.position.set(-11, 0, i * 16 - 8);
      api.add(near);
      const far = building(mats, 8, 6 + rng() * 8, 10);
      far.position.set(12, 0, i * 16 - 4);
      api.add(far);
    }
    return;
  }
  if (kind === "alley") {
    const ground = box(mats, 4.2, 0.08, 70, "dark");
    ground.position.set(0, 0.04, 6);
    api.add(ground);
    for (let i = 0; i < 5; i++) {
      for (const side of [-1, 1]) {
        const wall = building(mats, 5, 7 + rng() * 6, 12);
        wall.position.set(side * 4.6, 0, i * 14 - 6);
        api.add(wall);
      }
    }
    return;
  }
  if (kind === "plaza") {
    const square = box(mats, 36, 0.08, 36, "light");
    square.position.set(0, 0.04, 2);
    api.add(square);
    for (let i = 0; i < 4; i++) {
      const block = building(mats, 8, 8 + rng() * 8, 8);
      block.position.set(-12 + i * 8, 0, 22);
      api.add(block);
    }
    return;
  }
  const street = road(mats, 90, 12);
  street.position.z = 10;
  api.add(street);
  for (let i = 0; i < 10; i++) {
    const line = laneLine(mats);
    line.position.set(0, 0.1, i * 8 - 6);
    api.add(line);
  }
  for (let i = 0; i < 6; i++) {
    for (const side of [-1, 1]) {
      const h = 14 + rng() * 28;
      const block = building(mats, 10, h, 12);
      block.position.set(side * 12, 0, i * 16 - 10);
      api.add(block);
    }
  }
}

function gaitMetres(anim: string): number {
  if (anim.startsWith("run")) return 22;
  if (anim.startsWith("jog")) return 14;
  if (anim.startsWith("walk")) return 9;
  return 0;
}

/** The sentence places trees (runtime/builder.js WANTS_TREES, the shot path's own rule). */
const WANTS_TREES = /\b(trees?|tree-lined|treeline|forests?|woods|woodland|orchard|grove|jungle|pines?|palms?|oaks?|birch(es)?|willows?)\b/i;

export function dressDescribed(spec: PrevizSpec, mats: Mats, api: Dress): void {
  const rng = mulberry32(spec.seed >>> 0);
  const props = spec.environment.props;
  const place = PLACES.has(props[0] ?? "") ? props[0] : "park";
  const extra = new Set(props.slice(1));
  // R2-2: an existing layout read back (a shot made before the shot path, an old shot's frame 0) hides its kit trees
  // too, unless the sentence places them: a grey tree ball came back as a ball in the render. The forest keeps its trees.
  const trees = place === "forest" || WANTS_TREES.test(spec.notes ?? "");
  if (place === "city") {
    const kind = extra.has("sidewalk") ? "sidewalk" : extra.has("alley") ? "alley" : extra.has("plaza") ? "plaza" : "avenue";
    dressCity(kind, mats, api, rng);
  } else dressPlace(place, mats, api, rng, trees);
  if (extra.has("house") && place !== "house") {
    const home = house(mats, 8, 6, 7);
    home.position.set(place === "suburbs" ? -8 : -7, 0, place === "suburbs" ? -2 : -9);
    api.add(home);
  }
  if (extra.has("trees") && place !== "forest") addTrees(api, mats, rng, 3, 7, true);
  if (extra.has("bench")) api.add(bench(mats));
  if (extra.has("fountain")) {
    const piece = fountain(mats);
    piece.position.set(2.4, 0, -1.6);
    api.add(piece);
  }
  if (extra.has("table")) {
    const piece = table(mats);
    piece.position.set(-1.7, 0, 0.2);
    api.add(piece);
  }
  if (extra.has("post")) {
    const piece = box(mats, 0.28, 1.3, 0.28, "dark");
    piece.position.set(2.2, 0.65, 1.1);
    api.add(piece);
  }
  if (extra.has("rock")) {
    const piece = new THREE.Mesh(new THREE.DodecahedronGeometry(0.45), mats.dark);
    piece.position.set(1.6, 0.28, -0.6);
    piece.castShadow = true;
    api.add(piece);
  }

  spec.subjects.forEach((s, index) => {
    const riding = s.kind === "car" || s.kind === "boat" || s.kind === "bike" || s.kind === "skate";
    const metres = riding && (s.anim === "track" || s.anim === "drive") ? (s.kind === "boat" ? 18 : 36) : gaitMetres(s.anim);
    const actor: Actor =
      s.kind === "humanoid"
        ? person(s.height_m, mats, s.anim)
        : s.kind === "dog"
          ? dog(s.height_m, mats, s.anim)
          : s.kind === "cat"
            ? cat(s.height_m, mats, s.anim)
            : s.kind === "car"
              ? car(s.height_m, mats, metres)
              : s.kind === "bike"
                ? bicycle(s.height_m, mats, s.anim === "track" || s.id === "moto", s.id === "moto")
                : s.kind === "skate"
                  ? skateboard(s.height_m, mats, s.anim === "track")
                  : s.kind === "boat"
                    ? boat(mats, s.anim !== "park")
                    : person(s.height_m, mats, "stand");
    actor.home = [s.pos[0], s.pos[1], s.pos[2]];
    actor.origin = [s.pos[0], s.pos[1], s.pos[2]];
    const handles = axisHandles(s.height_m);
    handles.position.y = Math.max(0.12, s.height_m * 0.5);
    actor.handles = handles;
    actor.object.add(handles);
    actor.object.position.set(s.pos[0], s.pos[1], s.pos[2]);
    const inner = actor.update.bind(actor);
    const follow = s.anim === "track" || s.anim.endsWith("_follow");
    actor.update = (t, speed) => {
      inner(t, speed);
      const along = metres > 0 ? travelMetres(t, speed, metres) : 0;
      const origin = actor.origin ?? actor.home ?? s.pos;
      actor.object.position.set(origin[0], origin[1], origin[2] + along);
      actor.travel = follow ? along : 0;
    };
    api.actors[index] = actor;
    api.add(actor.object);
  });
}
