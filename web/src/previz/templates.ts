// Five presets. Each one is a spec plus the props that spec names.
import * as THREE from "three";
import { travelMetres } from "./animate";
import { building, car, cloud, house, humanoid, laneLine, pedestal, product, ring, road, tree, type Actor, type Mats } from "./primitives";
import { mulberry32 } from "./seed";
import type { PrevizSpec, TemplateId, Vec3 } from "./spec";
import { PREVIZ_FPS, PREVIZ_H, PREVIZ_W } from "./spec";

const base = {
  version: 1 as const,
  fps: PREVIZ_FPS as 24,
  size: [PREVIZ_W, PREVIZ_H] as [1920, 1088],
  lens: { focal_mm: 24, film_gauge_mm: 36 as const },
};

/** 97 frames: one pass of Previz to final on a 24 GB card, and under the cap on a bigger one. */
const DURATION = 97 / PREVIZ_FPS;

function key(t: number, pos: Vec3, look: Vec3, focal?: number): PrevizSpec["camera"]["keys"][number] {
  return focal ? { t, pos, look, focal_mm: focal } : { t, pos, look };
}

export function templateSpec(id: TemplateId): PrevizSpec {
  switch (id) {
    case "giant_orbit":
      return {
        ...base,
        template: id,
        duration_s: DURATION,
        lens: { focal_mm: 24, film_gauge_mm: 36 },
        environment: { ground: "plane", sky: "gradient", fog: [80, 2200], props: ["houses_small", "trees", "clouds"] },
        subjects: [{ id: "giant", kind: "humanoid", height_m: 300, pos: [0, 0, 0], anim: "walk_slow" }],
        camera: {
          interp: "catmull_rom",
          ease: "in_out",
          keys: [key(0, [3, 1, 4], [0, 2, 0], 24), key(1, [350, 200, 350], [0, 150, 0], 35)],
        },
        seed: 1,
        speed: 1,
        notes: "A giant walks through a small town. The camera starts on a foot and pulls wide until the figure fills the sky. Houses and clouds show the scale. Overcast daylight.",
      };
    case "street_run":
      return {
        ...base,
        template: id,
        duration_s: DURATION,
        lens: { focal_mm: 20, film_gauge_mm: 36 },
        environment: { ground: "none", sky: "gradient", fog: [30, 260], props: ["road", "buildings", "lane_lines", "lights"] },
        subjects: [{ id: "car", kind: "car", height_m: 1.45, pos: [1.6, 0, 28], anim: "drive" }],
        camera: {
          interp: "catmull_rom",
          ease: "linear",
          keys: [key(0, [0, 1.1, 0], [0, 1.15, 18], 20), key(1, [0, 1.25, 170], [0, 1.3, 196], 20)],
        },
        seed: 2,
        speed: 1,
        notes: "A low camera races down a city street between tall buildings. Lane lines and lights stream past. Late afternoon.",
      };
    case "tunnel":
      return {
        ...base,
        template: id,
        duration_s: DURATION,
        lens: { focal_mm: 18, film_gauge_mm: 36 },
        environment: { ground: "none", sky: "gradient", fog: [6, 90], props: ["rings"] },
        subjects: [{ id: "car", kind: "car", height_m: 1.4, pos: [0, 0, 22], anim: "drive" }],
        camera: {
          interp: "catmull_rom",
          ease: "linear",
          keys: [key(0, [0, 1.5, 0], [0, 1.5, 12], 18), key(1, [0, 1.5, 150], [0, 1.5, 168], 18)],
        },
        seed: 3,
        speed: 1,
        notes: "The camera flies through a ring tunnel. Edge lights streak past. The car stays ahead.",
      };
    case "car_track":
      return {
        ...base,
        template: id,
        duration_s: DURATION,
        lens: { focal_mm: 35, film_gauge_mm: 36 },
        environment: { ground: "none", sky: "gradient", fog: [20, 400], props: ["road", "trees", "lane_lines"] },
        subjects: [{ id: "car", kind: "car", height_m: 1.5, pos: [0, 0, 0], anim: "track" }],
        camera: {
          interp: "catmull_rom",
          ease: "in_out",
          keys: [key(0, [6.2, 1.7, -2.2], [0, 0.7, 0.4], 35), key(1, [2.4, 1.15, 1.6], [0, 0.65, 2.4], 50)],
        },
        seed: 4,
        speed: 1,
        notes: "A car drives along an open road. The camera tracks alongside, then pushes in. Daylight.",
      };
    case "product_orbit":
      return {
        ...base,
        template: id,
        duration_s: DURATION,
        lens: { focal_mm: 50, film_gauge_mm: 36 },
        environment: { ground: "plane", sky: "gradient", fog: [8, 40], props: ["pedestal"] },
        subjects: [{ id: "object", kind: "product", height_m: 0.42, pos: [0, 1.05, 0], anim: "spin" }],
        camera: {
          interp: "catmull_rom",
          ease: "in_out",
          keys: [key(0, [2.3, 1.7, 0.15], [0, 1.15, 0], 50), key(1, [-1.15, 1.28, 0.35], [0, 1.12, 0], 75)],
        },
        seed: 5,
        speed: 1,
        notes: "A plain object sits on a pedestal. The camera orbits halfway around and pushes in. Soft studio light.",
      };
  }
}

export interface Dress {
  add(obj: THREE.Object3D): void;
  actors: Actor[];
}

/** Props for this template, plus one actor per subject. Positions are metres. */
export function dress(spec: PrevizSpec, mats: Mats, api: Dress): void {
  const rng = mulberry32(spec.seed >>> 0);
  const props = new Set(spec.environment.props);
  if (props.has("road")) {
    const strip = road(mats, spec.template === "street_run" ? 420 : 220, spec.template === "street_run" ? 14 : 10);
    strip.position.z = spec.template === "street_run" ? 180 : 40;
    api.add(strip);
  }
  if (props.has("lane_lines")) {
    const n = spec.template === "street_run" ? 48 : 24;
    for (let i = 0; i < n; i++) {
      const line = laneLine(mats);
      line.position.set(0, 0.1, i * 8);
      api.add(line);
    }
  }
  if (props.has("buildings")) {
    for (let i = 0; i < 16; i++) {
      for (const side of [-1, 1]) {
        const h = 16 + rng() * 48;
        const block = building(mats, 12, h, 14);
        block.position.set(side * (11 + rng() * 3), 0, i * 18 - 4);
        api.add(block);
        const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 0.45), mats.emit);
        lamp.position.set(side * 7.2, 3.2 + rng(), i * 18);
        api.add(lamp);
      }
    }
  }
  if (props.has("houses_small")) {
    for (let i = 0; i < 10; i++) {
      const ang = rng() * Math.PI * 2;
      const dist = 40 + rng() * 90;
      const h = house(mats, 6 + rng() * 4, 5 + rng() * 3, 6);
      h.position.set(Math.cos(ang) * dist, 0, Math.sin(ang) * dist);
      h.rotation.y = rng() * Math.PI;
      api.add(h);
    }
    const scale = humanoid(1.75, mats, false);
    scale.object.position.set(18, 0, 12);
    api.add(scale.object);
  }
  if (props.has("trees")) {
    const n = spec.template === "giant_orbit" ? 14 : 18;
    for (let i = 0; i < n; i++) {
      const t = tree(mats, spec.template === "giant_orbit" ? 8 + rng() * 6 : 5 + rng() * 3);
      if (spec.template === "car_track") t.position.set((i % 2 === 0 ? -1 : 1) * (8 + rng() * 4), 0, i * 12 - 10);
      else {
        const ang = rng() * Math.PI * 2;
        const dist = 25 + rng() * 110;
        t.position.set(Math.cos(ang) * dist, 0, Math.sin(ang) * dist);
      }
      api.add(t);
    }
  }
  if (props.has("clouds")) {
    const h = spec.subjects[0]?.height_m ?? 100;
    for (let i = 0; i < 6; i++) {
      const c = cloud(mats, 8 + rng() * 10);
      c.position.set((rng() - 0.5) * h, h * (0.55 + rng() * 0.35), (rng() - 0.5) * h);
      api.add(c);
    }
  }
  if (props.has("rings")) {
    for (let i = 0; i < 32; i++) {
      const hoop = ring(mats, 3.4);
      hoop.position.set(0, 2.4, i * 5.5);
      api.add(hoop);
      for (const x of [-3.3, 3.3]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.28, 4.8, 0.28), mats.dark);
        post.position.set(x, 2.4, i * 5.5);
        post.castShadow = true;
        api.add(post);
      }
    }
  }
  if (props.has("pedestal")) {
    api.add(pedestal(mats, spec.subjects[0]?.pos[1] ?? 1));
  }
  if (props.has("lights") && !props.has("buildings")) {
    for (let i = 0; i < 12; i++) {
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), mats.emit);
      lamp.position.set((i % 2 === 0 ? -1 : 1) * 4, 3, i * 12);
      api.add(lamp);
    }
  }

  spec.subjects.forEach((s, index) => {
    const moving = s.anim === "walk_slow";
    const metres = s.anim === "drive" || s.anim === "track" ? (spec.template === "tunnel" ? 90 : 70) : 0;
    const actor =
      s.kind === "humanoid" ? humanoid(s.height_m, mats, moving) : s.kind === "car" ? car(s.height_m, mats, metres) : product(s.height_m, mats);
    actor.object.position.set(s.pos[0], s.pos[1], s.pos[2]);
    if (s.anim !== "spin") actor.update = wrap(actor, s.pos[2], s.anim === "track");
    api.actors[index] = actor;
    api.add(actor.object);
  });
}

function wrap(actor: Actor, baseZ: number, track: boolean): Actor["update"] {
  const inner = actor.update.bind(actor);
  const metres = actor.travel;
  return (t, speed) => {
    inner(t, speed);
    const z = travelMetres(t, speed, metres);
    actor.object.position.z = baseZ + z;
    actor.travel = track ? z : 0;
  };
}
