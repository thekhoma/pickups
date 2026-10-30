// What a previz frame actually shows: a grid of rays from the camera, the first thing each one hits. Rule-based, no model.
// Used for frame 0 (the start frame is made from what's in it and nothing else) and to say why a frame came out blank.
import * as THREE from "three";

/** Plain words for the scene's pieces, from their names (kit geometry types, asset ids, item ids). */
const WORDS: [RegExp, string][] = [
  [/^(person|people|man|woman|kid|child|sitter|walker|rower|runner|jogger|dancer|crowd|extra|pedestrian|driver)s?$/, "people"],
  [/^(water|lake|sea|ocean|river|pond|canal)$/, "water"],
  [/^(tree|trees|oak|pine|palm|birch)$/, "trees"],
  [/^(bush|hedge|shrub)s?$/, "hedges"],
  [/^(planter|plantpot)s?$/, "a planter"],
  [/^(box|crate)(es)?$/, "boxes"],
  [/^(wall|parapet)s?$/, "walls"],
  [/^(grass|lawn)$/, "grass"],
  [/^(path|trail|footpath)s?$/, "a path"],
  [/^(road|street|lane|highway)s?$/, "the road"],
  [/^(sidewalk|pavement|kerb|curb)s?$/, "the pavement"],
  [/^(building|house|shop|shopfront|tower|block|facade)s?$/, "buildings"],
  [/^(car|van|truck|suv|vehicle|taxi|bus)s?$/, "cars"],
  [/^(boat|rowboat|ship|yacht)s?$/, "a boat"],
  [/^(bench)(es)?$/, "a bench"],
  [/^(rock|stone|boulder)s?$/, "rocks"],
  [/^(fence|railing|rail)s?$/, "a fence"],
  [/^(lamp|streetlight|lamppost)s?$/, "street lamps"],
  [/^(post|pole|bollard|sign|mailbox|hydrant)s?$/, "a post"],
  [/^(table)s?$/, "a table"],
  [/^(chair|stool)s?$/, "chairs"],
  [/^(desk)s?$/, "a desk"],
  [/^(mountain|hill)s?$/, "mountains"],
  [/^(dune|sand|desert)s?$/, "sand dunes"],
  [/^(bridge)s?$/, "a bridge"],
  [/^(dog)s?$/, "a dog"],
  [/^(cat)s?$/, "a cat"],
  [/^(bike|bicycle)s?$/, "a bike"],
  [/^(tunnel)s?$/, "a tunnel"],
  [/^(snowcap|snow)$/, "snow"],
];
/** Names that are only scaffolding: never a thing in the picture. */
const SKIP = /^(ground|sky|handles|knots|set|items|dash|kit|root|scene|duneWind|duneLee|duneCrest)$/i;

function tokens(name: string): string[] {
  return name
    .replace(/^(geo|inst|kit)_/i, "")
    .split(/[_\-\s.]+|(?<=[a-z])(?=[A-Z])/)
    .map((t) => t.toLowerCase().replace(/\d+$/, ""))
    .filter(Boolean);
}

/** The plain word for one object's name, or null. People win over places ("lake_sitter" is a person). */
export function wordFor(name: string): string | null {
  if (!name || /^kit_/.test(name) || SKIP.test(name.replace(/^(geo|inst)_/i, ""))) return null;
  const ts = tokens(name);
  const people = WORDS[0];
  if (ts.some((t) => people[0].test(t))) return people[1];
  for (const t of ts) for (const [re, word] of WORDS.slice(1)) if (re.test(t)) return word;
  return null;
}

/** The word for a mesh: the nearest ancestor (itself first) whose name has one. */
function wordOf(o: THREE.Object3D | null, stop: Set<THREE.Object3D>): string | null {
  for (let p = o; p; p = p.parent) {
    // The kit's own groups ("kit_lake", "set", "items") name the place, not the thing that was hit.
    if (stop.has(p) || /^(kit_.*|set|items)$/.test(p.name)) return null;
    const w = wordFor(p.name);
    if (w) return w;
  }
  return null;
}

export interface FrameView {
  /** Share of the frame each word covers (first hit per ray). */
  share: Record<string, number>;
  /** Where each word sits across the frame: the mean of its rays, 0 left to 1 right. */
  x: Record<string, number>;
  /** Words in the scene that no ray reached: out of frame or hidden. */
  outside: string[];
  /** Main subjects (the sentence's actors) that rays reached, by index. */
  actors: number[];
  /** Share of rays that hit something within 0.35 m of the lens: the camera is inside or against an object. */
  blocked: number;
  /** What most of those near rays hit: "the subject" (a sentence's actor), a word, or null. */
  near: string | null;
  /** Share of rays that hit only bare ground or nothing. */
  bare: number;
  /** The thing the camera sits inside (a hedge, a wall), or null. Rays from inside a mesh don't see it, so this is a box test. */
  inside: string | null;
}

/** Cast `cols` x `rows` rays through the frame and sort what they hit. The camera must be updated for the frame. */
export function frameView(scene: THREE.Scene, camera: THREE.PerspectiveCamera, actors: THREE.Object3D[], cols = 40, rows = 22): FrameView {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const stop = new Set<THREE.Object3D>([scene]);
  const ray = new THREE.Raycaster();
  ray.near = 0.01;
  ray.far = camera.far;
  const counts: Record<string, number> = {};
  const xs: Record<string, number> = {};
  const seenActors = new Set<number>();
  let blocked = 0;
  let bare = 0;
  const nearBy: Record<string, number> = {};
  const targets: THREE.Object3D[] = [];
  scene.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && o.visible) targets.push(o);
  });
  const actorOf = (o: THREE.Object3D | null): number => {
    for (let p = o; p; p = p.parent) {
      const i = actors.indexOf(p);
      if (i >= 0) return i;
    }
    return -1;
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ray.setFromCamera(new THREE.Vector2(((c + 0.5) / cols) * 2 - 1, 1 - ((r + 0.5) / rows) * 2), camera);
      const hit = ray.intersectObjects(targets, false).find((h) => isVisibleChain(h.object));
      if (!hit) {
        bare++;
        continue;
      }
      const a = actorOf(hit.object);
      if (hit.distance < 0.35) {
        blocked++;
        const w = a >= 0 ? "the subject" : wordOf(hit.object, stop);
        if (w) nearBy[w] = (nearBy[w] ?? 0) + 1;
      }
      if (a >= 0) {
        seenActors.add(a);
        continue;
      }
      const w = wordOf(hit.object, stop);
      if (w) {
        counts[w] = (counts[w] ?? 0) + 1;
        xs[w] = (xs[w] ?? 0) + (c + 0.5) / cols;
      } else bare++;
    }
  }
  const n = cols * rows;
  const present = new Set<string>();
  for (const o of targets) {
    if (actorOf(o) >= 0) continue;
    const w = wordOf(o, stop);
    if (w) present.add(w);
  }
  const share: Record<string, number> = {};
  const x: Record<string, number> = {};
  for (const [w, k] of Object.entries(counts)) {
    share[w] = Math.round((k / n) * 1000) / 1000;
    x[w] = Math.round((xs[w] / k) * 100) / 100;
  }
  return {
    inside: insideOf(targets, camera.position, stop, actorOf),
    share,
    x,
    outside: [...present].filter((w) => !(w in share)).sort(),
    actors: [...seenActors].sort((a, b) => a - b),
    blocked: Math.round((blocked / n) * 1000) / 1000,
    near: Object.entries(nearBy).sort((p, q) => q[1] - p[1])[0]?.[0] ?? null,
    bare: Math.round((bare / n) * 1000) / 1000,
  };
}

/** The word for the smallest object whose box holds the point (instanced pieces one by one). */
function insideOf(targets: THREE.Object3D[], at: THREE.Vector3, stop: Set<THREE.Object3D>, actorOf: (o: THREE.Object3D) => number): string | null {
  let best: { vol: number; word: string } | null = null;
  const box = new THREE.Box3();
  const m = new THREE.Matrix4();
  const size = new THREE.Vector3();
  const consider = (word: string | null) => {
    if (!word || word === "water" || !box.containsPoint(at)) return;
    const vol = box.getSize(size).x * size.y * size.z;
    if (!best || vol < best.vol) best = { vol, word };
  };
  for (const o of targets) {
    if (actorOf(o) >= 0) continue;
    const word = wordOf(o, stop);
    if (!word) continue;
    const mesh = o as THREE.Mesh;
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    const local = mesh.geometry.boundingBox;
    if (!local) continue;
    const inst = o as THREE.InstancedMesh;
    if (inst.isInstancedMesh) {
      for (let i = 0; i < inst.count; i++) {
        inst.getMatrixAt(i, m);
        box.copy(local).applyMatrix4(m.premultiply(inst.matrixWorld));
        consider(word);
      }
    } else {
      box.copy(local).applyMatrix4(mesh.matrixWorld);
      consider(word);
    }
  }
  return (best as { vol: number; word: string } | null)?.word ?? null;
}

/** Why a frame that looks blank is blank, in one line with what to change; null when the frame has something in it. */
export function blankReason(v: FrameView | null, frame: number): string | null {
  if (!v) return `Frame ${frame} came out blank. Check the preview, then try Make shot again.`;
  const the = (w: string) => (/^(a|the) /.test(w) ? w : `the ${w}`);
  if (v.inside) return `Frame ${frame}: the camera is inside ${the(v.inside)}. Move the camera out of it, then Make shot again.`;
  if (v.blocked >= 0.4) {
    const what = v.near ? the(v.near) : "something";
    return what === "the subject"
      ? `Frame ${frame}: the subject comes right up against the lens. Move the camera back, then Make shot again.`
      : `Frame ${frame}: ${what} is right up against the lens. Move the camera away from it, then Make shot again.`;
  }
  // A subject or set pieces in view: flat ground filling most of it (a desert under a crane) isn't blank.
  if (v.actors.length || Object.values(v.share).some((x) => x >= 0.01)) return null;
  return `Frame ${frame} shows only empty ground or sky. Aim the camera at the subject, then Make shot again.`;
}

function isVisibleChain(o: THREE.Object3D): boolean {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return false;
  return true;
}
