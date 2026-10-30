// The director's previs library. Off, or before a file has loaded, the gray stand-ins in primitives.ts stay.
import * as THREE from "three";
import type { PrevizSpec } from "./spec";
import { CORE_MODELS, createLibrary } from "./runtime/loader.js";

type Library = ReturnType<typeof createLibrary>;
import { registerAnimalPoses, registerPoses, resolvePoseId } from "./runtime/poser.js";

/** Flip this off and every kit word goes back to the procedural mesh. `?library=0` does the same for one tab. */
export const USE_LIBRARY = true;

/** Compressed bundle. Step 1's GLBs stay at /previz/<id>.glb and are not what this loads. */
export const LIBRARY_BASE = "/previz/assets/";

const KIND_GLB: Record<string, string> = {
  dog: "dog",
  cat: "cat",
  car: "car",
  bike: "bicycle",
  skate: "skateboard",
  boat: "boat",
};

const PROP_GLB: Record<string, string> = {
  bench: "bench",
  tree: "tree",
  trees: "tree",
  house: "house",
  fountain: "fountain",
  table: "table",
  building: "building",
  block: "building",
  rock: "rock",
  post: "post",
  cup: "cup",
  phone: "phone",
};

const TREE_PLACES = new Set(["park", "suburbs", "forest", "lake", "river"]);
const HOUSE_PLACES = new Set(["suburbs", "house", "outskirts"]);

/** Names the current compiler emits that the bundle aliases table does not list. */
const POSE_GAP: Record<string, string> = {
  stand_sip: "sip",
  track: "walk",
  walk_slow: "walk",
  park: "stand",
  spin: "stand",
};

export const previzTimes = { firstMs: 0, coldLoadMs: 0, lastLoadMs: 0, lastBuildMs: 0, rebuildMs: 0, boot: 0 };

const sizes = new Map<string, THREE.Vector3>();
let lib: Library | null = null;
let poses: Promise<void> | null = null;
let personQueue: string[] = [];

export function libraryOn(): boolean {
  if (!USE_LIBRARY) return false;
  if (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("library") === "0") return false;
  return true;
}

function books(): Library {
  if (!lib) lib = createLibrary({ baseUrl: LIBRARY_BASE });
  return lib;
}

/** Close-up and extreme close use the hero mesh. Everything else uses the lite mesh. */
export function heroPersonId(spec: PrevizSpec): "person" | "person__lite" {
  const notes = (spec.notes || "").toLowerCase();
  const close = /\bextreme[\s-]?close\b/.test(notes) || /\bclose[\s-]?up\b/.test(notes) || /\bclose shot\b/.test(notes);
  return close ? "person" : "person__lite";
}

function peopleIn(spec: PrevizSpec): string[] {
  const ids: string[] = [];
  const hero = heroPersonId(spec);
  const next = () => ids.push(ids.length === 0 ? hero : "person__crowd");
  for (const subject of spec.subjects) {
    if (subject.kind === "humanoid") next();
    else if (subject.kind === "bike" && (subject.anim === "track" || subject.id === "moto")) next();
    else if (subject.kind === "skate" && subject.anim === "track") next();
    else if (subject.kind === "boat" && subject.anim !== "park") next();
  }
  return ids;
}

/** Reset which person mesh the next figure takes. Called at the start of every scene build. */
export function beginShot(spec: PrevizSpec): void {
  personQueue = peopleIn(spec);
}

/** First person is the subject. Later people are extras. */
export function claimPerson(): string {
  return personQueue.shift() ?? "person__crowd";
}

/** Current animation name, through the bundle aliases (lay, sit_sip, phone) and the few names the compiler still emits. */
export function poseNameFor(anim: string): string {
  const base = anim.endsWith("_follow") ? anim.slice(0, -"_follow".length) : anim;
  return resolvePoseId(base) ?? resolvePoseId(POSE_GAP[base] ?? "stand") ?? "stand";
}

export function assetsFor(spec: PrevizSpec): string[] {
  const ids = new Set<string>(peopleIn(spec));
  const props = spec.environment.props;
  const place = props[0] ?? "park";
  if (place === "city") ids.add("building");
  if (TREE_PLACES.has(place)) ids.add("tree");
  if (HOUSE_PLACES.has(place)) ids.add("house");
  for (const prop of props) {
    const id = PROP_GLB[prop];
    if (id) ids.add(id);
  }
  for (const subject of spec.subjects) {
    const id = KIND_GLB[subject.kind];
    if (id) ids.add(id);
    if (subject.anim.includes("sip")) ids.add("cup");
    if (subject.anim.includes("phone")) ids.add("phone");
  }
  return [...ids];
}

export function shotLibrary(): Library {
  return books();
}

export function ensurePoses(): Promise<void> {
  return registerBundle();
}

/** The shared set, fetched once. A kit's own models wait until a sentence picks that kit. */
export function preloadCore(): Promise<void> {
  if (!libraryOn()) return Promise.resolve();
  return ensureLibrary([...CORE_MODELS]);
}

function registerBundle(): Promise<void> {
  if (!poses) {
    poses = (async () => {
      const [people, animals] = await Promise.all([
        fetch("/previz/data/person_poses.json").then((res) => {
          if (!res.ok) throw new Error("person poses " + res.status);
          return res.json() as Promise<object>;
        }),
        fetch("/previz/data/animal_poses.json").then((res) => {
          if (!res.ok) throw new Error("animal poses " + res.status);
          return res.json() as Promise<object>;
        }),
      ]);
      registerPoses(people);
      registerAnimalPoses(animals);
    })();
    poses.catch(() => {
      poses = null;
    });
  }
  return poses;
}

function remember(id: string): void {
  if (sizes.has(id)) return;
  const root = books().template(id);
  if (!root) return;
  root.updateWorldMatrix(true, true);
  sizes.set(id, new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3()));
}

/** Fetch each GLB once. Later scenes clone the cached scene graph. createLibrary sets MeshoptDecoder on its loader. */
export function ensureLibrary(ids: string[]): Promise<void> {
  if (!libraryOn() || ids.length === 0) return Promise.resolve();
  const started = performance.now();
  return registerBundle()
    .then(() => books().preload(ids))
    .then(() => {
      for (const id of ids) remember(id);
      previzTimes.lastLoadMs = performance.now() - started;
      if (!previzTimes.coldLoadMs) previzTimes.coldLoadMs = previzTimes.lastLoadMs;
    });
}

/** A clone of a cached GLB, or null when the flag is off or the file is not in yet. */
export function takeModel(id: string): THREE.Group | null {
  if (!libraryOn()) return null;
  const book = books();
  if (!book.isLoaded(id)) return null;
  remember(id);
  const clone = book.instance(id) as THREE.Group;
  if (id.startsWith("person")) clone.userData.asset = "person";
  else if (id === "dog" || id === "cat") clone.userData.asset = id;
  return clone;
}

export function modelSize(id: string): THREE.Vector3 | null {
  const size = sizes.get(id);
  return size ? size.clone() : null;
}
