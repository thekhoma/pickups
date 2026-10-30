// Type declarations for the Previz runtime (plain ES modules for three r170). Each runtime/<module>.d.ts re-exports
// from here, so editors and TypeScript projects get types for `import ... from './runtime/<module>.js'`.
import type { Object3D, Group, Box3, Vector3, PerspectiveCamera, Scene, WebGLRenderer, LoadingManager, Material, Matrix4 } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ---------------------------------------------------------------- shot spec (parser -> builder contract, SHOT_SPEC.md)
export type RelationType = 'in' | 'on' | 'rides' | 'walks' | 'holds' | 'near' | 'facing' | 'with' | 'pair';
/** library v1.4 two-person pair, self-contained: poses for a and b, b's offset in a's frame (metres, +Z = where a faces), b's turn in degrees */
export interface PairSpec { id: 'hug' | 'kiss' | 'shake_hands' | 'fight' | 'couple' | string; poses: [string, string]; offset: [number, number, number]; facing: number }
export type NearSide = 'beside' | 'front' | 'behind' | 'across' | 'against' | 'around' | 'under' | 'along' | 'toward';
export interface Relation {
  type: RelationType;
  /** id of the target actor ('a1') or prop ('p1') */
  target: string;
  /** 'in' a vehicle: one seat per counted person (null = no free seat, stands beside it) */
  seats?: (string | null)[];
  poses?: string[];
  /** 'on' a bench or sofa: slot index per counted person (3 per piece) */
  slots?: number[];
  side?: NearSide;
  /** 'holds': 'L', 'foot_R', 'hands', 'strap', v1.4 'push' (cart) and 'airborne' (kite on a line); default right hand */
  hand?: 'L' | 'R' | 'foot_R' | 'hands' | 'strap' | 'push' | 'airborne';
  /** 'pair' (v1.4): the target actor is b of this pair */
  pair?: PairSpec;
}
export interface Actor {
  id: string;
  kind: 'person' | 'animal';
  asset: string;
  word: string;
  count: number;
  pose: string;
  preset?: 'adult' | 'child';
  presets?: ('adult' | 'child')[];
  gender?: 'male' | 'female';
  genders?: (string | null)[];
  size?: number;
  formation?: 'facing';
  implied?: boolean;
  role?: string;
  statue?: boolean;
  /** v1.4: one actor with count 2 is a pair (objects 0 and 1) */
  pair?: PairSpec;
  relations: Relation[];
}
export interface Prop {
  id: string;
  kind: 'vehicle' | 'rideable' | 'furniture' | 'handheld' | 'set';
  asset: string;
  word: string;
  count: number;
  scale?: number | [number, number, number];
  size?: number;
  implied?: boolean;
  relations?: Relation[];
}
export interface CameraSpec {
  /** actor or prop id the camera frames (the first-mentioned person) */
  subject: string | null;
  shot: 'extreme_close' | 'close' | 'medium' | 'full' | 'wide' | null;
  angle: 'low' | 'high' | 'overhead' | 'behind' | 'side' | 'front' | 'over_shoulder' | 'pov' | 'through_windshield' | null;
  move: 'push_in' | 'pull_out' | 'orbit' | 'pan' | 'pan_left' | 'pan_right' | 'track' | 'follow' | 'crane_up' | 'crane_down' | 'crane_reveal' | 'dolly_along' | 'dolly_zoom' | 'boom_up' | 'boom_down' | 'rise' | 'tilt_up' | 'tilt_down' | 'truck_left' | 'truck_right' | 'aerial' | 'body_mount' | 'fpv' | 'swoop' | 'kit' | 'static' | null;
  /** set when the move starts at the interior kit's doorway camera */
  from?: 'doorway' | null;
  facing?: 'toward' | 'away' | 'across' | 'left_to_right' | null;
  lensMm?: number | null;
  dutch?: number | null;
  subjectSpeed?: number | null;
  reveal?: boolean;
  handheld?: { seed: number; pos_m: number; rot_deg: number; hz: number } | null;
  segments?: { move: string; from_t: number; to_t: number; params?: Record<string, string | number | boolean>; tracking: number }[];
}
export interface ShotSpec {
  version: 1;
  text: string;
  /** FNV-1a hash of the normalized text: same sentence, same scene */
  seed: number;
  place: { id: string; word: string | null; source: 'word' | 'implied' | 'default'; /** v1.4: the word names a place with no kit of its own yet; id is the nearest kit */ standin?: boolean };
  /** time of day and weather words (rule table grammar.world_words) */
  world: WorldSpec;
  actors: Actor[];
  props: Prop[];
  camera: CameraSpec;
  matched: { text: string; as: string }[];
  /** words the kit has nothing for (show "No X in the kit") */
  unmatched: string[];
  notes: string[];
  beats?: { actor?: string; type: string; from_t: number; to_t: number; params?: Record<string, string | number | boolean> }[];
}

// ---------------------------------------------------------------- loader.js
export const PERSON_TIERS: { hero: 'person'; lite: 'person__lite'; crowd: 'person__crowd' };
export const CORE_MODELS: string[];
export interface Library {
  url(id: string): string;
  load(id: string): Promise<Object3D>;
  create(id: string): Promise<Object3D>;
  /** synchronous clone of a loaded model (shares geometry and materials) */
  instance(id: string): Object3D;
  preload(ids: string[]): Promise<Object3D[]>;
  isLoaded(id: string): boolean;
  template(id: string): Object3D | null;
  loadedIds(): string[];
  timings: Record<string, number>;
  gltfLoader: GLTFLoader;
}
export function createLibrary(options?: { baseUrl?: string; manager?: LoadingManager }): Library;
export function prepareTemplate(scene: Object3D, id: string): Object3D;

// ---------------------------------------------------------------- poser.js
export interface PoseData { [joint: string]: [number, number, number] }
export function registerPoses(json: object): void;
export function registerAnimalPoses(json: object): void;
export function poseIds(): string[];
export function resolvePoseId(id: string): string | null;
export function getPoseData(id: string): PoseData | null;
export function frozenVehiclePose(kind: string, scaleKey: 'adult' | 'child', key: string): PoseData | null;
export function frozenRider(kind: string, scaleKey: 'adult' | 'child'): object | null;
export function rigOf(person: Object3D): Object3D;
export function jointsOf(obj: Object3D): Record<string, Object3D>;
export function attachPoint(obj: Object3D, name: string): Object3D | null;
export function setPersonScale(person: Object3D, s: number): Object3D;
export function scaleKey(person: Object3D): 'adult' | 'child' | null;
/** degrees, Euler XYZ per joint, clamped to userData.limits; re-grounds the feet unless ground is false */
export function setPose(person: Object3D, pose: string | PoseData, options?: { ground?: boolean; clamp?: boolean }): Object3D;
export function groundObject(obj: Object3D, lift?: number, mover?: Object3D | null): void;
export function setAnimalPose(animal: Object3D, pose: string): Object3D;
/** animal kinds with a pose table (v1.4: dog, cat, cow, giraffe, elephant, bird, horse) */
export function animalKinds(): string[];
/** v1.4 pair table from person_poses.json */
export function getPair(id: string): { a: string; b: string; offset: [number, number, number]; facing: number } | null;
export function pairIds(): string[];
export function mirrorPose(p: PoseData): PoseData;

// ---------------------------------------------------------------- attach.js
export const SEAT_ALIASES: Record<string, string>;
export const VEHICLES: string[];
export const GUITAR_MOUNT: { pos: [number, number, number]; rot: [number, number, number] };
export function setBookOpen(book: Object3D, deg: number): Object3D;
export function attachActionProp(person: Object3D, slot: 'hand_L' | 'hand_R' | 'hands' | 'foot_L' | 'foot_R' | 'strap' | 'push' | 'airborne', prop: Object3D, options?: { upright?: boolean }): Object3D;
/** v1.4: poses a and b and puts b at the pair offset in a's frame (pair id from person_poses.json pairs, or a PairSpec) */
export function placePair(pair: string | PairSpec, a: Object3D, b: Object3D): [Object3D, Object3D];
export const KITE_MOUNT: { pos: [number, number, number]; rot: [number, number, number] };
export function attachProp(person: Object3D, prop: Object3D, point?: string, options?: { upright?: boolean }): Object3D;
export function placeOnTop(base: Object3D, obj: Object3D, offset?: [number, number]): Object3D;
export function seatOn(seat: Object3D, person: Object3D, options?: { slot?: number | string | null; pose?: string; point?: string }): Object3D;
export function attachRider(vehicle: Object3D, person: Object3D, options?: object): Object3D;
export function vehicleSeats(vehicle: Object3D): string[];
/** frozen in-vehicle poses only; needs the interior GLB (not __exterior) */
export function seatInVehicle(person: Object3D, vehicle: Object3D, seat?: string, pose?: string | null, options?: { litInside?: boolean }): Object3D;
export function makeLeash(fromObj: Object3D, fromPoint: string, toObj: Object3D, toPoint: string, options?: { sag?: number; material?: Material | null; radius?: number }): Object3D;
export function updateLeash(mesh: Object3D): Object3D;

// ---------------------------------------------------------------- lexicon.js
export const CATEGORY: Record<string, Prop['kind'] | 'person' | 'animal'>;
export const SEATING: string[];
export const SURFACES: string[];
export interface LexEntry { kind: string; [k: string]: unknown }
export interface Lexicon {
  words: Record<string, LexEntry>;
  verbs: Record<string, LexEntry>;
  phrases: { tokens: string[]; text: string; entry: LexEntry }[];
  lemma(word: string): string;
  vocab: Set<string>;
  maxPhrase: number;
}
export function tokenize(text: string): string[];
export function makeLemmatizer(vocab: Set<string>, exceptions?: Record<string, string>): (w: string) => string;
export function manifestVerb(word: string, action: object): LexEntry;
export function buildLexicon(manifest: object, grammar: object): Lexicon;

// ---------------------------------------------------------------- grammar.js
export const SPEC_VERSION: 1;
export function hashSeed(text: string): number;
export function tagTokens(text: string, lex: Lexicon): { text: string; lemma: string; entry: LexEntry | null; phrase?: boolean; plural?: boolean }[];
/** pure: the same text always gives the same spec */
export function parseShot(text: string, lex: Lexicon, grammar: object): ShotSpec;

// ---------------------------------------------------------------- kit.js
export interface KitLayout { id: string; label: string; move?: string; world?: object; geometry: object[]; items: object[]; links?: object[]; cameras?: object; /** a wall with a window opening: normal points into the room, `at` is its inner face along the normal, span runs along the wall */ windowWall?: { normal: [number, number]; at: number; span: [number, number]; y: [number, number] } }
export interface KitStage { stage: { pos: [number, number]; rot: number; y: number }; vehicleStage: { pos: [number, number]; rot: number; kitItem: string | null } }
export interface KitOptions { /** v1.4: items with subjectSwap matching this subject's pose or props are left out (BuiltKit.dropped) */ subject?: { pose?: string; props?: string[] }; skipActors?: boolean; instancing?: boolean; personTier?: 'hero' | 'lite' | 'crowd'; vehicleLOD?: boolean; farDist?: number; maxCrowd?: number }
export interface BuiltKit extends KitStage {
  root: Group; set: Group; dressing: Group; byId: Record<string, Object3D>; cameras: object; world: object; dropped: string[]; windowWall: KitLayout['windowWall'] | null;
  stats: { instanced: number; cloned: number; instancedModels: number; farDist: number; crowdSkipped: number };
}
export const ACTOR_ASSETS: string[];
export const INSTANCE_ASSETS: string[];
export const FAR_ASSETS: string[];
export const SET_COLORS: Record<string, number>;
export function setMaterials(): Record<string, Material>;
export function modelIdFor(item: object, personTier?: string, vehicleLOD?: boolean, far?: boolean): string;
export function referencedIds(layout: KitLayout): Set<string>;
export function kitModelIds(layout: KitLayout, options?: KitOptions): string[];
export function isCrowdItem(item: object): boolean;
export function kitStage(layout: KitLayout): KitStage;
export function swapMatches(swap: string, subject: { pose?: string; props?: string[] | Record<string, string> } | null): boolean;
export function instanceModel(template: Object3D, matrices: Matrix4[], name: string): Group;
export function buildKit(layout: KitLayout, lib: Library, options?: KitOptions): BuiltKit;

// ---------------------------------------------------------------- world.js
export const WORLD: Record<string, unknown>;
export interface WorldSpec { time: 'day' | 'golden' | 'dusk' | 'night' | null; weather: 'clear' | 'overcast' | 'rain' | 'fog' | 'snow' | null }
export const LOOKS: { time: Record<string, object>; weather: Record<string, object> };
export function setupRenderer(renderer: WebGLRenderer, options?: { quality?: 'fast' | 'high' }): WebGLRenderer;
export interface World { key: Object3D; hemi: Object3D; sky: Object3D; apply(kitWorld?: object, look?: Partial<WorldSpec>): void }
export function createWorld(renderer: WebGLRenderer, scene: Scene, options?: { quality?: 'fast' | 'high'; environment?: boolean }): World;
export function disposeEnvironment(renderer: WebGLRenderer): void;

// ---------------------------------------------------------------- builder.js
export const BUILD_DEFAULTS: { subjectTier: 'hero' | 'lite'; liteCount: number; crowdCount: number; instancing: boolean };
export interface BuiltItem { id: string; index: number; kind: string; object: Object3D; related?: boolean }
export interface BuiltShot {
  root: Group;
  subject: { id: string; kind: string; object: Object3D; yaw: number; vehicle: Object3D | null } | null;
  actors: BuiltItem[]; props: BuiltItem[];
  kit: BuiltKit; layout: KitLayout; undo: (() => void)[];
  stats: { triangles: number; drawCalls: number; instances: number };
  timing: { loadMs: number; buildMs: number };
}
export function mulberry32(seed: number): () => number;
export function shotModelIds(spec: ShotSpec, layout: KitLayout | null, options?: Partial<typeof BUILD_DEFAULTS>): string[];
export function createShotBuilder(o: { lib: Library; getLayout: (id: string) => Promise<KitLayout> }): {
  build(spec: ShotSpec, options?: Partial<typeof BUILD_DEFAULTS>): Promise<BuiltShot>;
  teardown(): void;
  readonly current: BuiltShot | null;
  kits: Map<string, BuiltKit>;
};
export function crowdFigures(matrices: Matrix4[]): Group;
export function sceneStats(root: Object3D): { triangles: number; drawCalls: number; instances: number };

// ---------------------------------------------------------------- camera.js
export interface CamState { pos: [number, number, number]; target: [number, number, number]; fov: number; t?: number; focal_mm?: number }
export const SHOT_HEIGHT: Record<string, number | null>;
export function frameShot(built: BuiltShot, spec: ShotSpec, options?: { aspect?: number; fov?: number; obstacles?: Object3D[] }): { start: CamState; end: CamState; mid?: CamState; keys?: CamState[]; label: string };
export function fitDistance(box: Box3, target: Vector3, dir: Vector3, fov: number, aspect: number, margin?: number): number;
export function applyCamera(camera: PerspectiveCamera, a: CamState, b: CamState, t?: number): void;

// ---------------------------------------------------------------- previz.js
export interface Previz {
  scene: Scene; camera: PerspectiveCamera; world: World | null; lib: Library; lexicon: Lexicon;
  data: { manifest: object; grammar: object; poses: object };
  parse(text: string): ShotSpec;
  show(textOrSpec: string | ShotSpec, options?: Partial<typeof BUILD_DEFAULTS> & { move?: CameraSpec['move']; shot?: CameraSpec['shot']; angle?: CameraSpec['angle']; world?: Partial<WorldSpec> }): Promise<{ spec: ShotSpec; built: BuiltShot; cams: { start: CamState; end: CamState; label: string }; timing: { parseMs: number; loadMs: number; buildMs: number; totalMs: number } }>;
  /** camera along the shot's move, t = 0 start, 1 end */
  frame(t?: number): void;
  getLayout(id: string): Promise<KitLayout>;
  readonly shot: BuiltShot | null;
  readonly cams: { start: CamState; end: CamState; label: string } | null;
}
export function createPreviz(options?: { renderer?: WebGLRenderer; baseUrl?: string; quality?: 'fast' | 'high'; preload?: string[]; environment?: boolean }): Promise<Previz>;

// ---------------------------------------------------------------- wordlog.js
export interface WordLogEntry { word: string; count: number; sentences: string[]; last: number }
export interface WordLog { record(spec: ShotSpec): boolean; entries(): WordLogEntry[]; top(n?: number): WordLogEntry[]; exportJSON(): string; clear(): void }
export function createWordLog(options?: { storage?: Storage | null; max?: number }): WordLog;
