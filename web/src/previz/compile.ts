// A sentence becomes a shot spec. Words hit a catalog; placement is rules. Same sentence, same spec.
import { scan } from "./lexicon.ts";
import { readShot, rememberPick, shotSpecOn } from "./shotLex.ts";
import type { Prop, ShotSpec } from "./runtime/types";
import { PREVIZ_DEFAULT_S, type PrevizSpec, type Vec3 } from "./spec.ts";

const W = 1920;
const H = 1088;

export interface Compiled {
  spec: PrevizSpec;
  title: string;
  matched: string[];
  missed: string[];
  camera: string;
  /** Full parser output when USE_SHOT_SPEC is on. Counts, pairs, relations and world stay here for later steps. */
  shot?: ShotSpec;
}

export interface PresetLine {
  id: string;
  label: string;
  text: string;
}

export const PRESETS: PresetLine[] = [
  {
    id: "park",
    label: "Park",
    text: "A park with a bench and a person sitting at the bench. The camera is pushing in from behind and rotates around until we see them in front as they take a sip of coffee from their coffee cup.",
  },
  {
    id: "suburbs",
    label: "Suburbs",
    text: "A car parked in the suburbs in front of a house. The camera tracks alongside and pushes in.",
  },
  {
    id: "house",
    label: "House",
    text: "A person standing in front of a house. The camera pushes in.",
  },
  {
    id: "interior",
    label: "Interior",
    text: "Inside a house, a person sitting. The camera pushes in from the doorway.",
  },
  {
    id: "city",
    label: "City",
    text: "A person walking in the city. The camera follows from behind.",
  },
  {
    id: "outskirts",
    label: "Outskirts",
    text: "A car on the outskirts. The camera circles around to the front.",
  },
  {
    id: "desert",
    label: "Desert",
    text: "A person standing in the desert. The camera pulls wide.",
  },
  {
    id: "forest",
    label: "Forest",
    text: "A dog in the forest. The camera pushes in.",
  },
  {
    id: "ocean",
    label: "Ocean",
    text: "A person standing at the ocean. The camera moves in from behind.",
  },
  {
    id: "lake",
    label: "Lake",
    text: "A cat by the lake. The camera orbits around.",
  },
  {
    id: "river",
    label: "River",
    text: "A person walking by the river. The camera tracks alongside.",
  },
];

const PLACES: { id: string; label: string; phrases: string[] }[] = [
  { id: "interior", label: "Interior", phrases: ["inside the house", "inside a house", "in the house", "in a house", "living room", "kitchen", "bedroom", "indoors", "interior", "inside"] },
  { id: "suburbs", label: "Suburbs", phrases: ["suburbs", "suburb", "suburban", "neighborhood", "neighbourhood"] },
  { id: "outskirts", label: "Outskirts", phrases: ["outskirts", "edge of town"] },
  { id: "desert", label: "Desert", phrases: ["desert", "dunes"] },
  { id: "forest", label: "Forest", phrases: ["forest", "woods", "woodland"] },
  { id: "ocean", label: "Ocean", phrases: ["ocean", "the sea", "at sea", "beach", "shore"] },
  { id: "lake", label: "Lake", phrases: ["lake", "lakeside"] },
  { id: "river", label: "River", phrases: ["river", "riverside", "creek", "stream"] },
  { id: "city", label: "City", phrases: ["city", "downtown", "skyline", "urban", "alley"] },
  { id: "house", label: "House", phrases: ["house", "home"] },
  { id: "park", label: "Park", phrases: ["park", "garden"] },
];

/**
 * Run 3 (the director, R&D's (d)): the place when the words name none the parser knows. The chip shows no place until the user's
 * words imply one (first match wins); never Park unless they say park, lawn, bench or path. An empty place lets the start
 * frame take the ground from the previz objects and the Look words.
 */
const INFER_PLACE: [RegExp, string][] = [
  [/\b(tracks?|farmhouses?|farms?)\b/i, "Country"],
  [/\b(sidewalks?|streets?)\b/i, "Street"],
  [/\b(roofs?|rooftops?)\b/i, "Rooftop"],
  [/\b(shores?|lakes?)\b/i, "Lake"],
  [/\b(booths?|diners?)\b/i, "Diner"],
  [/\b(parks?|lawns?|benche?s?|paths?)\b/i, "Park"],
];

export function inferPlace(text: string): string | null {
  return INFER_PLACE.find(([re]) => re.test(text))?.[1] ?? null;
}

/** The place said on the chip and sent with frame 0: the parser's own unless it only fell back to its default. */
export function shownPlace(text: string, kitLabel: string | undefined, source: string | null | undefined): string | undefined {
  return source === "default" ? (inferPlace(text) ?? undefined) : kitLabel;
}

const STOP = new Set(
  `a an the of and with to from in on at by for as is are was were be their them they we he she his her our my it its into then while over some there this that just really very kind like slowly quickly gently soft shot scene camera move moves moving look looking see until from their take takes taking cup`.split(
    /\s+/,
  ),
);

const IGNORE = new Set(
  "night day sunny sun rain raining morning evening dusk dawn cinematic beautiful nice big small light lights today outside around front back behind doorway down off onto toward towards across through along next under over into pushing push pushes pushed dolly closer rotating rotates rotate rotation orbit orbits orbiting circling circles circle tracking tracks track follows follow alongside parallel pulling pulls pull wide sip sips sipping drink drinks drinking sitting sits sit seated standing stands stand walking walks walk jogging jogs jog running runs run sprint sprints parked driving drives drive shoulder until them see their medium aerial drone overhead slowly starts start out".split(
    /\s+/,
  ),
);

function has(text: string, pattern: RegExp): boolean {
  return pattern.test(text);
}

function r3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function vec(x: number, y: number, z: number): Vec3 {
  return [r3(x), r3(y), r3(z)];
}

function clampNum(n: number | undefined, min: number, max: number, fallback: number): number {
  if (n == null || !Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function polar(target: Vec3, angle: number, dist: number, camY: number): Vec3 {
  return vec(target[0] + Math.sin(angle) * dist, camY, target[2] + Math.cos(angle) * dist);
}

type Move = "around" | "push-behind" | "push" | "orbit" | "track" | "parallel" | "pull" | "hold" | "close" | "wide" | "close-wide" | "drone" | "fpv" | "reveal";
type ShotSize = "close" | "medium" | "wide" | "aerial";

export const CAMERA_PRESETS = [
  { id: "close", label: "Close-up" },
  { id: "wide", label: "Wide" },
  { id: "close-wide", label: "Close to wide" },
  { id: "dolly-in", label: "Push in" },
  { id: "dolly-out", label: "Pull back" },
  { id: "orbit", label: "Orbit" },
  { id: "drone", label: "Drone" },
  { id: "fpv", label: "Chase from behind" },
  { id: "track", label: "Tracking" },
  { id: "parallel", label: "Parallel" },
  { id: "locked", label: "Still camera" },
] as const;

export type CameraPick = (typeof CAMERA_PRESETS)[number]["id"];

const PICK_MOVE: Record<CameraPick, { move: Move; label: string }> = {
  close: { move: "close", label: "Close-up" },
  wide: { move: "wide", label: "Wide" },
  "close-wide": { move: "close-wide", label: "Close to wide" },
  "dolly-in": { move: "push", label: "Push in" },
  "dolly-out": { move: "pull", label: "Pull back" },
  orbit: { move: "orbit", label: "Orbit" },
  drone: { move: "drone", label: "Drone" },
  fpv: { move: "fpv", label: "Chase from behind" },
  track: { move: "track", label: "Tracking" },
  parallel: { move: "parallel", label: "Parallel" },
  locked: { move: "hold", label: "Still camera" },
};

const SIZE_NAME: Record<ShotSize, string> = { close: "Close-up", medium: "Medium", wide: "Wide", aerial: "aerial" };
const SIZE_ORDER: ShotSize[] = ["close", "medium", "wide", "aerial"];

/** Shot sizes in the order they were said. "Wide aerial" is one aerial, not a wide and then an aerial. */
function sizeHits(text: string): { size: ShotSize; at: number }[] {
  const found: { size: ShotSize; at: number }[] = [];
  const add = (size: ShotSize, re: RegExp) => {
    for (const hit of text.matchAll(re)) if (hit.index != null) found.push({ size, at: hit.index });
  };
  add("aerial", /\b(aerials?|drone|overhead|birds eye|birdseye|from above)\b/g);
  add("close", /\b(close-up|closeup|close up)\b/g);
  add("medium", /\b(medium|mid shot|waist up)\b/g);
  add("wide", /\bwide\b/g);
  found.sort((a, b) => a.at - b.at);
  const collapsed: { size: ShotSize; at: number }[] = [];
  for (const hit of found) {
    const prev = collapsed[collapsed.length - 1];
    if (prev && hit.at - prev.at < 18 && ((prev.size === "wide" && hit.size === "aerial") || (prev.size === "aerial" && hit.size === "wide"))) {
      prev.size = "aerial";
      continue;
    }
    collapsed.push({ ...hit });
  }
  return collapsed;
}

/** "Starts medium and pulls out into a wide aerial" is a move from one size to another. */
function sizeArc(text: string): { from: ShotSize; to: ShotSize } | null {
  const hits = sizeHits(text);
  if (hits.length >= 2 && hits[0].size !== hits[hits.length - 1].size) return { from: hits[0].size, to: hits[hits.length - 1].size };
  const pull = /\bpull\w*\b|\bwiden\w*\b|\bcrane\b/.test(text);
  const push = /\b(push\w*|dolly in|moves? in|moving in)\b/.test(text);
  if (hits.length === 1 && pull) {
    const to = hits[0].size;
    const from = SIZE_ORDER[Math.max(0, SIZE_ORDER.indexOf(to) - 1)];
    if (from && from !== to) return { from, to };
  }
  if (hits.length === 1 && push) {
    const from = hits[0].size;
    const to = SIZE_ORDER[Math.max(0, SIZE_ORDER.indexOf(from) - 1)];
    if (to && to !== from) return { from, to };
  }
  return null;
}

function sizePose(size: ShotSize, lookY: number): { dist: number; camY: number; focal: number } {
  if (size === "close") return { dist: 1.7, camY: lookY, focal: 65 };
  if (size === "medium") return { dist: 2.55, camY: lookY + 0.04, focal: 50 };
  if (size === "wide") return { dist: 9, camY: lookY + 1.8, focal: 28 };
  return { dist: 12, camY: lookY + 9, focal: 24 };
}

function cameraMove(text: string): { move: Move; label: string; from?: ShotSize; to?: ShotSize } {
  const arc = sizeArc(text);
  if (arc && !/\bfpv\b/.test(text)) {
    const to = arc.to === "aerial" ? "aerial" : SIZE_NAME[arc.to].toLowerCase();
    return { move: "reveal", label: `${SIZE_NAME[arc.from]} to ${to}`, from: arc.from, to: arc.to };
  }
  const behind = has(text, /\b(from behind|behind them|behind him|behind her|behind the|from the back|over the shoulder|over their shoulder|from the doorway|doorway)\b/) || has(text, /\bbehind\b/);
  const toFront = has(text, /\b(in front|to the front|see them|see him|see her|their face|his face|her face|until we see)\b/);
  const orbit = has(text, /\b(rotat\w*|orbit\w*|circl\w*|turns around|turn around|goes around|come around|comes around)\b/) || (has(text, /\baround\b/) && (behind || toFront));
  const push = has(text, /\b(push\w*|dolly|closer|moves? in|moving in|comes? in)\b/);
  const pull = has(text, /\b(pull\w*|wide shot|widen|crane)\b/);
  const parallel = has(text, /\b(parallel|alongside|side on)\b/);
  const track = has(text, /\b(track\w*|follows|follow)\b/);
  const hold = has(text, /\b(locked off|static|holds still|camera holds|does not move|doesn't move)\b/);
  const fpv = has(text, /\bfpv\b/);
  const drone = has(text, /\b(drone|aerial|overhead)\b/);
  const close = has(text, /\b(close-up|closeup|close up)\b/);
  const wide = has(text, /\bwide\b/);
  if (fpv) return { move: "fpv", label: "Chase from behind" };
  if (drone) return { move: "drone", label: "Drone" };
  if (close && wide) return { move: "close-wide", label: "Close to wide" };
  if (close) return { move: "close", label: "Close-up" };
  if (hold) return { move: "hold", label: "Hold" };
  if ((behind || push) && (orbit || toFront)) return { move: "around", label: "From behind, around to the front" };
  if ((track || parallel) && behind) return { move: "push-behind", label: "Follow from behind" };
  if (parallel) return { move: "parallel", label: "Parallel" };
  if (track) return { move: "track", label: push ? "Track and push in" : "Track" };
  if (pull) return { move: "pull", label: "Pull wide" };
  if (wide) return { move: "wide", label: "Wide" };
  if (orbit) return { move: "orbit", label: "Orbit" };
  if (behind) return { move: "push-behind", label: "Push in from behind" };
  if (push) return { move: "push", label: "Push in" };
  return { move: "push", label: "Push in" };
}

function keysFor(move: Move, target: Vec3, lookY: number, room: boolean, arc?: { from: ShotSize; to: ShotSize }): PrevizSpec["camera"]["keys"] {
  const look = vec(target[0], lookY, target[2]);
  const y = lookY + (room ? 0.15 : 0.35);
  const scale = room ? 0.42 : 1;
  const key = (t: number, angle: number, dist: number, camY = y, focalMm?: number) => {
    const row = { t, pos: polar(target, angle, dist * scale, camY), look };
    return focalMm ? { ...row, focal_mm: focalMm } : row;
  };
  if (move === "close") return [key(0, 0.35, 2.1, lookY, 50), key(1, 0.2, 1.4, lookY, 65)];
  if (move === "wide") return [key(0, 0.5, 8, y + 1.4, 24), key(1, 0.4, 12, y + 2.8, 18)];
  if (move === "close-wide") return [key(0, 0.3, 1.6, lookY, 50), key(1, 0.45, 11, y + 2.4, 18)];
  if (move === "drone") return [key(0, Math.PI, 14, lookY + 11, 24), key(0.55, Math.PI * 0.65, 9, lookY + 6, 28), key(1, 0.35, 5.5, lookY + 2.4, 32)];
  if (move === "fpv") return fpvKeys(target, lookY, room);
  if (move === "hold") return [key(0, 0.6, 4.2), key(1, 0.6, 4.2)];
  if (move === "around") return [key(0, Math.PI, 5.4), key(0.5, Math.PI * 0.5, 3.8, y + 0.25), key(1, 0.15, 3.1)];
  if (move === "push-behind") return [key(0, Math.PI, 6.2), key(1, Math.PI * 0.85, 3.2)];
  if (move === "orbit") return [key(0, 2.4, 4.6), key(0.5, 1.2, 4.2, y + 0.4), key(1, 0.2, 3.6)];
  if (move === "parallel") return [key(0, Math.PI / 2, 4.5, lookY + 0.15), key(1, Math.PI / 2, 4.5, lookY + 0.15)];
  if (move === "track") return [key(0, 1.15, 5.5), key(1, 0.75, 3.4)];
  if (move === "pull") return [key(0, 0.4, 2.8, y + 0.1), key(1, 0.5, 8.5, y + 2.2)];
  if (move === "reveal" && arc) {
    const a = sizePose(arc.from, lookY);
    const b = sizePose(arc.to, lookY);
    const angle = 0.15;
    const mix = (p: number, q: number, u: number) => p + (q - p) * u;
    return [
      key(0, angle, a.dist, a.camY, a.focal),
      key(0.45, angle + 0.06, mix(a.dist, b.dist, 0.32), mix(a.camY, b.camY, 0.28), Math.round(mix(a.focal, b.focal, 0.35))),
      key(1, angle + 0.16, b.dist, b.camY, b.focal),
    ];
  }
  return [key(0, 0.7, 5.2), key(1, 0.35, 3.1)];
}

/** A chase that stays behind the rider. Starts about 3 m back and closes to about 1.5 m, just above the shoulder. */
function fpvKeys(target: Vec3, lookY: number, _room: boolean): PrevizSpec["camera"]["keys"] {
  const steps = 16;
  const poses: Array<{ t: number; pos: Vec3; look: Vec3; roll: number }> = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const back = 3 - 1.5 * t;
    const drift = Math.sin(t * Math.PI * 2) * 0.14;
    const bob = Math.sin(t * Math.PI * 3) * 0.04;
    poses.push({
      t,
      pos: vec(target[0] + drift, Math.max(0.45, lookY + 0.18 + bob), target[2] - back),
      look: vec(target[0] + drift * 0.25, lookY, target[2]),
      roll: Math.round(Math.sin(t * Math.PI * 2) * 40) / 10,
    });
  }
  return poses.map((p) => ({ t: p.t, pos: p.pos, look: p.look, focal_mm: 16, roll: p.roll }));
}

const DRESS_PLACE: Record<string, string> = {
  bedroom: "interior",
  kitchen: "interior",
  office: "interior",
  restaurant: "interior",
  store: "interior",
  gym: "interior",
  car_interior: "interior",
  stage: "interior",
  highway: "outskirts",
  field: "outskirts",
  sports_field: "park",
  rooftop: "city",
  station: "city",
  snow_mountain: "forest",
};

const KNOWN_PLACE = new Set(["park", "suburbs", "house", "interior", "city", "outskirts", "desert", "forest", "ocean", "lake", "river"]);

const PLACE_LABEL: Record<string, string> = {
  park: "Park",
  suburbs: "Suburbs",
  house: "House",
  interior: "Interior",
  city: "City",
  outskirts: "Outskirts",
  desert: "Desert",
  forest: "Forest",
  ocean: "Ocean",
  lake: "Lake",
  river: "River",
  bedroom: "Bedroom",
  kitchen: "Kitchen",
  office: "Office",
  restaurant: "Restaurant",
  store: "Store",
  gym: "Gym",
  car_interior: "Car interior",
  highway: "Highway",
  field: "Field",
  sports_field: "Field",
  rooftop: "Rooftop",
  stage: "Stage",
  station: "Station",
  snow_mountain: "Mountain",
};

/** Poses the library can play. Anything else (a camera verb the parser kept) stands. */
const BODY_POSE = new Set([
  "carry", "catch", "climb", "cook", "crouch", "dance", "drive", "eat", "fight_guard", "fight_punch", "fly_kite", "forehand", "give", "hold",
  "hold_phone", "hold_umbrella", "hug", "jog", "jump", "kick", "kiss", "kneel", "lean_window", "lie_down", "play_guitar", "point", "pull", "push",
  "push_cart", "read", "ride_bike", "ride_horse", "ride_skate", "ride_ski", "run", "serve", "shake_hands", "sing", "sip", "sit", "sit_boat",
  "sit_passenger", "stand", "swing", "talk", "throw", "type", "walk", "wave",
]);

const SET_PROP: Record<string, string> = {
  bench: "bench",
  table: "table",
  cup: "cup",
  phone: "phone",
  fountain: "fountain",
  house: "house",
  tree: "trees",
  post: "post",
  rock: "rock",
  building: "building",
};

type CompileOpts = { duration_s?: number; focal_mm?: number; camera?: CameraPick | null; speed?: number; taper?: number; distance?: number; frame?: [number, number] | null };

function dressPlace(id: string): string {
  if (KNOWN_PLACE.has(id)) return id;
  return DRESS_PLACE[id] ?? "park";
}

function placeLabel(id: string): string {
  return PLACE_LABEL[id] ?? id.replace(/_/g, " ").replace(/\b[a-z]/g, (c) => c.toUpperCase());
}

function rideKind(asset: string): { id: string; kind: "car" | "bike" | "skate" | "boat" } | null {
  if (asset === "car" || asset === "suv" || asset === "truck" || asset === "van" || asset === "bus" || asset === "taxi") return { id: "car", kind: "car" };
  if (asset === "bicycle") return { id: "bike", kind: "bike" };
  if (asset === "motorcycle" || asset === "scooter") return { id: "moto", kind: "bike" };
  if (asset === "skateboard") return { id: "skate", kind: "skate" };
  if (asset === "boat") return { id: "boat", kind: "boat" };
  return null;
}

function propById(props: Prop[], id: string): Prop | undefined {
  return props.find((p) => p.id === id);
}

/** Names the current person() already turns into a cup or a phone. The spec pose is kept for the library. */
function playablePose(pose: string, holdsCup: boolean, holdsPhone: boolean): string {
  if (holdsPhone || pose === "hold_phone") return "phone";
  if ((pose === "sit" || pose === "sip") && holdsCup) return "sit_sip";
  if (pose === "sip") return "stand_sip";
  if (pose === "lie_down") return "lay";
  if (BODY_POSE.has(pose)) return pose;
  return "stand";
}

function whoLabel(anim: string): string {
  const base = anim.replace(/_follow$/, "");
  if (base === "walk") return "Person walking";
  if (base === "jog") return "Person jogging";
  if (base === "run") return "Person running";
  if (base === "sit" || base === "sit_sip" || base === "sit_passenger" || base === "sit_boat") return "Person sitting";
  if (base === "phone") return "Person texting";
  if (base === "lay") return "Person lying";
  if (base === "drive") return "Person driving";
  if (base === "jump") return "Person jumping";
  return "Person";
}

function shotFraming(cam: ShotSpec["camera"], walking: boolean): { move: Move; label: string; medium: boolean; extreme: boolean } {
  const extreme = cam.shot === "extreme_close";
  if (cam.shot === "close" || extreme) return { move: "close", label: extreme ? "Extreme close-up" : "Close-up", medium: false, extreme };
  if (cam.move === "pull_out") return { move: "pull", label: "Pull wide", medium: false, extreme: false };
  if (cam.shot === "wide") return { move: "wide", label: "Wide", medium: false, extreme: false };
  if (cam.angle === "overhead") return { move: "drone", label: "Drone", medium: false, extreme: false };
  if (cam.angle === "pov") return { move: "fpv", label: "Chase from behind", medium: false, extreme: false };
  if (cam.angle === "through_windshield") return { move: "fpv", label: "Through the windshield", medium: false, extreme: false };
  if (cam.move === "orbit" || cam.move === "pan") return { move: "orbit", label: "Orbit", medium: false, extreme: false };
  if (cam.move === "crane_up" || cam.move === "crane_reveal") return { move: "drone", label: "Drone", medium: false, extreme: false };
  if (cam.move === "dolly_along" || cam.angle === "side") return { move: "parallel", label: "Parallel", medium: false, extreme: false };
  if (cam.move === "static") return { move: "hold", label: "Hold", medium: false, extreme: false };
  if (cam.move === "track" || cam.move === "follow") return { move: walking ? "track" : "push", label: walking ? "Track" : "Push in", medium: false, extreme: false };
  if (cam.angle === "behind" || cam.angle === "over_shoulder") return { move: "push-behind", label: walking ? "Follow from behind" : "Push in from behind", medium: false, extreme: false };
  if (cam.shot === "medium") return { move: "push", label: "Medium", medium: true, extreme: false };
  return { move: "push", label: "Push in", medium: false, extreme: false };
}

/** One person, the first dog or cat, and the first vehicle. Pairs, counts and held props stay on the shot spec. */
function compileShot(raw: string, shot: ShotSpec, opts?: CompileOpts): Compiled {
  const duration_s = opts?.duration_s && opts.duration_s > 0 ? opts.duration_s : PREVIZ_DEFAULT_S;
  const focal = opts?.focal_mm && opts.focal_mm >= 8 ? Math.min(200, opts.focal_mm) : 35;
  const camSpeed = clampNum(opts?.speed, 0.25, 3, 1);
  const place = dressPlace(shot.place.id);
  const person = shot.actors.find((a) => a.kind === "person");
  const animal = shot.actors.find((a) => a.kind === "animal" && (a.asset === "dog" || a.asset === "cat"));
  const held = (asset: string) =>
    !!person && person.relations.some((r) => r.type === "holds" && propById(shot.props, r.target)?.asset === asset);
  const ridden = person?.relations.find((r) => r.type === "rides");
  const riddenProp = ridden ? propById(shot.props, ridden.target) : undefined;
  const riddenKind = riddenProp ? rideKind(riddenProp.asset) : null;
  const mount = riddenKind && riddenKind.kind !== "car" ? riddenKind : null;
  const vehicleProp = shot.props.find((p) => p.kind === "vehicle" || (p.kind === "rideable" && rideKind(p.asset)));
  const vehicle = vehicleProp && vehicleProp !== riddenProp ? rideKind(vehicleProp.asset) : null;
  const showVehicle = !!vehicle;

  const props = [place];
  const seen = new Set<string>([place]);
  for (const prop of shot.props) {
    const name = SET_PROP[prop.asset];
    if (!name || seen.has(name)) continue;
    seen.add(name);
    props.push(name);
  }
  if (held("cup") && !seen.has("cup")) props.push("cup");
  if ((held("phone") || person?.pose === "hold_phone") && !seen.has("phone")) props.push("phone");

  const subjects: PrevizSpec["subjects"] = [];
  const height = person?.preset === "child" ? 1.2 : 1.75;
  const anim = person ? playablePose(person.pose, held("cup"), held("phone") || person.pose === "hold_phone") : "stand";
  const onSeat = !!person && person.relations.some((r) => r.type === "on");
  if (person && !mount) {
    subjects.push({ id: "person", kind: "humanoid", height_m: height, pos: vec(0, onSeat ? 0.46 : 0, 0), anim });
  }
  if (animal) {
    const beside = !!person && !mount;
    subjects.push({
      id: animal.asset,
      kind: animal.asset === "cat" ? "cat" : "dog",
      height_m: animal.asset === "cat" ? 0.32 : 0.55,
      pos: vec(beside ? -0.72 : 0, 0, beside ? 0.2 : 0),
      anim: animal.pose || "stand",
    });
  }
  if (mount) {
    subjects.push({ id: mount.id, kind: mount.kind, height_m: mount.kind === "boat" ? 1.4 : height, pos: vec(0, 0, mount.kind === "boat" ? 1.2 : 0), anim: "track" });
  }
  if (showVehicle && vehicle) {
    const withPerson = subjects.some((s) => s.kind === "humanoid");
    subjects.push({
      id: vehicle.id,
      kind: vehicle.kind,
      height_m: vehicle.kind === "boat" ? 1.4 : vehicle.kind === "car" ? 1.5 : 1.1,
      pos: vec(withPerson ? 2.4 : 0, 0, withPerson ? 1.4 : vehicle.kind === "car" ? 2.2 : 0.4),
      anim: "park",
    });
  }

  const walking = anim === "walk" || anim === "jog" || anim === "run";
  const framing = shotFraming(shot.camera, walking);
  const chosen = opts?.camera ? PICK_MOVE[opts.camera] : framing;
  const move = chosen.move;
  const camera = chosen.label;
  if (move === "track" || move === "push-behind" || move === "parallel") {
    const lead = subjects.find((s) => s.anim === "walk" || s.anim === "jog" || s.anim === "run");
    if (lead) lead.anim = `${lead.anim}_follow`;
  }
  const camTaper = clampNum(opts?.taper, 0, 1, move === "fpv" ? 0.55 : move === "track" || move === "parallel" ? 0 : 1);
  const primary = subjects[0];
  let cameraKeys: PrevizSpec["camera"]["keys"];
  if (!primary) cameraKeys = landscapeKeys();
  else {
    const seated = anim.startsWith("sit");
    const lookY =
      primary.kind === "car" || primary.kind === "boat" ? primary.pos[1] + 0.9
      : primary.kind === "dog" ? 0.4
      : primary.kind === "cat" ? 0.28
      : primary.kind === "skate" ? 1.1
      : primary.pos[1] + primary.height_m * (seated || primary.kind === "bike" ? 0.5 : 0.62);
    const room = place === "interior";
    if (!opts?.camera && framing.medium) {
      const a = sizePose("medium", lookY);
      const look = vec(primary.pos[0], lookY, primary.pos[2]);
      cameraKeys = [
        { t: 0, pos: polar(primary.pos, 0.4, a.dist, a.camY), look, focal_mm: a.focal },
        { t: 1, pos: polar(primary.pos, 0.25, a.dist * 0.86, a.camY), look, focal_mm: a.focal },
      ];
    } else cameraKeys = keysFor(move, primary.pos, lookY, room);
    if (framing.extreme && !opts?.camera) cameraKeys = cameraKeys.map((k) => ({ ...k, focal_mm: 85 }));
  }
  cameraKeys = cameraKeys.map((k) => ({ ...k, focal_mm: k.focal_mm ?? (place === "interior" ? Math.min(focal, 28) : focal) }));

  const kit = shownPlace(raw, placeLabel(shot.place.id), shot.place.source) ?? "";
  const matched = kit ? [kit] : [];
  if (person && !mount) matched.push(whoLabel(anim));
  if (mount) matched.push("Person riding");
  if (animal) matched.push(animal.asset === "cat" ? "Cat" : "Dog");
  if (showVehicle && vehicle) matched.push(vehicle.kind === "car" ? "Car" : vehicle.kind === "boat" ? "Boat" : vehicle.kind === "skate" ? "Skateboard" : "Bicycle");
  if (props.includes("bench")) matched.push("Seat");
  if (props.includes("cup")) matched.push("Cup");
  if (props.includes("phone")) matched.push("Phone");
  const who = matched.find((m) => /Person|Dog|Cat|Car|Bicycle|Skateboard|Boat/.test(m));
  const title = who ? (kit ? `${kit}, ${who.toLowerCase()}` : who) : kit || "Shot";
  const spec: PrevizSpec = {
    version: 1,
    template: "described",
    duration_s,
    fps: 24,
    size: [W, H],
    lens: { focal_mm: cameraKeys[0].focal_mm ?? focal, film_gauge_mm: 36 },
    environment: { ground: place === "interior" ? "none" : "plane", sky: "gradient", fog: fogFor(place), props },
    subjects,
    camera: {
      interp: "catmull_rom",
      ease: move === "track" || move === "fpv" || move === "parallel" ? "linear" : "in_out",
      speed: camSpeed,
      taper: camTaper,
      distance: clampNum(opts?.distance, 0.35, 4, 1),
      ...(opts?.frame && (Math.abs(opts.frame[0]) > 0.001 || Math.abs(opts.frame[1]) > 0.001) ? { frame: opts.frame } : {}),
      ...(opts?.camera ? { pick: opts.camera } : {}),
      keys: cameraKeys,
    },
    seed: shot.seed % 1_000_000_000,
    speed: 1,
    notes: raw.trim().slice(0, 2000),
  };
  if (cameraKeys.some((k) => k.pos[1] > 6)) spec.environment.fog = [spec.environment.fog[0], Math.max(spec.environment.fog[1], 150)];
  // An orbit that names its sides ("starts from behind ... around to the front") used those words (Run 14).
  const sided = (shot.camera?.segments ?? []).some((s: { move?: string; params?: Record<string, unknown> }) => s.move === "orbit" && (s.params?.from_deg != null || s.params?.to_deg != null));
  const missed = shot.unmatched.filter((w: string) => !(sided && ORBIT_SIDE_WORDS.test(w)));
  return rememberPick({ spec, title, matched, missed: missed.slice(0, 8), camera, shot }, opts?.camera ?? null);
}

const ORBIT_SIDE_WORDS = /^(starts?|starting|begins?|beginning|from|front|behind|back|side|around|round|ends?|ending)$/i;

function landscapeKeys(): PrevizSpec["camera"]["keys"] {
  const look0 = vec(0, 1.4, 10);
  const look1 = vec(0, 1.2, 18);
  return [
    { t: 0, pos: vec(0, 2.2, -4), look: look0 },
    { t: 1, pos: vec(1.5, 2.6, 6), look: look1 },
  ];
}

export function compile(raw: string, opts?: CompileOpts): Compiled {
  raw = raw.trim().slice(0, 2000);
  if (shotSpecOn()) {
    const shot = readShot(raw);
    if (shot) return compileShot(raw, shot, opts);
  }
  const text = raw.toLowerCase().replace(/[’']/g, "'").replace(/[^a-z0-9'\s]/g, " ").replace(/\s+/g, " ").trim();
  const duration_s = opts?.duration_s && opts.duration_s > 0 ? opts.duration_s : PREVIZ_DEFAULT_S;
  const focal = opts?.focal_mm && opts.focal_mm >= 8 ? Math.min(200, opts.focal_mm) : 35;
  const camSpeed = clampNum(opts?.speed, 0.25, 3, 1);

  const heard = scan(text);
  let place = heard.place ?? "park";
  if (!heard.place) {
    if (heard.boat) place = "ocean";
    else if (heard.car && !heard.person) place = "suburbs";
    else if (heard.skate) place = "city";
    else if (heard.cat && !heard.person && !heard.dog) place = "interior";
  }
  const sitting = heard.sit || (heard.bench && heard.person);
  const walking = heard.walk;
  const movingRide = heard.ride || heard.bike || heard.skate || heard.moto;
  const mountBike = heard.bike && heard.person;
  const mountSkate = heard.skate && heard.person;
  const mountMoto = heard.moto && heard.person;
  const mountBoat = heard.boat && heard.person;
  const driverInside = heard.car && heard.person && heard.ride && !heard.bike && !heard.skate && !heard.moto;
  const showPerson = heard.person && !mountBike && !mountSkate && !mountMoto && !mountBoat && !driverInside;

  const props = [place];
  if (heard.bench) props.push("bench");
  if (heard.trees && place !== "forest") props.push("trees");
  const namedHouse = /\b(house|home|cabin|cottage|mansion|villa|barn)\b/.test(text);
  if ((heard.house || namedHouse) && place !== "house" && place !== "interior") props.push("house");
  if (heard.cup) props.push("cup");
  if (heard.phone) props.push("phone");
  if (heard.fountain) props.push("fountain");
  if (heard.table) props.push("table");
  if (heard.post) props.push("post");
  if (heard.rock) props.push("rock");
  if (heard.city) props.push(heard.city);

  const subjects: PrevizSpec["subjects"] = [];
  const gait = heard.gait ?? (walking ? "walk" : null);
  const personAnim = heard.action ?? (sitting && heard.cup ? "sit_sip" : sitting ? "sit" : heard.cup ? "stand_sip" : heard.phone ? "phone" : gait ?? "stand");
  const height = heard.child ? 1.2 : 1.75;
  const footX = heard.city === "sidewalk" ? -4.2 : 0;
  if (showPerson) {
    const y = sitting ? 0.46 : 0;
    const z = sitting ? 0.08 : heard.bench ? 0.85 : 0;
    subjects.push({ id: "person", kind: "humanoid", height_m: height, pos: vec(footX, y, z), anim: personAnim });
  }
  if (heard.dog) {
    const beside = showPerson || heard.bench;
    subjects.push({
      id: "dog",
      kind: "dog",
      height_m: 0.55,
      pos: vec(beside ? footX - 0.72 : footX, 0, beside ? 0.2 : 0),
      anim: !showPerson && sitting ? "sit" : walking ? "walk" : "stand",
    });
  }
  if (heard.cat) {
    const onBench = heard.bench && !showPerson;
    subjects.push({
      id: "cat",
      kind: "cat",
      height_m: 0.32,
      pos: vec(showPerson ? -0.85 : onBench ? 0.35 : 0.2, onBench ? 0.52 : 0, showPerson ? 0.45 : 0.15),
      anim: "stand",
    });
  }
  if (mountBike) subjects.push({ id: "bike", kind: "bike", height_m: height, pos: vec(0, 0, 0), anim: movingRide ? "track" : "park" });
  if (mountSkate) subjects.push({ id: "skate", kind: "skate", height_m: height, pos: vec(0, 0, 0), anim: "track" });
  if (mountMoto) subjects.push({ id: "moto", kind: "bike", height_m: height, pos: vec(0, 0, 0), anim: "track" });
  if (heard.boat) subjects.push({ id: "boat", kind: "boat", height_m: 1.4, pos: vec(0, 0, 1.2), anim: !mountBoat ? "park" : movingRide ? "track" : "rider" });
  if (heard.car) {
    subjects.push({
      id: "car",
      kind: "car",
      height_m: 1.5,
      pos: vec(showPerson ? 2.4 : 0, 0, showPerson ? 1.4 : 2.2),
      anim: heard.ride ? "track" : "park",
    });
  }
  if (heard.bike && !mountBike) subjects.push({ id: "bike", kind: "bike", height_m: 1.1, pos: vec(1.4, 0, 0.4), anim: "park" });
  if (heard.skate && !mountSkate) subjects.push({ id: "skate", kind: "skate", height_m: 1.1, pos: vec(0.8, 0, 0.6), anim: "park" });

  const parsed = cameraMove(text);
  const chosen = opts?.camera ? PICK_MOVE[opts.camera] : parsed;
  const { move, label: camera } = chosen;
  const wantsFollow = /\b(follow\w*|track\w*)\b/.test(text);
  if (move === "track" || move === "push-behind" || move === "parallel" || move === "reveal" || wantsFollow) {
    const walking = (s: (typeof subjects)[number]) => s.anim === "walk" || s.anim === "jog" || s.anim === "run";
    const lead = subjects.find((s) => s.kind === "humanoid" && walking(s)) ?? subjects.find(walking);
    if (lead) lead.anim = `${lead.anim}_follow`;
  }
  const camTaper = clampNum(opts?.taper, 0, 1, move === "fpv" ? 0.55 : move === "track" || move === "parallel" ? 0 : 1);
  const primary = subjects[0];
  let cameraKeys: PrevizSpec["camera"]["keys"];
  if (!primary) cameraKeys = landscapeKeys();
  else {
    const lookY =
      primary.kind === "car" || primary.kind === "boat" ? primary.pos[1] + 0.9
      : primary.kind === "dog" ? 0.4
      : primary.kind === "cat" ? 0.28
      : primary.kind === "skate" ? 1.1
      : primary.pos[1] + primary.height_m * (sitting || primary.kind === "bike" ? 0.5 : 0.62);
    const room = place === "interior";
    const arc = !opts?.camera && parsed.from && parsed.to ? { from: parsed.from, to: parsed.to } : undefined;
    cameraKeys = keysFor(move, primary.pos, lookY, room, arc);
    if (primary.kind === "car" && !heard.ride && (move === "push" || move === "track")) {
      cameraKeys = cameraKeys.map((k) => ({ ...k, pos: vec(k.pos[0] * 1.35, k.pos[1], k.pos[2] * 1.35) }));
    }
  }
  cameraKeys = cameraKeys.map((k) => ({ ...k, focal_mm: k.focal_mm ?? (place === "interior" ? Math.min(focal, 28) : focal) }));

  // The park is only the set it builds on: the chip says a place the words name or imply (Run 3).
  const placeLabel = (heard.place || place !== "park" ? PLACES.find((p) => p.id === place)?.label : inferPlace(raw)) ?? "";
  const matched = placeLabel ? [placeLabel] : [];
  if (heard.city === "sidewalk") matched.push("Sidewalk");
  else if (heard.city === "alley") matched.push("Alley");
  else if (heard.city === "plaza") matched.push("Plaza");
  if (heard.bench) matched.push("Seat");
  if (heard.trees && place !== "forest") matched.push("Tree");
  if ((heard.house || namedHouse) && place !== "house" && place !== "interior") matched.push("House");
  const actionLabel = heard.action === "jump" ? "Person jumping" : heard.action === "lay" ? "Person lying" : heard.action === "kneel" ? "Person kneeling" : heard.action === "crouch" ? "Person crouching" : heard.action === "wave" ? "Person waving" : null;
  const onFoot = gait === "run" ? "Person running" : gait === "jog" ? "Person jogging" : gait === "walk" ? "Person walking" : null;
  if (showPerson) matched.push(actionLabel ?? (sitting ? "Person sitting" : onFoot ?? (mountBike || mountSkate || mountMoto ? "Person riding" : "Person")));
  if (mountBike || mountSkate || mountMoto || mountBoat) matched.push("Person riding");
  if (heard.dog) matched.push("Dog");
  if (heard.cat) matched.push("Cat");
  if (heard.bike) matched.push("Bicycle");
  if (heard.skate) matched.push("Skateboard");
  if (heard.moto) matched.push("Motorcycle");
  if (heard.boat) matched.push("Boat");
  if (heard.car) matched.push(heard.ride ? "Car driving" : "Car");
  if (heard.cup) matched.push("Cup");
  if (heard.phone) matched.push("Phone");
  if (heard.fountain) matched.push("Fountain");
  if (heard.table) matched.push("Table");
  if (heard.post) matched.push("Post");
  if (heard.rock) matched.push("Rock");

  const used = heard.used;
  const missed: string[] = [];
  for (const word of text.split(" ")) {
    if (!word || word.length < 3 || STOP.has(word) || IGNORE.has(word) || used.has(word)) continue;
    if (!missed.includes(word)) missed.push(word);
  }

  const who = matched.find((m) => /Person|Dog|Cat|Car|Bicycle|Skateboard|Motorcycle|Boat/.test(m));
  const title = who ? (placeLabel ? `${placeLabel}, ${who.toLowerCase()}` : who) : placeLabel || "Shot";
  const fog = fogFor(place);
  const spec: PrevizSpec = {
    version: 1,
    template: "described",
    duration_s,
    fps: 24,
    size: [W, H],
    lens: { focal_mm: cameraKeys[0].focal_mm ?? focal, film_gauge_mm: 36 },
    environment: { ground: place === "interior" ? "none" : "plane", sky: "gradient", fog, props },
    subjects,
    camera: {
      interp: "catmull_rom",
      ease: move === "track" || move === "fpv" || move === "parallel" ? "linear" : "in_out",
      speed: camSpeed,
      taper: camTaper,
      distance: clampNum(opts?.distance, 0.35, 4, 1),
      ...(opts?.frame && (Math.abs(opts.frame[0]) > 0.001 || Math.abs(opts.frame[1]) > 0.001) ? { frame: opts.frame } : {}),
      ...(opts?.camera ? { pick: opts.camera } : {}),
      keys: cameraKeys,
    },
    seed: hash(text || "park") % 1_000_000_000,
    speed: 1,
    notes: raw.trim().slice(0, 2000),
  };
  if (cameraKeys.some((k) => k.pos[1] > 6)) spec.environment.fog = [spec.environment.fog[0], Math.max(spec.environment.fog[1], 150)];
  return rememberPick({ spec, title, matched, missed: missed.slice(0, 8), camera }, opts?.camera ?? null);
}

function fogFor(place: string): [number, number] {
  switch (place) {
    case "interior":
      return [6, 18];
    case "forest":
      return [8, 28];
    case "city":
      return [20, 90];
    case "desert":
      return [30, 160];
    case "ocean":
      return [20, 140];
    case "house":
      return [12, 50];
    default:
      return [14, 70];
  }
}
