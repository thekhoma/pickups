// "Motion and camera" for a made previz: the sentence, then what the previz itself does: the camera move, what comes into
// view and where, and where a group stands. Rule-based, from the spec and frame 0's view; no model.
// Fresh-9 user test: users write story ("she sprints"), not blocking, and LTX follows the words over the playblast. The
// fixes that worked named the previz's own blocking: "the lake comes into view on the left" (s03), "side by side in a
// line, nobody crosses" (s06).
import type { CastMember, Facing, PrevizFraming } from "../types";
import { CAMERA, castIn, withoutLookLine } from "./lookPrompt.ts";
import type { PrevizSpec } from "./spec.ts";
import { planCamera } from "./runtime/cameraPlan.js";

type V = [number, number, number];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: V) => Math.hypot(a[0], a[1], a[2]);
const ending = (s: string) => (/[.!?]$/.test(s) ? s : `${s}.`);
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);

/** The camera's move in words, from its first and last keys (so it holds for any template or a hand-placed path).
 * who: what it moves around ("her", "him", "it", "the person"); Run 2 (R&D #15): one person is never "them". */
export function cameraWords(spec: Pick<PrevizSpec, "camera">, who = "them"): string {
  return cameraWordsThem(spec).replace(/\bthem\b/g, who);
}

function cameraWordsThem(spec: Pick<PrevizSpec, "camera">): string {
  const keys = spec.camera.keys;
  const a = keys[0];
  const b = keys[keys.length - 1];
  // Push and pull on the ground: a crane up moves away too, but only upwards.
  const flat = (v: V): V => [v[0], 0, v[2]];
  const d0 = len(flat(sub(a.pos as V, a.look as V)));
  const d1 = len(flat(sub(b.pos as V, b.look as V)));
  const fwd = sub(a.look as V, a.pos as V);
  // The camera's right at the start: forward x up.
  const right: V = [-fwd[2], 0, fwd[0]];
  const rl = Math.hypot(right[0], right[2]) || 1;
  const move = sub(b.pos as V, a.pos as V);
  const side = (move[0] * right[0] + move[2] * right[2]) / rl;
  // How far round the subject it swings (degrees), seen from above.
  const ang = (p: V, l: V) => Math.atan2(p[2] - l[2], p[0] - l[0]);
  let turn = ((ang(b.pos as V, b.look as V) - ang(a.pos as V, a.look as V)) * 180) / Math.PI;
  turn = ((turn + 540) % 360) - 180;
  const rise = b.pos[1] - a.pos[1];
  const parts: string[] = [];
  if (Math.abs(turn) > 25) parts.push(`orbits to the ${side >= 0 ? "right" : "left"} around them`);
  else if (Math.abs(side) > 0.6) parts.push(`tracks to the ${side >= 0 ? "right" : "left"}`);
  if (d1 < d0 * 0.8) parts.push("pushes in");
  else if (d1 > d0 * 1.25) parts.push("pulls back");
  if (rise > 0.8) parts.push(rise > 4 ? "rises high above them" : "cranes up");
  else if (rise < -0.8) parts.push("cranes down");
  const hand = spec.camera.handheld ? " Handheld." : "";
  if (!parts.length) return `The camera holds still.${hand}`;
  const said = parts.length < 2 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
  return `The camera ${said}.${hand}`;
}

/** The runtime's moves (cameraPlan.js), said the way cameraWords says them. */
export const MOVE_WORDS: Record<string, string> = {
  push_in: "pushes in", pull_out: "pulls back", crane_up: "cranes up", crane_down: "cranes down", boom_up: "cranes up",
  boom_down: "cranes down", crane_reveal: "cranes up", rise: "rises high above them", orbit: "orbits around them",
  track: "tracks alongside", dolly_along: "tracks alongside", truck_left: "tracks to the left", truck_right: "tracks to the right",
  follow: "follows behind them", fpv: "follows close behind", aerial: "flies over", pan: "pans", pan_left: "pans to the left",
  pan_right: "pans to the right", tilt_up: "tilts up", tilt_down: "tilts down", swoop: "swoops down", static: "holds still",
};

/** Framing words the runtime turns into a move (camera.js: "close to wide" is a pull back from a close shot). */
const FRAMING_MOVE: [RegExp, string][] = [
  [/\bclose to wide\b/i, "pulls back from close to wide"],
  [/\bwide to close\b/i, "pushes in from wide to close"],
];

/** The move the sentence asks for, as the runtime reads it; null when it names none. */
export function sentenceCameraWords(notes: string, handheld = false): string | null {
  const framed = FRAMING_MOVE.find(([re]) => re.test(notes));
  const moves = framed ? [framed[1]] : (planCamera(notes).segments as { move: string }[]).map((s) => MOVE_WORDS[s.move]).filter(Boolean);
  if (!moves.length) return null;
  if (moves.length === 1 && moves[0] === "holds still") return `The camera holds still.${handheld ? " Handheld." : ""}`;
  const said = moves.length < 2 ? moves[0] : `${moves.slice(0, -1).join(", ")}, then ${moves[moves.length - 1]}`;
  return `The camera ${said}.${handheld ? " Handheld." : ""}`;
}

/** The previz path's own words: what frame 0 recorded (framing.move), else the spec's keys. */
export function pathCameraWords(spec: Pick<MotionSpec, "camera">, framing?: PrevizFraming, who = "them"): string {
  const m = framing?.move;
  return m ? cameraWords({ camera: { ...spec.camera, keys: [{ t: 0, ...m.start }, { t: 1, ...m.end }] } } as unknown as Pick<PrevizSpec, "camera">, who) : cameraWords(spec, who);
}

/** v2.4 9: the motion text's camera line: the user's camera words first, then the previz path. */
export function shotCameraWords(spec: Pick<MotionSpec, "camera" | "notes">, framing?: PrevizFraming, who = "them"): string {
  return (sentenceCameraWords(spec.notes ?? "", !!spec.camera.handheld) ?? pathCameraWords(spec, framing, who)).replace(/\bthem\b/g, who);
}

/** Run 2 (R&D #15): what the camera moves around, by who is in view: "her", "him", "it", "the person", else "them". */
export function whoInView(framing?: PrevizFraming, personNoun?: string): string {
  const seen = inView(framing);
  const people = seen.filter((s) => s.kind === "person");
  const cars = seen.filter((s) => s.kind === "vehicle");
  if (people.length === 1 && !cars.length) {
    const w = (personNoun ?? people[0].word ?? "").toLowerCase();
    return /^(woman|girl)$/.test(w) ? "her" : /^(man|boy)$/.test(w) ? "him" : "the person";
  }
  if (cars.length === 1 && !people.length) return "it";
  return "them";
}

/** Each move said as one kind, so the user's words and the path can be compared. */
const KINDS: [RegExp, string][] = [
  [/pushes in/, "in"], [/pulls back/, "out"], [/cranes up|rises/, "up"], [/cranes down|swoops down/, "down"],
  [/orbits/, "orbit"], [/tracks/, "track"], [/holds still/, "still"],
];
const kinds = (words: string) => new Set(KINDS.filter(([re]) => re.test(words)).map(([, k]) => k));

/**
 * v2.4 9: when the previz path disagrees with the user's camera words (R&D's s06: "cranes down", the path "tracks to the
 * left and pushes in"), the path's words, for the page to flag as a previz mismatch; null when they agree or the words
 * name no comparable move.
 */
export function cameraMismatch(spec: Pick<MotionSpec, "camera" | "notes">, framing?: PrevizFraming): string | null {
  const said = sentenceCameraWords(spec.notes ?? "");
  if (!said) return null;
  const want = kinds(said);
  if (!want.size) return null;
  const path = pathCameraWords(spec, framing);
  const got = kinds(path);
  return [...want].some((k) => got.has(k)) ? null : path.replace(/^The camera /, "").replace(/\.( Handheld\.)?$/, "");
}

/** Where a group stands, so nobody swaps places: two or more people side by side, or spread out. */
export function groupWords(spec: Pick<PrevizSpec, "subjects">): string {
  const people = spec.subjects.filter((s) => s.kind === "humanoid");
  if (people.length < 2) return "";
  const xs = people.map((p) => p.pos[0]);
  const zs = people.map((p) => p.pos[2]);
  const spread = (v: number[]) => Math.max(...v) - Math.min(...v);
  const line = Math.min(spread(xs), spread(zs)) < 0.5 * Math.max(spread(xs), spread(zs), 0.01);
  // Said as what happens, never as what doesn't (prompt_words.py: a positive has no negations).
  return line ? "They stay side by side in a line, in the same order." : "Everyone keeps their place.";
}

/** "The lake comes into view on the left at about 2 s." for each thing frame 0 doesn't show. */
export function revealWords(framing: PrevizFraming | undefined): string[] {
  // inView's words are plural ("trees"), mass ("water", "grass") or one thing ("a bench", "the road").
  const plural = (w: string) => /s$/.test(w) && !/^(a|the) /.test(w) && w !== "grass";
  return (framing?.reveals ?? []).map((r) => {
    const where = r.side === "middle" ? "ahead" : `on the ${r.side}`;
    return cap(`${r.word} ${plural(r.word) ? "come" : "comes"} into view ${where} at about ${r.at_s} s.`);
  });
}

/** The sentence without its camera talk: the camera line is worked out from the move itself, so it isn't said twice. */
function action(notes: string): string {
  const all = notes.replace(/\s+/g, " ").split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  // "Walks toward camera" is the action, not camera talk.
  const kept = all.filter((s) => !CAMERA.test(s.replace(/\b(toward(s)?|to|facing|faces?) (the )?camera\b/gi, "")));
  return (kept.length ? kept : all.slice(0, 1)).map(ending).join(" ");
}

/** What the motion words read from a spec. */
export type MotionSpec = Pick<PrevizSpec, "notes" | "camera" | "subjects"> & { duration_s?: number };

const vec = (v: unknown): v is V => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number");

/** A shot's stored previz_spec, read for the motion words only (the server writes unset fields as null, which the page's
 * own schema refuses). Null when it isn't a made previz. */
export function motionSpecOf(raw: unknown): MotionSpec | null {
  const r = raw as { notes?: unknown; camera?: { keys?: unknown; handheld?: unknown }; subjects?: unknown } | null;
  const keys = r?.camera?.keys;
  if (!Array.isArray(keys) || keys.length < 2 || !keys.every((k) => vec(k?.pos) && vec(k?.look))) return null;
  const subjects = Array.isArray(r?.subjects) ? r.subjects.filter((s) => vec(s?.pos) && typeof s?.kind === "string") : [];
  const duration_s = typeof (raw as { duration_s?: unknown })?.duration_s === "number" ? (raw as { duration_s: number }).duration_s : undefined;
  return { notes: typeof r?.notes === "string" ? r.notes : "", camera: { keys, handheld: r?.camera?.handheld ?? null }, subjects, duration_s } as unknown as MotionSpec;
}

/**
 * A car as the subject, with the camera travelling with it: the mount, so LTX moves the world past a moving car instead
 * of parking it (step 17). No speed: numbers add nothing for LTX. Null for anything else (a locked-off camera a car
 * drives past keeps its own camera words).
 */
export function vehicleCameraWords(spec: Pick<MotionSpec, "camera"> & { notes?: string }, framing?: PrevizFraming, _duration_s = 4): string | null {
  const main = (framing?.subjects ?? []).find((s) => s.in_view) ?? framing?.subjects?.[0];
  if (!main || main.kind !== "vehicle") return null;
  // Run 2 (R&D #6): only a car that moves (the previz drives it, or the sentence says it drives); a camera circling a
  // parked car is never a camera car.
  const parked = /\b(park(ed|s)?|stationary|standing still|sits)\b/i.test(spec.notes ?? "");
  const says = !parked && /\b(driv|speed|rac|cruis|zoom|chase)\w*/i.test(spec.notes ?? "");
  if (!main.speed_ms && !says) return null;
  // The director (e) and R&D #7: the travel direction matches the car's facing, the world streams the other way.
  if (main.facing === "left" || main.facing === "right") {
    const other = main.facing === "left" ? "right" : "left";
    return `Filmed from a camera car driving alongside at the same speed, the car drives to frame ${main.facing}, the road and background streaming past to frame ${other}.`;
  }
  if (main.facing === "away") return "The camera follows low and close behind the car, the road streaming toward it.";
  if (main.facing === "camera") return "The camera travels ahead of the car, the road streaming away behind it.";
  return "Filmed from a camera car driving alongside at the same speed, the road and background streaming past.";
}

/** The director (e) 1 to 4: a parked car's camera line by the previz move, slow, naming the car and the place. */
export function parkedCarWords(cameraLine: string, place: string): string {
  const k = kinds(cameraLine);
  const side = /to the (left|right)/.exec(cameraLine)?.[1] ?? "right";
  if (k.has("orbit")) return `The camera circles slowly around the car${place ? `, the ${place} turning behind it` : ""}.`;
  if (k.has("in")) return "The camera pushes slowly in on the car.";
  if (k.has("out")) return "The camera pulls slowly back from the car.";
  if (k.has("down")) return "The camera cranes slowly down toward the car.";
  if (k.has("up")) return "The camera cranes slowly up above the car.";
  if (k.has("track")) return `The camera dollies slowly to the ${side}.`;
  return "The camera holds steady on the car.";
}

const COUNT = ["", "", "two", "three", "four", "five", "six"];
/** A lone person's word, from the previz or the cast: the Keep line names them as this, never "person" when known. */
const PERSON_WORDS = "woman|man|girl|boy";
const VEHICLE_KINDS = "car|sedan|saloon|coupe|hatchback|suv|truck|van|bus|taxi|motorbike|motorcycle|jeep|pickup";
const PERSON_NOUN = new RegExp(`\\b(${PERSON_WORDS})\\b`, "i");
const VEHICLE_NOUN = new RegExp(`\\b(${VEHICLE_KINDS})\\b`, "i");
const VEHICLE_WORDS = new Set(["car", "truck", "van", "bus", "taxi", "motorbike", "motorcycle"]);

type Seen = NonNullable<PrevizFraming["subjects"]>;
function inView(framing?: PrevizFraming): Seen {
  const all = framing?.subjects ?? [];
  return all.some((s) => s.in_view) ? all.filter((s) => s.in_view) : all;
}

/** What the driver must keep, as prompt_words.keep_subjects says it: "woman", "two people", "car", "man and car". */
export function keepWords(spec: Pick<MotionSpec, "subjects">, framing?: PrevizFraming, personNoun?: string): string {
  const seen = inView(framing);
  const people = seen.length ? seen.filter((s) => s.kind === "person").length : spec.subjects.filter((s) => s.kind === "humanoid").length;
  const cars = seen.length ? seen.filter((s) => s.kind === "vehicle").length : spec.subjects.filter((s) => s.kind === "car").length;
  const word = (kind: string) => (seen.find((s) => s.kind === kind)?.word ?? "").toLowerCase();
  const one = personNoun ?? (PERSON_NOUN.test(word("person")) ? word("person") : "person");
  const car = VEHICLE_WORDS.has(word("vehicle")) ? word("vehicle") : "car";
  const parts = [
    people ? (people === 1 ? one : `${COUNT[people] ?? "many"} people`) : "",
    cars ? (cars === 1 ? car : `${COUNT[cars] ?? "many"} ${car}s`) : "",
  ].filter(Boolean);
  return parts.join(" and ") || "shot";
}

// --- the short Layout-to-Render line (the director's FACING_AND_VIDEO_TUNING.md 2) ----------------------------------------------

/** Place words, the one-word noun the Keep line uses, and a 4 to 8 word phrase for when the user's own is shorter.
 * First match wins (the lake before the park, the rooftop before the city). */
const PLACES: [RegExp, string, string][] = [
  [/^(highway|freeway|motorway)s?$/i, "road", "along an open highway"],
  [/^(rooftop|roof)s?$/i, "rooftop", "on a city rooftop"],
  [/^(booth|diner)s?$/i, "booth", "in a diner booth"],
  [/^(farmhouse|farm)s?$/i, "farmhouse", "outside a country farmhouse"],
  [/^lakes?$/i, "lake", "on the shore of a lake"],
  [/^(beach|shore|sea|ocean)$/i, "beach", "on a beach by the sea"],
  [/^(river|canal)s?$/i, "river", "on the bank of a river"],
  [/^(park|path|garden)s?$/i, "park", "along a path in the park"],
  [/^(forest|woods)$/i, "forest", "among the trees of a forest"],
  [/^(field|meadow)s?$/i, "field", "out in an open field"],
  [/^(sidewalk|pavement)s?$/i, "street", "down a city sidewalk"],
  [/^(street|city|town|alley)s?$/i, "street", "on a city street"],
  [/^(road|track|lane)s?$/i, "road", "along an open country road"],
];
const placeOf = (word: string) => PLACES.find(([re]) => re.test(word));
/** The director (Run 2): the real things the previz only stands in for, said in the short line. */
const EMPTY_ROAD = "on an empty freeway stretching to the horizon";
const LAKE_PINES = "tall dark pines along the far shore";
const PREP = /^(on|in|at|by|along|across|through|beside|near|down|up|inside|outside|around|over|under|into|onto|past)$/i;
const STOP = /^(and|as|while|but|then|who|where|when|with)$/i;

/** The user's own place phrase in one sentence: the preposition nearest the first place word through the last place
 * word of that run ("along a gravel path in the park"), 8 words at most. */
function placeIn(sentence: string): { phrase: string; noun: string; full: string } | null {
  const words = sentence.replace(/[.!?]$/, "").split(/\s+/).map((w) => w.replace(/[,;:]$/, ""));
  const at = words.findIndex((w) => placeOf(w));
  if (at < 0) return null;
  let start = at;
  while (start > 0 && at - start < 4 && !PREP.test(words[start])) start--;
  if (!PREP.test(words[start])) return null;
  let end = at;
  for (let j = at + 1; j < Math.min(words.length, start + 8) && !STOP.test(words[j]); j++) if (placeOf(words[j])) end = j;
  const hit = placeOf(words[end])!;
  return { phrase: words.slice(start, end + 1).join(" "), noun: hit[1], full: hit[2] };
}

/** A sentence that names the previz itself (prompt_words.py BANNED, or a grey stand-in): the server drops it, so it is
 * never the action. */
const PREVIZ_TALK =
  /(?<![\w-])(previz|pre-viz|mannequins?|blockout|block-out|backdrop|studio|clay|3D|small vehicles)(?![\w-])|\bgr[ae]y\b.*\b(figures?|shapes?|surfaces?|blocks?|layout|proxy|proxies|stand-ins?)\b/i;

/** Light the user stated, as a phrase ("at night", "in warm light"). A light that changes is never said: L2R can't move
 * the light (Booth: three runs, +1.5% luma), so that is post or an image-edit relight. */
const LIGHT_CHANGE =
  /\b(lights?|headlights?|sun|sunlight|shadows?|glow|neon|lamps?|beams?)\b[^.!?,]*\b(sweep|sweeps|sweeping|swept|change|changes|fade|fades|dim|dims|brighten|brightens|darken|darkens|flicker|flickers|shift|shifts|goes out|go out|comes on|come on|sets|rises|moves?)\b|\b(sweep|sweeps|sweeping|flash|flashes)\b[^.!?,]*\b(lights?|headlights?|beams?)\b|\b(fades?|turns?|goes) to (night|day|dark|black)\b/i;
const LIGHT_SAID: RegExp[] = [
  /\bat (night|dusk|dawn|sunset|sunrise|twilight|midday|noon|golden hour|blue hour)\b/i,
  /\bin (the )?(early |late[- ])?(morning|afternoon|evening) (sun|sunlight|light)\b/i,
  /\bin (warm|soft|harsh|bright|low|cool|golden|dappled|hazy) (sun|sunlight|light)\b/i,
  /\bunder (the )?(street ?lights?|neon( lights)?|moonlight|fluorescent lights?)\b/i,
  /\bby (candlelight|moonlight|firelight)\b/i,
];
const BARE_LIGHT =
  /\b((warm|soft|harsh|bright|low|cool|golden|golden[- ]hour|dappled|hazy) ){1,2}((early |late[- ])?(morning|afternoon|evening) )?(sun|sunlight|light)\b|\b(early |late[- ])?(morning|afternoon|evening) (sun|sunlight|light)\b/i;

function lightIn(texts: string[]): string | null {
  for (const text of texts) {
    for (const s of text.split(/(?<=[.!?])\s+/)) {
      if (LIGHT_CHANGE.test(s)) continue;
      const m = LIGHT_SAID.map((re) => s.match(re)).find(Boolean);
      if (m) return m[0].toLowerCase();
      const b = s.match(BARE_LIGHT);
      if (b) return `in ${b[0].toLowerCase()}`;
    }
  }
  return null;
}

/** A sentence's clauses that promise a light change, taken out ("as headlights sweep across her"). */
function steadyLight(s: string): string {
  if (!LIGHT_CHANGE.test(s)) return s;
  const parts = s.replace(/[.!?]$/, "").split(/(,\s*|\s+(?:as|while|when|and)\s+)/);
  const kept: string[] = [];
  for (let i = 0; i < parts.length; i += 2) if (!LIGHT_CHANGE.test(parts[i])) kept.push((kept.length ? parts[i - 1] ?? " " : "") + parts[i]);
  return kept.join("").trim();
}

const COLOUR =
  /\b(black|white|red|blue|green|yellow|orange|pink|purple|brown|tan|beige|cream|ivory|navy|olive|khaki|mustard|teal|burgundy|maroon|gold|golden|silver|gr[ae]y|charcoal|auburn|blond|blonde|ginger|dark|light|pale|bright)\b/i;
const HAIR = /\b(hair|beard|braid|bob|curls|moustache|mustache|stubble|head|ponytail|bun|afro|puff)\b/i;
const article = (seg: string) =>
  /^(a|an|the|his|her|their)\s/i.test(seg) || /s$/i.test(seg) || HAIR.test(seg) ? seg : `${/^[aeiou]/i.test(seg) ? "an" : "a"} ${seg}`;
/** A real grey said the way prompt_words.py says it: grey stays grey (Run 2, R&D #10: charcoal rendered near black). */
const realGrey = (seg: string) => seg;

const bare = (seg: string) => seg.replace(/^(a|an|the)\s+/i, "");
/** A cast line cut to who they are (age or body type) and one or two coloured garments under one "in" (Run 3, the director):
 * "a man in his 60s in a navy wool peacoat and grey flat cap". Hair only when the line names no garment. */
export function personSubject(line: string): string {
  const clean = line.trim().replace(/[.!?]$/, "");
  // The head is who they are: up to "with", a comma or the "in" that starts the clothes ("in his 60s" is age).
  const cut = clean.search(/\s+with\s+|,|\s+in\s+(?!(his|her|their)\s+\d)(?=(a|an|the)\s|\w+\s)/);
  const head = cut < 0 ? clean : clean.slice(0, cut);
  const rest = cut < 0 ? "" : clean.slice(cut).replace(/^\s*(with\s+|in\s+|,\s*)/, "");
  const segs = rest.split(/\s*,\s*|\s+and\s+|\s+over\s+|\s+in\s+/).map((s) => s.trim()).filter((s) => s && COLOUR.test(s));
  const worn = segs.filter((s) => !HAIR.test(s)).slice(0, 2);
  const hair = worn.length ? [] : segs.filter((s) => HAIR.test(s)).slice(0, 1);
  const said =
    head + (hair.length ? ` with ${article(hair[0])}` : "") + (worn.length ? ` in ${article(worn[0])}${worn[1] ? ` and ${bare(worn[1])}` : ""}` : "");
  return realGrey(said.replace(/\bin in\b/g, "in"));
}

/** A vehicle's cast line cut to the car itself and its first "with" part, at most 6 words of it (Run 5, c09): "a powder
 * blue 1960s four-door sedan with a bright white painted roof". Only the line's own words, never more. */
export function vehicleSubject(line: string): string {
  const head = line.trim().replace(/^[^:]*:\s*/, "").replace(/^the same\s+/i, "").split(/,|\.|\s+shown\b/i)[0].trim();
  const [car, tail] = head.split(/\s+with\s+/i);
  const part = (tail ?? "").split(/\s+/).filter(Boolean).slice(0, 5);
  while (part.length && /^(a|an|the|and|or|of|from|in|on|to|at|by|clearly|very)$/i.test(part[part.length - 1])) part.pop();
  return realGrey(article(car.trim()) + (part.length ? ` with ${part.join(" ")}` : ""));
}

/** Run 4 (c09 drift): with no cast picked, the car's own words from the user's text, never invented ones: a described
 * car ("a 1960s four-door sedan with steel wheels"), else the car parts a word list names ("Powder blue paint, white
 * roof, chrome." gives "a car with powder blue paint, white roof and chrome"); null when the text describes no car. */
const NOT_A_MODIFIER = "a|an|the|of|and|or|its|his|her|their|is|are|was|behind|front|back|beside|near|by|on|in|at|to|from|with|toward|towards|camera";
const DESCRIBED_CAR = new RegExp(
  `\\b(?:a|an|the)\\s+((?:(?!(?:${NOT_A_MODIFIER})\\b)[\\w-]+\\s+){1,5})(${VEHICLE_KINDS})\\b(?!'|\\s+(?:chase|race|crash|ride|journey|wash)\\b)((?:\\s+with\\s+[^.,;!?]+)?)`, "gi");
const WEAK_MODIFIER = /^(same|hero|main|other|parked|lone|single|only|one|first|second)$/i;
const CAR_PART = /\b(paint|paintwork|roof|wheels?|hubcaps?|bumpers?|grille|chrome|bodywork|tyres?|tires?|fenders?|bonnet|trim)\b/i;
const TAIL_STOP = /^(is|sits|stands|parked|waits|drives|rests|idles|that|which|as|while|in|on|at|by|beside|near|along|outside)$/i;
export function carWordsIn(texts: string[], noun = "car"): string | null {
  const sentences = texts.flatMap((t) => (t ?? "").split(/(?<=[.!?])\s+/)).filter((s) => s.trim() && !PREVIZ_TALK.test(s));
  for (const s of sentences) {
    for (const m of s.matchAll(DESCRIBED_CAR)) {
      const mods = m[1].trim().split(/\s+/);
      if (mods.every((w) => WEAK_MODIFIER.test(w))) continue;
      const tail = m[3].trim().split(/\s+/);
      const upTo = tail.findIndex((w, i) => i > 0 && TAIL_STOP.test(w));
      const withWords = m[3] ? ` ${tail.slice(0, upTo < 0 ? 6 : Math.min(upTo, 6)).join(" ")}` : "";
      return `${article(`${mods.filter((w) => !WEAK_MODIFIER.test(w)).join(" ")} ${m[2]}`)}${withWords}`.toLowerCase();
    }
  }
  for (const s of sentences) {
    const segs = s.replace(/[.!?]$/, "").split(/\s*,\s*|\s+and\s+/).map((x) => x.trim()).filter(Boolean);
    // a word list ("Powder blue paint, white roof, chrome, weathered clapboard."), never a sentence with a verb in it
    if (segs.length < 2 || segs.some((x) => x.split(/\s+/).length > 4)) continue;
    const parts = segs.filter((x) => CAR_PART.test(x)).map((x) => x.toLowerCase());
    if (!parts.length) continue;
    const said = parts.length < 2 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
    return `a ${noun} with ${said}`;
  }
  return null;
}

/** What the short line knows beyond the spec: the project's cast (only to find the names the words use), the cast picked
 * for this shot, and the Look box's words. */
export type MotionContext = { cast?: CastMember[]; picked?: CastMember[]; look?: string };

const isCar = (c: CastMember) => VEHICLE_NOUN.test(`${c.name} ${c.line}`) && !PERSON_NOUN.test(c.line.split(/\s+with\s+|,/)[0]);
/** Run 5 (c07): the project's one car cast, to add to the picked cast: a shot with one car in view then names it, picked
 * or not. Empty when the project has no car or more than one. */
export function loneCar(cast: CastMember[] = []): CastMember[] {
  const cars = cast.filter(isCar);
  return cars.length === 1 ? cars : [];
}

/**
 * The Layout-to-Render card's prompt (the director's FACING_AND_VIDEO_TUNING.md 2): the subject (a cast line cut to age and
 * one or two colours), the action, a 4 to 8 word place and the user's light in one sentence; the camera in another; then
 * "Keep the same <woman / man / two people / car>, <place> and camera as the driver." The server cuts any longer text
 * the same way (prompt_words.l2r_prompt), so what the form shows is what LTX reads.
 */
export function motionPromptFor(spec: MotionSpec, framing?: PrevizFraming, ctx: MotionContext = {}): string {
  // The fixed look line is the app's, not the user's: its light and "the room" never steer the video line.
  const look = withoutLookLine(ctx.look ?? "");
  const said = action(spec.notes).split(/(?<=[.!?])\s+/).filter((s) => !PREVIZ_TALK.test(s)).map(steadyLight).filter(Boolean);
  // A sentence that is only the light ("Warm light.") is the light phrase, never the action.
  const lightOnly = (s: string) => s.split(/\s+/).length <= 4 && (BARE_LIGHT.test(s) || LIGHT_SAID.some((re) => re.test(s)));
  let first = said.find((s) => !lightOnly(s)) ?? "";

  // The subject: one person or one car in view takes its cast line's words.
  const seen = inView(framing);
  const people = seen.length ? seen.filter((s) => s.kind === "person").length : spec.subjects.filter((s) => s.kind === "humanoid").length;
  const cars = seen.length ? seen.filter((s) => s.kind === "vehicle").length : spec.subjects.filter((s) => s.kind === "car").length;
  // R2-9: only the cast this shot uses (named in its words, or picked for it); never someone else from the project. With
  // none, the sentence keeps its own word ("A man").
  const named = castIn(`${spec.notes} ${look}`, ctx.cast);
  const pool = [...named, ...(ctx.picked ?? []).filter((c) => !named.includes(c))];
  const who = people === 1 ? pool.find((c) => c.kind === "person" && !isCar(c)) : !people && cars === 1 ? pool.find(isCar) : undefined;
  const castSubject = who?.line.trim() ? (people ? personSubject(who.line) : vehicleSubject(who.line)) : "";
  // Run 4 (c09): a car with no cast takes its own words from the idea or the Look box, so it stays the same car
  const ownCar = !castSubject && !people && cars === 1 ? carWordsIn([spec.notes, look], first.match(VEHICLE_NOUN)?.[0]?.toLowerCase()) : null;
  const subject = ownCar && first.toLowerCase().includes(bare(ownCar)) ? "" : castSubject || ownCar || "";
  if (subject) {
    const esc = (who?.name ?? "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const byName = esc ? new RegExp(`@?\\b${esc}\\b`, "i") : /$^/;
    const byNoun = new RegExp(`\\b(a|an|the|one)\\s+(?:[\\w-]+\\s+){0,3}?(${people ? PERSON_WORDS : VEHICLE_KINDS})\\b`, "i");
    // "A car chase ...": the car word is a modifier, so the car is in it.
    const modifier = (m: string) => !people && /^\s+(chase|race|crash|ride|journey|wash)\b/i.test(first.slice(first.indexOf(m) + m.length));
    if (byName.test(first)) first = first.replace(byName, subject);
    else if (byNoun.test(first)) first = first.replace(byNoun, (m) => (modifier(m) ? `${subject} in ${m.toLowerCase()}` : subject));
    else if (!first) first = subject;
  }
  const personNoun = people === 1 ? (subject.match(PERSON_NOUN)?.[0] ?? first.match(PERSON_NOUN)?.[0])?.toLowerCase() : undefined;

  // The place: the user's own phrase when it is 4 to 8 words, else the place word's own phrase, said once.
  const usable = (p: { phrase: string; full: string }) => (p.phrase.split(/\s+/).length < 4 || /\bgr[ae]y\b/i.test(p.phrase) ? p.full : p.phrase);
  const own = first ? placeIn(first) : null;
  let place = own?.noun ?? "";
  if (own && first.includes(own.phrase)) first = first.replace(own.phrase, usable(own));
  if (!own) {
    const other = [...said.slice(1), ...look.split(/(?<=[.!?])\s+/).filter((s) => !CAMERA.test(s))].map(placeIn).find(Boolean);
    const known = placeOf(framing?.place ?? "");
    const phrase = other ? usable(other) : known?.[2];
    place = other?.noun ?? known?.[1] ?? "";
    if (phrase && first) first = `${first.replace(/[.!?]$/, "")} ${phrase}`;
  }

  // Run 2 (the director's verdict 4): an empty road said in positive words (the shot's "through traffic" brought trucks).
  const words = `${spec.notes} ${look}`;
  if (place === "road" && /\bempty\b/i.test(words)) {
    first = first.replace(/\s+(through|in|amid) (heavy |the )?traffic\b/i, "");
    if (!/\bempty\b/i.test(first)) first = `${first.replace(/[.!?]$/, "").replace(/\s+along an open (highway|country road)$/, "")} ${EMPTY_ROAD}`;
  }
  // Run 2 (the director's verdict 3): a lake with trees names the real trees, so the previz's stand-ins never read as topiary.
  const trees = Object.keys(framing?.view?.share ?? {}).includes("trees") || (framing?.reveals ?? []).some((r) => r.word === "trees") || /\b(pines?|forest)\b/i.test(words);
  if (place === "lake" && trees && first && !/\bpines?\b/i.test(first)) first = `${first.replace(/[.!?]$/, "")}, ${LAKE_PINES}`;

  // The light, only when the user said it.
  const light = lightIn([spec.notes, look]);
  if (light && first && !first.toLowerCase().includes(light.replace(/^(in|at|by|under) /, ""))) first = `${first.replace(/[.!?]$/, "")} ${light}`;

  // The camera: a moving car's mount, a parked car's slow move, else the move around whoever is in view.
  const around = whoInView(framing, personNoun);
  const moving = vehicleCameraWords(spec, framing, spec.duration_s ?? 4);
  const lone = inView(framing);
  const parked = !moving && lone.length === 1 && lone[0].kind === "vehicle";
  const path = shotCameraWords(spec, framing, around);
  const camera = (moving ?? (parked ? parkedCarWords(path, place) : path)).replace(/\.\s+Handheld\.$/, ", handheld.");

  const keep = [keepWords(spec, framing, personNoun), place].filter(Boolean).join(", ");
  return [first && ending(cap(first)), camera, `Keep the same ${keep} and camera as the driver.`].filter(Boolean).join(" ");
}

/** What the words say about facing, as start_frame.py FACING_WORDS reads them (first match wins). */
const FACING_WORDS: [Facing, RegExp][] = [
  ["away", /\b(from behind|backs? to (the )?camera|facing away|walks? away|drives? away|seen from the back)\b/i],
  ["camera", /\b(toward(s)? (the )?camera|facing (the )?camera|faces? (the )?camera|to camera|head-on|dead on)\b/i],
  ["left", /\b(facing|toward(s)?|nose to) (frame )?left\b/i],
  ["right", /\b(facing|toward(s)?|nose to) (frame )?right\b/i],
];
const SUBJECT_LEADS = ["he", "she", "they", "both", "the (?:car|vehicle|coupe|sedan|saloon|suv|van|truck|woman|man|girl|boy|couple|dancers)"];
const LIGHT_WORDS = /\b(sun|sunlight|sunset|sunrise|light|lights|lit|lamps?|shadows?|golden[- ]hour|blue hour|dusk|dawn|backlit|overcast|moonlight)\b/i;
const FACING_SAID: Record<Facing, string> ={ camera: "facing the camera", away: "facing away", left: "facing frame left", right: "facing frame right" };

/** When the words name a facing that no subject in view has in the previz, the flag's sentence; the previz wins
 * (start_frame.facing_conflicts takes those words out of the start frame), so the page only says so. */
export function facingMismatch(words: string, framing?: PrevizFraming, names: string[] = []): string | null {
  const all = (framing?.subjects ?? []).filter((s) => s.facing);
  const seen = all.some((s) => s.in_view) ? all.filter((s) => s.in_view) : all;
  if (!seen.length) return null;
  // Run 2 (start_frame.facing_conflicts): only a clause about a subject is a facing claim (its name, he, she, they or
  // the car, its own or the sentence's lead); a light or shadow sentence never is.
  const esc = names.filter(Boolean).map((n) => n.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const lead = new RegExp(`^\\s*(?:(?:and|but|then|while|as)\\s+)?@?(?:${[...esc, ...SUBJECT_LEADS].join("|")})\\b`, "i");
  for (const sentence of words.split(/(?<=[.!?])\s+/)) {
    if (LIGHT_WORDS.test(sentence)) continue;
    const clauses = sentence.replace(/[.!?]$/, "").split(/\s*[,;]\s*|\s+(?:and|but|while|as)\s+/);
    const led = lead.test(clauses[0] ?? "");
    const said = clauses.map((c) => (led || lead.test(c) ? FACING_WORDS.find(([, re]) => re.test(c))?.[0] : undefined)).find(Boolean);
    if (!said || seen.some((s) => s.facing === said)) continue;
    const one = seen[0];
    return seen.length === 1
      ? `The previz has the ${one.kind === "vehicle" ? "car" : "person"} ${FACING_SAID[one.facing!]}; the previz wins.`
      : `No one in the previz is ${FACING_SAID[said]}; the previz wins.`;
  }
  return null;
}
