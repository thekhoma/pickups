// The "Try:" row under the sentence: a place, an action and a camera move, built from grammar.json so it never drifts
// from the vocabulary. Every fragment is parsed before it's offered. The set rotates by seed, never at random.
import { readShot } from "./shotLex.ts";
import type { ShotSpec } from "./runtime/types";

type Parse = (text: string) => ShotSpec | null;

export interface TryFragment {
  kind: "place" | "action" | "camera";
  text: string;
}

type Grammar = { places?: Record<string, string>; verbs?: Record<string, { pose?: string }>; camera?: Record<string, { move?: string }> };

// Place phrases that read well, each kept only if its word is a grammar place and the phrase parses to that place.
const PLACES: [string, string][] = [
  ["in a park", "park"], ["on a city street", "street"], ["in a diner", "diner"], ["in a kitchen", "kitchen"],
  ["on a rooftop", "rooftop"], ["on a beach", "beach"], ["in a forest", "forest"], ["in the desert", "desert"],
  ["by a lake", "lake"], ["in an office", "office"], ["at a train station", "station"], ["in a living room", "living room"],
  ["on a highway", "highway"], ["in a parking lot", "parking lot"], ["by the river", "river"], ["in a field", "field"],
  ["on a stage", "stage"], ["in a store", "store"], ["in the suburbs", "suburbs"], ["on a snowy mountain", "mountain"],
];
// Moves a first-time user reads at a glance. The rest still parse; they just aren't suggested.
const MOVES = ["push in", "pull out", "orbit", "tracking shot", "crane up", "low dolly", "dolly in", "pan left", "aerial shot", "crane down", "locked off", "pull back"];
const PEOPLE = ["a woman", "a man", "two friends", "a girl", "an old man", "a couple"];
const ACTIONS = ["walk", "run", "jog", "sit", "dance", "wave", "talk", "skate", "read", "lean", "drink", "cross", "wait", "jump"];

function gerund(v: string): string {
  if (v === "lie") return "lying";
  if (/ie$/.test(v)) return v.slice(0, -2) + "ying";
  if (/[^aeiou]e$/.test(v) && v !== "see") return v.slice(0, -1) + "ing";
  if (/^(run|sit|jog|swim|hop|stop|skip|shop|chat|pet|nap|hug)$/.test(v)) return v + v.slice(-1) + "ing";
  return v + "ing";
}

let cache: { grammar: object; pools: Record<TryFragment["kind"], string[]> } | null = null;

/** Every fragment that parses as what it claims to be, per kind, in a fixed order. */
export function tryPools(grammar: object | null, parse: Parse = readShot): Record<TryFragment["kind"], string[]> | null {
  if (!grammar) return null;
  if (cache?.grammar === grammar) return cache.pools;
  const g = grammar as Grammar;
  const place: string[] = [];
  for (const [text, word] of PLACES) {
    const id = g.places?.[word];
    if (!id) continue;
    const shot = parse(`a person standing ${text}`);
    if (shot?.place.source === "word" && shot.place.id === id) place.push(text);
  }
  const action: string[] = [];
  const verbs = g.verbs ?? {};
  ACTIONS.filter((v) => v in verbs).forEach((v, i) => {
    const text = `${PEOPLE[i % PEOPLE.length]} ${gerund(v)}`;
    const shot = parse(text);
    if (shot && shot.actors.length > 0 && shot.actors[0].pose && shot.actors[0].pose !== "stand") action.push(text);
  });
  const camera: string[] = [];
  for (const m of MOVES) {
    if (!g.camera?.[m]?.move) continue;
    const text = m === "push in" ? "slow push in" : m;
    const shot = parse(`a man standing in a park, ${text}`);
    if (shot?.camera.move) camera.push(text);
  }
  cache = { grammar, pools: { place, action, camera } };
  return cache.pools;
}

/** Six fragments (two of each kind) for a seed. The same seed always gives the same six. */
export function tryFragments(grammar: object | null, seed: number, parse: Parse = readShot): TryFragment[] {
  const pools = tryPools(grammar, parse);
  if (!pools) return [];
  const out: TryFragment[] = [];
  const pick = (kind: TryFragment["kind"], step: number) => {
    const list = pools[kind];
    if (!list.length) return;
    const start = ((seed * step) % list.length + list.length) % list.length;
    for (let i = 0; i < Math.min(2, list.length); i++) out.push({ kind, text: list[(start + i * 7) % list.length] });
  };
  pick("place", 5);
  pick("action", 3);
  pick("camera", 2);
  const seen = new Set<string>();
  return out.filter((f) => (seen.has(f.text) ? false : (seen.add(f.text), true)));
}

/** Put a fragment into the sentence at the cursor, with a comma or a space where it reads right. */
export function insertFragment(text: string, at: number, fragment: string): { text: string; cursor: number } {
  const pos = Math.max(0, Math.min(text.length, at));
  const before = text.slice(0, pos).replace(/\s+$/, "");
  const after = text.slice(pos).replace(/^\s+/, "");
  let lead = "";
  if (before) lead = /[,.;!?]$/.test(before) ? " " : /^(on|in)\b/.test(fragment) ? " " : ", ";
  const sentenceStart = !before || /[.!?]$/.test(before);
  const body = sentenceStart ? fragment[0].toUpperCase() + fragment.slice(1) : fragment;
  const tail = after ? (/^[,.;!?]/.test(after) ? "" : /^[A-Z]/.test(after) ? ". " : " ") : "";
  const next = `${before}${lead}${body}${tail}${after}`;
  return { text: next, cursor: before.length + lead.length + body.length };
}
