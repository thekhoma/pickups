// The Look prompt a made previz arrives with: who the grey figure is and where, from the sentence and the cast.
// Rule-based, no model: the same sentence and cast always give the same words.
import type { CastMember } from "../types";

/** Camera talk the look frame doesn't need (Qwen paints one still; the move is the playblast's job). */
export const CAMERA =
  /\b(camera|shot|lens|frame|framing|push(es|ed)?|pull(s|ed)?|dolly|pan(s|ned)?|tilt(s|ed)?|zoom(s|ed)?|orbit(s|ing)?|crane(s|d)?|track(s|ing)?|boom(s)?|drone|aerial|close[- ]?up|wide|slow(ly)? (moves?|rises?|drifts?)|cut)\b/i;
const OPENER = /^(a|an|the|one|two|three|four|five|some|several|his|her|their|my)\b/i;

function sentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** "A man ..." becomes "a man ..." after "The grey figure is". Names keep their capital. */
function lowerFirst(s: string): string {
  return OPENER.test(s) ? s[0].toLowerCase() + s.slice(1) : s;
}

function ending(s: string): string {
  return /[.!?]$/.test(s) ? s : `${s}.`;
}

/** Cast entries the sentence names ("Maya", or "@Maya"). */
export function castIn(text: string, cast: CastMember[] | undefined): CastMember[] {
  if (!cast?.length) return [];
  return cast.filter((c) => c.name.trim() && new RegExp(`(^|[^\\w])@?${c.name.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(text));
}

/** The look line every made previz ends with (the creative director's words, verbatim): the first light the user's
 * sentence names wins, in this order, else the default. Never the word "camera" (the look frame is one still). */
const FILM = "A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean.";
export const LOOKS: { id: "night" | "interior" | "golden" | "default"; words: RegExp | null; line: string }[] = [
  { id: "night", words: /\b(night|midnight|dusk|neon|after dark)\b/i, line: `${FILM} Practical lights glow against deep blue shadows, soft highlight roll-off, fine grain.` },
  {
    id: "interior",
    words: /\b(interior|indoors|inside|room|kitchen|diner|booth|office|apartment|hallway)\b/i,
    line: `${FILM} Soft window light falls across the room with gentle falloff into the shadows, fine grain.`,
  },
  {
    id: "golden",
    words: /\b(golden hour|sunset|sunrise|dawn|early morning|late afternoon|low sun)\b/i,
    line: `${FILM} Low golden sun rakes across the scene with long soft shadows and warm highlights, fine grain.`,
  },
  { id: "default", words: null, line: `${FILM} Soft directional light that suits the place and time, shallow depth of field, fine grain.` },
];

export function lookLineFor(text: string): string {
  return LOOKS.find((l) => !l.words || l.words.test(text))!.line;
}

/** The Look words without the fixed look line: only what the sentence and the cast said (the video line reads that). */
export function withoutLookLine(look: string): string {
  let out = look;
  for (const l of LOOKS) out = out.split(l.line).join(" ");
  return out.replace(/\s+/g, " ").trim();
}

/** The time of day the sentence names (FableWalk: "dusk" was lost when it came last), as the Look's first words. */
const TIME = /\b(early morning|late afternoon|golden hour|blue hour|after dark|dawn|sunrise|morning|midday|noon|afternoon|sunset|dusk|twilight|evening|night|midnight)\b/i;
const CLOTHES =
  "(?:trench ?coat|overcoat|raincoat|coat|jacket|parka|anorak|dress|gown|suit|tuxedo|t-shirt|shirt|blouse|hoodie|sweater|jumper|cardigan|vest|uniform|armou?r|robe|kimono|apron|overalls|jeans|trousers|pants|shorts|skirt|leggings|boots|shoes|sneakers|trainers|hat|cap|helmet|beanie|scarf|gloves|sunglasses|glasses|mask|costume|poncho|cape|cloak)(?:es|s)?";
/** "in a long coat", "wearing a red dress and boots": what someone wears (only clothes, never "in a diner"). */
const WEARS = new RegExp(`\\b(?:in|wearing|dressed in)\\s+((?:(?:an?|the|his|her|their)\\s+)?(?:[\\w-]+\\s+){0,3}?${CLOTHES}(?:\\s+and\\s+(?:an?\\s+)?(?:[\\w-]+\\s+){0,2}?${CLOTHES})?)\\b`, "i");
const NUMBER = /^(two|three|four|five|several|some)\b/i;
const STOP = /^(in|wearing|dressed|with|who|and|is|are|was|on|at|by|near|walks?|stands?|sits?|sitting|standing|walking|runs?|running|rides?|riding|drives?|driving|waits?|waiting|looks?|looking)$/i;

/** Who wears it: the sentence's first person ("a man in…" → "The man"; "Maya in…" → "Maya"), plural when it says so. */
function wearer(sentence: string, people: number): { who: string; plural: boolean } {
  const words = sentence.replace(/[,.;!?]/g, " ").split(/\s+/).filter(Boolean);
  const opener = words[0] ?? "";
  if (OPENER.test(opener)) {
    const noun: string[] = [];
    for (const w of words.slice(1)) {
      if (STOP.test(w) || noun.length >= 3) break;
      noun.push(w.toLowerCase());
    }
    const plural = NUMBER.test(opener) || people > 1;
    if (noun.length) return { who: `The ${NUMBER.test(opener) ? `${opener.toLowerCase()} ` : ""}${noun.join(" ")}`, plural };
  }
  if (/^[A-Z][a-z]+$/.test(opener)) return { who: opener, plural: false };
  return { who: people > 1 ? "They" : "The figure", plural: people > 1 };
}

/** The Look's lead: the time of day, then what they wear, so a still model reads them first (the final render dropped
 * "dusk" and "long coat" when they came last: FableWalk, Oct 27 2026). Empty when the sentence names neither. */
export function lookLead(text: string, people = 1): string {
  const scene = sentences(text).filter((s) => !CAMERA.test(s));
  const out: string[] = [];
  const time = scene.map((s) => s.match(TIME)?.[1]).find(Boolean);
  if (time) out.push(`It is ${time.toLowerCase()}.`);
  for (const s of scene) {
    const m = s.match(WEARS);
    if (!m) continue;
    const { who, plural } = wearer(s, people);
    out.push(`${who} ${plural ? "wear" : "wears"} ${m[1]}.`);
    break;
  }
  return out.join(" ");
}

/** Verbs a previz sentence acts with ("A man in a long coat stands on a pier"), so its subject can be split off. */
const ACTS =
  "stand|walk|sit|run|ride|drive|wait|lean|turn|step|cross|move|stroll|jog|dance|talk|hold|carry|climb|kneel|lie|read|smoke|watch|wave|enter|leave|arrive|approach|head|hurry|race|cruise|pause|gaze|stare|sing|play|fight|swim|fall|jump|reach|smile|laugh|look|hug|kiss|sip|drink|eat|go|come|wander|linger|rest|pace|stop";
const ONE_ACTS = new RegExp(`^(?:is|was|has|who|that|(?:${ACTS})(?:s|es)|(?:carr|hurr)ies|lies)$`, "i");
const MANY_ACTS = new RegExp(`^(?:are|were|have|who|that|${ACTS})$`, "i");

/** "a man in a long coat stands on a pier" → "The grey figure is a man in a long coat who stands on a pier.": one verb,
 * plain English (Run 9 doubled it: "is a man in a long coat stands"). */
export function figureLine(first: string, people = 1): string {
  const s = lowerFirst(first).replace(/[.!?]+$/, "");
  const many = people > 1 || NUMBER.test(s);
  const words = s.split(" ");
  const at = words.findIndex((w, i) => i > 1 && (many ? MANY_ACTS : ONE_ACTS).test(w.replace(/[,;]$/, "")));
  const head = `The grey ${many ? "figures are" : "figure is"}`;
  if (at < 0) return `${head} ${s}.`;
  const verb = words[at].toLowerCase();
  const rest = words.slice(at).join(" ");
  return `${head} ${words.slice(0, at).join(" ")} ${verb === "who" || verb === "that" ? "" : "who "}${rest}.`.replace(/\s+/g, " ");
}

export function lookPromptFor(text: string, cast?: CastMember[], people = 1): string {
  const all = sentences(text);
  if (!all.length) return "";
  const scene = all.filter((s) => !CAMERA.test(s));
  const lines = scene.length ? scene : [all[0]];
  const parts: string[] = [];
  const lead = lookLead(text, people);
  if (lead) parts.push(lead);
  // Who the grey figure is, in one plain sentence (the server turns it into plain words for the models:
  // prompt_words.py).
  const first = lines[0];
  if (OPENER.test(first)) parts.push(figureLine(first, people));
  else parts.push(ending(first.charAt(0).toUpperCase() + first.slice(1)));
  parts.push(...lines.slice(1).map(ending));
  for (const c of castIn(text, cast)) {
    if (c.line.trim()) parts.push(ending(`${c.name.trim()} is ${c.line.trim()}`));
  }
  parts.push(lookLineFor(text));
  return parts.join(" ");
}
