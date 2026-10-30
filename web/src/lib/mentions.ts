// @-mentions of a project's cast. Parsing mirrors the backend (pickups/mentions.py): "@" + a cast name, longest
// name wins, case-insensitive, whole words. The expansion functions are a port used by the mock API.
import type { CastMember, MentionPreview } from "../types";

export interface Segment {
  text: string;
  cast?: CastMember;
}

function matchAt(text: string, i: number, cast: CastMember[]): CastMember | null {
  let best: CastMember | null = null;
  for (const c of cast) {
    const j = i + 1 + c.name.length;
    if (text.slice(i + 1, j).toLowerCase() !== c.name.toLowerCase()) continue;
    const next = text[j];
    if (j < text.length && next && /[\p{L}\p{N}_]/u.test(next)) continue;
    if (!best || c.name.length > best.name.length) best = c;
  }
  return best;
}

/** Mentions in order: [start, end, cast]. */
export function scan(text: string, cast: CastMember[]): Array<[number, number, CastMember]> {
  const out: Array<[number, number, CastMember]> = [];
  let i = 0;
  while ((i = text.indexOf("@", i)) !== -1) {
    const c = matchAt(text, i, cast);
    if (c) {
      const end = i + 1 + c.name.length;
      out.push([i, end, c]);
      i = end;
    } else i += 1;
  }
  return out;
}

/** Text split into plain runs and mention runs (for the chip overlay). */
export function segments(text: string, cast: CastMember[]): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const [s, e, c] of scan(text, cast)) {
    if (s > last) out.push({ text: text.slice(last, s) });
    out.push({ text: text.slice(s, e), cast: c });
    last = e;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

/** The "@query" being typed just before the caret, if any. */
export function mentionQuery(text: string, caret: number): { start: number; query: string } | null {
  const before = text.slice(0, caret);
  const at = before.lastIndexOf("@");
  if (at < 0) return null;
  if (at > 0 && /[\p{L}\p{N}_]/u.test(before[at - 1])) return null; // an e-mail, not a mention
  const query = before.slice(at + 1);
  if (query.includes("\n") || query.length > 32) return null;
  return { start: at, query };
}

export function matchingCast(cast: CastMember[], query: string): CastMember[] {
  const q = query.trim().toLowerCase();
  return cast
    .filter((c) => !q || c.name.toLowerCase().startsWith(q) || c.name.toLowerCase().split(/\s+/).some((w) => w.startsWith(q)))
    .slice(0, 8);
}

// --- expansion (port of mentions.py, for the mock) --------------------------------------------------------------

const NOUN: Record<string, string> = { person: "person", place: "place", prop: "object", look: "look" };
const END = ",.;:!?)";

export function forImages(text: string, cast: CastMember[], firstImage = 2): [string, CastMember[]] {
  const refs: CastMember[] = [];
  let out = "";
  let last = 0;
  for (const [s, e, c] of scan(text, cast)) {
    if (!refs.some((r) => r.id === c.id)) refs.push(c);
    const n = firstImage + refs.findIndex((r) => r.id === c.id);
    const sub =
      c.kind === "look" ? `the look of <image${n}>` : c.kind === "place" ? `${c.name} (the place in <image${n}>)` : `${c.name} (the ${NOUN[c.kind] ?? "person"} in <image${n}>)`;
    out += text.slice(last, s) + sub;
    last = e;
  }
  return [out + text.slice(last), refs];
}

export function forText(text: string, cast: CastMember[]): string {
  let out = "";
  let last = 0;
  const done = new Set<string>();
  for (const [s, e, c] of scan(text, cast)) {
    const line = (c.line || "").trim().replace(/\.$/, "");
    let sub: string;
    if (c.kind === "place" || c.kind === "look") sub = line || c.name;
    else if (done.has(c.id) || !line) sub = c.name;
    else {
      const next = text.slice(e, e + 1);
      sub = `${c.name}, ${line}${!next || END.includes(next) ? "" : ","}`;
    }
    done.add(c.id);
    out += text.slice(last, s) + sub;
    last = e;
  }
  return (out + text.slice(last)).replace(/\s{2,}/g, " ").trim();
}

export function tagImages(text: string, nImages: number): string {
  const plain = /(?<![<\w])(?:the\s+)?(?:image|picture)\s*#?\s*(\d+)(?!\w)/gi;
  if (nImages >= 2) return text.replace(plain, (_m, n) => `<image${n}>`);
  return text.replace(/<image(\d+)>/g, "the image").replace(plain, "the image");
}

export function preview(text: string, cast: CastMember[], uploads: number): MentionPreview {
  const [img, refs] = forImages(text, cast, 2 + uploads);
  return {
    images: tagImages(img, 1 + uploads + refs.length),
    text: forText(text, cast),
    refs: refs.map((c, i) => ({ id: c.id, name: c.name, url: c.url, image: i + 2 + uploads })),
  };
}
