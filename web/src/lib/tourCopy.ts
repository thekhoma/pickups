// The demo's words: the project, its cast, the typed prompts. Built-in copy is generic; a machine that plays its
// own example loops can put matching copy in /examples/tour.json (served from ~/.pickups/examples, never in the repo).
import type { CastKind } from "../types";

export interface TourCastEntry {
  name: string;
  kind: CastKind;
  line: string;
  /** Relative to /examples/. Without it the mock's own pictures are used. */
  image?: string;
}

export interface TourCopy {
  project: string;
  cast: TourCastEntry[];
  previz_shot: string;
  /** Typed into Render previz → Look. `@Name` mentions are picked from the @-list as they're typed. */
  prompt: string;
  /** The Prompt Director's answer in the tour (optional; otherwise the mock writes one). */
  director_text?: string;
  /** The Change weather shot's name (optional in tour.json, like the previz shot's; the generic one otherwise). */
  weather_shot: string;
  image_prompt: string;
  image_cast_name: string;
  image_cast_kind: CastKind;
}

export const DEFAULT_TOUR_COPY: TourCopy = {
  project: "Stunt previz",
  cast: [
    { name: "Stunt driver", kind: "person", line: "an old stunt driver with a weathered face, a white walrus moustache and a cream open-face helmet with navy stars, in a sage-green work shirt" },
    { name: "Hood view", kind: "place", line: "the stripped cabin of the grey stunt car seen through its dusty windshield from the hood, with racks of wrecked cars behind" },
  ],
  previz_shot: "K03 hood push",
  prompt: "@Stunt driver sits behind the wheel in @Hood view as the camera pushes in; he draws a knuckle along his moustache and breathes out. Warm low evening sun, fine film grain.",
  weather_shot: "Castle on the loch",
  image_prompt: "@Stunt driver at the wheel in the salvage yard at golden hour, warm low sun through the dusty windshield, 35 mm film still.",
  image_cast_name: "Golden hour",
  image_cast_kind: "look",
};

const KINDS: CastKind[] = ["person", "place", "prop", "look"];
let loading: Promise<TourCopy | null> | null = null;

/** /examples/tour.json, validated and completed with the defaults; null when there isn't one. */
export function loadTourCopy(): Promise<TourCopy | null> {
  loading ??= (async () => {
    try {
      const ctrl = new AbortController();
      const t = window.setTimeout(() => ctrl.abort(), 1500);
      const res = await fetch("/examples/tour.json", { signal: ctrl.signal, headers: { Accept: "application/json" } });
      window.clearTimeout(t);
      if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) return null;
      const raw = (await res.json()) as Partial<Record<keyof TourCopy, unknown>>;
      if (!raw || typeof raw !== "object") return null;
      const str = (v: unknown, d: string) => (typeof v === "string" && v.trim() ? v.trim() : d);
      const cast = Array.isArray(raw.cast)
        ? (raw.cast as Array<Record<string, unknown>>)
            .filter((c) => c && typeof c.name === "string" && c.name.trim())
            .map((c) => ({
              name: String(c.name).trim(),
              kind: KINDS.includes(c.kind as CastKind) ? (c.kind as CastKind) : "person",
              line: str(c.line, ""),
              image: typeof c.image === "string" && c.image ? c.image : undefined,
            }))
        : [];
      if (!cast.length) return null;
      const d = DEFAULT_TOUR_COPY;
      return {
        project: str(raw.project, d.project),
        cast,
        previz_shot: str(raw.previz_shot, d.previz_shot),
        prompt: str(raw.prompt, `@${cast[0].name}`),
        director_text: typeof raw.director_text === "string" && raw.director_text.trim() ? raw.director_text.trim() : undefined,
        weather_shot: str(raw.weather_shot, d.weather_shot),
        image_prompt: str(raw.image_prompt, d.image_prompt),
        image_cast_name: str(raw.image_cast_name, `${cast[0].name} still`),
        image_cast_kind: KINDS.includes(raw.image_cast_kind as CastKind) ? (raw.image_cast_kind as CastKind) : "person",
      };
    } catch {
      return null;
    }
  })();
  return loading;
}

/** "@Stunt driver sits in @Hood view" → "{Stunt driver} sits in {Hood view}" (the autopilot picks each from the @-list). */
export function mentionScript(text: string, names: string[]): string {
  const sorted = [...names].sort((a, b) => b.length - a.length);
  let out = "";
  for (let i = 0; i < text.length; ) {
    if (text[i] === "@") {
      const hit = sorted.find((n) => text.slice(i + 1, i + 1 + n.length).toLowerCase() === n.toLowerCase() && !/[\p{L}\p{N}_]/u.test(text[i + 1 + n.length] ?? ""));
      if (hit) {
        out += `{${hit}}`;
        i += hit.length + 1;
        continue;
      }
    }
    out += text[i];
    i++;
  }
  return out;
}
