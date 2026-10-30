// The params Remove, Relight, Finish and Recast send, built in one place for the old forms and the step flows (StepFlow.tsx),
// so the same picks always give the same job (tests/previz/flow_params.mjs checks it).
import type { JobParams, Point, RecastEngine, RecastParams, Ref, RelightParams, RemoveParams } from "../types";
import { paramsOf, type FinishChoice } from "./finish.ts";

/** Remove: the pick (clicks on a frame, or words), the shadow switch, the bracket when it shows, and from a take's
 * settings its fill words (while the pick is still exactly the take's) and its seed. */
export function removeParams(
  pick: { points: Point[]; text: string; frame: number },
  shadow: boolean,
  win: { start: number; frames: number; sized?: boolean } | null,
  kept: { fill?: string; seed?: number } = {},
): RemoveParams {
  const params: RemoveParams = { points: pick.points, ...(pick.text ? { text: pick.text } : {}), shadow, frame: pick.frame };
  if (win) {
    params.start_frame = win.start;
    if (win.sized) params.frames = win.frames;
  }
  if (kept.fill) params.fill = kept.fill;
  if (typeof kept.seed === "number") params.seed = kept.seed;
  return params;
}

/** Relight any scene: a tested look by its key, or your own words with where the light comes from. */
export function relightParams(look: string, custom: string, direction: string): JobParams {
  return look === "custom" ? { look, custom: custom.trim(), direction: direction as RelightParams["direction"] } : { look };
}

/** Finish: the choice's upscale and HDR, and the quality (Best unless changed). */
export function finishParams(choice: FinishChoice, quality: "fast" | "best"): JobParams {
  return { ...paramsOf(choice), quality };
}

/** The tested looks (tasks/scene_relight.py LOOKS) and where the light comes from (DIRECTIONS), with their labels. */
export const RELIGHT_LOOKS = [
  { id: "sunset", label: "Sunset backlight" },
  { id: "golden", label: "Golden side light" },
  { id: "cool", label: "Cool low sun" },
  { id: "midday", label: "Midday sun" },
  { id: "afternoon", label: "Afternoon light" },
] as const;
export const RELIGHT_FROM = [
  { value: "behind_low", label: "Behind" },
  { value: "left_low", label: "Left" },
  { value: "back_right_low", label: "Back right" },
  { value: "front_right_high", label: "Front right" },
  { value: "front_mid", label: "Front" },
] as const;
export type RelightFrom = (typeof RELIGHT_FROM)[number]["value"];
/** What a relight is relighting to, in words: "Sunset backlight", or the user's own words. */
export function relightLabel(params: { look?: unknown; custom?: unknown } | null | undefined): string {
  if (params?.look === "custom") return String(params.custom || "your light").trim();
  return RELIGHT_LOOKS.find((l) => l.id === params?.look)?.label ?? "the new light";
}

/** The models each run uses, said before it starts (Run 14, walkthrough 12). */
export const RELIGHT_MODEL = "LTX-2.3 Relight LoRA";
export function finishModels(choice: FinishChoice): string {
  return [choice !== "hdr" ? "SeedVR2 3B upscale" : "", choice !== "4k" ? "LTX-2.5 SDR-to-HDR" : ""].filter(Boolean).join(" + ");
}

/** The old form's starting picks, which the flow starts from too. */
export const RELIGHT_DEFAULT = { look: "sunset", direction: "front_mid" as RelightFrom };

/** Who comes in for Replace a person: the first of these that is set (a cast member, a picked image, a photo, a take's
 * still), plus FLUX 3 klein's own caption when it makes them again from it. */
export interface RecastWho {
  cast_id?: string;
  who_ref?: Ref;
  ref_upload_id?: string;
  reference_take?: string;
  reference_prompt?: string;
}

/** Replace a person, for one person: the pick, the engine, who comes in, the words about them, Relight the new person,
 * what a take's settings keep while the person is the same (its prompt, scene and seed), and the bracket (LTX-2.5). */
export function recastParams(
  pick: { points: Point[]; text: string; frame: number },
  engine: RecastEngine,
  who: RecastWho,
  o: { describe?: string; relight?: boolean; kept?: { prompt?: string; scene?: string; seed?: number | null }; win?: { start: number; frames: number; sized?: boolean } | null } = {},
): RecastParams {
  // Replace a person: what was picked is a person (R&D, Run 9: a yellow mannequin's click was read as a car).
  const params: RecastParams = { points: pick.points, ...(pick.text ? { text: pick.text } : {}), kind: "person", frame: pick.frame, engine };
  if (who.cast_id) params.cast_id = who.cast_id;
  else if (who.who_ref) params.who_ref = who.who_ref;
  else if (who.ref_upload_id) params.ref_upload_id = who.ref_upload_id;
  else if (who.reference_take) params.reference_take = who.reference_take;
  if (who.reference_prompt) params.reference_prompt = who.reference_prompt;
  if (o.describe) params.describe = o.describe;
  if (o.relight) params.relight = true;
  if (o.kept) {
    if (o.kept.prompt) params.prompt = o.kept.prompt;
    if (o.kept.scene) params.scene = o.kept.scene;
    if (typeof o.kept.seed === "number") params.seed = o.kept.seed;
  }
  // The part in the bracket is the part rendered, moved or not (what you see is what you get).
  if (engine === "ltx" && o.win) {
    params.start_frame = o.win.start;
    if (o.win.sized) params.frames = o.win.frames;
  }
  return params;
}
