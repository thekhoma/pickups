import { CARS } from "./features";
import type { Example, FinishParams, TaskId } from "../types";

/** Chained workflows on the home screen. Steps run one after another on the latest take. */
export interface Recipe {
  id: string;
  title: string;
  /** What the step chips say. */
  chips: string[];
  /** The steps as a sentence (slates, screen readers). */
  line: string;
  /** The tasks that actually run, in order. */
  tasks: TaskId[];
  /** What the first step's form starts from (a preset's settings). */
  params?: Record<string, unknown>;
}

export const RECIPES: Recipe[] = [
  { id: "recast", title: "Replace a person and finish", chips: ["Replace a person", "Finish"], line: "Replace a person, then finish", tasks: ["recast", "finish"] },
  { id: "clean-plate", title: "Clean plate", chips: ["Remove a person", "Finish"], line: "Remove a person, then finish", tasks: ["remove", "finish"] },
  { id: "new-weather", title: "Weather and finish", chips: ["Change the weather", "Finish"], line: "Change the weather, then finish", tasks: ["environment", "finish"] },
  { id: "blockout", title: "Previz to master", chips: ["Previz to final", "Finish"], line: "Previz to final, then finish", tasks: ["previz", "finish"] },
  { id: "clean-up", title: "Two-pass clean-up", chips: ["Remove a person", "Remove again"], line: "Remove a person, then remove again", tasks: ["remove", "remove"] },
];

/** Finish runs by itself inside a recipe, with its form defaults. */
export const DEFAULT_FINISH: FinishParams = { upscale_4k: true, hdr: true, quality: "best" };

/** One-tap presets under the home Create bar: one task, its form prefilled. "character" opens the bar's Character mode. */
export interface Preset extends Recipe {
  /** The chip's word ("Snow"). */
  chip: string;
  icon: "snow" | "night" | "autumn" | "remove" | "car" | "character" | "previz" | "relight" | "finish";
  /** Builds a character sheet instead of opening a shot. */
  character?: boolean;
}

const preset = (id: string, chip: string, icon: Preset["icon"], task: TaskId, line: string, params?: Record<string, unknown>): Preset => ({
  id: `preset:${id}`,
  chip,
  icon,
  title: chip,
  chips: [chip],
  line,
  tasks: [task],
  params,
});

export const PRESETS: Preset[] = [
  preset("snow", "Snow", "snow", "environment", "Change the weather to snow", { change: "snow" }),
  preset("night", "Night", "night", "environment", "Turn day into night", { change: "night" }),
  preset("autumn", "Autumn", "autumn", "environment", "Change the season to autumn", { change: "autumn" }),
  preset("relight", "Relight a scene", "relight", "relight", "New light on the whole shot", { look: "sunset" }),
  preset("remove-person", "Remove a person", "remove", "remove", "Take a person out of the shot", { text: "person" }),
  preset("swap-car", "Swap a car", "car", "recast", "Put a different car in the shot", { text: "car" }),
  { ...preset("character", "New character", "character", "recast", "Build a character sheet"), character: true },
  preset("previz", "Previz to final", "previz", "previz", "Turn a previz into a finished shot"),
  preset("finish", "Finish", "finish", "finish", "Upscale and grade a master", { ...DEFAULT_FINISH }),
];

/** Presets not offered yet: they don't pass the quality bar on two clips (the code paths stay; a saved take still
 * opens its preset). */
export const HIDDEN_PRESETS = new Set(["night", "autumn", "fog"]);
/** Preset ids are "preset:<name>"; HIDDEN_PRESETS holds the names (the weather form shares them). Car swap is off in
 * this build (lib/features). */
export const OFFERED_PRESETS = PRESETS.filter(
  (p) => !HIDDEN_PRESETS.has(p.id.replace(/^preset:/, "")) && (CARS || p.icon !== "car"),
);

/** Presets made on the fly ("Use on my clip" on an example that has no shot to open here). */
const made = new Map<string, Recipe>();

/** The settings of an example that make sense on someone else's clip: the look, not its frames and clicks. */
const PORTABLE: Record<TaskId, string[]> = {
  remove: ["text", "shadow"],
  recast: ["text", "describe", "engine"],
  environment: ["change", "custom", "keep_text", "control"],
  previz: [],
  relight: ["look", "custom", "direction"],
  reshoot: ["move", "amount"],
  finish: ["upscale_4k", "hdr", "quality"],
};

/** "Use on my clip": a one-step preset from an example, waiting for a clip. */
export function presetFromExample(ex: Example): Recipe {
  const keep = PORTABLE[ex.task] ?? [];
  const params: Record<string, unknown> = {};
  for (const k of keep) if (ex.params && ex.params[k] != null && ex.params[k] !== "") params[k] = ex.params[k];
  // A custom weather is its words: keep them only when the example says what it changes into.
  if (params.change === "custom" && !params.custom) delete params.change;
  const r: Recipe = { id: `example:${ex.id}`, title: ex.title, chips: [ex.title], line: ex.line, tasks: [ex.task], params };
  made.set(r.id, r);
  return r;
}

export const recipeById = (id: string | null | undefined): Recipe | null =>
  !id ? null : RECIPES.find((r) => r.id === id) ?? PRESETS.find((r) => r.id === id) ?? made.get(id) ?? null;

/** A one-step preset (no recipe strip, no chain): its form just opens prefilled. */
export const isPreset = (r: Recipe | null | undefined): boolean => !!r && (r.id.startsWith("preset:") || r.id.startsWith("example:"));
