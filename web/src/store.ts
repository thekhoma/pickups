// A tiny global store (useSyncExternalStore). Selectors must return existing references or primitives.
import { LAYOUT_SWITCH } from "./lib/features";
import { loadDrafts, type Draft } from "./lib/draftStore";
import type { ModelsDownload, ModelsHelp } from "./lib/servers";
import { useSyncExternalStore } from "react";
import type { CastMember, CropAspect, Delivery, DirectorSettings, Example, ImageModels, Job, JobRequest, ModelCatalog, Point, Project, ProjectImage, Sample, Shot, Status, TaskId, TrashEntry } from "./types";

export type ViewMode = "before" | "after" | "receipt";

export type Rail =
  | { kind: "tiles" }
  | { kind: "form"; task: TaskId }
  | { kind: "running"; jobId: string }
  | { kind: "result"; versionId: string }
  /** Trim & crop: pick in and out and a shape, then make a new shot from the original. */
  | { kind: "prepare" };

export interface ShotUi {
  rail: Rail;
  /** The version shown in the viewer (and the base for the next task). */
  versionId: string;
  view: ViewMode;
  wipe: boolean;
  wipePos: number;
  trimDismissed: boolean;
  /** The take on the wipe's left side (default: the take this one was made from). */
  beforeId: string | null;
  showHidden: boolean;
  /** The take a step flow's Review is approving from the rail: the player bar leaves its Approve to it. */
  railApprove?: string | null;
}

export interface Selection {
  shotId: string;
  frame: number;
  points: Point[];
  text: string;
  maskUrl: string | null;
  loading: boolean;
  error: string | null;
  seq: number;
  /** A take's pick put back as it was ("the last take", "the example"): selected without a find, so no mask yet. */
  kept?: string;
  /** Recast's people finder made this pick and the user hasn't changed it: a click on someone else switches to them. */
  auto?: boolean;
  /** How many of `points` (the first ones) came from the finder, when the user then clicked more on that person. */
  autoBase?: number;
  /** Recast's people finder is still looking (the picture's chip says so too). */
  finding?: boolean;
  /** Recast anyone: a click on someone else the finder found swaps them too (the picture's chip says so). */
  many?: boolean;
}

/** Trim & crop's draft for the open shot: in and out frames (the out frame is kept), the shape ("original": no crop)
 * and the crop box in the source's pixels. Drawn on the filmstrip and over the picture while the rail is open. */
export interface PrepareDraft {
  shotId: string;
  inFrame: number;
  outFrame: number;
  aspect: "original" | CropAspect;
  box: { x: number; y: number; w: number; h: number } | null;
  busy: boolean;
  error: string | null;
  /** Opened from the shot menu: the second button is Cancel. A new clip still says Skip. */
  fromMenu?: boolean;
  /** Opened from a task form ("Long shot → Trim"): the new clip opens on that form again. */
  task?: TaskId;
}

/** Recast's window on the timeline: LTX-2.5 renders `frames` frames of a longer shot, from `start`. */
export interface RenderWindow {
  shotId: string;
  start: number;
  frames: number;
  /** The user set its length with the in and out handles (or dragged it), so the form sends `frames` too. */
  sized?: boolean;
  /** The user dragged it, so the form sends `start_frame`. Otherwise the backend places it where they're on screen most. */
  moved: boolean;
  /** It can't move (SCAIL-2 works on the first frames of the shot). */
  fixed: boolean;
}

export type ProjectTab = "shots" | "images" | "cast" | "deliveries";

export type Route =
  | { name: "home" }
  | { name: "projects" }
  | { name: "create"; panel: "home" | "image" | "edit" }
  | { name: "assets" }
  | { name: "previz"; projectId: string | null }
  | { name: "project"; id: string; tab: ProjectTab }
  | { name: "shot"; id: string };

export interface Toast {
  id: number;
  text: string;
  /** "queued": the work is waiting for the GPU, nothing is done yet. */
  tone: "info" | "error" | "success" | "queued";
  action?: { label: string; run: () => void };
}

/** The home Create bar's ways in: open a clip to edit, make an image, build a character. */
export type CreateMode = "clip" | "image" | "t2v" | "i2v" | "character" | "previz";

export interface UploadState {
  state: "idle" | "uploading" | "processing" | "error";
  name?: string;
  progress?: number;
  error?: string;
}

export type OnboardingStep = "connect" | "scan" | "choose" | "fix" | "done";

export interface AppState {
  mode: "live" | "mock" | null;
  status: Status | null;
  statusFailed: boolean;
  samples: Sample[] | null;
  samplesFailed: boolean;
  /** Assets > Trash (null until it's opened). */
  trash: TrashEntry[] | null;
  shots: Record<string, Shot>;
  shotsFailed: boolean;
  shotErrors: Record<string, string>;
  jobs: Record<string, Job>;
  jobOrder: string[];
  /** Requests sent this session, for "Try again". */
  requests: Record<string, JobRequest>;
  route: Route;
  /** The shot a page was opened from (a cast sheet from Previz to final): that page's Back returns to it. */
  backToShot: string | null;
  /** `section`: a Setup section to scroll to (its title, as "prompt-director"). */
  setup: { open: boolean; focus: TaskId | null; section?: string };
  /** The first-run flow (welcome, connect, what this machine can do), also reopened by "Change server". */
  /** The guided setup: connect, scan, choose features, fix what's missing, done. `task` opens Choose on that card. */
  onboarding: { open: boolean; step: OnboardingStep; task?: string | null };
  /** A model download running on this computer, or how to get the models onto another one. */
  modelsDownload: ModelsDownload | null;
  modelsHelp: ModelsHelp | null;
  ui: Record<string, ShotUi>;
  selection: Selection | null;
  upload: UploadState;
  sampleOpening: string | null;
  queueOpen: boolean;
  dismissed: Record<string, true>;
  toasts: Toast[];
  modelsFetching: boolean;
  examples: Example[] | null;
  /** Each home workflow's default models (null until loaded, or on a backend without them). */
  defaultModels: ModelCatalog | null;
  /** The downloadable example gallery (only fetched when there are no examples). */
  examplesPack: (import("./types").ExamplesPack & { busy?: boolean }) | null;
  /** A recipe picked on Home, waiting for a shot. */
  pendingRecipe: string | null;
  /** Recipes in progress, per shot: which step is next. */
  recipeRuns: Record<string, { recipeId: string; step: number }>;
  /** Which recipe step a job belongs to. */
  jobSteps: Record<string, { recipeId: string; step: number }>;
  path: { busy: boolean; error: string | null };
  exampleOpening: string | null;
  projects: Project[] | null;
  projectsFailed: boolean;
  /** Where new shots go. Remembered per browser ("" until the projects load and one is picked). */
  currentProjectId: string;
  /** Shot ids per project, newest first (null while loading). */
  projectShots: Record<string, string[] | null>;
  cast: Record<string, CastMember[]>;
  deliveries: Record<string, Delivery[] | null>;
  /** "Try again with changes": the params a task form starts from. */
  prefill: Record<string, { task: TaskId; params: Record<string, unknown>; seq: number } | undefined>;
  imageModels: ImageModels | null;
  imageModelsFailed: boolean;
  /** Images per project, newest first (null while loading). */
  images: Record<string, ProjectImage[] | null>;
  /** Prompt director settings (null until loaded). */
  director: DirectorSettings | null;
  /** GET /api/settings → comfy_url (for "Open ComfyUI"). */
  comfyUrl: string | null;
  /** Settings handed to the Image page from elsewhere ("Edit", "Same settings"). */
  imageDraft: { seq: number; patch: Record<string, unknown> } | null;
  /** Recast's render window, drawn on the open shot's filmstrip while the form is open. */
  renderWindow: RenderWindow | null;
  /** Trim & crop's draft (null: not open). */
  prepare: PrepareDraft | null;
  /** The home Create bar: its mode, and words handed to it (a gallery card's prompt). `seq` changes on every hand-off. */
  createBar: { mode: CreateMode; seq: number; text?: string; file?: File | null };
  /** The left panel: collapsed to icons (remembered per browser), and open as a drawer (phones). */
  nav: { collapsed: boolean; drawer: boolean };
  /** Side panel, or the same links in the top bar. Remembered so the two layouts can be compared. */
  chrome: Chrome;
  /** What forms and step flows have typed and picked, by `${shotId}:${task}` (lib/drafts.ts), mirrored to
   * localStorage `pickups.drafts.v1`. */
  drafts: Record<string, Draft>;
}

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [a, b, c] = h.split("/");
  if (a === "library" || a === "projects") return { name: "projects" };
  // #/image is the old Image tab. It opens Create → Image.
  if (a === "image" || (a === "create" && b === "image")) return { name: "create", panel: "image" };
  if (a === "create" && b === "edit") return { name: "create", panel: "edit" };
  if (a === "create") return { name: "create", panel: "home" };
  if (a === "assets") return { name: "assets" };
  if (a === "previz") return { name: "previz", projectId: null };
  if (a === "project" && b && c === "previz") return { name: "previz", projectId: decodeURIComponent(b) };
  if (a === "project" && b) return { name: "project", id: decodeURIComponent(b), tab: c === "cast" || c === "deliveries" || c === "images" ? c : "shots" };
  if (a === "shot" && b) return { name: "shot", id: decodeURIComponent(b) };
  return { name: "home" };
}

/** Mock / tour sessions remember their own project, so they never disturb the real workspace. */
const MOCKED = typeof window !== "undefined" && /[?&](mock|tour)(=|&|$)/.test(window.location.search);
const PROJECT_KEY = MOCKED ? "pickups.mock.project" : "pickups.project";

/** Where the main links live. "side" is the left panel; "top" is the bar, with no panel. */
export type Chrome = "side" | "top";

const NAV_KEY = "pickups.nav.collapsed";
const CHROME_KEY = "pickups.chrome";

export function rememberedNav(): boolean {
  try {
    return window.localStorage.getItem(NAV_KEY) === "1";
  } catch {
    return false;
  }
}

export function rememberNav(collapsed: boolean): void {
  try {
    window.localStorage.setItem(NAV_KEY, collapsed ? "1" : "0");
  } catch {
    /* private mode: just don't remember */
  }
}

export function rememberedChrome(): Chrome {
  try {
    // Top for a new install; once someone picks a layout, theirs.
    if (!LAYOUT_SWITCH) return "side";
    return window.localStorage.getItem(CHROME_KEY) === "side" ? "side" : "top";
  } catch {
    return "top";
  }
}

export function rememberChrome(chrome: Chrome): void {
  try {
    window.localStorage.setItem(CHROME_KEY, chrome);
  } catch {
    /* private mode: just don't remember */
  }
}

export function rememberedProject(): string {
  try {
    return window.localStorage.getItem(PROJECT_KEY) || "";
  } catch {
    return "";
  }
}

export function rememberProject(id: string): void {
  try {
    window.localStorage.setItem(PROJECT_KEY, id);
  } catch {
    /* private mode: just don't remember */
  }
}

let state: AppState = {
  mode: null,
  status: null,
  statusFailed: false,
  samples: null,
  samplesFailed: false,
  trash: null,
  shots: {},
  shotsFailed: false,
  shotErrors: {},
  jobs: {},
  jobOrder: [],
  requests: {},
  route: parseHash(window.location.hash),
  backToShot: null,
  setup: { open: false, focus: null },
  onboarding: { open: false, step: "connect" },
  modelsDownload: null,
  modelsHelp: null,
  ui: {},
  selection: null,
  upload: { state: "idle" },
  sampleOpening: null,
  queueOpen: false,
  dismissed: {},
  toasts: [],
  modelsFetching: false,
  examples: null,
  defaultModels: null,
  examplesPack: null,
  pendingRecipe: null,
  recipeRuns: {},
  jobSteps: {},
  path: { busy: false, error: null },
  exampleOpening: null,
  projects: null,
  projectsFailed: false,
  currentProjectId: rememberedProject(),
  projectShots: {},
  cast: {},
  deliveries: {},
  prefill: {},
  imageModels: null,
  imageModelsFailed: false,
  images: {},
  director: null,
  comfyUrl: null,
  imageDraft: null,
  renderWindow: null,
  prepare: null,
  createBar: { mode: "clip", seq: 0 }, // footage first: Edit clip until something asks for another tab
  nav: { collapsed: rememberedNav(), drawer: false },
  chrome: rememberedChrome(),
  drafts: loadDrafts(typeof window !== "undefined" ? window.localStorage : null),
};

const listeners = new Set<() => void>();

export function getState(): AppState {
  return state;
}

export function setState(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)): void {
  const p = typeof patch === "function" ? patch(state) : patch;
  state = { ...state, ...p };
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useApp<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(subscribe, () => selector(state));
}

export const DEFAULT_UI: ShotUi = {
  rail: { kind: "tiles" },
  versionId: "orig",
  view: "after",
  wipe: true,
  wipePos: 50,
  trimDismissed: false,
  beforeId: null,
  showHidden: false,
};
