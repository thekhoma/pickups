// Types for the Pickups API contract (docs/ui-spec.md → "API contract").

export type TaskId = "remove" | "recast" | "environment" | "previz" | "relight" | "reshoot" | "finish";
/** Rail order. */
export const TASKS: TaskId[] = ["remove", "recast", "environment", "relight", "previz", "reshoot", "finish"];
/** Jobs also make images, character sheets and new videos (text or image to video: a new shot) for a project. */
export type JobTask = TaskId | "image" | "sheet" | "video";

/** A picked reference: a cast entry, a project image or an upload. */
export interface Ref {
  kind: "cast" | "image" | "upload";
  id: string;
}

export type Verdict = "ready" | "slower" | "untested" | "remote" | "offline";

export interface MissingModel {
  file: string;
  gb: number;
  gated: boolean;
  url: string;
  /** The model's page, where a gated licence is accepted. */
  page?: string;
}

export interface MissingPack {
  pack: string;
  url: string;
  by: string;
}

export interface TileStatus {
  ready: boolean;
  eta_s: number | null;
  missing: MissingModel[];
  /** Custom node packs the task needs that this ComfyUI doesn't have (never auto-installed). */
  missing_nodes?: MissingPack[];
  /** Recast: the time each engine takes here (eta_s is the default engine's). */
  eta_by_engine?: Partial<Record<RecastEngine, number>>;
  /** Finish: each choice's own readiness and missing files (4K alone needs only SeedVR2). */
  choices?: Partial<Record<"4k-hdr" | "hdr" | "4k", { ready: boolean; missing: MissingModel[] }>>;
}

export interface Machine {
  gpu: string;
  vram_gb: number;
  ram_gb: number;
  comfy_version: string;
  /** Optional friendly machine name (e.g. "Studio"). Not in the v1 contract; shown in the status pill when present. */
  name?: string;
  /** Optional free VRAM right now, in GB ("18 GB FREE" in the status). */
  vram_free_gb?: number;
  /** Where the GPU is: this computer, a PC on the network, or a RunPod pod. */
  kind?: MachineKind;
}

export type MachineKind = "local" | "lan" | "runpod";

/** Setup → Machines: a ComfyUI Pickups can use, and whether it answers right now (null: not checked). */
export interface SavedMachine {
  id: string;
  name: string;
  kind: MachineKind;
  url: string;
  status?: MachineProbe | null;
}

export interface MachineProbe {
  reachable: boolean;
  gpu: string | null;
  vram_gb: number | null;
}

export interface MachinesList {
  active: string;
  machines: SavedMachine[];
  /** After an add: the new machine's id and what its address answered. */
  added?: string;
  probe?: MachineProbe;
  /** After a switch: the new status. */
  status?: Status;
}

export interface AddMachineRequest {
  kind: MachineKind;
  name?: string;
  /** A LAN machine's address, or a RunPod proxy address. */
  url?: string;
  /** A RunPod pod ID (or its proxy address). */
  pod?: string;
  port?: string;
}

/** Setup → Benchmarks: one task (and engine and size class) on one GPU. Measured rows are medians of finished runs
 * (PICKUPS_HOME/benchmarks.jsonl); a GPU with none yet gets the RTX 4090 reference scaled to it (measured: false). */
export interface Benchmark {
  /** "Recast (SCAIL-2)", "Remove, locked-off" */
  task: string;
  /** Median end-to-end time. */
  seconds: number;
  frames: number | null;
  size: string;
  /** "RTX 4090". Older servers leave the GPU fields out (the connected GPU). */
  gpu?: string;
  runs?: number;
  gpu_seconds?: number | null;
  s_per_frame?: number | null;
  fastest?: number | null;
  slowest?: number | null;
  measured?: boolean;
  /** The GPU connected now. */
  current?: boolean;
  /** "local", "remote", "runpod" */
  host?: string | null;
}

export interface Status {
  connected: boolean;
  comfy_url: string;
  machine: Machine | null;
  verdict: Verdict;
  /** Per task, plus "image" (the default image model's readiness) and "video" (LTX-2.5 for Create's video tabs). */
  tiles: Partial<Record<TaskId | "image" | "video", TileStatus>>;
  benchmarks: Benchmark[];
  /** ComfyUI's own queue. `outside`: it's running something Pickups didn't send (new renders wait behind it). */
  comfy_queue?: { running: number; pending: number; outside: boolean } | null;
  /** The video models work on at most this many frames from the start of a shot, at the shot's own frame rate
   * (97: 4 s at 24 fps, 1.6 s at 60). A take made from less than the whole shot says so in details.made_from. */
  max_frames?: number;
  /** Tasks whose limit on this GPU is shorter than max_frames (Change the weather on a 24 GB card). */
  max_frames_by_task?: Record<string, number>;
  /** The largest size work is done at (long side, px): 3840. Bigger clips are scaled down. */
  max_long?: number;
  /** The shortest clip a task takes, in frames. */
  min_frames?: number;
  /** Frames Previz to final renders in one pass on this GPU: 97, or 169 on a big card. */
  previz_frames?: number;
}

/** Trim & crop's shapes (width:height); "original" keeps the clip's own. */
export type CropAspect = "16:9" | "9:16" | "1:1" | "4:5" | "2.39:1" | "4:3";

/** POST /api/shots/{id}/prepare: a new shot from part of this one, cropped. Frames are 0-based; the out frame is kept.
 * The crop is in the source's pixels (as displayed). */
export interface PrepareRequest {
  in_frame: number;
  out_frame: number;
  crop: { aspect: CropAspect | null; x: number; y: number; w: number; h: number } | null;
  name?: string;
}

export interface Sample {
  id: string;
  name: string;
  thumb_url: string;
  /** Optional: what to try on it, and which task. */
  hint?: string;
  try?: TaskId;
}

/** A before/after example on the home screen (`GET /api/examples`). One has hero: true. */
export interface Example {
  id: string;
  task: TaskId;
  title: string;
  line: string;
  slate: string;
  sample_id: string;
  before_url: string;
  after_url: string;
  receipt_url?: string;
  poster_url: string;
  hero?: boolean;
  /** Examples that need their project (previz with its cast) open this real shot instead of a copy. */
  shot_id?: string;
  /** false: there's no shot to open on this machine (the card still plays). */
  can_try?: boolean;
  params?: Record<string, unknown>;
  /** Its workflow's default models (newer backends; the tile prefers the catalog from /api/models/defaults). */
  models?: ModelSlot[];
  /** The models this example's own take ran (its details.models), when they aren't the workflow's defaults: its slide's
   * model pill shows these. */
  models_used?: ModelSlot[];
}

/** One job a model does in a workflow (models_catalog.py). */
export interface ModelSlot {
  /** What it does there, in a word or two ("video", "frame paint", "tracking"). */
  role: string;
  /** As takes record it in details.models ("LTX-2.5 + Union Control IC-LoRA"). */
  name: string;
  /** The chip label ("LTX-2.5"). */
  short: string;
  /** What the slot can run instead today; the first is the default. */
  options: string[];
  /** Only runs sometimes ("on moving shots"). */
  when?: string;
}

/** GET /api/models/defaults: each home workflow's default models, keyed by workflow id ("character", "weather", ...). */
export type ModelCatalog = Record<string, { task: JobTask; models: ModelSlot[] }>;

/** The downloadable example gallery (a fresh install has none). */
export interface ExamplesPack {
  available: boolean;
  installed: boolean;
  size_mb?: number;
  /** How many examples are installed. */
  examples?: number;
  /** A pack found next to the app is being installed (first run). */
  importing?: boolean;
  /** Why the last import failed. */
  error?: string | null;
  /** Where a pack next to the app was found. */
  found?: string | null;
}

export interface ExampleOpenResult {
  shot: Shot;
  task: TaskId;
  /** The take's settings: the task form opens prefilled with them. */
  params?: Record<string, unknown>;
}

/** Shares of the frame. The backend sends percentages (0–100); fractions (0–1) are accepted too. */
export interface Receipt {
  original: number;
  borrowed: number;
  /** Finish's upscale: every pixel resampled from the footage (new detail, no new content). Never counted as AI. */
  upscaled?: number;
  ai: number;
  /** With `upscaled`: the upscaler ("SeedVR2 3B", "NVIDIA RTX Video Super Resolution") and its factor (2 = 2×). */
  upscaler?: string;
  scale?: number;
  /** A Finish made from a take (the backend composes it at read time, receipts.compose): the shares above are of the
   * shot's original footage, `highlights` is the Finish's own HDR share (% of pixels LTX-2.5 rebuilt) and `source` the
   * take it was made from, with that take's shares. */
  highlights?: number;
  source?: { take: string; label: string; receipt?: Pick<Receipt, "original" | "borrowed" | "upscaled" | "ai"> };
  /** An HDR-only Finish of a take that was upscaled: `upscaler` and `scale` are that take's. */
  source_upscaled?: boolean;
}

export type DownloadCodec = "prores" | "h264" | "h265" | "hlg" | "exr";
export type DownloadSize = "UHD" | "HD" | "native";

export interface Download {
  label: string;
  url: string;
  /** Newer backends: what the file is. Older ones send label/url only (the UI groups by label then). */
  codec?: DownloadCodec;
  size?: DownloadSize;
  bytes?: number;
  /** NLE-safe file name, e.g. harbour_pier_T02_remove_ProRes422HQ_UHD.mov. */
  filename?: string;
}

/** One ComfyUI workflow a take ran (API format), in the order it ran. */
export interface TakeWorkflow {
  /** e.g. "look", "previz", "remove". */
  step: string;
  /** What it did, from the graph: "Pick · SAM 3.1 · woman", "Look frame · Qwen-Image 2.1 · 2 references". Older servers: none. */
  title?: string;
  url: string;
  filename?: string;
}

/** Social cut-downs of a take: 9:16 stacked before/after, or a 1:1 wipe. */
export type ShareFormat = "9x16" | "1x1";
export interface ShareResult {
  url: string;
  filename?: string;
}

/** Timeline exports of a project's approved takes. */
export type TimelineFormat = "xml" | "edl";

export interface Note {
  id: string;
  frame: number;
  text: string;
  created_at?: number | string;
}

export interface ModelLicence {
  name: string;
  /** null: no model licence of its own (Canny is built into ComfyUI) or a name Pickups doesn't know. */
  licence: string | null;
  /** false: not for commercial use (shown as NON-COMMERCIAL). null: depends on its terms. */
  commercial: boolean | null;
  /** A model of the take this take was made from ("Take 02"): its pixels are in this take too. */
  via?: string;
}

export interface Version {
  id: string;
  label: string;
  /** null / "original" for the original. */
  task: TaskId | "original" | null;
  /** ISO string or epoch seconds. */
  created_at: string | number | null;
  video_url: string;
  thumb_url: string;
  receipt_url?: string;
  receipt?: Receipt;
  /** Optional aligned "before" clip for the wipe (falls back to the original). */
  before_url?: string;
  downloads: Download[];
  details: Record<string, unknown>;
  /** The licence of each model in details.models, in order (derived by the server, never stored). */
  licences?: ModelLicence[];
  /** "" or "approved". The original can be approved too. */
  status?: "" | "approved";
  hidden?: boolean;
  notes?: Note[];
  /** The take this one was made from ("orig" or a take id). */
  input_version?: string | null;
  /** What it ran with (for "Try again with changes"). */
  params?: Record<string, unknown>;
  look_url?: string;
  first_frame_url?: string;
  /** Recast: the still FLUX 3 klein made from a description (takes since Oct 17 keep it; older ones have none). */
  reference_url?: string;
  /** The exact ComfyUI workflows this take ran, in order. Older takes: [] or missing. */
  workflows?: TakeWorkflow[];
}

export interface Shot {
  id: string;
  name: string;
  width: number;
  height: number;
  fps: number;
  frames: number;
  duration_s: number;
  thumb_url: string;
  video_url: string;
  versions: Version[];
  project_id?: string;
  created_at?: number | string;
  /** Thumbnail of the newest take. */
  latest_thumb_url?: string;
  /** Number of approved takes. */
  approved?: number;
  /** Present when the shot was made in the Previz tab: the layout that rendered it. */
  previz_spec?: Record<string, unknown>;
  /** sha256 of the first exported frame. Same spec, same bytes, same hash. */
  previz_frame0?: string;
  /** Frame 0's framing from the previz page: the subject's box, the horizon, what the frame shows and the place. */
  previz_framing?: PrevizFraming;
  /** Frame 0's subject matte from the previz renderer (people and vehicles white), when the shot has one. */
  previz_matte_url?: string;
  /** Frame 0's depth pass from the previz renderer (start_frame.py method zbuf), when the shot has one. */
  previz_depth_url?: string;
  /** Present when Create made the shot (text or image to video): how it was made. */
  made?: ShotMade;
}

/** How Create made a shot (pickups/tasks/video.py). */
export interface ShotMade {
  task: "video";
  mode: "text" | "image";
  model: string;
  prompt: string;
  aspect: string;
  resolution: string;
  width: number;
  height: number;
  frames: number;
  fps: number;
  seed: number;
  /** Lock seed was on when it was made. */
  seed_locked?: boolean;
  image?: Ref;
  job_id?: string;
  /** Measured seconds from start to done. */
  took_s?: number;
}

/** POST /api/projects/{id}/videos/generate (pickups/video.py request). */
export interface VideoGenerateRequest {
  mode: "text" | "image";
  prompt: string;
  aspect: string;
  resolution: "720p" | "1080p";
  /** Seconds: 1 to 4 (25, 49, 73 or 97 frames at 24 fps). */
  duration: number;
  seed?: number;
  seed_locked?: boolean;
  /** Image to video: the picture to start from. */
  image?: Ref | null;
}

export interface Project {
  id: string;
  name: string;
  created_at?: number | string;
  updated_at?: number | string;
  archived?: boolean;
  /** "unsorted" (My shots) and "examples" are built in. */
  builtin?: boolean;
  shots: number;
  takes: number;
  approved: number;
  cover_url?: string | null;
  /** The clip behind the cover (the newest take's): it plays when the card is hovered. None when the cover is an image. */
  cover_video_url?: string | null;
  cast: number;
  /** Newer backends: how many images the project has. */
  images?: number;
}

export type CastKind = "person" | "place" | "prop" | "look";

export interface CastMember {
  id: string;
  name: string;
  kind: CastKind;
  /** One describing line: "a woman in her thirties with short black hair and a red wool coat". */
  line: string;
  file?: string;
  /** Full-size image (detail and zoom only). */
  url: string;
  /** A small JPEG (≤ 720 px) for grids, chips and the receipt. */
  thumb_url?: string;
  created_at?: number | string;
  /** GET /api/projects/{id}: how many shots of the project use it (@-mention or reference). */
  uses?: number;
  /** GET /api/cast: where it lives. */
  project_id?: string;
  project_name?: string;
  /** A character sheet built for them (front, back, close-up). `url` is then the sheet too; `thumb_url` its front view. */
  sheet?: CastSheet;
}

/** FRONT | BACK | CLOSE-UP in one 16:9 picture (pickups/sheets.py), with each view on its own. */
export interface CastSheet {
  url: string;
  /** About 960 px wide, for cards. */
  thumb_url?: string;
  views: { front?: string; back?: string; face?: string };
  /** The frame editor that made the views: "Qwen-Image 2.1" or "Qwen-Image-Edit 2511". */
  editor?: string;
  models?: string[];
  licences?: ModelLicence[];
  seed?: number;
  /** Lock seed was on: Redo sheet uses the same seed. */
  seed_locked?: boolean;
  /** Measured seconds its job took, start to done. */
  took_s?: number;
  /** What it was built from. */
  made_from?: "upload" | "image" | "cast" | "words";
  /** The front view is the editor's (the picture wasn't a clean full-length front view). */
  made_front?: boolean;
  created_at?: number;
}

/** `POST /api/projects/{id}/cast/sheet`: one source (a picture, a cast member to upgrade) or a description alone. */
export interface SheetRequest {
  upload_id?: string;
  image_id?: string;
  cast_id?: string;
  describe?: string;
  /** Needed for a new cast member. */
  name?: string;
  seed?: number;
  seed_locked?: boolean;
}

export interface ProjectDetail {
  project: Project;
  shots: Shot[];
  cast: CastMember[];
}

export interface MentionPreview {
  /** What the image model gets: mentions become "Maya (the person in <image2>)". */
  images: string;
  /** What text-only video models get: mentions become "Maya, a woman in her thirties with…". */
  text: string;
  refs: Array<{ id: string; name: string; url: string; image: number }>;
}

export interface Delivery {
  shot_id: string;
  shot_name: string;
  take: Version;
}

export interface Point {
  x: number;
  y: number;
  positive: boolean;
}

export interface SelectRequest {
  frame: number;
  points: Point[];
  text?: string;
}

export interface SelectResult {
  mask_url: string;
  bbox: number[] | Record<string, number> | null;
}

/** Check tracking: `POST /api/shots/{id}/track` body (pickups/tracking.py). The selection as Recast sends it. */
export interface TrackRequest {
  frame: number;
  points: Point[];
  text?: string;
  /** The take Recast works on (default: the original). */
  version_id?: string;
  /** LTX-2.5's window start, when the shot is longer than the render (the tracked part reaches it). */
  start_frame?: number;
}

/** The selection followed through the shot, as the render would follow it. The masks are kept for the render. */
export interface TrackResult {
  /** The shot with the selection tinted amber (H.264, at most 960 px wide). */
  video_url: string;
  /** Frames tracked, from the shot's first frame. */
  frames: number;
  fps: number;
  /** Per frame, the share of the picture the selection covers. */
  area: number[];
  /** Frames where the selection suddenly shrinks (a part, or the person, lost). */
  flagged: number[];
  /** Frames where nothing is selected. */
  empty: number[];
  cache_key: string;
  /** This selection was already checked: nothing ran. */
  cached?: boolean;
}

/** Recast's people finder: `POST /api/shots/{id}/people` body (all optional; pickups/people.py). */
export interface PeopleRequest {
  /** The take Recast works on (default: the original). */
  version_id?: string;
  /** Look again instead of using the cached answer. */
  refresh?: boolean;
}

/** One person the finder saw on its frame (source pixels, what Recast's `points` use). */
export interface PersonFound {
  index: number;
  /** [x, y, w, h]. */
  box: number[];
  /** Share of the frame they cover (0–1). */
  area: number;
  /** The deepest point inside them (usually the chest). */
  point: { x: number; y: number };
  /** Where to click so SAM takes the whole person: Recast's `params.points` for them. Older answers may omit it. */
  points?: Point[];
  /** A 3:4 cutout: them in colour, the rest dimmed. */
  thumb_url: string;
  /** Their matte on `frame`, the same overlay as /select's mask_url. */
  mask_url?: string;
  /** On how many of the probed frames they were found. */
  seen?: number;
  /** The one Recast replaces unless told otherwise (largest and longest on screen). */
  main: boolean;
}

export interface PeopleResult {
  /** The frame the people were found on (`params.frame`). */
  frame: number;
  /** Left to right. */
  people: PersonFound[];
  /** `index` of the main person, or null when nobody was found. */
  main: number | null;
  probed?: number[];
  /** The frames Recast works on (people are looked for only there). */
  window?: number;
  width?: number;
  height?: number;
  cached?: boolean;
  /** SAM failed part way: what it found, not cached. */
  partial?: boolean;
}

export interface UploadResult {
  id: string;
  url: string;
}

export type ReshootMove = "push_in" | "pull_out" | "orbit_left" | "orbit_right" | "crane_up";
export type Amount = "small" | "medium" | "large";

export interface RemoveParams {
  points: Point[];
  text?: string;
  shadow: boolean;
  /** Frame the points were clicked on (extra; the contract's select call carries it too). */
  frame?: number;
  /** Kept from the take by "Try again with changes" while its pick is unchanged (no field for them). */
  fill?: string;
  seed?: number;
  /** First frame of the part of a long shot to work on. */
  start_frame?: number;
  /** How many frames of the shot to work on (the bracket's length), up to status.max_frames. */
  frames?: number;
}
export interface RecastParams {
  points: Point[];
  /** A typed selection ("the man in the coat"). */
  text?: string;
  /** What was picked. Replace a person always says "person", so the server never asks whether a click is a car. */
  kind?: "person" | "vehicle";
  /** Who's in: an uploaded photo, a cast person, or any picked reference (e.g. a project image). */
  ref_upload_id?: string;
  cast_id?: string;
  who_ref?: Ref;
  describe?: string;
  /** Person swap (LTX): relight the new person to the shot's light first. Off unless true. */
  relight?: boolean;
  /** A take on this shot whose reference still (reference_url) is the person again ("Try again with changes"). */
  reference_take?: string;
  /** FLUX 3 klein's caption for the reference still (instead of one written from `describe`). */
  reference_prompt?: string;
  /** The video model's prompt (LTX-2.5 or SCAIL-2), kept from the take while the person is unchanged. */
  prompt?: string;
  scene?: string;
  seed?: number;
  frame?: number;
  /** Which renderer. Default "ltx". */
  engine?: RecastEngine;
  /** LTX: the first frame of the part of a long shot to work on (otherwise where the person is on screen most). */
  start_frame?: number;
  /** How many frames of the shot to work on (the bracket's length), up to status.max_frames. */
  frames?: number;
  /** Recast anyone: two or more people, each to their own character (authoritative when sent; LTX only). One person
   * sends no `swaps`: the fields above say who and with whom. */
  swaps?: RecastSwap[];
}
/** One person (or vehicle) a Recast swaps, and who they become (one of cast_id, ref_upload_id, who_ref, describe). */
export interface RecastSwap {
  /** The people finder's frame (PeopleResult.frame). */
  frame: number;
  /** Their points (all positive). */
  points: Point[];
  text?: string;
  kind?: "person" | "vehicle";
  /** "the person on the left", for the take's details. */
  label?: string;
  cast_id?: string;
  ref_upload_id?: string;
  who_ref?: Ref;
  describe?: string;
}
/** Recast's renderers: LTX-2.5 (the whole costume, lit by the shot) or SCAIL-2 (follows the exact performance). */
export type RecastEngine = "ltx" | "scail";
/** Render previz. All optional; the older single-reference fields are still accepted by the backend. */
export interface PrevizParams {
  /** References in order: they are images 2, 3, … (image 1 is the playblast frame). Up to 4. */
  look_upload_ids?: string[];
  /** Qwen-Image 2.1 instruction for the look frame. */
  look_prompt?: string;
  /** LTX-2.5 video prompt: subject, action, camera, light, film look. */
  prompt?: string;
  /** Also pin the start still as the shot's real first frame (off: it is a reference at position -1 only). */
  pin_start?: boolean;
  /** Mid and last keyframes, an agent's ask only: the app never sends it and keys are off for every shot (tasks/previz.py
   * keyframes_on); a keyed render that cuts is made again start-only. */
  keyframes?: boolean;
  /** How the start still was made ("zbuf", "B3"), for the take's model list. */
  start_method?: string;
  /** The start still's seed, and whether Lock seed was on: a reopened shot shows them again. */
  start_seed?: number;
  seed_locked?: boolean;
  look_frame_upload_id?: string;
  /** References picked from the cast or the project's images (after the uploaded ones). */
  look_refs?: Ref[];
  /** A project image used as the look frame. */
  look_frame_ref?: Ref;
  /** The playblast frame the look frame is painted over and pinned to (default 0). */
  look_at?: number;
  grain?: Grain;
  /** How closely the render follows the playblast: "exact" (Layout-To-Render, the default) or "loose" (camera and
   * blocking kept, stand-ins such as dust spheres or grey blocks become what the prompt describes). */
  follow?: PrevizFollow;
  /** First frame of the part of a long shot to render. */
  start_frame?: number;
  /** Older fields. */
  look_upload_id?: string;
  look_upload_id_2?: string;
  describe?: string;
}

export type PrevizFollow = "exact" | "loose";

/** Film grain on generated video (which measures zero grain, so skin reads as plastic). Default "fine". */
export type Grain = "off" | "fine" | "35mm";

export type WeatherChange = "snow" | "rain" | "fog" | "night" | "golden_hour" | "autumn";

/** Change weather: a preset `change`, or `custom` text. */
export interface EnvironmentParams {
  change?: WeatherChange;
  custom?: string;
  /** What's in the shot, in one sentence. */
  describe?: string;
  /** Advanced: the first-frame instruction (Qwen), the video prompt (LTX), structure control and seed. */
  look_prompt?: string;
  prompt?: string;
  control?: "" | "depth" | "canny";
  seed?: number;
  grain?: Grain;
  /** A project image used as the restyled first frame. */
  first_frame_ref?: Ref;
  /** Or an uploaded one. */
  first_frame_upload_id?: string;
  /** Keep exactly as shot: what SAM 3.1 finds and follows ("the car"), put back over the render in its new light. */
  keep_text?: string;
  /** Or clicks on it (in the shot's pixels), on this frame of the shot. */
  keep_points?: Point[];
  keep_frame?: number;
  /** First frame of the part of a long shot to change. */
  start_frame?: number;
  /** How many frames of the shot to work on (the bracket's length), up to status.max_frames. */
  frames?: number;
}
export interface ReshootParams {
  move: ReshootMove;
  amount: Amount;
}
export interface FinishParams {
  upscale_4k: boolean;
  hdr: boolean;
  quality: "fast" | "best";
}

/** Relight any scene (tasks/scene_relight.py): a tested look, or "custom" with the light in words and where it comes from. */
export interface RelightParams {
  look: "sunset" | "golden" | "cool" | "midday" | "afternoon" | "custom" | string;
  custom?: string;
  direction?: "behind_low" | "left_low" | "back_right_low" | "front_right_high" | "front_mid";
}

export type JobParams = RemoveParams | RecastParams | EnvironmentParams | PrevizParams | RelightParams | ReshootParams | FinishParams;

export interface JobRequest {
  shot_id: string;
  version_id?: string;
  task: TaskId;
  params: JobParams;
  /** A step flow's draft and step: on the job record, never in params (params go into every take's receipt). */
  draft_key?: string;
  step?: string;
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "canceled";

export interface Job {
  id: string;
  /** Set by Move next in the queue (epoch seconds): it runs before the other waiting runs. */
  next_at?: number;
  /** null for image jobs (they belong to a project). */
  shot_id: string;
  task: JobTask;
  project_id?: string | null;
  /** Image jobs: the images they made. */
  result_image_ids?: string[];
  /** Sheet jobs: the cast member that has the sheet now. */
  result_cast_id?: string | null;
  /** Video jobs: the new shot the video became. */
  result_shot_id?: string | null;
  /** Sheet jobs: each view's picture as soon as it is made (front, back, face), before the sheet is done. */
  partial?: { front?: string; back?: string; face?: string };
  /** For labels when the shot or project isn't loaded. */
  shot_name?: string | null;
  project_name?: string | null;
  /** Image jobs: the model id. */
  model?: string | null;
  /** Set when ComfyUI has reported no progress for 10 minutes; clears itself when progress resumes. */
  stalled?: boolean;
  stall_message?: string | null;
  status: JobStatus;
  stage: string;
  stages: string[];
  progress: number;
  eta_s: number | null;
  /** What went wrong, in plain words (app/comfy.py plain_error). */
  error?: string | null;
  /** The raw engine text behind error, for Details. */
  error_detail?: string | null;
  /** Contract: the resulting version. The backend may also send the input version here. */
  version_id?: string | null;
  /** Backend extension: the version this job produced. Preferred over version_id when present. */
  result_version_id?: string | null;
  params?: JobParams;
  created_at?: number | string;
  started_at?: number | string;
  finished_at?: number | string;
  /** UI-only: the last real stage seen, so a failure can say where it happened. */
  _lastStage?: string;
  /** "inline": work a route runs itself (the start frame, the look frame), shown in the queue but never on the GPU
   * queue. It can't be stopped and makes no take. */
  lane?: "inline";
  /** The flow step this job is for ("start_frame", "look"), and the page's draft that started it. */
  step?: string;
  draft_key?: string;
  /** An inline job's picture, once done. */
  result?: { upload_id: string; url: string; seed?: number; took_s?: number } | null;
  /** What the run is timed as (pickups/app/timings.py kind_of): the task, its model and size. */
  kind?: string;
  /** The median of the last measured runs of this kind on this machine, whole seconds; null before the first. */
  usual_s?: number | null;
  /** Done: measured seconds from start to done. */
  took_s?: number;
  /** Failed because the model refused the prompt (Ideogram 4's grey card): nothing was saved, and the seed isn't kept. */
  blocked?: boolean;
}

export interface PrevizLookRequest {
  shot_id: string;
  version_id?: string;
  look_upload_ids?: string[];
  look_prompt?: string;
  /** Older fields, sent too so an older backend still works. */
  look_upload_id?: string;
  look_upload_id_2?: string;
  describe?: string;
  /** References picked from the project (cast entries, images), read after the uploads. */
  look_refs?: { kind: "cast" | "image" | "upload"; id: string }[];
  /** A fresh seed per preview, so "Try another" gives another picture. */
  seed?: number;
  /** The page's draft, for the inline queue row. */
  draft_key?: string;
}

export interface PrevizLookResult {
  url: string;
  upload_id: string;
  seed?: number;
  references?: number;
}

/** Make start frame from previz: the shot, its references, and the Look words only when the user wrote their own. */
export interface PrevizStartFrameRequest {
  shot_id: string;
  version_id?: string;
  look_upload_ids?: string[];
  look_refs?: { kind: "cast" | "image" | "upload"; id: string }[];
  user_look?: string;
  seed?: number;
  /** The arm and the words, without running anything. */
  plan_only?: boolean;
  /** The page's draft: it rides on the inline queue row, so a reloaded page finds the start frame again. */
  draft_key?: string;
}

/** What made it: the method ("B3"), the template's words as sent, whether the subjects were masked out of the canny, and
 * what the template still lacks. */
export interface PrevizStartFrameResult extends Partial<PrevizLookResult> {
  /** Measured seconds it took. */
  took_s?: number;
  /** zbuf: Z-Image Turbo from the previz's own depth (the default); B3 and commercial-safe: the fallbacks. */
  method: "zbuf" | "B3" | "commercial-safe";
  subject_mask: boolean;
  /** Where the user's words disagreed with the previz (the previz won), e.g. the facing. */
  flags?: string[];
  /** zbuf: the strength the kept still used, its size check, and a flag when the retry failed too. */
  strength?: number;
  size_check?: SizeCheck;
  flag?: "size" | "ghost" | "count" | null;
  /** How many people the app's detector found in the kept still (null: the detector couldn't run). */
  people?: number | null;
  /** Said when the people count still differs from the previz after the retry. */
  warning?: string;
  /** "Remake this shot": its frame 0 lacks what only the Previz page can render. */
  action?: string | null;
  positive: string;
  negative: string;
  references: number;
  todo: string[];
  steps: number;
  cfg: number;
  template: string;
}

/** Frame 0 of a playblast: the main subject's box [x0, y0, x1, y1] and the horizon height, as fractions of the frame. */
export interface PrevizFraming {
  subject?: [number, number, number, number];
  horizon?: number;
  view?: { share: Record<string, number>; x?: Record<string, number>; outside: string[]; actors: number[]; blocked?: number; bare?: number };
  place?: string;
  reveals?: { word: string; at_s: number; side: "left" | "right" | "middle" }[];
  /** The camera's real move at the first and last frame (runtime.ts moveOf). */
  move?: { start: { pos: [number, number, number]; look: [number, number, number] }; end: { pos: [number, number, number]; look: [number, number, number] } };
  camera?: { height_m: number; pitch_deg: number; focal_mm: number };
  subjects?: { kind: "person" | "vehicle" | "animal"; x: number; distance_m: number; in_view: boolean; doing?: string; word?: string; speed_ms?: number; facing?: Facing; w?: number; h?: number }[];
  world?: { time?: string | null; weather?: string | null };
  light?: { dir: "camera left" | "camera right" | "behind camera" | "backlit" };
}

/** Which way a previz figure or vehicle faces at frame 0, from its rotation: toward the camera, away, or a side on screen. */
export type Facing = "camera" | "away" | "left" | "right";

/** The start still's size against the previz (start_frame.size_state): within 15% passes, more warns; over 1.5x fails and
 * under 1/1.5 is small (both retry once). */
export interface SizeCheck {
  ratio: number | null;
  state: "pass" | "warn" | "fail" | "small" | "unmeasured";
  warning: string | null;
}

export interface PrevizFramingResult {
  warning: string | null;
  checked: boolean;
  size_check?: SizeCheck;
}

/** A take, a shot, a project (its folder and shots), a cast entry or a project image in the trash (library/trash/<id>/,
 * under their library paths). Restore puts it back; Empty Trash removes it for good. */
export interface TrashEntry {
  id: string;
  kind: "take" | "shot" | "project" | "cast" | "image";
  /** Takes and shots. */
  shot_id?: string;
  take_id?: string | null;
  /** Cast entries and images. */
  asset_id?: string;
  /** A project: how many shots went with it. */
  shots?: number;
  name: string;
  project_id: string;
  trashed_at: number;
  bytes: number;
  /** Files the ComfyUI machine still holds for it (from the saved workflows). Pickups can't remove those. */
  remote_files: number;
  thumb_url?: string;
}

export interface EmptyTrashResult {
  items: number;
  bytes: number;
  remote_files: number;
}

/** Setup > Storage: where the library is and its total size. */
export interface StorageInfo {
  path: string;
  bytes: number;
}

export interface ApiEvents {
  onJob: (job: Job) => void;
  onStatus: (status: Status) => void;
  onShot?: (shot: Shot) => void;
  /** A shot went to the trash: drop it from every open view. */
  onShotRemoved?: (event: { id: string; project_id?: string }) => void;
  /** A project changed (e.g. new images). */
  onProject?: (event: { id: string; images?: string[]; cast?: string[] }) => void;
  onOpen?: () => void;
  onError?: () => void;
}

// --- images (docs/images-spec.md) ---------------------------------------------------------------------------------

export type ImageResolution = "1K" | "2K";

export interface ImageModel {
  id: string;
  name: string;
  maker: string;
  licence?: string;
  /** false: the licence doesn't allow commercial use (shown as NON-COMMERCIAL). */
  commercial?: boolean;
  /** "generate", "edit", "style". */
  modes: string[];
  max_refs: number;
  max_side: number;
  /** "~5 s at 1K on a 4090". */
  speed: string;
  sponsor?: boolean;
  notes?: string;
  /** false: files not installed; null: unknown (ComfyUI offline). */
  ready: boolean | null;
  sizes: Partial<Record<ImageResolution, Record<string, [number, number]>>>;
}

export interface ImageModels {
  default: string;
  aspects: string[];
  resolutions: ImageResolution[];
  models: ImageModel[];
}

export interface ProjectImage {
  id: string;
  url: string;
  thumb_url: string;
  width: number;
  height: number;
  model: string;
  model_name?: string;
  mode?: string;
  prompt: string;
  prompt_sent?: string;
  seed?: number;
  /** Lock seed was on when it was made. */
  seed_locked?: boolean;
  aspect?: string;
  resolution?: ImageResolution;
  refs?: Ref[];
  created_at?: number | string;
  job_id?: string;
  /** Measured seconds its job took, start to done. */
  took_s?: number;
  /** How many of the project's shots use it (a reference in a take's settings). */
  uses?: number;
}

export interface ImageGenerateRequest {
  model: string;
  prompt: string;
  aspect: string;
  resolution: ImageResolution;
  count: number;
  seed?: number;
  seed_locked?: boolean;
  refs: Ref[];
}

// --- prompt director ------------------------------------------------------------------------------------------------

export type EnhanceTarget = "qwen-image" | "qwen-edit" | "flux3" | "ltx";

export interface EnhanceRequest {
  target: EnhanceTarget;
  text: string;
  project_id?: string;
  shot_id?: string;
  version_id?: string;
  frame?: number;
  refs?: Ref[];
  seed?: number;
}

export interface EnhanceResult {
  text: string;
  model: string;
  runs_on: string;
  sent: string;
  target?: EnhanceTarget;
}

/** "Describe it": what the description is for. scene = "What's in the shot"; motion = Render previz's action and camera
 * (a playblast described as the live-action shot it stands for); look = a still, for the image models. */
export type DescribeField = "scene" | "motion" | "look";

/** `POST /api/shots/{id}/describe` body (all optional; pickups/describe.py). */
export interface DescribeRequest {
  /** The take being worked from (default: the original). */
  version_id?: string;
  /** The model the field's prompt is for (default "ltx"). */
  target?: EnhanceTarget;
  field?: DescribeField;
  /** Image targets: describe this one frame. */
  frame?: number;
  /** Look again instead of using the cached answer ("Try again"). */
  refresh?: boolean;
  seed?: number;
}

export interface DescribeResult {
  text: string;
  target: EnhanceTarget;
  field: DescribeField;
  /** "orig" or the take's id. */
  version_id: string;
  /** The frames the model looked at (first, middle, last of the part the video model renders). */
  frames: number[];
  /** The camera move measured between the first and the last frame. */
  camera?: "static" | "push" | "pull" | "moving" | null;
  model: string;
  /** "ComfyUI", or the local server's vision model. */
  runs_on: string;
  seconds?: number;
  cached?: boolean;
}

export interface DirectorSettings {
  /** "openai" = a local OpenAI-compatible server (LM Studio, Ollama); "comfy" = the GPU machine's enhancers. */
  backend: "comfy" | "openai" | "off";
  url: string;
  model: string;
  vision_model: string;
  /** The user chose "Off" and confirmed it. */
  off_acknowledged?: boolean;
}

export interface DetectedServer {
  url: string;
  /** "lmstudio", "ollama", … */
  kind: string;
  models: string[];
  suggested_model?: string | null;
  suggested_vision_model?: string | null;
}

export interface PromptTestResult {
  seconds: number;
  [k: string]: unknown;
}

export interface Settings {
  comfy_url?: string;
  allow_nvidia_rtx?: boolean;
  open_weights_only?: boolean;
  commercial_safe?: boolean;
  director?: DirectorSettings;
}
