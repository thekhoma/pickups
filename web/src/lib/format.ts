import { TASKS, type Job, type JobTask, type Machine, type ModelLicence, type Receipt, type Shot, type Status, type TaskId, type Version } from "../types";

/** HH:MM:SS:FF (non-drop, timebase = rounded fps). */
export function timecode(frame: number, fps: number): string {
  const base = Math.max(1, Math.round(fps || 24));
  const f = Math.max(0, Math.floor(frame));
  const ff = f % base;
  const totalS = Math.floor(f / base);
  const ss = totalS % 60;
  const mm = Math.floor(totalS / 60) % 60;
  const hh = Math.floor(totalS / 3600);
  return [hh, mm, ss, ff].map((n) => String(n).padStart(2, "0")).join(":");
}

/** How many frames a take really has: tasks can trim (previz and weather render at most 97), the original is the shot. */
export function takeFrames(shot: Pick<Shot, "frames">, v: Version | null | undefined): number {
  const n = Number(v?.details?.frames);
  return !isOriginal(v) && Number.isFinite(n) && n > 0 ? Math.round(n) : shot.frames;
}

/** How many frames Recast's LTX-2.5 renders of a shot of `total` frames (tasks/recast_ltx.py: 8k+1, at most
 * `maxFrames`). */
export function ltxWindowFrames(maxFrames: number, total: number): number {
  const n = Math.max(1, Math.min(maxFrames, total));
  return 8 * Math.floor((n - 1) / 8) + 1;
}

/** The shortest bracket: about a second, on LTX's 8-frame grid (8k+1). */
export function minWindowFrames(fps: number, total: number): number {
  return ltxWindowFrames(Math.max(9, Math.round(fps > 0 ? fps : 24) + 1), total);
}

/** "4 s", "5.5 s": the length to the nearest half second. */
export function halfSeconds(frames: number, fps: number): string {
  const half = Math.max(0.5, Math.round((frames / (fps > 0 ? fps : 24)) * 2) / 2);
  return `${Number.isInteger(half) ? half : half.toFixed(1)} s`;
}

/** Where a window of `frames` can start on a shot of `total` frames: on LTX's 8-frame grid, inside the shot (the very
 * end counts too, as recast_ltx.choose_window clamps). */
export function windowStart(raw: number, frames: number, total: number): number {
  const last = Math.max(0, total - frames);
  return Math.max(0, Math.min(last, Math.round(raw / 8) * 8));
}

/** A take's length: "4.0 s · 97 frames". */
export function takeLength(frames: number, fps: number): string {
  const s = frames / (fps > 0 ? fps : 24);
  return `${s < 10 ? `${s.toFixed(1)} s` : duration(s)} · ${frames} frames`;
}

/** A take's label as shown. Older takes keep the previous names on disk. */
/** A take's label in the task's one name (the tile's, the panel's, the queue's), whatever an older take saved: "Remove"
 * → "Remove a person", "Weather · Snow" → "Change the weather · Snow". The take's own detail (Snow, 4K HDR) stays. */
export function takeLabel(label: string): string {
  return label
    .replace(/^(?:Environment|Weather|Change weather|Change the weather)(?= · |$)/, "Change the weather")
    .replace(/^Recast(?= · |$)/, "Replace a person")
    .replace(/^Remove(?= · |$)/, "Remove a person")
    .replace(/^Relight any scene(?= · |$)/, "Relight")
    .replace(/^Render previz$/, "Previz to final")
    .replace(/^Finish to (?=\S)/, "Finish ");
}

/** "~4 min" for tile badges. */
export function etaBadge(eta_s: number | null | undefined): string | null {
  if (eta_s == null || !isFinite(eta_s) || eta_s <= 0) return null;
  if (eta_s < 60) return `~${Math.max(5, Math.round(eta_s / 5) * 5)} s`;
  return `~${Math.max(1, Math.round(eta_s / 60))} min`;
}

/** "About 4 min" for form footers. */
export function aboutTime(eta_s: number | null | undefined): string | null {
  if (eta_s == null || !isFinite(eta_s) || eta_s <= 0) return null;
  if (eta_s < 60) return "Under a minute";
  return `About ${Math.max(1, Math.round(eta_s / 60))} min`;
}

/** "about 2 min left" for the running card. */
export function etaLeft(eta_s: number | null | undefined): string {
  if (eta_s == null || !isFinite(eta_s)) return "Working out how long…";
  if (eta_s <= 3) return "almost done";
  if (eta_s < 60) return `about ${Math.max(5, Math.ceil(eta_s / 5) * 5)} s left`;
  if (eta_s < 90) return "about 1 min left";
  return `about ${Math.round(eta_s / 60)} min left`;
}

/** "2:10 left" for the queue bar. */
export function etaClock(eta_s: number | null | undefined): string {
  if (eta_s == null || !isFinite(eta_s)) return "starting";
  const s = Math.max(0, Math.round(eta_s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} left`;
}

/** "3 min 12 s" */
export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m >= 60) return `${Math.floor(m / 60)} h ${m % 60} min`;
  return r ? `${m} min ${r} s` : `${m} min`;
}

export function shotLength(shot: Pick<Shot, "duration_s">): string {
  const d = shot.duration_s || 0;
  return d < 10 ? `${d.toFixed(1)} s` : duration(d);
}

/** A receipt as whole percentages that add up to 100 (`upscaled` is 0 for every take but a Finish upscale). */
export interface ReceiptShares {
  original: number;
  borrowed: number;
  upscaled: number;
  ai: number;
}

/** Normalise a receipt to whole percentages that add up to 100 (receipts.percents). */
export function receiptPercents(r: Receipt | undefined | null): ReceiptShares | null {
  if (!r) return null;
  const vals = [Number(r.original) || 0, Number(r.borrowed) || 0, Number(r.upscaled) || 0, Number(r.ai) || 0];
  const sum = vals.reduce((a, b) => a + b, 0);
  if (sum <= 0) return null;
  const scaled = vals.map((v) => (v / sum) * 100);
  const floors = scaled.map(Math.floor);
  let left = 100 - floors.reduce((a, b) => a + b, 0);
  const order = scaled.map((v, i) => [v - Math.floor(v), i] as const).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) {
    if (left <= 0) break;
    floors[i] += 1;
    left -= 1;
  }
  return { original: floors[0], borrowed: floors[1], upscaled: floors[2], ai: floors[3] };
}

export type ShareKey = keyof ReceiptShares;
const SHARE_LABEL: Record<ShareKey, string> = { original: "Original", borrowed: "Other frames", upscaled: "Upscaled", ai: "AI" };
const SHARE_KEYS = Object.keys(SHARE_LABEL) as ShareKey[];

/** A whole percent as shown (receipts.percent_texts): a share that is there but rounds to nothing reads "<1%", and one
 * rounded up to 100% while another share is there reads ">99%". */
export function percentText(r: Receipt | undefined | null, key: ShareKey): string {
  const p = receiptPercents(r);
  if (!p || !r) return "";
  const raw = (k: ShareKey) => Number(r[k]) || 0;
  if (p[key] === 0 && raw(key) > 0) return "<1%";
  if (p[key] === 100 && SHARE_KEYS.some((k) => k !== key && raw(k) > 0)) return ">99%";
  return `${p[key]}%`;
}

/** The shares a receipt names, in the bar's order (receipts.shares): Original, then any share that is actually there.
 * A 0% row ("0% other frames") is left out. Original stays even at 0%, so a fully generated take still says so.
 * An upscale is the footage resampled, its own neutral share: never AI. `text` is the share as shown ("13%", "<1%", ">99%"). */
export function receiptShares(r: Receipt | undefined | null): Array<{ key: ShareKey; label: string; pct: number; text: string }> {
  const p = receiptPercents(r);
  if (!p) return [];
  return SHARE_KEYS.filter((k) => k === "original" || p[k] > 0).map((k) => ({
    key: k,
    label: SHARE_LABEL[k],
    pct: p[k],
    text: percentText(r, k),
  }));
}

/** A fully generated take: nothing of the original, of other frames or upscaled from the footage is left (receipts.all_ai). */
export function allAi(r: Receipt | undefined | null): boolean {
  return !!r && !!receiptPercents(r) && !(Number(r.original) > 0) && !(Number(r.borrowed) > 0) && !(Number(r.upscaled) > 0);
}

/** The take was upscaled (Finish 4K): an upscaled share, or (a whole frame generated upstream) the upscaler named. */
function wasUpscaled(r: Receipt | undefined | null): r is Receipt {
  return !!r && (Number(r.upscaled) > 0 || !!r.upscaler || Number(r.scale) > 0);
}

/** " 2×" for a receipt that knows its factor (receipts.scale_text). */
function scaleText(r: Receipt): string {
  const scale = Number(r.scale);
  return scale > 0 ? ` ${+scale.toFixed(1)}×` : "";
}

/** "87% upscaled · 13% AI", "97% original · 3% other frames · 0% AI", "100% AI · whole frame generated (upscaled 2×)"
 * (receipts.summary). */
export function receiptSummary(r: Receipt | undefined | null): string | null {
  if (allAi(r)) return `100% AI · whole frame generated${wasUpscaled(r) ? ` (upscaled${scaleText(r)})` : ""}`;
  const parts = receiptShares(r);
  return parts.length ? parts.map((s) => `${s.text} ${s.key === "ai" ? "AI" : s.label.toLowerCase()}`).join(" · ") : null;
}

/** "Upscaled 2× from your footage by SeedVR2 3B: new detail, no new content." ("from Take 02" for a Finish made from a
 * take; receipts.upscale_sentence). */
export function upscaleSentence(r: Receipt | undefined | null): string | null {
  if (!wasUpscaled(r)) return null;
  const who = r.upscaler ?? "";
  const n = r.source?.take;
  // SeedVR2 restores detail as it upscales; NVIDIA's RTX Video Super Resolution only sharpens.
  const what = !who || who.startsWith("SeedVR2") ? "new detail, no new content" : "no new content";
  const by = who ? ` by ${who}` : "";
  if (r.source_upscaled && n) return `Take ${n} was upscaled${scaleText(r)} from your footage${by}: ${what}.`;
  return `Upscaled${scaleText(r)} from ${n ? `Take ${n}` : "your footage"}${by}: ${what}.`;
}

/** What the take a Finish was made from already was: "Take 02 (Remove) is 95% original · 4% other frames · 1% AI."
 * (receipts.source_sentence; null when made from the original or when the whole frame is generated). */
export function sourceSentence(r: Receipt | undefined | null): string | null {
  const s = r?.source;
  if (!s?.take || !s.receipt || allAi(r)) return null;
  const what = receiptSummary(s.receipt as Receipt);
  return what ? `Take ${s.take}${s.label ? ` (${takeLabel(s.label)})` : ""} is ${what}.` : null;
}

/** A Finish take's HDR share as shown: "13%" or "under 1%" (receipts.highlights_text). Its own highlights when it was
 * made from a take, else its AI share as the bar rounds it. */
export function highlightsText(r: Receipt | undefined | null): string | null {
  if (!r) return null;
  const own = r.highlights != null;
  const hl = Number(own ? r.highlights : r.ai) || 0;
  if (hl <= 0) return null;
  const whole = own ? Math.floor(hl + 0.5) : (receiptPercents(r)?.ai ?? 0);
  return whole ? `${whole}%` : "under 1%";
}

/** A Finish take's own AI share: "HDR highlights rebuilt by LTX-2.5 on 13% of pixels." (receipts.highlights_sentence). */
export function highlightsSentence(r: Receipt | undefined | null): string | null {
  const t = highlightsText(r);
  return t ? `HDR highlights rebuilt by LTX-2.5 on ${t} of pixels.` : null;
}

/** What a Finish take did to the picture, in words (the receipt sheet's sentence): "Upscaled 2× from your footage by
 * SeedVR2 3B: new detail, no new content. HDR highlights rebuilt by LTX-2.5 on 13% of pixels." Made from a take, it
 * names the take and what it already was. Null for other tasks, whose bar says it all. */
export function finishSentence(v: Pick<Version, "task" | "receipt"> | null | undefined): string | null {
  if (v?.task !== "finish") return null;
  return [upscaleSentence(v.receipt), sourceSentence(v.receipt), highlightsSentence(v.receipt)].filter(Boolean).join(" ") || null;
}

export function shortGpu(gpu: string | undefined | null): string {
  if (!gpu) return "GPU";
  return gpu.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+Laptop GPU$/i, " Laptop").trim();
}

export function machineLabel(m: Machine | null | undefined): string {
  if (!m) return "";
  return [m.name, shortGpu(m.gpu)].filter(Boolean).join(" · ");
}

export const TASK_VERB: Record<JobTask, string> = {
  image: "Image",
  sheet: "Character sheet",
  video: "New video",
  remove: "Remove a person",
  recast: "Replace a person",
  environment: "Change the weather",
  previz: "Previz to final",
  relight: "Relight",
  reshoot: "Reshoot",
  finish: "Finish",
};

/** The verb as chips, slates and the queue show it (the same names as the task list). */
export const TASK_SHORT: Record<JobTask, string> = {
  image: "Image",
  sheet: "Sheet",
  video: "Video",
  remove: "Remove a person",
  recast: "Replace a person",
  environment: "Change the weather",
  previz: "Previz to final",
  relight: "Relight",
  reshoot: "Reshoot",
  finish: "Finish",
};

export function isTask(t: unknown): t is TaskId {
  return t === "remove" || t === "recast" || t === "environment" || t === "previz" || t === "relight" || t === "reshoot" || t === "finish";
}

export function isOriginal(v: Version | null | undefined): boolean {
  return !v || v.id === "orig" || !isTask(v.task);
}

/** "Remove v1", "Finish 4K HDR v2": numbers versions of the same task in order. */
export function versionLabels(shot: Shot): Record<string, string> {
  const out: Record<string, string> = {};
  const count: Record<string, number> = {};
  for (const v of shot.versions) {
    if (isOriginal(v)) {
      out[v.id] = v.label || "Original";
      continue;
    }
    const key = String(v.task);
    count[key] = (count[key] ?? 0) + 1;
    const base = takeLabel(v.label || TASK_VERB[v.task as TaskId] || "Version");
    out[v.id] = /\bv\d+$/i.test(base) ? base : `${base} v${count[key]}`;
  }
  return out;
}

export function toEpochMs(t: string | number | null | undefined): number | null {
  if (t == null) return null;
  if (typeof t === "number") return t < 1e12 ? t * 1000 : t;
  const n = Date.parse(t);
  return isNaN(n) ? null : n;
}

export function relativeTime(t: string | number | null | undefined): string {
  const ms = toEpochMs(t);
  if (ms == null) return "";
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86400);
  if (d < 7) return d === 1 ? "yesterday" : `${d} days ago`;
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** How long a finished version took, from its details or the job that made it. */
export function versionTook(v: Version, job?: Job | null): number | null {
  const d = v.details ?? {};
  for (const k of ["total_seconds", "duration_s", "elapsed_s", "gpu_seconds", "gpu_time_s"]) {
    const n = Number(d[k]);
    if (isFinite(n) && n > 0) return n;
  }
  if (job) {
    const a = toEpochMs(job.started_at ?? job.created_at);
    const b = toEpochMs(job.finished_at);
    if (a && b && b > a) return (b - a) / 1000;
  }
  return null;
}

export function jobResultVersionId(job: Job): string | null {
  return job.result_version_id ?? (job.status === "done" ? (job.version_id ?? null) : null);
}

export function isActive(job: Job): boolean {
  return job.status === "queued" || job.status === "running";
}

export type PillState = "checking" | "ready" | "setup" | "offline";

export function pillState(status: Status | null, failed: boolean): PillState {
  if (!status) return failed ? "offline" : "checking";
  if (!status.connected || status.verdict === "offline") return "offline";
  // Beta tiles (Reshoot) don't turn the whole app amber.
  const core = (Object.entries(status.tiles ?? {}) as Array<[string, Status["tiles"][TaskId]]>).filter(([k]) => k !== "reshoot");
  if (status.verdict === "remote" || core.some(([, t]) => t && !t.ready)) return "setup";
  return "ready";
}

/** Label for a tile that can't run: something to set up, or simply not available on this build/machine yet. */
export function notReadyLabel(status: Status | null, task: TaskId): string {
  if (!status?.connected) return "Needs setup";
  return status.tiles?.[task]?.missing?.length ? "Needs setup" : "Not available yet";
}

export function missingFor(status: Status | null, task: TaskId | "image") {
  const t = status?.tiles?.[task];
  const files = t?.missing ?? [];
  const gb = files.reduce((a, m) => a + (Number(m.gb) || 0), 0);
  return { files, gb, count: files.length };
}

/** A file size from bytes: "3.2 GB", "420 MB", "12 KB" (decimal units, as Finder shows them). */
export function size(bytes: number): string {
  if (!isFinite(bytes) || bytes <= 0) return "0 KB";
  if (bytes >= 1e9) return gb(bytes / 1e9);
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

export function gb(n: number): string {
  if (!isFinite(n)) return "";
  if (n < 1) return `${Math.max(1, Math.round(n * 1000))} MB`; // never "0.0 GB"
  return n >= 10 ? `${Math.round(n)} GB` : `${n.toFixed(1)} GB`;
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent);
export const ALT_KEY = isMac ? "Option" : "Alt";
export const MOD_KEY = isMac ? "⌘" : "Ctrl";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

// --- v2: timecode discipline, takes, status line -----------------------------------------------------------------

/** How long a finished take took, spoken: "5 min 57 s", "42 s", "1 h 2 min 3 s". */
export function tookText(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor(s / 60) % 60;
  const sec = s % 60;
  const parts: string[] = [];
  if (h) parts.push(`${h} h`);
  if (m) parts.push(`${m} min`);
  if (sec || !parts.length) parts.push(`${sec} s`);
  return parts.join(" ");
}

/** Countdowns and durations as 00:01:42 (HH:MM:SS), never "1m 42s". */
export function clock(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "--:--:--";
  const s = Math.max(0, Math.round(seconds));
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((n) => String(n).padStart(2, "0")).join(":");
}

/** A countdown as remaining time: "~5 min left", "~40 s left", "~1 h 10 min left". */
export function leftText(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "Starting";
  const s = Math.max(0, Math.round(seconds));
  if (s < 5) return "Almost done";
  if (s < 60) return `~${Math.max(5, Math.round(s / 5) * 5)} s left`;
  const m = Math.round(s / 60);
  if (m < 60) return `~${m} min left`;
  return `~${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ""} left`;
}

/** Tile times as a mono "~00:04:00" (estimates round to the minute, or to 5 s under a minute). */
export function etaMono(eta_s: number | null | undefined): string | null {
  if (eta_s == null || !isFinite(eta_s) || eta_s <= 0) return null;
  const rounded = eta_s < 60 ? Math.max(5, Math.round(eta_s / 5) * 5) : Math.max(1, Math.round(eta_s / 60)) * 60;
  return `~${clock(rounded)}`;
}

export interface Take {
  /** "02" */
  num: string;
  /** "Remove", "Finish 4K HDR", "Original" */
  label: string;
}

/** Versions are takes: TAKE 01 Original, TAKE 02 Remove, … */
export function takes(shot: Shot): Record<string, Take> {
  const out: Record<string, Take> = {};
  shot.versions.forEach((v, i) => {
    const label = isOriginal(v) ? "Original" : takeLabel(v.label || TASK_VERB[v.task as TaskId] || "Take").replace(/\s+v\d+$/i, "");
    out[v.id] = { num: String(i + 1).padStart(2, "0"), label };
  });
  return out;
}

export function takeName(t: Take | undefined): string {
  return t ? `Take ${t.num} · ${t.label}` : "Take";
}

/** "STUDIO · RTX 4090 · READY · 18 GB FREE" */
/** The features a first run sets up (Setup's Models rows and the Connect scan). */
const SETUP_FEATURES = ["recast", "remove", "environment", "relight", "previz", "finish", "image"] as const;

/** "gpu-pc", "This computer", "RunPod abc123": where the connected ComfyUI is, as the pill names it. */
function serverLabel(status: Status | null): string | null {
  if (status?.machine?.name) return status.machine.name;
  try {
    const u = new URL(status?.comfy_url ?? "");
    if (["127.0.0.1", "localhost", "[::1]"].includes(u.hostname)) return "This computer";
    return u.hostname.endsWith(".proxy.runpod.net") ? `RunPod ${u.hostname.split("-")[0]}` : u.hostname;
  } catch {
    return null;
  }
}

export function statusLine(status: Status | null, failed: boolean): string {
  const state = pillState(status, failed);
  if (state === "checking") return "Checking…";
  if (state === "offline") return "Connect ComfyUI";
  if (state === "setup") {
    const tiles = status?.tiles as Partial<Record<string, { ready?: boolean }>> | undefined;
    const ready = SETUP_FEATURES.filter((t) => tiles?.[t]?.ready).length;
    return [serverLabel(status), status?.machine ? shortGpu(status.machine.gpu) : null, `${ready} of ${SETUP_FEATURES.length} ready`].filter(Boolean).join(" · ");
  }
  const m = status?.machine;
  if (status?.comfy_queue?.outside) return [serverLabel(status), m ? shortGpu(m.gpu) : null, "Busy with another ComfyUI job"].filter(Boolean).join(" · ");
  const free = m?.vram_free_gb != null && isFinite(m.vram_free_gb) ? `${gb(m.vram_free_gb)} free` : null;
  return [serverLabel(status), m ? shortGpu(m.gpu) : null, "Ready", free].filter(Boolean).join(" · ");
}

/** Receipt as a short slate fragment: "0% AI", "<1% AI"; for an upscale (Finish) "Upscaled 2× · 9% AI", as the share
 * video's slate (share.stat_line): the upscaled share is the footage, not AI, and a bare "9% AI" would read as 91%
 * original. */
export function aiShare(r: Receipt | undefined | null): string | null {
  if (!receiptPercents(r) || !r) return null;
  const ai = `${percentText(r, "ai")} AI`;
  return Number(r.upscaled) > 0 ? `Upscaled${scaleText(r)} · ${ai}` : ai;
}

/** A model as a receipt names it: "Qwen-Image 2.1", or "Qwen-Image 2.1 (via Take 02)" for one that made the take this
 * take was made from (receipts.licence_name). */
export function licenceName(m: ModelLicence): string {
  return m.via ? `${m.name} (via ${m.via})` : m.name;
}

/** The models a take ran (or its input chain ran: "via Take 02") whose licence doesn't allow paid work. */
export function nonCommercialModels(v: Pick<Version, "licences"> | null | undefined): string[] {
  return (v?.licences ?? []).filter((m) => m.commercial === false).map(licenceName);
}

/** The NON-COMMERCIAL pill's guidance: which models, and for the default frame editor, what to do about it. The
 * client's receipt sheet only says the take isn't for paid work; the guidance is for the editor, in the app. */
export function nonCommercialHint(names: string[]): string {
  if (!names.length) return "";
  const which = `${names.join(", ")} ${names.length === 1 ? "is" : "are"} licensed for non-commercial use only.`;
  const editor = names.find((n) => /^Qwen-Image 2\.1\b/.test(n));
  if (!editor) return which;
  const via = /\(via (Take \d+)\)$/.exec(editor)?.[1];
  return `${which} For paid work, open Setup → Licenses → Frame editor to switch to the Apache-2.0 editor, then make ${via ? `${via}, then this take,` : "the take"} again.`;
}

/** Tasks to offer: Reshoot is a Labs tool and only shows when this machine can actually run it. */
export function visibleTasks(status: Status | null): TaskId[] {
  return TASKS.filter((t) => t !== "reshoot" || !!status?.tiles?.reshoot?.ready);
}
