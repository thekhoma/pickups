// Drafts (SPEC section 2): what a form or a step flow has typed and picked, per shot and task, so switching shot, task or
// rail never loses it. Plain functions over a record, so they run (and are tested) without React or a browser.

/** A picture a step made (an upload): the source of truth when a job id is long gone. */
export interface DraftOutput {
  upload_id: string;
  url: string;
  seed?: number;
}

export interface Draft {
  /** The step the flow is on ("who", "render", "review"; "" for a plain form). */
  step: string;
  data: Record<string, unknown>;
  outputs: Record<string, DraftOutput>;
  /** Job ids per step, best effort: jobs live in the server's memory, so an unknown id means gone, never busy. */
  jobs: Record<string, string>;
  inputsHash?: string;
  /** A step whose inline work (the start frame) was in flight when the page last saw it: re-attach by the draft key. */
  pending?: string | null;
  /** Only an inline row created after this (the server's clock: the newest row for the step when the click happened)
   * is the pending one, never an older try. */
  pendingAfter?: number;
  updatedAt: number;
}

export const DRAFTS_KEY = "pickups.drafts.v1";
export const MAX_DRAFTS = 50;
export const DRAFT_TTL_MS = 14 * 24 * 3600 * 1000;

export const draftKey = (shotId: string, task: string): string => `${shotId}:${task}`;

export function emptyDraft(step = "", data: Record<string, unknown> = {}, now = Date.now()): Draft {
  return { step, data, outputs: {}, jobs: {}, updatedAt: now };
}

/** Drop drafts older than 14 days, then keep the 50 most recently touched. */
export function pruneDrafts(all: Record<string, Draft>, now = Date.now()): Record<string, Draft> {
  const live = Object.entries(all).filter(([, d]) => d && typeof d.updatedAt === "number" && now - d.updatedAt <= DRAFT_TTL_MS);
  live.sort((a, b) => b[1].updatedAt - a[1].updatedAt);
  return Object.fromEntries(live.slice(0, MAX_DRAFTS));
}

/** One draft changed (merged over what's there, touched now), the rest pruned. */
export function putDraft(all: Record<string, Draft>, key: string, patch: Partial<Draft>, now = Date.now()): Record<string, Draft> {
  const prev = all[key] ?? emptyDraft("", {}, now);
  return pruneDrafts({ ...all, [key]: { ...prev, ...patch, updatedAt: Math.max(now, prev.updatedAt + 1) } }, now);
}

/** "Try again with changes" (a prefill) starts a fresh draft from the take's settings: it wins over a stale one. */
export function seedDraft(all: Record<string, Draft>, key: string, step: string, data: Record<string, unknown>, now = Date.now()): Record<string, Draft> {
  return pruneDrafts({ ...all, [key]: { ...emptyDraft(step, data, now), updatedAt: Math.max(now, (all[key]?.updatedAt ?? 0) + 1) } }, now);
}

export function dropDraft(all: Record<string, Draft>, key: string): Record<string, Draft> {
  if (!(key in all)) return all;
  const next = { ...all };
  delete next[key];
  return next;
}

interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export function loadDrafts(storage: Storage | null | undefined, now = Date.now()): Record<string, Draft> {
  try {
    const raw = storage?.getItem(DRAFTS_KEY);
    const all = raw ? (JSON.parse(raw) as Record<string, Draft>) : {};
    return all && typeof all === "object" ? pruneDrafts(all, now) : {};
  } catch {
    return {};
  }
}

export function saveDrafts(storage: Storage | null | undefined, all: Record<string, Draft>): void {
  try {
    storage?.setItem(DRAFTS_KEY, JSON.stringify(all));
  } catch {
    /* private mode or full: the drafts live for this session only */
  }
}
