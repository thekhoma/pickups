// Drafts in the app (SPEC section 2): useDraft keeps a form's or a step flow's words and picks per shot and task, in the
// store and in localStorage, so switching shot, task or rail (or reloading) never loses them. The flags for the step
// flows live here too.
import { useEffect, useState } from "react";
import { getState, setState, useApp } from "../store";
import { draftKey, dropDraft, emptyDraft, putDraft, saveDrafts, seedDraft, type Draft } from "./draftStore";

let saveTimer: number | undefined;

function storage(): globalThis.Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** Mirror to localStorage, at most every 300 ms (and once more as the page goes). */
function persist(): void {
  if (typeof window === "undefined") return;
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => saveDrafts(storage(), getState().drafts), 300);
}

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    window.clearTimeout(saveTimer);
    saveDrafts(storage(), getState().drafts);
  });
}

export function patchDraft(key: string, patch: Partial<Draft> | ((d: Draft) => Partial<Draft>)): void {
  setState((s) => {
    const cur = s.drafts[key] ?? emptyDraft();
    return { drafts: putDraft(s.drafts, key, typeof patch === "function" ? patch(cur) : patch) };
  });
  persist();
}

export function clearDraft(key: string): void {
  setState((s) => ({ drafts: dropDraft(s.drafts, key) }));
  persist();
}

export function getDraft(shotId: string, task: string): Draft | undefined {
  return getState().drafts[draftKey(shotId, task)];
}

/**
 * A form's state that survives switching shot, task or rail. `initial` is the form's own default (and fills keys a saved
 * draft lacks); `fresh` (a "Try again with changes" prefill) starts a new draft from `initial`, winning over a stale one.
 */
export function useDraft<D extends Record<string, unknown>>(shotId: string, task: string, initial: () => D, opts: { fresh?: boolean; step?: string } = {}) {
  const key = draftKey(shotId, task);
  const [base] = useState(initial);
  // Seeded on first render (a prefill, or no draft yet), written once mounted: never a store write during render.
  const [seeded] = useState<Draft | null>(() => (opts.fresh || !getState().drafts[key] ? emptyDraft(opts.step ?? "", base) : null));
  useEffect(() => {
    if (seeded) {
      setState((s) => ({ drafts: seedDraft(s.drafts, key, seeded.step, seeded.data) }));
      persist();
    }
  }, [key]);
  const stored = useApp((s) => s.drafts[key]);
  const draft: Draft = seeded && (!stored || stored.updatedAt < seeded.updatedAt) ? seeded : stored ?? emptyDraft(opts.step ?? "", base);
  const data = { ...base, ...draft.data } as D;
  return {
    key,
    draft,
    data,
    set: (p: Partial<D>) => patchDraft(key, (d) => ({ data: { ...base, ...d.data, ...p } })),
    patch: (p: Partial<Draft> | ((d: Draft) => Partial<Draft>)) => patchDraft(key, p),
    clear: () => clearDraft(key),
  };
}

/** Which step flows are on when `pickups.flow.<task>` isn't set. Remove, Relight and Finish have no other version. */
export const FLOW_DEFAULTS: Record<string, boolean> = { recast: true, character: false };

/** A step flow's switch: `pickups.flow.<task>` in localStorage ("1" on, "0" off), else FLOW_DEFAULTS. Never in the UI:
 * it is the way back to a task's old long form while it is kept. */
export function flowOn(task: string): boolean {
  try {
    const v = storage()?.getItem(`pickups.flow.${task}`);
    if (v === "1" || v === "true" || v === "on") return true;
    if (v === "0" || v === "false" || v === "off") return false;
  } catch {
    /* no storage: the default */
  }
  return FLOW_DEFAULTS[task] ?? false;
}
