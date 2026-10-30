// Inline queue rows (the start frame, the look frame: SPEC section 3), found again by the page's draft.
import type { Job } from "../types";

const at = (j: Job): number => (typeof j.created_at === "number" ? j.created_at : Number(j.created_at ?? 0) || 0);

/** The newest inline row this draft started for a step, created after `after` (the newest such row when the click
 * happened): an older row, such as the last try that failed, is never taken for this one. */
export function newestInline(jobs: Record<string, Job>, key: string, step: string, after: number | null | undefined): Job | undefined {
  let best: Job | undefined;
  for (const j of Object.values(jobs)) {
    if (j.lane !== "inline" || j.draft_key !== key || j.step !== step) continue;
    if (after != null && at(j) <= after) continue;
    if (!best || at(j) > at(best)) best = j;
  }
  return best;
}

/** When the newest row for this draft and step was created (the server's clock), or 0 with none: what a new try must
 * come after. */
export function lastInlineAt(jobs: Record<string, Job>, key: string, step: string): number {
  const j = newestInline(jobs, key, step, null);
  return j ? at(j) : 0;
}
