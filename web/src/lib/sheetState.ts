// A cast member's character sheet, as one state everywhere it shows (the cast grid, Recast's Character step, the
// full-screen viewer): not made, making ("2 of 3"), ready, or failed. Read from the member and its newest sheet job, so
// every place agrees and a reload finds the same state. Plain functions: tested without React (tests/previz).
import type { CastMember, Job, SheetRequest } from "../types";

/** sheets.STAGES (pickups/sheets.py): one stage per view, then the composite. */
export const SHEET_STAGES = {
  front: "Making the front view",
  back: "Making the back view",
  face: "Making the close-up",
  sheet: "Putting the sheet together",
} as const;

export type SheetState =
  | { s: "none" }
  | { s: "making"; n: number; queued: boolean; job: Job }
  | { s: "ready" }
  | { s: "failed"; error: string; job: Job };

const active = (j: Job) => j.status === "queued" || j.status === "running";

/** Which of the 3 views is being made: front 1, back 2, close-up 3, and the composite still 3 (0 before the first). */
export function sheetStep(stage: string | undefined): number {
  switch (stage) {
    case SHEET_STAGES.front:
      return 1;
    case SHEET_STAGES.back:
      return 2;
    case SHEET_STAGES.face:
    case SHEET_STAGES.sheet:
    case "Done":
      return 3;
    default:
      return 0;
  }
}

const createdAt = (j: Job) => (typeof j.created_at === "number" ? j.created_at : Date.parse(String(j.created_at ?? "")) / 1000 || 0);

/** The newest sheet job for this member (it upgrades them: `cast_id`, or it made them: `result_cast_id`). */
export function sheetJobFor(castId: string | null | undefined, jobs: Iterable<Job>): Job | undefined {
  if (!castId) return undefined;
  let best: Job | undefined;
  for (const j of jobs) {
    if (j.task !== "sheet") continue;
    if ((j.params as SheetRequest | undefined)?.cast_id !== castId && j.result_cast_id !== castId) continue;
    if (!best || createdAt(j) >= createdAt(best)) best = j;
  }
  return best;
}

/**
 * The member's sheet state. A job at work wins; a failed one counts until a sheet exists (a failed redo leaves the
 * good sheet in place). A wide picture is a sheet already. A canceled job is as if it never ran.
 */
export function sheetStateOf(member: Pick<CastMember, "sheet"> | null | undefined, job: Job | undefined, wide = false): SheetState {
  if (job && active(job)) return { s: "making", n: sheetStep(job.stage), queued: job.status === "queued", job };
  if (member?.sheet || wide) return { s: "ready" };
  if (job && job.status === "failed") return { s: "failed", error: job.error || "", job };
  return { s: "none" };
}

/** The step's one primary for the state: Make sheet, Making the sheet (off), Use <name>, Try again. */
export function sheetPrimary(st: SheetState, name: string): { label: string; disabled: boolean } {
  switch (st.s) {
    case "none":
      return { label: "Make sheet", disabled: false };
    case "making":
      return { label: "Making the sheet", disabled: true };
    case "ready":
      return { label: `Use ${name}`, disabled: false };
    case "failed":
      return { label: "Try again", disabled: false };
  }
}

/** The short state line: "No sheet", "Queued", "2 of 3", "Sheet", "Failed" (one word for a failure everywhere). */
export function sheetLabel(st: SheetState): string {
  switch (st.s) {
    case "none":
      return "No sheet";
    case "making":
      return st.queued ? "Queued" : `${Math.max(1, st.n)} of 3`;
    case "ready":
      return "Sheet";
    case "failed":
      return "Failed";
  }
}

/** The failure's own words without a "Try again." on the end (the button under it says that already). */
export function sheetError(text: string | null | undefined): string {
  return (text ?? "").replace(/\s*(?:Please\s+)?try again\.?\s*$/i, "").trim();
}

/**
 * The request for Make sheet and Try again: always by `cast_id` (the route refuses a second member with the same
 * name, server.py api_cast_sheet), with a new seed when it's made again.
 */
export function sheetRequest(castId: string, again: boolean, seed: number): SheetRequest {
  return again ? { cast_id: castId, seed } : { cast_id: castId };
}

/** What the viewer shows for a member: each view, then the whole sheet; while making, the views made so far. */
export function sheetViews(member: Pick<CastMember, "sheet" | "url" | "thumb_url" | "name"> | null | undefined, st: SheetState): { key: string; label: string; url: string | null }[] {
  if (!member) return [];
  const views = st.s === "making" ? { ...(st.job.partial ?? {}) } : member.sheet?.views ?? {};
  const three = [
    { key: "front", label: "Front", url: views.front ?? null },
    { key: "back", label: "Back", url: views.back ?? null },
    { key: "face", label: "Close-up", url: views.face ?? null },
  ];
  if (st.s === "making") return three;
  if (member.sheet) return [...three.filter((v) => v.url), { key: "sheet", label: "Sheet", url: member.sheet.url }];
  return [{ key: "picture", label: member.name, url: member.url || member.thumb_url || null }];
}
