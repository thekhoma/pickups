// After a delete (Run 6): where new work goes once a project is gone, the confirm's words, and a reopened form's settings
// without the cast entries and images that are gone. Pure, so tests/previz/run6_delete.mjs checks them.
import type { Project } from "../types";

const at = (p: Project) => {
  const t = p.updated_at ?? p.created_at;
  return typeof t === "number" ? t : Date.parse(String(t ?? "")) / 1000 || 0;
};

/** The current project, unless it was the one deleted: then the most recent project left (never Examples, Test runs
 * or an archived one), else the fresh empty My shots the server makes, so there's always a project. */
export function projectAfterDelete(left: Project[], deletedId: string, currentId: string): string {
  if (currentId !== deletedId && left.some((p) => p.id === currentId)) return currentId;
  const open = left.filter((p) => p.id !== deletedId && p.id !== "examples" && p.id !== "qc" && !p.archived);
  return open.sort((a, b) => at(b) - at(a))[0]?.id ?? "unsorted";
}

/** The delete confirm's one line under "Delete <name>?": a project's shots or a clip's takes that go with it, or the
 * shots that use a picture (they keep their takes). Null when there's nothing to say. */
export function deleteLine(kind: "project" | "shot" | "asset", count: number): string | null {
  const n = (one: string) => `${count} ${count === 1 ? one : `${one}s`}`;
  if (kind === "project") return !count ? "No shots in it." : `${n("shot")} ${count === 1 ? "goes" : "go"} with it.`;
  if (kind === "shot") return count ? `${n("take")} ${count === 1 ? "goes" : "go"} with it.` : null;
  return count ? `Used in ${n("shot")}.` : null;
}

/** The danger button of a cast entry's confirm on the Cast tab: a person is a character; a prop or a place an asset. */
export function castDeleteLabel(kind: string): string {
  return kind === "person" ? "Delete character" : "Delete asset";
}

/** The references a take's settings keep ({kind, id}, one or a list). */
const REF_LISTS = ["look_refs", "refs"];
const REF_ONES = ["look_frame_ref", "first_frame_ref", "who_ref"];

/**
 * A take's settings for "Try again with changes" without the cast entries and images deleted since: the form opens
 * without them (no blank picture), as the server skips them. `cast` and `images`: the ids the project has now; null
 * when not loaded yet (then nothing is dropped). A gone cast_id goes too, also from each person of a multi-person Recast
 * (`swaps`), who then keeps their words.
 */
export function withoutGoneRefs(params: Record<string, unknown>, cast: string[] | null, images: string[] | null): Record<string, unknown> {
  const gone = (r: unknown) => {
    const ref = r as { kind?: string; id?: string } | null;
    if (!ref || typeof ref !== "object") return false;
    if (ref.kind === "cast") return !!cast && !cast.includes(String(ref.id));
    if (ref.kind === "image") return !!images && !images.includes(String(ref.id));
    return false;
  };
  const strip = (o: Record<string, unknown>) => {
    const out = { ...o };
    for (const k of REF_LISTS) if (Array.isArray(out[k])) out[k] = (out[k] as unknown[]).filter((r) => !gone(r));
    for (const k of REF_ONES) if (gone(out[k])) delete out[k];
    if (cast && typeof out.cast_id === "string" && out.cast_id && !cast.includes(out.cast_id)) delete out.cast_id;
    return out;
  };
  const out = strip(params);
  if (Array.isArray(out.swaps)) out.swaps = out.swaps.map((s) => (s && typeof s === "object" ? strip(s as Record<string, unknown>) : s));
  return out;
}
