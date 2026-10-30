// An older previz shot's frame 0, re-rendered from the spec it was made with (step 18b): the framing a new shot records
// (its people and vehicles, the camera, the key light, what's in view) and the subject matte, so its start frame gets the
// words and the mask a new shot gets. Off screen, in the same runtime the Previz page uses: frame i is always the same
// picture, so this is the frame the shot was exported from.
import { PrevizRuntime, type Frame0 } from "./runtime";
import { PrevizSpecSchema, type PrevizSpec } from "./spec";

/** The server writes a spec's unset fields as null; the page's schema wants them left out. */
function withoutNulls(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(withoutNulls);
  if (v && typeof v === "object") {
    return Object.fromEntries(Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== null).map(([k, x]) => [k, withoutNulls(x)]));
  }
  return v;
}

/** A saved shot's spec as the Previz page opens it (nulls dropped, then the page's schema), or null if it doesn't parse. */
export function openedSpec(rawSpec: unknown): PrevizSpec | null {
  const parsed = PrevizSpecSchema.safeParse(withoutNulls(rawSpec));
  return parsed.success ? parsed.data : null;
}

/** Frame 0, its subject matte and the depth passes, or null when this browser can't rebuild the scene (no WebGL, a spec
 * it can't read). */
export async function frame0Again(rawSpec: unknown): Promise<{ framing: Frame0 | null; matte: string | null; depth: Record<string, string> | null } | null> {
  const parsed = PrevizSpecSchema.safeParse(withoutNulls(rawSpec));
  if (!parsed.success || typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = 1920;
  canvas.height = 1088;
  let rt: PrevizRuntime | null = null;
  try {
    rt = new PrevizRuntime(canvas);
    await rt.setSpec(parsed.data, 0, [1920, 1088]);
    const framing = rt.framing0();
    const matte = rt.subjectMatte();
    const depth = rt.depthPasses();
    return framing || matte ? { framing, matte, depth } : null;
  } catch {
    return null;
  } finally {
    rt?.dispose();
  }
}

/** A made previz shot whose frame 0 lacks what the start frame needs (made before steps 17 and 17b, or before the depth
 * pass and facing were recorded). */
export function frame0IsThin(shot: { previz_spec?: unknown; previz_framing?: { subjects?: { facing?: unknown }[]; camera?: unknown }; previz_matte_url?: string; previz_depth_url?: string }): boolean {
  const subjects = shot.previz_framing?.subjects;
  return !!shot.previz_spec && (!subjects || !shot.previz_framing?.camera || !shot.previz_matte_url || !shot.previz_depth_url
    || !subjects.some((s) => s.facing));
}
