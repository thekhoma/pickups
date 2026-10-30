// Every Previz error a person can see is one plain sentence: what happened and what to do. The raw text goes to the
// console for whoever is debugging, never into the page.

export const LIBRARY_FAILED = "The shot library didn't load. Reload the page.";

export type PrevizErrorWhere = "build" | "upload" | "finish" | "export";

const SAID: Record<PrevizErrorWhere, string> = {
  build: "That shot didn't build, so this is the plain grey scene. Try rewording it.",
  upload: "The frames didn't upload. Check that Pickups is still running, then try Make shot again.",
  finish: "The shot didn't save. Try Make shot again.",
  export: "Couldn't make the shot. Try Make shot again.",
};

/** File names, status codes, paths and stack text: what a plain sentence never contains. */
const RAW = /(\.(glb|gltf|json|js|ts|png|jpe?g|mov|mp4)\b)|(\b[1-5]\d\d\b)|(\/[\w.-]+\/)|(\bat [\w.<>]+ \()|(Error\b)|(\bundefined\b)|(\bnull\b)|(\bfetch\b)/i;

/** A sentence safe to show as it is: no file names, codes, paths or stack text. */
export function isPlain(text: string): boolean {
  return !!text && !RAW.test(text);
}

/** A message from our own server that is already a plain sentence (PrevizError lines, the GPU cap). */
function plainServerLine(e: unknown): string | null {
  // ApiError (api.ts), read by shape so this file has no imports.
  const err = e as { message?: unknown; status?: unknown } | null;
  if (!err || typeof err.message !== "string" || !err.message || err.status !== 400) return null;
  return RAW.test(err.message) ? null : err.message;
}

export function plainError(e: unknown, where: PrevizErrorWhere): string {
  console.error(`[previz ${where}]`, e);
  if (where !== "build") {
    const line = plainServerLine(e);
    if (line) return line;
  }
  const text = e instanceof Error ? e.message : String(e ?? "");
  if (where === "build" && /\b(kit|model|asset|library|manifest|grammar|glb|loader?)\b|404|failed to fetch|networkerror/i.test(text)) {
    return "Part of the 3D kit didn't load. Reload the page, then try again.";
  }
  return SAID[where];
}
