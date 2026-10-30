// The shot-spec parser. Loaded once in the browser, then parseShot runs on every keystroke.
import { buildLexicon } from "./runtime/lexicon.js";
import { parseShot } from "./runtime/grammar.js";
import type { Lexicon, ShotSpec } from "./runtime/types";
import type { PrevizSpec } from "./spec.ts";
import type { CameraPick } from "./compile.ts";

/** Off, the keyword lexicon in lexicon.ts stays the scene. `?shotspec=0` does the same for one tab. */
export const USE_SHOT_SPEC = true;

let lex: Lexicon | null = null;
let grammar: object | null = null;
let pending: Promise<void> | null = null;
let failed = false;
const failWatch = new Set<(failed: boolean) => void>();

/** True once the manifest or the grammar failed to load: the page says so instead of building the keyword scene. */
export function shotLibraryFailed(): boolean {
  return failed;
}

export function onShotLibrary(fn: (failed: boolean) => void): () => void {
  failWatch.add(fn);
  return () => {
    failWatch.delete(fn);
  };
}

function setFailed(next: boolean): void {
  if (failed === next) return;
  failed = next;
  for (const fn of failWatch) fn(next);
}

/** The loaded grammar (grammar.json), for the "Try:" row. null until it loads. */
export function shotGrammar(): object | null {
  return grammar;
}
const picks = new WeakMap<PrevizSpec, CameraPick | null>();

/** The camera button that compiled this spec, if the user pressed one. */
export function rememberPick<T extends { spec: PrevizSpec }>(out: T, pick: CameraPick | null): T {
  picks.set(out.spec, pick);
  return out;
}

export function pickFor(spec: PrevizSpec): CameraPick | null {
  return spec.camera.pick ?? picks.get(spec) ?? null;
}

export function shotSpecOn(): boolean {
  if (!USE_SHOT_SPEC) return false;
  if (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("shotspec") === "0") return false;
  return true;
}

export function readShot(text: string): ShotSpec | null {
  if (!lex || !grammar) return null;
  return parseShot(text, lex, grammar);
}

/** Fetch the manifest and the grammar once. Node, and a tab with the flag off, never call the network. */
export function ensureShotLexicon(): Promise<void> {
  if (!shotSpecOn() || typeof window === "undefined") return Promise.resolve();
  if (lex && grammar) return Promise.resolve();
  if (pending) return pending;
  pending = Promise.all([
    fetch("/previz/data/library_manifest.json").then((res) => {
      if (!res.ok) throw new Error("library manifest " + res.status);
      return res.json() as Promise<object>;
    }),
    fetch("/previz/data/grammar.json").then((res) => {
      if (!res.ok) throw new Error("grammar " + res.status);
      return res.json() as Promise<object>;
    }),
  ])
    .then(([manifest, rules]) => {
      grammar = rules;
      lex = buildLexicon(manifest, rules);
      setFailed(false);
    })
    .catch((err: unknown) => {
      pending = null;
      console.error("[previz] shot library", err);
      setFailed(true);
      throw err;
    });
  return pending;
}

if (typeof window !== "undefined") ensureShotLexicon().catch(() => undefined);
