// Where this Previz session saves. The page itself is not inside a project until you say so.
import type { Compiled } from "./compile";

export type PrevizHome = { kind: "project"; id: string } | { kind: "file" };

let home: PrevizHome | null = null;
let gate = 0;
let rememberedLine = "";
let rememberedBuilt: Compiled | null = null;
let rendering = false;
const listeners = new Set<() => void>();

function emit(): void {
  for (const fn of listeners) fn();
}

export function subscribePreviz(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

export function previzHome(): PrevizHome | null {
  return home;
}

export function previzGate(): number {
  return gate;
}

export function previzMemory(): { line: string; built: Compiled | null } {
  return { line: rememberedLine, built: rememberedBuilt };
}

export function rememberPreviz(line: string, built: Compiled | null): void {
  rememberedLine = line;
  rememberedBuilt = built;
}

/** Make shot is rendering: the nav's Previz must not wipe the scene under it. */
export function setPrevizRendering(on: boolean): void {
  rendering = on;
}

export function previzRendering(): boolean {
  return rendering;
}

/** Nav Previz, Home "New previz" and "New scene": back to the scene grid. The save choice for this session stays. */
export function resetPrevizGate(): void {
  gate += 1;
  rememberedLine = "";
  rememberedBuilt = null;
  emit();
}

export function setPrevizProject(id: string): void {
  home = { kind: "project", id };
  emit();
}

export function setPrevizFile(): void {
  home = { kind: "file" };
  emit();
}

/** Project id for a save, or null when the user has not chosen yet. */
export function previzSaveId(): string | null {
  if (!home) return null;
  return home.kind === "project" ? home.id : "unsorted";
}
