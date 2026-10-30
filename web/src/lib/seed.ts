// Lock seed: one control wherever a generation starts from a random seed (Create's images and videos, the start frame,
// a character sheet). The page always picks the seed it sends, so the seed shows on the result at once: locked, the
// last seed again; off, a fresh one each run. pickups/seeds.py does the same on the server for callers that send none.

export const SEED_MAX = 2_147_483_647;

/** A fresh random seed, 1..SEED_MAX. */
export function freshSeed(rand: () => number = Math.random): number {
  return 1 + Math.floor(rand() * (SEED_MAX - 1));
}

export interface SeedState {
  /** The seed the last run used (or the one a reopened result was made with); null before any. */
  seed: number | null;
  locked: boolean;
}

/** Lock seed is on only with a seed to reuse (Run 10: after a reload, a lock pressed with no seed sent a random seed
 * as "locked"). */
export function isLocked(s: SeedState): boolean {
  return s.locked && s.seed != null;
}

/** The seed the next run sends: the same one while locked (and known), else a fresh one. */
export function nextSeed(s: SeedState, rand?: () => number): number {
  return s.locked && s.seed != null ? s.seed : freshSeed(rand);
}

/** After a run: the seed it used becomes the one Lock seed keeps. */
export function usedSeed(s: SeedState, seed: number | null | undefined): SeedState {
  return seed == null || !Number.isFinite(seed) ? s : { ...s, seed: Math.round(seed) };
}

/** A result's saved seed and lock, for a reopened form ("Same settings", a made shot): locked only with a seed. */
export function seedFrom(seed: unknown, locked: unknown): SeedState {
  const n = typeof seed === "number" && Number.isFinite(seed) ? Math.round(seed) : typeof seed === "string" && /^\d+$/.test(seed) ? Number(seed) : null;
  return { seed: n, locked: n != null && locked === true };
}
