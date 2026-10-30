export function askedAnimals(sentence: string): Set<string>;
export function guardShot(input: {
  built: unknown;
  keys: { t: number; pos: [number, number, number]; look: [number, number, number]; focal_mm: number }[];
  aspect: number;
  sentence: string;
  shot?: unknown;
  locked?: boolean;
}): { keys: { t: number; pos: [number, number, number]; look: [number, number, number]; focal_mm: number }[]; notes: string[] };
export function auditShot(input: {
  built: unknown;
  keys: { t: number; pos: [number, number, number]; look: [number, number, number]; focal_mm: number }[];
  aspect: number;
  sentence: string;
  shot?: unknown;
}): string[];
