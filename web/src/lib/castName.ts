// Auto names for new cast members never collide: the name as it is when it's free, else the next free number
// ("Knight", "Knight 2", "Knight 3"). Plain function: tested without React (tests/previz).

/** `name`, or `name 2`, `name 3`… the first one no member has (case-insensitive). */
export function freeName(name: string, names: Iterable<string>): string {
  const taken = new Set([...names].map((n) => n.trim().toLowerCase()));
  let n = 1;
  while (taken.has((n === 1 ? name : `${name} ${n}`).toLowerCase())) n += 1;
  return n === 1 ? name : `${name} ${n}`;
}
