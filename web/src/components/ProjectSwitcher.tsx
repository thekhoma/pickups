// Project counts, shared by the Projects page and project headers. (The top-bar project dropdown that lived here is
// gone: projects are in the left panel, components/Sidebar.tsx.)
import type { Project } from "../types";

export function projectCounts(p: Project): string {
  const n = (x: number, one: string, many: string) => `${x} ${x === 1 ? one : many}`;
  return [n(p.shots, "shot", "shots"), n(p.takes, "take", "takes"), p.approved ? `${p.approved} approved` : null].filter(Boolean).join(" · ");
}
