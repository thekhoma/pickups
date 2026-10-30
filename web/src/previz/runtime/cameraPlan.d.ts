export function planCamera(text: string): {
  move: string | null;
  segments: { move: string; from_t: number; to_t: number; params: Record<string, unknown>; tracking: number }[];
  handheld: { seed: number; pos_m: number; rot_deg: number; hz: number } | null;
  dutch: boolean;
  lensMm: number | null;
  facing: "toward" | "away" | "left_to_right" | "across" | null;
  reveal: boolean;
  chips: string[];
  speed: number | null;
};
