// Finish's one choice: 4K HDR (default), 1080p HDR, or 4K. It maps onto the task's two params, so presets and
// recipes that saved upscale_4k and hdr keep working.
export type FinishChoice = "4k-hdr" | "hdr" | "4k";

export const FINISH_CHOICES: Array<{ value: FinishChoice; label: string }> = [
  { value: "4k-hdr", label: "4K HDR" },
  { value: "hdr", label: "1080p HDR" },
  { value: "4k", label: "4K" },
];

/** A source already 3840 or more on its long side can't go up to 4K: only HDR is offered (labelled plainly). */
export function finishChoices(sourceLong: number): Array<{ value: FinishChoice; label: string }> {
  return sourceLong >= 3840 ? [{ value: "hdr", label: "HDR" }] : FINISH_CHOICES;
}

export function choiceOf(params: { upscale_4k?: unknown; hdr?: unknown } | null | undefined, sourceLong = 0): FinishChoice {
  const k4 = params?.upscale_4k !== false;
  const hdr = params?.hdr !== false;
  if (sourceLong >= 3840) return "hdr";
  return k4 && hdr ? "4k-hdr" : hdr ? "hdr" : k4 ? "4k" : "4k-hdr";
}

export function paramsOf(choice: FinishChoice): { upscale_4k: boolean; hdr: boolean } {
  return { upscale_4k: choice !== "hdr", hdr: choice !== "4k" };
}

/** Share of the 4K HDR time each choice takes (RTX 4090 runs: the 4K upscale about 890 s, HDR about 360 s). */
export const FINISH_SHARE: Record<FinishChoice, number> = { "4k-hdr": 1, "4k": 890 / 1250, hdr: 360 / 1250 };

/** A finish job's short name with its preset, as its take will be named: "Finish 4K HDR", "Finish 1080p HDR". */
export function finishLabel(params: { upscale_4k?: unknown; hdr?: unknown } | null | undefined): string {
  const c = choiceOf(params);
  return `Finish ${FINISH_CHOICES.find((x) => x.value === c)?.label ?? "4K HDR"}`;
}

export function finishTitle(choice: FinishChoice, sourceLong = 0): string {
  if (choice === "hdr") return sourceLong >= 3840 ? "Finish to HDR" : "Finish to 1080p HDR";
  return choice === "4k" ? "Finish to 4K" : "Finish to 4K HDR";
}
