import { Clapperboard, CloudSnow, Eraser, Move3d, Sparkles, SunMedium, UserRoundCog, type LucideIcon } from "lucide-react";
import type { TaskId } from "../types";
import { CARS } from "../lib/features";

export const TASK_META: Record<TaskId, { verb: string; line: string; icon: LucideIcon; beta?: boolean }> = {
  remove: { verb: "Remove a person", line: "Erase a person, boom or sign.", icon: Eraser },
  recast: { verb: "Replace a person", line: CARS ? "Same performance, new person or car." : "Same performance, new person.", icon: UserRoundCog },
  environment: { verb: "Change the weather", line: "Snow, rain or golden hour.", icon: CloudSnow },
  relight: { verb: "Relight", line: "New light on the whole shot.", icon: SunMedium },
  previz: { verb: "Previz to final", line: "Turn a previz into a finished shot.", icon: Clapperboard },
  /** Labs: shown only when this machine can run it. */
  reshoot: { verb: "Reshoot", line: "Move the camera.", icon: Move3d, beta: true },
  finish: { verb: "Finish", line: "Upscale and grade a master.", icon: Sparkles },
};

/** "Keep going →" suggestions after a result. */
export const NEXT_STEPS: Record<TaskId, Array<{ label: string; task: TaskId }>> = {
  remove: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Remove something else", task: "remove" },
  ],
  recast: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Remove something", task: "remove" },
  ],
  environment: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Try another weather", task: "environment" },
  ],
  relight: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Try another light", task: "relight" },
  ],
  previz: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Change the weather", task: "environment" },
  ],
  reshoot: [
    { label: "Upres to 4K / HDR", task: "finish" },
    { label: "Remove something", task: "remove" },
  ],
  finish: [
    { label: "Remove something", task: "remove" },
    { label: "Replace a person", task: "recast" },
  ],
};
