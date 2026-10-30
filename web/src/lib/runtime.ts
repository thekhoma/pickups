// Render times, measured only: how long a running job has been going, how long the same kind usually takes here (the
// server's median of the last measured runs, pickups/app/timings.py), and how long a finished one took. No estimates.
// The plain maths is in lib/timing.ts (node tests import it); this adds the words and the ticking hook.
import { useEffect, useState } from "react";
import type { Job } from "../types";
import { clock, tookText } from "./format";
import { elapsedOf } from "./timing";

export { elapsedOf, epochS, kindOf, medianOf, tookOf } from "./timing";

/** The elapsed clock, as every duration in the app: 00:00:07 (lib/format clock). */
export const elapsedClock = (s: number | null | undefined): string => clock(s ?? 0);

/** "52 s", "3 min 5 s" (lib/format tookText). */
export const secondsText = tookText;

/** "Made in 52 s" from a measured time; "" when there is none. */
export function madeIn(s: number | null | undefined): string {
  return s != null && Number.isFinite(s) && s > 0 ? `Made in ${secondsText(s)}` : "";
}

/** "Usually about 52 s" after at least one measured run of the same kind; "" before. */
export function usually(s: number | null | undefined): string {
  return s != null && Number.isFinite(s) && s > 0 ? `Usually about ${secondsText(s)}` : "";
}

/** Seconds a running job has been going, ticking once a second. */
export function useElapsed(job: Pick<Job, "status" | "started_at"> | null | undefined): number | null {
  const running = job?.status === "running";
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [running]);
  return elapsedOf(job, now);
}
