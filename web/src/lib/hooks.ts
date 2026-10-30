import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useApp } from "../store";
import type { Job } from "../types";
import { isActive, toEpochMs } from "./format";

/** Counts an ETA down locally between server updates. Returns seconds left (or null when unknown). */
export function useCountdown(eta_s: number | null | undefined, running: boolean): number | null {
  // Re-anchor the countdown whenever the server sends a new ETA.
  const at = useMemo(() => Date.now(), [eta_s]);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [running]);
  if (eta_s == null || !isFinite(eta_s)) return null;
  if (!running) return eta_s;
  return Math.max(0, eta_s - Math.max(0, now - at) / 1000);
}

/** Where a queued job stands: Pickups jobs ahead of it, and ComfyUI jobs Pickups didn't send that it waits behind
 * (status.comfy_queue.outside: ComfyUI is busy with someone else's work). */
export function useQueuePlace(job: Job | null | undefined): { ahead: number; outside: number } {
  const jobs = useApp((s) => s.jobs);
  const q = useApp((s) => s.status?.comfy_queue);
  const ahead = useMemo(() => {
    if (!job || job.status !== "queued") return 0;
    const mine = toEpochMs(job.created_at) ?? 0;
    // Only the GPU queue is ahead of it: inline rows (the start frame) run beside it, never in front.
    return Object.values(jobs).filter((j) => j.id !== job.id && j.lane !== "inline" && isActive(j) && (toEpochMs(j.created_at) ?? 0) < mine).length;
  }, [jobs, job]);
  const outside = job?.status === "queued" && q?.outside ? Math.max(1, (q.running || 0) + (q.pending || 0)) : 0;
  return { ahead, outside };
}

/** "Behind 1 ComfyUI job", "2 jobs ahead", "Starting soon": where a queued job is, in a few words. */
export function queueWords({ ahead, outside }: { ahead: number; outside: number }): string {
  if (outside && !ahead) return `Behind ${outside} ComfyUI ${outside === 1 ? "job" : "jobs"}`;
  const n = ahead + outside;
  return n ? `${n} ${n === 1 ? "job" : "jobs"} ahead` : "Starting soon";
}

/** The same, shorter, after a label ("Making the sheet · 3 ahead"). */
export function queueShort({ ahead, outside }: { ahead: number; outside: number }): string {
  if (outside && !ahead) return `${outside} ahead`;
  const n = ahead + outside;
  return n ? `${n} ahead` : "next";
}

const mediaQueries = new Map<string, MediaQueryList>();

/** Whether a media query matches, kept up to date ("(max-width: 767px)"). */
export function useMedia(query: string): boolean {
  let mq = mediaQueries.get(query);
  if (!mq && typeof window !== "undefined") {
    mq = window.matchMedia(query);
    mediaQueries.set(query, mq);
  }
  return useSyncExternalStore(
    (on) => {
      mq?.addEventListener("change", on);
      return () => mq?.removeEventListener("change", on);
    },
    () => !!mq?.matches,
  );
}
