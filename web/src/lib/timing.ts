// Measured render times, the plain maths (no React, no words): a job's elapsed and finished seconds, the kind a run is
// timed as (pickups/app/timings.py kind_of says the same) and the median the page calls "usually".
import type { Job } from "../types";

/** Epoch seconds from a job's time field (seconds, milliseconds or an ISO string). */
export function epochS(t: number | string | null | undefined): number | null {
  if (t == null) return null;
  if (typeof t === "number") return t > 1e12 ? t / 1000 : t;
  const ms = Date.parse(t);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

/** Seconds a job has been running, from its start (the server's clock) to `now` (ms). Not running: null. */
export function elapsedOf(job: Pick<Job, "status" | "started_at"> | null | undefined, now: number): number | null {
  if (!job || job.status !== "running") return null;
  const start = epochS(job.started_at);
  return start == null ? null : Math.max(0, now / 1000 - start);
}

/** A finished job's measured time: took_s, else its start and finish. Not done: null. */
export function tookOf(job: Pick<Job, "status" | "took_s" | "started_at" | "finished_at"> | null | undefined): number | null {
  if (!job || job.status !== "done") return null;
  if (job.took_s != null) return job.took_s;
  const a = epochS(job.started_at);
  const b = epochS(job.finished_at);
  return a != null && b != null && b >= a ? b - a : null;
}

/** The kind a run is timed as: the same task, model and size take about the same time. */
export function kindOf(task: string, p: Record<string, unknown> = {}): string {
  const v = (k: string, d = "") => (p[k] == null || p[k] === "" ? d : String(p[k]));
  if (task === "image") return ["image", v("model"), v("aspect", "16:9"), v("resolution", "1K"), v("count", "1")].join("|");
  if (task === "video") {
    const size = p.width && p.height ? `${p.width}x${p.height}` : `${v("aspect")}@${v("resolution")}`;
    return ["video", v("mode", "text"), size, v("frames") || v("duration")].join("|");
  }
  if (task === "sheet") return ["sheet", p.describe && !(p.upload_id || p.image_id || p.cast_id) ? "words" : "picture"].join("|");
  if (task === "start_frame" || task === "look_frame") return [task, v("method")].join("|");
  return [task, v("engine") || v("model")].join("|");
}

/** "Usually": the median of the last five measured runs, whole seconds (at least 1); null before any. */
export function medianOf(runs: number[]): number | null {
  const last = runs.slice(-5).sort((a, b) => a - b);
  if (!last.length) return null;
  const mid = Math.floor(last.length / 2);
  const m = last.length % 2 ? last[mid] : (last[mid - 1] + last[mid]) / 2;
  return Math.max(1, Math.round(m));
}
