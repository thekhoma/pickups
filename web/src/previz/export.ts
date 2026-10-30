// Render every frame to JPEG and upload it. Batches of 16, with cancel.
import { getApi } from "../actions";
import type { PrevizFraming, Shot } from "../types";
import { plainError } from "./errors";
import { blankReason } from "./inView";
import { PrevizSpecSchema, shotFrames, type PrevizSpec } from "./spec";
import { frameIsBlank, type PrevizRuntime } from "./runtime";

const BATCH = 16;

export class ExportCanceled extends Error {
  constructor() {
    super("Canceled");
    this.name = "ExportCanceled";
  }
}

/** Frames whose pixels are checked for a failed draw: every 8th and the last (a context lost halfway shows up). */
export function checkedFrame(i: number, n: number): boolean {
  return i % 8 === 0 || i === n - 1;
}

function sessionId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return `pv_${[...bytes].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}

function jpeg(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Couldn't read a frame."))), "image/jpeg", 0.95);
  });
}

export async function exportPreviz(
  rt: PrevizRuntime,
  spec: PrevizSpec,
  opts: {
    cap: number; projectId: string; name: string; signal: AbortSignal; onProgress: (fraction: number) => void; framing?: PrevizFraming | null;
    /** Frame 0's subject matte (runtime.subjectMatte), a PNG data URL. */
    matte?: string | null;
    /** The depth passes (runtime.depthPasses): frame number to a PNG data URL. */
    depth?: Record<string, string> | null;
  },
): Promise<Shot> {
  let parsed: PrevizSpec;
  try {
    parsed = PrevizSpecSchema.parse(spec);
  } catch {
    throw new Error("This layout couldn't be saved. Reset the camera, then try Make shot again.");
  }
  const wanted = Math.max(1, Math.round(parsed.duration_s * parsed.fps + 1e-6));
  if (wanted > opts.cap) throw new Error(`This GPU renders up to ${opts.cap} frames. Shorten the shot.`);
  const n = shotFrames(parsed.duration_s, parsed.fps, opts.cap);
  if (opts.signal.aborted) throw new ExportCanceled();

  if (getApi().mode === "mock") {
    for (let i = 1; i <= 8; i++) {
      if (opts.signal.aborted) throw new ExportCanceled();
      opts.onProgress(i / 8);
      await new Promise((r) => setTimeout(r, 70));
    }
    return getApi().previzFinish("pv_mock", { spec: parsed, project_id: opts.projectId, name: opts.name, framing: opts.framing, matte: opts.matte, depth: opts.depth }, opts.signal);
  }

  const session = sessionId();
  const canvas = rt.renderer.domElement;
  let batch: Blob[] = [];
  let start = 0;
  const flush = async () => {
    if (!batch.length) return;
    try {
      await getApi().previzFrames(session, start, batch, opts.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") throw new ExportCanceled();
      throw new Error(plainError(e, "upload"));
    }
    start += batch.length;
    batch = [];
  };
  rt.setClean(true);
  try {
    for (let i = 0; i < n; i++) {
      if (opts.signal.aborted) throw new ExportCanceled();
      rt.renderFrame(i);
      if (checkedFrame(i, n) && frameIsBlank(rt.renderer)) {
        // Five flat pixels: say why (the camera in a hedge, nothing but sky), or carry on when the rays find the subject
        // or set pieces (a desert under a crane). The server still checks every sampled frame's full picture.
        const why = blankReason(rt.viewAt(i), i + 1);
        rt.renderFrame(i);
        if (why) throw new Error(why);
      }
      batch.push(await jpeg(canvas));
      if (batch.length === BATCH) await flush();
      opts.onProgress(((i + 1) / n) * 0.85);
      await new Promise((r) => requestAnimationFrame(() => r(undefined)));
    }
    await flush();
    opts.onProgress(0.92);
    let shot: Shot;
    try {
      shot = await getApi().previzFinish(session, { spec: parsed, project_id: opts.projectId, name: opts.name, framing: opts.framing, matte: opts.matte, depth: opts.depth }, opts.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") throw new ExportCanceled();
      throw new Error(plainError(e, "finish"));
    }
    opts.onProgress(1);
    return shot;
  } catch (e) {
    void getApi().previzCancel(session).catch(() => undefined);
    if ((e as Error).name === "AbortError") throw new ExportCanceled();
    throw e;
  } finally {
    rt.setClean(false);
  }
}
