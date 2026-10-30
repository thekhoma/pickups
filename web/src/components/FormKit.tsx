// Shared pieces of the task forms: the shell (header, readiness notice, footer), the primary action, uploads, fields.
import { ChevronDown, ImageOff } from "lucide-react";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { clearPrefill, getApi, openPrepare, openOnboarding, openSetup, railBack, setRenderWindow, submitJob } from "../actions";
import { aboutTime, cx, gb, isOriginal, halfSeconds, ltxWindowFrames, minWindowFrames, missingFor, shortGpu, takes, windowStart } from "../lib/format";
import { withoutGoneRefs } from "../lib/deleted";
import { draftKey } from "../lib/draftStore";
import { getState, useApp, type ShotUi } from "../store";
import type { JobParams, Ref, Shot, TaskId } from "../types";
import { RailBody, RailFooter, RailHeader } from "./RailChrome";
import { TASK_META } from "./taskMeta";
import { Button, Hint, Notice, Tag, type SlotValue } from "./ui";

export interface FormProps {
  shot: Shot;
  ui: ShotUi;
}

/** "Try again with changes": the params this form should start from (read once, then cleared), without the cast entries
 * and images deleted since. */
export function usePrefill(shotId: string, task: TaskId): Record<string, unknown> | null {
  const [p] = useState(() => {
    const s = getState();
    const x = s.prefill[shotId];
    if (!x || x.task !== task) return null;
    const pid = s.shots[shotId]?.project_id ?? s.currentProjectId;
    return withoutGoneRefs(x.params, s.cast[pid]?.map((c) => c.id) ?? null, s.images[pid]?.map((im) => im.id) ?? null);
  });
  useEffect(() => {
    if (p) clearPrefill(shotId);
  }, [p, shotId]);
  return p;
}

/** A made shot opened again with no draft and no "Try again with changes": its newest take of this task's settings
 * (read once), so the form shows what made it (the Look words, references, start frame and seed), not an empty form.
 * Null when there is a prefill or a draft (they win), or no such take. */
export function useLastTake(shot: Shot, task: TaskId, prefill: Record<string, unknown> | null): Record<string, unknown> | null {
  const [p] = useState(() => {
    const s = getState();
    if (prefill || s.drafts[draftKey(shot.id, task)]) return null;
    const v = shot.versions
      .filter((x) => x.task === task && !x.hidden && !isOriginal(x) && x.params)
      .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0))[0];
    if (!v) return null;
    const pid = shot.project_id ?? s.currentProjectId;
    return withoutGoneRefs(v.params as Record<string, unknown>, s.cast[pid]?.map((c) => c.id) ?? null, s.images[pid]?.map((im) => im.id) ?? null);
  });
  return p;
}

/** Where a reopened form's kept references came from: "the example" (Try it), "the preset" (a home preset chip) or
 * "the last take" (Try again). */
export function keptFrom(pre: Record<string, unknown> | null): string {
  return pre?._from === "example" ? "the example" : pre?._from === "preset" ? "the preset" : "the last take";
}

export const str = (v: unknown, d = ""): string => (typeof v === "string" ? v : d);
export const bool = (v: unknown, d: boolean): boolean => (typeof v === "boolean" ? v : d);

/** A take's settings keep references by id: where to see one ("" when there's no picture). */
export function keptUrl(ref: Ref, projectId?: string): string {
  try {
    return getApi().refUrl(ref, projectId) ?? "";
  } catch {
    return "";
  }
}

/** A reference's picture, or a quiet "Missing" when there's none or its file is gone. */
export function RefPicture({ url, alt = "", className }: { url?: string | null; alt?: string; className?: string }) {
  const [failed, setFailed] = useState<string | null>(null);
  if (!url || failed === url) {
    return (
      <span className={cx("flex flex-col items-center justify-center gap-1 bg-raised text-ink3", className)} title="This file isn't in the library any more.">
        <ImageOff size={14} strokeWidth={1.5} aria-hidden="true" />
        <span className="text-[11px]">Missing</span>
      </span>
    );
  }
  return <img src={url} alt={alt} onError={() => setFailed(url)} className={cx("thumb-face", className)} />;
}

// --------------------------------------------------------------------------------------------------------

export function useTask(task: TaskId, shot: Shot, ui: ShotUi) {
  const status = useApp((s) => s.status);
  const tile = status?.tiles?.[task];
  const ready = !!(status?.connected && tile?.ready);
  const [submitting, setSubmitting] = useState(false);
  const baseId = shot.versions.some((v) => v.id === ui.versionId) ? ui.versionId : "orig";
  const about = aboutTime(tile?.eta_s);
  const gpu = status?.machine ? shortGpu(status.machine.gpu) : null;
  /** The one way a form starts its job. `stay`: a step flow keeps the rail and shows the job itself (its Render step);
   * the job's id comes back (null when it didn't start). `draft_key` and `step` go on the job record, beside params. */
  const submit = async (params: JobParams, opts: { stay?: boolean; draft_key?: string; step?: string } = {}): Promise<string | null> => {
    setSubmitting(true);
    const job = await submitJob(
      {
        shot_id: shot.id,
        task,
        params,
        ...(baseId !== "orig" ? { version_id: baseId } : {}),
        ...(opts.draft_key ? { draft_key: opts.draft_key, step: opts.step } : {}),
      },
      !opts.stay,
    );
    if (!job || opts.stay) setSubmitting(false);
    return job?.id ?? null;
  };
  return {
    status,
    ready,
    submitting,
    submit,
    baseId,
    about,
    /** "About 4 min on your RTX 4090." */
    etaLine: about ? `${about}${gpu ? ` on your ${gpu}` : ""}.` : null,
  };
}

export function FormShell({
  task,
  shot,
  ui,
  footer,
  children,
  lengthNote = true,
  need,
  top,
  plainSub,
  title,
}: FormProps & {
  task: TaskId;
  footer: ReactNode;
  children: ReactNode;
  /** The header when the pick names it better than the task (Finish to 4K, Finish to 4K HDR). */
  title?: string;
  /** false: the form says how much of a long shot it uses itself (Recast, under its button). */
  lengthNote?: boolean;
  /** The option picked needs less than the whole task (Finish 4K: SeedVR2 only). */
  need?: { ready: boolean; missing: { count: number; gb: number } };
  /** Pinned under the header, above the scrolling body (a step flow's tabs: they never scroll away). */
  top?: ReactNode;
  /** Just the shot's name under the title (Render and Review: the take they're on is already in the viewer). */
  plainSub?: boolean;
}) {
  const meta = TASK_META[task];
  const task_ = useTask(task, shot, ui);
  const { status, baseId } = task_;
  const ready = need ? !!status?.connected && need.ready : task_.ready;
  const tk = useMemo(() => takes(shot), [shot]);
  const missing = need?.missing ?? missingFor(status, task);
  return (
    <>
      <RailHeader
        title={title ?? meta.verb}
        badge={meta.beta ? <Tag>Labs</Tag> : undefined}
        sub={
          baseId !== "orig" && !plainSub ? (
            <>
              <span className="min-w-0 truncate">{shot.name}</span>
              <span className="inline-flex max-w-full items-center rounded-[3px] border border-line px-1.5 py-px text-ink2" title={`Made from take ${tk[baseId]?.num ?? ""}`}>
                <span className="truncate">
                  On take {tk[baseId]?.num ?? ""} · {tk[baseId]?.label ?? ""}
                </span>
              </span>
            </>
          ) : (
            shot.name
          )
        }
        onBack={() => railBack(shot.id)}
      />
      {top ? <div className="shrink-0 px-6 pb-1 pt-4">{top}</div> : null}
      <RailBody className={cx("space-y-6", !!top && "pt-4")}>
        {lengthNote ? <LongShotNote shot={shot} task={task} /> : null}
        {status && !ready ? (
          !status.connected ? (
            <Notice stack tone="warn" text="Pickups can't reach your ComfyUI." action={{ label: "Connect", onClick: () => openOnboarding("connect") }} />
          ) : (
            <Notice
              stack
              tone="warn"
              text={
                missing.count
                  ? `${meta.verb} needs ${missing.count} ${missing.count === 1 ? "model" : "models"} (${gb(missing.gb)}).`
                  : `${meta.verb} isn't ready on this machine.`
              }
              action={{ label: missing.count ? "Get them" : "Open setup", onClick: () => openSetup(task) }}
            />
          )
        ) : null}
        {children}
      </RailBody>
      <RailFooter>{footer}</RailFooter>
    </>
  );
}

/** Where each task's length limit applies (seconds). Locked-off Remove handles up to 10 s; Finish has none. */
const LENGTH_LIMIT: Partial<Record<TaskId, number>> = { remove: 10, recast: 4, environment: 4, previz: 4, relight: 5, reshoot: 4 };
const LONG_OK_KEY = "pickups.longShotOk";

function longShotOk(id: string): boolean {
  try {
    return (JSON.parse(window.localStorage.getItem(LONG_OK_KEY) || "[]") as string[]).includes(id);
  } catch {
    return false;
  }
}
function rememberLongShotOk(id: string) {
  try {
    const ids = (JSON.parse(window.localStorage.getItem(LONG_OK_KEY) || "[]") as string[]).filter((x) => x !== id);
    window.localStorage.setItem(LONG_OK_KEY, JSON.stringify([id, ...ids].slice(0, 300)));
  } catch {
    /* ignore */
  }
}

/** "Long shot": only in the forms whose model has a length limit, until the user answers or the shot has a take. */
function LongShotNote({ shot, task }: { shot: Shot; task: TaskId }) {
  const limit = LENGTH_LIMIT[task];
  const [ok, setOk] = useState(() => longShotOk(shot.id));
  const hasTakes = shot.versions.some((v) => !isOriginal(v));
  if (!limit || ok || hasTakes || shot.duration_s <= limit + 0.05) return null;
  return (
    <Notice
      stack
      label="Long shot"
      text={`Works best on shots up to ${limit} s. Trim it to the part you want?`}
      action={{ label: "Trim", onClick: () => openPrepare(shot.id, { task }) }}
      secondary={{
        label: "Use anyway",
        onClick: () => {
          rememberLongShotOk(shot.id);
          setOk(true);
        },
      }}
    />
  );
}

/** How much of the shot the video models use: status.max_frames from the start (4 s if the backend doesn't say).
 * `over`: a real part of the shot is left out (at least 0.25 s and 5 % of it), so only the first `label` is worked on.
 * A few frames over (a 4.0 s shot against a 97-frame window) isn't worth a line. `label` is to the nearest 0.5 s. */
export function useLengthCap(shot: Shot): { seconds: number; label: string; over: boolean } {
  const maxFrames = useApp((s) => s.status?.max_frames);
  const fps = shot.fps > 0 ? shot.fps : 24;
  const seconds = maxFrames && maxFrames > 0 ? maxFrames / fps : 4;
  const length = shot.duration_s > 0 ? shot.duration_s : shot.frames > 0 ? shot.frames / fps : 0;
  const left = length - seconds;
  const over = left >= 0.25 && left >= 0.05 * length;
  const half = Math.max(0.5, Math.round(seconds * 2) / 2);
  const label = `${Number.isInteger(half) ? half : half.toFixed(1)} s`;
  return { seconds, label, over };
}

/** The filmstrip bracket. On a shot longer than one render it shows the part rendered; with `resizable` (Remove, Replace a
 * person, Change the weather) it also has in and out handles, so its length is the user's, from about a second up to
 * status.max_frames (what the GPU renders without quality loss), and the form sends it as `frames`. Recast waits until
 * the person is found (`waitForAnchor`); the other tasks show it from the first frame, or centred on `anchor`, until it
 * is dragged. */
export function useShotWindow(shot: Shot, initial: number | null, anchor: number | null, fixed: boolean, waitForAnchor: boolean, resizable = false, task?: string) {
  const cap = useLengthCap(shot);
  const maxFrames = useApp((s) => (task ? s.status?.max_frames_by_task?.[task] : undefined) ?? s.status?.max_frames);
  const total = Math.max(1, shot.frames);
  const fps = shot.fps > 0 ? shot.fps : 24;
  const maxLen = ltxWindowFrames(maxFrames && maxFrames > 0 ? maxFrames : Math.round(4 * fps) + 1, total);
  const minLen = Math.min(maxLen, minWindowFrames(fps, total));
  const [len, setLen] = useState<number | null>(null);
  const winLen = resizable && len != null ? Math.max(minLen, Math.min(maxLen, len)) : maxLen;
  const show = resizable ? total > minLen : cap.over && total > winLen;
  const [held, setHeld] = useState<number | null>(initial);
  const win = useApp((s) => (s.renderWindow?.shotId === shot.id ? s.renderWindow : null));
  useEffect(() => {
    if (win?.moved && !win.fixed && win.start !== held) setHeld(win.start);
    if (resizable && win?.sized && win.frames !== len) setLen(win.frames);
  }, [win?.moved, win?.start, win?.sized, win?.frames]);
  useEffect(() => {
    const sized = resizable && len != null;
    if (!show) setRenderWindow(null);
    else if (fixed) setRenderWindow({ shotId: shot.id, start: 0, frames: winLen, moved: false, fixed: true });
    else if (held != null) {
      let start = Math.min(Math.max(0, held), Math.max(0, total - winLen));
      // Remove and weather track from the frame you clicked: the bracket can't be dragged off it.
      if (anchor != null && !waitForAnchor) start = Math.min(Math.max(start, anchor - winLen + 1, 0), Math.min(anchor, Math.max(0, total - winLen)));
      setRenderWindow({ shotId: shot.id, start, frames: winLen, moved: true, fixed: false, sized });
    }
    else if (anchor != null) setRenderWindow({ shotId: shot.id, start: windowStart(anchor - (winLen - 1) / 2, winLen, total), frames: winLen, moved: false, fixed: false, sized });
    else if (!waitForAnchor) setRenderWindow({ shotId: shot.id, start: 0, frames: winLen, moved: false, fixed: false, sized });
    else setRenderWindow(null);
  }, [show, fixed, winLen, total, anchor, held, shot.id, waitForAnchor, len, resizable]);
  useEffect(() => () => setRenderWindow(null), []);
  return { show, win, label: win && resizable ? halfSeconds(win.frames, fps) : cap.label, maxLabel: `Up to ${halfSeconds(maxLen, fps)} on your GPU` };
}

/** "More options": what most people never need, closed until asked for. */
export function MoreOptions({ children, title = "More options" }: { children: ReactNode; title?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-line pt-4">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-[14px] text-ink2 transition-colors hover:text-ivory"
      >
        {title}
        <ChevronDown size={15} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open ? <div className="mt-4 space-y-5 animate-fade-in">{children}</div> : null}
    </div>
  );
}

/** One quiet line that opens to show more (e.g. "Prompt" → the prompt receipt). */
export function Fold({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="inline-flex items-center gap-1 text-[13px] text-ink2 transition-colors hover:text-ivory">
        {label}
        <ChevronDown size={13} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open ? <div className="animate-fade-in">{children}</div> : null}
    </div>
  );
}

export function PrimaryAction({ children, disabled, loading, onClick, hint }: { children: string; disabled?: boolean; loading?: boolean; onClick: () => void; hint?: string | null }) {
  return (
    <>
      {hint ? <p className="text-[13px] leading-snug text-ink3">{hint}</p> : null}
      <Button variant="primary" size="lg" block disabled={disabled} loading={loading} onClick={onClick}>
        {children}
      </Button>
    </>
  );
}

export function useUploadSlot() {
  const [value, setValue] = useState<SlotValue | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const onFile = async (f: File) => {
    setBusy(true);
    setError(null);
    setProgress(0);
    try {
      const res = await getApi().upload(f, setProgress);
      setValue({ id: res.id, url: res.url, name: f.name });
    } catch (e) {
      setError(e instanceof ApiError && e.message ? e.message : "That image didn't upload. Try again.");
    } finally {
      setBusy(false);
    }
  };
  // clear(): the slot is empty again, with no error from an earlier file either.
  const clear = () => {
    setValue(null);
    setError(null);
    setProgress(0);
  };
  return { value, busy, progress, error, onFile: (f: File) => void onFile(f), clear, set: setValue };
}

/** A labelled field. With a `hint`, the label gets a "?" and the field is labelled by id (a <label> can't hold a button). */
export function Field({ label, hint, children }: { label?: string; hint?: ReactNode; children: ReactNode }) {
  const id = useId();
  if (hint && label) {
    return (
      <div className="block" role="group" aria-labelledby={id}>
        <span className="mb-1.5 flex items-center gap-1.5 text-[13px] font-medium text-ink2">
          <span id={id}>{label}</span>
          <Hint text={hint} label={`About ${label.toLowerCase()}`} />
        </span>
        {children}
      </div>
    );
  }
  return (
    <label className="block">
      {label ? <span className="mb-1.5 block text-[13px] font-medium text-ink2">{label}</span> : null}
      {children}
    </label>
  );
}

export function Chip({ children, onClick, active }: { children: ReactNode; onClick: () => void; active?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        "inline-flex h-7 items-center rounded-full border px-3 text-[12px] transition-colors duration-150 ease-out",
        active ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

