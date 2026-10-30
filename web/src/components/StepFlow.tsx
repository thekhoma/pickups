// Step flows (SPEC section 1): a task as numbered steps inside the existing FormShell. The task's own steps (Who, Light,
// Finish), then a shared Render (the job's real stages) and a shared Review (compare, Try again, Note, Approve). One
// body, one Advanced fold per step, one pinned primary. State lives in the task's draft (lib/drafts.ts), so leaving
// and coming back (or reloading) returns to the same step. useTask.submit stays the only way a job starts.
import { Check, ChevronDown } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { loadJobs, setApproved, setRail, setUi, tryAgainWithChanges } from "../actions";
import type { useDraft } from "../lib/drafts";
import { finishLabel } from "../lib/finish";
import { relightLabel } from "../lib/flowParams";
import { cx } from "../lib/format";
import { usePlayer } from "../lib/player";
import { getState, useApp, type ShotUi } from "../store";
import type { Job, JobParams, Shot, TaskId, Version } from "../types";
import { FormShell, PrimaryAction, useTask } from "./FormKit";
import { startNote } from "./Notes";
import { RunStrip } from "./RunStatus";
import { Button, Notice, ProgressLine, Spinner } from "./ui";

type Draft<D extends Record<string, unknown>> = ReturnType<typeof useDraft<D>>;

/** What a step's body can do to the flow: go on as its primary would without its `run` (a viewer's "Use <name>",
 * Advanced's "Use without a sheet"). On the last task step that starts the job; `canNext` is false while it can't. */
export interface FlowNav {
  next(): void;
  canNext: boolean;
}

/** One of the task's own steps. Its body is one visual choice; anything for pros goes in `advanced`. */
export interface FlowStep<D> {
  id: string;
  label: string;
  /** The row a done step folds into: a thumb, a few words, and Edit (or Change). */
  summary(d: D): { thumb?: string | null; text: string; edit?: "Edit" | "Change" } | null;
  body(d: D, set: (p: Partial<D>) => void, flow: FlowNav): ReactNode;
  /** The step's one button. By default it goes on (the last task step starts the job); `run` does the step's own
   * work instead (Make sheet), and the flow stays on the step. */
  primary(d: D): { label: string; disabled?: boolean; run?: () => void | Promise<void> };
  advanced?(d: D, set: (p: Partial<D>) => void, flow: FlowNav): ReactNode;
  /** Goes on by itself when this returns a key (once per key): Replace a person's sheet is ready, so it renders. */
  autoNext?(d: D): string | null;
}

const RENDER = "render";
const REVIEW = "review";

export function StepFlow<D extends Record<string, unknown>>({
  task,
  shot,
  ui,
  draft,
  steps,
  params,
  need,
  lengthNote,
  footerNote,
  reviewExtra,
  title,
}: {
  /** The header, when the picks name the job (Finish: "Finish to 4K"). */
  title?: string;
  task: TaskId;
  shot: Shot;
  ui: ShotUi;
  draft: Draft<D>;
  steps: FlowStep<D>[];
  /** The job's params from the steps' picks (lib/flowParams.ts: the same as the old form's for the same picks). */
  params(d: D): JobParams;
  need?: { ready: boolean; missing: { count: number; gb: number } };
  lengthNote?: boolean;
  /** A line under the primary on the task's own steps (Remove's bracket). */
  footerNote?: ReactNode;
  /** What Review shows for the take besides the summary (Finish: its downloads). */
  reviewExtra?: (v: Version) => ReactNode;
}) {
  const t = useTask(task, shot, ui);
  const player = usePlayer();
  const { data, set } = draft;
  const all = [...steps.map((s) => ({ id: s.id, label: s.label })), { id: RENDER, label: "Render" }, { id: REVIEW, label: "Review" }];
  const stepId = all.some((s) => s.id === draft.draft.step) ? draft.draft.step : steps[0].id;
  const at = all.findIndex((s) => s.id === stepId);
  const jobId = draft.draft.jobs?.[RENDER];
  const job = useApp((s) => (jobId ? s.jobs[jobId] : undefined));
  const takeId = typeof data._take === "string" ? data._take : null;
  const take = takeId ? shot.versions.find((v) => v.id === takeId) ?? null : null;
  const base = typeof data._base === "string" ? data._base : "orig";
  const [advanced, setAdvanced] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = (id: string) => draft.patch({ step: id });

  // Back on a task step from Render or Review: the viewer shows the base again, so a new run starts from it.
  const editStep = (id: string) => {
    if (stepId === RENDER || stepId === REVIEW) setUi(shot.id, { versionId: base });
    go(id);
  };

  // Review shows the take against what it was made from, the take on the right.
  const showTake = (id: string) => setUi(shot.id, { versionId: id, beforeId: base, wipe: true, view: "after" });

  // The job finished: on to Review with its take. Failed or canceled: Render says so.
  useEffect(() => {
    if (stepId === RENDER && job?.status === "done" && job.result_version_id) {
      draft.patch((d) => ({ step: REVIEW, data: { ...d.data, _take: job.result_version_id } }));
      showTake(job.result_version_id);
    }
  }, [stepId, job?.status, job?.result_version_id]);

  // Reopened on Review: the viewer shows the take again.
  useEffect(() => {
    if (stepId === REVIEW && take && ui.versionId !== take.id) showTake(take.id);
  }, [stepId, take?.id]);

  // A job id the server doesn't know after asking it (it restarted): gone, never busy forever. Back to the first step
  // with the picks kept; a take it already made still opens Review.
  useEffect(() => {
    if (stepId !== RENDER || !jobId || job) return;
    let live = true;
    void loadJobs().then(() => {
      if (live && !getState().jobs[jobId]) draft.patch((d) => ({ step: takeId ? REVIEW : steps[0].id, jobs: { ...d.jobs, [RENDER]: "" } }));
    });
    return () => {
      live = false;
    };
  }, [stepId, jobId, !!job]);

  const submit = async () => {
    setBusy(true);
    const p = params(data);
    const id = await t.submit(p, { stay: true, draft_key: draft.key, step: RENDER });
    setBusy(false);
    if (id) draft.patch((d) => ({ step: RENDER, jobs: { ...d.jobs, [RENDER]: id }, data: { ...d.data, _base: t.baseId, _take: null, _sent: JSON.stringify(p) } }));
  };

  // Changing a task step's picks after a run makes the later steps stale: their tabs wait for the next run.
  const sent = typeof data._sent === "string" ? data._sent : null;
  const stale = at < steps.length && !!sent && sent !== JSON.stringify(params(data));
  const reachable = (i: number) =>
    i < at || (!stale && ((all[i].id === RENDER && !!jobId) || (all[i].id === REVIEW && !!take)));

  // Review's Approve is the rail's: the player bar hides its own for this take while Review shows it (⇧A still works).
  const reviewTake = stepId === REVIEW && take ? take.id : null;
  useEffect(() => {
    setUi(shot.id, { railApprove: reviewTake });
    return () => {
      if (getState().ui[shot.id]?.railApprove === reviewTake) setUi(shot.id, { railApprove: null });
    };
  }, [shot.id, reviewTake]);

  const ready = need ? !!t.status?.connected && need.ready : t.ready;
  const lastTask = at === steps.length - 1;
  const flow: FlowNav = {
    canNext: at < steps.length && !busy && !t.submitting && (!lastTask || ready),
    next: () => {
      if (at >= steps.length || busy || t.submitting) return;
      if (!lastTask) go(steps[at + 1].id);
      else if (ready) void submit();
    },
  };

  // A step that goes on by itself (its work landed while the user waited on it): once per key, only while it's shown.
  const autoKey = at < steps.length ? steps[at].autoNext?.(data) ?? null : null;
  const autoDone = useRef<string | null>(null);
  useEffect(() => {
    if (!autoKey || autoDone.current === autoKey || !flow.canNext) return;
    autoDone.current = autoKey;
    flow.next();
  }, [autoKey, flow.canNext]);

  // --- the pinned primary ---
  let primary: ReactNode;
  if (at < steps.length) {
    const s = steps[at];
    const p = s.primary(data);
    const last = lastTask;
    primary = (
      <>
        <PrimaryAction
          disabled={!!p.disabled || (last && !p.run && !ready)}
          loading={busy || t.submitting}
          onClick={async () => {
            if (p.run) {
              setBusy(true);
              try {
                await p.run();
              } finally {
                setBusy(false);
              }
            } else if (last) void submit();
            else go(steps[at + 1].id);
          }}
        >
          {p.label}
        </PrimaryAction>
        {footerNote}
      </>
    );
  } else if (stepId === RENDER) {
    primary = job?.status === "failed" ? (
      <PrimaryAction loading={busy} onClick={() => void submit()}>
        Try again
      </PrimaryAction>
    ) : (
      <PrimaryAction onClick={() => setRail(shot.id, { kind: "tiles" })}>Back to the shot</PrimaryAction>
    );
  } else {
    primary = (
      <PrimaryAction
        disabled={!take || take.status === "approved"}
        hint={take?.status === "approved" ? null : "Approve puts this take in the project's Deliveries."}
        loading={busy}
        onClick={async () => {
          if (!take) return;
          setBusy(true);
          await setApproved(shot.id, take.id, true);
          setBusy(false);
          draft.clear();
          setRail(shot.id, { kind: "result", versionId: take.id });
        }}
      >
        {take?.status === "approved" ? "Approved" : "Approve"}
      </PrimaryAction>
    );
  }

  const summaries = (upTo: number) =>
    steps.slice(0, upTo).map((s) => {
      const sum = s.summary(data);
      return sum ? <SummaryRow key={s.id} thumb={sum.thumb} text={sum.text} edit={sum.edit ?? "Edit"} onEdit={() => editStep(s.id)} /> : null;
    });

  let body: ReactNode;
  if (at < steps.length) {
    const s = steps[at];
    body = (
      <>
        {summaries(at)}
        {s.body(data, set, flow)}
        {s.advanced ? (
          <Advanced open={advanced === s.id} onToggle={() => setAdvanced((o) => (o === s.id ? null : s.id))}>
            {s.advanced(data, set, flow)}
          </Advanced>
        ) : null}
      </>
    );
  } else if (stepId === RENDER) {
    body = (
      <>
        {summaries(steps.length)}
        {job ? <RenderProgress job={job} /> : <Spinner size={14} className="text-ivory" />}
      </>
    );
  } else {
    body = (
      <>
        {summaries(steps.length)}
        {take && reviewExtra ? reviewExtra(take) : null}
        <div className="grid grid-cols-2 gap-2">
          <Button variant="secondary" size="md" disabled={!take} onClick={() => take && tryAgainWithChanges(shot.id, take)}>
            Try again
          </Button>
          <Button variant="secondary" size="md" disabled={!take} onClick={() => take && startNote(player, shot.id, take.id)}>
            Note
          </Button>
        </div>
      </>
    );
  }

  return (
    <FormShell
      task={task}
      shot={shot}
      ui={ui}
      title={title}
      footer={primary}
      need={need}
      lengthNote={(lengthNote ?? true) && at < steps.length}
      plainSub={at >= steps.length}
      top={<StepTabs steps={all} at={at} stale={stale} reachable={reachable} onGo={(i) => (i < steps.length ? editStep(all[i].id) : go(all[i].id))} />}
    >
      {/* Room under the last row: it scrolls clear of the pinned primary. */}
      <div className="space-y-4 pb-8">{body}</div>
    </FormShell>
  );
}

/** 1440: numbered tabs (done ones clickable, later ones shown but inert). 390: one line, "2 of 3 · Render". */
function StepTabs({ steps, at, stale, reachable, onGo }: { steps: { id: string; label: string }[]; at: number; stale: boolean; reachable: (i: number) => boolean; onGo: (i: number) => void }) {
  return (
    <nav aria-label="Steps">
      <p className="text-[13px] text-ink md:hidden" aria-current="step">
        {at + 1} of {steps.length} · {steps[at].label}
      </p>
      <ol className="flex gap-1.5 max-md:hidden">
        {steps.map((s, i) => {
          const on = i === at;
          const can = !on && reachable(i);
          return (
            <li key={s.id} className="min-w-0 flex-1">
              <button
                type="button"
                disabled={!can}
                aria-current={on ? "step" : undefined}
                onClick={() => onGo(i)}
                className={cx(
                  "block w-full border-t-2 pt-2 text-left text-[13px] transition-colors duration-150",
                  on ? "border-ivory text-ivory" : can ? "border-line2 text-ink2 hover:text-ivory" : "border-line text-ink4",
                  stale && i > at && "opacity-60",
                )}
              >
                <span className="truncate">
                  {i + 1} {s.label}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export function SummaryRow({ thumb, text, edit, onEdit }: { thumb?: string | null; text: string; edit: "Edit" | "Change"; onEdit: () => void }) {
  return (
    <div className="flex min-h-[52px] items-center gap-3 rounded-card border border-line px-2.5 py-2">
      {thumb ? <img src={thumb} alt="" className="thumb-face h-9 w-12 shrink-0 rounded-[3px]" /> : null}
      <span className="min-w-0 flex-1 truncate text-[13px] text-ink2">{text}</span>
      <button type="button" onClick={onEdit} className="shrink-0 rounded-full px-2.5 py-1 text-[12px] text-ink2 transition-colors hover:bg-line hover:text-ivory">
        {edit}
      </button>
    </div>
  );
}

/** One fold per step, closed by default. Opening it changes nothing; only its controls do (and it scrolls into view). */
function Advanced({ open, onToggle, children }: { open: boolean; onToggle: () => void; children: ReactNode }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open) box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [open]);
  return (
    <div ref={box} className="scroll-mb-6">
      <button type="button" aria-expanded={open} onClick={onToggle} className="flex w-full items-center justify-between py-1 text-[13px] text-ink3 transition-colors hover:text-ivory">
        Advanced
        <ChevronDown size={14} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open ? <div className="mt-3 space-y-4 animate-fade-in">{children}</div> : null}
    </div>
  );
}

/** What the run is doing, in words: "Relighting to Sunset backlight", "Finishing to 4K HDR". */
function renderLabel(job: Job): string {
  if (job.task === "relight") return `Relighting to ${relightLabel(job.params as { look?: unknown; custom?: unknown })}`;
  if (job.task === "finish") return finishLabel(job.params as { upscale_4k?: unknown; hdr?: unknown }).replace(/^Finish/, "Finishing to");
  return "Rendering";
}

/** Render: the shared RunStrip (time since it started, "Usually about N", Cancel), then the job's own stages. Leaving
 * doesn't stop it. */
function RenderProgress({ job }: { job: Job }) {
  const lastStage = job.stages.includes(job.stage) ? job.stage : job._lastStage;
  const idx = job.status === "done" ? job.stages.length : lastStage ? job.stages.indexOf(lastStage) : -1;
  if (job.status === "failed") {
    return <Notice stack tone="bad" label="Failed" text={job.error_detail && job.error ? job.error : lastStage ? `Something went wrong while ${lastStage.charAt(0).toLowerCase()}${lastStage.slice(1)}.` : "Something went wrong."} />;
  }
  if (job.status === "canceled") return <Notice stack label="Cancelled" text="Nothing was kept." />;
  return (
    <div className="space-y-3" aria-live="polite">
      {job.status === "done" ? <ProgressLine value={1} /> : <RunStrip job={job} label={job.stalled ? "Stalled" : renderLabel(job)} />}
      <ol aria-label="Stages">
        {job.stages.map((label, i) => {
          const state = i < idx ? "done" : i === idx ? "now" : "todo";
          return (
            <li key={label} className="flex items-center gap-3 border-b border-line py-2 last:border-b-0">
              <span
                className={cx(
                  "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                  state === "done" ? "border-ink3 bg-ink3 text-[#0D0C0B]" : state === "now" ? "border-ivory" : "border-line2",
                )}
              >
                {state === "done" ? <Check size={10} strokeWidth={3} /> : state === "now" && job.status === "running" ? <Spinner size={10} className="text-ivory" /> : null}
              </span>
              <span className={cx("min-w-0 flex-1 text-[13px]", state === "now" ? "text-ivory" : state === "done" ? "text-ink2" : "text-ink4")}>{label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
