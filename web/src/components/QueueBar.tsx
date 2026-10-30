import { ChevronDown, X } from "lucide-react";
import { finishLabel } from "../lib/finish";
import { useEffect, useMemo, useRef, useState } from "react";
import { cancelJob, dismissJob, moveJobNext, navigate, openShot, setCurrentProject, sheetName } from "../actions";
import { cx, isActive, jobResultVersionId, TASK_SHORT } from "../lib/format";
import { elapsedClock, useElapsed } from "../lib/runtime";
import { setState, useApp } from "../store";
import type { Job, TaskId } from "../types";
import { Button, IconButton, MENU_ITEM, ProgressLine, StateDot, TallyDot, useDismiss } from "./ui";

/** Open what a job made: its shot, the Image page for image jobs (in their project, which becomes current), or the
 * project's Cast for a character sheet. */
function openJob(job: Job) {
  setState({ queueOpen: false });
  if (job.task === "image") {
    if (job.project_id) setCurrentProject(job.project_id, true);
    navigate({ name: "create", panel: "image" });
    return;
  }
  if (job.task === "video") {
    // A new video is a shot of its own once done; until then, Create (where it was asked for).
    if (job.status === "done" && job.result_shot_id) openShot(job.result_shot_id);
    else navigate({ name: "create", panel: "home" });
    return;
  }
  if (job.task === "sheet") {
    // A sheet with no project still has a home: Assets lists every picture made.
    navigate(job.project_id ? { name: "project", id: job.project_id, tab: "cast" } : { name: "assets" });
    return;
  }
  if (job.lane === "inline") {
    // The start frame or look frame: back to the form that asked for it (its draft has the picture).
    openShot(job.shot_id, { rail: { kind: "form", task: job.task as TaskId } });
    return;
  }
  const vid = jobResultVersionId(job);
  if (job.status === "done" && vid) openShot(job.shot_id, { versionId: vid });
  else openShot(job.shot_id, { rail: { kind: "running", jobId: job.id } });
}

/** "REMOVE · pier_C115 · Harbour spot" / "IMAGE · FLUX 3 klein · Harbour spot". */
function useJobLabel(job: Job): { what: string; name: string; project: string | null } {
  const shot = useApp((s) => (job.shot_id ? s.shots[job.shot_id] : undefined));
  const pid = job.project_id ?? shot?.project_id;
  const projectName = useApp((s) => (pid ? s.projects?.find((p) => p.id === pid)?.name : undefined));
  const modelId = job.model ?? (job.params as { model?: string } | undefined)?.model ?? null;
  const modelName = useApp((s) => (modelId ? s.imageModels?.models.find((m) => m.id === modelId)?.name : undefined));
  if (job.task === "image") return { what: "Image", name: modelName ?? modelId ?? "image", project: job.project_name ?? projectName ?? null };
  if (job.task === "sheet") return { what: "Sheet", name: sheetName(job), project: job.project_name ?? projectName ?? null };
  if (job.task === "video") {
    const p = job.params as { mode?: string; resolution?: string } | undefined;
    return { what: p?.mode === "image" ? "Image to video" : "Text to video", name: `LTX-2.5${p?.resolution ? ` · ${p.resolution}` : ""}`, project: job.project_name ?? projectName ?? null };
  }
  // An inline row says its step: "Start frame · shot".
  if (job.lane === "inline") return { what: job.stages[0] ?? job.stage, name: job.shot_name ?? shot?.name ?? "shot", project: job.project_name ?? projectName ?? null };
  const what = job.task === "finish" ? finishLabel(job.params as { upscale_4k?: unknown; hdr?: unknown }) : TASK_SHORT[job.task];
  return { what, name: job.shot_name ?? shot?.name ?? "shot", project: job.project_name ?? projectName ?? null };
}

function useQueue() {
  const jobs = useApp((s) => s.jobs);
  const order = useApp((s) => s.jobOrder);
  const dismissed = useApp((s) => s.dismissed);
  return useMemo(() => {
    const list = order.map((id) => jobs[id]).filter((j): j is Job => !!j && (isActive(j) || !dismissed[j.id]));
    const rank = (j: Job) => (j.status === "running" ? 0 : j.status === "queued" ? 1 : 2);
    // Running first, then queued oldest-first, then finished newest-first.
    return list
      .map((j, i) => ({ j, i }))
      .sort((a, b) => rank(a.j) - rank(b.j) || (rank(a.j) === 1 ? (b.j.next_at ?? 0) - (a.j.next_at ?? 0) || b.i - a.i : a.i - b.i))
      .map((x) => x.j);
  }, [jobs, order, dismissed]);
}

function statusText(job: Job, elapsed: number | null) {
  if (job.status === "running" && job.stalled) return "Stalled";
  if (job.status === "running") return elapsedClock(elapsed);
  if (job.status === "queued") return "Queued";
  if (job.status === "done") return "Done";
  if (job.status === "failed") return "Error";
  return "Cancelled";
}

function JobDot({ job }: { job: Job }) {
  if (job.status === "running") return <TallyDot size={6} />;
  return <StateDot on={job.status === "done"} size={6} />;
}

function QueueRow({ job, ahead }: { job: Job; ahead: number }) {
  const label = useJobLabel(job);
  // How long it has been running (measured), never an estimate of what's left.
  const elapsed = useElapsed(job);
  const [stopping, setStopping] = useState(false);
  const status = job.status === "queued" ? (ahead === 0 ? "Next" : `Queued · ${ahead} ahead`) : statusText(job, elapsed);
  // The next one up is yellow: it starts as soon as the GPU is free.
  const next = job.status === "queued" && ahead === 0;
  // Right-click a waiting run: Move next (and Cancel). The same Move next sits on the row on hover, for a mouse
  // without a right button and for the keyboard.
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const canMove = job.status === "queued" && job.lane !== "inline" && ahead > 0;
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [menu]);
  return (
    <li
      className={cx("group/row relative border-b border-line px-4 py-3 last:border-b-0", next && "border-l-2 border-l-[#E3A34B] bg-[#2A2112]")}
      data-next={next || undefined}
      onContextMenu={(e) => {
        if (job.status !== "queued") return;
        e.preventDefault();
        const r = e.currentTarget.getBoundingClientRect();
        setMenu({ x: e.clientX - r.left, y: e.clientY - r.top });
      }}
    >
      {menu ? (
        <div
          role="menu"
          aria-label="Queue item"
          className="absolute z-40 w-[180px] overflow-hidden rounded-card border border-line2 bg-raised py-1 shadow-[0_24px_60px_-24px_rgba(0,0,0,0.9)]"
          style={{ left: Math.min(menu.x, 220), top: menu.y }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <button type="button" role="menuitem" disabled={!canMove} className={cx(MENU_ITEM, !canMove && "text-ink4")} onClick={() => (setMenu(null), void moveJobNext(job.id))}>
            {canMove ? "Move next" : "Already next"}
          </button>
          <button type="button" role="menuitem" className={MENU_ITEM} onClick={() => (setMenu(null), void cancelJob(job.id))}>
            Cancel
          </button>
        </div>
      ) : null}
      <div className="flex items-center gap-3">
        <JobDot job={job} />
        <button type="button" onClick={() => openJob(job)} className="min-w-0 flex-1 text-left">
          <span className="block truncate text-[14px] text-ink hover:text-ivory">
            {label.what}
            <span className="text-ink4"> · </span>
            <span className="font-mono text-[13px]">{label.name}</span>
            {label.project ? <span className="text-ink3"> · {label.project}</span> : null}
          </span>
          {job.status === "running" && job.stalled ? (
            <span className="mt-0.5 block text-[12px] leading-snug text-ink2">
              <span className="slate mr-2 text-ivory/70">Stalled</span>
              {job.stall_message || "No progress from ComfyUI for a while."}
            </span>
          ) : job.status === "failed" && job.error ? (
            <span className="mt-0.5 block text-[12px] leading-snug text-ink2">
              <span className="slate mr-2 text-ivory/70">Error</span>
              {job.error.split("\n")[0]}
            </span>
          ) : (
            <span className={cx("slate block truncate", next ? "text-[#E3A34B]" : "text-ink3")}>
              {job.status === "running" ? `${job.stage} · ${elapsedClock(elapsed)}` : next ? "Next up" : status}
            </span>
          )}
        </button>
        {canMove ? (
          <Button variant="ghost" size="sm" className="opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100" onClick={() => void moveJobNext(job.id)}>
            Move next
          </Button>
        ) : null}
        {isActive(job) ? (
          // One press, as everywhere a job runs (RunStrip): it stops this job's own prompt and keeps nothing.
          <Button
            variant="ghost"
            size="sm"
            loading={stopping || job.stage === "Canceling"}
            onClick={async () => {
              setStopping(true);
              await cancelJob(job.id);
              setStopping(false);
            }}
          >
            Cancel
          </Button>
        ) : (
          <IconButton icon={X} label="Dismiss" size="sm" onClick={() => dismissJob(job.id)} />
        )}
      </div>
      {job.status === "running" ? <ProgressLine value={job.progress} idle={job.lane === "inline"} className="mt-2.5" /> : null}
    </li>
  );
}

/** Top right of every page: what the GPU is doing, and the queue behind it, in a drop-down. */
export function QueueMenu() {
  const items = useQueue();
  const open = useApp((s) => s.queueOpen);
  const ref = useRef<HTMLDivElement>(null);
  useDismiss(open, () => setState({ queueOpen: false }), ref);
  const running = items.filter((j) => j.status === "running");
  const queued = items.filter((j) => j.status === "queued");
  const finished = items.filter((j) => !isActive(j));
  // The GPU's job leads the button; an inline row alone (a start frame) shows its step.
  const live = running.find((j) => j.lane !== "inline") ?? running[0];
  const elapsed = useElapsed(live);
  const label = live ? (live.stalled ? "Stalled" : elapsedClock(elapsed)) : queued.length ? "Queued" : "Queue";

  return (
    <div ref={ref} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setState({ queueOpen: !open })}
        aria-expanded={open}
        aria-label={`Queue: ${running.length} running, ${queued.length} waiting`}
        className={cx(
          "flex h-9 items-center gap-2 whitespace-nowrap rounded-full border px-3.5 text-[13px] font-medium transition-colors duration-150 hover:text-ivory",
          live || queued.length ? "border-line2 text-ivory" : "border-line text-ink2 hover:border-line2",
        )}
      >
        {live ? <TallyDot size={7} /> : <StateDot on={false} size={7} />}
        <span>{live ? `Queue · ${label}` : "Queue"}</span>
        {queued.length ? <span className="rounded-full bg-[#E3A34B] px-1.5 text-[12px] font-semibold text-[#0D0C0B]">+{queued.length}</span> : null}
        <ChevronDown size={13} className={cx("transition-transform", open && "rotate-180")} aria-hidden="true" />
      </button>
      {live ? <ProgressLine value={live.progress} idle={live.lane === "inline"} className="absolute inset-x-3 -bottom-[3px]" /> : null}
      {open ? (
        <div className="absolute right-0 top-full z-30 mt-2 w-[420px] max-w-[calc(100vw-24px)] animate-rise-in overflow-hidden rounded-card border border-line2 bg-panel max-[640px]:fixed max-[640px]:inset-x-3 max-[640px]:top-14 max-[640px]:mt-0 max-[640px]:w-auto">
          <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
            <span className="slate text-ink3">
              {running.length} running · {queued.length} waiting
            </span>
            <span className="flex items-center gap-1">
              {queued.length > 1 ? (
                <Button variant="ghost" size="sm" onClick={() => queued.forEach((j) => void cancelJob(j.id))}>
                  Cancel all waiting
                </Button>
              ) : null}
              {finished.length ? (
                <Button variant="ghost" size="sm" onClick={() => finished.forEach((j) => dismissJob(j.id))}>
                  Clear finished
                </Button>
              ) : null}
            </span>
          </div>
          {items.length ? (
            <ul className="max-h-[60vh] overflow-y-auto">
              {items.map((j) => (
                <QueueRow key={j.id} job={j} ahead={queued.indexOf(j)} />
              ))}
            </ul>
          ) : (
            <p className="px-4 py-5 text-[13px] text-ink3">Nothing rendering. Start as many runs as you like; they go one at a time, in order.</p>
          )}
        </div>
      ) : null}
    </div>
  );
}
