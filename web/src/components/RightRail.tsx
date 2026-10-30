import { ArrowRight, Check, ChevronDown, Crop, FolderOpen, RotateCcw, SlidersHorizontal, UserRound } from "lucide-react";
import { VersionStrip } from "./Transport";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { activeJob, cancelJob, isLocalHost, jobForVersion, loadJobs, openJob, openPrepare, openOnboarding, openSetup, retry, revealTake, selectVersion, setRail, setUi, tryAgainWithChanges } from "../actions";
import { choiceOf, finishTitle } from "../lib/finish";
import { queueWords, useQueuePlace } from "../lib/hooks";
import { elapsedClock, madeIn, useElapsed, usually } from "../lib/runtime";
import {
  aiShare,
  clock,
  cx,
  duration,
  tookText,
  finishSentence,
  isActive,
  isOriginal,
  nonCommercialHint,
  nonCommercialModels,
  pillState,
  shortGpu,
  takeLength,
  takes,
  TASK_VERB,
  versionTook,
  visibleTasks,
} from "../lib/format";
import { useApp, type ShotUi } from "../store";
import type { Job, ModelLicence, RecastParams, Shot, Status, TaskId, Version } from "../types";
import { DownloadControl } from "./DownloadControl";
import { keptUrl, RefPicture } from "./FormKit";
import { foundPersonThumb, TaskForm } from "./Forms";
import { NotesList } from "./Notes";
import { ComfyWorkflows, ShareControl } from "./TakeExtras";
import { RailBody, RailFooter, RailHeader, RecipeStrip } from "./RailChrome";
import { PrepareRail } from "./Prepare";
import { NEXT_STEPS, TASK_META } from "./taskMeta";
import { ModelLine, useWorkflowModels } from "./ModelChips";
import { Button, Notice, ProgressLine, ReceiptBar, Skeleton, Slate, Tag, TallyDot, useDismiss } from "./ui";

export function RightRail({ shot, ui }: { shot: Shot; ui: ShotUi }) {
  const rail = ui.rail;
  let content: ReactNode;
  if (rail.kind === "form") content = <TaskForm key={`${rail.task}`} task={rail.task} shot={shot} ui={ui} />;
  else if (rail.kind === "running") content = <Running key={rail.jobId} shot={shot} jobId={rail.jobId} />;
  else if (rail.kind === "result") content = <Result key={rail.versionId} shot={shot} versionId={rail.versionId} />;
  else if (rail.kind === "prepare") content = <PrepareRail shot={shot} />;
  else content = <Tiles shot={shot} ui={ui} />;
  return (
    <aside
      className="flex w-[392px] shrink-0 flex-col border-l border-line bg-panel max-[1180px]:w-[348px] max-[767px]:w-full max-[767px]:border-l-0 max-[767px]:border-t"
      aria-label="Tasks"
    >
      <RecipeStrip shotId={shot.id} />
      <RailVersions shot={shot} ui={ui} />
      {content}
    </aside>
  );
}

/** Every version of the shot (Original, Previz, Snow…), at the top of the panel: click one to see it. */
function RailVersions({ shot, ui }: { shot: Shot; ui: ShotUi }) {
  const busy = useApp((s) => s.jobOrder.some((id) => s.jobs[id]?.shot_id === shot.id && isActive(s.jobs[id])));
  if (shot.versions.length < 2 && !busy) return null;
  return (
    <div className="max-[767px]:hidden" data-testid="rail-versions">
      <div className="slate px-4 pb-1 pt-3 text-ink3">Versions</div>
      <VersionStrip shot={shot} ui={ui} vertical />
    </div>
  );
}

/** Stills and loops that ship with the app, for a task with no example on this machine. */
const TASK_MEDIA: Partial<Record<TaskId, { poster: string; loop: string }>> = {
  remove: { poster: "/mock/trail_remove.jpg", loop: "/mock/trail_remove.mp4" },
  recast: { poster: "/mock/glen_recast.jpg", loop: "/mock/glen_recast.mp4" },
  environment: { poster: "/mock/castle_dusk_environment.jpg", loop: "/mock/castle_dusk_environment.mp4" },
  previz: { poster: "/mock/hood_push_previz.jpg", loop: "/mock/hood_push_previz.mp4" },
  relight: { poster: "/mock/harbour_relight.jpg", loop: "/mock/harbour_relight.mp4" },
  finish: { poster: "/mock/coastal_drive_finish.jpg", loop: "/mock/coastal_drive_finish.mp4" },
};

/** "~6 min" (or "~40 s"): a task's usual time, said plainly. */
function aboutMinutes(eta_s: number | null | undefined): string {
  if (!eta_s) return "";
  return eta_s < 90 ? `~${Math.max(5, Math.round(eta_s / 5) * 5)} s` : `~${Math.round(eta_s / 60)} min`;
}

/** What a task makes: its example's after (a still; the loop plays only while the row is hovered or focused). */
function TaskPreview({ task, playing, dim }: { task: TaskId; playing: boolean; dim?: boolean }) {
  // Render previz shows playblast | final (the hood push), not just a final frame.
  const ex = useApp(
    (s) =>
      (task === "previz" ? s.examples?.find((e) => e.task === "previz" && /hood/i.test(e.id) && e.before_url) : null) ??
      s.examples?.find((e) => e.task === task && e.after_url) ??
      null,
  );
  const media = ex ? { poster: ex.poster_url, loop: ex.after_url } : TASK_MEDIA[task];
  const Icon = TASK_META[task].icon;
  const split = task === "previz" ? ex?.before_url || (!ex ? "/mock/hood_push.mp4" : null) : null;
  return (
    <span
      aria-hidden="true"
      className={cx(
        "relative mt-0.5 h-[46px] w-[82px] shrink-0 overflow-hidden rounded-[3px] bg-raised shadow-[inset_0_0_0_1px_rgba(238,233,224,0.08)] transition-opacity duration-150",
        dim && "opacity-50 grayscale",
      )}
    >
      {media ? (
        <>
          <img src={media.poster} alt="" loading="lazy" decoding="async" draggable={false} className="absolute inset-0 h-full w-full object-cover" />
          {playing && media.loop ? <video src={media.loop} autoPlay muted loop playsInline preload="auto" className="absolute inset-0 h-full w-full object-cover" /> : null}
          {split ? (
            <>
              <video
                src={`${split}#t=0.5`}
                autoPlay={playing}
                muted
                loop
                playsInline
                preload="auto"
                className="absolute inset-0 h-full w-full object-cover"
                style={{ clipPath: "inset(0 50% 0 0)" }}
              />
              <span className="wipe-line absolute inset-y-0 left-1/2" />
            </>
          ) : null}
        </>
      ) : (
        <span className="flex h-full w-full items-center justify-center text-ink3">
          <Icon size={18} strokeWidth={1.5} />
        </span>
      )}
    </span>
  );
}

/** One verb per row: name, one line, mono time on the right. Used for the tiles and "Keep going". */
function VerbRow({
  title,
  line,
  right,
  onClick,
  dim,
  badge,
  note,
  task,
}: {
  title: string;
  line?: string;
  right?: ReactNode;
  onClick: () => void;
  dim?: boolean;
  badge?: ReactNode;
  /** Spoken extra (e.g. "about 4 minutes", "needs setup"). */
  note?: string;
  /** Shows a preview of what the task makes. */
  task?: TaskId;
}) {
  const [hot, setHot] = useState(false);
  const models = useWorkflowModels(undefined, task);
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        onPointerEnter={() => setHot(true)}
        onPointerLeave={() => setHot(false)}
        onFocus={() => setHot(true)}
        onBlur={() => setHot(false)}
        aria-label={[title, line, note, task && models.length ? `Default models: ${models.map((m) => m.name).join(", ")}` : ""]
          .filter(Boolean)
          .map((s) => String(s).replace(/\.$/, ""))
          .join(". ")}
        className={cx("group flex w-full items-start border-b border-line text-left transition-colors duration-150 ease-out", task ? "gap-3.5 py-3" : "gap-4 py-4")}
      >
        {task ? <TaskPreview task={task} playing={hot} dim={dim} /> : null}
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className={cx("text-[16px] transition-colors duration-150", dim ? "text-ink3" : "text-ink group-hover:text-ivory")}>{title}</span>
            {badge}
          </span>
          {line ? <span className="mt-0.5 block text-[13px] leading-snug text-ink3">{line}</span> : null}
          {task ? <ModelLine models={models} className={cx("mt-1", dim && "opacity-60")} /> : null}
        </span>
        <span className="flex shrink-0 items-center gap-2 pt-1.5">
          {right}
          <ArrowRight
            size={14}
            strokeWidth={1.5}
            aria-hidden="true"
            className="-translate-x-1 text-ivory opacity-0 transition-[opacity,transform] duration-150 group-hover:translate-x-0 group-hover:opacity-100"
          />
        </span>
      </button>
    </li>
  );
}

function notReady(status: Status | null, task: TaskId) {
  if (!status?.connected) return "Setup";
  return status.tiles?.[task]?.missing?.length || status.tiles?.[task]?.missing_nodes?.length ? "Setup" : "N/A";
}

// --- tiles ----------------------------------------------------------------------------------------------------------

/** A clip Create made: how it was made (the prompt, the model, size, length, seed, how long it took). */
function ClipInfo({ made }: { made: NonNullable<Shot["made"]> }) {
  const rows: Array<[string, string]> = [
    ["Model", made.model || "LTX-2.5"],
    ["Made from", made.mode === "image" ? "A start frame and words" : "Words"],
    ["Size", made.width && made.height ? `${made.width}×${made.height}` : made.resolution],
    ["Length", made.frames && made.fps ? `${(made.frames / made.fps).toFixed(1)} s · ${made.frames} frames` : ""],
    ["Seed", made.seed != null ? String(made.seed) : ""],
    ["Render time", made.took_s ? tookText(made.took_s) : ""],
  ].filter((r): r is [string, string] => !!r[1]);
  return (
    <section className="mt-6 space-y-3 rounded-card border border-line p-4" aria-label="Clip info" data-testid="clip-info">
      <div className="slate text-ink3">Prompt</div>
      <p className="text-[14px] leading-relaxed text-ink">{made.prompt}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-1.5 border-t border-line pt-3 text-[13px]">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-ink3">{k}</dt>
            <dd className="text-ink2">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Tiles({ shot, ui }: { shot: Shot; ui: ShotUi }) {
  const status = useApp((s) => s.status);
  const failed = useApp((s) => s.statusFailed);
  const pill = pillState(status, failed);
  const tk = useMemo(() => takes(shot), [shot]);
  const base = shot.versions.find((v) => v.id === ui.versionId);
  const onTake = base && !isOriginal(base);

  return (
    <>
      <RailHeader title="Tasks" sub={shot.name} />
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
        {onTake ? (
          <p className="mt-6 flex flex-wrap items-baseline gap-x-2 text-[13px] text-ink3">
            <span className="slate">On take {tk[base.id]?.num} · {tk[base.id]?.label}</span>
            <button type="button" className="text-ink2 underline-offset-4 hover:text-ivory hover:underline" onClick={() => selectVersion(shot.id, "orig")}>
              Use original
            </button>
          </p>
        ) : null}
        {!onTake ? (
          <div className="mt-4 flex items-center gap-2">
            <Button variant="quiet" size="sm" icon={Crop} onClick={() => openPrepare(shot.id, { fromMenu: true })}>
              Trim &amp; crop
            </Button>
          </div>
        ) : null}
        {!onTake && shot.made ? <ClipInfo made={shot.made} /> : null}
        {pill === "offline" ? (
          <Notice className="mt-6" stack tone="warn" text="Pickups can't reach your ComfyUI." action={{ label: "Connect", onClick: () => openOnboarding("connect") }} />
        ) : null}
        <ul className="mt-6 border-t border-line">
          {visibleTasks(status).map((t) => {
            const meta = TASK_META[t];
            const tile = status?.tiles?.[t];
            const ready = !!(status?.connected && tile?.ready);
            return (
              <VerbRow
                key={t}
                task={t}
                title={meta.verb}
                line={meta.line}
                dim={!!status && !ready}
                badge={meta.beta ? <Tag>Labs</Tag> : null}
                note={!status ? undefined : ready ? (tile?.eta_s ? `about ${Math.max(1, Math.round(tile.eta_s / 60))} min` : undefined) : `${notReady(status, t)} needed`}
                onClick={() => {
                  // A job of this task under way on this shot: its progress, never a fresh form.
                  const running = activeJob(shot.id, t);
                  if (running) openJob(shot.id, running);
                  else if (status && !status.connected) openOnboarding("connect");
                  else if (status && !ready) openOnboarding("choose", { task: t });
                  else setRail(shot.id, { kind: "form", task: t });
                }}
                right={
                  !status ? (
                    <Skeleton className="h-3 w-12" />
                  ) : (
                    <span className={ready ? "mono whitespace-nowrap text-[12px] text-ink3" : "slate text-ink3"}>{ready ? aboutMinutes(tile?.eta_s) : notReady(status, t)}</span>
                  )
                }
              />
            );
          })}
        </ul>
        {/* The original's notes (and a take's, on this panel) once there are some; N leaves one. */}
        {base ? <NotesList shot={shot} version={base} hideEmpty className="mt-8" /> : null}
      </div>
    </>
  );
}

// --- running --------------------------------------------------------------------------------------------------------

function lowerFirst(s: string) {
  return s ? s[0].toLowerCase() + s.slice(1) : s;
}

/** What's being made, so the panel isn't empty while a job waits or runs: the shot (for Recast, the person found in it)
 * and, for Recast, who goes in ("pier_C115 → Alien"). A wrong character can be stopped in the first seconds rather than
 * found 6 minutes later. */
function MakingCard({ shot, job, onStop }: { shot: Shot; job: Job; onStop: () => void }) {
  const request = useApp((s) => s.requests[job.id]);
  const projectId = shot.project_id || "unsorted";
  const cast = useApp((s) => s.cast[projectId]);
  const tk = useMemo(() => takes(shot), [shot]);
  const baseId = request?.version_id && shot.versions.some((v) => v.id === request.version_id) ? request.version_id : "orig";
  const base = shot.versions.find((v) => v.id === baseId) ?? shot.versions[0];
  const from = baseId === "orig" ? "the original" : `take ${tk[baseId]?.num ?? ""}`;
  if (job.task !== "recast") {
    return (
      <div className="mb-6 flex items-center gap-3 rounded-card border border-line p-2 pr-3 animate-fade-in">
        <RefPicture url={base.thumb_url} className="h-12 w-[86px] shrink-0 rounded-[3px] bg-black" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] text-ink">{shot.name}</span>
          <span className="mt-0.5 block text-[12px] text-ink3">From {from}</span>
        </span>
      </div>
    );
  }
  const p = (job.params ?? request?.params ?? {}) as Partial<RecastParams>;
  const person = foundPersonThumb(shot.id, baseId, p.frame, p.points);
  const member = p.cast_id ? cast?.find((c) => c.id === p.cast_id) : p.who_ref?.kind === "cast" ? cast?.find((c) => c.id === p.who_ref?.id) : undefined;
  const url = member
    ? member.thumb_url || member.url
    : p.who_ref
      ? keptUrl(p.who_ref, projectId)
      : p.ref_upload_id
        ? keptUrl({ kind: "upload", id: p.ref_upload_id })
        : p.reference_take
          ? shot.versions.find((v) => v.id === p.reference_take)?.reference_url ?? ""
          : "";
  const words = (p.describe ?? "").trim();
  const name = member
    ? member.name
    : p.who_ref && words
      ? words[0].toUpperCase() + words.slice(1)
      : p.ref_upload_id
        ? "Your picture"
        : p.reference_take
          ? "The same character"
          : words
            ? words[0].toUpperCase() + words.slice(1)
            : "Your character";
  const tile = (src: string | null | undefined, label: string, sub: string, contain?: boolean) => (
    <span className="min-w-0 flex-1">
      <span className="relative block aspect-[4/5] overflow-hidden rounded-[3px] bg-black">
        {src ? (
          <RefPicture url={src} className={cx("h-full w-full", contain && "!object-contain")} />
        ) : (
          <span className="flex h-full w-full flex-col items-center justify-center gap-1.5 px-3 text-center text-ink3">
            <UserRound size={20} strokeWidth={1.5} aria-hidden="true" />
            <span className="text-[11px] leading-snug">From your words</span>
          </span>
        )}
      </span>
      <span className="mt-2 block truncate text-[13px] text-ink" title={label}>
        {label}
      </span>
      <span className="block truncate text-[12px] text-ink3">{sub}</span>
    </span>
  );
  return (
    <div className="mb-6 rounded-card border border-line p-3 animate-fade-in" aria-label={`${shot.name} to ${name}`}>
      <div className="flex items-start gap-2">
        {tile(person || base.thumb_url, shot.name, `From ${from}`)}
        <ArrowRight size={16} strokeWidth={1.5} className="mt-[calc(31%)] shrink-0 text-ink3" aria-hidden="true" />
        {tile(url, name, member?.sheet ? "Character sheet" : member ? "From your cast" : url ? "Your picture" : "From your words", !member && !!url)}
      </div>
      {url ? (
        <div className="mt-3 flex items-center justify-between gap-3 border-t border-line pt-2.5">
          <span className="text-[12px] leading-snug text-ink3">Wrong character?</span>
          <Button variant="quiet" size="sm" onClick={onStop}>
            Stop
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function Running({ shot, jobId }: { shot: Shot; jobId: string }) {
  const job = useApp((s) => s.jobs[jobId]);
  const [canceling, setCanceling] = useState(false);
  const stop = async () => {
    setCanceling(true);
    await cancelJob(jobId);
    setCanceling(false);
  };
  const [details, setDetails] = useState(false);
  // Measured time only: how long it has been running, and how long this kind usually takes here.
  const elapsed = useElapsed(job);
  const back = () => setRail(shot.id, { kind: "tiles" });
  const missing = !job;

  useEffect(() => {
    if (missing) void loadJobs();
  }, [missing]);

  const place = useQueuePlace(job);

  if (!job) {
    return (
      <>
        <RailHeader title="Working…" sub={shot.name} onBack={back} />
        <RailBody>
          <Skeleton className="h-12 w-48" />
          <Skeleton className="mt-6 h-px w-full" />
        </RailBody>
      </>
    );
  }

  // Finish says what it makes ("Finish to 4K"), from the job's own params.
  const verb = job.task === "finish" ? finishTitle(choiceOf(job.params as { upscale_4k?: unknown; hdr?: unknown })) : TASK_VERB[job.task];
  const active = isActive(job);
  const lastStage = job.stages.includes(job.stage) ? job.stage : job._lastStage;
  const idx = job.status === "done" ? job.stages.length : lastStage ? job.stages.indexOf(lastStage) : -1;
  const pct = Math.round(job.progress * 100);

  return (
    <>
      <RailHeader title={verb} sub={shot.name} onBack={back} />
      <RailBody>
        {job.status === "failed" ? (
          <div className="mb-8 space-y-3 animate-fade-in">
            <Notice
              stack
              tone="bad"
              text={job.error_detail && job.error ? job.error : lastStage ? `Something went wrong while ${lowerFirst(lastStage)}.` : "Something went wrong."}
              action={{ label: "Try again", onClick: () => void retry(job, shot.id, job.task as TaskId) }}
              secondary={{ label: details ? "Hide details" : "Details", onClick: () => setDetails((d) => !d) }}
            />
            {details ? (
              <pre className="whitespace-pre-wrap break-words rounded-card border border-line bg-bg p-3 font-mono text-[12px] leading-relaxed text-ink2">
                {job.error_detail || job.error || "No details."}
              </pre>
            ) : null}
          </div>
        ) : job.status === "canceled" ? (
          <Notice
            stack
            className="mb-8"
            label="Cancelled"
            text="Nothing was kept."
            action={{ label: "Try again", onClick: () => void retry(job, shot.id, job.task as TaskId) }}
          />
        ) : (
          <div className="pb-8" aria-live="polite">
            <Slate live={job.status === "running"}>{[verb, job.status === "queued" ? "Queued" : job.status === "done" ? "Done" : job.stalled ? "Stalled" : "Rendering"]}</Slate>
            <div className={cx("mono mt-4 text-[48px] leading-none tracking-[-0.02em]", job.stalled && job.status === "running" ? "text-ink3" : "text-ink")}>
              {job.status === "queued" ? "--:--:--" : job.status === "done" ? clock(job.took_s ?? 0) : elapsedClock(elapsed)}
            </div>
            {job.stalled && job.status === "running" ? (
              <div className="mt-5 rounded-card border border-line2 px-4 py-3 animate-fade-in" role="status">
                <div className="slate text-ivory/70">Stalled</div>
                <p className="mt-1.5 text-[14px] leading-snug text-ink">{job.stall_message || "No progress for a while. ComfyUI may be stuck."}</p>
              </div>
            ) : null}
            <div className="mt-4 text-[15px] text-ink">
              {job.status === "queued" ? queueWords(place) : job.status === "done" ? "Done." : `${job.stage}…`}
            </div>
            <ProgressLine value={job.status === "done" ? 1 : job.progress} idle={job.status === "queued"} className="mt-6" />
            <div className="slate mt-2 flex justify-between gap-3 text-ink3">
              {/* How long this kind usually takes here, measured (nothing before the first run); done, how long it took. */}
              <span data-testid="usually">{job.status === "done" ? madeIn(job.took_s) : usually(job.usual_s)}</span>
              <span>{job.status === "queued" ? "" : `${pct}%`}</span>
            </div>
          </div>
        )}

        {active ? <MakingCard shot={shot} job={job} onStop={() => void stop()} /> : null}

        <ol className="border-t border-line" aria-label="Stages">
          {job.stages.map((label, i) => {
            const state = i < idx ? "done" : i === idx && active ? "now" : i === idx && job.status === "failed" ? "failed" : "todo";
            return (
              <li key={label} className="flex items-center gap-3 border-b border-line py-2.5">
                <span className="mono w-6 shrink-0 text-[11px] text-ink3">{String(i + 1).padStart(2, "0")}</span>
                <span
                  className={cx(
                    "min-w-0 flex-1 text-[13px]",
                    state === "now" ? "text-ivory" : state === "done" ? "text-ink3" : state === "failed" ? "text-ivory" : "text-ink4",
                  )}
                >
                  {label}
                </span>
                {state === "done" ? (
                  <Check size={13} strokeWidth={2} className="text-ink3" aria-label="done" />
                ) : state === "now" ? (
                  <TallyDot size={6} />
                ) : state === "failed" ? (
                  <span className="slate">Error</span>
                ) : null}
              </li>
            );
          })}
        </ol>
      </RailBody>
      {active ? (
        <RailFooter>
          {/* One press, as everywhere a job runs (RunStrip): it stops this job's own prompt and keeps nothing. */}
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="md" className="flex-1" onClick={back}>
              Run in background
            </Button>
            <Button variant="quiet" className="px-3" loading={canceling || job.stage === "Canceling"} onClick={() => void stop()}>
              Cancel
            </Button>
          </div>
        </RailFooter>
      ) : job.status === "done" && job.result_version_id ? (
        <RailFooter>
          <Button variant="primary" size="lg" block onClick={() => selectVersion(shot.id, job.result_version_id!)}>
            See the result
          </Button>
        </RailFooter>
      ) : null}
    </>
  );
}

// --- result ---------------------------------------------------------------------------------------------------------

/** A quiet NON-COMMERCIAL pill for a take that ran a model whose licence doesn't allow paid work, or was made from a
 * take that did (take header, Deliveries). A focusable button: a click, Enter or a tap opens a small note naming the
 * model(s) and, for the default frame editor, what to do about it, with a way into Setup. Guidance for the editor in the
 * app, never on the client's receipt sheet. */
export function NonCommercialTag({ version, className }: { version: Pick<Version, "licences">; className?: string }) {
  const names = nonCommercialModels(version);
  const [open, setOpen] = useState(false);
  const [alignEnd, setAlignEnd] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, ref);
  const id = useId();
  if (!names.length) return null;
  const hint = nonCommercialHint(names);
  const editor = names.some((n) => /^Qwen-Image 2\.1\b/.test(n));
  return (
    <span ref={ref} className={cx("relative inline-flex shrink-0", className)}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => {
          // Open towards the side with room: the take panel sits at the right edge of the window.
          const r = ref.current?.getBoundingClientRect();
          setAlignEnd(!!r && r.left + 288 > window.innerWidth - 16);
          setOpen((o) => !o);
        }}
        className="group/nc rounded-[3px] outline-none focus-visible:ring-1 focus-visible:ring-ivory/60"
      >
        <Tag className="cursor-pointer transition-colors duration-150 group-hover/nc:border-ink3 group-hover/nc:text-ink">Non-commercial</Tag>
      </button>
      {open ? (
        <span
          id={id}
          role="note"
          className={cx(
            "absolute top-full z-50 mt-2 w-[280px] max-w-[calc(100vw-32px)] animate-fade-in rounded-[6px] border border-line2 bg-panel p-3 text-left text-[12px] leading-5 text-ink2 shadow-[0_8px_24px_rgba(0,0,0,0.35)]",
            alignEnd ? "right-0" : "left-0",
          )}
        >
          <span className="block">{hint}</span>
          {editor ? (
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                openSetup();
              }}
              className="mt-2 text-ink underline underline-offset-4 hover:text-ivory"
            >
              Open Setup → Licenses
            </button>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}

const DETAIL_LABELS: Record<string, string> = {
  total_seconds: "Total time",
  gpu_seconds: "GPU time",
  gpu_time_s: "GPU time",
  duration_s: "Total time",
  gpu: "GPU",
  models: "Models",
  resolution: "Resolution",
  input: "Input",
  output: "Output",
  frames: "Frames",
  fps: "Frame rate",
  camera: "Camera",
  quality: "Quality",
  move: "Move",
  hdr: "HDR",
  performance_size: "Performance size",
  settings: "Settings",
  seed: "Seed",
  references: "References",
  pinned_first_frame: "First frame locked",
  change: "Change",
  control: "Structure",
  grain: "Film grain",
  swaps: "Swaps",
  left_as_filmed: "Left as filmed",
};

/** Prompts get their own read-only, copyable blocks. */
const PROMPT_KEYS: Record<string, string> = { look_prompt: "Look prompt", prompt: "Video prompt" };
const CHANGE_LABEL: Record<string, string> = { snow: "Snow", rain: "Rain", fog: "Fog", night: "Night", golden_hour: "Golden hour", autumn: "Autumn", custom: "Custom" };
const CONTROL_LABEL: Record<string, string> = { depth: "Depth (surfaces change)", canny: "Edges (light only)" };
const GRAIN_LABEL: Record<string, string> = { off: "Off", fine: "Fine", "35mm": "35mm" };

function PromptBlock({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked: the text is selectable */
    }
  };
  return (
    <div className="border-t border-line pt-3">
      <div className="flex items-center justify-between gap-3">
        <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="inline-flex items-center gap-1 text-[13px] text-ink3 hover:text-ivory">
          {label}
          <ChevronDown size={13} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
        </button>
        <button type="button" onClick={() => void copy()} className="slate rounded-full px-2 py-0.5 text-ink3 transition-colors hover:bg-raised hover:text-ivory">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p className={cx("mono mt-1.5 select-text whitespace-pre-wrap break-words text-[12px] leading-5 text-ink2", !open && "line-clamp-2")}>{text}</p>
    </div>
  );
}

/** Shown as Resolution, Rendered at (or Native), Length and Models above the other details. */
const SUMMARY_KEYS = new Set(["resolution", "rendered_at", "output", "frames", "fps", "models"]);
const DELIVERY_SIZE: Record<string, string> = { UHD: "3840×2160", HD: "1920×1080" };

/** "1920x1088" → "1920×1088" */
function dims(v: unknown): string | null {
  return typeof v === "string" && v ? v.replace(/^(\d+)\s*x\s*(\d+)$/i, "$1×$2") : null;
}

/** The size a take is delivered at, and the size it was rendered at when that differs (previz: HD 1920×1080 from
 * 1920×1088). Older takes recorded only the size they were rendered at; their downloads say what was delivered. */
function takeSizes(v: Version): { delivered: string | null; rendered: string | null } {
  const d = v.details ?? {};
  const made = dims(d.rendered_at) ?? dims(d.resolution) ?? dims(d.output);
  const files = v.downloads ?? [];
  const top = ["UHD", "HD"].find((s) => files.some((f) => f.size === s || new RegExp(`\\b${s}\\b`).test(f.label)));
  const delivered = (top && DELIVERY_SIZE[top]) || dims(d.resolution) || made;
  return { delivered, rendered: made && made !== delivered ? made : null };
}

function takeModels(v: Version): ModelLicence[] {
  if (v.licences) return v.licences;
  const names = Array.isArray(v.details?.models) ? (v.details.models as unknown[]) : [];
  return names.filter((n): n is string => typeof n === "string").map((name) => ({ name, licence: null, commercial: null }));
}

function detailValue(key: string, v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v === "string" && /^\d+\s*x\s*\d+$/i.test(v)) return dims(v) ?? v;
  if (key === "change" && typeof v === "string") return CHANGE_LABEL[v] ?? v;
  if (key === "control" && typeof v === "string") return CONTROL_LABEL[v] ?? (v || "Auto");
  if (key === "grain" && typeof v === "string") return GRAIN_LABEL[v] ?? v;
  if (/(_seconds|_s)$/.test(key) && typeof v === "number") return clock(v);
  // Recast anyone: "the person on the left → Knight, the person on the right → Alien".
  if (key === "swaps" && Array.isArray(v)) {
    return v
      .map((x) => (x && typeof x === "object" ? (x as Record<string, unknown>) : {}))
      .map((x) => `${String(x.who || x.label || "someone")} → ${String(x.character || x.cast_name || x.describe || "a new character")}`)
      .join(", ");
  }
  // The start still's size against the previz (start_frame.size_state): "Not measured", "Pass (1.04×)", the warning.
  if (key === "size_check" && v && typeof v === "object") {
    const s = v as { ratio?: number | null; state?: string; warning?: string | null };
    if (!s.state || s.state === "unmeasured" || s.ratio == null) return "Not measured";
    return `${s.state[0].toUpperCase()}${s.state.slice(1)} (${s.ratio.toFixed(2)}×)${s.warning ? `. ${s.warning}` : ""}`;
  }
  if (Array.isArray(v)) return v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ");
  if (typeof v === "object") {
    // Raw data never shows "null": empty fields are left out (REVIEW F9).
    return Object.entries(v as Record<string, unknown>)
      .filter(([, x]) => x != null && x !== "")
      .map(([k, x]) => `${k.replace(/_/g, " ")}: ${typeof x === "object" ? JSON.stringify(x) : String(x)}`)
      .join(" · ") || "—";
  }
  return String(v);
}

function humanKey(k: string) {
  return DETAIL_LABELS[k] ?? k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function Result({ shot, versionId }: { shot: Shot; versionId: string }) {
  const status = useApp((s) => s.status);
  const view = useApp((s) => s.ui[shot.id]?.view);
  useApp((s) => s.jobs); // re-render when the job that made this take arrives
  const tk = useMemo(() => takes(shot), [shot]);
  // The take's facts (prompts, models, sizes, time) show at once: clicking a clip shows how it was made.
  const [open, setOpen] = useState(true);
  const version = shot.versions.find((v) => v.id === versionId);
  const back = () => setRail(shot.id, { kind: "tiles" });

  if (!version || isOriginal(version)) {
    return (
      <>
        <RailHeader title="Take" sub={shot.name} onBack={back} />
        <RailBody>
          <Notice stack text="This take isn't here anymore." action={{ label: "Back", onClick: back }} />
        </RailBody>
      </>
    );
  }

  const task = version.task as TaskId;
  const take = tk[version.id];
  const job = jobForVersion(version.id);
  const took = versionTook(version, job);
  const allDetails = Object.entries(version.details ?? {}).filter(([k, v]) => v !== "" && v != null && !k.startsWith("_") && k !== "params");
  const details = allDetails.filter(([k]) => !(k in PROMPT_KEYS) && !SUMMARY_KEYS.has(k));
  const prompts = allDetails.filter(([k, v]) => k in PROMPT_KEYS && typeof v === "string") as Array<[string, string]>;
  const sizes = takeSizes(version);
  const frames = Number(version.details?.frames);
  const summary: Array<[string, string]> = [];
  if (sizes.delivered) summary.push(["Resolution", sizes.delivered]);
  // Remove, Recast and Finish keep the source's frame: an open-gate 3840×2748 master is its native size, not a render
  // size. Previz and weather really render at a working size.
  const native = sizes.rendered === `${shot.width}×${shot.height}` && task !== "previz" && task !== "environment";
  if (sizes.rendered) summary.push([native ? "Native" : "Rendered at", sizes.rendered]);
  if (frames > 0) summary.push(["Length", takeLength(frames, Number(version.details?.fps) || shot.fps)]);
  const models = takeModels(version);
  // The default frame editor's licence is non-commercial; it may have come in with the take this one was made from.
  const qwen = models.find((m) => m.name === "Qwen-Image 2.1" && m.commercial === false);
  const receiptOn = view === "receipt";
  // Finish: the grey share is an upscale of the footage (no new content), the AI share its HDR highlights.
  const sentence = finishSentence(version);
  // The GPU the take was made on (recorded since benchmarks); older takes: the one connected now.
  const madeOn = typeof version.details?.gpu === "string" ? version.details.gpu : null;
  const gpu = madeOn ? shortGpu(madeOn) : status?.machine ? shortGpu(status.machine.gpu) : null;

  return (
    <>
      <RailHeader title={take?.label ?? TASK_VERB[task]} sub={`${shot.name} · take ${take?.num ?? ""}`} onBack={back} />
      <RailBody className="space-y-8">
        <div className="animate-rise-in">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
            <Slate live>{[`Take ${take?.num}`, take?.label, aiShare(version.receipt)]}</Slate>
            <NonCommercialTag version={version} />
          </div>
          {took ? (
            <p className="mt-3 text-[13px] text-ink3" title={duration(took)}>
              Made in {tookText(took)}
              {gpu ? ` · ${gpu}` : ""}
            </p>
          ) : null}
        </div>

        {version.receipt ? (
          <div>
            <div className="slate mb-3 text-ink3">What changed</div>
            <ReceiptBar receipt={version.receipt} />
            {sentence ? <p className="mt-2.5 text-[13px] leading-snug text-ink2">{sentence}</p> : null}
            {version.receipt_url ? (
              <button
                type="button"
                onClick={() => setUi(shot.id, { view: receiptOn ? "after" : "receipt" })}
                className="mt-3 inline-flex items-center gap-2 text-[13px] text-ink2 underline-offset-4 transition-colors duration-150 hover:text-ivory hover:underline"
              >
                {receiptOn ? "Hide on picture" : "Show on picture"}
                <span className="kbd">R</span>
              </button>
            ) : null}
          </div>
        ) : null}

        <div className="space-y-3">
          <DownloadControl downloads={version.downloads ?? []} />
          <ShareControl key={version.id} shot={shot} version={version} />
        </div>

        {/* Notes once there are some (the Note button under the picture, or N, leaves one). */}
        <NotesList shot={shot} version={version} hideEmpty />

        <ComfyWorkflows shotId={shot.id} version={version} />

        {NEXT_STEPS[task]?.some((n) => n.task === "finish") ? (
          <Button
            variant="go"
            size="lg"
            block
            onClick={() => {
              const ready = !!(status?.connected && status.tiles?.finish?.ready);
              if (status && !ready) {
                openSetup("finish");
                return;
              }
              setUi(shot.id, { versionId: version.id });
              setRail(shot.id, { kind: "form", task: "finish" });
            }}
          >
            Upres to 4K / HDR
          </Button>
        ) : null}

        <div>
          <div className="slate mb-1 text-ink3">Next</div>
          <ul>
            {NEXT_STEPS[task]?.filter((n) => n.task !== "finish").map((n) => {
              const ready = !!(status?.connected && status.tiles?.[n.task]?.ready);
              return (
                <VerbRow
                  key={n.label}
                  title={n.label}
                  note={!ready && status ? `${notReady(status, n.task)} needed` : undefined}
                  dim={!!status && !ready}
                  right={!ready && status ? <span className="slate text-ink3">{notReady(status, n.task)}</span> : null}
                  onClick={() => {
                    if (status && !ready) {
                      openSetup(n.task);
                      return;
                    }
                    setUi(shot.id, { versionId: version.id });
                    setRail(shot.id, { kind: "form", task: n.task });
                  }}
                />
              );
            })}
          </ul>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-x-4">
          <div className="flex flex-wrap items-center gap-x-4">
            <Button variant="quiet" icon={RotateCcw} onClick={() => void retry(job, shot.id, task, version.id)}>
              Try again
            </Button>
            <Button variant="quiet" icon={SlidersHorizontal} onClick={() => tryAgainWithChanges(shot.id, version)}>
              Change settings
            </Button>
            {isLocalHost() ? (
              <Button variant="quiet" icon={FolderOpen} onClick={() => void revealTake(shot.id, version.id)}>
                Reveal in Finder
              </Button>
            ) : null}
          </div>
          {allDetails.length ? (
            <button
              type="button"
              aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
              className="inline-flex h-8 items-center gap-1 text-[13px] text-ink2 transition-colors duration-150 hover:text-ivory"
            >
              Details
              <ChevronDown size={15} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
            </button>
          ) : null}
        </div>
        {open ? (
          <div className="-mt-4 space-y-4 animate-fade-in">
            {details.length || summary.length || models.length ? (
              <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 border-t border-line pt-4 text-[13px]">
                {summary.map(([label, value]) => (
                  <div key={label} className="contents">
                    <dt className="text-ink3">{label}</dt>
                    <dd className="mono break-words text-[12px] leading-5 text-ink2">{value}</dd>
                  </div>
                ))}
                {models.length ? (
                  <div className="contents">
                    <dt className="text-ink3">Models</dt>
                    <dd className="mono break-words text-[12px] leading-5 text-ink2">
                      <ul className="space-y-1">
                        {models.map((m) => (
                          <li key={m.name}>
                            {m.name}
                            {m.via ? <span className="text-ink3"> · via {m.via}</span> : null}
                            {m.licence ? <span className="text-ink3"> · {m.licence}</span> : null}
                            {m.commercial === false ? <Tag className="ml-1.5">Non-commercial</Tag> : null}
                          </li>
                        ))}
                      </ul>
                    </dd>
                  </div>
                ) : null}
                {details.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ink3">{humanKey(k)}</dt>
                    <dd className="mono break-words text-[12px] leading-5 text-ink2">{detailValue(k, v)}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
            {/* The receipt sheet says a take isn't for paid work; what to do about it is said here, for the editor. */}
            {qwen ? (
              <p className="text-[12px] leading-5 text-ink3">
                For paid work, open{" "}
                <button type="button" onClick={() => openSetup()} className="text-ink2 underline underline-offset-4 hover:text-ivory">
                  Setup → Licenses → Frame editor
                </button>{" "}
                to switch to the Apache-2.0 editor and remake {qwen.via ? `${qwen.via}, then this take` : "this take"}.
              </p>
            ) : null}
            {prompts.map(([k, v]) => (
              <PromptBlock key={k} label={PROMPT_KEYS[k]} text={v} />
            ))}
          </div>
        ) : null}
      </RailBody>
    </>
  );
}

