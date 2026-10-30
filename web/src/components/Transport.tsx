import { Check, Ellipsis, EyeOff, MessageSquarePlus, Pause, Play, SquareSplitHorizontal, StepBack, StepForward, Trash2 } from "lucide-react";
import { finishLabel } from "../lib/finish";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { moveRenderWindow, openJob, selectVersion, setHidden, setUi, toggleApproved, trashTake } from "../actions";
import { cx, isActive, isOriginal, ltxWindowFrames, minWindowFrames, TASK_SHORT, takeFrames, takes, timecode, windowStart } from "../lib/format";
import { usePlayer, usePlayerState } from "../lib/player";
import { useApp, type RenderWindow, type ShotUi, type ViewMode } from "../store";
import type { Note, Shot, Version } from "../types";
import { Thumb } from "./MediaCard";
import { TrimMarks } from "./Prepare";
import { framePct, NoteComposer, noteNear, NoteTicks, startNote, useNoteDraft, useNoteKey } from "./Notes";
import { Hint, IconButton, Segmented, TallyDot, Tip } from "./ui";

// --- transport row -----------------------------------------------------------------------------------------------

export function Transport({
  shot,
  compare,
  hasReceipt,
  view,
  wipe,
  locked,
  onView,
  onToggleWipe,
  version,
  beforeId,
  onBefore,
}: {
  shot: Shot;
  compare: boolean;
  hasReceipt: boolean;
  view: ViewMode;
  wipe: boolean;
  locked: boolean;
  onView: (v: ViewMode) => void;
  onToggleWipe: () => void;
  version: Version;
  beforeId: string;
  onBefore: (id: string) => void;
}) {
  const player = usePlayer();
  const snap = usePlayerState();
  const canCompare = compare && !locked;
  const tk = useMemo(() => takes(shot), [shot]);
  const approved = version.status === "approved";
  // A step flow's Review approves this take from the rail: one Approve on screen, not two.
  const railApproves = useApp((s) => {
    const u = s.ui[shot.id];
    return u?.rail.kind === "form" && u.railApprove === version.id;
  });
  useNoteKey(shot.id, version.id);
  const noting = useNoteDraft(shot.id, version.id);
  return (
    <div className="transport relative flex h-12 shrink-0 items-center gap-0.5 pt-2">
      <NoteComposer shot={shot} version={version} />
      <IconButton icon={snap.playing ? Pause : Play} label={snap.playing ? "Pause" : "Play"} hint="Space" onClick={player.toggle} />
      <IconButton icon={StepBack} label="Previous frame" hint="←" className="tp-step" onClick={() => player.step(-1)} />
      <IconButton icon={StepForward} label="Next frame" hint="→" className="tp-step" onClick={() => player.step(1)} />
      {/* Playhead / duration, as monitors in Premiere and Resolve show it: 144 frames at 24 fps read 00:00:06:00. */}
      <div className="tp-time mono ml-3 shrink-0 select-none whitespace-nowrap text-[13px]" aria-live="off">
        <span className="text-ink">{timecode(snap.frame, shot.fps)}</span>
        <span className="tp-duration text-ink4"> / {timecode(Math.min(shot.frames, takeFrames(shot, version)), shot.fps)}</span>
      </div>
      <div className="tp-tools ml-auto flex items-center gap-1.5">
        {canCompare ? (
          <label className="tp-left mr-1 flex items-center gap-2">
            <span className="tp-left-label slate text-ink3">Left</span>
            <select
              value={beforeId}
              onChange={(e) => onBefore(e.target.value)}
              aria-label="The take on the wipe's left side"
              className="tp-left-select mono h-7 max-w-[180px] cursor-pointer truncate rounded-full border border-line bg-bg px-2.5 text-[11px] uppercase tracking-[0.06em] text-ink2 hover:border-line2 hover:text-ivory"
            >
              {shot.versions
                .filter((v) => v.id !== version.id && (!v.hidden || v.id === beforeId))
                .map((v) => (
                  <option key={v.id} value={v.id}>
                    {`Take ${tk[v.id]?.num} · ${tk[v.id]?.label}`}
                  </option>
                ))}
            </select>
          </label>
        ) : null}
        <Segmented<ViewMode>
          size="sm"
          label="View"
          value={canCompare ? view : "after"}
          disabled={!canCompare}
          onChange={onView}
          options={[
            { value: "before", label: "Before", hint: "B" },
            { value: "after", label: "After", hint: "A" },
            // The tint comes with Remove and Replace takes; elsewhere it isn't offered at all.
            ...(hasReceipt && canCompare ? [{ value: "receipt" as ViewMode, label: "Changes", hint: "R" }] : []),
          ]}
        />
        {hasReceipt && canCompare ? <Hint text="Tints what the model changed" /> : null}
        <IconButton
          icon={SquareSplitHorizontal}
          label={wipe ? "Wipe on" : "Wipe off"}
          hint="W"
          active={canCompare && view === "after" && wipe}
          disabled={!canCompare || view !== "after"}
          onClick={onToggleWipe}
        />
        <IconButton
          icon={MessageSquarePlus}
          label="Note"
          hint="N"
          active={noting}
          tipHidden={noting}
          onClick={() => startNote(player, shot.id, version.id)}
        />
        {isOriginal(version) || railApproves ? null : (
          <Tip text={approved ? "Approved: it's in the project's Deliveries · ⇧A to undo" : "Approve this take: it goes to the project's Deliveries, ready to finish and download · ⇧A"} align="end">
            <button
              type="button"
              aria-pressed={approved}
              onClick={() => toggleApproved(shot.id, version.id)}
              className={cx(
                "tp-approve ml-1 inline-flex h-8 items-center gap-1.5 rounded-full border px-3.5 text-[13px] transition-colors duration-150",
                approved ? "border-ivory bg-ivory text-[#0D0C0B]" : "border-line2 text-ink2 hover:border-ivory/50 hover:text-ivory",
              )}
            >
              <Check size={14} strokeWidth={approved ? 2.25 : 1.5} />
              <span className="tp-approve-text">{approved ? "Approved" : "Approve"}</span>
            </button>
          </Tip>
        )}
      </div>
    </div>
  );
}

// --- filmstrip / scrubber ------------------------------------------------------------------------------------------

const stripCache = new Map<string, string>();

function once(el: HTMLVideoElement, ev: string, ms: number) {
  return new Promise<void>((resolve, reject) => {
    const t = window.setTimeout(() => {
      cleanup();
      reject(new Error("timeout"));
    }, ms);
    const ok = () => {
      cleanup();
      resolve();
    };
    const bad = () => {
      cleanup();
      reject(new Error("error"));
    };
    const cleanup = () => {
      window.clearTimeout(t);
      el.removeEventListener(ev, ok);
      el.removeEventListener("error", bad);
    };
    el.addEventListener(ev, ok);
    el.addEventListener("error", bad);
  });
}

async function extractStrip(url: string, n: number, th: number, aspect: number): Promise<string> {
  const v = document.createElement("video");
  v.muted = true;
  v.preload = "auto";
  v.playsInline = true;
  v.src = url;
  try {
    await once(v, "loadeddata", 10000);
    const tw = Math.round(th * aspect);
    const c = document.createElement("canvas");
    c.width = tw * n;
    c.height = th;
    const g = c.getContext("2d")!;
    const dur = isFinite(v.duration) ? v.duration : 1;
    for (let i = 0; i < n; i++) {
      v.currentTime = Math.min(dur - 0.01, ((i + 0.5) / n) * dur);
      await once(v, "seeked", 4000);
      g.drawImage(v, i * tw, 0, tw, th);
    }
    return c.toDataURL("image/jpeg", 0.72);
  } finally {
    v.removeAttribute("src");
    v.load();
  }
}

function useFilmstrip(url: string, n: number, th: number, aspect: number) {
  const key = `${url}|${n}`;
  const [state, setState] = useState<{ src: string | null; failed: boolean }>({ src: stripCache.get(key) ?? null, failed: false });
  useEffect(() => {
    if (!n) return;
    const hit = stripCache.get(key);
    if (hit) {
      setState({ src: hit, failed: false });
      return;
    }
    let cancelled = false;
    const t = window.setTimeout(() => {
      extractStrip(url, n, th, aspect)
        .then((src) => {
          stripCache.set(key, src);
          if (!cancelled) setState({ src, failed: false });
        })
        .catch(() => {
          if (!cancelled) setState((s) => ({ src: s.src, failed: !s.src }));
        });
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [key, url, n, th, aspect]);
  return state;
}

export function Filmstrip({ shot, url, notes }: { shot: Shot; url: string; notes?: Note[] }) {
  const player = usePlayer();
  const snap = usePlayerState();
  // Recast's window on a long shot: the part it renders, bracketed (the rest dimmed), with a bar above to drag it.
  const win = useApp((s) => (s.renderWindow?.shotId === shot.id ? s.renderWindow : null));
  // Trim & crop: the in and out handles.
  const trimming = useApp((s) => s.ui[shot.id]?.rail.kind === "prepare");
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [hoverNote, setHoverNote] = useState<Note | null>(null);
  // A press on a note's tick lands exactly on its frame; dragging on from there scrubs as usual.
  const drag = useRef<{ resume: boolean; snapX: number | null } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const H = 48;
  const aspect = shot.width / shot.height || 16 / 9;
  // Bucket the tile count so small resizes don't re-extract.
  const n = width ? Math.min(32, Math.max(2, Math.ceil(width / (H * aspect) / 2) * 2)) : 0;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const strip = useFilmstrip(url, n, Math.round(H * dpr), aspect);
  const frames = Math.max(1, shot.frames);

  const frameAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return Math.max(0, Math.min(frames - 1, Math.floor(((clientX - r.left) / r.width) * frames)));
  };
  const noteAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return noteNear(notes, frames, r.width, clientX - r.left);
  };
  const pct = (f: number) => `${framePct(f, frames)}%`;

  const edge = (f: number) => `${(Math.max(0, Math.min(frames, f)) / frames) * 100}%`;

  return (
    <div className="relative mt-1 shrink-0">
      {win && !win.fixed ? <WindowBar shot={shot} win={win} frames={frames} /> : null}
      {hoverNote ? (
        <span
          className={cx(
            // As wide as the text up to 360 px (not squeezed near the right edge), then up to three lines.
            "pointer-events-none absolute bottom-full z-10 mb-1.5 flex w-max max-w-[min(360px,100%)] items-baseline gap-2 rounded-[2px] bg-raised px-1.5 py-0.5 text-[12px] leading-snug text-ink",
            // Near an edge the text opens away from it, so it stays on screen and next to its tick.
            framePct(hoverNote.frame, frames) < 20 ? "-translate-x-1.5" : framePct(hoverNote.frame, frames) > 80 ? "-translate-x-[calc(100%-6px)]" : "-translate-x-1/2",
          )}
          style={{ left: pct(hoverNote.frame) }}
        >
          <span className="mono shrink-0 text-[11px] text-ink3">{timecode(hoverNote.frame, shot.fps)}</span>
          <span className="line-clamp-3 min-w-0 whitespace-normal break-words">{hoverNote.text}</span>
        </span>
      ) : hover != null ? (
        <span
          className="mono pointer-events-none absolute bottom-full z-10 mb-1.5 -translate-x-1/2 rounded-[2px] bg-raised px-1.5 py-0.5 text-[11px] text-ink"
          style={{ left: pct(hover) }}
        >
          {timecode(hover, shot.fps)}
        </span>
      ) : null}
      <div
        ref={ref}
        role="slider"
        tabIndex={0}
        aria-label="Scrub"
        aria-valuemin={0}
        aria-valuemax={frames - 1}
        aria-valuenow={snap.frame}
        aria-valuetext={timecode(snap.frame, shot.fps)}
        className={cx(
          "relative h-11 touch-none select-none overflow-hidden rounded-[2px] bg-panel outline outline-1 outline-line",
          hoverNote ? "cursor-pointer" : "cursor-ew-resize",
        )}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          const near = noteAt(e.clientX);
          drag.current = { resume: player.getSnapshot().playing, snapX: near ? e.clientX : null };
          if (drag.current.resume) player.pause();
          player.seek(near ? near.frame : frameAt(e.clientX));
        }}
        onPointerMove={(e) => {
          const f = frameAt(e.clientX);
          setHover(f);
          const d = drag.current;
          if (d && d.snapX != null && Math.abs(e.clientX - d.snapX) < 4) return;
          if (d) d.snapX = null;
          setHoverNote(d ? null : noteAt(e.clientX));
          if (d) player.seek(f);
        }}
        onPointerUp={(e) => {
          if (!drag.current) return;
          e.currentTarget.releasePointerCapture(e.pointerId);
          const resume = drag.current.resume;
          drag.current = null;
          if (resume) player.play();
        }}
        onPointerLeave={() => {
          setHover(null);
          setHoverNote(null);
        }}
      >
        {strip.src ? (
          <img src={strip.src} alt="" draggable={false} className="h-full w-full object-cover object-left opacity-75" />
        ) : strip.failed ? null : (
          <div className="skeleton absolute inset-0 rounded-none" />
        )}
        {win ? (
          <>
            <div className="pointer-events-none absolute inset-y-0 left-0 bg-black/60" style={{ width: edge(win.start) }} />
            <div className="pointer-events-none absolute inset-y-0 right-0 bg-black/60" style={{ left: edge(win.start + win.frames) }} />
            <div className="pointer-events-none absolute inset-y-0" style={{ left: edge(win.start), width: `calc(${edge(win.start + win.frames)} - ${edge(win.start)})` }}>
              <span className="absolute inset-y-0 left-0 w-1.5 border-y-2 border-l-2 border-ivory" />
              <span className="absolute inset-y-0 right-0 w-1.5 border-y-2 border-r-2 border-ivory" />
            </div>
          </>
        ) : null}
        {trimming ? <TrimMarks shot={shot} frames={frames} track={ref} /> : null}
        <NoteTicks notes={notes} frames={frames} active={hoverNote?.id ?? null} />
        {hover != null && !hoverNote ? <div className="pointer-events-none absolute inset-y-0 w-px bg-ivory/40" style={{ left: pct(hover) }} /> : null}
        <div className="wipe-line pointer-events-none absolute inset-y-0 -translate-x-1/2" style={{ left: pct(snap.frame) }}>
          <span className="absolute -top-px left-1/2 h-0 w-0 -translate-x-1/2 border-x-[4px] border-t-[5px] border-x-transparent border-t-ivory" />
        </div>
      </div>
    </div>
  );
}

/** The bar above the filmstrip that carries Recast's window: drag it (or press the track, or use the arrow keys) to move
 * which part of a long shot is recast, on LTX-2.5's 8-frame grid. */
function WindowBar({ shot, win, frames }: { shot: Shot; win: RenderWindow; frames: number }) {
  const task = useApp((s) => {
    const rail = s.ui[shot.id]?.rail;
    return rail && rail.kind === "form" ? rail.task : "recast";
  });
  const track = useRef<HTMLDivElement>(null);
  const grab = useRef<number | null>(null); // where in the window the pointer holds it (frames from its start)
  const [dragging, setDragging] = useState(false);
  const fps = shot.fps > 0 ? shot.fps : 24;
  const half = Math.max(0.5, Math.round((win.frames / fps) * 2) / 2);
  const length = `${Number.isInteger(half) ? half : half.toFixed(1)} s`;
  const at = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return ((clientX - r.left) / Math.max(1, r.width)) * frames;
  };
  const moveTo = (start: number) => moveRenderWindow(windowStart(start, win.frames, frames));
  // In and out handles: the length is the user's, from about a second up to what the GPU renders (status.max_frames).
  // Frames stay on LTX's grid: the bracket starts on a multiple of 8 and holds 8k+1 frames.
  const maxFrames = useApp((s) => s.status?.max_frames_by_task?.[task] ?? s.status?.max_frames);
  const maxLen = ltxWindowFrames(maxFrames && maxFrames > 0 ? maxFrames : Math.round(4 * fps) + 1, frames);
  const minLen = Math.min(maxLen, minWindowFrames(fps, frames));
  const last = win.start + win.frames - 1;
  const edge = useRef<"in" | "out" | null>(null);
  const setIn = (f: number) => {
    const s = Math.round(Math.min(Math.max(f, last - maxLen + 1), last - minLen + 1) / 8) * 8;
    const start = Math.max(0, s);
    moveRenderWindow(start, last - start + 1);
  };
  const setOut = (f: number) => {
    const len = Math.max(minLen, Math.min(maxLen, ltxWindowFrames(Math.max(1, Math.round(f) - win.start + 1), frames - win.start)));
    moveRenderWindow(win.start, len);
  };
  const handle = (side: "in" | "out") => (
    <span
      role="presentation"
      title={side === "in" ? "Drag to set where it starts" : "Drag to set where it ends"}
      className={cx("absolute inset-y-0 z-10 w-2.5 cursor-ew-resize touch-none bg-ivory/60 hover:bg-ivory", side === "in" ? "left-0" : "right-0")}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.setPointerCapture(e.pointerId);
        edge.current = side;
        setDragging(true);
      }}
      onPointerMove={(e) => {
        if (edge.current !== side) return;
        e.stopPropagation();
        (side === "in" ? setIn : setOut)(at(e.clientX));
      }}
      onPointerUp={(e) => {
        if (edge.current !== side) return;
        e.stopPropagation();
        edge.current = null;
        setDragging(false);
        e.currentTarget.releasePointerCapture(e.pointerId);
      }}
      onPointerCancel={() => {
        edge.current = null;
        setDragging(false);
      }}
    />
  );
  const step = 8 * Math.max(1, Math.round(fps / 8)); // Shift + arrow: about a second
  const pct = (f: number) => `${(f / frames) * 100}%`;
  return (
    <div
      ref={track}
      className="relative mb-1 h-4 touch-none select-none"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        const f = at(e.clientX);
        // On the bar: hold it where it was pressed. On the empty track: it comes to the pointer, centred.
        const inside = f >= win.start && f <= win.start + win.frames;
        grab.current = inside ? f - win.start : win.frames / 2;
        if (!inside) moveTo(f - grab.current);
        setDragging(true);
      }}
      onPointerMove={(e) => {
        if (grab.current == null) return;
        moveTo(at(e.clientX) - grab.current);
      }}
      onPointerUp={(e) => {
        if (grab.current == null) return;
        grab.current = null;
        setDragging(false);
        e.currentTarget.releasePointerCapture(e.pointerId);
      }}
      onPointerCancel={() => {
        grab.current = null;
        setDragging(false);
      }}
    >
      <div
        role="slider"
        tabIndex={0}
        aria-label="The part of the shot to change"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, frames - win.frames)}
        aria-valuenow={win.start}
        aria-valuetext={`${timecode(win.start, shot.fps)} to ${timecode(win.start + win.frames - 1, shot.fps)}`}
        title={`Drag to move it. Up to ${Number.isInteger(Math.round((maxLen / fps) * 2) / 2) ? Math.round((maxLen / fps) * 2) / 2 : (Math.round((maxLen / fps) * 2) / 2).toFixed(1)} s on your GPU`}
        onKeyDown={(e) => {
          const d = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
          if (d) {
            e.preventDefault();
            e.stopPropagation();
            moveTo(win.start + d * (e.shiftKey ? step : 8));
          } else if (e.key === "Home" || e.key === "End") {
            e.preventDefault();
            e.stopPropagation();
            moveTo(e.key === "Home" ? 0 : frames);
          }
        }}
        className={cx(
          "absolute inset-y-0 flex items-center justify-center overflow-hidden whitespace-nowrap rounded-[2px] border border-ivory/80 transition-colors duration-150",
          dragging ? "cursor-grabbing bg-ivory/30" : "cursor-grab bg-ivory/15 hover:bg-ivory/25",
        )}
        style={{ left: pct(win.start), width: pct(win.frames) }}
      >
        {handle("in")}
        <span className="slate text-[10px] leading-none text-ivory">{dragging ? `${timecode(win.start, shot.fps)} · ${length}` : `${TASK_SHORT[task]} · ${length}`}</span>
        {handle("out")}
      </div>
    </div>
  );
}

// --- versions -----------------------------------------------------------------------------------------------------

/** Versions are takes: TAKE 01 Original, TAKE 02 Remove… The take on screen has its tally lit.
 * vertical: the list at the top of the right panel (desktop), one take per row, bigger and easier to read; the
 * row under the picture is the phone's. */
export function VersionStrip({ shot, ui, vertical = false }: { shot: Shot; ui: ShotUi; vertical?: boolean }) {
  const tk = useMemo(() => takes(shot), [shot]);
  const jobs = useApp((s) => s.jobs);
  const order = useApp((s) => s.jobOrder);
  const pending = useMemo(
    () =>
      order
        .map((id) => jobs[id])
        .filter((j) => j && j.shot_id === shot.id && isActive(j))
        .reverse(),
    [order, jobs, shot.id],
  );
  const current = shot.versions.some((v) => v.id === ui.versionId) ? ui.versionId : "orig";
  const hiddenCount = shot.versions.filter((v) => v.hidden).length;
  const shown = shot.versions.filter((v) => !v.hidden || ui.showHidden || v.id === current);
  return (
    <div
      className={vertical ? "flex max-h-[214px] shrink-0 flex-col gap-1.5 overflow-y-auto border-b border-line px-3 pb-3 pt-1" : "flex h-[58px] shrink-0 items-center gap-2 overflow-x-auto pt-2"}
      role="tablist"
      aria-label="Takes"
      aria-orientation={vertical ? "vertical" : "horizontal"}
    >
      {shown.map((v) => {
        const on = v.id === current;
        const t = tk[v.id];
        const from = v.input_version && v.input_version !== "orig" ? tk[v.input_version]?.num : null;
        return (
          <div key={v.id} className={cx("group/take relative shrink-0", v.hidden && "opacity-50")}>
            <button
              type="button"
              role="tab"
              aria-selected={on}
              onClick={() => selectVersion(shot.id, v.id)}
              className={cx(
                "flex items-center gap-2.5 rounded-card border pl-1 pr-3.5 transition-colors duration-150 ease-out",
                vertical ? "h-[52px] w-full pr-10" : "h-11",
                on ? (vertical ? "border-ivory/70 bg-raised" : "border-ivory/50 bg-panel") : "border-line hover:border-line2",
              )}
            >
              <Thumb eager src={v.thumb_url} className={vertical ? "h-[42px] w-[74px] rounded-[3px]" : "h-[34px] w-[60px] rounded-[3px]"} />
              <span className="text-left">
                <span className={cx("slate flex items-center gap-1.5", on ? "text-ivory/80" : "text-ink3")}>
                  {on ? <TallyDot size={5} /> : null}Take {t?.num}
                  {from ? <span className="text-ink4">← {from}</span> : null}
                  {v.status === "approved" ? <Check size={11} strokeWidth={2.5} className="text-ivory" aria-label="approved" /> : null}
                  {v.hidden ? <span className="text-ink4">hidden</span> : null}
                </span>
                <span className={cx("block leading-tight", vertical ? "text-[14px]" : "text-[13px]", on ? "text-ivory" : "text-ink2")}>{t?.label}</span>
              </span>
            </button>
            {v.id !== "orig" ? <TakeMenu shotId={shot.id} take={v} num={t?.num} canTrash={shot.project_id !== "examples"} /> : null}
          </div>
        );
      })}
      {pending.map((j, i) => (
        <button
          key={j.id}
          type="button"
          onClick={() => openJob(shot.id, j)}
          aria-label={`Take ${String(shot.versions.length + i + 1).padStart(2, "0")}, ${TASK_SHORT[j.task]}, ${j.status === "queued" ? "queued" : `${Math.round(j.progress * 100)}%`}: show progress`}
          className={cx("flex shrink-0 items-center gap-2.5 rounded-card border border-line px-3.5 text-left transition-colors duration-150 hover:border-line2", vertical ? "h-[52px] w-full" : "h-11")}
        >
          <span>
            <span className="slate flex items-center gap-1.5 text-ink3">
              {j.status === "running" ? <TallyDot size={5} /> : null}Take {String(shot.versions.length + i + 1).padStart(2, "0")}
            </span>
            <span className="block text-[13px] leading-tight text-ink2">
              {j.task === "finish" ? finishLabel(j.params as { upscale_4k?: unknown; hdr?: unknown }) : TASK_SHORT[j.task]} <span className="mono text-[11px] text-ink3">{j.status === "queued" ? "queued" : `${Math.round(j.progress * 100)}%`}</span>
            </span>
          </span>
        </button>
      ))}
      {hiddenCount ? (
        <button
          type="button"
          onClick={() => setUi(shot.id, { showHidden: !ui.showHidden })}
          className="slate ml-1 shrink-0 rounded-full px-2.5 py-1 text-ink3 transition-colors hover:bg-raised hover:text-ivory"
        >
          {ui.showHidden ? "Hide hidden" : `Show hidden (${hiddenCount})`}
        </button>
      ) : null}
    </div>
  );
}

/** "…" on a take in the strip: Hide or show it, or Move to Trash (Undo in the toast). The menu sits in a portal so the
 * strip's sideways scroll can't clip it. */
function TakeMenu({ shotId, take, num, canTrash }: { shotId: string; take: Version; num?: number | string; canTrash: boolean }) {
  // Below the button, or above it when the strip sits near the bottom of the window (it usually does).
  const [at, setAt] = useState<{ top?: number; bottom?: number; left: number } | null>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const close = () => setAt(null);
  useEffect(() => {
    if (!at) return;
    const onDown = (e: PointerEvent) => {
      if (!menu.current?.contains(e.target as Node) && !btn.current?.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    // A scroll that moves the button (the page, or the strip) closes it; the focus nudge on open doesn't.
    const from = btn.current?.getBoundingClientRect();
    const onScroll = () => {
      const r = btn.current?.getBoundingClientRect();
      if (!r || !from || Math.abs(r.top - from.top) > 4 || Math.abs(r.left - from.left) > 4) close();
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [at]);
  const item = "flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-[14px] text-ink transition-colors duration-150 hover:bg-line hover:text-ivory focus-visible:bg-line focus-visible:outline-none";
  return (
    <span className={cx("absolute -right-1.5 -top-1.5 z-[2] transition-opacity duration-150 focus-within:opacity-100 group-hover/take:opacity-100", at ? "opacity-100" : "opacity-0 max-[767px]:opacity-100")}>
      <button
        ref={btn}
        type="button"
        aria-haspopup="menu"
        aria-expanded={!!at}
        aria-label={`Take ${num ?? ""}: more`}
        onClick={() => {
          if (at) return close();
          const r = btn.current!.getBoundingClientRect();
          const left = Math.min(r.left, window.innerWidth - 228);
          setAt(r.bottom + 120 > window.innerHeight ? { bottom: window.innerHeight - r.top + 6, left } : { top: r.bottom + 6, left });
        }}
        className="flex h-5 w-5 items-center justify-center rounded-full border border-line2 bg-raised text-ink3 hover:text-ivory"
      >
        <Ellipsis size={11} strokeWidth={1.75} />
      </button>
      {at
        ? createPortal(
            <div ref={menu} role="menu" aria-label={`Take ${num ?? ""}`} style={{ top: at.top, bottom: at.bottom, left: at.left }} className="fixed z-[70] w-[220px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
              <button
                type="button"
                role="menuitem"
                className={item}
                onClick={() => {
                  close();
                  void setHidden(shotId, take.id, !take.hidden);
                }}
              >
                <EyeOff size={14} strokeWidth={1.5} className="text-ink3" /> {take.hidden ? "Show take" : "Hide take"}
              </button>
              {canTrash ? (
                <button
                  type="button"
                  role="menuitem"
                  className={item}
                  onClick={() => {
                    close();
                    void trashTake(shotId, take.id);
                  }}
                >
                  <Trash2 size={14} strokeWidth={1.5} className="text-ink3" /> Move to Trash
                </button>
              ) : null}
            </div>,
            document.body,
          )
        : null}
    </span>
  );
}
