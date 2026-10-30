// Trim & crop: pick in and out on the filmstrip and a shape on the picture, then make a new clip from the original
// (POST /api/shots/{id}/prepare). Offered right after a clip is added, and from the shot's menu. Also the source limits
// line shown where clips are added.
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { cropRatio, fullCrop, makePreparedClip, setPrepare, skipPrepare } from "../actions";
import { cx, timecode } from "../lib/format";
import { activePlayer, usePlayer, usePlayerState } from "../lib/player";
import { getState, setState, useApp, type PrepareDraft } from "../store";
import type { CropAspect, Shot } from "../types";
import { RailBody, RailFooter, RailHeader } from "./RailChrome";
import { Button, Hint, Notice } from "./ui";

const SHAPES: Array<"original" | CropAspect> = ["original", "16:9", "9:16", "1:1", "4:5", "2.39:1"];
const MIN_CROP = 64; // px, as the backend (media.MIN_CROP)

function minFrames(): number {
  return getState().status?.min_frames ?? 9;
}

function secs(frames: number, fps: number): string {
  return `${(frames / (fps > 0 ? fps : 24)).toFixed(1)} s`;
}

/** The draft for this shot while its Trim & crop is open (null otherwise). */
function useDraft(shotId: string): PrepareDraft | null {
  return useApp((s) => (s.prepare?.shotId === shotId && s.ui[shotId]?.rail.kind === "prepare" ? s.prepare : null));
}

/** In and out moved to `frame`, keeping at least min_frames between them and both inside the clip. */
function withIn(d: PrepareDraft, frame: number, total: number): Pick<PrepareDraft, "inFrame" | "outFrame"> {
  const n = minFrames();
  const inFrame = Math.max(0, Math.min(Math.round(frame), total - n));
  return { inFrame, outFrame: Math.max(d.outFrame, Math.min(total - 1, inFrame + n - 1)) };
}
function withOut(d: PrepareDraft, frame: number, total: number): Pick<PrepareDraft, "inFrame" | "outFrame"> {
  const n = minFrames();
  const outFrame = Math.min(total - 1, Math.max(Math.round(frame), n - 1));
  return { outFrame, inFrame: Math.min(d.inFrame, Math.max(0, outFrame - n + 1)) };
}

function FrameIcon({ aspect }: { aspect: string }) {
  const [w, h] = aspect.split(":").map(Number);
  const k = 12 / Math.max(w, h);
  return <span aria-hidden="true" className="inline-block shrink-0 rounded-[1px] border border-current" style={{ width: Math.max(4, w * k), height: Math.max(4, h * k) }} />;
}

// --- the rail -------------------------------------------------------------------------------------------------------

export function PrepareRail({ shot }: { shot: Shot }) {
  const draft = useApp((s) => (s.prepare?.shotId === shot.id ? s.prepare : null));
  const maxFrames = useApp((s) => s.status?.max_frames) ?? 97;
  const player = usePlayer();
  const snap = usePlayerState();
  const total = Math.max(1, shot.frames);
  const fps = shot.fps > 0 ? shot.fps : 24;

  // Opened without a draft (or with another shot's): the whole clip, its own shape.
  useEffect(() => {
    if (!draft) {
      setState({ prepare: { shotId: shot.id, inFrame: 0, outFrame: total - 1, aspect: "original", box: null, busy: false, error: null } });
    }
  }, [draft, shot.id, total]);

  // I and O set in and out at the playhead, as in an editor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) return;
      const k = e.key.toLowerCase();
      if (k !== "i" && k !== "o") return;
      const d = getState().prepare;
      if (!d || d.shotId !== shot.id || getState().setup.open) return;
      e.preventDefault();
      const f = activePlayer.current?.getSnapshot().frame ?? 0;
      setPrepare(k === "i" ? withIn(d, f, total) : withOut(d, f, total));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shot.id, total]);

  if (!draft) return null;
  const count = draft.outFrame - draft.inFrame + 1;
  const trimmed = draft.inFrame > 0 || draft.outFrame < total - 1;
  const cropped = draft.aspect !== "original" && !!draft.box && (draft.box.w < shot.width - 1 || draft.box.h < shot.height - 1);
  const size = cropped && draft.box ? `${draft.box.w}×${draft.box.h}` : `${shot.width}×${shot.height}`;
  const runSeconds = (maxFrames / fps).toFixed(1);
  // Worth a line only when a real part is left out (as useLengthCap: at least 0.25 s and 5 %).
  const longer = count - maxFrames >= 0.25 * fps && count - maxFrames >= 0.05 * count;

  const point = (label: "In" | "Out", frame: number, key: "I" | "O", set: () => void) => (
    <div className="min-w-0 rounded-card border border-line p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="slate text-ink3">{label}</span>
        <button
          type="button"
          onClick={() => player.seek(frame)}
          className="mono text-[13px] text-ink transition-colors hover:text-ivory"
          aria-label={`Go to the ${label.toLowerCase()} point, ${timecode(frame, fps)}`}
        >
          {timecode(frame, fps)}
        </button>
      </div>
      <Button variant="secondary" size="sm" block className="mt-2.5" onClick={set}>
        Set {label.toLowerCase()} <span className="kbd ml-0.5">{key}</span>
      </Button>
    </div>
  );

  return (
    <>
      <RailHeader title="Trim & crop" sub="A new clip · the original stays" onBack={() => skipPrepare(shot.id)} />
      <RailBody className="space-y-8">
        <section aria-label="Length">
          <div className="mb-2.5 flex items-center gap-1.5">
            <span className="slate text-ink3">Length</span>
            <Hint text="Drag the handles at each end of the strip under the picture, or play to a frame and press I or O. The out frame is kept." />
          </div>
          <div className="grid grid-cols-2 gap-2">
            {point("In", draft.inFrame, "I", () => setPrepare(withIn(draft, snap.frame, total)))}
            {point("Out", draft.outFrame, "O", () => setPrepare(withOut(draft, snap.frame, total)))}
          </div>
          <p className="mt-3 flex items-baseline justify-between gap-3 text-[14px] text-ink">
            <span>
              <span className="mono">{secs(count, fps)}</span>
              <span className="text-ink3"> · {count} frames</span>
            </span>
            <span className="mono text-[12px] text-ink3">of {secs(total, fps)}</span>
          </p>
          {longer ? (
            <p className="mt-2 flex items-start gap-1.5 text-[13px] leading-snug text-ink3">
              <span>Each run changes up to {runSeconds} s. On a long shot, drag the bracket to choose the part.</span>
              <Hint
                className="mt-px"
                text={`That's ${maxFrames} frames at ${Math.round(fps * 1000) / 1000} fps. Remove a person, Replace a person, Change the weather and Previz to final use the bracket. Remove a person on a locked-off shot can do up to 10 s.`}
              />
            </p>
          ) : null}
        </section>

        <section aria-label="Shape">
          <div className="mb-2.5 flex items-center gap-1.5">
            <span className="slate text-ink3">Shape</span>
            <Hint text="The new clip is cut to this shape. Drag the frame on the picture to place it, and drag a corner to resize it. Takes and deliveries keep the shape." />
          </div>
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Shape">
            {SHAPES.map((a) => {
              const on = a === draft.aspect;
              return (
                <button
                  key={a}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  onClick={() => setPrepare({ aspect: a, box: a === "original" ? null : fullCrop(shot.width, shot.height, a) })}
                  className={cx(
                    "mono inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-[11px] transition-colors duration-150",
                    on ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
                  )}
                >
                  {a === "original" ? <FrameIcon aspect={`${shot.width}:${shot.height}`} /> : <FrameIcon aspect={a} />}
                  {a === "original" ? "Original" : a}
                </button>
              );
            })}
          </div>
          <p className="mt-3 text-[14px] text-ink">
            <span className="text-ink3">New clip </span>
            <span className="mono">{size}</span>
          </p>
        </section>

        {draft.error ? <Notice tone="bad" stack text={draft.error} /> : null}
      </RailBody>
      <RailFooter>
        <Button variant="primary" size="lg" block loading={draft.busy} disabled={!trimmed && !cropped} onClick={() => void makePreparedClip()}>
          Make clip
        </Button>
        {!trimmed && !cropped && !draft.busy ? <p className="text-center text-[13px] leading-snug text-ink3">Trim or pick a shape to make a clip.</p> : null}
        <Button variant="ghost" size="md" block disabled={draft.busy} onClick={() => skipPrepare(shot.id)}>
          {draft.fromMenu ? "Cancel" : "Skip"}
        </Button>
      </RailFooter>
    </>
  );
}

// --- in and out on the filmstrip ---------------------------------------------------------------------------------

/** The part kept, bracketed on the filmstrip (the rest dimmed), with a handle at each end to drag. */
export function TrimMarks({ shot, frames, track }: { shot: Shot; frames: number; track: RefObject<HTMLDivElement> }) {
  const draft = useDraft(shot.id);
  const player = usePlayer();
  const grab = useRef<"in" | "out" | null>(null);
  const [active, setActive] = useState<"in" | "out" | null>(null);
  if (!draft) return null;
  const edge = (f: number) => `${(Math.max(0, Math.min(frames, f)) / frames) * 100}%`;
  const at = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return Math.max(0, Math.min(frames - 1, Math.floor(((clientX - r.left) / Math.max(1, r.width)) * frames)));
  };
  const move = (which: "in" | "out", clientX: number) => {
    const d = getState().prepare;
    if (!d) return;
    const f = at(clientX);
    const next = which === "in" ? withIn(d, f, frames) : withOut(d, f, frames);
    setPrepare(next);
    player.seek(which === "in" ? next.inFrame : next.outFrame);
  };
  const handle = (which: "in" | "out") => ({
    onPointerDown: (e: ReactPointerEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation(); // not a scrub
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      grab.current = which;
      setActive(which);
      if (player.getSnapshot().playing) player.pause();
      move(which, e.clientX);
    },
    onPointerMove: (e: ReactPointerEvent) => {
      if (grab.current !== which) return;
      e.stopPropagation();
      move(which, e.clientX);
    },
    onPointerUp: (e: ReactPointerEvent) => {
      if (grab.current !== which) return;
      e.stopPropagation();
      grab.current = null;
      setActive(null);
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    },
  });
  const end = draft.outFrame + 1;
  return (
    <>
      <div className="pointer-events-none absolute inset-y-0 left-0 bg-black/65" style={{ width: edge(draft.inFrame) }} />
      <div className="pointer-events-none absolute inset-y-0 right-0 bg-black/65" style={{ left: edge(end) }} />
      <div className="pointer-events-none absolute inset-y-0 border-y-2 border-ivory" style={{ left: edge(draft.inFrame), width: `calc(${edge(end)} - ${edge(draft.inFrame)})` }} />
      {(["in", "out"] as const).map((w) => (
        <div
          key={w}
          role="slider"
          tabIndex={0}
          aria-label={w === "in" ? "In point" : "Out point"}
          aria-valuemin={0}
          aria-valuemax={frames - 1}
          aria-valuenow={w === "in" ? draft.inFrame : draft.outFrame}
          aria-valuetext={timecode(w === "in" ? draft.inFrame : draft.outFrame, shot.fps)}
          onKeyDown={(e) => {
            const dir = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
            if (!dir) return;
            e.preventDefault();
            e.stopPropagation();
            const d = getState().prepare;
            if (!d) return;
            const step = dir * (e.shiftKey ? Math.max(1, Math.round(shot.fps)) : 1);
            const next = w === "in" ? withIn(d, d.inFrame + step, frames) : withOut(d, d.outFrame + step, frames);
            setPrepare(next);
            player.seek(w === "in" ? next.inFrame : next.outFrame);
          }}
          className={cx(
            "group/h absolute inset-y-0 z-[2] w-4 cursor-ew-resize touch-none outline-none",
            w === "in" ? "-translate-x-[2px]" : "-translate-x-[calc(100%-2px)]",
          )}
          style={{ left: edge(w === "in" ? draft.inFrame : end) }}
          {...handle(w)}
        >
          <span
            className={cx(
              "absolute inset-y-0 w-2.5 bg-ivory shadow-[0_0_0_1px_#000] transition-[width] duration-150 group-hover/h:w-3 group-focus-visible/h:w-3",
              w === "in" ? "left-0 rounded-l-[2px]" : "right-0 rounded-r-[2px]",
              active === w && "w-3",
            )}
          >
            <span className="absolute inset-y-3 left-1/2 w-px -translate-x-1/2 bg-black/60" />
          </span>
        </div>
      ))}
    </>
  );
}

// --- the crop frame over the picture --------------------------------------------------------------------------------

type Corner = "nw" | "ne" | "sw" | "se";

/** The crop frame: locked to the chosen shape, darkened outside. Drag inside to move it, a corner to resize it. */
export function CropLayer({ shot, width, height }: { shot: Shot; width: number; height: number }) {
  const draft = useDraft(shot.id);
  const drag = useRef<{ mode: "move" | Corner; x: number; y: number; box: NonNullable<PrepareDraft["box"]> } | null>(null);
  if (!draft || draft.aspect === "original" || !draft.box || !width || !height) return null;
  const W = shot.width;
  const H = shot.height;
  const k = width / W; // screen px per source px
  const r = cropRatio(draft.aspect);
  const b = draft.box;
  const even = (v: number) => Math.max(2, Math.round(v / 2) * 2);

  const apply = (clientX: number, clientY: number) => {
    const g = drag.current;
    if (!g) return;
    const dx = (clientX - g.x) / k;
    const dy = (clientY - g.y) / k;
    const o = g.box;
    if (g.mode === "move") {
      const x = Math.max(0, Math.min(W - o.w, o.x + dx));
      const y = Math.max(0, Math.min(H - o.h, o.y + dy));
      setPrepare({ box: { ...o, x: Math.floor(x / 2) * 2, y: Math.floor(y / 2) * 2 } });
      return;
    }
    // A corner: the opposite one stays put; the size follows the pointer, in the chosen shape, inside the frame.
    const sx = g.mode.includes("e") ? 1 : -1;
    const sy = g.mode.includes("s") ? 1 : -1;
    const ax = sx > 0 ? o.x : o.x + o.w;
    const ay = sy > 0 ? o.y : o.y + o.h;
    let w = Math.max(o.w + sx * dx, (o.h + sy * dy) * r);
    const maxW = Math.min(sx > 0 ? W - ax : ax, (sy > 0 ? H - ay : ay) * r);
    const minW = Math.max(MIN_CROP, MIN_CROP * r);
    w = Math.max(Math.min(w, maxW), Math.min(minW, maxW));
    const h = w / r;
    const ew = even(Math.min(w, maxW));
    const eh = Math.min(even(h), sy > 0 ? H - ay : ay);
    setPrepare({ box: { w: ew, h: eh, x: Math.max(0, Math.round((sx > 0 ? ax : ax - ew) / 2) * 2), y: Math.max(0, Math.round((sy > 0 ? ay : ay - eh) / 2) * 2) } });
  };
  const start = (mode: "move" | Corner) => (e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { mode, x: e.clientX, y: e.clientY, box: b };
  };
  const track = {
    onPointerMove: (e: ReactPointerEvent) => {
      if (!drag.current) return;
      e.stopPropagation();
      apply(e.clientX, e.clientY);
    },
    onPointerUp: (e: ReactPointerEvent) => {
      if (!drag.current) return;
      drag.current = null;
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    },
    onPointerCancel: () => {
      drag.current = null;
    },
  };

  return (
    <div
      role="group"
      aria-label={`Crop frame, ${draft.aspect}`}
      tabIndex={0}
      className="absolute z-[1] cursor-move touch-none shadow-[0_0_0_1px_#EEE9E0,0_0_0_2px_rgba(0,0,0,.75),0_0_0_9999px_rgba(0,0,0,.62)] outline-none focus-visible:shadow-[0_0_0_2px_#EEE9E0,0_0_0_3px_rgba(0,0,0,.75),0_0_0_9999px_rgba(0,0,0,.62)]"
      style={{ left: b.x * k, top: b.y * k, width: b.w * k, height: b.h * k }}
      onClick={(e) => e.stopPropagation()}
      onPointerDown={start("move")}
      onKeyDown={(e) => {
        const dx = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
        const dy = e.key === "ArrowUp" ? -1 : e.key === "ArrowDown" ? 1 : 0;
        if (!dx && !dy) return;
        e.preventDefault();
        e.stopPropagation();
        const step = e.shiftKey ? Math.round(Math.max(W, H) / 20 / 2) * 2 : 2;
        setPrepare({ box: { ...b, x: Math.max(0, Math.min(W - b.w, b.x + dx * step)), y: Math.max(0, Math.min(H - b.h, b.y + dy * step)) } });
      }}
      {...track}
    >
      {/* Thirds, faint, to place the frame by. */}
      <span className="pointer-events-none absolute inset-y-0 left-1/3 w-px bg-ivory/25" />
      <span className="pointer-events-none absolute inset-y-0 left-2/3 w-px bg-ivory/25" />
      <span className="pointer-events-none absolute inset-x-0 top-1/3 h-px bg-ivory/25" />
      <span className="pointer-events-none absolute inset-x-0 top-2/3 h-px bg-ivory/25" />
      {(["nw", "ne", "sw", "se"] as const).map((c) => (
        <span
          key={c}
          aria-hidden="true"
          onPointerDown={start(c)}
          {...track}
          className={cx(
            "absolute h-5 w-5 touch-none",
            c === "nw" && "-left-2.5 -top-2.5 cursor-nwse-resize",
            c === "se" && "-bottom-2.5 -right-2.5 cursor-nwse-resize",
            c === "ne" && "-right-2.5 -top-2.5 cursor-nesw-resize",
            c === "sw" && "-bottom-2.5 -left-2.5 cursor-nesw-resize",
          )}
        >
          <span className="absolute left-1/2 top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-[1px] bg-ivory shadow-[0_0_0_1px_#000]" />
        </span>
      ))}
      <span className="slate pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[2px] bg-black/70 px-1.5 py-[3px] text-ivory/85">
        {draft.aspect} · {b.w}×{b.h}
      </span>
    </div>
  );
}

// --- source limits, where clips are added ----------------------------------------------------------------------------

/** The "?" beside "Drop a clip": what Pickups takes (the numbers come from the server). */
export function SourceLimits({ className, onMedia }: { className?: string; onMedia?: boolean }) {
  const st = useApp((s) => s.status);
  const maxFrames = st?.max_frames ?? 97;
  const maxLong = st?.max_long ?? 3840;
  const min = st?.min_frames ?? 9;
  const four = maxLong >= 3840 ? "4K" : `${maxLong} px`;
  const text = (
    <span className="block space-y-1.5">
      <span className="block">
        <b className="font-medium text-ivory">Files</b> MOV, MP4, M4V, MKV, AVI, MXF or WebM.
      </span>
      <span className="block">
        <b className="font-medium text-ivory">Length</b> Any, from {min} frames. Each run changes up to {maxFrames} frames ({(maxFrames / 24).toFixed(1)} s at 24
        fps). On a long shot, drag the bracket to choose the part.
      </span>
      <span className="block">
        <b className="font-medium text-ivory">Size</b> Any. Work is done at up to {four}, so 6K and 8K are scaled down. Remove a person and Replace a person return your size up
        to {four}; Change the weather and Previz to final return HD (Finish makes {four}).
      </span>
      <span className="block">
        <b className="font-medium text-ivory">Shape</b> Any. Takes keep it. Previz to final makes 16:9, and Change the weather can trim up to 8% off two edges. Crop for another shape.
      </span>
      <span className="block">
        <b className="font-medium text-ivory">Frame rate</b> Kept.
      </span>
    </span>
  );
  return (
    <span className={cx("inline-flex items-center text-[13px]", onMedia ? "text-ivory/60" : "text-ink3", className)}>
      <Hint text={text} label="What clips work" />
    </span>
  );
}
