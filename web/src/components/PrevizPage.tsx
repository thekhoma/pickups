// Previz: a gray 3D layout in the browser. Make shot turns it into a shot and opens Previz to final.
import { ChevronRight, Maximize, Pause, Play } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { loadShots, newPrevizScene, openMadePreviz, openShot, takePrevizSentence, takePrevizSpec, targetProject } from "../actions";
import { cx } from "../lib/format";
import { curveOf, dollyPose, keyTime, lookPose, orbitPose, panLook, truckPose, walkPose, type CameraPose } from "../previz/cameraMath";
import { aimPoint, canRemove, insertPoint, keyAtTime, MAX_KEYS, movePath, movePoint, removePoint, setViewAt } from "../previz/pathEdit";
import { openedSpec } from "../previz/remake";
import { CAMERA_PRESETS, compile, PRESETS, type CameraPick, type Compiled } from "../previz/compile";
import { ensureShotLexicon, onShotLibrary, readShot, shotGrammar, shotLibraryFailed } from "../previz/shotLex";
import { ExportCanceled, exportPreviz } from "../previz/export";
import { isPlain, LIBRARY_FAILED, plainError } from "../previz/errors";
import { castIn, lookPromptFor } from "../previz/lookPrompt";
import { loneCar, motionPromptFor } from "../previz/motionPrompt";
import { insertFragment, tryFragments, type TryFragment } from "../previz/tryRow";
import { PrevizRuntime, type TopHit } from "../previz/runtime";
import { askSaveTo, SaveFolderRow } from "./SaveTo";
import { previzGate, previzMemory, rememberPreviz, setPrevizRendering, subscribePreviz } from "../previz/home";
import { shotFrames, PREVIZ_FPS, PREVIZ_MAX_S, type PrevizSpec } from "../previz/spec";
import { useReducedMotion } from "../lib/loops";
import { useMedia } from "../lib/hooks";
import { getState, useApp } from "../store";
import type { Shot } from "../types";
import { CreateComposer } from "./CreateBar";
import { Button, Hint, IconButton, Notice, ProgressLine, StepHeader, Tag } from "./ui";

/** Stop the ivory line at the middle of the 15 px knob. */
function fillOf(value: number, min: number, max: number): string {
  const span = max - min;
  const t = span <= 0 ? 0 : Math.min(1, Math.max(0, (value - min) / span));
  return `calc(${t} * (100% - 15px) + 7.5px)`;
}

function formatTune(n: number, digits: number): string {
  const p = 10 ** digits;
  const rounded = Math.round(n * p) / p;
  return digits === 0 ? String(rounded) : rounded.toFixed(digits);
}

/** One control for every camera setting: a typed number and a slider that share it. */
function Tune({
  label,
  hint,
  value,
  min,
  max,
  step,
  digits,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits: number;
  onChange: (n: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = Number.isFinite(value) ? value : min;
  const commit = (raw: string) => {
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    onChange(Math.min(max, Math.max(min, n)));
  };
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[11px] text-ink3">
          <span>{label}</span>
          {hint ? <Hint text={hint} /> : null}
        </span>
        <input
          type="text"
          inputMode="decimal"
          aria-label={label}
          className="h-7 w-16 rounded-card border border-line bg-bg px-1.5 text-right font-mono text-[12px] text-ink outline-none hover:border-line2 focus:border-ivory/60"
          value={draft ?? formatTune(shown, digits)}
          onFocus={(e) => {
            setDraft(formatTune(shown, digits));
            e.target.select();
          }}
          onChange={(e) => {
            const raw = e.target.value;
            setDraft(raw);
            const n = Number(raw);
            if (Number.isFinite(n) && n >= min && n <= max) onChange(n);
          }}
          onBlur={() => {
            if (draft != null) commit(draft);
            setDraft(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      </div>
      <input
        type="range"
        className="range w-full"
        aria-label={`${label} slider`}
        min={min}
        max={max}
        step={step}
        value={shown}
        style={{ ["--fill" as string]: fillOf(shown, min, max) }}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
      />
    </div>
  );
}

/**
 * A new compile of the sentence, keeping a camera the user drew (a bent path, or start and end) while the cast is the
 * same people and vehicles. A different cast drops it (the caller says so, with Undo).
 */
export function keepPath(prev: Compiled, next: Compiled): { next: Compiled; dropped: boolean } {
  const cam = prev.spec.camera;
  if (!cam.free) return { next, dropped: false };
  const cast = (s: PrevizSpec) => s.subjects.map((x) => x.kind).join(",");
  if (cast(prev.spec) !== cast(next.spec)) return { next, dropped: true };
  return { next: { ...next, camera: prev.camera, spec: { ...next.spec, lens: prev.spec.lens, camera: structuredClone(cam) } }, dropped: false };
}

/** Top view's scale: a bar of a round number of metres, at the bottom left of the map. */
function ScaleBar({ mpp, split, canvas, scale }: { mpp: number; split: boolean; canvas: RefObject<HTMLCanvasElement>; scale: () => number }) {
  const [box, setBox] = useState<{ left: number; top: number; w: number; h: number } | null>(null);
  useLayoutEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const fit = () => setBox({ left: el.offsetLeft, top: el.offsetTop, w: el.clientWidth, h: el.clientHeight });
    fit();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(fit) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [canvas]);
  if (!box) return null;
  const now = scale() || mpp;
  const metres = [0.5, 1, 2, 5, 10, 20, 50, 100, 200].filter((m) => m / now <= 120).pop() ?? 0.5;
  return (
    <div
      className="pointer-events-none absolute z-10 flex flex-col items-start gap-1"
      style={{ left: box.left + (split ? Math.floor(box.w / 2) : 0) + 12, top: box.top + box.h - 34 }}
      aria-label="Map scale"
      data-metres={metres}
    >
      <span className="slate text-ivory/80">{metres} m</span>
      <span className="block h-[4px] border-x border-b border-ivory/80" style={{ width: Math.round(metres / now) }} />
    </div>
  );
}

const LOOP_W = 480;
const LOOP_H = 272;
const LOOP_FRAMES = 16;

function SceneLoop({ frames }: { frames?: ImageBitmap[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !frames?.length) return;
    let raf = 0;
    const start = performance.now();
    const draw = (now: number) => {
      const i = reduced || frames.length < 2 ? 0 : Math.floor(((now - start) / 1000) * 8) % frames.length;
      const f = frames[i];
      // a bitmap closed when the gate let go of its tiles has no size: skip it
      if (f && f.width) ctx.drawImage(f, 0, 0, canvas.width, canvas.height);
      if (!reduced && frames.length > 1) raf = requestAnimationFrame(draw);
    };
    draw(start);
    return () => cancelAnimationFrame(raf);
  }, [frames, reduced]);
  return <canvas ref={ref} width={LOOP_W} height={LOOP_H} className={cx("h-full w-full", frames?.length ? "bg-black" : "bg-raised")} aria-hidden="true" />;
}

type TileJob = { id: string; spec: PrevizSpec; done: (frames: ImageBitmap[]) => void };

/** Resolves once the page has painted and gone quiet, so the sentence box answers before any tile builds. */
function quiet(): Promise<void> {
  return new Promise((resolve) => {
    const idle = (window as Window & { requestIdleCallback?: (fn: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
    const after = () => (idle ? idle(() => resolve(), { timeout: 1500 }) : window.setTimeout(resolve, 300));
    window.setTimeout(after, 1200);
  });
}

/** One offscreen renderer bakes the starter tiles one at a time: only tiles that are on screen or hovered, and only
 * after the gate is idle. A hovered tile jumps the queue. */
function useTileBaker(): (job: TileJob, urgent?: boolean) => void {
  const reduced = useReducedMotion();
  const api = useRef<(job: TileJob, urgent?: boolean) => void>(() => undefined);
  useEffect(() => {
    let dead = false;
    let rt: PrevizRuntime | null = null;
    const queue: TileJob[] = [];
    const asked = new Set<string>();
    const made: ImageBitmap[] = [];
    let running = false;
    const ready = quiet();
    const stats = ((window as Window & { __previzTiles?: { firstStart: number | null; built: number } }).__previzTiles ??= { firstStart: null, built: 0 });
    const pump = async () => {
      if (running) return;
      running = true;
      await ready;
      while (queue.length && !dead) {
        const job = queue.shift()!;
        if (!rt) {
          try {
            rt = new PrevizRuntime(document.createElement("canvas"));
          } catch {
            break;
          }
        }
        if (stats.firstStart == null) stats.firstStart = performance.now();
        const n = Math.max(1, shotFrames(job.spec.duration_s, job.spec.fps, PREVIZ_MAX_S * PREVIZ_FPS));
        const count = reduced ? 1 : LOOP_FRAMES;
        const frames: ImageBitmap[] = [];
        await rt.setSpec(job.spec, 0, [LOOP_W, LOOP_H]);
        for (let i = 0; i < count && !dead; i++) {
          // Fresh shadows on every baked frame: a still scene doesn't redraw them, and the camera moves between frames.
          rt.renderer.shadowMap.needsUpdate = true;
          rt.renderFrame(Math.round((i / Math.max(1, count - 1)) * (n - 1)));
          const bitmap = await createImageBitmap(rt.renderer.domElement);
          if (dead) {
            bitmap.close();
            break;
          }
          frames.push(bitmap);
          made.push(bitmap);
          if (i === 0) job.done(frames.slice());
          // Give typing a turn between frames.
          await new Promise((r) => window.setTimeout(r, 0));
        }
        if (!dead) {
          job.done(frames.slice());
          stats.built += 1;
        }
      }
      running = false;
    };
    api.current = (job, urgent) => {
      if (asked.has(job.id) || dead) return;
      asked.add(job.id);
      if (urgent) queue.unshift(job);
      else queue.push(job);
      void pump();
    };
    return () => {
      dead = true;
      rt?.dispose();
      for (const bitmap of made) bitmap.close();
    };
  }, [reduced]);
  return (job, urgent) => api.current(job, urgent);
}

/** The camera chip for an example's move tag ("Push in from behind" is a push, "Pull wide" a dolly out). */
function chipForMove(move: string): CameraPick | null {
  const m = move.toLowerCase();
  if (/close to wide/.test(m)) return "close-wide";
  if (/pull|dolly out/.test(m)) return "dolly-out";
  if (/push|dolly in/.test(m)) return "dolly-in";
  if (/orbit|circle/.test(m)) return "orbit";
  if (/fpv/.test(m)) return "fpv";
  if (/track|follow/.test(m)) return "track";
  if (/parallel/.test(m)) return "parallel";
  if (/drone|aerial/.test(m)) return "drone";
  if (/locked|static|hold/.test(m)) return "locked";
  if (/close/.test(m)) return "close";
  if (/wide/.test(m)) return "wide";
  return null;
}

/** A finished example: its name, the sentence it was made from, its camera move, and the spec its tile is baked from. */
type PrevizExample = { id: string; label: string; move: string; text: string; spec: PrevizSpec };

function SceneTile({ preset, bake, onPick }: { preset: PrevizExample; bake: (job: TileJob, urgent?: boolean) => void; onPick: (example: PrevizExample) => void }) {
  const ref = useRef<HTMLButtonElement>(null);
  const [frames, setFrames] = useState<ImageBitmap[]>();
  const ask = (urgent = false) => bake({ id: preset.id, spec: preset.spec, done: setFrames }, urgent);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      ask();
      return;
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        ask();
        io.disconnect();
      }
    });
    io.observe(el);
    return () => io.disconnect();
    // one observer per tile
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset.id]);
  return (
    <button
      ref={ref}
      type="button"
      onClick={() => onPick(preset)}
      onPointerEnter={() => ask(true)}
      onFocus={() => ask(true)}
      className="overflow-hidden rounded-card border border-line bg-raised text-left transition-colors hover:border-line2"
    >
      <span className="relative block" style={{ aspectRatio: "1920 / 1088" }}>
        <SceneLoop frames={frames} />
        <Tag className="pointer-events-none absolute left-2 top-2 border-white/20 bg-black/55 text-ivory/85">{preset.move}</Tag>
        <span className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 to-transparent px-3 pb-2 pt-8">
          <span className="block text-[13px] leading-tight text-ivory">{preset.label}</span>
          <span className="mt-0.5 block truncate text-[11px] leading-tight text-ink2" title={preset.text}>{preset.text}</span>
        </span>
      </span>
    </button>
  );
}

/** The hero example: the last tile of the Examples grid. */
const WINDOW_SEAT = {
  id: "window-seat",
  label: "Window Seat",
  text: "A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and slowly pushes in, and at the end he turns and looks at the camera.",
};

/** "Your previz": previz shots in the current project, newest first, each distinct shot once. Hidden until there's one. */
function RecentPreviz() {
  const target = useApp((s) => (s.currentProjectId && s.currentProjectId !== "examples" ? s.currentProjectId : "unsorted"));
  const ids = useApp((s) => s.projectShots[target]);
  const shots = useApp((s) => s.shots);
  useEffect(() => {
    void loadShots(target);
  }, [target]);
  const when = (sh: Shot) => (typeof sh.created_at === "number" ? sh.created_at : Date.parse(String(sh.created_at ?? "")) / 1000 || 0);
  const list = (ids ?? [])
    .map((id) => shots[id])
    .filter((sh): sh is Shot => !!sh?.previz_spec)
    .sort((a, b) => when(b) - when(a))
    .filter((sh, i, all) => {
      const key = `${sh.name}\n${JSON.stringify(sh.previz_spec)}`;
      return all.findIndex((o) => `${o.name}\n${JSON.stringify(o.previz_spec)}` === key) === i;
    })
    .slice(0, 12);
  if (!list.length) return null;
  return (
    <section className="mb-8 w-full" aria-label="Your previz">
      <h2 className="slate mb-3 text-ink3">Your previz</h2>
      <div className="flex gap-3 overflow-x-auto pb-1 [scrollbar-width:thin]">
        {list.map((sh) => (
          <button
            key={sh.id}
            type="button"
            onClick={() => openShot(sh.id, { rail: { kind: "form", task: "previz" } })}
            className="w-[208px] shrink-0 overflow-hidden rounded-card border border-line bg-panel text-left transition-colors hover:border-line2 max-[640px]:w-[160px]"
          >
            <img src={sh.latest_thumb_url || sh.thumb_url} alt="" className="aspect-video w-full bg-black object-cover" draggable={false} />
            <span className="block truncate px-3 py-2 text-[13px] text-ink">{sh.name}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

/** The Examples grid: each starter scene and Window Seat, compiled exactly as the editor opens them (12 tiles, so every
 * column count, 4, 3 or 2, ends on a full row). */
function useExamples(lexReady: boolean): PrevizExample[] {
  return useMemo(() => {
    if (!lexReady) return [];
    return [...PRESETS, WINDOW_SEAT].map((preset) => {
      const compiled = compile(preset.text);
      return { ...preset, move: preset.id === WINDOW_SEAT.id ? "Push in" : compiled.camera, spec: compiled.spec };
    });
  }, [lexReady]);
}

function PrevizGate({ onPreset, padBottom, lexReady }: { onPreset: (example: PrevizExample) => void; padBottom: number; lexReady: boolean }) {
  const examples = useExamples(lexReady);
  const bake = useTileBaker();
  return (
    // The fade over the dock is 48 px taller than the dock: the last row scrolls clear of both.
    <div className="h-full overflow-y-auto" style={{ paddingBottom: padBottom + 48 }}>
      <div className="mx-auto flex w-full max-w-[1180px] flex-col items-center px-6 pt-8 max-[640px]:px-3 max-[640px]:pt-4">
        <h1 className="sr-only">Previz</h1>
        {/* Your own previz first: the shots you came back for never sit under the docked composer. */}
        <RecentPreviz />
        <h2 className="slate mb-3 flex items-center gap-2 self-start text-ink3">
          Examples
        </h2>
        <div className="grid w-full grid-cols-4 gap-3 max-[900px]:grid-cols-3 max-[640px]:grid-cols-2">
          {examples.map((preset) => (
            <SceneTile key={preset.id} preset={preset} bake={bake} onPick={onPreset} />
          ))}
        </div>
      </div>
    </div>
  );
}

/** REVIEW F5: a huge paste built a 30-figure rooftop; the sentence is capped. */
const PREVIZ_MAX_CHARS = 500;

/** The New previz tab's body in the shared composer: the sentence, the "Try:" row and one primary button. */
function PrevizBody({
  value,
  onChange,
  onSubmit,
  submitLabel,
  disabled,
  areaRef,
  fragments,
  onFragment,
  notice,
  example = null,
  compact = false,
  understood = null,
  primary = true,
  go = false,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  submitLabel: string;
  disabled?: boolean;
  areaRef: RefObject<HTMLTextAreaElement>;
  fragments: TryFragment[];
  onFragment: (text: string) => void;
  notice: string | null;
  /** The example the editor opened, while its sentence is unchanged: the Example tag and the Remix hint. */
  example?: { label: string } | null;
  /** A phone's collapsed dock: one line and the button, nothing else (it opens up when the field is focused). */
  compact?: boolean;
  /** REVIEW F5: what the built scene understood of the words (chips), and the words it didn't use (dimmed). */
  understood?: { matched: string[]; missed: string[]; camera: string } | null;
  /** Build is the primary until there is a scene; then Make shot is, and Rebuild steps back (REVIEW F4). */
  primary?: boolean;
  /** The camera move changed since the last build: Rebuild turns green. */
  go?: boolean;
}) {
  return (
    <form
      className={compact ? "flex items-center gap-2 p-2" : "p-4 max-[640px]:p-3"}
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      {compact ? null : (
        <div className="mb-2 flex items-center gap-1.5">
          <span className="text-[14px] font-medium text-ivory">Describe the shot</span>
          <Hint text="Who is where, and how the camera moves. Build turns it into a 3D layout you can adjust. The words also guide the final look." />
          {example ? (
            <span className="ml-auto flex items-center gap-2">
              <Tag>Example</Tag>
              <span className="text-[13px] text-ink2">Remix</span>
              <Hint text="Edit the sentence, then Rebuild: that makes a new shot of your own. The example never changes." />
            </span>
          ) : null}
        </div>
      )}
      {/* One textarea in both states (so focusing it to open the dock keeps the caret). */}
      <textarea
        ref={areaRef}
        rows={compact ? 1 : 2}
        value={value}
        disabled={disabled}
        maxLength={PREVIZ_MAX_CHARS}
        onChange={(e) => onChange(e.target.value.slice(0, PREVIZ_MAX_CHARS))}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="A man sitting alone in a diner at night. The camera slowly pushes in."
        aria-label="Describe the shot"
        className={cx("field resize-none text-[16px] leading-relaxed", compact ? "h-10 min-w-0 flex-1 overflow-hidden whitespace-nowrap py-2" : "h-auto min-h-[64px] py-3")}
      />
      {compact ? (
        <Button type="submit" variant="primary" size="md" disabled={disabled || !value.trim()} className="shrink-0">
          {submitLabel}
        </Button>
      ) : null}
      {notice && !compact ? <Notice tone="bad" text={notice} className="mt-3" /> : null}
      {understood && !compact ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5" aria-label="Understood">
          {[...understood.matched, understood.camera].filter(Boolean).map((w, i) => (
            <span key={`m${i}`} className="inline-flex h-6 items-center rounded-full border border-line2 px-2.5 text-[12px] text-ink">
              {w}
            </span>
          ))}
          {value.length >= PREVIZ_MAX_CHARS ? <span className="slate text-ink3">{PREVIZ_MAX_CHARS} characters max</span> : null}
        </div>
      ) : null}
      {/* Words the kit has no piece for, said in words (not crossed out): the final shot's Look still carries them. */}
      {understood?.missed.length && !compact ? (
        <p className="mt-1.5 text-[13px] text-ink3" data-testid="previz-missed">
          Not in the previz kit: <span className="text-ink2">{understood.missed.join(", ")}</span>
        </p>
      ) : null}
      <div className={cx("mt-3 flex items-center gap-3 max-[640px]:flex-wrap", compact && "hidden")}>
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2 max-[640px]:basis-full max-[640px]:flex-nowrap max-[640px]:overflow-x-auto max-[640px]:[scrollbar-width:none]" role="group" aria-label="Try">
          {fragments.length ? <span className="slate shrink-0 text-ink3">Try:</span> : null}
          {fragments.map((f) => (
            <Button
              key={f.text}
              className="shrink-0"
              variant="secondary"
              size="sm"
              disabled={disabled}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => onFragment(f.text)}
            >
              {f.text}
            </Button>
          ))}
        </div>
        {primary ? null : (
          <Hint text="Rebuild makes the scene again from your words, with the camera move you picked. The scene already follows your edits; use Rebuild to start the layout fresh." />
        )}
        <Button type="submit" variant={go ? "go" : primary ? "primary" : "secondary"} size="lg" disabled={disabled || !value.trim()} className="shrink-0 max-[640px]:w-full">
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}

/** The camera's keyframes as marks on the timeline: click one to go to its frame; a selected middle one can go (x). */
function KeyMarks({ frames, keyFrames, selected, disabled, onPick, onRemove, canRemove }: {
  frames: number;
  keyFrames: number[];
  selected: number | null;
  disabled: boolean;
  onPick: (index: number, frame: number) => void;
  onRemove: (index: number) => void;
  canRemove: (index: number) => boolean;
}) {
  const span = Math.max(1, frames - 1);
  return (
    <div className="pointer-events-none absolute inset-x-0 -top-3 h-3" aria-label="Keyframes" role="group">
      {keyFrames.map((f, i) => {
        const on = selected === i;
        return (
          <span key={i} className="absolute top-0 -translate-x-1/2" style={{ left: `calc(${(f / span) * 100}% + ${(0.5 - f / span) * 14}px)` }}>
            <button
              type="button"
              disabled={disabled}
              aria-label={`Keyframe ${i + 1}, frame ${f + 1}`}
              aria-pressed={on}
              data-testid="keyframe-mark"
              onClick={() => onPick(i, f)}
              className={cx("pointer-events-auto block h-2.5 w-2.5 rotate-45 border transition-colors", on ? "border-ivory bg-ivory" : "border-ivory/70 bg-bg hover:bg-ivory/60")}
            />
            {on && canRemove(i) ? (
              <button
                type="button"
                disabled={disabled}
                aria-label={`Delete keyframe ${i + 1}`}
                onClick={() => onRemove(i)}
                className="pointer-events-auto absolute -top-6 left-1/2 inline-flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border border-line2 bg-raised text-[12px] leading-none text-ink hover:text-ivory"
              >
                ×
              </button>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

const PREVIEW_ASPECT = 1920 / 1088;
const PREVIEW_ASPECT_CSS = "1920 / 1088";

/** The biggest 1920 x 1088 box (at most 1600 wide) that fits w x h: the preview is never stretched. */
export function fitPreview(w: number, h: number, maxW = 1600): { w: number; h: number } {
  const width = Math.max(1, Math.floor(Math.min(w, maxW, h * PREVIEW_ASPECT)));
  return { w: width, h: Math.round(width / PREVIEW_ASPECT) };
}

export function PrevizPage() {
  const cap = useApp((s) => s.status?.previz_frames) ?? 97;
  const previewCap = PREVIZ_MAX_S * PREVIZ_FPS;

  const memory = previzMemory();
  // A shot's saved spec handed over from Previz to final (a path the user drew): the editor opens on it.
  const [handed] = useState(() => openedSpec(takePrevizSpec()));
  const [line, setLine] = useState(handed ? handed.notes || "" : memory.line);
  const [cam, setCam] = useState<CameraPick | null>(null);
  const [lensUserSet, setLensUserSet] = useState(false);
  const [built, setBuilt] = useState<Compiled>(() =>
    handed ? { ...compile(handed.notes || "a person standing in a park"), spec: handed } : memory.built ?? compile(memory.line || "a person standing in a park"),
  );
  const [guardNotes, setGuardNotes] = useState<string[]>([]);
  const spec = built.spec;
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [canceled, setCanceled] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [posePhase, setPosePhase] = useState<"start" | "end" | null>(null);
  const [pathFlash, setPathFlash] = useState<string | null>(null);
  const flashTimer = useRef<number | undefined>(undefined);
  const [aiming, setAiming] = useState(false);
  // The camera move was changed (a chip, the path, W A S D) since the scene was last built: Rebuild turns green.
  const [moveChanged, setMoveChanged] = useState(false);
  const [opened, setOpened] = useState(() => handed != null || memory.built != null);
  const [draft, setDraft] = useState(() => takePrevizSentence() ?? "");
  const [lexReady, setLexReady] = useState(false);
  const [libFailed, setLibFailed] = useState(shotLibraryFailed);
  const [dockH, setDockH] = useState(220);
  const rootRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const seenGate = useRef(previzGate());
  const [view, setView] = useState<"eye" | "split" | "top">("eye");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  // The preview keeps its 1920 x 1088 shape inside its box at any window size (a capped height used to stretch it
  // sideways): its width and height come from the box, never from the CSS alone. Phones keep the full-width flow.
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageBox, setStageBox] = useState<{ w: number; h: number } | null>(null);
  useLayoutEffect(() => {
    const box = stageRef.current;
    if (!box || typeof ResizeObserver === "undefined") return;
    const mq = window.matchMedia("(max-width: 767px)");
    const fit = () => {
      if (mq.matches) return setStageBox(null);
      const cs = getComputedStyle(box);
      const w = box.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      const h = box.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
      setStageBox(w > 0 && h > 0 ? fitPreview(w, h) : null);
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(box);
    mq.addEventListener("change", fit);
    return () => {
      ro.disconnect();
      mq.removeEventListener("change", fit);
    };
  }, [opened]);
  const viewRef = useRef(view);
  viewRef.current = view;
  // Top view editing: the selected path point, the map's scale, a drag in progress, and an undo stack of cameras.
  const [topSel, setTopSel] = useState<number | null>(null);
  const topSelRef = useRef<number | null>(null);
  topSelRef.current = topSel;
  const [mpp, setMpp] = useState(0);
  const [pathNote, setPathNote] = useState<{ text: string; undo: Compiled } | null>(null);
  const [pathFull, setPathFull] = useState(false);
  const topDrag = useRef<{ kind: "point" | "turn" | "subject" | "pan" | "still" | "stillTurn"; index: number; last: { x: number; y: number }; moved: boolean; undo: boolean } | null>(null);
  const undoRef = useRef<PrevizSpec["camera"][]>([]);
  const [undoCount, setUndoCount] = useState(0);
  const rtRef = useRef<PrevizRuntime | null>(null);
  const specRef = useRef(spec);
  const frameRef = useRef(frame);
  const capRef = useRef(cap);
  const abortRef = useRef<AbortController | null>(null);
  const pickedRef = useRef<number | null>(null);
  const posePhaseRef = useRef<"start" | "end" | null>(null);
  const draftRef = useRef<CameraPose | null>(null);
  const startPoseRef = useRef<CameraPose | null>(null);
  const keysRef = useRef<Set<string>>(new Set());
  const aimDrag = useRef<{ x: number; y: number; mode: "pan" | "look" | "orbit" | "truck" } | null>(null);
  const downRef = useRef<{ x: number; y: number; hit: boolean; button: number } | null>(null);
  const flyingRef = useRef(false);
  const movedRef = useRef(false);
  const busyRef = useRef(false);
  const grabPoseRef = useRef<() => CameraPose | null>(() => null);
  const lockRef = useRef<() => boolean>(() => false);
  specRef.current = spec;
  const lineRef = useRef(line);
  const camRef = useRef(cam);
  lineRef.current = line;
  camRef.current = cam;
  useEffect(() => onShotLibrary(setLibFailed), []);

  // Room for the docked composer: nothing in the dock covers the transport or Make shot.
  useLayoutEffect(() => {
    const box = rootRef.current?.querySelector<HTMLElement>(".previz-dock > div");
    if (!box || typeof ResizeObserver === "undefined") return;
    const fit = () => setDockH(Math.ceil(box.getBoundingClientRect().height) + 24 + 12);
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(box);
    return () => ro.disconnect();
  }, [opened]);

  useEffect(() => {
    let dead = false;
    ensureShotLexicon().then(() => {
      if (dead) return;
      setLexReady(true);
      setBuilt((cur) => keepPath(cur, compile(lineRef.current, {
        duration_s: cur.spec.duration_s,
        focal_mm: cur.spec.lens.focal_mm,
        camera: camRef.current,
        speed: cur.spec.camera.speed,
        taper: cur.spec.camera.taper,
        distance: cur.spec.camera.distance,
        frame: cur.spec.camera.frame ?? null,
      })).next);
    }).catch(() => setLibFailed(true));
    return () => { dead = true; };
  }, []);
  posePhaseRef.current = posePhase;
  busyRef.current = busy;
  useEffect(() => {
    setPrevizRendering(busy);
    return () => setPrevizRendering(false);
  }, [busy]);
  frameRef.current = frame;
  capRef.current = cap;

  const frames = shotFrames(spec.duration_s, spec.fps, previewCap);
  const keyFrames = useMemo(() => {
    const n = Math.max(2, frames);
    return spec.camera.keys.map((k) => {
      let best = 0;
      let bestD = Infinity;
      for (let f = 0; f < n; f++) {
        const d = Math.abs(keyTime(spec.camera, f / (n - 1)) - k.t);
        if (d < bestD) {
          bestD = d;
          best = f;
        }
      }
      return best;
    });
  }, [spec.camera, frames]);
  // The slider stops at what this GPU renders, so Make shot never refuses a length.
  const maxS = Math.min(PREVIZ_MAX_S, cap / PREVIZ_FPS);
  useEffect(() => {
    if (spec.duration_s > maxS + 1e-9) setBuilt((b) => ({ ...b, spec: { ...b.spec, duration_s: maxS } }));
  }, [maxS, spec.duration_s]);

  useEffect(() => {
    if (opened) rememberPreviz(line, built);
  }, [line, built, opened]);

  useEffect(
    () =>
      subscribePreviz(() => {
        const next = previzGate();
        if (next !== seenGate.current) {
          seenGate.current = next;
          setOpened(false);
          setLine("");
          setDraft("");
          setCam(null);
          setError(null);
          setCanceled(false);
        }
      }),
    [],
  );

  const exitPose = () => {
    posePhaseRef.current = null;
    draftRef.current = null;
    startPoseRef.current = null;
    flyingRef.current = false;
    movedRef.current = false;
    rtRef.current?.setPose(null);
    setPosePhase(null);
    setAiming(false);
  };

  grabPoseRef.current = () => {
    const rt = rtRef.current;
    if (!rt || busyRef.current) return null;
    if (draftRef.current && (posePhaseRef.current || flyingRef.current)) return draftRef.current;
    setPlaying(false);
    const pose = rt.viewPose();
    if (!pose) return null;
    draftRef.current = { pos: [...pose.pos], look: [...pose.look] };
    flyingRef.current = true;
    rt.setPose(draftRef.current);
    setAiming(true);
    return draftRef.current;
  };

  const applyLine = (value: string, camera: CameraPick | null = cam) => {
    if (posePhaseRef.current) exitPose();
    const compiled = compile(value, {
      duration_s: spec.duration_s,
      focal_mm: lensUserSet ? spec.lens.focal_mm : undefined,
      camera,
      speed: spec.camera.speed,
      taper: spec.camera.taper,
      distance: spec.camera.distance,
      frame: spec.camera.frame,
    });
    // Run 7 (B3): a camera path the user drew survives an edit of the sentence while the cast stays the same
    const kept = keepPath(built, compiled);
    if (kept.dropped) setPathNote({ text: "Camera path reset for the new scene.", undo: built });
    setLine(value);
    setBuilt(kept.next);
    setFrame(0);
    setPlaying(true);
    setPicked(null);
    setTopSel(null);
  };

  const pickCamera = (id: CameraPick) => {
    const next = cam === id ? null : id;
    setCam(next);
    setMoveChanged(true);
    if (posePhaseRef.current) exitPose();
    // a camera chip is a new move: it replaces a drawn path, with Undo
    if (spec.camera.free) setPathNote({ text: "Camera path replaced.", undo: built });
    setBuilt(compile(line, { duration_s: spec.duration_s, focal_mm: spec.lens.focal_mm, camera: next, speed: spec.camera.speed, taper: spec.camera.taper, distance: spec.camera.distance, frame: spec.camera.frame }));
    setFrame(0);
    setPlaying(true);
    setPicked(null);
    setTopSel(null);
  };

  /** The camera, changed by a Top view edit: the page and the runtime both get it now (no wait for React). */
  const setCamera = (camera: PrevizSpec["camera"]) => {
    setMoveChanged(true);
    const next = { ...specRef.current, camera };
    specRef.current = next;
    rtRef.current?.setSpec(next, frameRef.current);
    setBuilt((b) => ({ ...b, spec: { ...b.spec, camera } }));
  };
  const pushUndo = () => {
    undoRef.current = [...undoRef.current.slice(-19), structuredClone(specRef.current.camera)];
    setUndoCount(undoRef.current.length);
  };
  const undoPath = () => {
    const last = undoRef.current.pop();
    setUndoCount(undoRef.current.length);
    if (!last) return false;
    setCamera(last);
    setTopSel((s) => (s != null && s < last.keys.length ? s : null));
    return true;
  };
  /** Before the first edit: the keys on screen become the user's (B1), still following the subject if they did (B2). */
  const adopt = (): PrevizSpec["camera"] | null => {
    const cur = specRef.current;
    if (cur.camera.free) return cur.camera;
    const camera = rtRef.current?.adoptPath();
    if (!camera) return null;
    setCamera(camera);
    setBuilt((b) => ({ ...b, camera: "Your path" }));
    return camera;
  };
  /** The key under the camera on this frame, made if there is none (dragging the camera bends the path there). */
  const keyAtFrame = (camera: PrevizSpec["camera"]): { camera: PrevizSpec["camera"]; index: number } => {
    const rt = rtRef.current;
    const u = rt ? rt.timeOf(Math.round(frameRef.current)) : 0;
    const t = keyTime(camera, u);
    const n = Math.max(2, shotFrames(specRef.current.duration_s, specRef.current.fps, PREVIZ_MAX_S * PREVIZ_FPS));
    const at = keyAtTime(camera.keys, t, Math.max(0.004, 0.6 / n));
    if (at >= 0) return { camera, index: at };
    const made = insertPoint(camera.keys, t, specRef.current.lens.focal_mm, curveOf(camera));
    if (!made) {
      if (camera.keys.length >= MAX_KEYS) setPathFull(true);
      let best = 0;
      camera.keys.forEach((k, i) => { if (Math.abs(k.t - t) < Math.abs(camera.keys[best].t - t)) best = i; });
      return { camera, index: best };
    }
    const next = { ...camera, keys: made.keys };
    setCamera(next);
    return { camera: next, index: made.index };
  };
  /** The camera moved by hand (W A S D, a drag in the shot): let go and this frame gets a keyframe with that view.
   * A path that follows a moving subject keeps the old click-to-lock. Returns true when a keyframe was set. */
  const keyThisView = (): boolean => {
    const rt = rtRef.current;
    const pose = draftRef.current;
    if (!rt || busyRef.current || posePhaseRef.current || !movedRef.current || !pose || !flyingRef.current) return false;
    if (specRef.current.camera.follow) return false;
    pushUndo();
    const own = adopt();
    if (!own) return false;
    const u = rt.timeOf(Math.round(frameRef.current));
    const n = Math.max(2, shotFrames(specRef.current.duration_s, specRef.current.fps, PREVIZ_MAX_S * PREVIZ_FPS));
    const made = setViewAt(own.keys, keyTime(own, u), pose, Math.max(0.004, 0.6 / n), specRef.current.lens.focal_mm, curveOf(own));
    if (!made) {
      setPathFull(true);
      return false;
    }
    rt.setPose(null);
    flyingRef.current = false;
    movedRef.current = false;
    draftRef.current = null;
    setAiming(false);
    setCamera({ ...own, keys: made.keys });
    setBuilt((b) => ({ ...b, camera: "Your path" }));
    setTopSel(made.index);
    setPathFlash(`Keyframe set · frame ${Math.round(frameRef.current) + 1}`);
    window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setPathFlash(null), 2600);
    return true;
  };
  const keyThisViewRef = useRef(keyThisView);
  keyThisViewRef.current = keyThisView;
  const removeSelected = () => {
    const i = topSelRef.current;
    const keys = specRef.current.camera.keys;
    if (i == null || !canRemove(keys, i)) return false;
    pushUndo();
    setCamera({ ...specRef.current.camera, keys: removePoint(keys, i) });
    setTopSel(null);
    setPathFull(false);
    return true;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let rt: PrevizRuntime;
    try {
      rt = new PrevizRuntime(canvas);
    } catch {
      setPreviewError("The preview couldn't start on this machine.");
      return;
    }
    rtRef.current = rt;
    rt.fallback = (reason) => setNotice(reason || null);
    rt.onGuard = (notes) => setGuardNotes(notes);
    // a drawn path rebuilds as "locked" (its keys are the user's): the chip keeps saying it's their path
    rt.onLabel = (label) => setBuilt((b) => (b.camera === label || (label === "locked" && b.spec.camera.free) ? b : { ...b, camera: label }));
    rt.setSpec(specRef.current, frameRef.current);
    const regionOf = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const u = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
      const v = rect.height > 0 ? (e.clientY - rect.top) / rect.height : 0;
      const point = (x: number, y: number) => ({ x: x * 2 - 1, y: -(y * 2 - 1) });
      const mode = viewRef.current;
      if (mode === "top") return { kind: "top" as const, ndc: point(u, v) };
      if (mode === "split" && u >= 0.5) return { kind: "top" as const, ndc: point((u - 0.5) / 0.5, v) };
      if (mode === "split") {
        // the shot sits letterboxed in the left half at its own aspect
        const band = Math.min(1, (rect.width / 2 / (1920 / 1088)) / Math.max(1, rect.height));
        return { kind: "eye" as const, ndc: point(u / 0.5, (v - (1 - band) / 2) / band) };
      }
      return { kind: "eye" as const, ndc: point(u, v) };
    };
    /** The Top pane's NDC for any pointer, even off the pane (a drag that leaves it keeps going). */
    const topNdc = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const u = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
      const v = rect.height > 0 ? (e.clientY - rect.top) / rect.height : 0;
      const x = viewRef.current === "split" ? (u - 0.5) / 0.5 : u;
      return { x: x * 2 - 1, y: -(v * 2 - 1) };
    };
    const onDbl = (e: MouseEvent) => {
      const at = regionOf(e as PointerEvent);
      if (at.kind !== "top") return;
      const hit = rt.pickTop(at.ndc.x, at.ndc.y);
      if (hit.kind === "point") {
        topSelRef.current = hit.index;
        removeSelected();
      } else if (hit.kind === "none") rt.fitTop();
    };
    rt.onTopFrame = setMpp;
    const modifier = (e: PointerEvent) => e.metaKey || e.ctrlKey || keysRef.current.has("m");
    const focalAspect = () => [specRef.current.lens.focal_mm, 1920 / 1088] as const;
    const hold = (next: CameraPose) => {
      draftRef.current = next;
      movedRef.current = true;
      rt.setPose(next);
    };
    const beginOrbit = (e: PointerEvent) => {
      e.preventDefault();
      const region = regionOf(e);
      let pivot: [number, number, number] | null = null;
      if (region.kind === "eye") {
        const hit = rt.pick(region.ndc.x, region.ndc.y);
        if (hit && !hit.axis) pivot = rt.pivotOf(hit.index);
      }
      aimDrag.current = { x: e.clientX, y: e.clientY, mode: "orbit" };
      canvas.setPointerCapture(e.pointerId);
      const pose = grabPoseRef.current();
      if (pose && pivot) {
        const aimed = { pos: [...pose.pos] as [number, number, number], look: pivot };
        draftRef.current = aimed;
        rt.setPose(aimed);
      }
      setPlaying(false);
    };
    const cursorFor = (hit: TopHit): string =>
      hit.kind === "handle" || hit.kind === "camera" || hit.kind === "point" || hit.kind === "subject" ? "grab" : hit.kind === "path" ? "crosshair" : "default";
    /** Top view: drag the camera, its turn handle, a point or a person; click the path to add a point; drag the map. */
    const onTopDown = (e: PointerEvent, ndc: { x: number; y: number }) => {
      e.preventDefault();
      // the middle button pans; the right button and Option do nothing here (B14)
      if (e.button === 2 || (e.altKey && e.button === 0)) return;
      if (busyRef.current) return;
      const start = (kind: "point" | "turn" | "subject" | "pan" | "still" | "stillTurn", index = -1, undo = false) => {
        topDrag.current = { kind, index, last: ndc, moved: false, undo };
        canvas.setPointerCapture(e.pointerId);
        canvas.style.cursor = kind === "pan" ? "grabbing" : "grabbing";
      };
      if (e.button === 1) return start("pan");
      if (e.button !== 0) return;
      const hit = rt.pickTop(ndc.x, ndc.y);
      if (hit.kind === "none") {
        setTopSel(null);
        return start("pan");
      }
      setPlaying(false);
      if (hit.kind === "subject") {
        setPicked(hit.index);
        pickedRef.current = hit.index;
        return start("subject", hit.index);
      }
      if (hit.kind === "point") {
        setTopSel(hit.index);
        return start("point", hit.index, true);
      }
      pushUndo();
      const own = adopt();
      if (!own) return;
      if (hit.kind === "path") {
        const t = keyTime(own, hit.u);
        const made = insertPoint(own.keys, t, specRef.current.lens.focal_mm, curveOf(own));
        if (!made) {
          if (own.keys.length >= MAX_KEYS) setPathFull(true);
          return;
        }
        setCamera({ ...own, keys: made.keys });
        setTopSel(made.index);
        return start("point", made.index);
      }
      // a still camera (every key in one place) moves and turns as one, so it stays still
      const k0 = own.keys[0];
      if (own.keys.every((k) => Math.hypot(k.pos[0] - k0.pos[0], k.pos[1] - k0.pos[1], k.pos[2] - k0.pos[2]) < 0.01)) {
        return start(hit.kind === "handle" ? "stillTurn" : "still", 0);
      }
      const at = keyAtFrame(own);
      setTopSel(at.index);
      start(hit.kind === "handle" ? "turn" : "point", at.index);
    };
    const onTopMove = (e: PointerEvent, ndc: { x: number; y: number }) => {
      const drag = topDrag.current;
      if (!drag) {
        if (!e.buttons) canvas.style.cursor = cursorFor(rt.pickTop(ndc.x, ndc.y));
        return;
      }
      const last = drag.last;
      drag.last = ndc;
      if (drag.kind === "pan") {
        rt.panTop(last, ndc);
        return;
      }
      const ground = rt.groundAt(ndc.x, ndc.y);
      if (!ground) return;
      if (!drag.moved) {
        drag.moved = true;
        // a point drag adopts the keys on its first move, so a plain click only selects
        if (drag.undo) pushUndo();
        if (drag.kind !== "subject" && !adopt()) return;
      }
      if (drag.kind === "subject") {
        const index = drag.index;
        setBuilt((b) => {
          const current = b.spec.subjects[index];
          if (!current) return b;
          const pos: [number, number, number] = [Math.round(ground.x * 10) / 10, current.pos[1], Math.round(ground.z * 10) / 10];
          if (pos[0] === current.pos[0] && pos[2] === current.pos[2]) return b;
          const subjects = b.spec.subjects.map((s, i) => (i === index ? { ...s, pos } : s));
          return { ...b, spec: { ...b.spec, subjects } };
        });
        return;
      }
      const camera = specRef.current.camera;
      if (drag.kind === "still") {
        const from = rt.groundAt(last.x, last.y);
        if (from) setCamera({ ...camera, keys: movePath(camera.keys, ground.x - from.x, ground.z - from.z) });
        return;
      }
      if (drag.kind === "stillTurn") {
        const shown = rt.drawnKey(0);
        if (!shown) return;
        let keys = camera.keys;
        keys.forEach((k, i) => { keys = aimPoint(keys, i, k.pos[0] + (ground.x - shown.x), k.pos[2] + (ground.z - shown.z)); });
        setCamera({ ...camera, keys });
        return;
      }
      if (drag.kind === "point") {
        const at = rt.keyGround(drag.index, ground.x, ground.z);
        if (at) setCamera({ ...camera, keys: movePoint(camera.keys, drag.index, at[0], at[1]) });
        return;
      }
      // the turn handle: the lens faces the pointer
      const k = camera.keys[drag.index];
      const shown = rt.drawnKey(drag.index);
      if (!k || !shown) return;
      setCamera({ ...camera, keys: aimPoint(camera.keys, drag.index, k.pos[0] + (ground.x - shown.x), k.pos[2] + (ground.z - shown.z)) });
    };
    const onDown = (e: PointerEvent) => {
      const first = regionOf(e);
      if (first.kind === "top") {
        onTopDown(e, first.ndc);
        return;
      }
      if (e.button === 0 && e.altKey) {
        beginOrbit(e);
        return;
      }
      if (e.button === 1 && !e.shiftKey) {
        beginOrbit(e);
        return;
      }
      if (modifier(e) && e.button === 0) {
        e.preventDefault();
        aimDrag.current = { x: e.clientX, y: e.clientY, mode: e.shiftKey ? "orbit" : "pan" };
        canvas.setPointerCapture(e.pointerId);
        setPlaying(false);
        setAiming(true);
        if (e.shiftKey || posePhaseRef.current || flyingRef.current) grabPoseRef.current();
        return;
      }
      if (e.button === 1) {
        e.preventDefault();
        aimDrag.current = { x: e.clientX, y: e.clientY, mode: "truck" };
        canvas.setPointerCapture(e.pointerId);
        grabPoseRef.current();
        setPlaying(false);
        return;
      }
      if (e.button === 2) {
        e.preventDefault();
        aimDrag.current = { x: e.clientX, y: e.clientY, mode: "look" };
        canvas.setPointerCapture(e.pointerId);
        grabPoseRef.current();
        setPlaying(false);
        return;
      }
      if (e.button !== 0) return;
      // Setting start and end: a click anywhere in the shot (on a person too) sets the view; nothing is picked.
      if (posePhaseRef.current) {
        downRef.current = { x: e.clientX, y: e.clientY, hit: false, button: 0 };
        return;
      }
      const point = first.ndc;
      const hit = rt.pick(point.x, point.y);
      if (hit?.axis) {
        rt.beginDrag(hit.index, hit.axis, point.x, point.y);
        canvas.setPointerCapture(e.pointerId);
        setPlaying(false);
        return;
      }
      downRef.current = { x: e.clientX, y: e.clientY, hit: !!hit, button: 0 };
      pickedRef.current = hit ? hit.index : null;
      setPicked(pickedRef.current);
    };
    const onMove = (e: PointerEvent) => {
      const at = regionOf(e);
      if (topDrag.current || (at.kind === "top" && !aimDrag.current && !downRef.current)) {
        onTopMove(e, topDrag.current && at.kind !== "top" ? topNdc(e) : at.ndc);
        return;
      }
      if (canvas.style.cursor && !aimDrag.current) canvas.style.cursor = "";
      const down = downRef.current;
      if (!aimDrag.current && down && down.hit && e.buttons === 1 && pickedRef.current != null) {
        if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) return;
        const region = regionOf(e);
        if (region.kind !== "eye") return;
        const ground = rt.dropAt(region.ndc.x, region.ndc.y);
        const index = pickedRef.current;
        if (!ground) return;
        setBuilt((b) => {
          const current = b.spec.subjects[index];
          if (!current) return b;
          const pos: [number, number, number] = [Math.round(ground.x * 10) / 10, current.pos[1], Math.round(ground.z * 10) / 10];
          if (pos[0] === current.pos[0] && pos[2] === current.pos[2]) return b;
          const subjects = b.spec.subjects.map((s, i) => (i === index ? { ...s, pos } : s));
          return { ...b, spec: { ...b.spec, subjects } };
        });
        return;
      }
      if (!aimDrag.current && down && !down.hit && e.buttons === 1) {
        if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) {
          aimDrag.current = { x: e.clientX, y: e.clientY, mode: "look" };
          canvas.setPointerCapture(e.pointerId);
          grabPoseRef.current();
        }
      }
      const drag = aimDrag.current;
      if (drag) {
        const rect = canvas.getBoundingClientRect();
        const dx = rect.width > 0 ? (e.clientX - drag.x) / rect.width : 0;
        const dy = rect.height > 0 ? (e.clientY - drag.y) / rect.height : 0;
        drag.x = e.clientX;
        drag.y = e.clientY;
        if (!dx && !dy) return;
        const pose = draftRef.current;
        const free = !!(pose && (posePhaseRef.current || flyingRef.current));
        if (drag.mode === "look" && pose) hold(lookPose(pose, -dx * Math.PI, -dy * 1.3));
        else if (drag.mode === "orbit" && pose) hold(orbitPose(pose, -dx * 2.2, dy * 1.2));
        else if (drag.mode === "truck" && pose) hold(truckPose(pose, dx, -dy, ...focalAspect()));
        else if (drag.mode === "pan" && free && pose) hold(panLook(pose, dx, -dy, ...focalAspect()));
        else if (drag.mode === "pan") {
          setBuilt((b) => {
            const cur = b.spec.camera.frame ?? [0, 0];
            const x = Math.min(1.2, Math.max(-1.2, Math.round((cur[0] + dx * 1.15) * 1000) / 1000));
            const y = Math.min(1.2, Math.max(-1.2, Math.round((cur[1] - dy * 1.15) * 1000) / 1000));
            if (x === cur[0] && y === cur[1]) return b;
            return { ...b, spec: { ...b.spec, camera: { ...b.spec.camera, frame: [x, y] } } };
          });
        }
        return;
      }
      const point = regionOf(e).ndc;
      const next = rt.moveDrag(point.x, point.y);
      const index = pickedRef.current;
      if (!next || index == null) return;
      setBuilt((b) => {
        const current = b.spec.subjects[index];
        if (!current) return b;
        const subjects = b.spec.subjects.map((s, i) => (i === index ? { ...s, pos: next } : s));
        return { ...b, spec: { ...b.spec, subjects } };
      });
    };
    const onUp = (e: PointerEvent) => {
      const down = downRef.current;
      const dragged = !!aimDrag.current || !!topDrag.current;
      if (topDrag.current) {
        topDrag.current = null;
        const at = regionOf(e);
        canvas.style.cursor = at.kind === "top" ? cursorFor(rt.pickTop(at.ndc.x, at.ndc.y)) : "";
      }
      aimDrag.current = null;
      downRef.current = null;
      if (!flyingRef.current && !posePhaseRef.current) setAiming(false);
      rt.endDrag();
      if (dragged && keyThisViewRef.current()) return;
      if (e.button === 0 && down && !down.hit && !dragged) lockRef.current();
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.buttons) return;
      // Top view: the wheel zooms the map about the pointer; the shot's distance is the Shot view's (B5)
      const at = regionOf(e as unknown as PointerEvent);
      if (at.kind === "top") {
        rt.zoomTop(Math.exp(e.deltaY * 0.0015), at.ndc.x, at.ndc.y);
        return;
      }
      if ((posePhaseRef.current || flyingRef.current) && draftRef.current) {
        hold(dollyPose(draftRef.current, Math.exp(e.deltaY * 0.0012)));
        return;
      }
      const factor = Math.exp(e.deltaY * 0.0015);
      setBuilt((b) => {
        const current = b.spec.camera.distance ?? 1;
        const distance = Math.min(4, Math.max(0.35, Math.round(current * factor * 100) / 100));
        if (distance === current) return b;
        return { ...b, spec: { ...b.spec, camera: { ...b.spec.camera, distance } } };
      });
    };
    const onMenu = (e: Event) => e.preventDefault();
    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 1 || e.altKey) e.preventDefault();
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("contextmenu", onMenu);
    canvas.addEventListener("dblclick", onDbl);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("contextmenu", onMenu);
      canvas.removeEventListener("dblclick", onDbl);
      rt.dispose();
      rtRef.current = null;
    };
  }, [opened]);

  useEffect(() => {
    const rt = rtRef.current;
    if (!rt) return;
    const f = Math.min(frameRef.current, frames - 1);
    rt.setSpec(spec, f);
    if (f !== frameRef.current) setFrame(f);
    rt.showHandles(pickedRef.current);
  }, [spec, frames]);

  useEffect(() => {
    rtRef.current?.setView(view);
    if (view === "eye" && canvasRef.current) canvasRef.current.style.cursor = "";
  }, [view, opened]);

  // the selected point follows the path; a point that's gone lets go
  const keyCount = spec.camera.keys.length;
  useEffect(() => {
    if (topSel != null && topSel >= keyCount) setTopSel(null);
    rtRef.current?.setTopSelection(topSel != null && topSel < keyCount ? topSel : null);
  }, [topSel, keyCount, opened]);
  useEffect(() => {
    if (keyCount < MAX_KEYS) setPathFull(false);
  }, [keyCount]);

  useEffect(() => {
    rtRef.current?.renderFrame(frame);
  }, [frame]);

  useEffect(() => {
    pickedRef.current = picked;
    rtRef.current?.showHandles(picked);
  }, [picked]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const specNow = specRef.current;
      const n = shotFrames(specNow.duration_s, specNow.fps, PREVIZ_MAX_S * PREVIZ_FPS);
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      setFrame((f) => {
        const next = f + dt * specNow.fps;
        return next >= n ? next % n : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    const typing = () => {
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!el?.isContentEditable;
    };
    const walkKey = (key: string) =>
      key === "w" || key === "a" || key === "s" || key === "d" || key === "q" || key === "e" || key.startsWith("arrow");
    const down = (e: KeyboardEvent) => {
      if (typing() || busyRef.current) return;
      const key = e.key.toLowerCase();
      // the camera path: Cmd/Ctrl-Z undoes the last edit; Delete removes the selected point; Escape lets go of it
      if (key === "z" && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        if (undoRef.current.length) {
          e.preventDefault();
          undoPath();
        }
        return;
      }
      if ((key === "delete" || key === "backspace") && topSelRef.current != null) {
        if (removeSelected()) e.preventDefault();
        return;
      }
      if (key === "escape" && topSelRef.current != null) {
        setTopSel(null);
        return;
      }
      // in Top view the keys fly nothing: the shot's camera isn't on screen (B11)
      if (viewRef.current === "top") return;
      if (key === "m" || walkKey(key) || key === "shift") {
        if (walkKey(key)) {
          e.preventDefault();
          grabPoseRef.current();
        }
        keysRef.current.add(key);
      }
    };
    const up = (e: KeyboardEvent) => {
      keysRef.current.delete(e.key.toLowerCase());
      if (walkKey(e.key.toLowerCase()) && ![...keysRef.current].some(walkKey)) keyThisViewRef.current();
    };
    const blur = () => {
      keysRef.current.clear();
    };
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const keys = keysRef.current;
      const pose = draftRef.current;
      let forward = 0;
      let strafe = 0;
      let lift = 0;
      if (keys.has("w") || keys.has("arrowup")) forward += 1;
      if (keys.has("s") || keys.has("arrowdown")) forward -= 1;
      if (keys.has("d") || keys.has("arrowright")) strafe += 1;
      if (keys.has("a") || keys.has("arrowleft")) strafe -= 1;
      if (keys.has("e")) lift += 1;
      if (keys.has("q")) lift -= 1;
      if (pose && (forward || strafe || lift) && (flyingRef.current || posePhaseRef.current)) {
        const speed = (keys.has("shift") ? 8 : 3) * dt;
        const next = walkPose(pose, forward * speed, strafe * speed * 0.28, lift * speed * 0.65);
        draftRef.current = next;
        movedRef.current = true;
        rtRef.current?.setPose(next);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " || e.altKey || e.metaKey || e.ctrlKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable) return;
      e.preventDefault();
      if (!busy) setPlaying((p) => !p);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy]);

  const beginCustom = () => {
    const rt = rtRef.current;
    if (!rt || busy) return;
    // start and end are set through the shot's own eye (B12)
    setView("eye");
    rt.setView("eye");
    setTopSel(null);
    setPlaying(false);
    rt.setPose(null);
    rt.renderFrame(0);
    const pose = rt.viewPose();
    if (!pose) return;
    draftRef.current = pose;
    startPoseRef.current = null;
    movedRef.current = false;
    rt.setPose(pose);
    setFrame(0);
    setPosePhase("start");
  };

  const nextCustom = () => {
    const rt = rtRef.current;
    const pose = draftRef.current;
    if (!rt || !pose) return;
    const held = { pos: [...pose.pos] as [number, number, number], look: [...pose.look] as [number, number, number] };
    startPoseRef.current = held;
    draftRef.current = { pos: [...held.pos], look: [...held.look] };
    movedRef.current = false;
    const n = shotFrames(specRef.current.duration_s, specRef.current.fps, PREVIZ_MAX_S * PREVIZ_FPS);
    rt.setPose(draftRef.current);
    setFrame(Math.max(0, n - 1));
    setPosePhase("end");
  };

  const resetCamera = () => {
    if (spec.camera.free) pushUndo();
    setTopSel(null);
    setLensUserSet(false);
    if (posePhaseRef.current) exitPose();
    const next = compile(line, {
      duration_s: spec.duration_s,
      focal_mm: spec.lens.focal_mm,
      camera: cam,
      speed: spec.camera.speed,
      taper: spec.camera.taper,
      distance: spec.camera.distance,
    });
    setBuilt(next);
    setFrame(0);
    setPlaying(true);
  };

  const backCustom = () => {
    const start = startPoseRef.current;
    if (!start) return;
    draftRef.current = start;
    rtRef.current?.setPose(start);
    setFrame(0);
    setPosePhase("start");
  };

  const saveCustom = () => {
    const start = startPoseRef.current;
    const end = draftRef.current;
    if (!start || !end) return;
    const focal = specRef.current.lens.focal_mm;
    rtRef.current?.setPose(null);
    posePhaseRef.current = null;
    flyingRef.current = false;
    movedRef.current = false;
    setAiming(false);
    setBuilt((b) => ({
      ...b,
      camera: "Start to end",
      spec: {
        ...b.spec,
        camera: {
          interp: "catmull_rom",
          ease: "in_out",
          speed: 1,
          taper: 1,
          distance: 1,
          free: true,
          keys: [
            { t: 0, pos: start.pos, look: start.look, focal_mm: focal, roll: 0 },
            { t: 1, pos: end.pos, look: end.look, focal_mm: focal, roll: 0 },
          ],
        },
      },
    }));
    setPosePhase(null);
    setFrame(0);
    setPlaying(true);
    // Said where the clicks were: the path took.
    setPathFlash("Path set · start to end");
    window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setPathFlash(null), 2600);
  };

  lockRef.current = () => {
    if (busyRef.current) return false;
    // Setting start and end: a click in the shot keeps the view as it is now, moved or not (the banner said so).
    if (posePhaseRef.current === "start") {
      movedRef.current = false;
      nextCustom();
      return true;
    }
    if (posePhaseRef.current === "end") {
      movedRef.current = false;
      saveCustom();
      return true;
    }
    const pose = draftRef.current;
    if (!movedRef.current || !pose || !flyingRef.current) return false;
    const focal = specRef.current.lens.focal_mm;
    rtRef.current?.setPose(null);
    flyingRef.current = false;
    movedRef.current = false;
    draftRef.current = null;
    setAiming(false);
    setBuilt((b) => ({
      ...b,
      camera: "Locked camera",
      spec: {
        ...b.spec,
        camera: {
          interp: "catmull_rom",
          ease: "linear",
          speed: 1,
          taper: 0,
          distance: 1,
          free: true,
          keys: [
            { t: 0, pos: [...pose.pos], look: [...pose.look], focal_mm: focal, roll: 0 },
            { t: 1, pos: [...pose.pos], look: [...pose.look], focal_mm: focal, roll: 0 },
          ],
        },
      },
    }));
    setPlaying(false);
    return true;
  };

  // Build and Rebuild: the scene again from the sentence, keeping the camera move picked on the right.
  const openEditor = (text: string) => {
    const keep = opened ? cam : null;
    setExample(null);
    setCam(keep);
    applyLine(text, keep);
    setOpened(true);
    setMoveChanged(false);
  };
  // An example opens exactly as its tile was made: its sentence compiled fresh, with none of the last scene's camera,
  // lens, length or start and end carried over.
  const [example, setExample] = useState<PrevizExample | null>(null);
  // An unchanged example shows its own move on the camera chips (picking one makes it yours, like any edit).
  const exampleCam = opened && example && line === example.text ? chipForMove(example.move) : null;
  const openExample = (ex: PrevizExample) => {
    if (posePhaseRef.current) exitPose();
    setCam(null);
    setLensUserSet(false);
    setLine(ex.text);
    setBuilt(compile(ex.text));
    setFrame(0);
    setPlaying(true);
    setPicked(null);
    setExample(ex);
    setOpened(true);
    setMoveChanged(false);
  };

  const make = async () => {
    const rt = rtRef.current;
    if (!rt || busy) return;
    const projectId = await askSaveTo("Save the shot to", "Make shot");
    if (!projectId) return;
    setPlaying(false);
    setBusy(true);
    setError(null);
    setCanceled(false);
    setProgress(0);
    rt.showHandles(null);
    // The playblast is always the camera's eye, moving: Top or Split and a held start or end pose are put back after.
    const keptView = rt.currentView();
    const keptPose = rt.heldPose();
    rt.setPose(null);
    rt.setView("eye", false);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const framing = rt.framing0();
      const matte = rt.subjectMatte();
      const depth = rt.depthPasses();
      const shot = await exportPreviz(rt, specRef.current, {
        cap: capRef.current,
        projectId,
        name: built.title,
        signal: ctrl.signal,
        onProgress: setProgress,
        framing,
        matte,
        depth,
      });
      const people = readShot(lineRef.current)?.actors.reduce((n, a) => n + (a.kind === "person" ? a.count : 0), 0) || 1;
      const cast = getState().cast[shot.project_id || targetProject()];
      const look = lookPromptFor(lineRef.current, cast, people);
      const ctx = { cast: castIn(lineRef.current, cast), picked: loneCar(cast), look };
      openMadePreviz(shot, motionPromptFor(specRef.current, shot.previz_framing ?? framing ?? undefined, ctx), look);
    } catch (e) {
      if (e instanceof ExportCanceled) setCanceled(true);
      else setError(e instanceof Error && isPlain(e.message) ? e.message : plainError(e, "export"));
    } finally {
      const live = rtRef.current;
      if (live) {
        live.setView(keptView, false);
        if (keptPose) live.setPose(keptPose);
        else live.renderFrame(frameRef.current);
        live.showHandles(pickedRef.current);
      }
      setBusy(false);
      abortRef.current = null;
    }
  };

  const fragments = useMemo(() => (lexReady ? tryFragments(shotGrammar(), seenGate.current) : []), [lexReady, opened]);
  const sentence = opened ? line : draft;
  const setSentence = (v: string) => (opened ? applyLine(v) : setDraft(v));
  const addFragment = (text: string) => {
    const el = areaRef.current;
    const at = el && document.activeElement === el ? el.selectionStart : sentence.length;
    const next = insertFragment(sentence, at, text);
    setSentence(next.text);
    requestAnimationFrame(() => {
      areaRef.current?.focus();
      areaRef.current?.setSelectionRange(next.cursor, next.cursor);
    });
  };
  const phone = useMedia("(max-width: 767px)");
  const [dockFocus, setDockFocus] = useState(false);
  const compactDock = phone && !dockFocus;
  const body = (
    <PrevizBody
      value={sentence}
      onChange={setSentence}
      onSubmit={() => {
        const text = sentence.trim();
        if (text && !busy) openEditor(text);
      }}
      submitLabel={opened ? "Rebuild" : "Build"}
      primary={!opened}
      go={opened && moveChanged}
      understood={opened ? { matched: built.matched.slice(0, 8), missed: built.missed.slice(0, 6), camera: built.camera } : null}
      disabled={busy}
      areaRef={areaRef}
      fragments={fragments}
      onFragment={addFragment}
      notice={libFailed ? LIBRARY_FAILED : null}
      example={opened && example && line === example.text ? example : null}
      compact={compactDock}
    />
  );
  // A phone gets the dock as one line until it's used, so the scenes, the camera chips and Make shot stay in view.
  const dock = (
    <div
      onFocus={() => setDockFocus(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDockFocus(false);
      }}
    >
      <CreateComposer
        placement="dock"
        tab="previz"
        previz={body}
        compact={compactDock}
        className={cx("previz-dock pointer-events-none [&>*]:pointer-events-auto", opened && "right-[380px] max-[1180px]:right-[340px] max-[767px]:right-0")}
      />
    </div>
  );

  if (!opened) {
    return (
      <div ref={rootRef} className="relative h-full overflow-hidden bg-bg">
        <PrevizGate onPreset={openExample} padBottom={dockH} lexReady={lexReady} />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-[linear-gradient(to_top,#0D0C0B_0%,rgba(13,12,11,0.85)_45%,rgba(13,12,11,0)_100%)]" style={{ height: dockH + 48 }} />
        {dock}
      </div>
    );
  }

  return (
    <div ref={rootRef} className="relative h-full overflow-hidden bg-bg">
    {/* The dock sits under the picture column only, so the panel keeps the full height (Camera path, Make shot). */}
    <div className="flex h-full min-h-0 max-[767px]:flex-col max-[767px]:overflow-y-auto" style={phone ? { paddingBottom: dockH } : undefined}>
      <div className="flex min-w-0 flex-1 flex-col max-[767px]:flex-none" style={phone ? undefined : { paddingBottom: dockH }}>
        <div ref={stageRef} className="relative flex min-h-0 flex-1 items-center justify-center bg-bg p-4 max-[767px]:flex-none max-[767px]:p-3 max-[767px]:pt-14">
          {previewError ? (
            <Notice tone="bad" text={previewError} />
          ) : (
            <canvas
              ref={canvasRef}
              data-testid="previz-canvas"
              className="h-auto max-h-full w-full max-w-[1600px] touch-none bg-black"
              style={stageBox ? { aspectRatio: PREVIEW_ASPECT_CSS, width: stageBox.w, height: stageBox.h, maxHeight: "none" } : { aspectRatio: PREVIEW_ASPECT_CSS }}
              aria-label="Previz preview"
            />
          )}
          {previewError ? null : (
            <>
              {new URLSearchParams(window.location.search).get("debug") === "1" ? (
                <p id="previz-debug" className="mono pointer-events-none absolute bottom-4 left-4 z-10 max-w-[70%] whitespace-pre-wrap rounded-lg bg-bg/80 px-2.5 py-2 text-[12px] leading-snug text-ink2" />
              ) : null}
              <Button variant="secondary" size="sm" disabled={busy} onClick={newPrevizScene} className="absolute right-4 top-4 z-10 bg-panel/90">
                New scene
              </Button>
              <div className="absolute left-4 top-4 z-10 flex items-center gap-1.5" role="group" aria-label="View">
                {(
                  [
                    ["eye", "Shot"],
                    ["split", "Split"],
                    ["top", "Top"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={view === id}
                    disabled={id !== "eye" && !!posePhase}
                    onClick={() => setView(id)}
                    className={cx(
                      "h-8 rounded-full border bg-panel/90 px-3 text-[13px] backdrop-blur-sm transition-colors disabled:text-ink4",
                      view === id ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:text-ivory",
                    )}
                  >
                    {label}
                  </button>
                ))}
                {view !== "eye" ? (
                  <IconButton icon={Maximize} label="Fit the map" size="sm" onClick={() => rtRef.current?.fitTop()} className="border border-line bg-panel/90" />
                ) : null}
                {!posePhase && !aiming && spec.camera.frame ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="bg-panel/90"
                    onClick={() =>
                      setBuilt((b) => {
                        const camera = { ...b.spec.camera };
                        delete camera.frame;
                        return { ...b, spec: { ...b.spec, camera } };
                      })
                    }
                  >
                    Center
                  </Button>
                ) : null}
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel/90">
                <Hint text={view === "eye" ? "Drag to look, W A S D to move, scroll to zoom." : "Drag the camera, its handle or a point. Click the line to add a point; double-click one to remove it. ⌘Z undoes."} />
                </span>
              </div>
              {view !== "eye" && mpp > 0 ? <ScaleBar mpp={mpp} split={view === "split"} canvas={canvasRef} scale={() => rtRef.current?.topScale() ?? mpp} /> : null}
            </>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-3 border-t border-line px-4 py-3">
          <button
            type="button"
            aria-label={playing ? "Pause" : "Play"}
            disabled={busy || !!previewError}
            onClick={() => setPlaying((p) => !p)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-line2 text-ink hover:text-ivory disabled:text-ink4"
          >
            {playing ? <Pause size={16} /> : <Play size={16} className="translate-x-px" />}
          </button>
          <div className="relative min-w-0 flex-1">
          <input
            type="range"
            className="range w-full"
            min={0}
            max={Math.max(0, frames - 1)}
            step={1}
            value={Math.min(frames - 1, Math.round(frame))}
            style={{ ["--fill" as string]: fillOf(Math.min(frames - 1, Math.round(frame)), 0, Math.max(1, frames - 1)) }}
            aria-label="Frame"
            disabled={busy || !!previewError}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
          />
          <KeyMarks
            frames={frames}
            keyFrames={keyFrames}
            selected={topSel}
            disabled={busy || !!previewError}
            onPick={(i, f) => {
              setPlaying(false);
              setFrame(f);
              setTopSel(i);
            }}
            onRemove={(i) => {
              const own = adopt();
              if (!own || !canRemove(own.keys, i)) return;
              topSelRef.current = i;
              removeSelected();
            }}
            canRemove={(i) => canRemove(spec.camera.keys, i)}
          />
          </div>
          {posePhase || aiming || pathFlash ? (
            <span className="slate shrink-0 text-ivory" role="status" data-testid="path-status">
              {posePhase === "start"
                ? "Click the shot to set the start"
                : posePhase === "end"
                  ? "Start set · click to set the end"
                  : aiming
                    ? specRef.current.camera.follow
                      ? "Click the shot to lock the camera here"
                      : `Let go to set a keyframe · frame ${Math.min(frames - 1, Math.round(frame)) + 1}`
                    : pathFlash}
            </span>
          ) : null}
          <span className="mono w-24 shrink-0 text-right text-[12px] text-ink3 max-[767px]:w-auto">
            {Math.min(frames - 1, Math.round(frame)) + 1} / {frames}
          </span>
          {/* On a phone Make shot rides with the transport, above the dock and without scrolling. */}
          <div className="hidden shrink-0 max-[767px]:block">
            <Button variant="primary" size="sm" disabled={busy || !!previewError} loading={busy} onClick={() => void make()}>
              Make shot
            </Button>
          </div>
        </div>
        {/* ...and so does what stopped it (a post against the lens), where the tap was. */}
        {error ? <Notice tone="bad" text={error} className="mx-4 mb-3 mt-1 hidden max-[767px]:flex" /> : null}
      </div>

      <aside className="flex w-[380px] shrink-0 flex-col border-l border-line bg-panel max-[1180px]:w-[340px] max-[767px]:w-full max-[767px]:border-l-0 max-[767px]:border-t" aria-label="Previz">
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-6 py-5 max-[767px]:overflow-visible max-[767px]:px-4">
          <h1 className="sr-only">Previz</h1>
          <div className="flex items-center gap-1.5">
            <StepHeader n={1} title="Adjust the camera move" />
            <Hint text="Pick a move, or set your own path. Edit the sentence and the scene follows. Drag people, the camera or a path point in Top." />
          </div>

          <section>
            <div className="mb-2 flex items-center gap-1.5">
              <span className="slate text-ink3">Camera</span>
              <Hint text="Sets the move for this shot. Click the same one again to follow the sentence instead." />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {CAMERA_PRESETS.map((preset) => {
                const on = (cam ?? exampleCam) === preset.id;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    aria-pressed={on}
                    disabled={busy}
                    onClick={() => pickCamera(preset.id)}
                    className={cx(
                      "h-8 rounded-full border px-3 text-[13px] transition-colors duration-150",
                      on ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
                    )}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </section>

          <section>
            <div className="mb-2 flex items-center gap-1.5">
              <span className="slate text-ink3">Camera path</span>
              <Hint text="Move the camera with W A S D (Q and E for up and down) or drag in the shot: let go and that frame gets a keyframe. Keyframes show as diamonds on the timeline; click one, then x to delete it. Set start and end makes a new two-point path. Reset goes back to the sentence." />
            </div>
            {posePhase === null ? (
              <div className="space-y-2">
                {topSel != null && topSel < spec.camera.keys.length ? (
                  <div className="flex items-center justify-between gap-2" aria-label="Selected point">
                    <span className="text-[13px] text-ink2">
                      Point {topSel + 1} of {spec.camera.keys.length}
                    </span>
                    <Button variant="quiet" size="sm" disabled={busy || !canRemove(spec.camera.keys, topSel)} onClick={removeSelected}>
                      Remove point
                    </Button>
                  </div>
                ) : null}
                {spec.camera.keys.length >= 3 ? (
                  <label className="flex items-center gap-3" data-testid="path-curve">
                    <span className="flex w-16 shrink-0 items-center gap-1 text-[13px] text-ink2">
                      Curve
                      <Hint text="How the camera travels between keyframes: left is straight lines, the middle is smooth, right is rounder. It always passes through every keyframe." />
                    </span>
                    <input
                      type="range"
                      className="range min-w-0 flex-1"
                      min={0}
                      max={1.5}
                      step={0.05}
                      disabled={busy}
                      value={curveOf(spec.camera)}
                      style={{ ["--fill" as string]: fillOf(curveOf(spec.camera), 0, 1.5) }}
                      aria-label="Path curve"
                      onPointerDown={() => pushUndo()}
                      onKeyDown={() => pushUndo()}
                      onChange={(e) => {
                        const own = adopt();
                        if (own) setCamera({ ...own, curve: Number(e.target.value) });
                      }}
                    />
                    <span className="mono w-14 shrink-0 text-right text-[12px] text-ink3">
                      {curveOf(spec.camera) < 0.15 ? "Straight" : curveOf(spec.camera) <= 1.1 ? "Smooth" : "Round"}
                    </span>
                  </label>
                ) : null}
                <div className="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={beginCustom}
                    className={cx(
                      "h-8 rounded-full border px-3 text-[13px] transition-colors duration-150",
                      spec.camera.free ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
                    )}
                  >
                    Set start and end
                  </button>
                  {spec.camera.free ? (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={resetCamera}
                      className="h-8 rounded-full border border-line px-3 text-[13px] text-ink2 transition-colors duration-150 hover:text-ink"
                    >
                      Reset
                    </button>
                  ) : null}
                  {undoCount ? (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={undoPath}
                      className="h-8 rounded-full border border-line px-3 text-[13px] text-ink2 transition-colors duration-150 hover:text-ink"
                    >
                      Undo
                    </button>
                  ) : null}
                </div>
                {pathFull ? <Notice text={`${MAX_KEYS} points is the most a path can have.`} /> : null}
                {pathNote ? (
                  <Notice
                    text={pathNote.text}
                    action={{
                      label: "Undo",
                      onClick: () => {
                        setLine(pathNote.undo.spec.notes || line);
                        setBuilt(pathNote.undo);
                        setPathNote(null);
                      },
                    }}
                  />
                ) : null}
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-[13px] text-ink2">{posePhase === "start" ? "Start frame" : "End frame"}</p>
                <div className="flex flex-wrap gap-1.5">
                  {posePhase === "start" ? (
                    <button type="button" onClick={nextCustom} className="h-8 rounded-full border border-ivory/60 px-3 text-[13px] text-ivory">
                      Next
                    </button>
                  ) : (
                    <>
                      <button type="button" onClick={saveCustom} className="h-8 rounded-full border border-ivory/60 px-3 text-[13px] text-ivory">
                        Save
                      </button>
                      <button type="button" onClick={backCustom} className="h-8 rounded-full border border-line px-3 text-[13px] text-ink2 hover:text-ink">
                        Back
                      </button>
                    </>
                  )}
                  <button type="button" onClick={exitPose} className="h-8 rounded-full border border-line px-3 text-[13px] text-ink2 hover:text-ink">
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </section>

          <section className="space-y-1.5">
            {notice && !(libFailed && notice === LIBRARY_FAILED) ? <Notice tone="bad" text={notice} /> : null}
            {guardNotes.map((note) => <p key={note} className="text-[13px] text-ink3">{note}</p>)}
          </section>

          {/* Opened, it scrolls into view: its sliders sat below the panel's fold, under Make shot (FableWalk). */}
          <details
            className="group scroll-mb-6"
            onToggle={(e) => {
              const el = e.currentTarget;
              if (el.open) window.requestAnimationFrame(() => el.scrollIntoView({ block: "end", behavior: "smooth" }));
            }}
          >
            <summary className="flex h-8 cursor-pointer list-none items-center gap-1.5 text-[13px] text-ink2 hover:text-ink [&::-webkit-details-marker]:hidden">
              <ChevronRight size={14} className="transition-transform duration-150 group-open:rotate-90" aria-hidden="true" />
              Advanced
            </summary>
          <section className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4">
            <Tune
              label="Duration (s)"
              hint={`Up to ${(Math.floor((cap / PREVIZ_FPS) * 10) / 10).toFixed(1)} s on this GPU.`}
              value={spec.duration_s}
              min={1 / PREVIZ_FPS}
              max={maxS}
              step={1 / PREVIZ_FPS}
              digits={2}
              onChange={(duration_s) =>
                setBuilt((b) => ({ ...b, spec: { ...b.spec, duration_s: Math.min(maxS, Math.max(1 / PREVIZ_FPS, duration_s)) } }))
              }
            />
            <Tune
              label="Lens (mm)"
              hint="Lower is wider."
              value={spec.lens.focal_mm}
              min={8}
              max={200}
              step={1}
              digits={0}
              onChange={(focal_mm) => {
                setLensUserSet(true);
                setBuilt((b) => {
                  const prev = b.spec.lens.focal_mm || focal_mm;
                  const ratio = prev > 0 ? focal_mm / prev : 1;
                  return {
                    ...b,
                    spec: {
                      ...b.spec,
                      lens: { ...b.spec.lens, focal_mm },
                      camera: { ...b.spec.camera, keys: b.spec.camera.keys.map((k) => ({ ...k, focal_mm: Math.round(((k.focal_mm ?? prev) * ratio) * 10) / 10 })) },
                    },
                  };
                });
              }}
            />
            <Tune
              label="Speed"
              hint="How fast the camera move plays. It does not change how fast people walk."
              value={spec.camera.speed ?? 1}
              min={0.25}
              max={3}
              step={0.05}
              digits={2}
              onChange={(speed) =>
                setBuilt((b) => ({ ...b, spec: { ...b.spec, camera: { ...b.spec.camera, speed } } }))
              }
            />
            <Tune
              label="Smooth"
              hint="100 slows the start and the end."
              value={Math.round((spec.camera.taper ?? (spec.camera.ease === "linear" ? 0 : 1)) * 100)}
              min={0}
              max={100}
              step={1}
              digits={0}
              onChange={(ease) =>
                setBuilt((b) => ({ ...b, spec: { ...b.spec, camera: { ...b.spec.camera, taper: ease / 100 } } }))
              }
            />
            <Tune
              label="Distance"
              hint="Scroll on the shot. Lower is closer."
              value={spec.camera.distance ?? 1}
              min={0.35}
              max={4}
              step={0.05}
              digits={2}
              onChange={(distance) =>
                setBuilt((b) => ({ ...b, spec: { ...b.spec, camera: { ...b.spec.camera, distance } } }))
              }
            />
          </section>
          </details>

          {picked != null && spec.subjects[picked] ? (
            <section className="space-y-2">
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] uppercase tracking-wide text-ink3">Place</span>
                <Hint text="Drag an arrow in the picture, or type a number. X is sideways, Y is up, Z runs along the shot. A following camera stays locked on." />
              </div>
              <div className="grid grid-cols-3 gap-2">
                {(["X", "Y", "Z"] as const).map((label, axis) => (
                  <label key={label} className="flex items-center gap-1.5">
                    <span className="text-[11px] text-ink3">{label}</span>
                    <input
                      type="number"
                      aria-label={`Place ${label}`}
                      disabled={busy}
                      step={0.1}
                      value={formatTune(spec.subjects[picked].pos[axis] ?? 0, 1)}
                      onChange={(e) => {
                        const raw = Number(e.target.value);
                        if (!Number.isFinite(raw)) return;
                        const lo = axis === 1 ? 0 : -40;
                        const value = Math.min(40, Math.max(lo, raw));
                        setBuilt((b) => {
                          const subjects = b.spec.subjects.map((s, i) => {
                            if (i !== picked) return s;
                            const pos: [number, number, number] = [s.pos[0], s.pos[1], s.pos[2]];
                            pos[axis] = value;
                            return { ...s, pos };
                          });
                          return { ...b, spec: { ...b.spec, subjects } };
                        });
                      }}
                      className="h-7 w-full rounded border border-line bg-bg px-2 text-[13px] text-ink"
                    />
                  </label>
                ))}
              </div>
            </section>
          ) : null}
        </div>

        <div className="shrink-0 space-y-3 border-t border-line px-6 py-4">
          <div className="space-y-3 max-[767px]:hidden">
            <StepHeader n={2} title="Make the shot" />
            <SaveFolderRow compact />
          </div>
          {busy ? <ProgressLine value={progress} /> : null}
          {error ? <Notice tone="bad" text={error} className="max-[767px]:hidden" /> : null}
          {canceled && !busy ? <Notice tone="neutral" label="Canceled" text="Nothing was saved." /> : null}
          <div className={cx("flex gap-2", !busy && "max-[767px]:hidden")}>
            <Button variant="primary" size="lg" block disabled={busy || !!previewError} loading={busy} onClick={() => void make()}>
              Make shot<span className="mono ml-2 text-[12px] opacity-70">{frames} frames · {spec.duration_s.toFixed(1)} s</span>
            </Button>
            {busy ? (
              <Button
                variant="secondary"
                onClick={() => abortRef.current?.abort()}
              >
                Cancel
              </Button>
            ) : null}
          </div>
        </div>
      </aside>
    </div>
    {dock}
    </div>
  );
}
