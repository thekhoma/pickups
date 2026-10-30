// "Examples": one tile per workflow, front and centre on Home. Each tile cycles through that workflow's examples
// (a soft crossfade every ~5 s, clickable dots), pauses on hover or focus so the before/after wipe can be scrubbed,
// and has one action: Try it (opens the current example's shot with the task ready, or the Create bar's mode).
// Top left, the workflow's default models (or the models a slide's own take ran); top right, the slide dots.
import { ArrowRight } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { installExamplesPack, openExample, openSetup, setCreateMode, startPreset } from "../actions";
import { cx } from "../lib/format";
import { useOnScreen, useReducedMotion } from "../lib/loops";
import { presetFromExample } from "../lib/recipes";
import { useApp } from "../store";
import type { Example } from "../types";
import { Button, Notice, Skeleton, Spinner } from "./ui";
import { ModelPill, useWorkflowModels } from "./ModelChips";
import { WipeMedia } from "./WipeMedia";

/** Stills that ship with the app (the demo library). */
const LIB = (f: string) => `/mock/${f}`;

interface CharacterCard {
  id: string;
  sheet: string;
  photo?: string;
  words?: string;
}

const CHARACTERS: CharacterCard[] = [
  { id: "char-photo", photo: LIB("still_driver_1.jpg"), sheet: LIB("cast_driver_sheet.jpg") },
  { id: "char-knight", words: "a knight in polished plate armour and a red lion tabard", sheet: LIB("fun_knight_sheet.jpg") },
  { id: "char-alien", words: "a slim green alien in a silver flight suit", sheet: LIB("fun_alien_sheet.jpg") },
  { id: "char-astronaut", words: "an astronaut in a worn white spacesuit with a gold visor", sheet: LIB("fun_astronaut_sheet.jpg") },
];

const IMAGES: Array<{ id: string; prompt: string; src: string }> = [
  {
    id: "img-loch",
    prompt: "A misty sea loch at blue hour, a small stone castle on a tidal island joined by an arched stone bridge, wet seaweed on the rocks, soft cold light, fine film grain",
    src: LIB("image_loch_1.jpg"),
  },
  {
    id: "img-dusk",
    prompt: "A stone castle in deep fresh snow at blue hour, warm light glowing in every window, a dark still loch below, heavy snowflakes, cinematic",
    src: LIB("cast_winter_dusk.jpg"),
  },
];

type Slide =
  | { key: string; kind: "example"; ex: Example }
  | { key: string; kind: "character"; c: CharacterCard }
  | { key: string; kind: "image"; src: string; prompt: string };

interface Workflow {
  id: string;
  title: string;
  slides: Slide[];
}

/** A slate's AI share in words ("REMOVE · 0% AI · REAL PIXELS" → "REMOVE · REAL FRAMES KEPT"). */
export function plainSlate(slate: string): string {
  const m = /(\d+(?:\.\d+)?)\s*%\s*AI/i.exec(slate);
  if (!m) return slate;
  const ai = Number(m[1]);
  const words = ai >= 100 ? "Fully generated" : ai <= 10 ? "Real frames kept" : ai < 50 ? "Mostly your footage" : "Mostly generated";
  return slate
    .split("·")
    .map((p) => p.trim())
    .filter((p) => p && !/real pixels/i.test(p))
    .map((p) => (/%\s*AI/i.test(p) ? words : p))
    .join(" · ");
}

/** The examples grouped into workflows, in the page's order; empty workflows are left out. */
function workflows(examples: Example[]): Workflow[] {
  const of = (pick: (e: Example) => boolean): Slide[] => examples.filter(pick).map((ex) => ({ key: ex.id, kind: "example", ex }));
  const all: Workflow[] = [
    { id: "character", title: "Replace a person", slides: of((e) => e.task === "recast") },
    { id: "weather", title: "Change the weather", slides: of((e) => e.task === "environment") },
    { id: "remove", title: "Remove a person", slides: of((e) => e.task === "remove") },
    { id: "relight", title: "Relight", slides: of((e) => e.task === "relight") },
    { id: "previz", title: "Previz to final", slides: of((e) => e.task === "previz") },
    { id: "sheet", title: "Character sheet", slides: CHARACTERS.map((c) => ({ key: c.id, kind: "character", c })) },
    { id: "finish", title: "Finish", slides: of((e) => e.task === "finish") },
    { id: "image", title: "New image", slides: IMAGES.map((im) => ({ key: im.id, kind: "image", src: im.src, prompt: im.prompt })) },
  ];
  return all.filter((w) => w.slides.length);
}

export function Gallery() {
  const examples = useApp((s) => s.examples);
  const pack = useApp((s) => s.examplesPack);
  const flows = useMemo(() => workflows(examples ?? []), [examples]);
  const showPack = !!examples && !examples.length && !!pack?.available && !pack.installed;
  // No pack to download and none installed: say where the examples come from (one line, one action).
  const noPack = !!examples && !examples.length && !showPack;
  const loading = examples === null;
  const featured = flows.slice(0, 2);
  const rest = flows.slice(2);

  return (
    <section className="pt-8" aria-labelledby="make-title">
      <h2 id="make-title" className="display text-[36px] leading-[1.05]">
        Examples
      </h2>

      {loading ? (
        <div className="mt-8 grid grid-cols-2 gap-6 max-[767px]:grid-cols-1">
          <Skeleton className="aspect-video" />
          <Skeleton className="aspect-video" />
        </div>
      ) : (
        <>
          {showPack ? (
            <div className="mt-8 max-w-[640px]">
              <PackCard />
            </div>
          ) : null}
          {noPack ? (
            <Notice
              className="mt-8"
              text={pack?.importing ? "Installing the examples pack…" : "No example clips yet. Put pickups_examples.zip in Downloads and they install by themselves."}
              action={pack?.importing ? undefined : { label: "Add examples", onClick: () => openSetup(null, "examples") }}
            />
          ) : null}
          <div className="mt-8 grid grid-cols-2 gap-x-6 gap-y-12 max-[900px]:grid-cols-1">
            {featured.map((w, i) => (
              <Tile key={w.id} w={w} size="lg" order={i} />
            ))}
          </div>
          {rest.length ? (
            <div className="mt-14 grid grid-cols-3 gap-x-6 gap-y-12 max-[1180px]:grid-cols-2 max-[640px]:grid-cols-1">
              {rest.map((w, i) => (
                <Tile key={w.id} w={w} size="md" order={i + featured.length} />
              ))}
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

// --- tiles ----------------------------------------------------------------------------------------------------------

const EVERY = 5000;
const FADE = 700;

function Tile({ w, size, order }: { w: Workflow; size: "lg" | "md"; order: number }) {
  const n = w.slides.length;
  const root = useRef<HTMLDivElement>(null);
  const visible = useOnScreen(root, "0px");
  const reduced = useReducedMotion();
  const [idx, setIdx] = useState(0);
  const [prev, setPrev] = useState<number | null>(null);
  const [hovering, setHovering] = useState(false);
  const [focused, setFocused] = useState(false);
  const first = useRef(true);
  const cur = w.slides[Math.min(idx, n - 1)];
  const opening = useApp((s) => (cur.kind === "example" ? s.exampleOpening === cur.ex.id : false));
  const busy = useApp((s) => !!s.exampleOpening);
  const paused = hovering || focused || opening;
  const firstEx = w.slides.find((sl) => sl.kind === "example");
  const defaults = useWorkflowModels(w.id, undefined, firstEx?.kind === "example" ? firstEx.ex.models : undefined);
  const models = (cur.kind === "example" && cur.ex.models_used?.length ? cur.ex.models_used : null) ?? defaults;

  const go = (to: number) => {
    if (to === idx) return;
    setPrev(idx);
    setIdx(to);
  };

  // Auto-advance while on screen and not paused; tiles start staggered so they don't all change at once.
  useEffect(() => {
    if (n < 2 || reduced || paused || !visible) return;
    const delay = EVERY + (first.current ? order * 650 : 0);
    const t = window.setTimeout(() => {
      first.current = false;
      setPrev(idx);
      setIdx((idx + 1) % n);
    }, delay);
    return () => window.clearTimeout(t);
  }, [idx, n, reduced, paused, visible, order]);

  // The outgoing slide stays under the incoming one until the fade is done.
  useEffect(() => {
    if (prev === null) return;
    const t = window.setTimeout(() => setPrev(null), FADE + 50);
    return () => window.clearTimeout(t);
  }, [prev, idx]);

  // Mounted: the current slide, the one fading out, and (on a visible tile) the next one's poster, ready to fade in.
  const next = (idx + 1) % n;
  const mounted = new Set<number>([idx]);
  if (prev !== null) mounted.add(prev);
  if (visible && n > 1 && !reduced) mounted.add(next);

  const useMine = cur.kind === "example" && cur.ex.can_try === false;
  const run = () => {
    if (cur.kind === "example") {
      if (busy) return;
      if (cur.ex.can_try !== false) void openExample(cur.ex);
      else startPreset(presetFromExample(cur.ex).id);
    } else if (cur.kind === "character") setCreateMode("character", cur.c.words);
    else setCreateMode("image", cur.prompt);
  };

  return (
    <div
      ref={root}
      className={cx("group min-w-0 self-start transition-opacity duration-200", opening && "pointer-events-none opacity-60")}
      onPointerEnter={() => setHovering(true)}
      onPointerLeave={() => setHovering(false)}
      onFocus={() => setFocused(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocused(false);
      }}
      aria-busy={opening || undefined}
    >
      <div className="hairline-frame relative aspect-video cursor-pointer overflow-hidden rounded-card bg-black" onClick={run}>
        {w.slides.map((sl, i) =>
          mounted.has(i) ? (
            <div
              key={sl.key}
              className="absolute inset-0 transition-opacity ease-out motion-reduce:transition-none"
              style={{ opacity: i === idx || i === prev ? 1 : 0, zIndex: i === idx ? 2 : i === prev ? 1 : 0, transitionDuration: `${FADE}ms` }}
              aria-hidden={i !== idx || undefined}
            >
              <SlideArt sl={sl} wake={i === idx && focused} />
            </div>
          ) : null,
        )}
        <div className="pointer-events-none absolute inset-x-2.5 top-2 z-[4] flex items-center justify-between gap-2">
          <span className="pointer-events-auto flex min-w-0">
            <ModelPill models={models} max={size === "lg" ? 3 : 2} />
          </span>
          {n > 1 ? (
            <div role="group" aria-label={`${w.title} examples`} className="pointer-events-auto shrink-0">
              <div className="flex items-center rounded-full bg-black/45 px-1 backdrop-blur-[2px]">
                {w.slides.map((sl, i) => (
                  <button
                    key={sl.key}
                    type="button"
                    aria-label={`Example ${i + 1} of ${n}`}
                    aria-current={i === idx || undefined}
                    onClick={(e) => {
                      e.stopPropagation();
                      go(i);
                    }}
                    className="flex h-5 w-4 items-center justify-center"
                  >
                    <span className={cx("block h-1.5 rounded-full transition-all duration-300", i === idx ? "w-3 bg-ivory" : "w-1.5 bg-ivory/45 hover:bg-ivory/80")} />
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <h3 className={cx("display", size === "lg" ? "mt-5 text-[36px] leading-[1.05]" : "mt-4 text-[24px]")}>{w.title}</h3>
      {opening ? (
        <span className="slate mt-3 inline-flex items-center gap-2 text-ink2" role="status">
          <Spinner size={11} />
          Opening…
        </span>
      ) : (
        <button
          type="button"
          onClick={run}
          aria-label={useMine ? `Use on my clip: ${w.title}` : `Try it: ${w.title}`}
          className="slate mt-3 inline-flex items-center gap-1.5 rounded-[2px] text-ink3 transition-colors duration-150 hover:text-ivory focus-visible:text-ivory group-hover:text-ivory"
        >
          {useMine ? "Use on my clip" : "Try it"} <ArrowRight size={12} strokeWidth={1.75} aria-hidden="true" className="transition-transform duration-150 group-hover:translate-x-0.5" />
        </button>
      )}
    </div>
  );
}

function SlideArt({ sl, wake }: { sl: Slide; wake: boolean }): ReactNode {
  if (sl.kind === "example") return <WipeMedia ex={sl.ex} variant="card" lazy wake={wake} />;
  if (sl.kind === "character") return <CharacterArt c={sl.c} />;
  return <ImageArt src={sl.src} />;
}

/** The sheet; a photo-made one shows its photo small in the corner (words go under the picture, not over it). */
function CharacterArt({ c }: { c: CharacterCard }) {
  return (
    <>
      <img src={c.sheet} alt="" loading="lazy" decoding="async" draggable={false} className="absolute inset-0 h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.02]" />
      {c.photo ? (
        <span className="absolute bottom-2.5 left-2.5 z-[2] flex items-end gap-1.5">
          <span className="relative block w-[22%] min-w-[72px] overflow-hidden rounded-[3px] shadow-[0_0_0_1px_rgba(238,233,224,0.7)]">
            <img src={c.photo} alt="" loading="lazy" decoding="async" draggable={false} className="thumb-face aspect-video w-full" />
          </span>
          <span className="slate rounded-[2px] bg-black/70 px-1.5 py-[2px] text-[10px] text-ivory">Photo</span>
        </span>
      ) : null}
    </>
  );
}

function ImageArt({ src }: { src: string }) {
  return <img src={src} alt="" loading="lazy" decoding="async" draggable={false} className="thumb-face absolute inset-0 h-full w-full transition-transform duration-500 ease-out group-hover:scale-[1.02]" />;
}

/** No examples on this machine yet: offer the download in the grid. */
function PackCard() {
  const pack = useApp((s) => s.examplesPack);
  if (!pack) return null;
  const mb = pack.size_mb ? Math.round(pack.size_mb) : 40;
  return (
    <div className="flex aspect-video flex-col justify-between rounded-card border border-dashed border-line2 p-6">
      <span className="slate text-ink3">{mb} MB</span>
      <div>
        <h3 className="display text-[24px]">Examples</h3>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          {pack.busy ? (
            <span className="inline-flex items-center gap-2 font-mono text-[12px] text-ink2" role="status" aria-live="polite">
              <Spinner size={12} /> Downloading…
            </span>
          ) : (
            <Button variant="primary" size="md" onClick={() => void installExamplesPack()}>
              Download examples
            </Button>
          )}
          {pack.error && !pack.busy ? <span className="text-[13px] text-ink">{pack.error}</span> : null}
        </div>
      </div>
    </div>
  );
}
