// A before/after loop with the signature wipe line, for the home hero and example cards.
// Hero: loops and sweeps before→after on its own. Cards: rest on the after, scrub with the cursor on hover,
// and do one staggered idle sweep every ~8 s. Offscreen media pauses; one loop plays at a time.
// `lazy` cards (a big gallery) show their poster and load nothing until the pointer (or focus, via `wake`) reaches them.
import { useEffect, useId, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { cx } from "../lib/format";
import { ease, loops, useActiveLoop, useOnScreen, useReducedMotion } from "../lib/loops";
import type { Example } from "../types";

const HOLD_BEFORE = 1200;
const SWEEP = 2400;
const HOLD_AFTER = 2400;
const HERO_PERIOD = HOLD_BEFORE + SWEEP + HOLD_AFTER + SWEEP;
const CARD_EVERY = 8000;

/** Hero wipe position over time: 100 = all before, 0 = all after. */
function heroPos(t: number) {
  const k = t % HERO_PERIOD;
  if (k < HOLD_BEFORE) return 100;
  if (k < HOLD_BEFORE + SWEEP) return 100 - 100 * ease((k - HOLD_BEFORE) / SWEEP);
  if (k < HOLD_BEFORE + SWEEP + HOLD_AFTER) return 0;
  return 100 * ease((k - HOLD_BEFORE - SWEEP - HOLD_AFTER) / SWEEP);
}

export function WipeMedia({
  ex,
  variant,
  index = 0,
  className,
  lazy,
  wake,
}: {
  ex: Example;
  variant: "hero" | "card";
  index?: number;
  className?: string;
  /** Poster only until hovered (or woken): for big galleries. */
  lazy?: boolean;
  /** Load now (the card got keyboard focus). */
  wake?: boolean;
}) {
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const before = useRef<HTMLVideoElement>(null);
  const after = useRef<HTMLVideoElement>(null);
  const visible = useOnScreen(root);
  const reduced = useReducedMotion();
  const active = useActiveLoop();
  const [armed, setArmed] = useState(false);
  const hover = useRef(false);
  const [hovering, setHovering] = useState(false);
  // Everything rests on the after, which has a poster (the before has none): the hero starts sweeping once the
  // before clip has a frame, so neither side ever paints black.
  const pos = useRef(0);
  const hasVideo = !!(ex.before_url && ex.after_url);
  const [beforeLoaded, setBeforeLoaded] = useState<string | null>(null);
  const beforeReady = hasVideo && beforeLoaded === ex.before_url;

  const apply = (p: number) => {
    pos.current = p;
    const el = root.current;
    if (!el) return;
    el.style.setProperty("--wipe", `${p}%`);
    el.style.setProperty("--line", p <= 0.5 || p >= 99.5 ? "0" : "1");
  };

  useEffect(() => {
    if (visible && (!lazy || wake)) setArmed(true);
  }, [visible, lazy, wake]);

  useEffect(() => {
    if (reduced && beforeReady) apply(50);
  }, [reduced, beforeReady]);

  const plays = hasVideo && armed && visible && !reduced && (variant === "hero" ? active === null || active === id : active === id);

  // Play / pause, and keep the two clips frame-locked while playing.
  useEffect(() => {
    const a = after.current;
    const b = before.current;
    if (!a || !b) return;
    if (!plays) {
      a.pause();
      b.pause();
      return;
    }
    void a.play().catch(() => undefined);
    void b.play().catch(() => undefined);
    let raf = 0;
    const tick = () => {
      if (Math.abs(b.currentTime - a.currentTime) > 0.08) b.currentTime = a.currentTime;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [plays]);

  // Hero: the continuous before→after sweep.
  useEffect(() => {
    if (variant !== "hero" || !beforeReady || reduced || !visible || !(active === null || active === id)) return;
    let raf = 0;
    // Pick up from wherever the wipe is: on the after (where it rests while loading), join the loop at its hold on the after.
    const t0 = performance.now() - (pos.current <= 0.5 ? HOLD_BEFORE + SWEEP : 0);
    const tick = (now: number) => {
      apply(heroPos(now - t0));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [variant, beforeReady, reduced, visible, active, id]);

  // Cards: one idle sweep every ~8 s, staggered by position.
  useEffect(() => {
    if (variant !== "card" || !beforeReady || reduced || !visible) return;
    let raf = 0;
    let timer = 0;
    const sweep = () => {
      if (hover.current || loops.get() !== null) {
        timer = window.setTimeout(sweep, CARD_EVERY);
        return;
      }
      const t0 = performance.now();
      const dur = SWEEP;
      const tick = (now: number) => {
        if (hover.current) return;
        const k = Math.min(1, (now - t0) / dur);
        // after → before → after
        apply(k < 0.5 ? 100 * ease(k * 2) : 100 - 100 * ease((k - 0.5) * 2));
        if (k < 1) raf = requestAnimationFrame(tick);
        else timer = window.setTimeout(sweep, CARD_EVERY);
      };
      raf = requestAnimationFrame(tick);
    };
    timer = window.setTimeout(sweep, 1800 + index * 1600);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(timer);
    };
  }, [variant, beforeReady, reduced, visible, index]);

  const settle = (to: number) => {
    const from = pos.current;
    const t0 = performance.now();
    const tick = (now: number) => {
      if (hover.current) return;
      const k = Math.min(1, (now - t0) / 320);
      apply(from + (to - from) * ease(k));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const cardHandlers =
    variant === "card" && hasVideo
      ? {
          onPointerEnter: () => {
            hover.current = true;
            setHovering(true);
            setArmed(true);
            loops.claim(id);
          },
          onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => {
            // Until the before has a frame, stay on the after (its poster): the left side would paint black.
            if (!beforeReady) return;
            const r = e.currentTarget.getBoundingClientRect();
            apply(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)));
          },
          onPointerLeave: () => {
            hover.current = false;
            setHovering(false);
            loops.release(id);
            settle(reduced ? 50 : 0);
          },
        }
      : {};

  const init = reduced && beforeReady ? 50 : pos.current;
  const style = { "--wipe": `${init}%`, "--line": init <= 0.5 || init >= 99.5 ? "0" : "1" } as CSSProperties;

  return (
    <div ref={root} className={cx("absolute inset-0", className)} style={style} {...cardHandlers}>
      {hasVideo ? (
        <>
          <video
            ref={before}
            src={armed ? ex.before_url : undefined}
            onLoadedData={() => setBeforeLoaded(ex.before_url)}
            muted
            loop
            playsInline
            preload={armed ? "auto" : "none"}
            className="absolute inset-0 h-full w-full object-cover"
            aria-hidden="true"
          />
          <video
            ref={after}
            src={armed ? ex.after_url : undefined}
            poster={ex.poster_url}
            muted
            loop
            playsInline
            preload={armed ? "auto" : "none"}
            className="absolute inset-0 h-full w-full object-cover"
            style={{ clipPath: "inset(0 0 0 var(--wipe))", WebkitClipPath: "inset(0 0 0 var(--wipe))" }}
            aria-hidden="true"
          />
          <div className="wipe-line pointer-events-none absolute inset-y-0" style={{ left: "var(--wipe)", opacity: "var(--line)" }} />
          {/* BEFORE / AFTER only while the cursor scrubs (the signature's labels). */}
          {variant === "card" && hovering ? (
            <>
              <span className="slate pointer-events-none absolute bottom-3 left-3 z-[2] rounded-[2px] bg-black/60 px-1.5 py-[3px]">Before</span>
              <span className="slate pointer-events-none absolute bottom-3 right-3 z-[2] rounded-[2px] bg-black/60 px-1.5 py-[3px]">After</span>
            </>
          ) : null}
          {variant === "card" && hovering && armed && !beforeReady ? (
            <span className="pointer-events-none absolute bottom-3 right-3 z-[2] flex h-5 w-5 items-center justify-center rounded-full bg-black/60 text-ivory/80" aria-hidden="true">
              <span className="inline-block h-2.5 w-2.5 animate-spin rounded-full border border-current border-r-transparent" />
            </span>
          ) : null}
        </>
      ) : (
        <img src={ex.poster_url} alt="" className="absolute inset-0 h-full w-full object-cover" draggable={false} />
      )}
    </div>
  );
}
