// One active loop at a time: the Home hero loops unless a card is being hovered, and a hovered video card (a wipe card
// or any HoverVideo) is the only one playing.
import { useEffect, useId, useState, useSyncExternalStore, type RefObject } from "react";

let active: string | null = null;
const subs = new Set<() => void>();
const emit = () => subs.forEach((f) => f());

export const loops = {
  claim(id: string) {
    if (active !== id) {
      active = id;
      emit();
    }
  },
  release(id: string) {
    if (active === id) {
      active = null;
      emit();
    }
  },
  get: () => active,
  subscribe(f: () => void) {
    subs.add(f);
    return () => {
      subs.delete(f);
    };
  },
};

export function useActiveLoop(): string | null {
  return useSyncExternalStore(loops.subscribe, loops.get);
}

const reducedQuery = typeof window !== "undefined" ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => !!reducedQuery?.matches);
  useEffect(() => {
    if (!reducedQuery) return;
    const on = () => setReduced(reducedQuery.matches);
    reducedQuery.addEventListener("change", on);
    return () => reducedQuery.removeEventListener("change", on);
  }, []);
  return reduced;
}

/** True while the element is (partly) on screen. */
export function useOnScreen(ref: RefObject<Element>, margin = "120px"): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setOn(e.isIntersecting), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [ref, margin]);
  return on;
}

/**
 * Hover to play, the one rule for every video card (HoverVideo; WipeMedia's cards claim the same way): the pointer on
 * `ref`'s element for a beat claims the one active loop; leaving releases it. True while this one is the one playing,
 * so a new hover anywhere (or the Home hero) stops it. Never with reduced motion.
 */
export function useHoverLoop(ref: RefObject<HTMLElement>, enabled = true, delay = 120): boolean {
  const id = useId();
  const active = useActiveLoop();
  const reduced = useReducedMotion();
  const on = enabled && !reduced;
  useEffect(() => {
    const el = ref.current;
    if (!el || !on) return;
    let timer = 0;
    const enter = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => loops.claim(id), delay);
    };
    const leave = () => {
      window.clearTimeout(timer);
      loops.release(id);
    };
    el.addEventListener("pointerenter", enter);
    el.addEventListener("pointerleave", leave);
    return () => {
      el.removeEventListener("pointerenter", enter);
      el.removeEventListener("pointerleave", leave);
      leave();
    };
  }, [ref, on, delay, id]);
  return on && active === id;
}

/** Ease in-out (cubic). */
export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
