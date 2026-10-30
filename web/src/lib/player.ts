// Keeps stacked <video> elements (after = main, before / receipt = followers) frame-locked, and publishes the
// current frame to the transport, timecode and scrubber without re-rendering the whole shot view.
import { createContext, useContext, useSyncExternalStore } from "react";

export interface PlayerSnapshot {
  playing: boolean;
  frame: number;
  ready: boolean;
  failed: boolean;
}

export class Player {
  fps = 24;
  frames = 1;
  private main: HTMLVideoElement | null = null;
  private followers = new Set<HTMLVideoElement>();
  private snap: PlayerSnapshot = { playing: false, frame: 0, ready: false, failed: false };
  private listeners = new Set<() => void>();
  private raf = 0;
  private wantPlay = false;
  private cleanup = new Map<HTMLVideoElement, () => void>();

  configure(fps: number, frames: number) {
    this.fps = fps > 0 ? fps : 24;
    this.frames = Math.max(1, frames | 0);
  }

  // --- subscription ----------------------------------------------------------------------------------------------
  subscribe = (l: () => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };
  getSnapshot = () => this.snap;
  private set(p: Partial<PlayerSnapshot>) {
    const next = { ...this.snap, ...p };
    if (next.playing === this.snap.playing && next.frame === this.snap.frame && next.ready === this.snap.ready && next.failed === this.snap.failed) return;
    this.snap = next;
    this.listeners.forEach((l) => l());
  }

  // --- elements --------------------------------------------------------------------------------------------------
  setMain = (el: HTMLVideoElement | null) => {
    if (el === this.main) return;
    if (this.main) this.detach(this.main);
    this.main = el;
    if (!el) return;
    this.set({ ready: el.readyState >= 2, failed: false });
    const onMeta = () => {
      this.seekElement(el, this.snap.frame);
      if (this.wantPlay) void el.play().catch(() => undefined);
    };
    const onData = () => this.set({ ready: true, failed: false });
    const onErr = () => this.set({ failed: true });
    const onSeeked = () => {
      if (!this.wantPlay) this.set({ frame: this.frameOf(el.currentTime) });
    };
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("loadeddata", onData);
    el.addEventListener("error", onErr);
    el.addEventListener("seeked", onSeeked);
    if (el.readyState >= 1) onMeta();
    this.cleanup.set(el, () => {
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("loadeddata", onData);
      el.removeEventListener("error", onErr);
      el.removeEventListener("seeked", onSeeked);
    });
  };

  addFollower = (el: HTMLVideoElement | null, prev?: HTMLVideoElement | null) => {
    if (prev) {
      this.followers.delete(prev);
      this.detach(prev);
    }
    if (!el) return;
    this.followers.add(el);
    const onMeta = () => {
      el.currentTime = this.main?.currentTime ?? this.timeOf(this.snap.frame);
      if (this.wantPlay) void el.play().catch(() => undefined);
    };
    el.addEventListener("loadedmetadata", onMeta);
    if (el.readyState >= 1) onMeta();
    this.cleanup.set(el, () => el.removeEventListener("loadedmetadata", onMeta));
  };

  removeFollower = (el: HTMLVideoElement) => {
    this.followers.delete(el);
    this.detach(el);
  };

  private detach(el: HTMLVideoElement) {
    this.cleanup.get(el)?.();
    this.cleanup.delete(el);
  }

  // --- transport -------------------------------------------------------------------------------------------------
  play = () => {
    if (!this.main) return;
    this.wantPlay = true;
    if (this.snap.frame >= this.frames - 1) this.seek(0);
    void this.main.play().catch(() => {
      this.wantPlay = false;
      this.set({ playing: false });
    });
    this.followers.forEach((f) => void f.play().catch(() => undefined));
    this.set({ playing: true });
    this.loop();
  };

  pause = () => {
    this.wantPlay = false;
    cancelAnimationFrame(this.raf);
    this.main?.pause();
    this.followers.forEach((f) => f.pause());
    if (this.main) {
      const frame = this.frameOf(this.main.currentTime);
      this.seek(frame);
    }
    this.set({ playing: false });
  };

  toggle = () => (this.snap.playing ? this.pause() : this.play());

  seek = (frame: number) => {
    const f = Math.max(0, Math.min(this.frames - 1, Math.round(frame)));
    if (this.main) this.seekElement(this.main, f);
    this.followers.forEach((v) => this.seekElement(v, f));
    this.set({ frame: f });
  };

  step = (delta: number) => {
    if (this.snap.playing) this.pause();
    this.seek(this.snap.frame + delta);
  };

  stepSeconds = (s: number) => this.step(Math.round(s * this.fps));

  /** Stop the playback loop. Element listeners are released by setMain(null) / removeFollower as React unmounts. */
  stop() {
    this.wantPlay = false;
    cancelAnimationFrame(this.raf);
  }

  // --- internals -------------------------------------------------------------------------------------------------
  private timeOf(frame: number) {
    return (frame + 0.5) / this.fps;
  }
  private frameOf(t: number) {
    return Math.max(0, Math.min(this.frames - 1, Math.floor(t * this.fps + 1e-3)));
  }
  private seekElement(el: HTMLVideoElement, frame: number) {
    if (el.readyState < 1) return;
    const dur = isFinite(el.duration) ? el.duration : Infinity;
    el.currentTime = Math.min(this.timeOf(frame), Math.max(0, dur - 0.001));
  }

  private loop = () => {
    cancelAnimationFrame(this.raf);
    const tick = () => {
      const m = this.main;
      if (!m || !this.wantPlay) return;
      // Loop within the shot's frame range (also when the proxy is longer than a trimmed shot).
      const end = this.frames / this.fps;
      if (m.ended || m.currentTime >= end) {
        m.currentTime = this.timeOf(0);
        if (m.paused) void m.play().catch(() => undefined);
      }
      const t = m.currentTime;
      this.set({ frame: this.frameOf(t), playing: true });
      const tol = 0.75 / this.fps;
      this.followers.forEach((f) => {
        if (f.readyState < 1) return;
        if (Math.abs(f.currentTime - t) > tol) f.currentTime = t;
        if (f.paused && !m.paused) void f.play().catch(() => undefined);
      });
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  };
}

export const PlayerContext = createContext<Player | null>(null);

export function usePlayer(): Player {
  const p = useContext(PlayerContext);
  if (!p) throw new Error("usePlayer outside PlayerContext");
  return p;
}

export function usePlayerState(): PlayerSnapshot {
  const p = usePlayer();
  return useSyncExternalStore(p.subscribe, p.getSnapshot);
}

/** The player of the open shot, for global keyboard shortcuts. */
export const activePlayer: { current: Player | null } = { current: null };
