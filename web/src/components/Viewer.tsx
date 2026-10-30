import { Maximize2, Minimize2, Minus, Plus } from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";
import { addPoint } from "../actions";
import { ALT_KEY, cx } from "../lib/format";
import { usePlayer, usePlayerState } from "../lib/player";
import { useApp, type ViewMode } from "../store";
import type { Shot, TaskId, Version } from "../types";
import { CropLayer } from "./Prepare";
import { Button, Hint, ReceiptLegend, Slate, Spinner } from "./ui";

const VIDEO_PROPS = {
  muted: true,
  playsInline: true,
  preload: "auto" as const,
  disablePictureInPicture: true,
  controls: false,
  draggable: false,
};

/** Fit a w×h frame inside the element, like object-fit: contain, and report the box. */
function useFit(ref: RefObject<HTMLElement>, w: number, h: number) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const cw = el.clientWidth;
      const ch = el.clientHeight;
      if (!cw || !ch || !w || !h) return;
      const a = w / h;
      const fit = cw / ch > a ? { w: Math.round(ch * a), h: ch } : { w: cw, h: Math.round(cw / a) };
      setSize((s) => (s.w === fit.w && s.h === fit.h ? s : fit));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, w, h]);
  return size;
}

function FollowerVideo({ src, poster, className, style }: { src: string; poster?: string; className?: string; style?: CSSProperties }) {
  const player = usePlayer();
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    player.addFollower(el);
    return () => player.removeFollower(el);
  }, [player, src]);
  return <video ref={ref} src={src} poster={poster || undefined} className={className} style={style} {...VIDEO_PROPS} aria-hidden="true" />;
}

const isVideoUrl = (u: string) => /\.(mp4|webm|mov|m4v)(\?|#|$)/i.test(u);

export interface StageProps {
  shot: Shot;
  version: Version;
  beforeUrl: string;
  /** The before take's still, shown on its side of the wipe until its video has a frame. */
  beforePoster?: string;
  compare: boolean;
  view: ViewMode;
  wipe: boolean;
  wipePos: number;
  onWipe: (pos: number) => void;
  selecting: TaskId | null;
  /** Trim & crop is open: the crop frame goes over the picture. */
  cropping?: boolean;
  /** Burn-in slate parts for the take on screen. */
  slate: Array<string | null | undefined>;
  /** Slate for the wipe's left side (shown in Before view). */
  beforeSlate?: Array<string | null | undefined>;
}

export function Stage({ shot, version, beforeUrl, beforePoster, compare, view, wipe, wipePos, onWipe, selecting, cropping, slate, beforeSlate }: StageProps) {
  const player = usePlayer();
  const snap = usePlayerState();
  const box = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const size = useFit(box, shot.width, shot.height);
  const [reload, setReload] = useState(0);
  const [everReady, setEverReady] = useState(false);
  const [livePos, setLivePos] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [full, setFull] = useState(false);
  const panDrag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const panMoved = useRef(false);
  const zoomRef = useRef(1);
  zoomRef.current = zoom;

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [shot.id]);

  useEffect(() => {
    const on = () => setFull(document.fullscreenElement === box.current);
    document.addEventListener("fullscreenchange", on);
    return () => document.removeEventListener("fullscreenchange", on);
  }, []);

  useEffect(() => {
    if (snap.ready) setEverReady(true);
  }, [snap.ready]);

  const showCompare = compare && !selecting;
  const mode: ViewMode = showCompare ? view : "after";
  const liveZoom = selecting || cropping ? 1 : zoom;
  const setZoomTo = (z: number) => {
    setZoom(z);
    if (z === 1) setPan({ x: 0, y: 0 });
  };
  const clampPan = (x: number, y: number, z: number) => {
    const maxX = Math.max(0, ((z - 1) * size.w) / 2);
    const maxY = Math.max(0, ((z - 1) * size.h) / 2);
    return { x: Math.max(-maxX, Math.min(maxX, x)), y: Math.max(-maxY, Math.min(maxY, y)) };
  };

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (selecting || cropping) return;
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      const next = Math.min(8, Math.max(1, zoomRef.current * factor));
      const snapped = next < 1.02 ? 1 : next;
      if (snapped === 1) setPan({ x: 0, y: 0 });
      setZoom(snapped);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [selecting, cropping]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable) return;
      if (e.key.toLowerCase() !== "f" || e.metaKey || e.ctrlKey || e.altKey) return;
      const node = box.current;
      if (!node) return;
      e.preventDefault();
      if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
      else void node.requestFullscreen().catch(() => undefined);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const wipeOn = showCompare && mode === "after" && wipe;
  const dragging = livePos != null;
  const pos = livePos ?? wipePos;
  const receiptUrl = version.receipt_url;

  const mainClip = wipeOn ? `inset(0 0 0 ${pos}%)` : mode === "before" ? "inset(0 0 0 100%)" : undefined;

  return (
    <div ref={box} className={cx("relative flex min-h-0 flex-1 items-center justify-center overflow-hidden", full && "bg-black")}>
      <div
        ref={frameRef}
        data-stage=""
        className={cx("relative overflow-hidden bg-black outline outline-1 outline-offset-0 outline-line", size.w ? "" : "opacity-0")}
        style={{
          width: size.w || undefined,
          height: size.h || undefined,
          aspectRatio: size.w ? undefined : `${shot.width} / ${shot.height}`,
          transform: liveZoom === 1 ? undefined : `translate(${pan.x}px, ${pan.y}px) scale(${liveZoom})`,
          cursor: liveZoom > 1 && !selecting && !cropping ? "grab" : undefined,
        }}
        onPointerDown={(e) => {
          if (e.button !== 0 || liveZoom <= 1 || selecting || cropping) return;
          panMoved.current = false;
          panDrag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = panDrag.current;
          if (!d) return;
          const dx = e.clientX - d.x;
          const dy = e.clientY - d.y;
          if (Math.hypot(dx, dy) > 3) panMoved.current = true;
          setPan(clampPan(d.px + dx, d.py + dy, liveZoom));
        }}
        onPointerUp={() => {
          panDrag.current = null;
        }}
        onClick={() => {
          if (panMoved.current) {
            panMoved.current = false;
            return;
          }
          if (!selecting) player.toggle();
        }}
      >
        {/* First load: a shimmer under the videos. Each video shows its own take's still (poster) until it has a
            frame, so opening a shot or switching takes never paints black, and each side of the wipe shows its own take. */}
        {!snap.ready && !snap.failed && !everReady ? <div className="skeleton pointer-events-none absolute inset-0 rounded-none" /> : null}
        {showCompare ? (
          <FollowerVideo
            key={`before:${beforeUrl}:${reload}`}
            src={beforeUrl}
            // A before still that's the same image as the after's can't be the before: no poster beats the wrong one.
            poster={beforePoster && beforePoster !== version.thumb_url ? beforePoster : undefined}
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : null}
        <video
          key={`main:${version.video_url}:${reload}`}
          ref={player.setMain}
          src={version.video_url}
          poster={version.thumb_url || undefined}
          className="absolute inset-0 h-full w-full object-contain"
          style={{ clipPath: mainClip, WebkitClipPath: mainClip }}
          {...VIDEO_PROPS}
          aria-label={`${shot.name} video`}
        />

        {mode === "receipt" && receiptUrl ? (
          isVideoUrl(receiptUrl) ? (
            <FollowerVideo
              key={`receipt:${receiptUrl}:${reload}`}
              src={receiptUrl}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              style={{ mixBlendMode: "screen", opacity: 0.55 }}
            />
          ) : (
            <img src={receiptUrl} alt="" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ mixBlendMode: "screen", opacity: 0.55 }} />
          )
        ) : null}

        {selecting ? <SelectionLayer shot={shot} task={selecting} width={size.w} height={size.h} /> : null}
        {cropping ? <CropLayer shot={shot} width={size.w} height={size.h} /> : null}

        {wipeOn ? (
          <WipeHandle
            pos={pos}
            dragging={dragging}
            frameRef={frameRef}
            onDrag={setLivePos}
            onCommit={(p) => {
              setLivePos(null);
              onWipe(p);
            }}
          />
        ) : null}

        {/* The burn-in slate for the take on screen. While comparing without the wipe, say which side is showing. */}
        <Slate at="tl" live>
          {showCompare && mode === "before" ? beforeSlate ?? ["Take 01", "Original", "Before"] : slate}
        </Slate>
        {mode === "receipt" ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-3 z-[2] flex justify-center animate-fade-in">
            {receiptUrl ? <ReceiptLegend receipt={version.receipt} /> : <Slate className="rounded-[2px] bg-black/70 px-2 py-1">This take has no receipt</Slate>}
          </div>
        ) : null}

        {/* Loading / error */}
        {!snap.ready && !snap.failed && everReady ? (
          <span className="pointer-events-none absolute right-3 top-3 z-[2] text-ivory/70">
            <Spinner size={13} />
          </span>
        ) : null}
        {snap.failed ? (
          <div className="absolute inset-0 z-[3] flex flex-col items-center justify-center gap-4 bg-bg text-center" onClick={(e) => e.stopPropagation()}>
            <span className="slate">Error</span>
            <p className="text-[15px] text-ink">This video didn't load.</p>
            <Button variant="secondary" size="sm" onClick={() => setReload((n) => n + 1)}>
              Try again
            </Button>
          </div>
        ) : null}
      </div>
      <div className="absolute bottom-3 right-3 z-[4] flex items-center gap-1.5">
        <ZoomButton label="100%" on={Math.abs(zoom - 1) < 0.05} onClick={() => setZoomTo(1)} />
        <ZoomButton label="200%" on={Math.abs(zoom - 2) < 0.05} onClick={() => setZoomTo(2)} />
        <button
          type="button"
          aria-label={full ? "Leave full screen" : "Full screen"}
          aria-keyshortcuts="F"
          onClick={() => {
            const node = box.current;
            if (!node) return;
            if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
            else void node.requestFullscreen().catch(() => undefined);
          }}
          className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-black/70 text-ivory hover:bg-black"
        >
          {full ? <Minimize2 size={14} strokeWidth={1.5} aria-hidden="true" /> : <Maximize2 size={14} strokeWidth={1.5} aria-hidden="true" />}
        </button>
        <Hint text="Scroll or pinch to zoom. Drag to move. F fills the screen, Esc leaves it. The wipe zooms with the picture." />
      </div>
    </div>
  );
}

function ZoomButton({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        "h-7 rounded-full px-2 font-mono text-[11px] tracking-[0.08em]",
        on ? "bg-ivory text-bg" : "bg-black/70 text-ivory/80 hover:bg-black hover:text-ivory",
      )}
    >
      {label}
    </button>
  );
}

/**
 * The signature: a 1 px ivory line with a black outer stroke. BEFORE / AFTER labels appear only while dragging.
 * The grip is a small ivory bar so the line can be found without a big knob.
 */
function WipeHandle({
  pos,
  dragging,
  frameRef,
  onDrag,
  onCommit,
}: {
  pos: number;
  dragging: boolean;
  frameRef: RefObject<HTMLDivElement>;
  onDrag: (p: number) => void;
  onCommit: (p: number) => void;
}) {
  const active = useRef(false);
  const last = useRef(pos);
  const at = (clientX: number) => {
    const r = frameRef.current?.getBoundingClientRect();
    if (!r) return pos;
    return Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
  };
  const handlers = {
    onPointerDown: (e: ReactPointerEvent) => {
      e.stopPropagation();
      e.preventDefault();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      active.current = true;
      last.current = at(e.clientX);
      onDrag(last.current);
    },
    onPointerMove: (e: ReactPointerEvent) => {
      if (!active.current) return;
      last.current = at(e.clientX);
      onDrag(last.current);
    },
    onPointerUp: (e: ReactPointerEvent) => {
      if (!active.current) return;
      active.current = false;
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      onCommit(last.current);
    },
    onClick: (e: ReactMouseEvent) => e.stopPropagation(),
  };
  return (
    <div className="absolute inset-y-0 z-[1]" style={{ left: `${pos}%` }}>
      <div className="wipe-line pointer-events-none absolute inset-y-0 -translate-x-1/2" />
      <div className="absolute inset-y-0 w-7 -translate-x-1/2 cursor-ew-resize touch-none" {...handlers} />
      <div
        role="slider"
        tabIndex={0}
        aria-label="Wipe between before and after"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos)}
        aria-valuetext={`${Math.round(pos)}% before`}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
            e.preventDefault();
            e.stopPropagation();
            const d = (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 10 : 2);
            onCommit(Math.max(0, Math.min(100, pos + d)));
          }
        }}
        className="absolute top-1/2 h-9 w-[5px] -translate-x-1/2 -translate-y-1/2 cursor-ew-resize touch-none rounded-full bg-ivory shadow-[0_0_0_1px_#000]"
        {...handlers}
      />
      {dragging ? (
        <>
          <span className="slate pointer-events-none absolute right-3 top-3 whitespace-nowrap rounded-[2px] bg-black/70 px-1.5 py-[3px] text-ivory">Before</span>
          <span className="slate pointer-events-none absolute left-3 top-3 whitespace-nowrap rounded-[2px] bg-black/70 px-1.5 py-[3px] text-ivory">After</span>
        </>
      ) : null}
    </div>
  );
}

// --- click-to-select ---------------------------------------------------------------------------------------------

const SELECT_PROMPT: Partial<Record<TaskId, string>> = {
  remove: "Click what should disappear",
  recast: "Click who or what to replace",
  environment: "Click what to keep as shot",
};

function SelectionLayer({ shot, task, width, height }: { shot: Shot; task: TaskId; width: number; height: number }) {
  const player = usePlayer();
  const sel = useApp((s) => (s.selection && s.selection.shotId === shot.id ? s.selection : null));
  const snap = usePlayerState();
  if (!sel) return null;
  const onFrame = !sel.points.length || sel.frame === snap.frame;
  // Recast keeps the keep-an-area hint under its "More options", so the picture only says what most people need. Change
  // weather's pick is what to keep, so "keep an area" would say the opposite.
  const keepHint = task === "recast" || task === "environment" ? "" : ` · ${ALT_KEY}-click to keep an area`;
  return (
    <div
      className="absolute inset-0 z-[1] cursor-crosshair"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        const r = e.currentTarget.getBoundingClientRect();
        const x = Math.round(((e.clientX - r.left) / r.width) * shot.width);
        const y = Math.round(((e.clientY - r.top) / r.height) * shot.height);
        if (player.getSnapshot().playing) player.pause();
        addPoint({ x, y, positive: !e.altKey }, player.getSnapshot().frame);
      }}
      onClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {sel.maskUrl ? <MaskOverlay url={sel.maskUrl} width={width} height={height} dim={!onFrame || sel.loading} /> : null}
      {sel.points.map((p, i) => (
        <span
          key={i}
          className={cx(
            "pointer-events-none absolute flex h-4 w-4 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full animate-fade-in",
            p.positive ? "bg-ivory text-black shadow-[0_0_0_1.5px_#000]" : "bg-black text-ivory shadow-[0_0_0_1.5px_#EEE9E0]",
            !onFrame && "opacity-40",
          )}
          style={{ left: `${(p.x / shot.width) * 100}%`, top: `${(p.y / shot.height) * 100}%` }}
        >
          {p.positive ? <Plus size={10} strokeWidth={3} /> : <Minus size={10} strokeWidth={3} />}
        </span>
      ))}
      {/* On a phone the zoom buttons fill the bottom edge, so the line sits above them. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3 max-[767px]:bottom-14">
        <span className="slate inline-flex items-center gap-2 rounded-[2px] bg-black/70 px-2 py-1 text-center text-ivory/85">
          {sel.loading ? (
            <>
              <Spinner size={11} /> Finding it…
            </>
          ) : sel.error ? (
            <>Error · {sel.error}</>
          ) : sel.maskUrl && sel.auto && sel.many ? (
            <>Click someone else to swap them too</>
          ) : sel.maskUrl && sel.auto ? (
            <>Click someone else to switch</>
          ) : sel.maskUrl ? (
            <>Got it · click to add more{keepHint}</>
          ) : sel.kept === "the preset" ? (
            <>From the preset · click to change</>
          ) : sel.kept ? (
            <>Same pick as {sel.kept} · click to add more</>
          ) : sel.finding && !sel.points.length && !sel.text ? (
            // Until the finder has answered, the picture doesn't ask for a click it's about to make for you.
            <>
              <Spinner size={11} /> Finding who's in the shot…
            </>
          ) : (
            <>
              {SELECT_PROMPT[task] ?? "Click to select"}
              {keepHint}
            </>
          )}
        </span>
      </div>
    </div>
  );
}

/** A mask PNG drawn as an ivory tint with the signature outline (1 px ivory inside a black stroke). */
export function MaskOverlay({ url, width, height, dim }: { url: string; width: number; height: number; dim?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!width || !height) return;
    let cancelled = false;
    const img = new Image();
    img.decoding = "async";
    img.onload = () => {
      if (cancelled || !ref.current) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W = Math.round(width * dpr);
      const H = Math.round(height * dpr);
      // 1. mask → alpha (luminance × alpha), at most 1280 px wide for speed.
      const s = Math.min(1, 1280 / img.naturalWidth);
      const mw = Math.max(1, Math.round(img.naturalWidth * s));
      const mh = Math.max(1, Math.round(img.naturalHeight * s));
      const m = document.createElement("canvas");
      m.width = mw;
      m.height = mh;
      const mg = m.getContext("2d", { willReadFrequently: true })!;
      mg.drawImage(img, 0, 0, mw, mh);
      const data = mg.getImageData(0, 0, mw, mh);
      const d = data.data;
      for (let i = 0; i < d.length; i += 4) {
        const a = ((0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) * d[i + 3]) / 255;
        d[i] = 238;
        d[i + 1] = 233;
        d[i + 2] = 224;
        d[i + 3] = a > 24 ? 255 : 0;
      }
      mg.putImageData(data, 0, 0);
      const layer = () => {
        const c = document.createElement("canvas");
        c.width = W;
        c.height = H;
        return [c, c.getContext("2d")!] as const;
      };
      const [shape, sg] = layer();
      sg.drawImage(m, 0, 0, W, H);
      const dilate = (r: number) => {
        const [c, g] = layer();
        for (let k = 0; k < 16; k++) {
          const t = (k / 16) * Math.PI * 2;
          g.drawImage(shape, Math.cos(t) * r, Math.sin(t) * r);
        }
        return c;
      };
      const d1 = dilate(1 * dpr);
      const d2 = dilate(2 * dpr);
      const c = ref.current;
      c.width = W;
      c.height = H;
      const g = c.getContext("2d")!;
      g.clearRect(0, 0, W, H);
      // Black outer stroke.
      g.drawImage(d2, 0, 0);
      g.globalCompositeOperation = "source-in";
      g.fillStyle = "rgba(0,0,0,0.9)";
      g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = "destination-out";
      g.drawImage(d1, 0, 0);
      // Ivory 1 px line.
      g.globalCompositeOperation = "source-over";
      const [ring, rg] = layer();
      rg.drawImage(d1, 0, 0);
      rg.globalCompositeOperation = "destination-out";
      rg.drawImage(shape, 0, 0);
      g.drawImage(ring, 0, 0);
      // Tint.
      g.globalAlpha = 0.28;
      g.drawImage(shape, 0, 0);
      g.globalAlpha = 1;
    };
    img.src = url;
    return () => {
      cancelled = true;
    };
  }, [url, width, height]);
  return (
    <canvas
      ref={ref}
      className={cx("pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-150", dim ? "opacity-40" : "opacity-100")}
      aria-hidden="true"
    />
  );
}
