import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { cx } from "../lib/format";
import { useHoverLoop } from "../lib/loops";
import { thumbSeen } from "../lib/thumbs";
import { Skeleton, Slate } from "./ui";

/** A cropped thumbnail. A portrait or square picture (a person, most often) is face first (.thumb-face: its top
 * stays); a video frame or a wide plate stays centred. With `video`, hovering the card's picture plays it in place
 * (HoverVideo). */
export function Thumb({ src, alt = "", className, eager, video }: { src?: string; alt?: string; className?: string; eager?: boolean; video?: string }) {
  // State is per src: a new src starts over instead of keeping the old "error" (which drew black).
  const [st, setSt] = useState<{ src?: string; s: "loading" | "ok" | "err"; tall?: boolean }>(() => ({ src, s: !src ? "err" : thumbSeen.has(src) ? "ok" : "loading" }));
  const state = st.src === src ? st.s : !src ? "err" : thumbSeen.has(src) ? "ok" : "loading";
  const tall = st.src === src && !!st.tall;
  const ref = useRef<HTMLImageElement>(null);
  const loaded = (el: HTMLImageElement, s: string) => {
    thumbSeen.add(s);
    setSt({ src: s, s: "ok", tall: el.naturalHeight >= el.naturalWidth * 0.9 });
  };
  useLayoutEffect(() => {
    const el = ref.current;
    // Cached images can finish before React listens.
    if (el && src && el.complete && el.naturalWidth > 0) loaded(el, src);
  }, [src]);
  return (
    <div className={cx("relative overflow-hidden bg-black", className)}>
      {state === "loading" ? <div className="skeleton absolute inset-0 rounded-none" /> : null}
      {src ? (
        <img
          ref={ref}
          src={src}
          alt={alt}
          loading={eager ? "eager" : "lazy"}
          decoding="async"
          draggable={false}
          onLoad={(e) => loaded(e.currentTarget, src)}
          onError={() => setSt({ src, s: "err" })}
          className={cx("h-full w-full transition-opacity duration-150 ease-out", tall ? "thumb-face" : "object-cover", state === "ok" ? "opacity-100" : "opacity-0")}
        />
      ) : null}
      {video ? <HoverVideo src={video} /> : null}
    </div>
  );
}

/**
 * A video card's hover: the pointer on the card's picture (this element's parent) plays the clip muted, in place, over
 * the still; leaving goes back to the still. Light: nothing loads until the hover (preload none), one plays at a time
 * (useHoverLoop), and leaving pauses and unloads it.
 */
export function HoverVideo({ src }: { src: string }) {
  const anchor = useRef<HTMLSpanElement>(null);
  const frame = useRef<HTMLElement | null>(null);
  // The Thumb's frame (its slates and badges too), known before useHoverLoop starts listening to it.
  useLayoutEffect(() => {
    frame.current = anchor.current?.parentElement?.parentElement ?? null;
  }, []);
  const playing = useHoverLoop(frame);
  return (
    <span ref={anchor} className="contents">
      {playing ? <LoopVideo src={src} /> : null}
    </span>
  );
}

function LoopVideo({ src }: { src: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [shown, setShown] = useState(false);
  // The source is set here, not as a prop, so it comes back whenever the effect does.
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    v.src = src;
    void v.play().catch(() => undefined);
    return () => {
      // Unload, not only pause: the browser lets go of the clip.
      v.pause();
      v.removeAttribute("src");
      v.load();
    };
  }, [src]);
  return (
    <video
      ref={ref}
      muted
      loop
      playsInline
      preload="none"
      data-hover-video=""
      aria-hidden="true"
      onPlaying={() => setShown(true)}
      className={cx("pointer-events-none absolute inset-0 h-full w-full object-cover transition-opacity duration-150 ease-out", shown ? "opacity-100" : "opacity-0")}
    />
  );
}

/** A take in the Library: media first, burn-in slate, mono name. */
export function MediaCard({
  thumb,
  title,
  slate,
  live,
  corner,
  meta,
  video,
  onClick,
}: {
  thumb?: string;
  /** A clip or take: hovering the picture plays it in place. */
  video?: string;
  title: string;
  slate?: string[];
  live?: boolean;
  corner?: string;
  meta?: ReactNode;
  onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} className="group flex w-full min-w-0 flex-col rounded-card text-left">
      <div className="hairline-frame relative aspect-video w-full overflow-hidden rounded-card bg-black">
        <Thumb src={thumb} video={video} className="absolute inset-0 h-full w-full transition-transform duration-300 ease-out group-hover:scale-[1.02]" />
        {slate ? (
          <Slate at="tl" live={live}>
            {slate}
          </Slate>
        ) : null}
        {corner ? <Slate at="br">{corner}</Slate> : null}
      </div>
      <div className="mt-3 flex min-w-0 items-baseline justify-between gap-3">
        {/* The whole name (a made shot's is its first sentence): two lines before it ends in "…". */}
        <span className="line-clamp-2 break-words font-mono text-[13px] leading-snug text-ink transition-colors group-hover:text-ivory" title={title}>
          {title}
        </span>
        {meta ? <span className="slate shrink-0 text-ink3">{meta}</span> : null}
      </div>
    </button>
  );
}

export function MediaCardSkeleton() {
  return (
    <div aria-hidden="true">
      <Skeleton className="aspect-video w-full" />
      <Skeleton className="mt-3 h-3 w-2/3" />
    </div>
  );
}
