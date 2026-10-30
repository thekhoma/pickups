// Create's Text to video and Image to video tabs: LTX-2.5, the one video model. Words (and, for Image to video, a
// picture to start from), then aspect, size and length. The picture covers the frame and the overflow is trimmed from
// the middle out, as the graph does (ImageScale crop center): the slot shows what stays. Done, the video is a new shot.
import { ImagePlus, Upload, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent, type RefObject } from "react";
import { createPortal } from "react-dom";
import { ensureProject, generateVideo, loadImages, openSetup, openShot, targetProject, uploadPicture } from "../actions";
import { cx, isActive } from "../lib/format";
import { madeIn, tookOf } from "../lib/runtime";
import { isLocked, nextSeed, type SeedState } from "../lib/seed";
import { coverCrop, hasSize, resolutionFor, VIDEO_ASPECTS, VIDEO_DEFAULTS, VIDEO_DURATIONS, type VideoAspect, type VideoResolution } from "../lib/video";
import { useApp } from "../store";
import type { Ref } from "../types";
import { placeMenu, type MenuPlace } from "../lib/placeMenu";
import { FrameIcon, Picker } from "./Picker";
import { Cancelled, RunStrip } from "./RunStatus";
import { Button, Notice, SeedLock, Spinner, Hint } from "./ui";

interface Picture {
  ref: Ref;
  url: string;
  name: string;
}

const isImage = (f: File) => f.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(f.name);

/** The composer's last run and its Lock seed, kept while you move around the app. */
let last: SeedState & { jobId: string | null } = { seed: null, locked: false, jobId: null };

export function VideoMode({ mode, text }: { mode: "text" | "image"; text?: string }) {
  const status = useApp((s) => s.status);
  const [prompt, setPrompt] = useState(text ?? "");
  const [aspect, setAspect] = useState<VideoAspect>(VIDEO_DEFAULTS.aspect);
  const [res, setRes] = useState<VideoResolution>(VIDEO_DEFAULTS.resolution);
  const [duration, setDuration] = useState(VIDEO_DEFAULTS.duration);
  const [picture, setPicture] = useState<Picture | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState(last);
  useEffect(() => {
    last = run;
  }, [run]);
  const job = useApp((s) => (run.jobId ? s.jobs[run.jobId] : undefined));
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (text) area.current?.focus();
  }, [text]);

  const tile = status?.tiles?.video;
  // Known to be missing something (models or nodes): say so and link to Setup, never a dead button.
  const needsSetup = !!status?.connected && tile?.ready === false && (!!tile.missing?.length || !!tile.missing_nodes?.length);
  const used = resolutionFor(aspect, res);
  const ready = !!prompt.trim() && (mode === "text" || !!picture) && !busy && !needsSetup;

  const go = async () => {
    if (!ready) return;
    setBusy(true);
    setError(null);
    const seed = nextSeed(run);
    const res = await generateVideo({
      mode, prompt: prompt.trim(), aspect, resolution: used, duration, image: mode === "image" ? picture?.ref ?? null : null, seed, seed_locked: isLocked(run),
    });
    setBusy(false);
    if (res.error) setError(res.error);
    const made = res.job;
    if (made) {
      const sent = (made.params as { seed?: number } | undefined)?.seed ?? seed;
      setRun((r) => ({ ...r, seed: sent, jobId: made.id }));
    }
  };
  const took = madeIn(tookOf(job));

  return (
    <form
      className="p-4 max-[640px]:p-3"
      onSubmit={(e) => {
        e.preventDefault();
        void go();
      }}
    >
      <div className={cx("flex gap-3", mode === "image" && "items-stretch")}>
        {mode === "image" ? <StartPicture picture={picture} aspect={aspect} onPick={(p) => (setPicture(p), setError(null))} onError={setError} /> : null}
        <textarea
          ref={area}
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              void go();
            }
          }}
          placeholder={mode === "image" ? "Describe the motion" : "Describe the shot"}
          aria-label={mode === "image" ? "Describe the motion" : "Describe the shot"}
          className="field h-auto min-h-[96px] min-w-0 flex-1 resize-none py-3 text-[16px] leading-relaxed"
        />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="inline-flex h-9 items-center gap-2 rounded-full border border-line px-3.5 text-[13px] text-ink" aria-label="Model: LTX-2.5">
          <span className="slate text-ink4">Model</span>
          LTX-2.5
        </span>
        {mode === "image" ? (
          <span className="inline-flex h-9 items-center gap-1.5 rounded-full border border-line px-3.5 text-[13px] text-ink" aria-label="The picture is the start frame">
            <span className="slate text-ink4">Picture</span>
            Start frame
            <Hint text="LTX-2.5 starts the video on this picture: it is frame 1, and your words describe what happens next. Reference pictures that guide a character or a place without being frame 1 aren't available with LTX-2.5 here yet." />
          </span>
        ) : null}
        <Picker
          label="Aspect"
          value={aspect}
          options={VIDEO_ASPECTS.map((a) => ({ id: a, label: a, left: <FrameIcon aspect={a} /> }))}
          onChange={(v) => setAspect(v as VideoAspect)}
        />
        <Picker
          label="Size"
          value={used}
          options={[
            { id: "720p", label: "720p" },
            { id: "1080p", label: "1080p", sub: hasSize(aspect, "1080p") ? undefined : "16:9 only", disabled: !hasSize(aspect, "1080p") },
          ]}
          onChange={(v) => setRes(v as VideoResolution)}
        />
        <Picker
          label="Length"
          value={String(duration)}
          options={VIDEO_DURATIONS.map((d) => ({ id: String(d.seconds), label: `${d.seconds} s` }))}
          onChange={(v) => setDuration(Number(v))}
        />
        <SeedLock size="md" seed={run.seed} locked={run.locked} onChange={(locked) => setRun((r) => ({ ...r, locked }))} />
        {needsSetup ? (
          <span className="ml-auto flex items-center gap-2 max-[640px]:w-full">
            <span className="slate text-ink3">LTX-2.5 needs setup</span>
            <Button variant="quiet" size="sm" onClick={() => openSetup()}>
              Open setup
            </Button>
          </span>
        ) : null}
        <Button type="submit" variant="primary" size="lg" loading={busy} disabled={!ready} className={cx(!needsSetup && "ml-auto", "max-[640px]:w-full")}>
          Generate
        </Button>
      </div>
      {error ? <Notice className="mt-3" tone="bad" text={error} /> : null}
      {job && isActive(job) ? <RunStrip job={job} label="Making your video" className="mt-4" /> : null}
      {job?.status === "canceled" ? <Cancelled className="mt-3" onRetry={() => void go()} /> : null}
      {job?.status === "done" && job.result_shot_id ? (
        <div className="mt-3 flex items-center gap-3" data-testid="video-done">
          <span className="slate text-ink3">{["Done", took].filter(Boolean).join(" · ")}</span>
          <Button variant="quiet" size="sm" onClick={() => openShot(job.result_shot_id!)}>
            Open
          </Button>
        </div>
      ) : null}
    </form>
  );
}

/** The picture Image to video starts from: drop one, upload one or choose one of the project's. Shown with the part the
 * chosen aspect keeps (the rest dimmed), centred as the render crops it. */
function StartPicture({ picture, aspect, onPick, onError }: { picture: Picture | null; aspect: VideoAspect; onPick: (p: Picture | null) => void; onError: (e: string | null) => void }) {
  const [open, setOpen] = useState(false);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const slot = useRef<HTMLButtonElement>(null);

  const upload = async (f: File | null | undefined) => {
    if (!f) return;
    if (!isImage(f)) {
      onError("That's not a picture. Use a JPG, PNG or WebP.");
      return;
    }
    setOpen(false);
    setUploading(true);
    try {
      const up = await uploadPicture(f);
      setDims(null);
      onPick({ ref: up.ref, url: up.url, name: f.name });
    } catch (e) {
      onError(e instanceof Error ? e.message : "That picture didn't upload. Try again.");
    } finally {
      setUploading(false);
    }
  };

  const drop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setOver(false);
    void upload(e.dataTransfer.files?.[0]);
  };

  const BOX_W = 152;
  const BOX_H = 96;
  const fit = dims ? Math.min(BOX_W / dims.w, BOX_H / dims.h) : 1;
  const crop = dims ? coverCrop(dims.w, dims.h, aspect) : null;

  return (
    <div className="relative shrink-0" onDragOver={(e) => (e.preventDefault(), setOver(true))} onDragLeave={() => setOver(false)} onDrop={drop}>
      <input ref={input} type="file" accept="image/*" className="hidden" tabIndex={-1} aria-hidden="true" onChange={(e) => (void upload(e.target.files?.[0]), (e.target.value = ""))} />
      <button
        ref={slot}
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={picture ? `Start frame: ${picture.name}. Change it` : "Choose the start frame"}
        onClick={() => setOpen((o) => !o)}
        className={cx(
          "relative flex h-full min-h-[96px] w-[152px] items-center justify-center overflow-hidden rounded-card border bg-black transition-colors duration-150",
          over ? "border-ivory/70" : picture ? "border-line" : "border-dashed border-line2 hover:border-ivory/50",
        )}
      >
        {uploading ? (
          <Spinner size={14} />
        ) : picture ? (
          <span className="relative block" style={dims ? { width: dims.w * fit, height: dims.h * fit } : { width: BOX_W, height: BOX_H }}>
            <img
              src={picture.url}
              alt=""
              draggable={false}
              onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
              className="absolute inset-0 h-full w-full object-contain"
            />
            {crop ? (
              <span
                aria-hidden="true"
                className="absolute rounded-[2px] shadow-[0_0_0_999px_rgba(13,12,11,0.62)] outline outline-1 outline-ivory/80"
                style={{ left: `${crop.x * 100}%`, top: `${crop.y * 100}%`, width: `${crop.w * 100}%`, height: `${crop.h * 100}%` }}
              />
            ) : null}
          </span>
        ) : (
          <span className="flex flex-col items-center gap-1.5 text-ink3">
            <ImagePlus size={18} strokeWidth={1.5} aria-hidden="true" />
            <span className="text-[12px]">Start frame</span>
          </span>
        )}
      </button>
      {picture && !uploading ? (
        <button
          type="button"
          onClick={() => (onPick(null), setDims(null))}
          aria-label="Remove the start frame"
          className="absolute right-1 top-1 rounded-full bg-black/70 p-1 text-ink2 transition-colors hover:text-ivory"
        >
          <X size={12} />
        </button>
      ) : null}
      {open ? (
        <PictureChooser
          anchor={slot}
          onClose={() => setOpen(false)}
          onUpload={() => input.current?.click()}
          onPick={(p) => {
            setOpen(false);
            setDims(null);
            onError(null);
            onPick(p);
          }}
        />
      ) : null}
    </div>
  );
}

/** The project's pictures (made images first, then the cast), and Upload. Portaled and placed like the pickers. */
function PictureChooser({ anchor, onClose, onUpload, onPick }: { anchor: RefObject<HTMLElement>; onClose: () => void; onUpload: () => void; onPick: (p: Picture) => void }) {
  const pid = targetProject();
  const images = useApp((s) => s.images[pid]);
  const cast = useApp((s) => s.cast[pid]);
  const box = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState<MenuPlace | null>(null);
  useEffect(() => {
    void loadImages(pid);
    void ensureProject(pid);
  }, [pid]);

  const items = useMemo(
    () => [
      ...(images ?? []).map((im) => ({ key: `image:${im.id}`, ref: { kind: "image", id: im.id } as Ref, thumb: im.thumb_url || im.url, url: im.url || im.thumb_url, name: im.prompt?.trim() || "Image" })),
      ...(cast ?? []).map((c) => ({ key: `cast:${c.id}`, ref: { kind: "cast", id: c.id } as Ref, thumb: c.thumb_url || c.url, url: c.url, name: c.name })),
    ],
    [images, cast],
  );

  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (box.current?.contains(t) || anchor.current?.contains(t)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [anchor, onClose]);

  useLayoutEffect(() => {
    const go = () => {
      const r = anchor.current?.getBoundingClientRect();
      const m = box.current;
      if (!r || !m) return;
      setPlace(placeMenu(r, { height: m.scrollHeight, width: m.offsetWidth }, { width: window.innerWidth, height: window.innerHeight }));
    };
    go();
    window.addEventListener("resize", go);
    return () => window.removeEventListener("resize", go);
  }, [anchor, items.length]);

  return createPortal(
    <div
      ref={box}
      role="dialog"
      aria-label="Start frame"
      className="fixed z-[80] flex w-[360px] flex-col overflow-hidden rounded-card border border-line2 bg-raised shadow-[0_24px_60px_-24px_rgba(0,0,0,0.9)]"
      style={place ? { left: place.left, top: place.top, bottom: place.bottom, maxHeight: place.maxH } : { left: 0, top: 0, visibility: "hidden" }}
    >
      <div className="shrink-0 border-b border-line p-1.5">
        <button type="button" onClick={onUpload} className="flex h-8 w-full items-center gap-2 rounded-[4px] px-2 text-left text-[14px] text-ivory hover:bg-line">
          <Upload size={14} strokeWidth={1.5} aria-hidden="true" />
          Upload a picture
        </button>
      </div>
      {items.length ? (
        <div className="grid min-h-0 flex-1 grid-cols-3 gap-1.5 overflow-y-auto overscroll-contain p-1.5">
          {items.map((it) => (
            <button
              key={it.key}
              type="button"
              onClick={() => onPick({ ref: it.ref, url: it.url, name: it.name })}
              aria-label={`Start from ${it.name}`}
              className="group overflow-hidden rounded-[3px] bg-black text-left outline-none focus-visible:ring-1 focus-visible:ring-ivory/70"
            >
              <img src={it.thumb} alt="" loading="lazy" draggable={false} className="aspect-video w-full object-cover transition-opacity group-hover:opacity-85" />
              <span className="block truncate px-1 py-0.5 text-[11px] text-ink2 group-hover:text-ivory">{it.name}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>,
    document.body,
  );
}
