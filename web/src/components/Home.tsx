// Home: a short way in, then the examples. The full prompt (model, aspect, a photo, a clip) lives on Create.
// A one-line bar hands words to Create, or opens Previz. Preset workflows sit under it. The gallery is the page.
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { clearPathError, loadProjects, loadShots, openOnboarding, openPath, resetUpload, setCreateMode, uploadShotFile } from "../actions";
import { cx, isOriginal, MOD_KEY, pillState } from "../lib/format";
import { openFilePicker } from "../lib/filePicker";
import { useApp } from "../store";
import type { Shot } from "../types";
import { useReducedMotion } from "../lib/loops";
import { CreateComposer, DemoClipNotice } from "./CreateBar";
import { Gallery } from "./Gallery";
import { SourceLimits } from "./Prepare";
import { Button, Notice, Slate } from "./ui";

const LEGIBLE = { textShadow: "0 1px 2px rgba(0,0,0,.35), 0 0 36px rgba(0,0,0,.45)" };

export function Home() {
  const over = useWindowFileDrop();
  const seq = useApp((s) => s.createBar.seq);
  const pending = useApp((s) => s.pendingRecipe);
  const failed = useApp((s) => s.shotsFailed);
  const scroller = useRef<HTMLDivElement>(null);

  // A preset, a gallery card or a dropped photo hands the Create bar something: bring it into view.
  useEffect(() => {
    if (seq || pending) scroller.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [seq, pending]);

  return (
    <div ref={scroller} className="h-full overflow-y-auto">
      <SetupNotice />
      <Hero over={over} />
      <div className="mx-auto w-full max-w-[1440px] px-10 pb-28 max-[767px]:px-4">
        {failed ? (
          <Notice className="mt-8" tone="neutral" text="Your projects didn't load." action={{ label: "Retry", onClick: () => void Promise.all([loadProjects(), loadShots()]) }} />
        ) : null}
        <Gallery />
      </div>
    </div>
  );
}

/** A cold start says so before anything else: no ComfyUI. (Missing models: the top bar says "6 of 7 ready".) */
function SetupNotice() {
  const status = useApp((s) => s.status);
  const failed = useApp((s) => s.statusFailed);
  const mode = useApp((s) => s.mode);
  const state = pillState(status, failed);
  if (mode !== "live" || state !== "offline") return null;
  return (
    <div className="mx-auto w-full max-w-[1440px] px-10 pt-6 max-[767px]:px-4">
      <Notice tone="warn" text="Pickups can't reach your ComfyUI." action={{ label: "Connect", onClick: () => openOnboarding("connect") }} />
    </div>
  );
}

/** The whole page takes a dropped file: a clip opens as a shot, a picture goes to the Character mode. */
function useWindowFileDrop() {
  const setupOpen = useApp((s) => s.setup.open);
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => {
      if (!hasFiles(e) || setupOpen) return;
      depth.current += 1;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setOver(false);
    };
    const drop = (e: DragEvent) => {
      depth.current = 0;
      setOver(false);
      if (setupOpen || e.defaultPrevented) return;
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      e.preventDefault();
      if (f.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(f.name)) setCreateMode("character", undefined, f);
      else void uploadShotFile(f);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, [setupOpen]);
  return over;
}

// --- hero -------------------------------------------------------------------------------------------------------

/** The headline and the Create bar, over the hero example's before/after loop (dimmed: the bar is the star). */
function Hero({ over }: { over: boolean }) {
  const examples = useApp((s) => s.examples);
  const hero = examples?.find((e) => e.hero && e.after_url) ?? examples?.find((e) => e.after_url) ?? null;
  const reduced = useReducedMotion();
  return (
    <section aria-label="Create" className="relative isolate overflow-hidden border-b border-line">
      {hero ? (
        <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden="true">
          {/* The after alone (no wipe line: a seam behind the headline reads as a glitch), dimmed, covering the band. */}
          <video
            key={hero.after_url}
            src={hero.after_url}
            poster={hero.poster_url}
            muted
            loop
            playsInline
            autoPlay={!reduced}
            preload="auto"
            className="absolute inset-0 h-full w-full object-cover opacity-[0.42]"
          />
          <div className="absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_45%,rgba(13,12,11,0.35)_0%,rgba(13,12,11,0.85)_60%,#0D0C0B_100%)]" />
          <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-bg to-transparent" />
        </div>
      ) : (
        <div className="absolute inset-0 -z-10" style={{ background: "radial-gradient(90% 70% at 50% 0%, #1D1B18 0%, #0D0C0B 70%)" }} aria-hidden="true" />
      )}
      <div className="mx-auto w-full max-w-[1100px] px-6 pb-7 pt-7 text-center max-[767px]:px-4 max-[767px]:pb-6 max-[767px]:pt-6">
        <h1 className="sr-only">Pickups</h1>
        <CreateComposer placement="hero" />
      </div>
      {over ? <DropOverlay /> : null}
    </section>
  );
}

function DropOverlay() {
  return (
    <div className="pointer-events-none fixed inset-3 z-[60] flex items-center justify-center rounded-card border border-ivory/80 bg-black/40 animate-fade-in">
      <Slate className="rounded-[2px] bg-black/80 px-2 py-1 text-ivory">Drop a clip or a photo</Slate>
    </div>
  );
}

// --- drop / open / path (the project page's Shots tab uses these) ------------------------------------------------

/** `next` sits right after "Drop a shot" (the fresh-install hero's second step). */
export function DropActions({ onMedia, size = "lg", className, next }: { onMedia?: boolean; size?: "lg" | "md"; className?: string; next?: ReactNode }) {
  const upload = useApp((s) => s.upload);
  const path = useApp((s) => s.path);
  const demo = useApp((s) => s.mode === "mock");
  const [value, setValue] = useState("");
  const busy = upload.state === "uploading" || upload.state === "processing";

  if (busy) {
    const pct = Math.round((upload.progress ?? 0) * 100);
    return (
      <div className={cx("w-[min(440px,100%)]", className)} role="status" style={onMedia ? LEGIBLE : undefined}>
        <div className="flex items-baseline justify-between gap-4">
          <span className="slate truncate text-ivory/80">{upload.state === "uploading" ? `Uploading · ${upload.name}` : `Preparing · ${upload.name}`}</span>
          <span className="slate text-ivory/80">{upload.state === "uploading" ? `${pct}%` : ""}</span>
        </div>
        <div className="mt-2.5 h-px w-full bg-ivory/20">
          <div className="h-px bg-ivory transition-[width] duration-300 ease-linear" style={{ width: `${upload.state === "uploading" ? pct : 100}%` }} />
        </div>
      </div>
    );
  }

  if (upload.state === "error" && demo) return <DemoClipNotice name={upload.name} />;

  if (upload.state === "error") {
    return (
      <div className={cx("flex flex-wrap items-center gap-4", className)} role="alert">
        <p className="flex items-baseline gap-3 text-[15px] text-ivory/85" style={onMedia ? LEGIBLE : undefined}>
          <span className="slate">Error</span>
          {upload.error ?? "That file didn't upload."}
        </p>
        <Button
          variant="primary"
          size="md"
          onClick={() => {
            resetUpload();
            openFilePicker();
          }}
        >
          Choose another
        </Button>
      </div>
    );
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;
    if (await openPath(value)) setValue("");
  };

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" size={size === "lg" ? "lg" : "md"} onClick={openFilePicker} className={size === "lg" ? "px-7" : undefined}>
          Open a clip
        </Button>
        <SourceLimits onMedia={onMedia} />
        {next}
        <form
          onSubmit={submit}
          className={cx(
            "flex items-center rounded-full border pl-4 pr-1 transition-colors duration-150 focus-within:border-ivory/60 max-[767px]:w-full",
            size === "lg" ? "h-11" : "h-9",
            onMedia ? "border-ivory/25 bg-black/60" : "border-line2 bg-bg",
          )}
        >
          <input
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              clearPathError();
            }}
            placeholder="or paste a file path"
            aria-label="File path on this machine"
            spellCheck={false}
            autoComplete="off"
            className="mono w-[min(320px,36vw)] bg-transparent text-[13px] text-ivory placeholder:text-ivory/45 focus-visible:outline-none max-[767px]:w-full max-[767px]:min-w-0"
          />
          <Button
            type="submit"
            variant={value.trim() ? "secondary" : "ghost"}
            size="sm"
            loading={path.busy}
            disabled={!value.trim()}
            title={value.trim() ? undefined : "Paste a path first"}
            className={onMedia && !value.trim() ? "disabled:text-ivory/40" : undefined}
          >
            Open…
          </Button>
        </form>
        <span className={cx("slate hidden items-center gap-1 lg:inline-flex", onMedia ? "text-ivory/55" : "text-ink3")}>
          <span className="kbd">{MOD_KEY}</span>
          <span className="kbd">O</span>
        </span>
      </div>
      {path.error ? (
        <p className="mt-3 flex items-baseline gap-3 text-[13px] text-ivory/85" role="alert" style={onMedia ? LEGIBLE : undefined}>
          <span className="slate">Error</span>
          {path.error}
        </p>
      ) : null}
    </div>
  );
}

// --- shots ---------------------------------------------------------------------------------------------------------

/** "ORIGINAL + 2 TAKES" (hidden takes don't count). */
export function stackBadge(shot: Shot): string {
  const n = shot.versions.filter((v) => !isOriginal(v) && !v.hidden).length;
  return n ? `Original + ${n} ${n === 1 ? "take" : "takes"}` : "Original";
}

