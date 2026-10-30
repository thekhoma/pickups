// The home Create bar: the two ways in, at a glance.
// Edit a clip (drop, browse or paste a path; the tasks come next) or start from scratch: an Image from words, or a
// Character sheet (front, back, close-up) from one photo or a few words. The preset chips under it start a workflow in
// one tap; they wait for a clip (open one, or pick a shot) when they need one.
import { Car, ChevronDown, Clapperboard, Eraser, Leaf, Moon, Plus, Snowflake, Sparkles, SunMedium, Upload, UserRoundPlus, X, type LucideIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  buildCharacter,
  characterName,
  chooseRecipe,
  clearPathError,
  ensureProject,
  generateFromHome,
  loadImageModels,
  newProject,
  openExample,
  openPath,
  openPreviz,
  openSetup,
  openShot,
  resetUpload,
  setCurrentProject,
} from "../actions";
import { cx, MOD_KEY } from "../lib/format";
import { openFilePicker } from "../lib/filePicker";
import { PRESETS, recipeById, type Preset } from "../lib/recipes";
import { getState, useApp, type CreateMode } from "../store";
import type { ImageResolution } from "../types";
import { Thumb } from "./MediaCard";
import { askSaveTo } from "./SaveTo";
import { FrameIcon, Picker } from "./Picker";
import { Button, Composer, Notice, Spinner, type ComposerTab } from "./ui";
import { VideoMode } from "./VideoMode";

// Words only: six tabs and Saves-to fit one row of the composer at 1280.
const MODES: ComposerTab<CreateMode>[] = [
  { id: "clip", label: "Edit clip", short: "Edit" },
  { id: "image", label: "New image", short: "Image" },
  { id: "t2v", label: "Text to video", short: "Video" },
  { id: "i2v", label: "Image to video", short: "Animate" },
  { id: "character", label: "New character", short: "Character" },
  { id: "previz", label: "New previz", short: "Previz" },
];

const PRESET_ICON: Record<Preset["icon"], LucideIcon> = {
  snow: Snowflake,
  night: Moon,
  autumn: Leaf,
  remove: Eraser,
  car: Car,
  character: UserRoundPlus,
  previz: Clapperboard,
  relight: SunMedium,
  finish: Sparkles,
};

/** Where new work goes (the current project, never Examples). */
function useTarget() {
  const currentId = useApp((s) => s.currentProjectId);
  const projects = useApp((s) => s.projects);
  const id = currentId && currentId !== "examples" ? currentId : "unsorted";
  return { id, name: projects?.find((p) => p.id === id)?.name ?? "My shots" };
}

/** The composer as every page uses it (ui.tsx Composer, wired to Create): Home in the hero, Create docked at the bottom,
 * Previz docked with "New previz" picked.
 *
 * tab: the tab it opens on (and comes back to when the page sends it a new one); otherwise the last one asked for.
 * previz: the New previz tab's body. Without it, picking New previz opens the Previz page (Home, Create).
 * presets: the Presets / Recipes strip (on by default in the hero). */
export function CreateComposer({
  placement = "hero",
  tab,
  previz,
  compact = false,
  className,
}: {
  placement?: "hero" | "dock";
  tab?: CreateMode;
  previz?: ReactNode;
  /** The collapsed phone dock: only the tab's body (see ui.tsx Composer). */
  compact?: boolean;
  className?: string;
}) {
  const bar = useApp((s) => s.createBar);
  // Nothing to edit yet (no shots anywhere but the examples): it opens on New image until something asks for a tab.
  const noShots = useApp((s) => !!s.projects && s.projects.every((p) => p.id === "examples" || !p.shots));
  const start = (m: CreateMode) => (m === "previz" && !previz ? "image" : m === "clip" && noShots && !bar.seq && !tab ? "image" : m);
  const [mode, setMode] = useState<CreateMode>(start(tab ?? bar.mode));
  useEffect(() => setMode(start(tab ?? bar.mode)), [bar.seq, bar.mode, tab, noShots]);
  const pick = (m: CreateMode) => {
    // Previz has its own page: without a previz body here, the tab opens it.
    if (m === "previz" && !previz) {
      openPreviz();
      return;
    }
    setMode(m);
    // A preset waiting for a clip doesn't apply to an image or a character.
    if (m !== "clip" && getState().pendingRecipe) chooseRecipe(null);
  };

  return (
    <Composer<CreateMode>
      tabs={MODES}
      tab={mode}
      onTab={pick}
      savesTo={<Destination />}
      placement={placement}
      compact={compact}
      className={className}
    >
      <SteadyBody off={compact || mode === "previz"}>
      {mode === "clip" ? (
        <ClipMode />
      ) : mode === "image" ? (
        <ImageMode key={`i${bar.seq}`} text={bar.mode === "image" ? bar.text : undefined} />
      ) : mode === "t2v" || mode === "i2v" ? (
        <VideoMode key={`${mode}${bar.seq}`} mode={mode === "t2v" ? "text" : "image"} text={bar.mode === mode ? bar.text : undefined} />
      ) : mode === "character" ? (
        <CharacterMode key={`c${bar.seq}`} text={bar.mode === "character" ? bar.text : undefined} file={bar.mode === "character" ? bar.file : undefined} />
      ) : (
        previz
      )}
      </SteadyBody>
    </Composer>
  );
}

/** The old name (Create's bottom bar): `studio` was the docked one, without presets. */
export function CreateBar({ studio = false }: { studio?: boolean }) {
  return <CreateComposer placement={studio ? "dock" : "hero"} />;
}

/** The box keeps its height when you switch tabs (Edit clip, New image, Text to video...): it grows to the tallest tab
 * it has shown and never shrinks back, so the tab row stays where your mouse is. A window resize starts it over. */
function SteadyBody({ children, off = false }: { children: ReactNode; off?: boolean }) {
  const inner = useRef<HTMLDivElement>(null);
  const [minH, setMinH] = useState(0);
  useLayoutEffect(() => {
    if (off) return setMinH(0);
    const el = inner.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const phone = window.matchMedia("(max-width: 767px)");
    const grow = () => {
      if (phone.matches) return setMinH(0);
      setMinH((h) => Math.max(h, Math.ceil(el.getBoundingClientRect().height)));
    };
    const reset = () => setMinH(0);
    const ro = new ResizeObserver(grow);
    ro.observe(el);
    window.addEventListener("resize", reset);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", reset);
    };
  }, [off]);
  return (
    <div style={minH ? { minHeight: minH } : undefined} className="flex flex-col">
      <div ref={inner}>{children}</div>
    </div>
  );
}

/** "Saves to · Stunt previz ▾": the project new work goes into. The menu is portaled so a parent can't clip it. */
function Destination() {
  const projects = useApp((s) => s.projects);
  const target = useTarget();
  const [open, setOpen] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const [box, setBox] = useState<{ right: number; top: number; maxH: number; up: boolean } | null>(null);
  const list = (projects ?? []).filter((p) => !p.archived && p.id !== "examples");

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
      setNaming(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setOpen(false);
      setNaming(false);
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const r = btnRef.current?.getBoundingClientRect();
      if (!r) return;
      const above = r.top - 12;
      const below = window.innerHeight - r.bottom - 12;
      const up = above > below;
      setBox({ right: Math.max(8, window.innerWidth - r.right), top: up ? r.top - 8 : r.bottom + 8, maxH: Math.max(160, Math.min(280, up ? above : below)), up });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const n = name.trim();
    if (!n || busy) return;
    setBusy(true);
    setFailed(false);
    const p = await newProject(n, false);
    setBusy(false);
    if (!p) {
      setFailed(true);
      return;
    }
    setCurrentProject(p.id);
    setName("");
    setNaming(false);
    setOpen(false);
  };

  return (
    <div ref={ref} className="relative flex min-w-0 items-center gap-2 pr-1 max-[640px]:w-full max-[640px]:pl-1.5">
      <span className="slate shrink-0 text-ink3">Saves to</span>
      <button
        ref={btnRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Saves to ${target.name}. Change project`}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex h-8 min-w-0 items-center gap-1.5 rounded-full border border-line px-3 text-[13px] text-ink transition-colors duration-150 hover:border-line2 hover:text-ivory"
      >
        <span className="truncate">{target.name}</span>
        <ChevronDown size={13} strokeWidth={1.5} className={cx("shrink-0 text-ink3 transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open && box
        ? createPortal(
            <div
              ref={menuRef}
              role="listbox"
              aria-label="Project"
              className="fixed z-[80] flex w-[280px] animate-rise-in flex-col overflow-hidden rounded-card border border-line2 bg-raised shadow-[0_24px_60px_-24px_rgba(0,0,0,0.9)]"
              style={{ right: box.right, maxHeight: box.maxH, top: box.up ? undefined : box.top, bottom: box.up ? window.innerHeight - box.top : undefined }}
            >
              <div className="shrink-0 border-b border-line bg-raised p-1.5">
                {naming ? (
                  <form onSubmit={create} className="space-y-1.5">
                    <div className="flex items-center gap-1.5">
                    <input
                      autoFocus
                      value={name}
                      onChange={(e) => {
                        setName(e.target.value);
                        setFailed(false);
                      }}
                      placeholder="Project name"
                      aria-label="New project name"
                      className="field h-8 min-w-0 flex-1 px-2.5 text-[13px]"
                    />
                    <Button type="submit" variant="primary" size="sm" className="shrink-0" loading={busy} disabled={!name.trim() || busy}>
                      Create
                    </Button>
                    </div>
                    {failed ? <Notice tone="bad" text="That project wasn't created. Check that Pickups is still running, then try again." stack /> : null}
                  </form>
                ) : (
                  <button
                    type="button"
                    onClick={() => setNaming(true)}
                    className="flex h-8 w-full items-center gap-2 rounded-[4px] px-2 text-left text-[14px] text-ivory hover:bg-line"
                  >
                    <Plus size={14} strokeWidth={1.5} aria-hidden="true" />
                    New project
                  </button>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain py-1">
                {list.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    role="option"
                    aria-selected={p.id === target.id}
                    onClick={() => {
                      setOpen(false);
                      setCurrentProject(p.id, false);
                    }}
                    className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors duration-150 hover:bg-line focus-visible:bg-line"
                  >
                    <Thumb src={p.cover_url ?? undefined} className="h-6 w-10 shrink-0 rounded-[3px]" />
                    <span className={cx("min-w-0 flex-1 truncate text-[14px]", p.id === target.id ? "text-ivory" : "text-ink")}>{p.name}</span>
                    {p.id === target.id ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ivory/80" aria-hidden="true" /> : null}
                  </button>
                ))}
              </div>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}


// --- Edit a clip ------------------------------------------------------------------------------------------------

function ClipMode() {
  const upload = useApp((s) => s.upload);
  const path = useApp((s) => s.path);
  const demo = useApp((s) => s.mode === "mock");
  const pending = recipeById(useApp((s) => s.pendingRecipe));
  const examples = useApp((s) => s.examples);
  const [value, setValue] = useState("");
  const [pathOpen, setPathOpen] = useState(false);
  const [over, setOver] = useState(false);
  const busy = upload.state === "uploading" || upload.state === "processing";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;
    if (await openPath(value)) setValue("");
  };

  const tryFirst = examples?.find((e) => e.can_try !== false) ?? null;
  const pathField = (
    <form onSubmit={submit} className="mt-3 flex items-center gap-2 rounded-full border border-line bg-bg pl-4 pr-1 transition-colors duration-150 focus-within:border-ivory/60">
      <input
        autoFocus
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          clearPathError();
        }}
        placeholder="/path/to/clip.mov"
        aria-label="File path on this machine"
        spellCheck={false}
        autoComplete="off"
        className="mono h-9 min-w-0 flex-1 bg-transparent text-[13px] text-ivory placeholder:text-ink3 focus-visible:outline-none"
      />
      <button
        type="submit"
        disabled={!value.trim() || path.busy}
        className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-full border border-line2 px-3 text-[13px] text-ink transition-colors hover:border-ivory/50 hover:text-ivory disabled:border-transparent disabled:text-ink4"
      >
        {path.busy ? <Spinner size={11} /> : null}
        Open
      </button>
    </form>
  );

  return (
    <div className="p-4 max-[640px]:p-3">
      {pending ? <PendingStrip /> : null}
      {upload.state === "error" ? (
        demo ? (
          <DemoClipNotice name={upload.name} />
        ) : (
          <Notice
            tone="bad"
            text={upload.error ?? "That file didn't upload."}
            action={{
              label: "Choose another",
              onClick: () => {
                resetUpload();
                openFilePicker();
              },
            }}
          />
        )
      ) : busy ? (
        <div className="rounded-card border border-line px-5 py-7" role="status">
          <div className="flex items-baseline justify-between gap-4">
            <span className="slate truncate text-ivory/80">{upload.state === "uploading" ? `Uploading · ${upload.name}` : `Getting it ready · ${upload.name}`}</span>
            <span className="slate text-ivory/80">{upload.state === "uploading" ? `${Math.round((upload.progress ?? 0) * 100)}%` : ""}</span>
          </div>
          <div className="mt-3 h-px w-full bg-ivory/20">
            <div className="h-px bg-ivory transition-[width] duration-300 ease-linear" style={{ width: `${upload.state === "uploading" ? Math.round((upload.progress ?? 0) * 100) : 100}%` }} />
          </div>
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            if (Array.from(e.dataTransfer.types).includes("Files")) setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={() => setOver(false)}
          className={cx("rounded-card border border-dashed px-5 py-4 transition-colors duration-150 max-[640px]:px-4", over ? "border-ivory/70 bg-raised" : "border-line2")}
        >
          <div className="flex items-center gap-4 max-[640px]:flex-col max-[640px]:items-stretch">
            <span className="flex min-w-0 flex-1 items-center gap-4">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-line2 text-ink2" aria-hidden="true">
                <Upload size={17} strokeWidth={1.5} />
              </span>
              <span className="min-w-0">
                <span className="block text-[16px] text-ink">
                  Drop a clip,{" "}
                  <button type="button" onClick={openFilePicker} className="text-ivory underline decoration-ivory/40 underline-offset-4 hover:decoration-ivory">
                    browse
                  </button>
                  , or{" "}
                  <button
                    type="button"
                    aria-expanded={pathOpen}
                    onClick={() => setPathOpen((o) => !o)}
                    className="text-ivory underline decoration-ivory/40 underline-offset-4 hover:decoration-ivory"
                  >
                    paste a path
                  </button>
                </span>
              </span>
            </span>
            {demo && tryFirst ? (
              <Button variant="primary" size="lg" className="shrink-0" onClick={() => void openExample(tryFirst)}>
                Try an example
              </Button>
            ) : (
              <Button variant="primary" size="lg" className="shrink-0" onClick={openFilePicker}>
                Choose a clip
                <span className="hidden items-center gap-1 lg:inline-flex" aria-hidden="true">
                  <span className="kbd border-black/20 text-black/50">{MOD_KEY}</span>
                  <span className="kbd border-black/20 text-black/50">O</span>
                </span>
              </Button>
            )}
          </div>
          {pathOpen ? pathField : null}
        </div>
      )}
      {path.error ? (
        <Notice className="mt-2" tone="bad" text={path.error} />
      ) : null}

      {/* The latest shots (or the examples) to start on, always: an obvious next step besides a new file. */}
      <PickShot />


    </div>
  );
}

function scrollToGallery(e?: { preventDefault: () => void }) {
  e?.preventDefault();
  document.getElementById("make-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Demo mode can't take someone's own file: say so plainly and offer the example shots. Not an error. */
export function DemoClipNotice({ name }: { name?: string }) {
  const examples = useApp((s) => s.examples);
  const first = examples?.find((e) => e.can_try !== false) ?? null;
  return (
    <div className="rounded-card border border-line2 bg-raised/60 px-5 py-5" role="status">
      <span className="slate text-ink3">Demo{name ? ` · ${name}` : ""}</span>
      <p className="mt-2 max-w-[60ch] text-[15px] leading-snug text-ink">The demo can't open your own clips. Run Pickups on a GPU machine to edit them.</p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {first ? (
          <Button
            variant="primary"
            size="md"
            onClick={() => {
              resetUpload();
              void openExample(first);
            }}
          >
            Try an example
          </Button>
        ) : null}
        <button
          type="button"
          onClick={() => {
            resetUpload();
            scrollToGallery();
          }}
          className="inline-flex h-9 items-center rounded-full border border-line2 px-4 text-[14px] text-ink hover:border-ivory/50 hover:text-ivory"
        >
          All examples
        </button>
        <button type="button" onClick={resetUpload} className="inline-flex h-9 items-center rounded-full px-3 text-[14px] text-ink3 hover:text-ivory">
          Close
        </button>
      </div>
    </div>
  );
}

/** "SNOW · Open a clip or pick a shot to start ✕" while a preset or recipe waits for a clip. */
function PendingStrip() {
  const recipe = recipeById(useApp((s) => s.pendingRecipe));
  if (!recipe) return null;
  const preset = PRESETS.find((p) => p.id === recipe.id);
  const I = preset ? PRESET_ICON[preset.icon] : Sparkles;
  return (
    <div className="mb-3 flex items-center gap-3 rounded-card border border-ivory/40 bg-raised py-2 pl-3 pr-1.5 animate-fade-in" role="status">
      <I size={15} strokeWidth={1.5} className="shrink-0 text-ivory" aria-hidden="true" />
      <span className="min-w-0 flex-1 text-[14px] text-ink">
        <span className="text-ivory">{recipe.title}</span>
        <span className="text-ink3"> · Choose a clip or pick a shot.</span>
      </span>
      <button type="button" onClick={() => chooseRecipe(null)} aria-label="Cancel" className="rounded-full p-1.5 text-ink3 transition-colors hover:text-ivory">
        <X size={14} />
      </button>
    </div>
  );
}

/** While a preset waits: the current project's latest shots (or, with none, the example shots) to start on. */
function PickShot() {
  const target = useTarget();
  const order = useApp((s) => s.projectShots[target.id] ?? null);
  const shots = useApp((s) => s.shots);
  const examples = useApp((s) => s.examples);
  useEffect(() => {
    if (!order) void ensureProject(target.id);
  }, [order, target.id]);
  const list = (order ?? []).map((id) => shots[id]).filter(Boolean).slice(0, 5);
  const exs = (examples ?? []).filter((e) => e.can_try !== false).slice(0, 5);
  if (!list.length && !exs.length) return null;
  return (
    <div className="mt-3 flex items-center gap-3 rounded-card border border-line px-3 py-2.5 max-[640px]:flex-col max-[640px]:items-stretch">
      <span className="slate shrink-0 text-ink3">Or start on a shot</span>
      <div className="flex min-w-0 flex-1 gap-2 overflow-x-auto">
        {list.length
          ? list.map((s) => (
              <button key={s.id} type="button" onClick={() => openShot(s.id, { applyPreset: true })} aria-label={`Start on ${s.name}`} className="group w-[112px] shrink-0 text-left">
                <Thumb src={s.latest_thumb_url || s.thumb_url} className="aspect-video w-full rounded-[4px] transition-opacity group-hover:opacity-85" />
                <span className="mt-1 block truncate font-mono text-[10px] text-ink2 group-hover:text-ivory">{s.name}</span>
              </button>
            ))
          : exs.map((e) => (
              <button key={e.id} type="button" onClick={() => void openExample(e)} aria-label={`Start on the example: ${e.title}`} className="group w-[112px] shrink-0 text-left">
                <Thumb src={e.poster_url} className="aspect-video w-full rounded-[4px] transition-opacity group-hover:opacity-85" />
                <span className="mt-1 block truncate text-[11px] text-ink2 group-hover:text-ivory">{e.title}</span>
              </button>
            ))}
      </div>
    </div>
  );
}

// --- Image ------------------------------------------------------------------------------------------------------

const ASPECTS = ["16:9", "1:1", "9:16", "4:5", "21:9"];

function ImageMode({ text }: { text?: string }) {
  const catalog = useApp((s) => s.imageModels);
  const booted = useApp((s) => s.mode);
  const [prompt, setPrompt] = useState(text ?? "");
  const [aspect, setAspect] = useState("16:9");
  const [model, setModel] = useState("");
  const [res, setRes] = useState<ImageResolution>("1K");
  const [busy, setBusy] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    void loadImageModels();
    if (text) area.current?.focus();
  }, [text, booted]);
  const models = (catalog?.models ?? []).filter((m) => m.modes.includes("generate"));
  const chosen = models.find((m) => m.id === model) ?? models.find((m) => m.id === catalog?.default) ?? models[0];
  const has2k = !!chosen && chosen.max_side >= 2048;
  const needsSetup = chosen?.ready === false;
  const aspects = catalog?.aspects?.length ? ASPECTS.filter((a) => catalog.aspects.includes(a)) : ASPECTS;
  const go = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    await generateFromHome({ prompt: prompt.trim(), aspect, model: chosen?.id, resolution: has2k ? res : "1K" });
    setBusy(false);
  };
  return (
    <form
      className="p-4 max-[640px]:p-3"
      onSubmit={(e) => {
        e.preventDefault();
        void go();
      }}
    >
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
        placeholder="A stone castle on a loch at blue hour, soft mist"
        aria-label="Describe the image"
        className="field h-auto min-h-[96px] resize-none py-3 text-[16px] leading-relaxed"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Picker
          label="Model"
          value={chosen?.id ?? ""}
          options={
            models.length
              ? models.map((m) => ({ id: m.id, label: m.name, sub: m.ready === false ? "Needs setup" : m.speed.split(" at ")[0] }))
              : [{ id: "", label: catalog ? "No model" : "Loading" }]
          }
          onChange={setModel}
        />
        <Picker
          label="Aspect"
          value={aspect}
          options={aspects.map((a) => ({ id: a, label: a, left: <FrameIcon aspect={a} /> }))}
          onChange={setAspect}
        />
        <Picker
          label="Size"
          value={has2k ? res : "1K"}
          options={[
            { id: "1K", label: "1K" },
            { id: "2K", label: "2K", sub: has2k ? undefined : "Not on this model", disabled: !has2k },
          ]}
          onChange={(v) => setRes(v === "2K" ? "2K" : "1K")}
        />

        {needsSetup ? (
          <span className="ml-auto flex items-center gap-2 max-[640px]:w-full">
            <span className="slate text-ink3">Model needs setup</span>
            <Button variant="quiet" size="sm" onClick={() => openSetup()}>
              Open setup
            </Button>
          </span>
        ) : null}
        <Button
          type="submit"
          variant="primary"
          size="lg"
          loading={busy}
          disabled={!prompt.trim() || busy || needsSetup}
          className={cx(!needsSetup && "ml-auto", "max-[640px]:w-full")}
        >
          Generate
        </Button>
      </div>
    </form>
  );
}

// --- Character --------------------------------------------------------------------------------------------------

const isImage = (f: File) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|tiff?)$/i.test(f.name);

function CharacterMode({ text, file: handed }: { text?: string; file?: File | null }) {
  const target = useTarget();
  const cast = useApp((s) => s.cast[target.id]);
  const [words, setWords] = useState(text ?? "");
  const [file, setFile] = useState<File | null>(handed ?? null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void ensureProject(target.id);
  }, [target.id]);
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!file) area.current?.focus();
  }, [file]);

  const taken = useMemo(() => (cast ?? []).map((c) => c.name), [cast]);
  const fromFile = file ? file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim() : "";
  const suggested = characterName(file ? (/^(img|dsc|photo|image|pxl)\s*\d/i.test(fromFile) ? "" : fromFile) : words, taken);
  // Named silently (renamed later in the cast): "Old stunt driver", never one the cast has.
  const finalName = suggested;
  const ready = (!!file || !!words.trim()) && !!finalName.trim() && !busy;

  const take = (f: File | null | undefined) => {
    if (!f) return;
    if (!isImage(f)) {
      setError("That's not a picture. Use a JPG, PNG or WebP.");
      return;
    }
    setError(null);
    setFile(f);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    if (!(await askSaveTo("Save the character to", "Make character"))) return;
    setBusy(true);
    setError(null);
    const err = await buildCharacter({ file, describe: words, name: finalName });
    setBusy(false);
    if (err) setError(err);
  };

  return (
    <form onSubmit={submit} className="p-4 max-[640px]:p-3">
      <input ref={input} type="file" accept="image/*" className="hidden" tabIndex={-1} aria-hidden="true" onChange={(e) => (take(e.target.files?.[0]), (e.target.value = ""))} />
      <div
        className={cx("flex gap-3 rounded-card border p-2.5 transition-colors duration-150", over ? "border-ivory/70 bg-raised" : "border-line bg-bg focus-within:border-ivory/60")}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOver(false);
          take(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="flex min-w-0 flex-1 flex-col">
          {file && preview ? (
            <span className="flex items-center gap-3 px-1 py-1">
              <span className="relative h-[72px] w-[72px] shrink-0 overflow-hidden rounded-[4px] bg-black">
                <img src={preview} alt="Your photo" className="h-full w-full object-cover" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[15px] text-ink">{file.name}</span>

              </span>
              <button type="button" onClick={() => setFile(null)} aria-label="Remove the photo" className="rounded-full p-1.5 text-ink3 hover:text-ivory">
                <X size={14} />
              </button>
            </span>
          ) : (
            <textarea
              ref={area}
              rows={2}
              value={words}
              onChange={(e) => setWords(e.target.value)}
              placeholder="An old stunt driver in a cream helmet, or drop a photo"
              aria-label="Drop a photo or describe them"
              className="min-h-[60px] min-w-0 flex-1 resize-none bg-transparent px-1 py-1 text-[16px] leading-relaxed text-ivory placeholder:text-ink3 focus-visible:outline-none"
            />
          )}
          {file ? null : (
            <button
              type="button"
              onClick={() => input.current?.click()}
              className="mt-1 inline-flex h-7 w-fit items-center gap-1.5 rounded-full px-1.5 text-[12px] text-ink3 transition-colors hover:text-ivory"
            >
              <Plus size={13} strokeWidth={1.5} aria-hidden="true" />
              Add a photo
            </button>
          )}
        </div>
        <SheetPreview />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-3">
        <Button type="submit" variant="primary" size="lg" loading={busy} disabled={!ready} className="ml-auto max-[640px]:w-full">
          New character
        </Button>
      </div>
      {error ? (
        <Notice className="mt-3" tone="bad" text={error} />
      ) : null}
    </form>
  );
}

/** An example sheet (the stunt driver the placeholder describes): front, back, close-up. */
function SheetPreview() {
  return (
    <span className="hidden shrink-0 flex-col gap-1 sm:flex" aria-hidden="true">
      <span className="slate text-[9px] leading-[12px] text-ink4">Example</span>
      <span className="flex gap-[2px] overflow-hidden rounded-[3px] shadow-[0_0_0_1px_rgba(238,233,224,0.12)]">
        {["front", "back", "face"].map((v) => (
          <img key={v} src={`/mock/cast_driver_${v}.jpg`} alt="" className="h-[58px] w-[38px] object-cover object-top" loading="lazy" draggable={false} />
        ))}
      </span>
      <span className="flex justify-between px-0.5 font-mono text-[9px] uppercase tracking-[0.08em] text-ink3">
        <span>Front</span>
        <span>Back</span>
        <span>Close</span>
      </span>
    </span>
  );
}

