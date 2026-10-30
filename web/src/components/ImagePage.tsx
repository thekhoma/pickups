// Image: pick a model, an aspect, a size, write a prompt, generate. Images live in the current project, so a still
// can become cast, a look frame or a first frame in one click (docs/images-spec.md).
import {
  Clapperboard,
  CloudSnow,
  Download,
  ImagePlus,
  PenLine,
  Plus,
  RotateCcw,
  UserPlus,
  X,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { generateImages, getApi, imageToCast, loadImageModels, loadImages, navigate, openSetup, setCreateMode, useImageIn, type ImageUse } from "../actions";
import { cx, isActive, relativeTime } from "../lib/format";
import { madeIn } from "../lib/runtime";
import { isLocked, nextSeed, seedFrom } from "../lib/seed";
import { getState, setState, useApp } from "../store";
import type { CastKind, ImageModel, ImageResolution, ProjectImage, Ref, Shot } from "../types";
import { useEnhance } from "./Enhance";
import { MentionField, PromptReceipt, useCast } from "./Mentions";
import { Fold } from "./FormKit";
import { Thumb } from "./MediaCard";
import { Cancelled, RunStrip } from "./RunStatus";
import { Button, Hint, Modal, Notice, SeedLock, Segmented, Skeleton, Slate, Spinner, Tag, Tip, useDismiss } from "./ui";

// --- form state (kept while you move around the app) ---------------------------------------------------------

interface PickedRef {
  ref: Ref;
  url: string;
  busy?: boolean;
}

interface Form {
  model: string;
  aspect: string;
  res: ImageResolution;
  prompt: string;
  refs: PickedRef[];
  count: number;
  /** The seed the last run used (or a result's, from Same settings); `locked` sends it again (lib/seed.ts). */
  seed: string;
  locked?: boolean;
}

let lastForm: Form | null = null;

const PLACEHOLDER: Record<string, string> = {
  flux: "e.g. a woman in a red wool coat on a rooftop at dusk",
  qwen: "A poster that says “Open late” in warm neon above a rainy street",
  krea: "Golden-hour portrait in a field of tall grass, soft film look",
  edit: "Change the jacket to red leather",
};

function projectOf(currentId: string): string {
  return currentId && currentId !== "examples" ? currentId : "unsorted";
}

function ModelLine({ m, on }: { m: ImageModel; on?: boolean }) {
  return (
    <>
      <span className="flex items-baseline justify-between gap-3">
        <span className={cx("truncate text-[14px]", on ? "text-ivory" : "text-ink")}>{m.name}</span>
        <span className="mono shrink-0 text-[11px] text-ink3">{m.speed.split(" at ")[0]}</span>
      </span>
      <span className="mt-1 flex flex-wrap items-center gap-1.5">
        <span className="text-[12px] text-ink3">{m.maker}</span>
        {m.modes.includes("edit") ? <Tag>Edit</Tag> : null}
        {m.commercial === false ? <Tag>Non-commercial</Tag> : null}
      </span>
    </>
  );
}

function FrameIcon({ aspect }: { aspect: string }) {
  const [w, h] = aspect.split(":").map(Number);
  const k = 12 / Math.max(w, h);
  return <span aria-hidden="true" className="inline-block shrink-0 rounded-[1px] border border-current" style={{ width: Math.max(4, w * k), height: Math.max(4, h * k) }} />;
}

// --- the page ---------------------------------------------------------------------------------------------------

export function ImagePage() {
  const currentId = useApp((s) => s.currentProjectId);
  const pid = projectOf(currentId);
  const project = useApp((s) => s.projects?.find((p) => p.id === pid));
  const catalog = useApp((s) => s.imageModels);
  const failed = useApp((s) => s.imageModelsFailed);
  const draft = useApp((s) => s.imageDraft);
  const cast = useCast(pid);

  const [f, setF] = useState<Form>(() => lastForm ?? { model: "", aspect: "16:9", res: "1K", prompt: "", refs: [], count: 1, seed: "" });
  const [note, setNote] = useState<string | null>(null);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [lastJob, setLastJob] = useState<string | null>(null);
  const promptBox = useRef<HTMLDivElement>(null);
  const modelBox = useRef<HTMLElement>(null);
  const changeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    lastForm = f;
  }, [f]);

  // The model list closes on a click outside it, and on Esc when nothing else (Setup, the queue, a menu) took the key.
  useEffect(() => {
    if (!modelsOpen) return;
    const onDown = (e: PointerEvent) => {
      if (modelBox.current && !modelBox.current.contains(e.target as Node)) setModelsOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      e.preventDefault();
      const hadFocus = !!modelBox.current?.contains(document.activeElement);
      setModelsOpen(false);
      if (hadFocus) changeBtn.current?.focus();
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [modelsOpen]);
  useEffect(() => {
    void loadImageModels();
  }, []);
  useEffect(() => {
    if (catalog && !f.model) setF((x) => ({ ...x, model: catalog.default }));
  }, [catalog, f.model]);

  const models = catalog?.models ?? [];
  const model: ImageModel | undefined = models.find((m) => m.id === f.model) ?? models.find((m) => m.id === catalog?.default) ?? models[0];
  const canEdit = !!model?.modes.includes("edit");
  const editMode = f.refs.length > 0 && canEdit;
  const size = model?.sizes[f.res]?.[f.aspect];
  const has2k = !!model && model.max_side >= 2048 && !!model.sizes["2K"];

  // Settings handed over from a result ("Edit", "Same settings"), here or on the project page.
  useEffect(() => {
    if (!draft) return;
    // A result's seed comes locked ("Same settings" makes the same picture again).
    const patch = draft.patch as Partial<Form>;
    setF((x) => ({ ...x, ...patch, ...("seed" in patch ? { locked: seedFrom(patch.seed, true).locked } : {}) }));
    setState({ imageDraft: null });
    promptBox.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [draft]);

  // References switch to a model that can edit (with a one-line note).
  const setRefs = (refs: PickedRef[]) => {
    setF((x) => {
      let modelId = x.model;
      const cur = models.find((m) => m.id === x.model);
      if (refs.length && cur && !cur.modes.includes("edit")) {
        const alt = models.find((m) => m.modes.includes("edit") && m.ready !== false);
        if (alt) {
          modelId = alt.id;
          setNote(`Switched to ${alt.name}: it can edit with references.`);
        }
      }
      return { ...x, refs, model: modelId };
    });
  };

  const valid = !!model && model.ready !== false && (!!f.prompt.trim() || f.refs.length > 0) && !f.refs.some((r) => r.busy);
  const generate = async () => {
    if (!valid || !model || submitting) return;
    setSubmitting(true);
    const locked = isLocked({ ...seedFrom(f.seed, true), locked: !!f.locked });
    const seed = nextSeed({ ...seedFrom(f.seed, locked), locked });
    const job = await generateImages(pid, {
      model: model.id,
      prompt: f.prompt.trim(),
      aspect: f.aspect,
      resolution: f.res,
      count: f.count,
      seed,
      seed_locked: locked,
      refs: f.refs.map((r) => r.ref),
    });
    setSubmitting(false);
    if (job) {
      setLastJob(job.id);
      const used = (job.params as { seed?: number } | undefined)?.seed ?? seed;
      setF((x) => ({ ...x, seed: String(used) }));
    }
  };
  const genRef = useRef(generate);
  genRef.current = generate;

  // A refused run (Ideogram's grey card) saved nothing: its seed isn't one to keep, so the lock lets go of it.
  const refused = useApp((s) => (lastJob && s.jobs[lastJob]?.status === "failed" && s.jobs[lastJob]?.blocked ? lastJob : null));
  useEffect(() => {
    if (refused) setF((x) => ({ ...x, seed: "", locked: false }));
  }, [refused]);

  // ⌘↵ / Ctrl+↵ generates.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        void genRef.current();
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const target = editMode ? "qwen-edit" : model?.id.startsWith("qwen") ? "qwen-image" : "flux3";
  const enh = useEnhance({
    request: f.prompt.trim() ? { target, project_id: pid, ...(editMode ? { refs: f.refs.map((r) => r.ref) } : {}) } : { target, project_id: pid },
    text: f.prompt,
    onUse: (v) => setF((x) => ({ ...x, prompt: v })),
  });

  const placeholder = editMode ? PLACEHOLDER.edit : model?.id.startsWith("qwen") ? PLACEHOLDER.qwen : model?.id.startsWith("krea") ? PLACEHOLDER.krea : PLACEHOLDER.flux;

  return (
    <div className="flex h-full min-h-0">
      <aside className="flex w-[380px] shrink-0 flex-col border-r border-line bg-panel max-[1180px]:w-[340px]" aria-label="Make an image">
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-6 py-5">
          <div className="flex items-baseline justify-between gap-3">
            <h1 className="display text-[36px]">Image</h1>
            <span className="flex min-w-0 items-center gap-2">
              <button type="button" onClick={() => setCreateMode("character")} className="shrink-0 text-[13px] text-ink3 hover:text-ivory">
                Character sheet
              </button>
              <Hint text="A front, a back, and a close-up, from a photo or a few words." />
              <a href={`#/project/${encodeURIComponent(pid)}/images`} className="truncate text-[13px] text-ink3 hover:text-ivory">
                Saved to {project?.name ?? "My shots"} →
              </a>
            </span>
          </div>

          {/* Prompt first */}
          <section ref={promptBox}>
            <div className="slate mb-1.5 text-ink3">Prompt</div>
            <MentionField label="Prompt" value={f.prompt} onChange={(prompt) => setF((x) => ({ ...x, prompt }))} rows={4} cast={cast} projectId={pid} placeholder={placeholder} />
            {/* Enhance and what each model gets: for pros, under one closed fold. */}
            <div className="mt-2">
              <Fold label="Advanced">
                <div className="space-y-2 pt-2">
                  {enh.button ? <div className="flex">{enh.button}</div> : null}
                  {enh.panel}
                  <PromptReceipt
                    projectId={pid}
                    text={f.prompt}
                    uploads={editMode ? f.refs.length : 0}
                    models={editMode ? ["qwen"] : ["gen"]}
                    genLabel={model?.name}
                  />
                </div>
              </Fold>
            </div>
          </section>

          {/* References */}
          <References pid={pid} refs={f.refs} max={model?.max_refs ?? 4} onChange={setRefs} editable={canEdit} />

          {/* Model: the selected one, "Change" opens the list */}
          <section ref={modelBox}>
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <span className="slate text-ink3">Model</span>
              {catalog && models.length > 1 ? (
                <button
                  ref={changeBtn}
                  type="button"
                  onClick={() => setModelsOpen((o) => !o)}
                  aria-expanded={modelsOpen}
                  className="text-[13px] text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline"
                >
                  {modelsOpen ? "Done" : "Change"}
                </button>
              ) : null}
            </div>
            {failed && !catalog ? (
              <Notice stack tone="bad" text="The image models didn't load." action={{ label: "Try again", onClick: () => void loadImageModels(true) }} />
            ) : !catalog ? (
              <Skeleton className="h-[58px]" />
            ) : modelsOpen ? (
              <div className="space-y-1.5 animate-fade-in" role="radiogroup" aria-label="Model">
                {models.map((m) => {
                  const on = m.id === model?.id;
                  const off = m.ready === false;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      aria-label={`${m.name}, ${m.maker}${off ? ", not installed" : `, ${m.speed.split(" at ")[0]}`}${m.commercial === false ? ", non-commercial license" : ""}`}
                      title={m.licence ? `License: ${m.licence}` : undefined}
                      onClick={() => {
                        if (off) {
                          openSetup();
                          return;
                        }
                        setF((x) => ({ ...x, model: m.id, res: m.max_side >= 2048 ? x.res : "1K" }));
                        setModelsOpen(false);
                      }}
                      className={cx(
                        "block w-full rounded-card border px-3.5 py-2.5 text-left transition-colors duration-150",
                        on ? "border-ivory/50 bg-raised" : "border-line hover:border-line2",
                        off && "opacity-60",
                      )}
                    >
                      <ModelLine m={m} on={on} />
                      <span className="mt-1 block text-[12px] leading-snug text-ink3">{off ? "Not installed · Setup" : m.notes}</span>
                      {m.licence && !off ? <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-[0.06em] text-ink4">{m.licence}</span> : null}
                    </button>
                  );
                })}
              </div>
            ) : model ? (
              <button
                type="button"
                onClick={() => setModelsOpen(true)}
                aria-label={`Model: ${model.name}, ${model.maker}${model.commercial === false ? ", non-commercial license" : ""}. Change`}
                title={model.licence ? `License: ${model.licence}` : undefined}
                className="block w-full rounded-card border border-ivory/50 bg-raised px-3.5 py-2.5 text-left transition-colors duration-150 hover:border-ivory/70"
              >
                <ModelLine m={model} on />
              </button>
            ) : null}
            {note ? (
              <p className="mt-2 flex items-start gap-2 text-[12px] text-ink2">
                <span className="flex-1">{note}</span>
                <button type="button" aria-label="Dismiss" onClick={() => setNote(null)} className="text-ink3 hover:text-ivory">
                  <X size={12} />
                </button>
              </p>
            ) : null}
          </section>

          {/* Aspect */}
          <section>
            <div className="slate mb-2 text-ink3">Aspect</div>
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Aspect">
              {(catalog?.aspects ?? ["1:1", "4:5", "3:4", "2:3", "9:16", "16:9", "3:2", "4:3", "21:9"]).map((a) => {
                const on = a === f.aspect;
                return (
                  <button
                    key={a}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    onClick={() => setF((x) => ({ ...x, aspect: a }))}
                    className={cx(
                      "mono inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-[11px] transition-colors duration-150",
                      on ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
                    )}
                  >
                    <FrameIcon aspect={a} />
                    {a}
                  </button>
                );
              })}
            </div>
          </section>

          {/* Size and count, side by side */}
          <section className="grid grid-cols-[1fr_auto] items-end gap-5">
            <div className="min-w-0">
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <span className="slate text-ink3">Size</span>
                <span className="mono truncate text-[12px] text-ink">{size ? `${size[0]} × ${size[1]}` : "—"}</span>
              </div>
              <Segmented<ImageResolution>
                block
                size="sm"
                label="Resolution"
                value={f.res}
                onChange={(res) => setF((x) => ({ ...x, res }))}
                options={[
                  { value: "1K", label: "1K" },
                  { value: "2K", label: "2K", disabled: !has2k },
                ]}
              />
            </div>
            <div>
              <div className="slate mb-2 text-ink3">Count</div>
              <Segmented<string>
                size="sm"
                label="Count"
                value={String(f.count)}
                onChange={(v) => setF((x) => ({ ...x, count: Number(v) }))}
                options={[
                  { value: "1", label: "1" },
                  { value: "2", label: "2" },
                  { value: "4", label: "4" },
                ]}
              />
            </div>
          </section>
          <div className="-mt-3 flex justify-end">
            <SeedLock seed={seedFrom(f.seed, false).seed} locked={!!f.locked} onChange={(locked) => setF((x) => ({ ...x, locked }))} />
          </div>
        </div>

        <div className="shrink-0 space-y-2 border-t border-line px-6 py-5">
          <Button variant="primary" size="lg" block disabled={!valid} loading={submitting} onClick={() => void generate()}>
            {editMode ? "Edit" : "Generate"}
          </Button>
          <p className="slate text-center text-ink4">
            <span className="kbd">⌘</span> <span className="kbd">↵</span>
          </p>
        </div>
      </aside>

      <section className="min-w-0 flex-1 overflow-y-auto px-8 py-6" aria-label="Results">
        <Results
          pid={pid}
          lastJob={lastJob}
          onRetry={() => void generate()}
          onEdit={(im) => setRefs([{ ref: { kind: "image", id: im.id }, url: im.thumb_url || im.url }])}
          onSame={(im) =>
            setF((x) => ({
              ...x,
              model: im.model || x.model,
              aspect: im.aspect || x.aspect,
              res: im.resolution || x.res,
              prompt: im.prompt ?? x.prompt,
              seed: im.seed != null ? String(im.seed) : "",
              locked: im.seed != null,
            }))
          }
        />
      </section>
    </div>
  );
}

// --- references -------------------------------------------------------------------------------------------------

function References({ pid, refs, max, onChange, editable }: { pid: string; refs: PickedRef[]; max: number; onChange: (r: PickedRef[]) => void; editable: boolean }) {
  const [open, setOpen] = useState(refs.length > 0);
  const [picker, setPicker] = useState<"cast" | "images" | null>(null);
  const cast = useCast(pid);
  const images = useApp((s) => s.images[pid] ?? null);
  const input = useRef<HTMLInputElement>(null);
  const pickRef = useRef<HTMLDivElement>(null);
  useDismiss(!!picker, () => setPicker(null), pickRef);
  useEffect(() => {
    if (refs.length) setOpen(true);
  }, [refs.length]);
  const room = Math.max(0, max - refs.length);

  const upload = (files: FileList | null) => {
    const list = Array.from(files ?? []).filter((f) => f.type.startsWith("image/")).slice(0, room);
    let next = [...refs];
    for (const file of list) {
      const tmp = { ref: { kind: "upload" as const, id: `pending-${Math.random()}` }, url: URL.createObjectURL(file), busy: true };
      next = [...next, tmp];
      getApi()
        .upload(file)
        .then((res) => onChangeLatest((cur) => cur.map((r) => (r === tmp ? { ref: { kind: "upload", id: res.id }, url: res.url } : r))))
        .catch(() => onChangeLatest((cur) => cur.filter((r) => r !== tmp)));
    }
    onChange(next);
  };
  // Uploads finish later; apply them to the latest list.
  const latest = useRef(refs);
  latest.current = refs;
  const onChangeLatest = (fn: (cur: PickedRef[]) => PickedRef[]) => onChange(fn(latest.current));

  const add = (r: PickedRef) => {
    if (refs.some((x) => x.ref.kind === r.ref.kind && x.ref.id === r.ref.id) || !room) return;
    onChange([...refs, r]);
    setPicker(null);
  };

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} className="inline-flex items-center gap-1.5 text-[13px] text-ink2 hover:text-ivory">
        <Plus size={14} strokeWidth={1.5} /> Add reference
      </button>
    );
  }
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <span className="slate text-ink3">References</span>
          <Hint text={editable ? "The first reference is the canvas (image 1). Refer to the others by number." : "This model borrows a style from references."} />
        </span>
        {!refs.length ? (
          <button type="button" onClick={() => setOpen(false)} className="text-[12px] text-ink3 hover:text-ivory">
            Close
          </button>
        ) : null}
      </div>
      <input ref={input} type="file" accept="image/*" multiple className="hidden" onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
      {refs.length ? (
        <div className="mb-3 grid grid-cols-3 gap-2">
          {refs.map((r, i) => (
            <div key={`${r.ref.kind}:${r.ref.id}`} className="relative aspect-square overflow-hidden rounded-[4px] bg-black">
              <img src={r.url} alt="" className="h-full w-full object-cover" />
              <Slate at="tl" className="!left-1.5 !top-1.5">{i === 0 ? "Image 1 · canvas" : `Image ${i + 1}`}</Slate>
              {r.busy ? (
                <span className="absolute inset-0 flex items-center justify-center bg-black/50 text-ivory/80">
                  <Spinner size={12} />
                </span>
              ) : null}
              <button
                type="button"
                aria-label="Remove reference"
                onClick={() => onChange(refs.filter((x) => x !== r))}
                className="absolute right-1 top-1 rounded-full bg-black/70 p-0.5 text-ivory/70 hover:text-ivory"
              >
                <X size={11} />
              </button>
            </div>
          ))}
        </div>
      ) : null}
      <div ref={pickRef} className="relative flex flex-wrap gap-1.5">
        <button type="button" disabled={!room} onClick={() => input.current?.click()} className="inline-flex h-7 items-center gap-1.5 rounded-full border border-line px-3 text-[12px] text-ink2 hover:border-line2 hover:text-ivory disabled:text-ink4">
          <ImagePlus size={13} strokeWidth={1.5} /> Upload
        </button>
        <button type="button" disabled={!room || !cast.length} onClick={() => setPicker(picker === "cast" ? null : "cast")} className="inline-flex h-7 items-center rounded-full border border-line px-3 text-[12px] text-ink2 hover:border-line2 hover:text-ivory disabled:text-ink4">
          From cast
        </button>
        <button type="button" disabled={!room || !images?.length} onClick={() => setPicker(picker === "images" ? null : "images")} className="inline-flex h-7 items-center rounded-full border border-line px-3 text-[12px] text-ink2 hover:border-line2 hover:text-ivory disabled:text-ink4">
          From images
        </button>
        {picker ? (
          <div className="absolute left-0 top-full z-40 mt-2 max-h-[300px] w-full overflow-y-auto rounded-card border border-line2 bg-raised p-2 animate-rise-in">
            <div className="grid grid-cols-3 gap-1.5">
              {picker === "cast"
                ? cast.map((c) => (
                    <button key={c.id} type="button" onClick={() => add({ ref: { kind: "cast", id: c.id }, url: c.thumb_url || c.url })} className="group text-left">
                      <img src={c.thumb_url || c.url} alt="" className="aspect-square w-full rounded-[3px] object-cover group-hover:opacity-80" />
                      <span className="mt-1 block truncate text-[11px] text-ink2">{c.name}</span>
                    </button>
                  ))
                : (images ?? []).map((im) => (
                    <button key={im.id} type="button" onClick={() => add({ ref: { kind: "image", id: im.id }, url: im.thumb_url || im.url })} className="group">
                      <img src={im.thumb_url || im.url} alt="" className="aspect-square w-full rounded-[3px] object-cover group-hover:opacity-80" />
                    </button>
                  ))}
            </div>
          </div>
        ) : null}
      </div>

    </section>
  );
}

// --- results ------------------------------------------------------------------------------------------------------

function groupByBatch(list: ProjectImage[]): ProjectImage[][] {
  const out: ProjectImage[][] = [];
  for (const im of list) {
    const last = out[out.length - 1];
    if (last && (im.job_id ? last[0].job_id === im.job_id : false)) last.push(im);
    else out.push([im]);
  }
  return out;
}

export function Results({ pid, lastJob, onEdit, onSame, onRetry, compact }: { pid: string; lastJob?: string | null; onEdit?: (im: ProjectImage) => void; onSame?: (im: ProjectImage) => void; onRetry?: () => void; compact?: boolean }) {
  const images = useApp((s) => s.images[pid] ?? null);
  const jobs = useApp((s) => s.jobs);
  const [detail, setDetail] = useState<{ im: ProjectImage; castOpen?: boolean } | null>(null);
  const [use, setUse] = useState<{ im: ProjectImage; kind: "look_frame" | "first_frame" | "reference" } | null>(null);

  useEffect(() => {
    void loadImages(pid);
  }, [pid]);

  const running = useMemo(
    () => Object.values(jobs).filter((j) => j.task === "image" && (j.project_id ?? pid) === pid && isActive(j)),
    [jobs, pid],
  );
  const batches = useMemo(() => groupByBatch(images ?? []), [images]);
  const [first, ...rest] = batches;
  const lastFailed = lastJob ? jobs[lastJob]?.status === "failed" ? jobs[lastJob] : null : null;
  const lastCanceled = !!lastJob && jobs[lastJob]?.status === "canceled";

  const actions = (im: ProjectImage) => ({
    download: () => {
      const a = document.createElement("a");
      a.href = im.url;
      a.download = `${im.id}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    },
    cast: () => setDetail({ im, castOpen: true }),
    edit: () => (onEdit ? onEdit(im) : handToImagePage({ refs: [{ ref: { kind: "image", id: im.id }, url: im.thumb_url || im.url }] })),
    look: () => setUse({ im, kind: "look_frame" }),
    first: () => setUse({ im, kind: "first_frame" }),
    ref: () => setUse({ im, kind: "reference" }),
    same: () =>
      onSame
        ? onSame(im)
        : handToImagePage({ model: im.model, aspect: im.aspect, res: im.resolution, prompt: im.prompt, seed: im.seed != null ? String(im.seed) : "" }),
  });

  if (images === null) {
    return (
      <div className="grid grid-cols-2 gap-4">
        <Skeleton className="aspect-video" />
        <Skeleton className="aspect-video" />
      </div>
    );
  }

  return (
    <div>
      {lastFailed ? (
        <Notice className="mb-6" stack tone="bad" text={lastFailed.error || "Something went wrong while making the images."} />
      ) : null}
      {lastCanceled && !running.length ? <Cancelled className="mb-6" onRetry={onRetry} /> : null}
      {running.length ? (
        <div className="mb-8 grid grid-cols-2 gap-4 max-[1100px]:grid-cols-1">
          {running.map((j) => (
            <div key={j.id} className="relative aspect-video overflow-hidden rounded-card border border-line bg-black">
              <div className="skeleton absolute inset-0 rounded-none" />
              <RunStrip job={j} label="Making your images" className="absolute inset-x-4 bottom-3" />
            </div>
          ))}
        </div>
      ) : null}
      {!images.length && !running.length && !lastCanceled ? (
        <Notice text="No images yet. Describe one and press Generate." />
      ) : null}
      {first && !compact ? (
        <section className="mb-10">
          <div className="slate mb-3 text-ink3" data-testid="batch-meta">
            {[relativeTime(first[0].created_at), first[0].model_name ?? first[0].model, seedsText(first), madeIn(first[0].took_s)].filter(Boolean).join(" · ")}
          </div>
          <div className={cx("grid gap-4", first.length === 1 ? "grid-cols-1" : "grid-cols-2")}>
            {first.map((im) => (
              <ImageTile key={im.id} im={im} large onOpen={() => setDetail({ im })} a={actions(im)} />
            ))}
          </div>
        </section>
      ) : null}
      {(compact ? batches.flat() : rest.flat()).length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">
          {(compact ? batches.flat() : rest.flat()).map((im) => (
            <ImageTile key={im.id} im={im} onOpen={() => setDetail({ im })} a={actions(im)} />
          ))}
        </div>
      ) : null}

      {detail ? <ImageDetail pid={pid} im={detail.im} castOpen={!!detail.castOpen} onClose={() => setDetail(null)} a={actions(detail.im)} /> : null}
      {use ? <ShotPicker pid={pid} im={use.im} kind={use.kind} onClose={() => setUse(null)} /> : null}
    </div>
  );
}

/** "Seed 41" for one picture, "Seeds 41–44" for a batch (each picture takes the next seed). */
function seedsText(batch: ProjectImage[]): string {
  const s = batch.map((im) => im.seed).filter((n): n is number => typeof n === "number").sort((a, b) => a - b);
  if (!s.length) return "";
  return s.length === 1 || s[0] === s[s.length - 1] ? `Seed ${s[0]}` : `Seeds ${s[0]}–${s[s.length - 1]}`;
}

function handToImagePage(patch: Record<string, unknown>) {
  setState({ imageDraft: { seq: (getState().imageDraft?.seq ?? 0) + 1, patch } });
  navigate({ name: "create", panel: "image" });
}

type Actions = Record<"download" | "cast" | "edit" | "look" | "first" | "ref" | "same", () => void>;

const ACTIONS: Array<{ key: keyof Actions; label: string; icon: LucideIcon }> = [
  { key: "download", label: "Download PNG", icon: Download },
  { key: "cast", label: "Save as character", icon: UserPlus },
  { key: "edit", label: "Edit", icon: PenLine },
  { key: "look", label: "Use as look frame", icon: Clapperboard },
  { key: "first", label: "Use as first frame", icon: CloudSnow },
  { key: "ref", label: "Use as reference", icon: ImagePlus },
  { key: "same", label: "Same settings", icon: RotateCcw },
];

function ImageTile({ im, onOpen, a, large }: { im: ProjectImage; onOpen: () => void; a: Actions; large?: boolean }) {
  return (
    <div className="group relative overflow-hidden rounded-card bg-black hairline-frame" style={{ aspectRatio: im.width && im.height ? `${im.width} / ${im.height}` : "16 / 9" }}>
      <button type="button" onClick={onOpen} className="absolute inset-0" aria-label="Open image">
        <Thumb src={large ? im.url : im.thumb_url || im.url} className="h-full w-full" />
      </button>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center p-2 opacity-0 transition-opacity duration-150 group-focus-within:opacity-100 group-hover:opacity-100">
        <div className="pointer-events-auto flex items-center gap-0.5 rounded-full bg-black/80 p-1">
          {ACTIONS.map((x) => (
            <Tip key={x.key} text={x.label}>
              <button type="button" aria-label={x.label} onClick={a[x.key]} className="flex h-7 w-7 items-center justify-center rounded-full text-ivory/75 transition-colors hover:bg-white/10 hover:text-ivory">
                <x.icon size={14} strokeWidth={1.5} />
              </button>
            </Tip>
          ))}
        </div>
      </div>
    </div>
  );
}

/** The kind an image most likely is, from its prompt: a car is a prop, a street is a place (Fresh-9: a coupe saved as a
 * Person). People first, since "a woman by her car" is a person. One click on the chips changes it. */
function kindFor(prompt: string): CastKind {
  const p = prompt.toLowerCase();
  if (/\b(man|woman|person|girl|boy|kid|child|people|portrait|face|actor|actress|he|she|character)\b/.test(p)) return "person";
  if (/\b(car|coupe|suv|sedan|truck|van|bike|bicycle|motorbike|boat|phone|bottle|bag|chair|lamp|watch|shoe|product|prop)s?\b/.test(p)) return "prop";
  if (/\b(street|road|room|kitchen|office|beach|forest|park|city|town|lake|field|desert|interior|exterior|landscape|location|place)s?\b/.test(p)) return "place";
  return "person";
}

/** A picture's prompt as a cast line: without a sheet caption's framing ("Car reference sheet on a plain light gray studio
 * background: the same ..."), and never cut mid-sentence: a long one keeps its whole clauses (start_frame.py cast_line). */
export function castLine(prompt: string, n = 140): string {
  const words = prompt
    .replace(/@/g, "")
    .replace(/^[^:]{0,90}\b(reference sheet|character sheet|turnaround|model sheet|studio background|backdrop)\b[^:]*:\s*(the same\s+)?/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (words.length <= n) return words;
  const cut = words.slice(0, n);
  const at = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf(", "), cut.lastIndexOf("; "));
  return at > 0 ? cut.slice(0, at) : words;
}

function ImageDetail({ pid, im, castOpen, onClose, a }: { pid: string; im: ProjectImage; castOpen: boolean; onClose: () => void; a: Actions }) {
  const [casting, setCasting] = useState(castOpen);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<CastKind>(() => kindFor(im.prompt));
  const [line, setLine] = useState(() => castLine(im.prompt));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recipe = [im.model_name ?? im.model, im.width && im.height ? `${im.width} × ${im.height}` : null, im.aspect, im.resolution, im.seed != null ? `seed ${im.seed}` : null, im.mode, madeIn(im.took_s) || null]
    .filter(Boolean)
    .join(" · ");
  const save = async () => {
    setBusy(true);
    const err = await imageToCast(pid, im.id, { name: name.trim(), kind, line: line.trim() });
    setBusy(false);
    if (err) setError(err);
    else setCasting(false);
  };
  return (
    <Modal open onClose={onClose} label="Image" width={1100}>
      <div className="grid grid-cols-[1fr_320px] max-[900px]:grid-cols-1">
        <div className="flex items-center justify-center bg-black p-4">
          <img src={im.url} alt="" className="max-h-[72vh] w-auto max-w-full object-contain" />
        </div>
        <div className="space-y-5 p-6">
          <div className="flex items-start justify-between">
            <Slate>{recipe}</Slate>
            <button type="button" onClick={onClose} aria-label="Close" className="rounded-full p-1 text-ink3 hover:text-ivory">
              <X size={15} />
            </button>
          </div>
          <div>
            <div className="slate mb-1 text-ink3">Prompt</div>
            <p className="text-[13px] leading-snug text-ink">{im.prompt || "—"}</p>
          </div>
          {im.prompt_sent && im.prompt_sent !== im.prompt ? (
            <div>
              <div className="slate mb-1 text-ink3">What the model read</div>
              <p className="mono whitespace-pre-wrap break-words text-[11px] leading-[17px] text-ink2">{im.prompt_sent}</p>
            </div>
          ) : null}
          {casting ? (
            <div className="space-y-2.5 border-t border-line pt-4">
              <div className="slate text-ink3">Save as character</div>
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Name, e.g. Maya" className="field h-9" aria-label="Name" />
              <div className="flex flex-wrap gap-1.5">
                {(["person", "place", "prop", "look"] as CastKind[]).map((k) => (
                  <button key={k} type="button" onClick={() => setKind(k)} className={cx("h-7 rounded-full border px-3 text-[12px] capitalize", kind === k ? "border-ivory/60 text-ivory" : "border-line text-ink2")}>
                    {k}
                  </button>
                ))}
              </div>
              <textarea rows={3} value={line} onChange={(e) => setLine(e.target.value)} className="field h-auto resize-none py-2 text-[13px]" aria-label="One line that describes it" />
              {error ? (
                <p className="flex items-baseline gap-2 text-[13px] text-ink">
                  <span className="slate">Error</span>
                  {error}
                </p>
              ) : null}
              <div className="flex gap-2">
                <Button variant="primary" size="sm" loading={busy} disabled={!name.trim()} onClick={() => void save()}>
                  Save as character
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setCasting(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-1 border-t border-line pt-4">
              {ACTIONS.map((x) => (
                <button
                  key={x.key}
                  type="button"
                  onClick={() => {
                    if (x.key === "cast") setCasting(true);
                    else {
                      if (x.key !== "download") onClose();
                      a[x.key]();
                    }
                  }}
                  className="flex w-full items-center gap-2.5 rounded-[4px] px-2 py-1.5 text-left text-[13px] text-ink2 transition-colors hover:bg-raised hover:text-ivory"
                >
                  <x.icon size={14} strokeWidth={1.5} />
                  {x.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}

// --- "Use as …": pick a shot ---------------------------------------------------------------------------------------

const USE_TITLE: Record<"look_frame" | "first_frame" | "reference", string> = {
  look_frame: "Use as look frame",
  first_frame: "Use as first frame",
  reference: "Use as reference",
};

function ShotPicker({ pid, im, kind, onClose }: { pid: string; im: ProjectImage; kind: "look_frame" | "first_frame" | "reference"; onClose: () => void }) {
  const ids = useApp((s) => s.projectShots[pid] ?? null);
  const shots = useApp((s) => s.shots);
  const [refFor, setRefFor] = useState<"previz_ref" | "recast_ref">("previz_ref");
  const list: Shot[] = (ids ?? []).map((x) => shots[x]).filter(Boolean);
  const pick = (shotId: string) => {
    const use: ImageUse = kind === "look_frame" ? "look_frame" : kind === "first_frame" ? "first_frame" : refFor;
    onClose();
    useImageIn(use, im, shotId);
  };
  const sub: ReactNode =
    kind === "look_frame"
      ? "Opens Previz to final with this as the look frame."
      : kind === "first_frame"
        ? "Opens Change the weather with this as the first frame."
        : null;
  return (
    <Modal open onClose={onClose} label={USE_TITLE[kind]} width={760}>
      <div className="flex items-start justify-between px-6 pb-2 pt-6">
        <div>
          <h2 className="display text-[36px]">{USE_TITLE[kind]}</h2>
          <p className="mt-2 text-[13px] text-ink2">{sub ?? "Pick a task and a shot."}</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close" className="rounded-full p-1 text-ink3 hover:text-ivory">
          <X size={15} />
        </button>
      </div>
      <div className="px-6 pb-6">
        {kind === "reference" ? (
          <div className="mb-4 mt-2">
            <Segmented<"previz_ref" | "recast_ref">
              label="Use it in"
              value={refFor}
              onChange={setRefFor}
              options={[
                { value: "previz_ref", label: "Previz to final" },
                { value: "recast_ref", label: "Replace a person" },
              ]}
            />
          </div>
        ) : null}
        <div className="slate mb-3 mt-4 text-ink3">Pick a shot</div>
        {!list.length ? (
          <Notice text="No shots in this project yet." action={{ label: "Close", onClick: onClose }} />
        ) : (
          <div className="grid max-h-[50vh] grid-cols-3 gap-3 overflow-y-auto">
            {list.map((s) => (
              <button key={s.id} type="button" onClick={() => pick(s.id)} className="group text-left">
                <Thumb src={s.latest_thumb_url || s.thumb_url} className="aspect-video w-full rounded-[4px] transition-opacity group-hover:opacity-80" />
                <span className="mt-1.5 block truncate font-mono text-[12px] text-ink2 group-hover:text-ivory">{s.name}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}

