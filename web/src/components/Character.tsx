// Characters in the step flows (SPEC section 4): the cast grid, a member's sheet with its state (not made, making
// "2 of 3", ready, failed), and New character (words, 4 options, pick one, name, Make sheet). Every picture opens the
// one full-screen viewer (CharacterViewer.tsx). The sheet state is read from the member and its newest sheet job
// (lib/sheetState.ts), so the cast page, Recast's Character step and the viewer always say the same.
import { Check, Expand } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { buildSheet, dropCharacterStill, ensureProject, getApi, loadImageModels, loadImages, loadJobs, makeCharacterStill } from "../actions";
import { useDraft } from "../lib/drafts";
import { cx, isActive } from "../lib/format";
import { freeName } from "../lib/castName";
import { sheetError, sheetJobFor, sheetLabel, sheetPrimary, sheetRequest, sheetStateOf, sheetViews, type SheetState } from "../lib/sheetState";
import { newSeed, personWords, stillRequest } from "../lib/still";
import { getState, useApp } from "../store";
import type { CastMember, Job } from "../types";
import { CharacterViewer, type ViewerItem } from "./CharacterViewer";
import { RunStrip } from "./RunStatus";
import { useEnhance } from "./Enhance";
import { Fold, keptUrl } from "./FormKit";
import { SummaryRow } from "./StepFlow";
import { Button, Modal, ProgressLine, Spinner, Tag } from "./ui";

/** Whether a picture is wide (a contact sheet with several views side by side, or a landscape photo). */
export function useWide(url: string | null | undefined): boolean {
  const [wide, setWide] = useState(false);
  useEffect(() => {
    setWide(false);
    if (!url) return;
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live) setWide(img.naturalWidth > img.naturalHeight * 1.3);
    };
    img.src = url;
    return () => {
      live = false;
    };
  }, [url]);
  return wide;
}

/** useWide, with null while the picture is still being measured (so a state that depends on it never flickers). */
function useWideKnown(url: string | null | undefined): boolean | null {
  const [st, setSt] = useState<{ url: string; wide: boolean } | null>(null);
  useEffect(() => {
    if (!url) return;
    let live = true;
    const img = new Image();
    const done = (wide: boolean) => live && setSt({ url, wide });
    img.onload = () => done(img.naturalWidth > img.naturalHeight * 1.3);
    img.onerror = () => done(false);
    img.src = url;
    return () => {
      live = false;
    };
  }, [url]);
  if (!url) return false;
  return st && st.url === url ? st.wide : null;
}

/** A cast name from the words ("a friendly green alien in a silver flight suit" → "Friendly green alien"), else the
 * photo's own name unless a camera gave it, made unique in the cast. */
export function castNameFrom(words: string, file: string, cast: CastMember[]): string {
  const fromWords = personWords(words)
    .replace(/^(?:an?|the|some)\s+/i, "")
    .split(/,|\s+(?:in|with|wearing|who|holding|and)\s+/i)[0]
    .trim();
  const stem = file.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  const camera = /^(?:img|dsc|dscn|pxl|photo|image|screenshot|whatsapp|signal|mvimg|\d)/i.test(stem);
  const base = (fromWords || (camera ? "" : stem) || "Character").slice(0, 32).trim();
  return freeName(base[0].toUpperCase() + base.slice(1), cast.map((c) => c.name));
}

// --- a member's sheet ------------------------------------------------------------------------------------------------

/** A member's sheet: its state, its views for the viewer, and Make sheet / Try again (always by cast_id). */
export function useSheet(projectId: string, member: CastMember | null) {
  const job = useApp((s) => sheetJobFor(member?.id, Object.values(s.jobs)));
  const wide = useWideKnown(member && !member.sheet ? member.url : null);
  const st = sheetStateOf(member, job, !!wide);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<{ id: string; text: string } | null>(null);
  const start = async () => {
    if (!member) return;
    setStarting(true);
    setError(null);
    try {
      await buildSheet(projectId, sheetRequest(member.id, st.s !== "none", newSeed()), true);
    } catch (e) {
      setError({ id: member.id, text: e instanceof ApiError && e.message ? e.message : "The sheet didn't start." });
    } finally {
      setStarting(false);
    }
  };
  return {
    st,
    views: sheetViews(member, st),
    start,
    starting,
    /** The picture is still being measured (a wide one is a sheet already): the primary waits. */
    measuring: wide === null,
    /** A wide picture standing in for a sheet (no views of its own). */
    widePicture: !!wide && !member?.sheet,
    error: error && error.id === member?.id ? error.text : null,
  };
}

/** "SHEET", "NO SHEET", "2 OF 3", "FAILED". */
export function SheetChip({ st, className }: { st: SheetState; className?: string }) {
  return <Tag className={cx(st.s === "ready" && "border-ivory/30 text-ivory", st.s === "failed" && "text-ink", className)}>{sheetLabel(st)}</Tag>;
}

/** A sheet view that isn't there: its name, and "Failed" when the sheet failed before it (a spinner while it's made). */
function ViewSlot({ label, state }: { label: string; state: "empty" | "making" | "failed" }) {
  return (
    <span className="absolute inset-0 flex flex-col items-center justify-center gap-1.5">
      {state === "making" ? <Spinner size={14} className="text-ink3" /> : null}
      <span className="slate text-ink4">{label}</span>
      {state === "failed" ? <span className="text-[12px] text-ink3">Failed</span> : null}
    </span>
  );
}

/** The selected member's sheet: front, back and close-up (the views so far while making, a spinner for the rest; their
 * names in the empty boxes, "Failed" in the ones a failed sheet never made) under their name, the state and the
 * progress while it's made. Each picture opens full screen, with `actions` (Use <name>) in the viewer. */
export function SheetPanel({
  member,
  sheet,
  actions,
}: {
  member: CastMember;
  sheet: ReturnType<typeof useSheet>;
  /** In the viewer, by its picture; `close` shuts the viewer first. */
  actions?: (item: ViewerItem, close: () => void) => ReactNode;
}) {
  const [viewing, setViewing] = useState<number | null>(null);
  const { st } = sheet;
  // Another member picked: their sheet comes into view (the rail scrolls on a laptop screen).
  const box = useRef<HTMLElement>(null);
  useEffect(() => {
    box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [member.id]);
  const picture: ViewerItem = { key: "picture", label: member.name, url: member.thumb_url || member.url };
  const rest = (failed: boolean): ViewerItem[] => [
    { key: "back", label: "Back", url: null, failed },
    { key: "face", label: "Close-up", url: null, failed },
  ];
  // Three boxes: the views, or the member's own picture first with the two a sheet adds.
  const tiles: ViewerItem[] =
    st.s === "ready" && member.sheet ? sheet.views.filter((v) => v.key !== "sheet") : st.s === "making" ? sheet.views : [picture, ...rest(st.s === "failed")];
  const items = sheet.views.length ? sheet.views : tiles;
  const making = st.s === "making" ? st : null;
  const error = sheetError(sheet.error || (st.s === "failed" ? st.error : ""));
  return (
    <section ref={box} className="scroll-mb-4 overflow-hidden rounded-card border border-line" aria-label={`${member.name}'s sheet`}>
      {/* Name and state first: on a laptop screen the rail ends near here, and they say which sheet this is. */}
      <div className="border-b border-line px-3 py-2.5" aria-live="polite">
        <div className="flex items-center justify-between gap-3">
          <span className="min-w-0 truncate text-[14px] text-ivory">{member.name}</span>
          <SheetChip st={st} className="shrink-0" />
        </div>
        {making ? <RunStrip job={making.job} label="Making the sheet" className="mt-2.5" /> : null}
        {error ? <p className="mt-1.5 line-clamp-2 text-[12px] leading-snug text-ink3">{error}</p> : null}
      </div>
      {sheet.widePicture ? (
        // A picture that is a sheet already: the whole of it, as it is.
        <button type="button" aria-label={`${member.name}: open full screen`} onClick={() => setViewing(0)} className="block h-28 w-full cursor-zoom-in bg-black">
          <img src={member.url} alt="" className="h-full w-full object-contain" />
        </button>
      ) : (
        <div className="grid grid-cols-3 gap-px bg-line">
          {tiles.slice(0, 3).map((it) => {
            const at = items.findIndex((x) => x.key === it.key);
            return it.url || (making && at >= 0) ? (
              <button
                key={it.key}
                type="button"
                aria-label={`${member.name}, ${it.label}: open full screen`}
                onClick={() => setViewing(Math.max(0, at))}
                className="group relative h-28 cursor-zoom-in overflow-hidden bg-black"
              >
                {it.url ? (
                  <img src={it.url} alt="" className="thumb-face h-full w-full transition-opacity duration-150 group-hover:opacity-90" />
                ) : (
                  <ViewSlot label={it.label} state="making" />
                )}
              </button>
            ) : (
              <span key={it.key} className="relative h-28 bg-black" role="img" aria-label={`${it.label}: ${it.failed ? "failed" : "not made"}`}>
                <ViewSlot label={it.label} state={it.failed ? "failed" : "empty"} />
              </span>
            );
          })}
        </div>
      )}
      {viewing != null ? (
        <CharacterViewer title={member.name} items={items} index={viewing} onIndex={setViewing} onClose={() => setViewing(null)} status={<SheetChip st={st} />} actions={actions ? (it) => actions(it, () => setViewing(null)) : undefined} />
      ) : null}
    </section>
  );
}

// --- the cast grid ---------------------------------------------------------------------------------------------------

/** The project's people as tiles: a click picks one; the corner button opens them full screen. */
export function CastGrid({ projectId, cast, value, onPick }: { projectId: string; cast: CastMember[]; value: string | null; onPick: (c: CastMember) => void }) {
  const [viewing, setViewing] = useState<{ id: string; i: number } | null>(null);
  const open = viewing ? cast.find((c) => c.id === viewing.id) ?? null : null;
  return (
    <div className="grid grid-cols-4 gap-2" role="radiogroup" aria-label="Your cast">
      {cast.map((c) => (
        <CastTile key={c.id} projectId={projectId} c={c} on={c.id === value} onPick={() => onPick(c)} onOpen={() => setViewing({ id: c.id, i: 0 })} />
      ))}
      {open && viewing ? <CastViewer projectId={projectId} member={open} index={viewing.i} onIndex={(i) => setViewing({ id: open.id, i })} onClose={() => setViewing(null)} onUse={() => onPick(open)} /> : null}
    </div>
  );
}

function CastTile({ projectId, c, on, onPick, onOpen }: { projectId: string; c: CastMember; on: boolean; onPick: () => void; onOpen: () => void }) {
  const { st } = useSheet(projectId, c);
  return (
    <div className="min-w-0">
      <div className={cx("group relative aspect-[4/5] overflow-hidden rounded-[4px] border bg-black transition [@media(max-height:820px)]:aspect-square-[border-color,box-shadow] duration-150", on ? "border-ivory shadow-[0_0_0_1px_#EEE9E0]" : "border-line hover:border-line2")}>
        <button type="button" role="radio" aria-checked={on} aria-label={c.name} onClick={onPick} className="absolute inset-0">
          <img src={c.thumb_url || c.url} alt="" className="thumb-face h-full w-full" loading="lazy" />
        </button>
        {on ? (
          <span className="pointer-events-none absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]" aria-hidden="true">
            <Check size={12} strokeWidth={3} />
          </span>
        ) : null}
        <OpenButton label={`Open ${c.name} full screen`} onClick={onOpen} />
        {/* No sheet is a quiet dot (the picked one's panel says it in words); a failure says so. */}
        {st.s === "none" ? (
          <span className="pointer-events-none absolute bottom-2 left-2 h-1.5 w-1.5 rounded-full bg-ink3 ring-1 ring-black/60" role="img" aria-label="No sheet" title="No sheet" />
        ) : st.s === "failed" ? (
          <SheetChip st={st} className="pointer-events-none absolute bottom-1.5 left-1.5 h-4 whitespace-nowrap bg-black/75 px-1 text-[9px]" />
        ) : null}
        {st.s === "making" ? <ProgressLine value={st.job.progress} idle={st.queued} className="absolute inset-x-0 top-0" /> : null}
      </div>
      {/* The name picks too (a click on it is a click on the card). */}
      <button type="button" tabIndex={-1} aria-hidden="true" onClick={onPick} className={cx("mt-1.5 line-clamp-2 w-full break-words text-left text-[12px] leading-tight", on ? "text-ivory" : "text-ink2 hover:text-ivory")}>
        {c.name}
      </button>
    </div>
  );
}

/** The corner button on a tile: opens the picture full screen (the tile itself picks). */
function OpenButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="absolute bottom-1.5 right-1.5 flex h-6 w-6 items-center justify-center rounded-[3px] bg-black/70 text-ivory/80 opacity-0 transition-opacity duration-150 hover:text-ivory focus-visible:opacity-100 group-hover:opacity-100"
    >
      <Expand size={12} />
    </button>
  );
}

/** A cast member full screen: their views and sheet, with Use <name> where they can be picked. */
export function CastViewer({ projectId, member, index, onIndex, onClose, onUse }: { projectId: string; member: CastMember; index: number; onIndex: (i: number) => void; onClose: () => void; onUse?: () => void }) {
  const sheet = useSheet(projectId, member);
  return (
    <CharacterViewer
      title={member.name}
      items={sheet.views}
      index={index}
      onIndex={onIndex}
      onClose={onClose}
      status={<SheetChip st={sheet.st} />}
      actions={
        onUse
          ? () => (
              <Button
                variant="primary"
                size="md"
                onClick={() => {
                  onUse();
                  onClose();
                }}
              >
                {`Use ${member.name}`}
              </Button>
            )
          : undefined
      }
    />
  );
}

// --- New character: words, 4 options, pick one, name, Make sheet ----------------------------------------------------

/** What New character keeps in its flow's draft: the words, the options job and the 4 pictures it made (image ids, the
 * outputs that outlive the job), the one picked, its name, and the cast member it became. */
export interface NewCharacterData {
  words: string;
  job: string | null;
  options: string[];
  picked: string | null;
  name: string;
  castId: string | null;
  error: string | null;
  /** The name was typed (an auto name moves to the next free number when someone takes it; a typed one stays). */
  nameTyped?: boolean;
}

export const NEW_CHARACTER: NewCharacterData = { words: "", job: null, options: [], picked: null, name: "", castId: null, error: null, nameTyped: false };

/** New character's state, for its step's one primary: what's needed next, in order. */
export function useNewCharacter(projectId: string, data: NewCharacterData, set: (p: Partial<NewCharacterData>) => void) {
  const cast = useApp((s) => s.cast[projectId] ?? []);
  const images = useApp((s) => s.images[projectId]);
  const job = useApp((s) => (data.job ? s.jobs[data.job] : undefined));
  const member = data.castId ? cast.find((c) => c.id === data.castId) ?? null : null;
  const sheet = useSheet(projectId, member);
  const canMake = useApp((s) => !!s.status?.connected && s.status.tiles?.image?.ready !== false);
  const words = data.words.trim();
  const name = data.name.trim().replace(/\s+/g, " ");
  const taken = !!name && cast.some((c) => c.name.toLowerCase() === name.toLowerCase());
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (data.options.length && images == null) void loadImages(projectId);
  }, [data.options.length, projectId]);
  // An auto name someone has since taken (a character made elsewhere, a reload): the next free number, never a clash.
  useEffect(() => {
    if (taken && data.nameTyped === false && !data.castId) set({ name: castNameFrom(data.words, "", cast) });
  }, [taken, data.nameTyped, data.castId]);

  // The options job finished: its 4 pictures are the draft's outputs from now on (the job id is best effort).
  useEffect(() => {
    if (!job || isActive(job)) return;
    if (job.status === "done" && job.result_image_ids?.length) {
      set({ job: null, options: job.result_image_ids, picked: null, error: null });
      void loadImages(projectId);
    } else if (job.status === "failed") set({ job: null, error: sheetError(job.error) || "The options didn't come out." });
    else set({ job: null });
  }, [job?.status]);
  // A job id the server doesn't know (it restarted): gone, never "making" forever.
  useEffect(() => {
    if (!data.job || job) return;
    let live = true;
    void loadJobs().then(() => {
      if (live && data.job && !getState().jobs[data.job]) set({ job: null });
    });
    return () => {
      live = false;
    };
  }, [data.job, !!job]);

  const urlOf = (id: string) => {
    const im = images?.find((i) => i.id === id);
    return im?.thumb_url || im?.url || keptUrl({ kind: "image", id }, projectId);
  };
  const fullOf = (id: string) => images?.find((i) => i.id === id)?.url || urlOf(id);

  const makeOptions = async () => {
    if (!words) return;
    if (data.job) dropCharacterStill(data.job);
    void loadImageModels();
    try {
      const j: Job = await makeCharacterStill(projectId, { ...stillRequest(personWords(words), newSeed()), count: 4 });
      set({ job: j.id, options: [], picked: null, error: null });
    } catch (e) {
      set({ error: sheetError(e instanceof ApiError && e.message ? e.message : "") || "The options didn't start." });
    }
  };

  /** The picked option joins the cast under its name, then its sheet is made (by cast_id, as every Try again is). */
  const makeSheet = async () => {
    if (!data.picked || !name || taken) return;
    setBusy(true);
    try {
      const c = await getApi().imageToCast(projectId, data.picked, { name, kind: "person", line: personWords(words).slice(0, 200) });
      set({ castId: c.id, error: null });
      await ensureProject(projectId, true);
      await buildSheet(projectId, sheetRequest(c.id, false, newSeed()), true);
    } catch (e) {
      set({ error: sheetError(e instanceof ApiError && e.message ? e.message : "") || "The sheet didn't start." });
    } finally {
      setBusy(false);
    }
  };

  const making = !!job && isActive(job);
  /** The step's primary for New, before the member exists; after, the sheet's own (sheetPrimary). */
  const primary: { label: string; disabled: boolean; run: () => Promise<void> } = !data.options.length
    ? { label: making ? "Making 4 options" : "Make 4 options", disabled: making || !words || !canMake, run: makeOptions }
    : { label: "Make sheet", disabled: !data.picked || !name || taken || busy, run: makeSheet };
  return { cast, member, sheet, job, making, canMake, taken, name, urlOf, fullOf, makeOptions, primary, busy };
}

/** New character's body: the words (with Enhance), then the 4 options, then the name for the one picked. Once it has
 * joined the cast, the step shows its sheet instead (the caller's SheetPanel). */
export function NewCharacterBody({
  projectId,
  data,
  set,
  nc,
  shotId,
}: {
  projectId: string;
  data: NewCharacterData;
  set: (p: Partial<NewCharacterData>) => void;
  nc: ReturnType<typeof useNewCharacter>;
  shotId?: string;
}) {
  const [viewing, setViewing] = useState<number | null>(null);
  const enh = useEnhance({ request: { target: "flux3", project_id: projectId, ...(shotId ? { shot_id: shotId } : {}) }, text: data.words, onUse: (v) => set({ words: v }) });
  const items: ViewerItem[] = data.options.map((id, i) => ({ key: id, label: `Option ${i + 1}`, url: nc.fullOf(id) }));
  const pick = (id: string) => set({ picked: id, name: data.name || castNameFrom(data.words, "", nc.cast), ...(data.name ? {} : { nameTyped: false }) });
  return (
    <div className="space-y-3">
      <div className="rounded-card border border-line bg-black/20 focus-within:border-line2">
        <textarea
          className="block w-full resize-none bg-transparent px-3.5 pb-1 pt-3 text-[14px] leading-[22px] text-ink outline-none placeholder:text-ink4"
          rows={2}
          value={data.words}
          placeholder="e.g. a knight in shining armour"
          aria-label="Who are they?"
          spellCheck
          onChange={(e) => set({ words: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !data.options.length && !nc.making) {
              e.preventDefault();
              void nc.makeOptions();
            }
          }}
        />
      </div>
      {enh.button ? (
        <Fold label="Advanced">
          <div className="space-y-2 pt-2">
            <div className="flex">{enh.button}</div>
            {enh.panel}
          </div>
        </Fold>
      ) : null}

      {nc.making && nc.job ? <RunStrip job={nc.job} label="Making 4 options" /> : null}
      {nc.making || data.options.length ? (
        <div className="grid grid-cols-4 gap-2" role="radiogroup" aria-label="Options">
          {(nc.making ? [0, 1, 2, 3].map(String) : data.options).map((id, i) =>
            nc.making ? (
              <span key={id} className="relative flex aspect-[2/3] items-center justify-center rounded-[4px] border border-line bg-black" role="status" aria-label={`Option ${i + 1}: being made`}>
                <Spinner size={13} className="text-ink3" />
              </span>
            ) : (
              <div key={id} className={cx("group relative aspect-[2/3] overflow-hidden rounded-[4px] border bg-black transition-[border-color,box-shadow] duration-150", data.picked === id ? "border-ivory shadow-[0_0_0_1px_#EEE9E0]" : "border-line hover:border-line2")}>
                <button type="button" role="radio" aria-checked={data.picked === id} aria-label={`Option ${i + 1}`} onClick={() => pick(id)} className="absolute inset-0">
                  <img src={nc.urlOf(id)} alt="" className="thumb-face h-full w-full" />
                </button>
                {data.picked === id ? (
                  <span className="pointer-events-none absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]" aria-hidden="true">
                    <Check size={10} strokeWidth={3} />
                  </span>
                ) : null}
                <OpenButton label={`Open option ${i + 1} full screen`} onClick={() => setViewing(i)} />
              </div>
            ),
          )}
        </div>
      ) : null}
      {data.options.length && !nc.making ? (
        <Button variant="secondary" size="md" block onClick={() => void nc.makeOptions()} disabled={!data.words.trim() || !nc.canMake}>
          4 more
        </Button>
      ) : null}

      {data.picked && !nc.making ? (
        <div className="animate-fade-in">
          <input className="field" value={data.name} onChange={(e) => set({ name: e.target.value, nameTyped: true })} placeholder="Name" aria-label="Name" maxLength={40} />
          {nc.taken ? <p className="mt-1.5 text-[12px] text-ink3">There's already a {nc.name} in the cast.</p> : null}
        </div>
      ) : null}
      {data.error ? (
        <p className="flex items-baseline gap-2 text-[13px] text-ink" role="alert">
          <span className="slate">Error</span>
          {data.error}
        </p>
      ) : null}

      {viewing != null && items.length ? (
        <CharacterViewer
          title="New character"
          items={items}
          index={viewing}
          onIndex={setViewing}
          onClose={() => setViewing(null)}
          actions={(it) => (
            <Button
              variant="primary"
              size="md"
              onClick={() => {
                pick(it.key);
                setViewing(null);
              }}
            >
              Use this one
            </Button>
          )}
        />
      ) : null}
    </div>
  );
}

// --- the Character flow on the cast page (pickups.flow.character) ----------------------------------------------------

/** New character on the cast page: the same body as Recast's New, in a dialog, with one primary that goes Make 4
 * options, Make sheet, Making the sheet, then Done (Try again if the sheet didn't finish). It keeps its draft per
 * project, so closing it (or reloading) keeps the words, the options and the pick. */
export function NewCharacterModal({ projectId, onClose, onImport }: { projectId: string; onClose: () => void; onImport: () => void }) {
  const dr = useDraft<{ fresh: NewCharacterData }>(`project_${projectId}`, "character", () => ({ fresh: NEW_CHARACTER }));
  const setFresh = (p: Partial<NewCharacterData>) => dr.patch((x) => ({ data: { ...x.data, fresh: { ...NEW_CHARACTER, ...(x.data.fresh as NewCharacterData), ...p } } }));
  const fresh = { ...NEW_CHARACTER, ...dr.data.fresh };
  const nc = useNewCharacter(projectId, fresh, setFresh);
  const member = nc.member;
  const [busy, setBusy] = useState(false);
  let primary: { label: string; disabled?: boolean; run: () => void | Promise<void> };
  if (!member) primary = nc.primary;
  else if (nc.sheet.st.s === "ready") {
    primary = {
      label: "Done",
      run: () => {
        dr.clear();
        onClose();
      },
    };
  } else {
    const p = sheetPrimary(nc.sheet.st, member.name);
    primary = { label: p.label, disabled: p.disabled || nc.sheet.starting || nc.sheet.measuring, run: nc.sheet.start };
  }
  return (
    <Modal open onClose={onClose} label="New character" width={600}>
      <div className="p-6">
        <h2 className="display text-[32px]">New character</h2>
        <div className="mt-5 space-y-3">
          {member ? (
            <>
              <SummaryRow thumb={fresh.picked ? nc.urlOf(fresh.picked) : null} text={personWords(fresh.words) || member.name} edit="Change" onEdit={() => setFresh({ castId: null })} />
              <SheetPanel member={member} sheet={nc.sheet} />
            </>
          ) : (
            <NewCharacterBody projectId={projectId} data={fresh} set={setFresh} nc={nc} />
          )}
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => {
              onClose();
              onImport();
            }}
            className="text-[13px] text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline"
          >
            Copy from another project
          </button>
          <Button
            variant="primary"
            size="md"
            disabled={primary.disabled}
            loading={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await primary.run();
              } finally {
                setBusy(false);
              }
            }}
          >
            {primary.label}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

/** A cast card's sheet action on the cast page: Make sheet, Try again or Redo sheet (none while it's made). */
export function CastSheetAction({ projectId, member }: { projectId: string; member: CastMember }) {
  const sheet = useSheet(projectId, member);
  const { st } = sheet;
  const label = st.s === "none" ? "Make sheet" : st.s === "failed" ? "Try again" : st.s === "ready" && member.sheet ? "Redo sheet" : null;
  return label && projectId !== "examples" && !sheet.measuring ? (
    <Button variant="secondary" size="sm" className="shrink-0" onClick={() => void sheet.start()} loading={sheet.starting}>
      {label}
    </Button>
  ) : null;
}

/** A cast card's failed sheet, on its picture ("FAILED"). The rest says itself: a sheet is the picture, making shows
 * its progress over it, and no sheet has Make sheet under it. */
export function CastSheetChip({ projectId, member }: { projectId: string; member: CastMember }) {
  const { st } = useSheet(projectId, member);
  return st.s === "failed" ? <SheetChip st={st} className="pointer-events-none absolute right-3 top-3 bg-black/70" /> : null;
}
