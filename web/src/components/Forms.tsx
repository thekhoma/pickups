import { choiceOf, FINISH_SHARE, finishChoices, finishTitle, type FinishChoice } from "../lib/finish";
import { ArrowRight, ArrowUp, Check, Expand, MousePointerClick, RotateCcw, RotateCw, Shrink, Sparkles, Undo2, UserRound, X, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ComponentType, type ReactNode } from "react";
import { ApiError } from "../api";
import {
  buildSheet,
  cancelJob,
  clearSelection,
  dropCharacterStill,
  endSelection,
  ensureProject,
  getApi,
  loadImageModels,
  loadImages,
  makeCharacterStill,
  pickPerson,
  restoreSelection,
  selectByText,
  setAutoClick,
  setSelectionFinding,
  setSelectionMany,
  showKeptSelection,
  startSelection,
  undoPoint,
} from "../actions";
import { aboutTime, ALT_KEY, cx, isActive, shortGpu } from "../lib/format";
import { CARS, RELIGHT } from "../lib/features";
import { queueShort, useQueuePlace } from "../lib/hooks";
import { elapsedClock, useElapsed } from "../lib/runtime";
import { scan } from "../lib/mentions";
import { usePlayer } from "../lib/player";
import { isStillPrompt, ltxRecastPrompt, newSeed, personWords, scailPrompt, stillRequest, stillWords } from "../lib/still";
import { getState, useApp, type Selection } from "../store";
import type { Amount, CastMember, JobParams, PeopleResult, PersonFound, Point, RecastEngine, RecastParams, RecastSwap, Ref as RefPick, ReshootMove, SheetRequest, Shot, TaskId, TrackResult } from "../types";
import { bool, Chip, Fold, FormShell, keptFrom, keptUrl, MoreOptions, PrimaryAction, RefPicture, str, useLengthCap, usePrefill, useShotWindow, useTask, useUploadSlot, type FormProps } from "./FormKit";
import { useEnhance } from "./Enhance";
import { MentionField, PromptReceipt, useCast } from "./Mentions";
import { EnvironmentForm, PrevizForm } from "./FormsCreative";
import { RadioRow } from "./DirectorSetup";
import { DownloadControl } from "./DownloadControl";
import { StepFlow, SummaryRow, type FlowStep } from "./StepFlow";
import { CastGrid, castNameFrom, NEW_CHARACTER, NewCharacterBody, SheetPanel, useNewCharacter, useSheet, useWide, type NewCharacterData } from "./Character";
import { sheetError, sheetPrimary } from "../lib/sheetState";
import { flowOn, useDraft } from "../lib/drafts";
import { finishModels, finishParams, recastParams, RELIGHT_MODEL, relightParams, removeParams, RELIGHT_DEFAULT, RELIGHT_FROM, RELIGHT_LOOKS, type RecastWho, type RelightFrom } from "../lib/flowParams";
import { Button, DropSlot, Hint, Icon, Notice, ProgressLine, Segmented, Spinner, SwitchRow, Tag } from "./ui";

/** A vehicle named in words (mirrors recast_ltx.VEHICLE_WORDS). */
const VEHICLE_WORDS = /^(?:the |a |an )?(?:[\w-]+ ){0,2}(car|suv|truck|pickup|van|bus|jeep|taxi|cab|sedan|hatchback|wagon|coupe|convertible|vehicle|motorbike|motorcycle|scooter|tractor|lorry)s?$/i;

export function TaskForm({ task, shot, ui }: FormProps & { task: TaskId }) {
  switch (task) {
    case "remove":
      return <Flow task="remove" shot={shot} ui={ui} Comp={RemoveFlow} />;
    case "recast":
      return flowOn("recast") ? <Flow task="recast" shot={shot} ui={ui} Comp={RecastFlow} /> : <RecastForm shot={shot} ui={ui} />;
    case "environment":
      return <EnvironmentForm shot={shot} ui={ui} />;
    case "previz":
      return <PrevizForm shot={shot} ui={ui} />;
    case "reshoot":
      return <ReshootForm shot={shot} ui={ui} />;
    case "finish":
      return <Flow task="finish" shot={shot} ui={ui} Comp={FinishFlow} />;
    case "relight":
      return <Flow task="relight" shot={shot} ui={ui} Comp={RelightFlow} />;
  }
}

/** A step flow, started again when "Try again" hands it a take's settings (a new prefill for its task): the rail keeps
 * the same form open, so the flow would never read them otherwise. */
function Flow({ task, shot, ui, Comp }: FormProps & { task: TaskId; Comp: ComponentType<FormProps> }) {
  const pf = useApp((s) => s.prefill[shot.id]);
  const seq = pf && pf.task === task ? pf.seq : null;
  const [gen, setGen] = useState(0);
  const last = useRef(seq);
  useEffect(() => {
    if (seq != null && seq !== last.current) setGen((g) => g + 1);
    last.current = seq;
  }, [seq]);
  return <Comp key={gen} shot={shot} ui={ui} />;
}

function useSelection(shotId: string) {
  return useApp((s) => (s.selection && s.selection.shotId === shotId ? s.selection : null));
}

/** A take's pick, for "Try again with changes" (and an example's "Try it"): where it was made and what it was. */
type Pick = { points: Point[]; frame: number; text: string; from: string };

/** Whether something is selected: a found mask, or a take's pick put back as it was (no find until it changes). */
function picked(sel: Selection | null): boolean {
  return !!sel && (!!sel.maskUrl || !!sel.kept) && !sel.loading;
}

/** Whether the selection is still exactly the take's pick (then what the take ran with for it still applies). */
function samePick(sel: Selection | null, pick: Pick | null): boolean {
  return !!sel && !!pick && sel.frame === pick.frame && sel.text === pick.text && JSON.stringify(sel.points) === JSON.stringify(pick.points);
}

/** Click-to-select for the form's lifetime: on while it's open, with a take's pick put back as it was. */
function useSelectionSession(shotId: string, restore: Pick | null | undefined) {
  const player = usePlayer();
  useEffect(() => {
    startSelection(shotId);
    if (restore) {
      player.seek(restore.frame);
      // Shown as selected straight away: no find runs until the pick changes (or "Show").
      restoreSelection(shotId, restore, restore.from);
    }
    return () => endSelection();
    // Only when the form opens.
  }, [shotId]);
}

/** The pick's status row: finding, an error, "Selected · 2 clicks" (with Show, Undo and Clear), or what to do. */
function SelectionStatus({ shot, idle, onClear }: { shot: Shot; idle: ReactNode; onClear?: () => void }) {
  const sel = useSelection(shot.id);
  const n = sel?.points.length ?? 0;
  const hasMask = !!sel?.maskUrl;
  const kept = !hasMask && sel?.kept ? sel.kept : null;
  return (
    <div
      className={cx(
        "flex min-h-[52px] items-center gap-3 rounded-card border px-3.5 py-2.5 text-[13px] transition-colors duration-150",
        sel?.error ? "border-line2" : hasMask || kept ? "border-ivory/40 bg-raised" : "border-line",
      )}
      aria-live="polite"
    >
      {sel?.loading ? (
        <>
          <Spinner size={14} className="text-ivory" />
          <span className="text-ink">Finding it…</span>
        </>
      ) : sel?.error ? (
        <span className="flex items-baseline gap-2 text-ink">
          <span className="slate">Error</span>
          {sel.error}
        </span>
      ) : hasMask || kept ? (
        <>
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]">
            <Check size={12} strokeWidth={3} />
          </span>
          {/* Where a kept pick came from sits on a line of its own, so a narrow rail never breaks it up. */}
          <span className="min-w-0 flex-1 text-ink">
            <span className="block truncate" title={sel?.text || undefined}>
              Selected
              <span className="text-ink3">{n ? ` · ${n} ${n === 1 ? "click" : "clicks"}` : sel?.text ? ` · “${sel.text}”` : ""}</span>
            </span>
            {kept ? <span className="block text-[12px] leading-4 text-ink3">Same as {kept}</span> : null}
          </span>
        </>
      ) : (
        <>
          <Icon icon={MousePointerClick} className="shrink-0 text-ivory/80" />
          <span className="text-ink2">{idle}</span>
        </>
      )}
      {(n > 0 || hasMask || kept) && !sel?.loading ? (
        <span className="ml-auto flex shrink-0 items-center gap-0.5">
          {kept ? (
            <button
              type="button"
              onClick={showKeptSelection}
              title={`Find it on frame ${sel?.frame ?? 0} to see it over the picture`}
              className="rounded-full px-2.5 py-1 text-[12px] text-ink2 transition-colors hover:bg-line hover:text-ivory"
            >
              Show
            </button>
          ) : null}
          {n > 0 ? (
            <button type="button" onClick={undoPoint} aria-label="Undo last click" className="rounded-full p-1.5 text-ink2 transition-colors hover:bg-line hover:text-ivory">
              <Undo2 size={16} />
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => {
              onClear?.();
              clearSelection();
            }}
            className="rounded-full px-2.5 py-1 text-[12px] text-ink2 transition-colors hover:bg-line hover:text-ivory"
          >
            Clear
          </button>
        </span>
      ) : null}
    </div>
  );
}

// --- Recast ----------------------------------------------------------------------------------------------------------
// Three things, one button: who to replace (found for you), your character (a cast pick, a photo or sheet, or a
// still FLUX 3 klein makes from your words), a few words about them. Everything else waits under "More options".

function restoreOf(pre: Record<string, unknown> | null): Pick | null {
  if (!pre) return null;
  const points = Array.isArray(pre.points) ? (pre.points as Point[]) : [];
  const text = str(pre.text);
  if (!points.length && !text) return null;
  return { points, frame: typeof pre.frame === "number" ? pre.frame : 0, text, from: keptFrom(pre) };
}

/** Who a Recast take put in when FLUX 3 klein made them from words inside the task: the still if the take kept one
 * (reference_url), the take it's on, and the words (`describe`, or FLUX's own caption `reference_prompt`). */
function madePerson(pre: Record<string, unknown> | null): { url: string; take: string; text: string; caption: boolean } | null {
  if (!pre || pre.cast_id || pre.who_ref || pre.ref_upload_id) return null;
  const describe = str(pre.describe);
  const caption = str(pre.reference_prompt);
  const url = str(pre._reference_url);
  // Words that @-mention the cast, with no still or caption: the cast entry was the reference, not a made still.
  if (!url && !caption && (!describe || MENTION.test(describe))) return null;
  return { url, take: url ? str(pre._take) : "", text: describe || caption, caption: !describe && !!caption };
}

const MENTION = /(^|\s)@\S/;
/** Before a character is picked the words say who they are (FLUX 3 klein makes them); after, what to change. */
const WHO_PLACEHOLDER = "Who are they? e.g. an alien in a silver flight suit";
const CHANGE_PLACEHOLDER = "e.g. no helmet, holding a lantern";
const SLOT_COPY = "A character sheet (front, back, close-up) works best. A full-body photo is fine.";

type Who = "cast" | "image" | "photo" | "kept" | "made" | "still" | "words";

/** A still FLUX 3 klein is making (or made) from these words, as a project image (lib/still.ts). */
interface Still {
  words: string;
  seed: number;
  jobId: string | null;
  imageId?: string;
  url?: string;
  error?: string;
}

// --- who to replace: the people finder ----------------------------------------------------------------------------

type Finder = { s: "finding" } | { s: "found"; res: PeopleResult } | { s: "failed" };

/** This session's answers, per shot and take (the backend caches them too, so a reopened form is instant either way). */
const peopleCache = new Map<string, PeopleResult>();

function usePeople(shotId: string, versionId: string): Finder {
  const key = `${shotId}:${versionId}`;
  const [st, setSt] = useState<Finder>(() => {
    const hit = peopleCache.get(key);
    return hit ? { s: "found", res: hit } : { s: "finding" };
  });
  useEffect(() => {
    const hit = peopleCache.get(key);
    if (hit) {
      setSt({ s: "found", res: hit });
      return;
    }
    let live = true;
    setSt({ s: "finding" });
    getApi()
      .people(shotId, versionId !== "orig" ? { version_id: versionId } : {})
      .then((res) => {
        if (!res.partial) peopleCache.set(key, res);
        if (live) setSt({ s: "found", res: { ...res, people: res.people ?? [] } });
      })
      // Offline, SAM missing, an older backend: the click panel takes over, quietly.
      .catch(() => live && setSt({ s: "failed" }));
    return () => {
      live = false;
    };
  }, [key]);
  return st;
}

/** Where to click for a person: the finder's points (chest, head, legs), or its one point on older answers. */
function personPoints(p: PersonFound): Point[] {
  const pts = p.points?.length ? p.points : [p.point];
  return pts.map((q) => ({ x: Math.round(q.x), y: Math.round(q.y), positive: true }));
}

function samePoints(a: Point[], b: Point[]): boolean {
  return a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y && p.positive === b[i].positive);
}

/** The finder's cutout of the person a Recast replaces, when the pick was the finder's (the running panel shows it). */
export function foundPersonThumb(shotId: string, versionId: string, frame: number | undefined, points: Point[] | undefined): string | null {
  const res = peopleCache.get(`${shotId}:${versionId}`);
  if (!res || !points?.length || frame !== res.frame) return null;
  const hit = res.people.find((p) => {
    const own = personPoints(p);
    return samePoints(points.slice(0, own.length), own);
  });
  return hit?.thumb_url ?? null;
}

function mainOf(res: PeopleResult): PersonFound | null {
  return res.people.find((p) => p.main) ?? res.people.find((p) => p.index === res.main) ?? res.people[0] ?? null;
}

/** The found person a click in the picture lands on: the smallest box it is in (the most specific one there). */
function personAt(people: PersonFound[], pt: Point): PersonFound | null {
  const inside = people.filter((p) => {
    const [x, y, w, h] = p.box;
    return pt.x >= x && pt.x <= x + w && pt.y >= y && pt.y <= y + h;
  });
  inside.sort((a, b) => a.box[2] * a.box[3] - b.box[2] * b.box[3]);
  return inside[0] ?? null;
}

const ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];

/** "the only person", "the person most on screen", "the person on the left" (people come left to right). */
function whoLabel(people: PersonFound[], i: number): string {
  const n = people.length;
  if (n === 1) return "the only person";
  if (people[i].main) return "the person most on screen";
  if (n === 2) return i === 0 ? "the person on the left" : "the person on the right";
  if (i === 0) return "the person on the far left";
  if (i === n - 1) return "the person on the far right";
  return `the ${ORDINAL[i] ?? `${i + 1}th`} person from the left`;
}

const narrowQuery = typeof window !== "undefined" ? window.matchMedia("(max-width: 767px)") : null;

/** Phone width: the picture sits above the rail (and is often scrolled away), and people tap rather than click. */
function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => !!narrowQuery?.matches);
  useEffect(() => {
    if (!narrowQuery) return;
    const on = () => setNarrow(narrowQuery.matches);
    narrowQuery.addEventListener("change", on);
    return () => narrowQuery.removeEventListener("change", on);
  }, []);
  return narrow;
}

/** Bring the picture back into view (phones, where it scrolls away above the rail). */
function showPicture() {
  const stage = document.querySelector<HTMLElement>("[data-stage]");
  const r = stage?.getBoundingClientRect();
  if (stage && r && (r.top < 0 || r.bottom > window.innerHeight)) stage.scrollIntoView({ behavior: "smooth", block: "center" });
}

function RowButton({ children, onClick, title, label }: { children: ReactNode; onClick: () => void; title?: string; label?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={label}
      className="shrink-0 rounded-full px-2.5 py-1 text-[12px] text-ink2 transition-colors hover:bg-line hover:text-ivory"
    >
      {children}
    </button>
  );
}

/** One line of Recast's sentence: "Replacing [cutout] the only person · Not them?", "With [thumb] Alien · Change". */
function SentenceRow({
  lead,
  thumb,
  icon = MousePointerClick,
  busy,
  contain,
  label,
  sub,
  tag,
  quiet,
  children,
}: {
  lead: string;
  thumb?: string | null;
  icon?: LucideIcon;
  /** The picture is on its way (a still being made). */
  busy?: boolean;
  /** Show the whole picture (photos and sheets), not a crop. */
  contain?: boolean;
  label: string;
  sub?: ReactNode;
  tag?: string;
  /** Not a choice (an error): no chosen border. */
  quiet?: boolean;
  children?: ReactNode;
}) {
  return (
    <div
      className={cx(
        "flex min-h-[52px] items-center gap-3 rounded-card border py-1.5 pl-1.5 pr-2 text-[13px] animate-fade-in",
        quiet ? "border-line2" : "border-ivory/40 bg-raised",
      )}
      aria-live="polite"
    >
      <span className="relative flex h-12 w-9 shrink-0 items-center justify-center overflow-hidden rounded-[3px] bg-black text-ivory/80" aria-hidden="true">
        {busy ? (
          <>
            <span className="skeleton absolute inset-0 rounded-none" />
            <Spinner size={13} className="relative" />
          </>
        ) : thumb ? (
          <RefPicture url={thumb} className={cx("h-full w-full", contain && "!object-contain")} />
        ) : (
          <Icon icon={icon} size={15} />
        )}
      </span>
      <span className="min-w-0 flex-1 leading-snug">
        <span className="line-clamp-2 text-ink3" title={label}>
          {lead} <span className="text-ink">{label}</span>
          {tag ? <Tag className="ml-1.5 h-4 align-[1px]">{tag}</Tag> : null}
        </span>
        {sub ? (
          <span className="mt-0.5 block truncate text-[12px] text-ink3" title={typeof sub === "string" ? sub : undefined}>
            {sub}
          </span>
        ) : null}
      </span>
      {children ? <span className="ml-auto flex shrink-0 items-center gap-0.5">{children}</span> : null}
    </div>
  );
}

function BusyRow({ text }: { text: string }) {
  return (
    <div className="flex min-h-[52px] items-center gap-3 rounded-card border border-line px-3.5 py-2.5 text-[13px]" role="status">
      <Spinner size={14} className="text-ink3" />
      <span className="text-ink2">{text}</span>
    </div>
  );
}

// --- Recast anyone: one row per person the finder found -------------------------------------------------------------

/** "[cutout] → [Knight] Knight · Change", or "[cutout] As filmed · Replace". */
function PersonRow({
  person,
  where,
  swapped,
  thumb,
  icon = UserRound,
  busy,
  contain,
  label,
  sub,
  active,
  children,
}: {
  /** Their cutout. */
  person: string;
  /** "the person on the left". */
  where: string;
  swapped: boolean;
  thumb?: string | null;
  icon?: LucideIcon;
  busy?: boolean;
  contain?: boolean;
  label: string;
  sub?: ReactNode;
  /** Their character is being picked. */
  active?: boolean;
  children?: ReactNode;
}) {
  return (
    <div
      className={cx(
        "flex min-h-[60px] items-center gap-2 rounded-card border py-1.5 pl-1.5 pr-2 text-[13px] transition-colors duration-150",
        active ? "border-ivory bg-raised" : swapped ? "border-ivory/40 bg-raised" : "border-line",
      )}
    >
      <span className="h-12 w-9 shrink-0 overflow-hidden rounded-[3px] bg-black" aria-hidden="true">
        <img src={person} alt="" className={cx("thumb-face h-full w-full", !swapped && "opacity-70")} />
      </span>
      {swapped ? (
        <>
          <ArrowRight size={13} strokeWidth={1.75} className="shrink-0 text-ink3" aria-hidden="true" />
          <span className="relative flex h-12 w-9 shrink-0 items-center justify-center overflow-hidden rounded-[3px] bg-black text-ivory/80" aria-hidden="true">
            {busy ? (
              <>
                <span className="skeleton absolute inset-0 rounded-none" />
                <Spinner size={13} className="relative" />
              </>
            ) : thumb ? (
              <RefPicture url={thumb} className={cx("h-full w-full", contain && "!object-contain")} />
            ) : (
              <Icon icon={icon} size={15} />
            )}
          </span>
        </>
      ) : null}
      <span className="ml-1 min-w-0 flex-1 leading-snug">
        <span className={cx("block truncate", swapped ? "text-ink" : "text-ink3")} title={label}>
          {label}
        </span>
        <span className="mt-0.5 line-clamp-2 text-[12px] text-ink3" title={where}>
          {where}
          {sub ? <> · {sub}</> : null}
        </span>
      </span>
      {children ? <span className="ml-auto flex shrink-0 items-center gap-0.5">{children}</span> : null}
    </div>
  );
}

function CastChips({ cast, castId, onPick }: { cast: CastMember[]; castId: string | null; onPick: (c: CastMember) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Your cast">
      {cast.map((c) => {
        const on = c.id === castId;
        return (
          <button
            key={c.id}
            type="button"
            role="radio"
            aria-checked={on}
            title={on ? `Keep ${c.name}` : c.line}
            onClick={() => onPick(c)}
            className={cx(
              "inline-flex h-8 items-center gap-2 rounded-full border pl-1 text-[13px] transition-colors duration-150",
              on ? "border-ivory bg-ivory pr-2.5 font-medium text-[#0D0C0B]" : "border-line pr-3 text-ink2 hover:border-line2 hover:text-ink",
            )}
          >
            <img src={c.thumb_url || c.url} alt="" className="thumb-face h-6 w-6 rounded-full" />
            {c.name}
            {on ? <Check size={13} strokeWidth={2.5} aria-hidden="true" /> : null}
          </button>
        );
      })}
    </div>
  );
}

/** Who another person becomes: a cast member, or words. */
function SwapPicker({
  cast,
  castId,
  describe,
  onPick,
  onCancel,
}: {
  cast: CastMember[];
  castId: string | null;
  describe: string;
  onPick: (who: { castId: string | null; describe: string }) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(describe);
  const words = text.trim();
  const applyWords = () => {
    if (words) onPick({ castId: null, describe: words });
  };
  return (
    <div className="ml-4 space-y-2.5 border-l border-line pl-3 animate-fade-in">
      {cast.length ? <CastChips cast={cast} castId={castId} onPick={(c) => onPick({ castId: c.id, describe: "" })} /> : null}
      <div className="flex items-center gap-1.5">
        <input
          className="field h-9 min-w-0 flex-1 text-[13px]"
          value={text}
          placeholder={cast.length ? "Or describe them" : "Who do they become?"}
          aria-label="Who do they become?"
          autoFocus={!cast.length}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              applyWords();
            } else if (e.key === "Escape") onCancel();
          }}
        />
        {words ? (
          <Button variant="secondary" size="sm" className="shrink-0" onClick={applyWords}>
            Use
          </Button>
        ) : null}
        <RowButton onClick={onCancel}>Cancel</RowButton>
      </div>
    </div>
  );
}

/** Whether a picture has loaded (or failed, or is slow): rows wait for their thumbnail rather than show an empty box. */
function useLoaded(url: string | null | undefined): boolean {
  const [done, setDone] = useState<string | null>(null);
  useEffect(() => {
    if (!url) return;
    let live = true;
    const end = () => {
      if (live) setDone(url);
    };
    const img = new Image();
    img.onload = end;
    img.onerror = end;
    img.src = url;
    if (img.complete) end();
    const t = window.setTimeout(end, 4000); // never hold the row for a slow thumbnail
    return () => {
      live = false;
      window.clearTimeout(t);
    };
  }, [url]);
  return !url || done === url;
}

function WhoToReplace({
  shot,
  finder,
  autoIdx,
  takenOver,
  kept,
  narrow,
  onChoose,
  onNotThem,
  onClear,
}: {
  shot: Shot;
  finder: Finder;
  autoIdx: number | null;
  takenOver: boolean;
  /** "Try again with changes": the take's pick, while the selection is still exactly it. */
  kept: Pick | null;
  narrow: boolean;
  onChoose: (p: PersonFound) => void;
  onNotThem: () => void;
  onClear: () => void;
}) {
  const sel = useSelection(shot.id);
  const people = finder.s === "found" ? finder.res.people : [];
  const chosen = autoIdx != null ? people[autoIdx] : null;
  const thumbReady = useLoaded(chosen?.thumb_url);
  // Until the finder's first pick is on show (its answer in, the main person picked, their cutout loaded), the row says
  // it's still looking: never an empty slot, never a "click them" that's about to be answered for you.
  const mine = !!sel && !sel.auto && (sel.points.length > 0 || !!sel.text || !!sel.kept);
  const settlingNow =
    !mine &&
    !takenOver &&
    !kept &&
    (finder.s === "finding" || (finder.s === "found" && people.length > 0 && (!chosen || !thumbReady || !!sel?.loading)));
  const revealed = useRef(false);
  if (!settlingNow) revealed.current = true;
  const settling = settlingNow && !revealed.current;
  const strip = !settling && (people.length > 1 || (people.length === 1 && !chosen));
  // Someone the finder found that the user then clicked more on: still them, with the user's clicks on top.
  const baseIdx =
    !chosen && sel?.autoBase && finder.s === "found" && sel.frame === finder.res.frame
      ? people.findIndex((p) => samePoints(sel.points.slice(0, sel.autoBase), personPoints(p)))
      : -1;
  const based = baseIdx >= 0 ? people[baseIdx] : null;
  const own = sel ? sel.points.length - (based ? sel.autoBase ?? 0 : 0) : 0; // the user's own clicks (Undo takes them back)
  const clear = () => {
    onClear();
    clearSelection();
  };

  let row: ReactNode;
  if (settling) {
    row = <BusyRow text="Finding who's in the shot…" />;
  } else if (chosen) {
    row = (
      <SentenceRow lead="Replacing" thumb={chosen.thumb_url} label={whoLabel(people, autoIdx!)}>
        <RowButton onClick={onNotThem}>Not them?</RowButton>
      </SentenceRow>
    );
  } else if (sel?.loading) {
    row = <BusyRow text="Finding it…" />;
  } else if (sel?.error) {
    row = (
      <div className="flex min-h-[52px] items-center gap-3 rounded-card border border-line2 px-3.5 py-2.5 text-[13px]" aria-live="polite">
        <span className="flex min-w-0 flex-1 items-baseline gap-2 text-ink">
          <span className="slate">Error</span>
          {sel.error}
        </span>
        <RowButton onClick={clear}>Clear</RowButton>
      </div>
    );
  } else if (sel && (sel.maskUrl || sel.kept)) {
    const label = sel.text
      ? `“${sel.text}”`
      : based
        ? whoLabel(people, baseIdx)
        : samePick(sel, kept)
          ? `your pick from ${kept!.from}`
          : "your pick";
    row = (
      <SentenceRow lead="Replacing" thumb={based?.thumb_url} label={label}>
        {sel.kept && !sel.maskUrl ? (
          <RowButton onClick={showKeptSelection} title={`Find it on frame ${sel.frame} to see it over the picture`}>
            Show
          </RowButton>
        ) : null}
        {own > 0 ? (
          <button type="button" onClick={undoPoint} aria-label="Undo last click" className="rounded-full p-1.5 text-ink2 transition-colors hover:bg-line hover:text-ivory">
            <Undo2 size={16} />
          </button>
        ) : null}
        <RowButton onClick={clear}>Change</RowButton>
      </SentenceRow>
    );
  } else {
    const none = finder.s === "found" && !people.length;
    const where = narrow ? "Tap the person to replace, above." : "Click the person to replace.";
    row = (
      <div className="flex min-h-[52px] items-center gap-3 rounded-card border border-line px-3.5 py-2.5 text-[13px]" aria-live="polite">
        <Icon icon={MousePointerClick} className="shrink-0 text-ivory/80" />
        <span className="text-ink2">{none ? `No one found. ${where}` : where}</span>
      </div>
    );
  }

  return (
    <section className="space-y-2.5" aria-label="Who to replace">
      {row}
      {strip ? (
        <div>
          {!chosen ? <div className="slate mb-2 text-ink3">{based ? "Others" : "In the shot"}</div> : null}
          <div className="flex gap-2 overflow-x-auto pb-0.5" role="radiogroup" aria-label="People in the shot">
            {people.map((p, i) => {
              const on = i === autoIdx;
              return (
                <button
                  key={`${p.index}:${p.thumb_url}`}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  aria-label={`Replace ${whoLabel(people, i)}`}
                  title={`Replace ${whoLabel(people, i)}`}
                  onClick={() => onChoose(p)}
                  className={cx(
                    "relative h-[72px] w-[54px] shrink-0 overflow-hidden rounded-[4px] border bg-black transition-[border-color,opacity,box-shadow] duration-150",
                    on ? "border-ivory shadow-[0_0_0_1px_#EEE9E0]" : "border-line opacity-70 hover:border-line2 hover:opacity-100",
                  )}
                >
                  <img src={p.thumb_url} alt="" className="thumb-face h-full w-full" />
                  {on ? (
                    <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]" aria-hidden="true">
                      <Check size={10} strokeWidth={3} />
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}

// --- check tracking: the selection followed through the shot before the render (optional) -----------------------------

/** Where the tracking slips, as times: the first frame of each run of flagged frames (runs half a second apart). */
function trackSlips(res: { flagged: number[]; fps: number }, max = 3): number[] {
  const fps = res.fps > 0 ? res.fps : 24;
  const gap = Math.max(2, Math.round(fps / 2));
  const starts: number[] = [];
  let prev = -Infinity;
  for (const f of [...res.flagged].sort((a, b) => a - b)) {
    if (f - prev > gap) starts.push(f);
    prev = f;
  }
  return starts.slice(0, max).map((f) => f / fps);
}

/** "Check tracking": SAM 3.1 follows the selection through the shot as the render will, and the shot plays back with
 * it in amber. Optional; the render reuses the result. A changed pick (points, words or frame) clears it. */
export function TrackCheck({ shot, versionId, startFrame }: { shot: Shot; versionId: string; startFrame?: number }) {
  const sel = useSelection(shot.id);
  const key = sel ? JSON.stringify([versionId, sel.frame, sel.points, sel.text]) : "";
  const [state, setState] = useState<{ key: string; busy?: boolean; res?: TrackResult; error?: string } | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const cur = state && state.key === key ? state : null;
  if (!sel || sel.loading || sel.error || (!sel.points.length && !sel.text)) return null;

  const run = async () => {
    const k = key;
    setState({ key: k, busy: true });
    try {
      const res = await getApi().track(shot.id, {
        frame: sel.frame,
        points: sel.points,
        ...(sel.text ? { text: sel.text } : {}),
        ...(versionId !== "orig" ? { version_id: versionId } : {}),
        ...(startFrame != null ? { start_frame: startFrame } : {}),
      });
      setState((s) => (s && s.key === k ? { key: k, res } : s));
    } catch (e) {
      const error = sheetError(e instanceof ApiError && e.message ? e.message : "") || "Couldn't check the tracking.";
      setState((s) => (s && s.key === k ? { key: k, error } : s));
    }
  };
  const seek = (t: number) => {
    const v = video.current;
    if (!v) return;
    v.pause();
    v.currentTime = t;
  };

  if (cur?.res) {
    const res = cur.res;
    const slips = trackSlips(res);
    return (
      <div className="space-y-2 rounded-card border border-line p-1.5 animate-fade-in">
        <video
          ref={video}
          key={res.video_url}
          src={res.video_url}
          autoPlay
          muted
          loop
          playsInline
          controls
          className="block w-full rounded-[3px] bg-black"
          aria-label="Your selection through the shot, in amber"
        />
        <div className="flex items-start gap-2 px-1.5 pb-1 text-[13px]" aria-live="polite">
          <span className="min-w-0 flex-1 leading-snug">
            <span className="block text-ink">
              {slips.length ? (
                <>
                  Loses them around{" "}
                  {slips.map((t, i) => (
                    <span key={t}>
                      {i ? (i === slips.length - 1 ? " and " : ", ") : null}
                      <button type="button" onClick={() => seek(t)} className="underline underline-offset-4 hover:text-ivory" title="Show this moment">
                        {t.toFixed(1)} s
                      </button>
                    </span>
                  ))}
                </>
              ) : (
                "Follows them all the way"
              )}
            </span>
            <span className="mt-0.5 block text-[12px] text-ink3">Looks wrong? Click them in the picture.</span>
          </span>
          <Hint text="Amber is what gets replaced. Check it stays on them, and on anything they carry, from start to end. Recast uses this, so it doesn't track them again." />
          <RowButton onClick={() => setState(null)} label="Hide the tracking check">
            Hide
          </RowButton>
        </div>
      </div>
    );
  }
  if (cur?.error) {
    return (
      <div className="flex min-h-[44px] items-center gap-3 rounded-card border border-line2 px-3.5 py-2 text-[13px]" aria-live="polite">
        <span className="flex min-w-0 flex-1 items-baseline gap-2 text-ink">
          <span className="slate">Failed</span>
          {cur.error}
        </span>
        <RowButton onClick={() => void run()}>Try again</RowButton>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      <Button variant="secondary" size="sm" loading={!!cur?.busy} onClick={() => void run()}>
        {cur?.busy ? "Tracking…" : "Check tracking"}
      </Button>
    </div>
  );
}

// --- your character -------------------------------------------------------------------------------------------------

function CardAction({ children, onClick }: { children: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className="text-[12px] text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline">
      {children}
    </button>
  );
}

/** "About 7 min on your RTX 4090.", for the engine picked (the tile's own time is the default engine's). */
function engineEta(t: ReturnType<typeof useTask>, engine: RecastEngine): string | null {
  const s = t.status?.tiles?.recast?.eta_by_engine?.[engine];
  if (s == null) return t.etaLine;
  const about = aboutTime(s);
  const gpu = t.status?.machine ? shortGpu(t.status.machine.gpu) : null;
  return about ? `${about}${gpu ? ` on your ${gpu}` : ""}.` : null;
}

/** Under one photo or a still made from words: build a full sheet (front, back, close-up) into the project's cast.
 * Optional; once it's built it becomes the character. */
function SheetOffer({ projectId, from, name, onBuilt }: { projectId: string; from: SheetRequest | null; name: string; onBuilt: (castId: string) => void }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [builtFor, setBuiltFor] = useState<string | null>(null);
  const job = useApp((s) => (jobId ? s.jobs[jobId] : undefined));
  const elapsed = useElapsed(job);
  const place = useQueuePlace(job);
  const key = from ? from.upload_id || from.image_id || "" : "";
  const keyNow = useRef(key);
  keyNow.current = key;
  const startedFor = useRef("");
  useEffect(() => setError(null), [key]); // another picture: an earlier sheet's error no longer applies
  useEffect(() => {
    if (!job || isActive(job)) return;
    setJobId(null);
    if (job.status === "done" && job.result_cast_id) {
      const id = job.result_cast_id;
      // Still the same picture: the sheet becomes the character (once the cast has them).
      if (keyNow.current === startedFor.current) {
        setBuiltFor(startedFor.current);
        void ensureProject(projectId, true).then(() => onBuilt(id));
      }
    } else if (job.status === "failed") setError(job.error || "Something went wrong on the GPU.");
  }, [job?.status]);
  if (!jobId && (!from || builtFor === key)) return null;
  const start = async () => {
    if (!from) return;
    setStarting(true);
    setError(null);
    try {
      const j = await buildSheet(projectId, { ...from, name }, true);
      startedFor.current = key;
      setJobId(j.id);
    } catch (e) {
      setError(e instanceof ApiError && e.message ? e.message : "The sheet didn't start. Try again.");
    } finally {
      setStarting(false);
    }
  };
  if (job && isActive(job)) {
    const queued = job.status === "queued";
    return (
      <div className="rounded-card border border-line px-3 py-2.5" aria-live="polite">
        <div className="flex items-baseline justify-between gap-3 text-[12px] leading-snug">
          <span className="min-w-0 truncate text-ink2" title="Building the full sheet">
            {queued ? `Queued · ${queueShort(place)}` : job.stage}
          </span>
          <span className="flex shrink-0 items-baseline gap-3">
            <span className="slate text-ink3">{queued ? "Queued" : elapsedClock(elapsed)}</span>
            <CardAction onClick={() => void cancelJob(job.id)}>Stop</CardAction>
          </span>
        </div>
        <ProgressLine value={job.progress} idle={queued} className="mt-2" />
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-card border border-line px-3 py-2.5">
      <p className="flex min-w-[140px] flex-1 items-center gap-1.5 text-[12px] leading-snug text-ink3">
        {error ? (
          `The sheet didn't build: ${error}`
        ) : (
          <>
            Character sheet
            <Hint text="Optional. Front, back and close-up, so they look right from behind. It's added to your cast." />
          </>
        )}
      </p>
      <Button variant="secondary" size="sm" className="shrink-0" onClick={() => void start()} loading={starting}>
        {error ? "Try again" : "Build · ~40 s"}
      </Button>
    </div>
  );
}

function RecastForm({ shot, ui }: FormProps) {
  const t = useTask("recast", shot, ui);
  const sel = useSelection(shot.id);
  const player = usePlayer();
  const pre = usePrefill(shot.id, "recast");
  const [pick] = useState(() => restoreOf(pre));
  const [made] = useState(() => madePerson(pre));
  const ref = useUploadSlot();
  const projectId = shot.project_id || "unsorted";
  const cast = useCast(projectId);
  const castPeople = cast.filter((c) => c.kind === "person");
  const cap = useLengthCap(shot);
  const narrow = useNarrow();
  const [castId, setCastId] = useState<string | null>(str(pre?.cast_id) || null);
  // "Change" on a cast pick: the chips come back with them still picked, until another is picked (or they are again).
  const [choosing, setChoosing] = useState(false);
  const [keptRef] = useState<string | null>(str(pre?.ref_upload_id) || null);
  const keptRefUrl = keptRef ? keptUrl({ kind: "upload", id: keptRef }) : "";
  const fromExample = keptFrom(pre) === "the example";
  const fromPreset = keptFrom(pre) === "the preset";
  const keptPhrase = (example: string, preset: string, last: string) => (fromExample ? example : fromPreset ? preset : last);
  const [whoRef, setWhoRef] = useState<{ ref: RefPick; url: string } | null>(
    pre?.who_ref ? { ref: pre.who_ref as RefPick, url: str(pre?._who_url) || keptUrl(pre.who_ref as RefPick, projectId) } : null,
  );
  const [useKept, setUseKept] = useState(!!keptRef && !pre?.cast_id);
  // A character the take made from words stays until "Change".
  const [keepMade, setKeepMade] = useState(!!made);
  // The words field: who they are until a character is in, then what to change about them. A made character's words
  // are theirs (their row), so the field starts empty.
  const [startText, setStartText] = useState(() => (made ? "" : str(pre?.describe)));
  const [describe, setDescribe] = useState(startText);
  const [still, setStill] = useState<Still | null>(null);
  const stillSeq = useRef(0);
  const [pending, setPending] = useState(false); // Recast pressed before the still existed: it goes when it lands
  const [pickText, setPickText] = useState(pick?.text ?? "");
  // LTX-2.5 unless the take being tried again was made with SCAIL-2 and said so.
  const [engine, setEngine] = useState<RecastEngine>(str(pre?.engine) === "scail" ? "scail" : "ltx");
  const [relight, setRelight] = useState(bool(pre?.relight, false)); // person swap, optional: the shot's light on the new person
  const [takenOver, setTakenOver] = useState(false); // "Not them?", a click or words: the user picks, not the finder
  // Words about who they are, cleared when a picture took their place ("Undo" puts them back, as changes).
  const [removedWords, setRemovedWords] = useState<string | null>(null);
  const describeWrap = useRef<HTMLDivElement>(null);
  const enh = useEnhance({ request: { target: "ltx", shot_id: shot.id, project_id: projectId }, text: describe, onUse: setDescribe });
  const desc = describe.trim();
  // The person in the words, without an instruction's lead-in ("turn him into a knight" → "a knight"): what FLUX 3
  // klein and the video model get.
  const words = personWords(desc);
  const chosen = castId ? castPeople.find((c) => c.id === castId) ?? null : null;

  // --- who to replace ---
  useSelectionSession(shot.id, pick);
  const finder = usePeople(shot.id, t.baseId);
  const autoIdx = useMemo(() => {
    if (finder.s !== "found" || !sel || sel.text || sel.frame !== finder.res.frame) return null;
    const i = finder.res.people.findIndex((p) => samePoints(sel.points, personPoints(p)));
    return i >= 0 ? i : null;
  }, [finder, sel]);
  const choose = (p: PersonFound) => {
    if (finder.s !== "found") return;
    setTakenOver(false);
    setPickText("");
    if (player.getSnapshot().playing) player.pause();
    player.seek(finder.res.frame);
    pickPerson(shot.id, { frame: finder.res.frame, points: personPoints(p), maskUrl: p.mask_url });
  };

  // --- Recast anyone: with several people found and the finder's pick in, anyone else can be swapped too, each to a
  // cast member or words. The main person keeps the form's own character; the rest stay as filmed unless swapped. ---
  const found = finder.s === "found" ? finder.res.people : [];
  const mainP = autoIdx != null ? found[autoIdx] ?? null : null;
  const multi = !!mainP && found.length > 1 && !takenOver;
  const [extras, setExtras] = useState<Array<{ index: number; castId: string | null; describe: string }>>([]);
  const [pickingFor, setPickingFor] = useState<number | null>(null);
  // Left to right, never the main person.
  const others = multi ? found.filter((p) => p.index !== mainP!.index).flatMap((p) => extras.filter((x) => x.index === p.index)) : [];
  const many = others.length > 0;
  // Several people render with LTX-2.5 only.
  const eng: RecastEngine = many ? "ltx" : engine;
  // A click in the picture on someone else the finder found picks their character (instead of switching to them).
  const multiClick = useRef<((p: PersonFound) => boolean) | null>(null);
  multiClick.current = multi
    ? (p) => {
        if (p.index === mainP!.index) return false;
        setPickingFor(p.index);
        return true;
      }
    : null;
  useEffect(() => {
    setSelectionMany(shot.id, multi);
  }, [multi, sel, shot.id]);
  // "Try again with changes": the take's other swaps come back (its main person is the form's own fields).
  const restoredSwaps = useRef(false);
  useEffect(() => {
    if (restoredSwaps.current || finder.s !== "found" || !Array.isArray(pre?.swaps)) return;
    restoredSwaps.current = true;
    const mainPts = Array.isArray(pre.points) ? (pre.points as Point[]) : [];
    const back: typeof extras = [];
    for (const s of pre.swaps as RecastSwap[]) {
      if (s.frame !== finder.res.frame || !s.points?.length || samePoints(s.points, mainPts)) continue;
      const p = finder.res.people.find((q) => samePoints(s.points, personPoints(q)));
      if (p && (s.cast_id || s.describe)) back.push({ index: p.index, castId: s.cast_id ?? null, describe: s.cast_id ? "" : s.describe ?? "" });
    }
    if (back.length) setExtras(back);
  }, [finder]);

  // While the finder looks, the picture's chip says so too ("Finding who's in the shot…").
  useEffect(() => {
    setSelectionFinding(shot.id, finder.s === "finding");
  }, [finder.s, shot.id]);
  // While the finder's pick is untouched, a click in the picture on someone else it found switches to them (with their
  // matte, no find); a click on the same person adds to them; a click anywhere else starts a new pick there.
  useEffect(() => {
    if (finder.s !== "found" || !finder.res.people.length) return;
    const res = finder.res;
    setAutoClick((pt, frame) => {
      const hit = frame === res.frame ? personAt(res.people, pt) : null;
      if (!hit) return "none";
      const cur = getState().selection;
      if (cur && samePoints(cur.points, personPoints(hit))) return "same";
      if (multiClick.current?.(hit)) return "switched";
      choose(hit);
      return "switched";
    });
    return () => setAutoClick(null);
  }, [finder]);
  // The finder's main person is picked for you; a kept pick ("Try again with changes") or anything the user does wins.
  useEffect(() => {
    if (finder.s !== "found" || !sel) return;
    // A kept pick that is someone the finder found: the same pick, now with their matte over the picture.
    if (sel.kept && autoIdx != null && !sel.maskUrl) {
      if (finder.res.people[autoIdx].mask_url) choose(finder.res.people[autoIdx]);
      return;
    }
    if (pick || takenOver) return;
    if (sel.points.length || sel.text || sel.kept) {
      if (autoIdx == null) setTakenOver(true);
      return;
    }
    const main = mainOf(finder.res);
    if (main) choose(main);
  }, [finder, sel, autoIdx, pick, takenOver]);
  // Words that name a vehicle: this build swaps people only, so say so now rather than after the render (the server
  // refuses too, recast_ltx.NO_CARS).
  const carWords = !CARS && !!sel?.text && VEHICLE_WORDS.test(sel.text.trim());
  const hasPerson = (autoIdx != null || picked(sel)) && !carWords;
  const userPick = !!sel && (sel.points.length > 0 || !!sel.text || !!sel.kept);
  // The finder has answered and is about to pick for you.
  const autoPicking = finder.s === "found" && finder.res.people.length > 0 && !takenOver && !pick && !userPick;

  // --- your character ---
  const canMake = !!t.status?.connected && t.status.tiles?.image?.ready !== false;
  const mentioned = (text: string): CastMember | null => scan(text, cast).map((m) => m[2]).find((c) => c.kind === "person") ?? null;
  const mention = mentioned(describe);
  const madeOn = !!made && keepMade;
  const photoWide = useWide(ref.value?.url);
  const keptWide = useWide(useKept ? keptRefUrl : null);
  const stillJob = useApp((s) => (still?.jobId ? s.jobs[still.jobId] : undefined));
  const stillImage = useApp((s) => (still?.imageId ? s.images[projectId]?.find((i) => i.id === still.imageId) : undefined));
  const stillUrl = still?.imageId ? still.url || stillImage?.url || stillImage?.thumb_url || keptUrl({ kind: "image", id: still.imageId }, projectId) : "";
  const stillState: "none" | "making" | "ready" | "error" = !still ? "none" : still.error ? "error" : still.imageId ? "ready" : "making";

  // Who's in: the first that applies, as the submit sends it.
  const source: Who | null = chosen
    ? "cast"
    : whoRef
      ? "image"
      : ref.value
        ? "photo"
        : useKept && keptRef
          ? "kept"
          : madeOn
            ? "made"
            : stillState === "ready"
              ? "still"
              : desc && !still
                ? "words"
                : null;
  // A character is in (or on its way): from here the words say what to change about them.
  const charIn = (!!source && source !== "words") || !!still;
  // The ways in (cast chips, a picture, words) show until a character is in, and again on a cast pick's "Change".
  const chooser = !charIn || (choosing && source === "cast");
  const reads = !charIn && words && words !== desc.replace(/\s+/g, " ").replace(/[.\s]+$/, "") ? words : "";
  // What the video model is told about them (`describe`): who they are (a made character's or a still's words, or the
  // words alone), then what to change.
  const join = (a: string, b: string) => (a && b ? `${a}, ${b}` : a || b);
  const sendDescribe =
    source === "made" && made ? (made.caption ? words : join(personWords(made.text), words)) : source === "still" && still ? join(still.words, words) : words;
  const who =
    source === "cast" && chosen
      ? { name: chosen.name, url: chosen.thumb_url || chosen.url }
      : source === "image" && whoRef
        ? { name: "An image from this project", url: whoRef.url }
        : source === "photo" && ref.value
          ? { name: ref.value.name, url: ref.value.url }
          : source === "kept"
            ? { name: keptPhrase("The example's reference", "The preset's reference", "The last take's reference"), url: keptRefUrl || undefined }
          : source === "made" && made
            ? made.url
              ? { name: keptPhrase("The example's character", "The preset's character", "The last take's character"), url: made.url }
                : { name: "A still made again from the description" }
              : source === "still"
                ? { name: "The still made from your words", url: stillUrl }
                : source === "words" && !mention
                  ? { name: "A still made from your words" }
                  : null;
  // The same character as the take (and the same words): what else it ran with still applies.
  const preWhoId = (pre?.who_ref as RefPick | undefined)?.id ?? "";
  const samePerson =
    !!pre &&
    !!source &&
    describe === startText &&
    (source === "cast"
      ? chosen?.id === str(pre.cast_id)
      : source === "image"
        ? !!pre.who_ref
        : source === "still"
          ? !!preWhoId && still?.imageId === preWhoId
          : source === "words"
            ? !pre.cast_id && !pre.who_ref && !pre.ref_upload_id
            : source !== "photo");
  // A kept prompt is the video model's own: only for the same engine (takes from before the choice were SCAIL-2's).
  // (and only for as many people: one person's prompt doesn't describe several).
  const keptPrompt = samePerson && (str(pre?.engine) || "scail") === eng && many === Array.isArray(pre?.swaps) ? str(pre?.prompt) : "";
  const ready = hasPerson && (!!source || stillState === "making") && !ref.busy;
  // A character set another way (cast, a still, "Undo"): an earlier file's upload error no longer applies.
  useEffect(() => {
    if (source && source !== "words" && ref.error) ref.clear();
  }, [source]);

  // A still the take made from words (a project image Recast used as who_ref) comes back as one: "Try another" re-rolls
  // it, and the take's words beyond the still's own are the changes. Any other picked image stays a picture.
  const preImageId = (pre?.who_ref as RefPick | undefined)?.kind === "image" ? preWhoId : "";
  const preImage = useApp((s) => (preImageId ? s.images[projectId]?.find((i) => i.id === preImageId) : undefined));
  const adopted = useRef(false);
  useEffect(() => {
    if (preImageId) void loadImages(projectId);
  }, [preImageId, projectId]);
  useEffect(() => {
    if (adopted.current || !preImage || whoRef?.ref.id !== preImage.id || !isStillPrompt(preImage.prompt)) return;
    adopted.current = true;
    const sw = personWords(stillWords(preImage.prompt) || desc);
    const rest = sw && desc.toLowerCase().startsWith(sw.toLowerCase()) ? desc.slice(sw.length).replace(/^[\s,;.]+/, "") : "";
    setStill({ words: sw, seed: preImage.seed ?? 0, jobId: null, imageId: preImage.id, url: whoRef.url });
    setWhoRef(null);
    setDescribe(rest);
    setStartText(rest);
  }, [preImage]);

  // Follow the still's image job until it lands (or doesn't).
  useEffect(() => {
    if (!still?.jobId || still.imageId || still.error || !stillJob) return;
    const id = still.jobId;
    if (stillJob.status === "done") {
      const imageId = stillJob.result_image_ids?.[0];
      setStill((s) => (s && s.jobId === id ? (imageId ? { ...s, imageId } : { ...s, error: "No picture came back. Try again." }) : s));
    } else if (stillJob.status === "failed" || stillJob.status === "canceled") {
      const error = stillJob.status === "failed" ? stillJob.error || "FLUX 3 klein couldn't make them. Try again." : "Stopped.";
      setStill((s) => (s && s.jobId === id ? { ...s, error } : s));
    }
  }, [stillJob?.status, still?.jobId]);

  const startStill = async (w: string, seed = newSeed()) => {
    if (still?.jobId && !still.imageId) dropCharacterStill(still.jobId);
    if (ref.error) ref.clear(); // an earlier file's error no longer applies
    const mine = ++stillSeq.current;
    setStill({ words: w, seed, jobId: null });
    void loadImageModels(); // the queue bar names the model ("FLUX 3 klein")
    try {
      const job = await makeCharacterStill(projectId, stillRequest(w, seed));
      if (stillSeq.current === mine) setStill((s) => (s && s.seed === seed ? { ...s, jobId: job.id } : s));
      else dropCharacterStill(job.id);
    } catch (e) {
      if (stillSeq.current === mine) setStill((s) => (s ? { ...s, error: e instanceof ApiError && e.message ? e.message : "Couldn't start making them. Try again." } : s));
    }
  };
  /** Words with no character yet make one: on Enter, from "Make from words", or on Recast. The words go with the still
   * (its row shows them) and the field is then for changes. */
  const wantsStill = (w: string) => !!w && !charIn && canMake && !mentioned(w);
  const commitWords = (text: string) => {
    const w = personWords(text);
    if (!wantsStill(w)) return;
    void startStill(w);
    setDescribe("");
  };

  // The part of a long shot LTX-2.5 renders: a bracket on the timeline. Until it's moved, centred on the person.
  const anchor = hasPerson && sel ? sel.frame : finder.s === "found" ? finder.res.frame : finder.s === "failed" ? 0 : null;
  const bracket = useShotWindow(shot, typeof pre?.start_frame === "number" ? Math.max(0, Math.round(pre.start_frame)) : null, anchor, eng === "scail", true, eng === "ltx");
  const showWindow = bracket.show;
  const win = bracket.win;

  const doSubmit = () => {
    if (!sel || !source || !hasPerson) return;
    // Who comes in: the first that applies, as the form shows it.
    const whoIn: RecastWho =
      source === "cast" && chosen
        ? { cast_id: chosen.id }
        : source === "image" && whoRef
          ? { who_ref: whoRef.ref }
          : source === "photo" && ref.value
            ? { ref_upload_id: ref.value.id }
            : source === "kept" && keptRef
              ? { ref_upload_id: keptRef }
              : source === "made" && made?.take
                ? { reference_take: made.take }
                : source === "still" && still?.imageId
                  ? { who_ref: { kind: "image", id: still.imageId } }
                  : {};
    // A made character's caption is FLUX 3 klein's own: it makes them again from it (reference_prompt).
    if (source === "made" && made?.caption) whoIn.reference_prompt = made.text;
    const params = recastParams({ points: sel.points, text: sel.text, frame: sel.frame }, eng, whoIn, {
      describe: sendDescribe,
      relight: RELIGHT && eng === "ltx" && !many && relight,
      kept: samePerson && pre ? { prompt: keptPrompt, scene: str(pre.scene), seed: typeof pre.seed === "number" ? pre.seed : null } : undefined,
      win: showWindow ? win : null,
    });
    // Recast anyone: the main person first (the fields above, kept for them too), then the others left to right. One
    // person sends no `swaps`.
    if (many && mainP && autoIdx != null && finder.s === "found") {
      const frame = finder.res.frame;
      const first: RecastSwap = { frame, points: sel.points, kind: "person", label: whoLabel(found, autoIdx) };
      if (params.cast_id) first.cast_id = params.cast_id;
      else if (params.who_ref) first.who_ref = params.who_ref;
      else if (params.ref_upload_id) first.ref_upload_id = params.ref_upload_id;
      const words = params.describe || params.reference_prompt;
      if (words) first.describe = words;
      params.swaps = [
        first,
        ...others.map((x): RecastSwap => {
          const i = found.findIndex((p) => p.index === x.index);
          return {
            frame,
            points: personPoints(found[i]),
            kind: "person",
            label: whoLabel(found, i),
            ...(x.castId ? { cast_id: x.castId } : { describe: x.describe }),
          };
        }),
      ];
    }
    void t.submit(params);
  };
  const submit = () => {
    if (stillState === "making") {
      setPending(true); // it goes as soon as the still lands
      return;
    }
    if (source === "words" && wantsStill(words)) {
      // Make the still they'd see, then render with it.
      void startStill(words);
      setDescribe("");
      setPending(true);
      return;
    }
    doSubmit();
  };

  // Recast pressed while the still was on its way: it goes as soon as the still exists, if there's still someone to
  // replace (a pick being found is waited for; "Not them?" or Clear calls it off, and so does another character).
  useEffect(() => {
    if (!pending) return;
    if (!hasPerson && !sel?.loading) setPending(false);
    else if (source === "still") {
      if (!hasPerson) return;
      setPending(false);
      if (t.ready) doSubmit();
    } else if (source || !still || still.error) setPending(false);
  }, [pending, source, still, hasPerson, sel?.loading, t.ready]);

  // --- picking a character ---
  const dropStill = () => {
    if (still?.jobId && !still.imageId) dropCharacterStill(still.jobId);
    stillSeq.current += 1; // a still still starting for these words no longer counts
    setStill(null);
  };
  /** Another character goes in: every other way in is let go (a still on its way is stopped). */
  const clearPicks = () => {
    setCastId(null);
    setChoosing(false);
    setWhoRef(null);
    setUseKept(false);
    setKeepMade(false);
    setPending(false);
    if (still) dropStill();
  };
  /** A picture in place of words about who they are: the words go ("Undo" puts them back, as changes). */
  const wordsForPicture = () => {
    if (charIn || !desc) return;
    setRemovedWords(describe);
    setDescribe("");
  };
  /** "Change" on a picture: it goes, and so do the changes written for it. */
  const unpick = (drop: () => void) => {
    drop();
    setDescribe("");
    setRemovedWords(null);
  };
  const focusDescribe = () => requestAnimationFrame(() => describeWrap.current?.querySelector<HTMLElement>("input, textarea")?.focus());
  const onPhoto = (f: File) => {
    wordsForPicture();
    clearPicks();
    ref.onFile(f);
  };
  const pickCast = (c: CastMember) => {
    if (c.id === castId) {
      setChoosing(false); // the same one again: keep them
      return;
    }
    wordsForPicture();
    clearPicks();
    ref.clear();
    setCastId(c.id);
  };
  const makeOne = () => {
    const was = charIn;
    clearPicks();
    ref.clear();
    if (was) setDescribe(""); // changes for the last character don't describe a new one
    else if (wantsStill(words)) {
      void startStill(words);
      setDescribe("");
    }
    focusDescribe();
  };
  const changeStill = () => {
    if (!still) return;
    const w = still.words;
    dropStill();
    setPending(false);
    setDescribe(w); // their words, to edit
    focusDescribe();
  };
  const changeMade = () => {
    setKeepMade(false);
    setDescribe(made && !made.caption ? personWords(made.text) : "");
  };
  const backToMade = () => {
    wordsForPicture();
    clearPicks();
    ref.clear();
    setKeepMade(true);
  };
  const onDescribe = (v: string) => {
    setRemovedWords(null);
    // One line: Enter (or a pasted line break) means "done", and makes their still.
    if (v.includes("\n")) {
      const clean = v.replace(/\s*\n+\s*/g, " ").trimEnd();
      setDescribe(clean);
      commitWords(clean);
      return;
    }
    setDescribe(v);
  };
  const runPickText = (text: string) => {
    setTakenOver(true);
    selectByText(text, player.getSnapshot().frame);
  };

  // --- the "With" row, once a character is in (with several people, the main person's row says the same) ---
  const changeButton = (onClick: () => void) => <RowButton onClick={onClick}>Change</RowButton>;
  type WithInfo = { thumb?: string | null; icon?: LucideIcon; busy?: boolean; contain?: boolean; label: string; sub?: ReactNode; tag?: string; quiet?: boolean; actions: ReactNode };
  let withInfo: WithInfo | null = null;
  if (source === "cast" && chosen) {
    withInfo = { thumb: chosen.thumb_url || chosen.url, label: chosen.name, tag: chosen.sheet ? "Sheet" : undefined, actions: changeButton(() => setChoosing(true)) };
  } else if (source === "image" && whoRef) {
    withInfo = {
      thumb: whoRef.url,
      label: whoRef.ref.kind === "cast" ? "someone from your cast" : "an image from this project",
      actions: changeButton(() => unpick(() => setWhoRef(null))),
    };
  } else if (source === "photo" && ref.value) {
    withInfo = { thumb: ref.value.url, contain: true, label: ref.value.name, tag: photoWide ? "Sheet" : undefined, actions: changeButton(() => unpick(ref.clear)) };
  } else if (source === "kept") {
    withInfo = {
      thumb: keptRefUrl,
      contain: true,
      label: keptPhrase("the example's picture", "the preset's picture", "the same picture as last time"),
      tag: keptWide ? "Sheet" : undefined,
      actions: changeButton(() => unpick(() => setUseKept(false))),
    };
  } else if (source === "made" && made) {
    withInfo = {
      thumb: made.url || null,
      icon: UserRound,
      label: made.url
        ? keptPhrase("the example's character", "the preset's character", "the same character as last time")
        : keptPhrase("the example's description", "the preset's description", "the same description as last time"),
      sub: made.url ? undefined : fromPreset ? "From the preset" : fromExample ? "From the example" : "Remade from its words",
      actions: changeButton(changeMade),
    };
  } else if (still) {
    withInfo =
      stillState === "error"
        ? {
            quiet: true,
            icon: UserRound,
            label: still.words,
            sub: `Couldn't make them. ${still.error}`,
            actions: (
              <>
                <RowButton onClick={() => void startStill(still.words)}>Try again</RowButton>
                {changeButton(changeStill)}
              </>
            ),
          }
        : stillState === "making"
          ? { busy: true, label: still.words, sub: "Making · ~10 s", actions: changeButton(changeStill) }
          : {
              thumb: stillUrl,
              label: still.words,
              sub: <CardAction onClick={() => void startStill(still.words)}>Try another</CardAction>,
              actions: changeButton(changeStill),
            };
  }
  let withRow: ReactNode = null;
  if (withInfo) {
    const { actions, ...row } = withInfo;
    withRow = (
      <SentenceRow lead="With" {...row}>
        {actions}
      </SentenceRow>
    );
  }

  // --- Recast anyone: one row per person found ---
  const notThem = () => {
    setTakenOver(true);
    setPending(false);
    setExtras([]);
    setPickingFor(null);
    clearSelection();
    if (narrow) showPicture();
  };
  const swapTo = (index: number, who: { castId: string | null; describe: string }) => {
    setExtras((xs) => [...xs.filter((x) => x.index !== index), { index, ...who }]);
    setPickingFor(null);
  };
  const unswap = (index: number) => {
    setExtras((xs) => xs.filter((x) => x.index !== index));
    if (pickingFor === index) setPickingFor(null);
  };
  /** The main person back to as filmed: the next one swapped takes their place (and the form's character is theirs). */
  const dropMain = () => {
    const next = others[0];
    const p = next ? found.find((q) => q.index === next.index) : null;
    if (!next || !p) return;
    setExtras((xs) => xs.filter((x) => x.index !== next.index));
    setPickingFor(null);
    clearPicks();
    ref.clear();
    setRemovedWords(null);
    setCastId(next.castId);
    setDescribe(next.castId ? "" : next.describe);
    choose(p);
  };
  const peopleRows: ReactNode = multi ? (
    <section className="space-y-2" aria-label="Who to replace">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[13px] text-ink2">
          {many ? `Replacing ${others.length + 1} of ${found.length}` : "In the shot"}
          <Hint text={`Replace anyone here, each with their own character. Anyone you leave stays as filmed. ${narrow ? "Tap" : "Click"} someone in the picture to replace them too.`} />
        </span>
        <RowButton onClick={notThem} title="Click or describe who to replace instead">
          Pick by hand
        </RowButton>
      </div>
      {found.map((p, i) => {
        const where = whoLabel(found, i);
        if (p.index === mainP!.index) {
          const drop = many ? (
            <button type="button" onClick={dropMain} aria-label={`Keep ${where} as filmed`} title="Keep as filmed" className="rounded-full p-1.5 text-ink2 transition-colors hover:bg-line hover:text-ivory">
              <X size={15} />
            </button>
          ) : null;
          if (!withInfo) {
            return (
              <PersonRow key={p.index} person={p.thumb_url} where={where} swapped active label="Pick a character below">
                {drop}
              </PersonRow>
            );
          }
          const { actions, quiet: _quiet, tag: _tag, ...row } = withInfo;
          return (
            <PersonRow key={p.index} person={p.thumb_url} where={where} swapped active={chooser} {...row}>
              {actions}
              {drop}
            </PersonRow>
          );
        }
        const mine = others.find((x) => x.index === p.index);
        const member = mine?.castId ? castPeople.find((c) => c.id === mine.castId) : null;
        const picking = pickingFor === p.index;
        return (
          <div key={p.index} className="space-y-2">
            {mine ? (
              <PersonRow
                person={p.thumb_url}
                where={where}
                swapped
                active={picking}
                thumb={member ? member.thumb_url || member.url : null}
                label={member ? member.name : mine.castId ? "Someone from your cast" : mine.describe}
              >
                {!picking ? <RowButton onClick={() => setPickingFor(p.index)}>Change</RowButton> : null}
                <button type="button" onClick={() => unswap(p.index)} aria-label={`Keep ${where} as filmed`} title="Keep as filmed" className="rounded-full p-1.5 text-ink2 transition-colors hover:bg-line hover:text-ivory">
                  <X size={15} />
                </button>
              </PersonRow>
            ) : (
              <PersonRow person={p.thumb_url} where={where} swapped={false} active={picking} label="As filmed">
                {!picking ? <RowButton onClick={() => setPickingFor(p.index)}>Replace</RowButton> : null}
              </PersonRow>
            )}
            {picking ? (
              <SwapPicker
                key={p.index}
                cast={castPeople}
                castId={mine?.castId ?? null}
                describe={mine?.describe ?? ""}
                onPick={(who) => swapTo(p.index, who)}
                onCancel={() => setPickingFor(null)}
              />
            ) : null}
          </div>
        );
      })}
    </section>
  ) : null;

  // "Build a full sheet": offered under one photo (not a sheet already) or a still made from words.
  const sheetFrom: SheetRequest | null = !canMake
    ? null
    : source === "photo" && ref.value && !photoWide
      ? { upload_id: ref.value.id }
      : source === "still" && still?.imageId
        ? { image_id: still.imageId }
        : null;
  const sheetCast = sheetFrom ? castNameFrom(source === "still" && still ? still.words : "", ref.value?.name ?? "", cast) : "";

  // One line under the words, the first that applies (before a character is in).
  const readsLine = reads ? `Reads as “${reads}”.` : "";
  const describeHint = charIn ? null : [readsLine, wantsStill(words) ? "Press Enter to preview." : ""].filter(Boolean).join(" ") || null;
  const longField = cast.length > 0;

  // What Recast still needs, said where the button is (the time once it can go).
  const needs = pending
    ? "Making your character…"
    : !hasPerson
      ? sel?.loading
        ? "Finding it…"
        : (finder.s === "finding" && !userPick) || autoPicking
          ? "Finding who's in the shot…"
          : carWords
        ? "Pick a person. This version replaces people only."
        : "First, click the person to replace in the picture."
      : ref.busy
        ? "Uploading…"
        : stillState === "making"
          ? "Making your character…"
          : !source
            ? "Pick a character above."
            : null;
  // What the video model gets, as tasks/recast.py and recast_ltx.py write it (a cast pick with no words is named by
  // their mention; LTX's sentence about what they do comes from the shot itself).
  const receiptWho = sendDescribe || (source === "cast" && chosen ? `@${chosen.name}` : "");
  const receiptScene = samePerson ? str(pre?.scene) : "";
  const receiptText = keptPrompt || (eng === "ltx" ? ltxRecastPrompt(receiptWho, receiptScene) : scailPrompt(receiptWho, receiptScene));
  const etaLine = engineEta(t, eng);

  return (
    <FormShell
      task="recast"
      shot={shot}
      ui={ui}
      lengthNote={false}
      footer={
        <>
          <PrimaryAction hint={needs ?? etaLine} disabled={!ready || !t.ready} loading={t.submitting || pending} onClick={submit}>
            Replace a person
          </PrimaryAction>
          {cap.over ? (
            <p className="text-[13px] leading-snug text-ink3">
              {eng === "scail"
                ? `Uses the first ${cap.label}.`
                : win
                  ? `Replaces the ${bracket.label} in the bracket. Drag to move it, or its ends to set the length. ${bracket.maxLabel}.`
                  : `Uses the ${cap.label} where they're on screen most.`}
            </p>
          ) : null}
        </>
      }
    >
      {/* One sentence: "Replacing [them] · With [your character] · Anything to change?" */}
      <div className="space-y-2.5">
        {multi ? (
          peopleRows
        ) : (
          <WhoToReplace
            shot={shot}
            finder={finder}
            autoIdx={autoIdx}
            takenOver={takenOver}
            kept={pick}
            narrow={narrow}
            onChoose={choose}
            onNotThem={notThem}
            onClear={() => {
              setTakenOver(true);
              setPending(false);
              setPickText("");
            }}
          />
        )}
        {hasPerson ? <TrackCheck shot={shot} versionId={t.baseId} startFrame={eng === "ltx" && showWindow && win ? win.start : undefined} /> : null}
        {!chooser && !multi ? withRow : null}
        <SheetOffer
          projectId={projectId}
          from={sheetFrom}
          name={sheetCast}
          onBuilt={(id) => {
            clearPicks();
            ref.clear();
            setCastId(id);
          }}
        />
      </div>

      {chooser ? (
        <section className="space-y-3 border-t border-line pt-5" aria-label="Your character">
          <h3 className="flex items-center gap-1.5 text-[16px] font-medium text-ivory">
            Character
            <Hint text={`${SLOT_COPY} Or pick someone from your cast, or describe them.`} />
          </h3>
          {castPeople.length ? <CastChips cast={castPeople} castId={castId} onPick={pickCast} /> : null}
          <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-2">
            <DropSlot compact value={null} busy={ref.busy} progress={ref.progress} error={ref.error} onFile={onPhoto} onClear={ref.clear} title="Drop a photo or sheet" />
            <button
              type="button"
              onClick={makeOne}
              className="flex h-16 flex-col items-center justify-center gap-1 rounded-card border border-line px-4 text-ink2 transition-colors duration-150 hover:border-line2 hover:text-ink"
            >
              <Sparkles size={15} strokeWidth={1.5} aria-hidden="true" />
              <span className="whitespace-nowrap text-[13px]">Make from words</span>
            </button>
          </div>

          {made && !keepMade && !charIn ? (
            <button type="button" onClick={backToMade} className="text-[12px] text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline">
              {keptPhrase("Use the example's character", "Use the preset's character", "Use the same character as last time")}
            </button>
          ) : null}
        </section>
      ) : null}

      <section className={cx("space-y-2", charIn ? "border-t border-line pt-5" : chooser && "!mt-3")} aria-label={charIn ? "Anything to change" : "Who they are"}>
        {charIn ? (
          <div className="text-[13px] text-ink2">
            {many && source === "cast" && chosen ? `Changes to ${chosen.name}` : "Changes"} <span className="text-ink3">(optional)</span>
          </div>
        ) : null}
        {/* Not on leaving the field: the ways in fold away once a still starts, and a click on a cast chip would miss. */}
        <div ref={describeWrap}>
          {longField ? (
            <MentionField
              label={charIn ? "Anything to change?" : "Who are they?"}
              value={describe}
              onChange={onDescribe}
              rows={charIn ? 1 : 2}
              cast={cast}
              projectId={projectId}
              placeholder={charIn ? CHANGE_PLACEHOLDER : WHO_PLACEHOLDER}
            />
          ) : (
            <textarea
              className="field h-auto resize-none py-2.5 leading-[22px]"
              rows={charIn ? 1 : 2}
              value={describe}
              placeholder={charIn ? CHANGE_PLACEHOLDER : WHO_PLACEHOLDER}
              aria-label={charIn ? "Anything to change?" : "Who are they?"}
              spellCheck
              onChange={(e) => onDescribe(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitWords(describe);
                }
              }}
            />
          )}
        </div>
        {removedWords != null && !desc ? (
          <p className="flex flex-wrap items-baseline gap-x-2 text-[12px] leading-snug text-ink3" aria-live="polite">
            Cleared your description.
            <CardAction
              onClick={() => {
                setDescribe(removedWords);
                setRemovedWords(null);
              }}
            >
              Undo
            </CardAction>
          </p>
        ) : describeHint ? (
          <p className="text-[12px] leading-snug text-ink3">{describeHint}</p>
        ) : null}
      </section>

      <MoreOptions>
        <div>
          <div id="recast-engine" className="mb-1.5 text-[13px] text-ink2">
            Engine
          </div>
          <div className="space-y-2" role="radiogroup" aria-labelledby="recast-engine">
            <RadioRow on={engine === "ltx"} onClick={() => setEngine("ltx")} title="LTX-2.5" badge={<Tag>Default</Tag>} sub="The whole costume, lit by your shot." />
            <RadioRow on={engine === "scail"} onClick={() => setEngine("scail")} title="SCAIL-2" sub="Follows the exact performance. Best for close-ups and turns." />
          </div>
          {many && engine === "scail" ? <p className="mt-1.5 text-[12px] leading-snug text-ink3">More than one person renders with LTX-2.5.</p> : null}
        </div>
        {RELIGHT && eng === "ltx" && !many ? (
          <SwitchRow checked={relight} onChange={setRelight} title="Relight the new person" hint="Match the new person's light to the shot. Not Relight any scene, which relights the whole shot." />
        ) : null}
        <div>
          <div className="mb-1.5 text-[13px] text-ink2">Who to replace, in words</div>
          <input
            className="field"
            value={pickText}
            placeholder="the person on the left"
            aria-label="Who to replace, in words"
            onChange={(e) => setPickText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                runPickText(pickText);
              }
            }}
            onBlur={() => {
              if (pickText.trim() && pickText.trim() !== sel?.text) runPickText(pickText);
            }}
          />
          <p className="mt-1.5 text-[12px] leading-snug text-ink3">
            Once you've picked someone, clicks in the picture add to them. {ALT_KEY}-click to keep an area as it is.
          </p>
        </div>
        {enh.button ? (
          <div>
            <div className="mb-1.5 flex items-center justify-between gap-3">
              <span className="text-[13px] text-ink2">Rewrite for the model</span>
              {enh.button}
            </div>
            {enh.panel}
          </div>
        ) : null}
        <Fold label="Prompt">
          <PromptReceipt projectId={projectId} text={receiptText} models={[eng]} reference={who} />
        </Fold>
      </MoreOptions>
    </FormShell>
  );
}

// --- Reshoot -------------------------------------------------------------------------------------------------------

const MOVES: Array<{ id: ReshootMove; label: string; icon: LucideIcon }> = [
  { id: "push_in", label: "Push in", icon: Shrink },
  { id: "pull_out", label: "Pull out", icon: Expand },
  { id: "orbit_left", label: "Orbit left", icon: RotateCcw },
  { id: "orbit_right", label: "Orbit right", icon: RotateCw },
  { id: "crane_up", label: "Crane up", icon: ArrowUp },
];
const AMOUNTS: Amount[] = ["small", "medium", "large"];
const AMOUNT_LABEL: Record<Amount, string> = { small: "Small", medium: "Medium", large: "Large" };

function ReshootForm({ shot, ui }: FormProps) {
  const t = useTask("reshoot", shot, ui);
  const pre = usePrefill(shot.id, "reshoot");
  const [move, setMove] = useState<ReshootMove>((str(pre?.move) as ReshootMove) || "push_in");
  const [amount, setAmount] = useState<Amount>((str(pre?.amount) as Amount) || "medium");
  const ai = AMOUNTS.indexOf(amount);
  return (
    <FormShell
      task="reshoot"
      shot={shot}
      ui={ui}
      footer={
        <PrimaryAction hint={t.etaLine} disabled={!t.ready} loading={t.submitting} onClick={() => void t.submit({ move, amount })}>
          Reshoot
        </PrimaryAction>
      }
    >
      <section className="space-y-3">
        <h3 className="flex items-center gap-1.5 text-[16px] font-medium text-ivory">
          Camera move
          <Hint text="Works best on locked-off shots." />
        </h3>
        <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Camera move">
          {MOVES.map((m) => {
            const on = m.id === move;
            return (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => setMove(m.id)}
                className={cx(
                  "flex h-[88px] flex-col items-center justify-center gap-2.5 rounded-card border text-[13px] font-medium transition-[border-color,background-color,color] duration-150 ease-out",
                  on ? "border-ivory/60 bg-raised text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
                )}
              >
                <span className={cx("flex h-8 w-8 items-center justify-center", on ? "text-ivory" : "text-ink3")}>
                  <span className={cx("inline-flex", on && `anim-${m.id}`)}>
                    <Icon icon={m.icon} size={22} />
                  </span>
                </span>
                {m.label}
              </button>
            );
          })}
        </div>
      </section>
      <section className="space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-[14px] text-ink">Amount</span>
          <span className="slate text-ink2">{AMOUNT_LABEL[amount]}</span>
        </div>
        <input
          type="range"
          min={0}
          max={2}
          step={1}
          value={ai}
          onChange={(e) => setAmount(AMOUNTS[Number(e.target.value)])}
          className="range"
          style={{ ["--fill" as string]: `${ai * 50}%` }}
          aria-label="Amount"
          aria-valuetext={AMOUNT_LABEL[amount]}
        />
        <div className="flex justify-between text-[12px] text-ink3">
          {AMOUNTS.map((a) => (
            <button key={a} type="button" onClick={() => setAmount(a)} className={cx("rounded-[4px] px-0.5 transition-colors hover:text-ink", a === amount && "text-ink2")}>
              {AMOUNT_LABEL[a]}
            </button>
          ))}
        </div>
      </section>

    </FormShell>
  );
}

// --- step flows (StepFlow.tsx) -------------------------------------------------------------------------------------------
// Remove, Relight and Finish are only these (their long forms are in git at 4fee8b6); Replace a person's long form above
// comes back with pickups.flow.recast = "0" (lib/drafts.ts flowOn). The same jobs for the same picks (lib/flowParams.ts),
// laid out as steps: one choice per step, with what pros want under each step's Advanced.

type RemoveDraft = {
  shadow: boolean;
  /** The pick, kept so a switched shot or task comes back to it ("Same as before"). */
  pick: Pick | null;
  /** What the Who row says once done: "Person 1", and their cutout. */
  who: string;
  thumb: string | null;
  /** From a take's settings ("Try again with changes"): its fill words, while the pick is still its own, and its seed. */
  kept: { pick: Pick | null; fill: string; seed: number | null };
};

function RemoveFlow({ shot, ui }: FormProps) {
  const t = useTask("remove", shot, ui);
  const sel = useSelection(shot.id);
  const pre = usePrefill(shot.id, "remove");
  const dr = useDraft<RemoveDraft>(
    shot.id,
    "remove",
    () => {
      const pick = restoreOf(pre);
      return { shadow: bool(pre?.shadow, true), pick, who: "", thumb: null, kept: { pick, fill: str(pre?.fill), seed: typeof pre?.seed === "number" ? pre.seed : null } };
    },
    { fresh: !!pre, step: "who" },
  );
  const onWho = dr.draft.step === "who" || !dr.draft.step;
  const player = usePlayer();
  // Click-to-select only while choosing who: Render and Review leave the picture alone.
  const [restore] = useState(() => dr.data.pick);
  useEffect(() => {
    if (!onWho) return;
    startSelection(shot.id);
    const cur = getState().selection;
    if (restore && !cur?.points.length && !cur?.text) {
      player.seek(restore.frame);
      restoreSelection(shot.id, restore, restore.from);
    }
    return () => endSelection();
  }, [shot.id, onWho]);
  const finder = usePeople(shot.id, t.baseId);
  const people = finder.s === "found" ? finder.res.people : [];
  const chosen = sel && finder.s === "found" && sel.frame === finder.res.frame ? people.findIndex((p) => samePoints(sel.points, personPoints(p))) : -1;
  // The person most on screen is picked for you, once, when nothing is picked yet.
  const autoDone = useRef(false);
  useEffect(() => {
    if (!onWho || autoDone.current || finder.s !== "found") return;
    autoDone.current = true;
    const s = getState().selection;
    if (s && s.shotId === shot.id && (s.points.length || s.text || s.kept)) return;
    const main = mainOf(finder.res);
    if (main) pickPerson(shot.id, { frame: finder.res.frame, points: personPoints(main), maskUrl: main.mask_url });
  }, [finder.s, onWho]);
  // What's picked, kept in the draft (and how the Who row will say it).
  useEffect(() => {
    if (!onWho || !sel || sel.loading) return;
    const has = sel.points.length > 0 || !!sel.text;
    const who = chosen >= 0 ? `Person ${chosen + 1}` : sel.text ? `“${sel.text}”` : has ? "Your pick" : "";
    dr.set({ pick: has ? { points: sel.points, frame: sel.frame, text: sel.text, from: "before" } : null, who, thumb: chosen >= 0 ? people[chosen].thumb_url : null });
  }, [sel?.seq, sel?.loading, chosen, onWho]);
  const anchor = sel && (sel.points.length || sel.text) ? sel.frame : null;
  const bracket = useShotWindow(shot, typeof pre?.start_frame === "number" ? Math.max(0, Math.round(pre.start_frame)) : null, anchor, false, false, true);

  const params = (d: RemoveDraft): JobParams => {
    const pick = sel && (sel.points.length || sel.text) ? { points: sel.points, text: sel.text, frame: sel.frame } : d.pick;
    if (!pick) return {};
    const k = d.kept.pick;
    const same = !!k && pick.frame === k.frame && pick.text === k.text && JSON.stringify(pick.points) === JSON.stringify(k.points);
    return removeParams(pick, d.shadow, bracket.show && bracket.win ? bracket.win : null, { fill: same ? d.kept.fill || undefined : undefined, seed: d.kept.seed ?? undefined });
  };

  const steps: FlowStep<RemoveDraft>[] = [
    {
      id: "who",
      label: "Who",
      summary: (d) => (d.who ? { thumb: d.thumb, text: `${d.who}${d.shadow ? " · shadow" : ""}`, edit: "Edit" } : null),
      body: (d, set) => (
        <>
          {finder.s === "finding" ? <BusyRow text="Finding who's in the shot…" /> : null}
          {people.length ? (
            <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="People in the shot">
              {people.map((p, i) => (
                <button
                  key={`${p.index}:${p.thumb_url}`}
                  type="button"
                  role="radio"
                  aria-checked={i === chosen}
                  aria-label={`Person ${i + 1}`}
                  onClick={() => finder.s === "found" && pickPerson(shot.id, { frame: finder.res.frame, points: personPoints(p), maskUrl: p.mask_url })}
                  className={cx(
                    "relative h-[84px] w-[64px] overflow-hidden rounded-[4px] border bg-black transition-[border-color,opacity,box-shadow] duration-150",
                    i === chosen ? "border-ivory shadow-[0_0_0_1px_#EEE9E0]" : "border-line opacity-70 hover:border-line2 hover:opacity-100",
                  )}
                >
                  <img src={p.thumb_url} alt="" className="thumb-face h-full w-full" />
                  {i === chosen ? (
                    <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]" aria-hidden="true">
                      <Check size={10} strokeWidth={3} />
                    </span>
                  ) : null}
                </button>
              ))}
            </div>
          ) : null}
          {/* What to click is said once, over the picture; the rail only shows a pick that isn't one of the people. */}
          {sel && chosen < 0 && (sel.points.length || sel.text || sel.kept || sel.loading || sel.error) ? <SelectionStatus shot={shot} idle="" /> : null}
          <SwitchRow checked={d.shadow} onChange={(v) => set({ shadow: v })} title="Include shadow" />
          {picked(sel) ? <TrackCheck shot={shot} versionId={t.baseId} /> : null}
        </>
      ),
      // Pros: add or take away clicks on the picture (Alt-click keeps an area), or say it in words.
      advanced: () => <PickInWords shot={shot} />,
      primary: () => ({ label: chosen >= 0 ? `Remove Person ${chosen + 1}` : "Remove a person", disabled: !picked(sel) }),
    },
  ];
  return <StepFlow task="remove" shot={shot} ui={ui} draft={dr} steps={steps} params={params} lengthNote={!bracket.show} />;
}

/** Remove's Advanced: the clicks so far (Undo, Clear), and the pick in words. It never starts or ends the selection. */
function PickInWords({ shot, placeholder = "or type it: person, boom mic, sign…", label = "Describe what to select" }: { shot: Shot; placeholder?: string; label?: string }) {
  const sel = useSelection(shot.id);
  const player = usePlayer();
  const [text, setText] = useState(sel?.text ?? "");
  const run = () => {
    if (text.trim() !== (sel?.text ?? "")) selectByText(text, player.getSnapshot().frame);
  };
  return (
    <div className="space-y-3">
      {sel && (sel.points.length || sel.text || sel.kept || sel.loading || sel.error) ? <SelectionStatus shot={shot} idle="" onClear={() => setText("")} /> : null}
      <input
        className="field"
        value={text}
        placeholder={placeholder}
        aria-label={label}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            run();
          }
        }}
        onBlur={run}
      />
    </div>
  );
}

type RelightDraft = { look: string; custom: string; direction: RelightFrom };

function RelightFlow({ shot, ui }: FormProps) {
  const pre = usePrefill(shot.id, "relight");
  const dr = useDraft<RelightDraft>(
    shot.id,
    "relight",
    () => ({
      look: str(pre?.look) || RELIGHT_DEFAULT.look,
      custom: str(pre?.custom),
      direction: (RELIGHT_FROM.find((d) => d.value === pre?.direction)?.value ?? RELIGHT_DEFAULT.direction) as RelightFrom,
    }),
    { fresh: !!pre, step: "light" },
  );
  const lookLabel = (d: RelightDraft) =>
    d.look === "custom" ? `${d.custom.trim() || "Custom"} · ${RELIGHT_FROM.find((x) => x.value === d.direction)?.label ?? ""}` : (RELIGHT_LOOKS.find((l) => l.id === d.look)?.label ?? d.look);
  // Relight peaks around 25.5 GB with the GPU's memory freed first: said only on a card below 24 GB (it reports ~25.8).
  const vram = useApp((s) => s.status?.machine?.vram_gb ?? null);
  const small = vram != null && vram > 0 && vram < 23;
  const relightEta = useApp((s) => s.status?.tiles?.relight?.eta_s ?? null);
  const steps: FlowStep<RelightDraft>[] = [
    {
      id: "light",
      label: "Light",
      summary: (d) => ({ text: lookLabel(d), edit: "Change" }),
      body: (d, set) => (
        <>
          {small ? <Notice stack tone="warn" text={`Relight needs a 24 GB card; this one has ${Math.round(vram)} GB.`} /> : null}
          <RunFacts models={RELIGHT_MODEL} eta={relightEta} />
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Light">
            {RELIGHT_LOOKS.map((l) => (
              <Chip key={l.id} active={d.look === l.id} onClick={() => set({ look: l.id })}>
                {l.label}
              </Chip>
            ))}
            <Chip active={d.look === "custom"} onClick={() => set({ look: "custom" })}>
              Custom
            </Chip>
          </div>
          {d.look === "custom" ? (
            <div className="space-y-3 animate-fade-in">
              <input className="field" value={d.custom} autoFocus onChange={(e) => set({ custom: e.target.value })} placeholder="soft warm evening light" aria-label="Describe the light" />
              <div className="space-y-2">
                <span className="block text-[13px] text-ink2">Light from</span>
                <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Light from">
                  {RELIGHT_FROM.map((x) => (
                    <Chip key={x.value} active={d.direction === x.value} onClick={() => set({ direction: x.value })}>
                      {x.label}
                    </Chip>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </>
      ),
      primary: (d) => ({ label: "Relight", disabled: d.look === "custom" && !d.custom.trim() }),
    },
  ];
  return <StepFlow task="relight" shot={shot} ui={ui} draft={dr} steps={steps} params={(d) => relightParams(d.look, d.custom, d.direction)} />;
}

/** Before a long run: which models it uses and about how long it takes on this GPU. One line. */
function RunFacts({ models, eta }: { models: string; eta: number | null | undefined }) {
  const gpu = useApp((s) => (s.status?.machine ? shortGpu(s.status.machine.gpu) : null));
  const about = aboutTime(eta);
  return (
    <p className="text-[13px] leading-snug text-ink2" data-testid="run-facts">
      <span className="text-ink">{models}</span>
      {about ? <span className="text-ink3">{` · ${about.toLowerCase()}${gpu ? ` on your ${gpu}` : ""}`}</span> : null}
    </p>
  );
}

type FinishDraft = { choice: FinishChoice; quality: "fast" | "best" };

function FinishFlow({ shot, ui }: FormProps) {
  const t = useTask("finish", shot, ui);
  const pre = usePrefill(shot.id, "finish");
  const long = Math.max(shot.width || 0, shot.height || 0);
  const dr = useDraft<FinishDraft>(shot.id, "finish", () => ({ choice: choiceOf(pre, long), quality: pre?.quality === "fast" ? "fast" : "best" }), {
    fresh: !!pre,
    step: "finish",
  });
  const choices = finishChoices(long);
  const pickOf = (d: FinishDraft) => (choices.some((c) => c.value === d.choice) ? d.choice : choices[0].value);
  // Each choice has its own files: 4K alone runs with only SeedVR2 installed (as the form).
  const own = t.status?.tiles?.finish?.choices?.[pickOf(dr.data)];
  const need = own ? { ready: own.ready, missing: { count: own.missing.length, gb: own.missing.reduce((s, m) => s + (Number(m.gb) || 0), 0) } } : undefined;
  const qualityLabel = (q: "fast" | "best") => (q === "fast" ? "Fast preview" : "Best");
  const steps: FlowStep<FinishDraft>[] = [
    {
      id: "finish",
      label: "Finish",
      summary: (d) => ({ text: `${choices.find((c) => c.value === pickOf(d))?.label ?? ""} · ${qualityLabel(d.quality)}`, edit: "Edit" }),
      body: (d, set) => (
        <>
          <Segmented<FinishChoice> block label="Finish" value={pickOf(d)} onChange={(v) => set({ choice: v })} options={choices} />
          <RunFacts models={finishModels(pickOf(d))} eta={t.status?.tiles?.finish?.eta_s ? t.status.tiles.finish.eta_s * FINISH_SHARE[pickOf(d)] : null} />
        </>
      ),
      advanced: (d, set) => (
        <div className="space-y-2">
          <span className="block text-[14px] text-ink">Quality</span>
          <Segmented<"fast" | "best">
            block
            label="Quality"
            value={d.quality}
            onChange={(v) => set({ quality: v })}
            options={[
              { value: "fast", label: "Fast preview" },
              { value: "best", label: "Best" },
            ]}
          />
        </div>
      ),
      primary: () => ({ label: "Finish" }),
    },
  ];
  return (
    <StepFlow
      task="finish"
      shot={shot}
      ui={ui}
      draft={dr}
      steps={steps}
      need={need}
      title={finishTitle(pickOf(dr.data), long)}
      params={(d) => finishParams(pickOf(d), d.quality)}
      reviewExtra={(v) => <DownloadControl downloads={v.downloads ?? []} />}
    />
  );
}

type RecastDraft = {
  /** The pick, kept so a switched shot or task comes back to it. */
  pick: Pick | null;
  /** What the Who row says once done: "Person 1", and their cutout. */
  who: string;
  thumb: string | null;
  /** Character: someone from the cast, or a new character made here. */
  mode: "cast" | "new";
  castId: string | null;
  fresh: NewCharacterData;
  engine: RecastEngine;
  relight: boolean;
  /** From a take's settings ("Try again with changes"): what it ran with, kept while the character is the same. */
  kept: { castId: string; prompt: string; scene: string; seed: number | null; engine: string };
  /** Make sheet was pressed here for this cast member: when their sheet is ready, the render starts by itself. */
  go: string | null;
};

/** Replace a person as steps: Who (a click or a chip), Character (Cast | New, with the sheet's state), then the shared
 * Render and Review. One person; several people stay on the old form. */
function RecastFlow({ shot, ui }: FormProps) {
  const t = useTask("recast", shot, ui);
  const sel = useSelection(shot.id);
  const pre = usePrefill(shot.id, "recast");
  const projectId = shot.project_id || "unsorted";
  const people = useCast(projectId).filter((c) => c.kind === "person");
  const dr = useDraft<RecastDraft>(
    shot.id,
    "recast",
    () => {
      const castId = str(pre?.cast_id) || null;
      const words = castId ? "" : str(pre?.describe);
      return {
        pick: restoreOf(pre),
        who: "",
        thumb: null,
        mode: words ? "new" : "cast",
        castId,
        fresh: { ...NEW_CHARACTER, words },
        engine: str(pre?.engine) === "scail" ? "scail" : "ltx",
        relight: bool(pre?.relight, false),
        kept: { castId: castId ?? "", prompt: str(pre?.prompt), scene: str(pre?.scene), seed: typeof pre?.seed === "number" ? pre.seed : null, engine: str(pre?.engine) || "scail" },
        go: null,
      };
    },
    { fresh: !!pre, step: "who" },
  );
  const d0 = dr.data;
  const onWho = dr.draft.step === "who" || !dr.draft.step;
  const player = usePlayer();

  // --- Who: click-to-select only while choosing who; Render and Review leave the picture alone ---
  const [restore] = useState(() => d0.pick);
  useEffect(() => {
    if (!onWho) return;
    startSelection(shot.id);
    const cur = getState().selection;
    if (restore && !cur?.points.length && !cur?.text) {
      player.seek(restore.frame);
      restoreSelection(shot.id, restore, restore.from);
    }
    return () => endSelection();
  }, [shot.id, onWho]);
  const finder = usePeople(shot.id, t.baseId);
  const found = finder.s === "found" ? finder.res.people : [];
  const chosen = sel && finder.s === "found" && sel.frame === finder.res.frame ? found.findIndex((p) => samePoints(sel.points, personPoints(p))) : -1;
  const choose = (p: PersonFound) => {
    if (finder.s !== "found") return;
    if (player.getSnapshot().playing) player.pause();
    player.seek(finder.res.frame);
    pickPerson(shot.id, { frame: finder.res.frame, points: personPoints(p), maskUrl: p.mask_url });
  };
  useEffect(() => {
    setSelectionFinding(shot.id, onWho && finder.s === "finding");
  }, [finder.s, shot.id, onWho]);
  // A click in the picture on someone the finder found picks them (their matte, no find); on the same person it adds.
  useEffect(() => {
    if (!onWho || finder.s !== "found" || !finder.res.people.length) return;
    const res = finder.res;
    setAutoClick((pt, frame) => {
      const hit = frame === res.frame ? personAt(res.people, pt) : null;
      if (!hit) return "none";
      const cur = getState().selection;
      if (cur && samePoints(cur.points, personPoints(hit))) return "same";
      choose(hit);
      return "switched";
    });
    return () => setAutoClick(null);
  }, [finder, onWho]);
  // The person most on screen is picked for you, once, when nothing is picked yet.
  const autoDone = useRef(false);
  useEffect(() => {
    if (!onWho || autoDone.current || finder.s !== "found") return;
    autoDone.current = true;
    const s = getState().selection;
    if (s && s.shotId === shot.id && (s.points.length || s.text || s.kept)) return;
    const main = mainOf(finder.res);
    if (main) choose(main);
  }, [finder.s, onWho]);
  // What's picked, kept in the draft (and how the Who row will say it).
  useEffect(() => {
    if (!onWho || !sel || sel.loading) return;
    const has = sel.points.length > 0 || !!sel.text;
    const who = chosen >= 0 ? `Person ${chosen + 1}` : sel.text ? `“${sel.text}”` : has ? "Your pick" : "";
    dr.set({ pick: has ? { points: sel.points, frame: sel.frame, text: sel.text, from: "before" } : null, who, thumb: chosen >= 0 ? found[chosen].thumb_url : null });
  }, [sel?.seq, sel?.loading, chosen, onWho]);
  // Words that name a vehicle: this build swaps people only (the server refuses too, recast_ltx.NO_CARS).
  const carWords = !CARS && !!sel?.text && VEHICLE_WORDS.test(sel.text.trim());

  // --- Character ---
  const setFresh = (p: Partial<NewCharacterData>) => dr.patch((x) => ({ data: { ...x.data, fresh: { ...NEW_CHARACTER, ...(x.data.fresh as NewCharacterData), ...p } } }));
  const nc = useNewCharacter(projectId, d0.fresh, setFresh);
  const mode = people.length ? d0.mode : "new";
  const member = mode === "new" ? nc.member : people.find((c) => c.id === d0.castId) ?? null;
  const sheet = useSheet(projectId, member);
  // A new character has joined the cast (Make sheet): they are the pick from here, New starts clean for another, and
  // the render starts by itself when their sheet is ready.
  const joined = d0.fresh.castId && people.some((c) => c.id === d0.fresh.castId) ? d0.fresh.castId : null;
  useEffect(() => {
    if (joined) dr.patch((x) => ({ data: { ...x.data, mode: "cast", castId: joined, fresh: { ...NEW_CHARACTER }, go: joined } }));
  }, [joined]);
  // Rendering: the auto-start is spent.
  useEffect(() => {
    if (dr.draft.step === "render" && d0.go) dr.set({ go: null });
  }, [dr.draft.step]);

  const anchor = d0.pick ? d0.pick.frame : finder.s === "found" ? finder.res.frame : finder.s === "failed" ? 0 : null;
  const bracket = useShotWindow(shot, typeof pre?.start_frame === "number" ? Math.max(0, Math.round(pre.start_frame)) : null, anchor, d0.engine === "scail", true, d0.engine === "ltx");

  const params = (d: RecastDraft): JobParams => {
    const pick = onWho && sel && (sel.points.length || sel.text) ? { points: sel.points, text: sel.text, frame: sel.frame } : d.pick;
    const id = (people.length ? d.mode : "new") === "new" ? d.fresh.castId : d.castId;
    if (!pick || !id) return {};
    const same = !!d.kept.castId && d.kept.castId === id;
    return recastParams(pick, d.engine, { cast_id: id }, {
      relight: RELIGHT && d.engine === "ltx" && d.relight,
      kept: same ? { prompt: d.kept.engine === d.engine ? d.kept.prompt : "", scene: d.kept.scene, seed: d.kept.seed } : undefined,
      win: bracket.show && bracket.win ? bracket.win : null,
    });
  };
  // What the video model is told, as recast_ltx.py and recast.py write it (a cast pick is named by their mention).
  const receipt = (d: RecastDraft) => {
    const p = params(d) as RecastParams;
    const whoWords = member ? `@${member.name}` : "";
    return p.prompt || (d.engine === "ltx" ? ltxRecastPrompt(whoWords, p.scene ?? "") : scailPrompt(whoWords, p.scene ?? ""));
  };

  const characterPrimary = (): { label: string; disabled?: boolean; run?: () => Promise<void> } => {
    if (!member) {
      if (mode === "new") return nc.primary;
      return { label: "Pick a character", disabled: true };
    }
    const p = sheetPrimary(sheet.st, member.name);
    if (sheet.st.s === "ready") return { label: p.label, disabled: !t.ready };
    const start = async () => {
      dr.set({ go: member.id });
      await sheet.start();
    };
    return { label: p.label, disabled: p.disabled || sheet.starting || sheet.measuring, run: p.disabled ? undefined : start };
  };

  const steps: FlowStep<RecastDraft>[] = [
    {
      id: "who",
      label: "Who",
      summary: (d) => (d.who ? { thumb: d.thumb, text: d.who, edit: "Edit" } : null),
      body: () => (
        <>
          {finder.s === "finding" ? <BusyRow text="Finding who's in the shot…" /> : null}
          {found.length ? (
            <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="People in the shot">
              {found.map((p, i) => (
                <button
                  key={`${p.index}:${p.thumb_url}`}
                  type="button"
                  role="radio"
                  aria-checked={i === chosen}
                  aria-label={`Person ${i + 1}`}
                  onClick={() => choose(p)}
                  className={cx(
                    "relative h-[84px] w-[64px] overflow-hidden rounded-[4px] border bg-black transition-[border-color,opacity,box-shadow] duration-150",
                    i === chosen ? "border-ivory shadow-[0_0_0_1px_#EEE9E0]" : "border-line opacity-70 hover:border-line2 hover:opacity-100",
                  )}
                >
                  <img src={p.thumb_url} alt="" className="thumb-face h-full w-full" />
                  {i === chosen ? (
                    <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-ivory text-[#0D0C0B]" aria-hidden="true">
                      <Check size={10} strokeWidth={3} />
                    </span>
                  ) : null}
                </button>
              ))}
            </div>
          ) : null}
          {/* What to click is said once, over the picture; the rail only shows a pick that isn't one of the people. */}
          {sel && chosen < 0 && (sel.points.length || sel.text || sel.kept || sel.loading || sel.error) ? <SelectionStatus shot={shot} idle="" /> : null}
          {carWords ? <p className="text-[13px] text-ink2">This version replaces people only.</p> : null}
          {picked(sel) ? <TrackCheck shot={shot} versionId={t.baseId} startFrame={d0.engine === "ltx" && bracket.show && bracket.win ? bracket.win.start : undefined} /> : null}
        </>
      ),
      advanced: () => <PickInWords shot={shot} placeholder="or type it: the person on the left" label="Who to replace, in words" />,
      primary: () => ({ label: chosen >= 0 ? `Use Person ${chosen + 1}` : "Use this pick", disabled: !picked(sel) || carWords }),
    },
    {
      id: "character",
      label: "Character",
      summary: () => (member ? { thumb: member.thumb_url || member.url, text: member.name, edit: "Change" } : null),
      body: (d, set, flow) => (
        <>
          {people.length ? (
            <Segmented<"cast" | "new">
              block
              label="Character"
              value={mode}
              onChange={(v) => set({ mode: v })}
              options={[
                { value: "cast", label: "Cast" },
                { value: "new", label: "New" },
              ]}
            />
          ) : null}
          {mode === "cast" ? (
            <CastGrid projectId={projectId} cast={people} value={d.castId} onPick={(c) => set({ castId: c.id })} />
          ) : member ? (
            <SummaryRow thumb={d.fresh.picked ? nc.urlOf(d.fresh.picked) : null} text={personWords(d.fresh.words) || member.name} edit="Change" onEdit={() => setFresh({ castId: null })} />
          ) : (
            <NewCharacterBody projectId={projectId} data={d.fresh} set={setFresh} nc={nc} shotId={shot.id} />
          )}
          {member ? (
            <SheetPanel
              member={member}
              sheet={sheet}
              actions={
                sheet.st.s === "ready"
                  ? (_it, close) => (
                      <Button
                        variant="primary"
                        size="sm"
                        disabled={!flow.canNext}
                        onClick={() => {
                          close();
                          flow.next();
                        }}
                      >
                        {`Use ${member.name}`}
                      </Button>
                    )
                  : undefined
              }
            />
          ) : null}
        </>
      ),
      advanced: (d, set, flow) => (
        <>
          {/* Make sheet stays the primary; a pro can go on with just their picture (recast uses it when there's no sheet). */}
          {member && (sheet.st.s === "none" || sheet.st.s === "failed") ? (
            <Button variant="secondary" size="sm" disabled={!flow.canNext} onClick={() => flow.next()}>
              Use without a sheet
            </Button>
          ) : null}
          <Segmented<RecastEngine>
            block
            label="Engine"
            value={d.engine}
            onChange={(v) => set({ engine: v })}
            options={[
              { value: "ltx", label: "LTX-2.5" },
              { value: "scail", label: "SCAIL-2" },
            ]}
          />
          {RELIGHT && d.engine === "ltx" ? <SwitchRow checked={d.relight} onChange={(v) => set({ relight: v })} title="Relight the new person" /> : null}
          <Fold label="Prompt">
            <PromptReceipt projectId={projectId} text={receipt(d)} models={[d.engine]} reference={member ? { name: member.name, url: member.thumb_url || member.url } : null} />
          </Fold>
        </>
      ),
      primary: characterPrimary,
      autoNext: (d) => (d.go && member?.id === d.go && sheet.st.s === "ready" ? `sheet:${d.go}` : null),
    },
  ];
  return <StepFlow task="recast" shot={shot} ui={ui} draft={dr} steps={steps} params={params} lengthNote={!bracket.show} />;
}
