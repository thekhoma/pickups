// Timecoded notes on the take on screen (the original too): N or the transport's Note button opens a small input at
// the playhead, notes show as ticks on the scrubber and as a list in the take panel, and Export timeline carries them
// into the edit as markers. A note on the original goes with the shot's approved take (nle.fold_original_notes), unless
// the original is approved itself; the composer and the list say so.
import { X } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { addNote, deleteNote } from "../actions";
import { cx, takes, timecode } from "../lib/format";
import { usePlayer, usePlayerState, type Player } from "../lib/player";
import { getState } from "../store";
import type { Note, Shot, Version } from "../types";
import { Hint } from "./ui";

// --- the open composer (one at a time) ----------------------------------------------------------------------------

type Draft = { shotId: string; takeId: string; n: number } | null;
let draft: Draft = null;
const subs = new Set<() => void>();
function setDraft(d: Draft) {
  draft = d;
  subs.forEach((l) => l());
}
const subscribe = (l: () => void) => {
  subs.add(l);
  return () => subs.delete(l);
};

/** Whether the note input is open on this take (the Note button hides its tooltip, which would sit on it). */
export function useNoteDraft(shotId: string, takeId: string): boolean {
  const d = useSyncExternalStore(subscribe, () => draft);
  return !!d && d.shotId === shotId && d.takeId === takeId;
}

/** Pause and open the note input at the playhead (again: just focus it). */
export function startNote(player: Player, shotId: string, takeId: string) {
  if (player.getSnapshot().playing) player.pause();
  setDraft({ shotId, takeId, n: (draft?.n ?? 0) + 1 });
}

function isTyping(t: EventTarget | null) {
  const el = t as HTMLElement | null;
  if (!el) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable;
}

/** N leaves a note on the take on screen (not while typing, not under Setup). */
export function useNoteKey(shotId: string, takeId: string) {
  const player = usePlayer();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "n" && e.key !== "N") return;
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat || e.defaultPrevented || isTyping(e.target)) return;
      const s = getState();
      if (s.setup.open || s.route.name !== "shot") return;
      e.preventDefault();
      startNote(player, shotId, takeId);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, shotId, takeId]);
}

/** Where a frame sits along the scrubber (the playhead's own formula). */
export const framePct = (frame: number, frames: number) => ((frame + 0.5) / Math.max(1, frames)) * 100;

export const isPending = (n: Note) => n.id.startsWith("tmp_");

/** The frames of the original a take shows, as the export places it (nle.take_start): from the sum of
 * params.start_frame along input_version, for details.frames (unknown: the whole shot). Null when the take can't be
 * placed (a broken or looping chain): the export folds nothing into it. */
function originalRange(shot: Shot, take: Version): { start: number; frames: number } | null {
  let start = 0;
  const seen = new Set<string>();
  for (let id = take.id; id !== "orig"; ) {
    const v = shot.versions.find((x) => x.id === id);
    if (!v || seen.has(id)) return null;
    seen.add(id);
    const s = Number(v.params?.start_frame ?? 0);
    start += Number.isFinite(s) && s > 0 ? Math.floor(s) : 0;
    id = v.input_version || "orig";
  }
  const frames = take.details?.frames;
  return { start, frames: typeof frames === "number" && frames > 0 ? frames : shot.frames };
}

const takeList = (nums: string[]) =>
  nums.length === 1 ? `Take ${nums[0]}` : `Takes ${nums.slice(0, -1).join(", ")} and ${nums[nums.length - 1]}`;

/** A note on the original (when the original isn't approved itself) reaches the edit with the shot's approved takes
 * that show its moment: Export timeline folds it into them. Those takes ([] when none shows it), or null when the note
 * goes with the take it's on, or no take is approved yet. */
function foldsInto(shot: Shot, version: Version, frame: number): Version[] | null {
  if (version.id !== "orig" || version.status === "approved") return null;
  const approved = shot.versions.filter((v) => v.id !== "orig" && v.status === "approved");
  if (!approved.length) return null;
  return approved.filter((v) => {
    const r = originalRange(shot, v);
    return !!r && frame >= r.start && frame < r.start + r.frames;
  });
}

/** The approved takes made from `version`, directly or along a chain (a Finish of it): the edit gets those, not it. */
function approvedFrom(shot: Shot, version: Version): Version[] {
  const byId = new Map(shot.versions.map((v) => [v.id, v]));
  return shot.versions.filter((v) => {
    if (v.id === version.id || v.status !== "approved") return false;
    const seen = new Set<string>();
    for (let id = v.input_version || "orig"; id !== "orig" && !seen.has(id); id = byId.get(id)?.input_version || "orig") {
      if (id === version.id) return true;
      seen.add(id);
    }
    return false;
  });
}

/** Where a note at `frame` reaches the edit, when that isn't simply with the take it's on (null). */
export function noteDestination(shot: Shot, version: Version, frame: number): string | null {
  const nums = (vs: Version[]) => vs.map((v) => takes(shot)[v.id]?.num ?? "");
  if (version.id !== "orig") {
    // A take made from this one is approved (its Finish): the edit gets that take, and this take's notes stay here.
    const later = version.status === "approved" ? [] : approvedFrom(shot, version);
    return later.length ? `Not in the edit: ${takeList(nums(later))} ${later.length === 1 ? "is" : "are"} approved` : null;
  }
  if (version.status === "approved") return null;
  const into = foldsInto(shot, version, frame);
  if (!into) return "Goes to the edit once a take of this shot is approved";
  if (!into.length) {
    const approved = shot.versions.filter((v) => v.id !== "orig" && v.status === "approved");
    return `This moment isn't in ${takeList(nums(approved))}, so it stays out of the edit`;
  }
  return `Goes to the edit with ${takeList(nums(into))}`;
}

/** The note whose tick is within `px` of x (pixels from the scrubber's left edge), nearest first. */
export function noteNear(notes: Note[] | undefined, frames: number, width: number, x: number, px = 5): Note | null {
  let best: Note | null = null;
  let dist = px + 0.001;
  for (const n of notes ?? []) {
    const d = Math.abs((framePct(n.frame, frames) / 100) * width - x);
    if (d < dist) {
      best = n;
      dist = d;
    }
  }
  return best;
}

/** Small ivory ticks along the bottom of the scrubber. They never take the pointer: the scrubber finds them. */
export function NoteTicks({ notes, frames, active }: { notes?: Note[]; frames: number; active: string | null }) {
  if (!notes?.length) return null;
  return (
    <>
      {notes.map((n) => (
        <span
          key={n.id}
          aria-hidden="true"
          className={cx(
            "pointer-events-none absolute bottom-0 z-[1] w-[3px] -translate-x-1/2 rounded-t-[1px] shadow-[0_0_0_1px_rgba(0,0,0,0.75)] transition-[height,background-color] duration-150",
            n.id === active ? "h-3.5 bg-ivory" : "h-2.5 bg-ivory/70",
            isPending(n) && "opacity-60",
          )}
          style={{ left: `${framePct(n.frame, frames)}%` }}
        />
      ))}
    </>
  );
}

// --- the input at the playhead -------------------------------------------------------------------------------------

/** Lives in the transport row and floats above it, over the playhead. */
export function NoteComposer({ shot, version }: { shot: Shot; version: Version }) {
  const d = useSyncExternalStore(subscribe, () => draft);
  // A note belongs to the take it was started on: switching takes or leaving the shot drops the input.
  useEffect(() => () => setDraft(null), [shot.id, version.id]);
  if (!d || d.shotId !== shot.id || d.takeId !== version.id) return null;
  return <Composer shot={shot} version={version} focusKey={d.n} />;
}

function Composer({ shot, version, focusKey }: { shot: Shot; version: Version; focusKey: number }) {
  const player = usePlayer();
  const snap = usePlayerState();
  const ref = useRef<HTMLInputElement>(null);
  const [text, setText] = useState("");
  const id = useId();
  const tc = timecode(snap.frame, shot.fps);
  const pct = framePct(snap.frame, shot.frames);
  // The note goes on the take on screen, whichever side (Before) is showing: say which, and where it reaches the edit.
  const on = version.id === "orig" ? "the original" : `Take ${takes(shot)[version.id]?.num ?? ""}`;
  const goes = noteDestination(shot, version, snap.frame);

  useEffect(() => {
    ref.current?.focus();
  }, [focusKey]);

  // Esc cancels before anything else hears it (the app's Esc steps back), also once the input has lost focus. Not an
  // Esc meant for another field, an open menu, list or dialog, or Setup: those keep theirs, and the draft stays.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (e.target !== ref.current) {
        const popup = document.querySelector('[role="menu"], [role="listbox"], [role="dialog"], [aria-modal="true"]');
        if (isTyping(e.target) || popup || getState().setup.open) return;
      }
      e.preventDefault();
      e.stopPropagation();
      setDraft(null);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  const save = () => {
    if (!text.trim()) return;
    void addNote(shot.id, version.id, player.getSnapshot().frame, text);
    setDraft(null);
  };

  return (
    <>
      <span
        aria-hidden="true"
        className="pointer-events-none absolute bottom-full z-20 mb-[3px] h-0 w-0 -translate-x-1/2 border-x-[5px] border-t-[5px] border-x-transparent border-t-line2"
        style={{ left: `${pct}%` }}
      />
      {/* Centred on the playhead by the outer box; the inner one animates (its transform would undo the centring). */}
      <div className="absolute bottom-full z-20 mb-2 w-[320px] -translate-x-1/2" style={{ left: `clamp(160px, ${pct}%, calc(100% - 160px))` }}>
        <div className="rounded-card border border-line2 bg-raised px-2.5 pb-2.5 pt-2 shadow-[0_8px_24px_rgba(0,0,0,0.45)] animate-rise-in">
          <label htmlFor={id} className="slate block truncate text-ink3">
            Note on {on} at <span className="text-ivory">{tc}</span>
          </label>
          {goes ? (
            <p id={`${id}-goes`} className="mt-0.5 text-[12px] leading-snug text-ink3">
              {goes}.
            </p>
          ) : null}
          <input
            id={id}
            ref={ref}
            aria-describedby={goes ? `${id}-goes ${id}-keys` : `${id}-keys`}
            value={text}
            maxLength={500}
            autoComplete="off"
            spellCheck
            placeholder="What should change here?"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
              e.preventDefault();
              save();
            }}
            onBlur={() => {
              if (!text.trim()) setDraft(null);
            }}
            className="field mt-1.5 h-8 px-2.5 text-[13px]"
          />
          <div id={`${id}-keys`} className="slate mt-1.5 text-right text-ink4">
            ↵ save · esc cancel
          </div>
        </div>
      </div>
    </>
  );
}

// --- the list in the take panel ------------------------------------------------------------------------------------

/** "Notes · 2", one row per note in frame order: click a row to go to its frame, × deletes (with Undo). */
export function NotesList({ shot, version, hideEmpty, className }: { shot: Shot; version: Version; hideEmpty?: boolean; className?: string }) {
  const player = usePlayer();
  const snap = usePlayerState();
  const notes = useMemo(() => [...(version.notes ?? [])].sort((a, b) => a.frame - b.frame), [version.notes]);
  if (!notes.length && hideEmpty) return null;
  // On the original: these go to the edit with the shot's approved take (a note outside it stays out, and says so).
  const orig = version.id === "orig" && version.status !== "approved";
  return (
    <section aria-label="Notes" className={className}>
      <div className="mb-1 flex items-center gap-1.5">
        <span className="slate text-ink3">Notes · {notes.length}</span>
        {orig && notes.length ? <Hint text="They go to the edit with this shot's approved take." /> : null}
      </div>
      {notes.length ? (
        <ul className="border-t border-line">
          {notes.map((n) => {
            const here = n.frame === snap.frame;
            const tc = timecode(n.frame, shot.fps);
            const left = orig && foldsInto(shot, version, n.frame)?.length === 0;
            return (
              <li key={n.id} className={cx("group/note flex items-start gap-1 border-b border-line", isPending(n) && "opacity-60")}>
                <button
                  type="button"
                  aria-label={`${tc}: ${n.text}`}
                  onClick={() => {
                    if (player.getSnapshot().playing) player.pause();
                    player.seek(n.frame);
                  }}
                  className="flex min-w-0 flex-1 items-baseline gap-3 py-2 text-left"
                >
                  <span className={cx("mono shrink-0 text-[12px] transition-colors duration-150", here ? "text-ivory" : "text-ink3")}>{tc}</span>
                  <span className={cx("min-w-0 break-words text-[13px] leading-snug transition-colors duration-150", here ? "text-ivory" : "text-ink group-hover/note:text-ivory")}>
                    {n.text}
                    {left ? <span className="ml-1.5 whitespace-nowrap text-[12px] text-ink4">· not in the approved take</span> : null}
                  </span>
                </button>
                {isPending(n) ? null : (
                  <button
                    type="button"
                    aria-label={`Delete the note at ${tc}`}
                    onClick={() => void deleteNote(shot.id, version.id, n)}
                    className="mt-1.5 shrink-0 rounded-full p-1 text-ink3 opacity-0 transition-opacity duration-150 hover:text-ivory focus-visible:opacity-100 group-hover/note:opacity-100"
                  >
                    <X size={13} />
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="text-[13px] text-ink3">
          Press Note under the picture, or <span className="kbd">N</span>, to leave a note at the playhead.
        </p>
      )}
    </section>
  );
}
