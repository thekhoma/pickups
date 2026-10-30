// Prompt fields with @-mentions of the project's cast (inline chips) and the live prompt receipt underneath:
// what each model will actually get (docs/projects-spec.md → "Consistency").
import { ChevronDown } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { getApi } from "../actions";
import { cx } from "../lib/format";
import { matchingCast, mentionQuery, segments } from "../lib/mentions";
import { useApp } from "../store";
import type { CastMember, MentionPreview } from "../types";

const EMPTY: CastMember[] = [];

/** The cast of a project (loaded with the project). */
export function useCast(projectId: string | undefined): CastMember[] {
  return useApp((s) => (projectId ? s.cast[projectId] : undefined)) ?? EMPTY;
}

const FIELD_TEXT = "px-3 py-2.5 text-[14px] leading-[22px] font-sans";

/**
 * A textarea with a mirror underneath that draws mentions as ivory chips (16 px thumbnail + name). The textarea's own
 * text is transparent, so the chips line up exactly with the characters being edited.
 */
export function MentionField({
  value,
  onChange,
  cast,
  projectId,
  placeholder,
  rows = 3,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  cast: CastMember[];
  projectId?: string;
  placeholder?: string;
  rows?: number;
  label: string;
}) {
  const ta = useRef<HTMLTextAreaElement>(null);
  const mirror = useRef<HTMLDivElement>(null);
  const [caret, setCaret] = useState(0);
  const [focused, setFocused] = useState(false);
  const [index, setIndex] = useState(0);
  const [dismissedAt, setDismissedAt] = useState<number | null>(null);

  const segs = useMemo(() => segments(value, cast), [value, cast]);
  const q = focused ? mentionQuery(value, caret) : null;
  const options = q ? matchingCast(cast, q.query) : [];
  // A finished mention ("@Mara " plus whatever follows) doesn't reopen the list.
  const complete = !!q && cast.some((c) => q.query.toLowerCase().startsWith(c.name.toLowerCase()) && /\s/.test(q.query.charAt(c.name.length)));
  const open = !!q && !complete && dismissedAt !== q.start && (options.length > 0 || cast.length === 0);

  useEffect(() => setIndex(0), [q?.query]);

  const syncCaret = () => setCaret(ta.current?.selectionStart ?? 0);

  const insert = (c: CastMember) => {
    if (!q) return;
    const before = value.slice(0, q.start);
    const after = value.slice(caret);
    const add = `@${c.name}${after.startsWith(" ") ? "" : " "}`;
    const next = before + add + after;
    onChange(next);
    const pos = before.length + add.length;
    setDismissedAt(q.start);
    requestAnimationFrame(() => {
      ta.current?.focus();
      ta.current?.setSelectionRange(pos, pos);
      setCaret(pos);
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!open) return;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      setDismissedAt(q!.start);
      return;
    }
    if (!options.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setIndex((i) => (i + (e.key === "ArrowDown" ? 1 : options.length - 1)) % options.length);
    } else if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault();
      insert(options[Math.min(index, options.length - 1)]);
    }
  };

  return (
    <div className="relative">
      <div className="relative overflow-hidden rounded-card border border-line bg-bg transition-colors duration-150 focus-within:border-ivory/60 hover:border-line2">
        <div
          ref={mirror}
          aria-hidden="true"
          style={{ scrollbarGutter: "stable" }}
          className={cx("pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap break-words text-ink", FIELD_TEXT)}
        >
          {segs.map((s, i) =>
            s.cast ? (
              <span key={i} className="rounded-[3px] bg-ivory text-[#0D0C0B] shadow-[0_0_0_2px_#EEE9E0]">
                <span className="relative inline-block">
                  <img src={s.cast.thumb_url || s.cast.url} alt="" className="absolute left-1/2 top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full thumb-face" />
                  <span className="invisible">@</span>
                </span>
                {s.text.slice(1)}
              </span>
            ) : (
              <span key={i}>{s.text}</span>
            ),
          )}
          {/* keep a trailing newline's height */}
          {value.endsWith("\n") ? " " : null}
        </div>
        <textarea
          ref={ta}
          rows={rows}
          aria-label={label}
          value={value}
          placeholder={placeholder}
          spellCheck
          onChange={(e) => {
            onChange(e.target.value);
            setCaret(e.target.selectionStart);
            setDismissedAt(null);
          }}
          onSelect={syncCaret}
          onKeyUp={syncCaret}
          onClick={syncCaret}
          onFocus={() => {
            setFocused(true);
            syncCaret();
          }}
          onBlur={() => window.setTimeout(() => setFocused(false), 120)}
          onKeyDown={onKeyDown}
          onScroll={(e) => {
            if (mirror.current) mirror.current.scrollTop = e.currentTarget.scrollTop;
          }}
          className={cx(
            "relative z-[1] block w-full resize-y bg-transparent text-transparent placeholder:text-ink3 focus-visible:outline-none",
            FIELD_TEXT,
          )}
          style={{ caretColor: "#EEE9E0", scrollbarGutter: "stable" }}
        />
      </div>
      {open ? (
        <div role="listbox" aria-label="Cast" className="absolute left-0 top-full z-40 mt-1 w-[280px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          {options.length ? (
            options.map((c, i) => (
              <button
                key={c.id}
                type="button"
                role="option"
                aria-selected={i === index}
                onMouseDown={(e) => {
                  e.preventDefault();
                  insert(c);
                }}
                onMouseEnter={() => setIndex(i)}
                className={cx("flex w-full items-center gap-2.5 px-3 py-1.5 text-left", i === index ? "bg-line" : "")}
              >
                <img src={c.thumb_url || c.url} alt="" className="thumb-face h-6 w-6 shrink-0 rounded-full" />
                <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{c.name}</span>
                <span className="slate shrink-0 text-ink3">{c.kind}</span>
              </button>
            ))
          ) : (
            <div className="px-3 py-2 text-[13px] text-ink2">
              No cast in this project yet.{" "}
              {projectId ? (
                <a href={`#/project/${encodeURIComponent(projectId)}/cast`} className="text-ivory underline-offset-4 hover:underline">
                  Add people and places
                </a>
              ) : null}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

// --- the prompt receipt ---------------------------------------------------------------------------------------

export type ReceiptModel = "qwen" | "ltx" | "scail" | "gen";

const MODEL_NAME: Record<ReceiptModel, string> = { qwen: "Qwen-Image 2.1", ltx: "LTX-2.5", scail: "SCAIL-2", gen: "Image model" };

function usePreview(projectId: string | undefined, text: string, uploads: number) {
  const [data, setData] = useState<MentionPreview | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!text.trim() || !projectId) {
      setData(null);
      return;
    }
    let cancelled = false;
    const t = window.setTimeout(() => {
      getApi()
        .mentionsPreview({ project_id: projectId, text, uploads })
        .then((d) => {
          if (!cancelled) {
            setData(d);
            setFailed(false);
          }
        })
        .catch(() => !cancelled && setFailed(true));
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [projectId, text, uploads]);
  return { data, failed };
}

function Thumb16({ url }: { url: string }) {
  return <img src={url} alt="" className="thumb-face inline-block h-4 w-4 shrink-0 rounded-full align-[-3px]" />;
}

function Expandable({ text, label }: { text: string; label: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="inline-flex items-center gap-1 text-[11px] text-ink3 hover:text-ivory">
        {label}
        <ChevronDown size={12} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open ? <p className="mono mt-1 whitespace-pre-wrap break-words text-[11px] leading-[17px] text-ink2">{text}</p> : null}
    </div>
  );
}

/**
 * One mono row per model, live: what the image model sees (numbered images, tagged instruction), what the video model
 * reads (mentions expanded to their description), and who Recast uses as the reference.
 */
/** Double spaces (from expanded mentions) read as typos in the receipt. */
function tidy(t: string): string {
  return t.replace(/[ \t]{2,}/g, " ").replace(/ +([,.;:!?])/g, "$1");
}

export function PromptReceipt({
  projectId,
  text,
  models,
  uploads = 0,
  reference,
  genLabel,
}: {
  projectId: string | undefined;
  text: string;
  models: ReceiptModel[];
  /** References uploaded as files before the mentions (they take images 2…). */
  uploads?: number;
  /** Recast's "Who's in", when it isn't a mention. */
  reference?: { name: string; url?: string } | null;
  /** Label for the "gen" row (the image model's name). */
  genLabel?: string;
}) {
  const { data, failed } = usePreview(projectId, text, uploads);
  if (!text.trim() && !reference) return null;
  const firstPerson = data?.refs[0];
  return (
    <div className="mt-2 space-y-2 rounded-[4px] border border-line px-3 py-2.5" aria-live="polite">
      <div className="slate text-ink4">What each model gets</div>
      {failed ? <p className="text-[12px] text-ink3">The preview isn't available right now.</p> : null}
      {models.map((m) => (
        <div key={m} className="space-y-1">
          <div className="slate flex flex-wrap items-center gap-x-2 gap-y-1 text-ink2">
            <span className="text-ivory/80">{m === "gen" && genLabel ? genLabel : MODEL_NAME[m]}</span>
            {m === "qwen" ? (
              <>
                {uploads ? <span>· image 2{uploads > 1 ? `–${uploads + 1}` : ""} = your references</span> : null}
                {(data?.refs ?? []).map((r) => (
                  <span key={r.id} className="inline-flex items-center gap-1">
                    · image {r.image} = <Thumb16 url={r.url} /> <span className="normal-case tracking-normal text-ink">{r.name}</span>
                  </span>
                ))}
                {!uploads && !data?.refs.length ? <span>· image 1 = the shot's frame</span> : null}
              </>
            ) : m === "scail" ? (
              <span className="inline-flex items-center gap-1">
                · reference ={" "}
                {reference ? (
                  <>
                    {reference.url ? <Thumb16 url={reference.url} /> : null}
                    <span className="normal-case tracking-normal text-ink">{reference.name}</span>
                  </>
                ) : firstPerson ? (
                  <>
                    <Thumb16 url={firstPerson.url} /> <span className="normal-case tracking-normal text-ink">{firstPerson.name}</span>
                  </>
                ) : (
                  <span>pick who's in</span>
                )}
              </span>
            ) : null}
          </div>
          {data && m === "qwen" ? <Expandable label="The instruction it reads" text={tidy(data.images)} /> : null}
          {data && m !== "qwen" ? <p className="mono line-clamp-3 break-words text-[11px] leading-[17px] text-ink2">{tidy(data.text)}</p> : null}
        </div>
      ))}
    </div>
  );
}
