// "Enhance": the prompt director rewrites a short idea into the prompt a model reads best. It never replaces the user's
// text by itself: the director's version shows under the field as a suggestion (Use · Keep mine · Try again).
import { PenLine, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { getApi, openSetup } from "../actions";
import { clock, cx } from "../lib/format";
import { useApp } from "../store";
import { segments } from "../lib/mentions";
import type { CastMember, EnhanceRequest, EnhanceResult } from "../types";

const EMPTY_CAST: CastMember[] = [];
import { Button, Spinner } from "./ui";

const HINT_KEY = "pickups.directorHint";

function hintSeen(): boolean {
  try {
    return window.localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return true;
  }
}
function markHintSeen() {
  try {
    window.localStorage.setItem(HINT_KEY, "1");
  } catch {
    /* ignore */
  }
}

/** Word-level diff (LCS), for showing what the director changed. */
function diffWords(a: string, b: string): Array<{ t: string; op: "same" | "add" | "del" }> {
  const A = a.split(/(\s+)/).filter(Boolean);
  const B = b.split(/(\s+)/).filter(Boolean);
  if (A.length * B.length > 250_000) return [{ t: b, op: "add" }];
  const n = A.length;
  const m = B.length;
  const L: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: Array<{ t: string; op: "same" | "add" | "del" }> = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) {
      out.push({ t: A[i], op: "same" });
      i++;
      j++;
    } else if (L[i + 1][j] >= L[i][j + 1]) out.push({ t: A[i++], op: "del" });
    else out.push({ t: B[j++], op: "add" });
  }
  while (i < n) out.push({ t: A[i++], op: "del" });
  while (j < m) out.push({ t: B[j++], op: "add" });
  return out;
}

type State =
  | { s: "idle" }
  | { s: "busy"; started: number; seq: number }
  | { s: "done"; result: EnhanceResult; base: string }
  | { s: "error"; message: string; off?: boolean };

/**
 * Returns the Enhance pill (for the field's label row) and the suggestion panel (for under the field).
 * Both are null when the director is turned off.
 */
export function useEnhance({
  request,
  text,
  onUse,
}: {
  /** Everything but the text: target, shot, frame, refs… */
  request: Omit<EnhanceRequest, "text"> | null;
  text: string;
  onUse: (v: string) => void;
}): { button: ReactNode; panel: ReactNode } {
  const director = useApp((s) => s.director);
  const [st, setSt] = useState<State>({ s: "idle" });
  const [hint, setHint] = useState(false);
  const [now, setNow] = useState(Date.now());
  const seed = useRef(0);
  const seq = useRef(0);

  useEffect(() => {
    if (st.s !== "busy") return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [st.s]);

  const off = director?.backend === "off";
  if (off || !request) return { button: null, panel: null };

  const run = async (again = false) => {
    if (!text.trim()) return;
    if (again) seed.current += 1;
    if (director && director.backend === "comfy" && !director.url && !hintSeen()) setHint(true);
    const mine = ++seq.current;
    const base = text;
    setSt({ s: "busy", started: Date.now(), seq: mine });
    setNow(Date.now());
    try {
      const result = await getApi().enhancePrompt({ ...request, text, ...(seed.current ? { seed: seed.current } : {}) });
      if (seq.current === mine) setSt({ s: "done", result, base });
    } catch (e) {
      if (seq.current !== mine) return;
      const offNow = e instanceof ApiError && e.status === 409;
      setSt({ s: "error", message: e instanceof ApiError && e.message ? e.message : "The prompt director couldn't run. Try again.", off: offNow });
    }
  };
  const close = () => {
    seq.current += 1;
    setSt({ s: "idle" });
  };

  const busy = st.s === "busy";
  const button = (
    <button
      type="button"
      onClick={() => void run()}
      disabled={!text.trim() || busy}
      title="Rewrite this the way the model reads best"
      className="inline-flex h-6 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-line px-2.5 text-[12px] text-ink2 transition-colors duration-150 hover:border-ivory/50 hover:text-ivory disabled:border-line disabled:text-ink4"
    >
      <PenLine size={12} strokeWidth={1.5} aria-hidden="true" />
      Enhance
    </button>
  );

  const panel =
    st.s === "idle" && !hint ? null : (
      <div className="mt-2 space-y-2" aria-live="polite">
        {hint ? (
          <div className="flex items-center gap-3 rounded-[4px] border border-line px-3 py-2 text-[12px] text-ink2">
            <span className="flex-1">Faster with a local LLM.</span>
            <button
              type="button"
              onClick={() => {
                markHintSeen();
                setHint(false);
                openSetup(null, "prompt-director");
              }}
              className="text-ivory/80 underline-offset-4 hover:text-ivory hover:underline"
            >
              Setup
            </button>
            <button
              type="button"
              aria-label="Dismiss"
              onClick={() => {
                markHintSeen();
                setHint(false);
              }}
              className="rounded-full p-0.5 text-ink3 hover:text-ivory"
            >
              <X size={12} />
            </button>
          </div>
        ) : null}
        {st.s === "busy" ? (
          <div className="flex items-center gap-2.5 px-1 text-ink3">
            <Spinner size={11} />
            <span className="font-mono text-[11px]">
              Directing… usually 5–20 s <span className="text-ink4">· {clock((now - st.started) / 1000)}</span>
            </span>
            <button type="button" onClick={close} className="ml-auto text-[12px] text-ink3 hover:text-ivory">
              Stop waiting
            </button>
          </div>
        ) : st.s === "error" ? (
          <div className="flex items-baseline gap-2.5 rounded-[4px] border border-line2 px-3 py-2 text-[13px] text-ink">
            <span className="slate shrink-0">Error</span>
            <span className="flex-1">{st.message}</span>
            {st.off ? (
              <button type="button" onClick={() => openSetup(null, "prompt-director")} className="shrink-0 text-[12px] text-ink2 hover:text-ivory">
                Setup
              </button>
            ) : (
              <button type="button" onClick={() => void run()} className="shrink-0 text-[12px] text-ink2 hover:text-ivory">
                Try again
              </button>
            )}
          </div>
        ) : st.s === "done" ? (
          <Suggestion
            projectId={request.project_id}
            result={st.result}
            base={st.base}
            onUse={() => {
              onUse(st.result.text);
              close();
            }}
            onKeep={close}
            onAgain={() => void run(true)}
          />
        ) : null}
      </div>
    );
  return { button, panel };
}

/** Text with @cast mentions drawn as inline chips; whitespace kept (the parent is pre-wrap). */
function ChipText({ text, cast, added, muted }: { text: string; cast: CastMember[]; added?: boolean[]; muted?: boolean }) {
  const segs = useMemo(() => {
    const known = segments(text, cast);
    if (known.some((x) => x.cast)) return known.map((x) => ({ text: x.text, chip: !!x.cast }));
    // No cast loaded: a bare @word still reads as a mention.
    const out: Array<{ text: string; chip: boolean }> = [];
    let last = 0;
    for (const m of text.matchAll(/(^|[^\p{L}\p{N}_])(@[\p{L}\p{N}_]+)/gu)) {
      const at = (m.index ?? 0) + m[1].length;
      if (at > last) out.push({ text: text.slice(last, at), chip: false });
      out.push({ text: m[2], chip: true });
      last = at + m[2].length;
    }
    if (last < text.length) out.push({ text: text.slice(last), chip: false });
    return out;
  }, [text, cast]);
  let offset = 0;
  return (
    <>
      {segs.map((seg, i) => {
        const from = offset;
        offset += seg.text.length;
        if (seg.chip) {
          return (
            <span
              key={i}
              className={cx("rounded-[3px] px-[3px] no-underline", muted ? "bg-line2 text-ink3" : "bg-ivory text-[#0D0C0B]")}
              style={muted ? { textDecoration: "line-through" } : undefined}
            >
              {seg.text}
            </span>
          );
        }
        if (!added) return <span key={i}>{seg.text}</span>;
        // Split the plain run into added / kept pieces.
        const runs: Array<{ t: string; add: boolean }> = [];
        for (let k = 0; k < seg.text.length; k++) {
          const add = !!added[from + k] && /\S/.test(seg.text[k]);
          const prev = runs[runs.length - 1];
          if (prev && (prev.add === add || /\s/.test(seg.text[k]))) prev.t += seg.text[k];
          else runs.push({ t: seg.text[k], add });
        }
        return (
          <span key={i}>
            {runs.map((r, j) => (
              <span key={j} className={r.add ? "bg-ivory/10 text-ivory" : "text-ink2"}>
                {r.t}
              </span>
            ))}
          </span>
        );
      })}
    </>
  );
}

function Suggestion({ projectId, result, base, onUse, onKeep, onAgain }: { projectId?: string; result: EnhanceResult; base: string; onUse: () => void; onKeep: () => void; onAgain: () => void }) {
  // The prompt's project (a shot's own project), which isn't always the current one.
  const cast = useApp((s) => s.cast[projectId ?? s.currentProjectId] ?? EMPTY_CAST);
  // Which characters of the director's version are new (word diff against the user's text).
  const added = useMemo(() => {
    const mask: boolean[] = [];
    for (const p of diffWords(base, result.text)) {
      if (p.op === "del") continue;
      for (let k = 0; k < p.t.length; k++) mask.push(p.op === "add");
    }
    return mask;
  }, [base, result.text]);
  return (
    <div className="rounded-[4px] border border-ivory/30 bg-raised/60 px-3 py-2.5 animate-rise-in">
      <div className="slate mb-1.5 text-ink3">
        Director's version <span className="text-ink4">· via</span> <span className="normal-case tracking-normal text-ink2">{result.runs_on || "ComfyUI"}</span>
      </div>
      {base.trim() ? (
        <p className="mb-2 whitespace-pre-wrap break-words text-[12px] leading-[19px] text-ink4 line-through decoration-ink4/70" aria-label="Your version">
          <ChipText text={base} cast={cast} muted />
        </p>
      ) : null}
      <p className="whitespace-pre-wrap break-words text-[13px] leading-[20px]">
        <ChipText text={result.text} cast={cast} added={added} />
      </p>
      <div className="mt-2.5 flex items-center gap-2">
        <Button variant="primary" size="sm" onClick={onUse}>
          Use
        </Button>
        <Button variant="ghost" size="sm" onClick={onKeep}>
          Keep mine
        </Button>
        <Button variant="quiet" className="ml-auto" onClick={onAgain}>
          Try again
        </Button>
      </div>
    </div>
  );
}
