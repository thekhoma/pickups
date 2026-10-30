// "Describe it": a local model looks at the shot (or the take being worked from) and writes what's in it, straight into
// the field as an editable suggestion with Undo. It describes, it doesn't rewrite: pickups/describe.py.
import { ScanEye, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { getApi } from "../actions";
import { clock } from "../lib/format";
import { useApp } from "../store";
import type { DescribeField, DescribeResult, EnhanceTarget } from "../types";
import { Spinner } from "./ui";

export interface DescribeFor {
  shotId: string;
  /** The take being worked from; the original when unset. */
  versionId?: string;
  field: DescribeField;
  target?: EnhanceTarget;
}

type State =
  | { s: "idle" }
  | { s: "busy"; started: number }
  | { s: "done"; result: DescribeResult; before: string }
  | { s: "error"; message: string };

function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    // A server from before this feature answers with FastAPI's own words.
    if (e.status === 405 || (e.status === 404 && e.message === "Not Found")) return "This server can't describe shots yet. Restart Pickups to update it.";
    if (e.message) return e.message;
  }
  return "Couldn't describe the shot. Try again.";
}

/**
 * Returns the "Describe it" pill (for the field's label row, beside Enhance) and the line under the field (working,
 * error, or Undo after it filled the field). Both are null when the prompt director is turned off.
 */
export function useDescribe({ what, value, onChange }: { what: DescribeFor | null; value: string; onChange: (v: string) => void }): {
  button: ReactNode;
  panel: ReactNode;
} {
  const director = useApp((s) => s.director);
  const [st, setSt] = useState<State>({ s: "idle" });
  const [now, setNow] = useState(Date.now());
  const seq = useRef(0);
  const seed = useRef(0);
  const scope = what ? `${what.shotId}:${what.versionId ?? "orig"}:${what.field}:${what.target ?? "ltx"}` : "";

  useEffect(() => {
    if (st.s !== "busy") return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [st.s]);
  // Another take or field: what was said about the last one no longer applies.
  useEffect(() => {
    seq.current += 1;
    seed.current = 0;
    setSt({ s: "idle" });
  }, [scope]);

  if (!what || director?.backend === "off") return { button: null, panel: null };

  const run = async (again = false) => {
    const mine = ++seq.current;
    const before = st.s === "done" && again ? st.before : value;
    if (again) seed.current += 1;
    setSt({ s: "busy", started: Date.now() });
    setNow(Date.now());
    try {
      const result = await getApi().describeShot(what.shotId, {
        target: what.target ?? "ltx",
        field: what.field,
        ...(what.versionId && what.versionId !== "orig" ? { version_id: what.versionId } : {}),
        ...(again ? { refresh: true, seed: seed.current } : {}),
      });
      if (seq.current !== mine) return;
      onChange(result.text);
      setSt({ s: "done", result, before });
    } catch (e) {
      if (seq.current === mine) setSt({ s: "error", message: errorText(e) });
    }
  };
  const close = () => {
    seq.current += 1;
    setSt({ s: "idle" });
  };

  const busy = st.s === "busy";
  // Once the words are edited they're the user's own: the Undo line goes, and the button describes afresh.
  const fresh = st.s === "done" && value.trim() === st.result.text.trim();
  const button = (
    <button
      type="button"
      onClick={() => void run(fresh)}
      disabled={busy}
      title="Let AI look at the shot and describe it"
      className="inline-flex h-6 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-line px-2.5 text-[12px] text-ink2 transition-colors duration-150 hover:border-ivory/50 hover:text-ivory disabled:border-line disabled:text-ink4"
    >
      <ScanEye size={12} strokeWidth={1.5} aria-hidden="true" />
      Describe it
    </button>
  );

  const panel =
    st.s === "busy" ? (
      <div className="mt-2 flex items-center gap-2.5 px-1 text-ink3" aria-live="polite">
        <Spinner size={11} />
        <span className="font-mono text-[11px]">
          Looking at the shot… <span className="text-ink4">· {clock((now - st.started) / 1000)}</span>
        </span>
        <button type="button" onClick={close} className="ml-auto text-[12px] text-ink3 hover:text-ivory">
          Stop waiting
        </button>
      </div>
    ) : st.s === "error" ? (
      <div className="mt-2 flex items-baseline gap-2.5 rounded-[4px] border border-line2 px-3 py-2 text-[13px] text-ink" aria-live="polite">
        <span className="slate shrink-0">Error</span>
        <span className="flex-1">{st.message}</span>
        <button type="button" onClick={() => void run()} className="shrink-0 text-[12px] text-ink2 hover:text-ivory">
          Try again
        </button>
        <button type="button" aria-label="Dismiss" onClick={close} className="shrink-0 rounded-full p-0.5 text-ink3 hover:text-ivory">
          <X size={12} />
        </button>
      </div>
    ) : fresh ? (
      <p className="mt-1.5 flex flex-wrap items-baseline gap-x-3 text-[12px] leading-snug text-ink3 animate-fade-in" aria-live="polite">
        <span className="min-w-0 flex-1">
          Described from the shot <span className="text-ink4">· via</span> <span className="text-ink2">{st.result.runs_on || "ComfyUI"}</span>. Edit it as you like.
        </span>
        <button
          type="button"
          onClick={() => {
            onChange(st.before);
            close();
          }}
          className="shrink-0 text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline"
        >
          Undo
        </button>
        <button type="button" onClick={() => void run(true)} className="shrink-0 text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline">
          Try again
        </button>
      </p>
    ) : null;
  return { button, panel };
}
