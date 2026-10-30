// The take panel's extras: the ComfyUI workflows a take ran, social cut-downs ("Share") and its receipt sheet.
import { ArrowUpRight, Download as DownloadIcon, ReceiptText, Share } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../api";
import { getApi } from "../actions";
import { clock, cx } from "../lib/format";
import { useApp } from "../store";
import type { ShareFormat, Shot, TakeWorkflow, Version } from "../types";
import { Hint, Icon, MenuPill, Notice, ProgressLine, Spinner } from "./ui";

/** A take's ComfyUI workflows as two actions: Open in ComfyUI (through Pickups' /comfy/, so the browser needn't reach the
 * GPU machine itself) and Download workflows (one zip; files one by one where there's no zip, as in demo mode). Drag a
 * JSON onto ComfyUI to tweak it. Hidden for takes from before workflows were kept. */
export function ComfyWorkflows({ shotId, version }: { shotId: string; version: Version }) {
  const comfyUrl = useApp((s) => s.comfyUrl || s.status?.comfy_url || null);
  const demo = useApp((s) => s.mode === "mock");
  const workflows = (version.workflows ?? []).filter((w) => w && w.url);
  if (!workflows.length) return null;
  const zip = (() => {
    try {
      return getApi().takeWorkflowsUrl(shotId, version.id);
    } catch {
      return null;
    }
  })();
  const one = (w: TakeWorkflow) => {
    const a = document.createElement("a");
    a.href = w.url;
    a.download = w.filename || `${w.step}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const pill = "inline-flex h-8 items-center gap-1.5 whitespace-nowrap rounded-full border border-line2 px-3.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ivory/50 hover:text-ivory";
  return (
    <section aria-label="ComfyUI workflows" className="flex flex-wrap items-center gap-2">
      {comfyUrl && !demo ? (
        <a href="/comfy/" target="_blank" rel="noopener noreferrer" className={pill}>
          Open in ComfyUI
          <ArrowUpRight size={13} strokeWidth={1.5} className="text-ink3" aria-hidden="true" />
        </a>
      ) : null}
      {zip ? (
        <a href={zip} download className={pill} aria-label={`Download workflows (${workflows.length})`}>
          <DownloadIcon size={14} strokeWidth={1.5} aria-hidden="true" />
          Download workflows
        </a>
      ) : (
        <button type="button" className={pill} aria-label={`Download workflows (${workflows.length})`} onClick={() => workflows.forEach((w, i) => window.setTimeout(() => one(w), i * 250))}>
          <DownloadIcon size={14} strokeWidth={1.5} aria-hidden="true" />
          Download workflows
        </button>
      )}
    </section>
  );
}

/** "Receipt sheet ↗": the take's receipt sheet in a new tab (what changed, models and licenses, prompts, timecode, files
 * and workflows on one printable page), to keep or hand to a client. Not the transport's Receipt view (R), which tints
 * the picture. A pill beside Share; a quiet link on Deliveries rows. */
export function ReceiptLink({ shotId, version, variant = "pill" }: { shotId: string; version: Version; variant?: "pill" | "link" }) {
  const url = useMemo(() => {
    try {
      return getApi().takeReceiptUrl(shotId, version.id);
    } catch {
      return null;
    }
  }, [shotId, version]);
  if (!url) return null;
  const title = "Printable: what changed, models, licenses, prompts and files";
  if (variant === "link") {
    return (
      <a href={url} target="_blank" rel="noopener" title={title} className="inline-flex items-center gap-1 text-[12px] text-ink3 underline-offset-4 hover:text-ivory hover:underline">
        Receipt sheet
        <ArrowUpRight size={12} strokeWidth={1.5} aria-hidden="true" />
      </a>
    );
  }
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener"
      title={title}
      className="inline-flex h-8 select-none items-center gap-1.5 whitespace-nowrap rounded-full border border-line2 px-3.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ivory/50 hover:text-ivory"
    >
      <Icon icon={ReceiptText} size={16} />
      Receipt sheet
      <ArrowUpRight size={13} strokeWidth={1.5} className="text-ink3" aria-hidden="true" />
    </a>
  );
}

const FORMATS: Record<ShareFormat, { title: string; note: string; aspect: string; width: string }> = {
  "9x16": { title: "9:16 stacked", note: "(vertical)", aspect: "aspect-[9/16]", width: "w-[112px]" },
  "1x1": { title: "1:1 wipe", note: "(feeds)", aspect: "aspect-square", width: "w-[148px]" },
};

type ShareState =
  | { kind: "idle" }
  | { kind: "rendering"; format: ShareFormat; started: number }
  | { kind: "done"; format: ShareFormat; url: string; filename: string }
  | { kind: "error"; format: ShareFormat; text: string };

/** "Share ▾": renders a social cut-down of this take (a few seconds), previews it, offers the file. Its receipt sits beside it. */
export function ShareControl({ shot, version }: { shot: Shot; version: Version }) {
  const [st, setSt] = useState<ShareState>({ kind: "idle" });
  const [now, setNow] = useState(() => Date.now());
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    if (st.kind !== "rendering") return;
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(t);
  }, [st.kind]);

  const render = async (format: ShareFormat) => {
    const started = Date.now();
    setNow(started);
    setSt({ kind: "rendering", format, started });
    try {
      const r = await getApi().shareTake(shot.id, version.id, format);
      if (!alive.current) return;
      const fallback = `${shot.name.replace(/[^A-Za-z0-9]+/g, "_")}_${format}.mp4`;
      setSt({ kind: "done", format, url: r.url, filename: r.filename || fallback });
    } catch (e) {
      if (!alive.current) return;
      setSt({ kind: "error", format, text: e instanceof ApiError ? e.message : "That didn't render. Try again." });
    }
  };

  const items = (Object.keys(FORMATS) as ShareFormat[]).map((f) => ({
    id: f,
    label: (
      <>
        {FORMATS[f].title} <span className="text-ink3">{FORMATS[f].note}</span>
      </>
    ),
    onSelect: () => void render(f),
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <MenuPill label="Share" icon={Share} items={items} size="sm" loading={st.kind === "rendering"} menuLabel="Share as" />
        <Hint text="Makes a short clip for social posts: the before and after stacked (9:16) or as a wipe (1:1). Download it when it's done." />
        <ReceiptLink shotId={shot.id} version={version} />
      </div>
      {st.kind === "rendering" ? (
        <div className="rounded-card border border-line px-4 py-3" role="status" aria-live="polite">
          <div className="flex items-center gap-2 font-mono text-[12px] text-ink2">
            <Spinner size={12} />
            <span>Rendering…</span>
            <span className="text-ink3">{clock(Math.max(0, (now - st.started) / 1000))}</span>
            <span className="ml-auto text-ink3">{FORMATS[st.format].title}</span>
          </div>
          <ProgressLine value={0} idle className="mt-3" />
        </div>
      ) : st.kind === "done" ? (
        <div className="flex animate-rise-in gap-4 rounded-card border border-line p-3">
          <video
            key={st.url}
            src={st.url}
            muted
            loop
            autoPlay
            playsInline
            className={cx("shrink-0 rounded-[3px] bg-black object-cover", FORMATS[st.format].aspect, FORMATS[st.format].width)}
            aria-label={`${FORMATS[st.format].title} preview`}
          />
          <div className="flex min-w-0 flex-1 flex-col justify-between gap-3 py-0.5">
            <div className="min-w-0">
              <div className="slate text-ink3">{FORMATS[st.format].title}</div>
              <div className="mt-1 truncate font-mono text-[11px] text-ink2" title={st.filename}>
                {st.filename}
              </div>
            </div>
            <div className="flex flex-col items-start gap-1">
              <a
                href={st.url}
                download={st.filename}
                className="inline-flex h-8 items-center gap-1.5 rounded-full border border-line2 px-3.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ivory/50 hover:text-ivory"
              >
                <DownloadIcon size={15} strokeWidth={1.75} aria-hidden="true" />
                Download
              </a>
              <button type="button" onClick={() => setSt({ kind: "idle" })} className="h-7 text-[12px] text-ink3 transition-colors hover:text-ivory">
                Close
              </button>
            </div>
          </div>
        </div>
      ) : st.kind === "error" ? (
        <Notice stack tone="bad" text={st.text} action={{ label: "Try again", onClick: () => void render(st.format) }} secondary={{ label: "Close", onClick: () => setSt({ kind: "idle" }) }} />
      ) : null}
    </div>
  );
}
