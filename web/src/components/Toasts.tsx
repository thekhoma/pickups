import { Check, Clock, X } from "lucide-react";
import { dismissToast, holdToast } from "../actions";
import { cx } from "../lib/format";
import { Button } from "./ui";
import { useApp } from "../store";

/** "Done" only for finished work; anything sent to the GPU queue says "Queued" until it lands. */
const LABEL = { error: "Error", success: "Done", info: "Note", queued: "Queued" } as const;
/** Done is green and Queued is yellow (the queue's next-up colour), so a finished render or a queued run reads at a glance. */
const TONE = {
  success: "border-[#3FB950]/70 bg-[#13251A]",
  queued: "border-[#E3A34B]/70 bg-[#2A2112]",
  error: "border-line2 bg-raised",
  info: "border-line2 bg-raised",
} as const;

export function Toasts() {
  const toasts = useApp((s) => s.toasts);
  // Bottom right on every page. On a shot that corner is the task rail's button (RightRail.tsx: 392 px, 348 px below
  // 1180 px; full width on a phone), so the stack sits at the bottom right of the picture instead.
  const onShot = useApp((s) => s.route.name === "shot");
  return (
    <div
      className={cx(
        "pointer-events-none fixed bottom-14 right-6 z-[60] flex w-[380px] max-w-[calc(100vw-48px)] flex-col gap-2",
        onShot && "min-[768px]:right-[416px] max-[1180px]:min-[768px]:right-[372px]",
      )}
      aria-live="polite"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.tone === "error" ? "alert" : "status"}
          onPointerEnter={() => holdToast(t.id, true)}
          onPointerLeave={() => holdToast(t.id, false)}
          onFocus={() => holdToast(t.id, true)}
          onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) holdToast(t.id, false);
          }}
          className={cx("pointer-events-auto flex items-center gap-3 rounded-card border py-3 pl-4 pr-2 animate-rise-in", TONE[t.tone])}
        >
          <span className="sr-only">{LABEL[t.tone]}</span>
          {t.tone === "success" ? <Check size={16} strokeWidth={2.5} className="shrink-0 text-[#3FB950]" aria-hidden="true" /> : null}
          {t.tone === "queued" ? <Clock size={16} strokeWidth={2} className="shrink-0 text-[#E3A34B]" aria-hidden="true" /> : null}
          {/* The whole message is the action too: a finished render opens with one click anywhere on it. */}
          {t.action ? (
            <button
              type="button"
              className="min-w-0 flex-1 text-left text-[14px] text-ivory hover:underline"
              onClick={() => {
                t.action!.run();
                dismissToast(t.id);
              }}
            >
              {t.text}
            </button>
          ) : (
            <span className="min-w-0 flex-1 text-[14px] text-ink">{t.text}</span>
          )}
          {t.action ? (
            <Button
              variant={t.tone === "success" ? "go" : "primary"}
              size="sm"
              className="shrink-0"
              onClick={() => {
                t.action!.run();
                dismissToast(t.id);
              }}
            >
              {t.action.label}
            </Button>
          ) : null}
          <button type="button" onClick={() => dismissToast(t.id)} aria-label="Dismiss" className="shrink-0 rounded-full p-1 text-ink3 transition-colors hover:text-ivory">
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
