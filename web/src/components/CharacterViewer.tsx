// The full-screen character viewer: one picture at a time (a sheet and its views, or the new-character options), with
// Left and Right (keys and buttons) between them, the tabs saying which one it is, and Esc or the close button to go back to the
// picture that opened it. One component for every place a picture opens large: a character's, or Previz to final's start
// frame (one item: no tabs, no arrows).
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { cx } from "../lib/format";
import { stepIndex, viewerPosition, type ViewerItem } from "../lib/viewer";
import { Spinner } from "./ui";

export type { ViewerItem };

export function CharacterViewer({
  title,
  items,
  index,
  onIndex,
  onClose,
  status,
  actions,
}: {
  /** The character's name, or "New character". */
  title: string;
  items: ViewerItem[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  /** The sheet's state ("2 of 3"), beside the title. */
  status?: ReactNode;
  /** What can be done with this picture (an option's "Use this one", a cast member's "Use <name>"). */
  actions?: (item: ViewerItem, index: number) => ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const at = Math.min(Math.max(0, index), Math.max(0, items.length - 1));
  const item = items[at];
  const many = items.length > 1;
  const live = useRef({ at, count: items.length, onIndex, onClose });
  live.current = { at, count: items.length, onIndex, onClose };

  // Focus comes in on open and goes back to the picture that opened it on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    return () => {
      if (opener && document.contains(opener)) opener.focus();
    };
  }, []);

  // Keys before anything else on the page (the player's arrows and Esc): this is on top.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const { at: i, count, onIndex: go, onClose: close } = live.current;
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close();
      } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && count > 1) {
        e.preventDefault();
        e.stopPropagation();
        go(stepIndex(i, e.key === "ArrowLeft" ? -1 : 1, count));
      } else if (e.key === "Tab" && box.current) {
        const nodes = box.current.querySelectorAll<HTMLElement>("button:not([disabled]), [href], input:not([disabled])");
        if (!nodes.length) return;
        const first = nodes[0];
        const last = nodes[nodes.length - 1];
        const inside = box.current.contains(document.activeElement);
        if (e.shiftKey && (document.activeElement === first || !inside)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && (document.activeElement === last || !inside)) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  if (!item) return null;
  const position = viewerPosition(items, at);
  return createPortal(
    <div
      ref={box}
      role="dialog"
      aria-modal="true"
      aria-label={`${title}: ${position}`}
      tabIndex={-1}
      data-viewer=""
      className="fixed inset-0 z-[70] flex flex-col bg-[#0B0A09]/[0.97] outline-none animate-fade-in"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close"
        className="absolute right-5 top-4 z-10 flex h-10 w-10 items-center justify-center rounded-full border border-line bg-panel/80 text-ink2 transition-colors hover:border-line2 hover:text-ivory"
      >
        <X size={18} />
      </button>

      <div className="relative flex min-h-0 flex-1 items-center justify-center px-24 pb-2 pt-16" onMouseDown={(e) => {
          if (e.target !== e.currentTarget) return;
          e.preventDefault(); // or the press would take focus from the picture it goes back to
          onClose();
        }}
      >
        {item.url ? (
          <img key={item.url} src={item.url} alt={`${title}, ${item.label}`} className="max-h-full max-w-full rounded-[3px] object-contain animate-fade-in" draggable={false} />
        ) : item.failed ? (
          <div className="flex aspect-[2/3] h-[60%] flex-col items-center justify-center gap-1 rounded-card border border-line bg-black" role="img" aria-label={`${item.label}: failed`}>
            <span className="slate text-ink3">{item.label}</span>
            <span className="text-[13px] text-ink4">Failed</span>
          </div>
        ) : (
          <div className="flex aspect-[2/3] h-[60%] items-center justify-center rounded-card border border-line bg-black" role="status" aria-label={`${item.label}: being made`}>
            <Spinner size={18} className="text-ink2" />
          </div>
        )}
        {many ? (
          <>
            <NavButton side="left" onClick={() => onIndex(stepIndex(at, -1, items.length))} />
            <NavButton side="right" onClick={() => onIndex(stepIndex(at, 1, items.length))} />
          </>
        ) : null}
      </div>

      <footer className="flex min-h-[80px] shrink-0 items-center gap-4 border-t border-line px-6 py-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="truncate text-[15px] text-ivory">{title}</span>
          {status ? <span className="shrink-0">{status}</span> : null}
          <span className="sr-only" aria-live="polite">
            {position}
          </span>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-3">
          {many ? (
            <div className="flex rounded-full border border-line p-1" role="tablist" aria-label="Pictures">
              {items.map((it, i) => (
                <button
                  key={it.key}
                  type="button"
                  role="tab"
                  aria-selected={i === at}
                  onClick={() => onIndex(i)}
                  className={cx(
                    "h-8 rounded-full px-3.5 text-[13px] transition-colors duration-150",
                    i === at ? "bg-ivory text-[#0D0C0B]" : "text-ink2 hover:text-ivory",
                  )}
                >
                  {it.label}
                </button>
              ))}
            </div>
          ) : null}
          {actions ? actions(item, at) : null}
        </div>
      </footer>
    </div>,
    document.body,
  );
}

function NavButton({ side, onClick }: { side: "left" | "right"; onClick: () => void }) {
  const Icon = side === "left" ? ChevronLeft : ChevronRight;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={side === "left" ? "Previous" : "Next"}
      className={cx(
        "absolute top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-line bg-panel/80 text-ink2 transition-colors hover:border-line2 hover:text-ivory",
        side === "left" ? "left-5" : "right-5",
      )}
    >
      <Icon size={20} />
    </button>
  );
}
