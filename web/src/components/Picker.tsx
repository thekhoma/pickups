// The composer's pickers (Model, Aspect, Size, Length): a pill that opens a list. The list is portaled to the page and
// placed against the window, so no parent can clip it: it opens below when it fits, else above (the docked composer),
// else on the roomier side with a max height and a scroll. The picked row is scrolled into view.
import { ChevronDown } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { cx } from "../lib/format";
import { placeMenu, type MenuPlace } from "../lib/placeMenu";

export interface PickerOption {
  id: string;
  label: string;
  sub?: string;
  left?: ReactNode;
  disabled?: boolean;
}

export function Picker({ label, value, options, onChange }: { label: string; value: string; options: PickerOption[]; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  const [box, setBox] = useState<MenuPlace | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const cur = options.find((o) => o.id === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (wrap.current?.contains(t) || menu.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setOpen(false);
      btn.current?.focus();
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) {
      setBox(null);
      return;
    }
    const place = () => {
      const r = btn.current?.getBoundingClientRect();
      const m = menu.current;
      if (!r || !m) return;
      setBox(placeMenu(r, { height: m.scrollHeight, width: m.offsetWidth }, { width: window.innerWidth, height: window.innerHeight }));
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, options.length]);

  const shown = useRef(false); // the picked row is brought into view once per opening, never against the user's scroll
  useEffect(() => {
    if (!box) {
      shown.current = false;
      return;
    }
    if (shown.current) return;
    shown.current = true;
    menu.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  }, [box]);

  return (
    <div ref={wrap} className="relative">
      <button
        ref={btn}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`${label}: ${cur?.label ?? ""}`}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex h-9 items-center gap-2 rounded-full border border-line px-3.5 text-[13px] text-ink transition-colors hover:border-line2 hover:text-ivory"
      >
        <span className="slate text-ink4">{label}</span>
        {cur?.left}
        <span className="max-w-[160px] truncate">{cur?.label}</span>
        <ChevronDown size={13} strokeWidth={1.5} className={cx("text-ink3 transition-transform duration-150", open && "rotate-180")} aria-hidden="true" />
      </button>
      {open
        ? createPortal(
            <div
              ref={menu}
              role="listbox"
              aria-label={label}
              className="fixed z-[80] min-w-[200px] overflow-y-auto overscroll-contain rounded-card border border-line2 bg-raised py-1 shadow-[0_24px_60px_-24px_rgba(0,0,0,0.9)]"
              style={
                box
                  ? { left: box.left, top: box.top, bottom: box.bottom, maxHeight: box.maxH }
                  : { left: 0, top: 0, visibility: "hidden" } // measured first, then placed
              }
            >
              {options.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  role="option"
                  aria-selected={o.id === value}
                  aria-disabled={o.disabled || undefined}
                  disabled={o.disabled}
                  onClick={() => {
                    if (o.disabled) return;
                    onChange(o.id);
                    setOpen(false);
                  }}
                  className={cx(
                    "flex w-full items-center gap-3 whitespace-nowrap px-3.5 py-2 text-left text-[13px] transition-colors hover:bg-line focus-visible:bg-line disabled:text-ink4 disabled:hover:bg-transparent",
                    o.id === value ? "text-ivory" : "text-ink",
                  )}
                >
                  {o.left}
                  <span className="flex-1">{o.label}</span>
                  {o.sub ? <span className="slate text-ink3">{o.sub}</span> : null}
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

/** A small frame drawn at an aspect ratio ("16:9"), for aspect pickers. */
export function FrameIcon({ aspect }: { aspect: string }) {
  const [w, h] = aspect.split(":").map(Number);
  const k = 12 / Math.max(w, h);
  return <span aria-hidden="true" className="inline-block shrink-0 rounded-[1px] border border-current" style={{ width: Math.max(4, w * k), height: Math.max(4, h * k) }} />;
}
