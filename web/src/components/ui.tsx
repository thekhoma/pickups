// Pickups UI kit, "the cutting room" (design v2.1).
// Pills are actions, 6 px rectangles are containers. Ivory 100% only for headlines and the primary pill.
// Tally red is for dots only. Errors carry no colour: ivory text, a mono ERROR slate and a hairline.
import { Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Lock, LockOpen, Plus, X, type LucideIcon } from "lucide-react";
import { createPortal } from "react-dom";
import {
  forwardRef,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type DragEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { cx, percentText, receiptPercents, receiptShares } from "../lib/format";
import type { Receipt } from "../types";

// --- icons ----------------------------------------------------------------------------------------------------------

export function Icon({ icon: I, size = 18, className }: { icon: LucideIcon; size?: number; className?: string }) {
  return <I size={size} strokeWidth={1.5} className={className} aria-hidden="true" />;
}

export function Spinner({ size = 14, className }: { size?: number; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cx("inline-block shrink-0 animate-spin rounded-full border border-current border-r-transparent", className)}
      style={{ width: size, height: size, borderWidth: Math.max(1, Math.round(size / 10)) }}
    />
  );
}

// --- slate & dots ----------------------------------------------------------------------------------------------------

type SlatePos = "tl" | "tr" | "bl" | "br" | "bc" | "tc";
const POS: Record<SlatePos, string> = {
  tl: "left-3 top-3",
  tr: "right-3 top-3",
  bl: "bottom-3 left-3",
  br: "bottom-3 right-3",
  bc: "bottom-3 left-1/2 -translate-x-1/2",
  tc: "top-3 left-1/2 -translate-x-1/2",
};

/**
 * Burn-in slate: 11 px mono, 0.08em tracking, ivory 70%. Over media (`at`) it sits 12 px in, on a black burn-in box.
 * A tiny tally dot is lit only on the active take.
 */
export function Slate({
  children,
  live,
  at,
  className,
}: {
  children: ReactNode;
  live?: boolean;
  at?: SlatePos;
  className?: string;
}) {
  // An array of plain strings is a slate line ("TAKE 02 · REMOVE"); anything else renders as given.
  const parts = Array.isArray(children) ? children : null;
  const plain = parts?.every((c) => c == null || c === false || typeof c === "string" || typeof c === "number");
  const text = parts && plain ? parts.filter((c) => c != null && c !== false && c !== "").join(" · ") : children;
  return (
    <span
      className={cx(
        "slate inline-flex items-center gap-1.5 whitespace-nowrap",
        at && "pointer-events-none absolute z-[2] rounded-[2px] bg-black/60 px-1.5 py-[3px]",
        at && POS[at],
        className,
      )}
    >
      {live ? <TallyDot size={5} /> : null}
      {text}
    </span>
  );
}

/** The tally light. Lit = red with a 1 px bloom; unlit = dark red-brown. */
export function TallyDot({ lit = true, size = 6, className }: { lit?: boolean; size?: number; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cx("inline-block shrink-0 rounded-full transition-[background-color,box-shadow] duration-150", lit ? "tally" : "bg-unlit", className)}
      style={{ width: size, height: size }}
    />
  );
}

/** Neutral state dot (no colour): on = ivory, off = hollow. */
export function StateDot({ on, size = 6 }: { on: boolean; size?: number }) {
  return (
    <span
      aria-hidden="true"
      className={cx("inline-block shrink-0 rounded-full", on ? "bg-ivory/80" : "border border-ink3")}
      style={{ width: size, height: size }}
    />
  );
}

// --- buttons --------------------------------------------------------------------------------------------------------

type Variant = "primary" | "secondary" | "ghost" | "quiet" | "danger" | "go";
type Size = "lg" | "md" | "sm";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  iconRight?: LucideIcon;
  loading?: boolean;
  block?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  primary:
    "rounded-full bg-ivory text-[#0D0C0B] font-medium hover:bg-ivory/90 active:translate-y-px disabled:bg-raised disabled:text-ink4 border border-transparent",
  secondary:
    "rounded-full border border-line2 text-ink font-medium hover:border-ivory/50 hover:text-ivory active:translate-y-px disabled:text-ink4 disabled:hover:border-line2",
  ghost: "rounded-full text-ink2 font-medium hover:text-ivory hover:bg-raised border border-transparent disabled:text-ink4 disabled:hover:bg-transparent",
  quiet: "rounded-[4px] text-ink2 font-medium hover:text-ivory border border-transparent disabled:text-ink4",
  /** The one next step that must stand out (Rebuild after a camera change, Upres to 4K / HDR): green. */
  go: "rounded-full bg-[#3FB950] text-[#0D0C0B] font-medium hover:bg-[#3FB950]/90 active:translate-y-px disabled:bg-raised disabled:text-ink4 border border-transparent",
  danger:
    "rounded-full border border-tally/70 text-tally font-medium hover:border-tally hover:bg-tally hover:text-ivory active:translate-y-px disabled:border-line2 disabled:text-ink4",
};
const SIZES: Record<Size, string> = {
  lg: "h-12 px-6 text-[15px] gap-2",
  md: "h-10 px-4 text-[14px] gap-2",
  sm: "h-9 px-3.5 text-[13px] gap-1.5",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, iconRight, loading, block, className, children, disabled, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cx(
        "inline-flex select-none items-center justify-center whitespace-nowrap transition-[background-color,border-color,color,transform] duration-150 ease-out",
        VARIANTS[variant],
        variant === "quiet" ? "h-8 gap-1.5 px-0 text-[13px]" : SIZES[size],
        block && "w-full",
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner size={size === "lg" ? 15 : 13} /> : icon ? <Icon icon={icon} size={size === "lg" ? 18 : 16} /> : null}
      {children}
      {iconRight && !loading ? <Icon icon={iconRight} size={size === "lg" ? 18 : 16} /> : null}
    </button>
  );
});

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: LucideIcon;
  label: string;
  hint?: string;
  active?: boolean;
  size?: "sm" | "md";
  tipAlign?: "center" | "end" | "start";
  /** No tooltip for now (e.g. while what the button opened sits where the tip would). */
  tipHidden?: boolean;
}

export function IconButton({ icon, label, hint, active, size = "md", tipAlign = "center", tipHidden, className, type = "button", ...rest }: IconButtonProps) {
  return (
    <Tip text={hint ? `${label} · ${hint}` : label} align={tipAlign} hidden={tipHidden}>
      <button
        type={type}
        aria-label={label}
        aria-pressed={active}
        className={cx(
          "inline-flex items-center justify-center rounded-full transition-colors duration-150 ease-out disabled:text-ink4 disabled:hover:bg-transparent",
          size === "sm" ? "h-8 w-8" : "h-9 w-9",
          active ? "bg-raised text-ivory shadow-[inset_0_0_0_1px_#3A3631]" : "text-ink2 hover:bg-raised hover:text-ivory",
          className,
        )}
        {...rest}
      >
        <Icon icon={icon} size={size === "sm" ? 16 : 18} />
      </button>
    </Tip>
  );
}

// --- tooltip ----------------------------------------------------------------------------------------------------------

export function Tip({
  text,
  children,
  side = "top",
  align = "center",
  className,
  hidden,
}: {
  text: string;
  children: ReactNode;
  side?: "top" | "bottom";
  align?: "center" | "end" | "start";
  hidden?: boolean;
  /** For the wrapper, e.g. min-w-0 so a truncating child can shrink in a flex row. */
  className?: string;
}) {
  return (
    <span className={cx("group/tip relative inline-flex", className)}>
      {children}
      <span
        role="tooltip"
        className={cx(
          "pointer-events-none absolute z-50 whitespace-nowrap rounded-[4px] border border-line bg-raised px-2 py-1 font-mono text-[11px] uppercase tracking-[0.06em] text-ink opacity-0 transition-opacity duration-150 ease-out group-hover/tip:opacity-100 group-hover/tip:delay-300 group-focus-within/tip:opacity-0",
          side === "top" ? "bottom-full mb-2" : "top-full mt-2",
          align === "center" ? "left-1/2 -translate-x-1/2" : align === "end" ? "right-0" : "left-0",
          hidden && "hidden",
        )}
      >
        {text}
      </span>
    </span>
  );
}

// --- hint -------------------------------------------------------------------------------------------------------------

/**
 * A small "?" that explains a control on demand, so the form itself stays short. Hover or keyboard focus shows it,
 * a tap or click pins it (tap again, Escape or a click elsewhere closes it). The text is the button's description.
 * Never put it inside another button or a <label> (use Field's `hint`).
 */
export function Hint({
  text,
  label = "More info",
  className,
  children,
}: {
  text: ReactNode;
  label?: string;
  className?: string;
  /** A trigger of its own (e.g. a chip) instead of the "?"; `className` then styles it whole. */
  children?: ReactNode;
}) {
  const id = useId();
  const btn = useRef<HTMLButtonElement>(null);
  const tip = useRef<HTMLSpanElement>(null);
  const [hover, setHover] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [at, setAt] = useState<{ left: number; top: number } | null>(null);
  const shown = hover || pinned;

  useLayoutEffect(() => {
    if (!shown) return setAt(null);
    const place = () => {
      const b = btn.current?.getBoundingClientRect();
      const t = tip.current?.getBoundingClientRect();
      if (!b || !t) return;
      const m = 8;
      const left = Math.min(Math.max(m, b.left + b.width / 2 - t.width / 2), window.innerWidth - t.width - m);
      const above = b.top - t.height - 8;
      setAt({ left, top: above >= m ? above : b.bottom + 8 });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [shown]);

  useEffect(() => {
    if (!shown) return;
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setHover(false);
      setPinned(false);
    };
    const down = (e: PointerEvent) => {
      if (btn.current?.contains(e.target as Node)) return;
      setPinned(false);
      setHover(false);
    };
    document.addEventListener("keydown", key, true);
    document.addEventListener("pointerdown", down, true);
    return () => {
      document.removeEventListener("keydown", key, true);
      document.removeEventListener("pointerdown", down, true);
    };
  }, [shown]);

  return (
    <>
      <button
        ref={btn}
        type="button"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={shown}
        onPointerEnter={(e) => e.pointerType === "mouse" && setHover(true)}
        onPointerLeave={(e) => e.pointerType === "mouse" && setHover(false)}
        onFocus={(e) => e.currentTarget.matches(":focus-visible") && setHover(true)}
        onBlur={() => {
          setHover(false);
          setPinned(false);
        }}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setPinned((p) => !p);
          if (pinned) setHover(false);
        }}
        className={
          children
            ? className
            : cx(
                "relative inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full align-middle text-ink3 transition-colors duration-150 before:absolute before:-inset-1.5 before:content-[''] hover:text-ivory focus-visible:text-ivory",
                shown && "text-ivory",
                className,
              )
        }
      >
        {children ?? <CircleHelp size={15} strokeWidth={1.75} aria-hidden="true" />}
        <span id={id} className="sr-only">
          {text}
        </span>
      </button>
      {shown
        ? createPortal(
            <span
              ref={tip}
              aria-hidden="true"
              className="pointer-events-none fixed z-[200] block max-w-[min(280px,calc(100vw-16px))] rounded-[6px] border border-line2 bg-raised px-3 py-2 text-left text-[13px] font-normal normal-case leading-snug tracking-normal text-ink shadow-[0_8px_24px_rgba(0,0,0,.45)] animate-fade-in"
              style={at ?? { left: -9999, top: 0 }}
            >
              {text}
            </span>,
            document.body,
          )
        : null}
    </>
  );
}

// --- toggles ----------------------------------------------------------------------------------------------------------

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        "relative inline-flex h-[18px] w-8 shrink-0 items-center rounded-full border transition-colors duration-150 ease-out",
        checked ? "border-ivory bg-ivory" : "border-line2 bg-transparent",
      )}
    >
      <span
        className={cx(
          "absolute h-3 w-3 rounded-full transition-transform duration-150 ease-out",
          checked ? "translate-x-[15px] bg-[#0D0C0B]" : "translate-x-[2px] bg-ink3",
        )}
      />
    </button>
  );
}

export function SwitchRow({ checked, onChange, title, sub, hint }: { checked: boolean; onChange: (v: boolean) => void; title: string; sub?: string; hint?: ReactNode }) {
  const id = useId();
  return (
    <div className="flex items-center justify-between gap-4 py-1">
      <span className="flex items-center gap-1.5">
        <label htmlFor={id} className="cursor-pointer" onClick={() => onChange(!checked)}>
          <div className="text-[14px] text-ink">{title}</div>
          {sub ? <div className="text-[13px] text-ink3">{sub}</div> : null}
        </label>
        {hint ? <Hint text={hint} label={`About ${title.toLowerCase()}`} /> : null}
      </span>
      <span id={id}>
        <Switch checked={checked} onChange={onChange} label={title} />
      </span>
    </div>
  );
}

function CheckBox({ checked }: { checked: boolean }) {
  return (
    <span
      className={cx(
        "flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border transition-colors duration-150",
        checked ? "border-ivory bg-ivory text-[#0D0C0B]" : "border-line2 text-transparent",
      )}
      aria-hidden="true"
    >
      <Check size={11} strokeWidth={3} />
    </span>
  );
}

/** On/off option as a 6 px container (Finish's 4K / HDR). */
export function ToggleCard({ checked, onChange, title, sub }: { checked: boolean; onChange: (v: boolean) => void; title: string; sub: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={`${title}: ${sub}`}
      onClick={() => onChange(!checked)}
      className={cx(
        "flex w-full items-start gap-3.5 rounded-card border px-4 py-3.5 text-left transition-colors duration-150 ease-out",
        checked ? "border-ivory/50 bg-raised" : "border-line hover:border-line2",
      )}
    >
      <span className="mt-1">
        <CheckBox checked={checked} />
      </span>
      <span className="min-w-0 flex-1">
        <span className={cx("block font-mono text-[14px] font-medium tracking-[0.02em]", checked ? "text-ivory" : "text-ink2")}>{title}</span>
        <span className="mt-0.5 block text-[13px] text-ink3">{sub}</span>
      </span>
    </button>
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
  disabled?: boolean;
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  size = "md",
  disabled,
  block,
  mono,
}: {
  value: T;
  onChange: (v: T) => void;
  options: SegOption<T>[];
  label: string;
  size?: "sm" | "md";
  disabled?: boolean;
  block?: boolean;
  /** Small mono caps (grid sort / filter). */
  mono?: boolean;
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cx("inline-flex rounded-full border border-line p-[2px]", block && "flex w-full", disabled && "opacity-40")}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={disabled || o.disabled}
            onClick={() => onChange(o.value)}
            className={cx(
              "inline-flex items-center justify-center gap-1.5 rounded-full transition-colors duration-150 ease-out disabled:text-ink4",
              mono ? "h-6 px-2.5 font-mono text-[10px] uppercase tracking-[0.08em]" : size === "sm" ? "h-7 px-3 text-[12px]" : "h-8 px-4 text-[13px]",
              block && "flex-1",
              on ? "bg-raised text-ivory shadow-[inset_0_0_0_1px_#3A3631]" : "text-ink3 hover:text-ink",
            )}
          >
            {o.label}
            {o.hint ? <span className="kbd">{o.hint}</span> : null}
          </button>
        );
      })}
    </div>
  );
}

/** Small mono label in a hairline box (BETA, step chips). A container, not an action. */
export function Tag({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex h-[18px] items-center rounded-[3px] border border-line2 px-1.5 font-mono text-[10px] uppercase tracking-[0.08em] text-ink2", className)}>
      {children}
    </span>
  );
}

/** A small rounded status pill, e.g. "Beta" on Previz. Green: thin outline, green text, a faint tint; nothing moves. */
export function Badge({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex h-[15px] shrink-0 items-center rounded-full border border-[#3FB950]/40 bg-[#3FB950]/[0.08] px-1.5 font-sans text-[10px] font-normal normal-case leading-none tracking-normal text-[#3FB950]", className)}>
      {children}
    </span>
  );
}

/** A long path or name on one line, cut in the middle so both ends stay readable; the whole text on hover. */
/**
 * A long path with its middle cut: "~/Pickups/…/home/library". The head gives way first, down to its first 8
 * characters, then the tail does, losing its own left end ("…home/library"). So both ends always show, even on a
 * phone, where the old fixed tail pushed the start of the path off the left edge.
 */
export function MiddleTruncate({ text, tail = 28, className }: { text: string; tail?: number; className?: string }) {
  const cut = Math.max(0, text.length - tail);
  return (
    <span className={cx("flex min-w-0", className)} title={text}>
      {/* The head takes no room of its own (basis 0, at least 8 characters) and grows into what the tail leaves. */}
      {cut ? <span className="min-w-[8ch] flex-1 basis-0 truncate">{text.slice(0, cut)}</span> : null}
      <span className="min-w-0 shrink truncate whitespace-pre [direction:rtl]">
        <span dir="ltr">{text.slice(cut)}</span>
      </span>
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton rounded-card", className)} aria-hidden="true" />;
}

/**
 * Empty / error state: one sentence + one action, no colour. A mono slate says what kind it is
 * (ERROR, SETUP, …). `stack` puts the actions under the sentence for narrow rails.
 */
export function Notice({
  text,
  tone = "neutral",
  label,
  action,
  secondary,
  className,
  stack,
}: {
  text: ReactNode;
  tone?: "neutral" | "warn" | "bad";
  label?: string;
  icon?: LucideIcon;
  action?: { label: string; onClick: () => void; loading?: boolean };
  secondary?: { label: string; onClick: () => void };
  className?: string;
  stack?: boolean;
}) {
  const slate = label ?? (tone === "bad" ? "Error" : tone === "warn" ? "Setup" : null);
  const buttons =
    action || secondary ? (
      <div className="flex shrink-0 items-center gap-1.5">
        {action ? (
          <Button variant="secondary" size="sm" onClick={action.onClick} loading={action.loading}>
            {action.label}
          </Button>
        ) : null}
        {secondary ? (
          <Button variant="ghost" size="sm" onClick={secondary.onClick}>
            {secondary.label}
          </Button>
        ) : null}
      </div>
    ) : null;
  return (
    <div
      role={tone === "bad" ? "alert" : "status"}
      className={cx(
        "rounded-card border",
        tone === "neutral" ? "border-line" : "border-line2",
        stack ? "space-y-3 p-4" : "flex items-center gap-4 px-4 py-3",
        className,
      )}
    >
      <div className={cx("min-w-0", stack ? "space-y-1.5" : "flex flex-1 items-baseline gap-3")}>
        {slate ? <span className="slate block shrink-0 text-ivory/70">{slate}</span> : null}
        <div className="min-w-0 flex-1 text-[14px] leading-snug text-ink">{text}</div>
      </div>
      {buttons}
    </div>
  );
}

// --- progress ---------------------------------------------------------------------------------------------------------

/** The record line: thin tally red over a hairline. */
/** Lock seed (lib/seed.ts): one small pill with the seed the last run used. Pressed, the next run uses it again; off,
 * each run gets a fresh one. Before any run it reads "Seed" alone. */
export function SeedLock({ seed, locked, onChange, size = "sm", className, testId = "seed-lock" }: {
  seed: number | null | undefined; locked: boolean; onChange: (locked: boolean) => void;
  /** sm: 28 px, beside chips; md: 36 px, in a row of pickers (the composer). */
  size?: "sm" | "md"; className?: string; testId?: string;
}) {
  // Nothing to lock before a run has used a seed: the pill comes with the first one (never pressed with no seed).
  if (seed == null) return null;
  const I = locked ? Lock : LockOpen;
  return (
    <button
      type="button"
      aria-pressed={locked}
      aria-label={`Lock seed ${seed}`}
      data-testid={testId}
      onClick={() => onChange(!locked)}
      className={cx(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border transition-colors duration-150",
        size === "md" ? "h-9 px-3.5 text-[12px]" : "h-7 px-2.5 text-[11px]",
        locked ? "border-ivory/60 text-ivory" : "border-line text-ink3 hover:border-line2 hover:text-ink",
        className,
      )}
    >
      <I size={12} strokeWidth={1.75} aria-hidden="true" />
      <span className="mono">Seed {seed}</span>
    </button>
  );
}

export function ProgressLine({ value, className, idle }: { value: number; className?: string; idle?: boolean }) {
  return (
    <div
      className={cx("relative h-[2px] w-full overflow-hidden rounded-full bg-line", className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}
    >
      <div
        className={cx("absolute inset-y-0 left-0", idle ? "bg-ink4" : "bg-tally shadow-[0_0_6px_rgba(245,48,30,0.55)]")}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, transition: "width 400ms linear" }}
      />
    </div>
  );
}

// --- receipt -----------------------------------------------------------------------------------------------------------

/** Bar colours per share: Finish's upscale (the footage resampled, no new content) is a neutral grey, never AI. */
const SHARE_BG = { original: "bg-ivory/40", borrowed: "bg-borrowed", upscaled: "bg-ink2", ai: "bg-ai" } as const;

export function ReceiptBar({ receipt, className }: { receipt?: Receipt | null; className?: string }) {
  const p = receiptPercents(receipt);
  if (!p) return null;
  return (
    <div className={className}>
      <div className="flex h-[3px] w-full overflow-hidden rounded-full bg-line" aria-hidden="true">
        {(["original", "borrowed", "upscaled", "ai"] as const).map((k) =>
          p[k] ? <div key={k} className={cx("h-full", SHARE_BG[k])} style={{ width: `${p[k]}%` }} /> : null,
        )}
      </div>
      <div className="mono mt-2.5 flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-ink2">
        {receiptShares(receipt).map((s) =>
          s.key === "original" ? (
            <span key={s.key} className="whitespace-nowrap">
              <span className="text-ivory">{s.text}</span> original
            </span>
          ) : (
            <span key={s.key} className="whitespace-nowrap">
              <span className={cx("mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle", SHARE_BG[s.key])} />
              {s.text} {s.key === "ai" ? "AI" : s.label.toLowerCase()}
            </span>
          ),
        )}
      </div>
    </div>
  );
}

export function ReceiptLegend({ receipt }: { receipt?: Receipt | null }) {
  const p = receiptPercents(receipt);
  const item = (swatch: string, label: string, key: "original" | "borrowed" | "upscaled" | "ai") => (
    <span className="flex items-center gap-1.5">
      <span className={cx("h-2 w-2 rounded-full", swatch)} />
      {label}
      {p ? <span className="text-ivory/50">{percentText(receipt, key)}</span> : null}
    </span>
  );
  return (
    <div className="slate flex items-center gap-4 rounded-[2px] bg-black/70 px-2.5 py-1.5">
      {item("border border-ivory/70", "Original", "original")}
      {item("bg-borrowed", "Other frames", "borrowed")}
      {p?.upscaled ? item(SHARE_BG.upscaled, "Upscaled", "upscaled") : null}
      {item("bg-ai", "AI", "ai")}
    </div>
  );
}

// --- menu ----------------------------------------------------------------------------------------------------------

/** A "…" menu's row, and its Delete row (last, after a hairline). */
export const MENU_ITEM =
  "flex w-full items-center px-4 py-2.5 text-left text-[14px] text-ink transition-colors duration-150 hover:bg-line hover:text-ivory focus-visible:bg-line focus-visible:outline-none";
export const MENU_DELETE =
  "flex w-full items-center px-4 py-2.5 text-left text-[14px] text-tally transition-colors duration-150 hover:bg-line focus-visible:bg-line focus-visible:outline-none";

/**
 * The one delete confirm (DESIGN_RULES, Destructive actions): the name, one short line, Cancel and the danger button.
 * It takes the place of the "…" menu it came from, never a browser dialog; Cancel (or Escape, or a click away) keeps it.
 */
export function ConfirmDelete({
  title,
  line,
  confirm,
  onConfirm,
  onCancel,
}: {
  title: string;
  line?: string | null;
  confirm: string;
  onConfirm: () => Promise<unknown> | void;
  onCancel: () => void;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <div role="alertdialog" aria-label={title} className="w-[280px] animate-fade-in p-4">
      <p className="text-[14px] text-ivory [overflow-wrap:anywhere]">{title}</p>
      {line ? <p className="mt-1 text-[13px] text-ink2">{line}</p> : null}
      <div className="mt-4 flex justify-end gap-1.5">
        <Button variant="ghost" size="sm" autoFocus onClick={onCancel}>
          Cancel
        </Button>
        <Button
          variant="danger"
          size="sm"
          loading={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await onConfirm();
            } finally {
              setBusy(false);
            }
          }}
        >
          {confirm}
        </Button>
      </div>
    </div>
  );
}

export function useDismiss(open: boolean, close: () => void, ref: RefObject<HTMLElement>) {
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
      }
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open, close, ref]);
}

export function SplitMenuButton({
  label,
  icon,
  items,
  disabled,
}: {
  label: string;
  icon?: LucideIcon;
  items: Array<{ label: string; href: string; sub?: string }>;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = () => setOpen(false);
  useDismiss(open, close, ref);
  return (
    <div ref={ref} className="relative">
      <Button variant="primary" size="lg" block icon={icon} iconRight={ChevronDown} disabled={disabled || !items.length} onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}>
        {label}
      </Button>
      {open ? (
        <div role="menu" className="absolute left-0 right-0 top-full z-40 mt-2 animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          {items.map((it) => (
            <a
              key={it.label}
              role="menuitem"
              href={it.href}
              download
              onClick={close}
              className="flex items-center justify-between gap-3 px-4 py-2.5 text-[14px] text-ink transition-colors duration-150 hover:bg-line hover:text-ivory focus-visible:bg-line"
            >
              <span>{it.label}</span>
              <span className="slate text-ink3">{it.sub ?? "Download"}</span>
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * A horizontal row that can scroll: soft edge fades and arrow buttons appear on the side that has more.
 * `bleed` lets the row run to the page edge (it undoes the page's 40 px gutter).
 */
export function ScrollRow({ children, className, label, bleed = true }: { children: ReactNode; className?: string; label: string; bleed?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });
  const update = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const left = el.scrollLeft > 4;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 4;
    setEdges((e) => (e.left === left && e.right === right ? e : { left, right }));
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    for (const c of Array.from(el.children)) ro.observe(c);
    el.addEventListener("scroll", update, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", update);
    };
  });
  const by = (dir: 1 | -1) => {
    const el = ref.current;
    if (el) el.scrollBy({ left: dir * Math.max(240, el.clientWidth * 0.8), behavior: "smooth" });
  };
  const arrow = "absolute top-[38%] z-[2] flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-full border border-line2 bg-raised/90 text-ivory shadow-[0_2px_12px_rgba(0,0,0,.5)] transition-colors duration-150 hover:border-ivory/50";
  return (
    <div className={cx("relative", bleed && "-mx-10")}>
      <div ref={ref} role="group" aria-label={label} className={cx("flex overflow-x-auto", bleed && "px-10", className)}>
        {children}
      </div>
      <div
        aria-hidden="true"
        className={cx("pointer-events-none absolute inset-y-0 left-0 w-20 bg-gradient-to-r from-bg to-transparent transition-opacity duration-200", edges.left ? "opacity-100" : "opacity-0")}
      />
      <div
        aria-hidden="true"
        className={cx("pointer-events-none absolute inset-y-0 right-0 w-24 bg-gradient-to-l from-bg to-transparent transition-opacity duration-200", edges.right ? "opacity-100" : "opacity-0")}
      />
      {edges.left ? (
        <button type="button" aria-label={`Scroll ${label} left`} onClick={() => by(-1)} className={cx(arrow, "left-4")}>
          <ChevronLeft size={18} strokeWidth={1.75} />
        </button>
      ) : null}
      {edges.right ? (
        <button type="button" aria-label={`Scroll ${label} right`} onClick={() => by(1)} className={cx(arrow, "right-4")}>
          <ChevronRight size={18} strokeWidth={1.75} />
        </button>
      ) : null}
    </div>
  );
}

/** A pill that opens a small menu of actions ("Share ▾", "Export timeline ▾"). */
export function MenuPill({
  label,
  icon,
  items,
  variant = "secondary",
  size = "md",
  align = "left",
  disabled,
  loading,
  menuLabel,
}: {
  label: string;
  icon?: LucideIcon;
  items: Array<{ id: string; label: ReactNode; sub?: ReactNode; onSelect: () => void }>;
  variant?: Variant;
  size?: Size;
  align?: "left" | "right";
  disabled?: boolean;
  loading?: boolean;
  menuLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, ref);
  return (
    <div ref={ref} className="relative">
      <Button
        variant={variant}
        size={size}
        icon={icon}
        iconRight={ChevronDown}
        disabled={disabled}
        loading={loading}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {label}
      </Button>
      {open ? (
        <div
          role="menu"
          aria-label={menuLabel ?? label}
          className={cx(
            "absolute top-full z-40 mt-2 min-w-[260px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1",
            align === "right" ? "right-0" : "left-0",
          )}
        >
          {items.map((it) => (
            <button
              key={it.id}
              type="button"
              role="menuitem"
              onClick={() => {
                close();
                it.onSelect();
              }}
              className="flex w-full items-center justify-between gap-4 px-4 py-2.5 text-left text-[14px] text-ink transition-colors duration-150 hover:bg-line hover:text-ivory focus-visible:bg-line focus-visible:outline-none"
            >
              <span>{it.label}</span>
              {it.sub ? <span className="slate shrink-0 text-ink3">{it.sub}</span> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// --- modal ---------------------------------------------------------------------------------------------------------

export function Modal({ open, onClose, label, children, width = 760 }: { open: boolean; onClose: () => void; label: string; children: ReactNode; width?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const t = window.setTimeout(() => {
      const first = ref.current?.querySelector<HTMLElement>("[data-autofocus]") ?? ref.current;
      first?.focus();
    }, 20);
    return () => {
      window.clearTimeout(t);
      prev?.focus?.();
    };
  }, [open]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 px-4 py-[6vh] animate-fade-in" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        className="relative w-full animate-rise-in rounded-card border border-line bg-panel"
        style={{ maxWidth: width }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            onClose();
            return;
          }
          if (e.key !== "Tab" || !ref.current) return;
          const nodes = ref.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])');
          if (!nodes.length) return;
          const first = nodes[0];
          const last = nodes[nodes.length - 1];
          if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
          } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }}
      >
        {children}
      </div>
    </div>
  );
}

// --- image drop slot -----------------------------------------------------------------------------------------------

export interface SlotValue {
  id: string;
  url: string;
  name: string;
}

export function DropSlot({
  value,
  onFile,
  onClear,
  busy,
  progress,
  error,
  title,
  sub,
  compact,
}: {
  value: SlotValue | null;
  onFile: (f: File) => void;
  onClear: () => void;
  busy?: boolean;
  progress?: number;
  error?: string | null;
  title: string;
  sub?: string;
  compact?: boolean;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pick = () => input.current?.click();
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };
  const hidden = (
    <input
      ref={input}
      type="file"
      accept="image/*,.exr,.tif,.tiff"
      className="hidden"
      onChange={(e) => {
        const f = e.target.files?.[0];
        if (f) onFile(f);
        e.target.value = "";
      }}
    />
  );
  if (value && !busy) {
    return (
      <div className={cx("relative overflow-hidden rounded-card border border-line bg-black", compact ? "h-20" : "")}>
        {hidden}
        <img src={value.url} alt="" className={cx("thumb-face w-full", compact ? "h-20" : "aspect-video")} />
        <Slate at="tl">{value.name}</Slate>
        <span className="absolute right-2 top-2 flex gap-1">
          <button type="button" onClick={pick} className="slate rounded-[2px] bg-black/70 px-1.5 py-[3px] hover:text-ivory">
            Replace
          </button>
          <button type="button" onClick={onClear} aria-label="Remove image" className="rounded-[2px] bg-black/70 p-1 text-ivory/70 hover:text-ivory">
            <X size={12} />
          </button>
        </span>
      </div>
    );
  }
  return (
    <div>
      {hidden}
      <button
        type="button"
        onClick={pick}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        disabled={busy}
        className={cx(
          "flex w-full items-center gap-3.5 rounded-card border text-left transition-colors duration-150 ease-out",
          compact ? "h-16 px-4" : "h-24 px-5",
          over ? "border-ivory/70 bg-raised" : "border-line hover:border-line2",
        )}
      >
        <span className={cx("flex h-8 w-8 shrink-0 items-center justify-center rounded-full border", over ? "border-ivory/70 text-ivory" : "border-line2 text-ink2")}>
          {busy ? <Spinner size={13} /> : <Plus size={15} strokeWidth={1.5} />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[14px] text-ink">{busy ? "Adding…" : title}</span>
          {sub && !busy ? <span className="block text-[12px] text-ink3">{sub}</span> : null}
          {busy && progress != null ? <ProgressLine value={progress} idle className="mt-2 w-40" /> : null}
        </span>
      </button>
      {error ? (
        <p className="mt-2 flex items-baseline gap-2 text-[13px] text-ink">
          <span className="slate">Error</span>
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function StepHeader({ n, title, done, dim }: { n: number; title: string; done?: boolean; dim?: boolean }) {
  return (
    <div className={cx("flex items-baseline gap-3", dim && "opacity-45")}>
      <span className="mono w-5 shrink-0 text-[11px] tracking-[0.08em] text-ink3">{done ? <Check size={12} strokeWidth={2.5} className="inline text-ivory" /> : String(n).padStart(2, "0")}</span>
      <h3 className="text-[16px] font-medium text-ivory">{title}</h3>
    </div>
  );
}

/** Mono section label with an index: "01 — What you can make". */
export function SectionLabel({ n, children, className }: { n?: number; children: ReactNode; className?: string }) {
  return (
    <div className={cx("slate text-ink3", className)}>
      {n != null ? <span className="mr-2 text-ink4">{String(n).padStart(2, "0")}</span> : null}
      {children}
    </div>
  );
}

// --- composer ---------------------------------------------------------------------------------------------------------

export interface ComposerTab<T extends string = string> {
  id: T;
  /** "Edit clip". */
  label: string;
  /** "Edit": below 640 px. */
  short: string;
  /** Optional: Create's six tabs go without, so they and Saves-to fit one row at 1280. */
  icon?: LucideIcon;
}

/** The one prompt box: tabs (Edit clip, New image, New character, New previz) with Saves-to on the right, an optional
 * strip (presets) under them, and the tab's body, whose primary button is the kit's lg primary. Home shows it in the
 * hero; Create (and Previz) dock it at the bottom of the page. Same width, radius and background everywhere.
 *
 * placement "hero": in the page flow, centred. "dock": pinned to the bottom of the nearest positioned parent, over
 * whatever the page shows behind it. */
export function Composer<T extends string>({
  tabs,
  tab,
  onTab,
  savesTo,
  strip,
  children,
  placement = "hero",
  label = "What do you want to start with?",
  compact = false,
  className,
}: {
  tabs: ComposerTab<T>[];
  tab: T;
  onTab: (t: T) => void;
  savesTo?: ReactNode;
  strip?: ReactNode;
  children: ReactNode;
  placement?: "hero" | "dock";
  label?: string;
  /** Just the body: no tab row or strip, tight padding (a phone's collapsed dock; the page expands it on focus). */
  compact?: boolean;
  className?: string;
}) {
  const box = (
    <div
      className={cx(
        "relative mx-auto w-full max-w-[900px] rounded-[16px] border border-line2 bg-panel text-left shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)]",
        placement === "hero" && className,
      )}
    >
      <div className={cx("flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-line px-2 py-2", compact && "hidden")}>
        <div role="radiogroup" aria-label={label} className="flex min-w-0 gap-1 overflow-x-auto [scrollbar-width:none]">
          {tabs.map(({ id, label: name, short, icon: I }) => {
            const on = tab === id;
            return (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => onTab(id)}
                className={cx(
                  // text-sm (14 px), not text-[14px]: the desktop type step (index.css) skips it, so the tabs and Saves to keep one row.
                  "inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3 text-sm transition-colors duration-150",
                  on ? "bg-raised text-ivory shadow-[inset_0_0_0_1px_#3A3631]" : "text-ink2 hover:text-ivory",
                )}
              >
                {I ? <I size={16} strokeWidth={1.5} aria-hidden="true" className="max-[400px]:hidden" /> : null}
                <span className="max-[640px]:hidden">{name}</span>
                <span className="hidden max-[640px]:inline">{short}</span>
              </button>
            );
          })}
        </div>
        {savesTo}
      </div>
      {strip && !compact ? <div className="border-b border-line px-3 py-2">{strip}</div> : null}
      {children}
    </div>
  );
  if (placement === "hero") return box;
  return <div className={cx("absolute inset-x-0 bottom-0 z-10 px-4 pb-6 pt-12 max-[767px]:px-3", compact && "pb-3 pt-3", className)}>{box}</div>;
}

// --- page frame -------------------------------------------------------------------------------------------------------

/** Every scrolling page: one width (1440), one gutter (40 px; 16 on a phone), one top and bottom margin. */
export function Page({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className="h-full overflow-y-auto">
      <div className={cx("mx-auto w-full max-w-[1440px] px-10 pb-28 pt-10 max-[767px]:px-4 max-[767px]:pt-6", className)}>{children}</div>
    </div>
  );
}

/** Every page's title row: a 36 px display title, one line under it, actions on the right. One per page. */
export function PageHeader({ title, sub, actions, className }: { title: ReactNode; sub?: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <header className={cx("flex flex-wrap items-end justify-between gap-x-6 gap-y-4 pb-8", className)}>
      <div className="min-w-0">
        <h1 className="display text-[36px] leading-[1.05]">{title}</h1>
        {sub ? <p className="mt-2 max-w-[60ch] text-[14px] leading-relaxed text-ink2">{sub}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

/** A section's title inside a page or panel: 16 px medium, ivory; an optional hint after it. */
export function SectionTitle({ children, hint, className }: { children: ReactNode; hint?: string; className?: string }) {
  return (
    <h2 className={cx("flex items-center gap-1.5 text-[16px] font-medium text-ivory", className)}>
      {children}
      {hint ? <Hint text={hint} /> : null}
    </h2>
  );
}
