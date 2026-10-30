// Download, grouped by purpose. The pill downloads your last choice (default ProRes at the top size);
// the chevron opens the menu. Works with the newer {codec, size, bytes} downloads and the older label-only ones.
import { ChevronDown, Download as DownloadIcon } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { cx } from "../lib/format";
import type { Download, DownloadCodec, DownloadSize } from "../types";
import { useDismiss } from "./ui";

const KEY = "pickups.download";

interface Item extends Omit<Download, "codec" | "size"> {
  codec: DownloadCodec | "other";
  size: DownloadSize;
}

const GROUPS: Array<{ id: string; title: string; hint: string; codecs: Array<Item["codec"]> }> = [
  { id: "prores", title: "ProRes 422 HQ", hint: "For editing and grading.", codecs: ["prores"] },
  { id: "h264", title: "MP4 · H.264", hint: "Plays everywhere.", codecs: ["h264"] },
  { id: "h265", title: "MP4 · H.265", hint: "Half the size. Modern devices.", codecs: ["h265"] },
  { id: "hdr", title: "HDR master (HLG) and EXR (ACEScg)", hint: "For HDR delivery and grading.", codecs: ["hlg", "exr"] },
  { id: "other", title: "Other files", hint: "", codecs: ["other"] },
];

const SIZE_RANK: Record<DownloadSize, number> = { UHD: 0, HD: 1, native: 2 };

function guessCodec(label: string): Item["codec"] {
  const l = label.toLowerCase();
  if (l.includes("prores")) return "prores";
  if (l.includes("h.265") || l.includes("h265") || l.includes("hevc")) return "h265";
  if (l.includes("h.264") || l.includes("h264") || l.includes("mp4")) return "h264";
  if (l.includes("hlg") || l.includes("hdr")) return "hlg";
  if (l.includes("exr")) return "exr";
  return "other";
}

function guessSize(label: string): DownloadSize {
  if (/\b(uhd|4k|2160)\b/i.test(label)) return "UHD";
  if (/\b(hd|1080)\b/i.test(label)) return "HD";
  return "native";
}

function normalise(d: Download): Item {
  return { ...d, codec: d.codec ?? guessCodec(d.label), size: d.size ?? guessSize(d.label) };
}

function fileSize(bytes?: number): string | null {
  if (bytes == null || !isFinite(bytes) || bytes <= 0) return null;
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(gb >= 10 ? 0 : 1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

const keyOf = (i: Item) => `${i.codec}:${i.size}`;
const sizeLabel = (s: DownloadSize) => (s === "native" ? "Native" : s);

function remembered(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

function remember(i: Item) {
  try {
    window.localStorage.setItem(KEY, keyOf(i));
  } catch {
    /* private mode etc.: just don't remember */
  }
}

function fetchFile(i: Item) {
  const a = document.createElement("a");
  a.href = i.url;
  a.download = i.filename || "";
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function DownloadControl({ downloads }: { downloads: Download[] }) {
  const items = useMemo(() => downloads.map(normalise), [downloads]);
  const [open, setOpen] = useState(false);
  const [last, setLast] = useState<string | null>(() => remembered());
  const ref = useRef<HTMLDivElement>(null);
  useDismiss(open, () => setOpen(false), ref);

  const groups = useMemo(
    () =>
      GROUPS.map((g) => ({
        ...g,
        items: items
          .filter((i) => g.codecs.includes(i.codec))
          .sort((a, b) => g.codecs.indexOf(a.codec) - g.codecs.indexOf(b.codec) || SIZE_RANK[a.size] - SIZE_RANK[b.size]),
      })).filter((g) => g.items.length),
    [items],
  );

  // Last-used choice, else ProRes at the top size, else the first file.
  const primary = useMemo(() => {
    const byKey = items.find((i) => keyOf(i) === last) ?? (last ? items.find((i) => i.codec === last.split(":")[0]) : undefined);
    if (byKey) return byKey;
    const prores = items.filter((i) => i.codec === "prores").sort((a, b) => SIZE_RANK[a.size] - SIZE_RANK[b.size]);
    return prores[0] ?? items[0] ?? null;
  }, [items, last]);

  const pick = (i: Item) => {
    remember(i);
    setLast(keyOf(i));
    setOpen(false);
    fetchFile(i);
  };

  const primaryDetail = primary
    ? [primary.codec === "other" ? primary.label : GROUPS.find((g) => g.codecs.includes(primary.codec))?.title.split(" and ")[0], sizeLabel(primary.size), fileSize(primary.bytes)]
        .filter(Boolean)
        .join(" · ")
    : null;

  return (
    <div ref={ref} className="relative">
      <div className={cx("flex h-12 overflow-hidden rounded-full", primary ? "bg-ivory text-[#0D0C0B]" : "bg-raised text-ink4")}>
        <button
          type="button"
          disabled={!primary}
          onClick={() => primary && pick(primary)}
          className="flex min-w-0 flex-1 items-center gap-3 pl-5 pr-3 text-left transition-colors duration-150 hover:bg-black/[0.06] disabled:cursor-not-allowed"
        >
          <DownloadIcon size={17} strokeWidth={1.75} aria-hidden="true" className="shrink-0" />
          <span className="min-w-0">
            <span className="block text-[14px] font-medium leading-tight">Download</span>
            {primaryDetail ? <span className="block truncate font-mono text-[10px] uppercase tracking-[0.08em] opacity-60">{primaryDetail}</span> : null}
          </span>
        </button>
        <button
          type="button"
          disabled={!items.length}
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="All downloads"
          className="flex w-12 shrink-0 items-center justify-center border-l border-black/15 transition-colors duration-150 hover:bg-black/[0.06] disabled:cursor-not-allowed"
        >
          <ChevronDown size={17} strokeWidth={1.75} className={cx("transition-transform duration-150", open && "rotate-180")} />
        </button>
      </div>
      {!items.length ? <p className="slate mt-2 text-ink3">Files are still being written</p> : null}
      {open ? (
        <div role="menu" className="absolute left-0 right-0 top-full z-40 mt-2 max-h-[60vh] animate-rise-in overflow-y-auto rounded-card border border-line2 bg-raised py-2">
          {groups.map((g, gi) => (
            <div key={g.id} className={cx(gi > 0 && "mt-1 border-t border-line pt-2")}>
              <div className="px-4 pb-1 pt-1">
                <div className="text-[13px] font-medium text-ink">{g.title}</div>
                {g.hint ? <div className="text-[12px] text-ink3">{g.hint}</div> : null}
              </div>
              {g.items.map((i) => {
                const chosen = primary && keyOf(i) === keyOf(primary) && i.url === primary.url;
                return (
                  <button
                    key={`${i.url}:${i.label}`}
                    type="button"
                    role="menuitem"
                    aria-label={`Download ${g.title}, ${g.id === "hdr" || g.id === "other" ? i.label : sizeLabel(i.size)}${fileSize(i.bytes) ? `, ${fileSize(i.bytes)}` : ""}${i.filename ? `, ${i.filename}` : ""}`}
                    onClick={() => pick(i)}
                    className="flex w-full items-center gap-3 px-4 py-2 text-left transition-colors duration-150 hover:bg-line focus-visible:bg-line"
                  >
                    <span className={cx("h-1 w-1 shrink-0 rounded-full", chosen ? "bg-ivory" : "bg-transparent")} aria-hidden="true" />
                    <span className="min-w-0 flex-1 truncate text-[13px] text-ink">
                      {g.id === "hdr" || g.id === "other" ? i.label.replace(/\s*·\s*(UHD|HD)$/i, "") : sizeLabel(i.size)}
                      {g.id === "hdr" && i.size !== "native" ? <span className="text-ink3"> · {i.size}</span> : null}
                    </span>
                    <span className="mono shrink-0 text-[12px] text-ink3">{fileSize(i.bytes) ?? "—"}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
