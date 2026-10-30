import { Check, Copy, ExternalLink, X } from "lucide-react";
import { SaveFolderRow } from "./SaveTo";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { closeSetup, downloadModels, getApi, importExamplesPack, loadExamplesPack, openOnboarding, revealLibrary, toast } from "../actions";
import { request } from "../api";
import { clock, cx, gb, missingFor, shortGpu, size } from "../lib/format";
import { useApp } from "../store";
import { TASKS, type Benchmark, type Status, type StorageInfo, type TaskId, type Verdict } from "../types";
import { DirectorSetup, RadioRow } from "./DirectorSetup";
import { ServerSetup } from "./ServerSetup";
import { Commands, diskNeedGb, FeatureCard, missingFiles } from "./Onboarding";
import { servers, type Capabilities, type Requirements } from "../lib/servers";
import { TASK_META } from "./taskMeta";
import { Button, Hint, IconButton, MiddleTruncate, Modal, Notice, ProgressLine, StateDot, Switch } from "./ui";

const VERDICT: Record<Verdict, string> = {
  ready: "Ready",
  slower: "Ready, but slower",
  untested: "Not tested on this GPU",
  remote: "Needs a remote GPU",
  offline: "Not connected",
};

/** "Prompt director" → "setup-prompt-director": openSetup(null, "prompt-director") scrolls to it. */
function sectionId(title: string) {
  return `setup-${title.toLowerCase().replace(/\s+/g, "-")}`;
}

function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section id={sectionId(title)} className="scroll-mt-6 border-t border-line pt-6">
      <div className="mb-4 flex min-w-0 items-baseline justify-between gap-4 [&>*:last-child]:min-w-0">
        <h3 className="text-[16px] font-medium text-ivory">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

/** Setup → Models: the tasks (Reshoot stays in Labs, off this list), then Images (the default image model). */
const MODEL_ROWS: Array<TaskId | "image"> = [...TASKS.filter((t) => t !== "reshoot"), "image"];

function allMissing(status: Status | null) {
  const files = new Map<string, number>();
  for (const t of MODEL_ROWS) for (const m of status?.tiles?.[t]?.missing ?? []) files.set(m.file, Number(m.gb) || 0);
  return { count: files.size, gb: Array.from(files.values()).reduce((a, b) => a + b, 0) };
}

const linkCls = "inline-flex shrink-0 items-center gap-1 text-ivory/80 underline-offset-4 transition-colors hover:text-ivory hover:underline";

export function SetupPanel() {
  const open = useApp((s) => s.setup.open);
  const focus = useApp((s) => s.setup.focus);
  const section = useApp((s) => s.setup.section);
  const status = useApp((s) => s.status);
  // Opened for one section (Enhance's "Setup" → Prompt director): bring it into view once the modal has drawn.
  useEffect(() => {
    if (!open || !section) return;
    const t = window.setTimeout(() => document.getElementById(sectionId(section))?.scrollIntoView({ block: "start", behavior: "smooth" }), 80);
    return () => window.clearTimeout(t);
  }, [open, section]);
  const fetching = useApp((s) => s.modelsFetching);

  const connected = !!status?.connected;
  const busy = !!status?.comfy_queue?.outside;
  const m = status?.machine;

  const focusMissing = focus ? missingFor(status, focus) : null;
  const [cap, setCap] = useState<Capabilities | null>(null);
  useEffect(() => {
    if (!open) return;
    let live = true;
    servers
      .capabilities()
      .then((c) => live && setCap(c))
      .catch(() => live && setCap(null));
    return () => {
      live = false;
    };
  }, [open, status]);
  // One source: once capabilities are in, the Models line and the Ready line read the same files and disk the cards and
  // the table show (a low-disk setup never says "All installed" above cards still missing files).
  const missing = useMemo(() => {
    if (!cap?.features?.length) return allMissing(status);
    const m = missingFiles(cap.features);
    return { count: m.files.length, gb: m.gb };
  }, [cap, status]);
  const req = cap?.requirements;
  const lowDisk =
    !!req &&
    ((req.models_free_gb != null && missing.count > 0 && req.models_free_gb < diskNeedGb(cap!.features)) ||
      (req.library_free_gb != null && req.library_free_gb < req.library_min_free_gb));
  const ready = status?.verdict === "ready" || status?.verdict === "slower";
  const line = ready && lowDisk ? "Low on disk" : ready && missing.count ? "Models missing" : VERDICT[status?.verdict ?? "offline"] ?? VERDICT.offline;

  return (
    <Modal open={open} onClose={closeSetup} label="Setup" width={760}>
      <div className="flex items-start justify-between px-8 pb-6 pt-8">
        <div>
          <h2 className="display text-[36px] leading-[1.05]">Setup</h2>
        </div>
        <IconButton icon={X} label="Close" hint="Esc" onClick={closeSetup} />
      </div>

      <div className="space-y-8 px-8 pb-8">
        {focus && connected && focusMissing && focusMissing.count ? (
          <Notice
            tone="warn"
            text={`${TASK_META[focus].verb} needs ${focusMissing.count} ${focusMissing.count === 1 ? "model" : "models"} (${gb(focusMissing.gb)}).`}
            action={{ label: "Get them", loading: fetching, onClick: () => void downloadModels(focus) }}
          />
        ) : null}

        <Section title="Server">
          <ServerSetup open={open} />
        </Section>

        <Section title="Your machine">
          <RequirementsTable req={cap?.requirements ?? null} cap={cap} fallback={m ? { gpu: shortGpu(m.gpu), vram: m.vram_gb, comfy: m.comfy_version } : null} />
          {/* Busy (ComfyUI is running something Pickups didn't send) takes over the line, so it never says Ready above it. */}
          <p className="mt-5 flex items-center gap-2.5 text-[14px] text-ink">
            <StateDot on={status?.verdict === "ready" && !busy && !lowDisk && !missing.count} />
            {busy ? "Busy with another ComfyUI job" : line}
            {busy ? <Hint text="New renders wait for it. If it never finishes, restart ComfyUI on the GPU machine." /> : null}
            {!busy && (status?.verdict === "remote" || !connected) ? (
              <Button variant="quiet" size="sm" onClick={() => openOnboarding("connect")}>
                {connected ? "Use another GPU" : "Connect"}
              </Button>
            ) : null}
          </p>
        </Section>

        <Section
          title="Models"
          right={
            connected ? (
              <span className="flex flex-wrap items-center justify-end gap-x-3 gap-y-2">
                {missing.count ? (
                  <Button variant="primary" size="sm" loading={fetching} onClick={() => openOnboarding("choose")}>
                    {`Get missing models · ${gb(missing.gb)}`}
                  </Button>
                ) : (
                  <span className="slate inline-flex items-center gap-1.5 text-ink2">
                    <Check size={13} strokeWidth={2} /> All installed
                  </span>
                )}
                <Button variant="quiet" size="sm" onClick={() => openOnboarding("scan")}>
                  Check setup
                </Button>
              </span>
            ) : null
          }
        >
          <ModelsProgress />
          {connected ? <SetupFeatures cap={cap} focus={focus} /> : (
            <Notice text="Not connected to ComfyUI." action={{ label: "Connect", onClick: () => openOnboarding("connect") }} />
          )}
        </Section>

        <Section title="Examples">
          <ExamplesSetup open={open} />
        </Section>

        <Section title="Storage">
          <StorageRow open={open} />
          <div className="mt-4">
            <SaveFolderRow />
          </div>
        </Section>

        <PhoneRow open={open} />

        <Benchmarks status={status} />

        <DirectorSetup Section={Section} />

        <Licences />
      </div>
    </Modal>
  );
}

/** A download running on this computer, or how to get the models onto the GPU machine when it's another one. */
function ModelsProgress() {
  const d = useApp((s) => s.modelsDownload);
  const help = useApp((s) => s.modelsHelp);
  if (d?.running) {
    const value = d.total ? (d.done ?? 0) / d.total : 0;
    return (
      <div className="mb-5 space-y-2">
        <div className="flex items-baseline justify-between text-[13px]">
          <span className="text-ink">Downloading{d.file ? ` ${d.file}` : "…"}</span>
          <span className="mono text-ink3">{Math.round(value * 100)}%</span>
        </div>
        <ProgressLine value={value} />
      </div>
    );
  }
  if (d && !d.running && d.error && !help) {
    const task = d.task as TaskId | "image" | null | undefined;
    return (
      <Notice
        className="mb-5"
        tone="bad"
        text={`The download stopped: ${d.error}`}
        action={{ label: "Try again", onClick: () => void downloadModels(task ?? undefined) }}
      />
    );
  }
  if (!help) return null;
  return (
    <div className="mb-5 space-y-4 rounded-card border border-line2 p-4">
      <p className="text-[14px] text-ink">{help.message}</p>
      {help.gated?.length ? (
        <ul className="space-y-1.5">
          {help.gated.map((page) => (
            <li key={page}>
              <a href={page} target="_blank" rel="noreferrer" className={linkCls}>
                Accept the license: {page.replace("https://huggingface.co/", "")} <ExternalLink size={11} />
              </a>
            </li>
          ))}
        </ul>
      ) : null}
      {help.command ? <Commands commands={[{ os: "GPU machine", label: "", command: help.command, note: "Needs a Pickups folder there (git clone, then this). Gated models also need HF_TOKEN set." }]} intro="On the computer running ComfyUI:" /> : null}
    </div>
  );
}

type BenchGroup = { gpu: string; current: boolean; host: string | null; rows: Benchmark[] };

/** Rows by GPU, the connected one first (older servers send rows without a GPU: they're the connected one's). */
function benchGroups(rows: Benchmark[], connectedGpu: string | undefined): BenchGroup[] {
  const out: BenchGroup[] = [];
  for (const b of rows) {
    const gpu = b.gpu ?? connectedGpu ?? "";
    let g = out.find((x) => x.gpu === gpu);
    if (!g) {
      const current = !!b.current || (!!connectedGpu && shortGpu(gpu) === shortGpu(connectedGpu));
      out.push((g = { gpu, current, host: b.host ?? null, rows: [] }));
    }
    g.rows.push(b);
  }
  return out.sort((a, b) => Number(b.current) - Number(a.current));
}

const HOSTS: Record<string, string> = { local: "This computer", remote: "Remote", runpod: "RunPod" };

function benchHint(b: Benchmark): string {
  if (b.measured === false) return "Estimated from the RTX 4090 reference";
  return [
    b.gpu_seconds != null ? `GPU ${clock(b.gpu_seconds)}` : null,
    b.s_per_frame != null ? `${b.s_per_frame} s per frame` : null,
    b.fastest != null && b.slowest != null && (b.runs ?? 0) > 1 ? `fastest ${clock(b.fastest)}, slowest ${clock(b.slowest)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** Setup → Benchmarks: median times of finished runs, per task, on every GPU measured (the connected one first). */
function Benchmarks({ status }: { status: Status | null }) {
  const m = status?.machine;
  const groups = useMemo(() => benchGroups(status?.benchmarks ?? [], m?.gpu), [status?.benchmarks, m?.gpu]);
  const measured = groups.some((g) => g.rows.some((b) => b.measured));
  const note = measured ? "Median of finished runs" : m ? `Estimated for ${shortGpu(m.gpu)}` : null;
  return (
    <Section title="Benchmarks" right={note ? <span className="slate text-ink3">{note}</span> : null}>
      {groups.length ? (
        <div className="overflow-x-auto">
          <table className="mono w-full border-t border-line text-left text-[12px]">
            <thead className="slate text-ink3">
              <tr>
                <th className="py-2 pr-3 font-normal sm:pr-4">Task</th>
                <th className="hidden py-2 pr-4 font-normal sm:table-cell">GPU</th>
                <th className="hidden py-2 pr-3 font-normal sm:table-cell sm:pr-4">Frames / size</th>
                <th className="py-2 pr-3 text-right font-normal sm:pr-4">Time</th>
                <th className="py-2 text-right font-normal">Runs</th>
              </tr>
            </thead>
            {groups.map((g) => {
              const where = g.current ? "Connected" : HOSTS[g.host ?? ""] ?? g.host ?? "";
              return (
                <tbody key={g.gpu} className="border-t border-line">
                  {/* Phones: the GPU heads its rows instead of taking a column. */}
                  <tr className="sm:hidden">
                    <td colSpan={3} className="slate pb-1 pt-3 text-ink2">
                      {[shortGpu(g.gpu), where].filter(Boolean).join(" · ")}
                    </td>
                  </tr>
                  {g.rows.map((b, i) => (
                    <tr key={`${b.task}-${i}`} className={i ? "border-t border-line/40" : undefined}>
                      <td className="py-2.5 pr-3 font-sans text-[13px] text-ink sm:pr-4">
                        {b.task}
                        {/* Phones: frames and size under the task, so the task's name has the row's width. */}
                        <span className="mono mt-0.5 block text-[12px] text-ink3 sm:hidden">
                          {b.frames ?? "–"} · {b.size || "–"}
                        </span>
                      </td>
                      <td className="hidden py-2.5 pr-4 align-top text-ink2 sm:table-cell">
                        {i === 0 ? (
                          <>
                            <div>{shortGpu(g.gpu)}</div>
                            <div className="slate mt-0.5 text-ink3">{where}</div>
                          </>
                        ) : null}
                      </td>
                      <td className="hidden py-2.5 pr-3 text-ink2 sm:table-cell sm:whitespace-nowrap sm:pr-4">
                        {b.frames ?? "–"}
                        <span className="text-ink3"> · </span>
                        <span className="whitespace-nowrap">{b.size || "–"}</span>
                      </td>
                      <td className="whitespace-nowrap py-2.5 pr-3 text-right text-ink sm:pr-4" title={benchHint(b) || undefined}>
                        {clock(b.seconds)}
                      </td>
                      <td className="py-2.5 text-right text-ink3">{b.measured === false ? "–" : b.runs ?? "–"}</td>
                    </tr>
                  ))}
                </tbody>
              );
            })}
          </table>
        </div>
      ) : (
        <p className="text-[13px] text-ink3">No runs on this machine yet.</p>
      )}
    </Section>
  );
}

type LicenceKey = "commercial_safe" | "allow_nvidia_rtx";

const RTX_TITLE = "Use NVIDIA RTX accelerators for fast previews";

/** Licence choices: the frame editor (Qwen-Image 2.1, research licence, or Apache-2.0 Qwen-Image-Edit 2511 for paid
 *  work; saved as commercial_safe) and NVIDIA's proprietary RTX accelerators, off by default. Saves are optimistic. */
function Licences() {
  const [st, setSt] = useState<Record<LicenceKey, boolean> | null>(null);
  const [failed, setFailed] = useState(false);
  // Latest save per key, so a slow failure doesn't roll back a newer choice.
  const seq = useRef<Record<LicenceKey, number>>({ commercial_safe: 0, allow_nvidia_rtx: 0 });
  useEffect(() => {
    getApi()
      .settings()
      .then((d) => setSt({ commercial_safe: !!d.commercial_safe, allow_nvidia_rtx: !!d.allow_nvidia_rtx }))
      .catch(() => setSt(null));
  }, []);
  if (st === null) return null;
  const save = async (key: LicenceKey, next: boolean) => {
    if (st[key] === next) return;
    const prev = st[key];
    const mine = ++seq.current[key];
    setFailed(false);
    setSt((s) => s && { ...s, [key]: next });
    try {
      await getApi().saveSettings({ [key]: next });
    } catch {
      if (seq.current[key] !== mine) return;
      setSt((s) => s && { ...s, [key]: prev });
      setFailed(true);
    }
  };
  return (
    <Section title="Licenses">
      <div className="space-y-6">
        <div>
          <div className="mb-3 flex items-center gap-1.5">
            <span id="frame-editor" className="text-[14px] text-ink">
              Frame editor
            </span>
            <Hint text="Edits a still frame for Remove a person, Previz to final and Change the weather." />
          </div>
          <div className="space-y-2" role="radiogroup" aria-labelledby="frame-editor">
            <RadioRow
              on={!st.commercial_safe}
              onClick={() => void save("commercial_safe", false)}
              title="Qwen-Image 2.1"
              sub="Best quality. Qwen Research License: not for commercial use."
            />
            <RadioRow
              on={st.commercial_safe}
              onClick={() => void save("commercial_safe", true)}
              title="Qwen-Image-Edit 2511"
              sub="Apache-2.0, fine for paid work. Slightly softer; reads 2 references, not 4."
            />
          </div>
        </div>
        <div className="flex items-start gap-6">
          <div className="flex-1">
            <div className="text-[14px] text-ink">{RTX_TITLE}</div>
            <p className="mt-1 text-[13px] leading-relaxed text-ink3">
              About 12× faster 4K previews. Closed source, so off by default.
            </p>
          </div>
          <div className="pt-0.5">
            <Switch checked={st.allow_nvidia_rtx} onChange={(v) => void save("allow_nvidia_rtx", v)} label={RTX_TITLE} />
          </div>
        </div>
        {failed ? (
          <Notice tone="bad" text="Couldn't save that. Try again." />
        ) : null}
      </div>
    </Section>
  );
}

/** Setup → Examples: Home's example gallery comes in a pack (a zip, or the folder it unzips to). Import it here; on first
 * run the app also looks for one next to itself. */
function ExamplesSetup({ open }: { open: boolean }) {
  const pack = useApp((s) => s.examplesPack);
  const mode = useApp((s) => s.mode);
  const [path, setPath] = useState("");
  const file = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (open) void loadExamplesPack();
  }, [open]);
  const busy = !!pack?.busy || !!pack?.importing;
  const n = pack?.examples ?? 0;
  return (
    <div className="space-y-4">
      <p className="flex items-center gap-2.5 text-[14px] text-ink">
        <StateDot on={!!pack?.installed && n > 0} />
        {busy ? "Installing…" : pack?.installed && n > 0 ? `${n} examples` : "No examples"}
        <Hint text="The examples come in a pickups_examples zip. Leave it in Downloads, on the Desktop or next to Pickups and it installs by itself (only when no examples are installed, so nothing is replaced), or import it here. Importing here again replaces the examples." />
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={file}
          type="file"
          accept=".zip,application/zip"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void importExamplesPack(f);
          }}
        />
        <Button variant="secondary" size="sm" loading={busy} disabled={mode !== "live" && mode !== "mock"} onClick={() => file.current?.click()}>
          Import pack…
        </Button>
        <form
          className="flex min-w-[260px] flex-1 items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (path.trim()) void importExamplesPack(path.trim()).then((ok) => ok && setPath(""));
          }}
        >
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="Zip or folder path" aria-label="Examples pack path" className="field mono h-8 min-w-0 flex-1 text-[12px]" spellCheck={false} />
          <Button type="submit" variant="ghost" size="sm" disabled={!path.trim() || busy}>
            Import
          </Button>
        </form>
      </div>
      {pack?.error && !busy ? <Notice tone="bad" text={pack.error} /> : null}
    </div>
  );
}

/** Setup > Storage: the library's folder and its size (the server walks it off its main thread), and Reveal. */
function StorageRow({ open }: { open: boolean }) {
  const [info, setInfo] = useState<StorageInfo | null>(null);
  const mode = useApp((s) => s.mode);
  useEffect(() => {
    if (!open) return;
    let live = true;
    getApi()
      .storage()
      .then((i) => live && setInfo(i))
      .catch(() => live && setInfo(null));
    return () => {
      live = false;
    };
  }, [open]);
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-3">
          <MiddleTruncate text={info?.path ?? "…"} className="mono flex-1 text-[13px] text-ink2" />
          {mode !== "mock" ? (
            <Button variant="secondary" size="sm" className="shrink-0" onClick={() => void revealLibrary()}>
              Reveal
            </Button>
          ) : null}
        </div>
        <div className="mono mt-1 text-[16px] text-ink">{info ? size(info.bytes) : "—"}</div>
      </div>
    </div>
  );
}

interface PhoneState {
  tailscale: boolean;
  enabled: boolean;
  url: string | null;
}

/** Setup > Use on your phone: with Tailscale here, a switch (off by default) that also serves Pickups on the Tailscale
 * address only, and then the phone's address with Copy and a QR code. Without it, one line and Get Tailscale. */
function PhoneRow({ open }: { open: boolean }) {
  const mode = useApp((s) => s.mode);
  const [st, setSt] = useState<PhoneState | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open || mode !== "live") return;
    let live = true;
    request<PhoneState>("GET", "/api/phone")
      .then((x) => live && setSt(x))
      .catch(() => live && setSt(null));
    return () => {
      live = false;
    };
  }, [open, mode]);
  if (!st) return null;
  const set = async (enabled: boolean) => {
    setBusy(true);
    try {
      setSt(await request<PhoneState>("POST", "/api/phone", { enabled }));
    } catch (e) {
      toast((e as Error).message || "Couldn't change that.", "error");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Section title="Use on your phone">
      {st.tailscale ? (
        <div className="space-y-5">
          <div className="flex items-center justify-between gap-6">
            <span className="text-[14px] text-ink">Allow phone access</span>
            <Switch checked={st.enabled} disabled={busy} onChange={(v) => void set(v)} label="Allow phone access" />
          </div>
          {st.enabled && st.url ? (
            <div className="flex flex-wrap items-center gap-6 animate-rise-in">
              <img src={`/api/phone/qr.svg?u=${encodeURIComponent(st.url)}`} alt={`QR code for ${st.url}`} className="h-36 w-36 rounded-card" />
              <div className="min-w-0 flex-1 space-y-3">
                <div className="mono break-all text-[14px] text-ink">{st.url}</div>
                <Button variant="secondary" size="sm" icon={Copy} onClick={() => void navigator.clipboard?.writeText(st.url!).then(() => toast("Copied", "success", undefined, 2000))}>
                  Copy
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
          <span className="text-[14px] text-ink">Open Pickups on your phone from anywhere.</span>
          <a href="https://tailscale.com/download" target="_blank" rel="noreferrer" className={linkCls}>
            Get Tailscale <ExternalLink size={12} strokeWidth={1.75} />
          </a>
        </div>
      )}
    </Section>
  );
}

/** Setup > Your machine: this computer against the minimum, one row each, found or short. GB are decimal, as the gates
 * read them (a 4090 is 25.8). */
function RequirementsTable({ req, cap, fallback }: {
  req: Requirements | null; cap: Capabilities | null; fallback: { gpu: string; vram: number; comfy: string | null } | null;
}) {
  const modelsNeed = diskNeedGb(cap?.features ?? []);
  const v = (x: number) => `${+x.toFixed(1)} GB`;
  type R = { label: string; value: string; min: string; ok: boolean | null };
  const rows: R[] = req
    ? [
        { label: "GPU", value: req.gpu ? shortGpu(req.gpu) : "—", min: "NVIDIA", ok: req.nvidia },
        { label: "VRAM", value: req.vram_gb ? v(req.vram_gb) : "—", min: v(req.min_vram_gb), ok: req.vram_gb >= req.min_vram_gb },
        ...(req.models_free_gb != null
          ? [{ label: "Models drive", value: `${v(req.models_free_gb)} free`, min: v(modelsNeed), ok: req.models_free_gb >= modelsNeed }]
          : []),
        ...(req.library_free_gb != null
          ? [{ label: "Library drive", value: `${v(req.library_free_gb)} free`, min: v(req.library_min_free_gb), ok: req.library_free_gb >= req.library_min_free_gb }]
          : []),
        { label: "ComfyUI", value: req.comfy_version ? req.comfy_version.replace(/^v/, "") : "—", min: `${req.min_comfy}+`, ok: req.comfy_ok },
        { label: "ffmpeg", value: req.ffmpeg != null ? String(req.ffmpeg) : "—", min: String(req.min_ffmpeg), ok: req.ffmpeg == null ? null : req.ffmpeg >= req.min_ffmpeg },
      ]
    : fallback
      ? [
          { label: "GPU", value: fallback.gpu, min: "NVIDIA", ok: null },
          { label: "VRAM", value: v(fallback.vram), min: "23.5 GB", ok: null },
          { label: "ComfyUI", value: fallback.comfy ?? "—", min: "0.41+", ok: null },
        ]
      : [];
  if (!rows.length) return <div className="h-24" aria-busy="true" />;
  return (
    <table className="w-full text-[14px]" aria-label="Requirements">
      <thead className="slate text-ink3">
        <tr>
          <th className="pb-2 text-left font-normal" />
          <th className="pb-2 text-left font-normal">This computer</th>
          <th className="pb-2 text-left font-normal">Minimum</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} className="border-t border-line">
            <td className="py-2 pr-3 text-ink2">{r.label}</td>
            <td className="py-2 pr-3">
              <span className="mono inline-flex items-center gap-2 text-ink">
                <span
                  className={cx("h-1.5 w-1.5 shrink-0 rounded-full", r.ok === null ? "border border-ink3" : r.ok ? "bg-[#3FB950]" : "bg-[#E3A34B]")}
                  aria-label={r.ok === null ? "unknown" : r.ok ? "found" : "short"}
                />
                {r.value}
              </span>
            </td>
            <td className="mono py-2 text-ink3">{r.min}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Setup > Models: the same feature cards as onboarding, each opening to its files and nodes, found or missing. */
function SetupFeatures({ cap, focus }: { cap: Capabilities | null; focus: TaskId | null }) {
  const [expanded, setExpanded] = useState<string | null>(focus);
  if (!cap) return <div className="h-24" aria-busy="true" />;
  return (
    <div className="grid grid-cols-1 gap-2">
      {cap.features.map((f) => (
        <FeatureCard key={f.task} f={f} expanded={expanded === f.task} onToggle={() => setExpanded(expanded === f.task ? null : f.task)} />
      ))}
    </div>
  );
}
