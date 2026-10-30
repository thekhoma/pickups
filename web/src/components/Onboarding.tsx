// The guided setup, five steps with one primary each: Connect (servers found as rows, one "Connect to <name>"), Scan (a
// checklist that shows its work), Choose (feature cards, nothing ticked), Fix (nodes, licences, models), Done.
// SETUP_ONBOARDING_PLAN.md. It reads /api/capabilities only, the same payload Setup reads, so the two never disagree.
// Nothing installs or downloads without a click, and nothing is ever written on a remote ComfyUI.
import { ChevronRight, Copy, ExternalLink, Lock, Search, X } from "lucide-react";
import { Children, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { closeOnboarding, navigate, openOnboarding, refreshStatus, toast } from "../actions";
import { request } from "../api";
import { FINISH_CHOICES, type FinishChoice } from "../lib/finish";
import { enterDemo } from "../lib/demo";
import { cx, gb } from "../lib/format";
import { useReducedMotion } from "../lib/loops";
import { scannedLine, serverName, servers, type Capabilities, type FoundServer, type Feature, type ListenCommand, type ModelsDownload, type ServerTest } from "../lib/servers";
import { setState, useApp, type OnboardingStep } from "../store";
import { Wordmark } from "./TopBar";
import { Button, Hint, IconButton, MiddleTruncate, Notice, ProgressLine, Segmented } from "./ui";

const STEPS: OnboardingStep[] = ["connect", "scan", "choose", "fix", "done"];
const MIN_COMFY = [0, 41];

export function Onboarding() {
  const { open, step, task } = useApp((s) => s.onboarding);
  const [cap, setCap] = useState<Capabilities | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [finish, setFinish] = useState<FinishChoice>("4k-hdr");
  const [runpod, setRunpod] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  // Finish's card, Fix and Done read the choice picked: 4K alone lists only SeedVR2.
  const view = useMemo(() => (cap ? { ...cap, features: cap.features.map((f) => withChoice(f, finish)) } : null), [cap, finish]);

  // Esc closes it like every other dialog (Skip for now before connecting, Close after).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopImmediatePropagation();
      closeOnboarding();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);
  // A feature that isn't ready opens Choose with its card ticked.
  useEffect(() => {
    if (open && task) setPicked((p) => (p.includes(task) ? p : [...p, task]));
  }, [open, task]);
  // Focus follows the step: its heading, so Tab reaches the primary next.
  useEffect(() => {
    if (open) heading.current?.focus({ preventScroll: true });
  }, [open, step]);

  if (!open) return null;
  const go = (s: OnboardingStep) => openOnboarding(s, { task: null });
  const at = STEPS.indexOf(step);
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-bg animate-fade-in" role="dialog" aria-modal="true" aria-label="Set up Pickups">
      <div className="mx-auto flex min-h-full w-full max-w-[720px] flex-col px-6 pb-28 pt-[7vh] max-[767px]:px-4 max-[767px]:pb-0 max-[767px]:pt-5">
        <div className="mb-10 flex items-center justify-between gap-4 max-[767px]:mb-7">
          <Wordmark size={26} />
          <ol className="flex items-center gap-2" aria-label="Steps">
            {STEPS.map((s, i) => (
              <li
                key={s}
                aria-current={i === at ? "step" : undefined}
                aria-label={s}
                className={cx("h-1.5 rounded-full transition-all duration-200 ease-out", i === at ? "w-5 bg-ivory" : i < at ? "w-1.5 bg-ivory/60" : "w-1.5 bg-line2")}
              />
            ))}
          </ol>
          <IconButton icon={X} label={cap?.connected ? "Close" : "Skip for now"} hint="Esc" tipAlign="end" onClick={closeOnboarding} />
        </div>
        {step === "connect" ? (
          <Connect heading={heading} onConnected={() => go("scan")} openRunpod={runpod} />
        ) : step === "scan" ? (
          <Scan heading={heading} onDone={(c) => { setCap(c); go("choose"); }} onLost={() => go("connect")} />
        ) : step === "choose" ? (
          <Choose
            heading={heading}
            cap={view}
            setCap={setCap}
            picked={picked}
            setPicked={setPicked}
            focusTask={task ?? null}
            onFix={() => go("fix")}
            finish={finish}
            setFinish={setFinish}
            onElsewhere={(pod) => {
              setRunpod(pod);
              go("connect");
            }}
          />
        ) : step === "fix" ? (
          <Fix heading={heading} cap={view} setCap={setCap} picked={picked} finish={finish} onDone={() => go("done")} onBack={() => go("choose")} />
        ) : (
          <Done heading={heading} cap={view} onChoose={() => go("choose")} />
        )}
      </div>
    </div>
  );
}

type HeadingRef = React.RefObject<HTMLHeadingElement>;

function Title({ heading, children }: { heading: HeadingRef; children: ReactNode }) {
  return (
    <h1 ref={heading} tabIndex={-1} className="display text-[36px] leading-[1.05] text-ivory outline-none max-[767px]:text-[28px]">
      {children}
    </h1>
  );
}

/** The step's one primary: inline on a wide screen, pinned to the bottom (44 px, full width) on a phone. */
function Primary({ children }: { children: ReactNode }) {
  if (!Children.toArray(children).length) return null; // nothing to do here: no empty sticky bar on a phone
  return (
    <div className="mt-10 flex flex-wrap items-center gap-3 max-[767px]:sticky max-[767px]:bottom-0 max-[767px]:z-10 max-[767px]:-mx-4 max-[767px]:mt-8 max-[767px]:border-t max-[767px]:border-line max-[767px]:bg-bg max-[767px]:px-4 max-[767px]:pb-[max(12px,env(safe-area-inset-bottom))] max-[767px]:pt-3 max-[767px]:[&>*]:flex-1">
      {children}
    </div>
  );
}

/** A check that draws itself in (200 ms), or an amber dot, or a ring while it works. */
function Mark({ state }: { state: "ok" | "warn" | "busy" | "off" }) {
  if (state === "busy") return <span className="inline-block h-3 w-3 shrink-0 animate-spin rounded-full border border-ink3 border-t-ivory" aria-hidden="true" />;
  if (state === "warn") return <span className="inline-block h-2 w-2 shrink-0 rounded-full bg-[#E3A34B]" aria-hidden="true" />;
  if (state === "off") return <span className="inline-block h-2 w-2 shrink-0 rounded-full border border-ink3" aria-hidden="true" />;
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" className="shrink-0 text-ivory" aria-hidden="true">
      <path d="M2.5 7.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="24" className="animate-draw" />
    </svg>
  );
}

function copy(text: string) {
  void navigator.clipboard?.writeText(text).then(() => toast("Copied", "success", undefined, 2000));
}

/** A command as one line with a Copy button (on a phone, its start in mono, never a wide code block). */
function CommandRow({ command, label }: { command: string; label?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-card border border-line bg-raised py-1.5 pl-3.5 pr-1.5">
      <code className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink" title={command}>
        {command}
      </code>
      <Button variant="secondary" size="sm" icon={Copy} onClick={() => copy(command)} aria-label={label ?? `Copy ${command.slice(0, 32)}`}>
        Copy
      </Button>
    </div>
  );
}

// --- 1. connect --------------------------------------------------------------------------------------------------------

type Row = { url: string; kind?: string; name: string; state: "probing" | "ok" | "off"; test?: ServerTest; recent: boolean; usedAt: number };

/** Pickups never downloads or runs an installer: Windows and Mac get ComfyUI's own download page, Linux its README's
 * install commands (github.com/Comfy-Org/ComfyUI). */
const COMFY_DOWNLOAD = "https://www.comfy.org/download";
const LINUX_INSTALL = [
  "git clone https://github.com/Comfy-Org/ComfyUI.git",
  "cd ComfyUI",
  "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130",
  "pip install -r requirements.txt",
  "python main.py",
].join("\n");
type InstallOs = "Windows" | "Mac" | "Linux";
const INSTALL_OSES: InstallOs[] = ["Windows", "Mac", "Linux"];
const guessOs = (): InstallOs => (/Mac|iPhone|iPad/.test(navigator.platform) ? "Mac" : /Linux|X11/.test(navigator.platform) ? "Linux" : "Windows");

const REASON: Record<string, (host: string) => string> = {
  refused: (h) => `${h} isn't accepting connections.`,
  dns: (h) => `Can't find ${h}.`,
  timeout: (h) => `${h} didn't answer.`,
  not_comfy: () => "That address isn't ComfyUI.",
  http: () => "ComfyUI answered with an error.",
  address: () => "Not an address.",
};

function Connect({ heading, onConnected, openRunpod = false }: { heading: HeadingRef; onConnected: () => void; openRunpod?: boolean }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [probing, setProbing] = useState(true);
  const [address, setAddress] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [failure, setFailure] = useState<ServerTest | null>(null);
  const [runpod, setRunpod] = useState(openRunpod);
  const [other, setOther] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [lan, setLan] = useState<"idle" | "scanning" | "patient" | "none" | "done">("idle");
  const [scanNote, setScanNote] = useState<string | null>(null);
  // A Mac can't render (no NVIDIA GPU), so on a Mac the install choice opens on Windows, the GPU machine's usual OS.
  const onMac = guessOs() === "Mac";
  const [installOs, setInstallOs] = useState<InstallOs>(() => (onMac ? "Windows" : guessOs()));
  const primaryRef = useRef<HTMLButtonElement>(null);

  const put = useCallback((row: Row) => setRows((rs) => [...rs.filter((r) => r.url !== row.url), row]), []);
  // Opening probes this computer and the recent servers at once; each row answers when it answers.
  const probe = useCallback(() => {
    setProbing(true);
    setRows([]);
    const local = servers
      .detect()
      .then((d) => {
        if (d.found?.url) put({ url: d.found.url, kind: "local", name: "This computer", state: "ok", test: d.found, recent: false, usedAt: 0 });
      })
      .catch(() => {});
    const recents = servers
      .list()
      .then((list) =>
        Promise.all(
          list.recent.slice(0, 5).map((r) => {
            const row: Row = { url: r.url, kind: r.kind, name: serverName(r.url, r.kind), state: "probing", recent: true, usedAt: r.used_at };
            put(row);
            return servers
              .test(r.url)
              .then((t) => put({ ...row, state: t.ok ? "ok" : "off", test: t }))
              .catch(() => put({ ...row, state: "off" }));
          }),
        ),
      )
      .catch(() => {});
    void Promise.all([local, recents]).then(() => setProbing(false));
  }, [put]);
  useEffect(probe, [probe]);

  // Find ComfyUI on my network: each server found joins the rows above, ready to connect. This computer, found again at
  // its network address, is listed only when its usual address didn't answer.
  // Nothing on the quick look: a second, more patient one (1.5 s per computer), then one plain line of what was looked
  // at and why it may have found nothing. The address box below stays for typing one in.
  const findOnNetwork = () => {
    setLan("scanning");
    setScanNote(null);
    const take = (found: FoundServer[]) => {
      const localOk = rows.some((r) => r.kind === "local" && r.state === "ok");
      const fresh = found.filter((f) => f.url && !(f.self && localOk) && !rows.some((r) => r.url === f.url && r.state === "ok"));
      fresh.forEach((f) => put({ url: f.url!, kind: "lan", name: f.name, state: "ok", test: f, recent: false, usedAt: 0 }));
      return fresh.length;
    };
    void servers
      .scan()
      .then(async (first) => {
        if (take(first.found)) return setLan("done");
        setLan("patient");
        const second = await servers.scan(true);
        if (take(second.found)) return setLan("done");
        setScanNote(scannedLine(second));
        setLan("none");
      })
      .catch(() => {
        setScanNote(null);
        setLan("none");
      });
  };

  const reachable = rows.filter((r) => r.state === "ok");
  // The best row: the last used NVIDIA one that answers, else this computer, else the first NVIDIA one that answers. A
  // computer that can't render (AMD, Apple) is still listed, never the primary.
  const best = [...reachable].filter((r) => /nvidia/i.test(r.test?.gpu_raw ?? r.test?.gpu ?? "")).sort((a, b) => b.usedAt - a.usedAt)[0];
  const order = [...rows].sort((a, b) => (a.state === "ok" ? 0 : a.state === "probing" ? 1 : 2) - (b.state === "ok" ? 0 : b.state === "probing" ? 1 : 2));
  useEffect(() => {
    if (best && !probing) primaryRef.current?.focus({ preventScroll: true });
  }, [best?.url, probing]); // eslint-disable-line react-hooks/exhaustive-deps

  const connect = async (url: string) => {
    setBusy(url);
    setFailure(null);
    try {
      const r = await servers.use(url);
      if (r.test && !r.test.ok) {
        setFailure(r.test);
        return;
      }
      await refreshStatus();
      onConnected();
    } catch (e) {
      const t = await servers.test(url).catch(() => null);
      setFailure(t && !t.ok ? t : { ok: false, url, reason: "refused", message: (e as Error).message });
    } finally {
      setBusy(null);
    }
  };
  const host = (u: string) => {
    try {
      return new URL(u.includes("://") ? u : `http://${u}`).hostname;
    } catch {
      return u;
    }
  };

  const none = !probing && !reachable.length;
  // Waiting: ask this computer every few seconds, and go on to Scan the moment ComfyUI answers.
  useEffect(() => {
    if (!waiting) return;
    let alive = true;
    let inFlight = false;
    const look = () => {
      if (inFlight || busy) return;
      inFlight = true;
      void servers
        .detect()
        .then((d) => {
          if (alive && d.found?.url) void connect(d.found.url);
        })
        .catch(() => {})
        .finally(() => {
          inFlight = false;
        });
    };
    look();
    const t = window.setInterval(look, 3000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [waiting]); // eslint-disable-line react-hooks/exhaustive-deps
  const checkNow = () => {
    setBusy("check");
    void servers
      .detect()
      .then((d) => (d.found?.url ? connect(d.found.url) : undefined))
      .catch(() => {})
      .finally(() => setBusy((b) => (b === "check" ? null : b)));
  };

  const addressForm = (
    <>
      <div className="mt-3 flex items-center gap-3 max-[767px]:flex-wrap">
        <Button variant="secondary" size="md" icon={Search} loading={lan === "scanning" || lan === "patient"} onClick={findOnNetwork}>
          {lan === "scanning" ? "Looking on your network…" : lan === "patient" ? "Looking again, more slowly…" : "Find ComfyUI on my network"}
        </Button>
        {lan === "none" && !scanNote ? (
          <span className="text-[14px] text-ink3" role="status">
            None found
          </span>
        ) : null}
      </div>
      {lan === "none" && scanNote ? (
        <p className="mt-2 text-[13px] leading-snug text-ink2" role="status" data-testid="scan-note">
          {scanNote}{" "}
          <span className="text-ink3">This computer may be blocking local network access for Pickups, or ComfyUI wasn't started with --listen.</span>
        </p>
      ) : null}
      <form
        className="mt-3 flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (address.trim()) void connect(address.trim());
        }}
      >
        <input
          className="field h-11 min-w-0 flex-1 text-[15px]"
          value={address}
          placeholder="http://gpu-pc:8188"
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          aria-label="Address"
          onChange={(e) => {
            setAddress(e.target.value);
            setFailure(null);
          }}
        />
        {address.trim() || (none && onMac) ? (
          <Button type="submit" variant={best ? "secondary" : "primary"} size="md" disabled={!address.trim()} loading={!!address.trim() && busy === address.trim()}>
            Connect
          </Button>
        ) : null}
      </form>
      <p className="mt-2 text-[13px] text-ink3">The GPU computer's name or IP, then :8188. Start ComfyUI there with --listen.</p>

      {failure ? (
        <div className="mt-4 space-y-3" role="alert">
          <Notice
            tone="warn"
            text={(REASON[failure.reason ?? "refused"] ?? REASON.refused)(host(failure.url ?? address))}
            action={failure.reason === "timeout" || failure.reason === "http" ? { label: "Try again", onClick: () => void connect(failure.url ?? address) } : undefined}
          />
          {failure.reason === "refused" && failure.listen?.length ? <Commands commands={failure.listen} defaultOs="Windows" /> : null}
        </div>
      ) : null}
    </>
  );
  const runpodRow = (
    <div className="mt-2">
      <button type="button" onClick={() => setRunpod((o) => !o)} aria-expanded={runpod} className="inline-flex h-9 items-center gap-1.5 text-[14px] text-ink2 transition-colors hover:text-ivory">
        <ChevronRight size={14} className={cx("transition-transform duration-150", runpod && "rotate-90")} /> RunPod
      </button>
      {runpod ? <RunPodSteps onConnect={(u) => void connect(u)} busy={!!busy} /> : null}
    </div>
  );

  // No ComfyUI anywhere yet: the whole app on example shots, one click away (Exit demo in the top bar comes back here).
  const demoRow = (
    <div className="mt-10 max-[767px]:[&>*]:w-full">
      <Button variant="secondary" size="lg" onClick={enterDemo}>
        Try the demo
      </Button>
    </div>
  );

  if (none && waiting) {
    return (
      <div className="animate-rise-in">
        <Title heading={heading}>Waiting for ComfyUI…</Title>
        <div className="mt-10 flex justify-center" role="status" aria-label="Waiting for ComfyUI">
          <span className="relative flex h-16 w-16 items-center justify-center">
            <span className="absolute inset-0 rounded-full bg-ivory/10 motion-safe:animate-probe" aria-hidden="true" />
            <span className="h-3 w-3 rounded-full bg-ivory motion-safe:animate-probe" aria-hidden="true" />
          </span>
        </div>
        <Primary>
          <Button ref={primaryRef} variant="primary" size="lg" loading={!!busy} onClick={checkNow}>
            Check now
          </Button>
          <Button variant="ghost" size="lg" onClick={() => setWaiting(false)}>
            Back
          </Button>
        </Primary>
      </div>
    );
  }

  if (none && onMac) {
    // REVIEW F1: a Mac can't render, so the judge's path is the GPU machine's address: the field and Connect first,
    // the install for that machine (Windows or Linux) folded under it.
    const offline = order.filter((r) => r.recent);
    return (
      <div className="animate-rise-in">
        <Title heading={heading}>Where does ComfyUI run?</Title>
        <div className="mt-6">{addressForm}</div>
        {offline.length ? (
          <ul className="mt-3 space-y-2">
            {offline.map((r) => (
              <li key={r.url}>
                <button type="button" onClick={() => setAddress(r.url)} className="flex h-11 w-full items-center gap-4 rounded-card border border-line px-4 text-left opacity-60 hover:opacity-100">
                  <span className="h-2 w-2 shrink-0 rounded-full border border-ink3" aria-hidden="true" />
                  <span className="min-w-0 flex-1 truncate text-[14px] text-ink">{r.name}</span>
                  <span className="slate shrink-0 text-ink3">{r.test?.reason === "not_comfy" ? "Not ComfyUI" : "Offline"}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        <div className="mt-10 border-t border-line pt-3">
          {runpodRow}
          <button type="button" onClick={() => setOther((o) => !o)} aria-expanded={other} className="inline-flex h-9 items-center gap-1.5 text-[14px] text-ink2 transition-colors hover:text-ivory">
            <ChevronRight size={14} className={cx("transition-transform duration-150", other && "rotate-90")} /> Install ComfyUI on the GPU computer
          </button>
          {other ? (
            <div className="animate-rise-in space-y-4 pb-2 pt-2">
              <Segmented value={installOs} onChange={(v) => setInstallOs(v as InstallOs)} options={INSTALL_OSES.filter((o) => o !== "Mac").map((o) => ({ value: o, label: o }))} label="Operating system" />
              {installOs === "Linux" ? (
                <pre className="overflow-x-auto whitespace-pre rounded-card border border-line bg-raised px-3.5 py-2.5 font-mono text-[12px] leading-relaxed text-ink [scrollbar-width:thin]">{LINUX_INSTALL}</pre>
              ) : null}
              <Button
                variant="secondary"
                size="md"
                icon={installOs === "Linux" ? Copy : ExternalLink}
                onClick={() => (installOs === "Linux" ? copy(LINUX_INSTALL) : window.open(COMFY_DOWNLOAD, "_blank", "noopener,noreferrer"))}
              >
                {installOs === "Linux" ? "Copy commands" : `Download for ${installOs}`}
              </Button>
            </div>
          ) : null}
        </div>
        {demoRow}
      </div>
    );
  }

  if (none) {
    const offline = order.filter((r) => r.recent);
    return (
      <div className="animate-rise-in">
        <Title heading={heading}>Install ComfyUI</Title>
        <div className="mt-6">
          <Segmented value={installOs} onChange={(v) => setInstallOs(v as InstallOs)} options={INSTALL_OSES.map((o) => ({ value: o, label: o }))} label="Operating system" />
        </div>
        {installOs === "Linux" ? (
          <div className="mt-4">
            <pre className="overflow-x-auto whitespace-pre rounded-card border border-line bg-raised px-3.5 py-2.5 font-mono text-[12px] leading-relaxed text-ink [scrollbar-width:thin]">
              {LINUX_INSTALL}
            </pre>
          </div>
        ) : null}
        {/* The install action sits with the OS it's for; the other ways in stay quiet below it. */}
        <div className="mt-4 max-[767px]:[&>*]:w-full">
          {installOs === "Linux" ? (
            <Button
              ref={primaryRef}
              variant="primary"
              size="lg"
              icon={Copy}
              onClick={() => {
                copy(LINUX_INSTALL);
                setWaiting(true);
              }}
            >
              Copy commands
            </Button>
          ) : (
            <Button
              ref={primaryRef}
              variant="primary"
              size="lg"
              icon={ExternalLink}
              onClick={() => {
                window.open(COMFY_DOWNLOAD, "_blank", "noopener,noreferrer");
                setWaiting(true);
              }}
            >
              Download for {installOs}
            </Button>
          )}
        </div>

        <div className="mt-10 border-t border-line pt-3">
          <button type="button" onClick={() => setOther((o) => !o)} aria-expanded={other} className="inline-flex h-9 items-center gap-1.5 text-[14px] text-ink2 transition-colors hover:text-ivory">
            <ChevronRight size={14} className={cx("transition-transform duration-150", other && "rotate-90")} /> ComfyUI runs on another computer
          </button>
          {other ? (
            <div className="animate-rise-in pb-2">
              {offline.length ? (
                <ul className="mt-2 space-y-2">
                  {offline.map((r) => (
                    <li key={r.url} className="flex h-11 items-center gap-4 rounded-card border border-line px-4 opacity-60">
                      <span className="h-2 w-2 shrink-0 rounded-full border border-ink3" aria-hidden="true" />
                      <span className="min-w-0 flex-1 truncate text-[14px] text-ink">{r.name}</span>
                      <span className="slate shrink-0 text-ink3">{r.test?.reason === "not_comfy" ? "Not ComfyUI" : "Offline"}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              {addressForm}
            </div>
          ) : null}
          {runpodRow}
        </div>
        {demoRow}
      </div>
    );
  }

  return (
    <div className="animate-rise-in">
      <Title heading={heading}>Where does ComfyUI run?</Title>
      <ul className="mt-8 space-y-2" aria-live="polite">
        {order.map((r, i) => (
          <li key={r.url} style={{ animationDelay: `${i * 50}ms` }} className="animate-rise-in">
            <button
              type="button"
              disabled={r.state !== "ok" || !!busy}
              onClick={() => void connect(r.url)}
              className={cx(
                "flex h-14 w-full items-center gap-4 rounded-card border px-4 text-left transition-colors duration-150",
                r.url === best?.url ? "border-ivory/60 bg-raised" : "border-line hover:border-line2",
                r.state === "off" && "opacity-60",
              )}
            >
              <span
                className={cx("h-2 w-2 shrink-0 rounded-full", r.state === "probing" ? "animate-probe bg-ivory" : r.state === "ok" ? "bg-[#3FB950]" : "border border-ink3")}
                aria-hidden="true"
              />
              <span className="shrink-0 text-[15px] text-ink">{r.name}</span>
              <span className="slate min-w-0 flex-1 truncate text-right text-ink3">
                {r.state === "ok" ? [r.test?.gpu, r.test?.vram_gb ? `${r.test.vram_gb.toFixed(1)} GB` : null].filter(Boolean).join(" · ") : r.state === "off" ? (r.test?.reason === "not_comfy" ? "Not ComfyUI" : "Offline") : ""}
              </span>
            </button>
          </li>
        ))}
        {probing && !rows.length ? (
          <li className="flex h-14 items-center gap-4 rounded-card border border-line px-4">
            <span className="h-2 w-2 shrink-0 animate-probe rounded-full bg-ivory" aria-hidden="true" />
            <span className="text-[15px] text-ink2">This computer</span>
          </li>
        ) : null}
      </ul>

      {addressForm}

      <div className="mt-4">{runpodRow}</div>

      <Primary>
        {best ? (
          <Button ref={primaryRef} variant="primary" size="lg" loading={busy === best.url} onClick={() => void connect(best.url)}>
            Connect to {best.name}
          </Button>
        ) : null}
      </Primary>
    </div>
  );
}

function RunPodSteps({ onConnect, busy }: { onConnect: (url: string) => void; busy: boolean }) {
  const [url, setUrl] = useState("");
  return (
    <ol className="mt-4 space-y-4 border-l border-line pl-4 text-[14px] text-ink2 animate-rise-in">
      <li className="flex items-center gap-2">
        <span className="text-ink">Deploy a pod with ComfyUI on port 8188</span>
        <Hint text="A GPU with 24 GB or more, ComfyUI with ComfyUI-LTXVideo, and a network volume for the models. RunPod bills by the minute while the pod runs, even idle: stop it there when you're done." />
        <a href="https://console.runpod.io/deploy" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-ivory/80 hover:text-ivory">
          Open <ExternalLink size={12} />
        </a>
      </li>
      <li>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (url.trim()) onConnect(url.trim());
          }}
        >
          <input className="field h-10 min-w-0 flex-1 text-[14px]" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://<pod-id>-8188.proxy.runpod.net" aria-label="Pod address" spellCheck={false} />
          <Button type="submit" variant="secondary" size="md" disabled={!url.trim()} loading={busy}>
            Connect
          </Button>
        </form>
      </li>
    </ol>
  );
}

/** How to start ComfyUI so other computers can reach it, per OS, with a copy button. */
export function Commands({ commands, intro, defaultOs }: { commands: ListenCommand[]; intro?: string; defaultOs?: string }) {
  const guess = defaultOs ?? (/Mac/.test(navigator.platform) ? "Mac" : /Linux/.test(navigator.platform) ? "Linux" : "Windows");
  const [os, setOs] = useState(commands.some((c) => c.os === guess) ? guess : commands[0]?.os ?? "Windows");
  if (!commands.length) return null;
  const c = commands.find((x) => x.os === os) ?? commands[0];
  return (
    <div className="space-y-3">
      {intro ? <p className="text-[14px] text-ink2">{intro}</p> : null}
      <Segmented value={os} onChange={(v) => setOs(v)} options={commands.map((x) => ({ value: x.os, label: x.os }))} label="Operating system" />
      <CommandRow command={c.command} />
    </div>
  );
}

// --- 2. scan -----------------------------------------------------------------------------------------------------------

function older(v: string | null | undefined): boolean {
  if (!v) return false;
  const [a, b] = v.split(".").map((x) => Number(x) || 0);
  return a < MIN_COMFY[0] || (a === MIN_COMFY[0] && b < MIN_COMFY[1]);
}

function Scan({ heading, onDone, onLost }: { heading: HeadingRef; onDone: (c: Capabilities) => void; onLost: () => void }) {
  const reduced = useReducedMotion();
  const [cap, setCap] = useState<Capabilities | null>(null);
  const [failed, setFailed] = useState(false);
  const [shown, setShown] = useState(0);
  const run = useCallback(() => {
    setCap(null);
    setFailed(false);
    setShown(0);
    servers.capabilities().then(setCap).catch(() => setFailed(true));
  }, []);
  useEffect(run, [run]);

  const m = cap?.machine;
  const nodesKnown = !!cap?.features.every((f) => f.nodes?.checked !== false);
  const old = older(cap?.comfy_version ?? m?.comfy_version);
  const rows: Array<{ key: string; label: string; state: "ok" | "warn" }> = cap?.connected
    ? [
        { key: "comfy", label: `ComfyUI ${cap.comfy_version ?? m?.comfy_version ?? ""}`.trim(), state: old ? "warn" : "ok" },
        { key: "gpu", label: [m?.gpu, m?.vram_gb ? `${m.vram_gb.toFixed(1)} GB` : null].filter(Boolean).join(" · ") || "GPU", state: "ok" },
        { key: "nodes", label: nodesKnown ? "Nodes" : "Couldn't read nodes", state: nodesKnown ? "ok" : "warn" },
        { key: "models", label: "Models", state: cap.features.some((f) => f.needs.some((n) => n.installed === null)) ? "warn" : "ok" },
        ...cap.features.map((f) => ({ key: f.task, label: f.label, state: (f.ready ? "ok" : "warn") as "ok" | "warn" })),
      ]
    : [];
  // A staged reveal of the one fetch: 60 ms a row, 600 ms at most (all at once with reduced motion).
  useEffect(() => {
    if (!rows.length) return;
    if (reduced) {
      setShown(rows.length);
      return;
    }
    const step = Math.min(60, 600 / rows.length);
    const t = window.setInterval(() => setShown((n) => (n >= rows.length ? n : n + 1)), step);
    return () => window.clearInterval(t);
  }, [rows.length, reduced]);
  // Done: hold 400 ms so the eye lands on the last check, then Choose (never on unread nodes).
  useEffect(() => {
    if (!cap?.connected || shown < rows.length || !nodesKnown) return;
    const t = window.setTimeout(() => onDone(cap), reduced ? 0 : 400);
    return () => window.clearTimeout(t);
  }, [cap, shown, rows.length, nodesKnown, reduced, onDone]);

  if (failed || (cap && !cap.connected))
    return (
      <div className="animate-rise-in">
        <Title heading={heading}>Lost the connection</Title>
        <Primary>
          <Button variant="primary" size="lg" onClick={onLost}>
            Connect again
          </Button>
        </Primary>
      </div>
    );
  const name = cap ? serverName(cap.server.url, cap.server.kind) : "";
  return (
    <div className="animate-rise-in">
      <Title heading={heading}>{cap ? name : "Checking"}</Title>
      <ul className="mt-8 space-y-0.5" aria-live="polite">
        {(cap ? rows.slice(0, shown) : [{ key: "wait", label: "ComfyUI", state: "busy" as const }]).map((r) => (
          <li key={r.key} className="flex h-10 items-center gap-3 border-b border-line text-[15px] text-ink animate-rise-in">
            <Mark state={r.state} />
            <span className={cx("min-w-0 flex-1 truncate", r.state === "warn" && "text-ink")}>{r.label}</span>
          </li>
        ))}
        {cap && shown < rows.length ? (
          <li className="flex h-10 items-center gap-3 border-b border-line text-[15px] text-ink3">
            <Mark state="busy" />
            <span>{rows[shown]?.label}</span>
          </li>
        ) : null}
      </ul>
      {cap && old && shown >= rows.length ? (
        <Notice
          className="mt-6"
          tone="warn"
          text={`ComfyUI ${cap.comfy_version ?? m?.comfy_version} is too old for ${cap.features.filter((f) => !f.ready).length} features.`}
          action={{ label: "Copy update command", onClick: () => copy("git pull && pip install -r requirements.txt") }}
        />
      ) : null}
      {cap && !nodesKnown && shown >= rows.length ? (
        <Primary>
          <Button variant="primary" size="lg" onClick={run}>
            Scan again
          </Button>
        </Primary>
      ) : null}
    </div>
  );
}

// --- 3. choose ---------------------------------------------------------------------------------------------------------

/** The files the picked features are missing, each once (LTX core is shared by most), and their total. */
export function missingFiles(features: Feature[]) {
  const by = new Map<string, Feature["needs"][number]>();
  for (const f of features) for (const n of f.needs) if (n.installed === false) by.set(n.file, n);
  const files = [...by.values()];
  return { files, gb: files.reduce((s, n) => s + n.gb, 0) };
}

/** Finish as the choice picked: that choice's files, size and readiness (the others untouched). */
export function withChoice(f: Feature, choice: FinishChoice): Feature {
  const c = f.task === "finish" ? f.choices?.[choice] : undefined;
  return c ? { ...f, needs: c.needs, missing_gb: c.missing_gb, ready: c.ready } : f;
}

/** Free space the models drive needs (SYSTEM_REQUIREMENTS.md c): the missing files once, plus the most headroom any
 * feature still missing files wants (+10 GB video, +20 Finish, +5 Image). With nothing missing, the headroom alone:
 * room for ComfyUI's input/ and output/ to grow. Fix and Setup both read this. */
export function diskNeedGb(features: Feature[]): number {
  const short = features.filter((f) => f.needs.some((n) => n.installed === false));
  return missingFiles(features).gb + Math.max(0, ...(short.length ? short : features).map((f) => f.headroom_gb ?? 10));
}

/** A run of words that never splits across lines (a GPU name, "4.2 GB free"). */
function Whole({ children }: { children: React.ReactNode }) {
  return <span className="whitespace-nowrap">{children}</span>;
}

function DiskLine({ free, need }: { free: number; need: number }) {
  return (
    <>
      <Whole>{gb(free)} free</Whole> · <Whole>needs {gb(need)}</Whole>
    </>
  );
}

/** A card's state, from its files and nodes. The GPU caveat (not supported, not tested) is said once above the cards, so
 * an installed feature on such a GPU shows no state rather than "Ready". */
function pill(f: Feature): { text: string; tone: "ok" | "warn" } {
  if (f.ready) return f.support && f.support !== "ok" ? { text: "", tone: "warn" } : { text: "Ready", tone: "ok" };
  if (f.blocked === "vram") return { text: `${+(f.vram_needed_gb ?? f.vram_gb ?? 24).toFixed(1)} GB GPU`, tone: "warn" };
  if (f.reason === "nodes_unknown") return { text: "Nodes unknown", tone: "warn" };
  if (f.reason === "comfy" || f.nodes?.core.length) return { text: "Update ComfyUI", tone: "warn" };
  if (f.missing_nodes.length) return { text: "Nodes", tone: "warn" };
  if (f.missing_gb) return { text: gb(f.missing_gb), tone: "warn" };
  return { text: "Unknown", tone: "warn" };
}

function useCapabilities(cap: Capabilities | null, setCap: (c: Capabilities) => void) {
  const [checking, setChecking] = useState(false);
  const check = useCallback(async () => {
    setChecking(true);
    try {
      await refreshStatus();
      setCap(await servers.capabilities());
    } finally {
      setChecking(false);
    }
  }, [setCap]);
  useEffect(() => {
    if (!cap) void check();
  }, [cap, check]);
  return { checking, check };
}

function Choose({ heading, cap, setCap, picked, setPicked, focusTask, onFix, finish, setFinish, onElsewhere }: {
  heading: HeadingRef; cap: Capabilities | null; setCap: (c: Capabilities) => void; picked: string[]; setPicked: (p: string[]) => void;
  focusTask: string | null; onFix: () => void; finish: FinishChoice; setFinish: (c: FinishChoice) => void;
  onElsewhere: (runpod: boolean) => void;
}) {
  useCapabilities(cap, setCap);
  const [open, setOpen] = useState<string | null>(focusTask);
  const primary = useRef<HTMLButtonElement>(null);
  const firstBox = useRef<HTMLInputElement>(null);
  const features = useMemo(
    () => [...(cap?.features ?? [])].sort((a, b) => Number(b.ready) - Number(a.ready) || a.missing_gb - b.missing_gb),
    [cap],
  );
  const chosen = features.filter((f) => picked.includes(f.task) && !f.ready);
  const total = missingFiles(chosen);
  useEffect(() => {
    if (!cap) return;
    if (focusTask) primary.current?.focus({ preventScroll: true });
    else if (features.every((f) => f.ready)) primary.current?.focus({ preventScroll: true });
    else firstBox.current?.focus({ preventScroll: true });
  }, [cap]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!cap)
    return (
      <div className="animate-rise-in">
        <Title heading={heading}>Features</Title>
        <p className="mt-8 flex items-center gap-3 text-[15px] text-ink3">
          <Mark state="busy" />
        </p>
      </div>
    );
  const toggle = (t: string) => setPicked(picked.includes(t) ? picked.filter((x) => x !== t) : [...picked, t]);
  const req = cap.requirements;
  const unsupported = req?.nvidia === false;
  const untested = !!req?.nvidia && req.vram_gb < req.min_vram_gb;
  let firstSet = false;
  return (
    <div className="animate-rise-in">
      <Title heading={heading}>Features</Title>
      {unsupported ? (
        <Notice className="mt-6" tone="warn" label="GPU" text={<>Not supported on this computer{req?.gpu ? <> <Whole>({req.gpu})</Whole></> : null}</>} />
      ) : untested ? (
        <Notice className="mt-6" tone="warn" label="GPU" text={<>Not tested on this GPU <Whole>({[req?.gpu, `${req?.vram_gb} GB`].filter(Boolean).join(" · ")})</Whole></>} />
      ) : null}
      <div className="mt-8 grid grid-cols-2 gap-3 max-[767px]:grid-cols-1">
        {features.map((f, i) => {
          const expanded = open === f.task;
          const box = !f.ready && !firstSet ? ((firstSet = true), firstBox) : undefined;
          return (
            <FeatureCard
              key={f.task}
              f={f}
              delay={i * 40}
              expanded={expanded}
              onToggle={() => setOpen(expanded ? null : f.task)}
              picked={picked.includes(f.task)}
              onPick={() => toggle(f.task)}
              boxRef={box}
              choice={f.task === "finish" && f.choices ? finish : undefined}
              onChoice={setFinish}
            />
          );
        })}
      </div>
      <Primary>
        {unsupported ? (
          <>
            <Button ref={primary} variant="primary" size="lg" onClick={() => onElsewhere(false)}>
              Use another computer
            </Button>
            <Button variant="secondary" size="lg" onClick={() => onElsewhere(true)}>
              RunPod
            </Button>
            {chosen.length ? (
              <Button variant="ghost" size="lg" onClick={onFix}>
                Install anyway
              </Button>
            ) : null}
          </>
        ) : chosen.length ? (
          <Button ref={primary} variant="primary" size="lg" onClick={onFix}>
            Fix {chosen.length} {chosen.length === 1 ? "feature" : "features"}
            {total.gb ? ` · ${gb(total.gb)}` : ""}
          </Button>
        ) : (
          <Button
            ref={primary}
            variant="primary"
            size="lg"
            onClick={() => {
              closeOnboarding();
              navigate({ name: "home" });
            }}
          >
            Start creating
          </Button>
        )}
      </Primary>
    </div>
  );
}

/** A feature as a card: its name and state, opening to its models and nodes; a tick when it isn't ready and can be
 * picked (Choose). Open, it spans the whole grid so no column is left empty. Shared by onboarding and Setup. */
export function FeatureCard({ f, expanded, onToggle, picked, onPick, boxRef, delay = 0, choice, onChoice }: {
  f: Feature; expanded: boolean; onToggle: () => void; picked?: boolean; onPick?: () => void; boxRef?: React.RefObject<HTMLInputElement>; delay?: number;
  choice?: FinishChoice; onChoice?: (c: FinishChoice) => void;
}) {
  const p = pill(f);
  // Not supported or not tested on this GPU (said once above the cards): the name dims, the pill stays readable.
  const dim = !!f.support && f.support !== "ok";
  return (
    <div style={{ animationDelay: `${delay}ms` }} className={cx("animate-rise-in rounded-card border transition-colors duration-150", expanded ? "col-span-full border-line2 bg-raised" : "border-line")}>
      <div className="flex items-center gap-3 px-4 py-3.5">
        <button type="button" onClick={onToggle} aria-expanded={expanded} className={cx("flex min-w-0 flex-1 items-center gap-2.5 text-left", dim && !expanded && "opacity-50")}>
          <ChevronRight size={14} className={cx("shrink-0 text-ink3 transition-transform duration-150 ease-out", expanded && "rotate-90")} />
          <span className="min-w-0 flex-1 truncate text-[16px] text-ink">{f.label}</span>
        </button>
        <span className={cx("slate inline-flex shrink-0 items-center gap-1.5", p.tone === "ok" ? "text-ink2" : "text-ink")}>
          {p.tone === "ok" ? <span className="h-1.5 w-1.5 rounded-full bg-[#3FB950]" aria-hidden="true" /> : null}
          {p.text}
        </span>
        {/* Nothing to fix on this computer: a GPU it can't run (too little VRAM, or not NVIDIA) offers no download. */}
        {!f.ready && onPick && f.blocked !== "vram" && f.support !== "unsupported" ? (
          <input ref={boxRef} type="checkbox" checked={!!picked} onChange={onPick} aria-label={`Fix ${f.label}`} className="h-4 w-4 shrink-0 accent-[#EEE9E0]" />
        ) : null}
      </div>
      {expanded && choice && onChoice ? (
        <div className="border-t border-line px-4 pt-3">
          <Segmented value={choice} onChange={(v) => onChoice(v as FinishChoice)} options={FINISH_CHOICES} label="Finish to" />
        </div>
      ) : null}
      {expanded ? <FeatureDetail f={f} /> : null}
    </div>
  );
}

/** A card's insides: its model files and node packs, each found or missing (shared by Choose and Setup). */
export function FeatureDetail({ f }: { f: Feature }) {
  const core = f.nodes?.core ?? [];
  return (
    <div className="space-y-4 border-t border-line px-4 pb-4 pt-3 animate-fade-in">
      <div>
        <div className="slate mb-1.5 text-ink3">Models</div>
        <ul className="space-y-1">
          {f.needs.map((n) => (
            <li key={n.file} className="flex items-start gap-2.5 text-[13px]">
              <span className="mt-[3px] flex h-3.5 w-3.5 shrink-0 items-center justify-center">
                <Mark state={n.installed ? "ok" : n.installed === null ? "off" : "warn"} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex min-w-0 items-start gap-1.5 text-ink">
                  <span className="min-w-0 break-all" title={n.file}>{n.file}</span>
                  {n.gated && n.installed === false ? <Lock size={11} className="mt-[3px] shrink-0 text-ink3" aria-label="needs a license" /> : null}
                </span>
                <span className="mono block text-[12px] text-ink3">{n.folder}</span>
              </span>
              <span className="mono shrink-0 text-[12px] text-ink3">{gb(n.gb)}</span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <div className="slate mb-1.5 text-ink3">Nodes</div>
        <ul className="space-y-1 text-[13px]">
          {f.nodes?.checked === false ? (
            <li className="flex items-center gap-2.5 text-ink">
              <Mark state="warn" /> Couldn't read nodes
            </li>
          ) : null}
          {core.length ? (
            <li className="flex items-center gap-2.5 text-ink">
              <Mark state="warn" /> ComfyUI <span className="mono text-[12px] text-ink3">{core.join(", ")}</span>
            </li>
          ) : null}
          {f.missing_nodes.map((p) => (
            <li key={p.pack} className="flex items-center gap-2.5 text-ink">
              <Mark state="warn" /> {p.pack}
            </li>
          ))}
          {f.nodes?.checked !== false && !core.length && !f.missing_nodes.length ? (
            <li className="flex items-center gap-2.5 text-ink2">
              <Mark state="ok" /> All found
            </li>
          ) : null}
        </ul>
      </div>
    </div>
  );
}

// --- 4. fix ------------------------------------------------------------------------------------------------------------

/** The Fix screen's install kinds: one tab row picks the commands for every section. */
type Os = "Windows" | "Mac" | "Linux" | "Desktop app";
const OSES: Os[] = ["Windows", "Mac", "Linux", "Desktop app"];

/** Update ComfyUI the way it was installed: the Windows portable's script, the Desktop app's own updater, or git. */
function updateCommand(os: Os): string | null {
  if (os === "Desktop app") return null;
  if (os === "Windows") return "update\\update_comfyui.bat";
  return "git pull && pip install -r requirements.txt";
}

/** A file's path in its Hugging Face repo, from its resolve URL. */
function repoPath(n: Feature["needs"][number]): string {
  return n.url.split("/resolve/main/")[1] ?? n.file;
}

function joinPath(base: string, folder: string, windows: boolean): string {
  const sep = windows ? "\\" : "/";
  return base.replace(/[\\/]+$/, "") + sep + folder;
}

/** `hf download` per file into the ComfyUI models folder: a path that starts with its folder goes to models/, any other
 * into models/<folder>/ (ComfyUI finds files in subfolders). */
function hfCommands(files: Feature["needs"], modelsDir: string, os: Os): string {
  const windows = os === "Windows" || (os === "Desktop app" && /^[A-Za-z]:\\/.test(modelsDir));
  const lines = ["pip install -U huggingface_hub", "hf auth login"];
  for (const n of files) {
    const path = repoPath(n);
    const dir = path === `${n.folder}/${n.file}` ? modelsDir : joinPath(modelsDir, n.folder, windows);
    lines.push(`hf download ${n.repo} ${path} --local-dir "${dir}"`);
  }
  return lines.join("\n");
}

function CommandBlock({ command, label }: { command: string; label: string }) {
  return (
    <div className="flex items-start gap-2 rounded-card border border-line bg-raised py-2 pl-3.5 pr-1.5">
      <pre className="min-w-0 flex-1 overflow-x-auto whitespace-pre pb-1 font-mono text-[12px] leading-relaxed text-ink [scrollbar-width:thin]">{command}</pre>
      <Button variant="secondary" size="sm" icon={Copy} onClick={() => copy(command)} aria-label={label}>
        Copy
      </Button>
    </div>
  );
}

function revealFolder(folder: string) {
  void request<{ path: string }>("POST", "/api/models/reveal", { folder }).catch((e) => toast((e as Error).message || "Couldn't open the folder.", "error"));
}

function Fix({ heading, cap, setCap, picked, finish, onDone, onBack }: {
  heading: HeadingRef; cap: Capabilities | null; setCap: (c: Capabilities) => void; picked: string[]; finish: FinishChoice; onDone: () => void;
  onBack: () => void;
}) {
  const reduced = useReducedMotion();
  const { checking, check } = useCapabilities(cap, setCap);
  const download = useApp((s) => s.modelsDownload);
  const [error, setError] = useState<string | null>(null);
  const machineOs = cap?.machine?.os ?? "";
  const [os, setOs] = useState<Os>(/nt|win/i.test(machineOs) ? "Windows" : /darwin|mac/i.test(machineOs) ? "Mac" : machineOs ? "Linux" : "Windows");
  const chosen = (cap?.features ?? []).filter((f) => picked.includes(f.task));
  const open = chosen.filter((f) => !f.ready);
  const packs = [...new Map(open.flatMap((f) => f.missing_nodes).map((p) => [p.pack, p])).values()];
  const core = [...new Set(open.flatMap((f) => f.nodes?.core ?? []))];
  const { files } = missingFiles(open);
  const gated = files.filter((n) => n.gated);
  const plain = files.filter((n) => !n.gated);
  const repos = [...new Map(gated.map((n) => [n.page, n])).values()];
  const here = !!cap?.downloads?.here;
  const modelsDir = (here ? cap?.models_dir_here : cap?.models_dir) ?? cap?.models_dir ?? "<ComfyUI>/models";
  const running = !!download?.running;
  const plainGb = plain.reduce((s, n) => s + n.gb, 0);
  const update = updateCommand(os);
  // Free space: the missing files once, plus the most headroom any picked feature wants (SYSTEM_REQUIREMENTS.md c).
  const req = cap?.requirements;
  const needGb = diskNeedGb(open);
  const lowModels = req?.models_free_gb != null && files.length > 0 && req.models_free_gb < needGb;
  const lowLibrary = req?.library_free_gb != null && req.library_free_gb < req.library_min_free_gb;

  // Everything picked resolved: Done after 400 ms (at once with reduced motion).
  useEffect(() => {
    if (!cap || !chosen.length || open.length) return;
    const t = window.setTimeout(onDone, reduced ? 0 : 400);
    return () => window.clearTimeout(t);
  }, [cap, chosen.length, open.length, reduced, onDone]);
  // A finished download re-checks what's there.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) void check();
    wasRunning.current = running;
  }, [running, check]);

  const start = async () => {
    setError(null);
    try {
      const d = await servers.download(open.filter((f) => f.needs.some((n) => n.installed === false && !n.gated)).map((f) => f.task), finish);
      setState({ modelsDownload: d, modelsFetching: d.running });
      followProgress();
    } catch (e) {
      setError((e as Error).message || "The download didn't start.");
    }
  };

  if (!cap) return null;
  const doneFiles = new Set((download?.files ?? []).slice(0, Math.max(0, (download?.files ?? []).indexOf(download?.file ?? ""))));
  const showOs = core.length > 0 || (!here && files.length > 0);
  return (
    <div className="animate-rise-in">
      <Title heading={heading}>Get what's missing</Title>
      {/* One short line each that fits a phone: the drive is the label, the numbers never split. */}
      {lowModels ? (
        <Notice className="mt-6" tone="warn" label="Models drive" text={<DiskLine free={req!.models_free_gb!} need={needGb} />} />
      ) : null}
      {lowLibrary ? (
        <Notice className="mt-3" tone="warn" label="Library drive" text={<DiskLine free={req!.library_free_gb!} need={req!.library_min_free_gb} />} />
      ) : null}
      {showOs ? (
        <div className="mt-6">
          <Segmented value={os} onChange={(v) => setOs(v as Os)} options={OSES.map((o) => ({ value: o, label: o }))} label="How ComfyUI is installed" />
        </div>
      ) : null}

      {packs.length || core.length ? (
        <section className="mt-8 space-y-3">
          <div className="slate text-ink3">Nodes</div>
          {core.length ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2.5 text-[15px] text-ink">
                <Mark state="warn" /> Update ComfyUI
              </div>
              {update ? <CommandRow command={update} label="Copy update command" /> : <p className="text-[14px] text-ink2">The Desktop app updates itself. Restart it to update.</p>}
            </div>
          ) : null}
          {packs.map((p) => (
            <div key={p.pack} className="space-y-2">
              <div className="flex items-center gap-2.5 text-[15px] text-ink">
                <Mark state="warn" /> {p.manager_name ?? p.pack}
                <a href={p.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[13px] text-ink3 hover:text-ivory">
                  GitHub <ExternalLink size={11} />
                </a>
                <Hint text={`In ComfyUI Manager, search for ${p.manager_name ?? p.pack}. Or run this in the ComfyUI folder, then restart ComfyUI.`} />
              </div>
              {p.install ? <CommandRow command={p.install} label={`Copy install command for ${p.pack}`} /> : null}
            </div>
          ))}
        </section>
      ) : null}

      {repos.length ? (
        <section className="mt-8 space-y-2">
          <div className="slate text-ink3">Licenses</div>
          {repos.map((n) => (
            <div key={n.page} className="flex items-center gap-3 rounded-card border border-line px-4 py-2.5">
              <Lock size={13} className="shrink-0 text-ink3" />
              <span className="min-w-0 flex-1 break-all text-[14px] text-ink">{n.repo}</span>
              <a href={n.page} target="_blank" rel="noreferrer" className="inline-flex shrink-0 items-center gap-1 text-[13px] text-ivory/85 hover:text-ivory">
                Accept license <ExternalLink size={11} />
              </a>
            </div>
          ))}
          <p className="text-[13px] text-ink3">Pickups never sees your Hugging Face token.</p>
        </section>
      ) : null}

      {files.length ? (
        <section className="mt-8 space-y-2">
          <div className="slate text-ink3">Models</div>
          {here && gated.length ? <MiddleTruncate text={modelsDir} className="mono text-[12px] text-ink3" /> : null}
          {here ? (
            <ul className="divide-y divide-line rounded-card border border-line">
              {files.map((n) => {
                const now = running && download?.file === n.file;
                const done = doneFiles.has(n.file);
                return (
                  <li key={n.file} className={cx("px-4 py-2.5", now && "bg-raised")}>
                    <div className="flex items-start gap-2.5 text-[13px]">
                      <span className="mt-[3px] flex h-3.5 w-3.5 shrink-0 items-center justify-center">
                        <Mark state={done ? "ok" : now ? "busy" : "off"} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block break-all text-ink" title={n.file}>{n.file}</span>
                        {n.gated ? (
                          <>
                            <a href={n.url} target="_blank" rel="noreferrer" className="mt-1.5 inline-flex items-center gap-1 text-[13px] text-ivory/85 hover:text-ivory">
                              Download <ExternalLink size={11} />
                            </a>
                            {/* The models folder is shown once above the list; each row names only its subfolder. */}
                            <span className="mt-1 flex items-center gap-3">
                              <span className="mono min-w-0 break-all text-[12px] text-ink3" title={joinPath(modelsDir, n.folder, /^[A-Za-z]:\\/.test(modelsDir))}>
                                {n.folder}
                              </span>
                              <button type="button" onClick={() => revealFolder(n.folder)} className="shrink-0 text-[13px] text-ivory/85 hover:text-ivory">
                                Reveal
                              </button>
                            </span>
                          </>
                        ) : null}
                      </span>
                      <span className="mono shrink-0 text-[12px] text-ink3">{gb(n.gb)}</span>
                    </div>
                    {now && download?.total ? <ProgressLine className="mt-2" value={((download.done ?? 0) / download.total) * 100} /> : null}
                  </li>
                );
              })}
            </ul>
          ) : (
            <div className="space-y-2">
              <CommandBlock command={hfCommands(files, modelsDir, os)} label="Copy download commands" />
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[13px] text-ivory/85 hover:text-ivory">
                Create a read-only token <ExternalLink size={11} />
              </a>
            </div>
          )}
        </section>
      ) : null}

      {error ? <Notice className="mt-6" tone="warn" text={error} action={{ label: "Try again", onClick: () => void start() }} /> : null}
      {download?.error && !running ? <Notice className="mt-6" tone="warn" text={download.error} action={{ label: "Try again", onClick: () => void start() }} /> : null}

      <Primary>
        {here && plain.length && !running ? (
          <Button variant="primary" size="lg" onClick={() => void start()}>
            Download {gb(plainGb)}
          </Button>
        ) : (
          <Button variant="primary" size="lg" loading={checking || running} onClick={() => void check()}>
            Check again
          </Button>
        )}
        <Button variant="ghost" size="lg" onClick={onBack}>
          Back
        </Button>
      </Primary>
    </div>
  );
}

let progressTimer = 0;
function followProgress(): void {
  window.clearTimeout(progressTimer);
  progressTimer = window.setTimeout(() => {
    void servers
      .downloadProgress()
      .then((d: ModelsDownload) => {
        setState({ modelsDownload: d, modelsFetching: d.running });
        if (d.running) followProgress();
      })
      .catch(() => {});
  }, 1000);
}

// --- 5. done -----------------------------------------------------------------------------------------------------------

function Done({ heading, cap, onChoose }: { heading: HeadingRef; cap: Capabilities | null; onChoose: () => void }) {
  const ready = (cap?.features ?? []).filter((f) => f.ready);
  const primary = useRef<HTMLButtonElement>(null);
  useEffect(() => primary.current?.focus({ preventScroll: true }), []);
  return (
    <div className="animate-rise-in">
      <Title heading={heading}>{cap ? serverName(cap.server.url, cap.server.kind) : "Ready"}</Title>
      <ul className="mt-8 flex flex-wrap gap-x-5 gap-y-2.5">
        {ready.map((f, i) => (
          <li key={f.task} style={{ animationDelay: `${i * 40}ms` }} className="inline-flex items-center gap-2 text-[15px] text-ink animate-rise-in">
            <Mark state="ok" /> {f.label}
          </li>
        ))}
      </ul>
      <Primary>
        {ready.length ? (
          <Button
            ref={primary}
            variant="primary"
            size="lg"
            onClick={() => {
              closeOnboarding();
              navigate({ name: "home" });
            }}
          >
            Start creating
          </Button>
        ) : (
          <Button ref={primary} variant="primary" size="lg" onClick={onChoose}>
            Choose features
          </Button>
        )}
      </Primary>
    </div>
  );
}
