import { X } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { getApi } from "../actions";
import { ApiError } from "../api";
import { cx, gb, shortGpu } from "../lib/format";
import { setState, useApp } from "../store";
import type { MachineKind, MachineProbe, MachinesList, SavedMachine } from "../types";
import { Button, Hint, IconButton, StateDot } from "./ui";

const KIND: Record<MachineKind, string> = { local: "This computer", lan: "Network", runpod: "RunPod" };

const RUNPOD_HINT =
  "The pod must run ComfyUI with port 8188 exposed as HTTP. You pay RunPod by the hour while it runs. Its volume keeps your models between sessions.";

function errText(e: unknown, fallback: string) {
  return e instanceof ApiError || e instanceof Error ? e.message || fallback : fallback;
}

function probeText(p: MachineProbe | null | undefined) {
  if (!p) return "Checking…";
  if (!p.reachable) return "Not reachable";
  return [p.gpu ? shortGpu(p.gpu) : "GPU", p.vram_gb ? gb(p.vram_gb) : null].filter(Boolean).join(" · ");
}

function Row({ m, active, onUse, onRemove, using }: { m: SavedMachine; active: boolean; onUse: () => void; onRemove: () => void; using: boolean }) {
  return (
    <li className="flex items-center gap-3 border-b border-line py-3">
      <StateDot on={!!m.status?.reachable} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className={cx("text-[14px]", active ? "text-ivory" : "text-ink")}>{m.name}</span>
          <span className="slate text-ink3">{KIND[m.kind] ?? m.kind}</span>
        </div>
        <div className="mt-0.5 flex gap-2 text-[12px]">
          <span className="mono truncate text-ink3">{m.url}</span>
          <span className="slate shrink-0 text-ink2">{probeText(m.status)}</span>
        </div>
      </div>
      {active ? (
        <span className="slate shrink-0 text-ink2">In use</span>
      ) : (
        <>
          <Button variant="secondary" size="sm" loading={using} onClick={onUse}>
            Use
          </Button>
          <IconButton icon={X} label={`Remove ${m.name}`} onClick={onRemove} />
        </>
      )}
    </li>
  );
}

/** Setup → Machines: saved GPU machines (this computer, a network PC, a RunPod pod), one click to switch. */
export function MachinesSetup({ open }: { open: boolean }) {
  const [list, setList] = useState<MachinesList | null>(null);
  const [usingId, setUsingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState<"runpod" | "lan" | null>(null);
  const [pod, setPod] = useState("");
  const [port, setPort] = useState("");
  const [name, setName] = useState("");
  const [addr, setAddr] = useState("");
  const [saving, setSaving] = useState(false);
  const [found, setFound] = useState<string | null>(null);

  const load = () =>
    getApi()
      .machines()
      .then(setList)
      .catch(() => setList(null));

  const comfyUrl = useApp((s) => s.status?.comfy_url);
  useEffect(() => {
    if (!open) return;
    setError(null);
    setFound(null);
    void load();
  }, [open]);
  // Another window changed the machine in use.
  useEffect(() => {
    if (open && list && comfyUrl && !list.machines.some((m) => m.id === list.active && m.url === comfyUrl)) void load();
  }, [comfyUrl]);

  const use = async (id: string) => {
    setUsingId(id);
    setError(null);
    try {
      const r = await getApi().useMachine(id);
      if (r.status) setState({ status: r.status, statusFailed: false });
      setList((l) => (l ? { ...l, active: r.active } : l));
      void load();
    } catch (e) {
      setError(errText(e, "Couldn't switch. Try again."));
    } finally {
      setUsingId(null);
    }
  };

  const remove = async (id: string) => {
    setError(null);
    try {
      const r = await getApi().removeMachine(id);
      setList((l) => (l ? { ...r, machines: r.machines.map((m) => ({ ...m, status: l.machines.find((x) => x.id === m.id)?.status })) } : r));
    } catch (e) {
      setError(errText(e, "Couldn't remove it."));
    }
  };

  const add = async (e: FormEvent) => {
    e.preventDefault();
    if (!adding) return;
    setSaving(true);
    setError(null);
    setFound(null);
    try {
      const r = await getApi().addMachine(
        adding === "runpod" ? { kind: "runpod", pod: pod.trim(), port: port.trim() || undefined } : { kind: "lan", url: addr.trim(), name: name.trim() || undefined },
      );
      setList(r);
      setFound(r.probe?.reachable ? `Found ${probeText(r.probe)}. Click Use to switch.` : "Saved. It doesn't answer yet. Is the pod running?");
      setAdding(null);
      setPod("");
      setPort("");
      setName("");
      setAddr("");
    } catch (err) {
      setError(errText(err, "Couldn't add it."));
    } finally {
      setSaving(false);
    }
  };

  const active = list?.machines.find((m) => m.id === list.active);

  return (
    <div className="mb-6">
      {list ? (
        <ul className="border-t border-line">
          {list.machines.map((m) => (
            <Row
              key={m.id}
              m={m}
              active={m.id === list.active}
              using={usingId === m.id}
              onUse={() => void use(m.id)}
              onRemove={() => void remove(m.id)}
            />
          ))}
        </ul>
      ) : null}

      {active && active.kind !== "local" ? <p className="mt-3 text-[12px] text-ink3">Big 4K clips upload first; allow a minute.</p> : null}

      {adding ? (
        <form onSubmit={add} className="mt-4 rounded-card border border-line p-4">
          {adding === "runpod" ? (
            <>
              <div className="mb-2 flex items-center gap-1.5">
                <label htmlFor="runpod-pod" className="text-[13px] text-ink2">
                  Pod ID or address
                </label>
                <Hint text={RUNPOD_HINT} />
              </div>
              <div className="flex gap-2">
                <input
                  id="runpod-pod"
                  data-autofocus
                  className="field mono min-w-0 flex-1"
                  value={pod}
                  onChange={(e) => setPod(e.target.value)}
                  placeholder="abc123xyz"
                  spellCheck={false}
                  autoComplete="off"
                />
                <input
                  aria-label="Port"
                  className="field mono w-[88px]"
                  value={port}
                  onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))}
                  placeholder="8188"
                  inputMode="numeric"
                  autoComplete="off"
                />
              </div>
            </>
          ) : (
            <>
              <label htmlFor="lan-addr" className="mb-2 block text-[13px] text-ink2">
                ComfyUI address
              </label>
              <div className="flex gap-2 max-[640px]:flex-col">
                <input
                  id="lan-addr"
                  data-autofocus
                  className="field mono min-w-0 flex-1"
                  value={addr}
                  onChange={(e) => setAddr(e.target.value)}
                  placeholder="http://192.168.1.20:8188"
                  spellCheck={false}
                  autoComplete="off"
                />
                <input aria-label="Name" className="field w-[160px] max-[640px]:w-full" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
              </div>
            </>
          )}
          <div className="mt-3 flex justify-end gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => setAdding(null)}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" size="sm" loading={saving} disabled={adding === "runpod" ? !pod.trim() : !addr.trim()}>
              Add
            </Button>
          </div>
        </form>
      ) : (
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" onClick={() => setAdding("runpod")}>
            Add a RunPod GPU
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setAdding("lan")}>
            Add a network PC
          </Button>
        </div>
      )}

      <div className="mt-3 min-h-[18px] text-[13px]" aria-live="polite">
        {error ? (
          <p className="flex items-baseline gap-2.5 text-ink">
            <span className="slate">Error</span>
            {error}
          </p>
        ) : found ? (
          <p className="text-ink2">{found}</p>
        ) : null}
      </div>
    </div>
  );
}
