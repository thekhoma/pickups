// Setup → Server: the connected ComfyUI, the recent five to switch between, and "Change server" (the Connect flow).
import { useEffect, useState } from "react";
import { openOnboarding, refreshStatus, toast } from "../actions";
import { useApp } from "../store";
import { serverName, servers, type Servers } from "../lib/servers";
import { shortGpu } from "../lib/format";
import { Button, StateDot } from "./ui";

export function ServerSetup({ open }: { open: boolean }) {
  const status = useApp((s) => s.status);
  const [list, setList] = useState<Servers | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const [forgetting, setForgetting] = useState<string | null>(null); // asked "Forget it?", waiting for the answer
  const load = () => {
    servers.list().then(setList).catch(() => setList(null));
  };
  useEffect(() => {
    if (open) load();
  }, [open, status?.comfy_url]);

  const active = list?.active.url ?? status?.comfy_url ?? "";
  const others = (list?.recent ?? []).filter((r) => r.url !== active);
  const use = async (url: string) => {
    setSwitching(url);
    try {
      setList(await servers.use(url));
      await refreshStatus();
    } catch (e) {
      toast((e as Error).message || "Couldn't switch to that server.", "error", undefined, 7000);
    } finally {
      setSwitching(null);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-4">
        <StateDot on={!!status?.connected} />
        <div className="min-w-0 flex-1">
          <span className="block text-[15px] text-ink">
            {active ? serverName(active, list?.active.kind) : "Not connected"}
            {status?.connected && status.machine ? <span className="text-ink3"> · {shortGpu(status.machine.gpu)}</span> : null}
          </span>
          <span className="block truncate font-mono text-[12px] text-ink3">{active || "No ComfyUI yet"}</span>
        </div>
        <Button variant="secondary" size="sm" onClick={() => openOnboarding("connect")}>
          Change server
        </Button>
      </div>
      {others.length ? (
        <div>
          <p className="slate mb-2 text-ink3">Recent</p>
          <ul className="divide-y divide-line border-y border-line">
            {others.map((r) => (
              <li key={r.url} className="flex items-center gap-4 py-2.5">
                <div className="min-w-0 flex-1">
                  <span className="block text-[14px] text-ink">
                    {serverName(r.url, r.kind)}
                    {r.gpu ? <span className="text-ink3"> · {r.gpu}</span> : null}
                  </span>
                  <span className="block truncate font-mono text-[12px] text-ink3">{r.url}</span>
                </div>
                {forgetting === r.url ? (
                  <span className="flex items-center gap-1.5 animate-fade-in">
                    <span className="mr-1 text-[13px] text-ink">Forget it?</span>
                    <Button variant="ghost" size="sm" onClick={() => setForgetting(null)}>
                      Keep
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => void servers.forget(r.url).then((l) => (setList(l), setForgetting(null)))}
                    >
                      Forget
                    </Button>
                  </span>
                ) : (
                  <>
                    <Button variant="ghost" size="sm" onClick={() => setForgetting(r.url)}>
                      Forget
                    </Button>
                    <Button variant="secondary" size="sm" loading={switching === r.url} onClick={() => void use(r.url)}>
                      Use
                    </Button>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
