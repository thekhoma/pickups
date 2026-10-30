// Setup · Prompt director: connect a local LLM (LM Studio, Ollama) in two clicks, use the GPU machine's official
// enhancers, or turn it off (a choice the user confirms). No addresses or model names are built in.
import { useEffect, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { getApi, saveDirector } from "../actions";
import { cx } from "../lib/format";
import { useApp } from "../store";
import type { DetectedServer } from "../types";
import { Button, Spinner, StateDot } from "./ui";

type Choice = "openai" | "comfy" | "off";

const KIND: Record<string, string> = { lmstudio: "LM Studio", "lm-studio": "LM Studio", ollama: "Ollama" };
const kindName = (k?: string) => (k ? KIND[k.toLowerCase()] ?? k : "Local server");
const hostOf = (url: string) => url.replace(/^https?:\/\//, "").replace(/\/v1\/?$/, "").replace(/\/$/, "");
const errText = (e: unknown, fallback: string) => (e instanceof ApiError && e.message ? e.message : fallback);

/** One option in a Setup radio group (also used by Licenses → Frame editor). */
export function RadioRow({ on, onClick, title, sub, badge }: { on: boolean; onClick: () => void; title: string; sub: string; badge?: ReactNode }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={on}
      onClick={onClick}
      className={cx("flex w-full items-start gap-3 rounded-card border px-4 py-3 text-left transition-colors duration-150", on ? "border-ivory/50 bg-raised" : "border-line hover:border-line2")}
    >
      <span className={cx("mt-1 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border", on ? "border-ivory" : "border-line2")}>
        {on ? <span className="h-1.5 w-1.5 rounded-full bg-ivory" /> : null}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <span className={cx("text-[14px]", on ? "text-ivory" : "text-ink")}>{title}</span>
          {badge}
        </span>
        <span className="mt-0.5 block text-[12px] leading-snug text-ink3">{sub}</span>
      </span>
    </button>
  );
}

function ErrorLine({ text }: { text: string }) {
  return (
    <p className="flex items-baseline gap-2.5 text-[13px] text-ink">
      <span className="slate shrink-0">Error</span>
      {text}
    </p>
  );
}

export function DirectorSetup({ Section }: { Section: (p: { title: string; children: ReactNode; right?: ReactNode }) => ReactNode }) {
  const director = useApp((s) => s.director);
  const saved: Choice = director?.backend === "openai" ? "openai" : director?.backend === "off" ? "off" : "comfy";
  const [choice, setChoice] = useState<Choice>(saved);
  const [confirmOff, setConfirmOff] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedNote, setSavedNote] = useState<string | null>(null);

  // Local LLM flow
  const [detect, setDetect] = useState<{ s: "idle" | "busy" | "done"; found: DetectedServer[] }>({ s: "idle", found: [] });
  const [server, setServer] = useState<DetectedServer | null>(null);
  const [model, setModel] = useState("");
  const [vision, setVision] = useState("");
  const [manual, setManual] = useState(false);
  const [manualUrl, setManualUrl] = useState("");
  const [tested, setTested] = useState<number | null>(null);
  const [howTo, setHowTo] = useState(false);

  useEffect(() => setChoice(saved), [saved]);

  const pickServer = (sv: DetectedServer) => {
    setServer(sv);
    setModel(sv.suggested_model || sv.models[0] || "");
    setVision(sv.suggested_vision_model || "");
    setTested(null);
  };

  const find = async () => {
    setError(null);
    setDetect({ s: "busy", found: [] });
    try {
      const res = await getApi().detectPromptServers();
      setDetect({ s: "done", found: res.found });
      if (res.found[0]) pickServer(res.found[0]);
    } catch (e) {
      setDetect({ s: "done", found: [] });
      setError(errText(e, "Couldn't look for a local server."));
    }
  };

  const testAndSave = async (url: string) => {
    setBusy(true);
    setError(null);
    setTested(null);
    try {
      const r = await getApi().testPromptServer({ url, ...(model ? { model } : {}) });
      const useModel = (r.model as string | undefined) || model;
      if (!server || server.url !== (r.url ?? url)) {
        pickServer({ url: r.url ?? url, kind: r.kind ?? "", models: r.models ?? [], suggested_model: useModel, suggested_vision_model: r.suggested_vision_model ?? null });
      }
      setTested(r.seconds);
      const err = await saveDirector({ backend: "openai", url: r.url ?? url, model: useModel, vision_model: vision });
      if (err) setError(err);
      else setSavedNote(`Connected · answered in ${r.seconds.toFixed(1)} s`);
    } catch (e) {
      setError(errText(e, "Couldn't reach that server. Is LM Studio's local server running (Developer → Start Server)?"));
    } finally {
      setBusy(false);
    }
  };

  const useGpu = async () => {
    setBusy(true);
    setError(null);
    const err = await saveDirector({ backend: "comfy" });
    setBusy(false);
    if (err) setError(err);
    else {
      setChoice("comfy");
      setSavedNote("Using the GPU machine.");
    }
  };

  const turnOff = async () => {
    setBusy(true);
    setError(null);
    const err = await saveDirector({ backend: "off", off_acknowledged: true });
    setBusy(false);
    setConfirmOff(false);
    if (err) setError(err);
  };

  const isOff = director?.backend === "off";

  return (
    <Section
      title="Prompt director"
      right={isOff ? <span className="slate text-ink3">Off</span> : saved === "openai" && director?.url ? <span className="slate flex items-center gap-1.5 text-ink2"><StateDot on /> Local LLM</span> : null}
    >

      {isOff && choice === "off" ? (
        <p className="text-[13px] text-ink2">
          The Enhance buttons are hidden.{" "}
          <button type="button" onClick={() => setChoice("openai")} className="text-ivory/80 underline-offset-4 hover:text-ivory hover:underline">
            Turn on
          </button>
        </p>
      ) : (
        <div className="space-y-2" role="radiogroup" aria-label="Where the prompt director runs">
          <RadioRow
            on={choice === "openai"}
            onClick={() => {
              setChoice("openai");
              setConfirmOff(false);
            }}
            title="Local LLM: LM Studio or Ollama"
            sub="Fast and private. Uses a model you load."
            badge={detect.found.length ? <span className="slate text-ink3">Recommended</span> : null}
          />
          <RadioRow
            on={choice === "comfy"}
            onClick={() => {
              setChoice("comfy");
              setConfirmOff(false);
              if (saved !== "comfy") void useGpu();
            }}
            title="On the GPU machine"
            sub="No setup. Slower, and shares the GPU with renders."
          />
          <RadioRow on={choice === "off"} onClick={() => { setChoice("off"); setConfirmOff(true); }} title="Off" sub="Prompts go to the models as written." />
        </div>
      )}

      {/* Local LLM */}
      {choice === "openai" && !isOff ? (
        <div className="mt-4 space-y-3 rounded-card border border-line p-4">
          {saved === "openai" && director?.url && !server ? (
            <p className="mono text-[12px] text-ink2">
              {hostOf(director.url)} · {director.model || "no model chosen"}
              {director.vision_model ? ` · vision: ${director.vision_model}` : ""}
            </p>
          ) : null}
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" size="sm" loading={detect.s === "busy"} onClick={() => void find()}>
              Find my LM Studio
            </Button>
            <button type="button" onClick={() => setManual((m) => !m)} className="text-[12px] text-ink3 underline-offset-4 hover:text-ivory hover:underline">
              Enter address manually
            </button>
          </div>

          {manual ? (
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (manualUrl.trim()) void testAndSave(manualUrl.trim());
              }}
            >
              <input value={manualUrl} onChange={(e) => setManualUrl(e.target.value)} placeholder="http://127.0.0.1:1234/v1" aria-label="Server address" className="field mono h-9" spellCheck={false} />
              <Button type="submit" variant="secondary" size="sm" className="h-9" loading={busy && !server} disabled={!manualUrl.trim()}>
                Test & save
              </Button>
            </form>
          ) : null}

          {detect.s === "done" && detect.found.length > 1 ? (
            <div className="flex flex-wrap gap-1.5">
              {detect.found.map((sv) => (
                <button
                  key={sv.url}
                  type="button"
                  onClick={() => pickServer(sv)}
                  className={cx("mono h-7 rounded-full border px-3 text-[11px]", server?.url === sv.url ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:text-ink")}
                >
                  {kindName(sv.kind)} · {hostOf(sv.url)}
                </button>
              ))}
            </div>
          ) : null}

          {server ? (
            <div className="space-y-3">
              <p className="mono text-[12px] text-ink">
                {kindName(server.kind)} · {hostOf(server.url)} · {server.models.length} {server.models.length === 1 ? "model" : "models"}
              </p>
              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="slate mb-1 block text-ink3">Text model</span>
                  <select value={model} onChange={(e) => { setModel(e.target.value); setTested(null); }} className="field mono h-9 text-[12px]">
                    {server.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <span className="slate mb-1 block text-ink3">Vision model · optional</span>
                  <select value={vision} onChange={(e) => setVision(e.target.value)} className="field mono h-9 text-[12px]">
                    <option value="">None</option>
                    {server.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="flex items-center gap-3">
                <Button variant="primary" size="sm" loading={busy} disabled={!model} onClick={() => void testAndSave(server.url)}>
                  Test & save
                </Button>
                {tested != null && savedNote ? (
                  <span className="slate flex items-center gap-1.5 text-ink2">
                    <StateDot on /> {savedNote}
                  </span>
                ) : busy ? (
                  <span className="slate flex items-center gap-1.5 text-ink3">
                    <Spinner size={10} /> Testing
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}

          {detect.s === "done" && !detect.found.length ? (
            <div className="space-y-3 rounded-[4px] border border-line bg-bg/60 p-3.5">
              <p className="text-[13px] leading-relaxed text-ink">
                No local LLM server found.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="secondary" size="sm" loading={busy} onClick={() => void useGpu()}>
                  Use the GPU machine
                </Button>
                <Button variant="ghost" size="sm" onClick={() => { setChoice("off"); setConfirmOff(true); }}>
                  Turn it off
                </Button>
                <button type="button" onClick={() => setHowTo((h) => !h)} aria-expanded={howTo} className="ml-auto text-[12px] text-ink3 underline-offset-4 hover:text-ivory hover:underline">
                  How to set up LM Studio
                </button>
              </div>
              {howTo ? (
                <ol className="list-decimal space-y-1 pl-5 text-[12px] leading-relaxed text-ink2">
                  <li>Install LM Studio from lmstudio.ai.</li>
                  <li>Download a small instruct model, such as a Qwen or Gemma.</li>
                  <li>Open Developer and press Start Server.</li>
                  <li>
                    Come back here and press{" "}
                    <button type="button" onClick={() => void find()} className="text-ivory/80 underline-offset-4 hover:text-ivory hover:underline">
                      Find again
                    </button>
                    .
                  </li>
                </ol>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Off: a choice the user confirms */}
      {choice === "off" && !isOff && confirmOff ? (
        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-card border border-line2 p-4 animate-fade-in">
          <p className="min-w-[240px] flex-1 text-[13px] text-ink">
            Turn off the prompt director? Enhance buttons will be hidden.
          </p>
          <Button variant="secondary" size="sm" loading={busy} onClick={() => void turnOff()}>
            Turn off
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setConfirmOff(false);
              setChoice(saved);
            }}
          >
            Cancel
          </Button>
        </div>
      ) : null}

      {/* One rule: a choice is saved when it's made. A local LLM needs a server first, so until Test & save the old
          choice still runs; say so rather than let closing Setup look like a save. */}
      {choice === "openai" && saved !== "openai" && !busy ? (
        <p className="mt-3 text-[13px] text-ink2">
          Not saved yet. Until you Test &amp; save, {saved === "off" ? "the prompt director stays off" : "prompts are directed on the GPU machine"}.
        </p>
      ) : null}
      {choice === "comfy" && savedNote && !error ? <p className="slate mt-3 flex items-center gap-1.5 text-ink2"><StateDot on /> {savedNote}</p> : null}
      {error ? <div className="mt-3"><ErrorLine text={error} /></div> : null}
    </Section>
  );
}
