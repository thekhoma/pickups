// A workflow's default models, small: "LTX-2.5 · Qwen-Image 2.1 +3". On a gallery tile it's a dark pill over the
// picture whose hint lists every model with its role; in the task list it's a quiet mono line (a title tooltip, since
// the row is already a button). The catalog comes from the backend (models_catalog.py via /api/models/defaults).
import { cx } from "../lib/format";
import { useApp } from "../store";
import type { JobTask, ModelCatalog, ModelSlot } from "../types";
import { Hint } from "./ui";

/** The default models of a home workflow ("character", "weather", ...), or of the first workflow that runs `task`. */
export function useWorkflowModels(workflow?: string, task?: JobTask, fallback?: ModelSlot[]): ModelSlot[] {
  const cat = useApp((s) => s.defaultModels);
  return modelsFor(cat, workflow, task) ?? fallback ?? [];
}

export function modelsFor(cat: ModelCatalog | null, workflow?: string, task?: JobTask): ModelSlot[] | null {
  if (!cat) return null;
  if (workflow && cat[workflow]) return cat[workflow].models;
  if (task) return Object.values(cat).find((w) => w.task === task)?.models ?? null;
  return null;
}

/** Chip labels, each once ("LTX-2.5" twice in a workflow is one chip). */
function labels(models: ModelSlot[]): string[] {
  return [...new Set(models.map((m) => m.short || m.name))];
}

/** "LTX-2.5 + Union Control IC-LoRA: video. Qwen-Image 2.1: frame paint. ..." for screen readers and titles. */
export function modelsSentence(models: ModelSlot[]): string {
  return models.map((m) => `${m.name}: ${m.role}${m.when ? ` (${m.when})` : ""}`).join(". ");
}

function ModelList({ models }: { models: ModelSlot[] }) {
  return (
    <span className="block">
      <span className="slate block text-[10px] text-ink3">Default models</span>
      <span className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        {models.map((m) => (
          <span key={`${m.role}-${m.name}`} className="contents">
            <span className="text-ink3">{m.role}</span>
            <span className="text-ink">
              {m.name}
              {m.when ? <span className="text-ink3"> · {m.when}</span> : null}
            </span>
          </span>
        ))}
      </span>
    </span>
  );
}

/** One line of chips: the first `max` labels, then "+N". */
function ChipText({ models, max }: { models: ModelSlot[]; max: number }) {
  const all = labels(models);
  const shown = all.slice(0, max);
  const more = all.length - shown.length;
  return (
    <>
      <span className="min-w-0 truncate">{shown.join(" · ")}</span>
      {more > 0 ? <span className="shrink-0 opacity-70">+{more}</span> : null}
    </>
  );
}

/** Over a gallery tile's picture: a small dark pill; hover, focus or tap lists every model with its role. */
export function ModelPill({ models, max = 2, className }: { models: ModelSlot[]; max?: number; className?: string }) {
  if (!models.length) return null;
  return (
    <Hint
      text={<ModelList models={models} />}
      label={`Default models: ${labels(models).join(", ")}`}
      className={cx(
        "inline-flex h-5 min-w-0 max-w-full items-center gap-1.5 rounded-full bg-black/60 px-2 font-mono text-[10px] leading-none tracking-[0.02em] text-ivory/90 backdrop-blur-[2px] transition-colors duration-150 hover:bg-black/75 hover:text-ivory focus-visible:bg-black/75",
        className,
      )}
    >
      <ChipText models={models} max={max} />
    </Hint>
  );
}

/** Inside a row that's already a button: a quiet mono line with a title tooltip. */
export function ModelLine({ models, max = 2, className }: { models: ModelSlot[]; max?: number; className?: string }) {
  if (!models.length) return null;
  return (
    <span className={cx("flex min-w-0 items-center gap-1.5 font-mono text-[11px] leading-4 text-ink3", className)} title={`Default models\n${models.map((m) => `${m.role}: ${m.name}${m.when ? ` (${m.when})` : ""}`).join("\n")}`}>
      <ChipText models={models} max={max} />
    </span>
  );
}
