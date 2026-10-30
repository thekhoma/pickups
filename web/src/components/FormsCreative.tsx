// Change weather (environment) and Render previz: the two forms that take prompts, one per model.
import { TrackCheck } from "./Forms";
import { ChevronDown, Info, MousePointerClick, Plus, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { clearSelection, endSelection, getApi, loadJobs, openPreviz, startSelection, updateShot } from "../actions";
import { useDraft } from "../lib/drafts";
import { lastInlineAt, newestInline } from "../lib/inlineJobs";
import { cx, timecode } from "../lib/format";
import { madeIn } from "../lib/runtime";
import { nextSeed } from "../lib/seed";
import { usePlayer } from "../lib/player";
import { HIDDEN_PRESETS } from "../lib/recipes";
import { getState, useApp } from "../store";
import { cameraMismatch, facingMismatch, loneCar, motionPromptFor, motionSpecOf } from "../previz/motionPrompt";
import { frame0Again, frame0IsThin } from "../previz/remake";
import type { CastMember, EnhanceRequest, EnvironmentParams, Grain, Point, PrevizFollow, PrevizParams, Ref as RefPick, Shot, WeatherChange } from "../types";
import { CharacterViewer } from "./CharacterViewer";
import { useDescribe, type DescribeFor } from "./Describe";
import { useEnhance } from "./Enhance";
import { MentionField, PromptReceipt, useCast, type ReceiptModel } from "./Mentions";
import { bool, Chip, Field, Fold, FormShell, keptFrom, keptUrl, PrimaryAction, RefPicture, str, useLastTake, usePrefill, useShotWindow, useTask, useUploadSlot, type FormProps } from "./FormKit";
import { Cancelled, RunStrip } from "./RunStatus";
import { Button, DropSlot, Hint, Notice, SeedLock, Segmented, Skeleton, Slate, Spinner, SwitchRow, useDismiss } from "./ui";

const GRAINS: Grain[] = ["off", "fine", "35mm"];
const GRAIN_OPTIONS: Array<{ value: Grain; label: string }> = [
  { value: "off", label: "Off" },
  { value: "fine", label: "Fine" },
  { value: "35mm", label: "35mm" },
];

function preGrain(v: unknown): Grain {
  return GRAINS.includes(v as Grain) ? (v as Grain) : "fine";
}

/** "Film grain: Off · Fine · 35mm" (default Fine): generated video has none, so skin reads as plastic. */
function GrainControl({ value, onChange }: { value: Grain; onChange: (g: Grain) => void }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-1.5 text-[13px] text-ink2">
        Film grain
        <Hint text="Generated video has no grain, so skin can look plastic. Fine puts it back." />
      </div>
      <Segmented<Grain> value={value} onChange={onChange} options={GRAIN_OPTIONS} label="Film grain" size="sm" />
    </div>
  );
}

// --- prompt tips -------------------------------------------------------------------------------------------------

type TipsId = "previz-look" | "previz-motion" | "env-look" | "env-video";

// Static for now; condensed from docs/prompting-guide.md.
const TIPS: Record<TipsId, { title: string; model: string; bullets: string[] }> = {
  "previz-look": {
    title: "Look",
    model: "Qwen-Image 2.1 · edit",
    bullets: [
      "Refer to references as image 2, image 3, or @-mention your cast; Pickups writes them the way each model expects.",
      "Say what each mannequin or grey block becomes, and where it is in the frame.",
      "Describe the light and the place in plain, physical words, not “epic, 8k”.",
      "End with one short clause about what must not change: the framing and every position.",
    ],
  },
  "previz-motion": {
    title: "Motion and camera",
    model: "LTX-2.5 · video",
    bullets: [
      "One paragraph in the present tense, in this order: subject, action, camera move, light, film look.",
      "@-mention your cast: the video model gets each one's description in your words.",
      "Describe the finished shot. Leave out “3D”, “clay”, “previz”, “Blender”, “grey” and “render”.",
      "Match the look frame: if it's dusk there, don't write noon.",
      "Write what should be there. Negatives (“no blur”) do nothing at this setting.",
    ],
  },
  "env-look": {
    title: "First-frame instruction",
    model: "Qwen-Image 2.1 · edit",
    bullets: [
      "Start with the change: “Change the weather in the image to a snowy winter day: …”.",
      "Say where it shows: snow on roofs, ledges and walls; wet, mirror-like ground.",
      "End with one short clause about what stays: framing, faces, clothes.",
      "Want people in winter coats? Ask for it here; otherwise their clothes stay as they are.",
    ],
  },
  "env-video": {
    title: "Video prompt",
    model: "LTX-2.5 · Union Control",
    bullets: [
      "Describe the look, not the motion: the camera and the people already move as in your footage.",
      "@-mention your cast or places; Pickups writes them the way each model expects.",
      "Use the same look words as the first-frame instruction.",
      "Do describe weather that moves: falling snow, drifting fog, rain rings in the puddles.",
      "Write what should be there. Negatives do nothing at this setting.",
    ],
  },
};

function PromptTips({ id }: { id: TipsId }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  useDismiss(open, () => setOpen(false), ref);
  const t = TIPS[id];
  return (
    <span ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="inline-flex shrink-0 items-center whitespace-nowrap gap-1 rounded-full text-[12px] text-ink3 underline-offset-4 transition-colors hover:text-ivory hover:underline"
      >
        <Info size={12} strokeWidth={1.5} aria-hidden="true" />
        Prompt tips
      </button>
      {open ? (
        <span
          role="dialog"
          aria-label={`${t.title} prompt tips`}
          className="absolute right-0 top-full z-40 mt-2 block w-[300px] animate-rise-in rounded-card border border-line2 bg-raised p-4 text-left"
        >
          <span className="slate block text-ink3">{t.model}</span>
          <ul className="mt-2.5 space-y-2">
            {t.bullets.map((b) => (
              <li key={b} className="flex gap-2.5 text-[13px] leading-snug text-ink">
                <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ivory/60" aria-hidden="true" />
                {b}
              </li>
            ))}
          </ul>
        </span>
      ) : null}
    </span>
  );
}

type PromptToolsProps = {
  shot: Shot;
  value: string;
  onChange: (v: string) => void;
  tips?: TipsId;
  hint?: ReactNode;
  /** Which models' rows to show in the prompt receipt. */
  receipt?: ReceiptModel[];
  uploads?: number;
  /** The prompt director's target and context for this field. */
  enhance?: Omit<EnhanceRequest, "text" | "project_id">;
  /** "Describe it": what a local model writes into this field after looking at the shot. */
  describe?: Omit<DescribeFor, "shotId">;
};

/** A prompt's tools for pros: its hint, Prompt tips, Describe it, Enhance and what each model gets. */
function PromptTools({ shot, value, onChange, tips, hint, receipt, uploads, enhance, describe }: PromptToolsProps) {
  const projectId = shot.project_id || "unsorted";
  const e = useEnhance({ request: enhance ? { ...enhance, project_id: projectId } : null, text: value, onUse: onChange });
  const d = useDescribe({ what: describe ? { ...describe, shotId: shot.id } : null, value, onChange });
  const buttons = tips || d.button || e.button;
  if (!hint && !buttons && !receipt) return null;
  return (
    <div className="space-y-2">
      {hint ? <p className="text-[13px] leading-snug text-ink3">{hint}</p> : null}
      {buttons ? (
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          {tips ? <PromptTips id={tips} /> : null}
          {d.button}
          {e.button}
        </span>
      ) : null}
      {d.panel}
      {e.panel}
      {receipt ? <PromptReceipt projectId={projectId} text={value} models={receipt} uploads={uploads} /> : null}
    </div>
  );
}

/** A labelled prompt: the label and the textarea. Its tools (PromptTools) wait under one closed "Advanced" below it;
 * `tools="inline"` shows them as they are (a field already inside an Advanced section), `tools="none"` leaves them to
 * the form, which shows them in its own Advanced (Previz to final's Look). */
function PromptField({
  label,
  placeholder,
  rows = 3,
  required,
  tools = "fold",
  ...rest
}: PromptToolsProps & {
  tools?: "fold" | "inline" | "none";
  label: string;
  placeholder?: string;
  rows?: number;
  /** Marks the field as needed (and says so) when nothing else tells the model what to make. */
  required?: boolean;
}) {
  const { shot, value, onChange } = rest;
  const projectId = shot.project_id || "unsorted";
  const cast = useCast(projectId);
  const has = !!(rest.hint || rest.tips || rest.receipt || rest.enhance || rest.describe);
  return (
    <div className="space-y-1.5">
      <span className="flex items-center gap-1.5">
        <label className="whitespace-nowrap text-[13px] text-ink2">{label}</label>
        {required ? <span className="slate text-ink3">Required</span> : null}
      </span>
      <MentionField label={label} value={value} onChange={onChange} placeholder={placeholder} rows={rows} cast={cast} projectId={projectId} />
      {has && tools === "inline" ? <PromptTools {...rest} /> : null}
      {has && tools === "fold" ? (
        <Fold label="Advanced">
          <div className="pt-2">
            <PromptTools {...rest} />
          </div>
        </Fold>
      ) : null}
    </div>
  );
}

function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-line pt-4">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-[14px] text-ink2 transition-colors hover:text-ivory"
      >
        {title}
        <ChevronDown size={15} strokeWidth={1.5} className={cx("transition-transform duration-150", open && "rotate-180")} />
      </button>
      {open ? <div className="mt-4 space-y-5 animate-fade-in">{children}</div> : null}
    </div>
  );
}

// --- Change weather -----------------------------------------------------------------------------------------------

/** Mirrors the backend presets (tasks/environment.py): the first-frame instruction and the video sentence. */
const WEATHER: Array<{ id: WeatherChange; label: string; look: string; video: string }> = [
  {
    id: "snow",
    label: "Snow",
    look: "Make it a winter day in heavy snowfall: thick fresh snow covers the ground, paths, roofs, ledges, walls and plants, large snowflakes fall through the air, the sky is a flat overcast white-grey and the light is soft and cold.",
    video: "Heavy snow falls steadily over the scene; thick fresh snow covers the ground, roofs and every ledge, and large flakes drift down through soft, cold, overcast light.",
  },
  {
    id: "rain",
    label: "Rain",
    look: "Make it a dark, rainy day: heavy rain is falling, the ground and every surface are soaking wet and reflective with puddles, the sky is low and dark grey, and the light is dim and cool.",
    video: "Heavy rain pours down, streaking through the air and splashing in puddles on the wet, reflective ground under a low, dark grey sky; every surface glistens in the dim, cool light.",
  },
  {
    id: "fog",
    label: "Fog",
    look: "Fill the scene with thick, soft morning fog: the background fades into pale grey mist, distant objects are barely visible, and the light is diffuse and cool.",
    video: "Thick morning fog drifts slowly through the scene, fading the distance into pale grey mist under diffuse, cool light.",
  },
  {
    id: "night",
    label: "Night",
    look: "Turn this into the same place at night: a deep blue night sky, warm street lamps, lit windows and lamps glowing, soft moonlight on surfaces, dark shadows, and every person lit by the nearest lights.",
    video: "At night, warm street lamps and lit windows glow against a deep blue sky, and pools of lamplight fall across the ground and the people moving through the soft shadows.",
  },
  {
    id: "golden_hour",
    label: "Golden hour",
    look: "Change the light to golden hour: a low, warm sun near the horizon, long soft shadows, a glowing amber and pink sky, and warm rim light on people and edges.",
    video: "In warm golden-hour light, a low sun near the horizon casts long soft shadows and rims every edge with amber light under a glowing sky.",
  },
  {
    id: "autumn",
    label: "Autumn",
    look: "Make it autumn: the trees and plants turn orange, red and gold, fallen leaves cover the ground, and the light is soft and warm.",
    video: "In autumn, trees in orange, red and gold leaves line the scene, fallen leaves scatter across the ground, and soft, warm light filters through.",
  },
];

type Structure = "" | "depth" | "canny";

/** Clicks on what to keep, on one frame of the shot. `from`: a take's pick ("the last take"), sent again as it was. */
type KeepPick = { points: Point[]; frame: number; from?: string };

function keptPick(pre: Record<string, unknown> | null): KeepPick | null {
  const points = Array.isArray(pre?.keep_points) ? (pre.keep_points as Point[]) : [];
  return points.length ? { points, frame: typeof pre?.keep_frame === "number" ? pre.keep_frame : 0, from: keptFrom(pre) } : null;
}

const clicksText = (n: number) => `${n} ${n === 1 ? "click" : "clicks"}`;

/** "Keep exactly as shot": one thing that comes back as filmed, in the new light. Its name, or clicks on it in the
 * picture while "Click it" is on (the picture is a player again after "Done"). */
function KeepRow({ shot, text, onText, pick, onPick }: { shot: Shot; text: string; onText: (t: string) => void; pick: KeepPick | null; onPick: (p: KeepPick | null) => void }) {
  const sel = useApp((s) => (s.selection && s.selection.shotId === shot.id ? s.selection : null));
  const n = sel?.points.length ?? 0;
  useEffect(() => () => endSelection(), []);
  const done = () => {
    if (sel && n && !sel.error) onPick({ points: sel.points, frame: sel.frame });
    endSelection();
  };
  const clear = () => {
    onPick(null);
    clearSelection();
    endSelection();
  };
  const status = sel?.loading
    ? "Finding it…"
    : sel?.error
      ? sel.error
      : n
        ? `Selected · ${clicksText(n)}`
        : sel
          ? "Click it in the picture, then Done."
          : pick
            ? pick.from
              ? `Same ${pick.points.length === 1 ? "click" : "clicks"} as ${pick.from}`
              : `Selected · ${clicksText(pick.points.length)}`
            : null;
  return (
    <section>
      <div className="mb-1.5 flex items-center gap-1.5">
        <label htmlFor="keep-as-shot" className="text-[13px] text-ink2">
          Keep as filmed
        </label>
        <Hint text="Optional. Something that must not change, like the couple or the car: type its name, or press Pick in picture and click it. The new weather goes around it; only its light changes. Check tracking plays the shot with it outlined, as the render will follow it." />
      </div>
      <div className="flex gap-2">
        <input id="keep-as-shot" className="field min-w-0 flex-1" value={text} onChange={(e) => onText(e.target.value)} placeholder="e.g. the couple" />
        <Button variant="secondary" icon={sel ? undefined : MousePointerClick} aria-pressed={!!sel} onClick={() => (sel ? done() : startSelection(shot.id))} className="h-10 shrink-0">
          {sel ? "Done" : "Pick in picture"}
        </Button>
      </div>
      <p className="mt-1.5 flex min-h-5 items-center gap-2 text-[12px] leading-snug text-ink3">
        <span className="min-w-0 flex-1">{status}</span>
        {sel || pick ? (
          <button type="button" onClick={clear} className="shrink-0 rounded-full px-2 text-ink2 transition-colors hover:text-ivory">
            Clear
          </button>
        ) : null}
      </p>
      {sel && n ? <TrackCheck shot={shot} versionId="orig" /> : null}
    </section>
  );
}

export function EnvironmentForm({ shot, ui }: FormProps) {
  const t = useTask("environment", shot, ui);
  const pre = usePrefill(shot.id, "environment");
  const [change, setChange] = useState<WeatherChange | "custom">(
    pre ? (WEATHER.some((w) => w.id === pre.change) ? (pre.change as WeatherChange) : str(pre.custom) ? "custom" : "snow") : "snow",
  );
  const [custom, setCustom] = useState(str(pre?.custom));
  // Only ever this shot's words: a take's, else the sentence a made shot was made from. Never another shot's.
  const [describe, setDescribe] = useState(pre ? str(pre.describe) : str(shot.made?.prompt));
  const [look, setLook] = useState(str(pre?.look_prompt));
  const [prompt, setPrompt] = useState(str(pre?.prompt));
  const [control, setControl] = useState<Structure>((str(pre?.control) as Structure) || "");
  const [seed, setSeed] = useState(typeof pre?.seed === "number" ? String(pre.seed) : "");
  const [grain, setGrain] = useState<Grain>(preGrain(pre?.grain));
  const [keep, setKeep] = useState(str(pre?.keep_text));
  const [keepPick, setKeepPick] = useState<KeepPick | null>(() => keptPick(pre));
  const keepSel = useApp((s) => (s.selection && s.selection.shotId === shot.id ? s.selection : null));
  // A first frame kept by "Try again with changes" (or picked in Images) is a project image or an upload.
  const [firstFrame, setFirstFrame] = useState<{ ref: RefPick; url: string } | null>(() => {
    const ref: RefPick | null = pre?.first_frame_ref
      ? (pre.first_frame_ref as RefPick)
      : str(pre?.first_frame_upload_id)
        ? { kind: "upload", id: str(pre?.first_frame_upload_id) }
        : null;
    return ref ? { ref, url: str(pre?._first_frame_url) || keptUrl(ref, shot.project_id) } : null;
  });
  const preset = WEATHER.find((w) => w.id === change) ?? null;
  const isCustom = change === "custom";
  const customText = custom.trim();
  const valid = !isCustom || !!customText;
  const keepAt = keepSel && keepSel.points.length && !keepSel.error ? keepSel.frame : keepPick ? keepPick.frame : null;
  const bracket = useShotWindow(shot, typeof pre?.start_frame === "number" ? Math.max(0, Math.round(pre.start_frame)) : null, keepAt, false, false, true, "environment");

  const submit = () => {
    const params: EnvironmentParams = isCustom ? { custom: customText } : { change: change as WeatherChange };
    if (describe.trim()) params.describe = describe.trim();
    if (look.trim()) params.look_prompt = look.trim();
    if (prompt.trim()) params.prompt = prompt.trim();
    params.control = control;
    const n = parseInt(seed, 10);
    if (Number.isFinite(n)) params.seed = n;
    params.grain = grain;
    if (firstFrame?.ref.kind === "upload") params.first_frame_upload_id = firstFrame.ref.id;
    else if (firstFrame) params.first_frame_ref = firstFrame.ref;
    if (keep.trim()) params.keep_text = keep.trim();
    const clicks = keepSel?.points.length && !keepSel.error ? { points: keepSel.points, frame: keepSel.frame } : keepPick;
    if (clicks) {
      params.keep_points = clicks.points;
      params.keep_frame = clicks.frame;
    }
    if (bracket.show && bracket.win) {
      params.start_frame = bracket.win.start;
      if (bracket.win.sized) params.frames = bracket.win.frames;
    }
    void t.submit(params);
  };

  return (
    <FormShell
      task="environment"
      shot={shot}
      ui={ui}
      lengthNote={!bracket.show}
      footer={
        <>
          <PrimaryAction hint={t.etaLine} disabled={!valid || !t.ready} loading={t.submitting} onClick={submit}>
            Change the weather
          </PrimaryAction>
        </>
      }
    >
      <section className="space-y-3">
        <h3 className="text-[16px] font-medium text-ivory">Change</h3>
        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Change">
          {WEATHER.filter((w) => !HIDDEN_PRESETS.has(w.id) || change === w.id).map((w) => (
            <Chip key={w.id} active={change === w.id} onClick={() => setChange(w.id)}>
              {w.label}
            </Chip>
          ))}
          <Chip active={isCustom} onClick={() => setChange("custom")}>
            Custom
          </Chip>
        </div>
        {isCustom ? (
          <input
            className="field animate-fade-in"
            value={custom}
            autoFocus
            onChange={(e) => setCustom(e.target.value)}
            placeholder="a sandstorm at dusk"
            aria-label="Describe the change"
          />
        ) : null}
      </section>

      <KeepRow shot={shot} text={keep} onText={setKeep} pick={keepPick} onPick={setKeepPick} />

      {firstFrame ? (
        <div className="flex items-center gap-3 rounded-card border border-line p-2 pr-3">
          <RefPicture url={firstFrame.url} className="h-14 w-24 shrink-0 rounded-[3px]" />
          <span className="flex-1">
            <span className="block text-[14px] text-ink">{firstFrame.ref.kind === "upload" ? `First frame from ${keptFrom(pre)}` : "First frame from Images"}</span>

          </span>
          <Button variant="ghost" size="sm" onClick={() => setFirstFrame(null)}>
            Remove
          </Button>
        </div>
      ) : null}

      <Disclosure title="Advanced">
        <PromptField
          shot={shot}
          tools="inline"
          label="What's in the shot"
          value={describe}
          onChange={setDescribe}
          rows={3}
          placeholder="Who's in it and what they do"
          receipt={["ltx"]}
          enhance={{ target: "ltx", shot_id: shot.id }}
          describe={{ field: "scene", ...(t.baseId !== "orig" ? { versionId: t.baseId } : {}) }}
        />
        <PromptField
          shot={shot}
          tools="inline"
          enhance={{ target: "qwen-edit", shot_id: shot.id, ...(t.baseId !== "orig" ? { version_id: t.baseId } : {}), frame: 0 }}
          label="First-frame instruction"
          tips="env-look"
          value={look}
          onChange={setLook}
          rows={4}
          placeholder={preset?.look ?? (customText ? `${customText.replace(/\.$/, "")}.` : "Describe the change for the first frame.")}
        />
        <PromptField
          shot={shot}
          tools="inline"
          enhance={{ target: "ltx", shot_id: shot.id }}
          label="Video prompt"
          tips="env-video"
          value={prompt}
          onChange={setPrompt}
          rows={3}
          placeholder={[describe.trim(), preset?.video ?? customText].filter(Boolean).join(" ") || "Describe the look of the shot."}
        />
        <div className="grid grid-cols-[1fr_96px] gap-3">
          <Field label="Structure">
            <select className="field" value={control} onChange={(e) => setControl(e.target.value as Structure)}>
              <option value="">Auto</option>
              <option value="depth">Depth (surfaces change)</option>
              <option value="canny">Edges (light only)</option>
            </select>
          </Field>
          <Field label="Seed">
            <input
              className="field mono"
              inputMode="numeric"
              value={seed}
              onChange={(e) => setSeed(e.target.value.replace(/[^\d]/g, ""))}
              placeholder="42"
            />
          </Field>
        </div>
        <GrainControl value={grain} onChange={setGrain} />
      </Disclosure>
    </FormShell>
  );
}

// --- Render previz ------------------------------------------------------------------------------------------------

const MAX_REFS = 4;

interface Ref {
  key: number;
  id?: string;
  url?: string;
  name: string;
  busy: boolean;
  error?: string;
  /** Uploaded files are "upload"; picks from the project are "image" or "cast" (sent as look_refs, after uploads). */
  kind?: "upload" | "image" | "cast";
}

/** Up to four references; they are images 2–5 (image 1 is the playblast frame). */
function References({ refs, onAdd, onRemove, cast = [], onCast, sent }: {
  refs: Ref[]; onAdd: (files: File[]) => void; onRemove: (key: number) => void; cast?: CastMember[]; onCast?: (c: CastMember) => void;
  /** How many went into the start frame as pictures (it takes two); the rest are dimmed: named in its words only. */
  sent?: number | null;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const room = MAX_REFS - refs.length;
  const take = (list: FileList | null | undefined) => {
    const files = Array.from(list ?? []).filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|tiff?|exr)$/i.test(f.name));
    if (files.length) onAdd(files.slice(0, room));
  };
  return (
    <div>
      <input ref={input} type="file" accept="image/*,.exr,.tif,.tiff" multiple className="hidden" onChange={(e) => { take(e.target.files); e.target.value = ""; }} />
      <div className="grid grid-cols-2 gap-2">
        {refs.map((r, i) => (
          <div
            key={r.key}
            data-sent={sent != null ? String(i < sent) : undefined}
            className={cx("relative aspect-video overflow-hidden rounded-card border border-line bg-black transition-opacity", sent != null && i >= sent && "opacity-40")}
          >
            {r.url || (r.id && !r.busy) ? (
              <RefPicture url={r.url} alt={r.name} className="h-full w-full" />
            ) : (
              <div className="skeleton absolute inset-0 rounded-none" />
            )}
            <Slate at="tl">{`Image ${i + 2}`}</Slate>
            {r.busy ? (
              <span className="absolute inset-0 flex items-center justify-center text-ivory/80">
                <Spinner size={14} />
              </span>
            ) : null}
            {r.error ? <span className="slate absolute inset-x-2 bottom-2 rounded-[2px] bg-black/80 px-1.5 py-1 normal-case tracking-normal">{r.error}</span> : null}
            <button
              type="button"
              onClick={() => onRemove(r.key)}
              aria-label={`Remove image ${i + 2}`}
              className="absolute right-1.5 top-1.5 rounded-full bg-black/70 p-1 text-ivory/70 transition-colors hover:text-ivory"
            >
              <X size={12} />
            </button>
          </div>
        ))}
        {room > 0 ? (
          <button
            type="button"
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setOver(false);
              take(e.dataTransfer.files);
            }}
            className={cx(
              "flex aspect-video flex-col items-center justify-center gap-1.5 rounded-card border transition-colors duration-150",
              over ? "border-ivory/70 bg-raised text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
            )}
          >
            <Plus size={16} strokeWidth={1.5} />
            <span className="text-[13px]">{refs.length ? "Add another reference" : "Add references"}</span>
            <span className="text-[12px] text-ink3">A character, a place or a product</span>
            {refs.length ? <span className="slate text-ink3">{`Image ${refs.length + 2}${room > 1 ? ` – ${refs.length + 1 + room}` : ""}`}</span> : null}
          </button>
        ) : null}
      </div>
      {/* From the cast: one click adds a cast member's sheet, with no download and upload again. */}
      {room > 0 && onCast && cast.some((c) => !refs.some((r) => r.kind === "cast" && r.id === c.id)) ? (
        <div className="mt-2 flex flex-wrap gap-1.5" role="group" aria-label="Add from the cast">
          {cast
            .filter((c) => !refs.some((r) => r.kind === "cast" && r.id === c.id))
            .map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => onCast(c)}
                aria-label={`Add ${c.name}`}
                className="inline-flex h-8 items-center gap-1.5 rounded-full border border-line pl-1 pr-3 text-[13px] text-ink2 transition-colors hover:border-line2 hover:text-ink"
              >
                <img src={c.thumb_url || c.url} alt="" className="thumb-face h-6 w-6 rounded-full" />
                <Plus size={11} strokeWidth={2} />
                {c.name}
              </button>
            ))}
        </div>
      ) : null}
    </div>
  );
}

let refKey = 1;

/** Previz to final's draft: the Look and motion words, the start frame it accepted, and that frame's seed. */
type PrevizDraft = {
  look: string;
  motion: string;
  /** The app wrote the motion line: it follows the picked cast and the Look until the user types in it (Run 5, c09). */
  motionAuto?: boolean;
  /** method: how the start still was made (zbuf, B3), for the take's model list. */
  accepted: { upload_id?: string; ref?: RefPick; url: string; method?: string } | null;
  seed: number | null;
  /** Lock seed: the next start frame uses `seed` again (the same picture from the same words). */
  seedLocked?: boolean;
  /** Measured seconds the accepted start frame took. */
  took?: number | null;
};

/** Start frames this page is waiting on itself, by draft key: their answer comes back to the page, so the queue row
 * not being there yet never means "gone". */
const inFlight = new Set<string>();

export function PrevizForm({ shot, ui }: FormProps) {
  const t = useTask("previz", shot, ui);
  const prefill = usePrefill(shot.id, "previz");
  // A made shot opened again (no draft, no "Try again with changes") starts from its newest take: its Look words,
  // references, start frame and seed, never an empty Look box.
  const lastTake = useLastTake(shot, "previz", prefill);
  const pre = prefill ?? lastTake;
  const graded = useUploadSlot();
  // "Try again with changes" keeps the references by id (their files stay on the server), so their pictures come by id.
  const [refs, setRefs] = useState<Ref[]>(() => {
    const urls = (pre?._ref_urls ?? {}) as Record<string, string>;
    const kept = Array.isArray(pre?.look_upload_ids) ? (pre!.look_upload_ids as string[]) : [str(pre?.look_upload_id), str(pre?.look_upload_id_2)].filter(Boolean);
    const uploads = kept.map(
      (id): Ref => ({ key: refKey++, id, url: keptUrl({ kind: "upload", id }), name: `From ${keptFrom(pre)}`, busy: false, kind: "upload" }),
    );
    const picked = (Array.isArray(pre?.look_refs) ? (pre!.look_refs as RefPick[]) : []).map(
      (r): Ref => ({
        key: refKey++,
        id: r.id,
        url: urls[r.id] || keptUrl(r, shot.project_id),
        name: r.kind === "cast" ? "From the cast" : r.kind === "upload" ? `From ${keptFrom(pre)}` : "From Images",
        busy: false,
        kind: r.kind === "cast" ? "cast" : r.kind === "upload" ? "upload" : "image",
      }),
    );
    return [...uploads, ...picked].slice(0, MAX_REFS);
  });
  const [mode, setMode] = useState<"make" | "have">("make");
  // A made previz opens with its own blocking in words even without a prefill (motionPrompt.ts).
  const spec = motionSpecOf(shot.previz_spec);
  // v2.4 9: the previz path disagreeing with the sentence's camera words is flagged, not papered over in the text.
  const mismatch = spec ? cameraMismatch(spec, shot.previz_framing) : null;
  const cast = useCast(shot.project_id);
  // The words, the start frame and its seed live in the shot's draft (lib/drafts.ts): switching shot, task or rail, or
  // reloading, comes back to them. A prefill ("Try again with changes") starts a fresh one.
  const dr = useDraft<PrevizDraft>(
    shot.id,
    "previz",
    () => {
      const look = str(pre?.look_prompt) || str(pre?.describe);
      // R2-9: the short video line names only the cast this shot uses: its picked cast references and whoever the words
      // name; never the rest of the project's cast.
      const picks = new Set((Array.isArray(pre?.look_refs) ? (pre!.look_refs as RefPick[]) : []).filter((r) => r.kind === "cast").map((r) => r.id));
      const picked = cast.filter((c) => picks.has(c.id));
      const auto = !!spec && (!!pre?.prompt_auto || !str(pre?.prompt));
      return {
        look,
        motion: auto && spec ? motionPromptFor(spec, shot.previz_framing, { cast, picked: [...picked, ...loneCar(cast)], look }) : str(pre?.prompt),
        motionAuto: auto,
        accepted: pre?.look_frame_ref
          ? { ref: pre.look_frame_ref as RefPick, url: str(pre?._look_frame_url) || keptUrl(pre.look_frame_ref as RefPick, shot.project_id) }
          : str(pre?.look_frame_upload_id)
            ? { upload_id: str(pre?.look_frame_upload_id), url: keptUrl({ kind: "upload", id: str(pre?.look_frame_upload_id) }) }
            : null,
        seed: typeof pre?.start_seed === "number" ? (pre.start_seed as number) : null,
        seedLocked: pre?.seed_locked === true && typeof pre?.start_seed === "number",
        took: null,
      };
    },
    { fresh: !!prefill },
  );
  const lookPrompt = dr.data.look;
  const setLookPrompt = (look: string) => dr.set({ look });
  const prompt = dr.data.motion;
  const setPrompt = (motion: string) => dr.set({ motion, motionAuto: false });
  // Facing comes from the previz figure's rotation; words that say otherwise are flagged (the previz wins).
  const facing = facingMismatch([spec?.notes ?? "", lookPrompt, prompt].join(" "), shot.previz_framing, cast.map((c) => c.name));
  // Off by default: the start still goes in as a reference at position -1 only (the Layout-to-Render card's recipe).
  const [pin, setPin] = useState(bool(pre?.pin_start, false));
  const [lookAt, setLookAt] = useState<number>(typeof pre?.look_at === "number" ? (pre.look_at as number) : 0);
  const [grain, setGrain] = useState<Grain>(preGrain(pre?.grain));
  const [follow, setFollow] = useState<PrevizFollow>(pre?.follow === "loose" ? "loose" : "exact");
  const player = usePlayer();
  const [gen, setGen] = useState<{ state: "idle" | "busy" | "ready" | "error" | "canceled"; url?: string; upload_id?: string; error?: string; seed?: number }>(() => ({
    state: "idle",
    ...(dr.data.seed != null ? { seed: dr.data.seed } : {}),
  }));
  const [drift, setDrift] = useState<string | null>(null);
  const [remake, setRemake] = useState(false);
  // How many references the start frame took as pictures (the server's, from its response).
  const [sent, setSent] = useState<number | null>(null);
  const accepted = dr.data.accepted;
  const setAccepted = (a: PrevizDraft["accepted"]) => dr.set({ accepted: a });

  // A start frame in flight when the page was left or reloaded: find its inline queue row by the draft's key. Running
  // shows busy; done fills the start frame; failed says so; gone (the server restarted) lets it go.
  // Only a row made after the click counts (QA's QC: a new try showed the last failed try's error).
  const after = dr.draft.pendingAfter ?? 0;
  const pendingJob = useApp((s) => (dr.draft.pending === "start_frame" ? newestInline(s.jobs, dr.key, "start_frame", after) : undefined));
  useEffect(() => {
    if (dr.draft.pending !== "start_frame") return;
    if (!pendingJob) {
      if (inFlight.has(dr.key)) return; // this page asked: the answer comes to it
      let live = true;
      void loadJobs().then(() => {
        if (live && !inFlight.has(dr.key) && !newestInline(getState().jobs, dr.key, "start_frame", after)) dr.patch({ pending: null });
      });
      return () => {
        live = false;
      };
    }
    if (pendingJob.status === "running") setGen((g) => (g.state === "busy" ? g : { state: "busy", seed: g.seed }));
    else if (pendingJob.status === "done" && pendingJob.result) {
      const r = pendingJob.result;
      dr.patch((d) => ({ pending: null, data: { ...d.data, accepted: { upload_id: r.upload_id, url: r.url }, seed: r.seed ?? null, took: r.took_s ?? pendingJob.took_s ?? null } }));
      setGen({ state: "idle", seed: r.seed });
    } else if (pendingJob.status === "canceled") {
      dr.patch({ pending: null });
      setGen((g) => ({ state: "canceled", seed: g.seed }));
    } else if (pendingJob.status === "failed") {
      dr.patch({ pending: null });
      setGen({ state: "error", error: pendingJob.error || "The start frame didn't come out. Try again." });
    }
  }, [dr.draft.pending, pendingJob?.status]);
  const lookRef = useRef<HTMLDivElement>(null);
  const bracket = useShotWindow(shot, typeof pre?.start_frame === "number" ? Math.max(0, Math.round(pre.start_frame)) : null, null, false, false);

  useEffect(() => {
    if (accepted?.url) lookRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [accepted?.url]);

  const addRefs = (files: File[]) => {
    for (const f of files) {
      const key = refKey++;
      setRefs((r) => (r.length >= MAX_REFS ? r : [...r, { key, name: f.name, busy: true, kind: "upload" }]));
      getApi()
        .upload(f)
        .then((res) => setRefs((r) => r.map((x) => (x.key === key ? { ...x, id: res.id, url: res.url, busy: false } : x))))
        .catch((e) =>
          setRefs((r) => r.map((x) => (x.key === key ? { ...x, busy: false, error: e instanceof ApiError && e.message ? e.message : "Didn't upload" } : x))),
        );
    }
    // A new reference changes the look; a previewed frame no longer matches.
    setAccepted(null);
  };
  const addCastRef = (c: CastMember) => {
    setRefs((r) => (r.length >= MAX_REFS ? r : [...r, { key: refKey++, id: c.id, url: c.thumb_url || c.url, name: c.name, busy: false, kind: "cast" }]));
    setAccepted(null);
  };
  const removeRef = (key: number) => {
    setRefs((r) => r.filter((x) => x.key !== key));
    setAccepted(null);
  };

  // Uploaded files come first (images 2…), then references picked from the project.
  const ordered = [...refs.filter((r) => (r.kind ?? "upload") === "upload"), ...refs.filter((r) => r.kind === "image" || r.kind === "cast")];
  const ids = ordered.filter((r) => r.id && (r.kind ?? "upload") === "upload").map((r) => r.id!);
  const picks: RefPick[] = ordered.filter((r) => r.id && (r.kind === "image" || r.kind === "cast")).map((r) => ({ kind: r.kind as "image" | "cast", id: r.id! }));
  const uploading = refs.some((r) => r.busy);
  const versionId = t.baseId !== "orig" ? t.baseId : undefined;

  // The app's own motion line follows the cast picked here and the Look box (and the project's cast once it loads),
  // until the user types in it.
  const castPicks = picks.filter((p) => p.kind === "cast").map((p) => p.id).join(" ");
  useEffect(() => {
    if (!dr.data.motionAuto || !spec) return;
    const ids = new Set(castPicks.split(" "));
    const picked = [...cast.filter((c) => ids.has(c.id)), ...loneCar(cast)];
    const line = motionPromptFor(spec, shot.previz_framing, { cast, picked, look: lookPrompt });
    if (line !== prompt) dr.set({ motion: line });
    // spec and the framing come with the shot
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dr.data.motionAuto, castPicks, cast, lookPrompt, shot.id]);

  // Make start frame from previz: the creative director's start-frame template, filled on the server from the shot's
  // frame 0 and cast (start_frame.py), made with method B3, then pinned as the shot's first frame. A fresh seed each
  // time, so "Try another" is another picture; with Lock seed on, the same seed (the same picture from the same words).
  const makeStartFrame = async () => {
    const seed = nextSeed({ seed: dr.data.seed, locked: !!dr.data.seedLocked });
    setGen({ state: "busy", seed });
    setRemake(false);
    // Pending in the draft until the answer comes: a reload or a switch finds it again by its inline queue row, the
    // first one made after this click (never the last try's).
    dr.patch({ pending: "start_frame", pendingAfter: lastInlineAt(getState().jobs, dr.key, "start_frame") });
    inFlight.add(dr.key);
    try {
      // An older previz shot: its frame 0 rebuilt from its own spec first, so the words and the mask are a new shot's.
      // Best effort: when it can't, the plan says what's missing and offers Remake this shot.
      if (frame0IsThin(shot)) {
        try {
          const again = await frame0Again(shot.previz_spec);
          if (again) updateShot(await getApi().previzFrame0(shot.id, again));
        } catch {
          /* the plan below decides */
        }
      }
      // The Look as shown, auto or edited: the server takes its fixed look line out and merges the rest (start_frame.plan).
      const look = lookPrompt.trim();
      const res = await getApi().previzStartFrame({
        shot_id: shot.id,
        ...(versionId ? { version_id: versionId } : {}),
        look_upload_ids: ids,
        ...(picks.length ? { look_refs: picks } : {}),
        ...(look ? { user_look: look } : {}),
        seed,
        draft_key: dr.key,
      });
      const { url, upload_id } = res;
      if (!url || !upload_id) throw new Error("no picture");
      setRemake(!!res.action);
      setGen({ state: "idle", seed: res.seed ?? seed });
      dr.patch((d) => ({ pending: null, data: { ...d.data, accepted: { upload_id, url, method: res.method }, seed: res.seed ?? seed, took: res.took_s ?? null } }));
      setSent(res.references ?? null);
      setLookAt(0);
    } catch (e) {
      dr.patch({ pending: null });
      setRemake(e instanceof ApiError && e.status === 422);
      // Cancel answers 409: nothing was kept, and Try again makes it again.
      if (e instanceof ApiError && e.status === 409) setGen({ state: "canceled", seed });
      else setGen({ state: "error", error: e instanceof ApiError && e.message ? e.message : "The start frame didn't come out. Try again." });
    } finally {
      inFlight.delete(dr.key);
    }
  };
  // The start frame's seed (Lock seed: the next one uses it again) and how long it took.
  const shownSeed = dr.data.seed ?? gen.seed ?? null;
  const seedRow = shownSeed != null ? (
    <div className="flex items-center justify-end gap-3">
      {accepted && dr.data.took ? <span className="slate text-ink3" data-testid="start-took">{madeIn(dr.data.took)}</span> : null}
      <SeedLock seed={shownSeed} locked={!!dr.data.seedLocked} onChange={(seedLocked) => dr.set({ seedLocked })} testId="look-seed" />
    </div>
  ) : null;
  // One action for a shot whose frame 0 can't give the start frame what it needs: the Previz page with its sentence (and
  // the camera path, if the user drew one).
  const remakeButton = remake ? (
    <Button variant="secondary" size="md" className="w-full" onClick={() => openPreviz(spec?.notes, spec?.camera?.free ? shot.previz_spec : undefined)}>
      Remake this shot
    </Button>
  ) : null;

  const firstFrame: { upload_id?: string; ref?: RefPick } | null = mode === "have" ? (graded.value ? { upload_id: graded.value.id } : null) : accepted;
  // With no references and no look frame, the Look words are all the model has to go on.
  const lookRequired = mode === "make" && !accepted && ordered.length === 0;
  const valid = !uploading && (mode === "make" || !!graded.value) && (!lookRequired || !!lookPrompt.trim());

  // A start still for frame 0 (made here, from Images, or your own graded frame) should share the previz framing and size,
  // pinned or not.
  const pinnedAt0 = !!firstFrame && !lookAt;
  const firstKey = firstFrame?.ref ? `${firstFrame.ref.kind}:${firstFrame.ref.id}` : firstFrame?.upload_id ?? "";
  useEffect(() => {
    setDrift(null);
    if (!pinnedAt0 || !firstKey || !shot.previz_frame0) return;
    let dead = false;
    getApi()
      .previzFraming({ shot_id: shot.id, ...(firstFrame?.ref ? { look_frame_ref: firstFrame.ref } : { look_frame_upload_id: firstFrame?.upload_id }) })
      .then((r) => {
        if (!dead) setDrift(r.warning);
      })
      .catch(() => undefined);
    return () => {
      dead = true;
    };
    // firstKey stands for firstFrame
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pinnedAt0, firstKey, shot.id, shot.previz_frame0]);

  const paramsOf = (): PrevizParams => {
    const params: PrevizParams = { pin_start: pin, look_at: lookAt, grain, follow };
    if (mode === "make" && accepted?.method) params.start_method = accepted.method;
    // The start frame's seed and Lock seed go with the take, so the shot opens with them again.
    if (mode === "make" && accepted && dr.data.seed != null) {
      params.start_seed = dr.data.seed;
      params.seed_locked = !!dr.data.seedLocked;
    }
    if (prompt.trim()) params.prompt = prompt.trim();
    // The Look shown goes with every made start: it joins the video line and is saved on the take (details.look_prompt).
    if (mode === "make" && lookPrompt.trim()) params.look_prompt = lookPrompt.trim();
    if (firstFrame?.ref) params.look_frame_ref = firstFrame.ref;
    else if (firstFrame?.upload_id) params.look_frame_upload_id = firstFrame.upload_id;
    else {
      // References only make a look frame; with a first frame they'd ride along unused.
      params.look_upload_ids = ids;
      if (picks.length) params.look_refs = picks;
      if (ids[0]) params.look_upload_id = ids[0];
    }
    if (bracket.show && bracket.win) params.start_frame = bracket.win.start;
    return params;
  };

  // Everything stays with the shot (its references and the render), so this queues with no question; the shot's
  // name in the top bar moves it into a project any time.
  const submit = () => {
    void t.submit(paramsOf());
  };

  return (
    <>
    <FormShell
      task="previz"
      shot={shot}
      ui={ui}
      lengthNote={!bracket.show}
      footer={
        <>
          <PrimaryAction hint={t.etaLine} disabled={!valid || !t.ready} loading={t.submitting} onClick={submit}>
            Previz to final
          </PrimaryAction>
        </>
      }
    >
      <button type="button" onClick={() => openPreviz()} className="text-[13px] text-ink2 underline-offset-4 hover:text-ivory hover:underline">
        No previz? Make one
      </button>
      <section className="space-y-4">
        {mode === "have" ? (
          <>
            <DropSlot
              value={graded.value}
              busy={graded.busy}
              progress={graded.progress}
              error={graded.error}
              onFile={graded.onFile}
              onClear={graded.clear}
              title="Drop your graded first frame"
            />
            <Chip onClick={() => setMode("make")}>Use references instead</Chip>
          </>
        ) : (
          <>
            <References refs={ordered} onAdd={addRefs} onRemove={removeRef} cast={cast} onCast={addCastRef} sent={accepted && !accepted.ref ? sent : null} />
            <PromptField
              shot={shot}
              receipt={["qwen"]}
              uploads={ids.length}
              enhance={{
                target: "qwen-edit",
                shot_id: shot.id,
                ...(versionId ? { version_id: versionId } : {}),
                frame: lookAt,
                refs: [...ids.map((id): RefPick => ({ kind: "upload", id })), ...picks],
              }}
              label="Look"
              required={lookRequired}
              tools="none"
              value={lookPrompt}
              onChange={(v) => {
                setLookPrompt(v);
                if (accepted) setAccepted(null);
              }}
              rows={4}
              placeholder="Time of day, what they wear, the place"
            />

            {accepted ? (
              <div className="space-y-2.5 animate-rise-in" ref={lookRef}>
                <StartFrame url={accepted.url} />
                {gen.state === "busy" ? (
                  // Try another: the next one's progress and Cancel, under the one it may replace.
                  pendingJob && pendingJob.status === "running" ? (
                    <RunStrip job={pendingJob} label="Making the start frame" />
                  ) : (
                    <div className="flex items-center gap-2 text-[13px] text-ink2" role="status">
                      <Spinner size={14} className="text-ivory" />
                      Making the start frame…
                    </div>
                  )
                ) : (
                  <>
                    {gen.state === "canceled" ? <Cancelled onRetry={uploading || !t.ready ? undefined : () => void makeStartFrame()} /> : null}
                    {seedRow}
                    <div className="flex gap-2">
                      <Button variant="secondary" size="md" className="flex-1" disabled={uploading || !t.ready} onClick={() => void makeStartFrame()}>
                        Try another
                      </Button>
                      <Button variant="ghost" size="md" onClick={() => setAccepted(null)}>
                        Remove
                      </Button>
                    </div>
                  </>
                )}
                {remakeButton}
              </div>
            ) : gen.state === "busy" ? (
              <div className="relative overflow-hidden rounded-card border border-line" role="status">
                <Skeleton className="aspect-video w-full rounded-none" />
                {pendingJob && pendingJob.status === "running" ? (
                  <RunStrip job={pendingJob} label="Making the start frame" className="absolute inset-x-4 bottom-3" />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[13px] text-ink2">
                    <Spinner size={14} className="text-ivory" />
                    Making the start frame…
                  </div>
                )}
              </div>
            ) : gen.state === "canceled" ? (
              <Cancelled onRetry={uploading || !t.ready ? undefined : () => void makeStartFrame()} />
            ) : (
              <div className="space-y-2">
                {gen.state === "error" ? (
                  <p className="flex items-baseline gap-2 text-[13px] text-ink">
                    <span className="slate">Error</span>
                    {gen.error}
                  </p>
                ) : null}
                {gen.state === "error" ? remakeButton : null}
                <div className="flex items-center gap-2.5">
                  <Button variant="secondary" size="md" className="flex-1" disabled={uploading || !t.ready} onClick={() => void makeStartFrame()}>
                    Make start frame from previz
                  </Button>
                  <Hint text="Turns frame 1 of the previz into a real photo, using the words in the Look box above and your reference images. The previz sets where everyone stands and the camera; the Look sets who they are, the place and the light. Optional: without it the render makes its own first frame." />
                </div>
                {dr.data.seed != null ? seedRow : null}
              </div>
            )}
            <Chip onClick={() => setMode("have")}>Use my own first frame</Chip>
          </>
        )}
        {drift ? (
          <div className="flex items-start gap-1.5">
            <Notice tone="warn" label="Framing" text={drift} className="flex-1" />
            <Hint text="The start frame doesn't match the previz's framing exactly (how many people, how big, or where they stand). The render follows the start frame, so make it again if that matters." />
          </div>
        ) : null}
      </section>

      <section className="space-y-4">
        {mismatch || facing ? (
          <Notice
            tone="warn"
            label="Previz mismatch"
            text={[mismatch ? `The path ${mismatch}.` : "", facing ?? ""].filter(Boolean).join(" ")}
          />
        ) : null}
        {/* The Look's tools, and the motion (written from the previz itself): for pros. */}
        <Disclosure title="Advanced">
          {mode === "make" ? (
            <div className="space-y-2">
              <span className="block text-[13px] text-ink2">Look</span>
              <PromptTools
                shot={shot}
                value={lookPrompt}
                onChange={(v) => {
                  setLookPrompt(v);
                  if (accepted) setAccepted(null);
                }}
                hint="Add character sheets, a location plate or a mood still above, then refer to them as image 2, image 3…"
                tips="previz-look"
                receipt={["qwen"]}
                uploads={ids.length}
                enhance={{
                  target: "qwen-edit",
                  shot_id: shot.id,
                  ...(versionId ? { version_id: versionId } : {}),
                  frame: lookAt,
                  refs: [...ids.map((id): RefPick => ({ kind: "upload", id })), ...picks],
                }}
              />
            </div>
          ) : null}
          <PromptField
            shot={shot}
            tools="inline"
            receipt={["ltx"]}
            enhance={{ target: "ltx", shot_id: shot.id }}
            describe={{ field: "motion", ...(versionId ? { versionId } : {}) }}
            label="Motion and camera"
            tips="previz-motion"
            value={prompt}
            onChange={setPrompt}
            rows={4}
            placeholder="What moves, and how the camera moves"
          />
          <div>
            <SwitchRow checked={pin} onChange={setPin} title="Lock the first frame" hint="Also pins the start frame as frame 1. Off, it guides the look only." />
          </div>
          <div>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="flex items-center gap-1.5 text-[13px] text-ink2">
                Look frame at
                <Hint text="For shots where people walk in after the first frame." />
              </span>
              <span className="mono text-[12px] text-ink">{lookAt ? `${timecode(lookAt, shot.fps)} · ${lookAt}` : "First frame"}</span>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, shot.frames - 1)}
              step={1}
              value={lookAt}
              onChange={(e) => setLookAt(Number(e.target.value))}
              className="range"
              style={{ ["--fill" as string]: `${(lookAt / Math.max(1, shot.frames - 1)) * 100}%` }}
              aria-label="Paint the look over frame"
              aria-valuetext={lookAt ? `frame ${lookAt}` : "first frame"}
            />
            <div className="mt-1 flex items-center justify-between gap-3">
              <button type="button" onClick={() => setLookAt(0)} disabled={!lookAt} className="text-[12px] text-ink3 hover:text-ivory disabled:text-ink4">
                First frame
              </button>
              <button type="button" onClick={() => setLookAt(player.getSnapshot().frame)} className="text-[12px] text-ink2 underline-offset-4 hover:text-ivory hover:underline">
                Use current frame
              </button>
            </div>

          </div>
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[13px] text-ink2">
              Follow the previz
              <Hint text="Loosely keeps the camera and the people but redraws stand-ins (dust, smoke, grey blocks) as what you describe." />
            </div>
            <Segmented<PrevizFollow>
              value={follow}
              onChange={setFollow}
              label="Follow the previz"
              size="sm"
              options={[
                { value: "exact", label: "Exactly" },
                { value: "loose", label: "Loosely" },
              ]}
            />

          </div>
          <GrainControl value={grain} onChange={setGrain} />
        </Disclosure>

      </section>
    </FormShell>
    </>
  );
}

/** Previz to final's made start frame: click it to see it large (the one viewer, CharacterViewer); Esc or a click
 * outside the picture closes it, and focus comes back to the picture. */
function StartFrame({ url }: { url: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Start frame: open it large"
        className="group relative block w-full cursor-zoom-in overflow-hidden rounded-card border border-line"
      >
        <img src={url} alt="Start frame" className="aspect-video w-full object-cover transition-opacity duration-150 group-hover:opacity-90" />
        <Slate at="tl">Start frame</Slate>
      </button>
      {open ? <CharacterViewer title="Start frame" items={[{ key: url, label: "Start frame", url }]} index={0} onIndex={() => undefined} onClose={() => setOpen(false)} /> : null}
    </>
  );
}
