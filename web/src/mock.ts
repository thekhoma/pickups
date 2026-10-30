// Mock implementation of the Pickups API, for building and demoing the UI without the backend.
// Used when VITE_MOCK=1 (`npm run dev:mock`), with `?mock=1` in dev, or automatically in dev when /api is down.
//
// Scenarios (dev URL query): ?mock=fresh (first run, no shots) · ?mock=setup (missing models) · ?mock=offline (no ComfyUI)
// · ?mock=fail (Remove fails) · ?mock=stall · ?mock=busy
// Media is real: the shots, takes, receipts, people cutouts and cast in public/mock were cut from takes Pickups made
// (scripts/make_mock_media.py), and the takes carry their real sizes, lengths, receipts and settings. Jobs run SPEED×
// faster than the ETAs they report, so a "5 min" job finishes in about 15 s while the UI still reads like the real thing.
import { CARS } from "./lib/features";
import { ApiError, type PickupsApi } from "./api";
import {
  aiShare,
  allAi,
  duration,
  highlightsSentence,
  highlightsText,
  licenceName,
  percentText,
  receiptPercents,
  receiptShares,
  receiptSummary as shareSummary,
  sourceSentence,
  takeLength,
  upscaleSentence,
} from "./lib/format";
import { preview as mentionPreview } from "./lib/mentions";
import { isStillPrompt, stillPrompt, stillWords } from "./lib/still";
import { latestVideo } from "./lib/thumbs";
import { kindOf, medianOf } from "./lib/runtime";
import { freshSeed } from "./lib/seed";
import { DEFAULT_TOUR_COPY, loadTourCopy, type TourCopy } from "./lib/tourCopy";
import { framesFor, hasSize, VIDEO_DURATIONS, VIDEO_FPS, videoSize, type VideoAspect } from "./lib/video";
import type {
  VideoGenerateRequest,
  DescribeRequest,
  DescribeResult,
  DetectedServer,
  DirectorSettings,
  EnhanceRequest,
  EnhanceResult,
  ImageGenerateRequest,
  ImageModel,
  ImageModels,
  ProjectImage,
  PromptTestResult,
  Settings,
  ExamplesPack,
  ModelCatalog,
  ModelSlot,
  ShareFormat,
  ShareResult,
  TakeWorkflow,
  TimelineFormat,
  TrackRequest,
  TrackResult,
} from "./types";
import type {
  ApiEvents,
  CastKind,
  CastMember,
  CastSheet,
  Delivery,
  MentionPreview,
  Note,
  Project,
  ProjectDetail,
  Download,
  EnvironmentParams,
  PrevizParams,
  Example,
  ExampleOpenResult,
  FinishParams,
  Job,
  JobRequest,
  ModelLicence,
  PeopleRequest,
  PeopleResult,
  PersonFound,
  PrevizFramingResult,
  PrevizLookRequest,
  PrevizLookResult,
  PrevizStartFrameRequest,
  PrevizStartFrameResult,
  Receipt,
  RecastParams,
  RelightParams,
  Ref,
  ReshootParams,
  Sample,
  SelectRequest,
  SelectResult,
  SheetRequest,
  PrepareRequest,
  Shot,
  Status,
  TaskId,
  TileStatus,
  UploadResult,
  Version,
  EmptyTrashResult,
  StorageInfo,
  TrashEntry,
} from "./types";

const SPEED = 20;
const TICK_MS = 250;
/** The demo autopilot (?tour=1): renders take ~4 s, the director answers fast. */
const TOUR = typeof window !== "undefined" && /[?&]tour(=|&|$)/.test(window.location.search);
const media = (f: string) => `/mock/${f}`;

/** A poster frame from a video, as a data URL (null when it can't be read in time). */
function posterFrom(url: string, at = 0.4, timeoutMs = 3000, width = 480): Promise<string | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "auto";
    v.playsInline = true;
    const done = (r: string | null) => {
      window.clearTimeout(timer);
      v.removeAttribute("src");
      v.load();
      resolve(r);
    };
    const timer = window.setTimeout(() => done(null), timeoutMs);
    v.onloadeddata = () => {
      v.currentTime = Math.min(at, Math.max(0, (v.duration || 1) - 0.1));
    };
    v.onseeked = () => {
      try {
        const w = Math.min(width, v.videoWidth || width);
        const h = Math.round((w * (v.videoHeight || 270)) / (v.videoWidth || 480));
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        c.getContext("2d")!.drawImage(v, 0, 0, w, h);
        done(c.toDataURL("image/jpeg", 0.82));
      } catch {
        done(null);
      }
    };
    v.onerror = () => done(null);
    v.src = url;
  });
}

/** Draws a video into a rectangle, cropped to fill it. */
function cover(g: CanvasRenderingContext2D, v: HTMLVideoElement, x: number, y: number, w: number, h: number) {
  const vw = v.videoWidth || 16;
  const vh = v.videoHeight || 9;
  const k = Math.max(w / vw, h / vh);
  const sw = w / k;
  const sh = h / k;
  g.drawImage(v, (vw - sw) / 2, (vh - sh) / 2, sw, sh, x, y, w, h);
}

/**
 * A social cut-down made in the browser (canvas + MediaRecorder), so the demo's Share preview shows the actual take:
 * 9:16 before over after, or a 1:1 wipe, with the slate burned in. Null when the browser can't record.
 */
async function composeShare(beforeUrl: string, afterUrl: string, format: ShareFormat, slate: string): Promise<string | null> {
  if (typeof MediaRecorder === "undefined" || typeof HTMLCanvasElement.prototype.captureStream !== "function") return null;
  const load = (url: string) =>
    new Promise<HTMLVideoElement>((resolve, reject) => {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      const t = window.setTimeout(() => reject(new Error("timeout")), 5000);
      v.oncanplay = () => {
        window.clearTimeout(t);
        resolve(v);
      };
      v.onerror = () => reject(new Error("load"));
      v.src = url;
    });
  try {
    const [b, a] = await Promise.all([load(beforeUrl), load(afterUrl)]);
    const W = format === "9x16" ? 540 : 720;
    const H = format === "9x16" ? 960 : 720;
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const g = c.getContext("2d")!;
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    const rec = new MediaRecorder(c.captureStream(30), type ? { mimeType: type, videoBitsPerSecond: 5_000_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const dur = Math.min(TOUR ? 3 : 4, a.duration || 4, b.duration || 4);
    await Promise.all([a.play(), b.play()]);
    rec.start(250);
    const t0 = performance.now();
    const mono = (size: number) => `500 ${size}px "Geist Mono", ui-monospace, monospace`;
    const text = (s: string, x: number, y: number, size: number, color: string) => {
      g.font = mono(size);
      g.fillStyle = color;
      g.fillText(s.toUpperCase(), x, y);
    };
    await new Promise<void>((resolve) => {
      const draw = () => {
        const t = (performance.now() - t0) / 1000;
        g.fillStyle = "#0D0C0B";
        g.fillRect(0, 0, W, H);
        if (format === "9x16") {
          const h = Math.round((W * 9) / 16);
          const y1 = 176;
          const y2 = y1 + h + 8;
          cover(g, b, 0, y1, W, h);
          cover(g, a, 0, y2, W, h);
          g.fillStyle = "#F5301E";
          g.beginPath();
          g.arc(30, 120, 5, 0, Math.PI * 2);
          g.fill();
          text(slate, 44, 125, 15, "#EEE9E0");
          text("Before", 24, y1 - 12, 12, "#A39D93");
          text("After", 24, y2 + h + 26, 12, "#A39D93");
          text("Made locally with Pickups", 24, H - 60, 13, "#857F77");
        } else {
          const p = 0.5 + 0.32 * Math.sin(t * 1.4);
          cover(g, a, 0, 0, W, H);
          g.save();
          g.beginPath();
          g.rect(0, 0, W * p, H);
          g.clip();
          cover(g, b, 0, 0, W, H);
          g.restore();
          g.fillStyle = "rgba(238,233,224,.95)";
          g.fillRect(W * p - 1, 0, 2, H);
          g.fillStyle = "rgba(13,12,11,.82)";
          g.fillRect(0, H - 76, W, 76);
          g.fillStyle = "#F5301E";
          g.beginPath();
          g.arc(30, H - 46, 5, 0, Math.PI * 2);
          g.fill();
          text(slate, 44, H - 41, 15, "#EEE9E0");
          text("Made locally with Pickups", 44, H - 18, 12, "#857F77");
        }
        if (t >= dur) resolve();
        else window.setTimeout(draw, 33);
      };
      draw();
    });
    const stopped = new Promise((r) => (rec.onstop = r));
    rec.stop();
    await stopped;
    a.pause();
    b.pause();
    return chunks.length ? URL.createObjectURL(new Blob(chunks, { type: "video/webm" })) : null;
  } catch {
    return null;
  }
}

const TASK_TITLE: Record<TaskId, string> = { remove: "Remove a person", recast: "Replace a person", environment: "Change the weather", previz: "Previz to final", relight: "Relight", reshoot: "Reshoot", finish: "Finish" };

/** Each task's ComfyUI steps, in the order they run, named the way the backend names them ({frame}: the clicked frame). */
const WORKFLOW_STEPS: Record<TaskId, Array<{ step: string; title: string }>> = {
  remove: [
    { step: "pick", title: "Pick · SAM 3.1 · frame {frame}" },
    { step: "track", title: "Track · SAM 3.1 · forward from frame {frame}" },
    { step: "track", title: "Track · SAM 3.1 · backward from frame {frame}" },
  ],
  recast: [
    { step: "pick", title: "Pick · SAM 3.1 · frame {frame}" },
    { step: "track", title: "Track · SAM 3.1 · forward from frame {frame}" },
    { step: "recast", title: "Recast · SCAIL-2" },
    { step: "recast_new", title: "Find · SAM 3.1 · person" },
  ],
  environment: [
    { step: "environment_key", title: "First frame · Qwen-Image 2.1" },
    { step: "environment", title: "Weather · LTX-2.5 Union Control · {structure}" },
  ],
  relight: [{ step: "relight_scene", title: "Relight · LTX-2.3 Relight IC-LoRA" }],
  previz: [
    { step: "look", title: "Look frame · Qwen-Image 2.1" },
    { step: "previz", title: "Render · LTX-2.5 Layout-To-Render" },
  ],
  reshoot: [{ step: "reshoot", title: "Reshoot · LTX-2.5" }],
  finish: [
    { step: "upscale", title: "4K · SeedVR2" },
    { step: "hdr", title: "HDR · LTX SDR-to-HDR" },
  ],
};

/** A small API-format graph (node id → class_type + inputs) as a data: URL. Generic model file names. */
function workflowUrl(step: string, shotName: string): string {
  const graph = {
    "1": { class_type: "LoadVideo", inputs: { file: `${shotName}.mov` }, _meta: { title: "Shot" } },
    "2": { class_type: "UNETLoader", inputs: { unet_name: `${step}_model.safetensors`, weight_dtype: "default" }, _meta: { title: "Model" } },
    "3": { class_type: "CLIPLoader", inputs: { clip_name: "text_encoder.safetensors", type: "default" }, _meta: { title: "Text encoder" } },
    "4": { class_type: "CLIPTextEncode", inputs: { text: `Pickups ${step}`, clip: ["3", 0] }, _meta: { title: "Prompt" } },
    "5": { class_type: "KSampler", inputs: { model: ["2", 0], positive: ["4", 0], negative: ["4", 0], latent_image: ["1", 0], seed: 42, steps: 8, cfg: 1, sampler_name: "euler", scheduler: "simple", denoise: 1 }, _meta: { title: "Sample" } },
    "6": { class_type: "SaveVideo", inputs: { video: ["5", 0], filename_prefix: `pickups/${shotName}_${step}`, format: "auto", codec: "auto" }, _meta: { title: "Save" } },
  };
  return `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(graph, null, 2))}`;
}

/** HH:MM:SS:FF (non-drop). */
function tc(frames: number, fps: number): string {
  const r = Math.max(1, Math.round(fps));
  const f = Math.max(0, Math.round(frames));
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(Math.floor(f / (r * 3600)))}:${p(Math.floor(f / (r * 60)) % 60)}:${p(Math.floor(f / r) % 60)}:${p(f % r)}`;
}

function xmlEscape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- receipts -----------------------------------------------------------------------------------------------------
// The take receipt, built in the browser from the mock's take data and laid out as the backend's
// (src/pickups/receipts.py): the same CSS (RECEIPT_CSS is a copy of receipts.CSS: keep the two in step), numbered
// sections, the print button, the non-commercial callout and the index's thumbnails. The page is a Blob URL on this
// origin, so it uses the app's own font files; the takes' thumbnails stand in for the stills the backend grabs.
const RECEIPT_CSS = `:root{--paper:#f6f3ee;--card:#fffdf9;--ink:#171513;--ink2:#56514a;--ink3:#857f77;--line:#e1dacf;--line2:#c9c1b4;
--orig:#cbc3b6;--borrowed:#3dd6f5;--upscaled:#99948d;--ai:#ff4fd8;--tally:#f5301e;
--serif:"Instrument Sans",ui-sans-serif,system-ui,sans-serif;--mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:"Instrument Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html{background:var(--paper);color:var(--ink);-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-underline-offset:3px}
.sheet{max-width:980px;margin:0 auto;padding:44px 48px 40px}
.mono{font-family:var(--mono)}
.slate{font-family:var(--mono);font-size:11px;line-height:1.5;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--tally);margin-right:9px;vertical-align:1px}
.top{display:flex;justify-content:space-between;align-items:center;gap:16px;padding-bottom:12px;border-bottom:1px solid var(--ink)}
.top .slate:first-child{color:var(--ink)}
.pill{font:500 13px/1 var(--sans);color:var(--ink);background:none;border:1px solid var(--line2);border-radius:999px;padding:8px 14px;
cursor:pointer;text-decoration:none}
.pill:hover{border-color:var(--ink)}
.tools{display:flex;justify-content:flex-end;gap:8px;max-width:980px;margin:0 auto;padding:18px 48px 0}
.kicker{margin:30px 0 0}
h1{font:400 64px/1 var(--serif);letter-spacing:-.01em;margin:10px 0 0;overflow-wrap:anywhere}
.sub{font:400 30px/1.15 var(--serif);margin:8px 0 0;color:var(--ink2)}
.sub em{color:var(--ink)}
.facts{display:grid;grid-template-columns:minmax(0,1.5fr) repeat(4,minmax(0,1fr));margin:30px 0 0;border-top:1px solid var(--line);
border-bottom:1px solid var(--line)}
.facts div{padding:12px 14px 13px 0}
.facts dt{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.facts dd{margin:5px 0 0;font:13.5px/1.35 var(--mono);white-space:nowrap}
section{margin-top:46px}
h2{display:flex;align-items:baseline;gap:12px;font:400 30px/1.1 var(--serif);margin:0 0 16px;break-after:avoid;page-break-after:avoid}
.stills,figure,.stats,.zero,.mapfig,.tc,.facts,.callout,.prompt,tr,.notes li,.flows li{break-inside:avoid;page-break-inside:avoid}
h2 .n{font:11px/1 var(--mono);letter-spacing:.08em;color:var(--ink3)}
.stills{display:grid;grid-template-columns:1fr 1fr;gap:14px}
figure{margin:0}
figure img{display:block;width:100%;height:auto;border-radius:3px;background:#0d0c0b}
figure .none{display:flex;align-items:center;justify-content:center;aspect-ratio:16/9;border-radius:3px;background:#e9e3d9;
color:var(--ink3)}
figcaption{display:flex;justify-content:space-between;gap:12px;margin-top:8px}
figcaption .live::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--tally);
margin-right:7px;vertical-align:1px}
.frame{margin:10px 0 0;text-align:center}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line)}
.bar i{display:block;height:100%}
.bar .o{background:var(--orig)}.bar .b{background:var(--borrowed)}.bar .u{background:var(--upscaled)}.bar .a{background:var(--ai)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin-top:18px}
.stat b{display:block;font:400 52px/1 var(--serif)}
.stat span{display:inline-flex;align-items:center;gap:7px;margin-top:6px}
.sw{display:inline-block;width:8px;height:8px;border-radius:50%}
.sw.o{background:var(--orig)}.sw.b{background:var(--borrowed)}.sw.u{background:var(--upscaled)}.sw.a{background:var(--ai)}
.plain{font-size:15px;color:var(--ink2);margin:18px 0 0;max-width:62ch}
.plain strong{color:var(--ink);font-weight:600}
.zero{display:inline-flex;align-items:baseline;gap:10px;margin-top:16px;padding:10px 14px;border:1px solid var(--line2);border-radius:6px}
.zero b{font:400 28px/1 var(--serif)}
.mapfig{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start;margin-top:22px}
.mapfig p{margin:0;color:var(--ink2);font-size:14px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:9px 28px;margin:0}
dl.kv dt{color:var(--ink3);font-size:14px}
dl.kv dd{margin:0;font:13px/1.55 var(--mono);overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font:400 10.5px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);
padding:0 14px 8px 0;border-bottom:1px solid var(--line2)}
td{padding:11px 14px 11px 0;border-bottom:1px solid var(--line);vertical-align:top}
td.num,th.num{text-align:right;padding-right:0;white-space:nowrap}
td.file{width:50%;font:12.5px/1.5 var(--mono);overflow-wrap:break-word}
.terms{font-size:13px;line-height:1.45;color:var(--ink2)}
.tag{display:inline-block;font:10px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;padding:4px 6px 3px;
border:1px solid var(--line2);border-radius:3px;color:var(--ink2);white-space:nowrap}
.tag.nc{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.tag.ok{border-color:var(--line2)}
.callout{margin:0 0 16px;padding:12px 16px;border-left:3px solid var(--ink);background:var(--card);font-size:14px}
.prompt{margin-top:18px}
.prompt:first-of-type{margin-top:0}
.prompt pre{margin:7px 0 0;padding:13px 15px;font:12.5px/1.65 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere;
background:var(--card);border:1px solid var(--line);border-radius:4px}
.prompt pre.neg{color:var(--ink3);font-size:11.5px;padding:9px 15px}
.tc{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.tc b{display:block;margin-top:6px;font:400 26px/1.1 var(--mono);letter-spacing:.01em}
.notes{list-style:none;margin:0;padding:0}
.notes li{display:grid;grid-template-columns:170px 1fr;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)}
.notes .when b{display:block;font:400 13.5px/1.4 var(--mono)}
.notes p{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.flows{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}
.flows li{display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)}
.flows .t{min-width:0;overflow-wrap:anywhere}
.flows .t .n{font:12px var(--mono);color:var(--ink3);margin-right:10px}
.flows .t .more{color:var(--ink3)}
.flows .dl{font:12px var(--mono);white-space:nowrap;color:var(--ink2)}
.flows .dl a{margin-left:10px}
.hint{margin:12px 0 0;color:var(--ink2);font-size:14px}
.muted{color:var(--ink3)}
footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 16px;margin-top:60px;padding-top:12px;border-top:1px solid var(--ink)}
.idx td{vertical-align:middle}
.idx .muted{font-size:13px}
.idx img{display:block;width:132px;height:auto;border-radius:3px;background:#0d0c0b}
.idx .bar{height:6px;width:150px;margin-bottom:7px}
.idx .shot{font:400 22px/1.15 var(--serif)}
.idx .sum{font:11.5px/1.5 var(--mono);color:var(--ink2)}
.nw{white-space:nowrap}
@page{size:A4;margin:14mm 14mm 16mm}
@media print{html{background:#fff}.sheet{max-width:none;padding:0}.noprint{display:none!important}h1{font-size:52px}
section{margin-top:30px}footer{margin-top:28px;break-inside:avoid}a{text-decoration:none}.page+.page{break-before:page}}
/* Phones only: an A4 page is narrower than 760 px too, and prints the desktop layout. */
@media screen and (max-width:760px){.sheet{padding:28px 16px}.tools{padding:14px 16px 0}h1{font-size:44px}.sub{font-size:24px}
.facts{grid-template-columns:repeat(2,minmax(0,1fr))}.facts dd{white-space:normal}.stills,.mapfig{grid-template-columns:1fr}.stat b{font-size:38px}
.flows li{flex-wrap:wrap;gap:2px 16px}.flows .dl{white-space:normal}.flows .dl a:first-child{margin-left:0}
.tc{grid-template-columns:1fr 1fr}.tc b{font-size:18px}.notes li{grid-template-columns:1fr;gap:4px}
.idx thead{display:none}.idx tr{display:grid;grid-template-columns:96px minmax(0,1fr);gap:6px 14px;padding:16px 0;
border-bottom:1px solid var(--line)}.idx td{display:block;padding:0;border:0}.idx .th{grid-row:span 5}.idx img{width:96px}
.idx .num{text-align:left}.idx .bar{width:100%}}`;

/** What "depends on its terms" means for the licences the tasks use (licences.TERMS). */
const LICENCE_TERMS: Record<string, string> = {
  "LTX-2.x Community License": "Allowed for a business under $10M annual revenue (with affiliates); above that, a paid license from Lightricks",
  "SAM License": "Allowed, within Meta's trade-control and end-use limits",
};
const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
const pad2 = (n: number) => String(n).padStart(2, "0");
const rx = (s: unknown) => xmlEscape(String(s ?? ""));
const absUrl = (u?: string) => (u ? new URL(u, window.location.href).href : "");

/** The app's own @font-face rules with absolute URLs (a Blob URL can't resolve the app's relative ones). */
function appFontFaces(): string {
  const out: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    let rules: CSSRuleList;
    try {
      rules = sheet.cssRules;
    } catch {
      continue;
    }
    for (const r of Array.from(rules)) {
      if (!(r instanceof CSSFontFaceRule)) continue;
      const family = r.style.getPropertyValue("font-family").replace(/["']/g, "").trim();
      if (!["Instrument Sans", "Geist Mono"].includes(family)) continue;
      out.push(r.cssText.replace(/url\((["']?)([^"')]+)\1\)/g, (_m, q: string, u: string) => `url(${q}${new URL(u, sheet.href ?? window.location.href).href}${q})`));
    }
  }
  return out.join("\n");
}

function receiptDoc(title: string, articles: string[]): string {
  return `<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><meta name="generator" content="Pickups"><title>${rx(title)}</title><style>${appFontFaces()}${RECEIPT_CSS}</style></head>
<body><div class="tools noprint"><button class="pill" id="print" type="button">Print or save as PDF</button></div>
${articles.join("\n")}
<script>document.getElementById("print").addEventListener("click",function(){window.print()});</script></body></html>`;
}

/** receipts._bar: a segment per share (Finish's upscale in its own grey), labelled with the shares the receipt names. */
function receiptBar(r?: Receipt | null): string {
  const p = receiptPercents(r);
  if (!p) return "";
  const segs = (["original", "borrowed", "upscaled", "ai"] as const).map((k) => (p[k] ? `<i class="${k[0]}" style="width:${p[k]}%"></i>` : "")).join("");
  const label = receiptShares(r).map((s) => `${s.text} ${s.key === "ai" ? "AI" : s.label.toLowerCase()}`).join(", ");
  return `<div class="bar" role="img" aria-label="${rx(label)}">${segs}</div>`;
}

/** receipts.summary: "87% upscaled · 13% AI", "100% AI · whole frame generated (upscaled 2×)". */
function receiptSummary(r?: Receipt | null): string {
  return shareSummary(r) ?? "No receipt";
}

function receiptDate(ts: unknown): string {
  const d = new Date(Number(ts) * 1000);
  if (isNaN(d.getTime())) return "—";
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()} · ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

function receiptBytes(n?: number): string {
  if (!n) return "";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

function licenceTag(m: ModelLicence): string {
  if (m.commercial === false) return '<span class="tag nc">Non-commercial</span>';
  if (m.commercial) return '<span class="tag ok">Commercial use OK</span>';
  const terms = LICENCE_TERMS[m.licence ?? ""];
  if (terms) return `<span class="terms">${rx(terms)}</span>`;
  return `<span class="muted">${m.licence ? "Check its terms" : "No model license"}</span>`;
}

function licenceFlags(lic: ModelLicence[]): string {
  const nc = lic.filter((m) => m.commercial === false).map(licenceName);
  if (nc.length) return `<span class="tag nc">Non-commercial</span><div class="muted" style="margin-top:5px">${rx(nc.join(", "))}</div>`;
  const open = lic.filter((m) => m.commercial == null && m.licence);
  const cond = open.filter((m) => LICENCE_TERMS[m.licence ?? ""]).map(licenceName);
  const check = open.filter((m) => !LICENCE_TERMS[m.licence ?? ""]).map(licenceName);
  const lines = [...(cond.length ? [`Conditions apply: ${cond.join(", ")}`] : []), ...(check.length ? [`Check terms: ${check.join(", ")}`] : [])];
  if (lines.length) return `<span class="muted">${lines.map(rx).join("<br>")}</span>`;
  return lic.length ? '<span class="tag ok">Commercial use OK</span>' : '<span class="muted">—</span>';
}

/** The workflow rows as the take panel lists them: neighbours merged, the Rebuild step Pickups does itself in place. */
function receiptFlows(v: Version): Array<{ title: string; files: TakeWorkflow[] }> {
  const rows: Array<{ title: string; files: TakeWorkflow[] }> = [];
  for (const w of v.workflows ?? []) {
    const title = w.title || w.step;
    const last = rows[rows.length - 1];
    if (last && last.title === title) last.files.push(w);
    else rows.push({ title, files: [w] });
  }
  const camera = String(v.details?.camera ?? "");
  const method = String(v.details?.method ?? "");
  if (v.task === "remove" && (/^(locked|near-locked)/.test(camera) || method.startsWith("real pixels"))) {
    let at = 0;
    rows.forEach((r, i) => {
      if (/^(Pick|Track|Find|Shadow)\b/.test(r.title)) at = i + 1;
    });
    rows.splice(at, 0, { title: "Rebuild · real pixels, in Pickups", files: [] });
  }
  return rows;
}

const RECEIPT_DETAILS: Array<[string, string]> = [
  ["change", "Change"], ["control", "Structure"], ["grain", "Film grain"], ["hdr", "HDR"], ["quality", "Quality"], ["move", "Move"],
  ["references", "References"], ["pinned_first_frame", "First frame locked"],
];
const RECEIPT_PROMPTS: Array<[string, string]> = [
  ["text", "Selection"], ["fill", "Fill"], ["reference_prompt", "Reference still"], ["look_prompt", "Look prompt"], ["prompt", "Video prompt"],
  ["prompt_sent", "Prompt"],
];

function receiptArticle(projectName: string, shot: Shot, v: Version, anchor = ""): string {
  const n = shot.versions.indexOf(v) + 1;
  const num = pad2(n);
  const base = shot.versions.find((b) => b.id === (v.input_version || "orig")) ?? shot.versions[0];
  const d = v.details ?? {};
  const params = (v.params ?? {}) as Record<string, unknown>;
  const fps = Number(d.fps) || shot.fps;
  const frames = Number(d.frames) || shot.frames;
  const start = 3600 * Math.round(fps); // the mock's sources start at 01:00:00:00, as its timeline export
  const verb = TASK_TITLE[v.task as TaskId] ?? "Take";
  const label = v.label || verb;
  const p = receiptPercents(v.receipt);
  const clicked = Number(params.frame);
  const frame = Number.isInteger(clicked) && clicked >= 0 && clicked < frames ? clicked : Math.floor(frames / 2);
  const dims = (x: unknown) => (typeof x === "string" ? x.replace(/^(\d+)\s*x\s*(\d+)$/i, "$1×$2") : null);
  const delivered = dims(d.resolution) ?? (v.downloads.some((f) => f.size === "UHD") ? "3840×2160" : v.downloads.some((f) => f.size === "HD") ? "1920×1080" : null);
  const img = (u: string | undefined, alt: string) => (u ? `<img src="${rx(absUrl(u))}" alt="${rx(alt)}">` : '<div class="none slate">No picture</div>');
  const kv = (rows: Array<[string, unknown]>) =>
    `<dl class="kv">${rows.filter(([, val]) => val != null && val !== "").map(([k, val]) => `<dt>${rx(k)}</dt><dd>${rx(typeof val === "boolean" ? (val ? "Yes" : "No") : val)}</dd>`).join("")}</dl>`;
  const sections: Array<[string, string]> = [];

  sections.push(["Before and after", `<div class="stills">
<figure>${img(base.thumb_url, "Before")}<figcaption><span class="slate">Before</span><span class="slate">Take ${pad2(shot.versions.indexOf(base) + 1)} · ${rx(base.label)}</span></figcaption></figure>
<figure>${img(v.thumb_url, "After")}<figcaption><span class="slate live">After</span><span class="slate">Take ${num} · ${rx(label)}</span></figcaption></figure>
</div><p class="frame slate">Frame ${frame} · ${tc(frame, fps)} · Source ${tc(start + frame, fps)}</p>`]);

  if (p && v.receipt) {
    const r = v.receipt;
    const raw = [Number(r.original) || 0, Number(r.borrowed) || 0, Number(r.ai) || 0];
    const pct = (["original", "borrowed", "ai"] as const).map((k) => percentText(r, k).replace("<1%", "Under 1%").replace(">99%", "Over 99%"));
    // As receipts.take_body: Finish's upscale is the footage resampled (no new content); its HDR highlights are its own
    // AI share. Made from a take, it also says what that take already was.
    const finish = v.task === "finish";
    const upscale = finish ? upscaleSentence(r) : null;
    const highlights = finish ? highlightsSentence(r) : null;
    const extra = [finish ? sourceSentence(r) : null, highlights].filter(Boolean).map(rx).join(" ");
    const plain = allAi(r)
      ? `<strong>100% AI: the whole frame is generated.</strong>${[upscale, highlights].filter(Boolean).map((x) => ` ${rx(x)}`).join("")}`
      : upscale && !r.source_upscaled
        ? `<strong>${rx(upscale)}</strong>${extra ? ` ${extra}` : ""}`
        : `<strong>${pct[0]} original footage, untouched.</strong>${raw[1] > 0 ? ` ${pct[1]} real footage from other frames.` : ""}${upscale ? ` ${rx(upscale)}` : ""}${extra ? ` ${extra}` : raw[2] > 0 ? ` ${pct[2]} AI-generated.` : " Nothing AI-generated."}`;
    const outside = d.pixels_changed_outside_selection;
    const zero = typeof outside === "number" ? `<div class="zero"><b>${outside.toLocaleString("en-GB")}</b><span class="slate">${outside === 1 ? "pixel" : "pixels"} changed outside the selection</span></div>` : "";
    const stats = receiptShares(r).map((s) => `<div class="stat"><b>${s.text}</b><span class="slate"><i class="sw ${s.key[0]}"></i>${s.label}</span></div>`).join("\n");
    sections.push(["What changed", `${receiptBar(r)}
<div class="stats">${stats}</div>
<p class="plain">${plain}</p>${zero}`]);
  }

  sections.push(["How it was made", kv([
    ["Task", verb], ["Method", d.method], ["Camera", d.camera], ["Delivered", delivered], ["Rendered at", dims(d.rendered_at)],
    ["Length", takeLength(frames, fps)], ["Frame rate", `${+fps.toFixed(3)} fps`],
    ...RECEIPT_DETAILS.map(([k, l]): [string, unknown] => [l, d[k]]),
  ])]);

  if (v.licences?.length) {
    const nc = v.licences.filter((m) => m.commercial === false).map(licenceName);
    const callout = nc.length ? `<p class="callout"><strong>Not for paid work as it stands:</strong> ${rx(nc.join(", "))} ${nc.length === 1 ? "is" : "are"} licensed for non-commercial use only.</p>` : "";
    sections.push(["Models and licenses", `${callout}<table><thead><tr><th>Model</th><th>License</th><th>Commercial use</th></tr></thead><tbody>${v.licences
      .map((m) => `<tr><td>${rx(m.name)}${m.via ? ` <span class="muted">via ${rx(m.via)}</span>` : ""}</td><td>${rx(m.licence ?? "—")}</td><td>${licenceTag(m)}</td></tr>`)
      .join("")}</tbody></table>`]);
  }

  const prompts = RECEIPT_PROMPTS.map(([k, l]) => [l, d[k] ?? params[k]] as const).filter((x): x is readonly [string, string] => typeof x[1] === "string" && !!x[1].trim());
  const seed = d.seed ?? params.seed;
  if (prompts.length || seed != null) {
    sections.push(["Prompts", prompts.map(([l, t]) => `<div class="prompt"><div class="slate">${rx(l)}</div><pre>${rx(t.trim())}</pre></div>`).join("") + (seed != null ? `<p class="hint">Seed ${rx(seed)}.</p>` : "")]);
  }

  sections.push(["Timecode", `<div class="tc"><div><span class="slate">In</span><b>${tc(start, fps)}</b></div><div><span class="slate">Out</span><b>${tc(start + frames, fps)}</b></div>
<div><span class="slate">Length</span><b>${frames} fr</b></div></div>
<p class="hint">Source timecode. Out is exclusive, as in an EDL.</p>`]);

  if (v.downloads.length) {
    sections.push(["Files", `<table><thead><tr><th>File</th><th>What it is</th><th class="num">Size</th></tr></thead><tbody>${v.downloads
      .map((f) => `<tr><td class="file">${rx(f.filename ?? f.label).replace(/_/g, "_<wbr>")}</td><td>${rx(f.label)}</td><td class="num mono">${receiptBytes(f.bytes)}</td></tr>`)
      .join("")}</tbody></table>`]);
  }

  // As the backend: the app's timecode first (from the take's first frame), the source timecode under it.
  // An approved take also carries the original's notes at its own frames, as the export folds them in.
  const orig = shot.versions[0];
  const from = v.status === "approved" && orig.status !== "approved" ? mockTakeStart(shot, v) : null;
  const folded = from == null ? [] : (orig.notes ?? []).map((x) => ({ ...x, frame: Math.round(x.frame) - from, text: `${x.text} (original)` }));
  const notes = [
    ...(v.notes ?? []).map((x) => ({ ...x, frame: Math.min(frames - 1, Math.max(0, Math.round(x.frame))) })),
    ...folded.filter((x) => x.frame >= 0 && x.frame < frames),
  ].sort((a, b) => a.frame - b.frame);
  if (notes.length) {
    sections.push(["Notes", `<ul class="notes">${notes
      .map((x) => `<li><span class="when"><b>${tc(x.frame, fps)}</b><span class="slate">Source ${tc(start + x.frame, fps)}</span></span><p>${rx(x.text)}</p></li>`)
      .join("")}</ul><p class="hint">Timecode from the take's first frame, as Pickups shows it; the source timecode under it is where the note sits in the edit.</p>`]);
  }

  const flows = receiptFlows(v);
  if (flows.some((r) => r.files.length)) {
    const lis = flows.map((r, i) => {
      const [what, ...rest] = r.title.split(" · ");
      const title = rx(what) + (rest.length ? `<span class="more"> · ${rx(rest.join(" · "))}</span>` : "") + (r.files.length > 1 ? `<span class="more"> ×${r.files.length}</span>` : "");
      const dl = r.files.length
        ? r.files.map((w) => `<a href="${rx(absUrl(w.url))}" download="${rx(w.filename ?? `${w.step}.json`)}">JSON</a>`).join("")
        : '<span class="muted">In Pickups, no workflow</span>';
      return `<li><span class="t"><span class="n">${pad2(i + 1)}</span>${title}</span><span class="dl">${dl}</span></li>`;
    });
    sections.push(["ComfyUI workflows", `<ul class="flows">${lis.join("")}</ul><p class="hint">Each one is the exact ComfyUI workflow this take ran, and opens from the take in Pickups too. Drag one onto ComfyUI to see what ran; running it again needs the source media and the same models.</p>`]);
  } else {
    sections.push(["ComfyUI workflows", '<p class="hint" style="margin:0">This take was made before Pickups kept its ComfyUI workflows.</p>']);
  }

  const stem = `${shot.name.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "shot"}_T${num}_${v.task ?? "take"}`;
  const facts: Array<[string, string]> = [
    ["Made", receiptDate(v.created_at)], ["Total time", d.total_seconds ? duration(Number(d.total_seconds)) : "—"],
    ["GPU time", d.gpu_seconds ? duration(Number(d.gpu_seconds)) : "—"], ["Delivered", delivered ?? "—"], ["Length", takeLength(frames, fps).split(" · ")[0]],
  ];
  const today = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  return `<article class="sheet page"${anchor ? ` id="${rx(anchor)}"` : ""}><header class="top"><span class="slate"><i class="dot"></i>Pickups receipt</span><span class="slate">${rx(stem)}</span></header>
<p class="kicker slate">${rx(projectName || "Project")}</p><h1>${rx(shot.name)}</h1><p class="sub">Take ${num} · <em>${rx(label)}</em></p>
<dl class="facts">${facts.map(([k, val]) => `<div><dt>${k}</dt><dd>${rx(val)}</dd></div>`).join("")}</dl>
${sections.map(([t, c], i) => `<section><h2><span class="n">${pad2(i + 1)}</span>${rx(t)}</h2>${c}</section>`).join("\n")}
<footer><span class="slate">Made with Pickups · open source</span><span class="slate">${rx(today)}</span></footer></article>`;
}

/** "<Project>" as the backend names a project's downloads (server.attachment, the name the UI saves under). */
function projectFileStem(name: string): string {
  return name.replace(/[^\p{L}\p{N}_]+/gu, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "Pickups";
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
let seq = 100;
/** For browser checks and agents: what the forms sent, as the server would get it (window.__mockSent, mock only). */
function sent(route: string, body: unknown): void {
  const w = globalThis as { __mockSent?: { route: string; body: unknown }[] };
  (w.__mockSent ??= []).push({ route, body: clone(body) });
}
const uid = (p: string) => `${p}_${(seq++).toString(36)}${Math.random().toString(36).slice(2, 6)}`;
/** As the backend: the Examples project lists only shots with something to compare (a take that isn't hidden). */
const listed = (s: Shot) => s.project_id !== "examples" || s.versions.some((v) => v.id !== "orig" && !v.hidden);

// --- media map ----------------------------------------------------------------------------------------------------
// Every shot has a media key: `<key>.mp4` / `<key>.jpg` is its original, `<key>_<task>` the real take a job on it
// returns, with the take's receipt overlay (`_receipt.mp4`) and an aligned before clip (`_before.mp4`: the take was
// rendered at another size, or from later in the shot) where it has them. The takes a shot starts with that aren't its
// `<key>_<task>` have names of their own (`glen_recast_b`). All of it is cut from real takes (scripts/make_mock_media.py).
const KNOWN_VIDEOS = new Set([
  "trail", "trail_remove", "castle_dusk", "castle_dusk_environment", "castle_walk", "castle_walk_recast", "castle_walk_environment",
  "alpine_ride", "alpine_ride_environment", "glen", "glen_recast", "glen_recast_b", "glen_long", "glen_long_recast",
  "offroad", "offroad_recast", "coast_curve", "coast_curve_recast", "misty_road", "misty_road_environment", "misty_road_environment_b",
  "coastal_drive", "coastal_drive_finish", "hood_push", "hood_push_previz", "hood_push_previz_a", "hood_push_previz_b",
  "driver_profile", "driver_profile_previz", "launch", "launch_previz", "ocean_person", "ocean_person_previz",
]);
/** Takes with a receipt overlay, `<name>_receipt.mp4`. */
const WITH_RECEIPT = new Set(["trail_remove", "castle_walk_recast", "glen_recast", "glen_recast_b", "glen_long_recast", "offroad_recast", "coast_curve_recast", "misty_road_environment"]);
/** Takes with an aligned before clip, `<name>_before.mp4`. */
const WITH_BEFORE = new Set(["castle_dusk_environment", "castle_walk_environment", "alpine_ride_environment", "glen_long_recast", "misty_road_environment"]);
/** Recast takes made from words keep the still FLUX 3 klein made, `<name>_reference.jpg`. */
const WITH_REFERENCE = new Set(["offroad_recast", "coast_curve_recast"]);
/** What each real take's receipt says, where it isn't the task's usual share (SHARES). */
const RESULT_SHARES: Record<string, Receipt> = {
  trail_remove: { original: 96.72, borrowed: 3.28, ai: 0 },
  castle_walk_recast: { original: 97.9, borrowed: 0, ai: 2.1 },
  glen_recast: { original: 95.43, borrowed: 0, ai: 4.57 },
  glen_recast_b: { original: 95.44, borrowed: 0, ai: 4.56 },
  glen_long_recast: { original: 95.69, borrowed: 0, ai: 4.31 },
  offroad_recast: { original: 78.79, borrowed: 0, ai: 21.21 },
  coast_curve_recast: { original: 74.81, borrowed: 0, ai: 25.19 },
  misty_road_environment_b: { original: 11.14, borrowed: 0, ai: 88.86 },
};
/** Finish: the share of pixels whose highlights LTX rebuilt on the real take (else SHARES.finish.ai). */
const RESULT_HIGHLIGHTS: Record<string, number> = { coastal_drive_finish: 44.38 };
/** What the real takes were: their length and render size, and how a Remove was done. */
const RESULT_FACTS: Record<string, Record<string, unknown>> = {
  trail_remove: { frames: 165, camera: "locked-off", method: "real pixels from other frames", pixels_changed_outside_selection: 0 },
  castle_dusk_environment: { frames: 97, rendered_at: "1920×1024" },
  castle_walk_recast: { frames: 97, pixels_changed_outside_selection: 0 },
  castle_walk_environment: { frames: 97, rendered_at: "1920×1024" },
  alpine_ride_environment: { frames: 97, rendered_at: "1920×1024" },
  glen_recast: { frames: 97, rendered_at: "3840×2748", pixels_changed_outside_selection: 0 },
  glen_recast_b: { frames: 97, rendered_at: "3840×2748", pixels_changed_outside_selection: 0 },
  glen_long_recast: { frames: 97, rendered_at: "3840×2748", pixels_changed_outside_selection: 0 },
  offroad_recast: { frames: 97, subject: "vehicle", pixels_changed_outside_selection: 0 },
  coast_curve_recast: { frames: 57, subject: "vehicle", pixels_changed_outside_selection: 0 },
  misty_road_environment: { frames: 97, rendered_at: "1920×1024" },
  misty_road_environment_b: { frames: 97, rendered_at: "1920×1024" },
  coastal_drive_finish: { frames: 97 },
  hood_push_previz: { frames: 145, rendered_at: "1920×1088" },
  hood_push_previz_a: { frames: 97, rendered_at: "1920×1088" },
  hood_push_previz_b: { frames: 97, rendered_at: "1920×1088" },
  driver_profile_previz: { frames: 97, rendered_at: "1920×1088" },
  launch_previz: { frames: 97, rendered_at: "1920×1088" },
  ocean_person_previz: { frames: 97, rendered_at: "1920×1088" },
};
/** Remove and Recast on a clip of your own (an upload, which has no real take): a receipt overlay drawn for the demo. */
const GENERIC_RECEIPT: Partial<Record<TaskId, string>> = { remove: "receipt_borrowed_center.png", recast: "receipt_ai_center.png" };
/** Change weather with "Keep exactly as shot": the kept thing's share of the picture, as the backend's receipt counts it
 * (the misty road's car). */
const KEPT_SHARE = 11.06;

/** What a Change weather take keeps as shot, in the prompts' words ("" for nothing): "car" is "the car". */
function keepOf(env: EnvironmentParams): string {
  const t = (env.keep_text ?? "").trim().replace(/\.$/, "");
  if (t) return /^(the|a|an|this|that|my|our|your|his|her|their)\s/i.test(t) ? t : `the ${t}`;
  return env.keep_points?.length ? "the object you clicked" : "";
}

const SHARES: Record<TaskId, Receipt> = {
  remove: { original: 95.2, borrowed: 3.9, ai: 0.9 },
  recast: { original: 93.8, borrowed: 0, ai: 6.2 },
  environment: { original: 0, borrowed: 0, ai: 100 },
  previz: { original: 0, borrowed: 0, ai: 100 },
  relight: { original: 0, borrowed: 0, ai: 100 },
  reshoot: { original: 82.5, borrowed: 0, ai: 17.5 },
  finish: { original: 91.4, borrowed: 0, ai: 8.6 },
};
// The names the backend's tasks write into details.models ("Qwen-Image 2.1" is the frame editor; commercial-safe
// mode swaps in "Qwen-Image-Edit 2511").
const MODELS: Record<TaskId, string[]> = {
  remove: ["SAM 3.1", "Qwen-Image 2.1"],
  recast: ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3", "SAM 3.1"],
  environment: ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3"],
  previz: ["Qwen-Image 2.1", "LTX-2.5 Layout-To-Render"],
  relight: ["LTX-2.3 + Relight IC-LoRA"],
  reshoot: ["ReCamMaster"],
  finish: ["SeedVR2 3B", "LTX-2.5 SDR-to-HDR"],
};

// Mirrors src/pickups/licences.py: [licence, commercial] per model name (null commercial: depends on its terms).
const LTX_LICENCE = "LTX-2.x Community License";
const LICENCES: Record<string, [string | null, boolean | null]> = {
  "LTX-2.5": [LTX_LICENCE, null],
  "LTX-2.5 + Union Control IC-LoRA": [LTX_LICENCE, null],
  "LTX-2.5 Layout-To-Render": [LTX_LICENCE, null],
  "LTX-2.5 SDR-to-HDR": [LTX_LICENCE, null],
  "SAM 3.1": ["SAM License", null],
  "Qwen-Image 2.1": ["Qwen Research License", false],
  "Qwen-Image-Edit 2511": ["Apache-2.0", true],
  "SCAIL-2 (Wan 2.1)": ["MIT", true],
  "SeedVR2 3B": ["Apache-2.0", true],
  "Depth Anything 3": ["Apache-2.0", true],
  "FLUX 3 klein": ["FLUX Kommunity License", null],
  "NVIDIA RTX Video Super Resolution": ["NVIDIA proprietary", null],
  Canny: [null, null],
};

/** receipts.compose: a Finish made from a take, as shares of the shot's original footage. With an upscale
 * ai = hl + (100 - hl) × in.ai, borrowed = (100 - hl) × in.borrowed, upscaled = the rest; without one the input's
 * shares scale by (100 - hl). Adds `highlights` (hl) and `source` (the take, its label and its shares). */
function composeReceipt(own: Receipt, source: Receipt | undefined, take: string, label: string): Receipt {
  const src: NonNullable<Receipt["source"]> = { take, label };
  const ins = source ? [source.original, source.borrowed, source.upscaled ?? 0, source.ai].map((x) => Number(x) || 0) : [];
  const sum = ins.reduce((a, b) => a + b, 0);
  if (!source || sum <= 0) return { ...own, source: src };
  const [io, ib, iu, ia] = ins.map((x) => x / sum);
  const [o, b, u, a] = [own.original, own.borrowed, own.upscaled ?? 0, own.ai].map((x) => Number(x) || 0);
  const r2 = (x: number) => Math.round(x * 100) / 100;
  const up = o * iu + u * (io + iu);
  const out: Receipt = { ...own, original: r2(o * io), borrowed: r2(o * ib + b * (io + ib + iu) + u * ib), ai: r2(a + (o + b + u) * ia), highlights: r2(a) };
  if (own.upscaled != null || up > 0) out.upscaled = r2(up);
  if (own.upscaled == null && up > 0) {
    out.source_upscaled = true; // this take didn't upscale; its input did
    if (source.upscaler) out.upscaler = source.upscaler;
    if (source.scale) out.scale = source.scale;
  }
  src.receipt = { original: source.original, borrowed: source.borrowed, ...(source.upscaled != null ? { upscaled: source.upscaled } : {}), ai: source.ai };
  return { ...out, source: src };
}

/** receipts.hdr_detail: a Finish take's details.hdr, rounded as its receipt is. */
function hdrDetail(r: Receipt): string | undefined {
  const t = highlightsText(r);
  if (!t) return undefined;
  const up = (Number(r.upscaled) > 0 || !!r.upscaler || Number(r.scale) > 0) && !r.source_upscaled;
  return `highlights rebuilt by LTX-2.5 on ${t} of pixels; everything else is ${r.source?.take ? `Take ${r.source.take}` : "your footage"}${up ? ", upscaled" : ""}`;
}

/** Where a take starts in its shot's original: params.start_frame added up along input_version (nle.take_start). */
function mockTakeStart(shot: Shot, take: Version): number | null {
  let start = 0;
  const seen = new Set<string>();
  for (let id = take.id; id !== "orig"; ) {
    const v = shot.versions.find((x) => x.id === id);
    if (!v || seen.has(id)) return null; // a broken or looping chain: the take can't be placed
    seen.add(id);
    const s = Number(v.params?.start_frame ?? 0);
    start += Number.isFinite(s) && s > 0 ? Math.floor(s) : 0;
    id = v.input_version || "orig";
  }
  return start;
}

function licencesFor(models: unknown): ModelLicence[] {
  const names = Array.isArray(models) ? models.filter((m): m is string => typeof m === "string") : [];
  return [...new Set(names)].map((name) => {
    const [licence, commercial] = LICENCES[name] ?? LICENCES[name.replace(/\s*\([^)]*\)$/, "")] ?? [null, null];
    return { name, licence, commercial };
  });
}

/** Details as the backend writes them: the top delivery size, the size it was rendered at when that differs. */
function sizeDetails(w: number, h: number, frames: number, fps: number): Record<string, unknown> {
  const [dw, dh] = Math.max(w, h) >= 3000 ? [3840, 2160] : [1920, 1080];
  return { resolution: `${dw}×${dh}`, ...(w !== dw || h !== dh ? { rendered_at: `${w}×${h}` } : {}), frames, fps };
}

// Stages mirror the backend's human copy, with its relative time estimates.
const STAGES: Record<TaskId, Array<[string, number]>> = {
  remove: [["Preparing the shot", 20], ["Finding what you clicked", 25], ["Rebuilding the background", 90], ["Painting what was never seen", 60], ["Blending at full resolution", 150], ["Writing your files", 30]],
  recast: [["Preparing the shot", 15], ["Finding the person", 35], ["Reading the character sheet", 10], ["Painting them into the shot", 40], ["Performing the scene", 120], ["Blending at full resolution", 95], ["Writing your files", 45]],
  environment: [["Preparing the shot", 20], ["Restyling the first frame", 90], ["Rendering the new weather", 140], ["Writing your files", 40]],
  relight: [["Preparing the shot", 15], ["Relighting the scene", 320], ["Writing your files", 30]],
  previz: [["Preparing the previz", 15], ["Making the look frame", 60], ["Rendering the shot", 170], ["Writing your files", 30]],
  reshoot: [["Reading the scene in 3D", 60], ["Moving the camera", 30], ["Rendering the new angle", 170], ["Writing your files", 20]],
  finish: [["Preparing the shot", 20], ["Upscaling to 4K", 450], ["Rebuilding the highlights", 240], ["Merging HDR at full resolution", 240], ["Writing your files", 90]],
};
// Change weather that keeps something as shot, as tasks/environment.py.
const ENV_KEEP_STAGES: Array<[string, number]> = [["Preparing the shot", 20], ["Finding what to keep", 45], ["Restyling the first frame", 90], ["Rendering the new weather", 140], ["Keeping it as shot", 45], ["Writing your files", 40]];
// Recast's SCAIL-2 engine (STAGES.recast is the default, LTX-2.5), as tasks/recast.py.
const RECAST_SCAIL_STAGES: Array<[string, number]> = [["Preparing the shot", 20], ["Finding the person", 25], ["Reading the character sheet", 40], ["Performing the scene", 180], ["Blending at full resolution", 120], ["Writing your files", 30]];
const REAL_SECONDS: Record<TaskId, number> = { remove: 14, recast: 18, environment: 14, previz: 16, relight: 15, reshoot: 13, finish: 18 };
const WEATHER_LABEL: Record<string, string> = { snow: "Snow", rain: "Rain", fog: "Fog", night: "Night", golden_hour: "Golden hour", autumn: "Autumn" };
const WEATHER_CONTROL: Record<string, string> = { snow: "depth", rain: "canny", fog: "depth", night: "canny", golden_hour: "canny", autumn: "depth" };

// --- samples ------------------------------------------------------------------------------------------------------
interface SampleDef extends Sample {
  key: string;
  shotName: string;
  fps: number;
  frames: number;
  width: number;
  height: number;
}
const NTSC = 30000 / 1001;
/** The demo's shots, as the library holds them: source size, frame rate and length. */
const SAMPLES: SampleDef[] = [
  { id: "castle_dusk", name: "Castle on the loch", thumb_url: media("castle_dusk.jpg"), key: "castle_dusk", shotName: "Castle on the loch", fps: NTSC, frames: 98, width: 3840, height: 2160 },
  { id: "trail", name: "Tre Cime trail", thumb_url: media("trail.jpg"), key: "trail", shotName: "Tre Cime trail", fps: NTSC, frames: 165, width: 3840, height: 2160 },
  { id: "castle_walk", name: "Castle walk", thumb_url: media("castle_walk.jpg"), key: "castle_walk", shotName: "Castle walk", fps: NTSC, frames: 97, width: 7680, height: 4320 },
  { id: "alpine_ride", name: "Alpine e-bike ride", thumb_url: media("alpine_ride.jpg"), key: "alpine_ride", shotName: "Alpine e-bike ride", fps: NTSC, frames: 98, width: 3840, height: 2160 },
  { id: "glen", name: "Glen walk-in", thumb_url: media("glen.jpg"), key: "glen", shotName: "Glen walk-in", fps: 24, frames: 101, width: 4224, height: 3024 },
  { id: "glen_long", name: "Glen walker", thumb_url: media("glen_long.jpg"), key: "glen_long", shotName: "Glen walker", fps: 24, frames: 260, width: 4224, height: 3024 },
  { id: "offroad", name: "Rural road SUV", thumb_url: media("offroad.jpg"), key: "offroad", shotName: "Rural road SUV", fps: 24, frames: 137, width: 1920, height: 1080 },
  { id: "coast_curve", name: "Coast road curve", thumb_url: media("coast_curve.jpg"), key: "coast_curve", shotName: "Coast road curve", fps: 24, frames: 59, width: 1920, height: 1080 },
  { id: "misty_road", name: "Misty road SUV", thumb_url: media("misty_road.jpg"), key: "misty_road", shotName: "Misty road SUV", fps: 24, frames: 126, width: 1920, height: 1080 },
  { id: "coastal_drive", name: "Coastal drive", thumb_url: media("coastal_drive.jpg"), key: "coastal_drive", shotName: "Coastal drive", fps: 24, frames: 97, width: 1920, height: 1080 },
  { id: "hood_push", name: "K03 hood push", thumb_url: media("hood_push.jpg"), key: "hood_push", shotName: "K03 hood push", fps: 24, frames: 145, width: 1280, height: 720 },
  { id: "driver_profile", name: "K05 driver profile", thumb_url: media("driver_profile.jpg"), key: "driver_profile", shotName: "K05 driver profile", fps: 24, frames: 217, width: 1280, height: 720 },
  { id: "launch", name: "K06 launch", thumb_url: media("launch.jpg"), key: "launch", shotName: "K06 launch", fps: 24, frames: 121, width: 1280, height: 720 },
  { id: "ocean_person", name: "Seafront pier previz", thumb_url: media("ocean_person.jpg"), key: "ocean_person", shotName: "Seafront pier previz", fps: 24, frames: 97, width: 1920, height: 1088 },
];
const sample = (id: string) => SAMPLES.find((x) => x.id === id)!;

// The look and video prompts the real takes ran with (Change weather's custom looks), for "Try it" and the receipts.
const DUSK_LOOK =
  "Change the season, time of day and light in the image to a snowy winter dusk: thick fresh snow covers the castle roofs, the battlements, the stone bridge parapets, the grassy island and the rocky seaweed foreshore, the sky is a deep blue-hour blue, warm candle-yellow light glows in the castle windows, the water of the loch is dark slate-blue, and large snowflakes fall through the cold air. Keep the exact framing, camera angle and geometry, every person's face, pose and clothing, and every vehicle's exact design, colour and markings; only the season, time of day and light change.";
const DUSK_PROMPT =
  "A snowy winter dusk at a stone castle on a sea loch, thick fresh snow on the roofs, battlements, bridge and rocky shore, a deep blue-hour sky, warm candle-yellow light in the castle windows and cold blue light on the snow. A few tiny figures on the arched stone bridge. Heavy snowflakes fall steadily and drift sideways in a light wind over the dark water.";
const AUTUMN_LOOK =
  "Change the season and light in the image to late autumn: the meadows turn golden-brown and amber, the grass along the road is dry and tawny, the larches among the dark green spruces glow yellow-gold, the first snow dusts the tops of the jagged peaks ahead, fallen leaves and needles scatter along the road edges, and a soft, warm, low afternoon sun from the right casts long shadows across the road. Keep the exact framing, camera angle and geometry, every person's face, pose and clothing, and every vehicle's exact design, colour and markings; only the season and light change.";
const AUTUMN_PROMPT =
  "Late autumn on an alpine plateau, golden-brown and amber meadows, yellow-gold larches among dark green spruces, the first snow dusting the jagged peaks ahead, soft warm low sunlight from the right with long shadows across the narrow paved road. A woman with long dark hair in a white T-shirt, black trousers and a white cap on an e-bike. Dry leaves skitter across the road in the breeze and the meadow grass ripples.";
const OFFROADER = "a vintage 1970s two-door off-road 4x4 in olive green with a white roof, round headlights, chrome bumpers and chunky off-road tyres";
const OFFROAD_SCENE = "It drives along a wet two-lane rural road across flat, frosty farmland under a flat grey winter sky, with power poles along the roadside; the camera tracks alongside it.";
const CONVERTIBLE = "a cherry-red 1960s convertible sports car with the top down, chrome bumpers, round headlights and whitewall tyres";
const CURVE_SCENE = "It drives around a wet coastal curve towards the camera, past a steel guardrail above a grey, stormy sea under heavy overcast cloud.";
const GLEN_SCENE = "He walks out across the green moorland of a wide Scottish glen towards a dark mountain under a heavy overcast sky, in soft diffuse daylight, seen from a static camera.";
const WALKER = "A woman with short curly red hair in a bright yellow hooded raincoat, dark green trousers and brown boots walks along the path.";

const ASTRONAUT = "an astronaut in a white spacesuit with a gold-tinted visor, white gloves, chunky white boots and a white life-support pack on his back";
const GLEN_PROMPT = `An astronaut in a white spacesuit with a gold-tinted visor, white gloves, chunky white boots and a white life-support pack on his back walks in from the left and away from the camera. ${GLEN_SCENE} They are lit by the scene's own natural light, with soft, realistic shadows and no glow around them; the camera and the place stay exactly as in the original shot.`;
const GLEN_LOOK_TURN = `Turn the man walking on the left of <image1> into the character from the character sheet in <image2> (${ASTRONAUT}): keep the man's exact pose, mid-stride, with the head, arms and legs exactly where they are, and the same position, size and direction in the picture. Change the face, hair, body and clothes to look exactly like the character in <image2>, from head to toe, lit by the same light as the scene in <image1>, with no rim light and no glow around them. Keep everything else in <image1> exactly as it is.`;
const GLEN_LOOK_REPLACE = `Replace the man walking on the left of <image1> with the character from the character sheet in <image2> (${ASTRONAUT}): the same position, size, pose and direction as the man, wearing exactly what the character wears in <image2>, from head to toe, lit by the same light as the scene in <image1>, with no rim light and no glow around them. Keep everything else in <image1> exactly as it is.`;
/** Recast's look and video prompts for a vehicle (tasks/recast_ltx.py), as the coast road takes ran them. */
const vehicleLook = (who: string) =>
  `Replace the SUV in the middle of <image1> with the vehicle in <image2> (${who}): the same position, size and angle in the picture, facing and travelling the same way, its wheels on the road exactly where the SUV's wheels are. Make it look exactly like the vehicle in <image2>, its body, colour, wheels and details, lit by the same light as the scene in <image1>, with matching reflections and its own soft shadow on the road. Keep everything else in <image1> exactly as it is.`;
const vehiclePrompt = (who: string, scene: string) =>
  `${who.charAt(0).toUpperCase()}${who.slice(1)} drives steadily along the road, the camera travelling with it. ${scene} It is lit by the scene's own light, with real reflections on its paint and glass and a soft shadow on the road; the camera and the place stay exactly as in the original shot.`;
const MISTY_DESCRIBE = "A blue SUV drives toward the camera along a curving two-lane road with a steel guardrail on the left and bare winter trees on both sides; the camera tracks backward ahead of it.";
const HOOD_LOOK =
  "The yellow mannequin is the old stunt driver from image 2: the same weathered face, white walrus moustache, cream open-face helmet with navy stars and sage-green work shirt, with a black four-point racing harness over his shoulders. The car's roll cage, seat, dashboard, dusty windshield and the salvage yard behind the car look exactly like image 3. Warm low evening sun from behind the camera lights his face.";
const HOOD_PROMPT =
  "An old stunt driver sits alone behind the wheel of a stripped stunt car, seen through the dusty windshield as the camera on the hood slowly pushes in toward his face. He has a weathered face, a white walrus moustache and a cream open-face helmet with navy stars, and wears a sage-green work shirt under a black racing harness. He looks back from the side window to the road ahead, draws a knuckle along his moustache and lets out a long, tired breath, his shoulders dropping. Warm, low evening sun from behind the camera lights his face; stacks of wrecked cars and a white box truck stand in the salvage yard behind the car. Shot on 65mm film, natural skin texture, fine grain.";
const PROFILE_LOOK =
  "The yellow mannequin is the old stunt driver from image 2: the same weathered face, white walrus moustache, cream helmet with navy stars and sage-green work shirt, under a black four-point harness, sitting in a black racing seat. The car is a stripped grey police sedan with a black roll cage, a worn black dashboard and a black steering wheel; through the windshield and side window, a sandy salvage yard with stacked wrecked cars in warm low evening sun.";
const PROFILE_PROMPT =
  "Inside a stripped stunt car, an old stunt driver grips the steering wheel with both hands, seen in profile from just behind the passenger seat. He has a weathered face, a white walrus moustache and a cream open-face helmet with navy stars, and wears a sage-green work shirt under a black racing harness. He stares down the road ahead, breathes out slowly and tightens his grip on the wheel. Warm low evening sunlight pours through the windshield; outside, a sandy salvage yard with stacks of wrecked cars. Live-action film, natural skin texture, fine grain.";
const LAUNCH_LOOK =
  "The white car is the dusty grey police sedan from image 2, seen from behind, with plain black steel wheels. The sandy lane, the walls of stacked wrecked cars on both sides, the low sun and the hazy city skyline look exactly like image 3.";
const LAUNCH_PROMPT =
  "A dusty grey stunt car accelerates away from the camera down a long sandy lane between towering walls of stacked wrecked cars, its rear wheels throwing up plumes of sand and dust. The low camera stays still as the car shrinks toward the setting sun and the hazy downtown skyline at the end of the lane. Golden backlight glows through the dust cloud and rims the car, and long shadows stretch toward the camera. Live-action film, fine grain.";
const LOCH_PROMPT =
  "Photograph, a misty Scottish sea loch at blue hour, a small stone castle on a tidal island joined by an arched stone bridge, wet seaweed on the rocks in the foreground, soft cold light, fine film grain";
// The notes the backend writes on a take (tasks/environment.py, tasks/previz.py).
const GENERATED_NOTE = "Every pixel is generated from your shot's structure; chain Finish for a 4K master.";
const KEPT_NOTE = "The car is your footage: its detail as shot, its light and colour from the render. Everything else is generated from your shot's structure; chain Finish for a 4K master.";
const PREVIZ_NOTE = "Camera and blocking from your previz; look from the look frame.";

/** A take the demo library starts with (MockApi.seedTake): the real take's media name, settings and what it reported. */
interface TakeSeed {
  /** The take's media name (KNOWN_VIDEOS), when it isn't the shot's `<key>_<task>`. */
  media?: string;
  status?: "approved";
  /** What the real take reported (merged over what the mock works out). */
  details?: Record<string, unknown>;
  /** The ComfyUI steps that ran: [step, title]. */
  workflows?: Array<[string, string]>;
  /** [frame, text] */
  notes?: Array<[number, string]>;
  /** Render previz: the look frame it was painted from. */
  look?: string;
}

/** The backend's models_catalog.catalog(), for demo mode: each home workflow's default models. */
function mockCatalog(safe: boolean): ModelCatalog {
  const editor = safe ? "Qwen-Image-Edit 2511" : "Qwen-Image 2.1";
  const other = safe ? "Qwen-Image 2.1" : "Qwen-Image-Edit 2511";
  const slot = (role: string, name: string, short = name, options: string[] = [name], when?: string): ModelSlot => ({ role, name, short, options, ...(when ? { when } : {}) });
  const paint = (role: string, when?: string) => slot(role, editor, editor, [editor, other], when);
  const union = "LTX-2.5 + Union Control IC-LoRA";
  const recast = (): ModelSlot[] => [
    slot("video", union, "LTX-2.5", [union, "SCAIL-2 (Wan 2.1)"]),
    paint("frame paint"),
    slot("tracking", "SAM 3.1"),
    slot("depth", "Depth Anything 3"),
    slot("still from text", "FLUX 3 klein", "FLUX 3 klein", ["FLUX 3 klein"], "when you describe them"),
  ];
  return {
    character: { task: "recast", models: recast() },
    car: { task: "recast", models: recast() },
    weather: {
      task: "environment",
      models: [slot("video", union, "LTX-2.5"), paint("first frame"), slot("depth", "Depth Anything 3", "Depth Anything 3", ["Depth Anything 3", "Canny"])],
    },
    remove: {
      task: "remove",
      models: [slot("tracking", "SAM 3.1"), paint("clean plate", "when part is never seen"), slot("fill", "LTX-2.5", "LTX-2.5", ["LTX-2.5"], "on moving shots")],
    },
    relight: { task: "relight", models: [slot("video", "LTX-2.3 + Relight IC-LoRA", "LTX-2.3 Relight")] },
    previz: { task: "previz", models: [slot("video", "LTX-2.5 Layout-To-Render", "LTX-2.5", ["LTX-2.5 Layout-To-Render", union]), paint("look frame")] },
    sheet: { task: "sheet", models: [paint("views"), slot("front from text", "FLUX 3 klein", "FLUX 3 klein", ["FLUX 3 klein"], "when you describe them")] },
    finish: {
      task: "finish",
      models: [slot("4K upscale", "SeedVR2 3B", "SeedVR2", ["SeedVR2 3B", "NVIDIA RTX Video Super Resolution"]), slot("HDR", "LTX-2.5 SDR-to-HDR", "LTX-2.5 HDR")],
    },
    image: { task: "image", models: [slot("image", "FLUX 3 klein", "FLUX 3 klein", ["FLUX 3 klein", ...IMAGE_MODELS.models.map((m) => m.name).filter((n) => n !== "FLUX 3 klein")])] },
  };
}

/** The models the seafront pier previz ran (its take's details.models), shown on its slide. */
const OCEAN_MODELS: ModelSlot[] = [
  { role: "look frame", name: "Qwen-Image 2.1", short: "Qwen-Image 2.1", options: ["Qwen-Image 2.1"] },
  { role: "first frame", name: "Z-Image Turbo + Fun ControlNet Union 2.1", short: "Z-Image Turbo", options: ["Z-Image Turbo + Fun ControlNet Union 2.1"] },
  { role: "video", name: "LTX-2.5 Layout-To-Render", short: "LTX-2.5", options: ["LTX-2.5 Layout-To-Render"] },
];

/** Car swap's example, shown only when the build offers car swap (lib/features). */
const CAR_EXAMPLES: Example[] = [
  {
    id: "ex_offroad_recast",
    task: "recast",
    title: "Swap a car",
    line: "The road, rain and camera stay real.",
    slate: "Replace a person",
    sample_id: "offroad",
    params: { engine: "ltx", frame: 60, points: [], text: "SUV", describe: OFFROADER, scene: OFFROAD_SCENE },
    before_url: media("offroad.mp4"),
    after_url: media("offroad_recast.mp4"),
    receipt_url: media("offroad_recast_receipt.mp4"),
    poster_url: media("offroad_recast.jpg"),
  },
];

const EXAMPLES: Example[] = [
  {
    id: "ex_trail_remove",
    task: "remove",
    title: "Remove a hiker",
    line: "Rebuilt from real frames.",
    slate: "Remove a person",
    sample_id: "trail",
    params: { frame: 82, points: [], text: "woman", shadow: false, fill: "the continuing green grass slopes, grey scree and winding hiking trails" },
    before_url: media("trail.mp4"),
    after_url: media("trail_remove.mp4"),
    receipt_url: media("trail_remove_receipt.mp4"),
    // The poster is the after's still (it stands in for the after video until that loads).
    poster_url: media("trail_remove.jpg"),
    hero: true,
  },
  {
    id: "ex_glen_recast",
    task: "recast",
    title: "Astronaut",
    line: "From a character sheet.",
    slate: "Replace a person",
    sample_id: "glen",
    // Opens the Fun cast's shot, where the Astronaut and the sheet live (picked on the people finder's frame, where the
    // demo has the walker's real matte).
    shot_id: "shot_glen",
    params: { engine: "ltx", frame: 48, points: [], text: "man", cast_id: "cast_astronaut", scene: GLEN_SCENE },
    before_url: media("glen.mp4"),
    after_url: media("glen_recast.mp4"),
    receipt_url: media("glen_recast_receipt.mp4"),
    poster_url: media("glen_recast.jpg"),
  },
  {
    id: "ex_castle_dusk",
    task: "environment",
    title: "Winter dusk",
    line: "Summer afternoon to snowy blue hour.",
    slate: "Change the weather",
    sample_id: "castle_dusk",
    params: { custom: "Castle in a winter dusk", look_prompt: DUSK_LOOK, prompt: DUSK_PROMPT, control: "depth" },
    before_url: media("castle_dusk_environment_before.mp4"),
    after_url: media("castle_dusk_environment.mp4"),
    poster_url: media("castle_dusk_environment.jpg"),
  },
  {
    id: "ex_ocean_person",
    task: "previz",
    title: "Seafront pier",
    line: "From a previz and a look frame.",
    slate: "Previz to final",
    sample_id: "ocean_person",
    params: { pin_start: true, follow: "exact", grain: "fine" },
    before_url: media("ocean_person.mp4"),
    after_url: media("ocean_person_previz.mp4"),
    poster_url: media("ocean_person_previz.jpg"),
    models_used: OCEAN_MODELS,
  },
  {
    id: "ex_alpine_autumn",
    task: "environment",
    title: "Autumn",
    line: "Same ride, a season later.",
    slate: "Change the weather",
    sample_id: "alpine_ride",
    params: { custom: "Summer ride to autumn", look_prompt: AUTUMN_LOOK, prompt: AUTUMN_PROMPT, control: "depth" },
    before_url: media("alpine_ride_environment_before.mp4"),
    after_url: media("alpine_ride_environment.mp4"),
    poster_url: media("alpine_ride_environment.jpg"),
  },
  {
    id: "ex_castle_walk",
    task: "recast",
    title: "New walker",
    line: "Only the person changes.",
    slate: "Replace a person",
    sample_id: "castle_walk",
    params: { engine: "scail", frame: 0, points: [], text: "person walking on the path", describe: WALKER },
    before_url: media("castle_walk.mp4"),
    after_url: media("castle_walk_recast.mp4"),
    receipt_url: media("castle_walk_recast_receipt.mp4"),
    poster_url: media("castle_walk_recast.jpg"),
  },
  {
    id: "ex_coastal_finish",
    task: "finish",
    title: "Coastal drive in 4K HDR",
    line: "HD in, UHD HLG master out.",
    slate: "Finish",
    sample_id: "coastal_drive",
    params: { upscale_4k: true, hdr: true, quality: "best" },
    before_url: media("coastal_drive.mp4"),
    after_url: media("coastal_drive_finish.mp4"),
    poster_url: media("coastal_drive_finish.jpg"),
  },
  ...(CARS ? CAR_EXAMPLES : []),
];

// --- status -------------------------------------------------------------------------------------------------------
// "busy": ComfyUI is running a job Pickups didn't send (status.comfy_queue.outside).
type Scenario = "ready" | "setup" | "offline" | "fail" | "fresh" | "stall" | "busy" | "refuse";

function scenario(): Scenario {
  const v = new URLSearchParams(window.location.search).get("mock");
  return v === "setup" || v === "offline" || v === "fail" || v === "fresh" || v === "stall" || v === "busy" || v === "refuse" ? v : "ready";
}

function tile(task: TaskId, missing: TileStatus["missing"] = []): TileStatus {
  const est = STAGES[task].reduce((a, [, s]) => a + s, 0);
  return { ready: missing.length === 0, eta_s: Math.round(est * 0.92), missing };
}

const MISSING: Partial<Record<TaskId, TileStatus["missing"]>> = {
  remove: [
    { file: "sam2.1_hiera_large.safetensors", gb: 0.9, gated: false, url: "https://huggingface.co/facebook/sam2.1-hiera-large" },
    { file: "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors", gb: 0.3, gated: true, url: "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate" },
  ],
  previz: [
    { file: "ltx-2.5-layout-to-render_fp8.safetensors", gb: 12.4, gated: true, url: "https://huggingface.co/Lightricks" },
  ],
  // As manifest.TASK_MODELS["relight"].
  relight: [
    { file: "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors", gb: 25.2, gated: false, url: "https://huggingface.co/Kijai/LTX2.3_comfy" },
    { file: "gemma_3_12B_it_fp8_scaled.safetensors", gb: 13.2, gated: false, url: "https://huggingface.co/Comfy-Org/ltx-2" },
    { file: "ltx-2.3_text_projection_bf16.safetensors", gb: 2.3, gated: false, url: "https://huggingface.co/Kijai/LTX2.3_comfy" },
    { file: "LTX23_video_vae_bf16.safetensors", gb: 1.5, gated: false, url: "https://huggingface.co/Kijai/LTX2.3_comfy" },
    { file: "LTX23_audio_vae_bf16.safetensors", gb: 0.4, gated: false, url: "https://huggingface.co/Kijai/LTX2.3_comfy" },
    { file: "ltx-2.3-22b-ic-lora-relight-1.0.safetensors", gb: 0.3, gated: true, url: "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Relight" },
  ],
};

/** Setup → Benchmarks rows for one GPU: [task, frames, size, median total s, median GPU s, runs, fastest s, slowest s]. */
function benchRows(gpu: string, current: boolean, host: string, rows: Array<[string, number, string, number, number, number, number, number]>): Status["benchmarks"] {
  return rows.map(([task, frames, size, seconds, gpu_seconds, runs, fastest, slowest]) => ({
    task, frames, size, seconds, gpu_seconds, runs, fastest, slowest, gpu, current, host, measured: true,
    s_per_frame: Math.round((seconds / frames) * 100) / 100,
  }));
}

function makeStatus(kind: "ready" | "setup" | "offline", comfy_url = "http://127.0.0.1:8188"): Status {
  if (kind === "offline") {
    const off: TileStatus = { ready: false, eta_s: null, missing: [] };
    return {
      connected: false,
      comfy_url,
      machine: null,
      verdict: "offline",
      tiles: { remove: off, recast: off, environment: off, relight: off, previz: off, reshoot: off, finish: off, image: off, video: off },
      benchmarks: [],
      max_frames: 97,
      max_long: 3840,
      min_frames: 9,
      previz_frames: 97,
    };
  }
  const miss = (t: TaskId) => (kind === "setup" ? MISSING[t] ?? [] : []);
  return {
    connected: true,
    comfy_url,
    machine: { name: "Studio", gpu: "NVIDIA GeForce RTX 4090", vram_gb: 25.8, ram_gb: 68.7, comfy_version: "0.41.2", vram_free_gb: 19.8 },
    verdict: "ready",
    tiles: {
      remove: tile("remove", miss("remove")),
      recast: {
        ...tile("recast", miss("recast")),
        eta_by_engine: { ltx: tile("recast").eta_s ?? 0, scail: Math.round(RECAST_SCAIL_STAGES.reduce((a, [, s]) => a + s, 0) * 0.92) },
      },
      environment: tile("environment", miss("environment")),
      relight: tile("relight", miss("relight")),
      previz: {
        ...tile("previz", miss("previz")),
        ...(kind === "setup" ? { ready: false, missing_nodes: [{ pack: "ComfyUI-LTXVideo", url: "https://github.com/Lightricks/ComfyUI-LTXVideo", by: "Lightricks" }] } : {}),
      },
      reshoot: { ...tile("reshoot"), ready: false },
      finish: tile("finish", miss("finish")),
      image: { ready: true, eta_s: 10, missing: [] },
      // As manifest.TASK_MODELS["video"]: LTX-2.5 alone. The setup scenario lacks the upscaler, so the video tabs say so.
      video:
        kind === "setup"
          ? { ready: false, eta_s: 60, missing: [{ file: "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", gb: 1.0, gated: false, url: "https://huggingface.co/Lightricks" }] }
          : { ready: true, eta_s: 60, missing: [] },
    },
    benchmarks: [
      ...benchRows("RTX 4090", true, "remote", [
        ["Remove, locked-off", 202, "3840×2748", 502, 25, 6, 470, 765],
        ["Remove, moving camera", 97, "3840×2160", 495, 205, 5, 465, 640],
        ["Replace a person (SCAIL-2)", 81, "3840×2160", 276, 179, 5, 244, 582],
        ["Replace a person (LTX-2.5)", 97, "3840×2160", 541, 379, 1, 541, 541],
        ["Change the weather", 97, "1920×1024", 185, 146, 8, 140, 333],
        ["Previz to final", 97, "1920×1088", 223, 206, 12, 210, 374],
        ["Finish, best quality", 97, "3840×2160", 1509, 1261, 3, 1485, 2489],
      ]),
      ...benchRows("RTX PRO 6000 Blackwell Server Edition", false, "runpod", [
        ["Remove, locked-off", 202, "3840×2748", 470, 102, 2, 455, 486],
        ["Replace a person (SCAIL-2)", 81, "3840×2160", 168, 96, 3, 151, 190],
        ["Change the weather", 97, "1920×1024", 118, 84, 2, 112, 124],
        ["Previz to final", 97, "1920×1088", 142, 125, 3, 131, 160],
      ]),
    ],
    max_frames: 97,
    max_long: 3840,
    min_frames: 9,
    previz_frames: 97,
  };
}

// --- images -------------------------------------------------------------------------------------------------------
const ASPECTS = ["1:1", "4:5", "3:4", "2:3", "9:16", "16:9", "3:2", "4:3", "21:9"];
function sizesFor(maxSide: number): ImageModel["sizes"] {
  const out: ImageModel["sizes"] = { "1K": {}, "2K": {} };
  for (const res of ["1K", "2K"] as const) {
    const area = res === "1K" ? 1024 * 1024 : Math.min(2048, maxSide) ** 2;
    for (const a of ASPECTS) {
      const [w, h] = a.split(":").map(Number);
      let W = Math.round(Math.sqrt((area * w) / h) / 64) * 64;
      let H = Math.round(W * (h / w) / 64) * 64;
      const k = Math.min(1, (res === "2K" ? Math.min(2048, maxSide) : 1536) / Math.max(W, H));
      W = Math.round((W * k) / 64) * 64;
      H = Math.round((H * k) / 64) * 64;
      out[res]![a] = [W, H];
    }
  }
  return out;
}
const IMAGE_MODELS: ImageModels = {
  default: "flux3-klein",
  aspects: ASPECTS,
  resolutions: ["1K", "2K"],
  models: [
    { id: "flux3-klein", name: "FLUX 3 klein", maker: "Black Forest Labs", licence: "FLUX Kommunity License", commercial: true, modes: ["generate", "edit"], max_refs: 10, max_side: 2048, speed: "~5 s at 1K on a 4090", sponsor: true, notes: "Fast all-rounder: photos, people, multi-reference edits", ready: true, sizes: sizesFor(2048) },
    { id: "qwen-image-2.1", name: "Qwen-Image 2.1", maker: "Qwen (Alibaba)", licence: "Qwen Research License (non-commercial)", commercial: false, modes: ["generate", "edit"], max_refs: 10, max_side: 2048, speed: "~25 s at 1K on a 4090", notes: "Best for instruction edits and lettering, native 2K", ready: true, sizes: sizesFor(2048) },
    { id: "qwen-image-2.1-turbo", name: "Qwen-Image 2.1 Turbo", maker: "Qwen (Alibaba)", licence: "Qwen Research License (non-commercial)", commercial: false, modes: ["generate", "edit"], max_refs: 10, max_side: 2048, speed: "~8 s at 1K on a 4090", notes: "Qwen-Image 2.1 in 8 steps", ready: true, sizes: sizesFor(2048) },
    { id: "krea2-turbo", name: "Krea 2 Turbo", maker: "Krea", modes: ["generate", "style"], max_refs: 3, max_side: 1024, speed: "~6 s at 1K on a 4090", notes: "Aesthetic stills; borrows a look from style references", ready: false, sizes: sizesFor(1024) },
    // The rest of pickups/images.py's registry, so the demo's Model list is the real one (all seven).
    { id: "z-image-turbo", name: "Z-Image Turbo", maker: "Tongyi-MAI (Alibaba)", licence: "Apache 2.0", commercial: true, modes: ["generate"], max_refs: 0, max_side: 2048, speed: "~4 s at 1K on a 4090", notes: "Fast photoreal, English and Chinese text", ready: true, sizes: sizesFor(2048) },
    { id: "ideogram4", name: "Ideogram 4", maker: "Ideogram", licence: "Ideogram 4 Non-Commercial", commercial: false, modes: ["generate"], max_refs: 0, max_side: 2048, speed: "~16 s at 1K on a 4090", notes: "Best for lettering and layout (JSON prompts)", ready: true, sizes: sizesFor(2048) },
    { id: "hidream-o1", name: "HiDream-O1", maker: "HiDream.ai", licence: "MIT", commercial: true, modes: ["generate", "edit"], max_refs: 10, max_side: 2560, speed: "~45 s at 2K on a 4090", notes: "Made for 2K: slow and detailed; 1K comes out soft", ready: true, sizes: sizesFor(2560) },
  ],
};
// --- the Fun cast: seven FLUX 3 klein characters, each with a character sheet (front, back, close-up) --------------------
const FUN_KEYS = ["astronaut", "knight", "trex", "android", "alien", "skeleton", "lizard"] as const;
type FunKey = (typeof FUN_KEYS)[number];
/** [name, line, words the FLUX still was made from, sheet seed] */
const FUN: Record<FunKey, [string, string, string, number]> = {
  astronaut: ["Astronaut", "White spacesuit, gold visor", "a person in a bulky white spacesuit with a lowered gold mirrored visor, white gloves, chunky white boots, an orange-and-blue mission patch on the shoulder and a silver life-support pack on the back", 1360671213],
  knight: ["Knight", "Plate armour, red lion tabard", "a knight in full polished steel plate armour with the visor raised on a bearded face, a red tabard with a gold lion over the breastplate, steel gauntlets and pointed steel sabatons", 784879625],
  trex: ["T-rex", "Inflatable dinosaur costume", "a person wearing an inflatable bright-green T-rex costume, the big friendly dinosaur head above their own, tiny dinosaur arms, an orange spotted belly, and the wearer's own white sneakers at the bottom of the legs", 1706070691],
  android: ["Android", "Chrome android in a navy suit", "a humanoid android with a polished chrome face and chrome hands, softly glowing blue eyes, human proportions, wearing a tailored navy suit, crisp white shirt and black tie, black leather shoes", 1221616152],
  alien: ["Alien", "Green alien, silver flight suit", "a slim humanoid alien with smooth pale-green skin, large glossy black almond-shaped eyes, a normal-sized head, human proportions, wearing a fitted silver flight suit with a high collar and white boots", 483276646],
  skeleton: ["Skeleton", "Skeleton costume, tuxedo", "a person in a full-body skeleton costume, white bones printed on black, a skeleton mask, wearing a black tuxedo jacket with a red bow tie and black shoes", 1397160375],
  lizard: ["Lizard explorer", "Lizard person in khaki", "a humanoid lizard person with green-gold scales and a reptile face with amber eyes, human proportions and no tail, wearing a khaki explorer shirt, cargo trousers, brown boots and a canvas satchel strap across the chest", 1702211739],
};
const FUN_WORDS: Record<FunKey, RegExp> = {
  astronaut: /astronaut|space ?suit|visor/i,
  knight: /knight|armou?r/i,
  trex: /t-?rex|dinosaur/i,
  android: /android|robot|chrome/i,
  alien: /alien/i,
  skeleton: /skeleton|bones/i,
  lizard: /lizard|reptile/i,
};
/** The Fun cast character a description asks for, else the next one in turn. */
function funFor(words: string, turn: number): FunKey {
  return FUN_KEYS.find((k) => FUN_WORDS[k].test(words)) ?? FUN_KEYS[turn % FUN_KEYS.length];
}
/** The real sheet pictures for a Fun cast still or the stunt driver (a character sheet made from it), else null. */
function funSheet(src: string): MockSheetPictures | null {
  if (/\/mock\/cast_driver_(front|sheet)\.jpg$/.test(src)) {
    return { sheet: media("cast_driver_sheet.jpg"), front: media("cast_driver_front.jpg"), back: media("cast_driver_back.jpg"), face: media("cast_driver_face.jpg") };
  }
  const m = /\/mock\/fun_([a-z]+)_\d\.jpg$/.exec(src);
  if (!m || !(FUN_KEYS as readonly string[]).includes(m[1])) return null;
  const k = m[1];
  return { sheet: media(`fun_${k}_sheet.jpg`), front: media(`fun_${k}_front.jpg`), back: media(`fun_${k}_back.jpg`), face: media(`fun_${k}_face.jpg`) };
}

/** The demo's "generated" stills, by what a prompt asks for (frames of real takes and FLUX 3 klein stills). */
const DRIVER_STILLS = ["still_driver_1.jpg", "still_driver_2.jpg", "still_driver_3.jpg", "still_driver_4.jpg"];
const STILLS: Array<[RegExp, string[]]> = [
  [/driver|stunt|salvage|helmet|moustache|hood view|the run\b/i, DRIVER_STILLS],
  [/astronaut|knight|t-?rex|dinosaur|android|robot|alien|skeleton|lizard/i, FUN_KEYS.map((k) => `fun_${k}_1.jpg`)],
  [/castle|loch|snow|winter|dusk|blue hour/i, ["image_loch_1.jpg", "cast_winter_dusk.jpg", "image_loch_2.jpg", "cast_castle_loch.jpg"]],
  [/\bcars?\b|road|convertible|off-?roader|suv|coast/i, ["cast_convertible.jpg", "cast_offroader.jpg", "cast_winter_light.jpg", "offroad_recast_reference.jpg"]],
  [/./, DRIVER_STILLS],
];
function stillsFor(prompt: string): string[] {
  return (STILLS.find(([rx]) => rx.test(prompt)) ?? STILLS[STILLS.length - 1])[1];
}

// --- the mock -----------------------------------------------------------------------------------------------------
interface MockProject {
  id: string;
  name: string;
  created_at: number;
  updated_at: number;
  builtin: boolean;
  archived?: boolean;
  cast: CastMember[];
}

interface RunningJob {
  job: Job;
  req: JobRequest;
  realMs: number;
  elapsedMs: number;
  weights: number[];
  /** Reported seconds per real second (the countdown runs this much faster than the clock). */
  speed: number;
  /** A sheet job whose pictures are being put on the cast member. */
  finishing?: boolean;
}

interface MockSheetPictures {
  sheet: string;
  front: string;
  back: string;
  face: string;
}

export class MockApi implements PickupsApi {
  readonly mode = "mock" as const;
  private scenario = scenario();
  private st: Status = this.scenario === "busy"
    ? { ...makeStatus("ready"), comfy_queue: { running: 1, pending: 0, outside: true } }
    : makeStatus(this.scenario === "fail" || this.scenario === "fresh" || this.scenario === "stall" || this.scenario === "refuse" ? "ready" : this.scenario);
  private shotsById = new Map<string, Shot>();
  private order: string[] = [];
  /** media key per version id, so results know which real take to show. */
  private keys = new Map<string, string>();
  private jobsById = new Map<string, RunningJob>();
  private jobOrder: string[] = [];
  private listeners = new Set<ApiEvents>();
  /** The trash: a take (with where it sat), a whole shot, a project with its shots and images, a cast entry or an image,
   * as the backend's library/trash keeps them. */
  private trashed = new Map<
    string,
    { entry: TrashEntry; shot?: Shot; version?: Version; at?: number; project?: MockProject; shots?: Shot[]; images?: ProjectImage[]; cast?: CastMember; image?: ProjectImage }
  >();
  private timer: number | undefined;
  /** "busy": when the outside ComfyUI job ends (set when the first job queues behind it). */
  private busyUntil: number | undefined;
  private lookFlip = 0;
  /** Uploaded files by id, so a take's settings can show them again. */
  private uploadUrls = new Map<string, string>();
  private projectsById = new Map<string, MockProject>();
  private imagesByProject = new Map<string, ProjectImage[]>();
  private stillIndex = 0;
  /** The people finder's answers per shot and take, as the backend caches them. */
  private peopleCache = new Map<string, PeopleResult>();
  private trackChecked = new Set<string>();
  /** "Describe it" answers per shot, take, target and field, as pickups/describe.py caches them. */
  private describeCache = new Map<string, DescribeResult>();
  /** Recast's character stills (lib/still.ts prompts) and sheets from words are the Fun cast's FLUX 3 klein stills:
   * the character the words ask for, else the next one in turn. */
  private funTurn = 0;
  /** Character sheets being drawn (by job id), and the picture each cast member's sheet was built from (for Redo). */
  private sheetWork = new Map<string, Promise<MockSheetPictures>>();
  private sheetSource = new Map<string, string>();
  private examplesList: Example[] = clone(EXAMPLES);
  /** tour.json's director text (the tour's Enhance answer). */
  private tourDirectorText: string | null = null;
  private settingsState: Settings = {
    comfy_url: "http://127.0.0.1:8188",
    allow_nvidia_rtx: false,
    open_weights_only: true,
    commercial_safe: false,
    director: { backend: "comfy", url: "", model: "", vision_model: "", off_acknowledged: false },
  };

  constructor() {
    const now = Date.now() / 1000;
    const ago = (hours: number) => now - hours * 3600;
    this.projectsById.set("unsorted", { id: "unsorted", name: "My shots", created_at: now - 40 * 86400, updated_at: now - 40 * 86400, builtin: true, cast: [] });
    this.projectsById.set("examples", { id: "examples", name: "Examples", created_at: now - 40 * 86400, updated_at: now - 40 * 86400, builtin: true, cast: [] });
    if (this.scenario === "fresh") return; // first run: no shots yet
    if (TOUR) this.settingsState.director = { backend: "openai", url: "", model: "", vision_model: "", off_acknowledged: false };
    const place = (id: string, name: string, kind: CastKind, line: string, file: string, hours: number): CastMember => ({
      id, name, kind, line, url: media(file), thumb_url: media(file), created_at: ago(hours),
    });

    // Highlands & Alps: castles, a trail and a ride (Remove, Recast with SCAIL-2, Change weather). The tour's weather
    // shot is its castle on the loch.
    this.projectsById.set("prj_highlands", {
      id: "prj_highlands", name: "Highlands & Alps", created_at: now - 9 * 86400, updated_at: 0, builtin: false,
      cast: [
        place("cast_castle_loch", "Castle on the loch", "place", "a stone castle on a tidal island in a sea loch, joined to the shore by an arched stone bridge", "cast_castle_loch.jpg", 200),
        place("cast_winter_dusk", "Winter dusk", "look", "snowy blue hour, warm candle-yellow windows, cold blue light on the snow, large flakes falling", "cast_winter_dusk.jpg", 190),
      ],
    });
    this.imagesByProject.set("prj_highlands", [2, 1].map((i) => this.stillImage(`img_loch_${i}`, `image_loch_${i}.jpg`, 1344, 768, "16:9", 190156912 + i, LOCH_PROMPT, ago(180 - i))));
    this.seedShot("castle_dusk", "prj_highlands", 2);
    const trail = this.seedShot("trail", "prj_highlands", 31);
    this.seedTake(trail, "remove", "trail_remove", 29, { frame: 82, points: [], text: "woman", shadow: false, fill: "the continuing green grass slopes, grey scree and winding hiking trails" }, {
      status: "approved",
      details: { models: ["SAM 3.1"], total_seconds: 714.9, gpu_seconds: 434.8 },
      workflows: [["pick", "Pick · SAM 3.1 · woman"], ["track", "Track · SAM 3.1 · forward from frame 82"], ["track", "Track · SAM 3.1 · backward from frame 82"]],
      notes: [[120, "The trail stays empty to the last frame. Good to grade."]],
    });
    const walk = this.seedShot("castle_walk", "prj_highlands", 52);
    this.seedTake(walk, "recast", "castle_walk_recast", 50, { engine: "scail", frame: 0, points: [], text: "person walking on the path", describe: WALKER }, {
      details: { engine: "scail", performance_size: "832×640", models: ["SAM 3.1", "SCAIL-2 (Wan 2.1)"], total_seconds: 644.2, gpu_seconds: 248.0 },
      workflows: [["pick", "Pick · SAM 3.1 · person walking on the path"], ["track", "Track · SAM 3.1 · forward from frame 0"], ["recast", "Recast · SCAIL-2"], ["recast_new", "Find · SAM 3.1 · person"]],
    });
    this.seedTake(walk, "environment", "castle_walk_environment", 27, { change: "snow", describe: "A woman in a yellow raincoat walks along a gravel path toward a ruined stone castle on a hillside." }, {
      details: {
        look_prompt: "Make it a winter day in heavy snowfall: thick fresh snow covers the ground, paths, roofs, ledges, walls and plants, large snowflakes fall through the air, the sky is a flat overcast white-grey and the light is soft and cold.",
        prompt: "A woman in a yellow raincoat walks along a gravel path toward a ruined stone castle on a hillside. Heavy snow falls steadily over the scene; thick fresh snow covers the ground, roofs and every ledge, and large flakes drift down through soft, cold, overcast light.",
        note: GENERATED_NOTE, total_seconds: 333.5, gpu_seconds: 248.9,
      },
    });
    const ride = this.seedShot("alpine_ride", "prj_highlands", 75);
    this.seedTake(ride, "environment", "alpine_ride_environment", 74, { custom: "Summer ride to autumn", look_prompt: AUTUMN_LOOK, prompt: AUTUMN_PROMPT, control: "depth" }, {
      details: { note: GENERATED_NOTE, total_seconds: 194.4, gpu_seconds: 143.5 },
    });

    // Fun cast: seven FLUX 3 klein characters with character sheets, and two walks across a glen to recast them into.
    // The Lizard explorer has no sheet yet (the "Make a character sheet" state); the rest do.
    this.projectsById.set("prj_funcast", {
      id: "prj_funcast", name: "Fun cast", created_at: now - 2 * 86400, updated_at: 0, builtin: false,
      cast: FUN_KEYS.map((k, i) => this.funCast(k, ago(40 - i), k !== "lizard")),
    });
    // Newest first, as the backend lists them: each character's two takes (seeds 7 and 8), made one after another.
    this.imagesByProject.set("prj_funcast", FUN_KEYS.flatMap((k, i) => [1, 2].map((n) =>
      this.stillImage(`img_${k}_${n}`, `fun_${k}_${n}.jpg`, 832, 1280, "2:3", 6 + n, stillPrompt(FUN[k][2]), ago(46 - i - n / 100)))).reverse());
    const glen = this.seedShot("glen", "prj_funcast", 9);
    const astronaut = { engine: "ltx", frame: 20, points: [], text: "man", cast_id: "cast_astronaut", scene: GLEN_SCENE };
    const glenDetails = (look: string, total: number, gpu: number) => ({
      engine: "ltx", performance_size: "1664×1152", reference: "a 3-view character sheet", made_from: "the first 4.0 s of 4.2 s", grain: "matched to your shot",
      look_prompt: look, prompt: GLEN_PROMPT, total_seconds: total, gpu_seconds: gpu,
    });
    const glenFlows: Array<[string, string]> = [["pick", "Pick · SAM 3.1 · man"], ["track", "Track · SAM 3.1 · forward from frame 20"], ["track", "Track · SAM 3.1 · backward from frame 20"],
      ["recast_paint", "Paint · Qwen-Image 2.1 · 1 reference"], ["recast_ltx", "Recast · LTX-2.5 Union Control · depth"], ["pick", "Pick · SAM 3.1 · person"],
      ["recast_track", "Track · SAM 3.1 · forward"], ["recast_track", "Track · SAM 3.1 · backward"]];
    this.seedTake(glen, "recast", "glen_recast_b", 8, astronaut, { details: glenDetails(GLEN_LOOK_REPLACE, 344.2, 186.5), workflows: glenFlows });
    this.seedTake(glen, "recast", "glen_recast", 5, astronaut, {
      status: "approved", details: glenDetails(GLEN_LOOK_TURN, 357.2, 184.1), workflows: glenFlows,
      notes: [[72, "The visor catches the sky. Keep this one."]],
    });
    // Recast anyone: the castle walk has two people on it (the woman, and a walker down the left-hand path).
    this.shotFromSample({ ...sample("castle_walk"), id: "castle_pair" }, ago(4), "prj_funcast");
    // 10.8 s, longer than Recast's 4 s window, so the window's bracket shows on the timeline; the Alien take starts 4 s in.
    const walker = this.seedShot("glen_long", "prj_funcast", 10);
    this.seedTake(walker, "recast", "glen_long_recast", 6, { engine: "ltx", frame: 145, points: [{ x: 2468, y: 1941, positive: true }, { x: 2518, y: 2274, positive: true }], cast_id: "cast_alien", start_frame: 95 }, {
      details: {
        engine: "ltx", performance_size: "1664×1152", reference: "one figure on a plain backdrop", made_from: "4.0 s from 4.0 s in, of 10.8 s", grain: "matched to your shot",
        look_prompt: "Turn the person walking on the left of <image1> into the character in <image2> (Alien, Green alien, silver flight suit): keep the person's exact pose, mid-stride, with the head, arms and legs exactly where they are, and the same position, size and direction in the picture. Change the face, hair, body and clothes to look exactly like the character in <image2>, from head to toe, lit by the same light as the scene in <image1>, with no rim light and no glow around them. Keep everything else in <image1> exactly as it is.",
        prompt: "Alien, Green alien, silver flight suit walks in from the left and away from the camera. They are lit by the scene's own natural light, with soft, realistic shadows and no glow around them; the camera and the place stay exactly as in the original shot.",
        total_seconds: 481.0, gpu_seconds: 276.7,
      },
      workflows: [["pick", "Pick · SAM 3.1 · frame 145"], ["track", "Track · SAM 3.1 · forward from frame 145"], ["track", "Track · SAM 3.1 · backward from frame 145"],
        ["recast_paint", "Paint · Qwen-Image 2.1 · 1 reference"], ["recast_ltx", "Recast · LTX-2.5 Union Control · depth"], ["pick", "Pick · SAM 3.1 · person"],
        ["recast_track", "Track · SAM 3.1 · forward"], ["recast_track", "Track · SAM 3.1 · backward"]],
    });

    // Coast road: a snowy road that keeps the car as shot and a 4K HDR finish (and, when the build offers car swap, the SUV recast into other cars).
    this.projectsById.set("prj_coast", {
      id: "prj_coast", name: "Coast road", created_at: now - 6 * 86400, updated_at: 0, builtin: false,
      cast: [
        ...(CARS ? [place("cast_offroader", "Off-roader", "prop", OFFROADER, "cast_offroader.jpg", 120),
          place("cast_convertible", "Convertible", "prop", CONVERTIBLE, "cast_convertible.jpg", 119)] : []),
        place("cast_winter_light", "Winter light", "look", "flat grey winter light, wet asphalt, bare trees in mist, cold blue-grey", "cast_winter_light.jpg", 118),
      ],
    });
    const vehicle = (made: string, look: string, prompt: string, total: number, gpu: number) => ({
      engine: "ltx", performance_size: "1920×1024", reference: "a FLUX 3 klein still from your description", made_from: made, grain: "matched to your shot",
      models: ["Qwen-Image 2.1", "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3", "SAM 3.1", "FLUX 3 klein"],
      look_prompt: look, prompt, total_seconds: total, gpu_seconds: gpu,
    });
    const vehicleFlows = (frame: number): Array<[string, string]> => [["pick", "Pick · SAM 3.1 · SUV"], ["track", `Track · SAM 3.1 · forward from frame ${frame}`], ["track", `Track · SAM 3.1 · backward from frame ${frame}`],
      ["cast", "Reference · FLUX 3 klein"], ["recast_paint", "Paint · Qwen-Image 2.1 · 1 reference"], ["recast_ltx", "Recast · LTX-2.5 Union Control · depth"], ["pick", "Pick · SAM 3.1 · car"], ["recast_track", "Track · SAM 3.1 · forward"]];
    if (CARS) {  // car swap's takes: off in this build (lib/features)
      const offroad = this.seedShot("offroad", "prj_coast", 100);
      this.seedTake(offroad, "recast", "offroad_recast", 98, { engine: "ltx", frame: 60, points: [], text: "SUV", describe: OFFROADER, scene: OFFROAD_SCENE }, {
        status: "approved",
        details: vehicle("the first 4.0 s of 5.7 s", vehicleLook(OFFROADER), vehiclePrompt(OFFROADER, OFFROAD_SCENE), 251.2, 190.9),
        workflows: vehicleFlows(60),
        notes: [[48, "The spray off the tyres sells it. Check the roof line in the last second."]],
      });
      const curve = this.seedShot("coast_curve", "prj_coast", 99);
      this.seedTake(curve, "recast", "coast_curve_recast", 96, { engine: "ltx", frame: 30, points: [], text: "SUV", describe: CONVERTIBLE, scene: CURVE_SCENE, start_frame: 2 }, {
        details: vehicle("2.4 s from 0.1 s in, of 2.5 s", vehicleLook(CONVERTIBLE), vehiclePrompt(CONVERTIBLE, CURVE_SCENE), 341.6, 300.5),
        workflows: vehicleFlows(30),
      });
    }
    const misty = this.seedShot("misty_road", "prj_coast", 101);
    const keepFlows: Array<[string, string]> = [["pick", "Pick · SAM 3.1 · car"], ["track", "Track · SAM 3.1 · forward"], ["environment_key", "First frame · Qwen-Image 2.1"], ["environment", "Weather · LTX-2.5 Union Control · depth"]];
    this.seedTake(misty, "environment", "misty_road_environment_b", 97, { change: "snow", keep_text: "the car" }, {
      details: {
        kept_as_shot: "the car: its detail as shot, its light from the render (11.14% of the picture)",
        look_prompt: "Make it a winter day in heavy snowfall: thick fresh snow covers the ground, paths, roofs, ledges, walls and plants, large snowflakes fall through the air, the sky is a flat overcast white-grey and the light is soft and cold.",
        prompt: "Heavy snow falls steadily over the scene; thick fresh snow covers the ground, roofs and every ledge, and large flakes drift down through soft, cold, overcast light. The car stays exactly as it is throughout, the same model, colour and shape.",
        note: KEPT_NOTE, made_from: "the first 4.0 s of 5.2 s", total_seconds: 203.1, gpu_seconds: 157.1,
      },
      workflows: keepFlows,
    });
    this.seedTake(misty, "environment", "misty_road_environment", 70, { change: "snow", control: "depth", keep_text: "the car", describe: MISTY_DESCRIBE }, {
      status: "approved",
      details: {
        look_prompt: "Make it a winter day in heavy snowfall: thick fresh snow covers the ground and settles on everything already in the picture, large snowflakes fall through the air, the sky is a flat overcast white-grey and the light is soft and cold. Add nothing new: no buildings, walls or objects that aren't there.",
        prompt: `${MISTY_DESCRIBE} Heavy snow falls steadily over the scene; thick fresh snow covers the ground and settles on everything in it, and large flakes drift down through soft, cold, overcast light. The car stays exactly as it is throughout, the same model, colour and shape.`,
        note: KEPT_NOTE, made_from: "the first 4.0 s of 5.2 s", total_seconds: 198.0, gpu_seconds: 155.8,
      },
      workflows: keepFlows,
    });
    const coastal = this.seedShot("coastal_drive", "prj_coast", 120);
    this.seedTake(coastal, "finish", "coastal_drive_finish", 118, { upscale_4k: true, hdr: true, quality: "best" }, {
      status: "approved",
      details: { total_seconds: 1509.1, gpu_seconds: 1261.3 },
      workflows: [["finish4k", "4K · SeedVR2"], ["finish_hdr", "HDR · LTX SDR-to-HDR"]],
    });

    // Stunt previz: Blender playblasts rendered into finished shots, with the stunt driver's character sheet and the set.
    this.projectsById.set("prj_previz", {
      id: "prj_previz", name: DEFAULT_TOUR_COPY.project, created_at: now - 8 * 86400, updated_at: 0, builtin: false,
      cast: [
        {
          ...place("cast_driver", "Stunt driver", "person", "an old stunt driver with a weathered face, a white walrus moustache and a cream open-face helmet with navy stars, in a sage-green work shirt", "cast_driver_front.jpg", 190),
          url: media("cast_driver_sheet.jpg"),
          sheet: {
            url: media("cast_driver_sheet.jpg"), thumb_url: media("cast_driver_sheet.jpg"),
            views: { front: media("cast_driver_front.jpg"), back: media("cast_driver_back.jpg"), face: media("cast_driver_face.jpg") },
            editor: "Qwen-Image 2.1", models: ["FLUX 3 klein", "Qwen-Image 2.1"], licences: licencesFor(["FLUX 3 klein", "Qwen-Image 2.1"]),
            seed: 914031, made_from: "words", made_front: false, created_at: ago(190),
          },
        },
        place("cast_hood_view", "Hood view", "place", "the stripped cabin of the grey stunt car seen through its dusty windshield from the hood, with racks of wrecked cars behind", "cast_hood_view.jpg", 189),
        place("cast_the_run", "The run", "place", "a long sandy lane between towering walls of stacked wrecked cars, the low sun and a hazy skyline at its end", "cast_the_run.jpg", 188),
        place("cast_video_village", "Video village", "place", "the video village under a blue pop-up canopy with a firing console, monitors and director's chairs in a salvage yard", "cast_video_village.jpg", 187),
      ],
    });
    const driverRefs = [{ kind: "cast", id: "cast_driver" }, { kind: "cast", id: "cast_hood_view" }];
    const previzFlows = (refs: number): Array<[string, string]> => [["look", `Look frame · Qwen-Image 2.1 · ${refs} reference${refs === 1 ? "" : "s"}`], ["previz", "Render · LTX-2.5 Layout-To-Render"]];
    const hood = this.seedShot("hood_push", "prj_previz", 168);
    this.seedTake(hood, "previz", "hood_push_previz_a", 150, { look_refs: driverRefs, look_prompt: HOOD_LOOK, prompt: HOOD_PROMPT, pin_start: true, seed: 42 }, {
      look: "look_frame.jpg", details: { references: 2, look_prompt: HOOD_LOOK, prompt: HOOD_PROMPT, note: PREVIZ_NOTE, total_seconds: 222.3, gpu_seconds: 200.7 }, workflows: previzFlows(2),
    });
    this.seedTake(hood, "previz", "hood_push_previz_b", 146, { look_refs: driverRefs, look_prompt: HOOD_LOOK, prompt: HOOD_PROMPT, pin_start: false, seed: 42 }, {
      status: "approved", look: "look_frame_2.jpg",
      details: { references: 2, look_prompt: HOOD_LOOK, prompt: HOOD_PROMPT, note: PREVIZ_NOTE, total_seconds: 250.4, gpu_seconds: 228.4 }, workflows: previzFlows(2),
      notes: [[60, "The knuckle along the moustache is the moment. Hold it in the edit."]],
    });
    const profile = this.seedShot("driver_profile", "prj_previz", 167);
    this.seedTake(profile, "previz", "driver_profile_previz", 148, { look_refs: [driverRefs[0]], look_prompt: PROFILE_LOOK, prompt: PROFILE_PROMPT, pin_start: true, seed: 42 }, {
      status: "approved", details: { references: 1, look_prompt: PROFILE_LOOK, prompt: PROFILE_PROMPT, note: PREVIZ_NOTE, total_seconds: 220.1, gpu_seconds: 204.4 }, workflows: previzFlows(1),
    });
    const launch = this.seedShot("launch", "prj_previz", 166);
    this.seedTake(launch, "previz", "launch_previz", 145, { look_refs: [{ kind: "cast", id: "cast_the_run" }], look_prompt: LAUNCH_LOOK, prompt: LAUNCH_PROMPT, pin_start: true, seed: 42 }, {
      status: "approved", details: { references: 2, look_prompt: LAUNCH_LOOK, prompt: LAUNCH_PROMPT, note: PREVIZ_NOTE, total_seconds: 278.0, gpu_seconds: 258.5 }, workflows: previzFlows(2),
    });

    // Most recent first, by each shot's latest take; each project last changed with its newest shot or take.
    const last = (s: Shot) => Math.max(...s.versions.map((v) => Number(v.created_at) || 0), Number(s.created_at) || 0);
    this.order = Array.from(this.shotsById.values()).sort((a, b) => last(b) - last(a)).map((s) => s.id);
    for (const p of this.projectsById.values()) {
      const times = Array.from(this.shotsById.values()).filter((s) => s.project_id === p.id).map(last);
      p.updated_at = times.length ? Math.max(...times) : p.created_at;
    }
  }

  /** A shot of the demo library, made `hours` ago. */
  private seedShot(sampleId: string, projectId: string, hours: number): Shot {
    return this.shotFromSample(sample(sampleId), Date.now() / 1000 - hours * 3600, projectId);
  }

  /** A take the demo library starts with: the real take's media, receipt, settings and what it reported. */
  private seedTake(shot: Shot, task: TaskId, name: string, hours: number, params: Record<string, unknown>, seed: TakeSeed): Version {
    const v = this.addVersion(shot, task, this.keyOf(shot), params as JobRequest["params"], Date.now() / 1000 - hours * 3600, null, { ...seed, media: name });
    if (seed.status) v.status = seed.status;
    v.notes = (seed.notes ?? []).map(([frame, text], i) => ({ id: `note_${shot.id}_${i}`, frame, text, created_at: Number(v.created_at) + 1800 }));
    this.derive(shot);
    return v;
  }

  /** A Fun cast character: its FLUX 3 klein still, and (`sheet`) the character sheet built from it. */
  private funCast(k: FunKey, at: number, sheet: boolean): CastMember {
    const [name, line, , seed] = FUN[k];
    const still = media(`fun_${k}_1.jpg`);
    const c: CastMember = { id: `cast_${k}`, name, kind: "person", line, url: still, thumb_url: still, created_at: at };
    this.sheetSource.set(c.id, still);
    if (!sheet) return c;
    const pics = funSheet(still)!;
    const models = ["Qwen-Image 2.1"];
    return {
      ...c, url: pics.sheet, thumb_url: pics.front,
      sheet: { url: pics.sheet, thumb_url: pics.sheet, views: { front: pics.front, back: pics.back, face: pics.face }, editor: "Qwen-Image 2.1", models, licences: licencesFor(models), seed, made_from: "cast", made_front: false, created_at: at + 3600 },
    };
  }

  /** A still on a project's Images page, made by FLUX 3 klein. */
  private stillImage(id: string, file: string, width: number, height: number, aspect: string, seed: number, prompt: string, at: number): ProjectImage {
    return {
      id, url: media(file), thumb_url: media(file), width, height, model: "flux3-klein", model_name: "FLUX 3 klein", mode: "generate",
      prompt, prompt_sent: prompt, seed, aspect, resolution: "1K", refs: [], created_at: at, job_id: `job_${id}`,
    };
  }

  /** A Fun cast still for a description: the character it names, else the next in turn; each ask alternates its two takes. */
  private funStill(words: string): string {
    const turn = this.funTurn++;
    return media(`fun_${funFor(words, Math.floor(turn / 2))}_${(turn % 2) + 1}.jpg`);
  }

  /** The tour's words (/examples/tour.json, on a machine that has one) rename its project, cast and shots, so they match
   * what the autopilot types. Only in the tour: the demo itself always shows the built-in names that go with its media. */
  async init(): Promise<void> {
    if (!TOUR) return;
    const copy = await loadTourCopy();
    if (copy) this.applyTourCopy(copy);
  }

  /** /examples/tour.json: the demo project, cast and shot names the tour types on this machine. */
  private applyTourCopy(copy: TourCopy) {
    const p = this.projectsById.get("prj_previz");
    if (!p) return;
    const fallback = [media("cast_driver_front.jpg"), media("cast_hood_view.jpg"), media("cast_the_run.jpg")];
    p.name = copy.project;
    p.cast = copy.cast.map((c, i) => {
      const img = c.image ? `/examples/${c.image}` : fallback[i % fallback.length];
      return { id: `cast_tour_${i + 1}`, name: c.name, kind: c.kind, line: c.line, url: img, thumb_url: img, created_at: p.created_at + 3600 };
    });
    const previz = this.shotsById.get("shot_hood_push");
    if (previz) previz.name = copy.previz_shot;
    const weather = this.shotsById.get("shot_castle_dusk");
    if (weather) weather.name = copy.weather_shot;
    this.tourDirectorText = copy.director_text ?? null;
  }

  /** Counts and the newest thumbnail, the way the backend reports them. */
  private derive(shot: Shot) {
    const visible = shot.versions.filter((v) => !v.hidden);
    shot.latest_thumb_url = (visible[visible.length - 1] ?? shot.versions[0]).thumb_url;
    shot.approved = shot.versions.filter((v) => v.status === "approved").length;
  }

  private projectOut(p: MockProject): Project {
    const shots = this.order.map((id) => this.shotsById.get(id)!).filter((x) => x && x.project_id === p.id && listed(x));
    return {
      id: p.id,
      name: p.name,
      created_at: p.created_at,
      updated_at: p.updated_at,
      archived: !!p.archived,
      builtin: p.builtin,
      shots: shots.length,
      takes: shots.reduce((a, x) => a + x.versions.length - 1, 0),
      approved: shots.reduce((a, x) => a + (x.approved ?? 0), 0),
      cover_url: shots[0]?.latest_thumb_url ?? null,
      cover_video_url: shots[0] ? latestVideo(shots[0]) : null,
      cast: p.cast.length,
    };
  }

  private getProject(id: string): MockProject {
    const p = this.projectsById.get(id);
    if (!p) throw new ApiError("That project isn't here.", 404);
    return p;
  }

  // -- helpers --
  private emit(fn: (h: ApiEvents) => void) {
    setTimeout(() => this.listeners.forEach((h) => fn(h)), 0);
  }
  private touch(id: string) {
    this.order = [id, ...this.order.filter((x) => x !== id)];
  }
  private getShot(id: string): Shot {
    const s = this.shotsById.get(id);
    if (!s) throw new ApiError("That shot isn't in the library.", 404);
    return s;
  }
  private keyOf(shot: Shot, versionId?: string | null): string {
    return this.keys.get(`${shot.id}:${versionId || "orig"}`) ?? "upload";
  }

  private shotFromSample(s: SampleDef, createdAt: number, projectId = "examples"): Shot {
    const id = `shot_${s.id}`;
    const orig: Version = {
      id: "orig",
      label: "Original",
      task: null,
      created_at: createdAt,
      video_url: media(`${s.key}.mp4`),
      thumb_url: media(`${s.key}.jpg`),
      downloads: [],
      details: {},
      status: "",
      hidden: false,
      notes: [],
      input_version: null,
      params: {},
    };
    const shot: Shot = {
      id,
      project_id: projectId,
      created_at: createdAt,
      name: s.shotName,
      width: s.width,
      height: s.height,
      fps: s.fps,
      frames: s.frames,
      duration_s: +(s.frames / s.fps).toFixed(3),
      thumb_url: orig.thumb_url,
      video_url: orig.video_url,
      versions: [orig],
    };
    this.shotsById.set(id, shot);
    this.keys.set(`${id}:orig`, s.key);
    if (!this.order.includes(id)) this.order.unshift(id);
    this.derive(shot);
    this.bumpProject(projectId);
    return shot;
  }

  private bumpProject(id: string | undefined) {
    const p = id ? this.projectsById.get(id) : null;
    if (p) p.updated_at = Date.now() / 1000;
  }

  /** A take on `shot`: the real take its media key has for `task` (`<key>_<task>`, or `seed.media`), else the shot as it
   * is (a clip of your own), with what the backend would report for it. `seed`: a take the demo starts with. */
  private addVersion(shot: Shot, task: TaskId, baseKey: string, params: JobRequest["params"], createdAt: number, baseVersionId?: string | null, seed?: TakeSeed): Version {
    const base = shot.versions.find((v) => v.id === (baseVersionId || "orig")) ?? shot.versions[0];
    const candidate = seed?.media ?? `${baseKey}_${task}`;
    const known = KNOWN_VIDEOS.has(candidate);
    const key = known ? candidate : baseKey;
    const video = known ? media(`${candidate}.mp4`) : base.video_url;
    const thumb = known ? media(`${candidate}.jpg`) : base.thumb_url;
    const fin = task === "finish" ? (params as FinishParams) : null;
    const total = STAGES[task].reduce((a, [, s]) => a + s, 0) * (0.85 + Math.random() * 0.25);
    const MB = 1024 * 1024;
    // The size each task renders at, as the backend does: weather at the shot's aspect within 1920x1088 (multiples of
    // 128), previz at 1920x1088, Finish 4K at UHD; the rest at the shot's own size.
    const fit = Math.sqrt((1920 * 1088) / (shot.width * shot.height));
    const [rw, rh] =
      task === "environment" ? [Math.max(128, Math.floor((shot.width * fit) / 128) * 128), Math.max(128, Math.floor((shot.height * fit) / 128) * 128)]
      : task === "previz" ? [1920, 1088]
      : fin?.upscale_4k ? [3840, 2160]
      : [shot.width, shot.height];
    const uhd = Math.max(rw, rh) >= 3000;
    const sizes: Array<"UHD" | "HD"> = uhd ? ["UHD", "HD"] : ["HD"];
    // As the backend's deliverables(): ProRes and H.264 at each size, H.265 at the top size, and the native master
    // when the render size isn't a delivery size.
    const dims = { UHD: "3840×2160", HD: "1920×1080" };
    const k = (size: "UHD" | "HD") => (size === "UHD" ? 4 : 1);
    const downloads: Download[] = [
      ...sizes.map((size): Download => ({ label: `ProRes 422 HQ · ${size} ${dims[size]}`, url: video, codec: "prores", size, bytes: Math.round(352 * MB * k(size)) })),
      ...sizes.map((size): Download => ({ label: `MP4 · H.264 · ${size} ${dims[size]}`, url: video, codec: "h264", size, bytes: Math.round(38 * MB * k(size)) })),
      { label: `MP4 · H.265 (HEVC) 10-bit · ${sizes[0]} ${dims[sizes[0]]}`, url: video, codec: "h265", size: sizes[0], bytes: Math.round(19 * MB * k(sizes[0])) },
    ];
    if (!sizes.some((size) => dims[size] === `${rw}×${rh}`)) {
      downloads.push({ label: `ProRes 422 HQ · native ${rw}×${rh}`, url: video, codec: "prores", size: "native", bytes: Math.round(352 * MB * ((rw * rh) / (1920 * 1080))) });
    }
    if (fin?.hdr) {
      downloads.push(
        { label: "HDR master (HLG)", url: video, codec: "hlg", size: uhd ? "UHD" : "HD", bytes: Math.round(96 * MB * (uhd ? 4 : 1)) },
        { label: "EXR sequence (ACEScg)", url: video, codec: "exr", size: uhd ? "UHD" : "HD", bytes: Math.round(2.9 * 1024 * MB * (uhd ? 4 : 1)) },
      );
    }
    let label: string = { remove: "Remove a person", recast: "Replace a person", environment: "Weather", previz: "Previz to final", relight: "Relight", reshoot: "Reshoot", finish: "Finish" }[task];
    const env = task === "environment" ? (params as EnvironmentParams) : null;
    const weather = env ? (env.change ? WEATHER_LABEL[env.change] ?? env.change : (env.custom ?? "Custom").slice(0, 40)) : null;
    if (env) label = `Weather · ${weather}`;
    if (task === "relight") {
      const rl = params as RelightParams;
      const looks: Record<string, string> = { sunset: "Sunset backlight", golden: "Golden side light", cool: "Cool low sun", midday: "Midday sun", afternoon: "Afternoon light" };
      label = `Relight · ${rl.look === "custom" ? (rl.custom ?? "Custom").slice(0, 40) : looks[rl.look] ?? "Sunset backlight"}`;
    }
    // As tasks/finish.py: "Finish 4K HDR", "Finish 1080p HDR", "Finish 4K".
    if (fin) label = `Finish ${[fin.upscale_4k ? "4K" : fin.hdr ? "1080p" : "", fin.hdr && "HDR"].filter(Boolean).join(" ")}`.trim();
    const editor = this.settingsState.commercial_safe ? "Qwen-Image-Edit 2511" : "Qwen-Image 2.1";
    const upscaler = fin?.quality !== "best" && this.settingsState.allow_nvidia_rtx ? "NVIDIA RTX Video Super Resolution" : "SeedVR2 3B";
    const details: Record<string, unknown> = {
      total_seconds: Math.round(total),
      gpu_seconds: Math.round(total * 0.93),
      gpu: "RTX 4090",
      ...(fin ? { input: `${shot.width}×${shot.height}` } : {}),
      ...sizeDetails(rw, rh, shot.frames, shot.fps),
      // What the real take was (its length, render size, how a Remove was done).
      ...(known ? RESULT_FACTS[candidate] : {}),
      models: fin
        ? MODELS.finish.map((m) => (m === "SeedVR2 3B" ? upscaler : m)).filter((m) => (fin.upscale_4k || m.startsWith("LTX")) && (fin.hdr || !m.startsWith("LTX")))
        : MODELS[task].map((m) => (m === "Qwen-Image 2.1" ? editor : m)),
    };
    if (task === "remove") details.camera = "locked-off";
    if (env) {
      details.change = env.change ?? "custom";
      details.control = env.control || (env.change ? WEATHER_CONTROL[env.change] : "depth");
      if (details.control === "canny") details.models = [editor, "LTX-2.5 + Union Control IC-LoRA", "Canny"];
      if (env.look_prompt) details.look_prompt = env.look_prompt;
      if (env.prompt) details.prompt = env.prompt;
      details.seed = env.seed ?? 42;
      details.grain = env.grain ?? "fine";
      // As tasks/environment.py: a kept thing is put back as shot (its detail, the render's light) and counted as original.
      if (keepOf(env)) {
        details.models = [...(details.models as string[]), "SAM 3.1"];
        details.kept_as_shot = `${keepOf(env)}: its detail as shot, its light from the render (${KEPT_SHARE}% of the picture)`;
      }
    }
    if (task === "previz") {
      const pv = params as PrevizParams;
      details.references = pv.look_upload_ids?.length ?? (pv.look_upload_id ? 1 : 0);
      details.pinned_first_frame = !!pv.pin_start;
      details.grain = pv.grain ?? "fine";
      // As tasks/previz.py: "loose" renders with Union Control from the playblast's depth instead of Layout-To-Render.
      details.follow = pv.follow === "loose" ? "loosely" : "exactly";
      details.look_prompt = pv.look_prompt || pv.describe || "Turn image 1, a 3D previz frame, into a photoreal live-action film still.";
      details.prompt = pv.prompt || pv.describe || "A photoreal cinematic live-action shot.";
    }
    if (task === "reshoot") details.move = (params as ReshootParams).move?.replace("_", " ");
    // Recast with a person made from words, as the backend: a FLUX 3 klein still kept with the take (reference_url);
    // reference_take uses that take's still again.
    let referenceUrl: string | undefined;
    if (task === "recast") {
      const rc = params as RecastParams;
      details.engine = rc.engine === "scail" ? "scail" : "ltx";
      if (rc.engine === "scail") details.models = ["SAM 3.1", "SCAIL-2 (Wan 2.1)"];
      // Recast anyone, as the backend's take details: who was swapped, and who they became.
      if (rc.swaps && rc.swaps.length > 1) {
        const cast = this.projectsById.get(shot.project_id ?? "")?.cast ?? [];
        details.swaps = rc.swaps.map((w) => ({
          who: w.label || "someone",
          character: w.cast_id ? cast.find((c) => c.id === w.cast_id)?.name ?? "someone from your cast" : w.describe?.trim() || (w.who_ref || w.ref_upload_id ? "a picture" : "a new character"),
        }));
      }
      const again = rc.reference_take ? shot.versions.find((x) => x.id === rc.reference_take)?.reference_url : undefined;
      const words = (rc.describe ?? "").trim() && !/(^|\s)@\S/.test(rc.describe ?? "");
      if (!rc.cast_id && !rc.who_ref && !rc.ref_upload_id && (again || words || (rc.reference_prompt ?? "").trim())) {
        // The still FLUX 3 klein made from the words: the real take's own, else the Fun cast still the words ask for.
        referenceUrl = again ?? (WITH_REFERENCE.has(candidate) ? media(`${candidate}_reference.jpg`) : media(`fun_${funFor(`${rc.describe ?? ""} ${rc.reference_prompt ?? ""}`, this.funTurn++)}_1.jpg`));
        details.models = [...(details.models as string[]), "FLUX 3 klein"];
      }
    }
    if (fin) details.quality = fin.quality === "best" ? "Best" : "Fast preview";
    // As tasks/finish.py (receipts.finish_receipt): an upscale is its own share, the footage resampled and never AI;
    // the HDR highlights LTX rebuilt are the only generated pixels.
    const highlights = fin?.hdr ? RESULT_HIGHLIGHTS[candidate] ?? SHARES.finish.ai : 0;
    const upscaled = !!fin?.upscale_4k && 3840 / Math.max(shot.width, shot.height) > 1.05;
    const own: Receipt = !fin
      ? RESULT_SHARES[candidate] ??
        (env && keepOf(env) ? { original: KEPT_SHARE, borrowed: 0, ai: +(100 - KEPT_SHARE).toFixed(2) } : SHARES[task])
      : upscaled
        ? { original: 0, borrowed: 0, upscaled: +(100 - highlights).toFixed(2), ai: highlights, upscaler, scale: +(rw / shot.width).toFixed(2) }
        : { original: +(100 - highlights).toFixed(2), borrowed: 0, ai: highlights };
    // As Store.shot(): a Finish made from a take is composed with that take's receipt (receipts.compose), so its shares
    // are of the shot's original footage, and its HDR detail says the same as its receipt (receipts.hdr_detail).
    const fromTake = base.id !== "orig";
    const numOf = (x: Version) => pad2(shot.versions.indexOf(x) + 1);
    const receipt: Receipt = fin && fromTake ? composeReceipt(own, base.receipt, numOf(base), base.label) : own;
    if (fin?.hdr) details.hdr = hdrDetail(receipt);
    // A take the demo starts with says what the real one reported (its models, prompts and times).
    if (seed?.details) Object.assign(details, seed.details);
    // ... and every take made from a take carries the models of its input chain, "via Take NN".
    const licences = licencesFor(details.models);
    const named = new Set(licences.map((m) => m.name));
    for (let id = base.id, seen = new Set<string>(); id !== "orig" && !seen.has(id); ) {
      seen.add(id);
      const x = shot.versions.find((y) => y.id === id);
      if (!x) break;
      for (const m of x.licences ?? []) {
        if (m.via || named.has(m.name)) continue;
        named.add(m.name);
        licences.push({ ...m, via: `Take ${numOf(x)}` });
      }
      id = x.input_version || "orig";
    }
    const takeNo = String(shot.versions.length + 1).padStart(2, "0");
    const stem = `${shot.name.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_|_$/g, "")}_T${takeNo}_${task}`;
    const CODEC: Record<string, [string, string]> = { prores: ["ProRes422HQ", "mov"], h264: ["H264", "mp4"], h265: ["H265", "mp4"], hlg: ["HLG", "mov"], exr: ["EXR_ACEScg", "zip"] };
    for (const d of downloads) {
      const [c, ext] = CODEC[d.codec ?? "prores"];
      d.filename = `${stem}_${c}${d.size && d.size !== "native" ? `_${d.size}` : ""}.${ext}`;
    }
    const clicked = String((params as { frame?: number }).frame ?? 0);
    // As the backend: no backward track when the click is on the first frame.
    const steps = fin
      ? WORKFLOW_STEPS.finish.filter((s) => (s.step === "upscale" ? fin.upscale_4k : fin.hdr))
      : WORKFLOW_STEPS[task].filter((s) => clicked !== "0" || !s.title.includes("backward"));
    const ran = seed?.workflows?.map(([step, title]) => ({ step, title })) ?? (steps.length ? steps : [{ step: "finish", title: "Finish" }]);
    const workflows: TakeWorkflow[] = ran.map(({ step, title }, i) => ({
      step,
      // Named after the graph that ran, as the backend does: the frame editor is whichever one Setup picked.
      title: title.replace("Qwen-Image 2.1", editor).replace("{frame}", clicked).replace("{structure}", details.control === "canny" ? "edges" : "depth"),
      url: workflowUrl(step, shot.name),
      filename: `${stem}_${String(i + 1).padStart(2, "0")}_${step}_comfyui.json`,
    }));
    // As the backend: Remove and Recast write the receipt on the picture (receipt.mp4), and so does a Change weather that
    // keeps something as shot. The real take's own; on a clip of your own, one drawn for the demo.
    const receiptUrl = WITH_RECEIPT.has(candidate) && (task !== "environment" || (env && keepOf(env)))
      ? media(`${candidate}_receipt.mp4`)
      : !known && GENERIC_RECEIPT[task] ? media(GENERIC_RECEIPT[task]!) : undefined;
    const v: Version = {
      id: uid("v"),
      label,
      task,
      created_at: createdAt,
      video_url: video,
      thumb_url: thumb,
      ...(receiptUrl ? { receipt_url: receiptUrl } : {}),
      receipt,
      downloads,
      workflows,
      details,
      licences,
      status: "",
      hidden: false,
      notes: [],
      input_version: base.id,
      params: JSON.parse(JSON.stringify(params)),
    };
    if (base.id !== "orig") v.before_url = base.video_url;
    else if (WITH_BEFORE.has(candidate)) v.before_url = media(`${candidate}_before.mp4`); // rendered at another size or from later in the shot
    if (referenceUrl) v.reference_url = referenceUrl;
    if (seed?.look) v.look_url = media(seed.look);
    shot.versions.push(v);
    this.keys.set(`${shot.id}:${v.id}`, key);
    this.derive(shot);
    this.bumpProject(shot.project_id);
    return v;
  }

  // -- API: status --
  async status(): Promise<Status> {
    await sleep(250);
    return clone(this.st);
  }

  async connect(comfy_url: string): Promise<Status> {
    await sleep(1100);
    if (!/^https?:\/\/[^\s/]+/i.test(comfy_url.trim())) {
      this.st = makeStatus("offline", comfy_url);
    } else {
      this.st = makeStatus(this.scenario === "setup" ? "setup" : "ready", comfy_url.trim().replace(/\/$/, ""));
    }
    const s = clone(this.st);
    this.emit((h) => h.onStatus(clone(s)));
    return s;
  }

  async downloadModels(): Promise<Status | null> {
    await sleep(300);
    const url = this.st.comfy_url;
    setTimeout(() => {
      this.st = makeStatus("ready", url);
      const s = clone(this.st);
      this.emit((h) => h.onStatus(s));
    }, 4000);
    return clone(this.st);
  }

  // -- API: shots --
  async samples(): Promise<Sample[]> {
    await sleep(300);
    return SAMPLES.map(({ id, name, thumb_url }) => ({ id, name, thumb_url }));
  }

  private packInstalled = this.scenario !== "fresh";

  async examples(): Promise<Example[]> {
    await sleep(250);
    if (!this.packInstalled) return [];
    const cat = mockCatalog(false);
    return clone(this.examplesList).map((e) => ({ ...e, models: Object.values(cat).find((w) => w.task === e.task)?.models ?? [] }));
  }

  async defaultModels(): Promise<ModelCatalog> {
    await sleep(120);
    return mockCatalog(!!this.settingsState.commercial_safe);
  }

  async examplesPack(): Promise<ExamplesPack> {
    await sleep(150);
    return { available: true, installed: this.packInstalled, size_mb: 40 };
  }

  async importExamplesPack(): Promise<{ examples: number; samples: number }> {
    await sleep(1200);
    this.packInstalled = true;
    return { examples: this.examplesList.length, samples: 0 };
  }

  async installExamplesPack(): Promise<{ examples: number }> {
    await sleep(TOUR ? 1500 : 3500);
    this.packInstalled = true;
    return { examples: this.examplesList.length };
  }

  async openExample(id: string): Promise<ExampleOpenResult> {
    const ex = this.examplesList.find((e) => e.id === id);
    if (!ex) throw new ApiError("That example isn't available.", 404);
    // Examples that need their project (previz with its cast) open the real shot, with the take's settings.
    if (ex.shot_id && this.shotsById.has(ex.shot_id)) {
      await sleep(400);
      this.touch(ex.shot_id);
      return { shot: clone(this.getShot(ex.shot_id)), task: ex.task, params: ex.params ?? {} };
    }
    const shot = await this.openSample(ex.sample_id);
    return { shot, task: ex.task, params: ex.params ?? {} };
  }

  async shotFromPath(path: string, projectId?: string): Promise<Shot> {
    await sleep(600);
    const p = path.trim().replace(/^["']|["']$/g, "");
    if (!p) throw new ApiError("Paste the full path to a video file.", 400);
    if (!/\.(mov|mp4|m4v|mkv|mxf|avi|webm)$/i.test(p)) throw new ApiError("That's not a video Pickups can read.", 415);
    // Demo: any path opens the glen walk-in under that file's name.
    const def = sample("glen");
    const shot = this.shotFromSample({ ...def, id: uid("path"), shotName: p.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, "") }, Date.now() / 1000, projectId || "unsorted");
    this.touch(shot.id);
    return clone(shot);
  }

  async openSample(id: string): Promise<Shot> {
    await sleep(500);
    const def = SAMPLES.find((s) => s.id === id);
    if (!def) throw new ApiError("That sample isn't available.", 404);
    // As the backend: "Try it" copies the clip into My shots as a new shot.
    const shot = this.shotFromSample({ ...def, id: uid(def.id) }, Date.now() / 1000, "unsorted");
    this.touch(shot.id);
    return clone(shot);
  }

  async uploadShot(file: File, onProgress?: (f: number) => void, projectId?: string): Promise<Shot> {
    const isVideo = file.type.startsWith("video/") || /\.(mov|mp4|m4v|mkv|webm|avi)$/i.test(file.name);
    if (!isVideo) throw new ApiError("That's not a video Pickups can read.", 415);
    for (let i = 1; i <= 10; i++) {
      await sleep(70);
      onProgress?.(i / 10);
    }
    const url = URL.createObjectURL(file);
    const probe = await probeVideo(url).catch(() => null);
    if (!probe) {
      URL.revokeObjectURL(url);
      throw new ApiError("This browser can't play that file in demo mode. Try an MP4.", 415);
    }
    const id = uid("shot");
    const fps = 24;
    const orig: Version = {
      id: "orig",
      label: "Original",
      task: null,
      created_at: Date.now() / 1000,
      video_url: url,
      thumb_url: probe.thumb,
      downloads: [],
      details: {},
      status: "",
      hidden: false,
      notes: [],
      input_version: null,
      params: {},
    };
    const shot: Shot = {
      id,
      project_id: projectId || "unsorted",
      created_at: Date.now() / 1000,
      name: file.name.replace(/\.[^.]+$/, ""),
      width: probe.width,
      height: probe.height,
      fps,
      frames: Math.max(1, Math.round(probe.duration * fps)),
      duration_s: +probe.duration.toFixed(3),
      thumb_url: probe.thumb,
      video_url: url,
      versions: [orig],
    };
    this.shotsById.set(id, shot);
    this.keys.set(`${id}:orig`, "upload");
    this.touch(id);
    this.derive(shot);
    this.bumpProject(shot.project_id);
    return clone(shot);
  }

  async shots(projectId?: string): Promise<Shot[]> {
    await sleep(350);
    return this.order
      .map((id) => this.getShot(id))
      .filter((x) => (!projectId || x.project_id === projectId) && listed(x))
      .map((x) => clone(x));
  }

  async patchShot(id: string, patch: { name?: string; project_id?: string }): Promise<Shot> {
    await sleep(150);
    const shot = this.getShot(id);
    if (patch.name?.trim()) shot.name = patch.name.trim();
    if (patch.project_id) {
      this.getProject(patch.project_id);
      const from = shot.project_id;
      shot.project_id = patch.project_id;
      this.bumpProject(from);
      this.bumpProject(patch.project_id);
    }
    return clone(shot);
  }

  // -- API: projects --
  async projects(): Promise<Project[]> {
    await sleep(200);
    // As the backend: the built-in Examples project only lists once it holds something (old libraries).
    const list = Array.from(this.projectsById.values())
      .map((p) => this.projectOut(p))
      .filter((p) => p.id !== "examples" || p.shots > 0 || p.cast > 0);
    const rank = (p: Project) => (p.id === "examples" ? 2 : 0);
    return list.sort((a, b) => rank(a) - rank(b) || Number(b.updated_at) - Number(a.updated_at));
  }

  async createProject(name: string): Promise<Project> {
    await sleep(200);
    const n = name.trim();
    if (!n) throw new ApiError("Give the project a name.", 400);
    const now = Date.now() / 1000;
    const p: MockProject = { id: uid("prj"), name: n, created_at: now, updated_at: now, builtin: false, cast: [] };
    this.projectsById.set(p.id, p);
    return this.projectOut(p);
  }

  async renameProject(id: string, name: string): Promise<Project> {
    await sleep(150);
    const p = this.getProject(id);
    if (name.trim()) p.name = name.trim();
    return this.projectOut(p);
  }

  async archiveProject(id: string, archived: boolean): Promise<Project> {
    await sleep(150);
    const p = this.getProject(id);
    if (!p.builtin) p.archived = archived;
    return this.projectOut(p);
  }

  private busyIn(projectId: string, uses?: string): boolean {
    return this.jobOrder.some((id) => {
      const j = this.jobsById.get(id)!;
      const pid = j.job.shot_id ? this.shotsById.get(j.job.shot_id)?.project_id : j.job.project_id;
      return pid === projectId && (j.job.status === "queued" || j.job.status === "running") && (!uses || JSON.stringify(j.req).includes(uses));
    });
  }

  async trashProject(id: string): Promise<{ entry: TrashEntry; projects: Project[] }> {
    await sleep(150);
    const p = this.getProject(id);
    if (id === "examples") throw new ApiError("Examples can't be moved to the trash.", 403);
    if (this.busyIn(id)) throw new ApiError("A job is working in this project. Cancel it or wait for it to finish.", 409);
    const shots = this.order.map((s) => this.shotsById.get(s)!).filter((s) => s.project_id === id);
    for (const s of shots) this.shotsById.delete(s.id);
    this.order = this.order.filter((s) => !shots.some((x) => x.id === s));
    const images = this.imagesByProject.get(id) ?? [];
    this.imagesByProject.delete(id);
    this.projectsById.delete(id);
    if (id === "unsorted") {
      const now = Date.now() / 1000;
      this.projectsById.set("unsorted", { id: "unsorted", name: "My shots", created_at: now, updated_at: now, builtin: true, cast: [] });
    }
    const entry: TrashEntry = { id: uid("tr"), kind: "project", project_id: id, name: p.name, shots: shots.length, trashed_at: Date.now() / 1000,
      bytes: shots.reduce((n, s) => n + 2_600_000_000 + 1_240_000_000 * (s.versions.length - 1), 0), remote_files: 0, thumb_url: shots[0]?.thumb_url };
    this.trashed.set(entry.id, { entry, project: p, shots, images });
    for (const s of shots) for (const h of this.listeners) h.onShotRemoved?.({ id: s.id, project_id: id });
    return { entry: clone(entry), projects: await this.projects() };
  }

  async trashAsset(projectId: string, kind: "cast" | "image", id: string): Promise<TrashEntry> {
    await sleep(120);
    const p = this.getProject(projectId);
    if (this.busyIn(projectId, id)) throw new ApiError("A job is using this. Cancel it or wait for it to finish.", 409);
    const images = this.imagesByProject.get(projectId) ?? [];
    const cast = kind === "cast" ? p.cast.find((c) => c.id === id) : undefined;
    const image = kind === "image" ? images.find((im) => im.id === id) : undefined;
    if (!cast && !image) throw new ApiError(kind === "cast" ? "No such cast entry" : "No such image", 404);
    if (cast) p.cast = p.cast.filter((c) => c !== cast);
    if (image) this.imagesByProject.set(projectId, images.filter((im) => im !== image));
    const entry: TrashEntry = { id: uid("tr"), kind, project_id: projectId, asset_id: id, name: cast?.name ?? (image?.prompt.trim() || "Image"),
      trashed_at: Date.now() / 1000, bytes: 2_400_000, remote_files: 0, thumb_url: cast?.thumb_url ?? cast?.url ?? image?.thumb_url };
    this.trashed.set(entry.id, { entry, cast, image });
    return clone(entry);
  }

  async allCast(): Promise<CastMember[]> {
    await sleep(250);
    const out: CastMember[] = [];
    for (const p of this.projectsById.values()) for (const c of p.cast) out.push({ ...clone(c), project_id: p.id, project_name: p.name });
    return out;
  }

  async importCast(projectId: string, fromProjectId: string, castId: string): Promise<CastMember> {
    await sleep(350);
    const p = this.getProject(projectId);
    const src = this.getProject(fromProjectId).cast.find((c) => c.id === castId);
    if (!src) throw new ApiError("That cast entry isn't there anymore.", 404);
    if (p.cast.some((c) => c.name.toLowerCase() === src.name.toLowerCase())) throw new ApiError(`There's already a ${src.name} in this project's cast.`, 400);
    const c: CastMember = { ...clone(src), id: uid("cast"), created_at: Date.now() / 1000 };
    delete c.uses;
    p.cast.push(c);
    this.bumpProject(projectId);
    return clone(c);
  }

  /** How many shots use each cast entry (an @-mention or a reference in any take's settings). */
  private castUses(p: MockProject, shots: Shot[]): Record<string, number> {
    const uses: Record<string, number> = {};
    for (const c of p.cast) uses[c.id] = 0;
    for (const shot of shots) {
      const blob = JSON.stringify(shot.versions.map((v) => v.params ?? {})).toLowerCase();
      for (const c of p.cast) if (blob.includes(`@${c.name.toLowerCase()}`) || blob.includes(c.id)) uses[c.id] += 1;
    }
    return uses;
  }

  async project(id: string): Promise<ProjectDetail> {
    await sleep(250);
    const p = this.getProject(id);
    const shots = this.order.map((sid) => this.getShot(sid)).filter((x) => x.project_id === id && listed(x));
    const uses = this.castUses(p, shots);
    return { project: this.projectOut(p), shots: clone(shots), cast: clone(p.cast).map((c) => ({ ...c, uses: uses[c.id] ?? 0 })) };
  }

  async patchTake(shotId: string, takeId: string, patch: { status?: "" | "approved"; hidden?: boolean }): Promise<Shot> {
    await sleep(120);
    const shot = this.getShot(shotId);
    const v = shot.versions.find((x) => x.id === takeId);
    if (!v) throw new ApiError("That take isn't here.", 404);
    if (patch.status !== undefined) v.status = patch.status;
    if (patch.hidden !== undefined && v.id !== "orig") v.hidden = patch.hidden;
    this.derive(shot);
    this.bumpProject(shot.project_id);
    return clone(shot);
  }

  async addNote(shotId: string, takeId: string, note: { frame: number; text: string }): Promise<Note> {
    await sleep(120);
    const v = this.getShot(shotId).versions.find((x) => x.id === takeId);
    if (!v) throw new ApiError("That take isn't here.", 404);
    if (!note.text.trim()) throw new ApiError("A note needs some text.", 400);
    if (!Number.isInteger(note.frame) || note.frame < 0) throw new ApiError("Frame must be a whole number, 0 or more.", 400);
    const n: Note = { id: uid("note"), frame: note.frame, text: note.text.trim(), created_at: Date.now() / 1000 };
    v.notes = [...(v.notes ?? []), n];
    return clone(n);
  }

  async deleteNote(shotId: string, takeId: string, noteId: string): Promise<void> {
    await sleep(100);
    const v = this.getShot(shotId).versions.find((x) => x.id === takeId);
    if (v) v.notes = (v.notes ?? []).filter((n) => n.id !== noteId);
  }

  async revealMedia(): Promise<void> {
    await sleep(100);
    throw new ApiError("Reveal in Finder works on the real app, not in demo mode.", 501);
  }

  async revealTake(): Promise<void> {
    await sleep(100);
    throw new ApiError("Reveal in Finder works on the real app, not in demo mode.", 501);
  }

  async trashTake(shotId: string, takeId: string): Promise<TrashEntry> {
    await sleep(120);
    const shot = this.getShot(shotId);
    if (shot.project_id === "examples") throw new ApiError("Examples can't be moved to the trash.", 403);
    if (takeId === "orig") throw new ApiError("The original stays with its shot. Move the shot to the trash instead.", 403);
    if (this.jobOrder.some((id) => { const j = this.jobsById.get(id)!.job; return j.shot_id === shotId && (j.status === "queued" || j.status === "running"); })) {
      throw new ApiError("A job is working on this shot. Cancel it or wait for it to finish.", 409);
    }
    const at = shot.versions.findIndex((v) => v.id === takeId);
    if (at < 0) throw new ApiError("No such take", 404);
    const [version] = shot.versions.splice(at, 1);
    this.derive(shot);
    const entry: TrashEntry = { id: uid("tr"), kind: "take", shot_id: shotId, take_id: takeId, name: `${shot.name} · ${version.label}`,
      project_id: shot.project_id ?? "unsorted", trashed_at: Date.now() / 1000, bytes: 1_240_000_000, remote_files: 14, thumb_url: version.thumb_url };
    this.trashed.set(entry.id, { entry, version, at });
    for (const h of this.listeners) h.onShot?.(clone(shot));
    return clone(entry);
  }

  async trashShot(shotId: string): Promise<TrashEntry> {
    await sleep(120);
    const shot = this.getShot(shotId);
    if (shot.project_id === "examples") throw new ApiError("Examples can't be moved to the trash.", 403);
    this.shotsById.delete(shotId);
    this.order = this.order.filter((id) => id !== shotId);
    const entry: TrashEntry = { id: uid("tr"), kind: "shot", shot_id: shotId, take_id: null, name: shot.name, project_id: shot.project_id ?? "unsorted",
      trashed_at: Date.now() / 1000, bytes: 2_600_000_000 + 1_240_000_000 * (shot.versions.length - 1), remote_files: 14 * (shot.versions.length - 1), thumb_url: shot.thumb_url };
    this.trashed.set(entry.id, { entry, shot });
    for (const h of this.listeners) h.onShotRemoved?.({ id: shotId, project_id: shot.project_id });
    return clone(entry);
  }

  async trash(): Promise<TrashEntry[]> {
    await sleep(100);
    return [...this.trashed.values()].map((t) => clone(t.entry)).sort((a, b) => b.trashed_at - a.trashed_at);
  }

  async restoreTrash(trashId: string): Promise<{ entry: TrashEntry; shot?: Shot }> {
    await sleep(120);
    const t = this.trashed.get(trashId);
    if (!t) throw new ApiError("Already gone.", 404);
    if (t.project) {
      const now = this.projectsById.get(t.project.id);
      if (now && t.project.id !== "unsorted") throw new ApiError("Something is already in its place.", 409);
      // My shots came back empty when it went: what's in it now stays, and the old things join it.
      this.projectsById.set(t.project.id, now ? { ...t.project, cast: [...t.project.cast, ...now.cast] } : t.project);
      this.imagesByProject.set(t.project.id, [...(t.images ?? []), ...(this.imagesByProject.get(t.project.id) ?? [])]);
      for (const s of t.shots ?? []) {
        this.shotsById.set(s.id, s);
        this.order.unshift(s.id);
        for (const h of this.listeners) h.onShot?.(clone(s));
      }
      this.trashed.delete(trashId);
      return { entry: clone(t.entry) };
    }
    if (t.cast || t.image) {
      const p = this.projectsById.get(t.entry.project_id);
      if (!p) throw new ApiError("Restore its project first.", 404);
      if (t.cast) {
        if (p.cast.some((c) => c.name.toLowerCase() === t.cast!.name.toLowerCase())) throw new ApiError("Something is already in its place.", 409);
        p.cast.push(t.cast);
      }
      if (t.image) this.imagesByProject.set(p.id, [t.image, ...(this.imagesByProject.get(p.id) ?? [])]);
      this.trashed.delete(trashId);
      return { entry: clone(t.entry) };
    }
    const shotId = t.entry.shot_id ?? "";
    if (t.shot) {
      this.shotsById.set(t.shot.id, t.shot);
      this.order.unshift(t.shot.id);
    } else if (t.version) {
      if (!this.shotsById.has(shotId)) throw new ApiError("Restore its shot first.", 404);
      const shot = this.getShot(shotId);
      shot.versions.splice(Math.min(t.at ?? shot.versions.length, shot.versions.length), 0, t.version);
      this.derive(shot);
    }
    this.trashed.delete(trashId);
    const shot = this.getShot(shotId);
    for (const h of this.listeners) h.onShot?.(clone(shot));
    return { entry: clone(t.entry), shot: clone(shot) };
  }

  async emptyTrash(): Promise<EmptyTrashResult> {
    await sleep(300);
    const all = [...this.trashed.values()].map((t) => t.entry);
    this.trashed.clear();
    return { items: all.length, bytes: all.reduce((n, e) => n + e.bytes, 0), remote_files: all.reduce((n, e) => n + e.remote_files, 0) };
  }

  private saveDir: string | null = null;
  async saveFolder(): Promise<{ path: string | null; suggested: string; shown: string | null; suggested_shown: string }> {
    await sleep(60);
    return { path: this.saveDir, suggested: "~/Movies/Pickups", shown: this.saveDir, suggested_shown: "~/Movies/Pickups" };
  }
  async setSaveFolder(path: string): Promise<{ path: string }> {
    await sleep(60);
    this.saveDir = path.trim() || "~/Movies/Pickups";
    return { path: this.saveDir };
  }
  async chooseSaveFolder(): Promise<{ path: string | null }> {
    await sleep(60);
    return { path: null };
  }
  async revealSaveFolder(): Promise<void> {
    await sleep(60);
    throw new ApiError("Reveal in Finder works on the real app, not in demo mode.", 501);
  }

  async storage(): Promise<StorageInfo> {
    await sleep(200);
    return { path: "~/.pickups/library", bytes: 38_400_000_000 };
  }

  async revealLibrary(): Promise<void> {
    await sleep(100);
    throw new ApiError("Reveal in Finder works on the real app, not in demo mode.", 501);
  }

  async addCast(projectId: string, file: File, fields: { name: string; kind: CastKind; line: string }): Promise<CastMember> {
    await sleep(400);
    const p = this.getProject(projectId);
    const name = fields.name.trim();
    if (!name) throw new ApiError("Give it a name.", 400);
    if (p.cast.some((c) => c.name.toLowerCase() === name.toLowerCase())) throw new ApiError(`There's already a ${name} in this project's cast.`, 400);
    const c: CastMember = { id: uid("cast"), name, kind: fields.kind, line: fields.line.trim(), url: URL.createObjectURL(file), file: file.name, created_at: Date.now() / 1000 };
    p.cast.push(c);
    return clone(c);
  }

  async patchCast(projectId: string, castId: string, patch: { name?: string; kind?: CastKind; line?: string }): Promise<CastMember> {
    await sleep(150);
    const p = this.getProject(projectId);
    const c = p.cast.find((x) => x.id === castId);
    if (!c) throw new ApiError("That cast member isn't here.", 404);
    const name = patch.name?.trim();
    if (name && p.cast.some((x) => x.id !== castId && x.name.toLowerCase() === name.toLowerCase())) {
      throw new ApiError(`There's already a ${name} in this project's cast.`, 400);
    }
    if (name) c.name = name;
    if (patch.kind) c.kind = patch.kind;
    if (patch.line !== undefined) c.line = patch.line.trim();
    return clone(c);
  }

  async mentionsPreview(req: { project_id: string; text: string; uploads: number }): Promise<MentionPreview> {
    await sleep(120);
    const p = this.projectsById.get(req.project_id || "unsorted");
    return mentionPreview(req.text, p?.cast ?? [], req.uploads || 0);
  }

  async deliveries(projectId: string): Promise<Delivery[]> {
    await sleep(200);
    const out: Delivery[] = [];
    for (const sid of this.order) {
      const shot = this.getShot(sid);
      if (shot.project_id !== projectId || !listed(shot)) continue;
      for (const v of shot.versions) if (v.status === "approved") out.push({ shot_id: shot.id, shot_name: shot.name, take: clone(v) });
    }
    return out.sort((a, b) => Number(b.take.created_at) - Number(a.take.created_at));
  }

  async exportTimeline(projectId: string, format: TimelineFormat): Promise<{ blob: Blob; filename: string }> {
    const list = await this.deliveries(projectId);
    if (!list.length) throw new ApiError("Approve some takes first (⇧A on a take); the timeline is made from approved takes.", 409);
    const name = this.projectsById.get(projectId)?.name ?? "Pickups";
    const slug = `${projectFileStem(name)}_approved`;
    const clips = list
      .slice()
      .reverse()
      .map((d) => {
        const shot = this.getShot(d.shot_id);
        const file = d.take.downloads.find((x) => x.codec === "prores")?.filename ?? d.take.downloads[0]?.filename ?? `${shot.name}.mov`;
        // Notes go along in frame order, kept inside the clip (as the backend's nle.markers). The original's notes go
        // with the shot's approved takes (unless it's approved itself), at the take's frame showing the same moment;
        // outside the take they stay behind (nle.fold_original_notes).
        const orig = shot.versions[0];
        const start = d.take.id !== "orig" && orig.status !== "approved" ? mockTakeStart(shot, d.take) : null;
        const folded = start == null ? [] : (orig.notes ?? []).map((x) => ({ frame: Math.round(x.frame) - start, text: x.text, original: true }));
        const notes = [
          ...(d.take.notes ?? []).map((x) => ({ frame: Math.max(0, Math.min(shot.frames - 1, Math.round(x.frame))), text: x.text, original: false })),
          ...folded.filter((x) => x.frame >= 0 && x.frame < shot.frames),
        ]
          .map((x) => ({ ...x, text: x.text.replace(/\s+/g, " ").trim() }))
          .filter((x) => x.text)
          .sort((a, b) => a.frame - b.frame);
        return { file, shot: shot.name, frames: shot.frames, fps: shot.fps, src: 3600 * Math.round(shot.fps), notes };
      });
    const fps = clips[0]?.fps ?? 24;
    let text: string;
    if (format === "edl") {
      const lines = [`TITLE: ${name}`, "FCM: NON-DROP FRAME", ""];
      let rec = 3600 * Math.round(fps);
      clips.forEach((c, i) => {
        const n = String(i + 1).padStart(3, "0");
        lines.push(`${n}  AX       V     C        ${tc(c.src, c.fps)} ${tc(c.src + c.frames, c.fps)} ${tc(rec, fps)} ${tc(rec + c.frames, fps)}`);
        lines.push(`* FROM CLIP NAME: ${c.file}`);
        c.notes.forEach((x) => lines.push(`* LOC: ${tc(rec + x.frame, fps)} BLUE    ${x.text}${x.original ? " (original)" : ""}`));
        lines.push("");
        rec += c.frames;
      });
      text = lines.join("\r\n");
    } else {
      let at = 0;
      const markers: string[] = [];
      const items = clips.map((c, i) => {
        const item = `        <clipitem id="clipitem-${i + 1}"><name>${xmlEscape(c.file)}</name><start>${at}</start><end>${at + c.frames}</end><in>0</in><out>${c.frames}</out><file id="file-${i + 1}"><name>${xmlEscape(c.file)}</name><pathurl>file://localhost/${encodeURIComponent(c.file)}</pathurl><timecode><frame>${c.src}</frame><rate><timebase>${Math.round(c.fps)}</timebase></rate></timecode></file></clipitem>`;
        // Sequence markers at the note's place in the edit (Resolve imports markers from <sequence> only).
        c.notes.forEach((x) =>
          markers.push(
            `    <marker><name>${xmlEscape(x.text.length > 60 ? `${x.text.slice(0, 59).trimEnd()}…` : x.text)}</name><comment>${xmlEscape(`${x.text} (${x.original ? `${c.shot} · original` : c.file})`)}</comment><in>${at + x.frame}</in><out>-1</out></marker>`,
          ),
        );
        at += c.frames;
        return item;
      });
      text = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE xmeml>",
        '<xmeml version="5">',
        `  <sequence id="sequence-1"><name>${xmlEscape(name)}</name><rate><timebase>${Math.round(fps)}</timebase><ntsc>FALSE</ntsc></rate>`,
        ...markers,
        "    <media><video><track>",
        ...items,
        "    </track></video></media>",
        "  </sequence>",
        "</xmeml>",
        "",
      ].join("\n");
    }
    await sleep(500);
    return { blob: new Blob([text], { type: format === "xml" ? "application/xml" : "text/plain" }), filename: `${slug}.${format}` };
  }

  /** The backend sends a zip (index, a page per take, workflows); the mock saves one HTML page with every receipt. */
  async exportReceipts(projectId: string): Promise<{ blob: Blob; filename: string }> {
    const list = (await this.deliveries(projectId)).filter((d) => d.take.id !== "orig");
    if (!list.length) throw new ApiError("Approve some takes first (⇧A on a take); receipts are made from approved takes.", 409);
    const name = this.projectsById.get(projectId)?.name ?? "";
    const takes = list
      .map((d) => ({ shot: this.getShot(d.shot_id), id: d.take.id }))
      .map(({ shot, id }) => ({ shot, v: shot.versions.find((x) => x.id === id)! }))
      .sort((a, b) => a.shot.name.toLowerCase().localeCompare(b.shot.name.toLowerCase()) || a.shot.versions.indexOf(a.v) - b.shot.versions.indexOf(b.v));
    // The backend's index (receipts.project_zip), linking down the page instead of to takes/<name>.html.
    const rows = takes.map(({ shot, v }, i) => {
      const verb = TASK_TITLE[v.task as TaskId] ?? "Take";
      const pieces = receiptSummary(v.receipt).split(" · ").map((x) => `<span class="nw">${rx(x)}</span>`).join(" · ");
      // What the grey share is, in words, and what the take it was made from already was.
      const upscale = [upscaleSentence(v.receipt), sourceSentence(v.receipt)].filter(Boolean).join(" ");
      const words = upscale ? `<div class="muted" style="margin-top:4px">${rx(upscale)}</div>` : "";
      return `<tr><td class="th"><img src="${rx(absUrl(v.thumb_url))}" alt="${rx(shot.name)}"></td>
<td class="sh"><div class="shot">${rx(shot.name)}</div></td>
<td class="tk"><span class="mono">T${pad2(shot.versions.indexOf(v) + 1)}</span> · ${rx(verb)}${v.label !== verb ? `<div class="muted">${rx(v.label)}</div>` : ""}</td>
<td class="wc">${receiptBar(v.receipt)}<span class="sum">${pieces}</span>${words}</td>
<td class="lc">${licenceFlags(v.licences ?? [])}</td>
<td class="num lk"><a href="#take-${i + 1}">Receipt ↓</a></td></tr>`;
    });
    const nc = takes.filter(({ v }) => v.licences?.some((m) => m.commercial === false)).length;
    const warn = nc
      ? `<p class="callout"><strong>${nc} of ${takes.length} ${nc === 1 ? "take uses" : "takes use"}</strong> a model licensed for non-commercial use only. Their receipts say which.</p>`
      : "";
    const today = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
    const index = `<article class="sheet page"><header class="top"><span class="slate"><i class="dot"></i>Pickups receipts</span><span class="slate">${takes.length} approved ${takes.length === 1 ? "take" : "takes"}</span></header>
<p class="kicker slate">Project</p><h1>${rx(name)}</h1>
<p class="sub">Every approved take: what changed, the models that made it, and their licenses.</p>
<section>${warn}<table class="idx"><thead><tr><th></th><th>Shot</th><th>Take</th><th>What changed</th><th>Licenses</th><th class="num"></th></tr></thead><tbody>${rows.join("")}</tbody></table>
<p class="hint">One receipt per take follows, a page each when printed.</p></section>
<footer><span class="slate">Made with Pickups · open source</span><span class="slate">${rx(today)}</span></footer></article>`;
    const html = receiptDoc(`Receipts · ${name}`, [index, ...takes.map(({ shot, v }, i) => receiptArticle(name, shot, v, `take-${i + 1}`))]);
    await sleep(700);
    return { blob: new Blob([html], { type: "text/html" }), filename: `${projectFileStem(name)}_receipts.html` };
  }

  private receiptUrls = new Map<string, string>();

  /** No zip in demo mode: Download workflows saves the files one by one. */
  takeWorkflowsUrl(): string | null {
    return null;
  }

  takeReceiptUrl(shotId: string, takeId: string): string | null {
    const shot = this.shotsById.get(shotId);
    const v = shot?.versions.find((x) => x.id === takeId);
    if (!shot || !v || v.id === "orig") return null;
    const name = this.projectsById.get(shot.project_id ?? "")?.name ?? "";
    const html = receiptDoc(`Receipt · ${shot.name} · Take ${String(shot.versions.indexOf(v) + 1).padStart(2, "0")} ${v.label}`, [receiptArticle(name, shot, v)]);
    const key = `${shotId}:${takeId}`;
    const old = this.receiptUrls.get(key);
    if (old) URL.revokeObjectURL(old); // a page already open keeps its copy
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    this.receiptUrls.set(key, url);
    return url;
  }

  async shareTake(shotId: string, takeId: string, format: ShareFormat): Promise<ShareResult> {
    const shot = this.getShot(shotId);
    const idx = shot.versions.findIndex((x) => x.id === takeId);
    const v = shot.versions[idx];
    if (!v || v.id === "orig") throw new ApiError("Share works on a take, not the original.", 400);
    const stem = `${shot.name.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_|_$/g, "")}_T${String(idx + 1).padStart(2, "0")}_${v.task}`;
    // Made from the take itself where the browser can record; otherwise a cut-down made the same way from a real take.
    const before = v.before_url || shot.versions.find((x) => x.id === (v.input_version || "orig"))?.video_url || shot.video_url;
    const verb: Record<string, string> = { remove: "Remove a person", recast: "Replace a person", environment: "Change the weather", previz: "Previz to final", reshoot: "Reshoot", finish: "Finish" };
    // Shares as the app rounds them ("<1% AI"); an upscale (Finish) says so, since its pixels are the footage, not AI.
    const slate = [verb[v.task ?? ""] ?? "Take", aiShare(v.receipt)].filter(Boolean).join(" · ");
    const t0 = Date.now();
    const made = await composeShare(before, v.video_url, format, slate);
    if (made) return { url: made, filename: `${stem}_share_${format}.webm` };
    await sleep(Math.max(0, 2400 - (Date.now() - t0)));
    return { url: media(format === "9x16" ? "share_9x16.mp4" : "share_1x1.mp4"), filename: `${stem}_share_${format}.mp4` };
  }

  async finishApproved(projectId: string): Promise<{ queued: number; jobs: Job[] }> {
    // As the backend: skip Finish takes, takes already finished (a visible Finish take made from them) and ones finishing now.
    const skip = new Set<string>();
    for (const r of this.jobsById.values())
      if (r.job.task === "finish" && (r.job.status === "queued" || r.job.status === "running")) skip.add(`${r.job.shot_id}:${r.job.version_id || "orig"}`);
    for (const s of this.shotsById.values())
      for (const v of s.versions) if (v.task === "finish" && !v.hidden) skip.add(`${s.id}:${v.input_version || "orig"}`);
    const jobs: Job[] = [];
    for (const d of await this.deliveries(projectId)) {
      if (d.take.task === "finish" || skip.has(`${d.shot_id}:${d.take.id}`)) continue;
      jobs.push(await this.createJob({ shot_id: d.shot_id, version_id: d.take.id, task: "finish", params: { upscale_4k: true, hdr: true, quality: "best" } }));
    }
    return { queued: jobs.length, jobs };
  }

  async shot(id: string): Promise<Shot> {
    await sleep(200);
    return clone(this.getShot(id));
  }

  async trimShot(shotId: string, seconds: number): Promise<Shot> {
    await sleep(700);
    const s = this.getShot(shotId);
    s.duration_s = Math.min(s.duration_s, seconds);
    s.frames = Math.round(s.duration_s * s.fps);
    return clone(s);
  }

  /** As the backend's Trim & crop: a new shot in the same project, the original left as it is. The demo can't cut
   * video, so the new shot plays its source's media; its size, length and name are what the real one reports. */
  async prepareShot(shotId: string, req: PrepareRequest): Promise<Shot> {
    await sleep(1100);
    const src = this.getShot(shotId);
    const a = Math.round(req.in_frame);
    const b = Math.round(req.out_frame);
    if (!(a >= 0 && b <= src.frames - 1 && b >= a)) throw new ApiError(`In and out must be inside the clip (frames 0 to ${src.frames - 1}), with out after in.`, 400);
    const frames = b - a + 1;
    if (frames < 9) throw new ApiError("That's too short to work on. Keep at least 9 frames.", 400);
    const even = (v: number) => Math.max(2, Math.floor(v / 2) * 2);
    const box = req.crop ? { w: even(Math.min(req.crop.w, src.width)), h: even(Math.min(req.crop.h, src.height)) } : null;
    if (box && Math.min(box.w, box.h) < 64) throw new ApiError("That crop is too small. Make it at least 64 px on each side.", 400);
    const cropped = !!box && (box.w < even(src.width) || box.h < even(src.height));
    const parts = [frames < src.frames ? `${(a / src.fps).toFixed(1)}–${((a + frames) / src.fps).toFixed(1)} s` : "", cropped ? req.crop?.aspect ?? "crop" : ""].filter(Boolean);
    const id = uid("shot");
    const now = Date.now() / 1000;
    const orig: Version = { ...clone(src.versions.find((v) => v.id === "orig") ?? src.versions[0]), created_at: now, notes: [], status: "" };
    const shot: Shot = {
      id,
      project_id: src.project_id,
      created_at: now,
      name: req.name?.trim() || `${src.name} (${parts.join(", ") || "copy"})`,
      width: cropped && box ? box.w : src.width,
      height: cropped && box ? box.h : src.height,
      fps: src.fps,
      frames,
      duration_s: +(frames / src.fps).toFixed(3),
      thumb_url: orig.thumb_url,
      video_url: orig.video_url,
      versions: [orig],
    };
    this.shotsById.set(id, shot);
    this.keys.set(`${id}:orig`, this.keyOf(src));
    this.touch(id);
    this.derive(shot);
    this.bumpProject(shot.project_id);
    return clone(shot);
  }

  // -- API: selection --
  async select(shotId: string, req: SelectRequest): Promise<SelectResult> {
    await sleep(350 + Math.random() * 300);
    const shot = this.getShot(shotId);
    const key = this.keyOf(shot);
    return drawMask(shot, key, req);
  }

  /** As pickups/tracking.py, offline: the selection "follows them all the way" (the demo shows the shot itself; no
   * masks are drawn). A selection checked before comes back at once, as the real cache does. */
  async track(shotId: string, req: TrackRequest): Promise<TrackResult> {
    const shot = this.getShot(shotId);
    const vid = req.version_id && req.version_id !== "orig" ? req.version_id : "orig";
    const version = shot.versions.find((v) => v.id === vid) ?? shot.versions[0];
    const key = JSON.stringify([shotId, vid, req.frame, req.points, (req.text ?? "").trim()]);
    const cached = this.trackChecked.has(key);
    await sleep(cached ? 200 : TOUR ? 900 : 2500);
    this.trackChecked.add(key);
    const frames = Math.max(1, shot.frames);
    return {
      video_url: version.video_url,
      frames,
      fps: shot.fps || 24,
      area: Array.from({ length: frames }, () => 0.12),
      flagged: [],
      empty: [],
      cache_key: `mock${this.trackChecked.size}`,
      cached,
    };
  }

  /** As pickups/people.py: who's on the probe frame with the most people (ties: nearest the middle), left to right,
   * each with click points, a 3:4 cutout and a matte; the biggest is `main`. The finder's real answers per shot (PEOPLE);
   * on a take of that shot, the same people cut out of the take's own frame. */
  async people(shotId: string, req: PeopleRequest = {}): Promise<PeopleResult> {
    const shot = this.getShot(shotId);
    const vid = req.version_id && req.version_id !== "orig" ? req.version_id : "orig";
    const cacheKey = `${shotId}:${vid}`;
    const hit = this.peopleCache.get(cacheKey);
    if (hit && !req.refresh) {
      await sleep(150);
      return { ...clone(hit), cached: true };
    }
    await sleep(TOUR ? 500 : 1300);
    const key = this.keyOf(shot, vid);
    const W = shot.width;
    const H = shot.height;
    const real = PEOPLE[this.keyOf(shot)];
    let res: PeopleResult;
    if (real) {
      const version = shot.versions.find((v) => v.id === vid) ?? shot.versions[0];
      const base = this.keyOf(shot);
      // A take shows the same people where they were (a Recast keeps them in place), cut out of the take's own frame.
      const still = vid === "orig" ? null : (await posterFrom(version.video_url, real.frame / (shot.fps || 24) + 0.01, 3000, 640)) ?? version.thumb_url;
      const people: PersonFound[] = [];
      for (const [i, p] of real.people.entries()) {
        const points = p.points.map((x) => ({ ...x, positive: true }));
        if (p.drawn) {
          const mask = drawMask(shot, base, { frame: real.frame, points }).mask_url;
          const frameUrl = still ?? (await posterFrom(version.video_url, real.frame / (shot.fps || 24) + 0.01, 3000, 640)) ?? version.thumb_url;
          const thumb = (await cutout(frameUrl, mask, p.box, shot)) ?? frameUrl;
          people.push({ index: i, box: p.box, area: p.area, point: { x: points[0].x, y: points[0].y }, points, thumb_url: thumb, mask_url: mask, seen: 1, main: i === real.main });
          continue;
        }
        const f = p.file ?? i;
        const mask = media(`${base}_person_${f}_mask.png`);
        const thumb = still ? (await cutout(still, mask, p.box, shot)) ?? media(`${base}_person_${f}.jpg`) : media(`${base}_person_${f}.jpg`);
        people.push({ index: i, box: p.box, area: p.area, point: { x: points[0].x, y: points[0].y }, points, thumb_url: thumb, mask_url: mask, seen: 1, main: i === real.main });
      }
      res = { frame: real.frame, people, main: people.length ? real.main : null, probed: real.probed, window: real.window, width: W, height: H, cached: false };
    } else {
      const n = Math.max(1, Math.min(97, shot.frames));
      const w = 4 * Math.floor((n - 1) / 4) + 1;
      const probed = [...new Set([0, 1, 2, 3].map((k) => Math.min(w - 1, Math.floor((k * w) / 4))))];
      const frame = probed.reduce((a, b) => (Math.abs(b - w / 2) < Math.abs(a - w / 2) ? b : a));
      // [x, y, w, h] in source pixels: someone in a clip of your own; the demo's other shots have nobody to recast (cars,
      // playblasts, a castle) except the cyclist on the alpine ride.
      const boxes: number[][] =
        key === "upload" ? [[0.34 * W, 0.16 * H, 0.22 * W, 0.8 * H], [0.7 * W, 0.4 * H, 0.08 * W, 0.34 * H]]
        : key.startsWith("alpine_ride") ? [[0.443 * W, 0.32 * H, 0.091 * W, 0.41 * H]]
        : [];
      const version = shot.versions.find((v) => v.id === vid) ?? shot.versions[0];
      const still = boxes.length ? (await posterFrom(version.video_url, frame / (shot.fps || 24) + 0.01, 3000, 640)) ?? version.thumb_url : null;
      const areas = boxes.map((b) => (b[2] * b[3] * 0.6) / (W * H));
      const big = areas.indexOf(Math.max(...areas));
      const people: PersonFound[] = [];
      for (const [i, b] of boxes.entries()) {
        const [x, y, bw, bh] = b;
        const at = (fx: number, fy: number) => ({ x: Math.round(x + fx * bw), y: Math.round(y + fy * bh), positive: true });
        const points = [at(0.5, 0.4), at(0.5, 0.08), at(0.5, 0.78)];
        const mask = drawMask(shot, key, { frame, points }).mask_url;
        const thumb = (await cutout(still, mask, b, shot)) ?? still ?? "";
        people.push({ index: i, box: b.map(Math.round), area: +areas[i].toFixed(4), point: { x: points[0].x, y: points[0].y }, points, thumb_url: thumb, mask_url: mask, seen: probed.length, main: i === big });
      }
      res = { frame, people, main: people.length ? big : null, probed, window: w, width: W, height: H, cached: false };
    }
    this.peopleCache.set(cacheKey, clone(res));
    return res;
  }

  async upload(file: File, onProgress?: (f: number) => void): Promise<UploadResult> {
    if (!file.type.startsWith("image/") && !/\.(png|jpe?g|webp|tiff?|exr)$/i.test(file.name)) {
      throw new ApiError("That's not an image Pickups can read.", 415);
    }
    for (let i = 1; i <= 5; i++) {
      await sleep(60);
      onProgress?.(i / 5);
    }
    const up = { id: uid("up"), url: URL.createObjectURL(file) };
    this.uploadUrls.set(up.id, up.url);
    return up;
  }

  refUrl(ref: Ref, projectId?: string): string | null {
    // Uploads from before this session (an example's settings) stand in as a mock still.
    if (ref.kind === "upload") return this.uploadUrls.get(ref.id) ?? media("look_frame.jpg");
    if (ref.kind === "cast") {
      const all = [this.projectsById.get(projectId ?? ""), ...this.projectsById.values()].flatMap((p) => p?.cast ?? []);
      const c = all.find((x) => x.id === ref.id);
      return c ? c.thumb_url || c.url : null;
    }
    const im = (this.imagesByProject.get(projectId ?? "") ?? []).find((x) => x.id === ref.id);
    return im ? im.thumb_url || im.url : null;
  }

  async previzLook(req: PrevizLookRequest): Promise<PrevizLookResult> {
    if (!(req as { _quick?: boolean })._quick) await sleep(2600);
    this.lookFlip += 1;
    const look = { url: media(this.lookFlip % 2 ? "look_frame.jpg" : "look_frame_2.jpg"), upload_id: uid("up_look"), seed: req.seed };
    this.uploadUrls.set(look.upload_id, look.url);
    return look;
  }

  async previzFrame0(shotId: string): Promise<Shot> {
    return this.getShot(shotId);
  }

  async previzStartFrame(req: PrevizStartFrameRequest): Promise<PrevizStartFrameResult> {
    sent("/api/previz/start_frame", req);
    const shot = this.getShot(req.shot_id);
    const now = Date.now() / 1000;
    const job: Job = {
      id: uid("job"), lane: "inline", shot_id: shot.id, project_id: shot.project_id ?? null, shot_name: shot.name, project_name: null,
      task: "previz", status: "running", stage: "Start frame", stages: ["Start frame"], progress: 0, eta_s: null, error: null,
      params: {} as Job["params"], result_image_ids: [], created_at: now, started_at: now, step: "start_frame",
      ...(req.draft_key ? { draft_key: req.draft_key } : {}),
    };
    this.timed(job, "start_frame", { method: "B3" });
    this.jobsById.set(job.id, { job, req: { shot_id: shot.id, task: "previz", params: {} } as JobRequest, realMs: 0, elapsedMs: 0, weights: [1], speed: 1 });
    this.jobOrder.unshift(job.id);
    this.emit((h) => h.onJob(clone(job)));
    for (let t = 0; t < 26 && job.status === "running"; t++) await sleep(100);
    if (job.status !== "running") throw new ApiError("Cancelled.", 409);
    const look = await this.previzLook({ shot_id: req.shot_id, seed: req.seed, _quick: true } as PrevizLookRequest);
    const took_s = this.took(job);
    Object.assign(job, { status: "done", stage: "Done", progress: 1, result: { upload_id: look.upload_id, url: look.url, seed: look.seed, took_s }, took_s, finished_at: Date.now() / 1000 });
    this.emit((h) => h.onJob(clone(job)));
    return { ...look, took_s, method: "B3", subject_mask: true, positive: "", negative: "", references: (req.look_refs?.length ?? 0) + (req.look_upload_ids?.length ?? 0), todo: [], steps: 30, cfg: 3, template: "v2.1" };
  }

  // -- API: jobs --
  async createJob(req: JobRequest): Promise<Job> {
    sent("/api/jobs", req);
    await sleep(200);
    const shot = this.getShot(req.shot_id);
    let stages = STAGES[req.task];
    if (req.task === "recast" && (req.params as RecastParams).engine === "scail") stages = RECAST_SCAIL_STAGES;
    if (req.task === "environment" && keepOf(req.params as EnvironmentParams)) stages = ENV_KEEP_STAGES;
    if (req.task === "finish") {
      const p = req.params as FinishParams;
      stages = stages.filter(([l]) => (p.upscale_4k || !l.includes("4K")) && (p.hdr || !/HDR|highlights/.test(l)));
    }
    const weights = stages.map(([, s]) => s);
    const est = weights.reduce((a, b) => a + b, 0);
    const job: Job = {
      id: uid("job"),
      shot_id: shot.id,
      task: req.task,
      status: "queued",
      stage: "Waiting for the GPU",
      stages: stages.map(([l]) => l),
      progress: 0,
      eta_s: Math.round(est),
      error: null,
      version_id: req.version_id ?? null,
      // As the server: a step flow's draft and step ride on the job (the take chip reopens its Render step).
      ...(req.draft_key ? { draft_key: req.draft_key, step: req.step } : {}),
      result_version_id: null,
      params: req.params,
      created_at: Date.now() / 1000,
      shot_name: shot.name,
      project_id: shot.project_id ?? null,
      project_name: this.projectsById.get(shot.project_id ?? "")?.name ?? null,
    };
    this.timed(job, req.task, req.params as unknown as Record<string, unknown>);
    const realMs = TOUR ? 3200 : REAL_SECONDS[req.task] * 1000 * (req.task === "finish" ? Math.max(0.45, est / 1040) : 1);
    // The tour: a 3-minute render plays in 4 s with the countdown visibly ticking from its real estimate.
    const speed = TOUR ? est / (realMs / 1000) : SPEED;
    this.jobsById.set(job.id, { job, req, realMs, elapsedMs: 0, weights, speed });
    this.jobOrder.unshift(job.id);
    this.touch(shot.id);
    this.emit((h) => h.onJob(clone(job)));
    this.ensureTicking();
    return clone(job);
  }

  async jobs(): Promise<Job[]> {
    await sleep(200);
    return this.jobOrder.map((id) => clone(this.jobsById.get(id)!.job));
  }

  async moveJobNext(id: string): Promise<Job> {
    await sleep(120);
    const r = this.jobsById.get(id);
    if (!r || r.job.status !== "queued") throw new ApiError("Only a waiting run can move.", 409);
    // The demo's queue runs oldest first (the end of jobOrder): this one goes to that end.
    this.jobOrder = [...this.jobOrder.filter((x) => x !== id), id];
    Object.assign(r.job, { next_at: Date.now() / 1000 });
    this.emit((h) => h.onJob(clone(r.job)));
    return clone(r.job);
  }

  async cancelJob(id: string): Promise<Job | null> {
    await sleep(250);
    const r = this.jobsById.get(id);
    if (!r) throw new ApiError("That job isn't running.", 404);
    if (r.job.status === "queued" || r.job.status === "running") {
      Object.assign(r.job, { status: "canceled", stage: "Canceled", eta_s: 0, finished_at: Date.now() / 1000 });
      // A stopped sheet or image keeps nothing; a stopped inline row (the start frame) answers its page with 409.
      this.emit((h) => h.onJob(clone(r.job)));
    }
    return clone(r.job);
  }

  subscribe(handlers: ApiEvents): () => void {
    this.listeners.add(handlers);
    setTimeout(() => handlers.onOpen?.(), 0);
    return () => this.listeners.delete(handlers);
  }

  // -- measured times (the demo's own, as pickups/app/timings.py keeps the server's) --
  private timings = new Map<string, number[]>();

  /** A new job's kind and how long that kind usually takes in this demo (null before one has finished). */
  private timed(job: Job, task: string, params: Record<string, unknown>): void {
    job.kind = kindOf(task, params);
    job.usual_s = medianOf(this.timings.get(job.kind) ?? []);
  }

  /** A finished run's measured seconds, counted for its kind. */
  private took(job: Job): number {
    const start = typeof job.started_at === "number" ? job.started_at : Date.now() / 1000;
    const t = Math.round((Date.now() / 1000 - start) * 10) / 10;
    if (job.kind) this.timings.set(job.kind, [...(this.timings.get(job.kind) ?? []), t].slice(-10));
    return t;
  }

  // -- the simulated GPU --
  private ensureTicking() {
    if (this.timer !== undefined) return;
    this.timer = window.setInterval(() => this.tick(), TICK_MS);
  }

  private tick() {
    // Oldest first; inline rows (the start frame) run beside the GPU queue, never in it.
    const all = this.jobOrder.map((id) => this.jobsById.get(id)!).filter((r) => r.job.lane !== "inline").reverse();
    let running = all.find((r) => r.job.status === "running");
    if (!running) {
      running = all.find((r) => r.job.status === "queued");
      if (!running) {
        window.clearInterval(this.timer);
        this.timer = undefined;
        return;
      }
      // "busy": ComfyUI finishes someone else's job first (8 s), then the queue moves and the status says Ready.
      if (this.scenario === "busy" && this.st.comfy_queue?.outside) {
        this.busyUntil ??= Date.now() + 8000;
        if (Date.now() < this.busyUntil) return;
        this.st = { ...this.st, comfy_queue: { running: 0, pending: 0, outside: false } };
        const st = clone(this.st);
        this.emit((h) => h.onStatus(st));
      }
      Object.assign(running.job, { status: "running", started_at: Date.now() / 1000, stage: running.job.stages[0] });
    }
    const r = running;
    if (this.scenario === "stall" && r.job.task !== "image" && r.job.task !== "sheet" && r.elapsedMs >= r.realMs * 0.4) {
      if (!r.job.stalled) {
        Object.assign(r.job, {
          stalled: true,
          stall_message: "No progress from ComfyUI for 12 minutes. It may be stuck: restart ComfyUI on the GPU machine, then run this again.",
        });
        this.emit((h) => h.onJob(clone(r.job)));
      }
      return;
    }
    r.elapsedMs += TICK_MS;
    const frac = Math.min(1, r.elapsedMs / r.realMs);
    const total = r.weights.reduce((a, b) => a + b, 0);
    let acc = 0;
    let idx = 0;
    for (let i = 0; i < r.weights.length; i++) {
      if (frac * total >= acc) idx = i;
      acc += r.weights[i];
    }
    const stage = r.job.stages[idx];

    if (this.scenario === "fail" && r.job.task === "remove" && /Blending/.test(stage) && frac > 0.8) {
      Object.assign(r.job, {
        status: "failed",
        stage,
        eta_s: 0,
        error: "ComfyUI ran out of GPU memory (23.7 of 24 GB) in the blend pass at 3840×2160.",
        finished_at: Date.now() / 1000,
      });
      this.emit((h) => h.onJob(clone(r.job)));
      return;
    }
    // ?mock=refuse: Ideogram 4 draws its refusal card twice (tasks/image.py): a failed job, marked blocked, nothing saved.
    if (this.scenario === "refuse" && r.job.task === "image" && r.job.model === "ideogram4" && frac > 0.8) {
      Object.assign(r.job, { status: "failed", stage, eta_s: 0, blocked: true, error: "Ideogram's built-in filter refused this prompt. Try other words.", finished_at: Date.now() / 1000 });
      this.emit((h) => h.onJob(clone(r.job)));
      return;
    }

    if (frac >= 1 && r.job.task === "sheet") {
      if (!r.finishing) {
        r.finishing = true;
        void this.finishSheet(r);
      }
      return;
    }
    if (frac >= 1 && r.job.task === "video") {
      const took_s = this.took(r.job);
      const shot = this.finishVideo(r, took_s);
      Object.assign(r.job, { status: "done", stage: "Done", progress: 1, eta_s: 0, result_shot_id: shot.id, took_s, finished_at: Date.now() / 1000 });
      this.emit((h) => {
        h.onShot?.(clone(shot));
        h.onJob(clone(r.job));
      });
      return;
    }
    if (frac >= 1 && r.job.task === "image") {
      const took_s = this.took(r.job);
      const ids = this.finishImages(r, took_s);
      Object.assign(r.job, { status: "done", stage: "Done", progress: 1, eta_s: 0, result_image_ids: ids, took_s, finished_at: Date.now() / 1000 });
      const pid = r.job.project_id ?? "unsorted";
      this.emit((h) => {
        h.onJob(clone(r.job));
        h.onProject?.({ id: pid, images: ids });
      });
      return;
    }
    if (frac >= 1) {
      const shot = this.getShot(r.job.shot_id);
      const baseKey = this.keyOf(shot, r.req.version_id);
      const v = this.addVersion(shot, r.req.task, baseKey, r.req.params, Date.now() / 1000, r.req.version_id);
      const elapsed = (r.elapsedMs / 1000) * r.speed;
      v.details.total_seconds = Math.round(elapsed);
      v.details.gpu_seconds = Math.round(elapsed * 0.93);
      this.touch(shot.id);
      Object.assign(r.job, {
        status: "done",
        stage: "Done",
        progress: 1,
        eta_s: 0,
        result_version_id: v.id,
        took_s: this.took(r.job),
        finished_at: Date.now() / 1000,
      });
      this.emit((h) => {
        h.onShot?.(clone(shot));
        h.onJob(clone(r.job));
      });
      return;
    }
    // Ease the bar a touch so it doesn't look mechanical; never reach 1 before done.
    const shown = Math.min(0.99, frac * (0.96 + 0.04 * Math.sin(frac * 9)));
    Object.assign(r.job, {
      stage,
      progress: shown,
      eta_s: Math.max(1, Math.round(((r.realMs - r.elapsedMs) / 1000) * r.speed)),
    });
    this.emit((h) => h.onJob(clone(r.job)));
  }

  private finishImages(r: RunningJob, took_s?: number): string[] {
    const req = r.req as unknown as { project_id: string; image: ImageGenerateRequest; person_url?: string };
    const g = req.image;
    const model = IMAGE_MODELS.models.find((m) => m.id === g.model) ?? IMAGE_MODELS.models[0];
    const [w, h] = model.sizes[g.resolution]?.[g.aspect] ?? [1344, 768];
    const p = this.projectsById.get(req.project_id);
    const list = this.imagesByProject.get(req.project_id) ?? [];
    const ids: string[] = [];
    const mode = g.refs.length && model.modes.includes("edit") ? "edit" : "generate";
    const sent = mentionPreview(g.prompt, p?.cast ?? [], g.refs.length);
    const stills = stillsFor(g.prompt);
    for (let i = 0; i < g.count; i++) {
      // Stills from real takes and FLUX 3 klein, picked by what the prompt asks for.
      const still = req.person_url ?? media(stills[this.stillIndex++ % stills.length]);
      const im: ProjectImage = {
        id: uid("img"),
        url: still,
        thumb_url: still,
        width: w,
        height: h,
        model: model.id,
        model_name: model.name,
        mode,
        prompt: g.prompt,
        prompt_sent: mode === "edit" ? sent.images : sent.text,
        seed: (g.seed ?? Math.floor(Math.random() * 1e6)) + i,
        seed_locked: !!g.seed_locked,
        aspect: g.aspect,
        resolution: g.resolution,
        refs: g.refs,
        created_at: Date.now() / 1000,
        job_id: r.job.id,
        ...(took_s != null ? { took_s } : {}),
      };
      list.unshift(im);
      ids.push(im.id);
    }
    this.imagesByProject.set(req.project_id, list);
    this.bumpProject(req.project_id);
    return ids;
  }

  // -- API: images --
  async imageModels(): Promise<ImageModels> {
    await sleep(200);
    return clone(IMAGE_MODELS);
  }

  async generateImages(projectId: string, g: ImageGenerateRequest): Promise<Job> {
    sent("/api/images", g);
    await sleep(200);
    this.getProject(projectId);
    if (!g.prompt.trim() && !g.refs.length) throw new ApiError("Write a prompt first.", 400);
    // As tasks/image.py: Ideogram 4 writes a JSON caption from plain words first.
    const caption = g.model === "ideogram4" && !/^\s*\{/.test(g.prompt);
    const stages: Array<[string, number]> = caption
      ? [["Loading the model", 15], ["Writing the caption", 25], ["Making your images", 30]]
      : [["Loading the model", 15], ["Making your images", 30]];
    const job: Job = {
      id: uid("job"),
      shot_id: null as unknown as string,
      project_id: projectId,
      task: "image",
      status: "queued",
      stage: "Waiting for the GPU",
      stages: stages.map(([l]) => l),
      progress: 0,
      eta_s: 6 * g.count,
      error: null,
      params: g as unknown as Job["params"],
      result_image_ids: [],
      created_at: Date.now() / 1000,
      model: g.model,
      project_name: this.projectsById.get(projectId)?.name ?? null,
    };
    // Recast's "Make one" (lib/still.ts): a full-length character on a plain backdrop, the Fun cast's FLUX 3 klein still
    // the words ask for ("Try another" gives its other take, then the next character).
    // As the server: the seed it uses rides on the job.
    const seed = g.seed ?? freshSeed();
    job.params = { ...g, seed } as unknown as Job["params"];
    this.timed(job, "image", job.params as unknown as Record<string, unknown>);
    let person_url: string | undefined;
    if (isStillPrompt(g.prompt)) person_url = this.funStill(stillWords(g.prompt));
    const req = { shot_id: "", task: "finish", params: {}, project_id: projectId, image: { ...clone(g), seed }, person_url } as unknown as JobRequest;
    const realMs = TOUR ? 3200 : 1800 + 900 * g.count;
    this.jobsById.set(job.id, { job, req, realMs, elapsedMs: 0, weights: stages.map(([, w]) => w), speed: TOUR ? (6 * g.count) / (realMs / 1000) : 3 });
    this.jobOrder.unshift(job.id);
    this.emit((h) => h.onJob(clone(job)));
    this.ensureTicking();
    return clone(job);
  }

  // -- API: new video (text or image to video, LTX-2.5): done, it is a new shot in the project --
  async generateVideo(projectId: string, g: VideoGenerateRequest): Promise<Job> {
    sent("/api/videos", g);
    await sleep(200);
    this.getProject(projectId);
    if (!g.prompt.trim()) throw new ApiError(g.mode === "image" ? "Describe the motion first." : "Describe the shot first.", 400);
    if (g.mode === "image" && !g.image) throw new ApiError("Choose a picture to start from.", 400);
    if (!VIDEO_DURATIONS.some((d) => d.seconds === g.duration)) throw new ApiError("Pick a length of 1 s, 2 s, 3 s, 4 s.", 400);
    if (!hasSize(g.aspect as VideoAspect, g.resolution)) throw new ApiError(`${g.resolution} is 16:9 only. Pick 720p for ${g.aspect}.`, 400);
    const stages: Array<[string, number]> = [["Loading LTX-2.5", 20], ["Making your video", 50]];
    const params = { ...g, seed: g.seed ?? Math.floor(Math.random() * 2 ** 31), model: "ltx-2.5" };
    const job: Job = {
      id: uid("job"),
      shot_id: null as unknown as string,
      project_id: projectId,
      task: "video",
      status: "queued",
      stage: "Waiting for the GPU",
      stages: stages.map(([l]) => l),
      progress: 0,
      eta_s: 60,
      error: null,
      params: params as unknown as Job["params"],
      result_image_ids: [],
      created_at: Date.now() / 1000,
      model: "ltx-2.5",
      project_name: this.projectsById.get(projectId)?.name ?? null,
    };
    const [vw, vh] = videoSize(g.aspect as VideoAspect, g.resolution);
    this.timed(job, "video", { ...params, width: vw, height: vh, frames: framesFor(g.duration) });
    const req = { shot_id: "", task: "finish", params: {}, project_id: projectId, video: clone(params) } as unknown as JobRequest;
    const realMs = TOUR ? 3200 : 4200;
    this.jobsById.set(job.id, { job, req, realMs, elapsedMs: 0, weights: stages.map(([, w]) => w), speed: 60 / (realMs / 1000) });
    this.jobOrder.unshift(job.id);
    this.emit((h) => h.onJob(clone(job)));
    this.ensureTicking();
    return clone(job);
  }

  /** A finished video job: a new shot in its project, from an existing demo video (the demo can't make new ones). */
  private finishVideo(r: RunningJob, took_s?: number): Shot {
    const req = r.req as unknown as { project_id: string; video: VideoGenerateRequest & { seed: number } };
    const g = req.video;
    const [w, h] = videoSize(g.aspect as VideoAspect, g.resolution);
    const frames = framesFor(g.duration);
    const clip = g.mode === "image" ? "hood_push_previz" : "castle_dusk_environment";
    const now = Date.now() / 1000;
    const id = uid("shot");
    const orig: Version = {
      id: "orig",
      label: "Original",
      task: null,
      created_at: now,
      video_url: media(`${clip}.mp4`),
      thumb_url: media(`${clip}.jpg`),
      downloads: [],
      details: {},
      status: "",
      hidden: false,
      notes: [],
      input_version: null,
      params: {},
    };
    // As video.shot_name: the whole first sentence (a very long one stops at a whole word by 90 characters).
    let name = g.prompt.trim().replace(/\s+/g, " ").split(/(?<=[.!?])\s/)[0].replace(/^[\s,.;:]+|[\s,.;:]+$/g, "");
    if (name.length > 90) name = name.slice(0, 90).replace(/\s+\S*$/, "").replace(/[\s,.;:]+$/, "");
    name = name || "New video";
    const shot: Shot = {
      id,
      project_id: req.project_id,
      created_at: now,
      name,
      width: w,
      height: h,
      fps: VIDEO_FPS,
      frames,
      duration_s: +(frames / VIDEO_FPS).toFixed(3),
      thumb_url: orig.thumb_url,
      video_url: orig.video_url,
      versions: [orig],
      made: {
        task: "video", mode: g.mode, model: "ltx-2.5", prompt: g.prompt, aspect: g.aspect, resolution: g.resolution, width: w, height: h,
        frames, fps: VIDEO_FPS, seed: g.seed, seed_locked: !!g.seed_locked, ...(g.image ? { image: g.image } : {}), job_id: r.job.id,
        ...(took_s != null ? { took_s } : {}),
      },
    };
    this.shotsById.set(id, shot);
    this.keys.set(`${id}:orig`, "upload");
    this.touch(id);
    this.derive(shot);
    this.bumpProject(shot.project_id ?? "unsorted");
    return shot;
  }

  async projectImages(projectId: string): Promise<ProjectImage[]> {
    await sleep(200);
    const blobs = this.order.map((id) => this.shotsById.get(id)!).filter((s) => s.project_id === projectId).map((s) => JSON.stringify(s.versions.map((v) => v.params ?? {})));
    return clone(this.imagesByProject.get(projectId) ?? []).map((im) => ({ ...im, uses: blobs.filter((b) => b.includes(im.id)).length }));
  }

  async imageToCast(projectId: string, imageId: string, fields: { name: string; kind: CastKind; line: string }): Promise<CastMember> {
    const im = (this.imagesByProject.get(projectId) ?? []).find((x) => x.id === imageId);
    if (!im) throw new ApiError("That image isn't here.", 404);
    const p = this.getProject(projectId);
    const name = fields.name.trim();
    if (!name) throw new ApiError("Give it a name.", 400);
    if (p.cast.some((c) => c.name.toLowerCase() === name.toLowerCase())) throw new ApiError(`There's already a ${name} in this project's cast.`, 400);
    await sleep(250);
    const c: CastMember = { id: uid("cast"), name, kind: fields.kind, line: fields.line.trim(), url: im.url, created_at: Date.now() / 1000 };
    p.cast.push(c);
    return clone(c);
  }

  // -- API: character sheets (pickups/sheets.py) --
  async buildSheet(projectId: string, req: SheetRequest): Promise<Job> {
    await sleep(200);
    const p = this.getProject(projectId);
    const given = (["upload_id", "image_id", "cast_id", "describe"] as const).filter((k) => String(req[k] ?? "").trim());
    if (!given.length) throw new ApiError("Add a picture of them, or describe them.", 400);
    if (given.length > 1) throw new ApiError("Use one picture or a description, not both.", 400);
    let name = (req.name ?? "").trim().replace(/\s+/g, " ");
    let source: string | null = null;
    if (req.cast_id) {
      const c = p.cast.find((x) => x.id === req.cast_id);
      if (!c) throw new ApiError("No such cast entry", 404);
      if (c.kind !== "person") throw new ApiError("Character sheets are for people.", 400);
      name = c.name;
      source = this.sheetSource.get(c.id) ?? c.thumb_url ?? c.url; // a redo starts again from the same picture
    } else {
      if (!name) throw new ApiError("Give them a name.", 400);
      if (p.cast.some((c) => c.name.toLowerCase() === name.toLowerCase())) throw new ApiError(`There's already a cast entry called ${name}.`, 400);
      if (req.upload_id) {
        source = this.uploadUrls.get(req.upload_id) ?? null;
        if (!source) throw new ApiError("That picture is gone. Add it again.", 404);
      } else if (req.image_id) {
        const im = (this.imagesByProject.get(projectId) ?? []).find((x) => x.id === req.image_id);
        if (!im) throw new ApiError("No such image", 404);
        source = im.url;
      }
    }
    // From words: FLUX 3 klein's front view is the Fun cast character the words ask for.
    if (!source) source = this.funStill(req.describe ?? "");
    const stages: Array<[string, number]> = [["Making the front view", 12], ["Making the back view", 13], ["Making the close-up", 12], ["Putting the sheet together", 3]];
    const params = { ...req, name };
    const job: Job = {
      id: uid("job"),
      shot_id: null as unknown as string,
      project_id: projectId,
      task: "sheet",
      status: "queued",
      stage: "Waiting for the GPU",
      stages: stages.map(([l]) => l),
      progress: 0,
      eta_s: 40,
      error: null,
      params: params as unknown as Job["params"],
      result_image_ids: [],
      created_at: Date.now() / 1000,
      model: null,
      project_name: p.name,
    };
    this.timed(job, "sheet", params as unknown as Record<string, unknown>);
    const realMs = TOUR ? 3600 : 5200;
    // A Fun cast still comes back as its real sheet; any other picture is laid out as one in the browser.
    const real = funSheet(source);
    this.sheetWork.set(job.id, real ? Promise.resolve(real) : mockSheet(source));
    const run = { shot_id: "", task: "finish", params: {}, project_id: projectId, sheet: params, source } as unknown as JobRequest;
    this.jobsById.set(job.id, { job, req: run, realMs, elapsedMs: 0, weights: stages.map(([, w]) => w), speed: 40 / (realMs / 1000) });
    this.jobOrder.unshift(job.id);
    this.emit((h) => h.onJob(clone(job)));
    this.ensureTicking();
    return clone(job);
  }

  private async finishSheet(r: RunningJob): Promise<void> {
    const req = r.req as unknown as { project_id: string; sheet: SheetRequest; source: string };
    const pics = await this.sheetWork.get(r.job.id)!.catch(() => null);
    this.sheetWork.delete(r.job.id);
    if (r.job.status !== "running") return; // stopped meanwhile
    const p = this.projectsById.get(req.project_id);
    if (!pics || !p) {
      Object.assign(r.job, { status: "failed", stage: "Putting the sheet together", eta_s: 0, error: "That picture can't be opened. Try a PNG or JPEG.", finished_at: Date.now() / 1000 });
      this.emit((h) => h.onJob(clone(r.job)));
      return;
    }
    const editor = this.settingsState.commercial_safe ? "Qwen-Image-Edit 2511" : "Qwen-Image 2.1";
    const g = req.sheet;
    const models = [...(g.describe ? ["FLUX 3 klein"] : []), editor];
    const sheet: CastSheet = {
      url: pics.sheet,
      thumb_url: pics.sheet,
      views: { front: pics.front, back: pics.back, face: pics.face },
      editor,
      models,
      licences: licencesFor(models),
      seed: g.seed ?? Math.floor(Math.random() * 2e9),
      seed_locked: !!g.seed_locked,
      took_s: this.took(r.job),
      made_from: g.cast_id ? "cast" : g.upload_id ? "upload" : g.image_id ? "image" : "words",
      made_front: !!(g.upload_id || g.image_id),
      created_at: Date.now() / 1000,
    };
    let c = g.cast_id ? p.cast.find((x) => x.id === g.cast_id) : undefined;
    if (c) Object.assign(c, { url: pics.sheet, thumb_url: pics.front, sheet });
    else {
      c = { id: uid("cast"), name: g.name ?? "Your character", kind: "person", line: (g.describe ?? "").trim(), url: pics.sheet, thumb_url: pics.front, sheet, created_at: Date.now() / 1000 };
      p.cast.push(c);
    }
    if (!this.sheetSource.has(c.id)) this.sheetSource.set(c.id, req.source);
    this.bumpProject(p.id);
    const cid = c.id;
    Object.assign(r.job, { status: "done", stage: "Done", progress: 1, eta_s: 0, result_cast_id: cid, finished_at: Date.now() / 1000 });
    this.emit((h) => {
      h.onJob(clone(r.job));
      h.onProject?.({ id: p.id, images: [], cast: [cid] });
    });
  }

  // -- API: prompt director --
  async enhancePrompt(req: EnhanceRequest): Promise<EnhanceResult> {
    const d = this.settingsState.director!;
    if (d.backend === "off") throw new ApiError("The prompt director is turned off. You can turn it on in Setup.", 409);
    if (!req.text.trim()) throw new ApiError("Write something first.", 400);
    await sleep(TOUR ? 1700 : d.backend === "openai" ? 2200 : 4200);
    if (TOUR && this.tourDirectorText) {
      return { text: this.tourDirectorText, model: req.target === "ltx" ? "LTX-2.5" : "Qwen-Image 2.1", runs_on: "LM Studio", sent: req.text, target: req.target };
    }
    // The director keeps the user's @mentions, so the chips and the receipt still work after "Use".
    const base = req.text.trim().replace(/\s+/g, " ").replace(/\.$/, "");
    const cap = base.charAt(0).toUpperCase() + base.slice(1);
    const text =
      req.target === "ltx"
        ? `${cap}. A medium shot at eye level; the camera holds steady, then drifts slowly closer as the moment plays out. Soft natural light from the side, muted colour, fine film grain and crisp detail.`
        : req.target === "flux3"
          ? `Photograph of ${/^(A|An|The)\b/.test(base) ? base.charAt(0).toLowerCase() + base.slice(1) : base}, shot on a 50 mm lens at eye level, soft window light from the left, shallow depth of field, natural skin and fabric texture, muted, true-to-life colour.`
          : req.target === "qwen-edit"
            ? `${cap}. Keep the framing, every person's face and pose, and everything else exactly as it is.`
            : `${cap}, a clean, balanced composition with soft natural light and true-to-life colour, fine detail throughout.`;
    return { text, model: req.target === "ltx" ? "LTX-2.5" : req.target === "flux3" ? "FLUX 3" : "Qwen-Image 2.1", runs_on: TOUR ? "LM Studio" : d.backend === "openai" ? d.model || "a local LLM" : "ComfyUI", sent: base, target: req.target };
  }

  /** As pickups/describe.py: the first, middle and last of the frames the video model renders, described for the
   * field (DESCRIPTIONS: written for each demo clip and take); cached per shot, take, target and field. */
  async describeShot(shotId: string, req: DescribeRequest = {}): Promise<DescribeResult> {
    const d = this.settingsState.director!;
    if (d.backend === "off") throw new ApiError("The prompt director is turned off. You can turn it on in Setup.", 409);
    const shot = this.getShot(shotId);
    const vid = req.version_id && req.version_id !== "orig" ? req.version_id : "orig";
    if (!shot.versions.some((v) => v.id === vid)) throw new ApiError("No such shot or take", 404);
    const target = req.target ?? "ltx";
    const field = req.field ?? (target === "ltx" ? "scene" : "look");
    const cacheKey = `${shotId}:${vid}:${target}:${field}`;
    const hit = this.describeCache.get(cacheKey);
    if (hit && !req.refresh) {
      await sleep(150);
      return { ...clone(hit), cached: true };
    }
    await sleep(TOUR ? 900 : 1800 + Math.random() * 900);
    const entry = DESCRIPTIONS[this.keyOf(shot, vid)] ?? DESCRIPTIONS[this.keyOf(shot)];
    let text = entry ? (field === "motion" ? entry.motion ?? entry.scene : entry.scene) : GENERIC_DESCRIPTION;
    // "Try again" reads a little differently (the model samples): the camera gets its own sentence.
    if (req.refresh && hit && hit.text === text) text = text.replace(/; (the camera [^.]*)\.$/i, (_, cam: string) => `. ${cam.charAt(0).toUpperCase()}${cam.slice(1)}.`);
    const last = Math.max(0, Math.min(shot.frames, 97) - 1);
    const frames = target === "ltx" ? [...new Set([0, Math.floor(last / 2), last])] : [Math.max(0, Math.min(shot.frames - 1, req.frame ?? 0))];
    const camera = target !== "ltx" ? null : /camera is static/i.test(text) ? "static" : /pushes in/i.test(text) ? "push" : "moving";
    const local = d.backend === "openai" && !!d.url && !!d.vision_model;
    const res: DescribeResult = {
      text, target, field, version_id: vid, frames, camera,
      model: local ? d.vision_model : "Qwen3-VL 8B", runs_on: local ? d.vision_model : "ComfyUI", seconds: 1.8, cached: false,
    };
    this.describeCache.set(cacheKey, clone(res));
    return res;
  }

  async detectPromptServers(): Promise<{ found: DetectedServer[] }> {
    await sleep(900);
    if (new URLSearchParams(window.location.search).get("director") === "none") return { found: [] };
    // Demo fixture: LM Studio's default local address.
    return { found: [{ url: "http://localhost:1234/v1", kind: "lmstudio", models: ["text-model-8b-instruct", "text-model-4b-instruct", "vision-model-7b"], suggested_model: "text-model-8b-instruct", suggested_vision_model: "vision-model-7b" }] };
  }

  async testPromptServer(req: { url: string; model?: string }): Promise<PromptTestResult & Partial<DetectedServer>> {
    await sleep(1200);
    if (!/^https?:\/\/[^\s/]+/i.test(req.url.trim())) throw new ApiError("Couldn't reach that server. Is LM Studio's local server running (Developer → Start Server)?", 502);
    const models = ["text-model-8b-instruct", "text-model-4b-instruct", "vision-model-7b"];
    return { url: req.url.trim().replace(/\/$/, ""), kind: "lmstudio", models, model: req.model || models[0], seconds: 1.4, suggested_model: models[0], suggested_vision_model: models[2] };
  }

  async settings(): Promise<Settings> {
    await sleep(120);
    return clone(this.settingsState);
  }

  // -- API: Setup → Machines (mock data) --
  private machineList: import("./types").MachinesList = {
    active: "m-local",
    machines: [
      { id: "m-local", name: "This computer", kind: "local", url: "http://127.0.0.1:8188", status: { reachable: true, gpu: "RTX 4090", vram_gb: 24 } },
      { id: "m-pod", name: "RunPod", kind: "runpod", url: "https://abc123xyz-8188.proxy.runpod.net", status: { reachable: true, gpu: "RTX 5090", vram_gb: 32 } },
    ],
  };

  async machines(): Promise<import("./types").MachinesList> {
    await sleep(400);
    return clone(this.machineList);
  }

  async addMachine(req: import("./types").AddMachineRequest): Promise<import("./types").MachinesList> {
    await sleep(900);
    let url = (req.url ?? "").trim().replace(/\/$/, "");
    if (req.kind === "runpod") {
      const raw = (req.pod || req.url || "").trim().toLowerCase();
      const m = raw.match(/([a-z0-9]+)-(\d+)\.proxy\.runpod\.net/);
      const pod = m ? m[1] : raw;
      if (!/^[a-z0-9]{6,32}$/.test(pod)) throw new ApiError("A pod ID is letters and numbers only, like abc123xyz.", 400);
      url = `https://${pod}-${req.port || (m ? m[2] : "8188")}.proxy.runpod.net`;
    } else if (!/^https?:\/\//.test(url)) throw new ApiError("Use a full address, like http://192.168.1.20:8188", 400);
    if (this.machineList.machines.some((x) => x.url === url)) throw new ApiError("That machine is already saved.", 409);
    const probe = { reachable: true, gpu: req.kind === "runpod" ? "RTX 5090" : "RTX 4090", vram_gb: req.kind === "runpod" ? 32 : 24 };
    const id = `m-${Date.now().toString(36)}`;
    const name = req.name?.trim() || (req.kind === "runpod" ? "RunPod" : "GPU PC");
    this.machineList.machines.push({ id, name, kind: req.kind, url, status: probe });
    return { ...clone(this.machineList), added: id, probe };
  }

  async useMachine(id: string): Promise<import("./types").MachinesList> {
    await sleep(800);
    const m = this.machineList.machines.find((x) => x.id === id);
    if (!m) throw new ApiError("No such machine.", 404);
    this.machineList.active = id;
    const status = await this.connect(m.url);
    return { ...clone(this.machineList), status };
  }

  async removeMachine(id: string): Promise<import("./types").MachinesList> {
    await sleep(200);
    if (id === this.machineList.active) throw new ApiError("Switch to another machine first.", 409);
    this.machineList.machines = this.machineList.machines.filter((x) => x.id !== id);
    return clone(this.machineList);
  }

  async saveSettings(patch: { director?: Partial<DirectorSettings> } & Record<string, unknown>): Promise<Settings> {
    await sleep(200);
    for (const k of ["allow_nvidia_rtx", "commercial_safe"] as const) {
      if (typeof patch[k] === "boolean") this.settingsState[k] = patch[k] as boolean;
    }
    this.settingsState.open_weights_only = !this.settingsState.allow_nvidia_rtx;
    if (patch.director) {
      const d = patch.director;
      const backend = d.backend === "openai" || d.backend === "off" ? d.backend : "comfy";
      this.settingsState.director = {
        backend,
        url: String(d.url ?? (backend === "openai" ? this.settingsState.director?.url : "") ?? ""),
        model: String(d.model ?? ""),
        vision_model: String(d.vision_model ?? ""),
        off_acknowledged: backend === "off" ? !!d.off_acknowledged : false,
      };
    }
    return clone(this.settingsState);
  }

  async previzFrames(): Promise<{ received: number }> {
    await sleep(40);
    return { received: 16 };
  }

  async previzFraming(): Promise<PrevizFramingResult> {
    await sleep(120);
    return { warning: null, checked: false };
  }

  async previzFinish(): Promise<Shot> {
    await sleep(180);
    const shot = this.shotsById.get("shot_hood_push") ?? this.shotsById.values().next().value;
    if (!shot) throw new ApiError("No shot to open.", 404);
    return clone(shot);
  }

  async previzCancel(): Promise<void> {
    await sleep(20);
  }
}

// --- helpers --------------------------------------------------------------------------------------------------------

function probeVideo(url: string): Promise<{ width: number; height: number; duration: number; thumb: string }> {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "auto";
    v.playsInline = true;
    const fail = window.setTimeout(() => reject(new Error("timeout")), 10000);
    v.onerror = () => {
      window.clearTimeout(fail);
      reject(new Error("decode"));
    };
    v.onloadedmetadata = () => {
      if (!v.videoWidth || !isFinite(v.duration)) {
        window.clearTimeout(fail);
        reject(new Error("no video"));
        return;
      }
      v.currentTime = Math.min(0.5, v.duration / 3);
    };
    v.onseeked = () => {
      window.clearTimeout(fail);
      const c = document.createElement("canvas");
      c.width = 480;
      c.height = Math.round((480 * v.videoHeight) / v.videoWidth);
      c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);
      resolve({ width: v.videoWidth, height: v.videoHeight, duration: v.duration, thumb: c.toDataURL("image/jpeg", 0.8) });
    };
    v.src = url;
  });
}

/** What "Describe it" says about each demo clip and take, written from their first, middle and last frames (as the
 * describer is told to: only what's visible, one or two present-tense sentences, the measured camera move). `motion`
 * is Render previz's version: the playblast described as the live-action shot it stands for. */
const DESCRIPTIONS: Record<string, { scene: string; motion?: string }> = {
  castle_dusk: { scene: "A grey stone castle with a square keep stands on a small grassy island in a loch, joined to the shore by an arched stone bridge, under a heavy overcast sky; the camera is static." },
  trail: { scene: "A woman with dark hair, in a white T-shirt and black trousers, walks into frame and away from the camera across a green alpine valley toward three jagged rock peaks under a cloudy sky; the camera is static." },
  castle_walk: { scene: "A woman in a white jacket and dark trousers walks away from the camera along a cobbled path through a green lawn toward a square stone castle, under a blue sky with white clouds; the camera follows behind her." },
  alpine_ride: { scene: "A woman with long dark hair, in a white top and dark trousers, rides a bicycle away from the camera down a country road through green alpine meadows toward jagged mountains in bright midday sun; the camera follows behind her." },
  glen: { scene: "A man in a dark gilet and a white long-sleeved top walks in from the left and away from the camera across green moorland toward a dark mountain under a heavy grey sky; the camera is static." },
  glen_long: { scene: "A man in a dark gilet and jeans walks across the frame and away from the camera over green moorland toward a dark mountain under heavy grey cloud; the camera is static." },
  offroad: { scene: "A dark blue SUV drives along a wet, straight road past flat snowy fields and telegraph poles under a pale overcast sky; the camera tracks alongside it." },
  coast_curve: { scene: "A blue SUV with its lights on comes round a wet curve toward the camera, with a guardrail and a grey, choppy sea behind it under heavy cloud; the camera leads ahead of it." },
  misty_road: { scene: "A blue SUV with its headlights on drives toward the camera along a wet, curving road with double yellow lines, past bare winter trees, a guardrail and patches of snow in grey mist; the camera leads ahead of it." },
  coastal_drive: { scene: "A grey SUV drives toward the camera along a coast road beside the sea, backlit by a low sun through thin cloud, and passes close by; the camera leads ahead of it." },
  hood_push: {
    scene: "A driver in a white helmet with blue stars sits at the wheel of a car with a roll cage, seen through the windscreen in soft daylight; the camera pushes in toward him.",
    motion: "A driver in a helmet sits at the wheel of a car with a roll cage, both hands on the steering wheel, looking straight ahead. The camera pushes in through the windscreen toward his head and shoulders.",
  },
  driver_profile: {
    scene: "A driver in a white helmet with blue stars sits in a car with a roll cage, both hands on the wheel, seen from the passenger side, with low buildings and containers outside; the camera is static.",
    motion: "A driver in a helmet sits in the driver's seat of a car with a roll cage, both hands on the wheel, looking ahead, with a yard of low buildings and containers outside. The camera is static.",
  },
  launch: {
    scene: "A white sedan drives away from the camera down a long straight road between high walls toward a low hazy sun and a distant city skyline; the camera is static at bumper height.",
    motion: "A white sedan pulls away from the camera and drives off down a long straight road between high walls, toward a low hazy sun, a distant city skyline and a dark vehicle far ahead. The camera is static at bumper height.",
  },
  alpine_ride_environment: { scene: "A woman with long dark hair in a white top rides a bicycle away from the camera down a country road through golden autumn grass and orange larches toward rocky mountains in warm sun; the camera follows behind her." },
  castle_dusk_environment: { scene: "A stone castle on a small snow-covered island in a loch glows with warm light in its windows at blue-hour dusk, joined to the shore by an arched stone bridge; the camera is static." },
  castle_walk_environment: { scene: "A woman in a white jacket and dark trousers walks away from the camera along a snowy path toward a square stone castle, between snow-laden trees in soft, flat winter light; the camera follows behind her." },
  castle_walk_recast: { scene: "A person with red curly hair in a yellow raincoat walks away from the camera along a cobbled path through a green lawn toward a square stone castle, under a blue sky with white clouds; the camera follows behind them." },
  coast_curve_recast: { scene: "A red open-top vintage roadster with chrome bumpers comes round a wet curve toward the camera, with a guardrail and a grey sea behind it under heavy cloud; the camera leads ahead of it." },
  coastal_drive_finish: { scene: "A grey SUV drives toward the camera along a coast road beside the sea, backlit by a low sun through thin cloud, and passes close by; the camera leads ahead of it." },
  driver_profile_previz: { scene: "An older man with a white moustache, in a cream helmet with blue stars and a sage-green work shirt, drives a stripped-out car with a roll cage, both hands on the wheel, past a yard of containers in warm late light; the camera is static." },
  glen_long_recast: { scene: "A figure in a silver suit with a green alien head walks in from the left and away from the camera across green moorland toward a dark mountain under heavy grey cloud; the camera is static." },
  glen_recast: { scene: "An astronaut in a white spacesuit and a gold-visored helmet walks in from the left and away from the camera across green moorland toward a dark mountain under heavy grey cloud; the camera is static." },
  glen_recast_b: { scene: "An astronaut in a white spacesuit and a gold-visored helmet walks in from the left and away from the camera across green moorland toward a dark mountain under heavy grey cloud; the camera is static." },
  hood_push_previz: { scene: "An older man with a white moustache, in a cream helmet with blue stars, grips the wheel of a stripped-out car and stares ahead wide-eyed, with stacks of scrapped cars behind him in warm late light; the camera pushes in through the windscreen toward him." },
  hood_push_previz_a: { scene: "An older man with a white moustache, in a cream helmet with blue stars, grips the wheel of a stripped-out car and stares ahead wide-eyed, with stacks of scrapped cars behind him in warm late light; the camera pushes in through the windscreen toward him." },
  hood_push_previz_b: { scene: "An older man with a white moustache, in a cream helmet with blue stars, rubs his chin, then grips the wheel of a stripped-out car, with stacks of scrapped cars behind him in warm late light; the camera pushes in through the windscreen toward him." },
  launch_previz: { scene: "A white sedan kicks up dust as it pulls away down a dirt road between towering stacks of scrapped cars, toward a low sun and a distant city skyline; the camera is static at bumper height." },
  misty_road_environment: { scene: "A blue SUV with its headlights on drives toward the camera along a snow-covered road past bare trees and a guardrail as heavy snow falls; the camera leads ahead of it." },
  misty_road_environment_b: { scene: "A blue SUV with its headlights on drives toward the camera along a snow-covered road past bare trees, a guardrail and a red brick building as heavy snow falls; the camera leads ahead of it." },
  offroad_recast: { scene: "A green off-roader with a white roof drives along a wet, straight road past flat snowy fields and telegraph poles under a pale overcast sky; the camera tracks alongside it." },
  trail_remove: { scene: "Three jagged grey rock peaks rise above a green alpine valley crossed by winding paths, under a cloudy sky; the camera is static." },
};
/** A clip of your own in the demo: the mock can't look at it, so it says something plain. */
const GENERIC_DESCRIPTION = "A person walks through the frame in soft natural daylight, past trees and a low wall; the camera is static.";

/** The people finder's real answers on the demo shots' originals (pickups/people.py): the probe frame, who it found
 * there (box and click points in source pixels) and the biggest. Cutouts and mattes: `<key>_person_<i>.jpg` / `_mask.png`;
 * `file`: which `<key>_person_<n>` files are theirs (default: their place). `drawn`: no files, their matte is drawn from
 * the box (someone small in the background). */
const PEOPLE: Record<string, { frame: number; probed: number[]; window: number; main: number; people: Array<{ box: number[]; area: number; points: Array<{ x: number; y: number }>; file?: number; drawn?: boolean }> }> = {
  glen: { frame: 48, probed: [0, 24, 48, 72], window: 97, main: 0, people: [{ box: [2232, 1631, 433, 1056], area: 0.018, points: [{ x: 2427, y: 1924 }, { x: 2489, y: 2270 }] }] },
  glen_long: { frame: 145, probed: [0, 24, 48, 72, 97, 145, 193, 241], window: 260, main: 0, people: [{ box: [2269, 1643, 400, 1044], area: 0.0172, points: [{ x: 2468, y: 1941 }, { x: 2518, y: 2274 }] }] },
  // Recast anyone: a walker further down the left-hand path, as well as the woman on the cobbles.
  castle_walk: { frame: 0, probed: [0, 24, 48, 72], window: 97, main: 1, people: [
    { box: [2030, 2800, 160, 390], area: 0.0012, points: [{ x: 2110, y: 2930 }, { x: 2110, y: 3080 }], drawn: true },
    { box: [2640, 2312, 624, 1560], area: 0.0168, points: [{ x: 2904, y: 3240 }, { x: 2904, y: 3588 }], file: 0 },
  ] },
  trail: { frame: 82, probed: [0, 41, 82, 123], window: 165, main: 0, people: [{ box: [1814, 1274, 534, 886], area: 0.027, points: [{ x: 2050, y: 1782 }, { x: 2050, y: 1968 }] }] },
};

const PERSON = /person|man|woman|guy|girl|boy|figure|people|someone|walker|him|her/i;

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

/** The people finder's 3:4 cutout (as people.py's _thumb): the person in colour, the rest of the frame dimmed grey. */
async function cutout(frameUrl: string | null, maskUrl: string, box: number[], shot: Shot): Promise<string | null> {
  if (!frameUrl) return null;
  try {
    const [frame, mask] = await Promise.all([loadImage(frameUrl), loadImage(maskUrl)]);
    const [x0, y0, bw, bh] = box;
    let w = bw * 1.28;
    let h = bh * 1.28;
    if (w / h > 0.75) h = w / 0.75;
    else w = h * 0.75;
    const cx = x0 + bw / 2;
    const cy = y0 + bh / 2;
    const src = (img: HTMLImageElement) => {
      const k = img.naturalWidth / shot.width;
      return [(cx - w / 2) * k, (cy - h / 2) * k, w * k, h * k] as const;
    };
    const out = document.createElement("canvas");
    out.width = 192;
    out.height = 256;
    const g = out.getContext("2d")!;
    g.fillStyle = "#000";
    g.fillRect(0, 0, 192, 256);
    g.drawImage(frame, ...src(frame), 0, 0, 192, 256);
    // Grey and dim everything...
    const px = g.getImageData(0, 0, 192, 256);
    for (let i = 0; i < px.data.length; i += 4) {
      const l = (px.data[i] * 0.3 + px.data[i + 1] * 0.59 + px.data[i + 2] * 0.11) * 0.32;
      px.data[i] = px.data[i + 1] = px.data[i + 2] = l;
    }
    g.putImageData(px, 0, 0);
    // ...then the person in colour, through their matte.
    const person = document.createElement("canvas");
    person.width = 192;
    person.height = 256;
    const p = person.getContext("2d")!;
    p.drawImage(frame, ...src(frame), 0, 0, 192, 256);
    p.globalCompositeOperation = "destination-in";
    p.drawImage(mask, ...src(mask), 0, 0, 192, 256);
    g.drawImage(person, 0, 0);
    return out.toDataURL("image/jpeg", 0.86);
  } catch {
    return null;
  }
}

/** A mock character sheet from one picture (pickups/sheets.py's layout, half size): the picture as the front view,
 * mirrored in cool grey as the "back view", its top as the "close-up", 4 px white gaps, a MOCK slate. */
async function mockSheet(src: string): Promise<MockSheetPictures> {
  const im = await loadImage(src);
  const W = 1000;
  const H = 562;
  const gap = 4;
  const pw = Math.round((W - 2 * gap) / 3);
  const canvas = (w: number, h: number) => {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    return [c, c.getContext("2d")!] as const;
  };
  const iw = im.naturalWidth || im.width || 400;
  const ih = im.naturalHeight || im.height || 600;
  // A full-length view: the whole height, centred, sides cropped or filled with the backdrop's grey.
  const view = (mirror: boolean, tint: boolean) => {
    const [c, g] = canvas(pw, H);
    g.fillStyle = "#cfcfcf";
    g.fillRect(0, 0, pw, H);
    const s = H / ih;
    const w = iw * s;
    g.save();
    if (mirror) {
      g.translate(pw, 0);
      g.scale(-1, 1);
    }
    if (tint) g.filter = "grayscale(0.55) brightness(0.82) contrast(1.05)";
    g.drawImage(im, (pw - w) / 2, 0, w, H);
    g.restore();
    return c;
  };
  // The close-up: the upper middle of the picture (a head, whether it's a full-length still or a bust), filling the panel.
  const face = () => {
    const [c, g] = canvas(pw, H);
    const sw = iw * 0.55;
    const sh = Math.min((sw * H) / pw, ih * 0.9);
    g.drawImage(im, (iw - sw) / 2, ih * 0.06, sw, sh, 0, 0, pw, H);
    return c;
  };
  const panels = [view(false, false), view(true, true), face()];
  const [sheet, g] = canvas(W, H);
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, W, H);
  panels.forEach((p, i) => g.drawImage(p, i * (pw + gap), 0));
  g.font = "600 13px ui-monospace, Menlo, monospace";
  g.fillStyle = "rgba(0,0,0,0.55)";
  g.fillRect(W - 66, H - 30, 56, 20);
  g.fillStyle = "#f3efe6";
  g.fillText("MOCK", W - 58, H - 15);
  const url = (c: HTMLCanvasElement) => c.toDataURL("image/jpeg", 0.86);
  return { sheet: url(sheet), front: url(panels[0]), back: url(panels[1]), face: url(panels[2]) };
}

function drawMask(shot: Shot, key: string, req: SelectRequest): SelectResult {
  const W = 480;
  const H = Math.round((W * shot.height) / shot.width);
  const k = W / shot.width; // source px → mask px
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  const pos = req.points.filter((p) => p.positive);
  const neg = req.points.filter((p) => !p.positive);
  let bbox: number[] = [0, 0, 0, 0];

  // A click on someone the finder knows (or words that name a person), near its probe frame: their real matte.
  const real = PEOPLE[key] as (typeof PEOPLE)[string] | undefined;
  const pad = 0.03 * shot.width;
  let person = -1;
  if (real && Math.abs(req.frame - real.frame) <= 12 && !neg.length) {
    person = real.people.findIndex(({ box: [x, y, w, h] }) => pos.some((p) => p.x > x - pad && p.x < x + w + pad && p.y > y - pad && p.y < y + h + pad));
    if (person < 0 && !pos.length && req.text && PERSON.test(req.text)) person = real.main;
  }

  if (real && person >= 0 && real.people[person].drawn) {
    // Someone small in the background: a matte the shape of their box.
    const [x, y, w, h] = real.people[person].box;
    g.beginPath();
    g.ellipse((x + w / 2) * k, (y + h / 2) * k, (w / 2) * k, (h / 2) * k, 0, 0, Math.PI * 2);
    g.fill();
    bbox = [x, y, w, h];
  } else if (real && person >= 0) {
    const [x, y, w, h] = real.people[person].box;
    return { mask_url: media(`${key}_person_${real.people[person].file ?? person}_mask.png`), bbox: [x - 4, y - 4, w + 8, h + 8] };
  } else if (pos.length || req.text) {
    let cx: number, cy: number, rx: number, ry: number;
    if (pos.length >= 2) {
      const xs = pos.map((p) => p.x);
      const ys = pos.map((p) => p.y);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      cx = (x0 + x1) / 2;
      cy = (y0 + y1) / 2;
      rx = (x1 - x0) / 2 + 0.05 * shot.width;
      ry = (y1 - y0) / 2 + 0.07 * shot.height;
    } else if (pos.length === 1) {
      cx = pos[0].x;
      cy = pos[0].y;
      rx = 0.06 * shot.width;
      ry = 0.17 * shot.height;
    } else {
      cx = shot.width * 0.5;
      cy = shot.height * 0.55;
      rx = 0.08 * shot.width;
      ry = 0.2 * shot.height;
    }
    g.beginPath();
    g.ellipse(cx * k, cy * k, rx * k, ry * k, 0, 0, Math.PI * 2);
    g.fill();
    bbox = [cx - rx, cy - ry, rx * 2, ry * 2];
  }
  // Negative points carve the mask.
  g.globalCompositeOperation = "destination-out";
  for (const p of neg) {
    g.beginPath();
    g.arc(p.x * k, p.y * k, 0.045 * shot.width * k, 0, Math.PI * 2);
    g.fill();
  }
  return { mask_url: c.toDataURL("image/png"), bbox: bbox.map(Math.round) };
}
