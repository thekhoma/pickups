// Connecting to any ComfyUI (this computer, another on the network or Tailscale, a RunPod pod) and what it can do.
// Live only: the mock has its own fixed machine.
import { request } from "../api";
import type { TaskId } from "../types";

export type ServerKind = "local" | "lan" | "runpod";

export interface ListenCommand {
  os: string;
  label: string;
  command: string;
  note: string;
}

export interface ServerTest {
  ok: boolean;
  url: string | null;
  kind?: ServerKind;
  reason: null | "address" | "dns" | "refused" | "timeout" | "not_comfy" | "http";
  message: string;
  gpu?: string | null;
  /** The device name ComfyUI reports, e.g. "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync". */
  gpu_raw?: string | null;
  vram_gb?: number | null;
  ram_gb?: number | null;
  os?: string | null;
  comfy_version?: string | null;
  ms?: number;
  listen?: ListenCommand[];
}

export type FoundServer = ServerTest & { name: string; self: boolean };

export interface RecentServer {
  url: string;
  kind: ServerKind;
  gpu: string | null;
  used_at: number;
}

export interface Servers {
  active: { url: string; kind: ServerKind };
  recent: RecentServer[];
  welcome_done: boolean;
  listen: ListenCommand[];
}

export interface Measured {
  gpu: string;
  runs: number;
  max_frames: number;
  max_seconds: number;
  fps: number;
  minutes: number | null;
}

export interface NeededModel {
  file: string;
  folder: string;
  gb: number;
  gated: boolean;
  repo: string;
  url: string;
  page: string;
  /** null: ComfyUI didn't list that folder, so it's unknown (not missing). */
  installed: boolean | null;
}

export interface MissingPack {
  pack: string;
  url: string;
  by: string;
  manager_name?: string;
  nodes?: string[];
  install?: string;
}

export interface Feature {
  task: TaskId | "image";
  label: string;
  ready: boolean;
  /** Why not: nodes_unknown (couldn't read nodes), comfy (older than 0.41), models_unknown, nodes, models; null when
   * ready (or only VRAM). */
  reason?: "nodes_unknown" | "comfy" | "models_unknown" | "nodes" | "models" | null;
  /** This GPU against the feature's rule: ok, untested (NVIDIA under its VRAM: allowed), unsupported (not NVIDIA). */
  support?: "ok" | "untested" | "unsupported";
  /** The VRAM it's tested at (decimal GB, as the gate reads it) and the free space it wants beyond its files. */
  vram_gb?: number;
  headroom_gb?: number;
  /** Finish only: what each choice needs (4K alone is SeedVR2). */
  choices?: Partial<Record<"4k-hdr" | "hdr" | "4k", { ready: boolean; needs: NeededModel[]; missing_gb: number }>>;
  needs: NeededModel[];
  missing_gb: number;
  missing_nodes: MissingPack[];
  nodes?: { checked: boolean; core: string[]; update: string | null; packs: MissingPack[] };
  blocked?: "vram" | null;
  vram_needed_gb?: number | null;
  measured: Measured | null;
}

export interface Capabilities {
  connected: boolean;
  server: { url: string; kind: ServerKind };
  machine?: { gpu: string | null; vram_gb: number | null; ram_gb: number | null; os: string | null; comfy_version: string | null; device: string | null };
  measured_gpu?: string | null;
  features: Feature[];
  downloads?: { here: boolean; command: string };
  comfy_version?: string | null;
  models_dir?: string | null;
  models_dirs?: string[];
  /** The models folder Pickups can write here (ComfyUI on this computer), or null. */
  models_dir_here?: string | null;
  requirements?: Requirements;
}

/** This computer against the minimum (SYSTEM_REQUIREMENTS.md). Free space in decimal GB; null when unknown. */
export interface Requirements {
  gpu: string | null;
  nvidia: boolean;
  vram_gb: number;
  min_vram_gb: number;
  comfy_version: string | null;
  comfy_ok: boolean | null;
  min_comfy: string;
  /** Where downloads land, only when that's this computer. */
  models_free_gb: number | null;
  library_free_gb: number | null;
  library_min_free_gb: number;
  ffmpeg: number | null;
  min_ffmpeg: number;
}

export interface ModelsDownload {
  running: boolean;
  task?: string | null;
  tasks?: string[];
  files?: string[];
  total?: number;
  done?: number;
  file?: string | null;
  error?: string | null;
}

/** Why the models can't be fetched from here, and what to do instead. */
export interface ModelsHelp {
  message: string;
  command?: string;
  gated?: string[];
  files?: NeededModel[];
}

/** A network scan: what answered, and what was looked at (said plainly when nothing did). */
export interface ScanResult {
  found: FoundServer[];
  tried: number;
  networks?: string[];
  peers?: number;
  ports?: number[];
  timeout?: number;
}

/** "Looked at 254 addresses on 192.168.1.0/24 and 3 Tailscale devices, port 8188." */
export function scannedLine(r: ScanResult): string {
  const where = [...(r.networks ?? []), ...(r.peers ? [`${r.peers} Tailscale ${r.peers === 1 ? "device" : "devices"}`] : [])];
  const ports = r.ports?.length ? `, port ${r.ports.join(", ")}` : "";
  if (!r.tried) return "No local network found to look on.";
  return `Looked at ${r.tried} ${r.tried === 1 ? "address" : "addresses"}${where.length ? ` on ${where.join(" and ")}` : ""}${ports}, twice. Nothing answered.`;
}

export const servers = {
  list: () => request<Servers>("GET", "/api/servers"),
  test: (address: string) => request<ServerTest>("POST", "/api/servers/test", { address }),
  detect: () => request<{ found: ServerTest | null; tried: ServerTest[]; listen: ListenCommand[] }>("GET", "/api/servers/detect"),
  /** ComfyUI on this computer's private networks and Tailscale devices; `self`: one of this computer's own addresses. */
  scan: (patient = false) => request<ScanResult>("GET", `/api/servers/scan${patient ? "?patient=1" : ""}`),
  use: (address: string) => request<Servers & { test: ServerTest }>("POST", "/api/servers/use", { address }),
  forget: (url: string) => request<Servers>("POST", "/api/servers/forget", { url }),
  welcomeDone: () => request<Servers>("POST", "/api/welcome"),
  capabilities: () => request<Capabilities>("GET", "/api/capabilities"),
  download: (task?: string | string[], finishChoice?: string) =>
    request<ModelsDownload>("POST", "/api/models/download", {
      ...(Array.isArray(task) ? { tasks: task } : task ? { task } : {}),
      ...(finishChoice ? { finish_choice: finishChoice } : {}),
    }),
  downloadProgress: () => request<ModelsDownload>("GET", "/api/models/download"),
};

/** "gpu-pc", "192.168.1.20", "RunPod pod", "This computer": a server's address as a person would name it. */
export function serverName(url: string, kind?: ServerKind): string {
  try {
    const u = new URL(url);
    if (kind === "local" || ["127.0.0.1", "localhost", "[::1]"].includes(u.hostname)) return "This computer";
    if (u.hostname.endsWith(".proxy.runpod.net")) return `RunPod ${u.hostname.split("-")[0]}`;
    return u.hostname;
  } catch {
    return url;
  }
}

/** "Up to 4.8 s per take · about 14 min each, measured on RTX 4090", or null (not tested on this GPU). */
export function measuredLine(m: Measured | null): string | null {
  if (!m) return null;
  const each = m.minutes != null ? ` · about ${Math.round(m.minutes)} min each` : "";
  return `Up to ${m.max_seconds} s per take${each}, measured on ${m.gpu}`;
}
