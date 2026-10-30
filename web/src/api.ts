// Typed client for the Pickups backend (docs/ui-spec.md → "API contract").
// The backend is served from the same origin; in dev, Vite proxies /api and /media to http://127.0.0.1:7788.
import type { PrevizSpec } from "./previz/spec";
import type {
  ApiEvents,
  VideoGenerateRequest,
  CastKind,
  DetectedServer,
  DirectorSettings,
  DescribeRequest,
  DescribeResult,
  EnhanceRequest,
  EnhanceResult,
  ImageGenerateRequest,
  ImageModels,
  ProjectImage,
  PromptTestResult,
  Settings,
  MachinesList,
  AddMachineRequest,
  ShareFormat,
  ShareResult,
  TimelineFormat,
  CastMember,
  Delivery,
  Example,
  ExampleOpenResult,
  ExamplesPack,
  ModelCatalog,
  Job,
  JobRequest,
  MentionPreview,
  Note,
  Project,
  ProjectDetail,
  PeopleRequest,
  PeopleResult,
  PrevizFraming,
  PrevizFramingResult,
  PrevizLookRequest,
  PrevizLookResult,
  PrevizStartFrameRequest,
  PrevizStartFrameResult,
  Ref,
  Sample,
  SelectRequest,
  SelectResult,
  SheetRequest,
  PrepareRequest,
  Shot,
  Status,
  TrackRequest,
  TrackResult,
  UploadResult,
  EmptyTrashResult,
  StorageInfo,
  TrashEntry,
} from "./types";

/** Save videos to: the folder (null until chosen), the suggestion, and both as a person reads them (~/Movies/Pickups). */
export interface SaveFolder {
  path: string | null;
  suggested: string;
  shown?: string | null;
  suggested_shown?: string | null;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface PickupsApi {
  readonly mode: "live" | "mock";
  status(): Promise<Status>;
  connect(comfy_url: string): Promise<Status>;
  samples(): Promise<Sample[]>;
  openSample(id: string): Promise<Shot>;
  /** Home-screen before/after examples. Throws ApiError(404) when the backend doesn't have them yet. */
  examples(): Promise<Example[]>;
  openExample(id: string): Promise<ExampleOpenResult>;
  /** GET /api/models/defaults: each workflow's default models. Throws ApiError(404) on older backends. */
  defaultModels(): Promise<ModelCatalog>;
  /** GET /api/examples/pack: whether the example gallery can be downloaded. */
  examplesPack(): Promise<ExamplesPack>;
  /** POST /api/examples/pack: downloads and unpacks it (10–30 s). */
  installExamplesPack(): Promise<{ examples: number }>;
  /** Install an examples pack: an uploaded zip, or a zip or folder path on the machine running Pickups. */
  importExamplesPack(src: File | string): Promise<{ examples: number; samples: number }>;
  /** Open a file that's already on the backend's disk / network share (no upload), into a project. */
  shotFromPath(path: string, projectId?: string): Promise<Shot>;
  uploadShot(file: File, onProgress?: (fraction: number) => void, projectId?: string): Promise<Shot>;
  shots(projectId?: string): Promise<Shot[]>;
  shot(id: string): Promise<Shot>;
  patchShot(id: string, patch: { name?: string; project_id?: string }): Promise<Shot>;

  // Projects, takes, cast, deliveries (docs/projects-spec.md).
  projects(): Promise<Project[]>;
  createProject(name: string): Promise<Project>;
  renameProject(id: string, name: string): Promise<Project>;
  /** Archived projects leave the switcher; nothing moves. */
  archiveProject(id: string, archived: boolean): Promise<Project>;
  /** Delete: the project's folder and its shots go to the trash (Restore puts them back). My shots comes back empty.
   * Returns the trash entry and the projects left. */
  trashProject(id: string): Promise<{ entry: TrashEntry; projects: Project[] }>;
  /** Delete a cast entry or a project image: its files go to the trash. Takes that used it keep their settings. */
  trashAsset(projectId: string, kind: "cast" | "image", id: string): Promise<TrashEntry>;
  /** Every cast entry in every project, with project_id / project_name. */
  allCast(): Promise<CastMember[]>;
  /** Copies a cast entry from another project into this one. */
  importCast(projectId: string, fromProjectId: string, castId: string): Promise<CastMember>;
  project(id: string): Promise<ProjectDetail>;
  /** Approve / hide a take ("orig" too). Returns the updated shot. */
  patchTake(shotId: string, takeId: string, patch: { status?: "" | "approved"; hidden?: boolean }): Promise<Shot>;
  addNote(shotId: string, takeId: string, note: { frame: number; text: string }): Promise<Note>;
  deleteNote(shotId: string, takeId: string, noteId: string): Promise<void>;
  /** Opens the take's folder in Finder on the server machine. */
  revealTake(shotId: string, takeId: string): Promise<void>;
  trashTake(shotId: string, takeId: string): Promise<TrashEntry>;
  trashShot(shotId: string): Promise<TrashEntry>;
  trash(): Promise<TrashEntry[]>;
  /** `shot` for a take or a shot. */
  restoreTrash(trashId: string): Promise<{ entry: TrashEntry; shot?: Shot }>;
  emptyTrash(): Promise<EmptyTrashResult>;
  storage(): Promise<StorageInfo>;
  /** Save videos to: the folder finished takes are copied into (one folder per project), and a suggestion. */
  saveFolder(): Promise<SaveFolder>;
  setSaveFolder(path: string): Promise<{ path: string; shown?: string | null }>;
  /** The computer's own folder picker (Pickups on this computer); null when canceled. */
  chooseSaveFolder(): Promise<{ path: string | null }>;
  revealSaveFolder(): Promise<void>;
  revealLibrary(): Promise<void>;
  /** Reveal any library file the app shows (an image, a sheet, a clip) by its /media/ URL. */
  revealMedia(url: string): Promise<void>;
  addCast(projectId: string, file: File, fields: { name: string; kind: CastKind; line: string }): Promise<CastMember>;
  patchCast(projectId: string, castId: string, patch: { name?: string; kind?: CastKind; line?: string }): Promise<CastMember>;
  /** Queues a character sheet (task "sheet", no shot): a new cast member, or `cast_id` upgraded / made again. */
  buildSheet(projectId: string, req: SheetRequest): Promise<Job>;
  mentionsPreview(req: { project_id: string; text: string; uploads: number }): Promise<MentionPreview>;
  deliveries(projectId: string): Promise<Delivery[]>;
  /** Queues Finish 4K + HDR for every approved take that isn't a Finish already. */
  finishApproved(projectId: string): Promise<{ queued: number; jobs: Job[] }>;
  /** The approved takes as a timeline file (FCP7 XML for Resolve / Premiere, or a CMX3600 EDL). 409 when nothing is approved. */
  exportTimeline(projectId: string, format: TimelineFormat): Promise<{ blob: Blob; filename: string }>;
  /** The approved takes' receipt sheets as a zip (index.html, one page per take, their ComfyUI workflows). The mock
   * saves one combined HTML page instead. 409 when nothing is approved. */
  exportReceipts(projectId: string): Promise<{ blob: Blob; filename: string }>;
  /** Where a take's receipt sheet opens (a printable page, in a new tab). The mock builds it as a Blob URL. null: none. */
  takeReceiptUrl(shotId: string, takeId: string): string | null;
  /** Every ComfyUI workflow of a take as one zip. null: none to offer (the page downloads the files one by one). */
  takeWorkflowsUrl(shotId: string, takeId: string): string | null;
  /** Renders a social cut-down of a take (takes a few seconds). */
  shareTake(shotId: string, takeId: string, format: ShareFormat): Promise<ShareResult>;

  // Images (docs/images-spec.md).
  imageModels(): Promise<ImageModels>;
  /** Queues a job (task "image", no shot). */
  generateImages(projectId: string, req: ImageGenerateRequest): Promise<Job>;
  /** Queues a job (task "video", no shot): text or image to video on LTX-2.5. Done, it is a new shot (result_shot_id). */
  generateVideo(projectId: string, req: VideoGenerateRequest): Promise<Job>;
  projectImages(projectId: string): Promise<ProjectImage[]>;
  imageToCast(projectId: string, imageId: string, fields: { name: string; kind: CastKind; line: string }): Promise<CastMember>;

  // Prompt director.
  enhancePrompt(req: EnhanceRequest): Promise<EnhanceResult>;
  /** "Describe it": a local model looks at the shot (or a take) and describes it for a prompt field. Cached per take. */
  describeShot(shotId: string, req?: DescribeRequest): Promise<DescribeResult>;
  detectPromptServers(): Promise<{ found: DetectedServer[] }>;
  testPromptServer(req: { url: string; model?: string }): Promise<PromptTestResult & Partial<DetectedServer>>;
  settings(): Promise<Settings>;
  /** Setup → Machines: saved GPU machines with a live check of each. */
  machines(): Promise<MachinesList>;
  addMachine(req: AddMachineRequest): Promise<MachinesList>;
  /** Make a machine active. 409 while a job is running. */
  useMachine(id: string): Promise<MachinesList>;
  removeMachine(id: string): Promise<MachinesList>;
  saveSettings(patch: { director?: Partial<DirectorSettings> } & Record<string, unknown>): Promise<Settings>;
  select(shotId: string, req: SelectRequest): Promise<SelectResult>;
  /** Check tracking: the selection followed through the shot (an amber overlay video and where it slips). */
  track(shotId: string, req: TrackRequest): Promise<TrackResult>;
  /** Recast's people finder: who's in the shot (or a take), with click points, cutouts and mattes. Cached per take. */
  people(shotId: string, req?: PeopleRequest): Promise<PeopleResult>;
  upload(file: File, onProgress?: (fraction: number) => void): Promise<UploadResult>;
  /** Where to see a reference a take's settings keep by id (an upload, a cast entry, a project image). null: none. */
  refUrl(ref: Ref, projectId?: string): string | null;
  previzLook(req: PrevizLookRequest): Promise<PrevizLookResult>;
  previzStartFrame(req: PrevizStartFrameRequest): Promise<PrevizStartFrameResult>;
  previzFrame0(shotId: string, body: { framing?: PrevizFraming | null; matte?: string | null; depth?: Record<string, string> | null }): Promise<Shot>;
  /** One batch of previz JPEGs. `start` is the index of the first frame in the batch. */
  previzFrames(session: string, start: number, files: Blob[], signal?: AbortSignal): Promise<{ received: number }>;
  /** Assemble the uploaded frames into a shot. */
  previzFinish(session: string, body: { spec: PrevizSpec; project_id?: string; name?: string; framing?: PrevizFraming | null; matte?: string | null; depth?: Record<string, string> | null }, signal?: AbortSignal): Promise<Shot>;
  /** Does a start frame pinned at frame 0 match the playblast's framing? A warning, never a block. */
  previzFraming(req: { shot_id: string; look_frame_ref?: Ref; look_frame_upload_id?: string }): Promise<PrevizFramingResult>;
  /** Drop a previz upload that was canceled. */
  previzCancel(session: string): Promise<void>;
  createJob(req: JobRequest): Promise<Job>;
  jobs(): Promise<Job[]>;
  cancelJob(id: string): Promise<Job | null>;
  /** Move a waiting run to the front of the GPU queue. */
  moveJobNext(id: string): Promise<Job>;
  /** Not in the v1 contract (see docs/ui-notes.md): fetch the missing models for every tile. */
  downloadModels(): Promise<Status | null>;
  /** Not in the v1 contract (see docs/ui-notes.md): trim a long shot to its first N seconds. */
  trimShot(shotId: string, seconds: number): Promise<Shot>;
  /** Trim & crop: a new shot (same project) from in..out of this one's source, cropped to a shape. The original stays. */
  prepareShot(shotId: string, req: PrepareRequest): Promise<Shot>;
  /** Server-Sent Events on /api/events. Returns an unsubscribe function. */
  subscribe(handlers: ApiEvents): () => void;
}

// ---------------------------------------------------------------------------------------------------------------

export async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      // A FormData body (a file) goes as multipart, its boundary set by the browser.
      headers: body === undefined || body instanceof FormData ? { Accept: "application/json" } : { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Pickups can't reach its server.", 0);
  }
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    let detail =
      data && typeof data === "object"
        ? ((data as Record<string, unknown>).detail ?? (data as Record<string, unknown>).error ?? (data as Record<string, unknown>).message)
        : data;
    // A structured detail ({message, command, files}) reads as its message; the rest stays on the error's body.
    if (detail && typeof detail === "object" && typeof (detail as Record<string, unknown>).message === "string") detail = (detail as Record<string, unknown>).message;
    throw new ApiError(typeof detail === "string" && detail ? detail : `Request failed (${res.status})`, res.status, data);
  }
  return data as T;
}

/** Multipart upload with progress (fetch has no upload progress, so this uses XHR). */
function uploadMultipart<T>(path: string, file: File, onProgress?: (f: number) => void, fields: Record<string, string> = {}): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError("Pickups can't reach its server.", 0));
    xhr.onload = () => {
      let data: unknown = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        data = xhr.responseText;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as T);
      else {
        const detail = data && typeof data === "object" ? (data as Record<string, unknown>).detail : null;
        reject(new ApiError(typeof detail === "string" ? detail : `Upload failed (${xhr.status})`, xhr.status, data));
      }
    };
    const form = new FormData();
    form.append("file", file, file.name);
    for (const [k, v] of Object.entries(fields)) form.append(k, v);
    xhr.send(form);
  });
}

const enc = encodeURIComponent;

/** File name from a Content-Disposition header (RFC 5987 filename* first). */
function dispositionName(h: string | null): string | null {
  if (!h) return null;
  const star = /filename\*\s*=\s*(?:UTF-8|utf-8)?''([^;]+)/.exec(h);
  if (star) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"|"$/g, "")).replace(/[\\/]/g, "_");
    } catch {
      /* fall through */
    }
  }
  const plain = /filename\s*=\s*("([^"]*)"|[^;]+)/.exec(h);
  const name = plain ? (plain[2] ?? plain[1]).trim() : "";
  return name ? name.replace(/[\\/]/g, "_") : null;
}

/** A file the server sends as an attachment (a timeline, the receipts zip), saved under its Content-Disposition name.
 * `what` finishes "This server can't … yet" for an older server that doesn't have the endpoint. */
async function fetchFile(path: string, fallback: string, what: string): Promise<{ blob: Blob; filename: string }> {
  let res: Response;
  try {
    res = await fetch(path);
  } catch {
    throw new ApiError("Pickups can't reach its server.", 0);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail: unknown = text;
    try {
      const data = JSON.parse(text) as Record<string, unknown>;
      detail = data.detail ?? data.error ?? data.message;
    } catch {
      /* plain text */
    }
    throw new ApiError(typeof detail === "string" && detail && detail.length < 300 ? detail : `Export failed (${res.status})`, res.status);
  }
  // An older server answers unknown /api paths with the app's index.html.
  if (/text\/html/i.test(res.headers.get("Content-Type") ?? "")) {
    throw new ApiError(`This server can't ${what} yet. Restart Pickups to update it.`, 404);
  }
  const blob = await res.blob();
  return { blob, filename: dispositionName(res.headers.get("Content-Disposition")) ?? fallback };
}

export class HttpApi implements PickupsApi {
  readonly mode = "live" as const;

  status() {
    return request<Status>("GET", "/api/status");
  }
  connect(comfy_url: string) {
    return request<Status>("POST", "/api/connect", { comfy_url });
  }
  samples() {
    return request<Sample[]>("GET", "/api/samples");
  }
  openSample(id: string) {
    return request<Shot>("POST", `/api/samples/${enc(id)}/open`);
  }
  async examples() {
    const data = await request<unknown>("GET", "/api/examples");
    // Before the endpoint exists, the SPA fallback answers with index.html.
    if (!Array.isArray(data)) throw new ApiError("Examples aren't available.", 404);
    return data as Example[];
  }
  async defaultModels() {
    const data = await request<unknown>("GET", "/api/models/defaults");
    if (!data || typeof data !== "object" || Array.isArray(data)) throw new ApiError("No model catalog.", 404);
    return data as ModelCatalog;
  }
  async examplesPack() {
    const data = await request<unknown>("GET", "/api/examples/pack");
    if (!data || typeof data !== "object" || !("available" in data)) throw new ApiError("No example pack.", 404);
    return data as ExamplesPack;
  }
  installExamplesPack() {
    return request<{ examples: number }>("POST", "/api/examples/pack");
  }
  importExamplesPack(src: File | string) {
    if (typeof src === "string") return request<{ examples: number; samples: number }>("POST", "/api/examples/import", { path: src });
    const fd = new FormData();
    fd.append("file", src);
    return request<{ examples: number; samples: number }>("POST", "/api/examples/import", fd);
  }
  async openExample(id: string) {
    const data = await request<ExampleOpenResult>("POST", `/api/examples/${enc(id)}/open`);
    if (!data || typeof data !== "object" || !("shot" in data)) throw new ApiError("Examples aren't available.", 404);
    return data;
  }
  shotFromPath(path: string, projectId?: string) {
    return request<Shot>("POST", "/api/shots/from_path", { path, ...(projectId ? { project_id: projectId } : {}) });
  }
  uploadShot(file: File, onProgress?: (f: number) => void, projectId?: string) {
    return uploadMultipart<Shot>("/api/shots", file, onProgress, projectId ? { project_id: projectId } : {});
  }
  shots(projectId?: string) {
    return request<Shot[]>("GET", projectId ? `/api/shots?project_id=${enc(projectId)}` : "/api/shots");
  }
  patchShot(id: string, patch: { name?: string; project_id?: string }) {
    return request<Shot>("PATCH", `/api/shots/${enc(id)}`, patch);
  }
  projects() {
    return request<Project[]>("GET", "/api/projects");
  }
  createProject(name: string) {
    return request<Project>("POST", "/api/projects", { name });
  }
  renameProject(id: string, name: string) {
    return request<Project>("PATCH", `/api/projects/${enc(id)}`, { name });
  }
  archiveProject(id: string, archived: boolean) {
    return request<Project>("PATCH", `/api/projects/${enc(id)}`, { archived });
  }
  trashProject(id: string) {
    return request<{ entry: TrashEntry; projects: Project[] }>("DELETE", `/api/projects/${enc(id)}`);
  }
  trashAsset(projectId: string, kind: "cast" | "image", id: string) {
    return request<TrashEntry>("DELETE", `/api/projects/${enc(projectId)}/${kind === "cast" ? "cast" : "images"}/${enc(id)}`);
  }
  async allCast() {
    const data = await request<unknown>("GET", "/api/cast");
    if (!Array.isArray(data)) throw new ApiError("This server can't share cast between projects yet. Restart Pickups to update it.", 404);
    return data as CastMember[];
  }
  importCast(projectId: string, fromProjectId: string, castId: string) {
    return request<CastMember>("POST", `/api/projects/${enc(projectId)}/cast/import`, { from_project_id: fromProjectId, cast_id: castId });
  }
  project(id: string) {
    return request<ProjectDetail>("GET", `/api/projects/${enc(id)}`);
  }
  patchTake(shotId: string, takeId: string, patch: { status?: "" | "approved"; hidden?: boolean }) {
    return request<Shot>("PATCH", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}`, patch);
  }
  addNote(shotId: string, takeId: string, note: { frame: number; text: string }) {
    return request<Note>("POST", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}/notes`, note);
  }
  async deleteNote(shotId: string, takeId: string, noteId: string) {
    await request<unknown>("DELETE", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}/notes/${enc(noteId)}`);
  }
  async revealTake(shotId: string, takeId: string) {
    await request<unknown>("POST", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}/reveal`);
  }
  trashTake(shotId: string, takeId: string) {
    return request<TrashEntry>("DELETE", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}`);
  }
  trashShot(shotId: string) {
    return request<TrashEntry>("DELETE", `/api/shots/${enc(shotId)}`);
  }
  trash() {
    return request<TrashEntry[]>("GET", "/api/trash");
  }
  restoreTrash(trashId: string) {
    return request<{ entry: TrashEntry; shot?: Shot }>("POST", `/api/trash/${enc(trashId)}/restore`);
  }
  emptyTrash() {
    return request<EmptyTrashResult>("DELETE", "/api/trash");
  }
  storage() {
    return request<StorageInfo>("GET", "/api/storage");
  }
  saveFolder() {
    return request<SaveFolder>("GET", "/api/save-folder");
  }
  setSaveFolder(path: string) {
    return request<{ path: string; shown?: string | null }>("POST", "/api/save-folder", { path });
  }
  chooseSaveFolder() {
    return request<{ path: string | null }>("POST", "/api/save-folder/choose");
  }
  async revealSaveFolder() {
    await request<unknown>("POST", "/api/save-folder/reveal");
  }
  async revealMedia(url: string) {
    await request<unknown>("POST", "/api/reveal", { url });
  }
  async revealLibrary() {
    await request<unknown>("POST", "/api/storage/reveal");
  }
  addCast(projectId: string, file: File, fields: { name: string; kind: CastKind; line: string }) {
    return uploadMultipart<CastMember>(`/api/projects/${enc(projectId)}/cast`, file, undefined, fields);
  }
  patchCast(projectId: string, castId: string, patch: { name?: string; kind?: CastKind; line?: string }) {
    return request<CastMember>("PATCH", `/api/projects/${enc(projectId)}/cast/${enc(castId)}`, patch);
  }
  async buildSheet(projectId: string, req: SheetRequest) {
    try {
      return await request<Job>("POST", `/api/projects/${enc(projectId)}/cast/sheet`, req);
    } catch (e) {
      // A server from before sheets answers "No such API route" (404) or 405.
      if (e instanceof ApiError && (e.status === 405 || (e.status === 404 && /API route/.test(e.message)))) {
        throw new ApiError("This server can't build character sheets yet. Restart Pickups to update it.", e.status);
      }
      throw e;
    }
  }
  mentionsPreview(req: { project_id: string; text: string; uploads: number }) {
    return request<MentionPreview>("POST", "/api/mentions/preview", req);
  }
  deliveries(projectId: string) {
    return request<Delivery[]>("GET", `/api/projects/${enc(projectId)}/deliveries`);
  }
  finishApproved(projectId: string) {
    return request<{ queued: number; jobs: Job[] }>("POST", `/api/projects/${enc(projectId)}/finish_approved`);
  }
  exportTimeline(projectId: string, format: TimelineFormat) {
    return fetchFile(`/api/projects/${enc(projectId)}/export/${format}`, `timeline.${format}`, "export timelines");
  }
  exportReceipts(projectId: string) {
    return fetchFile(`/api/projects/${enc(projectId)}/receipts`, "receipts.zip", "export receipts");
  }
  takeReceiptUrl(shotId: string, takeId: string) {
    return takeId === "orig" ? null : `/api/shots/${enc(shotId)}/takes/${enc(takeId)}/receipt`;
  }
  takeWorkflowsUrl(shotId: string, takeId: string) {
    return takeId === "orig" ? null : `/api/shots/${enc(shotId)}/versions/${enc(takeId)}/workflows.zip`;
  }
  async shareTake(shotId: string, takeId: string, format: ShareFormat) {
    const r = await request<ShareResult>("POST", `/api/shots/${enc(shotId)}/takes/${enc(takeId)}/share`, { format });
    if (!r || typeof r !== "object" || typeof r.url !== "string") {
      throw new ApiError("This server can't make share versions yet. Restart Pickups to update it.", 404);
    }
    return r;
  }
  imageModels() {
    return request<ImageModels>("GET", "/api/image/models");
  }
  generateImages(projectId: string, req: ImageGenerateRequest) {
    return request<Job>("POST", `/api/projects/${enc(projectId)}/images/generate`, req);
  }
  generateVideo(projectId: string, req: VideoGenerateRequest) {
    return request<Job>("POST", `/api/projects/${enc(projectId)}/videos/generate`, req);
  }
  projectImages(projectId: string) {
    return request<ProjectImage[]>("GET", `/api/projects/${enc(projectId)}/images`);
  }
  imageToCast(projectId: string, imageId: string, fields: { name: string; kind: CastKind; line: string }) {
    return request<CastMember>("POST", `/api/projects/${enc(projectId)}/images/${enc(imageId)}/to_cast`, fields);
  }
  enhancePrompt(req: EnhanceRequest) {
    return request<EnhanceResult>("POST", "/api/prompt/enhance", req);
  }
  describeShot(shotId: string, req: DescribeRequest = {}) {
    return request<DescribeResult>("POST", `/api/shots/${enc(shotId)}/describe`, req);
  }
  detectPromptServers() {
    return request<{ found: DetectedServer[] }>("GET", "/api/prompt/detect");
  }
  testPromptServer(req: { url: string; model?: string }) {
    return request<PromptTestResult & Partial<DetectedServer>>("POST", "/api/prompt/test", req);
  }
  async settings() {
    const data = await request<unknown>("GET", "/api/settings");
    return (data && typeof data === "object" ? data : {}) as Settings;
  }
  saveSettings(patch: { director?: Partial<DirectorSettings> } & Record<string, unknown>) {
    return request<Settings>("POST", "/api/settings", patch);
  }
  machines() {
    return request<MachinesList>("GET", "/api/machines");
  }
  addMachine(req: AddMachineRequest) {
    return request<MachinesList>("POST", "/api/machines", req);
  }
  useMachine(id: string) {
    return request<MachinesList>("POST", `/api/machines/${enc(id)}/use`);
  }
  removeMachine(id: string) {
    return request<MachinesList>("DELETE", `/api/machines/${enc(id)}`);
  }
  shot(id: string) {
    return request<Shot>("GET", `/api/shots/${enc(id)}`);
  }
  select(shotId: string, req: SelectRequest) {
    return request<SelectResult>("POST", `/api/shots/${enc(shotId)}/select`, req);
  }
  track(shotId: string, req: TrackRequest) {
    return request<TrackResult>("POST", `/api/shots/${enc(shotId)}/track`, req);
  }
  people(shotId: string, req: PeopleRequest = {}) {
    return request<PeopleResult>("POST", `/api/shots/${enc(shotId)}/people`, req);
  }
  upload(file: File, onProgress?: (f: number) => void) {
    return uploadMultipart<UploadResult>("/api/uploads", file, onProgress);
  }
  refUrl(ref: Ref, projectId?: string) {
    if (ref.kind === "upload") return `/api/uploads/${enc(ref.id)}`;
    return `/api/projects/${enc(projectId || "unsorted")}/refs/${enc(ref.kind)}/${enc(ref.id)}`;
  }
  previzLook(req: PrevizLookRequest) {
    return request<PrevizLookResult>("POST", "/api/previz/look", req);
  }
  previzStartFrame(req: PrevizStartFrameRequest) {
    return request<PrevizStartFrameResult>("POST", "/api/previz/start_frame", req);
  }
  previzFrame0(shotId: string, body: { framing?: PrevizFraming | null; matte?: string | null; depth?: Record<string, string> | null }) {
    return request<Shot>("POST", `/api/shots/${enc(shotId)}/previz_frame0`, body);
  }
  async previzFrames(session: string, start: number, files: Blob[], signal?: AbortSignal) {
    const body = new FormData();
    body.set("start", String(start));
    files.forEach((blob, i) => body.append("files", blob, `${String(start + i).padStart(4, "0")}.jpg`));
    let res: Response;
    try {
      res = await fetch(`/api/previz/${enc(session)}/frames`, { method: "POST", body, signal, headers: { Accept: "application/json" } });
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new ApiError("Couldn't upload the frames. Try again.", 0);
    }
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!res.ok) {
      const detail = data && typeof data === "object" ? (data as { detail?: unknown }).detail : null;
      throw new ApiError(typeof detail === "string" && detail ? detail : "Couldn't upload the frames. Try again.", res.status, data);
    }
    return data as { received: number };
  }
  previzFinish(session: string, body: { spec: PrevizSpec; project_id?: string; name?: string; framing?: PrevizFraming | null; matte?: string | null; depth?: Record<string, string> | null }, signal?: AbortSignal) {
    return request<Shot>("POST", `/api/previz/${enc(session)}/finish`, body, signal);
  }
  previzFraming(req: { shot_id: string; look_frame_ref?: Ref; look_frame_upload_id?: string }) {
    return request<PrevizFramingResult>("POST", "/api/previz/framing", req);
  }
  async previzCancel(session: string) {
    await request<unknown>("DELETE", `/api/previz/${enc(session)}`);
  }
  createJob(req: JobRequest) {
    return request<Job>("POST", "/api/jobs", req);
  }
  jobs() {
    return request<Job[]>("GET", "/api/jobs");
  }
  cancelJob(id: string) {
    return request<Job | null>("POST", `/api/jobs/${enc(id)}/cancel`);
  }
  moveJobNext(id: string) {
    return request<Job>("POST", `/api/jobs/${enc(id)}/next`);
  }
  downloadModels() {
    return request<Status | null>("POST", "/api/models/download", {});
  }
  trimShot(shotId: string, seconds: number) {
    return request<Shot>("POST", `/api/shots/${enc(shotId)}/trim`, { seconds });
  }
  prepareShot(shotId: string, req: PrepareRequest) {
    return request<Shot>("POST", `/api/shots/${enc(shotId)}/prepare`, req);
  }

  subscribe(h: ApiEvents): () => void {
    let es: EventSource | null = null;
    let closed = false;
    let retry: number | undefined;
    const parse = <T,>(e: MessageEvent): T | null => {
      try {
        return JSON.parse(e.data) as T;
      } catch {
        return null;
      }
    };
    const open = () => {
      if (closed) return;
      es = new EventSource("/api/events");
      es.onopen = () => h.onOpen?.();
      es.addEventListener("job", (e) => {
        const job = parse<Job>(e as MessageEvent);
        if (job) h.onJob(job);
      });
      es.addEventListener("status", (e) => {
        const s = parse<Status>(e as MessageEvent);
        if (s) h.onStatus(s);
      });
      es.addEventListener("shot", (e) => {
        const s = parse<Shot>(e as MessageEvent);
        if (s) h.onShot?.(s);
      });
      es.addEventListener("shot_removed", (e) => {
        const r = parse<{ id: string; project_id?: string }>(e as MessageEvent);
        if (r) h.onShotRemoved?.(r);
      });
      es.addEventListener("project", (e) => {
        const p = parse<{ id: string; images?: string[] }>(e as MessageEvent);
        if (p) h.onProject?.(p);
      });
      es.onerror = () => {
        h.onError?.();
        // EventSource retries by itself while CONNECTING; if it gave up (CLOSED), reopen after a pause.
        if (es && es.readyState === EventSource.CLOSED) {
          es = null;
          retry = window.setTimeout(open, 3000);
        }
      };
    };
    open();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      es?.close();
    };
  }
}

// ---------------------------------------------------------------------------------------------------------------

/**
 * Picks the API: the mock when VITE_MOCK=1 (or `npm run dev:mock`), or — in dev only — when the backend isn't
 * answering /api/status. Dev also accepts `?mock=1` to force the mock.
 */
export async function createApi(): Promise<PickupsApi> {
  const params = new URLSearchParams(window.location.search);
  // ?mock=1 / ?mock=fresh / ?tour=1 work in every build (the tour and the demo video rely on it).
  const forced = import.meta.env.VITE_MOCK === "1" || params.has("mock") || params.has("tour");
  if (forced) return loadMock();
  if (import.meta.env.DEV) {
    const reachable = await backendReachable();
    if (!reachable) {
      console.info("[pickups] Backend at /api isn't answering; using the mock API. Start the backend on :7788 and reload.");
      return loadMock();
    }
  }
  return new HttpApi();
}

async function backendReachable(): Promise<boolean> {
  const ctrl = new AbortController();
  const t = window.setTimeout(() => ctrl.abort(), 2500);
  try {
    const res = await fetch("/api/status", { signal: ctrl.signal, headers: { Accept: "application/json" } });
    // When the backend is down, Vite's proxy answers 5xx with an empty / plain-text body.
    if (res.status >= 500 && res.status !== 503) return false;
    const type = res.headers.get("content-type") ?? "";
    return type.includes("json");
  } catch {
    return false;
  } finally {
    window.clearTimeout(t);
  }
}

async function loadMock(): Promise<PickupsApi> {
  const { MockApi } = await import("./mock");
  const m = new MockApi();
  await m.init(); // real example loops, when this machine has them
  return m;
}
