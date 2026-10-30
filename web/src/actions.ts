// Everything that talks to the API and changes app state.
import { ApiError, createApi, type PickupsApi } from "./api";
import { isActive, isOriginal, isTask, jobResultVersionId, size, takes, TASK_VERB, toEpochMs } from "./lib/format";
import { patchDraft } from "./lib/drafts";
import { DEFAULT_UI, getState, parseHash, type OnboardingStep, rememberChrome, rememberNav, rememberProject, setState, type Chrome, type CreateMode, type PrepareDraft, type ProjectTab, type Rail, type RenderWindow, type Route, type ShotUi, type Toast } from "./store";
import { DEFAULT_FINISH, isPreset, recipeById } from "./lib/recipes";
import { draftKey } from "./lib/draftStore";
import { projectAfterDelete } from "./lib/deleted";
import { preloadThumbs } from "./lib/thumbs";
import { servers, type ModelsHelp } from "./lib/servers";
import { TOUR } from "./lib/tour";
import { leftDemo } from "./lib/demo";
import { previzRendering, resetPrevizGate } from "./previz/home";
import type { CastKind, CropAspect, Delivery, DirectorSettings, Example, ImageGenerateRequest, ImageResolution, Job, JobRequest, Note, Point, Project, ProjectImage, Ref, SheetRequest, Shot, TaskId, Version, VideoGenerateRequest } from "./types";

let api: PickupsApi | null = null;
let lastEventAt = 0;

export function getApi(): PickupsApi {
  if (!api) throw new Error("API not ready");
  return api;
}

function message(e: unknown, fallback: string): string {
  if (e instanceof ApiError && e.message) return e.message;
  return fallback;
}

// --- boot -------------------------------------------------------------------------------------------------------

export async function boot(): Promise<void> {
  window.addEventListener("hashchange", () => {
    if (/^#\/library(\/|$)/.test(window.location.hash)) {
      window.location.hash = "#/projects";
      return;
    }
    const route = parseHash(window.location.hash);
    setState({ route, backToShot: backFrom(getState().route, route) });
    onRoute(route);
  });
  if (/^#\/library(\/|$)/.test(window.location.hash)) window.location.hash = "#/projects";

  api = await createApi();
  setState({ mode: api.mode });
  // First open (nothing connected yet): the welcome, then Connect; the same right after Exit demo. The mock has its own machine.
  if (api.mode === "live") {
    const fromDemo = leftDemo();
    void servers
      .list()
      .then((s) => {
        if (!s.welcome_done || fromDemo) openOnboarding("welcome");
      })
      .catch(() => {});
    void servers.downloadProgress().then((d) => d.running && followDownload(d)).catch(() => {});
  }
  if (TOUR && api.mode === "mock") void import("./tour/autopilot").then((m) => m.startTour());

  void refreshStatus();
  void loadSamples().then(loadExamples);
  void loadDefaultModels();
  void loadProjects().then(() => loadShots());
  void loadJobs();
  void loadDirector();

  api.subscribe({
    onOpen: () => {
      lastEventAt = Date.now();
    },
    onJob: (job) => {
      lastEventAt = Date.now();
      applyJob(job);
    },
    onStatus: (status) => {
      lastEventAt = Date.now();
      const stillMissing = Object.values(status.tiles ?? {}).some((t) => t && t.missing?.length);
      setState((s) => ({ status, statusFailed: false, modelsFetching: s.modelsFetching && stillMissing }));
    },
    onShot: (shot) => {
      lastEventAt = Date.now();
      putShot(shot);
      void loadProjects();
    },
    onShotRemoved: (r) => {
      lastEventAt = Date.now();
      dropShot(r.id);
      void loadProjects();
    },
    onProject: (p) => {
      lastEventAt = Date.now();
      const route = getState().route;
      if (getState().images[p.id] !== undefined || (route.name === "create" && route.panel === "image")) void loadImages(p.id);
      if (p.cast?.length) void ensureProject(p.id, true); // a character sheet joined the cast
      void loadProjects();
    },
  });

  // Safety net: if the event stream goes quiet while jobs run, poll.
  window.setInterval(() => {
    const active = Object.values(getState().jobs).some(isActive);
    if (active && Date.now() - lastEventAt > 5000) void loadJobs();
  }, 3000);

  onRoute(getState().route);
}

function onRoute(route: Route) {
  if (route.name === "shot") void ensureShot(route.id).then((shot) => shot?.project_id && void ensureProject(shot.project_id));
  // Viewing a project or a shot never changes the current project; only an explicit choice does (setCurrentProject).
  if (route.name === "project") {
    void ensureProject(route.id, true);
    if (route.tab === "deliveries") void loadDeliveries(route.id);
  }
  if (route.name === "projects") void loadProjects();
  if (route.name === "create") {
    void loadImageModels();
    if (route.panel === "image") {
      void loadImages(targetProject());
      void ensureProject(targetProject());
    }
  }
  // A link to a project's Previz (#/project/<id>/previz) is a choice of project: the shot it makes goes there.
  if (route.name === "previz" && route.projectId) {
    setCurrentProject(route.projectId);
    navigate({ name: "previz", projectId: null });
  }
  if (route.name === "project" && route.tab === "images") void loadImages(route.id);
}

export async function refreshStatus(): Promise<void> {
  try {
    const status = await getApi().status();
    setState({ status, statusFailed: false });
  } catch {
    setState({ statusFailed: true });
  }
}

export async function loadSamples(): Promise<void> {
  try {
    const samples = await getApi().samples();
    setState({ samples, samplesFailed: false });
  } catch {
    setState({ samples: [], samplesFailed: true });
  }
}

/** Home examples. Until the backend has /api/examples, they're made from the samples (poster only, no loop). */
export async function loadExamples(): Promise<void> {
  try {
    const examples = await getApi().examples();
    setState({ examples });
    if (!examples.length) void loadExamplesPack();
  } catch {
    const samples = getState().samples ?? [];
    const examples: Example[] = samples
      .filter((x) => x.try)
      .map((x) => ({
        id: `sample:${x.id}`,
        task: x.try!,
        title: TASK_VERB[x.try!],
        line: x.hint ?? "",
        slate: TASK_VERB[x.try!],
        sample_id: x.id,
        before_url: "",
        after_url: "",
        poster_url: x.thumb_url,
      }));
    const hero = examples.find((e) => e.task === "remove") ?? examples[0];
    if (hero) hero.hero = true;
    setState({ examples });
  }
}

/** Each workflow's default models, for the gallery's and task list's model chips (none on an older backend). */
export async function loadDefaultModels(): Promise<void> {
  try {
    setState({ defaultModels: await getApi().defaultModels() });
  } catch {
    setState({ defaultModels: null });
  }
}

/** A fresh install has no gallery: can it be downloaded? */
export async function loadExamplesPack(): Promise<void> {
  try {
    const pack = await getApi().examplesPack();
    setState({ examplesPack: { ...pack, busy: false } });
    // First run with a pack found (next to the app, Downloads, Desktop): it's installing in the background, or about to.
    // Ask again until it lands, then the gallery and the sample thumbnails load (a fresh install showed only two).
    if (pack.importing || (pack.found && !pack.installed && !pack.error)) {
      window.setTimeout(() => {
        void loadExamples();
        void loadSamples();
      }, 3000);
    }
  } catch {
    setState({ examplesPack: null });
  }
}

/** Setup → Examples: install a pack from a picked zip, or a zip or folder path on the Pickups machine. */
export async function importExamplesPack(src: File | string): Promise<boolean> {
  const cur = getState().examplesPack ?? { available: false, installed: false };
  setState({ examplesPack: { ...cur, busy: true, error: null } });
  try {
    const r = await getApi().importExamplesPack(src);
    setState({ examplesPack: { ...cur, installed: true, examples: r.examples, busy: false, error: null } });
    await loadExamples();
    await loadSamples();
    toast(`Examples pack installed: ${r.examples} examples.`, "success");
    return true;
  } catch (e) {
    setState({ examplesPack: { ...cur, busy: false, error: message(e, "That pack didn't install.") } });
    return false;
  }
}

/** "Get the examples": download the gallery, then show it. */
export async function installExamplesPack(): Promise<void> {
  const cur = getState().examplesPack;
  if (!cur || cur.busy) return;
  setState({ examplesPack: { ...cur, busy: true, error: null } });
  try {
    await getApi().installExamplesPack();
    setState({ examplesPack: { ...cur, installed: true, busy: false, error: null } });
    await loadExamples();
  } catch (e) {
    setState({ examplesPack: { ...cur, busy: false, error: message(e, "The examples didn't download. Try again.") } });
  }
}

export async function openExample(ex: Example): Promise<void> {
  if (ex.can_try === false) return;
  setState({ exampleOpening: ex.id });
  try {
    let shot: Shot;
    let task: TaskId = ex.task;
    let params: Record<string, unknown> | undefined;
    if (ex.id.startsWith("sample:")) {
      shot = await getApi().openSample(ex.sample_id);
    } else {
      try {
        const res = await getApi().openExample(ex.id);
        shot = res.shot;
        task = res.task ?? ex.task;
        params = res.params;
      } catch (e) {
        if (!(e instanceof ApiError) || (e.status !== 404 && e.status !== 405)) throw e;
        shot = await getApi().openSample(ex.sample_id);
      }
    }
    putShot(shot);
    // A copy of the example's clip is a new shot in your workspace: work there. An example's own shot (previz with its
    // cast) just opens; its forms read that shot's project.
    if (shot.id !== ex.shot_id && shot.project_id) setCurrentProject(shot.project_id, true);
    if (params && Object.keys(params).length) {
      // _from: the form labels kept references "From the example", not "From the last take".
      const pre = { ...params, _from: "example" };
      setState((s) => ({ prefill: { ...s.prefill, [shot.id]: { task, params: pre, seq: (s.prefill[shot.id]?.seq ?? 0) + 1 } } }));
    }
    openShot(shot.id, { rail: { kind: "form", task }, versionId: "orig" });
  } catch (e) {
    toast(message(e, "That example didn't open. Try again."), "error");
  } finally {
    setState({ exampleOpening: null });
  }
}

/** Open a file that's already on the backend machine (NLE bins, network shares). */
export async function openPath(path: string): Promise<boolean> {
  const p = path.trim();
  if (!p) return false;
  setState({ path: { busy: true, error: null } });
  try {
    const shot = await getApi().shotFromPath(p, dropTarget());
    putShot(shot);
    setCurrentProject(shot.project_id || "unsorted", true);
    setState({ path: { busy: false, error: null } });
    openImported(shot);
    return true;
  } catch (e) {
    const missing = e instanceof ApiError && (e.status === 404 || e.status === 405);
    setState({
      path: { busy: false, error: missing ? "This build can't open paths yet. Use Choose a clip, or drop the file." : message(e, "Pickups couldn't open that path.") },
    });
    return false;
  }
}

export function clearPathError(): void {
  if (getState().path.error) setState({ path: { busy: false, error: null } });
}

// --- recipes ---------------------------------------------------------------------------------------------------

/** Pick a recipe: the next clip opened runs its steps one after another. The form lives on Create. */
export function chooseRecipe(id: string | null): void {
  setState({ pendingRecipe: id });
  if (id && getState().route.name !== "create") navigate({ name: "create", panel: "home" });
}

/** Open the Create page on one of its four ways in, and hand it words or a dropped photo. */
export function setCreateMode(mode: CreateMode, text?: string, file?: File | null): void {
  setState((s) => ({ createBar: { mode, seq: s.createBar.seq + 1, text, file } }));
  const route = getState().route;
  if (route.name !== "create") navigate({ name: "create", panel: "home" });
}

/** Arm a preset without leaving the page (Create → Edit a clip). The next imported clip runs it. */
export function holdPreset(id: string): void {
  setState((s) => ({
    pendingRecipe: s.pendingRecipe === id ? null : id,
    upload: s.upload.state === "error" ? { state: "idle" } : s.upload,
  }));
}

/** A preset chip: wait for a clip (open one, or pick a shot), or for "New character", the bar's Character mode. */
export function startPreset(id: string, character = false): void {
  if (character) {
    setState({ pendingRecipe: null });
    setCreateMode("character");
    return;
  }
  const cur = getState().pendingRecipe;
  setState((s) => ({
    pendingRecipe: cur === id ? null : id,
    createBar: { mode: "clip", seq: s.createBar.seq + 1 },
    // A new start: last try's upload message goes.
    upload: s.upload.state === "error" ? { state: "idle" } : s.upload,
  }));
  if (getState().route.name !== "create") navigate({ name: "create", panel: "home" });
}

/** A new cast member's name from a few words: "a green alien in a silver flight suit" → "Green alien". Never one the
 * cast has already (the backend refuses a duplicate): "Green alien 2". */
export function characterName(words: string, taken: string[]): string {
  const stop = /^(in|with|wearing|who|on|at|from|and|holding|that|carrying|under|by|of)$/i;
  const parts: string[] = [];
  for (const w of words.replace(/[.,;:!?"“”()]/g, " ").trim().split(/\s+/)) {
    if (!w) continue;
    if (!parts.length && /^(a|an|the|my|our|one)$/i.test(w)) continue;
    if (stop.test(w) || parts.length >= 3) break;
    parts.push(w.toLowerCase());
  }
  const base = parts.length ? parts.join(" ").replace(/^./, (c) => c.toUpperCase()).slice(0, 32) : "New character";
  const used = new Set(taken.map((t) => t.toLowerCase()));
  if (!used.has(base.toLowerCase())) return base;
  for (let n = 2; n < 100; n++) if (!used.has(`${base} ${n}`.toLowerCase())) return `${base} ${n}`;
  return `${base} ${Date.now() % 1000}`;
}

/** The Create bar's Character mode: one photo or a few words become a character sheet (front, back, close-up) in the
 * current project's cast, the same request as the cast's "Add a character". Lands on the Cast tab, where the sheet
 * builds. Returns an error message, or null. */
export async function buildCharacter(req: { file?: File | null; describe?: string; name: string }): Promise<string | null> {
  const pid = targetProject();
  const name = req.name.trim().replace(/\s+/g, " ");
  if (!name) return "Give them a name.";
  try {
    const sheet: SheetRequest = req.file ? { upload_id: (await getApi().upload(req.file)).id, name } : { describe: (req.describe ?? "").trim(), name };
    await buildSheet(pid, sheet);
    navigate({ name: "project", id: pid, tab: "cast" });
    void ensureProject(pid, true);
    return null;
  } catch (e) {
    return message(e, "The sheet didn't start. Try again.");
  }
}

/** The Create bar's Image mode: start the images, then show them on the Image page with the same settings (so the
 * next try is one click). Without a ready model it just hands the prompt over: the Image page says what's missing. */
export async function generateFromHome(req: { prompt: string; aspect: string; model?: string; resolution?: ImageResolution }): Promise<void> {
  const s = getState();
  const catalog = s.imageModels;
  const model = catalog?.models.find((m) => m.id === (req.model || catalog.default)) ?? null;
  const resolution: ImageResolution = req.resolution === "2K" && model && model.max_side >= 2048 ? "2K" : "1K";
  const patch: Record<string, unknown> = { prompt: req.prompt, aspect: req.aspect, resolution, ...(model ? { model: model.id } : {}) };
  if (model && model.ready !== false && req.prompt.trim()) {
    const job = await generateImages(targetProject(), { model: model.id, prompt: req.prompt.trim(), aspect: req.aspect, resolution, count: 1, refs: [] });
    if (!job) return;
  }
  setState((st) => ({ imageDraft: { seq: (st.imageDraft?.seq ?? 0) + 1, patch } }));
  navigate({ name: "create", panel: "image" });
}

// --- the left panel -------------------------------------------------------------------------------------------

export function setNavCollapsed(collapsed: boolean): void {
  rememberNav(collapsed);
  setState((s) => ({ nav: { ...s.nav, collapsed } }));
}

export function setNavDrawer(drawer: boolean): void {
  setState((s) => (s.nav.drawer === drawer ? {} : { nav: { ...s.nav, drawer } }));
}

/** Side panel or top bar. The choice sticks in this browser so the two can be compared. */
export function setChrome(chrome: Chrome): void {
  rememberChrome(chrome);
  setState((s) => ({ chrome, nav: { ...s.nav, drawer: false } }));
}

export function stopRecipe(shotId: string): void {
  setState((s) => {
    const runs = { ...s.recipeRuns };
    delete runs[shotId];
    return { recipeRuns: runs };
  });
}

/** The current project's shots (Home's "Your shots"). */
export async function loadShots(projectId = getState().currentProjectId): Promise<void> {
  try {
    const list = await getApi().shots(projectId);
    preloadThumbs(list.map((x) => x.latest_thumb_url || x.thumb_url));
    setState((s) => {
      const shots = { ...s.shots };
      for (const shot of list) shots[shot.id] = shot;
      return { shots, projectShots: { ...s.projectShots, [projectId]: list.map((x) => x.id) }, shotsFailed: false };
    });
  } catch {
    setState((s) => ({ projectShots: { ...s.projectShots, [projectId]: s.projectShots[projectId] ?? [] }, shotsFailed: true }));
  }
}

// --- projects -------------------------------------------------------------------------------------------------

export async function loadProjects(): Promise<void> {
  try {
    const projects = await getApi().projects();
    preloadThumbs(projects.map((p) => p.cover_url));
    setState((s) => {
      // The remembered project may be gone (or be Examples, which is never a target). With nothing remembered, start
      // in the most recent project that has shots, else My shots.
      const ok = projects.some((p) => p.id === s.currentProjectId && p.id !== "examples" && p.id !== "qc");
      if (ok) return { projects, projectsFailed: false };
      const recent = s.currentProjectId ? null : projects.find((p) => p.id !== "examples" && !p.archived && p.shots > 0);
      const id = recent?.id ?? "unsorted";
      rememberProject(id);
      return { projects, projectsFailed: false, currentProjectId: id };
    });
  } catch {
    setState((s) => ({ projects: s.projects ?? [], projectsFailed: true }));
  }
}

/** Shots, cast and counts for one project. */
export async function ensureProject(id: string, force = false): Promise<void> {
  const s = getState();
  if (!force && s.cast[id] && s.projectShots[id]) return;
  if (!api) return;
  try {
    const detail = await getApi().project(id);
    preloadThumbs([...detail.shots.map((x) => x.latest_thumb_url || x.thumb_url), ...detail.cast.map((c) => c.thumb_url || c.url)]);
    setState((st) => {
      const shots = { ...st.shots };
      for (const shot of detail.shots) shots[shot.id] = shot;
      const projects = st.projects ? st.projects.map((p) => (p.id === id ? detail.project : p)) : st.projects;
      return {
        shots,
        projects,
        projectShots: { ...st.projectShots, [id]: detail.shots.map((x) => x.id) },
        cast: { ...st.cast, [id]: detail.cast },
      };
    });
  } catch {
    setState((st) => ({ projectShots: { ...st.projectShots, [id]: st.projectShots[id] ?? [] } }));
  }
}

/**
 * The project new work goes into. It changes only by an explicit choice: the switcher, the Projects page, a new
 * project, a new shot added to a project. `announce`: say so when it changed (for flows that switch as a side effect).
 */
export function setCurrentProject(id: string, announce = false): void {
  // Examples and the archived Test runs project (QC runs) are never where new work goes.
  if (id === "examples" || id === "qc") return;
  const changed = id !== getState().currentProjectId;
  rememberProject(id);
  setState({ currentProjectId: id });
  void loadShots(id);
  if (announce && changed) {
    const name = getState().projects?.find((p) => p.id === id)?.name ?? (id === "unsorted" ? "My shots" : null);
    if (name) toast(`Now working in ${name}.`);
  }
}

/** Opening a project from the switcher or the Projects page makes it current (a link to it doesn't). */
export function openProject(id: string, tab: ProjectTab = "shots"): void {
  if (id !== "examples") setCurrentProject(id);
  navigate({ name: "project", id, tab });
}

export async function createProject(name: string, open = true): Promise<boolean> {
  return !!(await newProject(name, open));
}

/** Make a project; `open` goes to it (and makes it current). Returns it, or null (a toast says why). */
export async function newProject(name: string, open = true): Promise<Project | null> {
  try {
    const p = await getApi().createProject(name);
    setState((s) => ({ projects: [p, ...(s.projects ?? [])], projectShots: { ...s.projectShots, [p.id]: s.projectShots[p.id] ?? [] } }));
    if (open) openProject(p.id);
    void loadProjects();
    return p;
  } catch (e) {
    toast(message(e, "That project wasn't created. Try again."), "error");
    return null;
  }
}

/** "Move to → New project…": make the project, then move the shot into it. */
export async function moveShotToNewProject(shotId: string, name: string): Promise<boolean> {
  const p = await newProject(name, false);
  if (!p) return false;
  await moveShot(shotId, p.id);
  return true;
}

export async function renameProject(id: string, name: string): Promise<void> {
  try {
    const p = await getApi().renameProject(id, name);
    setState((s) => ({ projects: (s.projects ?? []).map((x) => (x.id === id ? { ...x, ...p } : x)) }));
  } catch (e) {
    toast(message(e, "Couldn't rename the project."), "error");
  }
}

/** Archive (or bring back) a project. Archived projects leave the switcher; nothing is deleted. */
export async function archiveProject(id: string, archived: boolean): Promise<void> {
  try {
    const p = await getApi().archiveProject(id, archived);
    setState((s) => ({ projects: (s.projects ?? []).map((x) => (x.id === id ? { ...x, ...p, archived } : x)) }));
    if (archived && getState().currentProjectId === id) setCurrentProject("unsorted");
    toast(archived ? `Archived ${p.name}.` : `${p.name} is back.`, "info", archived ? { label: "Undo", run: () => void archiveProject(id, false) } : undefined);
  } catch (e) {
    toast(message(e, archived ? "Couldn't archive the project." : "Couldn't bring the project back."), "error");
  }
}

/** Delete a project: its folder and shots go to the Trash (Undo, or Restore there). If it was current, work moves to
 * the most recent project left, else the empty My shots the server makes; there's always a project. */
export async function deleteProject(id: string): Promise<boolean> {
  // Read before the call: its shot_removed events can land first, and loadProjects then falls back to My shots.
  const moved = getState().currentProjectId === id;
  const gone = new Set(getState().projectShots[id] ?? []);
  try {
    const { entry, projects } = await getApi().trashProject(id);
    const next = projectAfterDelete(projects, id, moved ? id : getState().currentProjectId);
    setState((st) => ({
      projects,
      shots: Object.fromEntries(Object.entries(st.shots).filter(([k]) => !gone.has(k))),
      projectShots: Object.fromEntries(Object.entries(st.projectShots).filter(([k]) => k !== id)),
      cast: Object.fromEntries(Object.entries(st.cast).filter(([k]) => k !== id)),
      images: Object.fromEntries(Object.entries(st.images).filter(([k]) => k !== id)),
    }));
    if (moved) setCurrentProject(next);
    const route = getState().route;
    if ((route.name === "project" && route.id === id) || (route.name === "shot" && gone.has(route.id))) navigate({ name: "projects" });
    if (getState().trash) void loadTrash();
    // One toast, so its Undo stays: where work goes now rides along.
    const now = moved ? ` Now working in ${projects.find((p) => p.id === next)?.name ?? "My shots"}.` : "";
    toast(`Moved ${entry.name} to Trash.${now}`, "info", { label: "Undo", run: () => void restoreTrash(entry.id) });
    return true;
  } catch (e) {
    toast(message(e, "Couldn't delete the project. Try again."), "error");
    return false;
  }
}

/** Delete a cast entry or a project image: its files go to the Trash. Shots that used it keep their takes; a new take
 * just goes without it. */
export async function deleteAsset(projectId: string, kind: "cast" | "image", id: string): Promise<boolean> {
  try {
    const entry = await getApi().trashAsset(projectId, kind, id);
    setState((s) =>
      kind === "cast"
        ? { cast: { ...s.cast, [projectId]: (s.cast[projectId] ?? []).filter((c) => c.id !== id) } }
        : { images: { ...s.images, [projectId]: (s.images[projectId] ?? []).filter((im) => im.id !== id) } },
    );
    void loadProjects();
    if (getState().trash) void loadTrash();
    toast(`Moved ${entry.name} to Trash.`, "info", { label: "Undo", run: () => void restoreTrash(entry.id) });
    return true;
  } catch (e) {
    toast(message(e, "Couldn't delete that. Try again."), "error");
    return false;
  }
}

/** Rename a shot (breadcrumb menu). */
export async function renameShot(id: string, name: string): Promise<void> {
  const n = name.trim();
  if (!n || getState().shots[id]?.name === n) return;
  try {
    const shot = await getApi().patchShot(id, { name: n });
    putShot(shot);
  } catch (e) {
    toast(message(e, "Couldn't rename the shot."), "error");
  }
}

/** Move a shot (and its takes) to another project. */
export async function moveShot(id: string, projectId: string): Promise<void> {
  const from = getState().shots[id]?.project_id;
  if (!from || from === projectId) return;
  try {
    const shot = await getApi().patchShot(id, { project_id: projectId });
    setState((s) => {
      const projectShots = { ...s.projectShots };
      if (projectShots[from]) projectShots[from] = projectShots[from]!.filter((x) => x !== id);
      if (projectShots[projectId]) projectShots[projectId] = [id, ...projectShots[projectId]!.filter((x) => x !== id)];
      return { shots: { ...s.shots, [id]: shot }, projectShots };
    });
    void loadProjects();
    const name = getState().projects?.find((p) => p.id === projectId)?.name ?? "the project";
    toast(`Moved to ${name}.`, "info", { label: "Undo", run: () => void moveShot(id, from) });
  } catch (e) {
    toast(message(e, "Couldn't move the shot."), "error");
  }
}

/** Reuse a cast entry from another project (a copy). Returns an error message, or null. */
export async function importCast(projectId: string, fromProjectId: string, castId: string): Promise<string | null> {
  try {
    const c = await getApi().importCast(projectId, fromProjectId, castId);
    setState((s) => ({ cast: { ...s.cast, [projectId]: [...(s.cast[projectId] ?? []), { ...c, uses: 0 }] } }));
    void loadProjects();
    toast(`${c.name} joined the cast.`);
    return null;
  } catch (e) {
    return message(e, "That didn't come across. Try again.");
  }
}

/** Where a new shot goes: the current project, never Examples. */
export function targetProject(): string {
  const id = getState().currentProjectId;
  return id && id !== "examples" ? id : "unsorted";
}

/** Open Previz on its own page. */
let previzSentence: string | null = null;
let previzSpecHanded: unknown = null;

/** Previz, optionally with a sentence in the composer (Remake this shot: the old shot's own sentence, ready to Build).
 * With the shot's saved spec too, a camera path the user drew comes back with it (the editor opens on that spec). */
export function openPreviz(sentence?: string, spec?: unknown): void {
  previzSentence = typeof sentence === "string" ? sentence.trim() || null : null;
  previzSpecHanded = spec ?? null;
  // Bare (the nav, Home "New previz"): always the scene grid, a new scene. Only a Make shot still rendering keeps its scene.
  if (previzSentence == null && previzSpecHanded == null && !previzRendering()) resetPrevizGate();
  navigate({ name: "previz", projectId: null });
}

/** The saved spec openPreviz was given, once. */
export function takePrevizSpec(): unknown {
  const s = previzSpecHanded;
  previzSpecHanded = null;
  return s;
}

/** The sentence openPreviz was given, once. */
export function takePrevizSentence(): string | null {
  const s = previzSentence;
  previzSentence = null;
  return s;
}

/** A shot the server changed (a refreshed frame 0): the store's copy follows. */
export function updateShot(shot: Shot): void {
  putShot(shot);
}

/** Previz's "New scene": back to the scene grid with an empty sentence. */
export function newPrevizScene(): void {
  resetPrevizGate();
}

/** A layout just became a shot: open it on Previz to final, with the layout's notes as the motion prompt and the
 * sentence's character (and any cast it names) as the Look prompt. */
export function openMadePreviz(shot: Shot, prompt: string, lookPrompt = ""): void {
  putShot(shot);
  const params: Record<string, string> = {};
  // prompt_auto: the app wrote this line, so the form rebuilds it from the cast and Look until the user types in it.
  if (prompt.trim()) Object.assign(params, { prompt: prompt.trim(), prompt_auto: "1" });
  if (lookPrompt.trim()) params.look_prompt = lookPrompt.trim();
  const apply = () =>
    setState((s) => ({
      prefill: { ...s.prefill, [shot.id]: { task: "previz" as const, params, seq: (s.prefill[shot.id]?.seq ?? 0) + 1 } },
    }));
  apply();
  openShot(shot.id, { rail: { kind: "form", task: "previz" } });
  toast("Previz ready → Previz to final", "success", {
    label: "Previz to final",
    run: () => {
      // The form already opened once: its draft keeps any words the user typed, so only reopen it.
      if (!getState().drafts[draftKey(shot.id, "previz")]) apply();
      openShot(shot.id, { rail: { kind: "form", task: "previz" } });
    },
  });
  void loadProjects();
}

/** Where a dropped or opened file goes: the project page you're looking at, else the current project. */
function dropTarget(): string {
  const r = getState().route;
  return r.name === "project" && r.id !== "examples" ? r.id : targetProject();
}

// --- cast ----------------------------------------------------------------------------------------------------

export async function addCast(projectId: string, file: File, fields: { name: string; kind: CastKind; line: string }): Promise<string | null> {
  try {
    const c = await getApi().addCast(projectId, file, fields);
    setState((s) => ({ cast: { ...s.cast, [projectId]: [...(s.cast[projectId] ?? []), c] } }));
    void loadProjects();
    return null;
  } catch (e) {
    return message(e, "That didn't save. Try again.");
  }
}

export async function updateCast(projectId: string, castId: string, patch: { name?: string; kind?: CastKind; line?: string }): Promise<string | null> {
  try {
    const c = await getApi().patchCast(projectId, castId, patch);
    setState((s) => ({ cast: { ...s.cast, [projectId]: (s.cast[projectId] ?? []).map((x) => (x.id === castId ? c : x)) } }));
    return null;
  } catch (e) {
    return message(e, "That didn't save. Try again.");
  }
}

// --- character sheets (pickups/sheets.py) ------------------------------------------------------------------------

/** Sheet jobs a form started for itself (Recast's "Build a full sheet"): the form shows the result, so no toast. */
const formSheetJobs = new Set<string>();

/** Queue a character sheet: a new cast member from a picture or words, or `cast_id` upgraded / made again (a new
 * seed). `quiet`: the caller follows the job itself. Throws ApiError when it can't start. */
export async function buildSheet(projectId: string, req: SheetRequest, quiet = false): Promise<Job> {
  const job = await getApi().buildSheet(projectId, req);
  if (quiet) formSheetJobs.add(job.id);
  applyJob(job);
  return job;
}

/** The name a sheet job is for: the request's (a new member) or the member it upgrades. */
export function sheetName(job: Job): string {
  return (job.params as SheetRequest | undefined)?.name || "Your character";
}

async function onSheetDone(job: Job): Promise<void> {
  const pid = job.project_id || targetProject();
  const quiet = formSheetJobs.delete(job.id);
  if (job.status === "done") await ensureProject(pid, true);
  void loadProjects();
  if (quiet) {
    dismissJob(job.id);
    return;
  }
  const r = getState().route;
  const onCast = r.name === "project" && r.id === pid && r.tab === "cast";
  if (job.status === "done" && !onCast) {
    toast(`${sheetName(job)}'s sheet is ready`, "success", { label: "View", run: () => navigate({ name: "project", id: pid, tab: "cast" }) }, 8000);
  } else if (job.status === "failed") {
    // The Cast grid keeps the failed card with Try again (REVIEW F2); the toast says it in plain words and goes there.
    toast(`${sheetName(job)}'s sheet didn't build.`, "error", { label: "Try again", run: () => navigate({ name: "project", id: pid, tab: "cast" }) }, 8000);
  }
}

// --- takes: approve, hide, reveal, deliveries -------------------------------------------------------------------

export async function setApproved(shotId: string, takeId: string, approved: boolean): Promise<void> {
  try {
    const shot = await getApi().patchTake(shotId, takeId, { status: approved ? "approved" : "" });
    putShot(shot);
    void loadProjects();
    if (shot.project_id && getState().deliveries[shot.project_id]) void loadDeliveries(shot.project_id);
  } catch (e) {
    toast(message(e, "Couldn't change the approval. Try again."), "error");
  }
}

/** Deliveries' "Use this": approve a take's finished child (its Finish 4K HDR) in place of the take, so the edit and the
 * receipts get the master. Undo puts the approval back where it was. */
export async function approveFinishedTake(shotId: string, takeId: string, finishedId: string, undo = false): Promise<void> {
  try {
    await getApi().patchTake(shotId, undo ? takeId : finishedId, { status: "approved" });
    const shot = await getApi().patchTake(shotId, undo ? finishedId : takeId, { status: "" });
    putShot(shot);
    void loadProjects();
    if (shot.project_id) void loadDeliveries(shot.project_id);
    if (undo) return;
    const tk = takes(shot);
    toast(`Take ${tk[finishedId]?.num ?? ""} is approved in place of Take ${tk[takeId]?.num ?? ""}.`, "success", {
      label: "Undo",
      run: () => void approveFinishedTake(shotId, takeId, finishedId, true),
    });
  } catch (e) {
    toast(message(e, "Couldn't change the approval. Try again."), "error");
  }
}

export function toggleApproved(shotId: string, takeId: string): void {
  const v = getState().shots[shotId]?.versions.find((x) => x.id === takeId);
  // The original is the footage. Approving it would list an untouched clip in Deliveries.
  if (v && !isOriginal(v)) void setApproved(shotId, takeId, v.status !== "approved");
}

export async function setHidden(shotId: string, takeId: string, hidden: boolean): Promise<void> {
  if (takeId === "orig") return;
  try {
    const shot = await getApi().patchTake(shotId, takeId, { hidden });
    putShot(shot);
    const ui = getUi(shotId);
    if (hidden && ui.versionId === takeId && !ui.showHidden) {
      const v = shot.versions.find((x) => x.id === takeId);
      selectVersion(shotId, v?.input_version && shot.versions.some((x) => x.id === v.input_version && !x.hidden) ? v.input_version : "orig");
    }
    toast(hidden ? "Take hidden." : "Take shown.", "info", hidden ? { label: "Undo", run: () => void setHidden(shotId, takeId, false) } : undefined);
  } catch (e) {
    toast(message(e, "Couldn't hide that take. Try again."), "error");
  }
}

// --- trash: takes and shots go to library/trash with an Undo; Empty Trash removes them from this computer -----------

/** A shot left the library (trashed here or in another tab): out of every list, and off its page. */
function dropShot(shotId: string): void {
  setState((s) => {
    const shots = { ...s.shots };
    delete shots[shotId];
    const projectShots: typeof s.projectShots = {};
    for (const [pid, ids] of Object.entries(s.projectShots)) projectShots[pid] = ids ? ids.filter((x) => x !== shotId) : ids;
    return { shots, projectShots };
  });
  const route = getState().route;
  if (route.name === "shot" && route.id === shotId) navigate({ name: "project", id: getState().currentProjectId ?? "unsorted", tab: "shots" });
}

export async function trashTake(shotId: string, takeId: string): Promise<void> {
  if (takeId === "orig") return;
  try {
    const entry = await getApi().trashTake(shotId, takeId);
    const shot = getState().shots[shotId];
    if (shot) {
      const v = shot.versions.find((x) => x.id === takeId);
      const ui = getUi(shotId);
      if (ui.versionId === takeId) {
        selectVersion(shotId, v?.input_version && shot.versions.some((x) => x.id === v.input_version && x.id !== takeId) ? v.input_version : "orig");
      }
      putShot({ ...shot, versions: shot.versions.filter((x) => x.id !== takeId) });
    }
    if (getState().trash) void loadTrash();
    toast("Moved to Trash.", "info", { label: "Undo", run: () => void restoreTrash(entry.id) });
  } catch (e) {
    toast(message(e, "Couldn't move that take to the trash. Try again."), "error");
  }
}

export async function trashShot(shotId: string): Promise<void> {
  try {
    const entry = await getApi().trashShot(shotId);
    dropShot(shotId);
    void loadProjects();
    if (getState().trash) void loadTrash();
    toast("Moved to Trash.", "info", { label: "Undo", run: () => void restoreTrash(entry.id) });
  } catch (e) {
    toast(message(e, "Couldn't move that shot to the trash. Try again."), "error");
  }
}

export async function loadTrash(): Promise<void> {
  try {
    const trash = await getApi().trash();
    setState({ trash });
  } catch {
    setState((s) => ({ trash: s.trash ?? [] }));
  }
}

export async function restoreTrash(trashId: string): Promise<void> {
  try {
    const { entry, shot } = await getApi().restoreTrash(trashId);
    if (shot) putShot(shot);
    await loadProjects();
    // A project, cast entry or image: its project's shots, cast and images again.
    if (entry.kind === "project" || entry.kind === "cast" || entry.kind === "image") {
      await ensureProject(entry.project_id, true);
      void loadImages(entry.project_id);
      void loadShots(entry.project_id);
    }
    setState((s) => ({ trash: s.trash ? s.trash.filter((t) => t.id !== trashId) : s.trash }));
    toast("Restored.");
  } catch (e) {
    toast(message(e, "Couldn't restore that. Try again."), "error");
  }
}

export async function emptyTrash(): Promise<void> {
  try {
    const r = await getApi().emptyTrash();
    setState({ trash: [] });
    const remote = r.remote_files ? ` ${r.remote_files} ${r.remote_files === 1 ? "file stays" : "files stay"} on the GPU machine.` : "";
    toast(`Removed ${size(r.bytes)}.${remote}`, "success");
  } catch (e) {
    toast(message(e, "Couldn't empty the trash. Try again."), "error");
  }
}

export async function revealLibrary(): Promise<void> {
  try {
    await getApi().revealLibrary();
  } catch (e) {
    toast(message(e, "Couldn't open the folder."), "error");
  }
}

/** Reveal in Finder (Show in Explorer on Windows): the words for this computer. */
export function revealLabel(): string {
  return /win/i.test(navigator.platform) ? "Show in Explorer" : "Reveal in Finder";
}

export async function revealMedia(url: string): Promise<void> {
  try {
    await getApi().revealMedia(url);
  } catch (e) {
    toast(message(e, "Couldn't show that file."), "error");
  }
}

export async function revealTake(shotId: string, takeId: string): Promise<void> {
  try {
    await getApi().revealTake(shotId, takeId);
    toast("Shown in Finder");
  } catch (e) {
    toast(message(e, "Couldn't open the folder."), "error");
  }
}

// --- notes: timecoded, on any take (the original too); they travel into the exported timeline as markers ---------

function patchNotes(shotId: string, takeId: string, fn: (notes: Note[]) => Note[]): void {
  setState((s) => {
    const shot = s.shots[shotId];
    if (!shot) return {};
    const versions = shot.versions.map((v) => (v.id === takeId ? { ...v, notes: fn(v.notes ?? []) } : v));
    return { shots: { ...s.shots, [shotId]: { ...shot, versions } } };
  });
}

/** Shows at once (pending), then swaps in the saved note; taken back out if the save fails. */
export async function addNote(shotId: string, takeId: string, frame: number, text: string): Promise<void> {
  const clean = text.trim();
  if (!clean) return;
  const tmp: Note = { id: `tmp_${Date.now().toString(36)}`, frame: Math.max(0, Math.round(frame)), text: clean };
  patchNotes(shotId, takeId, (notes) => [...notes, tmp]);
  try {
    const saved = await getApi().addNote(shotId, takeId, { frame: tmp.frame, text: clean });
    patchNotes(shotId, takeId, (notes) => [...notes.filter((n) => n.id !== tmp.id && n.id !== saved.id), saved]);
  } catch (e) {
    patchNotes(shotId, takeId, (notes) => notes.filter((n) => n.id !== tmp.id));
    toast(message(e, "That note didn't save. Try again."), "error");
  }
}

export async function deleteNote(shotId: string, takeId: string, note: Note): Promise<void> {
  if (note.id.startsWith("tmp_")) return;
  patchNotes(shotId, takeId, (notes) => notes.filter((n) => n.id !== note.id));
  try {
    await getApi().deleteNote(shotId, takeId, note.id);
    toast("Note deleted.", "info", { label: "Undo", run: () => void addNote(shotId, takeId, note.frame, note.text) });
  } catch (e) {
    patchNotes(shotId, takeId, (notes) => (notes.some((n) => n.id === note.id) ? notes : [...notes, note]));
    toast(message(e, "Couldn't delete that note. Try again."), "error");
  }
}

export async function loadDeliveries(projectId: string): Promise<void> {
  if (!api) return;
  try {
    const list = await getApi().deliveries(projectId);
    // Warm the cards' stills so Deliveries opens on pictures, not empty frames.
    preloadThumbs(list.map((d) => d.take.thumb_url));
    setState((s) => ({ deliveries: { ...s.deliveries, [projectId]: list } }));
  } catch {
    setState((s) => ({ deliveries: { ...s.deliveries, [projectId]: s.deliveries[projectId] ?? [] } }));
  }
}

/** "Finish all approved": the takes it queues, as the backend picks them (not Finish takes, not finished or finishing). */
export function finishCandidates(list: Delivery[], shots: Record<string, Shot>, jobs: Record<string, Job>): Delivery[] {
  const skip = new Set<string>();
  for (const j of Object.values(jobs)) if (j.task === "finish" && isActive(j)) skip.add(`${j.shot_id}:${j.version_id || "orig"}`);
  for (const sid of new Set(list.map((d) => d.shot_id)))
    for (const v of shots[sid]?.versions ?? []) if (v.task === "finish" && !v.hidden) skip.add(`${sid}:${v.input_version || "orig"}`);
  return list.filter((d) => d.take.task !== "finish" && !skip.has(`${d.shot_id}:${d.take.id}`));
}

export async function finishApproved(projectId: string): Promise<void> {
  try {
    const res = await getApi().finishApproved(projectId);
    for (const j of res.jobs ?? []) applyJob(j);
    // Nothing is finished yet: the toast says "Queued", and each take says "ready" when it lands.
    if (res.queued) toast(`Finish 4K HDR for ${res.queued} approved ${res.queued === 1 ? "take" : "takes"}.`, "queued");
    else toast("Every approved take is finished or finishing.", "info");
  } catch (e) {
    toast(message(e, "Couldn't queue the finishes. Try again."), "error");
  }
}

/** "Try again with changes": reopen the task's form on the same input, filled with what this take ran with. */
export function tryAgainWithChanges(shotId: string, take: Version): void {
  const task = take.task as TaskId;
  // _take and _reference_url: Recast shows the take's own reference still and can use it again (params.reference_take).
  const params = { ...(take.params ?? {}), _from: "take", _take: take.id, ...(take.reference_url ? { _reference_url: take.reference_url } : {}) };
  setState((s) => ({ prefill: { ...s.prefill, [shotId]: { task, params, seq: (s.prefill[shotId]?.seq ?? 0) + 1 } } }));
  setUi(shotId, { versionId: take.input_version || "orig" });
  setRail(shotId, { kind: "form", task });
}

export function clearPrefill(shotId: string): void {
  if (getState().prefill[shotId]) setState((s) => ({ prefill: { ...s.prefill, [shotId]: undefined } }));
}

/** Re-run a take's selection (points on a frame, or words) so its form starts where it left off. */
export function restoreSelection(shotId: string, r: { points?: Point[]; frame?: number; text?: string }, from = "the last take"): void {
  const sel = getState().selection;
  if (!sel || sel.shotId !== shotId) return;
  const points = r.points ?? [];
  if (!points.length && !r.text) return;
  // Selected as it was, without a find: a find runs when the pick changes (a click, new words) or on "Show".
  setState({
    selection: { ...sel, points, frame: r.frame ?? 0, text: r.text ?? "", maskUrl: null, loading: false, error: null, seq: sel.seq + 1, kept: from, auto: undefined, autoBase: undefined },
  });
}

/** Recast's people finder: select someone it found (their click points on its frame). Their matte comes with them, so
 * the viewer shows who will be replaced without another find; without one it is found like a click. The pick is
 * `auto` until the user changes it: a click on someone else in the picture then switches to them (see addPoint). */
export function pickPerson(shotId: string, person: { frame: number; points: Point[]; maskUrl?: string | null }): void {
  const sel = getState().selection;
  if (!sel || sel.shotId !== shotId || !person.points.length) return;
  const next = { ...sel, frame: person.frame, points: person.points, text: "", kept: undefined, auto: true, autoBase: person.points.length };
  if (person.maskUrl) setState({ selection: { ...next, maskUrl: person.maskUrl, loading: false, error: null, seq: sel.seq + 1 } });
  else void runSelect(next);
}

/** "Show": find a kept pick on its frame so its mask draws over the picture (it stays the same pick). */
export function showKeptSelection(): void {
  const sel = getState().selection;
  if (sel?.kept && !sel.loading) void runSelect(sel, true);
}

export async function loadJobs(): Promise<void> {
  try {
    const list = await getApi().jobs();
    // Apply oldest first so the newest ends up first in jobOrder (the API order isn't specified).
    const t = (j: Job) => toEpochMs(j.created_at) ?? 0;
    const oldestFirst = list.some((j) => j.created_at != null) ? [...list].sort((a, b) => t(a) - t(b)) : [...list].reverse();
    for (const j of oldestFirst) applyJob(j, true);
  } catch {
    /* the queue just stays as it is */
  }
}

function putShot(shot: Shot): void {
  preloadThumbs([shot.latest_thumb_url, shot.thumb_url, ...shot.versions.map((v) => v.thumb_url)]);
  setState((s) => {
    const errors = { ...s.shotErrors };
    delete errors[shot.id];
    const projectShots = { ...s.projectShots };
    const pid = shot.project_id;
    if (pid && projectShots[pid]) {
      const rest = projectShots[pid]!.filter((x) => x !== shot.id);
      projectShots[pid] = listed(shot) ? [shot.id, ...rest] : rest;
    }
    return { shots: { ...s.shots, [shot.id]: shot }, projectShots, shotErrors: errors };
  });
}

/** The Examples project lists only shots with something to compare (a take that isn't hidden), as the backend does. */
function listed(shot: Shot): boolean {
  return shot.project_id !== "examples" || shot.versions.some((v) => v.id !== "orig" && !v.hidden);
}

export async function ensureShot(id: string, force = false): Promise<Shot | null> {
  const s = getState();
  if (s.shots[id] && !force) return s.shots[id];
  if (!api) return null;
  try {
    const shot = await getApi().shot(id);
    setState((st) => ({ shots: { ...st.shots, [id]: shot }, shotErrors: { ...st.shotErrors, [id]: "" } }));
    return shot;
  } catch (e) {
    const notFound = e instanceof ApiError && e.status === 404;
    setState((st) => ({
      shotErrors: { ...st.shotErrors, [id]: notFound ? "missing" : "failed" },
    }));
    return null;
  }
}

// --- navigation -------------------------------------------------------------------------------------------------

export function navigate(route: Route): void {
  const hash =
    route.name === "shot"
      ? `#/shot/${encodeURIComponent(route.id)}`
      : route.name === "project"
        ? `#/project/${encodeURIComponent(route.id)}${route.tab === "shots" ? "" : `/${route.tab}`}`
        : route.name === "projects"
          ? "#/projects"
          : route.name === "create"
            ? route.panel === "image"
              ? "#/create/image"
              : route.panel === "edit"
                ? "#/create/edit"
                : "#/create"
            : route.name === "assets"
              ? "#/assets"
            : route.name === "previz"
              ? route.projectId
                ? `#/project/${encodeURIComponent(route.projectId)}/previz`
                : "#/previz"
              : "#/";
  const from = getState().route;
  if (window.location.hash !== hash) window.location.hash = hash;
  setState({ route, backToShot: backFrom(from, route) });
}

/** Leaving a shot for another page keeps the way back to it; moving between that page's tabs keeps it; Home or the
 * shot itself clears it. */
function backFrom(from: Route, to: Route): string | null {
  if (to.name === "shot" || to.name === "home") return null;
  if (from.name === "shot") return from.id;
  return from.name === to.name ? getState().backToShot : null;
}

/** Back to the shot this page was opened from, its panel as it was. */
export function backToShot(): boolean {
  const id = getState().backToShot;
  if (!id || !getState().shots[id]) return false;
  navigate({ name: "shot", id });
  return true;
}

export function getUi(shotId: string): ShotUi {
  return getState().ui[shotId] ?? DEFAULT_UI;
}

export function setUi(shotId: string, patch: Partial<ShotUi>): void {
  setState((s) => ({ ui: { ...s.ui, [shotId]: { ...(s.ui[shotId] ?? DEFAULT_UI), ...patch } } }));
}

export function setRail(shotId: string, rail: Rail): void {
  const sel = getState().selection;
  if (sel && sel.shotId === shotId && rail.kind !== "form") setState({ selection: null });
  setUi(shotId, { rail });
}

/** A job's live progress on its shot: a step flow's own Render step (its draft goes back to it), else the running
 * panel. The take strip's running chip and a task tile with a job under way both open it. */
export function openJob(shotId: string, job: Job): void {
  if (job.draft_key && job.step === "render" && isTask(job.task)) {
    patchDraft(job.draft_key, (d) => ({ step: "render", jobs: { ...d.jobs, render: job.id } }));
    setRail(shotId, { kind: "form", task: job.task });
  } else setRail(shotId, { kind: "running", jobId: job.id });
}

/** The newest job under way for this shot and task, if any. */
export function activeJob(shotId: string, task: TaskId): Job | undefined {
  const s = getState();
  return s.jobOrder.map((id) => s.jobs[id]).find((j) => j && j.shot_id === shotId && j.task === task && isActive(j));
}

/** Open a shot. With a version, the viewer shows it (and its result card).
 * A preset waiting on Home applies only when `applyPreset` is set (the "start on" row, or a clip just imported).
 * Any other open — a project, the sidebar, the queue — drops the preset so it can't land on the wrong shot. */
export function openShot(id: string, opts: { versionId?: string; rail?: Rail; applyPreset?: boolean } = {}): void {
  const s = getState();
  const pending = recipeById(s.pendingRecipe);
  if (pending && opts.applyPreset) {
    // A chain runs its steps one after another (the recipe strip follows it); a one-step preset just opens its form,
    // prefilled with the preset's settings. An example that couldn't be opened is "the example", not "the preset".
    const chain = !isPreset(pending);
    const task = pending.tasks[0];
    const from = pending.id.startsWith("example:") ? "example" : "preset";
    setState((st) => ({
      pendingRecipe: null,
      recipeRuns: chain ? { ...st.recipeRuns, [id]: { recipeId: pending.id, step: 0 } } : st.recipeRuns,
      prefill:
        pending.params && Object.keys(pending.params).length
          ? { ...st.prefill, [id]: { task, params: { ...pending.params, _from: from }, seq: (st.prefill[id]?.seq ?? 0) + 1 } }
          : st.prefill,
    }));
    opts = { ...opts, rail: { kind: "form", task } };
  } else if (s.pendingRecipe) {
    setState({ pendingRecipe: null });
  }
  if (!s.ui[id]) {
    // Coming back to a shot with work in progress shows that work.
    const active = s.jobOrder.map((j) => s.jobs[j]).find((j) => j && j.shot_id === id && isActive(j));
    if (active) openJob(id, active);
    else setUi(id, { rail: { kind: "tiles" } });
  }
  if (opts.versionId) {
    const isOrig = opts.versionId === "orig";
    setUi(id, {
      versionId: opts.versionId,
      rail: opts.rail ?? (isOrig ? { kind: "tiles" } : { kind: "result", versionId: opts.versionId }),
      view: "after",
      wipe: true,
      wipePos: 50,
    });
  } else if (opts.rail) {
    setUi(id, { rail: opts.rail });
  }
  navigate({ name: "shot", id });
  void ensureShot(id);
}

export function openSetup(focus: TaskId | null = null, section?: string): void {
  // From the phone drawer too: the dialog comes up in front, so the drawer closes.
  setState((s) => ({ setup: { open: true, focus, section }, nav: { ...s.nav, drawer: false } }));
}

export function closeSetup(): void {
  setState({ setup: { open: false, focus: null } });
}

// --- toasts -------------------------------------------------------------------------------------------------------

let toastId = 1;
// Each toast's timer: paused while the pointer is on it (holdToast), so its action (Undo) can't vanish under the cursor.
const toastTimers = new Map<number, { timer: number; left: number; since: number }>();

function armToast(id: number, ms: number): void {
  toastTimers.set(id, { timer: window.setTimeout(() => dismissToast(id), ms), left: ms, since: Date.now() });
}

/** Toasts go away after 8 s. Hovering pauses the timer so an Undo click isn't lost. */
export function toast(text: string, tone: Toast["tone"] = "info", action?: Toast["action"], ms = 8000): void {
  const id = toastId++;
  setState({ toasts: [{ id, text, tone, action }] }); // one at a time: a new toast replaces the last
  armToast(id, ms);
}

/** Pause (hover) or resume a toast's timer; it resumes with what it had left, at least 2 s. */
export function holdToast(id: number, hold: boolean): void {
  const t = toastTimers.get(id);
  if (!t || hold === (t.timer === 0)) return; // timer 0: already held
  window.clearTimeout(t.timer);
  if (hold) toastTimers.set(id, { timer: 0, left: Math.max(2000, t.left - (Date.now() - t.since)), since: 0 });
  else armToast(id, t.left);
}

export function dismissToast(id: number): void {
  window.clearTimeout(toastTimers.get(id)?.timer);
  toastTimers.delete(id);
  setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}

// --- shots ---------------------------------------------------------------------------------------------------------

const VIDEO_EXT = /\.(mov|mp4|m4v|mkv|webm|avi|mxf)$/i;

export async function uploadShotFile(file: File): Promise<void> {
  // Progress shows in the drop actions on Home and on the project page.
  const r = getState().route;
  const projectId = dropTarget();
  const stay =
    r.name === "assets" || (r.name === "create" && r.panel === "edit");
  if (!stay && r.name !== "home" && !(r.name === "project" && r.tab === "shots")) navigate({ name: "home" });
  if (!file.type.startsWith("video/") && !VIDEO_EXT.test(file.name)) {
    setState({ upload: { state: "error", name: file.name, error: "That's not a video Pickups can read." } });
    return;
  }
  setState({ upload: { state: "uploading", name: file.name, progress: 0 } });
  try {
    const shot = await getApi().uploadShot(
      file,
      (p) => {
        setState({ upload: { state: p >= 1 ? "processing" : "uploading", name: file.name, progress: p } });
      },
      projectId,
    );
    putShot(shot);
    setCurrentProject(shot.project_id || projectId, true);
    setState({ upload: { state: "idle" } });
    openImported(shot);
  } catch (e) {
    setState({ upload: { state: "error", name: file.name, error: message(e, "That file didn't upload.") } });
  }
}

export function resetUpload(): void {
  setState({ upload: { state: "idle" } });
}

/** A picture dropped on Assets. It is saved with the current project so it stays after a refresh. */
export async function importImageFile(file: File): Promise<void> {
  if (!file.type.startsWith("image/") && !/\.(png|jpe?g|webp|gif)$/i.test(file.name)) {
    toast("Drop a video or an image.", "error");
    return;
  }
  const pid = targetProject();
  const base = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim().slice(0, 32) || "Imported image";
  let err = await addCast(pid, file, { name: base, kind: "look", line: "Imported" });
  if (err && /already a cast entry/i.test(err)) {
    err = await addCast(pid, file, { name: `${base.slice(0, 28)} 2`.trim(), kind: "look", line: "Imported" });
  }
  if (err) toast(err, "error");
}

export async function openSample(id: string): Promise<void> {
  setState({ sampleOpening: id });
  try {
    const shot = await getApi().openSample(id);
    putShot(shot);
    if (shot.project_id) setCurrentProject(shot.project_id, true);
    openShot(shot.id);
  } catch (e) {
    toast(message(e, "That sample didn't open. Try again."), "error");
  } finally {
    setState({ sampleOpening: null });
  }
}

// --- trim & crop ---------------------------------------------------------------------------------------------------

/** A clip just added: Trim & crop first (Skip keeps it as it is). A preset waiting for a clip opens its form instead. */
/** A new clip opens on its tasks (Trim & crop is a row above them, not a step in the way). */
function openImported(shot: Shot): void {
  if (getState().pendingRecipe) openShot(shot.id, { applyPreset: true });
  else openShot(shot.id, { versionId: "orig" });
}

/** The largest `aspect` box inside a w×h frame, centred (source pixels, even). */
export function fullCrop(w: number, h: number, aspect: CropAspect): NonNullable<PrepareDraft["box"]> {
  const r = cropRatio(aspect);
  const bw = w / h > r ? h * r : w;
  const bh = bw / r;
  const ew = Math.max(2, Math.floor(bw / 2) * 2);
  const eh = Math.max(2, Math.floor(bh / 2) * 2);
  return { x: Math.floor((w - ew) / 4) * 2, y: Math.floor((h - eh) / 4) * 2, w: ew, h: eh };
}

export function cropRatio(aspect: CropAspect): number {
  const [a, b] = aspect.split(":").map(Number);
  return a / b;
}

/** Open Trim & crop on a shot: the whole clip, its own shape, on the original.
 * `fromMenu` is the shot menu: the second button says Cancel. A new clip still says Skip. */
export function openPrepare(shotId: string, opts?: { fromMenu?: boolean; task?: TaskId }): void {
  const shot = getState().shots[shotId];
  // A shot still loading: the rail starts its draft once the shot is here.
  setState({
    prepare: shot
      ? { shotId, inFrame: 0, outFrame: Math.max(1, shot.frames) - 1, aspect: "original", box: null, busy: false, error: null, fromMenu: !!opts?.fromMenu, task: opts?.task }
      : null,
  });
  openShot(shotId, { versionId: "orig", rail: { kind: "prepare" } });
}

export function setPrepare(patch: Partial<PrepareDraft>): void {
  const p = getState().prepare;
  if (p) setState({ prepare: { ...p, ...patch, error: patch.error ?? null } });
}

/** Keep the clip as it is: back to the tasks. */
export function skipPrepare(shotId: string): void {
  const task = getState().prepare?.task;
  setState({ prepare: null });
  setRail(shotId, task ? { kind: "form", task } : { kind: "tiles" });
}

/** Make the new clip from the draft, then open it (the original stays in the project). */
export async function makePreparedClip(): Promise<void> {
  const p = getState().prepare;
  const shot = p ? getState().shots[p.shotId] : null;
  if (!p || !shot || p.busy) return;
  setPrepare({ busy: true });
  try {
    const crop = p.aspect !== "original" && p.box ? { aspect: p.aspect, ...p.box } : null;
    const made = await getApi().prepareShot(shot.id, { in_frame: p.inFrame, out_frame: p.outFrame, crop });
    putShot(made);
    // From a task form: the new clip opens on the same form. Nothing is carried over: an example's frames and clicks
    // belong to the old clip's timing, not the trimmed one's.
    setState({ prepare: null });
    setRail(shot.id, { kind: "tiles" });
    openShot(made.id, { rail: p.task ? { kind: "form", task: p.task } : { kind: "tiles" } });
    toast(`Made ${made.name}. The original is still in the project.`, "success");
  } catch (e) {
    const cur = getState().prepare;
    if (cur) setState({ prepare: { ...cur, busy: false, error: message(e, "Couldn't make the new clip. Try again.") } });
  }
}

export async function trimShot(shotId: string, seconds = 4): Promise<void> {
  try {
    const shot = await getApi().trimShot(shotId, seconds);
    putShot(shot);
    setUi(shotId, { trimDismissed: true });
  } catch (e) {
    toast(message(e, "Trimming didn't work. You can still use the whole shot."), "error");
  }
}

// --- selection (click-to-select) -------------------------------------------------------------------------------

export function startSelection(shotId: string): void {
  const sel = getState().selection;
  if (sel && sel.shotId === shotId) return;
  setState({ selection: { shotId, frame: 0, points: [], text: "", maskUrl: null, loading: false, error: null, seq: 0 } });
}

export function endSelection(): void {
  setState({ selection: null });
}

export function clearSelection(): void {
  const sel = getState().selection;
  if (!sel) return;
  setState({ selection: { ...sel, points: [], text: "", maskUrl: null, loading: false, error: null, seq: sel.seq + 1, kept: undefined, auto: undefined, autoBase: undefined } });
}

/** Recast's window on the timeline (null: none). The form places it; the filmstrip moves it. */
export function setRenderWindow(w: RenderWindow | null): void {
  const cur = getState().renderWindow;
  if (cur === w || (cur && w && cur.shotId === w.shotId && cur.start === w.start && cur.frames === w.frames && cur.moved === w.moved && cur.fixed === w.fixed && cur.sized === w.sized)) return;
  setState({ renderWindow: w });
}

/** The user dragged Recast's window, or one of its ends (`frames`): from now on the form sends its start and length. */
export function moveRenderWindow(start: number, frames?: number): void {
  const w = getState().renderWindow;
  if (!w || w.fixed) return;
  setRenderWindow({ ...w, start, frames: frames ?? w.frames, moved: true, sized: w.sized || (frames != null && frames !== w.frames) });
}

/** Recast's people finder is looking (true) or done: meanwhile the picture's chip says so. */
export function setSelectionFinding(shotId: string, finding: boolean): void {
  const sel = getState().selection;
  if (!sel || sel.shotId !== shotId || !!sel.finding === finding) return;
  setState({ selection: { ...sel, finding } });
}

/** Recast anyone: whether a click on someone else the finder found swaps them too (rather than switching to them). */
export function setSelectionMany(shotId: string, many: boolean): void {
  const sel = getState().selection;
  if (!sel || sel.shotId !== shotId || !!sel.many === many) return;
  setState({ selection: { ...sel, many } });
}

/** Where a click lands while the finder's pick is untouched: "switched" (on someone else it found: the form picked them),
 * "same" (on the person already picked: the click adds to them) or "none" (nobody it found: the click starts over). */
export type AutoClick = (point: Point, frame: number) => "switched" | "same" | "none";
let autoClick: AutoClick | null = null;

/** The Recast form registers this while its people finder has an answer (null when it closes). */
export function setAutoClick(fn: AutoClick | null): void {
  autoClick = fn;
}

export function addPoint(point: Point, frame: number): void {
  const sel = getState().selection;
  if (!sel) return;
  if (sel.auto && point.positive) {
    // The finder picked for the user: a click on someone else means "them instead", never "them as well".
    const hit = autoClick?.(point, frame) ?? "none";
    if (hit === "switched") return;
    if (hit === "none") {
      void runSelect({ ...sel, frame, points: [point], text: "", auto: false, autoBase: undefined });
      return;
    }
  }
  // Points belong to one frame; clicking on another frame starts over there.
  const restart = !!sel.points.length && sel.frame !== frame;
  const points = restart ? [point] : [...sel.points, point];
  void runSelect({ ...sel, frame, points, auto: false, autoBase: restart ? undefined : sel.autoBase });
}

export function undoPoint(): void {
  const sel = getState().selection;
  if (!sel || !sel.points.length) return;
  const points = sel.points.slice(0, -1);
  if (!points.length && !sel.text) {
    clearSelection();
    return;
  }
  // Back to just the finder's points: the finder's pick again.
  const base = sel.autoBase && points.length >= sel.autoBase ? sel.autoBase : undefined;
  void runSelect({ ...sel, points, auto: !!base && points.length === base && !sel.text, autoBase: base });
}

export function selectByText(text: string, frame: number): void {
  const sel = getState().selection;
  if (!sel) return;
  const t = text.trim();
  if (!t && !sel.text) return; // no words before, none now: nothing changes
  // Words take over from the finder: its points go; the user's own clicks stay with the words.
  const points = sel.auto ? [] : sel.points.slice(sel.autoBase ?? 0);
  if (!t && !points.length) {
    clearSelection();
    return;
  }
  void runSelect({ ...sel, text: t, points, frame: points.length ? sel.frame : frame, auto: false, autoBase: undefined });
}

async function runSelect(next: NonNullable<ReturnType<typeof getState>["selection"]>, keepPick = false): Promise<void> {
  const seq = next.seq + 1;
  // A changed pick is no longer the kept one; "Show" keeps it (still usable if the find fails) until its mask arrives.
  setState({ selection: { ...next, seq, loading: true, error: null, kept: keepPick ? next.kept : undefined } });
  try {
    const res = await getApi().select(next.shotId, {
      frame: next.frame,
      points: next.points,
      ...(next.text ? { text: next.text } : {}),
    });
    const cur = getState().selection;
    if (!cur || cur.seq !== seq) return; // a newer click won
    setState({ selection: { ...cur, maskUrl: res.mask_url, loading: false, kept: undefined } });
  } catch (e) {
    const cur = getState().selection;
    if (!cur || cur.seq !== seq) return;
    setState({ selection: { ...cur, loading: false, error: message(e, "Couldn't find that. Try clicking again.") } });
  }
}

// --- jobs ------------------------------------------------------------------------------------------------------

function applyJob(job: Job, fromList = false): void {
  const prev = getState().jobs[job.id];
  setState((s) => ({
    jobs: {
      ...s.jobs,
      [job.id]: { ...prev, ...job, _lastStage: job.stages?.includes(job.stage) ? job.stage : prev?._lastStage },
    },
    jobOrder: s.jobOrder.includes(job.id) ? s.jobOrder : [job.id, ...s.jobOrder],
    // Jobs that were already finished when the app opened don't need to sit in the queue bar.
    dismissed: fromList && !prev && !isActive(job) ? { ...s.dismissed, [job.id]: true } : s.dismissed,
  }));
  const finished = prev && isActive(prev) && !isActive(job);
  // An inline row (the start frame, the look frame) makes no take: the page that asked for it has the answer.
  if (job.lane === "inline") return;
  if (finished && job.task === "sheet") {
    void onSheetDone(job);
  } else if (finished && job.status === "done") {
    if (job.task === "image") void onImagesDone(job);
    else if (job.task === "video") void onVideoDone(job);
    else void onJobDone(job);
  } else if (!isActive(job) && formImageJobs.delete(job.id)) {
    dismissJob(job.id); // a form's still that failed or was dropped: the form says so
  }
}

async function onJobDone(job: Job): Promise<void> {
  const shot = await ensureShot(job.shot_id, true);
  if (!shot) return;
  const wanted = jobResultVersionId(job);
  const version = shot.versions.find((v) => v.id === wanted && v.id !== "orig") ?? shot.versions[shot.versions.length - 1];
  if (!version || version.id === "orig") return;
  const s = getState();
  const ui = s.ui[shot.id];
  const watching = ui?.rail.kind === "running" && ui.rail.jobId === job.id;
  if (watching) {
    setUi(shot.id, { rail: { kind: "result", versionId: version.id }, versionId: version.id, view: "after", wipe: true, wipePos: 50 });
  }
  if (await continueRecipe(job, shot, version.id, watching)) return;
  const onThisShot = s.route.name === "shot" && s.route.id === shot.id;
  // A step flow showing this job (its Render step) moves on to Review by itself: no toast over it.
  const inFlow = ui?.rail.kind === "form" && ui.rail.task === job.task && s.drafts[`${shot.id}:${job.task}`]?.jobs?.render === job.id;
  if (!((watching || inFlow) && onThisShot)) {
    // The app's take naming ("Take 03 · Finish 4K HDR"), as on the take strip and Deliveries.
    const t = takes(shot)[version.id];
    toast(`${t ? `Take ${t.num} is ready · ${t.label}` : `${TASK_VERB[job.task]} is ready`} · ${shot.name}`, "success", {
      label: "Open",
      run: () => openShot(shot.id, { versionId: version.id }),
    }, 12000);
  }
}

/** After a recipe step finishes: run Finish by itself, or open the next step's form on the new take. */
async function continueRecipe(job: Job, shot: Shot, versionId: string, watching: boolean): Promise<boolean> {
  const step = getState().jobSteps[job.id];
  const recipe = recipeById(step?.recipeId);
  if (!step || !recipe) return false;
  const next = step.step + 1;
  if (next >= recipe.tasks.length) {
    stopRecipe(shot.id);
    toast(`${recipe.title} · done on ${shot.name}`, "success", { label: "View", run: () => openShot(shot.id, { versionId }) }, 8000);
    return true;
  }
  setState((s) => ({ recipeRuns: { ...s.recipeRuns, [shot.id]: { recipeId: recipe.id, step: next } } }));
  const task = recipe.tasks[next];
  if (task === "finish") {
    await startJob({ shot_id: shot.id, version_id: versionId, task, params: DEFAULT_FINISH }, watching);
    return true;
  }
  setUi(shot.id, { versionId });
  if (watching) setRail(shot.id, { kind: "form", task });
  toast(`Next step: ${TASK_VERB[task]} · ${shot.name}`, "info", {
    label: "Go",
    run: () => openShot(shot.id, { versionId, rail: { kind: "form", task } }),
  }, 8000);
  return true;
}

export async function startJob(req: JobRequest, showInRail = true): Promise<boolean> {
  return !!(await submitJob(req, showInRail));
}

/** A run just went on the GPU queue: say so (yellow), with how many are ahead, and a way to see the queue. `always`:
 * the page doesn't show the run itself (Create's images and videos), so even one starting now gets the line. */
function confirmQueued(job: Job, what: string, always: boolean): void {
  const s = getState();
  const ahead = s.jobOrder.filter((id) => {
    const j = s.jobs[id];
    return j && j.id !== job.id && j.lane !== "inline" && (j.status === "running" || j.status === "queued");
  }).length;
  if (!ahead && !always) return;
  toast(`Queued · ${what} · ${ahead ? `${ahead} ahead` : "starting now"}`, "queued", { label: "See queue", run: () => setState({ queueOpen: true }) }, 6000);
}

/** Queue a job and hand it back (null when it didn't start; the toast says why). A step flow keeps its own rail. */
export async function submitJob(req: JobRequest, showInRail = true): Promise<Job | null> {
  try {
    const job = await getApi().createJob(req);
    const run = getState().recipeRuns[req.shot_id];
    const recipe = recipeById(run?.recipeId);
    setState((s) => ({
      requests: { ...s.requests, [job.id]: req },
      jobSteps: run && recipe && recipe.tasks[run.step] === req.task ? { ...s.jobSteps, [job.id]: run } : s.jobSteps,
    }));
    applyJob(job);
    if (showInRail) setRail(req.shot_id, { kind: "running", jobId: job.id });
    confirmQueued(job, TASK_VERB[req.task], false);
    return job;
  } catch (e) {
    toast(message(e, `${TASK_VERB[req.task]} didn't start. Try again.`), "error");
    return null;
  }
}

export async function moveJobNext(id: string): Promise<void> {
  try {
    applyJob(await getApi().moveJobNext(id));
    toast("Moved next in the queue.", "queued");
  } catch (e) {
    toast(message(e, "Couldn't move it. Try again."), "error");
  }
}

export async function cancelJob(id: string): Promise<void> {
  try {
    const job = await getApi().cancelJob(id);
    if (job) applyJob(job);
  } catch (e) {
    toast(message(e, "Couldn't cancel. Try again."), "error");
  }
}

/** The request that made a job, if we know it (sent this session, or echoed back by the backend). */
export function requestFor(job: Job): JobRequest | null {
  const s = getState();
  if (s.requests[job.id]) return s.requests[job.id];
  if (job.lane === "inline") return null; // not a queued job: its page runs it again
  if (job.params && job.task !== "image" && job.task !== "sheet" && job.task !== "video") {
    const input = job.result_version_id !== undefined ? job.version_id : undefined;
    return { shot_id: job.shot_id, task: job.task, params: job.params, ...(input && input !== "orig" ? { version_id: input } : {}) };
  }
  return null;
}

export function jobForVersion(versionId: string): Job | null {
  const s = getState();
  for (const id of s.jobOrder) {
    const j = s.jobs[id];
    if (j && j.status === "done" && jobResultVersionId(j) === versionId) return j;
  }
  return null;
}

/** "Try again": resubmit the same request (the backend picks a new variation), else reopen the form. */
export async function retry(job: Job | null, shotId: string, task: TaskId, versionId?: string): Promise<void> {
  const req = job ? requestFor(job) : null;
  if (req) {
    await startJob(req);
    return;
  }
  if (versionId) setUi(shotId, { versionId });
  setRail(shotId, { kind: "form", task });
}

export function dismissJob(id: string): void {
  setState((s) => ({ dismissed: { ...s.dismissed, [id]: true } }));
}

// --- setup -------------------------------------------------------------------------------------------------------

export async function connectComfy(url: string): Promise<boolean> {
  try {
    const status = await getApi().connect(url);
    setState({ status, statusFailed: false });
    return status.connected;
  } catch {
    return false;
  }
}

/** The one way into the guided setup: Connect (first run, Change server, the offline notice), Scan (Setup's Check
 * setup), or Choose with one feature's card open and ticked (a feature that isn't ready). */
export function openOnboarding(step: OnboardingStep | "welcome" = "connect", opts: { task?: string | null } = {}): void {
  const at: OnboardingStep = step === "welcome" ? "connect" : step;
  setState((s) => ({ onboarding: { open: true, step: at, task: opts.task ?? null }, setup: { open: false, focus: null }, nav: { ...s.nav, drawer: false } }));
}

export function closeOnboarding(): void {
  setState((s) => ({ onboarding: { ...s.onboarding, open: false } }));
  void servers.welcomeDone().catch(() => {});
  void loadExamples(); // Home shows what's installed now (a pack may have landed while setup was open)
}

/** Get one feature's missing models (or all of them). On this computer they download here with progress; for a
 * ComfyUI on another computer, Setup shows the command to run there and the file links. */
export async function downloadModels(task?: TaskId | "image"): Promise<void> {
  if (getState().mode !== "live") {
    setState({ modelsFetching: true });
    try {
      const status = await getApi().downloadModels();
      if (status) setState({ status });
    } catch (e) {
      setState({ modelsFetching: false });
      toast(message(e, "The download didn't start. Try again."), "error", undefined, 7000);
    }
    return;
  }
  setState({ modelsHelp: null });
  try {
    followDownload(await servers.download(task));
  } catch (e) {
    const detail = e instanceof ApiError ? (e.body as { detail?: unknown } | null)?.detail : null;
    if (detail && typeof detail === "object" && "message" in detail) setState({ modelsHelp: detail as ModelsHelp });
    else toast(message(e, "The download didn't start. Try again."), "error", undefined, 7000);
  }
}

let downloadTimer = 0;

/** Poll a running download until it finishes, then refresh what's installed. */
function followDownload(d: import("./lib/servers").ModelsDownload): void {
  setState({ modelsDownload: d, modelsFetching: d.running });
  window.clearTimeout(downloadTimer);
  if (!d.running) {
    if (d.error) toast(`The download stopped: ${d.error}`, "error", undefined, 9000);
    void refreshStatus();
    return;
  }
  downloadTimer = window.setTimeout(() => {
    void servers
      .downloadProgress()
      .then(followDownload)
      .catch(() => setState({ modelsFetching: false, modelsDownload: { ...d, running: false, error: "Pickups lost track of it." } }));
  }, 2000);
}

// --- versions ------------------------------------------------------------------------------------------------------

/** Show a version. A finished take opens its result, even from a form. The original stays a form's base. */
export function selectVersion(shotId: string, versionId: string): void {
  const ui = getUi(shotId);
  const version = getState().shots[shotId]?.versions.find((v) => v.id === versionId);
  if (ui.rail.kind === "form" && version && !isOriginal(version)) {
    clearSelection();
    setUi(shotId, { versionId, rail: { kind: "result", versionId }, view: "after", wipe: true, wipePos: 50 });
    return;
  }
  if (ui.rail.kind === "form") {
    if (ui.versionId !== versionId) {
      clearSelection();
      setUi(shotId, { versionId });
    }
    return;
  }
  setUi(shotId, {
    versionId,
    rail: versionId === "orig" ? { kind: "tiles" } : { kind: "result", versionId },
    view: "after",
    wipe: true,
    wipePos: 50,
  });
}

/** Escape: closes Setup, the queue or the open form. On the task list it returns to the project. */
/** The panel's Back: a form on a finished take goes back to that take, anything else to the task list. */
export function railBack(id: string): void {
  const ui = getUi(id);
  const version = getState().shots[id]?.versions.find((v) => v.id === ui.versionId);
  if (ui.rail.kind === "form" && version && !isOriginal(version)) setRail(id, { kind: "result", versionId: version.id });
  else setRail(id, { kind: "tiles" });
}

export function goBack(): boolean {
  const s = getState();
  if (s.setup.open) {
    closeSetup();
    return true;
  }
  if (s.queueOpen) {
    setState({ queueOpen: false });
    return true;
  }
  if (s.route.name !== "shot" && backToShot()) return true;
  if (s.route.name === "shot") {
    const id = s.route.id;
    if (getUi(id).rail.kind !== "tiles") {
      railBack(id);
      return true;
    }
    const projectId = s.shots[id]?.project_id;
    if (projectId) {
      navigate({ name: "project", id: projectId, tab: "shots" });
      return true;
    }
  }
  return false;
}

/** Reveal in Finder only makes sense when the browser runs on the server machine. */
export function isLocalHost(): boolean {
  return ["localhost", "127.0.0.1", "[::1]", "::1"].includes(window.location.hostname);
}


// --- images (docs/images-spec.md) ----------------------------------------------------------------------------------

export async function loadImageModels(force = false): Promise<void> {
  if (!api || (getState().imageModels && !force)) return;
  try {
    const imageModels = await getApi().imageModels();
    setState({ imageModels, imageModelsFailed: false });
  } catch {
    setState({ imageModelsFailed: true });
  }
}

export async function loadImages(projectId: string): Promise<void> {
  if (!api) return;
  try {
    const list = await getApi().projectImages(projectId);
    setState((s) => ({ images: { ...s.images, [projectId]: list } }));
  } catch {
    setState((s) => ({ images: { ...s.images, [projectId]: s.images[projectId] ?? [] } }));
  }
}

export async function generateImages(projectId: string, req: ImageGenerateRequest): Promise<Job | null> {
  try {
    const job = await getApi().generateImages(projectId, req);
    applyJob(job);
    confirmQueued(job, "New image", true);
    return job;
  } catch (e) {
    toast(message(e, "The images didn't start. Try again."), "error");
    return null;
  }
}

/** Image jobs a form started for itself (Recast's character still): the form shows the result, so no "ready" toast,
 * and the queue bar lets them go once they end. */
const formImageJobs = new Set<string>();

/** Recast's "Make one": a FLUX 3 klein still from words (lib/still.ts), queued as an ordinary image job. Throws
 * ApiError when it can't start. */
export async function makeCharacterStill(projectId: string, req: ImageGenerateRequest): Promise<Job> {
  const job = await getApi().generateImages(projectId, req);
  formImageJobs.add(job.id);
  applyJob(job);
  return job;
}

/** A character still nobody needs any more ("Try another", new words): cancelled if it hasn't run yet. */
export function dropCharacterStill(jobId: string): void {
  const job = getState().jobs[jobId];
  if (!job || !isActive(job)) return;
  void getApi()
    .cancelJob(jobId)
    .then((j) => j && applyJob(j))
    .catch(() => undefined);
}

async function onImagesDone(job: Job): Promise<void> {
  const pid = job.project_id || targetProject();
  if (formImageJobs.delete(job.id)) {
    dismissJob(job.id);
    await loadImages(pid);
    void loadProjects();
    return;
  }
  await loadImages(pid);
  void loadProjects();
  const n = job.result_image_ids?.length ?? 0;
  const here = getState().route;
  if (!(here.name === "create" && here.panel === "image")) {
    const view = () => {
      setCurrentProject(pid, true); // the Image page shows the current project's images
      navigate({ name: "create", panel: "image" });
    };
    toast(`${n} ${n === 1 ? "image" : "images"} ready`, "success", { label: "View", run: view }, 8000);
  }
}

/** Create's Text to video and Image to video (LTX-2.5): queue it in the current project. Returns the job, or an error
 * to show (the composer follows the job: its progress, Cancel, and how long it took; done, it is a new shot and a
 * toast opens it). */
export async function generateVideo(req: VideoGenerateRequest): Promise<{ job: Job | null; error: string | null }> {
  try {
    const job = await getApi().generateVideo(targetProject(), req);
    applyJob(job);
    confirmQueued(job, "LTX-2.5 video", true);
    return { job, error: null };
  } catch (e) {
    return { job: null, error: message(e, "The video didn't start. Try again.") };
  }
}

/** A picture from this computer for a form (Image to video's start picture): uploaded, as a reference. Throws a plain
 * sentence when it can't be. */
export async function uploadPicture(file: File): Promise<{ ref: Ref; url: string }> {
  try {
    const up = await getApi().upload(file);
    return { ref: { kind: "upload", id: up.id }, url: up.url };
  } catch (e) {
    throw new Error(message(e, "That picture didn't upload. Try again."));
  }
}

/** A new video is done: it is a shot of its own in the project. Show it on the wall and in Assets; offer to open it. */
async function onVideoDone(job: Job): Promise<void> {
  const pid = job.project_id || targetProject();
  const id = job.result_shot_id;
  const shot = id ? await ensureShot(id, true) : null;
  if (shot) putShot(shot);
  await loadShots(pid);
  void loadProjects();
  if (!shot) return;
  toast(`Video ready · ${shot.name}`, "success", { label: "View", run: () => openShot(shot.id) }, 8000);
}

export async function imageToCast(projectId: string, imageId: string, fields: { name: string; kind: CastKind; line: string }): Promise<string | null> {
  try {
    const c = await getApi().imageToCast(projectId, imageId, fields);
    setState((s) => ({ cast: { ...s.cast, [projectId]: [...(s.cast[projectId] ?? []), c] } }));
    void loadProjects();
    toast(`${c.name} added to the cast. Use @${c.name} in prompts.`, "success");
    return null;
  } catch (e) {
    return message(e, "That didn't save. Try again.");
  }
}

export type ImageUse = "look_frame" | "first_frame" | "previz_ref" | "recast_ref";

/** Hand a project image to a shot's task form: look frame / first frame / reference. */
export function useImageIn(use: ImageUse, image: ProjectImage, shotId: string): void {
  const ref: Ref = { kind: "image", id: image.id };
  const map: Record<ImageUse, { task: TaskId; params: Record<string, unknown> }> = {
    look_frame: { task: "previz", params: { look_frame_ref: ref, _look_frame_url: image.thumb_url || image.url } },
    first_frame: { task: "environment", params: { first_frame_ref: ref, _first_frame_url: image.thumb_url || image.url } },
    previz_ref: { task: "previz", params: { look_refs: [ref], _ref_urls: { [image.id]: image.thumb_url || image.url } } },
    recast_ref: { task: "recast", params: { who_ref: ref, _who_url: image.thumb_url || image.url } },
  };
  const { task, params } = map[use];
  setState((s) => ({ prefill: { ...s.prefill, [shotId]: { task, params, seq: (s.prefill[shotId]?.seq ?? 0) + 1 } } }));
  openShot(shotId, { rail: { kind: "form", task }, versionId: "orig" });
}

// --- prompt director --------------------------------------------------------------------------------------------

export async function loadDirector(): Promise<void> {
  try {
    const st = await getApi().settings();
    setState({ director: st.director ?? { backend: "comfy", url: "", model: "", vision_model: "" }, comfyUrl: st.comfy_url || null });
  } catch {
    setState({ director: { backend: "comfy", url: "", model: "", vision_model: "" } });
  }
}

export async function saveDirector(director: Partial<DirectorSettings>): Promise<string | null> {
  try {
    const st = await getApi().saveSettings({ director });
    setState({ director: st.director ?? { backend: "comfy", url: "", model: "", vision_model: "", ...director } as DirectorSettings });
    return null;
  } catch (e) {
    return message(e, "That didn't save. Try again.");
  }
}
