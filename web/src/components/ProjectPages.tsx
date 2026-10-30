// All projects (grid) and one project: Shots · Images · Cast · Deliveries (docs/projects-spec.md).
import { Ellipsis, FileDown, Folder, LayoutGrid, List, Pencil, Plus, ReceiptText, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import {
  addCast,
  approveFinishedTake,
  archiveProject,
  buildSheet,
  cancelJob,
  createProject,
  deleteAsset,
  deleteProject,
  dismissJob,
  ensureProject,
  finishApproved,
  finishCandidates,
  getApi,
  importCast,
  isLocalHost,
  loadDeliveries,
  navigate,
  openProject,
  openShot,
  renameProject,
  revealTake,
  setCurrentProject,
  sheetName,
  toast,
  updateCast,
} from "../actions";
import { ApiError } from "../api";
import { castDeleteLabel, deleteLine } from "../lib/deleted";
import { Results } from "./ImagePage";
import { NonCommercialTag } from "./RightRail";
import { ReceiptLink } from "./TakeExtras";
import { aboutTime, aiShare, cx, duration, finishSentence, isActive, isOriginal, nonCommercialModels, relativeTime, shotLength, takeLabel, takes, toEpochMs } from "../lib/format";
import { queueShort, useQueuePlace } from "../lib/hooks";
import { elapsedClock, madeIn, useElapsed } from "../lib/runtime";
import { nextSeed } from "../lib/seed";
import { TOUR } from "../lib/tour";
import { useApp, type ProjectTab } from "../store";
import type { CastKind, CastMember, Delivery, Download, Job, Project, SheetRequest, Shot, TimelineFormat, Version } from "../types";
import { DropActions, stackBadge } from "./Home";
import { CastSheetAction, CastSheetChip, CastViewer, NewCharacterModal } from "./Character";
import { flowOn } from "../lib/drafts";
import { latestVideo } from "../lib/thumbs";
import { MediaCardSkeleton, Thumb } from "./MediaCard";
import { ShotMoreMenu } from "./ShotMenu";
import { projectCounts } from "./ProjectSwitcher";
import { Button, ConfirmDelete, DropSlot, Hint, MENU_DELETE, MENU_ITEM, MenuPill, Modal, Notice, Page, PageHeader, ProgressLine, ReceiptBar, SectionLabel, SeedLock, Segmented, Skeleton, Slate, Spinner, StateDot, Switch, SwitchRow, Tag, useDismiss } from "./ui";

// --- All projects -----------------------------------------------------------------------------------------------

type ProjectSort = "recent" | "name";
type ProjectView = "grid" | "list";
const VIEW_KEY = "pickups.projects.view";
let projectPrefs: { sort: ProjectSort; view: ProjectView } = {
  sort: "recent",
  view: (() => {
    try {
      return window.localStorage.getItem(VIEW_KEY) === "list" ? "list" : "grid";
    } catch {
      return "grid";
    }
  })(),
};

function updatedAt(p: Project): number {
  return toEpochMs(p.updated_at ?? p.created_at) ?? 0;
}

export function ProjectsPage() {
  const all = useApp((s) => s.projects);
  const currentId = useApp((s) => s.currentProjectId);
  const [showArchived, setShowArchived] = useState(false);
  const [sort, setSort] = useState<ProjectSort>(projectPrefs.sort);
  const [view, setView] = useState<ProjectView>(projectPrefs.view);
  useEffect(() => {
    projectPrefs = { sort, view };
    try {
      window.localStorage.setItem(VIEW_KEY, view);
    } catch {
      /* private mode */
    }
  }, [sort, view]);
  const archivedCount = (all ?? []).filter((p) => p.archived).length;
  const projects = useMemo(() => {
    if (!all) return null;
    const list = all.filter((p) => showArchived || !p.archived);
    // My shots first (it's where work goes without a project), then by the sort.
    const by = sort === "name" ? (a: Project, b: Project) => a.name.localeCompare(b.name, undefined, { numeric: true }) : (a: Project, b: Project) => updatedAt(b) - updatedAt(a);
    return [...list].sort((a, b) => (a.id === "unsorted" ? -1 : b.id === "unsorted" ? 1 : by(a, b)));
  }, [all, showArchived, sort]);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    await createProject(name.trim());
    setBusy(false);
  };
  return (
    <Page>
      <PageHeader
        title="Projects"
        actions={
          naming ? (
          <form onSubmit={submit} className="flex flex-wrap items-center gap-2 ">
            <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name" aria-label="New project name" className="field h-9 w-64 max-[767px]:w-full" />
            <Button type="submit" variant="primary" size="md" loading={busy} disabled={!name.trim()}>
              Create
            </Button>
            <Button variant="ghost" size="md" onClick={() => setNaming(false)}>
              Cancel
            </Button>
          </form>
        ) : (
          <Button variant="primary" size="md" icon={Plus} onClick={() => setNaming(true)}>
            New project
          </Button>
          )
        }
      />
      <div className="mb-6 mt-2 flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
        <div className="flex flex-wrap items-center gap-4">
          <SectionLabel>{projects ? `${projects.length} ${projects.length === 1 ? "project" : "projects"}` : "Loading"}</SectionLabel>
          {archivedCount ? (
            <label className="flex items-center gap-2.5 text-[13px] text-ink2">
              <Switch checked={showArchived} onChange={setShowArchived} label="Show archived" />
              Show archived <span className="mono text-[11px] text-ink3">{archivedCount}</span>
            </label>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Segmented<ProjectSort>
            mono
            label="Sort"
            value={sort}
            onChange={setSort}
            options={[
              { value: "recent", label: "Recent" },
              { value: "name", label: "Name" },
            ]}
          />
          <div role="radiogroup" aria-label="View" className="inline-flex rounded-full border border-line p-[2px]">
            {(
              [
                ["grid", LayoutGrid, "Grid"],
                ["list", List, "List"],
              ] as const
            ).map(([v, I, label]) => (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={view === v}
                aria-label={label}
                title={label}
                onClick={() => setView(v)}
                className={cx(
                  "inline-flex h-6 w-8 items-center justify-center rounded-full transition-colors duration-150",
                  view === v ? "bg-raised text-ivory shadow-[inset_0_0_0_1px_#3A3631]" : "text-ink3 hover:text-ink",
                )}
              >
                <I size={13} strokeWidth={1.75} aria-hidden="true" />
              </button>
            ))}
          </div>
        </div>
      </div>
      {view === "grid" ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-6 gap-y-10">
          {projects
            ? projects.map((p) => <ProjectCard key={p.id} p={p} current={p.id === currentId} />)
            : [0, 1, 2].map((i) => <MediaCardSkeleton key={i} />)}
        </div>
      ) : (
        <ul className="border-t border-line" aria-label="Projects">
          {projects
            ? projects.map((p) => <ProjectRow key={p.id} p={p} current={p.id === currentId} />)
            : [0, 1, 2].map((i) => (
                <li key={i} className="flex items-center gap-4 border-b border-line py-3">
                  <Skeleton className="h-[45px] w-20" />
                  <Skeleton className="h-4 w-48" />
                </li>
              ))}
        </ul>
      )}
    </Page>
  );
}

function projectBadge(p: Project, current: boolean): string | null {
  return p.archived ? "Archived" : current ? "Current" : p.builtin ? "Built in" : null;
}

/** "…" on a project: Open, Rename, Archive (not for the built-in ones), then Delete (not for Examples), which asks first. */
function ProjectMenu({ p, onRename, className }: { p: Project; onRename: () => void; className?: string }) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => {
    setOpen(false);
    setConfirming(false);
  }, []);
  useDismiss(open, close, ref);
  const item = MENU_ITEM;
  return (
    <div ref={ref} className={cx("relative", className)}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${p.name}: more`}
        onClick={() => (open ? close() : setOpen(true))}
        className="inline-flex h-8 w-8 items-center justify-center rounded-full text-ink3 transition-colors duration-150 hover:bg-raised hover:text-ivory"
      >
        <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
      </button>
      {open && confirming ? (
        <div className="absolute right-0 top-full z-40 mt-1.5 animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised">
          <ConfirmDelete title={`Delete ${p.name}?`} line={deleteLine("project", p.shots)} confirm="Delete project" onCancel={close} onConfirm={async () => (await deleteProject(p.id)) && close()} />
        </div>
      ) : open ? (
        <div role="menu" aria-label={p.name} className="absolute right-0 top-full z-40 mt-1.5 w-[200px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          <button
            type="button"
            role="menuitem"
            className={item}
            onClick={() => {
              close();
              openProject(p.id);
            }}
          >
            Open
          </button>
          {!p.builtin ? (
            <>
              <button
                type="button"
                role="menuitem"
                className={item}
                onClick={() => {
                  close();
                  onRename();
                }}
              >
                Rename…
              </button>
              <button
                type="button"
                role="menuitem"
                className={item}
                onClick={() => {
                  close();
                  void archiveProject(p.id, !p.archived);
                }}
              >
                {p.archived ? "Unarchive" : "Archive"}
              </button>
            </>
          ) : null}
          {p.id !== "examples" ? (
            <div className="mt-1 border-t border-line pt-1">
              <button type="button" role="menuitem" className={MENU_DELETE} onClick={() => setConfirming(true)}>
                Delete
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** Rename in place: the name becomes a field (Enter saves, Esc cancels). */
function useRename(p: Project) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(p.name);
  const field = editing ? (
    <form
      className="min-w-0 flex-1"
      onSubmit={(e) => {
        e.preventDefault();
        if (name.trim() && name.trim() !== p.name) void renameProject(p.id, name.trim());
        setEditing(false);
      }}
    >
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onBlur={() => {
          if (name.trim() && name.trim() !== p.name) void renameProject(p.id, name.trim());
          setEditing(false);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            setName(p.name);
            setEditing(false);
          }
        }}
        onFocus={(e) => e.target.select()}
        aria-label="Project name"
        className="field h-9 w-full text-[15px]"
      />
    </form>
  ) : null;
  return {
    field,
    start: () => {
      setName(p.name);
      setEditing(true);
    },
  };
}

function ProjectCard({ p, current }: { p: Project; current: boolean }) {
  const rename = useRename(p);
  const badge = projectBadge(p, current);
  const updated = relativeTime(p.updated_at ?? p.created_at);
  return (
    <div className="group/card min-w-0">
      <button
        type="button"
        onClick={() => openProject(p.id)}
        aria-label={`${p.name}${current ? " (new work goes here)" : ""}, ${projectCounts(p)}`}
        className="group block w-full rounded-card text-left"
      >
        <div className="hairline-frame relative aspect-video overflow-hidden rounded-card bg-black">
          {p.cover_url ? (
            <Thumb src={p.cover_url} video={p.cover_video_url ?? undefined} className="absolute inset-0 h-full w-full transition-transform duration-300 ease-out group-hover:scale-[1.02]" />
          ) : (
            <span className="absolute inset-0 flex items-center justify-center bg-raised text-ink3" aria-hidden="true">
              <Folder size={28} strokeWidth={1.25} />
            </span>
          )}
          {badge ? <Slate at="tl">{badge}</Slate> : null}
        </div>
      </button>
      <div className="mt-4 flex items-start gap-2">
        {rename.field ?? (
          <button type="button" onClick={() => openProject(p.id)} className="min-w-0 flex-1 text-left" tabIndex={-1} aria-hidden="true">
            <h2 className="display truncate text-[24px] transition-colors hover:text-ivory">{p.name}</h2>
          </button>
        )}
        <ProjectMenu p={p} onRename={rename.start} className="shrink-0 pt-0.5" />
      </div>
      <div className="slate mt-1.5 flex flex-wrap gap-x-3 text-ink3">
        <span>{projectCounts(p)}</span>
        {updated ? <span className="text-ink4">Updated {updated}</span> : null}
      </div>
    </div>
  );
}

function ProjectRow({ p, current }: { p: Project; current: boolean }) {
  const rename = useRename(p);
  const badge = projectBadge(p, current);
  const updated = relativeTime(p.updated_at ?? p.created_at);
  return (
    <li className="group flex items-center gap-4 border-b border-line py-2.5">
      <button type="button" onClick={() => openProject(p.id)} aria-label={`${p.name}, ${projectCounts(p)}`} className="shrink-0 rounded-[4px]">
        {p.cover_url ? (
          <Thumb src={p.cover_url} className="h-[45px] w-20 rounded-[4px]" />
        ) : (
          <span className="flex h-[45px] w-20 items-center justify-center rounded-[4px] border border-line bg-raised text-ink3" aria-hidden="true">
            <Folder size={16} strokeWidth={1.5} />
          </span>
        )}
      </button>
      <div className="min-w-0 flex-1">
        {rename.field ?? (
          <button type="button" onClick={() => openProject(p.id)} tabIndex={-1} aria-hidden="true" className="block max-w-full truncate text-left text-[16px] text-ink transition-colors group-hover:text-ivory">
            {p.name}
          </button>
        )}
        {badge ? <span className="slate text-ink3">{badge}</span> : null}
      </div>
      <span className="slate hidden shrink-0 text-ink3 md:block">{projectCounts(p)}</span>
      <span className="slate hidden w-40 shrink-0 whitespace-nowrap text-right text-ink4 sm:block">{updated ? `Updated ${updated}` : ""}</span>
      <ProjectMenu p={p} onRename={rename.start} className="shrink-0" />
    </li>
  );
}

// --- One project ------------------------------------------------------------------------------------------------


const TABS: Array<{ id: ProjectTab; label: string }> = [
  { id: "shots", label: "Shots" },
  { id: "images", label: "Images" },
  { id: "cast", label: "Cast" },
  { id: "deliveries", label: "Deliveries" },
];

export function ProjectPage({ id, tab }: { id: string; tab: ProjectTab }) {
  const project = useApp((s) => s.projects?.find((p) => p.id === id) ?? null);
  const projects = useApp((s) => s.projects);
  const castCount = useApp((s) => s.cast[id]?.length);
  const imageCount = useApp((s) => s.images[id]?.length);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState("");

  if (projects && !project) {
    return (
      <Page>
        <Notice text="This project isn't here anymore." action={{ label: "All projects", onClick: () => navigate({ name: "projects" }) }} />
      </Page>
    );
  }

  const counts: Record<ProjectTab, number | undefined> = {
    shots: project?.shots,
    images: imageCount,
    cast: castCount ?? project?.cast,
    deliveries: project?.approved,
  };

  return (
    <Page>
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div className="min-w-0">
          {renaming && project ? (
            <form
              className="mt-2 flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (name.trim()) void renameProject(project.id, name.trim());
                setRenaming(false);
              }}
            >
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} className="field h-12 w-[min(560px,60vw)] font-sans text-[28px]" aria-label="Project name" />
              <Button type="submit" variant="primary" size="md">
                Save
              </Button>
              <Button variant="ghost" size="md" onClick={() => setRenaming(false)}>
                Cancel
              </Button>
            </form>
          ) : (
            <h1 className="display group flex items-baseline gap-3 text-[36px] leading-[1.05]">
              {project ? project.name : <Skeleton className="h-14 w-80" />}
              {project && !project.builtin ? (
                <button
                  type="button"
                  aria-label="Rename project"
                  onClick={() => {
                    setName(project.name);
                    setRenaming(true);
                  }}
                  className="rounded-full p-1.5 text-ink4 opacity-0 transition-opacity hover:text-ivory focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Pencil size={16} strokeWidth={1.5} />
                </button>
              ) : null}
            </h1>
          )}
          {project ? (
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1">
              <span className="slate text-ink3">
                {project.archived ? "Archived · " : ""}
                {projectCounts(project)}
              </span>
              {!project.builtin && !renaming ? (
                <span className="flex items-center gap-3 text-[13px]">
                  <button
                    type="button"
                    onClick={() => {
                      setName(project.name);
                      setRenaming(true);
                    }}
                    className="text-ink3 transition-colors hover:text-ivory"
                  >
                    Rename
                  </button>
                  <span className="text-ink4" aria-hidden="true">
                    ·
                  </span>
                  <button type="button" onClick={() => void archiveProject(project.id, !project.archived)} className="text-ink3 transition-colors hover:text-ivory">
                    {project.archived ? "Unarchive" : "Archive"}
                  </button>
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
        {tab === "shots" && id !== "examples" ? <DropActions size="md" className="pb-1" /> : null}
      </div>

      <nav className="mt-10 flex gap-8 border-b border-line" aria-label="Project">
        {TABS.map((t) => {
          const on = t.id === tab;
          return (
            <a
              key={t.id}
              href={`#/project/${encodeURIComponent(id)}${t.id === "shots" ? "" : `/${t.id}`}`}
              aria-current={on ? "page" : undefined}
              className={cx(
                "relative -mb-px flex items-baseline gap-2 border-b py-3 text-[15px] transition-colors duration-150",
                on ? "border-ivory text-ivory" : "border-transparent text-ink3 hover:text-ink",
              )}
            >
              {t.label}
              {counts[t.id] != null ? <span className="mono text-[11px] text-ink3">{counts[t.id]}</span> : null}
            </a>
          );
        })}
      </nav>

      <div className="mt-10">
        {tab === "shots" ? (
          <ShotsTab id={id} />
        ) : tab === "images" ? (
          <ImagesTab id={id} />
        ) : tab === "cast" ? (
          <CastTab id={id} />
        ) : (
          <DeliveriesTab id={id} />
        )}
      </div>
    </Page>
  );
}

// --- Shots ------------------------------------------------------------------------------------------------------

function ShotCard({ shot }: { shot: Shot }) {
  const approved = shot.approved ?? shot.versions.filter((v) => v.status === "approved").length;
  return (
    <div className="group/card relative min-w-0">
      <button
        type="button"
        onClick={() => openShot(shot.id)}
        aria-label={`${shot.name}, ${shotLength(shot)}, ${stackBadge(shot).toLowerCase()}${approved ? `, ${approved} approved` : ""}`}
        className="group block w-full text-left"
      >
        <div className="hairline-frame relative aspect-video overflow-hidden rounded-card bg-black">
          <Thumb src={shot.latest_thumb_url || shot.thumb_url} video={latestVideo(shot)} className="absolute inset-0 h-full w-full transition-transform duration-300 ease-out group-hover:scale-[1.02]" />
          <Slate at="tl">{stackBadge(shot)}</Slate>
          {approved ? (
            <span className="slate absolute bottom-3 left-3 z-[2] flex items-center gap-1.5 rounded-[2px] bg-black/60 px-1.5 py-[3px]" title={`${approved} approved`}>
              <StateDot on size={6} /> {approved > 1 ? `${approved} approved` : "Approved"}
            </span>
          ) : null}
          <Slate at="br">{shotLength(shot)}</Slate>
        </div>
        <div className="mt-3 flex items-baseline justify-between gap-3">
          <span className="truncate font-mono text-[13px] text-ink transition-colors group-hover:text-ivory">{shot.name}</span>
          <span className="slate shrink-0 text-ink3">{relativeTime(shot.created_at ?? shot.versions[0]?.created_at)}</span>
        </div>
      </button>
      <ShotMoreMenu shot={shot} />
    </div>
  );
}

type ShotSort = "newest" | "name";
type ShotFilter = "all" | "approved" | "takes";
let gridPrefs: { sort: ShotSort; filter: ShotFilter } = { sort: "newest", filter: "all" };

function created(s: Shot): number {
  const t = s.created_at ?? s.versions[0]?.created_at;
  return typeof t === "number" ? t : t ? Date.parse(t) / 1000 || 0 : 0;
}

function ShotsTab({ id }: { id: string }) {
  const ids = useApp((s) => s.projectShots[id] ?? null);
  const shots = useApp((s) => s.shots);
  const [sort, setSort] = useState<ShotSort>(gridPrefs.sort);
  const [filter, setFilter] = useState<ShotFilter>(gridPrefs.filter);
  useEffect(() => {
    gridPrefs = { sort, filter };
  }, [sort, filter]);
  const all = useMemo(() => (ids ?? []).map((x) => shots[x]).filter(Boolean), [ids, shots]);
  const list = useMemo(() => {
    const out = all.filter((s) =>
      filter === "approved"
        ? (s.approved ?? s.versions.filter((v) => v.status === "approved").length) > 0
        : filter === "takes"
          ? s.versions.some((v) => !isOriginal(v))
          : true,
    );
    return sort === "name" ? [...out].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })) : [...out].sort((a, b) => created(b) - created(a));
  }, [all, sort, filter]);
  if (!ids) {
    return (
      <Grid>
        {[0, 1, 2, 3].map((i) => (
          <MediaCardSkeleton key={i} />
        ))}
      </Grid>
    );
  }
  if (!all.length) {
    return id === "examples" ? (
      <Notice text="Examples you try go to My shots." action={{ label: "Examples", onClick: () => navigate({ name: "home" }) }} />
    ) : (
      <Notice text="No shots yet." />
    );
  }
  return (
    <>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <Segmented<ShotFilter>
          mono
          label="Show"
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "All" },
            { value: "approved", label: "Approved" },
            { value: "takes", label: "Has takes" },
          ]}
        />
        <Segmented<ShotSort>
          mono
          label="Sort"
          value={sort}
          onChange={setSort}
          options={[
            { value: "newest", label: "Newest" },
            { value: "name", label: "Name" },
          ]}
        />
      </div>
      {list.length ? (
        <Grid>
          {list.map((s) => (
            <ShotCard key={s.id} shot={s} />
          ))}
        </Grid>
      ) : (
        <Notice text={filter === "approved" ? "No approved shots yet. Open a take and press Approve under the picture (⇧A)." : "No shots with takes yet."} action={{ label: "Show all", onClick: () => setFilter("all") }} />
      )}
    </>
  );
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-5 gap-y-10">{children}</div>;
}

// --- Images -----------------------------------------------------------------------------------------------------

function ImagesTab({ id }: { id: string }) {
  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <span />
        {id !== "examples" ? (
          <Button
            variant="primary"
            size="md"
            onClick={() => {
              setCurrentProject(id, true); // the Image page makes images for the current project
              navigate({ name: "create", panel: "image" });
            }}
          >
            Make an image
          </Button>
        ) : null}
      </div>
      <Results pid={id} compact />
    </div>
  );
}

// --- Cast -------------------------------------------------------------------------------------------------------

const KINDS: Array<{ id: CastKind; label: string; placeholder: string }> = [
  { id: "person", label: "Person", placeholder: "a woman in her thirties with short black hair and a red wool coat" },
  { id: "place", label: "Place", placeholder: "an old town square with cobblestones and a fountain, shopfronts on three sides" },
  { id: "prop", label: "Prop", placeholder: "a vintage green bicycle with a wicker basket" },
  { id: "look", label: "Look", placeholder: "warm low evening sun, long soft shadows, fine film grain" },
];

function KindPicker({ value, onChange }: { value: CastKind; onChange: (k: CastKind) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Kind">
      {KINDS.map((k) => (
        <button
          key={k.id}
          type="button"
          role="radio"
          aria-checked={value === k.id}
          onClick={() => onChange(k.id)}
          className={cx(
            "inline-flex h-7 items-center rounded-full border px-3 text-[12px] transition-colors duration-150",
            value === k.id ? "border-ivory/60 text-ivory" : "border-line text-ink2 hover:border-line2 hover:text-ink",
          )}
        >
          {k.label}
        </button>
      ))}
    </div>
  );
}

/** The grid's first card: New character, and a drop target for a picture. With the character flow on it's the one
 * way in (phones too); without it, the button above says the same and phones hide it. */
function AddCard({ onAdd, only }: { onAdd: (file?: File) => void; only?: boolean }) {
  const [over, setOver] = useState(false);
  return (
    <button
      type="button"
      onClick={() => onAdd()}
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
        onAdd(e.dataTransfer.files?.[0]);
      }}
      className={cx(
        // Phones have the button above and nothing to drop: no second card saying the same (unless it's the only way in).
        "flex min-h-[260px] w-full flex-col items-center justify-center gap-3 rounded-card border border-dashed p-6 text-center transition-colors duration-150",
        !only && "max-[767px]:hidden",
        over ? "border-ivory/70 bg-raised" : "border-line2 hover:border-ivory/40",
      )}
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-full border border-line2 text-ink2">
        <Plus size={16} strokeWidth={1.5} />
      </span>
      <span className="text-[15px] text-ink">New character</span>
    </button>
  );
}

const isImage = (f: File) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|tiff?)$/i.test(f.name);

/** "Add a character": the one way into the cast. A picture (or, for a person, a few words) and a name; for a person,
 * Pickups can build a character sheet from it (front, back, close-up) so Recast gets them right from every side. */
function AddCastModal({ projectId, file: dropped, onClose, onImport }: { projectId: string; file?: File; onClose: () => void; onImport: () => void }) {
  const cast = useApp((s) => s.cast[projectId] ?? []);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [wide, setWide] = useState(false); // a wide picture is most likely a sheet already
  const [kind, setKind] = useState<CastKind>("person");
  const [name, setName] = useState("");
  const [line, setLine] = useState("");
  const [describe, setDescribe] = useState("");
  const [sheet, setSheet] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const who = name.trim().replace(/\s+/g, " ");
  const taken = !!who && cast.some((c) => c.name.toLowerCase() === who.toLowerCase());
  const person = kind === "person";
  const words = describe.trim();
  // A person from words alone is always a sheet; from a picture, unless it's a sheet already or they say no.
  const building = person && (file ? sheet : !!words);
  const ready = !!who && !taken && (!!file || (person && !!words));

  const take = (f: File | undefined | null) => {
    if (!f) return;
    if (!isImage(f)) {
      setError("That's not an image Pickups can read.");
      return;
    }
    setError(null);
    setFile(f);
    setWide(false);
    setSheet(true);
    setPreview(URL.createObjectURL(f));
    if (!name) setName(f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase()).slice(0, 32));
  };
  useEffect(() => take(dropped), [dropped]);
  useEffect(() => {
    if (!preview) return;
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (!live) return;
      const w = img.naturalWidth > img.naturalHeight * 1.3;
      setWide(w);
      if (w) setSheet(false);
    };
    img.src = preview;
    return () => {
      live = false;
      URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      if (building) {
        const req: SheetRequest = file ? { upload_id: (await getApi().upload(file)).id, name: who } : { describe: words, name: who };
        await buildSheet(projectId, req);
      } else {
        const err = await addCast(projectId, file!, { name: who, kind, line: line.trim() });
        if (err) throw new Error(err);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : "That didn't work. Try again.");
      setBusy(false);
    }
  };

  return (
    <Modal open onClose={onClose} label="Add to cast" width={560}>
      <form onSubmit={submit} className="p-6">
        <h2 className="flex items-center gap-2 display text-[36px]">
          Add to cast
          <Hint text="A person, a place, a prop or a look. Type @ in any prompt to use them." />
        </h2>
        <div className="mt-6 space-y-4">
          <KindPicker value={kind} onChange={setKind} />
          {file && preview ? (
            // The whole picture (a portrait, a sheet), not a crop of it.
            <div className="relative h-48 overflow-hidden rounded-card border border-line bg-black">
              <img src={preview} alt="" className="h-full w-full object-contain" />
              <Slate at="tl">{file.name}</Slate>
              <button
                type="button"
                onClick={() => {
                  setFile(null);
                  setPreview(null);
                  setWide(false);
                  setSheet(true);
                }}
                aria-label="Remove the picture"
                className="absolute right-2 top-2 rounded-[2px] bg-black/70 p-1 text-ivory/70 hover:text-ivory"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <DropSlot value={null} onFile={take} onClear={() => undefined} title="Drop a picture" sub={person ? "A photo, a still or a character sheet" : "A photo or a still"} />
          )}
          {!file && person ? (
            <div>
              <div className="slate mb-1.5 text-ink4">Or describe them</div>
              <textarea
                rows={2}
                value={describe}
                onChange={(e) => setDescribe(e.target.value)}
                placeholder="e.g. a green alien in a silver flight suit"
                aria-label="Describe them"
                className="field h-auto resize-none py-2.5 leading-snug"
              />
              {words ? <p className="mt-1.5 text-[12px] text-ink3">Makes a character sheet · ~40 s</p> : null}
            </div>
          ) : null}
          <div>
            <div className="slate mb-1.5 text-ink4">Name</div>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Maya" aria-label="Name" className="field" />
            {taken ? <p className="mt-1.5 text-[12px] text-ink3">There's already a {who} in this cast. Pick another name.</p> : null}
          </div>
          {!building && file ? (
            <div>
              <div className="slate mb-1.5 text-ink4">One line about {person ? "them" : "it"}</div>
              <textarea
                rows={2}
                value={line}
                onChange={(e) => setLine(e.target.value)}
                placeholder={`e.g. ${KINDS.find((k) => k.id === kind)?.placeholder ?? ""}`}
                aria-label="One line that describes it"
                className="field h-auto resize-none py-2.5 leading-snug"
              />
            </div>
          ) : null}
          {person && file ? (
            <div className="border-t border-line pt-3">
              <SwitchRow
                checked={sheet}
                onChange={setSheet}
                title="Build a character sheet"
                sub={wide ? "Looks like a sheet already." : "Front, back and close-up · ~40 s"}
              />
            </div>
          ) : null}
        </div>
        {error ? (
          <p className="mt-4 flex items-baseline gap-2 text-[13px] text-ink" role="alert">
            <span className="slate">Error</span>
            {error}
          </p>
        ) : null}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => {
              onClose();
              onImport();
            }}
            className="text-[13px] text-ink2 underline-offset-4 transition-colors hover:text-ivory hover:underline"
          >
            Copy from another project
          </button>
          <div className="flex gap-2">
            <Button variant="ghost" size="md" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" size="md" loading={busy} disabled={!ready}>
              {person ? "New character" : "Add to cast"}
            </Button>
          </div>
        </div>
      </form>
    </Modal>
  );
}

function CastCard({ projectId, c, job }: { projectId: string; c: CastMember; job?: Job }) {
  const [editing, setEditing] = useState(false);
  const [viewing, setViewing] = useState(false);
  // The Character flow (pickups.flow.character): every card opens the full-screen viewer, and says its sheet's state.
  const flow = flowOn("character");
  const [view, setView] = useState(0);
  const [wide, setWide] = useState(false); // a wide picture is most likely a sheet already
  const sheet = useSheetStart(projectId, c);
  const [name, setName] = useState(c.name);
  const [kind, setKind] = useState<CastKind>(c.kind);
  const [line, setLine] = useState(c.line);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    const err = await updateCast(projectId, c.id, { name: name.trim(), kind, line: line.trim() });
    setBusy(false);
    if (err) setError(err);
    else setEditing(false);
  };
  return (
    <div className="group">
      {flow ? (
        <button
          type="button"
          onClick={() => {
            setView(0);
            setViewing(true);
          }}
          aria-label={`${c.name}: open full screen`}
          className="hairline-frame relative block aspect-[4/3] w-full cursor-zoom-in overflow-hidden rounded-card bg-black"
        >
          <img
            src={c.sheet ? c.sheet.thumb_url || c.sheet.url : c.thumb_url || c.url}
            alt=""
            className={cx("h-full w-full transition-opacity duration-150 group-hover:opacity-90", c.sheet ? "object-contain" : c.kind === "person" ? "thumb-face" : "object-cover")}
            loading="lazy"
          />
          {c.kind !== "person" ? <Slate at="tl">{c.kind}</Slate> : null}
          {c.kind === "person" ? <CastSheetChip projectId={projectId} member={c} /> : null}
          {job ? <SheetProgress job={job} /> : null}
        </button>
      ) : c.sheet ? (
        <button
          type="button"
          onClick={() => setViewing(true)}
          aria-label={`${c.name}'s character sheet: front, back and close-up. Open it bigger.`}
          className="hairline-frame relative block aspect-[4/3] w-full cursor-zoom-in overflow-hidden rounded-card bg-black"
        >
          <img src={c.sheet.thumb_url || c.sheet.url} alt="" className="h-full w-full object-contain transition-opacity duration-150 group-hover:opacity-90" loading="lazy" />
          <Slate at="tl">{c.kind}</Slate>
          {job ? <SheetProgress job={job} /> : null}
        </button>
      ) : (
        <div className="hairline-frame relative aspect-[4/3] overflow-hidden rounded-card bg-black">
          <img
            src={c.thumb_url || c.url}
            alt=""
            className={cx("h-full w-full", c.kind === "person" ? "thumb-face" : "object-cover")}
            loading="lazy"
            onLoad={(e) => setWide(e.currentTarget.naturalWidth > e.currentTarget.naturalHeight * 1.3)}
          />
          <Slate at="tl">{c.kind}</Slate>
          {job ? <SheetProgress job={job} /> : null}
        </div>
      )}
      {viewing && flow ? <CastViewer projectId={projectId} member={c} index={view} onIndex={setView} onClose={() => setViewing(false)} /> : null}
      {viewing && !flow && c.sheet ? <SheetViewer projectId={projectId} c={c} job={job} onClose={() => setViewing(false)} /> : null}
      {editing ? (
        <form onSubmit={save} className="mt-3 space-y-3">
          <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Name" className="field h-9" />
          <KindPicker value={kind} onChange={setKind} />
          <textarea rows={3} value={line} onChange={(e) => setLine(e.target.value)} aria-label="Line" className="field h-auto resize-none py-2 text-[13px] leading-snug" />
          {error ? (
            <p className="flex items-baseline gap-2 text-[13px] text-ink">
              <span className="slate">Error</span>
              {error}
            </p>
          ) : null}
          <div className="flex gap-2">
            <Button type="submit" variant="primary" size="sm" loading={busy} disabled={!name.trim()}>
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setEditing(false);
                setName(c.name);
                setKind(c.kind);
                setLine(c.line);
                setError(null);
              }}
            >
              Cancel
            </Button>
          </div>
        </form>
      ) : (
        <div className="mt-3">
          <div className="flex items-baseline justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <span className="truncate text-[16px] text-ivory">{c.name}</span>
              {c.sheet && !flow ? <Tag className="shrink-0">Sheet</Tag> : null}
            </span>
            <span className="flex shrink-0 items-baseline gap-3">
              {flow && c.kind === "person" ? <CastSheetAction projectId={projectId} member={c} /> : null}
              {!flow && sheet.can && !job && (c.sheet || !wide) ? (
                <button
                  type="button"
                  onClick={sheet.start}
                  disabled={sheet.busy}
                  className="text-[12px] text-ink3 transition-colors hover:text-ivory disabled:text-ink4"
                >
                  {c.sheet ? "Redo sheet" : "Build a sheet"}
                </button>
              ) : null}
              <CastMenu projectId={projectId} c={c} onEdit={() => setEditing(true)} />
            </span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink2">{c.line || <span className="text-ink4">No line yet.</span>}</p>
          <p className="mono mt-2 flex flex-wrap gap-x-3 text-[11px] text-ink3">
            <span>@{c.name}</span>
            {usesLine(c.uses) ? <span className="uppercase tracking-[0.08em]">{usesLine(c.uses)}</span> : null}
          </p>
        </div>
      )}
    </div>
  );
}

/** A cast card's "…": Edit, then Delete (last, after a hairline), which turns the menu into the same confirm as a
 * project's or an asset's. The entry goes to the Trash (Undo on the toast, Restore in Assets > Trash); shots that used
 * it keep their takes. Shown on hover and on focus, like the card's other links. */
function CastMenu({ projectId, c, onEdit }: { projectId: string; c: CastMember; onEdit: () => void }) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => {
    setOpen(false);
    setConfirming(false);
    opener.current?.focus();
  }, []);
  useDismiss(open, close, ref);
  return (
    <span ref={ref} className="relative self-center">
      <button
        ref={opener}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${c.name}: more`}
        onClick={() => (open ? close() : setOpen(true))}
        className={cx(
          "-my-1.5 inline-flex h-7 w-7 items-center justify-center rounded-full text-ink3 transition-[opacity,color,background-color] duration-150 hover:bg-raised hover:text-ivory focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100",
          open ? "opacity-100" : "opacity-0",
        )}
      >
        <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
      </button>
      {open && confirming ? (
        <div className="absolute right-0 top-full z-40 mt-1.5 animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised">
          <ConfirmDelete
            title={`Delete ${c.name}?`}
            line={deleteLine("asset", c.uses ?? 0)}
            confirm={castDeleteLabel(c.kind)}
            onCancel={close}
            onConfirm={async () => {
              if (await deleteAsset(projectId, "cast", c.id)) setOpen(false);
            }}
          />
        </div>
      ) : open ? (
        <div role="menu" aria-label={c.name} className="absolute right-0 top-full z-40 mt-1.5 w-[200px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          <button
            type="button"
            role="menuitem"
            className={MENU_ITEM}
            onClick={() => {
              setOpen(false);
              onEdit();
            }}
          >
            Edit
          </button>
          <div className="mt-1 border-t border-line pt-1">
            <button type="button" role="menuitem" className={MENU_DELETE} onClick={() => setConfirming(true)}>
              Delete
            </button>
          </div>
        </div>
      ) : null}
    </span>
  );
}

// --- character sheets (pickups/sheets.py) ----------------------------------------------------------------------------

/** "Build a sheet" / "Redo sheet" for a cast member: people only, never in Examples. A redo uses a new seed, or the
 * sheet's own seed again with Lock seed on (`locked`, else what the sheet was made with). */
function useSheetStart(projectId: string, c: CastMember, locked?: boolean) {
  const [busy, setBusy] = useState(false);
  const start = async () => {
    setBusy(true);
    try {
      const lock = locked ?? !!c.sheet?.seed_locked;
      await buildSheet(projectId, {
        cast_id: c.id,
        ...(c.sheet ? { seed: nextSeed({ seed: c.sheet.seed ?? null, locked: lock }, Math.random), seed_locked: lock } : {}),
      });
    } catch (e) {
      toast(e instanceof ApiError && e.message ? e.message : "The sheet didn't start. Try again.", "error");
    } finally {
      setBusy(false);
    }
  };
  return { can: projectId !== "examples" && c.kind === "person", busy, start: () => void start() };
}

/** Sheet jobs at work in a project, oldest first. */
function useSheetJobs(projectId: string): Job[] {
  const jobs = useApp((s) => s.jobs);
  const order = useApp((s) => s.jobOrder);
  return useMemo(
    () => order.map((id) => jobs[id]).filter((j): j is Job => !!j && j.task === "sheet" && j.project_id === projectId && isActive(j)).reverse(),
    [jobs, order, projectId],
  );
}

/** New-character sheets that failed and haven't been dismissed: they stay in the grid with Try again (REVIEW F2). */
function useFailedSheets(projectId: string): Job[] {
  const jobs = useApp((s) => s.jobs);
  const order = useApp((s) => s.jobOrder);
  const dismissed = useApp((s) => s.dismissed);
  return useMemo(
    () =>
      order
        .map((id) => jobs[id])
        .filter((j): j is Job => !!j && j.task === "sheet" && j.project_id === projectId && j.status === "failed" && !dismissed[j.id] && !(j.params as SheetRequest | undefined)?.cast_id)
        .reverse(),
    [jobs, order, projectId, dismissed],
  );
}

/** A new character whose sheet didn't build: a plain sentence, Try again (the same request) and Remove. */
function FailedSheetCard({ projectId, job }: { projectId: string; job: Job }) {
  const [busy, setBusy] = useState(false);
  const again = async () => {
    setBusy(true);
    try {
      await buildSheet(projectId, job.params as SheetRequest);
      dismissJob(job.id);
    } catch (e) {
      toast(e instanceof ApiError && e.message ? e.message : "That didn't start. Try again.", "error");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div>
      <div className="hairline-frame relative flex aspect-[4/3] flex-col items-start justify-end gap-3 overflow-hidden rounded-card bg-black p-4" role="alert">
        <Slate at="tl">{["person", "sheet"]}</Slate>
        <p className="text-[13px] leading-snug text-ink2">{job.error || "The sheet didn't build."}</p>
        <Button variant="secondary" size="sm" onClick={() => void again()} loading={busy}>
          Try again
        </Button>
      </div>
      <div className="mt-3 flex items-baseline justify-between gap-3">
        <span className="truncate text-[16px] text-ivory">{sheetName(job)}</span>
        <button type="button" onClick={() => dismissJob(job.id)} className="text-[12px] text-ink3 transition-colors hover:text-ivory">
          Remove
        </button>
      </div>
    </div>
  );
}

/** Over a card or the sheet: the stage, how long it has been running and the record line. */
function SheetProgress({ job }: { job: Job }) {
  const queued = job.status === "queued";
  const elapsed = useElapsed(job);
  const place = useQueuePlace(job);
  return (
    <span className="absolute inset-x-0 bottom-0 flex flex-col justify-end bg-gradient-to-t from-black/90 via-black/70 to-transparent px-3 pb-3 pt-12 text-left animate-fade-in" aria-live="polite">
      <span className="flex items-baseline justify-between gap-3">
        <span className="truncate text-[13px] text-ink">{queued ? `Queued · ${queueShort(place)}` : job.stage}</span>
        <span className="slate shrink-0 text-ink3">{queued ? "Queued" : elapsedClock(elapsed)}</span>
      </span>
      <ProgressLine value={job.progress} idle={queued} className="mt-2" />
    </span>
  );
}

/** A new cast member whose sheet is on its way. */
function PendingSheetCard({ job }: { job: Job }) {
  return (
    <div>
      <div className="hairline-frame relative aspect-[4/3] overflow-hidden rounded-card bg-black">
        <span className="absolute inset-x-0 top-1/2 grid aspect-video -translate-y-1/2 grid-cols-3 gap-[3px]" aria-hidden="true">
          {[0, 1, 2].map((i) => (
            <span key={i} className="skeleton rounded-none" />
          ))}
        </span>
        <Slate at="tl">{["person", "sheet"]}</Slate>
        <SheetProgress job={job} />
      </div>
      <div className="mt-3">
        <div className="flex items-baseline justify-between gap-3">
          <span className="truncate text-[16px] text-ivory">{sheetName(job)}</span>
          <button type="button" onClick={() => void cancelJob(job.id)} className="text-[12px] text-ink3 transition-colors hover:text-ivory">
            Stop
          </button>
        </div>
        <p className="mt-1 text-[13px] leading-snug text-ink2">Building sheet…</p>
      </div>
    </div>
  );
}

/** A cast member's sheet, big: front, back, close-up, what made it, and Redo. */
function SheetViewer({ projectId, c, job, onClose }: { projectId: string; c: CastMember; job?: Job; onClose: () => void }) {
  const s = c.sheet!;
  const [locked, setLocked] = useState(!!s.seed_locked && s.seed != null);
  const redo = useSheetStart(projectId, c, locked);
  const made = [s.models?.length ? `Made with ${s.models.join(" and ")}` : s.editor ? `Made with ${s.editor}` : null, madeIn(s.took_s) || null]
    .filter(Boolean)
    .join(" · ");
  return (
    <Modal open onClose={onClose} label={`${c.name}'s character sheet`} width={1120}>
      <div className="p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h2 className="display text-[24px]">{c.name}</h2>
          <span className="slate text-ink3">Front · Back · Close-up</span>
        </div>
        <div className="hairline-frame relative mt-4 aspect-video overflow-hidden rounded-card bg-black">
          <img src={s.url} alt={`${c.name} from the front, from behind and close up`} className="h-full w-full object-contain" />
          {job ? <SheetProgress job={job} /> : null}
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <p className="flex flex-wrap items-center gap-2 text-[12px] text-ink3">
            {made}
            <NonCommercialTag version={{ licences: s.licences ?? [] }} />
          </p>
          <div className="flex items-center gap-2">
            {redo.can && s.seed != null ? <SeedLock seed={s.seed} locked={locked} onChange={setLocked} /> : null}
            {redo.can ? (
              <Button variant="secondary" size="sm" onClick={redo.start} loading={redo.busy} disabled={!!job}>
                Redo
              </Button>
            ) : null}
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}

/** "Copy from another project": every project's cast, pick one to copy it here. */
function CastImport({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const mine = useApp((s) => s.cast[projectId] ?? []);
  const [list, setList] = useState<CastMember[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    getApi()
      .allCast()
      .then((all) => alive && setList(all.filter((c) => c.project_id && c.project_id !== projectId)))
      .catch((e) => alive && setFailed(e instanceof ApiError && e.message ? e.message : "The other projects' cast didn't load."));
    return () => {
      alive = false;
    };
  }, [projectId]);
  const groups = useMemo(() => {
    const m = new Map<string, { name: string; items: CastMember[] }>();
    for (const c of list ?? []) {
      const g = m.get(c.project_id!) ?? { name: c.project_name ?? "Project", items: [] };
      g.items.push(c);
      m.set(c.project_id!, g);
    }
    return Array.from(m.entries());
  }, [list]);
  const taken = new Set(mine.map((c) => c.name.toLowerCase()));
  return (
    <Modal open onClose={onClose} label="Copy from another project" width={720}>
      <div className="p-6">
        <h2 className="display text-[36px]">Copy from another project</h2>
        <p className="mt-2 text-[14px] text-ink2">Copies don't change the original.</p>
        {error ? (
          <p className="mt-4 flex items-baseline gap-2 text-[13px] text-ink" role="alert">
            <span className="slate">Error</span>
            {error}
          </p>
        ) : null}
        <div className="mt-6 max-h-[60vh] space-y-8 overflow-y-auto pr-1">
          {failed ? (
            <Notice tone="bad" text={failed} />
          ) : !list ? (
            <div className="grid grid-cols-4 gap-3">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="aspect-square" />
              ))}
            </div>
          ) : !groups.length ? (
            <Notice text="No other project has a cast yet." />
          ) : (
            groups.map(([pid, g]) => (
              <section key={pid}>
                <div className="slate mb-3 text-ink3">{g.name}</div>
                <div className="grid grid-cols-4 gap-3 max-[700px]:grid-cols-3">
                  {g.items.map((c) => {
                    const dup = taken.has(c.name.toLowerCase());
                    return (
                      <button
                        key={c.id}
                        type="button"
                        disabled={dup || !!busy}
                        aria-label={`${c.name}, ${c.kind}, from ${g.name}${dup ? ", already in this cast" : ""}`}
                        onClick={async () => {
                          setBusy(c.id);
                          setError(null);
                          const err = await importCast(projectId, pid, c.id);
                          setBusy(null);
                          if (err) setError(err);
                          else onClose();
                        }}
                        className="group text-left disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <div className="hairline-frame relative aspect-square overflow-hidden rounded-card bg-black">
                          <img src={c.thumb_url || c.url} alt="" className="thumb-face h-full w-full transition-opacity group-hover:opacity-85" loading="lazy" />
                          {busy === c.id ? (
                            <span className="absolute inset-0 flex items-center justify-center bg-black/50 text-ivory">
                              <Spinner />
                            </span>
                          ) : null}
                        </div>
                        <div className="mt-2 truncate text-[14px] text-ink group-hover:text-ivory">{c.name}</div>
                        <div className="slate truncate text-ink3">{dup ? "Already in this cast" : c.kind}</div>
                      </button>
                    );
                  })}
                </div>
              </section>
            ))
          )}
        </div>
        <div className="mt-6 flex justify-end">
          <Button variant="ghost" size="md" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </Modal>
  );
}

function usesLine(n: number | undefined): string | null {
  if (n == null) return null;
  return n === 0 ? null : `Used in ${n} ${n === 1 ? "shot" : "shots"}`;
}

function CastTab({ id }: { id: string }) {
  const cast = useApp((s) => s.cast[id] ?? null);
  const [importing, setImporting] = useState(false);
  const [adding, setAdding] = useState<{ file?: File } | null>(null);
  const sheetJobs = useSheetJobs(id);
  const failedSheets = useFailedSheets(id);
  const jobFor = (castId: string) => sheetJobs.find((j) => (j.params as SheetRequest | undefined)?.cast_id === castId);
  useEffect(() => {
    void ensureProject(id);
  }, [id]);
  return (
    <div>
      {/* One way in: the grid's New character card (the flow), or this button with it (the old form). */}
      {id !== "examples" && !flowOn("character") ? (
        <div className="mb-8 flex justify-end">
          <Button variant="primary" size="md" icon={Plus} onClick={() => setAdding({})}>
            New character
          </Button>
        </div>
      ) : null}
      {importing ? <CastImport projectId={id} onClose={() => setImporting(false)} /> : null}
      {adding && flowOn("character") && !adding.file ? (
        <NewCharacterModal projectId={id} onClose={() => setAdding(null)} onImport={() => setImporting(true)} />
      ) : adding ? (
        <AddCastModal projectId={id} file={adding.file} onClose={() => setAdding(null)} onImport={() => setImporting(true)} />
      ) : null}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-x-5 gap-y-10">
        {id !== "examples" ? <AddCard only={flowOn("character")} onAdd={(file) => setAdding({ file })} /> : null}
        {sheetJobs
          .filter((j) => !(j.params as SheetRequest | undefined)?.cast_id)
          .map((j) => (
            <PendingSheetCard key={j.id} job={j} />
          ))}
        {failedSheets.map((j) => (
          <FailedSheetCard key={j.id} projectId={id} job={j} />
        ))}
        {cast ? cast.map((c) => <CastCard key={c.id} projectId={id} c={c} job={jobFor(c.id)} />) : [0, 1].map((i) => <Skeleton key={i} className="aspect-[4/3]" />)}
      </div>
    </div>
  );
}

// --- Deliveries ---------------------------------------------------------------------------------------------------

function fileSize(bytes?: number) {
  if (!bytes) return "";
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

const DELIVERY_SIZE: Record<string, [number, number]> = { HD: [1920, 1080], UHD: [3840, 2160] };

function frameSize(f: Download): [number, number] | null {
  const m = /(\d{3,5})\s*[×x]\s*(\d{3,5})/.exec(f.label);
  return m ? [Number(m[1]), Number(m[2])] : (f.size && DELIVERY_SIZE[f.size]) || null;
}

/** "ProRes 422 HQ · Native 1920×1088": what the file is. It still downloads under its NLE-safe file name. */
function fileLabel(f: Download): string {
  return f.label.replace(/·\s*native\b/, "· Native");
}

/** A native master within a few pixels of a delivery size (1920×1088 next to HD 1920×1080) reads as a duplicate here;
 * it stays in the take's Download menu. */
function listedFiles(files: Download[]): Download[] {
  return files.filter((f) => {
    const n = f.size === "native" ? frameSize(f) : null;
    if (!n) return true;
    return !files.some((o) => {
      const s = o !== f && o.codec === f.codec && o.size !== "native" ? frameSize(o) : null;
      return !!s && Math.abs(s[0] - n[0]) <= 16 && Math.abs(s[1] - n[1]) <= 16;
    });
  });
}

/** An approved take's Finish that is done but not approved: Deliveries offers it in the take's place, so "Finish all
 * approved" ends with the 4K HDR master in the edit (none once a Finish of the take is approved). */
function finishedChild(shot: Shot | undefined, take: Version): Version | null {
  if (!shot || take.task === "finish") return null;
  const kids = shot.versions.filter((v) => v.task === "finish" && !v.hidden && (v.input_version || "orig") === take.id);
  if (kids.some((v) => v.status === "approved")) return null;
  return kids[kids.length - 1] ?? null;
}

function DeliveryRow({ d }: { d: Delivery }) {
  const shot = useApp((s) => s.shots[d.shot_id]);
  const tk = shot ? takes(shot) : null;
  const t = tk?.[d.take.id];
  const finished = finishedChild(shot, d.take);
  const ft = finished ? tk?.[finished.id] : null;
  const from = d.take.input_version && tk?.[d.take.input_version] ? `From take ${tk[d.take.input_version].num}` : null;
  const files = listedFiles(d.take.downloads ?? []);
  const sentence = finishSentence(d.take); // Finish: upscaled from the footage (no new content), HDR highlights as AI
  return (
    <li className="grid grid-cols-[240px_1fr] gap-6 border-t border-line py-6 max-[900px]:grid-cols-1">
      <button type="button" onClick={() => openShot(d.shot_id, { versionId: d.take.id })} className="group text-left">
        <div className="hairline-frame relative aspect-video overflow-hidden rounded-card bg-black">
          <Thumb src={d.take.thumb_url} video={d.take.video_url} className="absolute inset-0 h-full w-full" />
          <Slate at="tl" live>
            {[t ? `Take ${t.num}` : "Take", t?.label ?? takeLabel(d.take.label)]}
          </Slate>
        </div>
      </button>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <span className="flex min-w-0 items-center gap-2.5">
            <span className="font-mono text-[14px] text-ink">{d.shot_name}</span>
            <NonCommercialTag version={d.take} />
          </span>
          <span className="slate text-ink3">{[from, isOriginal(d.take) ? null : aiShare(d.take.receipt), relativeTime(d.take.created_at)].filter(Boolean).join(" · ")}</span>
        </div>
        {d.take.receipt ? <ReceiptBar receipt={d.take.receipt} className="mt-3 max-w-[520px]" /> : null}
        {sentence ? <p className="mt-2 max-w-[520px] text-[12px] leading-snug text-ink3">{sentence}</p> : null}
        <ul className="mt-4 space-y-1">
          {files.length ? (
            files.map((f) => (
              <li key={f.url + f.label}>
                <a
                  href={f.url}
                  download={f.filename || ""}
                  title={f.filename || undefined}
                  className="group/f flex items-baseline justify-between gap-4 rounded-[4px] px-2 py-1.5 transition-colors hover:bg-raised"
                >
                  <span className="truncate text-[13px] text-ink2 group-hover/f:text-ivory">{fileLabel(f)}</span>
                  <span className="mono shrink-0 text-[11px] text-ink3">{fileSize(f.bytes)}</span>
                </a>
              </li>
            ))
          ) : (
            <li className="slate px-2 text-ink3">{isOriginal(d.take) ? "The original file" : "Files are still being written"}</li>
          )}
        </ul>
        {finished && ft ? (
          <div className="mt-3 flex max-w-[520px] flex-wrap items-center justify-between gap-x-4 gap-y-2 rounded-[6px] border border-line px-3 py-2">
            <span className="text-[13px] text-ink2">
              Finished as <span className="text-ink">Take {ft.num} · {ft.label.replace(/^Finish\s+/, "") || ft.label}</span>
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void approveFinishedTake(d.shot_id, d.take.id, finished.id)}
              title={`Approve Take ${ft.num} in place of Take ${t?.num ?? ""}`}
            >
              Use this
            </Button>
          </div>
        ) : null}
        {!isOriginal(d.take) || isLocalHost() ? (
          <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 px-2">
            {isOriginal(d.take) ? null : <ReceiptLink shotId={d.shot_id} version={d.take} variant="link" />}
            {isLocalHost() ? (
              <button type="button" onClick={() => void revealTake(d.shot_id, d.take.id)} className="text-[12px] text-ink3 underline-offset-4 hover:text-ivory hover:underline">
                Reveal in Finder
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </li>
  );
}

/** What Deliveries exports: a timeline (XML / EDL) or the receipts. */
type ExportKind = TimelineFormat | "receipts";

/** GET /api/projects/{id}/export/{xml|edl} or /receipts: a file download; 409 = nothing approved yet (said calmly). */
function useProjectExport(id: string) {
  const [busy, setBusy] = useState<ExportKind | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [problem, setProblem] = useState<{ text: string; calm: boolean; kind: ExportKind } | null>(null);
  const timer = useRef<number>();
  // One export at a time: both controls are disabled while one runs, and a run only clears its own busy state.
  const running = useRef(false);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const run = async (kind: ExportKind) => {
    if (running.current) return;
    running.current = true;
    setBusy(kind);
    setProblem(null);
    setSaved(null);
    try {
      const { blob, filename } = kind === "receipts" ? await getApi().exportReceipts(id) : await getApi().exportTimeline(id, kind);
      if (!TOUR) {
        // The tour shows the result without saving a file on the viewer's machine.
        const href = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = href;
        a.download = filename;
        a.rel = "noopener";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.setTimeout(() => URL.revokeObjectURL(href), 10_000);
      }
      setSaved(filename);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setSaved(null), 6000);
    } catch (e) {
      const calm = e instanceof ApiError && e.status === 409;
      setProblem({
        text: e instanceof ApiError && e.message ? e.message : kind === "receipts" ? "The receipts didn't export. Try again." : "The timeline didn't export. Try again.",
        calm,
        kind,
      });
      if (calm) {
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setProblem(null), 8000);
      }
    } finally {
      running.current = false;
      setBusy((b) => (b === kind ? null : b));
    }
  };
  return { busy, saved, problem, run, clear: () => setProblem(null) };
}

function DeliveriesTab({ id }: { id: string }) {
  const list = useApp((s) => s.deliveries[id] ?? null);
  const shots = useApp((s) => s.shots);
  const jobs = useApp((s) => s.jobs);
  const eta = useApp((s) => s.status?.tiles.finish?.eta_s ?? null);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void loadDeliveries(id);
  }, [id]);
  // Each Finish is a long GPU job, so "Finish all approved" says how many and roughly how long before it queues.
  const candidates = useMemo(() => finishCandidates(list ?? [], shots, jobs), [list, shots, jobs]);
  const n = candidates.length;
  const ncQueued = candidates.filter((d) => nonCommercialModels(d.take).length).length;
  // Finished takes not yet approved in place of the takes they were made from ("Use this" on their cards).
  const waiting = (list ?? []).filter((d) => finishedChild(shots[d.shot_id], d.take)).length;
  const ncApproved = (list ?? []).filter((d) => !isOriginal(d.take) && nonCommercialModels(d.take).length).length;
  useEffect(() => {
    if (!n && !busy) setConfirming(false);
  }, [n, busy]);
  // Long totals read in hours ("about 2 h 40 min"), to the nearest 5 min.
  const total = eta ? n * eta : 0;
  const about = !total ? null : total >= 90 * 60 ? `About ${duration(Math.round(total / 300) * 300)}` : aboutTime(total);
  const exp = useProjectExport(id);
  return (
    <div>
      <div className="mb-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <span />
          <div className="flex flex-wrap items-center gap-2">
            <MenuPill
              label="Export timeline"
              icon={FileDown}
              align="right"
              loading={exp.busy === "xml" || exp.busy === "edl"}
              disabled={exp.busy === "receipts"}
              menuLabel="Export timeline as"
              items={[
                { id: "xml", label: "Resolve / Premiere (XML)", onSelect: () => void exp.run("xml") },
                { id: "edl", label: "EDL (CMX3600)", onSelect: () => void exp.run("edl") },
              ]}
            />
            <Button
              variant="secondary"
              size="md"
              icon={ReceiptText}
              loading={exp.busy === "receipts"}
              disabled={exp.busy === "xml" || exp.busy === "edl"}
              onClick={() => void exp.run("receipts")}
              title="A printable receipt for every approved take: what changed, models and licenses, prompts, timecode, files and workflows"
            >
              Export receipts
            </Button>
            {confirming ? (
              <div
                className="flex flex-wrap items-center gap-2 animate-fade-in"
                role="group"
                aria-label="Finish all approved"
                onKeyDown={(e) => {
                  if (e.key !== "Escape" || busy) return;
                  e.preventDefault();
                  e.stopPropagation();
                  setConfirming(false);
                }}
              >
                <span className="mx-1 text-[14px] text-ink">
                  Finish {n} approved {n === 1 ? "take" : "takes"}
                  {about ? <span className="text-ink2"> · {about.charAt(0).toLowerCase() + about.slice(1)} of GPU</span> : null}
                  {ncQueued ? <span className="text-ink2"> · {n === 1 ? "it uses" : `${ncQueued} ${ncQueued === 1 ? "uses" : "use"}`} a non-commercial model</span> : null}
                </span>
                <Button
                  variant="primary"
                  size="md"
                  loading={busy}
                  onClick={async () => {
                    setBusy(true);
                    await finishApproved(id);
                    setBusy(false);
                    setConfirming(false);
                  }}
                >
                  Queue
                </Button>
                <Button variant="ghost" size="md" autoFocus onClick={() => setConfirming(false)}>
                  Cancel
                </Button>
              </div>
            ) : list?.length && !n ? (
              <span className={cx("inline-flex h-9 items-center font-mono text-[12px]", waiting ? "text-ink" : "text-ink3")}>
                {waiting
                  ? `${waiting} finished ${waiting === 1 ? "take is" : "takes are"} waiting for approval`
                  : "Every approved take is finished or finishing."}
              </span>
            ) : (
              <Button variant="secondary" size="md" disabled={!n} onClick={() => setConfirming(true)}>
                Finish all approved
              </Button>
            )}
          </div>
        </div>
        <p className={cx("mt-3 font-mono text-[12px] sm:text-right", exp.problem?.calm ? "text-ink" : "text-ink3")} role="status" aria-live="polite">
          {exp.problem?.calm
            ? exp.problem.text
            : exp.busy === "receipts"
              ? "Making receipts…"
              : exp.saved
                ? `Saved ${exp.saved}`
                : "Timeline uses source timecode."}
        </p>
        {ncApproved ? (
          <p className="mt-1 font-mono text-[12px] text-ink3 sm:text-right">
            {ncApproved} of {list?.length ?? 0} approved {list?.length === 1 ? "take" : "takes"} {ncApproved === 1 ? "uses" : "use"} a non-commercial model
          </p>
        ) : null}
        {exp.problem && !exp.problem.calm ? (
          <Notice
            className="mt-4"
            tone="bad"
            text={exp.problem.text}
            action={{ label: "Try again", onClick: () => void exp.run(exp.problem!.kind) }}
            secondary={{ label: "Dismiss", onClick: exp.clear }}
          />
        ) : null}
      </div>
      {list === null ? (
        <div className="py-10 text-ink3">
          <Spinner />
        </div>
      ) : !list.length ? (
        <Notice text="Nothing approved yet. Open a take and press Approve under the picture (⇧A)." action={{ label: "Go to shots", onClick: () => navigate({ name: "project", id, tab: "shots" }) }} />
      ) : (
        <ul className="border-b border-line">
          {list.map((d) => (
            <DeliveryRow key={`${d.shot_id}:${d.take.id}`} d={d} />
          ))}
        </ul>
      )}
    </div>
  );
}
