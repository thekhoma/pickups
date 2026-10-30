// The left panel: where you are and where your work lives (it replaces the top bar's project dropdown).
// Expanded: Home, Create, Previz, Assets, Setup, then your projects with a thumbnail and counts; the project you're in opens
// into its sections (Shots, Images, Cast, Deliveries). "All" opens the projects page. Collapsed: an icon rail
// with recent projects as thumbnails. Expand sits under the mark, in the same top band as Collapse.
// The choice is remembered per browser. Below 1024 px it stays collapsed; on a phone it's a drawer (the top bar's menu).
import { Box, Folder, House, LayoutGrid, Library, PanelLeftClose, PanelLeftOpen, Plus, SlidersHorizontal, Sparkles, X, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { navigate, newProject, openPreviz, openProject, openSetup, setNavCollapsed, setNavDrawer } from "../actions";
import { cx, pillState, shortGpu } from "../lib/format";
import { useMedia } from "../lib/hooks";
import { useApp, type ProjectTab, type Route } from "../store";
import type { Project, Shot } from "../types";
import { Thumb } from "./MediaCard";
import { Wordmark } from "./TopBar";
import { Badge, Button, StateDot } from "./ui";

const SECTIONS: Array<{ id: ProjectTab; label: string }> = [
  { id: "shots", label: "Shots" },
  { id: "images", label: "Images" },
  { id: "cast", label: "Cast" },
  { id: "deliveries", label: "Deliveries" },
];

/** "3 shots · 10 takes": short enough for the panel. */
function shortCounts(p: Project): string {
  const n = (x: number, one: string, many: string) => `${x} ${x === 1 ? one : many}`;
  return p.takes ? `${n(p.shots, "shot", "shots")} · ${n(p.takes, "take", "takes")}` : n(p.shots, "shot", "shots");
}

/** The project on screen: a project page, or the open shot's project. */
export function viewedProjectId(route: Route, shots: Record<string, Shot>): string | null {
  if (route.name === "project") return route.id;
  if (route.name === "previz") return route.projectId;
  if (route.name === "shot") return shots[route.id]?.project_id ?? null;
  return null;
}

/** My shots first (where work goes until you make a project), then your projects, newest first. Never Examples. */
function usePanelProjects(): Project[] | null {
  const projects = useApp((s) => s.projects);
  return useMemo(() => {
    if (!projects) return null;
    const live = projects.filter((p) => !p.archived && p.id !== "examples");
    const mine = live.find((p) => p.id === "unsorted");
    return [...(mine ? [mine] : []), ...live.filter((p) => p.id !== "unsorted")];
  }, [projects]);
}

export function Sidebar() {
  const collapsedPref = useApp((s) => s.nav.collapsed);
  const drawer = useApp((s) => s.nav.drawer);
  const route = useApp((s) => s.route);
  const phone = useMedia("(max-width: 767px)");
  const narrow = useMedia("(max-width: 1023px)");

  // A route change (or growing past phone width) closes the drawer.
  useEffect(() => {
    setNavDrawer(false);
  }, [route]);
  useEffect(() => {
    if (!phone) setNavDrawer(false);
  }, [phone]);

  if (phone) return drawer ? <Drawer /> : null;
  const collapsed = collapsedPref || narrow;
  return (
    <aside
      aria-label="Panel"
      className={cx(
        "relative z-30 flex h-full shrink-0 flex-col border-r border-line bg-panel transition-[width] duration-150 ease-out",
        collapsed ? "w-[60px]" : "w-[256px]",
      )}
    >
      {collapsed ? <Rail canExpand={!narrow} /> : <Panel />}
    </aside>
  );
}

// --- phone drawer -----------------------------------------------------------------------------------------------

function Drawer() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLElement>("a, button")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setNavDrawer(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      prev?.focus?.();
    };
  }, []);
  return (
    <div className="fixed inset-0 z-[70] flex">
      <div className="absolute inset-0 bg-black/70 animate-fade-in" onClick={() => setNavDrawer(false)} aria-hidden="true" />
      <div ref={ref} role="dialog" aria-modal="true" aria-label="Menu" className="relative flex h-full w-[min(320px,86vw)] flex-col border-r border-line bg-panel animate-slide-in">
        <Panel phone />
      </div>
    </div>
  );
}

// --- expanded ---------------------------------------------------------------------------------------------------

function Panel({ phone }: { phone?: boolean }) {
  const route = useApp((s) => s.route);
  const setupOpen = useApp((s) => s.setup.open);
  const status = useApp((s) => s.status);
  const failed = useApp((s) => s.statusFailed);
  const projects = usePanelProjects();
  const currentId = useApp((s) => s.currentProjectId);
  const shots = useApp((s) => s.shots);
  const viewed = viewedProjectId(route, shots);
  const treeId = viewed ?? currentId;
  const [adding, setAdding] = useState(() => pendingNewProject.current);
  useEffect(() => {
    pendingNewProject.current = false;
  }, []);
  const ready = pillState(status, failed) === "ready" && !status?.comfy_queue?.outside;
  const shown = (projects ?? []).slice(0, 10);
  const more = (projects ?? []).length - shown.length;

  return (
    <>
      <div className="flex h-14 shrink-0 items-center justify-between gap-2 border-b border-line pl-4 pr-2">
        <a
          href="#/"
          onClick={(e) => {
            e.preventDefault();
            navigate({ name: "home" });
          }}
          className="rounded-[4px]"
          aria-label="Pickups home"
        >
          <Wordmark size={26} />
        </a>
        {phone ? (
          <PanelButton icon={X} label="Close menu" onClick={() => setNavDrawer(false)} />
        ) : (
          <PanelButton icon={PanelLeftClose} label="Collapse panel" onClick={() => setNavCollapsed(true)} />
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-4 pt-3">
        <ul className="space-y-0.5 px-2" aria-label="Go to">
          <NavRow icon={House} label="Home" href="#/" active={route.name === "home" && !setupOpen} onClick={() => navigate({ name: "home" })} />
          <NavRow icon={Sparkles} label="Create" href="#/create" active={route.name === "create" && !setupOpen} onClick={() => navigate({ name: "create", panel: "home" })} />
          <NavRow icon={Box} label="Previz" href="#/previz" active={route.name === "previz" && !setupOpen} onClick={() => openPreviz()} badge={<Badge>Beta</Badge>} />
          <NavRow icon={LayoutGrid} label="Projects" href="#/projects" active={route.name === "projects" && !setupOpen} onClick={() => navigate({ name: "projects" })} />
          <NavRow icon={Library} label="Assets" href="#/assets" active={route.name === "assets" && !setupOpen} onClick={() => navigate({ name: "assets" })} />
          <NavRow icon={SlidersHorizontal} label="Setup" active={setupOpen} onClick={() => openSetup()} right={status && !ready ? <StateDot on={false} /> : null} />
        </ul>

        <div className="mt-6 flex items-center justify-between px-4">
          <span className="slate text-ink3">Projects</span>
          <a href="#/projects" className="slate text-ink3 transition-colors hover:text-ivory" aria-label={`All projects${more > 0 ? `, ${more} more` : ""}`} aria-current={route.name === "projects" && !setupOpen ? "page" : undefined}>
            {more > 0 ? `All · ${more} more` : "All"}
          </a>
        </div>
        <ul className="mt-2 space-y-0.5 px-2" aria-label="Projects">
          {projects === null
            ? [0, 1, 2].map((i) => (
                <li key={i} className="flex items-center gap-2.5 px-2 py-1.5">
                  <span className="skeleton h-7 w-11 rounded-[3px]" />
                  <span className="skeleton h-3 w-24 rounded-[2px]" />
                </li>
              ))
            : shown.map((p) => <ProjectRow key={p.id} p={p} open={p.id === treeId} current={p.id === currentId} route={route} />)}
        </ul>
        <div className="mt-1 px-2">
          {adding ? (
            <NewProjectForm onDone={() => setAdding(false)} />
          ) : (
            <button
              type="button"
              onClick={() => setAdding(true)}
              className="flex h-9 w-full items-center gap-2.5 rounded-[5px] px-2 text-left text-[13px] text-ink2 transition-colors duration-150 hover:bg-raised hover:text-ivory"
            >
              <span className="flex h-7 w-11 shrink-0 items-center justify-center rounded-[3px] border border-dashed border-line2">
                <Plus size={13} strokeWidth={1.5} aria-hidden="true" />
              </span>
              New project
            </button>
          )}
        </div>
      </div>

      {phone ? null : (
        <div className="shrink-0 border-t border-line px-4 py-3">
          <p className="slate truncate text-ink4">{ready && status?.machine?.gpu ? `${shortGpu(status.machine.gpu)} · Ready` : ["offline", "setup"].includes(pillState(status, failed)) ? "Setup needed" : "Open source · ComfyUI"}</p>
        </div>
      )}
    </>
  );
}

/** "+" on the collapsed rail opens the panel straight into "New project". */
const pendingNewProject = { current: false };

function NewProjectForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    const p = await newProject(name.trim(), true);
    setBusy(false);
    if (p) onDone();
  };
  return (
    <form onSubmit={submit} className="flex items-center gap-1.5 rounded-[5px] bg-raised p-1.5">
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            onDone();
          }
        }}
        placeholder="Project name"
        aria-label="New project name"
        className="field h-8 min-w-0 flex-1 px-2.5 text-[13px]"
      />
      <Button type="submit" variant="primary" size="sm" className="shrink-0" loading={busy} disabled={!name.trim() || busy}>
        Create
      </Button>
    </form>
  );
}

function NavRow({ icon: I, label, href, active, onClick, right, badge }: { icon: LucideIcon; label: string; href?: string; active?: boolean; onClick: () => void; right?: ReactNode; badge?: ReactNode }) {
  const cls = cx(
    "flex h-9 w-full items-center gap-3 rounded-[5px] px-2.5 text-left text-[14px] transition-colors duration-150",
    active ? "bg-raised text-ivory" : "text-ink2 hover:bg-raised/70 hover:text-ivory",
  );
  const inner = (
    <>
      <I size={17} strokeWidth={1.5} aria-hidden="true" className="shrink-0" />
      <span className="flex flex-1 items-center gap-2">
        {label}
        {badge}
      </span>
      {right}
    </>
  );
  return (
    <li>
      {href ? (
        <a
          href={href}
          aria-current={active ? "page" : undefined}
          onClick={(e) => {
            e.preventDefault();
            onClick();
          }}
          className={cls}
        >
          {inner}
        </a>
      ) : (
        <button type="button" onClick={onClick} aria-pressed={active} className={cls}>
          {inner}
        </button>
      )}
    </li>
  );
}

function ProjectRow({ p, open, current, route }: { p: Project; open: boolean; current: boolean; route: Route }) {
  const castCount = useApp((s) => s.cast[p.id]?.length);
  const imageCount = useApp((s) => s.images[p.id]?.length);
  const counts: Record<ProjectTab, number | undefined> = { shots: p.shots, images: imageCount ?? p.images, cast: castCount ?? p.cast, deliveries: p.approved };
  const tab = route.name === "project" && route.id === p.id ? route.tab : null;
  return (
    <li>
      <a
        href={`#/project/${encodeURIComponent(p.id)}`}
        onClick={(e) => {
          e.preventDefault();
          openProject(p.id);
        }}
        aria-current={open && tab === "shots" ? "page" : undefined}
        aria-label={`${p.name}${current ? " (new work goes here)" : ""}, ${shortCounts(p)}`}
        className={cx(
          "group flex items-center gap-2.5 rounded-[5px] px-2 py-1.5 transition-colors duration-150",
          open ? "bg-raised" : "hover:bg-raised/70",
        )}
      >
        <Cover p={p} className="h-7 w-11" />
        <span className="min-w-0 flex-1">
          <span className={cx("block truncate text-[14px] leading-5", open ? "text-ivory" : "text-ink group-hover:text-ivory")}>{p.name}</span>
          {/* open, its sections below carry the counts */}
          {open ? null : <span className="block truncate font-mono text-[10px] uppercase leading-4 tracking-[0.06em] text-ink3">{shortCounts(p)}</span>}
        </span>
      </a>
      {open ? (
        <ul className="mb-1.5 ml-[27px] mt-0.5 border-l border-line pl-2" aria-label={`${p.name} sections`}>
          {SECTIONS.map((s) => {
            const on = tab === s.id;
            return (
              <li key={s.id}>
                <a
                  href={`#/project/${encodeURIComponent(p.id)}${s.id === "shots" ? "" : `/${s.id}`}`}
                  onClick={(e) => {
                    e.preventDefault();
                    openProject(p.id, s.id);
                  }}
                  aria-current={on ? "page" : undefined}
                  className={cx(
                    "flex h-8 items-center justify-between gap-2 rounded-[4px] px-2.5 text-[13px] transition-colors duration-150",
                    on ? "bg-line/70 text-ivory" : "text-ink2 hover:bg-raised/70 hover:text-ivory",
                  )}
                >
                  {s.label}
                  {counts[s.id] != null ? <span className="mono text-[11px] text-ink3">{counts[s.id]}</span> : null}
                </a>
              </li>
            );
          })}
        </ul>
      ) : null}
    </li>
  );
}

/** A project's cover, or a quiet folder when it has no shots yet. */
function Cover({ p, className }: { p: Project; className: string }) {
  if (!p.cover_url) {
    return (
      <span className={cx("flex shrink-0 items-center justify-center rounded-[3px] border border-line bg-raised text-ink3", className)} aria-hidden="true">
        <Folder size={13} strokeWidth={1.5} />
      </span>
    );
  }
  return <Thumb src={p.cover_url} className={cx("shrink-0 rounded-[3px]", className)} />;
}

function PanelButton({ icon: I, label, onClick }: { icon: LucideIcon; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-ink3 transition-colors duration-150 hover:bg-raised hover:text-ivory"
    >
      <I size={17} strokeWidth={1.5} aria-hidden="true" />
    </button>
  );
}

// --- collapsed rail ---------------------------------------------------------------------------------------------

function Rail({ canExpand }: { canExpand: boolean }) {
  const route = useApp((s) => s.route);
  const setupOpen = useApp((s) => s.setup.open);
  const projects = usePanelProjects();
  const currentId = useApp((s) => s.currentProjectId);
  const shots = useApp((s) => s.shots);
  const viewed = viewedProjectId(route, shots) ?? currentId;
  const [tip, setTip] = useState<{ text: string; y: number } | null>(null);
  const show = (text: string) => (e: { currentTarget: HTMLElement }) => {
    const r = e.currentTarget.getBoundingClientRect();
    setTip({ text, y: r.top + r.height / 2 });
  };
  const hide = () => setTip(null);
  const tipProps = (text: string) => ({ onMouseEnter: show(text), onFocus: show(text), onMouseLeave: hide, onBlur: hide });
  const icon = "flex h-10 w-10 items-center justify-center rounded-[6px] transition-colors duration-150";

  const go = (r: Route, e: { preventDefault: () => void }) => {
    e.preventDefault();
    hide();
    navigate(r);
  };

  return (
    <>
      <div className="flex h-14 shrink-0 items-center justify-center border-b border-line">
        <a href="#/" onClick={(e) => go({ name: "home" }, e)} aria-label="Pickups home" className="flex items-baseline gap-[2px] rounded-[4px] px-1" {...tipProps("Home")}>
          <span className="font-sans text-[22px] font-bold lowercase leading-none tracking-[-0.04em] text-ivory">p</span>
        </a>
      </div>
      {canExpand ? (
        <div className="flex shrink-0 justify-center pt-2">
          <button
            type="button"
            onClick={() => {
              hide();
              setNavCollapsed(false);
            }}
            aria-label="Expand panel"
            className="inline-flex h-9 w-9 items-center justify-center rounded-full text-ink3 transition-colors duration-150 hover:bg-raised hover:text-ivory"
            {...tipProps("Expand panel")}
          >
            <PanelLeftOpen size={17} strokeWidth={1.5} aria-hidden="true" />
          </button>
        </div>
      ) : null}
      <div className="flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto overflow-x-hidden py-3" onScroll={hide}>
        {(
          [
            [House, "Home", "#/", { name: "home" }, route.name === "home"],
            [Sparkles, "Create", "#/create", { name: "create", panel: "home" }, route.name === "create"],
            [Box, "Previz", "#/previz", { name: "previz", projectId: null }, route.name === "previz"],
            [LayoutGrid, "Projects", "#/projects", { name: "projects" }, route.name === "projects"],
            [Library, "Assets", "#/assets", { name: "assets" }, route.name === "assets"],
          ] as Array<[LucideIcon, string, string, Route, boolean]>
        ).map(([I, label, href, r, on]) => (
          <a
            key={label}
            href={href}
            onClick={(e) => {
              if (label === "Previz") {
                e.preventDefault();
                hide();
                openPreviz();
                return;
              }
              go(r, e);
            }}
            aria-label={label}
            aria-current={on && !setupOpen ? "page" : undefined}
            className={cx(icon, on && !setupOpen ? "bg-raised text-ivory" : "text-ink2 hover:bg-raised/70 hover:text-ivory")}
            {...tipProps(label)}
          >
            <I size={18} strokeWidth={1.5} aria-hidden="true" />
          </a>
        ))}
        <button
          type="button"
          onClick={() => {
            hide();
            openSetup();
          }}
          aria-label="Setup"
          aria-pressed={setupOpen}
          className={cx(icon, setupOpen ? "bg-raised text-ivory" : "text-ink2 hover:bg-raised/70 hover:text-ivory")}
          {...tipProps("Setup")}
        >
          <SlidersHorizontal size={18} strokeWidth={1.5} aria-hidden="true" />
        </button>

        <span className="my-2 h-px w-6 shrink-0 bg-line" aria-hidden="true" />
        <ul className="flex flex-col items-center gap-1.5" aria-label="Projects">
          {(projects ?? []).slice(0, 7).map((p) => {
            const on = p.id === viewed;
            return (
              <li key={p.id}>
                <a
                  href={`#/project/${encodeURIComponent(p.id)}`}
                  onClick={(e) => {
                    e.preventDefault();
                    hide();
                    openProject(p.id);
                  }}
                  aria-label={`${p.name}${p.id === currentId ? " (new work goes here)" : ""}`}
                  aria-current={on ? "true" : undefined}
                  className={cx("block rounded-[5px] p-[2px] transition-shadow duration-150", on ? "shadow-[0_0_0_1.5px_rgba(238,233,224,0.85)]" : "hover:shadow-[0_0_0_1px_#3A3631]")}
                  {...tipProps(p.name)}
                >
                  <Cover p={p} className="h-8 w-8" />
                </a>
              </li>
            );
          })}
        </ul>
        <button
          type="button"
          onClick={() => {
            hide();
            if (!canExpand) {
              navigate({ name: "projects" });
              return;
            }
            pendingNewProject.current = true;
            setNavCollapsed(false);
          }}
          aria-label="New project"
          className={cx(icon, "mt-1 h-9 w-9 border border-dashed border-line2 text-ink3 hover:border-ivory/50 hover:text-ivory")}
          {...tipProps("New project")}
        >
          <Plus size={15} strokeWidth={1.5} aria-hidden="true" />
        </button>
      </div>
      {tip ? (
        <span
          role="tooltip"
          className="pointer-events-none fixed left-[66px] z-[80] -translate-y-1/2 whitespace-nowrap rounded-[4px] border border-line bg-raised px-2 py-1 font-mono text-[11px] uppercase tracking-[0.06em] text-ink animate-fade-in"
          style={{ top: tip.y }}
        >
          {tip.text}
        </span>
      ) : null}
    </>
  );
}
