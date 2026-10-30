// The top bar. In the side layout it is the breadcrumb (Projects / Stunt previz / K03 hood push), the demo tag
// and the machine; the links live in the left panel. In the top layout those links are the bar itself and the
// panel is gone. Side / Top switches between them.
import { Menu } from "lucide-react";
import { Fragment, useSyncExternalStore, type ReactNode } from "react";
import { navigate, openOnboarding, openPreviz, openSetup, setChrome, setNavDrawer } from "../actions";
import { LAYOUT_SWITCH } from "../lib/features";
import { pillState, statusLine } from "../lib/format";
import { leaveDemo } from "../lib/demo";
import { TOUR } from "../lib/tour";
import { useApp, type Route } from "../store";
import { QueueMenu } from "./QueueBar";
import { ShotTitle } from "./ShotMenu";
import { Badge, Button, Segmented, StateDot, Tip } from "./ui";

/** The mark on the i. From the yellow in the logo sheet. */
const MARK = "#F2FD74";

/** "pickups", bold and lowercase. The dot on the i is the yellow mark. */
export function Wordmark({ size = 32 }: { size?: number }) {
  const dot = Math.max(4, Math.round(size * 0.2));
  return (
    <span className="inline-flex items-baseline font-sans font-bold lowercase leading-none tracking-[-0.045em] text-ivory" style={{ fontSize: size }}>
      p
      <span className="relative inline-block">
        i
        <span
          aria-hidden="true"
          className="absolute left-1/2 -translate-x-1/2 rounded-full"
          style={{ width: dot, height: dot, top: size * 0.05, background: MARK }}
        />
      </span>
      ckups
    </span>
  );
}

// Below 1000 px the full status line ("STUDIO · RTX 4090 · BUSY WITH ANOTHER COMFYUI JOB") would truncate mid-word.
const narrowQuery = typeof window !== "undefined" ? window.matchMedia("not all and (min-width: 1000px)") : null;

function useNarrow(): boolean {
  return useSyncExternalStore(
    (on) => {
      narrowQuery?.addEventListener("change", on);
      return () => narrowQuery?.removeEventListener("change", on);
    },
    () => !!narrowQuery?.matches,
  );
}

/** The state alone ("Busy", "Ready"), for a narrow top bar. */
function shortStatus(state: ReturnType<typeof pillState>, busy: boolean): string {
  if (state === "checking") return "Checking…";
  if (state === "offline") return "Connect ComfyUI";
  if (state === "setup") return "Setup needed";
  return busy ? "Busy" : "Ready";
}

function StatusButton() {
  const status = useApp((s) => s.status);
  const failed = useApp((s) => s.statusFailed);
  const narrow = useNarrow();
  const state = pillState(status, failed);
  const busy = !!status?.comfy_queue?.outside;
  const line = statusLine(status, failed);
  const mode = useApp((s) => s.mode);
  return (
    // Narrow: the button says just the state, and the tooltip (and aria-label) carry the whole line.
    <Tip text={narrow ? line : "Machine, models and benchmarks"} side="bottom" align="end" className="min-w-0 max-[640px]:hidden">
      <button
        type="button"
        onClick={() => (state === "offline" && mode === "live" ? openOnboarding("connect") : openSetup())}
        aria-label={`Setup: ${line}`}
        className="slate flex h-8 min-w-0 items-center gap-2 whitespace-nowrap rounded-full px-3 text-ink2 transition-colors duration-150 hover:bg-raised hover:text-ivory"
      >
        <StateDot on={state === "ready" && !busy} />
        <span className="truncate">{narrow ? shortStatus(state, busy) : line}</span>
      </button>
    </Tip>
  );
}

const TAB_LABEL = { shots: "Shots", images: "Images", cast: "Cast", deliveries: "Deliveries" } as const;

/** Each crumb: a link, or (the last) the page itself. */
type Crumb = { label: string; href?: string; node?: ReactNode };

function useCrumbs(): Crumb[] {
  const route = useApp((s) => s.route);
  const projects = useApp((s) => s.projects);
  const shot = useApp((s) => (route.name === "shot" ? s.shots[route.id] : undefined));
  const nameOf = (id: string) => projects?.find((p) => p.id === id)?.name ?? (id === "unsorted" ? "My shots" : id === "examples" ? "Examples" : "Project");
  const projectCrumb = (id: string, last = false): Crumb => ({ label: nameOf(id), href: last ? undefined : `#/project/${encodeURIComponent(id)}` });
  const all: Crumb = { label: "Projects", href: "#/projects" };
  switch (route.name) {
    case "home":
      return [{ label: "Home" }];
    case "projects":
      return [{ label: "Projects" }];
    case "project":
      return route.tab === "shots" ? [all, projectCrumb(route.id, true)] : [all, projectCrumb(route.id), { label: TAB_LABEL[route.tab] }];
    case "create":
      if (route.panel === "home") return [{ label: "Create" }];
      return [{ label: "Create", href: "#/create" }, { label: route.panel === "image" ? "Image" : "Edit a clip" }];
    case "assets":
      return [{ label: "Assets" }];
    case "previz":
      return [{ label: "Previz" }];
    case "shot":
      if (!shot) return [all, { label: "Loading…" }];
      return [all, ...(shot.project_id ? [projectCrumb(shot.project_id)] : []), { label: shot.name, node: <ShotTitle shot={shot} /> }];
  }
}

function Breadcrumb() {
  const crumbs = useCrumbs();
  return (
    <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2 text-[14px]">
      {crumbs.map((c, i) => {
        const last = i === crumbs.length - 1;
        return (
          <Fragment key={`${i}-${c.label}`}>
            {i > 0 ? (
              <span className={i === 1 ? "text-ink4 max-[767px]:hidden" : "text-ink4"} aria-hidden="true">
                /
              </span>
            ) : null}
            {/* On a phone the first crumb ("Projects") goes: the panel has it. */}
            <span className={i === 0 && crumbs.length > 1 ? "contents max-[767px]:hidden" : "contents"}>
              {c.node ? (
                <span className="min-w-0">{c.node}</span>
              ) : c.href && !last ? (
                <a href={c.href} className="min-w-0 shrink truncate text-ink3 transition-colors duration-150 hover:text-ivory">
                  {c.label}
                </a>
              ) : (
                <span className="min-w-0 truncate text-ink" aria-current="page" title={c.label}>
                  {c.label}
                </span>
              )}
            </span>
          </Fragment>
        );
      })}
    </nav>
  );
}

function LayoutSwitch() {
  const chrome = useApp((s) => s.chrome);
  return (
    <Segmented
      mono
      label="Layout"
      value={chrome}
      onChange={setChrome}
      options={[
        { value: "side", label: "Side" },
        { value: "top", label: "Top" },
      ]}
    />
  );
}

function BarTools() {
  const mode = useApp((s) => s.mode);
  return (
    <div className="flex min-w-0 shrink-0 items-center gap-3">
      {LAYOUT_SWITCH ? (
        // In the demo, Exit demo takes the room: below 1400 px the layout switch steps aside so the nav keeps Setup.
        <span className={mode === "mock" && !TOUR ? "max-[1399px]:hidden" : "max-[767px]:hidden"}>
          <LayoutSwitch />
        </span>
      ) : null}
      {mode === "mock" && !TOUR ? (
        <Button variant="ghost" size="sm" className="shrink-0 whitespace-nowrap" onClick={leaveDemo}>
          Exit demo
        </Button>
      ) : null}
      <QueueMenu />
      <StatusButton />
    </div>
  );
}

/** The current section, so the top bar can light the same word the side panel would. */
function sectionOf(route: Route): "home" | "create" | "previz" | "projects" | "assets" | null {
  if (route.name === "home") return "home";
  if (route.name === "create") return "create";
  if (route.name === "previz") return "previz";
  if (route.name === "assets") return "assets";
  if (route.name === "projects" || route.name === "project" || route.name === "shot") return "projects";
  return null;
}

function TopLink({ label, href, active, onClick, dot, badge }: { label: string; href?: string; active: boolean; onClick: () => void; dot?: boolean; badge?: ReactNode }) {
  const cls = `relative inline-flex h-14 shrink-0 items-center px-2.5 font-sans text-xs font-normal uppercase tracking-[0.14em] transition-colors duration-150 ${active ? "text-ivory" : "text-ink3 hover:text-ivory"}`;
  const mark = active ? <span className="absolute inset-x-2.5 bottom-0 h-px bg-ivory" aria-hidden="true" /> : null;
  const inner = (
    <>
      {label}
      {dot ? <span className="ml-1.5"><StateDot on={false} /></span> : null}
      {/* in the bottom padding, under the word and clear of the underline: the row's height and spacing don't move */}
      {badge ? <span className="pointer-events-none absolute bottom-[5px] left-1/2 -translate-x-1/2">{badge}</span> : null}
      {mark}
    </>
  );
  if (!href) {
    return (
      <button type="button" onClick={onClick} aria-pressed={active} className={cls}>
        {inner}
      </button>
    );
  }
  return (
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
  );
}

/** Links across the top, and no left panel. Assets stays, so that page is still one click away. */
function TopNav() {
  const route = useApp((s) => s.route);
  const setupOpen = useApp((s) => s.setup.open);
  const status = useApp((s) => s.status);
  const failed = useApp((s) => s.statusFailed);
  const section = setupOpen ? null : sectionOf(route);
  const ready = pillState(status, failed) === "ready" && !status?.comfy_queue?.outside;
  const crumbs = useCrumbs();
  const nested = crumbs.length > 1;
  return (
    <header className="relative z-20 shrink-0 border-b border-line bg-bg">
      <div className="flex h-14 items-center gap-4 pl-5 pr-4 max-[767px]:gap-2 max-[767px]:pl-3 max-[767px]:pr-2">
        <a
          href="#/"
          onClick={(e) => {
            e.preventDefault();
            navigate({ name: "home" });
          }}
          className="shrink-0 rounded-[4px]"
          aria-label="Pickups home"
        >
          <Wordmark size={26} />
        </a>
        <nav aria-label="Go to" className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto [scrollbar-width:none]">
          <TopLink label="Home" href="#/" active={section === "home"} onClick={() => navigate({ name: "home" })} />
          <TopLink label="Create" href="#/create" active={section === "create"} onClick={() => navigate({ name: "create", panel: "home" })} />
          <TopLink label="Previz" href="#/previz" active={section === "previz"} onClick={() => openPreviz()} badge={<Badge>Beta</Badge>} />
          <TopLink label="Projects" href="#/projects" active={section === "projects"} onClick={() => navigate({ name: "projects" })} />
          <TopLink label="Assets" href="#/assets" active={section === "assets"} onClick={() => navigate({ name: "assets" })} />
          <TopLink label="Setup" active={setupOpen} onClick={() => openSetup()} dot={!!status && !ready} />
        </nav>
        <BarTools />
      </div>
      {nested ? (
        <div className="flex h-9 items-center border-t border-line px-5 max-[767px]:px-3">
          <Breadcrumb />
        </div>
      ) : null}
    </header>
  );
}

function SideChrome() {
  const home = useApp((s) => s.route.name === "home");
  return (
    <header className="relative z-20 flex h-14 shrink-0 items-center gap-3 border-b border-line bg-bg pl-6 pr-4 max-[767px]:pl-2 max-[767px]:pr-3">
      <button
        type="button"
        onClick={() => setNavDrawer(true)}
        aria-label="Open menu"
        className="hidden h-10 w-10 shrink-0 items-center justify-center rounded-full text-ink2 transition-colors hover:bg-raised hover:text-ivory max-[767px]:inline-flex"
      >
        <Menu size={19} strokeWidth={1.5} aria-hidden="true" />
      </button>
      <a
        href="#/"
        onClick={(e) => {
          e.preventDefault();
          navigate({ name: "home" });
        }}
        className={home ? "hidden shrink-0 rounded-[4px] max-[767px]:block" : "hidden"}
        aria-label="Pickups home"
      >
        <Wordmark size={26} />
      </a>
      <div className={home ? "min-w-0 flex-1 max-[767px]:invisible" : "min-w-0 flex-1 max-[767px]:pl-1"}>
        <Breadcrumb />
      </div>
      <BarTools />
    </header>
  );
}

export function TopBar() {
  const chrome = useApp((s) => s.chrome);
  return chrome === "top" ? <TopNav /> : <SideChrome />;
}
