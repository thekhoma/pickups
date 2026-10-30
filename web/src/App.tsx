import { ArrowLeft } from "lucide-react";
import { useEffect } from "react";
import { backToShot, boot, endSelection, getUi, goBack, selectVersion, setUi, toggleApproved, undoPoint, uploadShotFile } from "./actions";
import { AssetsPage } from "./components/AssetsPage";
import { CreatePage } from "./components/CreatePage";
import { Home } from "./components/Home";
import { ImagePage } from "./components/ImagePage";
import { PrevizPage } from "./components/PrevizPage";
import { ProjectPage, ProjectsPage } from "./components/ProjectPages";
import { SetupPanel } from "./components/SetupPanel";
import { Onboarding } from "./components/Onboarding";
import { Sidebar } from "./components/Sidebar";
import { ShotView } from "./components/ShotView";
import { SaveToDialog } from "./components/SaveTo";
import { Toasts } from "./components/Toasts";
import { TopBar } from "./components/TopBar";
import { isOriginal } from "./lib/format";
import { openFilePicker, registerFilePicker } from "./lib/filePicker";
import { activePlayer } from "./lib/player";
import { getState, rememberedChrome, setState, useApp, type ViewMode } from "./store";

let booted = false;

/** A page opened from a shot (a cast sheet from Previz to final, Assets): one click back to that shot, as it was. */
function BackToShot() {
  const route = useApp((s) => s.route);
  const name = useApp((s) => (s.backToShot ? s.shots[s.backToShot]?.name : undefined));
  if (!name || (route.name !== "project" && route.name !== "assets" && route.name !== "projects")) return null;
  return (
    <div className="flex h-12 shrink-0 items-center border-b border-line px-4 max-[767px]:hidden">
      <button
        type="button"
        onClick={() => backToShot()}
        title="Esc"
        className="inline-flex h-8 max-w-[40ch] items-center gap-1.5 rounded-full border border-line2 bg-raised px-3 text-[13px] text-ink transition-colors hover:border-ivory/60 hover:text-ivory"
      >
        <ArrowLeft size={14} strokeWidth={1.75} />
        <span className="truncate">Back to {name}</span>
      </button>
    </div>
  );
}

function isTyping(t: EventTarget | null) {
  const el = t as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

/** Space play/pause · ←/→ frame · Shift+←/→ 1 s · ↑/↓ takes · B/A/R view · ⇧A approve · W wipe · Esc back · ⌘O open file.
 * N (a note at the playhead) lives in components/Notes.tsx (useNoteKey), next to the input it opens. */
function useKeyboard() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        openFilePicker();
        return;
      }
      if (e.key === "Escape") {
        if (document.fullscreenElement) {
          e.preventDefault();
          void document.exitFullscreen();
          return;
        }
        if (isTyping(e.target)) {
          (e.target as HTMLElement).blur();
          return;
        }
        if (goBack()) e.preventDefault();
        return;
      }
      if (isTyping(e.target) || e.altKey || e.defaultPrevented) return;
      const s = getState();
      if (s.setup.open || s.route.name !== "shot") return;
      const shotId = s.route.id;

      if (mod && e.key.toLowerCase() === "z" && s.selection?.shotId === shotId) {
        e.preventDefault();
        undoPoint();
        return;
      }
      if (mod) return;

      const player = activePlayer.current;
      const target = e.target as HTMLElement | null;
      const onControl = !!target?.closest?.('[role="slider"], [role="switch"], [role="radio"], [role="menuitem"]');

      switch (e.key) {
        case " ":
          if (target?.closest?.('[role="switch"]')) return;
          e.preventDefault();
          player?.toggle();
          return;
        case "ArrowLeft":
        case "ArrowRight": {
          if (onControl) return;
          e.preventDefault();
          const dir = e.key === "ArrowLeft" ? -1 : 1;
          if (e.shiftKey) player?.stepSeconds(dir);
          else player?.step(dir);
          return;
        }
        case "ArrowUp":
        case "ArrowDown": {
          // ↑/↓ switch takes (in strip order, skipping hidden ones unless they're shown).
          if (onControl) return;
          const shot = s.shots[shotId];
          if (!shot) return;
          const ui = getUi(shotId);
          const list = shot.versions.filter((v) => !v.hidden || ui.showHidden || v.id === ui.versionId);
          const i = Math.max(0, list.findIndex((v) => v.id === ui.versionId));
          const next = list[Math.max(0, Math.min(list.length - 1, i + (e.key === "ArrowUp" ? -1 : 1)))];
          e.preventDefault();
          if (next && next.id !== ui.versionId) selectVersion(shotId, next.id);
          return;
        }
      }

      const k = e.key.toLowerCase();
      if (k === "a" && e.shiftKey) {
        // ⇧A toggles Approved on the take on screen (A alone is the After view).
        e.preventDefault();
        toggleApproved(shotId, getUi(shotId).versionId);
        return;
      }
      if (k === "b" || k === "a" || k === "r" || k === "w") {
        const shot = s.shots[shotId];
        const ui = getUi(shotId);
        const version = shot?.versions.find((v) => v.id === ui.versionId);
        const selecting = ui.rail.kind === "form" && (ui.rail.task === "remove" || ui.rail.task === "recast");
        if (!shot || !version || isOriginal(version) || selecting) return;
        e.preventDefault();
        if (k === "w") {
          setUi(shotId, { wipe: !ui.wipe, view: "after" });
          return;
        }
        const view: ViewMode = k === "b" ? "before" : k === "a" ? "after" : "receipt";
        if (view === "receipt" && !version.receipt_url) return;
        setUi(shotId, { view: ui.view === view && view !== "after" ? "after" : view });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/** Keep the browser from navigating away when a file is dropped outside a drop zone. */
function useDropGuard() {
  useEffect(() => {
    const stop = (e: DragEvent) => {
      if (Array.from(e.dataTransfer?.types ?? []).includes("Files")) e.preventDefault();
    };
    window.addEventListener("dragover", stop);
    window.addEventListener("drop", stop);
    return () => {
      window.removeEventListener("dragover", stop);
      window.removeEventListener("drop", stop);
    };
  }, []);
}

export function App() {
  const route = useApp((s) => s.route);
  const chrome = useApp((s) => s.chrome);
  useKeyboard();
  useDropGuard();

  useEffect(() => {
    if (booted) return;
    booted = true;
    void boot();
  }, []);

  // A phone always gets the side panel's drawer and menu button (the top nav has no room for six links); the
  // remembered pick comes back on a wider window.
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const apply = () => setState({ chrome: mq.matches ? "side" : rememberedChrome() });
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  // Leaving a shot ends any click-to-select.
  useEffect(() => {
    if (route.name !== "shot") endSelection();
  }, [route]);

  return (
    <div className="flex h-full bg-bg">
      {chrome === "side" ? <Sidebar /> : null}
      <div className="flex min-w-0 flex-1 flex-col">
      <TopBar />
      <main className="relative flex min-h-0 flex-1 flex-col">
        <BackToShot />
        <div className="relative min-h-0 flex-1">
        {route.name === "home" ? (
          <Home />
        ) : route.name === "projects" ? (
          <ProjectsPage />
        ) : route.name === "project" ? (
          <ProjectPage key={route.id} id={route.id} tab={route.tab} />
        ) : route.name === "create" ? (
          route.panel === "image" ? <ImagePage /> : <CreatePage panel={route.panel} />
        ) : route.name === "assets" ? (
          <AssetsPage />
        ) : route.name === "previz" ? (
          <PrevizPage />
        ) : (
          <ShotView key={route.id} id={route.id} />
        )}
        </div>
      </main>
      </div>
      <SetupPanel />
      <Onboarding />
      <SaveToDialog />
      <Toasts />
      <input
        ref={registerFilePicker}
        type="file"
        accept="video/*,.mov,.mp4,.m4v,.mkv,.avi,.mxf,.webm"
        className="hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void uploadShotFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
