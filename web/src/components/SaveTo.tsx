// "Save to": Make shot and New character ask where the new work goes, My shots or a project, or a new project.
// One dialog for the whole app (mounted in App); askSaveTo() opens it and resolves to a project id, or null on Cancel.
import { Plus } from "lucide-react";
import { useEffect, useState, useSyncExternalStore, type FormEvent } from "react";
import { getApi, isLocalHost, newProject, setCurrentProject, targetProject } from "../actions";
import { cx } from "../lib/format";
import { useApp } from "../store";
import { Button, Hint, MiddleTruncate, Modal } from "./ui";

// --- Save videos to: the folder finished takes are copied into, one folder per project (savecopy.py) -------------

type Folder = { path: string | null; suggested: string; shown?: string | null; suggested_shown?: string | null } | null;
let folder: Folder = null;
let folderLoading = false;
const folderListeners = new Set<() => void>();
const emitFolder = () => folderListeners.forEach((fn) => fn());

function loadFolder() {
  if (folderLoading) return;
  let api;
  try {
    api = getApi();
  } catch {
    return; // still booting: the row asks again when it is shown
  }
  folderLoading = true;
  api
    .saveFolder()
    .then((f) => {
      folder = f;
      emitFolder();
    })
    .catch(() => undefined)
    .finally(() => {
      folderLoading = false;
    });
}

export function useSaveFolder(): Folder {
  const f = useSyncExternalStore(
    (fn) => {
      folderListeners.add(fn);
      return () => folderListeners.delete(fn);
    },
    () => folder,
  );
  useEffect(() => {
    if (!folder) loadFolder();
  }, []);
  return f;
}

async function keepFolder(path: string): Promise<string | null> {
  try {
    const r = await getApi().setSaveFolder(path);
    folder = { ...folder, path: r.path, shown: r.shown ?? r.path, suggested: folder?.suggested ?? r.path };
    emitFolder();
    return null;
  } catch (e) {
    return e instanceof Error && e.message ? e.message : "That folder didn't work. Try another.";
  }
}

/** "Save videos to <folder> · Choose location": the computer's folder picker when Pickups runs here, else a typed path.
 * `first`: nothing chosen yet; the suggestion is shown, ready to keep. */
export function SaveFolderRow({ compact = false }: { compact?: boolean }) {
  const f = useSaveFolder();
  const [typing, setTyping] = useState(false);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const shown = (f?.path ? f.shown || f.path : f?.suggested_shown || f?.suggested) ?? "…";
  const choose = async () => {
    setErr(null);
    if (!isLocalHost()) {
      setText(shown);
      setTyping(true);
      return;
    }
    setBusy(true);
    try {
      const r = await getApi().chooseSaveFolder();
      if (r.path) setErr(await keepFolder(r.path));
      else if (!f?.path) {
        setText(shown);
        setTyping(true);
      }
    } catch {
      setText(shown);
      setTyping(true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className={cx("space-y-1.5", compact ? "" : "rounded-card border border-line p-3")} data-testid="save-folder">
      <div className={cx("flex items-center gap-2", compact && !typing && "hidden")}>
        <span className="slate shrink-0 text-ink3">Videos go to</span>
        <Hint text="Every finished take is copied here, in a folder per project. Pickups keeps its own working copy too." />
      </div>
      {typing ? (
        <form
          className="flex items-center gap-1.5"
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            const bad = await keepFolder(text);
            setBusy(false);
            setErr(bad);
            if (!bad) setTyping(false);
          }}
        >
          <input autoFocus value={text} onChange={(e) => setText(e.target.value)} aria-label="Folder for your videos" className="field mono h-8 min-w-0 flex-1 px-2 text-[12px]" />
          <Button type="submit" variant="primary" size="sm" loading={busy} disabled={!text.trim()}>
            Keep
          </Button>
        </form>
      ) : (
        <div className="flex items-center gap-2">
          {compact ? (
            <span className="flex shrink-0 items-center gap-1">
              <span className="slate text-ink3">To</span>
              <Hint text="Every finished take is copied here, in a folder per project. Pickups keeps its own working copy too." />
            </span>
          ) : null}
          <MiddleTruncate text={shown} className={cx("mono min-w-0 flex-1 text-[12px]", f?.path ? "text-ink" : "text-ink3")} />
          <Button variant={compact ? "ghost" : "secondary"} size="sm" className="shrink-0" loading={busy} onClick={() => void choose()}>
            Choose location
          </Button>
        </div>
      )}
      {err ? <p className="text-[12px] text-ink2">{err}</p> : null}
    </div>
  );
}

type Ask = { title: string; action: string; resolve: (id: string | null) => void };

let ask: Ask | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((fn) => fn());

/** Where to save: resolves to the picked project's id (it becomes the current project), or null when canceled. */
export function askSaveTo(title: string, action: string): Promise<string | null> {
  ask?.resolve(null);
  return new Promise((resolve) => {
    ask = { title, action, resolve };
    emit();
  });
}

function close(id: string | null) {
  const a = ask;
  ask = null;
  emit();
  if (id) setCurrentProject(id, false);
  a?.resolve(id);
}

export function SaveToDialog() {
  const a = useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    () => ask,
  );
  const projects = useApp((s) => s.projects);
  const [pick, setPick] = useState("unsorted");
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const list = (projects ?? []).filter((p) => !p.archived && p.id !== "examples" && p.id !== "qc" && p.id !== "unsorted");

  useEffect(() => {
    if (!a) return;
    setPick(targetProject());
    setNaming(false);
    setName("");
    setBusy(false);
  }, [a]);

  const fold = useSaveFolder();
  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    // First time: the suggested folder is kept unless another was chosen (the row shows it, with Choose location).
    if (fold && !fold.path && fold.suggested) await keepFolder(fold.suggested);
    if (!naming) return close(pick);
    const n = name.trim();
    if (!n) return;
    setBusy(true);
    const p = await newProject(n, false);
    setBusy(false);
    if (p) close(p.id);
  };

  const row = (id: string, label: string) => (
    <label
      key={id}
      className={cx(
        "flex h-11 cursor-pointer items-center gap-3 rounded-card border px-3 text-[15px] transition-colors",
        !naming && pick === id ? "border-ivory/70 bg-raised text-ivory" : "border-line text-ink hover:border-line2",
      )}
    >
      <input
        type="radio"
        name="save-to"
        className="accent-[#EDE6D6]"
        checked={!naming && pick === id}
        onChange={() => {
          setPick(id);
          setNaming(false);
        }}
      />
      <span className="truncate">{label}</span>
    </label>
  );

  return (
    <Modal open={!!a} onClose={() => close(null)} label={a?.title ?? "Save to"} width={460}>
      <form onSubmit={save} className="space-y-5 p-6">
        <h2 className="text-[20px] font-medium text-ivory">{a?.title}</h2>
        <div className="max-h-[40vh] space-y-2 overflow-y-auto" role="radiogroup" aria-label="Project">
          {row("unsorted", "My shots")}
          {list.map((p) => row(p.id, p.name))}
          {naming ? (
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="New project name"
              aria-label="New project name"
              className="field h-11 w-full px-3 text-[15px]"
            />
          ) : (
            <button
              type="button"
              onClick={() => setNaming(true)}
              className="flex h-11 w-full items-center gap-2 rounded-card border border-dashed border-line2 px-3 text-left text-[15px] text-ink hover:border-ivory/60 hover:text-ivory"
            >
              <Plus size={15} strokeWidth={1.5} aria-hidden="true" />
              New project
            </button>
          )}
        </div>
        <SaveFolderRow />
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="md" onClick={() => close(null)}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" size="md" data-autofocus loading={busy} disabled={busy || (naming && !name.trim())}>
            {a?.action ?? "Save"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
