// A shot's menu: Rename, Move to another project (or a new one). The shot's name in the top bar's breadcrumb opens it,
// and so does the "…" on a shot card (project pages).
import { ArrowLeft, ChevronDown, ChevronRight, Ellipsis, Plus } from "lucide-react";
import { useCallback, useRef, useState, type FormEvent } from "react";
import { moveShot, moveShotToNewProject, openPrepare, openShot, renameShot, trashShot } from "../actions";
import { cx } from "../lib/format";
import { useApp } from "../store";
import type { Shot } from "../types";
import { Button, Modal, useDismiss } from "./ui";

const ITEM =
  "flex w-full items-center justify-between gap-4 px-4 py-2.5 text-left text-[14px] text-ink transition-colors duration-150 hover:bg-line hover:text-ivory focus-visible:bg-line focus-visible:outline-none disabled:text-ink4 disabled:hover:bg-transparent";

/** The menu's two pages: its actions, and "Move to" (projects, then "New project…"). */
function ShotMenuBody({ shot, close, onRename, withOpen }: { shot: Shot; close: () => void; onRename: () => void; withOpen?: boolean }) {
  const projects = useApp((s) => s.projects);
  const [moving, setMoving] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const targets = (projects ?? []).filter((p) => p.id !== shot.project_id && p.id !== "examples" && !p.archived);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    const ok = await moveShotToNewProject(shot.id, name.trim());
    setBusy(false);
    if (ok) close();
  };

  if (!moving) {
    return (
      <>
        {withOpen ? (
          <button
            type="button"
            role="menuitem"
            className={ITEM}
            onClick={() => {
              close();
              openShot(shot.id);
            }}
          >
            Open
          </button>
        ) : null}
        <button
          type="button"
          role="menuitem"
          className={ITEM}
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
          className={ITEM}
          onClick={() => {
            close();
            openPrepare(shot.id, { fromMenu: true });
          }}
        >
          Trim & crop…
        </button>
        <button type="button" role="menuitem" className={ITEM} onClick={() => setMoving(true)}>
          <span>Move to project…</span>
          <ChevronRight size={14} strokeWidth={1.5} className="text-ink3" />
        </button>
        {shot.project_id !== "examples" ? (
          <div className="mt-1 border-t border-line pt-1">
            <button
              type="button"
              role="menuitem"
              className={ITEM}
              onClick={() => {
                close();
                void trashShot(shot.id);
              }}
            >
              Move to Trash
            </button>
          </div>
        ) : null}
      </>
    );
  }
  return (
    <>
      <button type="button" role="menuitem" className={cx(ITEM, "text-ink3")} onClick={() => setMoving(false)}>
        <span className="inline-flex items-center gap-2">
          <ArrowLeft size={13} strokeWidth={1.5} /> Move to
        </span>
      </button>
      <div className="max-h-[44vh] overflow-y-auto border-t border-line py-1">
        {targets.map((p) => (
          <button
            key={p.id}
            type="button"
            role="menuitem"
            aria-label={`Move to ${p.name}`}
            className={ITEM}
            onClick={() => {
              close();
              void moveShot(shot.id, p.id);
            }}
          >
            <span className="truncate">{p.name}</span>
            <span className="slate shrink-0 text-ink3">
              {p.shots} {p.shots === 1 ? "shot" : "shots"}
            </span>
          </button>
        ))}
      </div>
      <div className="border-t border-line py-1">
        {naming ? (
          <form onSubmit={create} className="flex items-center gap-2 px-3 py-1.5">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  setNaming(false);
                }
              }}
              placeholder="Project name"
              aria-label="New project name"
              className="field h-8 min-w-0 flex-1 text-[13px]"
            />
            <Button type="submit" variant="primary" size="sm" className="shrink-0" loading={busy} disabled={!name.trim() || busy}>
              Move
            </Button>
          </form>
        ) : (
          <button type="button" role="menuitem" className={ITEM} onClick={() => setNaming(true)}>
            <span className="inline-flex items-center gap-2">
              <Plus size={14} strokeWidth={1.5} /> New project…
            </span>
          </button>
        )}
      </div>
    </>
  );
}

/** Rename a shot in a small dialog (from a card's menu). */
function RenameDialog({ shot, onClose }: { shot: Shot; onClose: () => void }) {
  const [name, setName] = useState(shot.name);
  return (
    <Modal open onClose={onClose} label="Rename shot" width={440}>
      <form
        className="p-6"
        onSubmit={(e) => {
          e.preventDefault();
          void renameShot(shot.id, name);
          onClose();
        }}
      >
        <h2 className="display text-[36px] leading-[1.05]">Rename shot</h2>
        <input data-autofocus value={name} onChange={(e) => setName(e.target.value)} onFocus={(e) => e.target.select()} aria-label="Shot name" spellCheck={false} className="field mono mt-5" />
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" size="md" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" size="md" disabled={!name.trim()}>
            Save
          </Button>
        </div>
      </form>
    </Modal>
  );
}

/** "…" over a shot card: Open, Rename, Trim & crop, Move to project, Move to Trash. */
export function ShotMoreMenu({ shot, className }: { shot: Shot; className?: string }) {
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, ref);
  return (
    <div ref={ref} className={cx("absolute right-2 top-2 z-[3]", className)}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${shot.name}: more`}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className={cx(
          "inline-flex h-8 w-8 items-center justify-center rounded-full bg-black/70 text-ivory/85 transition-opacity duration-150 hover:bg-black/90 hover:text-ivory focus-visible:opacity-100",
          open ? "opacity-100" : "opacity-0 group-hover/card:opacity-100 max-[767px]:opacity-100",
        )}
      >
        <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
      </button>
      {open ? (
        <div role="menu" aria-label="Shot" className="absolute right-0 top-full z-40 mt-1.5 w-[260px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          <ShotMenuBody shot={shot} close={close} onRename={() => setRenaming(true)} withOpen />
        </div>
      ) : null}
      {renaming ? <RenameDialog shot={shot} onClose={() => setRenaming(false)} /> : null}
    </div>
  );
}

/** The shot's name in the breadcrumb: a menu (Rename inline, Move to). */
export function ShotTitle({ shot }: { shot: Shot }) {
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(shot.name);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, ref);

  if (renaming) {
    const save = (e?: FormEvent) => {
      e?.preventDefault();
      setRenaming(false);
      void renameShot(shot.id, name);
    };
    return (
      <form onSubmit={save} className="min-w-0">
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => save()}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              e.preventDefault();
              e.stopPropagation();
              setName(shot.name);
              setRenaming(false);
            }
          }}
          aria-label="Shot name"
          spellCheck={false}
          className="field mono h-8 w-[min(360px,50vw)] px-2 text-[13px]"
        />
      </form>
    );
  }

  return (
    <div ref={ref} className="relative min-w-0">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${shot.name}: shot menu`}
        onClick={() => (open ? close() : setOpen(true))}
        className="group flex h-8 min-w-0 items-center gap-1 rounded-[4px] px-1 text-ink transition-colors hover:text-ivory"
      >
        <h1 className="truncate font-mono text-[13px]">{shot.name}</h1>
        <ChevronDown size={12} strokeWidth={1.5} className={cx("shrink-0 text-ink3 transition-transform duration-150 group-hover:text-ivory", open && "rotate-180")} />
      </button>
      {open ? (
        <div role="menu" aria-label="Shot" className="absolute left-0 top-full z-40 mt-2 w-[min(280px,80vw)] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
          <ShotMenuBody
            shot={shot}
            close={close}
            onRename={() => {
              setName(shot.name);
              setRenaming(true);
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
