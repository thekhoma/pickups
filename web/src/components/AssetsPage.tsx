// Assets: clips and pictures you brought in, what Pickups made, and the Trash. Drop a file anywhere on this page.
import { Ellipsis } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deleteAsset, emptyTrash, ensureProject, importImageFile, isLocalHost, loadImages, loadTrash, navigate, openShot, restoreTrash, revealLabel, revealMedia, revealTake, setCurrentProject, toast, trashShot, trashTake, uploadShotFile } from "../actions";
import { cx, isOriginal, size, takes } from "../lib/format";
import { deleteLine } from "../lib/deleted";
import { useApp } from "../store";
import type { Project, TrashEntry } from "../types";
import { MediaCard, MediaCardSkeleton, Thumb } from "./MediaCard";
import { Button, ConfirmDelete, MENU_DELETE, MENU_ITEM, Notice, Page, PageHeader, Segmented, Slate, useDismiss } from "./ui";

type Section = "imported" | "made" | "trash";
type Kind = "all" | "video" | "image";

interface Item {
  key: string;
  section: Section;
  kind: "video" | "image";
  title: string;
  thumb?: string;
  /** A clip or take: plays when its card is hovered. */
  video?: string;
  meta: string;
  approved: boolean;
  open: () => void;
  /** Reveal in Finder: the file on this computer. */
  reveal?: () => void;
  /** Delete: what the confirm says, and what it does (everything goes to the Trash). */
  remove: { line: string | null; run: () => Promise<unknown> };
}

const VIDEO = /\.(mov|mp4|m4v|mkv|webm|avi|mxf)$/i;

function liveProjects(projects: Project[] | null): Project[] {
  return (projects ?? []).filter((p) => !p.archived && p.id !== "examples");
}

export function AssetsPage() {
  const projects = useApp((s) => s.projects);
  const shots = useApp((s) => s.shots);
  const projectShots = useApp((s) => s.projectShots);
  const images = useApp((s) => s.images);
  const cast = useApp((s) => s.cast);
  const trash = useApp((s) => s.trash);
  const [section, setSection] = useState<Section>("made");
  const [emptying, setEmptying] = useState(false);
  useEffect(() => {
    if (section === "trash") void loadTrash();
  }, [section]);
  const [kind, setKind] = useState<Kind>("all");
  const [approvedOnly, setApprovedOnly] = useState(false);

  useEffect(() => {
    for (const p of liveProjects(projects)) {
      void ensureProject(p.id);
      void loadImages(p.id);
    }
  }, [projects]);

  const picker = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const over = (e: DragEvent) => {
      if (Array.from(e.dataTransfer?.types ?? []).includes("Files")) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      const list = e.dataTransfer?.files;
      if (!list?.length) return;
      e.preventDefault();
      void takeFiles(list);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  const items = useMemo(() => {
    const out: Item[] = [];
    for (const p of liveProjects(projects)) {
      for (const id of projectShots[p.id] ?? []) {
        const shot = shots[id];
        if (!shot) continue;
        const tk = takes(shot);
        const orig = shot.versions.find((v) => isOriginal(v));
        out.push({
          key: `clip:${shot.id}`,
          section: shot.made || shot.previz_spec ? "made" : "imported", // a video Create made, or a previz, is Pickups' work
          kind: "video",
          title: shot.name,
          thumb: shot.thumb_url,
          video: shot.video_url,
          meta: p.name,
          approved: orig?.status === "approved",
          open: () => openShot(shot.id, { versionId: orig?.id }),
          reveal: () => void revealTake(shot.id, "orig"),
          remove: { line: deleteLine("shot", shot.versions.length - 1), run: () => trashShot(shot.id) },
        });
        for (const v of shot.versions) {
          if (isOriginal(v) || v.hidden) continue; // a hidden take is hidden here too, as in the shot
          const num = tk[v.id]?.num;
          out.push({
            key: `take:${shot.id}:${v.id}`,
            section: "made",
            kind: "video",
            title: num ? `${shot.name} · Take ${num}` : shot.name,
            thumb: v.thumb_url || shot.latest_thumb_url,
            video: v.video_url || undefined,
            meta: p.name,
            approved: v.status === "approved",
            open: () => openShot(shot.id, { versionId: v.id }),
            reveal: () => void revealTake(shot.id, v.id),
            remove: { line: null, run: () => trashTake(shot.id, v.id) },
          });
        }
      }
      for (const im of images[p.id] ?? []) {
        out.push({
          key: `img:${p.id}:${im.id}`,
          section: "made",
          kind: "image",
          title: im.prompt?.trim() || "Image",
          thumb: im.thumb_url,
          meta: p.name,
          approved: false,
          open: () => {
            setCurrentProject(p.id, true);
            navigate({ name: "create", panel: "image" });
          },
          reveal: im.url ? () => void revealMedia(im.url) : undefined,
          remove: { line: deleteLine("asset", im.uses ?? 0), run: () => deleteAsset(p.id, "image", im.id) },
        });
      }
      for (const c of cast[p.id] ?? []) {
        const made = !!c.sheet && c.sheet.made_from !== "upload";
        out.push({
          key: `cast:${p.id}:${c.id}`,
          section: made ? "made" : "imported",
          kind: "image",
          title: c.name,
          thumb: c.thumb_url || c.url,
          meta: p.name,
          approved: false,
          open: () => navigate({ name: "project", id: p.id, tab: "cast" }),
          reveal: c.url ? () => void revealMedia(c.url) : undefined,
          remove: { line: deleteLine("asset", c.uses ?? 0), run: () => deleteAsset(p.id, "cast", c.id) },
        });
      }
    }
    return out;
  }, [projects, shots, projectShots, images, cast]);

  const loading = projects === null;
  // Pictures are never approved (only takes are): the Approved filter is for video only.
  const approvable = kind !== "image";
  const onlyApproved = approvedOnly && approvable;
  const shown = items.filter((it) => it.section === section && (kind === "all" || it.kind === kind) && (!onlyApproved || it.approved));

  return (
    <Page>
        <PageHeader title="Assets" />
        {/* Clips and pictures, as a drop takes them: a clip opens as a shot, a picture joins the project. */}
        <input
          ref={picker}
          type="file"
          multiple
          hidden
          accept={ASSET_FILES}
          aria-label="Import clips or pictures"
          onChange={(e) => {
            if (e.target.files?.length) {
              setSection("imported");
              void takeFiles(e.target.files);
            }
            e.target.value = "";
          }}
        />

        <div className="flex flex-wrap items-center gap-3">
          <Segmented<Section>
            mono
            label="Section"
            value={section}
            onChange={setSection}
            options={[
              { value: "made", label: "Made in Pickups" },
              { value: "imported", label: "Imported" },
              { value: "trash", label: "Trash" },
            ]}
          />
          <Button variant="secondary" size="sm" onClick={() => picker.current?.click()}>
            Import
          </Button>
          {section === "trash" ? (
            trash?.length ? (
              <Button
                variant="secondary"
                size="sm"
                className="ml-auto"
                loading={emptying}
                onClick={async () => {
                  setEmptying(true);
                  await emptyTrash();
                  setEmptying(false);
                }}
              >
                Empty Trash · {size(trash.reduce((n, t) => n + t.bytes, 0))}
              </Button>
            ) : null
          ) : (
          <>
          <Segmented<Kind>
            mono
            label="Kind"
            value={kind}
            onChange={setKind}
            options={[
              { value: "all", label: "All" },
              { value: "video", label: "Video" },
              { value: "image", label: "Image" },
            ]}
          />
          {approvable ? (
            <Segmented<"all" | "approved">
              mono
              label="Approval"
              value={approvedOnly ? "approved" : "all"}
              onChange={(v) => setApprovedOnly(v === "approved")}
              options={[
                { value: "all", label: "Any" },
                { value: "approved", label: "Approved" },
              ]}
            />
          ) : null}
          </>
          )}
        </div>

        <div className="mt-8">
          {section === "trash" ? (
            trash === null ? (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-5 gap-y-10">
                <MediaCardSkeleton />
                <MediaCardSkeleton />
              </div>
            ) : trash.length ? (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-5 gap-y-10">
                {trash.map((t) => (
                  <TrashCard key={t.id} entry={t} />
                ))}
              </div>
            ) : (
              <Notice text="Trash is empty." />
            )
          ) : loading ? (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-5 gap-y-10">
              <MediaCardSkeleton />
              <MediaCardSkeleton />
              <MediaCardSkeleton />
            </div>
          ) : shown.length ? (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-x-5 gap-y-10">
              {shown.map((it) => (
                <AssetCard key={it.key} it={it} />
              ))}
            </div>
          ) : (
            <Notice
              text={
                onlyApproved
                  ? "Nothing approved in this section."
                  : section === "imported"
                    ? "Nothing imported yet. Drop a clip or a picture."
                    : "Nothing made in Pickups yet."
              }
              action={
                onlyApproved
                  ? { label: "Show all", onClick: () => setApprovedOnly(false) }
                  : section === "imported"
                    ? { label: "Choose files", onClick: () => picker.current?.click() }
                    : { label: "Go to Create", onClick: () => navigate({ name: "create", panel: "home" }) }
              }
            />
          )}
        </div>
    </Page>
  );
}

/** An asset: its card, and "…" (Open, Delete) on the picture's corner, shown on hover, on focus and on touch screens. */
function AssetCard({ it }: { it: Item }) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => {
    setOpen(false);
    setConfirming(false);
  }, []);
  useDismiss(open, close, ref);
  return (
    <div className="group/asset relative min-w-0">
      <MediaCard thumb={it.thumb} video={it.video} title={it.title} meta={it.meta} slate={it.approved ? ["Approved"] : undefined} onClick={it.open} />
      <div
        ref={ref}
        className={cx(
          "absolute right-2 top-2 z-10 transition-opacity duration-150 group-hover/asset:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100",
          open ? "opacity-100" : "opacity-0",
        )}
      >
        <button
          type="button"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={`${it.title}: more`}
          onClick={() => (open ? close() : setOpen(true))}
          className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-bg/80 text-ink transition-colors duration-150 hover:bg-raised hover:text-ivory"
        >
          <Ellipsis size={16} strokeWidth={1.75} aria-hidden="true" />
        </button>
        {open && confirming ? (
          <div className="absolute right-0 top-full mt-1.5 animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised">
            <ConfirmDelete title={`Delete ${it.title}?`} line={it.remove.line} confirm="Delete asset" onCancel={close} onConfirm={async () => (await it.remove.run(), close())} />
          </div>
        ) : open ? (
          <div role="menu" aria-label={it.title} className="absolute right-0 top-full mt-1.5 w-[200px] animate-rise-in overflow-hidden rounded-card border border-line2 bg-raised py-1">
            <button
              type="button"
              role="menuitem"
              className={MENU_ITEM}
              onClick={() => {
                close();
                it.open();
              }}
            >
              Open in Pickups
            </button>
            {it.reveal && isLocalHost() ? (
              <button
                type="button"
                role="menuitem"
                className={MENU_ITEM}
                onClick={() => {
                  close();
                  it.reveal!();
                }}
              >
                {revealLabel()}
              </button>
            ) : null}
            <div className="mt-1 border-t border-line pt-1">
              <button type="button" role="menuitem" className={MENU_DELETE} onClick={() => setConfirming(true)}>
                Delete
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

const TRASH_KIND: Record<TrashEntry["kind"], string> = { shot: "Shot", take: "Take", project: "Project", cast: "Cast", image: "Image" };

/** Something in the trash: its picture, what it is, name and size, and Restore. */
function TrashCard({ entry }: { entry: TrashEntry }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="flex min-w-0 flex-col">
      <div className="hairline-frame relative aspect-video w-full overflow-hidden rounded-card bg-black">
        <Thumb src={entry.thumb_url} className="absolute inset-0 h-full w-full opacity-70" />
        <Slate at="tl">{TRASH_KIND[entry.kind] ?? "Item"}</Slate>
        <Slate at="br">{size(entry.bytes)}</Slate>
      </div>
      <div className="mt-3 flex min-w-0 items-center justify-between gap-3">
        <span className="truncate font-mono text-[13px] text-ink">{entry.name}</span>
        <Button
          variant="secondary"
          size="sm"
          className="shrink-0"
          loading={busy}
          onClick={async () => {
            setBusy(true);
            await restoreTrash(entry.id);
            setBusy(false);
          }}
        >
          Restore
        </Button>
      </div>
    </div>
  );
}

/** What Import takes: the clips a shot opens from, and pictures. */
const ASSET_FILES = "video/*,.mov,.mp4,.m4v,.mkv,.avi,.mxf,.webm,image/*,.png,.jpg,.jpeg,.webp";

async function takeFiles(list: FileList): Promise<void> {
  const files = Array.from(list);
  const pictures = files.filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif)$/i.test(f.name));
  const clips = files.filter((f) => f.type.startsWith("video/") || VIDEO.test(f.name));
  for (const f of pictures) await importImageFile(f);
  // A clip opens as its shot, one at a time: say so rather than drop the rest without a word.
  if (clips.length > 1) toast(`Added ${clips[0].name}. Drop the other ${clips.length - 1 === 1 ? "clip" : `${clips.length - 1} clips`} one at a time.`, "info");
  if (clips[0]) await uploadShotFile(clips[0]);
}
