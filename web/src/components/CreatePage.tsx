// Create: the picture wall, and the prompt bar over it. Home is the examples; this is where the settings live.
import { useEffect, useMemo, useState } from "react";
import { ensureProject, loadImages, openExample, openShot, setCreateMode, startPreset, targetProject } from "../actions";
import { presetFromExample } from "../lib/recipes";
import { useApp } from "../store";
import type { Example, ProjectImage } from "../types";
import { CreateComposer } from "./CreateBar";

const isStill = (url: string) => !/\.(mp4|webm|mov|m4v|mkv)(\?|#|$)/i.test(url);
const isPortrait = (w: number, h: number) => h > w * 1.05;

const FALLBACK: Tile[] = [
  "/mock/image_loch_1.jpg",
  "/mock/cast_winter_dusk.jpg",
  "/mock/still_driver_1.jpg",
  "/mock/cast_driver_sheet.jpg",
  "/mock/fun_knight_sheet.jpg",
  "/mock/fun_alien_sheet.jpg",
  "/mock/fun_astronaut_sheet.jpg",
].map((src) => ({ src, label: "Start an image", portrait: false, open: () => setCreateMode("image") }));

interface Tile {
  src: string;
  label: string;
  portrait: boolean;
  open: () => void;
}

/** Examples that show the same person or vehicle. The wall shows one of each, so nothing repeats on a screen. */
const SUBJECT: Record<string, string> = {
  "pv-k03-hood-push": "helmet-driver",
  "pv-k05-driver": "helmet-driver",
  "pv-s08-hood": "helmet-driver",
  "pv-s02-brief": "helmet-driver",
  "pv-k06-launch": "grey-sedan",
  "pv-s18-wheel": "grey-sedan",
  "env-car-snow": "blue-suv",
  "env-savanna-winter": "blue-suv",
  "rl-suv-cool": "blue-suv",
  "rl-suv-midday": "blue-suv",
  "rl-walk-golden": "walking-couple",
  "rl-walk-midday": "walking-couple",
};
/** 12 stills: 4, 3 or 2 columns, always full rows, filling the screen. */
const WALL = 12;

function exampleTile(e: Example): Tile | null {
  if (!e.poster_url || !isStill(e.poster_url)) return null;
  return {
    src: e.poster_url,
    label: e.title,
    portrait: false,
    open: () => {
      if (e.can_try !== false) void openExample(e);
      else startPreset(presetFromExample(e).id);
    },
  };
}

function imageTile(im: ProjectImage): Tile | null {
  if (!im.thumb_url || !isStill(im.thumb_url)) return null;
  const prompt = im.prompt?.trim() ?? "";
  return {
    src: im.thumb_url,
    label: prompt || "Start an image",
    portrait: isPortrait(im.width, im.height),
    open: () => setCreateMode("image", prompt || undefined),
  };
}

export function CreatePage({ panel }: { panel: "home" | "edit" }) {
  const examples = useApp((s) => s.examples);
  const images = useApp((s) => s.images);
  const projects = useApp((s) => s.projects);
  const shots = useApp((s) => s.shots);
  const projectShots = useApp((s) => s.projectShots);

  useEffect(() => {
    if (panel === "edit") setCreateMode("clip");
  }, [panel]);

  useEffect(() => {
    for (const p of projects ?? []) {
      if (!p.archived && p.id !== "examples") void loadImages(p.id);
    }
    void loadImages(targetProject());
    void ensureProject(targetProject()); // its shots: the videos Create made
  }, [projects]);

  const tiles = useMemo(() => {
    const base: Tile[] = [];
    const seen = new Set<string>();
    // Videos Create made (text or image to video) lead the wall: each opens its shot, ready to edit.
    for (const ids of Object.values(projectShots)) {
      for (const id of ids ?? []) {
        const s = shots[id];
        if (s?.made && s.thumb_url) base.push({ src: s.thumb_url, label: s.name, portrait: isPortrait(s.width, s.height), open: () => openShot(s.id) });
      }
    }
    for (const e of examples ?? []) {
      const subject = SUBJECT[e.id] ?? e.id;
      if (seen.has(subject)) continue;
      const t = exampleTile(e);
      if (t) {
        seen.add(subject);
        base.push(t);
      }
    }
    for (const list of Object.values(images)) {
      for (const im of list ?? []) {
        const t = imageTile(im);
        if (t) base.push(t);
      }
    }
    // Each picture once (a clip made once was shown 12 times): the starters fill what's left, then repeats only if a
    // fresh library has fewer than 12 pictures in all.
    const out: Tile[] = [];
    const srcs = new Set<string>();
    for (const t of [...base, ...FALLBACK]) {
      if (srcs.has(t.src)) continue;
      srcs.add(t.src);
      out.push(t);
    }
    const unique = out.length;
    while (out.length < WALL) out.push(out[out.length % unique]);
    return out.slice(0, WALL);
  }, [examples, images, shots, projectShots]);

  return (
    <div className="relative h-full overflow-hidden bg-bg">
      <div className="absolute inset-0 grid auto-rows-fr grid-cols-4 gap-1.5 p-1.5 max-[900px]:grid-cols-3 max-[640px]:grid-cols-2">
        {tiles.map((tile, i) => (
          <WallTile key={`${tile.src}-${i}`} tile={tile} />
        ))}
      </div>
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_top,#0D0C0B_0%,rgba(13,12,11,0.55)_28%,rgba(13,12,11,0.15)_55%,rgba(13,12,11,0.35)_100%)]" />
      <CreateComposer placement="dock" />
    </div>
  );
}

/** A still on the wall. Tall pictures keep the head and upper body; wide ones fill the frame. */
function WallTile({ tile }: { tile: Tile }) {
  const [portrait, setPortrait] = useState(tile.portrait);
  return (
    <button
      type="button"
      onClick={tile.open}
      aria-label={tile.label}
      className="group/tile relative min-h-0 overflow-hidden bg-black text-left outline-none transition-[filter] duration-150 hover:brightness-110 focus-visible:brightness-110"
    >
      <span className="pointer-events-none absolute inset-x-0 top-0 z-[1] truncate bg-gradient-to-b from-black/70 to-transparent px-3 pb-6 pt-2 text-[13px] text-ivory opacity-0 transition-opacity duration-150 group-hover/tile:opacity-100 group-focus-visible/tile:opacity-100">
        {tile.label}
      </span>
      <img
        src={tile.src}
        alt=""
        draggable={false}
        onLoad={(e) => {
          const im = e.currentTarget;
          if (isPortrait(im.naturalWidth, im.naturalHeight)) setPortrait(true);
        }}
        className={portrait ? "h-full w-full object-cover object-top" : "h-full w-full object-cover"}
      />
    </button>
  );
}
