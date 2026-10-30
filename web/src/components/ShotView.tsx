import { useEffect, useMemo, useState } from "react";
import { ensureShot, navigate, setUi } from "../actions";
import { isOriginal, shotLength, takes } from "../lib/format";
import { activePlayer, Player, PlayerContext } from "../lib/player";
import { DEFAULT_UI, useApp } from "../store";
import { RightRail } from "./RightRail";
import { Filmstrip, Transport, VersionStrip } from "./Transport";
import { Notice, Skeleton } from "./ui";
import { Stage } from "./Viewer";

export function ShotView({ id }: { id: string }) {
  const shot = useApp((s) => s.shots[id]);
  const err = useApp((s) => s.shotErrors[id]);
  const mode = useApp((s) => s.mode);
  const ui = useApp((s) => s.ui[id]) ?? DEFAULT_UI;
  const [player] = useState(() => new Player());
  // Change weather picks what to keep by clicking only once its "Click it" is on (the picture stays a player till then).
  const picking = useApp((s) => !!s.selection && s.selection.shotId === id);

  useEffect(() => {
    activePlayer.current = player;
    return () => {
      if (activePlayer.current === player) activePlayer.current = null;
      player.stop();
    };
  }, [player]);

  useEffect(() => {
    if (!shot && mode) void ensureShot(id);
  }, [id, mode, shot]);

  const tk = useMemo(() => (shot ? takes(shot) : {}), [shot]);

  if (!shot) {
    if (err === "missing" || err === "failed") {
      return (
        <div className="flex h-full items-center justify-center px-6">
          <div className="w-full max-w-md">
            <Notice
              tone={err === "missing" ? "neutral" : "bad"}
              text={err === "missing" ? "This shot isn't here anymore." : "This shot didn't load."}
              action={
                err === "missing"
                  ? { label: "Back to Home", onClick: () => navigate({ name: "home" }) }
                  : { label: "Try again", onClick: () => void ensureShot(id, true) }
              }
            />
          </div>
        </div>
      );
    }
    return <ShotSkeleton />;
  }

  player.configure(shot.fps, shot.frames);
  const version = shot.versions.find((v) => v.id === ui.versionId) ?? shot.versions[0];
  const orig = shot.versions.find((v) => v.id === "orig") ?? shot.versions[0];
  const compare = !isOriginal(version);
  // The wipe's left side: any take, by default the one this take was made from.
  const inputId = version.input_version && shot.versions.some((v) => v.id === version.input_version) ? version.input_version : "orig";
  const before = shot.versions.find((v) => v.id === (ui.beforeId && ui.beforeId !== version.id ? ui.beforeId : inputId)) ?? orig;
  const beforeUrl = before.id === inputId && version.before_url ? version.before_url : before.video_url;
  const from = compare && inputId !== "orig" ? tk[inputId]?.num : compare && version.input_version === "orig" ? "01" : null;
  const selecting =
    ui.rail.kind === "form" && (ui.rail.task === "remove" || ui.rail.task === "recast" || (ui.rail.task === "environment" && picking))
      ? ui.rail.task
      : null;

  return (
    <PlayerContext.Provider value={player}>
      {/* Phones: the take panel stacks under the viewer and the page scrolls (side by side it covers the viewer). */}
      <div className="flex h-full min-h-0 max-[767px]:flex-col max-[767px]:overflow-y-auto">
        <section className="flex min-w-0 flex-1 flex-col bg-black px-6 pb-2 pt-3 max-[767px]:h-[440px] max-[767px]:flex-none max-[767px]:px-4" aria-label="Viewer">
          {/* The shot's name and menu (Rename, Move) are in the top bar's breadcrumb. */}
          <header className="flex h-9 shrink-0 items-center justify-end gap-4 max-[767px]:h-6">
            <span className="slate shrink-0 text-ink3 max-[767px]:hidden">
              {shot.width}×{shot.height} · {Math.round(shot.fps * 1000) / 1000} fps · {shotLength(shot)}
            </span>
          </header>

          <Stage
            shot={shot}
            version={version}
            beforeUrl={beforeUrl}
            beforePoster={before.thumb_url}
            beforeSlate={[`Take ${tk[before.id]?.num ?? "01"}`, tk[before.id]?.label ?? "Original", "Before"]}
            compare={compare}
            view={ui.view}
            wipe={ui.wipe}
            wipePos={ui.wipePos}
            onWipe={(wipePos) => setUi(shot.id, { wipePos })}
            selecting={selecting}
            cropping={ui.rail.kind === "prepare"}
            slate={[
              `Take ${tk[version.id]?.num ?? "01"}`,
              tk[version.id]?.label ?? "Original",
              from ? `From take ${from}` : null,
              version.status === "approved" ? "Approved" : null,
            ]}
          />
          <Transport
            shot={shot}
            compare={compare}
            hasReceipt={!!version.receipt_url}
            view={ui.view}
            wipe={ui.wipe}
            locked={!!selecting}
            onView={(view) => setUi(shot.id, { view })}
            onToggleWipe={() => setUi(shot.id, { wipe: !ui.wipe })}
            version={version}
            beforeId={before.id}
            onBefore={(beforeId) => setUi(shot.id, { beforeId })}
          />
          <Filmstrip shot={shot} url={version.video_url} notes={version.notes} />
          <div className="min-[768px]:hidden">
            <VersionStrip shot={shot} ui={ui} />
          </div>
        </section>
        <RightRail shot={shot} ui={ui} />
      </div>
    </PlayerContext.Provider>
  );
}

function ShotSkeleton() {
  return (
    <div className="flex h-full min-h-0" aria-busy="true" aria-label="Loading shot">
      <section className="flex min-w-0 flex-1 flex-col bg-black px-6 pb-2 pt-3">
        <div className="flex h-9 items-center gap-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-3 w-32" />
        </div>
        <div className="flex min-h-0 flex-1 items-center justify-center">
          <Skeleton className="aspect-video max-h-full w-full rounded-[2px]" />
        </div>
        <div className="flex h-12 items-center gap-2 pt-2">
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="ml-2 h-4 w-44" />
        </div>
        <Skeleton className="mt-1 h-12 w-full" />
        <div className="flex h-12 items-center gap-2 pt-2">
          <Skeleton className="h-9 w-32" />
        </div>
      </section>
      <aside className="flex w-[392px] shrink-0 flex-col gap-3 border-l border-line bg-panel p-6 max-[1180px]:w-[348px]">
        <Skeleton className="mb-4 h-9 w-64" />
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-[60px] w-full" />
        ))}
      </aside>
    </div>
  );
}
