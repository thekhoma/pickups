import { ArrowLeft, X } from "lucide-react";
import type { ReactNode } from "react";
import { stopRecipe } from "../actions";
import { cx } from "../lib/format";
import { recipeById } from "../lib/recipes";
import { useApp } from "../store";
import { IconButton, Slate } from "./ui";

// Shared rail chrome ----------------------------------------------------------------------------------------------

export function RailHeader({ title, sub, onBack, badge }: { title: string; sub?: ReactNode; onBack?: () => void; badge?: ReactNode }) {
  return (
    <div className="flex min-h-[64px] shrink-0 items-center gap-1.5 border-b border-line px-3 py-2">
      {onBack ? <IconButton icon={ArrowLeft} label="Back" hint="Esc" onClick={onBack} /> : null}
      <div className="min-w-0 flex-1 pl-1">
        <div className="flex items-center gap-2">
          <h2 className="truncate text-[16px] font-medium text-ivory">{title}</h2>
          {badge}
        </div>
        {sub ? (
          typeof sub === "string" ? (
            <div className="slate truncate text-ink3">{sub}</div>
          ) : (
            <div className="slate mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-ink3">{sub}</div>
          )
        ) : null}
      </div>
    </div>
  );
}

export function RailBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("min-h-0 flex-1 overflow-y-auto px-6 py-6", className)}>{children}</div>;
}

export function RailFooter({ children }: { children: ReactNode }) {
  // Phones scroll the whole page (the picture above the rail): the primary stays pinned to the screen's bottom edge.
  return (
    <div className="shrink-0 space-y-3 border-t border-line px-6 py-5 max-[767px]:sticky max-[767px]:bottom-0 max-[767px]:z-20 max-[767px]:bg-panel max-[767px]:pb-[max(1.25rem,env(safe-area-inset-bottom))]">
      {children}
    </div>
  );
}

/** "RECIPE · CLEAN PLATE, DELIVERED · STEP 1 OF 2" while a recipe runs on this shot. */
export function RecipeStrip({ shotId }: { shotId: string }) {
  const run = useApp((s) => s.recipeRuns[shotId]);
  const recipe = recipeById(run?.recipeId);
  if (!run || !recipe) return null;
  return (
    <div className="flex items-center gap-2 border-b border-line px-6 py-2.5">
      <Slate className="min-w-0 flex-1">
        <span className="truncate">{`Recipe · ${recipe.title}`}</span>
        <span className="ml-auto shrink-0 pl-3 text-ivory">{`Step ${run.step + 1}/${recipe.tasks.length}`}</span>
      </Slate>
      <button type="button" onClick={() => stopRecipe(shotId)} aria-label="Stop the recipe" className="rounded-full p-1 text-ink3 transition-colors hover:text-ivory">
        <X size={13} />
      </button>
    </div>
  );
}
