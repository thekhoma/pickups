// A generation's progress area, the same everywhere one runs (Create's images and videos, the start frame): what it is
// doing, how long it has been going, how long this kind usually takes here (measured only), and Cancel. Cancelled, it
// says so with Try again. lib/runtime.ts has the maths.
import { useState } from "react";
import { cancelJob } from "../actions";
import { elapsedClock, usually, useElapsed } from "../lib/runtime";
import type { Job } from "../types";
import { Button, ProgressLine, Slate } from "./ui";

/** One line under a running job: slate (label · elapsed), the progress line, "Usually about N s" and Cancel. */
export function RunStrip({ job, label, className, onCancel }: { job: Job; label: string; className?: string; onCancel?: () => void }) {
  const elapsed = useElapsed(job);
  const [stopping, setStopping] = useState(false);
  const running = job.status === "running";
  const said = usually(job.usual_s);
  return (
    <div className={className} data-testid="run-strip" aria-live="polite">
      <div className="flex items-center gap-3">
        <Slate live={running}>{running ? [label, elapsedClock(elapsed)] : ["Queued"]}</Slate>
        <span className="ml-auto flex shrink-0 items-center gap-3">
          {said ? <span className="text-[12px] text-ink3" data-testid="usually">{said}</span> : null}
          <Button
            variant="ghost"
            size="sm"
            loading={stopping || job.stage === "Canceling"}
            onClick={async () => {
              setStopping(true);
              if (onCancel) onCancel();
              else await cancelJob(job.id);
              setStopping(false);
            }}
          >
            Cancel
          </Button>
        </span>
      </div>
      <ProgressLine value={job.progress} idle={!running || job.lane === "inline"} className="mt-2" />
    </div>
  );
}

/** A cancelled run: one line, and Try again. */
export function Cancelled({ onRetry, className }: { onRetry?: () => void; className?: string }) {
  return (
    <div className={className} role="status" data-testid="cancelled">
      <div className="flex items-center gap-3 rounded-card border border-line px-4 py-3">
        <span className="slate text-ivory/70">Cancelled</span>
        {onRetry ? (
          <Button variant="secondary" size="sm" className="ml-auto" onClick={onRetry}>
            Try again
          </Button>
        ) : null}
      </div>
    </div>
  );
}
