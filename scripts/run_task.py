"""Run one Pickups task outside the app (tests, benchmarks): same code path as the app's job queue.

Usage: uv run python scripts/run_task.py --shot SHOT_ID --task previz --params '{"describe": "..."}' [--version V]
"""

import argparse
import json
import time

from pickups.app.comfy import Connection
from pickups.app.jobs import JobManager
from pickups.app.store import Store
from pickups.tasks import TASKS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--version")
    a = ap.parse_args()
    jm = JobManager(Store(), Connection.from_settings(), TASKS)
    job = jm.submit(a.shot, a.task, json.loads(a.params), a.version)
    last = None
    while job["status"] in ("queued", "running"):
        if job["stage"] != last:
            print(time.strftime("%H:%M:%S"), job["stage"], flush=True)
            last = job["stage"]
        time.sleep(2)
    print(json.dumps({k: job.get(k) for k in ("status", "error", "result_version_id")}))
    return 0 if job["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
