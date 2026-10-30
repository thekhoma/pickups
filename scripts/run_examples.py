"""Run the example library through a running Pickups server, exactly as a user would (shots, jobs, takes).

Reads a JSON list of examples:
  {"id", "sample_id" | "path", "project"?, "task", "params", "then"?: [{"task", "params"}], "title", "line", "slate",
   "hero"?, "public"?}
For each one: opens the sample (or a file path) as a shot in the named project (default: Examples), uploads any
"@upload:<path>" values found in params (look references, first frames), submits the task, waits, then runs the
"then" chain on the result (e.g. remove -> finish). Progress is written to ~/.pickups/examples/runs.json, which
make_examples.py turns into the home screen's gallery. Re-running skips examples that already finished.

Usage: uv run python scripts/run_examples.py examples.json [--only id1,id2] [--server http://127.0.0.1:7788]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HOME = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups")))
RUNS = HOME / "examples" / "runs.json"


def call(server: str, method: str, url: str, body: dict | None = None, form: list[str] | None = None) -> dict:
    args = ["curl", "-s", "-f", "-X", method, server + url]
    if body is not None:
        args += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    for f in form or []:
        args += ["-F", f]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{method} {url} failed: {r.stdout[:300]} {r.stderr[:300]}")
    return json.loads(r.stdout)


def resolve_uploads(server: str, value):
    """Replace "@upload:<path>" strings (anywhere in params) with upload ids."""
    if isinstance(value, str) and value.startswith("@upload:"):
        return call(server, "POST", "/api/uploads", form=[f"file=@{value[8:]}"])["id"]
    if isinstance(value, list):
        return [resolve_uploads(server, v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_uploads(server, v) for k, v in value.items()}
    return value


def project_id(server: str, name: str) -> str:
    for p in call(server, "GET", "/api/projects"):
        if p["name"] == name:
            return p["id"]
    return call(server, "POST", "/api/projects", {"name": name})["id"]


def wait(server: str, job_id: str) -> dict:
    last = None
    while True:
        job = next((j for j in call(server, "GET", "/api/jobs") if j["id"] == job_id), None)
        if job is None:
            raise RuntimeError(f"job {job_id} vanished (server restarted?)")
        if job.get("stage") != last:
            print(time.strftime("%H:%M:%S"), "  ", job.get("stage"), flush=True)
            last = job.get("stage")
        if job["status"] in ("done", "failed", "canceled"):
            return job
        time.sleep(5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--server", default="http://127.0.0.1:7788")
    ap.add_argument("--wait-samples", action="store_true", help="wait for make_samples.py to finish each sample first")
    a = ap.parse_args()
    examples = json.loads(a.spec.read_text())
    only = {x for x in a.only.split(",") if x}
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        if only and ex["id"] not in only:
            continue
        if ex.get("shot_name"):
            continue  # a gallery entry for a take that already exists (make_examples.py picks it up)
        if runs.get(ex["id"], {}).get("status") == "done":
            print("skip (done)", ex["id"])
            continue
        if a.wait_samples and ex.get("sample_id"):
            poster = HOME / "samples" / f"{ex['sample_id']}.jpg"  # written last, after the clip
            while not poster.exists():
                time.sleep(10)
        print("===", ex["id"], time.strftime("%H:%M:%S"), flush=True)
        try:
            pid = project_id(a.server, ex.get("project", "Examples"))
            if ex.get("sample_id"):
                shot = call(a.server, "POST", f"/api/samples/{ex['sample_id']}/open")
                if shot.get("project_id") != pid:
                    shot = call(a.server, "PATCH", f"/api/shots/{shot['id']}", {"project_id": pid})
            else:
                shot = call(a.server, "POST", "/api/shots/from_path", {"path": ex["path"], "project_id": pid})
            if ex.get("name"):
                shot = call(a.server, "PATCH", f"/api/shots/{shot['id']}", {"name": ex["name"]})
            rec = {"shot_id": shot["id"], "takes": [], "status": "running"}
            runs[ex["id"]] = rec
            version = None
            for step in [{"task": ex["task"], "params": ex.get("params", {})}] + ex.get("then", []):
                params = resolve_uploads(a.server, step.get("params", {}))
                job = call(a.server, "POST", "/api/jobs", {"shot_id": shot["id"], "task": step["task"], "params": params,
                                                           "version_id": version})
                job = wait(a.server, job["id"])
                if job["status"] != "done":
                    raise RuntimeError(f"{step['task']} {job['status']}: {job.get('error')}")
                version = job["result_version_id"]
                rec["takes"].append({"task": step["task"], "take_id": version})
            rec["status"] = "done"
            print("ok", ex["id"], rec["takes"], flush=True)
        except Exception as e:  # noqa: BLE001 - keep going; the failure is recorded for review
            runs.setdefault(ex["id"], {})["status"] = "failed"
            runs[ex["id"]]["error"] = str(e)[:500]
            print("FAILED", ex["id"], e, flush=True)
        RUNS.write_text(json.dumps(runs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
