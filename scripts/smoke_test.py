"""Smoke-test a running Pickups server: every read-only API route the UI uses answers sensibly. Starts no GPU jobs.

Usage: uv run python scripts/smoke_test.py [--server http://127.0.0.1:7788]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def get(server: str, path: str, method: str = "GET", body: dict | None = None):
    req = urllib.request.Request(server + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - local server
        data = r.read()
        ctype = r.headers.get("Content-Type", "")
        return r.status, (json.loads(data) if "json" in ctype else data.decode(errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:7788")
    a = ap.parse_args()
    s = a.server
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append((True, name, detail))
        except Exception as e:  # noqa: BLE001
            results.append((False, name, f"{type(e).__name__}: {e}"))

    check("home page", lambda: "Pickups" in get(s, "/")[1] or "page has no title")
    check("status", lambda: get(s, "/api/status")[1]["verdict"])
    check("settings", lambda: get(s, "/api/settings")[1]["director"]["backend"])
    projects = []
    check("projects", lambda: projects.extend(get(s, "/api/projects")[1]) or f"{len(projects)} projects")
    for p in projects[:6]:
        check(f"project {p['name']}", lambda p=p: (lambda d: f"{len(d['shots'])} shots, {len(d['cast'])} cast")(get(s, f"/api/projects/{p['id']}")[1]))
        check(f"images {p['name']}", lambda p=p: f"{len(get(s, f'/api/projects/{p['id']}/images')[1])} images")
        check(f"deliveries {p['name']}", lambda p=p: f"{len(get(s, f'/api/projects/{p['id']}/deliveries')[1])} approved")
    check("shots", lambda: f"{len(get(s, '/api/shots')[1])} shots")
    check("jobs", lambda: f"{len(get(s, '/api/jobs')[1])} jobs")
    check("examples", lambda: f"{len(get(s, '/api/examples')[1])} examples")
    check("samples", lambda: f"{len(get(s, '/api/samples')[1])} samples")
    check("image models", lambda: (lambda d: f"{len(d['models'])} models, default {d['default']}")(get(s, "/api/image/models")[1]))
    check("director detect", lambda: f"{len(get(s, '/api/prompt/detect')[1]['found'])} local LLM servers")
    if projects:
        pid = projects[0]["id"]
        check("mentions preview", lambda: get(s, "/api/mentions/preview", "POST", {"project_id": pid, "text": "hello"})[1]["text"])

        def export():
            try:
                return f"{len(get(s, f'/api/projects/{pid}/export/xml')[1])} bytes"
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    return "nothing approved (409, expected)"
                raise
        check("timeline export", export)
    width = max(len(n) for _, n, _ in results)
    for ok, name, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(not ok for ok, _, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
