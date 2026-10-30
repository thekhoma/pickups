"""Pre-publish audit: scan the repo for things that must never go public.

Looks for API keys and tokens, private IPs and Tailscale addresses, absolute user/volume paths, email addresses,
local usernames, and brand, client, people or shot names you list in PICKUPS_AUDIT_TERMS (comma-separated) or in
~/.pickups/audit_terms.txt (one per line, # for comments). Both stay out of the repo: listing them here would publish
them.
Skips build output, dependencies and binary files. Exit code 1 when anything is found.

Usage: uv run python scripts/audit_repo.py            (from the repo root)
       PICKUPS_AUDIT_TERMS="clientname,brand" uv run python scripts/audit_repo.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMS_FILE = Path.home() / ".pickups" / "audit_terms.txt"
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm", ".woff", ".woff2", ".ttf", ".otf", ".ico",
               ".safetensors", ".zip", ".exr", ".lock"}
CHECKS = [
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("api key", re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|ghp_[A-Za-z0-9]{30,})")),
    ("secret assignment", re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]")),
    ("tailscale / private ip", re.compile(r"\b(100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b")),
    ("absolute local path", re.compile(r"(/Users/[A-Za-z0-9._-]+|/Volumes/[A-Za-z0-9 ._-]+|[A-Z]:\\\\Users\\\\)")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]
ALLOWED_EMAIL = re.compile(r"(@example\.(com|org)|noreply@|@users\.noreply\.github\.com)")


def listed_terms() -> list[str]:
    """Deny-listed names from the environment and the local terms file."""
    terms = [t.strip() for t in os.environ.get("PICKUPS_AUDIT_TERMS", "").split(",") if t.strip()]
    if TERMS_FILE.exists():
        for line in TERMS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and line not in terms:
                terms.append(line)
    return terms


def main() -> int:
    terms = listed_terms()
    user = os.environ.get("USER", "")
    checks = list(CHECKS)
    if user and len(user) > 2:
        checks.append(("local username", re.compile(re.escape(user), re.IGNORECASE)))
    for t in terms:
        checks.append((f"listed term '{t}'", re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)))
    found = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIX or path.stat().st_size > 2_000_000:
            continue
        if path.name == "audit_repo.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for name, rx in checks:
                m = rx.search(line)
                if not m:
                    continue
                if name == "email address" and ALLOWED_EMAIL.search(m.group(0)):
                    continue
                found += 1
                print(f"{path.relative_to(ROOT)}:{n}: {name}: {line.strip()[:140]}")
    print(f"\n{found} finding(s)" if found else "Clean: nothing that shouldn't be public.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
