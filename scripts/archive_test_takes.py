"""Move leftover test shots out of a person's projects into the archived "Test runs" project. Nothing is deleted:
the shots keep every file and can be moved back from Projects (Show archived).

    PICKUPS_HOME=/path/to/pickups-home uv run python scripts/archive_test_takes.py            # dry run: lists them
    PICKUPS_HOME=/path/to/pickups-home uv run python scripts/archive_test_takes.py --apply    # moves them

A test shot is a previz shot that repeats another one in the same project (same name and the same scene), or any previz
shot named with --name. Of each repeated set, the newest stays where it is unless --name matches it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from pickups.app.store import EXAMPLES, QC, Store


def test_shots(store: Store, names: set[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for shot in store.shots():
        if shot.get("previz_spec") and shot["project_id"] not in (QC, EXAMPLES):
            groups[(shot["project_id"], shot["name"], json.dumps(shot["previz_spec"], sort_keys=True))].append(shot)
    out = []
    for (_, name, _), shots in groups.items():
        shots.sort(key=lambda s: s.get("created_at") or 0, reverse=True)
        out.extend(shots if name in names else shots[1:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", default=[], help="also archive every previz shot with this name")
    ap.add_argument("--apply", action="store_true", help="move them (without it, only list them)")
    a = ap.parse_args()
    store = Store()
    found = test_shots(store, set(a.name))
    for shot in found:
        print(f"{'moved' if a.apply else 'would move'} {shot['id']}  {shot['name']!r}  from {shot['project_id']}")
        if a.apply:
            store.move_shot(shot["id"], store.qc_project())
    print(f"{len(found)} test shot(s){'' if a.apply else ' (dry run: add --apply to move them)'}")


if __name__ == "__main__":
    main()
