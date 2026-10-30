"""first_five.mjs helper: build the Previz to final render graph for a made shot without sending it to ComfyUI.

Runs tasks/previz.run with the queued job's params up to the graph, then stops. The LTX graph's length comes from the
layout clip its PU_Layout node loads, so the frame count printed is that clip's: what the render would get.
Usage: PICKUPS_HOME=<home> uv run python web/tests/previz/first_five_graph.py <shot_id> '<params json>'
"""

import json
import sys
import tempfile
from pathlib import Path

from pickups import media
from pickups.app.store import Store
from pickups.tasks import previz


class Stop(Exception):
    pass


def main() -> None:
    shot_id, params = sys.argv[1], json.loads(sys.argv[2])
    store = Store()
    meta = store.shot_meta(shot_id)
    seen: dict = {}

    class Ctx:
        def __init__(self, vdir: Path):
            self.params = params
            self.input_path = store.source_path(shot_id)
            self.vdir = vdir
            self.conn = None
            self.project_id = meta.get("project_id")
            self.shot_id = shot_id
            self.store = store
            self.client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

        def stage(self, label):
            seen.setdefault("stages", []).append(label)

        def commercial_safe(self):
            return False

        def run(self, graph):
            layout = graph["PU_Layout"]["inputs"]["file"]
            seen["graph_layout_frames"] = media.probe(Path(layout)).frames
            seen["graph_nodes"] = len(graph)
            seen["guides"] = sorted(k for k in graph if k.endswith("_layout") or k.endswith("_look"))
            raise Stop()

    previz.bench.machine = lambda conn: {"vram_gb": 24}
    with tempfile.TemporaryDirectory() as d:
        try:
            previz.run(Ctx(Path(d)))
        except Stop:
            pass
    seen["source_frames"] = media.probe(store.source_path(shot_id)).frames
    print(json.dumps(seen))


if __name__ == "__main__":
    main()
