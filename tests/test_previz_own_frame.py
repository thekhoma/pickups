"""Use my own first frame (or a start frame already made): no hidden references are read, no look frame is made.

Fresh-9 user test: with "Use my own first frame" the job still sent look_upload_ids and listed "Making the look frame".
"""

from fractions import Fraction
from pathlib import Path

import pytest

from pickups.media import Probe
from pickups.tasks import TASKS, previz


def test_the_look_frame_stage_is_listed_only_when_one_is_made():
    labels = lambda p: [s.label for s in TASKS["previz"].stages_of(p)]  # noqa: E731
    assert "Making the look frame" in labels({"look_upload_ids": ["u1"]})
    assert "Making the look frame" not in labels({"look_frame_upload_id": "mine", "look_upload_ids": ["u1"]})
    assert "Making the look frame" not in labels({"look_frame_ref": {"kind": "image", "id": "img"}})
    assert "Making the look frame" in labels({"look_frame_upload_id": "mine", "keyframes": True})  # middle and last keys


def test_an_own_first_frame_reads_no_references_and_makes_no_look_frame(tmp_path, monkeypatch):
    seen = {"stages": [], "uploads": [], "refs": []}

    def prepare(src, out, start, frames, size, pre="", cover=False, pad=None, hold=0):
        out.write_bytes(b"work")
        return {"fps": 24.0, "frames": frames, "start_frame": start, "size": list(size)}

    monkeypatch.setattr(previz.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), 97, False, None))
    monkeypatch.setattr(previz.media, "prepare_work_clip", prepare)
    monkeypatch.setattr(previz.bench, "machine", lambda conn: {"vram_gb": 24})
    monkeypatch.setattr(previz, "look_frame", lambda *a, **k: pytest.fail("no look frame is made"))
    monkeypatch.setattr(previz.graphs, "layout_render", lambda layout, look, prompt, *a, **k: {"look": look, "prompt": prompt})

    class Store:
        def shot_cast(self, shot_id):
            return [], lambda c: None

        def shot_meta(self, shot_id):
            return {}

        def shot_dir(self, shot_id):
            return tmp_path

        def upload_path(self, uid):
            seen["uploads"].append(uid)
            return tmp_path / f"{uid}.png"

        def ref_path(self, project_id, ref):
            seen["refs"].append(ref)
            return tmp_path / "ref.png"

    class Ctx:
        params = {"prompt": "He walks.", "look_frame_upload_id": "mine", "look_upload_ids": ["sheet"],
                  "look_refs": [{"kind": "cast", "id": "c1"}], "look_prompt": "The grey figure is Max.", "pin_start": True}
        input_path = Path("in.mp4")
        vdir = tmp_path
        conn = None
        project_id = "prj"
        shot_id = "sht"
        store = Store()
        client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

        def stage(self, label):
            seen["stages"].append(label)

        def run(self, graph):
            seen["graph"] = graph
            raise RuntimeError("stop after the graph")

    with pytest.raises(RuntimeError, match="stop after the graph"):
        previz.run(Ctx())
    assert seen["uploads"] == ["mine"], seen  # only the first frame itself
    assert seen["refs"] == [], seen
    assert "Making the look frame" not in seen["stages"]
    assert seen["graph"]["look"].endswith("mine.png")
