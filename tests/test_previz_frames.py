"""A previz playblast reaches the render as 8k+1 frames. A short one is padded, never trimmed."""

from pathlib import Path

import pytest

from pickups.media import Probe
from pickups.tasks import previz
from fractions import Fraction


def _drive(tmp_path, monkeypatch, n_in):
    seen = {}

    def prepare(src, out, start, frames, size, pre="", cover=False, pad=None, hold=0):
        seen["prepare"] = {"frames": frames, "pad": hold, "start": start}
        out.write_bytes(b"work")
        return {"fps": 24.0, "frames": frames, "start_frame": start, "size": list(size)}

    def layout(layout, look, prompt, fps, *args, **kwargs):
        seen["layout"] = {"layout": layout, "frames": seen["prepare"]["frames"], "fps": fps}
        return {"layout": layout, "frames": seen["prepare"]["frames"]}

    monkeypatch.setattr(previz.media, "probe", lambda path, count=True: Probe(1920, 1088, Fraction(24), n_in, False, None))
    monkeypatch.setattr(previz.media, "prepare_work_clip", prepare)
    monkeypatch.setattr(previz.bench, "machine", lambda conn: {"vram_gb": 24})
    monkeypatch.setattr(previz.graphs, "layout_render", layout)

    class Store:
        def shot_cast(self, shot_id):
            return [], lambda c: None

        def shot_meta(self, shot_id):
            return {}

        def shot_dir(self, shot_id):
            return tmp_path

        def ref_path(self, project_id, ref):
            return tmp_path / "look.png"

    class Ctx:
        params = {"describe": "a man walking", "seed": 1, "start_frame": 0, "pin_start": False,
                  "look_frame_ref": {"kind": "image", "id": "img"}, "follow": "exact"}
        input_path = Path("in.mp4")
        vdir = tmp_path
        conn = None
        project_id = "prj"
        shot_id = "sht"
        store = Store()
        client = type("C", (), {"assets": type("A", (), {"from_file": staticmethod(lambda p: str(p))})()})()

        def stage(self, _label):
            return None

        def run(self, graph):
            seen["graph_frames"] = graph["frames"]
            raise RuntimeError("stop after the graph")

    with pytest.raises(RuntimeError, match="stop after the graph"):
        previz.run(Ctx())
    return seen


def test_a_97_frame_playblast_reaches_the_render_as_97(tmp_path, monkeypatch):
    seen = _drive(tmp_path, monkeypatch, 97)
    assert seen["prepare"] == {"frames": 97, "pad": 0, "start": 0}
    assert seen["graph_frames"] == 97
    assert seen["layout"]["frames"] == 97


def test_a_96_frame_playblast_is_padded_to_97(tmp_path, monkeypatch):
    seen = _drive(tmp_path, monkeypatch, 96)
    assert seen["prepare"]["frames"] == 97
    assert seen["prepare"]["pad"] == 1
    assert seen["graph_frames"] == 97
    assert seen["graph_frames"] != 89


def test_a_playblast_longer_than_the_cap_is_refused():
    with pytest.raises(ValueError, match="Shorten the shot"):
        previz.playblast_length(120, 97)


def test_new_spec_fields_round_trip_and_an_unknown_field_is_refused(request):
    import json
    import subprocess
    from pickups.app.previz_maker import parse_spec
    spec = {
        "version": 1, "template": "described", "duration_s": 97 / 24, "fps": 24, "size": [1920, 1088],
        "lens": {"focal_mm": 35, "film_gauge_mm": 36},
        "environment": {"ground": "plane", "sky": "gradient", "fog": [40, 400], "props": ["park"]},
        "subjects": [{"id": "person", "kind": "humanoid", "height_m": 1.75, "pos": [0, 0, 0], "anim": "walk", "speed": 0.5}],
        "camera": {
            "interp": "catmull_rom", "ease": "linear", "taper": 0,
            "keys": [{"t": 0, "pos": [0, 1.6, 4], "look": [0, 1.2, 0], "focal_mm": 35, "roll": 16},
                     {"t": 1, "pos": [0, 1.6, 3], "look": [0, 1.2, 0], "focal_mm": 35, "roll": 16}],
            "segments": [{"move": "push_in", "from_t": 0, "to_t": 1, "params": {"speed": 0.6}, "tracking": 1}],
            "handheld": {"seed": 1, "pos_m": 0, "rot_deg": 0, "hz": 0},
        },
        "beats": [{"actor": "person", "type": "look_at_camera", "from_t": 0.7, "to_t": 1, "params": {}}],
        "seed": 1,
    }
    parsed = parse_spec(spec)
    assert parsed.subjects[0].speed == 0.5
    assert parsed.camera.ease == "linear" and parsed.camera.taper == 0
    assert parsed.camera.segments[0].move == "push_in" and parsed.camera.segments[0].tracking == 1
    assert parsed.camera.handheld is not None and parsed.camera.handheld.hz == 0
    assert parsed.beats[0].type == "look_at_camera"
    with pytest.raises(Exception):
        parse_spec({**spec, "nope": 1})
    web = request.getfixturevalue("web_modules")  # the page's own schema (zod) must agree; skips without node_modules
    script = (
        "import { PrevizSpecSchema } from './src/previz/spec.ts';"
        "const spec = JSON.parse(process.argv[1]);"
        "const ok = PrevizSpecSchema.safeParse(spec);"
        "if (!ok.success) { console.error(ok.error.message); process.exit(1); }"
        "const bad = PrevizSpecSchema.safeParse({...spec, nope: 1});"
        "if (bad.success) process.exit(2);"
        "if (Math.round(spec.duration_s * 24) !== 97) process.exit(3);"
    )
    r = subprocess.run(["node", "--experimental-strip-types", "--eval", script, json.dumps(spec)],
                       capture_output=True, text=True, cwd=web)
    assert r.returncode == 0, r.stderr
