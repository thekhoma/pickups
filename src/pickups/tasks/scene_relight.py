"""Relight any scene: new light on the whole shot (a time of day, a sun direction), from Lightricks' LTX-2.3 Relight IC-LoRA.

The LoRA reads the light direction from a small grey ball in the top-right corner of the clip and paints it out of the
result. Pickups bakes one of five tested balls into a 1280x704 copy of the shot (the LoRA's size), renders up to 121
frames, and delivers that render as it comes: R&D's tested workflow exactly (graphs.scene_relight), seed 42, no
compositing afterwards (an earlier detail-keep blend onto the plate brought back the old light's contrast and a halo).

Settings are the ones tested on an RTX 4090 (Oct 22, 20 runs). It peaks around 25.5 GB of the card, so nothing else
may be loaded: the GPU's memory is freed first. Not a weather tool: it can't add rain, fog or snow. Not person Relight
(Replace a person, More options), which relights only the new person.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from pickups import graphs, images, media
from pickups.app.jobs import Stage
from pickups.tasks import common

STAGES = [
    Stage("Preparing the shot", 15),
    Stage("Relighting the scene", 320),
    Stage("Writing your files", 30),
]

BALLS = Path(__file__).resolve().parent.parent / "assets" / "relight"
PROMPT = "relight the video to match the light-direction ball. {}"

# The tested looks: (label, ball, the light in the LoRA's words). Ball = where the light comes from, and how high.
LOOKS: dict[str, tuple[str, str, str]] = {
    "sunset": ("Sunset backlight", "behind_low", "strong backlight with rim light from behind, low sun"),
    "golden": ("Golden side light", "left_low", "warm golden low side sun from the left, low sun"),
    "cool": ("Cool low sun", "back_right_low", "cool soft daylight from the back-right, low sun"),
    "midday": ("Midday sun", "front_right_high", "hard high-angle sunlight from the front-right, high sun"),
    "afternoon": ("Afternoon light", "front_mid", "soft warm afternoon light from the front, mid sun"),
}
# Where the light comes from, for a look in the user's own words.
DIRECTIONS = {"behind_low": "from behind, low sun", "left_low": "from the left, low sun", "back_right_low": "from the back-right, low sun",
              "front_right_high": "from the front-right, high sun", "front_mid": "from the front, mid sun"}

MAX_FRAMES = 121  # the longest tested (5 s at 24 fps); 25.5 GB peak on a 24 GB card
COND_FPS = 24.0   # as tested, whatever the shot's own rate
BALL_PX, BALL_MARGIN = 143, 22


def look_of(p: dict) -> tuple[str, str, str]:
    """(label, ball, light words) for the job's params: a tested look, or the user's words with a direction."""
    key = str(p.get("look") or "sunset")
    if key != "custom":
        if key not in LOOKS:
            raise RuntimeError("Pick one of the looks, or describe your own.")
        return LOOKS[key]
    words = " ".join(str(p.get("custom") or "").split()).rstrip(".")
    if not words:
        raise RuntimeError("Describe the light you want, for example: warm evening light.")
    ball = str(p.get("direction") or "front_mid")
    if ball not in DIRECTIONS:
        ball = "front_mid"
    return words[:1].upper() + words[1:], ball, f"{words} {DIRECTIONS[ball]}"


def bake_ball(clip: Path, ball: str, out: Path) -> None:
    """The ball in the top-right corner, as in the tests (143 px, 22 px in)."""
    png = BALLS / f"ball_{ball}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-i", str(png), "-filter_complex",
                    f"[1:v]scale={BALL_PX}:{BALL_PX}:flags=lanczos[b];[0:v][b]overlay=W-w-{BALL_MARGIN}:{BALL_MARGIN}",
                    "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)


def run(ctx) -> dict:
    p = ctx.params
    label, ball, light = look_of(p)
    src = media.probe(ctx.input_path)
    if src.height > src.width:
        raise RuntimeError("Relight any scene works on landscape shots for now.")
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)

    ctx.stage("Preparing the shot")
    start = max(0, int(p.get("start", 0) or 0))
    n = media.ltx_frames_down(min(MAX_FRAMES, src.frames - start))
    if n < 9:
        raise RuntimeError("This shot is too short to relight.")
    fps = float(src.fps)
    small = run_dir / "relight_in.mp4"
    media.prepare_work_clip(ctx.input_path, small, start, n, graphs.RELIGHT_SIZE, cover=True)
    baked = run_dir / "relight_in_ball.mp4"
    bake_ball(small, ball, baked)
    ctx.stage("Relighting the scene")
    ctx.conn.free_memory()  # it needs the whole card (about 25.5 GB at 121 frames)
    graph = graphs.scene_relight(ctx.client.assets.from_file(baked), PROMPT.format(light), n, COND_FPS)
    graph = images.localize(graph, ctx.conn.model_files())
    job = ctx.run(graph)
    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(raw / o.name)
    frames = sorted(raw.glob("*.png"))
    if not frames:
        raise RuntimeError("The relight came back empty. Try again.")
    ctx.conn.free_memory()

    ctx.stage("Writing your files")
    out = run_dir / "out"
    out.mkdir(exist_ok=True)
    for i, f in enumerate(frames[:n]):  # the render's frames, in order, as the master's sequence
        f.rename(out / f"r_{i + 1:05d}.png")
    downloads = common.finish_outputs(ctx.vdir, out / "r_%05d.png", fps, audio_from=small)
    from pickups.app.store import make_viewer
    make_viewer(small, ctx.vdir / "before.mp4")
    Image.open(out / "r_00001.png").save(ctx.vdir / "first_frame.png")
    note = "The relight as LTX-2.3 renders it, at 1280×704 (the LoRA's size); chain Finish for a 4K master."
    return {"label": f"Relight · {label}", "task": "relight", "receipt": {"original": 0.0, "borrowed": 0.0, "ai": 100.0},
            "downloads": downloads,
            "details": {**common.sequence_details(out, "r_*.png", fps), "look": p.get("look") or "sunset", "light": light,
                        "models": ["LTX-2.3 + Relight IC-LoRA"], "prompt": PROMPT.format(light), "note": note}}
