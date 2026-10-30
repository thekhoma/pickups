"""Render previz: a Blender/Unreal playblast + a look -> a finished shot (LTX-2.5 Layout-To-Render).

Step 1 (instant-ish): the "look frame" is made from the playblast's first frame and the user's look reference
(character sheet, mood still, graded frame) with Qwen-Image 2.1, or the user brings their own graded first frame.
Step 2: Layout-To-Render follows the playblast's camera and blocking and takes palette, light and characters from the
look frame (a reference outside the timeline, so it isn't forced in as frame 1). 1920x1088, up to 97 frames.

By default the start still is only that reference (temporal position -1): no frame-0 pin and no mid or last keyframes
(R&D's l2r_round4 s05 video held 0.98x to 1.01x of the previz size on every frame that way). `pin_start` pins it at frame
0 again. Keyframes are off and not supported in this release (agents: leave `params.keyframes` unset; it stays only as
the cut check's test hook). `keyframes` (off for every shot: keyframes_on) adds Z-Image stills from previz frames 48 and
96 by the start still's own recipe (method zbuf, from the page's depth passes; never the start still reused), or the older
look-frame edits when the shot has no depth pass, and a cut in that render remakes it start-only (cut_frames). The video
prompt is the short Layout-to-Render one (prompt_words.l2r_prompt).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import graphs, media, mentions, prompt_words, start_frame
from pickups.app import bench
from pickups.app.jobs import Stage
from pickups.tasks import common

STAGES = [
    Stage("Preparing the previz", 15),
    Stage("Making the look frame", 150),
    Stage("Rendering the shot", 170),
    Stage("Writing your files", 30),
]
W, H = 1920, 1088
ZBUF_STILL = "Z-Image Turbo + Fun ControlNet Union 2.1"  # models_catalog.ZBUF_STILL: the start still from the previz depth
# "Follow the playblast: loosely": depth through the Union IC-LoRA (multiples of 128, close to 16:9, 1.8 MP: fits 24 GB).
LOOSE_SIZE = (1792, 1024)
UNION_LOOSE = 0.6  # at 0.6 see-through dust stand-ins rendered as dust (Oct 16); 1.0 turned them into flying wreckage

# Qwen-Image 2.1 instructions, in Comfy's official format: with two or more images every image is referred to as
# <image1>, <image2>... and each one's role is stated. <image1> is always the playblast frame (the canvas: a Blender or
# Unreal render, never a model output, so the edit stays clean); <image2>.. are the user's references in the order they
# were added (character sheets, a location plate, a mood still). The user's look prompt names them ("the driver is the
# woman in image 2", or @Maya); mentions.tag_images turns plain "image 2" into the tag.
LOOK_INSTRUCTION = ("Convert {canvas}, a 3D previz frame, into a photoreal live-action film still{ref}. {look} "
                    "Keep {canvas_s} exact framing, camera angle, lens and the position, size and pose of every object and "
                    "person. Replace every mannequin and grey block with the real person or object it stands for.")

# For the middle and last keyframes the first look frame goes LAST, so the reference numbers in the user's look prompt
# still point at the same images.
KEY_INSTRUCTION = ("Convert <image1>, a 3D previz frame, into a photoreal live-action film still in exactly the look of "
                   "<image{first}>: the same place, the same people and objects with the same faces, colours and details, "
                   "the same light and grade. {look} Keep <image1>'s exact framing, camera angle and the position, size "
                   "and pose of every object and person.")


def _ref_clause(n_refs: int) -> str:
    if n_refs == 0:
        return ""
    if n_refs == 1:
        return ", taking the people, objects, palette and light from <image2>"
    return f", taking the people, objects, place, palette and light from <image2> to <image{n_refs + 1}>"


def look_frame(client_runner, store, playblast: Path, refs: list[Path] | Path | None, look: str, out: Path, seed: int = 42,
               at: int = 0, commercial_safe: bool = False) -> Path:
    """Make the look frame from playblast frame `at`. client_runner has .client and .run (a TaskContext or the quick runner).
    commercial_safe: Qwen-Image-Edit 2511 (Apache-2.0), which reads the playblast plus the first two references."""
    refs = [refs] if isinstance(refs, Path) else list(refs or [])
    if commercial_safe:
        refs = refs[:2]
    frame0 = out.with_name(f"playblast_f{at}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(playblast), "-vf",
                    f"select=eq(n\\,{at}),scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                    "-frames:v", "1", str(frame0)], check=True)
    canvas, canvas_s = ("<image1>", "<image1>'s") if refs else ("the image", "its")
    instruction = mentions.tag_images(LOOK_INSTRUCTION.format(canvas=canvas, canvas_s=canvas_s, ref=_ref_clause(len(refs)),
                                                              look=look.strip()), len(refs) + 1)
    assets = [client_runner.client.assets.from_file(r) for r in refs]
    g = graphs.image_edit(client_runner.client.assets.from_file(frame0), instruction, seed=seed, refs=assets, resolution=0,
                          prefix="pickups/look", commercial_safe=commercial_safe, size=graphs.edit_size(W, H))
    job = client_runner.run(g)
    job.get_outputs("PU_Plate")[0].to_file(out)
    return out


def look_in_window(look_at: int, start: int, frames: int) -> int:
    """`look_at` is a frame of the shot. The render is the window from `start`, so the pin is how far into that window,
    on LTX's 8-frame grid. A frame before the window pins to its first frame."""
    rel = max(0, int(look_at) - max(0, int(start)))
    return min((rel // 8) * 8, max(0, int(frames) - 1))


def playblast_length(available: int, cap: int) -> tuple[int, int]:
    """Frames the render gets, and how many of those are clones of the last frame.

    The count is 8k+1. A playblast that is short of that is padded, never trimmed by 8.
    A playblast longer than the GPU cap is refused.
    """
    available = max(0, int(available))
    cap = max(1, int(cap))
    if available > cap:
        raise ValueError(f"This GPU renders up to {cap} frames. Shorten the shot.")
    frames = media.ltx_frames(max(1, available))
    if frames > cap:
        raise ValueError(f"This GPU renders up to {cap} frames. Shorten the shot.")
    return frames, frames - available


ORBIT_DEG, CRANE_M = 25.0, 0.8  # motionPrompt.ts cameraWords: an orbit turns over 25 degrees, a crane rises or drops 0.8 m


def camera_move(spec: dict, framing: dict) -> str | None:
    """"orbit" or "crane" when the previz camera swings round the subject or rises or drops (motionPrompt.ts cameraWords,
    from frame 0's recorded move, else the spec's first and last keys); None for any other move."""
    m = framing.get("move") if isinstance(framing.get("move"), dict) else None
    keys = (spec.get("camera") or {}).get("keys") or []
    a, b = (m.get("start"), m.get("end")) if m else ((keys[0], keys[-1]) if len(keys) >= 2 else (None, None))
    try:
        (ap, al), (bp, bl) = [(list(map(float, k["pos"])), list(map(float, k["look"]))) for k in (a, b)]
    except (TypeError, KeyError, ValueError):
        return None
    turn = math.degrees(math.atan2(bp[2] - bl[2], bp[0] - bl[0]) - math.atan2(ap[2] - al[2], ap[0] - al[0]))
    turn = (turn + 540) % 360 - 180
    if abs(turn) > ORBIT_DEG:
        return "orbit"
    return "crane" if abs(bp[1] - ap[1]) > CRANE_M else None


def orbit_degrees(spec: dict, framing: dict) -> float:
    """How far the previz camera swings round its subject, seen from above (degrees, 0 when there is no move)."""
    m = framing.get("move") if isinstance(framing.get("move"), dict) else None
    keys = (spec.get("camera") or {}).get("keys") or []
    a, b = (m.get("start"), m.get("end")) if m else ((keys[0], keys[-1]) if len(keys) >= 2 else (None, None))
    try:
        (ap, al), (bp, bl) = [(list(map(float, k["pos"])), list(map(float, k["look"]))) for k in (a, b)]
    except (TypeError, KeyError, ValueError):
        return 0.0
    turn = math.degrees(math.atan2(bp[2] - bl[2], bp[0] - bl[0]) - math.atan2(ap[2] - al[2], ap[0] - al[0]))
    return abs((turn + 540) % 360 - 180)


# Run 3 (R&D R2-1, R2-6; the director's one rule): keyframes are off for every move. Run 2's keys cut (c09, s01) and shrank the
# s06 dancers; the start still alone was smooth in 1b. They may switch on by themselves only for an orbit over
# KEYS_ORBIT_DEG, and only once all three pieces are built and tested in motion: each key's plan from that frame's own
# framing, each key an image-to-image of the start still at KEY_STRENGTH, and the cut check (cut_frames, built). The page
# records frame 0's framing only, and the image-to-image key is not built, so KEYS_AUTO stays False. There is no keyframe
# control in the app. Run 4: keyframes are not supported for the freeze; params.keyframes stays only as the cut check's
# test hook (tests/test_run3.py), and an agent should leave it unset.
KEYS_AUTO = False
KEYS_ORBIT_DEG = 90.0
KEY_STRENGTH = 0.6
CUT_RATIO = 3.0  # a frame-to-frame difference over this many times the shot's median is a cut


def keyframes_on(p: dict, spec: dict, framing: dict) -> bool:
    """Mid and last keyframes (not supported in this release): params.keyframes when set (a test hook, not a feature),
    else only an orbit over KEYS_ORBIT_DEG once KEYS_AUTO is on (off: every shot is the start still alone, at position -1)."""
    if p.get("keyframes") is not None:
        return bool(p["keyframes"])
    return KEYS_AUTO and orbit_degrees(spec, framing) > KEYS_ORBIT_DEG


def cut_frames(frames: list[Path], ratio: float = CUT_RATIO) -> list[int]:
    """The frames a hard cut lands on: where the mean difference from the frame before is over `ratio` times the median
    difference of the shot (measured on 160 px greyscale copies). [] for a smooth shot."""
    if len(frames) < 3:
        return []
    prev, diffs = None, []
    for f in frames:
        with Image.open(f) as im:
            a = np.asarray(im.convert("L").resize((160, 90), Image.BILINEAR), dtype=np.float32)
        if prev is not None:
            diffs.append(float(np.abs(a - prev).mean()))
        prev = a
    med = float(np.median(diffs))
    return [i + 1 for i, d in enumerate(diffs) if d > ratio * max(med, 0.5)]


# The model that made a start frame on the inline lane, by its method (the start-frame endpoint's).
START_MODELS = {start_frame.ZBUF: ZBUF_STILL, start_frame.METHOD: graphs.edit_model(False), "commercial-safe": graphs.edit_model(True)}


def frame_models(edited: bool, safe: bool, start_method: str | None, zbuf_keys: bool) -> list[str]:
    """The still models a take used, for its receipt: the frame editor only when it ran here (a look frame, or look-made
    keyframes), the start frame's own model when the inline lane made it, Z-Image Turbo for depth keyframes."""
    models = [graphs.edit_model(safe)] if edited else []
    for m in (START_MODELS.get(start_method or ""), ZBUF_STILL if zbuf_keys else None):
        if m and m not in models:
            models.append(m)
    return models


def start_recipe(store, shot_id: str, upload_id: str | None) -> dict:
    """What the accepted start still was made from (the Look words, the picked cast, the seed), saved with it by the
    start-frame endpoint, so its keyframes are made the same way; {} when the still came from elsewhere."""
    if not upload_id:
        return {}
    try:
        return (json.loads((store.shot_dir(shot_id) / RECIPES).read_text()) or {}).get(upload_id) or {}
    except (OSError, ValueError, AttributeError):
        return {}


RECIPES = "looks/start_recipes.json"  # upload id -> {"user_look", "look_refs", "seed", "method"}


def has_first_frame(p: dict) -> bool:
    """The job brings its first frame (a project image or an upload), so no look frame is made from references."""
    return bool(p.get("look_frame_ref") or p.get("look_frame_upload_id"))


def stages_for(p: dict) -> list[Stage]:
    """With its own first frame and no extra keyframes the job never makes a look frame, so that stage isn't listed."""
    if has_first_frame(p) and p.get("keyframes") is not True:
        return [s for s in STAGES if s.label != "Making the look frame"]
    return STAGES


def run(ctx) -> dict:
    p = ctx.params
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)
    describe = (p.get("describe") or "").strip()
    # One prompt per model: `look_prompt` is the Qwen-Image 2.1 instruction for the look frame (who and what is in each
    # reference), `prompt` is the LTX-2.5 video prompt (the action, camera and light). `describe` is the older single field.
    look_prompt = (p.get("look_prompt") or describe).strip()
    video_prompt = (p.get("prompt") or describe).strip() or "A photoreal cinematic live-action shot."
    seed = int(p.get("seed", 42))

    try:
        meta = ctx.store.shot_meta(ctx.shot_id)
    except (OSError, ValueError):
        meta = {}
    framing = meta.get("previz_framing") if isinstance(meta.get("previz_framing"), dict) else {}
    # The Layout-to-Render card's prompt: frame 1 and the one action, then "Keep the same ... and camera as the driver."
    # The Look box (what the start still was made from) rides along, so the video line says the same time and wardrobe.
    video_prompt = prompt_words.l2r_prompt(video_prompt, framing.get("subjects") or [], look=look_prompt)

    ctx.stage("Preparing the previz")
    n_in = media.probe(ctx.input_path).frames
    start = int(p.get("start_frame", 0))
    cap = common.previz_frames(bench.machine(ctx.conn).get("vram_gb"))  # 4 s, or 7 s on a 96 GB card
    frames, pad = playblast_length(n_in - start, cap)
    work = run_dir / "work.mp4"
    info = media.prepare_work_clip(ctx.input_path, work, start, frames, (W, H), pre=p.get("pre", ""), cover=True, hold=pad)
    fps = info["fps"]
    # look_at: which playblast frame the look frame is painted over (0 = the first). Snapped to LTX's 8-frame latent
    # grid so the pinned frame lands exactly on a latent frame.
    # look_at is a frame of the shot. The render is the window from `start`, so the pin is how far into that window.
    look_at = look_in_window(int(p.get("look_at", 0) or 0), start, frames)

    # A first frame the user brought (or already made and accepted) is the look: references only make a look frame, so
    # they're neither read nor sent (see has_first_frame).
    own = has_first_frame(p)
    edited = False  # the frame editor ran in this job (the receipt lists it only then)
    ref_ids = [] if own else list(p.get("look_upload_ids") or ([p["look_upload_id"]] if p.get("look_upload_id") else []))
    refs = [ctx.store.upload_path(u) for u in ref_ids]
    # References picked from the project (cast entries, generated images): {"kind": "cast" | "image" | "upload", "id"}.
    refs += [] if own else ctx.store.live_refs(ctx.project_id, p.get("look_refs", []))
    # @-mentions of the project's cast: Qwen gets their images (numbered after any uploaded references) and
    # "Maya (the person in <image2>)"; LTX gets "Maya, a woman in her thirties with ..." because it reads text only.
    cast, cast_file = ctx.store.shot_cast(ctx.shot_id)
    if own and cast and "@" in video_prompt:
        video_prompt = mentions.for_text(video_prompt, cast)
    elif cast and "@" in look_prompt + video_prompt:
        look_src = (p.get("look_prompt") or p.get("prompt") or describe).strip()
        look_prompt, used = mentions.for_images(look_src, cast, first_image=2 + len(refs))
        refs += [cast_file(c) for c in used]
        video_prompt = mentions.for_text((p.get("prompt") or p.get("look_prompt") or describe).strip(), cast)
    if p.get("look_frame_ref"):
        look = ctx.store.ref_path(ctx.project_id, p["look_frame_ref"])
    elif p.get("look_frame_upload_id"):
        look = ctx.store.upload_path(p["look_frame_upload_id"])
    else:
        ctx.stage("Making the look frame")
        look = look_frame(ctx, ctx.store, work, refs, look_prompt, run_dir / "look.png", seed=seed, at=look_at,
                          commercial_safe=ctx.commercial_safe())
        edited = True

    keyframes = []
    if look_at:
        # The look frame was painted over a later playblast frame (the one where every character is in shot), so it
        # is pinned there: people who walk in after frame 1 keep their face and wardrobe instead of the mannequin's.
        keyframes.append((ctx.client.assets.from_file(look), look_at, float(p.get("pin_strength", 1.0))))
    elif p.get("pin_start", False):
        # Off by default (the card's recipe: the start still at position -1 only). On, the start still is also the shot's
        # real first frame.
        keyframes.append((ctx.client.assets.from_file(look), 0, float(p.get("pin_strength", 1.0))))
    pinned = list(keyframes)  # the start still's own pin; what follows are the mid and last keys
    keys_at =((frames // 2) // 8 * 8, frames - 1)
    depths = [ctx.store.shot_dir(ctx.shot_id) / start_frame.DEPTH_FILE.format(start + i) for i in keys_at]
    want_keys = keyframes_on(p, meta.get("previz_spec") or {}, framing)
    zbuf_keys = want_keys and frames >= 49 and all(d.is_file() for d in depths)
    patch = start_frame.zbuf_patch(ctx.conn) if zbuf_keys else None
    if zbuf_keys and patch:
        # The rescue keyframes: new Z-Image stills from previz frames 48 and 96 (the start still's recipe and words, the
        # depth of that frame), never the start still again (reusing it froze c07).
        ctx.stage("Making the look frame")
        # R&D #13: the start still's own recipe (its Look words, its picked cast, its seed), else the job's words.
        recipe = start_recipe(ctx.store, ctx.shot_id, p.get("look_frame_upload_id"))
        shot_cast = ctx.store.shot_cast(ctx.shot_id)[0]
        by_id = {c.get("id"): c for c in shot_cast}
        picked = [by_id.get(r.get("id")) for r in recipe.get("look_refs") or [] if isinstance(r, dict) and r.get("kind") == "cast"]
        try:
            # R2-1: frame 0's size and facing sentences are never forced on a later frame.
            plan = start_frame.plan(meta.get("previz_spec") or {}, framing, shot_cast, [], [c for c in picked if c],
                                    str(recipe.get("user_look", look_prompt)), method=start_frame.ZBUF, at_key=True)
        except start_frame.NoSubjects:
            plan = None
        key_seed = int(recipe.get("seed", seed))
        strength = start_frame.strengths(start_frame.hero_kind(framing))[0]
        for idx, depth in zip(keys_at, depths):
            if plan is None:
                break
            fr = run_dir / f"playblast_{idx}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", str(fr)], check=True)
            control = start_frame.zbuf_control(ctx, depth, fr, run_dir / f"key_{idx}_control.png")
            kf = run_dir / f"key_{idx}.png"
            ctx.run(start_frame.zbuf_graph(plan, ctx.client.assets.from_file(control), key_seed, strength, patch=patch,
                                           prefix="pickups/look_key")).get_outputs("PU_Plate")[0].to_file(kf)
            keyframes.append((ctx.client.assets.from_file(kf), idx, float(p.get("key_strength", KEY_STRENGTH))))
    elif want_keys and frames >= 49:
        # Look frames for the middle and the last playblast frame, made in the look of the first one, so small or
        # moving subjects keep their place and scale through the shot instead of drifting or doubling.
        ctx.stage("Making the look frame")
        safe = ctx.commercial_safe()
        edited = True
        ref_assets = [ctx.client.assets.from_file(r) for r in (refs[:1] if safe else refs)]  # 2511 reads three images
        for idx in ((frames // 2) // 8 * 8, frames - 1):
            fr = run_dir / f"playblast_{idx}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", str(fr)], check=True)
            imgs = ref_assets + [ctx.client.assets.from_file(look)]
            key_text = mentions.tag_images(KEY_INSTRUCTION.format(first=len(imgs) + 1, look=look_prompt), len(imgs) + 1)
            job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(fr), key_text, seed=seed, refs=imgs, resolution=0,
                                            prefix="pickups/look_key", commercial_safe=safe, size=graphs.edit_size(W, H)))
            kf = run_dir / f"key_{idx}.png"
            job.get_outputs("PU_Plate")[0].to_file(kf)
            keyframes.append((ctx.client.assets.from_file(kf), idx, float(p.get("key_strength", KEY_STRENGTH))))

    # The start still's size against the previz, on the take: the start-frame endpoint's measure (R2-3), the one saved
    # with the still when the app made it.
    size = None
    if own and not start and framing.get("subject"):
        saved = start_recipe(ctx.store, ctx.shot_id, p.get("look_frame_upload_id")).get("size_check")
        size = saved or start_frame.size_check_now(ctx, ctx.store.shot_dir(ctx.shot_id), framing, Path(look))

    ctx.stage("Rendering the shot")
    # layout_strength: the playblast guide's strength. Keep 1.0: at 0.75 (tried on a stunt shot full of see-through dust
    # spheres) LTX stopped rendering and played back the grey playblast itself. Dust and smoke stand-ins are better hidden
    # in the playblast and described in words.
    loose = str(p.get("follow", "exact")).lower() == "loose"
    lw, lh = LOOSE_SIZE
    if loose:
        # "Follow the playblast: loosely". Layout-To-Render copies the playblast's shapes literally (see-through dust
        # spheres come out as glass balls, a wall a foot from the lens stays a flat grey block). Here the shot's depth
        # (Depth Anything 3) steers LTX through the Union IC-LoRA at UNION_LOOSE: camera, blocking and big shapes stay,
        # the stand-ins become what the prompt and the look frame say.
        ctl = run_dir / "control.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work), "-vf",
                        f"scale={lw}:{lh}:force_original_aspect_ratio=increase,crop={lw}:{lh}", "-c:v", "libx264", "-crf", "4",
                        "-pix_fmt", "yuv420p", str(ctl)], check=True)

    def render(keys: list, out: Path) -> Path:
        if loose:
            at, st = keys[0][1:] if keys else (0, 1.0)
            job = ctx.run(graphs.drive(ctx.client.assets.from_file(ctl), ctx.client.assets.from_file(look), video_prompt, fps,
                                       lw, lh, control_mode="depth", union_strength=float(p.get("union_strength", UNION_LOOSE)),
                                       seed=seed, prefix="pickups/previz", first_at=at, first_strength=st))
        else:
            job = ctx.run(graphs.layout_render(ctx.client.assets.from_file(work), ctx.client.assets.from_file(look),
                                               video_prompt, fps, W, H, seed=seed, prefix="pickups/previz", keyframes=keys,
                                               strength=float(p.get("layout_strength", 1.0))))
        out.mkdir(exist_ok=True)
        for o in job.get_outputs("PU_Out"):
            o.to_file(out / o.name)
        for i, f in enumerate(sorted(out.glob("*.png"))):
            f.rename(out / f"r_{i + 1:05d}.png")
            if loose:  # back to the previz size, so the masters match the exact mode's
                Image.open(out / f"r_{i + 1:05d}.png").convert("RGB").resize((W, round(W * lh / lw)), Image.LANCZOS).crop(
                    (0, (round(W * lh / lw) - H) // 2, W, (round(W * lh / lw) - H) // 2 + H)).save(out / f"r_{i + 1:05d}.png")
        return out

    out = render(keyframes, run_dir / "out")
    cuts: list[int] = []
    if len(keyframes) > len(pinned):
        # R2-6: a key that doesn't match the start still snaps the shot. On a cut the shot is made again from the start
        # still alone (the keyed render stays in work/out_keys).
        cuts = cut_frames(sorted(out.glob("r_*.png")))
        if cuts:
            shutil.move(str(out), str(run_dir / "out_keys"))
            keyframes = pinned
            out = render(keyframes, run_dir / "out")

    grain = p.get("grain", "fine")
    common.grain_pngs(out, "r_*.png", grain)

    ctx.stage("Writing your files")
    downloads = common.finish_outputs(ctx.vdir, out / "r_%05d.png", fps)
    from pickups.app.store import make_viewer
    make_viewer(work, ctx.vdir / "before.mp4")
    (ctx.vdir / "look.png").write_bytes(Path(look).read_bytes())
    for f in run_dir.glob("key_*.png") if not cuts else []:
        (ctx.vdir / f.name).write_bytes(f.read_bytes())
    return {"label": "Previz to final", "task": "previz", "receipt": {"original": 0.0, "borrowed": 0.0, "ai": 100.0},
            "downloads": downloads,
            "details": {**common.sequence_details(out, "r_*.png", fps),
                        "models": frame_models(edited, ctx.commercial_safe(), p.get("start_method") or start_recipe(
                            ctx.store, ctx.shot_id, p.get("look_frame_upload_id")).get("method"), bool(zbuf_keys and patch)) + (
                            ["LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3"] if loose else ["LTX-2.5 Layout-To-Render"]),
                        "follow": "loosely" if loose else "exactly",
                        "references": len(refs), "pinned_first_frame": bool(p.get("pin_start", False)), "look_at": look_at,
                        "start_position": -1, "keyframes": [k[1] for k in keyframes],
                        **({"cut_check": {"cuts": cuts, "remade_start_only": True}} if cuts else {}),
                        **({"size_check": size} if size else {}),
                        "grain": grain,
                        "look_prompt": look_prompt, "prompt": video_prompt,
                        "note": "Camera and blocking from your previz; look from the look frame."}}
