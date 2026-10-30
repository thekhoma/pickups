"""Recast anyone: one Recast take that swaps several people (or cars) in a shot, each to their own character.

params.swaps is a list of {frame, points, text?, kind?, label?, name?} plus who each one becomes (cast_id,
ref_upload_id, who_ref, describe or reference_prompt, as a single Recast takes them). One swap is today's Recast
(recast_ltx.run, unchanged); two or more run here, on the same LTX-2.5 steps:

1. Each person is followed on their own: SAM 3.1 tracks them together, one object each (people who touch stay apart),
   and every object's own masks come back (common.track_each). A track that jumps to the other person is put back.
2. Window and anchor for everyone: the window where most of them are on screen most, the anchor (on LTX's grid) where
   they are all biggest and least cut off. The crop is around all of them.
3. Paint one person at a time onto the anchor frame, each from their own picture (and the view of their sheet their
   angle needs), named by where they stand ("the person on the left"). Each step must change its person and leave the
   others alone: a paint that changes someone else, or where SAM 3.1 can't find a character in that person's place, is
   tried once more with another seed. Whatever the painter still changed on the other people is put back.
4. One LTX-2.5 render with the painted frame pinned at the anchor.
5. Each new character is found from their own person's place on the render's anchor frame and followed on their own;
   a track that ends up on someone else's place is given back to its own person.
6. Composite: the union of every person's old + new matte, with the changed-pixels-only key. Outside it the footage is
   untouched.

Someone too small for LTX to draw well, or not on screen with the others, is left as filmed, and the take says so.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pickups import composite, features, graphs, media
from pickups.app.store import make_viewer
from pickups.tasks import common, recast
from pickups.tasks import recast_ltx as rl

WHO_KEYS = ("cast_id", "ref_upload_id", "who_ref", "describe", "reference_prompt", "reference_take")
MAX_SWAPS = 6
CUT_WEIGHT = 2.0      # window: a person half cut off by the frame's edge counts as about half on screen
ABSENT = 0.25         # at the anchor, smaller than this share of their biggest in the window: not there with the others
PAINT_REACH = 0.06    # protect other people outside this reach (share of the target's height) around the target
PAINT_BULK = 0.15     # a paint is kept only this far around its person (share of their height): a bulkier costume fits
PAINT_SIDE = 0.6      # the paint crop: this share of the person's width (or 40% of their height) each side...
PAINT_TOP = 0.15      # ...and this share of their height above and below
PAINT_PX = 1024 * 1024  # the crop is painted at about this many pixels
CHANGED_MIN = 0.25    # a paint must change at least this share of its person's pixels...
LEAK_MAX = 0.12       # ...and change at most this share of other people's pixels (well clear of the target)
COVER_MIN = 0.55      # the painted character covers at least this share of the person's place (else a new pose: LTX
                      # then draws a ghost of the old limbs, which the shot's depth still holds)
RING = 0.06           # composite: around each old person, this share of their height is replaced where the render
                      # differs from the footage (hair and motion blur SAM's outline misses)
DIFF = 40             # levels (max over RGB) that count as "changed" in those checks


# --- params ----------------------------------------------------------------------------------------------------------


def parse_swaps(p: dict) -> list[dict]:
    """params.swaps, checked: each with a frame, its points (and/or words) and who they become. A person with no one to
    become is left as filmed (dropped). A single swap with no who of its own takes the top-level one (today's form)."""
    raw = p.get("swaps")
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise RuntimeError("Recast couldn't read who to swap. Pick the people again.")
    out = []
    for s in raw:
        if not isinstance(s, dict):
            raise RuntimeError("Recast couldn't read who to swap. Pick the people again.")
        pts = [{"x": float(pt["x"]), "y": float(pt["y"]), "positive": bool(pt.get("positive", True))}
               for pt in (s.get("points") or []) if isinstance(pt, dict) and "x" in pt and "y" in pt]
        text = str(s.get("text") or "").strip()
        if not pts and not text:
            raise RuntimeError("Each person to recast needs a click in the picture.")
        who = {k: s[k] for k in WHO_KEYS if s.get(k)}
        if not who and len(raw) == 1:
            who = {k: p[k] for k in WHO_KEYS if p.get(k)}
        if not who:
            continue  # no character: as filmed
        try:
            frame = int(s.get("frame", p.get("frame", 0)) or 0)
        except (TypeError, ValueError):
            frame = 0
        sw = {"frame": max(0, frame), "points": pts, "text": text, **who}
        for k in ("kind", "label", "name"):
            if isinstance(s.get(k), str) and s[k].strip():
                sw[k] = s[k].strip()
        out.append(sw)
    if not out:
        raise RuntimeError("Pick a character for at least one person.")
    if len(out) > MAX_SWAPS:
        raise RuntimeError(f"Recast swaps up to {MAX_SWAPS} people in one take.")
    return out


def single_params(p: dict, swap: dict) -> dict:
    """Today's single-person params from one swap: its click and its character, the rest of the params as they are."""
    q = {k: v for k, v in p.items() if k != "swaps" and k not in WHO_KEYS}
    q.update(frame=swap["frame"], points=swap["points"], text=swap.get("text", ""))
    if swap.get("kind"):
        q["kind"] = swap["kind"]
    q.update({k: swap[k] for k in WHO_KEYS if swap.get(k)})
    return q


# --- where everyone is -------------------------------------------------------------------------------------------------


def on_screen(masks: list[np.ndarray]) -> np.ndarray:
    """rl.presence, less where the frame's edge cuts them (someone walking in through the edge is only partly there)."""
    pres = rl.presence(masks)
    cut = np.array([rl.cut_off(m) if m.any() else 1.0 for m in masks])
    return pres * np.clip(1.0 - CUT_WEIGHT * cut, 0.25, 1.0)


def choose_window_many(tracks: list[list[np.ndarray]], frames: int, total: int, start: int | None = None) -> int:
    """rl.choose_window for several people: the window where the most of them are on screen most (person-frames, bigger
    and whole counting a little more), the earliest of equals. The user's start wins."""
    last = max(0, total - frames)
    if start is not None:
        return max(0, min(int(start), last))
    n = min((len(t) for t in tracks), default=0)
    if last == 0 or not n:
        return 0
    score = np.sum([on_screen(t[:n]) for t in tracks], axis=0)
    sums = [float(score[s:s + frames].sum()) for s in range(min(last, n - 1) + 1)]
    return int(np.argmax(sums))


def anchor_many(tracks: list[list[np.ndarray]], grid: int = rl.GRID) -> int:
    """rl.anchor_frame for several people: on LTX's grid, the frame where they are all biggest and least cut off (each
    person's own score, as a share of their best, summed: someone missing or hidden behind the other pulls it down).
    Frame 0 when it is nearly as good (the in-place pin is strongest)."""
    n = min((len(t) for t in tracks), default=0)
    cands = list(range(0, n, grid))
    if not cands:
        return 0
    total = np.zeros(len(cands))
    for t in tracks:
        s = np.array([float(t[i].sum()) * max(0.0, 1.0 - 4.0 * rl.cut_off(t[i])) for i in cands])
        if s.max() > 0:
            total += s / s.max()
    best = int(np.argmax(total))
    if total[best] <= 0:
        return 0
    return 0 if total[0] >= rl.KEEP_FIRST * total[best] else cands[best]


def positions(masks: list[np.ndarray]) -> list[str]:
    """Where each person stands, left to right among them: "on the left", "on the right", "in the middle", "second from
    the left"... (to go in "the woman on the left of <image1>")."""
    xs = [float(np.nonzero(m)[1].mean()) if m.any() else 0.0 for m in masks]
    order = sorted(range(len(masks)), key=lambda j: xs[j])
    n = len(masks)
    names = {1: ["in the middle"], 2: ["on the left", "on the right"], 3: ["on the left", "in the middle", "on the right"]}
    ords = ["second", "third", "fourth", "fifth"]
    out = [""] * n
    for rank, j in enumerate(order):
        if n in names:
            out[j] = names[n][rank]
        elif rank == 0:
            out[j] = "on the far left"
        elif rank == n - 1:
            out[j] = "on the far right"
        elif rank < n / 2:
            out[j] = f"{ords[rank - 1]} from the left"
        else:
            out[j] = f"{ords[n - 2 - rank]} from the right"
    return out


def of_picture(where: str) -> str:
    """ "on the left" -> "on the left of", "second from the left" -> "second from the left in" (before <image1>)."""
    return f"{where} in" if "from the" in where else f"{where} of"


# --- keeping people apart ------------------------------------------------------------------------------------------------


def disjoint(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Pixels two people's masks both claim go to the one whose middle is nearest (a hug: the arm round her back stays
    his only where it is nearer his middle). Every mask the same shape."""
    if len(masks) < 2:
        return [m.copy() for m in masks]
    stack = np.stack(masks)
    both = stack.sum(0) > 1
    if not both.any():
        return [m.copy() for m in masks]
    ys, xs = np.nonzero(both)
    d = []
    for m in masks:
        if m.any():
            my, mx = (float(v.mean()) for v in np.nonzero(m))
            d.append(np.where(m[ys, xs], (ys - my) ** 2 + (xs - mx) ** 2, np.inf))
        else:
            d.append(np.full(len(ys), np.inf))
    win = np.argmin(np.stack(d), axis=0)
    out = [m.copy() for m in masks]
    for j, m in enumerate(out):
        m[ys, xs] = win == j
        m &= masks[j]
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _best_order(score: np.ndarray) -> list[int]:
    """Greedy assignment on a square score matrix (rows: tracks, columns: people): perm[j] = the person track j is."""
    n = score.shape[0]
    perm, taken = [-1] * n, set()
    for flat in np.argsort(-score, axis=None):
        j, q = divmod(int(flat), n)
        if perm[j] < 0 and q not in taken:
            perm[j] = q
            taken.add(q)
    return perm


def keep_identities(tracks: list[list[np.ndarray]], margin: float = 0.3, memory: int = 8) -> tuple[list[list[np.ndarray]], int]:
    """Two people's tracks swapping places (SAM's object 0 jumping onto the other person): when, from one frame on, the
    tracks fit the other people's last-seen masks clearly better than their own, they are swapped back for the rest of
    the shot. Returns (tracks, how many times that was done)."""
    k = len(tracks)
    n = min((len(t) for t in tracks), default=0)
    if k < 2 or not n:
        return tracks, 0
    out = [list(t[:n]) for t in tracks]
    last: list[tuple[int, np.ndarray] | None] = [None] * k
    fixes = 0
    for i in range(n):
        if i and all(last_j is not None for last_j in last):
            if all(i - last_j[0] <= memory for last_j in last):
                score = np.array([[_iou(out[j][i], last[q][1]) for q in range(k)] for j in range(k)])
                perm = _best_order(score)
                if perm != list(range(k)) and sum(score[j, perm[j]] for j in range(k)) > np.trace(score) + margin:
                    rest = [[out[j][f] for f in range(i, n)] for j in range(k)]
                    for j in range(k):
                        for f in range(i, n):
                            out[perm[j]][f] = rest[j][f - i]
                    fixes += 1
        for j in range(k):
            if out[j][i].any():
                last[j] = (i, out[j][i])
    return out, fixes


def own_person(new: list[np.ndarray], old: list[np.ndarray], reach: int, margin: float = 0.2) -> tuple[list[np.ndarray], bool]:
    """One rendered frame: each new character's mask should sit on its own old person. When the masks fit the other
    people's places clearly better (two characters swapped by the tracker), they are handed back. (masks, swapped)."""
    k = len(new)
    if k < 2:
        return new, False
    grown = [composite.grow(o, reach) if o.any() else o for o in old]
    score = np.array([[float((new[j] & grown[q]).sum()) / max(1, int(new[j].sum())) for q in range(k)] for j in range(k)])
    perm = _best_order(score)
    if perm == list(range(k)) or sum(score[j, perm[j]] for j in range(k)) <= np.trace(score) + margin:
        return new, False
    out = [np.zeros_like(new[0]) for _ in range(k)]
    for j in range(k):
        out[perm[j]] = new[j]
    return out, True


# --- painting one person at a time ------------------------------------------------------------------------------------------


def paint_effect(before: np.ndarray, after: np.ndarray, target: np.ndarray, others: np.ndarray, reach: int) -> tuple[float, float]:
    """(changed, leak): the share of the target's pixels the paint changed, and the share of other people's pixels
    (beyond `reach` of the target) it changed. RGB uint8 frames, masks at their size."""
    d = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(-1) > DIFF
    changed = float(d[target].mean()) if target.any() else 0.0
    guard = others & ~composite.grow(target, reach) if others.any() else others
    leak = float(d[guard].mean()) if guard.any() else 0.0
    return changed, leak


def keep_local(before: np.ndarray, after: np.ndarray, target: np.ndarray, others: np.ndarray, reach: int,
               bulk: int) -> np.ndarray:
    """The paint kept only around its own person (within `bulk` of them: a bulkier costume fits, a character the
    painter added elsewhere doesn't), and never on the other people (beyond `reach` of the target). Feathered."""
    near = composite.soft_matte(target, bulk, max(1.0, bulk / 3)) if target.any() else np.zeros(target.shape, np.float32)
    guard = others & ~composite.grow(target, reach) if others.any() else others
    if guard.any():
        near = near * (1.0 - composite.soft_matte(guard, max(1, reach // 3), max(1.0, reach / 3)))
    m = near[..., None]
    return np.clip(np.rint(after * m + before * (1 - m)), 0, 255).astype(np.uint8)


def paint_box(target: np.ndarray, side: float = PAINT_SIDE, top: float = PAINT_TOP, aspect: tuple[float, float] = (0.6, 1.8)
              ) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the crop the painter gets for one person. They stand in its middle, big, so "the person in the
    middle" is them (named by place in the whole frame, the painter added a new character in an empty corner instead).
    Some of the scene around them for its light; inside the frame; even numbers."""
    H, W = target.shape
    ys, xs = np.nonzero(target)
    if not len(ys):
        return 0, 0, W // 2 * 2, H // 2 * 2
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    padx, pady = side * max(bw, 0.4 * bh), top * bh
    cw, ch = bw + 2 * padx, bh + 2 * pady
    lo, hi = aspect
    if cw / ch < lo:
        cw = ch * lo
    elif cw / ch > hi:
        ch = cw / hi
    cw, ch = min(W, cw), min(H, ch)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0 = int(min(max(cx - cw / 2, 0), W - cw)) // 2 * 2
    y0 = int(min(max(cy - ch / 2, 0), H - ch)) // 2 * 2
    x1, y1 = min(W, x0 + int(cw) // 2 * 2), min(H, y0 + int(ch) // 2 * 2)
    return x0, y0, x1, y1


def paint_size(cw: int, ch: int, px: int = PAINT_PX) -> tuple[int, int]:
    """The size a crop is painted at: about `px` pixels, its own shape, multiples of 32."""
    k = (px / max(1, cw * ch)) ** 0.5
    return max(64, round(cw * k / 32) * 32), max(64, round(ch * k / 32) * 32)


def paint_order(masks: list[np.ndarray]) -> list[int]:
    """Who is painted first: the furthest (feet highest in the frame) first, so whoever stands in front is painted last,
    over them."""
    feet = [int(np.flatnonzero(m.any(axis=1))[-1]) if m.any() else 0 for m in masks]
    return sorted(range(len(masks)), key=lambda j: feet[j])


def facing_from(faces: list[np.ndarray] | None, person: np.ndarray) -> str:
    """rl.facing with the faces already found: "front" when a face is in the top third of the person, else "back";
    "" when faces couldn't be looked for."""
    if faces is None:
        return ""
    if not person.any():
        return "back"
    ys = np.flatnonzero(person.any(axis=1))
    top = np.zeros_like(person)
    top[ys[0]:ys[0] + max(1, (ys[-1] - ys[0]) // 3)] = True
    for f in faces:
        f = recast.upsize(f, (person.shape[1], person.shape[0]))
        if (f & composite.grow(person & top, 3)).any():
            return "front"
    return "back"


def ltx_prompt_many(people: list[dict], scene: str) -> str:
    """LTX-2.5's prompt for several new characters: each where they are and what they do, then the scene and its light."""
    parts = []
    for q in people:
        subject = q["who"].strip().rstrip(".") or ("the new vehicle" if q["vehicle"] else "the new character")
        parts.append(f"{q['where'][0].upper()}{q['where'][1:]}, {subject} {q['motion']}.")
    parts.append(scene.strip())
    parts.append("Everyone is lit by the scene's own natural light, with soft, realistic shadows and no glow around them; "
                 "the camera and the place stay exactly as in the original shot.")
    return " ".join(x for x in parts if x)


# --- the task ----------------------------------------------------------------------------------------------------------------


def _pick_masks(ctx, proxy: Path, swaps: list[dict], src_w: int, run_dir: Path) -> dict[int, list[np.ndarray]]:
    """Each swap's mask on its own frame: SAM 3.1 from its points (the other swaps' points on that frame as negatives,
    so a hugging couple comes back as two people) or its words, plus a bag or backpack touching them."""
    n = media.probe(proxy).frames
    scale = media.probe(proxy).width / src_w
    by_frame: dict[int, list[int]] = {}
    for j, s in enumerate(swaps):
        by_frame.setdefault(max(0, min(s["frame"], n - 1)), []).append(j)
    picks: dict[int, list[np.ndarray]] = {}
    for frame, js in by_frame.items():
        png = common._frame_png(proxy, frame, run_dir / f"pick_{frame}.png")
        masks = []
        for j in js:
            s = swaps[j]
            neg = [dict(pt, positive=False) for q in js if q != j for pt in swaps[q]["points"] if pt.get("positive", True)]
            m = common.select_preview(ctx, png, s["points"] + (neg if s["points"] else []), s.get("text", ""), scale,
                                      run_dir / f"pick_{frame}_{j}")
            if not m.any():
                raise RuntimeError(f"Nothing found where you clicked for {s.get('label') or 'one of the people'}. "
                                   "Try clicking in the middle of them.")
            masks.append(m)
        masks = disjoint(masks)
        people = [m for j, m in zip(js, masks) if swaps[j].get("kind") != "vehicle"]
        if people:
            carried = np.zeros_like(masks[0])
            for thing in ("backpack", "bag"):
                try:
                    found = common.select_preview(ctx, png, [], thing, scale, run_dir / f"pick_{frame}_{thing}")
                except Exception:  # noqa: BLE001 - an accessory we can't find is not an error
                    continue
                if found.shape == carried.shape:
                    carried |= found
            carried &= ~np.logical_or.reduce(masks)
            for piece in common.split_objects(carried, min_frac=0.0) if carried.any() else []:
                near = [int((piece & composite.grow(m, max(6, int(0.12 * common.person_height([m]))))).sum())
                        if swaps[j].get("kind") != "vehicle" else 0 for j, m in zip(js, masks)]
                if max(near) > 0:
                    masks[int(np.argmax(near))] |= piece
        picks[frame] = list(zip(js, masks))
    return picks


def run(ctx, swaps: list[dict]) -> dict:
    p = dict(ctx.params)
    run_dir = ctx.vdir / "work"
    run_dir.mkdir(exist_ok=True)
    seed = int(p.get("seed", 42))
    notes: list[str] = []

    ctx.stage("Preparing the shot")
    src = media.probe(ctx.input_path)
    frames = 8 * ((min(common.length_cap(p), src.frames) - 1) // 8) + 1
    if frames < media.MIN_FRAMES:
        raise RuntimeError("This shot is too short to recast. Try one that's at least half a second long.")
    given = p.get("start_frame")
    start = None if given in (None, "") else max(0, min(int(given), src.frames - frames))
    n_search = min(src.frames, max(rl.search_frames(), (start or 0) + frames))
    pw, ph = media.work_size(src.width, src.height, common.PROXY_LONG)
    proxy = run_dir / "proxy.mp4"
    media.prepare_work_clip(ctx.input_path, proxy, 0, n_search, (pw, ph))
    common.check_not_blank(proxy)

    ctx.stage("Finding the person")
    others: list[np.ndarray] = []  # the people no one swaps, tracked with the swapped: always kept as footage
    checked = common.checked_tracks(ctx, proxy, swaps, others_out=others)  # everyone's "Check tracking", when all were
    picks = {} if checked else _pick_masks(ctx, proxy, swaps, src.width, run_dir)
    tracks: list[list[np.ndarray]] = checked or [[] for _ in swaps]
    for g, (frame, pairs) in enumerate(sorted(picks.items())):
        besides = common.people_besides(ctx, run_dir / f"pick_{frame}.png", np.logical_or.reduce([m for _, m in pairs]), 1.0,
                                        run_dir / f"besides_{g}")
        each = common.track_each(ctx, proxy, frame, [m for _, m in pairs] + besides, run_dir, n_search,
                                 folder=f"track_{g}" if len(picks) > 1 else "")
        for (j, _), t in zip(pairs, each):
            tracks[j] = t
        for t in each[len(pairs):]:
            others = [o | m for o, m in zip(others, t)] if others else list(t)
    n_track = min(len(t) for t in tracks)
    tracks = [t[:n_track] for t in tracks]
    tracks, id_fixes = keep_identities(tracks)
    tracks = [list(x) for x in zip(*[disjoint(list(fr)) for fr in zip(*tracks)])]
    others = [o & ~np.logical_or.reduce([t[i] for t in tracks]) for i, o in enumerate(others[:n_track])]
    kinds = []
    for j, s in enumerate(swaps):
        vn = rl.vehicle_noun(s.get("text", "")) if str(s.get("kind", "")).lower() not in ("person",) else None
        if not vn and str(s.get("kind", "")).lower() == "vehicle":
            vn = "car"
        if not vn and not s.get("kind") and s["points"] and not s.get("text"):
            (run_dir / f"kind_{j}").mkdir(exist_ok=True)
            vn = rl.clicked_vehicle(ctx, proxy, min(s["frame"], n_track - 1), tracks[j], run_dir / f"kind_{j}")
        kinds.append(vn)
    if any(kinds) and not features.cars():
        # Car swap is cut from this build: one car among several swaps is refused like a single one (recast_ltx).
        raise RuntimeError(rl.NO_CARS)
    at_pick = [tracks[j][min(s["frame"], n_track - 1)] for j, s in enumerate(swaps)]
    pos0 = positions(at_pick)
    labels = [s.get("label") or f"the {rl.vehicle_words(kinds[j]) if kinds[j] else rl.person_noun(s.get('text', ''))} {pos0[j]}"
              for j, s in enumerate(swaps)]

    s0 = choose_window_many(tracks, frames, src.frames, start)
    blank = np.zeros_like(tracks[0][0])
    pms = [(t[s0:s0 + frames] + [blank] * frames)[:frames] for t in tracks]
    if start is None and s0:
        ctx.params["start_frame"] = s0

    # Too small to draw, or never on screen in the window: left as filmed (all of them: stop, as a single Recast does).
    keep = []
    for j, pm in enumerate(pms):
        h = common.person_height(pm)
        if not h:
            notes.append(f"{labels[j][0].upper()}{labels[j][1:]} isn't in this part of the shot, so they're left as filmed.")
            continue
        mh, mw = pm[0].shape
        if h * common.PROXY_LONG / max(mh, mw) < common.MIN_PERSON_PX:
            notes.append(f"{labels[j][0].upper()}{labels[j][1:]} is too small in the frame to recast well (about "
                         f"{round(h * src.height / mh)} px tall), so they're left as filmed.")
            continue
        keep.append(j)
    if not keep:
        common.require_size(pms[0], src.height)  # says why, as a single Recast does
        raise RuntimeError(common.LOST)
    k = anchor_many([pms[j] for j in keep])
    for j in list(keep):
        biggest = max(float(m.sum()) for m in pms[j])
        if len(keep) > 1 and float(pms[j][k].sum()) < ABSENT * biggest:
            notes.append(f"{labels[j][0].upper()}{labels[j][1:]} isn't in the frame with the others long enough to "
                         "recast, so they're left as filmed.")
            keep.remove(j)
    om = (others[s0:s0 + frames] + [blank] * frames)[:frames] if others else [blank] * frames
    for j in range(len(pms)):
        if j not in keep:  # left as filmed: kept as footage like anyone else
            om = [o | m for o, m in zip(om, pms[j])]
    swaps = [swaps[j] for j in keep]
    pms = [pms[j] for j in keep]
    kinds = [kinds[j] for j in keep]
    labels = [labels[j] for j in keep]

    full = run_dir / "work.mp4"
    info = common.work_clip(ctx, full, frames=frames, start=s0)
    W, H, fps = info["width"], info["height"], info["fps"]
    frames = info["frames"]
    pms = [pm[:frames] for pm in pms]
    om = om[:frames]
    union = [np.logical_or.reduce([pm[i] for pm in pms]) for i in range(frames)]
    vram = rl.gpu_vram(ctx)
    max_px = rl.max_pixels(vram)
    cx0, cy0, cw, ch = rl.crop_box(union, (W, H), max_px)
    w, h = rl.ltx_size(cw, ch, max_px)
    sx, sy = W / union[0].shape[1], H / union[0].shape[0]
    pbox = (cx0 / sx, cy0 / sy, (cx0 + cw) / sx, (cy0 + ch) / sy)
    clip = run_dir / "ltx_in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-vf", f"crop={cw}:{ch}:{cx0}:{cy0},scale={w}:{h}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", str(clip)], check=True)
    old_w = [[recast.mask_in(m, pbox, (w, h)) for m in pm] for pm in pms]
    where = positions([o[k] for o in old_w])

    ctx.stage("Reading the character sheet")
    chars = []
    base = {k_: v for k_, v in p.items() if k_ not in WHO_KEYS and k_ != "swaps"}
    for j, s in enumerate(swaps):
        sd = run_dir / f"who_{j}"
        sd.mkdir(exist_ok=True)
        sp = dict(base, **{k_: s[k_] for k_ in WHO_KEYS if s.get(k_)})
        picture, sp, what, views, made_still = rl.character_picture(ctx, sp, sd, vehicle=bool(kinds[j]))
        if made_still and (ctx.vdir / "reference.png").exists():
            (ctx.vdir / f"reference_{j + 1}.png").write_bytes((ctx.vdir / "reference.png").read_bytes())
        name = s.get("name") or next((c["name"] for c in ctx.store.shot_cast(ctx.shot_id)[0] if c["id"] == s.get("cast_id")), "") \
            or (sp.get("describe") or "").strip().rstrip(".") or "your character"
        chars.append({"picture": picture, "p": sp, "what": what, "views": views, "made_still": made_still,
                      "who": (sp.get("describe") or "").strip().rstrip("."), "name": name})

    ctx.stage("Painting them into the shot")
    anchor = common._frame_png(clip, k, run_dir / "anchor.png")
    before0 = np.asarray(Image.open(anchor).convert("RGB"))
    try:
        faces = common.select_instances(ctx, anchor, "face", 1.0, run_dir / "faces")
    except Exception:  # noqa: BLE001 - not knowing: the whole sheet, as before
        faces = None
    safe = ctx.commercial_safe()
    motions = []
    paints: list[dict] = [{} for _ in swaps]
    current = before0
    for j in paint_order([o[k] for o in old_w]):
        c, target = chars[j], old_w[j][k]
        others = np.logical_or.reduce([old_w[q][k] for q in range(len(swaps)) if q != j]) if len(swaps) > 1 else np.zeros_like(target)
        motion = rl.motion_words(pms[j])
        if kinds[j]:
            motion = rl.drive_words(motion)
            instruction = rl.paint_vehicle(kinds[j], "{where}", c["who"])
        else:
            instruction = rl.paint_instruction(rl.person_noun(swaps[j].get("text", "")), "{where}",
                                               motion.startswith("walks"), c["views"] > 1, c["who"],
                                               phrase=rl.person_phrase(swaps[j].get("text", "")))
        motions.append((j, motion))
        refs = [ctx.client.assets.from_file(c["picture"])]
        view = ""
        if c["views"] > 1 and not kinds[j]:
            view = facing_from(faces, target)
            cut = rl.sheet_view(c["picture"], view) if view else None
            if cut is not None:
                cut.save(run_dir / f"view_{j}_{view}.png")
                refs.append(ctx.client.assets.from_file(run_dir / f"view_{j}_{view}.png"))
                instruction = instruction.replace(
                    " Keep everything else in <image1> exactly as it is.",
                    f" From this angle they are seen {'from the front' if view == 'front' else 'from behind'}: copy their "
                    f"{'front' if view == 'front' else 'back'} exactly as <image3> shows it. Keep everything else in <image1> "
                    "exactly as it is.")
            else:
                view = ""
        if not kinds[j] and view != "front" and target.any() and \
                np.ptp(np.flatnonzero(target.any(axis=1))) >= rl.BACK_VIEW_MIN_H * target.shape[0] and \
                facing_from(faces, target) == "back":
            # Seen from behind (as recast_ltx): a front-facing still, or even a sheet's back view, had the painter turn
            # a runner running away to grin at the camera.
            view = "back"
            instruction = instruction.replace(" Keep everything else in <image1> exactly as it is.",
                                              " They are seen from behind, facing away: paint the back of the character's "
                                              "head and body, with no face showing. Keep everything else in <image1> exactly as it is.")
        tall = max(1, common.person_height([target]))
        reach, bulk = max(3, round(PAINT_REACH * tall)), max(4, round(PAINT_BULK * tall))
        # The painter works on a crop with this person in its middle, at about a megapixel.
        x0, y0, x1, y1 = paint_box(target)
        t_c, o_c = target[y0:y1, x0:x1], others[y0:y1, x0:x1]
        instruction = instruction.replace("{where}", rl.where_words(t_c))
        beside = [m for m in (common.split_objects(o_c) if o_c.any() else []) if m.sum() >= 0.05 * t_c.sum()]
        if beside:
            rel = positions([t_c] + beside)[0]
            n_in = len(beside) + 1
            instruction = instruction.replace(
                " Keep everything else in <image1> exactly as it is.",
                f" Change only that one person, the one {rel} of the {n_in if n_in > 2 else 'two'} people in <image1>; leave "
                f"the other {'person' if n_in == 2 else 'people'} exactly as they are, and keep everything else in <image1> "
                "exactly as it is.")
        pw_, ph_ = paint_size(x1 - x0, y1 - y0)
        src_png = run_dir / f"paint_in_{j}.png"
        Image.fromarray(current[y0:y1, x0:x1]).resize((pw_, ph_), Image.LANCZOS).save(src_png)
        tries = []
        for attempt in range(rl.PAINT_TRIES):
            pseed = seed + 1000 * attempt + j
            job = ctx.run(graphs.image_edit(ctx.client.assets.from_file(src_png), instruction, seed=pseed, refs=refs,
                                            resolution=0, prefix="pickups/recast_paint", commercial_safe=safe,
                                            size=graphs.edit_size(pw_, ph_)))
            raw = run_dir / f"paint_{j}_{attempt}.png"
            job.get_outputs("PU_Plate")[0].to_file(raw)
            big = Image.open(raw).convert("RGB").resize((pw_, ph_), Image.LANCZOS)
            after = current.copy()
            after[y0:y1, x0:x1] = np.asarray(big.resize((x1 - x0, y1 - y0), Image.LANCZOS))
            changed, leak = paint_effect(current, after, target, others, reach)
            found, cover = _found_there(ctx, np.asarray(big), recast.upsize(t_c, (pw_, ph_)), bool(kinds[j]),
                                        run_dir / f"paint_{j}_{attempt}_find")
            ok = changed >= CHANGED_MIN and leak <= LEAK_MAX and found is not None and cover >= COVER_MIN
            phone = ok and not kinds[j] and _holds_phone(ctx, np.asarray(big), recast.upsize(t_c, (pw_, ph_)),
                                                         f"{c['who']} {p.get('prompt') or ''}", run_dir / f"paint_{j}_{attempt}_find")
            ok = ok and not phone
            tries.append({"seed": pseed, "changed": round(changed, 3), "leak": round(leak, 3), "found": found,
                          "cover": round(cover, 3), "phone": phone, "ok": ok, "after": after})
            if ok:
                break
        best = next((t for t in tries if t["ok"]), None) or min(
            tries, key=lambda t: (t["found"] is None, t["leak"] > LEAK_MAX, t["phone"], -t["cover"], -t["changed"]))
        current = keep_local(current, best["after"], target, others, reach, bulk)
        paints[j] = {"attempts": len(tries), "seed": best["seed"], "changed": best["changed"], "leak": best["leak"],
                     "found": best["found"], "cover": best["cover"], "ok": best["ok"],
                     "gate": ["ok" if t["ok"] else "failed" for t in tries], "sheet_view": view or None, "instruction": instruction,
                     "crop": [x0, y0, x1 - x0, y1 - y0]}
    painted = run_dir / "painted.png"
    Image.fromarray(current).save(painted)

    ctx.stage("Performing the scene")
    motion_of = dict(motions)
    prompt = (p.get("prompt") or "").strip() or ltx_prompt_many(
        [{"who": chars[j]["who"], "motion": motion_of[j], "where": where[j], "vehicle": bool(kinds[j])} for j in range(len(swaps))],
        p.get("scene") or "")
    job = ctx.run(graphs.drive(ctx.client.assets.from_file(clip), ctx.client.assets.from_file(painted), prompt, fps, w, h,
                               control_mode="depth", seed=seed, prefix="pickups/recast_ltx", first_at=k,
                               first_strength=float(p.get("guide_strength", rl.GUIDE_STRENGTH)) if k else 1.0))
    gdir = run_dir / "gen"
    gdir.mkdir(exist_ok=True)
    for o in job.get_outputs("PU_Out"):
        o.to_file(gdir / o.name)
    gen = sorted(gdir.glob("*.png"))
    if not gen:
        raise RuntimeError("LTX-2.5 didn't send the shot back. Try again.")
    ddir = run_dir / "depth"  # the shot's own depth (the render's guide): what stands in front of the people
    ddir.mkdir(exist_ok=True)
    try:
        for o in job.get_outputs("PU_ControlPreview"):
            o.to_file(ddir / o.name)
    except Exception:  # noqa: BLE001 - no depth back: blend as before, nothing kept in front
        pass
    depth = sorted(ddir.glob("*.png"))
    if len(depth) != len(gen):
        depth = []
    gen_small = run_dir / "gen.mp4"
    sw_, sh_ = media.work_size(w, h, common.PROXY_LONG)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(gdir / "*.png"),
                    "-vf", f"scale={sw_}:{sh_}:flags=lanczos", "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", str(gen_small)],
                   check=True)

    ctx.stage("Blending at full resolution")
    track_new, seeded = find_new_many(ctx, gen_small, k, [o[k] for o in old_w], kinds, run_dir)
    m = len(swaps)
    reaches = [max(3, round(0.04 * max(1, common.person_height(old_w[j])))) for j in range(m)]
    new_w: list[list[np.ndarray]] = [[] for _ in range(m)]
    swapped_frames = 0
    for i in range(len(old_w[0])):
        raw = [recast.upsize(track_new[j][i], (w, h)) if i < len(track_new[j]) else np.zeros((h, w), bool) for j in range(m)]
        raw, swapped = own_person(raw, [old_w[j][i] for j in range(m)], max(reaches))
        swapped_frames += swapped
        raw = disjoint(raw)
        for j in range(m):
            new_w[j].append(rl.new_character(raw[j], old_w[j][i], reaches[j]))

    kf = max(W, H) / 1920
    grow_px, feather = max(2, round(6 * kf)), 4.0 * kf
    grain_on = str(p.get("grain", "fine")).lower() != "off"
    need = rl._grain_needed(full, gen, (cx0, cy0, cw, ch), run_dir) if grain_on else None
    master = ctx.vdir / "master.mov"
    out_w = common.FrameWriter(master, W, H, fps)
    rec_w = common.FrameWriter(ctx.vdir / "receipt.mp4", W, H, fps, kind="h264", scale_long=1920)
    qs = max(1, round(max(cw, ch) / 640))
    ring_px = max(grow_px, round(RING * common.person_height([m for pm in pms for m in pm]) * H / union[0].shape[0]))

    protected = [0]  # frames where someone no one swaps would have been pasted from the render and stayed footage

    def blend(i: int, frame: np.ndarray):
        out = frame.copy()
        if i >= len(gen) or i >= len(old_w[0]):
            return out, out, 0, 0, None, None
        old = np.logical_or.reduce([recast.mask_in(pm[i], pbox, (cw, ch)) for pm in pms])
        new = np.logical_or.reduce([recast.upsize(nw[i], (cw, ch)) for nw in new_w])
        protect = rl.protect_others(recast.mask_in(om[i], pbox, (cw, ch)), old, feather / 2) if i < len(om) else None
        drawn = np.logical_or.reduce([recast.upsize(recast.upsize(t[i], (w, h)), (cw, ch)) if i < len(t) else np.zeros((ch, cw), bool)
                                      for t in track_new]) | new
        matte = composite.soft_matte(old | new, grow_px, feather) if (old | new).any() else np.zeros((ch, cw), np.float32)
        if old.any():  # a wider keyed ring round the old people: their hair and motion blur go too
            matte = np.maximum(matte, composite.soft_matte(old, ring_px, feather))
        region = frame[cy0:cy0 + ch, cx0:cx0 + cw]
        if matte.any():
            g = np.asarray(Image.open(gen[i]).convert("RGB").resize((cw, ch), Image.LANCZOS))
            keep_out = np.maximum(matte, composite.grow(drawn, 2 * grow_px).astype(np.float32))
            occ = None
            if depth and old.any():  # a hand or a post in front of someone stays real footage (each person's own depth)
                dep = np.asarray(Image.open(depth[i]).convert("L").resize((cw, ch), Image.BILINEAR))
                occ = np.zeros((ch, cw), bool)
                both = [recast.mask_in(pms[j][i], pbox, (cw, ch)) | recast.upsize(new_w[j][i], (cw, ch)) for j in range(m)]
                for j in range(m):
                    o_j = recast.mask_in(pms[j][i], pbox, (cw, ch))
                    if o_j.any():
                        # The other swapped people are recast too, so never "in front" (his arm round her back).
                        others_j = np.logical_or.reduce([both[q] for q in range(m) if q != j]) if m > 1 else np.zeros_like(o_j)
                        occ |= composite.occluders(dep, o_j, composite.grow(both[j], 2 * grow_px),
                                                   speck=max(2, grow_px // 2)) & ~others_j
                occ = composite.soft_matte(occ, 1, feather / 2) if occ.any() else None
                if occ is not None:
                    keep_out = np.maximum(keep_out, occ)
            if protect is not None:
                keep_out = np.maximum(keep_out, protect)
            # Each character keeps its own colours (only a clamped brightness match); the fill around them is matched.
            subject = composite.soft_matte(drawn, max(1, grow_px // 2), feather) if drawn.any() else np.zeros_like(matte)
            g = np.clip(np.rint(composite.harmonize_around(g, region, keep_out, subject)), 0, 255).astype(np.uint8)
            core = composite.soft_matte(new, max(1, grow_px // 2), feather) if new.any() else np.zeros_like(matte)
            # Inside the old people the footage is them, never the background: there it is always the render (the key
            # kept the footage wherever LTX's fill looked like it, and the woman's long hair stayed beside the alien).
            inside = composite.soft_matte(composite.shrink(old, max(1, grow_px // 3)), 0, feather / 2) if old.any() else 0
            diff = composite.blur(np.abs(g.astype(np.float32) - region.astype(np.float32)).max(axis=-1), 2.0 * kf)
            matte = np.maximum(np.maximum(core, inside), matte * np.clip((diff - rl.KEY_LO) / (rl.KEY_HI - rl.KEY_LO), 0.0, 1.0))
            if occ is not None:
                matte = matte * (1.0 - occ)
            if protect is not None:
                if float((matte * protect).max()) > 0.05:
                    protected[0] += 1
                matte = matte * (1.0 - protect)
            if need is not None:
                g = rl._grain_in(g, matte > 0, need, seed=i)
            pasted, c = composite.paste(region, g, matte)
        else:
            pasted, c = region.copy(), 0
        out[cy0:cy0 + ch, cx0:cx0 + cw] = pasted
        touched = matte > 0
        tinted = out.copy()
        reg = pasted.astype(np.float32)
        reg[touched] = reg[touched] * 0.45 + np.array(recast.AI_TINT, np.float32) * 0.55
        tinted[cy0:cy0 + ch, cx0:cx0 + cw] = reg.astype(np.uint8)
        hit = bool(new.any())
        q = {"out": pasted[::qs, ::qs], "src": region[::qs, ::qs], "new": new[::qs, ::qs], "old": old[::qs, ::qs], "hit": hit,
             "coverage": recast.coverage(new, old),
             "halo": recast.halo_lift(pasted[::qs, ::qs], new[::qs, ::qs], kf / qs) if hit else None}
        each = [(bool(old_w[j][i].any()), bool(new_w[j][i].any()), recast.coverage(new_w[j][i], old_w[j][i])) for j in range(m)]
        return out, tinted, c, int(touched.sum()), q, each

    changed = touched_px = n = 0
    qa = recast.QA()
    per = [{"frames": 0, "empty": 0, "cov": []} for _ in range(m)]
    for i, (out, tinted, c, t, q, each) in enumerate(common.parallel_frames(common.read_frames(full), blend)):
        ctx.check()
        n += 1
        changed += c
        touched_px += t
        qa.add(q)
        for j, (there, hit, cov) in enumerate(each or []):
            per[j]["frames"] += there
            per[j]["empty"] += there and not hit
            if cov is not None:
                per[j]["cov"].append(cov)
        out_w.write(out)
        rec_w.write(tinted)
        ctx.progress((i + 1) / max(frames, 1))
    out_w.close()
    rec_w.close()

    ctx.stage("Writing your files")
    downloads = common.outputs_from_master(ctx.vdir, master, audio_from=full)
    make_viewer(full, ctx.vdir / "before.mp4")
    (ctx.vdir / "painted_frame.png").write_bytes(painted.read_bytes())
    total = W * H * max(n, 1)
    receipt = {"original": round(100 * (1 - touched_px / total), 2), "borrowed": 0.0, "ai": round(100 * touched_px / total, 2)}
    made = media.made_from(src, s0, n)
    any_still = any(c["made_still"] for c in chars)
    models = [graphs.edit_model(safe), "LTX-2.5 + Union Control IC-LoRA", "Depth Anything 3", "SAM 3.1"] + (["FLUX 3 klein"] if any_still else [])
    swaps_out = [{"who": labels[j], "character": chars[j]["name"], "reference": chars[j]["what"],
                  **({"subject": "vehicle"} if kinds[j] else {})} for j in range(m)]
    people_qa = [{"who": labels[j], "character": chars[j]["name"],
                  "coverage": round(float(np.median(per[j]["cov"])), 3) if per[j]["cov"] else None,
                  "frames_on_screen": per[j]["frames"], "empty_new_frames": per[j]["empty"], "new_seed": seeded[j],
                  "paint": {k_: v for k_, v in paints[j].items() if k_ != "instruction"}, "where": where[j]} for j in range(m)]
    return {"label": "Replace a person", "task": "recast", "receipt": receipt, "downloads": downloads,
            "details": {**common.size_details(W, H, n, fps), "engine": "ltx", "performance_size": f"{w}×{h}",
                        "reference": "; ".join(f"{c['name']}: {c['what']}" for c in chars),
                        "swaps": swaps_out, **({"left_as_filmed": notes} if notes else {}),
                        "pixels_changed_outside_selection": changed, "models": models,
                        **({"made_from": made} if made else {}),
                        "grain": "matched to your shot" if grain_on else "off",
                        "look_prompt": " | ".join(paints[j]["instruction"] for j in range(m)), "prompt": prompt,
                        "_qa": {**qa.summary(), "window_start": s0, "window_chosen": "yours" if start is not None else "auto",
                                "anchor_frame": k, "guide": "first frame" if k == 0 else f"frame {k}",
                                "people": people_qa, "identity_fixes_tracking": id_fixes,
                                "identity_fixes_render": int(swapped_frames), "protected_frames": protected[0],
                                "crop": [cx0, cy0, cw, ch], "work_size": [w, h], "max_pixels": max_px, "gpu_vram_gb": vram,
                                "searched_frames": n_search,
                                "grain_added": None if need is None else [round(float(x), 2) for x in need]}}}


def _found_there(ctx, img: np.ndarray, target: np.ndarray, vehicle: bool, out_dir: Path) -> tuple[str | None, float]:
    """Where SAM 3.1 finds the painted character on the painted crop: (the label it answers to, None when it isn't in
    the person's place; how much of the person's place it covers). A detection counts when it covers at least a third
    of the place. (Which way it faces can't be checked this way: SAM calls the back of a helmet or a bald alien head a
    "face".)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "painted.png"
    Image.fromarray(img).save(png)
    for label in (rl.VEHICLE_LABELS[:2] if vehicle else recast.NEW_LABELS[:2]):
        try:
            found = common.select_instances(ctx, png, label, 1.0, out_dir / label)
        except Exception:  # noqa: BLE001 - try the next word
            continue
        hits = [recast.upsize(m, (target.shape[1], target.shape[0])) for m in found]
        hits = [m for m in hits if (m & target).sum() >= target.sum() / 3]
        if hits:
            char = np.logical_or.reduce(hits)
            cover = float((char & target).sum()) / max(1, int(target.sum()))
            return label, cover
    return None, 0.0


def _holds_phone(ctx, img: np.ndarray, target: np.ndarray, wanted: str, out_dir: Path) -> bool:
    """rl.paint_problem's phone check on the painted crop: a phone at the person the character wasn't asked to hold."""
    if rl.HELD_WORDS.search(wanted or ""):
        return False
    try:
        held = common.select_instances(ctx, out_dir / "painted.png", "phone", 1.0, out_dir / "phone")
    except Exception:  # noqa: BLE001 - a check we can't run passes
        return False
    near = composite.grow(target, max(3, target.shape[0] // 100))
    return any((recast.upsize(m, (target.shape[1], target.shape[0])) & near).any() for m in held)


def assign(found: list[np.ndarray], old: list[np.ndarray], reach: int = 3) -> list[np.ndarray]:
    """SAM's detections on the render's anchor frame, each given to the person whose place it overlaps most (as a share
    of the detection): one seed per person, empty when none."""
    seeds = [np.zeros_like(o) for o in old]
    grown = [composite.grow(o, reach) if o.any() else o for o in old]
    for f in found:
        ov = [int((f & g).sum()) for g in grown]
        if max(ov, default=0) > 0:
            seeds[int(np.argmax(ov))] |= f
    return seeds


def find_new_many(ctx, gen_small: Path, k: int, old_k: list[np.ndarray], kinds: list[str | None],
                  run_dir: Path) -> tuple[list[list[np.ndarray]], list[str]]:
    """rl.find_new for several people: SAM 3.1 finds the new characters on the render's anchor frame, each detection
    goes to the person whose place it is on (a person it finds nothing for is seeded with their own outline), and they
    are tracked together, one object each."""
    png = common._frame_png(gen_small, k, run_dir / "new_anchor.png")
    size = Image.open(png).size
    old = [recast.upsize(o, size) for o in old_k]
    seeds, how = [np.zeros_like(o) for o in old], ["the person's outline"] * len(old)
    labels = rl.VEHICLE_LABELS if all(kinds) else recast.NEW_LABELS
    for label in labels:
        if all(s.any() for s in seeds):
            break
        try:
            found = [f for f in common.select_instances(ctx, png, label, 1.0, run_dir / f"new_{label}") if f.shape == old[0].shape]
        except Exception:  # noqa: BLE001 - try the next word
            continue
        for j, s in enumerate(assign(found, old)):
            if s.any() and not seeds[j].any():
                seeds[j], how[j] = s, label
    seeds = disjoint([s if s.any() else o for s, o in zip(seeds, old)])
    seeds = [s if s.any() else o for s, o in zip(seeds, old)]
    tracks = common.track_each(ctx, gen_small, k, seeds, run_dir, folder="new_track", prefix="pickups/recast_track")
    return tracks, how
