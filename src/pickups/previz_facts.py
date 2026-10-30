"""What an app-made previz already knows about its figure and camera, for Replace a person on a previz take.

Run 11 (R&D, Run 10 check 2): on the previz mannequin take (back to camera, the camera pushing in and orbiting) the
picture checks read the mannequin's head as a face ("front"), the orbit as a still camera, and the push-in as a walk
towards the camera: the new man faced the camera and walked. The previz recorded all three: the figure's facing at
frame 0 (runtime.ts facingOf), what it does and how fast (doing, speed_ms) and the camera's first and last frame
(framing.move). So on a previz take these are the truth, and the picture checks are only for imported footage.

The facing on a later frame comes from frame 0's facing and how far the camera has swung round the figure by then
(the orbit, seen from above). Near the line between two facings it is unsure (""), and nothing is said about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A take made from the previz that keeps its camera and blocking: the facts still hold on it.
SAME_BLOCKING = ("previz", "finish")
# A figure in one of these poses stays where it is even with a speed.
STILL_POSES = ("stand", "idle", "sit", "kneel", "lie", "wave", "talk", "read", "type", "sip", "hold_phone", "point", "sing")
POSTURE = {"sit": "sitting", "kneel": "kneeling", "lie": "lying down"}  # else "standing"
MOVE_M, TURN_DEG = 0.1, 2.0  # the camera moves when it travels this far, or turns or swings round this much
# The relative angle between the figure's forward and the line to the camera: 0 faces the camera, 180 faces away.
# A margin either side of facingOf's 53 degrees (cos 0.6) is unsure.
FRONT_MAX, BACK_MIN = 40.0, 140.0
SIDE_MIN, SIDE_MAX = 65.0, 115.0
ANGLE = {"camera": 0.0, "away": 180.0, "left": 90.0, "right": -90.0}


@dataclass
class Figure:
    """The previz figure being replaced: its facing at frame 0, whether it travels, and its posture word."""
    facing: str | None  # camera / away / left / right at frame 0
    moving: bool
    posture: str  # "standing", "sitting" ...


@dataclass
class Facts:
    figure: Figure | None
    camera_moves: bool | None  # None: the previz didn't record its camera
    turn_deg: float  # how far the camera swings round its subject over the shot, signed (seen from above)
    ease: str  # the camera's ease ("in_out", "linear")
    frames: int  # the previz shot's frames
    offset: int = 0  # the take's frame 0 on the previz (a previz-to-final take of a later window)

    def facing_at(self, frame: int) -> str:
        """The figure's facing on this frame of the take: camera / away / left / right, or "" when unsure."""
        if self.figure is None or self.figure.facing not in ANGLE:
            return ""
        t = min(1.0, max(0.0, (frame + self.offset) / max(1, self.frames - 1)))
        if self.ease == "in_out":
            t = t * t * (3 - 2 * t)
        turn = self.turn_deg * t
        if abs(turn) < 1e-6:
            return self.figure.facing
        return facing_of(ANGLE[self.figure.facing] - turn)

    def view_at(self, frame: int) -> str:
        """facing_at in the words recast_ltx.paint_view reads: "front", "back", "side-left", "side-right", or ""."""
        return {"camera": "front", "away": "back", "left": "side-left", "right": "side-right"}.get(self.facing_at(frame), "")

    def motion(self, frame: int) -> str:
        """What the figure does, in LTX's words: "is standing in place", or which way it walks (its own facing)."""
        if self.figure is None:
            return ""
        if not self.figure.moving:
            return f"is {self.figure.posture} in place"
        return {"camera": "walks towards the camera", "away": "walks away from the camera",
                "left": "walks to the left of the picture", "right": "walks to the right of the picture"}.get(
            self.facing_at(frame), "walks as in the original shot")


def facing_of(angle: float) -> str:
    """A relative angle (the figure's forward against the line to the camera, degrees) as a facing; "" near a line."""
    a = (angle + 180.0) % 360.0 - 180.0
    if abs(a) <= FRONT_MAX:
        return "camera"
    if abs(a) >= BACK_MIN:
        return "away"
    if SIDE_MIN <= abs(a) <= SIDE_MAX:
        return "left" if a > 0 else "right"
    return ""


def _ends(spec: dict, framing: dict) -> tuple[list[float], list[float], list[float], list[float]] | None:
    m = framing.get("move") if isinstance(framing.get("move"), dict) else None
    keys = ((spec or {}).get("camera") or {}).get("keys") or []
    a, b = (m.get("start"), m.get("end")) if m else ((keys[0], keys[-1]) if len(keys) >= 2 else (None, None))
    try:
        (ap, al), (bp, bl) = [(list(map(float, k["pos"])), list(map(float, k["look"]))) for k in (a, b)]
    except (TypeError, KeyError, ValueError):
        return None
    return ap, al, bp, bl


def camera_turn(spec: dict, framing: dict) -> float:
    """How far the camera swings round what it looks at over the shot, seen from above (signed degrees; 0 unknown)."""
    ends = _ends(spec, framing)
    if ends is None:
        return 0.0
    ap, al, bp, bl = ends
    turn = math.degrees(math.atan2(bp[2] - bl[2], bp[0] - bl[0]) - math.atan2(ap[2] - al[2], ap[0] - al[0]))
    return (turn + 540.0) % 360.0 - 180.0


def camera_moves(spec: dict, framing: dict) -> bool | None:
    """True when the previz camera travels, turns or swings round (a push, a pan, an orbit) or is handheld; None when
    the previz didn't record its camera."""
    cam = (spec or {}).get("camera") or {}
    if (cam.get("handheld") or {}).get("pos_m"):
        return True
    ends = _ends(spec, framing)
    if ends is None:
        return None
    ap, al, bp, bl = ends
    if math.dist(ap, bp) > MOVE_M or abs(camera_turn(spec, framing)) > TURN_DEG:
        return True
    da, db = [x - y for x, y in zip(al, ap)], [x - y for x, y in zip(bl, bp)]
    na, nb = math.hypot(*da), math.hypot(*db)
    if na and nb:
        cos = sum(x * y for x, y in zip(da, db)) / (na * nb)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > TURN_DEG
    return False


def figure(framing: dict, x: float | None = None) -> Figure | None:
    """The previz person in view at frame 0 (with several, the one nearest `x`, 0 left to 1 right; None when that
    can't be told)."""
    people = [s for s in framing.get("subjects") or [] if isinstance(s, dict) and s.get("kind") == "person" and s.get("in_view")]
    if not people:
        return None
    if len(people) > 1:
        if x is None:
            return None
        people = sorted(people, key=lambda s: abs(float(s.get("x", 0.5)) - x))
        if abs(float(people[1].get("x", 0.5)) - x) - abs(float(people[0].get("x", 0.5)) - x) < 0.1:
            return None  # two figures about as near: not sure which one
    s = people[0]
    doing = str(s.get("doing") or "")
    speed = s.get("speed_ms")
    moving = isinstance(speed, (int, float)) and speed > 0 and doing not in STILL_POSES
    return Figure(facing=s.get("facing") if s.get("facing") in ANGLE else None, moving=moving,
                  posture=POSTURE.get(doing, "standing"))


def facts(spec: dict, framing: dict, frames: int, x: float | None = None, offset: int = 0) -> Facts:
    cam = (spec or {}).get("camera") or {}
    return Facts(figure=figure(framing, x), camera_moves=camera_moves(spec, framing), turn_deg=camera_turn(spec, framing),
                 ease=str(cam.get("ease") or "linear"), frames=max(1, int(frames)), offset=max(0, int(offset)))


def of_take(store, shot_id: str | None, version_id: str | None) -> tuple[dict, dict, int, int] | None:
    """(spec, framing, offset, frames) when the take is an app-made previz (the Previz tab's original) or a take made
    from it that keeps its camera and blocking (SAME_BLOCKING); None for imported footage or a take that changed who's
    there. frames: the previz's own length (0 when the shot didn't record it)."""
    if store is None or not shot_id:
        return None
    try:
        meta = store.shot_meta(shot_id)
    except (OSError, ValueError, KeyError):
        return None
    framing = meta.get("previz_framing")
    if not isinstance(framing, dict) or not framing.get("subjects"):
        return None
    v, offset = version_id or "orig", 0
    for _ in range(32):
        if v == "orig":
            spec = meta.get("previz_spec")
            frames = meta.get("frames")
            return (spec if isinstance(spec, dict) else {}), framing, offset, int(frames) if isinstance(frames, (int, float)) else 0
        try:
            vm = store.version_meta(shot_id, v)
        except (OSError, ValueError, FileNotFoundError):
            return None
        if vm.get("task") not in SAME_BLOCKING:
            return None
        offset += int((vm.get("params") or {}).get("start_frame", 0) or 0) if vm.get("task") == "previz" else 0
        v = vm.get("input_version") or "orig"
    return None
