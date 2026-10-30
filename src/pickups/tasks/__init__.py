"""Pickups tasks. Each one turns a shot (or a version of it) into a new version, through ComfyUI workflows."""

from __future__ import annotations

from pickups.app.jobs import Stage, TaskSpec
from pickups.tasks import environment, finish, image, previz, recast, remove, scene_relight, sheet, video


def _reshoot(ctx) -> dict:
    raise RuntimeError("Reshoot is still being tuned. It will be back in a later build.")


TASKS: dict[str, TaskSpec] = {
    "remove": TaskSpec(remove.STAGES, remove.run),
    "recast": TaskSpec(recast.STAGES, recast.run, stages_for=recast.stages_for),
    "previz": TaskSpec(previz.STAGES, previz.run, stages_for=previz.stages_for),
    "environment": TaskSpec(environment.STAGES, environment.run),
    "image": TaskSpec(image.STAGES, image.run, scope="project", stages_for=image.stages_for),
    "sheet": TaskSpec(sheet.STAGES, sheet.run, scope="project"),
    "video": TaskSpec(video.STAGES, video.run, scope="project"),  # text or image to video: makes a new shot
    "finish": TaskSpec(finish.STAGES, finish.run, stages_for=finish.stages_for),
    "relight": TaskSpec(scene_relight.STAGES, scene_relight.run),
    "reshoot": TaskSpec([Stage("Reading the scene in 3D", 60), Stage("Moving the camera", 30), Stage("Rendering the new angle", 170)], _reshoot),
}
