"""What this build offers. Car swap and relight wait for the Car Pose control, so they are off for the contest build;
PICKUPS_CARS=1 turns them back on (their code stays as it is)."""

import os


def cars() -> bool:
    return os.environ.get("PICKUPS_CARS", "") == "1"
