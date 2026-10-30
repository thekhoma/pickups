"""Seeds: the number a generation starts its noise from. The same seed with the same settings gives the same picture.

Every generation that would pick a random seed (Create's images and videos, a character sheet, the start frame) picks
it here, and the seed it used goes back on the result. The page's Lock seed sends the last one again; off, it sends a
fresh one (or none, and the server picks).
"""

from __future__ import annotations

import random

MAX = 2**31 - 1  # what every seed node takes (RandomNoise, KSampler, SamplerCustom)


def given(value) -> bool:
    """True when the caller sent a seed (a number, or digits); "", None and "None" mean pick one."""
    return value is not None and str(value).strip() not in ("", "None")


def fresh() -> int:
    """A new random seed, 1..MAX."""
    return random.randint(1, MAX)


def pick(value=None) -> int:
    """The seed to use: the one sent, else a fresh random one. Raises ValueError for one that isn't a whole number."""
    if not given(value):
        return fresh()
    try:
        n = int(str(value).strip())
    except ValueError:
        raise ValueError("The seed must be a whole number.") from None
    if n < 0:
        raise ValueError("The seed must be a whole number, 0 or more.")
    return n
