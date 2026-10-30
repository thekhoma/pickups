"""The manifest covers what the graphs load: every class_type in graphs.py, images.py and tasks/ is a feature node, a pack
node, a stock node or an opt-in one, and every model file named there is in MODELS, an image alias or the unlisted set.
No ComfyUI needed. This is what catches the next file that a feature loads but Setup never checks (sdpose was one)."""

import glob
import re
from pathlib import Path

from pickups.app import manifest
from pickups.images import ALIASES

SRC = Path(__file__).resolve().parents[1] / "src" / "pickups"
FILES = [SRC / "graphs.py", SRC / "images.py", *map(Path, glob.glob(str(SRC / "tasks" / "*.py")))]


def _found(pattern: str) -> set[str]:
    out: set[str] = set()
    for f in FILES:
        out |= set(re.findall(pattern, f.read_text()))
    return out


def test_every_node_the_graphs_use_is_accounted_for():
    used = _found(r'"class_type":\s*"([A-Za-z0-9_]+)"')
    known = (set().union(*manifest.TASK_NODES.values()) | {n for p in manifest.NODE_PACKS.values() for n in p["nodes"]}
             | manifest.STOCK_NODES | manifest.OPTIONAL_NODES)
    assert used, "no class_type found: the scan pattern is stale"
    assert sorted(used - known) == []


def test_every_model_the_graphs_load_is_in_the_manifest():
    used = _found(r'["\']([A-Za-z0-9_.\-]+\.(?:safetensors|pth|ckpt|pt|gguf))["\']')
    aliases = set(ALIASES) | {a for alts in ALIASES.values() for a in alts}
    assert sorted(m for m in used if m not in manifest.MODELS and m not in aliases and m not in manifest.UNLISTED_MODELS) == []


def test_feature_nodes_never_overlap_the_stock_list():
    feature = set().union(*manifest.TASK_NODES.values())
    assert sorted(feature & manifest.STOCK_NODES) == []
