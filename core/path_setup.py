"""Shared sys.path setup for scripts that reach into GNC-Payload (a sibling
flight-software repo at a fixed absolute path on this machine, not vendored
under third_party/ -- see integration/batchopt_adapter.py's docstring for why).

GNC-Payload has its own top-level `scripts` package (scripts/run_dynamics.py),
which collides with this repo's scripts/evaluate.py: Python regular packages
(with __init__.py) shadow entirely rather than merging, so whichever root
sys.path resolves first wins for the *whole* `scripts` name.

The naive fix -- `if p not in sys.path: sys.path.insert(0, p)` for each root
in the right order -- is NOT actually safe: if one of the two roots is
already on sys.path from something else (e.g. `python -m pkg.module`
auto-inserts the repo root as sys.path[0] before this code ever runs), the
guard skips re-inserting it, silently leaving it behind the other root and
corrupting the intended order. Confirmed by direct reproduction: this exact
bug broke `scripts.evaluate` resolution under `python -m
integration.two_frame_od_demo` even though the insertion code "looked"
correct. This helper always moves both to the front, unconditionally, so the
final order is guaranteed regardless of what sys.path looked like before.
"""

import sys

GNC_PAYLOAD_ROOT = "/home/pvijayba/GNC-Payload"


def ensure_repo_root_first(repo_root: str) -> None:
    """Call once, near the top of any script that imports both this repo's
    `scripts` package and anything from GNC_PAYLOAD_ROOT. Leaves repo_root at
    sys.path[0] and GNC_PAYLOAD_ROOT at sys.path[1], no matter what sys.path
    contained before this call.
    """
    for p in (GNC_PAYLOAD_ROOT, repo_root):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
