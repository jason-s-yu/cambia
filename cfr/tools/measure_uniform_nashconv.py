"""cfr/tools/measure_uniform_nashconv.py

X2R1 (cambia-517): exact NashConv of the uniform-over-legal-actions policy on
the corrected {A,6} tiny Cambia tree.

The X2 respec preregistration (.docs/v0.4/phase2-throughput-pilot/
x2-respec-preregistration.md) needs a scale-anchored bar: bar_respec =
0.0340 * U, where U is the exact NashConv of the maximally-uninformed
(uniform) policy on the SAME tree the neural PRT-CFR policy is scored on
(``src.cfr.prtcfr_eval.build_tiny_tree`` / ``score_policy_on_tiny_game``).
This script computes U directly rather than approximating it, reusing the
exact best-response/NashConv routine (``tools.tiny_solver.exploitability``)
that ``score_policy_on_tiny_game`` calls, with a policy dict built by
assigning uniform mass over ``node.actions`` at every distinct perfect-recall
infoset (``src.cfr.prtcfr_eval.enumerate_infosets``).

Editable-install .pth trap
---------------------------
Script-mode python can silently import a DIFFERENT checkout's ``src`` package
via the ``__editable__.cambia_cfr-*.pth`` finder if sys.path resolution picks
up a stale entry (see project memory: uv-sync-active-trap /
editable-install-pth-trap, cambia-240). This module pins ``cfr/`` (this
file's parent) at the front of sys.path AND os.environ["PYTHONPATH"] before
any ``src.*`` import, then asserts the resolved module file lives under this
checkout. The run is invalid without the printed assertion line.

Usage:
  cd cfr && python tools/measure_uniform_nashconv.py
"""

from __future__ import annotations

import math
import os
import sys

# --- pin PYTHONPATH / sys.path to THIS checkout's cfr/ before any src.* import ---
_CFR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_CFR_DIR)  # .../cambia

if _CFR_DIR not in sys.path:
    sys.path.insert(0, _CFR_DIR)
os.environ["PYTHONPATH"] = _CFR_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402

from src.cfr import prtcfr_eval  # noqa: E402
from tools.tiny_solver import exploitability  # noqa: E402

_resolved = os.path.abspath(prtcfr_eval.__file__)
assert _resolved.startswith(_CFR_DIR + os.sep), (
    f"src.cfr.prtcfr_eval resolved OUTSIDE this checkout: {_resolved!r} "
    f"(expected under {_CFR_DIR!r}). Editable-install .pth trap -- abort."
)
print(f"[assert] src.cfr.prtcfr_eval.__file__ = {_resolved}")

# bar_respec = RESPEC_MULTIPLIER * U, per the X2 respec preregistration.
RESPEC_MULTIPLIER = 0.0340


def build_uniform_policy(root) -> dict:
    """{pkey: uniform_vector(nA)} over every distinct perfect-recall infoset."""
    nodes = prtcfr_eval.enumerate_infosets(root)
    policy: dict = {}
    for node in nodes:
        nA = len(node.actions)
        policy[node.pkey] = np.ones(nA, dtype=np.float64) / nA
    return policy


def round_sig(x: float, sig: int = 2) -> float:
    """Round to ``sig`` significant figures (not decimal places)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


def main() -> None:
    root, isets, nnodes, aborted = prtcfr_eval.build_tiny_tree()
    policy = build_uniform_policy(root)
    nashconv, components = exploitability(root, policy)
    br0, br1, onp0, onp1 = components
    bar_respec = round_sig(RESPEC_MULTIPLIER * nashconv, sig=2)

    print(f"num_infosets(builder)={len(isets)} num_infosets(policy)={len(policy)} "
          f"nnodes={nnodes} aborted={aborted}")
    print(f"components: br0={br0!r} br1={br1!r} onp0={onp0!r} onp1={onp1!r}")
    print(f"U (uniform-policy exact NashConv) = {nashconv!r}")
    print(f"bar_respec = {RESPEC_MULTIPLIER} * U = {bar_respec!r}")


if __name__ == "__main__":
    main()
