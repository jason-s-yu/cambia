"""
tests/test_ffi_crossing_budget.py

Crossing budget for the tabular traversal (cambia-1902).

The Go port of the traversal (cambia-1782) bought no throughput over the Python
engine because it read a node one export at a time: 20.9 foreign-function
crossings per applied action on the pinned gate run, of which only four (save,
apply, restore, snapshot free) are irreducible. ``cambia_game_apply_and_read``
folds the whole read into the apply's own crossing.

This counts the crossings a real traversal makes rather than asserting a
property of the accessor in isolation, because the budget is a property of the
call pattern: a caller that goes back to the single-purpose accessors, or drops
the cached view and re-reads it inside a node, puts the crossings straight back
without breaking any correctness test. The counting proxy wraps the loaded
library, so every call through ``GoEngine`` is counted whatever made it.
"""

import collections

import pytest

from src.ffi import bridge
from tests import tabular_table_gate as gate

#: Crossings per applied action the traversal is held to. Four of these are the
#: irreducible pair-per-branch (save, apply, restore, snapshot free); the rest
#: is per-traversal setup amortised over the run's applies.
BUDGET = 5.0

#: Enough iterations for the per-traversal setup to amortise and for the run to
#: reach snap windows and abilities, without making this a slow test.
ITERATIONS = 30


class _CountingLib:
    """The loaded library, counting every call made through it."""

    def __init__(self, lib, counts):
        self._lib = lib
        self._counts = counts

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        counts = self._counts

        def wrapper(*args):
            counts[name] += 1
            # cambia_game_apply_and_read applies when its action index is >= 0
            # and only reads when it is negative. Both are one crossing; only
            # the first is an applied action.
            if name == "cambia_game_apply_and_read":
                if int(args[1]) >= 0:
                    counts["applied"] += 1
            elif name == "cambia_game_apply_action":
                counts["applied"] += 1
            return fn(*args)

        return wrapper


def _go_available() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            engine = bridge.GoEngine(seed=0)
            engine.close()
        return True
    except Exception:
        return False


skip_if_no_go = pytest.mark.skipif(
    not _go_available(), reason="libcambia.so not available"
)


@pytest.fixture(scope="module")
def counts():
    """Crossings by export name over a pinned tabular run."""
    if not _go_available():
        pytest.skip("libcambia.so not available")
    tallies: collections.Counter = collections.Counter()
    original = bridge._LIB
    bridge._LIB = _CountingLib(bridge._get_lib(), tallies)
    try:
        gate.run_tables(iterations=ITERATIONS)
    finally:
        bridge._LIB = original
    return tallies


@skip_if_no_go
def test_the_run_actually_applied_actions(counts):
    """A budget over zero applies would pass by doing nothing."""
    assert counts["applied"] > 100, f"only {counts['applied']} actions applied"


@skip_if_no_go
def test_crossings_per_applied_action_stay_under_budget(counts):
    applied = counts["applied"]
    total = sum(n for name, n in counts.items() if name != "applied")
    per_apply = total / applied
    assert per_apply < BUDGET, (
        f"{per_apply:.3f} crossings per applied action, budget {BUDGET}. "
        f"Breakdown: {dict(counts.most_common())}"
    )


@skip_if_no_go
def test_the_node_read_shares_the_apply_crossing(counts):
    """The per-node state read is not a crossing of its own.

    Reads outnumbering applies by more than the one-per-traversal opening read
    means something dropped the cached view mid-node and re-read it.
    """
    reads = counts["cambia_game_apply_and_read"]
    assert reads <= counts["applied"] + ITERATIONS, (
        f"{reads} state reads for {counts['applied']} applies over "
        f"{ITERATIONS} traversals"
    )


@skip_if_no_go
def test_the_single_purpose_accessors_left_the_hot_path(counts):
    """Nothing the batched record carries is still read one field at a time."""
    folded = [
        "cambia_game_is_terminal",
        "cambia_game_acting_player",
        "cambia_game_decision_ctx",
        "cambia_game_turn_number",
        "cambia_game_stock_len",
        "cambia_game_discard_top_card",
        "cambia_game_cambia_caller",
        "cambia_game_get_snap_state",
        "cambia_game_get_pending",
        "cambia_game_get_hand",
        "cambia_agent_action_mask",
        "cambia_game_num_players",
        "cambia_game_get_utility",
        # Folded by cambia-1971, when the belief moved onto the Go agent: the
        # acting seat's key rides the same record as the engine state.
        "cambia_agent_get_own_hand",
        "cambia_agent_get_opp_belief",
        "cambia_agent_get_hand_lens",
        "cambia_agent_update",
        "cambia_agents_update_both",
    ]
    still_called = {name: counts[name] for name in folded if counts[name]}
    assert not still_called, f"folded accessors still called: {still_called}"
