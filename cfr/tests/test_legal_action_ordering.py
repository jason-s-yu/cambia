"""
tests/test_legal_action_ordering.py

Regression tests for canonical, hash-seed-independent ordering of
CambiaGameState.get_legal_actions().

Background: get_legal_actions() used to return a Set[GameAction]. Python set
iteration order for these NamedTuple actions depends on PYTHONHASHSEED
(string-field hashing is randomized per-process by default), so any consumer
that iterated the set directly -- e.g. `rng.choice(list(get_legal_actions()))`
with a fixed RNG seed -- got a different, process-dependent action sequence.
This caused tests/test_sequence_tokenizer.py::test_full_2p_roundtrip_lossless
to fail intermittently (~20-25% of fresh processes) depending on hash seed
and prior test-collection order.

The fix (src/game/_query_mixin.py): get_legal_actions() now returns a list,
sorted by _legal_action_sort_key -- (type(action).__name__, action) -- which
never touches Python's hash-randomized ordering.

These tests verify the fix at the engine boundary directly (not via the
tokenizer or PRT-CFR production worker, which are out of scope here and
already canonicalize independently at their own driver boundaries).
"""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.constants import GameAction
from src.game.engine import CambiaGameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _play_random_game(seed: int, rng_seed: int, max_steps: int = 300) -> List[GameAction]:
    """Play a full game from a seeded deal, picking actions via a seeded RNG
    directly from get_legal_actions()'s own ordering (mirrors the pattern
    that flaked in test_sequence_tokenizer.py). Returns the list of applied
    actions."""
    game = CambiaGameState(seed=seed)
    rng = random.Random(rng_seed)
    applied: List[GameAction] = []
    for _ in range(max_steps):
        if game.is_terminal():
            break
        actor = game.get_acting_player()
        if actor == -1:
            break
        legal = game.get_legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        applied.append(action)
        game.apply_action(action)
    return applied


# ---------------------------------------------------------------------------
# 1. get_legal_actions() returns a list (API contract)
# ---------------------------------------------------------------------------


def test_get_legal_actions_returns_a_list():
    game = CambiaGameState(seed=1)
    legal = game.get_legal_actions()
    assert isinstance(legal, list), (
        f"get_legal_actions() must return a list for deterministic iteration "
        f"order, got {type(legal)}"
    )
    assert len(legal) > 0, "fresh game should have legal start-of-turn actions"


# ---------------------------------------------------------------------------
# 2. Same game, two engine instances stepped identically -> byte-identical
#    get_legal_actions() sequences at every step.
# ---------------------------------------------------------------------------


def test_two_instances_stepped_identically_yield_byte_identical_legal_actions():
    """Two independently-constructed CambiaGameState instances with the same
    deal, driven by mirrored actions chosen from instance A's legal-action
    list, must report exactly the same legal-action list (same content, same
    order) at every decision point -- not merely the same set of legal moves."""
    seed = 4242
    rng_seed = 777
    game_a = CambiaGameState(seed=seed)
    game_b = CambiaGameState(seed=seed)
    rng = random.Random(rng_seed)

    steps = 0
    for _ in range(300):
        if game_a.is_terminal() or game_b.is_terminal():
            break
        actor_a = game_a.get_acting_player()
        actor_b = game_b.get_acting_player()
        assert actor_a == actor_b, "mirrored instances diverged on acting player"
        if actor_a == -1:
            break

        legal_a = game_a.get_legal_actions()
        legal_b = game_b.get_legal_actions()

        assert (
            legal_a == legal_b
        ), f"legal action lists diverged at step {steps}: {legal_a!r} != {legal_b!r}"
        # "byte-identical" -- compare the canonical string form too, not just
        # NamedTuple equality, so a hash-driven reordering that happened to
        # preserve == equality of the containers would still be caught.
        assert [repr(a) for a in legal_a] == [repr(a) for a in legal_b]

        if not legal_a:
            break
        action = rng.choice(legal_a)
        game_a.apply_action(action)
        game_b.apply_action(copy.deepcopy(action))
        steps += 1

    assert steps > 0, "mirrored replay produced no decision steps"


# ---------------------------------------------------------------------------
# 3. Cross-process determinism: identical PYTHONHASHSEED-independent ordering.
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = """
import random
import sys
from src.game.engine import CambiaGameState

def main():
    game = CambiaGameState(seed=137)
    rng = random.Random(999)
    trace = []
    for _ in range(300):
        if game.is_terminal():
            break
        actor = game.get_acting_player()
        if actor == -1:
            break
        legal = game.get_legal_actions()
        if not legal:
            break
        trace.append([repr(a) for a in legal])
        action = rng.choice(legal)
        game.apply_action(action)
    for frame in trace:
        print("|".join(frame))
    sys.exit(0)

main()
"""


def _run_with_hashseed(hashseed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess with PYTHONHASHSEED={hashseed} failed: "
        f"stdout={result.stdout!r} stderr={result.stderr[-4000:]!r}"
    )
    assert (
        result.stdout.strip()
    ), f"subprocess with PYTHONHASHSEED={hashseed} produced no decision steps"
    return result.stdout


def test_legal_action_order_identical_across_different_pythonhashseed():
    """Same seeded game, same RNG-driven replay, run in fresh subprocesses
    with deliberately different PYTHONHASHSEED values. The recorded
    get_legal_actions() sequence at every step must be byte-identical --
    this is the direct regression test for the original flake, which was
    caused by set-iteration order varying with the hash seed."""
    out_a = _run_with_hashseed("0")
    out_b = _run_with_hashseed("1")
    out_c = _run_with_hashseed("2147483647")

    assert out_a == out_b, (
        "get_legal_actions() ordering differs between PYTHONHASHSEED=0 and "
        "PYTHONHASHSEED=1 -- hash-seed dependence regressed"
    )
    assert out_a == out_c, (
        "get_legal_actions() ordering differs between PYTHONHASHSEED=0 and "
        "PYTHONHASHSEED=2147483647 -- hash-seed dependence regressed"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
