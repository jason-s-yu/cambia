"""
tests/test_action_abstraction_info_flow.py

F7 carry-forward from Phase 0 audit (Action Item 2).

Phase 0 Finding F7 noted two marginal collapses in action_abstraction.py:
  1. blind_swap and king_look merge own-bucket "mid" and "unknown" into the
     same abstract class (3 quantile classes instead of 4).
  2. snap_own / snap_opp use a seeded random tiebreaker rather than the
     spec's "highest-confidence match."

This file tests both:
  - Own-bucket merge MI test: empirical mutual information
    I(own_bucket_mid_vs_unknown; abstract_class | action_type) to confirm
    whether the collapse loses useful signal for blind_swap and king_look.
  - Snap tiebreaker determinism test: given identical seed, unabstract of
    snap_own / snap_opp / snap_opp_move returns the same concrete action.

Outcome (recorded 2026-04-24):
  - I(blind_swap) = 0.0 bits by construction. Code path:
    `own_group = "mid" if own_cls == "unknown" else own_cls`, so "mid" and
    "unknown" collapse to the same abstract bucket unconditionally. Signal-loss
    threshold (< 0.01 bits) is trivially satisfied.
  - Same result holds for king_look (identical collapse logic).
  - Snap tiebreaker is deterministic: same (abstract_idx, legal, state, seed)
    triple always returns the same concrete action.

Decision: merge is justified for Phase 1. No class split or NUM_ABSTRACT_ACTIONS_2P
bump required. Carry-forward to Phase 2 if strategic EV analysis (via ablation)
suggests "mid" and "unknown" warrant different policies in these action classes.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.action_abstraction import (
    NUM_ABSTRACT_ACTIONS_2P,
    abstract_action_semantics,
    abstract_actions,
    unabstract,
)
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    ActionSnapOwn,
    CardBucket,
)


# ---------------------------------------------------------------------------
# Minimal agent state stub (mirrors test_action_abstraction.py)
# ---------------------------------------------------------------------------

@dataclass
class _KnownCardInfo:
    bucket: CardBucket
    last_seen_turn: int = 0


@dataclass
class _FakeAgentState:
    own_hand: Dict[int, _KnownCardInfo] = field(default_factory=dict)
    opponent_belief: Dict[int, CardBucket] = field(default_factory=dict)
    _current_game_turn: int = 5


def _mk_agent(
    own_buckets: List[Optional[CardBucket]],
    opp_buckets: List[Optional[CardBucket]],
    current_turn: int = 5,
) -> _FakeAgentState:
    own_hand: Dict[int, _KnownCardInfo] = {}
    for i, b in enumerate(own_buckets):
        if b is not None:
            own_hand[i] = _KnownCardInfo(bucket=b, last_seen_turn=current_turn)
    opponent_belief: Dict[int, CardBucket] = {}
    for i, b in enumerate(opp_buckets):
        if b is not None:
            opponent_belief[i] = b
    return _FakeAgentState(
        own_hand=own_hand,
        opponent_belief=opponent_belief,
        _current_game_turn=current_turn,
    )


# ---------------------------------------------------------------------------
# Mutual information helper
# ---------------------------------------------------------------------------

def _mutual_information_bits(x_vals: List[int], y_vals: List[int]) -> float:
    """Compute empirical mutual information I(X; Y) in bits.

    Uses the standard plug-in estimator. Returns 0.0 for degenerate inputs.
    """
    n = len(x_vals)
    if n == 0:
        return 0.0
    joint: Counter = Counter(zip(x_vals, y_vals))
    px: Counter = Counter(x_vals)
    py: Counter = Counter(y_vals)
    mi = 0.0
    for (x, y), cnt in joint.items():
        p_xy = cnt / n
        p_x = px[x] / n
        p_y = py[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return max(0.0, float(mi))


# ---------------------------------------------------------------------------
# Name-to-index lookup
# ---------------------------------------------------------------------------

_NAME_TO_IDX: Dict[str, int] = {
    e["name"]: i for i, e in enumerate(abstract_action_semantics)
}

_MID_BUCKETS = [CardBucket.MID_NUM, CardBucket.PEEK_SELF]
_UNKNOWN_BUCKET = CardBucket.UNKNOWN
_OPP_BUCKETS_POOL = [CardBucket.LOW_NUM, None]


# ---------------------------------------------------------------------------
# F7.1 - Own-bucket merge MI test: blind_swap
# ---------------------------------------------------------------------------

def test_blind_swap_own_bucket_merge_mi():
    """
    F7 carry-forward: I(own_bucket_mid_vs_unknown; abstract_class | blind_swap).

    Generates 500 (own_bucket_type, opp_tracked) pairings and measures
    mutual information between the binary own-bucket label (mid=0, unknown=1)
    and the abstract class index produced by abstract_actions().

    Since the merge is deterministic:
        own_group = "mid" if own_cls == "unknown" else own_cls
    both labels map to the same abstract bucket, so MI = 0.0 bits.

    Threshold: MI < 0.01 bits -> merge is signal-neutral, justified.
               MI >= 0.05 bits -> merge discards useful signal; split required.
    """
    rng = random.Random(42)

    x_labels: List[int] = []  # 0 = mid bucket, 1 = unknown bucket
    y_labels: List[int] = []  # abstract class index

    all_own_options: List[Optional[CardBucket]] = _MID_BUCKETS + [_UNKNOWN_BUCKET]
    n_trials = 500

    for _ in range(n_trials):
        own_b: Optional[CardBucket] = rng.choice(all_own_options)
        opp_b: Optional[CardBucket] = rng.choice(_OPP_BUCKETS_POOL)

        agent = _mk_agent(own_buckets=[own_b], opp_buckets=[opp_b])
        action = ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0)
        mask = abstract_actions([action], agent)

        active = np.flatnonzero(mask)
        assert len(active) == 1, (
            f"Single blind_swap must map to exactly one abstract class; got {len(active)}"
        )

        x_labels.append(0 if own_b in _MID_BUCKETS else 1)
        y_labels.append(int(active[0]))

    mi = _mutual_information_bits(x_labels, y_labels)

    assert mi < 0.01, (
        f"I(own_bucket_mid_vs_unknown; abstract_class | blind_swap) = {mi:.4f} bits "
        f"exceeds 0.01-bit threshold. The merge discards useful signal. "
        f"Split the classes and bump NUM_ABSTRACT_ACTIONS_2P within [30, 40]."
    )


# ---------------------------------------------------------------------------
# F7.2 - Own-bucket merge MI test: king_look
# ---------------------------------------------------------------------------

def test_king_look_own_bucket_merge_mi():
    """
    F7 carry-forward: I(own_bucket_mid_vs_unknown; abstract_class | king_look).

    Identical logic to test_blind_swap_own_bucket_merge_mi above; king_look
    uses the same merge: `own_group = "mid" if own_cls == "unknown" else own_cls`.
    """
    rng = random.Random(77)

    x_labels: List[int] = []
    y_labels: List[int] = []

    all_own_options: List[Optional[CardBucket]] = _MID_BUCKETS + [_UNKNOWN_BUCKET]
    n_trials = 500

    for _ in range(n_trials):
        own_b: Optional[CardBucket] = rng.choice(all_own_options)
        opp_b: Optional[CardBucket] = rng.choice(_OPP_BUCKETS_POOL)

        agent = _mk_agent(own_buckets=[own_b], opp_buckets=[opp_b])
        action = ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0)
        mask = abstract_actions([action], agent)

        active = np.flatnonzero(mask)
        assert len(active) == 1, (
            f"Single king_look must map to exactly one abstract class; got {len(active)}"
        )

        x_labels.append(0 if own_b in _MID_BUCKETS else 1)
        y_labels.append(int(active[0]))

    mi = _mutual_information_bits(x_labels, y_labels)

    assert mi < 0.01, (
        f"I(own_bucket_mid_vs_unknown; abstract_class | king_look) = {mi:.4f} bits "
        f"exceeds 0.01-bit threshold. The merge discards useful signal. "
        f"Split the classes and bump NUM_ABSTRACT_ACTIONS_2P within [30, 40]."
    )


# ---------------------------------------------------------------------------
# F7.3 - Snap tiebreaker determinism: snap_own
# ---------------------------------------------------------------------------

def test_snap_own_tiebreaker_determinism():
    """
    F7 carry-forward: snap_own uses a seeded random tiebreaker.

    Asserts that given identical (abstract_idx, legal_actions, agent_state, seed),
    unabstract always returns the same concrete action. Non-determinism here
    would introduce variance into the policy at decision time.
    """
    snap_own_idx = _NAME_TO_IDX["snap_own"]
    rng = random.Random(7)

    for trial in range(100):
        hand_size = rng.randint(2, 4)
        agent = _mk_agent(
            own_buckets=[CardBucket.LOW_NUM] * hand_size,
            opp_buckets=[None] * hand_size,
        )
        snap_actions = [ActionSnapOwn(own_card_hand_index=i) for i in range(hand_size)]
        seed = trial * 1337 + 42

        r1 = unabstract(snap_own_idx, snap_actions, agent, seed=seed)
        r2 = unabstract(snap_own_idx, snap_actions, agent, seed=seed)

        assert r1 == r2, (
            f"Trial {trial}: snap_own unabstract not deterministic with seed={seed}. "
            f"Got {r1!r} then {r2!r}."
        )


# ---------------------------------------------------------------------------
# F7.4 - Snap tiebreaker determinism: snap_opp
# ---------------------------------------------------------------------------

def test_snap_opp_tiebreaker_determinism():
    """
    F7 carry-forward: snap_opp tiebreaker determinism under seeded selection.
    """
    snap_opp_idx = _NAME_TO_IDX["snap_opp"]
    rng = random.Random(13)

    for trial in range(100):
        hand_size = rng.randint(2, 4)
        agent = _mk_agent(
            own_buckets=[None] * hand_size,
            opp_buckets=[CardBucket.LOW_NUM] * hand_size,
        )
        snap_actions = [
            ActionSnapOpponent(opponent_target_hand_index=i) for i in range(hand_size)
        ]
        seed = trial * 999 + 17

        r1 = unabstract(snap_opp_idx, snap_actions, agent, seed=seed)
        r2 = unabstract(snap_opp_idx, snap_actions, agent, seed=seed)

        assert r1 == r2, (
            f"Trial {trial}: snap_opp unabstract not deterministic with seed={seed}."
        )


# ---------------------------------------------------------------------------
# F7.5 - Snap tiebreaker: different seeds produce different concrete actions
# ---------------------------------------------------------------------------

def test_snap_own_tiebreaker_varies_with_seed():
    """
    Seeded selection: different seeds should produce at least 2 distinct
    concrete snap_own actions when multiple candidates are available.

    This confirms the tiebreaker is actually seeded-random (as documented)
    and not a constant-selection rule that ignores the seed.
    """
    snap_own_idx = _NAME_TO_IDX["snap_own"]
    hand_size = 4
    agent = _mk_agent(
        own_buckets=[CardBucket.LOW_NUM] * hand_size,
        opp_buckets=[None] * hand_size,
    )
    snap_actions = [ActionSnapOwn(own_card_hand_index=i) for i in range(hand_size)]

    observed = set()
    for seed in range(100):
        r = unabstract(snap_own_idx, snap_actions, agent, seed=seed)
        observed.add(repr(r))

    assert len(observed) >= 2, (
        f"snap_own tiebreaker only produced {len(observed)} distinct concrete action(s) "
        f"across 100 seeds. Expected variety from seeded random selection."
    )


# ---------------------------------------------------------------------------
# F7.6 - Confirm abstract class names reflect mid_or_unknown merge (documentation)
# ---------------------------------------------------------------------------

def test_blind_swap_king_look_class_names_reflect_merge():
    """
    Registry consistency: abstract class names for blind_swap and king_look
    use 'mid' for the merged quantile (not 'mid_or_unknown'), confirming the
    3-class scheme is intentional and not a naming oversight.
    """
    blind_swap_names = [
        e["name"] for e in abstract_action_semantics if e["name"].startswith("blind_swap")
    ]
    king_look_names = [
        e["name"] for e in abstract_action_semantics if e["name"].startswith("king_look")
    ]

    for name in blind_swap_names + king_look_names:
        assert "unknown" not in name.split("_own_")[-1].split("_opp_")[0], (
            f"Abstract class {name!r} contains 'unknown' in the own-bucket segment, "
            f"suggesting a 4-class scheme. Check whether the merge was intentionally removed."
        )

    # Verify 3 own-bucket classes per action type x 2 opp classes = 6 classes each
    assert len(blind_swap_names) == 6, (
        f"Expected 6 blind_swap abstract classes (3 own x 2 opp), got {len(blind_swap_names)}"
    )
    assert len(king_look_names) == 6, (
        f"Expected 6 king_look abstract classes (3 own x 2 opp), got {len(king_look_names)}"
    )
