"""
tests/test_action_abstraction.py

Tier 1-2 tests for the v3.1 semantic action abstraction layer
(cfr/src/action_abstraction.py).

Covers:
- NUM_ABSTRACT_ACTIONS_2P in contract bound [30, 40].
- Registry shape and uniqueness of class names.
- No-orphan property: every legal concrete action maps to some abstract class.
- Round-trip idempotency: for every set bit in the abstract mask, `unabstract`
  returns a concrete action that re-maps to the same abstract index.
- Determinism: same (abstract_idx, legal, state, seed) always returns the same
  concrete action.
- Information-flow: abstract classes with multiple concrete candidates produce
  varying concrete outputs across a population of random states.
- Orphan-raises: `unabstract` raises ValueError on unset bits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.action_abstraction import (
    NUM_ABSTRACT_ACTIONS_2P,
    abstract_action_name,
    abstract_action_semantics,
    abstract_actions,
    unabstract,
)
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionCallCambia,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    ActionSnapOwn,
    CardBucket,
)

# ---------------------------------------------------------------------------
# Minimal AgentState stub
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
    own_ages: Optional[List[int]] = None,
    current_turn: int = 5,
) -> _FakeAgentState:
    """Build a _FakeAgentState with the given slot contents."""
    own_hand: Dict[int, _KnownCardInfo] = {}
    for i, b in enumerate(own_buckets):
        if b is None:
            continue
        age = 0 if own_ages is None else own_ages[i]
        own_hand[i] = _KnownCardInfo(bucket=b, last_seen_turn=age)
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
# Random state / legal-action factories
# ---------------------------------------------------------------------------

_ALL_BUCKETS = [
    CardBucket.ZERO,
    CardBucket.NEG_KING,
    CardBucket.ACE,
    CardBucket.LOW_NUM,
    CardBucket.MID_NUM,
    CardBucket.PEEK_SELF,
    CardBucket.PEEK_OTHER,
    CardBucket.SWAP_BLIND,
    CardBucket.HIGH_KING,
    CardBucket.UNKNOWN,
]


def _random_agent(rng: random.Random, hand_size: int = 4) -> _FakeAgentState:
    own = [
        rng.choice(_ALL_BUCKETS) if rng.random() < 0.8 else None for _ in range(hand_size)
    ]
    opp = [
        rng.choice(_ALL_BUCKETS) if rng.random() < 0.5 else None for _ in range(hand_size)
    ]
    ages = [rng.randint(0, 10) for _ in range(hand_size)]
    current = rng.randint(5, 20)
    return _mk_agent(own, opp, own_ages=ages, current_turn=current)


def _random_legal_actions(
    rng: random.Random,
    hand_size: int = 4,
) -> List:
    """
    Build a population of plausible concrete actions spanning every abstract
    class. Not tied to a real game phase; tests only need coverage of the
    action tags.
    """
    actions = [
        ActionDrawStockpile(),
        ActionDrawDiscard(),
        ActionCallCambia(),
        ActionDiscard(use_ability=False),
        ActionDiscard(use_ability=True),
        ActionPassSnap(),
    ]
    actions.extend(ActionReplace(target_hand_index=i) for i in range(hand_size))
    actions.extend(
        ActionAbilityPeekOwnSelect(target_hand_index=i) for i in range(hand_size)
    )
    actions.extend(
        ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
        for i in range(hand_size)
    )
    for i in range(hand_size):
        for j in range(hand_size):
            actions.append(
                ActionAbilityBlindSwapSelect(own_hand_index=i, opponent_hand_index=j)
            )
            actions.append(
                ActionAbilityKingLookSelect(own_hand_index=i, opponent_hand_index=j)
            )
    actions.append(ActionAbilityKingSwapDecision(perform_swap=True))
    actions.append(ActionAbilityKingSwapDecision(perform_swap=False))
    actions.extend(ActionSnapOwn(own_card_hand_index=i) for i in range(hand_size))
    actions.extend(
        ActionSnapOpponent(opponent_target_hand_index=i) for i in range(hand_size)
    )
    for i in range(hand_size):
        for j in range(hand_size):
            actions.append(
                ActionSnapOpponentMove(
                    own_card_to_move_hand_index=i,
                    target_empty_slot_index=j,
                )
            )
    rng.shuffle(actions)
    return actions


# ---------------------------------------------------------------------------
# Bound + registry properties
# ---------------------------------------------------------------------------


def test_num_abstract_actions_in_contract_bound():
    assert 30 <= NUM_ABSTRACT_ACTIONS_2P <= 40


def test_registry_shape_matches_constant():
    assert len(abstract_action_semantics) == NUM_ABSTRACT_ACTIONS_2P
    names = [entry["name"] for entry in abstract_action_semantics]
    assert len(set(names)) == len(names), "Abstract class names must be unique"
    for entry in abstract_action_semantics:
        assert "rule" in entry and isinstance(entry["rule"], str) and entry["rule"]


def test_abstract_action_name_roundtrip():
    for i in range(NUM_ABSTRACT_ACTIONS_2P):
        assert abstract_action_name(i) == abstract_action_semantics[i]["name"]


# ---------------------------------------------------------------------------
# Mask shape and dtype
# ---------------------------------------------------------------------------


def test_abstract_actions_mask_shape_and_dtype():
    rng = random.Random(0)
    agent = _random_agent(rng)
    actions = _random_legal_actions(rng)
    mask = abstract_actions(actions, agent)
    assert mask.shape == (NUM_ABSTRACT_ACTIONS_2P,)
    assert mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# No-orphan property
# ---------------------------------------------------------------------------


def test_no_orphan_concrete_actions():
    """Every legal concrete action must map to some abstract class."""
    from src.action_abstraction import _concrete_to_abstract  # type: ignore

    rng = random.Random(17)
    for trial in range(100):
        agent = _random_agent(rng)
        actions = _random_legal_actions(rng)
        for a in actions:
            idx = _concrete_to_abstract(a, agent)
            assert idx is not None, (
                f"Trial {trial}: action {a!r} mapped to None; update abstraction "
                f"registry or _concrete_to_abstract to handle this tag."
            )
            assert 0 <= idx < NUM_ABSTRACT_ACTIONS_2P


# ---------------------------------------------------------------------------
# Round-trip idempotency
# ---------------------------------------------------------------------------


def test_abstract_unabstract_roundtrip_idempotent():
    """For every set bit in the mask, unabstract returns an action that
    re-maps to that same abstract index.
    """
    from src.action_abstraction import _concrete_to_abstract  # type: ignore

    rng = random.Random(31)
    for trial in range(100):
        agent = _random_agent(rng)
        actions = _random_legal_actions(rng)
        mask = abstract_actions(actions, agent)
        for idx in np.flatnonzero(mask):
            concrete = unabstract(int(idx), actions, agent, seed=trial * 7 + int(idx))
            remapped = _concrete_to_abstract(concrete, agent)
            assert remapped == int(idx), (
                f"Trial {trial}: abstract {idx} -> concrete {concrete!r} re-mapped "
                f"to {remapped} (expected {idx})"
            )


# ---------------------------------------------------------------------------
# Determinism given seed
# ---------------------------------------------------------------------------


def test_unabstract_deterministic_given_seed():
    rng = random.Random(5)
    agent = _random_agent(rng)
    actions = _random_legal_actions(rng)
    mask = abstract_actions(actions, agent)
    for idx in np.flatnonzero(mask):
        a1 = unabstract(int(idx), actions, agent, seed=123)
        a2 = unabstract(int(idx), actions, agent, seed=123)
        assert a1 == a2


# ---------------------------------------------------------------------------
# Orphan-raise
# ---------------------------------------------------------------------------


def test_unabstract_raises_on_empty_class():
    """If no legal action maps to the requested abstract class, unabstract raises."""
    agent = _mk_agent(
        own_buckets=[CardBucket.LOW_NUM, None, None, None],
        opp_buckets=[None, None, None, None],
    )
    actions = [ActionDrawStockpile()]  # only draw_stockpile is legal
    # call_cambia (index = lookup) is not present -> expect ValueError
    cambia_idx = next(
        i for i, e in enumerate(abstract_action_semantics) if e["name"] == "call_cambia"
    )
    with pytest.raises(ValueError):
        unabstract(cambia_idx, actions, agent, seed=0)


def test_unabstract_raises_on_bad_index():
    agent = _mk_agent(own_buckets=[], opp_buckets=[])
    with pytest.raises(ValueError):
        unabstract(-1, [], agent, seed=0)
    with pytest.raises(ValueError):
        unabstract(NUM_ABSTRACT_ACTIONS_2P, [], agent, seed=0)


# ---------------------------------------------------------------------------
# Information-flow: varying state -> varying concrete outputs
# ---------------------------------------------------------------------------

_MULTI_CANDIDATE_CLASSES = [
    "snap_own",
    "snap_opp",
    "snap_opp_move",
]


def test_unabstract_varies_with_state_for_multi_candidate_classes():
    """
    For abstract classes that typically have >1 concrete candidate, confirm
    that sweeping the state population produces at least 2 distinct concrete
    outputs. This is the information-flow check from spec Section 6.
    """
    rng = random.Random(99)
    observed = {name: set() for name in _MULTI_CANDIDATE_CLASSES}
    name_to_idx = {e["name"]: i for i, e in enumerate(abstract_action_semantics)}

    for trial in range(100):
        agent = _random_agent(rng)
        actions = _random_legal_actions(rng)
        mask = abstract_actions(actions, agent)
        for name in _MULTI_CANDIDATE_CLASSES:
            idx = name_to_idx[name]
            if not mask[idx]:
                continue
            concrete = unabstract(idx, actions, agent, seed=trial)
            observed[name].add(repr(concrete))

    for name, concretes in observed.items():
        assert len(concretes) >= 2, (
            f"Abstract class {name!r} produced only {len(concretes)} distinct "
            "concrete actions across 100 states; expected >= 2 (information-flow "
            "collapse detected: split this class or adjust unabstract rule)."
        )


# ---------------------------------------------------------------------------
# Targeted collapse rules (sanity)
# ---------------------------------------------------------------------------


def test_replace_slot_quantile_mapping():
    """Replace by own slot bucket maps to the correct quantile class."""
    agent = _mk_agent(
        own_buckets=[
            CardBucket.LOW_NUM,
            CardBucket.MID_NUM,
            CardBucket.HIGH_KING,
            CardBucket.UNKNOWN,
        ],
        opp_buckets=[None, None, None, None],
    )
    actions = [ActionReplace(target_hand_index=i) for i in range(4)]
    mask = abstract_actions(actions, agent)
    expected = {
        "replace_slot_low",
        "replace_slot_mid",
        "replace_slot_high",
        "replace_slot_unknown",
    }
    name_to_idx = {e["name"]: i for i, e in enumerate(abstract_action_semantics)}
    for name in expected:
        assert mask[name_to_idx[name]], f"Expected mask bit for {name}"
    # Nothing outside expected should be set.
    lit = {abstract_action_name(int(i)) for i in np.flatnonzero(mask)}
    assert lit == expected


def test_discard_ability_split():
    """ActionDiscard(use_ability=True) vs False land in different abstract classes."""
    agent = _mk_agent(own_buckets=[], opp_buckets=[])
    actions = [
        ActionDiscard(use_ability=False),
        ActionDiscard(use_ability=True),
    ]
    mask = abstract_actions(actions, agent)
    name_to_idx = {e["name"]: i for i, e in enumerate(abstract_action_semantics)}
    assert mask[name_to_idx["discard_drawn_no_ability"]]
    assert mask[name_to_idx["discard_drawn_with_ability"]]


def test_peek_own_age_split():
    """Peek-own classes split by observation age on a known slot."""
    agent = _mk_agent(
        own_buckets=[CardBucket.LOW_NUM, CardBucket.LOW_NUM, CardBucket.UNKNOWN, None],
        opp_buckets=[None, None, None, None],
        own_ages=[
            5,
            0,
            0,
            0,
        ],  # slot0 recently_seen=5 (current turn), slot1 last_seen=0 (old)
        current_turn=5,
    )
    actions = [
        ActionAbilityPeekOwnSelect(target_hand_index=0),  # age 0 -> recent
        ActionAbilityPeekOwnSelect(target_hand_index=1),  # age 5 -> stale
        ActionAbilityPeekOwnSelect(target_hand_index=2),  # unknown
    ]
    mask = abstract_actions(actions, agent)
    name_to_idx = {e["name"]: i for i, e in enumerate(abstract_action_semantics)}
    assert mask[name_to_idx["peek_own_known_recent"]]
    assert mask[name_to_idx["peek_own_known_stale"]]
    assert mask[name_to_idx["peek_own_unknown"]]
