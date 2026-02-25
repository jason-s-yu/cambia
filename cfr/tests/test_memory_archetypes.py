"""
tests/test_memory_archetypes.py

Tests for memory archetype behavior (Perfect, Decaying, HumanLike) in AgentState.
"""
import random

import pytest

from src.agent_state import AgentState, KnownCardInfo
from src.constants import CardBucket, EpistemicTag, EP_PBS_MAX_ACTIVE_MASK, bucket_saliency


def _make_agent(memory_archetype="perfect", memory_decay_lambda=0.1, memory_capacity=3):
    """Build a minimal AgentState with stub config and the given memory archetype."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.cambia_rules.penaltyDrawCount = 2
    config.cambia_rules.use_jokers = 2

    # Provide deep_cfr mock so __post_init__ can read memory fields.
    deep_cfg = MagicMock()
    deep_cfg.memory_archetype = memory_archetype
    deep_cfg.memory_decay_lambda = memory_decay_lambda
    deep_cfg.memory_capacity = memory_capacity
    config.deep_cfr = deep_cfg

    agent = AgentState(
        player_id=0,
        opponent_id=1,
        memory_level=0,
        time_decay_turns=3,
        initial_hand_size=4,
        config=config,
    )
    return agent


def _add_priv_own_slot(agent, slot_idx, bucket_val):
    """Directly add a PrivOwn slot to agent's EP-PBS state."""
    agent.slot_tags[slot_idx] = EpistemicTag.PRIV_OWN
    agent.slot_buckets[slot_idx] = bucket_val
    if slot_idx not in agent.own_active_mask:
        agent.own_active_mask.append(slot_idx)


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------


def test_config_fields_exist():
    """Verify the config stub has all three memory archetype fields."""
    from src.config import DeepCfrConfig

    cfg = DeepCfrConfig()
    assert hasattr(cfg, "memory_archetype"), "DeepCfrConfig missing memory_archetype"
    assert hasattr(cfg, "memory_decay_lambda"), "DeepCfrConfig missing memory_decay_lambda"
    assert hasattr(cfg, "memory_capacity"), "DeepCfrConfig missing memory_capacity"
    assert cfg.memory_archetype == "perfect"
    assert cfg.memory_decay_lambda == 0.1
    assert cfg.memory_capacity == 3


# ---------------------------------------------------------------------------
# Perfect archetype tests
# ---------------------------------------------------------------------------


def test_perfect_no_decay():
    """MemoryPerfect: apply_memory_decay never removes any observations."""
    agent = _make_agent(memory_archetype="perfect")
    # Add 3 PrivOwn slots.
    for slot in [0, 1, 2]:
        _add_priv_own_slot(agent, slot, CardBucket.HIGH_KING.value)

    for _ in range(50):
        agent.apply_memory_decay()

    assert len(agent.own_active_mask) == 3
    for slot in [0, 1, 2]:
        assert agent.slot_tags[slot] == EpistemicTag.PRIV_OWN, (
            f"Slot {slot} should remain PRIV_OWN for MemoryPerfect"
        )


# ---------------------------------------------------------------------------
# Decaying archetype tests
# ---------------------------------------------------------------------------


def test_decaying_eventual_loss():
    """MemoryDecaying with high lambda: all slots decay over many turns."""
    agent = _make_agent(memory_archetype="decaying", memory_decay_lambda=10.0)
    _add_priv_own_slot(agent, 0, CardBucket.HIGH_KING.value)
    _add_priv_own_slot(agent, 1, CardBucket.ACE.value)

    rng = random.Random(42)
    agent.apply_memory_decay(rng=rng)

    # With lambda=10, p ≈ 1.0 — should decay everything in one step.
    assert len(agent.own_active_mask) == 0, (
        "Expected all slots decayed with lambda=10"
    )
    for slot in [0, 1]:
        assert agent.slot_tags[slot] == EpistemicTag.UNK


def test_decaying_low_lambda_retains_some():
    """MemoryDecaying with lambda=0: no decay (p=0)."""
    agent = _make_agent(memory_archetype="decaying", memory_decay_lambda=0.0)
    _add_priv_own_slot(agent, 0, CardBucket.HIGH_KING.value)
    _add_priv_own_slot(agent, 1, CardBucket.ACE.value)

    rng = random.Random(42)
    for _ in range(10):
        agent.apply_memory_decay(rng=rng)

    # lambda=0 → p=0, nothing decays.
    assert len(agent.own_active_mask) == 2, "lambda=0 should retain all slots"


def test_decaying_deterministic_seed():
    """Same seed → same decay pattern."""

    def run(seed):
        agent = _make_agent(memory_archetype="decaying", memory_decay_lambda=0.5)
        for slot in [0, 1, 2]:
            _add_priv_own_slot(agent, slot, CardBucket.MID_NUM.value)
        rng = random.Random(seed)
        for _ in range(5):
            agent.apply_memory_decay(rng=rng)
        return list(agent.own_active_mask), list(agent.slot_tags[:3])

    mask1, tags1 = run(1234)
    mask2, tags2 = run(1234)
    assert mask1 == mask2, "Deterministic seed must produce identical masks"
    assert tags1 == tags2, "Deterministic seed must produce identical tags"


# ---------------------------------------------------------------------------
# HumanLike archetype tests
# ---------------------------------------------------------------------------


def test_human_like_capacity_limit():
    """MemoryHumanLike: after having more slots than capacity, mask is trimmed."""
    agent = _make_agent(memory_archetype="human_like", memory_capacity=2)
    # Manually insert 3 slots.
    _add_priv_own_slot(agent, 0, CardBucket.HIGH_KING.value)  # saliency = 8.5
    _add_priv_own_slot(agent, 1, CardBucket.ACE.value)         # saliency = 3.5
    _add_priv_own_slot(agent, 2, CardBucket.ZERO.value)        # saliency = 4.5

    agent.apply_memory_decay()

    assert len(agent.own_active_mask) == 2, (
        f"Expected mask len 2, got {len(agent.own_active_mask)}"
    )


def test_human_like_saliency_eviction():
    """MemoryHumanLike: lowest-saliency slot is evicted first."""
    agent = _make_agent(memory_archetype="human_like", memory_capacity=2)
    # Slot 0: BucketAce, saliency = |1 - 4.5| = 3.5 — LOWEST
    # Slot 1: BucketHighKing, saliency = |13 - 4.5| = 8.5
    # Slot 2: BucketZero, saliency = |0 - 4.5| = 4.5
    _add_priv_own_slot(agent, 0, CardBucket.ACE.value)
    _add_priv_own_slot(agent, 1, CardBucket.HIGH_KING.value)
    _add_priv_own_slot(agent, 2, CardBucket.ZERO.value)

    agent.apply_memory_decay()

    # Slot 0 (ACE, lowest saliency) should be evicted.
    assert agent.slot_tags[0] == EpistemicTag.UNK, (
        "Slot 0 (BucketAce, lowest saliency 3.5) should have been evicted"
    )
    assert agent.slot_tags[1] == EpistemicTag.PRIV_OWN, "Slot 1 (HIGH_KING) should remain"
    assert agent.slot_tags[2] == EpistemicTag.PRIV_OWN, "Slot 2 (ZERO) should remain"
    assert 0 not in agent.own_active_mask, "Evicted slot 0 should not be in mask"


def test_human_like_no_eviction_when_at_capacity():
    """MemoryHumanLike: no eviction if mask is already at or below capacity."""
    agent = _make_agent(memory_archetype="human_like", memory_capacity=3)
    _add_priv_own_slot(agent, 0, CardBucket.HIGH_KING.value)
    _add_priv_own_slot(agent, 1, CardBucket.ACE.value)

    agent.apply_memory_decay()

    # Only 2 slots, capacity=3 — nothing should be evicted.
    assert len(agent.own_active_mask) == 2
    assert agent.slot_tags[0] == EpistemicTag.PRIV_OWN
    assert agent.slot_tags[1] == EpistemicTag.PRIV_OWN
