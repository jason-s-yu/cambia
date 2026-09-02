"""Tests for EP-PBS encoding."""

import numpy as np
import pytest
from src.encoding import (
    EP_PBS_INPUT_DIM,
    EpistemicTag,
    encode_infoset_eppbs,
    bucket_saliency,
    EP_PBS_MAX_ACTIVE_MASK,
)

try:
    from src.ffi.bridge import GoEngine, GoAgentState

    _HAS_GO = True
except Exception:  # pragma: no cover - libcambia not built
    _HAS_GO = False

skipgo = pytest.mark.skipif(not _HAS_GO, reason="libcambia.so not available")

# EP-PBS slot layout: own hand is slots 0-5, the opponent's is 6-11.
_OPP_SLOTS_START = 6

# Seeds swept for a King look followed by a swap. The driver forces the King line
# wherever it is legal, so a seed contributes as soon as a King reaches a discard with
# its ability live; seeds that never deal one are skipped, not failed.
_KING_SWAP_SEEDS = tuple(range(1, 61))


def _play_to_king_swap(seed: int):
    """Drive both engines in lockstep until the first King look followed by a swap.

    Returns ``(go_agents, py_agents, actor, own_idx, opp_idx, decision_ctx,
    drawn_bucket)`` at the state just after the swap, or None if the game ended first.
    Both engines are fed the same action indices, and the Python agents are updated
    through the production observation builder so the King look actually reveals cards.
    """
    from src.cfr.worker import _create_observation, _filter_observation
    from src.constants import (
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionPassSnap,
    )
    from src.encoding import action_to_index, encode_action_mask, NUM_ACTIONS
    from tests.test_cross_engine_samples import (
        _setup_python_game_matching_go,
        _is_snap_only,
        _PASS_SNAP_IDX,
        _TEST_RULES,
    )
    from tests.test_cross_validation import _build_py_agents, _make_config

    go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
    try:
        py_state = _setup_python_game_matching_go(seed)
        go_agents = [GoAgentState(go_engine, i) for i in range(2)]
        py_agents = _build_py_agents(py_state, _make_config())
        snap_indices = set(range(_PASS_SNAP_IDX, NUM_ACTIONS))

        looked = None  # (own_idx, opp_idx) once a King look has resolved
        for _ in range(300):
            if go_engine.is_terminal() or py_state.is_terminal():
                return None

            go_actions = set(np.where(go_engine.legal_actions_mask() > 0)[0].tolist())
            py_legal = py_state.get_legal_actions()
            py_actions = set(
                np.where(encode_action_mask(list(py_legal)).astype(np.uint8) > 0)[
                    0
                ].tolist()
            )

            # Snap phases resolve independently in the two engines; pass through them
            # without touching the agents, exactly as the sibling parity tests do.
            if _is_snap_only(go_actions):
                go_engine.apply_action(_PASS_SNAP_IDX)
                if py_state.snap_phase_active:
                    py_state.apply_action(ActionPassSnap())
                continue
            if py_state.snap_phase_active:
                py_state.apply_action(ActionPassSnap())
                continue

            common = sorted((go_actions & py_actions) - snap_indices)
            if not common:
                return None

            py_action = None
            for candidate in _prefer_king_line(common, py_legal):
                for a in py_legal:
                    try:
                        if action_to_index(a) == candidate:
                            py_action = a
                            break
                    except Exception:
                        continue
                if py_action is not None:
                    action_idx = candidate
                    break
            if py_action is None:
                return None

            actor = py_state.get_acting_player()
            pre_swap = None
            if (
                isinstance(py_action, ActionAbilityKingSwapDecision)
                and py_action.perform_swap
                and py_state.pending_action_data
            ):
                pad = py_state.pending_action_data
                if "own_idx" in pad and "opp_idx" in pad:
                    pre_swap = (pad["own_idx"], pad["opp_idx"])

            before = _py_fingerprint(py_state)
            had_pending = py_state.pending_action is not None
            go_engine.apply_action(action_idx)
            py_state.apply_action(py_action)
            # A declined action leaves the Python engine's pending state where it was
            # while Go advances; comparing the two from there compares different games.
            if _py_fingerprint(py_state) == before:
                return None
            if had_pending and repr(py_state.pending_action) == before[1]:
                return None
            go_engine.update_both(go_agents[0], go_agents[1])
            obs = _create_observation(
                None,
                py_action,
                py_state,
                actor,
                py_state.snap_results_log,
                king_swap_indices=pre_swap,
            )
            if obs is None:
                return None
            for pid, pa in enumerate(py_agents):
                pa.update(_filter_observation(obs, pid))

            if isinstance(py_action, ActionAbilityKingLookSelect):
                looked = (py_action.own_hand_index, py_action.opponent_hand_index)
            elif (
                isinstance(py_action, ActionAbilityKingSwapDecision)
                and py_action.perform_swap
                and looked is not None
            ):
                own_idx, opp_idx = pre_swap if pre_swap is not None else looked
                return (
                    go_agents,
                    py_agents,
                    actor,
                    own_idx,
                    opp_idx,
                    go_engine.decision_ctx(),
                    go_engine.get_drawn_card_bucket(),
                )
        return None
    finally:
        go_engine.close()


# Seeds swept for a successful opponent snap followed by its RULES.md 5 fill. The driver
# takes the snap line wherever it is legal; a seed that never reaches one is skipped.
_SNAP_FILL_SEEDS = tuple(range(1, 121))

_PASS_SNAP = 97
_SNAP_OPP_MIN, _SNAP_OPP_MAX = 104, 110
_SNAP_MOVE_MIN, _SNAP_MOVE_MAX = 110, 146


def _hands_agree(go_engine, py_state) -> bool:
    """True when both engines hold the same number of cards in both hands."""
    try:
        return all(
            len(go_engine.get_hand_indices(seat)) == py_state.get_player_card_count(seat)
            for seat in range(2)
        )
    except Exception:
        return False


def _slot_tags_from_encoding(enc):
    """Read the 12 one-hot slot tags back out of a 224-dim EP-PBS encoding."""
    return [int(np.argmax(enc[40 + 4 * s : 44 + 4 * s])) for s in range(12)]


def _play_to_snap_fill(seed: int):
    """Drive both engines to a successful opponent snap and the fill that answers it.

    Returns a list of ``(label, [go_tags_per_seat], [py_tags_per_seat])`` snapshots taken
    right after the snap (the victim's hand shrank) and right after the fill (it grew
    again), or None if the seed never reached one in lockstep. Any step that leaves the
    two engines holding different hand sizes ends the seed rather than comparing states
    that have drifted apart.
    """
    from src.cfr.worker import _create_observation, _filter_observation
    from src.constants import ActionSnapOpponent, ActionSnapOpponentMove
    from src.encoding import action_to_index, encode_action_mask
    from tests.test_cross_engine_samples import (
        _setup_python_game_matching_go,
        _TEST_RULES,
    )
    from tests.test_cross_validation import _build_py_agents, _make_config

    go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
    try:
        py_state = _setup_python_game_matching_go(seed)
        go_agents = [GoAgentState(go_engine, i) for i in range(2)]
        py_agents = _build_py_agents(py_state, _make_config())
        snapshots = []

        for _ in range(300):
            if go_engine.is_terminal() or py_state.is_terminal():
                return None
            if not _hands_agree(go_engine, py_state):
                return None

            go_actions = set(np.where(go_engine.legal_actions_mask() > 0)[0].tolist())
            py_legal = py_state.get_legal_actions()
            py_actions = set(
                np.where(encode_action_mask(list(py_legal)).astype(np.uint8) > 0)[
                    0
                ].tolist()
            )
            common = sorted(go_actions & py_actions)
            if not common:
                return None

            py_action = None
            for candidate in sorted(common, key=_snap_line_rank):
                for a in py_legal:
                    try:
                        if action_to_index(a) == candidate:
                            py_action = a
                            break
                    except Exception:
                        continue
                if py_action is not None:
                    action_idx = candidate
                    break
            if py_action is None:
                return None

            actor = py_state.get_acting_player()
            before = _py_fingerprint(py_state)
            go_engine.apply_action(action_idx)
            py_state.apply_action(py_action)
            if _py_fingerprint(py_state) == before:
                return None
            go_engine.update_both(go_agents[0], go_agents[1])
            obs = _create_observation(
                None, py_action, py_state, actor, py_state.snap_results_log
            )
            if obs is None:
                return None
            for pid, pa in enumerate(py_agents):
                pa.update(_filter_observation(obs, pid))

            if not _hands_agree(go_engine, py_state):
                return None
            if (
                isinstance(py_action, ActionSnapOpponentMove)
                and py_state.snap_results_log
            ):
                # Another snapper is still owed a decision, so the engine deliberately
                # keeps the snap log populated for the resumed snapper's observation. The
                # belief tracker then applies the same removal a second time, which is a
                # separate defect; this seed cannot measure the shift through it.
                return None

            label = None
            if isinstance(py_action, ActionSnapOpponent):
                label = "after the snap"
            elif isinstance(py_action, ActionSnapOpponentMove):
                label = "after the fill"
            if label is not None:
                ctx = go_engine.decision_ctx()
                drawn = go_engine.get_drawn_card_bucket()
                snapshots.append(
                    (
                        label,
                        [
                            _slot_tags_from_encoding(
                                go_agents[i].encode_eppbs(ctx, drawn)
                            )
                            for i in range(2)
                        ],
                        [list(py_agents[i].slot_tags) for i in range(2)],
                    )
                )
                if label == "after the fill":
                    return snapshots
        return None
    finally:
        go_engine.close()


def _snap_line_rank(idx: int) -> int:
    """Order action indices so a snap and its fill are taken ahead of anything else."""
    if _SNAP_MOVE_MIN <= idx < _SNAP_MOVE_MAX:
        return 0
    if _SNAP_OPP_MIN <= idx < _SNAP_OPP_MAX:
        return 1
    if idx == _PASS_SNAP:
        return 3
    return 2


def _py_fingerprint(py_state):
    """Enough of the Python state to tell a real apply from a declined one."""
    return (
        py_state.get_turn_number(),
        repr(py_state.pending_action),
        py_state.get_player_card_count(0),
        py_state.get_player_card_count(1),
        py_state.get_stockpile_size(),
        py_state.snap_phase_active,
    )


def _prefer_king_line(common, py_legal):
    """Order the common legal actions so the King line is taken wherever it is legal."""
    from src.constants import (
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionDiscard,
    )
    from src.encoding import action_to_index

    ranked = {}
    for a in py_legal:
        try:
            idx = action_to_index(a)
        except Exception:
            continue
        if isinstance(a, ActionAbilityKingSwapDecision):
            ranked[idx] = 0 if a.perform_swap else 3
        elif isinstance(a, ActionAbilityKingLookSelect):
            ranked[idx] = 1
        elif isinstance(a, ActionDiscard) and getattr(a, "use_ability", False):
            ranked[idx] = 2
    return sorted(common, key=lambda i: (ranked.get(i, 4), i))


class TestEPPBSEncoding:
    def test_dimension(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        assert out.shape == (EP_PBS_INPUT_DIM,)
        assert out.dtype == np.float32

    def test_public_features(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 3, 1, 2, 0, 2, drawn_card_bucket=5)
        # discard bucket 3 → out[3] = 1
        assert out[3] == 1.0
        # stock estimate 1 → out[11] = 1
        assert out[11] == 1.0
        # phase 2 → out[16] = 1
        assert out[16] == 1.0
        # context 0 → out[20] = 1
        assert out[20] == 1.0
        # cambia NONE=2 → out[28] = 1
        assert out[28] == 1.0
        # drawn bucket 5 → out[34] = 1
        assert out[34] == 1.0

    def test_slot_tags_one_hot(self):
        tags = [
            EpistemicTag.UNK,
            EpistemicTag.PRIV_OWN,
            EpistemicTag.PRIV_OPP,
            EpistemicTag.PUB,
        ] + [EpistemicTag.UNK] * 8
        buckets = [0, 2, 0, 8] + [0] * 8
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Slot 0 tag (offset 40): UNK=0
        assert out[40] == 1.0
        # Slot 1 tag (offset 44): PRIV_OWN=1
        assert out[45] == 1.0
        # Slot 2 tag (offset 48): PRIV_OPP=2
        assert out[50] == 1.0
        # Slot 3 tag (offset 52): PUB=3
        assert out[55] == 1.0

    def test_slot_identity_zeroing(self):
        """TagPrivOpp and TagUnk slots should have zero bucket encoding."""
        tags = [
            EpistemicTag.PRIV_OWN,
            EpistemicTag.PRIV_OPP,
            EpistemicTag.UNK,
            EpistemicTag.PUB,
        ] + [EpistemicTag.UNK] * 8
        buckets = [2, 5, 0, 8] + [0] * 8
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Slot 0 (PRIV_OWN, bucket 2): identity at offset 88+2
        assert out[88 + 2] == 1.0
        # Slot 1 (PRIV_OPP): identity should be all zeros
        assert np.all(out[88 + 9 : 88 + 18] == 0.0)
        # Slot 2 (UNK): identity should be all zeros
        assert np.all(out[88 + 18 : 88 + 27] == 0.0)
        # Slot 3 (PUB, bucket 8): identity at offset 88+27+8
        assert out[88 + 27 + 8] == 1.0

    def test_saliency_values(self):
        # BucketMidNum (midpoint 5.5) → saliency 1.0
        assert abs(bucket_saliency(4) - 1.0) < 1e-6
        # BucketHighKing (midpoint 13.0) → saliency 8.5
        assert abs(bucket_saliency(8) - 8.5) < 1e-6
        # BucketNegKing (midpoint -1.0) → saliency 5.5
        assert abs(bucket_saliency(1) - 5.5) < 1e-6

    def test_no_drawn_card(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0, drawn_card_bucket=-1)
        # NONE → index 10 in drawn card section (offset 29+10=39)
        assert out[39] == 1.0

    def test_all_zeros_except_one_hot(self):
        """Sum of all one-hot bits in public section equals the number of features encoded."""
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Public section: 6 one-hot groups, each should have exactly 1 bit set
        # discard[0:10], stock[10:14], phase[14:20], ctx[20:26], cambia[26:29], drawn[29:40]
        assert out[0:10].sum() == 1.0
        assert out[10:14].sum() == 1.0
        assert out[14:20].sum() == 1.0
        assert out[20:26].sum() == 1.0
        assert out[26:29].sum() == 1.0
        assert out[29:40].sum() == 1.0

    def test_out_of_range_values_ignored(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        # discard_top_bucket=10 → out of range [0-9], no bit set
        out = encode_infoset_eppbs(tags, buckets, 10, 0, 0, 0, 0)
        assert out[0:10].sum() == 0.0

    def test_padding_zeros(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        assert np.all(out[196:200] == 0.0)

    @skipgo
    def test_cross_engine_parity(self):
        """Go and Python agree on the slot tags a King look and swap leaves behind.

        The two backends used to disagree here: Go swapped the tags of the two slots, so
        both stayed known, while Python marked its own slot learned and forgot the
        opponent slot outright, and its own docstring claimed both went UNK (cambia-1553).
        The tags are one-hot in the 224-dim EP-PBS input on both sides, so the Go tag is
        read back out of ``cambia_agent_encode_eppbs`` rather than through a new export.

        Both perspectives are checked: the actor, who looked at both cards, and the
        observer, who saw neither and whose tags follow the cards all the same.
        """
        from src.encoding import EP_PBS_INPUT_DIM

        checked = 0
        for seed in _KING_SWAP_SEEDS:
            result = _play_to_king_swap(seed)
            if result is None:
                continue
            go_agents, py_agents, actor, own_idx, opp_idx, ctx, drawn = result
            checked += 1

            for observer in range(2):
                go_enc = go_agents[observer].encode_eppbs(ctx, drawn)
                assert go_enc.shape == (EP_PBS_INPUT_DIM,)
                py_tags = py_agents[observer].slot_tags

                # The slots the swap touched, named from each observer's own seat.
                if observer == actor:
                    slots = (own_idx, _OPP_SLOTS_START + opp_idx)
                else:
                    slots = (opp_idx, _OPP_SLOTS_START + own_idx)

                for slot in slots:
                    go_tag = int(np.argmax(go_enc[40 + 4 * slot : 44 + 4 * slot]))
                    assert go_tag == int(py_tags[slot]), (
                        f"seed {seed} observer {observer} slot {slot}: "
                        f"go tag {go_tag} != python tag {int(py_tags[slot])}"
                    )

                # Whichever seat is reading, the King look showed the actor both cards
                # and the swap only moved them, so neither slot may come out UNK. That is
                # what the pre-fix Python did to the opponent slot, and what the earlier
                # docstring claimed it did to both.
                known_to_us = (EpistemicTag.PRIV_OWN, EpistemicTag.PUB)
                known_to_them = (EpistemicTag.PRIV_OPP, EpistemicTag.PUB)
                expected = known_to_us if observer == actor else known_to_them
                for slot in slots:
                    assert int(py_tags[slot]) in expected, (
                        f"seed {seed} observer {observer}: slot {slot} tag "
                        f"{int(py_tags[slot])} is not one of {expected} after a King "
                        "look and swap"
                    )

        assert checked > 0, "no seed in the sweep reached a King look and swap"

    @skipgo
    def test_cross_engine_parity_across_a_hand_length_change(self):
        """Go and Python agree on the slot tags after a hand shrinks and grows again.

        Go shifts its EP-PBS slots whenever a hand does (removeOppCard closes the gap,
        insertOppUnknown opens one), so a tag keeps naming the card it was recorded for.
        Python reconciled whole hands and left the tag array where it was, so every tag
        past a snapped slot named a different card until the next full sync (cambia-1690).
        A successful opponent snap shrinks the victim's hand and the fill grows it back,
        which exercises both directions in one game.
        """
        checked = 0
        for seed in _SNAP_FILL_SEEDS:
            snapshots = _play_to_snap_fill(seed)
            if not snapshots:
                continue
            checked += 1
            for label, go_tags, py_tags in snapshots:
                for observer in range(2):
                    assert go_tags[observer] == py_tags[observer], (
                        f"seed {seed} {label}, observer {observer}: "
                        f"go tags {go_tags[observer]} != python tags {py_tags[observer]}"
                    )
            assert [s[0] for s in snapshots][-1] == "after the fill"

        assert checked > 0, "no seed in the sweep reached a snap and its fill"


class TestEPPBSAgentStateTracking:
    """Tests for AgentState EP-PBS epistemic tag tracking."""

    @pytest.fixture
    def agent_state(self):
        """Minimal AgentState for EP-PBS testing (no game engine required)."""
        from unittest.mock import MagicMock
        from src.agent_state import AgentState, KnownCardInfo
        from src.constants import CardBucket, EpistemicTag

        cfg = MagicMock()
        cfg.cambia_rules.penaltyDrawCount = 2
        cfg.cambia_rules.use_jokers = 2

        state = AgentState(
            player_id=0,
            opponent_id=1,
            memory_level=0,
            time_decay_turns=0,
            initial_hand_size=4,
            config=cfg,
        )
        # Manually set up minimal hand state (bypass full initialize)
        from src.constants import GamePhase, StockpileEstimate

        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=0)
            for i in range(4)
        }
        state.opponent_belief = {i: CardBucket.UNKNOWN for i in range(4)}
        state.opponent_last_seen_turn = {}
        state.opponent_card_count = 4
        state.game_phase = GamePhase.EARLY
        state.stockpile_estimate = StockpileEstimate.HIGH
        # EP-PBS is initialized by __post_init__
        return state

    def test_eppbs_fields_initialized(self, agent_state):
        """AgentState has EP-PBS fields after construction."""
        from src.constants import EpistemicTag

        assert len(agent_state.slot_tags) == 12
        assert all(t == EpistemicTag.UNK for t in agent_state.slot_tags)
        assert len(agent_state.slot_buckets) == 12
        assert len(agent_state.own_active_mask) == 0
        assert len(agent_state.opp_active_mask) == 0

    def test_eppbs_tag_transitions_unk_to_priv_own(self, agent_state):
        """UNK → PRIV_OWN when we learn a slot."""
        from src.constants import EpistemicTag

        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)  # LOW_NUM bucket
        assert agent_state.slot_tags[0] == EpistemicTag.PRIV_OWN
        assert agent_state.slot_buckets[0] == 3
        assert 0 in agent_state.own_active_mask

    def test_eppbs_tag_transitions_priv_opp_to_pub(self, agent_state):
        """PRIV_OPP → PUB when we also learn a slot the opponent already knew."""
        from src.constants import EpistemicTag

        # First opponent learns slot 0
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OPP)
        assert agent_state.slot_tags[0] == EpistemicTag.PRIV_OPP
        assert 0 in agent_state.opp_active_mask
        # Now we learn it too → PUB
        agent_state._eppbs_set_tag(0, EpistemicTag.PUB, 5)
        assert agent_state.slot_tags[0] == EpistemicTag.PUB
        assert 0 not in agent_state.opp_active_mask  # removed from opp mask
        assert 0 not in agent_state.own_active_mask  # PUB not in own mask

    def test_eppbs_tag_transitions_priv_own_to_pub(self, agent_state):
        """PRIV_OWN → PUB when opponent learns a slot we already knew."""
        from src.constants import EpistemicTag

        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 4)
        assert 2 in agent_state.own_active_mask
        # Opponent learns it → PUB
        agent_state._eppbs_set_tag(2, EpistemicTag.PUB, 4)
        assert agent_state.slot_tags[2] == EpistemicTag.PUB
        assert 2 not in agent_state.own_active_mask

    def test_eppbs_saliency_eviction(self, agent_state):
        """Peeking 4 own cards evicts the lowest-saliency one."""
        from src.constants import EpistemicTag

        # Peek 3 low-saliency cards first: MID_NUM(4,sal=1.0), MID_NUM(4), LOW_NUM(3,sal=1.5)
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 4)  # MID_NUM, sal=1.0
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 3)  # LOW_NUM, sal=1.5
        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 5)  # PEEK_SELF, sal=3.0
        assert agent_state.own_active_mask == [0, 1, 2]
        # Now peek a high-saliency card: HIGH_KING(8,sal=8.5)
        agent_state._eppbs_set_tag(3, EpistemicTag.PRIV_OWN, 8)  # HIGH_KING, sal=8.5
        # Slot 0 (sal=1.0) should have been evicted
        assert 0 not in agent_state.own_active_mask
        assert agent_state.slot_tags[0] == EpistemicTag.UNK
        assert 3 in agent_state.own_active_mask
        assert len(agent_state.own_active_mask) == 3

    def test_eppbs_saliency_no_eviction_if_new_lower(self, agent_state):
        """New card with lower saliency than all existing is NOT added."""
        from src.constants import EpistemicTag

        # Fill mask with high-saliency cards
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 8)  # HIGH_KING, sal=8.5
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 1)  # NEG_KING, sal=5.5
        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 6)  # PEEK_OTHER, sal=5.0
        # Now add very low saliency: MID_NUM(4, sal=1.0) - lower than minimum (5.0)
        agent_state._eppbs_set_tag(3, EpistemicTag.PRIV_OWN, 4)  # MID_NUM, sal=1.0
        # Slot 3 should NOT be in mask (lower saliency than slot 2's 5.0)
        assert 3 not in agent_state.own_active_mask
        # Slot 2 (min of {8.5,5.5,5.0}=5.0) should remain since new sal=1.0 < 5.0
        assert 2 in agent_state.own_active_mask

    def test_eppbs_fifo_eviction(self, agent_state):
        """Opponent peeking 4 slots evicts the oldest (FIFO)."""
        from src.constants import EpistemicTag

        # Opponent peeks slots 6,7,8 (3 opp slots)
        agent_state._eppbs_set_tag(6, EpistemicTag.PRIV_OPP)
        agent_state._eppbs_set_tag(7, EpistemicTag.PRIV_OPP)
        agent_state._eppbs_set_tag(8, EpistemicTag.PRIV_OPP)
        assert agent_state.opp_active_mask == [6, 7, 8]
        # Opponent peeks a 4th slot → FIFO evict slot 6
        agent_state._eppbs_set_tag(9, EpistemicTag.PRIV_OPP)
        assert 6 not in agent_state.opp_active_mask
        assert agent_state.slot_tags[6] == EpistemicTag.UNK
        assert 9 in agent_state.opp_active_mask
        assert agent_state.opp_active_mask == [7, 8, 9]

    def test_eppbs_clone_copies_state(self, agent_state):
        """clone() preserves EP-PBS state."""
        from src.constants import EpistemicTag

        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)
        agent_state._eppbs_set_tag(6, EpistemicTag.PRIV_OPP)
        cloned = agent_state.clone()
        assert cloned.slot_tags[0] == EpistemicTag.PRIV_OWN
        assert cloned.slot_buckets[0] == 3
        assert cloned.slot_tags[6] == EpistemicTag.PRIV_OPP
        assert 0 in cloned.own_active_mask
        assert 6 in cloned.opp_active_mask
        # Mutations to original don't affect clone
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 5)
        assert cloned.slot_tags[1] == EpistemicTag.UNK

    def test_eppbs_forget_slot(self, agent_state):
        """Setting tag to UNK removes from active masks."""
        from src.constants import EpistemicTag

        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)
        assert 0 in agent_state.own_active_mask
        agent_state._eppbs_set_tag(0, EpistemicTag.UNK)
        assert 0 not in agent_state.own_active_mask
        assert agent_state.slot_tags[0] == EpistemicTag.UNK
