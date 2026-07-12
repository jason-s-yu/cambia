"""tests/test_sequence_tokenizer.py

AC2 gate for the PRT-CFR perfect-recall sequence tokenizer (src/sequence_encoding.py).

The tokenizer turns a player's observation-action event stream into a flat token
sequence the GRU will consume. The gate this file enforces is LOSSLESS ROUND-TRIP:
replaying seeded games (the {A,6} 2-card plateau game AND full 2-player games),
the token sequence must reconstruct the per-player observation-action history with
no information lost relative to the X1 perfect-recall key content.

The X1 perfect-recall key (cfr/tools/tiny_solver.py, --perfect-recall) carries,
per acting player p:
  - priv_init[p]: p's peeked initial-hand cards (slot -> card)
  - priv_draw[p]: p's own stockpile draws, in order
  - pub_path:     the full public reveal sequence for ALL players, in order:
                  (actor, repr(action), repr(discard_top_after))

This test independently collects that exact content per player from the live
game (a "ground truth" GT trace, NOT via the tokenizer), encodes the per-player
filtered observation stream with the tokenizer, decodes it, and asserts the
decoded stream reproduces the GT content field-for-field.

Run (CPU-only, no torch):
  python -m pytest tests/test_sequence_tokenizer.py -v
"""

import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Make the package importable whether pytest is run from cfr/ or repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card import Card
from src.config import Config, CambiaRulesConfig, load_config
from src.constants import (
    NUM_PLAYERS,
    ActionCallCambia,
    ActionDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
)
from src.game.engine import CambiaGameState
import src.sequence_encoding as se
from src.cfr.worker import _create_observation, _filter_observation

# Helpers reused from existing cross-validation / parity infra (per AC2 task).
try:
    from tests.test_cross_engine_samples import _setup_python_game_matching_go
except ImportError:  # pragma: no cover - path fallback
    from test_cross_engine_samples import _setup_python_game_matching_go  # type: ignore


# ---------------------------------------------------------------------------
# Game builders
# ---------------------------------------------------------------------------


def _full_2p_config() -> Config:
    """Config matching the full-2P helper rules (Go defaults, 4 cards/hand)."""
    cfg = Config()
    cfg.cambia_rules = CambiaRulesConfig()
    cfg.cambia_rules.allowDrawFromDiscardPile = True
    cfg.cambia_rules.allowOpponentSnapping = True
    cfg.cambia_rules.max_game_turns = 46
    return cfg


def _build_tiny_game(seed: int) -> Tuple[CambiaGameState, Config]:
    """Build the {A,6} 2-card plateau game from tiny_2card_plateau.yaml.

    Mirrors the X1 solver's construction: CambiaGameState(house_rules=cfg.cambia_rules,
    _rng=Random(seed)). _setup_game runs in __post_init__ using deck_ranks /
    cards_per_player / initial_view_count from the config.
    """
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "tiny_2card_plateau.yaml",
    )
    cfg = load_config(cfg_path)
    game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(seed))
    return game, cfg


# ---------------------------------------------------------------------------
# Ground-truth perfect-recall trace (independent of the tokenizer)
# ---------------------------------------------------------------------------
#
# We collect, per observer, exactly what the X1 key carries:
#   - priv_init: sorted [(slot, (rank, suit)), ...] of peeked initial-hand cards
#   - priv_draw: [(rank, suit), ...] of own drawn cards surfaced to the observer
#   - pub_path:  [(actor, action, discard_top_identity), ...] public reveals
# plus snap outcomes and the cambia caller (public). Card identity is the
# (rank, suit) pair, matching repr(Card) distinguishability.


def _card_ident(card: Optional[Card]) -> Optional[Tuple[str, Optional[str]]]:
    return None if card is None else (card.rank, card.suit)


class _GTTrace:
    """Per-observer ground-truth perfect-recall content, collected live."""

    def __init__(self, observer_id: int):
        self.observer_id = observer_id
        self.priv_init: List[Tuple[int, Tuple[str, Optional[str]]]] = []
        # public-path entries (one per observation), and private/snap/cambia
        # events interleaved in temporal order, recorded as typed tuples so the
        # comparison against decoded frames is exact and order-sensitive.
        self.events: List[Tuple] = []

    def set_init_peek(self, hand: List[Card], peek_indices: Tuple[int, ...]):
        for slot in sorted(peek_indices):
            if slot < len(hand):
                self.priv_init.append((slot, _card_ident(hand[slot])))

    def record_observation(self, filtered_obs: Any):
        actor = filtered_obs.acting_player
        action = filtered_obs.action
        # drawn (private) precedes the public frame, matching the tokenizer.
        drawn = getattr(filtered_obs, "drawn_card", None)
        if drawn is not None and actor == self.observer_id:
            self.events.append(("drawn", _card_ident(drawn)))
        # public reveal entry
        self.events.append(
            (
                "public",
                actor,
                _action_repr_key(action),
                _card_ident(filtered_obs.discard_top_card),
            )
        )
        # cambia (public)
        if getattr(filtered_obs, "did_cambia_get_called", False) and isinstance(
            action, ActionCallCambia
        ):
            caller = getattr(filtered_obs, "who_called_cambia", None)
            self.events.append(("cambia", caller if caller is not None else actor))
        # snaps (public)
        for snap_info in getattr(filtered_obs, "snap_results", None) or []:
            outcome, slot = se._classify_snap(snap_info)
            self.events.append(("snap", snap_info.get("snapper"), outcome, slot))


def _action_repr_key(action: Any) -> Any:
    """Public-structure key for an action: the action OBJECT itself.

    Cambia actions are NamedTuples; their structural equality (==) captures
    exactly the public structure repr(action) carries (tag + slot/flag args,
    never card contents). Returning the object itself -- rather than routing
    through the tokenizer's own _action_local_id -- keeps the round-trip GT
    comparison INDEPENDENT of the codec under test, so a hypothetical encoder
    collision (two distinct actions -> one id) would be caught, not masked. The
    tokenizer's local-id bijection is separately checked by
    test_all_action_types_round_trip. Returns None for the no-action initial obs.
    """
    return action


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------


def _replay_collect(
    game: CambiaGameState,
    cfg: Config,
    rng: random.Random,
    max_steps: int = 400,
    avoid_cambia: bool = False,
) -> Tuple[
    Dict[int, List[Any]],  # per-observer filtered observation stream
    Dict[int, _GTTrace],  # per-observer ground-truth trace
    Dict[int, List[Card]],  # per-observer initial hand
    Dict[int, Tuple[int, ...]],  # per-observer initial peek indices
    int,  # number of decision steps taken
]:
    """Play a game with random legal actions, collecting per-player streams.

    For each applied action we build the FULL observation (private info
    populated) via the production _create_observation, then filter it per player
    with _filter_observation. That filtered stream is what the tokenizer consumes
    and what the GT trace records, so both see the same per-player information.
    """
    obs_streams: Dict[int, List[Any]] = {p: [] for p in range(NUM_PLAYERS)}
    gt: Dict[int, _GTTrace] = {p: _GTTrace(p) for p in range(NUM_PLAYERS)}
    init_hands: Dict[int, List[Card]] = {}
    init_peeks: Dict[int, Tuple[int, ...]] = {}

    for p in range(NUM_PLAYERS):
        init_hands[p] = list(game.players[p].hand)
        init_peeks[p] = tuple(game.players[p].initial_peek_indices)
        gt[p].set_init_peek(init_hands[p], init_peeks[p])

    steps = 0
    for _ in range(max_steps):
        if game.is_terminal():
            break
        actor = game.get_acting_player()
        if actor == -1:
            break
        legal = list(game.get_legal_actions())
        if not legal:
            break

        # Deterministic-but-varied choice via the test RNG. With avoid_cambia,
        # skip premature CallCambia so games run to natural length (cap stress).
        pool = legal
        if avoid_cambia:
            non_cambia = [a for a in legal if not isinstance(a, ActionCallCambia)]
            if non_cambia:
                pool = non_cambia
        action = rng.choice(pool)

        # Apply, then build the full observation from the post-action state.
        game.apply_action(action)
        snap_results = list(getattr(game, "snap_results_log", []) or [])
        full_obs = _create_observation(None, action, game, actor, snap_results)
        if full_obs is None:
            # Observation build failed (engine edge) -- skip this event for both
            # streams so encoder and GT stay aligned. Rare; not a tokenizer issue.
            steps += 1
            continue

        for observer in range(NUM_PLAYERS):
            filtered = _filter_observation(full_obs, observer)
            obs_streams[observer].append(filtered)
            gt[observer].record_observation(filtered)
        steps += 1

    return obs_streams, gt, init_hands, init_peeks, steps


# ---------------------------------------------------------------------------
# Lossless comparison: decoded frames vs ground-truth content
# ---------------------------------------------------------------------------


def _decoded_to_gt_events(
    decoded: List[se.DecodedEvent],
) -> Tuple[List[Tuple[int, Tuple[str, Optional[str]]]], List[Tuple]]:
    """Project decoded frames into the same (priv_init, events) shape as _GTTrace."""
    priv_init: List[Tuple[int, Tuple[str, Optional[str]]]] = []
    events: List[Tuple] = []
    for ev in decoded:
        if ev.kind == "init_peek":
            priv_init.append((ev.peek_slot, _card_ident(ev.peek_card)))
        elif ev.kind == "drawn":
            events.append(("drawn", _card_ident(ev.drawn_card)))
        elif ev.kind == "public":
            events.append(
                (
                    "public",
                    ev.actor,
                    _action_repr_key(ev.action),
                    _card_ident(ev.discard_top),
                )
            )
        elif ev.kind == "cambia":
            events.append(("cambia", ev.cambia_caller))
        elif ev.kind == "snap":
            events.append(("snap", ev.snap_actor, ev.snap_outcome, ev.snap_slot))
        else:  # pragma: no cover
            raise AssertionError(f"unexpected decoded kind {ev.kind}")
    return priv_init, events


def _assert_lossless(
    obs_streams: Dict[int, List[Any]],
    gt: Dict[int, _GTTrace],
    init_hands: Dict[int, List[Card]],
    init_peeks: Dict[int, Tuple[int, ...]],
    label: str,
) -> int:
    """Encode -> decode -> compare against GT for each observer. Returns max len."""
    max_len = 0
    for observer in range(NUM_PLAYERS):
        seq = se.encode_observation_sequence(
            init_hands[observer],
            init_peeks[observer],
            obs_streams[observer],
            observer,
        )
        max_len = max(max_len, len(seq))

        # Every token id must be in vocabulary range.
        assert all(0 <= t < se.VOCAB_SIZE for t in seq), (
            f"{label} obs={observer}: token id out of vocab range "
            f"(VOCAB_SIZE={se.VOCAB_SIZE})"
        )

        decoded = se.decode_sequence(seq)
        dec_priv_init, dec_events = _decoded_to_gt_events(decoded)

        assert dec_priv_init == gt[observer].priv_init, (
            f"{label} obs={observer}: priv_init mismatch\n"
            f"  GT     : {gt[observer].priv_init}\n"
            f"  decoded: {dec_priv_init}"
        )
        assert len(dec_events) == len(gt[observer].events), (
            f"{label} obs={observer}: event-count mismatch "
            f"GT={len(gt[observer].events)} decoded={len(dec_events)}"
        )
        for idx, (g, d) in enumerate(zip(gt[observer].events, dec_events)):
            assert g == d, (
                f"{label} obs={observer}: event {idx} mismatch\n"
                f"  GT     : {g}\n"
                f"  decoded: {d}"
            )
    return max_len


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# A spread of seeds; each full game yields many decision points across action
# types. Tiny game is tiny (<=3 turns) but exercises the exact X1 plateau game.
_TINY_SEEDS = [0, 1, 2, 3, 5, 7, 11, 13, 42, 137]
_FULL_SEEDS = [42, 137, 313, 1729, 2718, 3141, 4096, 12345, 54321, 99991]


def test_tiny_2card_plateau_roundtrip_lossless():
    """{A,6} 2-card plateau game: token stream reconstructs PR content losslessly."""
    overall_max = 0
    for seed in _TINY_SEEDS:
        game, cfg = _build_tiny_game(seed)
        rng = random.Random(10_000 + seed)
        obs_streams, gt, init_hands, init_peeks, steps = _replay_collect(
            game, cfg, rng, max_steps=64
        )
        # The tiny game must actually produce decision events.
        assert steps > 0, f"tiny seed {seed}: no decision steps taken"
        mx = _assert_lossless(
            obs_streams, gt, init_hands, init_peeks, f"tiny[seed={seed}]"
        )
        overall_max = max(overall_max, mx)
    # Tiny game is far under the cap.
    assert (
        overall_max <= se.SEQ_CAP
    ), f"tiny max token length {overall_max} exceeds cap {se.SEQ_CAP}"


def test_full_2p_roundtrip_lossless():
    """Full 2-player games: token stream reconstructs PR content losslessly."""
    overall_max = 0
    total_steps = 0
    for seed in _FULL_SEEDS:
        game = _setup_python_game_matching_go(seed)
        cfg = _full_2p_config()
        rng = random.Random(20_000 + seed)
        obs_streams, gt, init_hands, init_peeks, steps = _replay_collect(
            game, cfg, rng, max_steps=400
        )
        total_steps += steps
        mx = _assert_lossless(
            obs_streams, gt, init_hands, init_peeks, f"full[seed={seed}]"
        )
        overall_max = max(overall_max, mx)
    assert total_steps > 0, "full games produced no decision steps"
    # Report-worthy: max observed sequence length vs the 256 cap.
    print(
        f"\n[full 2P] max token sequence length over {len(_FULL_SEEDS)} games: "
        f"{overall_max} (cap {se.SEQ_CAP})"
    )
    assert overall_max <= se.SEQ_CAP, (
        f"full 2P max token length {overall_max} EXCEEDS cap {se.SEQ_CAP}; "
        f"truncation would lose information -- revisit SEQ_CAP or frame width"
    )


def test_full_2p_natural_length_exceeds_cap_and_truncation_is_frame_safe():
    """Full 2P games at NATURAL length exceed the 256 cap; truncation stays decodable.

    The unbiased-random driver ends games quickly (it picks CallCambia /
    game-ending snaps early), so those sequences are short. A self-play CFR
    policy plays full-length games. Driving games to natural length here (only
    avoiding premature CallCambia) shows the honest worst case: raw sequences
    reach ~1200 tokens (mean ~726) at ~6 tokens/event over ~120 events, FAR over
    the SEQ_CAP=256 design target. This test documents that and verifies the
    frame-aligned keep-most-recent truncation always yields a sequence that
    (a) fits the cap and (b) decodes cleanly (no partial leading frame).
    """
    raw_lengths: List[int] = []
    capped_lengths: List[int] = []
    seeds = list(range(60))
    for seed in seeds:
        game = _setup_python_game_matching_go(seed)
        cfg = _full_2p_config()
        rng = random.Random(40_000 + seed)
        obs_streams, _gt, init_hands, init_peeks, steps = _replay_collect(
            game, cfg, rng, max_steps=400, avoid_cambia=True
        )
        for observer in range(NUM_PLAYERS):
            # Raw (untruncated) length.
            raw = se.encode_observation_sequence(
                init_hands[observer],
                init_peeks[observer],
                obs_streams[observer],
                observer,
                seq_cap=10**9,
            )
            raw_lengths.append(len(raw))
            # Capped (default SEQ_CAP) length must fit and decode.
            capped = se.encode_observation_sequence(
                init_hands[observer],
                init_peeks[observer],
                obs_streams[observer],
                observer,
            )
            capped_lengths.append(len(capped))
            assert (
                len(capped) <= se.SEQ_CAP
            ), f"seed {seed} obs {observer}: capped length {len(capped)} > cap {se.SEQ_CAP}"
            # Frame-safe: decodes without raising on a partial leading frame.
            decoded = se.decode_sequence(capped)
            assert isinstance(decoded, list)

    import statistics

    raw_max = max(raw_lengths)
    print(
        f"\n[natural-length full 2P, {len(seeds)} seeds] "
        f"RAW token length: max={raw_max} mean={statistics.mean(raw_lengths):.0f}; "
        f"capped max={max(capped_lengths)} (SEQ_CAP={se.SEQ_CAP}). "
        f"raw sequences over cap: {sum(1 for x in raw_lengths if x > se.SEQ_CAP)}/{len(raw_lengths)}"
    )
    # The point of this test: natural games DO exceed the design cap.
    assert raw_max > se.SEQ_CAP, (
        "expected natural-length full 2P games to exceed SEQ_CAP; if this fails, "
        "the cap may now be adequate -- re-evaluate the truncation concern"
    )


def test_truncation_is_frame_aligned_and_a_suffix_of_full_decode():
    """Frame-aligned keep-most-recent truncation drops oldest WHOLE events.

    The decoded truncated stream must be an exact SUFFIX of the decoded full
    stream (truncation removes a prefix of events and keeps the rest verbatim),
    and must decode without a partial leading frame.
    """
    # Use a natural-length game so there are many frames to truncate.
    game = _setup_python_game_matching_go(137)
    cfg = _full_2p_config()
    rng = random.Random(99)
    obs_streams, _gt, init_hands, init_peeks, steps = _replay_collect(
        game, cfg, rng, max_steps=400, avoid_cambia=True
    )
    assert steps > 0
    observer = 0
    full_seq = se.encode_observation_sequence(
        init_hands[observer],
        init_peeks[observer],
        obs_streams[observer],
        observer,
        seq_cap=10**9,
    )
    full_events = se.decode_sequence(full_seq)

    # Force a cap below the natural length to exercise truncation.
    small_cap = max(8, len(full_seq) // 3)
    trunc = se.encode_observation_sequence(
        init_hands[observer],
        init_peeks[observer],
        obs_streams[observer],
        observer,
        seq_cap=small_cap,
    )
    assert len(trunc) <= small_cap, "truncated sequence exceeds the forced cap"
    # Frame-safe: decodes cleanly (no ValueError on a partial leading frame).
    trunc_events = se.decode_sequence(trunc)
    assert len(trunc_events) > 0 and len(trunc_events) < len(full_events)

    # Suffix property: the kept events equal the tail of the full event list.
    def _key(ev: se.DecodedEvent):
        return (
            ev.kind,
            ev.peek_slot,
            _card_ident(ev.peek_card),
            ev.actor,
            _action_repr_key(ev.action),
            _card_ident(ev.discard_top),
            _card_ident(ev.drawn_card),
            ev.snap_actor,
            ev.snap_outcome,
            ev.snap_slot,
            ev.cambia_caller,
        )

    tail = full_events[len(full_events) - len(trunc_events) :]
    assert [_key(e) for e in trunc_events] == [
        _key(e) for e in tail
    ], "frame-aligned truncation is not an exact suffix of the full decode"


def test_vocab_layout_is_contiguous_and_nonoverlapping():
    """Vocabulary blocks tile [0, VOCAB_SIZE) with no gaps or overlaps."""
    summ = se.vocab_summary()
    blocks = summ["blocks"]
    # Special ids occupy [0, NUM_SPECIAL).
    spans = [(0, se.NUM_SPECIAL)]
    for _name, (base, size) in blocks.items():
        spans.append((base, base + size))
    spans.sort()
    cursor = 0
    for start, end in spans:
        assert start == cursor, f"vocab gap/overlap: expected {cursor}, got {start}"
        cursor = end
    assert (
        cursor == se.VOCAB_SIZE
    ), f"vocab blocks end at {cursor}, VOCAB_SIZE={se.VOCAB_SIZE}"


def test_all_action_types_round_trip():
    """Every action type's token id decodes back to an equal action."""
    from src.constants import (
        ActionAbilityBlindSwapSelect,
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionAbilityPeekOtherSelect,
        ActionAbilityPeekOwnSelect,
        ActionDrawDiscard,
        ActionSnapOpponent,
        ActionSnapOpponentMove,
        ActionSnapOwn,
    )

    actions = [
        ActionDrawStockpile(),
        ActionDrawDiscard(),
        ActionCallCambia(),
        ActionDiscard(use_ability=True),
        ActionDiscard(use_ability=False),
        ActionReplace(target_hand_index=0),
        ActionReplace(target_hand_index=5),
        ActionAbilityPeekOwnSelect(target_hand_index=2),
        ActionAbilityPeekOtherSelect(target_opponent_hand_index=3),
        ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=4),
        ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=5),
        ActionAbilityKingSwapDecision(perform_swap=True),
        ActionAbilityKingSwapDecision(perform_swap=False),
        ActionPassSnap(),
        ActionSnapOwn(own_card_hand_index=2),
        ActionSnapOpponent(opponent_target_hand_index=1),
        ActionSnapOpponentMove(own_card_to_move_hand_index=0, target_empty_slot_index=3),
    ]
    for a in actions:
        local, _name = se._action_local_id(a)
        assert 0 <= local < se.NUM_ACTION_IDS
        back = se._decode_action_local_id(local)
        assert back == a, f"action round-trip failed: {a} -> {local} -> {back}"


def test_all_card_identities_round_trip():
    """Every distinct card identity (and the joker) round-trips through the codec."""
    for rank, suit in se.CARD_IDENTITIES:
        card = Card(rank=rank, suit=suit)
        local = se._card_local_id(card)
        back = se._local_card_id_to_card(local)
        assert back is not None and (back.rank, back.suit) == (rank, suit)
    # The none/unknown id maps to None.
    assert se._local_card_id_to_card(se.CARD_NONE_LOCAL) is None
