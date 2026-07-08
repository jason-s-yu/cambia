"""tests/test_discard_ability_from_discard_pile.py

cambia-261: regression guard for the immediate-discard ability source gate
(the divergence originally reproduced under cambia-247, now fixed).

RULES.md Section 4 ("Special Abilities"): "Abilities only trigger when you
draw from the Stockpile and discard immediately." Section 3B: "(You *cannot*
use Special Abilities when drawing from the discard pile)."

Go's engine enforces this at legal-action-generation time: engine/legal.go's
legalPostDraw only sets ActionDiscardWithAbility when drawnFrom ==
DrawnFromStockpile (see canUseAbility gating). The Python reference engine
tracks the same "drawn_from" provenance in pending_action_data and now
consults it in both the post-draw legal-action generator
(CambiaGameState._get_legal_pending_actions) and the ActionDiscard apply path
(AbilityMixin._handle_pending_action). ActionDiscard(use_ability=True) is
therefore no longer offered, nor executed, for a card drawn from the DISCARD
pile, matching RULES.md and the Go engine.

This is the gap tests/test_token_stream_parity.py's lockstep driver formerly
excluded via _DISCARD_WITH_ABILITY_IDX (index 4); with the gate in place that
exclusion has been removed and the parity suite runs un-narrowed. These tests
guard the same CambiaGameState.get_legal_actions() / apply_action() path that
src.encoding.encode_action_mask (and therefore every eval instrument that
runs the Python engine: mean_imp, LBR, head-to-head, tiny gates) consumes
directly.
"""

from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.card import Card
from src.constants import (
    ACE,
    TWO,
    THREE,
    JACK,
    HEARTS,
    CLUBS,
    DIAMONDS,
    SPADES,
    ActionDrawDiscard,
    ActionDiscard,
    ActionAbilityBlindSwapSelect,
)
from src.encoding import action_to_index, encode_action_mask


def _make_rules(**kwargs):
    from src.config import CambiaRulesConfig as _RC

    try:
        return _RC(**kwargs)
    except TypeError:
        obj = _RC()
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


def _make_game_with_ability_card_on_discard() -> CambiaGameState:
    """2-player game where an ability card (Jack) sits on top of the discard
    pile and allowDrawFromDiscardPile is enabled (the PRT-CFR production rule
    profile's setting -- prtcfr_production.yaml, prtcfr_worker.py)."""
    rules = _make_rules(allowDrawFromDiscardPile=True)

    p0_hand = [
        Card(rank=ACE, suit=HEARTS),
        Card(rank=TWO, suit=CLUBS),
        Card(rank=THREE, suit=DIAMONDS),
        Card(rank=TWO, suit=SPADES),
    ]
    p1_hand = [
        Card(rank=ACE, suit=CLUBS),
        Card(rank=TWO, suit=HEARTS),
        Card(rank=THREE, suit=SPADES),
        Card(rank=ACE, suit=DIAMONDS),
    ]
    players = [
        PlayerState(hand=p0_hand, initial_peek_indices=(0, 1)),
        PlayerState(hand=p1_hand, initial_peek_indices=(0, 1)),
    ]
    return CambiaGameState(
        players=players,
        stockpile=[Card(rank=THREE, suit=CLUBS)],
        discard_pile=[Card(rank=JACK, suit=SPADES)],  # ability card on top
        current_player_index=0,
        house_rules=rules,
    )


def test_discard_pile_draw_does_not_offer_ability_discard():
    """After ActionDrawDiscard, Python's legal_actions() must not include
    ActionDiscard(use_ability=True) for the discard-drawn ability card. Go
    (engine/legal.go legalPostDraw) and RULES.md Sec. 3B/4 both forbid this --
    ability discard is legal only for a card drawn from the STOCKPILE.
    """
    game = _make_game_with_ability_card_on_discard()
    game.apply_action(ActionDrawDiscard())

    legal = game.get_legal_actions()
    offending = [a for a in legal if isinstance(a, ActionDiscard) and a.use_ability]
    assert not offending, (
        "ActionDiscard(use_ability=True) must not be legal after a discard-pile "
        "draw (RULES.md Sec. 3B/4; Go engine/legal.go legalPostDraw). Found: "
        f"{offending}."
    )

    # Same object encode_action_mask (the function every Python-engine eval
    # instrument's action mask construction calls) receives -- confirm the
    # source gate survives that translation, index 4 masked off.
    mask = encode_action_mask(legal)
    idx = action_to_index(ActionDiscard(use_ability=True))
    assert idx == 4, "action index layout drifted; re-check _DISCARD_WITH_ABILITY_IDX = 4"
    assert not mask[idx], (
        "encode_action_mask must mark index 4 illegal after a discard-pile draw, "
        "so eval-side masking inherits the source gate."
    )


def test_discard_pile_draw_ability_discard_does_not_execute():
    """Applying ActionDiscard(use_ability=True) after a discard-pile draw must
    not fire the drawn card's special ability (here, Jack's BlindSwap). The
    action is illegal (see the legal-set test above), so the engine rejects it
    and stays in the post-draw pending state; the apply-path source guard
    additionally ensures the ability cannot fire even if a caller bypasses the
    mask. Either way, no BlindSwap selection becomes available -- matching
    RULES.md Sec. 3B/4."""
    game = _make_game_with_ability_card_on_discard()
    game.apply_action(ActionDrawDiscard())

    game.apply_action(ActionDiscard(use_ability=True))

    legal_after = game.get_legal_actions()
    assert not any(
        isinstance(a, ActionAbilityBlindSwapSelect) for a in legal_after
    ), (
        "Jack's BlindSwap ability must not fire after discarding a card drawn "
        "from the discard pile -- the source gate forbids abilities on "
        "discard-pile draws (RULES.md Sec. 3B/4)."
    )
