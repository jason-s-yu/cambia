"""tests/test_discard_ability_from_discard_pile.py

cambia-247: reproduces the divergence hidden by _DISCARD_WITH_ABILITY_IDX in
tests/test_token_stream_parity.py.

RULES.md Section 4 ("Special Abilities"): "Abilities only trigger when you
draw from the Stockpile and discard immediately." Section 3B: "(You *cannot*
use Special Abilities when drawing from the discard pile)."

Go's engine enforces this at legal-action-generation time: engine/legal.go's
legalPostDraw only sets ActionDiscardWithAbility when drawnFrom ==
DrawnFromStockpile (see canUseAbility gating). The Python reference engine
tracks the same "drawn_from" provenance in pending_action_data (see
src/game/_ability_mixin.py, used correctly by the AllowReplaceAbilities
branch) but its post-draw legal-action generator
(CambiaGameState._get_legal_pending_actions) and its ActionDiscard apply path
never consult it for the *immediate discard* choice -- only
card_has_discard_ability(drawn_card) and hand-count fizzle conditions gate
it. Consequently Python offers, and will actually execute, "discard with
ability" for a card drawn from the DISCARD pile, which RULES.md and the Go
engine both forbid.

This is exactly the gap tests/test_token_stream_parity.py's lockstep driver
excludes via _DISCARD_WITH_ABILITY_IDX (index 4) to keep the S1W11
reconciliation scoped to the snap/reshuffle divergences it actually fixed.
This test demonstrates the excluded divergence is real and reachable through
the same CambiaGameState.get_legal_actions() / apply_action() path that
src.encoding.encode_action_mask (and therefore every eval instrument that
runs the Python engine: mean_imp, LBR, head-to-head, tiny gates) consumes
directly, with no independent re-validation.
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


def test_discard_pile_draw_spuriously_offers_ability_discard():
    """BUG (pre-existing, out of scope for S1W11): after ActionDrawDiscard,
    Python's legal_actions() still includes ActionDiscard(use_ability=True)
    for the discard-drawn ability card. Go (engine/legal.go legalPostDraw)
    and RULES.md Sec. 3B/4 both forbid this -- ability discard should only
    ever be legal for a card drawn from the STOCKPILE.
    """
    game = _make_game_with_ability_card_on_discard()
    game.apply_action(ActionDrawDiscard())

    legal = game.get_legal_actions()
    offending = [
        a for a in legal if isinstance(a, ActionDiscard) and a.use_ability
    ]
    assert offending, (
        "Expected the pre-existing legal-gen gap to reproduce: "
        "ActionDiscard(use_ability=True) should be (incorrectly) legal here. "
        "If this now fails, the gap may have been fixed upstream of this "
        "test -- re-verify _DISCARD_WITH_ABILITY_IDX in "
        "test_token_stream_parity.py is still needed."
    )

    # Same object encode_action_mask (the function every Python-engine eval
    # instrument's action mask construction calls) receives -- prove the
    # spurious legality survives that translation, index 4 unmasked.
    mask = encode_action_mask(legal)
    idx = action_to_index(ActionDiscard(use_ability=True))
    assert idx == 4, "action index layout drifted; re-check _DISCARD_WITH_ABILITY_IDX = 4"
    assert mask[idx], (
        "encode_action_mask should (incorrectly) mark index 4 legal here, "
        "confirming eval-side masking inherits the gap with no independent "
        "re-validation."
    )


def test_discard_pile_draw_ability_discard_actually_executes():
    """Confirms the spurious legal action is not a harmless no-op: applying
    it actually triggers the drawn card's special ability (here, Jack's
    BlindSwap), in direct violation of RULES.md Sec. 3B/4."""
    game = _make_game_with_ability_card_on_discard()
    game.apply_action(ActionDrawDiscard())

    game.apply_action(ActionDiscard(use_ability=True))

    legal_after = game.get_legal_actions()
    assert any(
        isinstance(a, ActionAbilityBlindSwapSelect) for a in legal_after
    ), (
        "Expected the Jack's BlindSwap ability to have actually fired after "
        "discarding a card drawn from the discard pile -- this proves the "
        "gap is a real rule violation (ability executes), not just a "
        "cosmetic over-listing in get_legal_actions()."
    )
