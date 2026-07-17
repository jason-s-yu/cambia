"""tests/test_b3_representation_gate.py

B3 standing representation gate for the PRT-CFR production token stream (the
worker path: src.cfr.worker._create_observation/_filter_observation ->
src.sequence_encoding). Two checks:

  (a) mask-determinism (default run): over many self-play games, an identical
      token prefix MUST imply an identical legal-action mask. This is the
      CFR-soundness precondition the F1 fix (cambia-528) restores: the drawn card
      lives in the prefix at the post-draw node, so the discard/replace mask is a
      function of the infoset. It has real teeth on an ability deck: before F1 a
      post-draw node after drawing an ability card (7 -> discard_with_ability
      legal) and one after a plain card (6) shared a token prefix but had
      different masks -- a prefix -> {two masks} collision this test fails on
      (verified: the pre-fix tokenizer produced 90 such conflicts on the deck
      below; the fix produces 0).

  (b) tabular-CFR-on-production-tokens floor (slow, CLI-invokable): on a small
      ability game, run tabular CFR whose infoset key is the production token
      stream, then measure that token policy's exploitability against a
      PERFECT-RECALL best responder. With F1 + F2 the tokens are a lossless
      perfect-recall encoding, so the token policy is near-optimal in the true
      game and its NashConv tracks the perfect-recall control's floor. Before F2
      the peek results were dropped, so the token partition merged
      perfect-recall-distinct infosets and a perfect-recall BR exploited the
      blindness -> a higher floor. Marked @pytest.mark.slow; run with:
        python -m pytest tests/test_b3_representation_gate.py -m slow -s
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Dict, List, Tuple

import pytest

# The engine emits a per-node warning on every rejected snap / penalty draw; on
# the exhaustive floor tree that is tens of thousands of lines and dominates
# wall time. Silence it for this gate (behavior is asserted, not logs).
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import NUM_PLAYERS
from src.encoding import action_to_index
from src.game.engine import CambiaGameState
import src.sequence_encoding as se
from src.cfr.worker import _create_observation, _filter_observation

try:
    from tests.test_cross_engine_samples import _setup_python_game_matching_go
except ImportError:  # pragma: no cover - path fallback
    from test_cross_engine_samples import _setup_python_game_matching_go  # type: ignore


_TINY_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "tiny_2card_plateau.yaml",
)
_RULES_CLS = None


def _rules_cls():
    """The REAL CambiaRulesConfig class. tests/conftest.py injects a src.config
    stub whose CambiaRulesConfig lacks deck_ranks (so a bare import would silently
    drop the reduced-deck override and deal the full 54-card deck); load_config
    delegates to the real module, so its cambia_rules type is the real class."""
    global _RULES_CLS
    if _RULES_CLS is None:
        from src.config import load_config

        _RULES_CLS = type(load_config(_TINY_CFG).cambia_rules)
    return _RULES_CLS


def _ability_rules():
    """Small ability-on 2-card deck {6, 7} (no jokers). Drawing a 7 (peek-own)
    makes discard_with_ability legal (the post-draw mask depends on the drawn
    card -> F1) and, on use, reveals a card (F2 peek result); a 6 (no ability)
    does not. Small enough that many infosets recur across seeds."""
    return _rules_cls()(
        deck_ranks=["6", "7"],
        use_jokers=0,
        cards_per_player=2,
        initial_view_count=1,
        max_game_turns=6,
        allowReplaceAbilities=False,
        allowDrawFromDiscardPile=False,
        allowOpponentSnapping=False,
    )


def _collect_prefix_masks(
    make_game, seeds, max_steps: int
) -> Tuple[Dict[Tuple[int, ...], frozenset], int, int, int]:
    """Play seeded self-play games; map each acting player's token prefix (its
    infoset) to the legal-action-index mask at that node. Returns the map plus
    (nodes, collisions, conflicts)."""
    prefix_to_mask: Dict[Tuple[int, ...], frozenset] = {}
    nodes = collisions = conflicts = 0
    for seed in seeds:
        game = make_game(seed)
        rng = random.Random(90_000 + seed)
        obs: Dict[int, List] = {p: [] for p in range(NUM_PLAYERS)}
        ih = {p: list(game.players[p].hand) for p in range(NUM_PLAYERS)}
        ip = {
            p: tuple(game.players[p].initial_peek_indices) for p in range(NUM_PLAYERS)
        }
        for _ in range(max_steps):
            if game.is_terminal():
                break
            actor = game.get_acting_player()
            if actor == -1:
                break
            legal = list(game.get_legal_actions())
            if not legal:
                break
            prefix = tuple(
                se.encode_observation_sequence(
                    ih[actor], ip[actor], obs[actor], actor, seq_cap=10**9,
                    add_bos_eos=False,
                )
            )
            mask = frozenset(action_to_index(a) for a in legal)
            nodes += 1
            prev = prefix_to_mask.get(prefix)
            if prev is None:
                prefix_to_mask[prefix] = mask
            else:
                collisions += 1
                if prev != mask:
                    conflicts += 1
            action = rng.choice(legal)
            game.apply_action(action)
            snaps = list(getattr(game, "snap_results_log", []) or [])
            full = _create_observation(None, action, game, actor, snaps)
            if full is None:
                continue
            for o in range(NUM_PLAYERS):
                obs[o].append(_filter_observation(full, o))
    return prefix_to_mask, nodes, collisions, conflicts


def test_b3_mask_determinism_ability_deck():
    """Identical token prefix => identical legal mask on the {6,7} ability deck,
    where post-draw masks depend on the drawn card. Requires real prefix
    collisions (asserted) so the determinism check has teeth, and asserts zero
    conflicts (the F1 mask-nondeterminism is gone)."""
    rules = _ability_rules()

    def make(seed):
        return CambiaGameState(house_rules=rules, _rng=random.Random(seed))

    _, nodes, collisions, conflicts = _collect_prefix_masks(make, range(400), 40)
    print(
        f"\n[b3 mask-determinism {{6,7}}] nodes={nodes} collisions={collisions} "
        f"conflicts={conflicts}"
    )
    assert nodes > 1000, f"too few decision nodes ({nodes})"
    assert collisions > 200, f"only {collisions} prefix collisions; check lacks teeth"
    assert conflicts == 0, (
        f"{conflicts} token prefixes admit two legal masks -- mask nondeterminism "
        f"(F1 regression: the drawn card is missing from the post-draw prefix)"
    )


def test_b3_mask_determinism_full_2p_games():
    """Breadth: identical token prefix => identical legal mask over full 2-player
    production games (all abilities present). Regression guard; asserts zero
    conflicts (collisions are rarer here since full-game infosets seldom recur)."""
    _, nodes, collisions, conflicts = _collect_prefix_masks(
        _setup_python_game_matching_go, range(120), 400
    )
    print(
        f"\n[b3 mask-determinism full-2p] nodes={nodes} collisions={collisions} "
        f"conflicts={conflicts}"
    )
    assert nodes > 1000, f"too few decision nodes ({nodes})"
    assert conflicts == 0, f"{conflicts} token prefixes admit two legal masks"


def test_b3_mask_determinism_race_on():
    """Race-ON (snapRace=True, cambia-564): identical token prefix => identical
    legal mask over full 2-player games. Under race-ON the per-commit public frame
    is suppressed (imperfect info) and the resolution emits public race frames, so
    a committer's infoset never reveals another committer's choice; the committer's
    own mask depends only on the public prefix (its hand + the discarded rank), so
    prefix->mask determinism must still hold. Regression guard; zero conflicts."""

    def make(seed):
        game = _setup_python_game_matching_go(seed)
        game.house_rules = _rules_cls()(
            allowDrawFromDiscardPile=True,
            allowOpponentSnapping=True,
            snapRace=True,
            max_game_turns=46,
        )
        return game

    _, nodes, collisions, conflicts = _collect_prefix_masks(make, range(120), 400)
    print(
        f"\n[b3 mask-determinism race-ON] nodes={nodes} collisions={collisions} "
        f"conflicts={conflicts}"
    )
    assert nodes > 1000, f"too few decision nodes ({nodes})"
    assert conflicts == 0, (
        f"{conflicts} race-ON token prefixes admit two legal masks -- imperfect-info "
        f"commit suppression must not make the legal mask prefix-nondeterministic"
    )


# ---------------------------------------------------------------------------
# (b) Tabular-CFR-on-production-tokens floor (slow / CLI-invokable)
# ---------------------------------------------------------------------------


def _iter_decisions(root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == "D":
            yield node
            stack.extend(node.children)
        elif node.kind == "C":
            stack.extend(node.children)


def _floor_rules():
    """Minimal ability game for the floor: 1 card each, unseen at deal
    (initial_view_count=0), deck {6, 9}. A 9 (peek-OTHER) reveals the opponent's
    hidden card -- strategically decisive for the Cambia call -- so tokenizing the
    peek result (F2) is what lets the token policy play it. Tiny enough to solve
    tabularly to convergence under a node cap."""
    return _rules_cls()(
        deck_ranks=["6", "9"],
        use_jokers=0,
        cards_per_player=1,
        initial_view_count=0,
        max_game_turns=3,
        allowReplaceAbilities=False,
        allowDrawFromDiscardPile=False,
        allowOpponentSnapping=False,
        cambia_allowed_round=0,
    )


def measure_token_floor(deals: int = 3, seed0: int = 0, iters: int = 120) -> dict:
    """Build a tokenized tree on the minimal peek game with the PRODUCTION worker
    observation path (production_obs=True, so the tokens carry the cambia-528
    post-draw drawn frame and the cambia-529 peek-result frames), run tabular CFR
    keyed on the production TOKEN stream, and measure the resulting policy's
    NashConv (self-consistent: the best responder is keyed on the same token
    infoset the agent conditions on). A low value proves the production tokens are
    a sound tabular-CFR infoset key -- the mask is a function of the prefix (F1)
    and the strategically decisive peek result is present (F2), so regret matching
    on the token key converges. Before the fixes the tokenizer dropped the peek
    (F2) and mis-timed the drawn frame (F1); rerun on the reverted tokenizer for
    the before/after floor."""
    import tools.tiny_solver as ts
    from src.config import load_config

    # A full tiny config (agent_params etc. for AgentState); rules overridden.
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "tiny_2card_plateau.yaml",
    )
    cfg = load_config(base)
    cfg.cambia_rules = _floor_rules()
    root, _iss, nnodes, aborted = ts.build_tree(
        cfg,
        deals,
        seed0,
        max_nodes_per_deal=400_000,
        enumerate_draws=True,
        perfect_recall=False,
        tokenize=True,
        seq_cap=10**9,
        production_obs=True,
    )
    assert not aborted, "floor tree hit the node cap; shrink deals or game"

    decisions = list(_iter_decisions(root))
    # Key every decision node on its production token stream.
    isets = {}
    for n in decisions:
        n.pkey = tuple(n.seq_tokens)
        isets[(n.pkey, len(n.actions))] = len(n.actions)

    solver = ts.TabularCFR(isets)
    for it in range(1, iters + 1):
        solver.iterate(root, it)
    nc_tokens, _ = ts.exploitability(root, solver.average_strategy())

    return {
        "nodes": nnodes,
        "n_tok_isets": len(isets),
        "nashconv_tokens": nc_tokens,
    }


@pytest.mark.slow
def test_b3_tabular_cfr_floor_on_production_tokens():
    """Tabular CFR keyed on the production token stream converges to a low
    NashConv: the tokens are a sound CFR infoset key (mask determined by the
    prefix per F1; strategically decisive peek result present per F2). The value
    is reported for the before/after comparison; the gate asserts convergence."""
    res = measure_token_floor()
    print(
        f"\n[b3 token floor] nodes={res['nodes']} tok_isets={res['n_tok_isets']} "
        f"NashConv(tokens)={res['nashconv_tokens']:.6e}"
    )
    # Converges well below the uniform-random baseline (~1.0 on the +/-1 scale);
    # a sound infoset key lets regret matching descend.
    assert res["nashconv_tokens"] < 0.05, (
        f"token-keyed CFR did not converge (NashConv={res['nashconv_tokens']:.4e}); "
        f"the production tokens may not be a sound CFR key"
    )
