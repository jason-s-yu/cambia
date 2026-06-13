"""Scoped tests for the tiny-solver perfect-recall keying mode (X1 keystone).

Covers the two correctness properties the X1 verdict depends on:

  1. Imperfect-recall mode is UNCHANGED: every decision node's policy key is
     still the production belief-abstraction key (InfosetKey.astuple()).
  2. Perfect-recall mode keys are WELL FORMED and DISTINCT per distinct history:
       - every decision key is the "PR" tuple shape,
       - the key determines the legal-action count (cures BUG-1: the production
         belief key does NOT, which is the documented plateau cause),
       - the key determines the legal action set (no two histories that a
         perfect-recall player can tell apart share a key with different moves),
       - perfect recall is a strict refinement of the belief partition (it
         produces strictly more distinct decision keys on a game where the
         belief abstraction is known to alias).

These run on the 1-card config (sub-2s tree build) so the test is cheap; the
keystone NashConv measurement itself is a CLI run, not a unit test.
"""

import random

import pytest

from src.config import load_config
from src.game.engine import CambiaGameState
from src.agent_state import AgentState
from src.analysis_tools import AnalysisTools
from tools.tiny_solver import build_tree, Builder, _mk_agent


CONFIG_1CARD = "config/tiny_norecall.yaml"


def _walk_decisions(node, acc):
    """Collect every Decision node in the explicit tree."""
    if node.kind == "T":
        return
    if node.kind == "C":
        for c in node.children:
            _walk_decisions(c, acc)
        return
    acc.append(node)
    for c in node.children:
        _walk_decisions(c, acc)


def _build(perfect_recall):
    cfg = load_config(CONFIG_1CARD)
    root, isets, nnodes, aborted = build_tree(
        cfg, n_deals=5, seed0=0, max_nodes_per_deal=2_000_000,
        enumerate_draws=True, perfect_recall=perfect_recall,
    )
    assert aborted == 0, "tree truncated; raise max_nodes_per_deal"
    decisions = []
    _walk_decisions(root, decisions)
    assert decisions, "expected at least one decision node"
    return root, isets, decisions


def test_imperfect_recall_keying_unchanged():
    """Default mode: pkey is exactly the production belief key (astuple)."""
    _, _, decisions = _build(perfect_recall=False)
    for nd in decisions:
        # The imperfect-recall key is the InfosetKey.astuple() carried on .iset.
        assert nd.pkey == nd.iset.astuple()
        # And it is NOT the perfect-recall tuple shape.
        assert not (isinstance(nd.pkey, tuple) and nd.pkey and nd.pkey[0] == "PR")


def test_perfect_recall_key_shape():
    """Perfect-recall mode tags every decision key with the PR discriminator
    and the four-part (tag, init, draws, public-path) structure."""
    _, _, decisions = _build(perfect_recall=True)
    for nd in decisions:
        assert isinstance(nd.pkey, tuple) and len(nd.pkey) == 4
        assert nd.pkey[0] == "PR"
        # init prefix and draw/public sequences are tuples (hashable, dict-key safe)
        assert isinstance(nd.pkey[1], tuple)
        assert isinstance(nd.pkey[2], tuple)
        assert isinstance(nd.pkey[3], tuple)


def test_perfect_recall_key_determines_action_count():
    """BUG-1 fix: a perfect-recall key maps to exactly one legal-action count.
    This is the property the production belief key violates (~8% of keys / 37%
    of visits carry a varying nA), and the proximate cause of the plateau."""
    _, _, decisions = _build(perfect_recall=True)
    key_to_nas = {}
    for nd in decisions:
        key_to_nas.setdefault(nd.pkey, set()).add(len(nd.actions))
    offenders = {k: v for k, v in key_to_nas.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} perfect-recall keys map to multiple action counts; "
        f"key does not determine nA. Example: {next(iter(offenders.items()))}"
    )


def test_perfect_recall_key_determines_legal_set():
    """Stronger than the nA check: a perfect-recall key maps to exactly one
    legal action SET (by repr). Two histories sharing a key are genuinely
    indistinguishable to the acting player, so their legal moves must match."""
    _, _, decisions = _build(perfect_recall=True)
    key_to_sets = {}
    for nd in decisions:
        sig = tuple(repr(a) for a in nd.actions)
        key_to_sets.setdefault(nd.pkey, set()).add(sig)
    offenders = {k: v for k, v in key_to_sets.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} perfect-recall keys map to multiple legal action sets; "
        f"distinct-history collision. Example: {next(iter(offenders.items()))}"
    )


def test_perfect_recall_also_determines_acting_player():
    """A perfect-recall info state belongs to one player; the key must not be
    shared across players (that would be a cross-player history collision)."""
    _, _, decisions = _build(perfect_recall=True)
    key_to_players = {}
    for nd in decisions:
        key_to_players.setdefault(nd.pkey, set()).add(nd.player)
    offenders = {k: v for k, v in key_to_players.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} perfect-recall keys span multiple acting players. "
        f"Example: {next(iter(offenders.items()))}"
    )


def test_perfect_recall_is_strict_refinement_of_belief_partition():
    """Perfect recall must produce strictly more distinct decision keys than the
    belief abstraction on the 1-card game (where the belief key is known to
    alias distinct private histories). Confirms the refinement is non-trivial."""
    _, _, dec_imperfect = _build(perfect_recall=False)
    _, _, dec_perfect = _build(perfect_recall=True)
    n_imperfect = len({nd.pkey for nd in dec_imperfect})
    n_perfect = len({nd.pkey for nd in dec_perfect})
    assert n_perfect > n_imperfect, (
        f"perfect-recall keys ({n_perfect}) not a strict refinement of belief "
        f"keys ({n_imperfect})"
    )


def test_perfect_recall_distinguishes_distinct_dealt_hands_at_root():
    """Targeted regression for the code-map deviation: an own-action-only key
    would collapse every deal's first decision to the empty history. The genuine
    perfect-recall key carries the player's peeked initial hand, so distinct
    dealt hands at the root get distinct keys. Build two single-deal trees whose
    seeds yield different dealt hands and assert their root decision keys differ
    iff the acting player's peeked card differs."""
    cfg = load_config(CONFIG_1CARD)
    root_keys = {}  # peeked-init -> set of root-decision pkeys
    for seed in range(8):
        b = Builder(cfg, 2_000_000, enumerate_draws=True, perfect_recall=True)
        game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(seed))
        init_obs = AnalysisTools._create_observation_for_br(game, None, -1)
        ag = {0: _mk_agent(game, 0, 1, cfg, init_obs),
              1: _mk_agent(game, 1, 0, cfg, init_obs)}
        for pid in (0, 1):
            peeks = tuple(
                (i, repr(game.players[pid].hand[i]))
                for i in sorted(game.players[pid].initial_peek_indices)
                if i < len(game.players[pid].hand)
            )
            b.priv_init[pid] = peeks
        sub = b.build_decision_or_terminal(game, ag, 0)
        # find the first decision node (the root acting player's first move)
        first = sub
        while first.kind == "C":
            first = first.children[0]
        if first.kind != "D":
            continue
        acting = first.player
        peeked = b.priv_init[acting]
        root_keys.setdefault(peeked, set()).add(first.pkey)
    # Each distinct peeked initial hand must yield exactly one root-decision key
    # (no aliasing within a hand) and distinct hands must yield distinct keys.
    all_keys = set()
    for peeked, keys in root_keys.items():
        assert len(keys) == 1, f"peeked hand {peeked} aliases {len(keys)} root keys"
        all_keys |= keys
    assert len(all_keys) == len(root_keys), (
        "distinct dealt hands collapsed to the same root key (own-action-only bug)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
