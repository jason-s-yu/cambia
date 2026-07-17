"""tests/test_prtcfr_provenance_gate.py

Token-stream provenance gate for the PRT-CFR tiny-game scorer (cambia-612).

Two concerns, both guarding the validity of every X2/X4 gate number:

  1. VERSION GATE (src/cfr/prtcfr_eval.py): a checkpoint recorded under one
     tokenizer version must never be scored under another. The scorer reads the
     run's recorded tokenizer_version (run_meta.json) and hard-errors on a
     mismatch with the live sequence_encoding.TOKENIZER_VERSION; an unknown
     (unstamped, pre-cambia-612) version warns loudly and proceeds on the legacy
     path rather than firing spuriously.

  2. OBS SOURCE (tools/tiny_solver.build_tree production_obs): from tokenizer v2
     on, production training tokens carry peek-result (F2) and post-draw drawn
     (F1) frames. The scorer must build its tree through the PRODUCTION worker
     observation path (production_obs=True) so its tokens match training; the
     legacy analysis_tools BR path drops peeked cards and would score a
     representation the net never trained on.
"""

from __future__ import annotations

import json
import os
import random

import pytest

import src.cfr.prtcfr_eval as prtcfr_eval
import src.sequence_encoding as se
import tools.tiny_solver as ts
from src.cfr.worker import _create_observation, _filter_observation
from src.constants import NUM_PLAYERS
from src.game.engine import CambiaGameState

_TINY_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "tiny_2card_plateau.yaml",
)


# ---------------------------------------------------------------------------
# (1) Version gate: _resolve_scoring_obs_path + _recorded_tokenizer_version
# ---------------------------------------------------------------------------


def test_resolve_matching_version_selects_production_path():
    """recorded == live (v3) -> no error, production_obs=True (live >= 2)."""
    assert se.TOKENIZER_VERSION == 3  # guards the constants below
    assert prtcfr_eval._resolve_scoring_obs_path(se.TOKENIZER_VERSION) is True


def test_resolve_mismatch_hard_errors_naming_both_versions():
    """recorded != live -> ValueError naming both versions and the fix."""
    with pytest.raises(ValueError) as ei:
        prtcfr_eval._resolve_scoring_obs_path(se.TOKENIZER_VERSION - 1)
    msg = str(ei.value)
    assert "MISMATCH" in msg
    assert f"v{se.TOKENIZER_VERSION - 1}" in msg and f"v{se.TOKENIZER_VERSION}" in msg
    assert "training-era" in msg  # names the remedy


def test_resolve_unknown_warns_and_takes_legacy_path():
    """recorded is None (pre-cambia-612 runs) -> loud warning, legacy path
    (production_obs=False), NOT a spurious hard error."""
    with pytest.warns(UserWarning, match="PROVENANCE UNKNOWN"):
        assert prtcfr_eval._resolve_scoring_obs_path(None) is False


def test_recorded_version_read_from_run_meta(tmp_path):
    """_recorded_tokenizer_version reads tokenizer_version from the nearest
    ancestor run_meta.json, from a checkpoint file, a snapshot dir, or the run
    dir itself; returns None when absent or unstamped."""
    run_dir = tmp_path / "v0.4-prtcfr-run"
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"tokenizer_version": 3}))
    ckpt = snap_dir / "prtcfr_snapshot_iter_1.pt"
    ckpt.write_bytes(b"")

    assert prtcfr_eval._recorded_tokenizer_version(str(run_dir)) == 3
    assert prtcfr_eval._recorded_tokenizer_version(str(snap_dir)) == 3
    assert prtcfr_eval._recorded_tokenizer_version(str(ckpt)) == 3

    # No run_meta anywhere -> unknown.
    assert prtcfr_eval._recorded_tokenizer_version(str(tmp_path / "nope.pt")) is None

    # run_meta present but no tokenizer_version field (a pre-cambia-612 run) -> None.
    old_dir = tmp_path / "legacy-run"
    old_dir.mkdir()
    (old_dir / "run_meta.json").write_text(json.dumps({"run_id": 7}))
    assert prtcfr_eval._recorded_tokenizer_version(str(old_dir)) is None


def test_score_path_entrypoints_hard_error_on_recorded_mismatch(tmp_path):
    """Path-based entry points read run_meta and refuse a version-mismatched
    checkpoint BEFORE loading it (the assert fires ahead of _load_net)."""
    run_dir = tmp_path / "mismatched-run"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(
        json.dumps({"tokenizer_version": se.TOKENIZER_VERSION - 1})
    )
    (run_dir / "prtcfr_snapshot_iter_1.pt").write_bytes(b"")  # never loaded

    for entry in (
        prtcfr_eval.score_policy_on_tiny_game,
        prtcfr_eval.certify_policy_on_tiny_game,
    ):
        with pytest.raises(ValueError, match="MISMATCH"):
            entry(str(run_dir))


def test_score_with_loaded_nets_hard_errors_on_mismatch():
    """score_with_loaded_nets applies the same gate via its tokenizer_version
    arg; a mismatch raises before any tree build or net use (empty nets list
    proves the assert precedes scoring)."""
    with pytest.raises(ValueError, match="MISMATCH"):
        prtcfr_eval.score_with_loaded_nets([], tokenizer_version=se.TOKENIZER_VERSION - 1)


def test_trainer_resume_guard_rejects_stale_vocab_checkpoint():
    """The trainer resume/load path refuses a checkpoint whose token embedding
    predates the current vocab with a clear tokenizer-version error, not the raw
    torch size mismatch that F1/F2/F3 vintage checkpoints otherwise trigger deep
    in load_state_dict (cambia-612 work item 2, applied to the trainer loader)."""
    torch = pytest.importorskip("torch")
    from src.cfr.prtcfr_trainer import (
        PRTCFRResumeError,
        _assert_resume_tokenizer_compatible,
    )

    stale = {"encoder_state_dict": {"embed.weight": torch.zeros(325, 4)}}
    with pytest.raises(PRTCFRResumeError, match="TOKENIZER-VERSION MISMATCH") as ei:
        _assert_resume_tokenizer_compatible(stale)
    msg = str(ei.value)
    assert "325" in msg and str(se.VOCAB_SIZE) in msg  # names both vintages

    # A current-vocab checkpoint passes the guard untouched.
    ok = {"encoder_state_dict": {"embed.weight": torch.zeros(se.VOCAB_SIZE, 4)}}
    _assert_resume_tokenizer_compatible(ok)


# ---------------------------------------------------------------------------
# (1b) Entry points route provenance -> production_obs (wiring, net-free)
# ---------------------------------------------------------------------------


class _StopSpy(Exception):
    pass


def _spy_build_tiny_tree(monkeypatch):
    """Replace prtcfr_eval.build_tiny_tree with a spy that records production_obs
    and short-circuits (raising _StopSpy) so the wiring is asserted without the
    full 230k-node tree build + scoring."""
    captured = {}

    def spy(*args, **kwargs):
        captured["production_obs"] = kwargs.get("production_obs")
        raise _StopSpy

    monkeypatch.setattr(prtcfr_eval, "build_tiny_tree", spy)
    return captured


def test_score_with_loaded_nets_routes_matching_version_to_production_obs(monkeypatch):
    captured = _spy_build_tiny_tree(monkeypatch)
    with pytest.raises(_StopSpy):
        prtcfr_eval.score_with_loaded_nets([], tokenizer_version=se.TOKENIZER_VERSION)
    assert captured["production_obs"] is True


def test_score_with_loaded_nets_routes_unknown_version_to_legacy_obs(monkeypatch):
    captured = _spy_build_tiny_tree(monkeypatch)
    with pytest.warns(UserWarning, match="PROVENANCE UNKNOWN"):
        with pytest.raises(_StopSpy):
            prtcfr_eval.score_with_loaded_nets([], tokenizer_version=None)
    assert captured["production_obs"] is False


# ---------------------------------------------------------------------------
# (2) Obs source: scorer tokens == production worker tokens on a peek fixture
# ---------------------------------------------------------------------------

_SEED = 0
_DRAWN_MARKER = se.FRAME_BASE + se._FRAME_TO_LOCAL["drawn"]  # F1 post-draw frame


def _peek_ability_cfg():
    """Tiny peek fixture: a single-rank {7} deck (7 = peek-own). Every stockpile
    draw is a 7, so drawing then discarding with ability is always legal (a
    post-draw decision -> F1) and always reveals a card (peek-result frame -> F2).
    1 card each / max 3 turns keeps the enumerated tree small. Rules override the
    loaded tiny config; the real CambiaRulesConfig is used (via type(cfg.cambia_rules))
    so the reduced-deck override is honored (see the B3 gate's note on the stub)."""
    from src.config import load_config

    cfg = load_config(_TINY_CFG)
    rules_cls = type(cfg.cambia_rules)
    cfg.cambia_rules = rules_cls(
        deck_ranks=["7"],
        use_jokers=0,
        cards_per_player=1,
        initial_view_count=0,
        max_game_turns=3,
        allowReplaceAbilities=False,
        allowDrawFromDiscardPile=False,
        allowOpponentSnapping=False,
    )
    return cfg


def _build_scorer_tree(cfg, production_obs):
    """Build the scorer tree the way build_tiny_tree does (perfect_recall +
    tokenize) on the peek fixture. enumerate_draws=False -> no chance nodes below
    the deal root, so each decision's children map 1:1 to its (sorted) actions and
    an action path replays deterministically on a same-seed game."""
    root, _isets, _n, aborted = ts.build_tree(
        cfg,
        n_deals=1,
        seed0=_SEED,
        max_nodes_per_deal=200_000,
        enumerate_draws=False,
        perfect_recall=True,
        tokenize=True,
        seq_cap=10**9,
        production_obs=production_obs,
    )
    assert not aborted, "peek fixture tree hit the node cap; shrink the game"
    return root


def _find_decision_node(root, predicate):
    """DFS the scorer tree, tracking the action path (list of GameActions) from the
    deal root. Return (path, node) for the first decision node satisfying predicate,
    else None. Children[i] corresponds to node.actions[i] (build_tree appends one
    child per sorted legal action)."""

    def dfs(node, path):
        if node.kind == "T":
            return None
        if node.kind == "C":
            for ch in node.children:
                got = dfs(ch, path)
                if got:
                    return got
            return None
        if predicate(node):
            return (list(path), node)
        for act, ch in zip(node.actions, node.children):
            got = dfs(ch, path + [act])
            if got:
                return got
        return None

    return dfs(root, [])


def _worker_tokens_for_path(cfg, path):
    """Ground truth: replay `path` on a fresh same-seed game through the PRODUCTION
    worker pipeline (worker._create_observation/_filter_observation +
    encode_observation_sequence) and return the token stream of the player acting
    at the node reached. Actions are matched by repr against the live legal set so
    no tree-captured object is reapplied stale. Independent of tiny_solver."""
    game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(_SEED))
    obs = {p: [] for p in range(NUM_PLAYERS)}
    ih = {p: list(game.players[p].hand) for p in range(NUM_PLAYERS)}
    ip = {p: tuple(game.players[p].initial_peek_indices) for p in range(NUM_PLAYERS)}
    for want in path:
        actor = game.get_acting_player()
        legal = list(game.get_legal_actions())
        match = next((a for a in legal if repr(a) == repr(want)), None)
        assert match is not None, f"replay diverged: {want!r} not legal at step"
        game.apply_action(match)
        snaps = list(getattr(game, "snap_results_log", []) or [])
        full = _create_observation(None, match, game, actor, snaps)
        if full is not None:
            for o in range(NUM_PLAYERS):
                obs[o].append(_filter_observation(full, o))
    target = game.get_acting_player()
    # add_bos_eos defaults True in encode_observation_sequence; tiny_solver's
    # _encode_seq uses that default, so match it for a byte-for-byte comparison.
    return target, tuple(
        se.encode_observation_sequence(
            ih[target], ip[target], obs[target], target, seq_cap=10**9
        )
    )


def test_scorer_production_obs_tokens_equal_production_worker_tokens():
    """production_obs=True scorer-tree seq_tokens are byte-identical to the
    production worker tokens for the same trajectory, on a node reached AFTER a
    peek (F2) and a post-draw decision (F1). This is the representation the
    PRT-CFR net trains on; scoring on anything else is the RC-B train/eval
    mismatch this gate prevents."""
    cfg = _peek_ability_cfg()
    root = _build_scorer_tree(cfg, production_obs=True)

    # A decision node whose token prefix carries BOTH a peek frame (F2) and a
    # post-draw drawn frame (F1) -- the fixture guarantees such nodes exist.
    found = _find_decision_node(
        root,
        lambda nd: se.PEEK_FRAME_BASE in (nd.seq_tokens or ())
        and _DRAWN_MARKER in (nd.seq_tokens or ()),
    )
    assert found is not None, (
        "no decision node carries both a peek (F2) and post-draw drawn (F1) frame; "
        "production_obs=True is not producing the production representation"
    )
    path, node = found

    target_actor, worker_tokens = _worker_tokens_for_path(cfg, path)
    assert target_actor == node.player, "replay reached a different acting player"
    assert worker_tokens == tuple(node.seq_tokens), (
        "scorer-tree tokens diverge from production worker tokens on the same "
        "trajectory; the scorer is not building through the production obs path"
    )


def test_scorer_legacy_obs_drops_peek_frames():
    """The legacy analysis_tools BR path (production_obs=False) nulls peeked cards,
    so NO node on the same fixture carries a peek-result frame -- the exact silent
    representation gap the v>=2 production_obs flip closes. (production_obs=True is
    proven to carry peek frames by the equality test above.)"""
    cfg = _peek_ability_cfg()
    root = _build_scorer_tree(cfg, production_obs=False)
    peek = _find_decision_node(
        root, lambda nd: se.PEEK_FRAME_BASE in (nd.seq_tokens or ())
    )
    assert peek is None, "legacy BR path unexpectedly carries a peek-result frame"
