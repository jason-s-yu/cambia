"""X2 gate: trained PRT-CFR neural policy NashConv < 0.05 on the tiny {A,6} game.

Two kinds of test:

  1. test_x2_gate_trained_checkpoint -- THE gate. Parametrized by the
     ``PRTCFR_X2_CHECKPOINT`` env var (a single .pt checkpoint OR a directory of
     prtcfr_snapshot_iter_{t}.pt SD-CFR snapshots). Loads the policy, scores it on
     the perfect-recall {A,6} tree, asserts NashConv < 0.05. SKIPS when the env
     var is unset (no trained artifact yet), so it never red-bars the suite
     before integration. @chief sets PRTCFR_X2_CHECKPOINT to the trained
     snapshot dir to run the real verdict.

  2. test_x2_plumbing_* -- ALWAYS run. Drive the full
     enumerate -> tokenize -> SD-CFR average -> exploitability path on RANDOM-INIT
     PRTCFRNet(s) and assert exploitability returns a finite number. A random
     policy is far from the 0.05 bar (expected); this proves the pipeline wiring,
     not the bound.

Parallel-dev seam
-----------------
Core (prtcfr-core) owns src/cfr/prtcfr_net.py (PRTCFRNet + tiny_node_to_tokens).
It has landed, so these tests run against the REAL net. If that module is ever
absent (true parallel dev), this file injects a faithful LOCAL stub into
sys.modules BEFORE importing the eval module, matching the real contract:
PRTCFRNet.strategy_from_tokens(tokens[B,L], mask[B,146]) -> [B,146], and
tiny_node_to_tokens(node) -> node.seq_tokens. The injection is guarded: it fires
only when the real module is absent, so the real symbols win automatically once
core is present.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Parallel-dev stub for src.cfr.prtcfr_net (only if core has not landed).
# ---------------------------------------------------------------------------


def _core_available() -> bool:
    try:
        import importlib

        m = importlib.import_module("src.cfr.prtcfr_net")
        return hasattr(m, "PRTCFRNet") and hasattr(m, "tiny_node_to_tokens")
    except Exception:  # noqa: BLE001
        return False


_USING_STUB = not _core_available()


def _install_stub_prtcfr_net() -> None:
    """Install a stub of core's prtcfr_net into sys.modules, matching the real
    contract. Used only when src/cfr/prtcfr_net.py is absent (true parallel dev).

    tiny_node_to_tokens returns node.seq_tokens (the builder must be run with
    tokenize=True, which prtcfr_eval.build_tiny_tree does). PRTCFRNet is a small
    random-init torch module whose strategy_from_tokens(tokens[B,L], mask[B,146])
    returns a [B,146] regret-matched distribution restricted to the mask.
    """
    import torch
    import torch.nn.functional as F

    from src.encoding import NUM_ACTIONS
    from src.sequence_encoding import PAD_ID, VOCAB_SIZE

    def tiny_node_to_tokens(node, seq_cap: int = 256):
        toks = getattr(node, "seq_tokens", None)
        if toks is None:
            raise ValueError(
                "stub tiny_node_to_tokens: node.seq_tokens is None; build the tree "
                "with build_tree(..., tokenize=True)"
            )
        return list(toks[-seq_cap:]) if len(toks) > seq_cap else list(toks)

    class PRTCFRNet(torch.nn.Module):
        def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            embed_dim: int = 16,
            num_actions: int = NUM_ACTIONS,
            device="cpu",
            **_,
        ):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
            self.head = torch.nn.Linear(embed_dim, num_actions)
            self.num_actions = num_actions
            self.device = (
                torch.device(device) if device is not None else torch.device("cpu")
            )
            self.to(self.device)

        def raw_advantages(self, tokens, mask=None):
            if tokens.dtype != torch.long:
                tokens = tokens.long()
            emb = self.embed(tokens)  # (B, L, E)
            pooled = emb.mean(dim=1)  # (B, E)
            return self.head(pooled)  # (B, 146)

        def forward(self, tokens, mask=None):
            return self.raw_advantages(tokens, mask)

        @torch.no_grad()
        def strategy_from_tokens(self, tokens, mask):
            adv = self.raw_advantages(tokens, mask)
            positive = F.relu(adv) * mask.float()
            total = positive.sum(dim=-1, keepdim=True)
            uniform = mask.float()
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return torch.where(total > 0, positive / total.clamp(min=1e-10), uniform)

    stub = types.ModuleType("src.cfr.prtcfr_net")
    stub.PRTCFRNet = PRTCFRNet
    stub.tiny_node_to_tokens = tiny_node_to_tokens
    stub.__prtcfr_stub__ = True
    sys.modules["src.cfr.prtcfr_net"] = stub


if _USING_STUB:
    _install_stub_prtcfr_net()


# Import AFTER any stub injection so prtcfr_eval binds the right symbols.
from src.cfr import prtcfr_eval  # noqa: E402
from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402  (real or stub)


def _make_random_net(seed: int = 0):
    import torch

    torch.manual_seed(seed)
    net = PRTCFRNet(device="cpu")
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Plumbing tests (always run)
# ---------------------------------------------------------------------------


def test_x2_plumbing_random_net():
    """Full enumerate -> tokenize -> SD-CFR average -> exploitability path on a
    random-init PRTCFRNet returns a FINITE NashConv. Proves the pipeline, not
    the bound: a random policy is far from 0.05.
    """
    np.random.seed(0)
    net = _make_random_net(seed=0)

    # One snapshot at iter=1 (single-net SD-CFR average == that net's strategy).
    result = prtcfr_eval.score_with_loaded_nets([(1, net)])

    nashconv = result["nashconv"]
    assert np.isfinite(nashconv), f"NashConv not finite: {nashconv!r}"
    assert all(np.isfinite(x) for x in result["components"]), result["components"]
    assert (
        result["num_infosets"] > 100
    ), f"expected >100 PR infosets on the {{A,6}} tree, got {result['num_infosets']}"
    # A random-init policy is nowhere near equilibrium; EXPECTED. Documents that
    # this test proves plumbing, not the gate bound.
    assert nashconv > prtcfr_eval.X2_NASHCONV_BAR, (
        f"random-init policy unexpectedly scored NashConv={nashconv:.6e} below the "
        f"gate bar {prtcfr_eval.X2_NASHCONV_BAR}; a meaningful bound test cannot pass "
        f"with a random net -- inspect the scorer"
    )


def test_x2_plumbing_multi_snapshot_average():
    """SD-CFR linear-weighted average over multiple random snapshots yields a
    finite NashConv and a normalized policy. Exercises the multi-net averaging
    path the real verdict uses with a trained snapshot directory.

    Uses ``materialize_policy_incremental`` (not ``materialize_policy``): at
    production net dims (embed=64, hidden=256) and the real {A,6} tree's
    69636 infosets, a single-batch forward per net OOMs (RSS in the tens of
    GB -- see IncrementalPolicyAccumulator's docstring); the chunked
    accumulator is numerically equivalent and is what the real gate path
    (score_with_loaded_nets / score_policy_on_tiny_game) now uses.
    """
    nets = [(it, _make_random_net(seed=it)) for it in (10, 20, 30)]

    root, _isets, _n, _ab = prtcfr_eval.build_tiny_tree()
    policy = prtcfr_eval.materialize_policy_incremental(root, nets, weighting="linear")

    # Every materialized strategy is a valid distribution over its legal actions.
    for pkey, vec in policy.items():
        assert np.all(vec >= -1e-9), f"negative prob in {pkey!r}: {vec}"
        assert abs(vec.sum() - 1.0) < 1e-6, f"{pkey!r} sums to {vec.sum()}"

    from tools.tiny_solver import exploitability

    nashconv, components = exploitability(root, policy)
    assert np.isfinite(nashconv)
    assert all(np.isfinite(x) for x in components)


def test_incremental_policy_matches_materialize_policy_small_tree():
    """Equivalence gate: materialize_policy_incremental (chunked, additive
    src code) reproduces materialize_policy (the single-batch reference)
    exactly, on a tree small enough that the reference path itself is safe
    to run (the full {A,6} tree's 69636 infosets at production net dims is
    NOT -- that single-batch forward is exactly what OOMs; see
    IncrementalPolicyAccumulator's docstring).

    Small tree: 1 deal, draws NOT enumerated (realized-deck-order draws
    only) -- 14346 nodes / 4433 perfect-recall infosets, versus the real
    gate's 230206 / 69636. Same net dims (production defaults), same
    weighting, same seq_cap, 3 snapshots (exercises the SD-CFR multi-net
    accumulation path, not just the single-net case) plus a deliberately
    small ``chunk_size`` (67 -- not a divisor of 4433) so the chunked path
    actually exercises multiple partial-final-chunk batches rather than one
    chunk covering everything.
    """
    from src.config import load_config
    from tools.tiny_solver import build_tree

    cfg = load_config(prtcfr_eval.TINY_2CARD_CONFIG)
    root, _isets, nnodes, aborted = build_tree(
        cfg, 1, 0, 2_000_000, enumerate_draws=False,
        perfect_recall=True, tokenize=True, seq_cap=256,
    )
    assert aborted == 0
    assert nnodes < 20_000, f"small-tree fixture grew unexpectedly: {nnodes} nodes"

    nets = [(it, _make_random_net(seed=it)) for it in (5, 15, 25)]

    reference = prtcfr_eval.materialize_policy(root, nets, weighting="linear", seq_cap=256)
    incremental = prtcfr_eval.materialize_policy_incremental(
        root, nets, weighting="linear", seq_cap=256, chunk_size=67
    )

    assert set(reference.keys()) == set(incremental.keys())
    max_abs_diff = 0.0
    for pkey, ref_vec in reference.items():
        inc_vec = incremental[pkey]
        assert inc_vec.shape == ref_vec.shape
        max_abs_diff = max(max_abs_diff, float(np.abs(ref_vec - inc_vec).max()))
    # Chunking only changes which rows share a batched matmul call; no layer
    # in PRTCFRNet mixes across the batch dimension (GRU is per-row
    # recurrent, LayerNorm normalizes per-row over features), so results
    # match up to float32 matmul-reordering noise (the GRU forward itself
    # runs in float32; only the outer SD-CFR accumulation is float64) --
    # NOT a chunking bug. Empirically ~1.8e-6 on this fixture (single-
    # precision, ~7 significant digits). 1e-4 is far tighter than any real
    # divergence would be: a genuine chunking bug (e.g. an off-by-one at a
    # chunk boundary, a dropped/duplicated row) would show up as an O(1)
    # difference (a whole strategy entry wrong), not float32-noise-scale.
    assert max_abs_diff < 1e-4, (
        f"materialize_policy_incremental diverged from materialize_policy: "
        f"max abs diff {max_abs_diff:.3e} across {len(reference)} infosets"
    )


def test_x2_tokens_are_single_sourced():
    """The scorer's tokens come from node.seq_tokens (the builder's single source),
    not a reimplemented tokenizer. Guards the RC-B train/eval parity invariant:
    every enumerated infoset has a non-empty token stream identical to what
    tiny_node_to_tokens returns.
    """
    from src.cfr.prtcfr_net import tiny_node_to_tokens

    root, _isets, _n, _ab = prtcfr_eval.build_tiny_tree()
    nodes = prtcfr_eval.enumerate_infosets(root)
    assert nodes, "no infosets enumerated"
    for nd in nodes:
        toks = list(tiny_node_to_tokens(nd))
        src = list(nd.seq_tokens)
        assert toks, f"empty token stream for pkey {nd.pkey!r}"
        # The helper returns node.seq_tokens verbatim, or (only when the source
        # exceeds seq_cap) its most-recent-seq_cap suffix. Tiny-game streams are
        # well under the cap, so this is exact equality here; the suffix branch
        # documents the single overflow contract.
        assert toks == src or toks == src[-len(toks) :], (
            f"tiny_node_to_tokens diverged from node.seq_tokens (parity seam "
            f"broken) for pkey {nd.pkey!r}"
        )


def test_x2_tokens_are_cross_pkey_injective():
    """Distinct perfect-recall pkeys map to DISTINCT token streams.

    test_x2_tokens_are_single_sourced only checks per-node round-trip (the helper
    returns node.seq_tokens verbatim); it never compares tokens ACROSS pkeys. This
    is the gap that hid the post-draw collision bug: the acting player's freshly
    drawn stockpile card was missing from the token stream at the post-draw
    decision node, so genuinely distinct perfect-recall infosets shared
    byte-identical tokens. Pre-fix: 354 colliding groups (737 pkeys), 77 with
    divergent equilibria (max L1 1.33), capping NashConv at 0.0794 > the 0.05 bar.
    Post-fix: 0 collisions. A regressed scorer would otherwise feed identical net
    inputs to distinct infosets and silently re-impose the floor.
    """
    from collections import defaultdict

    from tools.tiny_solver import build_tree
    from src.config import load_config

    cfg = load_config("config/tiny_2card_plateau.yaml")
    root, _isets, _n, _ab = build_tree(
        cfg,
        5,
        0,
        2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=True,
        seq_cap=256,
    )

    # One representative token stream per distinct perfect-recall pkey. Every node
    # sharing a pkey has the same tokens (the acting player's perfect-recall stream
    # is a function of the pkey); we assert that consistency, then check that the
    # pkey -> tokens map is injective (no two distinct pkeys share a stream).
    reps: dict = {}
    stack = [root]
    while stack:
        nd = stack.pop()
        kind = nd.kind
        if kind == "T":
            continue
        if kind == "C":
            stack.extend(nd.children)
            continue
        assert (
            nd.seq_tokens is not None
        ), f"decision node pkey {nd.pkey!r} has no seq_tokens (built without tokenize?)"
        toks = tuple(nd.seq_tokens)
        prev = reps.get(nd.pkey)
        if prev is None:
            reps[nd.pkey] = toks
        else:
            assert prev == toks, (
                f"same pkey {nd.pkey!r} yielded two distinct token streams; the "
                f"perfect-recall stream is not a function of the pkey"
            )
        stack.extend(nd.children)

    assert reps, "no decision nodes enumerated"

    inv: dict = defaultdict(set)
    for pkey, toks in reps.items():
        inv[toks].add(pkey)
    colliding = {toks: pkeys for toks, pkeys in inv.items() if len(pkeys) > 1}

    assert not colliding, (
        f"{len(colliding)} token streams are shared by >1 distinct perfect-recall "
        f"pkey ({sum(len(p) for p in colliding.values())} pkeys collide); distinct "
        f"infosets would receive identical PRT-CFR net inputs, re-imposing the "
        f"NashConv floor. Sample collisions: "
        f"{[sorted(map(repr, p))[:2] for p in list(colliding.values())[:3]]}"
    )


# ---------------------------------------------------------------------------
# X2 gate (runs only with a trained artifact)
# ---------------------------------------------------------------------------

_X2_CHECKPOINT_ENV = "PRTCFR_X2_CHECKPOINT"


@pytest.mark.skipif(
    not os.environ.get(_X2_CHECKPOINT_ENV),
    reason=(
        f"set {_X2_CHECKPOINT_ENV} to a trained PRT-CFR checkpoint (.pt) or a "
        f"directory of prtcfr_snapshot_iter_*.pt snapshots to run the X2 verdict"
    ),
)
def test_x2_gate_trained_checkpoint():
    """X2 gate: trained PRT-CFR policy NashConv < 0.05 on the {A,6} tiny game.

    @chief runs this by exporting PRTCFR_X2_CHECKPOINT=<trained snapshot dir>.
    Uses the REAL PRTCFRNet/tiny_node_to_tokens; if the stub is somehow active
    the run is skipped to avoid a false verdict.
    """
    if _USING_STUB:
        pytest.skip(
            "src.cfr.prtcfr_net not landed; refusing to score the X2 gate against "
            "the stub net (false verdict). Re-run after integration."
        )

    checkpoint = os.environ[_X2_CHECKPOINT_ENV]
    weighting = os.environ.get("PRTCFR_X2_WEIGHTING", "linear")
    result = prtcfr_eval.score_policy_on_tiny_game(checkpoint, weighting=weighting)

    nashconv = result["nashconv"]
    assert np.isfinite(nashconv), f"NashConv not finite: {nashconv!r}"
    assert nashconv < prtcfr_eval.X2_NASHCONV_BAR, (
        f"X2 GATE FAILED: NashConv={nashconv:.6e} >= {prtcfr_eval.X2_NASHCONV_BAR} "
        f"on the {{A,6}} tiny game.\n"
        f"  snapshots: {result['num_snapshots']} (iters {result['snapshot_iters']})\n"
        f"  infosets:  {result['num_infosets']}\n"
        f"  components (br0,br1,onp0,onp1): {result['components']}\n"
        f"  checkpoint: {checkpoint}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
