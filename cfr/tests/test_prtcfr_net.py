"""Scoped tests for the PRT-CFR sequence net and the single-sourced token seam.

Covers:
  - net forward + output shape (GPU when available, else CPU);
  - regret-matching strategy: masking, normalization, uniform fallback;
  - parameter count (reported; the architecture is pinned by the interface
    contract, so the count is asserted against that exact architecture);
  - token parity: the same perfect-recall infoset tokenized via the traversal
    path and via a direct node lookup yields identical arrays.
"""

import numpy as np
import pytest
import torch

from src.config import load_config
from src.cfr.prtcfr_net import (
    PRTCFRNet,
    pad_tokens,
    tiny_node_to_token_array,
    tiny_node_to_tokens,
)
from src.encoding import NUM_ACTIONS
from src.sequence_encoding import PAD_ID, VOCAB_SIZE
from tools.tiny_solver import build_tree

CONFIG_2CARD = "config/tiny_2card_plateau.yaml"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_tree():
    """Tokenized perfect-recall tiny tree (built once for the module)."""
    cfg = load_config(CONFIG_2CARD)
    root, isets, nnodes, aborted = build_tree(
        cfg,
        n_deals=5,
        seed0=0,
        max_nodes_per_deal=2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=True,
        seq_cap=256,
    )
    assert aborted == 0, "tree truncated; raise max_nodes_per_deal"
    return root


def _all_decisions(node, out):
    if node.kind == "T":
        return
    if node.kind == "C":
        for c in node.children:
            _all_decisions(c, out)
        return
    out.append(node)
    for c in node.children:
        _all_decisions(c, out)


# ---------------------------------------------------------------------------
# Net forward / shape / strategy
# ---------------------------------------------------------------------------


def test_net_forward_shape():
    net = PRTCFRNet(device=_DEVICE)
    B, L = 8, 64
    tokens = torch.randint(0, VOCAB_SIZE, (B, L), device=net.device)
    mask = torch.zeros(B, NUM_ACTIONS, dtype=torch.bool, device=net.device)
    mask[:, :3] = True
    adv = net.raw_advantages(tokens, mask)
    assert adv.shape == (B, NUM_ACTIONS)
    assert adv.device.type == net.device.type


def test_strategy_regret_match_masking_and_normalization():
    net = PRTCFRNet(device=_DEVICE)
    B, L = 16, 40
    tokens = torch.randint(0, VOCAB_SIZE, (B, L), device=net.device)
    mask = torch.zeros(B, NUM_ACTIONS, dtype=torch.bool, device=net.device)
    mask[:, 0] = True
    mask[:, 5] = True
    mask[:, 9] = True
    strat = net.strategy_from_tokens(tokens, mask)
    assert strat.shape == (B, NUM_ACTIONS)
    # rows sum to 1
    row_sums = strat.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    # zero mass on illegal actions
    illegal = strat[~mask]
    assert torch.all(illegal == 0)


def test_strategy_uniform_fallback_when_all_advantages_nonpositive():
    net = PRTCFRNet(device=_DEVICE)
    # Force all-negative advantages so ReLU zeroes them and the fallback fires.
    adv = -torch.ones(2, NUM_ACTIONS, device=net.device)
    mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool, device=net.device)
    mask[:, 1] = True
    mask[:, 7] = True
    from src.cfr.prtcfr_net import _regret_match

    strat = _regret_match(adv, mask)
    # uniform over the two legal actions
    assert pytest.approx(0.5, abs=1e-6) == strat[0, 1].item()
    assert pytest.approx(0.5, abs=1e-6) == strat[0, 7].item()
    assert strat[0, 0].item() == 0.0


def test_param_count_matches_pinned_architecture():
    """The architecture is pinned (Embedding(326,64) -> GRU(64,256,2) -> LN;
    Linear(256,256)->ReLU->Linear(256,146)). That exact architecture is 766,738
    parameters. The vocab grew 325 -> 326 with the cambia-529 peek-result marker
    (embedding +64 params; 766,674 -> 766,738), which invalidates pre-cambia-529
    checkpoints -- acceptable, X4 is future work. The contract's "~1.2M" is an
    over-estimate; the pin (shared with the X2 scorer's checkpoint loader) is
    authoritative."""
    net = PRTCFRNet(device="cpu")
    n = net.num_parameters()
    # Embedding 326*64=20864; GRU(64,256,2)=642048; LayerNorm 512;
    # head 256*256+256 + 256*146+146 = 103314 -> 766738.
    assert n == 766_738, f"param count {n} != pinned-architecture count 766738"


def test_pad_tokens_widths_and_truncation():
    arr = pad_tokens([4, 5, 6], seq_cap=8)
    assert arr.dtype == np.int32
    assert arr.shape == (8,)
    assert arr.tolist() == [4, 5, 6, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
    # overflow keeps the most-recent tokens
    long = list(range(1, 21))
    arr2 = pad_tokens(long, seq_cap=5)
    assert arr2.tolist() == [16, 17, 18, 19, 20]


# ---------------------------------------------------------------------------
# Token parity (the critical seam)
# ---------------------------------------------------------------------------


def test_token_parity_traversal_vs_direct_lookup(tiny_tree):
    """The same infoset tokenized via a tree traversal that reaches a node and via
    a direct node lookup must yield identical arrays. Since seq_tokens is stored
    once on the node, the traversal and the lookup read the same object: parity is
    structural. This asserts it explicitly on every decision node."""
    decisions = []
    _all_decisions(tiny_tree, decisions)
    assert decisions
    for node in decisions:
        via_list = tiny_node_to_tokens(node)
        via_array = tiny_node_to_token_array(node)
        # The padded array's non-pad prefix must equal the list.
        assert via_array[: len(via_list)].tolist() == list(via_list)
        assert np.all(via_array[len(via_list) :] == PAD_ID)


def test_token_parity_same_pkey_same_tokens(tiny_tree):
    """Every perfect-recall infoset (pkey) reached by multiple tree nodes must map
    to a single token sequence. This is the train/eval parity guarantee: a worker
    sample and an eval query for the same infoset feed byte-identical tokens."""
    from collections import defaultdict

    decisions = []
    _all_decisions(tiny_tree, decisions)
    by_pkey = defaultdict(set)
    multi = 0
    for node in decisions:
        toks = tuple(tiny_node_to_tokens(node))
        by_pkey[node.pkey].add(toks)
    for pkey, variants in by_pkey.items():
        if len(variants) > 1:
            multi += 1
    assert multi == 0, f"{multi} perfect-recall infosets map to >1 token sequence"


def test_tiny_node_to_tokens_requires_tokenized_tree():
    """A node from a non-tokenized tree (seq_tokens is None) must raise, not
    silently feed an empty/garbage sequence."""
    cfg = load_config(CONFIG_2CARD)
    root, _isets, _n, _ab = build_tree(
        cfg,
        n_deals=2,
        seed0=0,
        max_nodes_per_deal=2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=False,
    )
    decisions = []
    _all_decisions(root, decisions)
    assert decisions
    with pytest.raises(ValueError, match="seq_tokens"):
        tiny_node_to_tokens(decisions[0])
