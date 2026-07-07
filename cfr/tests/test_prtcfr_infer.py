"""tests/test_prtcfr_infer.py

Scoped tests for PRTCFRInferenceService (src/cfr/prtcfr_infer.py), the batched
incremental GRU inference service for PRT-CFR production policy queries
(S1W3 stage 2).

The binding test here (v0.4 Phase 2 window-semantics decision, sign-off
condition 3): incremental bf16 hidden-state carry must match a full batch
re-encode within bf16 tolerance, on REAL multi-hundred-token sequences (not
synthetic short ones) -- this is what makes "incremental carry conditions on
the exact full prefix" true rather than an approximation.
"""

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_infer import PRTCFRInferenceService  # noqa: E402
from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402
from src.cfr.prtcfr_worker import (
    new_production_driver,
    _legal_mask,
    _sample_and_apply,
    uniform_policy_production,
)  # noqa: E402
from src.encoding import NUM_ACTIONS  # noqa: E402
from src.sequence_encoding import BOS_ID  # noqa: E402


def _small_net() -> PRTCFRNet:
    # Small dims for test speed; vocab/action space stay at production size
    # since token ids and the action head width are fixed by the tokenizer
    # and encoding.NUM_ACTIONS respectively.
    return PRTCFRNet(
        embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        head_hidden_dim=16,
        dropout=0.0,  # carry-vs-batch equivalence requires deterministic (no-dropout) eval
        device="cpu",
    )


def _real_long_token_sequence(min_len: int = 300) -> list:
    """Play a real full game via the production driver until player 0's
    full-recall token stream is at least ``min_len`` tokens; return it. Uses
    the actual tokenizer output (not synthetic ids), matching sign-off
    condition 3's "real multi-hundred-token sequences" requirement."""
    for seed in range(200):
        driver = new_production_driver(seed=seed)
        driver.seq_cap = 20000
        rng = random.Random(seed)
        for _ in range(3000):
            if driver.is_terminal():
                break
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)
            probs = uniform_policy_production([], mask)
            try:
                _sample_and_apply(driver, legal, probs, rng)
            except RuntimeError:
                break
            toks = driver.tokens(0)
            if len(toks) >= min_len:
                return toks
    pytest.fail("could not reach a real token sequence of the required length")


# ---------------------------------------------------------------------------
# Carry-vs-batch equivalence (sign-off condition 3)
# ---------------------------------------------------------------------------


def test_incremental_carry_matches_full_batch_reencode_bf16():
    """register()+step() through a REAL multi-hundred-token sequence, chunked
    a few tokens at a time (mimicking one-frame-per-tick production usage),
    must produce the same raw GRU hidden state (bf16 tolerance) as running the
    GRU over the whole sequence in one pack_padded pass from a zero initial
    hidden state, in the SAME (bf16) precision."""
    net = _small_net()
    tokens = _real_long_token_sequence(min_len=300)
    assert len(tokens) >= 300

    # Incremental: register with a short prefix, then step through the rest in
    # small chunks (frame-sized: 3-4 tokens), exactly the production usage
    # pattern (append one observation frame, advance hidden state once).
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)

    # Reference: one-shot bf16 GRU forward over the full sequence (no padding
    # needed, batch size 1, exact length), using the SAME bf16-cast weights
    # the service holds internally (service.net) -- apples-to-apples: same
    # model, same dtype, only the call pattern (one-shot vs incremental)
    # differs.
    with torch.no_grad():
        tok_tensor = torch.tensor([tokens], dtype=torch.long)
        emb = service.net.encoder["embed"](tok_tensor)
        _out, h_n_ref = service.net.encoder["gru"](emb)
    ref_top = h_n_ref[-1, 0, :].to(torch.float32)
    prefix_len = 6
    service.register("s0", tokens[:prefix_len])
    i = prefix_len
    chunk_sizes = [4, 3, 5, 2]
    ci = 0
    while i < len(tokens):
        size = chunk_sizes[ci % len(chunk_sizes)]
        ci += 1
        chunk = tokens[i : i + size]
        if not chunk:
            break
        service.step(["s0"], [chunk])
        i += size

    carry_top = service._hidden["s0"][-1, :].to(torch.float32)

    max_abs_diff = (ref_top - carry_top).abs().max().item()
    # bf16 has ~3 decimal digits of precision (2^-8 relative ~ 0.4%); allow a
    # generous but meaningful tolerance well below "clearly wrong" magnitude
    # (hidden state entries are tanh/sigmoid-gated, roughly O(1)).
    assert max_abs_diff < 0.05, (
        f"incremental carry diverged from full-batch re-encode: "
        f"max abs diff {max_abs_diff:.6f} (ref={ref_top[:5]}, carry={carry_top[:5]})"
    )


def test_incremental_carry_bf16_close_to_fp32_full_encode():
    """Precision sanity check (looser than the strict same-dtype equivalence
    test above): the service's bf16 incremental advantages should stay close
    to a full-precision (fp32) one-shot encode of the identical prefix via the
    ORIGINAL uncast net -- bounding how much the bf16 cast itself (not the
    incremental-vs-batch call pattern) can move the output."""
    net = _small_net()
    tokens = _real_long_token_sequence(min_len=200)

    with torch.no_grad():
        tok_tensor = torch.tensor([tokens], dtype=torch.long)
        ref_hidden = net.encode(tok_tensor)  # (1, hidden), post-LayerNorm, fp32
        ref_adv = net.head(ref_hidden)  # (1, 146)

    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    service.register("s1", tokens[:8])
    i = 8
    while i < len(tokens):
        chunk = tokens[i : i + 4]
        service.step(["s1"], [chunk])
        i += 4

    carry_adv = service.advantages(["s1"], masks=None)

    diff = (ref_adv.float() - carry_adv.float()).abs().max().item()
    assert diff < 0.5, f"advantages diverged too far: {diff:.4f}"


# ---------------------------------------------------------------------------
# Basic API sanity
# ---------------------------------------------------------------------------


def test_register_then_step_shapes():
    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    h = service.register("a", [BOS_ID, 5, 6])
    assert h.shape == (net.num_layers, net.hidden_dim)

    top = service.step(["a"], [[7, 8]])
    assert top.shape == (1, net.hidden_dim)


def test_step_batches_multiple_streams():
    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    service.register("a", [BOS_ID, 5])
    service.register("b", [BOS_ID, 9, 10])
    top = service.step(["a", "b"], [[6, 7], [11]])
    assert top.shape == (2, net.hidden_dim)


def test_step_unregistered_stream_raises_keyerror():
    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(KeyError):
        service.step(["nope"], [[1, 2]])


def test_strategy_is_masked_and_sums_to_one():
    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    service.register("a", [BOS_ID, 5, 6])
    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    mask[0, :5] = True
    probs = service.strategy(["a"], mask)
    assert probs.shape == (1, NUM_ACTIONS)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-4)
    assert torch.all(probs[0, 5:] == 0)


def test_drop_removes_stream():
    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.bfloat16)
    service.register("a", [BOS_ID])
    assert len(service) == 1
    service.drop("a")
    assert len(service) == 0
