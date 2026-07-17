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

import numpy as np
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
from src.constants import ActionCallCambia  # noqa: E402
from src.encoding import NUM_ACTIONS  # noqa: E402
from src.sequence_encoding import BOS_ID, EOS_ID  # noqa: E402


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
    condition 3's "real multi-hundred-token sequences" requirement.

    S1W11 fixed a snap-token-duplication bug that used to inflate sequential
    play lengths; on the corrected engine, the "natural" cohort (CallCambia
    played as soon as it is legal) terminates at ~69 tokens mean / p99 253 --
    too short to reliably clear a several-hundred-token bar. This helper
    instead drives the "avoid_cambia" cohort (mirrors
    scripts/prtcfr_p100_instrument.py and tests/test_sequence_tokenizer.py's
    established stress convention): never select ActionCallCambia while any
    other legal action exists, so the game runs to its natural
    non-Cambia-terminated length, bounded only by the engine's own 300-turn
    cap -- the same cohort observed reaching several thousand tokens. A single
    seed clears any realistic ``min_len`` under this policy; the seed loop and
    failure report below are a safety net, not the expected path."""
    longest = 0
    for seed in range(200):
        # Explicitly the Python stub backend: this helper mutates .seq_cap
        # directly to force an uncapped-window generation run, a
        # PythonEngineGameDriver-specific affordance (new_production_driver's
        # S1W13 default is Go-backed).
        driver = new_production_driver(seed=seed, backend="python")
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
            pool = legal
            non_cambia = [a for a in legal if not isinstance(a, ActionCallCambia)]
            if non_cambia:
                pool = non_cambia
            mask = _legal_mask(pool)
            probs = uniform_policy_production([], mask)
            try:
                _sample_and_apply(driver, pool, probs, rng)
            except RuntimeError:
                break
            toks = driver.tokens(0)
            if len(toks) >= min_len:
                return toks
        toks = driver.tokens(0)
        longest = max(longest, len(toks))
    pytest.fail(
        f"could not reach a real token sequence of length >= {min_len} tokens "
        f"across 200 seeds even under the avoid_cambia policy (longest single "
        f"game observed: {longest} tokens); this would indicate a real "
        f"regression in game length, not a test-tuning issue -- investigate "
        f"before lowering min_len."
    )


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


# ---------------------------------------------------------------------------
# cambia-472: vectorized _pad_batch + fused advance_query equivalence
# ---------------------------------------------------------------------------


def test_pad_batch_vectorized_matches_naive():
    """The vectorized (numpy-scatter) _pad_batch must reproduce the pre-fix
    per-stream reference grid exactly: right-pad to Lmax with PAD_ID, empty rows
    clamped to a single PAD (length 1). Ragged widths, an empty row, and a
    single-token row all exercised."""
    from src.sequence_encoding import PAD_ID

    net = _small_net()
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
    token_lists = [[BOS_ID, 5, 6, 7], [9], [], [BOS_ID, 3, 4], [2, 2]]

    batch, lengths = service._pad_batch(token_lists)

    lmax = max(max((len(t) for t in token_lists), default=0), 1)
    ref = np.full((len(token_lists), lmax), PAD_ID, dtype=np.int64)
    for i, toks in enumerate(token_lists):
        if toks:
            ref[i, : len(toks)] = np.asarray(toks, dtype=np.int64)
    assert np.array_equal(batch.cpu().numpy(), ref)
    assert lengths.tolist() == [max(1, len(t)) for t in token_lists]


def _drive_two_paths(dtype):
    """Drive an identical multi-tick (sids, news, transients, masks) schedule
    through the sequential advance()+query_transient() pair and the fused
    advance_query(), returning per-tick (ref_strat, fused_strat) and the final
    carried-hidden dict for each service. The schedule mixes: fresh (absent,
    non-empty new) streams, continuing streams (non-empty new), present
    empty-new streams (re-query from carry, no persist), an absent empty-new
    stream (defensive zero-hidden query path), ragged frame widths, and a
    fork/drop, so every advance_query branch is exercised."""
    net = _small_net()
    svc_ref = PRTCFRInferenceService(net, device="cpu", dtype=dtype)
    svc_fused = PRTCFRInferenceService(net, device="cpu", dtype=dtype)
    rng = random.Random(20472)
    seen: set = set()

    def _mask_tensor(masks_np):
        return torch.from_numpy(np.stack(masks_np))

    pairs = []
    saw_empty_present = False
    saw_absent = False
    for tick in range(12):
        active = sorted(rng.sample(range(9), rng.randint(2, 6)))
        sids, news, transients, masks_np = [], [], [], []
        for sid in active:
            if sid not in seen:
                # Fresh stream: first frame must be non-empty ([BOS]+body), as
                # the production manager's tokens[:-1] always is on first sight.
                new = [BOS_ID] + [rng.randint(3, 40) for _ in range(rng.randint(0, 3))]
                seen.add(sid)
                saw_absent = True
            else:
                # Continuing stream: 0-3 new body frames (0 -> empty-new re-query).
                k = rng.randint(0, 3)
                new = [rng.randint(3, 40) for _ in range(k)]
                if k == 0:
                    saw_empty_present = True
            sids.append(sid)
            news.append(new)
            transients.append([EOS_ID])
            m = np.zeros(NUM_ACTIONS, dtype=bool)
            legal = rng.sample(range(NUM_ACTIONS), rng.randint(1, 5))
            m[legal] = True
            masks_np.append(m)

        # One defensive absent-empty-new stream on a middle tick: absent stream
        # (never seen) with empty new -> query from a zero hidden, no persist.
        if tick == 5:
            sids.append(99)
            news.append([])
            transients.append([EOS_ID])
            m = np.zeros(NUM_ACTIONS, dtype=bool)
            m[:3] = True
            masks_np.append(m)

        masks = _mask_tensor(masks_np)

        svc_ref.advance(sids, news)
        ref = svc_ref.query_transient(sids, transients, masks.clone())
        fused = svc_fused.advance_query(sids, news, transients, masks.clone())
        pairs.append((ref.detach().cpu().numpy(), fused.detach().cpu().numpy()))

        # A fork + drop mid-run, applied identically to both, so the store's
        # slot reuse and fork clone interact with the fused path.
        if tick == 7:
            for svc in (svc_ref, svc_fused):
                svc.fork(active[0], 200 + active[0])
            for svc in (svc_ref, svc_fused):
                svc.drop(active[-1])
            seen.discard(active[-1])

    carry_ref = {s: svc_ref._hidden[s].clone() for s in svc_ref._hidden._slot}
    carry_fused = {s: svc_fused._hidden[s].clone() for s in svc_fused._hidden._slot}
    return pairs, carry_ref, carry_fused, saw_empty_present, saw_absent


def test_advance_query_fused_matches_sequential_fp32():
    """Fused advance_query() equals sequential advance()+query_transient()
    bit-close in fp32 (the GRU recurrence over new+transient from h0 has the
    same top hidden as a transient step from the advanced carry), at every tick
    and for the final carried hidden state of every stream."""
    pairs, carry_ref, carry_fused, saw_empty, saw_absent = _drive_two_paths(
        torch.float32
    )
    assert saw_empty and saw_absent, "schedule did not exercise both branches"
    for i, (ref, fused) in enumerate(pairs):
        assert ref.shape == fused.shape
        d = float(np.abs(ref - fused).max())
        assert d < 1e-6, f"tick {i}: fused strategy diverged from sequential: {d:.2e}"
    assert set(carry_ref) == set(carry_fused)
    for s in carry_ref:
        d = (carry_ref[s] - carry_fused[s]).abs().max().item()
        assert d < 1e-6, f"stream {s}: carried hidden diverged: {d:.2e}"


def test_advance_query_fused_matches_sequential_bf16():
    """Same fused-vs-sequential equivalence in bf16, within bf16 tolerance
    (well under the 5e-6 eval-wrapper precedent's spirit; bf16 round-off on the
    O(1) regret-matched probabilities is the only gap, and the persist carry is
    bit-identical since both paths run the identical GRU op)."""
    pairs, carry_ref, carry_fused, _, _ = _drive_two_paths(torch.bfloat16)
    for i, (ref, fused) in enumerate(pairs):
        d = float(np.abs(ref - fused).max())
        assert d < 1e-3, f"tick {i}: fused bf16 strategy diverged: {d:.2e}"
    assert set(carry_ref) == set(carry_fused)
    for s in carry_ref:
        assert torch.equal(carry_ref[s], carry_fused[s]), (
            f"stream {s}: bf16 carried hidden not bit-identical across paths"
        )
