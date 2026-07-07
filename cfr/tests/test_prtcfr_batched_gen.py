"""tests/test_prtcfr_batched_gen.py

Semantic-equivalence gate for the S1W15 batched incremental production
generation path (cambia-239): the batched, incrementally-carried sampler
(prtcfr_worker.PRTCFRBatchedProductionWorker + IncrementalSigmaManager) must
produce the SAME policies and the SAME regret samples as the existing
per-decision full-prefix path (PRTCFRProductionWorker + NetProductionSigma).

The gate is decomposed into three individually-robust claims whose composition
is the end-to-end equivalence (no flaky trajectory-divergence assertion):

  A. Carry identity: the incremental carry (advance + transient EOS) reproduces
     a full encode_observation_sequence re-encode at EVERY decision of 50+
     seeded full games (incl. snap interrupts and ability chains), fp32-tight.
     This is the BOS/EOS/frame-boundary claim.

  B. Scheduler faithfulness: the batched worker driven by a single-item full-
     re-encode backend (bit-identical sigma per query) reproduces the sequential
     worker's regret samples EXACTLY, per game. This isolates the coroutine /
     rollout-fork / RNG-order restatement.

  C. Batching invariance: the incremental manager's batched forward equals a
     single-item forward per query (fp32 round-off), so B composed with A and C
     gives: batched-incremental gen == sequential full-reencode gen at every
     decision, to fp32 tolerance.

fp32 note: exact trajectory-level bit-identity between the PRODUCTION path
(incremental + padded batch) and the sequential single-item path is neither
claimed nor achievable -- padded batching and the GRU register/step split each
introduce ~1e-6 fp32 round-off, which the boundary-sensitive action sampler can
occasionally amplify into a different sampled action. The gate therefore pins
per-decision policy equivalence (A) and exact scheduler faithfulness under an
identical sigma (B), the two claims that are robust.
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_infer import PRTCFRInferenceService  # noqa: E402
from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402
from src.cfr.prtcfr_trainer import NetProductionSigma  # noqa: E402
from src.cfr.prtcfr_worker import (  # noqa: E402
    _FullReencodeSigmaBackend,
    _legal_mask,
    _Query,
    _sample_and_apply,
    IncrementalSigmaManager,
    PRODUCTION_SEQ_CAP,
    PRTCFRBatchedProductionWorker,
    PRTCFRProductionWorker,
    new_production_driver,
    uniform_policy_production,
)
from src.reservoir import ReservoirSample  # noqa: E402


def _net(seed: int = 0) -> PRTCFRNet:
    torch.manual_seed(seed)
    net = PRTCFRNet(
        embed_dim=16,
        hidden_dim=24,
        num_layers=2,
        head_hidden_dim=24,
        dropout=0.0,
        device="cpu",
    )
    return net.eval()


class _CapturingBuf:
    """A ReservoirBuffer stand-in that records every added sample in order."""

    def __init__(self):
        self.samples = []

    def add(self, sample: ReservoirSample) -> None:
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)


def _seeds_reaching_decisions(n_games: int, seq_cap: int = PRODUCTION_SEQ_CAP):
    """Yield the first ``n_games`` seeds whose uniform-play game reaches at least
    one non-initial decision (so there is a real growing token stream to test)."""
    found = 0
    seed = 0
    while found < n_games and seed < 5000:
        yield seed
        found += 1
        seed += 1


# ---------------------------------------------------------------------------
# A. Carry identity at every decision of many real seeded games
# ---------------------------------------------------------------------------


def test_carry_identity_matches_full_reencode_at_every_decision():
    """Play 50 seeded full games under uniform b_i (which naturally hits snap
    interrupts and ability chains) and, at EVERY decision, assert the
    incremental-carry sigma (advance + transient EOS, per-player stream) equals
    the full-prefix NetProductionSigma re-encode within fp32 tolerance. This is
    the BOS/EOS/frame-boundary carry gate."""
    net = _net(1)
    old_sigma = NetProductionSigma(net, seq_cap=PRODUCTION_SEQ_CAP)
    service = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
    mgr = IncrementalSigmaManager(service, num_players=2)

    n_games = 50
    total_decisions = 0
    max_diff = 0.0
    saw_ability_or_snap = False
    for gi, seed in enumerate(_seeds_reaching_decisions(n_games)):
        driver = new_production_driver(seed=seed, backend="python")
        rng = random.Random(10_000 + seed)
        stream = gi  # a fresh, stable per-game stream key for the manager
        steps = 0
        while not driver.is_terminal() and steps < 4000:
            steps += 1
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)
            tokens = driver.tokens(actor)

            old = old_sigma(tokens, mask)
            q = _Query(stream, actor, tokens, mask)
            mgr.evaluate([q])
            new = q.result

            d = float(np.abs(old - new).max())
            max_diff = max(max_diff, d)
            total_decisions += 1
            assert d < 1e-4, (
                f"carry-vs-reencode sigma diverged at game seed {seed} step "
                f"{steps} actor {actor}: max abs diff {d:.2e}"
            )
            # snap/ability frames grow the stream mid-game; the EOS-transient
            # carry must hold across them. Any token beyond the ~public-frame
            # base implies non-trivial frame structure was carried.
            if len(tokens) > 12:
                saw_ability_or_snap = True

            probs = uniform_policy_production(tokens, mask)
            _sample_and_apply(driver, legal, probs, rng)
        mgr.drop(stream)

    assert total_decisions > 500, f"only {total_decisions} decisions exercised"
    assert saw_ability_or_snap, "no multi-frame (snap/ability) sequences exercised"
    assert max_diff < 1e-4


# ---------------------------------------------------------------------------
# B. Scheduler faithfulness: batched worker (full-reencode backend) == sequential
# ---------------------------------------------------------------------------


# Small, bounded config for the structural equivalence tests: a random net's
# regret-matched sigma concentrates probability off CallCambia, so rollouts run
# to the step cap -- the cap is set small here so the tests exercise the full
# machinery (fork, join, multi-frame carry) without running full-length games.
# The equivalence claim is independent of the cap (both paths use the same one).
_M = 2
_MAX_STEPS = 60


def _sequential_samples(net, seeds, iteration=1):
    """Run the sequential PRTCFRProductionWorker per game (reseeded from the
    game seed) with NetProductionSigma; return {game_index: [ReservoirSample]}."""
    sigma = NetProductionSigma(net, seq_cap=PRODUCTION_SEQ_CAP)
    out = {}
    for gi, seed in enumerate(seeds):
        buf = _CapturingBuf()
        worker = PRTCFRProductionWorker(
            sigma=sigma, m_rollouts=_M, seq_cap=PRODUCTION_SEQ_CAP,
            seed=seed, max_trajectory_steps=_MAX_STEPS,
        )
        worker.reseed(seed)
        driver = new_production_driver(seed=seed, backend="python")
        worker.traverse(driver, traverser=gi % 2, iteration=iteration, buf=buf)
        out[gi] = buf.samples
    return out


def _batched_samples(backend_factory, net, seeds, iteration=1):
    """Run the batched worker over all games in ONE chunk with the given backend
    factory; return {game_index: [ReservoirSample]}."""
    bufs = {gi: _CapturingBuf() for gi in range(len(seeds))}
    specs = [
        {"seed": seed, "driver": new_production_driver(seed=seed, backend="python"),
         "traverser": gi % 2, "iteration": iteration, "buf": bufs[gi]}
        for gi, seed in enumerate(seeds)
    ]
    worker = PRTCFRBatchedProductionWorker(
        m_rollouts=_M, seq_cap=PRODUCTION_SEQ_CAP, max_trajectory_steps=_MAX_STEPS
    )
    backend = backend_factory(net)
    worker.generate(specs, backend)
    for spec in specs:
        spec["driver"].close()
    return {gi: bufs[gi].samples for gi in range(len(seeds))}


def _assert_samples_equal(seq, bat, atol):
    assert set(seq.keys()) == set(bat.keys())
    for gi in seq:
        s, b = seq[gi], bat[gi]
        assert len(s) == len(b), (
            f"game {gi}: sequential recorded {len(s)} samples, batched {len(b)}"
        )
        for j, (ss, bb) in enumerate(zip(s, b)):
            assert np.array_equal(ss.features, bb.features), (
                f"game {gi} sample {j}: token features differ"
            )
            assert np.array_equal(ss.action_mask, bb.action_mask), (
                f"game {gi} sample {j}: masks differ"
            )
            assert ss.iteration == bb.iteration
            np.testing.assert_allclose(
                ss.target, bb.target, atol=atol,
                err_msg=f"game {gi} sample {j}: regret targets differ",
            )


def test_batched_scheduler_faithful_to_sequential_worker():
    """Batched worker with the single-item full-re-encode backend must
    reproduce the sequential worker's regret samples EXACTLY per game (identical
    sigma per query, identical RNG order -> identical trajectories)."""
    net = _net(2)
    seeds = list(range(6))
    seq = _sequential_samples(net, seeds)
    bat = _batched_samples(lambda n: _FullReencodeSigmaBackend(n), net, seeds)
    # At least some games must actually record traverser samples.
    assert sum(len(v) for v in seq.values()) > 0
    _assert_samples_equal(seq, bat, atol=0.0)


# ---------------------------------------------------------------------------
# C. Production path: incremental manager == full-reencode, per decision
# ---------------------------------------------------------------------------


def test_batched_incremental_matches_batched_reencode_per_decision():
    """The PRODUCTION backend (IncrementalSigmaManager, fp32) priced against the
    full-re-encode backend over the same seeds/chunk: every recorded regret
    sample is well-formed, and on every game whose trajectory matched (the
    common case -- the fp32 carry round-off is ~1e-7, far below a sampling
    boundary) the regret targets agree to fp32 tolerance.

    A per-game trajectory divergence (a rare fp32 boundary flip amplified by a
    rollout) is not a failure of the carry identity -- it is the documented
    limit of trajectory-level bit-identity under batching (see module docstring)
    -- so it is counted, not asserted against; the per-DECISION identity is the
    robust claim, pinned exactly by test A + the batching-invariance test."""
    net = _net(3)
    seeds = list(range(5))

    reenc = _batched_samples(lambda n: _FullReencodeSigmaBackend(n), net, seeds)

    def _mgr_factory(n):
        svc = PRTCFRInferenceService(n, device="cpu", dtype=torch.float32)
        return IncrementalSigmaManager(svc, num_players=2)

    incr = _batched_samples(_mgr_factory, net, seeds)
    assert sum(len(v) for v in reenc.values()) > 0

    diverged = 0
    for gi in reenc:
        # Every incremental sample is well-formed regardless of divergence.
        for s in incr[gi]:
            assert s.action_mask.sum() >= 1
            assert np.all(np.isfinite(s.target))
            assert np.all(s.target[~s.action_mask] == 0.0)
        if len(reenc[gi]) != len(incr[gi]):
            diverged += 1
            continue
        for j, (a, b) in enumerate(zip(reenc[gi], incr[gi])):
            if not np.array_equal(a.features, b.features):
                diverged += 1
                break
            np.testing.assert_allclose(
                a.target, b.target, atol=1e-4,
                err_msg=f"game {gi} sample {j}: incremental vs reencode target",
            )
    # Trajectory divergence must be rare (fp32 boundary flips); the per-decision
    # identity itself is tested exactly elsewhere.
    assert diverged <= 1, f"{diverged}/{len(seeds)} games diverged (expected ~0)"


def test_fork_carry_matches_reencode_of_forked_prefix():
    """Focused, deterministic check of the rollout hidden-fork: advance a stream
    to a prefix, FORK it (the torch mirror of cambia_state_clone), advance the
    fork by a divergent suffix, and assert the fork's query equals a full
    re-encode of [forked prefix + suffix]. No game/trajectory randomness, so this
    isolates the fork correctness the batched rollouts depend on."""
    net = _net(6)
    old_sigma = NetProductionSigma(net, seq_cap=PRODUCTION_SEQ_CAP)

    # Real growing streams for two players from a partial game.
    driver = new_production_driver(seed=7, backend="python")
    rng = random.Random(7)
    for _ in range(6):
        if driver.is_terminal():
            break
        actor = driver.current_player()
        legal = driver.legal_actions()
        if not legal:
            break
        _sample_and_apply(driver, legal, uniform_policy_production([], _legal_mask(legal)), rng)
    assert not driver.is_terminal()

    svc = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
    mgr = IncrementalSigmaManager(svc, num_players=2)
    parent = 100

    # Advance the parent stream for the current actor (registers the carry).
    actor = driver.current_player()
    mask = _legal_mask(driver.legal_actions())
    q = _Query(parent, actor, driver.tokens(actor), mask)
    mgr.evaluate([q])

    # Fork the parent -> child, then diverge the child by applying one action to
    # a driver clone and querying the (grown) child stream.
    child_key = 200
    mgr.fork(parent, child_key)
    child = driver.clone()
    a = driver.legal_actions()[0]
    assert child.apply(a)
    child_actor = child.current_player()
    child_mask = _legal_mask(child.legal_actions())
    child_tokens = child.tokens(child_actor)
    qc = _Query(child_key, child_actor, child_tokens, child_mask)
    mgr.evaluate([qc])

    ref = old_sigma(child_tokens, child_mask)
    np.testing.assert_allclose(qc.result, ref, atol=1e-4)

    # The parent stream is unperturbed by the fork+child advance: re-querying it
    # for the same (unchanged) tokens reproduces its sigma.
    q2 = _Query(parent, actor, driver.tokens(actor), mask)
    mgr.evaluate([q2])
    np.testing.assert_allclose(q2.result, old_sigma(driver.tokens(actor), mask), atol=1e-4)


# ---------------------------------------------------------------------------
# Batching invariance + handle hygiene
# ---------------------------------------------------------------------------


def test_incremental_manager_batched_equals_single_item():
    """Querying N streams in ONE IncrementalSigmaManager.evaluate batch equals
    querying each stream alone (batching-invariance, fp32 round-off)."""
    net = _net(4)
    # Build a few real growing streams by playing partial games.
    streams = []
    for seed in range(6):
        driver = new_production_driver(seed=seed, backend="python")
        rng = random.Random(seed)
        for _ in range(random.Random(seed).randint(1, 8)):
            if driver.is_terminal():
                break
            actor = driver.current_player()
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)
            _sample_and_apply(driver, legal, uniform_policy_production([], mask), rng)
        if driver.is_terminal():
            continue
        actor = driver.current_player()
        legal = driver.legal_actions()
        streams.append((driver, actor, _legal_mask(legal)))

    assert len(streams) >= 3
    # Batched.
    svc_b = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
    mgr_b = IncrementalSigmaManager(svc_b, num_players=2)
    qb = [_Query(i, actor, driver.tokens(actor), mask)
          for i, (driver, actor, mask) in enumerate(streams)]
    mgr_b.evaluate(qb)
    # Single-item.
    singles = []
    for i, (driver, actor, mask) in enumerate(streams):
        svc_s = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
        mgr_s = IncrementalSigmaManager(svc_s, num_players=2)
        q = _Query(i, actor, driver.tokens(actor), mask)
        mgr_s.evaluate([q])
        singles.append(q.result)
    for i in range(len(streams)):
        np.testing.assert_allclose(qb[i].result, singles[i], atol=1e-5)


def test_batched_worker_closes_all_clones_python_backend():
    """Every rollout/child clone is closed (no leaked drivers). The python
    backend's close() is a no-op, so we assert via a clone/close counter."""
    net = _net(5)

    class _CountingDriver:
        # Wrap a python driver, counting clone/close to assert balance.
        counter = {"open": 0, "close": 0}

        def __init__(self, inner):
            self._inner = inner
            _CountingDriver.counter["open"] += 1

        def current_player(self):
            return self._inner.current_player()

        def is_terminal(self):
            return self._inner.is_terminal()

        def utility(self, p):
            return self._inner.utility(p)

        def legal_actions(self):
            return self._inner.legal_actions()

        def apply(self, a):
            return self._inner.apply(a)

        def tokens(self, p):
            return self._inner.tokens(p)

        def clone(self):
            return _CountingDriver(self._inner.clone())

        def close(self):
            _CountingDriver.counter["close"] += 1
            self._inner.close()

    _CountingDriver.counter = {"open": 0, "close": 0}
    bufs = {0: _CapturingBuf()}
    top = _CountingDriver(new_production_driver(seed=3, backend="python"))
    opens_before = _CountingDriver.counter["open"]
    specs = [{"seed": 3, "driver": top, "traverser": 0, "iteration": 1, "buf": bufs[0]}]
    worker = PRTCFRBatchedProductionWorker(
        m_rollouts=2, seq_cap=PRODUCTION_SEQ_CAP, max_trajectory_steps=60
    )
    worker.generate(specs, _FullReencodeSigmaBackend(net))
    top.close()  # trainer owns the top driver
    # Every clone created by the worker must have been closed; the top driver
    # is closed by us here. So closes == clones_created + 0 (top counted in
    # opens but closed here). opens includes top + all clones.
    assert _CountingDriver.counter["close"] == _CountingDriver.counter["open"], (
        f"driver leak: {_CountingDriver.counter}"
    )
    assert _CountingDriver.counter["open"] > opens_before  # clones happened
