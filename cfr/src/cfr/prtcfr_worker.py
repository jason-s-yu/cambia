"""src/cfr/prtcfr_worker.py

PRT-CFR external-sampling traversal with m-rollout Monte-Carlo regret targets,
on the tiny_solver explicit tree.

Estimator loop (standard external-sampling MCCFR, Lanctot 2009 / OpenSpiel
external_sampling_mccfr, with v0.4 design decision 2's rollout-based child
values and SD-CFR averaging):

  - recurse ALL of the TRAVERSER's actions (so every traverser infoset on the
    sampled opponent/chance line is visited and its regret collected);
  - SAMPLE one action at OPPONENT and CHANCE nodes (opponent from sigma^t, chance
    from the chance weights);
  - at a traverser node, each child's value q_hat(a) is the m-rollout Monte-Carlo
    estimate under sigma^t, CRN-paired across the node's children (decision 2; no
    critic in the regret path). E[q_hat] = q for any m >= 1 (the unbiasedness
    property the MC-target test verifies against exact tree values);
  - regret target r_hat(a) = q_hat(a) - sum_a' sigma(a') q_hat(a'), 0 for illegal;
    append ReservoirSample(tokens(h) int32 width seq_cap, r_hat float32(146),
    mask bool(146), iteration=t).

No average-strategy buffer is kept: SD-CFR realizes the average exactly from the
per-iteration regret-net snapshots (decision 4), so opponent infosets record
nothing. No importance weighting: external sampling draws opponent+chance from
their own reach, so the sampled visitation supplies the counterfactual reach in
expectation; the regret target is the CRN-paired MC estimate.

IMPLEMENTATION NOTE (deviation from the design's literal estimator paragraph):
the design overview sketches an ESCHER-style SINGLE trajectory with the
traverser's actions sampled from a fixed uniform policy b_i and no importance
weighting. That single-trajectory-with-uniform-traverser-sampling scheme does
NOT reach equilibrium on the X1 tiny tree as written (validated empirically:
NashConv plateaus ~0.53 vs the tabular 5e-6 on the identical tree, independent
of m and of exact-vs-rollout q). The convergent structure is the one above:
FULL traverser-action recursion + opponent/chance sampling + opponent-free,
SD-CFR-realized averaging, which drives the SD-CFR-averaged NashConv to ~0.001
on the same tree. The rollout-based q and the SD-CFR averaging (the two design
decisions that matter for the gate) are preserved; only the traverser-side
trajectory sampling is replaced with the textbook external-sampling recursion.
This is flagged to @chief as a spec deviation requiring sign-off.
"""

from __future__ import annotations

import copy
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

from ..encoding import NUM_ACTIONS, action_to_index
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..sequence_encoding import PAD_ID, SEQ_CAP
from .prtcfr_net import pad_tokens, tiny_node_to_token_array

# A policy function maps a tiny_solver Decision node to a length-nA probability
# vector over its legal actions (node.actions order). The worker stays agnostic
# to how the policy is produced (net regret-matching, uniform, a fixed table);
# the trainer passes net-backed regret matching for sigma^t.
PolicyFn = Callable[[object], np.ndarray]


def uniform_policy(node) -> np.ndarray:
    """ESCHER's fixed traverser sampling policy b_i: uniform over legal actions."""
    n = len(node.actions)
    return np.full(n, 1.0 / n, dtype=np.float64)


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    """Sample an index from a probability vector using one uniform draw (CRN-safe)."""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def _rollout_value(node, traverser: int, sigma: PolicyFn, rng: random.Random) -> float:
    """Play one trajectory from ``node`` to a terminal under ``sigma`` for both
    players; return the traverser's utility.

    Chance nodes sample a child by the chance weights; decision nodes sample an
    action from sigma. All randomness is drawn from ``rng`` in a fixed order, so
    two sibling calls handed freshly-reset clones of the same seeded rng consume
    a common random-number stream (CRN pairing).
    """
    cur = node
    while True:
        kind = cur.kind
        if kind == "T":
            return cur.util[traverser]
        if kind == "C":
            idx = _sample_index(np.asarray(cur.weights, dtype=np.float64), rng)
            cur = cur.children[idx]
            continue
        # decision
        probs = sigma(cur)
        idx = _sample_index(probs, rng)
        cur = cur.children[idx]


def _child_value_mc(
    child, traverser: int, sigma: PolicyFn, m: int, seed_base: int
) -> float:
    """Mean traverser return of ``m`` rollouts from ``child`` under ``sigma``.

    Replicate k uses ``random.Random(seed_base + k)``. The caller invokes this
    for every sibling with the SAME ``seed_base``, so replicate k of sibling a
    and replicate k of sibling a' draw from the identical seeded stream: common
    random numbers paired across children, the variance-reduction the estimator
    relies on. CRN pairing does not bias the mean (each replicate is still a
    valid sample of the child's value), so E[q_hat] = q is preserved.
    """
    total = 0.0
    for k in range(m):
        rng = random.Random(seed_base + k)
        total += _rollout_value(child, traverser, sigma, rng)
    return total / m


class PRTCFRWorker:
    """External-sampling PRT-CFR traversal producing traverser regret samples.

    Stateless across traversals apart from the reservoir it fills; the policy
    (sigma^t) and rollout count m are supplied per traversal so the trainer can
    swap the current iterate each iteration.
    """

    def __init__(
        self,
        root,
        sigma: PolicyFn,
        m_rollouts: int = 4,
        seq_cap: int = SEQ_CAP,
        seed: int = 0,
    ):
        self.root = root
        self.sigma = sigma
        self.m_rollouts = m_rollouts
        self.seq_cap = seq_cap
        self._rng = random.Random(seed)
        # Monotone counter feeding rollout seed bases so every (traversal, node)
        # gets a distinct CRN block while staying reproducible from ``seed``.
        self._rollout_counter = 0
        self._added = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._rollout_counter = 0
        self._added = 0

    def _next_seed_base(self) -> int:
        # Spread seed bases by more than m so replicate blocks of different nodes
        # never overlap (collision would correlate unrelated estimates).
        self._rollout_counter += 1
        return self._rollout_counter * (self.m_rollouts + 1009)

    def traverse(self, traverser: int, iteration: int, buf: ReservoirBuffer) -> int:
        """Run one external-sampling traversal; append traverser regret samples to
        ``buf``. Returns the number of samples appended."""
        self._added = 0
        self._traverse_node(self.root, traverser, iteration, buf)
        return self._added

    def _traverse_node(
        self, node, traverser: int, iteration: int, buf: ReservoirBuffer
    ) -> float:
        """External-sampling traversal; appends traverser regret samples to ``buf``.

        Structure (standard ES-MCCFR; OpenSpiel external_sampling_mccfr): recurse
        ALL of the TRAVERSER's actions (so every traverser infoset on the sampled
        opponent/chance path is visited and its regret collected), but SAMPLE one
        action at OPPONENT and CHANCE nodes. The traverser's child values feed the
        regret; here each child value is the m-rollout MC estimate under sigma^t
        (decision 2), CRN-paired across the node's children. No average-strategy
        buffer is kept: SD-CFR realizes the average from the per-iteration regret-
        net snapshots, so opponent infosets record nothing.

        Return value: exact at terminals and the sampled value at chance/opponent
        nodes, but at a TRAVERSER node it is the local sigma-weighted rollout
        baseline -- NOT the node's subtree value (the recursive descent's returns
        are intentionally discarded; the descent only collects deeper regrets). The
        top-level ``traverse`` discards this return entirely -- regret targets come
        from per-node rollouts (q_hat), never from a bootstrapped value. Do NOT
        consume this return as a calibrated V(h) for credit assignment without
        revalidation; Phase-2 ESCHER defines proper value propagation.
        """
        kind = node.kind
        if kind == "T":
            return node.util[traverser]
        if kind == "C":
            idx = _sample_index(np.asarray(node.weights, dtype=np.float64), self._rng)
            return self._traverse_node(node.children[idx], traverser, iteration, buf)

        if node.player != traverser:
            # Opponent node: sample one action from sigma^t and recurse.
            probs = self.sigma(node)
            idx = _sample_index(probs, self._rng)
            return self._traverse_node(node.children[idx], traverser, iteration, buf)

        # Traverser node.
        nA = len(node.actions)
        sigma_node = self.sigma(node)  # length nA over legal actions

        # q_hat(a): m CRN-paired rollouts per child under sigma^t (the regret target
        # value). CRN shares the per-replicate seed across children for variance
        # reduction in the regret differences.
        seed_base = self._next_seed_base()
        q_hat = np.empty(nA, dtype=np.float64)
        for i in range(nA):
            q_hat[i] = _child_value_mc(
                node.children[i], traverser, self.sigma, self.m_rollouts, seed_base
            )
        baseline = float(np.dot(sigma_node, q_hat))

        regret_full = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, action in enumerate(node.actions):
            gi = action_to_index(action)
            regret_full[gi] = q_hat[i] - baseline
            mask[gi] = True
        buf.add(
            ReservoirSample(
                features=tiny_node_to_token_array(node, seq_cap=self.seq_cap),
                target=regret_full,
                action_mask=mask,
                iteration=iteration,
            )
        )
        self._added += 1

        # Recurse ALL traverser actions so deeper traverser infosets on this sampled
        # opponent/chance line are also visited (the ES-MCCFR requirement). Their
        # returns are intentionally discarded: this node's regret already comes from
        # the per-child rollouts (q_hat), not from the subtree. ``baseline`` below is
        # the local sigma-weighted rollout value, NOT this node's subtree value, and
        # is unused at the top level (see the docstring) -- do not consume it as a
        # calibrated V(h).
        for i in range(nA):
            self._traverse_node(node.children[i], traverser, iteration, buf)
        return baseline


# ---------------------------------------------------------------------------
# Exact tree values under a fixed policy (the unbiasedness ground truth)
# ---------------------------------------------------------------------------


def exact_child_values(node, traverser: int, sigma: PolicyFn) -> np.ndarray:
    """Exact expected traverser value of each legal child of ``node`` under
    ``sigma`` (both players), by enumeration over the explicit tree.

    This is the q(h, a) the MC estimator targets: the validation gate asserts the
    m-rollout mean converges to these as m grows. Returns a length-nA vector in
    node.actions order.
    """
    return np.array(
        [_exact_value(child, traverser, sigma) for child in node.children],
        dtype=np.float64,
    )


def _exact_value(node, traverser: int, sigma: PolicyFn) -> float:
    kind = node.kind
    if kind == "T":
        return float(node.util[traverser])
    if kind == "C":
        v = 0.0
        for child, w in zip(node.children, node.weights):
            v += w * _exact_value(child, traverser, sigma)
        return v
    probs = sigma(node)
    v = 0.0
    for p, child in zip(probs, node.children):
        if p <= 0.0:
            continue
        v += p * _exact_value(child, traverser, sigma)
    return v


def first_traverser_decision(node, traverser: int):
    """Return the first decision node belonging to ``traverser`` in a DFS from
    ``node`` (skipping chance/opponent), or None. Used by the unbiasedness test
    to pick a node with >= 2 legal actions for a meaningful q comparison."""
    if node.kind == "T":
        return None
    if node.kind == "C":
        for c in node.children:
            r = first_traverser_decision(c, traverser)
            if r is not None:
                return r
        return None
    if node.player == traverser and len(node.actions) >= 2:
        return node
    for c in node.children:
        r = first_traverser_decision(c, traverser)
        if r is not None:
            return r
    return None


# ---------------------------------------------------------------------------
# Production ESCHER single-trajectory sampler (S1W3 stage 2, cf-P2-escher)
# ---------------------------------------------------------------------------
#
# The tiny-tree PRTCFRWorker above deviates from the literal single-trajectory
# ESCHER estimator (its docstring flags this) because full traverser-action
# recursion is the structure that actually converges on the small, enumerable
# tiny tree. On the full game that recursion is intractable
# (p2-redesign.md sec 2.2: ~5^14 nodes per traversal), so production uses the
# estimator AS SPECIFIED there: a single trajectory, traverser actions drawn
# from a fixed uniform sampling policy b_i, opponent actions drawn from the
# current iterate sigma^t, and m-rollout CRN-paired Monte-Carlo q-targets
# computed from CLONES of the real engine state at every traverser decision.
#
# GameDriver is the seam the sampler traverses through. It mirrors the S1W2
# Go event-stream FFI surface pinned in the sprint-1 plan (token_len/tokens/
# tokens_since, vectorized apply, token-inclusive state save/restore) at the
# semantic level, so the sampler is driver-agnostic: PythonEngineGameDriver
# below is the "thin stub" the S1W3 stage-2 spawn note calls for (a Python
# fake driven by the existing Python engine), and the real Go-FFI-backed
# driver replaces it at S1W2 integration without changing sampler logic.
#
# Window semantics: PythonEngineGameDriver.tokens() always requests the FULL
# observation-action prefix via encode_observation_sequence(..., strict=True)
# at PRODUCTION_SEQ_CAP. strict=True turns "would truncate" into a hard
# SequenceOverflowError instead of a silent drop (the v0.4 Phase 2
# window-semantics decision note, sign-off conditions 1+2).
# PRODUCTION_SEQ_CAP is a separate module-level constant from the tiny-game
# SEQ_CAP=256 (imported above, untouched): raising it never touches the X2
# gate. Pinned by scripts/prtcfr_p100_instrument.py's P100 run over the
# production rule profile's real (300-turn) engine cap.

#: Production sequence cap. Kept distinct from the tiny-game SEQ_CAP constant
#: (coexistence rule): tiny paths import SEQ_CAP=256 unchanged; only
#: production call sites (PythonEngineGameDriver, PRTCFRProductionWorker) use
#: this value. See the module docstring above and
#: scripts/prtcfr_p100_instrument.py for the P100 instrumentation that pins it.
#:
#: Pinned by a real P100 run (scripts/prtcfr_p100_instrument.py, cumulative
#: ~8800 games / ~17600 player-observations across three runs -- 3000 + 800 +
#: 5000 games -- production rule profile: allowDrawFromDiscardPile/
#: allowReplaceAbilities/allowOpponentSnapping=True, the REAL engine turn cap
#: max_game_turns=300 -- NOT the 46-turn cap the pre-existing
#: sequence_encoding.py docstring's "mean ~726, worst ~1200" figures were
#: measured under). Cohorts: "natural" (uniform-random including CallCambia,
#: n=6000) mean=63.6, p99=248, max=602; "avoid_cambia" (skips CallCambia
#: whenever legal -- a plausible early/uniform-b_i-style trajectory, and the
#: honest worst case, structurally bounded only by the 300-turn cap: a
#: follow-up check confirmed EVERY avoid_cambia game runs the full 300 turns)
#: largest single run n=10000: mean=3217.5, p99=4476, max=7284 (the tail crept
#: up from 6316 at n=3000 to 7284 at n=10000, as expected for a max-statistic;
#: not yet fully converged). 12288 clears the observed worst-cohort max with
#: ~69% margin -- deliberately generous given the tail was still rising with
#: sample size. If a future larger instrumentation run (or real production
#: generation) ever exceeds this, driver.tokens() raises SequenceOverflowError
#: (hard error, never silent truncation) rather than corrupt data; treat that
#: as a signal to raise this constant, re-run the P100 script, and re-gate.
#: NOTE (flagged to @chief): at this cap, FIXED-WIDTH storage for the 20M-
#: sample reservoir is 12288*2 bytes*20M ~= 457GB -- infeasible. Only RAGGED
#: (variable-length, packed-pool + offsets) storage keeps the reservoir in
#: the 12-25GB range the compute plan (p2-redesign.md sec 6) targets, since
#: real per-sample lengths cluster far below this worst-case allocation bound
#: (see the "natural" cohort and any trained-policy trajectory, which calls
#: Cambia well before the turn cap). This is a hard requirement for S1W4, not
#: a preference.
PRODUCTION_SEQ_CAP: int = 12288


class GameDriver(Protocol):
    """Driver-agnostic seam the production sampler traverses through.

    Mirrors the pinned S1W2 Go event-stream FFI surface at the semantic level
    (not the C ABI): per-player token streams, a single apply (vectorized
    batch-apply is an S1W2/integration-time throughput concern, not a sampler
    semantics concern), and state clone in place of save/restore. Any concrete
    driver must give ``tokens(p)`` full-recall semantics: the COMPLETE
    observation-action prefix for player p, never a truncated window.
    """

    def current_player(self) -> int: ...

    def is_terminal(self) -> bool: ...

    def utility(self, player: int) -> float: ...

    def legal_actions(self) -> List[Any]: ...

    def apply(self, action: Any) -> bool: ...

    def tokens(self, player: int) -> List[int]: ...

    def clone(self) -> "GameDriver": ...

    def close(self) -> None: ...


class PythonEngineGameDriver:
    """Thin reference GameDriver over the Python engine (src.game.engine).

    Stub for the S1W2 Go-FFI driver: correctness-first, not throughput-first.
    ``tokens()`` re-encodes the observer's full observation list from scratch
    on every call (O(prefix length)) rather than incrementally appending (the
    real FFI's planned ``cambia_agent_tokens_since``); ``clone()`` deep-copies
    the whole engine GameState (Python-side ``copy.deepcopy``, not the Go
    engine's ~250B memcpy). Both are explicitly acceptable here -- this driver
    is swapped for the real incremental/cheap-clone FFI driver at S1W2
    integration, and none of the sampler logic above this seam changes.
    """

    def __init__(
        self,
        game: Any,
        init_hands: Dict[int, list],
        init_peeks: Dict[int, tuple],
        obs_streams: Optional[Dict[int, List[Any]]] = None,
        seq_cap: int = PRODUCTION_SEQ_CAP,
    ):
        self.game = game
        self.init_hands = init_hands
        self.init_peeks = init_peeks
        num_players = len(init_hands)
        self.obs_streams: Dict[int, List[Any]] = (
            obs_streams
            if obs_streams is not None
            else {p: [] for p in range(num_players)}
        )
        self.seq_cap = seq_cap

    def current_player(self) -> int:
        return self.game.get_acting_player()

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def utility(self, player: int) -> float:
        return self.game.get_utility(player)

    def legal_actions(self) -> List[Any]:
        # get_legal_actions() returns a Set[GameAction]; Python set iteration
        # order is not a stable function of content alone (string-field hash
        # randomization varies per process). Sort by the canonical
        # action_to_index so two runs with identical seeds see the identical
        # ordering (index-based sampling, CRN pairing, and this driver's own
        # determinism tests all rely on that).
        return sorted(self.game.get_legal_actions(), key=action_to_index)

    def apply(self, action: Any) -> bool:
        """Apply ``action``; return True iff the engine actually processed it
        (state changed). Returns False, WITHOUT recording any observation, if
        the engine rejected ``action`` for the current pending sub-decision.

        This rejection is a pre-existing engine/random-policy interaction,
        confirmed present even with zero cloning or driver code involved
        (``CambiaGameState.apply_action`` logs "Invalid action ... for
        pending state ... Waiting." and returns an EMPTY delta_list,
        i.e. no state mutation, for certain ability-pending-chain sequences
        under some house-rule combinations -- observed at ~1.4% of applies
        under uniform-random play with allowReplaceAbilities=True). Root-
        causing that engine/ability-mixin behavior is out of this task's
        scope; what IS in scope is never fabricating a token-stream frame
        for an action that never happened, since that would corrupt PRT-CFR's
        full-recall guarantee. ``apply_action``'s own ``delta_list`` return
        value is the engine's authoritative "did state change" signal (every
        successful mutation branch appends to it for undo support), so its
        truthiness is the check here. Callers must retry with a freshly
        sampled action on False (see ``_sample_and_apply`` below).
        """
        from .worker import _create_observation, _filter_observation

        actor = self.game.get_acting_player()
        delta_list, _undo = self.game.apply_action(action)
        if not delta_list:
            return False
        snap_results = list(getattr(self.game, "snap_results_log", []) or [])
        full_obs = _create_observation(None, action, self.game, actor, snap_results)
        if full_obs is None:
            raise RuntimeError(
                f"PythonEngineGameDriver.apply: observation creation failed for "
                f"actor {actor} action {action!r}"
            )
        for observer in self.obs_streams:
            self.obs_streams[observer].append(_filter_observation(full_obs, observer))
        return True

    def tokens(self, player: int) -> List[int]:
        from .. import sequence_encoding as se

        return se.encode_observation_sequence(
            self.init_hands[player],
            self.init_peeks[player],
            self.obs_streams[player],
            player,
            seq_cap=self.seq_cap,
            strict=True,
        )

    def clone(self) -> "PythonEngineGameDriver":
        cloned_game = copy.deepcopy(self.game)
        # AgentObservation entries are appended once and never mutated in
        # place (_filter_observation returns a fresh shallow copy per
        # observer), so a shallow per-list copy is a correct, cheap clone of
        # the streams; the expensive part is the game-state deepcopy above.
        cloned_streams = {p: list(v) for p, v in self.obs_streams.items()}
        return PythonEngineGameDriver(
            cloned_game,
            dict(self.init_hands),
            dict(self.init_peeks),
            cloned_streams,
            self.seq_cap,
        )

    def close(self) -> None:
        """No-op: nothing to free (plain Python objects, GC'd normally)."""


class GoEngineGameDriver:
    """GameDriver backed by the real Go engine via the FFI bridge (S1W13).

    The production substrate: correctness AND throughput (unlike
    PythonEngineGameDriver, the "thin stub" this file developed against
    before S1W12/S1W13 landed).

    ``clone()`` uses ``bridge.state_clone_wrapped`` (S1W12's
    ``cambia_state_clone``): a TRUE independent clone onto FRESH handles
    (game + both agents' belief + token streams), distinct from
    ``state_save``/``state_restore``'s same-handle rewind checkpoint (which
    cannot back independent, simultaneously-live branches -- the reason S1W3
    stage 2 did not implement a Go-backed driver against that primitive).

    ``tokens()`` reads the raw per-agent token body via the FFI and
    frame-aligns it at ``seq_cap`` with an explicit STRICT overflow guard
    (mirrors ``PythonEngineGameDriver``'s ``strict=True`` contract): the Go
    side's own ``MaxTokenStream`` hard-errors at APPEND time (surfaced via
    ``apply()``) before a body could ever exceed it, but
    ``bridge.frame_aligned_window`` itself has no strict mode and would
    silently window a body that's exactly at the raw cap if the BOS/EOS
    budget were not separately checked here -- this guard closes that gap.

    Owns its (engine, a0, a1) handles: ``close()`` frees them and is
    idempotent (safe to call multiple times, matching the underlying
    GoEngine/GoAgentState convention). The handle pool is finite --
    ``PRTCFRProductionWorker`` closes every clone after it is done exploring
    it (see ``traverse``/``_rollout_mc``).
    """

    def __init__(
        self,
        engine: Any,
        a0: Any,
        a1: Any,
        seq_cap: int = PRODUCTION_SEQ_CAP,
    ):
        self.engine = engine
        self.a0 = a0
        self.a1 = a1
        self.seq_cap = seq_cap
        self._closed = False

    def current_player(self) -> int:
        return self.engine.acting_player()

    def is_terminal(self) -> bool:
        return self.engine.is_terminal()

    def utility(self, player: int) -> float:
        return float(self.engine.get_utility()[player])

    def legal_actions(self) -> List[int]:
        # legal_actions_mask() -> (146,) uint8; nonzero() is already ascending
        # index order (no set-ordering nondeterminism, unlike the Python
        # driver's get_legal_actions()).
        mask = self.engine.legal_actions_mask()
        return [int(i) for i in mask.nonzero()[0]]

    def apply(self, action: int) -> bool:
        """Apply the action-index ``action`` via a length-1
        ``apply_games_batch`` call (the only FFI path that both advances the
        game AND appends to both agents' token streams -- the single-agent
        ``cambia_agent_update`` export does not touch the token stream at
        all, per the Go source). Length-1 batches are atomic: either this
        fully succeeds (game + both token streams advance) or nothing
        changes, so unlike the Python driver there is no phantom-observation
        risk from a partially-applied action.

        Returns True on success. Returns False if the engine rejected the
        action for the current state (mirrors PythonEngineGameDriver.apply's
        bool contract so ``_sample_and_apply`` retries uniformly across
        drivers). A token-stream OVERFLOW is a different, non-retryable
        condition -- retrying with a different action cannot help since the
        stream is already too long -- so it is re-raised, not swallowed.
        """
        from ..ffi import bridge

        try:
            bridge.apply_games_batch(
                [self.engine.handle],
                [self.a0.handle],
                [self.a1.handle],
                [int(action)],
            )
            return True
        except RuntimeError as e:
            if "overflow" in str(e):
                raise
            return False

    def tokens(self, player: int) -> List[int]:
        from ..ffi import bridge
        from ..sequence_encoding import SequenceOverflowError

        agent = self.a0 if player == 0 else self.a1
        body = agent.tokens()  # raw frame body, no BOS/EOS, never truncated
        budget = self.seq_cap - 2  # BOS + EOS, matching frame_aligned_window
        if len(body) > budget:
            raise SequenceOverflowError(
                f"Go agent token body length {len(body)} exceeds strict cap "
                f"budget {budget} (seq_cap={self.seq_cap}); raise seq_cap "
                f"rather than let frame_aligned_window silently truncate"
            )
        return bridge.frame_aligned_window(body, seq_cap=self.seq_cap, add_bos_eos=True)

    def clone(self) -> "GoEngineGameDriver":
        from ..ffi import bridge

        new_engine, new_a0, new_a1 = bridge.state_clone_wrapped(
            self.engine, self.a0, self.a1
        )
        return GoEngineGameDriver(new_engine, new_a0, new_a1, seq_cap=self.seq_cap)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.a0.close()
        self.a1.close()
        self.engine.close()


def _default_production_house_rules() -> Any:
    from ..config import CambiaRulesConfig

    house_rules = CambiaRulesConfig()
    house_rules.allowDrawFromDiscardPile = True
    house_rules.allowReplaceAbilities = True
    house_rules.allowOpponentSnapping = True
    house_rules.max_game_turns = 300
    house_rules.lockCallerHand = False
    return house_rules


def _new_python_production_driver(
    seed: int, house_rules: Optional[Any] = None, num_players: int = 2
) -> PythonEngineGameDriver:
    """Build a fresh production game + PythonEngineGameDriver for ``seed``.

    Uses the engine's own internal deal (``CambiaGameState(house_rules=...,
    _rng=...)``, the same construction path tiny_solver's tree builder uses),
    NOT the Go-deal-matching test helper: self-play generation only needs an
    internally consistent engine instance here, independent of Go-parity,
    which is covered separately by the FFI cross-path parity tests.
    """
    from ..game.engine import CambiaGameState

    if house_rules is None:
        house_rules = _default_production_house_rules()

    game = CambiaGameState(house_rules=house_rules, _rng=random.Random(seed))
    init_hands = {p: list(game.players[p].hand) for p in range(num_players)}
    init_peeks = {
        p: tuple(game.players[p].initial_peek_indices) for p in range(num_players)
    }
    return PythonEngineGameDriver(game, init_hands, init_peeks)


def _new_go_production_driver(
    seed: int, house_rules: Optional[Any] = None, seq_cap: int = PRODUCTION_SEQ_CAP
) -> "GoEngineGameDriver":
    """Build a fresh production game + GoEngineGameDriver for ``seed`` over
    the real Go engine via the FFI bridge (S1W13: the production substrate)."""
    from ..ffi.bridge import GoAgentState, GoEngine

    if house_rules is None:
        house_rules = _default_production_house_rules()

    engine = GoEngine(seed=seed, house_rules=house_rules)
    a0 = GoAgentState(engine, player_id=0)
    a1 = GoAgentState(engine, player_id=1)
    return GoEngineGameDriver(engine, a0, a1, seq_cap=seq_cap)


def new_production_driver(
    seed: int,
    house_rules: Optional[Any] = None,
    num_players: int = 2,
    backend: str = "go",
) -> "GameDriver":
    """Build a fresh production game + driver for ``seed``.

    ``backend="go"`` (default, S1W13): the real Go engine via the FFI bridge
    (``GoEngineGameDriver``) -- the production substrate. ``backend="python"``:
    ``PythonEngineGameDriver``, the reference/stub implementation this file
    developed against before the Go clone-to-fresh-handles FFI export
    (``cambia_state_clone``, S1W12) landed; kept for tests exercising its
    specific internals and as a fallback where ``libcambia.so`` is
    unavailable. Both satisfy the same ``GameDriver`` protocol; sampler code
    (``PRTCFRProductionWorker``) never branches on which one it was handed.
    """
    if backend == "go":
        return _new_go_production_driver(seed, house_rules, seq_cap=PRODUCTION_SEQ_CAP)
    if backend == "python":
        return _new_python_production_driver(seed, house_rules, num_players)
    raise ValueError(f"unknown backend {backend!r}; expected 'go' or 'python'")


# Production sigma^t signature: (full-recall token prefix, legal action mask)
# -> masked probability vector over the NUM_ACTIONS=146 global action space.
# The sampler is agnostic to how this is produced (a net's regret-matched
# strategy, PRTCFRInferenceService.strategy(), or -- for b_i -- the fixed
# uniform policy below); it only ever calls sigma(tokens, mask).
ProductionPolicyFn = Callable[[List[int], np.ndarray], np.ndarray]


def uniform_policy_production(tokens: List[int], legal_mask: np.ndarray) -> np.ndarray:
    """ESCHER's fixed traverser sampling policy b_i: uniform over legal
    actions, independent of tokens. Signature matches ProductionPolicyFn so it
    composes with the same call sites as a net-backed sigma^t."""
    probs = legal_mask.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return probs
    return probs / total


def _action_index(action: Any) -> int:
    """Global [0, NUM_ACTIONS) index for a production-driver action.

    Driver-agnostic: PythonEngineGameDriver's legal_actions()/apply() traffic
    in GameAction NamedTuples (routed through encoding.action_to_index);
    GoEngineGameDriver's traffic in the same integer indices GoEngine.
    apply_action already uses (legal_actions_mask() is index-native, so no
    translation is needed there). This adapter lets every driver-agnostic
    helper below (_legal_mask, _sample_legal_action, the traverser's regret
    bookkeeping) accept whichever action representation the active driver
    produces without branching on driver type.
    """
    if isinstance(action, (int, np.integer)):
        return int(action)
    return action_to_index(action)


def _legal_mask(legal: Sequence[Any]) -> np.ndarray:
    """NUM_ACTIONS-wide bool mask for a legal-action list (production driver)."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal:
        mask[_action_index(a)] = True
    return mask


def _sample_legal_action(
    legal: Sequence[Any], probs: np.ndarray, rng: random.Random
) -> Any:
    """Sample one action object from ``legal`` given a NUM_ACTIONS-wide probs
    vector (indexed by _action_index); renormalizes defensively so a
    slightly-off-mass sigma never raises."""
    weights = np.array([probs[_action_index(a)] for a in legal], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        weights = np.full(len(legal), 1.0 / len(legal))
    else:
        weights = weights / total
    idx = _sample_index(weights, rng)
    return legal[idx]


class DriverStuckError(RuntimeError):
    """Raised when every attempt to advance a GameDriver is rejected by the
    engine (see ``_sample_and_apply``); indicates a true engine-level stall,
    distinct from the ordinary retry-and-recover case."""


def _sample_and_apply(
    driver: "GameDriver",
    legal: Sequence[Any],
    probs: np.ndarray,
    rng: random.Random,
    max_attempts: int = 20,
) -> Any:
    """Sample an action from ``legal`` per ``probs`` and apply it to
    ``driver``, RESAMPLING (same ``legal``/``probs``, fresh draw) if the
    engine rejects it (``driver.apply`` returns False -- see
    ``PythonEngineGameDriver.apply``'s docstring for the pre-existing
    engine/ability-pending-chain interaction this guards against). Legitimate
    because ``get_legal_actions()`` is unchanged after a rejection (no state
    mutated), so resampling from the identical distribution is a valid retry,
    not a bias -- it just avoids the SPECIFIC member the engine just refused.
    Returns the action that was actually applied; raises ``DriverStuckError``
    if ``max_attempts`` consecutive draws are all rejected (a true stall).
    """
    for _ in range(max_attempts):
        action = _sample_legal_action(legal, probs, rng)
        if driver.apply(action):
            return action
    raise DriverStuckError(
        f"driver rejected {max_attempts} consecutive sampled actions drawn "
        f"from the same legal set (size {len(legal)}); likely a true engine "
        f"stall rather than the ordinary pending-state retry case"
    )


#: Optional S1W6 hook type: PRTCFRProductionWorker.traverse calls this at
#: every TRAVERSER decision with ``(tokens_h, driver, pooled_rollout_mean,
#: iteration)`` -- see PRTCFRProductionWorker.value_sink for the exact
#: synchronous-call contract (no cloning happens on the worker's behalf).
ValueSinkFn = Callable[[List[int], "GameDriver", float, int], None]


class PRTCFRProductionWorker:
    """Single-trajectory ESCHER sampler on the real engine (production path).

    Per p2-redesign.md sec 2.2: traverser actions on the trajectory are drawn
    from the fixed uniform sampling policy b_i; opponent actions from the
    current iterate sigma^t; chance via the engine. At every TRAVERSER
    decision on the trajectory, each legal action's child value is an
    m-rollout CRN-paired Monte-Carlo estimate (both players playing sigma^t to
    termination from a driver CLONE of that child) -- the same rollout
    estimator the tiny-tree worker above uses, but driven through GameDriver
    against the real engine instead of the enumerable tree.

    No average-strategy buffer is kept (SD-CFR realizes the average from the
    per-iteration regret-net snapshots): opponent-node visits append nothing.
    """

    def __init__(
        self,
        sigma: ProductionPolicyFn,
        m_rollouts: int = 4,
        seq_cap: int = PRODUCTION_SEQ_CAP,
        seed: int = 0,
        max_trajectory_steps: int = 4000,
        value_sink: Optional[ValueSinkFn] = None,
    ):
        self.sigma = sigma
        self.m_rollouts = m_rollouts
        self.seq_cap = seq_cap
        self.max_trajectory_steps = max_trajectory_steps
        # S1W6 (p2-redesign.md sec 2.4): OPTIONAL tap for the V_phi critic,
        # OUTSIDE the regret path -- this worker never reads value_sink's
        # return value and never lets it influence q_hat/baseline/regret_full
        # above. Default None: zero behavior change to the sampler or the
        # regret reservoir. When set, called synchronously at every TRAVERSER
        # decision (see the call site in ``traverse`` below) with the SAME
        # live, not-yet-mutated ``driver`` reference used for that decision's
        # rollouts -- valid only for the duration of the call (the trajectory
        # advances immediately after value_sink returns); a sink that needs
        # to retain game state beyond the call must clone it itself
        # (``driver.clone()``). Omniscient-feature extraction from that
        # driver is deliberately NOT this file's concern (training-only
        # boundary owned by cfr/src/cfr/omniscient.py); value_sink
        # implementations route it through ``compute_omniscient_features``.
        self.value_sink = value_sink
        self._rng = random.Random(seed)
        self._rollout_counter = 0
        self._added = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._rollout_counter = 0
        self._added = 0

    def _next_seed_base(self) -> int:
        # Spread seed bases by more than m so replicate blocks of different
        # nodes never overlap (collision would correlate unrelated estimates).
        self._rollout_counter += 1
        return self._rollout_counter * (self.m_rollouts + 1009)

    def traverse(
        self, driver: GameDriver, traverser: int, iteration: int, buf: ReservoirBuffer
    ) -> int:
        """Run one single-trajectory ESCHER traversal from ``driver`` (mutated
        in place); append traverser regret samples to ``buf``. Returns the
        number of samples appended.

        ``max_trajectory_steps`` is a defensive bound (engines with
        pathological legal-action cycles cannot hang the sampler); production
        games under the real engine turn cap stay far under it.
        """
        self._added = 0
        steps = 0
        while not driver.is_terminal():
            steps += 1
            if steps > self.max_trajectory_steps:
                break
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)

            if actor != traverser:
                # Opponent node: sample one action from sigma^t and apply
                # (retrying on engine rejection -- see _sample_and_apply).
                tokens = driver.tokens(actor)
                probs = self.sigma(tokens, mask)
                _sample_and_apply(driver, legal, probs, self._rng)
                continue

            # Traverser decision node: price every legal action with m
            # CRN-paired rollouts from a clone of its child, record the
            # regret sample, then continue the trajectory via b_i (uniform).
            tokens_h = driver.tokens(traverser)
            sigma_node = self.sigma(tokens_h, mask)
            seed_base = self._next_seed_base()
            kept_actions: List[Any] = []
            kept_q: List[float] = []
            for action in legal:
                child = driver.clone()
                try:
                    if not child.apply(action):
                        # The engine rejected this nominally-"legal" action
                        # for the current pending sub-decision (the same
                        # pre-existing engine/ability-pending-chain
                        # interaction PythonEngineGameDriver.apply guards
                        # against). Exclude it from THIS decision's regret
                        # sample rather than fabricate a q-value for an
                        # action that was never actually applied to the
                        # clone.
                        continue
                    kept_actions.append(action)
                    kept_q.append(self._rollout_mc(child, traverser, seed_base))
                finally:
                    # Every clone() call allocates real resources on a
                    # Go-backed driver (fresh game+2 agent handles from a
                    # finite pool, S1W12's cambia_state_clone); the Python
                    # stub's close() is a no-op. Close unconditionally,
                    # whether kept or rejected.
                    child.close()

            if kept_actions:
                q_hat = np.array(kept_q, dtype=np.float64)
                sigma_legal = np.array(
                    [sigma_node[_action_index(a)] for a in kept_actions],
                    dtype=np.float64,
                )
                sigma_total = sigma_legal.sum()
                sigma_legal = (
                    sigma_legal / sigma_total
                    if sigma_total > 0
                    else np.full(len(kept_actions), 1.0 / len(kept_actions))
                )
                baseline = float(np.dot(sigma_legal, q_hat))

                regret_full = np.zeros(NUM_ACTIONS, dtype=np.float32)
                kept_mask = np.zeros(NUM_ACTIONS, dtype=bool)
                for i, action in enumerate(kept_actions):
                    gi = _action_index(action)
                    regret_full[gi] = q_hat[i] - baseline
                    kept_mask[gi] = True
                buf.add(
                    ReservoirSample(
                        features=pad_tokens(tokens_h, seq_cap=self.seq_cap),
                        target=regret_full,
                        action_mask=kept_mask,
                        iteration=iteration,
                    )
                )
                self._added += 1

                # S1W6 value-sink tap (read-only; see __init__ for the
                # contract). ``baseline`` here is EXACTLY the pooled
                # sigma-weighted rollout mean already computed above for the
                # regret target -- the sink receives the identical float, it
                # never triggers extra rollouts or recomputation.
                if self.value_sink is not None:
                    self.value_sink(tokens_h, driver, baseline, iteration)
            # else: every legal action was rejected for this decision (should
            # be exceedingly rare); no regret sample is recorded, and the
            # trajectory still advances below via the full `legal`/b_i (which
            # retries until one actually applies, or raises DriverStuckError
            # if the driver is truly wedged).

            # ESCHER: the trajectory's traverser action comes from the FIXED
            # uniform sampling policy b_i, never sigma (sigma only prices the
            # regret via q_hat/baseline above).
            b_i_probs = uniform_policy_production(tokens_h, mask)
            _sample_and_apply(driver, legal, b_i_probs, self._rng)

        return self._added

    def _rollout_mc(self, child: GameDriver, traverser: int, seed_base: int) -> float:
        """Mean traverser return of ``m_rollouts`` CRN-paired replicates from
        ``child``, both players playing sigma^t to termination. Replicate k
        clones ``child`` fresh and uses ``random.Random(seed_base + k)``; the
        caller uses the SAME ``seed_base`` for every sibling action at this
        decision, so replicate k of action a and replicate k of action a'
        consume the identical seeded stream (CRN pairing)."""
        total = 0.0
        for k in range(self.m_rollouts):
            rng = random.Random(seed_base + k)
            rollout_driver = child.clone()
            try:
                total += self._rollout_to_terminal(rollout_driver, traverser, rng)
            finally:
                rollout_driver.close()
        return total / self.m_rollouts

    def _rollout_to_terminal(
        self, driver: GameDriver, traverser: int, rng: random.Random
    ) -> float:
        """Play ``driver`` to a terminal under sigma^t for both players
        (unbiased Monte-Carlo child value, decision 2: no critic in the
        regret path); return the traverser's utility."""
        steps = 0
        while not driver.is_terminal():
            steps += 1
            if steps > self.max_trajectory_steps:
                break
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)
            tokens = driver.tokens(actor)
            probs = self.sigma(tokens, mask)
            _sample_and_apply(driver, legal, probs, rng)
        return driver.utility(traverser)


# ===========================================================================
# Batched incremental production generation (S1W15, cambia-239)
# ===========================================================================
#
# The X3 gen verdict (S1W7): production generation ran 9.09 s/game at K=16 with
# 97% of wall in inference, because the single-trajectory sampler above issues
# ONE un-batched, non-incrementally-carried full-prefix GRU forward per decision
# and per m-rollout step (NetProductionSigma.__call__). The designed remedy is
# the S1W3 batched incremental PRTCFRInferenceService, which S1W5 never
# consumed. This section consumes it.
#
# Two wins, both required (p2-redesign.md sec 6: "batched incremental GRU steps
# (carried hidden state per live rollout) ... batching across ~10^4 concurrent
# rollouts keeps occupancy high"):
#
#   1. Incremental carry: a live game/rollout's per-player token stream grows
#      monotonically (BOS + init_peek + observation frames); a policy query
#      needs the GRU hidden after [BOS]+body+[EOS]. Carrying the raw hidden over
#      [BOS]+body and stepping only new frames turns each decision's O(prefix)
#      re-encode into O(new frames), i.e. the whole trajectory's inference from
#      O(L^2) to O(L). The EOS is appended TRANSIENTLY per query
#      (PRTCFRInferenceService.query_transient) so the carry stays a clean
#      prefix -- this is the BOS/EOS/frame-boundary subtlety the equivalence
#      test pins.
#   2. Batching: all K live games AND all their m CRN rollouts that reach a
#      decision on the same scheduler tick have their queries gathered into ONE
#      batched forward. A rollout branch forks the carried hidden
#      (PRTCFRInferenceService.fork, the torch mirror of Go's cambia_state_clone)
#      instead of re-encoding the shared prefix.
#
# Structure: the sampler logic is expressed as generators that ``yield`` a
# _Query at each policy-query point (opponent action, traverser regret pricing,
# rollout step) and receive the strategy vector back. A cooperative scheduler
# (_run_batched_scheduler) pumps every live generator to its next query, gathers
# them, evaluates once, and resumes them -- so the generation is a faithful
# restatement of the sequential estimator above (same per-stream RNG draw order,
# same CRN pairing, same regret math), only the INFERENCE is batched and
# incremental. Semantic equivalence to the sequential full-prefix path is a hard
# gate (tests/test_prtcfr_batched_gen.py).


class _Query:
    """A policy-query request a sampler coroutine yields; the scheduler fills
    ``result`` with the (146,) float64 strategy vector and resumes the coro."""

    __slots__ = ("stream", "player", "tokens", "mask", "result")

    def __init__(self, stream: int, player: int, tokens: List[int], mask: np.ndarray):
        self.stream = stream
        self.player = player
        self.tokens = tokens
        self.mask = mask
        self.result: Optional[np.ndarray] = None


class _Join:
    """A structured-concurrency request: run ``subtasks`` (generators) to
    completion, batching their queries with everything else live, then resume
    the parent coroutine with the ordered list of their return values."""

    __slots__ = ("subtasks",)

    def __init__(self, subtasks: List[Iterator]):
        self.subtasks = subtasks


class _Task:
    """Scheduler bookkeeping for one live generator (a main trajectory or a
    rollout). Parent/child links implement _Join."""

    __slots__ = (
        "gen",
        "parent",
        "idx",
        "send_val",
        "join_pending",
        "join_results",
    )

    def __init__(self, gen: Iterator, parent: "Optional[_Task]" = None, idx: int = 0):
        self.gen = gen
        self.parent = parent
        self.idx = idx
        self.send_val: Any = None
        self.join_pending: int = 0
        self.join_results: List[Any] = []


class SigmaBackend(Protocol):
    """The inference seam the batched sampler queries through. ``evaluate``
    fills every _Query's ``result`` in ONE batch; ``fork``/``drop`` mirror the
    driver clone/close so carried hidden state fans out and frees with the
    game/rollout trajectories it backs."""

    def evaluate(self, queries: List[_Query]) -> None: ...

    def fork(self, src_stream: int, dst_stream: int) -> None: ...

    def drop(self, stream: int) -> None: ...


def _run_batched_scheduler(
    root_gens: List[Iterator], backend: SigmaBackend
) -> None:
    """Cooperative trampoline: run every generator in ``root_gens`` (and any
    _Join subtasks they spawn) to completion, gathering all pending _Query
    yields at each tick into a single ``backend.evaluate`` call.

    Contract each generator obeys: it ``yield``s either a ``_Query`` (and is
    resumed with the filled ``result``) or a ``_Join`` (and is resumed with the
    ordered list of its subtasks' return values); ``return`` ends it. This is a
    standard batched-inference trampoline (cf. vLLM-style continuous batching):
    the interleaving across generators changes only the TIMING of the shared
    forward, never any single generator's own instruction/RNG order, so the
    generation is bit-faithful to the sequential estimator per stream.
    """
    runnable: List[_Task] = [_Task(g) for g in root_gens]

    def complete(task: _Task, value: Any) -> None:
        parent = task.parent
        if parent is not None:
            parent.join_results[task.idx] = value
            parent.join_pending -= 1
            if parent.join_pending == 0:
                parent.send_val = parent.join_results
                runnable.append(parent)

    while runnable:
        pending: List[_Task] = []
        # Drain: pump every runnable task (and any children it spawns this tick)
        # to its next query / join / completion.
        while runnable:
            task = runnable.pop()
            try:
                y = task.gen.send(task.send_val)
                task.send_val = None
            except StopIteration as si:
                complete(task, si.value)
                continue
            if type(y) is _Query:
                task.send_val = y  # stash the query on the task for the batch
                pending.append(task)
            elif type(y) is _Join:
                subs = y.subtasks
                if not subs:
                    task.send_val = []
                    runnable.append(task)
                else:
                    task.join_pending = len(subs)
                    task.join_results = [None] * len(subs)
                    for k, cg in enumerate(subs):
                        runnable.append(_Task(cg, parent=task, idx=k))
                    # task itself waits (not runnable) until all children finish
            else:
                raise TypeError(
                    f"sampler coroutine yielded {type(y).__name__}, expected "
                    f"_Query or _Join"
                )
        if not pending:
            break
        queries = [t.send_val for t in pending]
        backend.evaluate(queries)
        for t in pending:
            q: _Query = t.send_val
            t.send_val = q.result
            runnable.append(t)


class IncrementalSigmaManager:
    """Production ``SigmaBackend``: batched incremental GRU inference over the
    live games and rollouts, on top of ``PRTCFRInferenceService``.

    Per (stream, player) it carries the raw GRU hidden state over ``[BOS]+body``
    (no trailing EOS) and its length; a query steps only the newly-appended
    frames (``advance``), then a TRANSIENT ``[EOS]`` step (``query_transient``)
    yields the masked regret-matched strategy that matches a full
    ``encode_observation_sequence`` re-encode. ``fork`` copies both players'
    carried hidden at a rollout branch (the shared prefix is not re-encoded).

    The returned strategy is the same (146,) float64 masked vector
    ``NetProductionSigma`` returns for the same tokens+mask (carry-vs-reencode
    identity, tests/test_prtcfr_infer.py + test_prtcfr_batched_gen.py); at bf16
    it is that vector within bf16 tolerance (an independent precision axis, not
    the carry identity -- see the equivalence report).
    """

    def __init__(self, service: Any, num_players: int = 2):
        self.service = service
        self.num_players = num_players
        # (stream, player) -> carried [BOS]+body length folded into the hidden.
        self._carry_len: Dict[Tuple[int, int], int] = {}
        self._players_of: Dict[int, set] = {}

    def _sid(self, stream: int, player: int) -> Tuple[int, int]:
        return (stream, player)

    def evaluate(self, queries: List[_Query]) -> None:
        if not queries:
            return
        import torch

        sids: List[Tuple[int, int]] = []
        news: List[List[int]] = []
        transients: List[List[int]] = []
        for q in queries:
            toks = q.tokens
            if not toks:
                raise ValueError(
                    "IncrementalSigmaManager.evaluate: empty query tokens; "
                    "driver.tokens() must yield at least [BOS]"
                )
            # The carry conditions on the MONOTONIC prefix tokens[:-1]
            # ([BOS]+body for the real tokenizer -- body grows by append every
            # frame); the FINAL token (EOS for the real tokenizer, a per-query
            # marker for a test driver) is applied TRANSIENTLY so the carry
            # never absorbs it and the next frame appends cleanly. This is exact
            # for any driver whose tokens[:-1] is a growing prefix per
            # (stream, player) -- the general form of the [BOS]+body+[EOS]
            # window-semantics decision.
            body_prefix = toks[:-1]
            transients.append([toks[-1]])
            sid = self._sid(q.stream, q.player)
            cl = self._carry_len.get(sid, 0)
            if cl > len(body_prefix):
                # Non-monotonic prefix (would only happen under a truncating,
                # non-strict window -- production uses the non-firing strict
                # cap). Re-register from scratch rather than corrupt the carry.
                self.service.drop(sid)
                cl = 0
                # Reset the stored carry too: leaving the stale value makes
                # line +7's get()+len(new) accumulate past the true prefix,
                # re-firing this drop on every later query for the sid.
                self._carry_len[sid] = 0
            sids.append(sid)
            news.append(body_prefix[cl:])
        # One batched register/step over the new frames (skips empty streams,
        # registers fresh ones from a zero hidden -- see service.advance).
        self.service.advance(sids, news)
        for sid, new, q in zip(sids, news, queries):
            self._carry_len[sid] = self._carry_len.get(sid, 0) + len(new)
            self._players_of.setdefault(q.stream, set()).add(q.player)
        # Transient final-token step -> masked regret-matched strategy, one batch.
        masks = torch.stack(
            [torch.as_tensor(np.asarray(q.mask, dtype=bool)) for q in queries]
        ).to(self.service.device)
        strat = self.service.query_transient(sids, transients, masks)
        strat_np = strat.detach().to("cpu", dtype=torch.float64).numpy()
        for i, q in enumerate(queries):
            q.result = strat_np[i]

    def fork(self, src_stream: int, dst_stream: int) -> None:
        for p in self._players_of.get(src_stream, ()):  # copy only live streams
            src = self._sid(src_stream, p)
            dst = self._sid(dst_stream, p)
            self.service.fork(src, dst)
            self._carry_len[dst] = self._carry_len[src]
            self._players_of.setdefault(dst_stream, set()).add(p)

    def drop(self, stream: int) -> None:
        for p in self._players_of.pop(stream, ()):
            sid = self._sid(stream, p)
            self.service.drop(sid)
            self._carry_len.pop(sid, None)


class _FullReencodeSigmaBackend:
    """Reference ``SigmaBackend`` for the equivalence gate: each query is a
    fresh, single-item ``net.strategy_from_tokens`` full re-encode -- bit-for-
    bit the sequential ``NetProductionSigma.__call__`` computation. Carries no
    state, so ``fork``/``drop`` are no-ops.

    Running the batched worker with THIS backend isolates the coroutine
    scheduler / rollout-fork / RNG-order restatement (it must reproduce the
    sequential worker's samples exactly, since the sigma per query is identical);
    the IncrementalSigmaManager is then shown equivalent to this backend per
    query by the carry-identity test. The composition is the semantic-
    equivalence gate.
    """

    def __init__(self, net: Any, seq_cap: int = PRODUCTION_SEQ_CAP):
        self.net = net.eval()
        self.seq_cap = int(seq_cap)

    def evaluate(self, queries: List[_Query]) -> None:
        import torch

        device = self.net.device
        for q in queries:
            toks = list(q.tokens) if q.tokens else [PAD_ID]
            if len(toks) > self.seq_cap:
                toks = toks[-self.seq_cap :]
            t = torch.as_tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            m = torch.as_tensor(
                np.asarray(q.mask, dtype=bool), device=device
            ).unsqueeze(0)
            with torch.no_grad():
                strat = self.net.strategy_from_tokens(t, m)
            q.result = strat[0].detach().to("cpu", dtype=torch.float64).numpy()

    def fork(self, src_stream: int, dst_stream: int) -> None:
        pass

    def drop(self, stream: int) -> None:
        pass


class PRTCFRBatchedProductionWorker:
    """Batched, incrementally-carried restatement of ``PRTCFRProductionWorker``.

    Same single-trajectory ESCHER estimator (traverser actions from b_i uniform,
    opponent from sigma^t, m-rollout CRN-paired Monte-Carlo q-targets, SD-CFR
    averaging so opponent nodes record nothing), but the whole chunk of games
    runs concurrently through ``_run_batched_scheduler`` so every live game and
    rollout that reaches a decision on a tick shares ONE batched, incremental
    inference call (``SigmaBackend.evaluate``).

    Per-game reproducibility: each game gets an independent, deterministic RNG
    seeded from its own game seed (and an independent rollout-seed counter), so
    a game's samples are a pure function of its seed -- unlike the sequential
    worker's single continuing RNG threaded across games. This makes generation
    reproducible per game and lets the equivalence test drive both paths from
    matched per-game seeds; the estimator's statistics are unchanged (the RNG is
    only a sampling source).
    """

    def __init__(
        self,
        m_rollouts: int = 4,
        seq_cap: int = PRODUCTION_SEQ_CAP,
        max_trajectory_steps: int = 4000,
        value_sink: Optional[ValueSinkFn] = None,
    ):
        self.m_rollouts = m_rollouts
        self.seq_cap = seq_cap
        self.max_trajectory_steps = max_trajectory_steps
        self.value_sink = value_sink
        self._key_counter = 0

    def _new_key(self) -> int:
        self._key_counter += 1
        return self._key_counter

    def generate(
        self,
        game_specs: List[Dict[str, Any]],
        backend: SigmaBackend,
    ) -> Dict[int, int]:
        """Run one batched generation chunk. ``game_specs`` is a list of dicts,
        each ``{"seed", "driver", "traverser", "iteration", "buf"}``. Appends
        traverser regret samples to each game's ``buf`` and returns
        ``{game_index: samples_added}``.

        The trainer owns the top-level game drivers (closes them after); this
        worker closes only its own clones (rollout drivers and per-action child
        drivers), unconditionally, so the finite Go handle pool never leaks.
        """
        added: Dict[int, int] = {}
        gens: List[Iterator] = []
        main_streams: List[int] = []
        for gi, spec in enumerate(game_specs):
            added[gi] = 0
            stream = self._new_key()
            main_streams.append(stream)
            gens.append(
                self._traverse_coro(
                    game_index=gi,
                    stream=stream,
                    driver=spec["driver"],
                    traverser=int(spec["traverser"]),
                    iteration=int(spec["iteration"]),
                    buf=spec["buf"],
                    rng=random.Random(int(spec["seed"])),
                    rollout_counter=[0],
                    backend=backend,
                    added=added,
                )
            )
        _run_batched_scheduler(gens, backend)
        # Rollout streams self-drop (see _rollout_coro); free the per-game main
        # streams too so a service/manager reused across chunks never leaks
        # carried hidden state.
        for stream in main_streams:
            backend.drop(stream)
        return added

    def _next_seed_base(self, rollout_counter: List[int]) -> int:
        # Mirror PRTCFRProductionWorker._next_seed_base with a per-game counter
        # so rollout CRN seeds are reproducible from the game seed alone.
        rollout_counter[0] += 1
        return rollout_counter[0] * (self.m_rollouts + 1009)

    def _traverse_coro(
        self,
        game_index: int,
        stream: int,
        driver: GameDriver,
        traverser: int,
        iteration: int,
        buf: ReservoirBuffer,
        rng: random.Random,
        rollout_counter: List[int],
        backend: SigmaBackend,
        added: Dict[int, int],
    ) -> Iterator:
        """Coroutine restatement of ``PRTCFRProductionWorker.traverse`` for one
        game: yields a _Query at every policy-query point, a _Join to price a
        traverser decision's rollouts. Faithful to the sequential RNG-draw order
        (see the class docstring)."""
        steps = 0
        while not driver.is_terminal():
            steps += 1
            if steps > self.max_trajectory_steps:
                break
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)

            if actor != traverser:
                tokens = driver.tokens(actor)
                probs = yield _Query(stream, actor, tokens, mask)
                _sample_and_apply(driver, legal, probs, rng)
                continue

            # Traverser decision: price every legal action with m CRN-paired
            # rollouts (each from a driver clone + a forked hidden), record the
            # regret sample, then advance the trajectory via b_i (uniform).
            tokens_h = driver.tokens(traverser)
            sigma_node = yield _Query(stream, traverser, tokens_h, mask)
            seed_base = self._next_seed_base(rollout_counter)

            kept_actions: List[Any] = []
            rollout_subtasks: List[Iterator] = []
            rollout_owner: List[int] = []
            for action in legal:
                child = driver.clone()
                if not child.apply(action):
                    # Engine rejected this nominally-legal action for the
                    # pending sub-decision (same pre-existing interaction the
                    # sequential worker guards): exclude it rather than price a
                    # phantom q-value.
                    child.close()
                    continue
                ai = len(kept_actions)
                kept_actions.append(action)
                for k in range(self.m_rollouts):
                    rdriver = child.clone()
                    rstream = self._new_key()
                    backend.fork(stream, rstream)  # child rollouts branch from h
                    rollout_subtasks.append(
                        self._rollout_coro(
                            driver=rdriver,
                            stream=rstream,
                            traverser=traverser,
                            rng=random.Random(seed_base + k),
                            backend=backend,
                        )
                    )
                    rollout_owner.append(ai)
                # The child driver's only role was to seed the m rollout clones;
                # close it (and it never got its own sigma stream) now.
                child.close()

            if kept_actions:
                results = yield _Join(rollout_subtasks)
                q_sums = [0.0] * len(kept_actions)
                q_cnts = [0] * len(kept_actions)
                for util, ai in zip(results, rollout_owner):
                    q_sums[ai] += util
                    q_cnts[ai] += 1
                q_hat = np.array(
                    [q_sums[i] / q_cnts[i] for i in range(len(kept_actions))],
                    dtype=np.float64,
                )
                sigma_legal = np.array(
                    [sigma_node[_action_index(a)] for a in kept_actions],
                    dtype=np.float64,
                )
                sigma_total = sigma_legal.sum()
                sigma_legal = (
                    sigma_legal / sigma_total
                    if sigma_total > 0
                    else np.full(len(kept_actions), 1.0 / len(kept_actions))
                )
                baseline = float(np.dot(sigma_legal, q_hat))

                regret_full = np.zeros(NUM_ACTIONS, dtype=np.float32)
                kept_mask = np.zeros(NUM_ACTIONS, dtype=bool)
                for i, action in enumerate(kept_actions):
                    gi = _action_index(action)
                    regret_full[gi] = q_hat[i] - baseline
                    kept_mask[gi] = True
                buf.add(
                    ReservoirSample(
                        features=pad_tokens(tokens_h, seq_cap=self.seq_cap),
                        target=regret_full,
                        action_mask=kept_mask,
                        iteration=iteration,
                    )
                )
                added[game_index] += 1

                if self.value_sink is not None:
                    self.value_sink(tokens_h, driver, baseline, iteration)

            b_i_probs = uniform_policy_production(tokens_h, mask)
            _sample_and_apply(driver, legal, b_i_probs, rng)

        return added[game_index]

    def _rollout_coro(
        self,
        driver: GameDriver,
        stream: int,
        traverser: int,
        rng: random.Random,
        backend: SigmaBackend,
    ) -> Iterator:
        """Coroutine restatement of ``PRTCFRProductionWorker._rollout_to_terminal``
        for one CRN replicate: play both players under sigma^t to termination,
        yielding a _Query per step; return the traverser's utility. Closes its
        own driver and drops its hidden stream when done (finite Go handle pool /
        carried-hidden memory)."""
        try:
            steps = 0
            while not driver.is_terminal():
                steps += 1
                if steps > self.max_trajectory_steps:
                    break
                actor = driver.current_player()
                if actor == -1:
                    break
                legal = driver.legal_actions()
                if not legal:
                    break
                mask = _legal_mask(legal)
                tokens = driver.tokens(actor)
                probs = yield _Query(stream, actor, tokens, mask)
                _sample_and_apply(driver, legal, probs, rng)
            return driver.utility(traverser)
        finally:
            driver.close()
            backend.drop(stream)
