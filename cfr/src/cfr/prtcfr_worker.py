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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..encoding import NUM_ACTIONS, action_to_index
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..sequence_encoding import SEQ_CAP
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


def new_production_driver(
    seed: int, house_rules: Optional[Any] = None, num_players: int = 2
) -> PythonEngineGameDriver:
    """Build a fresh production game + driver for ``seed``.

    Uses the engine's own internal deal (``CambiaGameState(house_rules=...,
    _rng=...)``, the same construction path tiny_solver's tree builder uses),
    NOT the Go-deal-matching test helper: self-play generation only needs an
    internally consistent engine instance here, independent of Go-parity,
    which is covered separately by the FFI cross-path parity tests.
    """
    from ..config import CambiaRulesConfig
    from ..game.engine import CambiaGameState

    if house_rules is None:
        house_rules = CambiaRulesConfig()
        house_rules.allowDrawFromDiscardPile = True
        house_rules.allowReplaceAbilities = True
        house_rules.allowOpponentSnapping = True
        house_rules.max_game_turns = 300
        house_rules.lockCallerHand = False

    game = CambiaGameState(house_rules=house_rules, _rng=random.Random(seed))
    init_hands = {p: list(game.players[p].hand) for p in range(num_players)}
    init_peeks = {
        p: tuple(game.players[p].initial_peek_indices) for p in range(num_players)
    }
    return PythonEngineGameDriver(game, init_hands, init_peeks)


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


def _legal_mask(legal: Sequence[Any]) -> np.ndarray:
    """NUM_ACTIONS-wide bool mask for a legal-action list (production driver)."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal:
        mask[action_to_index(a)] = True
    return mask


def _sample_legal_action(
    legal: Sequence[Any], probs: np.ndarray, rng: random.Random
) -> Any:
    """Sample one action object from ``legal`` given a NUM_ACTIONS-wide probs
    vector (indexed by action_to_index); renormalizes defensively so a
    slightly-off-mass sigma never raises."""
    weights = np.array([probs[action_to_index(a)] for a in legal], dtype=np.float64)
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
    ):
        self.sigma = sigma
        self.m_rollouts = m_rollouts
        self.seq_cap = seq_cap
        self.max_trajectory_steps = max_trajectory_steps
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
                if not child.apply(action):
                    # The engine rejected this nominally-"legal" action for
                    # the current pending sub-decision (the same pre-existing
                    # engine/ability-pending-chain interaction
                    # PythonEngineGameDriver.apply guards against). Exclude it
                    # from THIS decision's regret sample rather than
                    # fabricate a q-value for an action that was never
                    # actually applied to the clone.
                    continue
                kept_actions.append(action)
                kept_q.append(self._rollout_mc(child, traverser, seed_base))

            if kept_actions:
                q_hat = np.array(kept_q, dtype=np.float64)
                sigma_legal = np.array(
                    [sigma_node[action_to_index(a)] for a in kept_actions],
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
                    gi = action_to_index(action)
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
            total += self._rollout_to_terminal(rollout_driver, traverser, rng)
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
