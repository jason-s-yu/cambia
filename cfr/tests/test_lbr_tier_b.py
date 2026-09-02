"""Tests for Tier-B sampled LBR, the collector sample-count fix (BUG-3), and the
Go-engine search substrate the estimators were ported onto (cambia-1427).

Tier-B LBR (cfr/src/cfr/lbr.py) tightens the Tier-A loose lower bound by:
  - generating trajectories against a strong fixed opponent (not random), and
  - rolling out the BR continuation under the agent's own policy at seat 0
    plus the strong opponent at seat 1 (agent-policy rollouts).

The collector also fixes BUG-3: the Tier-A sampler sized games_needed/sample_prob
for an assumed 40 P0 decisions/game while real Cambia games average ~3 under
random play, so requesting N infosets collected ~0.07*N. The corrected collector
loops until it has the requested count (subject to a safety cap).

Substrate coverage added with the port: branching is now a state_save /
state_restore rewind rather than a deep copy, and every playout holds FFI
handles out of a finite pool, so both the exactness of the rewind and the
absence of handle leaks are pinned here.
"""

import random
from dataclasses import dataclass, field

import pytest

from src.cfr.lbr import (
    DEFAULT_ROLLOUT_OPPONENT,
    DEFAULT_TRAJECTORY_OPPONENT,
    GoSearchState,
    _make_random_opponent,
    _make_strong_opponent,
    collect_infosets,
    replay_infoset,
    tier_b_lbr,
)
from src.cfr.sampled_lbr import sampled_lbr
from src.ffi.bridge import GoAgentState, get_handle_pool_stats

# ---------------------------------------------------------------------------
# Minimal config stubs (mirror tests/test_sampled_lbr.py, plus agents.greedy)
# ---------------------------------------------------------------------------


@dataclass
class _RulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 200
    lockCallerHand: bool = True
    num_decks: int = 1


@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(default_factory=_GreedyAgentConfig)


@dataclass
class _Config:
    cambia_rules: _RulesConfig = field(default_factory=_RulesConfig)
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _UniformWrapper:
    """Uniform-random target on the cambia-1427 policy boundary."""

    accepts_game_view = True

    def __init__(self, config, seed: int = 0):
        self._config = config
        self.player_id = 0
        self._rng = random.Random(seed)

    def choose_action(self, view, legal_actions):
        actions = list(legal_actions)
        return actions[self._rng.randrange(len(actions))]


class _RecordingWrapper(_UniformWrapper):
    """Uniform wrapper that counts choose_action calls (to prove the agent
    policy is actually used inside BR-continuation rollouts for Tier B)."""

    def __init__(self, config, seed: int = 0):
        super().__init__(config, seed)
        self.calls = 0

    def choose_action(self, view, legal_actions):
        self.calls += 1
        return super().choose_action(view, legal_actions)


class _HookRecordingWrapper(_UniformWrapper):
    """Records which optional episode hooks the estimator drove, and with what."""

    def __init__(self, config, seed: int = 0):
        super().__init__(config, seed)
        self.bound = []
        self.inits = 0
        self.transitions = []

    def bind_go_state(self, view, agent_state):
        self.bound.append((view, agent_state))

    def initialize_state(self, view):
        self.inits += 1

    def observe_transition(self, view, action, actor):
        self.transitions.append((action, actor))


# ---------------------------------------------------------------------------
# Tier-B result-shape and invariants
# ---------------------------------------------------------------------------


def test_tier_b_valid_result():
    config = _Config()
    agent = _UniformWrapper(config)
    result = tier_b_lbr(
        agent, config, num_infosets=20, br_rollouts_per_infoset=4, seed=42
    )

    assert isinstance(result, dict)
    for key in (
        "exploitability",
        "num_infosets_sampled",
        "std_err",
        "tier",
        "rollout_opponent",
        "seed",
    ):
        assert key in result, f"missing key {key}"
    assert result["tier"] == "B"
    assert result["seed"] == 42
    assert isinstance(result["exploitability"], float)
    assert isinstance(result["num_infosets_sampled"], int)
    assert result["num_infosets_sampled"] >= 0


def test_tier_b_non_negative_exploitability():
    config = _Config()
    agent = _UniformWrapper(config)
    result = tier_b_lbr(agent, config, num_infosets=40, br_rollouts_per_infoset=4, seed=7)
    assert result["exploitability"] >= 0.0
    assert result["std_err"] >= 0.0


def test_tier_b_deterministic_seed():
    config = _Config()
    a = _UniformWrapper(config)
    b = _UniformWrapper(config)
    ra = tier_b_lbr(a, config, num_infosets=25, br_rollouts_per_infoset=4, seed=99)
    rb = tier_b_lbr(b, config, num_infosets=25, br_rollouts_per_infoset=4, seed=99)
    assert ra["exploitability"] == rb["exploitability"]
    assert ra["num_infosets_sampled"] == rb["num_infosets_sampled"]


def test_tier_b_uses_agent_policy_in_rollouts():
    """The agent wrapper must be invoked for seat-0 continuation play, not only
    at the sampled infoset. With >1 rollout per action and multi-step games the
    call count must far exceed the number of sampled infosets."""
    config = _Config()
    agent = _RecordingWrapper(config)
    result = tier_b_lbr(agent, config, num_infosets=15, br_rollouts_per_infoset=4, seed=3)
    n = result["num_infosets_sampled"]
    if n == 0:
        pytest.skip("no infosets sampled in this tiny config")
    # Trajectory choose_action + per-infoset agent_value rollouts each call the
    # agent at least once; agent-policy continuation adds more. Strictly greater
    # than the sampled count proves continuation play used the agent.
    assert agent.calls > n


def test_tier_b_labels_the_rollout_opponent():
    """The row names the opponent the continuation was actually measured against,
    so a Tier-B run that fell back to uniform cannot read as a strong-opponent
    measurement (the port window: cambia-1426 has not yet put the heuristic
    baselines on the GameView protocol)."""
    config = _Config()
    result = tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=5, br_rollouts_per_infoset=2, seed=1
    )
    assert result["rollout_opponent"], "rollout_opponent must be populated"


# ---------------------------------------------------------------------------
# Policy boundary (cambia-1427 D1)
# ---------------------------------------------------------------------------


def test_policy_receives_a_game_view_and_named_actions():
    """The first argument satisfies the GameView protocol and the legal set is a
    list of GameAction NamedTuples, not raw indices."""
    from src.agents.game_view import GameView
    from src.constants import GameAction

    config = _Config()
    seen = {}

    class _Probe(_UniformWrapper):
        def choose_action(self, view, legal_actions):
            seen.setdefault("view", view)
            seen.setdefault("legal", list(legal_actions))
            return super().choose_action(view, legal_actions)

    collect_infosets(_Probe(config), config, num_infosets=3, seed=5)
    assert isinstance(seen["view"], GameView)
    assert seen["legal"], "policy saw an empty legal set"
    assert all(isinstance(a, GameAction) for a in seen["legal"])
    assert len(set(seen["legal"])) == len(seen["legal"]), "duplicate legal actions"


def test_legal_actions_are_in_ascending_engine_index_order():
    """The ordering handed to a policy is the engine's ascending index order,
    which is stable across processes (the Python engine's legal-action set was
    not: GameAction carries a str tag and set order is hash-salted)."""
    from src.agents.action_codec import actions_from_indices

    config = _Config()
    state = GoSearchState.new(config.cambia_rules, 31337)
    try:
        indices = state.legal_indices()
        assert indices == sorted(indices), "legal_indices is not ascending"
        assert state.legal_actions() == actions_from_indices(indices)
    finally:
        state.close()


def test_episode_hooks_are_driven_along_trajectories():
    """bind_go_state / initialize_state / observe_transition are all fed during
    collection, and observe_transition sees every applied action with its actor."""
    config = _Config()
    agent = _HookRecordingWrapper(config)
    collect_infosets(agent, config, num_infosets=5, seed=13)

    assert agent.inits > 0, "initialize_state never called"
    assert agent.bound, "bind_go_state never called"
    from src.ffi.bridge import GoAgentState

    view, agent_state = agent.bound[0]
    assert isinstance(agent_state, GoAgentState)
    assert agent.transitions, "observe_transition never called"
    assert all(actor in (0, 1) for _, actor in agent.transitions)


# ---------------------------------------------------------------------------
# Go search substrate: rewind exactness, replay exactness, handle hygiene
# ---------------------------------------------------------------------------


def test_state_rewind_is_exact():
    """state_save/state_restore must return the engine to the same decision."""
    config = _Config()
    state = GoSearchState.new(config.cambia_rules, 4242)
    try:
        before_legal = state.legal_indices()
        before_actor = state.acting_player()
        snap = state.save()
        try:
            for idx in before_legal[:1]:
                assert state.apply_index(idx)
            assert state.legal_indices() != before_legal or state.acting_player() != (
                before_actor
            ), "the test action did not change the state, so rewind proves nothing"
            state.restore(snap)
            assert state.legal_indices() == before_legal
            assert state.acting_player() == before_actor
        finally:
            GoSearchState.free_snapshot(snap)
    finally:
        state.close()


def test_replay_reconstructs_the_sampled_decision():
    """An infoset is stored as (deal_seed, action prefix); replaying it must land
    on exactly the recorded legal set."""
    config = _Config()
    agent = _UniformWrapper(config)
    infosets = collect_infosets(agent, config, num_infosets=8, seed=77)
    assert infosets, "collector produced nothing to replay"
    for infoset in infosets:
        state = replay_infoset(config.cambia_rules, infoset, agent)
        try:
            assert state.acting_player() == 0
            assert tuple(state.legal_indices()) == infoset.legal_indices
        finally:
            state.close()


def test_two_seat_guard():
    """The 146-action space is 2-seat only; a 3+ seat game must be refused rather
    than panic the Go runtime (cambia-1171)."""

    @dataclass
    class _ThreeSeatRules(_RulesConfig):
        num_players: int = 3

    with pytest.raises(ValueError, match="2-seat"):
        GoSearchState.new(_ThreeSeatRules(), 1)


def test_no_handle_leak_across_a_full_estimate():
    """Every playout allocates a game handle plus two agent handles, and every
    branch a snapshot handle, all out of finite pools; a full Tier-A and Tier-B
    run must return every one of them.

    The pool keys are exactly ``games`` / ``agents`` / ``snapshots``; asserting
    on any other name would make this test silently vacuous.
    """
    config = _Config()
    before = get_handle_pool_stats()
    assert set(before) >= {"games", "agents", "snapshots"}, before

    tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=6, br_rollouts_per_infoset=2, seed=8
    )
    sampled_lbr(
        _UniformWrapper(config), config, num_infosets=6, br_rollouts_per_infoset=2, seed=8
    )

    after = get_handle_pool_stats()
    for key in ("games", "agents", "snapshots"):
        assert (
            after[key] == before[key]
        ), f"handle leak in {key}: {before[key]} -> {after[key]} ({before} -> {after})"


def test_no_handle_leak_when_the_policy_raises_during_collection():
    """A policy that throws where the estimator does NOT catch must still not
    strand a game/agent handle: the finite pool is unrecoverable within a process
    once exhausted.

    The throw is placed in ``collect_infosets``, which calls ``choose_action``
    unguarded, so the exception really does propagate out with a state open --
    the path the ``finally: state.close()`` exists for. (A throw inside a branch
    rollout would instead be swallowed by ``_agent_policy_rollout``, making this
    assertion vacuous.)
    """

    class _Exploding(_UniformWrapper):
        def choose_action(self, view, legal_actions):
            raise RuntimeError("boom")

    config = _Config()
    before = get_handle_pool_stats()
    with pytest.raises(RuntimeError, match="boom"):
        collect_infosets(_Exploding(config), config, num_infosets=6, seed=4)
    after = get_handle_pool_stats()
    for key in ("games", "agents", "snapshots"):
        assert (
            after[key] == before[key]
        ), f"handle leak in {key} on the error path: {before} -> {after}"


def test_no_handle_leak_when_replay_fails():
    """A failed infoset replay is caught and skipped by both tiers; the state it
    had already opened must still be released."""
    config = _Config()
    agent = _UniformWrapper(config)
    infosets = collect_infosets(agent, config, num_infosets=4, seed=6)
    assert infosets

    # A prefix the engine cannot accept forces replay_infoset down its except
    # branch, which must close the state it opened before re-raising.
    broken = infosets[0]._replace(action_prefix=(145,) * 4)
    before = get_handle_pool_stats()
    with pytest.raises(Exception):
        replay_infoset(config.cambia_rules, broken, agent)
    after = get_handle_pool_stats()
    for key in ("games", "agents", "snapshots"):
        assert (
            after[key] == before[key]
        ), f"handle leak in {key} on the replay-failure path: {before} -> {after}"


# ---------------------------------------------------------------------------
# BUG-3: collector returns ~requested sample count (not ~0.07x)
# ---------------------------------------------------------------------------


def test_collector_meets_requested_count():
    """Requesting N infosets must collect ~N (the 14x over-request fix).

    Before the fix, num=200 collected ~14 samples (40/3 under-collection).
    After the fix the collector loops until it reaches the target (or the
    safety cap), so it must collect at least the requested count.
    """
    config = _Config()
    agent = _UniformWrapper(config)
    requested = 200
    infosets = collect_infosets(
        agent,
        config,
        num_infosets=requested,
        seed=11,
        trajectory_opponent_factory=DEFAULT_TRAJECTORY_OPPONENT,
    )
    # Must reach the requested count, never ~0.07x.
    assert len(infosets) >= requested, (
        f"collector under-collected: got {len(infosets)} of {requested} "
        f"(BUG-3 regression - sized for wrong decisions/game)"
    )
    # And it must not wildly over-collect (collection stops at the target).
    assert len(infosets) <= requested + 50


def test_collector_caps_to_avoid_runaway():
    """With an unreachable target the collector must terminate via its safety
    cap rather than loop forever, and return what it collected."""
    config = _Config()
    agent = _UniformWrapper(config)
    infosets = collect_infosets(
        agent,
        config,
        num_infosets=10_000_000,
        seed=5,
        trajectory_opponent_factory=DEFAULT_TRAJECTORY_OPPONENT,
        max_games=50,  # hard cap forces early termination
    )
    assert len(infosets) >= 0  # terminates; count bounded by max_games


def test_sampled_lbr_tier_a_also_meets_count():
    """The production Tier-A path (sampled_lbr) must also honor the requested
    count now that it shares the corrected collector."""
    config = _Config()
    agent = _UniformWrapper(config)
    requested = 150
    result = sampled_lbr(
        agent, config, num_infosets=requested, br_rollouts_per_infoset=2, seed=21
    )
    assert (
        result["num_infosets_sampled"] >= requested
    ), f"Tier-A under-collected: {result['num_infosets_sampled']} of {requested}"


# ---------------------------------------------------------------------------
# The two seat-1 roles are separate knobs (cambia-1793)
# ---------------------------------------------------------------------------


class _BeliefCarryingOpponent:
    """A continuation opponent that owns a belief, as the eval wrappers do.

    Records, at every binding, the length of the token stream it was handed
    against the length a belief built from nothing at that same position would
    have. That difference is the public history the opponent used to be denied.
    """

    accepts_game_view = True
    binds: list = []

    def __init__(self, player_id, config):
        self.player_id = player_id
        self._bound = None

    def bind_go_state(self, view, agent_state):
        self._bound = agent_state
        fresh = GoAgentState(view, self.player_id)
        try:
            type(self).binds.append(
                (int(agent_state.token_len()), int(fresh.token_len()))
            )
        finally:
            fresh.close()

    def choose_action(self, view, legal_actions):
        assert self._bound is not None, "decided before being handed a belief"
        return list(legal_actions)[0]


def test_tier_b_defaults_to_a_uniform_trajectory_and_a_strong_continuation():
    """The default pairing is the one whose number means what Tier B says it
    means: the sampled distribution held at Tier A's uniform opponent, and only
    the continuation strengthened (cambia-1793).

    A Tier-B row recorded before this default ran strong at both seats, which
    is why the row names them: a number measured under one pairing cannot be
    read against the other.
    """
    config = _Config()
    result = tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=6, br_rollouts_per_infoset=2, seed=2
    )
    assert result["trajectory_opponent"] == "UniformRandomPolicy"
    assert result["continuation_opponent"] == "ImperfectGreedyAgent"
    # Pinned against the factories themselves, so repointing a constant cannot
    # move the default without this failing.
    assert DEFAULT_TRAJECTORY_OPPONENT is _make_random_opponent
    assert DEFAULT_ROLLOUT_OPPONENT is _make_strong_opponent


def test_tier_b_row_names_both_seat_one_roles():
    """A Tier-B row says which distribution it measured as well as how hard the
    continuation was, so the two can never move together unnoticed again."""
    config = _Config()
    result = tier_b_lbr(
        _UniformWrapper(config),
        config,
        num_infosets=6,
        br_rollouts_per_infoset=2,
        seed=1,
        trajectory_opponent_factory=_make_random_opponent,
        rollout_opponent_factory=_make_strong_opponent,
    )
    assert result["trajectory_opponent"] == "UniformRandomPolicy"
    assert result["continuation_opponent"] == "ImperfectGreedyAgent"
    # The pre-cambia-1793 name still carries the continuation opponent, which is
    # what it always held; consumers written against it keep working.
    assert result["rollout_opponent"] == result["continuation_opponent"]


def test_tier_b_factories_drive_the_roles_they_name():
    """The trajectory factory builds only during collection and the
    continuation factory only inside the rollouts, so a leg can hold the
    sampled distribution fixed and vary the continuation alone."""
    config = _Config()
    built = {"trajectory": 0, "continuation": 0}

    def _traj(player_id, cfg):
        built["trajectory"] += 1
        return _make_random_opponent(player_id, cfg)

    def _cont(player_id, cfg):
        built["continuation"] += 1
        return _make_strong_opponent(player_id, cfg)

    result = tier_b_lbr(
        _UniformWrapper(config),
        config,
        num_infosets=8,
        br_rollouts_per_infoset=3,
        seed=42,
        trajectory_opponent_factory=_traj,
        rollout_opponent_factory=_cont,
    )
    if result["num_infosets_sampled"] == 0:
        pytest.skip("no infosets sampled in this tiny config")
    assert built["trajectory"] > 0
    # One continuation opponent per rollout, so many more than there are
    # infosets; the trajectory factory never enters that loop.
    assert built["continuation"] > result["num_infosets_sampled"]


def test_continuation_opponent_is_seeded_with_the_public_history():
    """A continuation opponent is built at the infoset, not at the deal, so it
    observed nothing on the way there. It must be handed the belief the engine
    carried through the sampled prefix (cambia-1793, from cambia-1479 F4);
    before that it took its first decision knowing only the current position."""
    config = _Config()
    _BeliefCarryingOpponent.binds = []
    result = tier_b_lbr(
        _UniformWrapper(config),
        config,
        num_infosets=10,
        br_rollouts_per_infoset=2,
        seed=4,
        rollout_opponent_factory=_BeliefCarryingOpponent,
    )
    if result["num_infosets_sampled"] == 0:
        pytest.skip("no infosets sampled in this tiny config")
    binds = _BeliefCarryingOpponent.binds
    assert binds, "no continuation opponent was handed a belief"
    assert result["continuation_opponent"] == "_BeliefCarryingOpponent"
    # A hook that raised would be absorbed onto the count rather than failing
    # the run, so a silent bind failure cannot pass this test.
    assert result["policy_errors"] == 0, result["policy_error_detail"]
    assert any(
        bound > fresh for bound, fresh in binds
    ), f"every bound belief was as empty as a fresh one: {binds[:5]}"
    assert all(
        bound >= fresh for bound, fresh in binds
    ), "a bound belief held less history than a belief built from nothing"
