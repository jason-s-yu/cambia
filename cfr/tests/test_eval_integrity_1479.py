"""Tests for the three eval-integrity defects of cambia-1479.

1. A policy that raises inside an estimator used to be caught and the playout
   broken off, so a broken agent produced a quietly low exploitability number
   instead of a failed run. Policy exceptions now leave as ``PolicyError``; the
   failures that ARE absorbed are counted onto the result as ``policy_errors``.
2. Nothing advanced the belief of a policy that owns one (every wrapper in
   ``src.evaluate_agents``): it was built at the deal and stayed there for the
   whole measurement. The search state now adopts that handle so the engine
   moves it with every applied action, with ``frozen_beliefs=True`` reproducing
   the old protocol for a historical comparison.
3. ``lbr._make_strong_opponent`` gated the Tier-B strong opponent on an
   ``accepts_game_view`` marker no baseline set, so every Tier-B run fell back
   to UniformRandomPolicy. The marker is claimed here per class, and claimed
   only for classes this file drives through a whole game on a GoEngine.
"""

import random
from dataclasses import dataclass, field

import pytest

from src.agents.baseline_agents import (
    AggressiveSnapAgent,
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    RandomAgent,
    RandomLateCambiaAgent,
    RandomNoCambiaAgent,
)
from src.cfr.ismcts_br import ismcts_br
from src.cfr.lbr import (
    GoSearchState,
    PolicyError,
    ToleratedFailures,
    _accepts_game_view,
    _begin_episode,
    _make_strong_opponent,
    choose_action_pos,
    collect_infosets,
    tier_b_lbr,
)
from src.cfr.sampled_lbr import sampled_lbr
from src.constants import ActionCallCambia
from src.ffi.bridge import GoAgentState

# ---------------------------------------------------------------------------
# Config stubs (same shape as tests/test_lbr_tier_b.py, kept local so this
# module stands alone)
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


# ---------------------------------------------------------------------------
# Planted policies
# ---------------------------------------------------------------------------


class _RaisingWrapper(_UniformWrapper):
    """Raises on its Nth decision, standing in for a broken agent."""

    def __init__(self, config, raise_after: int = 0, seed: int = 0):
        super().__init__(config, seed)
        self.raise_after = raise_after
        self.calls = 0

    def choose_action(self, view, legal_actions):
        self.calls += 1
        if self.calls > self.raise_after:
            raise ZeroDivisionError("planted policy failure")
        return super().choose_action(view, legal_actions)


class _IllegalActionWrapper(_UniformWrapper):
    """Returns an action outside the legal set: absorbed, but counted."""

    def choose_action(self, view, legal_actions):
        return ActionCallCambia()


class _BeliefWrapper(_UniformWrapper):
    """A policy that owns its belief the way the evaluation wrappers do.

    ``initialize_state`` builds a GoAgentState for the seat and
    ``belief_handle`` hands it out, which is exactly the surface
    ``GoSearchState.adopt_agent_belief`` looks for.
    """

    def __init__(self, config, seed: int = 0):
        super().__init__(config, seed)
        self._state = None
        self.token_lens = []

    def initialize_state(self, view):
        self.release()
        self._state = GoAgentState(view, self.player_id)

    def belief_handle(self) -> int:
        return -1 if self._state is None else int(self._state.handle)

    def choose_action(self, view, legal_actions):
        self.token_lens.append(int(self._state.token_len()))
        return super().choose_action(view, legal_actions)

    def release(self) -> None:
        if self._state is not None:
            self._state.close()
            self._state = None


# ---------------------------------------------------------------------------
# Defect 1: a raising policy fails the run
# ---------------------------------------------------------------------------


def test_choose_action_pos_wraps_the_policy_exception_with_context():
    config = _Config()
    state = GoSearchState.new(config.cambia_rules, 11)
    try:
        with pytest.raises(PolicyError) as excinfo:
            choose_action_pos(
                _RaisingWrapper(config),
                state.view(),
                state.legal_actions(),
                seat=0,
                phase="unit probe",
            )
    finally:
        state.close()
    message = str(excinfo.value)
    assert "_RaisingWrapper" in message
    assert "unit probe" in message
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_collector_does_not_swallow_a_raising_policy():
    config = _Config()
    with pytest.raises(PolicyError):
        collect_infosets(_RaisingWrapper(config), config, num_infosets=5, seed=3)


def test_tier_a_does_not_swallow_a_raising_policy():
    config = _Config()
    with pytest.raises(PolicyError):
        sampled_lbr(
            _RaisingWrapper(config, raise_after=2),
            config,
            num_infosets=5,
            br_rollouts_per_infoset=2,
            seed=3,
        )


def test_tier_b_does_not_swallow_a_raising_policy():
    config = _Config()
    with pytest.raises(PolicyError):
        tier_b_lbr(
            _RaisingWrapper(config, raise_after=2),
            config,
            num_infosets=5,
            br_rollouts_per_infoset=2,
            seed=3,
        )


def test_ismcts_br_does_not_swallow_a_raising_opponent():
    config = _Config()
    with pytest.raises(PolicyError):
        ismcts_br(
            _UniformWrapper(config),
            config,
            num_infosets=5,
            opponent_factory=lambda seat, cfg: _RaisingWrapper(cfg),
        )


@pytest.mark.parametrize("estimator", [sampled_lbr, tier_b_lbr])
def test_an_absorbed_failure_is_counted_on_the_result(estimator):
    """An action outside the legal set is recoverable, so it is absorbed -- but
    the count reaches the result so the row cannot read as a clean run."""
    config = _Config()
    result = estimator(
        _IllegalActionWrapper(config),
        config,
        num_infosets=6,
        br_rollouts_per_infoset=2,
        seed=5,
    )
    assert result["policy_errors"] > 0
    assert result["policy_error_detail"].get("illegal_action", 0) > 0


def test_a_clean_run_reports_no_absorbed_failures():
    config = _Config()
    result = sampled_lbr(
        _UniformWrapper(config),
        config,
        num_infosets=6,
        br_rollouts_per_infoset=2,
        seed=5,
    )
    assert result["policy_errors"] == 0
    assert result["policy_error_detail"] == {}


def test_tolerated_failures_logs_once_per_kind(caplog):
    counter = ToleratedFailures()
    with caplog.at_level("WARNING"):
        for _ in range(5):
            counter.record("illegal_action", "probe")
    assert counter.total == 5
    assert counter.as_dict() == {"illegal_action": 5}
    logged = [r for r in caplog.records if "illegal_action" in r.getMessage()]
    assert len(logged) == 1


# ---------------------------------------------------------------------------
# Defect 2: the agent's belief advances with the game
# ---------------------------------------------------------------------------


def test_the_search_state_adopts_the_agents_own_belief():
    config = _Config()
    agent = _BeliefWrapper(config)
    state = GoSearchState.new(config.cambia_rules, 21)
    try:
        own_handle = state.belief_handle(0)
        _begin_episode(state, agent, 0)
        assert state.belief_handle(0) == agent.belief_handle()
        assert state.belief_handle(0) != own_handle
        assert state.belief_handle(1) == int(state.a1.handle)
    finally:
        agent.release()
        state.close()


def test_frozen_beliefs_leaves_the_agent_on_its_own_unadvanced_handle():
    config = _Config()
    agent = _BeliefWrapper(config)
    state = GoSearchState.new(config.cambia_rules, 21)
    try:
        _begin_episode(state, agent, 0, frozen_beliefs=True)
        assert state.belief_handle(0) == int(state.a0.handle)
        assert state.belief_handle(0) != agent.belief_handle()
    finally:
        agent.release()
        state.close()


def test_collection_advances_the_agents_belief():
    """The whole defect in one assertion: under the default protocol the agent's
    token stream grows as the trajectory is played; under the frozen protocol
    every decision sees the stream it had at the deal."""
    config = _Config()

    advancing = _BeliefWrapper(config)
    try:
        collect_infosets(advancing, config, num_infosets=12, seed=31)
    finally:
        advancing.release()

    frozen = _BeliefWrapper(config)
    try:
        collect_infosets(frozen, config, num_infosets=12, seed=31, frozen_beliefs=True)
    finally:
        frozen.release()

    assert advancing.token_lens, "the agent was never asked to choose"
    assert len(set(advancing.token_lens)) > 1, "belief never advanced"
    assert len(set(frozen.token_lens)) == 1, "frozen belief moved"


@pytest.mark.parametrize("estimator", [sampled_lbr, tier_b_lbr])
def test_the_result_names_the_belief_protocol(estimator):
    config = _Config()
    advancing = estimator(
        _UniformWrapper(config), config, num_infosets=4, br_rollouts_per_infoset=2
    )
    frozen = estimator(
        _UniformWrapper(config),
        config,
        num_infosets=4,
        br_rollouts_per_infoset=2,
        frozen_beliefs=True,
    )
    assert advancing["belief_protocol"] == "advancing"
    assert frozen["belief_protocol"] == "frozen"


# ---------------------------------------------------------------------------
# Defect 3: Tier B gets its strong opponent
# ---------------------------------------------------------------------------

#: Every baseline claiming ``accepts_game_view``. HumanPlayerAgent is
#: deliberately absent: it reads a move off stdin, so it cannot be driven
#: headless and has no business being an automated opponent.
_GAME_VIEW_BASELINES = [
    RandomAgent,
    RandomNoCambiaAgent,
    RandomLateCambiaAgent,
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
]


@pytest.mark.parametrize("agent_class", _GAME_VIEW_BASELINES)
def test_a_marked_baseline_plays_a_whole_game_on_the_go_engine(agent_class):
    """The marker is a claim about behaviour, so it is verified by behaviour:
    the baseline plays every one of its seat's decisions against a real GoEngine
    and each choice comes back inside the engine's legal set."""
    config = _Config()
    assert _accepts_game_view(agent_class(1, config))

    rng = random.Random(0)
    decisions = 0
    for seed in (101, 202, 303):
        agent = agent_class(1, config)
        state = GoSearchState.new(config.cambia_rules, seed)
        try:
            turns = 0
            while not state.is_terminal() and turns < 200:
                turns += 1
                acting = state.acting_player()
                if acting == -1:
                    break
                legal = state.legal_actions()
                if not legal:
                    break
                if acting == 1:
                    chosen = agent.choose_action(state.view(), legal)
                    assert chosen in legal, f"{agent_class.__name__} chose {chosen!r}"
                    pos = legal.index(chosen)
                    decisions += 1
                else:
                    pos = rng.randrange(len(legal))
                if not state.apply_index(state.legal_indices()[pos]):
                    break
        finally:
            state.close()
    assert decisions > 0, f"{agent_class.__name__} never got a decision"


def test_the_strong_opponent_factory_builds_the_heuristic_baseline():
    config = _Config()
    assert isinstance(_make_strong_opponent(1, config), ImperfectGreedyAgent)


def test_tier_b_measures_against_the_strong_opponent():
    """The regression the fallback hid: a Tier-B row that says
    UniformRandomPolicy is a Tier-A continuation wearing a Tier-B label."""
    config = _Config()
    result = tier_b_lbr(
        _UniformWrapper(config),
        config,
        num_infosets=6,
        br_rollouts_per_infoset=2,
        seed=42,
    )
    assert result["rollout_opponent"] == "ImperfectGreedyAgent"
    assert result["policy_errors"] == 0
