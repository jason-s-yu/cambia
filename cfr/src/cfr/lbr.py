"""
cfr/src/cfr/lbr.py

Tier-B sampled Local Best Response (LBR) for approximate exploitability, plus
the shared Go-engine search substrate the LBR / ISMCTS-BR estimators run on.

Background
----------
Tier-A LBR (``src.cfr.sampled_lbr.sampled_lbr``) estimates exploitability with:
  - trajectories generated against a uniform-random opponent, and
  - best-response continuation rollouts where BOTH seats play uniform-random.

Random rollouts make Tier-A a LOOSE lower bound: a real adversary plays well
after a deviation, not randomly, so Tier-A understates the true exploitability.
The relative ordering across agents (measured identically) is trustworthy; the
absolute number is not.

Tier-B tightens the bound by making the continuation realistic:
  - trajectories are generated against a STRONG fixed opponent (heuristic), and
  - BR continuation rollouts roll seat 0 under the AGENT'S OWN policy and seat 1
    under the strong opponent ("agent-policy rollouts, strong trajectory
    opponent", per the E3 report / report.md build list).

Because the agent under test plays on after its own one-step deviation rather
than reverting to random, the best-response value is measured against a
realistic continuation, yielding a higher (tighter) exploitability estimate.

The 14x sampler fix
-------------------
The Tier-A collector sized ``games_needed`` and the per-decision sample
probability for an assumed 40 P0 decisions/game, but real Cambia games average
~3 P0 decisions/game under random play. Requesting N infosets therefore
collected only ~0.07*N. ``collect_infosets`` removes the brittle
decisions/game assumption: it plays games and samples P0 decisions until it has
the requested count (or hits a safety cap on games played), so requesting N
collects ~N. ``src.cfr.sampled_lbr.sampled_lbr`` consumes this same collector,
so the production ``cambia evaluate --lbr`` path is fixed too.

Engine (cambia-1427)
--------------------
The search runs on the Go engine through the FFI bridge, not on the Python
reference engine. Each playout owns ONE
``(GoEngine, GoAgentState seat 0, GoAgentState seat 1)`` triple -- the
``GoSearchState`` below -- and branches by rewinding it with the token-inclusive
``bridge.state_save`` / ``bridge.state_restore`` pair rather than deep-copying a
Python game object. Rewinding restores the game, BOTH seats' belief state, AND
both token streams together, which is what makes a one-ply deviation measurable
against a correctly rewound agent: under the old deep-copy path the shared
agent wrapper carried whatever prefix it had accumulated at collection time into
every branch rollout.

Sampled infosets are recorded as ``(deal_seed, applied action-index prefix)``
and REPLAYED on demand rather than held open, so the estimator never pins more
than one triple's worth of the finite FFI handle pool at a time. Replay is
exact: the Go deal is a pure function of the seed and the engine is
deterministic given an action sequence.

Policy boundary (cambia-1427 D1)
--------------------------------
A policy is anything exposing ``choose_action(view, legal_actions)`` where
``view`` satisfies the ``src.agents.game_view.GameView`` protocol (``GoEngine``
does) and ``legal_actions`` is a list of ``src.constants`` ``GameAction``
NamedTuples decoded from the engine's 146-bit legal mask. Optional hooks, called
when present:

  - ``bind_go_state(view, agent_state)``: hands a Go-native wrapper its own
    ``GoAgentState`` belief/token handle for the episode. This is the handle the
    save/restore rewind keeps consistent.
  - ``initialize_state(view)``: per-episode reset, at the start of a game and at
    the start of each infoset replay.
  - ``observe_transition(view, action, actor)``: post-action frame, fed for
    every applied action along a TRAJECTORY (collection and replay) so a wrapper
    that keeps its own Python-side stream stays in step. Deliberately NOT fed
    during branch rollouts: a rollout is rewound by ``state_restore``, which
    rewinds the ``GoAgentState`` but cannot rewind wrapper-private Python state,
    so feeding it there would leave such a wrapper desynced for the rest of the
    infoset.

Both tiers are pure eval-time measurement: no network or harness state is
mutated.
"""

import logging
import math
import random as _random_module
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from src.agents.action_codec import actions_from_mask, index_to_action
from src.agents.game_view import GameView
from src.constants import GameAction
from src.ffi.bridge import (
    GoAgentState,
    GoEngine,
    apply_games_batch,
    state_restore,
    state_save,
    state_snapshot_free,
)

logger = logging.getLogger(__name__)

# Exploitability is always measured from P0 (the agent under test).
_PLAYER_ID = 0
_OPPONENT_ID = 1

# Measured under random play (E3 report, 200-game sample): ~3 P0 decisions/game.
# Used only to size the games-played safety cap, never to gate collection.
_EST_P0_DECISIONS_PER_GAME = 3.0

# Factory signature: (player_id, config) -> policy exposing choose_action.
OpponentFactory = Callable[[int, Any], Any]


class SampledInfoset(NamedTuple):
    """A recorded P0 decision point, replayable on demand.

    Holds the deal seed and the action-index prefix that reaches the decision
    instead of a copy of the state: the Go deal is a pure function of the seed,
    so replaying the prefix reconstructs the exact game, both seats' beliefs and
    both token streams. Keeping the state open instead would pin three FFI
    handles per sampled infoset, and the pool is finite.
    """

    deal_seed: int
    action_prefix: Tuple[int, ...]
    legal_indices: Tuple[int, ...]
    agent_action_pos: int


class GoSearchState:
    """One ``(GoEngine, GoAgentState, GoAgentState)`` triple with rewind.

    The unit of search for every best-response estimator here. ``apply_index``
    goes through ``bridge.apply_games_batch`` because that is the only FFI path
    that advances the game AND appends to both agents' token streams (the
    single-agent ``cambia_agent_update`` export does not touch the token stream
    at all); length-1 batches are atomic, so a rejected action leaves nothing
    partially applied.

    ``save``/``restore`` are the token-inclusive ``cambia_state_save`` /
    ``cambia_state_restore`` pair: a same-handle rewind checkpoint, which is
    exactly right for a search that explores branches one at a time. (An
    independent, simultaneously-live branch would need ``state_clone`` and a
    fresh handle triple; nothing here needs one.)

    Owns its handles: ``close`` frees all three and is idempotent. Use as a
    context manager, or close in a ``finally`` -- the handle pool is finite.
    """

    __slots__ = ("engine", "a0", "a1", "_closed")

    def __init__(self, engine: GoEngine, a0: GoAgentState, a1: GoAgentState) -> None:
        self.engine = engine
        self.a0 = a0
        self.a1 = a1
        self._closed = False

    @classmethod
    def new(cls, house_rules: Any, seed: int) -> "GoSearchState":
        """Fresh deal from ``seed`` under ``house_rules``."""
        engine = GoEngine(seed=int(seed), house_rules=house_rules)
        return cls._with_agents(engine)

    @classmethod
    def from_deck(
        cls,
        house_rules: Any,
        deck_indices: Sequence[int],
        starting_player: int = 0,
    ) -> "GoSearchState":
        """Deal a pinned deck order (deterministic determinization injection)."""
        engine = GoEngine.from_deck(
            list(deck_indices),
            starting_player=starting_player,
            house_rules=house_rules,
        )
        return cls._with_agents(engine)

    @classmethod
    def _with_agents(cls, engine: GoEngine) -> "GoSearchState":
        num_players = int(engine.num_players())
        if num_players != 2:
            engine.close()
            raise ValueError(
                f"GoSearchState is a 2-seat search substrate, got "
                f"num_players={num_players}. Driving 3+ seats through the "
                "2-player 146-action space panics the Go runtime (its apply "
                "paths still resolve the opponent as 1-acting); an N-player "
                "best response has to go through nplayer_legal_actions_mask / "
                "apply_nplayer_action. See cambia-1171."
            )
        try:
            a0 = GoAgentState(engine, player_id=0)
        except Exception:
            engine.close()
            raise
        try:
            a1 = GoAgentState(engine, player_id=1)
        except Exception:
            a0.close()
            engine.close()
            raise
        return cls(engine, a0, a1)

    # --- Read surface ---

    def view(self) -> GameView:
        """The read-only game surface handed to policies."""
        return self.engine

    def agent_state(self, seat: int) -> GoAgentState:
        return self.a0 if seat == 0 else self.a1

    def acting_player(self) -> int:
        return int(self.engine.acting_player())

    def is_terminal(self) -> bool:
        return bool(self.engine.is_terminal())

    def utility(self, seat: int) -> float:
        return float(self.engine.get_utility()[seat])

    def legal_indices(self) -> List[int]:
        """Legal action indices, ascending.

        ``flatnonzero`` is already ascending, giving a canonical ordering that
        is stable across processes -- unlike the Python engine's
        ``get_legal_actions()`` set, whose iteration order varies with string
        hash randomization.
        """
        mask = self.engine.legal_actions_mask()
        return [int(i) for i in np.flatnonzero(mask)]

    def legal_actions(self) -> List[GameAction]:
        return actions_from_mask(self.legal_indices())

    # --- Mutation + rewind ---

    def apply_index(self, action_idx: int) -> bool:
        """Apply an action index; return False if the engine rejected it.

        A token-stream overflow is a different, non-retryable condition (the
        stream is already too long; a different action cannot help), so it
        propagates rather than being reported as a rejection.
        """
        try:
            apply_games_batch(
                [self.engine.handle],
                [self.a0.handle],
                [self.a1.handle],
                [int(action_idx)],
            )
            return True
        except RuntimeError as exc:
            if "overflow" in str(exc):
                raise
            return False

    def save(self) -> int:
        return state_save(self.engine.handle, self.a0.handle, self.a1.handle)

    def restore(self, snap_h: int) -> None:
        state_restore(self.engine.handle, snap_h, self.a0.handle, self.a1.handle)

    @staticmethod
    def free_snapshot(snap_h: int) -> None:
        state_snapshot_free(snap_h)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.a0.close()
        self.a1.close()
        self.engine.close()

    def __enter__(self) -> "GoSearchState":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class UniformRandomPolicy:
    """Uniform-random over the engine's ascending legal-action order.

    Driven by an injected ``random.Random`` so a whole estimator run is
    seed-deterministic. (``baseline_agents.RandomAgent`` draws from the GLOBAL
    ``random`` module over an unsorted set, which is reproducible only if the
    caller reseeds the global module first.)
    """

    __slots__ = ("player_id", "_rng")

    def __init__(self, player_id: int, rng: Optional[_random_module.Random] = None):
        self.player_id = player_id
        self._rng = rng if rng is not None else _random_module.Random()

    def choose_action(self, view: GameView, legal_actions) -> GameAction:
        actions = list(legal_actions)
        if not actions:
            raise ValueError(
                f"UniformRandomPolicy P{self.player_id} cannot choose from an "
                "empty legal set."
            )
        return actions[self._rng.randrange(len(actions))]


def _make_random_opponent(player_id: int, config: Any):
    """Uniform-random opponent, seeded off the global ``random`` stream.

    ``collect_infosets`` seeds that stream before a run, so a fresh opponent per
    game stays reproducible under the estimator's ``seed``.
    """
    return UniformRandomPolicy(player_id, _random_module.Random(
        _random_module.getrandbits(63)
    ))


# Set once the strong-opponent fallback has been reported (see below).
_STRONG_OPPONENT_WARNED = False


def _make_strong_opponent(player_id: int, config: Any):
    """The default strong fixed opponent (ImperfectGreedyAgent), if available.

    The heuristic baselines are being ported to the ``GameView`` protocol
    separately (cambia-1426). Until that lands they are still written against
    the Python reference engine and cannot read a ``GoEngine``, so this falls
    back to uniform-random and says so: a Tier-B run on the fallback is a Tier-A
    continuation wearing a Tier-B label, which the returned
    ``rollout_opponent`` field names so a row is never silently mislabelled.
    """
    global _STRONG_OPPONENT_WARNED
    try:
        from src.agents.baseline_agents import ImperfectGreedyAgent

        agent = ImperfectGreedyAgent(player_id, config)
        if not _accepts_game_view(agent):
            raise TypeError(
                "ImperfectGreedyAgent is not ported to the GameView protocol "
                "(cambia-1426)"
            )
        return agent
    except Exception as exc:  # JUSTIFIED: eval resilience across the port window
        # Warn once per process, not once per constructed opponent: Tier B builds
        # a fresh rollout opponent per branch rollout, so a per-call warning is
        # millions of identical lines on a production-sized run.
        if not _STRONG_OPPONENT_WARNED:
            _STRONG_OPPONENT_WARNED = True
            logger.warning(
                "lbr: strong opponent unavailable (%s); falling back to "
                "UniformRandomPolicy for the rest of this process. Tier-B "
                "numbers from this run are NOT measured against a strong "
                "continuation.",
                exc,
            )
        return _make_random_opponent(player_id, config)


def _accepts_game_view(agent: Any) -> bool:
    """True if ``agent`` declares itself runnable against a ``GameView``.

    The opt-in marker a ported baseline/wrapper sets; absent it, an agent is
    assumed to still want the Python reference engine.
    """
    return bool(getattr(agent, "accepts_game_view", False))


# Default Tier-B opponents: strong both for trajectory generation and for the
# adversary seat during agent-policy continuation rollouts.
DEFAULT_TRAJECTORY_OPPONENT: OpponentFactory = _make_strong_opponent
DEFAULT_ROLLOUT_OPPONENT: OpponentFactory = _make_strong_opponent


def _resolve_max_turns(config: Any) -> int:
    house_rules = config.cambia_rules
    max_turns = getattr(house_rules, "max_game_turns", 0)
    if max_turns <= 0:
        max_turns = 500
    return max_turns


def _notify(policy: Any, method: str, *args) -> bool:
    """Call an optional policy hook. Returns False if it raised."""
    fn = getattr(policy, method, None)
    if fn is None:
        return True
    try:
        fn(*args)
        return True
    except Exception as exc:  # JUSTIFIED: eval resilience for optional hooks
        logger.warning(
            "lbr: policy hook %s failed (%s: %s)", method, type(exc).__name__, exc
        )
        return False


def _begin_episode(state: GoSearchState, agent_wrapper: Any, seat: int) -> None:
    """Hand the agent its per-episode Go handles and reset its episode state."""
    _notify(agent_wrapper, "bind_go_state", state.view(), state.agent_state(seat))
    _notify(agent_wrapper, "initialize_state", state.view())


def hand_score_utility(view: GameView, seat: int, opponent_seat: int) -> float:
    """Utility estimate for a game cut short by the decision cap.

    Lower hand score wins, matching the Tier-A/Tier-B/ISMCTS timeout convention
    so the estimators stay comparable.
    """
    try:
        mine = sum(c.value for c in view.get_player_hand(seat))
        theirs = sum(c.value for c in view.get_player_hand(opponent_seat))
    except Exception:  # JUSTIFIED: eval resilience on odd states
        return 0.0
    if mine < theirs:
        return 1.0
    if mine > theirs:
        return -1.0
    return 0.0


def terminal_utility(
    state: GoSearchState, seat: int = _PLAYER_ID, opponent_seat: int = _OPPONENT_ID
) -> float:
    """Terminal utility for ``seat``, or a hand-score estimate on timeout."""
    if state.is_terminal():
        try:
            return state.utility(seat)
        except Exception:  # JUSTIFIED: eval resilience
            return 0.0
    return hand_score_utility(state.view(), seat, opponent_seat)


def replay_infoset(
    house_rules: Any, infoset: SampledInfoset, agent_wrapper: Any
) -> GoSearchState:
    """Rebuild the state at a sampled decision point.

    Replays the recorded action-index prefix onto a fresh deal, feeding the
    agent the same episode hooks the collector fed, so its per-episode state at
    the decision point matches collection. Caller owns the returned state and
    must close it.
    """
    state = GoSearchState.new(house_rules, infoset.deal_seed)
    try:
        _begin_episode(state, agent_wrapper, _PLAYER_ID)
        for action_idx in infoset.action_prefix:
            actor = state.acting_player()
            if not state.apply_index(action_idx):
                raise RuntimeError(
                    f"replay_infoset: engine rejected action {action_idx} while "
                    f"replaying deal {infoset.deal_seed}; the recorded prefix "
                    "and the engine have diverged."
                )
            _notify(
                agent_wrapper,
                "observe_transition",
                state.view(),
                index_to_action(action_idx),
                actor,
            )
    except Exception:
        state.close()
        raise
    return state


def collect_infosets(
    agent_wrapper,
    config,
    num_infosets: int,
    seed: int = 42,
    trajectory_opponent_factory: OpponentFactory = _make_random_opponent,
    sample_prob: float = 1.0,
    max_games: Optional[int] = None,
) -> List[SampledInfoset]:
    """Collect P0 decision points by play against a trajectory opponent.

    Plays games with ``agent_wrapper`` at seat 0 and a fresh
    ``trajectory_opponent_factory(1, config)`` opponent at seat 1, recording P0
    decision points until ``num_infosets`` are collected or ``max_games`` games
    have been played.

    This is the BUG-3 fix: collection is bounded by the requested COUNT, not by
    a games budget sized for a wrong decisions/game assumption. Requesting N
    infosets yields ~N (subject to the safety cap).

    Args:
        agent_wrapper: agent under test; ``choose_action(view, legal)`` plus the
            optional hooks named in the module docstring.
        config: config exposing ``cambia_rules`` (and ``agents`` for strong opps).
        num_infosets: target number of P0 infosets to collect.
        seed: RNG seed for reproducibility.
        trajectory_opponent_factory: builds the seat-1 opponent per game.
        sample_prob: per-eligible-P0-decision sampling probability. Defaults to
            1.0 (take every decision) so the target count is reached quickly.
        max_games: hard cap on games played (safety against unreachable targets).
            Defaults to a generous multiple of the games implied by the measured
            decisions/game, with an absolute ceiling.

    Returns:
        list of ``SampledInfoset``.
    """
    rng = np.random.default_rng(seed)
    _random_module.seed(seed)

    house_rules = config.cambia_rules
    max_turns = _resolve_max_turns(config)

    if max_games is None:
        implied = int(math.ceil(num_infosets / _EST_P0_DECISIONS_PER_GAME))
        # 4x margin over the implied games, floor 200, absolute ceiling so an
        # agent that never reaches a P0 decision cannot loop unbounded.
        max_games = min(max(200, implied * 4), 2_000_000)

    sampled: List[SampledInfoset] = []
    games_played = 0
    failed_observe_transitions = 0

    while len(sampled) < num_infosets and games_played < max_games:
        games_played += 1

        deal_seed = int(rng.integers(0, 2**31))
        opp_agent = trajectory_opponent_factory(_OPPONENT_ID, config)

        state = GoSearchState.new(house_rules, deal_seed)
        try:
            _begin_episode(state, agent_wrapper, _PLAYER_ID)
            prefix: List[int] = []

            turn = 0
            while not state.is_terminal() and turn < max_turns:
                turn += 1
                ap = state.acting_player()
                if ap == -1:
                    break
                legal_indices = state.legal_indices()
                if not legal_indices:
                    break
                legal_actions = actions_from_mask(legal_indices)

                if ap == _PLAYER_ID:
                    take = len(sampled) < num_infosets and rng.random() < sample_prob
                    chosen_action = agent_wrapper.choose_action(
                        state.view(), legal_actions
                    )
                    try:
                        pos = legal_actions.index(chosen_action)
                    except ValueError:
                        pos = 0
                    if take:
                        sampled.append(
                            SampledInfoset(
                                deal_seed=deal_seed,
                                action_prefix=tuple(prefix),
                                legal_indices=tuple(legal_indices),
                                agent_action_pos=pos,
                            )
                        )
                else:
                    chosen_action = opp_agent.choose_action(state.view(), legal_actions)
                    try:
                        pos = legal_actions.index(chosen_action)
                    except ValueError:
                        pos = 0

                chosen_idx = legal_indices[pos]
                if not state.apply_index(chosen_idx):
                    break
                prefix.append(chosen_idx)

                # Full-recall frame feed for agents that keep their own stream.
                # The GoAgentState token streams advance inside apply_index; this
                # hook is for wrapper-private Python state.
                if hasattr(agent_wrapper, "observe_transition"):
                    if not _notify(
                        agent_wrapper,
                        "observe_transition",
                        state.view(),
                        legal_actions[pos],
                        ap,
                    ):
                        # L5 (cambia-248): a dropped frame here desyncs the
                        # agent's prefix from the true trajectory for the REST
                        # of this game -- any further P0 samples taken this game
                        # would be measured against a stale prefix. Abort this
                        # game's measurement rather than keep sampling corrupted
                        # infosets.
                        failed_observe_transitions += 1
                        break

                if len(sampled) >= num_infosets:
                    break
        finally:
            state.close()

    if len(sampled) < num_infosets:
        logger.warning(
            "collect_infosets: collected %d of %d requested after %d games "
            "(safety cap hit).",
            len(sampled),
            num_infosets,
            games_played,
        )
    if failed_observe_transitions:
        logger.warning(
            "collect_infosets: %d game(s) aborted early due to "
            "observe_transition failures (see per-occurrence warnings above).",
            failed_observe_transitions,
        )
    return sampled


def _agent_policy_rollout(
    state: GoSearchState,
    agent_wrapper,
    rollout_opponent,
    max_turns: int,
) -> float:
    """Roll out from the current state under agent-policy play.

    Seat 0 plays the agent's own policy; seat 1 plays ``rollout_opponent``. No
    ``observe_transition`` hook is fired here: the caller rewinds this branch
    with ``state_restore``, which rewinds the ``GoAgentState`` (belief + tokens)
    but cannot rewind a wrapper's private Python stream.
    """
    turns = 0
    while not state.is_terminal() and turns < max_turns:
        turns += 1
        ap = state.acting_player()
        if ap == -1:
            break
        legal_indices = state.legal_indices()
        if not legal_indices:
            break
        legal_actions = actions_from_mask(legal_indices)
        try:
            if ap == _PLAYER_ID:
                act = agent_wrapper.choose_action(state.view(), legal_actions)
            else:
                act = rollout_opponent.choose_action(state.view(), legal_actions)
            pos = legal_actions.index(act)
        except Exception:  # JUSTIFIED: eval resilience
            break
        if not state.apply_index(legal_indices[pos]):
            break
    return terminal_utility(state)


def tier_b_lbr(
    agent_wrapper,
    config,
    num_infosets: int = 10000,
    br_rollouts_per_infoset: int = 100,
    seed: int = 42,
    trajectory_opponent_factory: OpponentFactory = DEFAULT_TRAJECTORY_OPPONENT,
    rollout_opponent_factory: OpponentFactory = DEFAULT_ROLLOUT_OPPONENT,
    max_games: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the Tier-B sampled LBR exploitability estimate.

    Algorithm:
      1. Collect P0 infosets along trajectories where the agent (seat 0) faces a
         strong fixed opponent (seat 1).
      2. At each sampled infoset, replay the state, then for each legal action:
           rewind to the decision point, apply the candidate action, and roll the
           continuation out under agent-policy play (seat 0 = agent, seat 1 =
           strong opponent). Average over ``br_rollouts_per_infoset``.
      3. BR value = max over actions of mean continuation utility.
         Agent value = mean continuation utility of the action the agent chose.
      4. Exploitability = mean(BR value - agent value) over infosets (>= 0).

    Returns a dict:
        exploitability: float (mean BR gap; >= 0 by construction)
        num_infosets_sampled: int
        std_err: float (standard error of the per-infoset gap mean)
        tier: "B"
        rollout_opponent: str (label of the rollout opponent class, for the row)
        seed: int (echoed, so a persisted row records what produced it)
    """
    max_turns = _resolve_max_turns(config)
    house_rules = config.cambia_rules

    sampled = collect_infosets(
        agent_wrapper,
        config,
        num_infosets=num_infosets,
        seed=seed,
        trajectory_opponent_factory=trajectory_opponent_factory,
        max_games=max_games,
    )

    opp_label = type(rollout_opponent_factory(_OPPONENT_ID, config)).__name__

    def _empty(reason: str) -> Dict[str, Any]:
        logger.warning("tier_b_lbr: %s", reason)
        return {
            "exploitability": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "tier": "B",
            "rollout_opponent": opp_label,
            "seed": seed,
        }

    if not sampled:
        return _empty("no infosets sampled.")

    gaps: List[float] = []
    for infoset in sampled:
        try:
            state = replay_infoset(house_rules, infoset, agent_wrapper)
        except Exception as exc:  # JUSTIFIED: eval resilience
            logger.warning("tier_b_lbr: infoset replay failed (%s); skipping.", exc)
            continue

        snap_h: Optional[int] = None
        try:
            snap_h = state.save()
            action_mean_utils: List[float] = []
            for action_idx in infoset.legal_indices:
                utils: List[float] = []
                for _ in range(br_rollouts_per_infoset):
                    state.restore(snap_h)
                    if not state.apply_index(action_idx):
                        utils.append(0.0)
                        continue
                    rollout_opp = rollout_opponent_factory(_OPPONENT_ID, config)
                    utils.append(
                        _agent_policy_rollout(
                            state, agent_wrapper, rollout_opp, max_turns
                        )
                    )
                action_mean_utils.append(float(np.mean(utils)) if utils else 0.0)
        finally:
            if snap_h is not None:
                GoSearchState.free_snapshot(snap_h)
            state.close()

        if not action_mean_utils:
            continue
        br_value = max(action_mean_utils)
        safe_pos = min(infoset.agent_action_pos, len(action_mean_utils) - 1)
        agent_value = action_mean_utils[safe_pos]
        gaps.append(br_value - agent_value)

    if not gaps:
        return _empty("no infoset produced a measurable gap.")

    gaps_arr = np.array(gaps)
    exploitability = float(np.mean(gaps_arr))
    std_err = (
        float(np.std(gaps_arr, ddof=1) / np.sqrt(len(gaps_arr)))
        if len(gaps_arr) > 1
        else 0.0
    )
    return {
        "exploitability": exploitability,
        "num_infosets_sampled": len(sampled),
        "std_err": std_err,
        "tier": "B",
        "rollout_opponent": opp_label,
        "seed": seed,
    }
