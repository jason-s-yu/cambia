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

Sampled infosets are recorded as ``(deal spec, applied action-index prefix)``
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
    save/restore rewind keeps consistent. Also fired at the Tier-B continuation
    seat, where it is the whole of that opponent's initialisation: the handle it
    receives is the one the engine advanced through the sampled prefix, so a
    belief-carrying opponent starts its rollout knowing the public history
    rather than knowing nothing (cambia-1793).
  - ``initialize_state(view)``: per-episode reset, at the start of a game and at
    the start of each infoset replay.
  - ``belief_handle()``: the ``GoAgentState`` handle a wrapper built for itself
    in ``initialize_state``. The search state ADOPTS it (see
    ``GoSearchState.adopt_agent_belief``) so the engine advances that belief
    with every applied action and the rewind restores it, which is what the
    head-to-head and mean_imp loops do through ``apply_games_batch``. Before
    cambia-1479 the estimators applied actions with their own handles only, so a
    wrapper's belief was built at the deal and never moved: a whole measurement
    ran on the opening knowledge. ``frozen_beliefs=True`` reproduces that
    protocol on demand, and every result names the protocol it ran under.
  - ``observe_transition(view, action, actor)``: post-action frame, fed for
    every applied action along a TRAJECTORY (collection and replay) so a wrapper
    that keeps its own Python-side stream stays in step. Deliberately NOT fed
    during branch rollouts: a rollout is rewound by ``state_restore``, which
    rewinds the ``GoAgentState`` but cannot rewind wrapper-private Python state,
    so feeding it there would leave such a wrapper desynced for the rest of the
    infoset.

A policy that raises while choosing is a defect, not a condition to route
around: ``choose_action_pos`` re-raises it as a ``PolicyError`` carrying the
seat and the phase, so a broken agent fails the run instead of scoring
truncated playouts and reporting a low exploitability (cambia-1479). An engine
read a utility depends on gets the same treatment as an ``EngineReadError``,
since the 0.0 those reads used to fall back to is the value of a draw. Both are
``MeasurementError``. The failures that ARE absorbed (a lost optional-hook
frame, a policy returning an action outside the legal set, an infoset whose
replay diverged) are counted on a ``ToleratedFailures`` and the count rides on
the result as ``policy_errors``.

Both tiers are pure eval-time measurement: no network or harness state is
mutated.
"""

import logging
import math
import random as _random_module
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from src.agents.action_codec import actions_from_indices, index_to_action
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


#: Rule fields the Go engine's FFI rules struct cannot express. Empty since
#: cambia-1478 crossed deck_ranks over as HouseRules.DeckRanks: the bridge now
#: passes all 13 of CambiaRulesConfig's rule fields to
#: cambia_game_new_with_rules, so a reduced-deck config deals the same game on
#: both engines and needs no guard. The gate stays because it is the thing that
#: caught the last silently-dropped field: a rule the Go deal cannot honor gets
#: named here and is refused loudly instead of quietly measuring another game.
_FFI_UNSUPPORTED_RULE_FIELDS: Tuple[str, ...] = ()


def _reject_unsupported_rules(house_rules: Any) -> None:
    """Refuse house rules the Go deal cannot honor.

    Without this, a config the FFI cannot carry measures a completely different
    game on the Go path and reports the number as if it were the configured one.
    An explicit deck (``GoSearchState.from_deck``, or the ``deal_decks`` pool the
    estimators accept) is the fallback for anything a rule cannot express, since
    a deck order fully determines the deal and needs no rule support.
    """
    for field in _FFI_UNSUPPORTED_RULE_FIELDS:
        value = getattr(house_rules, field, None)
        if value:
            raise ValueError(
                f"house_rules.{field}={value!r} cannot be expressed over the "
                "Go engine's FFI rules struct, so dealing from these rules "
                "would silently measure a different game. Pass an explicit deck "
                "instead: GoSearchState.from_deck(...), or the deal_decks= pool "
                "accepted by collect_infosets / sampled_lbr / tier_b_lbr / "
                "ismcts_br."
            )


class DealSpec(NamedTuple):
    """How to produce one deal: either a Go seed or an explicit deck order.

    A spec is self-contained, so a recorded infoset can be replayed without the
    caller holding on to whatever pool it came from.
    """

    seed: int = 0
    deck: Optional[Tuple[int, ...]] = None
    starting_player: int = 0

    def new_state(self, house_rules: Any) -> "GoSearchState":
        if self.deck is not None:
            return GoSearchState.from_deck(house_rules, self.deck, self.starting_player)
        return GoSearchState.new(house_rules, self.seed)


def normalize_deal_decks(deal_decks) -> List[DealSpec]:
    """Turn a deck pool into ``DealSpec``s.

    Accepts either bare deck orders or ``(deck, starting_player)`` pairs -- the
    pair form is what a reference implementation hands over, since which seat
    moves first is part of the deal it solved.
    """
    specs: List[DealSpec] = []
    for entry in deal_decks:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and not isinstance(entry[1], (list, tuple))
        ):
            deck, starting_player = entry
        else:
            deck, starting_player = entry, 0
        specs.append(
            DealSpec(
                deck=tuple(int(c) for c in deck), starting_player=int(starting_player)
            )
        )
    return specs


class SampledInfoset(NamedTuple):
    """A recorded P0 decision point, replayable on demand.

    Holds the deal spec and the action-index prefix that reaches the decision
    instead of a copy of the state: a Go deal is a pure function of its seed (or
    of its explicit deck), so replaying the prefix reconstructs the exact game,
    both seats' beliefs and both token streams. Keeping the state open instead
    would pin three FFI handles per sampled infoset, and the pool is finite.
    """

    deal: DealSpec
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

    __slots__ = ("engine", "a0", "a1", "_closed", "_belief_handles")

    def __init__(self, engine: GoEngine, a0: GoAgentState, a1: GoAgentState) -> None:
        self.engine = engine
        self.a0 = a0
        self.a1 = a1
        self._closed = False
        # The belief handles the game advances and the rewind restores.
        # They start as this state's own a0/a1 and are replaced per seat
        # by adopt_agent_belief when the policy at that seat owns its own
        # belief (cambia-1479 defect 2).
        self._belief_handles = [int(a0.handle), int(a1.handle)]

    @classmethod
    def new(cls, house_rules: Any, seed: int) -> "GoSearchState":
        """Fresh deal from ``seed`` under ``house_rules``.

        Refuses rules the Go deal cannot honor, rather than quietly dealing a
        different game (see ``_reject_unsupported_rules``).
        """
        _reject_unsupported_rules(house_rules)
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

    def belief_handle(self, seat: int) -> int:
        """The handle the game advances for ``seat``: adopted, or own."""
        return self._belief_handles[0 if seat == 0 else 1]

    def adopt_agent_belief(self, seat: int, policy: Any) -> bool:
        """Advance ``policy``'s own belief at ``seat`` instead of this state's.

        The evaluation wrappers own their belief: ``initialize_state``
        builds a GoAgentState for the seat, and the head-to-head and
        mean_imp loops hand THAT handle to apply_games_batch, which is
        what advances it. The estimators here applied every action with
        their own a0/a1 instead, so a wrapper's belief was built at the
        deal and never moved again: an entire measurement was taken on
        the opening knowledge (cambia-1479 defect 2). Adopting the
        wrapper's handle into the apply/save/restore triple advances it
        the way the evaluation loop does, and keeps it under the rewind,
        since state_restore rewinds whatever handles it is given.

        The adopted handle belongs to the policy and is never freed here.
        Returns False when the policy carries no belief of its own (the
        heuristic baselines, the uniform rollout policies), which leaves
        the seat on this state's handle.
        """
        handle_fn = getattr(policy, "belief_handle", None)
        if handle_fn is None:
            return False
        try:
            handle = int(handle_fn())
        except Exception as exc:  # JUSTIFIED: an optional wrapper hook
            logger.warning(
                "lbr: %s.belief_handle() failed (%s: %s); seat %d keeps the "
                "search state's own belief.",
                type(policy).__name__,
                type(exc).__name__,
                exc,
                seat,
            )
            return False
        if handle < 0:
            return False
        self._belief_handles[0 if seat == 0 else 1] = handle
        return True

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
        return actions_from_indices(self.legal_indices())

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
                [self._belief_handles[0]],
                [self._belief_handles[1]],
                [int(action_idx)],
            )
            return True
        except RuntimeError as exc:
            if "overflow" in str(exc):
                raise
            return False

    def save(self) -> int:
        return state_save(
            self.engine.handle, self._belief_handles[0], self._belief_handles[1]
        )

    def restore(self, snap_h: int) -> None:
        state_restore(
            self.engine.handle,
            snap_h,
            self._belief_handles[0],
            self._belief_handles[1],
        )

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
    return UniformRandomPolicy(
        player_id, _random_module.Random(_random_module.getrandbits(63))
    )


# Set once the strong-opponent fallback has been reported (see below).
_STRONG_OPPONENT_WARNED = False


def _make_strong_opponent(player_id: int, config: Any):
    """The default strong fixed opponent (ImperfectGreedyAgent), if available.

    The fallback to uniform-random stays for an agent that cannot read a
    ``GoEngine``: a Tier-B run on it is a Tier-A continuation wearing a Tier-B
    label, which the returned ``rollout_opponent`` field names so a row is never
    silently mislabelled. That fallback is no longer the normal case --
    ImperfectGreedyAgent claims ``accepts_game_view`` since cambia-1479, having
    only been ported (cambia-1426) and never marked, which put both Tier-B legs
    of 2026-09-01 on UniformRandomPolicy.
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

    The opt-in marker a ported baseline/wrapper sets (see
    ``src.agents.baseline_agents.BaseAgent.accepts_game_view``); absent it, an
    agent is assumed to still want the Python reference engine.
    """
    return bool(getattr(agent, "accepts_game_view", False))


# Default Tier-B opponents: strong both for trajectory generation and for the
# adversary seat during agent-policy continuation rollouts.
DEFAULT_TRAJECTORY_OPPONENT: OpponentFactory = _make_strong_opponent
DEFAULT_ROLLOUT_OPPONENT: OpponentFactory = _make_strong_opponent


class MeasurementError(RuntimeError):
    """Base for the failures an estimator refuses to absorb.

    Both members answer the same question the wrong way round: an estimator
    that routes around a failure still returns a number, and a number built on
    a failure it hid is worse than no number (cambia-1479). Catch this base to
    catch every such failure; the members say whether the policy or an engine
    read was the thing that broke.
    """


class PolicyError(MeasurementError):
    """A policy raised while being asked for an action at a measured decision.

    Every estimator here used to catch this and break out of the playout, so a
    broken agent scored a truncated game instead of failing the run and the
    exploitability estimate came back quietly low (cambia-1479 defect 1). The
    error now propagates, carrying the context a bare traceback lacks: which
    policy, which seat, which phase of the measurement, and how many actions
    were on offer.
    """


class EngineReadError(MeasurementError):
    """An engine read an estimator needs for a utility failed.

    ``terminal_utility`` and ``hand_score_utility`` used to answer 0.0 here,
    which is the value of a draw: an unreadable playout scored as a tie and
    pulled the estimate toward zero with nothing on the row to say so. There is
    no correct utility to substitute for a read that failed, so the run fails.
    """


class ToleratedFailures:
    """Counts the failures an estimator absorbs instead of raising.

    A silently absorbed failure is what lets a measurement come back depressed
    rather than broken, so each one is counted here and the first of its kind is
    logged with its exception. Per-occurrence logging is not an option: a
    production-sized run absorbs failures inside millions of rollouts. The total
    rides on the estimator's result dict so a non-zero count reaches the
    persisted row.
    """

    __slots__ = ("counts", "_logged")

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self._logged: set = set()

    def record(self, kind: str, detail: str, exc: Optional[BaseException] = None) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind in self._logged:
            return
        self._logged.add(kind)
        logger.warning(
            "lbr: tolerated %s failure (%s)%s. Further occurrences of this kind "
            "are counted, not logged; the total rides on the result as "
            "policy_errors.",
            kind,
            detail,
            "" if exc is None else f": {type(exc).__name__}: {exc}",
        )

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> Dict[str, int]:
        return dict(self.counts)


def choose_action_pos(
    policy: Any,
    view: GameView,
    legal_actions: Sequence[GameAction],
    *,
    seat: int,
    phase: str,
    tolerated: Optional[ToleratedFailures] = None,
) -> int:
    """Ask ``policy`` for an action and return its position in ``legal_actions``.

    A policy that RAISES is never absorbed: the exception is re-raised as a
    ``PolicyError`` naming the policy, the seat, the phase and the legal-set
    size, so the run fails where the defect is instead of reporting a number.

    A policy that returns an action OUTSIDE the legal set is a different, and
    recoverable, failure -- the heuristic baselines build actions without
    always checking legality. It falls back to the first legal action, which is
    the behaviour the collector already had, and is recorded in ``tolerated``
    when one is passed so the count reaches the row rather than vanishing. With
    no counter to record it in, it raises too.
    """
    try:
        action = policy.choose_action(view, legal_actions)
    except Exception as exc:
        raise PolicyError(
            f"{type(policy).__name__} P{seat} raised during {phase} with "
            f"{len(legal_actions)} legal actions: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        return legal_actions.index(action)
    except ValueError:
        detail = (
            f"{type(policy).__name__} P{seat} returned {action!r} during {phase}, "
            f"which is not in the {len(legal_actions)}-action legal set"
        )
        if tolerated is None:
            raise PolicyError(detail) from None
        tolerated.record("illegal_action", detail)
        return 0


def _resolve_max_turns(config: Any) -> int:
    house_rules = config.cambia_rules
    max_turns = getattr(house_rules, "max_game_turns", 0)
    if max_turns <= 0:
        max_turns = 500
    return max_turns


def _notify(
    policy: Any,
    method: str,
    *args,
    tolerated: Optional[ToleratedFailures] = None,
) -> bool:
    """Call an optional policy hook. Returns False if it raised.

    A hook is optional and a failure is recoverable, so this one absorbs; the
    absorb is counted on ``tolerated`` (and logged once per kind there) rather
    than logged per occurrence, so a run that lost frames says so on its result.
    """
    fn = getattr(policy, method, None)
    if fn is None:
        return True
    try:
        fn(*args)
        return True
    except Exception as exc:  # JUSTIFIED: eval resilience for optional hooks
        if tolerated is None:
            logger.warning(
                "lbr: policy hook %s failed (%s: %s)", method, type(exc).__name__, exc
            )
        else:
            tolerated.record(
                f"hook_{method}", f"{type(policy).__name__} hook {method}", exc
            )
        return False


#: How a measurement treated the belief of a policy that owns one. "advancing"
#: is the correct protocol: the engine moves the policy's belief with every
#: applied action. "frozen" is the pre-cambia-1479 protocol, kept switchable so
#: a historical number can be reproduced against the run that produced it.
BELIEF_PROTOCOL_ADVANCING = "advancing"
BELIEF_PROTOCOL_FROZEN = "frozen"


def belief_protocol_label(frozen_beliefs: bool) -> str:
    """The label a result records for the protocol it ran under."""
    return BELIEF_PROTOCOL_FROZEN if frozen_beliefs else BELIEF_PROTOCOL_ADVANCING


def _begin_episode(
    state: GoSearchState,
    agent_wrapper: Any,
    seat: int,
    frozen_beliefs: bool = False,
    tolerated: Optional[ToleratedFailures] = None,
) -> None:
    """Hand the agent its per-episode Go handles and reset its episode state.

    The state then adopts whatever belief the wrapper built in
    ``initialize_state``, so the engine advances it with every applied action
    (see ``GoSearchState.adopt_agent_belief``). ``frozen_beliefs`` skips the
    adoption and reproduces the pre-cambia-1479 protocol, where such a belief
    stayed at its initial state for the whole measurement.
    """
    _notify(
        agent_wrapper,
        "bind_go_state",
        state.view(),
        state.agent_state(seat),
        tolerated=tolerated,
    )
    _notify(agent_wrapper, "initialize_state", state.view(), tolerated=tolerated)
    if not frozen_beliefs:
        state.adopt_agent_belief(seat, agent_wrapper)


def _begin_continuation(
    state: GoSearchState,
    policy: Any,
    seat: int = _OPPONENT_ID,
    tolerated: Optional[ToleratedFailures] = None,
) -> None:
    """Seed a continuation opponent built mid-game with the history it missed.

    A Tier-B continuation opponent is constructed at the infoset, not at the
    deal, so it has observed nothing: before cambia-1793 it took its first
    decision with a belief built from thin air, which is why a belief-carrying
    opponent could not be measured against at all (cambia-1479 F4).

    It does not have to replay the prefix to catch up. The engine advanced
    ``seat``'s own ``GoAgentState`` through every action of that prefix during
    ``replay_infoset``, so the belief the opponent needs already exists on the
    search state; binding hands it over. That keeps this O(1) per rollout, which
    matters because Tier B builds one opponent per rollout (tens of thousands
    per leg) since the shared-opponent reuse was reverted.

    ``initialize_state`` is deliberately NOT fired here. It builds a fresh
    belief from the view it is handed, and the view at an infoset is mid-game:
    a wrapper would read the current position as an opening deal and take the
    seat's current cards for its initial peek. Binding is the only seeding that
    is true to the history.

    Neither is the seat's belief ADOPTED from the policy the way seat 0's is.
    The handle bound here is the search state's own, which ``apply_index``
    advances and ``restore`` rewinds, so the opponent's belief already moves
    with the rollout and unwinds with it. Swapping in a policy-owned handle
    per rollout would repoint the seat while the infoset's snapshot is still
    live, and that snapshot was taken against the handle it is now not holding.
    ``frozen_beliefs`` therefore governs the measured agent's belief only; it
    has never had anything to say about this seat.
    """
    _notify(
        policy,
        "bind_go_state",
        state.view(),
        state.agent_state(seat),
        tolerated=tolerated,
    )


def hand_score_utility(view: GameView, seat: int, opponent_seat: int) -> float:
    """Utility estimate for a game cut short by the decision cap.

    Lower hand score wins, matching the Tier-A/Tier-B/ISMCTS timeout convention
    so the estimators stay comparable. A hand that cannot be read raises
    ``EngineReadError`` rather than scoring the playout as a tie.
    """
    try:
        mine = sum(c.value for c in view.get_player_hand(seat))
        theirs = sum(c.value for c in view.get_player_hand(opponent_seat))
    except Exception as exc:
        raise EngineReadError(
            f"reading hands for the timeout hand-score utility failed "
            f"(seat {seat} vs seat {opponent_seat}): {type(exc).__name__}: {exc}"
        ) from exc
    if mine < theirs:
        return 1.0
    if mine > theirs:
        return -1.0
    return 0.0


def terminal_utility(
    state: GoSearchState, seat: int = _PLAYER_ID, opponent_seat: int = _OPPONENT_ID
) -> float:
    """Terminal utility for ``seat``, or a hand-score estimate on timeout.

    An unreadable terminal state raises ``EngineReadError``: the 0.0 this used
    to return is the value of a draw, so an engine read that failed scored as
    one and dragged the estimate toward zero without a trace.
    """
    if state.is_terminal():
        try:
            return state.utility(seat)
        except Exception as exc:
            raise EngineReadError(
                f"reading the terminal utility for seat {seat} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return hand_score_utility(state.view(), seat, opponent_seat)


def replay_infoset(
    house_rules: Any,
    infoset: SampledInfoset,
    agent_wrapper: Any,
    frozen_beliefs: bool = False,
    tolerated: Optional[ToleratedFailures] = None,
) -> GoSearchState:
    """Rebuild the state at a sampled decision point.

    Replays the recorded action-index prefix onto a fresh deal, feeding the
    agent the same episode hooks the collector fed, so its per-episode state at
    the decision point matches collection. Caller owns the returned state and
    must close it.
    """
    state = infoset.deal.new_state(house_rules)
    try:
        _begin_episode(state, agent_wrapper, _PLAYER_ID, frozen_beliefs, tolerated)
        for action_idx in infoset.action_prefix:
            actor = state.acting_player()
            if not state.apply_index(action_idx):
                raise RuntimeError(
                    f"replay_infoset: engine rejected action {action_idx} while "
                    f"replaying deal {infoset.deal}; the recorded prefix "
                    "and the engine have diverged."
                )
            _notify(
                agent_wrapper,
                "observe_transition",
                state.view(),
                index_to_action(action_idx),
                actor,
                tolerated=tolerated,
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
    deal_decks: Optional[Sequence[Any]] = None,
    frozen_beliefs: bool = False,
    tolerated: Optional[ToleratedFailures] = None,
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
        deal_decks: optional pool of explicit deck orders (bare decks, or
            ``(deck, starting_player)`` pairs) to draw deals from instead of
            seeding the Go dealer. Required for any config whose deck the FFI
            rules struct cannot express -- see ``_reject_unsupported_rules``.
        frozen_beliefs: reproduce the pre-cambia-1479 protocol, where a policy
            that owns a belief kept the one it built at the deal for the whole
            measurement. Default False: beliefs advance with every action, as
            they do in the head-to-head and mean_imp loops.
        tolerated: counter for the failures collection absorbs (a lost hook
            frame, a policy returning an action outside the legal set). A
            raising policy is never absorbed -- it leaves as a ``PolicyError``.

    Returns:
        list of ``SampledInfoset``.
    """
    deal_specs = normalize_deal_decks(deal_decks) if deal_decks else []
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

        if deal_specs:
            deal = deal_specs[int(rng.integers(len(deal_specs)))]
        else:
            deal = DealSpec(seed=int(rng.integers(0, 2**31)))
        opp_agent = trajectory_opponent_factory(_OPPONENT_ID, config)

        state = deal.new_state(house_rules)
        try:
            _begin_episode(state, agent_wrapper, _PLAYER_ID, frozen_beliefs, tolerated)
            # The opponent seat has the same belief lifecycle: a belief-carrying
            # opponent would otherwise play the whole trajectory on the deal's
            # opening knowledge. Both hooks are optional, so a heuristic
            # baseline or a uniform policy passes through untouched.
            _begin_episode(state, opp_agent, _OPPONENT_ID, frozen_beliefs, tolerated)
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
                legal_actions = actions_from_indices(legal_indices)

                if ap == _PLAYER_ID:
                    take = len(sampled) < num_infosets and rng.random() < sample_prob
                    pos = choose_action_pos(
                        agent_wrapper,
                        state.view(),
                        legal_actions,
                        seat=ap,
                        phase="trajectory collection",
                        tolerated=tolerated,
                    )
                    if take:
                        sampled.append(
                            SampledInfoset(
                                deal=deal,
                                action_prefix=tuple(prefix),
                                legal_indices=tuple(legal_indices),
                                agent_action_pos=pos,
                            )
                        )
                else:
                    pos = choose_action_pos(
                        opp_agent,
                        state.view(),
                        legal_actions,
                        seat=ap,
                        phase="trajectory collection (opponent)",
                        tolerated=tolerated,
                    )

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
                        tolerated=tolerated,
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
            "observe_transition failures; the failures themselves are counted "
            "on the estimator's policy_errors, logged once per kind.",
            failed_observe_transitions,
        )
    return sampled


def _agent_policy_rollout(
    state: GoSearchState,
    agent_wrapper,
    rollout_opponent,
    max_turns: int,
    tolerated: Optional[ToleratedFailures] = None,
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
        legal_actions = actions_from_indices(legal_indices)
        if ap == _PLAYER_ID:
            pos = choose_action_pos(
                agent_wrapper,
                state.view(),
                legal_actions,
                seat=ap,
                phase="Tier-B continuation rollout",
                tolerated=tolerated,
            )
        else:
            pos = choose_action_pos(
                rollout_opponent,
                state.view(),
                legal_actions,
                seat=ap,
                phase="Tier-B continuation rollout (opponent)",
                tolerated=tolerated,
            )
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
    deal_decks: Optional[Sequence[Any]] = None,
    frozen_beliefs: bool = False,
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

    The two seat-1 roles are separate knobs and the row records both
    (cambia-1793). ``trajectory_opponent_factory`` decides WHICH positions are
    measured, since it plays seat 1 while the infosets are collected;
    ``rollout_opponent_factory`` decides HOW HARD the continuation is, since it
    plays seat 1 after the candidate action. Both default to the strong
    opponent, so moving Tier B onto the real strong opponent (cambia-1479)
    moved both at once and the number could not say which change produced it.
    The tier's rationale is about the continuation alone.

    Each continuation opponent is seeded with the public history at the infoset
    before its first decision (see ``_begin_continuation``), so an opponent that
    carries a belief can be measured against here at all.

    ``frozen_beliefs`` reproduces the pre-cambia-1479 protocol, where a policy
    owning a belief kept the one it built at the deal for the whole measurement.
    Default False: beliefs advance with every applied action. The number a run
    produces is not comparable across the two, so the result names which ran.

    Returns a dict:
        exploitability: float (mean BR gap; >= 0 by construction)
        num_infosets_sampled: int
        std_err: float (standard error of the per-infoset gap mean)
        tier: "B"
        trajectory_opponent: str (class that played seat 1 during collection)
        continuation_opponent: str (class that plays seat 1 in the rollouts)
        rollout_opponent: str (the continuation opponent under the name this
            row carried before the two were named apart; every consumer
            written before cambia-1793 reads this one)
        seed: int (echoed, so a persisted row records what produced it)
        belief_protocol: "advancing" or "frozen"
        policy_errors: int (failures the run absorbed; a raising policy is not
            absorbed, it raises PolicyError)
        policy_error_detail: dict of absorbed-failure kind -> count
    """
    max_turns = _resolve_max_turns(config)
    house_rules = config.cambia_rules
    tolerated = ToleratedFailures()
    protocol = belief_protocol_label(frozen_beliefs)

    sampled = collect_infosets(
        agent_wrapper,
        config,
        num_infosets=num_infosets,
        seed=seed,
        trajectory_opponent_factory=trajectory_opponent_factory,
        max_games=max_games,
        deal_decks=deal_decks,
        frozen_beliefs=frozen_beliefs,
        tolerated=tolerated,
    )

    # A rollout opponent is built per rollout, which is the only shape that
    # needs no reset contract at all. Reuse was measured and bought nothing:
    # one ImperfectGreedyAgent costs 0.55 us to construct, so the tens of
    # thousands a leg builds are ~0.02s of its ~700s, and the back-to-back pair
    # of a counterbalanced A/B on 2026-09-02 put the reusing arm at 210.8s of
    # CPU against 196.5s for this one, on identical estimates (cambia-1479).
    opp_label = type(rollout_opponent_factory(_OPPONENT_ID, config)).__name__
    traj_label = type(trajectory_opponent_factory(_OPPONENT_ID, config)).__name__

    def _empty(reason: str) -> Dict[str, Any]:
        logger.warning("tier_b_lbr: %s", reason)
        return {
            "exploitability": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "tier": "B",
            "trajectory_opponent": traj_label,
            "continuation_opponent": opp_label,
            "rollout_opponent": opp_label,
            "seed": seed,
            "belief_protocol": protocol,
            "policy_errors": tolerated.total,
            "policy_error_detail": tolerated.as_dict(),
        }

    if not sampled:
        return _empty("no infosets sampled.")

    gaps: List[float] = []
    for infoset in sampled:
        try:
            state = replay_infoset(
                house_rules, infoset, agent_wrapper, frozen_beliefs, tolerated
            )
        except Exception as exc:  # JUSTIFIED: eval resilience
            # A skipped infoset shrinks the sample the estimate is built from,
            # so it is counted onto the row rather than only logged.
            tolerated.record("infoset_replay", "tier_b_lbr infoset replay", exc)
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
                    _begin_continuation(
                        state, rollout_opp, _OPPONENT_ID, tolerated=tolerated
                    )
                    utils.append(
                        _agent_policy_rollout(
                            state, agent_wrapper, rollout_opp, max_turns, tolerated
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
        "trajectory_opponent": traj_label,
        "continuation_opponent": opp_label,
        "rollout_opponent": opp_label,
        "seed": seed,
        "belief_protocol": protocol,
        "policy_errors": tolerated.total,
        "policy_error_detail": tolerated.as_dict(),
    }
