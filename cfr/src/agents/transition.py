"""src/agents/transition.py

The one per-transition step every game loop runs.

A transition is more than an engine step. A belief-carrying seat's
``GoAgentState`` -- and, for the token wrappers, the append-only event stream
hanging off it -- advances with the action, inside the same FFI crossing that
applies it, which is what makes an evaluated agent's belief follow the training
update rule by construction rather than by a Python re-derivation of it
(cambia-1426). A wrapper that keeps private Python state instead gets a
post-action frame through ``observe_transition``, the same hook the Tier-B
estimators feed along a trajectory (the contract is written out in
``src/cfr/lbr.py``).

Before cambia-711 each game loop carried its own version of that step:
``evaluate_agents._GoEvalGame.apply`` for the evaluation and head-to-head
loops, ``play._apply_action`` for the interactive one. The interactive one
advanced the ENGINE only, so an AI seat there played a whole game on the belief
and token prefix it was dealt at the deal, with nothing raised: a PRT-CFR seat
would have chosen every action off the opening prefix and scored as if that
were the game. This module is the single version both now call.

An agent carries an engine-side belief when it exposes ``belief_handle``. That
is the accessor the wrappers already publish so the estimators can adopt their
belief, so testing for it keeps one contract instead of two and keeps this
module off ``evaluate_agents``'s import graph.
"""

from typing import Any, Dict, List, Optional

from .action_codec import nplayer_index_for, relative_opponent_index
from .game_view import tracked_opponent_seat
from ..constants import GameAction
from ..ffi.bridge import apply_games_batch


def carries_belief(agent: Any) -> bool:
    """Whether the engine advances this agent's belief when an action is applied."""
    return agent is not None and callable(getattr(agent, "belief_handle", None))


class TransitionBroadcaster:
    """One game's transition step: apply an action, advance everything that tracks it.

    Built once per game, AFTER the engine and BEFORE the first action: the
    constructor fires each agent's per-game reset, and a belief attached
    anywhere but the initial state misses that seat's initial-peek knowledge
    (and, for the token wrappers, the private peek prefix their stream opens
    with).

    Action space by seat count, unchanged from the evaluation loop this was
    lifted out of:

      - TWO seats run the 2-player space (146 actions) and apply through
        ``apply_games_batch``, the only FFI path that advances the game AND both
        agents' belief and token streams in a single crossing. It is the path
        the production sampler uses.
      - ABOVE two seats the game runs in the N-player space (620 actions) and
        belief is advanced per seat with ``update_nplayer``; the token stream is
        unavailable there (no N-player token export), which is why the token
        wrappers refuse to be seated at a larger table.

    ``agents`` is seat-ordered and may hold ``None`` for a seat no agent drives
    (a human seat in interactive play); such a seat contributes no belief handle
    and no reset.
    """

    __slots__ = (
        "engine",
        "agents",
        "num_players",
        "_nplayer",
        "_belief_agents",
        "_batch_handles",
        "_observers",
        "_released",
    )

    def __init__(self, engine, agents: List[Any], num_players: int):
        self.engine = engine
        self.num_players = max(2, int(num_players))
        self._nplayer = self.num_players > 2
        self.agents = list(agents)
        self._released = False

        self._reset_agents()

        self._belief_agents = [a for a in self.agents if carries_belief(a)]
        self._observers = [
            a
            for a in self.agents
            if a is not None and callable(getattr(a, "observe_transition", None))
        ]
        # Handle vector for apply_games_batch: seat 0 in a0, seat 1 in a1, -1
        # for a seat whose agent carries no belief.
        if self._nplayer:
            self._batch_handles = None
        else:
            self._batch_handles = [
                (
                    self.agents[seat].belief_handle()
                    if carries_belief(self.agents[seat])
                    else -1
                )
                for seat in range(2)
            ]

    # --- Per-game lifecycle ---

    def _reset_agents(self) -> None:
        """Fire each agent's per-game reset, before any action is applied.

        ``initialize_state``, not ``attach_belief``: it is the hook the wrappers
        override, and several do real work in it (PRT-CFR samples the episode's
        snapshot and rebuilds its GRU cursor there; the PBS wrappers reset their
        ranges). Calling ``attach_belief`` directly would skip that and play
        every game with the first game's episode state.

        Baselines carry no such hook and detect a new game by the id() of what
        they are handed. Handles come from a pool and an address can be reused,
        so their sentinel is invalidated explicitly rather than trusted to
        differ (a stale hit keeps the previous game's memory and collapses games
        to an immediate Cambia call).
        """
        for agent in self.agents:
            if agent is None:
                continue
            if callable(getattr(agent, "initialize_state", None)):
                agent.initialize_state(self.engine)
            elif hasattr(agent, "_last_game_id"):
                agent._last_game_id = None

    @property
    def belief_agents(self) -> List[Any]:
        """The seats whose belief this game attached, in seat order."""
        return self._belief_agents

    def release(self) -> None:
        """Free every belief this game attached. Idempotent.

        Handles come from a finite pool and a game builds one per belief-carrying
        seat, so a loop that plays many games has to give them back.
        """
        if self._released:
            return
        self._released = True
        for agent in self._belief_agents:
            agent.release_belief()

    # --- The transition ---

    def apply(
        self,
        action: GameAction,
        index_map: Dict[GameAction, int],
        acting_player: Optional[int] = None,
    ) -> None:
        """Apply one action: the engine first, then every seat that tracks it.

        ``index_map`` is the action -> engine-index map the caller's legal-set
        decode produced. An action an agent built without checking legality is
        not in it (the baselines have such fallbacks); it is re-encoded here and
        left to the engine to reject, so an illegal choice still surfaces as an
        engine error rather than as a quietly dropped move.

        ``acting_player`` is the seat that chose, read off the engine when the
        caller does not supply it. It is only needed to re-encode an
        out-of-index action at N seats and to label the ``observe_transition``
        frame, so the two-seat hot path with no such wrapper seated pays no
        extra crossing for it.
        """
        idx = index_map.get(action)

        actor = -1
        if self._observers or (self._nplayer and idx is None):
            actor = (
                int(acting_player)
                if acting_player is not None
                else int(self.engine.acting_player())
            )

        if self._nplayer:
            if idx is None:
                idx = self._nplayer_index(action, actor)
            self.engine.apply_nplayer_action(int(idx))
            for agent in self._belief_agents:
                # ``agent_state`` rather than the handle: update_nplayer is a
                # method on the GoAgentState. PPOAgentWrapper keeps its belief
                # under a private name and so raises here above two seats,
                # exactly as it did in the evaluation loop this was lifted out
                # of; unifying that name is a wrapper change this did not take.
                agent.agent_state.update_nplayer(self.engine)
        else:
            if idx is None:
                idx = self._two_player_index(action)
            if self._belief_agents:
                apply_games_batch(
                    [self.engine.handle],
                    [self._batch_handles[0]],
                    [self._batch_handles[1]],
                    [int(idx)],
                )
            else:
                self.engine.apply_action(int(idx))

        # Post-action frame for a wrapper that keeps its own Python-side stream.
        # The GoAgentState streams moved above; this is the rest of the
        # trajectory contract src/cfr/lbr.py feeds, and the game loops fed none
        # of it before cambia-711. A failure here is not absorbed: a dropped
        # frame desyncs that wrapper's prefix for the rest of the game, and the
        # calling loops already count a raising turn as an error rather than
        # scoring it.
        for agent in self._observers:
            agent.observe_transition(self.engine, action, actor)

    # --- Index recovery for an action the decode did not hand out ---

    @staticmethod
    def _two_player_index(action: GameAction) -> int:
        # Imported here, not at module scope: src.encoding pulls the belief and
        # bucket modules in, and this is the rare fallback path.
        from ..encoding import action_to_index

        return action_to_index(action)

    def _nplayer_index(self, action: GameAction, actor: int) -> int:
        pending = self.engine.get_pending()
        if pending.target_seat is not None and pending.target_seat != actor:
            target = int(pending.target_seat)
        else:
            target = tracked_opponent_seat(actor, self.num_players)
        rel = relative_opponent_index(actor, target, self.num_players)
        return nplayer_index_for(action, rel)
