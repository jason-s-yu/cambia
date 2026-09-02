"""
src/agents/go_baselines.py

The mean_imp baselines, decided inside the Go engine (cambia-1487).

Each class here subclasses its pure-Python counterpart in baseline_agents.py
and overrides nothing but ``choose_action``: handed a GoEngine at two seats it
asks the engine for the decision through one FFI crossing, and handed anything
else (a PythonGameView, a table above two seats) it falls through to the
inherited Python body. The Python classes therefore remain the reference
implementation and the oracle the equality test scores against.

Why this exists: the mean_imp battery stopped tracking engine throughput after
the evaluation loop moved onto the Go engine, because each baseline decision
paid a decoded 146-action list plus a Python policy body, and that cost now
exceeds the engine's own. Deciding engine-side removes both.

Byte-for-byte identical decisions are the hard requirement, mean_imp being a
historical metric. Two things carry it:

  - The heuristics use no randomness at all, so their Go ports are decided by
    state alone and are checked action-for-action against the Python bodies in
    tests/test_go_baseline_parity.py.
  - The random policies keep their draw in CPython. The engine returns the
    candidate sequence its filter produced -- the same ascending sequence the
    Python body builds -- and the draw is ``rng.randrange(n)``, which consumes
    exactly the stream ``rng.choice(candidate_list)`` consumes and selects the
    same position.

Above two seats the engine's 2-player action space is not the space the runner
drives (see _GoEvalGame), so the fast path declines and the Python body runs.
"""

import logging
from typing import Optional, Sequence as TypingSequence

from .action_codec import NUM_ACTIONS, TWO_PLAYER_TABLE
from .baseline_agents import (
    AggressiveSnapAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    RandomAgent,
    RandomLateCambiaAgent,
    RandomNoCambiaAgent,
)
from ..constants import GameAction
from ..ffi import bridge

logger = logging.getLogger(__name__)

# baselines.Kind in engine/baselines/agents.go. Kept in the same order.
KIND_RANDOM = 0
KIND_RANDOM_NO_CAMBIA = 1
KIND_RANDOM_LATE_CAMBIA = 2
KIND_IMPERFECT_GREEDY = 3
KIND_MEMORY_HEURISTIC = 4
KIND_AGGRESSIVE_SNAP = 5

#: cambia_baseline_choose writes a two-int header before any candidate list.
_CHOOSE_HEADER = 2

#: Answer kinds in that header's first slot.
_ANSWER_DECIDED = 0
_ANSWER_UNIFORM = 1

#: Written length of a decided answer, which is the header alone. A uniform one
#: is always longer, so the return value alone identifies the common case.
_ANSWER_DECIDED_LEN = _CHOOSE_HEADER

#: Out-parameter length cambia_baseline_choose requires.
_BUF_LEN = _CHOOSE_HEADER + NUM_ACTIONS

#: The 2-player index -> action table, bound here so the hot path indexes it
#: directly instead of paying index_to_action's bounds check per decision.
_ACTION_TABLE = TWO_PLAYER_TABLE


class GoBaselineMixin:
    """Engine-side decision for a baseline, with the Python body as fallback.

    Mixed in ahead of the reference class, so ``super().choose_action`` is the
    reference body.
    """

    #: baselines.Kind this policy maps to. Set by each subclass.
    GO_KIND: int = -1

    #: Read by _GoEvalGame on a two-seat Go game, where this agent always takes
    #: the engine-side path: it reads the legal set off the engine itself, so
    #: the runner does not fetch or decode one for it.
    decides_engine_side = True

    # Declared on the class so _GoEvalGame's per-game reset finds the attribute
    # on a baseline that has not yet played a game (the reference classes only
    # grow it inside _init_memory).
    _last_game_id = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._go_handle: Optional[int] = None
        self._go_buf = None
        self._go_lib = None
        # The FFI entry points are bound once rather than looked up per
        # decision: this is the loop's hot path and an attribute chain on it
        # costs as much as the crossing.
        self._go_choose = None
        self._go_reset = None
        # Which game the path choice below was last made for, what it was, and
        # that game's engine handle.
        self._go_gid: Optional[int] = None
        self._go_fast = False
        self._go_game_h = -1

    # --- FFI plumbing ---

    def _go_threshold(self) -> int:
        """The Cambia threshold the engine-side policy reads.

        Only the two threshold policies have one; the reference's aggressive
        and random policies use their own constants or none.
        """
        return int(getattr(self, "cambia_threshold", 0))

    def _ensure_go_agent(self) -> int:
        """Allocate this seat's engine-side policy once, and return its handle."""
        if self._go_handle is not None:
            return self._go_handle
        lib = bridge.get_lib()
        handle = int(
            lib.cambia_baseline_new(
                self.GO_KIND,
                self.player_id,
                self._go_threshold(),
                int(getattr(self, "n_turns", 0)),
            )
        )
        if handle < 0:
            raise RuntimeError(
                f"cambia_baseline_new failed (returned {handle}) for kind "
                f"{self.GO_KIND} seat {self.player_id}"
            )
        self._go_lib = lib
        self._go_choose = lib.cambia_baseline_choose
        self._go_reset = lib.cambia_baseline_reset
        self._go_handle = handle
        self._go_buf = bridge.new_int32_buffer(_BUF_LEN)
        return handle

    def _draw_uniform(self, count: int) -> int:
        """Pick one of ``count`` candidates.

        Only the random policies reach this; a heuristic never returns a
        uniform answer. ``randrange(count)``, not ``choice(candidate_list)``:
        ``Random.choice`` indexes its argument with ``self._randbelow(len(seq))``
        and ``Random.randrange`` returns that same draw, so the two consume the
        same words of the same stream and land on the same position. The
        identity is pinned by tests/test_go_baseline_parity.py.
        """
        rng = getattr(self, "_rng", None)
        if rng is None:
            raise RuntimeError(
                f"{type(self).__name__} got a uniform answer from the engine but "
                "carries no RNG; only the random baselines draw."
            )
        return rng.randrange(count)

    # --- Decision ---

    def choose_action(
        self, game_state, legal_actions: TypingSequence[GameAction]
    ) -> GameAction:
        gid = id(game_state)
        # Pick the path once per game. _last_game_id being None is _GoEvalGame's
        # explicit new-game signal, which matters because handles come from a
        # pool and an id() can repeat; otherwise a changed id() is the signal.
        if self._go_gid != gid or self._last_game_id is None:
            self._go_gid = gid
            game_h = getattr(game_state, "handle", None)
            self._go_fast = game_h is not None and game_state.num_players() == 2
            if self._go_fast:
                self._go_game_h = game_h
                handle = self._ensure_go_agent()
                if int(self._go_reset(handle)) < 0:
                    raise RuntimeError(f"cambia_baseline_reset failed on handle {handle}")
                # Claim the game, so the inherited Python body's own new-game
                # check never fires behind this one and re-seeds a memory the
                # engine now owns.
                self._last_game_id = gid

        if not self._go_fast:
            return super().choose_action(game_state, legal_actions)

        buf = self._go_buf
        written = self._go_choose(self._go_handle, self._go_game_h, buf, _BUF_LEN)
        if written == _ANSWER_DECIDED_LEN and buf[0] == _ANSWER_DECIDED:
            return _ACTION_TABLE[buf[1]]
        if written == -3:
            # The engine refused the table: more than two seats, and the
            # 2-player action space is not the one being driven.
            self._go_fast = False
            self._last_game_id = None
            return super().choose_action(game_state, legal_actions)
        if written == -2:
            raise ValueError(
                f"{type(self).__name__} P{self.player_id} cannot choose from empty "
                "legal actions."
            )
        if written < _CHOOSE_HEADER:
            raise RuntimeError(
                f"cambia_baseline_choose failed (returned {written}) on handle "
                f"{self._go_handle}"
            )

        if buf[0] != _ANSWER_UNIFORM:
            raise RuntimeError(
                f"cambia_baseline_choose wrote an unknown answer kind {buf[0]}"
            )
        count = int(buf[1])
        if count <= 0:
            raise ValueError(
                f"{type(self).__name__} P{self.player_id} cannot choose from empty "
                "legal actions."
            )
        return _ACTION_TABLE[buf[_CHOOSE_HEADER + self._draw_uniform(count)]]

    # --- Lifecycle ---

    def release_go_agent(self) -> None:
        """Free the engine-side policy. Idempotent."""
        if self._go_handle is None:
            return
        try:
            self._go_lib.cambia_baseline_free(self._go_handle)
        finally:
            self._go_handle = None
            self._go_buf = None

    def __del__(self) -> None:
        try:
            self.release_go_agent()
        except Exception:  # JUSTIFIED: interpreter teardown must not raise
            pass


class GoRandomAgent(GoBaselineMixin, RandomAgent):
    """RandomAgent, decided engine-side."""

    GO_KIND = KIND_RANDOM


class GoRandomNoCambiaAgent(GoBaselineMixin, RandomNoCambiaAgent):
    """RandomNoCambiaAgent, decided engine-side."""

    GO_KIND = KIND_RANDOM_NO_CAMBIA


class GoRandomLateCambiaAgent(GoBaselineMixin, RandomLateCambiaAgent):
    """RandomLateCambiaAgent, decided engine-side.

    n_turns is handed to the engine at construction, so a non-default one is
    honoured rather than silently replaced by the engine's own default.
    """

    GO_KIND = KIND_RANDOM_LATE_CAMBIA


class GoImperfectGreedyAgent(GoBaselineMixin, ImperfectGreedyAgent):
    """ImperfectGreedyAgent, decided engine-side."""

    GO_KIND = KIND_IMPERFECT_GREEDY


class GoMemoryHeuristicAgent(GoBaselineMixin, MemoryHeuristicAgent):
    """MemoryHeuristicAgent, decided engine-side."""

    GO_KIND = KIND_MEMORY_HEURISTIC


class GoAggressiveSnapAgent(GoBaselineMixin, AggressiveSnapAgent):
    """AggressiveSnapAgent, decided engine-side."""

    GO_KIND = KIND_AGGRESSIVE_SNAP
