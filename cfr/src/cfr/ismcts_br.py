"""
cfr/src/cfr/ismcts_br.py

Information-Set Monte-Carlo Tree Search Best Response (ISMCTS-BR): a search-based
exploitability estimator that is a tighter (higher, closer-to-true) bound than the
one-ply Local Best Response in ``src.cfr.lbr`` / ``src.cfr.sampled_lbr``.

Motivation
----------
LBR estimates exploitability with a ONE-PLY lookahead: at each sampled infoset it
tries every immediate action, rolls the continuation out under a fixed policy, and
takes the max. A responder that can only improve its FIRST action understates the
true best response whenever the gain needs a coordinated multi-turn line. ISMCTS-BR
lifts the one-ply cap: it runs a determinized information-set tree search over the
responder's whole future decision sequence, so the estimated best-response value is
>= the one-ply LBR value for the same target policy (a tighter lower bound on true
exploitability).

Definition (matches tools/tiny_solver.py exact BR)
--------------------------------------------------
Exploitability of a target policy is measured from the responder seat (P0 by the
LBR convention):

    exploitability = BR_value - game_value

  - game_value = E[ utility_P0 | P0 plays the target, P1 plays the fixed opponent ].
  - BR_value   = E[ utility_P0 | P0 best-responds, P1 plays the fixed opponent ].

With the opponent fixed at the uniform-random policy and the target also uniform,
these are exactly ``tools.tiny_solver._policy_value`` (onp0) and
``tools.tiny_solver._br_value(root, 0, {})`` (br0): the empty-policy path in the
solver's ``_lookup`` returns a uniform distribution, so ISMCTS-BR calibrates
against the solver's EXACT best-response gap on the tiny {A,6} game (see
tests/test_ismcts_br.py for the measured tolerance).

Algorithm (SO-ISMCTS, single-observer = the responder)
------------------------------------------------------
Search phase (build a value tree keyed by the responder's information):

  Each simulation:
    1. Sample a determinization: a full game state drawn from the deal distribution
       (a fresh engine deal; ``deal_seeds`` restricts the pool for calibration so
       the search integrates over exactly the solver's K-deal chance root).
    2. Descend from the game start. At a responder node select an action by UCB1
       over the responder's own actions; at an opponent node sample from the fixed
       target/opponent policy; stockpile draws resolve from the determinization's
       deck (chance is fixed per determinization, averaged over the pool).
    3. Expand one new responder node, then roll the rest out (responder uniform,
       opponent fixed), and back the responder's terminal utility up the path.

  Tree nodes are keyed by the responder's PERFECT-RECALL information state --
  (initial peek + own draws + the public action/reveal sequence), the same key
  ``tools.tiny_solver`` builds with ``perfect_recall=True`` -- so states the
  responder cannot distinguish share statistics (no strategy fusion). Because the
  opponent is fixed, the responder faces a single-agent MDP and UCB1 converges to
  the exact information-set best response as the budget grows.

Evaluation phase (read the best-response value off the tree):

  The greedy responder policy (the most-visited action per information set --
  robust child -- uniform on unseen sets) is played out over the deal distribution
  to estimate BR_value; the target self-play line estimates game_value;
  exploitability is their difference.

Engine reuse
------------
This reuses the SAME game model, deep-copy, and rollout surface as the LBR/battery
paths (``src.game.engine.CambiaGameState``; opponents from
``src.agents.baseline_agents``): no second game model is introduced. The
information-key construction mirrors ``tools/tiny_solver.py``'s builder
(priv_init + priv_draw + pub_path) but keys card identity by RANK, not the full
suit-bearing repr (see ``_card_id``): suits are payoff- and dynamics-irrelevant in
Cambia, so a rank-keyed best response has the exact same value as the solver's
suit-distinguishing br0 while collapsing the information-set space enough for the
search to converge at a tractable budget. On the {A,6} game (no ability reveals
fire for the A/6 ranks) this is genuine perfect recall; on the full 54-card game
the same key omits ability-peek contents, so the estimator there is a constrained
(still multi-ply, still >= one-ply LBR) best response. The tiny-game calibration is
the correctness anchor.

Determinism
-----------
All randomness derives from ``seed`` (a ``random.Random`` master stream: per-deal
seeds, the UCB rollout stream, and the default opponent's stream). Two calls with
the same seed and arguments return identical numbers.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.game.engine import CambiaGameState
from src.constants import ActionDrawStockpile

logger = logging.getLogger(__name__)

# Engine loggers that emit expected per-decision chatter under uniform play (e.g.
# invalid-snap penalty warnings): muted for the duration of a search/eval run so
# the estimator does not flood output, then restored.
_QUIET_ENGINE_LOGGERS = ("src.game", "src.game.engine", "src.agent_state")

# Exploitability is measured from the responder seat; P0 by the LBR convention.
_RESPONDER_ID = 0

# UCB1 exploration constant on the +/-1 utility scale. The textbook sqrt(2) assumes
# rewards in [0, 1]; the exploration bonus must scale with the reward range, and
# these utilities span [-1, 1] (width 2), so the matched constant is 2*sqrt(2). That
# is equivalent to normalizing backed-up values to [0, 1] and keeping sqrt(2) (same
# argmax: the +1 offset is constant across actions). Calibrated worst |error| on the
# tiny-game exact BR stays within CALIB_TOL at this value (tests/test_ismcts_br.py).
_DEFAULT_UCB_C = 2.0 * math.sqrt(2)  # ~= 2.8284, matched to the width-2 utility range

# Factory signature: (player_id, config) -> agent exposing choose_action.
OpponentFactory = Callable[[int, Any], Any]

# A tree node: [visit_count, per_action_visits, per_action_value_sum].
# Keyed by (info_key, num_legal_actions).
_Node = List[Any]


def _resolve_decision_cap(config: Any) -> int:
    """Safety cap on the number of DECISIONS per playout.

    The engine terminates games on its own (``is_terminal`` fires at
    ``max_game_turns``); this is only a guard against a pathological
    non-terminating loop. It is NOT ``max_game_turns``: a single game turn spans
    many decisions (draw, post-draw discard/replace, snap phase), so capping the
    decision loop at ``max_game_turns`` would truncate games far too early and
    score them by the timeout hand-estimate instead of the engine terminal.
    """
    engine_turns = getattr(config.cambia_rules, "max_game_turns", 0)
    if engine_turns <= 0:
        engine_turns = 500
    return max(500, engine_turns * 8)


@contextlib.contextmanager
def _quiet_engine_logs():
    """Mute the engine's expected per-decision chatter for the duration of a run,
    then restore prior levels. Mirrors ``tools.tiny_solver._quiet_src_loggers``.
    """
    saved = {n: logging.getLogger(n).level for n in _QUIET_ENGINE_LOGGERS}
    for n in _QUIET_ENGINE_LOGGERS:
        logging.getLogger(n).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)


class _UniformAgent:
    """Uniform-random policy over the repr-sorted legal set, driven by an injected
    ``random.Random`` so the whole estimator is seed-deterministic regardless of
    set-iteration order (``baseline_agents.RandomAgent`` draws from the GLOBAL
    ``random`` module and an unsorted set, which is not reproducible under hash
    randomization). This is the default opponent and matches the solver's uniform
    (empty) policy exactly.
    """

    def __init__(self, player_id: int, rng: random.Random):
        self.player_id = player_id
        self._rng = rng

    def choose_action(self, game_state: CambiaGameState, legal_actions):
        actions = sorted(legal_actions, key=repr)
        return actions[self._rng.randrange(len(actions))]


def _default_opponent_factory(seed_stream: random.Random) -> OpponentFactory:
    def factory(player_id: int, config: Any):
        # Each opponent gets its own deterministic sub-stream so a fresh opponent
        # per game/simulation stays reproducible.
        sub = random.Random(seed_stream.getrandbits(63))
        return _UniformAgent(player_id, sub)

    return factory


def _new_deal(house_rules, deal_rng: random.Random, deal_seeds: Optional[List[int]]):
    """Sample a determinization: a fresh engine deal.

    ``deal_seeds`` (calibration) restricts the pool to the solver's K deals so the
    search integrates over the exact same chance root; otherwise deals are drawn
    fresh from the full distribution (production).
    """
    if deal_seeds:
        s = deal_seeds[deal_rng.randrange(len(deal_seeds))]
    else:
        s = deal_rng.getrandbits(31)
    return CambiaGameState(house_rules=house_rules, _rng=random.Random(s))


def _card_id(card) -> str:
    """Payoff-relevant card identity for the responder's information key: the RANK,
    not the full (suit-bearing) repr.

    Suits never affect utility, legal actions, snap matching, or abilities in
    Cambia (all rank-based), so two responder histories that differ only in suits
    have identical continuation payoffs under every action sequence. Keying by rank
    is therefore a lossless abstraction for best-response purposes: the rank-keyed
    BR value equals the suit-distinguishing BR value (``tools.tiny_solver``'s
    perfect-recall br0), while collapsing the information-set space by the suit
    multiplicity so the search is dense enough to converge at a tractable budget.
    """
    r = getattr(card, "rank", None)
    return r if r is not None else repr(card)


def _responder_priv_init(state: CambiaGameState, responder: int) -> Tuple:
    """The responder's initial private knowledge: (slot, rank) for every slot it
    peeked at deal time. Rank-keyed analogue of ``tools.tiny_solver``
    Builder.priv_init.
    """
    hand = state.players[responder].hand
    peeks = state.players[responder].initial_peek_indices
    return tuple((i, _card_id(hand[i])) for i in sorted(peeks) if i < len(hand))


class _InfoKey:
    """Perfect-recall responder info key that extends in O(1) and hashes in O(1).

    Content-equivalent to the former ``("PR", priv_init, tuple(priv_draw),
    tuple(pub_path))`` tuple: two keys are equal iff their ``priv_init`` and their
    ordered ``priv_draw`` / ``pub_path`` streams match, so states the responder
    cannot distinguish share a tree node exactly as the rebuilt tuple did (node
    sharing stays content-exact, not hash-approximate). The old builder rebuilt both
    growing lists at every responder decision (O(L) copy per decision, O(L^2) over an
    L-decision playout) and hashed an O(L) tuple; this carries each stream as a cons
    chain that extends in O(1) and folds each element into a per-stream running hash,
    so ``__hash__`` is O(1) while ``__eq__`` still walks the chains for an exact
    comparison. The two stream hashes are kept independent, so the pub/draw
    interleaving order along a playout cannot affect the hash: equal content (equal
    per-stream order) always hashes equal, which is required for correct node sharing.
    """

    __slots__ = ("priv_init", "draw", "pub", "_draw_h", "_pub_h", "_hash")

    def __init__(
        self, priv_init: Tuple, draw: Tuple, pub: Tuple, draw_h: int, pub_h: int
    ):
        self.priv_init = priv_init
        self.draw = draw
        self.pub = pub
        self._draw_h = draw_h
        self._pub_h = pub_h
        self._hash = hash((priv_init, draw_h, pub_h))

    @classmethod
    def root(cls, priv_init: Tuple) -> "_InfoKey":
        """Key at the responder's first decision: empty draw/pub streams."""
        return cls(priv_init, (), (), 0, 0)

    def extend_pub(self, entry: Tuple) -> "_InfoKey":
        """Append a common-knowledge public entry; O(1)."""
        return _InfoKey(
            self.priv_init,
            self.draw,
            (self.pub, entry),
            self._draw_h,
            hash((self._pub_h, entry)),
        )

    def extend_draw(self, token: str) -> "_InfoKey":
        """Append the responder's freshly drawn (private) stockpile card; O(1)."""
        return _InfoKey(
            self.priv_init,
            (self.draw, token),
            self.pub,
            hash((self._draw_h, token)),
            self._pub_h,
        )

    @classmethod
    def from_streams(cls, priv_init: Tuple, priv_draw, pub_path) -> "_InfoKey":
        """Rebuild a key from full streams: the reference the incremental descent
        must match. Each stream is folded in its own order, so the result does not
        depend on how pub/draw extensions interleave along the playout.
        """
        key = cls.root(priv_init)
        for entry in pub_path:
            key = key.extend_pub(entry)
        for token in priv_draw:
            key = key.extend_draw(token)
        return key

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if other is self:
            return True
        if not isinstance(other, _InfoKey):
            return NotImplemented
        return (
            self._hash == other._hash
            and self.priv_init == other.priv_init
            and self.pub == other.pub
            and self.draw == other.draw
        )


def _step_tokens(
    state: CambiaGameState, action, acting: int, responder: int
) -> Tuple[Tuple, Optional[str]]:
    """Info-key tokens produced by ``action`` (already applied to ``state``): the
    common-knowledge (actor, action-repr, post-action discard-top) public entry for
    every action, and the responder's freshly drawn stockpile card (hidden from the
    opponent) or None. Same public/private split as ``tools/tiny_solver``.
    """
    try:
        top = state.get_discard_top()
    except Exception:  # JUSTIFIED: eval resilience on odd terminal states
        top = None
    top_id = _card_id(top) if top is not None else None
    pub_entry = (acting, repr(action), top_id)
    draw_token: Optional[str] = None
    if acting == responder and isinstance(action, ActionDrawStockpile):
        try:
            if state.pending_action_player == responder:
                drawn = state.pending_action_data.get("drawn_card")
                if drawn is not None:
                    draw_token = _card_id(drawn)
        except Exception:  # JUSTIFIED: pending state absent -> no draw token
            pass
    return pub_entry, draw_token


def _apply_and_track(
    state: CambiaGameState,
    action,
    acting: int,
    responder: int,
    key: "_InfoKey",
) -> "_InfoKey":
    """Apply ``action`` and return the responder's info key extended in O(1) with
    this action's public entry (and, on a responder stockpile draw, the drawn card).
    """
    state.apply_action(action)
    pub_entry, draw_token = _step_tokens(state, action, acting, responder)
    key = key.extend_pub(pub_entry)
    if draw_token is not None:
        key = key.extend_draw(draw_token)
    return key


def _terminal_util(state: CambiaGameState, responder: int) -> float:
    """Responder terminal utility, or a hand-score estimate on turn-cap timeout
    (matching the LBR tie/timeout convention so the estimators are comparable).
    """
    if state.is_terminal():
        try:
            return float(state._utilities[responder])
        except Exception:  # JUSTIFIED: eval resilience
            return 0.0
    try:
        my = sum(c.value for c in state.players[responder].hand)
        opp = sum(c.value for c in state.players[1 - responder].hand)
        if my < opp:
            return 1.0
        if my > opp:
            return -1.0
        return 0.0
    except Exception:  # JUSTIFIED: eval resilience
        return 0.0


def _rollout(
    state: CambiaGameState,
    responder: int,
    opponent,
    rng: random.Random,
    max_turns: int,
) -> float:
    """Default MCTS rollout: responder uniform-random, opponent fixed. On the
    3-turn tiny game the tree covers almost the whole game, so the rollout only
    fills the last unexpanded ply.
    """
    turns = 0
    while not state.is_terminal() and turns < max_turns:
        turns += 1
        acting = state.get_acting_player()
        if acting == -1:
            break
        legal = sorted(state.get_legal_actions(), key=repr)
        if not legal:
            break
        try:
            if acting == responder:
                action = legal[rng.randrange(len(legal))]
            else:
                action = opponent.choose_action(state, legal)
            state.apply_action(action)
        except Exception:  # JUSTIFIED: eval resilience
            break
    return _terminal_util(state, responder)


def _ucb_select(node: _Node, c: float) -> int:
    _, na, wa = node
    total = node[0]
    log_total = math.log(total + 1.0)
    best_i, best_val = 0, -float("inf")
    for i in range(len(na)):
        if na[i] == 0:
            return i
        q = wa[i] / na[i]
        u = q + c * math.sqrt(log_total / na[i])
        if u > best_val:
            best_val, best_i = u, i
    return best_i


def _greedy_action_index(
    node: Optional[_Node], num_legal: int, rng: random.Random
) -> int:
    """Best responder action for the extracted policy: argmax mean value over
    visited actions, ties broken by visit count; uniform when the information set
    was never searched.
    """
    if node is None:
        return rng.randrange(num_legal)
    _, na, wa = node
    # Robust-child selection: the most-visited action, ties broken by mean value.
    # UCB1 spends its visits on the highest-value action, so the visit count is a
    # lower-variance pointer to the best response than the raw action-mean (which
    # is noisy at low-visit actions and biases the extracted BR downward).
    best_i, best_n, best_q = -1, -1, -float("inf")
    for i in range(len(na)):
        if na[i] == 0:
            continue
        q = wa[i] / na[i]
        if na[i] > best_n or (na[i] == best_n and q > best_q):
            best_n, best_q, best_i = na[i], q, i
    if best_i < 0:
        return rng.randrange(num_legal)
    return best_i


def _simulate(
    root_state: CambiaGameState,
    tree: Dict[Tuple, _Node],
    responder: int,
    opponent,
    rng: random.Random,
    ucb_c: float,
    max_turns: int,
) -> None:
    """One ISMCTS iteration over a single determinization (``root_state``)."""
    state = copy.deepcopy(root_state)
    priv_init = _responder_priv_init(state, responder)
    key = _InfoKey.root(priv_init)
    path: List[Tuple[Tuple, int]] = []
    turns = 0
    value = 0.0

    while True:
        if state.is_terminal() or turns >= max_turns:
            value = _terminal_util(state, responder)
            break
        acting = state.get_acting_player()
        if acting == -1:
            value = _terminal_util(state, responder)
            break
        legal = sorted(state.get_legal_actions(), key=repr)
        if not legal:
            value = _terminal_util(state, responder)
            break
        turns += 1

        if acting == responder:
            nkey = (key, len(legal))
            node = tree.get(nkey)
            if node is None:
                node = [0, [0] * len(legal), [0.0] * len(legal)]
                tree[nkey] = node
                a_idx = rng.randrange(len(legal))
                path.append((nkey, a_idx))
                key = _apply_and_track(state, legal[a_idx], acting, responder, key)
                value = _rollout(state, responder, opponent, rng, max_turns - turns)
                break
            untried = [i for i in range(len(legal)) if node[1][i] == 0]
            if untried:
                a_idx = untried[rng.randrange(len(untried))]
                path.append((nkey, a_idx))
                key = _apply_and_track(state, legal[a_idx], acting, responder, key)
                value = _rollout(state, responder, opponent, rng, max_turns - turns)
                break
            a_idx = _ucb_select(node, ucb_c)
            path.append((nkey, a_idx))
            key = _apply_and_track(state, legal[a_idx], acting, responder, key)
            continue

        # Opponent (or any non-responder) node: fixed policy.
        try:
            action = opponent.choose_action(state, legal)
        except Exception:  # JUSTIFIED: eval resilience
            action = legal[rng.randrange(len(legal))]
        key = _apply_and_track(state, action, acting, responder, key)

    for nkey, a_idx in path:
        node = tree[nkey]
        node[0] += 1
        node[1][a_idx] += 1
        node[2][a_idx] += value


def _play_greedy_br_game(
    root_state: CambiaGameState,
    tree: Dict[Tuple, _Node],
    responder: int,
    opponent,
    rng: random.Random,
    max_turns: int,
) -> float:
    """Play one game with the responder following the tree's greedy (extracted BR)
    policy and the opponent fixed; return the responder's utility.
    """
    state = copy.deepcopy(root_state)
    priv_init = _responder_priv_init(state, responder)
    key = _InfoKey.root(priv_init)
    turns = 0
    while not state.is_terminal() and turns < max_turns:
        turns += 1
        acting = state.get_acting_player()
        if acting == -1:
            break
        legal = sorted(state.get_legal_actions(), key=repr)
        if not legal:
            break
        if acting == responder:
            nkey = (key, len(legal))
            a_idx = _greedy_action_index(tree.get(nkey), len(legal), rng)
            action = legal[a_idx]
        else:
            try:
                action = opponent.choose_action(state, legal)
            except Exception:  # JUSTIFIED: eval resilience
                action = legal[rng.randrange(len(legal))]
        key = _apply_and_track(state, action, acting, responder, key)
    return _terminal_util(state, responder)


def _notify(agent, method: str, *args) -> None:
    fn = getattr(agent, method, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # JUSTIFIED: eval resilience for optional agent hooks
        pass


def _play_target_game(
    root_state: CambiaGameState,
    responder: int,
    target,
    opponent,
    rng: random.Random,
    max_turns: int,
) -> float:
    """Self-play baseline: the responder plays the ``target`` policy, the opponent
    is fixed; return the responder's utility. Optional PRT-CFR agent hooks
    (initialize_state / observe_transition) are fed best-effort when present.
    """
    state = copy.deepcopy(root_state)
    _notify(target, "initialize_state", state)
    turns = 0
    while not state.is_terminal() and turns < max_turns:
        turns += 1
        acting = state.get_acting_player()
        if acting == -1:
            break
        legal = sorted(state.get_legal_actions(), key=repr)
        if not legal:
            break
        if acting == responder:
            try:
                action = target.choose_action(state, legal)
            except Exception:  # JUSTIFIED: eval resilience
                action = legal[rng.randrange(len(legal))]
        else:
            try:
                action = opponent.choose_action(state, legal)
            except Exception:  # JUSTIFIED: eval resilience
                action = legal[rng.randrange(len(legal))]
        try:
            state.apply_action(action)
        except Exception:  # JUSTIFIED: eval resilience
            break
        _notify(target, "observe_transition", state, action, acting)
    return _terminal_util(state, responder)


def ismcts_br(
    agent_wrapper,
    config,
    num_infosets: int = 2000,
    ismcts_iterations: Optional[int] = None,
    eval_games: Optional[int] = None,
    seed: int = 42,
    opponent_factory: Optional[OpponentFactory] = None,
    deal_seeds: Optional[List[int]] = None,
    responder_id: int = _RESPONDER_ID,
    ucb_c: float = _DEFAULT_UCB_C,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate the exploitability of ``agent_wrapper`` by information-set MCTS
    best response.

    Args:
        agent_wrapper: the target policy under evaluation (plays the responder seat
            for the game-value baseline; exposes ``choose_action(state, legal)`` and
            optionally ``initialize_state`` / ``observe_transition``).
        config: exposes ``cambia_rules`` (house rules govern the games).
        num_infosets: search-budget scale. ``ismcts_iterations`` and ``eval_games``
            default to this; mirrors the LBR ``num_infosets`` knob so callers that
            pass only ``num_infosets`` (e.g. the baseline-exploitability script) get
            a sensibly sized run.
        ismcts_iterations: number of search simulations (default ``num_infosets``).
        eval_games: games for the BR-value and game-value estimates each (default
            ``num_infosets``).
        seed: master RNG seed; the whole estimate is deterministic under it.
        opponent_factory: builds the fixed opponent at the non-responder seat.
            Default: uniform-random (matches the solver's empty policy and the
            tiny-game calibration). Pass a strong opponent for a Tier-B-style
            continuation, or the target itself for a self-play (NashConv) reading.
        deal_seeds: restrict determinizations to this deal-seed pool (calibration:
            the solver's ``range(K)`` at ``seed0=0``); None samples fresh deals.
        responder_id: seat measured (0 by convention).
        ucb_c: UCB1 exploration constant.
        max_turns: per-playout DECISION safety cap; the engine terminates games on
            its own well before this (default: ``_resolve_decision_cap(config)``).

    Returns:
        dict:
          exploitability: float (BR_value - game_value; clamped to >= 0)
          br_value: float
          game_value: float
          num_infosets_sampled: int (distinct responder information sets searched)
          std_err: float (standard error of the exploitability estimate)
          estimator: "ismcts_br"
          ismcts_iterations, eval_games, ucb_c: echoed search knobs
    """
    responder = responder_id
    house_rules = config.cambia_rules
    if max_turns is None:
        max_turns = _resolve_decision_cap(config)

    iters = int(ismcts_iterations if ismcts_iterations is not None else num_infosets)
    games = int(eval_games if eval_games is not None else num_infosets)

    master = random.Random(seed)
    search_deal_rng = random.Random(master.getrandbits(63))
    search_rng = random.Random(master.getrandbits(63))
    br_deal_rng = random.Random(master.getrandbits(63))
    br_rng = random.Random(master.getrandbits(63))
    gv_deal_rng = random.Random(master.getrandbits(63))
    gv_rng = random.Random(master.getrandbits(63))
    opp_seed_stream = random.Random(master.getrandbits(63))

    if opponent_factory is None:
        opponent_factory = _default_opponent_factory(opp_seed_stream)

    # ---- Search phase: build the value tree. ----
    tree: Dict[Tuple, _Node] = {}
    with _quiet_engine_logs():
        for _ in range(iters):
            deal = _new_deal(house_rules, search_deal_rng, deal_seeds)
            # A fresh opponent per simulation keeps stateful opponents (and the
            # token prefix of a PRT-CFR opponent) from leaking across playouts.
            opponent = opponent_factory(1 - responder, config)
            _simulate(deal, tree, responder, opponent, search_rng, ucb_c, max_turns)

    if not tree:
        logger.warning("ismcts_br: empty search tree (no responder decisions).")
        return {
            "exploitability": 0.0,
            "br_value": 0.0,
            "game_value": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "estimator": "ismcts_br",
            "ismcts_iterations": iters,
            "eval_games": games,
            "ucb_c": ucb_c,
        }

    # ---- Eval phase: BR value (extracted greedy policy) and game value (target). ----
    br_outcomes: List[float] = []
    gv_outcomes: List[float] = []
    with _quiet_engine_logs():
        for _ in range(games):
            deal = _new_deal(house_rules, br_deal_rng, deal_seeds)
            opponent = opponent_factory(1 - responder, config)
            br_outcomes.append(
                _play_greedy_br_game(deal, tree, responder, opponent, br_rng, max_turns)
            )
        for _ in range(games):
            deal = _new_deal(house_rules, gv_deal_rng, deal_seeds)
            opponent = opponent_factory(1 - responder, config)
            gv_outcomes.append(
                _play_target_game(
                    deal, responder, agent_wrapper, opponent, gv_rng, max_turns
                )
            )

    br_arr = np.asarray(br_outcomes, dtype=np.float64)
    gv_arr = np.asarray(gv_outcomes, dtype=np.float64)
    br_value = float(br_arr.mean())
    game_value = float(gv_arr.mean())
    exploitability = br_value - game_value

    n_br = len(br_arr)
    n_gv = len(gv_arr)
    var_br = float(br_arr.var(ddof=1)) if n_br > 1 else 0.0
    var_gv = float(gv_arr.var(ddof=1)) if n_gv > 1 else 0.0
    std_err = math.sqrt(var_br / max(n_br, 1) + var_gv / max(n_gv, 1))

    return {
        # A best response can never do worse than the target's own play; clamp
        # away MC noise that would push a near-zero gap slightly negative.
        "exploitability": max(0.0, exploitability),
        "br_value": br_value,
        "game_value": game_value,
        "num_infosets_sampled": len(tree),
        "std_err": std_err,
        "estimator": "ismcts_br",
        "ismcts_iterations": iters,
        "eval_games": games,
        "ucb_c": ucb_c,
    }
