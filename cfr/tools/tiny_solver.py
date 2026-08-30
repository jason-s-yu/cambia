"""Exact-tree solver and exploitability for tiny-Cambia (research E1).

Builds an explicit game tree of a reduced-deck Cambia variant by recursively
expanding the Python CambiaGameState, then:
  - solves it with tabular CFR+ to near-exact equilibrium,
  - computes exact best-response exploitability of ANY policy over the same tree.

Chance handling. Two chance layers exist in Cambia: the initial deal and every
stockpile draw. We model both as explicit chance nodes:
  - Deal: a synthetic root with K children, one per sampled deal (seeds 0..K-1),
    each weight 1/K. Sampling deals (rather than enumerating all) makes the deal
    layer a Monte-Carlo subgame; CFR/BR run EXACTLY on this subgame, and the
    exploitability number is the exact exploitability of the policy on the
    sampled subgame (an unbiased-in-the-limit estimate of true exploitability,
    with a deal-sampling CI you control via K and repeated solver runs).
  - Draws: enumerated over distinct drawable cards, weighted by multiplicity.
    This makes future draws fair chance (the BR cannot read the deck order).

Infosets are keyed by the production agent-state machinery (analysis_tools
helpers + AgentState.get_infoset_key + DecisionContext), so the tabular table
and the deep pipeline share the exact same infoset partition on this game.

The tree is explicit (no memoization: a node is defined by full engine+belief
state, which is not cheaply hashable). Size is bounded by K and the reduced
deck. Build is the bottleneck; CFR iterations over a built tree are fast.
"""

import argparse
import contextlib
import logging as _logging
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from fractions import Fraction

import numpy as np

from src.config import load_config
from src.constants import NUM_PLAYERS, ActionDrawStockpile
from src.utils import InfosetKey
from src.sequence_encoding import encode_observation_sequence

# The Python reference engine and the belief/observation machinery layered on it
# (src.game.engine, src.agent_state, src.analysis_tools, src.cfr.worker) are
# imported LAZILY, inside the python-backend entry points only. Importing them at
# module scope would make every consumer of this module -- prtcfr_eval.py's X2
# scorer and tiny_exact.py's certifier included -- depend on the Python engine
# even when the tree is built on the Go engine (cambia-1429). src.config,
# src.constants (action NamedTuples), src.utils (InfosetKey) and
# src.sequence_encoding carry no engine dependency and stay eager.


# Index -> GameAction inverse of encoding.action_to_index, built by enumerating
# every action variant and inverting the map (so it cannot drift from the forward
# map it inverts). The Go engine's legal mask and apply path are index-native,
# but a Decision node has to carry the SAME GameAction NamedTuples the Python
# builder put there: Decision.actions is read by the PRT-CFR trainer, the tiny
# worker, prtcfr_net and the X2 scorer, all of which route actions through
# encoding.action_to_index. Handing them integers would fork the node contract
# per backend. Keeping one contract also keeps the child ORDER the same (both
# builders sort a node's legal actions by repr), which is what makes the two
# trees agree to the bit rather than to float64 summation noise.
_GO_ACTION_BY_INDEX = None
# The same map the other way, so the hot apply path translates a GameAction to
# its FFI index with a dict lookup instead of a call back into encoding.
_GO_INDEX_BY_ACTION = {}


def _go_action_table():
    """Build (and cache) the index -> GameAction table.

    Enumerates every constructible action variant over the [0, MAX_HAND) slot
    ranges and inverts action_to_index. Asserts injectivity, so a forward-map
    change that collided two actions would fail loudly here instead of silently
    mislabelling a node's legal set.
    """
    global _GO_ACTION_BY_INDEX
    if _GO_ACTION_BY_INDEX is not None:
        return _GO_ACTION_BY_INDEX

    from src.constants import (
        ActionAbilityBlindSwapSelect,
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionAbilityPeekOtherSelect,
        ActionAbilityPeekOwnSelect,
        ActionCallCambia,
        ActionDiscard,
        ActionDrawDiscard,
        ActionDrawStockpile,
        ActionPassSnap,
        ActionReplace,
        ActionSnapOpponent,
        ActionSnapOpponentMove,
        ActionSnapOwn,
    )
    from src.encoding import MAX_HAND, NUM_ACTIONS, action_to_index

    variants = [
        ActionDrawStockpile(),
        ActionDrawDiscard(),
        ActionCallCambia(),
        ActionDiscard(use_ability=True),
        ActionDiscard(use_ability=False),
        ActionAbilityKingSwapDecision(perform_swap=True),
        ActionAbilityKingSwapDecision(perform_swap=False),
        ActionPassSnap(),
    ]
    for i in range(MAX_HAND):
        variants.append(ActionReplace(target_hand_index=i))
        variants.append(ActionAbilityPeekOwnSelect(target_hand_index=i))
        variants.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
        variants.append(ActionSnapOwn(own_card_hand_index=i))
        variants.append(ActionSnapOpponent(opponent_target_hand_index=i))
        for j in range(MAX_HAND):
            variants.append(
                ActionAbilityBlindSwapSelect(own_hand_index=i, opponent_hand_index=j)
            )
            variants.append(
                ActionAbilityKingLookSelect(own_hand_index=i, opponent_hand_index=j)
            )
            variants.append(
                ActionSnapOpponentMove(
                    own_card_to_move_hand_index=i, target_empty_slot_index=j
                )
            )

    table = [None] * NUM_ACTIONS
    reverse = {}
    for action in variants:
        idx = action_to_index(action)
        if table[idx] is not None:
            raise AssertionError(
                f"action_to_index is not injective: {action!r} and {table[idx]!r} "
                f"both map to {idx}"
            )
        table[idx] = action
        reverse[action] = idx
    _GO_ACTION_BY_INDEX = tuple(table)
    _GO_INDEX_BY_ACTION.update(reverse)
    return _GO_ACTION_BY_INDEX


def _go_actions_from_mask(mask_indices):
    """Legal-action list for a Go mask, in build_tree_python's repr order.

    Raises on an index the inverse table does not cover: that means the Go engine
    offered an action encoding.action_to_index cannot name, which would make the
    node's policy vector meaningless rather than merely misordered.
    """
    table = _go_action_table()
    out = []
    for i in mask_indices:
        action = table[i]
        if action is None:
            raise AssertionError(
                f"Go legal mask offers action index {i}, which encoding's action "
                f"space does not name; the tree's legal set cannot be keyed"
            )
        out.append(action)
    out.sort(key=repr)
    return out


def _encode_seq(hand, peek_indices, observations, observer_id, seq_cap):
    """Tokenize one player's perfect-recall observation-action stream.

    Thin wrapper over src.sequence_encoding.encode_observation_sequence so the
    tiny-solver token path and the PRT-CFR worker/eval token path share one
    implementation. observations are the player's FILTERED post-action
    observations in temporal order (the production information boundary).
    """
    return encode_observation_sequence(
        hand, peek_indices, observations, observer_id, seq_cap=seq_cap
    )


# Suppress the engine's chatty per-node warnings during full-tree expansion.
# Scoped to the expansion call path (see _quiet_src_loggers / Builder.
# build_decision_or_terminal), NOT import time: setting logger levels as a
# module-level side effect at import would permanently mute every "src.*"
# logger for the rest of the process (or pytest session) the moment anything
# imports this module, even transitively (prtcfr_trainer.py / prtcfr_eval.py
# both `from tools.tiny_solver import build_tree`). That was the root cause
# of a session-wide test flake: unrelated tests asserting on src.* log output
# would silently fail depending on import order.
_QUIET_LOGGER_PREFIX = "src."
_QUIET_EXPLICIT_LOGGERS = ("src.game", "src.agent_state", "src.game.engine")
_quiet_depth = 0


@contextlib.contextmanager
def _quiet_src_loggers():
    """Mute src.* loggers for the duration of a tree expansion, then restore.

    Reentrant via a depth counter: only the outermost enter captures levels
    and mutes; only the outermost exit restores. Safe (and cheap) to nest:
    Builder.build_decision_or_terminal enters this once per top-level call
    (once per deal in build_tree), not once per recursively-expanded node.
    """
    global _quiet_depth
    if _quiet_depth == 0:
        names = set(_QUIET_EXPLICIT_LOGGERS) | {
            n
            for n in _logging.root.manager.loggerDict
            if n.startswith(_QUIET_LOGGER_PREFIX)
        }
        saved = {n: _logging.getLogger(n).level for n in names}
        for n in names:
            _logging.getLogger(n).setLevel(_logging.CRITICAL)
    else:
        saved = None
    _quiet_depth += 1
    try:
        yield
    finally:
        _quiet_depth -= 1
        if _quiet_depth == 0 and saved is not None:
            for n, lvl in saved.items():
                _logging.getLogger(n).setLevel(lvl)


# ---- Node types (lightweight; built once, traversed many times) ----


class Terminal:
    __slots__ = ("util",)
    kind = "T"

    def __init__(self, util):
        self.util = util  # tuple (u0, u1)


class Chance:
    __slots__ = ("children", "weights", "wfrac")
    kind = "C"

    def __init__(self):
        self.children = []
        self.weights = []
        # Exact-rational chance weights (list[fractions.Fraction]), parallel to
        # ``weights``. Populated only when the tree is built with
        # ``exact_weights=True`` (default off); None otherwise. The float64 fast
        # path reads ``weights``; the exact-rational NashConv certifier
        # (tools/tiny_exact.py, cambia-530) reads ``wfrac`` so chance mass is the
        # true rational (1/K for deals, cnt/total for draws), never a rounded
        # float. Kept separate so the hot float builder/scorer is untouched.
        self.wfrac = None


class Decision:
    __slots__ = ("player", "iset", "pkey", "actions", "children", "seq_tokens")
    kind = "D"

    def __init__(self, player, iset, pkey, actions):
        self.player = player
        self.iset = iset  # production infoset key (get_infoset_key + ctx)
        # PRT-CFR token sequence for the ACTING player at this node (perfect-recall
        # observation-action stream, tokenized via src.sequence_encoding). Populated
        # only when the tree is built with tokenize=True; None otherwise. This is the
        # single-sourced parity seam: both the PRT-CFR worker (training) and the X2
        # scorer (eval) read the same node.seq_tokens via prtcfr_net.tiny_node_to_tokens,
        # so train-time and eval-time token inputs are byte-identical by construction.
        self.seq_tokens = None
        # policy key = (iset, num_actions). The production infoset key does NOT
        # determine the legal-action count (~8% of keys / 37% of visits vary);
        # production CFR silently resets regret/strategy vectors on mismatch.
        # Keying the tabular policy by (iset, nA) makes it well-defined (a strict
        # refinement of production's partition); DESCA sidesteps this via its
        # fixed-32 abstraction. We report divergence on `iset` separately.
        self.pkey = pkey
        self.actions = actions  # list of GameAction (sorted)
        self.children = []  # one node per action


def _mk_agent(game, pid, opp, cfg, init_obs):
    from src.agent_state import AgentState

    a = AgentState(
        player_id=pid,
        opponent_id=opp,
        memory_level=cfg.agent_params.memory_level,
        time_decay_turns=cfg.agent_params.time_decay_turns,
        initial_hand_size=len(game.players[pid].hand),
        config=cfg,
    )
    a.initialize(init_obs, game.players[pid].hand, game.players[pid].initial_peek_indices)
    return a


def _advance(game, action, acting, ag):
    from src.analysis_tools import AnalysisTools

    obs = AnalysisTools._create_observation_for_br(game, action, acting)
    if obs is None:
        return None
    new = {}
    for pid, a in ag.items():
        na = a.clone()
        na.update(AnalysisTools._filter_observation_for_br(obs, pid))
        new[pid] = na
    return new


class Builder:
    def __init__(
        self,
        cfg,
        max_nodes,
        enumerate_draws=True,
        perfect_recall=False,
        tokenize=False,
        seq_cap=256,
        exact_weights=False,
        production_obs=False,
    ):
        self.cfg = cfg
        self.max_nodes = max_nodes
        self.enumerate_draws = enumerate_draws
        self.perfect_recall = perfect_recall
        # Attach exact-rational chance weights (Chance.wfrac) alongside the float
        # weights. Default off keeps the hot float builder allocation-identical.
        # See Chance.wfrac and tools/tiny_exact.py (cambia-530).
        self.exact_weights = exact_weights
        # production_obs (additive, default off): when tokenizing, build obs_path
        # through the PRODUCTION worker path (src.cfr.worker._create_observation /
        # _filter_observation) instead of the analysis_tools BR path. The BR path
        # nulls peeked_cards and surfaces the drawn card at draw time already, so
        # it does not reflect the cambia-528/529 worker-side fixes; the worker
        # path does (drawn frame at the post-draw node, ability-peek result
        # frames). Used by the B3 representation-floor gate to key tabular CFR on
        # the true production token stream. Default off preserves every existing
        # caller's behavior byte-for-byte.
        self.production_obs = production_obs
        # PRT-CFR tokenization (additive, default off). When on, each Decision node
        # gets seq_tokens: the acting player's perfect-recall observation-action token
        # stream, produced by src.sequence_encoding.encode_observation_sequence over the
        # per-player FILTERED observations accumulated along the descent path. Same data
        # source as the X1 perfect-recall pkey (priv_init + priv_draw + pub_path), routed
        # through the production observation filter and the real tokenizer so the tokens
        # are genuine, lossless perfect recall. tokenize implies perfect-recall semantics
        # but does not require perfect_recall keying to be on (they are independent flags).
        self.tokenize = tokenize
        self.seq_cap = seq_cap
        # Per-player initial private state (hand contents + peeked slots), seeded per
        # deal in build_tree. obs_path[p] is the ordered list of p's filtered
        # post-action observations along the current path (push on descend, pop on
        # ascend), matching pub_path one-to-one for the public events.
        self.tok_hand = {0: [], 1: []}
        self.tok_peek = {0: (), 1: ()}
        self.obs_path = {0: [], 1: []}
        self.n = 0
        self.aborted = False
        self.iset_actions = {}  # iset -> num actions (consistency check)
        self.iset_count = 0
        # Perfect-recall keying (X1 keystone). When on, a decision node's policy
        # key is the ACTING player's genuine perfect-recall information state
        # rather than the production imperfect-recall belief abstraction.
        #
        # A perfect-recall info state for player p at node h is everything p has
        # privately observed plus everything publicly observable along the path:
        #   - priv_init[p]: p's initial private knowledge (peeked initial-hand
        #     card contents at deal time; p sees its own peeked cards).
        #   - priv_draw[p]: p's own private draw observations (the stockpile card
        #     p drew and now holds in pending; hidden from the opponent).
        #   - pub_path: the common-knowledge action+reveal sequence for ALL
        #     players: (acting, repr(action), repr(discard_top_after)). Actions
        #     are NamedTuples whose repr carries only public structure (tags,
        #     slot indices, flags) and never card contents; discarded/replaced
        #     card identities become public via the post-action discard top.
        #
        # The acting player's key = ("PR", priv_init[p], tuple(priv_draw[p]),
        # tuple(pub_path)). This is a strict refinement of the belief partition;
        # because the public path is a deterministic function that fixes the
        # legal-action count, it cures BUG-1 (key not determining nA) by
        # construction.
        #
        # Deviation from the dispatch code-map: the map sketched pkey =
        # tuple(path[acting]) over the acting player's OWN actions only. That
        # under-keys: at the root every deal yields the empty own-action history,
        # collapsing distinct dealt hands a perfect-recall player CAN tell apart
        # (it peeked its hand). That collapse is coarser than perfect recall and
        # would corrupt the verdict, so the key here additionally carries p's
        # private prefix/draws and the full public reveal sequence. Accumulators
        # live in the builder (not AgentState, whose action_history is a lossy
        # 3-slot ring and whose infoset key is belief-only).
        self.priv_init = {0: (), 1: ()}
        self.priv_draw = {0: [], 1: []}
        self.pub_path = []

    def build_decision_or_terminal(self, game, ag, depth, quiet=True):
        """Expansion entry point. Recurses via _build_decision_or_terminal.

        quiet=True (default, matching prior CLI/solver behavior): mutes the
        engine's chatty src.* per-node warnings for the duration of this call
        via _quiet_src_loggers, restoring prior levels on return. The context
        is entered once here (not per recursive node) even though this method
        is itself recursive-in-spirit; the actual recursion runs through
        _build_decision_or_terminal, which does not re-enter the mute.
        """
        if quiet:
            with _quiet_src_loggers():
                return self._build_decision_or_terminal(game, ag, depth)
        return self._build_decision_or_terminal(game, ag, depth)

    def _build_decision_or_terminal(self, game, ag, depth):
        self.n += 1
        if self.n > self.max_nodes:
            self.aborted = True
            return Terminal((0.0, 0.0))
        if game.is_terminal():
            return Terminal((game.get_utility(0), game.get_utility(1)))
        acting = game.get_acting_player()
        legal = sorted(list(game.get_legal_actions()), key=repr)
        if acting == -1 or not legal:
            return Terminal((game.get_utility(0), game.get_utility(1)))

        from src.analysis_tools import AnalysisTools

        ctx = AnalysisTools._get_decision_context(game)
        base = ag[acting].get_infoset_key()
        iset = InfosetKey(*base, ctx.value if ctx else -1)
        nA = len(legal)
        # Policy key = the production plain-tuple infoset key (InfosetKey.astuple),
        # so a loaded production tabular policy (worker uses astuple, worker.py:251)
        # matches by dict key. The production infoset key does NOT determine nA
        # (~8% of keys / 37% of visits vary); production CFR silently resets the
        # regret/strategy vector on a length mismatch. Exploitability lookups fall
        # back to uniform when the stored vector length != node nA, reproducing
        # that semantics. For my own exact CFR table we still want per-(key,nA)
        # separation; iset_actions records nA per the LAST-seen count (only used
        # for the uniform null control + CFR sizing, both length-robust).
        if self.perfect_recall:
            # X1 keystone: key by the acting player's genuine perfect-recall
            # information state (see Builder.__init__ for the construction and
            # the deviation from the code-map). Determines nA by construction.
            pkey = (
                "PR",
                self.priv_init[acting],
                tuple(self.priv_draw[acting]),
                tuple(self.pub_path),
            )
        else:
            pkey = iset.astuple()
        # iset_actions is keyed by (pkey, nA) so my own exact CFR table and the
        # uniform null control are well-defined under variable action counts.
        # The production tabular policy is looked up by the BARE node.pkey.
        self.iset_actions[(pkey, nA)] = nA
        node = Decision(acting, iset, pkey, legal)
        if self.tokenize:
            # Acting player's perfect-recall token stream: BOS + their peeked initial
            # hand + every filtered observation they have received along this path.
            node.seq_tokens = _encode_seq(
                self.tok_hand[acting],
                self.tok_peek[acting],
                self.obs_path[acting],
                acting,
                self.seq_cap,
            )

        for action in legal:
            if (
                self.enumerate_draws
                and isinstance(action, ActionDrawStockpile)
                and game.stockpile
            ):
                node.children.append(self._draw_chance(game, action, acting, ag, depth))
            else:
                # enumerate_draws=False: draw follows the realized (pre-shuffled)
                # deck order -> a single child. Draw randomness is covered by
                # merging K independent deals (Monte-Carlo over deck orders).
                node.children.append(self._apply_one(game, action, acting, ag, depth))
            if self.aborted:
                # still return a (partial) node; caller handles via Terminal stubs
                pass
        return node

    def _draw_chance(self, game, action, acting, ag, depth):
        ch = Chance()
        if self.exact_weights:
            ch.wfrac = []
        distinct = {}
        for c in game.stockpile:
            k = (c.rank, c.suit)
            distinct[k] = distinct.get(k, 0) + 1
        total = len(game.stockpile)
        orig = list(game.stockpile)
        for (rank, suit), cnt in distinct.items():
            idx = next(
                i
                for i, c in enumerate(game.stockpile)
                if (c.rank, c.suit) == (rank, suit)
            )
            card = game.stockpile.pop(idx)
            game.stockpile.append(card)
            child = self._apply_one(game, action, acting, ag, depth)
            game.stockpile[:] = list(orig)
            ch.children.append(child)
            ch.weights.append(cnt / total)
            if self.exact_weights:
                # Exact draw-chance mass: multiplicity over stockpile size. Both
                # are integers read directly from the deck, so this is the true
                # rational, not the rounded cnt/total float above.
                ch.wfrac.append(Fraction(cnt, total))
            if self.aborted:
                break
        return ch

    def _apply_one(self, game, action, acting, ag, depth):
        state_delta, undo = game.apply_action(action)
        if not callable(undo):
            return Terminal((0.0, 0.0))
        new_ag = _advance(game, action, acting, ag)
        if new_ag is None:
            try:
                undo()
            except Exception:
                pass
            return Terminal((0.0, 0.0))
        # Perfect-recall path bookkeeping: push on descend, pop on ascend so the
        # accumulators reflect exactly the path to the current subtree. The
        # public token is common knowledge (action repr + post-action discard
        # top); the private draw token is the acting player's just-drawn card,
        # which after a stockpile draw sits in pending_action_data.
        pushed_pub = False
        pushed_priv = None
        if self.perfect_recall:
            try:
                top = game.get_discard_top()
            except Exception:
                top = None
            self.pub_path.append((acting, repr(action), repr(top)))
            pushed_pub = True
            if isinstance(action, ActionDrawStockpile):
                drawn = None
                try:
                    if game.pending_action_player == acting:
                        drawn = game.pending_action_data.get("drawn_card")
                except Exception:
                    drawn = None
                if drawn is not None:
                    self.priv_draw[acting].append(repr(drawn))
                    pushed_priv = acting
        # Tokenize-mode path bookkeeping: push each player's FILTERED post-action
        # observation onto obs_path so a decision node deeper in this subtree can
        # tokenize the acting player's full perfect-recall stream. Uses the same
        # observation + filter as _advance (the production information boundary).
        pushed_obs = False
        if self.tokenize:
            from src.analysis_tools import AnalysisTools
            from src.cfr.worker import _create_observation as _wk_create_observation
            from src.cfr.worker import _filter_observation as _wk_filter_observation

            if self.production_obs:
                snaps = list(getattr(game, "snap_results_log", []) or [])
                obs = _wk_create_observation(None, action, game, acting, snaps)
                filt = _wk_filter_observation
            else:
                obs = AnalysisTools._create_observation_for_br(game, action, acting)
                filt = AnalysisTools._filter_observation_for_br
            if obs is not None:
                for pid in (0, 1):
                    self.obs_path[pid].append(filt(obs, pid))
                pushed_obs = True
        child = self._build_decision_or_terminal(game, new_ag, depth + 1)
        if pushed_obs:
            for pid in (0, 1):
                self.obs_path[pid].pop()
        if pushed_priv is not None:
            self.priv_draw[pushed_priv].pop()
        if pushed_pub:
            self.pub_path.pop()
        try:
            undo()
        except Exception:
            pass
        return child


def build_tree_python(
    cfg,
    n_deals,
    seed0,
    max_nodes_per_deal,
    enumerate_draws=True,
    perfect_recall=False,
    tokenize=False,
    seq_cap=256,
    quiet=True,
    exact_weights=False,
    production_obs=False,
):
    """Synthetic root: K deals, each weight 1/K; each is a full chance-tree.

    PYTHON-ENGINE BACKEND. Retained as the reference implementation behind
    ``build_tree(..., backend="python")``; the default backend is the Go engine
    (``build_tree_go``, cambia-1429). Every X1/X2 number recorded before
    cambia-1429 came from this function.

    tokenize (default off): populate Decision.seq_tokens with each acting player's
    perfect-recall observation-action token stream (src.sequence_encoding), the
    single-sourced input for the PRT-CFR net. Independent of perfect_recall keying.

    quiet (default True, matching prior CLI/solver behavior): mute the engine's
    chatty src.* per-node warnings for each deal's expansion (see
    Builder.build_decision_or_terminal / _quiet_src_loggers). Set False to see
    the underlying warnings, e.g. while debugging engine behavior.

    exact_weights (default off): additionally attach exact-rational chance mass
    (Chance.wfrac) to every chance node: Fraction(1, K) at the deal root and
    Fraction(cnt, total) at each draw node. The float ``weights`` are unchanged;
    the exact-rational NashConv certifier (tools/tiny_exact.py, cambia-530) reads
    ``wfrac`` so no chance mass is ever a rounded float in the exact path.
    """
    # CORRECTNESS FENCE (cambia-564): the exact tree builder enumerates chance
    # branches explicitly (the deal and every stockpile draw). The race-ON snap
    # model resolves its N-way winner from an ENGINE-INTERNAL RNG draw
    # (engine race resolveSnapRace / _resolve_snap_race), which this builder cannot
    # enumerate: it would walk that stochastic transition as a single
    # sampled-deterministic step and silently corrupt any exact NashConv computed
    # on the tree. Refuse race-ON until the winner draw is exposed as an enumerable
    # chance point.
    if getattr(cfg.cambia_rules, "snapRace", False):
        raise ValueError(
            "tiny_solver.build_tree does not support snapRace=true (race-ON): the "
            "N-way snap winner is an engine-internal RNG draw that this exact tree "
            "builder cannot enumerate, so an exact tree would treat a stochastic "
            "transition as sampled-deterministic and corrupt NashConv. Exact "
            "solving of race-ON requires exposing the winner draw as an enumerable "
            "chance node first (cambia-564 follow-up)."
        )
    from src.analysis_tools import AnalysisTools
    from src.game.engine import CambiaGameState

    root = Chance()
    all_isets = {}
    total_nodes = 0
    aborted_deals = 0
    for d in range(n_deals):
        b = Builder(
            cfg,
            max_nodes_per_deal,
            enumerate_draws=enumerate_draws,
            perfect_recall=perfect_recall,
            tokenize=tokenize,
            seq_cap=seq_cap,
            exact_weights=exact_weights,
            production_obs=production_obs,
        )
        game = CambiaGameState(
            house_rules=cfg.cambia_rules, _rng=random.Random(seed0 + d)
        )
        init_obs = AnalysisTools._create_observation_for_br(game, None, -1)
        ag = {
            0: _mk_agent(game, 0, 1, cfg, init_obs),
            1: _mk_agent(game, 1, 0, cfg, init_obs),
        }
        if tokenize:
            # Seed each player's initial private state for the tokenizer: their dealt
            # hand contents and the slots they peeked at deal time. encode_observation_sequence
            # emits the BOS-anchored init_peek prefix from these (the X1 priv_init content).
            for pid in (0, 1):
                b.tok_hand[pid] = list(game.players[pid].hand)
                b.tok_peek[pid] = tuple(game.players[pid].initial_peek_indices)
        if perfect_recall:
            # Seed each player's private prefix with the cards it peeked at deal
            # time, keyed by hand index so identical contents at different slots
            # stay distinct. This is p's genuine initial private information; it
            # also separates distinct dealt hands at the root (the collapse the
            # code-map's own-action-only key would have caused). The deal index d
            # is NOT part of any key: two deals that produce the same observation
            # for p are genuinely indistinguishable to p and correctly merge.
            for pid in (0, 1):
                peeks = tuple(
                    (i, repr(game.players[pid].hand[i]))
                    for i in sorted(game.players[pid].initial_peek_indices)
                    if i < len(game.players[pid].hand)
                )
                b.priv_init[pid] = peeks
        sub = b.build_decision_or_terminal(game, ag, 0, quiet=quiet)
        root.children.append(sub)
        root.weights.append(1.0)  # normalized below
        total_nodes += b.n
        if b.aborted:
            aborted_deals += 1
        for k, v in b.iset_actions.items():
            all_isets[k] = v
    s = sum(root.weights)
    root.weights = [w / s for w in root.weights]
    if exact_weights:
        # Deal root: K equal-mass children (build appended 1.0 each above).
        # Exact mass is Fraction(1, K), never the 0.2 float of 1.0/5.
        k = len(root.children)
        root.wfrac = [Fraction(1, k)] * k
    return root, all_isets, total_nodes, aborted_deals


# ---- Go-engine tree builder (cambia-1429) ----
#
# Same explicit-tree contract as build_tree_python (Terminal / Chance / Decision
# nodes, perfect-recall pkeys, per-node token streams), built on the Go engine
# through the FFI bridge instead of src.game.engine. Three primitives carry it:
#
#   GoEngine.from_deck(deck, starting_player, rules) -- fixes the deal AND the
#     draw order: cambia_game_new_with_deck consumes deck[0] first, so the deck
#     array doubles as the draw script (see GoBuilder._draw_chance).
#   bridge.state_save / state_restore / state_snapshot_free -- token-inclusive
#     (game, both agents) checkpoints on the SAME handles: the enumeration's
#     backtracking, replacing the Python engine's undo closures.
#   bridge.apply_games_batch -- the only FFI path that advances the game AND
#     both agents' token streams, so Decision.seq_tokens comes off the Go
#     tokenizer (the stream production training consumes) rather than a Python
#     re-derivation of it.
#
# Draw enumeration. The Go engine exposes no stockpile accessor and no way to
# reorder the stockpile, so a card is forced to the top the only way the FFI
# allows: swap it into the next-consumed deck slot, rebuild the game from the
# edited deck and replay the action prefix (_materialize). Deck positions at or
# after the next-consumed slot do not affect the current node's state, so the
# rebuilt state is the same node.
#
# The deck channel dies at the first RESHUFFLE. When the stockpile empties the
# engine recycles the discard pile (engine/actions.go attemptReshuffle) and
# shuffles it with the game's own RNG; past that the stockpile corresponds to no
# deck suffix and no FFI surface can order it. Those draws are taken as a single
# engine-resolved child and counted in GoBuilder.unenumerated_draws, and every
# enumerated draw positively asserts that the card the engine drew is the one its
# deck slot promised -- a dead channel raises GoDeckChannelLost instead of
# silently yielding a tree whose chance weights do not match its branches.


def _go_card_key(card):
    """Suit-preserving hashable identity for a Card, for use in a key component.

    src.card.Card declares ``suit`` as ``field(compare=False)``, so Card objects
    compare and HASH BY RANK ALONE: putting them straight into a perfect-recall
    key silently merges all four suits of a rank and coarsens the infoset
    partition by up to 4x per card position. build_tree_python sidesteps this by
    keying on ``repr(card)``; this returns the same information as a tuple.
    ``(rank, suit)`` also matches the Python builder's granularity on jokers,
    which repr and this both conflate (a canonical index would separate 52 from
    53 and give a strictly finer partition).
    """
    if card is None:
        return None
    return (card.rank, card.suit)


def _go_deck_indices(cards):
    """Canonical card indices for a deal-order card list.

    Mirrors bridge.extract_deck_from_python_game's joker handling (the first
    joker seen in deal order takes index 52, the second 53);
    python_card_to_go_index alone collapses both onto 52.
    """
    from src.ffi.bridge import python_card_to_go_index

    out = []
    jokers = 0
    for c in cards:
        idx = python_card_to_go_index(c)
        if idx == 52:
            idx = 52 + min(jokers, 1)
            jokers += 1
        out.append(idx)
    return out


def go_deal_decks(cfg, n_deals, seed0):
    """The K deals build_tree_python draws, as (deck_indices, starting_player).

    Reproduces CambiaGameState._setup_game's RNG consumption without importing
    the engine: one random.Random(seed0 + d) per deal, shuffle() over
    create_standard_deck(...), then randint(0, num_players - 1) for the starting
    seat, with no RNG use in between. The Python deal and every Python draw pop
    from the END of that list while cambia_game_new_with_deck consumes deck[0]
    first, so the Go deck is the reversed list -- which lines the round-robin
    deal, the discard flip and the draw order up card for card. Asserted against
    the engine's own deal in tests/test_tiny_solver_go_backend.py.
    """
    from src.card import create_standard_deck

    rules = cfg.cambia_rules
    n_players = int(getattr(rules, "num_players", NUM_PLAYERS) or NUM_PLAYERS)
    out = []
    for d in range(n_deals):
        rng = random.Random(seed0 + d)
        deck = create_standard_deck(
            include_jokers=rules.use_jokers,
            num_decks=getattr(rules, "num_decks", 1),
            deck_ranks=getattr(rules, "deck_ranks", None),
        )
        rng.shuffle(deck)
        order = list(reversed(deck))
        starting_player = rng.randint(0, n_players - 1)
        out.append((_go_deck_indices(order), starting_player))
    return out


class GoDeckChannelLost(RuntimeError):
    """An enumerated draw did not yield the card its deck slot promised.

    Raised, never swallowed: it means the deck-order channel GoBuilder._draw_chance
    enumerates through no longer controls the stockpile (a reshuffle the fence
    failed to notice), so continuing would build a tree whose chance structure
    does not match its own weights.
    """


class GoBuilder:
    """One deal's chance-tree, expanded on the Go engine.

    Counterpart of Builder: same node types, same node-counter semantics (``n``
    counts decision-or-terminal expansions, not chance nodes), same perfect-recall
    key construction. The key components are Go-native and injective against the
    Python ones they replace -- card identities are the interned Card objects
    behind canonical indices (repr-equivalent), and an action is its global
    [0, NUM_ACTIONS) index, which encoding.action_to_index maps one-to-one from
    the GameAction NamedTuples Builder keyed on.
    """

    def __init__(
        self,
        cfg,
        deck,
        starting_player,
        max_nodes,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=False,
        seq_cap=256,
        exact_weights=False,
    ):
        from src.ffi import bridge

        self.cfg = cfg
        self.rules = cfg.cambia_rules
        self.bridge = bridge
        self.base_deck = [int(x) for x in deck]
        self.deck = list(self.base_deck)
        self.starting_player = int(starting_player)
        self.max_nodes = max_nodes
        self.enumerate_draws = enumerate_draws
        self.perfect_recall = perfect_recall
        self.tokenize = tokenize
        self.seq_cap = seq_cap
        self.exact_weights = exact_weights
        self.n = 0
        self.aborted = False
        self.iset_actions = {}
        self.prefix = []
        self.priv_init = {}
        self.priv_draw = {0: [], 1: []}
        self.pub_path = []
        # Reshuffles seen on the CURRENT root-to-node path (pushed on descend,
        # popped on ascend). Non-zero fences off draw enumeration: past a
        # reshuffle the deck array no longer describes the stockpile.
        self.reshuffles_on_path = 0
        # Build-wide diagnostics, aggregated by build_tree_go into its stats dict.
        self.unenumerated_draws = 0
        self.reshuffle_draws = 0
        self.rebuilds = 0
        self.unchecked_draws = 0
        # Actions the legal mask offered that the engine then refused. Should be 0:
        # a non-zero count means the mask and the apply path disagree, and the
        # builder stubbed a whole subtree out as a zero-utility Terminal.
        self.rejected_actions = 0
        self.eng = None
        self.a0 = None
        self.a1 = None

    # -- handle lifecycle --

    def open(self):
        from src.ffi.bridge import GoAgentState, GoEngine

        _go_action_table()  # prime the index maps before the first apply

        self.eng = GoEngine.from_deck(self.deck, self.starting_player, self.rules)
        self.a0 = GoAgentState(self.eng, 0)
        self.a1 = GoAgentState(self.eng, 1)
        hr = self.eng.get_house_rules()
        if hr.num_players != 2:
            self.close()
            raise NotImplementedError(
                f"GoBuilder is 2-player only (got num_players={hr.num_players}): "
                "the token-inclusive checkpoint it backtracks on "
                "(cambia_state_save) is a two-agent surface."
            )
        peek = min(int(hr.initial_view_count), int(hr.cards_per_player))
        for pid in (0, 1):
            hand = self.eng.get_player_hand(pid)
            self.priv_init[pid] = tuple(
                (i, _go_card_key(hand[i])) for i in range(peek) if i < len(hand)
            )
        return self

    def close(self):
        for h in (self.a1, self.a0, self.eng):
            if h is not None:
                h.close()
        self.a1 = self.a0 = self.eng = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- state plumbing --

    def _apply_raw(self, eng, a0, a1, action):
        """Advance the game plus both token streams by one action.

        ``action`` is a GameAction (the node contract); the FFI is index-native,
        so it is translated here, at the boundary, and nowhere else.
        """
        self.bridge.apply_games_batch(
            [eng.handle], [a0.handle], [a1.handle], [_GO_INDEX_BY_ACTION[action]]
        )

    def _materialize(self, deck, prefix):
        """Put the persistent (game, a0, a1) handles at (deck, prefix)'s state.

        Builds a throwaway game from ``deck``, replays ``prefix`` into it, then
        value-copies the result onto the persistent handles through
        state_save/state_restore. Going via a snapshot rather than swapping in the
        fresh handles is what keeps every OUTER frame's checkpoint restorable:
        cambia_state_restore copies into whichever handles it is handed, so the
        persistent handle ids never move under a recursion holding checkpoints
        against them.
        """
        from src.ffi.bridge import GoAgentState, GoEngine

        tmp = GoEngine.from_deck(deck, self.starting_player, self.rules)
        t0 = GoAgentState(tmp, 0)
        t1 = GoAgentState(tmp, 1)
        try:
            for a in prefix:
                self._apply_raw(tmp, t0, t1, a)
            snap = self.bridge.state_save(tmp.handle, t0.handle, t1.handle)
            try:
                self.bridge.state_restore(
                    self.eng.handle, snap, self.a0.handle, self.a1.handle
                )
            finally:
                self.bridge.state_snapshot_free(snap)
        finally:
            t1.close()
            t0.close()
            tmp.close()
        self.rebuilds += 1

    def _tokens(self, acting):
        agent = self.a0 if acting == 0 else self.a1
        return self.bridge.frame_aligned_window(
            agent.tokens(), seq_cap=self.seq_cap, add_bos_eos=True
        )

    def _terminal(self):
        u = self.eng.get_utility()
        return Terminal((float(u[0]), float(u[1])))

    # -- expansion --

    def build(self):
        return self._build(0)

    def _build(self, depth):
        self.n += 1
        if self.n > self.max_nodes:
            self.aborted = True
            return Terminal((0.0, 0.0))
        if self.eng.is_terminal():
            return self._terminal()
        acting = self.eng.acting_player()
        legal = _go_actions_from_mask(
            int(i) for i in self.eng.legal_actions_mask().nonzero()[0]
        )
        if acting < 0 or not legal:
            return self._terminal()

        nA = len(legal)
        # Perfect-recall key: the acting seat's initial private knowledge, its own
        # draws, and the full public action/reveal sequence -- the same three
        # components Builder keys on, in Go-native currency. Determines nA by
        # construction, the property the X1 keystone rests on.
        pkey = (
            "PR",
            self.priv_init[acting],
            tuple(self.priv_draw[acting]),
            tuple(self.pub_path),
        )
        self.iset_actions[(pkey, nA)] = nA
        # iset (the production imperfect-recall belief key) is None on this
        # backend: it comes from AgentState.get_infoset_key, which has no FFI
        # export. Nothing on the perfect-recall path reads it; build_tree_go
        # refuses perfect_recall=False rather than hand back a half-keyed node.
        node = Decision(acting, None, pkey, legal)
        if self.tokenize:
            node.seq_tokens = self._tokens(acting)

        can_enumerate = (
            self.enumerate_draws
            and self.reshuffles_on_path == 0
            and self.eng.stock_len() > 0
        )
        for action in legal:
            is_draw = isinstance(action, ActionDrawStockpile)
            enumerable = can_enumerate and is_draw
            if enumerable:
                node.children.append(self._draw_chance(action, acting, depth))
            else:
                if is_draw and self.enumerate_draws:
                    self.unenumerated_draws += 1
                node.children.append(self._apply_one(action, acting, depth))
        return node

    def _draw_chance(self, action, acting, depth):
        """Chance node over the distinct cards the stockpile can yield.

        Weights are multiplicity over stockpile size, matching
        Builder._draw_chance. The stockpile IS the deck suffix from the
        next-consumed slot ``m`` onward: cambia_game_new_with_deck loads deck[i]
        at Stockpile[len-1-i] and the deal plus every draw consume from the top,
        so ``m = len(deck) - stock_len``.
        """
        ch = Chance()
        if self.exact_weights:
            ch.wfrac = []
        stock = self.eng.stock_len()
        m = len(self.deck) - stock
        tail = self.deck[m:]
        total = len(tail)
        # Distinct cards in Builder._draw_chance's insertion order, so the two
        # backends lay a chance node's children out identically and the reductions
        # in _cfr / _policy_value / _br_eval sum the same terms in the same order.
        # The Python engine's stockpile list holds the next-drawn card LAST and it
        # iterates the list front to back, which is the deck suffix reversed.
        counts = {}
        for c in reversed(tail):
            counts[c] = counts.get(c, 0) + 1
        saved = list(self.deck)
        for card, cnt in counts.items():
            if self.deck[m] != card:
                j = self.deck.index(card, m)
                self.deck[m], self.deck[j] = self.deck[j], self.deck[m]
                self._materialize(self.deck, self.prefix)
            child = self._apply_one(action, acting, depth, expect_draw=card)
            ch.children.append(child)
            ch.weights.append(cnt / total)
            if self.exact_weights:
                # Exact draw-chance mass: multiplicity over stockpile size, both
                # integers read off the deck, so this is the true rational rather
                # than the rounded float above.
                ch.wfrac.append(Fraction(cnt, total))
            if self.deck != saved:
                self.deck[:] = saved
                self._materialize(self.deck, self.prefix)
            if self.aborted:
                break
        return ch

    def _apply_one(self, action, acting, depth, expect_draw=None):
        snap = self.bridge.state_save(self.eng.handle, self.a0.handle, self.a1.handle)
        try:
            stock_before = self.eng.stock_len()
            m_before = len(self.deck) - stock_before
            try:
                self._apply_raw(self.eng, self.a0, self.a1, action)
            except RuntimeError:
                # The engine refused an action its own legal mask offered.
                # Builder._apply_one returns a zero-utility Terminal stub in the
                # same situation (a non-callable undo), so the shape is mirrored --
                # but the stub silently replaces a whole subtree, so it is counted
                # and build_tree_go warns on any non-zero total.
                self.rejected_actions += 1
                self.bridge.state_restore(
                    self.eng.handle, snap, self.a0.handle, self.a1.handle
                )
                return Terminal((0.0, 0.0))

            # The stockpile only ever grows by a reshuffle: engine/actions.go
            # attemptReshuffle recycles the discard pile when a draw finds it
            # empty. That is where the deck-order channel dies, so the flag fences
            # off draw enumeration for the rest of this path.
            recycled = self.eng.stock_len() > stock_before

            pend = self.eng.get_pending()
            drawn = pend.drawn_card if pend.seat == acting else None
            if expect_draw is not None:
                expected_card = self.bridge.card_from_index(int(expect_draw))
                if drawn is None:
                    # No pending record to read the drawn card back from (the draw
                    # auto-resolved). Counted, not asserted.
                    self.unchecked_draws += 1
                elif _go_card_key(drawn) != _go_card_key(expected_card):
                    # Compared through _go_card_key, not ==: Card equality drops
                    # the suit, so a suit-wrong draw would pass an == check.
                    raise GoDeckChannelLost(
                        f"enumerated draw expected {expected_card!r} from deck slot "
                        f"{m_before} but the engine drew {drawn!r}: the deck-order "
                        f"channel no longer controls the stockpile (reshuffle). "
                        f"deck={self.deck} prefix={self.prefix}"
                    )

            pushed_priv = None
            if self.perfect_recall:
                self.pub_path.append(
                    (
                        acting,
                        repr(action),
                        _go_card_key(self.eng.get_discard_top()),
                    )
                )
                if isinstance(action, ActionDrawStockpile) and drawn is not None:
                    self.priv_draw[acting].append(_go_card_key(drawn))
                    pushed_priv = acting
            if recycled:
                self.reshuffles_on_path += 1
                self.reshuffle_draws += 1
            self.prefix.append(action)

            child = self._build(depth + 1)

            self.prefix.pop()
            if recycled:
                self.reshuffles_on_path -= 1
            if self.perfect_recall:
                if pushed_priv is not None:
                    self.priv_draw[pushed_priv].pop()
                self.pub_path.pop()
            self.bridge.state_restore(
                self.eng.handle, snap, self.a0.handle, self.a1.handle
            )
            return child
        finally:
            self.bridge.state_snapshot_free(snap)


def build_tree_go(
    cfg,
    n_deals,
    seed0,
    max_nodes_per_deal,
    enumerate_draws=True,
    perfect_recall=True,
    tokenize=False,
    seq_cap=256,
    exact_weights=False,
    stats=None,
):
    """Synthetic root over K Go-engine deals; the build_tree_python counterpart.

    perfect_recall must be True: the imperfect-recall belief key comes from
    AgentState.get_infoset_key, which the FFI does not export, so keying by it
    would need the Python engine back. Every X1/X2 consumer keys perfect-recall.

    ``stats`` (optional dict) receives the build's diagnostics --
    ``unenumerated_draws`` (draw points the deck channel could not enumerate,
    i.e. downstream of a reshuffle), ``reshuffle_draws`` (transitions where the
    engine recycled the discard pile), ``rebuilds`` (deck-surgery replays),
    ``unchecked_draws`` and ``rejected_actions``. A non-zero ``unenumerated_draws`` means the tree carries
    chance points collapsed onto one engine-sampled outcome; read that count
    before trusting any exactness claim about the tree.
    """
    if getattr(cfg.cambia_rules, "snapRace", False):
        raise ValueError(
            "build_tree_go does not support snapRace=true (race-ON): the N-way "
            "snap winner is an engine-internal RNG draw an exact tree builder "
            "cannot enumerate, so the tree would treat a stochastic transition as "
            "sampled-deterministic and corrupt NashConv (the same fence "
            "build_tree_python carries, cambia-564)."
        )
    if not perfect_recall:
        raise NotImplementedError(
            "build_tree_go supports perfect_recall=True only: the production "
            "imperfect-recall key is AgentState.get_infoset_key + DecisionContext "
            "and the FFI exports neither, so an imperfect-recall Go tree would "
            "reintroduce the Python engine this backend exists to retire. Use "
            "build_tree(..., backend='python') for belief-keyed trees."
        )
    root = Chance()
    all_isets = {}
    total_nodes = 0
    aborted_deals = 0
    agg = {
        "unenumerated_draws": 0,
        "reshuffle_draws": 0,
        "rebuilds": 0,
        "unchecked_draws": 0,
        "rejected_actions": 0,
    }
    for deck, starting_player in go_deal_decks(cfg, n_deals, seed0):
        b = GoBuilder(
            cfg,
            deck,
            starting_player,
            max_nodes_per_deal,
            enumerate_draws=enumerate_draws,
            perfect_recall=perfect_recall,
            tokenize=tokenize,
            seq_cap=seq_cap,
            exact_weights=exact_weights,
        )
        try:
            b.open()
            sub = b.build()
        finally:
            b.close()
        root.children.append(sub)
        root.weights.append(1.0)  # normalized below
        total_nodes += b.n
        if b.aborted:
            aborted_deals += 1
        for k, v in b.iset_actions.items():
            all_isets[k] = v
        agg["unenumerated_draws"] += b.unenumerated_draws
        agg["reshuffle_draws"] += b.reshuffle_draws
        agg["rebuilds"] += b.rebuilds
        agg["unchecked_draws"] += b.unchecked_draws
        agg["rejected_actions"] += b.rejected_actions
    s = sum(root.weights)
    root.weights = [w / s for w in root.weights]
    if exact_weights:
        k = len(root.children)
        root.wfrac = [Fraction(1, k)] * k
    if agg["rejected_actions"]:
        warnings.warn(
            f"build_tree_go: the engine refused {agg['rejected_actions']} actions "
            "its own legal mask offered, and each one stubbed a subtree out as a "
            "zero-utility Terminal. The mask and the apply path disagree; the tree "
            "is wrong wherever that happened.",
            stacklevel=2,
        )
    if agg["unchecked_draws"]:
        warnings.warn(
            f"build_tree_go: {agg['unchecked_draws']} enumerated draws left no "
            "pending record to read the drawn card back from, so neither the deck "
            "channel nor the acting seat's private-draw key component could be "
            "validated at those nodes (build_tree_python's priv_draw guard has the "
            "same blind spot). The tree is only as trustworthy as that count is "
            "small; it is 0 on config/tiny_norecall.yaml and "
            "config/tiny_2card_plateau.yaml.",
            stacklevel=2,
        )
    if agg["unenumerated_draws"]:
        warnings.warn(
            f"build_tree_go: {agg['unenumerated_draws']} draw points could not be "
            f"enumerated ({agg['reshuffle_draws']} transitions recycled the discard "
            "pile). Downstream of a reshuffle the stockpile corresponds to no deck "
            "suffix and the FFI exposes no way to order it, so those chance points "
            "collapse onto the one outcome the engine's own RNG produced. NashConv "
            "on this tree is exact FOR this tree, not for the game.",
            stacklevel=2,
        )
    if stats is not None:
        stats.update(agg)
    return root, all_isets, total_nodes, aborted_deals


def _go_tokenizer_version():
    from src.ffi.bridge import get_tokenizer_version

    try:
        return get_tokenizer_version()
    except Exception:  # noqa: BLE001 - diagnostics string only
        return "?"


def build_tree(
    cfg,
    n_deals,
    seed0,
    max_nodes_per_deal,
    enumerate_draws=True,
    perfect_recall=False,
    tokenize=False,
    seq_cap=256,
    quiet=True,
    exact_weights=False,
    production_obs=None,
    backend="go",
    stats=None,
):
    """Build the explicit tiny-Cambia tree on the selected engine backend.

    backend="go" (default, cambia-1429): build_tree_go, the Go engine through the
    FFI bridge. No Python-engine import happens anywhere on this path.
    backend="python": build_tree_python, the original src.game.engine expansion.

    ``quiet`` and ``production_obs`` are python-backend knobs. On the Go backend
    there is nothing to quiet (the engine emits no per-node warnings) and the
    token stream is the Go tokenizer's own -- which IS the production observation
    path, and the only stream that backend can produce.

    ``production_obs`` therefore defaults to None, "unspecified", rather than
    False: the python backend maps None to False (its historical default) while
    the Go backend warns only for an EXPLICIT False, the case where a caller is
    deliberately asking for the pre-cambia-528/529 legacy stream and would
    otherwise get live tokens under a legacy label. Warning on the default instead
    would fire on every ordinary build and train readers to ignore it.
    """
    if backend == "python":
        return build_tree_python(
            cfg,
            n_deals,
            seed0,
            max_nodes_per_deal,
            enumerate_draws=enumerate_draws,
            perfect_recall=perfect_recall,
            tokenize=tokenize,
            seq_cap=seq_cap,
            quiet=quiet,
            exact_weights=exact_weights,
            production_obs=bool(production_obs),
        )
    if backend != "go":
        raise ValueError(f"unknown backend {backend!r}; expected 'go' or 'python'")
    if tokenize and production_obs is False:
        warnings.warn(
            "build_tree(backend='go', tokenize=True, production_obs=False): the Go "
            "tokenizer emits the live (production) observation stream, so the "
            f"returned seq_tokens are v{_go_tokenizer_version()} tokens, NOT the "
            "pre-cambia-528/529 legacy stream production_obs=False names. Score a "
            "legacy-provenance checkpoint with backend='python'.",
            stacklevel=2,
        )
    return build_tree_go(
        cfg,
        n_deals,
        seed0,
        max_nodes_per_deal,
        enumerate_draws=enumerate_draws,
        perfect_recall=perfect_recall,
        tokenize=tokenize,
        seq_cap=seq_cap,
        exact_weights=exact_weights,
        stats=stats,
    )


# ---- Tabular CFR+ over the explicit tree ----


class TabularCFR:
    def __init__(self, isets):
        # isets: dict iset -> num_actions
        self.regret = {k: np.zeros(n, dtype=np.float64) for k, n in isets.items()}
        self.strat_sum = {k: np.zeros(n, dtype=np.float64) for k, n in isets.items()}
        self.isets = isets

    def _strategy(self, iset):
        r = np.maximum(self.regret[iset], 0.0)
        s = r.sum()
        if s > 1e-12:
            return r / s
        n = len(r)
        return np.ones(n) / n

    def iterate(self, root, iter_idx):
        # CFR+ : two passes (one per traverser) per iteration over chance root.
        for traverser in (0, 1):
            self._cfr(root, traverser, 1.0, 1.0, 1.0, iter_idx)

    def _cfr(self, node, traverser, p0, p1, pc, t):
        if node.kind == "T":
            return node.util[traverser]
        if node.kind == "C":
            v = 0.0
            for child, w in zip(node.children, node.weights):
                v += w * self._cfr(child, traverser, p0, p1, pc * w, t)
            return v
        # decision: my own CFR keys tables by (pkey, nA) (well-defined under
        # variable action counts; production tabular policy uses bare pkey).
        iset = (node.pkey, len(node.actions))
        sigma = self._strategy(iset)
        nA = len(node.actions)
        util_a = np.empty(nA, dtype=np.float64)
        node_util = 0.0
        for i in range(nA):
            if node.player == 0:
                v = self._cfr(node.children[i], traverser, p0 * sigma[i], p1, pc, t)
            else:
                v = self._cfr(node.children[i], traverser, p0, p1 * sigma[i], pc, t)
            util_a[i] = v
            node_util += sigma[i] * v
        if node.player == traverser:
            # counterfactual reach of the OTHER player(s) * chance
            cf = (p1 * pc) if traverser == 0 else (p0 * pc)
            self.regret[iset] += cf * (util_a - node_util)
            # CFR+ : floor regrets at 0
            np.maximum(self.regret[iset], 0.0, out=self.regret[iset])
            # strategy-sum weighted by own reach and iteration (linear CFR+)
            own = p0 if traverser == 0 else p1
            self.strat_sum[iset] += t * own * sigma
        return node_util

    def average_strategy(self):
        avg = {}
        for k, ss in self.strat_sum.items():
            s = ss.sum()
            if s > 1e-12:
                avg[k] = ss / s
            else:
                n = len(ss)
                avg[k] = np.ones(n) / n
        return avg


# ---- Exact infoset best response / exploitability over the explicit tree ----
#
# Infosets are SHARED across many tree nodes (imperfect information), so a
# per-node max would let the BR adapt to hidden state it cannot observe and
# OVER-state exploitability (Jensen). The correct BR commits one action per
# infoset. We compute it by policy iteration on the single-player MDP induced
# by fixing the opponent's policy: repeat {accumulate counterfactual action
# values per BR infoset under the current BR action choice; set each BR action
# to the argmax} until the action map stops changing. On a finite tree this
# converges to the exact infoset best response (standard result).


def _br_eval(node, br_player, policy, br_actions, cfav):
    """Value to br_player given fixed BR action map `br_actions` (iset->idx).
    Accumulates per-infoset counterfactual action values into `cfav` along the
    way (weighted by chance*opponent reach, which is propagated implicitly by
    multiplying child values by opponent/chance probs as we descend)."""
    return _br_eval_rec(node, br_player, policy, br_actions, cfav, 1.0)


def _br_eval_rec(node, br_player, policy, br_actions, cfav, cfreach):
    if node.kind == "T":
        return node.util[br_player]
    if node.kind == "C":
        v = 0.0
        for c, w in zip(node.children, node.weights):
            v += w * _br_eval_rec(c, br_player, policy, br_actions, cfav, cfreach * w)
        return v
    if node.player == br_player:
        nA = len(node.actions)
        # BR commits one action per (infoset, nA): the bare infoset key can carry
        # multiple action counts, and the BR action index is only meaningful
        # within a fixed action set. Keying the commitment by (pkey, nA) keeps it
        # well-defined; this is the BR's information (it knows its own legal set).
        bkey = (node.pkey, nA)
        # value of each action (downstream uses current br_actions choices)
        vals = np.empty(nA)
        for i in range(nA):
            vals[i] = _br_eval_rec(
                node.children[i], br_player, policy, br_actions, cfav, cfreach
            )
        acc = cfav.get(bkey)
        if acc is None:
            acc = np.zeros(nA)
            cfav[bkey] = acc
        acc += cfreach * vals
        chosen = br_actions.get(bkey, 0)
        if chosen >= nA:
            chosen = 0
        return vals[chosen]
    # opponent node: weight children by opponent policy (this is part of cfreach)
    nA = len(node.actions)
    dist = _lookup(policy, node)
    v = 0.0
    for i in range(nA):
        p = dist[i]
        if p <= 0:
            continue
        v += p * _br_eval_rec(
            node.children[i], br_player, policy, br_actions, cfav, cfreach * p
        )
    return v


def _br_value(node, br_player, policy, max_sweeps=64):
    """Exact infoset best-response value for br_player vs `policy`."""
    br_actions = {}
    last_val = None
    for _ in range(max_sweeps):
        cfav = {}
        val = _br_eval(node, br_player, policy, br_actions, cfav)
        # greedy update of BR action map from accumulated counterfactual values
        changed = False
        for iset, acc in cfav.items():
            best = int(np.argmax(acc))
            if br_actions.get(iset) != best:
                br_actions[iset] = best
                changed = True
        if not changed and last_val is not None:
            return val
        last_val = val
    # final consistent evaluation
    cfav = {}
    return _br_eval(node, br_player, policy, br_actions, cfav)


def _lookup(pol, node):
    """Look up a policy dist for a node. Tries the compound key (pkey, nA) first
    (my own CFR's keying) then the bare pkey (production tabular keying). Returns
    a length-nA vector or uniform if missing/length-mismatched."""
    nA = len(node.actions)
    d = pol.get((node.pkey, nA))
    if d is not None and len(d) == nA:
        return d
    d = pol.get(node.pkey)
    if d is not None and len(d) == nA:
        return d
    return np.ones(nA) / nA


def _policy_value(node, policy_by_player, who):
    """Value for player `who` when BOTH players play their given policies."""
    if node.kind == "T":
        return node.util[who]
    if node.kind == "C":
        return sum(
            w * _policy_value(c, policy_by_player, who)
            for c, w in zip(node.children, node.weights)
        )
    pol = policy_by_player[node.player]
    nA = len(node.actions)
    dist = _lookup(pol, node)
    v = 0.0
    for i in range(nA):
        p = dist[i]
        if p <= 0:
            continue
        v += p * _policy_value(node.children[i], policy_by_player, who)
    return v


def exploitability(root, policy):
    """Sum over players of (BR value - on-policy value). Zero-sum => this is the
    standard exploitability (a.k.a. NashConv) in utility units (+/-1 scale)."""
    pol_by_player = {0: policy, 1: policy}
    onp0 = _policy_value(root, pol_by_player, 0)
    onp1 = _policy_value(root, pol_by_player, 1)
    br0 = _br_value(root, 0, policy)
    br1 = _br_value(root, 1, policy)
    nc = (br0 - onp0) + (br1 - onp1)
    return nc, (br0, br1, onp0, onp1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deals", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--max-nodes-per-deal", type=int, default=2_000_000)
    ap.add_argument(
        "--enumerate-draws",
        action="store_true",
        help="enumerate stockpile-draw chance (exact draws; large tree). "
        "Default off: draws follow realized deck order, covered by K deals.",
    )
    ap.add_argument(
        "--perfect-recall",
        action="store_true",
        help="X1 keystone: key the tabular policy by each player's "
        "genuine perfect-recall information state (initial peek "
        "+ own draws + public action/reveal sequence) instead of "
        "the production imperfect-recall belief abstraction. "
        "Tests whether perfect recall cures the NashConv plateau.",
    )
    ap.add_argument("--save-tree", type=str, default=None)
    ap.add_argument("--save-policy", type=str, default=None)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(
        f"[build] deals={args.deals} seed0={args.seed0} deck={cfg.cambia_rules.deck_ranks} "
        f"cpp={cfg.cambia_rules.cards_per_player} maxturns={cfg.cambia_rules.max_game_turns} "
        f"perfect_recall={args.perfect_recall}",
        flush=True,
    )
    t0 = time.time()
    root, isets, nnodes, aborted = build_tree(
        cfg,
        args.deals,
        args.seed0,
        args.max_nodes_per_deal,
        enumerate_draws=args.enumerate_draws,
        perfect_recall=args.perfect_recall,
    )
    print(
        f"[build] nodes~{nnodes} infosets={len(isets)} aborted_deals={aborted} "
        f"build_time={time.time()-t0:.1f}s",
        flush=True,
    )
    if aborted:
        print(
            "[build] WARNING: some deals hit max-nodes; tree truncated (raise cap or shrink game).",
            flush=True,
        )

    if args.save_tree:
        with open(args.save_tree, "wb") as f:
            pickle.dump({"root": root, "isets": isets}, f)
        print(f"[build] tree saved -> {args.save_tree}", flush=True)

    solver = TabularCFR(isets)
    tcfr = time.time()
    for it in range(1, args.iters + 1):
        solver.iterate(root, it)
        if it % args.eval_every == 0 or it == args.iters:
            avg = solver.average_strategy()
            nc, parts = exploitability(root, avg)
            print(
                f"[cfr] iter={it} exploitability(NashConv)={nc:.6e} "
                f"br=({parts[0]:.4f},{parts[1]:.4f}) onp=({parts[2]:.4f},{parts[3]:.4f}) "
                f"t={time.time()-tcfr:.0f}s",
                flush=True,
            )

    avg = solver.average_strategy()
    if args.save_policy:
        # store as plain dict for the adapter
        out = {repr(k): (list(k), v.tolist()) for k, v in avg.items()}
        with open(args.save_policy, "wb") as f:
            pickle.dump({"policy": {k: v for k, v in avg.items()}, "isets": isets}, f)
        print(f"[cfr] avg policy saved -> {args.save_policy}", flush=True)
    nc, parts = exploitability(root, avg)
    print(f"[done] final exploitability(NashConv)={nc:.6e}", flush=True)


if __name__ == "__main__":
    main()
