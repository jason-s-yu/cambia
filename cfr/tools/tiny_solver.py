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
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

from src.config import load_config
from src.game.engine import CambiaGameState
from src.agent_state import AgentState
from src.analysis_tools import AnalysisTools
from src.constants import NUM_PLAYERS, ActionDrawStockpile
from src.utils import InfosetKey

# Suppress the engine's chatty per-node warnings during full-tree expansion.
import logging as _logging
_logging.getLogger("src.game").setLevel(_logging.CRITICAL)
_logging.getLogger("src.agent_state").setLevel(_logging.CRITICAL)
_logging.getLogger("src.game.engine").setLevel(_logging.CRITICAL)
for _n in list(_logging.root.manager.loggerDict):
    if _n.startswith("src."):
        _logging.getLogger(_n).setLevel(_logging.CRITICAL)


# ---- Node types (lightweight; built once, traversed many times) ----

class Terminal:
    __slots__ = ("util",)
    kind = "T"
    def __init__(self, util):
        self.util = util  # tuple (u0, u1)

class Chance:
    __slots__ = ("children", "weights")
    kind = "C"
    def __init__(self):
        self.children = []
        self.weights = []

class Decision:
    __slots__ = ("player", "iset", "pkey", "actions", "children")
    kind = "D"
    def __init__(self, player, iset, pkey, actions):
        self.player = player
        self.iset = iset            # production infoset key (get_infoset_key + ctx)
        # policy key = (iset, num_actions). The production infoset key does NOT
        # determine the legal-action count (~8% of keys / 37% of visits vary);
        # production CFR silently resets regret/strategy vectors on mismatch.
        # Keying the tabular policy by (iset, nA) makes it well-defined (a strict
        # refinement of production's partition); DESCA sidesteps this via its
        # fixed-32 abstraction. We report divergence on `iset` separately.
        self.pkey = pkey
        self.actions = actions      # list of GameAction (sorted)
        self.children = []          # one node per action


def _mk_agent(game, pid, opp, cfg, init_obs):
    a = AgentState(
        player_id=pid, opponent_id=opp,
        memory_level=cfg.agent_params.memory_level,
        time_decay_turns=cfg.agent_params.time_decay_turns,
        initial_hand_size=len(game.players[pid].hand),
        config=cfg,
    )
    a.initialize(init_obs, game.players[pid].hand, game.players[pid].initial_peek_indices)
    return a


def _advance(game, action, acting, ag):
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
    def __init__(self, cfg, max_nodes, enumerate_draws=True):
        self.cfg = cfg
        self.max_nodes = max_nodes
        self.enumerate_draws = enumerate_draws
        self.n = 0
        self.aborted = False
        self.iset_actions = {}     # iset -> num actions (consistency check)
        self.iset_count = 0

    def build_decision_or_terminal(self, game, ag, depth):
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
        pkey = iset.astuple()
        # iset_actions is keyed by (pkey, nA) so my own exact CFR table and the
        # uniform null control are well-defined under variable action counts.
        # The production tabular policy is looked up by the BARE node.pkey.
        self.iset_actions[(pkey, nA)] = nA
        node = Decision(acting, iset, pkey, legal)

        for action in legal:
            if self.enumerate_draws and isinstance(action, ActionDrawStockpile) and game.stockpile:
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
        distinct = {}
        for c in game.stockpile:
            k = (c.rank, c.suit)
            distinct[k] = distinct.get(k, 0) + 1
        total = len(game.stockpile)
        orig = list(game.stockpile)
        for (rank, suit), cnt in distinct.items():
            idx = next(i for i, c in enumerate(game.stockpile) if (c.rank, c.suit) == (rank, suit))
            card = game.stockpile.pop(idx)
            game.stockpile.append(card)
            child = self._apply_one(game, action, acting, ag, depth)
            game.stockpile[:] = list(orig)
            ch.children.append(child)
            ch.weights.append(cnt / total)
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
        child = self.build_decision_or_terminal(game, new_ag, depth + 1)
        try:
            undo()
        except Exception:
            pass
        return child


def build_tree(cfg, n_deals, seed0, max_nodes_per_deal, enumerate_draws=True):
    """Synthetic root: K deals, each weight 1/K; each is a full chance-tree."""
    root = Chance()
    all_isets = {}
    total_nodes = 0
    aborted_deals = 0
    for d in range(n_deals):
        b = Builder(cfg, max_nodes_per_deal, enumerate_draws=enumerate_draws)
        game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(seed0 + d))
        init_obs = AnalysisTools._create_observation_for_br(game, None, -1)
        ag = {0: _mk_agent(game, 0, 1, cfg, init_obs), 1: _mk_agent(game, 1, 0, cfg, init_obs)}
        sub = b.build_decision_or_terminal(game, ag, 0)
        root.children.append(sub)
        root.weights.append(1.0)  # normalized below
        total_nodes += b.n
        if b.aborted:
            aborted_deals += 1
        for k, v in b.iset_actions.items():
            all_isets[k] = v
    s = sum(root.weights)
    root.weights = [w / s for w in root.weights]
    return root, all_isets, total_nodes, aborted_deals


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
            vals[i] = _br_eval_rec(node.children[i], br_player, policy, br_actions, cfav, cfreach)
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
        v += p * _br_eval_rec(node.children[i], br_player, policy, br_actions, cfav, cfreach * p)
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
        return sum(w * _policy_value(c, policy_by_player, who) for c, w in zip(node.children, node.weights))
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
    ap.add_argument("--enumerate-draws", action="store_true",
                    help="enumerate stockpile-draw chance (exact draws; large tree). "
                         "Default off: draws follow realized deck order, covered by K deals.")
    ap.add_argument("--save-tree", type=str, default=None)
    ap.add_argument("--save-policy", type=str, default=None)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[build] deals={args.deals} seed0={args.seed0} deck={cfg.cambia_rules.deck_ranks} "
          f"cpp={cfg.cambia_rules.cards_per_player} maxturns={cfg.cambia_rules.max_game_turns}", flush=True)
    t0 = time.time()
    root, isets, nnodes, aborted = build_tree(
        cfg, args.deals, args.seed0, args.max_nodes_per_deal,
        enumerate_draws=args.enumerate_draws)
    print(f"[build] nodes~{nnodes} infosets={len(isets)} aborted_deals={aborted} "
          f"build_time={time.time()-t0:.1f}s", flush=True)
    if aborted:
        print("[build] WARNING: some deals hit max-nodes; tree truncated (raise cap or shrink game).", flush=True)

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
            print(f"[cfr] iter={it} exploitability(NashConv)={nc:.6e} "
                  f"br=({parts[0]:.4f},{parts[1]:.4f}) onp=({parts[2]:.4f},{parts[3]:.4f}) "
                  f"t={time.time()-tcfr:.0f}s", flush=True)

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
