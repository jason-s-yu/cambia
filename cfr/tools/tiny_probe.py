"""Tree-size probe for tiny-Cambia (research E1).

Measures the realized game tree of a reduced-deck Cambia variant by recursively
expanding the Python CambiaGameState. Stockpile draws are treated two ways:

  --chance: enumerate distinct drawable cards at each draw (proper chance node),
            weighted by multiplicity. This is the true game-tree size.
  default : follow the engine's pre-shuffled deck (one realized line per draw).
            This measures the per-deal decision-tree size for ONE deck order.

The initial deal is whatever the constructor deals (one realized deal); to bound
the full chance-root cost, pass --deals N to average node counts over N random
deals. Decision nodes report their infoset key via the production agent-state
machinery (analysis_tools helpers), so the count of DISTINCT infosets equals the
tabular CFR table size on this game.

Usage:
  python tools/tiny_probe.py --config config/tiny_cambia_tabular.yaml --chance --deals 3
"""

import argparse
import sys
import time
from collections import defaultdict

from src.config import load_config
from src.game.engine import CambiaGameState
from src.agent_state import AgentState
from src.analysis_tools import AnalysisTools
from src.constants import NUM_PLAYERS
from src.utils import InfosetKey


def _mk_agent(game, pid, opp, cfg, init_obs):
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


def _advance_agents(game, action, acting, ag):
    """Clone+update both agent views after `action` applied by `acting`."""
    obs = AnalysisTools._create_observation_for_br(game, action, acting)
    if obs is None:
        return None
    new = {}
    for pid, a in ag.items():
        na = a.clone()
        filt = AnalysisTools._filter_observation_for_br(obs, pid)
        na.update(filt)
        new[pid] = na
    return new


class Probe:
    def __init__(self, cfg, enumerate_chance, max_nodes):
        self.cfg = cfg
        self.enumerate_chance = enumerate_chance
        self.max_nodes = max_nodes
        self.nodes = 0
        self.terminals = 0
        self.decision_nodes = 0
        self.chance_nodes = 0
        self.max_depth = 0
        self.infosets = set()
        self.ctx_counts = defaultdict(int)
        self.aborted = False

    def run(self, game, ag, depth=0):
        if self.aborted:
            return
        self.nodes += 1
        if self.nodes > self.max_nodes:
            self.aborted = True
            return
        if depth > self.max_depth:
            self.max_depth = depth
        if game.is_terminal():
            self.terminals += 1
            return
        acting = game.get_acting_player()
        if acting == -1:
            return
        legal = sorted(list(game.get_legal_actions()), key=repr)
        if not legal:
            self.terminals += 1
            return

        ctx = AnalysisTools._get_decision_context(game)
        # Infoset key from acting player's own view (the production partition).
        try:
            base = ag[acting].get_infoset_key()
            self.infosets.add(InfosetKey(*base, ctx.value if ctx else -1))
            self.ctx_counts[ctx.name if ctx else "NONE"] += 1
        except Exception:
            pass
        self.decision_nodes += 1

        # Determine whether the NEXT transition for a draw should branch on chance.
        # We expand all decision branches. For ActionDrawStockpile we optionally
        # enumerate distinct top-of-stockpile cards.
        from src.constants import ActionDrawStockpile

        for action in legal:
            if (
                self.enumerate_chance
                and isinstance(action, ActionDrawStockpile)
                and game.stockpile
            ):
                # Enumerate distinct drawable cards (by identity rank/suit), weighted by count.
                # We mutate the stockpile top to each distinct card, recurse, then restore.
                self.chance_nodes += 1
                distinct = {}
                for c in game.stockpile:
                    key = (c.rank, c.suit)
                    distinct[key] = distinct.get(key, 0) + 1
                orig_stock = list(game.stockpile)
                for (rank, suit), _cnt in distinct.items():
                    # find an index of that card, move it to top (-1)
                    idx = next(
                        i
                        for i, c in enumerate(game.stockpile)
                        if (c.rank, c.suit) == (rank, suit)
                    )
                    card = game.stockpile.pop(idx)
                    game.stockpile.append(card)
                    self._apply_and_recurse(game, action, acting, ag, depth)
                    game.stockpile[:] = list(orig_stock)
                    if self.aborted:
                        return
            else:
                self._apply_and_recurse(game, action, acting, ag, depth)
                if self.aborted:
                    return

    def _apply_and_recurse(self, game, action, acting, ag, depth):
        state_delta, undo = game.apply_action(action)
        if not callable(undo):
            return
        new_ag = _advance_agents(game, action, acting, ag)
        if new_ag is None:
            try:
                undo()
            except Exception:
                pass
            return
        self.run(game, new_ag, depth + 1)
        try:
            undo()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chance", action="store_true", help="enumerate draw chance nodes")
    ap.add_argument(
        "--deals", type=int, default=1, help="number of random initial deals to probe"
    )
    ap.add_argument("--max-nodes", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(
        f"deck_ranks={cfg.cambia_rules.deck_ranks} cards_per_player={cfg.cambia_rules.cards_per_player} "
        f"use_jokers={cfg.cambia_rules.use_jokers} initial_view={cfg.cambia_rules.initial_view_count} "
        f"max_game_turns={cfg.cambia_rules.max_game_turns} cambia_allowed_round={cfg.cambia_rules.cambia_allowed_round}",
        flush=True,
    )

    import random

    agg_infosets = set()
    for d in range(args.deals):
        t0 = time.time()
        # Seeded rng -> reproducible deal (dataclass field consumed by __post_init__).
        game = CambiaGameState(
            house_rules=cfg.cambia_rules, _rng=random.Random(args.seed + d)
        )

        init_obs = AnalysisTools._create_observation_for_br(game, None, -1)
        ag = {
            0: _mk_agent(game, 0, 1, cfg, init_obs),
            1: _mk_agent(game, 1, 0, cfg, init_obs),
        }
        p = Probe(cfg, args.chance, args.max_nodes)
        p.run(game, ag, 0)
        agg_infosets |= p.infosets
        dt = time.time() - t0
        status = "ABORTED(>max_nodes)" if p.aborted else "complete"
        print(
            f"deal {d}: nodes={p.nodes} decision={p.decision_nodes} chance={p.chance_nodes} "
            f"terminals={p.terminals} max_depth={p.max_depth} infosets={len(p.infosets)} "
            f"[{status}] {dt:.2f}s ctx={dict(p.ctx_counts)}",
            flush=True,
        )
        sys.stdout.flush()

    print(f"UNION infosets across {args.deals} deals: {len(agg_infosets)}", flush=True)


if __name__ == "__main__":
    main()
