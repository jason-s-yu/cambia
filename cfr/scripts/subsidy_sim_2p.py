#!/usr/bin/env python3
"""cambia-1010 aggression-subsidy derivation: 2-player measurement, DATA ONLY.

Plays fixed-seed 2-player Cambia games between existing baseline heuristic
agents (src/agents/baseline_agents.py) using the Python reference engine
(src/game/engine.py), under prod-default house rules, and records raw
per-game outcomes. No win bonus, no false-Cambia penalty, no subsidies are
applied anywhere: the reference engine's CambiaGameState._calculate_final_scores
only ever produces +1/-1/tie utilities from the raw summed hand values plus
the RULES.md caller-tied-for-lowest-wins tie rule (cambia_caller_id in the
tied-lowest set wins outright) -- there is no bonus/penalty/subsidy knob to
disable.

House rules mirror src/cfr/prtcfr_worker.py::_default_production_house_rules(),
the codebase's canonical "production" CambiaRulesConfig used by PRT-CFR
self-play generation:
    allowDrawFromDiscardPile=True, allowReplaceAbilities=True,
    allowOpponentSnapping=True, max_game_turns=300, lockCallerHand=False.
All other CambiaRulesConfig fields are left at their pydantic defaults
(snapRace=False, penaltyDrawCount=2, use_jokers=2, cards_per_player=4,
initial_view_count=2, cambia_allowed_round=0, num_decks=1).

Configs (each >=20000 games, fixed per-game seeds via sha256):
    1. memory_heuristic vs memory_heuristic
    2. imperfect_greedy vs imperfect_greedy
    3. memory_heuristic vs imperfect_greedy

Seats alternate every game (game_index parity) so a heterogeneous matchup's
"caller identity split" is tracked by agent NAME, not by seat, canceling
first-mover seat bias -- matching the convention in
src/evaluate_agents.py::run_evaluation (seat_scheme="alternated").

Usage:
    python scripts/subsidy_sim_2p.py --smoke            # 100 games/config sanity gate
    python scripts/subsidy_sim_2p.py --games 20000 --workers 32

Output: scripts/subsidy_sim_2p_results.json (per-config aggregates only; raw
per-game tuples are not persisted, to keep the JSON small -- every requested
aggregate is computed from them before they are dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Engine/agent modules log every penalty draw and invalid-action retry at
# WARNING/INFO via the root logger's handler-of-last-resort -- expected agent
# behavior (baseline heuristics sometimes probe illegal actions and eat a
# penalty), not a script bug. Silenced up front so 60k games/config don't
# flood stdout or pay logging overhead; set before the Pool forks so workers
# inherit it.
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.ERROR)

# --- cambia-240 uv-sync-active-trap guard -----------------------------------
# The pyenv "cfr" venv's editable install (.pth) resolves the `src` package to
# the MAIN repo checkout, not this worktree, when PYTHONPATH isn't pinned.
# Fail loudly instead of silently measuring the wrong tree.
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

import src.game.engine as _engine_mod  # noqa: E402

_imported_from = os.path.abspath(_engine_mod.__file__)
_expected_prefix = os.path.abspath(_WORKTREE_ROOT)
if not _imported_from.startswith(_expected_prefix):
    raise RuntimeError(
        "uv-sync-active-trap: src.game.engine imported from "
        f"{_imported_from!r}, expected under worktree {_expected_prefix!r}. "
        "PYTHONPATH is not pinned to this worktree -- refusing to measure "
        "against the wrong source tree."
    )

from src.config import Config, CambiaRulesConfig  # noqa: E402
from src.game.engine import CambiaGameState  # noqa: E402
from src.agents.baseline_agents import (  # noqa: E402
    BaseAgent,
    MemoryHeuristicAgent,
    ImperfectGreedyAgent,
)
from src.constants import ActionCallCambia  # noqa: E402
from src.cfr.exceptions import GameStateError, AgentStateError  # noqa: E402


AGENT_CLASSES = {
    "memory_heuristic": MemoryHeuristicAgent,
    "imperfect_greedy": ImperfectGreedyAgent,
}

CONFIGS: List[Tuple[str, str, str]] = [
    ("memory_heuristic_vs_memory_heuristic", "memory_heuristic", "memory_heuristic"),
    ("imperfect_greedy_vs_imperfect_greedy", "imperfect_greedy", "imperfect_greedy"),
    ("memory_heuristic_vs_imperfect_greedy", "memory_heuristic", "imperfect_greedy"),
]

PERCENTILES = [10, 25, 50, 75, 90]


def build_house_rules() -> CambiaRulesConfig:
    """Mirrors src/cfr/prtcfr_worker.py::_default_production_house_rules()."""
    return CambiaRulesConfig(
        allowDrawFromDiscardPile=True,
        allowReplaceAbilities=True,
        allowOpponentSnapping=True,
        max_game_turns=300,
        lockCallerHand=False,
    )


def build_config() -> Config:
    return Config(cambia_rules=build_house_rules())


def deck_seed_for(config_name: str, game_index: int) -> int:
    """Deterministic per-game deck seed, fixed across reruns."""
    key = f"cambia-1010-subsidy2p|{config_name}|{game_index}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class GameRecord:
    game_index: int
    deck_seed: int
    # names of the agent occupying seat 0 / seat 1 this game (post-alternation)
    agent0_name: str
    agent1_name: str
    caller_seat: Optional[int]  # None, 0, or 1
    call_turn_engine: Optional[int]  # game_state._turn_number at the call
    call_action_index: Optional[int]  # action-loop counter at the call
    score0: float
    score1: float
    winner_seat: Optional[int]  # None (true tie, no-call only), 0, or 1
    placement0: int  # 1 or 2 (1 shared on a true tie)
    placement1: int
    turns_engine: int  # final game_state._turn_number
    turns_actions: int  # final action-loop counter
    error: bool


def play_one_game(
    config_name: str,
    game_index: int,
    agent_a_name: str,
    agent_b_name: str,
    engine_config: Config,
) -> GameRecord:
    """Plays one game. Seats alternate by game_index parity: agent_a occupies
    seat 0 on odd game_index (1-based), seat 1 on even game_index."""
    seed = deck_seed_for(config_name, game_index)
    a_is_seat0 = (game_index % 2 == 1)
    seat0_name = agent_a_name if a_is_seat0 else agent_b_name
    seat1_name = agent_b_name if a_is_seat0 else agent_a_name

    game_state = CambiaGameState(house_rules=engine_config.cambia_rules, seed=seed)
    agents: List[BaseAgent] = [
        AGENT_CLASSES[seat0_name](player_id=0, config=engine_config),
        AGENT_CLASSES[seat1_name](player_id=1, config=engine_config),
    ]

    max_turns = engine_config.cambia_rules.max_game_turns
    action_cap = (max_turns if max_turns > 0 else 300) * 16

    caller_seat: Optional[int] = None
    call_turn_engine: Optional[int] = None
    call_action_index: Optional[int] = None
    error = False
    action_idx = 0

    try:
        while not game_state.is_terminal() and action_idx < action_cap:
            action_idx += 1
            acting_player_id = game_state.get_acting_player()
            if acting_player_id == -1:
                error = True
                break

            current_agent = agents[acting_player_id]
            legal_actions = game_state.get_legal_actions()
            if not legal_actions:
                if game_state.is_terminal():
                    break
                error = True
                break

            chosen_action = current_agent.choose_action(game_state, legal_actions)

            if isinstance(chosen_action, ActionCallCambia) and caller_seat is None:
                caller_seat = acting_player_id
                call_turn_engine = game_state._turn_number
                call_action_index = action_idx

            _delta, undo_info = game_state.apply_action(chosen_action)
            if not callable(undo_info):
                error = True
                break
    except (GameStateError, AgentStateError):
        error = True
    except Exception:  # JUSTIFIED: measurement resilience, flagged via `error`
        error = True

    if not game_state.is_terminal():
        error = True

    score0 = score1 = float("nan")
    winner_seat: Optional[int] = None
    placement0 = placement1 = 2
    try:
        hands = [game_state.get_player_hand(i) for i in range(2)]
        score0 = float(sum(c.value for c in hands[0]))
        score1 = float(sum(c.value for c in hands[1]))
    except Exception:
        error = True

    try:
        winner_seat = game_state._winner
        utilities = game_state._utilities
        placement0 = 1 if utilities[0] >= 0.0 else 2
        placement1 = 1 if utilities[1] >= 0.0 else 2
    except Exception:
        error = True

    return GameRecord(
        game_index=game_index,
        deck_seed=seed,
        agent0_name=seat0_name,
        agent1_name=seat1_name,
        caller_seat=caller_seat,
        call_turn_engine=call_turn_engine,
        call_action_index=call_action_index,
        score0=score0,
        score1=score1,
        winner_seat=winner_seat,
        placement0=placement0,
        placement1=placement1,
        turns_engine=game_state._turn_number,
        turns_actions=action_idx,
        error=error,
    )


def _worker_chunk(
    args: Tuple[str, str, str, int, int, Config]
) -> List[GameRecord]:
    config_name, agent_a_name, agent_b_name, start, end, engine_config = args
    return [
        play_one_game(config_name, i, agent_a_name, agent_b_name, engine_config)
        for i in range(start, end + 1)
    ]


def run_config(
    config_name: str,
    agent_a_name: str,
    agent_b_name: str,
    num_games: int,
    workers: int,
    engine_config: Config,
) -> List[GameRecord]:
    game_indices = list(range(1, num_games + 1))
    if workers <= 1:
        return [
            play_one_game(config_name, i, agent_a_name, agent_b_name, engine_config)
            for i in game_indices
        ]

    chunk_size = max(1, -(-num_games // workers))  # ceil
    chunks = []
    for start in range(1, num_games + 1, chunk_size):
        end = min(start + chunk_size - 1, num_games)
        chunks.append((config_name, agent_a_name, agent_b_name, start, end, engine_config))

    records: List[GameRecord] = []
    with mp.Pool(processes=workers) as pool:
        for chunk_records in pool.imap_unordered(_worker_chunk, chunks):
            records.extend(chunk_records)
    records.sort(key=lambda r: r.game_index)
    return records


# --- Aggregation --------------------------------------------------------


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _sd(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var**0.5


def _percentile(xs_sorted: List[float], pct: float) -> Optional[float]:
    if not xs_sorted:
        return None
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    k = (len(xs_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


def _percentile_summary(xs: List[float]) -> Dict[str, Optional[float]]:
    xs_sorted = sorted(xs)
    return {f"p{p}": _percentile(xs_sorted, p) for p in PERCENTILES}


def aggregate_config(
    config_name: str, agent_a_name: str, agent_b_name: str, records: List[GameRecord]
) -> Dict:
    clean = [r for r in records if not r.error]
    n_errors = len(records) - len(clean)

    called = [r for r in clean if r.caller_seat is not None]
    no_call = [r for r in clean if r.caller_seat is None]

    n = len(clean)
    n_called = len(called)
    n_no_call = len(no_call)

    # Caller-relative series (only defined for `called` games).
    caller_scores: List[float] = []
    opp_scores: List[float] = []
    gaps: List[float] = []  # caller_score - opponent_score
    caller_won = 0
    caller_lost = 0
    exact_tie_called = 0
    gap_given_won: List[float] = []
    gap_given_lost: List[float] = []
    caller_name_counts: Dict[str, int] = {agent_a_name: 0, agent_b_name: 0}
    call_turns_engine: List[float] = []
    call_turns_action: List[float] = []

    for r in called:
        caller_score = r.score0 if r.caller_seat == 0 else r.score1
        opp_score = r.score1 if r.caller_seat == 0 else r.score0
        caller_scores.append(caller_score)
        opp_scores.append(opp_score)
        gap = caller_score - opp_score
        gaps.append(gap)
        caller_name = r.agent0_name if r.caller_seat == 0 else r.agent1_name
        caller_name_counts[caller_name] = caller_name_counts.get(caller_name, 0) + 1
        if r.call_turn_engine is not None:
            call_turns_engine.append(float(r.call_turn_engine))
        if r.call_action_index is not None:
            call_turns_action.append(float(r.call_action_index))
        if caller_score == opp_score:
            exact_tie_called += 1
        if r.winner_seat == r.caller_seat:
            caller_won += 1
            gap_given_won.append(gap)
        else:
            caller_lost += 1
            gap_given_lost.append(gap)

    abs_gaps = [abs(g) for g in gaps]

    # No-call control.
    no_call_scores0 = [r.score0 for r in no_call]
    no_call_scores1 = [r.score1 for r in no_call]
    no_call_exact_tie = sum(1 for r in no_call if r.score0 == r.score1)
    no_call_seat0_wins = sum(1 for r in no_call if r.winner_seat == 0)
    no_call_seat1_wins = sum(1 for r in no_call if r.winner_seat == 1)
    no_call_true_ties = sum(1 for r in no_call if r.winner_seat is None)
    no_call_agent_name_wins: Dict[str, int] = {agent_a_name: 0, agent_b_name: 0}
    for r in no_call:
        if r.winner_seat is None:
            continue
        winner_name = r.agent0_name if r.winner_seat == 0 else r.agent1_name
        no_call_agent_name_wins[winner_name] = (
            no_call_agent_name_wins.get(winner_name, 0) + 1
        )

    all_turns_engine = [float(r.turns_engine) for r in clean]
    all_turns_actions = [float(r.turns_actions) for r in clean]

    def gap_threshold_prob(threshold: int) -> Optional[float]:
        if not abs_gaps:
            return None
        return sum(1 for g in abs_gaps if g <= threshold) / len(abs_gaps)

    return {
        "config_name": config_name,
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "games_requested": len(records),
        "games_played_clean": n,
        "games_error": n_errors,
        "n_called": n_called,
        "n_no_call": n_no_call,
        "p_call": (n_called / n) if n else None,
        "p_no_call": (n_no_call / n) if n else None,
        "caller_identity_split": {
            "counts_by_agent_name": caller_name_counts,
            "fractions_by_agent_name": (
                {k: v / n_called for k, v in caller_name_counts.items()}
                if n_called
                else {}
            ),
        },
        "called_games": {
            "p_caller_wins_given_called": (caller_won / n_called) if n_called else None,
            "p_caller_loses_given_called": (
                (caller_lost / n_called) if n_called else None
            ),
            "p_exact_tie_given_called": (
                (exact_tie_called / n_called) if n_called else None
            ),
            "note": (
                "In 2-player games the RULES.md caller-tied-for-lowest-wins rule "
                "means any exact tie at the minimum score is always won by the "
                "caller (both players are trivially in the tied-lowest set), so "
                "p_exact_tie_given_called games are a subset of "
                "p_caller_wins_given_called, not a third mutually exclusive "
                "outcome. Reported per spec regardless."
            ),
            "caller_score": {"mean": _mean(caller_scores), "sd": _sd(caller_scores)},
            "opponent_score": {"mean": _mean(opp_scores), "sd": _sd(opp_scores)},
            "gap_caller_minus_opponent_given_won": {
                "mean": _mean(gap_given_won),
                "sd": _sd(gap_given_won),
                "n": len(gap_given_won),
            },
            "gap_caller_minus_opponent_given_lost": {
                "mean": _mean(gap_given_lost),
                "sd": _sd(gap_given_lost),
                "n": len(gap_given_lost),
            },
            "abs_gap_percentiles": _percentile_summary(abs_gaps),
            "p_abs_gap_le_1": gap_threshold_prob(1),
            "p_abs_gap_le_2": gap_threshold_prob(2),
            "p_abs_gap_le_3": gap_threshold_prob(3),
            "p_abs_gap_le_5": gap_threshold_prob(5),
            "call_turn_engine": {
                "mean": _mean(call_turns_engine),
                "sd": _sd(call_turns_engine),
                "percentiles": _percentile_summary(call_turns_engine),
            },
            "call_turn_action_index": {
                "mean": _mean(call_turns_action),
                "sd": _sd(call_turns_action),
                "percentiles": _percentile_summary(call_turns_action),
            },
        },
        "no_call_control": {
            "description": (
                "Games where cambia_caller_id stayed None at termination "
                "(engine max_game_turns cap or stalemate; no ActionCallCambia "
                "was ever legal+chosen)."
            ),
            "n": n_no_call,
            "p_seat0_wins": (no_call_seat0_wins / n_no_call) if n_no_call else None,
            "p_seat1_wins": (no_call_seat1_wins / n_no_call) if n_no_call else None,
            "p_true_tie": (no_call_true_ties / n_no_call) if n_no_call else None,
            "p_exact_score_tie": (
                (no_call_exact_tie / n_no_call) if n_no_call else None
            ),
            "wins_by_agent_name": no_call_agent_name_wins,
            "score_seat0": {"mean": _mean(no_call_scores0), "sd": _sd(no_call_scores0)},
            "score_seat1": {"mean": _mean(no_call_scores1), "sd": _sd(no_call_scores1)},
        },
        "turns": {
            "mean_turns_engine": _mean(all_turns_engine),
            "sd_turns_engine": _sd(all_turns_engine),
            "mean_turns_action_index": _mean(all_turns_actions),
            "sd_turns_action_index": _sd(all_turns_actions),
        },
    }


def sanity_check(records: List[GameRecord], config_name: str) -> List[str]:
    """Smoke-gate assertions. Returns a list of violation strings (empty = pass)."""
    problems: List[str] = []
    clean = [r for r in records if not r.error]
    if not clean:
        problems.append(f"[{config_name}] all {len(records)} games errored")
        return problems

    n_called = sum(1 for r in clean if r.caller_seat is not None)
    if n_called == 0:
        problems.append(f"[{config_name}] no caller ever recorded across {len(clean)} games")

    for r in clean:
        if r.caller_seat is not None:
            if r.caller_seat not in (0, 1):
                problems.append(f"[{config_name}] game {r.game_index}: bad caller_seat {r.caller_seat}")
            if r.call_turn_engine is None:
                problems.append(f"[{config_name}] game {r.game_index}: caller recorded but no call_turn_engine")
            caller_score = r.score0 if r.caller_seat == 0 else r.score1
            opp_score = r.score1 if r.caller_seat == 0 else r.score0
            # RULES.md tie rule: caller among tied-lowest always wins in 2p.
            if caller_score <= opp_score and r.winner_seat != r.caller_seat:
                problems.append(
                    f"[{config_name}] game {r.game_index}: caller_score={caller_score} <= "
                    f"opp_score={opp_score} but winner_seat={r.winner_seat} != caller_seat={r.caller_seat} "
                    "(caller-tie rule violated)"
                )
            if caller_score > opp_score and r.winner_seat == r.caller_seat:
                problems.append(
                    f"[{config_name}] game {r.game_index}: caller_score={caller_score} > "
                    f"opp_score={opp_score} but winner_seat==caller_seat (caller-tie rule violated)"
                )
        # placement/winner consistency
        if r.winner_seat == 0 and not (r.placement0 == 1 and r.placement1 == 2):
            problems.append(f"[{config_name}] game {r.game_index}: winner_seat=0 but placements={r.placement0},{r.placement1}")
        if r.winner_seat == 1 and not (r.placement1 == 1 and r.placement0 == 2):
            problems.append(f"[{config_name}] game {r.game_index}: winner_seat=1 but placements={r.placement0},{r.placement1}")
        if r.winner_seat is None and not (r.placement0 == 1 and r.placement1 == 1):
            problems.append(f"[{config_name}] game {r.game_index}: winner_seat=None but placements={r.placement0},{r.placement1} (expected tied 1/1)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20000, help="games per config")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--smoke", action="store_true", help="100-game smoke run per config, sanity gate only")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_WORKTREE_ROOT, "scripts", "subsidy_sim_2p_results.json"),
    )
    args = parser.parse_args()

    num_games = 100 if args.smoke else args.games
    engine_config = build_config()

    print(
        f"[subsidy_sim_2p] games/config={num_games} workers={args.workers} "
        f"smoke={args.smoke} src={_imported_from}",
        flush=True,
    )

    overall_start = time.time()
    results = []
    for config_name, agent_a, agent_b in CONFIGS:
        t0 = time.time()
        records = run_config(config_name, agent_a, agent_b, num_games, args.workers, engine_config)
        dt = time.time() - t0
        print(
            f"[subsidy_sim_2p] {config_name}: {len(records)} games in {dt:.1f}s "
            f"({len(records) / dt:.1f} games/s)",
            flush=True,
        )

        problems = sanity_check(records, config_name)
        if problems:
            print(f"[subsidy_sim_2p] SANITY CHECK FAILURES for {config_name}:", flush=True)
            for p in problems[:25]:
                print(f"  - {p}", flush=True)
            if len(problems) > 25:
                print(f"  ... and {len(problems) - 25} more", flush=True)
            if args.smoke:
                return 1

        agg = aggregate_config(config_name, agent_a, agent_b, records)
        agg["runtime_seconds"] = dt
        agg["sanity_check_violations"] = len(problems)
        results.append(agg)

    total_dt = time.time() - overall_start
    output = {
        "meta": {
            "ticket": "cambia-1010",
            "leg": "2-player",
            "purpose": "aggression-subsidy derivation measurement data (data only, no conclusions)",
            "smoke": args.smoke,
            "games_per_config": num_games,
            "workers": args.workers,
            "total_runtime_seconds": total_dt,
            "engine_source": _imported_from,
            "house_rules": {
                k: v
                for k, v in engine_config.cambia_rules.model_dump().items()
            },
            "scoring_note": (
                "Raw engine scoring only: CambiaGameState._calculate_final_scores "
                "produces +1/-1/tie utilities from summed raw hand card values "
                "(RULES.md: Red King=-1, Joker=0, Black King=13, others face "
                "value) with the caller-tied-for-lowest-wins tie rule. The "
                "engine has no win-bonus, false-Cambia-penalty, or subsidy knobs."
            ),
            "percentiles_reported": PERCENTILES,
        },
        "configs": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[subsidy_sim_2p] wrote {args.out} in {total_dt:.1f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
