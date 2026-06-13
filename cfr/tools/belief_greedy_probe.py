"""
tools/belief_greedy_probe.py

Phase 0 measurement-repair probe: RC-D eval-only share (U9).

Compares DESCA iter-1000 under two unabstraction strategies:
  - uniform: existing random selection within abstract class (baseline)
  - belief_greedy: pick the concrete action within the chosen class that
    ranks highest under the agent's current belief state

CRN-paired: both conditions see the same deals (same per-game RNG seed),
so the delta is a direct within-pair estimate of eval-time RC-D cost.

Usage:
    cd cfr
    python -m tools.belief_greedy_probe \\
        --checkpoint runs/desca-phase1-apcfr-mild/checkpoints/desca_checkpoint_iter_1000.pt \\
        --config config/desca_phase1_apcfr_mild.yaml \\
        --num-games 200 \\
        --baselines imperfect_greedy memory_heuristic

    # Full mean_imp run (5 baselines x 200 CRN pairs):
    python -m tools.belief_greedy_probe \\
        --checkpoint runs/desca-phase1-apcfr-mild/checkpoints/desca_checkpoint_iter_1000.pt \\
        --config config/desca_phase1_apcfr_mild.yaml \\
        --num-games 200

Notes:
  - Does NOT modify training code or evaluate_agents.py.
  - BeliefGreedyDESCAWrapper is a thin subclass of DESCAAgentWrapper that
    overrides only the unabstract step.
  - Belief-greedy value function uses agent.agent_state bucket beliefs to
    rank concrete candidates; falls back to uniform when belief is absent
    or uninformative.
  - CRN seeding: each game i runs with RNG seed BASE_SEED + i for both
    conditions, guaranteeing identical deals.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from collections import Counter
from typing import List, Optional, Sequence, Set

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow running as "python -m tools.belief_greedy_probe" from cfr/
# ---------------------------------------------------------------------------
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CFR_ROOT = os.path.dirname(_SCRIPT_DIR)
if _CFR_ROOT not in sys.path:
    sys.path.insert(0, _CFR_ROOT)

from src.config import load_config
from src.constants import (
    CardBucket,
    GameAction,
)
from src.evaluate_agents import (
    DESCAAgentWrapper,
    MEAN_IMP_BASELINES,
    NeuralAgentWrapper,
    get_agent,
)
from src.game.engine import CambiaGameState
from src.agent_state import AgentObservation
from src.action_abstraction import unabstract, _concrete_to_abstract  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket value map (lower = better card to keep; higher = better to replace)
# ---------------------------------------------------------------------------

# Estimated expected point value per bucket, used to rank which concrete
# replacement target is worst in hand (highest = most desirable to replace).
# NEG_KING has negative value (-1) so it's the best card; ZERO (Joker) = 0.
_BUCKET_EXPECTED_VALUE = {
    CardBucket.NEG_KING.value: -1.0,
    CardBucket.ZERO.value: 0.0,
    CardBucket.ACE.value: 1.0,
    CardBucket.LOW_NUM.value: 3.0,    # midpoint of 2-4
    CardBucket.MID_NUM.value: 5.5,    # midpoint of 5-6
    CardBucket.PEEK_SELF.value: 7.5,  # 7-8
    CardBucket.PEEK_OTHER.value: 9.5, # 9-10
    CardBucket.SWAP_BLIND.value: 11.5, # J-Q
    CardBucket.HIGH_KING.value: 13.0,
    CardBucket.UNKNOWN.value: 6.5,    # prior: slightly above neutral midpoint (4.5)
}

# For UNKNOWN slots, we assume a slightly above-average value, encouraging
# replacement of unknown cards less than known high-value cards.
_UNKNOWN_EV = _BUCKET_EXPECTED_VALUE[CardBucket.UNKNOWN.value]


def _own_bucket_ev(agent_state, slot_idx: int) -> float:
    """Return expected value for agent's own hand slot under belief."""
    own_hand = getattr(agent_state, "own_hand", None)
    if not own_hand or slot_idx not in own_hand:
        return _UNKNOWN_EV
    info = own_hand[slot_idx]
    bucket = getattr(info, "bucket", None)
    if bucket is None:
        return _UNKNOWN_EV
    bval = int(bucket.value if hasattr(bucket, "value") else bucket)
    return _BUCKET_EXPECTED_VALUE.get(bval, _UNKNOWN_EV)


def _opp_bucket_ev(agent_state, slot_idx: int) -> float:
    """Return expected value for opponent hand slot under belief."""
    belief = getattr(agent_state, "opponent_belief", None)
    if not belief or slot_idx not in belief:
        return _UNKNOWN_EV
    val = belief[slot_idx]
    bval = getattr(val, "value", val)
    try:
        bval = int(bval)
    except (TypeError, ValueError):
        return _UNKNOWN_EV
    return _BUCKET_EXPECTED_VALUE.get(bval, _UNKNOWN_EV)


def belief_greedy_unabstract(
    abstract_idx: int,
    legal_actions: Sequence[GameAction],
    agent_state,
    seed: int,
) -> GameAction:
    """
    Belief-greedy unabstraction: within the chosen abstract class, pick
    the concrete action with the highest estimated value under the agent's
    belief state. Falls back to uniform (seed-deterministic) when no
    meaningful differentiation is possible.

    Value heuristics by action class:
    - replace_slot_*: pick the slot with highest own bucket EV (replace worst card)
    - snap_own: pick the slot with highest own bucket EV (snap worst card)
    - snap_opp / snap_opp_move: pick the slot with lowest opp bucket EV (snap best opp card)
    - peek_own_*: pick the slot with highest uncertainty (UNKNOWN > known_stale > known_recent)
    - blind_swap_* / king_look_*: pick own=highest EV slot, opp=lowest EV slot
    - king_swap: 1:1, no choice
    - all others: uniform fallback (including draw, discard, cambia, pass_snap)
    """
    # Collect candidates for this abstract class.
    candidates: List[GameAction] = [
        a for a in legal_actions
        if _concrete_to_abstract(a, agent_state) == int(abstract_idx)
    ]
    if not candidates:
        raise ValueError(
            f"No legal concrete action maps to abstract class {abstract_idx}"
        )
    if len(candidates) == 1:
        return candidates[0]

    # Sort for deterministic tie-breaking.
    candidates.sort(key=lambda a: repr(a))

    tag = getattr(candidates[0], "tag", None)

    if tag == "replace":
        # Pick the slot with the highest expected value (worst card to keep -> best to replace).
        scored = [
            (_own_bucket_ev(agent_state, int(getattr(c, "target_hand_index", 0))), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x[0])  # descending by EV (highest = worst to keep)
        # If tied, fall through to uniform among top-tied.
        best_ev = scored[0][0]
        top = [c for ev, c in scored if ev == best_ev]
        if len(top) == 1:
            return top[0]
        top.sort(key=lambda a: repr(a))
        return top[int(np.random.default_rng(seed).integers(0, len(top)))]

    if tag == "snap_own":
        # Snap own card: pick slot with highest EV (snap the worst card from your own hand).
        # ActionSnapOwn uses field name: own_card_hand_index
        scored = [
            (_own_bucket_ev(agent_state, int(getattr(c, "own_card_hand_index", 0))), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        best_ev = scored[0][0]
        top = [c for ev, c in scored if ev == best_ev]
        top.sort(key=lambda a: repr(a))
        return top[int(np.random.default_rng(seed).integers(0, len(top)))]

    if tag == "snap_opp":
        # Snap opponent card: pick slot with lowest EV (snap the best card from their hand).
        # ActionSnapOpponent uses field name: opponent_target_hand_index
        scored = [
            (_opp_bucket_ev(agent_state, int(getattr(c, "opponent_target_hand_index", 0))), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0])  # ascending: smallest EV = best opp card to snap
        best_ev = scored[0][0]
        top = [c for ev, c in scored if ev == best_ev]
        top.sort(key=lambda a: repr(a))
        return top[int(np.random.default_rng(seed).integers(0, len(top)))]

    if tag == "snap_opp_move":
        # Move action after snapping opp: pick own slot with highest EV (give worst card).
        # ActionSnapOpponentMove uses field name: own_card_to_move_hand_index
        scored = [
            (_own_bucket_ev(agent_state, int(getattr(c, "own_card_to_move_hand_index", 0))), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        best_ev = scored[0][0]
        top = [c for ev, c in scored if ev == best_ev]
        top.sort(key=lambda a: repr(a))
        return top[int(np.random.default_rng(seed).integers(0, len(top)))]

    if tag == "peek_own":
        # Prefer unknown/stale slots (information gain maximizing).
        own_hand = getattr(agent_state, "own_hand", None)
        def peek_priority(c):
            slot = int(getattr(c, "target_hand_index", 0))
            if not own_hand or slot not in own_hand:
                return 0  # highest priority: unknown
            info = own_hand[slot]
            bucket = getattr(info, "bucket", None)
            if bucket is None:
                return 0
            bval = int(bucket.value if hasattr(bucket, "value") else bucket)
            if bval == CardBucket.UNKNOWN.value:
                return 0
            # For known slots, prefer stale (higher information gain) over recent.
            last_seen = getattr(info, "last_seen_turn", -1)
            cur = int(getattr(agent_state, "_current_game_turn", 0) or 0)
            age = max(0, cur - (last_seen or 0))
            return -(age + 1)  # more negative = lower priority
        scored = sorted(candidates, key=peek_priority)
        return scored[0]

    if tag == "peek_other":
        # Prefer opponent slots with unknown belief (more information gain).
        belief = getattr(agent_state, "opponent_belief", None)
        def opp_peek_priority(c):
            slot = int(getattr(c, "target_opponent_hand_index", 0))
            if not belief or slot not in belief:
                return 0  # unknown = highest priority
            val = belief[slot]
            bval = getattr(val, "value", val)
            try:
                bval = int(bval)
            except (TypeError, ValueError):
                return 0
            return 1 if bval != CardBucket.UNKNOWN.value else 0
        scored = sorted(candidates, key=opp_peek_priority)
        return scored[0]

    if tag in ("blind_swap", "king_look"):
        # Pick own slot with highest EV to give away (replace worst own with unknown opp).
        # Secondary: prefer opponent slots with unknown belief.
        own_attr = "own_hand_index"
        opp_attr = "opponent_hand_index"
        belief = getattr(agent_state, "opponent_belief", None)
        def swap_priority(c):
            own_slot = int(getattr(c, own_attr, 0))
            opp_slot = int(getattr(c, opp_attr, 0))
            own_ev = _own_bucket_ev(agent_state, own_slot)
            # Prefer swapping own high-value slots (gets rid of worst card).
            # Prefer unknown opp slots (more uncertain = more potential upside).
            opp_known = 0
            if belief and opp_slot in belief:
                val = belief[opp_slot]
                bval = getattr(val, "value", val)
                try:
                    bval = int(bval)
                except (TypeError, ValueError):
                    bval = CardBucket.UNKNOWN.value
                opp_known = 0 if bval == CardBucket.UNKNOWN.value else 1
            # Sort key: descending own_ev, ascending opp_known (prefer unknown opp)
            return (-own_ev, opp_known)
        scored = sorted(candidates, key=swap_priority)
        return scored[0]

    # All other tags: uniform fallback.
    return unabstract(abstract_idx, legal_actions, agent_state, seed=seed)


# ---------------------------------------------------------------------------
# Belief-greedy DESCA wrapper (subclasses DESCAAgentWrapper, overrides unabstract)
# ---------------------------------------------------------------------------

class BeliefGreedyDESCAWrapper(DESCAAgentWrapper):
    """DESCA with belief-greedy within-class concrete action selection.

    Only choose_action is overridden; everything else (network loading,
    encoding, abstract policy) is inherited unchanged from DESCAAgentWrapper.
    """

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Same as DESCAAgentWrapper.choose_action but replaces unabstract() call."""
        from src.action_abstraction import abstract_actions

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            features = self._encode_v2(decision_context)
            abstract_mask = abstract_actions(legal_list, self.agent_state)
        except Exception as e:
            logger.error("BeliefGreedyDESCA P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(abstract_mask).bool().unsqueeze(0).to(self.device)
            probs = self.avg_strategy_net(feat_t, mask_t)
            probs_np = probs.squeeze(0).cpu().numpy()

        legal_abstract = np.where(abstract_mask)[0]
        if len(legal_abstract) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_abstract]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_abstract)) / len(legal_abstract)
        else:
            legal_probs = legal_probs / prob_sum

        if self._use_argmax:
            chosen_local = int(np.argmax(legal_probs))
        else:
            chosen_local = int(np.random.choice(len(legal_abstract), p=legal_probs))

        chosen_abstract_idx = int(legal_abstract[chosen_local])
        seed = hash((id(game_state), chosen_abstract_idx)) & 0xFFFF_FFFF

        try:
            return belief_greedy_unabstract(
                chosen_abstract_idx, legal_list, self.agent_state, seed=seed
            )
        except Exception as e:
            logger.error("BeliefGreedyDESCA P%d unabstract error: %s", self.player_id, e)
            return random.choice(legal_list)


# ---------------------------------------------------------------------------
# CRN-paired game loop
# ---------------------------------------------------------------------------

_BASE_SEED = 0xCAFEBABE


def _run_paired_games(
    config,
    checkpoint_path: str,
    baseline_type: str,
    num_games: int,
    device: str,
) -> tuple[Counter, Counter]:
    """Run num_games CRN-paired games for uniform vs belief_greedy.

    Returns (uniform_results, belief_greedy_results).
    Both conditions share the same per-game deal via seeded RNG.
    """
    from src.cfr.exceptions import GameStateError

    uniform_agent = DESCAAgentWrapper(0, config, checkpoint_path, device=device)
    greedy_agent = BeliefGreedyDESCAWrapper(0, config, checkpoint_path, device=device)

    uniform_results: Counter = Counter()
    greedy_results: Counter = Counter()

    max_turns = (
        config.cambia_rules.max_game_turns
        if config.cambia_rules.max_game_turns > 0
        else 500
    )

    for game_num in range(num_games):
        seed = _BASE_SEED + game_num

        for cond, desca_agent, results in [
            ("uniform", uniform_agent, uniform_results),
            ("greedy", greedy_agent, greedy_results),
        ]:
            rng = random.Random(seed)
            game_state = CambiaGameState(
                house_rules=config.cambia_rules,
                _rng=rng,
            )

            baseline_agent = get_agent(baseline_type, player_id=1, config=config)
            agents = [desca_agent, baseline_agent]

            # Initialize stateful agents.
            for agent in agents:
                if isinstance(agent, NeuralAgentWrapper):
                    agent.initialize_state(game_state)

            turn = 0
            error = False
            while not game_state.is_terminal() and turn < max_turns:
                turn += 1
                acting_pid = game_state.get_acting_player()
                if acting_pid == -1:
                    results["Errors"] += 1
                    error = True
                    break

                current_agent = agents[acting_pid]
                try:
                    legal = game_state.get_legal_actions()
                    if not legal:
                        if game_state.is_terminal():
                            break
                        results["Errors"] += 1
                        error = True
                        break

                    chosen = current_agent.choose_action(game_state, legal)
                    game_state.apply_action(chosen)

                    # Update agent state (baseline agents don't need it).
                    if isinstance(desca_agent, NeuralAgentWrapper) and hasattr(
                        desca_agent, "_create_observation"
                    ):
                        obs = desca_agent._create_observation(
                            game_state, chosen, acting_pid
                        )
                        if obs:
                            desca_agent.update_state(obs)
                            if isinstance(baseline_agent, NeuralAgentWrapper):
                                baseline_agent.update_state(obs)

                except Exception as e:
                    logger.debug("Game %d cond=%s turn %d error: %s", game_num, cond, turn, e)
                    results["Errors"] += 1
                    error = True
                    break

            if not error:
                if game_state.is_terminal():
                    winner = getattr(game_state, "_winner", None)
                    if winner == 0:
                        results["P0 Wins"] += 1
                    elif winner == 1:
                        results["P1 Wins"] += 1
                    else:
                        results["Ties"] += 1
                elif turn >= max_turns:
                    results["MaxTurnTies"] += 1

    return uniform_results, greedy_results


def _win_rate(results: Counter) -> float:
    """Fraction of games won by P0 (DESCA). Ties contribute 0.5."""
    w = results.get("P0 Wins", 0)
    t = results.get("Ties", 0)
    total = sum(
        results.get(k, 0)
        for k in ("P0 Wins", "P1 Wins", "Ties", "MaxTurnTies")
    )
    if total == 0:
        return float("nan")
    return (w + 0.5 * t) / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="RC-D eval-only probe: uniform vs belief-greedy unabstraction"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/desca-phase1-apcfr-mild/checkpoints/desca_checkpoint_iter_1000.pt"
        ),
        help="Path to DESCA .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        default="config/desca_phase1_apcfr_mild.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="CRN-paired games per baseline matchup (each game runs twice)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(MEAN_IMP_BASELINES),
        help="Baseline agent types to run against",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (cpu / cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base CRN seed (default: 0xCAFEBABE)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _BASE_SEED
    if args.seed is not None:
        _BASE_SEED = args.seed

    # Resolve paths relative to cfr/ root.
    checkpoint = args.checkpoint
    config_path = args.config

    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(_CFR_ROOT, checkpoint)
    if not os.path.isabs(config_path):
        config_path = os.path.join(_CFR_ROOT, config_path)

    if not os.path.exists(checkpoint):
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    if config is None:
        print(f"ERROR: failed to load config: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Belief-greedy unabstraction probe (RC-D eval-only share, U9)")
    print(f"  checkpoint : {checkpoint}")
    print(f"  config     : {config_path}")
    print(f"  num_games  : {args.num_games} CRN pairs per baseline")
    print(f"  baselines  : {args.baselines}")
    print(f"  base_seed  : 0x{_BASE_SEED:08X}")
    print()

    summary = {}
    t0 = time.perf_counter()

    for baseline in args.baselines:
        print(f"  Running vs {baseline} ...", end=" ", flush=True)
        bt = time.perf_counter()
        try:
            uni_res, grdy_res = _run_paired_games(
                config, checkpoint, baseline, args.num_games, args.device
            )
        except Exception as e:
            print(f"FAILED: {e}")
            logger.exception("Error vs %s", baseline)
            summary[baseline] = {"error": str(e)}
            continue

        uni_wr = _win_rate(uni_res)
        grdy_wr = _win_rate(grdy_res)
        delta = grdy_wr - uni_wr
        elapsed = time.perf_counter() - bt

        summary[baseline] = {
            "uniform_wr": uni_wr,
            "greedy_wr": grdy_wr,
            "delta_pp": delta * 100.0,
            "uniform_raw": dict(uni_res),
            "greedy_raw": dict(grdy_res),
        }

        print(
            f"uniform={uni_wr*100:.1f}%  greedy={grdy_wr*100:.1f}%"
            f"  delta={delta*100:+.1f}pp  ({elapsed:.0f}s)"
        )

    total_elapsed = time.perf_counter() - t0

    # Summary table.
    print()
    print("=" * 70)
    print(f"{'Baseline':<28} {'Uniform WR':>12} {'Greedy WR':>12} {'Delta (pp)':>12}")
    print("-" * 70)
    deltas = []
    for bl in args.baselines:
        if "error" in summary.get(bl, {}):
            print(f"  {bl:<26} {'ERROR':>12}")
            continue
        d = summary[bl]
        u = d["uniform_wr"]
        g = d["greedy_wr"]
        delta = d["delta_pp"]
        deltas.append(delta)
        print(f"  {bl:<26} {u*100:>11.1f}%  {g*100:>11.1f}%  {delta:>+10.2f}pp")

    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        print("-" * 70)
        print(f"  {'mean across baselines':<26} {'':>12} {'':>12} {mean_delta:>+10.2f}pp")
    print("=" * 70)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print()
    print(
        "Interpretation: positive delta means belief-greedy recovers that many "
        "percentage points of win-rate at eval time (no retraining). This is "
        "RC-D's eval-only share (U9)."
    )

    # Dump machine-readable results to stdout as JSON.
    print()
    print("--- JSON summary ---")
    print(json.dumps({"baselines": summary, "mean_delta_pp": mean_delta if deltas else None}, indent=2))


if __name__ == "__main__":
    main()
