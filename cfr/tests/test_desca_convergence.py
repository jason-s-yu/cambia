"""
tests/test_desca_convergence.py

Stream D deliverables: Tier 3 micro-game convergence test and DESCAAgentWrapper
smoke test.

Tier 3 convergence test:
    Trains DESCATrainer on the Phase 0 micro-game harness (2-card hands, 20-card
    deck, no abilities) for 30 iterations with a minimal config. Asserts that
    the avg-strategy network achieves >= 5pp improvement in win rate against a
    random policy baseline at iter 30 vs iter 0.

    Gate: DO NOT lower the 5pp threshold if the test fails. A failure signals a
    bug in DESCATrainer / DESCAWorker (Streams A/B). Escalate to @chief.

    Marked @pytest.mark.slow; excluded from fast CI.

DESCAAgentWrapper smoke test:
    Loads an untrained (fresh-weights) DESCAAgentWrapper and asserts that every
    action returned on 50 random legal 2P micro-game states is contained in the
    legal-action set. Catches unabstract / encoding bugs without running training.

Depends on:
    - Stream A: src.desca_networks (AvgStrategyNetwork)
    - Stream B: src.cfr.desca_trainer (DESCATrainer)
    - Stream C: src.evaluate_agents (DESCAAgentWrapper)

When Stream B has not yet delivered, the Tier 3 test skips automatically. The
smoke test runs as long as Streams A + C are available.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    from src.desca_networks import AvgStrategyNetwork

    DESCA_NETWORKS_AVAILABLE = True
except ImportError:
    DESCA_NETWORKS_AVAILABLE = False

try:
    from src.cfr.desca_trainer import DESCATrainer

    DESCA_TRAINER_AVAILABLE = True
except ImportError:
    DESCA_TRAINER_AVAILABLE = False

try:
    from src.evaluate_agents import DESCAAgentWrapper

    DESCA_WRAPPER_AVAILABLE = True
except ImportError:
    DESCA_WRAPPER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------


def _make_agent_config(micro_rules=None):
    """Return a minimal config object for AgentState and DESCAAgentWrapper."""
    config = type("Config", (), {})()

    if micro_rules is None:
        from tests.micro_game import build_micro_rules

        micro_rules = build_micro_rules()
    config.cambia_rules = micro_rules

    agent_params = type("AgentParamsConfig", (), {})()
    agent_params.memory_level = 0
    agent_params.time_decay_turns = 0
    config.agent_params = agent_params

    agents_cfg = type("AgentsConfig", (), {})()
    agents_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_cambia_threshold = 5
    config.agents = agents_cfg

    return config


def _make_desca_config(
    iterations: int = 30,
    traversals_per_iter: int = 80,
    minibatch: int = 32,
    hidden_dim: int = 32,
    buffer_capacity: int = 20_000,
    inner_update: str = "apcfr_plus",
    regret_sgd_steps: int = 100,
    strategy_sgd_steps: int = 100,
    value_sgd_steps: int = 50,
):
    """Minimal DESCAConfig for fast convergence tests.

    regret_sgd_steps / strategy_sgd_steps / value_sgd_steps default to 100/100/50
    (vs trainer defaults 2000/2000/1000) to keep wall-clock under 5 min on CPU.
    """
    cfg = type("DESCAConfig", (), {})()
    cfg.encoding_version = 2
    cfg.hidden_dim = hidden_dim
    cfg.num_abstract_actions = 32
    cfg.iterations = iterations
    cfg.traversals_per_iter = traversals_per_iter
    cfg.minibatch = minibatch
    cfg.lr = 5e-4
    cfg.weight_decay = 1e-4
    cfg.grad_clip = 1.0
    cfg.dcfr_alpha = 1.5
    cfg.apcfr_asymmetry = 0.9
    cfg.buffer_capacity = buffer_capacity
    cfg.checkpoint_every = 1000  # effectively disabled for short test runs
    cfg.eval_every = 1000
    cfg.warmup_iters = 5
    cfg.inner_update = inner_update
    # SGD steps per iteration: use small values to keep CPU wall-clock under 5 min.
    # Production defaults (2000/2000/1000) would run ~5x longer on CPU.
    cfg.regret_sgd_steps = regret_sgd_steps
    cfg.strategy_sgd_steps = strategy_sgd_steps
    cfg.value_sgd_steps = value_sgd_steps
    stall = type("StallDetection", (), {})()
    stall.window_size_iters = 50
    stall.num_windows = 5
    stall.max_iter_abs = 3000
    cfg.stall_detection = stall
    return cfg


def _make_desca_checkpoint(path: str, hidden_dim: int = 64) -> None:
    """Save a freshly-initialized (untrained) DESCA checkpoint."""
    from src.desca_networks import AvgStrategyNetwork
    from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P
    from src.constants import EP_PBS_V2_INPUT_DIM

    net = AvgStrategyNetwork(
        input_dim=EP_PBS_V2_INPUT_DIM,
        hidden_dim=hidden_dim,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
    )
    checkpoint = {
        "avg_strategy_state_dict": net.state_dict(),
        "desca_config": {
            "hidden_dim": hidden_dim,
            "encoding_dim": EP_PBS_V2_INPUT_DIM,
            "num_abstract_actions": NUM_ABSTRACT_ACTIONS_2P,
        },
        "iteration": 0,
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# Python engine adapter for the DESCA worker
# ---------------------------------------------------------------------------


class _MicroGameEngine:
    """
    Adapter wrapping Python CambiaGameState to expose the DESCA worker's
    engine interface. Supports save/restore via deepcopy.

    Tracks context from the most recent apply_action call so agents can
    build AgentObservation objects in their update() method.
    """

    def __init__(self, game):
        self._game = game
        # Set by apply_action; read by _DESCATestAgent.update().
        self._last_actor: int = -1
        self._last_action = None

    def legal_actions(self) -> list:
        actions = self._game.get_legal_actions()
        return sorted(actions, key=repr)

    def is_terminal(self) -> bool:
        return self._game.is_terminal()

    def get_utility(self) -> list:
        if not self._game.is_terminal():
            return [0.0, 0.0]
        try:
            return [float(self._game.get_utility(i)) for i in range(2)]
        except Exception:
            return [0.0, 0.0]

    def get_acting_player(self) -> int:
        return int(self._game.current_player_index)

    def apply_action(self, action) -> None:
        self._last_actor = int(self._game.current_player_index)
        self._last_action = action
        try:
            self._game.apply_action(action)
        except Exception:
            pass

    def save(self):
        snap = copy.deepcopy(self._game)
        return snap

    def restore(self, snap) -> None:
        self._game.__dict__.update(snap.__dict__)
        self._last_actor = -1
        self._last_action = None

    def free_snapshot(self, snap) -> None:
        pass

    def get_decision_context(self) -> int:
        from src.constants import DecisionContext, ActionDiscard

        if getattr(self._game, "snap_phase_active", False):
            return DecisionContext.SNAP_DECISION.value
        pending = getattr(self._game, "pending_action", None)
        if pending is not None:
            if isinstance(pending, ActionDiscard):
                return DecisionContext.POST_DRAW.value
            return DecisionContext.ABILITY_SELECT.value
        return DecisionContext.START_TURN.value

    def get_drawn_card_bucket(self) -> int:
        # Drawn-card bucket is not tracked in the Python path at this level.
        return -1

    def _omniscient_features(self) -> np.ndarray:
        # Zeros: Python backend does not have direct card-identity access
        # through the worker interface. The history-value network will
        # receive zero-padded omniscient features during training. This is
        # acceptable for a convergence test; the network can still learn
        # from the regret signal.
        return np.zeros(120, dtype=np.float32)


# ---------------------------------------------------------------------------
# Python agent adapter for the DESCA worker
# ---------------------------------------------------------------------------


class _DESCATestAgent:
    """
    Thin composition wrapper around Python AgentState that overrides update()
    to accept a _MicroGameEngine rather than an AgentObservation.

    The DESCA worker calls agent.update(engine) after each apply_action.
    GoAgentState.update(engine) does this natively via FFI. For the Python
    path used in this convergence test, we intercept the call and build
    a minimal AgentObservation from the engine's stored last-action context.

    All other attribute accesses are forwarded to the underlying AgentState
    so the encoding function (encode_infoset_eppbs_interleaved_v2) can read
    slot_tags, slot_buckets, etc. directly.
    """

    def __init__(self, agent_state):
        # Store under a name that doesn't conflict with AgentState attributes.
        object.__setattr__(self, "_agent", agent_state)

    def update(self, engine: _MicroGameEngine) -> None:
        """Build an AgentObservation from the engine and update agent beliefs."""
        if not isinstance(engine, _MicroGameEngine):
            # Direct observation passed (should not happen in normal flow).
            try:
                self._agent.update(engine)
            except Exception:
                pass
            return

        if engine._last_action is None:
            return

        game = engine._game
        try:
            from src.agent_state import AgentObservation

            obs = AgentObservation(
                acting_player=engine._last_actor,
                action=engine._last_action,
                discard_top_card=game.get_discard_top(),
                player_hand_sizes=[game.get_player_card_count(i) for i in range(2)],
                stockpile_size=game.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=list(getattr(game, "snap_results_log", [])),
                did_cambia_get_called=getattr(game, "cambia_caller_id", None) is not None,
                who_called_cambia=getattr(game, "cambia_caller_id", None),
                is_game_over=game.is_terminal(),
                current_turn=game.get_turn_number(),
            )
            self._agent.update(obs)
        except Exception:
            pass

    def clone(self) -> "_DESCATestAgent":
        return _DESCATestAgent(copy.deepcopy(self._agent))

    def __getattr__(self, name: str):
        # Forward all unknown attribute lookups to the underlying agent state.
        return getattr(object.__getattribute__(self, "_agent"), name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_agent":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_agent"), name, value)


# ---------------------------------------------------------------------------
# Env factory for the convergence test
# ---------------------------------------------------------------------------


def _build_micro_env_factory(seed_base: int = 0):
    """
    Build an env_factory for the DESCA convergence test using the Python
    micro-game harness (2-card hands, 20-card deck, no abilities).

    The factory creates a fresh game + initialized agent states on each call,
    matching the DESCA worker's (engine, agents) interface.
    """
    from tests.micro_game import build_micro_game, build_micro_rules
    from src.agent_state import AgentState, AgentObservation

    micro_rules = build_micro_rules()
    agent_config = _make_agent_config(micro_rules)
    _counter = [0]

    def factory(rng=None):
        _counter[0] += 1
        seed = seed_base + _counter[0]
        game = build_micro_game(seed=seed)

        agents = []
        for pid in range(2):
            agent_state = AgentState(
                player_id=pid,
                opponent_id=1 - pid,
                memory_level=0,
                time_decay_turns=0,
                initial_hand_size=2,
                config=agent_config,
            )
            initial_hand = game.players[pid].hand
            initial_peeks = getattr(
                game.players[pid], "initial_peek_indices", tuple(range(2))
            )
            init_obs = AgentObservation(
                acting_player=-1,
                action=None,
                discard_top_card=game.get_discard_top(),
                player_hand_sizes=[game.get_player_card_count(i) for i in range(2)],
                stockpile_size=game.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=[],
                did_cambia_get_called=False,
                who_called_cambia=None,
                is_game_over=False,
                current_turn=0,
            )
            agent_state.initialize(init_obs, initial_hand, initial_peeks)
            agents.append(_DESCATestAgent(agent_state))

        engine = _MicroGameEngine(game)
        return engine, agents

    return factory


# ---------------------------------------------------------------------------
# Simple micro-game evaluator
# ---------------------------------------------------------------------------


def _run_micro_episode(policy0, policy1, seed: int, max_turns: int = 64) -> float:
    """
    Run one micro-game episode between two policies.

    policy_i: callable(game_state, legal_actions, player_id) -> action
    Returns: 1.0 if P0 wins, 0.0 if P0 loses, 0.5 on tie.
    """
    from tests.micro_game import build_micro_game

    game = build_micro_game(seed=seed)

    for _ in range(max_turns):
        if game.is_terminal():
            break
        legal = list(game.get_legal_actions())
        if not legal:
            break
        pid = game.current_player_index
        policy = policy0 if pid == 0 else policy1
        try:
            action = policy(game, legal, pid)
        except Exception:
            action = random.choice(legal)
        try:
            game.apply_action(action)
        except Exception:
            break

    try:
        u0 = float(game.get_utility(0))
        u1 = float(game.get_utility(1))
    except Exception:
        return 0.5

    if u0 > u1:
        return 1.0
    if u1 > u0:
        return 0.0
    return 0.5


def _random_policy(game_state, legal_actions: list, player_id: int):
    return random.choice(legal_actions)


def _evaluate_policy_vs_random(policy, n_games: int = 200, seed_offset: int = 0) -> float:
    """Win rate of `policy` vs random, alternating roles."""
    wins = 0.0
    for i in range(n_games):
        seed = seed_offset + i
        if i % 2 == 0:
            wins += _run_micro_episode(policy, _random_policy, seed=seed)
        else:
            wins += 1.0 - _run_micro_episode(_random_policy, policy, seed=seed)
    return wins / n_games


def _net_to_policy(avg_strategy_net):
    """
    Build a game-loop policy callable from an AvgStrategyNetwork.

    Uses abstract_actions / unabstract with a null agent state (zero features).
    Suitable for measuring whether the network has learned any useful bias.
    """
    from src.action_abstraction import abstract_actions, unabstract
    from src.constants import EP_PBS_V2_INPUT_DIM

    class _NullAgent:
        own_hand: dict = {}
        opponent_belief: dict = {}
        _current_game_turn: int = 0

    avg_strategy_net.eval()
    null_agent = _NullAgent()

    def policy(game_state, legal_actions: list, player_id: int):
        try:
            features = np.zeros(EP_PBS_V2_INPUT_DIM, dtype=np.float32)
            abstract_mask = abstract_actions(legal_actions, null_agent)
            feat_t = torch.from_numpy(features).float().unsqueeze(0)
            mask_t = torch.from_numpy(abstract_mask).bool().unsqueeze(0)
            with torch.inference_mode():
                probs = avg_strategy_net(feat_t, mask_t).squeeze(0).numpy()
            legal_abstract = np.where(abstract_mask)[0]
            if len(legal_abstract) == 0:
                return random.choice(legal_actions)
            legal_probs = probs[legal_abstract]
            prob_sum = legal_probs.sum()
            if prob_sum <= 0:
                legal_probs = np.ones(len(legal_abstract)) / len(legal_abstract)
            else:
                legal_probs /= prob_sum
            chosen_local = int(np.random.choice(len(legal_abstract), p=legal_probs))
            chosen_abstract_idx = int(legal_abstract[chosen_local])
            seed = (player_id * 997 + chosen_abstract_idx) & 0xFFFF_FFFF
            return unabstract(chosen_abstract_idx, legal_actions, null_agent, seed=seed)
        except Exception:
            return random.choice(legal_actions)

    return policy


# ---------------------------------------------------------------------------
# State-tracked evaluation for Tier 3 convergence test
# ---------------------------------------------------------------------------


def _run_episode_desca_vs_random(
    avg_strategy_net,
    desca_player: int,
    seed: int,
    agent_config,
    max_turns: int = 64,
) -> float:
    """
    One episode: avg_strategy_net as `desca_player`, random as the other.

    Initializes proper AgentState for each player and tracks it across turns
    so the DESCA policy receives real feature vectors (not zeros). This makes
    the convergence gate actually sensitive to whether the network learned.

    Returns 1.0 if DESCA wins, 0.0 if DESCA loses, 0.5 on tie.
    """
    from tests.micro_game import build_micro_game
    from src.agent_state import AgentState, AgentObservation
    from src.encoding import encode_infoset_eppbs_interleaved_v2
    from src.action_abstraction import abstract_actions, unabstract
    from src.constants import DecisionContext, ActionDiscard

    game = build_micro_game(seed=seed)

    # Initialize one AgentState per player
    agents_eval: List[AgentState] = []
    for pid in range(2):
        agent = AgentState(
            player_id=pid,
            opponent_id=1 - pid,
            memory_level=0,
            time_decay_turns=0,
            initial_hand_size=2,
            config=agent_config,
        )
        initial_hand = game.players[pid].hand
        initial_peeks = getattr(
            game.players[pid], "initial_peek_indices", tuple(range(2))
        )
        init_obs = AgentObservation(
            acting_player=-1,
            action=None,
            discard_top_card=game.get_discard_top(),
            player_hand_sizes=[game.get_player_card_count(i) for i in range(2)],
            stockpile_size=game.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=[],
            did_cambia_get_called=False,
            who_called_cambia=None,
            is_game_over=False,
            current_turn=0,
        )
        agent.initialize(init_obs, initial_hand, initial_peeks)
        agents_eval.append(agent)

    avg_strategy_net.eval()
    last_actor = -1
    last_action = None

    for _ in range(max_turns):
        if game.is_terminal():
            break
        legal = list(game.get_legal_actions())
        if not legal:
            break
        pid = int(game.current_player_index)

        if pid == desca_player:
            try:
                # Decision context
                if getattr(game, "snap_phase_active", False):
                    ctx = DecisionContext.SNAP_DECISION.value
                elif getattr(game, "pending_action", None) is not None:
                    if isinstance(getattr(game, "pending_action", None), ActionDiscard):
                        ctx = DecisionContext.POST_DRAW.value
                    else:
                        ctx = DecisionContext.ABILITY_SELECT.value
                else:
                    ctx = DecisionContext.START_TURN.value

                features = encode_infoset_eppbs_interleaved_v2(agents_eval[pid], ctx, -1)
                abstract_mask = abstract_actions(legal, agents_eval[pid])
                feat_t = torch.from_numpy(features).float().unsqueeze(0)
                mask_t = torch.from_numpy(abstract_mask).bool().unsqueeze(0)
                with torch.inference_mode():
                    probs = avg_strategy_net(feat_t, mask_t).squeeze(0).numpy()
                legal_abstract = np.where(abstract_mask)[0]
                if len(legal_abstract) == 0:
                    action = random.choice(legal)
                else:
                    legal_probs = probs[legal_abstract]
                    prob_sum = legal_probs.sum()
                    if prob_sum <= 0:
                        legal_probs = np.ones(len(legal_abstract)) / len(legal_abstract)
                    else:
                        legal_probs /= prob_sum
                    chosen_local = int(
                        np.random.choice(len(legal_abstract), p=legal_probs)
                    )
                    chosen_abstract_idx = int(legal_abstract[chosen_local])
                    seed_a = (pid * 997 + chosen_abstract_idx) & 0xFFFF_FFFF
                    action = unabstract(
                        chosen_abstract_idx, legal, agents_eval[pid], seed=seed_a
                    )
            except Exception:
                action = random.choice(legal)
        else:
            action = random.choice(legal)

        last_actor = pid
        last_action = action

        try:
            game.apply_action(action)
        except Exception:
            break

        # Update both agents' beliefs
        for agent in agents_eval:
            try:
                obs = AgentObservation(
                    acting_player=last_actor,
                    action=last_action,
                    discard_top_card=game.get_discard_top(),
                    player_hand_sizes=[game.get_player_card_count(j) for j in range(2)],
                    stockpile_size=game.get_stockpile_size(),
                    drawn_card=None,
                    peeked_cards=None,
                    snap_results=list(getattr(game, "snap_results_log", [])),
                    did_cambia_get_called=getattr(game, "cambia_caller_id", None)
                    is not None,
                    who_called_cambia=getattr(game, "cambia_caller_id", None),
                    is_game_over=game.is_terminal(),
                    current_turn=game.get_turn_number(),
                )
                agent.update(obs)
            except Exception:
                pass

    try:
        u_desca = float(game.get_utility(desca_player))
        u_rand = float(game.get_utility(1 - desca_player))
    except Exception:
        return 0.5

    if u_desca > u_rand:
        return 1.0
    if u_rand > u_desca:
        return 0.0
    return 0.5


def _evaluate_desca_vs_random(
    avg_strategy_net,
    agent_config,
    n_games: int = 200,
    seed_offset: int = 0,
) -> float:
    """
    Win rate of avg_strategy_net vs random baseline with proper agent state
    tracking. Alternates DESCA as P0 / P1 to cancel first-mover bias.
    """
    wins = 0.0
    for i in range(n_games):
        seed = seed_offset + i
        role = i % 2  # alternate: 0 = DESCA as P0, 1 = DESCA as P1
        wins += _run_episode_desca_vs_random(avg_strategy_net, role, seed, agent_config)
    return wins / n_games


# ---------------------------------------------------------------------------
# Deliverable 4: DESCAAgentWrapper smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DESCA_NETWORKS_AVAILABLE and DESCA_WRAPPER_AVAILABLE),
    reason="Requires desca_networks.py (Stream A) and DESCAAgentWrapper (Stream C)",
)
def test_desca_agent_wrapper_smoke_50_states():
    """
    Smoke test: untrained DESCAAgentWrapper returns a legal action on every one
    of 50 randomly sampled 2P micro-game states. Catches unabstract and encoding
    bugs without running training.
    """
    from tests.micro_game import build_micro_game, build_micro_rules

    micro_rules = build_micro_rules()
    config = _make_agent_config(micro_rules)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        _make_desca_checkpoint(ckpt_path, hidden_dim=64)
        agent = DESCAAgentWrapper(0, config, ckpt_path, device="cpu", use_argmax=False)

        for trial in range(50):
            game = build_micro_game(seed=trial)
            agent.initialize_state(game)
            legal = game.get_legal_actions()
            assert (
                len(legal) > 0
            ), f"Trial {trial}: micro-game has no legal actions at start"
            chosen = agent.choose_action(game, legal)
            assert chosen in legal, (
                f"Trial {trial}: DESCAAgentWrapper returned action not in legal set.\n"
                f"  chosen: {chosen!r}\n"
                f"  legal: {sorted([repr(a) for a in legal])}"
            )
    finally:
        try:
            os.unlink(ckpt_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Deliverable 1: Tier 3 micro-game convergence test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not DESCA_TRAINER_AVAILABLE,
    reason="DESCATrainer not available (Stream B). Will run once desca_trainer.py is delivered.",
)
def test_desca_micro_game_convergence_30_iters():
    """
    Tier 3 convergence test. Trains DESCA on the Phase 0 micro-game harness
    for 30 iterations and asserts >= 5pp win-rate improvement over random.

    Gate: if this test fails, the 5pp threshold MUST NOT be lowered. Failure
    indicates a bug in DESCA traversal or fitting (Streams A/B). Escalate.

    Target wall clock: < 5 min on CPU (K=80, sgd_steps=100/100/50, hidden_dim=32).
    Compared to production defaults (2000/2000/1000 SGD steps), these reduced
    step counts keep the test under the 5-min mark while providing enough
    gradient signal for the micro-game's small action space.
    """
    import shutil
    from src.desca_networks import RegretNetwork, AvgStrategyNetwork, HistoryValueNetwork
    from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P
    from src.constants import EP_PBS_V2_INPUT_DIM
    from tests.micro_game import build_micro_rules

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    micro_rules = build_micro_rules()
    agent_config = _make_agent_config(micro_rules)

    desca_cfg = _make_desca_config(
        iterations=30,
        traversals_per_iter=80,
        minibatch=32,
        hidden_dim=32,
        buffer_capacity=20_000,
    )

    regret_net = RegretNetwork(
        input_dim=EP_PBS_V2_INPUT_DIM,
        hidden_dim=32,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
    )
    avg_strategy_net = AvgStrategyNetwork(
        input_dim=EP_PBS_V2_INPUT_DIM,
        hidden_dim=32,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
    )
    history_value_net = HistoryValueNetwork(
        input_dim=EP_PBS_V2_INPUT_DIM,
        hidden_dim=32,
    )

    env_factory = _build_micro_env_factory(seed_base=42)

    # Baseline: iter-0 win rate with proper agent state tracking.
    # Uses real features so changes are meaningful and not noise-driven.
    wr_iter0 = _evaluate_desca_vs_random(
        avg_strategy_net, agent_config, n_games=200, seed_offset=10000
    )

    run_dir = tempfile.mkdtemp(prefix="desca_conv_test_")
    ckpt_path = os.path.join(run_dir, "desca_checkpoint.pt")

    try:
        trainer = DESCATrainer(
            desca_cfg,
            regret_net,
            avg_strategy_net,
            history_value_net,
            env_factory,
            device="cpu",
            checkpoint_path=ckpt_path,
            seed=0,
        )
        trainer.train(num_iterations=30)

        # Post-training win rate: same evaluation with real features.
        wr_iter30 = _evaluate_desca_vs_random(
            trainer.avg_strategy_net, agent_config, n_games=200, seed_offset=20000
        )

        improvement_pp = (wr_iter30 - wr_iter0) * 100.0

        assert improvement_pp >= 5.0, (
            f"Tier 3 convergence FAILED: iter-30 win rate {wr_iter30:.1%} is only "
            f"{improvement_pp:.1f}pp above iter-0 baseline {wr_iter0:.1%}.\n"
            f"Required >= 5pp improvement. DO NOT lower this threshold.\n"
            f"Investigate DESCATrainer / DESCAWorker for bugs and escalate to @chief."
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
