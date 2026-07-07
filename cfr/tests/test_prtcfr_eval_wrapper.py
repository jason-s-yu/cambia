"""tests/test_prtcfr_eval_wrapper.py

Scoped tests for the PRT-CFR SD-CFR eval mixture + wrapper (cambia-233):

  - src/cfr/prtcfr_mixture.py  (PRTCFRMixture: deployable-window loading, SD-CFR
    linear weights, per-episode single-snapshot sampling, policy query)
  - src/evaluate_agents.py::PRTCFRAgentWrapper (engine-token-stream feed +
    masked-advantage policy) and its registration
  - src/run_db.py + src/cli.py prt_cfr auto-detect / run-dir discovery

The gate-critical property (X4): SD-CFR is realized by sampling ONE snapshot
per EPISODE proportional to w_t = t, NOT by per-decision averaging, and the
policy query is byte-identical to the training-time sigma^t / inference service.
"""

import logging
import os
import sys
import random
from collections import Counter

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_mixture import (  # noqa: E402
    PRTCFRMixture,
    _sd_cfr_weights,
    _infer_arch_from_state,
    _discover_snapshot_iters,
)
from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402
from src.cfr.prtcfr_infer import PRTCFRInferenceService  # noqa: E402
from src.cfr.prtcfr_stability import BestSnapshotController, write_deployable_manifest  # noqa: E402
from src.cfr.prtcfr_worker import (  # noqa: E402
    new_production_driver,
    _legal_mask,
    _sample_and_apply,
    uniform_policy_production,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiny_net(seed: int) -> PRTCFRNet:
    torch.manual_seed(seed)
    return PRTCFRNet(
        embed_dim=8, hidden_dim=16, num_layers=2, head_hidden_dim=16,
        dropout=0.0, device="cpu",
    )


def _save_snapshot(net: PRTCFRNet, path: str, iteration: int) -> None:
    torch.save(
        {
            "encoder_state_dict": net.encoder_state_dict(),
            "head_state_dict": net.head_state_dict(),
            "iteration": iteration,
        },
        path,
    )


def _make_run_dir(tmp_path, iters, best_iteration=None):
    """Build a run dir with prtcfr_snapshot_iter_{t}.pt files + a deployable
    manifest pinned to <= best_iteration (or the whole set when None)."""
    snapdir = tmp_path / "snapshots"
    snapdir.mkdir()
    for t in iters:
        _save_snapshot(_tiny_net(t), str(snapdir / f"prtcfr_snapshot_iter_{t}.pt"), t)
    ctrl = BestSnapshotController()
    if best_iteration is not None:
        ctrl.best_iteration = best_iteration
    write_deployable_manifest(str(snapdir), ctrl, list(iters))
    return snapdir


def _real_tokens(min_len: int = 60) -> list:
    """A real full-recall token prefix from the production Python-engine driver."""
    logging.disable(logging.CRITICAL)
    try:
        for seed in range(200):
            driver = new_production_driver(seed=seed, backend="python")
            driver.seq_cap = 20000
            rng = random.Random(seed)
            for _ in range(3000):
                if driver.is_terminal():
                    break
                actor = driver.current_player()
                if actor == -1:
                    break
                legal = driver.legal_actions()
                if not legal:
                    break
                try:
                    _sample_and_apply(
                        driver, legal, uniform_policy_production([], _legal_mask(legal)), rng
                    )
                except RuntimeError:
                    break
                toks = driver.tokens(0)
                if len(toks) >= min_len:
                    return toks
        pytest.fail("could not generate a real token sequence")
    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# SD-CFR weights + per-episode sampling (mixture faithfulness)
# ---------------------------------------------------------------------------


def test_sd_cfr_linear_weights_are_proportional_to_iteration():
    w = _sd_cfr_weights([1, 2, 3, 4], "linear")
    assert np.allclose(w, np.array([1, 2, 3, 4]) / 10.0)
    assert abs(w.sum() - 1.0) < 1e-12


def test_sd_cfr_uniform_weights():
    w = _sd_cfr_weights([5, 10, 42], "uniform")
    assert np.allclose(w, np.ones(3) / 3.0)


def test_sd_cfr_weights_reject_unknown_scheme():
    with pytest.raises(ValueError):
        _sd_cfr_weights([1, 2], "reach")


def test_sample_episode_is_deterministic_under_seed():
    nets = [_tiny_net(i) for i in range(4)]
    mix_a = PRTCFRMixture([1, 2, 3, 4], nets)
    mix_b = PRTCFRMixture([1, 2, 3, 4], nets)
    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    seq_a = [mix_a.sample_episode(rng_a) for _ in range(50)]
    seq_b = [mix_b.sample_episode(rng_b) for _ in range(50)]
    assert seq_a == seq_b  # same seed -> identical snapshot sequence


def test_sample_episode_frequencies_match_linear_weights():
    nets = [_tiny_net(i) for i in range(4)]
    mix = PRTCFRMixture([1, 2, 3, 4], nets, weighting="linear")
    rng = np.random.default_rng(7)
    n = 40000
    counts = Counter(mix.sample_episode(rng) for _ in range(n))
    freqs = np.array([counts[i] / n for i in range(4)])
    # Expected linear weights w_t = t: [0.1, 0.2, 0.3, 0.4].
    assert np.allclose(freqs, mix.weights, atol=0.01)


def test_active_net_is_single_snapshot_not_average_within_episode():
    # Faithfulness guard: within one episode the SAME sampled net answers every
    # decision (SD-CFR trajectory sampling), never a per-decision average.
    nets = [_tiny_net(i) for i in range(3)]
    mix = PRTCFRMixture([1, 2, 3], nets)
    mix.sample_episode(np.random.default_rng(0))
    idx = mix._active_idx
    net_first = mix.active_net()
    mask = np.zeros(146, dtype=bool)
    mask[[0, 5, 9]] = True
    for _ in range(5):
        mix.strategy([2, 10, 20], mask)  # querying must not resample
    assert mix._active_idx == idx
    assert mix.active_net() is net_first
    assert mix.active_net() is nets[idx]


def test_active_net_before_sample_raises():
    mix = PRTCFRMixture([1], [_tiny_net(0)])
    with pytest.raises(RuntimeError):
        mix.active_net()


# ---------------------------------------------------------------------------
# Deployable-window loading (manifest seam)
# ---------------------------------------------------------------------------


def test_from_checkpoint_respects_deployable_manifest(tmp_path):
    snapdir = _make_run_dir(tmp_path, [1, 2, 3, 4], best_iteration=2)
    mix = PRTCFRMixture.from_checkpoint(str(snapdir / "prtcfr_snapshot_iter_4.pt"))
    assert mix.iters == [1, 2]  # manifest pins deployable to <= best_iteration
    assert np.allclose(mix.weights, np.array([1, 2]) / 3.0)


def test_from_checkpoint_unpinned_loads_all_snapshots(tmp_path):
    snapdir = tmp_path / "snapshots"
    snapdir.mkdir()
    for t in (1, 2, 3):
        _save_snapshot(_tiny_net(t), str(snapdir / f"prtcfr_snapshot_iter_{t}.pt"), t)
    # No manifest -> every snapshot deployable.
    mix = PRTCFRMixture.from_checkpoint(str(snapdir / "prtcfr_snapshot_iter_1.pt"))
    assert mix.iters == [1, 2, 3]
    assert _discover_snapshot_iters(str(snapdir)) == [1, 2, 3]


def test_from_checkpoint_max_iter_restricts_window(tmp_path):
    snapdir = _make_run_dir(tmp_path, [1, 2, 3, 4], best_iteration=None)
    mix = PRTCFRMixture.from_checkpoint(
        str(snapdir / "prtcfr_snapshot_iter_4.pt"), max_iter=3
    )
    assert mix.iters == [1, 2, 3]  # --epoch N "mixture as of iter N"


def test_from_checkpoint_single_file_degenerate(tmp_path):
    # A lone snapshot with no siblings and no manifest -> single-snapshot mixture.
    lone = tmp_path / "prtcfr_snapshot_iter_9.pt"
    _save_snapshot(_tiny_net(9), str(lone), 9)
    mix = PRTCFRMixture.from_checkpoint(str(lone))
    assert len(mix) == 1
    assert mix.iters == [9]


def test_arch_inferred_from_nondefault_state():
    net = PRTCFRNet(
        embed_dim=12, hidden_dim=20, num_layers=3, head_hidden_dim=24,
        dropout=0.0, device="cpu",
    )
    arch = _infer_arch_from_state(net.encoder_state_dict(), net.head_state_dict())
    assert arch["embed_dim"] == 12
    assert arch["hidden_dim"] == 20
    assert arch["num_layers"] == 3
    assert arch["head_hidden_dim"] == 24
    assert arch["num_actions"] == 146


# ---------------------------------------------------------------------------
# Policy query byte-identical to the production inference service (X4-critical)
# ---------------------------------------------------------------------------


def test_stream_replay_policy_equivalence_vs_inference_service():
    """The mixture's batch policy query equals the PRTCFRInferenceService's
    incremental register()+step() carry over the SAME token stream, in fp32
    (exact). This is the "stream-replay policy equivalence vs the inference
    service" faithfulness check: the eval wrapper's policy path is the identical
    object the production incremental service would serve."""
    net = _tiny_net(7)
    mix = PRTCFRMixture([1], [net])
    mix.sample_episode(np.random.default_rng(0))
    tokens = _real_tokens(min_len=60)

    mask = np.zeros(146, dtype=bool)
    mask[[0, 3, 7, 42]] = True
    probs_batch = mix.strategy(tokens, mask)

    svc = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
    svc.register("s", tokens[:6])
    i = 6
    while i < len(tokens):
        svc.step(["s"], [tokens[i : i + 4]])
        i += 4
    probs_carry = (
        svc.strategy(["s"], torch.as_tensor(mask).unsqueeze(0))[0]
        .to(torch.float64)
        .numpy()
    )
    assert np.abs(probs_batch - probs_carry).max() < 1e-5


def test_strategy_masks_illegal_actions_and_normalizes():
    net = _tiny_net(3)
    mix = PRTCFRMixture([1], [net])
    mix.sample_episode(np.random.default_rng(0))
    mask = np.zeros(146, dtype=bool)
    mask[[1, 4, 8]] = True
    probs = mix.strategy([2, 11, 22, 33], mask)
    assert probs[~mask].sum() == 0.0
    assert abs(probs[mask].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Registration + auto-detect
# ---------------------------------------------------------------------------


def test_registered_in_agent_registry_and_get_agent(tmp_path):
    from src.evaluate_agents import AGENT_REGISTRY, get_agent, PRTCFRAgentWrapper
    from src.config import load_config

    assert AGENT_REGISTRY["prt_cfr"] is PRTCFRAgentWrapper
    snapdir = _make_run_dir(tmp_path, [1, 2], best_iteration=2)
    cfg = load_config(os.path.join(PROJECT_ROOT, "config", "prtcfr_production.yaml"))
    agent = get_agent(
        "prt_cfr",
        player_id=0,
        config=cfg,
        checkpoint_path=str(snapdir / "prtcfr_snapshot_iter_2.pt"),
        device="cpu",
    )
    assert isinstance(agent, PRTCFRAgentWrapper)
    assert len(agent._mixture) == 2


def test_run_db_mapping_tables_and_infer_algorithm():
    from src.run_db import (
        algo_to_agent_type,
        algo_to_checkpoint_prefix,
        infer_algorithm,
    )

    assert algo_to_agent_type("prt-cfr") == "prt_cfr"
    assert algo_to_checkpoint_prefix("prt-cfr") == "prtcfr_checkpoint"
    assert infer_algorithm({"prt_cfr": {"iterations": 300}}) == "prt-cfr"
    assert (
        infer_algorithm({}, checkpoint_filename="prtcfr_snapshot_iter_5.pt")
        == "prt-cfr"
    )


def test_cli_run_dir_discovery_finds_prtcfr_snapshots(tmp_path):
    from pathlib import Path
    from src.cli import _find_run_dir_checkpoints, _checkpoint_matches_epoch

    snapdir = _make_run_dir(tmp_path, [1, 2, 3], best_iteration=3)
    found = _find_run_dir_checkpoints(Path(str(snapdir)), "prtcfr_checkpoint", "prt-cfr")
    names = sorted(p.name for p in found)
    assert names == [
        "prtcfr_snapshot_iter_1.pt",
        "prtcfr_snapshot_iter_2.pt",
        "prtcfr_snapshot_iter_3.pt",
    ]
    # --epoch N matcher picks the snapshot at that iter.
    assert _checkpoint_matches_epoch(found[1], 2, "prt-cfr")
    assert not _checkpoint_matches_epoch(found[1], 3, "prt-cfr")


# ---------------------------------------------------------------------------
# Engine-token-stream feed + end-to-end eval smoke
# ---------------------------------------------------------------------------


def _write_fast_config(tmp_path, max_turns=15):
    base = os.path.join(PROJECT_ROOT, "config", "prtcfr_production.yaml")
    cfgpath = tmp_path / "smoke_config.yaml"
    cfgpath.write_text(f"_base: {base}\ncambia_rules:\n  max_game_turns: {max_turns}\n")
    return str(cfgpath)


def test_observe_transition_grows_token_stream_and_captures_all_frames(tmp_path):
    """observe_transition must append a frame for EVERY applied action (both
    seats) so the full-recall prefix is complete -- including opponent-baseline
    moves the eval loop's public-observation sharing would drop."""
    from src.evaluate_agents import get_agent
    from src.config import load_config
    from src.game.engine import CambiaGameState

    snapdir = _make_run_dir(tmp_path, [1, 2], best_iteration=2)
    cfg = load_config(_write_fast_config(tmp_path))
    agent = get_agent(
        "prt_cfr", player_id=0, config=cfg,
        checkpoint_path=str(snapdir / "prtcfr_snapshot_iter_2.pt"), device="cpu",
    )
    game = CambiaGameState(house_rules=cfg.cambia_rules, seed=13)
    agent.initialize_state(game)
    assert agent._obs_stream == []  # fresh episode

    frames_before = 0
    steps = 0
    while not game.is_terminal() and steps < 200:
        steps += 1
        ap = game.get_acting_player()
        if ap == -1:
            break
        legal = game.get_legal_actions()
        if not legal:
            break
        # Both seats play uniformly at random here; only the feed is under test.
        action = random.choice(list(legal))
        game.apply_action(action)
        agent.observe_transition(game, action, ap)
        assert len(agent._obs_stream) == frames_before + 1  # one frame per action
        frames_before += 1

    assert frames_before > 0
    # The accumulated stream tokenizes into a non-trivial full-recall prefix.
    tokens = agent._encode_tokens()
    assert len(tokens) > 2  # more than just BOS/EOS


def test_end_to_end_eval_smoke(tmp_path):
    from src.evaluate_agents import run_evaluation

    logging.disable(logging.CRITICAL)
    try:
        snapdir = _make_run_dir(tmp_path, [1, 2], best_iteration=2)
        cfgpath = _write_fast_config(tmp_path)
        res = run_evaluation(
            cfgpath,
            "prt_cfr",
            "random_no_cambia",
            4,
            None,
            checkpoint_path=str(snapdir / "prtcfr_snapshot_iter_2.pt"),
            device="cpu",
            crn_seed_base=1,
        )
    finally:
        logging.disable(logging.NOTSET)

    assert res.get("Errors", 0) == 0
    decisive = res.get("P0 Wins", 0) + res.get("P1 Wins", 0)
    total = decisive + res.get("Ties", 0) + res.get("MaxTurnTies", 0)
    assert total == 4
    # T1-Cambia rate (an X4 battery instrument) is attached to the stats.
    assert "t1_cambia_rate" in res.stats


def test_end_to_end_argmax_mode_smoke(tmp_path):
    from src.evaluate_agents import run_evaluation

    logging.disable(logging.CRITICAL)
    try:
        snapdir = _make_run_dir(tmp_path, [1, 2], best_iteration=2)
        cfgpath = _write_fast_config(tmp_path)
        res = run_evaluation(
            cfgpath, "prt_cfr", "random_no_cambia", 2, None,
            checkpoint_path=str(snapdir / "prtcfr_snapshot_iter_2.pt"),
            device="cpu", use_argmax=True, crn_seed_base=5,
        )
    finally:
        logging.disable(logging.NOTSET)
    assert res.get("Errors", 0) == 0
