"""Scoped tests for the PRT-CFR tiny-game trainer.

Covers one training iteration end to end (sigma^t build, K traversals, refit from
scratch, snapshot + checkpoint write) and the SD-CFR snapshot / checkpoint
formats pinned by the interface contract.
"""

import os

import pytest
import torch

from src.config import load_config, PRTCFRConfig
from src.cfr.prtcfr_net import PRTCFRNet
from src.cfr.prtcfr_trainer import (
    NetSigmaProvider,
    PRTCFRTinyTrainer,
    _collect_decision_nodes,
    train_tiny_prtcfr,
)
from tools.tiny_solver import build_tree


CONFIG_2CARD = "config/tiny_2card_plateau.yaml"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_tree():
    cfg = load_config(CONFIG_2CARD)
    root, isets, nnodes, aborted = build_tree(
        cfg, n_deals=5, seed0=0, max_nodes_per_deal=2_000_000,
        enumerate_draws=True, perfect_recall=True, tokenize=True, seq_cap=256,
    )
    assert aborted == 0
    return root


def _fast_config():
    return PRTCFRConfig(
        m_rollouts=2,
        k_games_per_iter=15,
        iterations=2,
        train_steps_per_iter=20,
        batch_size=256,
        device=_DEVICE,
    )


def test_sigma_provider_returns_valid_distributions(tiny_tree):
    net = PRTCFRNet(device=_DEVICE)
    nodes = _collect_decision_nodes(tiny_tree)
    assert nodes
    sigma = NetSigmaProvider(net, nodes, seq_cap=256)
    for node in nodes[:200]:
        p = sigma.policy(node)
        assert p.shape == (len(node.actions),)
        assert abs(p.sum() - 1.0) < 1e-6
        assert (p >= 0).all()


def test_one_iteration_end_to_end_writes_snapshot(tiny_tree, tmp_path):
    cfg = _fast_config()
    snap_dir = str(tmp_path / "snaps")
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, snap_dir)
    st = trainer.run_iteration(t=1)
    assert st.iteration == 1
    assert st.samples_added >= 1
    assert st.buffer_size == st.samples_added
    assert os.path.exists(st.snapshot_path)
    assert os.path.exists(trainer.checkpoint_path())


def test_snapshot_and_checkpoint_formats(tiny_tree, tmp_path):
    cfg = _fast_config()
    snap_dir = str(tmp_path / "snaps")
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, snap_dir)
    trainer.run_iteration(t=1)

    snap = torch.load(trainer.snapshot_path(1), map_location="cpu", weights_only=False)
    assert set(snap.keys()) == {"encoder_state_dict", "head_state_dict", "iteration"}
    assert snap["iteration"] == 1

    ckpt = torch.load(trainer.checkpoint_path(), map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) == {"encoder_state_dict", "head_state_dict", "config", "iteration"}

    # the snapshot must load into a fresh net (the X2 scorer does exactly this)
    net = PRTCFRNet(device="cpu")
    net.load_encoder_head(snap["encoder_state_dict"], snap["head_state_dict"])


def test_buffer_accumulates_across_iterations(tiny_tree, tmp_path):
    cfg = _fast_config()
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, str(tmp_path / "snaps"))
    s1 = trainer.run_iteration(t=1)
    s2 = trainer.run_iteration(t=2)
    # the reservoir is shared and accumulates: iteration 2 sees iteration 1's samples
    assert s2.buffer_size > s1.buffer_size
    assert s2.buffer_size == s1.samples_added + s2.samples_added


def test_warm_start_carries_net_forward(tiny_tree, tmp_path):
    """warm_start gates the per-iteration re-init.

    From-scratch builds a net at init (sigma^1) and re-inits once per iteration:
    1 + 2 = 3 factory calls over 2 iterations. Warm-start builds only the sigma^1
    net and fine-tunes it in place every iteration: 1 call, and ``trainer.net`` is
    the same object after both iterations. The counting factory is the cleanest
    probe since the trainer already accepts ``net_factory``.
    """
    scratch_calls = {"n": 0}
    cfg_scratch = PRTCFRConfig(
        m_rollouts=2, k_games_per_iter=15, iterations=2, train_steps_per_iter=20,
        batch_size=256, warm_start=False, device=_DEVICE,
    )
    t_scratch = PRTCFRTinyTrainer(
        tiny_tree, cfg_scratch, str(tmp_path / "scratch"),
        net_factory=lambda: (scratch_calls.__setitem__("n", scratch_calls["n"] + 1)
                             or PRTCFRNet(device=_DEVICE)),
    )
    t_scratch.run_iteration(t=1)
    t_scratch.run_iteration(t=2)
    assert scratch_calls["n"] == 3

    warm_calls = {"n": 0}
    cfg_warm = PRTCFRConfig(
        m_rollouts=2, k_games_per_iter=15, iterations=2, train_steps_per_iter=20,
        batch_size=256, warm_start=True, device=_DEVICE,
    )
    t_warm = PRTCFRTinyTrainer(
        tiny_tree, cfg_warm, str(tmp_path / "warm"),
        net_factory=lambda: (warm_calls.__setitem__("n", warm_calls["n"] + 1)
                             or PRTCFRNet(device=_DEVICE)),
    )
    t_warm.run_iteration(t=1)
    net_after_iter1 = t_warm.net
    t_warm.run_iteration(t=2)
    assert warm_calls["n"] == 1
    assert t_warm.net is net_after_iter1


def test_train_tiny_prtcfr_entrypoint(tmp_path):
    snap_dir = str(tmp_path / "snaps")
    hist = train_tiny_prtcfr(
        CONFIG_2CARD,
        snap_dir,
        iterations=2,
        config_overrides={
            "m_rollouts": 2,
            "k_games_per_iter": 15,
            "train_steps_per_iter": 20,
            "batch_size": 256,
            "device": _DEVICE,
        },
    )
    assert len(hist) == 2
    assert hist[1].iteration == 2
    assert os.path.exists(os.path.join(snap_dir, "prtcfr_snapshot_iter_1.pt"))
    assert os.path.exists(os.path.join(snap_dir, "prtcfr_snapshot_iter_2.pt"))
    assert os.path.exists(os.path.join(snap_dir, "prtcfr_checkpoint.pt"))
