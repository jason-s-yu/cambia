"""src/cfr/prtcfr_mixture.py

SD-CFR snapshot mixture for PRT-CFR evaluation (v0.4 Phase 2, cambia-233).

PRT-CFR keeps NO strategy network: the average strategy is the guaranteed CFR
object, realized EXACTLY by SD-CFR snapshot sampling (v0.4 design decision 4;
design-overview.md: "sample iterate tau proportional to tau at game start").
This module is the deployment-time realizer of that object for FULL-GAME play:

  - ``PRTCFRMixture.from_checkpoint`` reads a run dir's deployable window (the
    S1W1 ``prtcfr_deployable.json`` manifest seam via
    ``prtcfr_stability.read_deployable_iters``), loads each deployable
    per-iteration regret-net snapshot, and computes the SD-CFR linear
    (``w_t = t``) snapshot weights.
  - ``sample_episode`` draws ONE snapshot per episode with probability
    proportional to its weight; the whole game is then played with that single
    snapshot's regret-matched policy. This is the SD-CFR trajectory-sampling
    procedure (Steinberger 2019): committing to one iterate for the whole game
    and sampling the iterate proportional to ``w_t`` produces trajectories
    distributed by the true average strategy sigma-bar -- the reach weighting
    emerges from the sampling, so no per-infoset reach term is needed.

Per-DECISION averaging of the snapshots (the naive weighted mean of per-net
strategies used by the deterministic tiny-game X2 scorer ``prtcfr_eval.py``) is
a DIFFERENT, reach-unweighted object and is deliberately NOT done here for
full-game play: it does not sample from sigma-bar. Mixture faithfulness is
gate-critical for X4, so this module realizes SD-CFR by sampling, never by
averaging.

The policy query itself (tokens + legal mask -> masked regret-matched strategy
over the 146 global actions) is byte-identical to the training-time sigma^t
(``prtcfr_trainer.NetProductionSigma``) and to
``prtcfr_infer.PRTCFRInferenceService.strategy``: all three route the snapshot
net's advantages through the single-sourced ``prtcfr_net._regret_match``. The
module has no dependency on ``evaluate_agents`` (the wrapper imports FROM here),
so the mixture math is unit-testable in isolation.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..sequence_encoding import PAD_ID
from .prtcfr_net import PRTCFRNet, _regret_match
from .prtcfr_worker import PRODUCTION_SEQ_CAP
from .prtcfr_stability import read_deployable_iters

SNAPSHOT_GLOB = "prtcfr_snapshot_iter_*.pt"
_SNAPSHOT_ITER_RE = re.compile(r"prtcfr_snapshot_iter_(\d+)\.pt$")


def _sd_cfr_weights(iters: Sequence[int], weighting: str = "linear") -> np.ndarray:
    """SD-CFR snapshot weights over ``iters``, normalized to sum 1.

    ``linear`` (the PRT-CFR default, design decision 4): ``w_t = t`` -- the
    iterate is sampled proportional to its iteration number, exactly the linear
    (``w_t = t``) CFR averaging weight the served average strategy is defined
    with. ``uniform``: equal weight. A degenerate all-zero/empty weight vector
    falls back to uniform so sampling never divides by zero.
    """
    it = list(iters)
    if not it:
        raise ValueError("_sd_cfr_weights: empty iteration list")
    if weighting == "uniform":
        w = np.ones(len(it), dtype=np.float64)
    elif weighting == "linear":
        w = np.array([max(float(t), 0.0) for t in it], dtype=np.float64)
    else:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected 'linear' or 'uniform'"
        )
    total = w.sum()
    if total <= 0:
        w = np.ones(len(it), dtype=np.float64)
        total = w.sum()
    return w / total


def _infer_arch_from_state(encoder_sd: dict, head_sd: dict) -> dict:
    """Recover the PRTCFRNet constructor dims from a snapshot's state dicts.

    The pinned snapshot format ({encoder_state_dict, head_state_dict,
    iteration}) carries no architecture config, so the shapes are read straight
    off the weights. This keeps the mixture independent of any config file and
    correct for a non-default-width run.
    """
    embed_w = encoder_sd["embed.weight"]
    vocab_size, embed_dim = int(embed_w.shape[0]), int(embed_w.shape[1])
    hidden_dim = int(encoder_sd["norm.weight"].shape[0])
    num_layers = sum(1 for k in encoder_sd if re.fullmatch(r"gru\.weight_ih_l\d+", k))
    head_hidden_dim = int(head_sd["0.weight"].shape[0])
    num_actions = int(head_sd["2.weight"].shape[0])
    return {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": max(num_layers, 1),
        "head_hidden_dim": head_hidden_dim,
        "num_actions": num_actions,
    }


def _build_net_from_state(encoder_sd: dict, head_sd: dict, device: str) -> PRTCFRNet:
    """Construct a PRTCFRNet with dims inferred from the state dicts and load it."""
    arch = _infer_arch_from_state(encoder_sd, head_sd)
    net = PRTCFRNet(
        vocab_size=arch["vocab_size"],
        embed_dim=arch["embed_dim"],
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        head_hidden_dim=arch["head_hidden_dim"],
        num_actions=arch["num_actions"],
        dropout=0.0,  # eval: dropout off (matches the inference service carry identity)
        device=device,
    )
    net.load_encoder_head(encoder_sd, head_sd)
    net.eval()
    return net


def _load_snapshot_file(path: str):
    """Return (encoder_sd, head_sd, iteration) from a snapshot / rolling checkpoint.

    Accepts both the per-iteration snapshot ({encoder_state_dict,
    head_state_dict, iteration}) and the rolling ``prtcfr_checkpoint.pt`` (same
    keys plus ``config``); either exposes the two state dicts this needs.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "encoder_state_dict" not in data or "head_state_dict" not in data:
        raise ValueError(
            f"{path}: not a PRT-CFR snapshot (missing encoder/head state dict)"
        )
    return (
        data["encoder_state_dict"],
        data["head_state_dict"],
        int(data.get("iteration", 0)),
    )


def _discover_snapshot_iters(snapshot_dir: str) -> List[int]:
    """All iterations with a ``prtcfr_snapshot_iter_{t}.pt`` file in ``snapshot_dir``."""
    iters = []
    for p in glob.glob(os.path.join(snapshot_dir, SNAPSHOT_GLOB)):
        m = _SNAPSHOT_ITER_RE.search(os.path.basename(p))
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


class PRTCFRMixture:
    """Deployable SD-CFR snapshot mixture with per-episode iterate sampling.

    Holds the deployable snapshot nets and their SD-CFR weights. ``sample_episode``
    selects one net for the coming game; ``strategy`` queries THAT net. A fresh
    game re-samples. The mixture is stateless with respect to the token stream
    (the wrapper owns that); it only owns the snapshot selection + policy query.
    """

    def __init__(
        self,
        snapshot_iters: Sequence[int],
        nets: Sequence[PRTCFRNet],
        weighting: str = "linear",
        seq_cap: int = PRODUCTION_SEQ_CAP,
        device: str = "cpu",
    ):
        if len(snapshot_iters) != len(nets) or not nets:
            raise ValueError(
                "PRTCFRMixture: snapshot_iters and nets must align and be non-empty"
            )
        self.iters: List[int] = list(snapshot_iters)
        self.nets: List[PRTCFRNet] = list(nets)
        self.weighting = weighting
        self.weights = _sd_cfr_weights(self.iters, weighting)
        self.seq_cap = int(seq_cap)
        self.device = torch.device(device)
        self._active_idx: Optional[int] = None

    # -- SD-CFR per-episode sampling ---------------------------------------

    def sample_episode(self, rng: np.random.Generator) -> int:
        """Draw one snapshot index for the coming episode (proportional to w_t).

        Returns the chosen index into ``self.nets``/``self.iters``. The whole
        episode is then played with ``active_net()``; a new episode calls this
        again. This is the SD-CFR realization: one iterate per trajectory,
        sampled proportional to its weight -- NOT a per-decision average.
        """
        self._active_idx = int(rng.choice(len(self.nets), p=self.weights))
        return self._active_idx

    def active_iter(self) -> int:
        if self._active_idx is None:
            raise RuntimeError("PRTCFRMixture.active_iter: call sample_episode() first")
        return self.iters[self._active_idx]

    def active_net(self) -> PRTCFRNet:
        if self._active_idx is None:
            raise RuntimeError("PRTCFRMixture.active_net: call sample_episode() first")
        return self.nets[self._active_idx]

    # -- policy query (byte-identical to training sigma^t) -----------------

    def strategy(self, tokens: Sequence[int], legal_mask: np.ndarray) -> np.ndarray:
        """Regret-matched strategy (146,) of the ACTIVE snapshot for ``tokens``.

        Mirrors ``prtcfr_trainer.NetProductionSigma.__call__`` exactly: keep-most-
        recent truncation to ``seq_cap`` (matching the sigma's own overflow
        tolerance), one forward, ``_regret_match`` masked normalization. Uses the
        episode's single sampled net -- never averages across snapshots.
        """
        net = self.active_net()
        toks = list(tokens) if len(tokens) else [PAD_ID]
        if len(toks) > self.seq_cap:
            toks = toks[-self.seq_cap :]
        t = torch.as_tensor(toks, dtype=torch.long, device=net.device).unsqueeze(0)
        mask_arr = np.asarray(legal_mask, dtype=bool)
        m = torch.as_tensor(mask_arr, device=net.device).unsqueeze(0)
        with torch.no_grad():
            adv = net.raw_advantages(t, m)
            strat = _regret_match(adv, m)  # (1, 146)
        return strat[0].detach().to("cpu", dtype=torch.float64).numpy()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        weighting: str = "linear",
        seq_cap: int = PRODUCTION_SEQ_CAP,
        max_iter: Optional[int] = None,
    ) -> "PRTCFRMixture":
        """Build the deployable mixture for a PRT-CFR run.

        ``checkpoint_path`` may point at any file inside the run's snapshot dir
        (a per-iteration snapshot or the rolling ``prtcfr_checkpoint.pt``); its
        DIRECTORY is the snapshot dir. The deployable window comes from
        ``prtcfr_deployable.json`` (``read_deployable_iters``); with no manifest,
        every ``prtcfr_snapshot_iter_{t}.pt`` in the dir is deployable (the
        unpinned default). ``max_iter`` further restricts the window to iters
        ``<= max_iter`` (the ``--epoch N`` "mixture as of iter N" query).

        Degenerate fallback: if the dir holds no per-iteration snapshots but
        ``checkpoint_path`` itself carries encoder/head state (e.g. a lone
        rolling checkpoint or a tiny test fixture), it becomes a single-snapshot
        mixture.
        """
        snapshot_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        deployable = read_deployable_iters(snapshot_dir)
        if deployable is None:
            deployable = _discover_snapshot_iters(snapshot_dir)
        if max_iter is not None:
            deployable = [t for t in deployable if t <= max_iter]

        iters: List[int] = []
        nets: List[PRTCFRNet] = []
        for t in sorted(deployable):
            snap_path = os.path.join(snapshot_dir, f"prtcfr_snapshot_iter_{t}.pt")
            if not os.path.isfile(snap_path):
                continue
            enc, head, it = _load_snapshot_file(snap_path)
            nets.append(_build_net_from_state(enc, head, device))
            iters.append(it if it > 0 else t)

        if not nets:
            # No per-iteration snapshots resolved: fall back to the pointed-at
            # file as a single-snapshot mixture (degenerate but valid).
            enc, head, it = _load_snapshot_file(checkpoint_path)
            nets.append(_build_net_from_state(enc, head, device))
            iters.append(it if it > 0 else 1)

        return cls(iters, nets, weighting=weighting, seq_cap=seq_cap, device=device)

    def __len__(self) -> int:
        return len(self.nets)
