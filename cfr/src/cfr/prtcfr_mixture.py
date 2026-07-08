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
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from ..sequence_encoding import BOS_ID, EOS_ID, PAD_ID
from .prtcfr_net import PRTCFRNet, _regret_match
from .prtcfr_worker import PRODUCTION_SEQ_CAP
from .prtcfr_stability import read_deployable_iters

# A mixture "slot" is either an already-materialized net (eager, e.g. tests
# constructing PRTCFRMixture directly) or a zero-arg loader that builds one
# on first use (lazy, the ``from_checkpoint`` path -- cambia-249).
NetOrLoader = Union[PRTCFRNet, Callable[[], PRTCFRNet]]

DEFAULT_LAZY_CACHE_SIZE = 8

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

    Snapshot LOADING is lazy (cambia-249): ``nets`` may hold already-built
    ``PRTCFRNet`` instances (eager, e.g. tests constructing the mixture
    directly) or zero-arg loader callables (``from_checkpoint``'s path) that
    are invoked only the first time their index is actually sampled and
    queried, via ``active_net()``. Loaded nets are cached (an LRU of
    ``lazy_cache_size`` entries) so a short eval that only ever samples a
    handful of the deployable snapshots never pays the full-run load cost;
    eagerly-provided nets are never evicted (there is no loader to re-invoke).
    SD-CFR weighting (``self.weights``, computed from ``snapshot_iters`` alone)
    and per-episode sampling are unaffected: both depend only on iteration
    numbers, never on the loaded net objects.
    """

    def __init__(
        self,
        snapshot_iters: Sequence[int],
        nets: Sequence[NetOrLoader],
        weighting: str = "linear",
        seq_cap: int = PRODUCTION_SEQ_CAP,
        device: str = "cpu",
        lazy_cache_size: int = DEFAULT_LAZY_CACHE_SIZE,
    ):
        if len(snapshot_iters) != len(nets) or not nets:
            raise ValueError(
                "PRTCFRMixture: snapshot_iters and nets must align and be non-empty"
            )
        self.iters: List[int] = list(snapshot_iters)
        self._slots: List[NetOrLoader] = list(nets)
        self.weighting = weighting
        self.weights = _sd_cfr_weights(self.iters, weighting)
        self.seq_cap = int(seq_cap)
        self.device = torch.device(device)
        self._active_idx: Optional[int] = None
        self._lazy_cache_size = max(1, int(lazy_cache_size))
        # LRU of index -> loaded net, for slots that came in as loader callables.
        self._loaded: "OrderedDict[int, PRTCFRNet]" = OrderedDict()

    # -- lazy resolution (cambia-249) ---------------------------------------

    def is_loaded(self, idx: int) -> bool:
        """True if slot ``idx`` is an eagerly-provided net or has already been
        loaded (and is still cache-resident) via a prior ``_resolve``."""
        slot = self._slots[idx]
        return isinstance(slot, PRTCFRNet) or idx in self._loaded

    def _resolve(self, idx: int) -> PRTCFRNet:
        slot = self._slots[idx]
        if isinstance(slot, PRTCFRNet):
            return slot
        if idx in self._loaded:
            self._loaded.move_to_end(idx)
            return self._loaded[idx]
        net = slot()  # invoke the loader: torch.load + _build_net_from_state
        self._loaded[idx] = net
        self._loaded.move_to_end(idx)
        if len(self._loaded) > self._lazy_cache_size:
            self._loaded.popitem(last=False)  # evict least-recently-used
        return net

    # -- SD-CFR per-episode sampling ---------------------------------------

    def sample_episode(self, rng: np.random.Generator) -> int:
        """Draw one snapshot index for the coming episode (proportional to w_t).

        Returns the chosen index into ``self._slots``/``self.iters``. The whole
        episode is then played with ``active_net()``; a new episode calls this
        again. This is the SD-CFR realization: one iterate per trajectory,
        sampled proportional to its weight -- NOT a per-decision average. Does
        NOT trigger a load: sampling depends only on ``self.weights`` (from
        iteration numbers), never on the net objects.
        """
        self._active_idx = int(rng.choice(len(self._slots), p=self.weights))
        return self._active_idx

    def active_iter(self) -> int:
        if self._active_idx is None:
            raise RuntimeError("PRTCFRMixture.active_iter: call sample_episode() first")
        return self.iters[self._active_idx]

    def active_net(self) -> PRTCFRNet:
        """The sampled episode's net, loading it from disk on first use if the
        slot is still a lazy loader (cambia-249)."""
        if self._active_idx is None:
            raise RuntimeError("PRTCFRMixture.active_net: call sample_episode() first")
        return self._resolve(self._active_idx)

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

    @staticmethod
    def _make_loader(snap_path: str, device: str) -> Callable[[], PRTCFRNet]:
        """A zero-arg loader closure: defers ``torch.load`` + net construction
        until actually invoked by ``_resolve`` (cambia-249 lazy loading)."""

        def _load() -> PRTCFRNet:
            enc, head, _it = _load_snapshot_file(snap_path)
            return _build_net_from_state(enc, head, device)

        return _load

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        weighting: str = "linear",
        seq_cap: int = PRODUCTION_SEQ_CAP,
        max_iter: Optional[int] = None,
        lazy_cache_size: int = DEFAULT_LAZY_CACHE_SIZE,
    ) -> "PRTCFRMixture":
        """Build the deployable mixture for a PRT-CFR run.

        ``checkpoint_path`` may point at any file inside the run's snapshot dir
        (a per-iteration snapshot or the rolling ``prtcfr_checkpoint.pt``); its
        DIRECTORY is the snapshot dir. The deployable window comes from
        ``prtcfr_deployable.json`` (``read_deployable_iters``); with no manifest,
        every ``prtcfr_snapshot_iter_{t}.pt`` in the dir is deployable (the
        unpinned default). ``max_iter`` further restricts the window to iters
        ``<= max_iter`` (the ``--epoch N`` "mixture as of iter N" query).

        Snapshot files are NOT read here (cambia-249): each deployable iter
        becomes a lazy loader closure, and ``self.iters`` is populated from the
        DEPLOYABLE ITERATION NUMBER ITSELF (the manifest/filename ``t``), not
        the value stored inside the file. In every real snapshot this equals
        the file's own ``iteration`` field (snapshots are always written as
        ``prtcfr_snapshot_iter_{t}.pt`` with ``"iteration": t``); reading it
        from the filename avoids opening every file just to confirm what its
        name already promises, which is the whole point of deferring the load.
        SD-CFR weighting depends only on this iters list, so sampling semantics
        are unchanged. ``lazy_cache_size`` bounds how many loaded nets are kept
        resident (LRU) across episodes that sample different snapshots.

        Degenerate fallback: if the dir holds no per-iteration snapshots but
        ``checkpoint_path`` itself carries encoder/head state (e.g. a lone
        rolling checkpoint or a tiny test fixture), it becomes a single-snapshot
        mixture; this rare path loads eagerly since it is a single well-defined
        file, not one further per-episode reload risk.
        """
        snapshot_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        deployable = read_deployable_iters(snapshot_dir)
        if deployable is None:
            deployable = _discover_snapshot_iters(snapshot_dir)
        if max_iter is not None:
            deployable = [t for t in deployable if t <= max_iter]

        iters: List[int] = []
        loaders: List[NetOrLoader] = []
        for t in sorted(deployable):
            snap_path = os.path.join(snapshot_dir, f"prtcfr_snapshot_iter_{t}.pt")
            if not os.path.isfile(snap_path):
                continue
            iters.append(t)
            loaders.append(cls._make_loader(snap_path, device))

        if not loaders:
            # No per-iteration snapshots resolved: fall back to the pointed-at
            # file as a single-snapshot mixture (degenerate but valid). Loaded
            # eagerly (one file) so the true stored iteration is known.
            enc, head, it = _load_snapshot_file(checkpoint_path)
            loaders.append(_build_net_from_state(enc, head, device))
            iters.append(it if it > 0 else 1)

        return cls(
            iters, loaders, weighting=weighting, seq_cap=seq_cap, device=device,
            lazy_cache_size=lazy_cache_size,
        )

    def __len__(self) -> int:
        return len(self._slots)


# ---------------------------------------------------------------------------
# Incremental per-episode GRU carry (cambia-249: O(1)-amortized eval queries)
# ---------------------------------------------------------------------------


class PRTCFRIncrementalCursor:
    """Per-episode incremental GRU carry for one sampled PRT-CFR net.

    ``PRTCFRAgentWrapper.choose_action`` previously re-tokenized and re-ran
    the GRU over the WHOLE accumulated observation prefix at every decision --
    O(n) work per query, O(n^2) per game. This cursor instead feeds only the
    BODY tokens (``sequence_encoding.initial_peek_frames`` once, then
    ``observation_frames`` per newly-appended observation) appended since the
    last query through the net's GRU from the carried raw
    (num_layers, 1, hidden) hidden state -- an O(1)-amortized append per
    decision. LayerNorm and the head are read-time ops applied only in
    ``query`` and are never part of the carried state, matching
    ``PRTCFRNet.encode``'s own read-time LayerNorm placement.

    ``query`` appends a TRANSIENT EOS token (matching
    ``encode_observation_sequence``'s trailing EOS) to compute the queried
    hidden state WITHOUT persisting it, so the next ``advance`` call appends
    cleanly onto the un-EOS'd carry -- the same transient-suffix technique as
    ``prtcfr_infer.PRTCFRInferenceService.query_transient``.

    Truncation (seq-cap keep-most-recent) invalidates the simple append-only
    recurrence: a truncating re-encode can drop OLDEST frames, which the
    carried hidden state has already irreversibly absorbed. ``advance`` sets
    ``overflowed`` and stops updating the carry the instant the accumulated
    body would exceed the cap, at EXACTLY the ``budget = seq_cap - 2`` (BOS +
    EOS) threshold ``encode_observation_sequence(strict=True)`` raises
    ``SequenceOverflowError`` at. Callers MUST fall back to a full stateless
    re-encode once ``overflowed`` is set (rare on natural-length games;
    correctness over speed per cambia-249).
    """

    def __init__(self, net: PRTCFRNet, seq_cap: int):
        self.net = net
        self.seq_cap = int(seq_cap)
        self._hidden: Optional[torch.Tensor] = None  # (num_layers, 1, hidden)
        self._body_len = 0
        self._registered = False
        self.overflowed = False

    @property
    def registered(self) -> bool:
        """True once ``advance`` has been called at least once this episode
        (the leading BOS has been fed)."""
        return self._registered

    @torch.no_grad()
    def advance(self, new_body_tokens: Sequence[int]) -> None:
        """Feed newly-appended BODY tokens (no BOS/EOS) through the GRU from
        the carried hidden state. The first call additionally feeds the
        leading BOS. No-op once ``overflowed``."""
        if self.overflowed:
            return
        prospective = self._body_len + len(new_body_tokens)
        if prospective + 2 > self.seq_cap:  # +2 = BOS + EOS budget, matches
            self.overflowed = True          # encode_observation_sequence's own cap.
            return
        chunk = list(new_body_tokens)
        if not self._registered:
            chunk = [BOS_ID] + chunk
            self._registered = True
        if not chunk:
            return
        t = torch.as_tensor(chunk, dtype=torch.long, device=self.net.device).unsqueeze(0)
        self._hidden = self.net.step_hidden(t, self._hidden)
        self._body_len = prospective

    @torch.no_grad()
    def query(self, legal_mask: np.ndarray) -> np.ndarray:
        """Regret-matched strategy (146,) from the carry plus a transient EOS
        suffix (not persisted). Raises if ``advance`` was never called or the
        cursor has overflowed -- callers must check ``overflowed`` and fall
        back to a full re-encode themselves rather than call this."""
        if self._hidden is None or self.overflowed:
            raise RuntimeError(
                "PRTCFRIncrementalCursor.query: no valid carried hidden state "
                "(call advance() first; if overflowed, the caller must fall "
                "back to a full stateless re-encode instead of calling query)"
            )
        eos = torch.as_tensor([EOS_ID], dtype=torch.long, device=self.net.device).unsqueeze(0)
        h_n = self.net.step_hidden(eos, self._hidden)  # transient: not stored back
        top = h_n[-1]  # (1, hidden) top-layer hidden after the transient EOS
        mask_t = torch.as_tensor(
            np.asarray(legal_mask, dtype=bool), device=self.net.device
        ).unsqueeze(0)
        strat = self.net.strategy_from_hidden(top, mask_t)
        return strat[0].detach().to("cpu", dtype=torch.float64).numpy()
