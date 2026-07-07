"""scripts/prtcfr_token_dist.py

S1W15 instrumentation (cambia-239, scope item 3): report the TRAINING
token-length distribution produced by PRT-CFR production generation over a
K>=256 near-uniform cell.

This resolves the open X3 fit-VRAM question. The P100 instrumentation
(scripts/prtcfr_p100_instrument.py) measured TERMINAL token lengths and found
two cohorts an order of magnitude apart: "natural" uniform play (including
random CallCambia) mean ~64, vs "avoid_cambia" (skips CallCambia, runs to the
300-turn cap) mean ~3217. The reservoir footprint and the per-iteration fit
batch VRAM are sized by the TRAINING prefix distribution -- the length of each
``tokens_h`` stored as a regret-sample feature at a TRAVERSER decision -- not by
terminal length. Production b_i is uniform over ALL legal actions (CallCambia
included), so this measures which cohort production actually generates.

The generation path is the real one: PRTCFRBatchedProductionWorker with a fresh
(near-uniform) PRTCFRNet as sigma^t, driving the production GameDriver. It
records, per stored regret sample, the natural (unpadded) length of the
decision prefix, plus each game's terminal per-player token length for
comparison with the P100 numbers.

Usage (from cfr/):
    python scripts/prtcfr_token_dist.py --k-games 256 --m-rollouts 4 \
        --backend python --seq-cap 12288
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402
from src.cfr.prtcfr_worker import (  # noqa: E402
    IncrementalSigmaManager,
    PRODUCTION_SEQ_CAP,
    PRTCFRBatchedProductionWorker,
    _FullReencodeSigmaBackend,
    _legal_mask,
    _sample_and_apply,
    new_production_driver,
    uniform_policy_production,
)
from src.reservoir import ReservoirSample  # noqa: E402
from src.sequence_encoding import PAD_ID  # noqa: E402


class _LenBuf:
    """A reservoir stand-in that records the natural (unpadded) length of every
    stored regret-sample feature (the decision-point prefix length)."""

    def __init__(self):
        self.lengths: list = []

    def add(self, sample: ReservoirSample) -> None:
        feats = np.asarray(sample.features)
        nz = np.nonzero(feats != PAD_ID)[0]
        self.lengths.append(int(nz[-1]) + 1 if nz.size else 0)

    def __len__(self) -> int:
        return len(self.lengths)


def _dist(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
        "max": int(a.max()),
        "min": int(a.min()),
    }


def _terminal_lengths(seed0: int, n_games: int, backend: str, seq_cap: int) -> list:
    """Play n_games under production b_i (uniform incl. CallCambia) to terminal
    and record each player's terminal token-body length."""
    out = []
    import random

    for k in range(n_games):
        driver = new_production_driver(seed=seed0 + k, backend=backend)
        if hasattr(driver, "seq_cap"):
            driver.seq_cap = seq_cap
        rng = random.Random(1_000 + seed0 + k)
        steps = 0
        while not driver.is_terminal() and steps < 4000:
            steps += 1
            actor = driver.current_player()
            if actor == -1:
                break
            legal = driver.legal_actions()
            if not legal:
                break
            mask = _legal_mask(legal)
            try:
                _sample_and_apply(driver, legal, uniform_policy_production([], mask), rng)
            except RuntimeError:
                break
        for p in range(2):
            try:
                out.append(len(driver.tokens(p)))
            except Exception:
                pass
        driver.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-games", type=int, default=256)
    ap.add_argument("--m-rollouts", type=int, default=4)
    ap.add_argument("--seq-cap", type=int, default=PRODUCTION_SEQ_CAP)
    ap.add_argument("--backend", default="python", choices=["python", "go"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="use the IncrementalSigmaManager backend (bf16/fp32 carry) instead "
        "of the full-reencode reference; distribution is backend-independent, "
        "this only exercises the production inference path end to end.",
    )
    ap.add_argument("--out", default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    net = PRTCFRNet(
        embed_dim=64, hidden_dim=256, num_layers=2, head_hidden_dim=256,
        dropout=0.0, device="cpu",
    ).eval()

    if args.incremental:
        import torch

        from src.cfr.prtcfr_infer import PRTCFRInferenceService

        svc = PRTCFRInferenceService(net, device="cpu", dtype=torch.float32)
        backend = IncrementalSigmaManager(svc, num_players=2)
    else:
        backend = _FullReencodeSigmaBackend(net, seq_cap=args.seq_cap)

    buf = _LenBuf()
    specs = []
    for k in range(args.k_games):
        traverser = k % 2
        driver = new_production_driver(seed=args.seed + k, backend=args.backend)
        if hasattr(driver, "seq_cap"):
            driver.seq_cap = args.seq_cap
        specs.append(
            {
                "seed": args.seed + 7_000_003 + k * 2_000_029,
                "driver": driver,
                "traverser": traverser,
                "iteration": 1,
                "buf": buf,
            }
        )
    worker = PRTCFRBatchedProductionWorker(
        m_rollouts=args.m_rollouts, seq_cap=args.seq_cap, max_trajectory_steps=4000
    )
    worker.generate(specs, backend)
    for spec in specs:
        spec["driver"].close()

    terminal = _terminal_lengths(args.seed, args.k_games, args.backend, args.seq_cap)

    report = {
        "cell": {
            "k_games": args.k_games,
            "m_rollouts": args.m_rollouts,
            "seq_cap": args.seq_cap,
            "backend": args.backend,
            "policy": "near-uniform (fresh-net sigma^t at opponent nodes, b_i "
            "uniform incl. CallCambia at traverser nodes)",
        },
        "training_prefix_len": _dist(buf.lengths),
        "terminal_body_len": _dist(terminal),
    }
    # Fit-VRAM implication: a fit batch pads to the longest row in the batch.
    tp = report["training_prefix_len"]
    if tp.get("n"):
        # bytes = batch * pad_width * 8 (int64 token ids on GPU) for one batch.
        for bs in (8192,):
            report.setdefault("fit_batch_vram_estimate", {})[str(bs)] = {
                "pad_to_p99_MB": round(bs * tp["p99"] * 8 / 1e6, 1),
                "pad_to_max_MB": round(bs * tp["max"] * 8 / 1e6, 1),
            }
        # reservoir footprint at 20M rows, ragged int16 storage.
        report["reservoir_20M_int16_GB"] = round(20_000_000 * tp["mean"] * 2 / 1e9, 1)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
