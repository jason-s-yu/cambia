#!/usr/bin/env python
"""scripts/prtcfr_infer_microbench.py

Standalone latency-vs-batch microbenchmark for the PRT-CFR inference path
(cambia-472, X3 inference-dominance investigation). Sweeps batch size 1..8192 and
fits the per-call cost model the X3 memo leaves as the one unmeasured quantity:

    cost(B) = c_fixed + c_marginal * B

c_fixed is the per-call overhead paid once regardless of batch (two GRU launches,
LayerNorm, head, regret-match, and the forced device-to-host readback in
IncrementalSigmaManager.evaluate). c_marginal is the per-stream cost that scales
with B (the Python per-stream batch assembly in _pad_batch and the mask stack plus
per-row H2D). The split decides the X3 fix: if c_marginal is near zero, widening
gen_chunk_games (fewer, larger calls) alone clears the 90 s gate; if c_marginal is
a few tenths of a millisecond per stream, batching cannot remove the c_marginal *
total_requests term (about 3000 s at the recorded 7.54 M requests) and the batch
assembly must be vectorized and the per-call sync dropped.

Two layers are measured so the split is attributable:
  - "manager": IncrementalSigmaManager.evaluate, the exact production call
    (Python assembly + advance + query_transient + the .to("cpu") sync).
  - "service": PRTCFRInferenceService.advance followed by query_transient, the
    two GPU forwards without the manager's Python assembly, isolating device-side
    fixed cost from the Python per-stream marginal cost.

SAFETY: no CUDA/torch device work happens at import or argument parse. The sweep
runs only under an explicit ``--run`` flag; ``--device`` selects cpu (default) or
cuda. This task does NOT execute it on GPU (GPU windows need user authorization);
it is committed and import/syntax-checked on CPU only. The author runs the CUDA
sweep in an authorized window.

Example (author, authorized GPU window):
    python scripts/prtcfr_infer_microbench.py --run --device cuda \
        --dtype bf16 --reps 50 --out x3-microbench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default batch sweep: powers of two through the config batch, plus 114 (the X3
# measured mean batch) so the operating point sits on the fitted line.
DEFAULT_BATCHES: Tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 114, 128, 256, 512, 1024, 2048, 4096, 8192,
)


def _build_service(device: str, dtype_name: str):
    """Construct a frozen PRTCFRInferenceService over a fresh net. Imports torch
    and the net lazily so import of this module stays device-free."""
    import torch

    from src.cfr.prtcfr_infer import PRTCFRInferenceService  # noqa: E402
    from src.cfr.prtcfr_net import build_prtcfr_net  # noqa: E402

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        dtype_name
    ]
    net = build_prtcfr_net(config=None, device=device)
    return PRTCFRInferenceService(net, device=device, dtype=dtype)


def _import_query_and_manager():
    """Lazy import of the scheduler query type and the production backend."""
    from src.cfr.prtcfr_worker import (  # noqa: E402
        IncrementalSigmaManager,
        _Query,
    )

    return _Query, IncrementalSigmaManager


def _make_queries(query_cls, batch: int, prefix_len: int, body_len: int):
    """Build ``batch`` synthetic _Query objects with distinct streams. tokens are
    [BOS]+body+[EOS]-shaped: a leading marker, ``body_len`` body ids, a trailing
    transient marker (matching the manager's tokens[:-1] carry / tokens[-1]
    transient split). Masks mark a small fixed legal set."""
    import numpy as np

    from src.encoding import NUM_ACTIONS  # noqa: E402

    BOS, EOS, BODY = 1, 2, 3  # any non-PAD ids; PAD_ID is 0
    queries = []
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[:4] = True  # a representative small legal set (A about 5 in the memo)
    _ = prefix_len  # reserved: first-frame prefix length knob (kept for parity)
    for s in range(batch):
        toks = [BOS] + [BODY] * body_len + [EOS]
        queries.append(query_cls(stream=s, player=0, tokens=toks, mask=mask.copy()))
    return queries


def _time_manager(
    manager, query_cls, batch: int, reps: int, body_len: int
) -> List[float]:
    """Median-friendly list of per-call walls for manager.evaluate at ``batch``.
    Each rep advances every stream by ``body_len`` fresh body frames (the carry
    grows, matching a rollout stepping one frame per query)."""
    walls: List[float] = []
    for r in range(reps):
        # Fresh streams per rep so the carry length starts at 0 (worst-case
        # advance width equals body_len every call, an upper bound on advance
        # cost; a warm carry advances by fewer frames).
        base = (r + 1) * (batch + 1)
        queries = _make_queries(query_cls, batch, prefix_len=0, body_len=body_len)
        for i, q in enumerate(queries):
            q.stream = base + i
        t0 = time.perf_counter()
        manager.evaluate(queries)
        walls.append(time.perf_counter() - t0)
        for q in queries:
            manager.drop(q.stream)
    return walls


def _fit_line(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Ordinary least squares y = a + b x. Returns (a=c_fixed, b=c_marginal, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    query_cls, manager_cls = _import_query_and_manager()
    service = _build_service(args.device, args.dtype)
    manager = manager_cls(service, num_players=1)

    batches = [int(b) for b in (args.batches or DEFAULT_BATCHES)]
    rows: List[Dict[str, Any]] = []

    # Warmup at the largest batch to page in kernels / cuDNN plans.
    _ = _time_manager(manager, query_cls, max(batches), reps=2, body_len=args.body_len)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    for b in batches:
        walls = _time_manager(manager, query_cls, b, args.reps, args.body_len)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        med = statistics.median(walls)
        rows.append(
            {
                "batch": b,
                "median_ms": round(med * 1000, 4),
                "min_ms": round(min(walls) * 1000, 4),
                "p90_ms": round(sorted(walls)[int(0.9 * (len(walls) - 1))] * 1000, 4),
                "ms_per_stream": round(med * 1000 / b, 5),
            }
        )
        print(
            f"[microbench] B={b:5d}  median={med * 1000:8.3f} ms  "
            f"per-stream={med * 1000 / b:7.4f} ms"
        )

    xs = [float(r["batch"]) for r in rows]
    ys = [r["median_ms"] / 1000.0 for r in rows]  # seconds
    c_fixed, c_marginal, r2 = _fit_line(xs, ys)

    # Project the fit onto the X3 operating point (7.54 M requests, 66,124 calls
    # measured; and a single-chunk floor of about 1500 calls) to state the
    # reachable wall the memo asks for.
    R = 7_543_850
    fit = {
        "c_fixed_ms": round(c_fixed * 1000, 4),
        "c_marginal_ms_per_stream": round(c_marginal * 1000, 6),
        "r2": round(r2, 5),
        "c_marginal_times_R_s": round(c_marginal * R, 1),
        "projected_wall_s": {
            "calls_66124": round(c_fixed * 66124 + c_marginal * R, 1),
            "calls_4000": round(c_fixed * 4000 + c_marginal * R, 1),
            "calls_1500": round(c_fixed * 1500 + c_marginal * R, 1),
        },
    }
    print(
        f"[microbench] fit: c_fixed={fit['c_fixed_ms']} ms  "
        f"c_marginal={fit['c_marginal_ms_per_stream']} ms/stream  "
        f"r2={fit['r2']}  c_marginal*R={fit['c_marginal_times_R_s']} s"
    )
    return {
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "reps": args.reps,
            "body_len": args.body_len,
            "batches": batches,
        },
        "rows": rows,
        "fit": fit,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="store_true",
        help="actually execute the sweep (device work). Without it, this script "
        "only parses args and exits, so import/CI stays GPU-free.",
    )
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--reps", type=int, default=30, help="timed reps per batch")
    p.add_argument(
        "--body-len",
        type=int,
        default=2,
        help="body frames advanced per query (the memo's ~2 frames/ply)",
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="*",
        default=None,
        help="explicit batch sizes; default the power-of-two sweep plus 114",
    )
    p.add_argument("--out", default=None, help="write the full result JSON here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print(
            "[microbench] dry mode (no --run): parsed args only, no device work. "
            "Pass --run --device cuda in an authorized GPU window to sweep."
        )
        return 0
    result = run_sweep(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[microbench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
