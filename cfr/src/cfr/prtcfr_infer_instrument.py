"""src/cfr/prtcfr_infer_instrument.py

Non-invasive instrumentation for the PRT-CFR batched inference path (cambia-472,
X3 inference-dominance investigation). Wraps
``IncrementalSigmaManager.evaluate`` -- the SigmaBackend seam the scheduler calls
once per tick -- to record, per call:

  - batch size (len(queries)): the per-call request count, so the batch-size
    HISTOGRAM (not just the mean) is observable. The X3 derivation put the mean
    at about 114 against a config batch of 8192; the histogram shows the sawtooth
    (a traverser decision's rollouts open the batch, then it drains as rollouts
    terminate at staggered depths).
  - queue wait: wall between the end of the previous call and the start of this
    one (the gap the scheduler spends pumping coroutines and stepping the Go
    engine between inference calls). Large gaps at small batch are the
    call-count-bound signature.
  - GPU-util timeline: an optional coarse ``nvidia-smi`` sample tagged with the
    call index, so util-over-time lines up against batch size (the X3 cell sat at
    5.3% mean util, the occupancy-starved signature this confirms in situ).

Design: this NEVER edits prtcfr_infer.py. ``instrument_evaluate`` is a context
manager that monkeypatches the method for the duration of a ``with`` block and
restores it on exit (mirrors prtcfr_bench.py's temporary-patch discipline). Zero
import-time side effects; safe to import on CPU with no torch GPU call. The GPU
sampler shells out to ``nvidia-smi`` and is opt-in (``gpu_util=True``); with it
off, the module touches no device at all.

Usage (author, on a GPU window, NOT run in this task):

    from src.cfr.prtcfr_infer_instrument import instrument_evaluate
    with instrument_evaluate(gpu_util=True) as rec:
        trainer.run_iteration(t)
    rec.dump_json("x3-evaluate-trace.json")
    rec.print_summary()
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluateTrace:
    """Per-call records for one instrumented generation phase.

    Parallel lists indexed by call number: ``batch_sizes[i]`` is the request
    count of call i, ``call_wall_s[i]`` its inference wall, ``queue_wait_s[i]``
    the gap before it, ``gpu_util_pct[i]`` a coarse util sample (NaN when GPU
    sampling is off or a sample failed).
    """

    batch_sizes: List[int] = field(default_factory=list)
    call_wall_s: List[float] = field(default_factory=list)
    queue_wait_s: List[float] = field(default_factory=list)
    gpu_util_pct: List[float] = field(default_factory=list)

    def add(
        self,
        batch: int,
        call_wall: float,
        queue_wait: float,
        gpu_util: float = float("nan"),
    ) -> None:
        self.batch_sizes.append(int(batch))
        self.call_wall_s.append(float(call_wall))
        self.queue_wait_s.append(float(queue_wait))
        self.gpu_util_pct.append(float(gpu_util))

    # -- reductions ---------------------------------------------------------

    def n_calls(self) -> int:
        return len(self.batch_sizes)

    def total_requests(self) -> int:
        return int(sum(self.batch_sizes))

    def mean_batch(self) -> float:
        n = self.n_calls()
        return self.total_requests() / n if n else float("nan")

    def batch_histogram(self, bins: Optional[List[int]] = None) -> Dict[str, int]:
        """Bucket batch sizes into right-open ranges. Default bins bracket the
        regimes the X3 derivation predicts (starved < 128, partial, near-full)."""
        edges = bins or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        counts = {f"[{edges[i]},{edges[i + 1]})": 0 for i in range(len(edges) - 1)}
        counts[f">={edges[-1]}"] = 0
        for b in self.batch_sizes:
            placed = False
            for i in range(len(edges) - 1):
                if edges[i] <= b < edges[i + 1]:
                    counts[f"[{edges[i]},{edges[i + 1]})"] += 1
                    placed = True
                    break
            if not placed and b >= edges[-1]:
                counts[f">={edges[-1]}"] += 1
        return counts

    def summary(self) -> Dict[str, Any]:
        n = self.n_calls()
        inf_wall = float(sum(self.call_wall_s))
        wait_wall = float(sum(self.queue_wait_s))
        utils = [u for u in self.gpu_util_pct if u == u]  # drop NaN
        return {
            "n_calls": n,
            "total_requests": self.total_requests(),
            "mean_batch": round(self.mean_batch(), 2) if n else None,
            "inference_wall_s": round(inf_wall, 3),
            "mean_ms_per_call": round(inf_wall / n * 1000, 3) if n else None,
            "queue_wait_wall_s": round(wait_wall, 3),
            "gpu_util_pct_mean": round(sum(utils) / len(utils), 2) if utils else None,
            "gpu_util_pct_max": round(max(utils), 2) if utils else None,
            "batch_histogram": self.batch_histogram(),
        }

    def dump_json(self, path: str) -> None:
        payload = {
            "summary": self.summary(),
            "batch_sizes": self.batch_sizes,
            "call_wall_s": [round(x, 6) for x in self.call_wall_s],
            "queue_wait_s": [round(x, 6) for x in self.queue_wait_s],
            "gpu_util_pct": self.gpu_util_pct,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def print_summary(self) -> None:
        s = self.summary()
        print("[prtcfr-infer-instrument] evaluate() trace")
        for k in (
            "n_calls",
            "total_requests",
            "mean_batch",
            "inference_wall_s",
            "mean_ms_per_call",
            "queue_wait_wall_s",
            "gpu_util_pct_mean",
            "gpu_util_pct_max",
        ):
            print(f"  {k:22s}: {s[k]}")
        print("  batch_histogram:")
        for rng, c in s["batch_histogram"].items():
            if c:
                print(f"    {rng:14s}: {c}")


def _sample_gpu_util(gpu_index: int = 0) -> float:
    """Coarse GPU utilization percent via nvidia-smi. Returns NaN on any failure
    (no driver, parse error). Shells out; performs NO torch/CUDA call, so it is
    safe to leave importable on a CPU host (it simply returns NaN there)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode != 0:
            return float("nan")
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


@contextlib.contextmanager
def instrument_evaluate(
    gpu_util: bool = False,
    gpu_index: int = 0,
    gpu_sample_every: int = 1,
):
    """Temporarily wrap ``IncrementalSigmaManager.evaluate`` to fill and yield an
    ``EvaluateTrace``. Restores the original method on exit even on exception.

    ``gpu_util`` opt-in enables the ``nvidia-smi`` timeline; ``gpu_sample_every``
    throttles it (sample once per N calls) so the subprocess cost does not
    perturb the very wall it measures. The wrapper's own timing brackets only the
    inner call, so instrumentation overhead is excluded from ``call_wall_s`` and
    lands in the next call's ``queue_wait_s`` (a known, bounded skew).
    """
    from .prtcfr_worker import IncrementalSigmaManager

    rec = EvaluateTrace()
    original = IncrementalSigmaManager.evaluate
    state = {"last_end": None, "n": 0}

    def wrapper(self, queries, *a, **kw):
        now = time.perf_counter()
        wait = 0.0 if state["last_end"] is None else (now - state["last_end"])
        batch = len(queries)
        util = float("nan")
        if gpu_util and (state["n"] % max(1, gpu_sample_every) == 0):
            util = _sample_gpu_util(gpu_index)
        t0 = time.perf_counter()
        try:
            return original(self, queries, *a, **kw)
        finally:
            end = time.perf_counter()
            rec.add(batch, end - t0, wait, util)
            state["last_end"] = end
            state["n"] += 1

    IncrementalSigmaManager.evaluate = wrapper  # type: ignore[method-assign]
    try:
        yield rec
    finally:
        IncrementalSigmaManager.evaluate = original  # type: ignore[method-assign]
