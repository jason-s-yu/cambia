"""cfr/scripts/prtcfr_bench.py

X3 throughput bench harness (v0.4 Phase 2, S1W7): measures full-game PRT-CFR
generation and regret-fit wall-clock, per iteration, against the X3 gate
(phase2-throughput-pilot/contract.md AC1): <= 90 s/iter generation AND
<= 120 s/iter regret fit, at K=8192 games / m=4 rollouts, with a contention
annotation (GPU stats before + during, host load before + during) and a cost
profile breaking down where the time goes.

Design: this harness calls ``PRTCFRProductionTrainer.run_iteration(1)``
DIRECTLY -- the exact production gen/fit split (``gen_seconds``/``fit_seconds``
on the returned ``PRTCFRProductionTrainState``) -- so the measured numbers are
by construction the same quantities the trainer itself would log, not a
reimplementation that could drift from production behavior. Sub-phase
profiling (FFI vs inference vs reservoir IO vs fit forward/backward) is
layered on top via TEMPORARY method monkeypatches confined to this script
(applied in a ``contextlib.ExitStack`` and always restored in a ``finally``):
no production file (prtcfr_trainer.py / prtcfr_worker.py / prtcfr_net.py /
disk_reservoir.py) is ever edited on disk. This is strictly additive
instrumentation, per the sprint's "bench is additive" constraint.

Bucket attribution:
  gen phase   -- ffi (GameDriver method calls: apply/clone/tokens/
                 legal_actions/is_terminal/current_player/close on whichever
                 backend driver class is active), inference (the policy-query
                 seam of whichever gen path is active: with config.gen_batched
                 True -- the S1W15 default -- this is
                 IncrementalSigmaManager.evaluate, the SigmaBackend entry
                 _run_batched_scheduler calls once per tick with every live
                 game/rollout query that reached a decision that tick (a
                 single batched, carried-hidden GRU forward covering
                 potentially many queries); with --no-gen-batched this is the
                 original per-query NetProductionSigma.__call__ (one un-batched,
                 non-carried GRU forward per decision and per m*legal_actions
                 CRN rollout step -- see the trainer's own NetProductionSigma
                 docstring). Both are timed into the same "inference" bucket
                 so gen_seconds' cost breakdown is comparable across the two
                 paths even though the batched path's call COUNT reflects
                 scheduler ticks, not individual queries), reservoir_write
                 (_UnpaddingReservoir.add), other (derived remainder).
  fit phase   -- reservoir_sample (_MultiReservoirSampler.sample_batch, the
                 per-step IO), forward (PRTCFRNet.raw_advantages calls with
                 batch size > 1 -- this predicate is what separates the FIT's
                 batched forward from gen's single-item forwards, which also
                 route through raw_advantages via strategy_from_tokens but
                 are already counted under "inference"), backward_opt_misc
                 (derived remainder: loss compute + backward() + grad clip +
                 optimizer/scheduler step + host<->device transfer).

Scoping: the V_phi critic fit and the stability-controller manifest write run
inside ``run_iteration`` but AFTER ``fit_seconds`` stops (see
prtcfr_trainer.py's own timer placement) -- they are not part of the X3 gate,
so this harness disables them by default (``--critic-enabled`` to re-enable)
to keep bench wall-clock focused on exactly the gated quantities.

Usage (smoke, CPU, python backend, tiny cell):
  cd cfr && python scripts/prtcfr_bench.py --k-games 4 --m-rollouts 1 \\
      --batch-size 16 --train-steps 5 --backend python --device cpu \\
      --max-trajectory-steps 40 --no-wait-for-clean-host

Usage (a real measurement cell, Go backend, GPU):
  cd cfr && python scripts/prtcfr_bench.py --k-games 64 --m-rollouts 4 \\
      --backend go --device cuda --train-steps 200 --batch-size 2048 \\
      --out runs/x3-bench/cell_k64.json

Usage (the actual X3 gate cell -- WARNING: can run for a long time given the
production sampler's un-batched per-decision inference; see the sprint
append-log and the verdict doc this harness's output feeds):
  cd cfr && python scripts/prtcfr_bench.py --out runs/x3-bench/x3_cell.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PRTCFRConfig, load_config  # noqa: E402
from src.cfr.prtcfr_worker import (  # noqa: E402
    GoEngineGameDriver,
    IncrementalSigmaManager,
    PRODUCTION_SEQ_CAP,
    PythonEngineGameDriver,
    new_production_driver,
)
from src.cfr.prtcfr_trainer import (  # noqa: E402
    NetProductionSigma,
    PRTCFRProductionTrainer,
    _fit_from_scratch,
    _MultiReservoirSampler,
    _peak_lr_for_iter,
    _UnpaddingReservoir,
)
from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402

_DEFAULT_BASE_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "prtcfr_production.yaml",
)


# ---------------------------------------------------------------------------
# Contention snapshot (GPU + host load) -- contract.md AC1 contention-window
# rule: GPU stats before AND during each timed run, host load annotated.
# ---------------------------------------------------------------------------


#: Fallback nvidia-smi locations for hosts where it is not on PATH. WSL2 ships
#: the driver shim at /usr/lib/wsl/lib/nvidia-smi but does not always add that
#: dir to PATH, so the plain "nvidia-smi" lookup fails and the GPU annotation
#: came back empty (observed on this WSL host during the P5 cell). Try PATH
#: first, then the known WSL shim path.
_NVIDIA_SMI_CANDIDATES = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi")


def _resolve_nvidia_smi() -> Optional[str]:
    """Return the first usable nvidia-smi binary: PATH lookup first, then the
    WSL shim path. None if neither resolves."""
    import shutil

    for cand in _NVIDIA_SMI_CANDIDATES:
        resolved = shutil.which(cand)
        if resolved:
            return resolved
        if os.path.isabs(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _nvidia_smi_snapshot() -> Dict[str, Any]:
    binary = _resolve_nvidia_smi()
    if binary is None:
        return {"error": "nvidia-smi not found on PATH or at /usr/lib/wsl/lib"}
    try:
        out = subprocess.run(
            [
                binary,
                "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        first = out.splitlines()[0]
        util, used, free, total = (int(x.strip()) for x in first.split(","))
        return {
            "util_pct": util,
            "mem_used_mib": used,
            "mem_free_mib": free,
            "mem_total_mib": total,
        }
    except Exception as e:  # pragma: no cover - environment dependent
        return {"error": str(e)}


def contention_snapshot() -> Dict[str, Any]:
    """One point-in-time reading of GPU + host contention state."""
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:  # pragma: no cover - non-POSIX
        load1 = load5 = load15 = None
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": _nvidia_smi_snapshot(),
        "load1": load1,
        "load5": load5,
        "load15": load15,
    }


class ContentionMonitor:
    """Background sampler of ``contention_snapshot()`` for the duration of a
    timed run (the "during" half of the contention-window rule); the caller
    supplies the "before"/"after" snapshots itself."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(contention_snapshot())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ContentionMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5.0)
        return False

    def summary(self) -> Dict[str, Any]:
        if not self._samples:
            return {"samples": 0}
        utils = [
            s["gpu"]["util_pct"]
            for s in self._samples
            if isinstance(s.get("gpu"), dict) and "util_pct" in s["gpu"]
        ]
        frees = [
            s["gpu"]["mem_free_mib"]
            for s in self._samples
            if isinstance(s.get("gpu"), dict) and "mem_free_mib" in s["gpu"]
        ]
        loads = [s["load1"] for s in self._samples if s.get("load1") is not None]
        out: Dict[str, Any] = {"samples": len(self._samples)}
        if utils:
            out["gpu_util_pct"] = {
                "min": min(utils),
                "mean": round(statistics.mean(utils), 1),
                "max": max(utils),
            }
        if frees:
            out["gpu_mem_free_mib"] = {
                "min": min(frees),
                "mean": round(statistics.mean(frees), 1),
                "max": max(frees),
            }
        if loads:
            out["load1"] = {
                "min": min(loads),
                "mean": round(statistics.mean(loads), 2),
                "max": max(loads),
            }
        return out


class Heartbeat:
    """Throttled progress log so a future hang is localizable from the bench
    log (which phase/game it stalled in) instead of a silent `timeout` reap.
    ``tick`` is rate-limited to ``min_interval_s``; ``force`` always prints."""

    def __init__(self, min_interval_s: float = 2.0):
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._t0 = time.perf_counter()

    def force(self, label: str) -> None:
        now = time.perf_counter()
        print(f"[prtcfr-bench] heartbeat t+{now - self._t0:6.1f}s: {label}", flush=True)
        self._last = now

    def tick(self, label: str) -> None:
        now = time.perf_counter()
        if now - self._last >= self.min_interval_s:
            self.force(label)

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


def _cuda_warm_or_timeout(device: str, timeout_s: float = 20.0) -> None:
    """Fail fast if CUDA init / first allocation hangs, rather than let an
    outer `timeout` reap the process with no diagnostic (observed: a
    co-tenant spiking SM util to ~100% around context-creation time can stall
    it indefinitely). Runs a trivial alloc+sync on a background thread with a
    join timeout; the thread cannot be force-killed if CUDA itself is wedged,
    but the caller gets a clear, immediate, actionable error instead of
    silence. No-op for non-CUDA devices."""
    if not device.startswith("cuda"):
        return
    import torch

    result: Dict[str, Any] = {}

    def _warm() -> None:
        try:
            t0 = time.perf_counter()
            x = torch.zeros(1, device=device)
            torch.cuda.synchronize()
            _ = x.cpu()
            result["ok"] = True
            result["seconds"] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)

    th = threading.Thread(target=_warm, daemon=True)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        raise TimeoutError(
            f"CUDA init/first-alloc did not return within {timeout_s:.0f}s "
            f"(device={device}); a co-tenant likely holds the SM/allocator "
            "(check nvidia-smi util/mem). Failing fast rather than hanging "
            "under an outer timeout with no diagnostic."
        )
    if "error" in result:
        raise RuntimeError(f"CUDA init/first-alloc failed: {result['error']}")


def _wait_for_clean_host(max_load: float, poll_s: float, timeout_s: float) -> float:
    """Block until 1-min load avg <= ``max_load`` or ``timeout_s`` elapses
    (host protocol: a full-suite batch gate may run concurrently early in a
    lane; poll rather than pollute the measurement window)."""
    t0 = time.time()
    while True:
        load1 = os.getloadavg()[0]
        if load1 <= max_load:
            return load1
        if time.time() - t0 > timeout_s:
            print(
                f"[prtcfr-bench] WARNING: host load1={load1:.2f} still above "
                f"{max_load} after {timeout_s:.0f}s wait; proceeding anyway "
                "(annotated in results)."
            )
            return load1
        print(
            f"[prtcfr-bench] host load1={load1:.2f} > {max_load}; waiting "
            f"{poll_s:.0f}s for it to clear..."
        )
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Timers + additive instrumentation (temporary monkeypatches, always restored)
# ---------------------------------------------------------------------------


class BenchTimers:
    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    def add(self, bucket: str, dt: float) -> None:
        self.seconds[bucket] = self.seconds.get(bucket, 0.0) + dt
        self.calls[bucket] = self.calls.get(bucket, 0) + 1


@contextlib.contextmanager
def _patched(
    cls: type,
    method_name: str,
    timers: BenchTimers,
    bucket: str,
    predicate: Optional[Callable[[tuple, dict], bool]] = None,
):
    """Temporarily wrap ``cls.method_name`` with a wall-clock timer into
    ``timers[bucket]``. Restores the original method on exit unconditionally.
    Never touches the source file; the wrapped attribute lives only on the
    live class object for the ``with``/``ExitStack`` scope."""
    if not hasattr(cls, method_name):
        yield
        return
    orig = getattr(cls, method_name)

    def wrapper(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            dt = time.perf_counter() - t0
            if predicate is None or predicate(a, kw):
                timers.add(bucket, dt)

    setattr(cls, method_name, wrapper)
    try:
        yield
    finally:
        setattr(cls, method_name, orig)


@contextlib.contextmanager
def _heartbeat_patched(cls: type, method_name: str, hb: "Heartbeat", label: str):
    """Temporarily wrap ``cls.method_name`` with a throttled heartbeat tick
    (composes with ``_patched`` on the same method via ExitStack nesting: each
    context captures whatever is currently installed as its own "orig", so
    stacking a timing patch and a heartbeat patch on the same method chains
    both effects). Restores the original on exit."""
    if not hasattr(cls, method_name):
        yield
        return
    orig = getattr(cls, method_name)
    counter = {"n": 0}

    def wrapper(self, *a, **kw):
        counter["n"] += 1
        hb.tick(f"{label} #{counter['n']}")
        return orig(self, *a, **kw)

    setattr(cls, method_name, wrapper)
    try:
        yield
    finally:
        setattr(cls, method_name, orig)


def _batch_gt1_predicate(a: tuple, kw: dict) -> bool:
    """True iff the first positional arg is a tensor-like with batch dim > 1
    (separates fit's batched raw_advantages calls from gen's single-item
    strategy_from_tokens-routed calls, which are already timed under
    "inference")."""
    if not a:
        return False
    first = a[0]
    shape = getattr(first, "shape", None)
    return bool(shape) and shape[0] > 1


_FFI_METHODS = (
    "apply",
    "clone",
    "tokens",
    "legal_actions",
    "is_terminal",
    "current_player",
    "utility",
    "close",
)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_bench_config(args: argparse.Namespace) -> PRTCFRConfig:
    """Load the real production config (rule profile, net dims, LR schedule,
    etc.) and override only the X3-relevant knobs the harness exposes. Critic
    and stability are disabled by default (see module docstring: both run
    outside the gated gen/fit window; leaving them on only spends bench wall
    clock on non-gated quantities)."""
    base_cfg = load_config(args.base_config)
    prt_cfg = getattr(base_cfg, "prt_cfr", None)
    if prt_cfg is None:
        prt_cfg = PRTCFRConfig()
    device = _resolve_device(args.device)
    overrides = dict(
        k_games_per_iter=args.k_games,
        m_rollouts=args.m_rollouts,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        seq_cap=args.seq_cap,
        backend=args.backend,
        device=device,
        num_players=args.num_players,
        max_trajectory_steps=args.max_trajectory_steps,
        seed=args.seed,
        critic_enabled=args.critic_enabled,
        stability_enabled=False,
    )
    # S1W15 additive: select the generation path / batching knobs when the flag
    # is given; otherwise inherit the config default (batched, chunk 64, bf16).
    if getattr(args, "gen_batched", None) is not None:
        overrides["gen_batched"] = bool(args.gen_batched)
    if getattr(args, "gen_chunk_games", None) is not None:
        overrides["gen_chunk_games"] = int(args.gen_chunk_games)
    if getattr(args, "infer_dtype", None) is not None:
        overrides["infer_dtype"] = str(args.infer_dtype)
    merged = prt_cfg.model_dump()
    merged.update(overrides)
    return PRTCFRConfig(**merged)


# ---------------------------------------------------------------------------
# Atomic output (L4, cambia-248 quality review): both observed X3 cells
# died mid-run (OOM / external kill during a multi-hour generation) with NO
# JSON at all, discarding the config/contention/timing data already
# collected. ``_atomic_write_json`` never leaves a torn file at the target
# path; ``_write_partial_cell_result`` is the try/finally dump a cell writes
# on the way out when it fails, so a killed run still leaves a diagnosable
# artifact instead of nothing.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write ``obj`` as JSON to ``path`` via a same-directory temp file plus
    ``os.replace``, so a reader (or a process killed mid-write) never
    observes a partially-written file at ``path`` -- either the previous
    complete file is still there, or the new one is."""
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _write_partial_cell_result(
    out_path: str,
    mode: str,
    cfg: PRTCFRConfig,
    device: str,
    hb: Heartbeat,
    host_load_before: float,
    gpu_pre: Optional[Dict[str, Any]],
    contention_during: Optional[Dict[str, Any]],
    exc: BaseException,
) -> None:
    """Best-effort dump of whatever a bench cell had measured before it died
    (config, before-snapshot, contention samples collected so far, elapsed
    wall time, the exception) -- written atomically to ``out_path`` so a
    killed cell still leaves diagnosable output instead of nothing."""
    partial = {
        "mode": mode,
        "status": "failed",
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "elapsed_seconds": round(hb.elapsed(), 3),
        "config": {
            "k_games": cfg.k_games_per_iter,
            "m_rollouts": cfg.m_rollouts,
            "batch_size": cfg.batch_size,
            "train_steps": cfg.train_steps,
            "seq_cap": cfg.seq_cap,
            "backend": cfg.backend,
            "device": device,
            "num_players": cfg.num_players,
            "critic_enabled": cfg.critic_enabled,
        },
        "contention": {
            "host_load1_before": host_load_before,
            "gpu_before": gpu_pre,
            "during_summary": contention_during,
        },
    }
    _atomic_write_json(out_path, partial)
    print(
        f"[prtcfr-bench] cell FAILED after {hb.elapsed():.1f}s "
        f"({type(exc).__name__}: {exc}); wrote partial result to {out_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# One measured cell
# ---------------------------------------------------------------------------


def run_cell(
    args: argparse.Namespace,
    batch_size_override: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run exactly one PRT-CFR production iteration (gen + fit), instrumented.

    Returns a JSON-able dict: gen_seconds/fit_seconds (straight from the
    trainer's own PRTCFRProductionTrainState), the profile breakdown, and the
    contention annotation (GPU + host load before/during/after).
    """
    cfg = build_bench_config(args)
    if batch_size_override is not None:
        cfg = PRTCFRConfig(**{**cfg.model_dump(), "batch_size": batch_size_override})

    device = str(cfg.device)
    hb = Heartbeat(min_interval_s=args.heartbeat_interval_s)
    hb.force(f"start (device={device} backend={cfg.backend} k={cfg.k_games_per_iter})")

    pre = contention_snapshot()
    gpu_pre = pre.get("gpu") if device.startswith("cuda") else None
    if (
        device.startswith("cuda")
        and isinstance(gpu_pre, dict)
        and "mem_free_mib" in gpu_pre
    ):
        print(
            f"[prtcfr-bench] pre-launch GPU headroom: free={gpu_pre['mem_free_mib']}MiB "
            f"util={gpu_pre['util_pct']}% (of {gpu_pre['mem_total_mib']}MiB total)"
        )
        if gpu_pre["mem_free_mib"] < args.min_free_vram_mib:
            raise RuntimeError(
                f"insufficient free VRAM before launch: free={gpu_pre['mem_free_mib']}MiB "
                f"< required {args.min_free_vram_mib}MiB (co-tenant likely holding the "
                "card); re-check nvidia-smi and retry, or lower --min-free-vram-mib"
            )

    hb.force("cuda-init:start")
    _cuda_warm_or_timeout(device, timeout_s=args.cuda_init_timeout_s)
    hb.force("cuda-init:done")

    run_dir = args.run_dir or tempfile.mkdtemp(prefix="prtcfr_bench_")
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "cambia_runs_bench.db")

    def driver_factory(seed: int):
        hb.tick(f"driver-init game_seed={seed}")
        return new_production_driver(
            seed, num_players=cfg.num_players, backend=cfg.backend
        )

    hb.force("trainer-init:start")
    trainer = PRTCFRProductionTrainer(
        cfg,
        run_dir,
        driver_factory=driver_factory,
        db_path=db_path,
        run_name=f"x3-bench-{int(time.time())}",
    )
    hb.force("trainer-init:done")

    timers = BenchTimers()
    host_load_before = os.getloadavg()[0]
    print(f"[prtcfr-bench] host load1 before run: {host_load_before:.2f}")

    mon: Optional[ContentionMonitor] = None
    try:
        with contextlib.ExitStack() as stack:
            for cls in (GoEngineGameDriver, PythonEngineGameDriver):
                for meth in _FFI_METHODS:
                    stack.enter_context(_patched(cls, meth, timers, "ffi"))
                stack.enter_context(_heartbeat_patched(cls, "close", hb, "game-done"))
            # Sequential path (--no-gen-batched): per-query un-batched forward.
            stack.enter_context(
                _patched(NetProductionSigma, "__call__", timers, "inference")
            )
            # Batched path (config.gen_batched default True, S1W15): the
            # SigmaBackend.evaluate seam the scheduler calls once per tick
            # with the whole batch of live queries; only one of these two
            # patches ever actually fires for a given run, depending on which
            # generation path run_iteration takes.
            stack.enter_context(
                _patched(IncrementalSigmaManager, "evaluate", timers, "inference")
            )
            stack.enter_context(
                _patched(_UnpaddingReservoir, "add", timers, "reservoir_write")
            )
            stack.enter_context(
                _patched(
                    _MultiReservoirSampler, "sample_batch", timers, "reservoir_sample"
                )
            )
            stack.enter_context(
                _patched(
                    PRTCFRNet,
                    "raw_advantages",
                    timers,
                    "fit_forward",
                    predicate=_batch_gt1_predicate,
                )
            )
            with ContentionMonitor(interval_s=args.contention_interval_s) as mon:
                hb.force("gen+fit:start")
                state = trainer.run_iteration(1)
                hb.force("gen+fit:done")
    except BaseException as exc:
        # L4 (cambia-248): a mid-run death (OOM, external kill) previously
        # lost every timing/contention sample this cell had already collected
        # -- dump whatever is available now, atomically, before propagating.
        if out_path:
            _write_partial_cell_result(
                out_path,
                mode="gen_fit",
                cfg=cfg,
                device=device,
                hb=hb,
                host_load_before=host_load_before,
                gpu_pre=gpu_pre,
                contention_during=mon.summary() if mon is not None else None,
                exc=exc,
            )
        raise
    finally:
        trainer.close()

    host_load_after = os.getloadavg()[0]
    post = contention_snapshot()
    gpu_post = post.get("gpu") if device.startswith("cuda") else None

    gen_seconds = float(state.gen_seconds)
    fit_seconds = float(state.fit_seconds)

    gen_measured = (
        timers.seconds.get("ffi", 0.0)
        + timers.seconds.get("inference", 0.0)
        + timers.seconds.get("reservoir_write", 0.0)
    )
    gen_other = max(0.0, gen_seconds - gen_measured)

    fit_measured = timers.seconds.get("reservoir_sample", 0.0) + timers.seconds.get(
        "fit_forward", 0.0
    )
    fit_other = max(0.0, fit_seconds - fit_measured)

    profile = {
        "gen": {
            "ffi_s": round(timers.seconds.get("ffi", 0.0), 3),
            "inference_s": round(timers.seconds.get("inference", 0.0), 3),
            "reservoir_write_s": round(timers.seconds.get("reservoir_write", 0.0), 3),
            "other_s": round(gen_other, 3),
            "total_s": round(gen_seconds, 3),
        },
        "fit": {
            "reservoir_sample_s": round(timers.seconds.get("reservoir_sample", 0.0), 3),
            "forward_s": round(timers.seconds.get("fit_forward", 0.0), 3),
            "backward_opt_misc_s": round(fit_other, 3),
            "total_s": round(fit_seconds, 3),
        },
        "call_counts": dict(timers.calls),
    }

    return {
        "config": {
            "k_games": cfg.k_games_per_iter,
            "m_rollouts": cfg.m_rollouts,
            "batch_size": cfg.batch_size,
            "train_steps": cfg.train_steps,
            "seq_cap": cfg.seq_cap,
            "backend": cfg.backend,
            "device": device,
            "num_players": cfg.num_players,
            "critic_enabled": cfg.critic_enabled,
        },
        "gen_seconds": gen_seconds,
        "fit_seconds": fit_seconds,
        "samples_added": state.samples_added,
        "buffer_sizes": state.buffer_sizes,
        "profile": profile,
        "contention": {
            "host_load1_before": host_load_before,
            "host_load1_after": host_load_after,
            "gpu_before": gpu_pre,
            "gpu_after": gpu_post,
            "during_summary": mon.summary(),
        },
        "run_dir": run_dir,
    }


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return True
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


def run_cell_with_oom_backoff(
    args: argparse.Namespace, out_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run ``run_cell``, halving batch_size on CUDA OOM down to
    ``--min-batch-size`` rather than crashing the bench (GPU protocol).
    ``out_path``, if given, is forwarded to every attempt so a mid-run death
    (OOM retry exhaustion or any other exception) still leaves a partial
    result on disk (see ``_write_partial_cell_result``)."""
    batch_size = args.batch_size
    attempt = 0
    while True:
        try:
            return run_cell(args, batch_size_override=batch_size, out_path=out_path)
        except Exception as e:  # noqa: BLE001 - re-raised below if not OOM
            if not _is_oom(e):
                raise
            attempt += 1
            if attempt > args.max_oom_retries or batch_size <= args.min_batch_size:
                raise RuntimeError(
                    f"OOM persists after {attempt - 1} batch-size backoffs "
                    f"(final batch_size={batch_size}): {e}"
                ) from e
            new_batch = max(args.min_batch_size, batch_size // 2)
            print(
                f"[prtcfr-bench] OOM at batch_size={batch_size}; backing off to "
                f"{new_batch} and retrying"
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            batch_size = new_batch


# ---------------------------------------------------------------------------
# Fit-scale probe: measure fit_seconds at the REAL production batch_size /
# train_steps without waiting on a literal K=8192 generation (infeasible at
# the observed serial per-game gen cost -- see the verdict this harness's
# output feeds). Harvests a small, feasible batch of REAL generated samples,
# then replicates that real (not synthetic) content up to the requested batch
# scale via the reservoir's own public add_batch API, and times the
# UNMODIFIED production `_fit_from_scratch` against it.
# ---------------------------------------------------------------------------


def _replicate_reservoir_to_size(disk_reservoir: Any, target_size: int) -> None:
    """Grow ``disk_reservoir`` to >= ``target_size`` rows by cycling its
    current real content back through ``add_batch`` (natural-length rows,
    real token content -- not synthetic filler). No-op if already empty or
    already at/above ``target_size``."""
    current = len(disk_reservoir)
    if current == 0 or current >= target_size:
        return
    batch = disk_reservoir.sample_batch(current)  # every real row, natural lengths
    feats = [batch.features[i, : int(batch.lengths[i])] for i in range(current)]
    targets = batch.targets
    masks = batch.masks
    iterations = batch.iterations
    needed = target_size - current
    while needed > 0:
        take = min(needed, current)
        idx = list(range(take))
        disk_reservoir.add_batch(
            features=[feats[i] for i in idx],
            targets=targets[idx],
            masks=masks[idx] if masks is not None else None,
            iterations=iterations[idx],
        )
        needed -= take


def run_fit_scale_probe(
    args: argparse.Namespace,
    seed_k_games: int = 16,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure fit_seconds at ``args.batch_size`` / ``args.train_steps`` (the
    real production values by default) against a reservoir seeded from
    ``seed_k_games`` real generated games and replicated up to batch scale.

    The seed generation's own fit is a throwaway 1-step pass (irrelevant;
    reservoirs are populated during generation, before any fit runs); the
    REAL measurement is the second, explicit ``_fit_from_scratch`` call below,
    run at the caller's requested ``batch_size``/``train_steps``.
    """
    seed_args = argparse.Namespace(**vars(args))
    seed_args.k_games = seed_k_games
    seed_args.train_steps = 1
    cfg = build_bench_config(seed_args)
    device = str(cfg.device)

    hb = Heartbeat(min_interval_s=args.heartbeat_interval_s)
    hb.force(f"fit-scale-probe start (device={device} backend={cfg.backend})")
    hb.force("cuda-init:start")
    _cuda_warm_or_timeout(device, timeout_s=args.cuda_init_timeout_s)
    hb.force("cuda-init:done")

    pre_host_load = os.getloadavg()[0]
    pre_gpu = contention_snapshot().get("gpu") if device.startswith("cuda") else None

    run_dir = args.run_dir or tempfile.mkdtemp(prefix="prtcfr_fitprobe_")
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "cambia_runs_bench.db")

    def driver_factory(seed: int):
        hb.tick(f"seed-gen driver-init game_seed={seed}")
        return new_production_driver(
            seed, num_players=cfg.num_players, backend=cfg.backend
        )

    hb.force("trainer-init:start")
    trainer = PRTCFRProductionTrainer(
        cfg,
        run_dir,
        driver_factory=driver_factory,
        db_path=db_path,
        run_name=f"x3-fitprobe-{int(time.time())}",
    )
    hb.force("trainer-init:done")
    mon: Optional[ContentionMonitor] = None
    try:
        print(f"[prtcfr-bench] fit-scale probe: seeding {seed_k_games} real games...")
        hb.force("seed-gen:start")
        seed_gen_start = time.perf_counter()
        trainer.run_iteration(1)  # populates trainer.reservoirs; its own fit is throwaway
        seed_gen_seconds = time.perf_counter() - seed_gen_start
        hb.force("seed-gen:done")

        target_per_player = max(args.batch_size // cfg.num_players + 128, 256)
        for p in range(cfg.num_players):
            _replicate_reservoir_to_size(trainer.reservoirs[p].raw, target_per_player)
        reservoir_sizes = {p: len(trainer.reservoirs[p]) for p in range(cfg.num_players)}
        print(
            f"[prtcfr-bench] fit-scale probe: reservoir sizes after replication: {reservoir_sizes}"
        )

        sampler = _MultiReservoirSampler(
            [trainer.reservoirs[p] for p in range(cfg.num_players)]
        )
        peak_lr = _peak_lr_for_iter(
            cfg.lr, cfg.lr_min, 1, cfg.iterations, cfg.lr_schedule
        )

        timers = BenchTimers()
        host_before = os.getloadavg()[0]
        gpu_before = contention_snapshot().get("gpu")
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                _patched(
                    _MultiReservoirSampler, "sample_batch", timers, "reservoir_sample"
                )
            )
            stack.enter_context(
                _heartbeat_patched(PRTCFRNet, "raw_advantages", hb, "fit-step")
            )
            stack.enter_context(
                _patched(
                    PRTCFRNet,
                    "raw_advantages",
                    timers,
                    "fit_forward",
                    predicate=_batch_gt1_predicate,
                )
            )
            with ContentionMonitor(interval_s=args.contention_interval_s) as mon:
                hb.force(
                    f"fit:start batch_size={args.batch_size} train_steps={args.train_steps}"
                )
                t0 = time.perf_counter()
                loss = _fit_from_scratch(
                    trainer.net,
                    sampler,
                    lr=peak_lr,
                    batch_size=args.batch_size,
                    num_steps=args.train_steps,
                    weight_decay=cfg.weight_decay,
                    grad_clip=cfg.grad_clip,
                    lr_min=cfg.lr_min,
                )
                fit_seconds = time.perf_counter() - t0
                hb.force("fit:done")
        host_after = os.getloadavg()[0]
        gpu_after = contention_snapshot().get("gpu")
    except BaseException as exc:
        # L4 (cambia-248): mirror run_cell's partial dump for the fit-scale
        # probe path (its own seed-gen phase runs a full run_iteration too).
        if out_path:
            _write_partial_cell_result(
                out_path,
                mode="fit_scale_probe",
                cfg=cfg,
                device=device,
                hb=hb,
                host_load_before=pre_host_load,
                gpu_pre=pre_gpu,
                contention_during=mon.summary() if mon is not None else None,
                exc=exc,
            )
        raise
    finally:
        trainer.close()

    fit_measured = timers.seconds.get("reservoir_sample", 0.0) + timers.seconds.get(
        "fit_forward", 0.0
    )
    fit_other = max(0.0, fit_seconds - fit_measured)

    return {
        "mode": "fit_scale_probe",
        "config": {
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "seq_cap": cfg.seq_cap,
            "backend": cfg.backend,
            "device": str(cfg.device),
            "seed_k_games": seed_k_games,
            "seed_gen_seconds": round(seed_gen_seconds, 3),
            "reservoir_sizes_after_replication": reservoir_sizes,
        },
        "fit_loss": float(loss),
        "fit_seconds": fit_seconds,
        "profile": {
            "reservoir_sample_s": round(timers.seconds.get("reservoir_sample", 0.0), 3),
            "forward_s": round(timers.seconds.get("fit_forward", 0.0), 3),
            "backward_opt_misc_s": round(fit_other, 3),
            "total_s": round(fit_seconds, 3),
            "call_counts": dict(timers.calls),
        },
        "contention": {
            "host_load1_before": host_before,
            "host_load1_after": host_after,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "during_summary": mon.summary(),
        },
        "run_dir": run_dir,
        "caveat": (
            "Reservoir content is REAL token samples harvested from "
            f"{seed_k_games} real generated games, REPLICATED (not synthetic "
            "filler) up to the requested batch_size via the reservoir's own "
            "public add_batch API -- literally generating K=8192 games to "
            "reach this batch size at the observed per-game gen cost is "
            "infeasible in this bench (~9s/game serial => hours). This is an "
            "iteration-1 (near-uniform policy, natural-cohort, short-game) "
            "length distribution; later-iteration reservoirs may carry a "
            "longer tail (games running closer to the 300-turn cap), which "
            "would only INCREASE per-step packed-GRU compute -- so this "
            "measurement is a lower bound on steady-state production fit "
            "cost, not an upper bound."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def gate_verdict(
    result: Dict[str, Any], gate_gen_s: float, gate_fit_s: float
) -> Dict[str, str]:
    gen = result["gen_seconds"]
    fit = result["fit_seconds"]
    return {
        "gen": "PASS" if gen <= gate_gen_s else "FAIL",
        "fit": "PASS" if fit <= gate_fit_s else "FAIL",
    }


def human_summary(
    result: Dict[str, Any], gate_gen_s: float = 90.0, gate_fit_s: float = 120.0
) -> str:
    gen = result["gen_seconds"]
    fit = result["fit_seconds"]
    verdict = gate_verdict(result, gate_gen_s, gate_fit_s)
    p = result["profile"]
    c = result["config"]
    contention = result["contention"]
    lines = [
        f"K={c['k_games']} m={c['m_rollouts']} batch={c['batch_size']} "
        f"train_steps={c['train_steps']} backend={c['backend']} device={c['device']}",
        f"gen:  {gen:.2f}s  (gate <= {gate_gen_s:.0f}s)  [{verdict['gen']}]",
        f"  ffi={p['gen']['ffi_s']:.2f}s inference={p['gen']['inference_s']:.2f}s "
        f"reservoir_write={p['gen']['reservoir_write_s']:.2f}s other={p['gen']['other_s']:.2f}s",
        f"fit:  {fit:.2f}s  (gate <= {gate_fit_s:.0f}s)  [{verdict['fit']}]",
        f"  reservoir_sample={p['fit']['reservoir_sample_s']:.2f}s "
        f"forward={p['fit']['forward_s']:.2f}s "
        f"backward_opt_misc={p['fit']['backward_opt_misc_s']:.2f}s",
        f"contention: host_load1 before={contention['host_load1_before']:.2f} "
        f"after={contention['host_load1_after']:.2f} during={contention['during_summary']}",
    ]
    if contention.get("gpu_before"):
        lines.append(
            f"gpu: before={contention['gpu_before']} after={contention['gpu_after']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="PRT-CFR X3 throughput bench: gen/fit split, profiled, gate verdict."
    )
    ap.add_argument("--base-config", default=_DEFAULT_BASE_CONFIG)
    ap.add_argument("--k-games", type=int, default=8192)
    ap.add_argument("--m-rollouts", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--seq-cap", type=int, default=PRODUCTION_SEQ_CAP)
    ap.add_argument("--backend", choices=["go", "python"], default="go")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--max-trajectory-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    # S1W15 additive knobs (do not change any measurement semantics; they only
    # select which generation path run_iteration takes so the X3 gen cell can be
    # measured before/after the batched-incremental wiring).
    ap.add_argument(
        "--gen-batched",
        dest="gen_batched",
        action="store_true",
        default=None,
        help="force the batched incremental generation path (S1W15). Default: "
        "inherit the config (production default True).",
    )
    ap.add_argument(
        "--no-gen-batched",
        dest="gen_batched",
        action="store_false",
        help="force the original per-decision full-prefix generation path (the "
        "S1W7 X3 'before' baseline).",
    )
    ap.add_argument("--gen-chunk-games", type=int, default=None)
    ap.add_argument("--infer-dtype", default=None, choices=["bf16", "fp32"])
    ap.add_argument("--critic-enabled", action="store_true", default=False)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None, help="JSON results output path")
    ap.add_argument("--min-batch-size", type=int, default=64)
    ap.add_argument("--max-oom-retries", type=int, default=4)
    ap.add_argument("--min-free-vram-mib", type=int, default=800)
    ap.add_argument("--contention-interval-s", type=float, default=5.0)
    ap.add_argument(
        "--heartbeat-interval-s",
        type=float,
        default=2.0,
        help="Minimum seconds between throttled progress heartbeat log lines.",
    )
    ap.add_argument(
        "--cuda-init-timeout-s",
        type=float,
        default=20.0,
        help=(
            "Fail fast (raise) if CUDA init / first alloc does not return "
            "within this many seconds (a co-tenant at ~100%% SM util can "
            "stall context creation indefinitely)."
        ),
    )
    ap.add_argument("--max-host-load", type=float, default=8.0)
    ap.add_argument("--host-wait-timeout-s", type=float, default=900.0)
    ap.add_argument("--host-wait-poll-s", type=float, default=15.0)
    ap.add_argument("--no-wait-for-clean-host", action="store_true")
    ap.add_argument("--gate-gen-s", type=float, default=90.0)
    ap.add_argument("--gate-fit-s", type=float, default=120.0)
    ap.add_argument(
        "--fit-scale-probe",
        action="store_true",
        help=(
            "Measure fit_seconds at --batch-size/--train-steps against a "
            "reservoir seeded from --seed-k-games real games and replicated "
            "up to batch scale, instead of running a combined gen+fit cell "
            "(use when K games at the requested scale is not wall-clock "
            "feasible; see run_fit_scale_probe's docstring)."
        ),
    )
    ap.add_argument("--seed-k-games", type=int, default=16)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.no_wait_for_clean_host:
        _wait_for_clean_host(
            args.max_host_load, args.host_wait_poll_s, args.host_wait_timeout_s
        )
    if args.fit_scale_probe:
        result = run_fit_scale_probe(
            args, seed_k_games=args.seed_k_games, out_path=args.out
        )
        verdict = "PASS" if result["fit_seconds"] <= args.gate_fit_s else "FAIL"
        print(
            f"[fit-scale-probe] batch={result['config']['batch_size']} "
            f"train_steps={result['config']['train_steps']} "
            f"fit_seconds={result['fit_seconds']:.2f}s (gate <= {args.gate_fit_s:.0f}s) "
            f"[{verdict}]"
        )
        print(
            f"  reservoir_sample={result['profile']['reservoir_sample_s']:.2f}s "
            f"forward={result['profile']['forward_s']:.2f}s "
            f"backward_opt_misc={result['profile']['backward_opt_misc_s']:.2f}s"
        )
    else:
        result = run_cell_with_oom_backoff(args, out_path=args.out)
        print(human_summary(result, args.gate_gen_s, args.gate_fit_s))
    if args.out:
        # Success path: same atomic writer the partial-dump path uses (L4,
        # cambia-248), so the final JSON is never left torn either.
        _atomic_write_json(args.out, result)
        print(f"\n[prtcfr-bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
