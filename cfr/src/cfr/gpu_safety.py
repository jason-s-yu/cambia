"""src/cfr/gpu_safety.py

Failsafes for co-resident GPU jobs. Other processes may spawn and grab VRAM in
real time, so a long-running job should check before it allocates, bound its own
footprint, survive a transient mid-run OOM, and refuse to clobber a peer run of
itself.

Design constraint: do NOT fight PyTorch's caching allocator. The caching pool is
what makes repeated allocations fast; these helpers add only one-time startup
checks and rare-path (OOM) recovery, so steady-state throughput is unchanged.
`cap_process_vram` sets a ceiling, not a reservation: the allocator still grows
lazily and keeps its fast freed-block cache. Every function no-ops gracefully
when the target accelerator is unavailable, so CPU-only runs and tests import
and call them freely.

cuda and xpu (Intel Arc) expose near-identical memory/allocator APIs
(mem_get_info, set_per_process_memory_fraction, empty_cache), so every helper
below takes a ``device_type`` ("cuda" by default, for backward compat) and
dispatches to the matching ``torch.cuda``/``torch.xpu`` submodule.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_GB = 1024 ** 3


def _accel_module(device_type: str) -> Optional[Any]:
    """Return the ``torch.cuda``/``torch.xpu`` submodule for ``device_type``.

    None when ``device_type`` isn't an accelerator, the submodule doesn't
    exist on this torch build, or the accelerator isn't available.
    """
    try:
        import torch
    except Exception:
        return None
    if device_type == "cuda":
        return torch.cuda if torch.cuda.is_available() else None
    if device_type == "xpu":
        return torch.xpu if hasattr(torch, "xpu") and torch.xpu.is_available() else None
    return None


def accel_available(device_type: str = "cuda") -> bool:
    return _accel_module(device_type) is not None


def cuda_available() -> bool:
    return accel_available("cuda")


def cuda_mem_info(
    device: Optional[int] = None, device_type: str = "cuda"
) -> Optional[Tuple[int, int]]:
    """(free_bytes, total_bytes) for ``device`` (current device if None).

    Returns None when ``device_type``'s accelerator is unavailable.
    """
    mod = _accel_module(device_type)
    if mod is None:
        return None
    dev = mod.current_device() if device is None else device
    return mod.mem_get_info(dev)


def require_free_vram(
    need_gb: float,
    device: Optional[int] = None,
    wait_s: float = 120.0,
    poll_s: float = 3.0,
    label: str = "",
    device_type: str = "cuda",
) -> None:
    """Block until at least ``need_gb`` is free on the device, else raise.

    Handles realtime contention: if a peer process is transiently holding VRAM,
    wait up to ``wait_s`` for it to release before giving up, instead of starting
    a run that will OOM partway. No-op when ``device_type``'s accelerator is
    unavailable. Raises RuntimeError with a free/total/need breakdown if the
    budget is not met in ``wait_s``.
    """
    if not accel_available(device_type):
        return
    need = int(need_gb * _GB)
    waited = 0.0
    tag = f"[{label}] " if label else ""
    while True:
        free, total = cuda_mem_info(device, device_type)  # type: ignore[misc]
        if free >= need:
            if waited > 0:
                logger.info(
                    "%sVRAM ok after %.0fs: free=%.1fGB need=%.1fGB",
                    tag, waited, free / _GB, need_gb,
                )
            return
        if waited >= wait_s:
            raise RuntimeError(
                f"{tag}insufficient free VRAM: free={free / _GB:.1f}GB "
                f"total={total / _GB:.1f}GB need={need_gb:.1f}GB after waiting "
                f"{wait_s:.0f}s. A co-resident process is holding the GPU; retry "
                f"when it frees or lower the job's batch/seq."
            )
        logger.warning(
            "%swaiting for VRAM: free=%.1fGB need=%.1fGB (%.0f/%.0fs)",
            tag, free / _GB, need_gb, waited, wait_s,
        )
        time.sleep(poll_s)
        waited += poll_s


def cap_process_vram(
    cap_gb: float, device: Optional[int] = None, device_type: str = "cuda"
) -> None:
    """Cap this process's VRAM at ``cap_gb`` via the caching-allocator fraction.

    A ceiling, not a reservation: the caching allocator still grows lazily and
    keeps its freed-block cache, so throughput is unchanged as long as ``cap_gb``
    exceeds the working set. It only prevents this job from growing without bound
    and starving peers. No-op when ``device_type``'s accelerator is unavailable.
    """
    mod = _accel_module(device_type)
    if mod is None:
        return
    dev = mod.current_device() if device is None else device
    _free, total = mod.mem_get_info(dev)
    frac = max(0.0, min(1.0, (cap_gb * _GB) / total))
    mod.set_per_process_memory_fraction(frac, dev)
    logger.info(
        "VRAM cap: %.1fGB (%.1f%% of %.1fGB) on %s:%d",
        cap_gb, frac * 100, total / _GB, device_type, dev,
    )


def oom_retry(
    retries: int = 2,
    backoff_s: float = 5.0,
    device: Optional[int] = None,
    device_type: str = "cuda",
) -> Callable:
    """Decorator: retry a callable on accelerator OOM after empty_cache + backoff.

    For the realtime case where a peer spawns mid-run and momentarily exhausts
    VRAM. Frees our cached blocks, waits, and retries up to ``retries`` times;
    re-raises with a free/total diagnostic if it still fails. ``empty_cache`` runs
    on the OOM path only, so it never touches the steady-state hot loop. The
    wrapped callable is re-invoked from the top on retry, so wrap an idempotent
    unit (an eval pass, a single train step), not a stateful accumulator.

    Catches ``torch.OutOfMemoryError``, the single top-level exception torch
    raises for both cuda and xpu allocator OOMs (``torch.cuda.OutOfMemoryError``
    is the same class object), so this retries correctly regardless of which
    accelerator ``device_type`` names.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            mod = _accel_module(device_type)
            if mod is None:
                return fn(*args, **kwargs)
            import torch

            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except torch.OutOfMemoryError as exc:
                    if attempt >= retries:
                        info = cuda_mem_info(device, device_type)
                        extra = (
                            f" free={info[0] / _GB:.1f}GB total={info[1] / _GB:.1f}GB"
                            if info
                            else ""
                        )
                        raise RuntimeError(
                            f"{device_type} OOM in {fn.__name__} after {retries} "
                            f"retries{extra}. A co-resident process likely grabbed "
                            f"VRAM mid-run."
                        ) from exc
                    attempt += 1
                    logger.warning(
                        "%s OOM in %s; empty_cache + retry %d/%d after %.0fs",
                        device_type, fn.__name__, attempt, retries, backoff_s,
                    )
                    mod.empty_cache()
                    time.sleep(backoff_s)

        return wrapper

    return deco


def pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` currently exists (POSIX)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not signalable by us
    return True


class RunLock:
    """Exclusive run lock via an advisory ``flock`` on a lockfile.

    ``flock`` is atomic -- concurrent acquirers can never both win, so there is no
    check-then-create race -- and the kernel releases it when the holder exits for
    ANY reason (normal, crash, or SIGKILL), so a dead peer never blocks a new run
    and there is no stale-PID reclaim path to get wrong. The lockfile records the
    holder PID for diagnostics only; liveness is decided by the ``flock``, not the
    PID. The fd is held for the lock's lifetime (``release`` or process exit frees
    it); the file is intentionally NOT unlinked on release -- a persistent inode is
    what ``flock`` serializes on, unlink-then-flock reintroduces a race, and a
    leftover empty lockfile is harmless and reused. Usable as a context manager;
    pair ``acquire`` with ``atexit`` for tidy release (the kernel covers crashes).
    """

    def __init__(self, path: str, label: str = ""):
        self.path = path
        self.label = label or os.path.basename(path)
        self._fd: Optional[int] = None

    def acquire(self) -> "RunLock":
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            owner = self._read_pid(fd)
            os.close(fd)
            raise RuntimeError(
                f"run lock {self.path!r} held by a live peer (pid={owner}, "
                f"{self.label}); a concurrent run is active. Refusing to clobber. "
                f"Kill it or wait, then retry."
            )
        # Held: any prior holder is dead (its flock auto-released on exit). Stamp
        # our PID for diagnostics and keep the fd open so the flock persists.
        try:
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
        except OSError:
            pass
        self._fd = fd
        return self

    @staticmethod
    def _read_pid(fd: int) -> Optional[int]:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            data = os.read(fd, 32).decode(errors="ignore").strip()
            return int(data) if data else None
        except (ValueError, OSError):
            return None

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> bool:
        self.release()
        return False
