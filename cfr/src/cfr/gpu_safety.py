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
when CUDA is unavailable, so CPU-only runs and tests import and call them freely.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_GB = 1024 ** 3


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_mem_info(device: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """(free_bytes, total_bytes) for ``device`` (current device if None).

    Returns None when CUDA is unavailable.
    """
    if not cuda_available():
        return None
    import torch

    dev = torch.cuda.current_device() if device is None else device
    return torch.cuda.mem_get_info(dev)


def require_free_vram(
    need_gb: float,
    device: Optional[int] = None,
    wait_s: float = 120.0,
    poll_s: float = 3.0,
    label: str = "",
) -> None:
    """Block until at least ``need_gb`` is free on the device, else raise.

    Handles realtime contention: if a peer process is transiently holding VRAM,
    wait up to ``wait_s`` for it to release before giving up, instead of starting
    a run that will OOM partway. No-op when CUDA is unavailable. Raises
    RuntimeError with a free/total/need breakdown if the budget is not met in
    ``wait_s``.
    """
    if not cuda_available():
        return
    need = int(need_gb * _GB)
    waited = 0.0
    tag = f"[{label}] " if label else ""
    while True:
        free, total = cuda_mem_info(device)  # type: ignore[misc]
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


def cap_process_vram(cap_gb: float, device: Optional[int] = None) -> None:
    """Cap this process's VRAM at ``cap_gb`` via the caching-allocator fraction.

    A ceiling, not a reservation: the caching allocator still grows lazily and
    keeps its freed-block cache, so throughput is unchanged as long as ``cap_gb``
    exceeds the working set. It only prevents this job from growing without bound
    and starving peers. No-op when CUDA is unavailable.
    """
    if not cuda_available():
        return
    import torch

    dev = torch.cuda.current_device() if device is None else device
    _free, total = torch.cuda.mem_get_info(dev)
    frac = max(0.0, min(1.0, (cap_gb * _GB) / total))
    torch.cuda.set_per_process_memory_fraction(frac, dev)
    logger.info(
        "VRAM cap: %.1fGB (%.1f%% of %.1fGB) on cuda:%d",
        cap_gb, frac * 100, total / _GB, dev,
    )


def oom_retry(
    retries: int = 2, backoff_s: float = 5.0, device: Optional[int] = None
) -> Callable:
    """Decorator: retry a callable on CUDA OOM after empty_cache + backoff.

    For the realtime case where a peer spawns mid-run and momentarily exhausts
    VRAM. Frees our cached blocks, waits, and retries up to ``retries`` times;
    re-raises with a free/total diagnostic if it still fails. ``empty_cache`` runs
    on the OOM path only, so it never touches the steady-state hot loop. The
    wrapped callable is re-invoked from the top on retry, so wrap an idempotent
    unit (an eval pass, a single train step), not a stateful accumulator.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not cuda_available():
                return fn(*args, **kwargs)
            import torch

            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except torch.cuda.OutOfMemoryError as exc:
                    if attempt >= retries:
                        info = cuda_mem_info(device)
                        extra = (
                            f" free={info[0] / _GB:.1f}GB total={info[1] / _GB:.1f}GB"
                            if info
                            else ""
                        )
                        raise RuntimeError(
                            f"CUDA OOM in {fn.__name__} after {retries} retries"
                            f"{extra}. A co-resident process likely grabbed VRAM "
                            f"mid-run."
                        ) from exc
                    attempt += 1
                    logger.warning(
                        "CUDA OOM in %s; empty_cache + retry %d/%d after %.0fs",
                        fn.__name__, attempt, retries, backoff_s,
                    )
                    torch.cuda.empty_cache()
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
