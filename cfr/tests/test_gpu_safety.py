"""CPU tests for src.cfr.gpu_safety failsafes.

RunLock (flock-based) and pid_alive are fully CPU-testable. The VRAM helpers are
validated in their CUDA-absent no-op behavior (run with CUDA hidden); their
CUDA-present paths (cap, OOM retry under real pressure) are exercised when the X2
run and capacity probe actually run on the GPU.
"""

import os

import pytest

from src.cfr.gpu_safety import (
    RunLock,
    cap_process_vram,
    cuda_available,
    oom_retry,
    pid_alive,
    require_free_vram,
)


def test_pid_alive_self_and_dead():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_000_000_000) is False  # implausible pid
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_runlock_acquire_release_roundtrip(tmp_path):
    p = str(tmp_path / "run.lock")
    lock = RunLock(p).acquire()
    assert os.path.exists(p)
    with open(p) as f:
        assert int(f.read().strip()) == os.getpid()  # PID stamped for diagnostics
    lock.release()
    # flock freed (the file may persist by design); a fresh acquire must succeed.
    RunLock(p).acquire().release()


def test_runlock_context_manager(tmp_path):
    p = str(tmp_path / "run.lock")
    with RunLock(p):
        assert os.path.exists(p)
    # After the context exits the lock is freed -> re-acquirable.
    RunLock(p).acquire().release()


def test_runlock_refuses_live_peer(tmp_path):
    p = str(tmp_path / "run.lock")
    peer = RunLock(p, label="peer").acquire()  # a live peer run holds the flock
    try:
        with pytest.raises(RuntimeError, match="concurrent run is active"):
            RunLock(p, label="mine").acquire()
    finally:
        peer.release()
    # Peer released -> now acquirable.
    RunLock(p).acquire().release()


def test_runlock_reclaims_dead_peer(tmp_path):
    # A crashed peer left its PID in the file; the kernel released its flock on
    # death, so the lock is free -> acquire succeeds and restamps our PID. No
    # stale-PID check needed (that race is gone with flock).
    p = str(tmp_path / "run.lock")
    with open(p, "w") as f:
        f.write("424242")
    lock = RunLock(p).acquire()
    assert int(open(p).read().strip()) == os.getpid()
    lock.release()


def test_runlock_reclaims_empty(tmp_path):
    p = str(tmp_path / "run.lock")
    open(p, "w").close()  # empty/garbage lockfile (crashed mid-write), no holder
    lock = RunLock(p).acquire()
    assert int(open(p).read().strip()) == os.getpid()
    lock.release()


@pytest.mark.skipif(cuda_available(), reason="CPU-only no-op assertions")
def test_vram_helpers_noop_without_cuda():
    # No CUDA -> these must return immediately and never raise.
    require_free_vram(9999.0, wait_s=0.0)
    cap_process_vram(9999.0)

    @oom_retry(retries=3)
    def f(x):
        return x + 1

    assert f(41) == 42
