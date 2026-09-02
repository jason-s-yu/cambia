"""
tests/test_ffi_abi_version.py

The libcambia.so ABI version handshake (cambia-1689): bridge.py pins an
expected ABI generation (ABI_GENERATION, paired by hand with the Go constant
engine/cgo/abiver.Generation) and refuses to load a shared library that
reports anything else, including a library built before this handshake
existed, which exports no cambia_abi_generation symbol at all and is treated
as generation 0.

Without this, an appended constructor argument or a widened read-back record
is silently ignored by the SysV calling convention: a stale .so keeps loading
and keeps answering, just with the wrong rules, and only a caller lucky
enough to carry its own length check (cambia_game_get_house_rules) ever
notices, after the game was already built with the wrong rules.

The mismatch/missing-symbol cases below monkeypatch the loader rather than
building a real stale .so, so they run without libcambia.so present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ffi import bridge


class _FakeAbiLib:
    """Stands in for a cffi dlopen() result reporting a chosen ABI generation."""

    def __init__(self, generation: int):
        self._generation = generation

    def cambia_abi_generation(self):
        return self._generation

    def cambia_abi_commit(self, out_buf, buf_len):
        return 0


class _FakeNoVersionLib:
    """Stands in for a pre-handshake .so: no cambia_abi_generation export."""


@pytest.fixture
def _reset_lib_singleton():
    """bridge._LIB is a module-level singleton; isolate mutations to it."""
    saved = bridge._LIB
    bridge._LIB = None
    yield
    bridge._LIB = saved


def _patch_dlopen(monkeypatch, fake_lib, tmp_path) -> Path:
    fake_so = tmp_path / "libcambia.so"
    fake_so.write_bytes(b"")
    monkeypatch.setattr(bridge._ffi, "dlopen", lambda path: fake_lib)
    monkeypatch.setenv("LIBCAMBIA_PATH", str(fake_so))
    return fake_so


def test_generation_mismatch_refused(monkeypatch, tmp_path, _reset_lib_singleton):
    """A library reporting a different generation is refused, naming both
    numbers and the rebuild command."""
    wrong_generation = bridge.ABI_GENERATION + 1
    so_path = _patch_dlopen(monkeypatch, _FakeAbiLib(wrong_generation), tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        bridge._get_lib()

    message = str(exc_info.value)
    assert f"expects generation {bridge.ABI_GENERATION}" in message
    assert f"reports {wrong_generation}" in message
    assert "make libcambia" in message
    assert str(so_path) in message


def test_missing_version_symbol_treated_as_generation_zero(
    monkeypatch, tmp_path, _reset_lib_singleton
):
    """A library with no cambia_abi_generation export at all (older than this
    handshake) is refused as generation 0, not a crash on the missing symbol."""
    _patch_dlopen(monkeypatch, _FakeNoVersionLib(), tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        bridge._get_lib()

    message = str(exc_info.value)
    assert f"expects generation {bridge.ABI_GENERATION}" in message
    assert "reports 0" in message
    assert "make libcambia" in message


def test_matching_generation_loads(monkeypatch, tmp_path, _reset_lib_singleton):
    """A library reporting the expected generation loads without error."""
    _patch_dlopen(monkeypatch, _FakeAbiLib(bridge.ABI_GENERATION), tmp_path)

    lib = bridge._get_lib()

    assert lib.cambia_abi_generation() == bridge.ABI_GENERATION


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        engine = GoEngine.from_deck(list(range(54)))
        engine.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


@skip_if_no_go
def test_real_library_reports_expected_generation():
    """The libcambia.so this test run is actually loading passes the same
    handshake bridge.py enforces on every load. A failure here means the
    built .so and this bridge module have drifted; rebuild with
    `make libcambia`."""
    lib = bridge._get_lib()
    assert int(lib.cambia_abi_generation()) == bridge.ABI_GENERATION
