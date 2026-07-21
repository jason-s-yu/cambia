"""
tests/test_npz_size_guard.py

Tests for the cambia-559 decompression-bomb size guard (src/npz_size_guard.py)
and its two guarded call sites: ReservoirBuffer.load (src/reservoir.py) and
persistence.load_agent_data (src/persistence.py).

Threat: an attacker with rsync write access to ``runs/`` (the actor already
covered by the cambia-552 RCE hardening) plants a crafted ``.npz`` whose
member header declares a multi-GB array shape but whose actual data is
truncated/empty. ``np.load`` allocates the full declared size before reading
any data, so opening such a file OOM-kills the process even though it is only
a few KB on disk.

Bomb archives below are hand-built via numpy.lib.format's own header-writing
helpers, writing ONLY the header bytes (no data), so constructing the test
never allocates the huge array it declares.
"""

import io
import zipfile

import numpy as np
import pytest
from numpy.lib import format as npy_format

from src.cfr.exceptions import NpzSizeGuardError, ReservoirIOError
from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.npz_size_guard import guard_npz_size
from src.reservoir import ReservoirBuffer, ReservoirSample


def _write_bomb_member(zf: zipfile.ZipFile, name: str, shape, dtype="<f8"):
    """Write a .npy member whose header declares `shape` but whose data is
    truncated/empty -- the guard must reject this without numpy ever
    allocating `shape`."""
    buf = io.BytesIO()
    npy_format.write_array_header_1_0(
        buf, {"descr": dtype, "fortran_order": False, "shape": shape}
    )
    zf.writestr(name, buf.getvalue())  # header only, no data bytes


def _write_real_member(zf: zipfile.ZipFile, name: str, array: np.ndarray):
    buf = io.BytesIO()
    npy_format.write_array(buf, array)
    zf.writestr(name, buf.getvalue())


# ---------------------------------------------------------------------------
# Guard helper, exercised directly
# ---------------------------------------------------------------------------


class TestGuardHelperDirect:
    def test_rejects_disproportionate_declared_size(self, tmp_path):
        """A member declaring a huge shape backed by a near-empty compressed
        stream is the ratio-check bomb signature (declared well under the
        absolute cap, so only the ratio heuristic can catch it)."""
        path = tmp_path / "bomb_ratio.npz"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 2e9 float64 == ~16GB declared from a file that is a few hundred
            # bytes on disk.
            _write_bomb_member(zf, "features.npy", (2_000_000_000,))
        assert path.stat().st_size < 1024  # confirm it really is a tiny file

        with pytest.raises(NpzSizeGuardError, match="decompression bomb"):
            guard_npz_size(path)

    def test_rejects_total_declared_over_cap(self, tmp_path, monkeypatch):
        """Real (non-truncated) data whose total declared size exceeds the
        absolute cap is rejected even though its on-disk ratio is
        unremarkable."""
        import src.npz_size_guard as guard_mod

        monkeypatch.setattr(guard_mod, "RUNS_NPZ_MAX_DECLARED_BYTES", 1000)
        path = tmp_path / "over_cap.npz"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_real_member(zf, "features.npy", np.zeros(2000, dtype=np.float32))
        with pytest.raises(NpzSizeGuardError, match="size-guard cap"):
            guard_mod.guard_npz_size(path)

    def test_accepts_legit_small_archive(self, tmp_path):
        path = tmp_path / "legit.npz"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_real_member(
                zf, "features.npy", np.random.randn(100, INPUT_DIM).astype(np.float32)
            )
            _write_real_member(
                zf, "targets.npy", np.random.randn(100, NUM_ACTIONS).astype(np.float32)
            )
        guard_npz_size(path)  # must not raise


# ---------------------------------------------------------------------------
# Call site 1: ReservoirBuffer.load (src/reservoir.py)
# ---------------------------------------------------------------------------


class TestReservoirLoadGuarded:
    def _make_buffer_and_save(self, tmp_path, n=5):
        buf = ReservoirBuffer(
            capacity=10, input_dim=INPUT_DIM, target_dim=NUM_ACTIONS, has_mask=True
        )
        for i in range(n):
            buf.add(
                ReservoirSample(
                    features=np.full(INPUT_DIM, float(i), dtype=np.float32),
                    target=np.full(NUM_ACTIONS, float(i), dtype=np.float32),
                    action_mask=np.ones(NUM_ACTIONS, dtype=bool),
                    iteration=i,
                )
            )
        path = str(tmp_path / "reservoir.npz")
        buf.save(path)
        return buf, path

    def test_legit_round_trip_unaffected(self, tmp_path):
        """Guard must not disturb a normal save/load round trip."""
        buf, path = self._make_buffer_and_save(tmp_path)
        loaded = ReservoirBuffer(
            capacity=10, input_dim=INPUT_DIM, target_dim=NUM_ACTIONS, has_mask=True
        )
        loaded.load(path)  # must not raise
        assert loaded._size == buf._size
        np.testing.assert_array_equal(
            loaded._features[: loaded._size], buf._features[: buf._size]
        )

    def test_bomb_archive_rejected_before_load(self, tmp_path):
        path = tmp_path / "bomb.npz"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_bomb_member(zf, "features.npy", (2_000_000_000,))
            _write_real_member(zf, "meta.npy", np.array([0, 10], dtype=np.int64))

        buf = ReservoirBuffer(
            capacity=10, input_dim=INPUT_DIM, target_dim=NUM_ACTIONS, has_mask=True
        )
        with pytest.raises(ReservoirIOError, match="cambia-559"):
            buf.load(str(path))
        # Buffer must be left untouched -- never partially loaded.
        assert buf._size == 0


# ---------------------------------------------------------------------------
# Call site 2: persistence.load_agent_data (src/persistence.py)
# ---------------------------------------------------------------------------


class TestLoadAgentDataGuarded:
    def test_legit_round_trip_unaffected(self, tmp_path):
        from src.persistence import load_agent_data, save_agent_data

        data = {
            "regret_sum": {(1, 2): np.array([0.1, 0.2], dtype=np.float64)},
            "strategy_sum": {(1, 2): np.array([0.5, 0.5], dtype=np.float64)},
            "reach_prob_sum": {(1, 2): 1.0},
            "iteration": 3,
            "exploitability_results": [(1, 0.5)],
        }
        path = str(tmp_path / "agent.pt")
        assert save_agent_data(data, path) is True

        loaded = load_agent_data(path)
        assert loaded is not None
        assert loaded["iteration"] == 3
        np.testing.assert_array_equal(loaded["regret_sum"][(1, 2)], [0.1, 0.2])

    def test_bomb_archive_rejected_before_load(self, tmp_path):
        from src.persistence import load_agent_data

        path = tmp_path / "agent.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_bomb_member(zf, "regret_flat.npy", (2_000_000_000,))

        with pytest.raises(NpzSizeGuardError, match="cambia-559"):
            load_agent_data(str(path))
