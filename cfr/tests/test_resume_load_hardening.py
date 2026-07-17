"""
tests/test_resume_load_hardening.py

Round-trip + rejection tests for W3 resume-load hardening (cambia-552).

Threat: an attacker with rsync write access to ``runs/`` plants a malicious
pickle that executes on the next ``torch.load`` / ``joblib.load`` during resume
or eval on the runner. Every such site was an explicit ``weights_only=False``
opt-out (or unrestricted ``joblib.load``); W3 tightens each one. These tests
assert both halves of the fix: a normally-written artifact still loads under the
hardened loader, and a legacy / whole-module pickle is refused rather than run.

Sites:
  1. prtcfr_eval._load_net             -> weights_only=True, pinned-format only
  2. persistence.save/load_agent_data  -> npz (no pickle); legacy joblib refused
  3. prtcfr rolling-checkpoint payload -> loads under weights_only=True (w/ config)
  4. desca load_checkpoint payload     -> loads under weights_only=True (rng/opt)
  5. cli `info` .joblib branch          -> hardened loader, no unrestricted pickle
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


# ---------------------------------------------------------------------------
# Site 2: persistence agent-data npz round-trip + legacy-pickle rejection
# ---------------------------------------------------------------------------


def _sample_agent_data():
    """A tabular-CFR payload shaped exactly as data_manager_mixin.save_data
    produces it: keys are ``InfosetKey.astuple()`` tuples (two nested int-tuple
    fields + five int fields), regret/strategy values are ragged float64 arrays,
    reach-prob values are floats."""
    k1 = ((1, 2, 3), (4, 5), 2, 7, 1, 0, 3)
    k2 = ((9,), (), 0, 12, 2, 1, 5)
    return {
        "regret_sum": {
            k1: np.array([0.5, -1.25, 3.0], dtype=np.float64),
            k2: np.array([2.0], dtype=np.float64),
        },
        "strategy_sum": {
            k1: np.array([1.0, 1.0, 1.0], dtype=np.float64),
            k2: np.array([4.0], dtype=np.float64),
        },
        "reach_prob_sum": {k1: 2.5, k2: 0.75},
        "iteration": 42,
        "exploitability_results": [(1, 0.9), (10, 0.42)],
    }


class TestPersistenceNpzRoundTrip:
    def test_round_trip_equal(self, tmp_path):
        from src.persistence import load_agent_data, save_agent_data

        data = _sample_agent_data()
        path = str(tmp_path / "agent.pt")
        assert save_agent_data(data, path) is True

        loaded = load_agent_data(path)
        assert loaded is not None
        assert loaded["iteration"] == 42
        assert loaded["exploitability_results"] == [(1, 0.9), (10, 0.42)]
        for name in ("regret_sum", "strategy_sum"):
            assert set(loaded[name].keys()) == set(data[name].keys())
            for k in data[name]:
                np.testing.assert_array_equal(loaded[name][k], data[name][k])
                assert loaded[name][k].dtype == np.float64
        assert loaded["reach_prob_sum"] == data["reach_prob_sum"]

    def test_keys_reconstruct_as_hashable_infosetkeys(self, tmp_path):
        """Keys must come back as tuples with nested tuple fields so
        ``InfosetKey(*k)`` reconstructs a valid, hashable key (the exact path
        data_manager_mixin.load_data takes)."""
        from src.persistence import load_agent_data, save_agent_data
        from src.utils import InfosetKey

        path = str(tmp_path / "agent.pt")
        save_agent_data(_sample_agent_data(), path)
        loaded = load_agent_data(path)

        for k in loaded["regret_sum"]:
            assert isinstance(k, tuple)
            assert isinstance(k[0], tuple) and isinstance(k[1], tuple)
            ik = InfosetKey(*k)  # must not raise
            hash(ik)  # must be hashable

    def test_file_is_zip_archive_loadable_without_pickle(self, tmp_path):
        from src.persistence import save_agent_data

        path = str(tmp_path / "agent.pt")
        save_agent_data(_sample_agent_data(), path)
        with open(path, "rb") as fh:
            assert fh.read(4) == b"PK\x03\x04"  # numpy .npz == ZIP magic
        # And it deserializes with pickling fully disabled.
        with open(path, "rb") as fh:
            with np.load(fh, allow_pickle=False) as npz:
                assert int(npz["iteration"][0]) == 42

    def test_missing_and_empty_return_none(self, tmp_path):
        from src.persistence import load_agent_data

        assert load_agent_data(str(tmp_path / "nope.pt")) is None
        empty = tmp_path / "empty.pt"
        empty.write_bytes(b"")
        assert load_agent_data(str(empty)) is None

    def test_legacy_pickle_rejected_not_executed(self, tmp_path):
        """A legacy joblib/pickle artifact (or a poisoned one) must be refused
        with a clear error, never unpickled."""
        from src.persistence import load_agent_data

        legacy = tmp_path / "legacy.joblib"
        legacy.write_bytes(pickle.dumps(_sample_agent_data()))
        with pytest.raises(ValueError, match="cambia-552"):
            load_agent_data(str(legacy))

    def test_empty_dicts_round_trip(self, tmp_path):
        from src.persistence import load_agent_data, save_agent_data

        data = {
            "regret_sum": {},
            "strategy_sum": {},
            "reach_prob_sum": {},
            "iteration": 0,
            "exploitability_results": [],
        }
        path = str(tmp_path / "empty_dicts.pt")
        assert save_agent_data(data, path) is True
        loaded = load_agent_data(path)
        assert loaded["regret_sum"] == {}
        assert loaded["strategy_sum"] == {}
        assert loaded["reach_prob_sum"] == {}
        assert loaded["iteration"] == 0
        assert loaded["exploitability_results"] == []


# ---------------------------------------------------------------------------
# Site 1: prtcfr_eval._load_net -- pinned-format-only after fallback removal
# ---------------------------------------------------------------------------


class TestLoadNetHardened:
    def _prtcfr_eval(self):
        return pytest.importorskip("src.cfr.prtcfr_eval")

    def _make_net(self):
        prtcfr_net = pytest.importorskip("src.cfr.prtcfr_net")
        return prtcfr_net.PRTCFRNet(device="cpu")

    def test_pinned_snapshot_still_loads(self, tmp_path):
        """A normally-written snapshot ({encoder_state_dict, head_state_dict,
        iteration}) still loads after the whole-module fallback was removed, and
        reconstructs identical weights."""
        prtcfr_eval = self._prtcfr_eval()
        net = self._make_net()
        path = str(tmp_path / "prtcfr_snapshot_iter_5.pt")
        torch.save(
            {
                "encoder_state_dict": net.encoder_state_dict(),
                "head_state_dict": net.head_state_dict(),
                "iteration": 5,
            },
            path,
        )
        loaded = prtcfr_eval._load_net(path, device="cpu")
        assert isinstance(loaded, torch.nn.Module)

        orig = net.encoder_state_dict()
        got = loaded.encoder_state_dict()
        assert set(orig.keys()) == set(got.keys())
        for k in orig:
            assert torch.allclose(orig[k], got[k])

    def test_whole_module_pickle_refused(self, tmp_path):
        """A pickled whole ``nn.Module`` (the removed RCE-shaped fallback) is
        refused under weights_only=True instead of being deserialized."""
        prtcfr_eval = self._prtcfr_eval()
        net = self._make_net()
        path = str(tmp_path / "prtcfr_snapshot_iter_1.pt")
        torch.save(net, path)  # whole-module pickle
        with pytest.raises(Exception):
            prtcfr_eval._load_net(path, device="cpu")

    def test_non_pinned_dict_refused(self, tmp_path):
        """A dict that is not the pinned format is rejected with a clear error,
        not coerced through a state_dict fallback."""
        prtcfr_eval = self._prtcfr_eval()
        path = str(tmp_path / "prtcfr_snapshot_iter_1.pt")
        torch.save({"model_state_dict": {"w": torch.zeros(2)}}, path)
        with pytest.raises(ValueError, match="pinned"):
            prtcfr_eval._load_net(path, device="cpu")


# ---------------------------------------------------------------------------
# Site 3: prtcfr rolling-checkpoint payload loads under weights_only=True
# ---------------------------------------------------------------------------


class TestPrtcfrCheckpointWeightsOnly:
    def test_rolling_checkpoint_with_config_round_trips(self, tmp_path):
        """The rolling checkpoint carries a ``model_dump()`` config dict. Verify
        the real PRTCFRConfig schema survives weights_only=True (the site-3
        risk), matching the exact writer shape."""
        from src.config import PRTCFRConfig

        cfg = PRTCFRConfig().model_dump()
        payload = {
            "encoder_state_dict": {"gru.weight_ih_l0": torch.randn(4, 3)},
            "head_state_dict": {"lin.bias": torch.zeros(2)},
            "config": cfg,
            "iteration": 11,
        }
        path = str(tmp_path / "prtcfr_checkpoint.pt")
        torch.save(payload, path)

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["iteration"] == 11
        assert loaded["config"] == cfg
        assert torch.allclose(
            loaded["encoder_state_dict"]["gru.weight_ih_l0"],
            payload["encoder_state_dict"]["gru.weight_ih_l0"],
        )


# ---------------------------------------------------------------------------
# Site 4: desca load_checkpoint payload loads under weights_only=True
# ---------------------------------------------------------------------------


class TestDescaCheckpointWeightsOnly:
    def test_desca_payload_round_trips(self, tmp_path):
        """Mirror the desca save_checkpoint payload: net tensors + optimizer
        state + a numpy PCG64 bit-generator state dict + config + float
        histories. All must survive weights_only=True, and the restored RNG
        state must reproduce the stream (end-to-end net/optimizer equality is
        covered by test_desca_trainer.test_checkpoint_roundtrip_identical_state)."""
        rng = np.random.default_rng(123)
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()

        payload = {
            "algorithm": "desca",
            "iteration": 9,
            "regret_state_dict": model.state_dict(),
            "regret_optimizer_state_dict": opt.state_dict(),
            "desca_state_dict": {
                "iteration": 9,
                "config": {"hidden_dim": 512, "lr": 0.001},
                "rng_state": rng.bit_generator.state,
                "v_loss_history": [0.1, 0.2],
            },
        }
        path = str(tmp_path / "desca_iter_9.pt")
        torch.save(payload, path)

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["iteration"] == 9
        assert "state" in loaded["regret_optimizer_state_dict"]
        restored_rng_state = loaded["desca_state_dict"]["rng_state"]
        assert restored_rng_state == rng.bit_generator.state

        rng2 = np.random.default_rng()
        rng2.bit_generator.state = restored_rng_state
        assert rng.random() == rng2.random()


# ---------------------------------------------------------------------------
# Site 5: cli `info` .joblib branch no longer uses unrestricted joblib.load
# ---------------------------------------------------------------------------


class TestCliInfoHardened:
    def test_info_branch_drops_unrestricted_joblib_load(self):
        source = (_SRC_ROOT / "cli.py").read_text()
        assert "joblib.load" not in source
        assert "from .persistence import load_agent_data" in source
