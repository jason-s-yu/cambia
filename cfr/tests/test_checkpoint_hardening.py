"""
tests/test_checkpoint_hardening.py

Unit tests for Workstream A: Checkpoint Hardening.

Covers:
  - atomic_torch_save: write-then-rename atomicity, crash safety
  - atomic_npz_save: write-then-rename atomicity
  - weights_only=True at all torch.load sites
  - DeepCFRConfig mutual exclusion: pipeline_training + num_traversal_threads
"""

import inspect
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# atomic_torch_save
# ---------------------------------------------------------------------------


class TestAtomicTorchSave:
    def test_file_created(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        data = {"x": torch.tensor([1, 2, 3])}
        atomic_torch_save(data, path)
        assert os.path.exists(path)

    def test_content_correct(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        data = {"key": torch.tensor([42.0])}
        atomic_torch_save(data, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.allclose(loaded["key"], data["key"])

    def test_no_tmp_file_left_on_success(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        atomic_torch_save({"v": torch.zeros(1)}, path)
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_no_partial_file_on_failure(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")

        # Simulate torch.save raising mid-write
        with patch("torch.save", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_torch_save({"v": torch.zeros(1)}, path)

        # The target file must not exist
        assert not os.path.exists(path)
        # No stray .tmp files either
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_creates_parent_directory(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "nested" / "dir" / "checkpoint.pt")
        atomic_torch_save({"v": torch.zeros(1)}, path)
        assert os.path.exists(path)

    def test_overwrites_existing_file_atomically(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        atomic_torch_save({"v": torch.tensor([1.0])}, path)
        atomic_torch_save({"v": torch.tensor([2.0])}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["v"].item() == 2.0


# ---------------------------------------------------------------------------
# atomic_npz_save
# ---------------------------------------------------------------------------


class TestAtomicNpzSave:
    def _dummy_save_fn(self, arr, base_path):
        """Mimics ReservoirBuffer.save: writes <base_path>.npz."""
        np.savez(base_path, data=arr)

    def test_file_created(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([1, 2, 3])
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        assert os.path.exists(base + ".npz")

    def test_content_correct(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([10.0, 20.0])
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        loaded = np.load(base + ".npz")
        np.testing.assert_array_equal(loaded["data"], arr)

    def test_no_tmp_file_left_on_success(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.zeros(4)
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp") or f.endswith(".tmp.npz")]
        assert tmp_files == []

    def test_no_partial_file_on_failure(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")

        def failing_save(p):
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            atomic_npz_save(failing_save, base)

        assert not os.path.exists(base + ".npz")

    def test_accepts_path_with_npz_extension(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([5])
        # Pass path that already has .npz
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base + ".npz")
        assert os.path.exists(base + ".npz")


# ---------------------------------------------------------------------------
# weights_only=True at load sites — source-level inspection (no module import)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _read_source(rel_path: str) -> str:
    return (_SRC_ROOT / rel_path).read_text()


def _find_torch_load_calls(source: str):
    """Return list of torch.load(...) calls from a source string."""
    import re
    # Match multi-line torch.load(...) by finding the opening and scanning to close paren
    calls = re.findall(r"torch\.load\((?:[^()]*|\([^()]*\))*\)", source)
    return calls


class TestWeightsOnlyAtLoadSites:
    def test_deep_trainer_load_checkpoint_uses_weights_only_true(self):
        source = _read_source("cfr/deep_trainer.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in deep_trainer.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )

    def test_main_train_inspect_checkpoint_uses_weights_only_true(self):
        source = _read_source("main_train.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in main_train.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )

    def test_cli_uses_weights_only_true(self):
        source = _read_source("cli.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in cli.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )


# ---------------------------------------------------------------------------
# Mutual exclusion guard — source-level inspection + minimal dataclass exec
# ---------------------------------------------------------------------------


def _extract_deep_cfr_config_class() -> type:
    """
    Extract and eval the DeepCFRConfig dataclass from deep_trainer.py without
    importing the full module (which has unavailable deps like rich, typer).
    """
    import re
    from dataclasses import dataclass

    source = _read_source("cfr/deep_trainer.py")

    # Extract class body: from @dataclass\nclass DeepCFRConfig: to the next top-level def/class
    match = re.search(
        r"(@dataclass\nclass DeepCFRConfig:.*?)(?=\n@|\ndef |\nclass )",
        source,
        re.DOTALL,
    )
    assert match, "Could not find DeepCFRConfig class in deep_trainer.py"
    class_src = match.group(1)

    # Build a minimal namespace to exec the class definition
    from src.encoding import INPUT_DIM, NUM_ACTIONS
    namespace = {
        "dataclass": dataclass,
        "INPUT_DIM": INPUT_DIM,
        "NUM_ACTIONS": NUM_ACTIONS,
    }
    exec(compile(class_src, "<DeepCFRConfig>", "exec"), namespace)
    return namespace["DeepCFRConfig"]


class TestDeepCFRConfigMutualExclusion:
    @pytest.fixture(scope="class")
    def DeepCFRConfig(self):
        return _extract_deep_cfr_config_class()

    def test_pipeline_true_threads_1_ok(self, DeepCFRConfig):
        cfg = DeepCFRConfig(pipeline_training=True, num_traversal_threads=1)
        assert cfg.pipeline_training is True

    def test_pipeline_false_threads_many_ok(self, DeepCFRConfig):
        cfg = DeepCFRConfig(pipeline_training=False, num_traversal_threads=4)
        assert cfg.num_traversal_threads == 4

    def test_pipeline_true_threads_many_raises(self, DeepCFRConfig):
        with pytest.raises(ValueError, match="mutually exclusive"):
            DeepCFRConfig(pipeline_training=True, num_traversal_threads=2)

    def test_pipeline_true_threads_many_raises_message_helpful(self, DeepCFRConfig):
        with pytest.raises(ValueError) as exc_info:
            DeepCFRConfig(pipeline_training=True, num_traversal_threads=4)
        assert "pipeline_training" in str(exc_info.value)
        assert "num_traversal_threads" in str(exc_info.value)

    def test_default_config_valid(self, DeepCFRConfig):
        """Default config must not raise (pipeline_training=True, threads=1)."""
        cfg = DeepCFRConfig()
        assert cfg.pipeline_training is True
        assert cfg.num_traversal_threads == 1
