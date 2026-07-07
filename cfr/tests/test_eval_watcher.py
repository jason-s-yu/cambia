"""
tests/test_eval_watcher.py

Tests for scripts/eval_watcher.py's checkpoint-prefix inference, in
particular that it stays in sync with run_db.ALGO_TO_CHECKPOINT_PREFIX for
ppo instead of drifting via a duplicated local mapping.
"""

import sys
from pathlib import Path

# Ensure cfr root is importable (conftest.py also handles this).
_CFR_ROOT = Path(__file__).resolve().parent.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

from scripts.eval_watcher import _infer_checkpoint_prefix
from src.run_db import ALGO_TO_CHECKPOINT_PREFIX


class TestInferCheckpointPrefix:
    def test_ppo_matches_run_db_authoritative_value(self):
        """ppo agent_type must resolve to the same prefix run_db uses for the
        ppo algorithm (ppo_train.py's actual save-stem: "ppo_model"), not the
        stale "ppo_checkpoint" the local map used to carry."""
        assert _infer_checkpoint_prefix("ppo") == ALGO_TO_CHECKPOINT_PREFIX["ppo"]
        assert _infer_checkpoint_prefix("ppo") == "ppo_model"

    def test_explicit_prefix_overrides_inference(self):
        assert _infer_checkpoint_prefix("ppo", "custom_prefix") == "custom_prefix"
        assert _infer_checkpoint_prefix("deep_cfr", "custom_prefix") == "custom_prefix"

    def test_known_agent_types(self):
        assert _infer_checkpoint_prefix("rebel") == "rebel_checkpoint"
        assert _infer_checkpoint_prefix("deep_cfr") == "deep_cfr_checkpoint"
        assert _infer_checkpoint_prefix("sd_cfr") == "deep_cfr_checkpoint"
        assert _infer_checkpoint_prefix("escher") == "deep_cfr_checkpoint"
        assert _infer_checkpoint_prefix("nplayer") == "deep_cfr_checkpoint"
        assert _infer_checkpoint_prefix("gtcfr") == "gtcfr_checkpoint"
        assert _infer_checkpoint_prefix("sog") == "sog_checkpoint"
        assert _infer_checkpoint_prefix("sog_inference") == "sog_checkpoint"

    def test_unknown_agent_type_falls_back_to_deep_cfr(self):
        assert _infer_checkpoint_prefix("some_unknown_type") == "deep_cfr_checkpoint"
