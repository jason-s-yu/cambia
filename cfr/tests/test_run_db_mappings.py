"""Tests for infer_algorithm gtcfr/sog detection and algo mapping tables."""

import pytest

from src.run_db import (
    ALGO_TO_AGENT_TYPE,
    ALGO_TO_CHECKPOINT_PREFIX,
    algo_to_agent_type,
    algo_to_checkpoint_prefix,
    infer_algorithm,
)

# --- infer_algorithm: checkpoint_keys detection ---


def test_infer_algorithm_gtcfr_from_checkpoint_keys():
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "optimizer_state_dict", "epoch"},
    )
    assert result == "gtcfr"


def test_infer_algorithm_sog_from_checkpoint_keys():
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "sog_metadata", "optimizer_state_dict"},
    )
    assert result == "sog"


def test_infer_algorithm_sog_over_gtcfr():
    """When both cvpn_state_dict and sog_metadata present, sog wins."""
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "sog_metadata"},
    )
    assert result == "sog"


# --- infer_algorithm: config dict detection ---


def test_infer_algorithm_gtcfr_from_config():
    result = infer_algorithm(
        config_dict={"deep_cfr": {"gtcfr_epochs": 200}},
    )
    assert result == "gtcfr"


def test_infer_algorithm_sog_from_config():
    result = infer_algorithm(
        config_dict={"deep_cfr": {"sog_epochs": 500}},
    )
    assert result == "sog"


# --- algo_to_agent_type ---


def test_algo_to_agent_type_all_entries():
    expected = {
        "rebel": "rebel",
        "os-mccfr": "deep_cfr",
        "es-mccfr": "deep_cfr",
        "escher": "escher",
        "sd-cfr": "sd_cfr",
        "gtcfr": "gtcfr",
        "sog": "sog_inference",
        "psro": "deep_cfr",
    }
    for algo, agent_type in expected.items():
        assert algo_to_agent_type(algo) == agent_type, f"Failed for {algo}"
        assert ALGO_TO_AGENT_TYPE[algo] == agent_type, f"Dict mismatch for {algo}"


def test_algo_to_agent_type_fallback():
    assert algo_to_agent_type("unknown-algo") == "deep_cfr"


# --- algo_to_checkpoint_prefix ---


def test_algo_to_checkpoint_prefix_all_entries():
    expected = {
        "rebel": "rebel_checkpoint",
        "os-mccfr": "deep_cfr_checkpoint",
        "es-mccfr": "deep_cfr_checkpoint",
        "escher": "deep_cfr_checkpoint",
        "sd-cfr": "deep_cfr_checkpoint",
        "gtcfr": "gtcfr_checkpoint",
        "sog": "sog_checkpoint",
        "psro": "deep_cfr_checkpoint",
        "ppo": "ppo_model",
    }
    for algo, prefix in expected.items():
        assert algo_to_checkpoint_prefix(algo) == prefix, f"Failed for {algo}"
        assert ALGO_TO_CHECKPOINT_PREFIX[algo] == prefix, f"Dict mismatch for {algo}"


def test_algo_to_checkpoint_prefix_fallback():
    assert algo_to_checkpoint_prefix("unknown-algo") == "deep_cfr_checkpoint"


def test_algo_to_checkpoint_prefix_ppo_matches_e2_trainer_save_naming():
    """ppo_train.py saves under the agent_data_save_path stem (e.g. "ppo_model"),
    with periodic/eval callbacks writing "ppo_model_steps_<N>.zip" and
    "ppo_model_eval_<N>.zip" (see config/v0.4-e2-ppo-selfplay.yaml
    agent_data_save_path). Run-dir auto-detect must glob on that stem, not on
    a "ppo_checkpoint" prefix the trainer never writes.
    """
    assert algo_to_checkpoint_prefix("ppo") == "ppo_model"
    assert "ppo_model_steps_1000.zip".startswith(ALGO_TO_CHECKPOINT_PREFIX["ppo"])
    assert "ppo_model_eval_1000.zip".startswith(ALGO_TO_CHECKPOINT_PREFIX["ppo"])


def test_get_db_honors_cambia_run_db_env(tmp_path, monkeypatch):
    """CAMBIA_RUN_DB points get_db's default at the per-run journal the serving
    harness syncs (design 4.2); an explicit db_path argument still wins."""
    from src.run_db import get_db

    journal = tmp_path / "runs" / "some-run" / "run_db.sqlite"
    monkeypatch.setenv("CAMBIA_RUN_DB", str(journal))
    db = get_db()
    db.execute("SELECT 1")
    db.close()
    assert journal.exists()

    explicit = tmp_path / "explicit.db"
    db = get_db(str(explicit))
    db.close()
    assert explicit.exists()


def test_upsert_run_status_none_preserves_existing(tmp_path):
    """status=None attaches data without touching lifecycle status: an existing
    row keeps its status, a fresh insert lands on 'created'."""
    from src.run_db import get_db, upsert_run

    db = get_db(str(tmp_path / "t.db"))
    upsert_run(db, name="r1", algorithm="prt-cfr", status="completed")
    upsert_run(db, name="r1", algorithm="prt-cfr", status=None)
    row = db.execute("SELECT status FROM runs WHERE name='r1'").fetchone()
    assert row["status"] == "completed"

    upsert_run(db, name="r2", algorithm="prt-cfr", status=None)
    row = db.execute("SELECT status FROM runs WHERE name='r2'").fetchone()
    assert row["status"] == "created"
    db.close()
