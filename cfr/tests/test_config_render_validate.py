"""
tests/test_config_render_validate.py

Tests for the `cambia config render` / `cambia config validate` CLI commands
(Phase 2 Sprint 1, S1T4): _base resolution, dotted `--set` overrides with
int/float/bool coercion, schema validation via Config construction, and
materialized-YAML output.
"""

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# The conftest stub for src.config (extra="allow", no fields defined) makes
# the render/validate commands' schema walk vacuous -- every dotted key would
# "resolve" since the stub has no field list to check against. Swap in the
# genuine pydantic module for the duration of each test in this file so the
# commands see the real Config schema, then restore the stub for other test
# modules. Mirrors the bypass pattern already used in test_desca_cli.py /
# test_config_validation.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_real_config_module():
    saved = sys.modules.pop("src.config", None)
    real_mod = importlib.import_module("src.config")
    sys.modules["src.config"] = real_mod
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["src.config"] = saved
        else:
            sys.modules.pop("src.config", None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cambia_app():
    from src.cli import app

    return app


PRODUCTION_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "prtcfr_production.yaml"
)


# ---------------------------------------------------------------------------
# config render
# ---------------------------------------------------------------------------


def test_render_prtcfr_production_applies_override(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "prt_cfr.iterations=5",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()

    rendered = yaml.safe_load(out_path.read_text())
    assert "_base" not in rendered
    assert rendered["prt_cfr"]["iterations"] == 5

    # The materialized file is itself schema-valid.
    from src.config import Config

    Config.model_validate(rendered)


def test_render_multiple_sets_applied(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "prt_cfr.iterations=7",
            "--set",
            "prt_cfr.device=cpu",
            "--set",
            "prt_cfr.warm_start=false",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rendered = yaml.safe_load(out_path.read_text())
    assert rendered["prt_cfr"]["iterations"] == 7
    assert rendered["prt_cfr"]["device"] == "cpu"
    assert rendered["prt_cfr"]["warm_start"] is False


def test_render_resolves_base(runner, cambia_app, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("prt_cfr:\n  iterations: 10\n  device: cpu\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"_base: {base.name}\nprt_cfr:\n  seed: 42\n")

    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        ["config", "render", str(child), "-o", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    rendered = yaml.safe_load(out_path.read_text())
    assert "_base" not in rendered
    assert rendered["prt_cfr"]["iterations"] == 10
    assert rendered["prt_cfr"]["seed"] == 42


def test_render_bad_key_exits_nonzero(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "prt_cfr.not_a_real_field=5",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert not out_path.exists()


def test_render_bad_top_level_key_exits_nonzero(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "totally_bogus_section.foo=5",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert not out_path.exists()


def test_render_bad_value_exits_nonzero(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "prt_cfr.iterations=not_a_number",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert not out_path.exists()


def test_render_malformed_set_missing_equals(runner, cambia_app, tmp_path):
    out_path = tmp_path / "rendered.yaml"
    result = runner.invoke(
        cambia_app,
        [
            "config",
            "render",
            str(PRODUCTION_CONFIG),
            "--set",
            "prt_cfr.iterations",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# config validate
# ---------------------------------------------------------------------------


def test_validate_good_file_exits_zero(runner, cambia_app):
    result = runner.invoke(cambia_app, ["config", "validate", str(PRODUCTION_CONFIG)])
    assert result.exit_code == 0, result.output


def test_validate_broken_yaml_exits_nonzero(runner, cambia_app, tmp_path):
    broken = tmp_path / "broken.yaml"
    # Invalid YAML syntax: unbalanced flow mapping.
    broken.write_text("prt_cfr: [unclosed\n  iterations: 5\n")
    result = runner.invoke(cambia_app, ["config", "validate", str(broken)])
    assert result.exit_code != 0


def test_validate_schema_invalid_value_exits_nonzero(runner, cambia_app, tmp_path):
    bad = tmp_path / "bad_schema.yaml"
    bad.write_text("prt_cfr:\n  iterations: not_a_number\n")
    result = runner.invoke(cambia_app, ["config", "validate", str(bad)])
    assert result.exit_code != 0


def test_validate_missing_file_exits_nonzero(runner, cambia_app, tmp_path):
    result = runner.invoke(
        cambia_app, ["config", "validate", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Direct unit tests of the dotted-override helpers.
# ---------------------------------------------------------------------------


def test_coerce_override_value_bool_int_float_str():
    from src.cli import _coerce_override_value

    assert _coerce_override_value("true") is True
    assert _coerce_override_value("False") is False
    assert _coerce_override_value("5") == 5
    assert isinstance(_coerce_override_value("5"), int)
    assert _coerce_override_value("1.5") == 1.5
    assert _coerce_override_value("cpu") == "cpu"


def test_apply_dotted_override_sets_nested_value():
    from src.cli import _apply_dotted_override
    from src.config import Config

    merged = {"prt_cfr": {"iterations": 100}}
    _apply_dotted_override(merged, "prt_cfr.iterations", "5", Config)
    assert merged["prt_cfr"]["iterations"] == 5


def test_apply_dotted_override_creates_missing_nested_dict():
    from src.cli import _apply_dotted_override
    from src.config import Config

    merged = {}
    _apply_dotted_override(merged, "prt_cfr.seed", "7", Config)
    assert merged["prt_cfr"]["seed"] == 7


def test_apply_dotted_override_unknown_key_raises():
    from src.cli import _apply_dotted_override
    from src.config import Config

    with pytest.raises(ValueError):
        _apply_dotted_override({}, "prt_cfr.definitely_not_a_field", "1", Config)


def test_apply_dotted_override_cannot_descend_into_leaf():
    from src.cli import _apply_dotted_override
    from src.config import Config

    with pytest.raises(ValueError):
        _apply_dotted_override({}, "prt_cfr.iterations.sub", "1", Config)
