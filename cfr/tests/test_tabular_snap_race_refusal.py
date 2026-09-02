"""
tests/test_tabular_snap_race_refusal.py

A tabular config that turns snapRace on is refused when it loads (cambia-1782).

``src.cfr.br_state.GoBrState`` cannot drive snapRace: the race path resolves
every committed snap at once through the engine's ``_resolve_snap_race``, and
the snap-result reconstruction that rebuilds the observation stream does not
model it. Since cambia-1782 put the tabular traversal on that substrate, such a
config no longer trains. It failed in the least useful way available: the worker
caught the refusal per iteration, logged it to a run file and returned an empty
result, so the run went to completion having learned nothing.

The refusal is keyed on the file's own ``algorithm: tabular`` line, which is why
it can live in ``load_config`` without touching the deep, PRT-CFR, GT-CFR and
evaluation lanes, all of which run snapRace legitimately through other code.
"""

import importlib
import sys
import textwrap

import pytest


def _real_config_module():
    """The real src.config, past the stub tests/conftest.py installs.

    The stub's load_config delegates to the real one inside a bare
    ``except Exception: return None``, so a refusal raised at load time comes
    back as None rather than an error. Every config test in this tree reaches
    around it the same way (see tests/test_config_validation.py).
    """
    saved = sys.modules.pop("src.config", None)
    try:
        return importlib.import_module("src.config")
    finally:
        if saved is not None:
            sys.modules["src.config"] = saved


_config = _real_config_module()
TabularSnapRaceError = _config.TabularSnapRaceError
load_config = _config.load_config

_TABULAR_SNAP_RACE = """
algorithm: tabular

cambia_rules:
  snapRace: true
  use_jokers: 0
  cards_per_player: 2
  max_game_turns: 6

cfr_training:
  num_iterations: 10
"""


def _write(tmp_path, text: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(text))
    return str(path)


def test_a_tabular_config_with_snap_race_is_refused(tmp_path):
    with pytest.raises(TabularSnapRaceError) as caught:
        load_config(_write(tmp_path, _TABULAR_SNAP_RACE))
    message = str(caught.value)
    assert "snapRace" in message
    assert "cambia-1782" in message, "the refusal has to name the change that caused it"


def test_a_tabular_config_without_snap_race_still_loads(tmp_path):
    cfg = load_config(_write(tmp_path, _TABULAR_SNAP_RACE.replace("true", "false")))
    assert cfg.cambia_rules.snapRace is False
    assert cfg.cambia_rules.max_game_turns == 6


def test_the_refusal_does_not_reach_other_algorithms(tmp_path):
    """Only the tabular lane runs on the substrate that cannot take snapRace."""
    deep = _TABULAR_SNAP_RACE.replace("algorithm: tabular", "algorithm: deep")
    cfg = load_config(_write(tmp_path, deep))
    assert cfg.cambia_rules.snapRace is True


def test_a_config_that_names_no_algorithm_still_loads(tmp_path):
    """Most configs omit the key, and none of them should start failing."""
    anonymous = _TABULAR_SNAP_RACE.replace("algorithm: tabular\n", "")
    cfg = load_config(_write(tmp_path, anonymous))
    assert cfg.cambia_rules.snapRace is True


def test_the_refusal_is_not_swallowed_into_a_default_config(tmp_path):
    """load_config falls back to defaults on a bad file; this must not do that.

    A silent fallback would drop every other setting in the file and train on
    something the user never asked for, which is the failure mode the engine
    backend refusal was written to avoid.
    """
    with pytest.raises(TabularSnapRaceError):
        load_config(_write(tmp_path, _TABULAR_SNAP_RACE))


def test_the_shipped_tabular_configs_load(tmp_path):
    """The refusal is calibrated so nothing shipped trips it."""
    for name in (
        "config/tiny_cambia_tabular.yaml",
        "config/tiny_2card_plateau.yaml",
        "config/tiny_norecall.yaml",
    ):
        cfg = load_config(name)
        assert cfg.cambia_rules.snapRace is False, name
