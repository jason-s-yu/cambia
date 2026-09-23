"""Tests for `cambia train ppo`'s CLI wiring (cambia-1428/cambia-1482):
--num-players validation and the default fixed-baseline opponent.

Mocks src.ppo_train.train_ppo so these stay fast and do not need
sb3-contrib or a real training run: the CLI layer's job is argument parsing,
validation, and threading values through, which is what is under test here.
Full end-to-end smoke runs against the default baseline live in
tests/test_ppo_env_goengine.py.
"""

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    """Help output without ANSI escape sequences: typer's rich help forces a
    terminal under GITHUB_ACTIONS, so the CI runner's output carries SGR codes
    that split an option name (cambia-2400; the CI typer ships no click)."""
    return _ANSI.sub("", text)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cambia_app():
    from src.cli import app

    return app


@pytest.fixture
def ppo_train():
    """patch("src.ppo_train.train_ppo") imports src.ppo_train, which imports
    gymnasium and sb3-contrib (the rl extra) at module level, so the command
    tests skip where those are absent, as tests/test_ppo_env_goengine.py and
    tests/test_ppo_diagnostic.py do; the CI runner installs no rl extra
    (cambia-2400). The help tests stay: the help path imports neither."""
    pytest.importorskip("gymnasium", reason="gymnasium required for the PPO env")
    pytest.importorskip("sb3_contrib", reason="sb3-contrib required for PPO tests")


def test_train_ppo_help_names_the_default_opponent(runner, cambia_app):
    result = runner.invoke(cambia_app, ["train", "ppo", "--help"])
    assert result.exit_code == 0
    assert "imperfect_greedy" in _plain(result.stdout)


def test_train_ppo_help_names_num_players(runner, cambia_app):
    result = runner.invoke(cambia_app, ["train", "ppo", "--help"])
    assert result.exit_code == 0
    assert "--num-players" in _plain(result.stdout)


def test_train_ppo_defaults_opponent_to_imperfect_greedy(
    runner, cambia_app, tmp_path, ppo_train
):
    """No --opponent, no --self-play: the call reaching train_ppo names the
    GameView-ported imperfect_greedy baseline, not a self-play sentinel or an
    unresolved default."""
    with patch("src.ppo_train.train_ppo") as mock_train:
        result = runner.invoke(
            cambia_app,
            [
                "train",
                "ppo",
                "--config",
                "config/deep_train.yaml",
                "--save-path",
                str(tmp_path / "runs" / "x" / "checkpoints" / "m"),
                "--timesteps",
                "1",
            ],
        )
    assert result.exit_code == 0, result.stdout
    mock_train.assert_called_once()
    assert mock_train.call_args.kwargs["opponent"] == "imperfect_greedy"


@pytest.mark.parametrize("num_players", [2, 4, 8])
def test_train_ppo_accepts_num_players_in_range(
    runner, cambia_app, tmp_path, num_players, ppo_train
):
    with patch("src.ppo_train.train_ppo") as mock_train:
        result = runner.invoke(
            cambia_app,
            [
                "train",
                "ppo",
                "--config",
                "config/deep_train.yaml",
                "--save-path",
                str(tmp_path / "runs" / f"x{num_players}" / "checkpoints" / "m"),
                "--timesteps",
                "1",
                "--num-players",
                str(num_players),
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert mock_train.call_args.kwargs["num_players"] == num_players


@pytest.mark.parametrize("num_players", [0, 1, 9, 100])
def test_train_ppo_rejects_num_players_out_of_range(
    runner, cambia_app, tmp_path, num_players, ppo_train
):
    """Validated against engine.MaxPlayers (8) before train_ppo is ever
    called, so a bad value fails fast with a clear message rather than deep
    inside a SubprocVecEnv worker."""
    with patch("src.ppo_train.train_ppo") as mock_train:
        result = runner.invoke(
            cambia_app,
            [
                "train",
                "ppo",
                "--config",
                "config/deep_train.yaml",
                "--save-path",
                str(tmp_path / "runs" / "x" / "checkpoints" / "m"),
                "--timesteps",
                "1",
                "--num-players",
                str(num_players),
            ],
        )
    assert result.exit_code != 0
    mock_train.assert_not_called()
    assert "MaxPlayers" in result.output or "num-players" in result.output


def test_train_ppo_self_play_overrides_opponent(runner, cambia_app, tmp_path, ppo_train):
    """--self-play always wins regardless of --opponent (cambia-1482 must not
    regress the pre-existing self-play path while wiring the new default)."""
    from src.ppo_env import SELF_PLAY_OPPONENT

    with patch("src.ppo_train.train_ppo") as mock_train:
        result = runner.invoke(
            cambia_app,
            [
                "train",
                "ppo",
                "--config",
                "config/deep_train.yaml",
                "--save-path",
                str(tmp_path / "runs" / "x" / "checkpoints" / "m"),
                "--timesteps",
                "1",
                "--self-play",
                "--opponent",
                "imperfect_greedy",
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert mock_train.call_args.kwargs["opponent"] == SELF_PLAY_OPPONENT
