"""Scoped tests for tools/tiny_solver.py's logging-suppression scoping.

tiny_solver.py used to mute every "src.*" logger to CRITICAL as a
module-level (import-time) side effect, to keep the engine's chatty per-node
warnings out of full-tree expansion output. Because prtcfr_trainer.py and
prtcfr_eval.py both `from tools.tiny_solver import build_tree`, merely
importing either of those modules (directly or transitively) would silence
src.* logging for the rest of the process/pytest session -- the root cause of
a session-wide test flake for anything asserting on src.* log output.

These tests verify the fix: muting is scoped to the tree-expansion call path
(via _quiet_src_loggers / the `quiet` parameter), and importing tiny_solver
has zero logging side effects.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

_CFR_ROOT = Path(__file__).resolve().parent.parent


class TestImportHasNoLoggingSideEffects:
    def test_import_does_not_mutate_logger_levels(self):
        """Run in a fresh subprocess (not importlib.reload in-process): reloading
        a module already imported by sibling test files would swap out class
        identities (Builder, Terminal, ...) out from under any isinstance check
        elsewhere in the same pytest session. A subprocess is fully isolated."""
        script = (
            "import logging, sys\n"
            "for name, level in [('src.game', logging.INFO), "
            "('src.agent_state', logging.WARNING), "
            "('src.game.engine', logging.DEBUG), "
            "('src.some_new_module', logging.WARNING)]:\n"
            "    logging.getLogger(name).setLevel(level)\n"
            "before = {n: logging.getLogger(n).level for n in "
            "['src.game', 'src.agent_state', 'src.game.engine', 'src.some_new_module']}\n"
            "import tools.tiny_solver\n"
            "after = {n: logging.getLogger(n).level for n in "
            "['src.game', 'src.agent_state', 'src.game.engine', 'src.some_new_module']}\n"
            "assert before == after, (before, after)\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(_CFR_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "OK" in result.stdout


class TestQuietSrcLoggersContextManager:
    def test_mutes_during_and_restores_after(self):
        from tools.tiny_solver import _quiet_src_loggers

        logger = logging.getLogger("src.game")
        original = logger.level
        logger.setLevel(logging.WARNING)
        try:
            assert logger.level == logging.WARNING
            with _quiet_src_loggers():
                assert logger.level == logging.CRITICAL
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original)

    def test_restores_on_exception(self):
        from tools.tiny_solver import _quiet_src_loggers

        logger = logging.getLogger("src.agent_state")
        original = logger.level
        logger.setLevel(logging.INFO)
        try:
            with pytest.raises(ValueError):
                with _quiet_src_loggers():
                    assert logger.level == logging.CRITICAL
                    raise ValueError("boom")
            assert logger.level == logging.INFO
        finally:
            logger.setLevel(original)

    def test_nested_entry_only_restores_at_outermost_exit(self):
        from tools.tiny_solver import _quiet_src_loggers

        logger = logging.getLogger("src.game.engine")
        original = logger.level
        logger.setLevel(logging.WARNING)
        try:
            with _quiet_src_loggers():
                with _quiet_src_loggers():
                    assert logger.level == logging.CRITICAL
                # inner exit must not restore while the outer context is active
                assert logger.level == logging.CRITICAL
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original)

    def test_newly_registered_src_logger_is_covered(self):
        """A src.* logger created after tiny_solver import (not present in the
        explicit list) must still be muted/restored -- matches prior behavior
        of scanning loggerDict at mute time, not a fixed name list."""
        from tools.tiny_solver import _quiet_src_loggers

        logger = logging.getLogger("src.freshly_created_for_this_test")
        logger.setLevel(logging.INFO)
        try:
            with _quiet_src_loggers():
                assert logger.level == logging.CRITICAL
            assert logger.level == logging.INFO
        finally:
            logging.root.manager.loggerDict.pop("src.freshly_created_for_this_test", None)


class TestBuildTreeQuietParameter:
    """Exercise the `quiet` parameter through the real build_tree entry point
    on the smallest available config, so this is fast and still hits the
    actual expansion call path Builder.build_decision_or_terminal uses."""

    CONFIG_1CARD = "config/tiny_norecall.yaml"

    def test_build_tree_default_restores_logger_levels(self):
        from src.config import load_config
        from tools.tiny_solver import build_tree

        cfg = load_config(self.CONFIG_1CARD)
        logger = logging.getLogger("src.game.engine")
        original = logger.level
        logger.setLevel(logging.WARNING)
        try:
            build_tree(cfg, n_deals=1, seed0=0, max_nodes_per_deal=2_000_000)
            assert logger.level == logging.WARNING, (
                "build_tree(quiet=True default) must restore logger levels after returning"
            )
        finally:
            logger.setLevel(original)

    def test_build_tree_quiet_false_does_not_mute(self):
        from src.config import load_config
        from tools.tiny_solver import build_tree

        cfg = load_config(self.CONFIG_1CARD)
        logger = logging.getLogger("src.game.engine")
        original = logger.level
        logger.setLevel(logging.WARNING)
        try:
            build_tree(cfg, n_deals=1, seed0=0, max_nodes_per_deal=2_000_000, quiet=False)
            # quiet=False never touches the level; it should be untouched throughout
            # and after (nothing to restore since nothing was muted).
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original)
