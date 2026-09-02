"""
tests/test_tabular_estimator.py

Estimator tests for the tabular outcome-sampling traversal in
``src.cfr.worker``.

Two defects the 2026-08-19 soundness review confirmed are pinned here.

cambia-718: the CFR+ delayed linear averaging weight multiplied the regret
update as well as the strategy sum, so with the shipped ``averaging_delay`` of
100 every iteration up to the delay returned all-zero regret updates and the
RM+ floor kept the zeros. The weight belongs to the average-strategy
accumulation only (Tammelin et al. 2015); regret updates are unweighted.

"""

import numpy as np
import pytest

from src.config import load_config

from tests import tabular_table_gate as gate


def _go_available() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.ffi.bridge import GoEngine

            e = GoEngine(seed=0)
            e.close()
        return True
    except Exception:
        return False


skip_if_no_go = pytest.mark.skipif(
    not _go_available(), reason="libcambia.so not available"
)


# --- cambia-718: the averaging weight ---------------------------------------


def test_the_averaging_weight_is_zero_until_the_delay_elapses():
    """max(0, t - averaging_delay), with t the 1-based iteration number."""
    from src.cfr.worker import averaging_weight

    params = load_config(gate.CONFIG).cfr_plus_params
    assert params.averaging_delay == 100, "this test is written against the delay of 100"

    # iteration is 0-based, so iteration i is t = i + 1.
    assert averaging_weight(0, params) == 0.0
    assert averaging_weight(99, params) == 0.0
    assert averaging_weight(100, params) == 1.0
    assert averaging_weight(101, params) == 2.0


def test_the_averaging_weight_is_one_when_weighting_is_disabled():
    from src.cfr.worker import averaging_weight

    params = load_config(gate.CONFIG).cfr_plus_params.model_copy(
        update={"weighted_averaging_enabled": False}
    )
    assert averaging_weight(0, params) == 1.0
    assert averaging_weight(500, params) == 1.0


@skip_if_no_go
def test_the_first_iteration_moves_regrets_but_not_the_strategy_sum():
    """t = 1 under a delay of 100: regrets move, the average strategy does not.

    This is the cambia-718 defect stated as behaviour. Before the fix the
    iteration weight multiplied the regret update too, so the whole result was
    zeros and the first 100 iterations were wasted.
    """
    result = _run_iteration(iteration=0)

    assert _any_non_zero(result.regret_updates), (
        "iteration 1 produced no regret movement, so the averaging delay is "
        "still zeroing the regret update (cambia-718)"
    )
    assert not _any_non_zero(result.strategy_updates), (
        "iteration 1 contributed to the average strategy, but the delayed "
        "averaging weight is zero until the delay elapses"
    )
    assert all(v == 0.0 for v in result.reach_prob_updates.values())


@skip_if_no_go
def test_an_iteration_past_the_delay_moves_the_strategy_sum():
    """t = 101 under a delay of 100 is the first iteration that averages."""
    result = _run_iteration(iteration=100)

    assert _any_non_zero(result.regret_updates)
    assert _any_non_zero(
        result.strategy_updates
    ), "iteration 101 did not contribute to the average strategy"


# --- helpers -----------------------------------------------------------------


def _run_iteration(iteration: int):
    """One real worker traversal on a pinned Go deal, logging suppressed."""
    import logging
    import tempfile

    from src.cfr.br_state import DealSpec
    from src.cfr.worker import run_cfr_simulation_worker

    cfg = load_config(gate.CONFIG)
    np.random.seed(gate.NP_SEED + iteration)

    disabled_at = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        result = run_cfr_simulation_worker(
            (iteration, cfg, {}, None, None, 0, tempfile.mkdtemp(), "estimator"),
            deal=DealSpec(seed=90210 + iteration),
        )
    finally:
        logging.disable(disabled_at)

    assert result is not None, "the traversal returned no result"
    assert result.stats.error_count == 0, (
        f"the traversal logged {result.stats.error_count} errors, so its "
        "updates are not a clean sample"
    )
    return result


def _any_non_zero(updates) -> bool:
    return any(np.any(np.asarray(v) != 0.0) for v in updates.values())
