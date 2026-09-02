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

cambia-719: the outcome-sampling estimator subtracted ``sigma[a] * u`` per
action instead of the node-value estimate, sampled on-policy with no
exploration (so an action driven to zero probability could never be sampled
again), and threaded reach probabilities that were seeded to ones and passed
down unchanged. The estimator is now the canonical one of Lanctot et al. 2009.
"""

import numpy as np
import pytest

from src.config import load_config
from src.utils import get_rm_plus_strategy

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


# --- cambia-719: the outcome-sampling estimator ------------------------------


def test_sampled_regrets_subtract_the_node_value_estimate():
    """The review's example, with the canonical baseline.

    sigma = [.8, .1, .1], sampled a* = 0, u = +1. Sampling on-policy makes
    q = .8, so the sampled action value is u/q = 1.25 and every unsampled
    action's estimate is 0. The node value is therefore sigma[a*] * 1.25 = 1.0,
    and the regrets are that baseline subtracted from each action's estimate.

    The pre-fix code subtracted ``sigma[a] * 1.25`` instead, which gave
    [0.25, -0.125, -0.125]: the unsampled actions were understated eightfold
    and the RM+ floor rectified the drift asymmetrically.
    """
    from src.cfr.worker import sampled_regrets

    strategy = np.array([0.8, 0.1, 0.1])
    action_value = 1.0 / 0.8  # u / q

    regrets = sampled_regrets(strategy, chosen_index=0, action_value=action_value)

    assert regrets == pytest.approx([0.25, -1.0, -1.0])


def test_the_estimator_is_unbiased_over_the_sampling_policy():
    """The soundness claim the whole estimator rests on.

    Averaged over which action the behaviour policy draws, the sampled regrets
    are exactly the true counterfactual regrets ``u(a) - sum_b sigma(b) u(b)``,
    and stay so at every exploration rate, because the 1/q correction cancels
    the sampling probability. Enumerating the three outcomes and weighting by q
    computes that expectation exactly, so this is an equality rather than a
    Monte Carlo estimate.

    The pre-fix baseline gives [1.25, -1.85, 0.6] here against a true
    [0.5, -2.5, 2.5]: biased, and biased regardless of epsilon, since the fault
    is the baseline rather than the sampling.
    """
    from src.cfr.worker import sampled_regrets, sampling_policy

    strategy = np.array([0.5, 0.3, 0.2])
    utilities = np.array([1.0, -2.0, 3.0])
    true_regrets = utilities - float(np.dot(strategy, utilities))
    assert true_regrets == pytest.approx([0.5, -2.5, 2.5])

    for epsilon in (0.0, 0.2, 0.6, 1.0):
        q = sampling_policy(strategy, epsilon)
        expectation = np.zeros(len(strategy))
        for sampled in range(len(strategy)):
            expectation += q[sampled] * sampled_regrets(
                strategy, sampled, float(utilities[sampled] / q[sampled])
            )
        assert expectation == pytest.approx(
            true_regrets
        ), f"the estimator is biased at epsilon={epsilon}"


def test_the_estimator_stays_unbiased_two_levels_up():
    """The suffix correction, on a tree deep enough for it to matter.

    One level of recursion cannot show whether the suffix of a trajectory is
    corrected for, because there is no suffix. This walks a two-level tree:
    the root infoset I offers {a, b}, action a leads to a second decision node J
    offering {c, d}, and every other edge ends the game. The expectation is
    computed by enumerating both continuations and weighting by the behaviour
    policy, so it is exact.

    Leaving the sampling probability out of what J hands back up scores
    v(I, a) at 0.38 against a true 1.1: the trajectory prefix gets corrected and
    its suffix does not.
    """
    from src.cfr.worker import suffix_reach_ratio

    strategy_j = {"c": 0.7, "d": 0.3}
    sample_j = {"c": 0.4, "d": 0.6}
    sample_i_a = 0.5
    utility = {"c": 2.0, "d": -1.0}

    true_value = sum(strategy_j[k] * utility[k] for k in "cd")
    assert true_value == pytest.approx(1.1)

    expectation = 0.0
    for edge in "cd":
        trajectory_prob = sample_i_a * sample_j[edge]
        # What node J returns to node I, per the recursion's own rule.
        tail = suffix_reach_ratio(
            child_ratio=1.0,  # a terminal returns 1
            own_action_prob=strategy_j[edge],
            sampling_prob=sample_j[edge],
        )
        # What node I then makes of it: opp_reach is 1 with no opponent, and
        # the sampling reach through a is sample_i_a.
        action_value = 1.0 * tail * utility[edge] / sample_i_a
        expectation += trajectory_prob * action_value

    assert expectation == pytest.approx(true_value), (
        "the estimate of v(I, a) is biased, so the trajectory suffix is not "
        "being corrected for"
    )


def test_sampled_regrets_sum_to_zero_against_the_strategy():
    """sum_a sigma(a) r(a) = 0: the baseline is the strategy's own value."""
    from src.cfr.worker import sampled_regrets

    strategy = np.array([0.5, 0.3, 0.2])
    for chosen in range(3):
        regrets = sampled_regrets(strategy, chosen_index=chosen, action_value=2.5)
        assert float(np.dot(strategy, regrets)) == pytest.approx(0.0, abs=1e-12)


def test_the_sampling_policy_mixes_in_exploration():
    """An epsilon-mixed policy reaches actions the strategy has abandoned."""
    from src.cfr.worker import sampling_policy

    strategy = np.array([1.0, 0.0, 0.0])

    on_policy = sampling_policy(strategy, epsilon=0.0)
    assert on_policy == pytest.approx([1.0, 0.0, 0.0])

    mixed = sampling_policy(strategy, epsilon=0.6)
    assert mixed == pytest.approx([0.6, 0.2, 0.2])
    assert float(np.sum(mixed)) == pytest.approx(1.0)
    assert (mixed > 0).all(), "every legal action must stay reachable"


def test_an_action_driven_to_zero_probability_is_revived():
    """No permanent freeze: exploration reaches a zeroed action and RM+ keeps it.

    Action 1 carries no positive regret, so RM+ gives it probability zero and
    on-policy sampling can never select it again. With exploration the sampler
    reaches it; a positive outcome there is credited to action 1 alone, because
    the node-value baseline is sigma[a*] * u and sigma[1] is zero. The RM+
    floor keeps that positive regret and the action returns to the support.
    """
    from src.cfr.worker import sampled_regrets, sampling_policy

    regret_sum = np.array([4.0, 0.0, 0.0])
    strategy = get_rm_plus_strategy(regret_sum)
    assert strategy == pytest.approx([1.0, 0.0, 0.0]), "action 1 starts frozen out"

    sample_policy = sampling_policy(strategy, epsilon=0.6)
    assert sample_policy[1] > 0.0, "the frozen action is unreachable by the sampler"

    # Sample the frozen action and observe a positive utility for it.
    action_value = 1.0 / sample_policy[1]
    regrets = sampled_regrets(strategy, chosen_index=1, action_value=action_value)
    assert regrets[1] > 0.0, "the sampled frozen action earned no positive regret"

    revived = get_rm_plus_strategy(np.maximum(0.0, regret_sum + regrets))
    assert revived[1] > 0.0, "the action stayed frozen after a favourable sample"


@skip_if_no_go
def test_counterfactual_reaches_are_threaded_rather_than_held_at_one():
    """The recursion carries real reach and sampling probabilities downward.

    Before the fix every reach factor was the constant 1: the root seeded an
    array of ones and each level passed its own array down unchanged, so the
    1/q tail correction was inert and no counterfactual weighting was applied.
    """
    from src.cfr import worker

    seen = []
    real = worker._traverse_game_for_worker
    wanted = {"my_reach", "opp_reach", "sample_reach"}

    def recording(*args, **kwargs):
        if wanted <= set(kwargs):
            seen.append((kwargs["my_reach"], kwargs["opp_reach"], kwargs["sample_reach"]))
        return real(*args, **kwargs)

    worker._traverse_game_for_worker = recording
    try:
        _run_iteration(iteration=100)
    finally:
        worker._traverse_game_for_worker = real

    assert len(seen) > 1, (
        "the traversal recursed without passing the reaches by keyword, so "
        "this probe read nothing"
    )
    assert any(
        s < 1.0 for _, _, s in seen
    ), "every sampling reach was 1.0, so the 1/q tail correction is inert"
    assert any(
        o < 1.0 for _, o, _ in seen
    ), "every opponent reach was 1.0, so no counterfactual weighting is applied"


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
