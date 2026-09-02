"""
tests/test_tabular_table_gate.py

Table-equality gate for the tabular traversal (cambia-1782, re-based by
cambia-718 and cambia-719).

This runs the traversal over the pinned deals and seeds recorded in
``tests/fixtures/tabular_tables_cambia_718_719.json`` and asserts the result is
bit-for-bit the stored table: the same infoset keys, with the same float64
regret and strategy vectors.

The fixture started as the Python-engine traversal's output, so that the port
onto the Go engine had to reproduce it exactly. The estimator fixes changed the
tables on purpose, so the fixture is now the corrected traversal's own output
and the gate is a regression pin over sampling, reach threading and the
averaging weight. ``tests/tabular_table_gate.py`` carries the runner, why each
source of randomness is pinned the way it is, and how to regenerate the
fixture.

A gate against a stored table can rot into a tautology if the stored table is
empty or all zeros, so the coverage assertions below fail if the fixture stops
carrying non-zero entries in all three tables.
"""

import pytest

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


@pytest.fixture(scope="module")
def reference():
    return gate.load_fixture()


@pytest.fixture(scope="module")
def produced(reference):
    tables = gate.run_tables(
        decks=[(deck, start) for deck, start in reference["decks"]],
        config_path=reference["meta"]["config"],
        iterations=reference["meta"]["iterations"],
        deal_seed=reference["meta"]["deal_seed"],
        np_seed=reference["meta"]["np_seed"],
    )
    return gate.as_json(tables)


def test_the_fixture_is_a_clean_reference_run(reference):
    """The stored tables came from a traversal that logged no errors.

    Through cambia-1782 this asserted ``engine == "python"``, because the
    fixture was then the pre-port traversal's output and the gate's job was to
    show the port reproduced it. cambia-718 and cambia-719 corrected the
    estimator on purpose, so no Python-engine table is reachable any more and
    the fixture is the corrected traversal's own output.
    """
    assert reference["meta"]["engine"] == "go"
    assert reference["meta"]["error_count"] == 0


def test_the_fixture_is_not_a_tautology(reference):
    """Every table the gate compares carries non-zero entries."""
    non_zero = {
        field: sum(
            1
            for _, value in reference[field]
            if (any(value) if isinstance(value, list) else value)
        )
        for field in ("regret", "strategy", "reach")
    }
    assert non_zero["regret"] > 0, "the reference regret table is all zeros"
    assert non_zero["strategy"] > 0, "the reference strategy table is all zeros"
    assert non_zero["reach"] > 0, "the reference reach table is all zeros"


@skip_if_no_go
def test_the_traversal_runs_clean_on_the_pinned_deals(produced):
    assert produced["meta"]["engine"] == "go"
    assert produced["meta"]["error_count"] == 0


@skip_if_no_go
def test_infoset_keys_match(reference, produced):
    for field in ("regret", "strategy", "reach"):
        expected = [key for key, _ in reference[field]]
        actual = [key for key, _ in produced[field]]
        assert actual == expected, f"{field}: the infoset key set differs"


@skip_if_no_go
@pytest.mark.parametrize("field", ["regret", "strategy", "reach"])
def test_table_vectors_are_bit_identical(reference, produced, field):
    for (key, expected), (_, actual) in zip(reference[field], produced[field]):
        assert actual == expected, f"{field}: vector differs at infoset {key}"
