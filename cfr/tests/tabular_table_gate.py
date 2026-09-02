"""
tests/tabular_table_gate.py

Shared runner for the tabular table-equality gate (cambia-1782). Not a test
module: ``tests/test_tabular_table_gate.py`` is.

What the gate compares
----------------------
The port moved ``src.cfr.worker``'s outcome-sampling traversal from the Python
engine onto the Go engine. The claim it has to support is not "the new tables
look reasonable" but "the new tables are the old tables": the same infoset keys,
with the same float64 vectors, down to the last bit.

Making that comparison exact takes pinning both sources of randomness.

* The deal. Neither engine's seeded shuffle reproduces the other's, so the deal
  cannot come from a seed on either side. It is recorded as an explicit deck
  order plus a starting seat and both engines are dealt from it, which is what
  ``src.ffi.bridge.extract_deck_from_python_game`` and ``DealSpec(deck=...)``
  exist for.
* The action sampling. The traversal samples through the global numpy RNG, so
  numpy is re-seeded per iteration (``np_seed + t``) rather than once per run.
  With identical legal-action lists and identical strategies the two traversals
  then draw the same numbers at the same nodes; re-seeding per iteration means a
  divergence cannot silently shift every later iteration's stream.

Nothing else differs: the deal consumes the Python engine's game-local RNG and
the Go engine's deck argument, neither of which touches numpy.

The reshuffle caveat (memory cambia-1492) is why the gate config has to be one
that never exhausts the stockpile: past a reshuffle the two engines deal from
different shuffle RNGs and the comparison stops being about the traversal.

Regenerating the fixture
------------------------
The reference side is the Python-engine traversal as it stood at d2aff58 (the
commit this port started from), which is no longer in the tree. Check that
commit out somewhere, copy this file into its ``cfr/tests/``, and run:

    cd <d2aff58 checkout>/cfr
    PYTHONPATH=$PWD LIBCAMBIA_PATH=$PWD/libcambia.so \\
        python tests/tabular_table_gate.py --out tests/fixtures/<name>.json

The Python branch below drives that older ``run_cfr_simulation_worker``, which
has no ``deal`` argument, by replacing the ``CambiaGameState`` name it
constructs through. Once cambia-1430 removes the Python engine that branch is
dead and can go; the fixture and the Go side of the gate outlive it.
"""

import json
import logging
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import load_config

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "tabular_tables_d2aff58.json"
)

#: The gate's pinned run. tiny_cambia_tabular.yaml is the smallest shipped
#: tabular config (16-card deck, 2 cards a seat, 6 engine turns), so it never
#: reshuffles and the whole gate runs in a couple of seconds. The iteration
#: count clears the config's averaging_delay of 100, without which every
#: strategy and reach entry would be zero and two thirds of the comparison would
#: be checking nothing.
CONFIG = "config/tiny_cambia_tabular.yaml"
ITERATIONS = 150
DEAL_SEED = 90210
NP_SEED = 4242


def _uses_go_engine() -> bool:
    """True on the ported traversal, False on the d2aff58 Python one."""
    from src.cfr import worker

    return hasattr(worker, "GoBrState")


def _record_decks(rules, iterations: int, deal_seed: int) -> List[Tuple[List[int], int]]:
    """One deck order and starting seat per iteration, in deal order."""
    from src.ffi.bridge import extract_deck_from_python_game
    from src.game.engine import CambiaGameState

    decks = []
    for t in range(iterations):
        game = CambiaGameState(house_rules=rules, _rng=random.Random(deal_seed + t))
        deck, starting_player = extract_deck_from_python_game(game)
        decks.append(([int(c) for c in deck], int(starting_player)))
    return decks


def _run_iteration(worker, cfg, snapshot, deck, starting_player, deal_seed, t, log_dir):
    """One worker simulation on the pinned deal, on whichever engine is present."""
    args = (t, cfg, snapshot, None, None, 0, log_dir, "gate")
    if _uses_go_engine():
        from src.cfr.br_state import DealSpec

        return worker.run_cfr_simulation_worker(
            args, deal=DealSpec(deck=tuple(deck), starting_player=starting_player)
        )

    from src.ffi.bridge import extract_deck_from_python_game
    from src.game.engine import CambiaGameState

    def _pinned(house_rules):
        game = CambiaGameState(house_rules=house_rules, _rng=random.Random(deal_seed + t))
        recorded, _ = extract_deck_from_python_game(game)
        assert [int(c) for c in recorded] == deck, "the recorded deck drifted"
        return game

    original = worker.CambiaGameState
    worker.CambiaGameState = _pinned
    try:
        return worker.run_cfr_simulation_worker(args)
    finally:
        worker.CambiaGameState = original


def _merge(result, regret_sum, strategy_sum, reach_sum) -> None:
    """Accumulate one worker result, as CFRDataManagerMixin.merge_worker_results does."""
    touched = set()
    for table, updates in (
        (regret_sum, result.regret_updates),
        (strategy_sum, result.strategy_updates),
    ):
        for key, update in updates.items():
            if len(update) == 0:
                continue
            touched.add(key)
            current = table.get(key)
            if current is None or len(current) != len(update):
                table[key] = update.copy()
            else:
                table[key] = current + update
    for key, update in result.reach_prob_updates.items():
        touched.add(key)
        reach_sum[key] = reach_sum.get(key, 0.0) + float(update)
    for key in touched:  # RM+ floor
        if key in regret_sum and len(regret_sum[key]) > 0:
            regret_sum[key] = np.maximum(0.0, regret_sum[key])


def run_tables(
    decks: Optional[List[Tuple[List[int], int]]] = None,
    config_path: str = CONFIG,
    iterations: int = ITERATIONS,
    deal_seed: int = DEAL_SEED,
    np_seed: int = NP_SEED,
) -> Dict[str, Any]:
    """Run the pinned tabular gate and return its tables as comparable rows."""
    from src.cfr import worker

    cfg = load_config(config_path)
    if decks is None:
        decks = _record_decks(cfg.cambia_rules, iterations, deal_seed)
    assert len(decks) >= iterations, "the recorded deal list is shorter than the run"

    log_dir = tempfile.mkdtemp()
    regret_sum: Dict[Any, np.ndarray] = {}
    strategy_sum: Dict[Any, np.ndarray] = {}
    reach_sum: Dict[Any, float] = {}
    errors = 0

    disabled_at = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for t in range(iterations):
            deck, starting_player = decks[t]
            np.random.seed(np_seed + t)
            result = _run_iteration(
                worker,
                cfg,
                dict(regret_sum),
                deck,
                starting_player,
                deal_seed,
                t,
                log_dir,
            )
            assert result is not None, f"iteration {t} produced no result"
            errors += result.stats.error_count
            _merge(result, regret_sum, strategy_sum, reach_sum)
    finally:
        logging.disable(disabled_at)

    def rows(table, cast):
        return sorted([list(key.astuple()), cast(value)] for key, value in table.items())

    return {
        "meta": {
            "config": config_path,
            "iterations": iterations,
            "deal_seed": deal_seed,
            "np_seed": np_seed,
            "engine": "go" if _uses_go_engine() else "python",
            "error_count": errors,
        },
        "decks": [[deck, start] for deck, start in decks[:iterations]],
        "regret": rows(regret_sum, lambda v: [float(x) for x in v]),
        "strategy": rows(strategy_sum, lambda v: [float(x) for x in v]),
        "reach": rows(reach_sum, float),
    }


def as_json(tables: Dict[str, Any]) -> Dict[str, Any]:
    """The tables in the fixture's own representation.

    An infoset key holds nested tuples that JSON renders as lists, so a live run
    and a loaded fixture are only comparable once both have been through the same
    encoder. float64 round-trips exactly through ``repr``, which is what json
    writes, so this normalises the shape without loosening the comparison: it
    stays bit-for-bit.
    """
    return json.loads(json.dumps(tables, sort_keys=True))


def load_fixture(path: str = FIXTURE) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=FIXTURE)
    args = parser.parse_args()

    tables = run_tables()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tables, f, sort_keys=True, indent=0)
    print(
        f"{tables['meta']['engine']}: {len(tables['regret'])} regret keys, "
        f"{len(tables['strategy'])} strategy keys, {len(tables['reach'])} reach keys, "
        f"{tables['meta']['error_count']} worker errors -> {args.out}"
    )
