"""tests/parity_seeds.py

The single declared seed constant for the Go-vs-Python cross-engine parity
gate (cambia-1234).

Go is the RULES.md reference implementation; the Python engine in
cfr/src/game/ is the reference-mirror being retired (cambia-1424). While both
exist, every cross-engine test draws its seed list from here, so the gate's
breadth is one number in one place instead of a per-file literal.

  PARITY_SEED_COUNT  how many seeds each cross-engine test sweeps.
  PARITY_SEEDS       the seeds themselves: range(PARITY_SEED_COUNT).

Override for a wider (or faster) sweep without editing any test:

  CAMBIA_PARITY_SEEDS=200 make parity-gate

The default of 40 is the cambia-225 lockstep bar. This module is deliberately
import-light (stdlib only) so any test can read it without pulling in the
engine, torch, or the FFI bridge.
"""

from __future__ import annotations

import os

# The cambia-225 bar. Changing this default changes the gate's breadth
# everywhere at once.
PARITY_SEED_COUNT_DEFAULT = 40

_ENV_VAR = "CAMBIA_PARITY_SEEDS"


def _read_seed_count() -> int:
    raw = os.environ.get(_ENV_VAR)
    if raw is None or raw.strip() == "":
        return PARITY_SEED_COUNT_DEFAULT
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is not an integer; it sets how many seeds the "
            f"cross-engine parity gate sweeps (default {PARITY_SEED_COUNT_DEFAULT})."
        ) from None
    if count < 1:
        raise ValueError(
            f"{_ENV_VAR}={count} must be >= 1; it sets how many seeds the "
            f"cross-engine parity gate sweeps (default {PARITY_SEED_COUNT_DEFAULT})."
        )
    return count


PARITY_SEED_COUNT: int = _read_seed_count()

PARITY_SEEDS: tuple = tuple(range(PARITY_SEED_COUNT))

# The lockCallerHand=False leg (cambia-1239 FT1). PARITY_SEEDS above always drives
# _TEST_RULES, which defaults lockCallerHand True, so the 40-seed sweep never exercises
# the branch cambia-1118 added: the Cambia caller's hand reachable once the house rule
# is off (RULES.md 3C, MATCHMAKING.md 5.2, off in every ranked queue). This leg is
# intentionally smaller than the main sweep; it exists to cover the lock-off branch at
# all, not to match the main gate's breadth.
_ENV_VAR_LOCK_OFF = "CAMBIA_PARITY_SEEDS_LOCK_OFF"

PARITY_SEED_COUNT_LOCK_OFF_DEFAULT = 8


def _read_seed_count_lock_off() -> int:
    raw = os.environ.get(_ENV_VAR_LOCK_OFF)
    if raw is None or raw.strip() == "":
        return PARITY_SEED_COUNT_LOCK_OFF_DEFAULT
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(
            f"{_ENV_VAR_LOCK_OFF}={raw!r} is not an integer; it sets how many seeds the "
            f"lockCallerHand=False parity leg sweeps (default "
            f"{PARITY_SEED_COUNT_LOCK_OFF_DEFAULT})."
        ) from None
    if count < 1:
        raise ValueError(
            f"{_ENV_VAR_LOCK_OFF}={count} must be >= 1; it sets how many seeds the "
            f"lockCallerHand=False parity leg sweeps (default "
            f"{PARITY_SEED_COUNT_LOCK_OFF_DEFAULT})."
        )
    return count


PARITY_SEED_COUNT_LOCK_OFF: int = _read_seed_count_lock_off()

PARITY_SEEDS_LOCK_OFF: tuple = tuple(range(PARITY_SEED_COUNT_LOCK_OFF))
