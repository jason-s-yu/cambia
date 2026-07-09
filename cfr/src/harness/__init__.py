"""
src/harness

Client-side serving-harness support (cambia-256). v1 ships the reconciler that
ingests a runner's synced run_db.sqlite into the client's authoritative
cfr/runs/cambia_runs.db. Transport (rsync/pull), CLI wiring, and the Go control
plane live outside this package.
"""

from src.harness.reconciler import (
    ReconcilerCollisionError,
    ReconcilerError,
    ReconcilerValidationError,
    replay,
)

__all__ = [
    "replay",
    "ReconcilerError",
    "ReconcilerValidationError",
    "ReconcilerCollisionError",
]
