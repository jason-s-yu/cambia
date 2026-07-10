"""
src/harness

Client-side serving-harness support (cambia-256). The reconciler ingests a
runner's synced run_db.sqlite into the client's authoritative
cfr/runs/cambia_runs.db;
the `harness` CLI (submit/status/logs/cancel/resume/pull/push-run/watch) drives
the runner's control plane and the ssh/rsync data plane. The Go runnerd lives
outside this package.

Only the reconciler is imported eagerly; the CLI/transport/pull/config modules
carry the optional `harness` extra deps (websockets/pyjwt/cryptography) and are
imported on demand so the base package stays lightweight.
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
