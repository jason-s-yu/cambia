"""
src/run_db.py

SQLite run database for the Cambia CFR pipeline.

Tracks training runs, checkpoints, eval results, and head-to-head comparisons.
Uses standard library sqlite3 only — no extra dependencies.

Usage:
    db = get_db()                          # default path: cfr/runs/cambia_runs.db
    db = get_db("/path/to/cambia_runs.db") # explicit path
    run_id = upsert_run(db, name="my-run", algorithm="os-mccfr", config_yaml="...")
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.evaluate_agents import MEAN_IMP_BASELINES  # canonical source

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "runs" / "cambia_runs.db"

ALGO_TO_AGENT_TYPE: Dict[str, str] = {
    "rebel": "rebel",
    "os-mccfr": "deep_cfr",
    "es-mccfr": "deep_cfr",
    "escher": "escher",
    "sd-cfr": "sd_cfr",
    "gtcfr": "gtcfr",
    "sog": "sog_inference",
    "psro": "deep_cfr",
}

ALGO_TO_CHECKPOINT_PREFIX: Dict[str, str] = {
    "rebel": "rebel_checkpoint",
    "os-mccfr": "deep_cfr_checkpoint",
    "es-mccfr": "deep_cfr_checkpoint",
    "escher": "deep_cfr_checkpoint",
    "sd-cfr": "deep_cfr_checkpoint",
    "gtcfr": "gtcfr_checkpoint",
    "sog": "sog_checkpoint",
    "psro": "deep_cfr_checkpoint",
}


def algo_to_agent_type(algorithm: str) -> str:
    """Map algorithm name to eval agent_type string. Falls back to 'deep_cfr'."""
    return ALGO_TO_AGENT_TYPE.get(algorithm, "deep_cfr")


def algo_to_checkpoint_prefix(algorithm: str) -> str:
    """Map algorithm name to checkpoint filename prefix. Falls back to 'deep_cfr_checkpoint'."""
    return ALGO_TO_CHECKPOINT_PREFIX.get(algorithm, "deep_cfr_checkpoint")

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    algorithm TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    config_hash TEXT,
    house_rules_hash TEXT,
    config_schema_version INTEGER DEFAULT 1,
    engine_commit_hash TEXT,
    best_metric_name TEXT,
    best_metric_value REAL,
    best_metric_iter INTEGER,
    tags TEXT DEFAULT '[]',
    notes TEXT,
    parent_run_id INTEGER REFERENCES runs(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    config_yaml TEXT,
    config_hash TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    created_at TEXT NOT NULL,
    is_best INTEGER NOT NULL DEFAULT 0,
    is_retained INTEGER NOT NULL DEFAULT 1,
    compressed INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, iteration)
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    checkpoint_id INTEGER REFERENCES checkpoints(id),
    iteration INTEGER NOT NULL,
    baseline TEXT NOT NULL,
    win_rate REAL,
    ci_low REAL,
    ci_high REAL,
    games_played INTEGER,
    p0_wins INTEGER,
    p1_wins INTEGER,
    ties INTEGER,
    avg_game_turns REAL,
    t1_cambia_rate REAL,
    avg_score_margin REAL,
    adv_loss REAL,
    strat_loss REAL,
    seat_balanced INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    UNIQUE(run_id, iteration, baseline)
);

CREATE TABLE IF NOT EXISTS head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    iter_a INTEGER NOT NULL,
    iter_b INTEGER NOT NULL,
    label TEXT,
    a_wins INTEGER,
    b_wins INTEGER,
    ties INTEGER,
    a_win_rate REAL,
    avg_game_turns REAL,
    timestamp TEXT,
    UNIQUE(run_id, iter_a, iter_b, label)
);

CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_iter ON checkpoints(run_id, iteration);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_iter ON eval_results(run_id, iteration);
CREATE INDEX IF NOT EXISTS idx_head_to_head_run ON head_to_head(run_id, iter_a);
"""


def get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open (or create) the SQLite run database.

    Configures WAL mode and synchronous=NORMAL for performance.
    Creates all tables and indexes if they don't exist.

    Args:
        db_path: Path to the SQLite file. Defaults to cfr/runs/cambia_runs.db.

    Returns:
        sqlite3.Connection with row_factory=sqlite3.Row.
    """
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(_DDL)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(yaml_text: str) -> str:
    """SHA256 of sorted/normalized YAML text."""
    normalized = "\n".join(sorted(yaml_text.strip().splitlines()))
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def house_rules_hash(config_dict: Dict[str, Any]) -> str:
    """SHA256 of just the cambia_rules block."""
    rules = config_dict.get("cambia_rules", {})
    canonical = json.dumps(rules, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def infer_algorithm(
    config_dict: Dict[str, Any],
    checkpoint_keys: Optional[set] = None,
    checkpoint_filename: Optional[str] = None,
) -> str:
    """
    Infer algorithm name from config dict, checkpoint keys, or filename.

    Priority:
    1. checkpoint_keys contains "rebel_value_net_state_dict" → "rebel"
    2. checkpoint_filename matches "rebel_checkpoint*" → "rebel"
    3. config has rebel section → "rebel"
    4. traversal_method == "escher" → "escher"
    5. use_sd_cfr → "sd-cfr"
    6. use_psro → "psro"
    7. sampling_method == "external" → "es-mccfr"
    8. default → "os-mccfr"
    """
    deep_cfr_section = config_dict.get("deep_cfr", {}) or {}

    # Detect SoG (must come before GT-CFR since SoG checkpoints also contain cvpn_state_dict)
    if checkpoint_keys and "sog_metadata" in checkpoint_keys:
        return "sog"
    if "sog_epochs" in deep_cfr_section or "sog_games_per_epoch" in deep_cfr_section:
        return "sog"

    # Detect GT-CFR
    if checkpoint_keys and "cvpn_state_dict" in checkpoint_keys:
        return "gtcfr"
    if "gtcfr_epochs" in deep_cfr_section or "gtcfr_games_per_epoch" in deep_cfr_section:
        return "gtcfr"

    # Detect ReBeL by checkpoint contents or filename
    if checkpoint_keys and "rebel_value_net_state_dict" in checkpoint_keys:
        return "rebel"
    if checkpoint_filename:
        import os
        basename = os.path.basename(checkpoint_filename)
        if basename.startswith("rebel_checkpoint"):
            return "rebel"
    if config_dict.get("rebel"):
        return "rebel"
    traversal = deep_cfr_section.get("traversal_method", "")
    if traversal == "escher":
        return "escher"
    if deep_cfr_section.get("use_sd_cfr"):
        return "sd-cfr"
    if deep_cfr_section.get("use_psro"):
        return "psro"
    sampling = deep_cfr_section.get("sampling_method", "")
    if sampling == "external":
        return "es-mccfr"
    return "os-mccfr"


def _get_engine_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def upsert_run(
    db: sqlite3.Connection,
    name: str,
    algorithm: Optional[str] = None,
    config_yaml: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    status: str = "created",
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    parent_run_id: Optional[int] = None,
) -> int:
    """
    Insert or update a run record.

    On conflict (same name), updates algorithm, status, config hashes, and updated_at.

    Returns:
        run_id (integer primary key).
    """
    now = _now()
    cfg_hash = config_hash(config_yaml) if config_yaml else None
    hr_hash = house_rules_hash(config_dict) if config_dict else None
    engine_commit = _get_engine_commit()
    tags_json = json.dumps(tags or [])

    cur = db.execute(
        """
        INSERT INTO runs (name, algorithm, status, config_hash, house_rules_hash,
                          engine_commit_hash, tags, notes, parent_run_id,
                          created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            algorithm=excluded.algorithm,
            status=excluded.status,
            config_hash=excluded.config_hash,
            house_rules_hash=excluded.house_rules_hash,
            engine_commit_hash=excluded.engine_commit_hash,
            updated_at=excluded.updated_at
        """,
        (name, algorithm, status, cfg_hash, hr_hash,
         engine_commit, tags_json, notes, parent_run_id, now, now),
    )
    db.commit()

    # Get run_id (works for both insert and update)
    row = db.execute("SELECT id FROM runs WHERE name=?", (name,)).fetchone()
    run_id = row["id"]

    # Store config snapshot if yaml provided
    if config_yaml:
        existing = db.execute(
            "SELECT id FROM config_snapshots WHERE run_id=? AND config_hash=?",
            (run_id, cfg_hash),
        ).fetchone()
        if not existing:
            db.execute(
                "INSERT INTO config_snapshots (run_id, config_yaml, config_hash, created_at) VALUES (?,?,?,?)",
                (run_id, config_yaml, cfg_hash, now),
            )
            db.commit()

    return run_id


def update_run_status(db: sqlite3.Connection, run_id: int, status: str) -> None:
    """Update run status and updated_at."""
    db.execute(
        "UPDATE runs SET status=?, updated_at=? WHERE id=?",
        (status, _now(), run_id),
    )
    db.commit()


def update_best_metric(
    db: sqlite3.Connection,
    run_id: int,
    name: str,
    value: float,
    iter_num: int,
) -> None:
    """Update best metric if value is better (higher) than existing."""
    row = db.execute(
        "SELECT best_metric_value FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if row is None:
        return
    current = row["best_metric_value"]
    if current is None or value > current:
        db.execute(
            "UPDATE runs SET best_metric_name=?, best_metric_value=?, best_metric_iter=?, updated_at=? WHERE id=?",
            (name, value, iter_num, _now(), run_id),
        )
        db.commit()


def register_checkpoint(
    db: sqlite3.Connection,
    run_id: int,
    iteration: int,
    file_path: str,
    file_size_bytes: Optional[int] = None,
) -> int:
    """
    Register a checkpoint (INSERT OR IGNORE on duplicate run_id+iteration).

    Returns:
        checkpoint_id (integer primary key).
    """
    now = _now()
    if file_size_bytes is None:
        try:
            file_size_bytes = os.path.getsize(file_path)
        except OSError:
            file_size_bytes = None
    db.execute(
        """
        INSERT INTO checkpoints (run_id, iteration, file_path, file_size_bytes, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_id, iteration) DO UPDATE SET file_path=excluded.file_path, file_size_bytes=excluded.file_size_bytes, created_at=excluded.created_at
        """,
        (run_id, iteration, str(file_path), file_size_bytes, now),
    )
    db.commit()
    row = db.execute(
        "SELECT id FROM checkpoints WHERE run_id=? AND iteration=?",
        (run_id, iteration),
    ).fetchone()
    return row["id"]


def mark_best_checkpoint(
    db: sqlite3.Connection,
    run_id: int,
    ckpt_id: int,
) -> None:
    """
    Mark a checkpoint as best. Clears is_best on all other checkpoints for the run.
    Also creates a best.pt symlink in the checkpoint directory.
    """
    db.execute("UPDATE checkpoints SET is_best=0 WHERE run_id=?", (run_id,))
    db.execute("UPDATE checkpoints SET is_best=1 WHERE id=?", (ckpt_id,))
    db.commit()

    # Create best.pt symlink
    row = db.execute("SELECT file_path FROM checkpoints WHERE id=?", (ckpt_id,)).fetchone()
    if row and row["file_path"]:
        ckpt_path = Path(row["file_path"])
        if ckpt_path.exists():
            best_link = ckpt_path.parent / "best.pt"
            try:
                if best_link.is_symlink() or best_link.exists():
                    best_link.unlink()
                best_link.symlink_to(ckpt_path.name)
            except OSError:
                pass


def insert_eval_result(
    db: sqlite3.Connection,
    run_id: int,
    ckpt_id: Optional[int],
    row_dict: Dict[str, Any],
) -> None:
    """
    Insert or replace an eval result row.

    row_dict should contain: iteration, baseline, win_rate, games_played,
    p0_wins, p1_wins, ties, adv_loss, strat_loss, avg_game_turns,
    t1_cambia_rate, avg_score_margin, timestamp.
    """
    db.execute(
        """
        INSERT OR REPLACE INTO eval_results
            (run_id, checkpoint_id, iteration, baseline, win_rate, ci_low, ci_high,
             games_played, p0_wins, p1_wins, ties, avg_game_turns,
             t1_cambia_rate, avg_score_margin, adv_loss, strat_loss,
             seat_balanced, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            ckpt_id,
            row_dict.get("iter", row_dict.get("iteration", 0)),
            row_dict.get("baseline"),
            row_dict.get("win_rate"),
            row_dict.get("ci_low"),
            row_dict.get("ci_high"),
            row_dict.get("games_played"),
            row_dict.get("p0_wins"),
            row_dict.get("p1_wins"),
            row_dict.get("ties"),
            row_dict.get("avg_game_turns"),
            row_dict.get("t1_cambia_rate"),
            row_dict.get("avg_score_margin"),
            row_dict.get("adv_loss"),
            row_dict.get("strat_loss"),
            row_dict.get("seat_balanced", 0),
            row_dict.get("timestamp", _now()),
        ),
    )
    db.commit()


def insert_head_to_head(
    db: sqlite3.Connection,
    run_id: int,
    row_dict: Dict[str, Any],
) -> None:
    """Insert or replace a head-to-head result."""
    db.execute(
        """
        INSERT OR REPLACE INTO head_to_head
            (run_id, iter_a, iter_b, label, a_wins, b_wins, ties,
             a_win_rate, avg_game_turns, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row_dict.get("iter_a"),
            row_dict.get("iter_b"),
            row_dict.get("label"),
            row_dict.get("a_wins"),
            row_dict.get("b_wins"),
            row_dict.get("ties"),
            row_dict.get("a_win_rate"),
            row_dict.get("avg_game_turns"),
            row_dict.get("timestamp"),
        ),
    )
    db.commit()


def compute_retention_flags(
    db: sqlite3.Connection,
    run_id: int,
    keep_every_n: int = 50,
    keep_latest: int = 3,
) -> list:
    """
    Recompute is_retained flags for all checkpoints of a run.

    Retains:
    - The best checkpoint (is_best=1)
    - Every Nth checkpoint (iteration % keep_every_n == 0)
    - The keep_latest most recent checkpoints

    Returns list of non-retained checkpoint IDs.
    """
    rows = db.execute(
        "SELECT id, iteration, is_best FROM checkpoints WHERE run_id=? ORDER BY iteration",
        (run_id,),
    ).fetchall()
    if not rows:
        return []

    retained_ids = set()

    # Keep every N
    for row in rows:
        if keep_every_n > 0 and row["iteration"] % keep_every_n == 0:
            retained_ids.add(row["id"])
        if row["is_best"]:
            retained_ids.add(row["id"])

    # Keep latest N
    for row in rows[-keep_latest:]:
        retained_ids.add(row["id"])

    non_retained = []
    for row in rows:
        keep = row["id"] in retained_ids
        db.execute(
            "UPDATE checkpoints SET is_retained=? WHERE id=?",
            (1 if keep else 0, row["id"]),
        )
        if not keep:
            non_retained.append(row["id"])
    db.commit()
    return non_retained


def _wilson_ci(p_hat: float, n: int, z: float = 1.96):
    """Compute Wilson score interval. Returns (lower, center, upper)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - margin), center, min(1.0, center + margin))


def write_run_meta_json(
    db: sqlite3.Connection,
    run_id: int,
    run_dir: str,
) -> None:
    """Export run_meta.json to run_dir."""
    run_row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not run_row:
        return

    ckpt_count = db.execute(
        "SELECT COUNT(*) as cnt FROM checkpoints WHERE run_id=?", (run_id,)
    ).fetchone()["cnt"]

    best_ckpt = db.execute(
        "SELECT file_path, iteration FROM checkpoints WHERE run_id=? AND is_best=1",
        (run_id,),
    ).fetchone()

    meta = {
        "run_id": run_id,
        "name": run_row["name"],
        "algorithm": run_row["algorithm"],
        "status": run_row["status"],
        "config_hash": run_row["config_hash"],
        "house_rules_hash": run_row["house_rules_hash"],
        "engine_commit_hash": run_row["engine_commit_hash"],
        "best_metric_name": run_row["best_metric_name"],
        "best_metric_value": run_row["best_metric_value"],
        "best_metric_iter": run_row["best_metric_iter"],
        "best_checkpoint_iter": best_ckpt["iteration"] if best_ckpt else None,
        "best_checkpoint_path": best_ckpt["file_path"] if best_ckpt else None,
        "total_checkpoints": ckpt_count,
        "tags": json.loads(run_row["tags"] or "[]"),
        "notes": run_row["notes"],
        "created_at": run_row["created_at"],
        "updated_at": run_row["updated_at"],
    }

    out_path = Path(run_dir) / "run_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_eval_summary_jsonl(
    db: sqlite3.Connection,
    run_id: int,
    run_dir: str,
) -> None:
    """
    Export eval_summary.jsonl to run_dir.

    Each line is one iteration, with per-baseline win_rate + Wilson CI,
    mean_imp (average over MEAN_IMP_BASELINES), and best_checkpoint flag.
    """
    rows = db.execute(
        """
        SELECT iteration, baseline, win_rate, games_played,
               adv_loss, strat_loss, avg_game_turns, t1_cambia_rate, timestamp
        FROM eval_results
        WHERE run_id=?
        ORDER BY iteration, baseline
        """,
        (run_id,),
    ).fetchall()

    if not rows:
        return

    best_row = db.execute(
        "SELECT iteration FROM checkpoints WHERE run_id=? AND is_best=1", (run_id,)
    ).fetchone()
    best_iter = best_row["iteration"] if best_row else None

    # Group by iteration
    by_iter: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        it = row["iteration"]
        if it not in by_iter:
            by_iter[it] = {"baselines": {}}
        by_iter[it]["baselines"][row["baseline"]] = dict(row)

    out_path = Path(run_dir) / "eval_summary.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in sorted(by_iter.keys()):
            entry = by_iter[it]
            baselines = entry["baselines"]

            per_baseline = {}
            for bl, r in baselines.items():
                n = r["games_played"] or 0
                p = r["win_rate"] or 0.0
                lo, center, hi = _wilson_ci(p, n)
                per_baseline[bl] = {
                    "win_rate": round(p, 6),
                    "ci_lo": round(lo, 6),
                    "ci_hi": round(hi, 6),
                    "games_played": n,
                }

            # mean_imp = mean WR over MEAN_IMP_BASELINES that have data
            imp_wrs = [
                baselines[bl]["win_rate"]
                for bl in MEAN_IMP_BASELINES
                if bl in baselines and baselines[bl]["win_rate"] is not None
            ]
            mean_imp = round(sum(imp_wrs) / len(imp_wrs), 6) if imp_wrs else None

            # Pull metadata from any available baseline row
            first = next(iter(baselines.values()))
            line = {
                "iter": it,
                "mean_imp": mean_imp,
                "is_best_checkpoint": it == best_iter,
                "adv_loss": first.get("adv_loss"),
                "strat_loss": first.get("strat_loss"),
                "avg_game_turns": first.get("avg_game_turns"),
                "t1_cambia_rate": first.get("t1_cambia_rate"),
                "timestamp": first.get("timestamp"),
                "baselines": per_baseline,
            }
            f.write(json.dumps(line) + "\n")
