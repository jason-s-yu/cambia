"""
tests/test_eval_seat_crn_hygiene.py

Scoped tests for the Phase 0 eval-hygiene repair (task S1W3):

1. Seat alternation in run_evaluation (agent under test plays both seats; win
   attribution stays agent-relative regardless of physical seat).
2. Termination scale: the engine's own max_game_turns cap terminates games, so
   the action-count safety valve (MaxTurnTies) rarely fires against strategic
   baselines.
3. Common-random-numbers (CRN) seat pairing: a seeded deck makes paired games
   reproducible and identical across repeated runs.
4. Engine seed param: same seed -> identical deal; default None -> random.
5. New eval_results columns (selection_mode, crn_seed, seat_scheme) persist and
   migrate additively.
6. Database hygiene hooks (best-metric recompute, retention flags, stale-running).

These use the duck-typed config pattern from test_evaluate.py and patch
load_config so no YAML file is required.
"""

import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Shared config helper (no jokers, short games for fast termination)
# ---------------------------------------------------------------------------


def _make_config(max_game_turns: int = 60):
    """Minimal duck-typed config suitable for run_evaluation."""
    config = type("Config", (), {})()

    rules = type("CambiaRulesConfig", (), {})()
    rules.allowDrawFromDiscardPile = False
    rules.allowReplaceAbilities = False
    rules.snapRace = False
    rules.penaltyDrawCount = 2
    rules.use_jokers = 0
    rules.cards_per_player = 4
    rules.initial_view_count = 2
    rules.cambia_allowed_round = 0
    rules.allowOpponentSnapping = False
    rules.max_game_turns = max_game_turns
    rules.lockCallerHand = True
    rules.num_decks = 1
    config.cambia_rules = rules

    agent_params = type("AgentParamsConfig", (), {})()
    agent_params.memory_level = 1
    agent_params.time_decay_turns = 10
    config.agent_params = agent_params

    agents_cfg = type("AgentsConfig", (), {})()
    greedy_cfg = type("GreedyAgentConfig", (), {})()
    greedy_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_agent = greedy_cfg
    config.agents = agents_cfg

    return config


def _run(agent1, agent2, num_games, config, **kwargs):
    from src.evaluate_agents import run_evaluation

    with patch("src.evaluate_agents.load_config", return_value=config):
        return run_evaluation(
            config_path="dummy.yaml",
            agent1_type=agent1,
            agent2_type=agent2,
            num_games=num_games,
            strategy_path=None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 1. Engine seed param
# ---------------------------------------------------------------------------


class TestEngineSeed:
    def test_same_seed_identical_deal(self):
        from src.game.engine import CambiaGameState
        from src.config import CambiaRulesConfig

        r = CambiaRulesConfig()
        g1 = CambiaGameState(house_rules=r, seed=4242)
        g2 = CambiaGameState(house_rules=r, seed=4242)
        h1 = [[c.rank for c in p.hand] for p in g1.players]
        h2 = [[c.rank for c in p.hand] for p in g2.players]
        assert h1 == h2
        assert g1.current_player_index == g2.current_player_index
        d1 = g1.discard_pile[-1].rank if g1.discard_pile else None
        d2 = g2.discard_pile[-1].rank if g2.discard_pile else None
        assert d1 == d2
        assert g1.seed == 4242

    def test_different_seed_differs(self):
        from src.game.engine import CambiaGameState
        from src.config import CambiaRulesConfig

        r = CambiaRulesConfig()
        s1 = [c.rank for c in CambiaGameState(house_rules=r, seed=1).stockpile]
        s2 = [c.rank for c in CambiaGameState(house_rules=r, seed=2).stockpile]
        assert s1 != s2

    def test_default_none_is_random(self):
        from src.game.engine import CambiaGameState
        from src.config import CambiaRulesConfig

        r = CambiaRulesConfig()
        a = [c.rank for c in CambiaGameState(house_rules=r).stockpile]
        b = [c.rank for c in CambiaGameState(house_rules=r).stockpile]
        # Two unseeded 53-card shuffles colliding is astronomically unlikely.
        assert a != b


# ---------------------------------------------------------------------------
# 2. Seat alternation + agent-relative win attribution
# ---------------------------------------------------------------------------


class TestSeatAlternation:
    def test_outcome_count_invariant_preserved(self):
        """Alternation must not change the one-outcome-per-game invariant."""
        config = _make_config()
        results = _run("random", "random", 10, config, seat_scheme="alternated")
        total = (
            results.get("P0 Wins", 0)
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
            + results.get("Errors", 0)
        )
        assert total == 10, dict(results)

    def test_seat_balanced_true_when_alternated_and_decisive(self):
        """A strong vs weak matchup yields decisive games on both seats."""
        config = _make_config()
        results = _run("greedy", "random_no_cambia", 40, config, seat_scheme="alternated")
        stats = results.stats
        assert stats["seat_scheme"] == "alternated"
        # greedy (perfect info) wins decisively from either seat, so the agent
        # under test occupies both seats across decisive games.
        assert stats["seat_balanced"] is True, dict(results)

    def test_seat_balanced_false_when_fixed(self):
        config = _make_config()
        results = _run("greedy", "random_no_cambia", 40, config, seat_scheme="fixed")
        stats = results.stats
        assert stats["seat_scheme"] == "fixed"
        assert stats["seat_balanced"] is False

    def test_attribution_follows_agent_not_seat(self):
        """Agent-relative P0 Wins reflects the agent under test (greedy), which
        is strong, even though it sits at seat 1 in half the games."""
        config = _make_config()
        results = _run("greedy", "random_no_cambia", 60, config, seat_scheme="alternated")
        p0 = results.get("P0 Wins", 0)  # agent under test (greedy) wins
        p1 = results.get("P1 Wins", 0)  # opponent (random) wins
        scored = p0 + p1 + results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        assert scored > 0
        # greedy dominates regardless of seat; if attribution were seat-locked
        # to physical seat 0 we would see ~50% of greedy's wins vanish.
        assert p0 > p1, dict(results)

    def test_invalid_seat_scheme_falls_back_to_alternated(self):
        config = _make_config()
        results = _run("random", "random", 4, config, seat_scheme="banana")
        assert results.stats["seat_scheme"] == "alternated"

    def test_alternation_no_stale_memory_regression(self):
        """Guard against the id-reuse stale-memory bug that alternation exposed.

        Baseline agents detect a new game by game_state object id. When an agent
        instance is reused across non-consecutive games (alternation), a recycled
        address can collide with a stale _last_game_id, so the agent keeps the
        prior game's memory and calls Cambia immediately -- games collapse to ~4
        turns and the strong agent's win rate regresses toward 50%. The harness
        invalidates the id sentinel each game to prevent this. This test fails if
        that reset is removed: avg_turns would crater and the edge would vanish.

        Uses memory_heuristic vs random_no_cambia (no early-Cambia opponent) so
        healthy games are long; stale memory would short-circuit them.
        """
        config = _make_config(max_game_turns=60)
        results = _run("memory_heuristic", "random_no_cambia", 120, config,
                       seat_scheme="alternated", crn_seed_base=4242)
        avg_turns = results.stats.get("avg_game_turns", 0.0)
        assert avg_turns > 15, (
            f"avg_game_turns={avg_turns:.1f} too low under alternation -- stale "
            f"cross-game memory (id-reuse) is short-circuiting games."
        )


# ---------------------------------------------------------------------------
# 3. Termination scale / MaxTurnTies rarity
# ---------------------------------------------------------------------------


class TestTerminationScale:
    def test_engine_cap_terminates_not_safety_valve(self):
        """With the engine's max_game_turns cap active, games end terminal and
        are scored as P0/P1/Ties; the action-count safety valve (MaxTurnTies)
        should essentially never fire."""
        config = _make_config(max_game_turns=40)
        results = _run("memory_heuristic", "random_no_cambia", 80, config,
                       seat_scheme="alternated")
        max_turn_ties = results.get("MaxTurnTies", 0)
        errors = results.get("Errors", 0)
        scored = (
            results.get("P0 Wins", 0)
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + max_turn_ties
        )
        assert scored > 0
        assert errors == 0, dict(results)
        # The engine's own cap makes every game terminal; the safety valve is a
        # runaway guard only. Target: < 1% MaxTurnTies (was 5-8% on the hybrid).
        assert max_turn_ties / scored < 0.01, (
            f"MaxTurnTies rate {max_turn_ties}/{scored} too high; the local "
            f"action-count cap is firing before the engine terminates."
        )


# ---------------------------------------------------------------------------
# 4. CRN seat pairing reproducibility
# ---------------------------------------------------------------------------


class TestCRNSeatPairing:
    def test_crn_run_is_reproducible(self):
        """Same crn_seed_base + scheme + deterministic agents -> bit-identical
        results across runs. greedy and memory_heuristic do not sample, so the
        seeded deck is the only source of randomness."""
        config = _make_config()

        def snapshot(r):
            return (
                r.get("P0 Wins", 0), r.get("P1 Wins", 0),
                r.get("Ties", 0), r.get("MaxTurnTies", 0),
                round(r.stats.get("avg_game_turns", 0.0), 4),
            )

        r1 = _run("greedy", "memory_heuristic", 30, config,
                  seat_scheme="alternated", crn_seed_base=777)
        r2 = _run("greedy", "memory_heuristic", 30, config,
                  seat_scheme="alternated", crn_seed_base=777)
        assert r1.stats["crn_seed"] == 777
        assert snapshot(r1) == snapshot(r2), (
            f"CRN runs not reproducible: {snapshot(r1)} != {snapshot(r2)}"
        )

    def test_crn_seed_recorded_in_stats(self):
        config = _make_config()
        results = _run("random", "random", 4, config, crn_seed_base=12321)
        assert results.stats["crn_seed"] == 12321

    def test_no_crn_seed_leaves_none(self):
        config = _make_config()
        results = _run("random", "random", 4, config)  # crn_seed_base default None
        assert results.stats["crn_seed"] is None

    def test_crn_pairing_identical_deal_within_pair(self):
        """Under alternation, the two games of a seat-swap pair share one deck.

        Drive the engine directly with the same seeds the harness derives for a
        pair (pair_index identical) and confirm the deals match.
        """
        import hashlib
        from src.game.engine import CambiaGameState
        from src.config import CambiaRulesConfig

        base, ident, opp = 555, "ckptX", "random"
        # Games 1 and 2 -> pair_index 0 (alternated): same seed_key.
        seeds = []
        for game_num in (1, 2):
            pair_index = (game_num - 1) // 2
            key = f"{base}|{ident}|{opp}|{pair_index}"
            seeds.append(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        assert seeds[0] == seeds[1], "pair should share one deck seed"

        r = CambiaRulesConfig()
        g_a = CambiaGameState(house_rules=r, seed=seeds[0])
        g_b = CambiaGameState(house_rules=r, seed=seeds[1])
        deal_a = [[c.rank for c in p.hand] for p in g_a.players]
        deal_b = [[c.rank for c in p.hand] for p in g_b.players]
        assert deal_a == deal_b


# ---------------------------------------------------------------------------
# 4b. Multi-baseline stats survive the parallel process boundary
# ---------------------------------------------------------------------------


class TestMultiBaselineStatsPreserved:
    """A Counter's dynamic `.stats` attribute is dropped by pickling, so parallel
    eval used to lose enhanced + hygiene stats. The worker now returns stats
    separately and the parent reattaches them. These run with baseline agents so
    the spawned workers load a default config (no YAML needed)."""

    def test_parallel_preserves_stats(self):
        from src.evaluate_agents import run_evaluation_multi_baseline

        rm = run_evaluation_multi_baseline(
            config_path="runs/__nonexistent__.yaml",  # workers fall back to default config
            checkpoint_path="/tmp/ignored_by_baseline.pt",
            num_games=20,
            baselines=["random_no_cambia", "random_late_cambia"],
            device="cpu",
            agent_type="memory_heuristic",
            max_workers=2,  # force parallel spawn
            seat_scheme="alternated",
        )
        assert set(rm) == {"random_no_cambia", "random_late_cambia"}
        for bl, r in rm.items():
            stats = getattr(r, "stats", None)
            assert stats, f"{bl}: stats lost across process boundary"
            assert stats.get("seat_scheme") == "alternated"
            assert "avg_game_turns" in stats
            assert stats.get("crn_seed") is not None  # auto-derived from checkpoint

    def test_sequential_preserves_stats(self):
        from src.evaluate_agents import run_evaluation_multi_baseline

        rm = run_evaluation_multi_baseline(
            config_path="runs/__nonexistent__.yaml",
            checkpoint_path="/tmp/ignored_by_baseline.pt",
            num_games=10,
            baselines=["random_no_cambia"],
            device="cpu",
            agent_type="memory_heuristic",
            max_workers=1,  # sequential
            seat_scheme="fixed",
        )
        stats = getattr(rm["random_no_cambia"], "stats", None)
        assert stats and stats.get("seat_scheme") == "fixed"


# ---------------------------------------------------------------------------
# 5. New eval_results columns: DDL, migration, persistence
# ---------------------------------------------------------------------------


class TestEvalResultsColumns:
    def test_fresh_db_has_new_columns(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            cols = {r[1] for r in db.execute("PRAGMA table_info(eval_results)").fetchall()}
            assert {"selection_mode", "crn_seed", "seat_scheme"} <= cols
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_migration_adds_columns_to_old_db(self):
        """An old eval_results table (no hygiene columns) gains them on open,
        preserving existing rows with NULL new columns."""
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        old = sqlite3.connect(tmp)
        old.executescript(
            """
            CREATE TABLE eval_results (
              id INTEGER PRIMARY KEY, run_id INTEGER, checkpoint_id INTEGER,
              iteration INTEGER, baseline TEXT, win_rate REAL, ci_low REAL,
              ci_high REAL, games_played INTEGER, p0_wins INTEGER, p1_wins INTEGER,
              ties INTEGER, avg_game_turns REAL, t1_cambia_rate REAL,
              avg_score_margin REAL, adv_loss REAL, strat_loss REAL,
              seat_balanced INTEGER DEFAULT 0, timestamp TEXT,
              UNIQUE(run_id, iteration, baseline)
            );
            INSERT INTO eval_results (run_id, iteration, baseline, win_rate, games_played, timestamp)
            VALUES (1, 10, 'random_no_cambia', 0.42, 5000, '2020-01-01T00:00:00Z');
            """
        )
        old.commit()
        old.close()

        db = run_db.get_db(tmp)
        try:
            cols = {r[1] for r in db.execute("PRAGMA table_info(eval_results)").fetchall()}
            assert {"selection_mode", "crn_seed", "seat_scheme"} <= cols
            row = db.execute(
                "SELECT win_rate, selection_mode, crn_seed, seat_scheme "
                "FROM eval_results WHERE baseline='random_no_cambia'"
            ).fetchone()
            assert abs(row["win_rate"] - 0.42) < 1e-9
            assert row["selection_mode"] is None
            assert row["crn_seed"] is None
            assert row["seat_scheme"] is None
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_migration_idempotent(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        try:
            run_db.get_db(tmp).close()
            run_db.get_db(tmp).close()  # second open must not raise
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_insert_writes_hygiene_columns(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = run_db.upsert_run(db, name="cols-run", algorithm="desca")
            run_db.insert_eval_result(
                db, rid, None,
                {
                    "iter": 5, "baseline": "random_no_cambia", "win_rate": 0.5,
                    "games_played": 100, "timestamp": "2026-06-13T00:00:00Z",
                    "selection_mode": "stochastic", "crn_seed": 999,
                    "seat_scheme": "alternated", "seat_balanced": 1,
                },
            )
            row = db.execute(
                "SELECT selection_mode, crn_seed, seat_scheme, seat_balanced "
                "FROM eval_results WHERE run_id=?", (rid,)
            ).fetchone()
            assert row["selection_mode"] == "stochastic"
            assert row["crn_seed"] == "999"  # stored as TEXT
            assert row["seat_scheme"] == "alternated"
            assert row["seat_balanced"] == 1
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_persist_eval_results_records_hygiene_fields(self, tmp_path):
        """persist_eval_results carries stats-derived hygiene fields into the
        metrics.jsonl row."""
        import json
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "hyg-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 55, "P1 Wins": 40, "Ties": 5})
        results.stats = {
            "seat_scheme": "alternated",
            "seat_balanced": True,
            "selection_mode": "argmax",
            "crn_seed": 4242,
        }
        persist_eval_results(str(run_dir), 100, {"random_no_cambia": results})

        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert row["seat_scheme"] == "alternated"
        assert row["seat_balanced"] == 1
        assert row["selection_mode"] == "argmax"
        assert row["crn_seed"] == "4242"

    def test_persist_param_fallback_when_no_stats(self, tmp_path):
        """When the Counter has no stats, function-level overrides are used."""
        import json
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "fallback-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}
        persist_eval_results(
            str(run_dir), 100, {"random": results},
            selection_mode="stochastic", seat_scheme="fixed", crn_seed=7,
        )
        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert row["seat_scheme"] == "fixed"
        assert row["selection_mode"] == "stochastic"
        assert row["crn_seed"] == "7"
        # No alternation evidence -> seat_balanced stays 0.
        assert row["seat_balanced"] == 0


# ---------------------------------------------------------------------------
# 6. Database hygiene hooks
# ---------------------------------------------------------------------------


class TestDBHygiene:
    def _seed_run(self, db, run_db, name="hyg"):
        rid = run_db.upsert_run(db, name=name, algorithm="desca", status="running")
        for it, wr in [(10, 0.40), (20, 0.45)]:
            for bl in run_db.MEAN_IMP_BASELINES:
                run_db.insert_eval_result(
                    db, rid, None,
                    {"iter": it, "baseline": bl, "win_rate": wr,
                     "games_played": 5000, "timestamp": "2026-06-13T00:00:00Z"},
                )
        return rid

    def test_recompute_best_metric_picks_max(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = self._seed_run(db, run_db)
            best = run_db.recompute_best_metric(db, rid)
            assert abs(best - 0.45) < 1e-6
            row = db.execute(
                "SELECT best_metric_name, best_metric_value, best_metric_iter "
                "FROM runs WHERE id=?", (rid,)
            ).fetchone()
            assert row["best_metric_name"] == "mean_imp"
            assert row["best_metric_iter"] == 20
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_recompute_best_metric_no_data_returns_none(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = run_db.upsert_run(db, name="empty", algorithm="desca")
            assert run_db.recompute_best_metric(db, rid) is None
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_mark_stale_running_runs(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = run_db.upsert_run(db, name="stale", algorithm="desca", status="running")
            # Recent update -> not marked.
            assert run_db.mark_stale_running_runs(db, max_age_hours=24.0) == 0
            # Force old timestamp -> marked interrupted.
            db.execute(
                "UPDATE runs SET updated_at='2020-01-01T00:00:00Z' WHERE id=?", (rid,)
            )
            db.commit()
            assert run_db.mark_stale_running_runs(db, max_age_hours=24.0) == 1
            status = db.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()["status"]
            assert status == "interrupted"
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_checkpoint_retention_flags(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = run_db.upsert_run(db, name="ret", algorithm="desca")
            for it in (10, 20, 30):
                run_db.register_checkpoint(db, rid, it, f"/tmp/ck_{it}.pt")
            best_id = db.execute(
                "SELECT id FROM checkpoints WHERE iteration=10 AND run_id=?", (rid,)
            ).fetchone()["id"]
            run_db.mark_best_checkpoint(db, rid, best_id)
            dropped = run_db.apply_checkpoint_retention(db, rid, keep_last_n=1, keep_best=True)
            retained = {
                r["iteration"]
                for r in db.execute(
                    "SELECT iteration FROM checkpoints WHERE run_id=? AND is_retained=1", (rid,)
                ).fetchall()
            }
            assert dropped == 1
            assert retained == {10, 30}  # best=10, last=30
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_retention_keep_all_when_zero(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            rid = run_db.upsert_run(db, name="retall", algorithm="desca")
            for it in (1, 2, 3):
                run_db.register_checkpoint(db, rid, it, f"/tmp/k_{it}.pt")
            dropped = run_db.apply_checkpoint_retention(db, rid, keep_last_n=0)
            assert dropped == 0
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)

    def test_cleanup_database_aggregate(self):
        import src.run_db as run_db

        tmp = tempfile.mktemp(suffix=".db")
        db = run_db.get_db(tmp)
        try:
            self._seed_run(db, run_db, name="agg")
            summary = run_db.cleanup_database(db, stale_running_hours=24.0)
            assert summary["best_metrics_recomputed"] >= 1
            assert "stale_runs_marked" in summary
        finally:
            db.close()
            Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. PPO algorithm detection (eval baseline, not a CFR variant)
# ---------------------------------------------------------------------------


class TestPPOAlgorithmDetection:
    def test_declared_ppo_detected(self):
        import src.run_db as run_db

        assert run_db.infer_algorithm({"algorithm": "ppo"}) == "ppo"
        assert run_db.infer_algorithm({"algorithm": "PPO"}) == "ppo"

    def test_ppo_mappings(self):
        import src.run_db as run_db

        assert run_db.algo_to_agent_type("ppo") == "ppo"
        assert run_db.algo_to_checkpoint_prefix("ppo") == "ppo_checkpoint"

    def test_existing_detection_unchanged(self):
        import src.run_db as run_db

        assert run_db.infer_algorithm({}) == "os-mccfr"
        assert run_db.infer_algorithm({"algorithm": "desca"}) == "desca"
        assert run_db.algo_to_agent_type("unknown-algo") == "deep_cfr"
