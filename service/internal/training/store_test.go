package training

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	_ "modernc.org/sqlite"
)

// testDDL creates the same schema as the Python run_db.py.
const testDDL = `
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
`

// setupTestDB creates a temporary SQLite DB with fixture data and returns the TrainingStore.
func setupTestDB(t *testing.T) (*TrainingStore, string) {
	t.Helper()
	tmpDir := t.TempDir()

	dbPath := filepath.Join(tmpDir, "cambia_runs.db")
	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(testDDL); err != nil {
		t.Fatal(err)
	}

	// Insert fixture runs.
	_, err = db.Exec(`INSERT INTO runs (name, algorithm, status, best_metric_value, best_metric_iter, created_at, updated_at)
		VALUES ('test-run-1', 'os-mccfr', 'completed', 0.42, 500, '2026-03-01T00:00:00Z', '2026-03-02T00:00:00Z')`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`INSERT INTO runs (name, algorithm, status, best_metric_value, best_metric_iter, notes, tags, created_at, updated_at)
		VALUES ('test-run-2', 'sd-cfr', 'running', 0.35, 300, 'some notes', '["tag1"]', '2026-03-01T00:00:00Z', '2026-03-03T00:00:00Z')`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert config snapshot for run 1.
	_, err = db.Exec(`INSERT INTO config_snapshots (run_id, config_yaml, config_hash, created_at)
		VALUES (1, 'learning_rate: 0.001', 'abc123', '2026-03-01T00:00:00Z')`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert checkpoints for run 1.
	_, err = db.Exec(`INSERT INTO checkpoints (run_id, iteration, file_path, file_size_bytes, created_at, is_best)
		VALUES (1, 100, '/path/to/ckpt100.pt', 1024, '2026-03-01T01:00:00Z', 0)`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`INSERT INTO checkpoints (run_id, iteration, file_path, file_size_bytes, created_at, is_best)
		VALUES (1, 500, '/path/to/ckpt500.pt', 2048, '2026-03-01T05:00:00Z', 1)`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert eval results for run 1 at iteration 100 — all 5 baselines.
	for _, b := range meanImpBaselines {
		_, err = db.Exec(`INSERT INTO eval_results (run_id, iteration, baseline, win_rate, ci_low, ci_high, games_played, adv_loss, strat_loss, timestamp)
			VALUES (1, 100, ?, 0.40, 0.38, 0.42, 5000, 0.5, 0.3, '2026-03-01T01:00:00Z')`, b)
		if err != nil {
			t.Fatal(err)
		}
	}
	// Add a non-baseline eval.
	_, err = db.Exec(`INSERT INTO eval_results (run_id, iteration, baseline, win_rate, ci_low, ci_high, games_played, adv_loss, strat_loss, timestamp)
		VALUES (1, 100, 'random', 0.90, 0.88, 0.92, 5000, 0.5, 0.3, '2026-03-01T01:00:00Z')`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert iteration 200 with all 5 baselines at 0.50.
	for _, b := range meanImpBaselines {
		_, err = db.Exec(`INSERT INTO eval_results (run_id, iteration, baseline, win_rate, ci_low, ci_high, games_played, adv_loss, strat_loss, timestamp)
			VALUES (1, 200, ?, 0.50, 0.48, 0.52, 5000, 0.4, 0.2, '2026-03-01T02:00:00Z')`, b)
		if err != nil {
			t.Fatal(err)
		}
	}

	db.Close()

	store, err := NewTrainingStore(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return store, tmpDir
}

func TestListRuns(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	runs, err := store.ListRuns(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 2 {
		t.Fatalf("expected 2 runs, got %d", len(runs))
	}
	// Ordered by updated_at DESC.
	if runs[0].Name != "test-run-2" {
		t.Errorf("expected first run to be test-run-2, got %s", runs[0].Name)
	}
	if runs[1].Name != "test-run-1" {
		t.Errorf("expected second run to be test-run-1, got %s", runs[1].Name)
	}
	// test-run-2 has status "running" but no PID file → "stopped".
	if runs[0].Status != "stopped" {
		t.Errorf("expected stopped (no PID), got %s", runs[0].Status)
	}
}

func TestGetRun(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	detail, err := store.GetRun(ctx, "test-run-1")
	if err != nil {
		t.Fatal(err)
	}
	if detail == nil {
		t.Fatal("expected non-nil detail")
	}
	if detail.Name != "test-run-1" {
		t.Errorf("expected test-run-1, got %s", detail.Name)
	}
	if detail.ConfigYAML != "learning_rate: 0.001" {
		t.Errorf("expected config yaml, got %q", detail.ConfigYAML)
	}
	if detail.Algorithm != "os-mccfr" {
		t.Errorf("expected os-mccfr, got %s", detail.Algorithm)
	}
}

func TestGetRunNotFound(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	detail, err := store.GetRun(ctx, "nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	if detail != nil {
		t.Fatalf("expected nil, got %+v", detail)
	}
}

func TestGetMetrics(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	// All metrics for run 1.
	metrics, err := store.GetMetrics(ctx, "test-run-1", "")
	if err != nil {
		t.Fatal(err)
	}
	// 5 baselines + 1 random at iter 100, plus 5 baselines at iter 200 = 11 total.
	if len(metrics) != 11 {
		t.Fatalf("expected 11 metrics, got %d", len(metrics))
	}
}

func TestGetMetricsFilteredByBaseline(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	metrics, err := store.GetMetrics(ctx, "test-run-1", "random")
	if err != nil {
		t.Fatal(err)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected 1 metric for baseline=random, got %d", len(metrics))
	}
	if metrics[0].Baseline != "random" {
		t.Errorf("expected random, got %s", metrics[0].Baseline)
	}
}

func TestGetMeanImp(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	points, err := store.GetMeanImp(ctx, "test-run-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(points) != 2 {
		t.Fatalf("expected 2 mean_imp points, got %d", len(points))
	}
	if points[0].Iteration != 100 {
		t.Errorf("expected iteration 100, got %d", points[0].Iteration)
	}
	// All baselines at iter 100 have win_rate 0.40.
	if points[0].MeanImp < 0.399 || points[0].MeanImp > 0.401 {
		t.Errorf("expected ~0.40 mean_imp, got %f", points[0].MeanImp)
	}
	if points[1].Iteration != 200 {
		t.Errorf("expected iteration 200, got %d", points[1].Iteration)
	}
	if points[1].MeanImp < 0.499 || points[1].MeanImp > 0.501 {
		t.Errorf("expected ~0.50 mean_imp, got %f", points[1].MeanImp)
	}
}

func TestGetCheckpoints(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	cps, err := store.GetCheckpoints(ctx, "test-run-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(cps) != 2 {
		t.Fatalf("expected 2 checkpoints, got %d", len(cps))
	}
	if cps[0].Iteration != 100 {
		t.Errorf("expected iteration 100, got %d", cps[0].Iteration)
	}
	if cps[0].IsBest {
		t.Error("checkpoint 100 should not be best")
	}
	if !cps[1].IsBest {
		t.Error("checkpoint 500 should be best")
	}
	if cps[1].FileSizeBytes == nil || *cps[1].FileSizeBytes != 2048 {
		t.Errorf("expected file_size_bytes 2048, got %v", cps[1].FileSizeBytes)
	}
}

func TestCheckRunningWithPID(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	runDir := filepath.Join(tmpDir, "test-run-1")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Write our own PID — we know this process is alive.
	pid := os.Getpid()
	pidFile := filepath.Join(runDir, "train.pid")
	if err := os.WriteFile(pidFile, []byte(strconv.Itoa(pid)), 0644); err != nil {
		t.Fatal(err)
	}

	if !store.checkRunning("test-run-1") {
		t.Error("expected checkRunning to return true for our own PID")
	}
}

func TestCheckRunningWithFakePID(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	runDir := filepath.Join(tmpDir, "test-run-1")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Write a PID that almost certainly doesn't exist.
	pidFile := filepath.Join(runDir, "train.pid")
	if err := os.WriteFile(pidFile, []byte("9999999"), 0644); err != nil {
		t.Fatal(err)
	}

	if store.checkRunning("test-run-1") {
		t.Error("expected checkRunning to return false for fake PID")
	}
}

func TestCheckRunningNoPIDFile(t *testing.T) {
	store, _ := setupTestDB(t)
	if store.checkRunning("nonexistent-run") {
		t.Error("expected checkRunning to return false for missing PID file")
	}
}
