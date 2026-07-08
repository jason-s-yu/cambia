// Package training provides a read-only store and HTTP/WebSocket handlers
// for the training dashboard. It reads from the SQLite run database written
// by the Python CFR pipeline.
package training

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// meanImpBaselines is the canonical set of baselines used to compute mean_imp.
// Must match MEAN_IMP_BASELINES in cfr/src/evaluate_agents.py.
var meanImpBaselines = []string{
	"random_no_cambia",
	"random_late_cambia",
	"imperfect_greedy",
	"memory_heuristic",
	"aggressive_snap",
}

// Run is a summary of a training run.
type Run struct {
	ID              int      `json:"id"`
	Name            string   `json:"name"`
	Algorithm       string   `json:"algorithm"`
	Status          string   `json:"status"`
	BestMetricValue *float64 `json:"best_metric_value"`
	BestMetricIter  *int     `json:"best_metric_iter"`
	CreatedAt       string   `json:"created_at"`
	UpdatedAt       string   `json:"updated_at"`
	// Process is the Go-owned current-process-state record read from
	// runs/<name>/process.json, or nil when the run has none (an external
	// run_db-only run). It carries the live pid/pgid and lifecycle status.
	Process *ProcessState `json:"process,omitempty"`
}

// RunDetail extends Run with configuration and metadata.
type RunDetail struct {
	Run
	ConfigYAML string `json:"config_yaml"`
	Notes      string `json:"notes,omitempty"`
	Tags       string `json:"tags,omitempty"`
}

// EvalMetric is a single evaluation result row.
type EvalMetric struct {
	Iteration   int      `json:"iteration"`
	Baseline    string   `json:"baseline"`
	WinRate     *float64 `json:"win_rate"`
	CILow       *float64 `json:"ci_low"`
	CIHigh      *float64 `json:"ci_high"`
	GamesPlayed *int     `json:"games_played"`
	AdvLoss     *float64 `json:"adv_loss"`
	StratLoss   *float64 `json:"strat_loss"`
	Timestamp   string   `json:"timestamp"`
}

// MeanImpPoint is the mean win rate across the 5 baselines for one iteration.
type MeanImpPoint struct {
	Iteration int     `json:"iteration"`
	MeanImp   float64 `json:"mean_imp"`
}

// Checkpoint is a saved model checkpoint.
type Checkpoint struct {
	ID            int    `json:"id"`
	Iteration     int    `json:"iteration"`
	FilePath      string `json:"file_path"`
	FileSizeBytes *int64 `json:"file_size_bytes"`
	CreatedAt     string `json:"created_at"`
	IsBest        bool   `json:"is_best"`
}

// TrainingStore provides read-only access to the CFR run database.
type TrainingStore struct {
	runsDir    string
	db         *sql.DB
	mu         sync.RWMutex
	cachedRuns []Run
	cancel     context.CancelFunc
}

// NewTrainingStore opens the SQLite run database in read-only mode and starts
// a background goroutine to refresh the cached run list every 10 seconds.
func NewTrainingStore(runsDir string) (*TrainingStore, error) {
	dbPath := filepath.Join(runsDir, "cambia_runs.db")
	dsn := fmt.Sprintf("file:%s?mode=ro", dbPath)

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open training db: %w", err)
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping training db: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	s := &TrainingStore{
		runsDir: runsDir,
		db:      db,
		cancel:  cancel,
	}

	// Initial cache load.
	_ = s.refreshCache(ctx)

	// Background refresh.
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				_ = s.refreshCache(ctx)
			}
		}
	}()

	return s, nil
}

// Close shuts down the background goroutine and closes the database.
func (s *TrainingStore) Close() {
	s.cancel()
	s.db.Close()
}

// refreshCache queries all runs and caches them, overlaying process.json
// current state and merging in dashboard-created runs that have no run_db row.
func (s *TrainingStore) refreshCache(ctx context.Context) error {
	runs, err := s.queryRuns(ctx)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.cachedRuns = runs
	s.mu.Unlock()
	return nil
}

func (s *TrainingStore) queryRuns(ctx context.Context) ([]Run, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, name, algorithm, status,
		       best_metric_value, best_metric_iter,
		       created_at, updated_at
		FROM runs
		ORDER BY updated_at DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var runs []Run
	for rows.Next() {
		var r Run
		if err := rows.Scan(
			&r.ID, &r.Name, &r.Algorithm, &r.Status,
			&r.BestMetricValue, &r.BestMetricIter,
			&r.CreatedAt, &r.UpdatedAt,
		); err != nil {
			return nil, err
		}
		// Overlay the process.json current state (status + pid liveness).
		s.applyProcessState(&r)
		runs = append(runs, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Merge dashboard-created runs that exist only as process.json (created but
	// not yet registered in the run database by the trainer).
	seen := make(map[string]bool, len(runs))
	for i := range runs {
		seen[runs[i].Name] = true
	}
	states, _ := scanProcessStates(s.runsDir)
	for _, st := range states {
		if seen[st.Name] {
			continue
		}
		runs = append(runs, Run{
			Name:      st.Name,
			Algorithm: st.Algorithm,
			Status:    effectiveStatus(st),
			CreatedAt: st.CreatedAt,
			UpdatedAt: st.CreatedAt,
			Process:   st,
		})
	}

	// RFC3339 timestamps sort lexically in chronological order; newest first.
	sort.SliceStable(runs, func(i, j int) bool {
		return runs[i].UpdatedAt > runs[j].UpdatedAt
	})
	return runs, nil
}

// ListRuns returns the cached run list.
func (s *TrainingStore) ListRuns(ctx context.Context) ([]Run, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.cachedRuns == nil {
		return []Run{}, nil
	}
	out := make([]Run, len(s.cachedRuns))
	copy(out, s.cachedRuns)
	return out, nil
}

// GetRun returns detail for a single run by name, including the latest config snapshot.
func (s *TrainingStore) GetRun(ctx context.Context, name string) (*RunDetail, error) {
	var rd RunDetail
	var notes, tags sql.NullString
	err := s.db.QueryRowContext(ctx, `
		SELECT r.id, r.name, r.algorithm, r.status,
		       r.best_metric_value, r.best_metric_iter,
		       r.created_at, r.updated_at,
		       r.notes, r.tags
		FROM runs r
		WHERE r.name = ?
	`, name).Scan(
		&rd.ID, &rd.Name, &rd.Algorithm, &rd.Status,
		&rd.BestMetricValue, &rd.BestMetricIter,
		&rd.CreatedAt, &rd.UpdatedAt,
		&notes, &tags,
	)
	if err == sql.ErrNoRows {
		// No run_db row yet: a dashboard-created run lives only in process.json.
		return s.processOnlyDetail(name), nil
	}
	if err != nil {
		return nil, err
	}
	if notes.Valid {
		rd.Notes = notes.String
	}
	if tags.Valid {
		rd.Tags = tags.String
	}

	// Overlay the process.json current state (status + pid liveness + record).
	if st, ok := s.processStateFor(rd.Name); ok {
		rd.Status = effectiveStatus(st)
		rd.Process = st
	}

	// Fetch latest config snapshot.
	var configYAML sql.NullString
	_ = s.db.QueryRowContext(ctx, `
		SELECT config_yaml
		FROM config_snapshots
		WHERE run_id = ?
		ORDER BY created_at DESC
		LIMIT 1
	`, rd.ID).Scan(&configYAML)
	if configYAML.Valid {
		rd.ConfigYAML = configYAML.String
	}

	return &rd, nil
}

// GetMetrics returns evaluation metrics for a run, optionally filtered by baseline.
func (s *TrainingStore) GetMetrics(ctx context.Context, runName string, baseline string) ([]EvalMetric, error) {
	var query string
	var args []interface{}
	if baseline != "" {
		query = `
			SELECT e.iteration, e.baseline, e.win_rate, e.ci_low, e.ci_high,
			       e.games_played, e.adv_loss, e.strat_loss, e.timestamp
			FROM eval_results e
			JOIN runs r ON e.run_id = r.id
			WHERE r.name = ? AND e.baseline = ?
			ORDER BY e.iteration ASC
		`
		args = []interface{}{runName, baseline}
	} else {
		query = `
			SELECT e.iteration, e.baseline, e.win_rate, e.ci_low, e.ci_high,
			       e.games_played, e.adv_loss, e.strat_loss, e.timestamp
			FROM eval_results e
			JOIN runs r ON e.run_id = r.id
			WHERE r.name = ?
			ORDER BY e.iteration ASC, e.baseline ASC
		`
		args = []interface{}{runName}
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []EvalMetric
	for rows.Next() {
		var m EvalMetric
		if err := rows.Scan(
			&m.Iteration, &m.Baseline, &m.WinRate,
			&m.CILow, &m.CIHigh, &m.GamesPlayed,
			&m.AdvLoss, &m.StratLoss, &m.Timestamp,
		); err != nil {
			return nil, err
		}
		metrics = append(metrics, m)
	}
	if metrics == nil {
		metrics = []EvalMetric{}
	}
	return metrics, rows.Err()
}

// GetMeanImp computes mean win rate across the 5 canonical baselines per iteration.
func (s *TrainingStore) GetMeanImp(ctx context.Context, runName string) ([]MeanImpPoint, error) {
	// Build placeholders for the IN clause.
	placeholders := make([]string, len(meanImpBaselines))
	args := make([]interface{}, 0, len(meanImpBaselines)+1)
	args = append(args, runName)
	for i, b := range meanImpBaselines {
		placeholders[i] = "?"
		args = append(args, b)
	}

	query := fmt.Sprintf(`
		SELECT e.iteration, AVG(e.win_rate) as mean_imp
		FROM eval_results e
		JOIN runs r ON e.run_id = r.id
		WHERE r.name = ?
		  AND e.baseline IN (%s)
		  AND e.win_rate IS NOT NULL
		GROUP BY e.iteration
		HAVING COUNT(DISTINCT e.baseline) = %d
		ORDER BY e.iteration ASC
	`, strings.Join(placeholders, ","), len(meanImpBaselines))

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var points []MeanImpPoint
	for rows.Next() {
		var p MeanImpPoint
		if err := rows.Scan(&p.Iteration, &p.MeanImp); err != nil {
			return nil, err
		}
		points = append(points, p)
	}
	if points == nil {
		points = []MeanImpPoint{}
	}
	return points, rows.Err()
}

// GetCheckpoints returns checkpoints for a run.
func (s *TrainingStore) GetCheckpoints(ctx context.Context, runName string) ([]Checkpoint, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT c.id, c.iteration, c.file_path, c.file_size_bytes,
		       c.created_at, c.is_best
		FROM checkpoints c
		JOIN runs r ON c.run_id = r.id
		WHERE r.name = ?
		ORDER BY c.iteration ASC
	`, runName)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cps []Checkpoint
	for rows.Next() {
		var cp Checkpoint
		var isBest int
		if err := rows.Scan(
			&cp.ID, &cp.Iteration, &cp.FilePath, &cp.FileSizeBytes,
			&cp.CreatedAt, &isBest,
		); err != nil {
			return nil, err
		}
		cp.IsBest = isBest != 0
		cps = append(cps, cp)
	}
	if cps == nil {
		cps = []Checkpoint{}
	}
	return cps, rows.Err()
}

// runDirOf returns the run directory for name under runsDir.
func runDirOf(runsDir, name string) string {
	return filepath.Join(runsDir, name)
}

// effectiveStatus returns st.Status with pid liveness applied: a run recorded
// as running/starting/stopping whose pid is no longer alive is reported as
// crashed. This is the read-time view; Reconcile persists the same repair at
// server start.
func effectiveStatus(st *ProcessState) string {
	switch st.Status {
	case StatusRunning, StatusStarting, StatusStopping:
		if !pidAlive(st.PID) {
			return StatusCrashed
		}
	}
	return st.Status
}

// processStateFor reads runs/<name>/process.json, returning the state and
// whether it exists. process.json is the Go-owned current-state authority; it
// replaces the legacy train.pid liveness probe.
func (s *TrainingStore) processStateFor(name string) (*ProcessState, bool) {
	st, err := readProcessState(runDirOf(s.runsDir, name))
	if err != nil {
		return nil, false
	}
	return st, true
}

// applyProcessState overlays the process.json current state onto r: the
// effective status plus the full ProcessState record. A run_db row with no
// process.json keeps its stored status (externally launched, unsupervised).
func (s *TrainingStore) applyProcessState(r *Run) {
	if st, ok := s.processStateFor(r.Name); ok {
		r.Status = effectiveStatus(st)
		r.Process = st
	}
}

// processOnlyDetail builds a RunDetail from process.json and the materialized
// config.yaml for a run created through the dashboard but not yet registered in
// the run database. It returns nil when no process.json exists.
func (s *TrainingStore) processOnlyDetail(name string) *RunDetail {
	st, ok := s.processStateFor(name)
	if !ok {
		return nil
	}
	rd := &RunDetail{
		Run: Run{
			Name:      st.Name,
			Algorithm: st.Algorithm,
			Status:    effectiveStatus(st),
			CreatedAt: st.CreatedAt,
			UpdatedAt: st.CreatedAt,
			Process:   st,
		},
	}
	if data, err := os.ReadFile(filepath.Join(runDirOf(s.runsDir, name), "config.yaml")); err == nil {
		rd.ConfigYAML = string(data)
	}
	return rd
}
