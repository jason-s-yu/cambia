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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/service/internal/harnessproxy"
	_ "modernc.org/sqlite"
)

// TrainingStore implements procmgr.RunResolver: the process layer reads the run
// directory layout and pid-liveness-aware status through this interface rather
// than importing the store type directly.
var _ procmgr.RunResolver = (*TrainingStore)(nil)

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
	Process *procmgr.ProcessState `json:"process,omitempty"`
	// Host is the origin host of a remote (serving-harness) run, empty for a
	// local run. It comes from the run's process.json Host field or, on the
	// client, the runs.origin_host column / harness_sync row stamped by the reconciler.
	// A non-empty Host marks the run read-only on this dashboard in v1.
	Host string `json:"host,omitempty"`
	// LastSyncAt is the RFC3339 timestamp of the last successful pull for a
	// remote run (harness_sync.last_sync_at). Nil for a local run or a remote
	// run never yet synced.
	LastSyncAt *string `json:"last_sync_at,omitempty"`
	// Stale reports whether a remote run's synced projection is older than
	// 3 sync intervals (the bounded-stale threshold). Always false for a local
	// run and for a remote run whose last_sync_at is unknown/unparseable.
	Stale bool `json:"stale"`
	// RemoteControllable is true when this remote run's origin host matches the
	// dashboard's configured harness proxy, so stop/resume can be forwarded to
	// the runner control plane (cambia-295 v1.1). False for a local run, a
	// remote run from an unconfigured origin, or when no harness proxy is
	// configured. This is the single field the frontend reads to decide whether
	// to render remote process controls (mirrored as remote_controllable in
	// web/src/types/training.ts).
	RemoteControllable bool `json:"remote_controllable"`
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

// defaultSyncIntervalSec is the pull interval assumed when CAMBIA_SYNC_INTERVAL_SEC
// is unset or invalid. The reconciler pulls live remote runs on this cadence
// (design 4.1); a remote run is flagged stale once its last_sync_at is older
// than 3 intervals (design 4.5).
const defaultSyncIntervalSec = 60

// TrainingStore provides read-only access to the CFR run database.
type TrainingStore struct {
	runsDir    string
	db         *sql.DB
	mu         sync.RWMutex
	cachedRuns []Run
	cancel     context.CancelFunc
	// syncIntervalSec is the pull cadence used to compute the bounded-stale
	// threshold (3 * interval). Read once from CAMBIA_SYNC_INTERVAL_SEC.
	syncIntervalSec int
	// hasOriginHost and hasHarnessSync record whether the opened db carries the
	// serving-harness schema (runs.origin_host column and the harness_sync
	// table). An older local-only db predating those additions is read without
	// them rather than failing every query (design failure mode: stale/degraded,
	// never wrong-but-confident).
	hasOriginHost  bool
	hasHarnessSync bool
	// proxy is the harness control-plane client, or nil when no harness config
	// is present (the no-proxy state: remote runs stay read-only, log tails use
	// the synced file). It supplies the origin host used to mark a remote run
	// RemoteControllable and is the WS log-proxy dialer.
	proxy *harnessproxy.Client
}

// SetHarnessProxy injects the harness control-plane client (or nil for no
// proxy). Called once at construction in main.go before the store serves any
// request. A non-nil client makes remote runs whose origin matches its
// OriginHost controllable and enables the WS log proxy.
func (s *TrainingStore) SetHarnessProxy(c *harnessproxy.Client) {
	s.proxy = c
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
		runsDir:         runsDir,
		db:              db,
		cancel:          cancel,
		syncIntervalSec: syncIntervalFromEnv(),
	}
	s.detectHarnessSchema(ctx)

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

// syncIntervalFromEnv reads CAMBIA_SYNC_INTERVAL_SEC with the same defensive
// parsing as the dashboard's other env rails (main.go): a missing, non-numeric,
// or non-positive value falls back to defaultSyncIntervalSec.
func syncIntervalFromEnv() int {
	if v := os.Getenv("CAMBIA_SYNC_INTERVAL_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultSyncIntervalSec
}

// detectHarnessSchema probes for the serving-harness additions (runs.origin_host
// column and the harness_sync table) so remote-provenance queries are only
// issued against a db that carries them. A db predating those additions is
// served as local-only rather than 500ing every list/detail request.
func (s *TrainingStore) detectHarnessSchema(ctx context.Context) {
	rows, err := s.db.QueryContext(ctx, `PRAGMA table_info(runs)`)
	if err == nil {
		for rows.Next() {
			var cid int
			var name, ctype string
			var notnull, pk int
			var dflt sql.NullString
			if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
				continue
			}
			if name == "origin_host" {
				s.hasOriginHost = true
			}
		}
		rows.Close()
	}
	var tbl string
	err = s.db.QueryRowContext(ctx,
		`SELECT name FROM sqlite_master WHERE type='table' AND name='harness_sync'`).Scan(&tbl)
	s.hasHarnessSync = err == nil && tbl == "harness_sync"
}

// syncRow is a run's serving-harness pull record from the harness_sync table.
type syncRow struct {
	originHost string
	lastSyncAt *string
}

// harnessSyncFor returns the harness_sync record for one run, or nil when the
// table is absent or has no row for name. It is the current-state store for pull
// freshness (design 4.5), opened read-only through the same db as everything
// else.
func (s *TrainingStore) harnessSyncFor(ctx context.Context, name string) *syncRow {
	if !s.hasHarnessSync {
		return nil
	}
	var host sql.NullString
	var ts sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT origin_host, last_sync_at FROM harness_sync WHERE run_name = ?`, name).Scan(&host, &ts)
	if err != nil {
		return nil
	}
	row := &syncRow{}
	if host.Valid {
		row.originHost = host.String
	}
	if ts.Valid {
		v := ts.String
		row.lastSyncAt = &v
	}
	return row
}

// harnessSyncMap loads every harness_sync row into a name-keyed map for the run
// list overlay. Returns an empty map when the table is absent.
func (s *TrainingStore) harnessSyncMap(ctx context.Context) map[string]*syncRow {
	out := make(map[string]*syncRow)
	if !s.hasHarnessSync {
		return out
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT run_name, origin_host, last_sync_at FROM harness_sync`)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		var host, ts sql.NullString
		if err := rows.Scan(&name, &host, &ts); err != nil {
			continue
		}
		row := &syncRow{}
		if host.Valid {
			row.originHost = host.String
		}
		if ts.Valid {
			v := ts.String
			row.lastSyncAt = &v
		}
		out[name] = row
	}
	return out
}

// isStale reports whether lastSyncAt (RFC3339) is older than 3 sync intervals.
// An unparseable timestamp returns false: the dashboard degrades to "fresh
// unknown" rather than asserting staleness it cannot substantiate.
func (s *TrainingStore) isStale(lastSyncAt string) bool {
	t, err := time.Parse(time.RFC3339, lastSyncAt)
	if err != nil {
		return false
	}
	threshold := time.Duration(3*s.syncIntervalSec) * time.Second
	return time.Since(t) > threshold
}

// applyRemoteProvenance sets r.Host, r.LastSyncAt, and r.Stale from the run's
// process.json Host, the runs.origin_host column (dbOriginHost), and the
// harness_sync row (sr). When the run is remote it also stamps Host onto the
// ProcessState BEFORE any EffectiveStatus call so procmgr short-circuits the
// local pid probe (a client probe against a runner pid is the cross-host
// pid-reuse bug). Callers must invoke this before deriving r.Status from the
// process state. A local run (no host from any source) is left untouched.
func (s *TrainingStore) applyRemoteProvenance(r *Run, dbOriginHost string, sr *syncRow) {
	host := ""
	switch {
	case r.Process != nil && r.Process.Host != "":
		host = r.Process.Host
	case dbOriginHost != "":
		host = dbOriginHost
	case sr != nil && sr.originHost != "":
		host = sr.originHost
	}
	if host == "" {
		return
	}
	r.Host = host
	if r.Process != nil {
		r.Process.Host = host
	}
	if sr != nil && sr.lastSyncAt != nil {
		r.LastSyncAt = sr.lastSyncAt
		r.Stale = s.isStale(*sr.lastSyncAt)
	}
	// A remote run is controllable only when a harness proxy is configured and
	// this run's origin matches the proxy's origin host: an unknown-origin
	// remote run stays read-only (the dashboard has no pinned path to its
	// runner).
	if s.proxy != nil && host == s.proxy.OriginHost() {
		r.RemoteControllable = true
	}
}

// RemoteHost returns the origin host of a run when it is remote (serving-harness
// synced), or "" for a local run. It is the authority the process handlers use
// to refuse start/stop/resume on remote runs (read-only on this dashboard in
// v1): a run is remote if its process.json carries a Host, or the runs.origin_host
// column / harness_sync row records an origin host.
func (s *TrainingStore) RemoteHost(ctx context.Context, name string) string {
	if st, ok := s.processStateFor(name); ok && st.Host != "" {
		return st.Host
	}
	if s.hasOriginHost {
		var host sql.NullString
		if err := s.db.QueryRowContext(ctx,
			`SELECT origin_host FROM runs WHERE name = ?`, name).Scan(&host); err == nil && host.Valid && host.String != "" {
			return host.String
		}
	}
	if sr := s.harnessSyncFor(ctx, name); sr != nil && sr.originHost != "" {
		return sr.originHost
	}
	return ""
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
	originSel := "NULL AS origin_host"
	if s.hasOriginHost {
		originSel = "origin_host"
	}
	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
		SELECT id, name, algorithm, status,
		       best_metric_value, best_metric_iter,
		       created_at, updated_at, %s
		FROM runs
		ORDER BY updated_at DESC
	`, originSel))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	syncMap := s.harnessSyncMap(ctx)

	var runs []Run
	for rows.Next() {
		var r Run
		var originHost sql.NullString
		if err := rows.Scan(
			&r.ID, &r.Name, &r.Algorithm, &r.Status,
			&r.BestMetricValue, &r.BestMetricIter,
			&r.CreatedAt, &r.UpdatedAt, &originHost,
		); err != nil {
			return nil, err
		}
		// Overlay the process.json current state (status + pid liveness) and the
		// remote provenance (host + staleness).
		s.applyProcessState(&r, nullString(originHost), syncMap[r.Name])
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
	states, _ := procmgr.ScanProcessStates(s.runsDir)
	for _, st := range states {
		if seen[st.Name] {
			continue
		}
		run := Run{
			Name:      st.Name,
			Algorithm: st.Algorithm,
			CreatedAt: st.CreatedAt,
			UpdatedAt: st.CreatedAt,
			Process:   st,
		}
		// Stamp remote provenance (host) before EffectiveStatus so a synced
		// remote run short-circuits the local pid probe.
		s.applyRemoteProvenance(&run, "", syncMap[st.Name])
		run.Status = procmgr.EffectiveStatus(st)
		runs = append(runs, run)
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
	var notes, tags, originHost sql.NullString
	originSel := "NULL AS origin_host"
	if s.hasOriginHost {
		originSel = "r.origin_host"
	}
	err := s.db.QueryRowContext(ctx, fmt.Sprintf(`
		SELECT r.id, r.name, r.algorithm, r.status,
		       r.best_metric_value, r.best_metric_iter,
		       r.created_at, r.updated_at,
		       r.notes, r.tags, %s
		FROM runs r
		WHERE r.name = ?
	`, originSel), name).Scan(
		&rd.ID, &rd.Name, &rd.Algorithm, &rd.Status,
		&rd.BestMetricValue, &rd.BestMetricIter,
		&rd.CreatedAt, &rd.UpdatedAt,
		&notes, &tags, &originHost,
	)
	if err == sql.ErrNoRows {
		// No run_db row yet: a dashboard-created run lives only in process.json.
		return s.processOnlyDetail(ctx, name), nil
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

	// Overlay the process.json current state (status + pid liveness + record)
	// and the remote provenance (host + staleness). Host is stamped before
	// EffectiveStatus so a remote run short-circuits the local pid probe.
	sr := s.harnessSyncFor(ctx, rd.Name)
	if st, ok := s.processStateFor(rd.Name); ok {
		rd.Process = st
		s.applyRemoteProvenance(&rd.Run, nullString(originHost), sr)
		rd.Status = procmgr.EffectiveStatus(st)
	} else {
		s.applyRemoteProvenance(&rd.Run, nullString(originHost), sr)
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

// RunDir returns the run directory for name under the store's runs root. It is
// the procmgr.RunResolver hook the process layer uses instead of importing the
// store's path logic.
func (s *TrainingStore) RunDir(name string) string {
	return filepath.Join(s.runsDir, name)
}

// EffectiveStatus returns st.Status with pid liveness applied, delegating to the
// procmgr helper. It is the procmgr.RunResolver hook: a run recorded as
// running/starting/stopping whose pid is no longer alive reports as crashed.
func (s *TrainingStore) EffectiveStatus(st *procmgr.ProcessState) string {
	return procmgr.EffectiveStatus(st)
}

// processStateFor reads runs/<name>/process.json, returning the state and
// whether it exists. process.json is the Go-owned current-state authority; it
// replaces the legacy train.pid liveness probe.
func (s *TrainingStore) processStateFor(name string) (*procmgr.ProcessState, bool) {
	st, err := procmgr.ReadProcessState(s.RunDir(name))
	if err != nil {
		return nil, false
	}
	return st, true
}

// applyProcessState overlays the process.json current state onto r: the
// effective status plus the full ProcessState record, and the remote provenance
// (host + staleness) derived from dbOriginHost and sr. Host is stamped onto the
// process state before EffectiveStatus so a remote run short-circuits the local
// pid probe. A run_db row with no process.json keeps its stored status
// (externally launched, unsupervised) but still records remote provenance.
func (s *TrainingStore) applyProcessState(r *Run, dbOriginHost string, sr *syncRow) {
	if st, ok := s.processStateFor(r.Name); ok {
		r.Process = st
		s.applyRemoteProvenance(r, dbOriginHost, sr)
		r.Status = procmgr.EffectiveStatus(st)
		return
	}
	s.applyRemoteProvenance(r, dbOriginHost, sr)
}

// processOnlyDetail builds a RunDetail from process.json and the materialized
// config.yaml for a run created through the dashboard but not yet registered in
// the run database. It returns nil when no process.json exists.
func (s *TrainingStore) processOnlyDetail(ctx context.Context, name string) *RunDetail {
	st, ok := s.processStateFor(name)
	if !ok {
		return nil
	}
	rd := &RunDetail{
		Run: Run{
			Name:      st.Name,
			Algorithm: st.Algorithm,
			CreatedAt: st.CreatedAt,
			UpdatedAt: st.CreatedAt,
			Process:   st,
		},
	}
	// Stamp remote provenance (host) before EffectiveStatus so a synced remote
	// run short-circuits the local pid probe.
	s.applyRemoteProvenance(&rd.Run, "", s.harnessSyncFor(ctx, name))
	rd.Status = procmgr.EffectiveStatus(st)
	if data, err := os.ReadFile(filepath.Join(s.RunDir(name), "config.yaml")); err == nil {
		rd.ConfigYAML = string(data)
	}
	return rd
}

// nullString unwraps a sql.NullString to its string, or "" when NULL.
func nullString(ns sql.NullString) string {
	if ns.Valid {
		return ns.String
	}
	return ""
}
