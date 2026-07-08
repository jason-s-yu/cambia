package training

import (
	"context"
	"database/sql"
	"net/http"
	"strings"
)

// maxComparisonRuns is the upper bound on runs accepted by a single
// comparison request, keeping the batch run_db read bounded.
const maxComparisonRuns = 6

// RunComparison is one run's data for a multi-run comparison view: its
// mean_imp series, best-metric summary, and per-baseline win rates at the
// latest evaluated iteration.
type RunComparison struct {
	Name            string         `json:"name"`
	Algorithm       string         `json:"algorithm"`
	BestMetricValue *float64       `json:"best_metric_value"`
	BestMetricIter  *int           `json:"best_metric_iter"`
	MeanImp         []MeanImpPoint `json:"mean_imp"`
	FinalBaselines  []EvalMetric   `json:"final_baselines"`
}

// ComparisonResponse wraps the per-run comparison data for 1-6 runs.
type ComparisonResponse struct {
	Runs []RunComparison `json:"runs"`
}

// GetComparison assembles comparison data for the given run names: the
// mean_imp series (via GetMeanImp), the algorithm + best-metric summary from
// the runs table, and the per-baseline win rates at the latest evaluated
// iteration. This is a single batch read (one query per data facet, looped
// over names) so the frontend does not fan out per-run requests. An unknown
// name yields a RunComparison with empty series and nil best-metric fields
// rather than an error, so one bad name does not fail the whole comparison.
func (s *TrainingStore) GetComparison(ctx context.Context, names []string) (*ComparisonResponse, error) {
	resp := &ComparisonResponse{Runs: make([]RunComparison, 0, len(names))}
	for _, name := range names {
		rc := RunComparison{Name: name}

		var algorithm sql.NullString
		err := s.db.QueryRowContext(ctx, `
			SELECT algorithm, best_metric_value, best_metric_iter
			FROM runs
			WHERE name = ?
		`, name).Scan(&algorithm, &rc.BestMetricValue, &rc.BestMetricIter)
		if err != nil && err != sql.ErrNoRows {
			return nil, err
		}
		if algorithm.Valid {
			rc.Algorithm = algorithm.String
		}

		meanImp, err := s.GetMeanImp(ctx, name)
		if err != nil {
			return nil, err
		}
		rc.MeanImp = meanImp

		finalBaselines, err := s.getFinalBaselines(ctx, name)
		if err != nil {
			return nil, err
		}
		rc.FinalBaselines = finalBaselines

		resp.Runs = append(resp.Runs, rc)
	}
	return resp, nil
}

// getFinalBaselines returns the per-baseline eval rows at the latest
// evaluated iteration for runName. Returns an empty slice (not an error) for
// an unknown run or a run with no eval_results rows.
func (s *TrainingStore) getFinalBaselines(ctx context.Context, runName string) ([]EvalMetric, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT e.iteration, e.baseline, e.win_rate, e.ci_low, e.ci_high,
		       e.games_played, e.adv_loss, e.strat_loss, e.timestamp
		FROM eval_results e
		JOIN runs r ON e.run_id = r.id
		WHERE r.name = ?
		  AND e.iteration = (
		      SELECT MAX(e2.iteration)
		      FROM eval_results e2
		      JOIN runs r2 ON e2.run_id = r2.id
		      WHERE r2.name = ?
		  )
		ORDER BY e.baseline ASC
	`, runName, runName)
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

// HandleCompare returns comparison data for 1-6 runs via ?runs=a,b,c.
func (s *TrainingStore) HandleCompare(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	raw := strings.Split(r.URL.Query().Get("runs"), ",")
	names := make([]string, 0, len(raw))
	for _, n := range raw {
		n = strings.TrimSpace(n)
		if n != "" {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		http.Error(w, "runs parameter required", http.StatusBadRequest)
		return
	}
	if len(names) > maxComparisonRuns {
		http.Error(w, "at most 6 runs may be compared", http.StatusBadRequest)
		return
	}

	resp, err := s.GetComparison(r.Context(), names)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}
