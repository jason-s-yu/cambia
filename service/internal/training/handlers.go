package training

import (
	"encoding/json"
	"net/http"
	"strings"
)

// HandleListRuns returns all runs as JSON.
func (s *TrainingStore) HandleListRuns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	runs, err := s.ListRuns(r.Context())
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, runs)
}

// HandleGetRun returns detail for a single run.
func (s *TrainingStore) HandleGetRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		http.Error(w, "missing run name", http.StatusBadRequest)
		return
	}
	detail, err := s.GetRun(r.Context(), name)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	if detail == nil {
		http.Error(w, "run not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, detail)
}

// HandleGetMetrics returns evaluation metrics for a run.
// Query params:
//   - aggregate=mean_imp — return mean_imp points instead of raw metrics
//   - baseline=X — filter by baseline name
func (s *TrainingStore) HandleGetMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		http.Error(w, "missing run name", http.StatusBadRequest)
		return
	}

	if r.URL.Query().Get("aggregate") == "mean_imp" {
		points, err := s.GetMeanImp(r.Context(), name)
		if err != nil {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, points)
		return
	}

	baseline := r.URL.Query().Get("baseline")
	metrics, err := s.GetMetrics(r.Context(), name, baseline)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, metrics)
}

// HandleGetCheckpoints returns checkpoints for a run.
func (s *TrainingStore) HandleGetCheckpoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		http.Error(w, "missing run name", http.StatusBadRequest)
		return
	}
	cps, err := s.GetCheckpoints(r.Context(), name)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, cps)
}

// extractRunName parses the run name from the URL path.
// Expected patterns:
//
//	/training/runs/{name}
//	/training/runs/{name}/metrics
//	/training/runs/{name}/checkpoints
//	/ws/training/{name}/logs
func extractRunName(r *http.Request) string {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	// /training/runs/{name}... → parts[0]="training", parts[1]="runs", parts[2]=name
	if len(parts) >= 3 && parts[0] == "training" && parts[1] == "runs" {
		return parts[2]
	}
	// /ws/training/{name}/logs → parts[0]="ws", parts[1]="training", parts[2]=name
	if len(parts) >= 3 && parts[0] == "ws" && parts[1] == "training" {
		return parts[2]
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
