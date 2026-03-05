package training

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func setupTestServer(t *testing.T) (*TrainingStore, *httptest.Server) {
	t.Helper()
	store, _ := setupTestDB(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/training/runs", store.HandleListRuns)
	mux.HandleFunc("/training/runs/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) == 3 {
			store.HandleGetRun(w, r)
		} else if len(parts) == 4 {
			switch parts[3] {
			case "metrics":
				store.HandleGetMetrics(w, r)
			case "checkpoints":
				store.HandleGetCheckpoints(w, r)
			default:
				http.NotFound(w, r)
			}
		} else {
			http.NotFound(w, r)
		}
	})

	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return store, ts
}

func TestHandlerListRuns(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json, got %s", ct)
	}

	var runs []Run
	if err := json.NewDecoder(resp.Body).Decode(&runs); err != nil {
		t.Fatal(err)
	}
	if len(runs) != 2 {
		t.Fatalf("expected 2 runs, got %d", len(runs))
	}
}

func TestHandlerGetRun(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var detail RunDetail
	if err := json.NewDecoder(resp.Body).Decode(&detail); err != nil {
		t.Fatal(err)
	}
	if detail.Name != "test-run-1" {
		t.Errorf("expected test-run-1, got %s", detail.Name)
	}
	if detail.ConfigYAML != "learning_rate: 0.001" {
		t.Errorf("expected config yaml, got %q", detail.ConfigYAML)
	}
}

func TestHandlerGetRunNotFound(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", resp.StatusCode)
	}
}

func TestHandlerGetMetrics(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var metrics []EvalMetric
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		t.Fatal(err)
	}
	if len(metrics) != 11 {
		t.Fatalf("expected 11 metrics, got %d", len(metrics))
	}
}

func TestHandlerGetMetricsBaselineFilter(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1/metrics?baseline=random")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var metrics []EvalMetric
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		t.Fatal(err)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected 1 metric for baseline=random, got %d", len(metrics))
	}
}

func TestHandlerGetMetricsMeanImp(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1/metrics?aggregate=mean_imp")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var points []MeanImpPoint
	if err := json.NewDecoder(resp.Body).Decode(&points); err != nil {
		t.Fatal(err)
	}
	if len(points) != 2 {
		t.Fatalf("expected 2 mean_imp points, got %d", len(points))
	}
	if points[0].MeanImp < 0.399 || points[0].MeanImp > 0.401 {
		t.Errorf("expected ~0.40, got %f", points[0].MeanImp)
	}
}

func TestHandlerGetCheckpoints(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1/checkpoints")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var cps []Checkpoint
	if err := json.NewDecoder(resp.Body).Decode(&cps); err != nil {
		t.Fatal(err)
	}
	if len(cps) != 2 {
		t.Fatalf("expected 2 checkpoints, got %d", len(cps))
	}
	if !cps[1].IsBest {
		t.Error("checkpoint 500 should be best")
	}
}

func TestHandlerUnknownSubpath(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Get(ts.URL + "/training/runs/test-run-1/unknown")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", resp.StatusCode)
	}
}

func TestHandlerMethodNotAllowed(t *testing.T) {
	_, ts := setupTestServer(t)

	resp, err := http.Post(ts.URL+"/training/runs", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", resp.StatusCode)
	}
}
