package training

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestComparisonGetComparison(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	resp, err := store.GetComparison(ctx, []string{"test-run-1", "test-run-2"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Runs) != 2 {
		t.Fatalf("expected 2 runs, got %d", len(resp.Runs))
	}

	r1 := resp.Runs[0]
	if r1.Name != "test-run-1" {
		t.Fatalf("expected test-run-1 first, got %s", r1.Name)
	}
	if r1.Algorithm != "os-mccfr" {
		t.Errorf("expected os-mccfr, got %s", r1.Algorithm)
	}
	if r1.BestMetricValue == nil || *r1.BestMetricValue != 0.42 {
		t.Errorf("expected best_metric_value 0.42, got %v", r1.BestMetricValue)
	}
	if r1.BestMetricIter == nil || *r1.BestMetricIter != 500 {
		t.Errorf("expected best_metric_iter 500, got %v", r1.BestMetricIter)
	}
	if len(r1.MeanImp) != 2 {
		t.Fatalf("expected 2 mean_imp points, got %d", len(r1.MeanImp))
	}
	if len(r1.FinalBaselines) != 5 {
		t.Fatalf("expected 5 final baselines (latest iter=200), got %d", len(r1.FinalBaselines))
	}
	for _, m := range r1.FinalBaselines {
		if m.Iteration != 200 {
			t.Errorf("expected final baselines at iteration 200, got %d", m.Iteration)
		}
		if m.WinRate == nil || *m.WinRate != 0.50 {
			t.Errorf("expected win_rate 0.50 at iter 200, got %v", m.WinRate)
		}
	}

	r2 := resp.Runs[1]
	if r2.Name != "test-run-2" {
		t.Fatalf("expected test-run-2 second, got %s", r2.Name)
	}
	if r2.Algorithm != "sd-cfr" {
		t.Errorf("expected sd-cfr, got %s", r2.Algorithm)
	}
	if r2.BestMetricValue == nil || *r2.BestMetricValue != 0.35 {
		t.Errorf("expected best_metric_value 0.35, got %v", r2.BestMetricValue)
	}
	if len(r2.MeanImp) != 0 {
		t.Errorf("expected no mean_imp points for test-run-2 (no eval rows), got %d", len(r2.MeanImp))
	}
	if len(r2.FinalBaselines) != 0 {
		t.Errorf("expected no final baselines for test-run-2 (no eval rows), got %d", len(r2.FinalBaselines))
	}
}

func TestComparisonUnknownRun(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	resp, err := store.GetComparison(ctx, []string{"does-not-exist"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Runs) != 1 {
		t.Fatalf("expected 1 run entry, got %d", len(resp.Runs))
	}
	rc := resp.Runs[0]
	if rc.Name != "does-not-exist" {
		t.Errorf("expected name passthrough, got %s", rc.Name)
	}
	if rc.Algorithm != "" {
		t.Errorf("expected empty algorithm, got %s", rc.Algorithm)
	}
	if rc.BestMetricValue != nil {
		t.Errorf("expected nil best_metric_value, got %v", rc.BestMetricValue)
	}
	if rc.BestMetricIter != nil {
		t.Errorf("expected nil best_metric_iter, got %v", rc.BestMetricIter)
	}
	if len(rc.MeanImp) != 0 {
		t.Errorf("expected empty mean_imp, got %d", len(rc.MeanImp))
	}
	if len(rc.FinalBaselines) != 0 {
		t.Errorf("expected empty final_baselines, got %d", len(rc.FinalBaselines))
	}
}

func TestComparisonMixedKnownUnknown(t *testing.T) {
	store, _ := setupTestDB(t)
	ctx := context.Background()

	resp, err := store.GetComparison(ctx, []string{"test-run-1", "does-not-exist"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Runs) != 2 {
		t.Fatalf("expected 2 run entries, got %d", len(resp.Runs))
	}
	if resp.Runs[0].Name != "test-run-1" || len(resp.Runs[0].MeanImp) == 0 {
		t.Errorf("expected populated test-run-1 entry, got %+v", resp.Runs[0])
	}
	if resp.Runs[1].Name != "does-not-exist" || resp.Runs[1].Algorithm != "" {
		t.Errorf("expected empty unknown-run entry, got %+v", resp.Runs[1])
	}
}

func TestComparisonHandleSuccess(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=test-run-1,test-run-2", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestComparisonHandleTrimsAndDropsEmpty(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=%20test-run-1%20,,test-run-2", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestComparisonHandleEmptyBadRequest(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty runs, got %d", w.Code)
	}
}

func TestComparisonHandleMissingParamBadRequest(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing runs param, got %d", w.Code)
	}
}

func TestComparisonHandleTooManyBadRequest(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=a,b,c,d,e,f,g", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for 7 runs, got %d", w.Code)
	}
}

func TestComparisonHandleSixRunsOK(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=a,b,c,d,e,f", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for 6 runs (at cap), got %d", w.Code)
	}
}

func TestComparisonHandleMethodNotAllowed(t *testing.T) {
	store, _ := setupTestDB(t)

	req := httptest.NewRequest(http.MethodPost, "/training/compare?runs=test-run-1", nil)
	w := httptest.NewRecorder()
	store.HandleCompare(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}
