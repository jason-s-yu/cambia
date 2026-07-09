// cmd/server/main_test.go
//
// Wiring tests for registerTrainingRoutes: they assert the Phase 3 eval,
// resources, and compare endpoints are gated by RequireAuth, that the
// /training/runs/{name}/eval case dispatches GET->HandleList and
// POST->HandleTrigger, and that /ws/training/resources is exact-matched to the
// resource stream ahead of the log-stream fallback. The routing logic lives in a
// helper split out of main() so it can be exercised without standing up the full
// server.
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/middleware"
	"github.com/jason-s-yu/cambia/service/internal/training"
	"github.com/sirupsen/logrus"
)

// newTestTrainingMux builds a mux wired by registerTrainingRoutes against
// throwaway managers rooted at a temp runs dir. The read-only TrainingStore needs
// an existing DB file, so a minimal one is created first.
func newTestTrainingMux(t *testing.T) *http.ServeMux {
	t.Helper()
	runsDir := t.TempDir()

	dbPath := filepath.Join(runsDir, "cambia_runs.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?mode=rwc")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	db.Close()

	store, err := training.NewTrainingStore(runsDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })

	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	authWrap := func(h http.Handler) http.Handler {
		return middleware.LogMiddleware(logger)(middleware.RequireAuth(h))
	}

	cambiaBin := "cambia"
	procMgr := procmgr.NewProcessManager(runsDir, runsDir, cambiaBin, store, procmgr.TrainAlgorithms())
	procHandlers := training.NewProcessHandlers(training.ProcessHandlersConfig{
		Manager:   procMgr,
		Store:     store,
		CambiaBin: cambiaBin,
		CFRDir:    runsDir,
		RunsDir:   runsDir,
	})
	evalMgr := training.NewEvalManager(runsDir, runsDir, cambiaBin)
	evalMgr.SetMaxConcurrent(1)
	evalHandlers := training.NewEvalHandlers(training.EvalHandlersConfig{
		Manager: evalMgr,
		RunsDir: runsDir,
	})
	// A large poll interval keeps the sampler from shelling nvidia-smi during the
	// tests; the WS backfill and the one-off snapshot do not depend on it.
	resMon := training.NewResourceMonitor(runsDir, time.Hour)
	t.Cleanup(resMon.Close)

	mux := http.NewServeMux()
	registerTrainingRoutes(mux, authWrap, store, procHandlers, evalHandlers, resMon)
	return mux
}

// authCookie mints a fresh key pair and a signed token, returning the cookie
// value RequireAuth expects.
func authCookie(t *testing.T) string {
	t.Helper()
	auth.Init()
	token, err := auth.CreateJWT("wiring-test-user")
	if err != nil {
		t.Fatal(err)
	}
	return "auth_token=" + token
}

// TestTrainingRoutesRequireAuth asserts every new Phase 3 route (and the WS
// resource path) returns 401 without an auth_token cookie, i.e. all sit under
// authWrap.
func TestTrainingRoutesRequireAuth(t *testing.T) {
	mux := newTestTrainingMux(t)

	cases := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/training/system/resources"},
		{http.MethodGet, "/training/compare?runs=a,b"},
		{http.MethodGet, "/training/runs/demo/eval"},
		{http.MethodPost, "/training/runs/demo/eval"},
		{http.MethodGet, "/ws/training/resources"},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != http.StatusUnauthorized {
			t.Errorf("%s %s: expected 401 without cookie, got %d", tc.method, tc.path, w.Code)
		}
	}
}

// TestEvalRouteDispatch asserts the eval case routes by method: GET -> HandleList
// (200 with a jobs array), POST -> HandleTrigger (404 no_checkpoint on a run with
// no evaluable checkpoint). Distinct responses on the same path prove the split.
func TestEvalRouteDispatch(t *testing.T) {
	mux := newTestTrainingMux(t)
	cookie := authCookie(t)

	// GET -> HandleList.
	getReq := httptest.NewRequest(http.MethodGet, "/training/runs/demo/eval", nil)
	getReq.Header.Set("Cookie", cookie)
	getW := httptest.NewRecorder()
	mux.ServeHTTP(getW, getReq)
	if getW.Code != http.StatusOK {
		t.Fatalf("GET eval: expected 200, got %d (body %s)", getW.Code, getW.Body.String())
	}
	var listBody struct {
		Jobs []json.RawMessage `json:"jobs"`
	}
	if err := json.Unmarshal(getW.Body.Bytes(), &listBody); err != nil {
		t.Fatalf("GET eval: decode body: %v", err)
	}
	if listBody.Jobs == nil {
		t.Error("GET eval: expected a jobs array (HandleList), got none")
	}

	// POST -> HandleTrigger. No checkpoints under the temp run dir -> 404.
	postReq := httptest.NewRequest(http.MethodPost, "/training/runs/demo/eval", strings.NewReader("{}"))
	postReq.Header.Set("Cookie", cookie)
	postW := httptest.NewRecorder()
	mux.ServeHTTP(postW, postReq)
	if postW.Code != http.StatusNotFound {
		t.Fatalf("POST eval: expected 404 no_checkpoint, got %d (body %s)", postW.Code, postW.Body.String())
	}
	var errBody struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(postW.Body.Bytes(), &errBody); err != nil {
		t.Fatalf("POST eval: decode body: %v", err)
	}
	if errBody.Error != "no_checkpoint" {
		t.Errorf("POST eval: expected error no_checkpoint (HandleTrigger), got %q", errBody.Error)
	}
}

// TestResourceSnapshotRoute asserts GET /training/system/resources reaches
// HandleSnapshot (200 JSON) with a valid cookie.
func TestResourceSnapshotRoute(t *testing.T) {
	mux := newTestTrainingMux(t)
	cookie := authCookie(t)

	req := httptest.NewRequest(http.MethodGet, "/training/system/resources", nil)
	req.Header.Set("Cookie", cookie)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (body %s)", w.Code, w.Body.String())
	}
	var snap map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	if _, ok := snap["gpu_available"]; !ok {
		t.Error("snapshot missing gpu_available field")
	}
}

// TestCompareRouteReached asserts GET /training/compare reaches HandleCompare: an
// empty runs param yields a 400 bounds error (not 401/404), proving the route is
// wired to the handler.
func TestCompareRouteReached(t *testing.T) {
	mux := newTestTrainingMux(t)
	cookie := authCookie(t)

	req := httptest.NewRequest(http.MethodGet, "/training/compare?runs=", nil)
	req.Header.Set("Cookie", cookie)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on empty runs, got %d (body %s)", w.Code, w.Body.String())
	}
}

// TestWSResourceDisambiguation asserts /ws/training/resources routes to the
// resource stream, not the log streamer. A resource_backfill first frame is
// unique to resMon.HandleWS; the log streamer would treat "resources" as a run
// name and never emit it, so this frame proves the exact-match branch wins over
// the fallback.
func TestWSResourceDisambiguation(t *testing.T) {
	mux := newTestTrainingMux(t)
	cookie := authCookie(t)

	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/resources"
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	hdr := http.Header{}
	hdr.Set("Cookie", cookie)
	c, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPHeader: hdr})
	if err != nil {
		t.Fatalf("dial /ws/training/resources: %v", err)
	}
	defer c.Close(websocket.StatusNormalClosure, "done")

	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatalf("read first frame: %v", err)
	}
	var msg struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("decode first frame: %v", err)
	}
	if msg.Type != "resource_backfill" {
		t.Fatalf("expected resource_backfill from HandleWS, got %q (misrouted to log streamer?)", msg.Type)
	}
}
