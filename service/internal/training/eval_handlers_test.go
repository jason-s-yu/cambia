package training

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/middleware"
	"github.com/sirupsen/logrus"
)

// evalHandlerFixture wires an EvalHandlers over an EvalManager with a stub CLI.
type evalHandlerFixture struct {
	h       *EvalHandlers
	mgr     *EvalManager
	runsDir string
}

func newEvalHandlerFixture(t *testing.T, stubBody string) *evalHandlerFixture {
	t.Helper()
	base := t.TempDir()
	runsDir := filepath.Join(base, "runs")
	cfrDir := filepath.Join(base, "cfr")
	for _, d := range []string{runsDir, cfrDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	stub := filepath.Join(base, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(stubBody), 0o755); err != nil {
		t.Fatal(err)
	}
	mgr := NewEvalManager(runsDir, cfrDir, stub)
	h := NewEvalHandlers(EvalHandlersConfig{Manager: mgr, RunsDir: runsDir})
	h.gpuQuery = fakeQuery("", exec.ErrNotFound) // CPU host by default
	return &evalHandlerFixture{h: h, mgr: mgr, runsDir: runsDir}
}

func evalReq(h http.HandlerFunc, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h(w, req)
	return w
}

func TestEvalHandlerTriggerSuccess(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	seedCheckpoint(t, f.runsDir, "run1", "snapshots", "prtcfr_checkpoint.pt")

	w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/run1/eval",
		`{"device":"cpu","games":10}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Job *EvalJob `json:"job"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Job == nil || resp.Job.Status != EvalRunning {
		t.Fatalf("job = %+v, want status running", resp.Job)
	}
	if resp.Job.Target != "latest" {
		t.Errorf("target = %q, want latest", resp.Job.Target)
	}
	waitEvalStatus(t, f.mgr, "run1", EvalSucceeded, 10*time.Second)
}

func TestEvalHandlerNoCheckpoint404(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	// No checkpoint seeded.
	w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/none/eval", `{"device":"cpu"}`)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "no_checkpoint" {
		t.Errorf("error = %v, want no_checkpoint", body["error"])
	}
}

// TestEvalHandlerCudaPreflightBlock: a cuda eval below the VRAM threshold 409s
// preflight_failed; force overrides to 202. A cpu eval never runs the GPU check.
func TestEvalHandlerCudaPreflightBlock(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	seedCheckpoint(t, f.runsDir, "gpu-run", "snapshots", "prtcfr_checkpoint.pt")
	// 512 MiB free, default requirement 4 GiB -> block.
	f.h.gpuQuery = fakeQuery("512, 30, Fake GPU\n", nil)

	w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/gpu-run/eval",
		`{"device":"cuda"}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "preflight_failed" {
		t.Errorf("error = %v, want preflight_failed", body["error"])
	}
	checks, _ := body["checks"].([]any)
	if len(checks) == 0 {
		t.Error("preflight_failed body missing checks list")
	}

	// force overrides the overridable gpu_vram block.
	wf := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/gpu-run/eval",
		`{"device":"cuda","force":true}`)
	if wf.Code != http.StatusAccepted {
		t.Fatalf("force status = %d, want 202: %s", wf.Code, wf.Body.String())
	}
	waitEvalStatus(t, f.mgr, "gpu-run", EvalSucceeded, 10*time.Second)
}

// TestEvalHandlerCudaCPUHostPasses: on a CPU host (nvidia-smi absent) a cuda
// eval passes the GPU check.
func TestEvalHandlerCudaCPUHostPasses(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	seedCheckpoint(t, f.runsDir, "cpuhost", "snapshots", "prtcfr_checkpoint.pt")
	f.h.gpuQuery = fakeQuery("", exec.ErrNotFound)

	w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/cpuhost/eval",
		`{"device":"cuda"}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202 (CPU host passes GPU check): %s", w.Code, w.Body.String())
	}
	waitEvalStatus(t, f.mgr, "cpuhost", EvalSucceeded, 10*time.Second)
}

// TestEvalHandlerCapReached: at cap 1, a second trigger while the first blocks
// returns 409 eval_cap_reached.
func TestEvalHandlerCapReached(t *testing.T) {
	base := t.TempDir()
	runsDir := filepath.Join(base, "runs")
	cfrDir := filepath.Join(base, "cfr")
	for _, d := range []string{runsDir, cfrDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	release := filepath.Join(base, "release")
	stubBody := "#!/bin/sh\nwhile [ ! -f '" + release + "' ]; do sleep 0.02; done\nexit 0\n"
	stub := filepath.Join(base, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(stubBody), 0o755); err != nil {
		t.Fatal(err)
	}
	mgr := NewEvalManager(runsDir, cfrDir, stub)
	mgr.SetMaxConcurrent(1)
	h := NewEvalHandlers(EvalHandlersConfig{Manager: mgr, RunsDir: runsDir})
	h.gpuQuery = fakeQuery("", exec.ErrNotFound)
	seedCheckpoint(t, runsDir, "cap-a", "snapshots", "prtcfr_checkpoint.pt")
	seedCheckpoint(t, runsDir, "cap-b", "snapshots", "prtcfr_checkpoint.pt")
	t.Cleanup(func() { _ = os.WriteFile(release, []byte("x"), 0o644) })

	if w := evalReq(h.HandleTrigger, http.MethodPost, "/training/runs/cap-a/eval", `{"device":"cpu"}`); w.Code != http.StatusAccepted {
		t.Fatalf("first eval: status = %d, want 202: %s", w.Code, w.Body.String())
	}
	w := evalReq(h.HandleTrigger, http.MethodPost, "/training/runs/cap-b/eval", `{"device":"cpu"}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("second eval: status = %d, want 409: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "eval_cap_reached" {
		t.Errorf("error = %v, want eval_cap_reached", body["error"])
	}
	if err := os.WriteFile(release, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	waitEvalStatus(t, mgr, "cap-a", EvalSucceeded, 10*time.Second)
}

func TestEvalHandlerList(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	seedCheckpoint(t, f.runsDir, "listrun", "snapshots", "prtcfr_checkpoint.pt")

	if w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/listrun/eval", `{"device":"cpu"}`); w.Code != http.StatusAccepted {
		t.Fatalf("trigger: status = %d, want 202: %s", w.Code, w.Body.String())
	}
	waitEvalStatus(t, f.mgr, "listrun", EvalSucceeded, 10*time.Second)

	w := evalReq(f.h.HandleList, http.MethodGet, "/training/runs/listrun/eval", "")
	if w.Code != http.StatusOK {
		t.Fatalf("list status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Jobs []*EvalJob `json:"jobs"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Jobs) != 1 {
		t.Fatalf("jobs len = %d, want 1", len(resp.Jobs))
	}
	if resp.Jobs[0].Status != EvalSucceeded {
		t.Errorf("job status = %q, want succeeded", resp.Jobs[0].Status)
	}
	if len(resp.Jobs[0].Tail) == 0 {
		t.Error("list job missing log tail")
	}
}

// TestEvalHandlerNameGuard: a run name carrying ".." is rejected before any
// filesystem access (path-traversal guard, load-bearing).
func TestEvalHandlerNameGuard(t *testing.T) {
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	for _, name := range []string{"e..vil", "..", "foo..bar"} {
		w := evalReq(f.h.HandleTrigger, http.MethodPost, "/training/runs/"+name+"/eval", `{"device":"cpu"}`)
		if w.Code != http.StatusBadRequest {
			t.Errorf("name %q: status = %d, want 400", name, w.Code)
		}
	}
}

// TestEvalHandlerAuthGate confirms RequireAuth rejects an unauthenticated eval
// trigger and lets an authenticated one reach the handler (which then 404s for
// the missing checkpoint, proving the handler body ran past auth).
func TestEvalHandlerAuthGate(t *testing.T) {
	auth.Init()
	f := newEvalHandlerFixture(t, evalArgsEchoStub)
	logger := logrus.New()
	logger.SetOutput(io.Discard)

	wrapped := middleware.LogMiddleware(logger)(middleware.RequireAuth(http.HandlerFunc(f.h.HandleTrigger)))

	req := httptest.NewRequest(http.MethodPost, "/training/runs/x/eval", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("no cookie: status = %d, want 401", w.Code)
	}

	token, err := auth.CreateJWT("test-user")
	if err != nil {
		t.Fatal(err)
	}
	req2 := httptest.NewRequest(http.MethodPost, "/training/runs/x/eval", nil)
	req2.Header.Set("Cookie", "auth_token="+token)
	w2 := httptest.NewRecorder()
	wrapped.ServeHTTP(w2, req2)
	if w2.Code == http.StatusUnauthorized {
		t.Fatalf("valid cookie still 401; auth gate rejected a valid token")
	}
	if w2.Code != http.StatusNotFound {
		t.Fatalf("valid cookie: status = %d, want 404 (handler ran, no checkpoint): %s", w2.Code, w2.Body.String())
	}
}
