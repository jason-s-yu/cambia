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
	"syscall"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/middleware"
	"github.com/sirupsen/logrus"
)

// cambiaStub emulates the cambia CLI for handler tests. `config render` writes a
// valid tiny config unless a --set value contains BADVALUE; `config validate`
// fails when the file contains INVALID; any other invocation (train prtcfr ...)
// exits 0 immediately.
const cambiaStub = `#!/bin/sh
if [ "$1" = "config" ] && [ "$2" = "render" ]; then
  out=""
  shift 2
  while [ $# -gt 0 ]; do
    case "$1" in
      -o|--output) out="$2"; shift 2 ;;
      --set)
        case "$2" in *BADVALUE*) echo "bad override" >&2; exit 1 ;; esac
        shift 2 ;;
      *) shift ;;
    esac
  done
  printf 'prt_cfr:\n  iterations: 2\n  device: cpu\n' > "$out"
  exit 0
fi
if [ "$1" = "config" ] && [ "$2" = "validate" ]; then
  if grep -q INVALID "$3"; then echo "invalid config" >&2; exit 1; fi
  exit 0
fi
exit 0
`

// handlerFixture wires a ProcessHandlers against a fixture store, a stub CLI,
// and a template dir. gpuQuery reports a CPU host so start preflight passes.
type handlerFixture struct {
	ph          *ProcessHandlers
	store       *TrainingStore
	mgr         *ProcessManager
	runsDir     string
	templateDir string
}

func newHandlerFixture(t *testing.T) *handlerFixture {
	t.Helper()
	store, runsDir := setupTestDB(t)
	cfrDir := t.TempDir()

	templateDir := t.TempDir()
	for _, n := range []string{"prtcfr_production.yaml", "prtcfr_smoke.yaml", "deep_train.yaml"} {
		if err := os.WriteFile(filepath.Join(templateDir, n), []byte("prt_cfr: {}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(templateDir, "prtcfr_subdir.yaml"), 0o755); err != nil {
		t.Fatal(err)
	}

	stub := filepath.Join(cfrDir, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(cambiaStub), 0o755); err != nil {
		t.Fatal(err)
	}

	mgr := NewProcessManager(runsDir, cfrDir, stub, store)
	ph := NewProcessHandlers(ProcessHandlersConfig{
		Manager:       mgr,
		Store:         store,
		CambiaBin:     stub,
		CFRDir:        cfrDir,
		RunsDir:       runsDir,
		TemplateDir:   templateDir,
		MaxConcurrent: 10,
	})
	ph.gpuQuery = fakeQuery("", exec.ErrNotFound) // CPU host

	t.Cleanup(func() {
		mgr.mu.Lock()
		for _, p := range mgr.procs {
			_ = syscall.Kill(-p.pgid, syscall.SIGKILL)
		}
		mgr.mu.Unlock()
	})
	return &handlerFixture{ph: ph, store: store, mgr: mgr, runsDir: runsDir, templateDir: templateDir}
}

func doReq(h http.HandlerFunc, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h(w, req)
	return w
}

type createResp struct {
	Run     *RunDetail    `json:"run"`
	Process *ProcessState `json:"process"`
}

func TestHandleCreateSuccess(t *testing.T) {
	f := newHandlerFixture(t)

	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
		`{"name":"new-run","template":"prtcfr_production.yaml","overrides":{"prt_cfr.iterations":5}}`)
	if w.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201: %s", w.Code, w.Body.String())
	}
	var resp createResp
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Process == nil || resp.Process.Status != StatusCreated {
		t.Errorf("process = %+v, want status created", resp.Process)
	}
	if resp.Run == nil || resp.Run.Name != "new-run" {
		t.Errorf("run = %+v, want name new-run", resp.Run)
	}
	// process.json and config.yaml materialized on disk.
	runDir := filepath.Join(f.runsDir, "new-run")
	if _, err := os.Stat(filepath.Join(runDir, "process.json")); err != nil {
		t.Errorf("process.json missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(runDir, "config.yaml")); err != nil {
		t.Errorf("config.yaml missing: %v", err)
	}
}

func TestHandleCreateVerbatimYAML(t *testing.T) {
	f := newHandlerFixture(t)
	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
		`{"name":"yaml-run","yaml":"prt_cfr:\n  iterations: 2\n"}`)
	if w.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201: %s", w.Code, w.Body.String())
	}
}

func TestHandleCreateInvalidName(t *testing.T) {
	f := newHandlerFixture(t)
	before := dirNames(t, f.runsDir)
	for _, name := range []string{"../evil", "a/b", "..", "/etc/passwd", "bad name"} {
		w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
			`{"name":"`+name+`","template":"prtcfr_production.yaml"}`)
		if w.Code != http.StatusBadRequest {
			t.Errorf("name %q: status = %d, want 400", name, w.Code)
		}
	}
	// A rejected name must never create a new entry under runs/ (path-traversal
	// guard is load-bearing).
	if after := dirNames(t, f.runsDir); len(after) != len(before) {
		t.Errorf("runs dir gained entries after invalid-name creates: before=%v after=%v", before, after)
	}
}

// dirNames returns the sorted directory-entry names under dir.
func dirNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

func TestHandleCreateCollision(t *testing.T) {
	f := newHandlerFixture(t)
	body := `{"name":"dup","template":"prtcfr_production.yaml"}`
	if w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs", body); w.Code != http.StatusCreated {
		t.Fatalf("first create: status = %d, want 201: %s", w.Code, w.Body.String())
	}
	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs", body)
	if w.Code != http.StatusConflict {
		t.Fatalf("second create: status = %d, want 409", w.Code)
	}
	var body2 map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body2)
	if body2["error"] != "collision" {
		t.Errorf("error = %v, want collision", body2["error"])
	}
}

func TestHandleCreateInvalidConfigRender(t *testing.T) {
	f := newHandlerFixture(t)
	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
		`{"name":"bad-render","template":"prtcfr_production.yaml","overrides":{"k":"BADVALUE"}}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", w.Code, w.Body.String())
	}
	// A failed render must not leave a created run.
	if _, ok := f.mgr.GetState("bad-render"); ok {
		t.Error("run created despite render failure")
	}
}

func TestHandleCreateInvalidConfigValidate(t *testing.T) {
	f := newHandlerFixture(t)
	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
		`{"name":"bad-yaml","yaml":"INVALID: [broken"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", w.Code, w.Body.String())
	}
}

// TestNewProcessHandlersDefaultsPreflightOn confirms the safety rails default
// on: a config that leaves MinVRAMGB/MinDiskGB unset (the zero value) must not
// silently disable the GPU/disk preflight checks (F1).
func TestNewProcessHandlersDefaultsPreflightOn(t *testing.T) {
	h := NewProcessHandlers(ProcessHandlersConfig{})
	if h.minVRAMGB != DefaultMinVRAMGB {
		t.Errorf("minVRAMGB = %v, want default %v", h.minVRAMGB, DefaultMinVRAMGB)
	}
	if h.minDiskGB != DefaultMinDiskGB {
		t.Errorf("minDiskGB = %v, want default %v", h.minDiskGB, DefaultMinDiskGB)
	}
}

// TestNewProcessHandlersRespectsExplicitConfig confirms an explicit positive
// config value is not clobbered by the default.
func TestNewProcessHandlersRespectsExplicitConfig(t *testing.T) {
	h := NewProcessHandlers(ProcessHandlersConfig{MinVRAMGB: 8, MinDiskGB: 20})
	if h.minVRAMGB != 8 {
		t.Errorf("minVRAMGB = %v, want 8 (explicit config must win)", h.minVRAMGB)
	}
	if h.minDiskGB != 20 {
		t.Errorf("minDiskGB = %v, want 20 (explicit config must win)", h.minDiskGB)
	}
}

// TestHandleStartRequestNonPositiveOverrideFallsBackNotDisable confirms a
// request-level min_free_disk_gb of 0 or negative falls back to the
// configured default rather than disabling the check (F1). The fixture is
// pinned to an impossible configured default so the only way this could pass
// is if the non-positive override wrongly disabled the check.
func TestHandleStartRequestNonPositiveOverrideFallsBackNotDisable(t *testing.T) {
	f := newHandlerFixture(t)
	f.ph.minDiskGB = 1e9 // configured default: intentionally impossible
	mustCreate(t, f, "zero-override-blocked")

	for _, v := range []string{"0", "-5"} {
		w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/zero-override-blocked/start",
			`{"min_free_disk_gb":`+v+`}`)
		if w.Code != http.StatusConflict {
			t.Errorf("min_free_disk_gb=%s: status = %d, want 409 (must fall back to configured default, not disable)", v, w.Code)
		}
	}
}

func TestHandleTemplates(t *testing.T) {
	f := newHandlerFixture(t)
	w := doReq(f.ph.HandleTemplates, http.MethodGet, "/training/config/templates", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var names []string
	if err := json.Unmarshal(w.Body.Bytes(), &names); err != nil {
		t.Fatal(err)
	}
	// Only prtcfr*.yaml files, sorted; deep_train.yaml and the subdir excluded.
	want := []string{"prtcfr_production.yaml", "prtcfr_smoke.yaml"}
	if len(names) != len(want) || names[0] != want[0] || names[1] != want[1] {
		t.Errorf("templates = %v, want %v", names, want)
	}
}

func TestHandleStartSuccess(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "start-ok")

	w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/start-ok/start", `{}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
	waitTerminal(t, f.mgr, "start-ok")
}

func TestHandleStartPreflightBlock(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "start-block")

	w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/start-block/start",
		`{"min_free_disk_gb":1000000000}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "preflight_failed" {
		t.Errorf("error = %v, want preflight_failed", body["error"])
	}
	if body["override"] != "force" {
		t.Errorf("override = %v, want force", body["override"])
	}
}

func TestHandleStartForceOverride(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "start-force")

	w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/start-force/start",
		`{"force":true,"min_free_disk_gb":1000000000}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202 (disk is overridable): %s", w.Code, w.Body.String())
	}
	waitTerminal(t, f.mgr, "start-force")
}

func TestHandleStartNotFound(t *testing.T) {
	f := newHandlerFixture(t)
	w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/ghost/start", `{}`)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
}

func TestHandleStop(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "stop-me")
	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/stop-me/stop", `{}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
}

func TestHandleResumeNoCheckpoint(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "resume-none")
	w := doReq(f.ph.HandleResume, http.MethodPost, "/training/runs/resume-none/resume", `{}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "no_resumable_checkpoint" {
		t.Errorf("error = %v, want no_resumable_checkpoint", body["error"])
	}
}

func TestHandleResumeWithCheckpoint(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "resume-ok")

	// Real PRT-CFR resume artifacts: a rolling checkpoint under snapshots/ plus
	// resume_state.json at the run-dir root (see prtcfr_trainer.py:345,1106,1178).
	snapDir := filepath.Join(f.runsDir, "resume-ok", "snapshots")
	if err := os.MkdirAll(snapDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapDir, "prtcfr_checkpoint.pt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.runsDir, "resume-ok", "resume_state.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	w := doReq(f.ph.HandleResume, http.MethodPost, "/training/runs/resume-ok/resume", `{}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
	waitTerminal(t, f.mgr, "resume-ok")
}

// TestHandleResumeCheckpointsDirOnly is a regression test for the old (wrong)
// resume gate, which checked runs/<name>/checkpoints/ for any file. The
// PRT-CFR trainer never creates that directory: it writes its rolling
// checkpoint to snapshots/prtcfr_checkpoint.pt and resume_state.json at the
// run-dir root. A checkpoints/ dir with a file in it (the legacy layout) must
// still 409, since the trainer's own resume path cannot use it.
func TestHandleResumeCheckpointsDirOnly(t *testing.T) {
	f := newHandlerFixture(t)
	mustCreate(t, f, "resume-legacy-dir")

	ckptDir := filepath.Join(f.runsDir, "resume-legacy-dir", "checkpoints")
	if err := os.MkdirAll(ckptDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ckptDir, "prtcfr_checkpoint_iter_2.pt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	w := doReq(f.ph.HandleResume, http.MethodPost, "/training/runs/resume-legacy-dir/resume", `{}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "no_resumable_checkpoint" {
		t.Errorf("error = %v, want no_resumable_checkpoint", body["error"])
	}
}

// TestHandlersAuthGate confirms the wrapped mux rejects unauthenticated requests
// and passes authenticated ones through to the handler.
func TestHandlersAuthGate(t *testing.T) {
	auth.Init()
	f := newHandlerFixture(t)
	logger := logrus.New()
	logger.SetOutput(io.Discard)

	wrapped := middleware.LogMiddleware(logger)(middleware.RequireAuth(http.HandlerFunc(f.ph.HandleTemplates)))

	// No cookie -> 401, handler never runs.
	req := httptest.NewRequest(http.MethodGet, "/training/config/templates", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("no cookie: status = %d, want 401", w.Code)
	}

	// Valid cookie -> 200.
	token, err := auth.CreateJWT("test-user")
	if err != nil {
		t.Fatal(err)
	}
	req2 := httptest.NewRequest(http.MethodGet, "/training/config/templates", nil)
	req2.Header.Set("Cookie", "auth_token="+token)
	w2 := httptest.NewRecorder()
	wrapped.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("valid cookie: status = %d, want 200: %s", w2.Code, w2.Body.String())
	}
}

// waitTerminal polls until the run reaches a terminal state. The stub exits
// immediately, so a spawned run settles fast; waiting guarantees the child is
// reaped (and done writing its log) before t.TempDir cleanup runs RemoveAll,
// which would otherwise race the still-exiting process.
func waitTerminal(t *testing.T, mgr *ProcessManager, name string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if st, ok := mgr.GetState(name); ok {
			switch st.Status {
			case StatusStopped, StatusCrashed:
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("run %q did not reach a terminal state", name)
}

// mustCreate creates a run through the handler and fails the test on error.
func mustCreate(t *testing.T, f *handlerFixture, name string) {
	t.Helper()
	w := doReq(f.ph.HandleCreate, http.MethodPost, "/training/runs",
		`{"name":"`+name+`","template":"prtcfr_production.yaml"}`)
	if w.Code != http.StatusCreated {
		t.Fatalf("create %q: status = %d, want 201: %s", name, w.Code, w.Body.String())
	}
}
