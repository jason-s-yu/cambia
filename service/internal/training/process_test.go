package training

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// longRunningStub traps SIGINT and exits 0 gracefully; otherwise it stays alive
// via a backgrounded sleep. It ignores the train/prtcfr CLI args the manager
// passes.
const longRunningStub = "#!/bin/sh\ntrap 'exit 0' INT\nsleep 30 &\nwait\n"

// crashStub exits nonzero immediately, simulating a training crash.
const crashStub = "#!/bin/sh\nexit 7\n"

// newTestManager builds a ProcessManager whose cambiaBin is a stub script with
// the given body. It returns the manager and the runs root. A cleanup kills any
// surviving process groups.
func newTestManager(t *testing.T, stubBody string) (*ProcessManager, string) {
	t.Helper()
	base := t.TempDir()

	runsDir := filepath.Join(base, "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfrDir := filepath.Join(base, "cfr")
	if err := os.MkdirAll(cfrDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stub := filepath.Join(base, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(stubBody), 0o755); err != nil {
		t.Fatal(err)
	}

	m := NewProcessManager(runsDir, cfrDir, stub, nil)
	t.Cleanup(func() {
		m.mu.Lock()
		for _, p := range m.procs {
			_ = syscall.Kill(-p.pgid, syscall.SIGKILL)
		}
		m.mu.Unlock()
	})
	return m, runsDir
}

// createRun writes a throwaway config and calls Create.
func createRun(t *testing.T, m *ProcessManager, name, algo string) {
	t.Helper()
	cfg := filepath.Join(t.TempDir(), "src-config.yaml")
	if err := os.WriteFile(cfg, []byte("prt_cfr:\n  iterations: 2\n  device: cpu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create(CreateRequest{Name: name, Algorithm: algo, ConfigPath: cfg}); err != nil {
		t.Fatalf("create %q: %v", name, err)
	}
}

// waitForStatus polls process.json until status reaches want or the timeout.
func waitForStatus(t *testing.T, m *ProcessManager, name, want string, timeout time.Duration) *ProcessState {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if st, ok := m.GetState(name); ok && st.Status == want {
			return st
		}
		time.Sleep(20 * time.Millisecond)
	}
	st, _ := m.GetState(name)
	t.Fatalf("run %q did not reach status %q within %s (last: %+v)", name, want, timeout, st)
	return nil
}

func TestProcessCreate(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	st, err := func() (*ProcessState, error) {
		cfg := filepath.Join(t.TempDir(), "src-config.yaml")
		if err := os.WriteFile(cfg, []byte("prt_cfr:\n  iterations: 2\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return m.Create(CreateRequest{Name: "run-create", Algorithm: "prt-cfr", ConfigPath: cfg})
	}()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if st.Status != StatusCreated {
		t.Errorf("status = %q, want created", st.Status)
	}
	if st.Algorithm != "prt-cfr" {
		t.Errorf("algorithm = %q, want prt-cfr", st.Algorithm)
	}

	runDir := filepath.Join(runsDir, "run-create")
	if _, err := os.Stat(filepath.Join(runDir, "logs")); err != nil {
		t.Errorf("logs dir not created: %v", err)
	}
	if _, err := os.Stat(filepath.Join(runDir, "process.json")); err != nil {
		t.Errorf("process.json not written: %v", err)
	}
	cfgData, err := os.ReadFile(filepath.Join(runDir, "config.yaml"))
	if err != nil {
		t.Fatalf("config.yaml not materialized: %v", err)
	}
	if len(cfgData) == 0 {
		t.Error("config.yaml is empty")
	}
	if st.ConfigPath != filepath.Join(runDir, "config.yaml") {
		// ConfigPath is absolute; compare to the absolute run-dir config.
		abs, _ := filepath.Abs(filepath.Join(runDir, "config.yaml"))
		if st.ConfigPath != abs {
			t.Errorf("config_path = %q, want %q", st.ConfigPath, abs)
		}
	}
}

func TestProcessCreateNameValidation(t *testing.T) {
	m, _ := newTestManager(t, crashStub)

	bad := []string{"../evil", "a/b", "..", "/etc/passwd", ".hidden", "", "a b"}
	for _, name := range bad {
		if _, err := m.Create(CreateRequest{Name: name, Algorithm: "prt-cfr"}); !errors.Is(err, ErrInvalidName) {
			t.Errorf("Create(%q): err = %v, want ErrInvalidName", name, err)
		}
	}

	// A valid name must be accepted.
	if _, err := m.Create(CreateRequest{Name: "good-name_1.2", Algorithm: "prt-cfr"}); err != nil {
		t.Errorf("Create(valid): unexpected err %v", err)
	}
}

func TestProcessCreateCollision(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	createRun(t, m, "dup", "prt-cfr")

	if _, err := m.Create(CreateRequest{Name: "dup", Algorithm: "prt-cfr"}); !errors.Is(err, ErrNameCollision) {
		t.Errorf("second Create: err = %v, want ErrNameCollision", err)
	}
}

func TestProcessStartStopGraceful(t *testing.T) {
	m, _ := newTestManager(t, longRunningStub)
	createRun(t, m, "run-graceful", "prt-cfr")

	st, err := m.Start("run-graceful", StartOpts{})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if st.Status != StatusRunning {
		t.Fatalf("status = %q, want running", st.Status)
	}
	if st.PID <= 0 || st.PGID <= 0 {
		t.Fatalf("pid=%d pgid=%d, want both > 0", st.PID, st.PGID)
	}

	// Let the stub install its SIGINT trap.
	time.Sleep(150 * time.Millisecond)

	stopped, err := m.Stop("run-graceful", false)
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if stopped.Status != StatusStopping && stopped.Status != StatusStopped {
		t.Errorf("post-stop status = %q, want stopping or stopped", stopped.Status)
	}

	final := waitForStatus(t, m, "run-graceful", StatusStopped, 10*time.Second)
	if final.ExitCode == nil {
		t.Error("exit_code not recorded on stop")
	}
	if final.FinishedAt == "" {
		t.Error("finished_at not recorded on stop")
	}
}

func TestProcessStopForce(t *testing.T) {
	m, _ := newTestManager(t, longRunningStub)
	createRun(t, m, "run-force", "prt-cfr")

	if _, err := m.Start("run-force", StartOpts{}); err != nil {
		t.Fatalf("Start: %v", err)
	}
	time.Sleep(100 * time.Millisecond)

	if _, err := m.Stop("run-force", true); err != nil {
		t.Fatalf("Stop(force): %v", err)
	}
	waitForStatus(t, m, "run-force", StatusStopped, 10*time.Second)
}

func TestProcessCrashOnNonzeroExit(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	createRun(t, m, "run-crash", "prt-cfr")

	if _, err := m.Start("run-crash", StartOpts{}); err != nil {
		t.Fatalf("Start: %v", err)
	}

	final := waitForStatus(t, m, "run-crash", StatusCrashed, 10*time.Second)
	if final.ExitCode == nil || *final.ExitCode != 7 {
		t.Errorf("exit_code = %v, want 7", final.ExitCode)
	}
	if final.LastError == "" {
		t.Error("last_error not set on crash")
	}
}

func TestProcessReconcileStaleRunning(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	name := "stale-run"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Fake a running run whose pid is dead.
	if err := writeProcessState(runDir, &ProcessState{
		Name:      name,
		Status:    StatusRunning,
		Algorithm: "prt-cfr",
		PID:       9999999,
		PGID:      9999999,
		CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	m.Reconcile()

	st, ok := m.GetState(name)
	if !ok {
		t.Fatal("state missing after reconcile")
	}
	if st.Status != StatusCrashed {
		t.Errorf("status = %q, want crashed", st.Status)
	}
	if st.LastError == "" {
		t.Error("last_error not set by reconcile")
	}
}

func TestProcessReconcileLeavesTerminalAndLiveAlone(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	// A stopped run must not be touched.
	stoppedDir := filepath.Join(runsDir, "already-stopped")
	if err := os.MkdirAll(stoppedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	code := 0
	if err := writeProcessState(stoppedDir, &ProcessState{
		Name: "already-stopped", Status: StatusStopped, PID: 9999999,
		ExitCode: &code, CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	// A live run (our own pid) must stay running.
	liveDir := filepath.Join(runsDir, "live-run")
	if err := os.MkdirAll(liveDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeProcessState(liveDir, &ProcessState{
		Name: "live-run", Status: StatusRunning, PID: os.Getpid(),
		PGID: os.Getpid(), CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	m.Reconcile()

	if st, _ := m.GetState("already-stopped"); st.Status != StatusStopped {
		t.Errorf("stopped run changed to %q", st.Status)
	}
	if st, _ := m.GetState("live-run"); st.Status != StatusRunning {
		t.Errorf("live run changed to %q", st.Status)
	}
}

func TestProcessStartUnsupportedAlgorithm(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	createRun(t, m, "run-badalgo", "es-mccfr")

	if _, err := m.Start("run-badalgo", StartOpts{}); !errors.Is(err, ErrUnsupportedAlgorithm) {
		t.Errorf("Start: err = %v, want ErrUnsupportedAlgorithm", err)
	}
}

func TestProcessStartMissingConfig(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	// Write a created state with no config.yaml on disk.
	name := "no-config"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeProcessState(runDir, &ProcessState{
		Name: name, Status: StatusCreated, Algorithm: "prt-cfr",
		ConfigPath: filepath.Join(runDir, "config.yaml"), CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := m.Start(name, StartOpts{}); !errors.Is(err, ErrConfigMissing) {
		t.Errorf("Start: err = %v, want ErrConfigMissing", err)
	}
}

func TestProcessGetState(t *testing.T) {
	m, _ := newTestManager(t, crashStub)

	if _, ok := m.GetState("nope"); ok {
		t.Error("GetState(nope) ok = true, want false")
	}

	createRun(t, m, "exists", "prt-cfr")
	st, ok := m.GetState("exists")
	if !ok {
		t.Fatal("GetState(exists) ok = false")
	}
	if st.Status != StatusCreated {
		t.Errorf("status = %q, want created", st.Status)
	}
}

func TestProcessStartInvalidName(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	if _, err := m.Start("../evil", StartOpts{}); !errors.Is(err, ErrInvalidName) {
		t.Errorf("Start(../evil): err = %v, want ErrInvalidName", err)
	}
}
