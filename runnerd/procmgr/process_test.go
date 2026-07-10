package procmgr

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
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

	m := NewProcessManager(runsDir, cfrDir, stub, nil, TrainAlgorithms())
	t.Cleanup(func() { m.KillAll() })
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
	if err := WriteProcessState(runDir, &ProcessState{
		Name:      name,
		Status:    StatusRunning,
		Algorithm: "prt-cfr",
		PID:       9999999,
		PGID:      9999999,
		CreatedAt: NowRFC3339(),
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
	if err := WriteProcessState(stoppedDir, &ProcessState{
		Name: "already-stopped", Status: StatusStopped, PID: 9999999,
		ExitCode: &code, CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	// A live run (our own pid) must stay running.
	liveDir := filepath.Join(runsDir, "live-run")
	if err := os.MkdirAll(liveDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(liveDir, &ProcessState{
		Name: "live-run", Status: StatusRunning, PID: os.Getpid(),
		PGID: os.Getpid(), CreatedAt: NowRFC3339(),
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

// TestProcessReconcileSkipsRemoteRow is the cross-host pid-reuse guard for
// startup reconciliation: a remote row (Host set) recorded running with a dead
// local pid must be left untouched. Its liveness is the runner's authority,
// refreshed by the pull loop; a local pidAlive probe against a foreign pid would
// wrongly flip the row to crashed.
func TestProcessReconcileSkipsRemoteRow(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	name := "v0.4-prtcfr-remote"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Host: "runner1", Status: StatusRunning, Algorithm: "prt-cfr",
		PID: 9999999, PGID: 9999999, CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	m.Reconcile()

	st, ok := m.GetState(name)
	if !ok {
		t.Fatal("state missing after reconcile")
	}
	if st.Status != StatusRunning {
		t.Errorf("remote row status = %q, want running (reconcile must skip remote rows)", st.Status)
	}
	if st.LastError != "" {
		t.Errorf("remote row last_error = %q, want empty (never reconciled to crashed)", st.LastError)
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
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Status: StatusCreated, Algorithm: "prt-cfr",
		ConfigPath: filepath.Join(runDir, "config.yaml"), CreatedAt: NowRFC3339(),
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

// TestProcessLaunchConcurrencyCap is the hard-backstop regression for MED-2:
// two goroutines racing Start on different created runs with cap 1 must not
// both succeed. The check-then-launch happens atomically under m.mu, closing
// the TOCTOU window a disk-scanning preflight check alone cannot.
func TestProcessLaunchConcurrencyCap(t *testing.T) {
	m, _ := newTestManager(t, longRunningStub)
	m.SetMaxConcurrent(1)
	createRun(t, m, "cap-a", "prt-cfr")
	createRun(t, m, "cap-b", "prt-cfr")

	names := []string{"cap-a", "cap-b"}
	results := make([]error, len(names))
	var wg sync.WaitGroup
	wg.Add(len(names))
	for i, n := range names {
		i, n := i, n
		go func() {
			defer wg.Done()
			_, err := m.Start(n, StartOpts{})
			results[i] = err
		}()
	}
	wg.Wait()

	successes := 0
	capBlocked := 0
	for _, err := range results {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, ErrConcurrencyCapReached):
			capBlocked++
		}
	}
	if successes != 1 {
		t.Fatalf("successes = %d, want 1 (results=%v)", successes, results)
	}
	if capBlocked != 1 {
		t.Fatalf("cap-blocked = %d, want 1 (results=%v)", capBlocked, results)
	}

	// Clean up whichever run actually launched.
	for _, n := range names {
		if st, ok := m.GetState(n); ok && st.Status == StatusRunning {
			_, _ = m.Stop(n, true)
			waitForStatus(t, m, n, StatusStopped, 10*time.Second)
		}
	}
}

// TestProcessLaunchConcurrencyCapDisabled confirms max <= 0 disables the
// manager-level backstop (the zero value from an unconfigured manager).
func TestProcessLaunchConcurrencyCapDisabled(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	createRun(t, m, "nocap-a", "prt-cfr")
	createRun(t, m, "nocap-b", "prt-cfr")

	if _, err := m.Start("nocap-a", StartOpts{}); err != nil {
		t.Fatalf("Start(nocap-a): %v", err)
	}
	if _, err := m.Start("nocap-b", StartOpts{}); err != nil {
		t.Fatalf("Start(nocap-b): %v", err)
	}
	// Wait for both to be reaped before tempdir cleanup races the still-exiting
	// child (crashStub exits immediately, but log-fd close is asynchronous).
	waitForStatus(t, m, "nocap-a", StatusCrashed, 10*time.Second)
	waitForStatus(t, m, "nocap-b", StatusCrashed, 10*time.Second)
}

// TestProcessReconcilePidReuseFalseAlive is the MED-3 regression: a process.json
// recording a pid that is alive (passes the bare signal-0 probe) but whose
// StartTicks does not match the pid's actual /proc/<pid>/stat starttime must be
// reconciled to crashed, not left running. This simulates a pid the kernel
// recycled for an unrelated process since this run's process exited, using our
// own test-binary pid (guaranteed alive) with a deliberately wrong StartTicks.
func TestProcessReconcilePidReuseFalseAlive(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	name := "reused-pid"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Status: StatusRunning, Algorithm: "prt-cfr",
		PID: os.Getpid(), PGID: os.Getpid(), StartTicks: 1,
		CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	m.Reconcile()

	st, ok := m.GetState(name)
	if !ok {
		t.Fatal("state missing after reconcile")
	}
	if st.Status != StatusCrashed {
		t.Errorf("status = %q, want crashed (starttime mismatch must not read as alive)", st.Status)
	}
}

// TestProcessReconcileRealStarttimeMatches is the positive-path complement:
// Reconcile must NOT flag a live run whose recorded StartTicks actually
// matches the pid's current /proc/<pid>/stat starttime.
func TestProcessReconcileRealStarttimeMatches(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	ticks, err := readProcStarttime(os.Getpid())
	if err != nil {
		t.Skipf("cannot read /proc/%d/stat on this host: %v", os.Getpid(), err)
	}
	boot, _ := readBootTime()

	name := "real-starttime"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Status: StatusRunning, Algorithm: "prt-cfr",
		PID: os.Getpid(), PGID: os.Getpid(), StartTicks: ticks, BootID: boot,
		CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	m.Reconcile()

	st, ok := m.GetState(name)
	if !ok {
		t.Fatal("state missing after reconcile")
	}
	if st.Status != StatusRunning {
		t.Errorf("status = %q, want running (starttime matches, must not be reconciled away)", st.Status)
	}
}

// TestProcessStopRefusesSignalOnStarttimeMismatch is the MED-3 regression for
// Stop's untracked-run path: it must not signal a pid it can no longer
// positively verify by starttime. killGroupFunc is swapped for a spy so the
// test never sends a real signal to the test binary's own process group.
func TestProcessStopRefusesSignalOnStarttimeMismatch(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	name := "stale-signal"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Status: StatusRunning, Algorithm: "prt-cfr",
		PID: os.Getpid(), PGID: os.Getpid(), StartTicks: 1,
		CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	orig := killGroupFunc
	signaled := false
	killGroupFunc = func(pgid int, sig syscall.Signal) error {
		signaled = true
		return nil
	}
	t.Cleanup(func() { killGroupFunc = orig })

	// Not tracked by this manager instance (never Start()ed here), so Stop
	// takes the disk-state fallback path that reads pid/starttime off process.json.
	if _, err := m.Stop(name, false); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if signaled {
		t.Error("Stop signaled a pid it could not verify by starttime (pid-reuse guard failed)")
	}
}

// TestProcessStopSignalsOnStarttimeMatch is the positive-path complement: Stop
// must still signal a pid whose recorded starttime matches.
func TestProcessStopSignalsOnStarttimeMatch(t *testing.T) {
	m, runsDir := newTestManager(t, crashStub)

	ticks, err := readProcStarttime(os.Getpid())
	if err != nil {
		t.Skipf("cannot read /proc/%d/stat on this host: %v", os.Getpid(), err)
	}
	boot, _ := readBootTime()

	name := "verified-signal"
	runDir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(runDir, &ProcessState{
		Name: name, Status: StatusRunning, Algorithm: "prt-cfr",
		PID: os.Getpid(), PGID: os.Getpid(), StartTicks: ticks, BootID: boot,
		CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}

	orig := killGroupFunc
	signaled := false
	killGroupFunc = func(pgid int, sig syscall.Signal) error {
		signaled = true
		return nil
	}
	t.Cleanup(func() { killGroupFunc = orig })

	if _, err := m.Stop(name, false); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if !signaled {
		t.Error("Stop did not signal a pid whose starttime matches")
	}
}

func TestProcessStartInvalidName(t *testing.T) {
	m, _ := newTestManager(t, crashStub)
	if _, err := m.Start("../evil", StartOpts{}); !errors.Is(err, ErrInvalidName) {
		t.Errorf("Start(../evil): err = %v, want ErrInvalidName", err)
	}
}
