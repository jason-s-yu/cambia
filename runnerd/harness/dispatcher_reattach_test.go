package harness

import (
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	_ "modernc.org/sqlite"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// reattachedChild is a live process standing in for a job forked by a previous
// daemon incarnation: process.json on disk names it, but this daemon never
// launched it, so procmgr has no managedProc and can never waitpid it.
type reattachedChild struct {
	cmd    *exec.Cmd
	killed bool
}

// kill terminates the stand-in and REAPS it. Reaping is load-bearing: a killed
// but unreaped child stays a zombie, /proc/<pid> still answers, and
// procmgr.EffectiveStatus would keep reporting it running forever.
func (c *reattachedChild) kill(t *testing.T) {
	t.Helper()
	if c.killed {
		return
	}
	_ = c.cmd.Process.Kill()
	_, _ = c.cmd.Process.Wait()
	c.killed = true
}

// reap waits out a stand-in that something ELSE signalled (the daemon's own
// stop path), for the same zombie reason: this test process forked it, so only
// this test process can reap it.
func (c *reattachedChild) reap(t *testing.T) {
	t.Helper()
	if c.killed {
		return
	}
	_, _ = c.cmd.Process.Wait()
	c.killed = true
}

// startReattachedChild spawns a long-lived child in its own process group and
// writes the runs/<name>/ pair a live job leaves behind: a process.json row at
// `running` carrying the pid/pgid/start_ticks/boot_id identity triple (so
// liveness is starttime-validated, not a bare pid probe), plus the jobspec.json
// the dispatcher reads back at Reconcile.
func startReattachedChild(t *testing.T, runsDir, name, kind string, exclusive bool) *reattachedChild {
	t.Helper()
	cmd := exec.Command("sleep", "60")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // own group: a force-stop cannot reach the test process
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	child := &reattachedChild{cmd: cmd}
	t.Cleanup(func() { child.kill(t) })

	pid := cmd.Process.Pid
	ticks, err := procStarttime(pid)
	if err != nil {
		t.Skipf("cannot read /proc/%d/stat on this host: %v", pid, err)
	}
	st := &procmgr.ProcessState{
		Name:       name,
		Status:     procmgr.StatusRunning,
		Algorithm:  kind,
		PID:        pid,
		PGID:       pid,
		StartTicks: ticks,
		BootID:     bootTime(),
		CreatedAt:  procmgr.NowRFC3339(),
		StartedAt:  procmgr.NowRFC3339(),
	}
	if err := procmgr.WriteProcessState(filepath.Join(runsDir, name), st); err != nil {
		t.Fatal(err)
	}
	spec := &JobSpec{Name: name, Kind: kind, Config: "cfr/config/x.yaml", SubmitSeq: 1, Exclusive: exclusive}
	if err := writeJobSpec(filepath.Join(runsDir, name), spec); err != nil {
		t.Fatal(err)
	}
	return child
}

// procStarttime mirrors procmgr's unexported /proc/<pid>/stat field-22 read
// (the package-private helper is not reachable from this package). The comm
// field is parenthesized and may contain spaces, so parsing anchors on the
// last ')'.
func procStarttime(pid int) (int64, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, err
	}
	s := string(data)
	idx := strings.LastIndexByte(s, ')')
	if idx < 0 || idx+2 > len(s) {
		return 0, fmt.Errorf("unparseable /proc/%d/stat", pid)
	}
	fields := strings.Fields(s[idx+2:])
	const starttimeIdx = 22 - 3
	if len(fields) <= starttimeIdx {
		return 0, fmt.Errorf("too few fields in /proc/%d/stat", pid)
	}
	return strconv.ParseInt(fields[starttimeIdx], 10, 64)
}

// bootTime reads /proc/stat btime, or 0 when unavailable (pidAlive then skips
// the boot disambiguation, which is fine inside a single test run).
func bootTime() int64 {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if rest, ok := strings.CutPrefix(line, "btime "); ok {
			v, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 64)
			if err != nil {
				return 0
			}
			return v
		}
	}
	return 0
}

// writeRunDBStatus creates runs/<name>/run_db.sqlite with a single `runs` row
// at the given status, the shape the trainer's journal leaves behind (design
// 4.2): a clean end is `completed`, an abandoned one `interrupted`.
func writeRunDBStatus(t *testing.T, runsDir, name, status string) {
	t.Helper()
	dbPath := filepath.Join(runsDir, name, "run_db.sqlite")
	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE runs (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		status TEXT NOT NULL DEFAULT 'created',
		created_at TEXT NOT NULL,
		updated_at TEXT NOT NULL
	)`); err != nil {
		t.Fatal(err)
	}
	now := procmgr.NowRFC3339()
	if _, err := db.Exec(`INSERT INTO runs (name, status, created_at, updated_at) VALUES (?, ?, ?, ?)`,
		name, status, now, now); err != nil {
		t.Fatal(err)
	}
}

// waitForProcessState polls runs/<name>/process.json until pred holds.
func waitForProcessState(t *testing.T, runsDir, name string, timeout time.Duration, pred func(*procmgr.ProcessState) bool) *procmgr.ProcessState {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last *procmgr.ProcessState
	for time.Now().Before(deadline) {
		st, err := procmgr.ReadProcessState(filepath.Join(runsDir, name))
		if err == nil {
			last = st
			if pred(st) {
				return st
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("process.json for %s never satisfied the predicate within %s: %+v", name, timeout, last)
	return nil
}

// TestReattachedNonExclusiveHoldsSlotAndFinalizes is the cambia-723 regression.
// Before the fix a job the daemon reattached at Reconcile was invisible to
// admission (only exclusive reattachments were accounted), so an unrelated job
// launched straight into an already-occupied runner, and nothing ever watched
// the reattached process: its process.json stayed `running` forever and a
// dependent gated on it was stranded. This pins both halves with a real live
// process: it consumes the single slot while alive, and when it dies the
// watcher finalizes the row, frees the slot, and re-dispatches the dependent.
func TestReattachedNonExclusiveHoldsSlotAndFinalizes(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted", "fake", false)

	r.disp.Reconcile()

	// Slot accounting: an unrelated job with no dependency stays queued while
	// the reattached job holds the only slot. This is the assertion that fails
	// on the pre-fix code (it would launch immediately, active having stayed 0).
	if _, err := r.disp.Submit(exclSpec("unrelated", "fake", false)); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("unrelated", 300*time.Millisecond)
	// Drop it so the freed slot below is contested by the dependent alone.
	if _, err := r.disp.Cancel("unrelated", false); err != nil {
		t.Fatal(err)
	}

	// A dependent gated on the reattached job: blocked while the parent lives.
	dep := exclSpec("dep-run", "fake", false)
	dep.After = []string{"adopted"}
	dep.OnFailure = OnFailureRun
	if _, err := r.disp.Submit(dep); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("dep-run", 200*time.Millisecond)

	child.kill(t)

	// The watcher observes the pid death, finalizes the row (no run_db here, so
	// the exit status is unknown -> crashed with no exit code), and frees the slot.
	st := waitForProcessState(t, r.runsDir, "adopted", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	if st.Status != procmgr.StatusCrashed {
		t.Fatalf("adopted status = %q, want crashed", st.Status)
	}
	if st.ExitCode != nil {
		t.Fatalf("adopted exit_code = %d, want none (unknowable for a reattached job)", *st.ExitCode)
	}
	if !strings.Contains(st.LastError, "exit status unknown (run_db status=absent)") {
		t.Fatalf("adopted last_error = %q, want the reattached-unknown-status wording", st.LastError)
	}
	if st.FinishedAt == "" {
		t.Fatal("adopted finished_at not stamped")
	}

	// Slot released and the dependent re-dispatched (on_failure=run admits it
	// despite the non-success parent).
	r.waitForState("dep-run", procmgr.StatusRunning, 3*time.Second)
}

// TestReattachedFinalizeInfersCleanExitFromRunDB pins the exit-status oracle:
// a reattached job whose own journal recorded `completed` exited cleanly, so it
// is finalized stopped with exit_code 0 and a dependent's SUCCESS gate fires
// (on_failure=skip, which would otherwise skip the dependent).
func TestReattachedFinalizeInfersCleanExitFromRunDB(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted-ok", "fake", false)
	writeRunDBStatus(t, r.runsDir, "adopted-ok", "completed")

	r.disp.Reconcile()

	dep := exclSpec("dep-skip-ok", "fake", false)
	dep.After = []string{"adopted-ok"}
	dep.OnFailure = OnFailureSkip
	if _, err := r.disp.Submit(dep); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("dep-skip-ok", 200*time.Millisecond)

	child.kill(t)

	st := waitForProcessState(t, r.runsDir, "adopted-ok", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	if st.Status != procmgr.StatusStopped {
		t.Fatalf("adopted-ok status = %q, want stopped", st.Status)
	}
	if st.ExitCode == nil || *st.ExitCode != 0 {
		t.Fatalf("adopted-ok exit_code = %v, want 0", st.ExitCode)
	}
	if !strings.Contains(st.LastError, "inferred from run_db status completed") {
		t.Fatalf("adopted-ok last_error = %q, want the run_db-inference wording", st.LastError)
	}

	// A clean parent means the success gate fires even under on_failure=skip.
	r.waitForState("dep-skip-ok", procmgr.StatusRunning, 3*time.Second)
}

// TestReattachedFinalizeWithoutRunDBSkipsDependent is the conservative half of
// the same oracle: with no journal to certify a clean exit the job is recorded
// crashed with NO fabricated exit code, so an on_failure=skip dependent is
// skipped rather than launched on an unverified success.
func TestReattachedFinalizeWithoutRunDBSkipsDependent(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted-bad", "fake", false)

	r.disp.Reconcile()

	dep := exclSpec("dep-skip-bad", "fake", false)
	dep.After = []string{"adopted-bad"}
	dep.OnFailure = OnFailureSkip
	if _, err := r.disp.Submit(dep); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("dep-skip-bad", 200*time.Millisecond)

	child.kill(t)

	st := waitForProcessState(t, r.runsDir, "adopted-bad", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	if st.Status != procmgr.StatusCrashed {
		t.Fatalf("adopted-bad status = %q, want crashed", st.Status)
	}
	if st.ExitCode != nil {
		t.Fatalf("adopted-bad exit_code = %d, want none", *st.ExitCode)
	}

	r.waitForState("dep-skip-bad", StateSkipped, 3*time.Second)
}

// TestReattachedFinalizeInterruptedRunDBCrashes covers the third journal state
// the stale sweep leaves behind: `interrupted` is not a clean end, so it lands
// crashed with the observed status quoted in last_error (an operator reading
// the row can tell "the journal said interrupted" from "there was no journal").
func TestReattachedFinalizeInterruptedRunDBCrashes(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted-int", "fake", false)
	writeRunDBStatus(t, r.runsDir, "adopted-int", "interrupted")

	r.disp.Reconcile()
	child.kill(t)

	st := waitForProcessState(t, r.runsDir, "adopted-int", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	if st.Status != procmgr.StatusCrashed {
		t.Fatalf("adopted-int status = %q, want crashed", st.Status)
	}
	if !strings.Contains(st.LastError, "run_db status=interrupted") {
		t.Fatalf("adopted-int last_error = %q, want the observed run_db status quoted", st.LastError)
	}
}

// TestRunDBRunStatus pins the journal reader directly: an exact name match, the
// single-row fallback a per-run-dir journal relies on when the trainer
// registered under a different name, and the empty-string verdict for an
// absent or unreadable file (which must never create one).
func TestRunDBRunStatus(t *testing.T) {
	runsDir := t.TempDir()

	// Absent journal -> "" and no file created as a side effect.
	if err := os.MkdirAll(filepath.Join(runsDir, "none"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := runDBRunStatus(filepath.Join(runsDir, "none"), "none"); got != "" {
		t.Fatalf("absent run_db status = %q, want \"\"", got)
	}
	if _, err := os.Stat(filepath.Join(runsDir, "none", "run_db.sqlite")); !os.IsNotExist(err) {
		t.Fatal("runDBRunStatus created run_db.sqlite on a read")
	}

	// Exact name match.
	if err := os.MkdirAll(filepath.Join(runsDir, "hit"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeRunDBStatus(t, runsDir, "hit", "completed")
	if got := runDBRunStatus(filepath.Join(runsDir, "hit"), "hit"); got != runDBStatusCompleted {
		t.Fatalf("name-matched status = %q, want completed", got)
	}

	// Name mismatch falls back to the newest row in the per-run journal.
	if got := runDBRunStatus(filepath.Join(runsDir, "hit"), "some-other-name"); got != runDBStatusCompleted {
		t.Fatalf("fallback status = %q, want completed", got)
	}

	// Unreadable garbage -> "" rather than a panic or a wrong verdict.
	if err := os.MkdirAll(filepath.Join(runsDir, "junk"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, "junk", "run_db.sqlite"), []byte("not a database"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := runDBRunStatus(filepath.Join(runsDir, "junk"), "junk"); got != "" {
		t.Fatalf("corrupt run_db status = %q, want \"\"", got)
	}
}

// TestReconcileTwiceDoesNotDoubleCountReattached guards the accounting against a
// repeat Reconcile: the same live row is re-found on the second pass, and
// claiming it twice would leak a slot (and an exclusive hold) that no watcher
// ever releases, wedging admission for the daemon's lifetime.
func TestReconcileTwiceDoesNotDoubleCountReattached(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted-twice", "fake", false)

	r.disp.Reconcile()
	r.disp.Reconcile()

	r.disp.mu.Lock()
	active := r.disp.active
	r.disp.mu.Unlock()
	if active != 1 {
		t.Fatalf("active = %d after two Reconciles, want 1", active)
	}

	// The single watcher still finalizes and frees the slot on exit.
	child.kill(t)
	waitForProcessState(t, r.runsDir, "adopted-twice", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		r.disp.mu.Lock()
		active = r.disp.active
		r.disp.mu.Unlock()
		if active == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("active = %d after the reattached job exited, want 0", active)
}

// TestReattachedOperatorStopFinalizesCanceled is the third finalize branch: an
// operator DELETE on a reattached job must not read back as a crash. procmgr's
// unsupervised Stop path records `stopping` before signalling the group, and
// the watcher turns that into `canceled` with no exit code -- the same terminal
// any other operator-requested stop produces -- so a dependent gates on it by
// on_failure exactly as it would on a canceled parent.
func TestReattachedOperatorStopFinalizesCanceled(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	child := startReattachedChild(t, r.runsDir, "adopted-stop", "fake", false)

	r.disp.Reconcile()

	dep := exclSpec("dep-after-stop", "fake", false)
	dep.After = []string{"adopted-stop"}
	dep.OnFailure = OnFailureSkip
	if _, err := r.disp.Submit(dep); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("dep-after-stop", 200*time.Millisecond)

	// Operator DELETE (graceful, no force) through the dispatcher's Cancel path.
	st, err := r.disp.Cancel("adopted-stop", false)
	if err != nil {
		t.Fatal(err)
	}
	if st.Status != procmgr.StatusStopping {
		t.Fatalf("status right after Cancel = %q, want stopping (the durable trace of the request)", st.Status)
	}
	child.reap(t) // the daemon signalled it; only this process can reap it

	final := waitForProcessState(t, r.runsDir, "adopted-stop", 3*time.Second, func(st *procmgr.ProcessState) bool {
		return isTerminal(st.Status)
	})
	if final.Status != StateCanceled {
		t.Fatalf("adopted-stop status = %q, want canceled (not crashed)", final.Status)
	}
	if final.ExitCode != nil {
		t.Fatalf("adopted-stop exit_code = %d, want none (still unknowable)", *final.ExitCode)
	}
	if final.LastError != "reattached: stopped by operator request; exit status unknown" {
		t.Fatalf("adopted-stop last_error = %q", final.LastError)
	}

	// A canceled parent is a non-success terminal: on_failure=skip skips.
	r.waitForState("dep-after-stop", StateSkipped, 3*time.Second)
}
