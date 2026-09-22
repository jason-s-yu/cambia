package nodeagent

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// adoptedLauncher stands in for procLauncher over a row a previous agent
// incarnation forked. Nothing in this incarnation waits on that process, so its
// raw process.json status never leaves the live set; only the pid-validated
// Effective view sees it go (procmgr.EffectiveStatus). Stop mirrors procmgr's
// unsupervised branch: it records stopping on the row, and the signalled group
// dies.
type adoptedLauncher struct {
	mu    sync.Mutex
	st    ProcessStatus
	stops int
}

func newAdoptedLauncher() *adoptedLauncher {
	return &adoptedLauncher{st: ProcessStatus{
		Status: procmgr.StatusRunning, Effective: procmgr.StatusRunning, PID: 4242, Found: true,
	}}
}

func (l *adoptedLauncher) Ensure(string, string) error { return nil }

func (l *adoptedLauncher) Start(string, Launch) (int, error) { return 0, nil }

func (l *adoptedLauncher) Stop(string, bool) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.stops++
	l.st.Status = procmgr.StatusStopping
	l.st.Effective = procmgr.StatusCrashed
	return nil
}

func (l *adoptedLauncher) Status(string) ProcessStatus {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.st
}

func (l *adoptedLauncher) stopCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.stops
}

// exit is the adopted process dying on its own: the pid probe fails while the
// row on disk still reads running.
func (l *adoptedLauncher) exit() {
	l.mu.Lock()
	l.st.Effective = procmgr.StatusCrashed
	l.mu.Unlock()
}

// resumeAdoptedJob persists a launched lease record, restarts the agent's view
// of it through reattach and register, and resumes it. It returns once the
// resumed job has posted progress, so its supervise loop is running.
func resumeAdoptedJob(t *testing.T, ctx context.Context, agent *Agent, stub *stubCoordinator, spec Spec) *leaseRecord {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(agent.cfg.RunsDir, spec.Name), 0o755); err != nil {
		t.Fatal(err)
	}
	rec := &leaseRecord{
		JobID: spec.Name, LeaseID: "01JADOPTED0000000000000000", LeaseEpoch: 5,
		Token: "lease-token-adopted", Attempt: 1, Commit: spec.Commit,
		Spec: specJSON(t, spec), Policy: fastPolicy(), Launched: true,
		Phase: nashnet.PhaseRunning,
	}
	stub.setLease(rec.Token, rec.LeaseEpoch)
	if err := writeLeaseRecord(agent.cfg.BaseDir, rec); err != nil {
		t.Fatal(err)
	}

	live, pending := agent.reattach()
	rebound, err := agent.register(ctx, live)
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	agent.resumePending(ctx, pending, rebound)
	if got := agent.slots.Active(); got != 1 {
		t.Fatalf("a resumed job holds %d slots, want 1", got)
	}

	deadline := time.Now().Add(10 * time.Second)
	for {
		progress, _, _, _, _ := stub.snapshotState()
		if len(progress) > 0 {
			return rec
		}
		if time.Now().After(deadline) {
			t.Fatalf("the resumed job never posted progress")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// agentIdle reports when every job goroutine the agent started has returned.
func agentIdle(agent *Agent) chan struct{} {
	done := make(chan struct{})
	go func() {
		agent.wg.Wait()
		close(done)
	}()
	return done
}

// writeRunDBStatus writes the one runs row a job's journal carries, which is
// the exit witness for a process nothing waited on (D35).
func writeRunDBStatus(t *testing.T, runDir, name, status string) {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(runDir, runDBName))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE runs (name TEXT PRIMARY KEY, status TEXT, updated_at TEXT)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO runs (name, status, updated_at) VALUES (?, ?, ?)`,
		name, status, time.Now().UTC().Format(time.RFC3339Nano)); err != nil {
		t.Fatal(err)
	}
}

// TestReattachedJobExitIsObserved pins cambia-2357: a job this agent
// reattached after a restart has no wait goroutine, so its raw process.json
// status stays running after the process is gone. The node must still see the
// exit off the effective status, finish the job, post its result, and free
// its slot. The exit code is unknowable, so the verdict comes from the run's
// own journal the way the coordinator's finalizer reads it: completed is a
// clean exit, anything else a crash with no exit code. An exit the forking
// incarnation recorded before it went down keeps its real code, journal or not.
func TestReattachedJobExitIsObserved(t *testing.T) {
	cases := []struct {
		name  string
		runDB string
		// recorded is an exit code the forking incarnation's wait wrote to
		// the row before the agent restarted.
		recorded  *int
		wantState string
		wantCode  *int
		wantError string
	}{
		{"journal completed", "completed", nil, nashnet.ResultStopped, new(int), "reattached: exit code inferred from run_db status completed"},
		{"journal interrupted", "interrupted", nil, nashnet.ResultCrashed, nil, "reattached: process exited; exit status unknown (run_db status=interrupted)"},
		{"no journal", "", nil, nashnet.ResultCrashed, nil, "reattached: process exited; exit status unknown (run_db status=absent)"},
		{"exit recorded before the restart", "interrupted", new(int), nashnet.ResultStopped, new(int), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stub := newStubCoordinator(t)
			launcher := newAdoptedLauncher()
			agent, _ := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(func() {
				cancel()
				agent.wg.Wait()
			})

			spec := Spec{Kind: KindTrain, Name: "job-adopted", Commit: strings.Repeat("c", 40), Exclusive: true}
			if tc.runDB != "" {
				runDir := filepath.Join(agent.cfg.RunsDir, spec.Name)
				if err := os.MkdirAll(runDir, 0o755); err != nil {
					t.Fatal(err)
				}
				writeRunDBStatus(t, runDir, spec.Name, tc.runDB)
			}
			if tc.recorded != nil {
				launcher.st = ProcessStatus{
					Status: procmgr.StatusStopped, Effective: procmgr.StatusStopped,
					PID: 4242, ExitCode: tc.recorded, Found: true,
				}
			}
			rec := resumeAdoptedJob(t, ctx, agent, stub, spec)

			launcher.exit()
			waitFor(t, agentIdle(agent))

			_, results, _, commits, _ := stub.snapshotState()
			if len(results) != 1 {
				t.Fatalf("expected one result for the reattached job, got %+v", results)
			}
			res := results[0]
			if res.State != tc.wantState {
				t.Fatalf("result state = %q, want %q", res.State, tc.wantState)
			}
			switch {
			case tc.wantCode == nil && res.ExitCode != nil:
				t.Fatalf("result exit code = %d, want none (unknowable for a reattached job)", *res.ExitCode)
			case tc.wantCode != nil && (res.ExitCode == nil || *res.ExitCode != *tc.wantCode):
				t.Fatalf("result exit code = %v, want %d", res.ExitCode, *tc.wantCode)
			}
			if res.LastError != tc.wantError {
				t.Fatalf("result last_error = %q, want %q", res.LastError, tc.wantError)
			}
			if len(commits) == 0 || !commits[len(commits)-1].Final {
				t.Fatalf("the reattached job posted a result without a final manifest")
			}
			if n := launcher.stopCount(); n != 0 {
				t.Fatalf("a job that exited on its own was stopped %d times", n)
			}
			if got := agent.slots.Active(); got != 0 {
				t.Fatalf("the finished job still holds %d slots", got)
			}
			if got := agent.slots.ExclusiveHolds(); got != 0 {
				t.Fatalf("the finished exclusive job still holds the node (%d holds)", got)
			}
			if _, err := os.Stat(leasePath(agent.cfg.BaseDir, rec.JobID)); !os.IsNotExist(err) {
				t.Fatalf("the finished job's lease record survived: %v", err)
			}
		})
	}
}

// TestReattachedJobStopObservesTheExit is the stop path of the same defect: a
// revoke of a reattached job signals a group this incarnation cannot wait on,
// so the row reads stopping until someone else rewrites it. The node must see
// the exit off the effective status and post canceled within a poll, not sit
// out the stop wait.
func TestReattachedJobStopObservesTheExit(t *testing.T) {
	stub := newStubCoordinator(t)
	launcher := newAdoptedLauncher()
	agent, _ := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		agent.wg.Wait()
	})

	spec := Spec{Kind: KindTrain, Name: "job-adopted-stop", Commit: strings.Repeat("d", 40)}
	rec := resumeAdoptedJob(t, ctx, agent, stub, spec)

	agent.applyEvents(nashnet.EventsResponse{
		Events:    []nashnet.Event{{Type: nashnet.EventRevoke, LeaseID: rec.LeaseID}},
		NodeEpoch: 3,
	})
	select {
	case <-agentIdle(agent):
	case <-time.After(stopWait / 2):
		t.Fatalf("a revoked reattached job did not settle within %s; its exit went unobserved", stopWait/2)
	}

	_, results, _, _, _ := stub.snapshotState()
	if len(results) != 1 || results[0].State != nashnet.ResultCanceled {
		t.Fatalf("expected one canceled result, got %+v", results)
	}
	if launcher.stopCount() == 0 {
		t.Fatalf("the revoke never stopped the job group")
	}
	if got := agent.slots.Active(); got != 0 {
		t.Fatalf("the stopped job still holds %d slots", got)
	}
}
