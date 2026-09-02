package nodeagent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestIncrementalCommitOnProgressTick covers the mid-run half of the artifact
// cadence (D51): a running job commits what it has written without waiting for
// its terminal, and the progress post that renews the lease keeps arriving
// while the upload runs.
func TestIncrementalCommitOnProgressTick(t *testing.T) {
	stub := newStubCoordinator(t)
	launcher := &fakeLauncher{status: ProcessStatus{Status: "running", PID: 21, Found: true}}
	agent, _ := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })

	job := newRunningJob(t, agent, stub, "job-incremental")
	if err := os.WriteFile(filepath.Join(job.runDir(), "metrics.jsonl"), []byte(`{"iter":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	done := runJobInBackground(t, agent, job)

	deadline := time.Now().Add(15 * time.Second)
	var incremental bool
	for time.Now().Before(deadline) && !incremental {
		_, _, _, commits, _ := stub.snapshotState()
		for _, c := range commits {
			if !c.Final {
				incremental = true
			}
		}
		if !incremental {
			time.Sleep(50 * time.Millisecond)
		}
	}
	job.requestStop(stopReason{kind: stopOrphaned, detail: "test teardown"})
	waitFor(t, done)

	if !incremental {
		t.Fatalf("a running job never committed before its terminal")
	}
	progress, _, _, _, _ := stub.snapshotState()
	if len(progress) == 0 {
		t.Fatalf("the tick never renewed the lease")
	}
}
