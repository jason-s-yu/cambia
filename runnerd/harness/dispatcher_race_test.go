package harness

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// raceSpec is a minimal cpu "fake" (sleep 30) spec for the cancel/launch race
// tests: if the job ever launches it runs sleep 30, so a dropped cancel is
// observable as a job stuck running rather than reaching a terminal state.
func raceSpec(name string) JobSpec {
	return JobSpec{
		Kind:   "fake",
		Commit: strings.Repeat("a", 40),
		Name:   name,
		Config: "cfr/config/prtcfr_prod.yaml",
		Device: "cpu",
	}
}

// TestCancelAtLaunchBoundaryHonored proves L7: a Cancel that arrives at the
// exact check-then-launch boundary must abort the job before StartWithOpts, not
// be dropped. The launchGateHook fires the cancel deterministically at that
// boundary; with the race window open the job would launch sleep 30 and never
// reach canceled.
func TestCancelAtLaunchBoundaryHonored(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})

	var once sync.Once
	launchGateHook = func(name string) {
		once.Do(func() {
			// Injected at the boundary, with d.mu not held, so Cancel can take it.
			_, _ = r.disp.Cancel(name, false)
		})
	}
	t.Cleanup(func() { launchGateHook = nil })

	if _, err := r.disp.Submit(raceSpec("boundary-1")); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// The boundary cancel must win: job canceled, no process ever launched.
	r.waitForState("boundary-1", StateCanceled, 3*time.Second)
	if st, ok := r.pm.GetState("boundary-1"); ok && st.Status == procmgr.StatusRunning {
		t.Fatalf("job launched despite a boundary cancel: status=%s", st.Status)
	}
}

// TestCancelRacesLaunchNeverLeavesRunning stresses the cancel/launch race
// concurrently (run under -race): every submitted job is canceled in a racing
// goroutine, and each must settle into a terminal state (canceled, or stopped if
// the cancel lost the race and took the running-job Stop path). A job left
// running would mean a dropped cancel.
func TestCancelRacesLaunchNeverLeavesRunning(t *testing.T) {
	launchGateHook = nil // no deterministic injection; exercise the real timing
	const n = 60
	r := newRig(t, rigConfig{maxJobs: 8, maxQueue: n + 4})

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("race-%d", i)
		if _, err := r.disp.Submit(raceSpec(name)); err != nil {
			t.Fatalf("submit %s: %v", name, err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = r.disp.Cancel(name, false)
		}()
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		name := fmt.Sprintf("race-%d", i)
		deadline := time.Now().Add(6 * time.Second)
		var last string
		settled := false
		for time.Now().Before(deadline) {
			s, ok := r.getState(name)
			if ok {
				last = s
				if isTerminal(s) {
					settled = true
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
		}
		if !settled {
			t.Fatalf("job %s left non-terminal (state=%q): a cancel was dropped", name, last)
		}
	}
}
