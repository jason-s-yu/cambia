package harness

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// exclSpec is a minimal cpu spec carrying the exclusive flag (cambia-655). kind
// selects the fake script behavior (fake=sleep, fake-quick=exit 0).
func exclSpec(name, kind string, exclusive bool) JobSpec {
	return JobSpec{
		Kind:      kind,
		Commit:    strings.Repeat("a", 40),
		Name:      name,
		Config:    "cfr/config/prtcfr_prod.yaml",
		Device:    "cpu",
		Exclusive: exclusive,
	}
}

// assertQueuedFor asserts a job stays StateQueued for the whole window, i.e. it
// is held (not admitted) despite the daemon having free capacity.
func (r *testRig) assertQueuedFor(id string, window time.Duration) {
	r.t.Helper()
	deadline := time.Now().Add(window)
	for time.Now().Before(deadline) {
		if s, _ := r.getState(id); s != StateQueued {
			r.t.Fatalf("%s state = %q, want queued", id, s)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestExclusiveAdmissionUnit (white-box) pins canLaunchLocked: an exclusive job
// launches only into an idle daemon, while any exclusive job is active nothing
// else launches, and a normal job respects the plain concurrency cap.
func TestExclusiveAdmissionUnit(t *testing.T) {
	runsDir := t.TempDir()
	fe := &fakeEnv{runsDir: runsDir}
	pm := procmgr.NewProcessManager(runsDir, t.TempDir(), "cambia", NewRunResolver(runsDir), fakeAlgos())
	disp := NewDispatcher(pm, fe, runsDir, 3, 16, 15*time.Millisecond)

	disp.mu.Lock()
	defer disp.mu.Unlock()

	excl := &job{spec: JobSpec{Name: "e", Exclusive: true}, state: StateQueued}
	norm := &job{spec: JobSpec{Name: "n"}, state: StateQueued}

	if !disp.canLaunchLocked(excl) {
		t.Fatal("exclusive should launch when active==0")
	}
	disp.active = 1
	if disp.canLaunchLocked(excl) {
		t.Fatal("exclusive must not launch when active!=0")
	}
	if !disp.canLaunchLocked(norm) {
		t.Fatal("normal job should launch with a free slot (1<3)")
	}

	disp.active = 0
	disp.activeExclusive = true
	if disp.canLaunchLocked(norm) {
		t.Fatal("normal job must not launch while an exclusive job is active")
	}
	if disp.canLaunchLocked(excl) {
		t.Fatal("second exclusive must not launch while an exclusive job is active")
	}

	disp.activeExclusive = false
	disp.active = 3
	if disp.canLaunchLocked(norm) {
		t.Fatal("normal job must not launch when full (3>=3)")
	}
}

// TestExclusiveJSONDefaultsFalse pins the wire + on-disk contract: an absent
// `exclusive` field decodes false (backward compat), a set field decodes true,
// and the flag round-trips through jobspec.json (omitted when false).
func TestExclusiveJSONDefaultsFalse(t *testing.T) {
	var absent JobSpec
	if err := json.Unmarshal([]byte(`{"kind":"train","name":"n"}`), &absent); err != nil {
		t.Fatal(err)
	}
	if absent.Exclusive {
		t.Fatal("absent exclusive field should decode false")
	}
	var set JobSpec
	if err := json.Unmarshal([]byte(`{"kind":"train","name":"n","exclusive":true}`), &set); err != nil {
		t.Fatal(err)
	}
	if !set.Exclusive {
		t.Fatal("exclusive:true should decode true")
	}

	// jobspec.json round-trip: a true value persists and reads back.
	dir := t.TempDir()
	if err := writeJobSpec(dir, &JobSpec{Name: "n", Kind: "train", Exclusive: true}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, jobSpecFile))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"exclusive": true`) {
		t.Fatalf("jobspec.json missing exclusive: %s", data)
	}
	if got := readJobSpec(dir); got == nil || !got.Exclusive {
		t.Fatalf("readJobSpec lost exclusive: %+v", got)
	}

	// A false value is omitted (omitempty) and reads back false.
	dir2 := t.TempDir()
	if err := writeJobSpec(dir2, &JobSpec{Name: "n", Kind: "train"}); err != nil {
		t.Fatal(err)
	}
	data2, err := os.ReadFile(filepath.Join(dir2, jobSpecFile))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data2), "exclusive") {
		t.Fatalf("false exclusive should be omitted from jobspec.json: %s", data2)
	}
	if got := readJobSpec(dir2); got == nil || got.Exclusive {
		t.Fatalf("absent exclusive should read false: %+v", got)
	}
}

// TestExclusiveSubmitThroughAPI proves the flag travels the HTTP submit path,
// persists into jobspec.json, and surfaces on the JobView.
func TestExclusiveSubmitThroughAPI(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})
	// fake-quick (exit 0) keeps this a pure submit-path/persist/view check without
	// leaving a long-running process for cleanup to drain.
	body := baseSpec("api-excl", "fake-quick")
	body["exclusive"] = true
	resp := r.do(http.MethodPost, "/harness/jobs", body)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit exclusive: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()

	var jr jobResp
	resp = r.do(http.MethodGet, "/harness/jobs/api-excl", nil)
	decodeBody(t, resp, &jr)
	if !jr.Job.Exclusive {
		t.Fatal("submitted exclusive job view missing exclusive=true")
	}
	if got := readJobSpec(filepath.Join(r.runsDir, "api-excl")); got == nil || !got.Exclusive {
		t.Fatalf("jobspec.json missing exclusive: %+v", got)
	}

	// An old-client body (no exclusive field) still submits and defaults false.
	resp = r.do(http.MethodPost, "/harness/jobs", baseSpec("api-shared", "fake-quick"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit shared: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	var shared jobResp
	resp = r.do(http.MethodGet, "/harness/jobs/api-shared", nil)
	decodeBody(t, resp, &shared)
	if shared.Job.Exclusive {
		t.Fatal("shared job (no exclusive field) should default exclusive=false")
	}
}

// TestMaxJobsThreeAdmitsThree proves the raised concurrency default: with
// maxJobs=3 three non-exclusive jobs run at once.
func TestMaxJobsThreeAdmitsThree(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})
	for _, name := range []string{"c1", "c2", "c3"} {
		if _, err := r.disp.Submit(exclSpec(name, "fake", false)); err != nil {
			t.Fatalf("submit %s: %v", name, err)
		}
	}
	r.waitForState("c1", procmgr.StatusRunning, 3*time.Second)
	r.waitForState("c2", procmgr.StatusRunning, 3*time.Second)
	r.waitForState("c3", procmgr.StatusRunning, 3*time.Second)
}

// TestExclusiveWaitsForDrain: an exclusive job queued behind a running job
// launches only once the daemon drains to idle.
func TestExclusiveWaitsForDrain(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})
	if _, err := r.disp.Submit(exclSpec("blk", "fake", false)); err != nil {
		t.Fatal(err)
	}
	r.waitForState("blk", procmgr.StatusRunning, 3*time.Second)

	if _, err := r.disp.Submit(exclSpec("excl", "fake", true)); err != nil {
		t.Fatal(err)
	}
	// active!=0, so the exclusive job is held even though two slots are free.
	r.assertQueuedFor("excl", 300*time.Millisecond)

	// Drain the blocker -> daemon idle -> exclusive launches.
	if _, err := r.disp.Cancel("blk", true); err != nil {
		t.Fatal(err)
	}
	r.waitForState("excl", procmgr.StatusRunning, 3*time.Second)
}

// TestNothingLaunchesWhileExclusiveRuns: while an exclusive job runs, later
// non-exclusive jobs stay queued despite free concurrency slots.
func TestNothingLaunchesWhileExclusiveRuns(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})
	if _, err := r.disp.Submit(exclSpec("excl", "fake", true)); err != nil {
		t.Fatal(err)
	}
	r.waitForState("excl", procmgr.StatusRunning, 3*time.Second)

	for _, name := range []string{"s1", "s2"} {
		if _, err := r.disp.Submit(exclSpec(name, "fake", false)); err != nil {
			t.Fatalf("submit %s: %v", name, err)
		}
	}
	r.assertQueuedFor("s1", 300*time.Millisecond)
	if s, _ := r.getState("s2"); s != StateQueued {
		t.Fatalf("s2 state = %q, want queued while exclusive runs", s)
	}

	// Drain the exclusive job -> the held jobs admit.
	if _, err := r.disp.Cancel("excl", true); err != nil {
		t.Fatal(err)
	}
	r.waitForState("s1", procmgr.StatusRunning, 3*time.Second)
}

// TestExclusiveHeadBarriersLaterReady: a deferred exclusive job at the queue head
// barriers a later ready non-exclusive job, so the small job cannot pass it into
// a free slot and starve it. Once the daemon drains the exclusive job launches
// first; the small job launches only after the exclusive job clears.
func TestExclusiveHeadBarriersLaterReady(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})
	if _, err := r.disp.Submit(exclSpec("blk", "fake", false)); err != nil {
		t.Fatal(err)
	}
	r.waitForState("blk", procmgr.StatusRunning, 3*time.Second)

	// excl (head, deferred for occupancy) then s1 (ready, would fit a free slot).
	if _, err := r.disp.Submit(exclSpec("excl", "fake", true)); err != nil {
		t.Fatal(err)
	}
	if _, err := r.disp.Submit(exclSpec("s1", "fake", false)); err != nil {
		t.Fatal(err)
	}
	// Barrier: despite two free slots, s1 does not pass the deferred exclusive head.
	r.assertQueuedFor("excl", 300*time.Millisecond)
	if s, _ := r.getState("s1"); s != StateQueued {
		t.Fatalf("s1 state = %q, want queued (barriered behind exclusive head)", s)
	}

	// Drain the blocker -> exclusive launches first; s1 still held (exclusive runs).
	if _, err := r.disp.Cancel("blk", true); err != nil {
		t.Fatal(err)
	}
	r.waitForState("excl", procmgr.StatusRunning, 3*time.Second)
	if s, _ := r.getState("s1"); s != StateQueued {
		t.Fatalf("s1 state = %q, want queued while exclusive runs", s)
	}

	// Drain the exclusive job -> s1 finally admits.
	if _, err := r.disp.Cancel("excl", true); err != nil {
		t.Fatal(err)
	}
	r.waitForState("s1", procmgr.StatusRunning, 3*time.Second)
}

// TestReconcileReattachedExclusiveHoldsDaemon proves the exclusive hold is
// restored at daemon start for a reattached running exclusive job: a job
// submitted after Reconcile stays queued until the reattached job exits, then
// the watcher clears the hold and it launches.
func TestReconcileReattachedExclusiveHoldsDaemon(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 3})

	// A real live process stands in for a reattached running exclusive job: its
	// pid makes EffectiveStatus report running until it is killed.
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	killed := false
	t.Cleanup(func() {
		if !killed {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		}
	})

	name := "excl-re"
	st := &procmgr.ProcessState{Name: name, Status: procmgr.StatusRunning, PID: cmd.Process.Pid, Algorithm: "fake"}
	if err := procmgr.WriteProcessState(filepath.Join(r.runsDir, name), st); err != nil {
		t.Fatal(err)
	}
	if err := writeJobSpec(filepath.Join(r.runsDir, name), &JobSpec{Name: name, Kind: "fake", Exclusive: true}); err != nil {
		t.Fatal(err)
	}

	r.disp.Reconcile()

	// Hold restored: a normal job submitted now stays queued despite free slots.
	if _, err := r.disp.Submit(exclSpec("held", "fake", false)); err != nil {
		t.Fatal(err)
	}
	r.assertQueuedFor("held", 300*time.Millisecond)

	// Reattached exclusive exits -> watcher observes pid death -> hold clears.
	_ = cmd.Process.Kill()
	_, _ = cmd.Process.Wait()
	killed = true
	r.waitForState("held", procmgr.StatusRunning, 3*time.Second)
}
