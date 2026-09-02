package harness

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// TestJobViewRendersFanInAfterAll (AC7, D29): a fan-in job's JobView keeps
// `after` in its pre-r2 string shape, carrying the first parent, and adds
// `after_all` carrying every parent. The raw emitted JSON (not just the Go
// struct) is asserted, matching the shape src/harness/client.py's
// parents_of() parses (cfr/tests/test_harness_client.py
// test_parents_of_matches_go_job_view_shape mirrors this exact literal).
func TestJobViewRendersFanInAfterAll(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 4})
	parents := []string{"fav-p1", "fav-p2", "fav-p3"}
	for _, name := range parents {
		writeTerminalParent(t, r.runsDir, name, procmgr.StatusStopped, 0)
	}

	child := depSpecFanIn("fav-child", "fake", parents, "skip")
	if _, err := r.disp.Submit(child); err != nil {
		t.Fatal(err)
	}
	r.waitForState("fav-child", procmgr.StatusRunning, 3*time.Second)

	resp := r.do(http.MethodGet, "/harness/jobs/fav-child", nil)
	var envelope map[string]any
	decodeBody(t, resp, &envelope)
	raw, ok := envelope["job"].(map[string]any)
	if !ok {
		t.Fatalf("job = %v (%T), want an object", envelope["job"], envelope["job"])
	}

	after, ok := raw["after"].(string)
	if !ok {
		t.Fatalf("after = %v (%T), want a string", raw["after"], raw["after"])
	}
	if after != parents[0] {
		t.Fatalf("after = %q, want first parent %q", after, parents[0])
	}
	afterAllRaw, ok := raw["after_all"].([]any)
	if !ok {
		t.Fatalf("after_all = %v (%T), want a list", raw["after_all"], raw["after_all"])
	}
	afterAll := make([]string, len(afterAllRaw))
	for i, v := range afterAllRaw {
		afterAll[i], _ = v.(string)
	}
	if len(afterAll) != len(parents) {
		t.Fatalf("after_all = %v, want %v", afterAll, parents)
	}
	for i, p := range parents {
		if afterAll[i] != p {
			t.Fatalf("after_all[%d] = %q, want %q (after_all = %v)", i, afterAll[i], p, afterAll)
		}
	}
}

// TestQueueSnapshotRendersFanInAfterAll (AC7, D29): the /ws/harness/queue
// snapshot renders a still-queued fan-in job's after/after_all the same way
// as the single-job view: after carries the first parent as a string,
// after_all carries every parent.
func TestQueueSnapshotRendersFanInAfterAll(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 4})
	parents := []string{"qsv-p1", "qsv-p2", "qsv-p3"}
	for i, name := range parents {
		// qsv-p3 (the last) stays non-terminal so the child stays queued long
		// enough to read the snapshot; see TestFanInLaunchesOnlyWhenAllParentsSucceed
		// for why the on-disk record must exist even for the pending override.
		writeTerminalParent(t, r.runsDir, name, procmgr.StatusStopped, 0)
		if i == len(parents)-1 {
			r.disp.mu.Lock()
			r.disp.pending[name] = &job{spec: JobSpec{Name: name}, state: StateQueued}
			r.disp.mu.Unlock()
		}
	}

	child := depSpecFanIn("qsv-child", "fake", parents, "skip")
	if _, err := r.disp.Submit(child); err != nil {
		t.Fatal(err)
	}
	if s, _ := r.getState("qsv-child"); s != StateQueued {
		t.Fatalf("qsv-child state = %q, want queued", s)
	}

	snap := r.disp.Snapshot()
	data, err := json.Marshal(snap)
	if err != nil {
		t.Fatal(err)
	}
	var raw struct {
		Queue []map[string]any `json:"queue"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	var entry map[string]any
	for _, q := range raw.Queue {
		if q["job_id"] == "qsv-child" {
			entry = q
		}
	}
	if entry == nil {
		t.Fatalf("qsv-child not found in queue snapshot: %s", data)
	}
	if entry["after"] != parents[0] {
		t.Fatalf("after = %v, want %q", entry["after"], parents[0])
	}
	afterAllRaw, ok := entry["after_all"].([]any)
	if !ok || len(afterAllRaw) != len(parents) {
		t.Fatalf("after_all = %v, want %v", entry["after_all"], parents)
	}
	for i, p := range parents {
		if afterAllRaw[i] != p {
			t.Fatalf("after_all[%d] = %v, want %q", i, afterAllRaw[i], p)
		}
	}
}

// TestJobViewOmitsAfterAllWithoutDependency pins the omitempty side: a job
// with no after list carries neither field, unchanged from pre-D29 behavior.
func TestJobViewOmitsAfterAllWithoutDependency(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	if _, err := r.disp.Submit(depSpec("no-dep", "fake-quick", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("no-dep", procmgr.StatusStopped, 3*time.Second)

	resp := r.do(http.MethodGet, "/harness/jobs/no-dep", nil)
	var envelope map[string]any
	decodeBody(t, resp, &envelope)
	raw, ok := envelope["job"].(map[string]any)
	if !ok {
		t.Fatalf("job = %v (%T), want an object", envelope["job"], envelope["job"])
	}
	if _, ok := raw["after"]; ok {
		t.Fatalf("after present on a job with no dependency: %v", raw)
	}
	if _, ok := raw["after_all"]; ok {
		t.Fatalf("after_all present on a job with no dependency: %v", raw)
	}
}
