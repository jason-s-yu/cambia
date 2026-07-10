package harness

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// -----------------------------------------------------------------------
// JobSpec unit tests (warm_start field: kind-scoped, optional, guards) --
// mirrors the target field tests in evaluate_target_test.go (design
// cambia-334).
// -----------------------------------------------------------------------

func TestJobSpecWarmStartForbidden(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"train-with-warm-start", JobSpec{Kind: KindTrain, WarmStart: "prior-run/snapshots/x.pt"}, false},
		{"train-without-warm-start", JobSpec{Kind: KindTrain}, false},
		{"evaluate-with-warm-start", JobSpec{Kind: KindEvaluate, WarmStart: "x"}, true},
		{"head-to-head-with-warm-start", JobSpec{Kind: KindHeadToHead, WarmStart: "x"}, true},
		{"bench-with-warm-start", JobSpec{Kind: KindBench, WarmStart: "x"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.warmStartForbidden(); got != tc.want {
				t.Errorf("warmStartForbidden() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecContainedWarmStart(t *testing.T) {
	s := JobSpec{Kind: KindTrain, WarmStart: "prior-run/snapshots/x.pt"}
	got := s.containedWarmStart()
	if len(got) != 1 || got[0].label != "warm_start" || got[0].value != "prior-run/snapshots/x.pt" {
		t.Fatalf("containedWarmStart() = %+v, want [{warm_start prior-run/snapshots/x.pt}]", got)
	}

	// warm_start is optional even for train: an empty value adds no entry,
	// unlike containedTarget's implicit required-ness for evaluate.
	s2 := JobSpec{Kind: KindTrain, WarmStart: ""}
	if got := s2.containedWarmStart(); got != nil {
		t.Fatalf("containedWarmStart() for empty warm_start = %+v, want nil", got)
	}

	// Non-train kinds get no containment entry (already rejected earlier by
	// warmStartForbidden in the real submit path).
	s3 := JobSpec{Kind: KindEvaluate, WarmStart: "x"}
	if got := s3.containedWarmStart(); got != nil {
		t.Fatalf("containedWarmStart() for evaluate = %+v, want nil", got)
	}
}

// -----------------------------------------------------------------------
// Submit-time guards (design cambia-334): warm_start gets the same lexical +
// containment guard as target/checkpoints, plus a submit-time existence
// check, and is rejected outright on a non-train kind.
// -----------------------------------------------------------------------

// warmStartAlgos maps kind "train" (and "evaluate", for the kind-scoping
// rejection case) to the fake script's fast-exit subcommand, mirroring
// evalGuardAlgos.
func warmStartAlgos() map[string][]string {
	return map[string][]string{
		"train":    {"quick"},
		"evaluate": {"quick"},
	}
}

// trainSpec returns a minimal train job spec body, with warm_start included
// only when non-empty.
func trainSpec(name, warmStart string) map[string]any {
	spec := map[string]any{
		"kind":   "train",
		"commit": strings.Repeat("a", 40),
		"name":   name,
		"config": "cfr/config/prtcfr_prod.yaml",
		"device": "cpu",
	}
	if warmStart != "" {
		spec["warm_start"] = warmStart
	}
	return spec
}

func TestSubmitWarmStartGuards(t *testing.T) {
	reject := []struct {
		name, warmStart string
	}{
		{"warm-abs", "/etc/passwd"},
		{"warm-dotdot", "../../etc/passwd"},
		{"warm-nested-dotdot", "ok/a/../../../escape.pt"},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, rigConfig{algos: warmStartAlgos()})
			resp := r.do(http.MethodPost, "/harness/jobs", trainSpec(tc.name, tc.warmStart))
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("got %d, want 400", resp.StatusCode)
			}
			var body map[string]string
			decodeBody(t, resp, &body)
			if body["error"] != "invalid_path" {
				t.Fatalf("error = %q, want invalid_path", body["error"])
			}
			if !strings.Contains(body["detail"], "warm_start") {
				t.Fatalf("detail = %q, want mention of warm_start", body["detail"])
			}
		})
	}

	t.Run("evaluate-with-warm-start-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: warmStartAlgos()})
		spec := evalSpec("eval-bad-warm-start", "prior-run", 0)
		spec["warm_start"] = "prior-run/snapshots/x.pt"
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_warm_start" {
			t.Fatalf("error = %q, want invalid_warm_start", body["error"])
		}
	})

	t.Run("train-without-warm-start-accepted", func(t *testing.T) {
		// warm_start is optional: a train spec that omits it entirely must not
		// be rejected as missing (unlike evaluate's required target).
		r := newRig(t, rigConfig{algos: warmStartAlgos()})
		resp := r.do(http.MethodPost, "/harness/jobs", trainSpec("warm-absent", ""))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201 for a train job with no warm_start", resp.StatusCode)
		}
		resp.Body.Close()
		r.waitForState("warm-absent", procmgr.StatusStopped, 5*time.Second)
	})

	t.Run("missing-snapshot-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: warmStartAlgos()})
		// The referenced run dir exists but the snapshot file itself does not.
		if err := os.MkdirAll(filepath.Join(r.runsDir, "prior-run", "snapshots"), 0o755); err != nil {
			t.Fatal(err)
		}
		resp := r.do(http.MethodPost, "/harness/jobs",
			trainSpec("warm-missing", "prior-run/snapshots/prtcfr_snapshot_iter_530.pt"))
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "warm_start_not_found" {
			t.Fatalf("error = %q, want warm_start_not_found", body["error"])
		}
	})

	t.Run("accept-contained-and-existing", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: warmStartAlgos()})
		snapshot := filepath.Join(r.runsDir, "prior-run", "snapshots", "prtcfr_snapshot_iter_530.pt")
		if err := os.MkdirAll(filepath.Dir(snapshot), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(snapshot, []byte("snap"), 0o644); err != nil {
			t.Fatal(err)
		}
		resp := r.do(http.MethodPost, "/harness/jobs",
			trainSpec("warm-ok", "prior-run/snapshots/prtcfr_snapshot_iter_530.pt"))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201 for contained + existing warm_start", resp.StatusCode)
		}
		resp.Body.Close()
		r.waitForState("warm-ok", procmgr.StatusStopped, 5*time.Second)
	})
}
