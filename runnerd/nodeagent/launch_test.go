package nodeagent

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestHeadToHeadArgv is the relocated table test for the head-to-head argv
// tail and its containment-refusal re-resolve (design 5.4), moved here with
// the launch template it exercises (D1). Launch happens in a later goroutine
// against the persisted spec, so both checkpoints are re-resolved through the
// same runs-dir guard they got at submit.
func TestHeadToHeadArgv(t *testing.T) {
	runsDir := t.TempDir()

	cases := []struct {
		name       string
		spec       Spec
		wantErrSub string // non-empty: want an error containing this substring
		wantArgv   []string
	}{
		{
			name: "contained-defaults-games-and-device",
			spec: Spec{
				Kind:        KindHeadToHead,
				CheckpointA: "run-a/snapshots/prtcfr_checkpoint.pt",
				CheckpointB: "run-b/snapshots/prtcfr_checkpoint.pt",
			},
			wantArgv: []string{
				"--checkpoint-a", filepath.Join(runsDir, "run-a", "snapshots", "prtcfr_checkpoint.pt"),
				"--checkpoint-b", filepath.Join(runsDir, "run-b", "snapshots", "prtcfr_checkpoint.pt"),
				"--games", "5000",
				"--device", "cpu",
			},
		},
		{
			name: "contained-explicit-games-and-device",
			spec: Spec{
				Kind:        KindHeadToHead,
				CheckpointA: "run-a/snap.pt",
				CheckpointB: "run-b/snap.pt",
				Games:       250,
				Device:      "xpu",
			},
			wantArgv: []string{
				"--checkpoint-a", filepath.Join(runsDir, "run-a", "snap.pt"),
				"--checkpoint-b", filepath.Join(runsDir, "run-b", "snap.pt"),
				"--games", "250",
				"--device", "xpu",
			},
		},
		{
			name:       "checkpoint-a-absolute-rejected",
			spec:       Spec{Kind: KindHeadToHead, CheckpointA: "/etc/passwd", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
		{
			name:       "checkpoint-b-parent-traversal-rejected",
			spec:       Spec{Kind: KindHeadToHead, CheckpointA: "run-a/snap.pt", CheckpointB: "../../etc/passwd"},
			wantErrSub: "checkpoint_b",
		},
		{
			name:       "checkpoint-a-nested-traversal-rejected",
			spec:       Spec{Kind: KindHeadToHead, CheckpointA: "ok/a/../../../escape.pt", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
		{
			name:       "checkpoint-a-empty-rejected",
			spec:       Spec{Kind: KindHeadToHead, CheckpointA: "", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := headToHeadArgv(tc.spec, runsDir)
			if tc.wantErrSub != "" {
				if err == nil {
					t.Fatalf("headToHeadArgv() = %v, nil error; want error containing %q", got, tc.wantErrSub)
				}
				if !strings.Contains(err.Error(), tc.wantErrSub) {
					t.Fatalf("err = %q, want it to contain %q", err.Error(), tc.wantErrSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("headToHeadArgv() unexpected error: %v", err)
			}
			if len(got) != len(tc.wantArgv) {
				t.Fatalf("argv = %v, want %v", got, tc.wantArgv)
			}
			for i := range got {
				if got[i] != tc.wantArgv[i] {
					t.Fatalf("argv = %v, want %v", got, tc.wantArgv)
				}
			}
		})
	}
}
