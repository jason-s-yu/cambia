package main

import (
	"os"
	"syscall"
	"testing"
)

// TestKillJobsOnSignal pins the shutdown policy (cambia-655). The load-bearing
// case is SIGTERM with no override: systemd sends SIGTERM on every stop and
// restart, so a redeploy must detach from the job process groups rather than
// kill a training run that has been going for weeks. SIGINT is a foreground
// Ctrl-C and keeps the abrupt kill. RUNNERD_KILL_JOBS_ON_STOP restores the old
// kill-on-SIGTERM behavior for an operator who wants a stop to take everything
// down, and is parsed with strconv.ParseBool so the usual 1/true/TRUE spellings
// all work while an unset or malformed value falls back to the safe default.
func TestKillJobsOnSignal(t *testing.T) {
	cases := []struct {
		name string
		sig  os.Signal
		env  string
		want bool
	}{
		{"sigterm default detaches", syscall.SIGTERM, "", false},
		{"sigterm env 0 detaches", syscall.SIGTERM, "0", false},
		{"sigterm env false detaches", syscall.SIGTERM, "false", false},
		{"sigterm env garbage detaches", syscall.SIGTERM, "yes-please", false},
		{"sigterm env 1 kills", syscall.SIGTERM, "1", true},
		{"sigterm env true kills", syscall.SIGTERM, "true", true},
		{"sigterm env TRUE kills", syscall.SIGTERM, "TRUE", true},
		{"sigint always kills", syscall.SIGINT, "", true},
		{"sigint kills despite env 0", syscall.SIGINT, "0", true},
		{"sighup default detaches", syscall.SIGHUP, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := killJobsOnSignal(tc.sig, tc.env); got != tc.want {
				t.Fatalf("killJobsOnSignal(%v, %q) = %v, want %v", tc.sig, tc.env, got, tc.want)
			}
		})
	}
}

// TestBuildCommitDefault pins the unstamped default: a binary built without
// -ldflags "-X main.buildCommit=<sha>" reports "dev" on GET /harness/health
// rather than an empty string a monitoring consumer would have to special-case.
func TestBuildCommitDefault(t *testing.T) {
	if buildCommit != "dev" {
		t.Fatalf("buildCommit = %q, want \"dev\" for an unstamped build", buildCommit)
	}
}
