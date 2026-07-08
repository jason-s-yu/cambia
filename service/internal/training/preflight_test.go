package training

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// fakeQuery builds a gpuQueryFunc returning canned output or an error.
func fakeQuery(out string, err error) gpuQueryFunc {
	return func() (string, error) { return out, err }
}

func TestGPUVRAMCheckNoNvidiaSmi(t *testing.T) {
	// A CPU host (nvidia-smi absent) passes regardless of the requirement.
	c := gpuVRAMCheck(8.0, fakeQuery("", exec.ErrNotFound))
	if !c.OK {
		t.Errorf("no-nvidia-smi should pass, got %+v", c)
	}
	if c.Name != "gpu_vram" {
		t.Errorf("name = %q, want gpu_vram", c.Name)
	}
}

func TestGPUVRAMCheckNoRequirement(t *testing.T) {
	// minGB <= 0 never queries and always passes.
	c := gpuVRAMCheck(0, fakeQuery("", errors.New("should not be called")))
	if !c.OK {
		t.Errorf("zero requirement should pass, got %+v", c)
	}
}

func TestGPUVRAMCheckBelowThreshold(t *testing.T) {
	// 512 MiB free, need 1 GiB -> block.
	c := gpuVRAMCheck(1.0, fakeQuery("512, 30, Fake GPU\n", nil))
	if c.OK {
		t.Errorf("512 MiB free vs 1 GiB requirement should block, got %+v", c)
	}
}

func TestGPUVRAMCheckAboveThreshold(t *testing.T) {
	// 8192 MiB free, need 1 GiB -> pass.
	c := gpuVRAMCheck(1.0, fakeQuery("8192, 30, Fake GPU\n", nil))
	if !c.OK {
		t.Errorf("8 GiB free vs 1 GiB requirement should pass, got %+v", c)
	}
}

func TestGPUVRAMCheckNvidiaSmiFails(t *testing.T) {
	// nvidia-smi present but errored (not ErrNotFound) -> hard block.
	c := gpuVRAMCheck(1.0, fakeQuery("", errors.New("driver error")))
	if c.OK {
		t.Errorf("failing nvidia-smi should block, got %+v", c)
	}
}

func TestDiskSpaceCheckNoRequirement(t *testing.T) {
	c := diskSpaceCheck(t.TempDir(), 0)
	if !c.OK {
		t.Errorf("zero requirement should pass, got %+v", c)
	}
}

func TestDiskSpaceCheckImpossibleRequirement(t *testing.T) {
	// No real filesystem has 1e9 GiB free.
	c := diskSpaceCheck(t.TempDir(), 1e9)
	if c.OK {
		t.Errorf("impossible disk requirement should block, got %+v", c)
	}
	if c.Name != "disk_space" {
		t.Errorf("name = %q, want disk_space", c.Name)
	}
}

func TestConcurrencyCapCheck(t *testing.T) {
	runsDir := t.TempDir()
	// Two live runs (our own pid).
	for _, n := range []string{"run-a", "run-b"} {
		dir := filepath.Join(runsDir, n)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := writeProcessState(dir, &ProcessState{
			Name: n, Status: StatusRunning, PID: os.Getpid(), PGID: os.Getpid(),
			CreatedAt: nowRFC3339(),
		}); err != nil {
			t.Fatal(err)
		}
	}

	if c := concurrencyCapCheck(runsDir, 0); !c.OK {
		t.Errorf("cap 0 (disabled) should pass, got %+v", c)
	}
	if c := concurrencyCapCheck(runsDir, 1); c.OK {
		t.Errorf("2 live runs at cap 1 should block, got %+v", c)
	}
	if c := concurrencyCapCheck(runsDir, 5); !c.OK {
		t.Errorf("2 live runs under cap 5 should pass, got %+v", c)
	}
}

func TestConcurrencyCapIgnoresDeadRuns(t *testing.T) {
	runsDir := t.TempDir()
	dir := filepath.Join(runsDir, "dead")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Recorded running but pid is dead -> effectiveStatus is crashed, not counted.
	if err := writeProcessState(dir, &ProcessState{
		Name: "dead", Status: StatusRunning, PID: 9999999, PGID: 9999999,
		CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}
	if c := concurrencyCapCheck(runsDir, 1); !c.OK {
		t.Errorf("a dead run must not count toward the cap, got %+v", c)
	}
}

func TestNameCollisionCheck(t *testing.T) {
	runsDir := t.TempDir()
	if c := nameCollisionCheck(runsDir, "fresh"); !c.OK {
		t.Errorf("fresh name should be available, got %+v", c)
	}

	dir := filepath.Join(runsDir, "taken")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeProcessState(dir, &ProcessState{
		Name: "taken", Status: StatusCreated, CreatedAt: nowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}
	if c := nameCollisionCheck(runsDir, "taken"); c.OK {
		t.Errorf("existing process.json should collide, got %+v", c)
	}
}

func TestPreflightPassesForceMatrix(t *testing.T) {
	okGPU := PreflightCheck{"gpu_vram", true, ""}
	badGPU := PreflightCheck{"gpu_vram", false, ""}
	badDisk := PreflightCheck{"disk_space", false, ""}
	badName := PreflightCheck{"name_collision", false, ""}

	cases := []struct {
		desc   string
		checks []PreflightCheck
		force  bool
		want   bool
	}{
		{"all ok", []PreflightCheck{okGPU}, false, true},
		{"all ok with force", []PreflightCheck{okGPU}, true, true},
		{"overridable fail no force", []PreflightCheck{badGPU}, false, false},
		{"overridable fail with force", []PreflightCheck{badGPU}, true, true},
		{"multiple overridable with force", []PreflightCheck{badGPU, badDisk}, true, true},
		{"non-overridable fail with force", []PreflightCheck{badName}, true, false},
		{"mixed with force blocked by name", []PreflightCheck{badGPU, badName}, true, false},
	}
	for _, tc := range cases {
		got, failed := preflightPasses(tc.checks, tc.force)
		if got != tc.want {
			t.Errorf("%s: passes = %v, want %v (failed=%+v)", tc.desc, got, tc.want, failed)
		}
	}
}
