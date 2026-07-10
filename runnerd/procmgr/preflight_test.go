package procmgr

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// fakeQuery builds a GPUQueryFunc returning canned output or an error.
func fakeQuery(out string, err error) GPUQueryFunc {
	return func() (string, error) { return out, err }
}

func TestGPUVRAMCheckNoNvidiaSmi(t *testing.T) {
	// A CPU host (nvidia-smi absent) passes regardless of the requirement.
	c := GPUVRAMCheck(8.0, fakeQuery("", exec.ErrNotFound))
	if !c.OK {
		t.Errorf("no-nvidia-smi should pass, got %+v", c)
	}
	if c.Name != "gpu_vram" {
		t.Errorf("name = %q, want gpu_vram", c.Name)
	}
}

func TestGPUVRAMCheckNoRequirement(t *testing.T) {
	// minGB <= 0 never queries and always passes.
	c := GPUVRAMCheck(0, fakeQuery("", errors.New("should not be called")))
	if !c.OK {
		t.Errorf("zero requirement should pass, got %+v", c)
	}
}

func TestGPUVRAMCheckBelowThreshold(t *testing.T) {
	// 512 MiB free, need 1 GiB -> block.
	c := GPUVRAMCheck(1.0, fakeQuery("512, 30, Fake GPU\n", nil))
	if c.OK {
		t.Errorf("512 MiB free vs 1 GiB requirement should block, got %+v", c)
	}
}

func TestGPUVRAMCheckAboveThreshold(t *testing.T) {
	// 8192 MiB free, need 1 GiB -> pass.
	c := GPUVRAMCheck(1.0, fakeQuery("8192, 30, Fake GPU\n", nil))
	if !c.OK {
		t.Errorf("8 GiB free vs 1 GiB requirement should pass, got %+v", c)
	}
}

func TestGPUVRAMCheckNvidiaSmiFails(t *testing.T) {
	// nvidia-smi present but errored (not ErrNotFound) -> hard block.
	c := GPUVRAMCheck(1.0, fakeQuery("", errors.New("driver error")))
	if c.OK {
		t.Errorf("failing nvidia-smi should block, got %+v", c)
	}
}

func TestDiskSpaceCheckNoRequirement(t *testing.T) {
	c := DiskSpaceCheck(t.TempDir(), 0)
	if !c.OK {
		t.Errorf("zero requirement should pass, got %+v", c)
	}
}

func TestDiskSpaceCheckImpossibleRequirement(t *testing.T) {
	// No real filesystem has 1e9 GiB free.
	c := DiskSpaceCheck(t.TempDir(), 1e9)
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
		if err := WriteProcessState(dir, &ProcessState{
			Name: n, Status: StatusRunning, PID: os.Getpid(), PGID: os.Getpid(),
			CreatedAt: NowRFC3339(),
		}); err != nil {
			t.Fatal(err)
		}
	}

	if c := ConcurrencyCapCheck(runsDir, 0); !c.OK {
		t.Errorf("cap 0 (disabled) should pass, got %+v", c)
	}
	if c := ConcurrencyCapCheck(runsDir, 1); c.OK {
		t.Errorf("2 live runs at cap 1 should block, got %+v", c)
	}
	if c := ConcurrencyCapCheck(runsDir, 5); !c.OK {
		t.Errorf("2 live runs under cap 5 should pass, got %+v", c)
	}
}

func TestConcurrencyCapIgnoresDeadRuns(t *testing.T) {
	runsDir := t.TempDir()
	dir := filepath.Join(runsDir, "dead")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Recorded running but pid is dead -> EffectiveStatus is crashed, not counted.
	if err := WriteProcessState(dir, &ProcessState{
		Name: "dead", Status: StatusRunning, PID: 9999999, PGID: 9999999,
		CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}
	if c := ConcurrencyCapCheck(runsDir, 1); !c.OK {
		t.Errorf("a dead run must not count toward the cap, got %+v", c)
	}
}

func TestNameCollisionCheck(t *testing.T) {
	runsDir := t.TempDir()
	if c := NameCollisionCheck(runsDir, "fresh"); !c.OK {
		t.Errorf("fresh name should be available, got %+v", c)
	}

	dir := filepath.Join(runsDir, "taken")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteProcessState(dir, &ProcessState{
		Name: "taken", Status: StatusCreated, CreatedAt: NowRFC3339(),
	}); err != nil {
		t.Fatal(err)
	}
	if c := NameCollisionCheck(runsDir, "taken"); c.OK {
		t.Errorf("existing process.json should collide, got %+v", c)
	}
}

// writeRunConfig writes runs/<name>/config.yaml with the given content.
func writeRunConfig(t *testing.T, runsDir, name, content string) {
	t.Helper()
	dir := filepath.Join(runsDir, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestResolveRunDeviceMissingRun(t *testing.T) {
	if got := ResolveRunDevice(t.TempDir(), "ghost"); got != "auto" {
		t.Errorf("missing run: device = %q, want auto", got)
	}
}

func TestResolveRunDeviceMissingField(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "no-device", "prt_cfr:\n  iterations: 2\n")
	if got := ResolveRunDevice(runsDir, "no-device"); got != "auto" {
		t.Errorf("no device field: device = %q, want auto", got)
	}
}

func TestResolveRunDevicePRTCFRSection(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "section-cpu", "prt_cfr:\n  iterations: 2\n  device: cpu\n")
	if got := ResolveRunDevice(runsDir, "section-cpu"); got != "cpu" {
		t.Errorf("section device: got %q, want cpu", got)
	}
}

func TestResolveRunDeviceRootLevel(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "root-cpu", "device: cpu\n")
	if got := ResolveRunDevice(runsDir, "root-cpu"); got != "cpu" {
		t.Errorf("root device: got %q, want cpu", got)
	}
}

func TestResolveRunDeviceSectionWinsOverRoot(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "mixed",
		"device: cuda\nprt_cfr:\n  iterations: 2\n  device: cpu\n")
	if got := ResolveRunDevice(runsDir, "mixed"); got != "cpu" {
		t.Errorf("mixed device: got %q, want cpu (section wins)", got)
	}
}

func TestResolveRunDeviceCuda(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "cuda-run", "prt_cfr:\n  device: cuda\n")
	if got := ResolveRunDevice(runsDir, "cuda-run"); got != "cuda" {
		t.Errorf("cuda device: got %q, want cuda", got)
	}
}

func TestResolveRunDeviceExplicitAuto(t *testing.T) {
	runsDir := t.TempDir()
	writeRunConfig(t, runsDir, "auto-run", "prt_cfr:\n  device: auto\n")
	if got := ResolveRunDevice(runsDir, "auto-run"); got != "auto" {
		t.Errorf("auto device: got %q, want auto", got)
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
		got, failed := PreflightPasses(tc.checks, tc.force)
		if got != tc.want {
			t.Errorf("%s: passes = %v, want %v (failed=%+v)", tc.desc, got, tc.want, failed)
		}
	}
}

// TestEffectiveStatusRemoteShortCircuit is the cross-host pid-reuse guard for
// the read-time status view: a remote row (Host set) whose recorded pid is dead
// in THIS host's pid space must still report its synced status verbatim, never
// crashed. Without the Host short-circuit, the dead-pid probe would flip a live
// remote run to crashed on the dashboard.
func TestEffectiveStatusRemoteShortCircuit(t *testing.T) {
	remote := &ProcessState{
		Name: "v0.4-prtcfr-r12", Host: "runner1", Status: StatusRunning,
		PID: 9999999, PGID: 9999999, CreatedAt: NowRFC3339(),
	}
	if got := EffectiveStatus(remote); got != StatusRunning {
		t.Errorf("remote running row: EffectiveStatus = %q, want running (Host short-circuit)", got)
	}

	remote.Status = StatusStopping
	if got := EffectiveStatus(remote); got != StatusStopping {
		t.Errorf("remote stopping row: EffectiveStatus = %q, want stopping (no local probe)", got)
	}

	// The same dead pid on a LOCAL row (Host empty) must still resolve to crashed:
	// the short-circuit is scoped to remote rows only.
	local := &ProcessState{
		Name: "local-run", Status: StatusRunning,
		PID: 9999999, PGID: 9999999, CreatedAt: NowRFC3339(),
	}
	if got := EffectiveStatus(local); got != StatusCrashed {
		t.Errorf("local dead-pid row: EffectiveStatus = %q, want crashed", got)
	}
}
