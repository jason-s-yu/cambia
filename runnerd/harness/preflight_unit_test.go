package harness

import (
	"net/http"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

func TestForceMatrix(t *testing.T) {
	fail := func(name string) procmgr.PreflightCheck {
		return procmgr.PreflightCheck{Name: name, OK: false, Detail: "fail"}
	}
	cases := []struct {
		name     string
		check    procmgr.PreflightCheck
		force    bool
		wantPass bool
	}{
		{"gpu not forced", fail("gpu_vram"), false, false},
		{"gpu forced (overridable)", fail("gpu_vram"), true, true},
		{"disk not forceable", fail("disk_space"), true, false},
		{"ram not forceable", fail("min_free_ram"), true, false},
		{"collision never forceable", fail("name_collision"), true, false},
	}
	for _, c := range cases {
		got, _ := preflightPasses([]procmgr.PreflightCheck{c.check}, c.force)
		if got != c.wantPass {
			t.Errorf("%s: preflightPasses = %v, want %v", c.name, got, c.wantPass)
		}
	}

	// All-ok passes regardless of force.
	ok := []procmgr.PreflightCheck{{Name: "disk_space", OK: true}, {Name: "min_free_ram", OK: true}}
	if pass, _ := preflightPasses(ok, false); !pass {
		t.Error("all-ok should pass")
	}
}

func TestMinFreeRAMCheck(t *testing.T) {
	if c := MinFreeRAMCheck(8, func() (float64, error) { return 16, nil }); !c.OK {
		t.Errorf("16 GiB available >= 8 floor should pass: %+v", c)
	}
	if c := MinFreeRAMCheck(8, func() (float64, error) { return 4, nil }); c.OK {
		t.Errorf("4 GiB available < 8 floor should fail: %+v", c)
	}
	// A query error is a hard block.
	if c := MinFreeRAMCheck(8, func() (float64, error) { return 0, errFake }); c.OK {
		t.Errorf("query error should fail closed: %+v", c)
	}
	// No floor is a pass.
	if c := MinFreeRAMCheck(0, func() (float64, error) { return 0, errFake }); !c.OK {
		t.Errorf("no floor should pass: %+v", c)
	}
}

func TestSubmitDiskFloorNotForceable(t *testing.T) {
	// A disk floor larger than any real free space fails the preflight; force
	// must NOT bypass it (runner force matrix).
	r := newRig(t, rigConfig{minDiskGB: 1e12})

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("disk-1", "fake"))
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("disk fail: got %d, want 412", resp.StatusCode)
	}
	resp.Body.Close()

	spec := baseSpec("disk-2", "fake")
	spec["force"] = true
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("disk fail with force: got %d, want 412 (not forceable)", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestSubmitGPUForceableButSkippedOnCPU(t *testing.T) {
	// device=cuda with a failing GPU query: without force -> 412; with force ->
	// admitted (gpu_vram is the sole runner-overridable check). disk/ram pass.
	lowGPU := func() (string, error) { return "128, 10, TestGPU", nil } // 128 MiB free
	r := newRig(t, rigConfig{gpuQuery: lowGPU, minRAMGB: 1, minDiskGB: 0.001})

	spec := baseSpec("gpu-1", "fake")
	spec["device"] = "cuda"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("gpu fail: got %d, want 412", resp.StatusCode)
	}
	resp.Body.Close()

	spec2 := baseSpec("gpu-2", "fake")
	spec2["device"] = "cuda"
	spec2["force"] = true
	resp = r.do(http.MethodPost, "/harness/jobs", spec2)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("gpu fail with force: got %d, want 201 (gpu forceable)", resp.StatusCode)
	}
	resp.Body.Close()
}

// errFake is a sentinel query error.
var errFake = errTest("boom")

type errTest string

func (e errTest) Error() string { return string(e) }
