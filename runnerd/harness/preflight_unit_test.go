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
	// cuda must be explicitly enabled on the rig -- the capability gate default
	// is cpu-only.
	lowGPU := func() (string, error) { return "128, 10, TestGPU", nil } // 128 MiB free
	r := newRig(t, rigConfig{
		gpuQuery:       lowGPU,
		minRAMGB:       1,
		minDiskGB:      0.001,
		allowedDevices: map[string]bool{"cpu": true, "cuda": true},
	})

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

// TestSubmitInvalidDeviceShape covers the shape check (jobspec.go
// deviceValid): a device outside {cpu,cuda,xpu} is rejected before the
// per-runner capability gate even runs.
func TestSubmitInvalidDeviceShape(t *testing.T) {
	r := newRig(t, rigConfig{})
	spec := baseSpec("bad-device", "fake")
	spec["device"] = "tpu"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("got %d, want 400", resp.StatusCode)
	}
	var body map[string]string
	decodeBody(t, resp, &body)
	if body["error"] != "invalid_device" {
		t.Fatalf("error = %q, want invalid_device", body["error"])
	}
}

// TestSubmitDeviceCapabilityGateBothDirections covers cambia-329's runner
// capability gate on a default (cpu-only) rig: a shape-valid device NOT in
// RUNNERD_ALLOWED_DEVICES is rejected as device_unsupported, structured and
// not forceable, while cpu (the always-allowed default) is admitted normally.
// This is the required end-to-end admission-rejection assertion: a device:xpu
// spec against a default-config test server.
func TestSubmitDeviceCapabilityGateBothDirections(t *testing.T) {
	r := newRig(t, rigConfig{})

	// Disallowed direction: xpu is shape-valid but not enabled on this
	// (cpu-only default) runner.
	spec := baseSpec("xpu-rejected", "fake")
	spec["device"] = "xpu"
	spec["force"] = true // capability gate is not forceable
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("xpu on cpu-only runner: got %d, want 400", resp.StatusCode)
	}
	var body map[string]string
	decodeBody(t, resp, &body)
	if body["error"] != "device_unsupported" {
		t.Fatalf("error = %q, want device_unsupported", body["error"])
	}

	// Allowed direction: cpu is enabled by default and reaches admission.
	resp2 := r.do(http.MethodPost, "/harness/jobs", baseSpec("cpu-allowed", "fake"))
	if resp2.StatusCode != http.StatusCreated {
		t.Fatalf("cpu on cpu-only runner: got %d, want 201", resp2.StatusCode)
	}
	resp2.Body.Close()
}

// TestSubmitXPURenderNodeRequired covers the xpu preflight's hard-required
// render-node check: absent, it fails as xpu_render_node (a name outside the
// runner force matrix, so force=true does not bypass it), distinct from the
// forceable gpu_vram check.
func TestSubmitXPURenderNodeRequired(t *testing.T) {
	noNode := func() ([]string, error) { return nil, nil }
	r := newRig(t, rigConfig{
		allowedDevices: map[string]bool{"cpu": true, "xpu": true},
		renderNodeGlob: noNode,
		minRAMGB:       1,
		minDiskGB:      0.001,
	})

	spec := baseSpec("xpu-no-node", "fake")
	spec["device"] = "xpu"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("no render node: got %d, want 412", resp.StatusCode)
	}
	resp.Body.Close()

	spec2 := baseSpec("xpu-no-node-forced", "fake")
	spec2["device"] = "xpu"
	spec2["force"] = true
	resp2 := r.do(http.MethodPost, "/harness/jobs", spec2)
	if resp2.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("no render node with force: got %d, want 412 (xpu_render_node not forceable)", resp2.StatusCode)
	}
	resp2.Body.Close()
}

// TestSubmitXPUVRAMForceable mirrors TestSubmitGPUForceableButSkippedOnCPU
// for the xpu path: with a render node present and xpu-smi reporting low free
// memory, admission fails as gpu_vram (the forceable check) and force=true
// admits it.
func TestSubmitXPUVRAMForceable(t *testing.T) {
	hasNode := func() ([]string, error) { return []string{"/dev/dri/renderD128"}, nil }
	lowXPU := func() (string, error) { return "free: 128 MiB", nil } // 128 MiB free
	r := newRig(t, rigConfig{
		allowedDevices: map[string]bool{"cpu": true, "xpu": true},
		renderNodeGlob: hasNode,
		xpuQuery:       lowXPU,
		minRAMGB:       1,
		minDiskGB:      0.001,
	})

	spec := baseSpec("xpu-vram-1", "fake")
	spec["device"] = "xpu"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("low xpu vram: got %d, want 412", resp.StatusCode)
	}
	resp.Body.Close()

	spec2 := baseSpec("xpu-vram-2", "fake")
	spec2["device"] = "xpu"
	spec2["force"] = true
	resp2 := r.do(http.MethodPost, "/harness/jobs", spec2)
	if resp2.StatusCode != http.StatusCreated {
		t.Fatalf("low xpu vram with force: got %d, want 201 (gpu_vram forceable)", resp2.StatusCode)
	}
	resp2.Body.Close()
}

// errFake is a sentinel query error.
var errFake = errTest("boom")

type errTest string

func (e errTest) Error() string { return string(e) }
