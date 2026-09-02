package capability

import (
	"errors"
	"testing"
)

// TestRequiresNormalizeDefaults covers AC(2): defaults derived from device
// reproduce today's behavior for a spec with no requires, and
// needs_libcambia defaults true for every kind.
func TestRequiresNormalizeDefaults(t *testing.T) {
	out := Requires{}.Normalize("cpu")
	if out.Device != "cpu" {
		t.Errorf("Device = %q, want cpu (from spec.device())", out.Device)
	}
	if out.MinCores != 1 {
		t.Errorf("MinCores = %d, want 1", out.MinCores)
	}
	if out.NeedsLibcambia == nil || !*out.NeedsLibcambia {
		t.Errorf("NeedsLibcambia = %v, want true", out.NeedsLibcambia)
	}

	for _, kind := range []string{"train", "evaluate", "head-to-head", "bench", "measure"} {
		r := Requires{}.Normalize("cuda")
		if r.NeedsLibcambia == nil || !*r.NeedsLibcambia {
			t.Errorf("kind %s: NeedsLibcambia default is not true", kind)
		}
	}

	// An explicit device is never overwritten by the spec default.
	explicit := Requires{Device: "cuda"}.Normalize("cpu")
	if explicit.Device != "cuda" {
		t.Errorf("Device = %q, want cuda (explicit wins over spec default)", explicit.Device)
	}

	// An explicit needs_libcambia: false survives normalization.
	no := false
	explicitFalse := Requires{NeedsLibcambia: &no}.Normalize("cpu")
	if explicitFalse.NeedsLibcambia == nil || *explicitFalse.NeedsLibcambia {
		t.Errorf("explicit needs_libcambia:false was overwritten")
	}
}

func TestRequiresValidateMalformed(t *testing.T) {
	cases := []struct {
		name string
		r    Requires
	}{
		{"unrecognized device", Requires{Device: "tpu"}},
		{"min_vram_gb on cpu", Requires{Device: "cpu", MinVRAMGB: f(8)}},
		{"negative min_vram_gb", Requires{Device: "cuda", MinVRAMGB: f(-1)}},
		{"negative min_cores", Requires{Device: "cpu", MinCores: -1}},
		{"negative min_ram_gb", Requires{Device: "cpu", MinRAMGB: f(-1)}},
		{"negative min_disk_gb", Requires{Device: "cpu", MinDiskGB: f(-1)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := c.r.Normalize(c.r.Device)
			err := r.Validate()
			if err == nil {
				t.Fatalf("expected an error")
			}
			if !errors.Is(err, ErrInvalidRequires) {
				t.Fatalf("error %v does not wrap ErrInvalidRequires", err)
			}
		})
	}
}

func TestRequiresValidateWellFormed(t *testing.T) {
	cases := []Requires{
		{},
		{Device: "cuda", MinVRAMGB: f(8)},
		{Device: "xpu", MinVRAMGB: f(4)},
		{Device: "cpu", MinCores: 8, MinRAMGB: f(8), MinDiskGB: f(20)},
	}
	for _, r := range cases {
		norm := r.Normalize("cpu")
		if err := norm.Validate(); err != nil {
			t.Errorf("Requires %+v should validate, got %v", r, err)
		}
	}
}
