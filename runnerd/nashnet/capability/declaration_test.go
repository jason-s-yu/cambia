package capability

import (
	"errors"
	"math"
	"testing"
)

func f(v float64) *float64 { return &v }
func i(v int) *int         { return &v }

func validCPUDevice() Device {
	return Device{ID: "cpu", Kind: "cpu", Cores: i(32), RAMTotalGB: f(62.0)}
}

func TestDeclarationValidate(t *testing.T) {
	base := Declaration{Schema: 1, Devices: []Device{validCPUDevice()}}

	if err := base.Validate(); err != nil {
		t.Fatalf("base declaration should validate: %v", err)
	}

	cases := []struct {
		name string
		mut  func(d Declaration) Declaration
	}{
		{"schema below 1", func(d Declaration) Declaration { d.Schema = 0; return d }},
		{"negative slots", func(d Declaration) Declaration { d.Slots = -1; return d }},
		{"too many kinds", func(d Declaration) Declaration {
			for n := 0; n <= MaxKinds; n++ {
				d.Kinds = append(d.Kinds, "train")
			}
			return d
		}},
		{"nan disk_total_gb", func(d Declaration) Declaration { d.DiskTotalGB = f(math.NaN()); return d }},
		{"negative disk_total_gb", func(d Declaration) Declaration { d.DiskTotalGB = f(-1); return d }},
		{"device with empty id", func(d Declaration) Declaration {
			d.Devices = append(d.Devices, Device{Kind: "cuda"})
			return d
		}},
		{"device with bad kind", func(d Declaration) Declaration {
			d.Devices = append(d.Devices, Device{ID: "tpu:0", Kind: "tpu"})
			return d
		}},
		{"device with negative cores", func(d Declaration) Declaration {
			d.Devices = append(d.Devices, Device{ID: "cpu2", Kind: "cpu", Cores: i(-1)})
			return d
		}},
		{"device with inf vram", func(d Declaration) Declaration {
			d.Devices = append(d.Devices, Device{ID: "cuda:0", Kind: "cuda", VRAMTotalGB: f(math.Inf(1))})
			return d
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := c.mut(base)
			err := d.Validate()
			if err == nil {
				t.Fatalf("expected an error")
			}
			if !errors.Is(err, ErrInvalidDeclaration) {
				t.Fatalf("error %v does not wrap ErrInvalidDeclaration", err)
			}
		})
	}
}

// TestDeviceFactOmittedNeverDefaulted covers AC(6): an XPU node whose probe
// cannot measure total VRAM declares no vram_total_gb at all, not a zero, and
// that declaration validates (an absent optional fact is not malformed).
func TestDeviceFactOmittedNeverDefaulted(t *testing.T) {
	d := Declaration{
		Schema: 1,
		Kinds:  []string{"train"},
		Devices: []Device{
			validCPUDevice(),
			{ID: "xpu:0", Kind: "xpu"}, // no VRAMTotalGB: the probe could not measure it
		},
	}
	if err := d.Validate(); err != nil {
		t.Fatalf("declaration with an omitted xpu vram_total_gb should validate: %v", err)
	}
	xpu, ok := deviceOfKind(d.Devices, "xpu")
	if !ok {
		t.Fatal("expected an xpu device")
	}
	if xpu.VRAMTotalGB != nil {
		t.Fatalf("VRAMTotalGB = %v, want nil (never defaulted)", *xpu.VRAMTotalGB)
	}
}
