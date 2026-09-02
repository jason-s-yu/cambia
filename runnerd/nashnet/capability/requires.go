package capability

import (
	"errors"
	"fmt"
	"math"
)

// ErrInvalidRequires is wrapped by Requires.Validate for a semantically
// malformed requires block (D10: "a malformed block is 400 invalid_requires").
var ErrInvalidRequires = errors.New("invalid_requires")

// Requires is a job's placement constraint block (D10). A zero value places
// with the spec's defaults applied by Normalize: device from the job's
// resolved device, min_cores 1, needs_libcambia true.
type Requires struct {
	Device         string   `json:"device,omitempty"`
	MinVRAMGB      *float64 `json:"min_vram_gb,omitempty"`
	MinCores       int      `json:"min_cores,omitempty"`
	MinRAMGB       *float64 `json:"min_ram_gb,omitempty"`
	MinDiskGB      *float64 `json:"min_disk_gb,omitempty"`
	NeedsLibcambia *bool    `json:"needs_libcambia,omitempty"`
	Node           string   `json:"node,omitempty"`
	LabelsAny      []string `json:"labels_any,omitempty"`
}

// Normalize returns r with every D10 default filled in: Device from
// specDevice (the job's resolved JobSpec.device(), jobspec.go:155) when unset,
// MinCores 1 when unset, and NeedsLibcambia true for every kind, because
// Prepare's ensureLibcambia call is unconditional (ingest.go:205) with no kind
// switch before it. Call this before Match or Validate.
func (r Requires) Normalize(specDevice string) Requires {
	out := r
	if out.Device == "" {
		out.Device = specDevice
	}
	if out.MinCores == 0 {
		out.MinCores = 1
	}
	if out.NeedsLibcambia == nil {
		t := true
		out.NeedsLibcambia = &t
	}
	return out
}

// Validate reports a malformed requires block: an unrecognized device, a
// negative or non-finite floor, or min_vram_gb set against a device kind that
// never carries VRAM (cpu). Call after Normalize so Device is never empty.
func (r Requires) Validate() error {
	switch r.Device {
	case "cpu", "cuda", "xpu":
	default:
		return fmt.Errorf("%w: device %q is not one of cpu, cuda, xpu", ErrInvalidRequires, r.Device)
	}
	if r.MinVRAMGB != nil {
		if r.Device != "cuda" && r.Device != "xpu" {
			return fmt.Errorf("%w: min_vram_gb requires device cuda or xpu, got %q", ErrInvalidRequires, r.Device)
		}
		if err := validateFiniteNonNegativeAs(*r.MinVRAMGB, "min_vram_gb"); err != nil {
			return err
		}
	}
	if r.MinCores < 0 {
		return fmt.Errorf("%w: min_cores must be >= 0", ErrInvalidRequires)
	}
	if r.MinRAMGB != nil {
		if err := validateFiniteNonNegativeAs(*r.MinRAMGB, "min_ram_gb"); err != nil {
			return err
		}
	}
	if r.MinDiskGB != nil {
		if err := validateFiniteNonNegativeAs(*r.MinDiskGB, "min_disk_gb"); err != nil {
			return err
		}
	}
	return nil
}

// validateFiniteNonNegativeAs mirrors validateFiniteNonNegative but wraps
// ErrInvalidRequires instead of ErrInvalidDeclaration, since the two error
// classes map to different HTTP reasons (400 invalid_requires vs the
// declaration-store validator).
func validateFiniteNonNegativeAs(v float64, field string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%w: %s must be finite", ErrInvalidRequires, field)
	}
	if v < 0 {
		return fmt.Errorf("%w: %s must be >= 0", ErrInvalidRequires, field)
	}
	return nil
}
