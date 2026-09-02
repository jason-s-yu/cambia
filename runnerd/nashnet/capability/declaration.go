// Package capability holds the nashnet compute-pool capability declaration
// (D9), the job placement requires block and its defaults (D10), the
// grant-caps clamp (D47), and the pure two-sided Match function (D11). See
// .docs/serving-harness/v1.1-compute-pool-design.md section 2.
//
// Every type here is a wire/value type; nothing in this package performs I/O,
// so Match and Clamp are table-testable offline with no injected clock, no
// network, and no GPU.
package capability

import (
	"errors"
	"fmt"
	"math"
)

// ErrInvalidDeclaration is wrapped by Declaration.Validate and Device.Validate
// for a shape or bounds violation the coordinator refuses to store (D47:
// "the coordinator validates shape and bounds... and stores it verbatim").
var ErrInvalidDeclaration = errors.New("invalid_declaration")

// Bounds on a declaration's shape, generous enough for any real node (a
// multi-accelerator box) while capping what a hostile or buggy node can make
// the coordinator store (D47).
const (
	MaxKinds       = 32
	MaxDevices     = 64
	MaxLabels      = 32
	MaxHaveCommits = 256
	MaxShortString = 128
)

// Declaration is the wire shape a node sends at register and refreshes on
// every heartbeat and claim (D9). Numeric device facts the node cannot
// measure are nil, never a fabricated zero; Match and the D47 clamp both rely
// on that distinction to report capability_undeclared instead of comparing
// against a guess.
type Declaration struct {
	Schema            int       `json:"schema"`
	AgentVersion      string    `json:"agent_version"`
	PlatformTag       string    `json:"platform_tag"`
	Slots             int       `json:"slots"`
	Kinds             []string  `json:"kinds"`
	Devices           []Device  `json:"devices"`
	DiskTotalGB       *float64  `json:"disk_total_gb,omitempty"`
	Toolchain         Toolchain `json:"toolchain"`
	CanBuildLibcambia bool      `json:"can_build_libcambia"`
	Labels            []string  `json:"labels,omitempty"`
	HaveCommits       []string  `json:"have_commits,omitempty"`
}

// Device is one compute device a node declared: the single "cpu" entry every
// node ships, plus zero or more accelerator entries (cuda:N, xpu:N). A field
// the probe that built this declaration could not measure is the zero value
// of its pointer (nil), never a defaulted number.
type Device struct {
	ID                string   `json:"id"`
	Kind              string   `json:"kind"` // "cpu", "cuda", or "xpu"
	Cores             *int     `json:"cores,omitempty"`
	Model             string   `json:"model,omitempty"`
	Name              string   `json:"name,omitempty"`
	RAMTotalGB        *float64 `json:"ram_total_gb,omitempty"`
	VRAMTotalGB       *float64 `json:"vram_total_gb,omitempty"`
	ComputeCapability string   `json:"compute_capability,omitempty"`
}

// Toolchain records the node's build tool versions, informational only (no
// Match constraint reads it as of this ticket).
type Toolchain struct {
	Go     string `json:"go,omitempty"`
	UV     string `json:"uv,omitempty"`
	Python string `json:"python,omitempty"`
	Git    string `json:"git,omitempty"`
}

// Validate reports a shape or bounds violation: a schema below 1, a negative
// slots count, an oversized list, an oversized string, or a non-finite or
// negative disk_total_gb. It never fills a missing field; that is Match's
// job via capability_undeclared, not this function's.
func (d Declaration) Validate() error {
	if d.Schema < 1 {
		return fmt.Errorf("%w: schema must be >= 1, got %d", ErrInvalidDeclaration, d.Schema)
	}
	if d.Slots < 0 {
		return fmt.Errorf("%w: slots must be >= 0, got %d", ErrInvalidDeclaration, d.Slots)
	}
	if len(d.Kinds) > MaxKinds {
		return fmt.Errorf("%w: kinds has %d entries, max %d", ErrInvalidDeclaration, len(d.Kinds), MaxKinds)
	}
	if len(d.Devices) > MaxDevices {
		return fmt.Errorf("%w: devices has %d entries, max %d", ErrInvalidDeclaration, len(d.Devices), MaxDevices)
	}
	if len(d.Labels) > MaxLabels {
		return fmt.Errorf("%w: labels has %d entries, max %d", ErrInvalidDeclaration, len(d.Labels), MaxLabels)
	}
	if len(d.HaveCommits) > MaxHaveCommits {
		return fmt.Errorf("%w: have_commits has %d entries, max %d", ErrInvalidDeclaration, len(d.HaveCommits), MaxHaveCommits)
	}
	if len(d.AgentVersion) > MaxShortString {
		return fmt.Errorf("%w: agent_version exceeds %d chars", ErrInvalidDeclaration, MaxShortString)
	}
	if len(d.PlatformTag) > MaxShortString {
		return fmt.Errorf("%w: platform_tag exceeds %d chars", ErrInvalidDeclaration, MaxShortString)
	}
	if d.DiskTotalGB != nil {
		if err := validateFiniteNonNegative(*d.DiskTotalGB, "disk_total_gb"); err != nil {
			return err
		}
	}
	for i, dev := range d.Devices {
		if err := dev.validate(i); err != nil {
			return err
		}
	}
	return nil
}

// validate checks one device entry at index i (used in the error message).
func (dev Device) validate(i int) error {
	if dev.ID == "" {
		return fmt.Errorf("%w: devices[%d].id is empty", ErrInvalidDeclaration, i)
	}
	if len(dev.ID) > MaxShortString || len(dev.Kind) > MaxShortString || len(dev.Model) > MaxShortString || len(dev.Name) > MaxShortString {
		return fmt.Errorf("%w: devices[%d] has a field exceeding %d chars", ErrInvalidDeclaration, i, MaxShortString)
	}
	switch dev.Kind {
	case "cpu", "cuda", "xpu":
	default:
		return fmt.Errorf("%w: devices[%d].kind %q is not one of cpu, cuda, xpu", ErrInvalidDeclaration, i, dev.Kind)
	}
	if dev.Cores != nil {
		if *dev.Cores < 0 {
			return fmt.Errorf("%w: devices[%d].cores must be >= 0", ErrInvalidDeclaration, i)
		}
	}
	if dev.RAMTotalGB != nil {
		if err := validateFiniteNonNegative(*dev.RAMTotalGB, fmt.Sprintf("devices[%d].ram_total_gb", i)); err != nil {
			return err
		}
	}
	if dev.VRAMTotalGB != nil {
		if err := validateFiniteNonNegative(*dev.VRAMTotalGB, fmt.Sprintf("devices[%d].vram_total_gb", i)); err != nil {
			return err
		}
	}
	return nil
}

func validateFiniteNonNegative(v float64, field string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%w: %s must be finite", ErrInvalidDeclaration, field)
	}
	if v < 0 {
		return fmt.Errorf("%w: %s must be >= 0", ErrInvalidDeclaration, field)
	}
	return nil
}

// devicesOfKind returns every declared device of the given kind, in
// declaration order.
func devicesOfKind(devices []Device, kind string) []Device {
	var out []Device
	for _, d := range devices {
		if d.Kind == kind {
			out = append(out, d)
		}
	}
	return out
}

// deviceOfKind returns the first declared device of the given kind. Every
// schema example ships exactly one "cpu" entry, so callers reading
// core/RAM facts (which live only on that entry) use this rather than
// devicesOfKind.
func deviceOfKind(devices []Device, kind string) (Device, bool) {
	for _, d := range devices {
		if d.Kind == kind {
			return d, true
		}
	}
	return Device{}, false
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func intersects(a, b []string) bool {
	for _, v := range a {
		if containsString(b, v) {
			return true
		}
	}
	return false
}
