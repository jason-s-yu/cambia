// Package gates holds the nashnet node-owned gate configuration and its pure
// evaluator (D46): windows, floors, ceilings, co-tenant, drain, power, and
// concurrency, evaluated against a probe snapshot with an injected clock. See
// .docs/serving-harness/v1.1-compute-pool-design.md section 2.
//
// Gates are a node's own policy: the coordinator has no gate config for any
// node and no default number for any host (D46). Nothing in this package
// performs I/O; Evaluate is table-testable offline with an injected clock and
// a hand-built Snapshot.
package gates

import "sort"

// OnBreach names the runtime response a gate's config attaches to a breach
// while a job is running (D46): finish lets the job complete and stops
// claiming, drain stops claiming until the gate reopens, stop sends SIGINT
// with the grace period. Evaluate reports which policy applies per breaching
// check; applying it is the dispatcher's job, not this package's.
type OnBreach string

const (
	OnBreachFinish OnBreach = "finish"
	OnBreachDrain  OnBreach = "drain"
	OnBreachStop   OnBreach = "stop"
)

// Config is the node's gates: block (D46). Every sub-gate is a pointer so an
// omitted gate is distinguishable from a configured-but-permissive one, and
// Evaluate skips a nil sub-gate's checks entirely.
type Config struct {
	Drain          *DrainGate       `yaml:"drain,omitempty"`
	Concurrency    *ConcurrencyGate `yaml:"concurrency,omitempty"`
	KindsAllowed   []string         `yaml:"kinds_allowed,omitempty"`
	DevicesAllowed []string         `yaml:"devices_allowed,omitempty"`
	Windows        []Window         `yaml:"windows,omitempty"`
	Floors         *FloorsGate      `yaml:"floors,omitempty"`
	Ceilings       *CeilingsGate    `yaml:"ceilings,omitempty"`
	Cotenants      *CotenantsGate   `yaml:"cotenants,omitempty"`
	Power          *PowerGate       `yaml:"power,omitempty"`
	JobPolicy      *JobPolicyGate   `yaml:"job_policy,omitempty"`
}

// DrainGate refuses further claims while File is present on disk. Evaluate
// reads its presence from Snapshot.DrainFilePresent (the filesystem stat is
// the caller's job, keeping Evaluate I/O-free).
type DrainGate struct {
	File     string   `yaml:"file,omitempty"`
	OnBreach OnBreach `yaml:"on_breach,omitempty"`
}

// ConcurrencyGate caps how many jobs (and how many accelerator-using jobs)
// this node runs at once. It has no on_breach: a full node simply offers no
// slots, it does not preempt what is already running.
type ConcurrencyGate struct {
	MaxSlots           *int `yaml:"max_slots,omitempty"`
	MaxAcceleratorJobs *int `yaml:"max_accelerator_jobs,omitempty"`
}

// Window is one admissible usage interval (D46). Days uses the three-letter
// lowercase names mon..sun; From and To are "HH:MM" in 24h time, with "24:00"
// meaning end-of-day, and To <= From meaning the window crosses midnight (a
// 22:00-07:00 night window). TZ is a fixed IANA zone name resolved through
// time.LoadLocation, so DST transitions are handled by the Go time package
// rather than by manual offset arithmetic.
type Window struct {
	Days     []string `yaml:"days"`
	From     string   `yaml:"from"`
	To       string   `yaml:"to"`
	TZ       string   `yaml:"tz"`
	OnBreach OnBreach `yaml:"on_breach,omitempty"`
}

// FloorsGate refuses admission when free resources drop below a minimum.
// FreeVRAMGB is keyed by device id (e.g. "cuda:0"), matching the id-keyed
// check name floors.free_vram_gb.<id> D46's gate_report example shows.
type FloorsGate struct {
	FreeRAMGB  *float64           `yaml:"free_ram_gb,omitempty"`
	FreeDiskGB *float64           `yaml:"free_disk_gb,omitempty"`
	FreeVRAMGB map[string]float64 `yaml:"free_vram_gb,omitempty"`
	OnBreach   OnBreach           `yaml:"on_breach,omitempty"`
}

// CeilingsGate refuses admission when load or accelerator utilization runs
// too hot. Load1PerCore divides Snapshot.Load1 by Snapshot.Cores; the busy
// and temperature ceilings apply to every declared accelerator device that
// reports the corresponding observed fact.
type CeilingsGate struct {
	Load1PerCore *float64 `yaml:"load1_per_core,omitempty"`
	GPUBusyPct   *float64 `yaml:"gpu_busy_pct,omitempty"`
	GPUTempC     *float64 `yaml:"gpu_temp_c,omitempty"`
	OnBreach     OnBreach `yaml:"on_breach,omitempty"`
}

// CotenantsGate refuses admission when another process already holds
// accelerator memory this node would need, or when a running process name
// matches a denylist pattern (a foreign inference server sharing the box).
type CotenantsGate struct {
	DenyWhenVRAMHeldByOthersGB map[string]float64 `yaml:"deny_when_vram_held_by_others_gb,omitempty"`
	DenyWhenProcessMatches     []string           `yaml:"deny_when_process_matches,omitempty"`
	OnBreach                   OnBreach           `yaml:"on_breach,omitempty"`
}

// PowerGate refuses admission on battery power below a floor, or off AC
// entirely when required (a laptop or UPS-backed node).
type PowerGate struct {
	RequireAC         bool     `yaml:"require_ac,omitempty"`
	MinBatteryPercent *float64 `yaml:"min_battery_percent,omitempty"`
	OnBreach          OnBreach `yaml:"on_breach,omitempty"`
}

// JobPolicyGate is the node's own per-job runtime policy. Evaluate publishes it
// as a passing check carrying its required value rather than testing it, since
// this package sees no job: the coordinator reads the number off the gate
// report, refuses to place a job asking for longer, and lowers the lease
// lifetime cap of D4 to it (D46, D63).
type JobPolicyGate struct {
	MaxRuntimeHours *float64 `yaml:"max_runtime_hours,omitempty"`
}

// DeviceRef is the (id, kind) pair devices_allowed resolves against: the
// node's own declared device inventory. Kept local to this package (rather
// than importing capability.Device) so gates has no dependency on the
// capability package.
type DeviceRef struct {
	ID   string
	Kind string
}

// ResolveDevicesAllowed turns a devices_allowed list of kind and/or id
// entries into the concrete, deduplicated, sorted set of device ids they
// name (D46): a kind entry ("cuda") matches every declared device of that
// kind, an id entry ("cuda:0") matches that one device, and a mixed list is
// legal. A nil or empty patterns list resolves to every declared device
// (the gate is absent, so nothing is restricted).
func ResolveDevicesAllowed(patterns []string, devices []DeviceRef) []string {
	if len(patterns) == 0 {
		out := make([]string, 0, len(devices))
		for _, d := range devices {
			out = append(out, d.ID)
		}
		return sortedUnique(out)
	}
	var out []string
	for _, p := range patterns {
		for _, d := range devices {
			if d.Kind == p || d.ID == p {
				out = append(out, d.ID)
			}
		}
	}
	return sortedUnique(out)
}

func sortedUnique(ids []string) []string {
	seen := make(map[string]bool, len(ids))
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}
