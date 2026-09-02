package gates

// DeviceSnapshot is one device's observed facts at evaluation time. A fact
// the probe that built it could not measure is nil, never a fabricated zero
// (D9's rule applies equally to the gate report's observed facts).
type DeviceSnapshot struct {
	ID          string
	Kind        string // "cpu", "cuda", or "xpu"
	Name        string
	VRAMFreeGB  *float64
	VRAMTotalGB *float64
	BusyPct     *float64
	TempC       *float64
	// VRAMHeldByOthersGB is the co-tenant fact cotenants.deny_when_vram_held_by_others_gb
	// reads: VRAM in use by a process this node does not attribute to its own
	// jobs. Distinct from VRAMFreeGB (this node's own headroom); populated by
	// whatever probe computes per-process attribution, which is out of this
	// ticket's scope (the accelerator probe here only extends the per-device
	// query fields, D46). Left nil by probes.go.
	VRAMHeldByOthersGB *float64
}

// Snapshot is everything Evaluate reads about the node's current state: the
// probe results plus admission-relevant counts. Every field a probe could
// not measure or a caller did not supply is the zero value, and Evaluate
// skips the corresponding check rather than treating it as a breach, so a
// gate whose config is unset never fires.
type Snapshot struct {
	RAMFreeGB      *float64
	DiskFreeGB     *float64
	Load1          *float64
	Cores          int // for ceilings.load1_per_core; 0 means unknown, so that ceiling is skipped
	Devices        []DeviceSnapshot
	ACPower        *bool
	BatteryPercent *float64
	// ProcessNames is the running process list cotenants.deny_when_process_matches
	// matches its regex patterns against.
	ProcessNames []string
	// DrainFilePresent is whether the drain gate's configured file exists;
	// the stat is the caller's job, keeping Evaluate free of filesystem I/O.
	DrainFilePresent bool
	// RunningSlots and RunningAcceleratorJobs are this node's own live-job
	// counts, compared against ConcurrencyGate's ceilings.
	RunningSlots           int
	RunningAcceleratorJobs int
}
