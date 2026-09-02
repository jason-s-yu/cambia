package gates

import (
	"fmt"
	"regexp"
	"time"
)

// Check is one gate's verdict, the wire shape D46 shows inside gate_report's
// checks list: {gate, ok, observed, required, next_eligible_at?, detail}.
type Check struct {
	Gate           string     `json:"gate"`
	OK             bool       `json:"ok"`
	Observed       *float64   `json:"observed,omitempty"`
	Required       *float64   `json:"required,omitempty"`
	NextEligibleAt *time.Time `json:"next_eligible_at,omitempty"`
	Detail         string     `json:"detail,omitempty"`
	// OnBreach is the config's response policy for this gate, echoed even
	// when OK so a caller reading a passing report still knows what would
	// happen on a future breach. Empty for a gate with no on_breach concept
	// (concurrency, devices_allowed).
	OnBreach OnBreach `json:"on_breach,omitempty"`
}

// ObservedDevice mirrors one entry of the report's observed.devices list.
type ObservedDevice struct {
	ID         string   `json:"id"`
	VRAMFreeGB *float64 `json:"vram_free_gb,omitempty"`
	BusyPct    *float64 `json:"busy_pct,omitempty"`
	TempC      *float64 `json:"temp_c,omitempty"`
}

// Observed mirrors D46's gate_report.observed block: measurements the node
// took, verbatim, for an operator or the coordinator to read (never used by
// the coordinator as an admission input, per D11).
type Observed struct {
	RAMFreeGB  *float64         `json:"ram_free_gb,omitempty"`
	DiskFreeGB *float64         `json:"disk_free_gb,omitempty"`
	Load1      *float64         `json:"load1,omitempty"`
	Devices    []ObservedDevice `json:"devices,omitempty"`
}

// Report is the full gate_report a node attaches to every heartbeat, claim,
// and progress tick (D46).
type Report struct {
	EvaluatedAt    time.Time  `json:"evaluated_at"`
	Admit          bool       `json:"admit"`
	SlotsOffered   int        `json:"slots_offered"`
	NextEligibleAt *time.Time `json:"next_eligible_at,omitempty"`
	Checks         []Check    `json:"checks"`
	Observed       Observed   `json:"observed"`
	// DevicesAllowed is the devices_allowed gate's resolved id set (D46: "the
	// gate report echoes the resolved id set, so an operator reading it sees
	// devices rather than a rule").
	DevicesAllowed []string `json:"devices_allowed,omitempty"`
}

// Evaluate is the pure gate evaluator (D46): every check reads only cfg and
// snap, with now as the sole source of wall-clock truth, so it is
// table-testable with no clock, no I/O, and no GPU. admit is the AND of every
// produced check; a nil sub-gate in cfg contributes no checks (an unconfigured
// gate never blocks).
func Evaluate(cfg Config, devices []DeviceRef, snap Snapshot, now time.Time) Report {
	var checks []Check

	if cfg.Drain != nil {
		checks = append(checks, evaluateDrain(*cfg.Drain, snap))
	}
	if len(cfg.Windows) > 0 {
		checks = append(checks, evaluateWindows(cfg.Windows, now))
	}
	if cfg.Floors != nil {
		checks = append(checks, evaluateFloors(*cfg.Floors, snap)...)
	}
	if cfg.Ceilings != nil {
		checks = append(checks, evaluateCeilings(*cfg.Ceilings, snap)...)
	}
	if cfg.Cotenants != nil {
		checks = append(checks, evaluateCotenants(*cfg.Cotenants, snap)...)
	}
	if cfg.Power != nil {
		checks = append(checks, evaluatePower(*cfg.Power, snap)...)
	}
	if cfg.Concurrency != nil {
		checks = append(checks, evaluateConcurrency(*cfg.Concurrency, snap)...)
	}

	admit := true
	var nextEligible *time.Time
	for _, c := range checks {
		if !c.OK {
			admit = false
		}
		if c.NextEligibleAt != nil && (nextEligible == nil || c.NextEligibleAt.Before(*nextEligible)) {
			nextEligible = c.NextEligibleAt
		}
	}

	slotsOffered := 0
	if admit {
		slotsOffered = 1
		if cfg.Concurrency != nil && cfg.Concurrency.MaxSlots != nil {
			slotsOffered = *cfg.Concurrency.MaxSlots - snap.RunningSlots
			if slotsOffered < 0 {
				slotsOffered = 0
			}
		}
	}

	return Report{
		EvaluatedAt:    now,
		Admit:          admit,
		SlotsOffered:   slotsOffered,
		NextEligibleAt: nextEligible,
		Checks:         checks,
		Observed:       buildObserved(snap),
		DevicesAllowed: ResolveDevicesAllowed(cfg.DevicesAllowed, devices),
	}
}

func evaluateDrain(g DrainGate, snap Snapshot) Check {
	if snap.DrainFilePresent {
		return Check{Gate: "drain", OK: false, OnBreach: g.OnBreach, Detail: fmt.Sprintf("drain file present at %s", g.File)}
	}
	return Check{Gate: "drain", OK: true, OnBreach: g.OnBreach, Detail: "no drain file"}
}

func evaluateWindows(windows []Window, now time.Time) Check {
	for _, w := range windows {
		if inWindow(now, w) {
			return Check{Gate: "windows", OK: true, OnBreach: w.OnBreach,
				Detail: fmt.Sprintf("inside %s %s-%s %s", w.From, w.To, w.TZ, w.Days)}
		}
	}
	c := Check{Gate: "windows", OK: false, Detail: "outside every configured window"}
	if next, ob, found := nextWindowOpen(now, windows); found {
		t := next
		c.NextEligibleAt = &t
		c.OnBreach = ob
		c.Detail = fmt.Sprintf("outside every configured window; next opens %s", next.Format(time.RFC3339))
	}
	return c
}

func evaluateFloors(g FloorsGate, snap Snapshot) []Check {
	var out []Check
	if g.FreeRAMGB != nil {
		out = append(out, thresholdCheck("floors.free_ram_gb", snap.RAMFreeGB, g.FreeRAMGB, g.OnBreach, false))
	}
	if g.FreeDiskGB != nil {
		out = append(out, thresholdCheck("floors.free_disk_gb", snap.DiskFreeGB, g.FreeDiskGB, g.OnBreach, false))
	}
	for id, floor := range g.FreeVRAMGB {
		floor := floor
		dev, found := findDevice(snap.Devices, id)
		observed := deviceFact(dev, found, func(d DeviceSnapshot) *float64 { return d.VRAMFreeGB })
		out = append(out, thresholdCheck("floors.free_vram_gb."+id, observed, &floor, g.OnBreach, false))
	}
	return out
}

func evaluateCeilings(g CeilingsGate, snap Snapshot) []Check {
	var out []Check
	if g.Load1PerCore != nil {
		var observed *float64
		if snap.Load1 != nil && snap.Cores > 0 {
			v := *snap.Load1 / float64(snap.Cores)
			observed = &v
		}
		out = append(out, thresholdCheck("ceilings.load1_per_core", observed, g.Load1PerCore, g.OnBreach, true))
	}
	if g.GPUBusyPct != nil {
		for _, d := range snap.Devices {
			if d.BusyPct == nil {
				continue
			}
			out = append(out, thresholdCheck("ceilings.gpu_busy_pct."+d.ID, d.BusyPct, g.GPUBusyPct, g.OnBreach, true))
		}
	}
	if g.GPUTempC != nil {
		for _, d := range snap.Devices {
			if d.TempC == nil {
				continue
			}
			out = append(out, thresholdCheck("ceilings.gpu_temp_c."+d.ID, d.TempC, g.GPUTempC, g.OnBreach, true))
		}
	}
	return out
}

func evaluateCotenants(g CotenantsGate, snap Snapshot) []Check {
	var out []Check
	for id, limit := range g.DenyWhenVRAMHeldByOthersGB {
		limit := limit
		dev, found := findDevice(snap.Devices, id)
		observed := deviceFact(dev, found, func(d DeviceSnapshot) *float64 { return d.VRAMHeldByOthersGB })
		out = append(out, thresholdCheck("cotenants.deny_when_vram_held_by_others_gb."+id, observed, &limit, g.OnBreach, true))
	}
	if len(g.DenyWhenProcessMatches) > 0 {
		matched := ""
		for _, pattern := range g.DenyWhenProcessMatches {
			re, err := regexp.Compile(pattern)
			if err != nil {
				continue
			}
			for _, proc := range snap.ProcessNames {
				if re.MatchString(proc) {
					matched = proc
					break
				}
			}
			if matched != "" {
				break
			}
		}
		if matched != "" {
			out = append(out, Check{Gate: "cotenants.deny_when_process_matches", OK: false, OnBreach: g.OnBreach,
				Detail: fmt.Sprintf("process %q matches a denylisted pattern", matched)})
		} else {
			out = append(out, Check{Gate: "cotenants.deny_when_process_matches", OK: true, OnBreach: g.OnBreach, Detail: "no denylisted process running"})
		}
	}
	return out
}

func evaluatePower(g PowerGate, snap Snapshot) []Check {
	var out []Check
	if g.RequireAC {
		ok := snap.ACPower != nil && *snap.ACPower
		detail := "on AC power"
		if !ok {
			detail = "not on AC power"
		}
		out = append(out, Check{Gate: "power.require_ac", OK: ok, OnBreach: g.OnBreach, Detail: detail})
	}
	if g.MinBatteryPercent != nil {
		out = append(out, thresholdCheck("power.min_battery_percent", snap.BatteryPercent, g.MinBatteryPercent, g.OnBreach, false))
	}
	return out
}

func evaluateConcurrency(g ConcurrencyGate, snap Snapshot) []Check {
	var out []Check
	if g.MaxSlots != nil {
		ok := snap.RunningSlots < *g.MaxSlots
		observed := float64(snap.RunningSlots)
		required := float64(*g.MaxSlots)
		out = append(out, Check{Gate: "concurrency.max_slots", OK: ok, Observed: &observed, Required: &required,
			Detail: fmt.Sprintf("%d running (cap %d)", snap.RunningSlots, *g.MaxSlots)})
	}
	if g.MaxAcceleratorJobs != nil {
		ok := snap.RunningAcceleratorJobs < *g.MaxAcceleratorJobs
		observed := float64(snap.RunningAcceleratorJobs)
		required := float64(*g.MaxAcceleratorJobs)
		out = append(out, Check{Gate: "concurrency.max_accelerator_jobs", OK: ok, Observed: &observed, Required: &required,
			Detail: fmt.Sprintf("%d running (cap %d)", snap.RunningAcceleratorJobs, *g.MaxAcceleratorJobs)})
	}
	return out
}

// thresholdCheck builds a floor or ceiling Check. ceiling=false means observed
// must be >= required (a floor); ceiling=true means observed must be <=
// required. A nil observed (the node never measured that fact) fails closed:
// admission cannot rest on a fact nobody took (mirrors D11's "no admission
// decision may rest on a node-reported number where the coordinator holds an
// independent one" by never inventing one here either).
func thresholdCheck(gate string, observed, required *float64, onBreach OnBreach, ceiling bool) Check {
	c := Check{Gate: gate, Required: required, OnBreach: onBreach, Observed: observed}
	if observed == nil {
		c.OK = false
		c.Detail = "not measured"
		return c
	}
	if ceiling {
		c.OK = *observed <= *required
	} else {
		c.OK = *observed >= *required
	}
	if c.OK {
		c.Detail = fmt.Sprintf("%.2f within %.2f", *observed, *required)
	} else {
		c.Detail = fmt.Sprintf("%.2f breaches %.2f", *observed, *required)
	}
	return c
}

func findDevice(devices []DeviceSnapshot, id string) (DeviceSnapshot, bool) {
	for _, d := range devices {
		if d.ID == id {
			return d, true
		}
	}
	return DeviceSnapshot{}, false
}

func deviceFact(dev DeviceSnapshot, found bool, get func(DeviceSnapshot) *float64) *float64 {
	if !found {
		return nil
	}
	return get(dev)
}

func buildObserved(snap Snapshot) Observed {
	o := Observed{RAMFreeGB: snap.RAMFreeGB, DiskFreeGB: snap.DiskFreeGB, Load1: snap.Load1}
	for _, d := range snap.Devices {
		o.Devices = append(o.Devices, ObservedDevice{ID: d.ID, VRAMFreeGB: d.VRAMFreeGB, BusyPct: d.BusyPct, TempC: d.TempC})
	}
	return o
}
