package nodeagent

import (
	"os/exec"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
)

// GoToolchainPin is the Go toolchain the libcambia build is pinned to. It
// mirrors the unexported goToolchainPin of runnerd/ingest so the declaration
// probe asks about the same toolchain the build would use; a drift between
// the two is a wrong can_build_libcambia, which is exactly what D63's breaker
// exists to recover from.
const GoToolchainPin = "go1.26.0"

// running is the node's own live-job accounting, the input to the concurrency
// gate and to the slots_free the claim advertises.
type running struct {
	slots           int
	acceleratorJobs int
}

// ExecRunFunc is the production capability.RunFunc: it runs the command with
// the given extra environment and returns its combined output.
func ExecRunFunc(name string, args []string, env []string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	if len(env) > 0 {
		cmd.Env = append(cmd.Environ(), env...)
	}
	return cmd.CombinedOutput()
}

// buildDeclaration assembles the host-agnostic capability declaration (D9)
// from one observation. It declares facts and never a role: no host name, no
// default table, and no fact the probe could not measure.
func buildDeclaration(cfg Config, obs Observation, canBuildLibcambia bool, haveCommits []string) capability.Declaration {
	devices := make([]capability.Device, 0, len(obs.Devices))
	for _, d := range obs.Devices {
		dev := capability.Device{ID: d.ID, Kind: d.Kind, Name: d.Name}
		switch d.Kind {
		case "cpu":
			if obs.Cores > 0 {
				cores := obs.Cores
				dev.Cores = &cores
			}
			dev.RAMTotalGB = obs.RAMTotalGB
		default:
			dev.VRAMTotalGB = d.VRAMTotalGB
		}
		devices = append(devices, dev)
	}
	return capability.Declaration{
		Schema:            1,
		AgentVersion:      cfg.AgentVersion,
		PlatformTag:       cfg.PlatformTag,
		Slots:             cfg.Slots,
		Kinds:             append([]string(nil), cfg.Kinds...),
		Devices:           devices,
		DiskTotalGB:       obs.DiskTotalGB,
		Toolchain:         probeToolchain(),
		CanBuildLibcambia: canBuildLibcambia,
		Labels:            append([]string(nil), cfg.Labels...),
		HaveCommits:       haveCommits,
	}
}

// probeToolchain reports the node's build tool versions. Every field is
// informational: no Match constraint reads one, so a tool that is absent is
// left empty rather than failing the declaration.
func probeToolchain() capability.Toolchain {
	return capability.Toolchain{
		Go:     firstLine(ExecRunFunc("go", []string{"version"}, nil)),
		UV:     firstLine(ExecRunFunc("uv", []string{"--version"}, nil)),
		Python: firstLine(ExecRunFunc("python3", []string{"--version"}, nil)),
		Git:    firstLine(ExecRunFunc("git", []string{"--version"}, nil)),
	}
}

// firstLine trims a version banner to its first line, or "" on any failure.
func firstLine(out []byte, err error) string {
	if err != nil {
		return ""
	}
	s := string(out)
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' || s[i] == '\r' {
			return trimSpace(s[:i])
		}
	}
	return trimSpace(s)
}

func trimSpace(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}

// deviceRefs is the (id, kind) inventory devices_allowed resolves against.
func deviceRefs(obs Observation) []gates.DeviceRef {
	refs := make([]gates.DeviceRef, 0, len(obs.Devices))
	for _, d := range obs.Devices {
		refs = append(refs, gates.DeviceRef{ID: d.ID, Kind: d.Kind})
	}
	return refs
}

// gateSnapshot turns an observation plus the node's own live-job counts into
// the pure evaluator's input.
func gateSnapshot(cfg Config, obs Observation, run running) gates.Snapshot {
	snap := gates.Snapshot{
		RAMFreeGB:              obs.RAMFreeGB,
		DiskFreeGB:             obs.DiskFreeGB,
		Load1:                  obs.Load1,
		Cores:                  obs.Cores,
		Devices:                obs.Devices,
		ACPower:                obs.ACPower,
		BatteryPercent:         obs.BatteryPercent,
		ProcessNames:           obs.ProcessNames,
		RunningSlots:           run.slots,
		RunningAcceleratorJobs: run.acceleratorJobs,
	}
	if cfg.Gates.Drain != nil && cfg.Gates.Drain.File != "" {
		snap.DrainFilePresent = fileExists(cfg.Gates.Drain.File)
	}
	return snap
}

// evaluateGates is the node's gate pass: one observation, one pure Evaluate.
// It runs at the three points of D46 (before every claim, immediately before
// Prepare, and on every progress tick while running).
func evaluateGates(cfg Config, prober Prober, run running, now time.Time) (gates.Report, Observation) {
	obs := prober.Observe(cfg.RunsDir)
	return gates.Evaluate(cfg.Gates, deviceRefs(obs), gateSnapshot(cfg, obs, run), now), obs
}

// failingChecks returns the checks that did not pass, in report order.
func failingChecks(rep gates.Report) []gates.Check {
	var out []gates.Check
	for _, c := range rep.Checks {
		if !c.OK {
			out = append(out, c)
		}
	}
	return out
}

// breachCooldown is the nack cooldown for a gate breach found before Prepare
// (D8): min(next_eligible_at - now, 3600), falling back to the default nack
// cooldown when the gate names no reopen time.
func breachCooldown(rep gates.Report, now time.Time) int {
	const maxCooldown = 3600
	if rep.NextEligibleAt == nil {
		return defaultNackCooldownSeconds
	}
	d := int(rep.NextEligibleAt.Sub(now) / time.Second)
	if d <= 0 {
		return defaultNackCooldownSeconds
	}
	if d > maxCooldown {
		return maxCooldown
	}
	return d
}

// defaultNackCooldownSeconds is the D8 default: a nacked job is not re-matched
// to this node for five minutes.
const defaultNackCooldownSeconds = 300

// prepareFailedCooldownSeconds is the D63 cooldown for a node-attributable
// Prepare failure.
const prepareFailedCooldownSeconds = 600
