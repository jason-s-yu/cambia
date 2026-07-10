package harness

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Default preflight floors (design 6). Higher than the dashboard's because the
// runner is a dedicated single-tenant serving host, not a shared dev box: DiskReservoir
// growth (~16G per player pool) drives the 20G disk floor.
const (
	DefaultMinFreeRAMGB  = 8.0
	DefaultMinFreeDiskGB = 20.0
)

// runnerOverridable is the runner force matrix (design 5.6). Only gpu_vram is
// forceable over the API, and it is moot on the runner (cpu-only, so the check is
// skipped entirely). name_collision is never forceable (filesystem identity);
// disk_space and min_free_ram are NOT forceable over the API -- raising those
// floors is an operator action (env change + restart), so a captured token
// cannot ask the daemon to OOM or fill the container.
var runnerOverridable = map[string]bool{
	"gpu_vram": true,
}

// RAMQueryFunc returns the available RAM in GiB. It is a seam so tests inject a
// value without depending on the host's live memory.
type RAMQueryFunc func() (float64, error)

// DefaultRAMQuery reads MemAvailable from /proc/meminfo (lxcfs presents
// container-scoped values inside the runner LXC) and returns it in GiB.
func DefaultRAMQuery() (float64, error) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "MemAvailable:" {
			continue
		}
		kb, perr := strconv.ParseFloat(fields[1], 64)
		if perr != nil {
			return 0, perr
		}
		return kb / (1 << 20), nil // kB -> GiB
	}
	return 0, fmt.Errorf("MemAvailable not found in /proc/meminfo")
}

// MinFreeRAMCheck passes when at least minGB of RAM is available, or when no RAM
// floor is set (minGB <= 0). A query error is a hard block: an unreadable
// /proc/meminfo must not silently launch a run that then OOMs the container.
func MinFreeRAMCheck(minGB float64, query RAMQueryFunc) procmgr.PreflightCheck {
	const name = "min_free_ram"
	if minGB <= 0 {
		return procmgr.PreflightCheck{Name: name, OK: true, Detail: "no RAM requirement"}
	}
	freeGB, err := query()
	if err != nil {
		return procmgr.PreflightCheck{Name: name, OK: false, Detail: fmt.Sprintf("MemAvailable read failed: %v", err)}
	}
	if freeGB >= minGB {
		return procmgr.PreflightCheck{Name: name, OK: true, Detail: fmt.Sprintf("%.1f GiB available (need %.1f)", freeGB, minGB)}
	}
	return procmgr.PreflightCheck{Name: name, OK: false, Detail: fmt.Sprintf("only %.1f GiB available (need %.1f)", freeGB, minGB)}
}

// preflightPasses reports whether the checks permit admission under the runner
// force matrix. It passes when every check is ok, or when force is set and every
// failing check is runner-overridable (only gpu_vram). The second return is the
// failing checks for the 412 body. This intentionally does NOT reuse
// procmgr.PreflightPasses, whose overridable set (gpu_vram, disk_space,
// concurrency_cap) is the dashboard matrix; the runner matrix is stricter.
func preflightPasses(checks []procmgr.PreflightCheck, force bool) (bool, []procmgr.PreflightCheck) {
	var failed []procmgr.PreflightCheck
	for _, c := range checks {
		if !c.OK {
			failed = append(failed, c)
		}
	}
	if len(failed) == 0 {
		return true, nil
	}
	if force {
		for _, c := range failed {
			if !runnerOverridable[c.Name] {
				return false, failed
			}
		}
		return true, failed
	}
	return false, failed
}
