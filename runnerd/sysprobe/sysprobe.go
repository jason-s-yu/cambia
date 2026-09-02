// Package sysprobe holds the host measurement primitives shared by the
// dashboard preflight rails (runnerd/harness) and the nashnet node gate
// evaluator (runnerd/nashnet/gates): available RAM and free disk. It imports
// no HTTP code so a node-side package can probe the host without pulling in
// the control-plane server.
//
// RAM and disk probing moved here from runnerd/harness/preflight.go and
// runnerd/harness/server.go verbatim (design .docs/serving-harness/v1.1-compute-pool-design.md
// D46); harness keeps thin forwarders under the old names so its existing
// callers and tests are unaffected. The GPU and XPU probes stay in
// runnerd/procmgr/preflight.go: GPUVRAMCheck is the submit-time boolean and is
// untouched, and the nashnet accelerator probe (runnerd/nashnet/gates/probes.go)
// is new code with a wider field list, not a reuse of this package.
package sysprobe

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// RAMQueryFunc returns the available RAM in GiB. It is a seam so tests inject
// a value without depending on the host's live memory.
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

// DiskFreeGB returns the unprivileged-available space in GiB on the
// filesystem backing path (Bavail, matching procmgr.DiskSpaceCheck
// semantics). A statfs failure reads as 0 free, same as the harness original.
func DiskFreeGB(path string) float64 {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0
	}
	return float64(st.Bavail*uint64(st.Bsize)) / (1 << 30)
}
