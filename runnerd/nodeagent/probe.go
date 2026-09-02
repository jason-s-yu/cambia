package nodeagent

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/sysprobe"
)

// Observation is one measurement pass over the host: everything the gate
// evaluator reads plus the totals the capability declaration reports. A fact
// the probe could not measure is nil, never a fabricated zero (D9), so a
// requires constraint over it fails capability_undeclared on the coordinator
// rather than matching against a guess.
type Observation struct {
	Cores          int
	RAMFreeGB      *float64
	RAMTotalGB     *float64
	DiskFreeGB     *float64
	DiskTotalGB    *float64
	Load1          *float64
	Devices        []gates.DeviceSnapshot
	ACPower        *bool
	BatteryPercent *float64
	ProcessNames   []string
}

// Prober measures the host. It is an interface so the agent's gate evaluation,
// declaration assembly, and progress ticks are all testable with a fixed
// observation and no GPU, no /proc, and no nvidia-smi.
type Prober interface {
	Observe(runsDir string) Observation
}

// HostProber is the production Prober. Each query is a field so a test can
// replace one probe without stubbing the rest, and so a host missing
// nvidia-smi contributes no accelerator devices rather than failing.
type HostProber struct {
	RAMAvailable func() (float64, error)
	RAMTotal     func() (float64, bool)
	DiskFree     func(string) float64
	DiskTotal    func(string) (float64, bool)
	Load1        func() (float64, bool)
	GPU          gates.GPUQueryFunc
	XPU          gates.XPUQueryFunc
	Cores        int
}

// NewHostProber returns the default probe set: available RAM and free disk
// from runnerd/sysprobe, the accelerator rows from the nashnet gate probes,
// and the totals from /proc/meminfo and statfs. sysprobe exposes only the free
// and available halves, so the totals the declaration needs are measured here.
func NewHostProber() *HostProber {
	return &HostProber{
		RAMAvailable: sysprobe.DefaultRAMQuery,
		RAMTotal:     ramTotalGB,
		DiskFree:     sysprobe.DiskFreeGB,
		DiskTotal:    diskTotalGB,
		Load1:        load1,
		GPU:          gates.DefaultGPUQuery,
		XPU:          gates.DefaultXPUQuery,
		Cores:        runtime.NumCPU(),
	}
}

// Observe takes one measurement pass. runsDir is the filesystem whose free
// space the disk floor and the declaration report, because that is where every
// staged artifact of a job lands.
func (p *HostProber) Observe(runsDir string) Observation {
	obs := Observation{Cores: p.Cores}
	if p.RAMAvailable != nil {
		if v, err := p.RAMAvailable(); err == nil {
			obs.RAMFreeGB = &v
		}
	}
	if p.RAMTotal != nil {
		if v, ok := p.RAMTotal(); ok {
			obs.RAMTotalGB = &v
		}
	}
	if p.DiskFree != nil {
		v := p.DiskFree(runsDir)
		obs.DiskFreeGB = &v
	}
	if p.DiskTotal != nil {
		if v, ok := p.DiskTotal(runsDir); ok {
			obs.DiskTotalGB = &v
		}
	}
	if p.Load1 != nil {
		if v, ok := p.Load1(); ok {
			obs.Load1 = &v
		}
	}
	obs.Devices = append(obs.Devices, cpuDevice())
	obs.Devices = append(obs.Devices, p.accelerators()...)
	return obs
}

// accelerators returns the cuda and xpu rows the accelerator probes report. A
// probe that is absent or fails contributes nothing: a node with no nvidia-smi
// declares no cuda device rather than declaring one with unknown facts.
func (p *HostProber) accelerators() []gates.DeviceSnapshot {
	var out []gates.DeviceSnapshot
	if p.GPU != nil {
		if csv, err := p.GPU(); err == nil {
			out = append(out, gates.ParseGPURows(csv)...)
		}
	}
	if p.XPU != nil {
		if raw, err := p.XPU(); err == nil {
			if dev, ok := gates.ParseXPURow(raw); ok {
				out = append(out, dev)
			}
		}
	}
	return out
}

// cpuDevice is the single cpu entry every node ships (D9). Its cores and
// ram_total_gb are the host facts, so a min_cores or min_ram_gb requirement
// has something declared to match against.
func cpuDevice() gates.DeviceSnapshot {
	return gates.DeviceSnapshot{ID: "cpu", Kind: "cpu"}
}

// ramTotalGB reads MemTotal from /proc/meminfo. sysprobe reads MemAvailable
// from the same file; the declaration reports the total and the gate report
// the available half, so both are measured.
func ramTotalGB() (float64, bool) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			continue
		}
		kb, perr := strconv.ParseFloat(fields[1], 64)
		if perr != nil {
			return 0, false
		}
		return kb / (1 << 20), true
	}
	return 0, false
}

// diskTotalGB returns the total size of the filesystem backing path, the
// declaration's disk_total_gb. sysprobe.DiskFreeGB reports the available half
// of the same statfs.
func diskTotalGB(path string) (float64, bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, false
	}
	return float64(st.Blocks*uint64(st.Bsize)) / (1 << 30), true
}

// load1 reads the one-minute load average, the ceilings.load1_per_core input.
func load1() (float64, bool) {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, false
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
