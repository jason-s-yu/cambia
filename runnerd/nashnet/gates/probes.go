package gates

import (
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// GPUQueryFunc returns the raw CSV output of an nvidia-smi query over every
// GPU row. It is a seam so tests inject fixture output without touching a
// real GPU.
type GPUQueryFunc func() (string, error)

// DefaultGPUQuery is new code, not a reuse of procmgr.DefaultGPUQuery
// (runnerd/procmgr/preflight.go:52-57): that query reads
// memory.free,utilization.gpu,name and GPUVRAMCheck (preflight.go:63) parses
// only fields[0] and fields[2] of the first row, staying the untouched
// submit-time boolean. The gate report's ceilings.gpu_temp_c needs
// temperature.gpu and D9's per-device vram_total_gb needs memory.total,
// neither of which that query requests, so this queries the wider field list
// and ParseGPURows below parses every row into per-device facts.
func DefaultGPUQuery() (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.free,memory.total,utilization.gpu,temperature.gpu,name",
		"--format=csv,noheader,nounits").Output()
	return string(out), err
}

// ParseGPURows parses DefaultGPUQuery's CSV output into one DeviceSnapshot
// per row, ids assigned "cuda:<row index>" in nvidia-smi's own device
// enumeration order (nvidia-smi does not need an explicit index column
// requested to emit rows index-ordered). A field a row does not carry (a
// short or malformed line) leaves that fact nil rather than a fabricated
// zero, matching D9's rule that an unmeasured device fact is omitted, never
// defaulted.
func ParseGPURows(csv string) []DeviceSnapshot {
	var out []DeviceSnapshot
	idx := 0
	for _, line := range strings.Split(csv, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		dev := DeviceSnapshot{ID: fmt.Sprintf("cuda:%d", idx), Kind: "cuda"}
		if v, ok := parseMiBToGB(fields, 0); ok {
			dev.VRAMFreeGB = v
		}
		if v, ok := parseMiBToGB(fields, 1); ok {
			dev.VRAMTotalGB = v
		}
		if v, ok := parseFloatField(fields, 2); ok {
			dev.BusyPct = v
		}
		if v, ok := parseFloatField(fields, 3); ok {
			dev.TempC = v
		}
		if len(fields) > 4 && fields[4] != "" {
			dev.Name = fields[4]
		}
		out = append(out, dev)
		idx++
	}
	return out
}

func parseFloatField(fields []string, i int) (*float64, bool) {
	if i >= len(fields) {
		return nil, false
	}
	v, err := strconv.ParseFloat(fields[i], 64)
	if err != nil {
		return nil, false
	}
	return &v, true
}

func parseMiBToGB(fields []string, i int) (*float64, bool) {
	v, ok := parseFloatField(fields, i)
	if !ok {
		return nil, false
	}
	gb := *v / 1024.0
	return &gb, true
}

// XPUQueryFunc mirrors GPUQueryFunc for xpu-smi's output.
type XPUQueryFunc func() (string, error)

// DefaultXPUQuery runs the same command as procmgr.DefaultXPUQuery
// (runnerd/procmgr/preflight.go:137-140): xpu-smi's per-device memory query
// for device 0. Duplicated here rather than imported because
// xpuFreeMiBPattern below mirrors procmgr's private pattern of the same
// name (preflight.go:128), which is unexported and so not reusable across
// packages; keeping the query alongside its parser avoids splitting one
// probe's two halves across an import.
func DefaultXPUQuery() (string, error) {
	out, err := exec.Command("xpu-smi", "stats", "-d", "0", "-j").Output()
	return string(out), err
}

// xpuFreeMiBPattern extracts the first numeric token following a "free"
// marker (case-insensitive), the same pattern and known limits as
// procmgr's xpuFreeMiBPattern: unverified against real Intel Arc hardware
// (preflight.go:131-136).
var xpuFreeMiBPattern = regexp.MustCompile(`(?i)free[^0-9]*([0-9]+(?:\.[0-9]+)?)`)

// ParseXPURow parses DefaultXPUQuery's output into a single device-0
// DeviceSnapshot (the probe's known limit: one device, free memory only).
// VRAMTotalGB, BusyPct, and TempC are never populated -- xpu-smi's queried
// output carries none of them at this probe's field set, so those facts stay
// nil rather than a fabricated zero (D9; AC(6)'s XPU-node case).
func ParseXPURow(out string) (DeviceSnapshot, bool) {
	m := xpuFreeMiBPattern.FindStringSubmatch(out)
	if m == nil {
		return DeviceSnapshot{}, false
	}
	freeMB, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return DeviceSnapshot{}, false
	}
	freeGB := freeMB / 1024.0
	return DeviceSnapshot{ID: "xpu:0", Kind: "xpu", VRAMFreeGB: &freeGB}, true
}
