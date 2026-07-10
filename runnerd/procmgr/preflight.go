package procmgr

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
)

// PreflightCheck is one gate evaluated before a training run is started or
// resumed. Name is one of "gpu_vram", "disk_space", "name_collision",
// "concurrency_cap". It marshals to the pinned wire shape
// {name, ok, detail} consumed by the frontend.
type PreflightCheck struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// overridableChecks names the preflight checks a caller may bypass with
// force=true. Name validation and name_collision are never overridable: they
// guard filesystem identity and the path-traversal boundary.
var overridableChecks = map[string]bool{
	"gpu_vram":        true,
	"disk_space":      true,
	"concurrency_cap": true,
}

// DefaultMinVRAMGB and DefaultMinDiskGB are the contract's baseline preflight
// thresholds (.docs/dashboard/phase2-process-mgmt/contract.md). NewProcessHandlers
// applies them whenever its config leaves MinVRAMGB/MinDiskGB <= 0 (the zero
// value), so the GPU/disk rails are on by default rather than silently
// disabled by an unset config field.
const (
	DefaultMinVRAMGB = 4.0
	DefaultMinDiskGB = 5.0
)

// GPUQueryFunc returns the raw CSV output of an nvidia-smi VRAM query. It is a
// seam so tests can inject fake output (or an exec.ErrNotFound to simulate a
// CPU host) without touching a real GPU.
type GPUQueryFunc func() (string, error)

// DefaultGPUQuery runs nvidia-smi for per-GPU free memory, utilization, and
// name. A missing binary yields an *exec.Error wrapping exec.ErrNotFound, which
// GPUVRAMCheck treats as a CPU host rather than a failure.
func DefaultGPUQuery() (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.free,utilization.gpu,name",
		"--format=csv,noheader,nounits").Output()
	return string(out), err
}

// GPUVRAMCheck passes when the first GPU reports at least minGB free, when no
// VRAM is required (minGB <= 0), or when nvidia-smi is absent (CPU host). A
// present-but-failing nvidia-smi is a hard block so a misconfigured GPU host
// does not silently launch under contention.
func GPUVRAMCheck(minGB float64, query GPUQueryFunc) PreflightCheck {
	const name = "gpu_vram"
	if minGB <= 0 {
		return PreflightCheck{name, true, "no VRAM requirement"}
	}
	out, err := query()
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return PreflightCheck{name, true, "no nvidia-smi (CPU host)"}
		}
		return PreflightCheck{name, false, fmt.Sprintf("nvidia-smi failed: %v", err)}
	}
	line := firstNonEmptyLine(out)
	if line == "" {
		return PreflightCheck{name, false, "nvidia-smi returned no GPU rows"}
	}
	fields := strings.Split(line, ",")
	freeMB, perr := strconv.ParseFloat(strings.TrimSpace(fields[0]), 64)
	if perr != nil {
		return PreflightCheck{name, false, fmt.Sprintf("unparseable memory.free %q", strings.TrimSpace(fields[0]))}
	}
	freeGB := freeMB / 1024.0
	gpuName := "GPU0"
	if len(fields) >= 3 {
		gpuName = strings.TrimSpace(fields[2])
	}
	if freeGB >= minGB {
		return PreflightCheck{name, true, fmt.Sprintf("%.1f GiB free on %s (need %.1f)", freeGB, gpuName, minGB)}
	}
	return PreflightCheck{name, false, fmt.Sprintf("only %.1f GiB free on %s (need %.1f)", freeGB, gpuName, minGB)}
}

// RenderNodeGlobFunc lists Intel GPU render node device paths under /dev/dri
// (design cambia-329). It is a seam so tests inject fake results without
// touching real devices.
type RenderNodeGlobFunc func() ([]string, error)

// DefaultRenderNodeGlob lists real Intel Arc render nodes. It accepts both
// device-node shapes seen in deployment: the conventional /dev/dri/renderD<N>
// name, and /dev/dri/by-path/pci-*-render, which is where a Proxmox `dev0:`
// LXC device passthrough mounts the node when given a by-path source (the
// compute runtime's device discovery scans by-path too, so the node is fully
// usable there).
func DefaultRenderNodeGlob() ([]string, error) {
	nodes, err := filepath.Glob("/dev/dri/renderD*")
	if err != nil {
		return nil, err
	}
	byPath, err := filepath.Glob("/dev/dri/by-path/*-render")
	if err != nil {
		return nil, err
	}
	return append(nodes, byPath...), nil
}

// XPUQueryFunc returns the raw output of an xpu-smi free-memory query. It
// mirrors GPUQueryFunc: a seam so tests inject fake output (or
// exec.ErrNotFound to simulate an absent xpu-smi binary) without touching a
// real device.
type XPUQueryFunc func() (string, error)

// xpuFreeMiBPattern extracts the first numeric token following a "free"
// marker (case-insensitive) from xpu-smi output, tolerating both a simple
// tabular/CSV shape and JSON key ordering (e.g. `"memory_free_mib": 8192` or
// `Memory Free (MiB): 8192`).
var xpuFreeMiBPattern = regexp.MustCompile(`(?i)free[^0-9]*([0-9]+(?:\.[0-9]+)?)`)

// DefaultXPUQuery runs xpu-smi's per-device memory query for device 0 and
// returns raw output for XPUChecks to parse against xpuFreeMiBPattern. NOTE:
// unverified against real Intel Arc hardware -- no XPU host is reachable as of
// this writing. Confirm the command and the parsing pattern together against a
// real xpu-smi install before trusting the VRAM number in production; the
// render-node check in XPUChecks is the load-bearing hard requirement and does
// not depend on this.
func DefaultXPUQuery() (string, error) {
	out, err := exec.Command("xpu-smi", "stats", "-d", "0", "-j").Output()
	return string(out), err
}

// XPUChecks builds the xpu-device preflight (design cambia-329, restructured
// per-device from GPUVRAMCheck). A real Intel GPU render node is hard-required
// and reported under its own name (xpu_render_node), distinct from gpu_vram,
// so it sits outside the runner force matrix (harness/preflight.go's
// runnerOverridable names gpu_vram only) and can never be bypassed with
// force=true. When a render node is present and xpu-smi's VRAM query
// succeeds, the result is folded into a gpu_vram check with the given minGB
// floor, forceable exactly like the cuda path. When xpu-smi is absent, only
// the render-node check is returned; submitPreflights fills the missing
// gpu_vram slot with a passing, "unverifiable" note.
func XPUChecks(minGB float64, renderNode RenderNodeGlobFunc, query XPUQueryFunc) []PreflightCheck {
	const renderName = "xpu_render_node"
	nodes, err := renderNode()
	if err != nil {
		return []PreflightCheck{{renderName, false, fmt.Sprintf("render node probe failed: %v", err)}}
	}
	if len(nodes) == 0 {
		return []PreflightCheck{{renderName, false, "no Intel GPU render node found under /dev/dri/renderD* or /dev/dri/by-path/*-render"}}
	}
	checks := []PreflightCheck{{renderName, true, fmt.Sprintf("render node present (%s)", nodes[0])}}

	const vramName = "gpu_vram"
	if minGB <= 0 {
		return append(checks, PreflightCheck{vramName, true, "no VRAM requirement"})
	}

	out, qerr := query()
	if qerr != nil {
		if errors.Is(qerr, exec.ErrNotFound) {
			return append(checks, PreflightCheck{vramName, true, "no xpu-smi; VRAM floor unverifiable"})
		}
		return append(checks, PreflightCheck{vramName, false, fmt.Sprintf("xpu-smi failed: %v", qerr)})
	}
	m := xpuFreeMiBPattern.FindStringSubmatch(out)
	if m == nil {
		return append(checks, PreflightCheck{vramName, false, "unparseable xpu-smi output (no free-memory field found)"})
	}
	freeMB, perr := strconv.ParseFloat(m[1], 64)
	if perr != nil {
		return append(checks, PreflightCheck{vramName, false, fmt.Sprintf("unparseable free-memory value %q", m[1])})
	}
	freeGB := freeMB / 1024.0
	if freeGB >= minGB {
		return append(checks, PreflightCheck{vramName, true, fmt.Sprintf("%.1f GiB free (need %.1f)", freeGB, minGB)})
	}
	return append(checks, PreflightCheck{vramName, false, fmt.Sprintf("only %.1f GiB free (need %.1f)", freeGB, minGB)})
}

// ResolveRunDevice returns the training device configured for run name's
// materialized runs/<name>/config.yaml, via a light line-based YAML scan (the
// full Config schema is Python-side; Go only needs this one scalar). It
// tolerates a root-level `device:` key and a `prt_cfr:` section's nested
// `device:` key, with the section value winning when both are present (the
// PRT-CFR trainer reads its own section). A missing file, missing key, or an
// explicit "auto" all resolve to "auto" so callers treat the run as
// GPU-relevant by default: only an explicit "cpu" is safe to skip the VRAM
// check for.
func ResolveRunDevice(runsDir, name string) string {
	data, err := os.ReadFile(filepath.Join(runDirOf(runsDir, name), "config.yaml"))
	if err != nil {
		return "auto"
	}
	var root, section string
	inPRTCFR := false
	for _, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimRight(raw, "\r")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " \t"))
		if indent == 0 {
			inPRTCFR = trimmed == "prt_cfr:" || strings.HasPrefix(trimmed, "prt_cfr:")
			if v, ok := yamlScalarField(trimmed, "device"); ok {
				root = v
			}
			continue
		}
		if inPRTCFR {
			if v, ok := yamlScalarField(trimmed, "device"); ok {
				section = v
			}
		}
	}
	if section != "" {
		return section
	}
	if root != "" {
		return root
	}
	return "auto"
}

// yamlScalarField reports whether trimmed is a "key: value" line for key,
// returning the trimmed, quote-stripped, comment-stripped value.
func yamlScalarField(trimmed, key string) (string, bool) {
	prefix := key + ":"
	if !strings.HasPrefix(trimmed, prefix) {
		return "", false
	}
	v := strings.TrimSpace(trimmed[len(prefix):])
	if i := strings.Index(v, "#"); i >= 0 {
		v = strings.TrimSpace(v[:i])
	}
	return strings.Trim(v, `"'`), true
}

// DiskSpaceCheck passes when the filesystem backing path has at least minGB of
// space available to an unprivileged writer, or when no disk is required
// (minGB <= 0).
func DiskSpaceCheck(path string, minGB float64) PreflightCheck {
	const name = "disk_space"
	if minGB <= 0 {
		return PreflightCheck{name, true, "no disk requirement"}
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return PreflightCheck{name, false, fmt.Sprintf("statfs %s failed: %v", path, err)}
	}
	freeBytes := st.Bavail * uint64(st.Bsize)
	freeGB := float64(freeBytes) / (1 << 30)
	if freeGB >= minGB {
		return PreflightCheck{name, true, fmt.Sprintf("%.1f GiB free (need %.1f)", freeGB, minGB)}
	}
	return PreflightCheck{name, false, fmt.Sprintf("only %.1f GiB free (need %.1f)", freeGB, minGB)}
}

// NameCollisionCheck fails when a run of that name already has a process.json.
// It is not force-overridable: reusing a name would clobber another run's
// current-state store. Bare name validation is enforced separately by
// ValidateName before any path is constructed.
func NameCollisionCheck(runsDir, name string) PreflightCheck {
	if _, err := ReadProcessState(runDirOf(runsDir, name)); err == nil {
		return PreflightCheck{"name_collision", false, fmt.Sprintf("a run named %q already exists", name)}
	}
	return PreflightCheck{"name_collision", true, "name available"}
}

// ConcurrencyCapCheck fails when the count of live runs (running, starting, or
// stopping with an alive pid) is already at or above max. max <= 0 disables the
// cap.
func ConcurrencyCapCheck(runsDir string, max int) PreflightCheck {
	const name = "concurrency_cap"
	if max <= 0 {
		return PreflightCheck{name, true, "no concurrency cap"}
	}
	states, _ := ScanProcessStates(runsDir)
	running := 0
	for _, st := range states {
		switch EffectiveStatus(st) {
		case StatusRunning, StatusStarting, StatusStopping:
			running++
		}
	}
	if running >= max {
		return PreflightCheck{name, false, fmt.Sprintf("%d live runs at cap %d", running, max)}
	}
	return PreflightCheck{name, true, fmt.Sprintf("%d live runs (cap %d)", running, max)}
}

// PreflightPasses reports whether the checks permit the action. It passes when
// every check is ok, or when force is set and every failing check is
// overridable. The second return value is the list of failing checks (nil when
// all pass), for the 409 preflight_failed body.
func PreflightPasses(checks []PreflightCheck, force bool) (bool, []PreflightCheck) {
	var failed []PreflightCheck
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
			if !overridableChecks[c.Name] {
				return false, failed
			}
		}
		return true, failed
	}
	return false, failed
}

// firstNonEmptyLine returns the first line of s with content, trimmed.
func firstNonEmptyLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			return t
		}
	}
	return ""
}

// runDirOf returns the run directory for name under runsDir. Callers must have
// validated name first.
func runDirOf(runsDir, name string) string {
	return filepath.Join(runsDir, name)
}

// EffectiveStatus returns st.Status with pid liveness applied: a run recorded
// as running/starting/stopping whose pid is no longer alive is reported as
// crashed. This is the read-time view; Reconcile persists the same repair at
// server start.
func EffectiveStatus(st *ProcessState) string {
	// A remote row (Host set) is a bounded-stale projection synced from another
	// host. Its recorded pid names a process in THAT host's pid space, so a local
	// pidAlive probe would test an unrelated pid here (the cross-host pid-reuse
	// bug). Short-circuit before any probe and return the synced status verbatim;
	// freshness is surfaced separately via last_sync_at, not local liveness.
	if st.Host != "" {
		return st.Status
	}
	switch st.Status {
	case StatusRunning, StatusStarting, StatusStopping:
		if !pidAlive(st) {
			return StatusCrashed
		}
	}
	return st.Status
}
