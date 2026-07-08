package training

import (
	"errors"
	"fmt"
	"os/exec"
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

// gpuQueryFunc returns the raw CSV output of an nvidia-smi VRAM query. It is a
// seam so tests can inject fake output (or an exec.ErrNotFound to simulate a
// CPU host) without touching a real GPU.
type gpuQueryFunc func() (string, error)

// defaultGPUQuery runs nvidia-smi for per-GPU free memory, utilization, and
// name. A missing binary yields an *exec.Error wrapping exec.ErrNotFound, which
// gpuVRAMCheck treats as a CPU host rather than a failure.
func defaultGPUQuery() (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.free,utilization.gpu,name",
		"--format=csv,noheader,nounits").Output()
	return string(out), err
}

// gpuVRAMCheck passes when the first GPU reports at least minGB free, when no
// VRAM is required (minGB <= 0), or when nvidia-smi is absent (CPU host). A
// present-but-failing nvidia-smi is a hard block so a misconfigured GPU host
// does not silently launch under contention.
func gpuVRAMCheck(minGB float64, query gpuQueryFunc) PreflightCheck {
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

// diskSpaceCheck passes when the filesystem backing path has at least minGB of
// space available to an unprivileged writer, or when no disk is required
// (minGB <= 0).
func diskSpaceCheck(path string, minGB float64) PreflightCheck {
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

// nameCollisionCheck fails when a run of that name already has a process.json.
// It is not force-overridable: reusing a name would clobber another run's
// current-state store. Bare name validation is enforced separately by
// validateName before any path is constructed.
func nameCollisionCheck(runsDir, name string) PreflightCheck {
	if _, err := readProcessState(runDirOf(runsDir, name)); err == nil {
		return PreflightCheck{"name_collision", false, fmt.Sprintf("a run named %q already exists", name)}
	}
	return PreflightCheck{"name_collision", true, "name available"}
}

// concurrencyCapCheck fails when the count of live runs (running, starting, or
// stopping with an alive pid) is already at or above max. max <= 0 disables the
// cap.
func concurrencyCapCheck(runsDir string, max int) PreflightCheck {
	const name = "concurrency_cap"
	if max <= 0 {
		return PreflightCheck{name, true, "no concurrency cap"}
	}
	states, _ := scanProcessStates(runsDir)
	running := 0
	for _, st := range states {
		switch effectiveStatus(st) {
		case StatusRunning, StatusStarting, StatusStopping:
			running++
		}
	}
	if running >= max {
		return PreflightCheck{name, false, fmt.Sprintf("%d live runs at cap %d", running, max)}
	}
	return PreflightCheck{name, true, fmt.Sprintf("%d live runs (cap %d)", running, max)}
}

// preflightPasses reports whether the checks permit the action. It passes when
// every check is ok, or when force is set and every failing check is
// overridable. The second return value is the list of failing checks (nil when
// all pass), for the 409 preflight_failed body.
func preflightPasses(checks []PreflightCheck, force bool) (bool, []PreflightCheck) {
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
