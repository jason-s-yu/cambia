package training

import (
	"errors"
	"net/http"
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// EvalHandlers serves the eval HTTP surface: POST trigger (with cuda preflight
// reuse and a no-checkpoint 404) and GET list (with a per-job log tail). It
// composes the EvalManager (job lifecycle) with the preflight.go GPU/disk checks
// reused verbatim from Phase 2.
type EvalHandlers struct {
	mgr     *EvalManager
	runsDir string

	minVRAMGB float64
	minDiskGB float64

	// gpuQuery is the nvidia-smi seam; tests replace it so preflight never
	// depends on live VRAM or touches the GPU. Mirrors ProcessHandlers.
	gpuQuery procmgr.GPUQueryFunc
}

// EvalHandlersConfig configures NewEvalHandlers. Zero-value MinVRAMGB/MinDiskGB
// fall back to procmgr.DefaultMinVRAMGB/procmgr.DefaultMinDiskGB, so the safety rails are on by
// default rather than silently disabled by an unset field.
type EvalHandlersConfig struct {
	Manager   *EvalManager
	RunsDir   string
	MinVRAMGB float64
	MinDiskGB float64
}

// NewEvalHandlers builds the eval handler set. gpuQuery defaults to nvidia-smi.
func NewEvalHandlers(cfg EvalHandlersConfig) *EvalHandlers {
	minVRAM := cfg.MinVRAMGB
	if minVRAM <= 0 {
		minVRAM = procmgr.DefaultMinVRAMGB
	}
	minDisk := cfg.MinDiskGB
	if minDisk <= 0 {
		minDisk = procmgr.DefaultMinDiskGB
	}
	return &EvalHandlers{
		mgr:       cfg.Manager,
		runsDir:   cfg.RunsDir,
		minVRAMGB: minVRAM,
		minDiskGB: minDisk,
		gpuQuery:  procmgr.DefaultGPUQuery,
	}
}

// triggerEvalRequest is the POST /training/runs/{name}/eval body.
type triggerEvalRequest struct {
	Epoch         *int     `json:"epoch"`
	Device        string   `json:"device"`
	Games         int      `json:"games"`
	Argmax        bool     `json:"argmax"`
	Force         bool     `json:"force"`
	MinFreeVRAMGB *float64 `json:"min_free_vram_gb"`
	MinFreeDiskGB *float64 `json:"min_free_disk_gb"`
}

// HandleTrigger validates the run name, 404s when nothing is evaluable, runs the
// disk preflight (always) plus the GPU preflight (device=="cuda"), and spawns
// the eval. A cuda block returns 409 preflight_failed unless force overrides; a
// cap hit returns 409 eval_cap_reached; success returns 202 {job}.
func (h *EvalHandlers) HandleTrigger(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", "missing run name")
		return
	}
	if err := procmgr.ValidateName(name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	var req triggerEvalRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}

	device := req.Device
	if device == "" {
		device = "cpu"
	}

	// No-checkpoint 404 before preflight: an unevaluable run is a not-found, not
	// a resource-contention 409, regardless of device.
	if !hasEvaluableCheckpoint(filepath.Join(h.runsDir, name)) {
		writeJSONError(w, http.StatusNotFound, "no_checkpoint", "no evaluable checkpoint under checkpoints/ or snapshots/")
		return
	}

	minVRAM := h.minVRAMGB
	minDisk := h.minDiskGB
	// A non-positive request override falls back to the configured default
	// rather than disabling the check; disabling it is what force=true is for.
	if req.MinFreeVRAMGB != nil && *req.MinFreeVRAMGB > 0 {
		minVRAM = *req.MinFreeVRAMGB
	}
	if req.MinFreeDiskGB != nil && *req.MinFreeDiskGB > 0 {
		minDisk = *req.MinFreeDiskGB
	}

	checks := []procmgr.PreflightCheck{procmgr.DiskSpaceCheck(h.runsDir, minDisk)}
	if device == "cuda" {
		checks = append(checks, procmgr.GPUVRAMCheck(minVRAM, h.gpuQuery))
	}
	if ok, failed := procmgr.PreflightPasses(checks, req.Force); !ok {
		writePreflightFailed(w, failed)
		return
	}

	job, err := h.mgr.Trigger(name, EvalOpts{
		Epoch:  req.Epoch,
		Device: device,
		Games:  req.Games,
		Argmax: req.Argmax,
	})
	if err != nil {
		switch {
		case errors.Is(err, ErrNoCheckpoint):
			writeJSONError(w, http.StatusNotFound, "no_checkpoint", err.Error())
		case errors.Is(err, ErrEvalCapReached):
			writeJSONError(w, http.StatusConflict, "eval_cap_reached", err.Error())
		case errors.Is(err, procmgr.ErrInvalidName):
			writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		default:
			writeJSONError(w, http.StatusInternalServerError, "eval_failed", err.Error())
		}
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"job": job})
}

// HandleList returns the run's eval jobs newest-first, each with a log tail.
func (h *EvalHandlers) HandleList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", "missing run name")
		return
	}
	if err := procmgr.ValidateName(name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	jobs := h.mgr.Jobs(name)
	if jobs == nil {
		jobs = []*EvalJob{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs})
}
