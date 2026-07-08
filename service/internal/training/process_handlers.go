package training

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// defaultAlgorithm is used when a create request omits the algorithm.
const defaultAlgorithm = "prt-cfr"

// templatePrefix filters the config template directory to the PRT-CFR family
// (the only algorithm this sprint launches). The template directory itself is
// the cfr config root, set by the caller (see main.go: cfr/config).
const templatePrefix = "prtcfr"

// ProcessHandlers serves the process-management HTTP surface: create, start,
// stop, resume, and the config template listing. It composes the ProcessManager
// (lifecycle) with the TrainingStore (run detail) and shells to the cambia CLI
// for config composition (Go never merges YAML).
type ProcessHandlers struct {
	mgr   *ProcessManager
	store *TrainingStore

	cambiaBin   string
	cfrDir      string
	runsDir     string
	templateDir string

	maxConcurrent int
	minVRAMGB     float64
	minDiskGB     float64

	// gpuQuery is the nvidia-smi seam; tests replace it so preflight never
	// depends on live VRAM or touches the GPU.
	gpuQuery gpuQueryFunc
}

// ProcessHandlersConfig configures NewProcessHandlers.
type ProcessHandlersConfig struct {
	Manager       *ProcessManager
	Store         *TrainingStore
	CambiaBin     string
	CFRDir        string
	RunsDir       string
	TemplateDir   string
	MaxConcurrent int
	MinVRAMGB     float64
	MinDiskGB     float64
}

// NewProcessHandlers builds the handler set. gpuQuery defaults to nvidia-smi.
// A non-positive MinVRAMGB/MinDiskGB in cfg (including the zero value of an
// unset field) falls back to DefaultMinVRAMGB/DefaultMinDiskGB: the safety
// rails are on by default, not silently disabled by an unconfigured caller.
func NewProcessHandlers(cfg ProcessHandlersConfig) *ProcessHandlers {
	minVRAM := cfg.MinVRAMGB
	if minVRAM <= 0 {
		minVRAM = DefaultMinVRAMGB
	}
	minDisk := cfg.MinDiskGB
	if minDisk <= 0 {
		minDisk = DefaultMinDiskGB
	}
	return &ProcessHandlers{
		mgr:           cfg.Manager,
		store:         cfg.Store,
		cambiaBin:     cfg.CambiaBin,
		cfrDir:        cfg.CFRDir,
		runsDir:       cfg.RunsDir,
		templateDir:   cfg.TemplateDir,
		maxConcurrent: cfg.MaxConcurrent,
		minVRAMGB:     minVRAM,
		minDiskGB:     minDisk,
		gpuQuery:      defaultGPUQuery,
	}
}

// createRunRequest is the POST /training/runs body.
type createRunRequest struct {
	Name      string         `json:"name"`
	Template  string         `json:"template"`
	Algorithm string         `json:"algorithm"`
	Overrides map[string]any `json:"overrides"`
	YAML      string         `json:"yaml"`
}

// startRunRequest is the POST /training/runs/{name}/start body.
type startRunRequest struct {
	Force         bool     `json:"force"`
	MinFreeVRAMGB *float64 `json:"min_free_vram_gb"`
	MinFreeDiskGB *float64 `json:"min_free_disk_gb"`
}

// stopRunRequest is the POST /training/runs/{name}/stop body.
type stopRunRequest struct {
	Force bool `json:"force"`
}

// resumeRunRequest is the POST /training/runs/{name}/resume body.
type resumeRunRequest struct {
	Force         bool     `json:"force"`
	MinFreeVRAMGB *float64 `json:"min_free_vram_gb"`
}

// HandleCreate materializes the config (via `cambia config render`/`validate`)
// and creates the run. The config is rendered to a temp file first, then Create
// copies it into runs/<name>/config.yaml only after the collision check passes
// under lock, so a name collision never clobbers an existing run's config.
func (h *ProcessHandlers) HandleCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req createRunRequest
	if err := decodeCreateBody(r, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}

	if err := validateName(req.Name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	algo := req.Algorithm
	if algo == "" {
		algo = defaultAlgorithm
	}
	if _, err := algoSubcommand(algo); err != nil {
		writeJSONError(w, http.StatusBadRequest, "unsupported_algorithm", err.Error())
		return
	}

	// Early, race-safe-backed collision check for a clean 409 before doing any
	// render work. Create re-checks under its lock as the authority.
	if c := nameCollisionCheck(h.runsDir, req.Name); !c.OK {
		writeCollision(w, c)
		return
	}

	tmpPath, cleanup, err := h.materializeConfig(req)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_config", err.Error())
		return
	}
	defer cleanup()

	st, err := h.mgr.Create(CreateRequest{Name: req.Name, Algorithm: algo, ConfigPath: tmpPath})
	if err != nil {
		switch {
		case errors.Is(err, ErrNameCollision):
			writeCollision(w, PreflightCheck{"name_collision", false, err.Error()})
		case errors.Is(err, ErrInvalidName):
			writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		default:
			writeJSONError(w, http.StatusInternalServerError, "create_failed", err.Error())
		}
		return
	}

	var rd *RunDetail
	if h.store != nil {
		rd, _ = h.store.GetRun(r.Context(), req.Name)
	}
	writeJSON(w, http.StatusCreated, map[string]any{"run": rd, "process": st})
}

// materializeConfig renders the run config into a temp file and returns its
// path plus a cleanup func. A verbatim yaml body is written and validated; a
// template body is composed with `cambia config render` and its --set
// overrides. Either way the CLI validates against the Config schema.
func (h *ProcessHandlers) materializeConfig(req createRunRequest) (string, func(), error) {
	f, err := os.CreateTemp("", "cambia-config-*.yaml")
	if err != nil {
		return "", func() {}, fmt.Errorf("create temp config: %w", err)
	}
	tmpPath := f.Name()
	f.Close()
	cleanup := func() { os.Remove(tmpPath) }

	if req.YAML != "" {
		if err := os.WriteFile(tmpPath, []byte(req.YAML), 0o644); err != nil {
			cleanup()
			return "", func() {}, fmt.Errorf("write yaml: %w", err)
		}
		if err := h.runCambia("config", "validate", tmpPath); err != nil {
			cleanup()
			return "", func() {}, err
		}
		return tmpPath, cleanup, nil
	}

	if req.Template == "" {
		cleanup()
		return "", func() {}, errors.New("template or yaml required")
	}
	if strings.Contains(req.Template, "/") || strings.Contains(req.Template, "..") {
		cleanup()
		return "", func() {}, fmt.Errorf("invalid template name %q", req.Template)
	}
	tmplPath := filepath.Join(h.templateDir, req.Template)

	args := []string{"config", "render", tmplPath}
	for _, k := range sortedKeys(req.Overrides) {
		args = append(args, "--set", k+"="+stringifyOverride(req.Overrides[k]))
	}
	args = append(args, "-o", tmpPath)
	if err := h.runCambia(args...); err != nil {
		cleanup()
		return "", func() {}, err
	}
	return tmpPath, cleanup, nil
}

// HandleStart runs the start preflight then launches the run.
func (h *ProcessHandlers) HandleStart(w http.ResponseWriter, r *http.Request) {
	h.launchHandler(w, r, false)
}

// HandleResume gates on a resumable checkpoint, runs the preflight, then
// launches with --resume.
func (h *ProcessHandlers) HandleResume(w http.ResponseWriter, r *http.Request) {
	h.launchHandler(w, r, true)
}

// launchHandler is the shared start/resume path.
func (h *ProcessHandlers) launchHandler(w http.ResponseWriter, r *http.Request, resume bool) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", "missing run name")
		return
	}
	if err := validateName(name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	if _, ok := h.mgr.GetState(name); !ok {
		writeJSONError(w, http.StatusNotFound, "not_found", "run not found")
		return
	}

	var force bool
	minVRAM := h.minVRAMGB
	minDisk := h.minDiskGB
	if resume {
		var req resumeRunRequest
		if err := decodeJSONBody(r, &req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
			return
		}
		force = req.Force
		// A non-positive request override falls back to the configured
		// default rather than disabling the check; disabling it is what
		// force=true is for.
		if req.MinFreeVRAMGB != nil && *req.MinFreeVRAMGB > 0 {
			minVRAM = *req.MinFreeVRAMGB
		}
		if !hasCheckpoint(h.runsDir, name) {
			writeJSONError(w, http.StatusConflict, "no_resumable_checkpoint",
				"no checkpoint found under runs/"+name+"/snapshots/prtcfr_checkpoint.pt")
			return
		}
	} else {
		var req startRunRequest
		if err := decodeJSONBody(r, &req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
			return
		}
		force = req.Force
		if req.MinFreeVRAMGB != nil && *req.MinFreeVRAMGB > 0 {
			minVRAM = *req.MinFreeVRAMGB
		}
		if req.MinFreeDiskGB != nil && *req.MinFreeDiskGB > 0 {
			minDisk = *req.MinFreeDiskGB
		}
	}

	checks := []PreflightCheck{
		gpuVRAMCheck(minVRAM, h.gpuQuery),
		diskSpaceCheck(h.runsDir, minDisk),
		concurrencyCapCheck(h.runsDir, h.maxConcurrent),
	}
	if ok, failed := preflightPasses(checks, force); !ok {
		writePreflightFailed(w, failed)
		return
	}

	var st *ProcessState
	var err error
	if resume {
		st, err = h.mgr.Resume(name, StartOpts{})
	} else {
		st, err = h.mgr.Start(name, StartOpts{})
	}
	if err != nil {
		h.writeLaunchError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"process": st})
}

// HandleStop stops (or force-kills) a run.
func (h *ProcessHandlers) HandleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := extractRunName(r)
	if name == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", "missing run name")
		return
	}
	if err := validateName(name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	if _, ok := h.mgr.GetState(name); !ok {
		writeJSONError(w, http.StatusNotFound, "not_found", "run not found")
		return
	}
	var req stopRunRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	st, err := h.mgr.Stop(name, req.Force)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "stop_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"process": st})
}

// HandleTemplates lists the PRT-CFR config templates (basenames) under the
// configured template directory (cfr/config).
func (h *ProcessHandlers) HandleTemplates(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	names := []string{}
	entries, err := os.ReadDir(h.templateDir)
	if err == nil {
		for _, e := range entries {
			n := e.Name()
			if !e.IsDir() && strings.HasSuffix(n, ".yaml") && strings.HasPrefix(n, templatePrefix) {
				names = append(names, n)
			}
		}
	}
	sort.Strings(names)
	writeJSON(w, http.StatusOK, names)
}

// writeLaunchError maps ProcessManager start/resume errors to status codes.
func (h *ProcessHandlers) writeLaunchError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrAlreadyRunning):
		writeJSONError(w, http.StatusConflict, "already_running", err.Error())
	case errors.Is(err, ErrConcurrencyCapReached):
		writeJSONError(w, http.StatusConflict, "concurrency_cap_reached", err.Error())
	case errors.Is(err, ErrConfigMissing):
		writeJSONError(w, http.StatusBadRequest, "config_missing", err.Error())
	case errors.Is(err, ErrUnsupportedAlgorithm):
		writeJSONError(w, http.StatusBadRequest, "unsupported_algorithm", err.Error())
	case errors.Is(err, ErrInvalidName):
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
	default:
		writeJSONError(w, http.StatusInternalServerError, "launch_failed", err.Error())
	}
}

// runCambia invokes the cambia CLI with cwd = cfrDir. It returns a combined
// stdout+stderr error on nonzero exit.
func (h *ProcessHandlers) runCambia(args ...string) error {
	cmd := exec.Command(h.cambiaBin, args...)
	cmd.Dir = h.cfrDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cambia %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// hasCheckpoint reports whether runs/<name>/ holds a resumable PRT-CFR
// checkpoint. The PRT-CFR trainer writes its rolling checkpoint to
// snapshots/prtcfr_checkpoint.pt and commits resume_state.json at the run-dir
// root last, once the checkpoint and reservoir state are durable (see
// prtcfr_trainer.py _save_resume_state). Resume gates on both files existing:
// the checkpoint alone can be mid-write, and resume_state.json alone (without
// its checkpoint) cannot be resumed from either.
func hasCheckpoint(runsDir, name string) bool {
	runDir := filepath.Join(runsDir, name)
	if _, err := os.Stat(filepath.Join(runDir, "snapshots", "prtcfr_checkpoint.pt")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(runDir, "resume_state.json")); err != nil {
		return false
	}
	return true
}

// decodeCreateBody decodes a create request, preserving numeric override values
// as json.Number so integer overrides stay integral (e.g. iterations=5, not
// 5.0). An empty body is tolerated (name validation catches the missing name).
func decodeCreateBody(r *http.Request, v *createRunRequest) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	return nil
}

// decodeJSONBody decodes a typed request body, tolerating an empty body.
func decodeJSONBody(r *http.Request, v any) error {
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	return nil
}

// stringifyOverride renders an override value for a `--set key=value` argument.
func stringifyOverride(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		if x {
			return "true"
		}
		return "false"
	case json.Number:
		return x.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

// sortedKeys returns m's keys sorted, for deterministic --set ordering.
func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// writeJSONError writes {error, detail} with the given status.
func writeJSONError(w http.ResponseWriter, status int, code, detail string) {
	writeJSON(w, status, map[string]string{"error": code, "detail": detail})
}

// writePreflightFailed writes the 409 preflight_failed body per the pinned HTTP
// surface: the failing checks plus the override keyword.
func writePreflightFailed(w http.ResponseWriter, failed []PreflightCheck) {
	writeJSON(w, http.StatusConflict, map[string]any{
		"error":    "preflight_failed",
		"checks":   failed,
		"override": "force",
	})
}

// writeCollision writes the 409 name-collision body.
func writeCollision(w http.ResponseWriter, c PreflightCheck) {
	writeJSON(w, http.StatusConflict, map[string]any{
		"error":  "collision",
		"checks": []PreflightCheck{c},
	})
}
