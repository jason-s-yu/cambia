package harness

import (
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// handleCreateJob is POST /harness/jobs. It runs the design 2.6 validation
// order (validateName -> collision -> kind allowlist -> device shape ->
// device capability gate -> path guards -> preflights -> render gate), then
// admits and enqueues the job. Status mapping: 201 accepted, 409 name
// collision, 400 invalid name/kind/device/path, 412 preflight failure (per
// check), 429 queue full.
func (s *Server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	var spec JobSpec
	if err := decodeJSON(r, &spec); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}

	// 1. validateName
	if err := procmgr.ValidateName(spec.Name); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	// 2. name collision (procmgr.Create re-checks atomically under lock)
	if c := procmgr.NameCollisionCheck(s.runsDir, spec.Name); !c.OK {
		writeJSON(w, http.StatusConflict, map[string]any{
			"error":  "name_collision",
			"checks": []procmgr.PreflightCheck{c},
		})
		return
	}
	// 3. kind allowlist
	if _, ok := s.algos[spec.Kind]; !ok {
		writeJSONError(w, http.StatusBadRequest, "invalid_kind", "kind not in allowlist: "+spec.Kind)
		return
	}
	// 3b. target kind-scoping (design 2.6): evaluate-only field, forbidden for
	// train. Evaluate's requiredness is enforced below, in the containment loop.
	if spec.targetForbidden() {
		writeJSONError(w, http.StatusBadRequest, "invalid_target", "target is not valid for kind=train")
		return
	}
	// 3b'. warm_start kind-scoping (design cambia-334): train-only field, like
	// target is evaluate-only. warm_start is optional even for train, so there
	// is no required-ness check here (unlike target for evaluate below).
	if spec.warmStartForbidden() {
		writeJSONError(w, http.StatusBadRequest, "invalid_warm_start", "warm_start is only valid for kind=train")
		return
	}
	// 3c. device shape validation: device must be one of cpu/cuda/xpu.
	if !spec.deviceValid() {
		writeJSONError(w, http.StatusBadRequest, "invalid_device", "device not supported: "+spec.device())
		return
	}
	// 3d. device capability gate (design cambia-329): device must be enabled on
	// this runner via RUNNERD_ALLOWED_DEVICES. Not forceable -- a captured
	// token cannot ask a cpu-only runner to admit a GPU job.
	if !s.allowedDevices[spec.device()] {
		writeJSONError(w, http.StatusBadRequest, "device_unsupported", "device not enabled on this runner: "+spec.device())
		return
	}
	// 3e. cross-job dependency (cambia-352): after names a single parent job that
	// must already exist; a terminal parent is allowed (the gate resolves its
	// outcome at dispatch). Self-reference is rejected; cycles are structurally
	// impossible since a parent must exist strictly before its child. on_failure
	// governs only the parent-failure branch.
	if spec.After != "" {
		if err := procmgr.ValidateName(spec.After); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_after", err.Error())
			return
		}
		if spec.After == spec.Name {
			writeJSONError(w, http.StatusBadRequest, "invalid_after", "after must not reference the job itself")
			return
		}
		if _, err := procmgr.ReadProcessState(filepath.Join(s.runsDir, spec.After)); err != nil {
			writeJSONError(w, http.StatusBadRequest, "after_not_found", "parent job not found: "+spec.After)
			return
		}
	}
	if !spec.onFailureValid() {
		writeJSONError(w, http.StatusBadRequest, "invalid_on_failure", "on_failure must be one of skip|run|fail")
		return
	}
	// 4. path guards (config, checkpoints): lexical shape (reject absolute + ..).
	for _, p := range spec.guardedPaths() {
		if err := pathguard.CheckRel(p.value); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_path", p.label+": "+err.Error())
			return
		}
	}
	// 4b. checkpoint + target containment (design 5.4): head-to-head
	// checkpoints and an evaluate target must resolve inside the runner runs
	// dir. config containment is deferred to ingest render, where the worktree
	// base is staged.
	for _, p := range append(spec.containedCheckpoints(), spec.containedTarget()...) {
		if _, err := pathguard.Resolve(s.runsDir, p.value); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_path", p.label+": "+err.Error())
			return
		}
	}
	// 4c. warm_start containment + existence (design cambia-334): a train job's
	// warm_start must resolve inside the runs directory (same containment path
	// as checkpoints/target above), and the referenced snapshot must actually
	// exist -- a submit-time failure beats a job that launches and only then
	// dies in prepare.
	for _, p := range spec.containedWarmStart() {
		resolved, err := pathguard.Resolve(s.runsDir, p.value)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid_path", p.label+": "+err.Error())
			return
		}
		if _, err := os.Stat(resolved); err != nil {
			writeJSONError(w, http.StatusBadRequest, "warm_start_not_found", p.label+": "+resolved)
			return
		}
	}
	// 5. preflights (disk, RAM; gpu skipped for cpu) under the runner force matrix
	checks := s.submitPreflights(&spec)
	if ok, failed := preflightPasses(checks, spec.Force); !ok {
		writeJSON(w, http.StatusPreconditionFailed, map[string]any{
			"error":    "preflight_failed",
			"checks":   failed,
			"override": "force (gpu_vram only; disk/ram floors are operator-set)",
		})
		return
	}
	// 6. render gate (M2 stub)
	if err := s.renderGate(&spec); err != nil {
		writeJSONError(w, http.StatusUnprocessableEntity, "render_failed", err.Error())
		return
	}

	view, err := s.disp.Submit(spec)
	if err != nil {
		switch {
		case errors.Is(err, ErrQueueFull):
			writeJSONError(w, http.StatusTooManyRequests, "queue_full", err.Error())
		case errors.Is(err, procmgr.ErrNameCollision):
			writeJSONError(w, http.StatusConflict, "name_collision", err.Error())
		case errors.Is(err, procmgr.ErrInvalidName):
			writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		default:
			writeJSONError(w, http.StatusInternalServerError, "submit_failed", err.Error())
		}
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{
		"job_id":    view.JobID,
		"state":     view.State,
		"queue_pos": view.QueuePos,
	})
}

// submitPreflights builds the admission preflight checks, routed by the job's
// device (design cambia-329): cpu skips the GPU check entirely; cuda runs the
// existing nvidia-smi VRAM check; xpu runs the render-node + xpu-smi VRAM
// check (procmgr.XPUChecks). Disk and RAM floors always apply.
func (s *Server) submitPreflights(spec *JobSpec) []procmgr.PreflightCheck {
	var checks []procmgr.PreflightCheck
	switch spec.device() {
	case "cuda":
		checks = append(checks, procmgr.GPUVRAMCheck(s.minVRAMGB, s.gpuQuery))
	case "xpu":
		checks = append(checks, procmgr.XPUChecks(s.minVRAMGB, s.renderNodeGlob, s.xpuQuery)...)
	default: // cpu
		checks = append(checks, procmgr.PreflightCheck{Name: "gpu_vram", OK: true, Detail: "device=cpu, GPU check skipped"})
	}
	checks = append(checks, procmgr.DiskSpaceCheck(s.runsDir, s.minDiskGB))
	checks = append(checks, MinFreeRAMCheck(s.minRAMGB, s.ramQuery))
	return checks
}

// renderGate is the M2 stub of the config render+validate gate (design 3.4). In
// M3 it runs `cambia config render ... -o runDir/config.yaml` then `cambia
// config validate` inside Prepare. In M2 the config is not materialized until
// ingest, so there is nothing to render yet.
func (s *Server) renderGate(spec *JobSpec) error { return nil }

// handleListJobs is GET /harness/jobs.
func (s *Server) handleListJobs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"jobs": s.disp.List()})
}

// handleGetJob is GET /harness/jobs/{id}: full view + resolved sha + env.json
// summary (env.json is written by M3 ingest; absent in M2).
func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	view, ok := s.disp.resolveView(id)
	if !ok {
		writeJSONError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}
	resp := map[string]any{"job": view, "resolved_sha": view.Commit}
	if env := readEnvJSON(s.runsDir, id); env != nil {
		resp["env"] = env
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleDeleteJob is DELETE /harness/jobs/{id}. ?purge=true removes a terminal
// job's run dir (refused for a non-terminal job); otherwise it cancels: a queued
// job is dropped, a running job is stopped (?force = SIGKILL, else SIGINT + 30s
// grace).
func (s *Server) handleDeleteJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	q := r.URL.Query()
	if q.Get("purge") == "true" {
		cascade := q.Get("cascade") == "true"
		switch err := s.disp.Purge(id, cascade); {
		case err == nil:
			writeJSON(w, http.StatusOK, map[string]any{"job_id": id, "purged": true})
		case errors.Is(err, ErrNotTerminal):
			writeJSONError(w, http.StatusConflict, "not_terminal", "purge refuses a non-terminal job")
		case errors.Is(err, ErrHasDependents):
			writeJSONError(w, http.StatusConflict, "has_dependents",
				"purge refused: job has queued dependents (retry with cascade=true to skip them)")
		case errors.Is(err, ErrNotFound):
			writeJSONError(w, http.StatusNotFound, "not_found", "job not found")
		default:
			writeJSONError(w, http.StatusInternalServerError, "purge_failed", err.Error())
		}
		return
	}

	force := q.Get("force") == "true"
	if _, err := s.disp.Cancel(id, force); err != nil {
		if errors.Is(err, ErrNotFound) {
			writeJSONError(w, http.StatusNotFound, "not_found", "job not found")
			return
		}
		writeJSONError(w, http.StatusInternalServerError, "cancel_failed", err.Error())
		return
	}
	view, _ := s.disp.resolveView(id)
	writeJSON(w, http.StatusOK, map[string]any{"job": view})
}

// handleResumeJob is POST /harness/jobs/{id}/resume: gate on the PRT-CFR resume
// contract, then re-enqueue as a resume launch.
func (s *Server) handleResumeJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	view, err := s.disp.Resume(id)
	if err != nil {
		switch {
		case errors.Is(err, ErrNotFound):
			writeJSONError(w, http.StatusNotFound, "not_found", "job not found")
		case errors.Is(err, ErrNoResumableState):
			writeJSONError(w, http.StatusConflict, "no_resumable_state",
				"resume requires runs/"+id+"/snapshots/prtcfr_checkpoint.pt and resume_state.json")
		case errors.Is(err, ErrAlreadyQueued):
			writeJSONError(w, http.StatusConflict, "already_queued", err.Error())
		case errors.Is(err, ErrQueueFull):
			writeJSONError(w, http.StatusTooManyRequests, "queue_full", err.Error())
		default:
			writeJSONError(w, http.StatusInternalServerError, "resume_failed", err.Error())
		}
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"job_id":    view.JobID,
		"state":     view.State,
		"queue_pos": view.QueuePos,
	})
}

// handleHealth is GET /harness/health.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	snap := s.disp.Snapshot()
	freeRAM, _ := s.ramQuery()
	writeJSON(w, http.StatusOK, map[string]any{
		"reconciled_at": snap.ReconciledAt,
		"jobs_running":  snap.JobsRunning,
		"queue_depth":   snap.QueueDepth,
		"free_ram_gb":   round1(freeRAM),
		"free_disk_gb":  round1(diskFreeGB(s.runsDir)),
	})
}

// readEnvJSON reads runs/<name>/env.json as a raw map (M3 provenance record),
// returning nil when absent or unparseable.
func readEnvJSON(runsDir, name string) map[string]any {
	data, err := os.ReadFile(filepath.Join(runsDir, name, "env.json"))
	if err != nil {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil
	}
	return m
}

// round1 rounds to one decimal place for the health payload.
func round1(v float64) float64 { return math.Round(v*10) / 10 }
