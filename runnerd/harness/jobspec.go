package harness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Runnerd-level job states layered over the procmgr ProcessState enum
// (design 2.3). queued/preparing/canceled/failed are owned by runnerd;
// created/running/stopping/stopped/crashed live in process.json. queued and
// preparing are memory-only (lost on restart, then swept); canceled and failed
// are persisted into process.json Status so they survive a restart and are not
// re-swept as orphans.
const (
	StateQueued    = "queued"
	StatePreparing = "preparing"
	StateCanceled  = "canceled"
	StateFailed    = "failed"
	// StateSkipped is the terminal state of a dependent whose parent reached a
	// non-success terminal and whose on_failure policy is skip (cambia-352). A
	// sibling ticket's state enum carries the same name; keep it exact.
	StateSkipped = "skipped"
)

// on_failure policies (cambia-352). They govern only the parent-failure branch
// of an `after` dependency: a parent success always runs the dependent. skip is
// the default (an empty on_failure).
const (
	OnFailureSkip = "skip"
	OnFailureRun  = "run"
	OnFailureFail = "fail"
)

// terminalStates is the set of states from which a job does not transition
// further (resume creates a new launch, not a transition). It spans both the
// runnerd-level terminals and the procmgr terminals.
var terminalStates = map[string]bool{
	StateCanceled:         true,
	StateFailed:           true,
	StateSkipped:          true,
	procmgr.StatusStopped: true,
	procmgr.StatusCrashed: true,
}

// isTerminal reports whether state is terminal.
func isTerminal(state string) bool { return terminalStates[state] }

// Job kinds (design 1, 2.6). v1 trains allowlist prtcfr only, expressed by the
// algos table: kind=train maps to the "train prtcfr" subcommand and no other
// train algorithm is registered.
const (
	KindTrain      = "train"
	KindEvaluate   = "evaluate"
	KindHeadToHead = "head-to-head"
	KindBench      = "bench"
	// KindMeasure runs a pinned script under an allowlisted root instead of a
	// cambia subcommand (design D38, cambia-1072): no cambia subcommand exists
	// for it, so it is registered in HarnessAlgorithms only to pass the submit
	// allowlist below; the launch template never consults that entry (see
	// Dispatcher.launchOpts).
	KindMeasure = "measure"
)

// measureScriptRoot is the allowlisted worktree-relative root a measure job's
// script must resolve under (design D38). script is validated against it
// lexically at submit (handlers.go, before a worktree exists) and the staged
// file is required to exist there at launch (Dispatcher.launchOpts).
const measureScriptRoot = "cfr/scripts/"

// maxMeasureArgLen caps a single measure args entry in bytes (design D38:
// "each entry rejected on a NUL or over a length cap"). 4096 comfortably
// covers a path or flag value while bounding a submitter's ability to inflate
// argv.
const maxMeasureArgLen = 4096

// HarnessAlgorithms returns the runnerd job-kind -> cambia-subcommand allowlist
// injected into the ProcessManager. It is the superset the daemon supervises
// (the dashboard registers only TrainAlgorithms). A fresh map is returned per
// call so callers cannot mutate a shared instance. kind=train resolves to
// `cambia train prtcfr`, encoding the v1 prtcfr-only train allowlist.
// kind=bench resolves to `cambia benchmark all`: cli.py registers the perf
// benchmarks under a `benchmark` typer group (`all`/`network`/`traversal`/...),
// not a top-level `bench` command, and the job spec carries no field to select
// a specific sub-benchmark, so "all" is the only well-defined target.
func HarnessAlgorithms() map[string][]string {
	return map[string][]string{
		KindTrain:      {"train", "prtcfr"},
		KindEvaluate:   {"evaluate"},
		KindHeadToHead: {"head-to-head"},
		KindBench:      {"benchmark", "all"},
		// KindMeasure carries no cambia subcommand (design D38): its launch
		// template builds argv from the staged script + args directly and never
		// calls AlgoSubcommand, so this value is unused. The entry exists only
		// so the kind allowlist check at handlers.go admits it.
		KindMeasure: {},
	}
}

// JobSpec is the submitted job description (design 2.6). It is decoded from the
// POST /harness/jobs body. Numeric override values are preserved via
// json.Number so integer overrides stay integral.
type JobSpec struct {
	Kind        string         `json:"kind"`
	Commit      string         `json:"commit"`
	Name        string         `json:"name"`
	Config      string         `json:"config"`
	Overrides   map[string]any `json:"overrides"`
	Resume      bool           `json:"resume"`
	Device      string         `json:"device"`
	CheckpointA string         `json:"checkpoint_a"`
	CheckpointB string         `json:"checkpoint_b"`
	Target      string         `json:"target"`
	Games       int            `json:"games"`
	Priority    string         `json:"priority"`
	Force       bool           `json:"force"`
	// WarmStart optionally names another run's staged snapshot (relative to the
	// runs dir, e.g. "prior-run/snapshots/prtcfr_snapshot_iter_530.pt") a train
	// job initializes from (design cambia-334). Empty means no warm start. Valid
	// only for kind=train; see warmStartForbidden.
	WarmStart string `json:"warm_start"`
	// After optionally gates this job on one or more parent jobs finishing first
	// (cambia-352, widened to an AND-join list by D29/cambia-1713). An empty list
	// means no dependency; allowed on every kind. The dependent launches only
	// once every named parent has reached a clean terminal; any parent reaching a
	// non-success terminal routes the whole dependent through OnFailure. Every
	// parent must exist at submit (validated in the handler); the gate resolves
	// each parent's outcome at dispatch time. The wire shape accepts a bare
	// string (the pre-r2 single-parent shape) or a list of 0..N names; see
	// UnmarshalJSON.
	After []string `json:"after,omitempty"`
	// OnFailure selects the failure-branch behavior when After's parent reaches a
	// non-success terminal (crashed/failed/canceled/skipped, a graceful-stop with
	// a nonzero exit, or a purged run dir): skip (default) marks this job
	// StateSkipped, run launches it anyway, fail marks it StateFailed. Empty means
	// skip. Inert without After.
	OnFailure string `json:"on_failure,omitempty"`
	// SubmitSeq is the server-assigned monotonic admission sequence, persisted in
	// jobspec.json so the FIFO order can be rebuilt after a daemon restart (queue
	// persistence). Assigned by Submit; never supplied by the client.
	SubmitSeq int64 `json:"submit_seq,omitempty"`
	// HubItem is an optional Codebridge hub work-item handle (cambia-353), e.g.
	// "cambia-359". Telemetry-only provenance: the runner never acts on it; it is
	// carried so writeJobSpec persists it into jobspec.json and it surfaces in the
	// JobView, letting the client-side reflector link a job's note to a hub item
	// (recoverable from a pulled run dir). Empty means an unlinked job.
	HubItem string `json:"hub_item,omitempty"`
	// Script is the worktree-relative path to a kind=measure job's pinned
	// script (design D38), e.g. "cfr/scripts/measure_gate_gap.py". Required for
	// measure, forbidden for every other kind: it must resolve under
	// measureScriptRoot and must exist at the pinned commit (checked at launch,
	// once the worktree is staged).
	Script string `json:"script,omitempty"`
	// Args is a kind=measure job's argv tail, appended verbatim after the
	// staged script path with no shell involved (design D38). Forbidden for
	// every other kind.
	Args []string `json:"args,omitempty"`
	// Reads names other run directories a kind=measure job reads as read-only
	// seeds (design D38), resolved through pathguard against the runs dir and
	// required to already exist at submit. Exported to the job process as
	// CAMBIA_MEASURE_READ_DIRS (os.pathsep-joined). Forbidden for every other
	// kind.
	Reads []string `json:"reads,omitempty"`
	// Exclusive marks a timing-sensitive job that must run alone (cambia-655): the
	// dispatcher launches it only when no other job is active and holds every other
	// job while it prepares or runs. Absent decodes false (a normal job that shares
	// the concurrency pool). A deferred exclusive job at the queue head barriers the
	// later ready jobs of that dispatch pass so a stream of small jobs cannot pass
	// it and starve it (see dispatchLocked). The submit path never uses
	// DisallowUnknownFields, so an old client (no field) still submits and a new
	// client against an old daemon is the only degradation: the field is dropped
	// and the job runs shared.
	Exclusive bool `json:"exclusive,omitempty"`
	// Requires is the placement constraint block of D10, matched against a
	// node's grant-clamped capability declaration. Absent means the defaults
	// derived from the spec alone (device from device(), min_cores 1,
	// needs_libcambia true), which is what every v1.0 spec places under.
	Requires *capability.Requires `json:"requires,omitempty"`
	// MaxRuntimeHours optionally lowers the pool's own lease-lifetime cap for
	// this job (D4). Zero means the pool cap alone applies. It is a coordinator
	// bound, checked from granted_at regardless of renewals, and is not the
	// node's own job_policy gate.
	MaxRuntimeHours float64 `json:"max_runtime_hours,omitempty"`
}

// jobSpecAlias has JobSpec's exact field set but none of its methods, so
// decoding into it cannot recurse into JobSpec.UnmarshalJSON.
type jobSpecAlias JobSpec

// UnmarshalJSON decodes a JobSpec, accepting `after` as either a bare string
// (the pre-r2 single-parent wire shape) or a JSON array of 0..N parent names
// (D29 fan-in, cambia-1713), so a jobspec.json written before this change still
// decodes (AC1). Every other field keeps standard decoding, including the
// json.Number preservation for Overrides that decodeJSON's UseNumber() gives
// the request path: since a custom UnmarshalJSON receives only raw bytes with
// no access to the caller's decoder options, this re-establishes UseNumber()
// on an inner decoder rather than losing it.
func (s *JobSpec) UnmarshalJSON(data []byte) error {
	var raw struct {
		jobSpecAlias
		After json.RawMessage `json:"after,omitempty"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	*s = JobSpec(raw.jobSpecAlias)
	s.After = nil
	if len(raw.After) == 0 || string(raw.After) == "null" {
		return nil
	}
	var single string
	if err := json.Unmarshal(raw.After, &single); err == nil {
		s.After = []string{single}
		return nil
	}
	var list []string
	if err := json.Unmarshal(raw.After, &list); err != nil {
		return fmt.Errorf("after: must be a string or an array of strings: %w", err)
	}
	s.After = list
	return nil
}

// maxDependencyDepth caps the length of an after-chain a submission may extend
// (D29): parents must already be admitted at submit, which makes the graph a
// DAG by construction, and this bounds how long a chain within that DAG may
// grow.
const maxDependencyDepth = 32

// dependencyDepth returns the longest after-chain ending at a job whose direct
// parents are `after`, reading each ancestor's persisted jobspec.json from
// runsDir. budget is how many more ancestor levels may be counted below this
// point; it returns ok=false the instant a still-unexplored parent would need
// more budget than remains, so a submission with too deep an ancestry fails
// fast instead of walking the rest of the DAG. The top-level caller passes
// budget=maxDependencyDepth. Called at submit time only; every named parent
// already exists on disk by then (validated before this runs).
func dependencyDepth(runsDir string, after []string, budget int) (int, bool) {
	if len(after) == 0 {
		return 0, true
	}
	if budget <= 0 {
		return 0, false
	}
	depth := 0
	for _, parent := range after {
		var parentAfter []string
		if spec := readJobSpec(filepath.Join(runsDir, parent)); spec != nil {
			parentAfter = spec.After
		}
		d, ok := dependencyDepth(runsDir, parentAfter, budget-1)
		if !ok {
			return 0, false
		}
		if d+1 > depth {
			depth = d + 1
		}
	}
	return depth, true
}

// onFailureOrDefault returns the spec's on_failure policy, defaulting to skip.
func (s *JobSpec) onFailureOrDefault() string {
	if s.OnFailure == "" {
		return OnFailureSkip
	}
	return s.OnFailure
}

// onFailureValid reports whether on_failure is empty (default skip) or one of
// the three policy values. Shape validation only.
func (s *JobSpec) onFailureValid() bool {
	switch s.OnFailure {
	case "", OnFailureSkip, OnFailureRun, OnFailureFail:
		return true
	}
	return false
}

// device returns the resolved device, defaulting to cpu (the v1 baseline; cuda
// and xpu are opt-in per runner via the capability gate below).
func (s *JobSpec) device() string {
	if s.Device == "" {
		return "cpu"
	}
	return s.Device
}

// validDevices is the runner-supported device set. Anything else fails the
// shape check in handleCreateJob before the per-runner capability gate
// (Server.allowedDevices) even runs.
var validDevices = map[string]bool{
	"cpu":  true,
	"cuda": true,
	"xpu":  true,
}

// deviceValid reports whether the spec's resolved device is one of the
// runner-supported values. This is shape validation only; whether the value is
// enabled on THIS runner is the separate capability gate.
func (s *JobSpec) deviceValid() bool {
	return validDevices[s.device()]
}

// gamesOrDefault returns the evaluate/head-to-head games count, defaulting to
// 5000 (the run-dir-mode default cli.py falls back to, cfr/src/cli.py
// evaluate) when the spec left it unset. head-to-head's own bare-CLI default
// is 2000, but the runner always passes --games explicitly, so this default
// governs both kinds.
func (s *JobSpec) gamesOrDefault() int {
	if s.Games <= 0 {
		return 5000
	}
	return s.Games
}

// targetForbidden reports whether the spec sets target on a kind that forbids
// it (design 2.6: target selects what an evaluate job evaluates; train has no
// use for it).
func (s *JobSpec) targetForbidden() bool {
	return s.Kind == KindTrain && s.Target != ""
}

// warmStartForbidden reports whether the spec sets warm_start on a kind that
// forbids it (design cambia-334): warm_start initializes a train job from
// another run's staged snapshot; evaluate, head-to-head, and bench have no use
// for it.
func (s *JobSpec) warmStartForbidden() bool {
	return s.Kind != KindTrain && s.WarmStart != ""
}

// scriptRequired reports whether kind=measure has left script unset (design
// D38: script is required for measure).
func (s *JobSpec) scriptRequired() bool {
	return s.Kind == KindMeasure && s.Script == ""
}

// scriptForbidden reports whether script is set on a kind other than measure
// (design D38: script is measure-only, like target is evaluate-only).
func (s *JobSpec) scriptForbidden() bool {
	return s.Kind != KindMeasure && s.Script != ""
}

// scriptRootValid reports whether script (already passed through
// pathguard.CheckRel, so it carries no ".." segment and is not absolute)
// lexically resolves under measureScriptRoot. This is a fixed-literal-prefix
// check, not filesystem containment: the worktree does not exist yet at
// submit time. The staged file's existence is checked at launch instead.
func (s *JobSpec) scriptRootValid() bool {
	return strings.HasPrefix(s.Script, measureScriptRoot) && s.Script != measureScriptRoot
}

// argsForbidden reports whether args is set on a kind other than measure
// (design D38).
func (s *JobSpec) argsForbidden() bool {
	return len(s.Args) > 0 && s.Kind != KindMeasure
}

// validateArgs rejects a measure job's args entries containing a NUL byte or
// exceeding maxMeasureArgLen (design D38). A no-op when args is empty (every
// other kind, or a measure job with no argv tail).
func (s *JobSpec) validateArgs() error {
	for i, a := range s.Args {
		if strings.IndexByte(a, 0) >= 0 {
			return fmt.Errorf("args[%d]: contains a NUL byte", i)
		}
		if len(a) > maxMeasureArgLen {
			return fmt.Errorf("args[%d]: exceeds %d bytes", i, maxMeasureArgLen)
		}
	}
	return nil
}

// readsForbidden reports whether reads is set on a kind other than measure
// (design D38).
func (s *JobSpec) readsForbidden() bool {
	return len(s.Reads) > 0 && s.Kind != KindMeasure
}

// containedReads returns a measure job's reads entries for the same
// containment-resolve guard as containedTarget/containedWarmStart (design
// D38): each read names an existing run directory under the runs dir, so its
// containment base is known at submit time.
func (s *JobSpec) containedReads() []struct{ label, value string } {
	if s.Kind != KindMeasure {
		return nil
	}
	out := make([]struct{ label, value string }, len(s.Reads))
	for i, r := range s.Reads {
		out[i] = struct{ label, value string }{fmt.Sprintf("reads[%d]", i), r}
	}
	return out
}

// overridesStr renders the dotted-key overrides as a string map for the ingest
// Prepare call (which renders them into the config). Numeric values from a
// json.Number decode stringify without a trailing ".0".
func (s *JobSpec) overridesStr() map[string]string {
	if len(s.Overrides) == 0 {
		return nil
	}
	out := make(map[string]string, len(s.Overrides))
	for _, k := range sortedKeys(s.Overrides) {
		out[k] = stringifyOverride(s.Overrides[k])
	}
	return out
}

// guardedPaths returns the spec's config path (any kind, when set) and a train
// job's warm_start (when set) for the lexical CheckRel guard in the design 2.6
// validation order (after the kind allowlist). Config's containment base (the
// job worktree) is not staged until ingest render, so config gets only the
// lexical guard at submit. warm_start's containment base (the runs dir) is
// already known at submit, so it also goes through containedWarmStart below;
// listing it here too just fails it fast with the same invalid_path code
// before the containment step runs. Checkpoints and target are guarded by
// their contained* helpers + pathguard.Resolve, which layers containment over
// the same lexical check against the already-known runs dir.
func (s *JobSpec) guardedPaths() []struct{ label, value string } {
	var paths []struct{ label, value string }
	if s.Config != "" {
		paths = append(paths, struct{ label, value string }{"config", s.Config})
	}
	if s.WarmStart != "" {
		paths = append(paths, struct{ label, value string }{"warm_start", s.WarmStart})
	}
	return paths
}

// containedCheckpoints returns the head-to-head checkpoint spec fields that must
// resolve inside the runner runs dir (design 5.4). It is the containment half of
// the guard, layered over the lexical CheckRel in guardedPaths: checkpoints name
// existing staged run dirs under runsDir, so their containment base is known at
// submit time. config is not returned here because at submit its containment base
// (the job worktree) is not yet staged; config keeps the lexical guard only until
// ingest render.
func (s *JobSpec) containedCheckpoints() []struct{ label, value string } {
	if s.Kind != KindHeadToHead {
		return nil
	}
	return []struct{ label, value string }{
		{"checkpoint_a", s.CheckpointA},
		{"checkpoint_b", s.CheckpointB},
	}
}

// containedTarget returns the evaluate target field for the same
// containment-resolve guard as containedCheckpoints (design 5.4): target
// names a runner-local run dir or checkpoint file, so its containment base
// (the runs dir) is known at submit time. An empty target on an evaluate spec
// also fails Resolve's underlying empty-path check, so "required for
// evaluate" is enforced through the same guard checkpoint_a/b already use.
func (s *JobSpec) containedTarget() []struct{ label, value string } {
	if s.Kind != KindEvaluate {
		return nil
	}
	return []struct{ label, value string }{{"target", s.Target}}
}

// containedWarmStart returns the train-only warm_start spec field for the same
// containment-resolve guard as containedCheckpoints/containedTarget (design
// 5.4, cambia-334): warm_start names another run's staged snapshot file, so
// its containment base (the runs dir) is known at submit time. It is
// optional: an empty warm_start (or a non-train kind, already rejected by
// warmStartForbidden before this runs) adds no entry, unlike containedTarget's
// implicit required-ness for evaluate.
func (s *JobSpec) containedWarmStart() []struct{ label, value string } {
	if s.Kind != KindTrain || s.WarmStart == "" {
		return nil
	}
	return []struct{ label, value string }{{"warm_start", s.WarmStart}}
}

// sortedKeys returns m's keys sorted, for deterministic override ordering.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
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
	case float64:
		// A plain JSON decode (no UseNumber) yields float64; render integers
		// without a trailing ".0".
		if x == float64(int64(x)) {
			return fmt.Sprintf("%d", int64(x))
		}
		return fmt.Sprintf("%g", x)
	default:
		return fmt.Sprintf("%v", v)
	}
}
