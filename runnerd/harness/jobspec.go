package harness

import (
	"encoding/json"
	"fmt"
	"sort"

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
)

// terminalStates is the set of states from which a job does not transition
// further (resume creates a new launch, not a transition). It spans both the
// runnerd-level terminals and the procmgr terminals.
var terminalStates = map[string]bool{
	StateCanceled:         true,
	StateFailed:           true,
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
)

// HarnessAlgorithms returns the runnerd job-kind -> cambia-subcommand allowlist
// injected into the ProcessManager. It is the superset the daemon supervises
// (the dashboard registers only TrainAlgorithms). A fresh map is returned per
// call so callers cannot mutate a shared instance. kind=train resolves to
// `cambia train prtcfr`, encoding the v1 prtcfr-only train allowlist.
func HarnessAlgorithms() map[string][]string {
	return map[string][]string{
		KindTrain:      {"train", "prtcfr"},
		KindEvaluate:   {"evaluate"},
		KindHeadToHead: {"head-to-head"},
		KindBench:      {"bench"},
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

// gamesOrDefault returns the evaluate games count, defaulting to 5000 (the
// run-dir-mode default cli.py falls back to, cfr/src/cli.py evaluate) when the
// spec left it unset.
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
