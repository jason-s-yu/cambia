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
	Games       int            `json:"games"`
	Priority    string         `json:"priority"`
	Force       bool           `json:"force"`
}

// device returns the resolved device, defaulting to cpu (v1 is cpu-only).
func (s *JobSpec) device() string {
	if s.Device == "" {
		return "cpu"
	}
	return s.Device
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

// guardedPaths returns the spec's file-path fields that require path guarding,
// keyed by kind. config is guarded for train/evaluate/bench; the two checkpoints
// for head-to-head. The handler runs these through pathguard.CheckRel in the
// design 2.6 validation order (after the name-collision check).
func (s *JobSpec) guardedPaths() []struct{ label, value string } {
	var out []struct{ label, value string }
	switch s.Kind {
	case KindHeadToHead:
		out = append(out,
			struct{ label, value string }{"checkpoint_a", s.CheckpointA},
			struct{ label, value string }{"checkpoint_b", s.CheckpointB},
		)
		if s.Config != "" {
			out = append(out, struct{ label, value string }{"config", s.Config})
		}
	default:
		if s.Config != "" {
			out = append(out, struct{ label, value string }{"config", s.Config})
		}
	}
	return out
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
