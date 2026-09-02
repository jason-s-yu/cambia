package nodeagent

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Job kinds, mirrored from the harness job-spec vocabulary. The node decodes
// the spec from the claim response as JSON rather than importing the harness
// package, which would be a cycle once the dispatcher's launch path moves into
// this package (D1).
const (
	KindTrain      = "train"
	KindEvaluate   = "evaluate"
	KindHeadToHead = "head-to-head"
	KindBench      = "bench"
	KindMeasure    = "measure"
)

// defaultGames is the evaluate/head-to-head game count when the spec sets none.
const defaultGames = 5000

// Spec is the launch template's view of a job: the node's decode of the
// persisted JobSpec a claim carries, and what the coordinator maps its own
// JobSpec onto before it launches an embedded run. Only the fields a launch
// acts on are declared; anything else in the coordinator's spec is ignored
// rather than re-validated, since the coordinator already admitted it.
type Spec struct {
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
	WarmStart   string         `json:"warm_start"`
	Exclusive   bool           `json:"exclusive,omitempty"`
	Script      string         `json:"script,omitempty"`
	Args        []string       `json:"args,omitempty"`
	Reads       []string       `json:"reads,omitempty"`
}

// decodeSpec decodes the claim's spec and checks the one field the node itself
// turns into a path: the run name.
func decodeSpec(raw json.RawMessage) (Spec, error) {
	var s Spec
	if len(raw) == 0 {
		return s, fmt.Errorf("claim carried no spec")
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		return s, fmt.Errorf("decode spec: %w", err)
	}
	if err := procmgr.ValidateName(s.Name); err != nil {
		return s, fmt.Errorf("spec name: %w", err)
	}
	if s.Kind == "" {
		return s, fmt.Errorf("spec carries no kind")
	}
	return s, nil
}

// device is the spec's device, defaulting to cpu exactly as the coordinator's
// JobSpec.device does, so the venv extra and the config rail agree across the
// two sides.
func (s Spec) device() string {
	if s.Device == "" {
		return "cpu"
	}
	return s.Device
}

// games returns the spec's game count or the default.
func (s Spec) games() int {
	if s.Games > 0 {
		return s.Games
	}
	return defaultGames
}

// overridesStr renders the dotted-key overrides as the string map ingest's
// render step consumes, in sorted key order so a rendered config is stable.
func (s Spec) overridesStr() map[string]string {
	if len(s.Overrides) == 0 {
		return nil
	}
	keys := make([]string, 0, len(s.Overrides))
	for k := range s.Overrides {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(map[string]string, len(keys))
	for _, k := range keys {
		out[k] = stringifyOverride(s.Overrides[k])
	}
	return out
}

// stringifyOverride renders one override value. json.Number is preserved
// verbatim so an integer override stays integral through the render.
func stringifyOverride(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case bool:
		return strconv.FormatBool(t)
	case json.Number:
		return t.String()
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Sprint(t)
		}
		return string(b)
	}
}

// fileExists reports whether path names something on disk.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
