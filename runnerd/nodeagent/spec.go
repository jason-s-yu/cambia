package nodeagent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
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

// Spec is the node's decode of the persisted JobSpec the claim carries. Only
// the fields the node acts on are declared; anything else in the coordinator's
// spec is ignored rather than re-validated, since the coordinator already
// admitted it.
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

// Launch is the node-side launch description: the argv, cwd, interpreter, and
// harness environment for one job.
type Launch struct {
	Python string
	Argv   []string
	Cwd    string
	Env    []string
	Resume bool
}

// buildLaunch builds the parameterized launch for one staged job. It mirrors
// the dispatcher's per-kind argv template so a job runs identically wherever
// it is placed; D1 moves that template here in a later ticket, at which point
// this becomes the single copy.
func buildLaunch(spec Spec, prepared *ingestapi.Prepared, runsDir string) (Launch, error) {
	if prepared == nil || prepared.VenvPython == "" {
		return Launch{}, fmt.Errorf("staged environment carries no interpreter")
	}
	runDir := filepath.Join(runsDir, spec.Name)
	if spec.Kind == KindMeasure {
		return measureLaunch(spec, prepared, runsDir)
	}

	sub, err := subcommand(spec.Kind)
	if err != nil {
		return Launch{}, err
	}
	argv := append([]string{"-m", "src.cli"}, sub...)
	journalRun := spec.Name

	switch spec.Kind {
	case KindEvaluate:
		// evaluate takes a positional run-dir target and no --config: run-dir
		// mode reads the target's own config.yaml. Its eval rows join the
		// evaluated run's journal.
		tail, terr := evaluateArgv(spec, runsDir)
		if terr != nil {
			return Launch{}, terr
		}
		argv = append(argv, tail...)
		journalRun = spec.Target
	case KindHeadToHead:
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		tail, herr := headToHeadArgv(spec, runsDir)
		if herr != nil {
			return Launch{}, herr
		}
		argv = append(argv, tail...)
	case KindBench:
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		argv = append(argv, "--output-dir", runDir, "--device", spec.device())
	default:
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		if spec.Kind == KindTrain {
			argv = append(argv, "--run-name", spec.Name, "--save-path", runDir)
		}
	}
	if spec.Resume {
		argv = append(argv, "--resume")
	}

	env := append([]string(nil), prepared.Env...)
	env = append(env, "CAMBIA_RUN_DB="+filepath.Join(runsDir, journalRun, runDBName))
	return Launch{
		Python: prepared.VenvPython,
		Argv:   argv,
		Cwd:    filepath.Join(prepared.WorktreeDir, "cfr"),
		Env:    env,
		Resume: spec.Resume,
	}, nil
}

// measureLaunch builds a measure job's launch (D38): the staged script path
// followed by its args verbatim, with no cambia subcommand and no shell. Its
// read-only seeds are exported as CAMBIA_MEASURE_READ_DIRS.
func measureLaunch(spec Spec, prepared *ingestapi.Prepared, runsDir string) (Launch, error) {
	scriptAbs := filepath.Join(prepared.WorktreeDir, spec.Script)
	if _, err := os.Stat(scriptAbs); err != nil {
		return Launch{}, &specFatalError{reason: fmt.Sprintf("script: not found at pinned commit: %s", spec.Script)}
	}
	argv := append([]string{scriptAbs}, spec.Args...)

	env := append([]string(nil), prepared.Env...)
	env = append(env, "CAMBIA_RUN_DB="+filepath.Join(runsDir, spec.Name, runDBName))
	if len(spec.Reads) > 0 {
		reads := make([]string, 0, len(spec.Reads))
		for i, r := range spec.Reads {
			resolved, rerr := pathguard.Resolve(runsDir, r)
			if rerr != nil {
				return Launch{}, fmt.Errorf("reads[%d]: %w", i, rerr)
			}
			reads = append(reads, resolved)
		}
		env = append(env, "CAMBIA_MEASURE_READ_DIRS="+strings.Join(reads, string(os.PathListSeparator)))
	}
	return Launch{
		Python: prepared.VenvPython,
		Argv:   argv,
		Cwd:    filepath.Join(prepared.WorktreeDir, "cfr"),
		Env:    env,
	}, nil
}

// evaluateArgv builds the evaluate tail: the resolved run-dir target,
// --latest, --games, --device. File mode is refused because it leaves the
// agent type at its deep_cfr default and reports plausible, wrong numbers.
func evaluateArgv(spec Spec, runsDir string) ([]string, error) {
	targetAbs, err := pathguard.Resolve(runsDir, spec.Target)
	if err != nil {
		return nil, fmt.Errorf("target: %w", err)
	}
	info, err := os.Stat(targetAbs)
	if err != nil {
		return nil, fmt.Errorf("target: %w", err)
	}
	if !info.IsDir() {
		return nil, &specFatalError{reason: fmt.Sprintf("target %q: evaluate requires a run directory", spec.Target)}
	}
	return []string{targetAbs, "--latest", "--games", strconv.Itoa(spec.games()), "--device", spec.device()}, nil
}

// headToHeadArgv builds the head-to-head tail: two resolved checkpoints,
// --games, --device.
func headToHeadArgv(spec Spec, runsDir string) ([]string, error) {
	a, err := pathguard.Resolve(runsDir, spec.CheckpointA)
	if err != nil {
		return nil, fmt.Errorf("checkpoint_a: %w", err)
	}
	b, err := pathguard.Resolve(runsDir, spec.CheckpointB)
	if err != nil {
		return nil, fmt.Errorf("checkpoint_b: %w", err)
	}
	return []string{
		"--checkpoint-a", a,
		"--checkpoint-b", b,
		"--games", strconv.Itoa(spec.games()),
		"--device", spec.device(),
	}, nil
}

// subcommand maps a job kind to its cambia subcommand, the same allowlist the
// coordinator injects into its own ProcessManager.
func subcommand(kind string) ([]string, error) {
	switch kind {
	case KindTrain:
		return []string{"train", "prtcfr"}, nil
	case KindEvaluate:
		return []string{"evaluate"}, nil
	case KindHeadToHead:
		return []string{"head-to-head"}, nil
	case KindBench:
		return []string{"benchmark", "all"}, nil
	default:
		return nil, &specFatalError{reason: "unsupported job kind " + strconv.Quote(kind)}
	}
}

// fileExists reports whether path names something on disk.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
