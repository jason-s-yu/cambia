package nodeagent

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
)

// Launch is the node-side launch description: the argv, cwd, interpreter, and
// harness environment for one job.
type Launch struct {
	Python string
	Argv   []string
	Cwd    string
	Env    []string
	Resume bool
}

// AlgoResolver maps a job kind to its cambia subcommand. It is injected
// because the coordinator and a node hold the same allowlist through different
// doors: the coordinator resolves through the table it gave its own
// ProcessManager, which its suites replace with fake kinds, and a node
// resolves through Subcommand below. The table is the only parameter of the
// launch template; the argv it builds is one piece of code either way.
type AlgoResolver func(kind string) ([]string, error)

// BuildLaunch builds the parameterized launch for one staged job. It is the
// single per-kind argv template (D1): the coordinator's embedded node and a
// remote node run the same code, so a job's command line does not depend on
// where it was placed.
func BuildLaunch(spec Spec, prepared *ingestapi.Prepared, runsDir string, algo AlgoResolver) (Launch, error) {
	if prepared == nil || prepared.VenvPython == "" {
		return Launch{}, fmt.Errorf("staged environment carries no interpreter")
	}
	if algo == nil {
		algo = Subcommand
	}
	runDir := filepath.Join(runsDir, spec.Name)
	if spec.Kind == KindMeasure {
		return measureLaunch(spec, prepared, runsDir)
	}

	sub, err := algo(spec.Kind)
	if err != nil {
		return Launch{}, err
	}
	argv := append([]string{"-m", "src.cli"}, sub...)

	switch spec.Kind {
	case KindEvaluate:
		// evaluate takes a positional run-dir target and no --config: run-dir
		// mode reads the target's own config.yaml. --metrics-dir points
		// metrics.jsonl and evaluations/ at the job's own run dir, which is
		// the dir the lease holds; on a node the seeded target is read-only,
		// and on the coordinator the target is a run this job does not own
		// either way (D64). The eval rows reach the target through the
		// client's merge, not through a cross-job write.
		tail, terr := evaluateArgv(spec, runsDir)
		if terr != nil {
			return Launch{}, terr
		}
		argv = append(argv, tail...)
		argv = append(argv, "--metrics-dir", runDir)
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
			// Without an explicit name the trainer registers its run_db row
			// under a config-derived default, decoupling the journal row from
			// the run dir the reconciler replays. Without an explicit save
			// path it resolves runs/<name> against the worktree cwd, so
			// resume_state.json and metrics.jsonl land in the worktree and die
			// with its cleanup. Both were found live in the M5 battery.
			argv = append(argv, "--run-name", spec.Name, "--save-path", runDir)
		}
	}
	if spec.Resume {
		argv = append(argv, "--resume")
	}

	// CAMBIA_RUN_DB points every run_db write of the job process at the job's
	// own per-run journal, an evaluate job included (D64): a lease owns its
	// own run dir and nothing else, and the journal's single runs row is named
	// for the evaluated target, which the coordinator's own validator asserts
	// (D55).
	env := append([]string(nil), prepared.Env...)
	env = append(env, "CAMBIA_RUN_DB="+filepath.Join(runDir, runDBName))
	return Launch{
		Python: prepared.VenvPython,
		Argv:   argv,
		Cwd:    filepath.Join(prepared.WorktreeDir, "cfr"),
		Env:    env,
		Resume: spec.Resume,
	}, nil
}

// measureLaunch builds a measure job's launch (D38): the staged script path
// followed by its args verbatim, with no "-m src.cli" prefix, no cambia
// subcommand, and no shell. The staged script's presence at the pinned commit
// is checked here, once the worktree exists; its absence fails the job with a
// named error before launch. Its read-only seeds are exported as
// CAMBIA_MEASURE_READ_DIRS, os.pathsep-joined, so the script locates them
// without re-deriving the runs dir itself.
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
// --latest, --games, --device. The target is re-resolved through the same
// containment guard it got at submit, since launch happens in a later
// goroutine against the persisted spec. File mode is refused: cli.py leaves
// agent_type at its deep_cfr default and only recovers a run dir under
// checkpoints/, so a PRT-CFR snapshot would evaluate under the wrong wrapper
// and report plausible, wrong numbers instead of failing.
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
		return nil, &specFatalError{reason: fmt.Sprintf(
			"target %q: evaluate requires a run directory, not a checkpoint file (file mode misdetects agent type)",
			spec.Target)}
	}
	return []string{targetAbs, "--latest", "--games", strconv.Itoa(spec.games()), "--device", spec.device()}, nil
}

// headToHeadArgv builds the head-to-head tail: two resolved checkpoints,
// --games, --device. Unlike evaluate's target, a checkpoint has no
// dir-vs-file ambiguity to guard against: `cambia head-to-head` declares both
// as typer Path(exists=True), so a missing one fails at launch with a clear
// CLI error rather than a silent misinterpretation.
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

// Subcommand maps a job kind to its cambia subcommand. It is the node's own
// copy of the allowlist the coordinator injects into its ProcessManager, and
// the default AlgoResolver.
func Subcommand(kind string) ([]string, error) {
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
