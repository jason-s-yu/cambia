package harness

import (
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// runResolver is runnerd's own thin procmgr.RunResolver: it maps a run name to
// its run dir and applies pid-liveness to a process state. It severs the
// manager's dependency on any concrete store (the dashboard injects its
// TrainingStore instead). Stateless and safe for concurrent use.
type runResolver struct{ runsDir string }

// NewRunResolver builds the thin RunResolver for a runs root.
func NewRunResolver(runsDir string) procmgr.RunResolver { return runResolver{runsDir: runsDir} }

func (r runResolver) RunDir(name string) string { return filepath.Join(r.runsDir, name) }

func (r runResolver) EffectiveStatus(st *procmgr.ProcessState) string {
	return procmgr.EffectiveStatus(st)
}
