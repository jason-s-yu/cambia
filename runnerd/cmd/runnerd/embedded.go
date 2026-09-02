package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// embeddedKeyFile is where --role both keeps the key its own node acts under.
// It is derived-id material like any node key (D60), and it lives beside the
// coordinator's other state rather than in the operator's nodes directory: the
// grant that admits it is authored in memory, never as a file an operator
// would have to mint or could accidentally delete.
const embeddedKeyFile = "embedded-node.key"

// embeddedSettings is what the coordinator's own environment says about the
// host it runs on. Every value already exists for the v1.0 daemon; --role both
// re-reads them as the embedded node's declaration and gates rather than
// introducing a second set of knobs (D40).
type embeddedSettings struct {
	baseDir string
	runsDir string
	// slots is RUNNERD_MAX_CONCURRENT_JOBS: the node's declared concurrency
	// and, through gates.concurrency, its own enforcement of the same number.
	slots int
	// devices is RUNNERD_ALLOWED_DEVICES, which becomes devices_allowed. It is
	// the node's gate now, not the submit gate: a job for a device no node
	// offers waits instead of being refused (D40).
	devices map[string]bool
	// minRAMGB and minDiskGB are RUNNERD_MIN_FREE_RAM_GB and
	// RUNNERD_MIN_FREE_DISK_GB, which become the node's floor gates (D19).
	minRAMGB  float64
	minDiskGB float64
	// requireSignedCommits and allowedSignersPath reach the node's own ingest
	// Manager, so the pin is checked by the same code either side of the move.
	requireSignedCommits bool
	allowedSignersPath   string
}

// embeddedNode is the node half of --role both before it has a transport: the
// key, the derived id, and the config. It is built before the coordinator pool
// because the pool has to know which node id materializes in place and which
// public key its in-process grant admits.
type embeddedNode struct {
	cfg    nodeagent.Config
	signer *nodeagent.Signer
}

// newEmbeddedNode loads or mints the embedded node's key and derives its
// config from the coordinator's own settings. Nothing here is operator-facing:
// a fresh key needs no enrollment, because the coordinator admits it through
// the in-process grant rather than through a signed file (D40, D60).
func newEmbeddedNode(s embeddedSettings) (*embeddedNode, error) {
	keyPath := filepath.Join(s.baseDir, "nashnet", embeddedKeyFile)
	priv, created, err := nodeagent.LoadOrCreateKey(keyPath)
	if err != nil {
		return nil, err
	}
	signer, err := nodeagent.NewSigner(priv, nil)
	if err != nil {
		return nil, err
	}
	if created {
		log.Printf("nashnet: minted the embedded node key at %s", keyPath)
	}

	slots := s.slots
	cfg := nodeagent.Config{
		NodeID:  signer.NodeID(),
		KeyPath: keyPath,
		// The node shares the coordinator's base dir and runs dir. That
		// sharing is the whole of in-place execution: the mirror, the
		// worktrees, the venv and libcambia caches, and the run dirs are
		// already where the launch looks for them, so nothing is fetched and
		// nothing is promoted (D40).
		BaseDir:              s.baseDir,
		RunsDir:              s.runsDir,
		Slots:                slots,
		RequireSignedCommits: s.requireSignedCommits,
		AllowedSignersPath:   s.allowedSignersPath,
		Gates: gates.Config{
			DevicesAllowed: sortedDevices(s.devices),
			Concurrency:    &gates.ConcurrencyGate{MaxSlots: &slots},
			Floors: &gates.FloorsGate{
				FreeRAMGB:  floatOrNil(s.minRAMGB),
				FreeDiskGB: floatOrNil(s.minDiskGB),
			},
		},
	}
	if err := cfg.NormalizeEmbedded(); err != nil {
		return nil, err
	}
	return &embeddedNode{cfg: cfg, signer: signer}, nil
}

// nodeID is the id the coordinator knows this node by.
func (e *embeddedNode) nodeID() string { return e.signer.NodeID() }

// run starts the agent against the coordinator's own routed handler and blocks
// until ctx is done. handler is Server.Handler(): the request reaches the same
// mux, the same credential middlewares, the same fence, and the same commit
// transaction a remote node's request reaches, with the socket and the TLS
// handshake the only things missing (D65).
func (e *embeddedNode) run(ctx context.Context, handler http.Handler, pm *procmgr.ProcessManager) error {
	agent, err := nodeagent.New(nodeagent.Options{
		Config:            e.cfg,
		Signer:            e.signer,
		Client:            nodeagent.NewLoopbackClient(handler, e.signer),
		Env:               nodeagent.NewEmbeddedEnvironment(e.cfg, runtime.NumCPU()-2),
		Launcher:          nodeagent.NewLauncher(pm),
		CanBuildLibcambia: capability.ProbeCanBuildLibcambia(nodeagent.GoToolchainPin, nodeagent.ExecRunFunc),
		InPlace:           true,
	})
	if err != nil {
		return fmt.Errorf("embedded node: %w", err)
	}
	return agent.Run(ctx)
}

// sortedDevices renders an allowed-devices set as the node's devices_allowed
// gate, stable so a restart does not churn the declaration.
func sortedDevices(devices map[string]bool) []string {
	out := make([]string, 0, len(devices))
	for d, ok := range devices {
		if ok {
			out = append(out, d)
		}
	}
	sort.Strings(out)
	return out
}

// floatOrNil turns a non-positive floor into an unset one, so a disabled env
// leaves the gate absent rather than declaring a floor of zero.
func floatOrNil(v float64) *float64 {
	if v <= 0 {
		return nil
	}
	return &v
}
