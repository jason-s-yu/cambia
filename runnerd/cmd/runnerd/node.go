package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/jason-s-yu/cambia/runnerd/harness"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Role values for --role / RUNNERD_ROLE (design D1). One binary, three roles:
// the coordinator owns the queue, placement, leases, quarantine, and the API;
// a node owns gate evaluation, snapshot fetch, ingest, procmgr, uploads, and
// the result post; both runs a coordinator with one embedded node.
const (
	roleCoordinator = "coordinator"
	roleNode        = "node"
	roleBoth        = "both"
)

// validRole reports whether the role string names a role.
func validRole(role string) bool {
	switch role {
	case roleCoordinator, roleNode, roleBoth:
		return true
	}
	return false
}

// runNode is the --role node entry point. A node opens no listener, holds no
// ssh key, and needs none of the coordinator's TLS or JWT environment: its
// complete set of secrets is its own ed25519 key and the coordinator's pinned
// certificate fingerprint (D27).
func runNode(configPath string) {
	if configPath == "" {
		log.Fatal("--role node requires --node-config (or RUNNERD_NASHNET_CONFIG): the nashnet: section of the node's yaml")
	}
	cfg, err := nodeagent.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("node config: %v", err)
	}

	priv, created, err := nodeagent.LoadOrCreateKey(cfg.KeyPath)
	if err != nil {
		log.Fatalf("node key: %v", err)
	}
	signer, err := nodeagent.NewSigner(priv, nil)
	if err != nil {
		log.Fatalf("node key: %v", err)
	}
	if created {
		// Enrollment is an operator-signed grant, not a join endpoint (D60), so
		// a fresh key is useless until the operator mints one against it.
		log.Printf("minted a new node key at %s", cfg.KeyPath)
		log.Printf("node id %s; enroll with: cambia harness node grant --pubkey %s",
			signer.NodeID(), signer.PublicKeyBase64())
	}

	client, err := nodeagent.NewClient(cfg, signer)
	if err != nil {
		log.Fatalf("coordinator client: %v", err)
	}
	if err := os.MkdirAll(cfg.RunsDir, 0o755); err != nil {
		log.Fatalf("create runs dir %q: %v", cfg.RunsDir, err)
	}

	pm := procmgr.NewProcessManager(cfg.RunsDir, "", "cambia",
		harness.NewRunResolver(cfg.RunsDir), harness.HarnessAlgorithms())
	pm.SetMaxConcurrent(cfg.Slots)

	agent, err := nodeagent.New(nodeagent.Options{
		Config:            cfg,
		Signer:            signer,
		Client:            client,
		Env:               nodeagent.NewEnvironment(cfg, runtime.NumCPU()-2),
		Launcher:          nodeagent.NewLauncher(pm),
		CanBuildLibcambia: capability.ProbeCanBuildLibcambia(nodeagent.GoToolchainPin, nodeagent.ExecRunFunc),
	})
	if err != nil {
		log.Fatalf("node agent: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// The same job-preserving signal policy the coordinator uses: SIGTERM (a
	// redeploy) leaves the job process groups running for the next incarnation
	// to reattach, SIGINT takes them down with the agent.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-stop
		if killJobsOnSignal(sig, os.Getenv("RUNNERD_KILL_JOBS_ON_STOP")) {
			log.Printf("shutdown (%s): killing %d job process groups", sig, pm.SupervisedCount())
			pm.KillAll()
		} else {
			log.Printf("shutdown (%s): leaving %d job process groups running for reattach", sig, pm.SupervisedCount())
		}
		cancel()
		os.Exit(0)
	}()

	log.Printf("nashnet node %s serving %s (runs=%s, slots=%d)",
		signer.NodeID(), cfg.Coordinator.URL, cfg.RunsDir, cfg.Slots)
	if err := agent.Run(ctx); err != nil {
		log.Fatalf("node agent: %v", err)
	}
}
