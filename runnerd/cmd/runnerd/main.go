// Command runnerd is the cambia serving-harness runner daemon (design 2). It
// serves the HTTPS + Bearer-JWT control plane, supervises training/eval jobs
// through the extracted procmgr process layer, and drives the in-memory FIFO
// queue and job state machine. It never serves plaintext and never auto-launches
// at startup: it reconciles inherited state and reports, and an operator resumes
// explicitly (design 2.3/6).
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/harness"
	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// buildCommit is the runnerd source commit, stamped at link time with
// -ldflags "-X main.buildCommit=<sha>" and reported by GET /harness/health so an
// operator can tell which binary a running daemon is. It stays "dev" for an
// unstamped local build.
var buildCommit = "dev"

// killJobsOnSignal decides whether a shutdown signal takes the job process
// groups down with the daemon (cambia-655). SIGTERM is systemd's stop/restart
// signal: a redeploy must never kill a training job, so it detaches and leaves
// the process groups running for the next incarnation to reattach at Reconcile.
// SIGINT is the interactive Ctrl-C of a foreground dev daemon, where an orphaned
// job would be a surprise, so it keeps the abrupt SIGKILL behavior. env is
// RUNNERD_KILL_JOBS_ON_STOP: a true value restores kill-on-SIGTERM for an
// operator who wants a stop to take everything down.
func killJobsOnSignal(sig os.Signal, env string) bool {
	if sig == syscall.SIGINT {
		return true
	}
	if b, err := strconv.ParseBool(env); err == nil && b {
		return true
	}
	return false
}

func main() {
	// `cambia-runnerd node <verb>` (currently just init, design D25) is
	// offline operator tooling with its own flag set, dispatched ahead of the
	// daemon's --listen/--role/--node-config flags so it never needs the
	// coordinator's TLS/JWT environment.
	if len(os.Args) > 1 && os.Args[1] == "node" {
		runNodeSubcommand(os.Args[2:])
		return
	}

	listen := flag.String("listen", envOr("RUNNERD_LISTEN", "127.0.0.1:8090"),
		"control-plane listen address (dev default 127.0.0.1:8090; prod binds the runner's LAN address)")
	role := flag.String("role", envOr("RUNNERD_ROLE", roleBoth),
		"coordinator | node | both: one binary, three roles (design D1)")
	nodeConfig := flag.String("node-config", os.Getenv("RUNNERD_NASHNET_CONFIG"),
		"path to the node's yaml, whose nashnet: section configures --role node")
	flag.Parse()

	if !validRole(*role) {
		log.Fatalf("--role %q: want coordinator, node, or both", *role)
	}
	if *role == roleNode {
		runNode(*nodeConfig)
		return
	}

	baseDir := envOr("RUNNERD_BASE_DIR", "/srv/cambia")
	runsDir := envOr("RUNNERD_RUNS_DIR", "/srv/cambia/runs")
	cfrDir := envOr("RUNNERD_CFR_DIR", "/srv/cambia/cfr")
	cambiaBin := envOr("RUNNERD_CAMBIA_BIN", "cambia")
	allowedOrigin := os.Getenv("RUNNERD_ALLOWED_ORIGIN")

	pubKeyPath := os.Getenv("RUNNERD_JWT_PUBKEY")
	tlsCert := os.Getenv("RUNNERD_TLS_CERT")
	tlsKey := os.Getenv("RUNNERD_TLS_KEY")

	// Defensive env parsing (service main.go:154-164 idiom): a malformed or
	// non-positive value falls back to the safe default rather than disabling a
	// rail or the cap.
	maxJobs := 1
	if v := os.Getenv("RUNNERD_MAX_CONCURRENT_JOBS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxJobs = n
		}
	}
	maxQueue := 128
	if v := os.Getenv("RUNNERD_MAX_QUEUE_DEPTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxQueue = n
		}
	}
	minRAM := harness.DefaultMinFreeRAMGB
	if v := os.Getenv("RUNNERD_MIN_FREE_RAM_GB"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			minRAM = f
		}
	}
	minDisk := harness.DefaultMinFreeDiskGB
	if v := os.Getenv("RUNNERD_MIN_FREE_DISK_GB"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			minDisk = f
		}
	}
	allowedDevices := harness.ParseAllowedDevices(envOr("RUNNERD_ALLOWED_DEVICES", "cpu"))

	// Signed-commit enforcement (cambia-550, W1). Default off: an unset or
	// malformed flag leaves verify-commit disabled, preserving current behavior.
	requireSignedCommits := false
	if v := os.Getenv("RUNNERD_REQUIRE_SIGNED_COMMITS"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			requireSignedCommits = b
		}
	}
	allowedSignersPath := os.Getenv("RUNNERD_ALLOWED_SIGNERS_PATH")

	if pubKeyPath == "" {
		log.Fatal("RUNNERD_JWT_PUBKEY is required (verify-only ed25519 public key)")
	}
	if tlsCert == "" || tlsKey == "" {
		log.Fatal("RUNNERD_TLS_CERT and RUNNERD_TLS_KEY are required (HTTPS-only control plane)")
	}
	if allowedOrigin == "" {
		log.Fatal("RUNNERD_ALLOWED_ORIGIN is required (single allowed WS origin; never wildcard)")
	}

	verifier, err := authtoken.Load(pubKeyPath)
	if err != nil {
		log.Fatalf("load JWT public key: %v", err)
	}

	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		log.Fatalf("create runs dir %q: %v", runsDir, err)
	}

	pm := procmgr.NewProcessManager(runsDir, cfrDir, cambiaBin, harness.NewRunResolver(runsDir), harness.HarnessAlgorithms())
	pm.SetMaxConcurrent(maxJobs)

	nashCfg, poolEnabled := loadNashnetConfig(baseDir)
	env := ingest.New(ingest.Config{
		BaseDir:              baseDir,
		RunsDir:              runsDir,
		CoresCap:             runtime.NumCPU() - 2,
		RequireSignedCommits: requireSignedCommits,
		AllowedSignersPath:   allowedSignersPath,
		SnapshotDir:          nashCfg.snapshotDir,
		MaxSnapshots:         nashCfg.snapshotCache,
	})
	disp := harness.NewDispatcher(pm, env, runsDir, maxJobs, maxQueue, 0)

	// Reconcile-then-report: never auto-launch (design 2.3/6).
	disp.Reconcile()

	srv, err := harness.NewServer(harness.ServerConfig{
		Dispatcher:     disp,
		Verifier:       verifier,
		RunsDir:        runsDir,
		AllowedOrigin:  allowedOrigin,
		MinFreeRAMGB:   minRAM,
		MinFreeDiskGB:  minDisk,
		Algos:          harness.HarnessAlgorithms(),
		AllowedDevices: allowedDevices,
		BuildCommit:    buildCommit,
		KillJobsOnStop: killJobsOnSignal(syscall.SIGTERM, os.Getenv("RUNNERD_KILL_JOBS_ON_STOP")),
	})
	if err != nil {
		log.Fatalf("build server: %v", err)
	}

	// The nashnet coordinator is opt-in on RUNNERD_NASHNET_NODES_DIR: with it
	// unset the daemon serves exactly the v1.0 surface (D39, D40) and the
	// dispatcher launches jobs itself, with no node and no lease anywhere.
	if poolEnabled {
		// --role both runs one embedded node over the loopback transport, and
		// it is what makes a pool-enabled daemon run jobs at all: with a pool
		// attached, placement owns every ready job, so a coordinator with no
		// node would queue forever (D1, D40). --role coordinator is the
		// deliberate opposite, a daemon that only places.
		var embedded *embeddedNode
		if *role == roleBoth {
			node, nerr := newEmbeddedNode(embeddedSettings{
				baseDir:              baseDir,
				runsDir:              runsDir,
				slots:                maxJobs,
				devices:              allowedDevices,
				minRAMGB:             minRAM,
				minDiskGB:            minDisk,
				requireSignedCommits: requireSignedCommits,
				allowedSignersPath:   allowedSignersPath,
			})
			if nerr != nil {
				log.Fatalf("embedded node: %v", nerr)
			}
			embedded = node
		}

		sweeper, serr := startNashnet(nashCfg, srv, disp, env, runsDir, pubKeyPath, minDisk, embedded)
		if serr != nil {
			log.Fatalf("start nashnet coordinator: %v", serr)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go sweeper.Run(ctx)
		log.Printf("nashnet coordinator enabled (nodes=%s)", nashCfg.nodesDir)

		if embedded != nil {
			// The handler is built after AttachPool so the node's requests
			// reach the /nashnet/ routes, and it is the same routed handler the
			// TLS listener serves.
			handler := srv.Handler()
			log.Printf("nashnet: embedded node %s (slots=%d, devices=%s, in place)",
				embedded.nodeID(), maxJobs, sortedDeviceList(allowedDevices))
			go func() {
				if err := embedded.run(ctx, handler, pm); err != nil {
					log.Printf("embedded node stopped: %v", err)
				}
			}()
		}
	}

	// Job-preserving restart (cambia-655): SIGTERM (systemd stop/restart) detaches
	// the daemon from its jobs and exits without signalling them, so a redeploy
	// never kills a multi-week training run; the next incarnation reattaches them
	// at Reconcile and a watcher finalizes each one when its process exits. Jobs
	// are stopped only through the API. SIGINT (interactive) still kills the
	// process groups. systemd KillMode=process keeps the unit stop from killing
	// the children the daemon deliberately left behind.
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
		os.Exit(0)
	}()

	log.Printf("runnerd serving HTTPS on %s (runs=%s, max_jobs=%d, queue=%d, devices=%s)", *listen, runsDir, maxJobs, maxQueue, sortedDeviceList(allowedDevices))
	if err := srv.ListenAndServeTLS(*listen, tlsCert, tlsKey); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

// envOr returns the env var value or a fallback.
func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// sortedDeviceList renders an allowed-devices set as a stable, comma-joined
// string for the startup log line.
func sortedDeviceList(devices map[string]bool) string {
	names := make([]string, 0, len(devices))
	for d := range devices {
		names = append(names, d)
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}
