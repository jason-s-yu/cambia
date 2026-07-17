// Command runnerd is the cambia serving-harness runner daemon (design 2). It
// serves the HTTPS + Bearer-JWT control plane, supervises training/eval jobs
// through the extracted procmgr process layer, and drives the in-memory FIFO
// queue and job state machine. It never serves plaintext and never auto-launches
// at startup: it reconciles inherited state and reports, and an operator resumes
// explicitly (design 2.3/6).
package main

import (
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

func main() {
	listen := flag.String("listen", envOr("RUNNERD_LISTEN", "127.0.0.1:8090"),
		"control-plane listen address (dev default 127.0.0.1:8090; prod binds the runner's LAN address)")
	flag.Parse()

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

	env := ingest.New(ingest.Config{
		BaseDir:              baseDir,
		RunsDir:              runsDir,
		CoresCap:             runtime.NumCPU() - 2,
		RequireSignedCommits: requireSignedCommits,
		AllowedSignersPath:   allowedSignersPath,
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
	})
	if err != nil {
		log.Fatalf("build server: %v", err)
	}

	// Best-effort abrupt stop on signal (design 6): SIGKILL job process groups so
	// they do not outlive the daemon; recovery is reconcile-on-next-boot. systemd
	// KillMode=mixed + TimeoutStopSec=35 sits above the 30s per-job grace.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		log.Print("shutdown signal received; killing job process groups")
		pm.KillAll()
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
