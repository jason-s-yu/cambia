package main

import (
	"crypto/ed25519"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/harness"
	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// nashnetConfig is the coordinator's own pool configuration, read from
// RUNNERD_NASHNET_* alone. Every value has a documented default, so enabling
// the pool is one variable: the nodes directory the operator drops grants into.
// The node role's own configuration is a nashnet: YAML section on the node
// (design D46) and is not read here.
type nashnetConfig struct {
	nodesDir         string
	quarantineDir    string
	snapshotDir      string
	snapshotCache    int
	grantLifetime    time.Duration
	revokeGrace      time.Duration
	leaseTTL         time.Duration
	maxLeaseSeconds  time.Duration
	maxLeasesPerNode int
	maxClaimWaiters  int
	nodeTTL          time.Duration
	sessionGrace     time.Duration
	unplaceableGrace time.Duration
	maxRunDBBytes    int64
	maxFileBytes     int64
	maxLeaseBytes    int64
	originHost       string
}

// loadNashnetConfig reads the coordinator pool settings. It returns enabled
// false when RUNNERD_NASHNET_NODES_DIR is unset, which is the zero-node daemon:
// no /nashnet/ route is registered and the v1.0 surface is untouched (D39, D40).
func loadNashnetConfig(baseDir string) (nashnetConfig, bool) {
	dir := os.Getenv("RUNNERD_NASHNET_NODES_DIR")
	if dir == "" {
		return nashnetConfig{}, false
	}
	c := nashnetConfig{
		nodesDir:         dir,
		quarantineDir:    envOr("RUNNERD_NASHNET_QUARANTINE_DIR", filepath.Join(baseDir, "nashnet", "quarantine")),
		snapshotDir:      os.Getenv("RUNNERD_NASHNET_SNAPSHOT_DIR"),
		snapshotCache:    envInt("RUNNERD_NASHNET_SNAPSHOT_CACHE", 20),
		grantLifetime:    envDuration("RUNNERD_NASHNET_GRANT_LIFETIME", authtoken.DefaultGrantLifetime),
		revokeGrace:      envDuration("RUNNERD_NASHNET_REVOKE_GRACE", 0),
		leaseTTL:         envDuration("RUNNERD_NASHNET_LEASE_TTL", nashnet.DefaultLeaseTTLSeconds*time.Second),
		maxLeaseSeconds:  envDuration("RUNNERD_NASHNET_MAX_LEASE_SECONDS", nashnet.DefaultMaxLeaseSeconds*time.Second),
		maxLeasesPerNode: envInt("RUNNERD_NASHNET_MAX_LEASES_PER_NODE", nashnet.DefaultMaxLeasesPerNode),
		maxClaimWaiters:  envInt("RUNNERD_NASHNET_MAX_CLAIM_WAITERS", harness.DefaultMaxClaimWaiters),
		nodeTTL:          envDuration("RUNNERD_NASHNET_NODE_TTL", nashnet.DefaultNodeTTLSeconds*time.Second),
		sessionGrace:     envDuration("RUNNERD_NASHNET_SESSION_GRACE", nashnet.DefaultSessionGraceSeconds*time.Second),
		unplaceableGrace: envDuration("RUNNERD_NASHNET_UNPLACEABLE_GRACE", harness.DefaultUnplaceableGrace),
		maxRunDBBytes:    int64(envInt("RUNNERD_NASHNET_MAX_RUNDB_BYTES", 0)),
		maxFileBytes:     int64(envInt("RUNNERD_NASHNET_MAX_FILE_BYTES", 0)),
		maxLeaseBytes:    int64(envInt("RUNNERD_NASHNET_MAX_LEASE_BYTES", 0)),
		originHost:       os.Getenv("RUNNERD_NASHNET_ORIGIN_HOST"),
	}
	return c, true
}

// startNashnet builds the coordinator pool and attaches it to the server: the
// grant store that authenticates node tokens, the lease store restored from
// runs/*/lease.json, the node registry seeded from the epochs those leases
// carry, the quarantine store, and the expiry sweeper. It returns the sweeper
// so the caller runs it beside the listener.
func startNashnet(cfg nashnetConfig, srv *harness.Server, disp *harness.Dispatcher,
	mgr *ingest.Manager, runsDir, pubKeyPath string, minDiskGB float64) (*nashnet.Sweeper, error) {

	opKey, err := os.ReadFile(pubKeyPath)
	if err != nil {
		return nil, fmt.Errorf("read operator public key: %w", err)
	}
	if len(opKey) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("operator public key has %d bytes, want %d", len(opKey), ed25519.PublicKeySize)
	}
	grants, err := authtoken.NewGrantStore(authtoken.GrantStoreConfig{
		Dir:              cfg.nodesDir,
		OperatorKey:      ed25519.PublicKey(opKey),
		MaxGrantLifetime: cfg.grantLifetime,
	})
	if err != nil {
		return nil, fmt.Errorf("load enrollment grants: %w", err)
	}

	policy := nashnet.DefaultPolicy()
	policy.LeaseTTLSeconds = int(cfg.leaseTTL / time.Second)
	policy.MaxLeaseSeconds = int(cfg.maxLeaseSeconds / time.Second)
	if cfg.maxFileBytes > 0 {
		policy.MaxFileBytes = cfg.maxFileBytes
	}
	if cfg.maxLeaseBytes > 0 {
		policy.MaxLeaseBytes = cfg.maxLeaseBytes
	}

	leases, err := nashnet.NewLeaseStore(nashnet.StoreConfig{
		RunsDir:     runsDir,
		Policy:      policy,
		RevokeGrace: cfg.revokeGrace,
	})
	if err != nil {
		return nil, fmt.Errorf("build lease store: %w", err)
	}
	restored, skipped, err := leases.Restore()
	if err != nil {
		return nil, fmt.Errorf("restore leases: %w", err)
	}
	log.Printf("nashnet: %d leases restored, %d skipped", restored, skipped)

	registry := nashnet.NewNodeRegistry(nashnet.RegistryConfig{
		NodeTTL:      cfg.nodeTTL,
		SessionGrace: cfg.sessionGrace,
	})
	// A restored lease is fenced on the node epoch it carries, so the registry
	// starts above it and a node that never noticed the restart is not fenced
	// out (D34).
	for _, l := range leases.LiveForNode("") {
		registry.SeedEpoch(l.NodeID, l.NodeEpoch)
	}

	limits := quarantine.DefaultLimits()
	limits.MinFreeDiskGB = minDiskGB
	if cfg.maxFileBytes > 0 {
		limits.MaxFileBytes = cfg.maxFileBytes
	}
	if cfg.maxLeaseBytes > 0 {
		limits.MaxLeaseBytes = cfg.maxLeaseBytes
	}
	quarCfg := quarantine.Config{
		QuarantineDir: cfg.quarantineDir,
		RunsDir:       runsDir,
		Limits:        limits,
	}
	if cfg.maxRunDBBytes > 0 {
		quarCfg.RunDB.MaxBytes = cfg.maxRunDBBytes
	}
	quar, err := quarantine.New(quarCfg)
	if err != nil {
		return nil, fmt.Errorf("build quarantine store: %w", err)
	}
	log.Printf("nashnet: quarantine at %s, materialize_mode=%s (%s)",
		cfg.quarantineDir, quar.MaterializeMode(), quar.MaterializeReason())

	pool, err := harness.NewPool(harness.PoolConfig{
		Dispatcher:       disp,
		Grants:           grants,
		Leases:           leases,
		Registry:         registry,
		Quarantine:       quar,
		Bundles:          mgr,
		RunsDir:          runsDir,
		NodesDir:         cfg.nodesDir,
		OriginHost:       cfg.originHost,
		Policy:           policy,
		Ceilings:         harness.Ceilings{MaxClaimWaiters: cfg.maxClaimWaiters},
		MaxLeasesPerNode: cfg.maxLeasesPerNode,
		UnplaceableGrace: cfg.unplaceableGrace,
		NodeTTL:          cfg.nodeTTL,
		SessionGrace:     cfg.sessionGrace,
	})
	if err != nil {
		return nil, err
	}
	srv.AttachPool(pool)
	return pool.Sweeper(), nil
}

// envInt reads a positive integer env var, falling back on anything malformed
// or non-positive rather than disabling a cap.
func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return fallback
}

// envDuration reads a Go duration env var (30s, 5m, 72h), falling back on
// anything malformed or non-positive.
func envDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return fallback
}
