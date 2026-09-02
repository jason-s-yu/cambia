// internal/historian/main_test.go
package historian

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/jason-s-yu/cambia/service/internal/testutil"
)

// redisAddr is the address Redis-dependent tests connect to. It follows the
// same REDIS_ADDR env var (default "localhost:6379") that
// internal/cache.ConnectRedis and cmd/db/historian.go read, so a dev machine
// pointing REDIS_ADDR elsewhere is honored consistently.
var redisAddr string

// redisAvailable reports whether redisAddr is reachable. Redis-dependent
// tests check this flag and skip cleanly on machines without a running dev
// Redis instead of failing the whole package.
var redisAvailable bool

// dbAvailable reports whether the dev Postgres is reachable. The end-to-end
// test needs both halves of the historian's world, so it checks this alongside
// redisAvailable. The probe and the bounded pool behind it come from
// internal/testutil, the same ones internal/game and internal/database use
// (cambia-1830).
var dbAvailable bool

func TestMain(m *testing.M) {
	// The historian binary this package launches reads its Postgres settings
	// from the environment, so loading service/.env here covers the subprocess
	// too: it inherits whatever this load puts in place.
	testutil.LoadServiceEnv()

	redisAddr = os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	redisAvailable = pingTestRedis(redisAddr)
	dbAvailable = testutil.PingPostgres()
	os.Exit(m.Run())
}

// pingTestRedis attempts a short-timeout connection to addr. It never fails
// fatally: an unreachable Redis is an expected condition on dev machines and
// callers use the returned bool to skip Redis-dependent tests.
func pingTestRedis(addr string) bool {
	rdb := redis.NewClient(&redis.Options{Addr: addr})
	defer rdb.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	return rdb.Ping(ctx).Err() == nil
}
