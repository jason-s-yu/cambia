// internal/historian/main_test.go
package historian

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
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

func TestMain(m *testing.M) {
	redisAddr = os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	redisAvailable = pingTestRedis(redisAddr)
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
