// internal/cache/redis.go
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// Rdb is the global Redis client. Connect it once at application startup.
var Rdb *redis.Client

// DefaultQueueName is the Redis list (queue) name for game action logs.
var DefaultQueueName = "cambia_actions"

// GameActionRecord holds the minimal info needed by the historian microservice.
type GameActionRecord struct {
	GameID        uuid.UUID              `json:"game_id"`
	ActionIndex   int                    `json:"action_index"`
	ActorUserID   uuid.UUID              `json:"actor_user_id"`
	ActionType    string                 `json:"action_type"`
	ActionPayload map[string]interface{} `json:"action_payload"`
	Timestamp     int64                  `json:"timestamp"`
}

// ConnectRedis initializes the global Redis client with environment variables:
//   - REDIS_ADDR (default "localhost:6379")
//   - REDIS_DB (optional, default 0)
func ConnectRedis() error {
	addr := getEnv("REDIS_ADDR", "localhost:6379")
	dbIdx := getEnvInt("REDIS_DB", 0)

	Rdb = redis.NewClient(&redis.Options{
		Addr: addr,
		DB:   dbIdx,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := Rdb.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("failed to connect to Redis at %s: %w", addr, err)
	}
	return nil
}

// PublishGameAction serializes the given record to JSON, then pushes it to the Redis queue.
// This does not block the calling logic (other than a quick network send).
func PublishGameAction(ctx context.Context, record GameActionRecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal GameActionRecord: %w", err)
	}

	queueName := getEnv("HISTORIAN_QUEUE_NAME", DefaultQueueName)
	if err := Rdb.RPush(ctx, queueName, data).Err(); err != nil {
		return fmt.Errorf("failed to RPush to Redis list '%s': %w", queueName, err)
	}
	return nil
}

// getEnv is a helper to read an environment variable or return a default value.
func getEnv(key, def string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return def
}

// getEnvInt is a helper to parse an environment variable as integer, else a default value.
func getEnvInt(key string, def int) int {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return v
}
