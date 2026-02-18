// internal/historian/historian_test.go
package historian

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// For demonstration, we'll do a minimal test that writes one action to Redis
// and ensures we can parse it. A deeper test would require a running Redis + DB instance.
func TestBasicHistorianFlow(t *testing.T) {
	// (Optional) start a real or mock Redis
	rdb := redis.NewClient(&redis.Options{
		Addr: "localhost:6379", // must have a real local redis for full integration
	})
	defer rdb.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// push a fake action
	action := map[string]interface{}{
		"game_id":        uuid.New().String(),
		"action_index":   1,
		"actor_user_id":  uuid.New().String(),
		"action_type":    "action_draw_stockpile",
		"action_payload": map[string]interface{}{"extra": "test"},
		"timestamp":      time.Now().UnixMilli(),
	}
	data, _ := json.Marshal(action)
	if err := rdb.RPush(ctx, "cambia_actions", data).Err(); err != nil {
		t.Fatalf("failed to rpush: %v", err)
	}

	// We won't actually spin up the entire historian service in this test. If we did,
	// we'd check the DB for inserted rows. For now, we just confirm the push succeeded.
	// In a real environment, you'd launch historian in a goroutine and let it process.
	t.Log("Pushed a sample action to Redis.")
}

// You can also do a more complete test if your environment includes
// Docker-based Redis + Postgres and you run everything end-to-end. See README.
func TestHistorianEndToEnd(t *testing.T) {
	// This is just a placeholder
	// Typically: start cambia-historian, push actions, wait, check DB
	t.Skip("not implemented end-to-end test here")
}

// TODO:: test inactivity logic
