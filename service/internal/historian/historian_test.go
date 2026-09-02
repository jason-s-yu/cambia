// internal/historian/historian_test.go
package historian

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/testutil"
)

// For demonstration, we'll do a minimal test that writes one action to Redis
// and ensures we can parse it. A deeper test would require a running Redis + DB instance.
func TestBasicHistorianFlow(t *testing.T) {
	if !redisAvailable {
		t.Skip("skipping: no Redis reachable via REDIS_ADDR (default localhost:6379); set REDIS_ADDR to point at a running dev Redis to run this test")
	}

	// (Optional) start a real or mock Redis
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr, // resolved in TestMain from REDIS_ADDR, default localhost:6379
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

// endToEndActions is the number of ordinary actions pushed before the terminal
// one. Two batches' worth at the batch size the test configures, so the run
// exercises a size-triggered flush and the ticker flush that carries the
// remainder, rather than one of the two alone.
const endToEndActions = 4

// TestHistorianEndToEnd drives the shipped historian the way production runs
// it: build cmd/db, point the binary at this environment's Redis and Postgres,
// push a game's action stream onto its queue, and read back what landed. It
// covers what no unit test can - the binary's env wiring, the Redis pop loop,
// the batch flush, the rows it writes, the finalize branch that closes a game
// on its terminal action, and a clean SIGTERM shutdown. Replaces a placeholder
// that only described this and always skipped, which is how it stayed green
// while nothing ran it (cambia-1246).
func TestHistorianEndToEnd(t *testing.T) {
	if !redisAvailable {
		t.Skip("skipping: no Redis reachable via REDIS_ADDR (default localhost:6379); set REDIS_ADDR to point at a running dev Redis to run this test")
	}
	if !dbAvailable {
		t.Skip(testutil.SkipMessage)
	}

	ctx := t.Context()

	pool, err := testutil.NewBoundedPool(ctx)
	if err != nil {
		t.Fatalf("connect to the test database: %v", err)
	}
	// Registered before every cleanup that needs the pool, so LIFO order closes
	// it last. A plain defer would close it while those still had rows to
	// delete, since a test's defers run ahead of its cleanups.
	t.Cleanup(pool.Close)

	// The schema has to exist before the fixture rows are inserted. The binary
	// migrates on boot as well; Migrate is idempotent and holds an advisory
	// lock, so the two passes cannot race each other.
	if err := database.Migrate(ctx, pool); err != nil {
		t.Fatalf("migrate the test database: %v", err)
	}

	gameID := uuid.New()
	actorID := uuid.New()
	if _, err := pool.Exec(ctx,
		`INSERT INTO users (id, username, is_ephemeral) VALUES ($1, $2, TRUE)`,
		actorID, "historian-e2e-"+actorID.String()); err != nil {
		t.Fatalf("insert the acting user: %v", err)
	}

	// The lobby and the in-progress games row the game server writes at
	// game start (database.UpsertInitialGameState), which is the state the
	// historian meets in production: it appends to a games row somebody else
	// created rather than creating one.
	var lobbyID uuid.UUID
	if err := pool.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type) VALUES ($1, 'private') RETURNING id`,
		actorID).Scan(&lobbyID); err != nil {
		t.Fatalf("insert the lobby: %v", err)
	}
	if _, err := pool.Exec(ctx,
		`INSERT INTO games (id, lobby_id, status, start_time) VALUES ($1, $2, 'in_progress', NOW())`,
		gameID, lobbyID); err != nil {
		t.Fatalf("insert the games row: %v", err)
	}

	// The lobby cascades to games, which cascades to game_actions.
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(cleanupCtx, `DELETE FROM lobbies WHERE id = $1`, lobbyID); err != nil {
			t.Logf("cleanup: delete lobbies row %s: %v", lobbyID, err)
		}
		if _, err := pool.Exec(cleanupCtx, `DELETE FROM users WHERE id = $1`, actorID); err != nil {
			t.Logf("cleanup: delete users row %s: %v", actorID, err)
		}
	})

	// A queue of this run's own, so a concurrent historian or the leftover
	// record TestBasicHistorianFlow pushes onto the default queue cannot be
	// read as this test's data.
	queue := "cambia_actions_e2e_" + uuid.NewString()
	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
	t.Cleanup(func() {
		defer rdb.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := rdb.Del(cleanupCtx, queue).Err(); err != nil {
			t.Logf("cleanup: delete redis queue %s: %v", queue, err)
		}
	})

	// Queued before the binary starts: BLPop takes whatever is waiting, so
	// there is no readiness handshake to get wrong, and FIFO order puts the
	// terminal action last where the finalize branch expects it.
	for i := 0; i < endToEndActions; i++ {
		pushAction(t, ctx, rdb, queue, gameID, actorID, i, "action_draw_stockpile")
	}
	pushAction(t, ctx, rdb, queue, gameID, actorID, endToEndActions, "action_end_game")

	logPath := startHistorian(t, queue)

	// Poll rather than sleep on a fixed delay: the flush is driven by a batch
	// size and a ticker, so the write lands at a time this test does not
	// control.
	var actionCount int
	var status string
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		if err := pool.QueryRow(ctx,
			`SELECT count(*) FROM game_actions WHERE game_id = $1`, gameID).Scan(&actionCount); err != nil {
			t.Fatalf("count persisted actions: %v", err)
		}
		if err := pool.QueryRow(ctx,
			`SELECT status FROM games WHERE id = $1`, gameID).Scan(&status); err != nil {
			t.Fatalf("read the games row: %v", err)
		}
		if actionCount == endToEndActions+1 && status == "completed" {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}

	if actionCount != endToEndActions+1 {
		t.Fatalf("historian persisted %d of %d actions\nhistorian log:\n%s",
			actionCount, endToEndActions+1, readLog(t, logPath))
	}
	if status != "completed" {
		t.Fatalf("terminal action left the games row at status %q, want \"completed\"\nhistorian log:\n%s",
			status, readLog(t, logPath))
	}

	// The rows themselves, not just their count: the historian is the only
	// writer of this table, so a wrong actor, a dropped index, or a payload
	// that did not survive the JSON round trip is a silent data-loss bug.
	rows, err := pool.Query(ctx,
		`SELECT action_index, actor_user_id, action_type, action_payload
		   FROM game_actions WHERE game_id = $1 ORDER BY action_index`, gameID)
	if err != nil {
		t.Fatalf("read persisted actions: %v", err)
	}
	defer rows.Close()

	seen := 0
	for rows.Next() {
		var index int
		var actor uuid.UUID
		var actionType string
		var payload map[string]any
		if err := rows.Scan(&index, &actor, &actionType, &payload); err != nil {
			t.Fatalf("scan persisted action: %v", err)
		}
		if index != seen {
			t.Fatalf("action %d persisted out of order, at index %d", seen, index)
		}
		if actor != actorID {
			t.Fatalf("action %d persisted actor %s, want %s", index, actor, actorID)
		}
		want := "action_draw_stockpile"
		if index == endToEndActions {
			want = "action_end_game"
		}
		if actionType != want {
			t.Fatalf("action %d persisted type %q, want %q", index, actionType, want)
		}
		if got := payload["step"]; got != float64(index) {
			t.Fatalf("action %d persisted payload %v, want step %d", index, payload, index)
		}
		seen++
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate persisted actions: %v", err)
	}
	if seen != endToEndActions+1 {
		t.Fatalf("read back %d actions, want %d", seen, endToEndActions+1)
	}
}

// pushAction queues one action record in the wire shape cmd/db unmarshals.
func pushAction(t *testing.T, ctx context.Context, rdb *redis.Client, queue string,
	gameID, actorID uuid.UUID, index int, actionType string) {
	t.Helper()

	data, err := json.Marshal(map[string]any{
		"game_id":        gameID.String(),
		"action_index":   index,
		"actor_user_id":  actorID.String(),
		"action_type":    actionType,
		"action_payload": map[string]any{"step": index},
		"timestamp":      time.Now().UnixMilli(),
	})
	if err != nil {
		t.Fatalf("marshal action %d: %v", index, err)
	}
	if err := rdb.RPush(ctx, queue, data).Err(); err != nil {
		t.Fatalf("push action %d onto %s: %v", index, queue, err)
	}
}

// startHistorian builds and runs cmd/db against the calling test's environment,
// returning the path its output is written to. The process is stopped with the
// SIGTERM production sends it, and the test fails if that shutdown does not
// come back clean. Output goes to a file rather than a bytes.Buffer so reading
// it from a failure message cannot race os/exec's copier goroutine.
func startHistorian(t *testing.T, queue string) string {
	t.Helper()

	dir := t.TempDir()
	bin := filepath.Join(dir, "cambia-historian")
	build := exec.Command("go", "build", "-o", bin, "../../cmd/db")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build the historian binary: %v\n%s", err, out)
	}

	logPath := filepath.Join(dir, "historian.log")
	logFile, err := os.Create(logPath)
	if err != nil {
		t.Fatalf("create the historian log file: %v", err)
	}

	cmd := exec.Command(bin)
	// The PG_* vars come in from the test process; os/exec keeps the last
	// occurrence of a duplicated key, so these override anything inherited.
	cmd.Env = append(os.Environ(),
		"REDIS_ADDR="+redisAddr,
		"RUN_MIGRATIONS=true",
		"HISTORIAN_QUEUE_NAME="+queue,
		"HISTORIAN_BATCH_SIZE=2",
		"HISTORIAN_FLUSH_MS=100",
	)
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		logFile.Close()
		t.Fatalf("start the historian binary: %v", err)
	}

	t.Cleanup(func() {
		defer logFile.Close()
		if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
			t.Errorf("signal the historian: %v", err)
			return
		}
		if err := cmd.Wait(); err != nil {
			t.Errorf("historian did not shut down cleanly on SIGTERM: %v\nhistorian log:\n%s",
				err, readLog(t, logPath))
		}
	})

	return logPath
}

// readLog returns the historian's captured output for a failure message.
func readLog(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("(could not read %s: %v)", path, err)
	}
	return string(data)
}

// TODO:: test inactivity logic
