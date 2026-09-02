// cmd/db/flush_retry_test.go
//
// The flush path against a real Postgres and Redis: that the historian writes
// no games row of its own, that an action arriving ahead of its games row is
// retried until the row appears, and that one that never appears is
// dead-lettered without taking the rest of its batch with it (cambia-1881).
//
// These need the dev stack (service/docker-compose.yml) and skip cleanly
// without it, the same way every other DB-backed package in this module does.
package main

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"

	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/testutil"
)

// dbAvailable and testRedisAddr are resolved once in TestMain and read by the
// tests that need the dev stack.
var (
	dbAvailable    bool
	redisAvailable bool
	testRedisAddr  string
)

func TestMain(m *testing.M) {
	// The historian reads its Postgres and Redis settings from the
	// environment, and `go test` runs this binary in cmd/db rather than the
	// module root, so service/.env is only found by walking up to go.mod
	// (cambia-1830).
	testutil.LoadServiceEnv()

	testRedisAddr = os.Getenv("REDIS_ADDR")
	if testRedisAddr == "" {
		testRedisAddr = "localhost:6379"
	}
	dbAvailable = testutil.PingPostgres()
	redisAvailable = pingTestRedis(testRedisAddr)

	os.Exit(m.Run())
}

// pingTestRedis reports whether addr answers. An unreachable Redis is an
// expected condition on a machine with no dev stack up, so it never fails
// fatally; callers skip on the result.
func pingTestRedis(addr string) bool {
	rdb := redis.NewClient(&redis.Options{Addr: addr})
	defer rdb.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	return rdb.Ping(ctx).Err() == nil
}

// flushFixture is one test's slice of the dev stack: a bounded pool wired into
// database.DB the way the binary wires its own, a historian pointed at a
// dead-letter list of this run's own, and the user and lobby a games row needs.
type flushFixture struct {
	pool    *pgxpool.Pool
	rdb     *redis.Client
	hs      *HistorianService
	userID  uuid.UUID
	lobbyID uuid.UUID
}

// newFlushFixture skips the calling test when the dev stack is not up.
func newFlushFixture(t *testing.T, retryAttempts int, retryBase time.Duration) *flushFixture {
	t.Helper()

	if !dbAvailable {
		t.Skip(testutil.SkipMessage)
	}
	if !redisAvailable {
		t.Skip("skipping: no Redis reachable via REDIS_ADDR (default localhost:6379); set REDIS_ADDR to point at a running dev Redis to run this test")
	}

	ctx := t.Context()

	pool, err := testutil.NewBoundedPool(ctx)
	if err != nil {
		t.Fatalf("connect to the test database: %v", err)
	}
	// Registered ahead of the cleanups that delete rows through it, so LIFO
	// order closes the pool last.
	t.Cleanup(pool.Close)

	if err := database.Migrate(ctx, pool); err != nil {
		t.Fatalf("migrate the test database: %v", err)
	}

	prevDB := database.DB
	database.DB = pool
	t.Cleanup(func() { database.DB = prevDB })

	userID := uuid.New()
	if _, err := pool.Exec(ctx,
		`INSERT INTO users (id, username, is_ephemeral) VALUES ($1, $2, TRUE)`,
		userID, "historian-flush-"+userID.String()); err != nil {
		t.Fatalf("insert the acting user: %v", err)
	}

	var lobbyID uuid.UUID
	if err := pool.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type) VALUES ($1, 'private') RETURNING id`,
		userID).Scan(&lobbyID); err != nil {
		t.Fatalf("insert the lobby: %v", err)
	}

	// The lobby cascades to games, which cascades to game_actions.
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(cleanupCtx, `DELETE FROM lobbies WHERE id = $1`, lobbyID); err != nil {
			t.Logf("cleanup: delete lobbies row %s: %v", lobbyID, err)
		}
		if _, err := pool.Exec(cleanupCtx, `DELETE FROM users WHERE id = $1`, userID); err != nil {
			t.Logf("cleanup: delete users row %s: %v", userID, err)
		}
	})

	// A dead-letter list of this run's own: the shipped default is a single
	// shared list, so a concurrent checkout's failures would otherwise read as
	// this test's.
	deadLetter := DeadLetterQueueName + "_test_" + uuid.NewString()
	t.Setenv("HISTORIAN_DEAD_LETTER_QUEUE_NAME", deadLetter)
	t.Setenv("REDIS_ADDR", testRedisAddr)

	hs := NewHistorianService()
	hs.retryAttempts = retryAttempts
	hs.retryBase = retryBase

	rdb := redis.NewClient(&redis.Options{Addr: testRedisAddr})
	t.Cleanup(func() {
		defer rdb.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := rdb.Del(cleanupCtx, deadLetter).Err(); err != nil {
			t.Logf("cleanup: delete redis list %s: %v", deadLetter, err)
		}
	})

	return &flushFixture{pool: pool, rdb: rdb, hs: hs, userID: userID, lobbyID: lobbyID}
}

// insertGamesRow writes the row the game server writes at game start
// (database.UpsertInitialGameState), which is the state the historian meets in
// production.
func (f *flushFixture) insertGamesRow(t *testing.T, gameID uuid.UUID) {
	t.Helper()

	if _, err := f.pool.Exec(context.Background(),
		`INSERT INTO games (id, lobby_id, status, start_time) VALUES ($1, $2, 'in_progress', NOW())`,
		gameID, f.lobbyID); err != nil {
		t.Fatalf("insert the games row: %v", err)
	}
}

func (f *flushFixture) record(gameID uuid.UUID, index int, actionType string) GameActionRecord {
	return GameActionRecord{
		GameID:        gameID,
		ActionIndex:   index,
		ActorUserID:   f.userID,
		ActionType:    actionType,
		ActionPayload: map[string]interface{}{"step": index},
		Timestamp:     time.Now().UnixMilli(),
	}
}

// persistedIndexes returns the action_index values that landed for gameID.
func (f *flushFixture) persistedIndexes(t *testing.T, gameID uuid.UUID) []int {
	t.Helper()

	rows, err := f.pool.Query(context.Background(),
		`SELECT action_index FROM game_actions WHERE game_id = $1 ORDER BY action_index`, gameID)
	if err != nil {
		t.Fatalf("read persisted actions: %v", err)
	}
	defer rows.Close()

	var out []int
	for rows.Next() {
		var index int
		if err := rows.Scan(&index); err != nil {
			t.Fatalf("scan persisted action: %v", err)
		}
		out = append(out, index)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate persisted actions: %v", err)
	}
	return out
}

// deadLettered returns the entries on this fixture's dead-letter list.
func (f *flushFixture) deadLettered(t *testing.T) []deadLetterEntry {
	t.Helper()

	raw, err := f.rdb.LRange(context.Background(), f.hs.deadLetterQueue, 0, -1).Result()
	if err != nil {
		t.Fatalf("read the dead-letter list: %v", err)
	}

	entries := make([]deadLetterEntry, 0, len(raw))
	for i, item := range raw {
		var entry deadLetterEntry
		if err := json.Unmarshal([]byte(item), &entry); err != nil {
			t.Fatalf("dead-letter entry %d is not the documented shape: %v\n%s", i, err, item)
		}
		entries = append(entries, entry)
	}
	return entries
}

// The historian writes game_actions and nothing else. Before cambia-1881 every
// flush opened with an upsert of the games row, which both failed its NOT NULL
// check on lobby_id and, had it succeeded, would have reset a finished game to
// 'in_progress'. Asserting on a 'completed' row catches either half coming
// back: a re-created row loses its lobby_id, and a resurrected one loses its
// status.
func TestFlushWritesNoGamesRow(t *testing.T) {
	f := newFlushFixture(t, defaultFlushRetryAttempts, time.Millisecond)

	gameID := uuid.New()
	f.insertGamesRow(t, gameID)
	if _, err := f.pool.Exec(context.Background(),
		`UPDATE games SET status = 'completed', end_time = NOW() WHERE id = $1`, gameID); err != nil {
		t.Fatalf("close the games row: %v", err)
	}

	f.hs.appendToBatch(f.record(gameID, 0, "action_draw_stockpile"))
	f.hs.flushBatchToDB()

	if got := f.persistedIndexes(t, gameID); len(got) != 1 || got[0] != 0 {
		t.Fatalf("expected the action to land, got indexes %v", got)
	}
	if entries := f.deadLettered(t); len(entries) != 0 {
		t.Fatalf("expected nothing dead-lettered, got %d entries", len(entries))
	}

	var status string
	var lobbyID *uuid.UUID
	if err := f.pool.QueryRow(context.Background(),
		`SELECT status, lobby_id FROM games WHERE id = $1`, gameID).Scan(&status, &lobbyID); err != nil {
		t.Fatalf("read the games row back: %v", err)
	}
	if status != "completed" {
		t.Fatalf("the flush moved the games row to status %q; it must not write games rows", status)
	}
	if lobbyID == nil || *lobbyID != f.lobbyID {
		t.Fatalf("the flush changed the games row's lobby_id to %v, want %v", lobbyID, f.lobbyID)
	}
}

// An action can reach the historian before its games row commits: the server
// writes that row from a background goroutine and queues the game's first
// actions in parallel. The foreign-key miss that produces is transient, so the
// flush waits it out rather than dropping the action.
func TestFlushRetriesUntilTheGamesRowAppears(t *testing.T) {
	// Six attempts at a 100ms base is a 3.1s window against a row that
	// appears after 200ms, so the test does not turn on scheduling luck.
	f := newFlushFixture(t, 6, 100*time.Millisecond)

	gameID := uuid.New()
	f.hs.appendToBatch(f.record(gameID, 0, "action_draw_stockpile"))

	// The insert runs off the test goroutine, so its error comes back on a
	// channel rather than through t.Fatalf, which is only valid on the
	// goroutine running the test.
	inserted := make(chan error, 1)
	timer := time.AfterFunc(200*time.Millisecond, func() {
		_, err := f.pool.Exec(context.Background(),
			`INSERT INTO games (id, lobby_id, status, start_time) VALUES ($1, $2, 'in_progress', NOW())`,
			gameID, f.lobbyID)
		inserted <- err
	})
	defer timer.Stop()

	f.hs.flushBatchToDB()

	select {
	case err := <-inserted:
		if err != nil {
			t.Fatalf("insert the games row mid-flush: %v", err)
		}
	default:
		t.Fatal("the flush returned before the games row was written; the retry window did not hold")
	}

	if got := f.persistedIndexes(t, gameID); len(got) != 1 || got[0] != 0 {
		t.Fatalf("the retry did not land the action once its games row appeared, got indexes %v", got)
	}
	if entries := f.deadLettered(t); len(entries) != 0 {
		t.Fatalf("expected nothing dead-lettered, got %d entries: %+v", len(entries), entries)
	}
}

// A games row that never appears is the case the retry cannot fix. The action
// is dead-lettered with its reason rather than dropped, and the historian
// still writes no games row to make it fit.
func TestFlushDeadLettersWhenTheGamesRowNeverAppears(t *testing.T) {
	f := newFlushFixture(t, 2, time.Millisecond)

	gameID := uuid.New()
	rec := f.record(gameID, 7, "action_draw_stockpile")
	f.hs.appendToBatch(rec)
	f.hs.flushBatchToDB()

	if got := f.persistedIndexes(t, gameID); len(got) != 0 {
		t.Fatalf("expected no action to land without a games row, got indexes %v", got)
	}

	var games int
	if err := f.pool.QueryRow(context.Background(),
		`SELECT count(*) FROM games WHERE id = $1`, gameID).Scan(&games); err != nil {
		t.Fatalf("count games rows: %v", err)
	}
	if games != 0 {
		t.Fatalf("the flush created %d games rows; it must not write games rows", games)
	}

	entries := f.deadLettered(t)
	if len(entries) != 1 {
		t.Fatalf("expected 1 dead-lettered entry, got %d: %+v", len(entries), entries)
	}
	got := entries[0]
	if got.Record.GameID != gameID || got.Record.ActionIndex != rec.ActionIndex {
		t.Fatalf("dead-lettered the wrong record: %+v", got.Record)
	}
	if got.Record.ActionType != rec.ActionType {
		t.Fatalf("dead-lettered record lost its action type: %q", got.Record.ActionType)
	}
	// 23503 is foreign_key_violation, the class the retry exists for; naming
	// it in the entry is what makes a dead-letter list triageable.
	if got.SQLState != "23503" {
		t.Fatalf("dead-lettered entry reported SQLSTATE %q, want \"23503\" (reason: %s)", got.SQLState, got.Reason)
	}
	if got.Attempts != 2 {
		t.Fatalf("dead-lettered entry reported %d attempts, want 2", got.Attempts)
	}
	if got.Reason == "" {
		t.Fatal("dead-lettered entry carries no reason")
	}
}

// A batch is whatever the queue happened to hold, so it can mix games. One
// record that will never land must cost only itself: before cambia-1881 the
// whole batch shared one transaction and one bad record rolled back the rest.
func TestFlushLandsTheRestOfTheBatchWhenOneRecordDeadLetters(t *testing.T) {
	f := newFlushFixture(t, 2, time.Millisecond)

	goodID := uuid.New()
	f.insertGamesRow(t, goodID)
	orphanID := uuid.New()

	f.hs.appendToBatch(f.record(goodID, 0, "action_draw_stockpile"))
	f.hs.appendToBatch(f.record(orphanID, 0, "action_draw_stockpile"))
	f.hs.appendToBatch(f.record(goodID, 1, "action_discard"))
	f.hs.flushBatchToDB()

	got := f.persistedIndexes(t, goodID)
	if len(got) != 2 || got[0] != 0 || got[1] != 1 {
		t.Fatalf("one unwritable record held back the rest of its batch, landed indexes %v", got)
	}

	entries := f.deadLettered(t)
	if len(entries) != 1 {
		t.Fatalf("expected only the orphan dead-lettered, got %d entries: %+v", len(entries), entries)
	}
	if entries[0].Record.GameID != orphanID {
		t.Fatalf("dead-lettered game %v, want the orphan %v", entries[0].Record.GameID, orphanID)
	}
}

// A terminal action is an action like any other. The historian used to close
// the game out on one, setting games.status and games.end_time; that write went
// with the rest of the games writes, and the server's RecordGameAndResults is
// what marks a game completed. Both spellings are covered: "action_end_game",
// which the deleted branch keyed on, and "game_end", which is what the server
// actually emits.
func TestFlushDoesNotCloseOutTheGame(t *testing.T) {
	f := newFlushFixture(t, defaultFlushRetryAttempts, time.Millisecond)

	for _, actionType := range []string{"action_end_game", "game_end"} {
		gameID := uuid.New()
		f.insertGamesRow(t, gameID)

		f.hs.appendToBatch(f.record(gameID, 0, "action_draw_stockpile"))
		f.hs.appendToBatch(f.record(gameID, 1, actionType))
		f.hs.flushBatchToDB()

		if got := f.persistedIndexes(t, gameID); len(got) != 2 {
			t.Fatalf("%s: expected both actions to land, got indexes %v", actionType, got)
		}

		var status string
		var endTime *time.Time
		if err := f.pool.QueryRow(context.Background(),
			`SELECT status, end_time FROM games WHERE id = $1`, gameID).Scan(&status, &endTime); err != nil {
			t.Fatalf("%s: read the games row: %v", actionType, err)
		}
		if status != "in_progress" {
			t.Fatalf("%s: the historian moved the games row to status %q; games is the server's to write", actionType, status)
		}
		if endTime != nil {
			t.Fatalf("%s: the historian set games.end_time to %v; games is the server's to write", actionType, *endTime)
		}
	}
}

// uuid.Nil is the game server's "no actor" value for a game's own events, and
// game_actions.actor_user_id is nullable for exactly those rows. Stored as the
// literal nil uuid it has no matching users row, so every game event failed
// SQLSTATE 23503 on game_actions_actor_user_id_fkey and dead-lettered; mapping
// it to NULL is what lets a game event persist.
func TestFlushStoresANilActorAsNull(t *testing.T) {
	f := newFlushFixture(t, 2, time.Millisecond)

	gameID := uuid.New()
	f.insertGamesRow(t, gameID)

	// The four action types game.logAction passes uuid.Nil for, plus one
	// ordinary player action to show the actor is still recorded when there
	// is one.
	events := []string{"game_pregame_start", "game_start", "game_initial_state_saved", "game_end"}
	for i, actionType := range events {
		rec := f.record(gameID, i, actionType)
		rec.ActorUserID = uuid.Nil
		f.hs.appendToBatch(rec)
	}
	f.hs.appendToBatch(f.record(gameID, len(events), "action_draw_stockpile"))
	f.hs.flushBatchToDB()

	if entries := f.deadLettered(t); len(entries) != 0 {
		t.Fatalf("a nil actor dead-lettered %d records: %+v", len(entries), entries)
	}

	rows, err := f.pool.Query(context.Background(),
		`SELECT action_index, action_type, actor_user_id FROM game_actions WHERE game_id = $1 ORDER BY action_index`, gameID)
	if err != nil {
		t.Fatalf("read persisted actions: %v", err)
	}
	defer rows.Close()

	seen := 0
	for rows.Next() {
		var index int
		var actionType string
		var actor *uuid.UUID
		if err := rows.Scan(&index, &actionType, &actor); err != nil {
			t.Fatalf("scan persisted action: %v", err)
		}
		if index != seen {
			t.Fatalf("action %d persisted out of order, at index %d", seen, index)
		}
		if index < len(events) {
			if actionType != events[index] {
				t.Fatalf("action %d persisted type %q, want %q", index, actionType, events[index])
			}
			if actor != nil {
				t.Fatalf("game event %q persisted actor %v, want NULL", actionType, *actor)
			}
		} else if actor == nil || *actor != f.userID {
			t.Fatalf("player action %d persisted actor %v, want %v", index, actor, f.userID)
		}
		seen++
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate persisted actions: %v", err)
	}
	if seen != len(events)+1 {
		t.Fatalf("read back %d actions, want %d", seen, len(events)+1)
	}
}

// The dead-letter list has one documented name, and the environment override
// exists so a test can isolate itself without that name drifting.
func TestDeadLetterQueueDefaultsToTheDocumentedName(t *testing.T) {
	t.Setenv("HISTORIAN_DEAD_LETTER_QUEUE_NAME", "")

	if got := NewHistorianService().deadLetterQueue; got != DeadLetterQueueName {
		t.Fatalf("default dead-letter list is %q, want %q", got, DeadLetterQueueName)
	}
}

// A retry count below one would dead-letter the first transient miss, which is
// the failure the schedule exists to prevent.
func TestRetryAttemptsFloorAtOne(t *testing.T) {
	t.Setenv("HISTORIAN_RETRY_ATTEMPTS", "0")

	if got := NewHistorianService().retryAttempts; got != 1 {
		t.Fatalf("configured 0 retry attempts became %d, want the floor of 1", got)
	}
}

// The backoff doubles per attempt and is capped, so a large configured base
// cannot produce one wait longer than the whole retry budget.
func TestBackoffDoublesAndCaps(t *testing.T) {
	hs := &HistorianService{retryBase: 50 * time.Millisecond}

	for attempt, want := range map[int]time.Duration{
		1: 50 * time.Millisecond,
		2: 100 * time.Millisecond,
		3: 200 * time.Millisecond,
		4: 400 * time.Millisecond,
	} {
		if got := hs.backoff(attempt); got != want {
			t.Fatalf("backoff(%d) = %v, want %v", attempt, got, want)
		}
	}

	big := &HistorianService{retryBase: time.Minute}
	if got := big.backoff(1); got != maxFlushRetryWindow {
		t.Fatalf("a one-minute base produced a %v wait, want the %v cap", got, maxFlushRetryWindow)
	}
}
