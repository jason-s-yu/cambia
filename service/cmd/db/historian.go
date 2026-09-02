// cmd/db/historian.go is an asynchronous historian service that pops message data from a Redis queue and persists it to a PostgreSQL database.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/redis/go-redis/v9"
)

// GameActionRecord holds minimal info about a single action to be persisted.
type GameActionRecord struct {
	GameID        uuid.UUID              `json:"game_id"`
	ActionIndex   int                    `json:"action_index"`
	ActorUserID   uuid.UUID              `json:"actor_user_id"`
	ActionType    string                 `json:"action_type"`
	ActionPayload map[string]interface{} `json:"action_payload"`
	Timestamp     int64                  `json:"timestamp"` // epoch millis or similar
}

// HistorianService encapsulates the Redis + DB logic for capturing game actions
// and marking games abandoned when a certain inactivity threshold is reached.
//
// The historian is the only writer of game_actions and creates no games row of
// its own. The game server owns that row and writes it at game start
// (database.UpsertInitialGameState); the historian reads it as a foreign key,
// so an action whose games row has not committed yet is retried rather than
// inserted around, and one that never gets a row is dead-lettered rather than
// dropped. Its only writes to games advance an existing in_progress row, to
// 'completed' on the terminal action (insertGameActionTx) and to 'abandoned'
// after the inactivity timeout (markGameAbandoned). See writeBatch for the
// retry and DeadLetterQueueName for where the rest goes.
type HistorianService struct {
	redisClient  *redis.Client
	batchSize    int
	flushDelay   time.Duration
	inactivity   time.Duration // duration until a game is marked "abandoned"
	lastActivity sync.Map      // map[uuid.UUID]time.Time for tracking last activity per game

	// retryAttempts and retryBase drive the per-record retry a failed batch
	// falls back to; deadLetterQueue is the Redis list a record lands on once
	// those attempts are spent.
	retryAttempts   int
	retryBase       time.Duration
	deadLetterQueue string

	batchMu  sync.Mutex
	batch    []GameActionRecord
	ctx      context.Context
	cancelFn context.CancelFunc

	// readerDone closes when readRedisLoop has stopped appending, so the final
	// flush cannot race a record popped just before cancellation. stopped closes
	// once that flush has been written, which is what Stop waits on.
	readerDone chan struct{}
	stopped    chan struct{}
}

// shutdownDrainTimeout bounds the wait for the Redis reader to exit before the
// final flush. A reader wedged in a network call must not hold shutdown open
// past the container's stop grace period.
const shutdownDrainTimeout = 5 * time.Second

// DeadLetterQueueName is the Redis list a game action lands on when it has
// failed every write attempt a flush allows. It is the historian's own dead
// letter box: nothing else pushes to it and nothing pops it automatically, so
// an operator can inspect what did not persist and replay it by pushing the
// entries' "record" field back onto the live action queue
// (cache.DefaultQueueName). HISTORIAN_DEAD_LETTER_QUEUE_NAME overrides it, the
// way HISTORIAN_QUEUE_NAME overrides the live queue.
const DeadLetterQueueName = "cambia_actions_dead_letter"

// Retry schedule for a flush whose write fails. The failure this exists for is
// SQLSTATE 23503 on game_actions.game_id: the game server writes the games row
// from a background goroutine (game.persistInitialGameState) and pushes the
// game's first actions onto the queue in parallel, so an action can reach the
// historian in the window before its games row commits. That miss clears on
// its own, which makes waiting and retrying the right answer and inserting a
// games row of the historian's own the wrong one. Every other failure class is
// retried on the same schedule rather than being classified, since a write
// that succeeds on a later attempt is a write that should not have been
// thrown away whatever its error code was.
//
// The schedule is defaultFlushRetryBaseMs doubling per attempt over
// defaultFlushRetryAttempts passes: 50, 100, 200, 400 ms between five passes,
// 750 ms of waiting in total. HISTORIAN_RETRY_ATTEMPTS and
// HISTORIAN_RETRY_BASE_MS override both.
//
// maxFlushRetryWindow is the hard ceiling on one flush's retrying, whatever
// those two are set to, because the final flush runs during shutdown: the
// drain costs at most shutdownDrainTimeout + maxFlushRetryWindow +
// deadLetterPushTimeout, 12s against the historian container's 15s
// stop_grace_period (deploy/hawking/docker-compose.yml). Raising any of the
// three past that budget makes a SIGTERM drain a kill.
const (
	defaultFlushRetryAttempts = 5
	defaultFlushRetryBaseMs   = 50
	maxFlushRetryWindow       = 5 * time.Second
	deadLetterPushTimeout     = 2 * time.Second
)

// NewHistorianService constructs a HistorianService instance from environment variables or defaults.
func NewHistorianService() *HistorianService {
	batchSize := getEnvInt("HISTORIAN_BATCH_SIZE", 20)
	flushMs := getEnvInt("HISTORIAN_FLUSH_MS", 500)
	inactivitySec := getEnvInt("GAME_INACTIVITY_TIMEOUT_SEC", 600) // default 10 min
	retryAttempts := getEnvInt("HISTORIAN_RETRY_ATTEMPTS", defaultFlushRetryAttempts)
	retryBaseMs := getEnvInt("HISTORIAN_RETRY_BASE_MS", defaultFlushRetryBaseMs)

	// A configured zero or negative would turn the retry off and dead-letter
	// the first transient miss, which is the bug this schedule exists to
	// prevent; one attempt with no wait is the floor.
	if retryAttempts < 1 {
		retryAttempts = 1
	}
	if retryBaseMs < 0 {
		retryBaseMs = 0
	}

	redisAddr := getEnv("REDIS_ADDR", "localhost:6379")
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	ctx, cancel := context.WithCancel(context.Background())
	return &HistorianService{
		redisClient:     rdb,
		batchSize:       batchSize,
		flushDelay:      time.Duration(flushMs) * time.Millisecond,
		inactivity:      time.Duration(inactivitySec) * time.Second,
		retryAttempts:   retryAttempts,
		retryBase:       time.Duration(retryBaseMs) * time.Millisecond,
		deadLetterQueue: getEnv("HISTORIAN_DEAD_LETTER_QUEUE_NAME", DeadLetterQueueName),
		batch:           make([]GameActionRecord, 0, batchSize),
		ctx:             ctx,
		cancelFn:        cancel,
		readerDone:      make(chan struct{}),
		stopped:         make(chan struct{}),
	}
}

// Run starts the two main loops:
//  1. A loop that reads from the Redis queue, accumulates messages in a batch, and flushes them to the DB.
//  2. A periodic check for inactivity to mark games as abandoned.
func (hs *HistorianService) Run() {
	// Connect to the database.
	database.ConnectDB()

	// Start the background loops.
	go hs.readRedisLoop()
	go hs.inactivityLoop()

	log.Println("cambia-historian service started.")
	<-hs.ctx.Done()
	log.Println("cambia-historian shutting down.")

	select {
	case <-hs.readerDone:
	case <-time.After(shutdownDrainTimeout):
		log.Println("cambia-historian: redis reader did not stop in time, flushing anyway.")
	}
	hs.flushBatchToDB()
	close(hs.stopped)
}

// readRedisLoop continuously uses BLPop to retrieve messages from the Redis queue.
func (hs *HistorianService) readRedisLoop() {
	defer close(hs.readerDone)

	ticker := time.NewTicker(hs.flushDelay)
	defer ticker.Stop()

	queueName := getEnv("HISTORIAN_QUEUE_NAME", "cambia_actions")

	for {
		select {
		case <-hs.ctx.Done():
			return

		case <-ticker.C:
			hs.flushBatchToDB()

		default:
			// Use BLPop with a 3-second timeout so that context cancellation is handled.
			res, err := hs.redisClient.BLPop(hs.ctx, 3*time.Second, queueName).Result()
			if err != nil && !errors.Is(err, redis.Nil) {
				log.Printf("[ERROR] BLPop: %v\n", err)
				continue
			}
			if len(res) < 2 {
				// No message popped.
				continue
			}

			// res[0] is the queue name and res[1] the payload.
			payload := res[1]
			var record GameActionRecord
			if err := json.Unmarshal([]byte(payload), &record); err != nil {
				log.Printf("invalid action record: %v\n", err)
				continue
			}

			// Track last activity for the game.
			hs.lastActivity.Store(record.GameID, time.Now())

			hs.appendToBatch(record)
		}
	}
}

// appendToBatch adds a record to the in-memory batch and flushes if the threshold is reached.
// The flush happens after the lock is dropped: batchMu is not reentrant, and it
// must not be held across a database round trip.
func (hs *HistorianService) appendToBatch(record GameActionRecord) {
	hs.batchMu.Lock()
	hs.batch = append(hs.batch, record)
	full := len(hs.batch) >= hs.batchSize
	hs.batchMu.Unlock()

	if full {
		hs.flushBatchToDB()
	}
}

// takeBatch removes and returns the accumulated records, or nil if there are none.
func (hs *HistorianService) takeBatch() []GameActionRecord {
	hs.batchMu.Lock()
	defer hs.batchMu.Unlock()

	if len(hs.batch) == 0 {
		return nil
	}
	batchCopy := make([]GameActionRecord, len(hs.batch))
	copy(batchCopy, hs.batch)
	hs.batch = hs.batch[:0]
	return batchCopy
}

// flushBatchToDB flushes the current batch to the database.
//
// The context is Background rather than the service context: the final flush
// runs after Run's context has been cancelled, and a cancelled context there
// would throw away exactly the records shutdown exists to save.
func (hs *HistorianService) flushBatchToDB() {
	batchCopy := hs.takeBatch()
	if len(batchCopy) == 0 {
		return
	}
	hs.writeBatch(context.Background(), batchCopy)
}

// failedWrite pairs a record the database refused with the refusal, so a
// record that is dead-lettered carries the reason it could not be written
// rather than whichever error the batch reported last.
type failedWrite struct {
	record GameActionRecord
	err    error
}

// writeBatch persists batch, retrying what does not land and dead-lettering
// what still does not.
//
// The first attempt is one transaction for the whole batch, which is the
// common case and one round trip. A batch that fails falls back to a
// transaction per record: a batch is a slice of whatever the queue happened to
// hold, so one record whose game row will never appear must not keep the other
// nineteen out of the table. Records that still fail are retried together on
// the backoff schedule above, and whatever survives every attempt is
// dead-lettered rather than dropped.
func (hs *HistorianService) writeBatch(ctx context.Context, batch []GameActionRecord) {
	err := beginTxFunc(ctx, database.DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, rec := range batch {
			if err := insertGameActionTx(ctx, tx, rec); err != nil {
				return fmt.Errorf("insertGameActionTx game %v action %d: %w", rec.GameID, rec.ActionIndex, err)
			}
		}
		return nil
	})
	if err == nil {
		log.Printf("Flushed %d actions to DB.\n", len(batch))
		return
	}
	log.Printf("[WARN] flushBatchToDB: batch of %d did not commit, retrying record by record: %v", len(batch), err)

	pending := batch
	deadline := time.Now().Add(maxFlushRetryWindow)
	for attempt := 1; ; attempt++ {
		failures := hs.writeRecords(ctx, pending)
		if written := len(pending) - len(failures); written > 0 {
			log.Printf("Flushed %d actions to DB on retry attempt %d.\n", written, attempt)
		}
		if len(failures) == 0 {
			return
		}

		if attempt >= hs.retryAttempts || time.Now().After(deadline) {
			hs.deadLetter(ctx, failures, attempt)
			return
		}

		wait := hs.backoff(attempt)
		if remaining := time.Until(deadline); wait > remaining {
			wait = remaining
		}
		if wait > 0 {
			time.Sleep(wait)
		}

		pending = make([]GameActionRecord, 0, len(failures))
		for _, f := range failures {
			pending = append(pending, f.record)
		}
	}
}

// writeRecords writes each record in its own transaction and returns the ones
// that failed, so a single bad record costs only itself.
func (hs *HistorianService) writeRecords(ctx context.Context, records []GameActionRecord) []failedWrite {
	var failures []failedWrite
	for _, rec := range records {
		rec := rec
		err := beginTxFunc(ctx, database.DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return insertGameActionTx(ctx, tx, rec)
		})
		if err != nil {
			failures = append(failures, failedWrite{record: rec, err: err})
		}
	}
	return failures
}

// backoff returns the wait before the given retry attempt: retryBase doubled
// once per attempt already spent, capped at maxFlushRetryWindow so a large
// configured base cannot produce a single wait longer than the whole budget.
func (hs *HistorianService) backoff(attempt int) time.Duration {
	wait := hs.retryBase
	for i := 1; i < attempt && wait < maxFlushRetryWindow; i++ {
		wait *= 2
	}
	if wait > maxFlushRetryWindow {
		wait = maxFlushRetryWindow
	}
	return wait
}

// deadLetterEntry is the shape pushed onto DeadLetterQueueName. It wraps the
// original record rather than replacing it so an operator can replay one by
// pushing its "record" field back onto the live queue, and carries why it did
// not land so a triage does not have to go hunting in container logs.
type deadLetterEntry struct {
	Record   GameActionRecord `json:"record"`
	Reason   string           `json:"reason"`
	SQLState string           `json:"sql_state,omitempty"`
	// Attempts is the number of per-record retry passes spent on this record,
	// not counting the batch transaction that failed ahead of them.
	Attempts int   `json:"attempts"`
	FailedAt int64 `json:"failed_at"`
}

// deadLetter logs each unwritable record with its game id and pushes it onto
// the dead-letter list. A record is never dropped silently: if the push itself
// fails, the entry is logged in full so the container's log still holds it.
func (hs *HistorianService) deadLetter(ctx context.Context, failures []failedWrite, attempts int) {
	pushCtx, cancel := context.WithTimeout(ctx, deadLetterPushTimeout)
	defer cancel()

	for _, f := range failures {
		sqlState := ""
		var pgErr *pgconn.PgError
		if errors.As(f.err, &pgErr) {
			sqlState = pgErr.Code
		}
		log.Printf("[ERROR] historian: game %v action %d (%s) failed its batch write and %d retry passes (SQLSTATE %q), dead-lettering to %q: %v",
			f.record.GameID, f.record.ActionIndex, f.record.ActionType, attempts, sqlState, hs.deadLetterQueue, f.err)

		data, err := json.Marshal(deadLetterEntry{
			Record:   f.record,
			Reason:   f.err.Error(),
			SQLState: sqlState,
			Attempts: attempts,
			FailedAt: time.Now().UnixMilli(),
		})
		if err != nil {
			log.Printf("[ERROR] historian: game %v action %d could not be marshaled for the dead-letter list: %v; record: %+v",
				f.record.GameID, f.record.ActionIndex, err, f.record)
			continue
		}
		if err := hs.redisClient.RPush(pushCtx, hs.deadLetterQueue, data).Err(); err != nil {
			log.Printf("[ERROR] historian: game %v action %d could not be pushed to %q (%v); the entry follows so it is not lost: %s",
				f.record.GameID, f.record.ActionIndex, hs.deadLetterQueue, err, data)
		}
	}
}

// inactivityLoop periodically checks if any game has been inactive beyond the configured threshold,
// and marks such games as abandoned.
func (hs *HistorianService) inactivityLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-hs.ctx.Done():
			return

		case <-ticker.C:
			now := time.Now()
			hs.lastActivity.Range(func(key, val interface{}) bool {
				gameID, ok1 := key.(uuid.UUID)
				last, ok2 := val.(time.Time)
				if ok1 && ok2 && now.Sub(last) > hs.inactivity {
					hs.markGameAbandoned(gameID)
					hs.lastActivity.Delete(gameID)
				}
				return true
			})
		}
	}
}

// markGameAbandoned marks a game as 'abandoned' in the database if it was still marked as
// 'in_progress'. Like the terminal-action finalize, this advances a row the game server created
// and creates nothing: no matching in_progress row updates nothing.
func (hs *HistorianService) markGameAbandoned(gameID uuid.UUID) {
	ctx := context.Background()
	err := beginTxFunc(ctx, database.DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		q := `
			UPDATE games
			SET status = 'abandoned', end_time = NOW()
			WHERE id = $1 AND status = 'in_progress'
		`
		_, e := tx.Exec(ctx, q, gameID)
		return e
	})
	if err != nil {
		log.Printf("failed to mark game %v abandoned: %v", gameID, err)
	} else {
		log.Printf("Marked game %v as 'abandoned' due to inactivity.", gameID)
	}
}

// insertGameActionTx inserts a single action record into the game_actions
// table. If the action indicates game end, it closes the game out.
//
// It does not create a games row. The game server owns that row and writes it
// at game start (database.UpsertInitialGameState, cambia-458 and cambia-1240),
// supplying the lobby_id and the initial state the historian has no way to
// know; game_actions.game_id is a foreign key onto it and nothing more.
//
// This function used to open with an upsert of its own, a fallback from before
// the server wrote the row. Migration 5 made games.lobby_id NOT NULL with no
// default (5_add_lobby_persistence.sql, cambia-450) and the upsert never
// supplied it, so every flush failed SQLSTATE 23502 and no game action reached
// the table from that migration until cambia-1881. It failed even for a games
// row that already existed, because Postgres checks NOT NULL against the
// proposed tuple before it resolves ON CONFLICT; repairing the upsert would
// have needed a lobby id the historian does not have, so ownership moved to
// the one writer that does. The window where the row has not committed yet is
// handled by writeBatch's retry, not by writing around it.
func insertGameActionTx(ctx context.Context, tx pgx.Tx, rec GameActionRecord) error {
	actionInsertQ := `
		INSERT INTO game_actions (
			game_id, action_index, actor_user_id, action_type, action_payload
		) VALUES ($1, $2, $3, $4, $5)
	`
	jsonPayload, err := json.Marshal(rec.ActionPayload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, actionInsertQ,
		rec.GameID, rec.ActionIndex, rec.ActorUserID, rec.ActionType, jsonPayload,
	)
	if err != nil {
		return err
	}

	if rec.ActionType == "action_end_game" {
		// A conditional advance of a row the game server already created, not
		// a write that can bring one into existence: no matching in_progress
		// row updates nothing and the action still lands. It is what sets
		// games.end_time (database.GetUserGameHistory reads it).
		finalizeQ := `
			UPDATE games
			SET status = 'completed', end_time = NOW()
			WHERE id = $1 AND status = 'in_progress'
		`
		_, err = tx.Exec(ctx, finalizeQ, rec.GameID)
		if err != nil {
			return err
		}
	}
	return nil
}

// beginTxFunc is a helper that starts a transaction using the provided pool,
// calls the function f with the transaction, and commits or rollbacks as needed.
func beginTxFunc(ctx context.Context, pool *pgxpool.Pool, txOptions pgx.TxOptions, f func(tx pgx.Tx) error) error {
	tx, err := pool.BeginTx(ctx, txOptions)
	if err != nil {
		return err
	}
	// If f returns an error, rollback and return the error.
	if err := f(tx); err != nil {
		if rbErr := tx.Rollback(ctx); rbErr != nil {
			return fmt.Errorf("tx rollback error: %v; original error: %w", rbErr, err)
		}
		return err
	}
	return tx.Commit(ctx)
}

// Stop gracefully stops the historian service and blocks until Run has written
// the final batch, so records already popped off the Redis queue are not lost
// on SIGTERM.
func (hs *HistorianService) Stop() {
	hs.cancelFn()
	<-hs.stopped
}

// main is the entrypoint.
func main() {
	hs := NewHistorianService()
	go hs.Run()

	// Block until an OS signal is received.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	hs.Stop()
	log.Println("Historian shutdown complete.")
}

// getEnv retrieves an environment variable's value or returns a default.
func getEnv(key, defVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defVal
}

// getEnvInt retrieves an integer value from an environment variable or returns a default value.
func getEnvInt(key string, defVal int) int {
	v := os.Getenv(key)
	if v == "" {
		return defVal
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return defVal
	}
	return i
}
