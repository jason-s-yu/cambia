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
type HistorianService struct {
	redisClient  *redis.Client
	batchSize    int
	flushDelay   time.Duration
	inactivity   time.Duration // duration until a game is marked "abandoned"
	lastActivity sync.Map      // map[uuid.UUID]time.Time for tracking last activity per game

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

// NewHistorianService constructs a HistorianService instance from environment variables or defaults.
func NewHistorianService() *HistorianService {
	batchSize := getEnvInt("HISTORIAN_BATCH_SIZE", 20)
	flushMs := getEnvInt("HISTORIAN_FLUSH_MS", 500)
	inactivitySec := getEnvInt("GAME_INACTIVITY_TIMEOUT_SEC", 600) // default 10 min

	redisAddr := getEnv("REDIS_ADDR", "localhost:6379")
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	ctx, cancel := context.WithCancel(context.Background())
	return &HistorianService{
		redisClient: rdb,
		batchSize:   batchSize,
		flushDelay:  time.Duration(flushMs) * time.Millisecond,
		inactivity:  time.Duration(inactivitySec) * time.Second,
		batch:       make([]GameActionRecord, 0, batchSize),
		ctx:         ctx,
		cancelFn:    cancel,
		readerDone:  make(chan struct{}),
		stopped:     make(chan struct{}),
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

// flushBatchToDB flushes the current batch to the database in a single transaction.
func (hs *HistorianService) flushBatchToDB() {
	batchCopy := hs.takeBatch()
	if len(batchCopy) == 0 {
		return
	}

	ctx := context.Background()
	err := beginTxFunc(ctx, database.DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Insert each action record within the transaction.
		for _, rec := range batchCopy {
			if err := insertGameActionTx(ctx, tx, rec); err != nil {
				return fmt.Errorf("insertGameActionTx: %w", err)
			}
		}
		return nil
	})
	if err != nil {
		log.Printf("[ERROR] flushBatchToDB: %v\n", err)
	} else {
		log.Printf("Flushed %d actions to DB.\n", len(batchCopy))
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

// markGameAbandoned marks a game as 'abandoned' in the database if it was still marked as 'in_progress'.
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

// insertGameActionTx inserts a single action record into the game_actions table and
// upserts the game row if necessary. If the action indicates game end, finalizes the game.
func insertGameActionTx(ctx context.Context, tx pgx.Tx, rec GameActionRecord) error {
	upsertGameQ := `
		INSERT INTO games (id, status, start_time)
		VALUES ($1, 'in_progress', NOW())
		ON CONFLICT (id)
		DO UPDATE SET status = 'in_progress'
	`
	_, err := tx.Exec(ctx, upsertGameQ, rec.GameID)
	if err != nil {
		return err
	}

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
