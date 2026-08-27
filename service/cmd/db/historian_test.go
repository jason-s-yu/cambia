// cmd/db/historian_test.go
//
// Batch accumulation and the shutdown drain. The batch tests do not touch
// Postgres: flushBatchToDB takes the records out from under batchMu before it
// opens a transaction, so takeBatch is the part that can be exercised alone.
package main

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

func newTestHistorian(batchSize int) *HistorianService {
	hs := NewHistorianService()
	hs.batchSize = batchSize
	return hs
}

// withUnreachableDB points database.DB at a pool that will never connect.
// pgxpool dials lazily, so this stands in for the real binary's connected pool
// (Run calls ConnectDB before any flush can happen) while keeping the flush
// path exercised: the transaction fails and is logged instead of writing rows.
func withUnreachableDB(t *testing.T) {
	t.Helper()

	pool, err := pgxpool.New(t.Context(), "postgres://cambia:cambia@127.0.0.1:1/cambia-test")
	if err != nil {
		t.Fatalf("failed to build the stand-in pool: %v", err)
	}
	prev := database.DB
	database.DB = pool
	t.Cleanup(func() {
		database.DB = prev
		pool.Close()
	})
}

func testRecord(i int) GameActionRecord {
	return GameActionRecord{
		GameID:      uuid.New(),
		ActionIndex: i,
		ActionType:  "action_draw_stockpile",
		Timestamp:   time.Now().UnixMilli(),
	}
}

// appendToBatch used to call flushBatchToDB while holding batchMu, which
// self-deadlocks on the non-reentrant mutex as soon as a batch fills. The
// timeout is what makes this a regression test: a deadlock hangs rather than
// failing an assertion.
func TestAppendToBatchDoesNotDeadlockOnFullBatch(t *testing.T) {
	hs := newTestHistorian(2)
	withUnreachableDB(t)

	done := make(chan struct{})
	go func() {
		defer close(done)
		hs.appendToBatch(testRecord(0))
		hs.appendToBatch(testRecord(1)) // fills the batch, triggering the flush path
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("appendToBatch deadlocked when the batch filled")
	}
}

func TestTakeBatchDrainsAndResets(t *testing.T) {
	hs := newTestHistorian(100)

	for i := 0; i < 3; i++ {
		hs.appendToBatch(testRecord(i))
	}

	got := hs.takeBatch()
	if len(got) != 3 {
		t.Fatalf("expected 3 records, got %d", len(got))
	}
	for i, rec := range got {
		if rec.ActionIndex != i {
			t.Fatalf("record %d out of order: ActionIndex=%d", i, rec.ActionIndex)
		}
	}

	if again := hs.takeBatch(); again != nil {
		t.Fatalf("expected an empty second take, got %d records", len(again))
	}
}

func TestTakeBatchEmptyReturnsNil(t *testing.T) {
	if got := newTestHistorian(10).takeBatch(); got != nil {
		t.Fatalf("expected nil for an empty batch, got %v", got)
	}
}

// Stop must block until the shutdown drain has run, so a SIGTERM does not cut
// the process off before the final batch is written.
func TestStopWaitsForShutdownDrain(t *testing.T) {
	hs := newTestHistorian(100)

	// Stand in for Run's shutdown half without connecting to Postgres or Redis.
	close(hs.readerDone)
	go func() {
		<-hs.ctx.Done()
		close(hs.stopped)
	}()

	done := make(chan struct{})
	go func() {
		defer close(done)
		hs.Stop()
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Stop did not return after the drain completed")
	}
}
