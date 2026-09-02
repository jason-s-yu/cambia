// internal/game/panic_guard_test.go
// A panic inside a game-owned timer or goroutine ends that game, not the process (cambia-1243).
//
// The fault is injected through the Emitter, which is the real shape of the risk: the hub's Emit
// runs on whichever goroutine the game fires an event from, so a timer callback is the one place a
// panic in event delivery used to unwind past the runtime and take every other table with it. Both
// tests here would fail by killing the test binary if the boundary were missing, so the package
// completing at all is half the assertion.
package game

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// panicOnceEmitter forwards to the emitter it wraps, except for the first broadcast after arm(),
// which panics. Panicking once rather than always is what lets the abort path run a real endGame
// afterwards, so the test can assert the game actually ended rather than just that nothing crashed.
type panicOnceEmitter struct {
	inner  Emitter
	armed  atomic.Bool
	fired  atomic.Bool
	events atomic.Int32
}

func (p *panicOnceEmitter) arm() { p.armed.Store(true) }

func (p *panicOnceEmitter) Emit(eventType string, payload any) {
	p.events.Add(1)
	if p.armed.CompareAndSwap(true, false) {
		p.fired.Store(true)
		panic("emitter blew up delivering " + eventType)
	}
	p.inner.Emit(eventType, payload)
}

func (p *panicOnceEmitter) EmitTo(userID uuid.UUID, eventType string, payload any) {
	p.inner.EmitTo(userID, eventType, payload)
}

// endSignal wires OnGameEnd to a buffered channel. The callback runs with g.mu held, so it must
// not touch the game.
func endSignal(g *CambiaGame) chan struct{} {
	ended := make(chan struct{}, 1)
	g.OnGameEnd = func(_ uuid.UUID, _ uuid.UUID, _ map[uuid.UUID]int, _ map[uuid.UUID]string, _ map[uuid.UUID]int, _ uuid.UUID, _ []FinalHand, _ EndReason) {
		select {
		case ended <- struct{}{}:
		default:
		}
	}
	return ended
}

// TestPanicInsideTheTurnTimerEndsTheGameNotTheProcess drives the production turn timer into a
// panicking event delivery. The callback is the real one scheduleNextTurnTimerEngine arms, so what
// is under test is the boundary that callback now runs under, not a stand-in for it.
func TestPanicInsideTheTurnTimerEndsTheGameNotTheProcess(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(15, 2))
	ended := endSignal(g)

	pe := &panicOnceEmitter{inner: mb}
	g.mu.Lock()
	g.Emitter = pe
	g.TurnDuration = 100 * time.Millisecond
	g.scheduleNextTurnTimerEngine()
	g.mu.Unlock()
	// Armed only after the scheduling call, whose own deadline broadcast would otherwise take the
	// panic on this goroutine instead of the timer's.
	pe.arm()

	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("a panic in the turn timer must have ended the game")
	}

	require.True(t, pe.fired.Load(), "precondition: the injected panic must have fired")
	g.mu.Lock()
	defer g.mu.Unlock()
	assert.True(t, g.GameOver, "the game the panic happened in must be over")
	assert.Nil(t, g.turnTimer, "and must have no clock left armed")
}

// TestPanicInsideAGameGoroutineEndsTheGame covers the detached-goroutine half of the boundary: the
// persistence and action-log publishers run with no lock held and nothing above them on the stack.
func TestPanicInsideAGameGoroutineEndsTheGame(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, testHouseRules(0, 2))
	ended := endSignal(g)

	go g.runGuarded("a test goroutine", func() { panic("detached goroutine blew up") })

	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("a panic in a game-owned goroutine must have ended the game")
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.True(t, g.GameOver, "the game the panic happened in must be over")
}

// TestGuardedCallbackPassesTheNonPanickingBodyThrough is the negative: the wrapper must be
// invisible to a body that returns normally, since every production timer now goes through it.
func TestGuardedCallbackPassesTheNonPanickingBodyThrough(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	ran := false
	g.guarded("a well-behaved body", func() { ran = true })()

	assert.True(t, ran, "the guard must run the body it wraps")
	g.mu.Lock()
	defer g.mu.Unlock()
	assert.False(t, g.GameOver, "and must not end a game that never panicked")
}

// TestAPanicEndedGameNamesTheInternalErrorOnItsResultsFrame pins what the table is told when the
// guard aborts a game (cambia-1831). The abort ends the game through the same endGame every
// ordinary ending runs through, so before this the players were handed a plain results frame
// carrying scores read off whatever hands the panic left mid-move, indistinguishable from a game
// somebody won. The reason names it instead, and it is the same value that withholds the rating.
func TestAPanicEndedGameNamesTheInternalErrorOnItsResultsFrame(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	g.Rated = true
	ended := endSignal(g)

	go g.runGuarded("a test goroutine", func() { panic("detached goroutine blew up") })

	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("a panic in a game-owned goroutine must have ended the game")
	}

	ev := mb.findEventByType(EventGameEnd)
	require.NotNil(t, ev, "an aborted game must still tell the table that its game is over")
	assert.Equal(t, string(EndReasonInternalError), ev.Payload["reason"],
		"game_end must name the internal error: the scores beside it are not a result anyone played to")

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, EndReasonInternalError, g.endReason, "the abort must record why it ended the game")
	assert.False(t, g.ratePerGame(), "a rated game the guard aborted must not feed the rating system")
}

// TestAnOrdinaryEndingNamesNoReason is the negative half: a game that reaches one of its rulebook
// endings keeps the frame it has always had, with no reason field for a client to read, and a
// rated one still rates.
func TestAnOrdinaryEndingNamesNoReason(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	g.Rated = true

	g.EndGame()

	ev := mb.findEventByType(EventGameEnd)
	require.NotNil(t, ev, "the game must report its ending")
	_, named := ev.Payload["reason"]
	assert.False(t, named, "a game that ended on its own terms says nothing about why")

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, EndReasonNormal, g.endReason)
	assert.True(t, g.ratePerGame(), "and a rated one still rates")
}

// TestAPanicEndedGameRecordsItsResultsButRatesNobody drives a real rated 2-player game to the
// point where a game-owned goroutine panics, then reads the record path back out of the database
// (cambia-1831). The split is the point: the game_results rows are still written, so what the
// abort found survives for anyone diagnosing it, but no ratings row exists and neither player's
// rating moved. Before this, the abort ran the ordinary rating update on scores read off hands the
// panic had left mid-move.
//
// The counterpart that must keep passing is TestEndGameRecordsResultsAndRating: this narrows the
// per-game rating path, it does not remove it.
func TestAPanicEndedGameRecordsItsResultsButRatesNobody(t *testing.T) {
	setupGameDBTest(t)

	userA := createGameDBTestUser(t, "panic-end-a-"+uuid.NewString())
	userB := createGameDBTestUser(t, "panic-end-b-"+uuid.NewString())

	g := NewCambiaGame()
	g.Emitter = newMockBroadcaster()
	g.LobbyID = uuid.New()
	g.HostUserID = userA.ID
	g.LobbyType = "private"
	g.Rated = true
	g.HouseRules = *testHouseRules(0, 2)
	g.TurnDuration = 0
	g.PersistWG = &sync.WaitGroup{}
	// OnGameEnd fires inside endGame, after persistFinalGameState has Add'd both of its write
	// goroutines, so receiving it is the happens-before edge awaitPersistence needs (cambia-942
	// F3).
	ended := endSignal(g)

	playerA := &models.Player{ID: userA.ID, Connected: true, User: &models.User{ID: userA.ID}}
	playerB := &models.Player{ID: userB.ID, Connected: true, User: &models.User{ID: userB.ID}}
	g.AddPlayer(playerA)
	g.AddPlayer(playerB)

	g.BeginPreGame()
	g.StartGame()
	// The games row has to exist before the abort reaches RecordGameAndResults' completion UPDATE.
	waitForGameStatus(t, g.ID, "in_progress", 2*time.Second)

	go g.runGuarded("a test goroutine", func() { panic("detached goroutine blew up mid-game") })

	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("a panic in a game-owned goroutine must have ended the game")
	}
	awaitPersistence(t, g.PersistWG, 5*time.Second)

	ctx := context.Background()

	var resultRows int
	require.NoError(t, database.DB.QueryRow(ctx,
		`SELECT count(*) FROM game_results WHERE game_id = $1`, g.ID).Scan(&resultRows))
	require.Equal(t, 2, resultRows, "an aborted game is still recorded: what the abort found is the evidence")

	var ratingRows int
	require.NoError(t, database.DB.QueryRow(ctx,
		`SELECT count(*) FROM ratings WHERE game_id = $1`, g.ID).Scan(&ratingRows))
	require.Zero(t, ratingRows, "a game the panic guard ended must write no ratings row")

	afterA, err := database.GetUserByID(ctx, userA.ID)
	require.NoError(t, err)
	afterB, err := database.GetUserByID(ctx, userB.ID)
	require.NoError(t, err)
	require.Equal(t, 1500, afterA.Elo1v1, "no rating moves on a game nobody played to a result")
	require.Equal(t, 1500, afterB.Elo1v1, "no rating moves on a game nobody played to a result")
	require.Equal(t, 350.0, afterA.Phi1v1, "and no rating deviation moves either")
	require.Equal(t, 350.0, afterB.Phi1v1, "and no rating deviation moves either")
}
