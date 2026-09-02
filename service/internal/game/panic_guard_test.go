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
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	g.OnGameEnd = func(_ uuid.UUID, _ uuid.UUID, _ map[uuid.UUID]int, _ map[uuid.UUID]string, _ map[uuid.UUID]int, _ uuid.UUID, _ []FinalHand) {
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
