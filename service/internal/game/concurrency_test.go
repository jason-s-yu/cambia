// internal/game/concurrency_test.go
package game

import (
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// noopEmitter is a stateless Emitter for the race test. It holds no mutable state, so it is
// safe to call concurrently; the point of the test is to race CambiaGame's own state, and
// every emit already happens under the game mutex.
type noopEmitter struct{}

func (noopEmitter) Emit(string, any)              {}
func (noopEmitter) EmitTo(uuid.UUID, string, any) {}

// TestConcurrentActionsAndTimers stresses the cambia-465 serialization: a running game is
// driven by an aggressively short turn timer (which fires handleTimeoutEngine from a timer
// goroutine) while several goroutines hammer the public action entry points and a state read.
// Run with -race: any unsynchronized access between a timer fire and an action is reported.
// Without the CambiaGame mutex this fails immediately under the race detector.
func TestConcurrentActionsAndTimers(t *testing.T) {
	deadline := time.Now().Add(400 * time.Millisecond)
	rounds := 0
	for time.Now().Before(deadline) {
		runConcurrentRound(t)
		rounds++
	}
	if rounds == 0 {
		t.Fatal("no concurrent rounds executed")
	}
	t.Logf("completed %d concurrent game rounds under the race detector", rounds)
}

// runConcurrentRound builds a fresh 2-player game with a sub-timer-tick turn duration, then
// overlaps action goroutines with the autonomous turn timer until the game ends or a hard cap
// elapses. The game is torn down deterministically (EndGame stops any in-flight timer).
func runConcurrentRound(t *testing.T) {
	t.Helper()

	g := NewCambiaGame()
	g.Emitter = noopEmitter{}
	// ForfeitOnDisconnect off keeps scoring simple; TurnTimerSec>0 arms a turn timer that we
	// then shrink below its 1s floor via TurnDuration before StartGame reads it.
	g.HouseRules = HouseRules{TurnTimerSec: 1, PenaltyDrawCount: 2, ForfeitOnDisconnect: false}

	ids := make([]uuid.UUID, 2)
	for i := range ids {
		ids[i] = uuid.New()
		g.AddPlayer(&models.Player{
			ID:        ids[i],
			Connected: true,
			User:      &models.User{ID: ids[i], Username: "P"},
		})
	}

	g.BeginPreGame() // deals, arms the 10s pre-game timer
	// Shrink the turn timer well below a scheduler tick so it re-fires continuously against the
	// action goroutines. Set before StartGame, which arms the first turn timer from TurnDuration.
	g.TurnDuration = 2 * time.Millisecond
	g.StartGame() // stops the pre-game timer, begins the turn cycle

	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Action goroutines only touch the locked public entry points and pass payloads that need no
	// peek at internal state, so the test itself introduces no unsynchronized reads of g. Actions
	// may be logically rejected (wrong turn, stale card id); the mutation pressure comes from the
	// draws/cambia that succeed plus the timer's autonomous draw+discard progression.
	worker := func(seed int) {
		defer wg.Done()
		pid := ids[seed%2]
		bogus := uuid.New().String()
		for {
			select {
			case <-stop:
				return
			default:
			}
			g.HandlePlayerAction(pid, models.GameAction{ActionType: "action_draw_stockpile"})
			g.HandlePlayerAction(pid, models.GameAction{
				ActionType: "action_discard",
				Payload:    map[string]interface{}{"id": bogus},
			})
			g.HandlePlayerAction(pid, models.GameAction{
				ActionType: "action_snap",
				Payload:    map[string]interface{}{"id": bogus},
			})
			g.ProcessSpecialAction(pid, "skip", nil, nil)
			// Concurrent locked read of the full snapshot.
			_ = g.GetCurrentObfuscatedGameState(pid)
			time.Sleep(150 * time.Microsecond)
		}
	}

	for w := 0; w < 4; w++ {
		wg.Add(1)
		go worker(w)
	}

	// One goroutine occasionally calls Cambia to drive the game to a natural end, exercising the
	// EndGame path concurrently with the timer.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			g.HandlePlayerAction(ids[0], models.GameAction{ActionType: "action_cambia"})
			time.Sleep(2 * time.Millisecond)
		}
	}()

	// Controller: end the round when the game is over or a hard cap elapses, then join.
	roundCap := time.Now().Add(120 * time.Millisecond)
	for time.Now().Before(roundCap) {
		if g.GetCurrentObfuscatedGameState(ids[0]).GameOver {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(stop)
	wg.Wait()

	// Deterministically stop any still-armed turn timer (no-op if the game already ended).
	g.EndGame()
}
