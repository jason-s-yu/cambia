// internal/game/session_summary.go
package game

import "github.com/google/uuid"

// SessionSummary is a read-only view of a live game, for callers outside the hub goroutine
// (the active-session REST endpoint) that need routing facts without touching game state.
type SessionSummary struct {
	GameID uuid.UUID // Identity of the summarised game.
	// Seated reports whether the queried user holds a seat in this game, as opposed to
	// merely being a member of the lobby that started it.
	Seated bool
	// GameOver mirrors the lifecycle flag: a finished game is not resumable even while the
	// lobby still points at it (the lobby is cleared asynchronously by OnGameEnd).
	GameOver    bool
	PlayerCount int
}

// SessionSummaryFor snapshots the game's lifecycle state and whether userID is seated in it.
//
// It acquires the game's serialization lock, so the caller must hold neither the GameStore
// lock nor a Lobby lock: endGame runs under this lock and reaches for both through the
// OnGameEnd callback, so acquiring them in the opposite order would close a deadlock cycle.
func (g *CambiaGame) SessionSummaryFor(userID uuid.UUID) SessionSummary {
	g.mu.Lock()
	defer g.mu.Unlock()

	s := SessionSummary{
		GameID:      g.ID,
		GameOver:    g.GameOver,
		PlayerCount: len(g.Players),
	}
	for _, p := range g.Players {
		if p != nil && p.ID == userID {
			s.Seated = true
			break
		}
	}
	return s
}
