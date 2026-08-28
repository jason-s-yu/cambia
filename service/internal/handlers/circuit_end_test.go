// internal/handlers/circuit_end_test.go
package handlers

import (
	"context"
	"testing"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// TestAttachOnGameEnd_CircuitUsesRawScoresAndRealCaller verifies attachOnGameEnd's circuit
// branch (cambia-1008, cambia-1009) feeds CircuitState.RecordRound the raw (pre-WinBonus) hand
// scores and the real Cambia caller's engine seat, never the display-adjusted scores or the -1
// "no caller" sentinel this used to hardcode. It invokes g.OnGameEnd directly with the two
// argument sets endGame() now produces (adjusted vs. raw + caller), matching how this package's
// other OnGameEnd tests (game_flow_test.go) simulate the engine's own invocation.
func TestAttachOnGameEnd_CircuitUsesRawScoresAndRealCaller(t *testing.T) {
	gs := NewGameServer()

	hostID := uuid.New()
	p2ID := uuid.New()

	lob := lobby.NewLobbyWithDefaults(hostID)
	lob.JoinUser(hostID)
	lob.JoinUser(p2ID)
	lob.Circuit.Enabled = true
	lob.Circuit.Mode = "quick"
	// Deliberately large and distinct so any leak into circuit scoring is unmistakable.
	lob.Circuit.Rules.WinBonus = -100
	gs.LobbyStore.AddLobby(lob)

	h := hub.NewHub(lob)
	gs.HubStore.CreateHub(h)

	g := gs.CreateGameInstance(context.Background(), lob.ID, hostID, lob.GameMode, lob.Type, false, lob.HouseRules, lob.Circuit, []uuid.UUID{hostID, p2ID}, nil, h)
	if g == nil {
		t.Fatal("failed to create circuit game instance")
	}

	circuitState, playerMap := gs.CircuitStore.Get(lob.ID)
	if circuitState == nil || playerMap == nil {
		t.Fatal("circuit state was not created")
	}
	hostSeat, ok := playerMap[hostID]
	if !ok {
		t.Fatal("host not in circuit player map")
	}
	p2Seat, ok := playerMap[p2ID]
	if !ok {
		t.Fatal("player 2 not in circuit player map")
	}

	// Fabricate what endGame() hands OnGameEnd for a round where the host wins with WinBonus
	// baked into the display scores, and the host is also the Cambia caller tied with player 2
	// on the raw hand score.
	rawScores := map[uuid.UUID]int{hostID: 8, p2ID: 8}
	adjustedScores := map[uuid.UUID]int{hostID: 8 + lob.Circuit.Rules.WinBonus, p2ID: 8}

	g.OnGameEnd(lob.ID, hostID, adjustedScores, map[uuid.UUID]string{}, rawScores, hostID)

	if len(circuitState.Rounds) != 1 {
		t.Fatalf("expected 1 circuit round recorded, got %d", len(circuitState.Rounds))
	}
	round := circuitState.Rounds[0]

	// RecordRound must have seen the raw score (8), never the WinBonus-adjusted one (-92).
	if round.PlayerScores[hostSeat] != 8 {
		t.Errorf("circuit round host score: want raw 8, got %d (WinBonus leaked into circuit scoring)", round.PlayerScores[hostSeat])
	}
	if round.PlayerScores[p2Seat] != 8 {
		t.Errorf("circuit round player2 score: want 8, got %d", round.PlayerScores[p2Seat])
	}

	// The host is the real Cambia caller, so the tie-break must place them first and pay the
	// H2H 1st-place subsidy (-3), not the untied fallback the -1 sentinel used to produce.
	if round.CambiaCallerID != hostSeat {
		t.Errorf("circuit round CambiaCallerID: want host seat %d, got %d", hostSeat, round.CambiaCallerID)
	}
	if len(round.Placements) == 0 || round.Placements[0] != hostSeat {
		t.Errorf("circuit round Placements[0]: want host seat %d (caller wins tie) first, got %v", hostSeat, round.Placements)
	}
	if round.Subsidies[hostSeat] != -3 {
		t.Errorf("host subsidy: want -3 (H2H 1st place), got %d", round.Subsidies[hostSeat])
	}

	// The cumulative total must reflect raw score + subsidy only, excluding WinBonus entirely.
	for _, p := range circuitState.Players {
		if p.PlayerID != hostSeat {
			continue
		}
		want := 8 + round.Subsidies[hostSeat] // 8 + -3 = 5, not 8 + -100 + -3
		if p.CumulativeScore != want {
			t.Errorf("host cumulative score: want %d (WinBonus excluded), got %d", want, p.CumulativeScore)
		}
	}
}
