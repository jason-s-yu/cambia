// internal/game/circuit_endgame_test.go
package game

import (
	"testing"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

// TestEndGameSeparatesRawScoresFromCircuitAdjustments drives a real 2-player game to a Cambia
// call and completion, then verifies endGame() hands OnGameEnd the raw (pre-WinBonus) hand
// scores and the real Cambia caller UUID, distinct from the display-adjusted scores that still
// carry WinBonus (cambia-1009) and from the -1-sentinel caller a circuit round used to receive
// (cambia-1008). FalseCambiaPenalty is zeroed so the only display adjustment in play is
// WinBonus, keeping the expected raw/adjusted diff a single deterministic value regardless of
// which player the random deal makes the round's winner.
func TestEndGameSeparatesRawScoresFromCircuitAdjustments(t *testing.T) {
	rules := testHouseRules(0, 2)
	g, players, _ := setupTestGame(t, 2, rules)
	g.Circuit.Enabled = true
	g.Circuit.Rules.WinBonus = -7
	g.Circuit.Rules.FalseCambiaPenalty = 0

	var capturedScores, capturedRaw map[uuid.UUID]int
	var capturedCaller uuid.UUID
	g.OnGameEnd = func(_ uuid.UUID, _ uuid.UUID, scores map[uuid.UUID]int, _ map[uuid.UUID]string, rawScores map[uuid.UUID]int, cambiaCallerID uuid.UUID, _ []FinalHand) {
		capturedScores = scores
		capturedRaw = rawScores
		capturedCaller = cambiaCallerID
	}

	first := currentTurnPlayer(g)
	var firstP, secondP *models.Player
	if first.ID == players[0].ID {
		firstP, secondP = players[0], players[1]
	} else {
		firstP, secondP = players[1], players[0]
	}

	doSimpleTurn := func(player *models.Player) {
		g.HandlePlayerAction(player.ID, models.GameAction{ActionType: "action_draw_stockpile"})
		engineIdx := g.PlayerToEngine[player.ID]
		drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
		if drawnUUID != uuid.Nil {
			g.HandlePlayerAction(player.ID, models.GameAction{
				ActionType: "action_discard",
				Payload:    map[string]interface{}{"id": drawnUUID.String()},
			})
			if g.SpecialAction.Active && g.SpecialAction.PlayerID == player.ID {
				g.ProcessSpecialAction(player.ID, "skip", nil, nil)
			}
		}
	}

	doSimpleTurn(firstP)
	g.HandlePlayerAction(secondP.ID, models.GameAction{ActionType: "action_cambia"})
	doSimpleTurn(firstP)

	if !g.GameOver {
		t.Fatal("game should be over after the final turn")
	}
	if capturedRaw == nil {
		t.Fatal("OnGameEnd was not invoked with rawScores")
	}

	// The real caller (secondP) must reach OnGameEnd, not uuid.Nil.
	if capturedCaller != secondP.ID {
		t.Errorf("cambiaCallerID: want the real caller %s, got %s", secondP.ID, capturedCaller)
	}

	// With FalseCambiaPenalty zeroed, the only legal raw/adjusted diff per player is 0 (not a
	// winner) or the WinBonus (-7, a winner) - and since the round always has at least one
	// winner and WinBonus != 0, at least one player must show that diff.
	sawWinBonus := false
	for _, p := range players {
		diff := capturedScores[p.ID] - capturedRaw[p.ID]
		switch diff {
		case 0:
		case g.Circuit.Rules.WinBonus:
			sawWinBonus = true
		default:
			t.Errorf("player %s adjusted-raw diff: want 0 or %d, got %d (scores=%d raw=%d)",
				p.ID, g.Circuit.Rules.WinBonus, diff, capturedScores[p.ID], capturedRaw[p.ID])
		}
	}
	if !sawWinBonus {
		t.Errorf("expected at least one player's display score to carry the WinBonus (-7) that rawScores must exclude")
	}
}
