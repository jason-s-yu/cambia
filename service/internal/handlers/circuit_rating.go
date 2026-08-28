// internal/handlers/circuit_rating.go
package handlers

import (
	"context"
	"log"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// finalizeCircuitRatings applies the single rating update a circuit tournament produces, from the
// standings it concludes on. RULES.md T6 and MATCHMAKING.md 6.2/6.3 rate a multi-round format
// strictly once, at its conclusion, from the final cumulative scores: the rounds are recorded and
// displayed but never rated (game.CambiaGame.ratePerGame), and this is the update that stands for
// all of them. Finishers within rating.CircuitTieMargin cumulative points are recorded as a tie
// (database.RecordCircuitRatings).
//
// g supplies the id the ratings rows are attributed to (the circuit's final round, the only games
// row a circuit has) and whether the circuit was rated at all. standings must be read from the
// CircuitState before it is dropped from the CircuitStore, and playerMap is that circuit's
// engine-id mapping.
//
// Called from OnGameEnd, which runs inside endGame while the game mutex is held, so the database
// work is handed to a goroutine rather than run inline: this returns immediately. It reads only
// fields fixed at game creation (id, rated), so it never re-enters that mutex.
//
// The goroutine is registered on GameServer.PersistWG (test-only; nil in production) before it
// starts, the same discipline persistFinalGameState follows, so a test can drain it. The Add runs
// later in OnGameEnd than the game_results broadcast, so a test that means to cover this write
// must order its Wait after something this call precedes - the circuit_complete emit, or the call
// itself - rather than after the results broadcast alone (see awaitGameEndPersistence for why
// sync.WaitGroup demands that ordering).
func (gs *GameServer) finalizeCircuitRatings(g *game.CambiaGame, standings []engine.CircuitPlayerState, playerMap map[uuid.UUID]int) {
	if g == nil || !g.Rated || database.DB == nil {
		return
	}
	roster, cumulative := circuitCumulativeScores(standings, playerMap)
	if len(roster) == 0 {
		log.Printf("Game %s: circuit concluded with no rateable players; skipping rating update.", g.ID)
		return
	}

	gameID := g.ID
	wg := gs.PersistWG
	if wg != nil {
		wg.Add(1)
	}
	go func() {
		if wg != nil {
			defer wg.Done()
		}
		if err := database.RecordCircuitRatings(context.Background(), gameID, roster, cumulative); err != nil {
			log.Printf("Game %s: failed to record circuit ratings: %v", gameID, err)
		}
	}()
}

// circuitCumulativeScores translates a circuit's final standings into the roster and cumulative
// score map the rating path reads: engine player ids become service player UUIDs via playerMap,
// and each player's CumulativeScore (subsidies already applied, lower is better) becomes its
// score. The roster keeps the standings order, so the update does not depend on map iteration
// order. A standing with no service-side player is dropped.
func circuitCumulativeScores(standings []engine.CircuitPlayerState, playerMap map[uuid.UUID]int) ([]uuid.UUID, map[uuid.UUID]int) {
	byEngineID := make(map[int]uuid.UUID, len(playerMap))
	for playerUUID, engineID := range playerMap {
		byEngineID[engineID] = playerUUID
	}

	roster := make([]uuid.UUID, 0, len(standings))
	cumulative := make(map[uuid.UUID]int, len(standings))
	for _, s := range standings {
		playerUUID, ok := byEngineID[s.PlayerID]
		if !ok {
			log.Printf("Circuit standings: no player mapped to engine id %d; excluded from the rating update.", s.PlayerID)
			continue
		}
		roster = append(roster, playerUUID)
		cumulative[playerUUID] = s.CumulativeScore
	}
	return roster, cumulative
}
