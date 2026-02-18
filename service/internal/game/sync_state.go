// internal/game/sync_state.go
package game

import (
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/google/uuid"
)

// ObfCard represents a card's state for client synchronization, potentially hiding details.
type ObfCard struct {
	ID    uuid.UUID `json:"id"`
	Known bool      `json:"known"` // True if Rank/Suit/Value should be revealed to the requesting client.
	Rank  string    `json:"rank,omitempty"`
	Suit  string    `json:"suit,omitempty"`
	Value int       `json:"value,omitempty"`
	Idx   *int      `json:"idx,omitempty"` // Pointer to allow omitting zero index (relevant for hand cards).
}

// ObfPlayerState represents the state of a single player, obfuscated for a specific observer.
type ObfPlayerState struct {
	PlayerID        uuid.UUID `json:"playerId"`
	Username        string    `json:"username"`
	HandSize        int       `json:"handSize"`
	HasCalledCambia bool      `json:"hasCalledCambia"`
	Connected       bool      `json:"connected"`
	IsCurrentTurn   bool      `json:"isCurrentTurn"`
	// RevealedHand is populated only for the player requesting the state ('self').
	RevealedHand []ObfCard `json:"revealedHand,omitempty"`
	// DrawnCard is populated only for the player requesting the state ('self').
	DrawnCard *ObfCard `json:"drawnCard,omitempty"`
}

// ObfGameState represents the overall game state, obfuscated for a specific observer.
type ObfGameState struct {
	GameID          uuid.UUID        `json:"gameId"`
	PreGameActive   bool             `json:"preGameActive"`
	Started         bool             `json:"started"`
	GameOver        bool             `json:"gameOver"`
	CurrentPlayerID uuid.UUID        `json:"currentPlayerId"`
	TurnID          int              `json:"turnId"`
	StockpileSize   int              `json:"stockpileSize"`
	DiscardSize     int              `json:"discardSize"`
	DiscardTop      *ObfCard         `json:"discardTop,omitempty"`
	Players         []ObfPlayerState `json:"players"`
	CambiaCalled    bool             `json:"cambiaCalled"`
	CambiaCallerID  uuid.UUID        `json:"cambiaCallerId,omitempty"`
	HouseRules      HouseRules       `json:"houseRules"`
}

// GetCurrentObfuscatedGameState generates a snapshot of the game state,
// tailored to the perspective of the requesting user (`forUser`).
// Reads from engine state as the authoritative source.
// This function assumes the game lock is HELD by the caller.
func (g *CambiaGame) GetCurrentObfuscatedGameState(forUser uuid.UUID) ObfGameState {
	obf := ObfGameState{
		GameID:        g.ID,
		PreGameActive: g.PreGameActive,
		Started:       g.Started,
		GameOver:      g.Engine.IsTerminal() || g.GameOver,
		TurnID:        int(g.Engine.TurnNumber),
		StockpileSize: int(g.Engine.StockLen),
		DiscardSize:   int(g.Engine.DiscardLen),
		CambiaCalled:  g.Engine.IsCambiaCalled(),
		HouseRules:    g.HouseRules,
	}

	// Current player.
	if g.Started && !obf.GameOver && len(g.Players) > 0 {
		actingIdx := g.Engine.ActingPlayer()
		if int(actingIdx) < len(g.Players) {
			obf.CurrentPlayerID = g.EngineToPlayer[actingIdx]
		}
	}

	// Cambia caller.
	if g.Engine.CambiaCaller >= 0 && int(g.Engine.CambiaCaller) < engine.MaxPlayers {
		obf.CambiaCallerID = g.EngineToPlayer[uint8(g.Engine.CambiaCaller)]
	}

	// Discard top card (always public knowledge).
	if g.Engine.DiscardLen > 0 {
		topIdx := g.Engine.DiscardLen - 1
		topCard := g.Engine.DiscardPile[topIdx]
		topUUID := g.CardTracker.DiscardUUIDs[topIdx]
		if topUUID != uuid.Nil {
			obf.DiscardTop = &ObfCard{
				ID:    topUUID,
				Known: true,
				Rank:  engineRankToString(topCard.Rank()),
				Suit:  engineSuitToString(topCard.Suit()),
				Value: int(topCard.Value()),
			}
		}
	}

	// Player states.
	obf.Players = make([]ObfPlayerState, len(g.Players))
	for i, pl := range g.Players {
		engineIdx, hasMapping := g.PlayerToEngine[pl.ID]
		isSelf := (pl.ID == forUser)

		ps := ObfPlayerState{
			PlayerID:  pl.ID,
			Username:  pl.User.Username,
			Connected: pl.Connected,
		}

		if hasMapping {
			ps.HandSize = int(g.Engine.Players[engineIdx].HandLen)
			ps.HasCalledCambia = (g.Engine.CambiaCaller == int8(engineIdx))
			ps.IsCurrentTurn = (g.Engine.ActingPlayer() == engineIdx && g.Started && !obf.GameOver)

			if isSelf {
				// Reveal hand details for self.
				handLen := g.Engine.Players[engineIdx].HandLen
				ps.RevealedHand = make([]ObfCard, handLen)
				for j := uint8(0); j < handLen; j++ {
					card := g.Engine.Players[engineIdx].Hand[j]
					cardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[j]
					idx := int(j)
					ps.RevealedHand[j] = ObfCard{
						ID:    cardUUID,
						Known: true,
						Rank:  engineRankToString(card.Rank()),
						Suit:  engineSuitToString(card.Suit()),
						Value: int(card.Value()),
						Idx:   &idx,
					}
				}

				// Drawn card (pending discard in engine).
				if g.Engine.Pending.Type == engine.PendingDiscard &&
					g.Engine.Pending.PlayerID == engineIdx {
					drawnCard := engine.Card(g.Engine.Pending.Data[0])
					drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
					if drawnUUID != uuid.Nil {
						ps.DrawnCard = &ObfCard{
							ID:    drawnUUID,
							Known: true,
							Rank:  engineRankToString(drawnCard.Rank()),
							Suit:  engineSuitToString(drawnCard.Suit()),
							Value: int(drawnCard.Value()),
						}
					}
				}
			}
		}

		obf.Players[i] = ps
	}

	return obf
}
