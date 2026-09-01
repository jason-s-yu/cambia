// internal/game/sync_state.go
package game

import (
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
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
	// Forfeited is set once the player's reconnect window has closed (or immediately on the drop
	// where the grace is 0). Connected alone cannot carry this: inside the window a player is
	// disconnected but still in the game, and the two states read differently at the table
	// (cambia-955).
	Forfeited bool `json:"forfeited"`
	// ReconnectDeadline is the epoch-ms time this player's reconnect window closes, present only
	// while one is open. It lets a client that joins or resyncs mid-window render the same
	// countdown as the clients that saw the player_reconnecting event.
	ReconnectDeadline *int64 `json:"reconnectDeadline,omitempty"`
	// RevealedHand is populated only for the player requesting the state ('self').
	RevealedHand []ObfCard `json:"revealedHand,omitempty"`
	// DrawnCard is populated only for the player requesting the state ('self').
	DrawnCard *ObfCard `json:"drawnCard,omitempty"`
}

// ObfSpecialActionState is the public-safe projection of SpecialActionState serialized into
// sync_state: which player owes a pending multi-step special action and what rank triggered it.
// Peeked card values (SpecialActionState.Card1/Card2) are intentionally omitted here - those are
// private to the acting player and already delivered via private_special_action_success; leaking
// them through a state any client can request would break the King/peek information model.
type ObfSpecialActionState struct {
	Active   bool      `json:"active"`
	PlayerID uuid.UUID `json:"playerId"`
	CardRank string    `json:"cardRank"`
	// Mandatory says the ability cannot be declined, so a client restoring this prompt after a
	// reconnect knows not to offer a skip that the server would only refuse (cambia-1125).
	Mandatory bool `json:"mandatory,omitempty"`
}

// ObfSnapMoveState is one outstanding snap fill (RULES.md 5, cambia-936): the snapper owes a card
// into VictimID's hand at Slot. Everything here was already public the moment the snap resolved
// (player_snap_success names the victim and the slot), and which card pays it is not decided yet,
// so the projection leaks nothing. It is serialized so a client that resyncs mid-obligation - a
// reconnect, a tab refresh, a repair after a dropped frame - restores the prompt instead of sitting
// on a table that refuses its every action until the deadline fires.
type ObfSnapMoveState struct {
	SnapperID uuid.UUID `json:"snapperId"`
	VictimID  uuid.UUID `json:"victimId"`
	Slot      int       `json:"slot"`
	// Deadline is the epoch-ms time the server fills for the snapper, omitted when the table plays
	// without a turn timer and nothing is armed.
	Deadline *int64 `json:"deadline,omitempty"`
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
	// SpecialAction reports a pending multi-step special action (e.g. a King's look-then-swap),
	// letting a client that resyncs mid-action (reconnect, tab refresh) restore its pendingAction
	// UI instead of waiting for the next event. Nil/omitted when no special action is pending
	// (cambia-763 F1: previously never serialized, so reconnecting mid-action left the client
	// unable to restore its pendingAction state).
	SpecialAction *ObfSpecialActionState `json:"specialAction,omitempty"`
	// SnapMoves lists every outstanding snap fill, one per snapper who owes one. A list rather than
	// a single entry because two players can each owe one off the same discard when snapRace is off
	// (cambia-936).
	SnapMoves []ObfSnapMoveState `json:"snapMoves,omitempty"`
	// TurnDeadline is the absolute server-clock epoch-ms time the current turn's timer expires.
	// Omitted (null) when no turn timer is configured/active, in which case the client falls back
	// to an informational (non-counting-down) render.
	TurnDeadline *int64 `json:"turnDeadline,omitempty"`
	// ServerNow is this snapshot's server-clock epoch-ms send time, letting the client compute a
	// serverNow-clientNow clock skew offset to apply against TurnDeadline.
	ServerNow int64 `json:"serverNow"`
}

// GetCurrentObfuscatedGameState is the public entry point for reading an observer-tailored
// state snapshot. It acquires mu and delegates to getCurrentObfuscatedGameState. Internal
// callers that already hold mu (sendSyncState) call getCurrentObfuscatedGameState directly.
func (g *CambiaGame) GetCurrentObfuscatedGameState(forUser uuid.UUID) ObfGameState {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.getCurrentObfuscatedGameState(forUser)
}

// getCurrentObfuscatedGameState generates a snapshot of the game state,
// tailored to the perspective of the requesting user (`forUser`).
// Reads from engine state as the authoritative source.
// This function assumes the game lock is HELD by the caller.
func (g *CambiaGame) getCurrentObfuscatedGameState(forUser uuid.UUID) ObfGameState {
	obf := ObfGameState{
		GameID:        g.ID,
		PreGameActive: g.PreGameActive,
		Started:       g.Started,
		GameOver:      g.Engine.IsTerminal() || g.GameOver,
		TurnID:        int(g.Engine.TurnNumber),
		StockpileSize: int(g.Engine.StockLen),
		// Both piles are reported as the table sees them: discardSize counts the ability card that
		// has been announced onto the pile but not yet applied to the engine (cambia-1033).
		DiscardSize:  g.discardSize(),
		CambiaCalled: g.Engine.IsCambiaCalled(),
		HouseRules:   g.HouseRules,
		ServerNow:    time.Now().UnixMilli(),
	}

	// Turn deadline: only advertised while a turn timer is actually armed (TurnDeadline is the
	// zero value otherwise - see scheduleNextTurnTimerEngine) AND the game is actually live.
	// endGame() stops turnTimer but does not clear TurnDeadline, so without the Started/GameOver
	// guard a finished game would keep echoing its last (never-firing) deadline in sync_state.
	if g.Started && !obf.GameOver && !g.TurnDeadline.IsZero() {
		deadlineMs := g.TurnDeadline.UnixMilli()
		obf.TurnDeadline = &deadlineMs
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

	// Pending special action (cambia-763 F1). Public-safe projection only; see ObfSpecialActionState.
	if g.SpecialAction.Active {
		obf.SpecialAction = &ObfSpecialActionState{
			Active:    true,
			PlayerID:  g.SpecialAction.PlayerID,
			CardRank:  g.SpecialAction.CardRank,
			Mandatory: g.SpecialAction.MustResolve(),
		}
	}

	// Outstanding snap fills (cambia-936). Walked in seating order rather than map order so the
	// list is stable between snapshots.
	if len(g.snapFills) > 0 {
		for _, pl := range g.Players {
			fill, owed := g.snapFills[pl.ID]
			if !owed {
				continue
			}
			entry := ObfSnapMoveState{
				SnapperID: fill.SnapperID,
				VictimID:  fill.VictimID,
				Slot:      int(fill.Slot),
			}
			if !fill.Deadline.IsZero() {
				deadlineMs := fill.Deadline.UnixMilli()
				entry.Deadline = &deadlineMs
			}
			obf.SnapMoves = append(obf.SnapMoves, entry)
		}
	}

	// Discard top card (always public knowledge). Read through the same effective view the snap path
	// judges against: while an ability discard is buffered, the top is the card every client was
	// shown, not the one it covers. A snapshot naming the covered card contradicted the pile the
	// clients had already rendered, and a client that resynced mid-window (reconnect, tab refresh)
	// adopted it (cambia-1033).
	if topCard, topUUID, ok := g.effectiveDiscardTop(); ok && topUUID != uuid.Nil {
		obf.DiscardTop = &ObfCard{
			ID:    topUUID,
			Known: true,
			Rank:  engineRankToString(topCard.Rank()),
			Suit:  engineSuitToString(topCard.Suit()),
			Value: int(topCard.Value()),
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
			Forfeited: g.forfeited[pl.ID],
		}
		if deadline, open := g.graceDeadlines[pl.ID]; open {
			deadlineMs := deadline.UnixMilli()
			ps.ReconnectDeadline = &deadlineMs
		}

		if hasMapping {
			ps.HandSize = int(g.Engine.Players[engineIdx].HandLen)
			ps.HasCalledCambia = (g.Engine.CambiaCaller == int8(engineIdx))
			ps.IsCurrentTurn = (g.Engine.ActingPlayer() == engineIdx && g.Started && !obf.GameOver)

			// The round-end reveal (RULES.md 3C, cambia-1542). Once the round is over every scored
			// hand turns face-up, own and opponent alike, off the same obf.GameOver this snapshot
			// already computed. Nothing below it changes before that point: while the round runs,
			// every slot is still an id and an index with Known:false, which is the cambia-1094
			// rule and the reason this is a carve-out from it rather than a rollback of it.
			//
			// A forfeited seat stays face-down. It is not scored (computeScoresFromEngine) and is
			// left out of the game_end reveal too (buildFinalReveal), so revealing it here would put
			// a hand on the table that the results frame does not name.
			revealFinal := obf.GameOver && !g.forfeited[pl.ID]

			if isSelf {
				// Self-view: expose every hand slot as an id+index reference with the face hidden
				// ALWAYS (Known:false, no rank/suit/value), exactly like an opponent's hand. No own
				// card is ever persistently face-up: the physical game gives you the pregame peek and
				// then turns every card down, and you play the rest of the round on memory
				// (cambia-1094, replacing the cambia-505 memory aid).
				//
				// The reveals a player is entitled to travel in their own events, never in a
				// snapshot: the pregame peek in private_initial_cards (re-fired on a reconnect that
				// lands inside the pregame window, see HandleReconnect), a drawn card in
				// private_draw_stockpile and DrawnCard below, an ability look in
				// private_special_action_success. The client shows each of those for its window and
				// then turns the card down. A snapshot that repeated them would make every one of
				// those windows permanent, which is the bug.
				//
				// The slot ids and indices stay: ability targeting names an own card by id
				// (peek_self, swap_blind, swap_peek) and the client needs a real UUID per slot
				// (cambia-509).
				ps.RevealedHand = g.handSlots(engineIdx, revealFinal)

				// Drawn card (pending discard in engine). An ability card whose discard is still
				// buffered stays pending in the engine, but the table has already been told it was
				// discarded and this snapshot reports it as the pile's top: naming it here as well
				// showed the same card twice, and the client reads a drawn card as "discard or
				// replace", which put the discarder back on that prompt instead of the ability one it
				// actually owes (web/src/stores/gameStore.ts, private_sync_state; cambia-1033).
				if g.Engine.Pending.Type == engine.PendingDiscard &&
					g.Engine.Pending.PlayerID == engineIdx && !g.pendingDiscardAbilityChoice {
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
			} else {
				// Opponent view: expose each hand slot as an id+index reference with the face
				// hidden ALWAYS (Known:false, no rank/suit/value). Card UUIDs are already public
				// knowledge (draw/discard/special event payloads broadcast them), so id+slot leaks
				// nothing. This gives the client a real UUID to target opponent-facing abilities
				// (9/T peek_other, J/Q swap_blind, K swap_peek) instead of fabricating an
				// unparseable placeholder id (cambia-509).
				ps.RevealedHand = g.handSlots(engineIdx, revealFinal)
			}
		}

		obf.Players[i] = ps
	}

	return obf
}

// handSlots projects one seat's hand into snapshot slots. Both views build the same thing, so
// they build it here: a card id and its slot index per card, face hidden.
//
// reveal turns the faces up, and only the round-end reveal passes it true (RULES.md 3C,
// cambia-1542). With it false the result is the cambia-1094 projection unchanged, which is what
// every call during a live round makes.
//
// Assumes the lock is held by the caller.
func (g *CambiaGame) handSlots(engineIdx uint8, reveal bool) []ObfCard {
	handLen := g.Engine.Players[engineIdx].HandLen
	slots := make([]ObfCard, handLen)
	for j := uint8(0); j < handLen; j++ {
		idx := int(j)
		slots[j] = ObfCard{
			ID:    g.CardTracker.Players[engineIdx].HandUUIDs[j],
			Known: false,
			Idx:   &idx,
		}
		if reveal {
			card := g.Engine.Players[engineIdx].Hand[j]
			slots[j].Known = true
			slots[j].Rank = engineRankToString(card.Rank())
			slots[j].Suit = engineSuitToString(card.Suit())
			slots[j].Value = int(card.Value())
		}
	}
	return slots
}
