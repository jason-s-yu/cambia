// internal/game/nplayer_targeting_test.go
//
// N-player (3+ seat) targeting for opponent-facing abilities and snaps (cambia-946).
//
// The adapter used to derive the opponent seat as `1 - engineIdx`, which is only correct for a
// 2-seat game: from seat 2 the uint8 subtraction wraps to 255 and indexes the engine's fixed
// 8-slot Players array, panicking the whole process; from seat 0 or 1 it silently targeted the
// wrong player in a 4-seat game. These tests drive a 4-player game where seats 2 and 3 act, and
// assert the seat the client named is the seat the engine mutates.
package game

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// neutralCard is a rank with no ability, so a draw-and-discard turn never opens a special action.
func neutralCard() engine.Card { return engine.NewCard(engine.SuitClubs, engine.RankTwo) }

// playNeutralTurn draws a no-ability card for the acting player and discards it, ending the turn.
func playNeutralTurn(t *testing.T, g *CambiaGame) {
	t.Helper()
	p := currentTurnPlayer(g)
	require.NotNil(t, p, "acting player must resolve")
	forceStockTop(g, neutralCard())
	g.HandlePlayerAction(p.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	engineIdx := g.PlayerToEngine[p.ID]
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "draw should set a drawn card UUID")
	g.HandlePlayerAction(p.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	})
}

// advanceToSeat plays neutral turns until the given engine seat is the acting seat.
func advanceToSeat(t *testing.T, g *CambiaGame, seat uint8) *models.Player {
	t.Helper()
	for i := 0; i < 12 && g.Engine.ActingPlayer() != seat; i++ {
		playNeutralTurn(t, g)
	}
	require.Equal(t, seat, g.Engine.ActingPlayer(), "should have reached seat %d", seat)
	return currentTurnPlayer(g)
}

// privateEventsOfType returns every private event of the given type captured for playerID.
func privateEventsOfType(mb *mockBroadcaster, playerID uuid.UUID, t GameEventType) []GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	var out []GameEvent
	for _, ev := range mb.playerEvents[playerID] {
		if ev.Type == t {
			out = append(out, ev)
		}
	}
	return out
}

// publicEventsOfType returns every broadcast event of the given type.
func publicEventsOfType(mb *mockBroadcaster, t GameEventType) []GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	var out []GameEvent
	for _, ev := range mb.allEvents {
		if ev.Type == t {
			out = append(out, ev)
		}
	}
	return out
}

// TestFourPlayerPeekOtherTargetsNamedSeat drives 9/T peek_other from seat 2 against seats 0, 1 and
// 3 and asserts the revealed card is the one the client named, in the named seat's hand.
func TestFourPlayerPeekOtherTargetsNamedSeat(t *testing.T) {
	for _, targetSeat := range []uint8{0, 1, 3} {
		targetSeat := targetSeat
		t.Run("target_seat", func(t *testing.T) {
			g, _, mb := setupTestGame(t, 4, testHouseRules(0, 2))
			actor := advanceToSeat(t, g, 2)
			targetID := g.EngineToPlayer[targetSeat]
			targetUUID := g.CardTracker.Players[targetSeat].HandUUIDs[1]
			wantCard := g.Engine.Players[targetSeat].Hand[1]
			mb.clear()

			drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankNine), "9")
			g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(targetUUID, targetID, 1), nil)

			revealed := privateEventsOfType(mb, actor.ID, EventPrivateSpecialSuccess)
			require.NotEmpty(t, revealed, "peek_other should reveal a card to the actor")
			ev := revealed[len(revealed)-1]
			require.NotNil(t, ev.Card1)
			assert.Equal(t, targetUUID, ev.Card1.ID, "revealed card must be the named card")
			require.NotNil(t, ev.Card1.User)
			assert.Equal(t, targetID, ev.Card1.User.ID, "revealed card must name the target seat's owner")
			assert.Equal(t, engineRankToString(wantCard.Rank()), ev.Card1.Rank, "revealed rank must be the target seat's card")
		})
	}
}

// TestFourPlayerSwapBlindTargetsNamedSeat drives J/Q swap_blind from seat 2 against seat 3 and
// asserts the engine swapped seat 2 and seat 3 hands, leaving seats 0 and 1 untouched.
func TestFourPlayerSwapBlindTargetsNamedSeat(t *testing.T) {
	g, _, mb := setupTestGame(t, 4, testHouseRules(0, 2))
	actor := advanceToSeat(t, g, 2)
	const actorSeat, targetSeat = uint8(2), uint8(3)
	targetID := g.EngineToPlayer[targetSeat]

	ownUUID := g.CardTracker.Players[actorSeat].HandUUIDs[0]
	oppUUID := g.CardTracker.Players[targetSeat].HandUUIDs[2]
	ownCard := g.Engine.Players[actorSeat].Hand[0]
	oppCard := g.Engine.Players[targetSeat].Hand[2]
	seat0Before := g.Engine.Players[0].Hand
	seat1Before := g.Engine.Players[1].Hand
	mb.clear()

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankJack), "J")
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownUUID, actor.ID, 0),
		cardTarget(oppUUID, targetID, 2))

	assert.Equal(t, oppCard, g.Engine.Players[actorSeat].Hand[0], "actor slot must hold the target seat's card")
	assert.Equal(t, ownCard, g.Engine.Players[targetSeat].Hand[2], "target seat must hold the actor's card")
	assert.Equal(t, oppUUID, g.CardTracker.Players[actorSeat].HandUUIDs[0], "tracker must follow the swap")
	assert.Equal(t, ownUUID, g.CardTracker.Players[targetSeat].HandUUIDs[2], "tracker must follow the swap")
	assert.Equal(t, seat0Before, g.Engine.Players[0].Hand, "seat 0 must be untouched")
	assert.Equal(t, seat1Before, g.Engine.Players[1].Hand, "seat 1 must be untouched")

	pub := publicEventsOfType(mb, EventPlayerSpecialAction)
	require.NotEmpty(t, pub, "swap_blind should broadcast a special action")
	ev := pub[len(pub)-1]
	require.NotNil(t, ev.Card2)
	require.NotNil(t, ev.Card2.User)
	assert.Equal(t, targetID, ev.Card2.User.ID, "swap event must name the target seat's owner")
}

// TestFourPlayerKingTargetsNamedSeat drives the King look-and-swap from seat 3 against seat 1.
func TestFourPlayerKingTargetsNamedSeat(t *testing.T) {
	g, _, mb := setupTestGame(t, 4, testHouseRules(0, 2))
	actor := advanceToSeat(t, g, 3)
	const actorSeat, targetSeat = uint8(3), uint8(1)
	targetID := g.EngineToPlayer[targetSeat]

	ownUUID := g.CardTracker.Players[actorSeat].HandUUIDs[1]
	oppUUID := g.CardTracker.Players[targetSeat].HandUUIDs[0]
	ownCard := g.Engine.Players[actorSeat].Hand[1]
	oppCard := g.Engine.Players[targetSeat].Hand[0]
	mb.clear()

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankKing), "K")
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownUUID, actor.ID, 1),
		cardTarget(oppUUID, targetID, 0))

	revealed := privateEventsOfType(mb, actor.ID, EventPrivateSpecialSuccess)
	require.NotEmpty(t, revealed, "King look should reveal both cards to the actor")
	look := revealed[len(revealed)-1]
	require.NotNil(t, look.Card2)
	require.NotNil(t, look.Card2.User)
	assert.Equal(t, targetID, look.Card2.User.ID, "King look must name the target seat's owner")
	assert.Equal(t, engineRankToString(oppCard.Rank()), look.Card2.Rank, "King look must reveal the target seat's card")

	g.ProcessSpecialAction(actor.ID, "swap_peek_swap", nil, nil)

	assert.Equal(t, oppCard, g.Engine.Players[actorSeat].Hand[1], "actor slot must hold the target seat's card")
	assert.Equal(t, ownCard, g.Engine.Players[targetSeat].Hand[0], "target seat must hold the actor's card")
	assert.Equal(t, oppUUID, g.CardTracker.Players[actorSeat].HandUUIDs[1], "tracker must follow the King swap")
	assert.Equal(t, ownUUID, g.CardTracker.Players[targetSeat].HandUUIDs[0], "tracker must follow the King swap")
}

// TestFourPlayerSnapOfThirdPlayerCard has seat 2 snap a card held by seat 3: the pre-fix opponent
// search only ever looked at seat `1 - engineIdx`, so a third player's card fell through to the
// failed-snap penalty (and wrapped to seat 255 from seat 2 outright).
func TestFourPlayerSnapOfThirdPlayerCard(t *testing.T) {
	g, _, mb := setupTestGame(t, 4, testHouseRules(0, 2))
	snapper := advanceToSeat(t, g, 2)
	const snapperSeat, victimSeat = uint8(2), uint8(3)
	victimID := g.EngineToPlayer[victimSeat]

	// Put a matching rank in the victim's hand and on top of the discard pile.
	match := engine.NewCard(engine.SuitHearts, engine.RankFive)
	g.Engine.Players[victimSeat].Hand[1] = match
	victimUUID := g.CardTracker.Players[victimSeat].HandUUIDs[1]
	g.CardTracker.Registry[victimUUID] = engineCardToDetails(match, victimUUID)

	discardTop := engine.NewCard(engine.SuitSpades, engine.RankFive)
	g.Engine.DiscardPile[g.Engine.DiscardLen] = discardTop
	topUUID := uuid.New()
	g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen] = topUUID
	g.CardTracker.Registry[topUUID] = engineCardToDetails(discardTop, topUUID)
	g.Engine.DiscardLen++
	g.CardTracker.DiscardLen = g.Engine.DiscardLen
	g.snapUsedForThisDiscard = false

	victimHandLen := g.Engine.Players[victimSeat].HandLen
	snapperHandLen := g.Engine.Players[snapperSeat].HandLen
	mb.clear()

	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": victimUUID.String()},
	})

	success := publicEventsOfType(mb, EventPlayerSnapSuccess)
	require.NotEmpty(t, success, "snapping a third player's matching card must succeed")
	ev := success[len(success)-1]
	require.NotNil(t, ev.User)
	assert.Equal(t, snapper.ID, ev.User.ID, "snap event must name the snapper")
	require.NotNil(t, ev.Card)
	require.NotNil(t, ev.Card.User)
	assert.Equal(t, victimID, ev.Card.User.ID, "snap event must name the card's owner")
	assert.Empty(t, publicEventsOfType(mb, EventPlayerSnapFail), "a matching snap must not be penalized")

	assert.Equal(t, victimHandLen-1, g.Engine.Players[victimSeat].HandLen, "victim hand must shrink by one")
	assert.Equal(t, snapperHandLen, g.Engine.Players[snapperSeat].HandLen, "snapper hand must be unchanged")
}

// TestFourPlayerAbilityRejectsBadTarget covers the reject-and-wait paths for an N-player target:
// an unknown owner, the actor targeting themselves as the opponent, and an out-of-range slot in
// the named seat's hand. Each must fire a private fail and leave the buffered discard pending.
func TestFourPlayerAbilityRejectsBadTarget(t *testing.T) {
	cases := []struct {
		name         string
		target       func(g *CambiaGame, actor *models.Player) map[string]interface{}
		wantAccepted bool
	}{
		{
			name: "unknown owner",
			target: func(g *CambiaGame, actor *models.Player) map[string]interface{} {
				return cardTarget(g.CardTracker.Players[0].HandUUIDs[0], uuid.New(), 0)
			},
		},
		{
			name: "owner is the actor",
			target: func(g *CambiaGame, actor *models.Player) map[string]interface{} {
				return cardTarget(g.CardTracker.Players[2].HandUUIDs[0], actor.ID, 0)
			},
		},
		{
			// An unknown card id at an out-of-range slot has nothing to fall back on: the named
			// index is past the target's hand and the id is in nobody's hand.
			name: "slot out of range",
			target: func(g *CambiaGame, actor *models.Player) map[string]interface{} {
				seat := uint8(3)
				return cardTarget(uuid.New(), g.EngineToPlayer[seat], int(g.Engine.Players[seat].HandLen))
			},
		},
		{
			// A stale index the client still believes in, with a live card id, self-heals to the
			// slot that id actually occupies rather than rejecting.
			name: "stale index heals to the card id",
			target: func(g *CambiaGame, actor *models.Player) map[string]interface{} {
				seat := uint8(3)
				return cardTarget(g.CardTracker.Players[seat].HandUUIDs[2], g.EngineToPlayer[seat], 9)
			},
			wantAccepted: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			g, _, mb := setupTestGame(t, 4, testHouseRules(0, 2))
			actor := advanceToSeat(t, g, 2)
			mb.clear()

			drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankNine), "9")
			g.ProcessSpecialAction(actor.ID, "peek_other", tc.target(g, actor), nil)

			if tc.wantAccepted {
				assert.Empty(t, actorFailEvents(mb, actor.ID), "a healable target must not fail")
				assert.NotEmpty(t, privateEventsOfType(mb, actor.ID, EventPrivateSpecialSuccess),
					"a healable target must reveal the named card")
				return
			}

			assert.NotEmpty(t, actorFailEvents(mb, actor.ID), "a bad target must fire a private fail")
			assert.True(t, g.SpecialAction.Active, "the ability must stay pending after a rejected target")
			assert.True(t, g.pendingDiscardAbilityChoice, "the buffered discard must stay unapplied")
			assert.Equal(t, uint8(2), g.Engine.ActingPlayer(), "a rejected target must not advance the turn")
		})
	}
}

// TestNoTwoPlayerOpponentWraparound guards the fix: `1 - engineIdx` (in any spacing) is only valid
// for a 2-seat game and wraps to seat 255 from seat 2, so it must not reappear in this package's
// production sources. Opponent seats come from resolveOpponentTarget / opponentSeatForAction,
// whose single 2-seat fallback is bounds-checked (nplayer_targeting.go).
func TestNoTwoPlayerOpponentWraparound(t *testing.T) {
	pattern := regexp.MustCompile(`1\s*-\s*(int\()?(engineIdx|actorEngineIdx|snapperIdx|seat)`)
	entries, err := os.ReadDir(".")
	require.NoError(t, err)

	var hits []string
	for _, e := range entries {
		// nplayer_targeting.go is the one sanctioned home for the 2-seat convention: its single
		// fallback rejects any seat above 1 before subtracting.
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") ||
			strings.HasSuffix(e.Name(), "_test.go") || e.Name() == "nplayer_targeting.go" {
			continue
		}
		body, err := os.ReadFile(filepath.Join(".", e.Name()))
		require.NoError(t, err)
		for i, line := range strings.Split(string(body), "\n") {
			if comment := strings.Index(line, "//"); comment >= 0 {
				line = line[:comment]
			}
			if pattern.MatchString(line) {
				hits = append(hits, e.Name()+":"+strings.TrimSpace(line)+" (line "+itoa(i+1)+")")
			}
		}
	}
	assert.Empty(t, hits, "opponent seats must be resolved from the target's owner id, not by 1 - engineIdx")
}

// itoa is a local strconv.Itoa to keep the guard test's imports minimal.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
