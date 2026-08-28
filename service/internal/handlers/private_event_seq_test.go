// internal/handlers/private_event_seq_test.go
//
// Private per-player frames must not strand the other player behind the hub sequence
// (cambia-878). Hub.seq is a staleness gate: dispatch() drops any client message whose last_seq
// trails h.seq and answers it with a sync_state repair. EmitTo consumed a seq for a frame only
// one connection ever receives, so every private event (a failed snap's penalty detail, an
// ability reveal, a player's own drawn card) left every other client one seq behind the hub with
// no way to learn the new number, and their next action was dropped. The cambia-848 table sweep
// saw it as draws that only landed on a retry.
//
// These are end-to-end over real WebSocket connections: the contract under test is what the wire
// carries, and the client cannot tell a private frame from a broadcast.
package handlers

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// mark returns the client's current frame count. Waits and counts taken from a mark ignore
// everything recorded before it, so a test driving many turns can assert on the frames one
// action produced rather than on the first matching frame of the whole run.
func (c *wsTestClient) mark() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.frames)
}

// findTypeFrom returns the first frame of msgType recorded at or after index start, without
// waiting. The second return is that frame's index.
func (c *wsTestClient) findTypeFrom(start int, msgType string) (*wsEnvelope, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := start; i < len(c.frames); i++ {
		if c.frames[i].Type == msgType {
			env := c.frames[i]
			return &env, i
		}
	}
	return nil, -1
}

// waitForTypeFrom polls findTypeFrom until it hits or the timeout elapses.
func (c *wsTestClient) waitForTypeFrom(start int, msgType string, timeout time.Duration) (*wsEnvelope, int) {
	deadline := time.Now().Add(timeout)
	for {
		if env, idx := c.findTypeFrom(start, msgType); env != nil {
			return env, idx
		}
		if time.Now().After(deadline) {
			return nil, -1
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// countTypeFrom counts frames of msgType recorded at or after index start.
func (c *wsTestClient) countTypeFrom(start int, msgType string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for i := start; i < len(c.frames); i++ {
		if c.frames[i].Type == msgType {
			n++
		}
	}
	return n
}

// seqsSeen returns every envelope seq this client received, and the lowest of them.
func (c *wsTestClient) seqsSeen() (map[uint64]struct{}, uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	set := make(map[uint64]struct{}, len(c.frames))
	var min uint64
	first := true
	for i := range c.frames {
		set[c.frames[i].Seq] = struct{}{}
		if first || c.frames[i].Seq < min {
			min = c.frames[i].Seq
			first = false
		}
	}
	return set, min
}

// sendAccepted sends a message and waits for the frame that acknowledges it, retrying past a
// sync_state bounce the way a real client recovers from the staleness gate. It exists for the
// setup steps of a test whose assertion is about some *other* action: the steps that must land
// on the first try are sent with the plain send/sendWith and asserted directly. Returns the
// acknowledging frame, or nil if the message was never accepted.
func (c *wsTestClient) sendAccepted(msgType string, extra map[string]interface{}, ackType string, timeout time.Duration) *wsEnvelope {
	for attempt := 0; attempt < 6; attempt++ {
		m := c.mark()
		if extra == nil {
			c.send(msgType)
		} else {
			c.sendWith(msgType, extra)
		}
		deadline := time.Now().Add(timeout)
		for time.Now().Before(deadline) {
			if env, _ := c.findTypeFrom(m, ackType); env != nil {
				return env
			}
			if c.countTypeFrom(m, "sync_state") > 0 {
				break // bounced as stale; readLoop has taken the repair's seq, so retry
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	return nil
}

// startTwoPlayerGame drives a fresh public lobby through ready -> countdown -> pre-game reveal
// and returns once both clients have seen the first game_player_turn, which is the point where
// both are caught up to the hub sequence and the game accepts actions.
func startTwoPlayerGame(t *testing.T) (*game.CambiaGame, uuid.UUID, uuid.UUID, *wsTestClient, *wsTestClient) {
	t.Helper()
	gs, ts := newForfeitTestServer(t)

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	t.Cleanup(host.close)
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	t.Cleanup(p2.close)
	host.settle()
	p2.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	if _, _ = host.waitForTypeFrom(0, "game_player_turn", 10*time.Second); host.countType("game_player_turn") == 0 {
		t.Fatalf("host never received game_player_turn (game never left pre-game)")
	}
	if _, _ = p2.waitForTypeFrom(0, "game_player_turn", 10*time.Second); p2.countType("game_player_turn") == 0 {
		t.Fatalf("player 2 never received game_player_turn (game never left pre-game)")
	}
	host.settle()
	p2.settle()

	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobbyID)
	}
	return g, hostID, p2ID, host, p2
}

// actingPair returns the client whose turn it is and its opponent.
func actingPair(t *testing.T, g *game.CambiaGame, hostID, p2ID uuid.UUID, host, p2 *wsTestClient) (uuid.UUID, *wsTestClient, uuid.UUID, *wsTestClient) {
	t.Helper()
	st := g.GetCurrentObfuscatedGameState(hostID)
	if st.GameOver || !st.Started {
		t.Fatalf("game is not in a playable state (started=%v over=%v)", st.Started, st.GameOver)
	}
	switch st.CurrentPlayerID {
	case hostID:
		return hostID, host, p2ID, p2
	case p2ID:
		return p2ID, p2, hostID, host
	default:
		t.Fatalf("current player %s is neither seated client", st.CurrentPlayerID)
		return uuid.Nil, nil, uuid.Nil, nil
	}
}

// handTarget builds a special-action card target for one slot of owner's hand, read from the
// obfuscated state the way a client reads it off sync_state (every slot carries an id and an
// index, opponents' included, since cambia-509).
func handTarget(t *testing.T, g *game.CambiaGame, viewer, owner uuid.UUID, slot int) map[string]interface{} {
	t.Helper()
	st := g.GetCurrentObfuscatedGameState(viewer)
	for _, p := range st.Players {
		if p.PlayerID != owner {
			continue
		}
		if slot >= len(p.RevealedHand) {
			t.Fatalf("player %s has %d hand slots, wanted slot %d", owner, len(p.RevealedHand), slot)
		}
		return map[string]interface{}{
			"id":   p.RevealedHand[slot].ID.String(),
			"idx":  slot,
			"user": map[string]interface{}{"id": owner.String()},
		}
	}
	t.Fatalf("player %s not found in the obfuscated state", owner)
	return nil
}

// TestFailedSnapPrivatePenaltyDoesNotDropThePeersDraw is the cambia-878 reproduction: one player
// fails a snap, which ends in a private_snap_penalty only they receive, and the other player's
// very next draw must be applied rather than answered with a sync_state and dropped.
//
// The snap failure is deterministic: a card id that is in nobody's hand can never match the
// discard top, so handleSnapViaEngine falls through to handleSnapFailure whatever the deal was.
func TestFailedSnapPrivatePenaltyDoesNotDropThePeersDraw(t *testing.T) {
	g, hostID, p2ID, host, p2 := startTwoPlayerGame(t)

	_, actor, _, peer := actingPair(t, g, hostID, p2ID, host, p2)

	// The player whose turn it is NOT fails a snap (snapping is legal off-turn).
	snapMark := peer.mark()
	peer.sendReliableWith("action_snap", map[string]interface{}{
		"card": map[string]interface{}{"id": uuid.New().String()},
	})
	if env, _ := peer.waitForTypeFrom(snapMark, "private_snap_penalty", 5*time.Second); env == nil {
		t.Fatalf("the failed snap never produced a private_snap_penalty for the snapping player")
	}
	// Let every frame the failure broadcast reach the acting player before it acts, so its
	// last_seq is genuinely the freshest seq it was ever told about.
	actor.settle()

	drawMark := actor.mark()
	actor.send("action_draw_stockpile")

	if env, _ := actor.waitForTypeFrom(drawMark, "private_draw_stockpile", 5*time.Second); env == nil {
		t.Fatalf("the acting player's draw was never applied after the peer's failed snap "+
			"(sync_state repairs since the draw: %d): a private frame to the peer consumed a seq "+
			"this client never observed", actor.countTypeFrom(drawMark, "sync_state"))
	}
	if n := actor.countTypeFrom(drawMark, "sync_state"); n != 0 {
		t.Fatalf("the acting player's draw drew %d sync_state repair(s); it must be applied on the first attempt", n)
	}
}

// TestAbilityRevealDoesNotStrandThePeer covers the other private-event family named in
// cambia-878: private_special_action_success, the ability reveal a 7/8, 9/T or King sends to the
// acting player alone. It plays real turns until an ability that reveals fires, then asserts two
// things: first, that the seq the private frame carried is one the peer also observed, which is
// the contract itself (dispatch() gates every inbound message against h.seq, so a seq the hub
// consumes has to be observable by every client) and the assertion that fails pre-fix; second,
// that the peer's next draw is applied without a sync_state bounce, end to end.
//
// The reveal is followed by a public player_special_action, so on this path the peer is dragged
// back level a frame later and only the seq assertion catches the defect; the failed-snap test
// above is where the dropped action is directly reproducible. Both are the same bug.
func TestAbilityRevealDoesNotStrandThePeer(t *testing.T) {
	g, hostID, p2ID, host, p2 := startTwoPlayerGame(t)

	const maxTurns = 20
	var reveal *wsEnvelope
	var peer *wsTestClient
	var peerID uuid.UUID

	for turn := 0; turn < maxTurns && reveal == nil; turn++ {
		actorID, actor, oppID, opp := actingPair(t, g, hostID, p2ID, host, p2)

		drawn := actor.sendAccepted("action_draw_stockpile", nil, "private_draw_stockpile", 5*time.Second)
		if drawn == nil {
			t.Fatalf("turn %d: the acting player's draw was never accepted", turn)
		}
		var drawPayload struct {
			Card struct {
				ID   string `json:"id"`
				Rank string `json:"rank"`
			} `json:"card"`
		}
		if err := json.Unmarshal(drawn.Payload, &drawPayload); err != nil {
			t.Fatalf("turn %d: decode private_draw_stockpile: %v", turn, err)
		}

		special := revealingSpecialFor(drawPayload.Card.Rank)
		turnMark := actor.mark()
		if ack := actor.sendAccepted("action_discard", map[string]interface{}{
			"card": map[string]interface{}{"id": drawPayload.Card.ID},
		}, "player_discard", 5*time.Second); ack == nil {
			t.Fatalf("turn %d: the acting player's discard was never accepted", turn)
		}

		if special == "" {
			// Either a plain card or a J/Q blind swap, neither of which reveals anything
			// privately. Skip any pending ability and take the next turn.
			if env, _ := actor.waitForTypeFrom(turnMark, "player_special_choice", 300*time.Millisecond); env != nil {
				if ack := actor.sendAccepted("action_special", map[string]interface{}{"special": "skip"},
					"game_player_turn", 5*time.Second); ack == nil {
					t.Fatalf("turn %d: skipping the ability never advanced the turn", turn)
				}
				continue
			}
			if env, _ := actor.waitForTypeFrom(turnMark, "game_player_turn", 5*time.Second); env == nil {
				t.Fatalf("turn %d: the discard never advanced the turn", turn)
			}
			continue
		}

		if env, _ := actor.waitForTypeFrom(turnMark, "player_special_choice", 5*time.Second); env == nil {
			t.Fatalf("turn %d: a rank %q discard never offered its ability", turn, drawPayload.Card.Rank)
		}

		body := map[string]interface{}{"special": special}
		switch special {
		case "peek_self":
			body["card1"] = handTarget(t, g, actorID, actorID, 0)
		case "peek_other":
			body["card1"] = handTarget(t, g, actorID, oppID, 0)
		case "swap_peek":
			body["card1"] = handTarget(t, g, actorID, actorID, 0)
			body["card2"] = handTarget(t, g, actorID, oppID, 0)
		}

		reveal = actor.sendAccepted("action_special", body, "private_special_action_success", 5*time.Second)
		if reveal == nil {
			t.Fatalf("turn %d: the %s ability never produced a private_special_action_success", turn, special)
		}
		if special == "swap_peek" {
			// The King's second step is still owed; declining it ends the turn.
			if ack := actor.sendAccepted("action_special", map[string]interface{}{"special": "skip"},
				"game_player_turn", 5*time.Second); ack == nil {
				t.Fatalf("turn %d: declining the King swap never advanced the turn", turn)
			}
		}
		peer, peerID = opp, oppID
	}

	if reveal == nil {
		t.Fatalf("no revealing ability (7/8, 9/T, King) came up in %d turns", maxTurns)
	}

	peerSeqs, _ := peer.seqsSeen()
	if _, ok := peerSeqs[reveal.Seq]; !ok {
		t.Fatalf("private_special_action_success carried seq %d, which the peer never observed: "+
			"a private frame consumed a sequence number the hub then gates the peer's next action against",
			reveal.Seq)
	}
	assertSeqStreamsAgree(t, host, p2)

	// The turn is the peer's now; its draw must land on the first attempt.
	actingID, _, _, _ := actingPair(t, g, hostID, p2ID, host, p2)
	if actingID != peerID {
		t.Fatalf("expected the turn to pass to %s after the ability, got %s", peerID, actingID)
	}
	peer.settle()
	drawMark := peer.mark()
	peer.send("action_draw_stockpile")
	if env, _ := peer.waitForTypeFrom(drawMark, "private_draw_stockpile", 5*time.Second); env == nil {
		t.Fatalf("the peer's draw was never applied after the ability reveal (sync_state repairs since: %d)",
			peer.countTypeFrom(drawMark, "sync_state"))
	}
	if n := peer.countTypeFrom(drawMark, "sync_state"); n != 0 {
		t.Fatalf("the peer's draw drew %d sync_state repair(s); it must be applied on the first attempt", n)
	}
}

// revealingSpecialFor maps a drawn card's rank to the special action that answers it with a
// private reveal. J/Q (blind swap) resolve entirely in public, so they map to "".
func revealingSpecialFor(rank string) string {
	switch rank {
	case "7", "8":
		return "peek_self"
	case "9", "T":
		return "peek_other"
	case "K":
		return "swap_peek"
	default:
		return ""
	}
}

// assertSeqStreamsAgree checks the hub's sequence contract from the wire: over the window both
// clients were connected for, neither may hold a seq the other never saw. Every seq the hub
// consumes gates every client's next message, so one that reaches a single connection strands
// everyone else.
func assertSeqStreamsAgree(t *testing.T, a, b *wsTestClient) {
	t.Helper()
	aSeqs, aMin := a.seqsSeen()
	bSeqs, bMin := b.seqsSeen()
	start := aMin
	if bMin > start {
		start = bMin
	}
	for seq := range aSeqs {
		if seq < start {
			continue
		}
		if _, ok := bSeqs[seq]; !ok {
			t.Fatalf("client A observed seq %d that client B never did (window from %d)", seq, start)
		}
	}
	for seq := range bSeqs {
		if seq < start {
			continue
		}
		if _, ok := aSeqs[seq]; !ok {
			t.Fatalf("client B observed seq %d that client A never did (window from %d)", seq, start)
		}
	}
}
