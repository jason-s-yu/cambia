// internal/hub/panic_isolation_test.go
//
// Panic isolation (cambia-946). A panic inside one hub's message handling used to unwind past
// Run and kill the process, so one table's bug ended every concurrent game on the server. The
// hub now recovers at its event loop: the affected hub tells its own clients the session is over
// and dissolves, every other hub keeps serving, and the game's mutex is released on the way out
// (CambiaGame's public entry points unlock by defer), so the wrecked game is not left locked.
package hub

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// panicGame builds a started 2-player game whose seat mapping for playerID points past the
// engine's fixed 8-slot Players array, so the next action from that player panics inside
// HandlePlayerAction with the game's mutex held: the shape of the cambia-946 production panic,
// injected without depending on the bug that is now fixed.
func panicGame(t *testing.T, h *Hub, playerID uuid.UUID) *game.CambiaGame {
	t.Helper()
	g := game.NewCambiaGame()
	g.Emitter = h
	g.HouseRules.TurnTimerSec = 0
	g.TurnDuration = 0

	for i, id := range []uuid.UUID{playerID, uuid.New()} {
		g.AddPlayer(&models.Player{
			ID:        id,
			Connected: true,
			User:      &models.User{ID: id, Username: "P" + string(rune('A'+i))},
		})
	}
	g.BeginPreGame()
	g.StartGame()
	require.True(t, g.Started, "game should have started")

	g.PlayerToEngine[playerID] = 200
	return g
}

// TestHubPanicDissolvesOnlyItsOwnHub drives a panic through one hub's game handler and asserts
// the process survives, that hub reports and dissolves, and a second hub still answers.
func TestHubPanicDissolvesOnlyItsOwnHub(t *testing.T) {
	victimUser := uuid.New()
	victimLobby := lobby.NewLobbyWithDefaults(victimUser)
	victimLobby.JoinUser(victimUser)
	victim := NewHub(victimLobby)

	dissolved := make(chan uuid.UUID, 1)
	victim.OnDissolve = func(hubID uuid.UUID) { dissolved <- hubID }

	bystanderUser := uuid.New()
	bystanderLobby := lobby.NewLobbyWithDefaults(bystanderUser)
	bystanderLobby.JoinUser(bystanderUser)
	bystander := NewHub(bystanderLobby)

	// Phase and Game belong to the Run goroutine, so the in-game hub is wired before it starts.
	g := panicGame(t, victim, victimUser)
	victim.Game = g
	victim.Phase = PhaseInGame

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go victim.Run(ctx)
	go bystander.Run(ctx)
	waitAlive(t, victim, true)
	waitAlive(t, bystander, true)
	defer bystander.Shutdown()

	victimConn := newFakeConn(victimUser, "victim")
	victim.Join(victimConn)
	require.NotNil(t, waitEnvelope(t, victimConn, "lobby_state", time.Second))

	bystanderConn := newFakeConn(bystanderUser, "bystander")
	bystander.Join(bystanderConn)
	require.NotNil(t, waitEnvelope(t, bystanderConn, "lobby_state", time.Second))

	// A snap skips the turn check, so it reaches the corrupted seat mapping directly.
	body, err := json.Marshal(map[string]interface{}{"card": map[string]interface{}{"id": uuid.New().String()}})
	require.NoError(t, err)
	victim.Incoming() <- ClientMsg{
		UserID:  victimUser,
		Type:    "action_snap",
		LastSeq: ^uint64(0),
		Body:    body,
	}

	// The panicking hub tells its own clients and dissolves.
	fatal := waitEnvelope(t, victimConn, "error", 2*time.Second)
	require.NotNil(t, fatal, "the affected hub must send its clients a terminal error")
	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(fatal.Payload, &payload))
	assert.Equal(t, "hub_fatal", payload["code"], "the terminal error must be marked fatal")

	select {
	case id := <-dissolved:
		assert.Equal(t, victim.ID, id, "the affected hub must dissolve through OnDissolve")
	case <-time.After(2 * time.Second):
		t.Fatal("the affected hub never dissolved")
	}
	waitAlive(t, victim, false)

	// The recovery left no lock held on the wrecked game: a call that takes the game's mutex
	// still returns.
	done := make(chan bool, 1)
	go func() { done <- g.HasPlayer(victimUser) }()
	select {
	case has := <-done:
		assert.True(t, has, "the wrecked game must still answer under its own lock")
	case <-time.After(2 * time.Second):
		t.Fatal("the game mutex was left held by the recovered panic")
	}

	// Every other hub keeps serving: this is the whole point of the boundary.
	assert.True(t, bystander.Alive(), "an unrelated hub must survive another hub's panic")
	bystander.Incoming() <- ClientMsg{UserID: bystanderUser, Type: "chat", LastSeq: ^uint64(0),
		Body: []byte(`{"msg":"still here"}`)}
	chat := waitEnvelope(t, bystanderConn, "chat", 2*time.Second)
	require.NotNil(t, chat, "an unrelated hub must still answer after another hub panicked")
}
