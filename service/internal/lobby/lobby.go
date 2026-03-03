// internal/lobby/lobby.go
package lobby

import (
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// Lobby is pure state: rules, readiness, and lifecycle. WebSocket connections
// are managed by the hub package; lobbying is stateless here.
type Lobby struct {
	ID         uuid.UUID `json:"id"`
	HostUserID uuid.UUID `json:"hostUserID"`
	Type       string    `json:"type"`
	GameMode   string    `json:"gameMode"`

	// Users maps userID -> joined (true) or invited (false).
	Users map[uuid.UUID]bool `json:"-"`

	// ReadyStates holds userID -> bool for "is ready".
	ReadyStates map[uuid.UUID]bool `json:"-"`

	GameInstanceCreated bool      `json:"-"`
	GameID              uuid.UUID `json:"gameId,omitempty"`
	InGame              bool      `json:"inGame"`

	CountdownTimer *time.Timer `json:"-"`

	HouseRules game.HouseRules `json:"houseRules"`
	Circuit    game.Circuit    `json:"circuit"`

	LobbySettings LobbySettings `json:"lobbySettings"`

	Visibility string `json:"visibility"` // "private" or "public"
	Mode       string `json:"mode"`       // "casual" or "ranked"
	QueueID    string `json:"queueID,omitempty"`
	Searching  bool   `json:"searching"`

	// OnEmpty is called when all users have left.
	OnEmpty func(lobbyID uuid.UUID) `json:"-"`

	Mu sync.Mutex
}

// LobbySettings holds lobby-level behavior settings.
type LobbySettings struct {
	AutoStart bool `json:"autoStart"`
}

// NewLobbyWithDefaults creates an ephemeral lobby with default house rules.
func NewLobbyWithDefaults(hostID uuid.UUID) *Lobby {
	lobbyID, _ := uuid.NewRandom()
	defaultHouseRules := game.HouseRules{
		AllowDrawFromDiscardPile: false,
		AllowReplaceAbilities:    false,
		SnapRace:                 false,
		ForfeitOnDisconnect:      true,
		PenaltyDrawCount:         2,
		AutoKickTurnCount:        3,
		TurnTimerSec:             15,
	}
	defaultCircuit := game.Circuit{
		Enabled: false,
		Rules: game.CircuitRules{
			TargetScore:            100,
			WinBonus:               -1,
			FalseCambiaPenalty:     1,
			FreezeUserOnDisconnect: true,
		},
	}

	return &Lobby{
		ID:          lobbyID,
		HostUserID:  hostID,
		Type:        "private",
		GameMode:    "head_to_head",
		Users:       make(map[uuid.UUID]bool),
		ReadyStates: make(map[uuid.UUID]bool),
		HouseRules:  defaultHouseRules,
		Circuit:     defaultCircuit,
		LobbySettings: LobbySettings{
			AutoStart: true,
		},
		Visibility: "private",
		Mode:       "casual",
	}
}

// JoinUser marks a user as joined (Users[userID] = true) and initialises their ready state.
// Acquires lock.
func (l *Lobby) JoinUser(userID uuid.UUID) {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	l.Users[userID] = true
	if _, ok := l.ReadyStates[userID]; !ok {
		l.ReadyStates[userID] = false
	}
}

// RemoveUser removes a user's state and calls OnEmpty if the lobby is empty. Acquires lock.
func (l *Lobby) RemoveUser(userID uuid.UUID) {
	l.Mu.Lock()
	delete(l.Users, userID)
	delete(l.ReadyStates, userID)
	isEmpty := len(l.Users) == 0
	onEmpty := l.OnEmpty
	if l.CountdownTimer != nil {
		l.CancelCountdownUnsafe()
	}
	l.Mu.Unlock()

	if isEmpty && onEmpty != nil {
		log.Printf("Lobby %s is now empty. Triggering OnEmpty.", l.ID)
		onEmpty(l.ID)
	}
}

// InviteUser marks a user as invited (Users[userID] = false) if not already present.
// Assumes lock is held.
func (l *Lobby) InviteUser(userID uuid.UUID) {
	l.inviteUserUnsafe(userID)
}

func (l *Lobby) inviteUserUnsafe(userID uuid.UUID) {
	if _, exists := l.Users[userID]; !exists {
		l.Users[userID] = false
		log.Printf("Lobby %s: User %s invited.", l.ID, userID)
	}
}

// MarkUserReadyUnsafe sets a user's ready state to true.
// Returns true if all joined users are now ready (countdown should start).
// Assumes lock is held.
func (l *Lobby) MarkUserReadyUnsafe(userID uuid.UUID) bool {
	joined, ok := l.Users[userID]
	if !ok || !joined {
		log.Printf("Lobby %s: Cannot mark non-joined user %s as ready.", l.ID, userID)
		return false
	}
	if l.ReadyStates[userID] {
		return false // already ready
	}
	l.ReadyStates[userID] = true
	log.Printf("Lobby %s: User %s marked READY.", l.ID, userID)

	allReady := l.AreAllReadyUnsafe()
	return allReady && l.LobbySettings.AutoStart && !l.InGame && l.JoinedCount() >= 2
}

// MarkUserReady calls the unsafe version. Assumes lock is held.
func (l *Lobby) MarkUserReady(userID uuid.UUID) bool {
	return l.MarkUserReadyUnsafe(userID)
}

// MarkUserUnreadyUnsafe sets a user's ready state to false and cancels countdown.
// Assumes lock is held.
func (l *Lobby) MarkUserUnreadyUnsafe(userID uuid.UUID) {
	joined, ok := l.Users[userID]
	if !ok || !joined {
		return
	}
	if !l.ReadyStates[userID] {
		return // already unready
	}
	l.ReadyStates[userID] = false
	log.Printf("Lobby %s: User %s marked UNREADY.", l.ID, userID)
	l.CancelCountdownUnsafe()
}

// MarkUserUnready calls the unsafe version. Assumes lock is held.
func (l *Lobby) MarkUserUnready(userID uuid.UUID) {
	l.MarkUserUnreadyUnsafe(userID)
}

// AreAllReadyUnsafe returns true if all joined users are ready and there are at least 2.
// Assumes lock is held.
func (l *Lobby) AreAllReadyUnsafe() bool {
	joined := 0
	for userID, isJoined := range l.Users {
		if !isJoined {
			continue
		}
		joined++
		if !l.ReadyStates[userID] {
			return false
		}
	}
	return joined >= 2
}

// AreAllReady acquires the lock and calls the unsafe version.
func (l *Lobby) AreAllReady() bool {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	return l.AreAllReadyUnsafe()
}

// JoinedCount returns the number of fully-joined users. Assumes lock is held.
func (l *Lobby) JoinedCount() int {
	count := 0
	for _, joined := range l.Users {
		if joined {
			count++
		}
	}
	return count
}

// JoinedUsers returns a slice of UUIDs for all fully-joined users.
// Acquires lock.
func (l *Lobby) JoinedUsers() []uuid.UUID {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	out := make([]uuid.UUID, 0, len(l.Users))
	for uid, joined := range l.Users {
		if joined {
			out = append(out, uid)
		}
	}
	return out
}

// StartCountdownUnsafe begins a countdown timer. Returns false if already in-game or timer running.
// Assumes lock is held.
func (l *Lobby) StartCountdownUnsafe(seconds int, callback func(*Lobby)) bool {
	if l.InGame || l.CountdownTimer != nil {
		return false
	}
	if l.JoinedCount() < 2 {
		return false
	}
	log.Printf("Lobby %s: Starting %d second countdown.", l.ID, seconds)
	var timer *time.Timer
	timer = time.AfterFunc(time.Duration(seconds)*time.Second, func() {
		l.Mu.Lock()
		if l.CountdownTimer == timer {
			l.CountdownTimer = nil
			l.Mu.Unlock()
			callback(l)
		} else {
			l.Mu.Unlock()
		}
	})
	l.CountdownTimer = timer
	return true
}

// StartCountdown calls the unsafe version. Assumes lock is held.
func (l *Lobby) StartCountdown(seconds int, callback func(*Lobby)) bool {
	return l.StartCountdownUnsafe(seconds, callback)
}

// CancelCountdownUnsafe stops any active countdown. Assumes lock is held.
func (l *Lobby) CancelCountdownUnsafe() {
	if l.CountdownTimer != nil {
		l.CountdownTimer.Stop()
		l.CountdownTimer = nil
	}
}

// CancelCountdown calls the unsafe version. Assumes lock is held.
func (l *Lobby) CancelCountdown() {
	l.CancelCountdownUnsafe()
}

// GetLobbyStatusPayloadUnsafe returns a summary of joined users and their ready states.
// Assumes lock is held.
func (l *Lobby) GetLobbyStatusPayloadUnsafe() map[string]interface{} {
	users := []map[string]interface{}{}
	for userID, joined := range l.Users {
		if !joined {
			continue
		}
		users = append(users, map[string]interface{}{
			"id":       userID.String(),
			"is_host":  userID == l.HostUserID,
			"is_ready": l.ReadyStates[userID],
		})
	}
	return map[string]interface{}{"users": users}
}

// UpdateUnsafe applies partial settings updates. Assumes lock is held.
func (l *Lobby) UpdateUnsafe(rules map[string]interface{}) error {
	changed := false

	tempHR := l.HouseRules
	if hrData, ok := rules["houseRules"].(map[string]interface{}); ok {
		if err := tempHR.Update(hrData); err != nil {
			return err
		}
		if tempHR != l.HouseRules {
			l.HouseRules = tempHR
			changed = true
		}
	}

	tempCircuit := l.Circuit
	madeChange := false
	if cData, ok := rules["circuit"].(map[string]interface{}); ok {
		if enabled, ok := cData["enabled"].(bool); ok && tempCircuit.Enabled != enabled {
			tempCircuit.Enabled = enabled
			madeChange = true
		}
		if cRules, ok := cData["rules"].(map[string]interface{}); ok {
			if ts, ok := cRules["targetScore"].(float64); ok && tempCircuit.Rules.TargetScore != int(ts) {
				tempCircuit.Rules.TargetScore = int(ts)
				madeChange = true
			}
			if wb, ok := cRules["winBonus"].(float64); ok && tempCircuit.Rules.WinBonus != int(wb) {
				tempCircuit.Rules.WinBonus = int(wb)
				madeChange = true
			}
			if fcp, ok := cRules["falseCambiaPenalty"].(float64); ok && tempCircuit.Rules.FalseCambiaPenalty != int(fcp) {
				tempCircuit.Rules.FalseCambiaPenalty = int(fcp)
				madeChange = true
			}
			if fud, ok := cRules["freezeUserOnDisconnect"].(bool); ok && tempCircuit.Rules.FreezeUserOnDisconnect != fud {
				tempCircuit.Rules.FreezeUserOnDisconnect = fud
				madeChange = true
			}
		}
		if madeChange {
			l.Circuit = tempCircuit
			changed = true
		}
	}

	tempLS := l.LobbySettings
	if lsData, ok := rules["settings"].(map[string]interface{}); ok {
		if autoStart, ok := lsData["autoStart"].(bool); ok && tempLS.AutoStart != autoStart {
			tempLS.AutoStart = autoStart
			l.LobbySettings = tempLS
			changed = true
		}
	}

	if changed {
		log.Printf("Lobby %s: Rules updated.", l.ID)
	}
	return nil
}

// Update calls UpdateUnsafe. Assumes lock is held.
func (l *Lobby) Update(rules map[string]interface{}) error {
	return l.UpdateUnsafe(rules)
}
