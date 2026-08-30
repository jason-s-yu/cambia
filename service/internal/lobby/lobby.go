// internal/lobby/lobby.go
package lobby

import (
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// SystemHostUserID is the reserved HostUserID of a lobby whose host role belongs to the system
// rather than to any player: a matchmade lobby, from the moment the matchmaker seats a match in
// it (cambia-1087). Every host-gated action compares an authenticated user id against
// HostUserID, and no authenticated user can hold the nil UUID, so the sentinel refuses all of
// them without a second flag to keep in sync. Read through SystemHosted rather than compared
// inline, and never persisted: users has no row for it, so CreatorUserID is what the lobbies FK
// is satisfied with (see NewCambiaGameFromLobby).
var SystemHostUserID = uuid.Nil

// Lobby is pure state: rules, readiness, and lifecycle. WebSocket connections
// are managed by the hub package; lobbying is stateless here.
type Lobby struct {
	ID         uuid.UUID `json:"id"`
	HostUserID uuid.UUID `json:"hostUserID"`
	Type       string    `json:"type"`
	GameMode   string    `json:"gameMode"`

	// CreatorUserID is whoever called POST /lobby/create, stamped once and never reassigned:
	// unlike HostUserID it is not a role, so it survives both host migration and the handover to
	// the system host. It exists because games.lobby_id points at a lobbies row whose
	// host_user_id is NOT NULL with an FK to users, and a system-hosted lobby has no player in
	// that field to persist; the creator is a real authenticated user for the lobby's whole
	// lifetime and is the honest answer to "who opened this table" (cambia-1087). Not
	// serialized: no client reads it, and the create response shape is documented in
	// service/doc/rest_api.md.
	CreatorUserID uuid.UUID `json:"-"`

	// Name is an optional, host-supplied display name for the lobby. Empty when unset.
	Name string `json:"name"`

	// Users maps userID -> joined (true) or invited (false).
	Users map[uuid.UUID]bool `json:"-"`

	// ReadyStates holds userID -> bool for "is ready".
	ReadyStates map[uuid.UUID]bool `json:"-"`

	GameInstanceCreated bool `json:"-"`
	// GameID carries no omitempty: uuid.UUID is a fixed-size [16]byte array, a type
	// encoding/json's omitempty never treats as empty (unlike a slice, map, string, pointer, or
	// numeric/bool zero value), so a zero GameID would still serialize as
	// "gameId":"00000000-0000-0000-0000-000000000000" either way (cambia-907 L3).
	GameID uuid.UUID `json:"gameId"`
	InGame bool      `json:"inGame"`

	// CreatedAt is stamped once at construction and never mutated after (mutation in tests
	// aside, to simulate age). GET /lobby/list reads it to grant a fresh lobby a short creation
	// grace before the presence filter applies (cambia-887 F3), and it is json-tagged rather
	// than internal-only because, unlike Mu/Users/ReadyStates, it is a plain value with no
	// invariant a client could violate by reading it, and a lobby browser is a reasonable place
	// to show how long a table has been open.
	CreatedAt time.Time `json:"createdAt"`

	CountdownTimer *time.Timer `json:"-"`

	HouseRules game.HouseRules `json:"houseRules"`
	Circuit    game.Circuit    `json:"circuit"`

	// PresetID names the ruleset this lobby carries (presets.go), empty for a sheet that is
	// nobody's preset. Recorded rather than re-derived from the rules, because MATCHMAKING.md
	// 5.2 is one ruleset for every ranked queue: the six queue presets are byte-identical to
	// each other and differ only in player and round count, so matching a lobby's rules against
	// the preset list returns whichever one is listed first and a lobby created from H2H Rapid
	// reads back as H2H Quick (cambia-1123). Written where a preset is accepted
	// (CreateLobbyHandler, UpdateUnsafe, HandleMatchFormed) and cleared by the first rule edit
	// that departs from it.
	//
	// omitempty like QueueID and unlike GameID: it is a plain string, so a lobby on no preset
	// leaves the key out rather than naming one, and a client that reads it back can only get
	// an id the service actually recorded.
	PresetID string `json:"presetId,omitempty"`

	LobbySettings LobbySettings `json:"lobbySettings"`

	// Mode and QueueID/Searching serve matchmaking; Type alone (see above) carries
	// public/private, so there is no separate visibility field to hold in sync with it
	// (cambia-907 F1).
	Mode      string `json:"mode"` // "casual" or "ranked"
	QueueID   string `json:"queueID,omitempty"`
	Searching bool   `json:"searching"`

	// OnEmpty is called when all users have left.
	OnEmpty func(lobbyID uuid.UUID) `json:"-"`

	// joinOrder stamps each member with the order in which they first joined, and joinSeq is
	// the counter it draws from. Users is a map, so its iteration order is deliberately
	// randomised and nothing about it is stable enough to pick a successor host from; join
	// order is what makes migration on a host's departure deterministic (see nextHostUnsafe).
	joinOrder map[uuid.UUID]uint64
	joinSeq   uint64

	// Mu guards every mutable field above. Marked `json:"-"` because the struct doubles as the
	// REST payload for a lobby (POST /lobby/create, GET /lobby/list): an exported sync.Mutex
	// serialises as an empty "Mu" object, which no client reads and nothing should ship
	// (cambia-884).
	Mu sync.Mutex `json:"-"`
}

// LobbySettings holds lobby-level behavior settings.
type LobbySettings struct {
	AutoStart bool `json:"autoStart"`
}

// NewLobbyWithDefaults creates an ephemeral lobby with default house rules.
func NewLobbyWithDefaults(hostID uuid.UUID) *Lobby {
	lobbyID, _ := uuid.NewRandom()
	defaultHouseRules := game.DefaultHouseRules()
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
		ID:            lobbyID,
		HostUserID:    hostID,
		CreatorUserID: hostID,
		Type:          "private",
		GameMode:      "head_to_head",
		CreatedAt:     time.Now(),
		Users:         make(map[uuid.UUID]bool),
		ReadyStates:   make(map[uuid.UUID]bool),
		joinOrder:     make(map[uuid.UUID]uint64),
		HouseRules:    defaultHouseRules,
		Circuit:       defaultCircuit,
		LobbySettings: LobbySettings{
			AutoStart: true,
		},
		Mode: "casual",
	}
}

// AdoptSystemHostUnsafe hands the host role to the system, permanently: the lobby is now run by
// the queue that seated the match in it, and no player holds host powers over it. Assumes the
// lock is held.
//
// Called once, at match formation (handlers.HandleMatchFormed), not at lobby creation: a
// matchmaking lobby is a party before it is a match, and its party leader is the one who cancels
// the search and whose departure the lobby is emptied by. Nothing reverses it. The lobby that
// held a match keeps the system host for the rest of its life, so an aborted start cannot hand
// the role back to a player who could then edit a ranked match's rules (cambia-1087).
func (l *Lobby) AdoptSystemHostUnsafe() {
	if l.HostUserID == SystemHostUserID {
		return
	}
	log.Printf("Lobby %s: host role handed to the system; no player hosts a matchmade lobby.", l.ID)
	l.HostUserID = SystemHostUserID
}

// SystemHostedUnsafe reports whether the host role belongs to the system rather than a player.
// Assumes the lock is held.
func (l *Lobby) SystemHostedUnsafe() bool {
	return l.HostUserID == SystemHostUserID
}

// SystemHosted is the locking form of SystemHostedUnsafe.
func (l *Lobby) SystemHosted() bool {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	return l.SystemHostedUnsafe()
}

// JoinUser marks a user as joined (Users[userID] = true) and initialises their ready state.
// Acquires lock.
func (l *Lobby) JoinUser(userID uuid.UUID) {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	l.MarkJoinedUnsafe(userID)
}

// MarkJoinedUnsafe promotes a user to a fully joined member, initialises their ready state and
// stamps their join order the first time they join. Assumes the lock is held.
//
// Every path that grants membership goes through here - POST /lobby/{id}/join, the WebSocket
// upgrade, and JoinUser - so that join order is recorded once and stays recorded across a
// reconnect. Losing a socket does not release membership (cambia-807), so it must not restamp
// the order either: the successor host is the member who has been here longest.
func (l *Lobby) MarkJoinedUnsafe(userID uuid.UUID) {
	if l.Users == nil {
		l.Users = make(map[uuid.UUID]bool)
	}
	if l.ReadyStates == nil {
		l.ReadyStates = make(map[uuid.UUID]bool)
	}
	if l.joinOrder == nil {
		l.joinOrder = make(map[uuid.UUID]uint64)
	}
	l.Users[userID] = true
	if _, ok := l.ReadyStates[userID]; !ok {
		l.ReadyStates[userID] = false
	}
	if _, ok := l.joinOrder[userID]; !ok {
		l.joinSeq++
		l.joinOrder[userID] = l.joinSeq
	}
}

// RemoveUser releases a user's membership and calls OnEmpty when the last joined member is
// gone. Acquires lock. Returns false when the user held no membership, in which case nothing
// was changed and OnEmpty is not fired: a repeated leave must not run teardown twice.
//
// This is the deliberate-leave path only (POST /lobby/{id}/leave). A dropped WebSocket must
// never reach it: membership is what lets a member reconnect and what GET /lobby/active
// reports as resumable (cambia-783), so a transient disconnect has to leave it intact.
//
// Emptiness counts joined members, not map entries. A lobby whose remaining entries are all
// invitations nobody accepted has no one left to play: the host is auto-invited at creation
// (cambia-771) and hosts that never connect would otherwise pin a lobby open forever.
//
// A departing host hands the role on. HostUserID is what every host-gated action is checked
// against - rules, start, search, and the private-lobby WebSocket gate - so a host that left
// without migrating left the remaining members with a lobby nobody could change (cambia-835).
// A system-hosted lobby has no such role to migrate and is skipped explicitly: the departing
// user is an authenticated one and can never equal the sentinel, so the guard is a statement of
// the invariant rather than a live branch (cambia-1087).
func (l *Lobby) RemoveUser(userID uuid.UUID) bool {
	l.Mu.Lock()
	if _, present := l.Users[userID]; !present {
		l.Mu.Unlock()
		return false
	}
	delete(l.Users, userID)
	delete(l.ReadyStates, userID)
	delete(l.joinOrder, userID)
	isEmpty := l.JoinedCount() == 0
	if !isEmpty && !l.SystemHostedUnsafe() && l.HostUserID == userID {
		if next := l.nextHostUnsafe(); next != uuid.Nil {
			l.HostUserID = next
			log.Printf("Lobby %s: host %s left; host migrated to %s.", l.ID, userID, next)
		}
	}
	onEmpty := l.OnEmpty
	if l.CountdownTimer != nil {
		l.CancelCountdownUnsafe()
	}
	l.Mu.Unlock()

	if isEmpty && onEmpty != nil {
		log.Printf("Lobby %s is now empty. Triggering OnEmpty.", l.ID)
		onEmpty(l.ID)
	}
	return true
}

// nextHostUnsafe picks the successor host: the joined member who joined earliest, with the lower
// user id breaking a tie (entries that predate join-order tracking all carry sequence zero).
// Returns uuid.Nil when no joined member remains. Assumes the lock is held.
//
// Earliest joiner rather than, say, the lowest id: the role should land on whoever has been in
// the lobby longest, which is both predictable to the people in it and stable under reconnects,
// since a lost socket never restamps join order.
func (l *Lobby) nextHostUnsafe() uuid.UUID {
	var best uuid.UUID
	var bestSeq uint64
	for uid, joined := range l.Users {
		if !joined {
			continue
		}
		seq := l.joinOrder[uid]
		if best == uuid.Nil || seq < bestSeq || (seq == bestSeq && uid.String() < best.String()) {
			best, bestSeq = uid, seq
		}
	}
	return best
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
//
// A "presetId" key names a whole ruleset (see presets.go, cambia-1088) and is expanded before
// the field-by-field keys, so an explicit houseRules or settings object in the same message
// lands on top of the preset rather than under it.
//
// PresetID follows the sheet: an update that leaves the lobby playing exactly the preset it
// named records that id, and one that moves a rule the preset covers without naming a preset
// clears it (cambia-1123). Nothing derives it from the values afterwards, which is the whole
// point: every queue preset holds the same rules.
func (l *Lobby) UpdateUnsafe(rules map[string]interface{}) error {
	changed := false
	// sheetChanged tracks the fields a preset can express - house rules and lobby settings -
	// separately from changed, which also covers circuit scoring. A preset says nothing about
	// circuit settings (presets.go), so toggling one is not a departure from it.
	sheetChanged := false

	preset, err := l.resolvePresetUnsafe(rules)
	if err != nil {
		return err
	}

	tempHR := l.HouseRules
	if preset != nil {
		tempHR = preset.HouseRules
	}
	if hrData, ok := rules["houseRules"].(map[string]interface{}); ok {
		if err := tempHR.Update(hrData); err != nil {
			return err
		}
	}
	if tempHR != l.HouseRules {
		l.HouseRules = tempHR
		changed = true
		sheetChanged = true
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
	if preset != nil {
		tempLS = preset.Settings
	}
	if lsData, ok := rules["settings"].(map[string]interface{}); ok {
		if autoStart, ok := lsData["autoStart"].(bool); ok {
			tempLS.AutoStart = autoStart
		}
	}
	if tempLS != l.LobbySettings {
		l.LobbySettings = tempLS
		changed = true
		sheetChanged = true
	}

	// Record or release the preset the lobby carries. A named preset is only recorded when what
	// actually landed is still that preset: explicit fields in the same message land on top of
	// it, and a sheet that departed from the preset in the very call that named it is not on it.
	switch {
	case preset != nil:
		if l.HouseRules == preset.HouseRules && l.LobbySettings == preset.Settings {
			l.PresetID = preset.ID
		} else {
			l.PresetID = ""
		}
	case sheetChanged:
		l.PresetID = ""
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
