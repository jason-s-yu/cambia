// internal/lobby/presets.go
//
// Ruleset presets: one named, whole-sheet rule configuration a custom lobby can adopt instead
// of the host setting fourteen house rules by hand (cambia-1088).
//
// There is exactly one definition of each preset and it lives here. The New lobby dialog reads
// it over GET /lobby/presets, the lobby rule sheet reads the same list to fill itself and to
// tell a departed sheet from an untouched one, POST /lobby/create expands a preset id through
// UpdateUnsafe, and the games a queue produces are built from the same values
// (handlers.NewCambiaGameFromLobby). A second copy of any of it - in the client, in the
// handler, or in the game factory - is what makes a preset a claim the service does not keep.
package lobby

import (
	"fmt"

	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// DefaultPresetID names the preset a lobby created with no preset already plays: the house
// rules NewLobbyWithDefaults builds it with. It is first in Presets() and is the one the
// dialog opens on.
const DefaultPresetID = "default"

// Preset is a named ruleset plus the lobby shape it implies.
//
// GameMode is empty for a preset that fixes no player count (the default one), and the client
// leaves its game-mode control free in that case. Rounds is carried for display only: a custom
// lobby has no round count to set, because a multi-round match's TotalRounds is hub state taken
// from the queue config at search time, so a queue preset's rounds figure describes the queue
// and not the lobby the preset was applied to (cambia-466 owns the round lifecycle).
//
// Circuit settings are deliberately absent. A preset cannot express a round count, so filling
// the sheet with one would silently reset a host's circuit configuration to say something the
// preset never meant.
type Preset struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	GameMode    string          `json:"gameMode"`
	Players     int             `json:"players"`
	Rounds      int             `json:"rounds"`
	Ranked      bool            `json:"ranked"`
	HouseRules  game.HouseRules `json:"houseRules"`
	Settings    LobbySettings   `json:"settings"`
}

// gameModeForPlayers maps a queue's player count onto the lobby game mode that seats it, the
// same mapping CreateLobbyHandler applies to a matchmaking lobby. An unsupported count yields
// an empty mode, which drops the queue from the preset list rather than offering a preset no
// lobby can be created from.
func gameModeForPlayers(players int) string {
	switch players {
	case 2:
		return "head_to_head"
	case 4:
		return "group_of_4"
	default:
		return ""
	}
}

// rankedQueueHouseRules is the fixed rule configuration every ranked queue plays, per
// MATCHMAKING.md 5.2. The four departures from the defaults are the T1C (turn-1 Cambia) fix
// and what it needs to work: an unlocked caller's hand turns the call from a declaration of
// victory into a gamble the opponents' final turns can punish, and the discard pile, replaced
// abilities and snap race are the public-information tools that make punishing it a skill.
// The reconnect grace is the queue's own (MATCHMAKING.md 8, cambia-955); everything else is
// the default.
func rankedQueueHouseRules(cfg matchmaking.QueueConfig) game.HouseRules {
	hr := game.DefaultHouseRules()
	hr.AllowDrawFromDiscardPile = true
	hr.AllowReplaceAbilities = true
	hr.SnapRace = true
	hr.LockCallerHand = false
	hr.DisconnectGraceSec = cfg.DisconnectGraceSec
	return hr
}

// defaultPreset is the ruleset a lobby starts on: exactly what NewLobbyWithDefaults builds,
// so selecting it in the dialog and creating a lobby without a preset produce the same lobby.
func defaultPreset() Preset {
	return Preset{
		ID:          DefaultPresetID,
		Name:        "Default",
		Description: "The rules a new lobby starts with.",
		GameMode:    "",
		Rounds:      1,
		HouseRules:  game.DefaultHouseRules(),
		Settings:    LobbySettings{AutoStart: true},
	}
}

// queuePreset is the ruleset a matchmaking queue plays, offered to a custom lobby under the
// queue's own display name.
func queuePreset(cfg matchmaking.QueueConfig) Preset {
	return Preset{
		ID:          cfg.QueueID,
		Name:        cfg.DisplayName,
		Description: "Ranked queue rules. A custom lobby plays a single round.",
		GameMode:    gameModeForPlayers(cfg.Players),
		Players:     cfg.Players,
		Rounds:      cfg.Rounds,
		Ranked:      cfg.Ranked,
		HouseRules:  rankedQueueHouseRules(cfg),
		Settings:    LobbySettings{AutoStart: true},
	}
}

// Presets returns every selectable ruleset: the default first, then one per configured queue
// in the queue list's own order, so the dialog's dropdown and the dashboard's queue cards
// agree on sequence and naming.
func Presets() []Preset {
	ids := matchmaking.OrderedQueueIDs()
	out := make([]Preset, 0, len(ids)+1)
	out = append(out, defaultPreset())
	for _, id := range ids {
		cfg := matchmaking.QueueConfigs[id]
		if gameModeForPlayers(cfg.Players) == "" {
			continue
		}
		out = append(out, queuePreset(cfg))
	}
	return out
}

// resolvePresetUnsafe reads an optional "presetId" key out of an update and returns the ruleset
// it names, or nil when the update carries no preset. Assumes the lock is held.
//
// A ranked or matchmade lobby refuses one outright: its rules come from the queue it entered,
// which is why update_rules refuses that lobby entirely (hub.go, cambia-966). The rule is
// enforced here as well so it holds for every caller of UpdateUnsafe, CreateLobbyHandler
// included, and not only on the WebSocket path that happens to gate first.
func (l *Lobby) resolvePresetUnsafe(rules map[string]interface{}) (*Preset, error) {
	raw, present := rules["presetId"]
	if !present || raw == nil {
		return nil, nil
	}
	id, ok := raw.(string)
	if !ok {
		return nil, fmt.Errorf("invalid type for presetId, expected string")
	}
	if id == "" {
		return nil, nil
	}
	if l.Type == "matchmaking" || l.Mode == "ranked" {
		return nil, fmt.Errorf("ruleset presets are not accepted for a ranked matchmaking lobby")
	}
	p, known := GetPreset(id)
	if !known {
		return nil, fmt.Errorf("unknown ruleset preset %q", id)
	}
	return &p, nil
}

// GetPreset looks a preset up by id. The queue presets are keyed by queue id, so a queue id is
// a valid preset id and NewCambiaGameFromLobby resolves a queued lobby's rules through here.
func GetPreset(id string) (Preset, bool) {
	if id == DefaultPresetID {
		return defaultPreset(), true
	}
	cfg, known := matchmaking.GetQueueConfig(id)
	if !known || gameModeForPlayers(cfg.Players) == "" {
		return Preset{}, false
	}
	return queuePreset(cfg), true
}
