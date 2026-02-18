package models

// GameAction captures a player's in-game move
type GameAction struct {
	ActionType string                 `json:"action_type"`
	Payload    map[string]interface{} `json:"payload"`
}
