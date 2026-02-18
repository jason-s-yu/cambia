// internal/game/rules.go
package game

import "fmt"

// HouseRules defines optional game rules that can modify standard play.
type HouseRules struct {
	AllowDrawFromDiscardPile bool `json:"allowDrawFromDiscardPile"` // Allow players to draw from the discard pile instead of the stockpile.
	AllowReplaceAbilities    bool `json:"allowReplaceAbilities"`    // Allow cards discarded via replacement to trigger their special abilities.
	SnapRace                 bool `json:"snapRace"`                 // Only the first player to successfully snap gets the benefit; others are penalized.
	ForfeitOnDisconnect      bool `json:"forfeitOnDisconnect"`      // If a player disconnects, their game is forfeited. If false, they can rejoin.
	PenaltyDrawCount         int  `json:"penaltyDrawCount"`         // Number of cards to draw as penalty for an invalid snap.
	AutoKickTurnCount        int  `json:"autoKickTurnCount"`        // Number of consecutive turns a player can time out before being kicked (0 disables).
	TurnTimerSec             int  `json:"turnTimerSec"`             // Duration (in seconds) for each player's turn (0 disables timer).
}

// Update applies changes from a map to the HouseRules struct.
// It validates input types and ranges where applicable.
func (rules *HouseRules) Update(newRules map[string]interface{}) error {
	var ok bool
	var err error // Declare error variable

	// Helper function to handle type assertion and assignment for booleans.
	assignBool := func(field *bool, key string) error {
		if val, exists := newRules[key]; exists && val != nil {
			*field, ok = val.(bool)
			if !ok {
				return fmt.Errorf("invalid type for %s, expected boolean", key)
			}
		}
		return nil
	}

	// Helper function to handle type assertion and assignment for integers.
	assignInt := func(field *int, key string, minVal int, validationMsg string) error {
		if val, exists := newRules[key]; exists && val != nil {
			// JSON numbers are often float64, handle conversion gracefully.
			var floatVal float64
			floatVal, ok = val.(float64)
			if !ok {
				// Try int if float64 fails.
				var intVal int
				intVal, ok = val.(int)
				if !ok {
					return fmt.Errorf("invalid type for %s, expected number", key)
				}
				*field = intVal
			} else {
				// Check if float has fractional part before converting.
				if floatVal != float64(int(floatVal)) {
					return fmt.Errorf("invalid value for %s, must be a whole number", key)
				}
				*field = int(floatVal)
			}

			if *field < minVal {
				return fmt.Errorf("%s", validationMsg)
			}
		}
		return nil
	}

	// Apply updates using helpers.
	if err = assignBool(&rules.AllowDrawFromDiscardPile, "allowDrawFromDiscardPile"); err != nil {
		return err
	}
	if err = assignBool(&rules.AllowReplaceAbilities, "allowReplaceAbilities"); err != nil {
		return err
	}
	if err = assignBool(&rules.SnapRace, "snapRace"); err != nil {
		return err
	}
	if err = assignBool(&rules.ForfeitOnDisconnect, "forfeitOnDisconnect"); err != nil {
		return err
	}
	if err = assignInt(&rules.PenaltyDrawCount, "penaltyDrawCount", 0, "penaltyDrawCount must be non-negative"); err != nil {
		return err
	}
	if err = assignInt(&rules.AutoKickTurnCount, "autoKickTurnCount", 0, "autoKickTurnCount must be non-negative"); err != nil {
		return err
	}
	if err = assignInt(&rules.TurnTimerSec, "turnTimerSec", 0, "turnTimerSec must be non-negative"); err != nil {
		return err
	}

	return nil
}

// ParseRules is deprecated as HouseRules.Update provides the same functionality directly.
// Keeping for potential compatibility but recommend direct use of Update.
// Deprecated: Use the Update method directly on a HouseRules instance.
func ParseRules(rules map[string]interface{}, current HouseRules) (HouseRules, error) {
	houseRules := current
	err := houseRules.Update(rules)
	return houseRules, err
}
