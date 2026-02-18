// internal/game/utils.go
package game

import (
	"encoding/json"
	"log"
)

// convertEventToBytes marshals a GameEvent into JSON bytes.
// Logs a warning and returns empty JSON "{}" on marshalling error.
func convertEventToBytes(ev GameEvent) []byte {
	data, err := json.Marshal(ev)
	if err != nil {
		// Log detailed error including event type if possible
		log.Printf("WARNING: Failed to marshal GameEvent type %s: %v", ev.Type, err)
		return []byte("{}") // Return empty JSON to prevent downstream crashes
	}
	return data
}
