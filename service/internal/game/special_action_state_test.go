// internal/game/special_action_state_test.go
//
// The truth table for SpecialActionState.MustResolve (cambia-1099 Q7). The predicate used to be
// written out at three call sites - the skip refusal, the turn timeout's auto-resolve, and the
// Mandatory flag sync_state sends the client - and the three had to agree for a table not to wedge
// (cambia-1125): a client told an ability is declinable that the server then refuses to decline
// leaves the turn unplayable.
package game

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSpecialActionStateMustResolve(t *testing.T) {
	for _, tc := range []struct {
		name          string
		mandatory     bool
		rank          string
		firstStepDone bool
		want          bool
		why           string
	}{
		{"armed queen", true, "Q", false, true, "an ability the engine armed cannot be declined"},
		{"armed king, first step", true, "K", false, true, "the King's look is still the armed ability"},
		{"armed king, second step", true, "K", true, false, "the engine models declining the King swap as ActionKingSwapNo"},
		{"armed queen, stale first step", true, "Q", true, true, "FirstStepDone exempts the King and nothing else"},
		{"played queen", false, "Q", false, false, "an ability played from the drawn card is the player's choice"},
		{"played king, first step", false, "K", false, false, "same, and the rank does not make it mandatory"},
		{"played king, second step", false, "K", true, false, "still the player's choice"},
		{"cleared state", false, "", false, false, "a zeroed state answers false"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := SpecialActionState{
				Active:        true,
				Mandatory:     tc.mandatory,
				CardRank:      tc.rank,
				FirstStepDone: tc.firstStepDone,
			}
			assert.Equal(t, tc.want, s.MustResolve(), tc.why)
		})
	}
}
