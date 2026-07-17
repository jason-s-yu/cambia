package engine

import "testing"

// TestHouseRulesValidate exercises HouseRules.Validate() directly (no cgo,
// no GameState construction) so the invalid cases can be tested without ever
// touching the array-index panic path they used to leave reachable
// (cambia-542 F3: NumPlayers had no upper-bound check anywhere prior to
// this).
func TestHouseRulesValidate(t *testing.T) {
	cases := []struct {
		name    string
		n       uint8
		wantErr bool
	}{
		{"zero_defaults_to_two", 0, false},
		{"one_below_minimum", 1, true},
		{"two_minimum", 2, false},
		{"seven_ok", 7, false},
		{"eight_maximum", MaxPlayers, false},
		{"nine_exceeds_maximum", MaxPlayers + 1, true},
		{"max_uint8", 255, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules()
			r.NumPlayers = c.n
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("NumPlayers=%d: expected an error, got nil", c.n)
			}
			if !c.wantErr && err != nil {
				t.Errorf("NumPlayers=%d: unexpected error: %v", c.n, err)
			}
		})
	}
}

// TestHouseRulesNumPlayersAccessorClamps is defense-in-depth coverage for
// the numPlayers() accessor's clamp (belt-and-suspenders behind Validate()):
// every [MaxPlayers]-sized array access in the engine flows through this
// accessor, so it must never hand back a value outside [2, MaxPlayers] no
// matter how HouseRules was constructed.
func TestHouseRulesNumPlayersAccessorClamps(t *testing.T) {
	cases := []struct {
		raw  uint8
		want uint8
	}{
		{0, 2},
		{1, 2},
		{2, 2},
		{8, 8},
		{9, MaxPlayers},
		{255, MaxPlayers},
	}
	for _, c := range cases {
		r := DefaultHouseRules()
		r.NumPlayers = c.raw
		if got := r.numPlayers(); got != c.want {
			t.Errorf("raw NumPlayers=%d: numPlayers() = %d, want %d", c.raw, got, c.want)
		}
	}
}
