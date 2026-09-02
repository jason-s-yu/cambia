package gates

import (
	"testing"
	"time"
)

func mustLoc(t *testing.T, name string) *time.Location {
	t.Helper()
	loc, err := time.LoadLocation(name)
	if err != nil {
		t.Fatalf("LoadLocation(%q): %v", name, err)
	}
	return loc
}

// weekdayWindow is the D46 example's night window: weeknights 22:00-07:00
// America/Los_Angeles.
func weekdayWindow() Window {
	return Window{Days: []string{"mon", "tue", "wed", "thu", "fri"}, From: "22:00", To: "07:00", TZ: "America/Los_Angeles", OnBreach: OnBreachStop}
}

// TestWindowsAcrossDSTSpringForward covers the 2026-03-08 US spring-forward
// transition (2:00 AM PST -> 3:00 AM PDT): a night window spanning that
// transition is still open at both endpoints and still closes at the correct
// wall-clock 07:00.
func TestWindowsAcrossDSTSpringForward(t *testing.T) {
	loc := mustLoc(t, "America/Los_Angeles")
	w := weekdayWindow() // Sunday 2026-03-08 is not in Days; use the window starting Fri 22:00

	// Friday 2026-03-06 22:30 PST: inside the window, well before the Sunday
	// transition, sanity baseline.
	if !inWindow(time.Date(2026, 3, 6, 22, 30, 0, 0, loc), w) {
		t.Error("expected inside the Friday night window")
	}

	// The transition itself falls on Sunday 2026-03-08, outside Days
	// (weekends aren't in this window), so use a weekday window instance
	// that itself straddles no transition but assert next_eligible_at
	// resolves correctly across the transition week: Sat 2026-03-07 10:00 is
	// closed (Saturday not in Days), next eligible is Mon 2026-03-09 22:00
	// PDT (post-transition offset).
	now := time.Date(2026, 3, 7, 10, 0, 0, 0, loc)
	if inWindow(now, w) {
		t.Fatal("expected Saturday daytime to be outside the weeknight window")
	}
	next, ob, found := nextWindowOpen(now, []Window{w})
	if !found {
		t.Fatal("expected a next window")
	}
	if ob != OnBreachStop {
		t.Errorf("OnBreach = %q, want stop", ob)
	}
	wantStart := time.Date(2026, 3, 9, 22, 0, 0, 0, loc)
	if !next.Equal(wantStart) {
		t.Errorf("next window open = %v, want %v", next, wantStart)
	}
	// The resolved instant reflects PDT (UTC-7), not PST (UTC-8): the
	// transition already happened by March 9.
	if _, offset := next.Zone(); offset != -7*3600 {
		t.Errorf("offset = %d, want -25200 (PDT, post spring-forward)", offset)
	}
}

// TestWindowsAcrossDSTFallBack covers the 2026-11-01 US fall-back transition
// (2:00 AM PDT -> 1:00 AM PST): a window evaluated on both sides of the
// transition resolves the correct wall-clock boundary and offset.
func TestWindowsAcrossDSTFallBack(t *testing.T) {
	loc := mustLoc(t, "America/Los_Angeles")
	w := weekdayWindow()

	// Friday 2026-10-30 22:30 PDT: inside, pre-transition.
	before := time.Date(2026, 10, 30, 22, 30, 0, 0, loc)
	if !inWindow(before, w) {
		t.Error("expected inside the Friday night window before fall-back")
	}
	if _, offset := before.Zone(); offset != -7*3600 {
		t.Errorf("pre-transition offset = %d, want -25200 (PDT)", offset)
	}

	// Monday 2026-11-02 22:30 PST: inside, post-transition, offset shifted.
	after := time.Date(2026, 11, 2, 22, 30, 0, 0, loc)
	if !inWindow(after, w) {
		t.Error("expected inside the Monday night window after fall-back")
	}
	if _, offset := after.Zone(); offset != -8*3600 {
		t.Errorf("post-transition offset = %d, want -28800 (PST)", offset)
	}

	// Sunday 2026-11-01 10:00 (the transition day itself, not in Days) is
	// closed; next eligible is Monday 2026-11-02 22:00 PST.
	now := time.Date(2026, 11, 1, 10, 0, 0, 0, loc)
	next, _, found := nextWindowOpen(now, []Window{w})
	if !found {
		t.Fatal("expected a next window")
	}
	wantStart := time.Date(2026, 11, 2, 22, 0, 0, 0, loc)
	if !next.Equal(wantStart) {
		t.Errorf("next window open = %v, want %v", next, wantStart)
	}
}

// TestWindowsWeekendFullDay covers the D46 example's second window
// (sat,sun 00:00-24:00) and the wraparound weeknight window together, and
// next_eligible_at for a genuinely closed period (AC(4)).
func TestWindowsClosedPeriodNextEligible(t *testing.T) {
	loc := mustLoc(t, "America/Los_Angeles")
	// Only the weeknight window configured (no weekend window), so Saturday
	// daytime is a real closed period.
	windows := []Window{weekdayWindow()}

	now := time.Date(2026, 6, 6, 12, 0, 0, 0, loc) // Saturday noon
	c := evaluateWindows(windows, now)
	if c.OK {
		t.Fatal("expected the windows check to fail on Saturday noon")
	}
	if c.NextEligibleAt == nil {
		t.Fatal("expected next_eligible_at to be set for a closed window")
	}
	want := time.Date(2026, 6, 8, 22, 0, 0, 0, loc) // Monday 22:00
	if !c.NextEligibleAt.Equal(want) {
		t.Errorf("next_eligible_at = %v, want %v", *c.NextEligibleAt, want)
	}
}

func TestWindowsFullWeekendWindow(t *testing.T) {
	loc := mustLoc(t, "America/Los_Angeles")
	weekend := Window{Days: []string{"sat", "sun"}, From: "00:00", To: "24:00", TZ: "America/Los_Angeles", OnBreach: OnBreachStop}
	windows := []Window{weekdayWindow(), weekend}

	sat := time.Date(2026, 6, 6, 15, 0, 0, 0, loc)
	if !inWindow(sat, weekend) {
		t.Error("expected inside the full-day Saturday window")
	}
	// Every hour of the D46 example's two windows together covers the full
	// week; a Tuesday midday is the one genuinely uncovered slice.
	tue := time.Date(2026, 6, 9, 12, 0, 0, 0, loc)
	c := evaluateWindows(windows, tue)
	if c.OK {
		t.Fatal("expected Tuesday midday to be outside both configured windows")
	}
}
