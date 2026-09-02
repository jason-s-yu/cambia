package gates

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

var weekdayNames = map[time.Weekday]string{
	time.Sunday:    "sun",
	time.Monday:    "mon",
	time.Tuesday:   "tue",
	time.Wednesday: "wed",
	time.Thursday:  "thu",
	time.Friday:    "fri",
	time.Saturday:  "sat",
}

func containsDay(days []string, wd time.Weekday) bool {
	name := weekdayNames[wd]
	for _, d := range days {
		if strings.EqualFold(strings.TrimSpace(d), name) {
			return true
		}
	}
	return false
}

// parseHHMM parses "HH:MM" into minutes since midnight. "24:00" (the D46
// example's full-day window terminator) parses as 1440.
func parseHHMM(s string) (int, error) {
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		return 0, fmt.Errorf("gates: %q is not HH:MM", s)
	}
	h, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, fmt.Errorf("gates: %q is not HH:MM: %w", s, err)
	}
	m, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, fmt.Errorf("gates: %q is not HH:MM: %w", s, err)
	}
	if h < 0 || h > 24 || m < 0 || m >= 60 || (h == 24 && m != 0) {
		return 0, fmt.Errorf("gates: %q is out of range", s)
	}
	return h*60 + m, nil
}

// inWindow reports whether now falls inside w. now is compared as an
// absolute instant; only the calendar date and weekday used to anchor the
// window's start and end are read in w.TZ's location, so a DST transition
// inside the window is handled by time.Date's own normalization against that
// location rather than by fixed-duration arithmetic.
func inWindow(now time.Time, w Window) bool {
	loc, err := time.LoadLocation(w.TZ)
	if err != nil {
		return false
	}
	fromMin, err := parseHHMM(w.From)
	if err != nil {
		return false
	}
	toMin, err := parseHHMM(w.To)
	if err != nil {
		return false
	}
	if toMin <= fromMin {
		toMin += 24 * 60
	}
	anchor := now.In(loc)
	for _, delta := range [2]int{-1, 0} {
		day := anchor.AddDate(0, 0, delta)
		if !containsDay(w.Days, day.Weekday()) {
			continue
		}
		y, mo, d := day.Date()
		start := time.Date(y, mo, d, fromMin/60, fromMin%60, 0, 0, loc)
		end := time.Date(y, mo, d, toMin/60, toMin%60, 0, 0, loc)
		if !now.Before(start) && now.Before(end) {
			return true
		}
	}
	return false
}

// nextWindowOpen finds the soonest instant at or after now that some window
// in windows opens, searching up to 8 calendar days ahead so a Days list
// naming only one weekday still resolves. It returns the on_breach policy of
// whichever window that soonest start belongs to, for the report's windows
// check when currently closed (D46: a breach's on_breach names the response
// to that gate closing).
func nextWindowOpen(now time.Time, windows []Window) (time.Time, OnBreach, bool) {
	var best time.Time
	var bestOB OnBreach
	found := false
	for _, w := range windows {
		loc, err := time.LoadLocation(w.TZ)
		if err != nil {
			continue
		}
		fromMin, err := parseHHMM(w.From)
		if err != nil {
			continue
		}
		anchor := now.In(loc)
		for delta := 0; delta <= 8; delta++ {
			day := anchor.AddDate(0, 0, delta)
			if !containsDay(w.Days, day.Weekday()) {
				continue
			}
			y, mo, d := day.Date()
			start := time.Date(y, mo, d, fromMin/60, fromMin%60, 0, 0, loc)
			if start.Before(now) {
				continue
			}
			if !found || start.Before(best) {
				best, bestOB, found = start, w.OnBreach, true
			}
			break
		}
	}
	return best, bestOB, found
}
