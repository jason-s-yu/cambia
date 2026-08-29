// internal/database/party_rating_test.go
package database

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/rating"
)

// TestLoadPartyRatingsNoRatingRowUsesDefaults is the cambia-1041 no-rating-row regression:
// LoadPartyRatings must never return a zero-value Elo/RD for a player it can't resolve a row
// for - a fresh random user id, standing in for a player who has never queued before. This holds
// whether or not a dev Postgres is reachable in this environment (DB == nil takes the same
// defaults branch as a resolved DB returning no row), so the test runs unconditionally rather
// than skip on a missing DB.
func TestLoadPartyRatingsNoRatingRowUsesDefaults(t *testing.T) {
	strangers := []uuid.UUID{uuid.New(), uuid.New()}
	got := LoadPartyRatings(context.Background(), strangers, rating.Mode1v1)
	if len(got) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(got))
	}
	for i, pr := range got {
		if pr.Elo != rating.DefaultMu {
			t.Errorf("entry %d: Elo = %v, want pool default %v", i, pr.Elo, rating.DefaultMu)
		}
		if pr.RD != rating.DefaultPhi {
			t.Errorf("entry %d: RD = %v, want pool default %v", i, pr.RD, rating.DefaultPhi)
		}
		if pr.OpenSkillMu != rating.DefaultOpenSkillMu {
			t.Errorf("entry %d: OpenSkillMu = %v, want pool default %v", i, pr.OpenSkillMu, rating.DefaultOpenSkillMu)
		}
	}
}

// TestLoadPartyRatingsUnsupportedModeUsesDefaults pins the mode=="" branch (rating.ModeForPlayerCount
// returns "" for a roster size no queue uses): every player degrades to defaults rather than
// PoolFields' own 1v1 fallback silently substituting a pool that was never selected.
func TestLoadPartyRatingsUnsupportedModeUsesDefaults(t *testing.T) {
	got := LoadPartyRatings(context.Background(), []uuid.UUID{uuid.New()}, rating.RatingMode(""))
	if len(got) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(got))
	}
	if got[0].Elo != rating.DefaultMu || got[0].RD != rating.DefaultPhi {
		t.Fatalf("expected pool defaults for an unsupported mode, got %+v", got[0])
	}
}

// TestLoadPartyRatingsReadsRealRow proves the pool-aware read path end to end against a live
// Postgres: a user with a real elo_4p/phi_4p row must come back with those exact values, not the
// pool default, and the pool the mode selects (4p) must not bleed into another pool's fields
// (elo_1v1 stays untouched by this test's UPDATE).
func TestLoadPartyRatingsReadsRealRow(t *testing.T) {
	setupGameTest(t)
	u := createGameTestUser(t, "party-rating-4p-"+uuid.New().String()[:8])

	_, err := DB.Exec(context.Background(),
		`UPDATE users SET elo_4p = $1, phi_4p = $2, open_skill_mu = $3 WHERE id = $4`,
		1812, 48.5, 31.25, u.ID,
	)
	if err != nil {
		t.Fatalf("failed to seed rating columns: %v", err)
	}

	got := LoadPartyRatings(context.Background(), []uuid.UUID{u.ID}, rating.Mode4p)
	if len(got) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(got))
	}
	pr := got[0]
	if pr.Elo != 1812 {
		t.Errorf("Elo = %v, want 1812", pr.Elo)
	}
	if pr.RD != 48.5 {
		t.Errorf("RD = %v, want 48.5", pr.RD)
	}
	if pr.OpenSkillMu != 31.25 {
		t.Errorf("OpenSkillMu = %v, want 31.25", pr.OpenSkillMu)
	}
}

// TestAggregatePartyGlicko covers the reduction QueuedLobby.AvgRating/MaxRD are built from: the
// average Elo across the party and the worst (highest) RD, plus the empty-party default that
// keeps a caller with no ratings from comparing against a zero.
func TestAggregatePartyGlicko(t *testing.T) {
	avg, maxRD := AggregatePartyGlicko(nil)
	if avg != rating.DefaultMu || maxRD != rating.DefaultPhi {
		t.Fatalf("empty party: got avg=%v maxRD=%v, want defaults %v/%v", avg, maxRD, rating.DefaultMu, rating.DefaultPhi)
	}

	avg, maxRD = AggregatePartyGlicko([]PartyRating{
		{Elo: 1400, RD: 60},
		{Elo: 1600, RD: 120},
	})
	if avg != 1500 {
		t.Errorf("avg = %v, want 1500", avg)
	}
	if maxRD != 120 {
		t.Errorf("maxRD = %v, want 120 (the worse of the two)", maxRD)
	}
}

// TestPartyOpenSkillMu confirms the extraction ValidateParty's FFA-4 spread check consumes.
func TestPartyOpenSkillMu(t *testing.T) {
	got := PartyOpenSkillMu([]PartyRating{{OpenSkillMu: 20}, {OpenSkillMu: 30}})
	if len(got) != 2 || got[0] != 20 || got[1] != 30 {
		t.Fatalf("expected [20 30], got %v", got)
	}
}
