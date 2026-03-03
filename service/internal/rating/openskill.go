package rating

import (
	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// FinalizeCircuitRatings updates OpenSkill ratings for all players after a circuit tournament.
// scores maps player UUID to their final cumulative score (lower = better).
// Returns updated users with new OpenSkillMu/OpenSkillSigma.
func FinalizeCircuitRatings(players []models.User, scores map[uuid.UUID]int) []models.User {
	n := len(players)
	if n < 2 {
		return players
	}

	// Build ratings from current user state
	ratings := make([]engine.OpenSkillRating, n)
	scoreSlice := make([]int, n)
	for i, p := range players {
		if p.OpenSkillMu == 0 && p.OpenSkillSigma == 0 {
			ratings[i] = engine.NewOpenSkillRating()
		} else {
			ratings[i] = engine.OpenSkillRating{Mu: p.OpenSkillMu, Sigma: p.OpenSkillSigma}
		}
		scoreSlice[i] = scores[p.ID]
	}

	// Convert scores to ranks with 3-point tie margin (T6)
	ranks := engine.RanksFromScores(scoreSlice, 3)

	// Update ratings: beta=8.0, tau=0.0 (T6)
	updated := engine.UpdateOpenSkill(ratings, ranks, engine.DefaultOpenSkillBeta(), 0.0)

	// Apply back to users
	for i := range players {
		players[i].OpenSkillMu = updated[i].Mu
		players[i].OpenSkillSigma = updated[i].Sigma
	}
	return players
}
