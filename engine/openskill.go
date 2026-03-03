package engine

import "math"

// OpenSkillRating represents a player's skill rating using the Plackett-Luce model.
type OpenSkillRating struct {
	Mu    float64 // Mean skill estimate. Default 25.0.
	Sigma float64 // Uncertainty (std dev). Default 25/3 ≈ 8.333.
}

// NewOpenSkillRating returns a default rating (mu=25, sigma=25/3).
func NewOpenSkillRating() OpenSkillRating {
	return OpenSkillRating{Mu: 25.0, Sigma: 25.0 / 3.0}
}

// DefaultOpenSkillBeta returns the default scale parameter (8.0).
func DefaultOpenSkillBeta() float64 {
	return 8.0
}

// UpdateOpenSkill computes updated ratings after a match using the Plackett-Luce model.
// ratings: each player's current rating
// ranks: 1-indexed ordinal ranks (1 = best). Equal ranks = tie.
// beta: scale parameter (use DefaultOpenSkillBeta() = 8.0)
// tau: additive dynamics factor (use 0.0 for circuit mode)
// Returns: new ratings for each player.
func UpdateOpenSkill(ratings []OpenSkillRating, ranks []int, beta, tau float64) []OpenSkillRating {
	n := len(ratings)
	sigmas := make([]float64, n)
	mus := make([]float64, n)
	for i, r := range ratings {
		mus[i] = r.Mu
		if tau > 0 {
			sigmas[i] = math.Sqrt(r.Sigma*r.Sigma + tau*tau)
		} else {
			sigmas[i] = r.Sigma
		}
	}

	deltaMu := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			cij := math.Sqrt(2*beta*beta + sigmas[i]*sigmas[i] + sigmas[j]*sigmas[j])
			expI := math.Exp(mus[i] / cij)
			expJ := math.Exp(mus[j] / cij)
			pij := expI / (expI + expJ)

			var sij float64
			if ranks[i] < ranks[j] {
				sij = 1.0
			} else if ranks[i] > ranks[j] {
				sij = 0.0
			} else {
				sij = 0.5
			}
			deltaMu[i] += (sigmas[i] * sigmas[i] / cij) * (sij - pij)
		}
	}

	result := make([]OpenSkillRating, n)
	for i := 0; i < n; i++ {
		var info float64
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			cij := math.Sqrt(2*beta*beta + sigmas[i]*sigmas[i] + sigmas[j]*sigmas[j])
			gammaIJ := sigmas[i] / cij
			info += gammaIJ * (1 - gammaIJ)
		}
		newSigma := sigmas[i] * math.Sqrt(math.Max(1-sigmas[i]*sigmas[i]*info, 0.0001))
		result[i] = OpenSkillRating{
			Mu:    mus[i] + deltaMu[i],
			Sigma: newSigma,
		}
	}
	return result
}

// RanksFromScores converts cumulative scores to ordinal ranks with a tie margin.
// Players within tieMargin points of each other receive the same rank.
// Lower score = better rank (rank 1 is best).
// Returns: 1-indexed ranks aligned with the input scores slice.
func RanksFromScores(scores []int, tieMargin int) []int {
	n := len(scores)
	// Build index-score pairs
	type indexedScore struct {
		idx   int
		score int
	}
	pairs := make([]indexedScore, n)
	for i, s := range scores {
		pairs[i] = indexedScore{i, s}
	}
	// Sort ascending by score
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if pairs[j].score < pairs[i].score {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	ranks := make([]int, n)
	groupMin := pairs[0].score
	groupRank := 1
	for pos, p := range pairs {
		if pos == 0 {
			ranks[p.idx] = 1
			continue
		}
		if p.score-groupMin > tieMargin {
			// Start new group
			groupMin = p.score
			groupRank = pos + 1
		}
		ranks[p.idx] = groupRank
	}
	return ranks
}
