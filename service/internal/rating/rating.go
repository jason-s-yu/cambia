package rating

import (
	"math"
	"sort"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// FinalizeRatings runs a single Glicko-2 rating-period update on the entire group of
// players based on their final "score" (lower is better in Cambia). This function is
// called once at game end to produce updated rating fields for each player.
//
//  1. We convert final scores into a fraction from 0..1, where 1 is best rank and 0 is worst.
//  2. We call SingleOrMultiPlayerGlicko2 to apply exactly one Glicko-2 rating-period update
//     per player, per the Glickman spec (a rating period is applied once, not iterated).
//
// Note: for a true persistent Glicko-2, we store each user's phi, sigma in the DB, then
// feed them into the next match.
// TODO: Modify returns ephemeral updated ELO only
func FinalizeRatings(players []models.User, scoresMap map[uuid.UUID]int) []models.User {
	// 1) Build a rank-based fraction for each user
	type userScore struct {
		UserID uuid.UUID
		Score  int
	}
	var arr []userScore
	for _, p := range players {
		arr = append(arr, userScore{p.ID, scoresMap[p.ID]})
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Score < arr[j].Score // ascending
	})

	// We'll assign fractional scores: top rank => 1.0, last => 0.0, ties share fraction
	rankFrac := make(map[uuid.UUID]float64, len(arr))
	i := 0
	for i < len(arr) {
		j := i + 1
		for j < len(arr) && arr[j].Score == arr[i].Score {
			j++
		}
		// players i..j-1 are tied
		// midRank fraction => 1 - (avgRank / (count-1))
		avgRank := float64(i+(j-1)) / 2
		fr := 1.0 - (avgRank / float64(len(arr)-1))
		for k := i; k < j; k++ {
			rankFrac[arr[k].UserID] = fr
		}
		i = j
	}

	// Build slices for Glicko
	scores := make([]float64, len(players))
	userIndex := make(map[uuid.UUID]int)
	for i, p := range players {
		userIndex[p.ID] = i
	}
	for _, p := range players {
		idx := userIndex[p.ID]
		scores[idx] = rankFrac[p.ID]
	}

	return SingleOrMultiPlayerGlicko2(players, scores)
}

// Update1v1 applies one Glicko2 rating-period update for a single decisive 1v1 result:
// the winner scores 1 against the loser, the loser scores 0 against the winner. Both
// updates read the pre-match opponent rating, and a fresh user (zero Phi1v1/Sigma1v1)
// falls back to the baseline deviation and volatility via ratingFromUser.
func Update1v1(winner, loser models.User) (models.User, models.User) {
	wR := ratingFromUser(winner)
	lR := ratingFromUser(loser)

	newW := updateGlicko(wR, lR, 1.0)
	newL := updateGlicko(lR, wR, 0.0)

	winner.Elo1v1 = int(math.Round(newW.ToElo()))
	winner.Phi1v1 = newW.Phi * GlickoScale
	winner.Sigma1v1 = newW.Sigma

	loser.Elo1v1 = int(math.Round(newL.ToElo()))
	loser.Phi1v1 = newL.Phi * GlickoScale
	loser.Sigma1v1 = newL.Sigma

	return winner, loser
}
