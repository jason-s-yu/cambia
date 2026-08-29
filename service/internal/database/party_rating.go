// internal/database/party_rating.go
package database

import (
	"context"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/rating"
)

// PartyRating holds one party member's rating fields as they stand at matchmaking enqueue time:
// the Glicko-2 Elo/RD for the pool the queue's player count selects, and OpenSkill mu (the FFA-4
// party spread check reads mu, not Elo - see matchmaking.ValidateParty).
type PartyRating struct {
	Elo         float64
	RD          float64
	OpenSkillMu float64
}

// LoadPartyRatings loads each of userIDs' live rating fields for mode (see
// rating.ModeForPlayerCount), substituting the pool default - rating.DefaultMu/DefaultPhi for
// Glicko-2, rating.DefaultOpenSkillMu for OpenSkill - for any player this can't resolve a row
// for: the user has never played a rated game in the pool (fresh row, DB defaults already equal
// these constants), has no row at all (e.g. an ephemeral session), or DB itself is unreachable
// (DB == nil, mirroring the nil-DB guard in circuit_rating.go and ws.go). A no-rating-row party
// must still be allowed to queue, so this never errors; it degrades to defaults instead.
//
// mode == "" (an unsupported roster size, see rating.ModeForPlayerCount) also degrades to
// defaults for every player, since there is no pool to read Elo/RD from.
func LoadPartyRatings(ctx context.Context, userIDs []uuid.UUID, mode rating.RatingMode) []PartyRating {
	out := make([]PartyRating, len(userIDs))
	for i, id := range userIDs {
		pr := PartyRating{
			Elo:         rating.DefaultMu,
			RD:          rating.DefaultPhi,
			OpenSkillMu: rating.DefaultOpenSkillMu,
		}
		if DB != nil && mode != "" {
			if u, err := GetUserByID(ctx, id); err == nil {
				elo, phi, _ := rating.PoolFields(*u, mode)
				pr.Elo = float64(elo)
				if phi > 0 {
					pr.RD = phi
				}
				pr.OpenSkillMu = u.OpenSkillMu
			}
		}
		out[i] = pr
	}
	return out
}

// AggregatePartyGlicko reduces a party's per-member ratings to the two figures
// matchmaking.QueuedLobby carries for the quality gate: the party's average Elo and its worst
// (highest) RD. An empty party reads as pool-default, the same value a solo unrated player would
// carry, so a Matchmaker with no PartyRating callers still spreads-checks predictably rather than
// comparing against a zero.
func AggregatePartyGlicko(ratings []PartyRating) (avgElo, maxRD float64) {
	if len(ratings) == 0 {
		return rating.DefaultMu, rating.DefaultPhi
	}
	var sum float64
	for _, r := range ratings {
		sum += r.Elo
		if r.RD > maxRD {
			maxRD = r.RD
		}
	}
	return sum / float64(len(ratings)), maxRD
}

// PartyOpenSkillMu extracts the OpenSkill mu values matchmaking.ValidateParty's FFA-4 spread
// check reads.
func PartyOpenSkillMu(ratings []PartyRating) []float64 {
	out := make([]float64, len(ratings))
	for i, r := range ratings {
		out[i] = r.OpenSkillMu
	}
	return out
}
