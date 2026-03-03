package matchmaking

import "fmt"

// QueueConfig describes the rules for a named matchmaking queue.
type QueueConfig struct {
	QueueID      string
	Players      int    // target player count
	Rounds       int    // rounds per match
	RatingPool   string // e.g. "h2h_qp", "h2h_ranked", "ffa4"
	Ranked       bool
	HiddenRating bool // true for h2h_quickplay
}

// QueueConfigs is the authoritative list of supported queues.
var QueueConfigs = map[string]QueueConfig{
	"h2h_quickplay":  {QueueID: "h2h_quickplay", Players: 2, Rounds: 1, RatingPool: "h2h_qp", Ranked: true, HiddenRating: true},
	"h2h_blitz":      {QueueID: "h2h_blitz", Players: 2, Rounds: 4, RatingPool: "h2h_ranked", Ranked: true},
	"h2h_rapid":      {QueueID: "h2h_rapid", Players: 2, Rounds: 8, RatingPool: "h2h_ranked", Ranked: true},
	"h2h_classical":  {QueueID: "h2h_classical", Players: 2, Rounds: 16, RatingPool: "h2h_ranked", Ranked: true},
	"ffa4_standard":  {QueueID: "ffa4_standard", Players: 4, Rounds: 8, RatingPool: "ffa4", Ranked: true},
	"ffa4_classical": {QueueID: "ffa4_classical", Players: 4, Rounds: 12, RatingPool: "ffa4", Ranked: true},
}

// GetQueueConfig looks up a queue configuration by ID.
func GetQueueConfig(queueID string) (QueueConfig, bool) {
	cfg, ok := QueueConfigs[queueID]
	return cfg, ok
}

// ValidateParty checks that a party is eligible to enter the given queue.
//
// Rules:
//   - Queue must exist in QueueConfigs.
//   - H2H queues (Players==2): partySize must be 1 (solo queue only).
//   - FFA-4 queues (Players==4): partySize must be ≤ 2.
//   - FFA-4 parties of 2: rating spread must be ≤ 360 (Glicko-2 H2H) or ≤ 15 (OpenSkill FFA-4).
//     We use ≤ 15 μ units (OpenSkill) as the FFA-4 spread limit since ratings is in μ for ffa4.
func ValidateParty(queueID string, partySize int, ratings []float64) error {
	cfg, ok := QueueConfigs[queueID]
	if !ok {
		return fmt.Errorf("matchmaking: unknown queue %q", queueID)
	}

	if partySize <= 0 {
		return fmt.Errorf("matchmaking: partySize must be > 0")
	}

	switch cfg.Players {
	case 2:
		// H2H: solo queue only.
		if partySize != 1 {
			return fmt.Errorf("matchmaking: queue %q requires solo queue (partySize=1), got %d", queueID, partySize)
		}
	case 4:
		// FFA-4: parties of 1 or 2.
		if partySize > 2 {
			return fmt.Errorf("matchmaking: queue %q allows parties up to 2, got %d", queueID, partySize)
		}
		// For parties of 2, enforce rating spread.
		if partySize == 2 && len(ratings) >= 2 {
			spread := ratingSpread(ratings)
			// FFA-4 uses OpenSkill μ; tiers are ~3-5 μ apart, spread limit = 15.
			const ffa4SpreadLimit = 15.0
			if spread > ffa4SpreadLimit {
				return fmt.Errorf("matchmaking: party rating spread %.2f exceeds limit %.2f for queue %q",
					spread, ffa4SpreadLimit, queueID)
			}
		}
	}

	return nil
}

// ratingSpread returns max - min of a slice of ratings.
func ratingSpread(ratings []float64) float64 {
	if len(ratings) == 0 {
		return 0
	}
	min, max := ratings[0], ratings[0]
	for _, r := range ratings[1:] {
		if r < min {
			min = r
		}
		if r > max {
			max = r
		}
	}
	return max - min
}
