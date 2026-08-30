// src/utils/ratingPool.ts
import type { Tier } from '@/components/ds/data/TierBadge';

/**
 * Rating pool identifiers, matching the service's pools (validLeaderboardPools in
 * service/internal/handlers/leaderboard.go and ratingPools in
 * service/internal/database/history.go): the per-pool elo and phi column pairs on
 * the users table.
 */
export type RatingPool = '1v1' | '4p' | '7p8p';

/** Display order and labels for the rating pools. */
export const RATING_POOLS: Array<[RatingPool, string]> = [
	['1v1', 'Head to Head'],
	['4p', '4 Player'],
	['7p8p', '7-8 Player']
];

const POOL_LABELS: Record<string, string> = Object.fromEntries(RATING_POOLS);

/** Maps a pool id to its label, falling back to the raw id for an unknown pool. */
export function ratingPoolLabel(pool: string): string {
	return POOL_LABELS[pool] ?? pool;
}

/**
 * Matchmaking queue pool identifiers (QueueConfig.RatingPool in
 * service/internal/matchmaking/validation.go). Distinct id space from RatingPool above:
 * these group queues for matchmaking/rating-update purposes (h2h_qp is the hidden-rating
 * quickplay pool, separate from the visible h2h_ranked pool), they are not the '1v1' /
 * '4p' / '7p8p' leaderboard pool ids returned by the ratings summary endpoint.
 */
export type QueuePoolId = 'h2h_qp' | 'h2h_ranked' | 'ffa4';

const QUEUE_POOL_LABELS: Record<string, string> = {
	h2h_qp: 'H2H Quickplay pool',
	h2h_ranked: 'H2H Ranked pool',
	ffa4: 'FFA-4 pool'
};

/** Maps a matchmaking queue's rating pool id to a player-readable label. */
export function queuePoolLabel(pool: string): string {
	return QUEUE_POOL_LABELS[pool] ?? pool;
}

const TIER_CUTOFFS: Array<[number, Tier]> = [
	[2000, 'grandmaster'],
	[1850, 'master'],
	[1700, 'diamond'],
	[1550, 'platinum'],
	[1400, 'gold'],
	[1250, 'silver']
];

/**
 * Client-side rating -> tier bucketing. The server contract carries only numeric
 * rating/rd, no tier field, so tiers are derived here for display. All pools rank on
 * the same Glicko-2 elo scale, so one cutoff table applies until the server exposes
 * tiers. Shared by the leaderboard and the profile rating summary so the two surfaces
 * cannot bucket the same rating differently.
 */
export function tierFromRating(rating: number): Tier {
	for (const [cutoff, tier] of TIER_CUTOFFS) {
		if (rating >= cutoff) return tier;
	}
	return 'bronze';
}

/** Rating with its deviation, the form both the leaderboard and the profile display. */
export function formatRating(rating: number, rd: number): string {
	return `${Math.round(rating)} ± ${Math.round(rd)}`;
}
