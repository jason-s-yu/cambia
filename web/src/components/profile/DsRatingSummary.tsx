// src/components/profile/DsRatingSummary.tsx
// Profile rating summary (cambia-784): the caller's standing in every Glicko-2 pool
// plus their lifetime record, read from GET /user/ratings via historyStore. Pools the
// player has never played are still shown, at their baseline, so the panel has a stable
// shape instead of appearing and disappearing row by row.
import React from 'react';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import Spinner from '@/components/ds/core/Spinner';
import Badge from '@/components/ds/core/Badge';
import TierBadge from '@/components/ds/data/TierBadge';
import StatRow from '@/components/ds/data/StatRow';
import type { PoolRating, RatingSummary } from '@/services/historyService';
import { ratingPoolLabel, tierFromRating } from '@/utils/ratingPool';

const winRate = (wins: number, games: number): string =>
	games > 0 ? `${Math.round((wins / games) * 100)}%` : '-';

/** Eyebrow over a tabular value, one cell of the pool card's stat strip. */
const Stat: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
	<span style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
		{/* The stat strip's lighter eyebrow: shared style, its own weight. */}
		<span style={{ ...EYEBROW, fontWeight: 'var(--weight-regular)' }}>{label}</span>
		<span style={{ fontSize: 'var(--ds-text-sm)', fontWeight: 'var(--weight-medium)', color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>{children}</span>
	</span>
);

const PoolCard: React.FC<{ pool: PoolRating }> = ({ pool }) => {
	const unplayed = pool.games === 0;
	return (
		<div
			style={{
				background: 'var(--surface-2)',
				border: '1px solid var(--border-default)',
				borderRadius: 'var(--ds-radius-md)',
				padding: '12px 14px',
				display: 'flex',
				flexDirection: 'column',
				gap: 8,
				minWidth: 0
			}}
		>
			<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', minHeight: 24 }}>
				<span style={{ fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)' }}>{ratingPoolLabel(pool.pool)}</span>
				{unplayed
					? <Badge tone='neutral'>unranked</Badge>
					: <TierBadge tier={tierFromRating(pool.rating)} size='sm' />}
			</div>
			<div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap', fontVariantNumeric: 'tabular-nums' }}>
				<span
					style={{
						fontSize: 'var(--ds-text-3xl)',
						fontWeight: 'var(--weight-black)',
						lineHeight: 'var(--ds-leading-tight)',
						letterSpacing: 'var(--ds-tracking-tight)',
						color: unplayed ? 'var(--text-tertiary)' : 'var(--text-primary)'
					}}
				>
					{Math.round(pool.rating)}
				</span>
				<span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>
					± {Math.round(pool.rd)}
				</span>
			</div>
			<div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8, marginTop: 2 }}>
				<Stat label='Games'>{pool.games}</Stat>
				<Stat label='Won'>
					{pool.wins}
					{pool.games > 0 && <span style={{ color: 'var(--text-tertiary)', fontWeight: 'var(--weight-regular)' }}> ({winRate(pool.wins, pool.games)})</span>}
				</Stat>
				<Stat label='Peak'>{Math.round(pool.peak)}</Stat>
			</div>
		</div>
	);
};

interface DsRatingSummaryProps {
	summary: RatingSummary | null;
	error: string | null;
}

/**
 * No isLoading prop: with no summary and no error the fetch is either in flight or has
 * not started, and a spinner is the right answer for both. Keying the placeholder off
 * isLoading instead would blank the panel for the frame between mount and the effect
 * that starts the fetch.
 */
const DsRatingSummary: React.FC<DsRatingSummaryProps> = ({ summary, error }) => {
	if (!summary) {
		return (
			<Panel title='Ratings'>
				{error
					? <div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>{error}</div>
					: (
						<div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
							<Spinner label='Loading ratings' />
						</div>
					)}
			</Panel>
		);
	}

	const { record } = summary;
	const neverPlayed = record.games === 0;

	// record.games (lifetime, rated or not) and a pool's games (rated only) come from
	// separate queries and can disagree, e.g. a rated pool seeded without a matching
	// game_results row. hasRatedPool checks the pools directly so the badge and the
	// onboarding paragraph never claim no rating exists while a pool card below is
	// printing a real one. Requiring both (rather than !hasRatedPool alone) keeps a
	// real lifetime record on unranked-only play from being replaced by the "no games
	// yet" state (cambia-929 F5, cambia-949 CP4).
	const hasRatedPool = summary.pools.some((pool) => pool.games > 0);

	return (
		<Panel
			title='Ratings'
			action={
				neverPlayed && !hasRatedPool
					? <Badge tone='neutral'>no games yet</Badge>
					: <Badge tone='info'>{record.wins}W · {record.games - record.wins}L</Badge>
			}
		>
			{neverPlayed && !hasRatedPool && (
				<p style={{ margin: '0 0 12px', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
					Ranked games set your rating. Everyone starts at 1500 with a wide deviation that narrows as you play.
				</p>
			)}
			<div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 }}>
				{summary.pools.map((p) => <PoolCard key={p.pool} pool={p} />)}
			</div>
			<div style={{ marginTop: 14 }}>
				{/* OpenSkill mu/sigma sit on a ~25 +/- 8.3 scale, so they are shown to two
				    decimals rather than through formatRating's Glicko-scale rounding. */}
				<StatRow
					label='OpenSkill (circuit)'
					value={`${summary.openSkill.mu.toFixed(2)} ± ${summary.openSkill.sigma.toFixed(2)}`}
					style={{ borderBottom: 'none' }}
				/>
			</div>
		</Panel>
	);
};

export default DsRatingSummary;
