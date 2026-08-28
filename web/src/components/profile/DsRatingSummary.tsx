// src/components/profile/DsRatingSummary.tsx
// Profile rating summary (cambia-784): the caller's standing in every Glicko-2 pool
// plus their lifetime record, read from GET /user/ratings via historyStore. Pools the
// player has never played are still shown, at their baseline, so the panel has a stable
// shape instead of appearing and disappearing row by row.
import React from 'react';
import Panel from '@/components/ds/chrome/Panel';
import Spinner from '@/components/ds/core/Spinner';
import Badge from '@/components/ds/core/Badge';
import TierBadge from '@/components/ds/data/TierBadge';
import StatRow from '@/components/ds/data/StatRow';
import type { PoolRating, RatingSummary } from '@/services/historyService';
import { ratingPoolLabel, tierFromRating } from '@/utils/ratingPool';

const winRate = (wins: number, games: number): string =>
	games > 0 ? `${Math.round((wins / games) * 100)}%` : '—';

const PoolCard: React.FC<{ pool: PoolRating }> = ({ pool }) => {
	const unplayed = pool.games === 0;
	return (
		<div
			style={{
				border: '1.5px solid var(--border-subtle)',
				borderRadius: 'var(--ds-radius-md)',
				padding: '12px 14px',
				display: 'flex',
				flexDirection: 'column',
				gap: 8,
				opacity: unplayed ? 0.68 : 1
			}}
		>
			<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
				<span style={{ fontWeight: 'var(--weight-bold)' }}>{ratingPoolLabel(pool.pool)}</span>
				{unplayed
					? <Badge tone='neutral'>unranked</Badge>
					: <TierBadge tier={tierFromRating(pool.rating)} size='sm' />}
			</div>
			<div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
				<span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-bold)' }}>
					{Math.round(pool.rating)}
				</span>
				<span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', fontFamily: 'var(--ds-font-mono)' }}>
					± {Math.round(pool.rd)}
				</span>
			</div>
			<div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)' }}>
				<span>
					<span style={{ color: 'var(--text-tertiary)' }}>Games </span>
					<span style={{ fontFamily: 'var(--ds-font-mono)' }}>{pool.games}</span>
				</span>
				<span>
					<span style={{ color: 'var(--text-tertiary)' }}>Won </span>
					<span style={{ fontFamily: 'var(--ds-font-mono)' }}>{pool.wins}</span>
					{pool.games > 0 && <span style={{ color: 'var(--text-tertiary)' }}> ({winRate(pool.wins, pool.games)})</span>}
				</span>
				<span>
					<span style={{ color: 'var(--text-tertiary)' }}>Peak </span>
					<span style={{ fontFamily: 'var(--ds-font-mono)' }}>{Math.round(pool.peak)}</span>
				</span>
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
							<Spinner label='Loading ratings…' />
						</div>
					)}
			</Panel>
		);
	}

	const { record } = summary;
	const neverPlayed = record.games === 0;

	return (
		<Panel
			title='Ratings'
			action={
				neverPlayed
					? <Badge tone='neutral'>no games yet</Badge>
					: <Badge tone='info' mono>{record.wins}W · {record.games - record.wins}L</Badge>
			}
		>
			{neverPlayed && (
				<p style={{ margin: '0 0 12px', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
					Play a ranked game to start a rating. Everyone begins at 1500 with a wide deviation, which narrows as you play.
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
