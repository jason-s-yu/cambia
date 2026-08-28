// src/pages/LeaderboardPage.tsx
import React, { useEffect } from 'react';
import TierBadge from '@/components/ds/data/TierBadge';
import Panel from '@/components/ds/chrome/Panel';
import Spinner from '@/components/ds/core/Spinner';
import { useLeaderboardStore, type LeaderboardPool } from '@/stores/leaderboardStore';
import type { LeaderboardRow } from '@/services/leaderboardService';
import { RATING_POOLS, formatRating, tierFromRating } from '@/utils/ratingPool';

const GRID_WITH_PEAK = '56px 1fr 150px 130px 70px 130px';
const GRID_NO_PEAK = '56px 1fr 150px 130px 70px';

// Pool list, labels, tier cutoffs and rating formatting live in utils/ratingPool so
// this page and the profile rating summary cannot drift apart on any of them.
const POOLS: Array<[LeaderboardPool, string]> = RATING_POOLS;

interface LbRowProps {
	r: LeaderboardRow;
	showPeak: boolean;
	you?: boolean;
}

const LbRow: React.FC<LbRowProps> = ({ r, showPeak, you = false }) => {
	return (
		<div
			style={{
				display: 'grid',
				gridTemplateColumns: showPeak ? GRID_WITH_PEAK : GRID_NO_PEAK,
				alignItems: 'center',
				gap: 10,
				padding: '9px 14px',
				background: you ? 'rgba(223,174,71,0.08)' : 'transparent',
				borderTop: '1px solid var(--border-subtle)',
				border: you ? '1.5px solid var(--honey-600)' : undefined,
				borderRadius: you ? 'var(--ds-radius-md)' : 0
			}}
		>
			<span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', color: r.rank <= 3 ? 'var(--honey-400)' : 'var(--text-tertiary)' }}>#{r.rank}</span>
			<span style={{ display: 'flex', alignItems: 'center', gap: 9, fontWeight: 'var(--weight-bold)', minWidth: 0 }}>
				<span style={{ width: 24, height: 24, flex: 'none', borderRadius: '50%', background: 'var(--surface-inset)', border: '1.5px solid var(--outline-ink)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'var(--text-primary)' }}>{(r.username[0] ?? '?').toUpperCase()}</span>
				<span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.username}{you ? ' (you)' : ''}</span>
			</span>
			<TierBadge tier={tierFromRating(r.rating)} size='sm' />
			<span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-sm)' }}>{formatRating(r.rating, r.rd)}</span>
			<span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)' }}>{r.games}</span>
			{showPeak && (r.peak != null
				? <TierBadge tier={tierFromRating(r.peak)} size='sm' />
				: <span style={{ color: 'var(--text-tertiary)' }}>—</span>)}
		</div>
	);
};

/**
 * Real-data leaderboard page (cambia-485), ported from the DS preview
 * pages/ds/LeaderboardScreen.tsx (cambia-438). Wires the ds preview's tab
 * layout and row design to GET /leaderboard via leaderboardStore, replacing
 * the mock H2H_ROWS/YOU_ROW sample data.
 */
const LeaderboardPage: React.FC = () => {
	const [pool, setPool] = React.useState<LeaderboardPool>('1v1');
	const fetchPool = useLeaderboardStore((state) => state.fetchPool);
	const poolState = useLeaderboardStore((state) => state.pools[pool]);

	useEffect(() => {
		fetchPool(pool);
	}, [pool, fetchPool]);

	const { rows, you, isLoading, error } = poolState;
	const showPeak = rows.some((r) => r.peak != null) || (you?.peak != null);

	return (
		<div style={{ padding: 22, maxWidth: 1000, margin: '0 auto', width: '100%' }}>
			<div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
				<div>
					<h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-3xl)', fontWeight: 'var(--weight-regular)' }}>Leaderboard</h1>
					<p style={{ margin: '4px 0 0', color: 'var(--text-secondary)' }}>Glicko-2 ratings by pool · updated after every rated game</p>
				</div>
				<div style={{ display: 'flex', gap: 6, background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)', padding: 4 }}>
					{POOLS.map(([id, label]) => (
						<button
							key={id}
							onClick={() => setPool(id)}
							style={{
								padding: '6px 16px',
								borderRadius: 'var(--radius-pill)',
								cursor: 'pointer',
								fontFamily: 'var(--font-ui)',
								fontWeight: 'var(--weight-bold)',
								fontSize: 'var(--ds-text-sm)',
								whiteSpace: 'nowrap',
								background: pool === id ? 'var(--ember-500)' : 'transparent',
								color: pool === id ? 'var(--text-on-ember)' : 'var(--text-secondary)',
								border: pool === id ? '1.5px solid var(--outline-ink)' : '1.5px solid transparent'
							}}
						>{label}</button>
					))}
				</div>
			</div>
			<Panel style={{ marginTop: 18, padding: '6px 4px' }}>
				{isLoading && (
					<div style={{ display: 'flex', justifyContent: 'center', padding: '32px 14px' }}>
						<Spinner label='Loading leaderboard…' />
					</div>
				)}
				{!isLoading && error && (
					<div style={{ padding: '24px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>{error}</div>
				)}
				{!isLoading && !error && rows.length === 0 && (
					<div style={{ padding: '24px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>No ranked players yet for this pool.</div>
				)}
				{!isLoading && !error && rows.length > 0 && (
					<>
						<div style={{ display: 'grid', gridTemplateColumns: showPeak ? GRID_WITH_PEAK : GRID_NO_PEAK, gap: 10, padding: '8px 14px', fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>
							<span>Rank</span><span>Player</span><span>Tier</span><span>Rating</span><span>Games</span>{showPeak && <span>Peak</span>}
						</div>
						{rows.map((r) => <LbRow key={r.userId} r={r} showPeak={showPeak} />)}
						{you && (
							<>
								<div style={{ padding: '10px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)' }}>···</div>
								<LbRow r={you} showPeak={showPeak} you />
							</>
						)}
					</>
				)}
			</Panel>
		</div>
	);
};

export default LeaderboardPage;
