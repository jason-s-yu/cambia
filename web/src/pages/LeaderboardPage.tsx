// src/pages/LeaderboardPage.tsx
import React, { useEffect, useState } from 'react';
import TierBadge from '@/components/ds/data/TierBadge';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import Spinner from '@/components/ds/core/Spinner';
import { useLeaderboardStore, type LeaderboardPool } from '@/stores/leaderboardStore';
import type { LeaderboardRow } from '@/services/leaderboardService';
import { RATING_POOLS, formatRating, tierFromRating } from '@/utils/ratingPool';

// Row grid. Narrow viewports collapse to rank / player / rating, with the tier under
// the name and the game count under the rating; from the sm breakpoint every column
// gets its own track. Full class strings so Tailwind's scanner sees each candidate.
const GRID_BASE = 'grid items-center gap-2.5 grid-cols-[40px_minmax(0,1fr)_auto]';
const GRID_WITH_PEAK = `${GRID_BASE} sm:grid-cols-[56px_minmax(0,1fr)_140px_130px_70px_130px]`;
const GRID_NO_PEAK = `${GRID_BASE} sm:grid-cols-[56px_minmax(0,1fr)_140px_130px_70px]`;

// Pool list, labels, tier cutoffs and rating formatting live in utils/ratingPool so
// this page and the profile rating summary cannot drift apart on any of them.
const POOLS: Array<[LeaderboardPool, string]> = RATING_POOLS;

const CELL_PAD: React.CSSProperties = { padding: '9px 14px' };

// A segmented control, announced as pressed buttons. It was carrying role='tab'
// with aria-selected but none of the rest of the pattern (no tabpanel, no roving
// tabindex, no arrow keys), so it promised a keyboard model it did not have;
// aria-pressed describes what these actually are (cambia-876, DL-5 review F5).
const PoolTab: React.FC<{ label: string; active: boolean; onSelect: () => void }> = ({ label, active, onSelect }) => {
	const [hover, setHover] = useState(false);
	return (
		<button
			type='button'
			aria-pressed={active}
			onClick={onSelect}
			onMouseEnter={() => setHover(true)}
			onMouseLeave={() => setHover(false)}
			style={{
				padding: '5px 14px',
				borderRadius: 'var(--radius-pill)',
				cursor: 'pointer',
				fontWeight: 'var(--weight-bold)',
				fontSize: 'var(--ds-text-sm)',
				whiteSpace: 'nowrap',
				background: active ? 'var(--accent-gold)' : hover ? 'var(--interactive-hover)' : 'transparent',
				color: active ? 'var(--text-on-gold)' : 'var(--text-secondary)',
				border: '1px solid transparent',
				transition: 'background var(--dur-fast) var(--ds-ease-out), color var(--dur-fast) var(--ds-ease-out)'
			}}
		>
			{label}
		</button>
	);
};

interface LbRowProps {
	r: LeaderboardRow;
	showPeak: boolean;
	you?: boolean;
}

const LbRow: React.FC<LbRowProps> = ({ r, showPeak, you = false }) => {
	const tier = tierFromRating(r.rating);
	// The own row is painted --interactive-selected, a gold tint that lands
	// lighter than the card in dark; --text-tertiary fell to 4.14:1 on it. The
	// emphasized row steps its metadata up a tier rather than lifting the tier
	// everywhere it is used (cambia-935, R1).
	const meta = you ? 'var(--text-secondary)' : 'var(--text-tertiary)';
	return (
		<div
			className={showPeak ? GRID_WITH_PEAK : GRID_NO_PEAK}
			style={{
				...CELL_PAD,
				fontVariantNumeric: 'tabular-nums',
				background: you ? 'var(--interactive-selected)' : 'transparent',
				borderTop: you ? undefined : '1px solid var(--border-subtle)',
				border: you ? '1px solid var(--border-accent)' : undefined,
				borderRadius: you ? 'var(--ds-radius-md)' : 0
			}}
		>
			{/* Gold as text takes --accent-gold-text: the fill token measures 2.45:1
			    on the own row in light, the text token 5.79:1 (cambia-935, R1). */}
			<span style={{ fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)', color: r.rank <= 3 ? 'var(--accent-gold-text)' : meta }}>
				#{r.rank}
			</span>
			<span style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
				<span
					aria-hidden
					style={{
						width: 24,
						height: 24,
						flex: 'none',
						borderRadius: '50%',
						background: 'var(--surface-inset)',
						border: '1px solid var(--border-default)',
						display: 'inline-flex',
						alignItems: 'center',
						justifyContent: 'center',
						fontSize: 'var(--text-2xs)',
						fontWeight: 'var(--weight-bold)',
						color: 'var(--text-primary)'
					}}
				>
					{(r.username[0] ?? '?').toUpperCase()}
				</span>
				<span style={{ display: 'flex', flexDirection: 'column', minWidth: 0, gap: 2 }}>
					<span style={{ fontWeight: 'var(--weight-bold)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
						{r.username}{you ? ' (you)' : ''}
					</span>
					{/* Narrow-viewport placement of the tier chip; the sm+ grid gives it a column. */}
					<span className='flex sm:hidden'>
						<TierBadge tier={tier} size='sm' style={{ gap: 5 }} />
					</span>
				</span>
			</span>
			<span className='hidden sm:flex'>
				<TierBadge tier={tier} size='sm' />
			</span>
			<span className='flex flex-col items-end gap-0.5 sm:items-start' style={{ fontSize: 'var(--ds-text-sm)' }}>
				<span style={{ fontWeight: 'var(--weight-medium)' }}>{formatRating(r.rating, r.rd)}</span>
				<span className='sm:hidden' style={{ fontSize: 'var(--text-2xs)', color: meta }}>{r.games} games</span>
			</span>
			<span className='hidden sm:block' style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>{r.games}</span>
			{showPeak && (
				<span className='hidden sm:flex'>
					{/* A bare hyphen in a column of real values reads as a stray mark;
					    say what is missing (cambia-876, DL-5 review F6). */}
					{r.peak != null
						? <TierBadge tier={tierFromRating(r.peak)} size='sm' />
						: <span style={{ fontSize: 'var(--text-2xs)', color: meta }}>none yet</span>}
				</span>
			)}
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
	const [pool, setPool] = useState<LeaderboardPool>('1v1');
	const fetchPool = useLeaderboardStore((state) => state.fetchPool);
	const poolState = useLeaderboardStore((state) => state.pools[pool]);

	useEffect(() => {
		fetchPool(pool);
	}, [pool, fetchPool]);

	const { rows, you, isLoading, error } = poolState;
	const showPeak = rows.some((r) => r.peak != null) || (you?.peak != null);

	// A copy, not the shared object: an alias hands every eyebrow in the app to
	// whatever this page later adds to its header cells (cambia-892, DL-7 F5).
	const headerCell: React.CSSProperties = { ...EYEBROW };

	return (
		<div style={{ padding: 'var(--space-6) var(--space-5)', maxWidth: 1000, margin: '0 auto', width: '100%' }}>
			<div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
				<div>
					<h1
						style={{
							margin: 0,
							fontSize: 'var(--ds-text-2xl)',
							fontWeight: 'var(--weight-bold)',
							letterSpacing: 'var(--ds-tracking-tight)',
							lineHeight: 'var(--ds-leading-tight)'
						}}
					>
						Leaderboard
					</h1>
					<p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>Glicko-2 ratings by pool. Updated after every rated game.</p>
				</div>
				<div
					role='group'
					aria-label='Rating pool'
					style={{
						display: 'flex',
						gap: 4,
						maxWidth: '100%',
						background: 'var(--surface-inset)',
						border: '1px solid var(--border-default)',
						borderRadius: 'var(--radius-pill)',
						padding: 3
					}}
				>
					{POOLS.map(([id, label]) => (
						<PoolTab key={id} label={label} active={pool === id} onSelect={() => setPool(id)} />
					))}
				</div>
			</div>
			<Panel style={{ marginTop: 'var(--space-5)', padding: 'var(--space-2) var(--space-1)' }}>
				{isLoading && (
					<div style={{ display: 'flex', justifyContent: 'center', padding: '32px 14px' }}>
						<Spinner label='Loading leaderboard' />
					</div>
				)}
				{!isLoading && error && (
					<div style={{ padding: '24px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>{error}</div>
				)}
				{!isLoading && !error && rows.length === 0 && (
					<div style={{ padding: '24px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>No ranked players in this pool yet.</div>
				)}
				{!isLoading && !error && rows.length > 0 && (
					<>
						<div className={showPeak ? GRID_WITH_PEAK : GRID_NO_PEAK} style={{ padding: '8px 14px' }}>
							<span style={headerCell}>Rank</span>
							<span style={headerCell}>Player</span>
							<span className='hidden sm:block' style={headerCell}>Tier</span>
							<span className='text-right sm:text-left' style={headerCell}>Rating</span>
							<span className='hidden sm:block' style={headerCell}>Games</span>
							{showPeak && <span className='hidden sm:block' style={headerCell}>Peak</span>}
						</div>
						{rows.map((r) => <LbRow key={r.userId} r={r} showPeak={showPeak} />)}
						{you && (
							<>
								<div style={{ padding: '8px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-xs)', letterSpacing: 'var(--ds-tracking-wide)' }}>···</div>
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
