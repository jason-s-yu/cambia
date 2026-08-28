// src/components/profile/DsGameHistory.tsx
// Profile match history (cambia-784): the caller's own past games, newest first, read
// from GET /user/history via historyStore. One row per game carries the outcome, the
// score line, who was at the table, and the rating change when the game was rated.
import React from 'react';
import Panel from '@/components/ds/chrome/Panel';
import Spinner from '@/components/ds/core/Spinner';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
import type { HistoryGame } from '@/services/historyService';
import { ratingPoolLabel } from '@/utils/ratingPool';

/** Absolute date, plus a relative hint for anything inside the last week. */
const formatPlayedAt = (iso: string): string => {
	const then = new Date(iso);
	if (Number.isNaN(then.getTime())) return 'Unknown date';

	const absolute = then.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
	const elapsedMs = Date.now() - then.getTime();
	if (elapsedMs < 0 || elapsedMs > 7 * 24 * 60 * 60 * 1000) return absolute;

	const hours = Math.floor(elapsedMs / (60 * 60 * 1000));
	if (hours < 1) return 'Just now';
	if (hours < 24) return `${hours}h ago`;
	const days = Math.floor(hours / 24);
	return days === 1 ? 'Yesterday' : `${days}d ago`;
};

/** "1v1", "4 players" etc. from the seat count the server recorded. */
const tableLabel = (playerCount: number): string => {
	if (playerCount === 2) return '1v1';
	if (playerCount > 0) return `${playerCount} players`;
	return 'Unknown table';
};

const GameRow: React.FC<{ game: HistoryGame }> = ({ game }) => {
	// didWin is nullable in the schema; an unrecorded outcome renders as neither a win
	// nor a loss rather than silently as a loss.
	const won = game.didWin === true;
	const lost = game.didWin === false;
	const delta = game.rating?.delta ?? null;

	return (
		<div
			style={{
				display: 'flex',
				alignItems: 'flex-start',
				gap: 12,
				padding: '11px 2px',
				borderTop: '1px solid var(--border-subtle)'
			}}
		>
			<span
				aria-hidden
				style={{
					width: 3,
					alignSelf: 'stretch',
					flex: 'none',
					borderRadius: 2,
					background: won ? 'var(--status-success)' : lost ? 'var(--status-danger)' : 'var(--border-strong)'
				}}
			/>
			<div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 5 }}>
				<div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
					{won && <Badge tone='success'>win</Badge>}
					{lost && <Badge tone='danger'>loss</Badge>}
					{game.didWin === null && <Badge tone='neutral'>no result</Badge>}
					{game.rated ? <Badge tone='gold'>ranked</Badge> : <Badge tone='neutral'>casual</Badge>}
					<span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>{tableLabel(game.playerCount)}</span>
				</div>
				<div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', overflowWrap: 'anywhere' }}>
					{game.opponents.length > 0
						? <>vs {game.opponents.map((o) => o.username || 'Unknown player').join(', ')}</>
						: <span style={{ color: 'var(--text-tertiary)' }}>Opponents not recorded</span>}
				</div>
				{game.rating && (
					<div style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>
						{ratingPoolLabel(game.rating.pool)} {Math.round(game.rating.old)} -&gt; {Math.round(game.rating.new)}
					</div>
				)}
			</div>
			<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flex: 'none', fontVariantNumeric: 'tabular-nums' }}>
				<span style={{ fontWeight: 'var(--weight-black)', fontSize: 'var(--ds-text-lg)', lineHeight: 'var(--ds-leading-tight)', color: 'var(--text-primary)' }}>
					{game.score != null ? game.score : '-'}
				</span>
				{delta != null && (
					<span
						style={{
							fontSize: 'var(--ds-text-xs)',
							fontWeight: 'var(--weight-medium)',
							color: delta >= 0 ? 'var(--status-success)' : 'var(--status-danger)'
						}}
					>
						{delta >= 0 ? '+' : ''}{delta}
					</span>
				)}
				<span style={{ fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)', whiteSpace: 'nowrap' }}>
					{formatPlayedAt(game.playedAt)}
				</span>
			</div>
		</div>
	);
};

interface DsGameHistoryProps {
	games: HistoryGame[];
	total: number;
	isLoading: boolean;
	loaded: boolean;
	error: string | null;
	onLoadMore: () => void;
}

const DsGameHistory: React.FC<DsGameHistoryProps> = ({ games, total, isLoading, loaded, error, onLoadMore }) => {
	const hasMore = games.length < total;
	// The first load has nothing to show yet; a "load more" page keeps the rows on screen
	// and shows its own spinner under them. Gated on `loaded` rather than `isLoading` so
	// the panel is not momentarily empty between mount and the effect that starts the
	// fetch, when neither flag is set yet.
	const initialLoading = games.length === 0 && !loaded && !error;

	return (
		<Panel
			title='Match history'
			action={total > 0 ? <Badge tone='neutral'>{total}</Badge> : undefined}
		>
			{initialLoading && (
				<div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
					<Spinner label='Loading history' />
				</div>
			)}

			{error && games.length === 0 && !isLoading && (
				<div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>{error}</div>
			)}

			{!isLoading && !error && loaded && games.length === 0 && (
				<div style={{ padding: '18px 0', textAlign: 'center' }}>
					<p style={{ margin: 0, fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>No games yet.</p>
					<p style={{ margin: '4px 0 0', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>
						Finished games land here with your score, the table, and any rating change.
					</p>
				</div>
			)}

			{games.length > 0 && (
				<div style={{ display: 'flex', flexDirection: 'column' }}>
					{games.map((g) => <GameRow key={g.gameId} game={g} />)}
				</div>
			)}

			{error && games.length > 0 && (
				<div style={{ padding: '10px 0 0', textAlign: 'center', color: 'var(--status-danger)', fontSize: 'var(--ds-text-xs)' }}>{error}</div>
			)}

			{hasMore && (
				<div style={{ display: 'flex', justifyContent: 'center', paddingTop: 12 }}>
					{isLoading && games.length > 0
						? <Spinner size={18} label='Loading' />
						: <Button variant='ghost' size='sm' onClick={onLoadMore}>Load more</Button>}
				</div>
			)}
		</Panel>
	);
};

export default DsGameHistory;
