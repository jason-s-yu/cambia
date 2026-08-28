// src/pages/ProfilePage.tsx
import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useHistoryStore } from '@/stores/historyStore';
import Panel from '@/components/ds/chrome/Panel';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
import Input from '@/components/ds/core/Input';
import Spinner from '@/components/ds/core/Spinner';
import DsRatingSummary from '@/components/profile/DsRatingSummary';
import DsGameHistory from '@/components/profile/DsGameHistory';

/** Eyebrow label over a plain value, for the account panel's identity fields. */
const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
	<div style={{ minWidth: 0 }}>
		<p
			style={{
				margin: '0 0 4px',
				fontSize: 'var(--text-2xs)',
				fontWeight: 'var(--weight-bold)',
				letterSpacing: 'var(--tracking-caps)',
				textTransform: 'uppercase',
				color: 'var(--text-tertiary)'
			}}
		>
			{label}
		</p>
		<p style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--text-primary)', overflowWrap: 'anywhere' }}>{children}</p>
	</div>
);

const ProfilePage: React.FC = () => {
	const user = useAuthStore((state) => state.user);
	const isLoading = useAuthStore((state) => state.isLoading);
	const error = useAuthStore((state) => state.error);
	const clearError = useAuthStore((state) => state.clearError);
	const claimAccount = useAuthStore((state) => state.claimAccount);

	const userId = user?.id;
	const fetchGames = useHistoryStore((state) => state.fetchGames);
	const fetchMoreGames = useHistoryStore((state) => state.fetchMoreGames);
	const fetchRatings = useHistoryStore((state) => state.fetchRatings);
	const resetHistory = useHistoryStore((state) => state.reset);
	const games = useHistoryStore((state) => state.games);
	const gamesTotal = useHistoryStore((state) => state.total);
	const gamesLoading = useHistoryStore((state) => state.gamesLoading);
	const gamesLoaded = useHistoryStore((state) => state.gamesLoaded);
	const gamesError = useHistoryStore((state) => state.gamesError);
	const ratings = useHistoryStore((state) => state.ratings);
	const ratingsError = useHistoryStore((state) => state.ratingsError);

	// Keyed on the user id so a claim (guest -> account) or an account switch refetches
	// against the new session rather than leaving the previous account's rows on screen.
	useEffect(() => {
		if (!userId) return;
		resetHistory();
		fetchGames();
		fetchRatings();
	}, [userId, resetHistory, fetchGames, fetchRatings]);

	const [claimUsername, setClaimUsername] = useState('');
	const [claimEmail, setClaimEmail] = useState('');
	const [claimPassword, setClaimPassword] = useState('');

	const handleClaim = async (e: React.FormEvent) => {
		e.preventDefault();
		clearError();
		if (!claimEmail || !claimPassword) {
			useAuthStore.setState({ error: 'Email and password are required to claim this account.' });
			return;
		}
		// On success the store refreshes `user` via checkAuth(), which flips
		// is_ephemeral to false and this form unmounts itself.
		await claimAccount({ email: claimEmail, password: claimPassword, username: claimUsername || undefined });
	};

	if (isLoading && !user) {
		return (
			<div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 'var(--space-16) var(--space-5)' }}>
				<Spinner label='Loading profile' />
			</div>
		);
	}

	if (!user) {
		return <Navigate to="/login" replace />;
	}

	const formatDate = (iso?: string) => {
		if (!iso) return 'Unknown';
		return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
	};

	return (
		<div
			style={{
				padding: 'var(--space-6) var(--space-5)',
				maxWidth: 820,
				margin: '0 auto',
				width: '100%',
				display: 'flex',
				flexDirection: 'column',
				gap: 'var(--space-5)'
			}}
		>
			<div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
				<div style={{ minWidth: 0 }}>
					<h1
						style={{
							margin: 0,
							fontSize: 'var(--ds-text-2xl)',
							fontWeight: 'var(--weight-bold)',
							letterSpacing: 'var(--ds-tracking-tight)',
							lineHeight: 'var(--ds-leading-tight)',
							overflowWrap: 'anywhere'
						}}
					>
						{user.username}
					</h1>
					<p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>
						Ratings, record and match history.
					</p>
				</div>
				{user.is_ephemeral ? <Badge tone='warning'>guest</Badge> : <Badge tone='neutral'>account</Badge>}
			</div>

			<Panel title='Account'>
				{user.is_ephemeral && (
					<div
						style={{
							marginBottom: 'var(--space-4)',
							padding: '10px 12px',
							background: 'var(--status-warning-bg)',
							border: '1px solid var(--status-warning-border)',
							borderRadius: 'var(--ds-radius-md)',
							color: 'var(--status-warning)',
							fontSize: 'var(--ds-text-sm)',
							lineHeight: 'var(--ds-leading-snug)'
						}}
					>
						Guest account. It lives in this browser only. Claim it below to keep it.
					</div>
				)}
				<div className="grid gap-4 sm:grid-cols-2">
					<Field label='Username'>{user.username}</Field>
					{user.email && <Field label='Email'>{user.email}</Field>}
					<Field label='Member since'>{formatDate(user.created_at)}</Field>
					<Field label='Last login'>{formatDate(user.last_login)}</Field>
				</div>
			</Panel>

			<DsRatingSummary summary={ratings} error={ratingsError} />

			<DsGameHistory
				games={games}
				total={gamesTotal}
				isLoading={gamesLoading}
				loaded={gamesLoaded}
				error={gamesError}
				onLoadMore={fetchMoreGames}
			/>

			{user.is_ephemeral && (
				<Panel title='Claim account'>
					<p style={{ margin: '0 0 var(--space-4)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
						Add an email and password to keep this account and its ratings.
					</p>
					<form onSubmit={handleClaim} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
						{error && (
							<div
								role='alert'
								style={{
									display: 'flex',
									alignItems: 'flex-start',
									gap: 'var(--space-3)',
									padding: '10px 12px',
									background: 'var(--status-danger-bg)',
									border: '1px solid var(--status-danger-border)',
									borderRadius: 'var(--ds-radius-md)',
									color: 'var(--status-danger)',
									fontSize: 'var(--ds-text-sm)',
									lineHeight: 'var(--ds-leading-snug)'
								}}
							>
								<span style={{ flex: 1 }}>{error}</span>
								<button
									type='button'
									onClick={clearError}
									aria-label='Dismiss error'
									style={{
										flex: 'none',
										background: 'transparent',
										border: 'none',
										padding: 0,
										cursor: 'pointer',
										color: 'inherit',
										fontSize: 'var(--ds-text-lg)',
										lineHeight: 1
									}}
								>
									&times;
								</button>
							</div>
						)}
						<Input
							label='Username'
							type='text'
							value={claimUsername}
							onChange={(e) => setClaimUsername(e.target.value)}
							placeholder={user.username}
							disabled={isLoading}
						/>
						<Input
							label='Email'
							type='email'
							value={claimEmail}
							onChange={(e) => setClaimEmail(e.target.value)}
							placeholder='you@example.com'
							disabled={isLoading}
						/>
						<Input
							label='Password'
							type='password'
							value={claimPassword}
							onChange={(e) => setClaimPassword(e.target.value)}
							placeholder='Create a password'
							disabled={isLoading}
						/>
						<div>
							<Button disabled={isLoading}>{isLoading ? 'Claiming' : 'Claim account'}</Button>
						</div>
					</form>
				</Panel>
			)}
		</div>
	);
};

export default ProfilePage;
