// src/pages/ProfilePage.tsx
import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useHistoryStore } from '@/stores/historyStore';
import Input from '@/components/common/Input';
import Button from '@/components/common/Button';
import ErrorMessage from '@/components/common/ErrorMessage';
import DsRatingSummary from '@/components/profile/DsRatingSummary';
import DsGameHistory from '@/components/profile/DsGameHistory';

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

	if (isLoading) {
		return (
			<div className="flex justify-center items-center h-48">
				<p className="text-gray-500 dark:text-gray-400">Loading...</p>
			</div>
		);
	}

	if (!user) {
		return <Navigate to="/login" replace />;
	}

	const formatDate = (iso?: string) => {
		if (!iso) return '—';
		return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
	};

	return (
		<div className="max-w-3xl mx-auto space-y-6">
			<h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">Profile</h2>

			<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
				{user.is_ephemeral && (
					<div className="bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700 rounded px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
						Playing as guest. This account is tied to your browser and can be lost — claim it below to keep it permanently.
					</div>
				)}
				<div>
					<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Username</p>
					<p className="text-lg font-medium text-gray-800 dark:text-gray-100">{user.username}</p>
				</div>

				{user.email && (
					<div>
						<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Email</p>
						<p className="text-gray-800 dark:text-gray-100">{user.email}</p>
					</div>
				)}

				<hr className="border-gray-200 dark:border-gray-700" />

				<div>
					<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Member Since</p>
					<p className="text-gray-800 dark:text-gray-100">{formatDate(user.created_at)}</p>
				</div>

				<div>
					<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Last Login</p>
					<p className="text-gray-800 dark:text-gray-100">{formatDate(user.last_login)}</p>
				</div>
			</div>

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
				<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
					<div>
						<h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Claim this account</h3>
						<p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
							Add an email and password to keep this account and its ratings permanently.
						</p>
					</div>
					<form onSubmit={handleClaim} className="space-y-4">
						<ErrorMessage message={error} onClear={clearError} />
						<Input
							label="Username"
							id="claim-username"
							type="text"
							value={claimUsername}
							onChange={(e) => setClaimUsername(e.target.value)}
							autoComplete="username"
							placeholder={user.username}
							disabled={isLoading}
						/>
						<Input
							label="Email Address"
							id="claim-email"
							type="email"
							value={claimEmail}
							onChange={(e) => setClaimEmail(e.target.value)}
							required
							autoComplete="email"
							placeholder="you@example.com"
							disabled={isLoading}
						/>
						<Input
							label="Password"
							id="claim-password"
							type="password"
							value={claimPassword}
							onChange={(e) => setClaimPassword(e.target.value)}
							required
							autoComplete="new-password"
							placeholder="Create a password"
							disabled={isLoading}
						/>
						<Button type="submit" isLoading={isLoading} disabled={isLoading}>
							Claim Account
						</Button>
					</form>
				</div>
			)}
		</div>
	);
};

export default ProfilePage;
