// src/pages/ProfilePage.tsx
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

const ProfilePage: React.FC = () => {
	const user = useAuthStore((state) => state.user);
	const isLoading = useAuthStore((state) => state.isLoading);

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

	const glickoRating = user.elo != null
		? `${Math.round(user.elo)} ± ${Math.round(user.rd ?? 0)}`
		: '—';

	const openSkillRating = user.open_skill_mu != null
		? `${user.open_skill_mu.toFixed(2)} ± ${(user.open_skill_sigma ?? 0).toFixed(2)}`
		: '—';

	return (
		<div className="max-w-lg mx-auto space-y-6">
			<h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">Profile</h2>

			<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
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
					<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Glicko-2 Rating</p>
					<p className="text-gray-800 dark:text-gray-100">{glickoRating}</p>
				</div>

				<div>
					<p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">OpenSkill Rating</p>
					<p className="text-gray-800 dark:text-gray-100">{openSkillRating}</p>
				</div>

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
		</div>
	);
};

export default ProfilePage;
