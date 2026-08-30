// src/layouts/AppLayout.tsx
import React, { useEffect, useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import TopBar, { type TopBarNavItem, type TopBarUser } from '@/components/ds/chrome/TopBar';
import { useAuthStore } from '@/stores/authStore';
import { useUiStore } from '@/stores/uiStore';
import { useHistoryStore } from '@/stores/historyStore';
import { formatRating } from '@/utils/ratingPool';

/** Primary navigation for authenticated users. */
const NAV_ITEMS: TopBarNavItem[] = [
	{ label: 'Home', path: '/dashboard' },
	{ label: 'Leaderboard', path: '/leaderboard' },
	{ label: 'Training', path: '/training' },
	{ label: 'Profile', path: '/profile' }
];

/**
 * Main application layout for authenticated users, built on the design-system
 * chrome (TopBar + Wordmark). The TopBar carries navigation, the theme toggle,
 * the identity chip and log out; nested routes render full-bleed into the
 * Outlet and manage their own padding, matching the design-system screen
 * contract.
 */
const AppLayout: React.FC = () => {
	const navigate = useNavigate();
	const location = useLocation();
	const logout = useAuthStore((state) => state.logout);
	const user = useAuthStore((state) => state.user);
	const theme = useUiStore((state) => state.theme);
	const setTheme = useUiStore((state) => state.setTheme);
	const ratings = useHistoryStore((state) => state.ratings);
	const fetchRatings = useHistoryStore((state) => state.fetchRatings);

	// AppLayout wraps every authenticated route and stays mounted across
	// navigation, so this is the single place the app fetches the caller's
	// ratings for the session: once on login/account-switch (keyed on user id),
	// not on every render. Nested pages (DashboardPage) read the same
	// historyStore state rather than fetching again.
	const userId = user?.id;
	useEffect(() => {
		if (!userId) return;
		fetchRatings();
	}, [userId, fetchRatings]);

	// Effective light flag for the theme toggle, resolving 'system' against
	// the OS preference so the toggle reflects what is actually on screen.
	const [light, setLight] = useState(false);
	useEffect(() => {
		const mq = window.matchMedia('(prefers-color-scheme: dark)');
		const resolve = () => setLight(theme === 'light' ? true : theme === 'dark' ? false : !mq.matches);
		resolve();
		mq.addEventListener('change', resolve);
		return () => mq.removeEventListener('change', resolve);
	}, [theme]);

	// A pinned tab's logout only drops its own token and re-probes the shared
	// cookie (authStore.logout, cambia-1149): that re-probe can land signed in
	// as whoever the cookie belongs to. Routing to /login unconditionally threw
	// that answer away, so a reload came back on the dashboard as the cookie's
	// guest instead of the login screen the tab had just navigated to. The
	// store's post-logout state is what /login is actually gated on: navigate
	// there only when it says signed out.
	const handleLogout = async () => {
		await logout();
		if (!useAuthStore.getState().isAuthenticated) {
			navigate('/login');
		}
	};

	// Headline rating: the 1v1 (head-to-head) pool, matching the dashboard hero
	// and formatted the same way (utils/ratingPool) as the profile page so the
	// number agrees everywhere it appears. 'Unrated' is a fact about the player
	// and is only claimed once the ratings actually arrive; while the fetch is
	// pending or failed the chip says nothing (cambia-876, DL-2 review F3).
	const headlinePool = ratings?.pools.find((p) => p.pool === '1v1') ?? null;
	const topBarUser: TopBarUser = {
		name: user?.username || 'Player',
		rating: !ratings
			? '--'
			: headlinePool && headlinePool.games > 0
				? formatRating(headlinePool.rating, headlinePool.rd)
				: 'Unrated'
	};

	return (
		<div
			style={{
				minHeight: '100vh',
				display: 'flex',
				flexDirection: 'column',
				background: 'var(--surface-0)',
				color: 'var(--text-primary)'
			}}
		>
			<TopBar
				items={NAV_ITEMS}
				activePath={location.pathname}
				onNav={(path) => navigate(path)}
				light={light}
				onToggleTheme={(nextLight) => setTheme(nextLight ? 'light' : 'dark')}
				user={topBarUser}
				onLogout={handleLogout}
			/>
			<main style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
				<Outlet />
			</main>
		</div>
	);
};

export default AppLayout;
