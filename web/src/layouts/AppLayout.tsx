// src/layouts/AppLayout.tsx
import React, { useEffect, useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import TopBar, { type TopBarNavItem, type TopBarUser } from '@/components/ds/chrome/TopBar';
import { useAuthStore } from '@/stores/authStore';
import { useUiStore } from '@/stores/uiStore';

/** Primary navigation for authenticated users. */
const NAV_ITEMS: TopBarNavItem[] = [
	{ label: 'Home', path: '/dashboard' },
	{ label: 'Leaderboard', path: '/leaderboard' },
	{ label: 'Training', path: '/training' },
	{ label: 'Profile', path: '/profile' }
];

/**
 * Main application layout for authenticated users, built on the design-system
 * chrome (TopBar + Wordmark). The TopBar carries navigation, the theme switch,
 * the profile chip and logout; nested routes render full-bleed into the Outlet
 * and manage their own padding, matching the design-system screen contract.
 */
const AppLayout: React.FC = () => {
	const navigate = useNavigate();
	const location = useLocation();
	const logout = useAuthStore((state) => state.logout);
	const user = useAuthStore((state) => state.user);
	const theme = useUiStore((state) => state.theme);
	const setTheme = useUiStore((state) => state.setTheme);

	// Effective light flag for the day/night switch, resolving 'system' against
	// the OS preference so the switch reflects what is actually on screen.
	const [light, setLight] = useState(false);
	useEffect(() => {
		const mq = window.matchMedia('(prefers-color-scheme: dark)');
		const resolve = () => setLight(theme === 'light' ? true : theme === 'dark' ? false : !mq.matches);
		resolve();
		mq.addEventListener('change', resolve);
		return () => mq.removeEventListener('change', resolve);
	}, [theme]);

	const handleLogout = async () => {
		await logout();
		navigate('/login');
	};

	const topBarUser: TopBarUser = {
		name: user?.username || 'Player',
		rating: user?.elo !== undefined ? `${Math.round(user.elo)} ± ${Math.round(user.rd ?? 0)}` : 'unrated'
	};

	return (
		<div
			style={{
				minHeight: '100vh',
				display: 'flex',
				flexDirection: 'column',
				background: 'var(--surface-page)',
				color: 'var(--text-primary)',
				fontFamily: 'var(--font-ui)'
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
