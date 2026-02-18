// src/layouts/AppLayout.tsx
import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import Button from '@/components/common/Button';
import ThemeToggle from '@/components/common/ThemeToggle';

/**
 * Main application layout for authenticated users.
 * Includes a header with navigation, user info, theme toggle, logout button,
 * the main content area rendered by <Outlet>, and a simple footer.
 */
const AppLayout: React.FC = () => {
	const logout = useAuthStore((state) => state.logout);
	const user = useAuthStore((state) => state.user);

	return (
		<div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
			<header className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-20">
				<nav className="container mx-auto px-4 py-3 flex justify-between items-center">
					<Link to="/dashboard" className="text-xl font-bold text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors">
						Cambia Online
					</Link>
					<div className='flex items-center space-x-4'>
						<span className="text-sm text-gray-700 dark:text-gray-300 hidden sm:inline">
							Welcome, <strong className='font-medium'>{user?.username ?? 'Player'}</strong>!
						</span>
						<ThemeToggle />
						<Button onClick={logout} variant='secondary' size='sm'>
							Logout
						</Button>
					</div>
				</nav>
			</header>

			{/* Main content area where nested routes are rendered */}
			<main className="flex-grow container mx-auto p-4 md:p-6 lg:p-8">
				<Outlet />
			</main>

			<footer className="text-center text-xs text-gray-500 dark:text-gray-400 p-3 bg-gray-100 dark:bg-gray-800 border-t dark:border-gray-700">
				Cambia Client v0.1.0 - © 2025
			</footer>
		</div>
	);
};

export default AppLayout;