// src/layouts/AuthLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import ThemeToggle from '@/components/common/ThemeToggle';

/**
 * Layout specifically for authentication pages (Login, Register).
 * Provides a centered card structure with branding and theme toggle.
 */
const AuthLayout: React.FC = () => {
	return (
		<div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-800 dark:via-gray-900 dark:to-indigo-900 p-4 transition-colors duration-200">
			{/* Theme toggle positioned top-right */}
			<div className="absolute top-4 right-4">
				<ThemeToggle />
			</div>

			<div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 sm:p-8">
				<h1 className="text-3xl font-bold text-center text-blue-600 dark:text-blue-400 mb-6">
					Cambia Online
				</h1>
				{/* Login or Register form renders here */}
				<Outlet />
			</div>

			<footer className="text-center text-xs text-gray-500 dark:text-gray-400 p-2 mt-4">
				© 2025
			</footer>
		</div>
	);
};

export default AuthLayout;