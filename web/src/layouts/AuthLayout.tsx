// src/layouts/AuthLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import ThemeToggle from '@/components/common/ThemeToggle';
import Wordmark from '@/components/ds/chrome/Wordmark';

/**
 * Layout for the entry pages (sign in, register): a single centered card on
 * the app ground, the wordmark above the form, and the theme toggle in the
 * top corner. Flat: the card separates by the surface step and a 1px border.
 */
const AuthLayout: React.FC = () => {
	return (
		<div className='relative min-h-screen flex flex-col items-center justify-center bg-surface-0 text-text-primary p-4 sm:p-6'>
			<div className='absolute top-4 right-4'>
				<ThemeToggle />
			</div>

			<div className='w-full max-w-md bg-surface-1 border border-border-default rounded-ds-lg p-6 sm:p-8'>
				<div className='flex justify-center mb-6'>
					<Wordmark size={28} />
				</div>
				<Outlet />
			</div>
		</div>
	);
};

export default AuthLayout;
