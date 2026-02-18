// src/hooks/useInitializeAuth.ts
import { useEffect } from 'react';
import { useAuthStore } from '@/stores/authStore';

/**
 * Checks authentication status via the auth store when the application loads.
 * It triggers the `checkAuth` action only if the state is currently loading
 * and the user isn't already marked as authenticated.
 *
 * @returns An object containing the current `isLoading` status from the auth store.
 */
export function useInitializeAuth() {
	const checkAuth = useAuthStore((state) => state.checkAuth);
	const isLoading = useAuthStore((state) => state.isLoading);
	const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

	useEffect(() => {
		// Only check auth if not already authenticated and store indicates initial loading.
		// This prevents redundant checks after login/logout actions manage state directly.
		if (!isAuthenticated && isLoading) {
			checkAuth();
		}
	}, [checkAuth, isAuthenticated, isLoading]); // Dependencies ensure effect runs only when these change

	return { isLoading };
}