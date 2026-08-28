// src/hooks/useInitializeAuth.ts
import { useEffect } from 'react';
import { useAuthStore } from '@/stores/authStore';

/**
 * Runs the initial authentication check once, when the application loads.
 *
 * @returns `initialised`: false until the first check settles. Callers gate the
 * app shell on this, never on `isLoading`, which every later auth call raises
 * too (cambia-876).
 */
export function useInitializeAuth() {
	const checkAuth = useAuthStore((state) => state.checkAuth);
	const initialised = useAuthStore((state) => state.initialised);

	useEffect(() => {
		// Gated on `initialised`, the flag the first settled check sets, not on
		// `isLoading`. Every later auth call raises isLoading too, so this effect
		// re-fired during a login POST and sent a GET /user/me alongside it; the
		// probe settled first, cleared isLoading while the POST was still in
		// flight, and the form's submit button came back to life mid-request
		// (cambia-914, DL-8 R6).
		if (!initialised) {
			checkAuth();
		}
	}, [checkAuth, initialised]);

	return { initialised };
}