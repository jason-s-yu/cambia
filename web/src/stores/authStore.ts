/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/authStore.ts
import { create } from 'zustand';
import type { User } from '@/types';
import { loginUser, registerUser, fetchMe } from '@/services/authService';

interface LoginCredentials {
	email: string;
	password?: string;
}

interface RegisterDetails {
	email: string;
	password?: string;
	username: string;
}

interface AuthState {
	isAuthenticated: boolean;
	user: User | null;
	isLoading: boolean; // Tracks initial auth check and ongoing auth operations
	error: string | null;
	login: (credentials: LoginCredentials) => Promise<boolean>;
	register: (details: RegisterDetails) => Promise<boolean>;
	logout: () => void;
	checkAuth: () => Promise<void>; // Action to verify authentication status on app load
	setUser: (user: User | null) => void; // Action to directly set user data (e.g., after WS guest creation)
	clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
	// Optionally persist part of the auth state (e.g., isAuthenticated, user)
	// to localStorage. Be mindful of security implications of storing user data locally.
	// persist(
	(set, get) => ({
		isAuthenticated: false,
		user: null,
		isLoading: true, // Assume loading initially until checkAuth completes
		error: null,

		login: async (credentials) => {
			set({ isLoading: true, error: null });
			try {
				const success = await loginUser(credentials);
				if (success) {
					// Fetch user details after successful login to populate the store
					await get().checkAuth();
					// Note: checkAuth sets isAuthenticated and isLoading internally
					return true;
				} else {
					// Should ideally not happen if loginUser throws on failure
					set({ error: 'Login failed. Please check credentials.', isLoading: false });
					return false;
				}
			} catch (err: any) {
				console.error('Login error:', err);
				// Display error message from API response or a generic one
				set({ error: err.response?.data?.message || 'An unknown error occurred during login.', isLoading: false });
				return false;
			}
		},

		register: async (details) => {
			set({ isLoading: true, error: null });
			try {
				const newUser = await registerUser(details);
				if (newUser) {
					// Attempt to auto-login after successful registration
					await get().login({ email: details.email, password: details.password });
					// isLoading and isAuthenticated will be set by login/checkAuth
					return true;
				} else {
					set({ error: 'Registration failed.', isLoading: false });
					return false;
				}
			} catch (err: any) {
				console.error('Registration error:', err);
				set({ error: err.response?.data?.message || 'An unknown error occurred during registration.', isLoading: false });
				return false;
			}
		},

		logout: () => {
			// TODO: Call a '/user/logout' API endpoint if implemented on backend.
			set({ isAuthenticated: false, user: null, error: null, isLoading: false });
			// If using persist middleware, ensure persisted state is cleared appropriately on logout.
		},

		checkAuth: async () => {
			set({ isLoading: true, error: null });
			try {
				const currentUser = await fetchMe();
				if (currentUser) {
					set({ isAuthenticated: true, user: currentUser, isLoading: false });
				} else {
					// No user returned (e.g., 401/403 response, or token invalid)
					set({ isAuthenticated: false, user: null, isLoading: false });
				}
			} catch (error) {
				// Catch unexpected errors during fetchMe (e.g., network issue)
				// Auth errors (401/403) are typically handled by returning null from fetchMe
				console.log('Auth check failed:', error);
				set({ isAuthenticated: false, user: null, isLoading: false });
			}
		},

		setUser: (user) => {
			set({ user, isAuthenticated: !!user });
		},

		clearError: () => {
			set({ error: null });
		}
	})
	// Example persist configuration:
	// {
	//   name: 'auth-storage',
	//   partialize: (state) => ({ isAuthenticated: state.isAuthenticated, user: state.user }),
	// }
	// )
);