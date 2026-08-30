/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/axios';
import { isPinned, pinTab } from '@/lib/tabSession';
import { loginToTab, mintTabGuest } from '@/services/devSessionService';
import type { User } from '@/types'; // Import common types

// Define interfaces for API request bodies inline or import if defined elsewhere
interface LoginCredentials {
	email: string;
	password?: string; // Password might be optional for guest->claimed flow later?
}

interface RegisterDetails {
	email: string;
	password?: string; // Allow potential passwordless registration for guests?
	username: string;
}

/**
 * Attempts to log in a user via the backend API.
 * @param credentials User's email and password.
 * @returns True if login is successful (token cookie set by backend), false otherwise.
 * @throws {Error} If the API request fails (e.g., network error, 500). Axios interceptor handles 401/403.
 */
export const loginUser = async (credentials: LoginCredentials): Promise<boolean> => {
	try {
		// A tab already pinned to its own identity signs in as that tab, not as
		// the browser: the request carries the tab-session header, the service
		// answers with a token and no Set-Cookie, and the new token replaces the
		// pin (cambia-1149). Without this the form would set the shared cookie
		// (signing every other tab in as this account) while the tab kept its old
		// token, so the login would look like it had done nothing.
		if (isPinned()) {
			const identity = await loginToTab(credentials.email, credentials.password ?? '');
			pinTab(identity.token, credentials.email.split('@')[0] || identity.label);
			return true;
		}
		const response = await api.post<{ token: string }>('/user/login', credentials);
		// Backend sets HttpOnly cookie, presence of token in response body confirms success.
		return !!response.data.token;
	} catch (error: any) {
		console.error('Login API call failed:', error.response?.data || error.message);
		// Rethrow the error so the calling store action can catch it and display a message.
		throw error;
	}
};

/**
 * Registers a new user via the backend API.
 * @param details User's email, password, and username.
 * @returns The newly created User object upon success.
 * @throws {Error} If the API request fails (e.g., 409 Conflict, 500).
 */
export const registerUser = async (details: RegisterDetails): Promise<User | null> => {
	try {
		const response = await api.post<User>('/user/create', details);
		return response.data;
	} catch (error: any) {
		console.error('Register API call failed:', error.response?.data || error.message);
		// Rethrow the error so the calling store action can catch it and display a message.
		throw error;
	}
};

/**
 * Logs out the current user by clearing the auth_token cookie on the server.
 */
export const logoutUser = async (): Promise<void> => {
	await api.post('/user/logout');
};

/**
 * Creates an ephemeral guest session via the backend API.
 * The backend sets the `auth_token` cookie; the response body only carries the
 * new user's id, so callers must follow up with fetchMe() to hydrate the full
 * user record (username, is_ephemeral, etc.).
 * @returns The new guest user's id.
 * @throws {Error} If the API request fails.
 */
export const guestLogin = async (): Promise<{ id: string }> => {
	try {
		// Same rule as loginUser: a pinned tab takes its guest as a token, so the
		// cookie the other tabs share is left alone (cambia-1149).
		if (isPinned()) {
			const guest = await mintTabGuest();
			pinTab(guest.token, guest.label);
			return { id: guest.id };
		}
		const response = await api.post<{ id: string }>('/user/guest');
		return response.data;
	} catch (error: any) {
		console.error('Guest login API call failed:', error.response?.data || error.message);
		throw error;
	}
};

interface ClaimDetails {
	email: string;
	password: string;
	username?: string;
}

/**
 * Converts the currently authenticated ephemeral (guest) user into a
 * persistent account by attaching email and password credentials.
 * Relies on the auth_token cookie identifying the ephemeral user.
 * @throws {Error} If the API request fails (e.g., account already claimed, email in use).
 */
export const claimAccount = async (details: ClaimDetails): Promise<void> => {
	try {
		await api.post('/user/claim', details);
	} catch (error: any) {
		console.error('Claim account API call failed:', error.response?.data || error.message);
		throw error;
	}
};

/**
 * Fetches the currently authenticated user's details from the backend.
 * Relies on the auth_token cookie being sent automatically by the browser.
 * @returns The User object if authenticated, null otherwise (e.g., 401/403 response).
 * @throws {Error} If the API request fails with non-auth errors (e.g., network error, 500).
 */
export const fetchMe = async (): Promise<User | null> => {
	try {
		const response = await api.get<User>('/user/me');
		return response.data;
	} catch (error: any) {
		// Axios interceptor should handle 401/403 by logging out.
		// If it's another error, log it but return null as user is not fetched.
		if (error.response?.status !== 401 && error.response?.status !== 403) {
			console.error('Fetch /user/me API call failed:', error.response?.data || error.message);
		}
		// We don't rethrow here because a 401/403 is expected for non-logged-in users.
		return null;
	}
};