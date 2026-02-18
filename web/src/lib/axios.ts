// src/lib/axios.ts
import axios, { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/stores/authStore';
import type { ApiErrorResponse } from '@/types';

// Create an Axios instance with default configuration
const api = axios.create({
	baseURL: import.meta.env.VITE_API_URL, // Use environment variable for base API URL
	withCredentials: true, // Ensure cookies (like auth_token) are sent with requests
	headers: {
		'Content-Type': 'application/json',
		'Accept': 'application/json',
	},
	timeout: 10000, // Set a request timeout (e.g., 10 seconds)
});

// --- Request Interceptor ---
// Can be used to add tokens to headers if not using cookies
api.interceptors.request.use(
	(config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
		// Modify config here if needed, e.g., add Authorization header
		// const token = useAuthStore.getState().token; // Example if token were in store
		// if (token) {
		//   config.headers.Authorization = `Bearer ${token}`;
		// }
		return config;
	},
	(error: AxiosError) => {
		// Handle request errors (e.g., network issue before sending)
		console.error('Axios request error:', error);
		return Promise.reject(error);
	}
);


// --- Response Interceptor ---
// Handles common response errors, particularly authentication issues (401/403).
api.interceptors.response.use(
	(response) => response, // Pass through successful responses (status 2xx)
	(error: AxiosError<ApiErrorResponse>) => {
		const { response } = error;

		if (response && (response.status === 401 || response.status === 403)) {
			// Unauthorized or Forbidden response.
			// Trigger logout action via Zustand store.
			// Avoid triggering logout if the error came from a logout attempt itself.
			const requestedUrl = error.config?.url ?? '';
			if (!requestedUrl.endsWith('/user/logout')) { // Adjust if your logout endpoint differs
				console.error(`Authentication error (${response.status}) on ${requestedUrl}. Logging out. Message: ${response.data?.message ?? 'N/A'}`);
				useAuthStore.getState().logout(); // Access Zustand store outside React component
				// Optionally redirect using window.location or let UI handle redirect based on auth state
				// Example: window.location.assign('/login');
			}
		} else if (response) {
			// Handle other known server errors (e.g., 400 Bad Request, 404 Not Found, 5xx Server Errors)
			console.error(`HTTP error ${response.status} on ${error.config?.url ?? ''}:`, response.data?.message ?? error.message);
		} else if (error.request) {
			// Handle network errors where no response was received
			console.error(`Network error or no response on ${error.config?.url ?? ''}:`, error.message);
		} else {
			// Handle errors during request setup
			console.error('Axios setup error:', error.message);
		}

		// Always return the rejected promise so calling code (e.g., in services or components)
		// can still handle the error specifically if needed (e.g., display specific messages).
		return Promise.reject(error);
	}
);

export default api;