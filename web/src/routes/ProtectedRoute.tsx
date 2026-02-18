import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import LoadingSpinner from '@/components/common/LoadingSpinner';

interface ProtectedRouteProps {
	children?: React.ReactNode; // Allow wrapping components or just using Outlet
}

/**
 * A component that renders its children or an <Outlet> only if the user
 * is authenticated. Otherwise, it redirects to the /login page.
 * It also displays a loading spinner while the authentication status is being checked.
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
	const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
	const isLoading = useAuthStore((state) => state.isLoading); // Check loading state

	if (isLoading) {
		// Show loading indicator while checking auth status on initial load.
		return (
			<div className="flex items-center justify-center h-screen">
				<LoadingSpinner />
			</div>
		);
	}

	if (!isAuthenticated) {
		// Redirect to login page if not authenticated.
		return <Navigate to="/login" replace />;
	}

	// Render the wrapped children or the nested routes via Outlet.
	return children ? <>{children}</> : <Outlet />;
};

export default ProtectedRoute;