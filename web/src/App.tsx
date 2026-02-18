// src/App.tsx
import { Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import DashboardPage from '@/pages/DashboardPage';
import LobbyPage from '@/pages/LobbyPage';
import GamePage from '@/pages/GamePage';
import AuthLayout from '@/layouts/AuthLayout';
import AppLayout from '@/layouts/AppLayout';
import ProtectedRoute from '@/routes/ProtectedRoute';
import { useAuthStore } from '@/stores/authStore';
import NotFoundPage from './pages/NotFoundPage';
import { useInitializeAuth } from '@/hooks/useInitializeAuth';
import LoadingSpinner from './components/common/LoadingSpinner';
import { useTheme } from './hooks/useTheme';

function App() {
	const { isLoading } = useInitializeAuth(); // Check auth status on load
	const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
	useTheme(); // Initialize and manage theme

	if (isLoading) {
		return (
			<div className='flex items-center justify-center h-screen'>
				<LoadingSpinner />
			</div>
		);
	}

	return (
		<Routes>
			{/* Public Routes */}
			<Route element={<AuthLayout />}>
				<Route path="/login" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <LoginPage />} />
				<Route path="/register" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <RegisterPage />} />
			</Route>

			{/* Protected Routes */}
			<Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
				<Route path="/dashboard" element={<DashboardPage />} />
				<Route path="/lobby/:lobbyId" element={<LobbyPage />} />
				<Route path="/game/:gameId" element={<GamePage />} />
			</Route>

			{/* Redirect root path */}
			<Route path="/" element={<Navigate to={isAuthenticated ? '/dashboard' : '/login'} replace />} />

			{/* Fallback 404 Route */}
			<Route path="*" element={<NotFoundPage />} />
		</Routes>
	);
}

export default App;