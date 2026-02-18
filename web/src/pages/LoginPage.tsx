import React from 'react';
import LoginForm from '@/components/auth/LoginForm';
import { Link } from 'react-router-dom';

/**
 * Page component for user login. Displays the LoginForm
 * and provides a link to the registration page.
 */
const LoginPage: React.FC = () => {
	return (
		<div>
			<LoginForm />
			<p className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
				Don&apos;t have an account?{' '}
				<Link to="/register" className="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300">
					Register here
				</Link>
			</p>
		</div>
	);
};

export default LoginPage;