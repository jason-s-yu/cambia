import React from 'react';
import RegisterForm from '@/components/auth/RegisterForm';
import { Link } from 'react-router-dom';

/**
 * Page component for user registration. Displays the RegisterForm
 * and provides a link to the login page.
 */
const RegisterPage: React.FC = () => {
	return (
		<div>
			<RegisterForm />
			<p className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
				Already have an account?{' '}
				<Link to="/login" className="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300">
					Login here
				</Link>
			</p>
		</div>
	);
};

export default RegisterPage;