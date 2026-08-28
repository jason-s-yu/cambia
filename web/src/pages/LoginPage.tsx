import React from 'react';
import { Link } from 'react-router-dom';
import LoginForm from '@/components/auth/LoginForm';

/**
 * Sign-in page: heading, the LoginForm, and the path to registration.
 * Rendered inside AuthLayout's centered card.
 */
const LoginPage: React.FC = () => {
	return (
		<div>
			<h1 className='m-0 mb-5 text-ds-xl font-ds-bold tracking-ds-tight leading-ds-tight text-text-primary'>Sign in</h1>
			<LoginForm />
			<p className='mt-6 mb-0 text-center text-ds-sm text-text-secondary'>
				No account?{' '}
				<Link to='/register' className='font-ds-medium text-text-primary underline underline-offset-4'>
					Register
				</Link>
			</p>
		</div>
	);
};

export default LoginPage;
