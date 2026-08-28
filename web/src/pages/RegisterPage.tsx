import React from 'react';
import { Link } from 'react-router-dom';
import RegisterForm from '@/components/auth/RegisterForm';

/**
 * Registration page: heading, the RegisterForm, and the path back to sign-in.
 * Rendered inside AuthLayout's centered card.
 */
const RegisterPage: React.FC = () => {
	return (
		<div>
			<h1 className='m-0 mb-5 text-ds-xl font-ds-bold tracking-ds-tight leading-ds-tight text-text-primary'>Create account</h1>
			<RegisterForm />
			<p className='mt-6 mb-0 text-center text-ds-sm text-text-secondary'>
				Have an account?{' '}
				<Link to='/login' className='font-ds-medium text-text-primary underline underline-offset-4'>
					Sign in
				</Link>
			</p>
		</div>
	);
};

export default RegisterPage;
