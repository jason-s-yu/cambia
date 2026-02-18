import React from 'react';
import { Link } from 'react-router-dom';
import Button from '@/components/common/Button';

/**
 * Simple 404 Not Found page displayed for invalid routes.
 */
const NotFoundPage: React.FC = () => {
	return (
		<div className="flex flex-col items-center justify-center min-h-screen text-center px-4 bg-gray-100 dark:bg-gray-900">
			<h1 className="text-6xl font-bold text-blue-600 dark:text-blue-400 mb-4">404</h1>
			<h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-2">Page Not Found</h2>
			<p className="text-gray-600 dark:text-gray-400 mb-6">
				Sorry, the page you are looking for does not exist or has been moved.
			</p>
			<Link to="/">
				<Button variant="primary">Go to Homepage</Button>
			</Link>
		</div>
	);
};

export default NotFoundPage;