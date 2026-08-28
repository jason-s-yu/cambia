import React from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '@/components/ds/core/Button';
import Wordmark from '@/components/ds/chrome/Wordmark';

/**
 * 404 page for unmatched routes. Rendered outside both layouts, so it paints
 * its own ground and carries the wordmark.
 */
const NotFoundPage: React.FC = () => {
	const navigate = useNavigate();
	return (
		<div
			style={{
				minHeight: '100vh',
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'center',
				justifyContent: 'center',
				gap: 'var(--space-4)',
				padding: 'var(--space-6)',
				textAlign: 'center',
				background: 'var(--surface-0)',
				color: 'var(--text-primary)'
			}}
		>
			<Wordmark size={20} style={{ color: 'var(--text-secondary)' }} />
			<div
				style={{
					fontSize: 'var(--ds-text-5xl)',
					fontWeight: 'var(--weight-black)',
					letterSpacing: 'var(--ds-tracking-tight)',
					lineHeight: 1,
					fontVariantNumeric: 'tabular-nums',
					color: 'var(--accent-gold)'
				}}
			>
				404
			</div>
			<h1 style={{ margin: 0, fontSize: 'var(--ds-text-xl)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>
				Page not found
			</h1>
			<p style={{ margin: 0, maxWidth: 360, color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>
				That page does not exist or has moved.
			</p>
			<Button variant='primary' onClick={() => navigate('/')} style={{ marginTop: 'var(--space-2)' }}>
				Back to home
			</Button>
		</div>
	);
};

export default NotFoundPage;
