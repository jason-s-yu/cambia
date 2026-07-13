// src/pages/LeaderboardPage.tsx
import React from 'react';
import Panel from '@/components/ds/chrome/Panel';

/**
 * Placeholder leaderboard page. The full standings view is built on a parallel
 * branch; this trivial shell only reserves the /leaderboard route so the shared
 * shell builds standalone.
 */
const LeaderboardPage: React.FC = () => {
	return (
		<div style={{ padding: 22, maxWidth: 1240, margin: '0 auto', width: '100%' }}>
			<Panel title="Leaderboard">
				<h1
					style={{
						margin: 0,
						fontFamily: 'var(--font-display)',
						fontSize: 'var(--ds-text-2xl)',
						fontWeight: 'var(--weight-regular)',
						color: 'var(--text-primary)'
					}}
				>
					Leaderboard
				</h1>
			</Panel>
		</div>
	);
};

export default LeaderboardPage;
