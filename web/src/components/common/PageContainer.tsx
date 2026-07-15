import React from 'react';

interface PageContainerProps {
	children: React.ReactNode;
	className?: string;
}

/**
 * Self-owned max-width + padding wrapper for pages rendered inside the
 * full-bleed ds-chrome AppLayout (cambia-489). The rebuilt AppLayout no
 * longer applies container/mx-auto padding around <Outlet>, so pages that
 * need a bounded reading width (tables, forms) opt in here rather than
 * relying on the shell. Padding/max-width match the ds screens' single-column
 * convention (see src/pages/ds/LeaderboardScreen.tsx).
 */
const PageContainer: React.FC<PageContainerProps> = ({ children, className = '' }) => (
	<div className={`max-w-[1200px] w-full mx-auto p-[22px] ${className}`}>
		{children}
	</div>
);

export default PageContainer;
