// src/components/training/HostBadge.tsx
import React from 'react';

interface HostBadgeProps {
	/** Origin host of a remote run. A local run passes undefined/empty and the
	 * badge renders nothing. */
	host?: string;
	className?: string;
}

// HostBadge marks a run synced from another machine (the serving harness) with
// its origin host. Local runs carry no host and render nothing, so the badge can
// be mounted unconditionally next to a run name.
const HostBadge: React.FC<HostBadgeProps> = ({ host, className = '' }) => {
	if (!host) return null;
	return (
		<span
			title={`Remote run synced from ${host} (read-only on this dashboard)`}
			className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200 ${className}`}
		>
			<span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-500" />
			{host}
		</span>
	);
};

export default HostBadge;
