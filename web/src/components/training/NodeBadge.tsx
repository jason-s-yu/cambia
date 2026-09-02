// src/components/training/NodeBadge.tsx
import React from 'react';

interface NodeBadgeProps {
	/** Which nashnet node produced this run's numbers (serving-harness v1.1
	 * design D23); a run the pool never touched passes undefined/empty and the
	 * badge renders nothing. */
	executedOn?: string;
	className?: string;
}

// NodeBadge marks a run with the nashnet node that executed it, distinct from
// HostBadge (which host owns/serves the run). It renders for a pool run
// executed by the coordinator's own embedded node as well as a genuinely
// remote one -- both carry executed_on -- and renders nothing for a run the
// pool never touched, so it can be mounted unconditionally next to HostBadge.
const NodeBadge: React.FC<NodeBadgeProps> = ({ executedOn, className = '' }) => {
	if (!executedOn) return null;
	return (
		<span
			title={`Executed on nashnet node ${executedOn}`}
			className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200 ${className}`}
		>
			<span className="inline-block w-1.5 h-1.5 rounded-full bg-teal-500" />
			{executedOn}
		</span>
	);
};

export default NodeBadge;
