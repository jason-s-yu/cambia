// src/components/training/StatusBadge.tsx
import React from 'react';
import type { ProcessStatus } from '@/types/training';

const STATUS_STYLES: Record<ProcessStatus, string> = {
	created: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200',
	starting: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200',
	running: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-200',
	stopping: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-200',
	stopped: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400',
	crashed: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200',
};

const STATUS_DOT: Record<ProcessStatus, string> = {
	created: 'bg-gray-400',
	starting: 'bg-blue-500 animate-pulse',
	running: 'bg-green-500',
	stopping: 'bg-yellow-500 animate-pulse',
	stopped: 'bg-gray-400',
	crashed: 'bg-red-500',
};

interface StatusBadgeProps {
	status: ProcessStatus;
	className?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
	const style = STATUS_STYLES[status] ?? STATUS_STYLES.created;
	const dot = STATUS_DOT[status] ?? STATUS_DOT.created;

	return (
		<span
			className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium ${style} ${className}`}
		>
			<span className={`inline-block w-1.5 h-1.5 rounded-full ${dot}`} />
			{status}
		</span>
	);
};

export default StatusBadge;
