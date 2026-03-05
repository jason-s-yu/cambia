// src/components/training/CheckpointTable.tsx
import React from 'react';
import { formatDistanceToNow, parseISO } from 'date-fns';
import type { Checkpoint } from '@/types/training';

interface CheckpointTableProps {
	checkpoints: Checkpoint[];
}

function formatFileSize(bytes: number | null): string {
	if (bytes == null) return '--';
	if (bytes < 1024) return `${bytes} B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
	return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const CheckpointTable: React.FC<CheckpointTableProps> = ({ checkpoints }) => {
	const sorted = [...checkpoints].sort((a, b) => b.iteration - a.iteration);

	if (sorted.length === 0) {
		return (
			<p className="text-gray-500 dark:text-gray-400 text-sm">No checkpoints recorded.</p>
		);
	}

	return (
		<div className="overflow-x-auto">
			<table className="min-w-full text-sm">
				<thead>
					<tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-400">
						<th className="py-2 pr-4 font-medium">Iteration</th>
						<th className="py-2 pr-4 font-medium">File Size</th>
						<th className="py-2 pr-4 font-medium">Created</th>
						<th className="py-2 pr-4 font-medium">Best</th>
					</tr>
				</thead>
				<tbody>
					{sorted.map((cp) => (
						<tr
							key={cp.id}
							className="border-b border-gray-100 dark:border-gray-800 text-gray-800 dark:text-gray-200"
						>
							<td className="py-2 pr-4 font-mono">{cp.iteration}</td>
							<td className="py-2 pr-4">{formatFileSize(cp.file_size_bytes)}</td>
							<td className="py-2 pr-4 text-gray-500 dark:text-gray-400">
								{formatDistanceToNow(parseISO(cp.created_at), { addSuffix: true })}
							</td>
							<td className="py-2 pr-4">
								{cp.is_best && (
									<span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
										Best
									</span>
								)}
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
};

export default CheckpointTable;
