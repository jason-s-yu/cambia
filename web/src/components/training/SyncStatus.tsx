// src/components/training/SyncStatus.tsx
import React from 'react';
import { format, formatDistanceToNow, parseISO } from 'date-fns';

interface SyncStatusProps {
	/** Origin host of a remote run; a local run passes undefined and nothing
	 * renders. */
	host?: string;
	/** RFC3339 timestamp of the last successful pull. */
	lastSyncAt?: string;
	/** Server-computed staleness (now - last_sync_at past 3 sync intervals). */
	stale?: boolean;
	className?: string;
}

// SyncStatus renders per-run pull freshness for a remote run next to its status
// badge (design 4.5). Past 3 sync intervals the server flags the run stale and
// this degrades to "stale, last synced HH:MM"; while fresh it shows the sync age.
// A local run (no host) or a remote run never yet synced renders nothing.
const SyncStatus: React.FC<SyncStatusProps> = ({ host, lastSyncAt, stale, className = '' }) => {
	if (!host || !lastSyncAt) return null;

	let synced: Date;
	try {
		synced = parseISO(lastSyncAt);
	} catch {
		return null;
	}

	if (stale) {
		return (
			<span
				title={`Last synced ${lastSyncAt}`}
				className={`inline-flex items-center gap-1 text-xs font-medium text-red-600 dark:text-red-400 ${className}`}
			>
				<span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500" />
				stale, last synced {format(synced, 'HH:mm')}
			</span>
		);
	}

	return (
		<span
			title={`Last synced ${lastSyncAt}`}
			className={`inline-flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 ${className}`}
		>
			<span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400" />
			synced {formatDistanceToNow(synced, { addSuffix: true })}
		</span>
	);
};

export default SyncStatus;
