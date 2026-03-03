import React, { useEffect, useState } from 'react';
import type { QueueInfo } from '@/services/matchmakingService';

const QUEUE_DISPLAY_NAMES: Record<string, string> = {
  h2h_quickplay: 'Quick Play',
  h2h_blitz: 'H2H Blitz',
  h2h_rapid: 'H2H Rapid',
  h2h_classical: 'H2H Classical',
  ffa4_standard: 'FFA-4 Standard',
  ffa4_classical: 'FFA-4 Classical',
};

interface QueueCardProps {
  queue: QueueInfo;
  userRating?: { mu: number; rd: number; tier: string };
  isSearching: boolean;
  searchTime?: number;
  onClick: () => void;
  onCancel: () => void;
}

const QueueCard: React.FC<QueueCardProps> = ({
  queue,
  userRating,
  isSearching,
  searchTime = 0,
  onClick,
  onCancel,
}) => {
  const [elapsed, setElapsed] = useState(searchTime);

  useEffect(() => {
    if (!isSearching) {
      setElapsed(0);
      return;
    }
    setElapsed(searchTime);
    const interval = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isSearching, searchTime]);

  const displayName = QUEUE_DISPLAY_NAMES[queue.queueId] ?? queue.name;
  const estimatedMinutes = Math.ceil((queue.players * queue.rounds * 3) / 60);

  const formatElapsed = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  return (
    <div
      className={`rounded-xl border bg-white dark:bg-gray-800 shadow-sm transition-all ${
        isSearching
          ? 'border-blue-400 dark:border-blue-500'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md'
      }`}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-3">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">{displayName}</h3>
        <div className="flex flex-wrap gap-2 mt-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            {queue.players} Players
          </span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            Bo{queue.rounds}
          </span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            ~{estimatedMinutes} min
          </span>
          {queue.ranked && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
              Ranked
            </span>
          )}
        </div>
      </div>

      {/* Rating section */}
      {queue.ranked && (
        <div className="px-5 py-2 border-t border-gray-100 dark:border-gray-700">
          {queue.hiddenRating ? (
            <span className="text-sm text-gray-500 dark:text-gray-400 italic">Hidden MMR</span>
          ) : userRating ? (
            <div className="flex items-center gap-2">
              <span
                className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-700 text-white"
              >
                {userRating.tier}
              </span>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {Math.round(userRating.mu)} ± {Math.round(userRating.rd)}
              </span>
            </div>
          ) : (
            <span className="text-sm text-gray-500 dark:text-gray-400">Unplaced</span>
          )}
        </div>
      )}

      {/* Queue stats */}
      <div className="px-5 py-2 text-xs text-gray-400 dark:text-gray-500">
        {queue.playerCount} in queue &bull; ~{queue.avgWaitSec}s wait
      </div>

      {/* Footer */}
      <div className="px-5 pb-5 pt-2">
        {isSearching ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              <span>Searching... {formatElapsed(elapsed)}</span>
            </div>
            <button
              onClick={onCancel}
              className="w-full py-2 px-4 rounded-lg bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 text-sm font-medium hover:bg-red-200 dark:hover:bg-red-800 transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={onClick}
            className="w-full py-2 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition-colors"
          >
            Find Match
          </button>
        )}
      </div>
    </div>
  );
};

export default QueueCard;
