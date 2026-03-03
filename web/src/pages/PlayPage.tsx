import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { getQueues, startSearch, cancelSearch } from '@/services/matchmakingService';
import type { QueueInfo } from '@/services/matchmakingService';
import QueueCard from '@/components/common/QueueCard';
import RatingDisplay from '@/components/dashboard/RatingDisplay';
import type { RatingInfo } from '@/components/dashboard/RatingDisplay';

const PlayPage: React.FC = () => {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const createAndJoinLobby = useCurrentLobbyStore((state) => state.createAndJoinLobby);

  const [queues, setQueues] = useState<QueueInfo[]>([]);
  const [queuesLoading, setQueuesLoading] = useState(true);
  const [queuesError, setQueuesError] = useState<string | null>(null);

  const [searchingQueueId, setSearchingQueueId] = useState<string | null>(null);
  const [searchLobbyId, setSearchLobbyId] = useState<string | null>(null);
  const [searchStartTime, setSearchStartTime] = useState<number>(0);
  const [searchElapsed, setSearchElapsed] = useState<number>(0);

  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch queues on mount
  useEffect(() => {
    setQueuesLoading(true);
    getQueues()
      .then((data) => {
        setQueues(data);
        setQueuesLoading(false);
      })
      .catch(() => {
        setQueuesError('Failed to load queues.');
        setQueuesLoading(false);
      });
  }, []);

  // Elapsed timer for active search
  useEffect(() => {
    if (searchingQueueId) {
      elapsedTimerRef.current = setInterval(() => {
        setSearchElapsed(Math.floor((Date.now() - searchStartTime) / 1000));
      }, 1000);
    } else {
      setSearchElapsed(0);
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    }
    return () => {
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    };
  }, [searchingQueueId, searchStartTime]);

  const handleQueueClick = useCallback(async (queue: QueueInfo) => {
    if (searchingQueueId) return;
    try {
      const lobbyId = await createAndJoinLobby({
        type: 'matchmaking',
        gameMode: queue.queueId,
      });
      if (!lobbyId) return;
      setSearchLobbyId(lobbyId);
      setSearchingQueueId(queue.queueId);
      setSearchStartTime(Date.now());
      setSearchElapsed(0);
      await startSearch(lobbyId);
    } catch (err) {
      console.error('Failed to start matchmaking search:', err);
      setSearchingQueueId(null);
      setSearchLobbyId(null);
    }
  }, [searchingQueueId, createAndJoinLobby]);

  const handleCancel = useCallback(async () => {
    if (!searchLobbyId) return;
    try {
      await cancelSearch(searchLobbyId);
    } catch (err) {
      console.error('Failed to cancel search:', err);
    } finally {
      setSearchingQueueId(null);
      setSearchLobbyId(null);
      setSearchElapsed(0);
    }
  }, [searchLobbyId]);

  // Build user ratings from auth store for display
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const u = user as any;
  const userRatings = {
    h2h_qp: u?.elo != null
      ? ({ mu: u.elo as number, rd: (u.rd as number) ?? 0, gamesPlayed: 0 } as RatingInfo)
      : undefined,
    h2h_ranked: u?.elo != null
      ? ({ mu: u.elo as number, rd: (u.rd as number) ?? 0, gamesPlayed: 0 } as RatingInfo)
      : undefined,
    ffa4: u?.open_skill_mu != null
      ? ({ mu: u.open_skill_mu as number, rd: (u.open_skill_sigma as number) ?? 0, gamesPlayed: 0 } as RatingInfo)
      : undefined,
  };

  const rankedQueues = queues.filter((q) => q.ranked);
  const casualQueues = queues.filter((q) => !q.ranked);

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">Play</h2>

      {/* Casual Section */}
      <section className="space-y-4">
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-200">Casual</h3>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => navigate('/lobby/create')}
            className="py-2 px-5 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 font-medium hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors shadow-sm"
          >
            Create Game
          </button>
          <button
            onClick={() => navigate('/dashboard')}
            className="py-2 px-5 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 font-medium hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors shadow-sm"
          >
            Browse Games
          </button>
        </div>

        {/* Casual queues from API if any */}
        {casualQueues.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-2">
            {casualQueues.map((queue) => (
              <QueueCard
                key={queue.queueId}
                queue={queue}
                isSearching={searchingQueueId === queue.queueId}
                searchTime={searchingQueueId === queue.queueId ? searchElapsed : 0}
                onClick={() => handleQueueClick(queue)}
                onCancel={handleCancel}
              />
            ))}
          </div>
        )}
      </section>

      {/* Ranked Section */}
      <section className="space-y-4">
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-200">Ranked</h3>

        <RatingDisplay ratings={userRatings} />

        {queuesLoading && (
          <p className="text-sm text-gray-500 dark:text-gray-400">Loading queues...</p>
        )}

        {queuesError && (
          <p className="text-sm text-red-500 dark:text-red-400">{queuesError}</p>
        )}

        {!queuesLoading && !queuesError && rankedQueues.length === 0 && (
          <p className="text-sm text-gray-500 dark:text-gray-400">No ranked queues available.</p>
        )}

        {rankedQueues.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {rankedQueues.map((queue) => (
              <QueueCard
                key={queue.queueId}
                queue={queue}
                isSearching={searchingQueueId === queue.queueId}
                searchTime={searchingQueueId === queue.queueId ? searchElapsed : 0}
                onClick={() => handleQueueClick(queue)}
                onCancel={handleCancel}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default PlayPage;
