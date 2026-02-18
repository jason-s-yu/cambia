// src/components/lobby/PlayerList.tsx
import React from 'react';
import type { LobbyUser } from '@/types';
import { useAuthStore } from '@/stores/authStore';

interface PlayerListProps {
  players: LobbyUser[];
}

const PlayerList: React.FC<PlayerListProps> = ({ players }) => {
  const currentUserId = useAuthStore((state) => state.user?.id);

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-100">Players ({players.length})</h3>
      <ul className="space-y-2">
        {players.map((player) => (
          <li key={player.id} className={`flex justify-between items-center p-2 rounded ${player.id === currentUserId ? 'bg-blue-100 dark:bg-blue-900' : 'bg-gray-50 dark:bg-gray-700'}`}>
            <span className="font-medium text-gray-700 dark:text-gray-200 truncate" title={player.id}>
              {player.username}
              {player.is_host ? ' (Host)' : ''}
              {player.id === currentUserId ? ' (You)' : ''}
            </span>
            <span className={`text-sm px-2 py-0.5 rounded-full ${player.is_ready ? 'bg-green-200 text-green-800 dark:bg-green-700 dark:text-green-100' : 'bg-yellow-200 text-yellow-800 dark:bg-yellow-700 dark:text-yellow-100'}`}>
              {player.is_ready ? 'Ready' : 'Not Ready'}
            </span>
          </li>
        ))}
        {players.length === 0 && (
            <li className="text-center text-sm text-gray-500 dark:text-gray-400 py-2">Waiting for players...</li>
        )}
      </ul>
    </div>
  );
};

export default PlayerList;