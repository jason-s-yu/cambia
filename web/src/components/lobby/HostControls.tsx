import React from 'react';
import Button from '@/components/common/Button';
import type { LobbyUser } from '@/types';

interface HostControlsProps {
  sendMessage: (message: { type: string }) => void;
  players: LobbyUser[];
}

const HostControls: React.FC<HostControlsProps> = ({ sendMessage, players }) => {
  const handleStartGame = () => {
    sendMessage({ type: 'start_game' });
    // Server will validate if game can start
  };

  const allReady = players.length > 0 && players.every(p => p.is_ready);
  const canStart = players.length >= 2;

  return (
    <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mt-auto space-y-2">
        <h4 className="text-sm font-semibold text-center text-gray-600 dark:text-gray-300">Host Actions</h4>
      <Button
        onClick={handleStartGame}
        variant="primary"
        className="w-full justify-center"
        disabled={!allReady || !canStart}
        title={!canStart ? 'Need at least 2 players' : (!allReady ? 'Waiting for all players to be ready' : 'Start the game')}
      >
        Force Start Game
      </Button>
      {/* TODO: Add Invite button later if needed */}
      {/* <Button variant="secondary" className="w-full justify-center" onClick={() => {}}>Invite Friend</Button> */}
    </div>
  );
};

export default HostControls;