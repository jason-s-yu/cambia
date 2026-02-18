// src/components/lobby/ReadyButton.tsx
import React from 'react';
import Button from '@/components/common/Button';
import type { LobbyUser } from '@/types';
import { useAuthStore } from '@/stores/authStore';

interface ReadyButtonProps {
  sendMessage: (message: { type: string }) => void;
  players: LobbyUser[];
}

const ReadyButton: React.FC<ReadyButtonProps> = ({ sendMessage, players }) => {
  const currentUserId = useAuthStore((state) => state.user?.id);
  const currentUser = players.find(p => p.id === currentUserId);
  const isReady = currentUser?.is_ready ?? false;

  const handleToggleReady = () => {
    if (!currentUser) return;
    sendMessage({ type: isReady ? 'unready' : 'ready' });
  };

  // Don't show button if user isn't in the player list yet
  if (!currentUser) {
    return null;
  }

  return (
    <Button
      onClick={handleToggleReady}
      variant={isReady ? 'secondary' : 'primary'}
      className="w-full justify-center"
    >
      {isReady ? 'Mark as Not Ready' : 'Mark as Ready'}
    </Button>
  );
};

export default ReadyButton;