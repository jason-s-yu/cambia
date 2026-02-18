// src/components/game/GameInfoPanel.tsx
import React from 'react';
import type { ObfPlayerState } from '@/types/game';

interface GameInfoPanelProps {
	currentPlayerId: string | null;
	players: ObfPlayerState[];
	turnId: number;
}

const GameInfoPanel: React.FC<GameInfoPanelProps> = ({ currentPlayerId, players, turnId }) => {
	const currentPlayer = players.find(p => p.playerId === currentPlayerId);
	const turnText = currentPlayer ? `${currentPlayer.username}'s Turn` : 'Waiting...';

	return (
		<div className="bg-black/50 text-white p-2 rounded shadow-md text-center text-sm mb-2">
			<p className='font-semibold'>{turnText}</p>
			<p className='text-xs opacity-80'>Turn: {turnId}</p>
			{/* Add turn timer display later */}
		</div>
	);
};

export default GameInfoPanel;