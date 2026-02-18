// src/components/game/OpponentHand.tsx
import React from 'react';
import type { ObfPlayerState } from '@/types/game';
import Card from './Card'; // Assuming Card component exists

interface OpponentHandProps {
	playerState: ObfPlayerState;
}

const OpponentHand: React.FC<OpponentHandProps> = ({ playerState }) => {
	const cardCount = playerState.handSize;
	const isCurrent = playerState.isCurrentTurn;

	return (
		<div className={`flex flex-col items-center p-2 rounded ${isCurrent ? 'bg-yellow-300 dark:bg-yellow-700 ring-2 ring-yellow-500' : 'bg-gray-400 dark:bg-gray-600 opacity-80'}`}>
			<span className="text-xs font-semibold mb-1 truncate max-w-[100px] text-black dark:text-white">{playerState.username}</span>
			<div className="flex justify-center space-x-[-15px]">
				{cardCount === 0 && <div className="text-gray-600 dark:text-gray-300 italic h-[60px] flex items-center text-xs">Empty</div>}
				{Array.from({ length: cardCount }).map((_, index) => (
					<Card
						key={`${playerState.playerId}-card-${index}`}
						id={`${playerState.playerId}-card-${index}`} // Placeholder ID
						faceUp={false} // Opponent cards always face down
						className="w-10 h-auto shadow-sm" // Smaller size for opponents
					/>
				))}
			</div>
			<span className={`text-xs mt-1 ${playerState.connected ? 'text-green-800 dark:text-green-300' : 'text-red-800 dark:text-red-300'}`}>
				{playerState.connected ? 'Online' : 'Offline'} {playerState.hasCalledCambia ? '(C!)' : ''}
			</span>
		</div>
	);
};

export default OpponentHand;