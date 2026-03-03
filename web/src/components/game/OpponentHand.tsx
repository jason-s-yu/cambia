// src/components/game/OpponentHand.tsx
import React from 'react';
import type { ObfPlayerState } from '@/types/game';
import Card from './Card';

interface OpponentHandProps {
	playerState: ObfPlayerState;
	isTargetable: boolean;
	onCardClick?: (playerId: string, idx: number) => void;
}

const OpponentHand: React.FC<OpponentHandProps> = ({ playerState, isTargetable, onCardClick }) => {
	const { playerId, username, handSize, isCurrentTurn, connected, hasCalledCambia } = playerState;

	return (
		<div
			className={`flex flex-col items-center p-2 rounded-lg transition-all ${
				isCurrentTurn
					? 'bg-yellow-300/80 dark:bg-yellow-700/80 ring-2 ring-yellow-400'
					: 'bg-black/40'
			}`}
		>
			<span className="text-xs font-semibold mb-1 truncate max-w-[100px] text-white drop-shadow">
				{username}
			</span>
			<div className="flex">
				{handSize === 0 ? (
					<div className="text-gray-300 italic h-[67px] flex items-center text-xs">Empty</div>
				) : (
					Array.from({ length: handSize }).map((_, idx) => (
						<Card
							key={`${playerId}-card-${idx}`}
							id={`${playerId}-card-${idx}`}
							faceUp={false}
							compact
							isTargetable={isTargetable}
							isInteractive={isTargetable && !!onCardClick}
							onClick={isTargetable && onCardClick ? () => onCardClick(playerId, idx) : undefined}
							className={idx > 0 ? '-ml-3' : ''}
						/>
					))
				)}
			</div>
			<span className={`text-[10px] mt-1 ${connected ? 'text-green-300' : 'text-red-300'}`}>
				{connected ? 'Online' : 'Offline'}{hasCalledCambia ? ' · C!' : ''}
			</span>
		</div>
	);
};

export default OpponentHand;
